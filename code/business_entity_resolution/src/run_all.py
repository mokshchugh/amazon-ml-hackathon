"""End-to-end pipeline (SPEC section 6, Task 14).

    run_all.py --split {train,test} [--limit-s1 N] [--model-tag TAG]
               [--train-sample 200000] [--learning-rate 0.1] [--max-rounds 1500]
               [--max-cands-per-source K] [--baseline]
               [--out-dir DIR]
    run_all.py --package

train: cache -> splits -> normalized frames / siblings / candidates (the
    precomputed production caches in CACHE_DIR/prod are reused when present,
    else computed with the same production choices and saved there) ->
    features -> LightGBM on a sample of non-holdout S1 records -> score the
    holdout -> tune the decision layer -> save MODELS_DIR/TAG -> report.
    Also tunes the candidates-only baseline threshold (MODELS_DIR/baseline.json).
test: load TAG -> per country, feature/predict chunks of <= 2,000,000 pairs ->
    decide -> output/candidate_pairs.tsv + output/matching_results.tsv (every
    test S1 id present). ``--baseline`` writes the candidates-only insurance
    submission instead.
``--limit-s1 N``: fast mode (tests): N seeded S1 records + their GT S2/S3 +
    ~10x N same-country decoys, computed from scratch under
    CACHE_DIR/limit/<split>_<N>/ (prod caches are never read or written).

Context features: ``features.add_context_features`` runs on S1-aligned chunks
of the per-country candidate table (every row of an S1 record is in the same
chunk, so the per-S1 ranks, gaps, name frequency, source, sibling size and
search mask are exact); the two claimant columns, which need every claimant
of an S2/S3 record, are computed once on the WHOLE per-country table here
(``claimant_features``) and overwrite the chunk values. The result equals
``add_context_features`` on the whole per-country table (tested) without
holding a 100M+-row copy of it in memory.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import gc
import json
import pickle
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import config
import evaluate
import features
import io_utils
import normalize
import splits
import train

PROD_DIR = config.CACHE_DIR / "prod"
LIMIT_DIR = config.CACHE_DIR / "limit"
EXPERIMENTS_MD = config.REPO_ROOT / "experiments.md"

CHUNK_ROWS = 2_000_000       # feature / predict chunk (pairs)
TFIDF_SAMPLE = 2_000_000     # names the char TF-IDF is fitted on (R27)
DECOY_FACTOR = 10            # limit mode: same-country S2/S3 decoys per S1
P_FLOOR = 1e-3               # scored pairs below this p are dropped before the decision layer
TRAIN_SAMPLE = 200_000       # R32
LEARNING_RATE = 0.1          # R32
MAX_ROUNDS = 1500            # R32

CAND_COLS = ["s1_id", "s23_id", "source", "search_mask", "best_score"]
REC_COLS = list(dict.fromkeys(["entity_id", "country", "source"] + list(features._RECORD_COLUMNS)))
S1_CTX_COLS = ["entity_id", "name_sorted", "country"]
SOURCE_TSV_COLS = ["entity_id", "business_name", "business_address", "country"]


# ---------------------------------------------------------------------------
# Run bookkeeping: stage timings and peak memory
# ---------------------------------------------------------------------------

class RunLog:
    """Stage wall-clock timings and peak RSS (sampled every 0.5 s)."""

    def __init__(self) -> None:
        import psutil

        self._proc = psutil.Process()
        self.peak = self._proc.memory_info().rss
        self.stages: dict[str, float] = {}
        self._stop = threading.Event()
        self._t0 = time.time()
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:  # noqa: BLE001 - monitoring must never kill the run
                pass
            self._stop.wait(0.5)

    def stage(self, name: str):
        log = self

        class _Stage:
            def __enter__(self):
                self.t = time.time()
                log.msg(f"START {name}")

            def __exit__(self, *exc):
                dt = time.time() - self.t
                log.stages[name] = log.stages.get(name, 0.0) + dt
                log.msg(f"DONE  {name} {dt:.1f}s rss={log.rss_gb():.2f}GB peak={log.peak_gb():.2f}GB")

        return _Stage()

    def rss_gb(self) -> float:
        return self._proc.memory_info().rss / 2 ** 30

    def peak_gb(self) -> float:
        return max(self.peak, self._proc.memory_info().rss) / 2 ** 30

    def total(self) -> float:
        return time.time() - self._t0

    def close(self) -> None:
        self._stop.set()

    def msg(self, text: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _arrow_strings(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and df[c].dtype != "string[pyarrow]":
            df[c] = df[c].astype("string[pyarrow]")
    return df


def _pa_str(values) -> pa.Array:
    arr = features._str_array(values)
    return arr.cast(pa.string()) if pa.types.is_large_string(arr.type) else arr


def _read_frame(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    return _arrow_strings(df, [c for c in ("entity_id", "country", "source") if c in df.columns])


def _truth_map(gt: pd.DataFrame, ids) -> dict[str, set[str]]:
    sub = gt[gt["s1_id"].isin(ids)]
    out: dict[str, set[str]] = {}
    for a, b in zip(sub["s1_id"].tolist(), sub["s23_id"].tolist()):
        out.setdefault(a, set()).add(b)
    return out


def write_source_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write raw source records as an organiser-style source TSV (for validate())."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(SOURCE_TSV_COLS) + "\n")
        cols = [df[c].astype("string").fillna("").tolist() for c in SOURCE_TSV_COLS]
        for row in zip(*cols):
            f.write("\t".join(row) + "\n")




def _concat_id_list_parts(parts: Sequence[Path], out_path: Path, value_col: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"source1_entity_id\t{value_col}\n")
        for part in parts:
            with open(part, "r", encoding="utf-8", newline="") as f:
                next(f, None)
                shutil.copyfileobj(f, out)


# ---------------------------------------------------------------------------
# Limit-mode slice sampling
# ---------------------------------------------------------------------------

def sample_slice(s1: pd.DataFrame, s23_index: pd.DataFrame, gt: pd.DataFrame | None, n: int,
                 seed: int = config.SEED, decoy_factor: int = DECOY_FACTOR) -> tuple[pd.DataFrame, np.ndarray]:
    """Seeded slice for fast mode: ``n`` S1 records (row-order independent),
    every GT S2/S3 partner of theirs present in ``s23_index``, and a seeded
    sample of ``decoy_factor`` x (slice S1 count of that country) same-country
    S2/S3 records per country. Returns (S1 slice sorted by entity_id, sorted
    unique S2/S3 ids)."""
    rng = np.random.default_rng(seed)
    s1s = s1.sort_values("entity_id", kind="stable").reset_index(drop=True)
    pick = np.sort(rng.choice(len(s1s), size=min(n, len(s1s)), replace=False))
    s1s = s1s.iloc[pick].reset_index(drop=True)

    idx = s23_index.sort_values("entity_id", kind="stable").reset_index(drop=True)
    idx_ids = idx["entity_id"].astype(str).to_numpy(dtype=object)
    chosen = []
    if gt is not None:
        partners = gt.loc[gt["s1_id"].isin(set(s1s["entity_id"].astype(str))), "s23_id"].astype(str)
        chosen.append(np.intersect1d(partners.to_numpy(dtype=object).astype(str), idx_ids.astype(str)))
    counts = s1s["country"].astype(str).value_counts()
    idx_country = idx["country"].astype(str).to_numpy(dtype=object)
    for country in sorted(counts.index):
        pool = idx_ids[idx_country == country]
        k = min(decoy_factor * int(counts[country]), len(pool))
        if k:
            chosen.append(pool[np.sort(rng.choice(len(pool), size=k, replace=False))].astype(str))
    s23_ids = np.unique(np.concatenate(chosen)) if chosen else np.array([], dtype=str)
    return s1s, s23_ids.astype(object)


# ---------------------------------------------------------------------------
# Candidates-only baseline (insurance submission)
# ---------------------------------------------------------------------------

def tune_baseline_threshold(top1: pd.DataFrame, truth: Mapping[str, set[str]],
                            s1_ids: Sequence[str]) -> tuple[float, float]:
    """Threshold t for "predict the top-1 candidate iff best_score >= t" that
    maximises macro F0.5 over ``s1_ids`` (exact over every distinct score, plus
    t = inf = predict nothing; ties -> the higher threshold). Returns (t, F)."""
    ids = list(s1_ids)
    n_truth = np.array([len(truth.get(s, ())) for s in ids], dtype=np.float64)
    f_empty = (n_truth == 0).astype(np.float64)
    top = dict(zip(top1["s1_id"].tolist(), zip(top1["s23_id"].tolist(), top1["best_score"].tolist())))
    score = np.full(len(ids), -np.inf)
    f_pred = np.zeros(len(ids))
    for i, s in enumerate(ids):
        hit = top.get(s)
        if hit is None:
            continue
        score[i] = hit[1]
        if hit[0] in truth.get(s, ()):
            f_pred[i] = 1.25 / (0.25 * n_truth[i] + 1.0)  # P = 1, R = 1/n
    base = f_empty.sum()
    has = np.isfinite(score)
    sc, gain = score[has], (f_pred - f_empty)[has]
    best_t, best_total = float("inf"), base
    if len(sc):
        order = np.argsort(-sc, kind="stable")
        sc, cum = sc[order], np.cumsum(gain[order])
        last = np.r_[sc[1:] != sc[:-1], True]  # end of each tie run
        totals = base + cum[last]
        k = int(np.argmax(totals))
        if totals[k] > best_total:
            best_t, best_total = float(sc[last][k]), float(totals[k])
    n = max(len(ids), 1)
    return best_t, best_total / n




# ---------------------------------------------------------------------------
# Candidate tables as int codes
# ---------------------------------------------------------------------------

def _large_str(values) -> pa.Array:
    """large_string arrow array; arrow inputs converted without a Python round trip."""
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array) and (pa.types.is_string(values.type) or pa.types.is_large_string(values.type)):
        return pc.fill_null(values.cast(pa.large_string()), "")
    return features._str_array(values)


def sorted_ids(values) -> pa.Array:
    """Unique ids, sorted (large_string)."""
    arr = pc.unique(_large_str(values))
    return arr.take(pc.sort_indices(arr))


def _codes_in(values, universe: pa.Array, missing_ok: bool = False) -> np.ndarray:
    """Position of each value in ``universe`` (-1 if absent, KeyError unless ``missing_ok``)."""
    idx = pc.index_in(_large_str(values), value_set=universe)
    if idx.null_count and not missing_ok:
        raise KeyError(f"{idx.null_count} candidate ids not found in the record frames")
    return np.asarray(idx.fill_null(-1), dtype=np.int32)


class CandCodes:
    """A candidate table held as int32 codes into SORTED id arrays ``u1``
    (S1) and ``u23`` (S2/S3): code order == lexicographic id order, the tie
    order ``add_context_features`` uses (factorize sort=True). About 14 bytes
    per pair instead of ~45 for the string table."""

    def __init__(self, u1: pa.Array, u23: pa.Array, s1c: np.ndarray, s23c: np.ndarray,
                 is_s3: np.ndarray, mask: np.ndarray, score: np.ndarray) -> None:
        self.u1, self.u23 = u1, u23
        self.s1c, self.s23c, self.is_s3, self.mask, self.score = s1c, s23c, is_s3, mask, score

    def __len__(self) -> int:
        return len(self.s1c)

    def subset(self, rows) -> "CandCodes":
        return CandCodes(self.u1, self.u23, self.s1c[rows], self.s23c[rows], self.is_s3[rows],
                         self.mask[rows], self.score[rows])

    def s1_ids(self, a: int = 0, b: int | None = None) -> pa.Array:
        return self.u1.take(pa.array(self.s1c[a:b]))

    def s23_ids(self, a: int = 0, b: int | None = None) -> pa.Array:
        return self.u23.take(pa.array(self.s23c[a:b]))

    def frame(self, a: int = 0, b: int | None = None) -> pd.DataFrame:
        """Rows a:b as the string candidate frame (CAND_COLS)."""
        return pd.DataFrame({
            "s1_id": pd.Series(pd.arrays.ArrowStringArray(self.s1_ids(a, b).cast(pa.string()))),
            "s23_id": pd.Series(pd.arrays.ArrowStringArray(self.s23_ids(a, b).cast(pa.string()))),
            "source": pd.Series(np.where(self.is_s3[a:b], "S3", "S2"), dtype="string[pyarrow]"),
            "search_mask": self.mask[a:b],
            "best_score": self.score[a:b],
        })

    @classmethod
    def from_frame(cls, df: pd.DataFrame, u1: pa.Array | None = None, u23: pa.Array | None = None) -> "CandCodes":
        u1 = sorted_ids(df["s1_id"]) if u1 is None else u1
        u23 = sorted_ids(df["s23_id"]) if u23 is None else u23
        return cls(u1, u23, _codes_in(df["s1_id"], u1), _codes_in(df["s23_id"], u23),
                   np.asarray(pc.equal(features._str_array(df["source"]), "S3")),
                   df["search_mask"].to_numpy(), df["best_score"].to_numpy())


def load_cand_codes(path: Path, u1: pa.Array, u23: pa.Array, batch_rows: int = 20_000_000) -> CandCodes:
    """Stream the candidate parquet into codes (one pass, bounded memory).
    Parquet batches are pooled into blocks of ~``batch_rows`` rows, because
    each ``index_in`` call rebuilds the hash of the (10M-id) value set."""
    parts, pending, n_pending = [], [], 0

    def flush():
        tbl = pa.Table.from_batches(pending).combine_chunks()
        parts.append((_codes_in(tbl.column("s1_id"), u1), _codes_in(tbl.column("s23_id"), u23),
                      np.asarray(pc.equal(_large_str(tbl.column("source")), "S3")),
                      tbl.column("search_mask").to_numpy(), tbl.column("best_score").to_numpy()))
        pending.clear()

    for batch in pq.ParquetFile(path).iter_batches(batch_size=min(batch_rows, 2_000_000), columns=CAND_COLS):
        pending.append(batch)
        n_pending += batch.num_rows
        if n_pending >= batch_rows:
            flush()
            n_pending = 0
    if pending:
        flush()
    if not parts:
        return CandCodes(u1, u23, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, bool),
                         np.zeros(0, np.int8), np.zeros(0, np.float32))
    return CandCodes(u1, u23, *[np.concatenate([p[i] for p in parts]) for i in range(5)])


class GroupedLists:
    """Read-only ``.get`` mapping s1_id -> list of s23_ids over a CandCodes
    table whose S1 codes form contiguous runs (memory-lean input for
    ``io_utils.write_id_lists`` on 100M+ candidate pairs)."""

    def __init__(self, t: CandCodes) -> None:
        self._t = t
        n = len(t)
        self._idx: dict[str, tuple[int, int]] = {}
        if n == 0:
            return
        starts = np.flatnonzero(np.r_[True, t.s1c[1:] != t.s1c[:-1]])
        ends = np.r_[starts[1:], n]
        keys = t.u1.take(pa.array(t.s1c[starts])).to_pylist()
        if len(set(keys)) != len(keys):
            raise ValueError("s1_id runs are not contiguous")
        self._idx = {k: (int(a), int(b)) for k, a, b in zip(keys, starts, ends)}

    @classmethod
    def from_ids(cls, s1_ids, s23_ids) -> "GroupedLists":
        df = pd.DataFrame({"s1_id": list(s1_ids), "s23_id": list(s23_ids), "source": "S2",
                           "search_mask": 0, "best_score": 0.0})
        return cls(CandCodes.from_frame(df))

    def get(self, key, default=None):
        r = self._idx.get(key)
        if r is None:
            return default
        return self._t.s23_ids(r[0], r[1]).to_pylist()


def top1_per_s1(t: CandCodes) -> pd.DataFrame:
    """Highest-best_score candidate per S1 record (ties -> smaller s23_id)."""
    if len(t) == 0:
        return pd.DataFrame({"s1_id": pd.Series([], dtype=object), "s23_id": pd.Series([], dtype=object),
                             "best_score": np.zeros(0)})
    order = np.lexsort((t.s23c, -t.score.astype(np.float64), t.s1c))
    s1o = t.s1c[order]
    sub = t.subset(order[np.r_[True, s1o[1:] != s1o[:-1]]])
    return pd.DataFrame({"s1_id": sub.s1_ids().to_pylist(), "s23_id": sub.s23_ids().to_pylist(),
                         "best_score": sub.score.astype(np.float64)})


def order_and_trim(t: CandCodes, max_k: int | None) -> CandCodes:
    """Rows grouped by S1 (ascending id); with ``max_k`` keep the top-K rows
    per (S1, source) by best_score (ties -> smaller s23_id)."""
    if len(t) == 0:
        return t
    if max_k is not None:
        order = np.lexsort((t.s23c, -t.score.astype(np.float64), t.is_s3, t.s1c))
        g = t.s1c[order].astype(np.int64) * 2 + t.is_s3[order]
        start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
        order = order[rank < max_k]
        del g, rank
    elif np.all(t.s1c[1:] >= t.s1c[:-1]):
        return t
    else:
        order = np.argsort(t.s1c, kind="stable")
    return t.subset(order)


def claimant_features(t: CandCodes) -> tuple[np.ndarray, np.ndarray]:
    """c_n_claimants / c_rank_among_claimants over the WHOLE table ``t``,
    exactly as ``features.add_context_features`` defines them."""
    if len(t) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    c1 = t.s1c.astype(np.int64)
    c23 = t.s23c.astype(np.int64)
    n23 = max(len(t.u23), 1)
    pairs = features._unique_sorted(c1 * n23 + c23)
    n_claim = np.bincount(pairs % n23, minlength=n23)[c23].astype(np.float32)
    del pairs
    rank = features._rank_within(c23, c1, t.score.astype(np.float64)).astype(np.float32)
    return n_claim, rank


def s1_chunk_bounds(s1, max_rows: int = CHUNK_ROWS) -> list[tuple[int, int]]:
    """[start, end) row ranges of <= ``max_rows`` rows that never split a run
    of equal (contiguous) S1 codes / ids."""
    n = len(s1)
    if n == 0:
        return []
    if isinstance(s1, np.ndarray) and s1.dtype.kind in "iu":
        codes = s1
    else:
        codes = np.asarray(pc.dictionary_encode(features._str_array(s1)).indices, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    # greedy: close the chunk before the S1 group that would overflow it
    bounds, a, prev = [], 0, 0
    for s in list(starts[1:]) + [n]:
        if s - a > max_rows and prev > a:
            bounds.append((a, prev))
            a = prev
        prev = int(s)
    bounds.append((a, n))
    return bounds


def context_chunks(t: CandCodes, s1_ctx: pd.DataFrame, sib: pd.DataFrame,
                   n_claim: np.ndarray, rank_claim: np.ndarray, max_rows: int = CHUNK_ROWS):
    """Yield (start, end, ctx) over S1-aligned chunks of ``t`` (grouped by S1)
    with the context columns of the whole table (see module doc)."""
    for a, b in s1_chunk_bounds(t.s1c, max_rows):
        ctx = features.add_context_features(t.frame(a, b), s1_ctx, sib)
        ctx["c_n_claimants"] = n_claim[a:b]
        ctx["c_rank_among_claimants"] = rank_claim[a:b]
        yield a, b, ctx

# ---------------------------------------------------------------------------
# Split preparation (prod caches or limit-mode slice)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SplitData:
    s1n: pd.DataFrame          # record columns the features read + country
    s23n: pd.DataFrame
    sib: pd.DataFrame
    cands_path: Path
    s1_raw_for_tsv: pd.DataFrame | None = None


def _prod_paths(split: str) -> dict[str, Path]:
    return {
        "tokens": PROD_DIR / f"tokens_{split}.json",
        "vocab": PROD_DIR / f"city_vocab_{split}.pkl",
        "source1": PROD_DIR / f"{split}_source1.parquet",
        "source2": PROD_DIR / f"{split}_source2.parquet",
        "source3": PROD_DIR / f"{split}_source3.parquet",
        "sib": PROD_DIR / f"{split}_sib.parquet",
        "cands": PROD_DIR / f"{split}_cands.parquet",
    }


def _learn_tokens(gt: pd.DataFrame, s1_raw: pd.DataFrame, s23_raw: pd.DataFrame) -> dict:
    pairs = gt.merge(s1_raw[["entity_id", "business_name"]].rename(columns={"business_name": "s1_name"}),
                     left_on="s1_id", right_on="entity_id", how="inner").drop(columns="entity_id")
    pairs = pairs.merge(s23_raw[["entity_id", "business_name"]].rename(columns={"business_name": "s23_name"}),
                        left_on="s23_id", right_on="entity_id", how="inner")
    return normalize.learn_token_table(pairs[["s23_name", "s1_name"]])


def _ensure_raw_cache(split: str) -> None:
    if not all((config.CACHE_DIR / f"{split}_{s}.parquet").exists() for s in ("source1", "source2", "source3")) \
            or (split == "train" and not (config.CACHE_DIR / "gt_pairs.parquet").exists()):
        io_utils.build_cache(split)
    if split == "test" and not (config.CACHE_DIR / "gt_pairs.parquet").exists():
        io_utils.build_cache("train")


def prepare_prod(split: str, log: RunLog, ids_only: bool = False) -> SplitData:
    """Reuse CACHE_DIR/prod (R28) or compute + save it with the same choices:
    token table from non-holdout train GT (train) / all train GT (test); city
    vocab from the S1 of ``split``."""
    p = _prod_paths(split)
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    if not all(p[s].exists() for s in ("source1", "source2", "source3")):
        with log.stage("normalize (compute prod)"):
            if p["tokens"].exists():
                table = normalize.load_token_table(p["tokens"])
            else:
                gt = pd.read_parquet(config.CACHE_DIR / "gt_pairs.parquet")
                if split == "train":
                    gt = gt[~gt["s1_id"].isin(splits.load_split("holdout"))]
                s1r = io_utils.load("train", "source1")
                s23r = pd.concat([io_utils.load("train", "source2"), io_utils.load("train", "source3")],
                                 ignore_index=True)
                table = _learn_tokens(gt, s1r, s23r)
                normalize.save_token_table(table, p["tokens"])
                del gt, s1r, s23r
                gc.collect()
            if p["vocab"].exists():
                with open(p["vocab"], "rb") as f:
                    vocab = pickle.load(f)
            else:
                s1r = io_utils.load(split, "source1")
                vocab = normalize.build_city_vocab(s1r["business_address"], s1r["country"], min_count=3)
                with open(p["vocab"], "wb") as f:
                    pickle.dump(vocab, f)
                del s1r
            for s in ("source1", "source2", "source3"):
                if not p[s].exists():
                    normalize.normalize_frame(io_utils.load(split, s), table, vocab).to_parquet(p[s], index=False)
                    gc.collect()
    if not p["sib"].exists():
        with log.stage("siblings (compute prod)"):
            import siblings

            s23 = pd.concat([pd.read_parquet(p["source2"]), pd.read_parquet(p["source3"])], ignore_index=True)
            siblings.sibling_groups(s23).to_parquet(p["sib"], index=False)
            del s23
            gc.collect()
    if not p["cands"].exists():
        with log.stage("blocking (compute prod)"):
            import blocking

            s1 = pd.read_parquet(p["source1"])
            s23 = pd.concat([pd.read_parquet(p["source2"]), pd.read_parquet(p["source3"])], ignore_index=True)
            blocking.generate_candidates(s1, s23).to_parquet(p["cands"], index=False)
            del s1, s23
            gc.collect()
    with log.stage("load frames"):
        if ids_only:  # --baseline: candidates alone need only ids and countries
            s1n = _read_frame(p["source1"], ["entity_id", "country"])
            s23n = pd.concat([_read_frame(p["source2"], ["entity_id"]), _read_frame(p["source3"], ["entity_id"])],
                             ignore_index=True)
            return SplitData(s1n, s23n, pd.DataFrame(), p["cands"])
        s1n = _read_frame(p["source1"], REC_COLS)
        s23n = pd.concat([_read_frame(p["source2"], REC_COLS), _read_frame(p["source3"], REC_COLS)],
                         ignore_index=True)
        sib = _read_frame(p["sib"])
    return SplitData(s1n, s23n, sib, p["cands"])


def prepare_limit(split: str, n: int, work: Path, log: RunLog,
                  slice_hook: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> SplitData:
    """Fast mode: sample a slice and compute everything for it from scratch."""
    import blocking
    import siblings

    work.mkdir(parents=True, exist_ok=True)
    with log.stage("limit: sample slice"):
        s1_raw = io_utils.load(split, "source1")
        src_paths = [config.CACHE_DIR / f"{split}_{s}.parquet" for s in ("source2", "source3")]
        s23_index = pd.concat([pq.read_table(pth, columns=["entity_id", "country"]).to_pandas()
                               for pth in src_paths], ignore_index=True)
        gt = pd.read_parquet(config.CACHE_DIR / "gt_pairs.parquet") if split == "train" else None
        s1s, s23_ids = sample_slice(s1_raw, s23_index, gt, n)
        del s1_raw, s23_index
        value_set = _pa_str(list(s23_ids))
        s23_raw = pd.concat([ds.dataset(str(pth), format="parquet").to_table(
            filter=pc.field("entity_id").isin(value_set)).to_pandas() for pth in src_paths], ignore_index=True)
        s23_raw = _arrow_strings(s23_raw, list(s23_raw.columns))
        if slice_hook is not None:
            s1s = slice_hook(s1s)
        log.msg(f"slice: {len(s1s)} S1, {len(s23_raw)} S2/S3")
    with log.stage("limit: normalize"):
        if gt is not None:
            holdout = splits.load_split("holdout")
            gt_slice = gt[gt["s1_id"].isin(set(s1s["entity_id"].astype(str)) - holdout)]
            table = _learn_tokens(gt_slice, s1s, s23_raw)
        else:
            table = {}
        vocab = normalize.build_city_vocab(s1s["business_address"], s1s["country"], min_count=3)
        s1n = normalize.normalize_frame(s1s, table, vocab)
        s23n = normalize.normalize_frame(s23_raw, table, vocab)
    with log.stage("limit: siblings"):
        sib = siblings.sibling_groups(s23n)
    with log.stage("limit: blocking"):
        cands = blocking.generate_candidates(s1n, s23n)
        cands_path = work / f"{split}_cands.parquet"
        cands.to_parquet(cands_path, index=False)
        log.msg(f"slice candidates: {len(cands)}")
    s1n = _arrow_strings(s1n[REC_COLS].copy(), ["entity_id", "country", "source"])
    s23n = _arrow_strings(s23n[REC_COLS].copy(), ["entity_id", "country", "source"])
    write_source_tsv(s1s, work / f"{split}_source1.tsv")
    return SplitData(s1n, s23n, sib, cands_path, s1_raw_for_tsv=s1s)


@dataclasses.dataclass
class TextModels:
    idf: dict
    tfidf: object
    addr_idf: dict


def build_text_models(data: SplitData, log: RunLog) -> TextModels:
    """IDF / TF-IDF / address IDF, once per split (R7/R27)."""
    with log.stage("idf/tfidf/addr_idf"):
        idf = features.build_idf(pd.concat([data.s1n["name_clean"], data.s23n["name_clean"]], ignore_index=True))
        names = pd.concat([data.s1n["name_sorted"], data.s23n["name_sorted"]], ignore_index=True)
        if len(names) > TFIDF_SAMPLE:
            rng = np.random.default_rng(config.SEED)
            names = names.iloc[np.sort(rng.choice(len(names), size=TFIDF_SAMPLE, replace=False))]
        tfidf = features.build_tfidf(names.reset_index(drop=True))
        addr_idf = features.build_idf(pd.concat([data.s1n["addr_clean"], data.s23n["addr_clean"]],
                                                ignore_index=True))
    return TextModels(idf, tfidf, addr_idf)



@dataclasses.dataclass
class Universe:
    """Sorted S1 / S2-S3 id arrays of a split, the country of each S1 code,
    and the candidate table as codes."""
    u1: pa.Array
    u23: pa.Array
    countries: list[str]
    s1_cc: np.ndarray        # country index per S1 code
    cands: CandCodes

    def country_rows(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.s1_cc[self.cands.s1c] == k)

    def ids_where(self, keep: np.ndarray) -> list[str]:
        return self.u1.filter(pa.array(keep)).to_pylist()


def build_universe(data: SplitData, log: RunLog) -> Universe:
    with log.stage("load candidates"):
        u1 = sorted_ids(data.s1n["entity_id"])
        u23 = sorted_ids(data.s23n["entity_id"])
        pos = _codes_in(data.s1n["entity_id"], u1)
        cc, countries = pd.factorize(data.s1n["country"].astype("string").fillna("").to_numpy(dtype=object),
                                     sort=True)
        s1_cc = np.empty(len(u1), dtype=np.int32)
        s1_cc[pos] = cc
        cands = load_cand_codes(data.cands_path, u1, u23)
        log.msg(f"{len(u1)} S1, {len(u23)} S2/S3, {len(cands)} candidate pairs")
    return Universe(u1, u23, [str(c) for c in countries], s1_cc, cands)


def _s1_ctx(data: SplitData, country: str) -> pd.DataFrame:
    c = data.s1n["country"].astype("string").fillna("")
    return data.s1n.loc[(c == country).to_numpy(), S1_CTX_COLS]


def _features(ctx: pd.DataFrame, data: SplitData, tm: TextModels) -> pd.DataFrame:
    return features.compute_features(ctx, data.s1n, data.s23n, data.sib, tm.idf,
                                     tfidf=tm.tfidf, addr_idf=tm.addr_idf)


def _score_rows(t: CandCodes, n_claim, rank_claim, s1_ctx, data, tm, booster, cal) -> pd.DataFrame:
    """Features + calibrated p for every row of ``t``; keeps p >= P_FLOOR.
    Returns (s1_id, s23_id, p)."""
    keep_rows, keep_p = [], []
    for a, b, ctx in context_chunks(t, s1_ctx, data.sib, n_claim, rank_claim):
        X = _features(ctx, data, tm)
        p = train.predict_proba(booster, cal, X[features.FEATURE_COLUMNS])
        keep = np.flatnonzero(p >= P_FLOOR)
        keep_rows.append(a + keep)
        keep_p.append(p[keep])
        del X, ctx
    rows = np.concatenate(keep_rows) if keep_rows else np.zeros(0, np.int64)
    sub = t.subset(rows)
    out = sub.frame()[["s1_id", "s23_id"]]
    out["p"] = np.concatenate(keep_p) if keep_p else np.zeros(0, np.float32)
    return out


def _decision_params_path(tag: str) -> Path:
    return Path(config.MODELS_DIR) / tag / "decision_params.json"


def _params_json(by_country: Mapping, default) -> str:
    return json.dumps({"default": dataclasses.asdict(default),
                       "by_country": {c: dataclasses.asdict(p) for c, p in sorted(by_country.items())}}, indent=2)


def load_decision_params(text: str):
    """(per-country params, default for other countries) from decision_params.json.
    A flat v1 file (one setting) applies to every country."""
    import decide

    raw = json.loads(text)
    if "default" not in raw:
        return {}, decide.DecisionParams(**raw)
    return ({c: decide.DecisionParams(**p) for c, p in raw["by_country"].items()},
            decide.DecisionParams(**raw["default"]))


def tune_by_country(scored: pd.DataFrame, sib: pd.DataFrame, truth: Mapping[str, set[str]],
                    report_ids: Sequence[str], s1_country: Mapping[str, str]):
    """Tune the decision layer separately per country on the holdout (candidate
    lists never cross countries, so the one-owner rule is unaffected); countries
    not in the holdout get the field-wise strictest setting."""
    import decide

    by_country = {}
    row_country = scored["s1_id"].map(s1_country).to_numpy()
    for country in sorted({s1_country[s] for s in report_ids}):
        ids_c = [s for s in report_ids if s1_country[s] == country]
        sc = scored[row_country == country].reset_index(drop=True)
        by_country[country] = decide.tune(sc, sib, truth, ids_c)
    return by_country, decide.conservative(by_country.values())


def decide_by_country(scored: pd.DataFrame, sib: pd.DataFrame, s1_country: Mapping[str, str],
                      by_country: Mapping, default) -> dict[str, list[str]]:
    import decide

    matches: dict[str, list[str]] = {}
    row_country = scored["s1_id"].map(s1_country).to_numpy()
    for country in pd.unique(row_country):
        sc = scored[row_country == country].reset_index(drop=True)
        matches.update(decide.decide(sc, sib, by_country.get(country, default)))
    return matches


def _in_sorted(keys: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[pos] == keys


def _write_candidate_file(out_dir: Path, pieces: Sequence[tuple[list[str], CandCodes | None]]) -> None:
    """candidate_pairs.tsv from per-country (S1 ids, grouped table) pieces."""
    parts = []
    for i, (ids, t) in enumerate(pieces):
        part = out_dir / f"_cands_part{i}.tsv"
        io_utils.write_id_lists(part, "candidate_entity_ids", ids, GroupedLists(t) if t is not None else {})
        parts.append(part)
    _concat_id_list_parts(parts, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    for part in parts:
        part.unlink()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(args, log: RunLog, slice_hook=None) -> dict:
    limit = args.limit_s1
    work = LIMIT_DIR / f"train_{limit}" if limit else None
    tag = args.model_tag or (f"limit{limit}" if limit else "v1")
    with log.stage("cache+splits"):
        _ensure_raw_cache("train")
        splits.ensure_splits()
        holdout = splits.load_split("holdout")
        gt = _arrow_strings(pd.read_parquet(config.CACHE_DIR / "gt_pairs.parquet"), ["s1_id", "s23_id"])
    data = prepare_limit("train", limit, work, log, slice_hook) if limit else prepare_prod("train", log)
    tm = build_text_models(data, log)
    uni = build_universe(data, log)
    u1, n23 = uni.u1, max(len(uni.u23), 1)

    is_hold = np.asarray(pc.is_in(u1, value_set=features._str_array(sorted(holdout))))
    nonhold = np.flatnonzero(~is_hold)
    rng = np.random.default_rng(config.SEED)
    is_train = np.zeros(len(u1), dtype=bool)
    is_train[nonhold[rng.choice(len(nonhold), size=min(args.train_sample, len(nonhold)), replace=False)]] = True
    is_eval = np.ones(len(u1), dtype=bool) if limit else is_hold
    report_ids = uni.ids_where(is_hold)
    report_set = set(report_ids)
    log.msg(f"S1: {len(u1)}; train sample {int(is_train.sum())}; eval {int(is_eval.sum())}; "
            f"holdout {len(report_ids)}")

    g1 = _codes_in(gt["s1_id"], u1, missing_ok=True).astype(np.int64)
    g23 = _codes_in(gt["s23_id"], uni.u23, missing_ok=True).astype(np.int64)
    ok = (g1 >= 0) & (g23 >= 0)
    gt_keys = np.unique(g1[ok] * n23 + g23[ok])
    gt_hold_keys = np.unique(g1[ok & is_hold[np.maximum(g1, 0)]] * n23 + g23[ok & is_hold[np.maximum(g1, 0)]])
    truth = _truth_map(gt, report_set)
    n_true = sum(len(v) for v in truth.values())
    del g1, g23, ok

    # ---- pass 1: training features; keep eval rows for pass 2 --------------
    X_parts, y_parts, g_parts, eval_tables, top1_parts = [], [], [], [], []
    blk_hit, n_cand_rows = 0, 0
    with log.stage("features (train sample)"):
        for k, country in enumerate(uni.countries):
            t = order_and_trim(uni.cands.subset(uni.country_rows(k)), args.max_cands_per_source)
            n_cand_rows += len(t)
            n_claim, rank_claim = claimant_features(t)
            m_train, m_eval, m_rep = is_train[t.s1c], is_eval[t.s1c], is_hold[t.s1c]
            if m_rep.any():
                rep = t.subset(m_rep)
                top1_parts.append(top1_per_s1(rep))
                blk_hit += int(np.isin(gt_hold_keys, rep.s1c.astype(np.int64) * n23 + rep.s23c).sum())
                del rep
            eval_tables.append((k, t.subset(m_eval), n_claim[m_eval], rank_claim[m_eval]))
            tr, nc_t, rc_t = t.subset(m_train), n_claim[m_train], rank_claim[m_train]
            del t, n_claim, rank_claim
            gc.collect()
            s1_ctx = _s1_ctx(data, country)
            for a, b, ctx in context_chunks(tr, s1_ctx, data.sib, nc_t, rc_t):
                X = _features(ctx, data, tm)
                y_parts.append(_in_sorted(tr.s1c[a:b].astype(np.int64) * n23 + tr.s23c[a:b], gt_keys))
                X_parts.append(X[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32))
                g_parts.append(tr.s1c[a:b])
                del X, ctx
            log.msg(f"country {country!r}: train rows so far {sum(len(y) for y in y_parts)}")
            del tr, nc_t, rc_t
            gc.collect()
    uni.cands = None
    gc.collect()

    # ---- baseline threshold (holdout candidates) ---------------------------
    top1 = pd.concat(top1_parts, ignore_index=True) if top1_parts else top1_per_s1(
        CandCodes(u1, uni.u23, *[np.zeros(0, d) for d in (np.int32, np.int32, bool, np.int8, np.float32)]))
    t_base, f_base = tune_baseline_threshold(top1, truth, report_ids)
    blocking_recall = blk_hit / n_true if n_true else 1.0
    tag_dir = Path(config.MODELS_DIR) / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    base_json = json.dumps({"threshold": t_base, "holdout_macro_f05": f_base, "tag": tag,
                            "max_cands_per_source": args.max_cands_per_source}, indent=2)
    (tag_dir / "baseline.json").write_text(base_json, encoding="utf-8")
    if not limit:
        (Path(config.MODELS_DIR) / "baseline.json").write_text(base_json, encoding="utf-8")
    log.msg(f"baseline: threshold={t_base:.4f} holdout macro F0.5={f_base:.4f}; blocking recall={blocking_recall:.4f}")

    # ---- train ---------------------------------------------------------------
    with log.stage("train"):
        X = pd.DataFrame(np.concatenate(X_parts), columns=features.FEATURE_COLUMNS)
        del X_parts
        y = np.concatenate(y_parts).astype(np.float64)
        groups = np.concatenate(g_parts)
        del y_parts, g_parts
        gc.collect()
        n_train_rows, n_pos = len(X), int(y.sum())
        log.msg(f"training rows {n_train_rows}, positives {n_pos}")
        params = dict(train.LGB_PARAMS, learning_rate=args.learning_rate)
        t_train = time.time()
        booster, oof = train.train_model(X, y, groups, max_rounds=args.max_rounds, params=params)
        cal = train.fit_calibrator(oof, y)
        train_seconds = time.time() - t_train
        del X, y, groups, oof
        gc.collect()
        train.save(booster, cal, tag)
        log.msg(f"trained in {train_seconds:.0f}s, {booster.current_iteration()} rounds")

    # ---- pass 2: score eval rows ----------------------------------------------
    with log.stage("score eval"):
        scored_parts = []
        for k, t, n_claim, rank_claim in eval_tables:
            scored_parts.append(_score_rows(t, n_claim, rank_claim, _s1_ctx(data, uni.countries[k]),
                                            data, tm, booster, cal))
        scored = pd.concat(scored_parts, ignore_index=True)
        del scored_parts
        log.msg(f"scored rows kept (p >= {P_FLOOR}): {len(scored)}")
        scored.to_parquet(tag_dir / "holdout_scored.parquet", index=False)  # re-tune without re-scoring

    with log.stage("tune decision"):
        rep_scored = scored[scored["s1_id"].isin(report_set).to_numpy()].reset_index(drop=True)
        s1_country = dict(zip(u1.to_pylist(), [uni.countries[c] for c in uni.s1_cc]))
        params_by_country, params_d = tune_by_country(rep_scored, data.sib, truth, report_ids, s1_country)
        del rep_scored
        _decision_params_path(tag).write_text(_params_json(params_by_country, params_d), encoding="utf-8")
        log.msg(f"decision params: {params_by_country}; other countries {params_d}")
    with log.stage("decide + report"):
        matches = decide_by_country(scored, data.sib, s1_country, params_by_country, params_d)
        rep = evaluate.report({s: set(matches.get(s, ())) for s in report_ids}, truth,
                              {s: s1_country[s] for s in report_ids})
        rep["blocking_recall"] = blocking_recall
        rep["baseline_threshold"] = t_base
        rep["baseline_macro_f05"] = f_base
        log.msg("REPORT " + json.dumps(rep, default=float))

    out_dir = Path(args.out_dir) if args.out_dir else ((work / "output") if limit else tag_dir / "holdout_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    with log.stage("write outputs"):
        pieces = [(uni.ids_where(is_eval & (uni.s1_cc == k)), t) for k, t, _, _ in eval_tables]
        _write_candidate_file(out_dir, pieces)
        eval_ids = [s for ids, _ in pieces for s in ids]
        io_utils.write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", eval_ids, matches)
        if data.s1_raw_for_tsv is not None:
            write_source_tsv(data.s1_raw_for_tsv, out_dir / "source1_slice.tsv")

    summary = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"), "split": "train", "tag": tag,
        "limit_s1": limit, "train_sample": int(is_train.sum()), "max_cands_per_source": args.max_cands_per_source,
        "n_s1": len(u1), "n_holdout": len(report_ids), "n_cand_rows": n_cand_rows,
        "n_train_rows": n_train_rows, "n_train_pos": n_pos, "train_seconds": train_seconds,
        "learning_rate": args.learning_rate, "max_rounds": args.max_rounds,
        "rounds": booster.current_iteration(),
        "decision_params": {**{c: dataclasses.asdict(p) for c, p in params_by_country.items()},
                            "default": dataclasses.asdict(params_d)},
        "n_scored_kept": len(scored), "report": rep, "stages": log.stages, "total_seconds": log.total(),
        "peak_rss_gb": log.peak_gb(), "out_dir": str(out_dir),
    }
    (tag_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def run_test(args, log: RunLog, slice_hook=None) -> dict:
    limit = args.limit_s1
    work = LIMIT_DIR / f"test_{limit}" if limit else None
    tag = args.model_tag or "v1"
    with log.stage("cache"):
        _ensure_raw_cache("test")
    data = (prepare_limit("test", limit, work, log, slice_hook) if limit
            else prepare_prod("test", log, ids_only=bool(args.baseline)))
    out_dir = Path(args.out_dir) if args.out_dir else ((work / "output") if limit else config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline:
        base_path = Path(config.MODELS_DIR) / tag / "baseline.json"
        if not limit or not base_path.exists():
            base_path = Path(config.MODELS_DIR) / "baseline.json"
        t_base = float(json.loads(base_path.read_text(encoding="utf-8"))["threshold"])
        log.msg(f"baseline threshold {t_base} from {base_path}")
    else:
        import decide

        booster, cal = train.load(tag)
        params_by_country, params_d = load_decision_params(_decision_params_path(tag).read_text(encoding="utf-8"))
        tm = build_text_models(data, log)
        log.msg(f"model {tag}; decision params {params_by_country}; other countries {params_d}")
    uni = build_universe(data, log)

    matches: dict[str, list[str]] = {}
    pieces = []
    n_cand_rows = n_scored = 0
    with log.stage("baseline test" if args.baseline else "score test"):
        for k, country in enumerate(uni.countries):
            ids = uni.ids_where(uni.s1_cc == k)
            t = order_and_trim(uni.cands.subset(uni.country_rows(k)), args.max_cands_per_source)
            n_cand_rows += len(t)
            pieces.append((ids, t))
            if len(t) == 0:
                log.msg(f"country {country!r}: {len(ids)} S1, no candidates")
                continue
            if args.baseline:
                top1 = top1_per_s1(t)
                top1 = top1[top1["best_score"].to_numpy() >= t_base]
                matches.update({a: [b] for a, b in zip(top1["s1_id"], top1["s23_id"])})
                log.msg(f"country {country!r}: {len(ids)} S1, {len(t)} candidates, {len(top1)} baseline matches")
                continue
            n_claim, rank_claim = claimant_features(t)
            scored = _score_rows(t, n_claim, rank_claim, _s1_ctx(data, country), data, tm, booster, cal)
            n_scored += len(scored)
            del n_claim, rank_claim
            gc.collect()
            matches.update(decide.decide(scored, data.sib, params_by_country.get(country, params_d)))
            log.msg(f"country {country!r}: {len(ids)} S1, {len(t)} candidates, {len(scored)} kept, "
                    f"{sum(1 for s in ids if matches.get(s))} S1 with matches; rss={log.rss_gb():.1f}GB")
            del scored
            gc.collect()
        uni.cands = None
    with log.stage("write outputs"):
        all_s1 = data.s1n["entity_id"].astype(str).tolist()
        io_utils.write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", all_s1, matches)
        log.msg(f"MATCHING WRITTEN {out_dir / 'matching_results.tsv'}")
        _write_candidate_file(out_dir, pieces)
        log.msg(f"CANDIDATES WRITTEN {out_dir / 'candidate_pairs.tsv'}")
        if data.s1_raw_for_tsv is not None:
            write_source_tsv(data.s1_raw_for_tsv, out_dir / "source1_slice.tsv")
    return {
        "date": _dt.datetime.now().isoformat(timespec="seconds"), "split": "test", "tag": tag,
        "baseline": bool(args.baseline), "limit_s1": limit, "max_cands_per_source": args.max_cands_per_source,
        "n_s1": len(all_s1), "n_cand_rows": n_cand_rows, "n_scored_kept": n_scored,
        "n_s1_with_matches": sum(1 for v in matches.values() if v), "stages": log.stages,
        "total_seconds": log.total(), "peak_rss_gb": log.peak_gb(), "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# experiments.md + CLI
# ---------------------------------------------------------------------------

def log_experiment(summary: dict, path: Path = EXPERIMENTS_MD) -> None:
    s = summary
    lines = [f"\n## {s['date'][:10]} — run_all --split {s['split']} tag={s['tag']}"
             + (" (baseline)" if s.get("baseline") else ""), ""]
    if s["split"] == "train":
        r = s["report"]
        lines += [
            f"- Holdout macro F0.5: **{r['overall']:.4f}**; by country "
            + ", ".join(f"{k} {v:.4f}" for k, v in sorted(r["by_country"].items()))
            + f"; singletons {r['singletons'] if r['singletons'] is None else round(r['singletons'], 4)}",
            f"- Pair precision {r['pair_precision']:.4f}, pair recall {r['pair_recall']:.4f}; "
            f"blocking recall (holdout) {r['blocking_recall']:.4f}",
            f"- Baseline (top-1 if best_score >= {r['baseline_threshold']:.4f}): holdout macro F0.5 "
            f"{r['baseline_macro_f05']:.4f}",
            f"- Decision params: {s['decision_params']}",
            f"- Training: {s['train_sample']} S1 sample, {s['n_train_rows']} rows ({s['n_train_pos']} positive), "
            f"{s['rounds']} rounds, {s['train_seconds']:.0f} s "
            f"(lr {s.get('learning_rate', LEARNING_RATE)}, max_rounds {s.get('max_rounds', MAX_ROUNDS)})",
        ]
    else:
        lines += [f"- {s['n_s1']} S1, {s['n_cand_rows']} candidate pairs, {s['n_s1_with_matches']} S1 with matches"]
    lines += [
        f"- max_cands_per_source: {s['max_cands_per_source']}",
        "- Stage runtimes: " + ", ".join(f"{k} {v:.0f}s" for k, v in s["stages"].items()),
        f"- Total {s['total_seconds']:.0f} s; peak RSS {s['peak_rss_gb']:.1f} GB",
    ]
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# submission package (SPEC section 8 step 14)
# ---------------------------------------------------------------------------

ZIP_NAME = "Barely_Legal_submission.zip"
DOC_NAME = "Documentation_template.md"
_CODE_TOP_FILES = ["README.md", "requirements.txt", "requirements-rerank.txt", "requirements-dev.txt",
                   "requirements.lock.txt", "models.lock.json", "THIRD_PARTY_LICENSES.md", "LICENSE"]


def package(zip_path: Path | None = None, output_dir: Path | None = None,
            doc_path: Path | None = None) -> Path:
    """Build the submission zip: output/ TSVs, code/business_entity_resolution/
    (src/*.py plus README, requirements, locks and licenses) and the filled
    methodology document. No caches, models or dataset files go in."""
    import zipfile

    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
    doc_path = Path(doc_path) if doc_path else config.REPO_ROOT / DOC_NAME
    zip_path = Path(zip_path) if zip_path else config.OUTPUT_DIR / ZIP_NAME
    entries: list[tuple[Path, str]] = []
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        entries.append((output_dir / name, f"output/{name}"))
    code_arc = "code/business_entity_resolution"
    for f in sorted((config.CODE_DIR / "src").glob("*.py")):
        entries.append((f, f"{code_arc}/src/{f.name}"))
    for name in _CODE_TOP_FILES:
        entries.append((config.CODE_DIR / name, f"{code_arc}/{name}"))
    entries.append((doc_path, DOC_NAME))
    missing = [str(src) for src, _ in entries if not src.is_file()]
    if missing:
        raise FileNotFoundError("package: missing " + ", ".join(missing))
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for src, arc in entries:
            z.write(src, arc)
    return zip_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--package", action="store_true",
                    help=f"only build OUTPUT_DIR/{ZIP_NAME} from the existing output files")
    ap.add_argument("--limit-s1", type=int, default=None)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE)
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--max-cands-per-source", type=int, default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    if args.split is None and not args.package:
        ap.error("--split is required unless --package is given")
    return args


def main(argv=None, slice_hook=None) -> dict:
    args = parse_args(argv)
    config.ensure_dirs()
    if args.package:
        path = package()
        print(f"PACKAGE WRITTEN {path}")
        return {"zip": str(path)}
    log = RunLog()
    try:
        summary = (run_train if args.split == "train" else run_test)(args, log, slice_hook)
        summary["total_seconds"] = log.total()
        summary["peak_rss_gb"] = log.peak_gb()
        log.msg(f"TOTAL {log.total():.0f}s peak RSS {log.peak_gb():.2f}GB")
        if not args.limit_s1:
            log_experiment(summary)
        return summary
    finally:
        log.close()


if __name__ == "__main__":
    main(sys.argv[1:])
