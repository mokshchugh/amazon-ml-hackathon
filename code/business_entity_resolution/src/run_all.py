"""End-to-end pipeline (SPEC section 6, Task 14).

    run_all.py --split {train,test} [--limit-s1 N] [--model-tag TAG]
               [--train-sample 200000] [--max-cands-per-source K] [--baseline]
               [--out-dir DIR]

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


def _pair_keys(s1, s23) -> pa.Array:
    return pc.binary_join_element_wise(_pa_str(s1), _pa_str(s23), "\x1f")


def write_source_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write raw source records as an organiser-style source TSV (for validate())."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(SOURCE_TSV_COLS) + "\n")
        cols = [df[c].astype("string").fillna("").tolist() for c in SOURCE_TSV_COLS]
        for row in zip(*cols):
            f.write("\t".join(row) + "\n")


class GroupedLists:
    """Read-only ``.get`` mapping s1_id -> list of s23_ids over two parallel
    arrow arrays whose s1_id runs are contiguous (memory-lean input for
    ``io_utils.write_id_lists`` on 100M+ candidate pairs)."""

    def __init__(self, s1_ids, s23_ids) -> None:
        s1 = _pa_str(s1_ids)
        self._s23 = _pa_str(s23_ids)
        n = len(s1)
        if n == 0:
            self._idx: dict[str, tuple[int, int]] = {}
            return
        codes = np.asarray(pc.dictionary_encode(s1).indices, dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        ends = np.r_[starts[1:], n]
        keys = s1.take(pa.array(starts)).to_pylist()
        if len(set(keys)) != len(keys):
            raise ValueError("s1_id runs are not contiguous")
        self._idx = {k: (int(a), int(b)) for k, a, b in zip(keys, starts, ends)}

    def get(self, key, default=None):
        r = self._idx.get(key)
        if r is None:
            return default
        return self._s23.slice(r[0], r[1] - r[0]).to_pylist()


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

def top1_per_s1(cands: pd.DataFrame) -> pd.DataFrame:
    """Highest-best_score candidate per S1 record (ties -> smaller s23_id)."""
    if len(cands) == 0:
        return pd.DataFrame({"s1_id": [], "s23_id": [], "best_score": []})
    c1, _ = pd.factorize(cands["s1_id"], sort=True)
    c23, _ = pd.factorize(cands["s23_id"], sort=True)
    score = cands["best_score"].to_numpy(dtype=np.float64)
    order = np.lexsort((c23, -score, c1))
    first = order[np.r_[True, c1[order][1:] != c1[order][:-1]]]
    out = cands.iloc[first][["s1_id", "s23_id", "best_score"]].reset_index(drop=True)
    return out


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
# Per-country candidate tables and context features
# ---------------------------------------------------------------------------

def read_country_cands(cands_path: Path, ids) -> pd.DataFrame:
    """Candidate rows whose s1_id is in ``ids`` (lean columns, arrow strings)."""
    if len(ids) == 0:
        return pd.DataFrame({c: pd.Series([], dtype="string[pyarrow]") for c in ("s1_id", "s23_id", "source")}
                            | {"search_mask": pd.Series([], dtype=np.int64),
                               "best_score": pd.Series([], dtype=np.float32)})
    dset = ds.dataset(str(cands_path), format="parquet")
    tbl = dset.to_table(columns=CAND_COLS, filter=pc.field("s1_id").isin(_pa_str(list(ids))))
    df = tbl.to_pandas()
    del tbl
    return _arrow_strings(df, ["s1_id", "s23_id", "source"])


def order_and_trim(df: pd.DataFrame, max_k: int | None) -> pd.DataFrame:
    """Rows grouped by s1_id (sorted); with ``max_k`` keep the top-K rows per
    (s1_id, source) by best_score (ties -> smaller s23_id)."""
    if len(df) == 0:
        return df.reset_index(drop=True)
    c1, _ = pd.factorize(df["s1_id"], sort=True)
    if max_k is not None:
        c23, _ = pd.factorize(df["s23_id"], sort=True)
        cs, _ = pd.factorize(df["source"], sort=True)
        score = df["best_score"].to_numpy(dtype=np.float64)
        order = np.lexsort((c23, -score, cs, c1))
        g = c1[order].astype(np.int64) * (cs.max() + 1) + cs[order]
        start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
        order = order[rank < max_k]
        del c23, cs, score, g, rank
    elif np.all(c1[1:] >= c1[:-1]):
        return df.reset_index(drop=True)
    else:
        order = np.argsort(c1, kind="stable")
    return df.iloc[order].reset_index(drop=True)


def claimant_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """c_n_claimants / c_rank_among_claimants over the WHOLE table ``df``,
    exactly as ``features.add_context_features`` defines them."""
    n = len(df)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    c1, _ = pd.factorize(df["s1_id"], sort=True)
    c23, u23 = pd.factorize(df["s23_id"], sort=True)
    c1 = c1.astype(np.int64)
    c23 = c23.astype(np.int64)
    n23 = max(len(u23), 1)
    pairs = features._unique_sorted(c1 * n23 + c23)
    n_claim = np.bincount(pairs % n23, minlength=n23)[c23].astype(np.float32)
    del pairs
    rank = features._rank_within(c23, c1, df["best_score"].to_numpy(dtype=np.float64)).astype(np.float32)
    return n_claim, rank


def s1_chunk_bounds(s1_ids, max_rows: int = CHUNK_ROWS) -> list[tuple[int, int]]:
    """[start, end) row ranges of <= ``max_rows`` rows that never split a run
    of equal (contiguous) s1_ids."""
    n = len(s1_ids)
    if n == 0:
        return []
    codes = np.asarray(pc.dictionary_encode(_pa_str(s1_ids)).indices, dtype=np.int64)
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


def context_chunks(df: pd.DataFrame, s1_ctx: pd.DataFrame, sib: pd.DataFrame,
                   n_claim: np.ndarray, rank_claim: np.ndarray, max_rows: int = CHUNK_ROWS):
    """Yield (start, end, ctx) over S1-aligned chunks of ``df`` (grouped by
    s1_id) with the context columns of the whole table (see module doc)."""
    for a, b in s1_chunk_bounds(df["s1_id"], max_rows):
        sub = df.iloc[a:b].reset_index(drop=True)
        ctx = features.add_context_features(sub, s1_ctx, sib)
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


def prepare_prod(split: str, log: RunLog) -> SplitData:
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


def _countries(s1n: pd.DataFrame) -> list[tuple[str, list[str]]]:
    c = s1n["country"].astype("string").fillna("")
    out = []
    for country in sorted(c.unique().tolist()):
        out.append((country, s1n.loc[(c == country).to_numpy(), "entity_id"].astype(str).tolist()))
    return out


def _features(ctx: pd.DataFrame, data: SplitData, tm: TextModels) -> pd.DataFrame:
    return features.compute_features(ctx, data.s1n, data.s23n, data.sib, tm.idf,
                                     tfidf=tm.tfidf, addr_idf=tm.addr_idf)


def _score_rows(df, n_claim, rank_claim, s1_ctx, data, tm, booster, cal) -> pd.DataFrame:
    """Features + calibrated p for every row of ``df``; keeps p >= P_FLOOR."""
    parts = []
    for a, b, ctx in context_chunks(df, s1_ctx, data.sib, n_claim, rank_claim):
        X = _features(ctx, data, tm)
        p = train.predict_proba(booster, cal, X[features.FEATURE_COLUMNS])
        keep = p >= P_FLOOR
        parts.append(pd.DataFrame({"s1_id": X["s1_id"].to_numpy()[keep], "s23_id": X["s23_id"].to_numpy()[keep],
                                   "p": p[keep]}))
        del X, ctx
    if not parts:
        return pd.DataFrame({"s1_id": pd.Series([], dtype="string[pyarrow]"),
                             "s23_id": pd.Series([], dtype="string[pyarrow]"), "p": np.zeros(0, np.float32)})
    return _arrow_strings(pd.concat(parts, ignore_index=True), ["s1_id", "s23_id"])


def _decision_params_path(tag: str) -> Path:
    return Path(config.MODELS_DIR) / tag / "decision_params.json"


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(args, log: RunLog, slice_hook=None) -> dict:
    import decide

    limit = args.limit_s1
    work = LIMIT_DIR / f"train_{limit}" if limit else None
    tag = args.model_tag or (f"limit{limit}" if limit else "v1")
    with log.stage("cache+splits"):
        _ensure_raw_cache("train")
        splits.ensure_splits()
        holdout = splits.load_split("holdout")
        gt = pd.read_parquet(config.CACHE_DIR / "gt_pairs.parquet")
        gt = _arrow_strings(gt, ["s1_id", "s23_id"])
    data = prepare_limit("train", limit, work, log, slice_hook) if limit else prepare_prod("train", log)
    tm = build_text_models(data, log)

    all_s1 = data.s1n["entity_id"].astype(str).tolist()
    report_ids = sorted(set(all_s1) & holdout)
    eval_ids = sorted(all_s1) if limit else report_ids
    nonhold = sorted(set(all_s1) - holdout)
    rng = np.random.default_rng(config.SEED)
    sample_ids = set(np.asarray(nonhold, dtype=object)[
        np.sort(rng.choice(len(nonhold), size=min(args.train_sample, len(nonhold)), replace=False))].tolist())
    eval_set = set(eval_ids)
    report_set = set(report_ids)
    log.msg(f"S1: {len(all_s1)}; train sample {len(sample_ids)}; eval {len(eval_ids)}; holdout {len(report_ids)}")
    gt_keys = _pair_keys(gt["s1_id"], gt["s23_id"])

    # ---- pass 1: training features; keep eval rows for pass 2 --------------
    X_parts, y_parts, g_parts = [], [], []
    eval_tables = []
    top1_parts = []
    blk_hit = 0
    n_cand_rows = 0
    s1n_ctx_all = data.s1n[S1_CTX_COLS]
    with log.stage("features (train sample)"):
        for country, ids in _countries(data.s1n):
            df = order_and_trim(read_country_cands(data.cands_path, ids), args.max_cands_per_source)
            n_cand_rows += len(df)
            n_claim, rank_claim = claimant_features(df)
            s1_ctx = s1n_ctx_all[s1n_ctx_all["country"].astype("string").fillna("") == country]
            s1arr = _pa_str(df["s1_id"])
            m_train = np.asarray(pc.is_in(s1arr, value_set=_pa_str(sorted(sample_ids & set(ids)))))
            m_eval = np.asarray(pc.is_in(s1arr, value_set=_pa_str(sorted(eval_set & set(ids)))))
            m_rep = np.asarray(pc.is_in(s1arr, value_set=_pa_str(sorted(report_set & set(ids)))))
            if m_rep.any():
                rep = df[m_rep]
                top1_parts.append(top1_per_s1(rep))
                blk_hit += int(np.asarray(pc.is_in(gt_keys, value_set=_pair_keys(rep["s1_id"], rep["s23_id"])))
                               .sum()) if len(rep) else 0
            if m_eval.any():
                eval_tables.append((country, df[m_eval].reset_index(drop=True), n_claim[m_eval], rank_claim[m_eval]))
            tr = df[m_train].reset_index(drop=True)
            nc_t, rc_t = n_claim[m_train], rank_claim[m_train]
            del df, n_claim, rank_claim, s1arr
            gc.collect()
            for a, b, ctx in context_chunks(tr, s1_ctx, data.sib, nc_t, rc_t):
                X = _features(ctx, data, tm)
                keys = _pair_keys(X["s1_id"], X["s23_id"])
                y_parts.append(np.asarray(pc.is_in(keys, value_set=gt_keys)).astype(np.int8))
                X_parts.append(X[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32))
                g_parts.append(_pa_str(X["s1_id"]))
                del X, ctx, keys
            log.msg(f"country {country}: train rows so far {sum(len(y) for y in y_parts)}")
            del tr
            gc.collect()

    # ---- baseline threshold (holdout candidates) ---------------------------
    truth = _truth_map(gt, report_set)
    top1 = pd.concat(top1_parts, ignore_index=True) if top1_parts else top1_per_s1(pd.DataFrame(columns=CAND_COLS))
    t_base, f_base = tune_baseline_threshold(top1, truth, report_ids)
    n_true = sum(len(v) for v in truth.values())
    blocking_recall = blk_hit / n_true if n_true else 1.0
    tag_dir = Path(config.MODELS_DIR) / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    base_json = {"threshold": t_base, "holdout_macro_f05": f_base, "tag": tag,
                 "max_cands_per_source": args.max_cands_per_source}
    (tag_dir / "baseline.json").write_text(json.dumps(base_json, indent=2), encoding="utf-8")
    if not limit:
        (Path(config.MODELS_DIR) / "baseline.json").write_text(json.dumps(base_json, indent=2), encoding="utf-8")
    log.msg(f"baseline: threshold={t_base:.4f} holdout macro F0.5={f_base:.4f}; blocking recall={blocking_recall:.4f}")

    # ---- train ---------------------------------------------------------------
    with log.stage("train"):
        X = pd.DataFrame(np.concatenate(X_parts), columns=features.FEATURE_COLUMNS)
        del X_parts
        y = np.concatenate(y_parts).astype(np.float64)
        groups, _ = pd.factorize(pa.chunked_array(g_parts).to_pandas())
        del g_parts
        gc.collect()
        log.msg(f"training rows {len(X)}, positives {int(y.sum())}")
        params = dict(train.LGB_PARAMS, learning_rate=LEARNING_RATE)
        t_train = time.time()
        booster, oof = train.train_model(X, y, groups, max_rounds=MAX_ROUNDS, params=params)
        cal = train.fit_calibrator(oof, y)
        train_seconds = time.time() - t_train
        n_train_rows, n_pos = len(X), int(y.sum())
        del X, y, groups, oof
        gc.collect()
        train.save(booster, cal, tag)
        log.msg(f"trained in {train_seconds:.0f}s, {booster.current_iteration()} rounds")

    # ---- pass 2: score eval rows ----------------------------------------------
    scored_parts = []
    with log.stage("score eval"):
        for country, df, n_claim, rank_claim in eval_tables:
            s1_ctx = s1n_ctx_all[s1n_ctx_all["country"].astype("string").fillna("") == country]
            scored_parts.append(_score_rows(df, n_claim, rank_claim, s1_ctx, data, tm, booster, cal))
        eval_cands = [(c, df[["s1_id", "s23_id"]]) for c, df, _, _ in eval_tables]
        del eval_tables
        gc.collect()
        scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else _score_rows(
            pd.DataFrame(columns=CAND_COLS), np.zeros(0), np.zeros(0), None, data, tm, booster, cal)
        log.msg(f"scored rows kept (p >= {P_FLOOR}): {len(scored)}")

    with log.stage("tune decision"):
        rep_scored = scored[scored["s1_id"].isin(report_set)].reset_index(drop=True)
        params_d = decide.tune(rep_scored, data.sib, truth, report_ids)
        _decision_params_path(tag).write_text(json.dumps(dataclasses.asdict(params_d), indent=2), encoding="utf-8")
        log.msg(f"decision params: {params_d}")
    with log.stage("decide + report"):
        matches = decide.decide(scored, data.sib, params_d)
        s1_country = dict(zip(data.s1n["entity_id"].astype(str), data.s1n["country"].astype(str)))
        rep = evaluate.report({k: set(v) for k, v in matches.items() if k in report_set}, truth,
                              {s: s1_country[s] for s in report_ids})
        rep["blocking_recall"] = blocking_recall
        rep["baseline_threshold"] = t_base
        rep["baseline_macro_f05"] = f_base
        log.msg("REPORT " + json.dumps(rep, default=float))

    out_dir = Path(args.out_dir) if args.out_dir else ((work / "output") if limit else tag_dir / "holdout_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    with log.stage("write outputs"):
        cand_parts = []
        for i, (country, cdf) in enumerate(eval_cands):
            part = out_dir / f"_cands_part{i}.tsv"
            ids_c = [s for s in eval_ids if s1_country.get(s, "") == country]
            io_utils.write_id_lists(part, "candidate_entity_ids", ids_c, GroupedLists(cdf["s1_id"], cdf["s23_id"]))
            cand_parts.append(part)
        covered = {c for c, _ in eval_cands}
        rest = [s for s in eval_ids if s1_country.get(s, "") not in covered]
        if rest:
            part = out_dir / "_cands_part_rest.tsv"
            io_utils.write_id_lists(part, "candidate_entity_ids", rest, {})
            cand_parts.append(part)
        _concat_id_list_parts(cand_parts, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
        for part in cand_parts:
            part.unlink()
        io_utils.write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", eval_ids, matches)
        if data.s1_raw_for_tsv is not None:
            write_source_tsv(data.s1_raw_for_tsv, out_dir / "source1_slice.tsv")

    summary = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"), "split": "train", "tag": tag,
        "limit_s1": limit, "train_sample": len(sample_ids), "max_cands_per_source": args.max_cands_per_source,
        "n_s1": len(all_s1), "n_holdout": len(report_ids), "n_cand_rows": n_cand_rows,
        "n_train_rows": n_train_rows, "n_train_pos": n_pos, "train_seconds": train_seconds,
        "rounds": booster.current_iteration(), "decision_params": dataclasses.asdict(params_d),
        "report": rep, "stages": log.stages, "total_seconds": log.total(), "peak_rss_gb": log.peak_gb(),
        "out_dir": str(out_dir),
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
    data = prepare_limit("test", limit, work, log, slice_hook) if limit else prepare_prod("test", log)
    out_dir = Path(args.out_dir) if args.out_dir else ((work / "output") if limit else config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline:
        base_path = (Path(config.MODELS_DIR) / tag / "baseline.json") if limit else Path(config.MODELS_DIR) / "baseline.json"
        if not base_path.exists():
            base_path = Path(config.MODELS_DIR) / "baseline.json"
        t_base = float(json.loads(base_path.read_text(encoding="utf-8"))["threshold"])
        log.msg(f"baseline threshold {t_base} from {base_path}")
    else:
        import decide

        booster, cal = train.load(tag)
        params_d = decide.DecisionParams(**json.loads(_decision_params_path(tag).read_text(encoding="utf-8")))
        tm = build_text_models(data, log)
        log.msg(f"model {tag}; decision params {params_d}")

    all_s1 = data.s1n["entity_id"].astype(str).tolist()
    s1n_ctx_all = data.s1n[S1_CTX_COLS]
    matches: dict[str, list[str]] = {}
    cand_parts = []
    n_cand_rows = 0
    n_scored = 0
    with log.stage("score test" if not args.baseline else "baseline test"):
        for i, (country, ids) in enumerate(_countries(data.s1n)):
            df = order_and_trim(read_country_cands(data.cands_path, ids), args.max_cands_per_source)
            n_cand_rows += len(df)
            part = out_dir / f"_cands_part{i}.tsv"
            io_utils.write_id_lists(part, "candidate_entity_ids", ids, GroupedLists(df["s1_id"], df["s23_id"]))
            cand_parts.append(part)
            if len(df) == 0:
                log.msg(f"country {country!r}: {len(ids)} S1, no candidates")
                continue
            if args.baseline:
                top1 = top1_per_s1(df)
                top1 = top1[top1["best_score"].to_numpy(dtype=np.float64) >= t_base]
                matches.update({a: [b] for a, b in zip(top1["s1_id"].astype(str), top1["s23_id"].astype(str))})
                log.msg(f"country {country}: {len(ids)} S1, {len(df)} candidates, {len(top1)} baseline matches")
                del df
                continue
            n_claim, rank_claim = claimant_features(df)
            s1_ctx = s1n_ctx_all[s1n_ctx_all["country"].astype("string").fillna("") == country]
            scored = _score_rows(df, n_claim, rank_claim, s1_ctx, data, tm, booster, cal)
            n_scored += len(scored)
            del df, n_claim, rank_claim
            gc.collect()
            matches.update(decide.decide(scored, data.sib, params_d))
            log.msg(f"country {country}: {len(ids)} S1, {n_cand_rows} cand rows so far, "
                    f"{len(scored)} kept, {sum(1 for s in ids if matches.get(s))} S1 with matches")
            del scored
            gc.collect()
    with log.stage("write outputs"):
        _concat_id_list_parts(cand_parts, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
        for part in cand_parts:
            part.unlink()
        io_utils.write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", all_s1, matches)
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
            f"{s['rounds']} rounds, {s['train_seconds']:.0f} s (lr {LEARNING_RATE}, max_rounds {MAX_ROUNDS})",
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit-s1", type=int, default=None)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE)
    ap.add_argument("--max-cands-per-source", type=int, default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args(argv)


def main(argv=None, slice_hook=None) -> dict:
    args = parse_args(argv)
    config.ensure_dirs()
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
