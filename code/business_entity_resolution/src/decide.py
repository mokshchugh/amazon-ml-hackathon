"""F0.5 decision layer (SPEC section 7, step 11).

Turns calibrated pair probabilities ``(s1_id, s23_id, p)`` into one match
list per Source 1 record. Rules, applied in this order:

1. **One owner** (``resolve_owners``): each ``s23_id`` keeps only its
   highest-probability claim, and only if that claim beats the runner-up by
   at least ``margin``; a tie in ``p`` means no owner. A single claimant
   always survives.
2. **best_k**: per S1 record, candidates are sorted by ``p`` descending
   (ties: smaller ``s23_id`` first) and the top ``k`` are kept, where ``k``
   maximizes the plug-in expected F0.5 (``expected_f05``); ties -> smaller k.
3. **t_empty**: the list is emptied if the top ``p`` is below ``t_empty``.
4. **Sibling consistency**: only sibling-group members present in this S1
   record's (post-owner) candidate list count, and only groups with >= 2
   present members are touched. A candidate absent from the sibling table
   is a group of 1.

   * more than 50% of the present members kept -> also keep the remaining
     present members with ``p >= t_sib``;
   * exactly one present member kept (the others rejected) -> drop it unless
     ``p >= lone_keep``.

Everything is vectorized (sort / reduceat / bincount); nothing loops over
S1 records except the final construction of the output lists.

Assumption: sibling-table ``sib_group_id`` values identify one group across
the whole table passed in (``siblings.sibling_groups`` returns unique ids per
call; do not concatenate tables from separate calls without re-numbering).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

GRID_MARGIN = (0.05, 0.1, 0.15, 0.2, 0.3)
GRID_T_EMPTY = (0.3, 0.4, 0.5, 0.6, 0.7)
GRID_T_SIB = (0.3, 0.5, 0.7)

# Tolerance for the owner margin comparison (0.95 - 0.8 is 0.1499999... in
# binary floating point; it must still count as a gap of 0.15).
_MARGIN_EPS = 1e-9
_TIE_EPS = 1e-12


@dataclass(frozen=True)
class DecisionParams:
    margin: float = 0.15
    t_empty: float = 0.5
    t_sib: float = 0.5
    lone_keep: float = 0.95


# --------------------------------------------------------------- scoring math
def _f_curve(p_sorted: np.ndarray, starts: np.ndarray, gid: np.ndarray):
    """Per-row expected F0.5 for k = (rank within group + 1), and per-group F
    for k = 0. Groups are contiguous, rows sorted by p descending in each.
    Group results never depend on neighbouring groups (per-group cumsum)."""
    tp = pd.Series(p_sorted).groupby(gid, sort=False).cumsum().to_numpy()
    ends = np.append(starts[1:], len(p_sorted)) - 1
    total = tp[ends]
    k = (np.arange(len(p_sorted)) - starts[gid] + 1).astype(np.float64)
    fn = total[gid] - tp
    fp = k - tp
    f = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
    f0 = np.multiply.reduceat(1.0 - p_sorted, starts)
    return f, f0


def _best_k_groups(p_sorted: np.ndarray, starts: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """best_k per contiguous group (argmax over k in [0, n], ties -> smaller k)."""
    n = len(p_sorted)
    f, f0 = _f_curve(p_sorted, starts, gid)
    gmax = np.maximum.reduceat(f, starts)
    rank = np.arange(n) - starts[gid]
    first = np.minimum.reduceat(np.where(f == gmax[gid], rank, n), starts)
    return np.where(f0 >= gmax, 0, first + 1)


def _desc(probs) -> np.ndarray:
    return -np.sort(-np.asarray(probs, dtype=np.float64))


def expected_f05(probs_desc: np.ndarray, k: int) -> float:
    """Plug-in expected F0.5 of predicting the top ``k`` candidates.

    k = 0: prod(1 - p) (probability that there is no true match at all).
    k > 0: TP = sum of the top-k p, FP = k - TP, FN = sum of the rest,
    F = 1.25 TP / (1.25 TP + 0.25 FN + FP).
    """
    p = _desc(probs_desc)
    if not 0 <= k <= len(p):
        raise ValueError(f"k={k} out of range [0, {len(p)}]")
    if len(p) == 0:
        return 1.0
    starts = np.zeros(1, dtype=np.int64)
    f, f0 = _f_curve(p, starts, np.zeros(len(p), dtype=np.int64))
    return float(f0[0]) if k == 0 else float(f[k - 1])


def best_k(probs_desc: np.ndarray) -> int:
    """argmax_k expected_f05(probs_desc, k) over k in [0, n]; ties -> smaller k."""
    p = _desc(probs_desc)
    if len(p) == 0:
        return 0
    return int(_best_k_groups(p, np.zeros(1, dtype=np.int64), np.zeros(len(p), dtype=np.int64))[0])


# ------------------------------------------------------------ preparation
class _Prepared:
    """Integer-coded view of a scored frame (independent of all params)."""

    def __init__(self, scored: pd.DataFrame, sib: pd.DataFrame):
        s1c, s1u = pd.factorize(scored["s1_id"], sort=False)
        s23c, s23u = pd.factorize(scored["s23_id"], sort=True)  # codes follow lexical order
        p = scored["p"].to_numpy(dtype=np.float64)
        s1c = s1c.astype(np.int64)
        s23c = s23c.astype(np.int64)
        if (s1c < 0).any() or (s23c < 0).any() or np.isnan(p).any():
            raise ValueError("scored has missing s1_id / s23_id / p values")
        self.rows = np.arange(len(scored))
        # Duplicate (s1, s23) pairs: keep the highest p once.
        pair = s23c * max(len(s1u), 1) + s1c
        dup = pd.Series(pair).duplicated().to_numpy()
        if dup.any():
            o = np.lexsort((-p, pair))
            first = np.ones(len(o), dtype=bool)
            first[1:] = pair[o][1:] != pair[o][:-1]
            self.rows = np.sort(o[first])
            s1c, s23c, p = s1c[self.rows], s23c[self.rows], p[self.rows]
        self.s1c, self.s23c, self.p = s1c, s23c, p
        self.s1u = np.asarray(s1u, dtype=object)
        self.s23u = np.asarray(s23u, dtype=object)
        # sibling group id per s23 code (-1: not in the table -> group of 1)
        sib_ids = np.full(len(self.s23u), -1, dtype=np.int64)
        if len(sib):
            s = sib.drop_duplicates("entity_id")
            idx = pd.Index(s["entity_id"].astype(object)).get_indexer(self.s23u)
            gids = s["sib_group_id"].to_numpy(dtype=np.int64)
            sib_ids = np.where(idx >= 0, gids[np.maximum(idx, 0)], -1)
        self.sib = sib_ids[s23c]
        # owner statistics: sort by (s23, p desc, s1)
        o = np.lexsort((s1c, -p, s23c))
        s23s = s23c[o]
        st = np.flatnonzero(np.r_[True, s23s[1:] != s23s[:-1]]) if len(o) else np.zeros(0, np.int64)
        size = np.diff(np.r_[st, len(o)])
        top_p = p[o[st]]
        second_p = np.where(size > 1, p[o[np.minimum(st + 1, len(o) - 1)]], -np.inf)
        self._top_row, self._size, self._top_p, self._second_p = o[st], size, top_p, second_p

    def owner_mask(self, margin: float) -> np.ndarray:
        ok = (self._size == 1) | (
            (self._top_p > self._second_p) & (self._top_p - self._second_p >= margin - _MARGIN_EPS))
        mask = np.zeros(len(self.p), dtype=bool)
        mask[self._top_row[ok]] = True
        return mask


class _Stage:
    """Post-owner rows sorted by (s1, p desc, s23) with best_k and sibling
    bookkeeping (depends on margin only)."""

    def __init__(self, prep: _Prepared, margin: float):
        r = np.flatnonzero(prep.owner_mask(margin))
        o = r[np.lexsort((prep.s23c[r], -prep.p[r], prep.s1c[r]))]
        self.order = o
        self.s1c = prep.s1c[o]
        self.p = prep.p[o]
        n = len(o)
        if n == 0:
            self.keep = np.zeros(0, dtype=bool)
            self.top = np.zeros(0)
            self.skey = np.zeros(0, dtype=np.int64)
            self.present = np.zeros(0)
            return
        starts = np.flatnonzero(np.r_[True, self.s1c[1:] != self.s1c[:-1]])
        gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
        k = _best_k_groups(self.p, starts, gid)
        self.keep = (np.arange(n) - starts[gid]) < k[gid]
        self.top = self.p[starts][gid]
        # (s1 group, sibling group) key per row; -1 for no sibling group
        sib = prep.sib[o]
        has = sib >= 0
        skey = np.full(n, -1, dtype=np.int64)
        if has.any():
            sc, _ = pd.factorize(sib[has])
            codes, _ = pd.factorize(gid[has].astype(np.int64) * (int(sc.max()) + 1) + sc)
            skey[has] = codes
        self.skey = skey
        self.present = np.bincount(skey[has], minlength=int(skey.max()) + 1).astype(np.float64)

    def final(self, t_empty: float, t_sib: float, lone_keep: float) -> np.ndarray:
        return self._sib_rules(self.keep & (self.top >= t_empty), t_sib, lone_keep)

    def kept_counts(self, keep: np.ndarray) -> np.ndarray:
        has = self.skey >= 0
        return np.bincount(self.skey[has], weights=keep[has], minlength=len(self.present))

    def _sib_rules(self, keep, t_sib, lone_keep, kept=None):
        has = self.skey >= 0
        if not has.any():
            return keep
        if kept is None:
            kept = self.kept_counts(keep)
        key = np.where(has, self.skey, 0)
        pres = np.where(has, self.present[key], 0.0)
        kc = np.where(has, kept[key], 0.0)
        multi = pres >= 2
        add = multi & (2 * kc > pres) & ~keep & (self.p >= t_sib)
        drop = multi & (kc == 1) & keep & (self.p < lone_keep)
        return (keep | add) & ~drop


def _to_lists(prep: _Prepared, stage: _Stage, final: np.ndarray) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {s: [] for s in prep.s1u.tolist()}
    idx = np.flatnonzero(final)
    if len(idx) == 0:
        return out
    s1k = stage.s1c[idx]
    names = prep.s23u[prep.s23c[stage.order[idx]]]
    cut = np.flatnonzero(s1k[1:] != s1k[:-1]) + 1
    keys = prep.s1u[s1k[np.r_[0, cut]]].tolist()
    for key, arr in zip(keys, np.split(names, cut)):
        out[key] = arr.tolist()
    return out


# ------------------------------------------------------------------ public API
def resolve_owners(scored: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Keep, for each ``s23_id``, only its top claim, and only if it beats the
    second-best claim by at least ``margin`` (ties in p -> no owner; a single
    claimant always survives). Returns the surviving rows of ``scored``."""
    prep = _Prepared(scored, pd.DataFrame())
    return scored.iloc[prep.rows[prep.owner_mask(margin)]]


def decide(scored: pd.DataFrame, sib: pd.DataFrame, params: DecisionParams) -> dict[str, list[str]]:
    """Final match list for every ``s1_id`` in ``scored`` (possibly empty),
    sorted by p descending then ``s23_id``."""
    prep = _Prepared(scored, sib)
    stage = _Stage(prep, params.margin)
    final = stage.final(params.t_empty, params.t_sib, params.lone_keep)
    return _to_lists(prep, stage, final)


def tune_scores(scored: pd.DataFrame, sib: pd.DataFrame, truth: Mapping[str, set[str]],
                s1_ids: Iterable[str], lone_keep: float = DecisionParams.lone_keep) -> pd.DataFrame:
    """Macro F0.5 (as ``evaluate.macro_f05`` computes it) of ``decide`` for
    every grid point. Columns: margin, t_empty, t_sib, score (75 rows)."""
    prep = _Prepared(scored, sib)
    s1_ids = list(s1_ids)
    n1, n23 = len(prep.s1u), max(len(prep.s23u), 1)
    s1_index = pd.Index(prep.s1u)
    s23_index = pd.Index(prep.s23u)
    # true pairs present among the candidates
    t_s1, t_s23 = [], []
    for k, v in truth.items():
        for e in v:
            t_s1.append(k)
            t_s23.append(e)
    tc1 = s1_index.get_indexer(pd.Index(t_s1, dtype=object)) if t_s1 else np.zeros(0, dtype=np.int64)
    tc23 = s23_index.get_indexer(pd.Index(t_s23, dtype=object)) if t_s23 else np.zeros(0, dtype=np.int64)
    ok = (tc1 >= 0) & (tc23 >= 0)
    true_keys = np.unique(tc1[ok].astype(np.int64) * n23 + tc23[ok])
    is_true_all = np.isin(prep.s1c * n23 + prep.s23c, true_keys)
    # |truth| for every requested id, and its code in scored (-1: no candidates)
    nt = np.array([len(truth.get(s, ())) for s in s1_ids], dtype=np.float64)
    ids_code = s1_index.get_indexer(pd.Index(s1_ids, dtype=object)) if s1_ids else np.zeros(0, dtype=np.int64)
    in_scored = ids_code >= 0
    code = np.maximum(ids_code, 0)
    empty_score = (nt == 0).astype(np.float64)
    n_ids = len(s1_ids)

    def macro(stage: _Stage, final: np.ndarray) -> float:
        if n_ids == 0:
            return 0.0
        tp = np.bincount(stage.s1c, weights=final & is_true_all[stage.order], minlength=n1)
        npred = np.bincount(stage.s1c, weights=final, minlength=n1)
        tp_i = np.where(in_scored, tp[code], 0.0)
        np_i = np.where(in_scored, npred[code], 0.0)
        denom = 0.25 * nt + np_i
        f = np.where(np_i == 0, empty_score,
                     np.where(tp_i > 0, 1.25 * tp_i / np.where(denom > 0, denom, 1.0), 0.0))
        return float(f.sum() / n_ids)

    rows = []
    for m in GRID_MARGIN:
        stage = _Stage(prep, m)
        for te in GRID_T_EMPTY:
            keep = stage.keep & (stage.top >= te)
            kept = stage.kept_counts(keep)
            for ts in GRID_T_SIB:
                final = stage._sib_rules(keep, ts, lone_keep, kept)
                rows.append((m, te, ts, macro(stage, final)))
    return pd.DataFrame(rows, columns=["margin", "t_empty", "t_sib", "score"])


def tune(scored: pd.DataFrame, sib: pd.DataFrame, truth: Mapping[str, set[str]],
         s1_ids: Iterable[str]) -> DecisionParams:
    """Grid-search margin x t_empty x t_sib for the best macro F0.5; ties go to
    the grid point closest (Euclidean) to the defaults, then grid order."""
    grid = tune_scores(scored, sib, truth, s1_ids)
    d = DecisionParams()
    dist = np.sqrt((grid.margin - d.margin) ** 2 + (grid.t_empty - d.t_empty) ** 2
                   + (grid.t_sib - d.t_sib) ** 2)
    best = grid.score >= grid.score.max() - _TIE_EPS
    cand = grid[best].assign(dist=dist[best]).sort_values("dist", kind="stable")
    r = cand.iloc[0]
    return DecisionParams(margin=float(r.margin), t_empty=float(r.t_empty), t_sib=float(r.t_sib),
                          lone_keep=d.lone_keep)
