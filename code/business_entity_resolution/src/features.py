"""Pair features (SPEC step 8).

``add_context_features`` adds the ``c_*`` columns; it needs the WHOLE
per-country candidate table (claimant counts and ranks look at every
candidate). ``compute_features`` adds everything else and is safe to call
on any chunk of that table: every value depends only on the pair, the
record frames and the fitted IDF / TF-IDF models, never on the chunk.

Throughput design (about 165M test pairs):
  * record attributes are encoded ONCE per distinct record (integer codes,
    CSR token lists, first-house-number value, TF-IDF rows) and gathered
    into pairs by integer position -- no string merges per pair;
  * string similarities use ``rapidfuzz.process.cpdist`` (element-wise, C++);
  * set overlaps (IDF word Jaccard, shared house numbers, TF-IDF cosine) use
    one vectorised sorted-key ``searchsorted`` per chunk;
  * pairs are processed in 200k-row chunks on a joblib thread pool
    (numpy and rapidfuzz release the GIL).

There is deliberately no country feature. Missing values (either side
lacks the field) are NaN. Notes on definitions:
  * n_min_shared_idf = IDF of the rarest shared name word (the largest IDF
    among shared words; 0 when no word is shared);
  * n_sim_before_translit = fuzz.ratio of the raw lower-cased business names;
  * n_alt_best = best token_set_ratio of either side's alt_name against the
    other's name_clean / alt_name (NaN when neither has an alt_name);
  * h_* use the first house number (street chunk first) for eq / gap / edit;
    h_rel_gap = |a-b| / max(a, b, 1);
  * a_idf_jaccard uses the whitespace tokens of addr_clean with ``addr_idf``;
  * a_sib_best_set = token_set_ratio of the S1 address against the S2/S3
    record's sibling-group best address (NaN when not in ``sib``).
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

CHUNK = 200_000

NAME_FEATURES = [
    "n_ratio", "n_token_sort", "n_token_set", "n_partial", "n_jaro_winkler",
    "n_key_lev", "n_tfidf_cos", "n_idf_jaccard", "n_min_shared_idf",
    "n_unshared_cnt", "n_alt_best",
]
LEGAL_FEATURES = ["legal_state"]
SCRIPT_FEATURES = ["was_indic", "n_sim_before_translit"]
HOUSE_FEATURES = ["h_first_eq", "h_any_shared", "h_abs_gap", "h_rel_gap", "h_edit", "h_cnt_s1", "h_cnt_s23"]
ADDRESS_FEATURES = [
    "a_postcode_eq", "a_city_eq", "a_city_fuzzy", "a_state_eq", "a_street_set",
    "a_token_set", "a_idf_jaccard", "a_has_s1", "a_has_s23", "a_sib_best_set",
]
CONTEXT_FEATURES = [
    "c_rank_in_s1", "c_gap_to_best", "c_n_claimants", "c_rank_among_claimants",
    "c_name_freq", "c_source_is_s3", "c_sib_size", "c_search_mask",
]
FEATURE_COLUMNS: list[str] = (
    NAME_FEATURES + LEGAL_FEATURES + SCRIPT_FEATURES + HOUSE_FEATURES + ADDRESS_FEATURES + CONTEXT_FEATURES
)

# record columns the pair features read
_RECORD_COLUMNS = ["entity_id", "business_name", "name_clean", "name_sorted", "name_key", "legal", "alt_name",
                   "was_indic", "addr_clean", "house_nums", "street", "city", "state", "postcode", "has_addr"]

# legal_state values
LEGAL_SAME, LEGAL_COMPATIBLE, LEGAL_CONFLICT, LEGAL_MISSING = 0, 1, 2, 3

_LEADING_INT = r"^[a-z]{0,2}(\d+)"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _str_array(s) -> pa.Array:
    """String column / array -> pyarrow large_string array, missing -> ''.
    Arrow-backed pandas columns convert without copying through Python."""
    try:
        arr = pa.array(s, from_pandas=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        arr = None
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if arr is None or not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)
                           or pa.types.is_null(arr.type)):
        vals = np.asarray(s, dtype=object)
        vals = np.where(pd.isna(vals), "", vals).astype(str).astype(object) if len(vals) else vals
        arr = pa.array(vals, type=pa.large_string())
    return pc.fill_null(arr.cast(pa.large_string()), "")


def _obj(arr: pa.Array) -> np.ndarray:
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=object)


def _list_array(s: pd.Series) -> pa.ListArray:
    """list[str] column (lists / numpy arrays / None) -> pyarrow list array."""
    vals = [x if x is not None and not (isinstance(x, float) and math.isnan(x)) else [] for x in s.tolist()]
    return pa.array(vals, type=pa.large_list(pa.large_string()))


def _split_words(arr: pa.Array) -> pa.ListArray:
    """Space-separated tokens (inputs are whitespace-normalised; empty
    tokens are dropped later by ``_csr_codes``)."""
    return pc.split_pattern(arr, " ")


def _csr_codes(lists: pa.Array):
    """Flatten a list array -> (non-empty token strings, record index per token)."""
    flat = pc.list_flatten(lists)
    parent = np.asarray(pc.list_parent_indices(lists), dtype=np.int64)
    keep = np.asarray(pc.greater(pc.utf8_length(flat), 0).fill_null(False))
    return flat.filter(pa.array(keep)), parent[keep]


def _unique_sorted(x: np.ndarray) -> np.ndarray:
    """np.unique for big int64 arrays via one sort (faster than the hash path)."""
    x = np.sort(x)
    return x[np.r_[True, x[1:] != x[:-1]]] if len(x) else x


def _sorted_unique_csr(rec: np.ndarray, ids: np.ndarray, n_rec: int, width: int):
    """Per-record sorted unique ids as CSR (indptr, ids)."""
    key = _unique_sorted(rec.astype(np.int64) * width + ids.astype(np.int64))
    r = key // width
    ids = (key - r * width).astype(np.int64)
    indptr = np.zeros(n_rec + 1, dtype=np.int64)
    np.cumsum(np.bincount(r, minlength=n_rec), out=indptr[1:])
    return indptr, ids


def _gather(indptr: np.ndarray, rows: np.ndarray):
    """Element positions of the CSR rows ``rows`` -> (pair idx per element, element idx)."""
    lens = indptr[rows + 1] - indptr[rows]
    total = int(lens.sum())
    pair = np.repeat(np.arange(len(rows), dtype=np.int64), lens)
    if total == 0:
        return pair, np.zeros(0, dtype=np.int64)
    starts = np.cumsum(lens) - lens
    elem = np.repeat(indptr[rows] - starts, lens) + np.arange(total, dtype=np.int64)
    return pair, elem


def _match(ptr1, ids1, ptr2, ids2, i, j, width):
    """Shared elements of CSR rows ptr1[i] and ptr2[j], pair by pair.
    Rows must hold sorted unique ids. -> (pair, elem in ids1, elem in ids2)."""
    pa_, ea = _gather(ptr1, i)
    pb, eb = _gather(ptr2, j)
    if len(ea) == 0 or len(eb) == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    ka = pa_ * width + ids1[ea]
    kb = pb * width + ids2[eb]
    pos = np.minimum(np.searchsorted(kb, ka), len(kb) - 1)
    hit = kb[pos] == ka
    return pa_[hit], ea[hit], eb[pos[hit]]


def _nan_where(x: np.ndarray, missing: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x[missing] = np.nan
    return x


def _cp(a, b, scorer, valid: np.ndarray | None = None, workers: int = 1,
        key: np.ndarray | None = None) -> np.ndarray:
    """Element-wise rapidfuzz scores, NaN where not ``valid``. With ``key``
    (equal key <=> equal string pair) each distinct pair is scored once."""
    out = np.full(len(a), np.nan, dtype=np.float32)
    idx = np.arange(len(a)) if valid is None else np.flatnonzero(valid)
    if not len(idx):
        return out
    if key is None:
        out[idx] = process.cpdist(a[idx], b[idx], scorer=scorer, workers=workers, dtype=np.float32)
    else:
        _, first, inv = np.unique(key[idx], return_index=True, return_inverse=True)
        rep = idx[first]
        out[idx] = process.cpdist(a[rep], b[rep], scorer=scorer, workers=workers, dtype=np.float32)[inv]
    return out


# ---------------------------------------------------------------------------
# IDF / TF-IDF models
# ---------------------------------------------------------------------------

def build_idf(names: pd.Series) -> dict[str, float]:
    """Smoothed word IDF over ``names``: log((1+N)/(1+df)) + 1, df counted
    once per name. Words are whitespace tokens."""
    arr = _str_array(names)
    flat, rec = _csr_codes(_split_words(arr))
    if len(flat) == 0:
        return {}
    enc = pc.dictionary_encode(flat)
    codes = np.asarray(enc.indices, dtype=np.int64)
    words = enc.dictionary.to_pylist()
    width = len(words)
    key = _unique_sorted(rec * width + codes)
    df = np.bincount(key % width, minlength=width)
    n = float(len(arr))
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    return dict(zip(words, idf.tolist()))


def _char3(names: pa.Array):
    """char_wb 3-grams (as in sklearn: each word padded with one space) of
    each string -> (doc index per gram, gram code). A gram's code packs its
    three code points (21 bits each); cross-word grams (middle char a space)
    are dropped."""
    names = pc.utf8_trim_whitespace(names)
    sp1, empty = pa.scalar(" ", names.type), pa.scalar("", names.type)
    padded = pc.binary_join_element_wise(sp1, pc.replace_substring(names, " ", "  "), sp1, empty)
    lens = np.array(pc.utf8_length(padded), dtype=np.int64)
    lens[np.asarray(pc.equal(names, ""))] = 0  # empty name -> no grams
    joined = pc.if_else(pc.equal(names, ""), empty, padded).cast(pa.large_string())
    offs = np.frombuffer(joined.buffers()[1], dtype=np.int64)[joined.offset:joined.offset + len(joined) + 1]
    data = joined.buffers()[2]
    raw = memoryview(data)[offs[0]:offs[-1]] if data is not None and len(offs) else b""
    cp = np.frombuffer(bytes(raw).decode("utf-8").encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    starts = np.cumsum(lens) - lens
    n_g = np.maximum(lens - 2, 0)
    doc = np.repeat(np.arange(len(lens), dtype=np.int64), n_g)
    pos = np.repeat(starts - (np.cumsum(n_g) - n_g), n_g) + np.arange(int(n_g.sum()), dtype=np.int64)
    keep = cp[pos + 1] != 32
    doc, pos = doc[keep], pos[keep]
    return doc, (cp[pos] << 42) | (cp[pos + 1] << 21) | cp[pos + 2]


class CharTfidf:
    """Char 3-gram TF-IDF (char_wb grams, sublinear tf, smooth idf, L2 rows)
    in numpy, so transforming millions of names takes seconds."""

    def __init__(self, grams: np.ndarray, idf: np.ndarray):
        self.grams = grams  # sorted unique gram codes
        self.idf = idf
        self._index = None

    def _columns(self, code: np.ndarray) -> np.ndarray:
        if self._index is None:
            self._index = pd.Index(self.grams)
        return self._index.get_indexer(code)

    def transform(self, names):
        """-> scipy CSR (rows L2-normalised, column indices sorted)."""
        arr = names if isinstance(names, pa.Array) else _str_array(pd.Series(names, dtype=object))
        n, v = len(arr), max(len(self.grams), 1)
        doc, code = _char3(arr)
        col = self._columns(code)
        ok = col >= 0
        key = np.sort(doc[ok] * v + col[ok])
        first = np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, bool)
        uniq = key[first]
        tf = np.diff(np.r_[np.flatnonzero(first), len(key)]).astype(np.float64)
        d, c = uniq // v, uniq % v
        w = (1.0 + np.log(tf)) * self.idf[c] if len(c) else np.zeros(0)
        norm = np.sqrt(np.bincount(d, weights=w * w, minlength=n))
        w = w / np.where(norm[d] > 0, norm[d], 1.0)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(d, minlength=n), out=indptr[1:])
        return sp.csr_matrix((w.astype(np.float32), c, indptr), shape=(n, v))


def build_tfidf(names: pd.Series) -> CharTfidf:
    """Fit the char 3-gram TF-IDF used for ``n_tfidf_cos`` (on name_sorted):
    idf = ln((1+N)/(1+df)) + 1 over the N given names."""
    arr = _str_array(names)
    doc, code = _char3(arr)
    grams = _unique_sorted(code)
    v = max(len(grams), 1)
    key = _unique_sorted(doc * v + np.searchsorted(grams, code))
    df = np.bincount(key % v, minlength=len(grams))
    return CharTfidf(grams, np.log((1.0 + len(arr)) / (1.0 + df)) + 1.0)


# ---------------------------------------------------------------------------
# Context features (whole per-country candidate table)
# ---------------------------------------------------------------------------

def _rank_within(group: np.ndarray, tie: np.ndarray, score: np.ndarray) -> np.ndarray:
    """1-based rank by descending score inside each group, ties by smaller ``tie`` code."""
    n = len(group)
    order = np.lexsort((tie, -score, group))
    g = group[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if n else np.zeros(0, np.int64)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) + 1
    return rank


def add_context_features(cands: pd.DataFrame, s1n: pd.DataFrame, sib: pd.DataFrame) -> pd.DataFrame:
    """Return ``cands`` with the ``c_*`` columns (float32) added.

    Call on the WHOLE per-country candidate table: claimant counts and
    ranks need every candidate. Ties on best_score are broken by the smaller
    id (s23_id for c_rank_in_s1, s1_id for c_rank_among_claimants)."""
    out = cands.copy()
    n = len(out)
    score = out["best_score"].to_numpy(dtype=np.float64)
    c1, u1 = pd.factorize(out["s1_id"], sort=True)
    c23, u23 = pd.factorize(out["s23_id"], sort=True)

    # rank within the S1 record (ties -> smaller s23_id first)
    best = np.full(c1.max() + 1 if n else 0, -np.inf)
    np.maximum.at(best, c1, score)
    out["c_rank_in_s1"] = _rank_within(c1, c23, score).astype(np.float32)
    out["c_gap_to_best"] = (best[c1] - score).astype(np.float32) if n else np.zeros(0, np.float32)

    # claimants of each S2/S3 record: DISTINCT S1 records (duplicate pairs count once)
    n23 = c23.max() + 1 if n else 0
    pairs = _unique_sorted(c1.astype(np.int64) * max(n23, 1) + c23)
    n_claim = np.bincount(pairs % max(n23, 1), minlength=n23)
    out["c_n_claimants"] = n_claim[c23].astype(np.float32)
    out["c_rank_among_claimants"] = _rank_within(c23, c1, score).astype(np.float32)

    # name frequency among S1 records of the same country
    keys = [s1n["name_sorted"].astype("string").fillna("")]
    if "country" in s1n.columns:
        keys.insert(0, s1n["country"].astype("string").fillna(""))
    freq = s1n.groupby(keys, sort=False, dropna=False)["entity_id"].transform("size").to_numpy(dtype=np.float64)
    freq[(keys[-1] == "").to_numpy(dtype=bool)] = np.nan  # empty name: no frequency
    pos = _find(s1n["entity_id"], np.asarray(u1, dtype=object))
    out["c_name_freq"] = np.where(pos >= 0, np.append(freq, np.nan)[pos], np.nan)[c1].astype(np.float32)

    out["c_source_is_s3"] = np.asarray(pc.equal(_str_array(out["source"]), "S3")).astype(np.float32)
    pos = _find(sib["entity_id"], np.asarray(u23, dtype=object))
    size = np.append(sib["sib_group_size"].to_numpy(dtype=np.float64), 1.0)
    out["c_sib_size"] = size[pos][c23].astype(np.float32)
    out["c_search_mask"] = out["search_mask"].to_numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Record encoding (once per distinct record)
# ---------------------------------------------------------------------------

def _find(universe, ids: np.ndarray, required: bool = False) -> np.ndarray:
    """Position in ``universe`` (an id column) of each of the unique ``ids``,
    -1 if absent (KeyError if ``required``). Hashes only ``ids`` and scans
    the (possibly 10M-row) universe in parallel slices."""
    eid = _str_array(universe)
    value_set = pa.array(ids, type=pa.large_string())
    n, step = len(eid), 1_000_000

    def scan(start):
        return np.asarray(pc.index_in(eid.slice(start, step), value_set=value_set).fill_null(-1))

    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(scan, range(0, n, step)))
    where = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    rows = np.full(len(ids), -1, dtype=np.int64)
    hit = np.flatnonzero(where >= 0)
    rows[where[hit]] = hit
    if required and (rows < 0).any():
        missing = ids[rows < 0][:5]
        raise KeyError(f"candidate ids missing from the record frame: {list(missing)}")
    return rows


def _rows_for(frame: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """Row positions in ``frame`` of each id in ``ids`` (unique); KeyError if absent."""
    return _find(frame["entity_id"], ids, required=True)


def _take(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Rows ``rows`` of the record columns the features read (no full-frame copy)."""
    cols = [c for c in _RECORD_COLUMNS if c in frame.columns]
    with ThreadPoolExecutor(max_workers=8) as pool:
        taken = list(pool.map(lambda c: frame[c].iloc[rows].reset_index(drop=True), cols))
    return pd.DataFrame(dict(zip(cols, taken)))


def _legal_table(legal_uniq: list[str]) -> np.ndarray:
    """legal_state for every pair of distinct legal strings; the extra last
    row/column (index -1) is 'missing'."""
    sets = [frozenset(u.split()) for u in legal_uniq]
    k = len(sets)
    tab = np.full((k + 1, k + 1), LEGAL_MISSING, dtype=np.float32)
    for a in range(k):
        for b in range(k):
            sa, sb = sets[a], sets[b]
            if sa and sb:
                tab[a, b] = LEGAL_SAME if sa == sb else LEGAL_COMPATIBLE if (sa <= sb or sb <= sa) else LEGAL_CONFLICT
    return tab


def _enc_strings(arrs: dict) -> dict:
    e = {}
    for c in ("name_clean", "name_key", "alt_name", "street", "addr_clean", "city", "raw_lower"):
        e[c] = _obj(arrs[c])
    # distinct-string ids, so repeated string pairs are scored once
    for c in ("name_clean", "name_key", "street", "addr_clean", "city", "raw_lower"):
        enc = pc.dictionary_encode(arrs[c])
        e[c + "_id"] = (np.asarray(enc.indices, dtype=np.int64), max(len(enc.dictionary), 1))
    # categorical codes (-1 = missing), shared between both sides
    for c in ("city", "state", "postcode", "legal"):
        enc = pc.dictionary_encode(arrs[c])
        e[c + "_code"] = np.where(np.asarray(pc.equal(arrs[c], "")), -1, np.asarray(enc.indices, dtype=np.int64))
        if c == "legal":
            e["legal_tab"] = _legal_table(enc.dictionary.to_pylist())
    return e


def _enc_words(tag: str, arr: pa.Array, weights: Mapping[str, float]) -> dict:
    """Per-record sorted unique word ids + IDF weights (unseen word -> max IDF)."""
    n = len(arr)
    flat, rec = _csr_codes(_split_words(arr))
    enc = pc.dictionary_encode(flat)
    words = enc.dictionary.to_pylist()
    width = max(len(words), 1)
    indptr, ids = _sorted_unique_csr(rec, np.asarray(enc.indices, dtype=np.int64), n, width)
    default = max(weights.values()) if len(weights) else 1.0
    w = np.array([weights.get(t, default) for t in words], dtype=np.float64)
    wt = w[ids] if len(ids) else np.zeros(0)
    cnt = np.diff(indptr)
    return {tag: (indptr, ids, wt, width), tag + "_cnt": cnt,
            tag + "_sum": np.bincount(np.repeat(np.arange(n), cnt), weights=wt, minlength=n)}


def _enc_house(col: pd.Series) -> dict:
    n = len(col)
    flat, rec = _csr_codes(_list_array(col))
    enc = pc.dictionary_encode(flat)
    width = max(len(enc.dictionary), 1)
    codes = np.asarray(enc.indices, dtype=np.int64)
    cnt = np.bincount(rec, minlength=n)
    first_elem = np.cumsum(cnt) - cnt
    first_code = np.where(cnt > 0, codes[np.minimum(first_elem, max(len(codes) - 1, 0))] if len(codes) else -1, -1)
    dict_str = np.append(_obj(enc.dictionary), "")  # code -1 -> ""
    # numeric value of each distinct number string (leading digits after an optional 1-2 letter prefix)
    num = pd.Series(dict_str).str.lower().str.extract(_LEADING_INT, expand=False).str.slice(0, 15)
    dict_val = pd.to_numeric(num, errors="coerce").to_numpy(dtype=np.float64)
    return {"h_cnt": cnt, "h_first_code": first_code, "h_first_str": dict_str[first_code],
            "h_first_val": dict_val[first_code], "hn": _sorted_unique_csr(rec, codes, n, width) + (width,)}


def _enc_sib(ids: pd.Series, own_addr: pa.Array, sib: pd.DataFrame) -> dict:
    pos = _find(sib["entity_id"], np.asarray(ids, dtype=object))
    best = np.full(len(pos), "", dtype=object)
    hit = np.flatnonzero(pos >= 0)
    best[hit] = _obj(_str_array(sib["sib_best_addr"].iloc[pos[hit]]))
    return {"sib_best": best, "sib_is_own": best == _obj(own_addr)}


def _enc_tfidf(names: pa.Array, tfidf: CharTfidf) -> dict:
    enc = pc.dictionary_encode(names)
    x = tfidf.transform(enc.dictionary)
    return {"tf": (x.indptr.astype(np.int64), x.indices.astype(np.int64), x.data.astype(np.float64), x.shape[1]),
            "tf_row": np.asarray(enc.indices, dtype=np.int64), "ns_empty": np.asarray(pc.equal(names, ""))}


def _encode(sub1: pd.DataFrame, sub2: pd.DataFrame, sib: pd.DataFrame, idf: Mapping[str, float],
            addr_idf: Mapping[str, float], tfidf: CharTfidf) -> dict:
    """Encode every distinct record once (S1 rows first, then S2/S3 rows).
    Independent parts run on a thread pool (pyarrow / numpy release the GIL)."""
    n1 = len(sub1)
    both = pd.concat([sub1, sub2], ignore_index=True)

    def strings(col):
        return _str_array(both[col]) if col in both.columns else pa.array([""] * len(both), type=pa.large_string())

    arrs = {c: strings(c) for c in ("name_clean", "name_sorted", "name_key", "legal", "alt_name",
                                    "street", "addr_clean", "city", "state", "postcode")}
    arrs["raw_lower"] = pc.utf8_lower(strings("business_name"))
    e: dict = {"n1": n1,
               "was_indic": both["was_indic"].to_numpy(dtype=bool) if "was_indic" in both.columns
               else np.zeros(len(both), bool),
               "has_addr": both["has_addr"].to_numpy(dtype=bool)}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(_enc_strings, arrs),
                pool.submit(_enc_words, "nw", arrs["name_clean"], idf),
                pool.submit(_enc_words, "aw", arrs["addr_clean"], addr_idf),
                pool.submit(_enc_house, both["house_nums"]),
                pool.submit(_enc_sib, sub2["entity_id"], arrs["addr_clean"][n1:], sib),
                pool.submit(_enc_tfidf, arrs["name_sorted"], tfidf)]
        for job in jobs:
            e.update(job.result())
    return e


# ---------------------------------------------------------------------------
# Pair features for one chunk
# ---------------------------------------------------------------------------

def _overlap(e, tag, i, j):
    indptr, ids, wt, width = e[tag]
    p, a, b = _match(indptr, ids, indptr, ids, i, j, width)
    n = len(i)
    shared_w = np.bincount(p, weights=wt[a], minlength=n)
    shared_n = np.bincount(p, minlength=n)
    max_w = np.zeros(n)
    if len(p):
        np.maximum.at(max_w, p, wt[a])
    return shared_w, shared_n, max_w


def _chunk(e: dict, i1: np.ndarray, i2: np.ndarray, ctx: dict, workers: int = 1) -> dict:
    j = i2 + e["n1"]          # S2/S3 rows sit after the S1 rows
    i = i1
    f: dict[str, np.ndarray] = {}
    g = lambda c: (e[c][i], e[c][j])  # noqa: E731

    def key(c):
        ids, k = e[c + "_id"]
        return ids[i] * k + ids[j]

    # --- name ------------------------------------------------------------
    a, b = g("name_clean")
    ok = (a != "") & (b != "")
    nk = key("name_clean")
    f["n_ratio"] = _cp(a, b, fuzz.ratio, ok, workers=workers, key=nk)
    f["n_token_sort"] = _cp(a, b, fuzz.token_sort_ratio, ok, workers=workers, key=nk)
    f["n_token_set"] = _cp(a, b, fuzz.token_set_ratio, ok, workers=workers, key=nk)
    f["n_partial"] = _cp(a, b, fuzz.partial_ratio, ok, workers=workers, key=nk)
    f["n_jaro_winkler"] = _cp(a, b, JaroWinkler.normalized_similarity, ok, workers=workers, key=nk)
    ka, kb = g("name_key")
    f["n_key_lev"] = _cp(ka, kb, Levenshtein.distance, (ka != "") & (kb != ""), workers=workers, key=key("name_key"))

    # TF-IDF cosine, once per distinct (name_sorted, name_sorted) pair
    ti, tj = e["tf_row"][i], e["tf_row"][j]
    ptr, idx, dat, width = e["tf"]
    _, rep, inv = np.unique(ti * (len(ptr) - 1) + tj, return_index=True, return_inverse=True)
    p, ea, eb = _match(ptr, idx, ptr, idx, ti[rep], tj[rep], width)
    cos = np.bincount(p, weights=dat[ea] * dat[eb], minlength=len(rep))[inv]
    f["n_tfidf_cos"] = _nan_where(np.minimum(cos, 1.0), e["ns_empty"][i] | e["ns_empty"][j])

    # IDF word overlap, once per distinct (name_clean, name_clean) pair
    _, rep, inv = np.unique(nk, return_index=True, return_inverse=True)
    ri, rj = i[rep], j[rep]
    sw, sn, mw = (x[inv] for x in _overlap(e, "nw", ri, rj))
    union = e["nw_sum"][i] + e["nw_sum"][j] - sw
    empty = (e["nw_cnt"][i] == 0) | (e["nw_cnt"][j] == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["n_idf_jaccard"] = _nan_where(np.where(union > 0, sw / union, 0.0), empty)
    f["n_min_shared_idf"] = _nan_where(mw, empty)
    f["n_unshared_cnt"] = _nan_where((e["nw_cnt"][i] + e["nw_cnt"][j] - 2 * sn).astype(np.float64), empty)

    al1, al2 = g("alt_name")
    has_alt = (al1 != "") | (al2 != "")
    alt = np.full(len(i), np.nan, dtype=np.float32)
    if has_alt.any():
        s = np.flatnonzero(has_alt)
        cand = np.full((3, len(s)), np.nan, dtype=np.float32)
        for k, (x, y) in enumerate(((al1[s], b[s]), (a[s], al2[s]), (al1[s], al2[s]))):
            cand[k] = _cp(x, y, fuzz.token_set_ratio, (x != "") & (y != ""), workers=workers)
        with np.errstate(all="ignore"):
            best = np.where(np.isnan(cand).all(0), np.nan, np.nanmax(np.nan_to_num(cand, nan=-1), axis=0))
        alt[s] = best
    f["n_alt_best"] = alt

    # --- legal / script ----------------------------------------------------
    f["legal_state"] = e["legal_tab"][e["legal_code"][i], e["legal_code"][j]]
    wi = e["was_indic"][i] | e["was_indic"][j]
    f["was_indic"] = wi.astype(np.float32)
    ra, rb = g("raw_lower")
    f["n_sim_before_translit"] = _cp(ra, rb, fuzz.ratio, (ra != "") & (rb != ""), workers=workers, key=key("raw_lower"))

    # --- house numbers -----------------------------------------------------
    c1, c2 = e["h_cnt"][i], e["h_cnt"][j]
    miss = (c1 == 0) | (c2 == 0)
    f["h_first_eq"] = _nan_where((e["h_first_code"][i] == e["h_first_code"][j]).astype(np.float32), miss)
    hp, hwid = e["hn"][:2], e["hn"][2]
    p, _, _ = _match(hp[0], hp[1], hp[0], hp[1], i, j, hwid)
    f["h_any_shared"] = _nan_where((np.bincount(p, minlength=len(i)) > 0).astype(np.float32), miss)
    v1, v2 = e["h_first_val"][i], e["h_first_val"][j]
    gap = np.abs(v1 - v2)
    f["h_abs_gap"] = _nan_where(gap, miss)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["h_rel_gap"] = _nan_where(gap / np.maximum(np.maximum(v1, v2), 1.0), miss)
    fa, fb = g("h_first_str")
    f["h_edit"] = _cp(fa, fb, Levenshtein.distance, ~miss, workers=workers)
    f["h_cnt_s1"] = c1.astype(np.float32)
    f["h_cnt_s23"] = c2.astype(np.float32)

    # --- address -----------------------------------------------------------
    for col, name in (("postcode", "a_postcode_eq"), ("city", "a_city_eq"), ("state", "a_state_eq")):
        x, y = e[col + "_code"][i], e[col + "_code"][j]
        f[name] = _nan_where((x == y).astype(np.float32), (x < 0) | (y < 0))
    ca, cb = g("city")
    f["a_city_fuzzy"] = _cp(ca, cb, fuzz.ratio, (ca != "") & (cb != ""), workers=workers, key=key("city"))
    sa, sb = g("street")
    f["a_street_set"] = _cp(sa, sb, fuzz.token_set_ratio, (sa != "") & (sb != ""), workers=workers, key=key("street"))
    aa, ab = g("addr_clean")
    f["a_token_set"] = _cp(aa, ab, fuzz.token_set_ratio, (aa != "") & (ab != ""), workers=workers, key=key("addr_clean"))
    sw, _, _ = _overlap(e, "aw", i, j)
    union = e["aw_sum"][i] + e["aw_sum"][j] - sw
    empty = (e["aw_cnt"][i] == 0) | (e["aw_cnt"][j] == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["a_idf_jaccard"] = _nan_where(np.where(union > 0, sw / union, 0.0), empty)
    f["a_has_s1"] = e["has_addr"][i].astype(np.float32)
    f["a_has_s23"] = e["has_addr"][j].astype(np.float32)
    sib = e["sib_best"][i2]
    own = e["sib_is_own"][i2]  # sibling best address == this record's address -> same as a_token_set
    sbs = _cp(aa, sib, fuzz.token_set_ratio, (aa != "") & (sib != "") & ~own, workers=workers)
    f["a_sib_best_set"] = np.where(own & (sib != ""), f["a_token_set"], sbs)

    f.update(ctx)
    return f


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_features(cands: pd.DataFrame, s1n: pd.DataFrame, s23n: pd.DataFrame, sib: pd.DataFrame,
                     idf: Mapping[str, float], n_jobs: int = 16, tfidf: CharTfidf | None = None,
                     addr_idf: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Pair features for ``cands`` -> ``s1_id, s23_id`` + FEATURE_COLUMNS (float32).

    ``cands`` must already carry the ``c_*`` columns from
    ``add_context_features`` run on the WHOLE per-country table (ValueError
    otherwise); they are carried through unchanged. ``idf`` weights the
    name words (unseen words get the largest IDF). ``tfidf`` is the fitted
    char 3-gram model (``build_tfidf``) and ``addr_idf`` the address-token IDF
    (``build_idf`` on addresses); when omitted they are fitted on the names /
    addresses of the full ``s1n`` and ``s23n`` frames (deterministic, but slow
    on big frames -- pass them in production)."""
    missing = [c for c in CONTEXT_FEATURES if c not in cands.columns]
    if missing:
        raise ValueError(
            f"candidates lack context columns {missing}: call add_context_features on the WHOLE "
            "per-country candidate table first (ranks and claimant counts are wrong on a chunk)")
    n = len(cands)
    if n == 0:
        out = pd.DataFrame({"s1_id": cands["s1_id"].to_numpy(), "s23_id": cands["s23_id"].to_numpy()})
        return pd.concat([out, pd.DataFrame({c: np.zeros(0, np.float32) for c in FEATURE_COLUMNS})], axis=1)
    if tfidf is None:
        tfidf = build_tfidf(pd.concat([s1n["name_sorted"], s23n["name_sorted"]], ignore_index=True))
    if addr_idf is None:
        addr_idf = build_idf(pd.concat([s1n["addr_clean"], s23n["addr_clean"]], ignore_index=True))

    c1, u1 = pd.factorize(cands["s1_id"].astype("string"))
    c2, u2 = pd.factorize(cands["s23_id"].astype("string"))
    u1 = np.asarray(u1, dtype=object)
    u2 = np.asarray(u2, dtype=object)
    sub1 = _take(s1n, _rows_for(s1n, u1))
    sub2 = _take(s23n, _rows_for(s23n, u2))
    e = _encode(sub1, sub2, sib, idf, addr_idf, tfidf)

    ctx_all = {c: cands[c].to_numpy(dtype=np.float32) for c in CONTEXT_FEATURES}
    c1 = c1.astype(np.int64)
    c2 = c2.astype(np.int64)
    step = min(CHUNK, max(20_000, -(-n // (2 * max(1, n_jobs)))))  # <=200k rows, enough chunks for all cores
    bounds = [(s, min(s + step, n)) for s in range(0, n, step)]
    work = [delayed(_chunk)(e, c1[s:t], c2[s:t], {k: v[s:t] for k, v in ctx_all.items()}) for s, t in bounds]
    parts = Parallel(n_jobs=max(1, n_jobs), backend="threading")(work) if len(work) > 1 else [w[0](*w[1], **w[2]) for w in work]

    out = pd.DataFrame({"s1_id": cands["s1_id"].to_numpy(), "s23_id": cands["s23_id"].to_numpy()})
    cols = {c: (np.concatenate([p[c] for p in parts]) if parts else np.zeros(0)).astype(np.float32, copy=False)
            for c in FEATURE_COLUMNS}
    return pd.concat([out, pd.DataFrame(cols)], axis=1)
