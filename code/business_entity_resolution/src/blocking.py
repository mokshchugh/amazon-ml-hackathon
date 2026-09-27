"""Blocking: build the candidate shortlist (SPEC section 6, step 6).

For every country separately, several searches propose (Source 1, Source
2/3) pairs, which are merged, deduplicated, scored and capped:

* A. Name TF-IDF: character 3-grams of ``name_sorted`` (sublinear TF,
  min_df 2), top ``TOPN_NAME`` Source 2 and top ``TOPN_NAME`` Source 3
  records per Source 1 record with cosine >= ``MIN_COS``, plus the reverse
  direction (top ``REVERSE_TOPN`` Source 1 records of the full pool per
  Source 2/3 record). Also in A: identical name form (``name_sorted``,
  ``name_key`` or the alias of a "x dba / fka / aka / trading as y" name),
  alone and with the city.
* B. Address keys: exact ``postcode or city + first house number + first
  street word``, the looser ``city + street``, and any house number +
  street, and any house number + city (no name needed: domains, dba names,
  unconverted scripts).
* C. Rare word + city (the name word with the highest IDF), and any name
  word + any house number / the street / the city.

Key groups holding more than ``KEY_MAX`` records (S1 + S2 + S3) are
skipped. Search D (embeddings) is off in v1 (``EMBED_ENABLED``).

``best_score`` is a ranking score, NOT a cosine: it ranges over 0 to about
2.6. It is the name cosine -- replaced by 1.5 when the two records share an
identical name form (``name_sorted``, ``name_key`` or alias, compared on the
pair itself) -- plus bonuses for an agreeing house number (any of the first
three, 0.4), street (0.3), city (0.2) and postcode (0.2). It is computed for
every pair the same way whichever search found it, so address agreement
decides between the hundreds of same-name records of a chain. An alias
("x fka y" -> "y") is not taken from inside a run of single-letter initials,
and a one-word alias must not be one of the country's top 1% name words. At most
``CAP_PER_SOURCE`` candidates per (Source 1 record, source) are kept,
ranked by ``best_score`` (ties: smaller Source 2/3 id first).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as normalize_rows
from sparse_dot_topn import sp_matmul_topn

from normalize import _build_name_key

TOPN_NAME = 20
MIN_COS = 0.3
REVERSE_TOPN = 3
KEY_MAX = 200
CAP_PER_SOURCE = 60  # plan value 40; 60 recovers +0.6 pt proxy recall (Task 9 takeover)
EMBED_ENABLED = False

# search-A speed-ups (see "Search A" below)
A_QUERY_K = 5
A_REV_QUERY_K = 8
A_STAGE1_TOPN = 3 * TOPN_NAME
A_STAGE1_REV = 20

SEARCH_A, SEARCH_B, SEARCH_C = 1, 2, 4

log = logging.getLogger(__name__)

# key kinds: 0 exact address, 1 loose address (city + street), 2 rarest
# word + city, 3 identical name form (name_sorted / name_key / alias),
# 4 name word + house number, 5 name word + street, 6 name word + city,
# 7 house number + street (no city), 8 identical name form + city,
# 9 house number + city
_KIND_BIT = np.array([SEARCH_B, SEARCH_B, SEARCH_C, SEARCH_A, SEARCH_C, SEARCH_C, SEARCH_C, SEARCH_B,
                      SEARCH_A, SEARCH_B], dtype=np.int8)

# ranking score = name cosine + agreement bonuses (see ``_score``)
_W_IDENT, _W_HOUSE, _W_STREET, _W_CITY, _W_POSTCODE = 0.5, 0.4, 0.3, 0.2, 0.2

# alias phrases left inside name_clean by normalization ("X fka Y", ...),
# longest first; "d/b/a" etc. reach name_clean as "d b a"
_ALIAS_PHRASES = [("doing", "business", "as"), ("formerly", "known", "as"), ("d", "b", "a"),
                  ("f", "k", "a"), ("a", "k", "a"), ("trading", "as"), ("t", "a"),
                  ("dba",), ("fka",), ("aka",), ("formerly",)]
_ALIAS_HINT = r"\s(?:d b a|dba|doing business as|formerly|f k a|fka|a k a|aka|trading as|t a)\s"
_FREQUENT_SHARE = 0.01  # a one-word alias must not be one of the top 1% name words

OUT_COLUMNS = ["s1_id", "s23_id", "source", "search_mask", "best_score"]
_NEEDED = ["entity_id", "source", "country", "name_sorted", "name_clean", "name_key", "alt_name",
           "postcode", "city", "street", "house_nums"]

_N_THREADS = -1        # sparse_dot_topn: all cores but one
_COL_CHUNK = 250_000   # searched names per matmul (keeps the accumulator in cache)
_ROW_CHUNK = 200_000   # query names per matmul block (bounds memory)
_DOT_CHUNK = 2_000_000
_EMPTY_I = np.array([], dtype=np.int64)
_EMPTY_F = np.array([], dtype=np.float32)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "s1_id": pd.array([], dtype="string[pyarrow]"),
        "s23_id": pd.array([], dtype="string[pyarrow]"),
        "source": pd.array([], dtype="string[pyarrow]"),
        "search_mask": np.array([], dtype=np.int8),
        "best_score": np.array([], dtype=np.float32),
    })


def _text(s: pd.Series) -> pd.Series:
    """Arrow-backed strings, NA -> ""."""
    return s.astype("string[pyarrow]").fillna("").reset_index(drop=True)


def _codes(s: pd.Series) -> np.ndarray:
    """Dense int codes of a text column; "" -> -1."""
    t = _text(s)
    return pd.factorize(t.mask(t == ""))[0].astype(np.int64)


def _combine(*cols: np.ndarray) -> np.ndarray:
    """Dense code of the tuple of codes; -1 when any part is -1."""
    valid = np.logical_and.reduce([c >= 0 for c in cols])
    out = np.full(len(cols[0]), -1, dtype=np.int64)
    if valid.any():
        k = cols[0][valid]
        for c in cols[1:]:
            k = pd.factorize(k * (int(c.max()) + 1) + c[valid])[0].astype(np.int64)
        out[valid] = k
    return out


# ---------------------------------------------------------------------------
# Search A: name TF-IDF
# ---------------------------------------------------------------------------
#
# The TF-IDF weights are exactly those of
#   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
#                   min_df=2)
# fitted on all S1+S2+S3 ``name_sorted`` values of the country, but they are
# computed once per distinct name (document frequencies weighted by how many
# records carry each name). A plain ``sp_matmul_topn`` over all records takes
# hours on the full data (common trigrams such as "and" sit in ~250k names),
# so the top-n search runs in two stages:
#   1. shortlist: ``sp_matmul_topn`` over distinct names, where each query
#      name keeps only its ``A_QUERY_K`` rarest trigrams (``A_REV_QUERY_K``
#      for the reverse direction) and the searched names keep all of theirs,
#      keeping the best ``A_STAGE1_TOPN`` (``A_STAGE1_REV``) names. On a
#      300k-pair sample of true pairs with cos >= 0.3, a query's 5 rarest
#      trigrams hit the match in ~99.5% of cases, at a fraction of the cost;
#   2. exact cosine with the full vectors; pairs with cos >= ``MIN_COS`` are
#      expanded to records, keeping the top ``TOPN_NAME`` S2 and S3 records
#      per S1 record (``REVERSE_TOPN`` S1 records per S2/S3 record), ties by
#      id.

def _tfidf_unique(names: pd.Series):
    """-> (codes, X, df_share): record -> distinct-name code, the L2-normalised
    TF-IDF matrix of the distinct names, each column's record-level
    document-frequency share. ``None`` when the vocabulary is empty."""
    names = _text(names)
    codes, uniques = pd.factorize(names)
    counts = np.bincount(codes, minlength=len(uniques)).astype(np.float64)
    cv = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3), dtype=np.float32)
    try:
        xc = cv.fit_transform(list(uniques)).tocsr()
    except ValueError:  # empty vocabulary
        return None
    xb = xc.copy(); xb.data[:] = 1
    df = np.asarray(xb.T @ counts).ravel()
    del xb
    keep = np.flatnonzero(df >= 2)
    if len(keep) == 0:
        return None
    xc = xc[:, keep].tocsr(); df = df[keep]
    n = float(len(names))
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    xc.data = (1.0 + np.log(xc.data)).astype(np.float32)
    x = normalize_rows(xc @ sp.diags(idf), norm="l2", copy=False).astype(np.float32).tocsr()
    return codes.astype(np.int64), x, df / n


def _groups(u: np.ndarray, n_groups: int, rows: np.ndarray | None = None):
    """Members (positions) of each code in ``u``, in ascending position order;
    only positions where ``rows`` is True when given."""
    if rows is None:
        order = np.argsort(u, kind="stable")
    else:
        pos = np.flatnonzero(rows)
        order = pos[np.argsort(u[pos], kind="stable")]
    count = np.bincount(u[order], minlength=n_groups)
    start = np.cumsum(count) - count
    return order, start, count


def _expand(keys: np.ndarray, groups, limit: int | None = None):
    """For each key, its group members (first ``limit`` only):
    -> (index into keys, member)."""
    order, start, count = groups
    n = count[keys]
    if limit is not None:
        n = np.minimum(n, limit)
    rep = np.repeat(np.arange(len(keys)), n)
    within = np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)
    return rep, order[start[keys][rep] + within]


def _top_per(owner: np.ndarray, member: np.ndarray, score: np.ndarray, k: int):
    """Keep the top ``k`` (score desc, member asc) rows per owner."""
    if len(owner) == 0:
        return owner, member, score
    o = np.lexsort((member, -score, owner))
    owner, member, score = owner[o], member[o], score[o]
    new = np.r_[True, owner[1:] != owner[:-1]]
    first = np.maximum.accumulate(np.where(new, np.arange(len(owner)), 0))
    keep = (np.arange(len(owner)) - first) < k
    return owner[keep], member[keep], score[keep]


def _rarest(x, df_share: np.ndarray, k: int):
    """Each row of ``x`` reduced to its ``k`` rarest columns (ties: lower column)."""
    x = x.tocsr()
    rows = np.repeat(np.arange(x.shape[0], dtype=np.int64), np.diff(x.indptr))
    col_rank = np.empty(x.shape[1], dtype=np.int64)
    col_rank[np.lexsort((np.arange(x.shape[1]), df_share))] = np.arange(x.shape[1])
    o = np.argsort(rows * x.shape[1] + col_rank[x.indices])  # unique keys
    rank = np.empty(len(o), dtype=np.int64)
    rank[o] = np.arange(len(o)) - x.indptr[rows[o]]
    y = x.copy()
    y.data = np.where(rank < k, y.data, 0).astype(y.data.dtype)
    y.eliminate_zeros()
    return y


def _shortlist(q, x, left: np.ndarray, right: np.ndarray, top_n: int):
    """Stage 1 over distinct names (queries ``q[left]`` against ``x[right]``)
    -> (left name, right name) pairs: per left name, the best ``top_n``
    when the searched names fit one chunk; otherwise the best
    ``max(2 * top_n / n_chunks, top_n / 4)`` of every chunk (no merge sort -- the
    exact stage re-ranks them anyway)."""
    if len(left) == 0 or len(right) == 0:
        return _EMPTY_I, _EMPTY_I
    col_parts = [right[s:s + _COL_CHUNK] for s in range(0, len(right), _COL_CHUNK)]
    col_mats = [x[part].T.tocsr() for part in col_parts]
    if len(col_parts) > 1:
        top_n = min(top_n, max(-(-2 * top_n // len(col_parts)), -(-top_n // 4)))
    out_a, out_b = [], []
    for r0 in range(0, len(left), _ROW_CHUNK):
        lq = q[left[r0:r0 + _ROW_CHUNK]]
        rr, cc = [], []
        for part, mat in zip(col_parts, col_mats):
            c = sp_matmul_topn(lq, mat, top_n=top_n, threshold=0.0, n_threads=_N_THREADS)
            rr.append(np.repeat(np.arange(c.shape[0], dtype=np.int64), np.diff(c.indptr)))
            cc.append(part[c.indices.astype(np.int64)])
        r, cidx = np.concatenate(rr), np.concatenate(cc)
        out_a.append(left[r0:r0 + _ROW_CHUNK][r]); out_b.append(cidx)
    return np.concatenate(out_a), np.concatenate(out_b)


def _search_a(tf, n1: int, src2: np.ndarray, n_src: int, q1: np.ndarray, q2: np.ndarray):
    """Forward searches for the S1 records flagged in ``q1``, reverse searches
    for the S2/S3 records flagged in ``q2`` (all records outside a proxy run)."""
    codes, x, df_share = tf
    n_names = x.shape[0]
    u1, u2 = codes[:n1], codes[n1:]
    g1q = _groups(u1, n_names, q1)
    names1q = np.unique(u1[q1])
    ii, jj, ss = [], [], []
    t0 = time.time()
    q = _rarest(x, df_share, A_QUERY_K)
    for s in range(n_src):  # forward: top TOPN_NAME records of source s per S1 record
        js = np.flatnonzero(src2 == s)
        gs = _groups(u2[js], n_names)
        a, b = _shortlist(q, x, names1q, np.unique(u2[js]), A_STAGE1_TOPN)
        cos = _row_dot(x, x, a, b)
        ok = cos >= MIN_COS - 1e-6
        a, b, cos = a[ok], b[ok], cos[ok]
        rep, jpos = _expand(b, gs, TOPN_NAME)
        ua, j, sc = _top_per(a[rep], js[jpos], cos[rep], TOPN_NAME)
        rep, i = _expand(ua, g1q)
        ii.append(i); jj.append(j[rep]); ss.append(sc[rep])
    log.info("    A forward: %d pairs (%.0fs)", sum(map(len, ii)), time.time() - t0)
    t0 = time.time()
    # reverse: top REVERSE_TOPN S1 records (of the full pool) per S2/S3 record
    del q
    g1 = _groups(u1, n_names)
    g2 = _groups(u2, n_names, q2)
    q = _rarest(x, df_share, A_REV_QUERY_K)
    a, b = _shortlist(q, x, np.unique(u2[q2]), np.unique(u1), A_STAGE1_REV)
    del q
    cos = _row_dot(x, x, a, b)
    ok = cos >= MIN_COS - 1e-6
    a, b, cos = a[ok], b[ok], cos[ok]
    rep, i = _expand(b, g1, REVERSE_TOPN)
    ub, i, sc = _top_per(a[rep], i, cos[rep], REVERSE_TOPN)
    rep, j = _expand(ub, g2)
    i, j, sc = i[rep], j, sc[rep]
    keep = q1[i]
    ii.append(i[keep]); jj.append(j[keep]); ss.append(sc[keep])
    log.info("    A reverse: %d pairs (%.0fs)", len(ii[-1]), time.time() - t0)
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(ss).astype(np.float32)


def _row_dot(x1, x2, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    out = np.zeros(len(i), dtype=np.float32)
    for s in range(0, len(i), _DOT_CHUNK):
        e = s + _DOT_CHUNK
        prod = x1[i[s:e]].multiply(x2[j[s:e]])
        out[s:e] = np.asarray(prod.sum(axis=1)).ravel()
    return out


# ---------------------------------------------------------------------------
# Searches B and C: key joins
# ---------------------------------------------------------------------------

def _word_table(names: pd.Series):
    """Words of each record's name.

    -> (rec, word): every distinct (record, word) pair; ``rare``: per
    record, the code of its name word with the highest IDF (lowest
    record-level document frequency over all records given; ties ->
    alphabetically first word), -1 when the name has no word; and the word
    of each code."""
    names = _text(names)
    rec_u, uniq = pd.factorize(names)
    rare = np.full(len(names), -1, dtype=np.int64)
    if len(uniq) == 0:
        return _EMPTY_I, _EMPTY_I, rare, []
    cnt = np.bincount(rec_u, minlength=len(uniq)).astype(np.float64)
    lists = pc.split_pattern(pa.array(np.asarray(uniq, dtype=object), type=pa.large_string()), " ")
    parent = pc.list_parent_indices(lists).to_numpy().astype(np.int64)
    words = pd.Series(pd.arrays.ArrowStringArray(pc.list_flatten(lists)))
    wcode, wuniq = pd.factorize(words.mask(words == ""))
    wcode = wcode.astype(np.int64)
    ok = wcode >= 0
    parent, wcode = parent[ok], wcode[ok]
    if len(wcode) == 0:
        return _EMPTY_I, _EMPTY_I, rare, []
    n_w = len(wuniq)
    pair = np.unique(parent * n_w + wcode)
    parent, wcode = pair // n_w, pair % n_w
    df = np.bincount(wcode, weights=cnt[parent], minlength=n_w)
    wrank = np.empty(n_w, dtype=np.int64)
    wrank[np.argsort(np.asarray(wuniq, dtype=object), kind="stable")] = np.arange(n_w)
    o = np.lexsort((wrank[wcode], df[wcode], parent))
    first = np.r_[True, parent[o][1:] != parent[o][:-1]]
    best = np.full(len(uniq), -1, dtype=np.int64)
    best[parent[o][first]] = wcode[o][first]
    rep, recs = _expand(parent, _groups(rec_u.astype(np.int64), len(uniq)))
    return recs, wcode[rep], best[rec_u], list(wuniq)


def _rare_word_codes(names: pd.Series) -> np.ndarray:
    """Per record, the code of its rarest name word (see ``_word_table``)."""
    return _word_table(names)[2]


def _house_table(col: pd.Series):
    """-> (rec, house): every distinct (record, house number) pair, and the
    code of each record's first house number (-1 if none)."""
    recs, vals, firsts = [], [], []
    for r, h in enumerate(col.tolist()):
        f = ""
        if h is not None and not isinstance(h, float) and len(h):
            f = h[0]
            for v in dict.fromkeys(h):
                if v:
                    recs.append(r); vals.append(v)
        firsts.append(f)
    codes = _codes(pd.Series(vals + firsts, dtype="string[pyarrow]"))
    return np.asarray(recs, dtype=np.int64), codes[: len(vals)], codes[len(vals):]


def _first_k(rec: np.ndarray, code: np.ndarray, n: int, k: int = 3) -> np.ndarray:
    """(n, k) array: each record's first ``k`` codes in table order, -1 padded."""
    out = np.full((n, k), -1, dtype=np.int64)
    if len(rec) == 0:
        return out
    new = np.r_[True, rec[1:] != rec[:-1]]
    first = np.maximum.accumulate(np.where(new, np.arange(len(rec)), 0))
    pos = np.arange(len(rec)) - first
    ok = pos < k
    out[rec[ok], pos[ok]] = code[ok]
    return out


def _alias_ok(words: list[str], frequent: set[str]) -> bool:
    """An alias needs 2+ words, or a word outside the frequent name words."""
    return len(words) >= 2 or any(w not in frequent for w in words)


def _alias_split(name: str, frequent: set[str]) -> str:
    """The part after the first valid alias phrase of ``name`` ("x fka y" ->
    "y"), "" if none. A phrase next to a single-letter token is part of a run
    of initials ("m a k a enterprises"), not an alias phrase."""
    toks = name.split()
    for p in range(1, len(toks) - 1):
        for ph in _ALIAS_PHRASES:
            e = p + len(ph)
            if e >= len(toks) or tuple(toks[p:e]) != ph:
                continue
            if len(toks[p - 1]) == 1 or len(toks[e]) == 1:
                continue
            if _alias_ok(toks[e:], frequent):
                return " ".join(toks[e:])
    return ""


def _alias(name_clean: pd.Series, alt_name: pd.Series, frequent: set[str]) -> pd.Series:
    """The alternative name of each record: ``alt_name`` (dba), else the part
    after an alias phrase left in ``name_clean`` (``_alias_split``); "" if
    none. A one-word alias made of a word in ``frequent`` is dropped."""
    nc = _text(name_clean)
    alt = _text(alt_name).to_numpy(dtype=object)
    out = np.full(len(nc), "", dtype=object)
    for r in np.flatnonzero(alt != ""):
        if _alias_ok(alt[r].split(), frequent):
            out[r] = alt[r]
    has = nc.str.contains(_ALIAS_HINT, regex=True).to_numpy(dtype=bool) & (out == "")
    vals = nc.to_numpy(dtype=object)
    for r in np.flatnonzero(has):
        out[r] = _alias_split(vals[r], frequent)
    return pd.Series(out, dtype="string[pyarrow]")


def _frequent_words(w_code: np.ndarray, words: list[str]) -> set[str]:
    """The top ``_FREQUENT_SHARE`` of distinct name words by record count
    (at least one; ties -> smaller code)."""
    if len(words) == 0:
        return set()
    df = np.bincount(w_code, minlength=len(words))
    k = max(1, int(np.ceil(_FREQUENT_SHARE * len(words))))
    return {words[c] for c in np.argsort(-df, kind="stable")[:k]}


def _sorted_words(s: pd.Series) -> pd.Series:
    lists = pc.split_pattern(pa.array(_text(s).array), " ")
    return pd.Series([" ".join(sorted(w for w in v if w)) for v in lists.to_pylist()],
                     dtype="string[pyarrow]")


def _name_forms(both: pd.DataFrame, frequent: set[str]):
    """Identity forms of each record's name: name_sorted, name_key, and the
    sorted words / key of its alias (``_alias``).

    -> (rec, form code) for every distinct pair, and the (n, 4) array of each
    record's form codes (-1 = none) used for the identity test in ``_score``."""
    n = len(both)
    alias = _alias(both["name_clean"], both["alt_name"], frequent)
    has = np.flatnonzero((alias != "").to_numpy())
    al = alias.iloc[has].reset_index(drop=True)
    al_sorted = _sorted_words(al)
    al_key = pd.Series([_build_name_key(v.split()) for v in al.tolist()], dtype="string[pyarrow]")
    forms = pd.concat([
        "s " + _text(both["name_sorted"]), "k " + _text(both["name_key"]),
        "s " + al_sorted, "k " + al_key,
    ], ignore_index=True)
    forms = forms.mask(forms.isin(["s ", "k "]))
    rec = np.concatenate([np.arange(n), np.arange(n), has, has]).astype(np.int64)
    code = pd.factorize(forms)[0].astype(np.int64)
    h = len(has)
    forms4 = np.full((n, 4), -1, dtype=np.int64)
    forms4[:, 0], forms4[:, 1] = code[:n], code[n:2 * n]
    forms4[has, 2], forms4[has, 3] = code[2 * n:2 * n + h], code[2 * n + h:]
    ok = code >= 0
    rec, code = rec[ok], code[ok]
    m = int(code.max()) + 1 if len(code) else 1
    pair = np.unique(rec * m + code)
    return pair // m, pair % m, forms4


def _key_pairs(rec: np.ndarray, key: np.ndarray, n1: int, q1: np.ndarray):
    """S1 x S2/S3 record pairs sharing a key. ``rec`` are record positions
    (S1 first, then S2/S3); each (rec, key) pair is distinct; -1 = no key.
    Keys held by more than ``KEY_MAX`` records are skipped. Only S1 records
    flagged in ``q1`` are paired."""
    valid = key >= 0
    rec, key = rec[valid], key[valid]
    if len(key) == 0:
        return _EMPTY_I, _EMPTY_I
    ok = np.bincount(key)[key] <= KEY_MAX
    rec, key = rec[ok], key[ok]
    s1 = rec < n1
    left = s1.copy()
    left[s1] = q1[rec[s1]]
    left = pd.DataFrame({"k": key[left], "i": rec[left]})
    right = pd.DataFrame({"k": key[~s1], "j": rec[~s1] - n1})
    m = left.merge(right, on="k", how="inner", sort=False)
    return m["i"].to_numpy(np.int64), m["j"].to_numpy(np.int64)


def _cross(rec_a, code_a, rec_b, code_b):
    """Per record, every (a, b) combination -> (rec, combined code)."""
    m = pd.DataFrame({"r": rec_a, "a": code_a}).merge(pd.DataFrame({"r": rec_b, "b": code_b}), on="r")
    r = m["r"].to_numpy(np.int64)
    return r, _combine(m["a"].to_numpy(np.int64), m["b"].to_numpy(np.int64))


def _search_keys(a: pd.DataFrame, b: pd.DataFrame, q1: np.ndarray):
    """Searches B and C, plus the identical-name-form key of search A.

    -> (i, j, kind) pairs (kinds as in ``_KIND_BIT``) and the per-record
    address codes used by ``_score`` (records: S1 first, then S2/S3)."""
    n1 = len(a)
    cols = ["postcode", "city", "street", "house_nums", "name_clean", "name_sorted", "name_key", "alt_name"]
    both = pd.concat([a[cols], b[cols]], ignore_index=True)
    n = len(both)
    street_t = _text(both["street"])
    postcode, city, street = _codes(both["postcode"]), _codes(both["city"]), _codes(street_t)
    first_word = _codes(street_t.str.replace(r"\s.*$", "", regex=True))
    h_rec, h_code, house = _house_table(both["house_nums"])
    w_rec, w_code, rare, words = _word_table(both["name_clean"])
    f_rec, f_code, forms4 = _name_forms(both, _frequent_words(w_code, words))
    del both, street_t
    feats = {"postcode": postcode, "city": city, "street": street, "house3": _first_k(h_rec, h_code, n),
             "forms": forms4}
    recs = np.arange(n, dtype=np.int64)
    keys = [
        lambda: (recs, _combine(postcode, house, first_word), 0),   # B exact: postcode + house + street word
        lambda: (recs, _combine(city, house, first_word), 0),       # B exact: city + house + street word
        lambda: (recs, _combine(city, street), 1),                  # B loose: city + street
        lambda: (recs, _combine(rare, city), 2),                    # C: rarest word + city
        lambda: (f_rec, f_code, 3),                                 # A: identical name form
        lambda: (*_cross(w_rec, w_code, h_rec, h_code), 4),         # C: any name word + any house number
        lambda: (w_rec, _combine(w_code, street[w_rec]), 5),        # C: any name word + street
        lambda: (w_rec, _combine(w_code, city[w_rec]), 6),          # C: any name word + city
        lambda: (*_cross(h_rec, h_code, recs, street), 7),          # B: any house number + street
        lambda: (f_rec, _combine(f_code, city[f_rec]), 8),          # A: identical name form + city
        lambda: (*_cross(h_rec, h_code, recs, city), 9),            # B: any house number + city
    ]
    ii, jj, kk = [], [], []
    for make in keys:
        rec, key, kind = make()
        i, j = _key_pairs(rec, key, n1, q1)
        del rec, key
        log.info("    key kind %d: %d pairs", kind, len(i))
        ii.append(i.astype(np.int32)); jj.append(j.astype(np.int32)); kk.append(np.full(len(i), kind, dtype=np.int8))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(kk), feats


def _any_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise: does any valid (>= 0) code of ``a`` equal any code of ``b``."""
    hit = np.zeros(len(a), dtype=bool)
    for x in range(a.shape[1]):
        for y in range(b.shape[1]):
            hit |= (a[:, x] == b[:, y]) & (a[:, x] >= 0)
    return hit


def _score(i, j, cos, n1, feats):
    """Ranking score: name cosine (1 + ``_W_IDENT`` when a name form is
    identical) + bonuses for agreeing house number (any of the first three),
    street, city, postcode."""
    jj = j.astype(np.int64) + n1
    # an identical name form (e.g. the alias) counts as a full name match,
    # tested on the pair itself whichever search found it
    ident = _any_equal(feats["forms"][i], feats["forms"][jj])
    sc = np.where(ident, np.float32(1.0 + _W_IDENT), cos.astype(np.float32))
    for name, w in (("street", _W_STREET), ("city", _W_CITY), ("postcode", _W_POSTCODE)):
        c = feats[name]
        ci, cj = c[i], c[jj]
        sc += w * ((ci == cj) & (ci >= 0))
    h = feats["house3"]
    sc += _W_HOUSE * _any_equal(h[i], h[jj])
    return sc.astype(np.float32)


# ---------------------------------------------------------------------------
# One country
# ---------------------------------------------------------------------------

_SHARD = 250_000  # S1 records per merge shard (bounds merge memory)


def _merge_shard(i, j, mask, acos, n1, n2, tf, feats, src2):
    """One row per (i, j): OR-ed search mask, name cosine, ranking score;
    then the top ``CAP_PER_SOURCE`` per (i, source)."""
    order = np.argsort(i.astype(np.int64) * n2 + j, kind="stable")
    i, j = i[order], j[order]
    starts = np.flatnonzero(np.r_[True, (i[1:] != i[:-1]) | (j[1:] != j[:-1])])
    mask = np.bitwise_or.reduceat(mask[order], starts)
    cos = np.maximum.reduceat(acos[order], starts)
    i, j = i[starts].astype(np.int64), j[starts].astype(np.int64)
    del order, starts
    need = cos < 0
    if tf is not None:
        cos[need] = _row_dot(tf[1], tf[1], tf[0][i[need]], tf[0][n1 + j[need]])
    else:
        cos[need] = 0.0
    score = _score(i, j, cos, n1, feats)
    del cos
    # cap per (i, source): rank by score desc, then j (= s23 id order)
    s = src2[j]
    order = np.lexsort((j, -score, s, i))
    i, j, s, score, mask = i[order], j[order], s[order], score[order], mask[order]
    new_group = np.r_[True, (i[1:] != i[:-1]) | (s[1:] != s[:-1])]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(i)), 0))
    keep = (np.arange(len(i)) - group_start) < CAP_PER_SOURCE
    return i[keep], j[keep], s[keep], score[keep], mask[keep]


def _country(a: pd.DataFrame, b: pd.DataFrame, q1: np.ndarray | None = None,
             q2: np.ndarray | None = None) -> pd.DataFrame | None:
    """Candidates of one country. ``q1`` / ``q2`` (proxy runs only) flag the
    S1 records to return candidates for and the S2/S3 records whose reverse
    name search runs; the searched pools are always complete."""
    n1, n2 = len(a), len(b)
    q1 = np.ones(n1, dtype=bool) if q1 is None else q1
    q2 = np.ones(n2, dtype=bool) if q2 is None else q2
    src_text = _text(b["source"])
    src_names = sorted(set(src_text.unique()))
    src2 = pd.Categorical(src_text, categories=src_names).codes.astype(np.int8)

    t0 = time.time()
    tf = _tfidf_unique(pd.concat([_text(a["name_sorted"]), _text(b["name_sorted"])], ignore_index=True))
    if tf is not None:
        ai, aj, ascore = _search_a(tf, n1, src2, len(src_names), q1, q2)
    else:
        ai, aj, ascore = _EMPTY_I, _EMPTY_I, _EMPTY_F
    log.info("  A: %d pairs (%d distinct names, %.0fs)", len(ai),
             0 if tf is None else tf[1].shape[0], time.time() - t0)

    t0 = time.time()
    ki, kj, kind, feats = _search_keys(a, b, q1)
    log.info("  B+C: %d pairs (%.0fs)", len(ki), time.time() - t0)

    t0 = time.time()
    i = np.concatenate([ai.astype(np.int32), ki]); j = np.concatenate([aj.astype(np.int32), kj])
    if len(i) == 0:
        return None
    mask = np.concatenate([np.full(len(ai), SEARCH_A, np.int8), _KIND_BIT[kind]])
    acos = np.concatenate([ascore, np.full(len(ki), -1.0, np.float32)])
    del ai, aj, ki, kj, kind, ascore
    out = []
    for lo in range(0, n1, _SHARD):
        sel = np.flatnonzero((i >= lo) & (i < lo + _SHARD))
        if len(sel):
            out.append(_merge_shard(i[sel], j[sel], mask[sel], acos[sel],
                                    n1, n2, tf, feats, src2))
    del i, j, mask, acos, tf
    i, j, s, score, mask = (np.concatenate(p) for p in zip(*out))
    log.info("  merge+cap: %d pairs kept (%.0fs)", len(i), time.time() - t0)

    ids1 = pa.array(_text(a["entity_id"]).array)
    ids2 = pa.array(_text(b["entity_id"]).array)
    src_arr = pa.array(src_names, type=pa.large_string())
    return pd.DataFrame({
        "s1_id": pd.arrays.ArrowStringArray(ids1.take(pa.array(i))),
        "s23_id": pd.arrays.ArrowStringArray(ids2.take(pa.array(j))),
        "source": pd.arrays.ArrowStringArray(src_arr.take(pa.array(s.astype(np.int64)))),
        "search_mask": mask.astype(np.int8),
        "best_score": score,
    })


def generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """Candidate (S1, S2/S3) pairs from searches A-C, run per country.

    Both frames are normalized (``normalize_frame``) and carry
    ``entity_id, source, country``. Inputs are not modified. The output is
    independent of input row order: sorted by country, ``s1_id``, source,
    ``best_score`` (descending) and ``s23_id``.
    """
    return _generate(s1, s23)


def _generate(s1: pd.DataFrame, s23: pd.DataFrame, s1_query=None, s23_query=None) -> pd.DataFrame:
    """``generate_candidates``; a proxy run passes id sets ``s1_query`` (S1
    records to return candidates for) and ``s23_query`` (S2/S3 records whose
    reverse name search runs) -- see ``_country``."""
    parts = []
    c1 = _text(s1["country"]).to_numpy(dtype=object)
    c23 = _text(s23["country"]).to_numpy(dtype=object)
    for country in sorted((set(c1) & set(c23)) - {""}):
        frames = []
        for df, c in ((s1, c1), (s23, c23)):
            rows = np.flatnonzero(c == country)
            ids = _text(df["entity_id"].iloc[rows]).to_numpy(dtype=object)
            rows = rows[np.argsort(ids, kind="stable")]
            frames.append(df[_NEEDED].iloc[rows].reset_index(drop=True))
        a, b = frames
        q1 = None if s1_query is None else _text(a["entity_id"]).isin(s1_query).to_numpy()
        q2 = None if s23_query is None else _text(b["entity_id"]).isin(s23_query).to_numpy()
        log.info("country %s: %d S1 x %d S2/S3", country, len(a), len(b))
        out = _country(a, b, q1, q2)
        del a, b, frames
        if out is not None and len(out):
            parts.append(out)
    if not parts:
        return _empty()
    return pd.concat(parts, ignore_index=True)


def to_lists(cands: pd.DataFrame) -> dict[str, list[str]]:
    """s1_id -> its candidate s23_ids, in the frame's row order."""
    out: dict[str, list[str]] = {}
    for a, b in zip(cands["s1_id"].tolist(), cands["s23_id"].tolist()):
        out.setdefault(a, []).append(b)
    return out
