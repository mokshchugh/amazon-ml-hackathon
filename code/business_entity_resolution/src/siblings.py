"""Sibling groups within Source 2 and Source 3 (SPEC section 6, step 7).

Within a (country, source) pair, two records are linked ("siblings") when:

* they share a non-empty ``addr_clean`` and their ``name_clean`` values have
  a RapidFuzz ``token_set_ratio`` of at least ``FUZZ_THRESHOLD``; or
* they share both a non-empty ``name_key`` and a non-empty ``city``; or
* they share a non-empty ``name_key``, at least one of the two has an empty
  ``addr_clean`` **and** an empty ``city`` (a name-only record, e.g.
  "PORTER & NALL [LLC]"), and that ``name_key`` -- within this (country,
  source) pair -- is attached to exactly one distinct non-empty
  ``addr_clean`` value.

The third rule is what lets the SPEC's own example hold: the address-less
"PORTER & NALL [LLC]" record joins "porternall.com"'s group because the
name_key "porternall" is attached to only one address anywhere in this
(country, source); a name_key seen at several different addresses (a common
chain name) is left alone, since there is then no way to tell which address
the name-only record belongs to.

Connected components (``scipy.sparse.csgraph.connected_components``) over
these edges become sibling groups; a record with no links is its own group
of size 1 (``sib_group_size == 1``). Never links across ``source`` or
``country``.

Scale guard: within a (country, source, addr_clean) group, records with an
identical ``name_clean`` are linked directly (fully vectorized, no
RapidFuzz call needed -- an exact match always satisfies the fuzzy
threshold). When a group holds more than one distinct ``name_clean`` value,
up to ``MAX_REPS`` representative names (the group's distinct names, sorted
for determinism) are compared pairwise with RapidFuzz -- every distinct
name against those representatives, not an all-pairs scan of the group's
members. Groups of identical ``addr_clean`` larger than ``KEY_MAX`` skip the
fuzzy link entirely and are reported via a log line; such a group's members
can still link to each other through the name_key rules.

Determinism: the input is sorted by ``entity_id`` before anything else, and
``sib_group_id`` is the dense rank (0-based) of a group's smallest
``entity_id`` among all groups' smallest ``entity_id``s. ``sib_best_addr``
is the group's longest non-empty ``addr_clean`` value; ties broken
lexicographically (smallest wins).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from scipy.sparse.csgraph import connected_components

from blocking import _codes, _combine, _expand, _groups, _text

log = logging.getLogger(__name__)

FUZZ_THRESHOLD = 80
KEY_MAX = 200   # addr_clean groups bigger than this skip the fuzzy link
MAX_REPS = 50   # bounded representative names compared per address group

OUT_COLUMNS = ["entity_id", "sib_group_id", "sib_group_size", "sib_best_addr"]

_EMPTY = np.array([], dtype=np.int64)


def _star_from_groups(order: np.ndarray, start: np.ndarray, count: np.ndarray):
    """Edges linking every member of each group (size >= 2) to the group's
    first member -- a star topology, which is enough for connectivity and
    avoids an O(size^2) edge list."""
    keep = np.flatnonzero(count >= 2)
    if len(keep) == 0:
        return _EMPTY, _EMPTY
    rep, member = _expand(keep, (order, start, count))
    firsts = order[start[keep]]
    i, j = firsts[rep], member
    ok = i != j
    return i[ok].astype(np.int64), j[ok].astype(np.int64)


def _star_edges_from_code(code: np.ndarray):
    """Star edges (see ``_star_from_groups``) over groups of equal, valid
    (>= 0) values of ``code``."""
    valid = code >= 0
    if not valid.any():
        return _EMPTY, _EMPTY
    n_groups = int(code[valid].max()) + 1
    order, start, count = _groups(code, n_groups, valid)
    return _star_from_groups(order, start, count)


def _addr_fuzzy_edges(effective_addr: np.ndarray, name_clean: np.ndarray, skipped_sizes: list[int]):
    """Bounded fuzzy-name edges within each valid ``effective_addr`` group
    (see module docstring). Exact-name matches are handled elsewhere
    (``_star_edges_from_code`` on the address+name key), so only distinct
    name buckets are compared here."""
    valid = effective_addr >= 0
    if not valid.any():
        return _EMPTY, _EMPTY
    n_groups = int(effective_addr[valid].max()) + 1
    order, start, count = _groups(effective_addr, n_groups, valid)
    ii, jj = [], []
    for g in np.flatnonzero(count >= 2):
        members = order[start[g]:start[g] + count[g]]
        if len(members) > KEY_MAX:
            skipped_sizes.append(int(len(members)))
            continue
        names = name_clean[members]
        uniq, inv = np.unique(names, return_inverse=True)
        if len(uniq) == 1:
            continue  # identical names: already linked by the exact-key star edges
        bucket_first = np.empty(len(uniq), dtype=np.int64)
        for u in range(len(uniq)):
            bucket_first[u] = members[np.flatnonzero(inv == u)[0]]
        reps = uniq[:MAX_REPS]
        sim = process.cdist(uniq.tolist(), reps.tolist(), scorer=fuzz.token_set_ratio,
                             score_cutoff=FUZZ_THRESHOLD)
        a_idx, b_idx = np.nonzero(sim)
        keep = a_idx != b_idx
        ii.extend(bucket_first[a_idx[keep]].tolist())
        jj.extend(bucket_first[b_idx[keep]].tolist())
    if not ii:
        return _EMPTY, _EMPTY
    return np.array(ii, dtype=np.int64), np.array(jj, dtype=np.int64)


def _orphan_namekey_edges(effective_nk: np.ndarray, addr_code: np.ndarray, city_code: np.ndarray, n: int):
    """Edges for the third link rule: a name-only record (empty addr_clean
    and empty city) joins the single address group of its (country,
    source)-scoped name_key, when that name_key has exactly one distinct
    non-empty address."""
    valid = effective_nk >= 0
    if not valid.any():
        return _EMPTY, _EMPTY
    pos = np.arange(n, dtype=np.int64)
    df = pd.DataFrame({"nk": effective_nk[valid], "addr": addr_code[valid],
                        "city": city_code[valid], "pos": pos[valid]})
    has_addr = df[df["addr"] >= 0]
    if has_addr.empty:
        return _EMPTY, _EMPTY
    counts = has_addr.groupby("nk")["addr"].nunique()
    single = counts.index[counts == 1]
    if len(single) == 0:
        return _EMPTY, _EMPTY
    single_set = set(single.tolist())
    anchor_rows = has_addr[has_addr["nk"].isin(single_set)].sort_values("pos", kind="stable")
    anchors = anchor_rows.groupby("nk", sort=False)["pos"].first()
    orphans = df[(df["addr"] < 0) & (df["city"] < 0) & (df["nk"].isin(single_set))]
    if orphans.empty:
        return _EMPTY, _EMPTY
    merged = orphans.merge(anchors.rename("anchor_pos"), left_on="nk", right_index=True)
    return merged["anchor_pos"].to_numpy(np.int64), merged["pos"].to_numpy(np.int64)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_id": pd.array([], dtype="string[pyarrow]"),
        "sib_group_id": np.array([], dtype=np.int64),
        "sib_group_size": np.array([], dtype=np.int32),
        "sib_best_addr": pd.array([], dtype="string[pyarrow]"),
    })


def sibling_groups(s23: pd.DataFrame) -> pd.DataFrame:
    """Sibling groups of the given Source 2 / Source 3 records (see module
    docstring for the link rules). Works per ``(country, source)``; never
    links across either. Input is not modified; row order does not affect
    the result. Columns: ``entity_id``, ``sib_group_id`` (int64),
    ``sib_group_size`` (int32), ``sib_best_addr`` (str).
    """
    if len(s23) == 0:
        return _empty_result()

    entity_ids_in = _text(s23["entity_id"]).to_numpy(dtype=object)
    order = np.argsort(entity_ids_in, kind="stable")
    df = s23.iloc[order].reset_index(drop=True)
    n = len(df)

    entity_id = _text(df["entity_id"])
    country_code = _codes(df["country"])
    source_code = _codes(df["source"])
    cs_code = _combine(country_code, source_code)

    addr_code = _codes(df["addr_clean"])
    namekey_code = _codes(df["name_key"])
    city_code = _codes(df["city"])
    name_code = _codes(df["name_clean"])
    name_clean = _text(df["name_clean"]).to_numpy(dtype=object)

    effective_addr = _combine(cs_code, addr_code)
    effective_nk = _combine(cs_code, namekey_code)
    exact_addr_name = _combine(effective_addr, name_code)
    nk_city = _combine(effective_nk, city_code)

    edges_i, edges_j = [], []
    for i, j in (_star_edges_from_code(exact_addr_name), _star_edges_from_code(nk_city)):
        edges_i.append(i); edges_j.append(j)

    skipped_sizes: list[int] = []
    fi, fj = _addr_fuzzy_edges(effective_addr, name_clean, skipped_sizes)
    edges_i.append(fi); edges_j.append(fj)
    if skipped_sizes:
        log.info("sibling_groups: %d addr_clean group(s) skipped the fuzzy link "
                  "(KEY_MAX=%d); largest sizes %s", len(skipped_sizes), KEY_MAX,
                  sorted(skipped_sizes, reverse=True)[:10])

    oi, oj = _orphan_namekey_edges(effective_nk, addr_code, city_code, n)
    edges_i.append(oi); edges_j.append(oj)

    i_all = np.concatenate(edges_i) if edges_i else _EMPTY
    j_all = np.concatenate(edges_j) if edges_j else _EMPTY

    graph = sp.csr_matrix((np.ones(len(i_all), dtype=np.int8), (i_all, j_all)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    labels = labels.astype(np.int64)

    first_pos = np.full(n_comp, n, dtype=np.int64)
    np.minimum.at(first_pos, labels, np.arange(n, dtype=np.int64))
    order_comp = np.argsort(first_pos, kind="stable")
    rank_of_comp = np.empty(n_comp, dtype=np.int64)
    rank_of_comp[order_comp] = np.arange(n_comp, dtype=np.int64)
    sib_group_id = rank_of_comp[labels]

    sizes_per_comp = np.bincount(labels, minlength=n_comp).astype(np.int32)
    sib_group_size = sizes_per_comp[labels]

    addr_vals = _text(df["addr_clean"]).to_numpy(dtype=object)
    lengths = np.array([len(a) for a in addr_vals])
    non_empty = lengths > 0
    best_per_comp = np.full(n_comp, "", dtype=object)
    if non_empty.any():
        sub = pd.DataFrame({"comp": labels[non_empty], "addr": addr_vals[non_empty],
                            "len": lengths[non_empty]})
        sub = sub.sort_values(["comp", "len", "addr"], ascending=[True, False, True], kind="stable")
        best = sub.groupby("comp", sort=False)["addr"].first()
        best_per_comp[best.index.to_numpy()] = best.to_numpy()
    sib_best_addr = best_per_comp[labels]

    return pd.DataFrame({
        "entity_id": entity_id,
        "sib_group_id": sib_group_id,
        "sib_group_size": sib_group_size,
        "sib_best_addr": pd.array(sib_best_addr, dtype="string[pyarrow]"),
    })
