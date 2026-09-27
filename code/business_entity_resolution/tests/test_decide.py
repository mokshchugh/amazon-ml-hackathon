"""Tests for src/decide.py: the F0.5 decision layer (SPEC section 7)."""
import numpy as np
import pandas as pd
import pytest

from decide import (
    DecisionParams,
    best_k,
    decide,
    expected_f05,
    resolve_owners,
    tune,
    tune_scores,
)
from evaluate import macro_f05

EMPTY_SIB = pd.DataFrame({
    "entity_id": pd.Series([], dtype=object),
    "sib_group_id": pd.Series([], dtype="int64"),
    "sib_group_size": pd.Series([], dtype="int32"),
    "sib_best_addr": pd.Series([], dtype=object),
})


def _sib(groups: dict[str, int]) -> pd.DataFrame:
    ids = list(groups)
    gid = [groups[i] for i in ids]
    size = pd.Series(gid).map(pd.Series(gid).value_counts()).to_numpy()
    return pd.DataFrame({
        "entity_id": ids,
        "sib_group_id": np.array(gid, dtype=np.int64),
        "sib_group_size": size.astype(np.int32),
        "sib_best_addr": [""] * len(ids),
    })


# ---------------------------------------------------------------- brief tests
def test_best_k():
    assert best_k(np.array([0.9, 0.9, 0.1])) == 2
    assert best_k(np.array([0.3])) == 0
    assert best_k(np.array([0.95])) == 1


def test_one_owner():
    df = pd.DataFrame({"s1_id": ["A", "B", "C", "D"], "s23_id": ["x", "x", "y", "y"], "p": [0.9, 0.8, 0.9, 0.7]})
    out = resolve_owners(df, margin=0.15)
    assert set(zip(out.s1_id, out.s23_id)) == {("C", "y")}   # x dropped (gap 0.1 < 0.15), y kept by C


def test_t_empty():
    scored = pd.DataFrame({"s1_id": ["A"], "s23_id": ["x"], "p": [0.45]})
    assert decide(scored, EMPTY_SIB, DecisionParams())["A"] == []


def test_lone_sibling_dropped():
    scored = pd.DataFrame({"s1_id": ["A"]*3, "s23_id": ["x1", "x2", "x3"], "p": [0.9, 0.05, 0.05]})
    sib = pd.DataFrame({"entity_id": ["x1", "x2", "x3"], "sib_group_id": [7, 7, 7], "sib_group_size": [3, 3, 3], "sib_best_addr": [""]*3})
    assert decide(scored, sib, DecisionParams())["A"] == []      # lone kept member, 0.9 < lone_keep 0.95
    scored.loc[0, "p"] = 0.97
    assert decide(scored, sib, DecisionParams())["A"] == ["x1"]  # 0.97 >= 0.95 survives


# ---------------------------------------------------------------- extra tests
def test_defaults():
    p = DecisionParams()
    assert (p.margin, p.t_empty, p.t_sib, p.lone_keep) == (0.15, 0.5, 0.5, 0.95)


def test_expected_f05_values():
    probs = np.array([0.9, 0.9, 0.1])
    assert expected_f05(probs, 0) == pytest.approx(0.1 * 0.1 * 0.9)
    # k=2: TP=1.8, FP=0.2, FN=0.1 -> 2.25 / (2.25 + 0.025 + 0.2)
    assert expected_f05(probs, 2) == pytest.approx(2.25 / 2.475)
    # k=3: TP=1.9, FP=1.1, FN=0 -> 2.375 / 3.475
    assert expected_f05(probs, 3) == pytest.approx(2.375 / 3.475)


def test_best_k_is_argmax_of_expected_f05():
    rng = np.random.default_rng(3)
    for _ in range(200):
        probs = np.sort(rng.random(rng.integers(1, 8)))[::-1]
        vals = [expected_f05(probs, k) for k in range(len(probs) + 1)]
        assert best_k(probs) == int(np.argmax(vals))


def test_best_k_empty():
    assert best_k(np.array([])) == 0


def test_resolve_owners_tie():
    # equal top claims -> nobody owns x, even with margin 0; single claimant always survives
    df = pd.DataFrame({"s1_id": ["A", "B", "C"], "s23_id": ["x", "x", "y"], "p": [0.8, 0.8, 0.2]})
    for m in (0.0, 0.15):
        out = resolve_owners(df, margin=m)
        assert set(zip(out.s1_id, out.s23_id)) == {("C", "y")}


def test_resolve_owners_margin_boundary_and_columns():
    df = pd.DataFrame({"s1_id": ["A", "B", "C"], "s23_id": ["x", "x", "x"], "p": [0.3, 0.95, 0.8]})
    out = resolve_owners(df, margin=0.15)       # gap 0.95-0.8 is 0.15 up to float rounding
    assert list(out.columns) == ["s1_id", "s23_id", "p"]
    assert set(zip(out.s1_id, out.s23_id)) == {("B", "x")}
    assert resolve_owners(df, margin=0.2).empty


def test_decide_owner_loser_gets_empty_list_and_order():
    scored = pd.DataFrame({
        "s1_id": ["A", "A", "A", "B"],
        "s23_id": ["z", "b", "a", "z"],
        "p": [0.99, 0.97, 0.97, 0.99],
    })
    out = decide(scored, EMPTY_SIB, DecisionParams())
    assert out["B"] == []                       # z tied between A and B -> dropped for both
    assert out["A"] == ["a", "b"]               # equal p -> smaller s23_id first


def test_sibling_add():
    sib = _sib({"x1": 7, "x2": 7, "x3": 7})
    scored = pd.DataFrame({"s1_id": ["A"]*3, "s23_id": ["x1", "x2", "x3"], "p": [0.9, 0.9, 0.55]})
    # best_k keeps 2 of 3 (> 50%) -> x3 added since 0.55 >= t_sib
    assert best_k(np.array([0.9, 0.9, 0.55])) == 2
    assert decide(scored, sib, DecisionParams())["A"] == ["x1", "x2", "x3"]
    assert decide(scored, sib, DecisionParams(t_sib=0.7))["A"] == ["x1", "x2"]
    # without the sibling table nothing is added
    assert decide(scored, EMPTY_SIB, DecisionParams())["A"] == ["x1", "x2"]


def test_sibling_rules_only_count_present_members():
    # group 7 has 3 members but only x1 is among A's candidates -> no lone rule
    sib = _sib({"x1": 7, "x2": 7, "x3": 7})
    scored = pd.DataFrame({"s1_id": ["A", "A"], "s23_id": ["x1", "y"], "p": [0.9, 0.05]})
    assert decide(scored, sib, DecisionParams())["A"] == ["x1"]


# ------------------------------------------------ vectorized vs slow reference
def _reference_decide(scored, sib, params):
    """Slow, obviously-correct per-record implementation of the rules."""
    rows = list(zip(scored.s1_id, scored.s23_id, scored.p.astype(float)))
    rows = list(dict(((s1, s23), p) for s1, s23, p in rows).items())  # unique pairs
    by_s23 = {}
    for (s1, s23), p in rows:
        by_s23.setdefault(s23, []).append((p, s1))
    owner = {}
    for s23, claims in by_s23.items():
        claims.sort(key=lambda c: -c[0])
        if len(claims) == 1:
            owner[s23] = claims[0][1]
        elif claims[0][0] > claims[1][0] and claims[0][0] - claims[1][0] >= params.margin - 1e-9:
            owner[s23] = claims[0][1]
    cands = {s1: [] for (s1, _), _ in rows}
    for (s1, s23), p in rows:
        if owner.get(s23) == s1:
            cands[s1].append((p, s23))
    gid = dict(zip(sib.entity_id, sib.sib_group_id))
    out = {}
    for s1, cl in cands.items():
        cl.sort(key=lambda c: (-c[0], c[1]))
        probs = np.array([c[0] for c in cl])
        k = best_k(probs)
        if len(cl) == 0 or cl[0][0] < params.t_empty:
            k = 0
        keep = [i < k for i in range(len(cl))]
        groups = {}
        for i, (_, s23) in enumerate(cl):
            if s23 in gid:
                groups.setdefault(gid[s23], []).append(i)
        new_keep = list(keep)
        for members in groups.values():
            if len(members) < 2:
                continue
            n_kept = sum(keep[i] for i in members)
            if 2 * n_kept > len(members):
                for i in members:
                    if not keep[i] and cl[i][0] >= params.t_sib:
                        new_keep[i] = True
            elif n_kept == 1:
                for i in members:
                    if keep[i] and cl[i][0] < params.lone_keep:
                        new_keep[i] = False
        out[s1] = [cl[i][1] for i in range(len(cl)) if new_keep[i]]
    return out


def _random_case(seed, n_s1=80, n_s23=500, block=10):
    """Each S1 claims a random block of consecutive s23 ids (overlapping
    blocks -> contested owners); sibling groups are runs of consecutive ids,
    so a candidate list holds several members of the same group."""
    rng = np.random.default_rng(seed)
    s1, s23 = [], []
    for i in range(n_s1):
        start = int(rng.integers(0, n_s23 - block))
        for j in range(start, start + int(rng.integers(1, block + 1))):
            s1.append(i)
            s23.append(j)
    n_rows = len(s1)
    # coarse, mostly-confident probabilities -> many ties (like isotonic plateaus)
    p = np.round(rng.beta(1.2, 0.5, n_rows) * 20) / 20
    scored = pd.DataFrame({
        "s1_id": [f"A{i:03d}" for i in s1],
        "s23_id": [f"x{i:03d}" for i in s23],
        "p": p,
    }).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    ids = [f"x{i:03d}" for i in range(n_s23)]
    gid = np.cumsum(rng.random(n_s23) < 0.35)          # runs of ~3 consecutive ids
    in_sib = rng.random(n_s23) < 0.9
    groups = {e: int(g) for e, g, keep in zip(ids, gid, in_sib) if keep}
    return scored, _sib(groups)


@pytest.mark.parametrize("seed", range(6))
def test_decide_matches_reference(seed):
    scored, sib = _random_case(seed)
    for params in (DecisionParams(), DecisionParams(margin=0.05, t_empty=0.3, t_sib=0.3),
                   DecisionParams(margin=0.0, t_empty=0.7, t_sib=0.7)):
        got = decide(scored, sib, params)
        want = _reference_decide(scored, sib, params)
        assert got == want


# ----------------------------------------------------------------------- tune
def _truth_for(scored, seed):
    rng = np.random.default_rng(seed)
    truth = {}
    for s1, s23, p in zip(scored.s1_id, scored.s23_id, scored.p):
        if rng.random() < p:
            truth.setdefault(s1, set()).add(s23)
    return truth


def test_tune_scores_match_macro_f05():
    scored, sib = _random_case(5)
    truth = _truth_for(scored, 5)
    s1_ids = sorted(set(scored.s1_id)) + ["NOCAND1", "NOCAND2"]
    truth["NOCAND2"] = {"zzz"}                  # true match never reached the candidates
    grid = tune_scores(scored, sib, truth, s1_ids)
    assert len(grid) == 75
    for row in grid.itertuples():
        params = DecisionParams(margin=row.margin, t_empty=row.t_empty, t_sib=row.t_sib)
        want = macro_f05({k: set(v) for k, v in decide(scored, sib, params).items()}, truth, s1_ids)
        assert row.score == pytest.approx(want, abs=1e-12)


def test_tune_returns_best_grid_point():
    scored, sib = _random_case(8)
    truth = _truth_for(scored, 8)
    s1_ids = sorted(set(scored.s1_id))
    grid = tune_scores(scored, sib, truth, s1_ids)
    best = tune(scored, sib, truth, s1_ids)
    assert isinstance(best, DecisionParams)
    assert best.lone_keep == 0.95
    hit = grid[(grid.margin == best.margin) & (grid.t_empty == best.t_empty) & (grid.t_sib == best.t_sib)]
    assert hit.score.iloc[0] == pytest.approx(grid.score.max(), abs=1e-12)


def test_tune_ties_prefer_defaults():
    # every grid point scores 1.0 -> the defaults win
    scored = pd.DataFrame({"s1_id": ["A"], "s23_id": ["x"], "p": [0.99]})
    best = tune(scored, EMPTY_SIB, {"A": {"x"}}, ["A"])
    assert (best.margin, best.t_empty, best.t_sib) == (0.15, 0.5, 0.5)


def test_tune_grid_values():
    scored = pd.DataFrame({"s1_id": ["A"], "s23_id": ["x"], "p": [0.99]})
    grid = tune_scores(scored, EMPTY_SIB, {"A": {"x"}}, ["A"])
    assert sorted(set(grid.margin)) == [0.05, 0.1, 0.15, 0.2, 0.3]
    assert sorted(set(grid.t_empty)) == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert sorted(set(grid.t_sib)) == [0.3, 0.5, 0.7]
    assert len(set(zip(grid.margin, grid.t_empty, grid.t_sib))) == 75


def test_missing_ids_rejected():
    scored = pd.DataFrame({"s1_id": ["A", None], "s23_id": ["x", "y"], "p": [0.9, 0.9]})
    with pytest.raises(ValueError):
        decide(scored, EMPTY_SIB, DecisionParams())
