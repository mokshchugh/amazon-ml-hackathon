"""Unit tests for run_all helpers (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

import features
import io_utils
import run_all


# ---------------------------------------------------------------------------
# limit-mode slice sampling
# ---------------------------------------------------------------------------

def _toy_sources():
    s1 = pd.DataFrame({
        "entity_id": [f"S1-{i:03d}" for i in range(100)],
        "business_name": [f"n{i}" for i in range(100)],
        "business_address": ["a"] * 100,
        "country": ["US"] * 60 + ["India"] * 40,
    })
    s23 = pd.DataFrame({
        "entity_id": [f"S2-{i:04d}" for i in range(3000)] + [f"S3-{i:04d}" for i in range(3000)],
        "country": (["US"] * 1500 + ["India"] * 1500) * 2,
    })
    # each S1 i has GT partners S2-i and S3-(i+1500 if India else i)
    rows = []
    for i in range(100):
        off = 1500 if i >= 60 else 0
        rows += [(f"S1-{i:03d}", f"S2-{i + off:04d}"), (f"S1-{i:03d}", f"S3-{i + off:04d}")]
    gt = pd.DataFrame(rows, columns=["s1_id", "s23_id"])
    return s1, s23, gt


def test_sample_slice_contents():
    s1, s23, gt = _toy_sources()
    s1s, ids = run_all.sample_slice(s1, s23, gt, n=10, seed=42, decoy_factor=10)
    assert len(s1s) == 10 and s1s["entity_id"].is_unique
    ids = set(ids)
    partners = set(gt.loc[gt["s1_id"].isin(s1s["entity_id"]), "s23_id"])
    assert partners <= ids                                   # every GT partner present
    country = dict(zip(s23["entity_id"], s23["country"]))
    per_country = s1s["country"].value_counts()
    for c, k in per_country.items():
        n_c = sum(1 for x in ids if country[x] == c)
        assert 10 * k <= n_c <= 10 * k + len(partners)       # ~10x decoys, same country
    assert set(country[x] for x in ids) <= set(s1s["country"])  # no other-country records


def test_sample_slice_deterministic_and_order_independent():
    s1, s23, gt = _toy_sources()
    a1, a2 = run_all.sample_slice(s1, s23, gt, n=10, seed=42)
    b1, b2 = run_all.sample_slice(s1.sample(frac=1, random_state=1), s23.sample(frac=1, random_state=2), gt,
                                  n=10, seed=42)
    assert a1["entity_id"].tolist() == b1["entity_id"].tolist()
    assert list(a2) == list(b2)
    c1, _ = run_all.sample_slice(s1, s23, gt, n=10, seed=7)
    assert c1["entity_id"].tolist() != a1["entity_id"].tolist()


def test_sample_slice_without_gt():
    s1, s23, _ = _toy_sources()
    s1s, ids = run_all.sample_slice(s1, s23, None, n=5, seed=42, decoy_factor=10)
    assert len(s1s) == 5 and len(ids) == 50


# ---------------------------------------------------------------------------
# baseline threshold
# ---------------------------------------------------------------------------

def test_tune_baseline_threshold_hand_computed():
    # f if predicted: A 1.0 (1 of 1), B 1.25/(0.5+1)=0.8333 (1 of 2), C 0 (singleton), D 0 (wrong)
    # f if empty:     C 1 (singleton), E 1 (singleton, no candidate)
    top1 = pd.DataFrame({"s1_id": ["A", "B", "C", "D"], "s23_id": ["x", "y", "q", "v"],
                         "best_score": [0.9, 0.8, 0.7, 0.95]})
    truth = {"A": {"x"}, "B": {"y", "z"}, "D": {"w"}}
    t, f = run_all.tune_baseline_threshold(top1, truth, ["A", "B", "C", "D", "E"])
    # t=0.95: 2; t=0.9: 3; t=0.8: 3.8333; t=0.7: 2.8333; t=inf: 2
    assert t == pytest.approx(0.8)
    assert f == pytest.approx((2 + 1 + 1.25 / 1.5) / 5)


def test_tune_baseline_threshold_predicts_nothing_when_useless():
    top1 = pd.DataFrame({"s1_id": ["A"], "s23_id": ["x"], "best_score": [0.5]})
    t, f = run_all.tune_baseline_threshold(top1, {}, ["A", "B"])
    assert t == float("inf") and f == pytest.approx(1.0)


def test_top1_per_s1_ties_to_smaller_id():
    c = pd.DataFrame({"s1_id": ["A", "A", "A", "B"], "s23_id": ["z", "b", "c", "k"],
                      "best_score": [0.5, 0.9, 0.9, 0.1]})
    c["source"] = "S2"
    c["search_mask"] = 1
    out = run_all.top1_per_s1(run_all.CandCodes.from_frame(c))
    assert dict(zip(out["s1_id"], out["s23_id"])) == {"A": "b", "B": "k"}


# ---------------------------------------------------------------------------
# per-country tables, chunked context == whole-table context
# ---------------------------------------------------------------------------

def _toy_cands(seed=0, n1=30, n23=20):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n1):
        for j in rng.choice(n23, size=rng.integers(1, 9), replace=False):
            rows.append((f"S1-{i:02d}", f"S{2 + j % 2}-{j:02d}", f"S{2 + j % 2}", int(rng.integers(1, 8)),
                         float(np.round(rng.random(), 1))))
    df = pd.DataFrame(rows, columns=run_all.CAND_COLS).sample(frac=1, random_state=seed).reset_index(drop=True)
    s1 = pd.DataFrame({"entity_id": [f"S1-{i:02d}" for i in range(n1)],
                       "name_sorted": [f"name{i % 7}" if i % 5 else "" for i in range(n1)],
                       "country": ["US"] * n1})
    sib = pd.DataFrame({"entity_id": [f"S2-{j:02d}" for j in range(0, n23, 2)],
                        "sib_group_id": [j // 4 for j in range(0, n23, 2)],
                        "sib_group_size": [2] * (n23 // 2), "sib_best_addr": [""] * (n23 // 2)})
    return df, s1, sib


def test_order_and_trim_groups_and_topk():
    df, _, _ = _toy_cands()
    t = run_all.CandCodes.from_frame(df)
    out = run_all.order_and_trim(t, None).frame()
    assert len(out) == len(df)
    codes = pd.factorize(out["s1_id"], sort=True)[0]
    assert np.all(np.diff(codes) >= 0)
    top2 = run_all.order_and_trim(t, 2).frame()
    assert top2.groupby(["s1_id", "source"]).size().max() <= 2
    for (s1, src), g in df.groupby(["s1_id", "source"]):
        want = g.sort_values(["best_score", "s23_id"], ascending=[False, True]).head(2)["s23_id"].tolist()
        got = top2[(top2["s1_id"] == s1) & (top2["source"] == src)]["s23_id"].tolist()
        assert got == want


def test_s1_chunk_bounds_never_split_a_record():
    ids = pd.Series(np.repeat([f"S1-{i}" for i in range(50)], np.arange(1, 51) % 7 + 1))
    bounds = run_all.s1_chunk_bounds(ids, max_rows=10)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(ids)
    for (a, b), (c, _) in zip(bounds, bounds[1:]):
        assert b == c
    for a, b in bounds:
        assert b - a <= 10
        if a:
            assert ids[a] != ids[a - 1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chunked_context_equals_whole_table(seed):
    df, s1, sib = _toy_cands(seed)
    t = run_all.order_and_trim(run_all.CandCodes.from_frame(df), None)
    whole = features.add_context_features(t.frame(), s1, sib)
    n_claim, rank_claim = run_all.claimant_features(t)
    parts = [ctx for _, _, ctx in run_all.context_chunks(t, s1, sib, n_claim, rank_claim, max_rows=10)]
    assert len(parts) > 3
    chunked = pd.concat(parts, ignore_index=True)
    for c in features.CONTEXT_FEATURES:
        np.testing.assert_array_equal(chunked[c].to_numpy(), whole[c].to_numpy(), err_msg=c)


def test_grouped_lists_with_write_id_lists(tmp_path):
    lists = run_all.GroupedLists.from_ids(["A", "A", "B", "D", "D", "D"], ["x", "y", "z", "u", "u", "v"])
    path = tmp_path / "c.tsv"
    io_utils.write_id_lists(path, "candidate_entity_ids", ["A", "B", "C", "D"], lists)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "source1_entity_id\tcandidate_entity_ids", "A\tx,y", "B\tz", "C\t", "D\tu,v"]


def test_grouped_lists_rejects_non_contiguous():
    with pytest.raises(ValueError):
        run_all.GroupedLists.from_ids(["A", "B", "A"], ["x", "y", "z"])


def test_whole_table_context_matches_string_table():
    """Code tables reproduce add_context_features on the original string frame
    (tie order by id), for every row, in the grouped order."""
    df, s1, sib = _toy_cands(5)
    t = run_all.order_and_trim(run_all.CandCodes.from_frame(df), None)
    ref = features.add_context_features(df, s1, sib)
    got = features.add_context_features(t.frame(), s1, sib)
    key = ["s1_id", "s23_id", "best_score"]
    ref = ref.sort_values(key).reset_index(drop=True)
    got = got.sort_values(key).reset_index(drop=True)
    for c in features.CONTEXT_FEATURES:
        np.testing.assert_array_equal(got[c].to_numpy(), ref[c].to_numpy(), err_msg=c)


def test_load_cand_codes_roundtrip(tmp_path):
    df, _, _ = _toy_cands(3)
    df["search_mask"] = df["search_mask"].astype(np.int8)
    df["best_score"] = df["best_score"].astype(np.float32)
    path = tmp_path / "c.parquet"
    df.to_parquet(path, index=False)
    u1, u23 = run_all.sorted_ids(df["s1_id"]), run_all.sorted_ids(df["s23_id"])
    t = run_all.load_cand_codes(path, u1, u23, batch_rows=7)
    back = t.frame()
    for c in run_all.CAND_COLS:
        assert back[c].astype(str).tolist() == df[c].astype(str).tolist(), c
    with pytest.raises(KeyError):
        run_all.load_cand_codes(path, u1, run_all.sorted_ids(df["s23_id"].iloc[:3]))
