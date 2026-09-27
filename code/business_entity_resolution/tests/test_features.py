import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, add_context_features, build_idf, compute_features
from normalize import normalize_frame

CITY_VOCAB = {"US": {"indianapolis", "springfield"}}

RAW = [
    # entity_id, business_name, business_address
    ("S1-1", "Porter & Nall LLC", "3220 Gale Street, Indianapolis, IN 46201"),
    ("S1-2", "Acme Plumbing Pvt Ltd", "12 Main St, Springfield, IL 62701"),
    ("S1-3", "Porter & Nall", ""),
    ("S2-1", "PORTER AND NALL", "3220. Gale St, Indianapolis, IN 46201"),
    ("S2-2", "Porter & Nall, Inc.", "3228 Gale Street, Indianapolis, IN 46201"),
    ("S3-1", "Acme Plumbing Private Limited", "12 Main Street, Springfield, IL 62701"),
    ("S3-2", "Acme Plumbing", "99 Oak Ave, Springfield, IL 62702"),
    ("S3-3", "Porter Nall dba PN Legal Services", "3220 Gale St, Indianapolis, IN 46201"),
]


def _norm(rows):
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address"])
    df["country"] = "US"
    df["source"] = df["entity_id"].str.split("-").str[0]
    return normalize_frame(df, {}, CITY_VOCAB)


@pytest.fixture(scope="module")
def frames():
    n = _norm(RAW)
    s1n = n[n["source"] == "S1"].reset_index(drop=True)
    s23n = n[n["source"] != "S1"].reset_index(drop=True)
    sib = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "sib_group_id": np.array([7, 8], dtype=np.int64),
        "sib_group_size": np.array([3, 1], dtype=np.int32),
        "sib_best_addr": ["3220 gale street indianapolis in", "3228 gale street indianapolis in"],
    })
    cands = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2", "S1-3", "S1-3"],
        "s23_id": ["S2-1", "S2-2", "S3-3", "S3-1", "S3-2", "S2-1", "S2-2"],
        "source": ["S2", "S2", "S3", "S3", "S3", "S2", "S2"],
        "search_mask": np.array([7, 7, 1, 5, 1, 2, 2], dtype=np.int8),
        "best_score": np.array([2.4, 1.9, 1.9, 2.2, 1.1, 1.5, 1.6], dtype=np.float32),
    })
    idf = build_idf(pd.concat([s1n["name_clean"], s23n["name_clean"]]))
    return cands, s1n, s23n, sib, idf


@pytest.fixture(scope="module")
def feats(frames):
    cands, s1n, s23n, sib, idf = frames
    ctx = add_context_features(cands, s1n, sib)
    return compute_features(ctx, s1n, s23n, sib, idf, n_jobs=2)


def _row(out, s1, s23):
    r = out[(out["s1_id"] == s1) & (out["s23_id"] == s23)]
    assert len(r) == 1
    return r.iloc[0]


def test_true_pair_house_and_street(feats):
    r = _row(feats, "S1-1", "S2-1")
    assert r["h_first_eq"] == 1
    assert r["a_street_set"] == 100
    assert r["a_postcode_eq"] == 1 and r["a_city_eq"] == 1 and r["a_state_eq"] == 1
    assert r["n_token_set"] == 100


def test_decoy_house_number(feats):
    r = _row(feats, "S1-1", "S2-2")
    assert r["h_first_eq"] == 0
    assert r["h_abs_gap"] == 8
    assert r["h_any_shared"] == 0
    assert r["h_edit"] == 1


def test_no_country_feature():
    assert set(FEATURE_COLUMNS).isdisjoint({"country"})
    assert not any("country" in c for c in FEATURE_COLUMNS)


def test_output_shape_order_and_dtype(frames, feats):
    cands = frames[0]
    assert len(feats) == len(cands)
    assert list(feats.columns) == ["s1_id", "s23_id"] + FEATURE_COLUMNS
    assert feats["s1_id"].tolist() == cands["s1_id"].tolist()
    assert feats["s23_id"].tolist() == cands["s23_id"].tolist()
    assert all(feats[c].dtype == np.float32 for c in FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 39


def test_rank_in_s1_and_gap(feats):
    r = feats.set_index(["s1_id", "s23_id"])
    assert r.loc[("S1-1", "S2-1"), "c_rank_in_s1"] == 1
    # tie on best_score 1.9 -> smaller s23_id first
    assert r.loc[("S1-1", "S2-2"), "c_rank_in_s1"] == 2
    assert r.loc[("S1-1", "S3-3"), "c_rank_in_s1"] == 3
    assert r.loc[("S1-1", "S2-1"), "c_gap_to_best"] == 0
    assert r.loc[("S1-1", "S2-2"), "c_gap_to_best"] == pytest.approx(0.5, abs=1e-5)
    assert r.loc[("S1-3", "S2-2"), "c_rank_in_s1"] == 1


def test_claimants(feats):
    r = feats.set_index(["s1_id", "s23_id"])
    assert r.loc[("S1-1", "S2-1"), "c_n_claimants"] == 2
    assert r.loc[("S1-1", "S2-1"), "c_rank_among_claimants"] == 1  # 2.4 beats 1.5
    assert r.loc[("S1-3", "S2-1"), "c_rank_among_claimants"] == 2
    assert r.loc[("S1-3", "S2-2"), "c_rank_among_claimants"] == 2  # S1-1 has 1.9 > 1.6
    assert r.loc[("S1-2", "S3-1"), "c_n_claimants"] == 1
    assert r.loc[("S1-2", "S3-1"), "c_rank_among_claimants"] == 1


def test_context_misc(feats):
    r = feats.set_index(["s1_id", "s23_id"])
    # S1-1 and S1-3 share name_sorted ("and nall porter")
    assert r.loc[("S1-1", "S2-1"), "c_name_freq"] == 2
    assert r.loc[("S1-2", "S3-1"), "c_name_freq"] == 1
    assert r.loc[("S1-1", "S2-1"), "c_sib_size"] == 3
    assert r.loc[("S1-2", "S3-1"), "c_sib_size"] == 1  # absent from sib -> 1
    assert r.loc[("S1-2", "S3-1"), "c_source_is_s3"] == 1
    assert r.loc[("S1-1", "S2-1"), "c_source_is_s3"] == 0
    assert r.loc[("S1-2", "S3-1"), "c_search_mask"] == 5


def test_legal_state(feats):
    r = feats.set_index(["s1_id", "s23_id"])
    assert r.loc[("S1-1", "S2-2"), "legal_state"] == 2  # llc vs inc
    assert r.loc[("S1-1", "S2-1"), "legal_state"] == 3  # llc vs none
    assert r.loc[("S1-2", "S3-1"), "legal_state"] in (0, 1)  # pvt ltd vs private limited


def test_missing_address_gives_nan(feats):
    r = _row(feats, "S1-3", "S2-1")
    for c in ["h_first_eq", "h_abs_gap", "a_postcode_eq", "a_city_eq", "a_street_set", "a_token_set"]:
        assert np.isnan(r[c]), c
    assert r["a_has_s1"] == 0 and r["a_has_s23"] == 1
    assert r["h_cnt_s1"] == 0


def test_alt_name(feats):
    assert np.isnan(_row(feats, "S1-1", "S2-1")["n_alt_best"])
    assert not np.isnan(_row(feats, "S1-1", "S3-3")["n_alt_best"])


def test_sib_best_and_name_scores(feats):
    r = _row(feats, "S1-1", "S2-1")
    assert r["a_sib_best_set"] == 100
    assert np.isnan(_row(feats, "S1-2", "S3-1")["a_sib_best_set"])
    assert r["n_tfidf_cos"] == pytest.approx(1.0, abs=1e-5)
    assert r["n_idf_jaccard"] == pytest.approx(1.0, abs=1e-5)
    assert r["n_unshared_cnt"] == 0 and r["n_key_lev"] == 0
    d = _row(feats, "S1-2", "S3-2")
    assert d["n_idf_jaccard"] == pytest.approx(1.0, abs=1e-5)  # legal words stripped
    assert d["h_first_eq"] == 0 and d["a_postcode_eq"] == 0


def test_chunk_invariance(frames):
    cands, s1n, s23n, sib, idf = frames
    ctx = add_context_features(cands, s1n, sib)
    whole = compute_features(ctx, s1n, s23n, sib, idf, n_jobs=1)
    parts = pd.concat([
        compute_features(ctx.iloc[:3], s1n, s23n, sib, idf, n_jobs=1),
        compute_features(ctx.iloc[3:], s1n, s23n, sib, idf, n_jobs=1),
    ], ignore_index=True)
    pd.testing.assert_frame_equal(whole, parts)


def test_compute_features_requires_context_columns(frames):
    # context features need the whole per-country table; computing them from
    # a chunk would silently change ranks/claimants with chunk boundaries
    cands, s1n, s23n, sib, idf = frames
    with pytest.raises(ValueError, match="add_context_features"):
        compute_features(cands, s1n, s23n, sib, idf, n_jobs=1)
    ctx = add_context_features(cands, s1n, sib)
    with pytest.raises(ValueError, match="c_rank_in_s1"):
        compute_features(ctx.drop(columns=["c_rank_in_s1"]), s1n, s23n, sib, idf, n_jobs=1)


def _ctx_frames(names):
    s1n = pd.DataFrame({"entity_id": [f"S1-{k}" for k in range(len(names))], "country": "US",
                        "name_sorted": names})
    sib = pd.DataFrame({"entity_id": pd.Series([], dtype=object), "sib_group_size": np.array([], np.int32)})
    return s1n, sib


def test_n_claimants_counts_distinct_s1():
    s1n, sib = _ctx_frames(["a", "b"])
    cands = pd.DataFrame({
        "s1_id": ["S1-0", "S1-0", "S1-1"],   # (S1-0, S2-9) appears twice
        "s23_id": ["S2-9", "S2-9", "S2-9"],
        "source": "S2", "search_mask": np.int8(1),
        "best_score": np.array([2.0, 2.0, 1.0], dtype=np.float32),
    })
    out = add_context_features(cands, s1n, sib)
    assert out["c_n_claimants"].tolist() == [2, 2, 2]


def test_name_freq_nan_for_empty_name():
    s1n, sib = _ctx_frames(["", "", "acme"])
    cands = pd.DataFrame({"s1_id": ["S1-0", "S1-1", "S1-2"], "s23_id": ["S2-1", "S2-2", "S2-3"],
                          "source": "S2", "search_mask": np.int8(1),
                          "best_score": np.ones(3, dtype=np.float32)})
    out = add_context_features(cands, s1n, sib)
    assert np.isnan(out["c_name_freq"].iloc[0]) and np.isnan(out["c_name_freq"].iloc[1])
    assert out["c_name_freq"].iloc[2] == 1


def test_build_idf_rare_word_scores_higher():
    idf = build_idf(pd.Series(["acme shop", "best shop", "zeta shop", ""]))
    assert idf["acme"] > idf["shop"]
    assert set(idf) == {"acme", "best", "zeta", "shop"}


def test_empty_candidates(frames):
    cands, s1n, s23n, sib, idf = frames
    ctx = add_context_features(cands.iloc[:0], s1n, sib)
    out = compute_features(ctx, s1n, s23n, sib, idf, n_jobs=1)
    assert len(out) == 0
    assert list(out.columns) == ["s1_id", "s23_id"] + FEATURE_COLUMNS


def test_char_tfidf_matches_sklearn():
    from sklearn.feature_extraction.text import TfidfVectorizer

    from features import build_tfidf

    names = ["and nall porter", "acme plumbing", "a b", "", "cafe bar", "nall porter", "x"]
    x = build_tfidf(pd.Series(names)).transform(names)
    ref = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True).fit(names).transform(names)
    np.testing.assert_allclose((x @ x.T).toarray(), (ref @ ref.T).toarray(), atol=1e-6)
