"""Tests for blocking / candidate generation (SPEC step 6, Task 9)."""
import pandas as pd
import pytest

import blocking
from blocking import generate_candidates, to_lists
from normalize import normalize_frame

V = {
    "US": {"indianapolis", "chicago", "hartford", "springfield", "dayton", "carmel"},
    "India": {"jaipur", "pune"},
}

S1_RAW = [
    ("S1-u1", "Porter & Nall", "3220 Gale Street, Indianapolis, IN", "US"),
    ("S1-u2", "Blue Heron Bakery", "12 Elm Street, Hartford, CT", "US"),
    ("S1-u3", "Sunrise Dental Care", "400 Main Street, Springfield, IL", "US"),
    ("S1-u4", "Acme Tools", "77 River Road, Dayton, OH", "US"),
    ("S1-i1", "Sharma Electricals", "Shop No 12, MG Road, Jaipur, Rajasthan 302001", "India"),
    ("S1-i2", "Gupta Sweets", "45 Station Road, Pune, Maharashtra 411001", "India"),
]

S23_RAW = [
    # the Porter & Nall cluster (SPEC section 6): all four are true matches of S1-u1
    ("S2-a1", "PORTER & NALL [LLC]", "", "US", "S2"),
    ("S2-a2", "porternall.com", "3220 GALE SAINT, INDIANAPOLIS, IN", "US", "S2"),
    ("S3-a3", "Porter & Ngial", "3220. Gale St, Indianapolis, Indiana", "US", "S3"),
    ("S3-a4", "Porter & Nall", "3220. Gale Street, Indianapolis, Indiana", "US", "S3"),
    # same-name decoy in another city
    ("S2-d1", "Porter & Nall", "88 Pine Street, Chicago, IL", "US", "S2"),
    # identical name + address, but India: must never pair with a US record
    ("S2-x1", "Porter & Nall", "3220 Gale Street, Jaipur, Rajasthan 302001", "India", "S2"),
    ("S3-x2", "Blue Heron Bakery", "12 Elm Street, Jaipur, Rajasthan 302001", "India", "S3"),
    # matches / noise for the other S1 records
    ("S2-b1", "Blue Heron Bakery LLC", "12 Elm St, Hartford, CT", "US", "S2"),
    ("S3-b2", "blue heron bakry", "12 Elm Street, Hartford, Connecticut", "US", "S3"),
    ("S2-c1", "Sunrise Dental", "400 Main St, Springfield, IL", "US", "S2"),
    ("S3-c2", "SUNRISE DENTAL CARE PC", "400 Main Street, Springfield, Illinois", "US", "S3"),
    ("S2-e1", "Acme Tool Co", "77 River Rd, Dayton, OH", "US", "S2"),
    ("S3-e2", "Riverside Grill", "12 Lake Avenue, Carmel, IN", "US", "S3"),
    ("S2-f1", "Sharma Electrical", "Shop 12, M G Road, Jaipur, Rajasthan 302001", "India", "S2"),
    ("S3-f2", "SHARMA ELECTRICALS", "Shop No 12, MG Road, Jaipur 302001", "India", "S3"),
    ("S2-g1", "Gupta Sweet House", "45 Station Rd, Pune 411001", "India", "S2"),
    ("S3-g2", "Gupta Sweets", "45 Station Road, Pune, Maharashtra", "India", "S3"),
    ("S2-h1", "Mehta Textiles", "9 Nehru Nagar, Pune 411002", "India", "S2"),
    ("S3-h2", "Kumar Traders", "3 Gandhi Road, Jaipur 302003", "India", "S3"),
    ("S2-h3", "Delta Plumbing", "5 Oak Avenue, Chicago, IL", "US", "S2"),
]


def _frame(rows, source=None):
    cols = ["entity_id", "business_name", "business_address", "country"]
    if source is None:
        cols = cols + ["source"]
    df = pd.DataFrame(rows, columns=cols)
    if source is not None:
        df["source"] = source
    for c in df.columns:
        df[c] = df[c].astype("string")
    return normalize_frame(df, {}, V)


@pytest.fixture(scope="module")
def toy():
    s1 = _frame(S1_RAW, source="S1")
    s23 = _frame(S23_RAW)
    return s1, s23, generate_candidates(s1, s23)


def _pairs(cands):
    return set(zip(cands["s1_id"], cands["s23_id"]))


def test_constants():
    assert blocking.TOPN_NAME == 20 and blocking.MIN_COS == 0.3 and blocking.REVERSE_TOPN == 3
    assert blocking.KEY_MAX == 200 and blocking.CAP_PER_SOURCE == 60 and blocking.EMBED_ENABLED is False


def test_output_schema(toy):
    _, _, cands = toy
    assert list(cands.columns) == ["s1_id", "s23_id", "source", "search_mask", "best_score"]
    assert cands["best_score"].dtype == "float32"
    assert pd.api.types.is_integer_dtype(cands["search_mask"])
    assert cands["search_mask"].between(1, 7).all()
    assert set(cands["source"]) <= {"S2", "S3"}


def test_planted_pairs_found(toy):
    _, _, cands = toy
    pairs = _pairs(cands)
    for s23_id in ("S2-a1", "S2-a2", "S3-a3", "S3-a4"):
        assert ("S1-u1", s23_id) in pairs


def test_empty_address_found_by_name(toy):
    _, _, cands = toy
    row = cands[(cands["s1_id"] == "S1-u1") & (cands["s23_id"] == "S2-a1")]
    assert len(row) == 1 and int(row["search_mask"].iloc[0]) & 1


def test_no_cross_country(toy):
    s1, s23, cands = toy
    c1 = dict(zip(s1["entity_id"], s1["country"]))
    c23 = dict(zip(s23["entity_id"], s23["country"]))
    assert len(cands) > 0
    assert all(c1[a] == c23[b] for a, b in zip(cands["s1_id"], cands["s23_id"]))
    assert ("S1-u1", "S2-x1") not in _pairs(cands) and ("S1-u2", "S3-x2") not in _pairs(cands)


def test_candidates_deduplicated(toy):
    _, _, cands = toy
    assert not cands.duplicated(["s1_id", "s23_id"]).any()
    # S3-a4 has the same name and address as S1-u1: found by A and B
    row = cands[(cands["s1_id"] == "S1-u1") & (cands["s23_id"] == "S3-a4")]
    assert int(row["search_mask"].iloc[0]) & 3 == 3


def test_address_key_finds_domain_name(toy):
    _, _, cands = toy
    row = cands[(cands["s1_id"] == "S1-u1") & (cands["s23_id"] == "S2-a2")]
    assert int(row["search_mask"].iloc[0]) & 2


def test_cap(toy):
    _, _, cands = toy
    assert cands.groupby(["s1_id", "source"]).size().max() <= blocking.CAP_PER_SOURCE


def test_cap_enforced_on_big_cluster():
    # 80 S2 records share S1's name and address: A gives 20, B gives all 80,
    # the cap keeps CAP_PER_SOURCE (60).
    s1 = _frame([("S1-1", "Porter & Nall", "3220 Gale Street, Indianapolis, IN", "US")], source="S1")
    rows = [(f"S2-{k:03d}", "Porter & Nall", "3220 Gale Street, Indianapolis, IN", "US", "S2") for k in range(80)]
    rows += [(f"S3-{k:03d}", "Porter & Nall", "3220 Gale Street, Indianapolis, IN", "US", "S3") for k in range(5)]
    cands = generate_candidates(s1, _frame(rows))
    per = cands.groupby("source").size()
    assert per["S2"] == blocking.CAP_PER_SOURCE and per["S3"] == 5


def test_key_max_group_skipped():
    # 250 unrelated names at one address: the address key group is too big.
    s1 = _frame([("S1-1", "Zeta Qux", "1 Main Street, Dayton, OH", "US")], source="S1")
    rows = [(f"S2-{k:03d}", f"Shop{k:03d} Wx{k:03d}", "1 Main Street, Dayton, OH", "US", "S2") for k in range(250)]
    cands = generate_candidates(s1, _frame(rows))
    assert not (cands["search_mask"] & 2).any()


def test_deterministic_and_inputs_untouched(toy):
    s1, s23, cands = toy
    s1_copy, s23_copy = s1.copy(), s23.copy()
    again = generate_candidates(s1.iloc[::-1], s23.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(s1, s1_copy)
    pd.testing.assert_frame_equal(s23, s23_copy)
    pd.testing.assert_frame_equal(cands.reset_index(drop=True), again.reset_index(drop=True))


def test_to_lists(toy):
    _, _, cands = toy
    lists = to_lists(cands)
    assert set(lists) == set(cands["s1_id"])
    assert sum(len(v) for v in lists.values()) == len(cands)
    assert "S2-a1" in lists["S1-u1"]


def test_empty_inputs():
    s1 = _frame(S1_RAW, source="S1")
    cands = generate_candidates(s1, _frame(S23_RAW).iloc[0:0])
    assert len(cands) == 0 and list(cands.columns) == ["s1_id", "s23_id", "source", "search_mask", "best_score"]


def test_tfidf_matches_sklearn():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    names = list(_frame(S1_RAW, source="S1")["name_sorted"]) + list(_frame(S23_RAW)["name_sorted"])
    ref = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2).fit_transform(names)
    codes, x, _ = blocking._tfidf_unique(pd.Series(names))
    got = x[codes]
    np.testing.assert_allclose((got @ got.T).toarray(), (ref @ ref.T).toarray(), atol=1e-5)


def test_chunked_search_matches_unchunked(toy, monkeypatch):
    s1, s23, cands = toy
    monkeypatch.setattr(blocking, "_COL_CHUNK", 3)
    monkeypatch.setattr(blocking, "_ROW_CHUNK", 2)
    pd.testing.assert_frame_equal(generate_candidates(s1, s23), cands)


def test_rare_word_codes():
    codes = blocking._rare_word_codes(pd.Series(["alpha beta", "beta gamma", "beta", "", None, "gamma alpha"]))
    # document frequencies: alpha 2, beta 3, gamma 2 -> alpha/gamma tie -> alpha first
    assert codes[0] == codes[5] and codes[1] != codes[0] and codes[3] == -1 and codes[4] == -1
    assert codes[2] >= 0


def test_same_name_crowd_resolved_by_address():
    # 65 same-name decoys crowd name search A (top 20, ties) and the cap; the true match
    # differs in city spelling but shares a name word + house number.
    s1 = _frame([("S1-1", "Sapphire Inc", "18049 Wilmont Road, King George County, VA", "US")], source="S1")
    rows = [(f"S2-{k:03d}", "Sapphire", f"{k + 100} Elm Street, Richmond, VA", "US", "S2") for k in range(65)]
    rows.append(("S2-zzz", "Sapphire  Inc", "18049 Wilmont Rd, King George, Virginia", "US", "S2"))
    cands = generate_candidates(s1, _frame(rows))
    assert "S2-zzz" in set(cands["s23_id"])
    assert len(cands) == blocking.CAP_PER_SOURCE


def test_alias_name_found_in_crowd():
    # "X formerly known as Y": the alias equals the S1 name, while 45 similar
    # names crowd forward name search A and 3 S1 look-alikes of the full
    # string take the reverse slots; the identical-alias key finds the match.
    s1 = _frame([("S1-1", "Secure Software Private Limited", "12 Elm Street, Dayton, OH", "US"),
                 ("S1-2", "Ariawexx Formerly Known As Secure", "1 Pine Street, Dayton, OH", "US"),
                 ("S1-3", "Ariawexx Formerly Known As Software", "2 Pine Street, Dayton, OH", "US"),
                 ("S1-4", "Ariawexx Formerly Known Secure Software", "3 Pine Street, Dayton, OH", "US")],
                source="S1")
    rows = [(f"S2-{k:03d}", "Secure Software Systems", f"{k + 100} Oak Avenue, Chicago, IL", "US", "S2")
            for k in range(45)]
    rows.append(("S2-zzz", "Ariawexx formerly known as Secure Software Private Limited", "", "US", "S2"))
    cands = generate_candidates(s1, _frame(rows))
    row = cands[(cands["s1_id"] == "S1-1") & (cands["s23_id"] == "S2-zzz")]
    assert len(row) == 1 and int(row["search_mask"].iloc[0]) & 1


def test_house_and_street_found_without_city():
    # unrelated name, city differs: only house number + street agree
    s1 = _frame([("S1-1", "Mullin & Jones LLC", "7125 Twin Lakes Road, Dayton, OH", "US")], source="S1")
    s23 = _frame([("S2-1", "Deltajax", "7125 Twin Lakes Rd, Carmel, OH", "US", "S2"),
                  ("S2-2", "Qorvix", "12 Twin Lakes Road, Carmel, OH", "US", "S2")])
    cands = generate_candidates(s1, s23)
    assert set(cands["s23_id"]) == {"S2-1"}
    assert int(cands["search_mask"].iloc[0]) & 2


def test_proxy_query_restriction(toy):
    # a proxy run returns exactly the full run's rows for the queried S1 records
    s1, s23, cands = toy
    q = {"S1-u1", "S1-i1"}
    part = blocking._generate(s1, s23, s1_query=q, s23_query=set(s23["entity_id"]))
    full = cands[cands["s1_id"].isin(q)].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.reset_index(drop=True), full)


def _alias_of(names, frequent=frozenset()):
    df = _frame([(f"S2-{k}", n, "", "US", "S2") for k, n in enumerate(names)])
    return list(blocking._alias(df["name_clean"], df["alt_name"], set(frequent)))


def test_alias_not_split_inside_initials():
    # "a k a" / "t a" inside a run of single-letter initials is not an alias phrase
    assert _alias_of(["M A K A Enterprises", "R T A Logistics Co"]) == ["", ""]


def test_genuine_alias_kept():
    assert _alias_of(["Apex Traders aka Blue Ocean Imports"]) == ["blue ocean imports"]


def test_one_common_word_alias_rejected():
    # a one-word alias made only of a frequent name word is not an alias ...
    assert _alias_of(["Nike Aka Store"], frequent={"store"}) == [""]
    # ... but a one-word alias with a rarer word is
    assert _alias_of(["Nike Aka Synlyra"], frequent={"store"}) == ["synlyra"]


def test_common_word_alias_gives_no_identity_bonus():
    # "store" is the country's most frequent name word: S1 "Store" must not get
    # an identical-name bonus with "Nike Aka Store"
    s1 = _frame([("S1-1", "Store", "5 Oak Avenue, Dayton, OH", "US")], source="S1")
    rows = [("S2-000", "Nike Aka Store", "9 Pine Street, Carmel, IN", "US", "S2")]
    rows += [(f"S2-{k:03d}", f"Qx{k:02d} Store", f"{k + 10} Elm Street, Chicago, IL", "US", "S2")
             for k in range(1, 8)]
    cands = generate_candidates(s1, _frame(rows))
    row = cands[cands["s23_id"] == "S2-000"]
    assert len(row) == 0 or float(row["best_score"].iloc[0]) < 1.0


def test_identity_bonus_independent_of_key_max(monkeypatch):
    # identical names: the score must not depend on whether the (KEY_MAX-limited)
    # identical-name key fired or only name search A found the pair
    s1 = _frame([("S1-1", "Porter & Nall", "3220 Gale Street, Indianapolis, IN", "US")], source="S1")
    s23 = _frame([("S2-1", "PORTER NALL", "", "US", "S2"),
                  ("S2-2", "Porter & Nall LLC", "", "US", "S2")])
    ref = generate_candidates(s1, s23).set_index("s23_id")["best_score"]
    monkeypatch.setattr(blocking, "KEY_MAX", 1)
    got = generate_candidates(s1, s23).set_index("s23_id")["best_score"]
    assert set(got.index) == set(ref.index)
    for k in ref.index:
        assert got[k] == pytest.approx(ref[k])
