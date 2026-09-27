import pandas as pd
import pytest

from siblings import KEY_MAX, sibling_groups


def _frame(rows):
    """Build a minimal S2/S3-shaped frame from explicit column values (R6:
    tests don't call normalize in T10)."""
    cols = ["entity_id", "country", "source", "name_clean", "name_key", "addr_clean", "city"]
    return pd.DataFrame(rows, columns=cols)


def _group_of(out: pd.DataFrame, entity_id: str) -> int:
    row = out.loc[out["entity_id"] == entity_id]
    assert len(row) == 1
    return int(row["sib_group_id"].iloc[0])


def test_no_address_record_joins_named_sibling_via_name_key():
    # SPEC example: PORTER & NALL [LLC] (no address) joins porternall.com's
    # group through the shared name_key, and inherits its best address.
    out = sibling_groups(_frame([
        ("p1", "US", "S2", "porter and nall llc", "porternall", "", ""),
        ("p2", "US", "S2", "porternall com", "porternall",
         "3220 gale street indianapolis in", "indianapolis"),
    ]))
    assert _group_of(out, "p1") == _group_of(out, "p2")
    best = out.set_index("entity_id")["sib_best_addr"]
    assert best["p1"] == best["p2"] == "3220 gale street indianapolis in"
    sizes = out.set_index("entity_id")["sib_group_size"]
    assert sizes["p1"] == sizes["p2"] == 2


def test_same_address_low_name_similarity_stays_separate():
    out = sibling_groups(_frame([
        ("p2", "US", "S2", "porternall com", "porternall",
         "3220 gale street indianapolis in", "indianapolis"),
        ("p3", "US", "S2", "zzz totally unrelated biz", "zzztotallyunrelatedbiz",
         "3220 gale street indianapolis in", "indianapolis"),
    ]))
    assert _group_of(out, "p2") != _group_of(out, "p3")
    sizes = out.set_index("entity_id")["sib_group_size"]
    assert sizes["p2"] == 1 and sizes["p3"] == 1


def test_s2_and_s3_never_share_a_group():
    out = sibling_groups(_frame([
        ("p2", "US", "S2", "porternall com", "porternall",
         "3220 gale street indianapolis in", "indianapolis"),
        ("q2", "US", "S3", "porternall com", "porternall",
         "3220 gale street indianapolis in", "indianapolis"),
    ]))
    assert _group_of(out, "p2") != _group_of(out, "q2")
    sizes = out.set_index("entity_id")["sib_group_size"]
    assert sizes["p2"] == 1 and sizes["q2"] == 1


def test_name_key_and_city_links_even_with_different_address_text():
    out = sibling_groups(_frame([
        ("r1", "US", "S2", "acme plumbing", "acmeplumbing", "12 main st springfield", "springfield"),
        ("r2", "US", "S2", "acme plumbing inc", "acmeplumbing", "", "springfield"),
    ]))
    assert _group_of(out, "r1") == _group_of(out, "r2")


def test_orphan_namekey_does_not_link_when_name_key_has_two_addresses():
    # A no-address/no-city record must NOT borrow an address when its
    # name_key is attached to more than one distinct address in this
    # (country, source) -- e.g. a common chain name at different locations.
    out = sibling_groups(_frame([
        ("o1", "US", "S2", "franchise co", "franchiseco", "", ""),
        ("o2", "US", "S2", "franchise co", "franchiseco", "1 first ave anytown", "anytown"),
        ("o3", "US", "S2", "franchise co", "franchiseco", "2 second ave otherville", "otherville"),
    ]))
    assert _group_of(out, "o1") != _group_of(out, "o2")
    assert _group_of(out, "o1") != _group_of(out, "o3")
    assert _group_of(out, "o2") != _group_of(out, "o3")
    sizes = out.set_index("entity_id")["sib_group_size"]
    assert sizes["o1"] == 1


def test_orphan_namekey_requires_both_address_and_city_empty():
    # A record with an empty address but a (mismatched) non-empty city must
    # not use the orphan name_key rule -- only same-name_key+city applies,
    # and cities differ here, so it stays separate.
    out = sibling_groups(_frame([
        ("n1", "US", "S2", "single site llc", "singlesitellc", "", "othertown"),
        ("n2", "US", "S2", "single site llc", "singlesitellc", "9 ninth st sometown", "sometown"),
    ]))
    assert _group_of(out, "n1") != _group_of(out, "n2")


def test_large_identical_address_group_skips_fuzzy_link():
    # A group of identical addr_clean bigger than KEY_MAX must skip the
    # fuzzy name link entirely; with distinct names/name_keys/cities, every
    # record stays its own singleton group.
    n = KEY_MAX + 5
    rows = [
        (f"g{k}", "US", "S2", f"unique biz name {k}", f"uniquebizname{k}",
         "999 shared road bigcity", f"city{k}")
        for k in range(n)
    ]
    out = sibling_groups(_frame(rows))
    sizes = out.set_index("entity_id")["sib_group_size"]
    assert (sizes == 1).all()
    assert out["sib_group_id"].nunique() == n


def test_output_columns_and_determinism():
    frame = _frame([
        ("z2", "US", "S2", "porternall com", "porternall",
         "3220 gale street indianapolis in", "indianapolis"),
        ("z1", "US", "S2", "porter and nall llc", "porternall", "", ""),
    ])
    out1 = sibling_groups(frame)
    out2 = sibling_groups(frame.iloc[::-1].reset_index(drop=True))
    assert list(out1.columns) == ["entity_id", "sib_group_id", "sib_group_size", "sib_best_addr"]
    assert out1["sib_group_id"].dtype == "int64"
    assert out1["sib_group_size"].dtype == "int32"
    pd.testing.assert_frame_equal(
        out1.sort_values("entity_id").reset_index(drop=True),
        out2.sort_values("entity_id").reset_index(drop=True),
    )
