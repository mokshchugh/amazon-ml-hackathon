import pytest
from evaluate import f05, macro_f05, pair_recall, reduction_ratio, report


def test_readme_example():
    assert f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) == pytest.approx(0.7142857, 1e-6)


def test_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0


def test_macro_counts_missing_as_empty():
    truth = {"a": {"x"}, "b": set()}
    assert macro_f05({"a": {"x"}}, truth, ["a", "b"]) == 1.0


def test_report_by_country():
    r = report({"a": {"x"}}, {"a": {"x"}, "b": {"y"}}, {"a": "US", "b": "France"})
    assert r["by_country"] == {"US": 1.0, "France": 0.0} and r["overall"] == 0.5


def test_pair_recall():
    # Test case 1: cands {"a": {"x","y"}, "b": {"z"}}, truth {"a": {"x","w"}, "b": {"z"}, "c": set()} → 2/3
    cands = {"a": {"x", "y"}, "b": {"z"}}
    truth = {"a": {"x", "w"}, "b": {"z"}, "c": set()}
    # True pairs: a-x, a-w (from a), b-z (from b), nothing from c = 3 total
    # Found pairs: a-x (match), a-y (no match), b-z (match) = 2 matches
    assert pair_recall(cands, truth) == pytest.approx(2.0 / 3.0, 1e-6)

    # Test case 2: truth with no pairs at all → 1.0
    assert pair_recall({"a": {"x"}}, {}) == 1.0


def test_reduction_ratio():
    assert reduction_ratio(10, 5, 4) == 0.5
    assert reduction_ratio(0, 5, 4) == 1.0


def test_report_pooled_scope():
    # s1_country has only {"a": "US"}, but pred and truth have keys "a" and "z"
    # record "z" should be excluded from pooled metrics
    s1_country = {"a": "US"}
    pred = {"a": {"x"}, "z": {"q"}}
    truth = {"a": {"x"}, "z": {"r"}}
    r = report(pred, truth, s1_country)
    # Only considering "a": pred={"x"}, truth={"x"} → perfect overlap
    # pair_precision = 1 pair found / 1 pair predicted = 1.0
    # pair_recall = 1 pair found / 1 pair in truth = 1.0
    assert r["pair_precision"] == 1.0
    assert r["pair_recall"] == 1.0
