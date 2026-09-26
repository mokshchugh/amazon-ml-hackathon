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
