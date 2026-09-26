import numpy as np
import pandas as pd
import pytest

import config
import splits


def _make_synthetic(n=10000, seed=0):
    rng = np.random.default_rng(seed)
    countries = rng.choice(["US", "India", "Other"], size=n, p=[0.5, 0.3, 0.2])
    entity_ids = [f"S1-{i}" for i in range(n)]
    counts = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 10], size=n)
    s1 = pd.DataFrame({"entity_id": entity_ids, "country": countries})
    match_counts = pd.Series(counts, index=entity_ids)
    return s1, match_counts


def test_holdout_size_within_tolerance():
    s1, match_counts = _make_synthetic()
    holdout = splits.make_holdout(s1, match_counts, frac=0.15, seed=42)
    n = len(s1)
    expected = 0.15 * n
    assert abs(len(holdout) - expected) <= 0.005 * n


def test_holdout_country_share_within_tolerance():
    s1, match_counts = _make_synthetic()
    holdout = splits.make_holdout(s1, match_counts, frac=0.15, seed=42)
    overall_share = s1["country"].value_counts(normalize=True)
    holdout_mask = s1["entity_id"].isin(holdout)
    holdout_share = s1.loc[holdout_mask, "country"].value_counts(normalize=True)
    for country in overall_share.index:
        diff = abs(holdout_share.get(country, 0.0) - overall_share[country])
        assert diff <= 0.01, f"{country}: {diff}"


def test_holdout_deterministic_same_seed():
    s1, match_counts = _make_synthetic()
    h1 = splits.make_holdout(s1, match_counts, frac=0.15, seed=42)
    h2 = splits.make_holdout(s1, match_counts, frac=0.15, seed=42)
    assert h1 == h2


def test_transfer_split_disjoint_and_pure():
    s1, _ = _make_synthetic()
    train_ids, eval_ids = splits.transfer_split(s1, "US", "India")
    assert train_ids.isdisjoint(eval_ids)
    us_ids = set(s1.loc[s1["country"] == "US", "entity_id"])
    india_ids = set(s1.loc[s1["country"] == "India", "entity_id"])
    assert train_ids == us_ids
    assert eval_ids == india_ids


def test_save_and_load_split_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    ids = {"S1-3", "S1-1", "S1-2"}
    splits.save_split(ids, "myname")
    out_path = tmp_path / "splits" / "myname.txt"
    assert out_path.is_file()
    content = out_path.read_bytes()
    assert content == b"S1-1\nS1-2\nS1-3\n"
    loaded = splits.load_split("myname")
    assert loaded == ids


def test_make_holdout_folds_tiny_stratum_without_crashing():
    # A country with a tiny stratum (only 1 member in the "6+" bucket) that is
    # too small for train_test_split's stratify on its own; make_holdout must
    # fold it into a neighbouring bucket rather than crashing.
    rng = np.random.default_rng(1)
    n_main = 200
    countries = ["Tiny"] * n_main
    entity_ids = [f"S1-{i}" for i in range(n_main)]
    # Most records in bucket 0 or 1, exactly one lone record in bucket "6+".
    counts = list(rng.choice([0, 1], size=n_main - 1)) + [9]
    s1 = pd.DataFrame({"entity_id": entity_ids, "country": countries})
    match_counts = pd.Series(counts, index=entity_ids)

    holdout = splits.make_holdout(s1, match_counts, frac=0.15, seed=42)
    assert isinstance(holdout, set)
    assert holdout <= set(entity_ids)
    n = len(s1)
    assert abs(len(holdout) - 0.15 * n) <= 0.05 * n
