import numpy as np
import pandas as pd
import pytest

import config
import io_utils
import splits


def _write_synthetic_cache(cache_dir, n=60, seed=3):
    """Write a tiny train_source1.parquet + gt_pairs.parquet under cache_dir."""
    rng = np.random.default_rng(seed)
    countries = rng.choice(["US", "India"], size=n, p=[0.5, 0.5])
    entity_ids = [f"S1-{i}" for i in range(n)]
    s1 = pd.DataFrame(
        {
            "entity_id": pd.array(entity_ids, dtype="string[pyarrow]"),
            "business_name": pd.array(["x"] * n, dtype="string[pyarrow]"),
            "business_address": pd.array(["y"] * n, dtype="string[pyarrow]"),
            "country": pd.array(countries, dtype="string[pyarrow]"),
            "source": pd.array(["S1"] * n, dtype="string[pyarrow]"),
        }
    )
    s1.to_parquet(cache_dir / "train_source1.parquet", index=False)

    # Give roughly a third of records 1-3 matches each, so match_counts has
    # some non-zero values and zero-match records too.
    s1_out, s23_out = [], []
    for i, entity_id in enumerate(entity_ids):
        n_matches = rng.integers(0, 4)
        for j in range(n_matches):
            s1_out.append(entity_id)
            s23_out.append(f"S2-{i}-{j}")
    gt = pd.DataFrame(
        {
            "s1_id": pd.array(s1_out, dtype="string[pyarrow]"),
            "s23_id": pd.array(s23_out, dtype="string[pyarrow]"),
        }
    )
    gt.to_parquet(cache_dir / "gt_pairs.parquet", index=False)
    return s1


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


def test_holdout_invariant_to_row_order():
    s1, match_counts = _make_synthetic(n=300, seed=7)
    holdout_a = splits.make_holdout(s1, match_counts, frac=0.15, seed=42)

    shuffled = s1.sample(frac=1.0, random_state=123).reset_index(drop=True)
    holdout_b = splits.make_holdout(shuffled, match_counts, frac=0.15, seed=42)

    assert holdout_a == holdout_b


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


EXPECTED_SPLIT_NAMES = [
    "holdout",
    "transfer_us_india_train",
    "transfer_us_india_eval",
    "transfer_india_us_train",
    "transfer_india_us_eval",
]


def test_ensure_splits_creates_five_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    s1 = _write_synthetic_cache(tmp_path)

    sizes = splits.ensure_splits()

    assert set(sizes.keys()) == set(EXPECTED_SPLIT_NAMES)
    for name in EXPECTED_SPLIT_NAMES:
        path = tmp_path / "splits" / f"{name}.txt"
        assert path.is_file()
        assert sizes[name] == len(splits.load_split(name))
    assert sizes["transfer_us_india_train"] == (s1["country"] == "US").sum()
    assert sizes["transfer_us_india_eval"] == (s1["country"] == "India").sum()
    assert sizes["transfer_india_us_train"] == (s1["country"] == "India").sum()
    assert sizes["transfer_india_us_eval"] == (s1["country"] == "US").sum()


def test_ensure_splits_no_force_leaves_existing_files_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    _write_synthetic_cache(tmp_path)

    sizes_first = splits.ensure_splits()
    before = {
        name: (tmp_path / "splits" / f"{name}.txt").read_bytes()
        for name in EXPECTED_SPLIT_NAMES
    }

    # Corrupt the source caches: if ensure_splits recomputed, it would see
    # different data. It must not touch the files since force is False.
    (tmp_path / "train_source1.parquet").unlink()

    sizes_second = splits.ensure_splits(force=False)
    after = {
        name: (tmp_path / "splits" / f"{name}.txt").read_bytes()
        for name in EXPECTED_SPLIT_NAMES
    }

    assert sizes_first == sizes_second
    assert before == after


def test_ensure_splits_force_true_rewrites_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    _write_synthetic_cache(tmp_path)

    sizes_first = splits.ensure_splits()
    before = {
        name: (tmp_path / "splits" / f"{name}.txt").read_bytes()
        for name in EXPECTED_SPLIT_NAMES
    }

    sizes_second = splits.ensure_splits(force=True)
    after = {
        name: (tmp_path / "splits" / f"{name}.txt").read_bytes()
        for name in EXPECTED_SPLIT_NAMES
    }

    assert sizes_first == sizes_second
    assert before == after
