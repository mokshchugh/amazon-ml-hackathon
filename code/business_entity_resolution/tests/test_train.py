"""Tests for src/train.py: LightGBM training, calibration, save/load."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import train
from features import FEATURE_COLUMNS

# Small, fast params for tests.
TEST_PARAMS = dict(train.LGB_PARAMS)
TEST_PARAMS.update(num_leaves=15, min_data_in_leaf=10, verbose=-1, verbosity=-1)


def _synthetic(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    n_groups = n // 5
    groups = rng.integers(0, n_groups, size=n)

    data = {c: rng.normal(size=n).astype(np.float32) for c in FEATURE_COLUMNS}
    # signal in two features
    sig1 = data[FEATURE_COLUMNS[0]]
    sig2 = data[FEATURE_COLUMNS[1]]
    logit = 2.5 * sig1 - 2.0 * sig2
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(np.int64)

    X = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    return X, y, groups.astype(np.int64)


@pytest.fixture(scope="module")
def synthetic_data():
    return _synthetic()


@pytest.fixture(scope="module")
def trained(synthetic_data):
    X, y, groups = synthetic_data
    booster, oof = train.train_model(X, y, groups, max_rounds=200, params=TEST_PARAMS)
    return booster, oof


def test_oof_auc_above_threshold(synthetic_data, trained):
    _, y, _ = synthetic_data
    _, oof = trained
    auc = roc_auc_score(y, oof)
    assert auc > 0.9


def test_calibrated_probabilities_in_bounds_and_monotone(synthetic_data, trained):
    X, y, _ = synthetic_data
    booster, oof = trained
    cal = train.fit_calibrator(oof, y)
    assert isinstance(cal, IsotonicRegression)

    proba = train.predict_proba(booster, cal, X)
    assert proba.dtype == np.float32
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    # monotone in the raw score: sort by raw booster score, calibrated proba should be non-decreasing
    raw = booster.predict(X[FEATURE_COLUMNS].values, num_iteration=booster.best_iteration)
    order = np.argsort(raw)
    sorted_proba = proba[order]
    assert np.all(np.diff(sorted_proba) >= -1e-9)


def test_save_load_roundtrip(tmp_path, monkeypatch, synthetic_data, trained):
    monkeypatch.setattr(train.config, "MODELS_DIR", tmp_path)
    X, y, _ = synthetic_data
    booster, oof = trained
    cal = train.fit_calibrator(oof, y)

    tag = "test_model"
    train.save(booster, cal, tag)

    assert (tmp_path / tag / "lgbm.txt").exists()
    assert (tmp_path / tag / "isotonic.pkl").exists()
    assert (tmp_path / tag / "features.json").exists()

    with open(tmp_path / tag / "features.json") as f:
        saved_cols = json.load(f)
    assert saved_cols == FEATURE_COLUMNS

    booster2, cal2 = train.load(tag)

    proba_before = train.predict_proba(booster, cal, X)
    proba_after = train.predict_proba(booster2, cal2, X)
    np.testing.assert_array_equal(proba_before, proba_after)


def test_feature_order_mismatch_raises(tmp_path, monkeypatch, synthetic_data, trained):
    """predict_proba/load should guard against a column-order mismatch."""
    monkeypatch.setattr(train.config, "MODELS_DIR", tmp_path)
    X, y, _ = synthetic_data
    booster, oof = trained
    cal = train.fit_calibrator(oof, y)

    tag = "test_model_mismatch"
    train.save(booster, cal, tag)

    booster2, cal2 = train.load(tag)

    bad_cols = list(FEATURE_COLUMNS[::-1])
    X_bad = X[bad_cols]
    with pytest.raises(ValueError):
        train.predict_proba(booster2, cal2, X_bad)


def test_same_seed_deterministic(synthetic_data):
    X, y, groups = synthetic_data
    booster1, oof1 = train.train_model(X, y, groups, max_rounds=200, params=TEST_PARAMS)
    booster2, oof2 = train.train_model(X, y, groups, max_rounds=200, params=TEST_PARAMS)

    proba1 = booster1.predict(X[FEATURE_COLUMNS].values, num_iteration=booster1.best_iteration)
    proba2 = booster2.predict(X[FEATURE_COLUMNS].values, num_iteration=booster2.best_iteration)

    np.testing.assert_array_equal(oof1, oof2)
    np.testing.assert_array_equal(proba1, proba2)
