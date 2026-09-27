"""LightGBM training with GroupKFold cross-validation and isotonic calibration.

SPEC section 6 step 9.
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

import config
from features import FEATURE_COLUMNS

LGB_PARAMS: dict = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    seed=config.SEED,
    deterministic=True,
    force_row_wise=True,
    num_threads=16,
    verbose=-1,
    verbosity=-1,
)

N_FOLDS = 3
EARLY_STOPPING_ROUNDS = 100


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    max_rounds: int = 3000,
    params: dict | None = None,
) -> tuple[lgb.Booster, np.ndarray]:
    """3-fold GroupKFold by s1_id with early stopping; final model trained on all
    rows for the mean best iteration (rounded). Returns (booster, oof_scores)."""
    if params is None:
        params = LGB_PARAMS
    params = dict(params)

    X_vals = X[FEATURE_COLUMNS].values
    y = np.asarray(y)
    n = len(X_vals)
    oof = np.zeros(n, dtype=np.float64)
    best_iterations = []

    gkf = GroupKFold(n_splits=N_FOLDS)
    for train_idx, valid_idx in gkf.split(X_vals, y, groups):
        dtrain = lgb.Dataset(
            X_vals[train_idx], label=y[train_idx], feature_name=FEATURE_COLUMNS,
            free_raw_data=True,
        )
        dvalid = lgb.Dataset(
            X_vals[valid_idx], label=y[valid_idx], feature_name=FEATURE_COLUMNS,
            free_raw_data=True, reference=dtrain,
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[valid_idx] = booster.predict(X_vals[valid_idx], num_iteration=booster.best_iteration)
        best_iterations.append(booster.best_iteration)

    final_rounds = int(round(float(np.mean(best_iterations))))
    final_rounds = max(final_rounds, 1)

    dall = lgb.Dataset(X_vals, label=y, feature_name=FEATURE_COLUMNS, free_raw_data=True)
    final_booster = lgb.train(params, dall, num_boost_round=final_rounds)

    return final_booster, oof.astype(np.float32)


def fit_calibrator(oof: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic calibrator mapping raw OOF probabilities to calibrated ones."""
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(oof, y)
    return cal


def _check_feature_order(cols: list[str]) -> None:
    if list(cols) != list(FEATURE_COLUMNS):
        raise ValueError(
            "Feature column order mismatch: expected FEATURE_COLUMNS order, got a different order."
        )


def predict_proba(booster: lgb.Booster, cal: IsotonicRegression, X: pd.DataFrame) -> np.ndarray:
    """Predict calibrated probabilities for X (must have FEATURE_COLUMNS order)."""
    _check_feature_order(list(X.columns))
    raw = booster.predict(X[FEATURE_COLUMNS].values, num_iteration=booster.best_iteration)
    calibrated = cal.predict(raw)
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _tag_dir(tag: str) -> Path:
    return Path(config.MODELS_DIR) / tag


def save(booster: lgb.Booster, cal: IsotonicRegression, tag: str) -> None:
    """Save booster, calibrator, and feature-column order under MODELS_DIR/tag/."""
    out_dir = _tag_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(out_dir / "lgbm.txt"))

    with open(out_dir / "isotonic.pkl", "wb") as f:
        pickle.dump(cal, f)

    with open(out_dir / "features.json", "w") as f:
        json.dump(list(FEATURE_COLUMNS), f)


def load(tag: str) -> tuple[lgb.Booster, IsotonicRegression]:
    """Load booster and calibrator from MODELS_DIR/tag/, asserting feature order matches."""
    out_dir = _tag_dir(tag)

    with open(out_dir / "features.json") as f:
        saved_cols = json.load(f)
    _check_feature_order(saved_cols)

    booster = lgb.Booster(model_file=str(out_dir / "lgbm.txt"))

    with open(out_dir / "isotonic.pkl", "rb") as f:
        cal = pickle.load(f)

    return booster, cal
