"""Holdout and country-transfer splits for entity resolution (SPEC §6 step 5).

The holdout is the team's only honest score (test labels are hidden), so the
split must be deterministic and stratified by country x match-count bucket.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config

BUCKET_ORDER = ["0", "1", "2-3", "4-5", "6+"]


def _bucket(count: int) -> str:
    """Map a match count to one of the fixed buckets: 0, 1, 2-3, 4-5, 6+."""
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    return "6+"


def _strata_labels(countries: pd.Series, buckets: pd.Series) -> list[str]:
    """Build country x bucket strata labels, folding tiny strata into a
    neighbouring bucket of the same country so every stratum has >= 2 members
    (sklearn's train_test_split requires this when stratifying).

    Buckets are folded in BUCKET_ORDER (0, 1, 2-3, 4-5, 6+): adjacent buckets
    within a country are merged, in order, until each merged group reaches at
    least 2 members. A leftover group that never reaches 2 is merged into the
    previous group (or, if it is the first group, left as its own group --
    only possible if the whole country has fewer than 2 records).
    """
    tmp = pd.DataFrame({"country": countries.values, "bucket": buckets.values})
    full_map: dict[tuple, str] = {}

    for country, group in tmp.groupby("country"):
        counts = group["bucket"].value_counts()
        assigned: dict[str, int] = {}
        current_group: list[str] = []
        current_size = 0
        group_id = 0
        for b in BUCKET_ORDER:
            c = counts.get(b, 0)
            if c == 0:
                continue
            current_group.append(b)
            current_size += c
            if current_size >= 2:
                for bb in current_group:
                    assigned[bb] = group_id
                group_id += 1
                current_group = []
                current_size = 0
        if current_group:
            target = group_id - 1 if group_id > 0 else 0
            for bb in current_group:
                assigned[bb] = target
        for b, gid in assigned.items():
            full_map[(country, b)] = f"{country}::{gid}"

    return [full_map[(c, b)] for c, b in zip(countries.values, buckets.values)]


def make_holdout(
    s1: pd.DataFrame,
    match_counts: pd.Series,
    frac: float = 0.15,
    seed: int = config.SEED,
) -> set[str]:
    """Select a stratified holdout of S1 entity_ids.

    Stratifies by country x match-count bucket (0, 1, 2-3, 4-5, 6+), folding
    strata that are too small for sklearn's stratify into a neighbouring
    bucket of the same country.
    """
    entity_ids = s1["entity_id"].reset_index(drop=True)
    countries = s1["country"].reset_index(drop=True)
    counts = match_counts.reindex(entity_ids).fillna(0).astype(int).reset_index(drop=True)
    buckets = counts.map(_bucket)

    strata = _strata_labels(countries, buckets)

    idx = list(range(len(entity_ids)))
    _train_idx, holdout_idx = train_test_split(
        idx, test_size=frac, random_state=seed, stratify=strata
    )
    return {entity_ids.iloc[i] for i in holdout_idx}


def transfer_split(
    s1: pd.DataFrame, train_country: str, eval_country: str
) -> tuple[set[str], set[str]]:
    """Return (train_ids, eval_ids): S1 ids for ``train_country`` and
    ``eval_country`` respectively. The two sets are disjoint and each is
    pure to its own country.
    """
    train_ids = set(s1.loc[s1["country"] == train_country, "entity_id"])
    eval_ids = set(s1.loc[s1["country"] == eval_country, "entity_id"])
    return train_ids, eval_ids


def _split_path(name: str) -> Path:
    return config.CACHE_DIR / "splits" / f"{name}.txt"


def save_split(ids, name: str) -> None:
    """Save a set of entity_ids, one per line, sorted, UTF-8, LF endings."""
    path = _split_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for entity_id in sorted(ids):
            f.write(f"{entity_id}\n")


def load_split(name: str) -> set[str]:
    """Load a saved split back into a set of entity_ids."""
    path = _split_path(name)
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
