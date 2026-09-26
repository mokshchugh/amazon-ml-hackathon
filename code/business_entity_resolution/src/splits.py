"""Holdout and country-transfer splits for entity resolution (SPEC §6 step 5).

The holdout is the team's only honest score (test labels are hidden), so the
split must be deterministic and stratified by country x match-count bucket.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import io_utils

BUCKET_ORDER = ["0", "1", "2-3", "4-5", "6+"]

SPLIT_NAMES = [
    "holdout",
    "transfer_us_india_train",
    "transfer_us_india_eval",
    "transfer_india_us_train",
    "transfer_india_us_eval",
]


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

    Sorts by entity_id first so the result is independent of ``s1``'s row
    order (sklearn's stratified shuffle draws a permutation of *positions*
    within each class, so the same input rows in a different order would
    otherwise select different entities).
    """
    s1_sorted = s1.sort_values("entity_id", kind="stable").reset_index(drop=True)
    entity_ids = s1_sorted["entity_id"]
    countries = s1_sorted["country"]
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


def ensure_splits(force: bool = False) -> dict[str, int]:
    """Create (or reuse) the real-data holdout and transfer splits.

    Loads the train S1 cache and ``gt_pairs.parquet`` from ``config.CACHE_DIR``,
    builds match counts (0 for S1 records with no match), and creates and
    saves "holdout", "transfer_us_india_train"/"_eval" and
    "transfer_india_us_train"/"_eval".

    If all five split files already exist and ``force`` is False, nothing is
    read or (re)computed -- the existing files are left untouched and only
    their sizes are returned. With ``force=True``, or when any file is
    missing, all five are (re)computed and saved.

    Returns a dict mapping split name -> number of entity_ids in that split.
    """
    paths = {name: _split_path(name) for name in SPLIT_NAMES}
    if not force and all(p.is_file() for p in paths.values()):
        return {name: len(load_split(name)) for name in SPLIT_NAMES}

    s1 = io_utils.load("train", "source1")
    gt = pd.read_parquet(config.CACHE_DIR / "gt_pairs.parquet")

    match_counts = gt.groupby("s1_id").size()
    match_counts = match_counts.reindex(s1["entity_id"]).fillna(0).astype(int)
    match_counts.index = s1["entity_id"].values

    holdout = make_holdout(s1, match_counts, frac=0.15, seed=config.SEED)
    save_split(holdout, "holdout")

    us_india_train, us_india_eval = transfer_split(s1, "US", "India")
    save_split(us_india_train, "transfer_us_india_train")
    save_split(us_india_eval, "transfer_us_india_eval")

    india_us_train, india_us_eval = transfer_split(s1, "India", "US")
    save_split(india_us_train, "transfer_india_us_train")
    save_split(india_us_eval, "transfer_india_us_eval")

    return {
        "holdout": len(holdout),
        "transfer_us_india_train": len(us_india_train),
        "transfer_us_india_eval": len(us_india_eval),
        "transfer_india_us_train": len(india_us_train),
        "transfer_india_us_eval": len(india_us_eval),
    }
