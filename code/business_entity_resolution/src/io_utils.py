"""TSV loading, parquet cache, and submission output writing.

Read options (binding, per Global Constraints):
    sep="\\t", encoding="utf-8", dtype="string[pyarrow]",
    quoting=csv.QUOTE_NONE, keep_default_na=False, na_filter=False.
Literal "null" / "<NULL>" text must stay text, never become NaN.

Output writing is done by hand (no pandas ``to_csv``): TAB-separated,
UTF-8, LF line endings, IDs joined by "," with no quoting, duplicates
removed keeping first-seen order.
"""
from __future__ import annotations

import csv
import gc
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

import config

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

READ_KWARGS = dict(
    sep="\t",
    encoding="utf-8",
    dtype="string[pyarrow]",
    quoting=csv.QUOTE_NONE,
    keep_default_na=False,
    na_filter=False,
)


def read_source(path: Path) -> pd.DataFrame:
    """Read a source TSV (train/test source1/2/3 file).

    Returns a DataFrame with columns entity_id, business_name,
    business_address, country (all string[pyarrow]) plus a derived
    ``source`` column ("S1"/"S2"/"S3") taken from the entity_id prefix.
    """
    df = pd.read_csv(path, usecols=SOURCE_COLUMNS, **READ_KWARGS)
    if len(df) == 0:
        df["source"] = pd.array([], dtype="string[pyarrow]")
    else:
        df["source"] = df["entity_id"].str.split("-", n=1).str[0]
    return df


def read_ground_truth(path: Path) -> pd.DataFrame:
    """Read the ground-truth TSV into long format.

    Input columns: source1_entity_id, matched_entity_ids (comma-joined,
    possibly empty). Output columns: s1_id, s23_id -- one row per match;
    singletons (empty matched_entity_ids) contribute no rows.
    """
    df = pd.read_csv(
        path,
        usecols=["source1_entity_id", "matched_entity_ids"],
        **READ_KWARGS,
    )
    s1 = df["source1_entity_id"].tolist()
    matched = df["matched_entity_ids"].tolist()

    s1_out = []
    s23_out = []
    for s1_id, matched_ids in zip(s1, matched):
        if not matched_ids:
            continue
        for s23_id in matched_ids.split(","):
            s1_out.append(s1_id)
            s23_out.append(s23_id)

    return pd.DataFrame(
        {
            "s1_id": pd.array(s1_out, dtype="string[pyarrow]"),
            "s23_id": pd.array(s23_out, dtype="string[pyarrow]"),
        }
    )


def _split_source_path(split: Literal["train", "test"], source: str) -> Path:
    filename = f"{split}_{source.lower()}.tsv"
    return config.DATA_DIR / split / filename


def build_cache(split: Literal["train", "test"]) -> None:
    """Read the raw TSVs for ``split`` and write parquet caches.

    Streams one source file at a time and frees it before reading the
    next, so peak memory stays bounded to a single file. Writes
    ``CACHE_DIR/{split}_{source}.parquet`` for source1/2/3, and for
    "train" also writes ``CACHE_DIR/gt_pairs.parquet``. Country
    filtering happens at load time, not here.
    """
    config.ensure_dirs()

    for source in ("source1", "source2", "source3"):
        path = _split_source_path(split, source)
        df = read_source(path)
        out_path = config.CACHE_DIR / f"{split}_{source}.parquet"
        df.to_parquet(out_path, index=False)
        del df
        gc.collect()

    if split == "train":
        gt_path = config.DATA_DIR / "train" / "train_ground_truth.tsv"
        gt_df = read_ground_truth(gt_path)
        gt_out_path = config.CACHE_DIR / "gt_pairs.parquet"
        gt_df.to_parquet(gt_out_path, index=False)
        del gt_df
        gc.collect()


def load(split: Literal["train", "test"], source: str, country: str | None = None) -> pd.DataFrame:
    """Load a cached parquet source file, optionally filtered by country."""
    path = config.CACHE_DIR / f"{split}_{source}.parquet"
    df = pd.read_parquet(path)
    if country is not None:
        df = df[df["country"] == country].reset_index(drop=True)
    return df


def write_id_lists(
    path: Path,
    value_col: Literal["matched_entity_ids", "candidate_entity_ids"],
    s1_ids: Sequence[str],
    lists: Mapping[str, Sequence[str]],
) -> None:
    """Write a submission-style id-list TSV.

    Every id in ``s1_ids`` gets exactly one row, even when it has no
    entries in ``lists`` (written as an empty trailing field, e.g.
    "S1-2\\t"). Duplicate IDs within a single record's list are removed,
    keeping first-seen order. Written by hand: TAB-separated, UTF-8,
    LF line endings, no quoting.
    """
    header = f"source1_entity_id\t{value_col}"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for s1_id in s1_ids:
            ids = lists.get(s1_id, [])
            seen = set()
            deduped = []
            for entity_id in ids:
                if entity_id not in seen:
                    seen.add(entity_id)
                    deduped.append(entity_id)
            f.write(f"{s1_id}\t{','.join(deduped)}\n")
