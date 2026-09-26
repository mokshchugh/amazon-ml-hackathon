import csv
import sys
from pathlib import Path

import config
import io_utils

# Make the organisers' validator importable.
sys.path.insert(0, str(config.REPO_ROOT / "student_resource" / "utils"))
from validate_submission import validate_id_list_file, MATCHING_HEADER  # noqa: E402


def test_read_source_keeps_null_text(tmp_path):
    path = tmp_path / "s2.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tnull\t<NULL>\tIndia\n",
        encoding="utf-8",
        newline="\n",
    )
    df = io_utils.read_source(path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["business_name"] == "null"
    assert row["business_address"] == "<NULL>"
    assert row["country"] == "India"
    assert row["entity_id"] == "S2-1"
    assert row["source"] == "S2"
    # dtype must be string[pyarrow] (pyarrow-backed StringDtype), not object/NaN-producing
    assert df["business_name"].dtype.storage == "pyarrow"
    assert not df["business_name"].isna().any()


def test_read_ground_truth_long(tmp_path):
    path = tmp_path / "gt.tsv"
    path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-5,S3-9\n"
        "S1-2\t\n",
        encoding="utf-8",
        newline="\n",
    )
    df = io_utils.read_ground_truth(path)
    rows_s1_1 = df[df["s1_id"] == "S1-1"]
    assert len(rows_s1_1) == 2
    assert set(rows_s1_1["s23_id"]) == {"S2-5", "S3-9"}
    rows_s1_2 = df[df["s1_id"] == "S1-2"]
    assert len(rows_s1_2) == 0
    assert len(df) == 2


def test_write_id_lists_empty_rows(tmp_path):
    out_path = tmp_path / "matching_results.tsv"
    s1_ids = ["S1-1", "S1-2"]
    lists = {
        "S1-1": ["S2-5", "S3-9", "S2-5"],  # duplicate S2-5, first-seen order kept
        # S1-2 intentionally absent -> no list
    }
    io_utils.write_id_lists(out_path, "matched_entity_ids", s1_ids, lists)

    raw = out_path.read_bytes()
    assert b"\r\n" not in raw  # LF line endings only
    text = raw.decode("utf-8")
    lines = text.split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert "S1-1\tS2-5,S3-9" in lines
    assert "S1-2\t" in lines

    # Round-trip through the organisers' validator.
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tA\tAddr\tUS\n"
        "S1-2\tB\tAddr\tUS\n",
        encoding="utf-8",
        newline="\n",
    )
    required = {"S1-1", "S1-2"}
    errors = []
    mapping = validate_id_list_file(
        str(out_path), MATCHING_HEADER, "matched_entity_ids", required, None, errors
    )
    assert errors == []
    assert mapping is not None
