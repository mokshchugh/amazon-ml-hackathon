"""Tests for Indic-script -> Latin transliteration (SPEC step 3, Task 6)."""
import pandas as pd
import pytest

from normalize import learn_token_table, save_token_table, load_token_table, to_latin


def _toy_pairs():
    # 6 identical toy pairs so that the token pair counts reach min_count=5.
    rows = [
        ("लक्ष्मी कंसल्टेंसी प्राइवेट लिमिटेड", "Lakshmi Consultancy Private Limited")
        for _ in range(6)
    ]
    return pd.DataFrame(rows, columns=["s23_name", "s1_name"])


def test_learn_table():
    table = learn_token_table(_toy_pairs())
    assert table["लक्ष्मी"] == "lakshmi"
    assert table["कंसल्टेंसी"] == "consultancy"


def test_learn_table_min_count_excludes_rare_tokens():
    # Only 1 occurrence (< default min_count=5) -> absent from the table.
    pairs = pd.DataFrame(
        [("अनोखा नाम", "Unique Name")],
        columns=["s23_name", "s1_name"],
    )
    table = learn_token_table(pairs)
    assert "अनोखा" not in table
    assert "नाम" not in table


def test_mixed_script():
    assert to_latin("Digital सिस्टम्स", {"सिस्टम्स": "systems"}) == "digital systems"


def test_fallback_ascii():
    out = to_latin("क्रिएटिव", {})
    assert out != ""
    assert out.isascii()


def test_save_and_load_token_table(tmp_path):
    table = {"लक्ष्मी": "lakshmi", "कंसल्टेंसी": "consultancy"}
    path = tmp_path / "indic_tokens.json"
    save_token_table(table, path)
    loaded = load_token_table(path)
    assert loaded == table
