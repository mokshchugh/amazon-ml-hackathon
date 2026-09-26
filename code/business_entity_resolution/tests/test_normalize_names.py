"""Tests for business-name normalization (SPEC step 2)."""
import pandas as pd
import pytest

from normalize import normalize_name, normalize_names, clean_text


@pytest.mark.parametrize("raw,clean,legal", [
    ("PORTER & NALL [LLC]", "porter and nall", "llc"),
    ("porternall.com", "porternall", ""),
    ("@wheelmanagement", "wheelmanagement", ""),
    ("Intelligence Go1den LLC", "intelligence golden", "llc"),
    ("allied sígnature audio installation inc", "allied signature audio installation", "inc"),
    ("M/s Quality Garments Pvt", "quality garments", "pvt"),
    ("Lakshmi Consultancy (Private)", "lakshmi consultancy", "pvt"),
    ("Wheel Management Pvt Ltd", "wheel management", "ltd pvt"),
    ("<< Team Ecole", "team ecole", ""),
    ("ZNB Club SARL", "znb club", "sarl"),
])
def test_normalize_name(raw, clean, legal):
    out = normalize_name(raw)
    assert out["name_clean"] == clean and out["legal"] == legal


def test_dba_split():
    out = normalize_name("Synpyra doing business as Golden Intelligence Holdings")
    assert out["name_clean"] == "synpyra" and out["alt_name"] == "golden intelligence holdings"


def test_sorted_and_key():
    out = normalize_name("Empire Inc Translational  Interstate")
    assert out["name_sorted"] == "empire interstate translational" and out["name_key"] == "empiretranslationalinterstate"


def test_empty_and_null_names():
    for raw in ["", "null", "<NULL>", "--"]:
        assert normalize_name(raw)["name_clean"] == ""


def test_name_key_glued_word_equivalence():
    # R3: name_key drops the connector token "and" and all spaces, so
    # "Porter & Nall" and "porternall.com" produce the same key.
    assert (
        normalize_name("Porter & Nall")["name_key"]
        == normalize_name("porternall.com")["name_key"]
        == "porternall"
    )
