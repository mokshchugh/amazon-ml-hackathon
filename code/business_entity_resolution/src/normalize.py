"""Business-name normalization (SPEC step 2).

Cleans raw ``business_name`` strings into a set of comparable forms:
``name_clean`` (word order kept), ``name_sorted`` (words sorted),
``name_key`` (letters/digits only, no spaces, no connector words),
plus the extracted ``legal`` suffix and any ``alt_name`` split out of a
"doing business as" / "dba" name.

Indic transliteration (SPEC step 3) is a later task; this module only
flags ``was_indic`` so a transliteration pass can be inserted between
substeps 4 and 5 without restructuring this pipeline.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from lexicons import HONORIFICS, LEGAL_CANON

_INDIC_RANGE = re.compile(r"[ऀ-෿]")

# Substep 2: junk tokens/markers stripped outright.
_JUNK_LITERALS = ("<<", "--", "##")
_JUNK_WORD = re.compile(r"\bnull\b")
_JUNK_PUNCT = re.compile(r"[\[\]\(\)<>]")
_WHITESPACE = re.compile(r"\s+")

# Substep 3: strip a leading handle marker and a trailing domain suffix.
_LEADING_HANDLE = re.compile(r"^[@#]+")
_DOMAIN_SUFFIX = re.compile(r"\.(com|net|org|co|in|io|biz|info)$")

# Substep 5: split a "doing business as" / "dba" name into two.
_DBA_SPLIT = re.compile(r"\bdoing business as\b|\bdba\b")

# Substep 6.
_AMPERSAND = re.compile(r"&")

# Substep 9: name_key keeps only letters/digits and drops "and".
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def clean_text(s: str) -> str:
    """NFKC/NFKD-fold, lowercase, strip junk tokens, collapse whitespace.

    This is SPEC step 2, substeps 1-2 only: it does not touch domains,
    digits-as-letters, dba splitting, "&", legal suffixes or honorifics.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for junk in _JUNK_LITERALS:
        s = s.replace(junk, " ")
    s = _JUNK_WORD.sub(" ", s)
    s = _JUNK_PUNCT.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _fix_leetspeak_digits(s: str) -> str:
    """Substep 4: 1->l, 0->o, 3->e, only when letters flank the digit."""
    s = re.sub(r"(?<=[a-z])1(?=[a-z])", "l", s)
    s = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", s)
    s = re.sub(r"(?<=[a-z])3(?=[a-z])", "e", s)
    return s


def _strip_domain_and_handle(s: str) -> str:
    """Substep 3: strip a leading @/# handle marker and a trailing TLD."""
    s = _LEADING_HANDLE.sub("", s)
    s = _DOMAIN_SUFFIX.sub("", s)
    return s


def _filter_words(words: list[str]) -> tuple[list[str], list[str]]:
    """Substeps 7-8: pull legal suffixes out, drop honorifics.

    Returns (kept_words, canonical_legal_words).
    """
    kept: list[str] = []
    legal: list[str] = []
    for word in words:
        if word in HONORIFICS:
            continue
        canon = LEGAL_CANON.get(word)
        if canon is not None:
            legal.append(canon)
        else:
            kept.append(word)
    return kept, legal


def _build_name_key(clean_words: list[str]) -> str:
    """Substep 9 + ruling R3: letters/digits of name_clean, "and" dropped."""
    parts = [w for w in clean_words if w != "and"]
    return "".join(_NON_ALNUM.sub("", w) for w in parts)


def normalize_name(s: str) -> dict:
    """Run SPEC step 2 (substeps 1-9) on one raw business name."""
    raw = s if s is not None else ""
    was_indic = bool(_INDIC_RANGE.search(raw))

    text = clean_text(raw)
    text = _strip_domain_and_handle(text)
    text = _fix_leetspeak_digits(text)

    match = _DBA_SPLIT.search(text)
    if match:
        primary_text = text[: match.start()].strip()
        alt_text = text[match.end():].strip()
    else:
        primary_text = text
        alt_text = ""

    primary_text = _AMPERSAND.sub(" and ", primary_text)
    primary_text = _WHITESPACE.sub(" ", primary_text).strip()
    clean_words, legal_words = _filter_words(primary_text.split())

    alt_text = _AMPERSAND.sub(" and ", alt_text)
    alt_text = _WHITESPACE.sub(" ", alt_text).strip()
    alt_words, _ = _filter_words(alt_text.split())

    name_clean = " ".join(clean_words)
    name_sorted = " ".join(sorted(clean_words))
    name_key = _build_name_key(clean_words)
    legal = " ".join(sorted(set(legal_words)))
    alt_name = " ".join(alt_words)

    return {
        "name_clean": name_clean,
        "name_sorted": name_sorted,
        "name_key": name_key,
        "legal": legal,
        "alt_name": alt_name,
        "was_indic": was_indic,
    }


_COLUMNS = ["name_clean", "name_sorted", "name_key", "legal", "alt_name", "was_indic"]


def normalize_names(names: pd.Series) -> pd.DataFrame:
    """Vectorized ``normalize_name`` over a pandas string Series.

    Maps over unique values only (real input has ~5M rows with heavy
    repetition), then joins the results back so the result is aligned to
    ``names.index``.
    """
    values = names.tolist()
    unique_values = list(dict.fromkeys(values))
    lookup = {v: normalize_name(v) for v in unique_values}

    out = pd.DataFrame(
        [lookup[v] for v in values],
        index=names.index,
        columns=_COLUMNS,
    )
    for col in ("name_clean", "name_sorted", "name_key", "legal", "alt_name"):
        out[col] = out[col].astype("string[pyarrow]")
    out["was_indic"] = out["was_indic"].astype("bool")
    return out
