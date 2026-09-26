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

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Mapping

import pandas as pd
from indic_transliteration import sanscript

from lexicons import HONORIFICS, LEGAL_CANON

_INDIC_RANGE = re.compile(r"[ऀ-෿]")

# R15 fallback: Unicode block -> indic_transliteration sanscript scheme.
_SCRIPT_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]

_NON_ALNUM_ASCII = re.compile(r"[^a-z0-9]")

# Substep 2: junk tokens/markers stripped outright.
_JUNK_LITERALS = ("<<", "--", "##")
_JUNK_WORD = re.compile(r"\bnull\b")
_JUNK_PUNCT = re.compile(r"[\[\]\(\)<>]")
_WHITESPACE = re.compile(r"\s+")

# Substep 3: strip a leading handle marker and a trailing domain suffix.
_LEADING_HANDLE = re.compile(r"^[@#]+")
_DOMAIN_SUFFIX = re.compile(r"\.(com|net|org|co|in|io|biz|info)$")

# R13: the "m/s" honorific must be matched (and removed) before the
# keep-list punctuation filter runs, since that filter would otherwise
# split "m/s" into the two bare letters "m" and "s".
_MS_HONORIFIC = re.compile(r"\bm/s\b")

# R13: apostrophes are deleted outright (not replaced with a space), so
# "Hargrove's" -> "hargroves", not "hargrove s".
_APOSTROPHE = re.compile(r"['’]")

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


def _keep_letters_marks_digits(s: str) -> str:
    """Controller ruling R13: punctuation filter run after the domain/
    handle strip and after the "m/s" honorific is matched.

    Deletes apostrophes outright, keeps any character in Unicode
    category L*, M* or N* plus space and "&", and replaces every other
    character with a space before collapsing whitespace. Category M is
    kept (not just L/N) so Indic vowel signs and viramas survive this
    step for Task 6's transliteration.
    """
    s = _APOSTROPHE.sub("", s)
    kept = []
    for ch in s:
        if ch == " " or ch == "&" or unicodedata.category(ch)[0] in ("L", "M", "N"):
            kept.append(ch)
        else:
            kept.append(" ")
    s = "".join(kept)
    return _WHITESPACE.sub(" ", s).strip()


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
    if s is None or pd.isna(s):
        raw = ""
    else:
        raw = s
    was_indic = bool(_INDIC_RANGE.search(raw))

    text = clean_text(raw)
    text = _strip_domain_and_handle(text)
    text = _MS_HONORIFIC.sub(" ", text)
    text = _keep_letters_marks_digits(text)
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


def _latin_tokens(s: str) -> list[str]:
    """R14: lowercase + the R13 character-keep rule, then split on space."""
    return _keep_letters_marks_digits(s.lower()).split()


def _detect_script(token: str) -> str | None:
    """R15: find the sanscript scheme for the first Indic char's block."""
    for ch in token:
        cp = ord(ch)
        for lo, hi, scheme in _SCRIPT_BLOCKS:
            if lo <= cp <= hi:
                return scheme
    return None


def _fallback_transliterate(token: str) -> str:
    """R15 fallback for an unmapped Indic token: ITRANS + phonetic squash."""
    scheme = _detect_script(token)
    if scheme is None:
        return _NON_ALNUM_ASCII.sub("", token.lower())
    itrans = sanscript.transliterate(token, scheme, sanscript.ITRANS)
    itrans = itrans.lower()
    itrans = itrans.replace("aa", "a")
    itrans = re.sub(r"ee|ii", "i", itrans)
    itrans = re.sub(r"oo|uu", "u", itrans)
    itrans = itrans.replace("sh", "s")
    itrans = _NON_ALNUM_ASCII.sub("", itrans)
    return itrans


def learn_token_table(
    pairs: pd.DataFrame, min_count: int = 5, min_share: float = 0.8
) -> dict[str, str]:
    """R14: learn an (Indic token -> Latin token) mapping from GT pairs.

    ``pairs`` has columns ``s23_name`` (raw) and ``s1_name`` (raw). Only
    rows where the S2/S3 name contains Indic chars, the S1 name contains
    none, and both have the same token count are used. Indic tokens are
    the raw whitespace-split tokens (full sequence, legal words kept);
    Latin tokens are lowercased and run through the R13 keep-list rule.
    A mapping is kept when its co-occurrence count is >= ``min_count``
    and its share of that Indic token's total alignments is >= ``min_share``.
    """
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()

    for s23_name, s1_name in zip(pairs["s23_name"], pairs["s1_name"]):
        if s23_name is None or pd.isna(s23_name):
            continue
        if s1_name is None or pd.isna(s1_name):
            continue
        s23_name = str(s23_name)
        s1_name = str(s1_name)
        if not _INDIC_RANGE.search(s23_name) or _INDIC_RANGE.search(s1_name):
            continue

        indic_tokens = s23_name.split()
        latin_tokens = _latin_tokens(s1_name)
        if len(indic_tokens) != len(latin_tokens) or not indic_tokens:
            continue

        for indic_tok, latin_tok in zip(indic_tokens, latin_tokens):
            counts[(indic_tok, latin_tok)] += 1
            totals[indic_tok] += 1

    table: dict[str, str] = {}
    for (indic_tok, latin_tok), count in counts.items():
        if count >= min_count and count / totals[indic_tok] >= min_share:
            table[indic_tok] = latin_tok
    return table


def save_token_table(table: dict, path: Path | str | None = None) -> None:
    """Write ``table`` as UTF-8 JSON (``ensure_ascii=False``)."""
    if path is None:
        import config

        path = config.MODELS_DIR / "indic_tokens.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)


def load_token_table(path: Path | str | None = None) -> dict:
    """Load a token table previously written by ``save_token_table``."""
    if path is None:
        import config

        path = config.MODELS_DIR / "indic_tokens.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_latin(s: str, table: Mapping[str, str]) -> str:
    """Convert an Indic/Latin-mixed string to an all-Latin, lowercase string.

    Each whitespace-split token is looked up in ``table`` (exact raw
    token); an unmapped Indic token falls back to ITRANS + phonetic
    squash (R15). Latin tokens are lowercased and pass through untouched.
    """
    if not s:
        return s
    out = []
    for token in s.split():
        if _INDIC_RANGE.search(token):
            mapped = table.get(token)
            if mapped is not None:
                out.append(mapped.lower())
            else:
                out.append(_fallback_transliterate(token))
        else:
            out.append(token.lower())
    return " ".join(w for w in out if w)


def normalize_names(
    names: pd.Series, table: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Vectorized ``normalize_name`` over a pandas string Series.

    Maps over unique values only (real input has ~5M rows with heavy
    repetition), then joins the results back so the result is aligned to
    ``names.index``. When ``table`` is given, ``to_latin`` is applied to
    raw values containing Indic characters before ``normalize_name`` runs
    (clean_text's NFKD fold would otherwise destroy Indic vowel signs and
    viramas first); ``was_indic`` always reflects the original raw text.
    """

    def _process(v):
        if v is None or pd.isna(v):
            raw = ""
        else:
            raw = v
        was_indic = bool(_INDIC_RANGE.search(raw))
        text = raw
        if table is not None and was_indic:
            text = to_latin(raw, table)
        result = normalize_name(text)
        result["was_indic"] = was_indic
        return result

    values = names.tolist()
    unique_values = list(dict.fromkeys(values))
    lookup = {v: _process(v) for v in unique_values}

    out = pd.DataFrame(
        [lookup[v] for v in values],
        index=names.index,
        columns=_COLUMNS,
    )
    for col in ("name_clean", "name_sorted", "name_key", "legal", "alt_name"):
        out[col] = out[col].astype("string[pyarrow]")
    out["was_indic"] = out["was_indic"].astype("bool")
    return out
