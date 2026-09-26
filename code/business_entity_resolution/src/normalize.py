"""Business-name normalization (SPEC step 2).

Cleans raw ``business_name`` strings into a set of comparable forms:
``name_clean`` (word order kept), ``name_sorted`` (words sorted),
``name_key`` (letters/digits only, no spaces, no connector words),
plus the extracted ``legal`` suffix and any ``alt_name`` split out of a
"doing business as" / "dba" name.

Indic transliteration (SPEC step 3): ``to_latin`` with a learned token
table plus an ITRANS fallback.

Address parsing (SPEC step 4): ``parse_address`` / ``normalize_addresses``
split an address into postcode, house numbers, street, city and state;
``normalize_frame`` is the single entry point that adds both the name and
the address columns to a loaded source frame.
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

from lexicons import (
    ADDR_PREFIX_WORDS, AMBIGUOUS_ABBR, COUNTRY_ALIASES, DIRECTIONS,
    FR_DEPT_TO_REGION, FR_REGION_ALIASES, FR_REGIONS, HONORIFICS, IN_STATES,
    LANDMARK_WORDS, LEGAL_CANON, NON_CITY_WORDS, ORDINAL_WORDS,
    STATE_NAMED_CITIES, STREET_ABBR, STREET_ABBR_EXTRA, STREET_WORDS,
    US_STATES,
)

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
    if not s.isascii():  # ASCII is unchanged by NFKC/NFKD: skip the fold
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


# ---------------------------------------------------------------------------
# Address normalization and parsing (SPEC step 4)
# ---------------------------------------------------------------------------

_INDIC_RUN = re.compile("[%s-%s%s%s]+" % (chr(0x0900), chr(0x0DFF), chr(0x200C), chr(0x200D)))  # Indic + ZWNJ/ZWJ
_HOUSE_NO = re.compile(r"\bh\s*\.?\s*no\b|\bhn\b")  # hn, h.no, h no -> house
_ADDR_JUNK = re.compile(r"[^\w,/\-]|_")
_LONE_SLASH = re.compile(r"(?<!\d)/|/(?!\d)")  # "/" kept only as in "4/1"
_LONE_HYPHEN = re.compile(r"(?<!\w)-|-(?!\w)")  # "-" kept only inside words
_LETTER_DIGIT_HYPHEN = re.compile(r"(?<=[a-z])-(?=\d)|(?<=\d)-(?=[a-z])")
_NUM_TOKEN = re.compile(r"[a-z]{0,2}\d+(?:[-/]\d+)*(?:bis|ter|[a-z])?")
_DIGIT = re.compile(r"\d")
_ORDINAL = re.compile(r"0*(\d+)(?:st|nd|rd|th)")
_LEADING_ZEROS = re.compile(r"^0+(?=\d)")
_AND_WORD = re.compile(r"\band\b|\bet\b")
_ST_WORD = re.compile(r"\bst\b")
_STE_WORD = re.compile(r"\bste\b")
_UNAMBIG_ABBR = {k: STREET_ABBR[k] for k in ("rd", "ave", "av", "dr", "ct", "ln", "blvd", "str")}
_UNAMBIG_ABBR.update(STREET_ABBR_EXTRA)
_POSTCODE_LEN = {"US": 5, "France": 5, "India": 6}
# Street types before abbreviation expansion (postcode rules run first).
_STREET_ANY = STREET_WORDS | set(STREET_ABBR) | set(STREET_ABBR_EXTRA)
# "Unit 12345" / "Private Road 67603" / "Box 12345": a number, not a postcode.
_NOT_BEFORE_POSTCODE = ADDR_PREFIX_WORDS | STREET_WORDS | set(_UNAMBIG_ABBR) | {
    "box", "pmb", "fm", "cr", "fl", "bldg", "building", "trailer", "lot", "space", "spc", "cs", "bp"}
_ADDR_COLUMNS = ["addr_clean", "addr_tokens", "postcode", "house_nums", "street", "city", "state", "has_addr"]


def _state_key(s: str) -> str:
    return _NON_ALNUM.sub("", _AND_WORD.sub(" ", s))


def _city_key(s: str) -> str:
    return _NON_ALNUM.sub("", _STE_WORD.sub("sainte", _ST_WORD.sub("saint", s)))


def _state_table(names: Mapping[str, str]) -> dict[str, str]:
    return {_state_key(clean_text(k)): v for k, v in names.items() if not _INDIC_RANGE.search(k)}


_STATE_TABLES = {
    "US": _state_table({**US_STATES, **{c.lower(): c for c in US_STATES.values()}}),
    "India": _state_table(IN_STATES),
    "France": _state_table({**{r: r for r in FR_REGIONS}, **FR_REGION_ALIASES, **FR_DEPT_TO_REGION}),
}
_NATIVE_STATES = sorted(
    ((unicodedata.normalize("NFC", k), v) for k, v in IN_STATES.items() if _INDIC_RANGE.search(k)),
    key=lambda kv: -len(kv[0]),
)


def _canon_country(country) -> str:
    if country is None or pd.isna(country):
        return ""
    return COUNTRY_ALIASES.get(str(country).strip().lower(), str(country))


def _canon_token(t: str) -> str:
    """"a-68" -> "a68"; ordinals "seventh" / "7st" / "7th" -> "7th"."""
    if "-" in t:
        t = _LETTER_DIGIT_HYPHEN.sub("", t)
    m = _ORDINAL.fullmatch(t)
    n = int(m.group(1)) if m else ORDINAL_WORDS.get(t)
    if n is None:
        return t
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _addr_chunks(raw: str, table: Mapping[str, str] | None) -> tuple[list[list[str]], str]:
    """Clean an address into comma chunks of tokens; also return any
    native-script state code (R16: matched on the raw text, before the
    NFKD fold in clean_text destroys Indic marks)."""
    native = ""
    if _INDIC_RANGE.search(raw):
        raw = unicodedata.normalize("NFC", raw)
        for name, code in _NATIVE_STATES:
            if name in raw:
                native = native or code
                raw = raw.replace(name, ",")
        if _INDIC_RANGE.search(raw):
            raw = _INDIC_RUN.sub(lambda m: " " + to_latin(m.group(), table or {}) + " ", raw)
    text = _HOUSE_NO.sub(" house ", clean_text(raw).replace("#", " "))
    text = _LONE_HYPHEN.sub(" ", _LONE_SLASH.sub(" ", _ADDR_JUNK.sub(" ", text)))
    chunks = []
    for part in text.split(","):
        toks = [_canon_token(t) for t in part.split()]
        for i in range(len(toks) - 1, 0, -1):  # "5 bis" -> "5bis"
            if toks[i] in ("bis", "ter") and toks[i - 1].isdigit():
                toks[i - 1] += toks.pop(i)
        if toks:
            chunks.append(toks)
    return chunks, native


def _expand_abbr(toks: list[str], is_num: list[bool]) -> None:
    """Ruling R4, in place. Lone-token chunks (state codes such as CT, FL)
    are never expanded."""
    n = len(toks)
    if n < 2:
        return
    has_digit = any(is_num) or bool(_DIGIT.search("".join(toks)))
    for i, t in enumerate(toks):
        after_num = i > 0 and is_num[i - 1]
        # "last" also covers a trailing direction or number ("Due Ave W").
        last = i == n - 1 or is_num[i + 1] or toks[i + 1] in DIRECTIONS
        if t in _UNAMBIG_ABBR:
            if last or after_num:
                toks[i] = _UNAMBIG_ABBR[t]
        elif t == "fl":
            toks[i] = "floor"
        elif t in AMBIGUOUS_ABBR and has_digit:
            # st/saint are street types only in the trailing US position.
            if t in ("r", "bd") or last:
                toks[i] = STREET_ABBR[t]


def _street_of(toks: list[str], is_num: list[bool]) -> str:
    i = 0
    while i < len(toks) and (is_num[i] or toks[i] in ADDR_PREFIX_WORDS):
        i += 1
    out = []
    for t in toks[i:]:
        if t in LANDMARK_WORDS:
            break
        out.append(t)
    return " ".join(out)


def _partial_city(toks: list[str], idx: Mapping[str, tuple]) -> str:
    """Longest chunk suffix, then longest chunk prefix, found in the vocab."""
    n = len(toks)
    for size in range(n - 1, 0, -1):
        for sub in (toks[n - size:], toks[:size]):
            hit = idx.get(_city_key(" ".join(sub)))
            if hit:
                return hit[1]
    return ""


def _parse(raw, country, city_idx: Mapping[str, tuple], table=None) -> tuple:
    raw = "" if raw is None or pd.isna(raw) else str(raw)
    cc = _canon_country(country)
    chunks, state = _addr_chunks(raw, table)
    nums = [[bool(_NUM_TOKEN.fullmatch(t)) for t in toks] for toks in chunks]

    # Postcode: 5 digits (US/France, last token of its chunk after a word, or
    # a lone final chunk; France also "59000 Lille"), 6 digits (India), else
    # (R19) the longest 4-6 digit run that is not the address's first token
    # nor the leading number of a street chunk.
    plen, postcode, pc_at = _POSTCODE_LEN.get(cc), "", None
    for ci, toks in enumerate(chunks):
        streety = not _STREET_ANY.isdisjoint(toks)
        for ti, t in enumerate(toks):
            if not t.isdigit():
                continue
            if plen == 6:
                ok = len(t) == 6 and t[0] != "0"
            elif plen == 5:
                ok = len(t) == 5 and ((ti == len(toks) - 1 and (
                    (ti > 0 and not nums[ci][ti - 1] and toks[ti - 1] not in _NOT_BEFORE_POSTCODE)
                    or (ti == 0 and ci == len(chunks) - 1)))
                    or (cc == "France" and ti == 0 and len(toks) > 1 and not nums[ci][1]
                        and not t.startswith("00") and not streety))
            else:
                ok = (4 <= len(t) <= 6 and len(t) >= len(postcode) and (ci, ti) != (0, 0)
                      and not (streety and ti == nums[ci].index(True)))
            if ok:
                postcode, pc_at = t, (ci, ti)
    if pc_at:
        nums[pc_at[0]][pc_at[1]] = False

    # State (native script first, then state-only chunks, then chunks that
    # are also a known city). City = the whole-chunk vocab hit most frequent
    # in Source 1 (ties: the later chunk), else a partial (suffix/prefix) hit.
    states = _STATE_TABLES.get(cc, {})
    state_only, duals, whole = {}, [], []
    for ci, toks in enumerate(chunks):
        words = [t for ti, t in enumerate(toks) if (ci, ti) != pc_at]
        text = " ".join(words)
        if not words or _DIGIT.search(text):
            continue
        st, city = states.get(_state_key(text)), city_idx.get(_city_key(text))
        if st and city:
            duals.append((ci, st, city))
        elif st:
            state_only[ci] = st
        elif city:
            whole.append((city[0], ci, city[1]))
    if not state and state_only:
        state = next(iter(state_only.values()))
    dual_state = None
    if not state and duals:
        dual_state, state = duals[0][0], duals[0][1]
    whole += [(c[0], ci, c[1]) for ci, _, c in duals if ci != dual_state]

    # Abbreviations, house numbers, street.
    chunk_nums, street_ci, fallback_ci = {}, None, None
    for ci, toks in enumerate(chunks):
        if ci in state_only:
            continue
        _expand_abbr(toks, nums[ci])
        for ti, t in enumerate(toks):
            if nums[ci][ti] and t[0].isdigit():
                toks[ti] = t = _LEADING_ZEROS.sub("", t)
            if nums[ci][ti]:
                chunk_nums.setdefault(ci, []).append(t)
        if any(t in STREET_WORDS for t in toks):
            if street_ci is None or (any(nums[ci]) and not any(nums[street_ci])):
                street_ci = ci
        elif fallback_ci is None and _street_of(toks, nums[ci]) and any(nums[ci]):
            fallback_ci = ci
    sci = street_ci if street_ci is not None else fallback_ci
    street = _street_of(chunks[sci], nums[sci]) if sci is not None else ""
    # Street chunk's numbers first (chunks get reordered), then text order.
    house_nums = list(chunk_nums.get(sci, []))
    for ci, found in chunk_nums.items():
        if ci != sci:
            house_nums.extend(found)

    city = max(whole)[2] if whole else ""
    if not city:
        for ci in range(len(chunks) - 1, -1, -1):
            if ci not in state_only and ci != sci and len(chunks[ci]) > 1:
                city = _partial_city(chunks[ci], city_idx)
                if city:
                    break
    if not city and dual_state is not None:
        city = duals[0][2][1]

    tokens = []
    for ci, toks in enumerate(chunks):
        tokens.extend([state_only[ci].lower()] if ci in state_only else toks)
    return (" ".join(tokens), tokens, postcode, house_nums, street, city, state, bool(tokens))


_CITY_IDX_CACHE: dict[int, tuple] = {}


class CitySet(set):
    """A set of city names that also keeps each name's Source 1 count
    (``.counts``), used to pick the most common city among candidates.
    A plain ``set`` works too; its cities all rank equally."""

    def __init__(self, counts: Mapping[str, int] = ()):
        super().__init__(counts)
        self.counts = dict(counts)

    def __reduce__(self):  # pickle / copy keep the counts
        return (CitySet, (self.counts,))


def _city_index(vocab) -> dict[str, tuple[int, str]]:
    """City-key -> (S1 count, canonical vocab entry), cached per vocab object."""
    if not vocab:
        return {}
    hit = _CITY_IDX_CACHE.get(id(vocab))
    if hit is None or hit[0] is not vocab or hit[1] != len(vocab):
        counts = getattr(vocab, "counts", {})
        idx: dict[str, tuple[int, str]] = {}
        for c in sorted(vocab):
            idx.setdefault(_city_key(c), (counts.get(c, 0), c))
        hit = _CITY_IDX_CACHE[id(vocab)] = (vocab, len(vocab), idx)
    return hit[2]


def _vocab_for(city_vocab, country):
    return city_vocab.get(country) or city_vocab.get(_canon_country(country)) or set()


def parse_address(s: str, country: str, city_vocab: Mapping[str, set[str]],
                  table: Mapping[str, str] | None = None) -> dict:
    """Parse one raw address (SPEC step 4) into its comparable parts."""
    idx = _city_index(_vocab_for(city_vocab, country))
    return dict(zip(_ADDR_COLUMNS, _parse(s, country, idx, table)))


def build_city_vocab(addresses: pd.Series, countries: pd.Series, min_count: int = 3) -> dict[str, set[str]]:
    """Cities per country: digit-free comma chunks that are not a state /
    region / departement (except STATE_NAMED_CITIES) and hold no street or
    unit word, seen in >= ``min_count`` of the given (Source 1) addresses.
    Spelling variants sharing a city key keep only the most frequent one.
    Each value is a ``CitySet`` carrying the counts."""
    pairs = pd.DataFrame({"a": addresses.to_numpy(), "c": countries.to_numpy()}).dropna()
    counts: Counter = Counter()
    variants: dict[tuple, Counter] = {}
    for (addr, country), n in pairs.value_counts().items():
        states = _STATE_TABLES.get(_canon_country(country), {})
        seen = set()
        for toks in _addr_chunks(addr, None)[0]:
            text = " ".join(toks)
            key = _city_key(text)
            if (len(key) < 3 or key in seen or _DIGIT.search(text) or NON_CITY_WORDS.intersection(toks)
                    or (_state_key(text) in states and text not in STATE_NAMED_CITIES)):
                continue
            seen.add(key)
            counts[(country, key)] += n
            variants.setdefault((country, key), Counter())[text] += n
    kept: dict[str, dict[str, int]] = {}
    for (country, key), n in counts.items():
        if n >= min_count:
            kept.setdefault(country, {})[variants[(country, key)].most_common(1)[0][0]] = n
    return {country: CitySet(c) for country, c in kept.items()}


def normalize_addresses(df: pd.DataFrame, city_vocab: Mapping[str, set[str]],
                        table: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Return ``df`` with the parsed address columns added. Parses each
    unique (business_address, country) pair once."""
    idx_by_country: dict = {}
    lookup: dict = {}
    rows = []
    for key in zip(df["business_address"].tolist(), df["country"].tolist()):
        res = lookup.get(key)
        if res is None:
            if key[1] not in idx_by_country:
                idx_by_country[key[1]] = _city_index(_vocab_for(city_vocab, key[1]))
            res = lookup[key] = _parse(key[0], key[1], idx_by_country[key[1]], table)
        rows.append(res)
    out = df.copy()
    cols = list(zip(*rows)) if rows else [[] for _ in _ADDR_COLUMNS]
    for name, values in zip(_ADDR_COLUMNS, cols):
        if name in ("addr_tokens", "house_nums"):
            out[name] = pd.Series(list(values), index=df.index, dtype=object)
        elif name == "has_addr":
            out[name] = pd.Series(values, index=df.index, dtype=bool)
        else:
            out[name] = pd.Series(values, index=df.index, dtype="string[pyarrow]")
    return out


def normalize_frame(df: pd.DataFrame, token_table: Mapping[str, str],
                    city_vocab: Mapping[str, set[str]]) -> pd.DataFrame:
    """Single normalization entry point (R5): name columns + address columns."""
    names = normalize_names(df["business_name"], table=token_table)
    return normalize_addresses(pd.concat([df, names], axis=1), city_vocab, table=token_table)
