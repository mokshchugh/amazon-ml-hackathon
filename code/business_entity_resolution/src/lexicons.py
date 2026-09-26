"""Hand-written domain dictionaries for name and address normalization.

No external lookups: every entry here is domain knowledge from SPEC step 2
(legal-suffix canonicalization) and step 4 (address abbreviations), not data
fetched from any registry or API.
"""

# Legal-suffix words from SPEC step 2, substep 7. Synonyms map to one
# canonical form; every other legal word maps to itself.
_LEGAL_WORDS = [
    "inc", "llc", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "lp", "plc", "sarl", "sas", "sasu", "eurl",
    "sa", "sci", "cie", "gmbh", "groupe",
]

_LEGAL_SYNONYMS = {
    "private": "pvt",
    "limited": "ltd",
    "corporation": "corp",
    "company": "co",
}

LEGAL_CANON: dict[str, str] = {
    word: _LEGAL_SYNONYMS.get(word, word) for word in _LEGAL_WORDS
}

# Honorifics stripped from names (SPEC step 2, substep 8).
HONORIFICS = {"m/s", "smt", "shri", "sri", "the"}
