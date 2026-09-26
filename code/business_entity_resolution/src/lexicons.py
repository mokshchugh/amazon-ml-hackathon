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

# ---------------------------------------------------------------------------
# Address dictionaries (SPEC step 4). Hand-written domain knowledge only.
# ---------------------------------------------------------------------------

# Step 4.2 abbreviations, verbatim from the SPEC. "h no" / "h.no" are
# two-token forms and are matched by a regex in normalize.py.
STREET_ABBR: dict[str, str] = {
    "st": "street", "saint": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "dr": "drive",
    "ct": "court",
    "ln": "lane",
    "blvd": "boulevard", "bd": "boulevard",
    "r": "rue",
    "hn": "house", "h.no": "house", "h no": "house",
    "fl": "floor",
}

# Ruling R4: these are also ordinary words / names ("Saint Louis",
# "R Nagar") and expand only in a numbered street context.
AMBIGUOUS_ABBR = {"st", "saint", "r", "bd"}

# Common suffix abbreviations beyond the SPEC list; treated like the
# unambiguous STREET_ABBR entries.
STREET_ABBR_EXTRA: dict[str, str] = {
    "pkwy": "parkway", "hwy": "highway", "cir": "circle", "pl": "place",
    "trl": "trail", "sq": "square", "flr": "floor",
}

# Canonical street-type words: a chunk holding one of these is the street.
STREET_WORDS = {
    "street", "road", "avenue", "drive", "court", "lane", "boulevard", "rue",
    "way", "place", "circle", "parkway", "highway", "terrace", "trail",
    "square", "allee", "impasse", "chemin", "route", "cours", "quai",
    "passage", "marg", "path", "pike", "plaza", "alley", "expressway",
    "freeway",
}

# Leading words dropped from the street string ("No. 5", "Plot No 12").
ADDR_PREFIX_WORDS = {
    "no", "n", "number", "num", "house", "plot", "door", "shop", "flat",
    "unit", "apt", "apartment", "suite", "ste", "room", "floor",
}

# Compass directions after a street type ("Due Ave W").
DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west"}

# Spelled ordinals -> number ("Seventh Pl" == "7th Pl").
ORDINAL_WORDS = {
    w: i for i, w in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
        "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(), start=1)
}

# Words that start a landmark phrase; the street string stops there.
LANDMARK_WORDS = {"near", "nr", "opp", "opposite", "behind"}

# A digit-free chunk with one of these is not a city (for the vocab).
NON_CITY_WORDS = (STREET_WORDS - {"court", "place", "square", "passage"}) | {
    "rd", "ave", "av", "dr", "ln", "blvd", "floor", "unit", "apt", "suite",
    "flat", "plot", "shop", "block", "near", "nr", "opp", "opposite",
    "behind", "building", "tower", "complex", "maison",
}

# State/region names that are also common city names: kept in the city
# vocabulary even though they match a state.
STATE_NAMED_CITIES = {
    "new york", "washington", "delhi", "chandigarh", "puducherry",
    "pondicherry", "paris", "vienne",
}

# Country strings (lowercased) -> the canonical key used by the dicts.
COUNTRY_ALIASES = {
    "us": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "united states": "US", "united states of america": "US",
    "india": "India", "in": "India", "ind": "India", "bharat": "India",
    "france": "France", "fr": "France", "fra": "France",
}

# US states: full name -> USPS code (50 states + DC).
US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Indian states / UTs: English names, common aliases, codes and
# native-script names -> canonical code.
IN_STATES: dict[str, str] = {
    "andhra pradesh": "AP", "arunachal pradesh": "AR", "assam": "AS",
    "bihar": "BR", "chhattisgarh": "CG", "chattisgarh": "CG",
    "chhatisgarh": "CG", "goa": "GA", "gujarat": "GJ", "haryana": "HR",
    "himachal pradesh": "HP", "jharkhand": "JH", "karnataka": "KA",
    "kerala": "KL", "keralam": "KL", "madhya pradesh": "MP",
    "maharashtra": "MH", "manipur": "MN", "meghalaya": "ML",
    "mizoram": "MZ", "nagaland": "NL", "odisha": "OD", "orissa": "OD",
    "punjab": "PB", "rajasthan": "RJ", "sikkim": "SK", "tamil nadu": "TN",
    "tamilnadu": "TN", "telangana": "TG", "tripura": "TR",
    "uttar pradesh": "UP", "uttarakhand": "UK", "uttaranchal": "UK",
    "west bengal": "WB", "andaman and nicobar islands": "AN",
    "andaman and nicobar": "AN", "chandigarh": "CH",
    "dadra and nagar haveli and daman and diu": "DH",
    "dadra and nagar haveli": "DH", "daman and diu": "DH", "delhi": "DL",
    "nct of delhi": "DL", "jammu and kashmir": "JK", "ladakh": "LA",
    "lakshadweep": "LD", "puducherry": "PY", "pondicherry": "PY",
    # codes, including older / alternative ones
    "ap": "AP", "ar": "AR", "as": "AS", "br": "BR", "cg": "CG", "ct": "CG",
    "ga": "GA", "gj": "GJ", "hr": "HR", "hp": "HP", "jh": "JH", "ka": "KA",
    "kl": "KL", "mp": "MP", "mh": "MH", "mn": "MN", "ml": "ML", "mz": "MZ",
    "nl": "NL", "od": "OD", "or": "OD", "pb": "PB", "rj": "RJ", "sk": "SK",
    "tn": "TN", "tg": "TG", "ts": "TG", "tr": "TR", "up": "UP", "uk": "UK",
    "ut": "UK", "wb": "WB", "an": "AN", "ch": "CH", "dh": "DH", "dn": "DH",
    "dd": "DH", "dl": "DL", "jk": "JK", "la": "LA", "ld": "LD", "py": "PY",
    # native-script names (matched on the raw text, before clean_text)
    "महाराष्ट्र": "MH", "दिल्ली": "DL", "उत्तर प्रदेश": "UP", "ಕರ್ನಾಟಕ": "KA",
    "தமிழ்நாடு": "TN", "পশ্চিমবঙ্গ": "WB", "ગુજરાત": "GJ", "తెలంగాణ": "TG",
    "हरियाणा": "HR", "राजस्थान": "RJ", "കേരളം": "KL", "बिहार": "BR",
    "मध्य प्रदेश": "MP", "ఆంధ్రప్రదేశ్": "AP", "ਪੰਜਾਬ": "PB", "ଓଡ଼ିଶା": "OD",
    "पंजाब": "PB", "गुजरात": "GJ", "कर्नाटक": "KA", "केरल": "KL",
    "तमिलनाडु": "TN", "तमिल नाडु": "TN", "पश्चिम बंगाल": "WB",
    "तेलंगाना": "TG", "आंध्र प्रदेश": "AP", "ओडिशा": "OD", "असम": "AS",
    "অসম": "AS", "झारखंड": "JH", "छत्तीसगढ़": "CG", "उत्तराखंड": "UK",
    "हिमाचल प्रदेश": "HP", "गोवा": "GA", "जम्मू और कश्मीर": "JK",
    "चंडीगढ़": "CH", "त्रिपुरा": "TR", "মহারাষ্ট্র": "MH",
}

# France: the 13 metropolitan régions (canonical, lowercase hyphenated).
FR_REGIONS = [
    "auvergne-rhone-alpes", "bourgogne-franche-comte", "bretagne",
    "centre-val-de-loire", "corse", "grand-est", "hauts-de-france",
    "ile-de-france", "normandie", "nouvelle-aquitaine", "occitanie",
    "pays-de-la-loire", "provence-alpes-cote-d-azur",
]
FR_REGION_ALIASES = {
    "paca": "provence-alpes-cote-d-azur", "brittany": "bretagne",
    "normandy": "normandie", "corsica": "corse",
}

_FR_DEPTS_BY_REGION = {
    "auvergne-rhone-alpes": [
        "ain", "allier", "ardeche", "cantal", "drome", "isere", "loire",
        "haute-loire", "puy-de-dome", "rhone", "savoie", "haute-savoie"],
    "bourgogne-franche-comte": [
        "cote-d-or", "doubs", "jura", "nievre", "haute-saone",
        "saone-et-loire", "yonne", "territoire de belfort"],
    "bretagne": ["cotes-d-armor", "finistere", "ille-et-vilaine", "morbihan"],
    "centre-val-de-loire": [
        "cher", "eure-et-loir", "indre", "indre-et-loire", "loir-et-cher",
        "loiret"],
    "corse": ["corse-du-sud", "haute-corse"],
    "grand-est": [
        "ardennes", "aube", "marne", "haute-marne", "meurthe-et-moselle",
        "meuse", "moselle", "bas-rhin", "haut-rhin", "vosges"],
    "hauts-de-france": ["aisne", "nord", "oise", "pas-de-calais", "somme"],
    "ile-de-france": [
        "paris", "seine-et-marne", "yvelines", "essonne", "hauts-de-seine",
        "seine-saint-denis", "val-de-marne", "val-d-oise"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine-maritime"],
    "nouvelle-aquitaine": [
        "charente", "charente-maritime", "correze", "creuse", "dordogne",
        "gironde", "landes", "lot-et-garonne", "pyrenees-atlantiques",
        "deux-sevres", "vienne", "haute-vienne"],
    "occitanie": [
        "ariege", "aude", "aveyron", "gard", "haute-garonne", "gers",
        "herault", "lot", "lozere", "hautes-pyrenees", "pyrenees-orientales",
        "tarn", "tarn-et-garonne"],
    "pays-de-la-loire": [
        "loire-atlantique", "maine-et-loire", "mayenne", "sarthe", "vendee"],
    "provence-alpes-cote-d-azur": [
        "alpes-de-haute-provence", "hautes-alpes", "alpes-maritimes",
        "bouches-du-rhone", "var", "vaucluse"],
}

# All 96 metropolitan départements -> région.
FR_DEPT_TO_REGION: dict[str, str] = {
    dept: region
    for region, depts in _FR_DEPTS_BY_REGION.items()
    for dept in depts
}
