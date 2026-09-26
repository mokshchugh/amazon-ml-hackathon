"""Tests for address normalization and parsing (SPEC step 4, Task 7)."""
import pandas as pd

from normalize import build_city_vocab, normalize_addresses, normalize_frame, parse_address

V = {"US": {"indianapolis", "saint louis"}, "France": {"lille", "la teste-de-buch"}, "India": {"jaipur"}}


# --- brief tests -----------------------------------------------------------

def test_us_variants_agree():
    a = parse_address("3220. Gale St, Indianapolis, Indiana", "US", V)
    b = parse_address("3220 GALE SAINT, INDIANAPOLIS, IN", "US", V)
    for k in ("street", "city", "state"): assert a[k] == b[k]
    assert a["house_nums"] == ["3220"] and a["street"] == "gale street" and a["state"] == "IN" and a["city"] == "indianapolis"


def test_french():
    a = parse_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France", V)
    assert a["house_nums"] == ["63"] and a["street"] == "rue de dieppe" and a["state"] == "hauts-de-france"
    assert parse_address("5 bis Rue Pierre Dignac, Gironde", "France", V)["house_nums"] == ["5bis"]
    assert parse_address("5 bis Rue Pierre Dignac, Gironde", "France", V)["state"] == "nouvelle-aquitaine"


def test_india_pin_and_native_state():
    a = parse_address("Hn 286 10-A, Hansa Vihar, Jaipur, राजस्थान 302012", "India", V)
    assert a["postcode"] == "302012" and a["state"] == "RJ" and a["city"] == "jaipur"


def test_saint_in_city_not_expanded():
    a = parse_address("9007 Kathlyn Drive, Saint Louis, MO", "US", V)
    assert a["city"] == "saint louis" and a["street"] == "kathlyn drive"


def test_empty_and_null_addresses():
    for raw in ["", "null", "<NULL>, <NULL>"]:
        a = parse_address(raw, "US", V); assert a["has_addr"] is False and a["house_nums"] == []


# --- controller-ruling tests -----------------------------------------------

def test_mack_rd_expanded_as_last_token():  # R4
    assert parse_address("Mack Rd, Haltom City, Texas", "US", V)["street"] == "mack road"


def test_unambiguous_abbr_after_house_number():  # R4
    a = parse_address("77 AV LEON JOUHAUX, LILLE", "France", V)
    assert a["street"] == "avenue leon jouhaux" and a["city"] == "lille"


def test_state_code_chunk_not_expanded():  # "CT" is Connecticut, not "court"
    assert parse_address("12 Elm Street, Hartford, CT", "US", V)["state"] == "CT"


def test_saint_hyphen_city_kept():
    a = parse_address("59 Route de Trebale, Saint-Nazaire, Pays de la Loire", "France", V)
    assert "saint-nazaire" in a["addr_tokens"] and a["state"] == "pays-de-la-loire"


def test_native_state_other_scripts_without_table():  # R16
    assert parse_address("Door No 183, Bangalore, ಕರ್ನಾಟಕ", "India", V)["state"] == "KA"
    assert parse_address("PLOT NO- #1434, BHUBANESWAR, ଓଡ଼ିଶା", "India", V)["state"] == "OD"
    assert parse_address("SHOP NO 101/102, NAGPUR, महाराष्ट्र", "India", V)["state"] == "MH"


def test_na_input():
    a = parse_address(pd.NA, "US", V)
    assert a["has_addr"] is False and a["addr_tokens"] == [] and a["postcode"] == ""


def test_leading_zeros_stripped_from_house_numbers():
    a = parse_address("00765 WESTWOOD DRIVE, SAINT LOUIS, MO", "US", V)
    b = parse_address("765 Westwood Drive, Saint Louis, MO", "US", V)
    assert a["house_nums"] == b["house_nums"] == ["765"]


def test_us_house_number_not_taken_as_postcode():
    a = parse_address("10047 FOREST AVE, CHICAGO, IL", "US", V)
    assert a["postcode"] == "" and a["house_nums"] == ["10047"] and a["street"] == "forest avenue"


def test_other_country_postcode_longest_run():
    assert parse_address("Hauptstrasse 5, 10115 Berlin", "Germany", {})["postcode"] == "10115"


def test_city_matches_chunk_suffix_and_key_variants():
    assert parse_address("12 Rue X, La Teste de Buch", "France", V)["city"] == "la teste-de-buch"
    assert parse_address("Main Road, Near Temple Jaipur, RJ", "India", V)["city"] == "jaipur"


def test_spelled_and_misspelled_ordinals_agree():
    a = parse_address("1667 7th Place, Mesa, AZ", "US", V)["street"]
    assert a == parse_address("1667- SEVENTH PL, AZ, MESA", "US", V)["street"] == "7th place"
    assert parse_address("241 28st Place, Ridgefield, WA", "US", V)["street"] == "28th place"


def test_abbr_before_direction_expanded():
    assert parse_address("620 Due Ave W, Unit 401, Nashville, TN", "US", V)["street"] == "due avenue w"


def test_city_choice_is_order_invariant_with_ranked_vocab():
    addrs = pd.Series(["Mahim, Mumbai, MH"] * 5 + ["Mahim, Thane, MH"] * 3)
    vocab = build_city_vocab(addrs, pd.Series(["India"] * 8), min_count=3)
    assert vocab["India"] == {"mahim", "mumbai", "thane"}
    for raw in ("12 Road X, Mahim, Mumbai, MH", "Mumbai, 12 Road X, Mahim, MH"):
        assert parse_address(raw, "India", vocab)["city"] == "mahim"  # most frequent in S1


def test_french_postcode_before_city():  # review fix 1
    a = parse_address("12 Rue X, 59000 LILLE, Nord", "France", V)
    assert a["postcode"] == "59000" and a["house_nums"] == ["12"] and a["city"] == "lille"
    # zero-padded house number before a street type is still a house number
    b = parse_address("00262 R DE LANNOY, ROUBAIX, Nord", "France", V)
    assert b["postcode"] == "" and b["house_nums"] == ["262"]


def test_street_number_first_regardless_of_chunk_order():  # review fix 2
    a = parse_address("Suite 5, 3220 Gale St, Indianapolis, IN", "US", V)
    b = parse_address("Indianapolis, #3220 GALE ST, Suite 5, IN", "US", V)
    assert a["house_nums"][0] == b["house_nums"][0] == "3220"


def test_named_risk_regressions():
    for raw in ("3220 Gale Street, Indianapolis, IN", "3220. Gale St, Indianapolis, Indiana",
                "#3220 GALE ST, INDIANAPOLIS, IN"):
        a = parse_address(raw, "US", V)
        assert a["house_nums"][0] == "3220" and a["street"] == "gale street", raw
    assert parse_address("3228 Gale St, Indianapolis, IN", "US", V)["house_nums"][0] == "3228"


def test_unknown_country_postcode_not_house_number():  # ruling R19
    for country in ("Atlantis", float("nan"), None):
        a = parse_address("3220 Gale St, Somewhere", country, V)
        assert a["postcode"] == "" and a["house_nums"] == ["3220"], country


def test_build_city_vocab():
    addrs = pd.Series(["1 Main St, Springfield, IL"] * 3 + ["2 Oak St, Shelbyville, IL"] * 2
                      + ["Rue A, Lille, Nord"] * 3)
    ctry = pd.Series(["US"] * 5 + ["France"] * 3)
    vocab = build_city_vocab(addrs, ctry, min_count=3)
    assert vocab["US"] == {"springfield"} and vocab["France"] == {"lille"}


def test_normalize_addresses_adds_columns():
    df = pd.DataFrame({"business_address": ["3220 Gale St, Indianapolis, IN", None],
                       "country": ["US", "US"]})
    out = normalize_addresses(df, V)
    assert len(out) == 2 and out.loc[0, "city"] == "indianapolis" and not out.loc[1, "has_addr"]


def test_normalize_frame():  # R5
    df = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3"],
        "business_name": ["Porter & Nall LLC", "Porter & Nall LLC", "Mack Inc"],
        "business_address": ["3220 Gale St, Indianapolis, IN", "3220 Gale St, Indianapolis, IN", ""],
        "country": ["US", "US", "US"],
        "source": ["S2", "S2", "S2"],
    })
    out = normalize_frame(df, {}, V)
    expected = set(df.columns) | {"name_clean", "name_sorted", "name_key", "legal", "alt_name", "was_indic",
                                  "addr_clean", "addr_tokens", "postcode", "house_nums", "street", "city",
                                  "state", "has_addr"}
    assert set(out.columns) == expected and len(out) == 3
    assert out.loc[0, "name_clean"] == "porter and nall" and out.loc[1, "street"] == "gale street"
