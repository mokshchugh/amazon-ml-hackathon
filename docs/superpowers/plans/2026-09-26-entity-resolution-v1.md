# Entity Resolution v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 pipeline that turns the 7 challenge TSVs into a validated `matching_results.tsv` and `candidate_pairs.tsv`. v1 is blocking → LightGBM → F0.5 decision layer, with no re-ranker.

**Architecture:** The pipeline runs per country:
- normalize names and addresses;
- generate a candidate shortlist (blocking);
- score each pair with a calibrated LightGBM model;
- turn the scores into match lists with a decision layer tuned for macro F0.5.

Every stage caches its output as parquet under `C:\Users\utkar\er_work\cache`. `run_all.py` chains the stages for `--split train` (train and holdout-evaluate) or `--split test` (inference only).

**Tech Stack:** Python 3.14.3, pandas 3.0.1 + pyarrow 23.0.1, scikit-learn 1.8.0, scipy 1.16.3, sparse-dot-topn 1.2.0, RapidFuzz 3.14.6, LightGBM 4.7.0, indic_transliteration 2.3.82, joblib 1.5.3, and pytest 9.1.1 for tests.

**Spec:** `SPEC.md` in the repo root. The master copy is https://claude.ai/code/artifact/473eeb77-30f5-48c5-9b6e-6563c89c9d10. Read the SPEC section cited in each task.

## Global Constraints

- **Python:** 3.14.3, in the venv `C:\Users\utkar\venvs\amazon-er`. Pinned packages are exactly those in SPEC §2.2 (core), §2.3 (re-rank, not installed in v1) and §2.5 (dev).
- **numpy:** 2.4.6, never the yanked 2.4.0.
- **Licenses:** only MIT, Apache-2.0, BSD-2/3, 0BSD, ISC, PSF-2.0, Zlib, CC0-1.0, ZPL-2.1, CNRI-Python or MPL-2.0. Never install unidecode, Levenshtein, python-Levenshtein or fuzzywuzzy.
- **No network:** no call to any network service while the pipeline runs (Fair Play).
- **Country:**
    - Treat `country` as an open set of strings; never hard-code {US, India}.
    - **No country feature** goes into the model.
    - Never compare records across countries.
- **Seeds:** 42 everywhere. LightGBM runs with `deterministic=True, force_row_wise=True`.
- **Paths:** code in `code/business_entity_resolution/`, outputs in `output/`.
    - Caches go in `C:\Users\utkar\er_work\cache`, models in `...\models`, submissions in `...\submissions`.
    - All paths come from `src/config.py` only.
- **Reading files:** `sep="\t"`, `encoding="utf-8"`, `dtype="string[pyarrow]"`, `quoting=csv.QUOTE_NONE`, `keep_default_na=False`.
- **Writing files:**
    - TAB-separated, UTF-8, header exactly `source1_entity_id\tmatched_entity_ids` or `source1_entity_id\tcandidate_entity_ids`.
    - One row per test S1 record. IDs joined by `,` with no quoting and no duplicates. An empty cell means no match.
    - Every matched ID also appears in the candidates.
- **Memory:** the budget is 23.7 GB RAM. Process one country at a time, and score candidate pairs in chunks of ≤2,000,000 rows.
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Commands:** run them from `code/business_entity_resolution/` with `PY=/c/Users/utkar/venvs/amazon-er/Scripts/python.exe`.

## Review Focus

1. **Empty, `null` or `<NULL>` names and addresses** (3.4% of S2/S3 addresses are empty). Normalization must return empty parts, not crash, and the record must still flow through every stage. *Tests: Task 5 `test_empty_and_null_names`, Task 7 `test_empty_and_null_addresses`.*
2. **A country never seen in training (France), or any new label.** Every S1 record of that country gets a row in both output files. *Test: Task 14 `test_unseen_country_rows_present`.*
3. **An S1 record with zero candidates.** It still gets exactly one row, with an empty list, in both files. *Test: Task 4 `test_write_id_lists_empty_rows`, Task 14 integration.*
4. **The same S2/S3 ID found by several searches, or claimed by several S1 records.** Output lists never contain duplicates, and an ID is never assigned to two S1 records. *Tests: Task 9 `test_candidates_deduplicated`, Task 13 `test_one_owner`.*
5. **Words like "Saint", "St" and "R" that are also city or name words** ("Saint Louis", "St Albans", "Saint-Nazaire"). Abbreviation expansion must not rewrite city names. *Test: Task 7 `test_saint_in_city_not_expanded`.*

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `code/business_entity_resolution/requirements.txt`, `requirements-rerank.txt`, `requirements-dev.txt`, `requirements.lock.txt` | pinned environments | 1 |
| `src/config.py` | every path, seed and constant shared across stages | 1 |
| `tests/conftest.py`, `pytest.ini` | put `src/` on the import path; define the `integration` marker | 1 |
| `src/check_env.py` | version and license gate; writes `THIRD_PARTY_LICENSES.md` | 2 |
| `src/smoke_test.py` | quick checks that every pinned tool works | 2 |
| `models.lock.json`, `LICENSE`, `README.md` | model pins, MIT license, run instructions | 2, 17 |
| `src/evaluate.py` | macro F0.5, blocking recall, reports, error samples | 3, 16 |
| `src/io_utils.py` | read TSVs into the parquet cache; write output TSVs | 4 |
| `src/lexicons.py` | hand-written dictionaries: legal suffixes, honorifics, street abbreviations, states and regions. **A refinement of the SPEC layout** that keeps `normalize.py` focused. | 5, 7 |
| `src/normalize.py` | name cleaning, transliteration, address parsing | 5, 6, 7 |
| `src/splits.py` | holdout and country-transfer splits | 8 |
| `src/blocking.py` | candidate generation | 9 |
| `src/siblings.py` | duplicate groups inside S2 and S3 | 10 |
| `src/features.py` | pair features | 11 |
| `src/train.py` | LightGBM training, calibration, save/load | 12 |
| `src/decide.py` | owner resolution, expected-F0.5 top-k, sibling consistency, tuning | 13 |
| `src/run_all.py` | orchestration, command line, packaging | 14, 17 |
| `experiments.md` (repo root) | log of every benchmark and submission | 14–16 |

---

### Task 1: Environment, pins, config and test harness

**Files:**
- Create: `code/business_entity_resolution/requirements.txt`, `requirements-rerank.txt`, `requirements-dev.txt`, `requirements.lock.txt`, `pytest.ini`, `src/__init__.py` (empty), `src/config.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces from `config.py`:
    - `REPO_ROOT: Path` (repo root, 3 levels above `src/`)
    - `DATA_DIR = REPO_ROOT/"student_resource"/"dataset"`
    - `OUTPUT_DIR = REPO_ROOT/"output"`
    - `WORK_DIR = Path(os.environ.get("ER_WORK_DIR", r"C:\Users\utkar\er_work"))`, with `CACHE_DIR`, `MODELS_DIR` and `SUBMISSIONS_DIR` under it
    - `SEED = 42`
    - `ensure_dirs() -> None`, which creates the work and output directories

- [ ] **Step 1:** Write the three requirement files with exactly the pins in SPEC §2.2 (29 lines), §2.3 (`-r requirements.txt`, `--extra-index-url https://download.pytorch.org/whl/cu128`, 23 lines) and §2.5 (pytest 9.1.1, pluggy 1.6.0, iniconfig 2.3.0, packaging 26.3).
- [ ] **Step 2:** Create the venv and install:

  ```bash
  py -3.14 -m venv /c/Users/utkar/venvs/amazon-er
  $PY -m pip install --upgrade pip
  $PY -m pip install -r requirements.txt -r requirements-dev.txt
  $PY -m pip check
  ```

  Expected: `No broken requirements found.`
- [ ] **Step 3:** Generate `requirements.lock.txt` with hashes:

  ```bash
  $PY -m pip install --dry-run --ignore-installed --quiet --report lock.json -r requirements.txt
  ```

  Then turn `lock.json` into lines of the form `name==version --hash=sha256:<archive_info.hashes.sha256>`, sorted by name, and delete `lock.json`. Expected: 29 lines, each with one `--hash`.
- [ ] **Step 4: Write the failing test** `tests/test_config.py`:

  ```python
  def test_paths_and_seed(tmp_path, monkeypatch):
      monkeypatch.setenv("ER_WORK_DIR", str(tmp_path / "w"))
      import importlib, config; importlib.reload(config)
      assert config.SEED == 42
      assert config.DATA_DIR.name == "dataset" and config.DATA_DIR.parent.name == "student_resource"
      assert config.CACHE_DIR == tmp_path / "w" / "cache"
      config.ensure_dirs()
      assert config.CACHE_DIR.is_dir() and config.MODELS_DIR.is_dir() and config.SUBMISSIONS_DIR.is_dir()
  ```

  `tests/conftest.py` inserts `Path(__file__).parents[1]/"src"` into `sys.path`. `pytest.ini` sets `testpaths = tests` and `markers = integration: needs the real dataset`.
- [ ] **Step 5:** Run `$PY -m pytest tests/test_config.py -v`. Expected: FAIL (no module `config`).
- [ ] **Step 6:** Implement `src/config.py` with the interface above.
- [ ] **Step 7:** Run `$PY -m pytest tests -v`. Expected: 1 passed.
- [ ] **Step 8: Commit:** `git add code/business_entity_resolution && git commit -m "build: pinned environment, config and pytest harness"`.

### Task 2: License/version gate, smoke tests, offline proof

**Files:**
- Create: `src/check_env.py`, `src/smoke_test.py`, `tests/test_check_env.py`, `models.lock.json`, `LICENSE`, `README.md` (environment section only), `THIRD_PARTY_LICENSES.md` (generated)

**Interfaces:**
- Produces:
    - `classify_license(expression: str, license_field: str, classifiers: list[str]) -> Literal["allowed","banned","unknown"]`
    - `check_versions(req_files: list[Path]) -> list[str]` (the mismatches)
    - `run_checks(rerank: bool = False) -> list[str]` (all errors)
    - `main()`, which takes `--write-report` and `--rerank`, and exits 0 on PASS and 1 otherwise
    - `BANNED = {"unidecode","levenshtein","python-levenshtein","fuzzywuzzy"}`

- [ ] **Step 1: Write the failing tests** with these exact cases, taken from metadata seen during verification:

  ```python
  @pytest.mark.parametrize("expr,field,cls,want", [
      ("MIT", "", [], "allowed"),
      ("", "BSD 3-Clause License", ["License :: OSI Approved :: BSD License"], "allowed"),      # pandas
      ("", "Copyright (c) 2001-2002 Enthought, Inc.", ["License :: OSI Approved :: BSD License"], "allowed"),  # scipy
      ("MPL-2.0 AND MIT", "", [], "allowed"),                                                   # tqdm
      ("", "Apache 2.0 License", [], "allowed"),                                                # transformers
      ("Apache-2.0 AND CNRI-Python", "", [], "allowed"),                                        # regex
      ("GPL-2.0-or-later", "", [], "banned"),
      ("", "", ["License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)"], "banned"),
      ("", "", [], "unknown"),
  ])
  def test_classify_license(expr, field, cls, want):
      assert check_env.classify_license(expr, field, cls) == want

  def test_banned_names_fail(monkeypatch):
      monkeypatch.setattr(check_env, "installed", lambda: {"unidecode": "1.4.0"})
      assert any("unidecode" in e for e in check_env.check_banned())
  ```

- [ ] **Step 2:** Run `$PY -m pytest tests/test_check_env.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `check_env.py`.
    - Read installed distributions with `importlib.metadata.distributions()`.
    - Any GPL, AGPL, LGPL, SSPL or "Non-Commercial" text is banned; that check runs before the allowlist match.
    - `pip`, `setuptools` and `wheel` count as tooling, but are still license-checked.
    - `--write-report` writes a Markdown table (package, version, license) to `THIRD_PARTY_LICENSES.md`, then appends the 3 models from `models.lock.json`.
- [ ] **Step 4:** Write `models.lock.json` with the 3 models from SPEC §3.1 (repo ID, SHA, license, parameter count). Write `LICENSE` (MIT, `Copyright (c) 2026 Team Barely Legal`). In `README.md`, write the Environment section: the SPEC §2.4 commands plus `check_env.py`.
- [ ] **Step 5:** Run `$PY -m pytest tests -v`, then `$PY src/check_env.py --write-report`. Expected: all tests pass, the gate prints `PASS`, and `THIRD_PARTY_LICENSES.md` lists every installed distribution: the 29 core pins, the 4 dev-only pins (pytest, pluggy, iniconfig, packaging) and pip.
- [ ] **Step 6:** Implement `smoke_test.py` with the SPEC §9.2 core checks.
    - LightGBM trains on 1,000 synthetic rows with AUC > 0.9.
    - RapidFuzz `fuzz.token_set_ratio("porter and nall","porter and nall llc") >= 95`.
    - `sparse_dot_topn.sp_matmul_topn` on a 100×100 TF-IDF matrix matches a dense argmax.
    - `indic_transliteration.sanscript.transliterate("लक्ष्मी", DEVANAGARI, ITRANS)` returns non-empty ASCII.
    - It prints `SMOKE PASS`. Run it; expected `SMOKE PASS`.
- [ ] **Step 7: Offline proof.** A team member disables Wi-Fi (a firewall rule needs admin), then runs `$PY src/check_env.py && $PY src/smoke_test.py`. Expected: `PASS`, then `SMOKE PASS`. Record the result in `experiments.md`, which is created now with its header.
- [ ] **Step 8:** Commit: `feat: environment gate, smoke tests, license report`. Then `git tag env-v1`. Tick SPEC §9.5 in the Claude Doc and re-export `SPEC.md`.

### Task 3: Metric and blocking-quality evaluation (SPEC §7.1, §8)

**Files:** Create `src/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Produces:
    - `f05(pred: set[str], truth: set[str]) -> float`
    - `macro_f05(pred: Mapping[str, set[str]], truth: Mapping[str, set[str]], s1_ids: Iterable[str]) -> float`, where a missing key means an empty set
    - `pair_recall(cands: Mapping[str, set[str]], truth: Mapping[str, set[str]]) -> float`
    - `reduction_ratio(n_candidate_pairs: int, n_s1: int, n_s23: int) -> float`
    - `report(pred, truth, s1_country: Mapping[str,str]) -> dict`, with keys `overall`, `by_country`, `singletons`, `pair_precision`, `pair_recall`

- [ ] **Step 1: Write the failing tests:**

  ```python
  def test_readme_example():
      assert f05({"S2-00047","S2-00193","S3-00812"}, {"S2-00047","S3-00812"}) == pytest.approx(0.7142857, 1e-6)
  def test_singleton_rules():
      assert f05(set(), set()) == 1.0
      assert f05({"S2-1"}, set()) == 0.0
      assert f05(set(), {"S2-1"}) == 0.0
  def test_macro_counts_missing_as_empty():
      truth = {"a": {"x"}, "b": set()}
      assert macro_f05({"a": {"x"}}, truth, ["a", "b"]) == 1.0
  def test_report_by_country():
      r = report({"a": {"x"}}, {"a": {"x"}, "b": {"y"}}, {"a": "US", "b": "France"})
      assert r["by_country"] == {"US": 1.0, "France": 0.0} and r["overall"] == 0.5
  ```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the functions in `evaluate.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: macro F0.5 and blocking metrics`.

### Task 4: Loading, parquet cache, output writing (SPEC §6 step 1, §8 step 13)

**Files:** Create `src/io_utils.py`, `tests/test_io_utils.py`

**Interfaces:**
- Produces:
    - `read_source(path: Path) -> pd.DataFrame`, with columns `entity_id, business_name, business_address, country` (string[pyarrow]) and `source` ("S1" | "S2" | "S3", taken from the ID prefix)
    - `read_ground_truth(path: Path) -> pd.DataFrame`, with columns `s1_id, s23_id` (long format; singletons have no rows)
    - `build_cache(split: Literal["train","test"]) -> None`, which writes `CACHE_DIR/f"{split}_{source}.parquet"` and, for train, `gt_pairs.parquet`; country filtering happens at load time
    - `load(split, source, country: str | None = None) -> pd.DataFrame`
    - `write_id_lists(path: Path, value_col: Literal["matched_entity_ids","candidate_entity_ids"], s1_ids: Sequence[str], lists: Mapping[str, Sequence[str]]) -> None`

- [ ] **Step 1: Write the failing tests** using `tmp_path` TSVs.
    - `test_read_source_keeps_null_text`: the row `S2-1\tnull\t<NULL>\tIndia` gives `business_name == "null"` and does not become NaN.
    - `test_read_ground_truth_long`: the row `S1-1\tS2-5,S3-9` gives 2 rows, and `S1-2\t` gives 0 rows.
    - `test_write_id_lists_empty_rows`:
        - a record with no list writes the line `S1-2\t`;
        - duplicate IDs in the input are written once, in first-seen order;
        - the header matches exactly;
        - the file round-trips through `validate_submission.validate_id_list_file` with 0 errors (import from `student_resource/utils`).
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `io_utils.py`. Use the read options from Global Constraints and write lines by hand; do not use pandas `to_csv`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run `$PY -c "import io_utils; io_utils.build_cache('train'); io_utils.build_cache('test')"` from `src/`. Expected: 7 parquet files, and `gt_pairs.parquet` has 7,638,365 rows. Row counts match SPEC §1 (train 2,206,821 / 5,034,616 / 5,285,603; test 1,732,544 / 4,887,273 / 5,082,316).
- [ ] **Step 6:** Commit: `feat: TSV loading, parquet cache, output writer`.

### Task 5: Name normalization (SPEC §6 step 2)

**Files:** Create `src/lexicons.py`, `src/normalize.py`, `tests/test_normalize_names.py`

**Interfaces:**
- Produces in `lexicons.py`:
    - `LEGAL_CANON: dict[str,str]`. The keys are all legal words from SPEC step 2; synonyms map to one canonical form: private→pvt, limited→ltd, corporation→corp, company→co. Every other word maps to itself.
    - `HONORIFICS = {"m/s","smt","shri","sri","the"}`
- Produces in `normalize.py`:
    - `clean_text(s: str) -> str`: NFKC, then NFKD with combining marks removed, lowercase, junk tokens stripped, whitespace collapsed
    - `normalize_name(s: str) -> dict`, with keys `name_clean, name_sorted, name_key, legal, alt_name`. `legal` is the canonical suffixes, space-joined and sorted.
    - `normalize_names(names: pd.Series) -> pd.DataFrame`, with the same 5 columns plus `was_indic: bool`

- [ ] **Step 1: Write the failing tests** (SPEC §6 step 2 and §9.6):

  ```python
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
      out = normalize_name(raw); assert out["name_clean"] == clean and out["legal"] == legal

  def test_dba_split():
      out = normalize_name("Synpyra doing business as Golden Intelligence Holdings")
      assert out["name_clean"] == "synpyra" and out["alt_name"] == "golden intelligence holdings"

  def test_sorted_and_key():
      out = normalize_name("Empire Inc Translational  Interstate")
      assert out["name_sorted"] == "empire interstate translational" and out["name_key"] == "empiretranslationalinterstate"

  def test_empty_and_null_names():
      for raw in ["", "null", "<NULL>", "--"]:
          assert normalize_name(raw)["name_clean"] == ""
  ```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the functions in the order of SPEC step 2 (substeps 1–9).
    - Digit fix: `1→l, 0→o, 3→e` only when the digit has a letter on both sides.
    - `was_indic` is True when the name has any character in U+0900–U+0DFF.
    - `normalize_names` maps `normalize_name` over unique values only, then joins the results back.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: business name normalization`.

### Task 6: Indic-script → Latin (SPEC §6 step 3)

**Files:** Modify `src/normalize.py`. Create `tests/test_translit.py`.

**Interfaces:**
- Produces:
    - `learn_token_table(pairs: pd.DataFrame, min_count: int = 5, min_share: float = 0.8) -> dict[str,str]`. `pairs` has columns `s23_name, s1_name` (raw). It aligns tokens by position after removing legal words, and only for Indic/Latin pairs with the same number of tokens.
    - `save_token_table(d, path)` and `load_token_table(path) -> dict`, as JSON in `MODELS_DIR/"indic_tokens.json"`
    - `to_latin(s: str, table: Mapping[str,str]) -> str`: table lookup for each token; unmapped Indic tokens use ITRANS followed by the phonetic squash from SPEC step 3; Latin tokens pass through
    - `normalize_names(names, table: Mapping[str,str] | None = None)` gains the `table` argument and applies `to_latin` before `clean_text`

- [ ] **Step 1: Write the failing tests:**
    - `test_learn_table`: 6 toy pairs of (`लक्ष्मी कंसल्टेंसी प्राइवेट लिमिटेड`, `Lakshmi Consultancy Private Limited`) give `table["लक्ष्मी"] == "lakshmi"` and `table["कंसल्टेंसी"] == "consultancy"`.
    - Tokens seen fewer than 5 times are absent.
    - `test_mixed_script`: `to_latin("Digital सिस्टम्स", {"सिस्टम्स": "systems"}) == "digital systems"`.
    - `test_fallback_ascii`: `to_latin("क्रिएटिव", {})` is non-empty and ASCII.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the functions.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Coverage diagnostic only.** Learn a table from the full training GT and print its size and the share of Indic name tokens in train S2/S3 that it covers. Expected: coverage ≥ 90%. Log both numbers in `experiments.md`.
    - Do not save this table.
    - The production table is learned inside `run_all.py` (Task 14) from non-holdout pairs only, so the holdout stays unseen.
- [ ] **Step 6:** Commit: `feat: learned Indic token table with transliteration fallback`.

### Task 7: Address normalization and parsing (SPEC §6 step 4)

**Files:** Modify `src/lexicons.py` and `src/normalize.py`. Create `tests/test_normalize_address.py`.

**Interfaces:**
- Produces in `lexicons.py`:
    - `STREET_ABBR: dict[str,str]`, from SPEC step 4: st/saint/str→street, rd→road, ave/av→avenue, dr→drive, ct→court, ln→lane, blvd/bd→boulevard, r→rue, hn/h.no/h no→house, fl→floor
    - `US_STATES` (names ↔ 2-letter codes, 50 + DC)
    - `IN_STATES` (names, native-script names and codes such as RJ, MH, DL, HR, KA, OD ↔ canonical code)
    - `FR_DEPT_TO_REGION` (metropolitan départements → the 13 régions, lowercase and hyphenated)
- Produces in `normalize.py`:
    - `build_city_vocab(addresses: pd.Series, countries: pd.Series, min_count: int = 3) -> dict[str, set[str]]`. It collects comma chunks with no digits that are not a state or region and that appear in ≥ `min_count` S1 addresses of that country.
    - `parse_address(s: str, country: str, city_vocab: Mapping[str, set[str]]) -> dict`, with keys `addr_clean, addr_tokens (list[str]), postcode, house_nums (list[str]), street, city, state, has_addr`
    - `normalize_addresses(df: pd.DataFrame, city_vocab) -> pd.DataFrame`, which adds those columns

- [ ] **Step 1: Write the failing tests:**

  ```python
  V = {"US": {"indianapolis", "saint louis"}, "France": {"lille", "la teste-de-buch"}, "India": {"jaipur"}}
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
  ```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the functions.
    - **Abbreviation rule** (a plan decision that makes Review Focus 5 hold): expand a street abbreviation only when it is (a) the last token of a comma chunk that contains a digit or another street word, or (b) the token right after a house number.
    - **Postcodes:** 5 digits for US and France, 6 digits for India, and for any other country the longest run of 4–6 digits.
    - **City:** the chunk (or chunk suffix) found in `city_vocab[country]`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: address normalization and parsing`.

### Task 8: Holdout and transfer splits (SPEC §6 step 5)

**Files:** Create `src/splits.py`, `tests/test_splits.py`

**Interfaces:**
- Produces:
    - `make_holdout(s1: pd.DataFrame, match_counts: pd.Series, frac: float = 0.15, seed: int = 42) -> set[str]`. It stratifies by `country` × match-count bucket (0, 1, 2–3, 4–5, 6+).
    - `transfer_split(s1: pd.DataFrame, train_country: str, eval_country: str) -> tuple[set[str], set[str]]`
    - `save_split(ids, name)` and `load_split(name) -> set[str]`, stored in `CACHE_DIR/"splits"/f"{name}.txt"`

- [ ] **Step 1: Write the failing tests** on 10,000 synthetic S1 records:
    - the holdout size is within 15% ± 0.5%;
    - each country's share is within ±1% of its overall share;
    - the result is the same across two calls with seed 42;
    - `transfer_split` returns disjoint sets with pure countries.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `splits.py`, using `sklearn.model_selection.train_test_split` with `stratify`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Create and save `holdout`, `transfer_us_india` and `transfer_india_us` from the train cache.
- [ ] **Step 6:** Commit: `feat: holdout and country-transfer splits`.

### Task 9: Blocking (SPEC §6 step 6)

**Files:** Create `src/blocking.py`, `tests/test_blocking.py`

**Interfaces:**
- Consumes: the normalized frames from Tasks 5–7. Each record has `entity_id, source, country, name_sorted, name_clean, addr_*` columns.
- Produces:
    - Constants: `TOPN_NAME = 20`, `MIN_COS = 0.3`, `REVERSE_TOPN = 3`, `KEY_MAX = 200`, `CAP_PER_SOURCE = 40`, and `EMBED_ENABLED = False` (search D is off in v1).
    - `generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame`, with columns `s1_id, s23_id, source, search_mask (int: A=1, B=2, C=4), best_score (float32)`. It runs per country, internally.
    - `to_lists(cands: pd.DataFrame) -> dict[str, list[str]]`

- [ ] **Step 1: Write the failing tests** on a hand-made toy set of 6 S1 and 20 S2/S3 records, which includes the Porter & Nall cluster from SPEC §6 and a same-name decoy in another city.
    - `test_planted_pairs_found`: all 4 Porter & Nall matches are candidates.
    - `test_no_cross_country`: a US S1 never gets an India record.
    - `test_candidates_deduplicated`: each `(s1_id, s23_id)` appears once, even when searches A and B both find it (`search_mask == 3`).
    - `test_cap`: no S1 record has more than 40 candidates per source.
    - `test_empty_address_found_by_name`: `PORTER & NALL [LLC]`, which has no address, is found.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement searches A–C exactly as in the SPEC step 6 table.
    - A: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), sublinear_tf=True, min_df=2)`, fitted per country on S1+S2+S3 `name_sorted`, then `sp_matmul_topn` with `top_n=TOPN_NAME` and `threshold=MIN_COS`, in both directions.
    - B: address keys.
    - C: the rarest name word plus the city.
    - Merge the results, keep the top `CAP_PER_SOURCE` per `(s1_id, source)` by `best_score`, and drop any key group larger than `KEY_MAX`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Recall gate.** Run blocking for the holdout S1 against the full train S2/S3 pool, then compute `pair_recall` and `reduction_ratio` (Task 3).
    - Expected: `pair_recall >= 0.997`.
    - If it is lower: print a sample of 200 missed pairs grouped by cause, fix normalization or blocking, and re-run. Do not continue until the gate passes.
    - Log recall, reduction ratio and candidates per S1 in `experiments.md`.
- [ ] **Step 6:** Commit: `feat: multi-search blocking with recall gate`.

### Task 10: Sibling groups (SPEC §6 step 7)

**Files:** Create `src/siblings.py`, `tests/test_siblings.py`

**Interfaces:**
- Produces: `sibling_groups(s23: pd.DataFrame) -> pd.DataFrame`, with columns `entity_id, sib_group_id (int64), sib_group_size (int32), sib_best_addr (str)`. It works per `(country, source)`. It links two records when they have the same non-empty `addr_clean` and `fuzz.token_set_ratio(name_clean) >= 80`, **or** the same non-empty `name_key` and the same city. Connected components are found with scipy `connected_components`.

- [ ] **Step 1: Write the failing tests:**
    - `PORTER & NALL [LLC]` (no address) and `porternall.com` share a group, and `sib_best_addr` is `3220 gale street indianapolis in`;
    - an unrelated record with the same address but name similarity below 80 is in a different group;
    - records from S2 and S3 never share a group.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `sibling_groups`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: within-source sibling groups`.

### Task 11: Pair features (SPEC §6 step 8)

**Files:** Create `src/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: the Task 9 candidates, the normalized S1 and S2/S3 frames, and the Task 10 siblings.
- Produces:
    - `FEATURE_COLUMNS: list[str]`. It holds exactly the SPEC §6 step 8 features:
        - Name: `n_ratio, n_token_sort, n_token_set, n_partial, n_jaro_winkler, n_key_lev, n_tfidf_cos, n_idf_jaccard, n_min_shared_idf, n_unshared_cnt, n_alt_best`
        - Legal suffix: `legal_state` (0 same, 1 compatible, 2 conflict, 3 missing)
        - Script: `was_indic, n_sim_before_translit`
        - House number: `h_first_eq, h_any_shared, h_abs_gap, h_rel_gap, h_edit, h_cnt_s1, h_cnt_s23`
        - Address: `a_postcode_eq, a_city_eq, a_city_fuzzy, a_state_eq, a_street_set, a_token_set, a_idf_jaccard, a_has_s1, a_has_s23, a_sib_best_set`
        - Context: `c_rank_in_s1, c_gap_to_best, c_n_claimants, c_rank_among_claimants, c_name_freq, c_source_is_s3, c_sib_size, c_search_mask`
        - Compatible legal suffixes means one set is a subset of the other. Missing numeric values are NaN.
    - `compute_features(cands, s1n, s23n, sib, idf: Mapping[str,float], n_jobs: int = 16) -> pd.DataFrame`. It returns `s1_id, s23_id` plus `FEATURE_COLUMNS` as float32, and computes over 200k-row chunks with joblib.
    - `build_idf(names: pd.Series) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests:**
    - The Porter & Nall pair with `3220 Gale Street` vs `3220. Gale St` gives `h_first_eq == 1` and `a_street_set == 100`.
    - A decoy with `3228` gives `h_first_eq == 0` and `h_abs_gap == 8`.
    - `set(FEATURE_COLUMNS).isdisjoint({"country"})`, and no column name contains `country`.
    - The output row count equals the input candidate count.
    - `c_rank_in_s1` starts at 1 for the best `best_score`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement the functions with RapidFuzz scorers (`fuzz.*`, `distance.JaroWinkler`, `distance.Levenshtein`).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: pair features`.

### Task 12: LightGBM training and calibration (SPEC §6 step 9)

**Files:** Create `src/train.py`, `tests/test_train.py`

**Interfaces:**
- Produces:
    - `LGB_PARAMS`: `objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, deterministic=True, force_row_wise=True, num_threads=16`
    - `train_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, max_rounds: int = 3000) -> tuple[lgb.Booster, np.ndarray]`. It returns the final booster and out-of-fold scores.
        - Use 3-fold `GroupKFold` by `s1_id`, with early stopping at 100 on each fold's validation part.
        - Train the final model on all rows for the mean best iteration.
    - `fit_calibrator(oof: np.ndarray, y: np.ndarray) -> IsotonicRegression` (`out_of_bounds="clip"`)
    - `predict_proba(booster, cal, X) -> np.ndarray`
    - `save(booster, cal, tag: str)` and `load(tag) -> (booster, cal)`, stored in `MODELS_DIR/tag/`

- [ ] **Step 1: Write the failing tests** on 5,000 synthetic rows with signal in 2 features:
    - the OOF AUC is > 0.9;
    - calibrated probabilities are in [0, 1] and monotone in the raw score;
    - `save`/`load` round-trips identical predictions;
    - two runs with the same seed give identical predictions.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `train.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: LightGBM training with isotonic calibration`.

### Task 13: Decision layer (SPEC §7)

**Files:** Create `src/decide.py`, `tests/test_decide.py`

**Interfaces:**
- Produces:
    - `@dataclass DecisionParams`: `margin=0.15, t_empty=0.5, t_sib=0.5, lone_keep=0.95`. `t_sib=0.5` is only a starting point, because SPEC §7.3 tunes it and gives no start value.
    - `resolve_owners(scored: pd.DataFrame, margin: float) -> pd.DataFrame`. The input columns are `s1_id, s23_id, p`; the output keeps only the surviving claims.
    - `expected_f05(probs_desc: np.ndarray, k: int) -> float`. For k = 0 it returns `prod(1-p)`. For k > 0 it uses plug-in expectations: TP = sum of the top-k p, FP = k − TP, FN = sum of the remaining p, and F = 1.25·TP / (1.25·TP + 0.25·FN + FP).
    - `best_k(probs_desc: np.ndarray) -> int`
    - `decide(scored: pd.DataFrame, sib: pd.DataFrame, params: DecisionParams) -> dict[str, list[str]]`
    - `tune(scored: pd.DataFrame, sib, truth: Mapping[str,set[str]], s1_ids) -> DecisionParams`. It grid-searches `margin ∈ {0.05,0.1,0.15,0.2,0.3}`, `t_empty ∈ {0.3,0.4,0.5,0.6,0.7}` and `t_sib ∈ {0.3,0.5,0.7}` for the best `macro_f05`.

- [ ] **Step 1: Write the failing tests** (the values are computed by hand):

  ```python
  def test_best_k():
      assert best_k(np.array([0.9, 0.9, 0.1])) == 2
      assert best_k(np.array([0.3])) == 0
      assert best_k(np.array([0.95])) == 1
  def test_one_owner():
      df = pd.DataFrame({"s1_id": ["A","B","C","D"], "s23_id": ["x","x","y","y"], "p": [0.9,0.8,0.9,0.7]})
      out = resolve_owners(df, margin=0.15)
      assert set(zip(out.s1_id, out.s23_id)) == {("C","y")}   # x dropped (gap 0.1 < 0.15), y kept by C
  def test_t_empty():
      scored = pd.DataFrame({"s1_id": ["A"], "s23_id": ["x"], "p": [0.45]})
      assert decide(scored, EMPTY_SIB, DecisionParams())["A"] == []
  def test_lone_sibling_dropped():
      scored = pd.DataFrame({"s1_id": ["A"]*3, "s23_id": ["x1","x2","x3"], "p": [0.9, 0.05, 0.05]})
      sib = pd.DataFrame({"entity_id": ["x1","x2","x3"], "sib_group_id": [7,7,7], "sib_group_size": [3,3,3], "sib_best_addr": [""]*3})
      assert decide(scored, sib, DecisionParams())["A"] == []      # lone kept member, 0.9 < lone_keep 0.95
      scored.loc[0, "p"] = 0.97
      assert decide(scored, sib, DecisionParams())["A"] == ["x1"]  # 0.97 >= 0.95 survives
  ```

- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `decide` in this order:
    1. owners;
    2. per-S1 `best_k` on the probabilities sorted in descending order;
    3. `t_empty`;
    4. sibling consistency, where "most of a group kept" means > 50% of the group's members among this S1 record's candidates.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: F0.5 decision layer`.

### Task 14: Orchestration, integration test, v1 benchmark

**Files:** Create `src/run_all.py`, `tests/test_integration.py`. Modify `experiments.md`.

**Interfaces:**
- Consumes: every earlier task's interface.
- Produces:
    - The command line `run_all.py --split {train,test} [--limit-s1 N] [--model-tag TAG] [--train-sample 400000]`.
    - With `train`:
        - build the cache;
        - normalize and learn the Indic table (non-holdout only);
        - block;
        - compute siblings and features;
        - train on a sample of 400,000 non-holdout S1 records;
        - score the holdout;
        - tune the decision parameters;
        - save the model and `decision_params.json` in `MODELS_DIR/TAG`;
        - print the Task 3 `report`.
    - With `test`:
        - load TAG;
        - run everything per country, in feature/predict chunks of ≤2,000,000 pairs;
        - write `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

- [ ] **Step 1: Write the failing integration tests** (`@pytest.mark.integration`; skipped if `DATA_DIR` is missing):
    - `test_pipeline_slice`: `run_all --split train --limit-s1 5000` finishes. It writes both files for the slice. The organisers' `validate()` (with a temp `test_dir` holding the slice's source1 TSV) returns 0 errors. Every matched ID appears in the candidates.
    - `test_unseen_country_rows_present`: relabel 50 slice records to `"Atlantis"`. They still get exactly one row each in both files.
- [ ] **Step 2:** Run `$PY -m pytest -m integration -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `run_all.py`.
- [ ] **Step 4:** Run `$PY -m pytest -v`, the full suite including integration. Expected: all pass.
- [ ] **Step 5: v1 benchmark.**
    - Run `$PY src/run_all.py --split train --model-tag v1`.
    - Record the holdout macro F0.5 (overall, per country and for singletons), pair precision and recall, and blocking recall.
    - Also run the transfer check by repeating with training on US only and evaluating India.
    - Log everything in `experiments.md`.
- [ ] **Step 6:** Commit: `feat: end-to-end pipeline, v1 benchmark`, then `git tag model-v1`.
- [ ] **Step 7: Reminder (team request).** Tell the team the v1 benchmark numbers and ask whether to implement the optional step 10 re-ranker now. Do not build it without their yes.

### Task 15: First test submission (SPEC §8 step 13)

**Files:** Modify `experiments.md`.

- [ ] **Step 1:** Run `$PY src/check_env.py`. Expected: `PASS`.
- [ ] **Step 2:** Run `$PY src/run_all.py --split test --model-tag v1`. Expected: 1,732,544 data rows in each output file.
- [ ] **Step 3:** From `student_resource/`, run `python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test`. Expected: `PASS`, with no "matched IDs not present in candidate_pairs" warning.
- [ ] **Step 4:** Copy both files to `SUBMISSIONS_DIR/sub-20260926-1/`. Run `git tag sub-20260926-1`. The team uploads `matching_results.tsv` in the portal; this is manual, and 1 of 5 used today.
- [ ] **Step 5:** Log the tag, holdout score and leaderboard score in `experiments.md`, then commit.

### Task 16: Error-analysis round (SPEC §8 step 12), repeated per improvement

**Files:** Modify `src/evaluate.py`, `tests/test_evaluate.py` and `experiments.md`, plus whichever module the chosen fix touches.

**Interfaces:**
- Produces: `sample_errors(pred, truth, n: int = 100, seed: int = 42) -> pd.DataFrame`. It returns 50 wrong links and 50 missed links, weighted by F0.5 loss, with the raw S1 and S2/S3 text joined on.

- [ ] **Step 1: Write the failing test.** On toy data with known errors, the sample holds only real errors, in the 50/50 split, and is the same for the same seed.
- [ ] **Step 2:** Implement `sample_errors` until the test passes, then commit it.
- [ ] **Step 3 (each round):**
    - Sample errors from the latest holdout run and label each with a SPEC §8 cause.
    - Fix the largest cause test-first: add the failing case to the owning module's tests, fix it, then run the whole suite.
    - Re-run from the earliest affected stage and log the before/after scores in `experiments.md`.
    - Commit, and tag `model-vN` when the holdout improves.
- [ ] **Step 4 (France):** Compare the normal and conservative decision settings for countries not seen in training, using the transfer split. Submit both variants only if the budget allows (SPEC §8 plan, uploads 4–5).

### Task 17: Package the submission and methodology document (SPEC §8 step 14)

**Files:**
- Modify: `src/run_all.py` (add `--package`), `README.md`
- Create: `Documentation_template.md` (filled, in the repo root)

- [ ] **Step 1: Write the failing test** `test_package_layout`. `run_all.package(tmp_zip)` produces a zip with exactly these entries:
    - `output/matching_results.tsv` and `output/candidate_pairs.tsv`;
    - `code/business_entity_resolution/src/*`;
    - `README.md`, `requirements*.txt`, `models.lock.json`, `THIRD_PARTY_LICENSES.md` and `LICENSE`;
    - `Documentation_template.md`;
    - no parquet or model files, and no dataset files.
- [ ] **Step 2:** Implement `package()` with stdlib `zipfile`, writing `OUTPUT_DIR/"Barely_Legal_submission.zip"`. Make the test pass, then commit.
- [ ] **Step 3:** Complete `README.md` with the exact commands from a fresh machine to both output files, and the measured run time of each stage.
- [ ] **Step 4:** Fill in `Documentation_template.md`: about 2 pages of core sections plus an appendix.
    - Team: Barely Legal (Raghav Dokania, Moksh Chugh, Utkarsh Goel, Daksh Sajwan).
    - Include the SPEC EDA facts, the blocking numbers and features, the threshold method, the final holdout scores from `experiments.md`, and real examples of wrong and missed links.
- [ ] **Step 5:** Build the zip from the final chosen tag, check it with the Step 1 test, then commit and `git tag final`.
