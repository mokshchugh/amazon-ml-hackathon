# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Barely Legal
**Team Members:** Raghav Dokania, Moksh Chugh, Utkarsh Goel, Daksh Sajwan
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We resolve each Source 1 business against Sources 2 and 3 with a classic blocking + classifier pipeline, run separately per country: script- and address-aware normalization, four-way blocking (name TF-IDF, address keys, rare word + city), about 40 RapidFuzz pair features, a calibrated LightGBM pair classifier and an F0.5-aware decision layer. On a held-out 15% of training Source 1 records, v1 reaches **macro F0.5 0.9705** (US 0.9778, India 0.9595) with pair precision 0.9925. Everything is built from the challenge data only, with MIT/Apache/BSD-licensed packages and no model larger than LightGBM.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 (US 60%, India 40%); test 1,732,544 / 4,887,273 / 5,082,316 (India 47%, US 38%, France 15%, a country absent from training).
- **Ground truth:** 7,638,365 train pairs. Every S2/S3 record belongs to at most one S1 record, which we use as a hard constraint.
- **Name noise:** typos, legal-suffix changes (Pvt Ltd / Private Limited, LLC / Inc), word reordering, web domains and "dba" names, and Indic scripts (Devanagari and others) mixed with Latin.
- **Address noise:** abbreviations (Rd/Road, St/Street), missing PIN or state, landmark references ("Near SBI ATM"), reordered components, French leading postcodes. About 3.4% of S2/S3 records have an empty address.
- **Decoys:** look-alike businesses with the same or very similar name at a different house number, street or city, and duplicates inside S2/S3.
- **Metric:** F0.5 per S1 record, then averaged. A wrong link costs about twice a missed one, and a record with no true match only scores when its prediction is empty.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier, with a metric-aware decision layer.
**Core Innovation:** (1) an Indic-token → Latin table learned from training ground-truth pairs (95.97% coverage of Indic name tokens in train S2/S3), with rule-based transliteration as the fallback; (2) within-source sibling groups, so records with empty addresses inherit their group's best address; (3) a decision layer that chooses, per S1 record, the list length that maximizes expected F0.5 under calibrated probabilities, with one-owner-per-record and sibling-consistency rules.

---

## 3. Candidate Generation (Blocking)
All searches run per country. There is no country feature anywhere, so France is handled by the same code path.

- **Blocking keys used:**
  - **A. Name TF-IDF:** character 3-gram TF-IDF on the sorted, cleaned name (sublinear TF), `sparse_dot_topn` top-20 per S1 per source at cosine ≥ 0.3, plus a reverse search (top 3 S1 per S2/S3 record).
  - **B. Address keys:** exact `postcode or city + first house number + first street word`, and a looser `city + street`. Keys shared by more than 200 records are skipped.
  - **C. Rare word + city:** inverted index on the highest-IDF name word combined with the city.
  - Candidates from all searches are merged, scored (identity 0.5, house 0.4, street 0.3, city 0.2, postcode 0.2) and capped at 60 per S1 record per source.
- **Candidate pairs generated:** 166,628,832 on test (mean about 95 per S1 record). The reduction ratio on the holdout is 0.99999.
- **How you ensured true matches were not lost:** recall was measured on the holdout against the full train S2/S3 pool (10.3M records). The first version with only the planned keys reached 0.870. Adding searches A–C, the reverse search and a larger cap raised it to **0.9717** (US 0.9812, India 0.9575). The remaining misses were grouped by cause (Section 5). `candidate_pairs.tsv` is exactly the set the model scores.

---

## 4. Matching Model

**Features used** (about 40 per pair, float32):
- **Name features:** RapidFuzz ratio, token_sort_ratio, token_set_ratio and partial_ratio; Jaro-Winkler; Levenshtein on the name key; char-3gram TF-IDF cosine; IDF-weighted word Jaccard; IDF of the rarest shared word; count of unshared words; best alternate-name score; legal suffix same / compatible / conflicting / missing; Indic-script flag and similarity before and after conversion.
- **Address features:** first house number equal, any number shared, absolute and relative numeric gap, number edit distance and counts; postcode equal; city equal or fuzzy; state equal; street token_set_ratio; whole-address token_set_ratio and IDF Jaccard; has-address flags; best sibling-address score.
- **Other (context):** the candidate's rank among its S1 record's candidates and its gap to the best score; how many S1 records claim the candidate and this record's rank among them; name frequency in the country; source (2 or 3); sibling group size; which searches found it (bitmask).

**Model type:** LightGBM binary classifier (`objective=binary`, lr 0.1, early stopping), trained on 200,000 non-holdout S1 records (18.9M pairs, 671k positive; 427 rounds, about 17 minutes on CPU), followed by isotonic calibration.
**Threshold selection method:** direct macro-F0.5 grid search on the holdout over the decision-layer parameters. (1) Each S2/S3 record keeps only its best S1 claim, and only if it beats the runner-up by margin `m`. (2) For each S1 record, keep the top-k that maximizes expected F0.5, treating calibrated probabilities as independent Bernoulli variables. (3) Predict nothing if the best probability is below `t_empty`. (4) Add sibling-group members with p ≥ `t_sib`; drop a lone sibling unless p ≥ `lone_keep`. Chosen values: m = 0.3, t_empty = 0.7, t_sib = 0.7, lone_keep = 0.95.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, holdout of 331,024 S1 records):** **0.9705**. By country: India 0.9595, US 0.9778; records with a single true match: 0.9661. Pair precision 0.9925, pair recall 0.9326. The candidates-only baseline (top-1 if the blocking score is ≥ 1.5) scores 0.6170.
- **Common false positives (wrong merges):** same-brand branches and chains, where the names match but the house number or street differs; and records whose candidate has an empty address, so only the name and its sibling group give evidence. The margin and one-owner rules exist mainly to suppress these.
- **Common false negatives (missed matches):** most recall loss happens at blocking (0.9711 of true pairs reach the model; the classifier and decision layer then keep 0.9326 of all pairs). Of the 32,449 holdout pairs that blocking missed: the S2/S3 record has an empty address but a similar name (11,205); same city and street but a crowded key where the candidate list is full (4,900 similar names, 3,101 very different names); the city is spelled differently (4,597); the street differs (2,484). 60% of the missed pairs belong to a list that is full at the per-source cap. India is harder than the US mainly because of transliterated names and landmark-style addresses.

---

## 6. Conclusion
A carefully normalized, multi-search blocking stage plus a calibrated gradient-boosted pair model and an F0.5-aware decision layer gets v1 to 0.9705 macro F0.5 on an honest holdout without any external data or large models. The main lesson is that blocking recall sets the ceiling: the next gains would come from a larger or smarter per-source cap for crowded address keys and from recovering empty-address records through their siblings, and after that from the optional small cross-encoder re-ranker for uncertain pairs.

---

## Appendix

### A. Code Artefacts
All code is in `code/business_entity_resolution/`:

| File | Role |
| --- | --- |
| `src/config.py` | all paths (`ER_WORK_DIR` sets the cache and model folder) and the seed |
| `src/io_utils.py` | TSV loading, parquet cache, writing ID-list outputs |
| `src/normalize.py`, `src/lexicons.py` | name and address cleaning and parsing, the Indic token table and transliteration |
| `src/splits.py` | holdout and country-transfer splits |
| `src/blocking.py` | candidate generation (searches A–C) |
| `src/siblings.py` | within-source sibling groups |
| `src/features.py` | pair features |
| `src/train.py` | LightGBM training and isotonic calibration |
| `src/decide.py` | F0.5 decision layer |
| `src/evaluate.py` | macro F0.5 and blocking metrics |
| `src/run_all.py` | end-to-end entry point and submission packaging |
| `src/check_env.py`, `src/smoke_test.py` | version and license gate, smoke tests |

Entry points, from the project root with the pinned venv active:

```
python code/business_entity_resolution/src/run_all.py --split train --model-tag v1   # train + holdout report
python code/business_entity_resolution/src/run_all.py --split test  --model-tag v1   # writes output/*.tsv
python code/business_entity_resolution/src/run_all.py --package                      # builds the zip
```

### B. Additional Results
- **Holdout blocking recall by per-source cap K:** None (60) 0.9711; 40 0.9651; 30 0.9600; 25 0.9540; 20 0.9360.
- **Run times (16-core laptop, 23.7 GB RAM):** train run 2,444 s in total (features 518 s, training 1,017 s, holdout scoring 584 s), peak RSS 16.3 GB. Blocking for the full train pool takes 2,486 s, peak RSS 17.0 GB.
- **Licenses:** every installed package has a permissive license (MIT, BSD, Apache-2.0, ISC, PSF, ZPL; tqdm and certifi are MPL-2.0 and used unmodified), checked by `check_env.py` and listed in `THIRD_PARTY_LICENSES.md`. No external lookups or data are used.
