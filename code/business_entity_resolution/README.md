# Business Entity Resolution

## Environment

The pipeline runs on Python 3.14.3 in a dedicated venv outside OneDrive.

```
py -3.14 -m venv C:\Users\utkar\venvs\amazon-er
C:\Users\utkar\venvs\amazon-er\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r code/business_entity_resolution/requirements.txt
# optional, only if the re-ranker is enabled
python -m pip install -r code/business_entity_resolution/requirements-rerank.txt
python -m pip check
```

Before running anything else, verify the environment matches the pinned
requirements and that every installed license is permissive:

```
python src/check_env.py --write-report
```

This exits 0 and prints `PASS` when every installed package matches its pin
in `requirements.txt` / `requirements-dev.txt` (add `--rerank` to also check
`requirements-rerank.txt` and the local re-rank models), none of the banned
packages (`unidecode`, `Levenshtein`, `python-Levenshtein`, `fuzzywuzzy`) is
installed, and every installed distribution's license is on the allowlist.
It also regenerates `THIRD_PARTY_LICENSES.md` with `--write-report`.

Run `python src/smoke_test.py` to confirm LightGBM, RapidFuzz, sparse-dot-topn
and indic_transliteration all work end to end; it prints `SMOKE PASS`.

## Reproduce both output files

Put the organiser data in `student_resource/dataset/{train,test}/` (the
`.tsv` files as downloaded). Set `ER_WORK_DIR` to a folder outside any synced drive
for the parquet caches and models (the default is `C:\Users\utkar\er_work`).
From the project root, with the venv active:

```
python code/business_entity_resolution/src/check_env.py
python code/business_entity_resolution/src/run_all.py --split train --model-tag v1
python code/business_entity_resolution/src/run_all.py --split test --model-tag v1
cd student_resource
python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
cd ..
python code/business_entity_resolution/src/run_all.py --package
```

- `--split train` builds the caches, blocks candidates, trains LightGBM on
  200,000 non-holdout S1 records, tunes the decision layer on the holdout and
  saves the model to `ER_WORK_DIR/models/v1`. Measured on the team laptop
  (16 cores, 23.7 GB RAM): 2,444 s in total (features 518 s, training
  1,017 s, holdout scoring 584 s, other stages under 3 minutes), peak RSS
  16.3 GB, once the blocking caches exist. Building the blocking caches for the
  full train pool took 2,486 s on its own (peak RSS 17.0 GB).
- `--split test` writes `output/candidate_pairs.tsv` and
  `output/matching_results.tsv` (every test S1 id present).
  `--max-cands-per-source K` trims candidates for speed. `--baseline` writes
  the candidates-only insurance submission instead (773 s in total).
- `--package` builds `output/Barely_Legal_submission.zip` with the two output
  files, this folder's `src/`, README, requirements, locks and licenses, and the
  filled `Documentation_template.md` from the project root.

Every run appends its stage timings to `experiments.md`.
