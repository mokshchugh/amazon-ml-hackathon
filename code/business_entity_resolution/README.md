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
