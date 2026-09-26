# Experiments log

One entry per experiment or verification run: date, what was run, and the result.

## 2026-09-26 — Environment gate and smoke tests (online)

Ran `$PY src/check_env.py --write-report` and `$PY src/smoke_test.py` with
network available. Result: `PASS`, then `SMOKE PASS`. See
`.superpowers/sdd/2026-09-26-entity-resolution-v1/task-2-report.md` for full
output.

**Pending team action: offline proof (disable Wi-Fi, run check_env.py +
smoke_test.py, expect PASS / SMOKE PASS).**

## 2026-09-26 — Task 6 Indic token table coverage (diagnostic, not saved)

Learned `learn_token_table` from the full training ground truth (all
`gt_pairs` joined with `train_source1` + `train_source2`/`source3`
`business_name`, default `min_count=5`, `min_share=0.8`), then measured
coverage against every Indic token in train S2+S3 `business_name` values.

- Table size: 1467 entries
- Learn runtime: 97.55 s
- Coverage: 2,735,046 / 2,849,756 Indic name tokens in train S2+S3 =
  **95.97%** (target >= 90%, met)
- Total diagnostic runtime (learn + coverage scan): 105.74 s

This table was not saved; the production table is learned inside
`run_all.py` (Task 14) from non-holdout pairs only.
