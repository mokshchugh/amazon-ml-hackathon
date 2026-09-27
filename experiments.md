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

## 2026-09-27 — Task 9 blocking recall gate (SPEC step 6), commit 26d548b

`generate_candidates(full train S1, full train S2+S3)` with the gate script
`.superpowers/sdd/2026-09-26-entity-resolution-v1/t9_gate.py run`; recall on
holdout S1 (331,024 records, 1,146,208 true pairs) against the full train
S2/S3 pool (10,320,219 records). Token table from non-holdout GT only.

- **Pair recall: 0.97169 overall** (US 0.98116, India 0.95751) — target
  0.997 **not met**
- Reduction ratio: 0.9999908 (US 0.9999848, India 0.9999768)
- Candidates per S1: mean 94.6, p50 111, p95 120 (per S1 per source: mean
  47.3, p95 60)
- Runtime 2486 s for generate_candidates; peak RSS 17.0 GiB
- Constants: TOPN_NAME 20, MIN_COS 0.3, REVERSE_TOPN 3, KEY_MAX 200,
  CAP_PER_SOURCE 60 (plan: 40), EMBED_ENABLED False; search-A stage-1
  A_QUERY_K 5 / A_STAGE1_TOPN 60, A_REV_QUERY_K 8 / A_STAGE1_REV 20;
  score weights ident 0.5, house 0.4, street 0.3, city 0.2, postcode 0.2
- Previous attempt (run3, plan keys only, cap 40): recall 0.86969
  (US 0.91391, India 0.80349), 52.8 candidates per S1
- Remaining 32,449 misses: s23 empty address / name similar 11,205; same
  city+street but crowded (name similar 4,900, very different 3,101);
  city differs / name similar 4,597; street differs / name similar 2,484;
  others < 1,500 each. 60% of missed pairs belong to an (S1, source) list
  that is full at the cap.
