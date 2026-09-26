# Experiments log

One entry per experiment or verification run: date, what was run, and the result.

## 2026-09-26 — Environment gate and smoke tests (online)

Ran `$PY src/check_env.py --write-report` and `$PY src/smoke_test.py` with
network available. Result: `PASS`, then `SMOKE PASS`. See
`.superpowers/sdd/2026-09-26-entity-resolution-v1/task-2-report.md` for full
output.

**Pending team action: offline proof (disable Wi-Fi, run check_env.py +
smoke_test.py, expect PASS / SMOKE PASS).**
