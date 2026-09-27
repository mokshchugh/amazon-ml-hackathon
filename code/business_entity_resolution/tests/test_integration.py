"""End-to-end integration tests on a real-data slice (Task 14)."""
import shutil
import sys

import pytest

import config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not config.DATA_DIR.exists(), reason="real dataset (DATA_DIR) missing"),
]

N_SLICE = 5000


def _read_lists(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, ids = line.rstrip("\n").split("\t")
            rows.setdefault(s1, []).append(set(ids.split(",")) - {""})
    return rows


def _validate(out, tmp_path):
    sys.path.insert(0, str(config.REPO_ROOT / "student_resource" / "utils"))
    from validate_submission import validate

    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    shutil.copy(out / "source1_slice.tsv", test_dir / "test_source1.tsv")
    return validate(str(out / "matching_results.tsv"), str(out / "candidate_pairs.tsv"), str(test_dir))


def test_pipeline_slice(tmp_path):
    import run_all

    out = tmp_path / "out"
    run_all.main(["--split", "train", "--limit-s1", str(N_SLICE), "--model-tag", "itest_slice",
                  "--out-dir", str(out)])
    assert (out / "candidate_pairs.tsv").exists() and (out / "matching_results.tsv").exists()
    errors, _warnings = _validate(out, tmp_path)
    assert errors == []
    matched, cands = _read_lists(out / "matching_results.tsv"), _read_lists(out / "candidate_pairs.tsv")
    assert len(matched) == N_SLICE and len(cands) == N_SLICE
    for s1, [m] in matched.items():
        assert m <= cands[s1][0], s1
    assert sum(1 for [m] in matched.values() if m) > 0


def test_unseen_country_rows_present(tmp_path):
    import run_all

    relabeled = []

    def atlantis(s1):
        s1 = s1.copy()
        idx = s1.index[:50]
        relabeled.extend(s1.loc[idx, "entity_id"].astype(str))
        s1["country"] = s1["country"].astype(object)
        s1.loc[idx, "country"] = "Atlantis"
        s1["country"] = s1["country"].astype("string[pyarrow]")
        return s1

    out = tmp_path / "out"
    run_all.main(["--split", "train", "--limit-s1", str(N_SLICE), "--model-tag", "itest_atlantis",
                  "--out-dir", str(out)], slice_hook=atlantis)
    assert len(relabeled) == 50
    errors, _warnings = _validate(out, tmp_path)
    assert errors == []
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        rows = _read_lists(out / name)
        for s1 in relabeled:
            assert len(rows.get(s1, [])) == 1, (name, s1)
            assert rows[s1][0] == set(), (name, s1)  # no Atlantis S2/S3 -> no candidates, no matches
