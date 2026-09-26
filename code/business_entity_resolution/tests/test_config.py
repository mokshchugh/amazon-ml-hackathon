def test_paths_and_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("ER_WORK_DIR", str(tmp_path / "w"))
    import importlib, config; importlib.reload(config)
    assert config.SEED == 42
    assert config.DATA_DIR.name == "dataset" and config.DATA_DIR.parent.name == "student_resource"
    assert config.CACHE_DIR == tmp_path / "w" / "cache"
    config.ensure_dirs()
    assert config.CACHE_DIR.is_dir() and config.MODELS_DIR.is_dir() and config.SUBMISSIONS_DIR.is_dir()
