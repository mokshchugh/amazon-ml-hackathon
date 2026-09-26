"""Smoke tests for the pinned core environment (SPEC section 9.2).

Each check is small and fast (seconds each): import every pinned package,
train LightGBM on synthetic data, score a name pair with RapidFuzz, run
sparse-dot-topn against a dense reference, and transliterate a Devanagari
string to Latin. Prints SMOKE PASS and exits 0 if every check succeeds;
raises on the first failure otherwise.
"""
from __future__ import annotations

import sys

import config


def smoke_import_versions() -> None:
    """Import every pinned core package and print its version."""
    import importlib

    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "joblib",
        "lightgbm",
        "rapidfuzz",
        "sparse_dot_topn",
        "indic_transliteration",
        "tqdm",
    ]
    for package_name in packages:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  {package_name} {version}")


def smoke_lightgbm() -> None:
    """Train LightGBM on 1,000 synthetic rows and expect AUC > 0.9."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(config.SEED)
    n_rows, n_features = 1000, 8
    X = rng.normal(size=(n_rows, n_features))
    weights = rng.normal(size=n_features)
    logits = X @ weights
    y = (logits + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)

    model = LGBMClassifier(
        n_estimators=50,
        random_state=config.SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    model.fit(X, y)
    predicted = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, predicted)
    assert auc > 0.9, f"LightGBM AUC too low: {auc}"


def smoke_rapidfuzz() -> None:
    """RapidFuzz token_set_ratio on the SPEC 6 example must be >= 95."""
    from rapidfuzz import fuzz

    score = fuzz.token_set_ratio("porter and nall", "porter and nall llc")
    assert score >= 95, f"RapidFuzz token_set_ratio too low: {score}"


def smoke_sparse_dot_topn() -> None:
    """sp_matmul_topn on a 100x100 TF-IDF matrix must match a dense argmax."""
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn

    rng = np.random.default_rng(config.SEED)
    vocab = [f"tok{i}" for i in range(30)]
    documents = [
        " ".join(rng.choice(vocab, size=6, replace=True)) for _ in range(100)
    ]
    tfidf = TfidfVectorizer().fit_transform(documents).tocsr().astype(np.float64)

    top_n = 5
    sparse_result = sp_matmul_topn(tfidf, tfidf.T.tocsr(), top_n=top_n, threshold=0.0)

    dense_similarity = (tfidf @ tfidf.T).toarray()
    dense_top1 = dense_similarity.argmax(axis=1)

    sparse_dense = sparse_result.toarray()
    sparse_top1 = sparse_dense.argmax(axis=1)

    assert np.array_equal(dense_top1, sparse_top1), (
        "sparse_dot_topn top-1 neighbours disagree with the dense computation"
    )


def smoke_indic_transliteration() -> None:
    """Devanagari to ITRANS transliteration must return a non-empty ASCII string."""
    from indic_transliteration import sanscript

    result = sanscript.transliterate("लक्ष्मी", sanscript.DEVANAGARI, sanscript.ITRANS)
    assert result, "transliteration returned an empty string"
    assert result.isascii(), f"transliteration is not ASCII: {result!r}"


def main() -> None:
    checks = [
        smoke_import_versions,
        smoke_lightgbm,
        smoke_rapidfuzz,
        smoke_sparse_dot_topn,
        smoke_indic_transliteration,
    ]
    for check in checks:
        print(f"[smoke_test] {check.__name__} ...")
        check()

    print("SMOKE PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
