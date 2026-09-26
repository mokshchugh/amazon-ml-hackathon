import pytest

import check_env


@pytest.mark.parametrize(
    "expr,field,cls,want",
    [
        ("MIT", "", [], "allowed"),
        ("", "BSD 3-Clause License", ["License :: OSI Approved :: BSD License"], "allowed"),  # pandas
        (
            "",
            "Copyright (c) 2001-2002 Enthought, Inc.",
            ["License :: OSI Approved :: BSD License"],
            "allowed",
        ),  # scipy
        ("MPL-2.0 AND MIT", "", [], "allowed"),  # tqdm
        ("", "Apache 2.0 License", [], "allowed"),  # transformers
        ("Apache-2.0 AND CNRI-Python", "", [], "allowed"),  # regex
        ("GPL-2.0-or-later", "", [], "banned"),
        (
            "",
            "",
            ["License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)"],
            "banned",
        ),
        ("", "", [], "unknown"),
    ],
)
def test_classify_license(expr, field, cls, want):
    assert check_env.classify_license(expr, field, cls) == want


def test_banned_names_fail(monkeypatch):
    monkeypatch.setattr(check_env, "installed", lambda: {"unidecode": "1.4.0"})
    assert any("unidecode" in e for e in check_env.check_banned())
