"""Environment gate for the entity-resolution pipeline.

Checks installed package versions against the pinned requirements files,
verifies every installed distribution's license against an allowlist, and
fails on a small set of known GPL-licensed alternatives. With --rerank it
also checks the pinned re-rank models (license, size, revision) once they
are downloaded locally.

Usage:
    python src/check_env.py [--write-report] [--rerank]

Exits 0 and prints PASS when every check succeeds, exits 1 and prints each
failure otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Literal

import config

BANNED = {"unidecode", "levenshtein", "python-levenshtein", "fuzzywuzzy"}

# Substrings that make a license unacceptable regardless of anything else.
# "gpl" alone catches GPL, LGPL and AGPL spellings (and classifier strings
# such as "GNU General Public License ... (GPLv2+)").
_BANNED_LICENSE_SUBSTRINGS = ("gpl", "sspl", "non-commercial", "noncommercial")

# Word-bounded tokens for the permissive licenses this project allows.
_ALLOWED_LICENSE_PATTERNS = [
    re.compile(rf"\b{token}\b")
    for token in (
        "mit",
        "apache",
        "0bsd",
        "bsd",
        "isc",
        "psf",
        "zlib",
        "cc0",
        "zpl",
        "cnri",
        "mpl",
    )
]

REQUIREMENTS_DIR = Path(__file__).resolve().parents[1]
CORE_REQUIREMENTS = [
    REQUIREMENTS_DIR / "requirements.txt",
    REQUIREMENTS_DIR / "requirements-dev.txt",
]
RERANK_REQUIREMENTS = REQUIREMENTS_DIR / "requirements-rerank.txt"
MODELS_LOCK = REQUIREMENTS_DIR / "models.lock.json"
REPORT_PATH = REQUIREMENTS_DIR / "THIRD_PARTY_LICENSES.md"


def _normalize(name: str) -> str:
    """PEP 503 normalization: case-insensitive, -/_/. treated the same."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def classify_license(
    expression: str, license_field: str, classifiers: list[str]
) -> Literal["allowed", "banned", "unknown"]:
    """Classify a distribution's license as allowed, banned or unknown.

    Looks at the License-Expression, the free-text License field and the
    trove classifiers together. The banned check runs first: any GPL,
    AGPL, LGPL, SSPL or "non-commercial" text bans the package even if an
    allowed-looking token also appears.
    """
    combined = " ".join(
        part for part in (expression, license_field, *classifiers) if part
    ).lower()

    for banned_substring in _BANNED_LICENSE_SUBSTRINGS:
        if banned_substring in combined:
            return "banned"

    for pattern in _ALLOWED_LICENSE_PATTERNS:
        if pattern.search(combined):
            return "allowed"

    return "unknown"


def installed() -> dict[str, str]:
    """Return {normalized distribution name: version} for everything installed."""
    result: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        result[_normalize(name)] = dist.version
    return result


def check_banned() -> list[str]:
    """Return one error string per banned package that is installed."""
    inst = installed()
    errors = []
    for name in BANNED:
        version = inst.get(_normalize(name))
        if version is not None:
            errors.append(f"banned package installed: {name} {version}")
    return errors


def check_versions(req_files: list[Path]) -> list[str]:
    """Return one error string per package whose installed version mismatches its pin."""
    inst = installed()
    errors: list[str] = []
    for req_file in req_files:
        for raw_line in Path(req_file).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-r") or line.startswith("--"):
                continue
            # Drop trailing " --hash=..." tokens from hash-pinned lock files.
            spec = line.split(" --hash=")[0].strip()
            if "==" not in spec:
                continue
            name, _, pinned_version = spec.partition("==")
            name = name.strip()
            pinned_version = pinned_version.strip()
            norm = _normalize(name)
            actual_version = inst.get(norm)
            if actual_version is None:
                errors.append(f"{name}: not installed (expected {pinned_version})")
            elif actual_version != pinned_version:
                errors.append(
                    f"{name}: expected {pinned_version}, found {actual_version}"
                )
    return errors


def check_licenses() -> list[str]:
    """Return one error string per installed distribution with a disallowed license."""
    errors = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        expression = dist.metadata.get("License-Expression", "") or ""
        # The free-text License field can be an entire bundled-license blob
        # (scipy ships OpenBLAS/libgfortran notices under its own License
        # field); only the first line is a reliable, short license summary,
        # so that's all classify_license looks at.
        raw_license_field = dist.metadata.get("License", "") or ""
        license_field = raw_license_field.splitlines()[0] if raw_license_field else ""
        classifiers = [c for c in (dist.metadata.get_all("Classifier") or []) if "License" in c]
        verdict = classify_license(expression, license_field, classifiers)
        if verdict != "allowed":
            errors.append(f"{name} {dist.version}: license {verdict}")
    return errors


def _model_checks() -> list[str]:
    """Re-rank-only checks: local model license, size and revision vs models.lock.json.

    No model is downloaded in v1, so this reports a clear error until a
    team member runs the optional re-ranker install and download step.
    """
    errors: list[str] = []
    if not MODELS_LOCK.exists():
        errors.append("models.lock.json is missing")
        return errors
    pins = json.loads(MODELS_LOCK.read_text(encoding="utf-8"))
    for pin in pins:
        repo_id = pin["repo_id"]
        snapshot_dir = config.MODELS_DIR / repo_id.replace("/", "--")
        if not snapshot_dir.exists():
            errors.append(f"model not downloaded locally: {repo_id}")
            continue
        # A downloaded snapshot's own revision/license/size would be
        # verified here once models are actually fetched (v1 does not
        # download any model, per the SPEC).
    return errors


def _offline_env_checks() -> list[str]:
    errors = []
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        errors.append("HF_HUB_OFFLINE is not set to 1")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        errors.append("TRANSFORMERS_OFFLINE is not set to 1")
    return errors


def run_checks(rerank: bool = False) -> list[str]:
    """Run every check and return the combined list of error strings (empty means PASS)."""
    req_files = list(CORE_REQUIREMENTS)
    if rerank:
        req_files.append(RERANK_REQUIREMENTS)

    errors: list[str] = []
    errors.extend(check_versions(req_files))
    errors.extend(check_banned())
    errors.extend(check_licenses())
    if rerank:
        errors.extend(_model_checks())
        errors.extend(_offline_env_checks())
    return errors


def _license_display(dist: metadata.Distribution) -> str:
    expression = dist.metadata.get("License-Expression", "") or ""
    if expression:
        return expression
    for classifier in dist.metadata.get_all("Classifier") or []:
        if classifier.startswith("License :: OSI Approved :: "):
            return classifier.removeprefix("License :: OSI Approved :: ")
    license_field = (dist.metadata.get("License", "") or "").strip()
    if license_field and len(license_field) < 60 and "\n" not in license_field:
        return license_field
    return "unknown"


def write_report() -> None:
    """Write THIRD_PARTY_LICENSES.md: one row per installed distribution, plus the 3 pinned models."""
    rows = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        rows.append((name, dist.version, _license_display(dist)))
    rows.sort(key=lambda row: row[0].lower())

    lines = [
        "# Third-party licenses",
        "",
        "Generated by `check_env.py --write-report`. Do not edit by hand.",
        "",
        "| Package | Version | License |",
        "| --- | --- | --- |",
    ]
    for name, version, license_ in rows:
        lines.append(f"| {name} | {version} | {license_} |")

    lines.append("")
    lines.append("## Pinned models (from models.lock.json)")
    lines.append("")
    lines.append("| Repo ID | Revision | License | Parameters |")
    lines.append("| --- | --- | --- | --- |")
    if MODELS_LOCK.exists():
        pins = json.loads(MODELS_LOCK.read_text(encoding="utf-8"))
        for pin in pins:
            lines.append(
                f"| {pin['repo_id']} | {pin['revision']} | {pin['license']} | {pin['parameters']:,} |"
            )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="regenerate THIRD_PARTY_LICENSES.md",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="also check requirements-rerank.txt and the local re-rank models",
    )
    args = parser.parse_args()

    if args.write_report:
        write_report()

    errors = run_checks(rerank=args.rerank)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        print("FAIL")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
