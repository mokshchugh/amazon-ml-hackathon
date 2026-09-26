"""Central configuration: paths and seed for the entity-resolution pipeline.

All paths used elsewhere in this codebase must come from here.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "code" / "business_entity_resolution"

DATA_DIR = REPO_ROOT / "student_resource" / "dataset"
OUTPUT_DIR = REPO_ROOT / "output"

WORK_DIR = Path(os.environ.get("ER_WORK_DIR", r"C:\Users\utkar\er_work"))
CACHE_DIR = WORK_DIR / "cache"
MODELS_DIR = WORK_DIR / "models"
SUBMISSIONS_DIR = WORK_DIR / "submissions"

SEED = 42


def ensure_dirs() -> None:
    """Create the work and output directories if they do not already exist."""
    for directory in (CACHE_DIR, MODELS_DIR, SUBMISSIONS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
