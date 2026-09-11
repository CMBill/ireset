"""Central path constants for the ireset project.

All resource access goes through these constants so that read-only source
assets and derived outputs never mix. Derived data must be written only to
``derived/``; the resources under ``resources/`` are inputs (the OMOMO entry
is a symlink into source data owned outside this repo).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCES_DIR = PROJECT_ROOT / "resources"
ROBOTS_DIR = RESOURCES_DIR / "robots"
DATA_DIR = RESOURCES_DIR / "data"

# Read-only source assets (never write into these).
G1_HOLOSOMA_DIR = ROBOTS_DIR / "g1_holosoma"
LARGEBOX_DIR = ROBOTS_DIR / "largebox"
OMOMO_NEW_DIR = DATA_DIR / "OMOMO_new"  # symlink -> external source data
OMNIRETARGET_DIR = DATA_DIR / "omniretarget"
TEACHER_TRAJ_DIR = OMNIRETARGET_DIR / "robot-object"
TEACHER_MODELS_DIR = OMNIRETARGET_DIR / "models"

# Derived outputs (writable).
DERIVED_DIR = PROJECT_ROOT / "derived"
DERIVED_ROBOTS_DIR = DERIVED_DIR / "robots"
DERIVED_DATA_DIR = DERIVED_DIR / "data"
