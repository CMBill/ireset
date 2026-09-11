"""Validator tests: known violations must be detected; clean data must pass.

Uses one real teacher clip as the clean baseline, then injects each violation
class from plan section 10.1 / M1 acceptance ("known violation samples can be detected").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_data import convert_scene_qpos
from ireset.paths import TEACHER_TRAJ_DIR
from ireset.repair.validation import IndependentValidator, ValidationConfig

CLIP = TEACHER_TRAJ_DIR / "sub10_largebox_000_original.npz"


@pytest.fixture(scope="module")
def teacher_qpos() -> np.ndarray:
    q = np.load(CLIP, allow_pickle=True)["qpos"]
    return np.stack([convert_scene_qpos(row) for row in q])


@pytest.fixture(scope="module")
def validator() -> IndependentValidator:
    return IndependentValidator(cfg=ValidationConfig())


def test_clean_teacher_baseline_passes_geometry(teacher_qpos, validator) -> None:
    """Joint limits + penetration must pass on the real teacher clip.

    Teacher penetration was measured ~<=8mm in M0, so the full report may
    legitimately fail the strict 2mm gate; the geometry checks themselves
    must succeed and record finite worst values.
    """
    rep = validator.validate(teacher_qpos)
    assert rep.checks["joint_limits"].passed
    assert np.isfinite(rep.checks["penetration"].worst)
    assert rep.max_check_interval_m >= 0


def test_detects_penetration(teacher_qpos, validator) -> None:
    q = teacher_qpos.copy()
    t = q.shape[0] // 2
    # shove the box into the robot pelvis
    q[t:, 36:39] -= q[t:, 36:39] - q[t, 0:3]  # box position -> robot root pos
    rep = validator.validate(q[: t + 4])
    assert not rep.checks["penetration"].passed
    assert rep.checks["penetration"].worst > 0.002


def test_detects_joint_limit_violation(teacher_qpos, validator) -> None:
    q = teacher_qpos.copy()
    q[:, 7] = validator.spec.joint_range[0, 1] + 0.5  # left_hip_pitch beyond upper
    rep = validator.validate(q[:10])
    assert not rep.checks["joint_limits"].passed
    assert rep.checks["joint_limits"].worst >= 0.5 - 1e-9


def test_detects_excessive_velocity(teacher_qpos, validator) -> None:
    q = teacher_qpos.copy()
    t = 40
    q[t + 1, 7] = q[t, 7] + 1.0  # ~30 rad/s step in one frame
    rep = validator.validate(q[t : t + 3])
    assert not rep.checks["velocities"].passed


def test_detects_foot_slide_during_support(teacher_qpos, validator) -> None:
    """Horizontal root drift during inferred sole contact must trip foot checks.

    We translate the whole robot horizontally over a support segment (box
    fixed), which slides the feet while keeping them on the ground.
    """
    q = teacher_qpos.copy()
    rep0 = validator.validate(q)
    supports = [iv for iv in rep0.schedule.intervals if iv.support == "ground"]
    assert supports, "expected at least one inferred support interval"
    iv = supports[0]
    T = iv.end - iv.start
    drift = np.linspace(0.0, 0.2, T)[:, None] * np.array([[1.0, 0.0, 0.0]])
    q[iv.start : iv.end, 0:3] += drift
    rep = validator.validate(q)
    assert not rep.checks["foot_support"].passed


def test_detects_nonfinite(teacher_qpos, validator) -> None:
    q = teacher_qpos.copy().astype(float)
    q[10, 20] = np.nan
    # NaN must never produce a passing report (comparisons with NaN are False)
    try:
        rep = validator.validate(q)
    except (ValueError, FloatingPointError):
        pytest.skip("MuJoCo rejected NaN qpos directly, equally valid")
    assert not rep.passed
