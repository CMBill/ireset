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


def test_detects_foot_slide_during_support(validator) -> None:
    """A SLOW horizontal slide during inferred sole contact must trip the
    drift check. P0.1 L1 semantics: support := quasi-static (entry speed gate
    0.02 == check gate), so fast slides are excluded from support intervals
    (transition, uncertified) and only sub-gate-speed slides stay inside --
    those are caught by the 1cm anchor-drift gate.
    """
    from test_contacts import _root_height_for_tangent

    T = 45
    h = _root_height_for_tangent(validator)
    q = np.zeros((T, 43))
    q[:, 0:3] = [0.1, 0.0, h]
    q[:, 3] = 1.0
    q[:, 36:43] = [1.5, 0.0, 0.19, 1.0, 0, 0, 0]
    q[:, 0] += np.linspace(0.0, 0.013, T)   # 0.0087 m/s < 0.02, drift 13mm > 10mm
    rep = validator.validate(q)
    supports = [iv for iv in rep.schedule.intervals if iv.support == "ground"]
    assert supports, "slow slide stays inside inferred support"
    assert not rep.checks["foot_support"].passed


def test_detects_nonfinite(teacher_qpos, validator) -> None:
    """NaN joints: clean rejection with an explicit status, never a crash."""
    q = teacher_qpos.copy().astype(float)
    q[10, 20] = np.nan
    rep = validator.validate(q)
    assert rep.status == "rejected_nonfinite"
    assert not rep.passed
    assert rep.checks["finite"].location.startswith("t=")


def test_detects_nonfinite_box_quaternion(teacher_qpos, validator) -> None:
    """NaN box quat used to crash scipy Slerp; must be a clean rejection."""
    q = teacher_qpos.copy().astype(float)
    q[5, 40] = np.nan
    rep = validator.validate(q)
    assert rep.status == "rejected_nonfinite"
    assert not rep.passed


def test_detects_zero_norm_quaternion(teacher_qpos, validator) -> None:
    q = teacher_qpos.copy().astype(float)
    q[3, 3:7] = 0.0
    rep = validator.validate(q)
    assert rep.status == "rejected_degenerate_quaternion"
    assert not rep.passed


def test_adaptive_subdivision_reports_post_interval(teacher_qpos, validator) -> None:
    """Fast frame pairs subdivide; max_check_interval_m is post-subdivision."""
    q = teacher_qpos[:6].copy()
    rep = validator.validate(q)
    cfg = validator.cfg
    motion = validator._frame_motion_m(q[2], q[3])
    if motion > cfg.max_substep_motion_m:
        assert rep.max_check_interval_m < motion
    else:
        assert rep.max_check_interval_m <= cfg.max_substep_motion_m + 1e-9


def test_mid_frame_penetration_caught_by_subdivision(teacher_qpos, validator) -> None:
    """Penetration only between two clean keyframes must still be detected."""
    q = teacher_qpos[:3].copy()
    root = q[1, 0:3]
    far = root + np.array([2.0, 0.0, 0.0])
    hit = root.copy()
    s_star = 0.35
    a = (hit - s_star * far) / (1 - s_star)
    q[0, 36:39] = a
    q[1, 36:39] = far
    q[0, 39:43] = [1, 0, 0, 0]
    q[1, 39:43] = [1, 0, 0, 0]
    rep = validator.validate(q)
    assert not rep.checks["penetration"].passed, (
        f"interior penetration missed (worst={rep.checks['penetration'].worst})"
    )


# --- P0-fix F1: hand inference speed lives in the BOX frame ---------------------


def _box_beside_left_hand(validator: IndependentValidator, q43: np.ndarray) -> np.ndarray:
    """Box pose whose near face sits 2-8 mm from the LEFT hand geoms,
    placed beyond the hand along +x (nothing else reaches that far)."""
    import mujoco

    d = mujoco.MjData(validator.model)
    d.qpos = q43
    mujoco.mj_forward(validator.model, d)
    hg = validator._hand_geoms["l"]
    p = d.geom_xpos[hg].mean(0)
    for off in np.linspace(0.02, 0.60, 59):
        q = q43.copy()
        q[36:39] = p + np.array([off, 0.0, 0.0])
        q[39:43] = [1.0, 0, 0, 0]
        d.qpos = q
        mujoco.mj_forward(validator.model, d)
        dist = min(mujoco.mj_geomDistance(validator.model, d, g, validator.box_geom,
                                          1.0, np.zeros(6)) for g in hg)
        if 0.002 < dist < 0.008:
            return q[36:43].copy()
    raise AssertionError("no near-touch box placement found beside the left hand")


def _hand_box_scene(validator: IndependentValidator, T: int,
                    v_common: float = 0.0, drift_per_frame: np.ndarray | None = None,
                    yaw_rate: float = 0.0) -> np.ndarray:
    """Standing scene with the box just beyond the left hand (+x side).

    v_common: rigid common translation speed (m/s, +x) applied to robot
    root AND box. yaw_rate: rigid common yaw (rad/s) about the box center,
    rotating root and box together. drift_per_frame: [T] box-only +y
    offsets -- tangential to the near face, so the gap stays in-band while
    the box-frame hand position drifts.
    """
    from test_contacts import _root_height_for_tangent

    h = _root_height_for_tangent(validator)
    q0 = np.zeros(43)
    q0[2] = h
    q0[3] = 1.0
    box0 = _box_beside_left_hand(validator, q0)
    t = np.arange(T)
    scene = np.zeros((T, 43))
    scene[:, 3] = 1.0
    scene[:, 0:3] = [0.0, 0.0, h]
    if yaw_rate == 0.0:
        shift = np.stack([v_common * t / 30.0, np.zeros(T), np.zeros(T)], axis=1)
        scene[:, 0:3] += shift
        scene[:, 36:39] = box0[None, 0:3] + shift
        scene[:, 39:43] = box0[3:7]
    else:
        # rigid common rotation about the fixed box center: hand_local is
        # invariant by construction (quats are w-first; yaw lives in z)
        ang = yaw_rate * t / 30.0
        c0 = box0[0:3]
        rel = np.array([0.0, 0.0, h]) - c0
        scene[:, 0] = c0[0] + rel[0] * np.cos(ang) - rel[1] * np.sin(ang)
        scene[:, 1] = c0[1] + rel[0] * np.sin(ang) + rel[1] * np.cos(ang)
        scene[:, 2] = c0[2] + rel[2]
        scene[:, 3] = np.cos(ang / 2)
        scene[:, 6] = np.sin(ang / 2)
        scene[:, 36:39] = c0
        scene[:, 39] = np.cos(ang / 2)
        scene[:, 42] = np.sin(ang / 2)
    if drift_per_frame is not None:
        scene[:, 37] += drift_per_frame
    return scene


def test_hand_grasp_survives_common_translation(validator) -> None:
    """Review probe (P0-fix F1): hands translated WITH the box at 0.3 m/s
    are quasi-static in the box frame -- grasp intervals must not vanish,
    and the inference (intervals, box-frame anchors, stats) must equal the
    static case exactly."""
    T = 12
    sched_static = validator._infer_robot_schedule(_hand_box_scene(validator, T))
    sched_move = validator._infer_robot_schedule(_hand_box_scene(validator, T, v_common=0.3))
    l_static = [iv for iv in sched_static.intervals if iv.effector == "left_hand"]
    l_move = [iv for iv in sched_move.intervals if iv.effector == "left_hand"]
    assert l_static and l_move, (len(l_static), len(l_move))
    assert (l_static[0].start, l_static[0].end) == (l_move[0].start, l_move[0].end)
    assert np.allclose(l_static[0].anchor, l_move[0].anchor)
    assert sched_static.stats["hand_intervals"] == sched_move.stats["hand_intervals"]


def test_hand_grasp_survives_common_rotation(validator) -> None:
    """Rigid common yaw about the box center leaves hand_local invariant;
    inference must match the static case (invariance holds for hands only
    -- world-frame foot support is legitimately NOT invariant)."""
    T = 12
    sched_static = validator._infer_robot_schedule(_hand_box_scene(validator, T))
    sched_rot = validator._infer_robot_schedule(_hand_box_scene(validator, T, yaw_rate=0.3))
    l_static = [iv for iv in sched_static.intervals if iv.effector == "left_hand"]
    l_rot = [iv for iv in sched_rot.intervals if iv.effector == "left_hand"]
    assert l_static and l_rot, (len(l_static), len(l_rot))
    assert np.allclose(l_static[0].anchor, l_rot[0].anchor, atol=1e-9)


def test_slow_box_frame_drift_rejected_by_anchor_gate(validator) -> None:
    """Tangential box slide at 0.03 m/s (< entry 0.05, < rel-speed 0.8)
    keeps the interval alive and accumulates 30 mm of box-frame anchor
    drift > the recalibrated 20 mm gate: hand_grasp must fail (worst
    family = anchor). The
    scanned-hull surface is not flat, so the gap family drifts a few mm
    too; the structured first_failure_frame must equal the FIRST frame at
    which ANY family (gap or anchor) crosses its gate -- earlier than both
    the worst frame (30, the apex) and distinguishable from the interval
    start."""
    T = 61
    half = 30
    drift = 0.001 * np.minimum(np.arange(T), (T - 1) - np.arange(T)).astype(float)
    q = _hand_box_scene(validator, T, drift_per_frame=drift)
    rep = validator.validate(q)
    hand_iv = [iv for iv in rep.schedule.intervals if iv.effector == "left_hand"]
    assert hand_iv, rep.schedule.stats
    for name, chk in rep.checks.items():
        if name != "hand_grasp":
            assert chk.passed, (name, chk.location)
    chk = rep.checks["hand_grasp"]
    assert not chk.passed
    assert chk.first_failure_frame is not None
    iv = hand_iv[0]
    # either family may be the worst (hull-surface gap drifts too); the
    # essential assertions are the rejection and the first-crossing frame
    assert chk.worst > 1.0
    # independent recomputation of the first crossing of either family
    seg = slice(iv.start, iv.end)
    gap_t = validator._cache["hand_dist"]["l"][seg]
    local = validator._cache["hand_local"]["l"][seg]
    anchor_t = np.linalg.norm(local - iv.anchor, axis=-1).max(1)
    firsts = []
    for series, gate in ((gap_t, 0.01), (anchor_t, 0.02)):
        crossed = np.nonzero(series > gate)[0]
        if len(crossed):
            firsts.append(iv.start + int(crossed[0]))
    assert firsts, "some hand family must cross its gate"
    assert chk.first_failure_frame == min(firsts)
    assert chk.first_failure_frame > iv.start
    assert chk.first_failure_frame < iv.start + half   # first, not the apex


def test_foot_slide_reports_first_exceedance_frame(validator) -> None:
    """F3: the slow-slide rejection carries the FIRST drift exceedance
    (analytic: 0.013 m over 45 frames crosses 10 mm at frame 34 after the
    interval anchor)."""
    from test_contacts import _root_height_for_tangent

    T = 45
    h = _root_height_for_tangent(validator)
    q = np.zeros((T, 43))
    q[:, 0:3] = [0.1, 0.0, h]
    q[:, 3] = 1.0
    q[:, 36:43] = [1.5, 0.0, 0.19, 1.0, 0, 0, 0]
    q[:, 0] += np.linspace(0.0, 0.013, T)
    rep = validator.validate(q)
    iv = [i for i in rep.schedule.intervals if i.support == "ground"][0]
    assert not rep.checks["foot_support"].passed
    assert rep.checks["foot_support"].first_failure_frame == iv.start + 34
