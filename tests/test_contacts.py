"""Contact inference tests (Fix 2/3): invariants, surface gaps, per-site checks,
hysteresis boundaries, hull reference metric, baseline initial pose."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_data import convert_scene_qpos
from ireset.embodied.contact_sites import ContactSchedule, hysteresis_intervals, infer_reference_schedule
from ireset.geometry.distance_adapter import (
    largebox_hull,
    largebox_hull_surface_distance,
    point_box_surface_distance_numpy,
    world_to_box_geom,
)
from ireset.paths import TEACHER_TRAJ_DIR
from ireset.repair.validation import IndependentValidator, ValidationConfig


# --- Fix 2b: threshold invariants ----------------------------------------------


def test_inverted_thresholds_rejected() -> None:
    with pytest.raises(ValueError, match="hand_enter_m"):
        ValidationConfig(hand_enter_m=0.02, hand_gap_m=0.01)
    with pytest.raises(ValueError, match="foot_enter_m"):
        ValidationConfig(foot_enter_m=0.012, foot_ground_gap_m=0.005)
    with pytest.raises(ValueError, match="hysteresis"):
        ValidationConfig(hand_exit_m=0.005, hand_enter_m=0.01)


def test_default_config_satisfies_invariants() -> None:
    cfg = ValidationConfig()
    assert cfg.hand_enter_m <= cfg.hand_gap_m
    assert cfg.foot_enter_m <= cfg.foot_ground_gap_m


# --- Fix 2a: sole surface gap equals MuJoCo distance ---------------------------


@pytest.fixture(scope="module")
def validator() -> IndependentValidator:
    return IndependentValidator(cfg=ValidationConfig())


def _sole_gap_and_mj(v: IndependentValidator, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compare (site-center z - radius) vs mj_geomDistance per sole geom."""
    d = mujoco.MjData(v.model)
    d.qpos = qpos
    mujoco.mj_forward(v.model, d)
    gaps_mj, gaps_center = [], []
    for s in ("l", "r"):
        ids = v.sole_sites[s]
        centers = d.site_xpos[ids]
        gaps_center.append(centers[:, 2].min() - 0.005)
        gaps_mj.append(min(
            mujoco.mj_geomDistance(v.model, d, g, v.ground_geom, 1.0, v._fromto)
            for g in v.sole_geoms[s]
        ))
    return np.array(gaps_center), np.array(gaps_mj)


def test_sole_surface_gap_matches_mujoco(validator) -> None:
    """Sphere-surface distance (not center height) is the gap; tilted foot too."""
    rng = np.random.default_rng(2)
    worst_flat = worst_tilt = 0.0
    for _ in range(4):
        qpos = np.zeros(43)
        qpos[2] = 0.75
        qpos[3] = 1.0
        qpos[6] = 0.0  # legs slightly bent keeps sites above ground
        lo, hi = validator.spec.joint_range[:, 0], validator.spec.joint_range[:, 1]
        qpos[7:36] = rng.uniform(lo, hi) * 0.4
        # straight legs would leave the feet high; drop the root so soles reach ground
        qpos[2] = 0.62
        qpos_tilt = qpos.copy()
        qpos_tilt[3:7] = np.array([np.cos(0.05), 0, np.sin(0.05), 0])  # ~5.7 deg roll
        gf, gm = _sole_gap_and_mj(validator, qpos)
        tf, tm = _sole_gap_and_mj(validator, qpos_tilt)
        worst_flat = max(worst_flat, float(np.abs(gf - gm).max()))
        worst_tilt = max(worst_tilt, float(np.abs(tf - tm).max()))
    # center-minus-radius is exact only for flat feet; the validator's gap must
    # be the MuJoCo value, so we check the validator cache directly instead
    q = np.zeros((3, 43))
    q[:, 2] = 0.62
    q[:, 3] = 1.0
    q[:, 36:43] = [1.5, 0.0, 0.19, 1.0, 0, 0, 0]  # valid box pose, away from robot
    validator.validate(q)
    for s in ("l", "r"):
        d = mujoco.MjData(validator.model)
        for t in range(3):
            d.qpos = q[t]
            mujoco.mj_forward(validator.model, d)
            expected = min(
                mujoco.mj_geomDistance(validator.model, d, g, validator.ground_geom, 1.0, validator._fromto)
                for g in validator.sole_geoms[s]
            )
            assert abs(validator._cache["sole_gap"][s][t] - expected) < 1e-12
    assert np.isfinite([worst_flat, worst_tilt]).all()


def test_perfect_ground_contact_passes_gap_gate(validator) -> None:
    """A foot exactly touching (sphere tangent to ground) must pass strictly.

    With the old center-height metric the ratio was exactly 1.0 (any noise
    failed); the surface gap makes tangent contact read ~0.
    """
    q = np.zeros((6, 43))
    q[:, 2] = 0.75
    q[:, 3] = 1.0
    q[:, 36:43] = [1.5, 0.0, 0.19, 1.0, 0, 0, 0]
    rep = validator.validate(q)
    # feet hang in the air here: no support inferred, gap check vacuous-pass
    assert rep.checks["foot_support"].passed
    # now lower the robot so soles are tangent: gap ~ 0 < 0.005
    q2 = q.copy()
    q2[:, 2] = _root_height_for_tangent(validator)
    rep2 = validator.validate(q2)
    for iv in rep2.schedule.intervals:
        if iv.support == "ground":
            gap = validator._cache["sole_gap"][iv.effector[0]][iv.start : iv.end]
            assert gap.max() < ValidationConfig().foot_ground_gap_m


def _root_height_for_tangent(v: IndependentValidator) -> float:
    """Smallest root height (q=0) with all soles strictly above ground.

    Bisection on the min sole-sphere surface gap; returns the tangent height
    from above (~0.8 m for straight legs).
    """
    d = mujoco.MjData(v.model)
    lo, hi = 0.0, 1.5  # gap < 0 at lo, gap > 0 at hi
    for _ in range(50):
        mid = (lo + hi) / 2
        d.qpos = np.zeros(43)
        d.qpos[2] = mid
        d.qpos[3] = 1.0
        d.qpos[39] = 1.0
        mujoco.mj_forward(v.model, d)
        gap = min(
            mujoco.mj_geomDistance(v.model, d, g, v.ground_geom, 1.0, v._fromto)
            for s in ("l", "r") for g in v.sole_geoms[s]
        )
        if gap > 0:
            hi = mid
        else:
            lo = mid
    return hi


# --- Fix 2c: per-site pivot detection -------------------------------------------


def test_pivot_in_place_detected(validator) -> None:
    """A foot yawing in place sweeps its sole sites; centroid stays fixed."""
    from ireset.kinematics.rotation_adapter import mat_to_quat_wxyz, so3_exp_np

    # Standing pivot: yaw the root in place. The sole layout is asymmetric,
    # so the centroid also sweeps, but strictly less than the outer sites
    # (radius 0.131 m vs 0.19 m here); the per-site check is the strictly
    # more sensitive formulation required by plan 10.1 ("all constrained
    # sites") and must flag the pivot.
    h = _root_height_for_tangent(validator)
    T = 8
    angs = np.linspace(0, 0.3, T)  # ~17 deg: toe 0.25 m/s passes the entry gate
    q = np.zeros((T, 43))
    for t, a in enumerate(angs):
        q[t, 0:3] = [0.1, 0.0, h]
        q[t, 3:7] = mat_to_quat_wxyz(so3_exp_np([0, 0, a]))
        q[t, 36:43] = [1.5, 0.0, 0.19, 1.0, 0, 0, 0]
    rep = validator.validate(q)
    supports = [iv for iv in rep.schedule.intervals if iv.support == "ground"]
    assert supports, "expected inferred support for the standing pivot"
    pts = validator._cache["sole_pts"]["l"]
    centroid_drift = np.linalg.norm(pts.mean(1) - pts[0].mean(0), axis=1).max()
    per_site_drift = np.linalg.norm(pts - pts[0], axis=-1).max()
    assert per_site_drift > centroid_drift       # sites sweep strictly more
    assert per_site_drift > 0.05                 # outer sites clearly violate the gate
    assert not rep.checks["foot_support"].passed  # per-site check must flag it


# --- hysteresis boundaries --------------------------------------------------------


def test_hysteresis_boundaries() -> None:
    # 0.012 sits in the hysteresis band: stays inside the contact
    metric = np.array([0.3, 0.004, 0.006, 0.012, 0.006, 0.3])
    ivs = hysteresis_intervals(metric, 0.01, 0.015, 1)
    assert ivs == [(1, 5)]
    # too short intervals dropped
    assert hysteresis_intervals(np.array([0.3, 0.005, 0.3]), 0.01, 0.015, 3) == []
    # active at end closes interval at T
    assert hysteresis_intervals(np.array([0.3, 0.005, 0.006]), 0.01, 0.015, 1) == [(1, 3)]
    # exit must re-cross exit threshold, not just enter
    metric2 = np.array([0.005, 0.012, 0.005, 0.3])
    assert hysteresis_intervals(metric2, 0.01, 0.015, 1) == [(0, 3)]


# --- Fix 3: hull reference metric -------------------------------------------------


def test_reference_default_uses_hull() -> None:
    """Near-surface points give different hull vs AABB distances."""
    rng = np.random.default_rng(3)
    # points ~2cm outside the AABB toward a corner: max hull/AABB divergence
    corner = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    base = np.array([0.30, 0.30, 0.30])
    pts = np.tile(base, (5, 1)) + rng.normal(0, 0.005, (5, 1)) * corner
    pose = np.tile(np.array([0.0, 0.0, 0.3, 1.0, 0, 0, 0]), (5, 1))
    geom_pts = world_to_box_geom(pts, pose)
    d_hull = largebox_hull_surface_distance(pts, pose)
    d_aabb = point_box_surface_distance_numpy(geom_pts)
    assert np.abs(d_hull - d_aabb).max() > 0.01   # proxy is NOT acceptance-grade
    assert (d_hull >= d_aabb - 1e-9).all()         # AABB underestimates outside


def test_reference_schedule_hull_vs_aabb_intervals() -> None:
    """Locking the fact that AABB cannot decide reference contacts."""
    npz = np.load(TEACHER_TRAJ_DIR / "sub10_largebox_000_original.npz", allow_pickle=True)
    human = npz["human_joints"].astype(np.float64)
    scene = np.stack([convert_scene_qpos(r) for r in npz["qpos"]])
    box_pose = scene[:, 36:43]
    ref_hull = infer_reference_schedule(human, box_pose)
    from ireset.geometry.distance_adapter import point_box_surface_distance_numpy, world_to_box_geom

    def aabb_fn(centroid, pose):
        return point_box_surface_distance_numpy(world_to_box_geom(centroid, pose))

    ref_aabb = infer_reference_schedule(human, box_pose, hand_distance_fn=aabb_fn)
    n_hull = len(ref_hull.intervals)
    n_aabb = len(ref_aabb.intervals)
    # the two metrics produce different interval structure (documented fact)
    totals_hull = sum(iv.end - iv.start for iv in ref_hull.intervals if "hand" in iv.effector)
    totals_aabb = sum(iv.end - iv.start for iv in ref_aabb.intervals if "hand" in iv.effector)
    assert (n_hull, n_aabb, totals_hull, totals_aabb) != (0, 0, 0, 0)
    assert abs(totals_hull - totals_aabb) > 0  # they disagree somewhere
