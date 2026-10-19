"""Math and path contracts for w1_rotation_path_nlerp_v2 (plan §3/§11)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.model.rotation_fusion import (
    ANGULAR_SPEED_LIMIT_RADPS,
    MIN_BLEND_NORM,
    PATH_AMBIGUITY_DOT,
    PROTOCOL,
    RotationFusionRejected,
    angular_speed_radps,
    fuse_windows_v2,
)
from ireset.model.stitch import fuse_windows


def q_wxyz(axis_deg: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis_deg, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    xyzw = Rot.from_rotvec(np.deg2rad(angle_deg) * axis).as_quat()
    return xyzw[[3, 0, 1, 2]]


def qz(angle_deg: float) -> np.ndarray:
    return q_wxyz((0, 0, 1), angle_deg)


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = Rot.from_quat(np.asarray(a)[..., [1, 2, 3, 0]])
    rb = Rot.from_quat(np.asarray(b)[..., [1, 2, 3, 0]])
    return np.rad2deg((ra.inv() * rb).magnitude())


def assert_same_so3(a: np.ndarray, b: np.ndarray, atol_deg: float = 1e-9) -> None:
    diff = geodesic_deg(a, b)
    assert np.all(diff <= atol_deg), diff.max()


def identity_windows(n: int, f: int):
    return (np.zeros((n, f, 3)), np.tile(np.array([1.0, 0, 0, 0]), (n, f, 1)),
            np.zeros((n, f, 29)))


def test_protocol_name():
    assert PROTOCOL == "w1_rotation_path_nlerp_v2"


def test_three_window_middle_positive_weight_is_not_dropped():
    T, F, stride = 100, 60, 30
    starts = [0, 30, 40]
    pos, quat, joints = identity_windows(3, F)
    quat[1] = qz(60.0)
    pos[:, :, 0] = np.arange(F)[None, :] * 1e-3
    result = fuse_windows_v2(pos, quat, joints, starts, T, stride)

    legacy_pos, legacy_quat, legacy_joints, _ = fuse_windows(
        list(pos), list(quat), list(joints), starts, T, stride)
    assert np.array_equal(result.pos, legacy_pos)
    assert np.array_equal(result.joints, legacy_joints)
    assert result.diagnostics["starts"] == starts
    assert result.diagnostics["window_activation_frames"] == [0, 31, 41]
    # Triple overlap: all three windows are positive-weight contributors.
    assert result.diagnostics["frame_active_windows"][41] == [0, 1, 2]
    # Legacy only looks at the first/last slots -> identity; v2 keeps window 1.
    assert geodesic_deg(legacy_quat[41], [1, 0, 0, 0]) < 1e-9
    assert geodesic_deg(result.quat[41], [1, 0, 0, 0]) > 10.0
    # Zero-weight entry of the tail window does not cut the existing blend.
    # Compare against the same two contributors without the tail window up to
    # the zero-weight entry frame (frame 40).
    without_tail = fuse_windows_v2(pos[:2], quat[:2], joints[:2], [0, 30], 90, 30)
    assert_same_so3(result.quat[:41], without_tail.quat[:41], atol_deg=1e-12)
    assert np.array_equal(result.pos[:41], without_tail.pos[:41])
    # Last frame is covered by the non-integer-stride tail window.
    assert starts[-1] + F == T
    assert result.diagnostics["blend_norm"][-1] > MIN_BLEND_NORM


def _two_window_crossing(increasing: bool):
    T, F, stride = 90, 60, 30
    starts = [0, 30]
    pos = np.zeros((2, F, 3))
    joints = np.zeros((2, F, 29))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (2, F, 1))
    local = np.arange(F)
    angles = (2.0 + 0.08 * local) if increasing else (4.28 - 0.08 * local)
    quat[1] = np.stack([np.cos(angles / 2), np.zeros(F), np.zeros(F),
                        np.sin(angles / 2)], axis=1)
    return T, F, stride, starts, pos, quat, joints


@pytest.mark.parametrize("increasing", [True, False])
def test_relative_pi_crossing_does_not_switch_branch(increasing):
    T, F, stride, starts, pos, quat, joints = _two_window_crossing(increasing)
    result = fuse_windows_v2(pos, quat, joints, starts, T, stride, frame_dt=1 / 30)
    _, legacy_quat, _, _ = fuse_windows(list(pos), list(quat), list(joints), starts, T, stride)

    legacy_step = geodesic_deg(legacy_quat[:-1], legacy_quat[1:])
    fused_step = geodesic_deg(result.quat[:-1], result.quat[1:])
    # The legacy shortest-arc branch flips around the pi crossing.
    assert legacy_step.max() > 90.0
    # The fixed path stays on one branch; NLERP near the antipode can amplify
    # the local step, but never to the legacy ~150 deg class.
    assert fused_step.max() < 20.0
    assert fused_step.max() < legacy_step.max() / 5
    assert result.fusion_temporal_pass
    assert result.consumer_eligible
    # The model windows themselves are smooth; the legacy jump is fusion-added.
    assert result.diagnostics["window_angular_speed_max"][0]["max_radps"] == 0.0
    assert result.diagnostics["window_angular_speed_max"][1]["max_radps"] < ANGULAR_SPEED_LIMIT_RADPS
    assert result.diagnostics["pre_fusion_angular_speed_max_radps"] > 20.0
    assert result.diagnostics["angular_speed_global_max_radps"] < ANGULAR_SPEED_LIMIT_RADPS
    if increasing:
        assert result.diagnostics["window_signs"][1] == 1.0
    else:
        assert result.diagnostics["window_signs"][1] == -1.0


def test_per_frame_representation_signs_do_not_change_so3_result():
    T, F, stride = 100, 60, 30
    starts = [0, 30, 40]
    local = np.arange(T)
    angles = np.deg2rad(20.0 + 0.4 * local)
    truth = np.stack([np.cos(angles / 2), np.zeros(T), np.zeros(T),
                      np.sin(angles / 2)], axis=1)
    pos = np.stack([local * 1e-4, np.sin(local * 0.1) * 1e-3, np.zeros(T)], axis=1)
    joints = np.tile(local[:, None] * 1e-5, (1, 29))
    pos_w = np.stack([pos[s:s + F] for s in starts])
    quat = np.stack([truth[s:s + F] for s in starts])
    joints_w = np.stack([joints[s:s + F] for s in starts])
    base = fuse_windows_v2(pos_w, quat, joints_w, starts, T, stride)

    rng = np.random.default_rng(20261015)
    flipped = quat.copy()
    for i in range(flipped.shape[0]):
        for t in range(flipped.shape[1]):
            if rng.random() < 0.5:
                flipped[i, t] *= -1.0
    frozen = flipped.copy()
    variant = fuse_windows_v2(pos_w, flipped, joints_w, starts, T, stride)
    assert_same_so3(base.quat, variant.quat, atol_deg=1e-10)
    assert np.array_equal(base.pos, variant.pos)
    assert np.array_equal(base.joints, variant.joints)
    # Inputs are never mutated.
    assert np.array_equal(flipped, frozen)


def test_single_double_triple_segmentation_preserves_the_same_truth():
    T, F = 100, 60
    local = np.arange(T)
    angles = np.deg2rad(15.0 + 0.5 * local)
    truth = np.stack([np.cos(angles / 2), np.zeros(T), np.zeros(T),
                      np.sin(angles / 2)], axis=1)
    pos = np.stack([0.01 * local, 0.02 * np.sin(0.05 * local), 0.03 * np.cos(0.02 * local)], axis=1)
    joints = np.tile(np.sin(0.03 * local)[:, None], (1, 29))

    single = fuse_windows_v2(pos[None, :F], truth[None, :F], joints[None, :F], [0], F, 30)
    double = fuse_windows_v2(np.stack([pos[0:F], pos[30:30 + F]]),
                             np.stack([truth[0:F], truth[30:30 + F]]),
                             np.stack([joints[0:F], joints[30:30 + F]]),
                             [0, 30], 90, 30)
    triple = fuse_windows_v2(np.stack([pos[0:F], pos[30:30 + F], pos[40:40 + F]]),
                             np.stack([truth[0:F], truth[30:30 + F], truth[40:40 + F]]),
                             np.stack([joints[0:F], joints[30:30 + F], joints[40:40 + F]]),
                             [0, 30, 40], T, 30)
    assert_same_so3(single.quat, truth[:F], atol_deg=1e-9)
    assert_same_so3(double.quat, truth[:90], atol_deg=1e-9)
    assert_same_so3(triple.quat, truth, atol_deg=1e-9)
    assert np.allclose(single.pos, pos[:F], atol=1e-12, rtol=0)
    assert np.allclose(triple.pos, pos, atol=1e-12, rtol=0)
    assert np.allclose(triple.joints, joints, atol=1e-12, rtol=0)


def test_rigid_transform_equivariance_and_input_immutability():
    T, F = 90, 60
    starts = [0, 30]
    local = np.arange(T)
    angles = np.deg2rad(10.0 + 0.3 * local)
    truth = np.stack([np.cos(angles / 2), np.zeros(T), np.zeros(T),
                      np.sin(angles / 2)], axis=1)
    pos = np.stack([local * 1e-3, 0.01 * np.cos(0.2 * local), 0.02 * np.ones(T)], axis=1)
    joints = np.tile(np.cos(0.1 * local)[:, None], (1, 29))
    p_w = np.stack([pos[0:F], pos[30:30 + F]])
    q_w = np.stack([truth[0:F], truth[30:30 + F]])
    j_w = np.stack([joints[0:F], joints[30:30 + F]])
    p0, q0, j0 = p_w.copy(), q_w.copy(), j_w.copy()

    R = Rot.from_rotvec(np.deg2rad([12.0, -7.0, 31.0]))
    t0 = np.array([0.4, -0.2, 1.1])
    p_r = p_w @ R.as_matrix().T + t0
    q_xyzw = (R * Rot.from_quat(q_w.reshape(-1, 4)[:, [1, 2, 3, 0]])).as_quat()
    q_r = q_xyzw[:, [3, 0, 1, 2]].reshape(q_w.shape)

    base = fuse_windows_v2(p_w, q_w, j_w, starts, T, 30)
    moved = fuse_windows_v2(p_r, q_r, j_w, starts, T, 30)
    assert np.allclose(moved.pos, base.pos @ R.as_matrix().T + t0, atol=1e-12, rtol=0)
    expected_q = (R * Rot.from_quat(base.quat[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    assert_same_so3(moved.quat, expected_q, atol_deg=1e-9)
    assert np.array_equal(p_w, p0)
    assert np.array_equal(q_w, q0)
    assert np.array_equal(j_w, j0)


def test_activation_dot_threshold_boundaries():
    F, stride = 60, 30
    T = 90
    pos = np.zeros((2, F, 3))
    joints = np.zeros((2, F, 29))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (2, F, 1))
    boundary = np.array([0.999e-6, np.sqrt(1 - (0.999e-6) ** 2), 0.0, 0.0])
    quat[1] = boundary
    with pytest.raises(RotationFusionRejected) as exc:
        fuse_windows_v2(pos, quat, joints, [0, 30], T, stride)
    assert exc.value.code == "window_activation_ambiguous"
    assert exc.value.frame == 31
    assert exc.value.prefix["length"] == 31
    assert abs(exc.value.evidence["dot"]) <= PATH_AMBIGUITY_DOT

    above = boundary.copy()
    above[0] = 1.001e-6
    above[1] = np.sqrt(1 - above[0] ** 2)
    quat[1] = above
    result = fuse_windows_v2(pos, quat, joints, [0, 30], T, stride)
    assert result.diagnostics["window_activation_dots"][1] > PATH_AMBIGUITY_DOT


def test_lift_ambiguity_is_an_honest_rejection_with_prefix():
    T, F = 60, 60
    pos = np.zeros((1, F, 3))
    joints = np.zeros((1, F, 29))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (1, F, 1))
    quat[0, 1] = np.array([0.999e-6, np.sqrt(1 - (0.999e-6) ** 2), 0.0, 0.0])
    with pytest.raises(RotationFusionRejected) as exc:
        fuse_windows_v2(pos, quat, joints, [0], T, 30)
    assert exc.value.code == "window_lift_ambiguous"
    assert exc.value.frame == 1
    assert exc.value.window == 0
    assert exc.value.local_frame == 1
    assert exc.value.prefix["length"] == 1
    assert exc.value.prefix["quat"].shape == (1, 4)


def _blend_norm_case(dot: float):
    # Window 0 covers frames 0..2, window 1 covers 1..3.  Frame 1 is the
    # activation reference (window 1 weight is still zero); frame 2 has equal
    # positive weights.  All exact boundary components are binary fractions.
    F, stride, T = 3, 1, 4
    q01 = np.array([0.0, 0.0, 0.0, 1.0])
    q02 = np.array([0.5, 0.5, 0.5, 0.5])
    if dot == -0.5:
        q11 = np.array([-0.5, -0.5, -0.5, 0.5])
    else:
        # dot(q02, q11) = (z - 3a)/2 < -0.5 while q11_z > 0 keeps activation
        # unambiguous and positive.
        a = 0.55
        q11 = np.array([-a, -a, -a, np.sqrt(1 - 3 * a * a)])
    quat0 = np.stack([q01, q01, q02])
    quat1 = np.stack([q11, q11, q11])
    quat = np.stack([quat0, quat1])
    return quat, np.zeros((2, F, 3)), np.zeros((2, F, 29)), [0, 1], T, stride


def test_blend_norm_boundary_is_inclusive_at_one_half():
    quat, pos, joints, starts, T, stride = _blend_norm_case(-0.5)
    result = fuse_windows_v2(pos, quat, joints, starts, T, stride, frame_dt=1.0)
    assert result.diagnostics["blend_norm"][2] == pytest.approx(MIN_BLEND_NORM, abs=1e-12)
    assert result.fusion_math_valid

    quat, pos, joints, starts, T, stride = _blend_norm_case(-0.515)
    with pytest.raises(RotationFusionRejected) as exc:
        fuse_windows_v2(pos, quat, joints, starts, T, stride, frame_dt=1.0)
    assert exc.value.code == "quaternion_blend_ill_conditioned"
    assert exc.value.frame == 2
    assert exc.value.evidence["blend_norm"] < MIN_BLEND_NORM
    assert exc.value.prefix["length"] == 2


@pytest.mark.parametrize("mutate,code", [
    (lambda p, q, j: (p[:, :, :2], q, j), "invalid_pos_shape"),
    (lambda p, q, j: (np.where(np.arange(p.size).reshape(p.shape) == 5, np.nan, p), q, j), "nonfinite_pos"),
    (lambda p, q, j: (p, q * 2.0, j), "quaternion_norm_out_of_tolerance"),
    (lambda p, q, j: (p, np.where(np.arange(q.size).reshape(q.shape) == 7, np.inf, q), j),
     "nonfinite_quat"),
])
def test_bad_inputs_are_rejected(mutate, code):
    pos, quat, joints = identity_windows(2, 60)
    pos, quat, joints = mutate(pos, quat, joints)
    with pytest.raises(RotationFusionRejected) as exc:
        fuse_windows_v2(pos, quat, joints, [0, 30], 90, 30)
    assert exc.value.code == code


def test_bad_counts_and_coverage_are_rejected():
    pos, quat, joints = identity_windows(2, 60)
    cases = [
        (dict(starts=[0, 70], T=130, stride=30), "uncovered_frame"),
        (dict(starts=[0, 30, 30], T=90, stride=30), "invalid_starts"),
        (dict(starts=[0.0, 30], T=90, stride=30), "invalid_starts"),
        (dict(starts=[0, 30], T=90, stride=True), "invalid_stride"),
        (dict(starts=[0, 30], T=True, stride=30), "invalid_T"),
        (dict(starts=[0, 30], T=90, stride=30, frame_dt=True), "invalid_frame_dt"),
        (dict(starts=[0, 30], T=90, stride=30, frame_dt=np.inf), "invalid_frame_dt"),
        (dict(starts=[0, 30], T=91, stride=30), "starts_do_not_cover_last_frame"),
        (dict(starts=[1, 30], T=90, stride=30), "starts_do_not_begin_at_zero"),
    ]
    for kwargs, code in cases:
        starts = kwargs.pop("starts")
        T = kwargs.pop("T")
        with pytest.raises(RotationFusionRejected) as exc:
            fuse_windows_v2(pos, quat, joints, starts, T, **kwargs)
        assert exc.value.code == code, (kwargs, exc.value.code)


def test_model_own_overspeed_is_reported_not_rejected():
    T, F = 60, 60
    pos = np.zeros((1, F, 3))
    joints = np.zeros((1, F, 29))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (1, F, 1))
    quat[0, 10] = qz(120.0)
    result = fuse_windows_v2(pos, quat, joints, [0], T, 30, frame_dt=1 / 30)
    assert result.numerical_valid is True
    assert result.fusion_math_valid is True
    assert result.fusion_temporal_pass is False
    assert result.consumer_eligible is False
    assert result.diagnostics["window_angular_speed_max"][0]["max_radps"] > ANGULAR_SPEED_LIMIT_RADPS
    assert result.diagnostics["first_temporal_violation_frame"] == 10
    assert result.quat.shape == (T, 4)


def test_angular_speed_uses_so3_not_representation():
    identity = np.tile(np.array([1.0, 0, 0, 0]), (2, 1))
    flipped = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    assert angular_speed_radps(identity, 1.0)[0] == pytest.approx(0.0, abs=1e-12)
    assert angular_speed_radps(flipped, 1.0)[0] == pytest.approx(0.0, abs=1e-12)
    # Sign-flipped representations of the same path have identical speed.
    quat = np.stack([qz(10.0 * i) for i in range(5)])
    signed = quat.copy()
    signed[1] *= -1
    signed[3] *= -1
    assert np.allclose(angular_speed_radps(quat, 1.0), angular_speed_radps(signed, 1.0), atol=0, rtol=0)
