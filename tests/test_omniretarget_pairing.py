"""A1 freeze & pairing contracts (plan omniretarget_first_training_plan §4.3).

- similarity fit recovers an exact synthetic transform;
- source/target scene separation: a constant-offset original passes all
  pairing gates, and every gate has a dedicated counterexample (perturbed
  human, varying offset, rotated box, T mismatch, wrong fps, non-finite
  input, degenerate quaternion);
- hash-rule list selection is prefix-exact, per-subject, deterministic and
  identity-only;
- the derived source-scene box uses the project [x,y,z,qw,qx,qy,qz] layout
  and differs from the target box by exactly the constant offset.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.data import pairing  # noqa: E402

RNG = np.random.default_rng(20261008)
T = 80


def _rigid(rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rot.from_rotvec(axis * rng.uniform(0.3, 2.0)).as_matrix()


def make_scene(rng, offset_z=0.062, offset_varying=False, human_noise=0.0,
               box_rot_shift_deg=0.0):
    """Synthetic PT source scene + NPZ target scene under a known transform."""
    pt_human = np.cumsum(rng.normal(size=(T, 52, 3)) * 0.01, axis=0) + rng.normal(size=3)
    pt_obj = np.zeros((T, 7))  # [x,y,z,qx,qy,qz,qw]
    pt_obj[:, 0:3] = np.stack([np.linspace(0.2, 0.5, T), np.full(T, 0.1), np.linspace(0.1, 0.3, T)], axis=1)
    quat = Rot.from_rotvec(np.outer(np.linspace(0, 0.8, T), [0, 0, 1.0])).as_quat()  # xyzw
    pt_obj[:, 3:6], pt_obj[:, 6] = quat[:, 0:3], quat[:, 3]

    scale, R, t = 0.75, _rigid(rng), np.array([0.4, -0.2, 0.9])
    npz_human = scale * (pt_human.reshape(T, -1, 3) @ R.T) + t
    npz_human = npz_human + rng.normal(size=npz_human.shape) * human_noise

    src_pos = scale * (pt_obj[:, 0:3] @ R.T) + t
    src_rot = R[None] @ Rot.from_quat(pt_obj[:, 3:7]).as_matrix()  # xyzw layout
    if box_rot_shift_deg:
        shift = Rot.from_rotvec([0, 0, np.radians(box_rot_shift_deg)]).as_matrix()
        src_rot = shift[None] @ src_rot
    box_quat_wxyz = Rot.from_matrix(src_rot).as_quat()[:, [3, 0, 1, 2]]
    dz = np.linspace(0, offset_z, T) if offset_varying else np.full(T, offset_z)
    box_pos = src_pos + np.stack([np.zeros(T), np.zeros(T), dz], axis=1)

    qpos = np.zeros((T, 43))  # raw teacher layout [qw,qx,qy,qz,x,y,z]
    root_q = Rot.from_matrix(_rigid(rng)).as_quat()
    qpos[:, 0:4] = root_q[[3, 0, 1, 2]]
    qpos[:, 4:7] = [0.0, 0.0, 0.78]
    qpos[:, 7:36] = RNG.uniform(-0.5, 0.5, size=(T, 29))
    qpos[:, 36:40] = box_quat_wxyz
    qpos[:, 40:43] = box_pos
    return dict(pt_human=pt_human, pt_obj=pt_obj, qpos=qpos,
                npz_human=npz_human, scale=scale, R=R, t=t, offset_z=offset_z)


JOINT_RANGE = np.column_stack([np.full(29, -1.0), np.full(29, 1.0)])


def verify(scene, **kw):
    return pairing.verify_scene_pairing(
        "subX_largebox_000", scene["qpos"], scene["npz_human"],
        scene["pt_human"], scene["pt_obj"], fps=30.0,
        joint_range=JOINT_RANGE, subject_height_m=1.32 / scene["scale"], **kw,
    )


def test_fit_similarity_exact_recovery():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(500, 3))
    s, R_true, t_true = 1.3, _rigid(rng), rng.normal(size=3)
    B = s * (A @ R_true.T) + t_true
    s_fit, R_fit, t_fit, resid = pairing.fit_similarity(A, B)
    assert resid < 1e-12
    assert abs(s_fit - s) < 1e-12
    assert np.allclose(R_fit, R_true, atol=1e-10)
    assert np.allclose(t_fit, t_true, atol=1e-10)


def test_original_pairing_all_gates_pass_and_offset_recorded():
    scene = make_scene(np.random.default_rng(1))
    rec = verify(scene)
    assert rec.error is None and rec.passed, rec.gates
    assert abs(rec.box_offset_mean_m[2] - scene["offset_z"]) < 1e-9
    assert abs(rec.box_offset_mean_m[0]) < 1e-9 and abs(rec.box_offset_mean_m[1]) < 1e-9
    assert rec.joint_limit_max_violation_rad == 0.0
    d = rec.to_json_dict()
    assert d["scale_rel_diff"] is not None and d["scale_rel_diff"] < 1e-9


def test_source_box_separates_from_target_box():
    scene = make_scene(np.random.default_rng(2))
    scale, R, t = scene["scale"], scene["R"], scene["t"]
    src = pairing.source_obj_pose_in_target_scene(scene["pt_obj"], scale, R, t)
    assert src.shape == (T, 7)
    assert np.abs(np.linalg.norm(src[:, 3:7], axis=1) - 1).max() < 1e-12  # wxyz unit
    # position differs from the target box by exactly the constant offset
    np.testing.assert_allclose(
        scene["qpos"][:, 40:43] - src[:, 0:3] - np.array([0.0, 0.0, scene["offset_z"]]),
        0.0, atol=1e-9,
    )
    # orientation agrees with the target box (offset is translation-only)
    src_q_xyzw = src[:, [4, 5, 6, 3]]
    np.testing.assert_allclose(
        Rot.from_quat(src_q_xyzw).as_matrix(),
        Rot.from_quat(scene["qpos"][:, 36:40][:, [1, 2, 3, 0]]).as_matrix(),
        atol=1e-9,
    )


def test_counterexamples_each_gate_fails():
    # varying offset -> constancy gate fails, others still pass
    rec = verify(make_scene(np.random.default_rng(3), offset_varying=True))
    assert not rec.gates["box_offset_constant"] and not rec.passed
    assert rec.gates["human_residual"] and rec.gates["box_rot_mapping"]
    # perturbed human -> similarity residual gate fails
    rec = verify(make_scene(np.random.default_rng(4), human_noise=1e-3))
    assert not rec.gates["human_residual"] and not rec.passed
    assert rec.human_residual_max_m > pairing.GATE_HUMAN_RESIDUAL_M
    # box orientation shifted by 0.1 deg -> rot mapping gate fails
    rec = verify(make_scene(np.random.default_rng(5), box_rot_shift_deg=0.1))
    assert not rec.gates["box_rot_mapping"] and not rec.passed
    assert rec.box_rot_err_deg_max > pairing.GATE_BOX_ROT_DEG


def test_t_mismatch_and_fps_gates():
    scene = make_scene(np.random.default_rng(6))
    rec = pairing.verify_scene_pairing(
        "d", scene["qpos"][:-3], scene["npz_human"][:-3], scene["pt_human"],
        scene["pt_obj"], fps=30.0, joint_range=JOINT_RANGE,
    )
    assert not rec.gates["t_match"]
    rec = pairing.verify_scene_pairing(
        "d", scene["qpos"], scene["npz_human"], scene["pt_human"],
        scene["pt_obj"], fps=25.0, joint_range=JOINT_RANGE,
    )
    assert rec.gates["t_match"] and not rec.gates["fps_30"] and not rec.passed


def test_nonfinite_input_and_degenerate_quaternion():
    scene = make_scene(np.random.default_rng(7))
    bad_human = scene["npz_human"].copy()
    bad_human[10, 0, 0] = np.nan
    rec = pairing.verify_scene_pairing(
        "d", scene["qpos"], bad_human, scene["pt_human"], scene["pt_obj"],
        fps=30.0, joint_range=JOINT_RANGE,
    )
    assert not rec.passed and rec.error is not None and not rec.human_all_finite

    bad_q = scene["qpos"].copy()
    bad_q[5, 36:40] = 0.0  # degenerate box quaternion
    rec = pairing.verify_scene_pairing(
        "d", bad_q, scene["npz_human"], scene["pt_human"], scene["pt_obj"],
        fps=30.0, joint_range=JOINT_RANGE,
    )
    assert not rec.gates["box_quat_norm"] and not rec.passed


def test_joint_limit_violation_recorded():
    scene = make_scene(np.random.default_rng(8))
    scene["qpos"][3, 10] = 3.0  # far outside JOINT_RANGE
    rec = verify(scene)
    assert rec.joint_limit_max_violation_rad == pytest.approx(2.0)


def test_pt_obj_layout_slices():
    rng = np.random.default_rng(9)
    pt = np.zeros((4, 591))
    obj = rng.normal(size=(4, 7))
    pt[:, 162:318] = rng.normal(size=(4, 156))
    pt[:, 318:325] = obj
    pos, quat_wxyz = pairing.pt_obj_to_project(obj)
    np.testing.assert_allclose(pos, obj[:, 0:3])
    np.testing.assert_allclose(quat_wxyz, obj[:, [6, 3, 4, 5]])


def test_hash_rules_prefix_exact_per_subject_deterministic():
    demos = [
        "sub1_largebox_001", "sub1_largebox_002", "sub2_largebox_010",
        "sub2_largebox_011", "sub3_largebox_000",
    ]
    pilot = pairing.select_first_per_subject(demos, "20261008|teacher_pilot|")
    val6 = pairing.select_first_per_subject(demos, "20261008|")
    for subj_map in (pilot, val6):
        assert set(subj_map) == {"sub1", "sub2", "sub3"}
        for subj, demo in subj_map.items():
            ranked = sorted(
                (d for d in demos if d.startswith(subj + "_")),
                key=lambda d: hashlib.sha256(
                    (  # independent recomputation of the exact rule
                        ("20261008|teacher_pilot|" if subj_map is pilot else "20261008|") + d
                    ).encode("utf-8")
                ).hexdigest(),
            )
            assert demo == ranked[0]
    assert pairing.select_first_per_subject(demos, "20261008|teacher_pilot|") == pilot
    # selection must not depend on input order
    shuffled = list(reversed(demos))
    assert pairing.select_first_per_subject(shuffled, "20261008|teacher_pilot|") == pilot


def test_scene_pairing_json_record_roundtrip():
    scene = make_scene(np.random.default_rng(10))
    d = verify(scene).to_json_dict()
    assert d["demo"] == "subX_largebox_000" and d["passed"] is True
    assert d["gates"] and all(isinstance(v, bool) for v in d["gates"].values())
    assert d["scale_expected"] == pytest.approx(scene["scale"])
