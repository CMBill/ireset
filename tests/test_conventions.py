"""Convention tests for the locked G1 asset and project-wide layouts.

Run: uv run --locked pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.embodied.g1_spec import (
    HAND_SITES,
    JOINT_NAMES,
    KPT_TO_SMPLH,
    NQ_ROBOT,
    NQ_SCENE,
    SOLE_SITES,
    G1Spec,
)


@pytest.fixture(scope="module")
def spec() -> G1Spec:
    return G1Spec()


def test_locked_models_exist_and_compile(spec: G1Spec) -> None:
    for path in (spec.plain_xml, spec.scene_xml, spec.urdf):
        assert path.exists(), f"missing derived asset {path}; run scripts/build_locked_assets.py"
    m_plain = spec.load_plain()
    m_scene = spec.load_scene()
    assert m_plain.nq == NQ_ROBOT
    assert m_scene.nq == NQ_SCENE
    assert mujoco.MjModel.from_xml_path(str(spec.urdf)).nq == 29


def test_joint_order_and_limits_match_locked_model(spec: G1Spec) -> None:
    m = spec.load_plain()
    hinge = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(m.njnt)
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    assert tuple(hinge) == JOINT_NAMES
    assert len(JOINT_NAMES) == 29 and len(set(JOINT_NAMES)) == 29

    for k, name in enumerate(JOINT_NAMES):
        lo, hi = spec.joint_range[k]
        qadr = spec.qposadr[k]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert qadr == m.jnt_qposadr[jid]
        assert (lo, hi) == (m.jnt_range[jid, 0], m.jnt_range[jid, 1])
        assert lo < hi


def test_all_declared_sites_present(spec: G1Spec) -> None:
    m = spec.load_plain()
    for group in (KPT_TO_SMPLH, *[dict.fromkeys(v) for v in SOLE_SITES.values()],
                  *[dict.fromkeys(v.values()) for v in HAND_SITES.values()]):
        for site_name in group:
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name) >= 0, site_name
    expected = len(KPT_TO_SMPLH) + sum(len(v) for v in SOLE_SITES.values()) + sum(
        len(v) for v in HAND_SITES.values()
    )
    robot_sites = sum(
        1
        for i in range(m.nsite)
        if not mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i).startswith("unused_")
    )
    assert robot_sites >= expected


def test_root_storage_convention_is_xyz_quat_wxyz(spec: G1Spec) -> None:
    """MuJoCo freejoint qpos is [x, y, z, qw, qx, qy, qz]; project layout agrees."""
    m = spec.load_plain()
    d = mujoco.MjData(m)
    d.qpos[:3] = [0.1, 0.2, 0.9]
    d.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]  # 180 deg about z: wxyz
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert np.allclose(d.xpos[bid], d.qpos[:3])
    R_stored = np.zeros(9)
    mujoco.mju_quat2Mat(R_stored, np.ascontiguousarray(d.qpos[3:7]))
    R_state = np.zeros(9)
    mujoco.mju_quat2Mat(R_state, np.ascontiguousarray(d.xquat[bid]))
    assert np.allclose(R_stored, R_state, atol=1e-12)


def test_scene_split_roundtrip(spec: G1Spec) -> None:
    rng = np.random.default_rng(3)
    scene = rng.standard_normal((5, NQ_SCENE))
    robot, box = spec.split_robot_qpos(scene)
    assert robot.shape == (5, NQ_ROBOT) and box.shape == (5, 7)
    assert np.allclose(np.concatenate([robot, box], axis=-1), scene)
    pos, quat, joints = spec.split_robot_state(robot)
    assert pos.shape == (5, 3) and quat.shape == (5, 4) and joints.shape == (5, 29)
