"""FK adapter tests: tangent convention, Jacobians, cross-adapter agreement.

Plan section 6.1 acceptance: double-precision FK agreement < 1e-6 m/rad and
manifold finite-difference validation of the 35-dim Jacobian. The fast suite
here samples fewer states; ``scripts/validate_fk.py`` runs the full 1000-state
validation and records errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.embodied.g1_spec import G1Spec, KPT_BODIES, SOLE_BODIES
from ireset.kinematics.mujoco_adapter import G1Kinematics, N_TANGENT, RobotState, retract_batch
from ireset.kinematics.rotation_adapter import (
    mat_to_quat_wxyz,
    mat_to_rot6d_torch,
    quat_to_mat_wxyz,
    rot6d_to_mat_torch,
    so3_exp_np,
    so3_exp_torch,
    so3_log_np,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from ireset.kinematics.torch_adapter import G1TorchFK


def random_state(rng: np.random.Generator, spec: G1Spec) -> RobotState:
    pos = rng.uniform(-0.5, 0.5, 3)
    v = rng.standard_normal(4)
    quat = v / np.linalg.norm(v)
    q = np.zeros(29)
    for k in range(29):
        q[k] = rng.uniform(*spec.joint_range[k])
    return RobotState(pos[None], quat[None], q[None])


ALL_SITES = list(KPT_BODIES) + list(SOLE_BODIES) + [
    "hand_l_palm", "hand_l_thumb", "hand_l_pinky", "hand_r_palm", "hand_r_thumb", "hand_r_pinky",
]


@pytest.fixture(scope="module")
def kin() -> G1Kinematics:
    return G1Kinematics()


# --- rotation adapter --------------------------------------------------------


def test_quat_conversions_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.standard_normal(4)
        q = v / np.linalg.norm(v)
        assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(q)), q)
        R = quat_to_mat_wxyz(q)
        q2 = mat_to_quat_wxyz(R)
        # quaternion sign ambiguity
        assert min(np.abs(q2 - q).max(), np.abs(q2 + q).max()) < 1e-12


def test_so3_exp_log_roundtrip_and_torch_agreement() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        # principal-range rotvecs: |w| < pi so the log is the identity branch
        w = rng.standard_normal(3)
        w = w / np.linalg.norm(w) * rng.uniform(0, np.pi * 0.99)
        R = so3_exp_np(w)
        assert np.abs(so3_log_np(R) - w).max() < 1e-10
        Rt = so3_exp_torch(torch.tensor(w, dtype=torch.float64))
        assert np.abs(Rt.numpy() - R).max() < 1e-12
        r6 = mat_to_rot6d_torch(torch.tensor(R))
        R6 = rot6d_to_mat_torch(r6)
        assert np.abs(R6.numpy() - R).max() < 1e-12


# --- retraction --------------------------------------------------------------


def test_retract_zero_and_frames(kin: G1Kinematics) -> None:
    rng = np.random.default_rng(2)
    spec = kin.spec
    st = random_state(rng, spec)
    qpos0 = st.to_qpos()
    qpos_zero = retract_batch(kin.model, qpos0, np.zeros(N_TANGENT))[0]
    assert np.allclose(qpos_zero[0:7], qpos0[0, 0:7]) | np.allclose(
        qpos_zero[3:7], -qpos0[0, 3:7]
    ), "zero step must keep the root pose (quaternion up to sign)"
    assert np.allclose(qpos_zero[7:], qpos0[0, 7:])

    # world-frame rotation increment: Exp(delta) @ R for a site on the root body
    xi = np.zeros(N_TANGENT)
    xi[4] = 0.1  # +0.1 rad about world y
    qpos1 = retract_batch(kin.model, qpos0, xi)[0]
    R0 = quat_to_mat_wxyz(qpos0[0, 3:7])
    R1 = quat_to_mat_wxyz(qpos1[3:7])
    expected = so3_exp_np(np.array([0.0, 0.1, 0.0])) @ R0
    assert np.abs(R1 - expected).max() < 1e-12
    assert np.allclose(qpos1[0:3], qpos0[0, 0:3])


# --- Jacobians ----------------------------------------------------------------


def test_analytic_jacobian_matches_fd(kin: G1Kinematics) -> None:
    rng = np.random.default_rng(3)
    sites = ["kpt_l_wrist", "sole_l_1", "hand_r_palm", "kpt_pelvis"]
    worst = 0.0
    for _ in range(10):
        st = random_state(rng, kin.spec)
        J_analytic = kin.jacobian(st, sites)[0]
        J_fd = kin.fd_jacobian(st.to_qpos()[0], sites)
        denom = max(np.abs(J_fd).max(), 1e-9)
        worst = max(worst, np.abs(J_analytic - J_fd).max() / denom)
    assert worst < 1e-6, f"relative Jacobian error {worst}"


def test_jacobian_predicts_retraction_linearly(kin: G1Kinematics) -> None:
    """Small tangent step must move sites as J @ xi (sanity of convention)."""
    rng = np.random.default_rng(4)
    st = random_state(rng, kin.spec)
    sites = ["kpt_l_knee", "hand_r_thumb"]
    qpos0 = st.to_qpos()[0]
    kin._forward(qpos0)
    ids = [kin._site_id(s) for s in sites]
    x0 = kin.data.site_xpos[ids].copy()
    xi = rng.standard_normal(N_TANGENT) * np.array(
        [0.002, 0.002, 0.002, 0.002, 0.002, 0.002] + [0.003] * 29
    )
    qpos1 = retract_batch(kin.model, qpos0, xi)[0]
    kin._forward(qpos1)
    x1 = kin.data.site_xpos[ids]
    J = kin.jacobian(st, sites)[0]
    pred = np.einsum("sak,k->sa", J, xi)
    assert np.abs(pred - (x1 - x0)).max() < 5e-5  # second-order residual O(|xi|^2)


# --- torch adapter agreement ---------------------------------------------------


def test_torch_fk_agrees_with_mujoco(kin: G1Kinematics) -> None:
    fk = G1TorchFK()
    rng = np.random.default_rng(5)
    states = [random_state(rng, kin.spec) for _ in range(8)]
    batch = RobotState(
        np.concatenate([s.pos for s in states]),
        np.concatenate([s.quat for s in states]),
        np.concatenate([s.q for s in states]),
    )
    mj = kin.site_positions(batch, ALL_SITES)
    tk = (
        fk.site_world_positions(
            torch.tensor(batch.pos),
            torch.tensor(batch.quat),
            torch.tensor(batch.q),
            ALL_SITES,
        )
        .numpy()
    )
    err = np.abs(mj - tk).max()
    assert err < 1e-6, f"FK disagreement {err} m"


def test_torch_fk_gradients_flow(kin: G1Kinematics) -> None:
    fk = G1TorchFK()
    q = torch.zeros(1, 29, dtype=torch.float64, requires_grad=True)
    pos = torch.zeros(1, 3, dtype=torch.float64)
    quat = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    x = fk.site_world_positions(pos, quat, q, ["kpt_l_wrist"])
    x.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and (q.grad.abs() > 0).any()
