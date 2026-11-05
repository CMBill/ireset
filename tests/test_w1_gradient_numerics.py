"""Focused numerical regressions for the W1 gradient-stability repair.

N1 replaces the trace-only matrix -> quaternion adapter with the installed
``pytorch_kinematics`` best-conditioned branch.  N2 keeps the linear key bias
parameter in the attention graph with an exactly zero gradient.  Both are
small pure-tensor mechanisms: this module never loads a checkpoint, dataset,
real model or real training state, and never runs an optimizer on real
parameters.

The frozen output gate here is the same ``2e-8 + 2e-4 * |reference|`` used by
the W1 output-cotangent contract.  The old adapter is kept only as a negative
control so a future refactor cannot silently reintroduce the pi singularity.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.kinematics.rotation_adapter import (  # noqa: E402
    quat_apply_torch,
    rot6d_to_mat_torch,
)
from ireset.model.backbone import Attn  # noqa: E402
from ireset.model.tokens import (  # noqa: E402
    _mat_to_quat_torch,
    mat_to_quat_torch,
)
from pytorch_kinematics.transforms import (  # noqa: E402
    matrix_to_quaternion,
)

# Frozen dependency identity recorded with the repair.  ``uv.lock`` pins the
# package; this guards against a silent environment swap to a different
# branch-selection implementation.
PK_VERSION = "0.10.0"
PK_ROTATION_CONVERSIONS_SHA256 = (
    "fa4fae88af30ee99e524b1f0561e5eda7f236a51efc2d7320372f208f3a749ac")

# Exact output gate frozen by the W1 real training-update audit.
OUTPUT_GATE = {"atol": 2e-8, "rtol": 2e-4}

# The adapter's public value contract is tighter than the cotangent gate:
# float32 branch selection still rounds to a few 1e-7 on exact-pi matrices,
# while float64 must stay near machine epsilon.  These are frozen output
# tolerances for the adapter itself; the saved-frame intervention below also
# checks the original output gate where it applies.
ADAPTER_GATE = {
    torch.float32: {"atol": 1e-6, "rtol": 1e-5},
    torch.float64: {"atol": 1e-12, "rtol": 1e-9},
}

# Two saved diffusion u2/m0 prediction frames (slot 1, frames 0 and 6) from
# the failed real cohort.  Kept as literals so this regression needs no
# derived artifact directory.
SAVED_TWO_FRAME_R6 = np.array([
    [-0.9305329918861389, -0.01825786754488945, 0.2003178596496582,
     0.015529570169746876, -0.999880313873291, -0.20700129866600037],
    [-1.0155181884765625, -0.03407227620482445, 0.3138273358345032,
     0.06322026252746582, -1.011469841003418, -0.21101978421211243],
], dtype=np.float64)


def _rotation_matrix(quat_wxyz) -> np.ndarray:
    """Independent NumPy wxyz quaternion -> rotation matrix."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _edge_matrices() -> tuple[list[str], np.ndarray]:
    """The 42 frozen rotation inputs used by the local diagnostic."""
    quats: list[tuple[str, list[float]]] = [("identity", [1.0, 0.0, 0.0, 0.0])]
    for name, axis in (("x", [1, 0, 0]), ("y", [0, 1, 0]),
                       ("z", [0, 0, 1]), ("oblique", [1, 2, -3])):
        direction = np.asarray(axis, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        quats.append((f"exact_pi_{name}", [0.0, *direction.tolist()]))
        for delta in (-1e-3, -1e-6, 1e-6, 1e-3):
            angle = np.pi + delta
            quats.append((
                f"near_pi_{name}_{delta}",
                [np.cos(angle / 2), *(direction * np.sin(angle / 2))]))
    for eps in (-1e-6, 0.0, 1e-6):
        quats.extend([
            (f"w_x_seam_{eps}", [1 + eps, 1, 0, 0]),
            (f"x_y_seam_{eps}", [0, 1 + eps, 1, 0]),
            (f"four_way_seam_{eps}", [1 + eps, 1, 1, 1]),
        ])
    rng = np.random.default_rng(20261104)
    quats += [(f"random_{index}", rng.normal(size=4)) for index in range(12)]
    return ([name for name, _ in quats],
            np.stack([_rotation_matrix(value) for _, value in quats]))


def _r6_from_matrix(matrices: np.ndarray) -> np.ndarray:
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1)


def _action_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Apply wxyz quaternions to the basis: returns the transposed rotation."""
    basis = torch.eye(3, dtype=quat.dtype, device=quat.device)
    return torch.stack(
        [quat_apply_torch(quat, value.expand(quat.shape[0], 3))
         for value in basis], dim=1)


def _legacy_trace_to_quat(R: torch.Tensor) -> torch.Tensor:
    """The removed trace-only branch, kept only as a negative control."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = torch.sqrt((1.0 + tr).clamp_min(1e-12)) / 2.0
    denom = (4.0 * w).clamp_min(1e-12)
    x = (R[..., 2, 1] - R[..., 1, 2]) / denom
    y = (R[..., 0, 2] - R[..., 2, 0]) / denom
    z = (R[..., 1, 0] - R[..., 0, 1]) / denom
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _gate_ratio(got: np.ndarray, reference: np.ndarray,
                spec: dict | None = None) -> float:
    spec = OUTPUT_GATE if spec is None else spec
    denominator = spec["atol"] + spec["rtol"] * np.abs(reference)
    return float(np.max(np.abs(got - reference) / denominator))


def test_quaternion_dependency_version_and_source_are_pinned():
    assert importlib.metadata.version("pytorch-kinematics") == PK_VERSION
    source = Path(matrix_to_quaternion.__code__.co_filename)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == \
        PK_ROTATION_CONVERSIONS_SHA256
    assert "best-conditioned" in inspect.getsource(matrix_to_quaternion)
    # The public alias must remain the same callable object.
    assert mat_to_quat_torch is _mat_to_quat_torch


def test_runtime_identity_covers_the_quaternion_dependency():
    """A material that omits the N1 dependency must be refused, not skipped."""
    from copy import deepcopy

    from ireset.eval.w1_real_training_update_audit import (
        PYTORCH_KINEMATICS_SOURCE_MODULES,
        runtime_identity_facts,
        runtime_identity_problems,
    )

    assert PYTORCH_KINEMATICS_SOURCE_MODULES == (
        "pytorch_kinematics.transforms.rotation_conversions",)
    facts = runtime_identity_facts()
    assert facts["pytorch_kinematics"] == PK_VERSION
    assert facts["pytorch_kinematics_source_sha256"] == {
        PYTORCH_KINEMATICS_SOURCE_MODULES[0]:
            PK_ROTATION_CONVERSIONS_SHA256}
    assert runtime_identity_problems(facts, facts) == []

    legacy = deepcopy(facts)
    legacy.pop("pytorch_kinematics")
    legacy.pop("pytorch_kinematics_source_sha256")
    problems = runtime_identity_problems(legacy, facts)
    assert "runtime_identity:pytorch_kinematics" in problems
    assert ("runtime_identity:source:pytorch_kinematics.transforms."
            "rotation_conversions") in problems

    tampered = deepcopy(facts)
    tampered["pytorch_kinematics"] = "0.0.0"
    assert "runtime_identity:pytorch_kinematics" in \
        runtime_identity_problems(tampered, facts)

    tampered = deepcopy(facts)
    tampered["pytorch_kinematics_source_sha256"][
        PYTORCH_KINEMATICS_SOURCE_MODULES[0]] = "0" * 64
    assert ("runtime_identity:source:pytorch_kinematics.transforms."
            "rotation_conversions") in runtime_identity_problems(
                tampered, facts)

    tampered = deepcopy(facts)
    tampered["pytorch_kinematics_source_sha256"] = {}
    assert ("runtime_identity:source:pytorch_kinematics.transforms."
            "rotation_conversions") in runtime_identity_problems(
                tampered, facts)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adapter_recovers_rotation_at_pi_branches_and_seams(dtype):
    names, matrices = _edge_matrices()
    r6 = torch.tensor(_r6_from_matrix(matrices), dtype=dtype)
    R = rot6d_to_mat_torch(r6)
    quat = mat_to_quat_torch(R)
    action = _action_from_quat(quat)
    reference = R.transpose(-1, -2)

    assert quat.shape == (len(names), 4)
    assert quat.dtype == dtype
    assert bool(torch.isfinite(quat).all())
    norm_error = float((quat.norm(dim=-1) - 1).abs().max())
    assert norm_error <= (2e-6 if dtype == torch.float32 else 1e-12)
    # Historical public convention: real part non-negative (including +0).
    assert bool((quat[..., 0] >= 0).all())

    ratio = _gate_ratio(action.detach().double().numpy(),
                        reference.detach().double().numpy(),
                        ADAPTER_GATE[dtype])
    assert ratio <= 1.0, ratio
    # The old trace-only adapter really fails at exact/near pi; without this
    # negative control the regression could pass vacuously.
    legacy = _legacy_trace_to_quat(R)
    legacy_action = _action_from_quat(legacy)
    legacy_error = float((legacy_action - reference).abs().max())
    assert legacy_error > 1.0


def test_float64_adapter_gradient_matches_direct_R_finite_difference():
    names, matrices = _edge_matrices()
    r6 = torch.tensor(_r6_from_matrix(matrices), dtype=torch.float64,
                      requires_grad=True)
    R = rot6d_to_mat_torch(r6)
    action = _action_from_quat(mat_to_quat_torch(R))
    analytic = []
    for row in range(3):
        for column in range(3):
            analytic.append(torch.autograd.grad(
                action[:, row, column].sum(), r6, retain_graph=True)[0])
    analytic = torch.stack(analytic).detach().numpy()

    step = 1e-6
    finite_difference = []
    for column in range(6):
        offset = torch.zeros_like(r6)
        offset[:, column] = step
        plus = rot6d_to_mat_torch(r6.detach() + offset).transpose(-1, -2)
        minus = rot6d_to_mat_torch(r6.detach() - offset).transpose(-1, -2)
        finite_difference.append(
            ((plus - minus) / (2 * step)).reshape(len(names), 9).numpy())
    finite_difference = np.stack(finite_difference, axis=-1).transpose(1, 0, 2)

    assert np.isfinite(analytic).all()
    assert np.max(np.abs(analytic - finite_difference)) < 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saved_two_frames_match_direct_rotation_and_preserve_sign(dtype):
    r6 = torch.tensor(SAVED_TWO_FRAME_R6, dtype=dtype)
    R = rot6d_to_mat_torch(r6)
    quat = mat_to_quat_torch(R)
    action = _action_from_quat(quat)
    reference = R.transpose(-1, -2)
    assert bool((quat[..., 0] >= 0).all())
    ratio = _gate_ratio(action.detach().double().numpy(),
                        reference.detach().double().numpy())
    assert ratio <= 1.0, ratio


def test_adapter_preserves_shape_dtype_device_and_normalization():
    for dtype in (torch.float32, torch.float64):
        R = torch.eye(3, dtype=dtype).repeat(2, 4, 1, 1)
        R = R + 0.05 * torch.randn(2, 4, 3, 3, dtype=dtype)
        # Keep the inputs proper rotation matrices via the 6D decode.
        decoded = rot6d_to_mat_torch(
            torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1))
        quat = mat_to_quat_torch(decoded)
        assert quat.shape == (2, 4, 4)
        assert quat.dtype == dtype and quat.device == decoded.device
        assert torch.isfinite(quat).all()
        assert float((quat.norm(dim=-1) - 1).abs().max()) < (
            2e-6 if dtype == torch.float32 else 1e-12)


# ---------------------------------------------------------------------------
# N2: structural zero key bias
# ---------------------------------------------------------------------------


def _reference_attention(attn: Attn, x: torch.Tensor,
                         ctx: torch.Tensor | None,
                         mask: torch.Tensor | None) -> torch.Tensor:
    """The pre-repair explicit attention math, including the key bias."""
    kv = x if ctx is None else ctx
    batch, length, width = x.shape
    source = kv.shape[1]
    q = attn.q(x).view(batch, length, attn.h, attn.dk).transpose(1, 2)
    k = attn.k(kv).view(batch, source, attn.h, attn.dk).transpose(1, 2)
    v = attn.v(kv).view(batch, source, attn.h, attn.dk).transpose(1, 2)
    logits = q @ k.transpose(-2, -1) / (attn.dk ** 0.5)
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return attn.out((logits.softmax(dim=-1) @ v).transpose(1, 2).reshape(
        batch, length, width))


def _attention_case(dtype, *, cross: bool, masked: bool):
    torch.manual_seed(20261104 if cross else 20261105)
    attn = Attn(24, 4).to(dtype)
    batch, length, source = 2, 7, 11 if cross else 7
    x = torch.randn(batch, length, 24, dtype=dtype)
    ctx = torch.randn(batch, source, 24, dtype=dtype) if cross else None
    mask = None
    if masked:
        mask = torch.ones(1, 1, length, source, dtype=torch.bool)
        mask[:, :, :, -1] = False
        mask[:, :, 0, :] = False
        mask[:, :, 0, 0] = True
    return attn, x, ctx, mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_key_bias_has_zero_gradient_and_forward_is_unchanged(
        dtype, cross, masked):
    attn, x, ctx, mask = _attention_case(dtype, cross=cross, masked=masked)
    expected = _reference_attention(attn, x, ctx, mask)
    got = attn(x, ctx=ctx, mask=mask)
    # Removing the shared key bias changes only a softmax-invariant row
    # shift; the forward is equal up to float rounding in the logits.
    tight = ({"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32
             else {"rtol": 1e-10, "atol": 1e-12})
    torch.testing.assert_close(got, expected, **tight)

    got.sum().backward()
    bias_grad = attn.k.bias.grad
    assert bias_grad is not None
    assert torch.equal(bias_grad, torch.zeros_like(bias_grad))
    assert attn.k.bias.requires_grad

    # Other parameters keep the same gradients as the original projection.
    reference = Attn(24, 4).to(dtype)
    reference.load_state_dict(attn.state_dict())
    _reference_attention(reference, x, ctx, mask).sum().backward()
    for name, parameter in attn.named_parameters():
        if name == "k.bias":
            continue
        other = dict(reference.named_parameters())[name]
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, **tight)
    # Q/V biases and the output projection are untouched by the key rule.
    assert not torch.equal(reference.q.bias.grad,
                           torch.zeros_like(reference.q.bias.grad))
    assert not torch.equal(reference.v.bias.grad,
                           torch.zeros_like(reference.v.bias.grad))


def test_key_bias_rule_does_not_apply_to_nonlinear_key_projection():
    torch.manual_seed(7)
    source, width = 5, 16
    kv = torch.randn(2, source, width)
    weight = torch.randn(width, width)
    bias = torch.randn(width, requires_grad=True)
    with_bias = torch.tanh(torch.nn.functional.linear(kv, weight, bias))
    without_bias = torch.tanh(
        torch.nn.functional.linear(kv, weight, bias * 0.0))
    assert not torch.equal(with_bias, without_bias)
    # A nonlinearity makes the bias observable; the structural-zero shortcut
    # must never be applied by parameter name to this path.
    bias_grad = torch.autograd.grad(with_bias.sum(), bias)[0]
    assert bool(torch.isfinite(bias_grad).all())
    assert float(bias_grad.abs().max()) > 1e-3


def test_nonfinite_attention_input_is_not_sanitized():
    torch.manual_seed(11)
    attn = Attn(24, 4)
    x = torch.randn(2, 5, 24)
    x[0, 1, 2] = float("nan")
    out = attn(x)
    assert not bool(torch.isfinite(out).all())
    out.sum().backward()
    assert attn.k.bias.grad is not None
    assert not bool(torch.isfinite(attn.k.bias.grad).all())


def test_structural_zero_bias_scales_checkpoint_and_parameter_roster():
    torch.manual_seed(13)
    attn = Attn(24, 4)
    names = dict(attn.named_parameters())
    for key in ("k.weight", "k.bias", "q.weight", "q.bias",
                "v.weight", "v.bias", "out.weight", "out.bias"):
        assert key in names
    state = attn.state_dict()
    fresh = Attn(24, 4)
    missing, unexpected = fresh.load_state_dict(state, strict=True)
    assert missing == [] and unexpected == []
    assert torch.equal(fresh.k.bias, attn.k.bias)
    assert fresh.k.bias.requires_grad
    assert fresh.k.bias.dtype == attn.k.bias.dtype
    assert fresh.k.bias.shape == attn.k.bias.shape


def _manual_adamw(parameter, exp_avg, exp_avg_sq, step, *, lr,
                  weight_decay, beta1, beta2, eps):
    """PyTorch AdamW step with a zero gradient, including the new step count."""
    exp_avg = beta1 * exp_avg
    exp_avg_sq = beta2 * exp_avg_sq
    correction1 = 1.0 - beta1 ** step
    correction2 = 1.0 - beta2 ** step
    denom = exp_avg_sq.sqrt() / correction2 ** 0.5 + eps
    decayed = parameter * (1.0 - lr * weight_decay)
    return decayed - lr * (exp_avg / correction1) / denom


def test_zero_key_bias_gradient_consumes_existing_adamw_state():
    torch.manual_seed(17)
    attn = Attn(24, 4)
    x = torch.randn(2, 6, 24)
    attn(x).sum().backward()
    parameter = attn.k.bias
    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))

    lr, weight_decay, beta1, beta2, eps = 1e-2, 0.1, 0.9, 0.999, 1e-8
    optimizer = torch.optim.AdamW([parameter], lr=lr,
                                  weight_decay=weight_decay,
                                  betas=(beta1, beta2), eps=eps)
    first = torch.full_like(parameter, 2.5e-4)
    second = torch.full_like(parameter, 4.0e-6)
    step = 3
    optimizer.state[parameter]["step"] = torch.tensor(float(step))
    optimizer.state[parameter]["exp_avg"] = first.clone()
    optimizer.state[parameter]["exp_avg_sq"] = second.clone()
    before = parameter.detach().clone()
    expected = _manual_adamw(before, first, second, step + 1, lr=lr,
                             weight_decay=weight_decay, beta1=beta1,
                             beta2=beta2, eps=eps)
    optimizer.step()

    # Zero gradient must not be converted to None or used to clear state:
    # the stored momentum still produces the AdamW update and weight decay.
    torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-8)
    assert not torch.equal(parameter, before)
    assert torch.equal(optimizer.state[parameter]["exp_avg"], first * beta1)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"],
                               second * beta2, rtol=1e-6, atol=1e-12)
    assert float(optimizer.state[parameter]["step"]) == step + 1
