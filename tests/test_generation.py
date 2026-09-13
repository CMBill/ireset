"""M3 generation-layer tests: tokens, data schema/windows, backbone,
diffusion scheduler adaptation (x0-prediction), stitching, losses.

Heavy-data tests read one val-split NPZ (real teacher data); everything
else runs on synthetic tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.data.schema import ReferenceCondition
from ireset.data.teacher import clip_frame_from_human, world_to_clip
from ireset.data.windows import CACHE_DIR, TeacherWindowDataset, load_split_files, make_tokenizer
from ireset.model.tokens import RootStats, StateTokenizer, quat_to_mat_torch
from ireset.model.conditioner import ConditionEncoder, N_PER_FRAME, N_STATIC
from ireset.model.backbone import G1Transformer, assemble_x0
from ireset.model.deterministic import DeterministicG1
from ireset.model.stitch import fuse_windows, window_starts, window_weight
from ireset.model.generator import G1Generator
from ireset.diffusion.generator import DiffusionG1
from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz, so3_exp_np
from ireset.train.losses import LossAssembler, LossWeights

RNG = np.random.default_rng(20260913)


def _tokenizer() -> StateTokenizer:
    joint_range = np.stack([np.full(29, -2.5), np.full(29, 2.5)], axis=1)
    joint_range[3] = [-3.0, 0.5]  # asymmetric entry
    stats = RootStats(mean=np.array([0.1, -0.2, 0.8]), std=np.array([0.4, 0.3, 0.15]))
    return StateTokenizer(joint_range, stats)


def _random_state(T: int, tk: StateTokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = tk.lo, tk.hi
    q = lo + (hi - lo) * RNG.random((T, 29))
    p = RNG.normal([0.1, -0.2, 0.8], [0.4, 0.3, 0.15], (T, 3))
    quat = np.tile([1.0, 0, 0, 0], (T, 1)) + RNG.normal(0, 0.05, (T, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return np.concatenate([p, quat], axis=1), q, quat


def _wfc() -> np.ndarray:
    wfc = np.eye(4)
    wfc[:3, :3] = so3_exp_np(np.array([0.0, 0.0, 0.7]))
    wfc[:2, 3] = [1.5, -0.8]
    return wfc


# --- tokenizer ---------------------------------------------------------------

def test_tokenizer_roundtrip():
    tk = _tokenizer()
    root, q, _ = _random_state(50, tk)
    x0 = tk.encode(root, q, _wfc())
    assert x0.shape == (50, 38)
    assert np.abs(x0[:, 9:38]).max() <= 1.0 + 1e-12
    root2, q2, ok = tk.decode(x0, _wfc())
    assert ok.all()
    assert np.abs(root2 - root).max() < 1e-5
    assert np.abs(q2 - q).max() < 1e-9


def test_tokenizer_joint_limit_map():
    tk = _tokenizer()
    q = np.stack([tk.lo + 1e-9, tk.hi - 1e-9, (tk.lo + tk.hi) / 2])
    u = tk.joints_to_u(q)
    assert np.allclose(u[0], -1.0) and np.allclose(u[1], 1.0) and np.allclose(u[2], 0.0, atol=1e-12)
    u_ext = np.full((1, 29), 0.3)
    u_ext[0, 0], u_ext[0, 1] = -2.0, 2.0
    q2 = tk.u_to_joints(u_ext)
    assert np.abs(q2[0, 0] - tk.lo[0]) < 1e-12 and np.abs(q2[0, 1] - tk.hi[0]) < 1e-12


def test_tokenizer_rot6d_orthonormalize_and_degenerate():
    tk = _tokenizer()
    x0 = RNG.normal(0, 2, (10, 38))
    x0[:, 3:9] = x0[:, 3:9] * 3.0 + 5.0        # garbage 6D
    x0[5, 3:6] = 0.0                            # degenerate frame
    root2, _, ok = tk.decode(x0, _wfc())
    assert not ok[5] and bool(np.delete(ok, 5).all())
    R = quat_to_mat_wxyz(root2[ok][:, 3:7])
    eye = np.einsum("tij,tkj->tik", R, R)
    assert np.abs(eye - np.eye(3)).max() < 1e-5


def test_tokenizer_torch_matches_numpy():
    tk = _tokenizer()
    root, q, _ = _random_state(20, tk)
    wfc = _wfc()
    x0_np = tk.encode(root, q, wfc)
    mean, std = tk.root_stats.to_torch()
    lo = torch.tensor(tk.lo); hi = torch.tensor(tk.hi)
    x0_t = tk.encode_torch(
        torch.tensor(root[None, :, 0:3]), torch.tensor(root[None, :, 3:7]),
        torch.tensor(q[None]), torch.tensor(wfc[None]), mean, std, lo, hi)
    assert np.abs(x0_t[0].numpy() - x0_np).max() < 1e-6


# --- data layer --------------------------------------------------------------

def test_split_files_disjoint():
    splits = {s: set(load_split_files(s)) for s in ("train", "val", "test")}
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])
    assert sum(len(v) for v in splits.values()) == 1826


def test_clip_frame_removes_yaw_and_origin():
    human = RNG.normal(0, 0.3, (5, 52, 3))
    human[:, 0, 2] += 0.9
    wfc = clip_frame_from_human(human)
    kps = world_to_clip(human, wfc)
    # extra consistent yaw+translation of the whole scene leaves clip frame fixed
    R = so3_exp_np(np.array([0.0, 0.0, 1.3]))
    t = np.array([2.0, -1.0, 0.0])
    human2 = human @ R.T + t
    wfc2 = wfc.copy()
    wfc2[:3, :3] = R @ wfc[:3, :3]
    wfc2[:3, 3] = R @ wfc[:3, 3] + t
    kps2 = world_to_clip(human2, wfc2)
    assert np.abs(kps - kps2).max() < 1e-10


@pytest.mark.slow
def test_dataset_windows_and_hygiene():
    tk = make_tokenizer(RootStats(mean=np.zeros(3), std=np.ones(3)))
    ds = TeacherWindowDataset("val", tk, window=60, stride=30, max_files=2)
    assert len(ds) > 0
    item = ds[0]
    assert item["x0"].shape == (60, 38)
    assert item["kps_obj"].shape == (60, 52, 3)
    assert item["contact"].shape == (60, 4)
    assert set(item["contact"].unique().tolist()) <= {0.0, 1.0}
    # condition channels must not contain target root/joints: shapes differ and
    # the schedule comes from the human side only (0/1 flags)
    for lc in ds.clips:
        assert not lc.clip.validation.get("drop")


def test_to_repair_condition():
    T = 10
    cond = ReferenceCondition(
        source_clip_id="x",
        human_kps_world=np.zeros((T, 52, 3)),
        source_obj_pose=np.tile([0, 0, 0.2, 1, 0, 0, 0], (T, 1)),
        target_obj_pose=np.tile([0, 0, 0.2, 1, 0, 0, 0], (T, 1)),
    )
    rc = cond.to_repair_condition()
    assert rc.box_pose.shape == (T, 7) and rc.fps == 30.0


# --- backbone / conditioner ----------------------------------------------------

def _tiny_batch(B=2, F=8):
    return {
        "kps_obj": torch.randn(B, F, 52, 3),
        "kps_clip": torch.randn(B, F, 52, 3),
        "box_t_clip": torch.randn(B, F, 3),
        "box_quat": torch.nn.functional.normalize(torch.randn(B, F, 4), dim=-1),
        "contact": torch.randint(0, 2, (B, F, 4)).float(),
        "x0": torch.randn(B, F, 38),
    }


def test_conditioner_and_backbone_shapes():
    B, F, d = 2, 16, 32
    batch = _tiny_batch(B, F)
    cond = ConditionEncoder(d)
    mem = cond(batch)
    assert mem.shape == (B, N_STATIC + F * N_PER_FRAME, d)

    net = G1Transformer(d_model=d, n_layers=2, n_heads=4, ffn_mult=2, time_window=31)
    t = torch.randint(0, 1000, (B,))
    out = net(batch["x0"], t, mem)
    assert out["root"].shape == (B, F, 9)
    assert out["joint_logits"].shape == (B, F, 29)
    x0_pred = assemble_x0(out["root"], out["joint_logits"])
    assert x0_pred.shape == (B, F, 38)
    assert x0_pred[..., 9:].abs().max() < 1.0          # sigmoid map keeps joints inside

    det = G1Transformer(d_model=d, n_layers=2, n_heads=4, ffn_mult=2, use_time=False)
    out2 = det(None, None, mem, frames=F)
    assert out2["root"].shape == (B, F, 9)


def test_window_mask_semantics():
    net = G1Transformer(d_model=16, n_layers=1, n_heads=2, time_window=5)
    m = net._self_mask(10, torch.device("cpu"))
    assert m.shape == (10 * 30, 10 * 30)
    # token (t=0) may attend exactly frames 0..2 (half window 2)
    allowed = m[:30].reshape(30, 10, 30).any(-1)      # [30 query tokens, 10 frames]
    assert allowed[:, :3].all() and not allowed[:, 3:].any()


# --- diffusion adaptation -------------------------------------------------------

def test_ddim_perfect_x0_model_recovers_target():
    torch.manual_seed(0)
    diff = DiffusionG1(d_model=16, n_layers=1, n_heads=2)
    B, F = 2, 6
    target = torch.randn(B, F, 38)
    diff.forward = lambda x_t, t, memory: target   # perfect x0 prediction
    mem = torch.zeros(B, 4, 16)
    gen = torch.Generator().manual_seed(3)
    out = diff.sample(mem, F, steps=5, generator=gen)
    # last DDIM step returns the clean prediction exactly (final prev SNR = 1)
    assert torch.allclose(out, target, atol=1e-5)


def test_diffusion_seed_reproducibility():
    torch.manual_seed(0)
    diff = DiffusionG1(d_model=16, n_layers=1, n_heads=2)
    mem = torch.randn(2, 4, 16)
    a = diff.sample(mem, 6, steps=3, generator=torch.Generator().manual_seed(11))
    b = diff.sample(mem, 6, steps=3, generator=torch.Generator().manual_seed(11))
    c = diff.sample(mem, 6, steps=3, generator=torch.Generator().manual_seed(12))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_add_noise_matches_closed_form():
    diff = DiffusionG1()
    x0 = torch.tensor([[1.0, -2.0] + [0.0] * 36])
    eps = torch.tensor([[0.5, 0.25] + [0.0] * 36])
    t = torch.tensor([300])
    ab = diff.ddpm.alphas_cumprod[300]
    xt = diff.add_noise(x0, eps, t)
    expect = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    assert torch.allclose(xt, expect, atol=1e-6)
    sched_cfg = diff.ddpm.config
    assert sched_cfg.prediction_type == "sample" and sched_cfg.clip_sample is False
    assert diff.ddim.config.thresholding is False


# --- stitching ------------------------------------------------------------------

def test_window_starts_and_weights():
    assert window_starts(200, 60, 30) == [0, 30, 60, 90, 120, 140]
    assert window_starts(60, 60, 30) == [0]
    assert window_starts(30, 60, 30) == [0]
    w = window_weight(1, 3, 60, 30)
    assert w[0] == 0.0 and abs(w[-1] - 0.0) < 1e-12 and abs(w[29] - 1.0) < 1e-12
    w0 = window_weight(0, 3, 60, 30)
    assert w0[0] == 1.0                       # first window owns its opening


def test_fuse_identical_and_slerp():
    F, stride = 60, 30
    sig_p = RNG.normal(0, 0.01, (90, 3))
    sig_qj = RNG.normal(0, 0.05, (90, 29))
    pos = [sig_p[0:60], sig_p[30:90]]          # windows of one underlying signal
    qj = [sig_qj[0:60], sig_qj[30:90]]
    q_id = np.tile([1.0, 0, 0, 0], (F, 1))
    q = [q_id, q_id]
    p, qq, jj, diag = fuse_windows(pos, q, qj, [0, 30], 90, stride)
    assert np.abs(p - sig_p).max() < 1e-9        # weights sum to 1 exactly
    assert np.abs(jj - sig_qj).max() < 1e-9
    assert np.abs(np.linalg.norm(qq, axis=1) - 1).max() < 1e-9
    assert diag["n_seam_steps"] == 2

    # rotation blend: window B rotated 90 deg about z -> crossfade midpoint
    # (global frame 44, weights ~0.5/0.5) sits near half the arc
    q_b = np.tile(np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]), (F, 1))
    p2, q2, _, _ = fuse_windows(pos, [q[0], q_b], qj, [0, 30], 90, stride)
    R = quat_to_mat_wxyz(q2)
    angles = np.degrees(np.arccos(np.clip(R[:, 0, 0], -1, 1)))
    assert 35 < angles[44] < 55


# --- losses -----------------------------------------------------------------------

def test_loss_assembler_finite_and_differentiable():
    tk = _tokenizer()
    B, F = 2, 6
    batch = _tiny_batch(B, F)
    batch["wfc"] = torch.tensor(np.tile(_wfc(), (B, 1, 1)), dtype=torch.float32)
    la = LossAssembler(tk, LossWeights(), "cpu")
    x0_pred = torch.randn(B, F, 38, requires_grad=True)
    loss, parts = la(x0_pred, batch)
    for key in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2", "contact_m2", "total"):
        assert key in parts and np.isfinite(parts[key]), parts
    loss.backward()
    assert x0_pred.grad is not None and torch.isfinite(x0_pred.grad).all()


def test_contact_chain_matches_mujoco_box_frame():
    """Contact-loss clip->box transform vs MuJoCo FK + world_to_box_geom.

    Review round 3 (2026-09-13): the chain previously applied R_box_clip
    without the transpose — up to 0.10 m per-point deviation from the
    M1-verified geom frame on val data; the transposed form matched to
    5.9e-8 m. Guards the exact convention with a real G1 model.
    """
    from ireset.embodied.g1_spec import G1Spec
    from ireset.geometry.distance_adapter import world_to_box_geom
    from ireset.kinematics.mujoco_adapter import G1Kinematics, RobotState
    from ireset.train.losses import HAND_SITE_NAMES

    rng = np.random.default_rng(3)
    tk = StateTokenizer(G1Spec().joint_range, RootStats(mean=np.zeros(3), std=np.ones(3)))
    T = 7
    root = np.concatenate(
        [rng.normal([0, 0, 0.8], [0.3, 0.3, 0.1], (T, 3)),
         np.tile([1.0, 0, 0, 0], (T, 1)) + rng.normal(0, 0.08, (T, 4))], axis=1)
    root[:, 3:7] /= np.linalg.norm(root[:, 3:7], axis=1, keepdims=True)
    q = tk.lo + (tk.hi - tk.lo) * rng.random((T, 29))
    wfc = np.eye(4)
    wfc[:3, :3] = so3_exp_np(rng.normal(size=3) * 0.5)
    wfc[:3, 3] = rng.uniform(-0.5, 0.5, 3)
    box_pose = np.concatenate([rng.uniform(-0.7, 0.7, (T, 3)), rng.normal(size=(T, 4))], axis=1)
    box_pose[:, 3:7] /= np.linalg.norm(box_pose[:, 3:7], axis=1, keepdims=True)

    x0 = tk.encode(root, q, wfc)
    batch = {
        "x0": torch.as_tensor(x0, dtype=torch.float32)[None],
        "box_t_clip": torch.as_tensor(world_to_clip(box_pose[:, 0:3], wfc), dtype=torch.float32)[None],
        "box_quat": torch.as_tensor(box_pose[:, 3:7], dtype=torch.float32)[None],
        "wfc": torch.as_tensor(wfc, dtype=torch.float32)[None],
    }
    asm = LossAssembler(tk, LossWeights(), "cpu")
    pos_p, quat_p, q_p = asm.decode_clip(batch["x0"])
    flat = lambda a: a.reshape(-1, a.shape[-1])
    hand_p = asm.fk.site_world_positions(
        flat(pos_p), flat(quat_p), flat(q_p), HAND_SITE_NAMES).view(1, T, -1, 3)
    p_geom = asm.hand_points_box_geom(hand_p, batch)[0].detach().numpy()

    root_w, q_w, ok = tk.decode(x0, wfc)
    hand_w = G1Kinematics().site_positions(
        RobotState(root_w[:, 0:3], root_w[:, 3:7], q_w), HAND_SITE_NAMES)
    p_geom_gt = world_to_box_geom(hand_w, box_pose)
    assert ok.all()
    err = float(np.abs(p_geom - p_geom_gt).max())
    assert err < 1e-4, f"contact chain deviates from MuJoCo geom frame: {err} m"


# --- generator smoke ----------------------------------------------------------------

def test_generator_deterministic_smoke():
    torch.manual_seed(1)
    T = 100
    d = 24
    human = RNG.normal([0, 0, 0.9], 0.25, (T, 52, 3))
    box = np.tile([0.5, 0.1, 0.25, 1, 0, 0, 0], (T, 1))
    box[1:, 0:3] += np.cumsum(RNG.normal(0, 0.002, (T - 1, 3)), axis=0)
    cond = ReferenceCondition(
        source_clip_id="smoke",
        human_kps_world=human,
        source_obj_pose=box,
        target_obj_pose=box.copy(),
        frame_dt=1 / 30,
        world_from_clip=clip_frame_from_human(human),
    )
    tk = _tokenizer()
    model = DeterministicG1(d_model=d, n_layers=1, n_heads=4, ffn_mult=2)
    gen = G1Generator(model, ConditionEncoder(d), tk, device="cpu", window=60, stride=30)
    traj = gen.sample(cond, seed=0)
    assert traj.root.shape == (T, 7) and traj.q.shape == (T, 29)
    assert np.abs(np.linalg.norm(traj.root[:, 3:7], axis=1) - 1).max() < 1e-6
    assert (traj.q >= tk.lo - 1e-6).all() and (traj.q <= tk.hi + 1e-6).all()
    scene = traj.scene_qpos(cond.target_obj_pose)
    assert scene.shape == (T, 43) and np.allclose(scene[:, 36:43], box)
    assert traj.metadata["boundary"]["n_seam_steps"] == 4   # starts [0, 30, 40]
