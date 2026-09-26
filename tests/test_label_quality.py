"""Graded-label INTERFACE contracts (plan stage B): dataset entry + losses.

- data contracts: hash/T/fps/protocol/group/mask mismatches refuse to load;
  window masks and interval ids break at reference gaps and certified
  boundaries; label modification (joint clamping) is refused;
- loss and leakage: empty supervision -> zero contact loss and zero contact
  gradient; supervision masks never touch the conditioner or the motion /
  FK / velocity terms; the old dataset entry keeps its behaviour;
- manifest -> windows -> condition encoding -> loss forward/backward smoke
  (no training step, no checkpoint).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.data.labels import (LABEL_PROTOCOL_VERSION, grasp_pair_valid,  # noqa: E402
                                    side_label_arrays)
from ireset.data.teacher import load_teacher_npz  # noqa: E402
from ireset.data.windows import TeacherWindowDataset, load_split_files  # noqa: E402
from ireset.model.conditioner import ConditionEncoder  # noqa: E402
from ireset.train.losses import LossAssembler, LossWeights  # noqa: E402

# --- graded dataset entry: contracts -------------------------------------------------

VAL_DEMO_FILES = load_split_files("val")


def _val_demo() -> str:
    """A real val demo (source original NPZ, zero joint violations)."""
    for name in VAL_DEMO_FILES:
        if name.endswith("_original.npz"):
            return name[: -len("_original.npz")]
    pytest.skip("no val original NPZ available")


def _graded_env(tmp_path: Path, demo: str | None = None,
                window: int = 8, dist_pattern: str = "mixed"):
    """Fabricate a run directory: kept NPZ (copy of the source original),
    label NPZ with synthetic-but-consistent masks, manifest row."""
    from ireset.paths import TEACHER_TRAJ_DIR

    demo = demo or _val_demo()
    src = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    shutil.copy2(src, run_dir / f"{demo}.npz")
    traj, human, fps = load_teacher_npz(run_dir / f"{demo}.npz")
    T = traj.shape[0]

    ref = np.zeros(T, dtype=bool)
    ref[2:min(2 + max(window, 10), T)] = True
    ref[min(2 * max(window, 10) + 2, T - 1):min(2 * max(window, 10) + 2 + window, T)] = True
    cert = [(1, min(1 + max(window, 10) - 2, T)),          # straddles ref start
            (min(2 * max(window, 10) + 3, T - 1), min(2 * max(window, 10) + 3 + 4, T))]
    if dist_pattern == "mixed":
        dist = np.where(ref, 0.005, 0.5)
    elif dist_pattern == "at_threshold":
        dist = np.where(ref, 0.01, 0.5)                     # == enter -> no proximity
    else:
        dist = np.full(T, 0.5)
    sides = {}
    for i, side in enumerate(("l", "r")):
        sides[side] = side_label_arrays(dist if i == 0 else np.full(T, 0.5),
                                        ref if i == 0 else np.zeros(T, dtype=bool),
                                        cert if i == 0 else [], 0.01)
    prox = np.stack([sides[s]["proximity"] for s in ("l", "r")], axis=1)
    sup = np.stack([sides[s]["supervision"] for s in ("l", "r")], axis=1)
    ids = np.stack([sides[s]["interval_id"] for s in ("l", "r")], axis=1)
    meta = {"demo": demo, "T": int(T), "fps": float(fps),
            "label_protocol": LABEL_PROTOCOL_VERSION, "sides": ["l", "r"],
            "enter_m": 0.01, "proximity_operator": "<",
            "evidence": {}, "hands": {},
            "source_npz_sha256": None, "trajectory_npz_sha256": None}
    labels_dir = run_dir / "labels"
    labels_dir.mkdir()
    label_path = labels_dir / f"{demo}.npz"
    np.savez(label_path, hand_proximity_mask=prox,
             hand_contact_supervision_mask=sup,
             hand_supervision_interval_id=ids,
             meta_json=np.array(json.dumps(meta)))
    import hashlib

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    manifest = {
        "label_protocol": LABEL_PROTOCOL_VERSION,
        "rows": [{
            "demo": demo, "group": "val", "status": "accepted", "keep": True,
            "T": int(T), "fps": float(fps),
            "output_npz": {"path": str(run_dir / f"{demo}.npz"),
                           "sha256": sha(run_dir / f"{demo}.npz")},
            "label_npz": {"path": str(label_path), "sha256": sha(label_path)},
            "strict_recheck_passed": True,
            "label_eligibility": {"motion_fk": True, "proximity": True,
                                  "supervision_frames": [int(sup[:, i].sum()) for i in (0, 1)]},
        }],
    }
    manifest_path = run_dir / "label_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {"run_dir": run_dir, "manifest": manifest_path, "demo": demo, "T": T,
            "prox": prox, "sup": sup, "ids": ids, "window": window}


def _graded_tokenizer():
    from ireset.data.windows import make_tokenizer
    from ireset.model.tokens import RootStats

    return make_tokenizer(RootStats(mean=np.zeros(3), std=np.ones(3)))


@pytest.mark.slow
def test_graded_dataset_loads_and_slices_masks(tmp_path):
    env = _graded_env(tmp_path)
    tk = _graded_tokenizer()
    ds = TeacherWindowDataset("val", tk, window=env["window"],
                              stride=env["window"] // 2,
                              label_manifest=env["manifest"])
    assert len(ds.clips) == 1
    item = ds[0]
    F = env["window"]
    s = int(item["start"])
    for key, arr in (("hand_proximity_mask", env["prox"]),
                     ("hand_contact_supervision_mask", env["sup"]),
                     ("hand_supervision_interval_id", env["ids"])):
        got = item[key].numpy()
        assert got.shape == (F, 2)
        np.testing.assert_array_equal(got, arr[s:s + F])
    # the model condition channels are still the reference-human schedule
    assert item["contact"].shape == (F, 4)
    # interval ids: adjacency only inside one piece (never across the gaps)
    ids_col = env["ids"][:, 0]
    ok = grasp_pair_valid(ids_col)
    for t in range(1, env["T"]):
        if not ok[t - 1]:
            assert ids_col[t - 1] != ids_col[t] or ids_col[t - 1] < 0


@pytest.mark.slow
def test_graded_dataset_rejects_mismatches(tmp_path):
    """Hash/T/fps/protocol/group/mask-consistency errors all refuse to load;
    nothing falls back to the original teacher files."""
    tk = _graded_tokenizer()
    base = _graded_env(tmp_path)
    manifest = json.loads(base["manifest"].read_text())

    def fresh(mut):
        env = _graded_env(tmp_path / mut.__name__, demo=base["demo"])
        m = json.loads(env["manifest"].read_text())
        mut(m["rows"][0], env)
        env["manifest"].write_text(json.dumps(m))
        return env["manifest"]

    def tamper_out(row, env):
        p = Path(row["output_npz"]["path"])
        p.write_bytes(p.read_bytes() + b"x")

    def tamper_label(row, env):
        p = Path(row["label_npz"]["path"])
        p.write_bytes(p.read_bytes() + b"x")

    def wrong_t(row, env):
        row["T"] = row["T"] + 1

    def wrong_fps(row, env):
        row["fps"] = row["fps"] + 1.0

    def wrong_protocol(row, env):
        pass  # handled at manifest level below

    for mut in (tamper_out, tamper_label, wrong_t, wrong_fps):
        with pytest.raises((ValueError, FileNotFoundError)):
            TeacherWindowDataset("val", tk, window=8,
                                 label_manifest=fresh(mut))

    env = _graded_env(tmp_path / "proto")
    m = json.loads(env["manifest"].read_text())
    m["label_protocol"] = "graded_labels_v0"
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="label protocol"):
        TeacherWindowDataset("val", tk, window=8, label_manifest=env["manifest"])

    # group claim not backed by the locked split
    env = _graded_env(tmp_path / "group")
    m = json.loads(env["manifest"].read_text())
    m["rows"][0]["group"] = "train"          # the demo is a val demo
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="split"):
        TeacherWindowDataset("train", tk, window=8, label_manifest=env["manifest"])

    # no rows for the requested split at all
    with pytest.raises(ValueError, match="no rows"):
        TeacherWindowDataset("test", tk, window=8, label_manifest=base["manifest"])

    # masks disagreeing with the interval ids (hash refreshed so the failure
    # tested is the consistency rule, not the stale-hash guard)
    env = _graded_env(tmp_path / "maskdis")
    m = json.loads(env["manifest"].read_text())
    lp = Path(m["rows"][0]["label_npz"]["path"])
    with np.load(lp) as z:
        payload = {k: z[k] for k in z.files}
    payload["hand_contact_supervision_mask"] = np.zeros_like(
        payload["hand_contact_supervision_mask"])
    np.savez(lp, **payload)
    import hashlib

    m["rows"][0]["label_npz"]["sha256"] = hashlib.sha256(lp.read_bytes()).hexdigest()
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="disagree"):
        TeacherWindowDataset("val", tk, window=8, label_manifest=env["manifest"])


@pytest.mark.slow
def test_graded_dataset_refuses_label_modification(tmp_path):
    """A trajectory whose joints need clamping must be refused, not loaded
    with modified labels."""
    env = _graded_env(tmp_path)
    demo = env["demo"]
    traj, human, fps = load_teacher_npz(env["run_dir"] / f"{demo}.npz")
    traj[:, 7] = 3.5                              # far outside every limit
    np.savez(env["run_dir"] / f"{demo}.npz", qpos=traj, human_joints=human, fps=fps)
    # manifest sha256 is now stale -- refresh it so the failure tested is the
    # joint-limit one, not the hash one
    m = json.loads(env["manifest"].read_text())
    import hashlib

    m["rows"][0]["output_npz"]["sha256"] = hashlib.sha256(
        (env["run_dir"] / f"{demo}.npz").read_bytes()).hexdigest()
    env["manifest"].write_text(json.dumps(m))
    with pytest.raises(ValueError, match="labels would need modification"):
        TeacherWindowDataset("val", _graded_tokenizer(), window=8,
                             label_manifest=env["manifest"])


@pytest.mark.slow
def test_old_dataset_entry_has_no_graded_keys():
    ds = TeacherWindowDataset("val", _graded_tokenizer(), window=60,
                              stride=30, max_files=1)
    if not ds.clips:
        pytest.skip("val clip shorter than 60")
    assert not ds.graded
    assert "hand_contact_supervision_mask" not in ds[0]


# --- loss and leakage ---------------------------------------------------------------


def _loss_env():
    from test_generation import _tokenizer, _tiny_batch, _wfc

    tk = _tokenizer()
    batch = _tiny_batch(2, 6)
    batch["wfc"] = torch.tensor(np.tile(_wfc(), (2, 1, 1)), dtype=torch.float32)
    return tk, batch


def test_empty_supervision_mask_zero_loss_zero_gradient():
    """Empty graded mask: zero contact loss, finite total, zero contact
    gradient (identical gradients to w_contact=0)."""
    tk, batch = _loss_env()
    batch["hand_contact_supervision_mask"] = torch.zeros(2, 6, 2, dtype=torch.bool)
    la = LossAssembler(tk, LossWeights(), "cpu")
    x0_pred = torch.randn(2, 6, 38, requires_grad=True)
    loss, parts = la(x0_pred, batch)
    assert parts["contact_m2"] == 0.0
    assert np.isfinite(parts["total"])
    loss.backward()
    assert torch.isfinite(x0_pred.grad).all()

    # gradient equality with the contact term disabled
    la0 = LossAssembler(tk, LossWeights(w_contact=0.0), "cpu")
    x1 = x0_pred.detach().clone().requires_grad_(True)
    loss0, _ = la0(x1, batch)
    loss0.backward()
    torch.testing.assert_close(x0_pred.grad, x1.grad)


def test_supervision_mask_normalizes_by_valid_frames():
    """contact_m2 must equal the hand-computed per-side normalization over
    exactly the supervised frames (valid-frame normalization, /3 sites)."""
    from ireset.geometry.distance_adapter import point_box_sdf_torch

    tk, batch = _loss_env()
    B, F = 2, 6
    mask = torch.zeros(B, F, 2, dtype=torch.bool)
    mask[0, 2:4, 0] = True
    mask[1, 1, 1] = True
    batch["hand_contact_supervision_mask"] = mask
    la = LossAssembler(tk, LossWeights(), "cpu")
    x0 = batch["x0"]
    _, parts = la(x0, batch)

    pos, quat, q = la.decode_clip(x0)
    flat = lambda a: a.reshape(B * F, -1)
    from ireset.train.losses import HAND_SITE_NAMES
    hand_p = la.fk.site_world_positions(flat(pos), flat(quat), flat(q),
                                        HAND_SITE_NAMES).view(B, F, -1, 3)
    sdf = point_box_sdf_torch(la.hand_points_box_geom(hand_p, batch))   # [B,F,6]
    m = mask.to(sdf.dtype)
    left = sdf[..., 0:3] * m[..., 0:1]
    right = sdf[..., 3:6] * m[..., 1:2]
    expected = (left ** 2).sum() / m[..., 0].sum().clamp_min(1.0) / 3.0 \
        + (right ** 2).sum() / m[..., 1].sum().clamp_min(1.0) / 3.0
    assert parts["contact_m2"] == pytest.approx(float(expected), rel=1e-6)
    assert parts["contact_m2"] > 0.0


def test_supervision_mask_is_loss_only_no_conditioner_leakage():
    """Changing the supervision mask changes NOTHING except contact_m2:
    conditioner memory and every other loss part are bitwise identical."""
    tk, batch = _loss_env()
    mask_a = torch.zeros(2, 6, 2, dtype=torch.bool)
    mask_a[:, :, 0] = True
    mask_b = torch.zeros(2, 6, 2, dtype=torch.bool)
    cond = ConditionEncoder(32)
    mem_a = cond(batch)
    batch["hand_contact_supervision_mask"] = mask_a
    mem_b = cond(batch)
    torch.testing.assert_close(mem_a, mem_b)      # conditioner never sees the mask

    la = LossAssembler(tk, LossWeights(), "cpu")
    x0_pred = torch.randn(2, 6, 38)
    _, parts_a = la(x0_pred, batch)
    batch["hand_contact_supervision_mask"] = mask_b
    _, parts_b = la(x0_pred, batch)
    for key in ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2"):
        assert parts_a[key] == parts_b[key], key
    # velocity supervision stays full-window: the term is the plain
    # consecutive-frame difference error, identical under both masks


@pytest.mark.slow
def test_graded_manifest_to_loss_smoke(tmp_path):
    """Plan section 3.3 smoke: manifest -> windows -> condition encoding ->
    loss forward/backward. No training step, no checkpoint."""
    env = _graded_env(tmp_path, window=8)
    tk = _graded_tokenizer()
    ds = TeacherWindowDataset("val", tk, window=8, stride=4,
                              label_manifest=env["manifest"])
    assert len(ds) > 0
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    cond = ConditionEncoder(32)
    mem = cond(batch)
    assert torch.isfinite(mem).all()
    la = LossAssembler(tk, LossWeights(), "cpu")
    x0_pred = batch["x0"].clone().requires_grad_(True)
    loss, parts = la(x0_pred, batch)
    assert np.isfinite(parts["total"]) and "contact_m2" in parts
    loss.backward()
    assert torch.isfinite(x0_pred.grad).all()
