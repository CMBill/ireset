"""W1 A3/B/C interface contracts (omniretarget_w1_a3bc_plan §7).

Counterexamples per plan §7 row, on synthetic worlds (no derived/ or
resources/ dependency):

* entry (W0): tampered C2/UR verdicts are refused; registration is
  exclusive (reuse only after clean re-verification, never overwritten);
* list selection: per-subject hash-rank top-4 in canonical order;
* admission: structural damage (shape/fps/finiteness/quaternions/joint
  limits/T) fails with explicit reasons; legitimate normalization passes
  without touching the source; layout round-trip is sign-equivalent;
* scene split: source/target separation with the constant z offset kept;
  condition invariance under teacher-robot-only edits;
* consumption: manifest entries mutually exclusive; drifted
  inputs/derived/protocol and missing keyed schedules refuse with no
  fallback; not-admitted rows are excluded, never dropped from the record;
* stats/cache: val rows never move the train statistics; the schedule key
  covers input/code identity; windows never double-count frames;
* losses: w_contact=0 executes no contact branch (no hand-site FK/SDF/mask
  reads) while motion-FK gradients survive;
* validation: full-window coverage (no 40-batch cap), sample weighting,
  fixed t/noise, validation never perturbs the training RNG;
* generation: stitched windows cover the last frame; raw path builds zero
  RepairPipeline; model identity mismatch is refused; failures keep the
  expected-set denominator;
* S0: real optimizer-step vs micro-batch counting, fixed probe trend gate,
  budget stop -> incomplete, micro-batch gate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import pairing  # noqa: E402
from ireset.data import w1_training as w1  # noqa: E402
from ireset.data import windows as windows_mod  # noqa: E402
from ireset.data.teacher import build_interaction_clip  # noqa: E402
from ireset.model.tokens import RootStats, StateTokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SUBJECTS = ["sub10", "sub12", "sub16", "sub3", "sub7", "sub8"]


# --- synthetic demo construction ----------------------------------------------

def _joint_mid_range():
    from ireset.embodied.g1_spec import G1Spec

    rng_ = G1Spec().joint_range
    return (rng_[:, 0] + rng_[:, 1]) / 2.0, rng_


def make_synthetic_demo(demo: str, T: int = 90, seed: int = 0):
    """A published-like NPZ/PT pair whose pairing gates PASS exactly.

    npz human = exact similarity(pt human); npz box = similarity(pt box)
    + constant +z offset — the A1 §4.3 structure, bit-exact by
    construction.
    """
    rng = np.random.default_rng(seed)
    s = 1.1
    R = Rot.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    t = np.array([0.3, -0.2, 0.5])
    pt_human = np.cumsum(rng.normal(0, 0.01, (T, 52, 3)), axis=0) \
        + np.array([0.0, 0.0, 1.0])
    npz_human = (s * (pt_human.reshape(-1, 3) @ R.T) + t).reshape(T, 52, 3)

    pt_pos = np.cumsum(rng.normal(0, 0.004, (T, 3)), axis=0) + [0.6, 0.1, 0.25]
    q0 = rng.normal(size=4)
    q0 /= np.linalg.norm(q0)
    pt_quat_wxyz = q0[None, :].repeat(T, 0)
    Rpt = Rot.from_quat(pt_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    offset = np.array([0.0, 0.0, 0.12])
    npz_box_pos = s * (pt_pos @ R.T) + t + offset
    npz_box_quat = Rot.from_matrix(R[None] @ Rpt).as_quat()[:, [3, 0, 1, 2]]

    mid, jrange = _joint_mid_range()
    joints = mid[None, :] + rng.normal(0, 0.01, (T, 29))
    q1 = rng.normal(size=4)
    q1 /= np.linalg.norm(q1)
    root_quat = q1[None, :].repeat(T, 0)

    qpos_pub = np.zeros((T, 43))
    qpos_pub[:, 0:4] = root_quat
    qpos_pub[:, 4:7] = np.cumsum(rng.normal(0, 0.003, (T, 3)), axis=0)
    qpos_pub[:, 7:36] = joints
    qpos_pub[:, 36:40] = npz_box_quat
    qpos_pub[:, 40:43] = npz_box_pos

    pt = np.zeros((T, 325))
    pt[:, 162:318] = pt_human.reshape(T, -1)
    pt[:, 318:325] = np.concatenate(
        [pt_pos, pt_quat_wxyz[:, [1, 2, 3, 0]]], axis=1)
    return {"qpos": qpos_pub, "human": npz_human, "pt": pt, "fps": 30.0,
            "T": T, "subject": pairing.subject_of(demo)}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_demo_files(root: Path, demo: str, data: dict):
    npz_dir = root / "resources/data/omniretarget/robot-object"
    pt_dir = root / "resources/data/OMOMO_new"
    npz_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_dir / f"{demo}_original.npz", qpos=data["qpos"],
             human_joints=data["human"], fps=np.float64(data["fps"]))
    torch.save(torch.as_tensor(data["pt"], dtype=torch.float32),
               pt_dir / f"{demo}.pt")


def _append_index(dir_path: Path, entry: dict):
    dir_path.mkdir(parents=True, exist_ok=True)
    idx = dir_path / "attempts_index.json"
    attempts = json.loads(idx.read_text())["attempts"] if idx.is_file() else []
    attempts.append(entry)
    idx.write_text(json.dumps({"attempts": attempts}))


def _write_anchor_standins(root: Path, a1_dir: Path | None = None) -> dict[str, str]:
    """Write c2/ur/ur-prereg/source-box/a1 stand-ins AT the real anchor
    paths (inside the synthetic root) and return their hashes — the
    required nine-anchor set then re-verifies for every consumer."""
    if a1_dir is not None:
        for key, fname in (("a1_frozen_lists", "frozen_lists.json"),
                           ("a1_assets", "assets.json")):
            rel = w1.W1_PLAN_ANCHORS[key][0]
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(a1_dir / fname, dst)
    c2 = {"registration_status": {"complete": True},
          "baseline": {"verdict": "incomplete"},
          "recovery": {"status": "no_quality_gain"},
          "training": {"status": "not_ready"}}
    ur_prereg = {"experiment_id": "a2_recovery_step03_v1"}
    ur = {"verdict": "no_quality_gain",
          "admission": {"passed": True, "ur_coverage": {"missing": []}},
          "verdict_record": {"complete_evidence": True},
          "ur_identity": {"preregistration_sha256": None}}
    sources: dict[str, str] = {}
    for key, (rel, _) in w1.W1_PLAN_ANCHORS.items():
        dst = root / rel
        if key in ("c2_final_decision", "ur_full_comparison",
                   "ur_preregistration", "a1_pairing_source_boxes",
                   "split_manifest"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            if key == "c2_final_decision":
                dst.write_text(json.dumps(c2))
            elif key == "ur_preregistration":
                dst.write_text(json.dumps(ur_prereg))
            elif key == "a1_pairing_source_boxes":
                np.savez(dst)
    # bind the UR report to the ur-preregistration file hash
    sources["ur_preregistration"] = _sha(
        root / w1.W1_PLAN_ANCHORS["ur_preregistration"][0])
    ur["ur_identity"]["preregistration_sha256"] = sources["ur_preregistration"]
    ur_path = root / w1.W1_PLAN_ANCHORS["ur_full_comparison"][0]
    ur_path.write_text(json.dumps(ur))
    sources["ur_full_comparison"] = _sha(ur_path)
    sources["c2_final_decision"] = _sha(
        root / w1.W1_PLAN_ANCHORS["c2_final_decision"][0])
    sources["a1_pairing_source_boxes"] = _sha(
        root / w1.W1_PLAN_ANCHORS["a1_pairing_source_boxes"][0])
    sources["split_manifest"] = _sha(
        root / w1.W1_PLAN_ANCHORS["split_manifest"][0])
    return sources


def build_synthetic_world(tmp_path: Path, *, n_train: int = 2, n_val: int = 2,
                          T: int = 90, val_T: int | None = None,
                          seed0: int = 0) -> dict:
    """Full synthetic project root + W1 run root (prereg/audit/cache)."""
    root = tmp_path / "proj"
    demos: dict[str, dict] = {}
    train_demos, val_demos = [], []
    k = 0
    for i in range(n_train):
        d = f"{SUBJECTS[i % len(SUBJECTS)]}_largebox_{i:03d}"
        demos[d] = make_synthetic_demo(d, T=T, seed=seed0 + k)
        train_demos.append(d)
        k += 1
    for i in range(n_val):
        d = f"{SUBJECTS[(n_train + i) % len(SUBJECTS)]}_largebox_{100 + i:03d}"
        demos[d] = make_synthetic_demo(d, T=(val_T or T), seed=seed0 + k)
        val_demos.append(d)
        k += 1
    for d, data in demos.items():
        _write_demo_files(root, d, data)

    split_dir = root / "derived/data/split"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.json").write_text(json.dumps({"demos": sorted(train_demos),
                                                      "files": {}}))
    (split_dir / "val.json").write_text(json.dumps({"demos": sorted(val_demos),
                                                    "files": {}}))
    (split_dir / "manifest.json").write_text(
        json.dumps({"unsupervised_reserve_no_teacher": []}))

    a1 = root / "derived/experiments/omniretarget_first/a1/freeze"
    a1.mkdir(parents=True, exist_ok=True)
    npz_dir = root / "resources/data/omniretarget/robot-object"
    pt_dir = root / "resources/data/OMOMO_new"

    def entry(d):
        return {"demo": d, "T": demos[d]["T"],
                "npz": f"resources/data/omniretarget/robot-object/{d}_original.npz",
                "npz_sha256": _sha(npz_dir / f"{d}_original.npz"),
                "pt": f"resources/data/OMOMO_new/{d}.pt",
                "pt_sha256": _sha(pt_dir / f"{d}.pt")}

    val0 = sorted(val_demos)[0]
    (a1 / "frozen_lists.json").write_text(json.dumps({
        "teacher_pilot_train6_v1": [entry(train_demos[0])],
        "val_originals_v1": [entry(d) for d in sorted(val_demos)],
        "val6_pilot_v1": [entry(val0)],
    }))
    (a1 / "assets.json").write_text(json.dumps({
        "published": {"models": {
            f"resources/data/omniretarget/robot-object/{val0}_original.npz":
                {"bytes": (npz_dir / f"{val0}_original.npz").stat().st_size,
                 "sha256": _sha(npz_dir / f"{val0}_original.npz")}}},
        "upstream": {"height_dict":
                     f"resources/data/omniretarget/robot-object/{val0}_original.npz",
                     "height_dict_sha256": _sha(npz_dir / f"{val0}_original.npz"),
                     "subject_heights_m": {}},
    }))

    run = root / "derived/experiments/omniretarget_first/run1"
    freeze = run / "freeze"
    freeze.mkdir(parents=True, exist_ok=True)
    plan = root / "plan.md"
    plan.write_text("# synthetic w1 plan\n")
    train24 = w1.select_train24(sorted(train_demos))
    val31 = sorted(val_demos)
    # anchor stand-ins at the REAL W1_PLAN_ANCHORS paths (inside the
    # synthetic root) so the required-NINE-anchor set check passes and every
    # consumer's re-verification finds exactly what the prereg registers
    anchor_sources = _write_anchor_standins(root, a1_dir=a1)
    anchor_recs = {
        key: {"path": rel, "sha256": _sha(root / rel)}
        for key, (rel, _) in w1.W1_PLAN_ANCHORS.items()}
    assert set(anchor_recs) == set(w1.W1_PLAN_ANCHORS)    # all nine required
    prereg = {
        "identity": {"experiment_id": w1.EXPERIMENT_ID,
                     "protocol": w1.MOTION_PROTOCOL_VERSION,
                     "training_purpose": w1.TRAINING_PURPOSE,
                     "source_type": w1.SOURCE_TYPE,
                     "published_revision": w1.PUBLISHED_REVISION},
        "plan": {"path": "plan.md", "sha256": _sha(plan)},
        "anchors": anchor_recs,
        "selection": {"train24": train24,
                      "train24_canonical_sha256": w1.canonical_list_sha256(train24),
                      "val31": val31, "val6": [val31[0]],
                      "pilot6_from_a1": [train_demos[0]]},
        "inputs": {d: {"npz": _sha(npz_dir / f"{d}_original.npz"),
                       "pt": _sha(pt_dir / f"{d}.pt")}
                   for d in sorted(set(train24) | set(val31))},
        "assets": {"a1_freeze_path": str(a1.relative_to(root)),
                   "a1_assets_path": str((a1 / "assets.json").relative_to(root)),
                   "a1_assets_sha256": _sha(a1 / "assets.json")},
    }
    (freeze / "preregistration.json").write_text(json.dumps(prereg))

    audit_dir = run / "audit/attempt_001"
    derived_dir = audit_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, dd in (("train", train24), ("val", val31)):
        for d in dd:
            data = demos[d]
            Td = data["T"]
            src_box, sim_scale = w1.derive_source_box(
                data["pt"][:, 162:318].reshape(Td, 52, 3), data["human"],
                data["pt"][:, 318:325])
            dpath = derived_dir / f"{d}_w1_scene.npz"
            np.savez(dpath, qpos=data["qpos"], human_joints=data["human"],
                     fps=np.float64(30.0), source_obj_pose=src_box,
                     target_obj_pose=w1.published_to_project(data["qpos"])[:, 36:43],
                     similarity_scale=np.float64(sim_scale),
                     meta_json=json.dumps(
                         {"demo": d, "split": split,
                          "protocol": w1.MOTION_PROTOCOL_VERSION,
                          "npz_sha256": _sha(npz_dir / f"{d}_original.npz"),
                          "pt_sha256": _sha(pt_dir / f"{d}.pt"),
                          "similarity_scale": sim_scale,
                          "layout": "published_verbatim"}))
            rows.append({
                "demo": d, "split": split,
                "subject": pairing.subject_of(d),
                "admission": {"motion": True, "reasons": []},
                "inputs": {"npz_sha256": _sha(npz_dir / f"{d}_original.npz"),
                           "pt_sha256": _sha(pt_dir / f"{d}.pt")},
                "derived": {"path": str(dpath.relative_to(root)),
                            "sha256": _sha(dpath)},
                "T": Td, "fps": 30.0,
                "full_windows_F60_s30": w1.full_windows(Td),
            })
    manifest = {
        "manifest_name": w1.MANIFEST_NAME, "protocol": w1.MOTION_PROTOCOL_VERSION,
        "experiment_id": w1.EXPERIMENT_ID, "training_purpose": w1.TRAINING_PURPOSE,
        "source_type": w1.SOURCE_TYPE, "published_revision": w1.PUBLISHED_REVISION,
        "preregistration": {"path": str((freeze / "preregistration.json").relative_to(root)),
                            "sha256": _sha(freeze / "preregistration.json")},
        "selection": {"train24": train24, "val31": val31, "val6": [val31[0]]},
        "rows": rows,
    }
    (audit_dir / "training_manifest.json").write_text(json.dumps(manifest))

    for rel in w1.CONDITION_CODE_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    code_sha = w1.condition_code_sha256(root)
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    cache_dir = run / "cache/attempt_001"
    (cache_dir / "schedules").mkdir(parents=True, exist_ok=True)
    from ireset.embodied.contact_sites import infer_reference_schedule
    from ireset.data.teacher import schedule_to_cache

    schedule_rows = []
    for r in rows:
        key = w1.schedule_cache_key(r["inputs"]["npz_sha256"],
                                    r["inputs"]["pt_sha256"], assets_sha, code_sha)
        with np.load(root / r["derived"]["path"], allow_pickle=False) as z:
            human, src = z["human_joints"], z["source_obj_pose"]
            fps, scale = float(z["fps"]), float(z["similarity_scale"])
        sched = infer_reference_schedule(human, src, fps=fps, box_scale=scale)
        sched_file = cache_dir / "schedules" / w1.schedule_cache_filename(
            r["demo"], key)
        np.savez(sched_file, intervals=schedule_to_cache(sched))
        schedule_rows.append({"demo": r["demo"], "split": r["split"],
                              "key": key[:16], "sha256": _sha(sched_file),
                              "full_windows": r["full_windows_F60_s30"]})
    stats, facts = w1.fit_w1_root_stats(rows, root, cache_dir / "schedules",
                                        assets_sha, code_sha)
    stats.save(cache_dir / "root_stats.npz")
    (cache_dir / "cache_identity.json").write_text(json.dumps({
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "preregistration": {"sha256": _sha(freeze / "preregistration.json")},
        "training_manifest": {
            "path": str((audit_dir / "training_manifest.json").relative_to(root)),
            "sha256": _sha(audit_dir / "training_manifest.json")},
        "condition_code_sha256": code_sha,
        "assets_sha256": assets_sha,
        "schedule_cache_key_version": w1.SCHEDULE_KEY_VERSION,
        "complete": True,
        "root_stats": {"file_sha256": _sha(cache_dir / "root_stats.npz"),
                       "frames_fitted": facts["frames_fitted"],
                       "n_train_rows_used": facts["n_train_rows_used"],
                       "files": facts["files"]},
        "schedule_rows": schedule_rows,
    }))
    _append_index(run / "audit", {
        "attempt": "attempt_001",
        "manifest": str((audit_dir / "training_manifest.json").relative_to(root)),
        "scope": train24 + val31})
    _append_index(run / "cache", {
        "attempt": "attempt_001",
        "identity": str((cache_dir / "cache_identity.json").relative_to(root))})
    # the fixture cache attempt is pre-existing evidence: a real run root
    # ALWAYS carries its ledger row (round-3 three-way reconciliation in
    # precompute_m3_cache._historical_cache_cost_s refuses disk-without-
    # ledger history)
    (run / "costs.json").write_text(json.dumps({"invocations": [{
        "timestamp": "2026-10-12T00:00:00",
        "stage": "w1_cache", "attempt": "attempt_001",
        "status": "done", "wall_s": 0.0}]}))
    return {"root": root, "run": run, "prereg_path": freeze / "preregistration.json",
            "manifest_path": audit_dir / "training_manifest.json",
            "cache_dir": cache_dir, "demos": demos, "train24": train24,
            "val31": val31, "npz_dir": npz_dir, "pt_dir": pt_dir, "T": T}


def _patch_paths(monkeypatch, world):
    """Patch both the ireset.paths attributes (function-level imports) and
    the module-level bindings of the consumers under test."""
    import ireset.data.a2_integrity as integ
    import ireset.paths as paths_mod

    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(paths_mod, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(paths_mod, "OMOMO_NEW_DIR", world["pt_dir"])
    monkeypatch.setattr(windows_mod, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(integ, "DERIVED_DIR", world["root"] / "derived")


def _resync_cache_identity(world):
    """Reconcile the fixture cache-identity record with the CURRENT
    manifest's admitted set (file hashes unchanged — the files on disk are
    still the produced ones; only the set bookkeeping follows the edit)."""
    manifest = json.loads(world["manifest_path"].read_text())
    ident_path = world["cache_dir"] / "cache_identity.json"
    ident = json.loads(ident_path.read_text())
    ident["training_manifest"]["sha256"] = _sha(world["manifest_path"])
    ident["preregistration"]["sha256"] = _sha(world["prereg_path"])
    admitted = [r for r in manifest["rows"]
                if r.get("admission", {}).get("motion")]
    by_demo = {rec["demo"]: rec for rec in ident["schedule_rows"]}
    ident["schedule_rows"] = [by_demo[r["demo"]] for r in admitted
                              if r["demo"] in by_demo]
    train = [r for r in admitted if r["split"] == "train"]
    ident["root_stats"]["files"] = [r["demo"] for r in train]
    ident["root_stats"]["frames_fitted"] = sum(r.get("T", 0) for r in train)
    ident_path.write_text(json.dumps(ident))
    return manifest


def _tokenizer(world):
    from ireset.embodied.g1_spec import G1Spec

    return StateTokenizer(G1Spec().joint_range,
                          RootStats.load(world["cache_dir"] / "root_stats.npz"))


def _repin(world, edit):
    """Apply an edit to the manifest dict and re-pin the cache binding."""
    manifest = json.loads(world["manifest_path"].read_text())
    edit(manifest)
    world["manifest_path"].write_text(json.dumps(manifest))
    ident = json.loads((world["cache_dir"] / "cache_identity.json").read_text())
    ident["training_manifest"]["sha256"] = _sha(world["manifest_path"])
    (world["cache_dir"] / "cache_identity.json").write_text(json.dumps(ident))
    return manifest


# --- list selection -----------------------------------------------------------

def test_select_train24_canonical_order_and_cap():
    demos = [f"subB_largebox_{i:03d}" for i in range(6)] + \
            [f"subA_largebox_{i:03d}" for i in range(6)]
    sel = w1.select_train24(demos)
    assert len(sel) == 8
    subjects = [pairing.subject_of(d) for d in sel]
    assert subjects == sorted(subjects)
    for subj in ("subA", "subB"):
        want = pairing.rank_by_hash(
            [d for d in demos if d.startswith(subj)], w1.TRAIN24_HASH_PREFIX)[:4]
        assert [d for d in sel if d.startswith(subj)] == want
    assert w1.canonical_list_sha256(sel) != w1.canonical_list_sha256(sel[::-1])


# --- admission (§3.2) ----------------------------------------------------------

def _good_arrays():
    return make_synthetic_demo("sub10_largebox_000", T=90, seed=3)


def test_admission_counterexamples_each_fail_explicitly():
    _, jrange = _joint_mid_range()
    data = _good_arrays()
    ok = w1.check_published_admission(
        "d", data["qpos"], data["human"], 30.0, 90, jrange)
    assert ok["motion"] and not ok["reasons"]

    bad_human = data["human"][..., :2]
    r = w1.check_published_admission("d", data["qpos"], bad_human, 30.0, 90, jrange)
    assert not r["motion"] and any("human_shape" in x for x in r["reasons"])

    r = w1.check_published_admission("d", data["qpos"], data["human"], 50.0, 90, jrange)
    assert not r["motion"] and any("fps_not_30" in x for x in r["reasons"])

    nf = data["qpos"].copy()
    nf[10, 20] = np.nan
    r = w1.check_published_admission("d", nf, data["human"], 30.0, 90, jrange)
    assert not r["motion"] and "qpos_not_finite" in r["reasons"]

    deg = data["qpos"].copy()
    deg[:, 0:4] *= 0.5
    r = w1.check_published_admission("d", deg, data["human"], 30.0, 90, jrange)
    assert not r["motion"] and any("quat_norm_out_of_range" in x for x in r["reasons"])

    jl = data["qpos"].copy()
    jl[:, 7] = jrange[0, 0] - 0.5
    r = w1.check_published_admission("d", jl, data["human"], 30.0, 90, jrange)
    assert not r["motion"] and any("joint_limit_violation" in x for x in r["reasons"])

    r = w1.check_published_admission("d", data["qpos"], data["human"], 30.0, 80, jrange)
    assert not r["motion"] and any("T_mismatch_vs_pt" in x for x in r["reasons"])

    r = w1.check_published_admission("d", data["qpos"], data["human"], True, 90, jrange)
    assert not r["motion"] and "fps_missing_not_finite" in r["reasons"]


def test_admission_legal_normalization_registered_not_written_back():
    _, jrange = _joint_mid_range()
    data = _good_arrays()
    q = data["qpos"].copy()
    q[:, 0:4] *= 1.0 + 1e-9
    before = q.copy()
    r = w1.check_published_admission("d", q, data["human"], 30.0, 90, jrange)
    assert r["motion"], r["reasons"]
    assert r["facts"]["quat_normalization_registered"] is True
    assert np.array_equal(q, before)
    assert w1.published_roundtrip_ok(q)
    r2 = w1.check_published_admission("d", data["qpos"], data["human"], 30.0, 90, jrange)
    assert r2["facts"]["quat_normalization_registered"] is False


def test_published_roundtrip_sign_equivalence():
    data = _good_arrays()
    q = data["qpos"].copy()
    q[5, 0:4] *= -1.0
    assert w1.published_roundtrip_ok(q)


# --- scene split + condition invariance ----------------------------------------

def test_scene_split_facts_pure_z_offset_kept():
    data = _good_arrays()
    scene = w1.published_to_project(data["qpos"])
    src, scale = w1.derive_source_box(data["pt"][:, 162:318].reshape(90, 52, 3),
                                      data["human"], data["pt"][:, 318:325])
    assert 0.0 < scale < 2.0
    tgt = scene[:, 36:43]
    facts = w1.scene_split_facts(scene, src, tgt)
    assert facts["offset_pure_z"] is True
    assert facts["same_trajectory"] is False
    assert abs(facts["offset_mean_m"][2] - 0.12) < 1e-9
    resid = np.abs((tgt[:, 0:3] - src[:, 0:3]) - np.array([0, 0, 0.12])).max()
    assert resid < 1e-9


def test_condition_invariant_under_teacher_robot_only_edit(tmp_path):
    data = _good_arrays()
    d = tmp_path / "npz"
    d.mkdir()
    name = "sub10_largebox_000_original.npz"
    np.savez(d / name, qpos=data["qpos"], human_joints=data["human"],
             fps=np.float64(30.0))
    src, scale = w1.derive_source_box(data["pt"][:, 162:318].reshape(90, 52, 3),
                                      data["human"], data["pt"][:, 318:325])
    base = build_interaction_clip(name, teacher_traj_dir=d, source_obj_pose=src,
                                  source_box_scale=scale)

    edited = data["qpos"].copy()
    edited[:, 7:36] += 0.05
    edited[:, 4:7] += 0.03
    np.savez(d / name, qpos=edited, human_joints=data["human"],
             fps=np.float64(30.0))
    other = build_interaction_clip(name, teacher_traj_dir=d, source_obj_pose=src,
                                   source_box_scale=scale)
    a, b = base.condition, other.condition
    assert np.array_equal(a.human_kps_world, b.human_kps_world)
    assert np.array_equal(a.source_obj_pose, b.source_obj_pose)
    assert np.array_equal(a.target_obj_pose, b.target_obj_pose)
    assert np.array_equal(a.world_from_clip, b.world_from_clip)
    assert [(iv.start, iv.end) for iv in a.contacts] == \
           [(iv.start, iv.end) for iv in b.contacts]
    assert not np.array_equal(base.q, other.q)


def test_legacy_entry_keeps_source_equals_target(tmp_path):
    data = _good_arrays()
    d = tmp_path / "npz"
    d.mkdir()
    name = "sub10_largebox_000_original.npz"
    np.savez(d / name, qpos=data["qpos"], human_joints=data["human"],
             fps=np.float64(30.0))
    clip = build_interaction_clip(name, teacher_traj_dir=d)
    assert np.array_equal(clip.condition.source_obj_pose,
                          clip.condition.target_obj_pose)


# --- preregistration verification ----------------------------------------------

def test_verify_preregistration_flags_each_leg(tmp_path):
    world = build_synthetic_world(tmp_path)
    root = world["root"]
    prereg = json.loads(world["prereg_path"].read_text())
    ok = w1.verify_preregistration(prereg, root, world["npz_dir"], world["pt_dir"])
    assert ok["passed"], ok["problems"]

    bad = json.loads(json.dumps(prereg))
    bad["identity"]["protocol"] = "graded_labels_v1"
    assert "identity_mismatch" in w1.verify_preregistration(
        bad, root, world["npz_dir"], world["pt_dir"])["problems"]

    bad = json.loads(json.dumps(prereg))
    bad["plan"]["sha256"] = "0" * 64
    assert any(p.startswith("plan_drifted") for p in w1.verify_preregistration(
        bad, root, world["npz_dir"], world["pt_dir"])["problems"])

    bad = json.loads(json.dumps(prereg))
    bad["anchors"]["split_train"]["sha256"] = "0" * 64
    assert any(p.startswith("anchor_drifted") for p in w1.verify_preregistration(
        bad, root, world["npz_dir"], world["pt_dir"])["problems"])

    bad = json.loads(json.dumps(prereg))
    bad["selection"]["train24"] = list(reversed(bad["selection"]["train24"]))
    assert "train24_not_reproducible_from_split" in w1.verify_preregistration(
        bad, root, world["npz_dir"], world["pt_dir"])["problems"]

    victim = world["npz_dir"] / f"{world['train24'][0]}_original.npz"
    victim.write_bytes(victim.read_bytes() + b"x")
    got = w1.verify_preregistration(prereg, root, world["npz_dir"], world["pt_dir"])
    assert any(p.startswith("input_drifted") for p in got["problems"])


# --- dataset consumption ---------------------------------------------------------

def _w1_dataset(world, split, **kw):
    return windows_mod.TeacherWindowDataset(
        split, _tokenizer(world), window=60, stride=30,
        training_manifest=world["manifest_path"],
        schedule_cache_dir=world["cache_dir"] / "schedules", **kw)


def test_dataset_loads_and_counts_frames_once(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    ds = _w1_dataset(world, "train")
    T = world["T"]
    n_clips = len(world["train24"])
    assert len(ds.clips) == n_clips
    assert len(ds) == n_clips * w1.full_windows(T)
    assert ds.frame_count() == n_clips * T        # windows never double-count
    item = ds[0]
    assert item["x0"].shape == (60, 38)
    assert item["kps_obj"].shape == (60, 52, 3)


def test_dataset_mutually_exclusive_manifests(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    with pytest.raises(ValueError, match="mutually exclusive"):
        windows_mod.TeacherWindowDataset(
            "train", _tokenizer(world), training_manifest=world["manifest_path"],
            schedule_cache_dir=world["cache_dir"] / "schedules",
            label_manifest="whatever.json")


def test_dataset_refuses_drifted_input_derived_protocol(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)

    victim = world["npz_dir"] / f"{world['train24'][0]}_original.npz"
    victim.write_bytes(victim.read_bytes() + b"x")
    # the input drift is caught by the preregistration re-verification leg
    # (the registration pins the input hashes; the row leg never runs)
    with pytest.raises(ValueError, match="input_drifted"):
        _w1_dataset(world, "train")
    victim.write_bytes(victim.read_bytes()[:-1])

    rows = json.loads(world["manifest_path"].read_text())["rows"]
    drow = next(r for r in rows if r["demo"] == world["train24"][0])
    dpath = world["root"] / drow["derived"]["path"]
    dpath.write_bytes(dpath.read_bytes() + b"x")
    with pytest.raises(ValueError, match="derived representation missing or drifted"):
        _w1_dataset(world, "train")
    dpath.write_bytes(dpath.read_bytes()[:-1])

    def set_protocol(proto):
        manifest = json.loads(world["manifest_path"].read_text())
        manifest["protocol"] = proto
        world["manifest_path"].write_text(json.dumps(manifest))

    set_protocol("graded_labels_v1")
    with pytest.raises(ValueError, match="wrong source/protocol"):
        _w1_dataset(world, "train")
    set_protocol(w1.MOTION_PROTOCOL_VERSION)
    _w1_dataset(world, "train")


def test_dataset_refuses_missing_keyed_schedule_no_fallback(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    victim = sorted((world["cache_dir"] / "schedules").glob("*.npz"))[0]
    victim.unlink()
    # P1-6: the content re-hash catches the missing file at startup (before
    # any clip load); with content verification satisfied some other way the
    # keyed-file miss inside load_w1_derived_clip still refuses (no fallback)
    with pytest.raises(ValueError, match="schedule_file_missing"):
        _w1_dataset(world, "train")


def test_dataset_excludes_not_admitted_rows(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    demo = world["train24"][0]

    def edit(manifest):
        next(r for r in manifest["rows"]
             if r["demo"] == demo)["admission"] = {"motion": False,
                                                   "reasons": ["synthetic"]}

    _repin(world, edit)
    _resync_cache_identity(world)
    ds = _w1_dataset(world, "train")
    assert ds.excluded["not_admitted"] == [demo]
    assert all(lc.demo != demo for lc in ds.clips)


def test_dataset_condition_invariance_vs_teacher_edit(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    ds = _w1_dataset(world, "train")
    demo = world["train24"][0]

    def edit(manifest):
        row = next(r for r in manifest["rows"] if r["demo"] == demo)
        dpath = world["root"] / row["derived"]["path"]
        with np.load(dpath, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in ("qpos", "human_joints", "fps",
                                             "source_obj_pose", "target_obj_pose",
                                             "similarity_scale")}
            meta = json.loads(str(z["meta_json"]))
        data["qpos"][:, 7:36] += 0.04          # teacher joints only
        np.savez(dpath, meta_json=json.dumps(meta), **data)
        row["derived"]["sha256"] = _sha(dpath)

    _repin(world, edit)
    ds2 = _w1_dataset(world, "train")
    i1 = next(i for i, (ci, s) in enumerate(ds.index) if ds.clips[ci].demo == demo)
    i2 = next(i for i, (ci, s) in enumerate(ds2.index) if ds2.clips[ci].demo == demo)
    a, b = ds[i1], ds2[i2]
    for key in ("kps_obj", "kps_clip", "box_t_clip", "box_quat", "contact", "wfc"):
        assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["x0"], b["x0"])    # labels DID move


# --- stats/cache identity --------------------------------------------------------

def test_root_stats_train_only_val_never_contributes(tmp_path):
    world = build_synthetic_world(tmp_path)
    root = world["root"]
    code_sha = w1.condition_code_sha256(root)
    prereg = json.loads(world["prereg_path"].read_text())
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    rows = json.loads(world["manifest_path"].read_text())["rows"]
    stats_a, facts_a = w1.fit_w1_root_stats(rows, root,
                                            world["cache_dir"] / "schedules",
                                            assets_sha, code_sha)
    assert facts_a["n_train_rows_used"] == len(world["train24"])
    assert facts_a["n_rows_seen"] == len(rows)
    val_row = next(r for r in rows if r["split"] == "val")
    dpath = root / val_row["derived"]["path"]
    with np.load(dpath, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ("qpos", "human_joints", "fps",
                                         "source_obj_pose", "target_obj_pose")}
        meta = json.loads(str(z["meta_json"]))
    data["qpos"][:, 4:7] += 0.5
    np.savez(dpath, meta_json=json.dumps(meta), **data)
    stats_b, _ = w1.fit_w1_root_stats(rows, root, world["cache_dir"] / "schedules",
                                      assets_sha, code_sha)
    assert np.array_equal(stats_a.mean, stats_b.mean)
    assert np.array_equal(stats_a.std, stats_b.std)


def test_schedule_cache_key_covers_identity():
    base = w1.schedule_cache_key("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    assert base == w1.schedule_cache_key("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    variants = [("a2" + "0" * 62, "b" * 64, "c" * 64, "d" * 64),
                ("a" * 64, "b2" + "0" * 62, "c" * 64, "d" * 64),
                ("a" * 64, "b" * 64, "c2" + "0" * 62, "d" * 64),
                ("a" * 64, "b" * 64, "c" * 64, "d2" + "0" * 62)]
    for i, args in enumerate(variants):
        assert w1.schedule_cache_key(*args) != base, i
    assert w1.schedule_cache_filename("demo", base).startswith("demo__")


# --- losses gating ----------------------------------------------------------------

def _loss_batch(B=2, F=8):
    return {
        "x0": torch.randn(B, F, 38),
        "kps_obj": torch.randn(B, F, 52, 3),
        "kps_clip": torch.randn(B, F, 52, 3),
        "box_t_clip": torch.randn(B, F, 3),
        "box_quat": torch.nn.functional.normalize(torch.randn(B, F, 4), dim=-1),
        "contact": torch.zeros(B, F, 4),
        "wfc": torch.eye(4).repeat(B, 1, 1),
    }


def _tokenizer_for_losses():
    from ireset.embodied.g1_spec import G1Spec

    return StateTokenizer(G1Spec().joint_range,
                          RootStats(mean=np.zeros(3), std=np.ones(3)))


def test_w_contact_zero_executes_no_contact_branch(monkeypatch):
    import ireset.train.losses as L
    from ireset.train.losses import LossAssembler, LossWeights

    asm = LossAssembler(_tokenizer_for_losses(), LossWeights(w_contact=0.0),
                        device="cpu")
    fk_calls = {"n": 0}
    orig = asm.fk.site_world_positions

    def counting(*a, **k):
        fk_calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(asm.fk, "site_world_positions", counting)

    def sdf_must_not_run(*a, **k):
        raise AssertionError("SDF executed with w_contact=0")

    monkeypatch.setattr(L, "point_box_sdf_torch", sdf_must_not_run)
    batch = _loss_batch()
    x0 = batch["x0"].clone().requires_grad_(True)
    loss, parts = asm(x0, batch)
    assert "contact_m2" not in parts
    assert fk_calls["n"] == 2                     # pred+teacher FK only
    loss.backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()


def test_w_contact_positive_still_computes_contact():
    from ireset.train.losses import LossAssembler, LossWeights

    asm = LossAssembler(_tokenizer_for_losses(),
                        LossWeights(w_contact=0.5), device="cpu")
    batch = _loss_batch()
    loss, parts = asm(batch["x0"], batch)
    assert "contact_m2" in parts


# --- trainer validation ------------------------------------------------------------

class _FakeDS(torch.utils.data.Dataset):
    KEYS = ("x0", "kps_obj", "kps_clip", "box_t_clip", "box_quat",
            "contact", "wfc")

    def __init__(self, n, F=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x0 = torch.randn(n, F, 38, generator=g)
        self.kps_obj = torch.randn(n, F, 52, 3, generator=g)
        self.kps_clip = torch.randn(n, F, 52, 3, generator=g)
        self.box_t_clip = torch.randn(n, F, 3, generator=g)
        self.box_quat = torch.nn.functional.normalize(
            torch.randn(n, F, 4, generator=g), dim=-1)
        self.contact = torch.zeros(n, F, 4)
        self.wfc = torch.eye(4).repeat(n, 1, 1)

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, i):
        return {k: self.__dict__[k][i] for k in self.KEYS}


def _tiny_trainer(tmp_path, model_type, n_val, batch_size):
    from ireset.train.losses import LossWeights
    from ireset.train.trainer import TrainConfig, Trainer

    cfg = TrainConfig(model_type=model_type, d_model=32, n_layers=1, n_heads=2,
                      time_window=3, ffn_mult=2, device="cpu",
                      micro_batch=batch_size, num_workers=0, seed=11,
                      val_seed=13, loss=LossWeights(w_contact=0.0))
    return Trainer(cfg, _tokenizer_for_losses(), _FakeDS(8),
                   _FakeDS(n_val, seed=1),
                   tmp_path / f"out_{model_type}_{n_val}_{batch_size}")


def test_validate_covers_all_windows_and_weights_batches(tmp_path):
    tr = _tiny_trainer(tmp_path, "deterministic", n_val=6, batch_size=4)
    val = tr.validate()
    assert val["n_windows_evaluated"] == 6
    assert val["n_val_windows_total"] == 6
    per = []
    for i in range(6):
        single = {k: tr.val_ds.__dict__[k][i:i + 1] for k in _FakeDS.KEYS}
        _, parts = tr._forward(single)
        per.append(parts["pos"])
    assert abs(val["pos"] - float(np.mean(per))) < 1e-6


def test_validate_no_40_batch_truncation(tmp_path):
    tr = _tiny_trainer(tmp_path, "deterministic", n_val=200, batch_size=4)
    val = tr.validate()
    assert val["n_windows_evaluated"] == 200


def test_validate_fixed_noise_and_rng_isolation(tmp_path):
    tr = _tiny_trainer(tmp_path, "diffusion", n_val=8, batch_size=4)
    v1 = tr.validate()
    v2 = tr.validate()
    assert v1["total"] == v2["total"]
    torch.manual_seed(101)
    expect = torch.rand(4)
    torch.manual_seed(101)
    tr.validate()
    got = torch.rand(4)
    assert torch.equal(expect, got)
    tr.save("probe.pt", 0, v1, identity={"experiment_id": "x"})
    ck = torch.load(tr.out_dir / "probe.pt", weights_only=False)
    assert ck["identity"] == {"experiment_id": "x"}


# --- train config strictness --------------------------------------------------------

def test_load_train_config_rejects_unknown_keys(tmp_path):
    import train_generation as tg

    (tmp_path / "good.yaml").write_text(
        "model_type: diffusion\nsteps: 10\nloss:\n  w_contact: 0.0\n")
    cfg, meta = tg.load_train_config(tmp_path / "good.yaml")
    assert cfg.steps == 10 and cfg.loss.w_contact == 0.0
    (tmp_path / "bad.yaml").write_text("model_type: diffusion\nstepz: 10\n")
    with pytest.raises(SystemExit, match="unknown train-config keys"):
        tg.load_train_config(tmp_path / "bad.yaml")
    (tmp_path / "badloss.yaml").write_text("loss:\n  w_cntact: 0.0\n")
    with pytest.raises(SystemExit, match="unknown loss keys"):
        tg.load_train_config(tmp_path / "badloss.yaml")
    (tmp_path / "meta.yaml").write_text(
        "steps: 5\nformal_validation_contract:\n  full_generation_seed: 7\n")
    cfg, meta = tg.load_train_config(tmp_path / "meta.yaml")
    assert meta["formal_validation_contract"]["full_generation_seed"] == 7


def test_w1_train_mode_requires_all_explicit_args(tmp_path, monkeypatch):
    import train_generation as tg

    monkeypatch.setattr("sys.argv", [
        "prog", "--model", "diffusion",
        "--training-manifest", str(tmp_path / "m.json")])
    with pytest.raises(SystemExit, match="W1 mode requires explicit"):
        tg.main()


# --- generation: stitch + raw eval ---------------------------------------------------

def test_window_starts_cover_last_frame():
    from ireset.model.stitch import window_starts

    assert window_starts(90, 60, 30) == [0, 30]
    assert window_starts(100, 60, 30) == [0, 30, 40]
    assert window_starts(61, 60, 30) == [0, 1]
    assert window_starts(50, 60, 30) == [0]


class _StubGen:
    def __init__(self, fail=False):
        self.fail = fail

    def sample(self, cond, seed=0):
        if self.fail:
            raise RuntimeError("synthetic generation failure")
        from ireset.data.schema import RobotTrajectory

        n = cond.T
        root = np.zeros((n, 7))
        root[:, 6] = 1.0
        return RobotTrajectory(
            root=root, q=np.zeros((n, 29)), frame_dt=cond.frame_dt,
            metadata={"generation_wall_s": 0.1, "boundary": {},
                      "degenerate_windows": 0})


def test_w1_raw_eval_zero_repair_and_denominator_kept(tmp_path, monkeypatch):
    import eval_generation as eg

    world = build_synthetic_world(tmp_path, n_val=2, T=90, val_T=50)
    _patch_paths(monkeypatch, world)

    def boom(*a, **k):
        raise AssertionError("RepairPipeline constructed on the raw path")

    monkeypatch.setattr(eg, "RepairPipeline", boom)

    # shorten one val derived below the window length: the row stays in the
    # expected set as unsupported (val_T=50 already gives both val rows T=50;
    # instead rebuild with mixed lengths by extending the SECOND val demo)
    demos = world["demos"]
    long_demo = world["val31"][1]
    data = make_synthetic_demo(long_demo, T=90, seed=99)
    demos[long_demo] = data
    _write_demo_files(world["root"], long_demo, data)

    def edit(manifest):
        row = next(r for r in manifest["rows"] if r["demo"] == long_demo)
        dpath = world["root"] / row["derived"]["path"]
        src_box, sim_scale = w1.derive_source_box(
            data["pt"][:, 162:318].reshape(90, 52, 3), data["human"],
            data["pt"][:, 318:325])
        np.savez(dpath, qpos=data["qpos"], human_joints=data["human"],
                 fps=np.float64(30.0), source_obj_pose=src_box,
                 target_obj_pose=w1.published_to_project(data["qpos"])[:, 36:43],
                 similarity_scale=np.float64(sim_scale),
                 meta_json=json.dumps({"demo": long_demo, "split": "val",
                                       "protocol": w1.MOTION_PROTOCOL_VERSION,
                                       "similarity_scale": sim_scale,
                                       "layout": "published_verbatim"}))
        row.update({"T": 90, "full_windows_F60_s30": 1,
                    "derived": {"path": row["derived"]["path"],
                                "sha256": _sha(dpath)}})

    _repin(world, edit)
    # P1-8: the eval re-runs the preregistration legs, so the rewritten
    # demo's input hashes are re-registered (the edit is a legal re-audit,
    # not an attack) and the whole binding chain re-pinned
    prereg = json.loads(world["prereg_path"].read_text())
    prereg["inputs"][long_demo] = {
        "npz": _sha(world["npz_dir"] / f"{long_demo}_original.npz"),
        "pt": _sha(world["pt_dir"] / f"{long_demo}.pt")}
    world["prereg_path"].write_text(json.dumps(prereg))
    manifest = json.loads(world["manifest_path"].read_text())
    manifest["preregistration"]["sha256"] = _sha(world["prereg_path"])
    world["manifest_path"].write_text(json.dumps(manifest))
    _resync_cache_identity(world)
    manifest_sha = _sha(world["manifest_path"])
    import ireset.data.a2_integrity as _integ

    ckpt = {"identity": {"protocol": w1.MOTION_PROTOCOL_VERSION,
                         "model_type": "stub",
                         "training_manifest": {"sha256": manifest_sha},
                         "cache": {
                             "identity_sha256": _integ.sha256_file(
                                 world["cache_dir"] / "cache_identity.json"),
                             "root_stats_sha256": _integ.sha256_file(
                                 world["cache_dir"] / "root_stats.npz")}},
            "model": {}, "conditioner": {}, "config": {"d_model": 32},
            "root_stats": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
    (tmp_path / "models/stub").mkdir(parents=True)
    torch.save(ckpt, tmp_path / "models/stub/best.pt")
    monkeypatch.setattr(eg, "load_generator", lambda *a, **k: _StubGen())

    report_path = world["run"] / "eval/report.json"
    argv = ["prog", "--models", "stub", "--models-dir", str(tmp_path / "models"),
            "--training-manifest", str(world["manifest_path"]),
            "--expected-set", "val31",
            "--full-length", "--cache-dir", str(world["cache_dir"]),
            "--output", str(report_path)]
    monkeypatch.setattr("sys.argv", argv)
    eg.main()
    report = json.loads(report_path.read_text())
    assert report["repair_pipeline_invocations"] == 0
    assert report["counts"]["n_expected"] == len(world["val31"])
    assert report["counts"]["n_unsupported"] == 1
    assert report["counts"]["n_ok"] >= 1
    statuses = [r["status"] for r in report["rows"]]
    assert "unsupported_short_trajectory" in statuses and "ok" in statuses


def test_w1_raw_eval_refuses_foreign_checkpoint(tmp_path, monkeypatch):
    import eval_generation as eg

    world = build_synthetic_world(tmp_path, T=90)
    _patch_paths(monkeypatch, world)
    ckpt = {"identity": {"protocol": w1.MOTION_PROTOCOL_VERSION,
                         "model_type": "stub",
                         "training_manifest": {"sha256": "0" * 64}},
            "model": {}, "conditioner": {}, "config": {"d_model": 32},
            "root_stats": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
    (tmp_path / "models/stub").mkdir(parents=True)
    torch.save(ckpt, tmp_path / "models/stub/best.pt")
    argv = ["prog", "--models", "stub", "--models-dir", str(tmp_path / "models"),
            "--training-manifest", str(world["manifest_path"]),
            "--expected-set", "val31",
            "--full-length", "--cache-dir", str(world["cache_dir"]),
            "--output", str(world["run"] / "eval/r.json")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="checkpoint identity"):
        eg.main()


# --- S0 ------------------------------------------------------------------------------

def test_probe_error_is_unweighted_state_sum():
    from w1_s0 import probe_error

    tgt = torch.zeros(2, 4, 38)
    pred = torch.zeros(2, 4, 38)
    pred[..., 0] = 1.0
    pred[..., 3] = 2.0
    pred[..., 9] = 0.5
    err = probe_error(pred, tgt)
    # means over each state block: pos (3 dims), rot6d (6), joints (29)
    assert abs(err["pos"] - 1.0 / 3) < 1e-6
    assert abs(err["rot6d"] - 4.0 / 6) < 1e-6
    assert abs(err["joint"] - 0.25 / 29) < 1e-8
    assert abs(err["motion_reconstruction"]
               - (1.0 / 3 + 4.0 / 6 + 0.25 / 29)) < 1e-6


def _s0_cfg(model_type, steps):
    from ireset.train.losses import LossWeights
    from ireset.train.trainer import TrainConfig

    return TrainConfig(model_type=model_type, steps=steps,
                       micro_batch=2, grad_accum=2, d_model=32, n_layers=1,
                       n_heads=2, time_window=3, ffn_mult=2, device="cpu",
                       num_workers=0, seed=5, val_seed=7,
                       loss=LossWeights(w_contact=0.0))


def _s0_batch(ds):
    return {k: ds.__dict__[k].clone() for k in _FakeDS.KEYS}


def test_s0_runs_real_steps_trend_and_reload(tmp_path, monkeypatch):
    import w1_s0

    monkeypatch.setattr(w1, "S0_PROBE_EVERY", 2)
    monkeypatch.setattr(w1, "S0_TREND_WINDOW", 3)
    monkeypatch.setattr(w1, "S0_PROBE_MIN", 6)
    tok = _tokenizer_for_losses()
    ds = _FakeDS(6, F=8, seed=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True,
                                         generator=torch.Generator().manual_seed(1))
    out = tmp_path / "s0det"
    out.mkdir()
    summary = w1_s0.run_s0_model(
        "deterministic", _s0_cfg("deterministic", 40), tok, _s0_batch(ds),
        loader, "cpu", {"experiment_id": "x"}, out, time.perf_counter() + 600)
    assert summary["gates"]["min_100_optimizer_steps"] is False   # only 40 run
    assert summary["optimizer_steps"] == 40
    assert summary["micro_batches"] == 40 * 2                     # accum counted
    assert summary["gates"]["contact_branch_silent"] is True
    assert summary["gates"]["reload_reproducible"] is True
    errs = [p["motion_reconstruction"] for p in summary["probes"]]
    assert len(errs) >= 6
    assert errs[-1] < errs[0]
    assert (out / "last.pt").is_file()


def test_s0_budget_stop_is_incomplete(tmp_path, monkeypatch):
    import w1_s0

    monkeypatch.setattr(w1, "S0_PROBE_EVERY", 5)
    tok = _tokenizer_for_losses()
    ds = _FakeDS(6, F=8, seed=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    out = tmp_path / "s0det2"
    out.mkdir()
    summary = w1_s0.run_s0_model(
        "deterministic", _s0_cfg("deterministic", 10), tok, _s0_batch(ds),
        loader, "cpu", {"experiment_id": "x"}, out,
        time.perf_counter() - 1)               # deadline already passed
    assert summary["budget_stopped"] is True
    assert summary["verdict"] == "incomplete"
    assert summary["optimizer_steps"] == 0


def test_s0_micro_batch_gate(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path, n_train=1, n_val=1, T=90)
    _patch_paths(monkeypatch, world)
    import w1_s0

    monkeypatch.setattr(w1_s0, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(w1_s0, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(w1_s0, "OMOMO_NEW_DIR", world["pt_dir"])
    import train_generation
    big_batch_cfg = _s0_cfg("deterministic", 5)
    big_batch_cfg.micro_batch = 4          # 2 windows < 4 -> gate must stop
    monkeypatch.setattr(train_generation, "load_train_config",
                        lambda p: (big_batch_cfg, {}))
    argv = ["prog", "--preregistration", str(world["prereg_path"]),
            "--training-manifest", str(world["manifest_path"]),
            "--cache-dir", str(world["cache_dir"]), "--out", str(world["run"]),
            "--model", "deterministic", "--steps", "5"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="micro_batch"):
        w1_s0.main()


# --- W0 entry evidence + exclusive registration --------------------------------------

def _install_w0_anchors(world, monkeypatch):
    """Patch the anchor table onto the synthetic root's already-installed
    stand-ins (build_synthetic_world writes them at the REAL anchor paths,
    so the required-nine set re-verifies); relaxes the plan train24 sha to
    the synthetic selection."""
    root = world["root"]
    sources = {key: _sha(root / rel)
               for key, (rel, _) in w1.W1_PLAN_ANCHORS.items()}
    monkeypatch.setattr(w1, "W1_PLAN_ANCHORS", {
        key: (rel, sources[key])
        for key, (rel, _) in w1.W1_PLAN_ANCHORS.items()})
    monkeypatch.setattr(w1, "W1_PLAN_TRAIN24_SHA",
                        w1.canonical_list_sha256(world["train24"]))
    return sources


def test_w0_entry_evidence_checks(tmp_path, monkeypatch):
    import w1_preregister as w0

    world = build_synthetic_world(tmp_path)
    sources = _install_w0_anchors(world, monkeypatch)
    root = world["root"]
    good_c2 = w0.verify_c2_decision(root / w1.W1_PLAN_ANCHORS["c2_final_decision"][0])
    assert good_c2["passed"]

    tampered = json.loads((root / w1.W1_PLAN_ANCHORS["c2_final_decision"][0]).read_text())
    tampered["recovery"]["status"] = "local_preferred"
    p = root / "c2b.json"
    p.write_text(json.dumps(tampered))
    assert not w0.verify_c2_decision(p)["passed"]

    monkeypatch.setattr(
        "ireset.eval.teacher_comparison.reverify_verdict_record",
        lambda report, u_side="U0": {"problems": []})
    ur_path = root / w1.W1_PLAN_ANCHORS["ur_full_comparison"][0]
    assert w0.verify_ur_report(
        ur_path, root / w1.W1_PLAN_ANCHORS["ur_preregistration"][0])["passed"]

    bad = json.loads(ur_path.read_text())
    bad["verdict"] = "local_preferred"
    p = root / "urb.json"
    p.write_text(json.dumps(bad))
    r = w0.verify_ur_report(p, root / w1.W1_PLAN_ANCHORS["ur_preregistration"][0])
    assert any("verdict_not_no_quality_gain" in x for x in r["problems"])

    bad = json.loads(ur_path.read_text())
    bad["admission"]["passed"] = False
    p.write_text(json.dumps(bad))
    r = w0.verify_ur_report(p, root / w1.W1_PLAN_ANCHORS["ur_preregistration"][0])
    assert "admission_not_passed" in r["problems"]

    bad = json.loads(ur_path.read_text())
    bad["admission"]["ur_coverage"]["missing"] = ["subX_largebox_001"]
    p.write_text(json.dumps(bad))
    r = w0.verify_ur_report(p, root / w1.W1_PLAN_ANCHORS["ur_preregistration"][0])
    assert any("missing_demos" in x for x in r["problems"])


def test_w0_registration_exclusive_reuse_and_refusal(tmp_path, monkeypatch):
    import w1_preregister as w0

    world = build_synthetic_world(tmp_path, n_train=6, n_val=2)
    _patch_paths(monkeypatch, world)
    _install_w0_anchors(world, monkeypatch)
    monkeypatch.setattr(w0, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(w0, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(w0, "OMOMO_NEW_DIR", world["pt_dir"])
    monkeypatch.setattr(w0, "measure_repo_identity",
                        lambda: {"git_rev": "0" * 40, "dirty": False,
                                 "status_lines": [], "untracked_files": {},
                                 "tracked_diff_sha256": "0" * 64,
                                 "tracked_diff_bytes": 0})
    monkeypatch.setattr(
        "ireset.eval.teacher_comparison.reverify_verdict_record",
        lambda report, u_side="U0": {"problems": []})
    monkeypatch.setattr(w0.integ, "verify_assets_against_freeze",
                        lambda assets, root=None: {"passed": True})
    out_root = world["root"] / "derived/experiments/omniretarget_first"
    argv = ["prog", "--out-root", str(out_root),
            "--plan", str(world["root"] / "plan.md"),
            "--run-id", "w1run"]
    monkeypatch.setattr("sys.argv", argv)
    assert w0.main() == 0
    prereg_path = out_root / "w1run/freeze/preregistration.json"
    assert prereg_path.is_file()

    first = _sha(prereg_path)
    monkeypatch.setattr("sys.argv", argv)
    assert w0.main() == 0                       # clean re-verification -> reuse
    assert _sha(prereg_path) == first           # never rewritten

    victim = world["npz_dir"] / f"{world['train24'][0]}_original.npz"
    victim.write_bytes(victim.read_bytes() + b"x")
    monkeypatch.setattr("sys.argv", argv)
    assert w0.main() == 2                       # drifted input -> refused
    assert _sha(prereg_path) == first


def _complete_metrics(row: dict) -> dict:
    """A quality metrics dict with REAL measured fields (P1-5: the
    completeness predicate checks measurements, not the status string)."""
    return {
        "demo": row["demo"], "T": row["T"], "fps": 30.0, "valid": True,
        "error": None,
        "penetration_max_m": 0.012, "penetration_violation_ratio": 0.0,
        "task_mpjpe_p95_m": 0.08, "box_ground_penetration_max_m": 0.0,
        "foot": {"l": None, "r": None},       # legal zero-reference sides
        "hand": {"l": None, "r": None},
        "both": {"denominator_frames": 0, "valid_coverage": None,
                 "gap_series_p95_m": None},
        "validator_status": "pass", "validator_worst": {},
        "validator_failed_checks": [], "reference_intervals": [],
        "ref_support_frames": {"l": 0, "r": 0},
        "ref_grasp_frames": {"l": 0, "r": 0},
    }


def _run_real_s0_attempt(world, model, steps=24, deadline_s=600.0):
    """Run a REAL tiny S0 attempt into a fresh ``s0/<model>/attempt_NNN``
    and index it — internally consistent evidence (trained checkpoint,
    probe-replay artifact, contiguous loss history, on-cadence probes) that
    satisfies the readiness deep verification (round-2 review fixtures)."""
    import time as _time

    import ireset.data.a2_integrity as _integ
    import w1_s0

    tok = _tokenizer(world)
    manifest = json.loads(world["manifest_path"].read_text())
    prereg = json.loads(world["prereg_path"].read_text())
    pilot6 = list(prereg["selection"]["pilot6_from_a1"])
    ds = windows_mod.TeacherWindowDataset(
        "train", tok, window=w1.WINDOW_F, stride=w1.WINDOW_STRIDE,
        training_manifest=world["manifest_path"],
        schedule_cache_dir=world["cache_dir"] / "schedules")
    row_by_demo = {r["demo"]: r for r in manifest["rows"]}
    idxs, demos = [], []
    for demo in pilot6:
        row = row_by_demo.get(demo)
        if row is None or not row.get("admission", {}).get("motion"):
            continue
        lc_idx = next((i for i, lc in enumerate(ds.clips) if lc.demo == demo), None)
        if lc_idx is None:
            continue
        idxs.append(ds.index.index((lc_idx, 0)))
        demos.append(demo)
    items = [ds[i] for i in idxs]
    small_batch = {k: torch.stack([it[k] for it in items])
                   for k in items[0] if torch.is_tensor(items[0][k])}
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idxs), batch_size=max(1, len(idxs)),
        shuffle=True, generator=torch.Generator().manual_seed(3))
    identity = {
        "experiment_id": w1.EXPERIMENT_ID,
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "training_manifest": {"sha256": _sha(world["manifest_path"])},
        "cache": {"identity_sha256": _sha(world["cache_dir"] / "cache_identity.json"),
                  "root_stats_sha256": _sha(world["cache_dir"] / "root_stats.npz")},
        "small_set_demos": demos, "model_type": model,
    }
    s0_root = world["run"] / "s0" / model
    att = _integ.allocate_attempt(s0_root)
    cfg = _s0_cfg(model, steps)
    summary = w1_s0.run_s0_model(
        model, cfg, tok, small_batch, loader, "cpu", identity, att,
        _time.perf_counter() + deadline_s)
    _append_index(s0_root, {
        "attempt": att.name, "model": model, "verdict": summary["verdict"],
        "summary": str((att / "s0_summary.json").relative_to(world["root"]))})
    return att, summary


# --- readiness ------------------------------------------------------------------------

def test_readiness_reasons_on_missing_and_complete(tmp_path, monkeypatch):
    import w1_readiness as rd

    world = build_synthetic_world(tmp_path, n_train=6, n_val=2)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(rd, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(rd, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(rd, "OMOMO_NEW_DIR", world["pt_dir"])
    monkeypatch.setattr(
        "ireset.eval.teacher_comparison.reverify_verdict_record",
        lambda report, u_side="U0": {"problems": []})
    _install_w0_anchors(world, monkeypatch)

    def with_quality(manifest):
        for r in manifest["rows"]:
            r["quality"] = {"status": "complete", "reason": None,
                            "metrics": _complete_metrics(r)}
            r["short_trajectory"] = False

    _repin(world, with_quality)
    monkeypatch.setattr("sys.argv", ["prog", "--run-root", str(world["run"])])
    assert rd.main() == 2
    decisions = world["run"] / "decisions"
    report = json.loads(
        sorted(decisions.glob("attempt_*/w1_readiness.json"))[-1].read_text())
    assert report["ready_for_w1_d"] is False
    assert any("s0_deterministic_no_attempts" in x for x in report["unmet"])

    import ireset.data.a2_integrity as _integ

    # small S0 constants so a REAL tiny attempt satisfies the recomputed
    # gates (the deep verification rebuilds models and replays probes —
    # fabricated summaries can no longer stand in for training evidence)
    monkeypatch.setattr(w1, "S0_MIN_STEPS", 20)
    monkeypatch.setattr(w1, "S0_PROBE_EVERY", 2)
    monkeypatch.setattr(w1, "S0_TREND_WINDOW", 3)
    monkeypatch.setattr(w1, "S0_PROBE_MIN", 6)
    for model in ("deterministic", "diffusion"):
        att, summary = _run_real_s0_attempt(world, model, steps=24)
        assert summary["verdict"] == "passed", summary["gates"]

    monkeypatch.setattr("sys.argv", ["prog", "--run-root", str(world["run"])])
    rc = rd.main()
    report = json.loads(
        sorted(decisions.glob("attempt_*/w1_readiness.json"))[-1].read_text())
    assert report["ready_for_w1_d"] is True, report["unmet"]
    assert rc == 0

    # review fix 2 regression: a doctored verdict with broken facts must
    # fail the recomputed c gate
    victim_dir = sorted((world["run"] / "s0" / "deterministic").glob("attempt_*"))[-1]
    victim = victim_dir / "s0_summary.json"
    doctored = json.loads(victim.read_text())
    doctored["optimizer_steps"] = 0
    victim.write_text(json.dumps(doctored))
    monkeypatch.setattr("sys.argv", ["prog", "--run-root", str(world["run"])])
    assert rd.main() == 2
    report = json.loads(
        sorted(decisions.glob("attempt_*/w1_readiness.json"))[-1].read_text())
    assert report["c_complete"] is False
    assert any("min_100_optimizer_steps" in json.dumps(report["c_per_model"])
               or "recorded_verdict_not_reproducible" in x
               for x in report["unmet"])


def test_a3_exit_verdict_ignores_evidence_lists():
    """Regression (found on the real A3 run): ``all(gates.values())`` treated
    the empty ``missing_rows`` evidence list as a falsy gate — a full-pass
    audit exited FAIL. Only the boolean gates decide."""
    from audit_omniretarget_training import exit_criteria, exit_ok, summarize

    def _rows(statuses):
        rows = []
        for i, st in enumerate(statuses):
            row = {"demo": f"d{i}", "split": "train", "status": st,
                   "subject": SUBJECTS[i % 6],
                   "admission": {"motion": st == "processed"},
                   "quality": {"status": "complete", "reason": None,
                               "metrics": None},
                   "T": 90,
                   "full_windows_F60_s30": 1}
            row["quality"]["metrics"] = _complete_metrics(row)
            rows.append(row)
        return rows

    rows = _rows(["processed"] * 55)
    summary = summarize(rows, [f"d{i}" for i in range(55)])
    gates = exit_criteria(summary, rows)
    assert gates["missing_rows"] == [] and exit_ok(gates) is True

    rows[3]["status"] = "not_processed"
    summary = summarize(rows, [f"d{i}" for i in range(55)])
    gates = exit_criteria(summary, rows)
    assert gates["missing_rows"] == ["d3"]
    assert gates["all_candidates_processed"] is False   # reflected HERE
    assert exit_ok(gates) is False
    assert exit_ok({**gates, "all_candidates_processed": True},
                   n_budget_stop=1) is False
