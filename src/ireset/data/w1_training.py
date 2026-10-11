"""W1 weak-motion-supervision protocol (omniretarget_w1_a3bc_plan.md).

Constants and PURE contracts shared by the W0 run registration, the A3
published-label audit and the B-stage consumers (dataset / cache / train /
eval):

* identity: experiment ``w1_published_motion_v1``, protocol
  ``omniretarget_motion_v1``, purpose ``weak_motion_only``, source
  ``published_original`` at the pinned published revision;
* list selection: ``teacher_production_train24_v1`` — per subject, the
  first FOUR demos under ``SHA256("20261008|teacher_pilot|" + demo)`` over
  the locked train250 originals (canonical order: subject ascending, hash
  rank within subject); the canonical demo-array SHA-256 is part of the
  contract and is re-derived, never trusted from a file;
* admission (plan §3.2): structural/numeric/layout/joint-limit checks over
  the PUBLISHED ``*_original.npz`` labels — a file-level verdict separate
  from the quality registration (strict quality failures never cancel the
  weak-motion admission);
* preregistration / training-manifest verification: every consumer
  re-verifies anchors, lists, input hashes and assets against disk before
  consuming anything (a list file being unchanged does not mean the inputs
  and assets are unchanged).

Nothing here touches the disk at import time; the entries measure every
fact they record.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ireset.data import a2_integrity as integ
from ireset.data.pairing import (
    GATE_QUAT_NORM_ERR,
    EXPECTED_FPS,
    fit_similarity,
    rank_by_hash,
    subject_of,
)

# --- identity (plan §2.2) -----------------------------------------------------

EXPERIMENT_ID = "w1_published_motion_v1"
MOTION_PROTOCOL_VERSION = "omniretarget_motion_v1"
TRAINING_PURPOSE = "weak_motion_only"
SOURCE_TYPE = "published_original"
PUBLISHED_REVISION = "135259130158f1cb1799e4ab9b6566c938285fc9"
MANIFEST_NAME = "omniretarget_training_manifest_v1"
TRAIN24_LIST_ID = "teacher_production_train24_v1"
VAL31_LIST_ID = "val_originals_v1"
VAL6_LIST_ID = "val6_pilot_v1"
#: fields the published dataset does not carry for its own production env
NOT_PUBLISHED = "not_published"

TRAIN24_HASH_PREFIX = "20261008|teacher_pilot|"
TRAIN24_PER_SUBJECT = 4
WINDOW_F = 60
WINDOW_STRIDE = 30
SHORT_TRAJ_FRAMES = 60

#: plan §2.1 anchor table — the bytes this round is allowed to reason over.
#: Paths are repo-relative; the entries re-hash against the declared SHA-256.
W1_PLAN_ANCHORS = {
    "c2_final_decision": (
        "derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/"
        "decisions/attempt_010/a2_closure.json",
        "34a392f92a9cda5b8317a08d5b5b486f750c5fc83933174d36d361cb7b1d0fa2",
    ),
    "ur_full_comparison": (
        "derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/"
        "recovery/cmp/attempt_002/teacher_selection.json",
        "faa2df0cb9c1df37b325f0f73a7d2989c4ffc425270439679b4530d32724b00e",
    ),
    "ur_preregistration": (
        "derived/experiments/omniretarget_first/20261010_a39d763_a2_closure/"
        "recovery/preregistration.json",
        "f4676813b20c2784385237fc2c101636e9c77aaf280e2376f029b8252f9c42d9",
    ),
    "a1_frozen_lists": (
        "derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze/"
        "frozen_lists.json",
        "395094a30d69158ad08bf5b576f07c541c052187d5bf242f1cc7fde05e353d9d",
    ),
    "a1_assets": (
        "derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze/"
        "assets.json",
        "2f72c9ed0f1832b0c1d80e7d1f2e82382785c56ae83556a63a46ec6789c7ad79",
    ),
    "a1_pairing_source_boxes": (
        "derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze/"
        "pairing_source_boxes.npz",
        "39fce035ab63569bd69a4ddf61e1e6cd4fb70d4d1b2a12abf7051bd55ce3d09d",
    ),
    "split_train": (
        "derived/data/split/train.json",
        "93ff2ba4dd956e1ee8d12177cb48f5a0514343eb126b864696d94578aa196f1c",
    ),
    "split_val": (
        "derived/data/split/val.json",
        "7387d68cc56e5f87059ce4be2a56386510a49d96732fa77450ee53bb4433f15e",
    ),
    "split_manifest": (
        "derived/data/split/manifest.json",
        "fcfd13bf108766c05e92aa45fc751ffbf620bbc86de630a4a9e17faa1e052090",
    ),
}

#: the plan document itself (hashed into the registration)
W1_PLAN_DOC = "docs/plans/omniretarget_w1_a3bc_plan.md"
#: the canonical train24 demo-array SHA-256 declared by the plan (§3.1)
W1_PLAN_TRAIN24_SHA = (
    "8fbfd008b98f56196d31f81352582006ba95c16bdcf2a4da2f5da56eb8267948"
)

#: budgets (plan §9): W0+A3 and the B cache are 2h CPU wall each, S0 is 1h
#: GPU wall total with at most 1,000 optimizer steps per model
W0A3_BUDGET_S = 2 * 3600
CACHE_BUDGET_S = 2 * 3600
S0_BUDGET_S = 3600
S0_MAX_STEPS = 1000
S0_MIN_STEPS = 100
S0_PROBE_EVERY = 10
S0_PROBE_MIN = 40
S0_TREND_WINDOW = 20


# --- list selection (plan §3.1) -----------------------------------------------

def select_train24(train_demos: list[str]) -> list[str]:
    """Per subject the first four demos by SHA256(prefix + demo) rank.

    Canonical order: subject name ascending, hash rank within subject (ties
    fall back to demo-name order via ``sorted`` stability). Selection reads
    identity only — never quality or model results.
    """
    by_subj: dict[str, list[str]] = {}
    for d in rank_by_hash(sorted(train_demos), TRAIN24_HASH_PREFIX):
        by_subj.setdefault(subject_of(d), []).append(d)
    out: list[str] = []
    for subj in sorted(by_subj):
        out.extend(by_subj[subj][:TRAIN24_PER_SUBJECT])
    return out


def canonical_list_sha256(demos: list[str]) -> str:
    """SHA-256 of the canonical JSON demo array (§3.1: UTF-8, compact
    separators, no trailing newline) — verifies the ARRAY, never a
    substitute for the per-file input hashes."""
    blob = json.dumps(list(demos), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- published-label admission (plan §3.2, pure over arrays) -------------------

def published_to_project(qpos_raw: np.ndarray) -> np.ndarray:
    """Published [T,43] (quat-first freejoints) -> project scene layout.

    Same permutation as ``teacher.convert_teacher_qpos`` for the whole array;
    kept here so admission checks do not depend on the loader module.
    """
    q = np.asarray(qpos_raw, dtype=np.float64)
    out = np.empty_like(q)
    out[:, 0:3], out[:, 3:7] = q[:, 4:7], q[:, 0:4]
    out[:, 7:36] = q[:, 7:36]
    out[:, 36:39], out[:, 39:43] = q[:, 40:43], q[:, 36:40]
    return out


def published_roundtrip_ok(qpos_raw: np.ndarray) -> bool:
    """project<->published layout conversion must be an exact inverse.

    With normalization registered (norm err <= 1e-4) the quaternion columns
    may flip sign globally — |dot| == 1 counts as equal (§3.2 item 3).
    """
    scene = published_to_project(qpos_raw)
    back = np.empty_like(scene)
    back[:, 0:4], back[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
    back[:, 7:36] = scene[:, 7:36]
    back[:, 36:40], back[:, 40:43] = scene[:, 39:43], scene[:, 36:39]
    # published layout: root/box [qw,qx,qy,qz,x,y,z] — positions are cols
    # 4:7 and 40:43, joints 7:36; the quaternion blocks are compared below
    # up to global sign (registered normalization)
    pos_joint_cols = [4, 5, 6, *range(7, 36), 40, 41, 42]
    if not np.array_equal(back[:, pos_joint_cols], qpos_raw[:, pos_joint_cols]):
        return False
    for a, b in ((back[:, 0:4], qpos_raw[:, 0:4]), (back[:, 36:40], qpos_raw[:, 36:40])):
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        dot = np.abs((a / na * b / nb).sum(1))
        if not np.all(dot >= 1.0 - 1e-9):
            return False
    return True


def check_published_admission(
    demo: str,
    qpos_raw: np.ndarray,
    human: np.ndarray,
    fps: float,
    pt_T: int,
    joint_range: np.ndarray,
) -> dict:
    """Structural admission verdict over the published arrays (§3.2 items
    1-4). Returns ``{"motion": bool, "reasons": [...], "facts": {...}}``.

    Numeric/unit/identity/time damage fails admission; a failure reason is
    always explicit (a bool, NaN or missing value can never be laundered
    into a legal count or zero error). Legitimate quaternion normalization
    (norm error <= 1e-4) is REGISTERED, never written back.
    """
    reasons: list[str] = []
    q = np.asarray(qpos_raw)
    h = np.asarray(human)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) \
            or not math.isfinite(float(fps)):
        reasons.append("fps_missing_not_finite")
        fps = float("nan")
    elif abs(float(fps) - EXPECTED_FPS) > 1e-12:
        reasons.append(f"fps_not_30:{fps}")
    if q.ndim != 2 or q.shape[1] != 43:
        reasons.append(f"qpos_shape:{np.shape(q)}")
        q = np.zeros((0, 43))
    if h.ndim != 3 or h.shape[1:] != (52, 3):
        reasons.append(f"human_shape:{np.shape(h)}")
        h = np.zeros((q.shape[0], 52, 3))
    if q.shape[0] != h.shape[0]:
        reasons.append(f"T_disagree_qpos_human:{q.shape[0]}:{h.shape[0]}")
    if int(pt_T) != int(q.shape[0]):
        reasons.append(f"T_mismatch_vs_pt:{q.shape[0]}:{pt_T}")
    if not (np.isfinite(q).all() if q.size else True):
        reasons.append("qpos_not_finite")
    if not (np.isfinite(h).all() if h.size else True):
        reasons.append("human_not_finite")
    root_norm_err = box_norm_err = float("inf")
    quat_normalized = False
    if q.size:
        root_norm_err = float(np.abs(np.linalg.norm(q[:, 0:4], axis=1) - 1).max())
        box_norm_err = float(np.abs(np.linalg.norm(q[:, 36:40], axis=1) - 1).max())
        if max(root_norm_err, box_norm_err) > GATE_QUAT_NORM_ERR:
            reasons.append(
                f"quat_norm_out_of_range:root={root_norm_err:.3e}:box={box_norm_err:.3e}")
        else:
            # registered = a deliberate deviation beyond float representation
            quat_normalized = max(root_norm_err, box_norm_err) > 1e-12
        if not published_roundtrip_ok(q):
            reasons.append("layout_roundtrip_failed")
        lo, hi = joint_range[:, 0], joint_range[:, 1]
        joints = q[:, 7:36]
        viol = float(np.maximum(lo - joints, joints - hi).max()) if joints.size else 0.0
        if viol > 0.0:
            reasons.append(f"joint_limit_violation_rad:{viol:.6f}")
    else:
        viol = None
    return {
        "motion": not reasons,
        "reasons": reasons,
        "facts": {
            "T": int(q.shape[0]) if q.ndim == 2 else None,
            "fps": float(fps),
            "root_quat_norm_err_max": root_norm_err if q.size else None,
            "box_quat_norm_err_max": box_norm_err if q.size else None,
            "quat_normalization_registered": quat_normalized,
            "joint_limit_max_violation_rad": viol,
        },
    }


# --- scene split (plan §3.3, pure) ---------------------------------------------

def scene_split_facts(
    scene_project: np.ndarray, source_box: np.ndarray, target_box: np.ndarray,
    atol: float = 1e-6,
) -> dict:
    """Source/target scene separation facts for one demo.

    The reference human corresponds to the SOURCE box (the similarity-
    transformed PT object); the model's task condition is the TARGET box.
    Originals are known to carry a per-demo constant, z-dominant offset —
    registered here, never assumed away by assigning both fields the qpos
    box.
    """
    offset = np.asarray(target_box)[:, 0:3] - np.asarray(source_box)[:, 0:3]
    mean = offset.mean(0)
    resid = float(np.abs(offset - mean).max())
    return {
        "offset_mean_m": [float(x) for x in mean],
        "offset_norm_mean_m": float(np.linalg.norm(mean)),
        "offset_residual_max_m": resid,
        "offset_pure_z": bool(abs(mean[0]) <= atol and abs(mean[1]) <= atol),
        "same_trajectory": bool(
            resid == 0.0 and np.abs(offset).max() == 0.0),
    }


def derive_source_box(
    pt_human: np.ndarray, npz_human: np.ndarray, pt_obj: np.ndarray,
) -> np.ndarray:
    """Source-scene box in the target scene frame (§3.3, A1 mapping).

    Same estimator as A1's freeze so the recomputation is bit-comparable
    with the frozen ``pairing_source_boxes.npz`` entries.
    """
    from ireset.data.pairing import source_obj_pose_in_target_scene

    scale, R, t, _ = fit_similarity(
        np.asarray(pt_human).reshape(-1, 3), np.asarray(npz_human).reshape(-1, 3))
    return source_obj_pose_in_target_scene(pt_obj, scale, R, t)


# --- keyed schedule cache (plan §4.2) ------------------------------------------

#: bump when the schedule-inference/condition-construction semantics change
SCHEDULE_KEY_VERSION = 1
#: code that defines condition construction + schedule inference — the cache
#: key covers CODE identity, not just file names
CONDITION_CODE_FILES = (
    "src/ireset/data/teacher.py",
    "src/ireset/data/pairing.py",
    "src/ireset/embodied/contact_sites.py",
    "src/ireset/data/w1_training.py",
)


def condition_code_sha256(project_root: Path,
                          hash_file=integ.sha256_file) -> str:
    """Concatenated hash of the condition/schedule code files."""
    h = hashlib.sha256()
    for rel in CONDITION_CODE_FILES:
        h.update(hash_file(Path(project_root) / rel).encode())
    return h.hexdigest()


def schedule_cache_key(npz_sha256: str, pt_sha256: str, assets_sha256: str,
                       code_sha256: str) -> str:
    """Identity key for one demo's W1 reference-schedule cache file.

    Covers the actual input hashes, the scene-split mapping version, the
    source geometry, the asset closure, the protocol, the inference version
    and the condition CODE identity — the same file NAME never proves the
    same cache is reusable (§4.2).
    """
    blob = json.dumps({
        "v": SCHEDULE_KEY_VERSION,
        "npz_sha256": npz_sha256,
        "pt_sha256": pt_sha256,
        "scene_split": "a1_similarity_v1",
        "source_geometry": "largebox_locked",
        "assets_sha256": assets_sha256,
        "protocol": MOTION_PROTOCOL_VERSION,
        "schedule_inference": "infer_reference_schedule_v1",
        "condition_code_sha256": code_sha256,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def schedule_cache_filename(demo: str, key: str) -> str:
    return f"{demo}__{key[:16]}.npz"


# --- preregistration / manifest verification -----------------------------------

def anchor_paths(project_root: Path) -> dict[str, Path]:
    return {k: project_root / rel for k, (rel, _) in W1_PLAN_ANCHORS.items()}


def verify_preregistration(
    prereg: dict, project_root: Path, npz_dir: Path, pt_dir: Path,
    hash_file=integ.sha256_file, verify_assets: bool = True,
) -> dict:
    """Re-verify every registration leg against current disk state.

    Used by the A3 audit, the cache precompute, the W1 dataset entry and the
    S0 entry as the preflight: anchors (re-hashed, not existence-checked),
    the plan file, train24 recomputation from the locked split, val31/val6
    vs the A1 frozen lists, per-demo input hashes and the A1 asset closure.
    Any problem is a reason to refuse consuming the registration — an
    unchanged list file does not prove unchanged inputs and assets.
    ``verify_assets=False`` skips only the (comparatively expensive) asset
    re-hash leg; every identity/list/input leg always runs.
    """
    problems: list[str] = []
    root = Path(project_root)

    ident = prereg.get("identity", {})
    if (ident.get("experiment_id") != EXPERIMENT_ID
            or ident.get("protocol") != MOTION_PROTOCOL_VERSION
            or ident.get("training_purpose") != TRAINING_PURPOSE
            or ident.get("source_type") != SOURCE_TYPE
            or ident.get("published_revision") != PUBLISHED_REVISION):
        problems.append("identity_mismatch")

    plan_rec = prereg.get("plan") or {}
    plan_path = root / str(plan_rec.get("path", ""))
    if not plan_path.is_file() or hash_file(plan_path) != plan_rec.get("sha256"):
        problems.append("plan_drifted")

    for key, rec in (prereg.get("anchors") or {}).items():
        p = root / str(rec.get("path", ""))
        if not p.is_file():
            problems.append(f"anchor_missing:{key}")
        elif hash_file(p) != rec.get("sha256"):
            problems.append(f"anchor_drifted:{key}")

    sel = prereg.get("selection", {})
    train_demos = json.loads((root / "derived/data/split/train.json").read_text())["demos"]
    recomputed = select_train24(train_demos)
    if list(sel.get("train24", [])) != recomputed:
        problems.append("train24_not_reproducible_from_split")
    # self-consistency of the recorded canonical hash; the W0 entry
    # additionally enforces the plan-declared constant at registration time
    if sel.get("train24_canonical_sha256") not in (None, canonical_list_sha256(recomputed)):
        problems.append("train24_canonical_sha_inconsistent")
    lists_path = root / str((prereg.get("anchors") or {}).get(
        "a1_frozen_lists", {}).get("path", ""))
    if lists_path.is_file():
        lists = json.loads(lists_path.read_text())
        val31 = [e["demo"] for e in lists[VAL31_LIST_ID]]
        val6 = [e["demo"] for e in lists[VAL6_LIST_ID]]
        if list(sel.get("val31", [])) != val31:
            problems.append("val31_mismatch_vs_a1_freeze")
        if sorted(sel.get("val6", [])) != sorted(val6):
            problems.append("val6_mismatch_vs_a1_freeze")
        overlap = sorted(set(sel.get("train24", [])) & set(val31))
        if overlap:
            problems.append(f"train24_val31_overlap:{overlap}")

    for demo, rec in (prereg.get("inputs") or {}).items():
        for key, suffix, d in (("npz", "_original.npz", npz_dir), ("pt", ".pt", pt_dir)):
            p = Path(d) / f"{demo}{suffix}"
            if not p.is_file():
                problems.append(f"input_missing:{demo}:{key}")
            elif hash_file(p) != rec.get(key):
                problems.append(f"input_drifted:{demo}:{key}")

    assets_rec = prereg.get("assets") or {}
    a1_assets_path = root / str(assets_rec.get("a1_assets_path", ""))
    if not a1_assets_path.is_file() or hash_file(a1_assets_path) != assets_rec.get("a1_assets_sha256"):
        problems.append("a1_assets_drifted")
    elif verify_assets:
        check = integ.verify_assets_against_freeze(json.loads(a1_assets_path.read_text()),
                                                   root=root)
        if not check["passed"]:
            problems.append(f"assets_drifted:{check['mismatched'] + check['missing']}")
    return {"problems": problems, "passed": not problems}


def load_w1_derived_clip(row: dict, schedule_cache_dir: Path,
                         project_root: Path, assets_sha: str,
                         code_sha: str | None = None):
    """Build the :class:`InteractionClip` for one ADMITTED manifest row.

    THE condition constructor shared by training, validation and generation
    (plan §4.1): derived representation + keyed schedule cache + explicit
    source/target scene split. Teacher root/joints never enter the
    condition — changing only the teacher robot state leaves every
    condition channel bit-identical.
    """
    import numpy as np

    from ireset.data.teacher import build_interaction_clip

    if code_sha is None:
        code_sha = condition_code_sha256(project_root)
    dpath = Path(project_root) / str(row["derived"]["path"])
    inputs = row.get("inputs") or {}
    key = schedule_cache_key(inputs.get("npz_sha256", ""),
                             inputs.get("pt_sha256", ""),
                             assets_sha, code_sha)
    cache_file = (Path(schedule_cache_dir)
                  / schedule_cache_filename(row["demo"], key))
    with np.load(dpath, allow_pickle=False) as z:
        src_box = z["source_obj_pose"].copy()
    return build_interaction_clip(
        dpath.name, schedule_cache_file=cache_file,
        teacher_traj_dir=dpath.parent, source_obj_pose=src_box)


def w1_rows_for_split(manifest: dict, split: str) -> list[dict]:
    """Rows of the requested split from a W1 training manifest (identity
    only; the consumer still re-verifies hashes)."""
    return [r for r in manifest.get("rows", []) if r.get("split") == split]


def fit_w1_root_stats(rows: list[dict], project_root: Path,
                      schedule_cache_dir: Path, assets_sha: str,
                      code_sha: str | None = None) -> tuple["RootStats", dict]:
    """Root statistics over the ORIGINAL frames of the admitted TRAIN rows
    only (plan §4.2).

    Every frame once per admitted train file; val/test rows in ``rows`` are
    filtered out HERE (a val file whose content drifts can never move the
    train statistics), rejected rows never contribute, and the fit never
    sees windowing. Returns ``(stats, facts)`` with the files list and
    frame count for the identity record.
    """
    import numpy as np

    from ireset.data.teacher import world_to_clip
    from ireset.model.tokens import RootStats

    if code_sha is None:
        code_sha = condition_code_sha256(project_root)
    train_rows = [r for r in rows
                  if r.get("split") == "train" and r.get("admission", {}).get("motion")]
    pos = []
    for r in train_rows:
        clip = load_w1_derived_clip(r, schedule_cache_dir, project_root,
                                    assets_sha, code_sha)
        pos.append(world_to_clip(clip.root[:, 0:3], clip.condition.world_from_clip))
    stats = RootStats.fit(np.concatenate(pos, axis=0))
    facts = {
        "files": [r["demo"] for r in train_rows],
        "frames_fitted": int(sum(p.shape[0] for p in pos)),
        "n_rows_seen": len(rows),
        "n_train_rows_used": len(train_rows),
    }
    return stats, facts


def full_windows(T: int, f: int = WINDOW_F, stride: int = WINDOW_STRIDE) -> int:
    """Complete-window count only — no padding, no partial windows."""
    return max(0, (T - f) // stride + 1)
