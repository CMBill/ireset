"""Stage A1 freeze & pairing for the OmniRetarget first-training route.

Plan: docs/plans/omniretarget_first_training_plan.md §8 (A1), §3.2-§3.3,
§4.3, §7.1, §11. Produces the identity baseline that every later stage
(A2 teacher comparison, A3 labels, training) is checked against:

1. run identity (git/env/argv, untracked files not hidden);
2. inventory census + split verification (250/31/31 originals, 6 subjects,
   17 reserve PTs, variants follow their parent demo, T/fps, T<60 listing);
3. frozen lists: ``teacher_pilot_train6_v1`` (SHA256("20261008|teacher_pilot|"
   + demo) first per subject), ``val_originals_v1`` (31),
   ``val6_pilot_v1`` (SHA256("20261008|" + demo) first per subject);
4. asset/source hashes (published archive+models vs resources/manifest.json,
   upstream checkout, height metadata, locked G1/largebox assets);
5. §4.3 source/target scene pairing verification for the pilot six (gated)
   and all val originals (recorded), with derived source-scene box
   trajectories saved for I1/I2.

Read-only over resources/ and third_party/; writes only under
``derived/experiments/omniretarget_first/<run_id>/freeze/``. Exit code is
non-zero when any mandatory gate fails (plan §8: fix identity/conditions
first; missing evidence is never a pass).

Run: uv run --locked python scripts/omniretarget_freeze_pairing.py \
        [--out-root derived/experiments/omniretarget_first] [--run-id ID]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ireset.data import pairing
from ireset.paths import (
    DERIVED_DIR,
    DERIVED_ROBOTS_DIR,
    LARGEBOX_DIR,
    OMNIRETARGET_DIR,
    OMOMO_NEW_DIR,
    PROJECT_ROOT,
    TEACHER_TRAJ_DIR,
)

SPLIT_DIR = DERIVED_DIR / "data" / "split"
HOLOSOMA_DIR = PROJECT_ROOT / "third_party" / "holosoma"
HEIGHT_DICT = (
    HOLOSOMA_DIR / "src" / "holosoma_retargeting" / "holosoma_retargeting"
    / "demo_data" / "height_dict.pkl"
)
RESOURCE_MANIFEST = PROJECT_ROOT / "resources" / "manifest.json"

PILOT_HASH_PREFIX = "20261008|teacher_pilot|"
VAL6_HASH_PREFIX = "20261008|"
SHORT_TRAJ_FRAMES = 60  # plan §3.1: T<60 not used for first-round windows
EXPECTED_SUBJECTS = ["sub10", "sub12", "sub16", "sub3", "sub7", "sub8"]


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path = PROJECT_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def run_identity(argv: list[str]) -> dict:
    rev = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"]).splitlines()
    untracked = {}
    for line in status:
        if line.startswith("??"):
            rel = line[3:].strip().strip('"')
            untracked[rel] = pairing.sha256_hex(PROJECT_ROOT / rel)
    try:
        hs_rev = _git(["rev-parse", "HEAD"], cwd=HOLOSOMA_DIR)
        hs_dirty = bool(_git(["status", "--porcelain"], cwd=HOLOSOMA_DIR))
    except subprocess.CalledProcessError:
        hs_rev, hs_dirty = None, None
    import mujoco
    import torch

    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_revision": rev,
        "git_short": rev[:7],
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status),
        "git_status": status,
        "untracked_sha256": untracked,  # plan §11: never hide untracked files
        "holosoma_revision": hs_rev,
        "holosoma_dirty": hs_dirty,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "argv": argv,
    }


# ---------------------------------------------------------------------------
# census & split verification
# ---------------------------------------------------------------------------

def census_originals() -> tuple[dict, dict]:
    """Scan teacher NPZs; -> (census report, {demo: (T, fps)})."""
    names = sorted(p.name for p in TEACHER_TRAJ_DIR.glob("*.npz"))
    originals: list[str] = []
    rot_files = trans_files = 0
    variants_without_parent: list[str] = []
    demo_set: set[str] = set()
    for name in names:
        stem = name[: -len(".npz")]
        if stem.endswith("_original"):
            demo = stem[: -len("_original")]
            originals.append(demo)
            demo_set.add(demo)
    for name in names:
        stem = name[: -len(".npz")]
        if stem.endswith("_original"):
            continue
        if "_rot_" in stem:
            rot_files += 1
            parent = stem.rsplit("_rot_", 1)[0]
        elif "_trans_" in stem:
            trans_files += 1
            parent = stem.rsplit("_trans_", 1)[0]
        else:
            parent = ""
        if parent not in demo_set:
            variants_without_parent.append(name)

    frames = {}
    for demo in sorted(demo_set):
        npz = np.load(TEACHER_TRAJ_DIR / f"{demo}_original.npz", allow_pickle=True)
        q = npz["qpos"]
        frames[demo] = (int(q.shape[0]), float(npz["fps"]))

    pt_stems = sorted(p.stem for p in OMOMO_NEW_DIR.glob("*_largebox_*.pt"))
    report = {
        "npz_total": len(names),
        "original_demos": len(demo_set),
        "rot_variant_files": rot_files,
        "trans_variant_files": trans_files,
        "variant_files_without_parent_demo": variants_without_parent,
        "pt_largebox_files": len(pt_stems),
        "pt_without_teacher": sorted(set(pt_stems) - demo_set),
        "teacher_without_pt": sorted(demo_set - set(pt_stems)),
        "fps_not_30": {d: f for d, (_, f) in frames.items() if f != pairing.EXPECTED_FPS},
        "short_trajectories_T_lt_60": {d: t for d, (t, _) in frames.items() if t < SHORT_TRAJ_FRAMES},
        "originals_total_frames": int(sum(t for t, _ in frames.values())),
    }
    return report, frames


def verify_split(frames: dict, census_reserve: set[str]) -> dict:
    split_json = {
        s: json.loads((SPLIT_DIR / f"{s}.json").read_text()) for s in ("train", "val", "test")
    }
    demo_sets = {s: set(j["demos"]) for s, j in split_json.items()}
    union = demo_sets["train"] | demo_sets["val"] | demo_sets["test"]
    overlap = (
        (demo_sets["train"] & demo_sets["val"])
        | (demo_sets["train"] & demo_sets["test"])
        | (demo_sets["val"] & demo_sets["test"])
    )
    on_disk = set(frames)
    manifest_reserve = set(
        json.loads((SPLIT_DIR / "manifest.json").read_text())["unsupervised_reserve_no_teacher"]
    )
    per_split = {}
    for s, demos in demo_sets.items():
        by_subj = defaultdict(list)
        for d in sorted(demos):
            by_subj[pairing.subject_of(d)].append(d)
        per_split[s] = {
            "demos": len(demos),
            "subjects": sorted(by_subj),
            "demos_per_subject": {k: len(v) for k, v in sorted(by_subj.items())},
            "frames": int(sum(frames[d][0] for d in demos if d in frames)),
            "short_trajectories": {
                d: frames[d][0] for d in sorted(demos)
                if d in frames and frames[d][0] < SHORT_TRAJ_FRAMES
            },
        }
    checks = {
        "counts_250_31_31": [len(demo_sets[s]) for s in ("train", "val", "test")] == [250, 31, 31],
        "no_overlap": not overlap,
        "overlap_demos": sorted(overlap),
        "union_equals_disk": union == on_disk,
        "missing_on_disk": sorted(union - on_disk),
        "subjects_all_splits": all(
            per_split[s]["subjects"] == EXPECTED_SUBJECTS for s in ("train", "val", "test")
        ),
        "reserve_matches_manifest": {
            "census": sorted(census_reserve),
            "manifest": sorted(manifest_reserve),
            "equal": census_reserve == manifest_reserve,
        },
    }
    passed = (
        checks["counts_250_31_31"]
        and checks["no_overlap"]
        and checks["union_equals_disk"]
        and checks["subjects_all_splits"]
        and not checks["missing_on_disk"]
        and checks["reserve_matches_manifest"]["equal"]
    )
    return {"checks": checks, "per_split": per_split, "passed": passed}


# ---------------------------------------------------------------------------
# frozen lists
# ---------------------------------------------------------------------------

def _list_entry(demo: str, frames: dict) -> dict:
    t, fps = frames[demo]
    npz_path = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
    pt_path = OMOMO_NEW_DIR / f"{demo}.pt"
    return {
        "demo": demo,
        "subject": pairing.subject_of(demo),
        "T": t,
        "duration_s": round(t / fps, 3),
        "full_windows_F60_s30": max(0, (t - SHORT_TRAJ_FRAMES) // 30 + 1),
        "npz": str(npz_path.relative_to(PROJECT_ROOT)),
        "npz_sha256": pairing.sha256_hex(npz_path),
        "pt": str(pt_path.relative_to(PROJECT_ROOT)),
        "pt_sha256": pairing.sha256_hex(pt_path),
    }


def build_frozen_lists(frames: dict) -> dict:
    train_demos = sorted(
        d for d in json.loads((SPLIT_DIR / "train.json").read_text())["demos"] if d in frames
    )
    val_demos = sorted(
        d for d in json.loads((SPLIT_DIR / "val.json").read_text())["demos"] if d in frames
    )
    pilot6 = pairing.select_first_per_subject(train_demos, PILOT_HASH_PREFIX)
    val6 = pairing.select_first_per_subject(val_demos, VAL6_HASH_PREFIX)

    def with_key(demo: str, prefix: str) -> dict:
        e = _list_entry(demo, frames)
        e["sha256_rank_key"] = hashlib.sha256(f"{prefix}{demo}".encode("utf-8")).hexdigest()
        return e

    lists = {
        "selection_rules": {
            "teacher_pilot_train6_v1": (
                f'per subject, first by SHA256("{PILOT_HASH_PREFIX}" + demo) '
                "over train originals (plan §3.3)"
            ),
            "val_originals_v1": "all val originals, full length (plan §7.1)",
            "val6_pilot_v1": (
                f'per subject, first by SHA256("{VAL6_HASH_PREFIX}" + demo) '
                "over val originals (plan §7.1)"
            ),
        },
        "teacher_pilot_train6_v1": [with_key(d, PILOT_HASH_PREFIX) for d in pilot6.values()],
        "val_originals_v1": [_list_entry(d, frames) for d in val_demos],
        "val6_pilot_v1": [with_key(d, VAL6_HASH_PREFIX) for d in val6.values()],
    }
    gates = {
        "pilot6_six_subjects": len(pilot6) == 6 and set(pilot6) == set(EXPECTED_SUBJECTS),
        "pilot6_all_full_length": all(frames[d][0] >= SHORT_TRAJ_FRAMES for d in pilot6.values()),
        "val6_six_subjects": len(val6) == 6 and set(val6) == set(EXPECTED_SUBJECTS),
        "val6_subset_of_val31": set(val6.values()) <= set(val_demos),
        "val31_complete": len(val_demos) == 31,
    }
    return {
        "lists": lists,
        "gates": gates,
        "passed": all(gates.values()),
        "pilot6_demos": sorted(pilot6.values()),
        "val31_demos": val_demos,
        "val6_demos": sorted(val6.values()),
    }


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------

def hash_tree(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(PROJECT_ROOT))] = {
                "bytes": p.stat().st_size,
                "sha256": pairing.sha256_hex(p),
            }
    return out


def audit_assets() -> dict:
    res_manifest = json.loads(RESOURCE_MANIFEST.read_text())
    archive = OMNIRETARGET_DIR / "robot-object.zip"
    archive_sha = pairing.sha256_hex(archive)
    with open(HEIGHT_DICT, "rb") as f:
        heights = {str(k): float(v) for k, v in sorted(pickle.load(f).items())}
    return {
        "published": {
            "revision": res_manifest["omniretarget"]["revision"],
            "archive": str(archive.relative_to(PROJECT_ROOT)),
            "archive_sha256": archive_sha,
            "archive_sha256_matches_manifest": (
                archive_sha == res_manifest["omniretarget"]["archive_sha256"]
            ),
            "models": hash_tree(OMNIRETARGET_DIR / "models"),
            "models_file_count_manifest": res_manifest["omniretarget"]["verified_model_files"],
        },
        "upstream": {
            "url": res_manifest["holosoma"]["url"],
            "expected_commit": res_manifest["holosoma"]["commit"],
            "height_dict": str(HEIGHT_DICT.relative_to(PROJECT_ROOT)),
            "height_dict_sha256": pairing.sha256_hex(HEIGHT_DICT),
            "subject_heights_m": heights,
        },
        "locked_g1": hash_tree(DERIVED_ROBOTS_DIR),
        "largebox": hash_tree(LARGEBOX_DIR),
    }


# ---------------------------------------------------------------------------
# §4.3 pairing verification
# ---------------------------------------------------------------------------

def _failed_record(demo: str, exc: Exception) -> dict:
    return pairing.ScenePairing(
        demo=demo, t_npz=-1, t_pt=-1, fps=float("nan"), scale=float("nan"),
        scale_expected=None, rotation_wxyz=(0.0, 0.0, 0.0, 0.0),
        translation_xyz=(0.0, 0.0, 0.0), human_residual_max_m=float("inf"),
        box_rot_err_deg_max=float("inf"), box_offset_mean_m=(0.0, 0.0, 0.0),
        box_offset_residual_max_m=float("inf"), box_offset_xy_abs_mean_m=(0.0, 0.0),
        root_quat_norm_err_max=float("inf"), box_quat_norm_err_max=float("inf"),
        human_all_finite=False, teacher_all_finite=False, pt_all_finite=False,
        joint_limit_max_violation_rad=None, gates={}, passed=False,
        error=f"{type(exc).__name__}: {exc}",
    ).to_json_dict()


def verify_pairing_set(
    demos: list[str], joint_range: np.ndarray, heights: dict
) -> tuple[list[dict], dict[str, np.ndarray]]:
    records: list[dict] = []
    source_boxes: dict[str, np.ndarray] = {}
    for demo in demos:
        try:
            q_raw, human, fps = pairing.load_teacher_npz_raw(
                TEACHER_TRAJ_DIR / f"{demo}_original.npz"
            )
            pt_human, pt_obj = pairing.load_pt(OMOMO_NEW_DIR / f"{demo}.pt")
            rec = pairing.verify_scene_pairing(
                demo, q_raw, human, pt_human, pt_obj, fps,
                joint_range=joint_range,
                subject_height_m=heights.get(pairing.subject_of(demo)),
            )
            # Same fit as inside verify_scene_pairing (deterministic); kept
            # here so the derived source box uses the identical transform.
            scale, R, t, _ = pairing.fit_similarity(
                pt_human.reshape(-1, 3), human.reshape(-1, 3)
            )
            source_boxes[demo] = pairing.source_obj_pose_in_target_scene(pt_obj, scale, R, t)
            records.append(rec.to_json_dict())
        except Exception as exc:  # missing evidence is an explicit failure row
            print(f"    ! {demo}: {type(exc).__name__}: {exc}")
            records.append(_failed_record(demo, exc))
    return records, source_boxes


def summarize_pairing(records: list[dict]) -> dict:
    ok = [r for r in records if r["error"] is None]
    return {
        "n_expected": len(records),
        "n_processed": len(ok),
        "n_failed_rows": len(records) - len(ok),
        "all_passed": all(r["passed"] for r in records),
        "human_residual_max_m": max((r["human_residual_max_m"] for r in ok), default=None),
        "box_rot_err_deg_max": max((r["box_rot_err_deg_max"] for r in ok), default=None),
        "box_offset_residual_max_m": max(
            (r["box_offset_residual_max_m"] for r in ok), default=None
        ),
        "box_offset_norm_mean_m": [
            round(float(np.linalg.norm(r["box_offset_mean_m"])), 6) for r in ok
        ],
        "joint_limit_max_violation_rad": max(
            (r["joint_limit_max_violation_rad"] or 0.0 for r in ok), default=None
        ),
        "scale_rel_diff_max": max((r["scale_rel_diff"] or 0.0 for r in ok), default=None),
        "failed_gates_by_demo": {
            r["demo"]: (
                [f"error:{r['error']}"] if r["error"] is not None
                else [g for g, v in r["gates"].items() if not v]
            )
            for r in records
            if r["error"] is not None or not r["passed"]
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root", type=Path, default=DERIVED_DIR / "experiments" / "omniretarget_first"
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument(
        "--skip-val31-pairing", action="store_true",
        help="verify pairing on the pilot six only (A1 minimum)",
    )
    args = ap.parse_args()
    t0 = time.time()

    identity = run_identity(sys.argv)
    run_id = args.run_id or f"{_dt.date.today().strftime('%Y%m%d')}_{identity['git_short']}_a1"
    freeze_dir = args.out_root / run_id / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)

    print(f"[a1] run_id={run_id} rev={identity['git_short']} dirty={identity['git_dirty']}")
    print(f"[a1] census: reading {312} original NPZ headers ...")
    census, frames = census_originals()
    (freeze_dir / "census.json").write_text(json.dumps(census, indent=2, ensure_ascii=False))

    print("[a1] split verification ...")
    split = verify_split(frames, set(census["pt_without_teacher"]))
    (freeze_dir / "split_verification.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False)
    )

    print("[a1] frozen lists (hash rules) ...")
    lists = build_frozen_lists(frames)
    (freeze_dir / "frozen_lists.json").write_text(
        json.dumps(lists["lists"], indent=2, ensure_ascii=False)
    )
    print(f"    pilot6 = {lists['pilot6_demos']}")
    print(f"    val6   = {lists['val6_demos']}")

    print("[a1] asset hashes (archive/models/upstream/locked/largebox) ...")
    assets = audit_assets()
    heights = assets["upstream"]["subject_heights_m"]
    (freeze_dir / "assets.json").write_text(json.dumps(assets, indent=2, ensure_ascii=False))

    from ireset.embodied.g1_spec import G1Spec

    spec = G1Spec()
    joint_range = spec.joint_range

    pilot_records, pilot_boxes = verify_pairing_set(lists["pilot6_demos"], joint_range, heights)
    sum_pilot = summarize_pairing(pilot_records)
    (freeze_dir / "pairing_pilot6.json").write_text(
        json.dumps({"summary": sum_pilot, "records": pilot_records}, indent=2, ensure_ascii=False)
    )
    source_boxes = dict(pilot_boxes)
    summary_out: dict = {"pairing_pilot6": sum_pilot}
    if not args.skip_val31_pairing:
        val_only = [d for d in lists["val31_demos"] if d not in set(lists["pilot6_demos"])]
        val_records, val_boxes = verify_pairing_set(val_only, joint_range, heights)
        sum_val = summarize_pairing(val_records)
        (freeze_dir / "pairing_val31.json").write_text(
            json.dumps({"summary": sum_val, "records": val_records}, indent=2, ensure_ascii=False)
        )
        source_boxes.update(val_boxes)
        summary_out["pairing_val31"] = {
            k: sum_val[k] for k in ("n_expected", "n_processed", "all_passed")
        }
    print(f"[a1] §4.3 pairing: {sum_pilot['n_processed']}/{sum_pilot['n_expected']} pilot rows, "
          f"all_passed={sum_pilot['all_passed']}")

    np.savez(
        freeze_dir / "pairing_source_boxes.npz",
        **{demo: arr for demo, arr in sorted(source_boxes.items())},
    )

    summary = {
        "run_id": run_id,
        "stage": "A1 freeze & pairing (plan §8)",
        "duration_s": round(time.time() - t0, 1),
        "gates": {
            "census_structure": (
                census["npz_total"] == 1826
                and census["original_demos"] == 312
                and not census["variant_files_without_parent_demo"]
                and not census["fps_not_30"]
            ),
            "split_verification": split["passed"],
            "frozen_lists": lists["passed"],
            "assets_archive_matches_manifest": assets["published"]["archive_sha256_matches_manifest"],
            "pairing_pilot6": sum_pilot["all_passed"],
        },
        "joint_names_locked_order": list(spec.joint_names),
        **summary_out,
        "protocol_status": "omniretarget_motion_v1 not yet implemented (I1)",
    }
    (freeze_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (freeze_dir / "run_identity.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False)
    )

    print(json.dumps({k: summary[k] for k in ("gates", "pairing_pilot6")}, indent=2))
    print(f"\n[a1] freeze -> {freeze_dir}")
    ok = all(summary["gates"].values())
    print(f"[a1] overall: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
