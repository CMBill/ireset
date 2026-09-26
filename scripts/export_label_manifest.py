"""Graded-label export: quality manifest + supervision-mask NPZs for a run.

Label-quality plan section 3.2: for every demo of a regeneration run, ONE
manifest row carrying status, split group, input/output/asset/config/
protocol hashes, frame counts, fps, failure evidence, label eligibility and
cost; every kept trajectory additionally gets a lightweight label NPZ
(``hand_proximity_mask[T,2]``, ``hand_contact_supervision_mask[T,2]``,
``hand_supervision_interval_id[T,2]``, protocol metadata). All 24 (or 6)
demos keep a row -- rejected ones carry explicit missing reasons, never
zeroed metrics.

Masks are RECOMPUTED from the stored artifacts (kept NPZ + source original
NPZ) with the independent validator -- the same evidence chain as the
quality tables, so the manifest is verifiable without the run's process
state. A kept trajectory whose strict re-validation fails is recorded as
ineligible (``strict_recheck_passed: false``); the plan's transition gate
reads that flag instead of trusting the report.

Run: uv run --locked python scripts/export_label_manifest.py \
        --run derived/data/teachers_regenerated/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from closed_loop_metrics import sha256
from experiment_provenance import env_snapshot, git_state
from ireset.data.labels import (
    LABEL_PROTOCOL_VERSION,
    LABEL_SIDES,
    both_hands_metrics,
    side_label_arrays,
)
from ireset.data.teacher import load_teacher_npz
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.paths import PROJECT_ROOT, TEACHER_TRAJ_DIR
from ireset.repair.config_io import MODES, load_mode_configs
from ireset.repair.context import ReferenceCondition
from ireset.repair.validation import IndependentValidator

PROTOCOL_YAML = PROJECT_ROOT / "configs" / "labels" / "graded_labels_v1.yaml"
SPLIT_DIR = PROJECT_ROOT / "derived" / "data" / "split"


def _split_groups() -> dict[str, str]:
    """demo -> split group (train/val/test); demos outside every list map to
    ``unassigned`` (registered diagnosis sets may draw from any group)."""
    groups: dict[str, str] = {}
    for split in ("train", "val", "test"):
        d = json.loads((SPLIT_DIR / f"{split}.json").read_text())
        for demo in d["demos"]:
            groups[demo] = split
    return groups


def _side_block(vcfg, cond: ReferenceCondition, validator: IndependentValidator,
                rep, T: int) -> tuple[dict, dict[str, np.ndarray]]:
    """Per-side label arrays + compact evidence from ONE validator pass."""
    from ireset.data.labels import continuous_runs

    sides: dict[str, np.ndarray] = {}
    evidence = {}
    for side, eff in zip(LABEL_SIDES, ("left_hand", "right_hand")):
        ref_intervals = [(iv.start, iv.end) for iv in cond.ref_schedule.intervals
                         if iv.effector == eff]
        ref = np.zeros(T, dtype=bool)
        for a, b in ref_intervals:
            ref[a:b] = True
        cert_ivs = [(iv.start, iv.end) for iv in rep.schedule.intervals
                    if iv.effector == eff and iv.support == "object"]
        arrays = side_label_arrays(validator._cache["hand_dist"][side], ref,
                                   cert_ivs, float(vcfg.hand_enter_m))
        sides[side] = arrays
        evidence[side] = {
            "ref_frames": int(ref.sum()),
            "ref_intervals": [[int(a), int(b)] for a, b in ref_intervals],
            "certified_frames": int(arrays["certified"].sum()),
            "certified_in_ref_frames": int(arrays["supervision"].sum()),
            "intersection_intervals": [[int(a), int(b)] for a, b in
                                       continuous_runs(arrays["supervision"])],
            "proximity_frames": int(arrays["proximity"].sum()),
            "certified_intervals": [[int(a), int(b)] for a, b in cert_ivs],
            "enter_m": float(vcfg.hand_enter_m),
        }
    return evidence, sides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True,
                    help="regeneration run directory (report + kept NPZs)")
    ap.add_argument("--report", type=str, default=None,
                    help="explicit report path (default: the single "
                         "regen_report_*.json inside --run)")
    ap.add_argument("--validation-mode", choices=MODES, default="strict")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory for labels/ + label_manifest.json "
                         "(default: the run directory itself; historical run "
                         "directories are read-only -- pass a fresh --out)")
    ap.add_argument("--note", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    report_path = (Path(args.report) if args.report else None)
    if report_path is None:
        candidates = sorted(run_dir.glob("regen_report_*.json"))
        if len(candidates) != 1:
            raise SystemExit(
                f"expected exactly one regen_report_*.json in {run_dir}, "
                f"found {[c.name for c in candidates]} -- pass --report")
        report_path = candidates[0]
    report = json.loads(report_path.read_text())
    if report["mode"] != "regen":
        raise SystemExit("labels are only exported for --mode regen runs "
                         "(standard mode writes no NPZ)")
    if report["validation_mode"] != args.validation_mode:
        raise SystemExit(f"report validation_mode {report['validation_mode']} "
                         f"!= {args.validation_mode}")
    row_by_demo = {r["demo"]: r for r in report["rows"]}

    # input hashes must still match what the run consumed (manifest/source
    # consistency gate before any expansion)
    for inp in report.get("inputs") or []:
        cur = sha256(Path(inp["path"]))
        if cur != inp["sha256"]:
            raise SystemExit(f"input {inp['demo']} changed since the run: "
                             f"{cur} != {inp['sha256']}")

    rcfg, vcfg = load_mode_configs(args.validation_mode)
    validator = IndependentValidator(cfg=vcfg)
    groups = _split_groups()
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_labels = out_dir / "labels"
    out_labels.mkdir(parents=True, exist_ok=True)

    rows = []
    n_eligible = 0
    t_all = time.perf_counter()
    for demo, row in row_by_demo.items():
        t0 = time.perf_counter()
        out = {
            "demo": demo,
            "group": groups.get(demo, "unassigned"),
            "status": row["status"],
            "accepted": bool(row["accepted"]),
            "keep": bool(row.get("hand", {}).get("validator_passed", False)
                         and row.get("hand_screen") == "pass"),
            "T": int(row["T"]),
            "fps": float(row["fps"]),
            "input_npz": {"path": str(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
                          "sha256": next((i["sha256"] for i in report.get("inputs") or []
                                          if i["demo"] == demo), None)},
            "cost": {"regen_wall_s": row.get("wall_s"),
                     "regen_total_s": row.get("total_s"),
                     "regen_attempts": (row.get("cost") or {}).get("attempts"),
                     "label_export_s": None},
            "label_export_s": None,
        }
        npz_path = run_dir / f"{demo}.npz"
        if not out["keep"]:
            out.update({
                "output_npz": None,
                "label_npz": None,
                "missing_reason": (
                    f"no kept trajectory (status={row['status']}; "
                    f"hand_screen={row.get('hand_screen')})"),
                "failure_evidence": row.get("rejection") or row.get("suffix_retry"),
                "label_eligibility": {"motion_fk": False, "proximity": False,
                                      "reason": "trajectory not kept"},
            })
            rows.append(out)
            continue
        if not npz_path.exists():
            raise SystemExit(f"{demo}: report says keep but {npz_path} is missing")

        # masks + evidence recomputed from the stored artifacts
        traj, human, fps = load_teacher_npz(npz_path)
        if traj.shape[0] != out["T"]:
            raise SystemExit(f"{demo}: kept NPZ T={traj.shape[0]} != report T={out['T']}")
        if abs(fps - out["fps"]) > 1e-9:
            raise SystemExit(f"{demo}: kept NPZ fps={fps} != report fps={out['fps']}")
        src_scene, _h, _fps = load_teacher_npz(TEACHER_TRAJ_DIR / f"{demo}_original.npz")
        if src_scene.shape[0] != out["T"]:
            raise SystemExit(f"{demo}: source T={src_scene.shape[0]} != report T={out['T']}"
                             " (run used --frames?) -- labels require full clips")
        cond = ReferenceCondition(human, src_scene[:, 36:43],
                                  infer_reference_schedule(human, src_scene[:, 36:43]))
        rep = validator.validate(traj, exclude=("box_ground",))
        evidence, sides = _side_block(vcfg, cond, validator, rep, traj.shape[0])
        both = both_hands_metrics(
            {s: {"certified": a["certified"], "supervision": a["supervision"]}
             for s, a in sides.items()})

        T = traj.shape[0]
        prox = np.stack([sides[s]["proximity"] for s in LABEL_SIDES], axis=1)
        sup = np.stack([sides[s]["supervision"] for s in LABEL_SIDES], axis=1)
        ids = np.stack([sides[s]["interval_id"] for s in LABEL_SIDES], axis=1)
        meta = {
            "demo": demo, "T": int(T), "fps": float(fps),
            "label_protocol": LABEL_PROTOCOL_VERSION,
            "sides": list(LABEL_SIDES),
            "enter_m": float(vcfg.hand_enter_m),
            "proximity_operator": "<",
            "evidence": evidence, "hands": both,
            "source_npz_sha256": out["input_npz"]["sha256"],
            "trajectory_npz_sha256": sha256(npz_path),
        }
        label_path = out_labels / f"{demo}.npz"
        np.savez(label_path,
                 hand_proximity_mask=prox,
                 hand_contact_supervision_mask=sup,
                 hand_supervision_interval_id=ids,
                 meta_json=np.array(json.dumps(meta, default=float)))
        out["label_export_s"] = round(time.perf_counter() - t0, 3)
        out["cost"]["label_export_s"] = out["label_export_s"]
        out.update({
            "output_npz": {"path": str(npz_path), "sha256": sha256(npz_path)},
            "label_npz": {"path": str(label_path), "sha256": sha256(label_path)},
            "strict_recheck_passed": bool(rep.passed),
            "label_eligibility": {
                # motion/FK supervision: full kept trajectory (strict + task
                # gate + keep came with the run; strict is re-verified here)
                "motion_fk": bool(rep.passed),
                "proximity": bool(evidence["l"]["proximity_frames"] > 0
                                  or evidence["r"]["proximity_frames"] > 0),
                "supervision_frames": [evidence[s]["certified_in_ref_frames"]
                                       for s in LABEL_SIDES],
                "supervision_intervals": [evidence[s]["intersection_intervals"]
                                          for s in LABEL_SIDES],
                "proximity_frames": [evidence[s]["proximity_frames"]
                                     for s in LABEL_SIDES],
                "reason": (None if rep.passed else
                           "strict re-validation of the stored trajectory failed"),
            },
            "quality": {"sides": evidence, "hands": both},
        })
        if rep.passed:
            n_eligible += 1
        rows.append(out)
        print(f"[{demo}] strict_recheck={rep.passed} "
              f"supervision={out['label_eligibility']['supervision_frames']} "
              f"proximity={out['label_eligibility']['proximity_frames']} "
              f"({out['label_export_s']}s)", flush=True)

    manifest = {
        "label_protocol": LABEL_PROTOCOL_VERSION,
        "protocol_yaml_sha256": (sha256(PROTOCOL_YAML)
                                 if PROTOCOL_YAML.exists() else None),
        "source_run": {
            "dir": str(run_dir),
            "export_dir": str(out_dir),
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "mode": report["mode"],
            "validation_mode": report["validation_mode"],
            "task_mpjpe_tol_m": report["effective"]["task_mpjpe_tol_m"],
            "effective_config_sha256": report["effective"]["effective_config_sha256"],
            "min_hand_recall": report.get("min_hand_recall"),
            "min_ref_frames": report.get("min_ref_frames"),
            "git": report.get("freeze", {}).get("git") or git_state(),
            "assets": report.get("freeze", {}).get("assets"),
            "uv_lock_sha256": report.get("freeze", {}).get("uv_lock_sha256"),
        },
        "export": {
            "git": git_state(),
            "env": env_snapshot(),
            "wall_s": round(time.perf_counter() - t_all, 3),
            "validation_mode": args.validation_mode,
            "note": args.note,
        },
        "summary": {
            "n_demos": len(rows),
            "n_keep": sum(1 for r in rows if r["keep"]),
            "n_eligible_motion_fk": n_eligible,
            "all_kept_strict_recheck_passed": all(
                r.get("strict_recheck_passed", False) for r in rows if r["keep"]),
        },
        "rows": rows,
    }
    out_path = out_dir / "label_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=1, default=float))
    print(f"manifest -> {out_path} "
          f"(keep {manifest['summary']['n_keep']}/{len(rows)}, "
          f"eligible {n_eligible})")


if __name__ == "__main__":
    main()
