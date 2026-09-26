"""train24 label-experiment analysis (label-quality plan section 4.1 step 6).

Aggregates a regeneration run report (+ its label manifest when present)
with per-demo source statistics recomputed from the input NPZs: subject
distribution, duration, reference contact phase, box translation/rotation
and human motion range. Retained vs rejected sets are contrasted, and the
six repeatedly-diagnosed segments are listed SEPARATELY from the other 18
(the six have been through many solver iterations; their results are not
independent generalization evidence).

Window counts use the training windowing rule (60 frames, stride 30) over
the KEPT trajectories -- the same formula TeacherWindowDataset indexes by,
without loading the model.

Run: uv run --locked python scripts/train24_label_analysis.py \
        --run <train24 run dir> [--labels <label dir>] --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from ireset.embodied.contact_sites import SMPLH_INDEX, infer_reference_schedule
from ireset.data.teacher import load_teacher_npz
from ireset.paths import TEACHER_TRAJ_DIR

DIAGNOSIS6 = ("sub10_largebox_000", "sub12_largebox_001", "sub16_largebox_000",
              "sub3_largebox_000", "sub7_largebox_000", "sub8_largebox_000")


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0, q1)))
    return 2.0 * float(np.arccos(np.clip(d, 0.0, 1.0)))


def demo_stats(demo: str) -> dict:
    scene, human, fps = load_teacher_npz(TEACHER_TRAJ_DIR / f"{demo}_original.npz")
    T = scene.shape[0]
    sched = infer_reference_schedule(human, scene[:, 36:43])
    hand_frames = np.zeros(T, dtype=bool)
    both = np.zeros(T, dtype=bool)
    for eff, side in (("left_hand", 0), ("right_hand", 1)):
        m = sched.mask(eff, T)
        hand_frames |= m
        if side == 0:
            left = m
        else:
            both |= left & m
    pelvis = human[:, SMPLH_INDEX["Pelvis"]]
    span = pelvis.max(0) - pelvis.min(0)
    speed = np.concatenate(
        [[0.0], np.linalg.norm(np.diff(pelvis, axis=0), axis=1)]) * fps
    pos, quat = scene[:, 36:39], scene[:, 39:43]
    rot = sum(quat_angle(quat[t], quat[t + 1]) for t in range(T - 1))
    return {
        "T": int(T), "fps": float(fps), "duration_s": round(T / fps, 2),
        "ref_hand_frames": int(hand_frames.sum()),
        "ref_hand_fraction": round(float(hand_frames.mean()), 4),
        "ref_both_hand_frames": int(both.sum()),
        "ref_hand_intervals": int(sum(
            1 for iv in sched.intervals if iv.effector in ("left_hand", "right_hand"))),
        "box_translation_m": round(float(np.linalg.norm(pos[-1] - pos[0])), 3),
        "box_rotation_rad": round(float(rot), 3),
        "human_pelvis_span_m": [round(float(x), 3) for x in span],
        "human_pelvis_max_speed_mps": round(float(speed.max()), 3),
    }


def n_windows(T: int, window: int = 60, stride: int = 30) -> int:
    return max(0, (T - window) // stride + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train24 run dir (with the report)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--labels", default=None,
                    help="label-manifest dir (labels exported by export_label_manifest)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    report_path = Path(args.report) if args.report else \
        next(iter(sorted(run_dir.glob("regen_report_*.json"))))
    report = json.loads(report_path.read_text())
    label_rows = {}
    if args.labels:
        lm = json.loads((Path(args.labels) / "label_manifest.json").read_text())
        label_rows = {r["demo"]: r for r in lm["rows"]}

    rows = []
    for r in report["rows"]:
        demo = r["demo"]
        d = demo_stats(demo)
        keep = bool(r.get("hand", {}).get("validator_passed", False)
                    and r.get("hand_screen") == "pass")
        row = {
            "demo": demo,
            "subject": demo.split("_")[0],
            "diagnosis6": demo in DIAGNOSIS6,
            "status": r["status"], "keep": keep,
            "wall_s": r.get("wall_s"),
            "attempts": (r.get("cost") or {}).get("attempts"),
            "task_p95": (r.get("task_mpjpe") or {}).get("p95"),
            "hand_screen": r.get("hand_screen"),
            "windows": n_windows(d["T"]) if keep else 0,
        }
        hand = r.get("hand") or {}
        row["contact_recall"] = [hand.get("left_hand_contact_recall"),
                                 hand.get("right_hand_contact_recall")]
        if demo in label_rows:
            lr = label_rows[demo]
            el = lr.get("label_eligibility", {})
            row["supervision_frames"] = el.get("supervision_frames",
                                               el.get("reason"))
            row["proximity_frames"] = el.get("proximity_frames")
            row["strict_recheck"] = lr.get("strict_recheck_passed")
        row.update(d)
        rows.append(row)
        print(f"[{demo}] keep={keep} T={d['T']} ({d['duration_s']}s) "
              f"ref_hand={d['ref_hand_fraction']} box_t={d['box_translation_m']}m "
              f"rot={d['box_rotation_rad']}rad", flush=True)

    def agg(sel):
        n = sum(1 for r in rows if sel(r))
        return {
            "n": n,
            "subjects": sorted({r["subject"] for r in rows if sel(r)}),
            "windows": sum(r["windows"] for r in rows if sel(r)),
            "duration_s_sum": round(sum(r["duration_s"] for r in rows if sel(r)), 1),
            "mean_ref_hand_fraction": round(float(np.mean(
                [r["ref_hand_fraction"] for r in rows if sel(r)])), 4) if n else None,
            "mean_box_translation_m": round(float(np.mean(
                [r["box_translation_m"] for r in rows if sel(r)])), 3) if n else None,
            "mean_box_rotation_rad": round(float(np.mean(
                [r["box_rotation_rad"] for r in rows if sel(r)])), 3) if n else None,
            "mean_pelvis_span_m": (round(float(np.mean(
                [r["human_pelvis_span_m"][0] for r in rows if sel(r)])), 3),
                round(float(np.mean(
                    [r["human_pelvis_span_m"][1] for r in rows if sel(r)])), 3),
                round(float(np.mean(
                    [r["human_pelvis_span_m"][2] for r in rows if sel(r)])), 3)) if n else None,
            "zero_supervision_sides": sum(
                1 for r in rows if sel(r) and isinstance(r.get("supervision_frames"), list)
                and r["supervision_frames"] == [0, 0]),
        }

    kept = [r for r in rows if r["keep"]]
    out = {
        "report": str(report_path), "n_demos": len(rows),
        "n_keep": len(kept),
        "keep_rule": "strict + task gate 0.40 + per-side recall>=0.5 (>=10 ref frames)",
        "pre_registered_gate": {"quantity": ">=18/24 keep",
                                "denominator": "frozen 24 (calibration24_train_v1)"},
        "groups": {
            "kept": agg(lambda r: r["keep"]),
            "rejected": agg(lambda r: not r["keep"]),
            "diagnosis6": agg(lambda r: r["diagnosis6"]),
            "other18": agg(lambda r: not r["diagnosis6"]),
            "kept_diagnosis6": agg(lambda r: r["diagnosis6"] and r["keep"]),
            "kept_other18": agg(lambda r: not r["diagnosis6"] and r["keep"]),
        },
        "per_subject_kept": {
            s: [sum(1 for r in rows if r["subject"] == s and r["keep"]),
                sum(1 for r in rows if r["subject"] == s)]
            for s in sorted({r["subject"] for r in rows})},
        "rows": rows,
    }
    # motion-coverage floor: every subject with at least one usable window
    out["motion_coverage_floor"] = {
        s: any(r["windows"] > 0 for r in rows if r["subject"] == s)
        for s in sorted({r["subject"] for r in rows})}

    out_path = Path(args.out) if args.out else run_dir / "train24_analysis.json"
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nanalysis -> {out_path}")
    print(f"keep {out['n_keep']}/{out['n_demos']}  "
          f"(>=18/24 gate: {'PASS' if out['n_keep'] >= 18 else 'FAIL'})")
    for g in ("kept", "rejected", "diagnosis6", "other18"):
        print(f"  {g}: {json.dumps(out['groups'][g])}")
    print("  per-subject kept:", out["per_subject_kept"])
    print("  motion coverage floor (60f window per subject):",
          out["motion_coverage_floor"])


if __name__ == "__main__":
    main()
