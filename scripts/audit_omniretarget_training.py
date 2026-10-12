"""A3: published-label list & data audit for the W1 weak-motion round.

W1 plan §3 (unit I1/A3). Consumes the W0 registration (verified against
disk, never trusted), and produces the ``omniretarget_training_manifest_v1``
record for every candidate demo (train24 + val31 = 55):

* per-demo BASIC ADMISSION over the published ``*_original.npz`` labels
  (§3.2): structural shape/fps/T checks, finiteness, quaternion bounds with
  registered (never written-back) normalization, layout round-trip,
  joint-limit correspondence against the locked G1 asset, and the §4.3
  source/target pairing gates re-run on the actual NPZ+PT pair;
* scene split (§3.3): the source-scene box is derived from the PT through
  the same similarity fit A1 froze (bit-compared against the frozen
  ``pairing_source_boxes.npz`` wherever A1 has the demo — train18 pairs are
  new evidence), and the target box stays the qpos box; both are saved in a
  derived per-demo representation whose hash anchors later consumption;
* QUALITY registration (§3.3) with the frozen validator protocol
  (``TeacherQualityReplayer``: adaptive penetration, SLERP, per-site worst
  feet, frame-aligned task error, certified overlap) on the PUBLISHED
  labels — a strict quality failure is a registrable weak-supervision
  defect and NEVER cancels the motion admission; a run failure/timeout is
  ``incomplete``/``not_checked`` with the reason, never a zero or a pass.

Motion admission and quality status stay SEPARATE fields. Failures never
shrink the denominator: every candidate keeps exactly one final row
(``expected = processed + not_processed``), short trajectories are listed
separately, and nothing is re-selected because an audit row failed.

Exit criteria (§3.4): 55 final rows, admitted TRAIN labels carry a complete
quality registration, six subjects each have a usable full window. A miss
keeps the manifest and reports the reason (exit non-zero).

Run: uv run --locked python scripts/audit_omniretarget_training.py \
       --preregistration <run_root>/freeze/preregistration.json \
       --out <run_root>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data import a2_integrity as integ
from ireset.data import pairing
from ireset.data import w1_training as w1
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record

EXPECTED_SUBJECTS = ("sub10", "sub12", "sub16", "sub3", "sub7", "sub8")


def load_quality_replayer():
    from ireset.eval.teacher_comparison import TeacherQualityReplayer

    return TeacherQualityReplayer()


def audit_demo(
    demo: str,
    split: str,
    list_ids: list[str],
    replayer,
    joint_range: np.ndarray,
    heights: dict,
    a1_source_boxes: dict[str, np.ndarray],
    derived_dir: Path,
) -> dict:
    """One candidate -> one manifest row (admission/quality separated)."""
    row: dict = {
        "demo": demo,
        "subject": pairing.subject_of(demo),
        "variant": "original",
        "split": split,
        "list_ids": list_ids,
        "published_revision": w1.PUBLISHED_REVISION,
        "error": None,
        "admission": {"motion": False, "reasons": []},
        "review": None,
    }
    try:
        q_raw, human, fps = pairing.load_teacher_npz_raw(
            TEACHER_TRAJ_DIR / f"{demo}_original.npz")
        pt_human, pt_obj = pairing.load_pt(OMOMO_NEW_DIR / f"{demo}.pt")
    except Exception as exc:  # noqa: BLE001 -- explicit failure row
        row["error"] = f"load_failed:{type(exc).__name__}:{exc}"
        row["admission"]["reasons"] = [row["error"]]
        row["quality"] = {"status": "not_checked",
                          "reason": "inputs unreadable"}
        return row

    adm = w1.check_published_admission(demo, q_raw, human, fps,
                                       pt_human.shape[0], joint_range)
    row["admission"] = adm
    rec = pairing.verify_scene_pairing(
        demo, q_raw, human, pt_human, pt_obj, fps,
        joint_range=joint_range,
        subject_height_m=heights.get(pairing.subject_of(demo)),
    )
    row["pairing"] = rec.to_json_dict()
    if not rec.passed:
        row["admission"]["motion"] = False
        row["admission"]["reasons"] += [
            f"pairing_gate:{g}" for g, v in rec.gates.items() if not v]

    if not adm["motion"] and not rec.passed:
        row["quality"] = {"status": "not_checked",
                          "reason": "admission failed"}
        return row

    # scene split (§3.3): source box from the PT through the A1 estimator;
    # the fitted scale ALSO applies to the source box geometry (review fix
    # 1) — it is stored beside the pose and consumed by every schedule/
    # quality measurement against the source box
    source_box, sim_scale = w1.derive_source_box(pt_human, human, pt_obj)
    scene = w1.published_to_project(q_raw)
    target_box = scene[:, 36:43].copy()
    row["scene_split"] = w1.scene_split_facts(scene, source_box, target_box)
    row["scene_split"]["similarity_scale"] = sim_scale
    if demo in a1_source_boxes:
        match_a1 = bool(np.array_equal(source_box, a1_source_boxes[demo]))
        row["scene_split"]["matches_a1_frozen_source_box"] = match_a1
        if not match_a1:
            row["admission"]["motion"] = False
            row["admission"]["reasons"].append("source_box_mismatch_vs_a1")

    row["T"] = int(scene.shape[0])
    row["fps"] = float(fps)
    row["duration_s"] = round(scene.shape[0] / fps, 3)
    row["full_windows_F60_s30"] = w1.full_windows(scene.shape[0])
    row["short_trajectory"] = scene.shape[0] < w1.SHORT_TRAJ_FRAMES

    if row["admission"]["motion"]:
        npz_sha = integ.sha256_file(TEACHER_TRAJ_DIR / f"{demo}_original.npz")
        pt_sha = integ.sha256_file(OMOMO_NEW_DIR / f"{demo}.pt")
        row["inputs"] = {"npz_sha256": npz_sha, "pt_sha256": pt_sha}
        # qpos stays VERBATIM in the published layout so every consumer
        # rebuilds the scene through the one audited loader path; the scene
        # split (source/target boxes + geometry scale) travels beside it,
        # never inside qpos
        derived_path = derived_dir / f"{demo}_w1_scene.npz"
        np.savez(
            derived_path,
            qpos=q_raw,
            human_joints=human,
            fps=np.float64(fps),
            source_obj_pose=source_box,
            target_obj_pose=target_box,
            similarity_scale=np.float64(sim_scale),
            meta_json=json.dumps({
                "demo": demo, "split": split,
                "protocol": w1.MOTION_PROTOCOL_VERSION,
                "npz_sha256": npz_sha, "pt_sha256": pt_sha,
                "similarity_scale": sim_scale,
                "layout": "published_verbatim",
            }),
        )
        row["derived"] = {"path": path_for_record(derived_path),
                          "sha256": integ.sha256_file(derived_path)}

    if not row["admission"]["motion"]:
        row["quality"] = {"status": "not_checked",
                          "reason": "admission failed"}
        return row

    # quality registration: the frozen validator protocol on the PUBLISHED
    # labels; a strict failure is a registrable defect, a run failure is
    # incomplete/not_checked — never a zero, never a pass
    try:
        metrics = replayer.measure(demo, scene, human, source_box, fps,
                                   source_box_scale=sim_scale)
        row["quality"] = {"status": "complete", "reason": None,
                          "metrics": metrics.to_json()}
    except Exception as exc:  # noqa: BLE001 -- recorded, never raised
        row["quality"] = {"status": "incomplete",
                          "reason": f"{type(exc).__name__}: {exc}",
                          "metrics": None}
    return row


def _finite_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(float(x))


def quality_registration_complete(row: dict) -> tuple[bool, list[str]]:
    """Is this row's quality registration backed by ACTUAL measurements?

    Review fix 5 (deepened round 2): a ``status=complete`` flag without its
    measurement fields is a completeness claim, not evidence. Required:

    * the finite core measurements (penetration/task/box-ground AND the
      keyframe violation ratio) plus finite fps;
    * per-side hand/foot measurements must EXIST exactly where the recorded
      reference frames exist — ``None`` stays legal ONLY for a genuinely
      zero-reference side (checked against ``ref_grasp_frames`` /
      ``ref_support_frames``, themselves reconciled against the recorded
      ``reference_intervals``);
    * the both-hands block measured wherever its denominator is non-zero.

    Round-3 deepening: the per-side count KEYS must both exist (a deleted
    ``l`` key is missing evidence, never zero and never the recomputed
    value), and the both-hands denominator is recomputed from the reference
    intervals as the left∩right grasp-frame intersection.

    A deleted measurement must therefore surface as a gap (missing evidence),
    never as a legal zero-reference side.
    """
    q = row.get("quality") or {}
    m = q.get("metrics")
    if q.get("status") != "complete" or not isinstance(m, dict):
        return False, [f"status:{q.get('status')}"]
    problems: list[str] = []
    if m.get("valid") is not True:
        problems.append("metrics_not_valid")
    for k in ("penetration_max_m", "penetration_violation_ratio",
              "task_mpjpe_p95_m", "box_ground_penetration_max_m", "fps"):
        if not _finite_num(m.get(k)):
            problems.append(f"{k}:{m.get(k)}")

    # per-side reference-frame counts — the EXISTENCE contract for the
    # hand/foot measurements; reconciled against the recorded intervals.
    # Round-3 review fix 4: both ``l`` and ``r`` keys must EXIST — a missing
    # key was read through ``.get(side, <recomputed>)`` (laundering the
    # deletion) and then as zero (declaring a legal zero-reference side)
    ref_support = m.get("ref_support_frames")
    ref_grasp = m.get("ref_grasp_frames")
    if not isinstance(ref_support, dict) or not isinstance(ref_grasp, dict):
        problems.append("ref_frame_counts_missing")
        ref_support = ref_support if isinstance(ref_support, dict) else {}
        ref_grasp = ref_grasp if isinstance(ref_grasp, dict) else {}
    intervals = m.get("reference_intervals")
    recon_support = {"l": 0, "r": 0}
    recon_grasp = {"l": 0, "r": 0}
    grasp_frames = {"l": set(), "r": set()}
    if not isinstance(intervals, list):
        problems.append("reference_intervals_missing")
    else:
        for iv in intervals:
            if not isinstance(iv, dict):
                continue
            eff = str(iv.get("eff", iv.get("effector", "")))
            side = ("l" if eff.startswith("left")
                    else "r" if eff.startswith("right") else None)
            start, end = iv.get("start"), iv.get("end")
            if (side is None or not isinstance(start, int)
                    or not isinstance(end, int) or end < start):
                continue
            n = end - start
            if eff.endswith("_foot") and iv.get("sup") == "ground":
                recon_support[side] += n
            elif eff.endswith("_hand") and iv.get("sup") == "object":
                recon_grasp[side] += n
                grasp_frames[side].update(range(start, end))
    for side in ("l", "r"):
        if ref_support.get(side) != recon_support[side]:
            problems.append(
                f"foot[{side}]:ref_frames_vs_intervals:"
                f"{ref_support.get(side)}!={recon_support[side]}")
        if ref_grasp.get(side) != recon_grasp[side]:
            problems.append(
                f"hand[{side}]:ref_frames_vs_intervals:"
                f"{ref_grasp.get(side)}!={recon_grasp[side]}")

    for side in ("l", "r"):
        for group, counts, keys in (
                ("foot", ref_support, ("gap_p95_m", "speed_p95_mps",
                                       "drift_max_m")),
                ("hand", ref_grasp, ("gap_p95_m", "coverage"))):
            need = counts.get(side)
            if need is None:
                problems.append(f"{group}[{side}]:ref_frames_key_missing")
                continue
            if not isinstance(need, int) or isinstance(need, bool) or need < 0:
                problems.append(f"{group}[{side}]:ref_frames:{need}")
                continue
            entry = (m.get(group) or {}).get(side)
            if need == 0:
                # legal zero-reference side ONLY as an absent measurement
                if entry is not None:
                    problems.append(f"{group}[{side}]:measurement_without_reference")
                continue
            if entry is None:
                # reference frames EXIST but the measurement was deleted —
                # the reviewer's counterexample: NOT a legal zero side
                problems.append(f"{group}[{side}]:reference_without_measurement")
                continue
            if not isinstance(entry, dict):
                problems.append(f"{group}[{side}]:not_a_dict")
                continue
            nrf = entry.get("n_ref_frames")
            if nrf != need:
                problems.append(f"{group}[{side}].n_ref_frames:{nrf}!={need}")
            for k in keys:
                if not _finite_num(entry.get(k)):
                    problems.append(f"{group}[{side}].{k}:{entry.get(k)}")

    # both-hands block: the denominator is RECOMPUTED from the reference
    # intervals (left-grasp frames ∩ right-grasp frames) — a doctored 0 with
    # deleted measurements used to pass as a legal zero-reference (round-3
    # review fix 4)
    both = m.get("both")
    if not isinstance(both, dict) or "denominator_frames" not in both:
        problems.append("both_missing")
    else:
        den = both.get("denominator_frames")
        both_need = len(grasp_frames["l"] & grasp_frames["r"])
        if not isinstance(den, int) or isinstance(den, bool) or den < 0:
            problems.append(f"both.denominator_frames:{den}")
        elif den != both_need:
            problems.append(
                f"both.denominator_frames_vs_intervals:{den}!={both_need}")
        elif both_need == 0:
            if both.get("valid_coverage") is not None \
                    or both.get("gap_series_p95_m") is not None:
                problems.append("both:measurement_without_reference")
        else:
            for k in ("valid_coverage", "gap_series_p95_m"):
                if not _finite_num(both.get(k)):
                    problems.append(f"both.{k}:{both.get(k)}")
    if m.get("T") != row.get("T"):
        problems.append(f"T:{m.get('T')}!={row.get('T')}")
    return (not problems), problems


def summarize(rows: list[dict], scope: list[str]) -> dict:
    processed = [r for r in rows if r.get("status") != "not_processed"]
    by_split: dict[str, dict] = {}
    for r in processed:
        s = by_split.setdefault(r["split"], {"rows": 0, "admitted": 0,
                                             "rejected": 0})
        s["rows"] += 1
        if r["admission"]["motion"]:
            s["admitted"] += 1
        else:
            s["rejected"] += 1
    admitted = [r for r in processed if r["admission"]["motion"]]
    admitted_train = [r for r in admitted if r["split"] == "train"]
    subj_windows = {
        subj: sum(r.get("full_windows_F60_s30", 0) for r in admitted_train
                  if r["subject"] == subj)
        for subj in EXPECTED_SUBJECTS}
    # §3.4: the exit criterion is the TRAIN label quality registration; a
    # val label with incomplete quality keeps its row (its generation task
    # stays in the full expected set) and is listed separately. The
    # completeness predicate checks the MEASURED fields, not the status
    # string (review fix 5)
    train_quality_incomplete = [
        r["demo"] for r in admitted_train
        if not quality_registration_complete(r)[0]]
    val_quality_incomplete = [
        r["demo"] for r in admitted if r["split"] == "val"
        and not quality_registration_complete(r)[0]]
    return {
        "n_expected": len(scope),
        "n_processed": len(processed),
        "n_not_processed": len(rows) - len(processed),
        "n_admitted": len(admitted),
        "n_rejected": sum(1 for r in processed if not r["admission"]["motion"]),
        "by_split": by_split,
        "independent_frames_admitted": int(sum(
            r.get("T", 0) for r in admitted)),
        "independent_demos_admitted": len(admitted),
        "full_windows_admitted": int(sum(
            r.get("full_windows_F60_s30", 0) for r in admitted)),
        "short_trajectories": [r["demo"] for r in rows if r.get("short_trajectory")],
        "train_quality_incomplete": train_quality_incomplete,
        "val_quality_incomplete": val_quality_incomplete,
        "train_subject_full_windows": subj_windows,
    }


def exit_criteria(summary: dict, rows: list[dict]) -> dict:
    """§3.4: full denominator, complete train quality registration, six
    subjects with usable windows — each recomputed from the rows."""
    missing = [r["demo"] for r in rows if r.get("status") == "not_processed"]
    subj_ok = all(v >= 1 for v in summary["train_subject_full_windows"].values())
    return {
        "all_candidates_processed": (summary["n_processed"] == summary["n_expected"]
                                     and not missing),
        "train_label_quality_complete": not summary["train_quality_incomplete"],
        "six_subjects_usable_train_windows": subj_ok,
        "missing_rows": missing,
    }


BOOL_EXIT_GATES = ("all_candidates_processed", "train_label_quality_complete",
                   "six_subjects_usable_train_windows")


def exit_ok(gates: dict, n_budget_stop: int = 0) -> bool:
    """Only the BOOLEAN gates decide — ``missing_rows`` is the evidence list
    (an empty list is a PASS; ``all(gates.values())`` treated it as falsy)."""
    return (all(gates.get(k) is True for k in BOOL_EXIT_GATES)
            and n_budget_stop == 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preregistration", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="run root (writes audit/attempt_NNN inside)")
    ap.add_argument("--demos", type=str, default=None,
                    help="comma-separated retry scope (subset of train24+val31); "
                         "writes a NEW attempt; the denominator stays 55")
    args = ap.parse_args()
    t0 = time.time()

    prereg_path = args.preregistration.expanduser().resolve()
    run_root = integ.guard_output_dir(args.out)
    prereg = json.loads(prereg_path.read_text())
    check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                      TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    if not check["passed"]:
        raise SystemExit(f"preregistration re-verification failed: "
                         f"{check['problems']}")

    sel = prereg["selection"]
    train24, val31 = list(sel["train24"]), list(sel["val31"])
    scope_by_demo = {**{d: ("train", [w1.TRAIN24_LIST_ID]) for d in train24},
                     **{d: ("val", [w1.VAL31_LIST_ID,
                                    *( [w1.VAL6_LIST_ID] if d in sel["val6"] else [])])
                        for d in val31}}
    scope = list(train24) + list(val31)
    if args.demos:
        wanted = [d.strip() for d in args.demos.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in scope_by_demo]
        if unknown:
            raise SystemExit(f"--demos outside registered lists: {unknown}")
        if len(set(wanted)) != len(wanted):
            raise SystemExit("duplicate demo in --demos")
        scope = wanted

    audit_root = run_root / "audit"
    attempt_dir = integ.allocate_attempt(audit_root)
    derived_dir = attempt_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    from ireset.embodied.g1_spec import G1Spec

    joint_range = G1Spec().joint_range
    a1_freeze = PROJECT_ROOT / prereg["assets"]["a1_freeze_path"]
    assets = json.loads((a1_freeze / "assets.json").read_text())
    heights = assets["upstream"]["subject_heights_m"]
    a1_boxes: dict[str, np.ndarray] = {}
    boxes_path = a1_freeze / "pairing_source_boxes.npz"
    if boxes_path.is_file():
        with np.load(boxes_path, allow_pickle=False) as z:
            a1_boxes = {k: z[k] for k in z.files}
    replayer = load_quality_replayer()

    rows: list[dict] = []
    n_budget_stop = 0
    for demo in scope:
        elapsed = time.time() - t0
        if elapsed > w1.W0A3_BUDGET_S:
            rows.append({"demo": demo, "split": scope_by_demo[demo][0],
                         "admission": {"motion": False, "reasons": []},
                         "status": "not_processed",
                         "reason": f"w0_a3 budget exhausted at {elapsed:.0f}s"})
            n_budget_stop += 1
            continue
        split, list_ids = scope_by_demo[demo]
        row = audit_demo(demo, split, list_ids, replayer, joint_range,
                         heights, a1_boxes, derived_dir)
        row["status"] = "processed"
        rows.append(row)
        q = row.get("quality", {})
        print(f"[a3] {demo}: admission={row['admission']['motion']} "
              f"quality={q.get('status')}"
              f"{'' if row['admission']['motion'] else ' reasons=' + str(row['admission']['reasons'])}")

    summary = summarize(rows, scope)
    gates = exit_criteria(summary, rows)
    manifest = {
        "manifest_name": w1.MANIFEST_NAME,
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "experiment_id": w1.EXPERIMENT_ID,
        "training_purpose": w1.TRAINING_PURPOSE,
        "source_type": w1.SOURCE_TYPE,
        "published_revision": w1.PUBLISHED_REVISION,
        "attempt": attempt_dir.name,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "preregistration": {"path": path_for_record(prereg_path),
                            "sha256": integ.sha256_file(prereg_path)},
        "identity": {"git_rev": _git_rev(), "rows_schema": "a3_v1"},
        "selection": {
            "train24": train24,
            "train24_canonical_sha256": sel["train24_canonical_sha256"],
            "val31": val31,
            "val6": list(sel["val6"]),
        },
        "counts": summary,
        "exit_criteria": gates,
        "rows": rows,
        "total_wall_s": round(time.time() - t0, 1),
    }
    out = attempt_dir / "training_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    integ.append_attempts_index(audit_root, {
        "attempt": attempt_dir.name,
        "scope": scope,
        "expected_denominator": len(train24) + len(val31),
        "n_processed": summary["n_processed"],
        "n_admitted": summary["n_admitted"],
        "complete": gates["all_candidates_processed"],
        "manifest": path_for_record(out),
    })
    _append_costs(run_root, "a3_audit", time.time() - t0)

    print(json.dumps({"counts": {k: v for k, v in summary.items()
                                 if not isinstance(v, dict)},
                      "exit_criteria": gates}, indent=2))
    print(f"[a3] manifest -> {path_for_record(out)}")
    ok = exit_ok(gates, n_budget_stop)
    print(f"[a3] overall: {'PASS' if ok else 'FAIL'} "
          f"(denominator kept at {len(train24) + len(val31)})")
    return 0 if ok else 1


def _git_rev() -> str:
    import subprocess

    return subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def _append_costs(run_root: Path, stage: str, wall: float) -> None:
    costs_path = run_root / "costs.json"
    entries = []
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    entries.append({"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "stage": stage, "wall_s": round(wall, 1)})
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
