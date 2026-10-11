"""W0: run preregistration for the W1 weak-motion-supervision round.

W1 plan (docs/plans/omniretarget_w1_a3bc_plan.md) §2/§8 unit W0. This entry
CREATES the run registration — it does not audit data, build caches or train
anything. Before writing it re-verifies the entry evidence from disk:

* the plan §2.1 anchors re-hash to their declared SHA-256 (a list file that
  still exists is not evidence until its bytes match);
* the C2 final decision (attempt_010) is registration-complete with the
  unchanged verdicts (baseline ``incomplete`` / recovery ``no_quality_gain``
  / training ``not_ready``) — this round NEVER rewrites those;
* the UR comparison report is re-derived (``reverify_verdict_record``, the
  same leg the closure uses — never the top-level verdict flag alone):
  admission passed, complete evidence, no missing demo, verdict
  ``no_quality_gain`` re-derived from its own metric rows;
* train24 (``teacher_production_train24_v1``) is RECOMPUTED from the locked
  train split and its canonical demo-array SHA-256 must equal the declared
  one; val31/val6 come from the A1 frozen lists; no train/val overlap;
* the 55 NPZ/PT inputs are hashed; val31 entries must match the A1-frozen
  hashes (train18 are new records), and the A1 asset closure re-hashes
  clean;
* the registration environment is MEASURED (main-project interpreter,
  torch/mujoco/numpy, uv.lock); production-env fields the published dataset
  does not carry are recorded ``not_published`` (W1 plan §1.2).

The registration is written EXCLUSIVELY to ``<out>/<run_id>/freeze/
preregistration.json``: when the file exists it is only ever verified and
reused (exit 0) or refused with reasons (exit non-zero) — never overwritten
(§2.2). Costs append to ``costs.json`` per invocation.

Run: uv run --locked python scripts/w1_preregister.py \
       --out-root derived/experiments/omniretarget_first \
       [--run-id 20261011_<rev>_w1_a3bc]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data import a2_integrity as integ
from ireset.data import w1_training as w1
from ireset.data.pairing import subject_of
from ireset.eval.a2_closure import verify_plan_anchors
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record


def measure_repo_identity() -> dict:
    """Main-repo identity: rev, dirty state, untracked hashes, diff hash."""
    import hashlib as _hashlib

    def run(*a: str) -> str:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *a],
                              capture_output=True, text=True, check=True).stdout

    status = run("status", "--porcelain", "-uall")
    untracked = sorted(line[3:] for line in status.splitlines()
                       if line.startswith("??"))
    untracked_hashes = {
        rel: (integ.sha256_file(PROJECT_ROOT / rel)
              if (PROJECT_ROOT / rel).is_file() else "missing")
        for rel in untracked}
    tracked_diff = run("diff", "HEAD")
    return {
        "git_rev": run("rev-parse", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "status_lines": status.strip().splitlines()[:20],
        "untracked_files": untracked_hashes,
        "tracked_diff_sha256": _hashlib.sha256(tracked_diff.encode()).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff.encode()),
    }


def measure_env() -> dict:
    """Measured identity of the MAIN training environment (§1.2/§9).

    The published dataset ships no production-env record — those fields are
    explicitly ``not_published``; this round's own environment is measured,
    never assumed from dependency tables.
    """
    import mujoco
    import torch

    main_lock = PROJECT_ROOT / "uv.lock"
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "main_uv_lock_sha256": (integ.sha256_file(main_lock)
                                if main_lock.is_file() else None),
        "cuda_available": bool(torch.cuda.is_available()),
        "published_production_env": w1.NOT_PUBLISHED,
    }


def verify_c2_decision(c2_path: Path) -> dict:
    """The registered C2 terminal record must stay intact and unchanged."""
    record = json.loads(c2_path.read_text())
    reg = record.get("registration_status", {})
    problems = []
    if not reg.get("complete"):
        problems.append("c2_registration_not_complete")
    if record.get("baseline", {}).get("verdict") != "incomplete":
        problems.append("c2_baseline_verdict_changed")
    if record.get("recovery", {}).get("status") != "no_quality_gain":
        problems.append("c2_recovery_status_changed")
    if record.get("training", {}).get("status") != "not_ready":
        problems.append("c2_training_status_changed")
    return {"problems": problems, "passed": not problems,
            "facts": {"baseline_verdict": record.get("baseline", {}).get("verdict"),
                      "recovery_status": record.get("recovery", {}).get("status"),
                      "training_status": record.get("training", {}).get("status"),
                      "registration_complete": reg.get("complete")}}


def verify_ur_report(ur_path: Path, prereg_path: Path) -> dict:
    """UR entry evidence, re-derived — never the top-level flags alone."""
    from ireset.eval.teacher_comparison import reverify_verdict_record

    report = json.loads(ur_path.read_text())
    problems = []
    rever = reverify_verdict_record(report, u_side="UR")
    problems += [f"verdict:{p}" for p in rever["problems"]]
    admission = report.get("admission", {})
    if not admission.get("passed"):
        problems.append("admission_not_passed")
    missing = list(admission.get("ur_coverage", {}).get("missing", []))
    if missing:
        problems.append(f"missing_demos:{missing}")
    if report.get("verdict") != "no_quality_gain":
        problems.append(f"verdict_not_no_quality_gain:{report.get('verdict')}")
    if not report.get("verdict_record", {}).get("complete_evidence"):
        problems.append("evidence_not_complete")
    ur_identity = report.get("ur_identity", {})
    prereg_sha = integ.sha256_file(prereg_path)
    if ur_identity.get("preregistration_sha256") != prereg_sha:
        problems.append("report_not_bound_to_preregistration")
    return {"problems": problems, "passed": not problems,
            "verdict_reverification": rever,
            "facts": {"verdict": report.get("verdict"),
                      "admission_passed": bool(admission.get("passed")),
                      "complete_evidence": bool(
                          report.get("verdict_record", {}).get("complete_evidence")),
                      "missing": missing}}


def collect_inputs(demos: list[str], a1_lists: dict) -> tuple[dict, list[str]]:
    """Full hashes of the actual NPZ/PT inputs; val31 legs must equal the
    A1-frozen hashes (train18 entries are new records this round)."""
    inputs: dict[str, dict] = {}
    frozen = {e["demo"]: e for e in a1_lists["val_originals_v1"]}
    problems: list[str] = []
    for demo in demos:
        npz = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
        pt = OMOMO_NEW_DIR / f"{demo}.pt"
        rec = {"npz": integ.sha256_file(npz), "pt": integ.sha256_file(pt)}
        inputs[demo] = rec
        if demo in frozen:
            if rec["npz"] != frozen[demo].get("npz_sha256"):
                problems.append(f"npz_drifted_vs_a1:{demo}")
            if rec["pt"] != frozen[demo].get("pt_sha256"):
                problems.append(f"pt_drifted_vs_a1:{demo}")
    return inputs, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", type=Path,
                    default=PROJECT_ROOT / "derived" / "experiments" / "omniretarget_first")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--plan", type=Path,
                    default=PROJECT_ROOT / w1.W1_PLAN_DOC)
    args = ap.parse_args()
    t0 = time.time()

    out_root = integ.guard_output_dir(args.out_root)
    identity = measure_repo_identity()
    run_id = args.run_id or (
        f"{_dt.date.today().strftime('%Y%m%d')}_{identity['git_rev'][:7]}_w1_a3bc")
    run_dir = out_root / run_id
    freeze_dir = run_dir / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    prereg_path = freeze_dir / "preregistration.json"

    if prereg_path.is_file():
        # exclusive allocation: reuse only after a clean re-verification
        prereg = json.loads(prereg_path.read_text())
        check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                          TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
        print(f"[w0] registration exists: {path_for_record(prereg_path)}")
        print(f"[w0] re-verification: {'PASS' if check['passed'] else check['problems']}")
        return 0 if check["passed"] else 2

    # ---- anchors (re-hashed, not existence-checked) ----------------------
    paths = w1.anchor_paths(PROJECT_ROOT)
    anchors = verify_plan_anchors(
        paths, {k: sha for k, (_, sha) in w1.W1_PLAN_ANCHORS.items()},
        integ.sha256_file)
    if not anchors["passed"]:
        print(f"[w0] plan anchors failed: "
              f"{[k for k, r in anchors['anchors'].items() if not r['matches']]}",
              file=sys.stderr)
        return 2

    c2_check = verify_c2_decision(paths["c2_final_decision"])
    ur_check = verify_ur_report(paths["ur_full_comparison"],
                                paths["ur_preregistration"])
    if not c2_check["passed"] or not ur_check["passed"]:
        print(f"[w0] entry evidence failed: c2={c2_check['problems']} "
              f"ur={ur_check['problems']}", file=sys.stderr)
        return 2

    # ---- lists ------------------------------------------------------------
    a1_lists_path = paths["a1_frozen_lists"]
    a1_lists = json.loads(a1_lists_path.read_text())
    train_demos = json.loads(
        (PROJECT_ROOT / "derived/data/split/train.json").read_text())["demos"]
    train24 = w1.select_train24(train_demos)
    canonical_sha = w1.canonical_list_sha256(train24)
    val31 = [e["demo"] for e in a1_lists[w1.VAL31_LIST_ID]]
    val6 = [e["demo"] for e in a1_lists[w1.VAL6_LIST_ID]]
    list_problems: list[str] = []
    if canonical_sha != w1.W1_PLAN_TRAIN24_SHA:
        list_problems.append(f"train24_canonical_sha:{canonical_sha}")
    by_subj: dict[str, int] = {}
    for d in train24:
        by_subj[subject_of(d)] = by_subj.get(subject_of(d), 0) + 1
    if sorted(by_subj) != ["sub10", "sub12", "sub16", "sub3", "sub7", "sub8"] \
            or any(not (1 <= v <= w1.TRAIN24_PER_SUBJECT) for v in by_subj.values()):
        list_problems.append(f"train24_subject_coverage:{by_subj}")
    pilot6 = [e["demo"] for e in a1_lists["teacher_pilot_train6_v1"]]
    if not set(pilot6) <= set(train24):
        list_problems.append("pilot6_not_subset_of_train24")
    if sorted(set(val6)) != sorted(val6) or not set(val6) <= set(val31):
        list_problems.append("val6_not_subset_of_val31")
    overlap = sorted(set(train24) & set(val31))
    if overlap:
        list_problems.append(f"train24_val31_overlap:{overlap}")
    if list_problems:
        print(f"[w0] list problems: {list_problems}", file=sys.stderr)
        return 2

    # ---- inputs + assets ---------------------------------------------------
    all_demos = sorted(set(train24) | set(val31))
    inputs, input_problems = collect_inputs(all_demos, a1_lists)
    a1_assets_path = paths["a1_assets"]
    assets_check = integ.verify_assets_against_freeze(
        json.loads(a1_assets_path.read_text()), root=PROJECT_ROOT)
    if input_problems or not assets_check["passed"]:
        print(f"[w0] inputs={input_problems} assets mismatched="
              f"{assets_check['mismatched']} missing={assets_check['missing']}",
              file=sys.stderr)
        return 2

    env = measure_env()
    plan_rel = str(args.plan.relative_to(PROJECT_ROOT)
                   if args.plan.is_relative_to(PROJECT_ROOT) else args.plan)
    prereg = {
        "stage": "W0 run preregistration (W1 plan §2.2/§8)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "identity": {
            "experiment_id": w1.EXPERIMENT_ID,
            "protocol": w1.MOTION_PROTOCOL_VERSION,
            "training_purpose": w1.TRAINING_PURPOSE,
            "source_type": w1.SOURCE_TYPE,
            "published_revision": w1.PUBLISHED_REVISION,
            "run_id": run_id,
        },
        "repo_identity_at_registration": identity,
        "environment_measured": env,
        "plan": {"path": plan_rel, "sha256": integ.sha256_file(args.plan)},
        "anchors": {
            key: {"path": rel, "sha256": sha,
                  "measured_sha256": anchors["anchors"][key]["measured_sha256"]}
            for key, (rel, sha) in w1.W1_PLAN_ANCHORS.items()
        },
        "entry_evidence": {
            "c2_final_decision": c2_check["facts"],
            "ur_full_comparison": ur_check["facts"],
            "note": "U0 incomplete / UR no_quality_gain are NOT reopened by "
                    "this round; the registration only records that they "
                    "still hold on disk",
        },
        "selection": {
            "train24_list_id": w1.TRAIN24_LIST_ID,
            "train24_selection_rule": (
                f'per subject, first {w1.TRAIN24_PER_SUBJECT} by SHA256('
                f'"{w1.TRAIN24_HASH_PREFIX}" + demo) over train250 '
                "originals; identity-only, quality-blind"),
            "train24": train24,
            "train24_canonical_sha256": canonical_sha,
            "val31_list_id": w1.VAL31_LIST_ID,
            "val31": val31,
            "val6_list_id": w1.VAL6_LIST_ID,
            "val6": val6,
            "pilot6_from_a1": pilot6,
        },
        "inputs": inputs,
        "assets": {
            "a1_freeze_path": str(
                Path(w1.W1_PLAN_ANCHORS["a1_frozen_lists"][0]).parent),
            "a1_assets_path": str(w1.W1_PLAN_ANCHORS["a1_assets"][0]),
            "a1_assets_sha256": w1.W1_PLAN_ANCHORS["a1_assets"][1],
            "closure_verification": assets_check,
        },
        "labels_protocol": {
            "config": "configs/labels/omniretarget_motion_v1.yaml",
            "purpose": w1.TRAINING_PURPOSE,
            "w_contact": 0.0,
            "teacher_rewrite": "forbidden",
        },
        "budgets_and_stops": {
            "w0_a3_budget_s": w1.W0A3_BUDGET_S,
            "cache_budget_s": w1.CACHE_BUDGET_S,
            "s0_budget_s": w1.S0_BUDGET_S,
            "s0_max_steps_per_model": w1.S0_MAX_STEPS,
            "s0_min_steps_per_model": w1.S0_MIN_STEPS,
            "stop_conditions": [
                "A3: unprocessed/timeout rows keep the manifest open "
                "(incomplete), never a shrunken denominator",
                "S0: non-finite forward/backward/decode stops immediately "
                "(fit_not_demonstrated on trend failure; incomplete on "
                "missing evidence)",
                "no parameter/loss/data substitutions to obtain a pass; "
                "fixes land as new attempts with registered reasons",
            ],
        },
    }
    prereg_path.write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

    # costs accumulate per invocation at the run root (§9 layout)
    costs_path = run_dir / "costs.json"
    entries = []
    if costs_path.is_file():
        prev = json.loads(costs_path.read_text())
        entries = prev.get("invocations", [])
    entries.append({"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "stage": "w0_preregister", "wall_s": round(time.time() - t0, 1)})
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))

    print(f"[w0] train24={len(train24)} val31={len(val31)} val6={len(val6)} "
          f"inputs_hashed={len(inputs)} assets_checked={assets_check.get('n_checked')}")
    print(f"[w0] registration -> {path_for_record(prereg_path)}")
    print("[w0] NOTE: registration complete is bookkeeping, NOT a training "
          "authorization; U0 incomplete / UR no_quality_gain / training "
          "not_ready all stand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
