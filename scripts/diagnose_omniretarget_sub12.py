"""D0: sub12 limited fault diagnosis (closure plan §5) + G0 registration.

Parent runs in the MAIN environment: it verifies the inputs against the C0
baseline evidence, launches the wrapped-generation child inside the
independent upstream uv environment (argument list, never a shell line),
saves FULL stdout/stderr, and compares:

* control: ``sub16_largebox_000`` wrapped default generation must produce
  arrays bit-equal to the registered raw artifact — otherwise the wrapping
  is not equivalent and NO sub12 attribution is allowed (§5.3 step 1);
* two sub12 runs from frame 0 with the original default initialization:
  the ACTUAL failing ``frame_idx`` and in-frame solver ordinal are captured
  by the child (the historical 171/347 progress-bar number is never parsed
  back out of a log tail);
* Phase I (computed inside the failing child, on the original problem):
  rho = min ||dqa||2 over the original constraints minus the single
  step-size SOC; the §5.5 table decides whether the single 0.3 radius
  candidate may be registered (G0).

Subcommand ``register-recovery`` (G0, §6.2): writes the exclusive-create
``preregistration.json`` for ``a2_recovery_step03_v1`` — allowed ONLY when
the referenced diagnosis summary has status ``supports_step03_candidate``
and no UR products exist without prior registration.

Run: uv run --locked python scripts/diagnose_omniretarget_sub12.py \
       --baseline-evidence <run>/baseline/baseline_evidence.json \
       --out <run>/diagnosis
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from _omniretarget_diagnostic_child import classify_phase_one
from ireset.data import a2_integrity as integ
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, path_for_record

HOLOSOMA_PACKAGE = (
    PROJECT_ROOT / "third_party" / "holosoma" / "src" / "holosoma_retargeting"
    / "holosoma_retargeting"
)
DEFAULT_ENV_PYTHON = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"
CHILD_DRIVER = PROJECT_ROOT / "scripts" / "_omniretarget_diagnostic_child.py"
RETARGETER_FILE = HOLOSOMA_PACKAGE / "src" / "interaction_mesh_retargeter.py"

#: §5.5 two-run consistency: identical key arrays + rho agreement
RHO_AGREEMENT = 1e-6


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _recorded_path(recorded: str | Path) -> Path:
    """evidence_files records repo-relative paths inside the project and
    absolute paths outside; undo exactly that."""
    p = Path(recorded).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _hash_or_none(path: Path) -> str | None:
    return integ.sha256_file(path) if Path(path).is_file() else None


def compare_control(control_raw: Path, registered_raw: Path,
                    registered_sha: str | None) -> dict:
    """§5.3 step 1: wrapped generation must reproduce the registered raw."""
    rec: dict = {"control_raw": path_for_record(control_raw),
                 "registered_raw": path_for_record(registered_raw)}
    if registered_sha is not None:
        rec["registered_sha_matches"] = (
            _hash_or_none(registered_raw) == registered_sha)
        if not rec["registered_sha_matches"]:
            rec.update(equal=False, reason="registered_raw_drifted")
            return rec
    if not control_raw.is_file():
        rec.update(equal=False, reason="control_raw_missing")
        return rec
    a = np.load(control_raw)
    b = np.load(registered_raw)
    try:
        equal = (np.array_equal(a["qpos"], b["qpos"])
                 and np.array_equal(a["human_joints"], b["human_joints"])
                 and float(a["fps"]) == float(b["fps"]))
    except KeyError as exc:
        rec.update(equal=False, reason=f"missing_key:{exc}")
        return rec
    rec.update(equal=bool(equal))
    return rec


def _arrays_equal(npz_a: Path, npz_b: Path, keys: list[str]) -> dict:
    out: dict = {}
    if not (npz_a.is_file() and npz_b.is_file()):
        return {"all_equal": False, "missing": True}
    a, b = np.load(npz_a), np.load(npz_b)
    for k in keys:
        if k not in a or k not in b:
            out[k] = "missing"
        else:
            out[k] = bool(np.array_equal(a[k], b[k]))
    return {"all_equal": all(v is True for v in out.values()), "per_key": out}


def decide_diagnosis(control: dict, runs: list[dict]) -> dict:
    """Aggregate the §5.5 verdict. Pure; counterexample-testable."""
    if not control.get("equal", False):
        return {"status": "control_not_equivalent",
                "reason": control.get("reason"),
                "note": "wrapping not proven equivalent; no sub12 "
                        "attribution allowed (§5.3 step 1)"}
    failed, problems = [], []
    for run in runs:
        s = run.get("summary") or {}
        if run.get("timed_out"):
            problems.append(f"{run['label']}:timeout")
            continue
        if run.get("launch_error"):
            problems.append(f"{run['label']}:launch_failed")
            continue
        if not s:
            problems.append(f"{run['label']}:no_child_summary")
            continue
        failure = s.get("failure")
        if not failure:
            if "instrumentation" in str(s.get("outcome")):
                problems.append(f"{run['label']}:instrumentation_not_ready")
            else:
                problems.append(f"{run['label']}:unexpected_success")
            continue
        failed.append(run)
    if problems:
        return {"status": "diagnostic_incomplete",
                "problems": problems,
                "note": "a run neither failed at the original solver nor "
                        "produced usable evidence (§11 stop)"}
    # both runs failed with captured context: determinism first
    f0, f1 = (r["summary"]["failure"] for r in failed)
    same_frame = f0.get("failing_frame_idx", f0.get("frame_idx")) == \
        f1.get("failing_frame_idx", f1.get("frame_idx"))
    same_ordinal = f0.get("ordinal_in_frame") == f1.get("ordinal_in_frame")
    ctx_equal = _arrays_equal(
        Path(failed[0]["snapshot"]) / "failure_context.npz",
        Path(failed[1]["snapshot"]) / "failure_context.npz",
        ["q_locked", "q_a_n_last", "q_t_last", "target_laplacian",
         "obj_pts_local"])
    p0 = f0.get("phase_one", {}) or {}
    p1 = f1.get("phase_one", {}) or {}
    cats = [p.get("category") or f"phase1_{p.get('status', 'missing')}"
            for p in (p0, p1)]
    rho0, rho1 = p0.get("rho"), p1.get("rho")
    rho_agree = (rho0 is not None and rho1 is not None
                 and abs(rho0 - rho1) <= RHO_AGREEMENT)
    rec: dict = {
        "failing_frame_run1": f0.get("failing_frame_idx", f0.get("frame_idx")),
        "failing_frame_run2": f1.get("failing_frame_idx", f1.get("frame_idx")),
        "ordinal_run1": f0.get("ordinal_in_frame"),
        "ordinal_run2": f1.get("ordinal_in_frame"),
        "same_frame": bool(same_frame), "same_ordinal": bool(same_ordinal),
        "key_arrays_equal": ctx_equal,
        "phase1_categories": cats,
        "rho_run1": rho0, "rho_run2": rho1, "rho_agreement": bool(rho_agree),
    }
    if not (same_frame and same_ordinal and ctx_equal.get("all_equal")):
        rec.update(status="nondeterministic",
                   note="two runs disagree on the failure context; no "
                        "recovery candidate (§5.3 step 3)")
        return rec
    if all(c == "supports_step03_candidate" for c in cats) and rho_agree:
        rec.update(status="supports_step03_candidate",
                   note="both failing linearizations need a step larger "
                        "than 0.2 and within 0.3; register the single "
                        "step03 candidate before reading any new labels")
        return rec
    if all(c == "supports_step03_candidate" for c in cats) and not rho_agree:
        rec.update(status="nondeterministic",
                   note="identical failure contexts but Phase I rho "
                        f"disagrees beyond {RHO_AGREEMENT}; no candidate "
                        "(§5.5 two-run consistency)")
        return rec
    # terminal non-candidate rows of the §5.5 table
    if "rho_exceeds_candidate_range" in cats:
        terminal = "rho_exceeds_candidate_range"
    elif "retained_infeasible" in cats:
        terminal = "retained_infeasible"
    elif "rho_within_original_radius" in cats:
        terminal = "rho_within_original_radius"
    else:
        terminal = "diagnostic_incomplete"
    rec.update(status=f"no_recovery_candidate:{terminal}",
               note="finite-recovery hypothesis not supported; U0 stop line "
                    "stands (§6.1)")
    return rec


def run_child(env_python: Path, demo: str, data_path: Path, save_dir: Path,
              snapshot_dir: Path, retargeter_sha: str,
              timeout_s: int) -> dict:
    """Launch the wrapped generation child; FULL logs; group-kill timeout.

    The child runs in its own session (``start_new_session``), so a timeout
    kills the WHOLE process group and the partially-captured stdout/stderr
    is still saved — no orphan solver survives into a later attempt.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(env_python), str(CHILD_DRIVER),
           "--mode", "generation", "--demo", demo,
           "--data-path", str(data_path),
           "--save-dir", str(save_dir),
           "--snapshot-dir", str(snapshot_dir),
           "--holosoma-package", str(HOLOSOMA_PACKAGE),
           "--retargeter-sha256", retargeter_sha]
    t0 = time.time()
    rec: dict = {"label": demo, "argv": cmd, "snapshot": str(snapshot_dir),
                 "save_dir": str(save_dir), "timeout_s": timeout_s}
    try:
        proc = subprocess.Popen(cmd, cwd=str(HOLOSOMA_PACKAGE),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
    except OSError as exc:
        rec.update(launch_error=f"{type(exc).__name__}: {exc}",
                   timed_out=False)
        return rec
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # child leads its group
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
    (snapshot_dir / "child.stdout.txt").write_text(out or "")
    (snapshot_dir / "child.stderr.txt").write_text(err or "")
    rec.update(exit_code=None if timed_out else proc.returncode,
               wall_s=round(time.time() - t0, 1), timed_out=timed_out)
    if timed_out:
        (snapshot_dir / "TIMEOUT.txt").write_text(
            f"timeout after {timeout_s}s; process group killed; partial "
            f"logs saved\n")
        return rec
    summary_path = snapshot_dir / "child_summary.json"
    rec["summary"] = _load_json(summary_path) if summary_path.is_file() else None
    return rec


def diagnose(args) -> int:
    t0 = time.time()
    out_root = integ.guard_output_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- baseline evidence + anchors -----------------------------------
    baseline_evidence_path = Path(args.baseline_evidence).expanduser().resolve()
    if not baseline_evidence_path.is_file():
        raise SystemExit(f"baseline evidence missing: {baseline_evidence_path}")
    baseline = _load_json(baseline_evidence_path)
    evidence_files_path = baseline_evidence_path.parent / "evidence_files.json"
    if not evidence_files_path.is_file():
        raise SystemExit(f"evidence inventory missing: {evidence_files_path}")
    ev = _load_json(evidence_files_path)
    selection_path = _recorded_path(ev["authoritative_selection"]["path"])
    selection_now = _hash_or_none(selection_path)
    selection_ok = (selection_now == ev["authoritative_selection"]["sha256"]
                    == baseline["plan_anchors"]["anchors"][
                        "authoritative_selection"]["measured_sha256"])
    freeze_dir = _recorded_path(
        ev["a1_freeze"]["frozen_lists.json"]["path"]).parent

    # ---- identity gates --------------------------------------------------
    checkout = integ.verify_checkout_against_freeze(
        _load_json(freeze_dir / "assets.json"))
    env = integ.measure_env_identity(args.env_python)
    env_ok = (env["env_mujoco_version"] is not None
              and env["env_mujoco_version"] == env["env_lock_mujoco_version"])

    blocked: list[str] = []
    if not selection_ok:
        blocked.append("authoritative_selection_drifted")
    if not checkout["passed"]:
        blocked.append("upstream_checkout_mismatch")
    if not env_ok:
        blocked.append(f"env_probe_failed:{env.get('env_probe_error')
                                             or env['env_mujoco_version']}")
    idx_common = {"purpose": "sub12 limited diagnosis",
                  "control_demo": args.control_demo, "demo": args.demo,
                  "baseline_evidence": path_for_record(baseline_evidence_path),
                  "baseline_evidence_sha256":
                      integ.sha256_file(baseline_evidence_path)}
    if blocked:
        # a blocked diagnosis still allocates its own attempt and writes an
        # explicit record (never a crash with no report on disk)
        attempt_dir = integ.allocate_attempt(out_root)
        summary = {"status": "diagnostic_incomplete", "blocked": blocked,
                   "timestamp": _dt.datetime.now().isoformat(timespec="seconds")}
        (attempt_dir / "diagnosis_run.json").write_text(json.dumps(
            summary, indent=2, ensure_ascii=False))
        integ.append_attempts_index(out_root, {
            **idx_common, "attempt": attempt_dir.name,
            "status": summary["status"],
            "report": path_for_record(attempt_dir / "diagnosis_run.json")})
        print(json.dumps(summary, indent=2))
        return 2

    retargeter_sha = integ.sha256_file(RETARGETER_FILE)
    data_path = Path(args.data_path).expanduser().resolve()

    # §9 structure: attempt_001 = control + diagnosis run 1;
    # attempt_002 = diagnosis run 2 (the concluding verdict lives there)
    att1 = integ.allocate_attempt(out_root)
    control = run_child(args.env_python, args.control_demo, data_path,
                        att1 / "control", att1 / "control", retargeter_sha,
                        args.run_timeout_s)
    run1 = run_child(args.env_python, args.demo, data_path,
                     att1 / "sub12_run1", att1 / "sub12_run1",
                     retargeter_sha, args.run_timeout_s)
    reg_raw_info = (ev.get("u0_evidence", {}).get("attempt_001", {})
                    .get(args.control_demo, {}).get("raw", {}))
    control_cmp = {"run": {k: v for k, v in control.items()
                           if k != "summary"}}
    if reg_raw_info:
        control_cmp["comparison"] = compare_control(
            att1 / "control" / f"{args.control_demo}_original.npz",
            _recorded_path(reg_raw_info["path"]), reg_raw_info.get("sha256"))
    else:
        control_cmp["comparison"] = {"equal": False,
                                     "reason": "no registered raw artifact"}

    att2 = integ.allocate_attempt(out_root)
    run2 = run_child(args.env_python, args.demo, data_path,
                     att2 / "sub12_run2", att2 / "sub12_run2",
                     retargeter_sha, args.run_timeout_s)

    verdict = decide_diagnosis(control_cmp["comparison"], [run1, run2])
    summary = {
        "stage": "D0 sub12 limited diagnosis (closure plan §5)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "attempts": {"control_and_run1": att1.name, "run2": att2.name},
        "baseline": {
            "baseline_evidence": path_for_record(baseline_evidence_path),
            "baseline_evidence_sha256":
                integ.sha256_file(baseline_evidence_path),
            "evidence_files": path_for_record(evidence_files_path),
            "authoritative_selection": path_for_record(selection_path),
            "authoritative_selection_sha256": selection_now,
            "authoritative_selection_still_pinned": selection_ok,
        },
        "identity": {"checkout": checkout, "env": env,
                     "retargeter_sha256": retargeter_sha,
                     "child_driver_sha256": integ.sha256_file(CHILD_DRIVER),
                     "data_path": path_for_record(data_path)},
        "control": control_cmp,
        "runs": [{k: v for k, v in r.items() if k != "summary"}
                 | {"failure": (r.get("summary") or {}).get("failure"),
                    "outcome": (r.get("summary") or {}).get("outcome"),
                    "frames_completed": (r.get("summary") or {})
                    .get("frames_completed")}
                 for r in (run1, run2)],
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    out_json = att2 / "diagnosis_run.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                   default=str))
    integ.append_attempts_index(out_root, {
        "attempt": att1.name, "purpose": "sub16 control + sub12 diagnosis 1",
        "control_equal": control_cmp["comparison"].get("equal"),
        "run1_outcome": (run1.get("summary") or {}).get("outcome"),
        "report": path_for_record(att1 / "sub12_run1" / "child_summary.json")})
    integ.append_attempts_index(out_root, {
        "attempt": att2.name, "purpose": "sub12 diagnosis 2",
        "status": verdict["status"],
        "report": path_for_record(out_json)})
    print(json.dumps(verdict, indent=2, ensure_ascii=False, default=str))
    print(f"diagnosis -> {path_for_record(out_json)}")
    definitive = not verdict["status"].startswith("diagnostic_incomplete")
    return 0 if definitive else 2


# ---- G0: recovery preregistration (§6.2) ----------------------------------

PREREG_FIELDS = {
    "experiment_id": "a2_recovery_step03_v1",
    "source_id": "upstream_recovery_step03_v1",
    "single_config_difference": {"retargeter.step_size": {"from": 0.2,
                                                          "to": 0.3}},
    "held_constant": [
        "holosoma checkout (bccd4d7 pinned by the A1 freeze)",
        "upstream environment (lock + mujoco 3.10.0 triple check)",
        "A1 frozen assets and pilot six, full lengths, 30 Hz",
        "initialization (reference-only, augmentation off)",
        "foot sticking / non-penetration / joint limits semantics",
        "object/human/fps/T binding acceptance",
        "g1-g5 gates and the §3.4 verdict rules of the parent plan",
    ],
    "candidate_count": 1,
    "retry_policy": ("one normal run per demo; at most one same-config "
                     "retry for engineering failures or new-failure "
                     "verification; reasons and both costs recorded"),
    "stop_conditions": [
        "missing evidence or budget exhaustion -> incomplete",
        "no improvement -> stop; no additional candidates",
    ],
}


def register_recovery(args) -> int:
    """G0: exclusive-create preregistration.json (§6.2/§9)."""
    diag_path = Path(args.from_diagnosis).expanduser().resolve()
    if not diag_path.is_file():
        raise SystemExit(f"diagnosis summary missing: {diag_path}")
    diag = _load_json(diag_path)
    if diag.get("verdict", {}).get("status") != "supports_step03_candidate":
        raise SystemExit(
            "refusing to register: diagnosis status is "
            f"{diag.get('verdict', {}).get('status')!r}; the single 0.3 "
            "candidate is registrable only from supports_step03_candidate")
    out_root = integ.guard_output_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    gen_root = out_root / "generation"
    if gen_root.exists() and any(gen_root.iterdir()):
        raise SystemExit(
            "refusing to register: UR products already exist without a "
            "prior registration (§6.2 no retroactive registration)")
    ev = _load_json(Path(args.baseline_evidence).expanduser().resolve().parent
                    / "evidence_files.json")
    freeze_dir = _recorded_path(
        ev["a1_freeze"]["frozen_lists.json"]["path"]).parent
    pilot = [e["demo"] for e in _load_json(
        freeze_dir / "frozen_lists.json")["teacher_pilot_train6_v1"]]
    prereg = {
        **PREREG_FIELDS,
        "registered_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "parent": {
            "authoritative_selection": ev["authoritative_selection"]["path"],
            "authoritative_selection_sha256":
                ev["authoritative_selection"]["sha256"],
            "diagnosis": path_for_record(diag_path),
            "diagnosis_sha256": integ.sha256_file(diag_path),
        },
        "expected_demos": pilot,
        "expected_denominator": len(pilot),
        "recovery_dir": path_for_record(out_root),
    }
    target = out_root / "preregistration.json"
    try:
        with open(target, "x") as f:
            f.write(json.dumps(prereg, indent=2, ensure_ascii=False))
    except FileExistsError:
        raise SystemExit(f"preregistration already exists: {target} "
                         "(never overwritten; write a NEW experiment)")
    integ.append_attempts_index(out_root, {
        "attempt": "preregistration",
        "experiment_id": prereg["experiment_id"],
        "sha256": integ.sha256_file(target),
        "path": path_for_record(target)})
    print(f"preregistration -> {path_for_record(target)}")
    print(f"sha256 = {integ.sha256_file(target)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="run the D0 limited diagnosis")
    d.add_argument("--baseline-evidence", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True,
                   help="diagnosis dir (attempt_NNN inside)")
    d.add_argument("--env-python", type=Path, default=DEFAULT_ENV_PYTHON)
    d.add_argument("--demo", default="sub12_largebox_082")
    d.add_argument("--control-demo", default="sub16_largebox_000")
    d.add_argument("--data-path", type=Path, default=OMOMO_NEW_DIR)
    d.add_argument("--run-timeout-s", type=int, default=900,
                   help="per-generation budget (§11: <=15 min)")

    r = sub.add_parser("register-recovery", help="G0 preregistration")
    r.add_argument("--from-diagnosis", type=Path, required=True,
                   help="diagnosis_run.json with the final verdict")
    r.add_argument("--baseline-evidence", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True,
                   help="recovery dir (preregistration.json inside)")

    args = ap.parse_args()
    if args.cmd == "diagnose":
        return diagnose(args)
    return register_recovery(args)


if __name__ == "__main__":
    raise SystemExit(main())
