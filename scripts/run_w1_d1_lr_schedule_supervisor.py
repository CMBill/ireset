"""D1-only supervisor policy adapter plus independent actual-return observer.

The audited legacy supervisor is loaded as a private module, not modified.
Every comparison remains a candidate until this outer observer returns zero.
One fixed root/attempt is allowed for R′ and B independently; no retries.
"""

import time

_T0 = time.perf_counter()  # before uv, imports, reservation or subprocess

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = "w1_d1_lr_half_after_500_v1"


def load_core():
    spec = importlib.util.spec_from_file_location(
        "_lr_private_supervisor", ROOT / "scripts/run_w1_smallset_fit_real_supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WORKER = ROOT / "scripts/run_w1_d1_lr_schedule.py"
    mod.CHAIN_CAP = 2400.0
    mod.STAGE_CAPS = {"d1": 1800.0, "f": 600.0}
    old_sources = mod._source_hashes
    def sources():
        return {**old_sources(), **{str(p.relative_to(ROOT)): mod._sha(p) for p in (
            Path(__file__).resolve(),
            ROOT / "docs/plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md",
            ROOT / "tests/test_w1_d1_lr_schedule.py")}}
    mod._source_hashes = sources
    old_r = mod._r_summary_valid
    old_candidate = mod._candidate_problems

    def r_valid(doc):
        import copy
        projected = copy.deepcopy(doc)
        package = projected.get("package") or {}
        if (package.get("schema") != "w1_d1_lr_admission_package_v1"
                or package.get("cohort_id") != COHORT
                or package.get("schedule", {}).get("id") != "d1_half_after_500_v1"):
            return False
        package["schema"] = "w1_smallset_fit_admission_package_v1"
        return old_r(projected)

    def candidate(doc, launch_id, t0, mode, expected_cache_root=None):
        errors = old_candidate(doc, launch_id, t0, mode, expected_cache_root)
        if mode == "chain":
            try:
                if doc.get("authorized_includes") != ["D1", "F"] or doc.get("d6_ran") is not False:
                    errors.append("lr_d6_forbidden")
                binding = doc["comparison"]
                path = Path(doc["_output_dir"]) / "f/comparison.json"
                f = json.loads((path.parent / "f_report.json").read_bytes())
                cmp = json.loads(path.read_bytes())
                ledger = json.loads((path.parent.parent / "costs.json").read_bytes())
                if (binding != f.get("comparison") or binding["sha256"] != mod._sha(path)
                        or binding["bytes"] != path.stat().st_size
                        or cmp.get("state") != "pending_outer_return"
                        or cmp.get("conclusion") == "not_comparable"
                        or cmp.get("fit_verdict") != f.get("verdict")
                        or cmp.get("metrics") is None or cmp.get("problems") != []
                        or cmp.get("baseline_sha256") != doc.get("baseline_sha256")
                        or doc.get("stage_ledger", {}).get("sha256") != mod._sha(path.parent.parent / "costs.json")
                        or [row.get("stage") for row in ledger.get("rows", [])] != ["D1", "F"]
                        or any(row.get("status") != "sealed_candidate" for row in ledger.get("rows", []))):
                    errors.append("lr_comparison_binding_invalid")
            except (OSError, ValueError, TypeError, KeyError):
                errors.append("lr_comparison_unreadable")
        return errors

    # _candidate_problems needs the reserved output directory, supplied by
    # the proof reader (not a self-reported report field).
    old_proof = mod._proof_problems

    def proof(path, *args):
        problems, doc, observation, hashes = old_proof(path, *args)
        if path.name == "acceptance_summary.json":
            doc["_output_dir"] = str(path.parent)
        return problems, doc, observation, hashes

    mod._proof_problems = proof
    mod._r_summary_valid = r_valid
    mod._candidate_problems = candidate
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("r-admit", "chain"), default="chain")
    ap.add_argument("--output-root")
    ap.add_argument("--admission")
    ap.add_argument("--authorization")
    ap.add_argument("--cache-root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--inner-t0-hex", help=argparse.SUPPRESS)
    args = ap.parse_args()
    mod = load_core()
    if args.output_root is None:
        args.output_root = str(ROOT / ("derived/experiments/omniretarget_first/" + COHORT
                                      if args.mode == "chain" else "derived/audit/" + COHORT + "/r_admission"))
    args.fixture, args.impl_sha = args.synthetic, None
    args.wall_cap_s = 2400.0 if args.mode == "chain" else 120.0
    args.reserve_s = 60.0 if args.mode == "chain" else 15.0
    if args.synthetic and args.mode != "chain":
        ap.error("synthetic acceptance uses chain mode")
    if args.inner_t0_hex:
        if args.synthetic:
            args.admission = args.admission or "synthetic_worker_materials"
            args.authorization = args.authorization or "synthetic_worker_materials"
        # The synthetic worker publishes the actual fixture SHA bindings.
        if args.synthetic:
            # Use real fixture paths in the post-wait SHA check via Path args
            # resolved after the worker writes them, not placeholder digests.
            args.admission = str(Path(args.output_root) / "attempt_001/synthetic_materials/fixture_admission.json")
            args.authorization = str(Path(args.output_root) / "attempt_001/synthetic_materials/fixture_authorization.json")
        return mod.run(args, t0=float.fromhex(args.inner_t0_hex))
    output = Path(args.output_root).resolve()
    expected = ROOT / ("derived/experiments/omniretarget_first/" + COHORT
                       if args.mode == "chain" else "derived/audit/" + COHORT + "/r_admission")
    if not args.synthetic and output != expected:
        ap.error("fixed cohort root required; changing directories does not renew quota")
    if args.mode == "chain" and not args.synthetic and (not args.admission or not args.authorization):
        ap.error("NEW admission and explicit authorization required")
    output.mkdir(parents=True, exist_ok=True)
    # O_EXCL reservation plus nonempty-root refusal closes parallel launches,
    # failed/crashed/unknown histories and all attempt-renaming retry paths.
    if any(output.iterdir()):
        ap.error("once-only attempt consumed or history unknown")
    claim = output / "cohort_launch_claim.json"
    with claim.open("x") as out:
        json.dump({"cohort": COHORT, "mode": args.mode, "state": "running",
                   "t0_perf_counter": _T0, "synthetic": args.synthetic,
                   "max_attempts": 1,
                   "parent_consumption": {
                       "cohort": "w1_patched_smallset_fit_v1",
                       "run": "derived/experiments/omniretarget_first/attempt_002",
                       "authorization_consumed": True,
                       "old_incomplete_not_reclassified": True}}, out)
        out.flush()
        os.fsync(out.fileno())
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode,
               "--output-root", str(output), "--inner-t0-hex", _T0.hex()]
    for flag, value in (("--admission", args.admission), ("--authorization", args.authorization),
                        ("--cache-root", args.cache_root)):
        if value:
            command += [flag, str(Path(value).resolve())]
    if args.synthetic:
        command.append("--synthetic")
    with (output / "supervisor_stdout.log").open("xb") as stdout, (output / "supervisor_stderr.log").open("xb") as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            rc = proc.wait(timeout=max(0.01, _T0 + args.wall_cap_s - time.perf_counter()))
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
        returned = time.perf_counter()
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    attempts = list(output.glob("attempt_*"))
    if len(attempts) != 1:
        return 3
    attempt = attempts[0]
    try:
        artifact_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if artifact_bytes > 2 * (1 << 30):
            return 3
        verdict = json.loads((attempt / "acceptance_verdict.json").read_bytes())
        registered = verdict["registered_measured_s"]
        if (rc != 0 or verdict["verdict"] != "candidate_complete" or verdict["problems"] != []
                or type(registered) not in (int, float) or not math.isfinite(registered)
                or not returned - _T0 <= registered <= args.wall_cap_s):
            return 3
        record = {"schema": "w1_d1_lr_independent_return_v1", "supervisor_return_code": rc,
                  "t0_perf_counter": _T0, "supervisor_return_perf_counter": returned,
                  "elapsed_after_actual_wait_s": returned - _T0,
                  "registered_measured_s": registered, "wall_cap_s": args.wall_cap_s,
                  "comparison_candidate": args.mode == "chain",
                  "state": "requires_observer_actual_return_zero",
                  "artifacts_bytes_at_return": artifact_bytes,
                  "bindings": {str(p.relative_to(attempt)): mod._sha(p) for p in (
                       attempt / ("acceptance_summary.json" if args.mode == "chain" else "r_admission.json"),
                       attempt / "acceptance_verdict.json",
                      attempt / "outer_return_observation.json", attempt / "outer_stdout_observation.json",
                      *([attempt / "f/comparison.json", attempt / "f/f_report.json"] if args.mode == "chain" else []))}}
        path = output / "completion_verification.json"
        mod._write_json(path, record)
        after_record = time.perf_counter() - _T0
        mod._write_json(output / "observer_write_observation.json", {
            "completion_sha256": mod._sha(path), "elapsed_after_completion_write_s": after_record,
            "registered_measured_s": registered, "wall_cap_s": args.wall_cap_s})
        # No output or I/O after the passing final sample. Exit=0 is required
        # together with these immutable bindings; files alone cannot self-certify.
        return 0 if time.perf_counter() - _T0 <= registered <= args.wall_cap_s else 3
    except (OSError, ValueError, KeyError, TypeError):
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
