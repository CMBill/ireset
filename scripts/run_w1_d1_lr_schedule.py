"""Versioned LR diagnostic worker. Real training requires a NEW signed package.

--fixture is pure CPU synthetic acceptance, not a production authorization.
R′ metadata admission and real B are never implicitly started by acceptance.
"""

import time

_T0 = time.perf_counter()

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ireset.eval import w1_d1_lr_schedule as lr
from ireset.eval import w1_d1_lr_protocol as protocol
from ireset.eval import w1_d1_lr_artifacts as artifacts
from ireset.eval import w1_d1_lr_workflow as flow
from ireset.eval import w1_smallset_fit_diagnostic as fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("r-admit", "chain"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--authorization")
    parser.add_argument("--cache-root")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--outer-t0-hex", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--wall-cap-s", type=float, required=True)
    parser.add_argument("--reserve-s", type=float, required=True)
    args = parser.parse_args()
    t0 = float.fromhex(args.outer_t0_hex)
    cap, reserve = (120.0, 15.0) if args.mode == "r-admit" else (2400.0, 60.0)
    if (not math.isfinite(t0) or not 0 < t0 <= _T0
            or args.wall_cap_s != cap or args.reserve_s != reserve):
        parser.error("frozen clock/cap/reserve mismatch")
    output = Path(args.output_dir).resolve()
    if args.fixture:
        if not protocol.fixture_output_allowed(output, flow.PROJECT_ROOT):
            parser.error("synthetic output must be isolated from reserved real and historical roots")
    else:
        expected = Path(flow.PROJECT_ROOT) / (lr.RUN_ROOT if args.mode == "chain"
                                             else lr.AUDIT_ROOT + "/r_admission")
        if output.parent != expected:
            parser.error("fixed cohort root required")
    launch = {"launch_id": args.launch_id, "t0_perf_counter": t0,
              "cli_entry_perf_counter": _T0, "supervisor_declared": True,
              "worker_process_group": os.getpgrp()}
    budget = fit.StageBudget("R" if args.mode == "r-admit" else "chain", cap, reserve, t0=t0)
    try:
        record = lr.read_object(output / "outer_launch.json")
        if record.get("worker_process_group") != os.getpgrp():
            raise fit.FitDiagnosticError("lr_worker_not_in_supervised_process_group")
        if not args.fixture:
            flow.verify_launch_scope(output, launch, mode=args.mode)
        if args.mode == "r-admit":
            if args.fixture:
                parser.error("fixture acceptance uses chain mode only")
            out = flow.run_admission(out_dir=output, budget=budget,
                                     outer_launch=launch,
                                     cache_root=Path(args.cache_root) if args.cache_root else None)
            # This is only a template. The collector makes the unsigned draft
            # in the R root after it has observed the R observer return.
            doc = lr.read_object(Path(out["summary"]))
            if doc["admission_passed"]:
                lr.write_json(output / "authorization_draft.json", flow.authorization_draft(
                    doc["package"], flow.radm.sha256_file(Path(out["summary"]))))
        else:
            claim = lr.read_object(output.parent / "cohort_launch_claim.json")
            if (claim.get("observer_pid") != os.getpgrp()
                    or claim.get("artifact_space") != artifacts.policy()):
                raise fit.FitDiagnosticError("lr_worker_missing_supervised_artifact_policy")
            space = artifacts.ArtifactSpace(output.parent, supervised=True)
            budget.artifact_space = space
            if args.fixture:
                # Fixture generation is charged from the supervisor's t0.
                world, _, _, admission, authorization = flow.synthetic_materials(output, space=space)
            else:
                if not args.admission or not args.authorization or args.cache_root:
                    parser.error("chain requires admission/authorization, no cache override")
                world, admission, authorization = None, Path(args.admission), Path(args.authorization)
            out = flow.run_candidate(run_root=output, admission_path=admission,
                                      authorization_path=authorization, chain_t0=t0,
                                      outer_launch=launch, fixture=args.fixture, fixture_world=world,
                                      supervised=True)
        path = Path(out["summary"])
        doc = lr.read_object(path)
        print(json.dumps({"state": "pending_outer_return", "output_dir": str(output),
                          "mode": args.mode, "problems": doc.get("problems"),
                          "comparison": doc.get("comparison")}), flush=True)
        cov = fit.verify_return_coverage(path, budget,
                                         withdraw={"label_verdict": "budget_exceeded_after_seal"},
                                         **budget.closure_write_options())
        if not cov["covered"]:
            return 3
        doc = cov["final_doc"]
        if args.mode == "r-admit":
            return 0 if doc.get("admission_passed") is True else 1
        return 0 if (not doc.get("problems") and (doc.get("f") or {}).get("complete_evidence") is True
                     and (doc.get("f") or {}).get("replay_passed") is True) else 1
    except (fit.FitDiagnosticError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"LR diagnostic refused: {exc}", file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
