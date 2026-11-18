"""Metadata R′ and explicitly authorized D1′/F′ for the LR diagnostic.

R′ is a separate reviewable delivery. Neither admission nor this module's
existence authorizes real training. The legacy CLI cannot consume this schema.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import torch

from ireset.eval import w1_smallset_fit_diagnostic as fit
from ireset.eval import w1_smallset_fit_r_admission as radm
from ireset.eval import w1_d1_lr_schedule as lr
from ireset.eval import w1_d1_lr_protocol as protocol
from ireset.paths import PROJECT_ROOT

PACKAGE_SCHEMA = "w1_d1_lr_admission_package_v1"
AUTH_SCHEMA = "w1_d1_lr_authorization_v2"
CAPS = {"R": [120.0, 15.0], "D1": [1800.0, 60.0], "F": [600.0, 60.0]}
CHECKLIST = (*radm.AUTHORIZATION_CHECKLIST, "baseline_prefix_schedule_and_comparison_reviewed")
EXTRA_FILES = (lr.PLAN_PATH, "src/ireset/eval/w1_d1_lr_schedule.py",
               "src/ireset/eval/w1_d1_lr_workflow.py",
               "src/ireset/eval/w1_d1_lr_protocol.py",
               "scripts/run_w1_d1_lr_schedule.py",
               "scripts/run_w1_d1_lr_schedule_supervisor.py",
               "tests/test_w1_d1_lr_schedule.py",
               "tests/test_w1_d1_lr_review_regressions.py")
# The only old executable files allowed to differ from baseline A. Data,
# config, model, losses, tokenizer, FK and Trainer bytes must stay identical.
DIFF_WHITELIST = {"src/ireset/eval/w1_smallset_fit_diagnostic.py",
                  "src/ireset/eval/w1_smallset_fit_r_admission.py"}


def implementation_identity(root=PROJECT_ROOT):
    files = {rel: radm.sha256_file(Path(root) / rel)
             for rel in (*radm.R_IMPLEMENTATION_FILES, *EXTRA_FILES)}
    return {"implementation_files": files,
            "implementation_files_sha256": radm.canonical_sha256({"files": files})}


def frozen_matrix(plan: lr.SchedulePlan | None = None, baseline: dict | None = None) -> dict:
    p = plan or lr.SchedulePlan("D1")
    n = p.n_probes + 1
    d = {**fit.expected_call_matrix(p, 1), **radm._world_construction_counts(),
         "registered_window_reads": 1, "window_x0_encodes": 1,
         "model_constructions": 1, "conditioner_constructions": 1,
         "optimizer_constructions": 1, "checkpoint_loads": 0,
         "loss_production_decode_calls": 2 * p.steps * p.grad_accum,
         "saved_production_decode_calls": n, "saved_metric_decode_calls": n,
         "checkpoint_inspection_reads": 2, "reference_model_constructions": 1,
         "reference_conditioner_constructions": 1,
         "reference_contract_cache_hits": 1,
         "prefix_file_comparisons": p.switch_after // p.probe_every + 3,
         "prefix_admission_file_hashes": p.switch_after // p.probe_every + 3,
         "history_rows_compared": p.switch_after, "schedule_switches": 1,
         "optimizer_state_digest_reads": 3 * p.steps + 2,
         "cpu_evidence_baseline_rechecks": 2,
         "cpu_evidence_schedule_rows_verified": p.steps,
         "cpu_evidence_prefix_file_comparisons": p.switch_after // p.probe_every + 3,
         "cpu_evidence_history_rows_compared": p.switch_after,
         "lr_group_reads": 1 + p.n_probes + 2 + p.steps * (5 + p.grad_accum)}
    f = {**radm._world_construction_counts(), "registered_window_reads": 0,
         "window_x0_encodes": 0, "model_constructions": 1,
         "conditioner_constructions": 1, "optimizer_constructions": 0,
         "checkpoint_inspection_reads": 4, "checkpoint_loads": 2,
         "strict_module_loads": 4, "fresh_probe_calls": 2,
         "conditioner_forward_calls": 2, "backbone_forward_calls": 2,
         "production_fk_batch_calls": 2, "production_decode_calls": 2,
         "metric_decode_calls": 4, "saved_production_decode_calls": n,
         "saved_metric_decode_calls": n, "reference_model_constructions": 1,
         "reference_conditioner_constructions": 1, "reference_contract_cache_hits": 1,
         "loss_calls": 0, "backward_calls": 0, "clip_calls": 0,
         "adamw_step_calls": 0, "paired_probe_rows": p.n_probes,
         "prefix_file_comparisons": 2 * (p.switch_after // p.probe_every + 3),
         "history_rows_compared": 2 * p.switch_after}
    f["cpu_evidence_baseline_rechecks"] = 2
    f["cpu_evidence_schedule_rows_verified"] = 2 * p.steps
    # Decode classes are distinct: the top-level D1 counters exclude the
    # 8000 production decodes inside loss; the saved and fresh domains do not.
    doc = {"schema": "w1_d1_lr_call_matrix_v1", "D1": d, "D6": {}, "F": f,
           "R": {"model_constructions": 0, "forward_calls": 0,
                 "array_value_reads": 0, "checkpoint_loads": 0},
           "bounds": {"D6_all_calls": 0, "conditioner_forward_calls": d["conditioner_forward_calls"] + 2,
                      "backbone_forward_calls": d["backbone_forward_calls"] + 2,
                      "production_fk_batch_calls": d["production_fk_batch_calls"] + 2},
           "semantics": {"reference_constructions_and_hits": "independent upper bounds; process cache-dependent",
                         "g1_spec_constructions": "upper bound; scene cache-dependent",
                         "lr_group_reads": "per actual group; roster fixed at construction",
                         "baseline_byte_reads": "all bound files before materialization and comparison; no arrays",
                         "epochs_restarted": "driver fact, not a numerical call",
                         "scene": "one access per apparatus; compile/cache-hit recorded"}}
    if baseline is not None:
        total_bytes = sum(r["bytes"] for r in baseline["files"].values())
        doc["baseline_io_caps"] = {
            "file_count_per_recheck": len(baseline["files"]),
            "bytes_per_recheck": total_bytes,
            # One materialization recheck plus two driver/comparator scans per
            # phase; the initial R freeze hashes each artifact twice at most.
            "R_file_hashes": 3 * len(baseline["files"]),
            "D1_file_hashes": 3 * len(baseline["files"]),
            "F_file_hashes": 3 * len(baseline["files"]),
            "R_bytes_hashed": 3 * total_bytes,
            "D1_bytes_hashed": 3 * total_bytes, "F_bytes_hashed": 3 * total_bytes,
            "R_array_decodes": 0, "D1_A_array_decodes": 0, "F_A_array_decodes": 0}
    doc["sha256"] = radm.canonical_sha256(doc)
    return doc


def authorization_draft(package: dict, admission_sha: str, *, r_return=None) -> dict:
    return {"schema": AUTH_SCHEMA, "authorized": False, "template_only": True,
            "cohort_id": lr.COHORT, "includes": ["D1", "F"],
            "checklist": {k: None for k in CHECKLIST},
            "admission_package_sha256": admission_sha,
            "r_observer_actual_return": r_return,
            **{k: package[k] for k in ("plan_sha256", "baseline_sha256", "rules_sha256")},
            "implementation_files_sha256": package["code_identity"]["implementation_files_sha256"],
            "call_matrix_sha256": package["call_matrix"]["sha256"],
            "budget_caps": CAPS, "max_attempts_per_stage": 1}


def authorization_problems(doc, admission, admission_path, *, root=PROJECT_ROOT, admission_sha=None):
    if not isinstance(doc, dict) or doc.get("schema") != AUTH_SCHEMA:
        return ["lr_authorization_schema_invalid"]
    p = admission["package"]
    expected = authorization_draft(p, admission_sha or radm.sha256_file(Path(admission_path)))
    if p.get("fixture") is not True:
        try:
            expected["r_observer_actual_return"] = verify_admission_return(
                admission_path, doc.get("r_observer_actual_return"), root=root,
                expected_admission_sha=expected["admission_package_sha256"])
        except fit.FitDiagnosticError as exc:
            return [str(exc)]
    expected.update(authorized=True, template_only=False,
                    checklist={k: True for k in CHECKLIST})
    problems = ["lr_authorization_mismatch:" + k for k, v in expected.items()
                if not fit._json_value_equal(doc.get(k), v)]
    if implementation_identity(root)["implementation_files_sha256"] != expected["implementation_files_sha256"]:
        problems.append("lr_implementation_changed")
    return problems


def run_admission(*, out_dir, budget, outer_launch=None, fixture=False,
                  baseline_root=None, plan=None, **kwargs):
    """Wrap the audited metadata legs; extension executes before the R seal."""
    root = Path(kwargs.get("project_root") or PROJECT_ROOT)
    p = plan or lr.SchedulePlan("D1")
    if (budget.stage != "R" or [budget.wall_cap_s, budget.reserve_s] != CAPS["R"]
            or (not fixture and (root != PROJECT_ROOT or plan is not None and plan.synthetic))):
        raise fit.FitDiagnosticError("lr_r_budget_or_scope_invalid")
    if fixture and not protocol.fixture_output_allowed(out_dir, PROJECT_ROOT):
        raise fit.FitDiagnosticError("lr_fixture_output_not_isolated")
    if not fixture and not kwargs.get("cache_root"):
        budget.check("lr_frozen_A_cache_binding")
        a = lr.read_object(PROJECT_ROOT / lr.BASELINE_ROOT / "acceptance_summary.json")
        a_admission = lr.read_object(PROJECT_ROOT / a["admission"]["path"])
        kwargs["cache_root"] = PROJECT_ROOT / a_admission["package"]["cache_binding"]["path"]

    def extend(package, billing):
        baseline = lr.freeze_baseline(Path(baseline_root or PROJECT_ROOT / lr.BASELINE_ROOT),
                                      p, fixture=fixture, check=billing.check)
        impl = implementation_identity(root)
        differences = []
        if not fixture:
            a_summary = lr.read_object(PROJECT_ROOT / lr.BASELINE_ROOT / "acceptance_summary.json")
            old = lr.read_object(PROJECT_ROOT / a_summary["admission"]["path"])["package"]
            for rel, sha in old["code_identity"]["implementation_files"].items():
                if impl["implementation_files"].get(rel) != sha:
                    differences.append(rel)
            if set(differences) - DIFF_WHITELIST:
                raise fit.FitDiagnosticError(f"baseline_unapproved_code_diff:{differences}")
            for key in ("cache_binding", "root_stats", "source_binding"):
                if not fit._json_value_equal(package[key], old[key]):
                    raise fit.FitDiagnosticError(f"baseline_source_binding_mismatch:{key}")
            if package["d1_window_id"] != "sub10_largebox_097:0:60":
                raise fit.FitDiagnosticError("lr_window_mismatch")
        package.update(schema=PACKAGE_SCHEMA, cohort_id=lr.COHORT,
                       baseline=baseline, baseline_sha256=baseline["sha256"],
                       schedule=p.schedule(), fit_plan=p.as_dict(), rules=lr.RULES,
                       rules_sha256=radm.canonical_sha256(lr.RULES),
                       plan_sha256=radm.sha256_file(root / lr.PLAN_PATH),
                       call_matrix=frozen_matrix(p, baseline), fixture=fixture,
                       budget={"caps": {k: {"wall_cap_s": v[0], "reserve_s": v[1]}
                                        for k, v in CAPS.items()},
                               "chain_cap_s": 2400.0, "total_with_r_s": 2520.0},
                       code_identity={**package["code_identity"], **impl},
                       differences_from_A={"allowed": sorted(DIFF_WHITELIST),
                                           "actual": differences},
                       authorization_template={"schema": AUTH_SCHEMA, "authorized": False,
                                                "includes": ["D1", "F"],
                                                "checklist": {k: None for k in CHECKLIST}})
        return []

    return radm.run_r_admission(out_dir=out_dir, budget=budget,
                                outer_launch=outer_launch, extend_package=extend,
                                cohort_id=lr.COHORT, **kwargs)


def load_admission(path, *, doc=None):
    doc = lr.read_object(path) if doc is None else doc
    package, seal = doc.get("package"), doc.get("seal")
    if (doc.get("admission_passed") is not True or doc.get("problems") != []
            or not isinstance(seal, dict) or seal.get("status") != "sealed_within_budget"
            or not isinstance(package, dict) or type(package.get("fixture")) is not bool
            or (not package["fixture"] and seal.get("terminal") is not True)
            or package.get("schema") != PACKAGE_SCHEMA or package.get("cohort_id") != lr.COHORT):
        raise fit.FitDiagnosticError("lr_admission_not_consumable")
    return doc


def verify_admission_return(path, binding=None, *, root=PROJECT_ROOT, expected_admission_sha=None):
    """A completed inner supervisor's files cannot prove the observer exited."""
    path = Path(path).resolve()
    expected = Path(root).resolve() / protocol.R_ROOT / "attempt_001/r_admission.json"
    if path != expected:
        raise fit.FitDiagnosticError("lr_r_fixed_admission_required")
    try:
        protocol.verify_r_actual_return(path.parent, binding, expected_admission_sha=expected_admission_sha)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise fit.FitDiagnosticError(f"lr_admission_actual_return_missing_or_invalid:{exc}") from exc
    return dict(binding)


def recheck(admission, **kwargs):
    """Reuse SOURCE/cache/runtime legs without weakening their old contract.

    The old evaluator's constant execution envelope is checked on an explicit
    projection; the versioned envelope is independently checked below.
    """
    p = admission["package"]
    root = Path(kwargs.get("project_root") or PROJECT_ROOT)
    problems = []
    current = implementation_identity(root)
    plan = lr.SchedulePlan(**p["fit_plan"])
    for k, want in {"schema": PACKAGE_SCHEMA, "cohort_id": lr.COHORT,
                    "call_matrix": frozen_matrix(plan, p["baseline"]), "schedule": plan.schedule(),
                    "rules": lr.RULES, "rules_sha256": radm.canonical_sha256(lr.RULES),
                    "plan_sha256": radm.sha256_file(root / lr.PLAN_PATH)}.items():
        if not fit._json_value_equal(p.get(k), want):
            problems.append(f"lr_admission_drift:{k}")
    if any(not fit._json_value_equal(p["code_identity"].get(k), v) for k, v in current.items()):
        problems.append("lr_admission_implementation_drift")
    if p.get("budget") != {"caps": {k: {"wall_cap_s": v[0], "reserve_s": v[1]} for k, v in CAPS.items()},
                            "chain_cap_s": 2400.0, "total_with_r_s": 2520.0}:
        problems.append("lr_admission_budget_drift")
    problems += lr.baseline_problems(p["baseline"])
    if p.get("baseline_sha256") != p["baseline"]["sha256"]:
        problems.append("lr_admission_baseline_mismatch")
    legacy = copy.deepcopy(admission)
    legacy["package"]["budget"]["caps"] = {
        k: {"wall_cap_s": cap, "reserve_s": reserve} for k, (cap, reserve) in fit.BUDGET_CAPS.items()}
    legacy["package"]["call_matrix"] = radm.frozen_call_matrix()
    legacy["package"]["code_identity"]["implementation_files_sha256"] = radm.implementation_identity(root)["implementation_files_sha256"]
    kwargs.pop("impl_sha", None)
    problems += radm.admission_recheck_problems(legacy, **kwargs)
    return problems


def require_unused_root(root: Path, *, fixture: bool, mode="chain"):
    expected = PROJECT_ROOT / (lr.RUN_ROOT if mode == "chain" else lr.AUDIT_ROOT + "/r_admission")
    if not fixture and root.resolve() != expected.resolve():
        raise fit.FitDiagnosticError("lr_fixed_cohort_root_required")
    if fixture and not protocol.fixture_output_allowed(root, PROJECT_ROOT):
        raise fit.FitDiagnosticError("lr_fixture_output_not_isolated")
    if root.exists() and any(root.iterdir()):
        raise fit.FitDiagnosticError("lr_once_only_or_history_unknown")


def verify_launch_scope(run_root: Path, outer_launch: dict, *, mode="chain"):
    """Real workers cannot bypass the fixed-root once-only supervisor claim."""
    expected = PROJECT_ROOT / (lr.RUN_ROOT if mode == "chain" else lr.AUDIT_ROOT + "/r_admission")
    run_root = run_root.resolve()
    if run_root != expected / "attempt_001":
        raise fit.FitDiagnosticError("lr_fixed_attempt_required")
    claim = lr.read_object(expected / "cohort_launch_claim.json")
    launched = lr.read_object(run_root / "outer_launch.json")
    if (claim.get("mode") != mode or claim.get("state") != "running"
            or claim.get("cohort") != lr.COHORT
            or claim.get("synthetic") is not False
            or claim.get("t0_perf_counter") != outer_launch.get("t0_perf_counter")
            or launched.get("launch_id") != outer_launch.get("launch_id")
            or launched.get("t0_perf_counter") != outer_launch.get("t0_perf_counter")
            or launched.get("worker_process_group") != os.getpgrp()
            or outer_launch.get("worker_process_group") != os.getpgrp()
            or outer_launch.get("supervisor_declared") is not True
            or {p.name for p in expected.glob("attempt_*")} != {"attempt_001"}):
        raise fit.FitDiagnosticError("lr_launch_scope_or_history_invalid")


class StageLedger:
    """One durable row per stage; a partial/crashed attempt is never zero."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    def _write(self):
        lr.write_json(self.path, {"schema": protocol.LEDGER_SCHEMA, "rows": self.rows})

    def begin(self, budget):
        if any(row["stage"] == budget.stage for row in self.rows):
            raise fit.FitDiagnosticError("lr_stage_already_started")
        self.rows.append({"stage": budget.stage, "status": "running", "measured_wall_s": None,
                          "t0_perf_counter": budget.t0, "wall_cap_s": budget.wall_cap_s})
        self._write()

    def finish(self, budget, report_path, withdraw):
        """One coverage loop for ledger, report and proof, including withdrawal."""
        report, report_sha = protocol._object_and_sha(report_path)
        seal = report["seal"]
        registered = original = seal["measured_wall_s_at_summary"]
        if (not protocol.number(registered) or seal.get("terminal") is not True
                or seal.get("t0_perf_counter") != budget.t0 or seal.get("wall_cap_s") != budget.wall_cap_s):
            raise fit.FitDiagnosticError("lr_ledger_invalid_stage_seal")
        proof_path = Path(report_path).parent / "return_observation.json"
        row = next(r for r in self.rows if r["stage"] == budget.stage)
        complete = "local_complete_evidence" if budget.stage == "D1" else "complete_evidence"
        rejected = seal.get("status") != "sealed_within_budget" or original > budget.wall_cap_s
        reserve = 4.0

        def publish(elapsed, *, unproven=False):
            nonlocal registered, report_sha
            if rejected:
                registered = max(registered, elapsed + (0 if unproven else reserve))
                seal.update(status="budget_exceeded_after_seal", measured_wall_s_at_summary=registered,
                            seal_tail_reserve_s=registered - elapsed, return_coverage_elapsed_s=elapsed,
                            return_coverage_registered_s=original, cost_coverage_unproven=unproven,
                            note="lr_stage_ledger_return_withdrawn")
                report["budget"]["measured_wall_s_at_summary"] = registered
                report.update(withdraw)
                lr.write_json(report_path, report)
                report_sha = protocol.sha(report_path)
            row.update(status=("cost_unproven" if unproven else "sealed_candidate"
                               if not rejected and report.get(complete) is True else "incomplete"),
                       measured_wall_s=registered, report_seal=seal["status"])
            self._write()
            observed = budget.clock() - budget.t0
            lr.write_json(proof_path, {"schema": fit.RETURN_OBSERVATION_SCHEMA, "covered": not rejected,
                                       "summary_sha256": report_sha, "registered_measured_s": registered,
                                       "wall_cap_s": budget.wall_cap_s, "elapsed_s": observed,
                                       "cost_coverage_unproven": unproven, "stage_ledger_row": dict(row)})
            return observed

        for _ in range(4):
            elapsed = budget.clock() - budget.t0
            rejected = rejected or elapsed > registered or elapsed > budget.wall_cap_s
            observed = publish(elapsed)
            end = budget.clock()  # stage closure covers the final proof write itself
            if end - budget.t0 <= registered:
                return {"covered": not rejected, "final_doc": report, "end": end, "elapsed_s": observed}
            rejected = True
            reserve = max(4.0, 2 * (end - budget.t0 - elapsed) + 3.0)
        observed = publish(budget.clock() - budget.t0, unproven=True)
        return {"covered": False, "final_doc": report, "end": budget.clock(), "elapsed_s": observed}

    def failed(self, budget, reason):
        row = next((r for r in self.rows if r["stage"] == budget.stage), None)
        if row is not None:
            row.update(status="crashed", cost_coverage_unproven=True,
                       measured_wall_s_lower_bound=budget.clock() - budget.t0, reason=reason)
            self._write()


def _matrix_problems(driver, report, f_doc=None, f_cpu=None):
    want = frozen_matrix(driver.plan)
    problems = []
    if driver.lr_counts["schedule_switches"] != 1:
        problems.append("schedule_switch_count")
    for key in ("history_rows_compared", "prefix_file_comparisons", "prefix_admission_file_hashes", "lr_group_reads",
                "optimizer_state_digest_reads"):
        multiplier = len(driver.trainer.opt.param_groups) if key == "lr_group_reads" else 1
        if driver.lr_counts[key] != want["D1"][key] * multiplier:
            problems.append("D1:" + key)
    if driver.lr_counts.get("loss_production_decode_calls", 0) != want["D1"]["loss_production_decode_calls"]:
        problems.append("D1:loss_production_decode_calls")
    for key in ("production_decode_calls", "metric_decode_calls"):
        if report.get("verification_counters", {}).get(key) != driver.plan.n_probes + 1:
            problems.append("D1:saved_" + key)
    for key, cap in (("checkpoint_inspection_reads", 2), ("reference_model_constructions", 1),
                     ("reference_conditioner_constructions", 1), ("reference_contract_cache_hits", 1)):
        got = report.get("verification_counters", {}).get(key, 0)
        if got > cap or (key == "checkpoint_inspection_reads" and got != cap):
            problems.append("D1:" + key)
    for key in ("baseline_rechecks", "schedule_rows_verified", "prefix_file_comparisons", "history_rows_compared"):
        if driver.cpu_evidence_counts.get(key, 0) != want["D1"]["cpu_evidence_" + key]:
            problems.append("D1:cpu_evidence_" + key)
    if f_doc is not None:
        stages = f_doc.get("stages") or []
        if len(stages) != 1:
            problems.append("F:stage_count")
        else:
            fresh = stages[0].get("fresh_counters", {})
            for key in ("conditioner_forward_calls", "backbone_forward_calls", "production_fk_batch_calls",
                        "production_decode_calls", "metric_decode_calls", "checkpoint_loads", "strict_module_loads"):
                if fresh.get(key) != want["F"][key]:
                    problems.append("F:" + key)
            for key in ("production_decode_calls", "metric_decode_calls"):
                if stages[0].get("verification_counters", {}).get(key) != driver.plan.n_probes + 1:
                    problems.append("F:saved_" + key)
            if stages[0].get("verification_counters", {}).get("checkpoint_inspection_reads") != 4:
                problems.append("F:checkpoint_inspection_reads")
        if f_cpu is not None:
            for key in ("baseline_rechecks", "schedule_rows_verified", "paired_probe_rows",
                        "prefix_file_comparisons", "history_rows_compared"):
                target = want["F"].get("cpu_evidence_" + key, want["F"].get(key))
                if f_cpu.get(key, 0) != target:
                    problems.append("F:cpu_evidence_" + key)
    return problems


def run_candidate(*, run_root, admission_path, authorization_path, chain_t0,
                  outer_launch, fixture=False, fixture_world=None, **world_kwargs):
    """One candidate; every refused/failed path remains incomplete, never retried."""
    root = Path(world_kwargs.get("project_root") or PROJECT_ROOT)
    run_root = Path(run_root)
    if fixture:
        if not protocol.fixture_output_allowed(run_root, PROJECT_ROOT):
            raise fit.FitDiagnosticError("lr_fixture_output_not_isolated")
    else:
        verify_launch_scope(run_root, outer_launch)
        if world_kwargs or fixture_world is not None:
            raise fit.FitDiagnosticError("real_world_override_forbidden")
    chain = fit.StageBudget("chain", 2400.0, 60.0, t0=chain_t0)
    d1_budget = fit.StageBudget("D1", 1800.0, 60.0, t0=chain_t0)
    facts = {"kind": "lr_schedule_chain", "cohort_id": lr.COHORT, "fixture": fixture,
             "authorized_includes": ["D1", "F"], "d6_ran": False,
             "d6": {"skipped_reason": "d6_not_authorized"},
             "d1_handoff": {"handoff_ready": False}, "outer_launch": outer_launch,
             "label_verdict": "pending_outer_return"}
    facts["parent_consumption"] = {"baseline_root": lr.BASELINE_ROOT,
                                   "baseline_authorization_consumed": True,
                                   "baseline_updates": 1000, "baseline_probes": 101,
                                   "legacy_incomplete_unchanged": True,
                                   "T6_mandatory_gaps": 539}
    problems = []
    ledger = StageLedger(run_root / "costs.json")
    active_budget = d1_budget
    try:
        ledger.begin(d1_budget)
        d1_budget.check("lr_entry")
        admission_doc, admission_sha = protocol._object_and_sha(admission_path)
        auth, auth_sha = protocol._object_and_sha(authorization_path)
        facts.update(admission={"path": str(Path(admission_path).resolve()), "sha256": admission_sha},
                     authorization={"path": str(Path(authorization_path).resolve()), "sha256": auth_sha})
        admission = load_admission(admission_path, doc=admission_doc)
        p = admission["package"]
        if p["fixture"] is not fixture:
            raise fit.FitDiagnosticError("lr_fixture_mode_mismatch")
        errors = authorization_problems(auth, admission, admission_path, root=root, admission_sha=admission_sha)
        if errors:
            raise fit.FitDiagnosticError(f"lr_authorization_refused:{errors}")
        # Persistent, exclusive claim precedes array/model work. The supervisor
        # also reserves the fixed root, so parallel/changed-path invocations fail.
        with (run_root / "authorization_consumed.json").open("x") as out:
            json.dump({"authorization_sha256": facts["authorization"]["sha256"],
                       "admission_sha256": facts["admission"]["sha256"],
                       "cohort": lr.COHORT, "state": "consumed", "t0": chain_t0}, out)
            out.flush()
            os.fsync(out.fileno())
        plan = lr.SchedulePlan(**p["fit_plan"])
        if not fixture and plan.synthetic:
            raise fit.FitDiagnosticError("synthetic_plan_on_real_path")
        if not fixture and shutil.disk_usage(root).free < radm.R_REQUIRED_DISK_FREE_BYTES:
            raise fit.FitDiagnosticError("lr_disk_margin")
        guard = radm.ProductionRuntimeGuardShim(fixture)
        if fixture_world is None:
            world = radm.materialize_real_world(
                admission=admission, stage="D1", budget=d1_budget,
                device="cpu" if fixture else "cuda", fixture=fixture,
                binding_validator=recheck, **world_kwargs)
        else:
            if not fixture:
                raise fit.FitDiagnosticError("real_world_injection_refused")
            world = fixture_world
        view = world.view_for("D1")
        identity = {"module": __name__, "cohort_id": lr.COHORT,
                    "admission_package_sha256": facts["admission"]["sha256"],
                    "implementation_files_sha256": p["code_identity"]["implementation_files_sha256"]}
        d1_dir = run_root / "d1"
        counts = {}

        def constructed(unit):
            counts[unit + "_constructions"] = counts.get(unit + "_constructions", 0) + 1
            d1_budget.check("constructed_" + unit)

        build_args = {"construction_observer": constructed} if fixture_world is None else {}
        d1_budget.check("lr_d1_pre_trainer")
        guard.check_unit("lr_d1_pre_trainer")
        d1_budget.check("lr_d1_pre_trainer_post_runtime")
        trainer = world.build_trainer(plan, d1_dir, budget_deadline=d1_budget.work_deadline,
                                      train_ds=view["ds"], **build_args)
        driver = lr.ScheduleDriver(
            baseline=p["baseline"], trainer=trainer, plan=plan,
            registration=view["registration"], loss_fn=world.loss_fn,
            decoder=world.decoder, probe_batch=view["probe_batch"],
            budget=d1_budget, runtime_guard=guard, identity=identity,
            stage_dir=d1_dir, training_gen=torch.Generator().manual_seed(plan.seed),
            dataset_window_keys=view["ds"].window_keys)
        summary = driver.run()
        summary["materialization"] = getattr(world, "facts", {"fixture": True})
        summary["construction_counts"] = counts
        if fixture_world is None and counts != {"model_constructions": 1,
                                                "conditioner_constructions": 1,
                                                "optimizer_constructions": 1}:
            summary.update(local_complete_evidence=False, local_mean_fit_passed=None,
                           local_conclusion="incomplete")
            summary["call_matrix"]["problems"].append("D1:construction_counts")
        if summary["local_complete_evidence"]:
            matrix_problems = _matrix_problems(driver, summary)
            if matrix_problems:
                summary.update(local_complete_evidence=False, local_mean_fit_passed=None,
                               local_conclusion="incomplete")
                summary["call_matrix"]["problems"] += matrix_problems
        fit.finish_stage(d1_dir, summary, d1_budget)
        returned = time.perf_counter() - d1_budget.t0
        cov = ledger.finish(d1_budget, d1_dir / "stage_report.json", fit.I_WITHDRAW_ON_EXCEED)
        f_start = cov["end"]
        doc = cov["final_doc"]
        facts["d1"] = {k: doc.get(k) for k in ("local_complete_evidence", "local_mean_fit_passed", "local_conclusion")}
        facts["d1"].update(return_coverage=cov["covered"], stage_returned_elapsed_s=returned,
                             stage_closed_perf_counter=cov["end"],
                            call_matrix_problems=doc.get("call_matrix", {}).get("problems"),
                            seal=doc.get("seal", {}).get("status"))
        f_budget = fit.StageBudget("F", 600.0, 60.0, t0=f_start)
        active_budget = f_budget
        ledger.begin(f_budget)
        f_specs = []
        f_cpu_counts = {}
        if facts["d1"]["local_complete_evidence"] is True and cov["covered"]:
            apparatus = (radm.materialize_fit_apparatus(
                admission=admission, budget=f_budget,
                device="cpu" if fixture else "cuda", fixture=fixture,
                binding_validator=recheck, **world_kwargs)
                if fixture_world is None else None)
            f_specs.append({"stage_dir": d1_dir, "plan": plan, "registration": view["registration"],
                            "decoder": apparatus.decoder if apparatus else world.decoder,
                            "loss_fn": apparatus.loss_fn if apparatus else world.loss_fn,
                            "identity": driver.identity,
                            "verify_extension": lambda budget: lr.verify_schedule_stage(
                                d1_dir, plan, p["baseline"], budget, counters=f_cpu_counts)})
        build_counts = {}

        def build():
            # Same constructors/order as the audited factory, individually
            # guarded. No optimizer and no model work in the comparator.
            from ireset.model.deterministic import DeterministicG1
            from ireset.model.conditioner import ConditionEncoder
            f_budget.check("lr_f_model")
            model = DeterministicG1(**fit._backbone_kwargs(plan)).to(trainer.cfg.device)
            build_counts["model_constructions"] = 1
            f_budget.check("lr_f_conditioner")
            conditioner = ConditionEncoder(plan.d_model).to(trainer.cfg.device)
            build_counts["conditioner_constructions"] = 1
            f_budget.check("lr_f_constructed")
            return model, conditioner

        def comparison(final):
            comparison_doc = lr.build_comparison(stage_dir=d1_dir, baseline=p["baseline"],
                                                  plan=plan, f_candidate=final, budget=f_budget,
                                                  counters=f_cpu_counts)
            matrix_errors = _matrix_problems(driver, summary, final, f_cpu_counts) if final.get("complete_evidence") else []
            if matrix_errors:
                comparison_doc.update(lr.classify({}, {}, matrix_errors))
                final.update(complete_evidence=False, mean_fit_passed=None, verdict="incomplete")
            if comparison_doc["conclusion"] == "not_comparable":
                for key in ("paired_probes", "prefix", "tail", "final_spikes"):
                    comparison_doc[key] = None
                final.update(complete_evidence=False, mean_fit_passed=None, verdict="incomplete")
            path = run_root / "f/comparison.json"
            lr.write_json(path, comparison_doc)
            final["comparison"] = {"path": "comparison.json", **lr.file_record(path),
                                   "conclusion": comparison_doc["conclusion"]}
            final["construction_counts"] = build_counts
            final["cpu_evidence_counts"] = dict(f_cpu_counts)
            final["apparatus"] = apparatus.facts() if f_specs and apparatus else {"fixture": True}

        fit.run_f_verification(stage_specs=f_specs, out_dir=run_root / "f",
                                       budget=f_budget, runtime_guard=guard, build_modules=build,
                                       authorization={"includes": ["D1", "F"]}, finalize_evidence=comparison)
        returned = time.perf_counter() - f_budget.t0
        f_cov = ledger.finish(f_budget, run_root / "f/f_report.json", fit.F_WITHDRAW_ON_EXCEED)
        f_doc = f_cov["final_doc"]
        facts["f"] = {k: f_doc.get(k) for k in ("complete_evidence", "replay_passed", "mean_fit_passed", "verdict", "stages", "comparison")}
        facts["f"].update(return_coverage=f_cov["covered"], stage_returned_elapsed_s=returned,
                          stage_closed_perf_counter=f_cov["end"])
        facts["comparison"] = f_doc.get("comparison")
        facts["baseline_sha256"] = p["baseline_sha256"]
        facts["schedule"] = plan.schedule()
    except (fit.FitDiagnosticError, OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"lr_chain_incomplete:{type(exc).__name__}:{str(exc)[:500]}")
        ledger.failed(active_budget, str(exc)[:500])
    facts["stage_ledger"] = {"rows": ledger.rows,
                              "sha256": radm.sha256_file(ledger.path) if ledger.path.is_file() else None}
    size = sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file())
    facts["artifacts_bytes_before_chain_seal"] = size
    if size > radm.R_ARTIFACT_CAP_BYTES:
        problems.append("lr_artifact_cap_exceeded")
    facts["problems"] = problems
    facts["budget"] = chain.facts()
    facts["chain_budget"] = dict(facts["budget"])
    return fit.write_acceptance_outputs(run_root, facts, chain,
                                        withdraw={"label_verdict": "budget_exceeded_after_seal"})


def synthetic_materials(out_dir: Path, *, steps=4, switch_after=2):
    """Offline A/B control, labelled fixture; cannot be used by the real entry."""
    if not protocol.fixture_output_allowed(out_dir, PROJECT_ROOT):
        raise fit.FitDiagnosticError("lr_fixture_output_not_isolated")
    torch.set_num_threads(1)
    parent = Path(out_dir) / "synthetic_materials"
    world, constant, _, summary, stage_dir, _, _ = fit._mini_stage(
        parent / "A", "D1", n_windows=1, steps=steps, probe_every=1,
        d_model=16, lr=0.002, finish=True)
    if summary["local_complete_evidence"] is not True:
        raise fit.FitDiagnosticError("synthetic_A_incomplete")
    plan = lr.SchedulePlan(**constant.as_dict(), switch_after=switch_after)
    baseline = lr.freeze_baseline(stage_dir.parent, plan, fixture=True)
    package = {"schema": PACKAGE_SCHEMA, "cohort_id": lr.COHORT,
               "fixture": True, "fit_plan": plan.as_dict(),
               "baseline": baseline, "baseline_sha256": baseline["sha256"],
               "plan_sha256": radm.sha256_file(PROJECT_ROOT / lr.PLAN_PATH),
               "rules_sha256": radm.canonical_sha256(lr.RULES),
               "schedule": plan.schedule(), "rules": lr.RULES,
               "call_matrix": frozen_matrix(plan, baseline),
               "code_identity": implementation_identity(),
               "budget": {"caps": {k: {"wall_cap_s": v[0], "reserve_s": v[1]} for k, v in CAPS.items()},
                          "chain_cap_s": 2400.0, "total_with_r_s": 2520.0}}
    admission_path = parent / "fixture_admission.json"
    lr.write_json(admission_path, {"admission_passed": True, "problems": [],
                                  "seal": {"status": "sealed_within_budget"},
                                  "package": package})
    auth = authorization_draft(package, radm.sha256_file(admission_path))
    auth.update(authorized=True, template_only=False, checklist={k: True for k in CHECKLIST},
                synthetic_only=True)
    auth_path = parent / "fixture_authorization.json"
    lr.write_json(auth_path, auth)
    return world, plan, baseline, admission_path, auth_path
