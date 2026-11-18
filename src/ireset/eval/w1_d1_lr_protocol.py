"""Standard-library-only launch scope and return bindings for the LR diagnostic."""

import hashlib
import json
import math
from pathlib import Path

COHORT = "w1_d1_lr_half_after_500_v1"
RUN_ROOT = "derived/experiments/omniretarget_first/" + COHORT
AUDIT_ROOT = "derived/audit/" + COHORT
R_ROOT = AUDIT_ROOT + "/r_admission"
RETURN_SCHEMA = "w1_d1_lr_independent_return_v2"
R_RETURN_NAME = "r_observer_actual_return.json"
R_RETURN_SCHEMA = "w1_d1_lr_r_observer_actual_return_v1"
R_RETURN_FAILURE = "r_capture_failure.json"
LEDGER_SCHEMA = "w1_d1_lr_stage_costs_v2"
CAPS = {"r-admit": (120.0, 15.0), "chain": (2400.0, 60.0)}
STAGE_REPORTS = {"D1": ("d1/stage_report.json", 1800.0), "F": ("f/f_report.json", 600.0)}
COMPLETION_NAME = "completion_verification.json"
SAMPLE_NAME = "observer_write_observation.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object_and_sha(path):
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError(f"lr_return_not_object:{path}")
    return doc, hashlib.sha256(raw).hexdigest()


def fixture_output_allowed(output, root):
    """In-repo fixtures must stay in audit/synthetic; allow isolated scratch roots."""
    output, root = Path(output).resolve(), Path(root).resolve()
    for reserved in (root / RUN_ROOT, root / R_ROOT):
        reserved = reserved.resolve()
        if output.is_relative_to(reserved) or reserved.is_relative_to(output):
            return False
    synthetic = root / AUDIT_ROOT / "synthetic"
    return (not output.is_relative_to(root)
            or (output != synthetic and output.is_relative_to(synthetic)))


def attempt_bindings(mode):
    common = ("acceptance_verdict.json", "outer_return_observation.json",
              "outer_stdout_observation.json", "outer_launch.json", "return_observation.json")
    if mode == "r-admit":
        return ("r_admission.json", *common)
    return ("acceptance_summary.json", *common, "f/comparison.json", "f/f_report.json")


def number(value):
    try:
        return type(value) in (float, int) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def verify_observer_files(attempt, mode):
    """Verify bytes and clock closure, not the publishing observer's own exit."""
    attempt = Path(attempt)
    completion, completion_sha = _object_and_sha(attempt.parent / COMPLETION_NAME)
    sample, sample_sha = _object_and_sha(attempt.parent / SAMPLE_NAME)
    cap, reserve = CAPS[mode]
    reg = completion.get("registered_measured_s")
    wait = completion.get("elapsed_after_actual_wait_s")
    after = sample.get("elapsed_after_completion_write_s")
    t0 = completion.get("t0_perf_counter")
    returned = completion.get("supervisor_return_perf_counter")
    launch_id = completion.get("launch_id")
    if (completion.get("schema") != RETURN_SCHEMA or completion.get("mode") != mode
            or not isinstance(launch_id, str) or not launch_id
            or completion.get("state") != "requires_observer_actual_return_zero"
            or completion.get("comparison_candidate") is not (mode == "chain")
            or type(completion.get("supervisor_return_code")) is not int
            or completion["supervisor_return_code"] != 0
            or not all(number(v) for v in (reg, wait, after, t0, returned))
            or returned - t0 != wait or not wait <= after <= reg <= cap
            or completion.get("wall_cap_s") != cap
            or sample.get("registered_measured_s") != reg or sample.get("wall_cap_s") != cap
            or sample.get("completion_sha256") != completion_sha):
        raise ValueError("lr_observer_binding_invalid")
    bindings = completion.get("bindings")
    required = attempt_bindings(mode)
    if not isinstance(bindings, dict) or set(bindings) != set(required):
        raise ValueError("lr_observer_bindings_missing")
    for rel in required:
        if sha(attempt / rel) != bindings[rel]:
            raise ValueError(f"lr_observer_file_drift:{rel}")
    verdict, _ = _object_and_sha(attempt / "acceptance_verdict.json")
    receipt, _ = _object_and_sha(attempt / "outer_return_observation.json")
    stdout, _ = _object_and_sha(attempt / "outer_stdout_observation.json")
    launch, _ = _object_and_sha(attempt / "outer_launch.json")
    proof, _ = _object_and_sha(attempt / "return_observation.json")
    summary_name = required[0]
    summary, _ = _object_and_sha(attempt / summary_name)
    summary_launch = summary.get("outer_launch")
    seal = summary.get("seal")
    elapsed = (proof.get("elapsed_s"), receipt.get("wait_elapsed_s"),
               verdict.get("receipt_write_completed_elapsed_s"), stdout.get("stdout_flushed_elapsed_s"))
    if (any(d.get("launch_id") != launch_id or d.get("mode") != mode
            or d.get("wall_cap_s") != cap for d in (verdict, receipt, stdout, launch))
            or launch.get("t0_perf_counter") != t0 or launch.get("reserve_s") != reserve
            or receipt.get("t0_perf_counter") != t0
            or not isinstance(summary_launch, dict) or summary_launch.get("launch_id") != launch_id
            or summary_launch.get("t0_perf_counter") != t0
            or type(launch.get("worker_process_group")) is not int or launch["worker_process_group"] <= 0
            or summary_launch.get("worker_process_group") != launch["worker_process_group"]
            or not isinstance(seal, dict) or seal.get("status") != "sealed_within_budget"
            or seal.get("terminal") is not True or seal.get("t0_perf_counter") != t0
            or seal.get("wall_cap_s") != cap or seal.get("measured_wall_s_at_summary") != reg
            or proof.get("schema") != "n0_return_observation_v1" or proof.get("wall_cap_s") != cap
            or proof.get("covered") is not True or receipt.get("worker_hard_deadline_timeout") is not False
            or type(receipt.get("wait_return_code")) is not int or receipt["wait_return_code"] != 0
            or verdict.get("verdict") != "candidate_complete" or verdict.get("problems") != []
            or receipt.get("problems") != [] or summary.get("problems") != []
            or not all(number(v) for v in elapsed) or not 0 <= elapsed[0] <= elapsed[1] <= elapsed[2] <= elapsed[3] <= wait
            or any(d.get("registered_measured_s") != reg for d in (proof, receipt, verdict, stdout))
            or any(d.get("summary_sha256") != bindings[summary_name] for d in (proof, receipt, verdict))
            or receipt.get("in_process_proof_sha256") != bindings["return_observation.json"]
            or receipt.get("launch_sha256") != bindings["outer_launch.json"]
            or verdict.get("outer_return_observation_sha256") != bindings["outer_return_observation.json"]
            or stdout.get("verdict_sha256") != bindings["acceptance_verdict.json"]):
        raise ValueError("lr_outer_return_closure_invalid")
    return completion, completion_sha, sample, sample_sha, summary, receipt


def verify_r_actual_return(attempt, binding, *, expected_admission_sha=None):
    """Require the collector's wait and a separately confirmed collector exit."""
    attempt = Path(attempt)
    if ((attempt.parent / R_RETURN_FAILURE).exists() or not isinstance(binding, dict)
            or set(binding) != {"sha256", "capture_actual_return_code"}
            or type(binding["capture_actual_return_code"]) is not int
            or binding["capture_actual_return_code"] != 0):
        raise ValueError("lr_r_actual_capture_exit_required")
    observed, digest = _object_and_sha(attempt.parent / R_RETURN_NAME)
    if digest != binding["sha256"]:
        raise ValueError("lr_r_actual_return_digest_mismatch")
    completion, completion_sha, sample, sample_sha, summary, _ = verify_observer_files(attempt, "r-admit")
    claim, _ = _object_and_sha(attempt.parent / "cohort_launch_claim.json")
    elapsed = observed.get("elapsed_after_observer_wait_s")
    returned = observed.get("observer_return_perf_counter")
    t0 = completion["t0_perf_counter"]
    pid = observed.get("observer_pid")
    if (observed.get("schema") != R_RETURN_SCHEMA or type(observed.get("observer_return_code")) is not int
            or observed["observer_return_code"] != 0 or type(pid) is not int or pid <= 0
            or claim.get("observer_pid") != pid or claim.get("t0_perf_counter") != t0
            or claim.get("mode") != "r-admit" or claim.get("synthetic") is not False
            or observed.get("launch_id") != completion["launch_id"] or observed.get("t0_perf_counter") != t0
            or observed.get("wall_cap_s") != CAPS["r-admit"][0]
            or observed.get("registered_measured_s") != completion["registered_measured_s"]
            or not number(elapsed) or not number(returned) or returned - t0 != elapsed
            or not sample["elapsed_after_completion_write_s"] <= elapsed <= completion["registered_measured_s"]
            or observed.get("r_admission_sha256") != completion["bindings"]["r_admission.json"]
            or (expected_admission_sha is not None and observed["r_admission_sha256"] != expected_admission_sha)
            or observed.get("completion_sha256") != completion_sha
            or observed.get("observer_write_observation_sha256") != sample_sha
            or summary.get("admission_passed") is not True):
        raise ValueError("lr_r_actual_observer_return_invalid")
    return summary


def verify_stage_ledger(attempt, summary, receipt=None):
    """Independently reconcile the final ledger, both seals and both proofs."""
    attempt = Path(attempt)
    ledger, ledger_sha = _object_and_sha(attempt / "costs.json")
    rows = ledger.get("rows")
    if (ledger.get("schema") != LEDGER_SCHEMA or not isinstance(rows, list)
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("stage") for row in rows] != list(STAGE_REPORTS)
            or summary.get("stage_ledger") != {"rows": rows, "sha256": ledger_sha}):
        raise ValueError("lr_stage_ledger_binding_invalid")
    ends = summary["seal"]["t0_perf_counter"]
    stage_artifacts = None if receipt is None else receipt.get("stage_artifacts")
    if receipt is not None and (not isinstance(stage_artifacts, dict) or set(stage_artifacts) != {"d1", "f"}):
        raise ValueError("lr_stage_return_bindings_missing")
    for row in rows:
        stage = row["stage"]
        rel, cap = STAGE_REPORTS[stage]
        path = attempt / rel
        report, report_sha = _object_and_sha(path)
        proof, proof_sha = _object_and_sha(path.parent / "return_observation.json")
        seal = report["seal"]
        if not isinstance(seal, dict):
            raise ValueError(f"lr_stage_seal_not_object:{stage}")
        reg, t0, observed = seal["measured_wall_s_at_summary"], seal["t0_perf_counter"], proof.get("elapsed_s")
        facts = summary[stage.lower()]
        returned, closed = facts["stage_returned_elapsed_s"], facts["stage_closed_perf_counter"]
        if (row.get("status") != "sealed_candidate" or row.get("report_seal") != "sealed_within_budget"
                or row.get("wall_cap_s") != cap or row.get("t0_perf_counter") != t0
                or row.get("measured_wall_s") != reg
                or seal.get("status") != "sealed_within_budget" or seal.get("terminal") is not True
                or seal.get("wall_cap_s") != cap or proof.get("wall_cap_s") != cap
                or proof.get("registered_measured_s") != reg
                or proof.get("schema") != "n0_return_observation_v1" or proof.get("covered") is not True
                or proof.get("summary_sha256") != report_sha or proof.get("stage_ledger_row") != row
                or not all(number(v) for v in (reg, t0, observed, returned, closed))
                or not returned <= observed <= closed - t0 <= reg <= cap or t0 != ends
                or (receipt is not None and closed > receipt["wait_return_perf_counter"])
                or (stage_artifacts is not None and stage_artifacts.get(stage.lower()) != {
                    "report_sha256": report_sha, "return_observation_sha256": proof_sha})):
            raise ValueError(f"lr_stage_ledger_report_mismatch:{stage}")
        ends = closed
