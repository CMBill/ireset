"""Metadata-only preauthorization materials for the W1 D6 shared-gradient study.

This module deliberately does not open checkpoint, NPZ, source dataset, or
robot asset payloads. Its output is an unsigned review draft, never permission
to run the numerical worker.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


SCHEMA = "w1_d6_shared_gradient_real_admission_v1"
AUTH_SCHEMA = SCHEMA + "/authorization_v1"
ROOT_REL = Path("derived/audit/w1_d6_shared_gradient_step0_v1")
SOURCE_ROOT = Path("derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1")
PROPOSED_REAL_ROOT = Path("derived/experiments/omniretarget_first/w1_d6_shared_gradient_step0_v1/attempt_001")
R6_REPORT = Path("derived/audit/w1_d6_joint_fit_lr_half_v1/r6/attempt_001/r6_admission.json")
OUTCOME = Path("derived/audit/w1_d6_joint_fit_lr_half_v1/execution_review/verified_outcome.json")
INDEX = SOURCE_ROOT / "d6/artifacts_index.json"
IC_ACCEPTANCE = ROOT_REL / "synthetic/attempt_006/cuda_acceptance.json"
REVIEW_DIR = ROOT_REL / "real_admission"
METADATA_PINS = {
    "r6": (205291, "9e54107c7a6bf812b0f87098f1925e1d2f09293f4c1affd825179c28988eb527"),
    "outcome": (21510, "1de06dae2eca5d554b948db24e33399647f868ef60283ace4775e887e57e56b0"),
    "d6_index": (22707, "c5cca9bc313f27ae4733e46a40c0b89db8fb7b87caae5a742e45e3ddc865129c"),
    "ic_acceptance": (24981, "f05fc5054f27158b5a753ae751c982ea6758c737bc092e3d5aa00ae61ba32d73"),
}

WINDOWS = [
    "sub10_largebox_097:0:60", "sub12_largebox_082:0:60", "sub16_largebox_000:0:60",
    "sub3_largebox_027:0:60", "sub7_largebox_022:0:60", "sub8_largebox_035:0:60",
]
SELECTED_D6_FILES = ["d6/initial.pt", "d6/probe_inputs.npz", "d6/payload.json",
                     "d6/optimizer_roster.json", "d6/history.jsonl", "d6/probe_index.jsonl"]
FORBIDDEN_NOW = ("checkpoint_payload_reads", "npz_payload_reads", "npz_member_reads",
                 "dataset_value_reads", "asset_payload_reads", "model_constructions",
                 "optimizer_constructions", "forward_calls", "backward_calls", "parameter_updates")
ZERO_COUNTS = dict.fromkeys(FORBIDDEN_NOW, 0)
CHECKLIST = (
    "scope_and_partial_result_reviewed", "source_read_set_and_units_reviewed",
    "implementation_and_worker_closure_reviewed", "call_io_deadline_matrix_reviewed",
    "one_shot_300s_budget_and_zero_updates_reviewed", "independent_review_complete",
)


class AdmissionError(RuntimeError):
    pass


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise AdmissionError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_metadata(path: Path, limit: int = 64 << 20) -> tuple[bytes, dict]:
    """Open a bounded regular metadata file without following a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "metadata_file_type_or_size")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    _require(len(raw) == before.st_size and
             (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             "metadata_changed_during_read")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}


def _json(raw: bytes, code: str):
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(code) from exc
    _require(type(value) is dict, code + "_not_object")
    return value


def _source_read_set(index: dict) -> list[dict]:
    files = index.get("files")
    _require(type(files) is dict, "d6_index_files_invalid")
    rows = []
    for rel in SELECTED_D6_FILES:
        entry = files.get(rel)
        rec = entry.get("record") if type(entry) is dict else None
        _require(type(rec) is dict and type(rec.get("bytes")) is int and rec["bytes"] > 0
                 and type(rec.get("sha256")) is str and len(rec["sha256"]) == 64,
                 "selected_d6_file_not_bound:" + rel)
        rows.append({"path": str(SOURCE_ROOT / rel), "bytes": rec["bytes"], "sha256": rec["sha256"],
                     "role": entry.get("role"), "step": entry.get("step"),
                     "access": "future_single_read_after_authorization"})
    return rows


def build_materials(project_root: Path) -> tuple[dict, dict]:
    """Bind sealed prior evidence and current code; never read workload payloads."""
    root = project_root.resolve(strict=True)
    meta_records = {}
    parsed = {}
    for name, rel in (("r6", R6_REPORT), ("outcome", OUTCOME),
                      ("d6_index", INDEX), ("ic_acceptance", IC_ACCEPTANCE)):
        raw, rec = _read_metadata(root / rel)
        pinned_bytes, pinned_sha = METADATA_PINS[name]
        _require(rec["sha256"] == pinned_sha and (pinned_bytes is None or rec["bytes"] == pinned_bytes),
                 "historical_metadata_pin:" + name)
        meta_records[name] = rec
        parsed[name] = _json(raw, name + "_metadata_invalid")

    outcome, r6, index, ic = (parsed[k] for k in ("outcome", "r6", "d6_index", "ic_acceptance"))
    _require(outcome.get("schema") == "w1_d6_joint_fit_lr_half_v1/verified_outcome_v1"
             and outcome.get("complete_evidence") is True and outcome.get("replay_passed") is True
             and outcome.get("verdict") == "fit_insufficient", "historical_d6_anchor_invalid")
    _require(r6.get("metadata_passed") is True and r6.get("authorized") is False,
             "historical_r6_not_admitted")
    pkg = r6.get("package")
    _require(type(pkg) is dict and pkg.get("contract", {}).get("executable") is False,
             "r6_package_invalid")
    bindings = pkg.get("bindings")
    _require(type(bindings) is dict and [row.get("window_id") for row in bindings.get("windows", [])] == WINDOWS,
             "r6_window_roster_mismatch")
    _require(ic.get("schema") == "w1_d6_shared_gradient_synthetic_v1/cuda_synthetic_acceptance"
             and ic.get("passed") is True and ic.get("cases") == 18
             and len(ic.get("cases_summary", [])) == 18,
             "synthetic_ic_anchor_invalid")
    _require(index.get("schema") == "w1_d6_joint_fit_i6_v1/artifacts",
             "d6_artifact_index_schema_invalid")
    _require(index.get("identity", {}).get("r6_admission") ==
             {"bytes": METADATA_PINS["r6"][0], "sha256": METADATA_PINS["r6"][1]}
             and [row.get("window_id") for row in index.get("slots", [])] == WINDOWS,
             "d6_index_provenance_or_slots_mismatch")
    d6 = outcome.get("stages", {}).get("D6", {})
    _require(type(d6.get("report")) is dict and d6.get("report", {}).get("bytes", 0) > 0
             and len(d6.get("report", {}).get("sha256", "")) == 64
             and d6.get("wait", {}).get("launch", {}).get("sha256") == index["identity"].get("launch", {}).get("sha256"),
             "d6_outcome_index_binding_mismatch")

    implementation_paths = [
        "src/ireset/eval/w1_d6_shared_gradient.py",
        "src/ireset/eval/w1_d6_shared_gradient_admission.py",
        "src/ireset/eval/w1_d6_shared_gradient_real.py",
        "src/ireset/eval/w1_d6_joint_fit.py",
        "src/ireset/eval/w1_d6_joint_fit_evidence.py",
        "src/ireset/eval/w1_d6_joint_fit_protocol.py",
        "src/ireset/eval/w1_d6_runtime.py",
        "docs/plans/omniretarget_w1_d6_shared_gradient_prereg_draft.md",
        "scripts/verify_w1_d6_shared_gradient.py",
        "scripts/prepare_w1_d6_shared_gradient_admission.py",
        "scripts/supervise_w1_d6_shared_gradient.py",
        "tests/test_w1_d6_shared_gradient.py",
        "tests/test_w1_d6_shared_gradient_admission.py",
        "tests/test_w1_d6_shared_gradient_real.py",
        "src/ireset/train/trainer.py", "src/ireset/train/losses.py",
        "src/ireset/model/conditioner.py", "src/ireset/model/deterministic.py",
        "src/ireset/model/backbone.py", "src/ireset/model/tokens.py",
        "src/ireset/geometry/distance_adapter.py", "src/ireset/kinematics/rotation_adapter.py",
        "src/ireset/eval/w1_smallset_fit_r_admission.py", "src/ireset/embodied/g1_spec.py",
        "src/ireset/eval/w1_smallset_fit_diagnostic.py", "src/ireset/kinematics/torch_adapter.py",
        "src/ireset/eval/w1_real_fk_audit.py", "src/ireset/data/schema.py",
        "src/ireset/data/teacher.py", "src/ireset/paths.py",
        "src/ireset/eval/w1_d6_joint_fit_real.py", "src/ireset/eval/w1_d6_r6.py",
        "scripts/run_w1_d6_joint_fit_real.py", "scripts/run_w1_d6_r6.py",
        "pyproject.toml", "uv.lock",
    ]
    implementation = {}
    for rel in implementation_paths:
        raw, rec = _read_metadata(root / rel, 16 << 20)
        implementation[rel] = {"bytes": rec["bytes"], "sha256": rec["sha256"]}

    call_matrix = {
        "real_sources": {"selected_d6_artifacts": 6, "root_stats_npz": 1, "production_assets": 41},
        "numerics": {"checkpoint_cpu_load": 1, "strict_model_loads": 2, "trainer_predict": 1,
                     "loss_assembly": 1, "decode": 2, "production_fk": 2, "paired_vjp": 7,
                     "cuda_sync_points": 32},
        "control": {"optimizer_construct": 0, "loader_iterations": 0, "contact_branch": 0,
                    "parameter_updates": 0},
        "supervisor": {"worker_launches": 1, "worker_retries": 0, "return_witnesses": 1,
                       "pre_and_post_implementation_hash_passes": 2},
    }
    source_bindings = pkg["bindings"]
    production_assets = source_bindings["production_assets"]["files"]
    _require(type(production_assets) is dict and len(production_assets) == 41,
             "r6_asset_roster_invalid")
    io_matrix = {
        "metadata_preparation_now": {"metadata_documents_read_once": 4,
                                     "implementation_files_hashed_once": len(implementation_paths),
                                     "checkpoint_payload_reads": 0, "npz_payload_or_member_reads": 0,
                                     "dataset_or_asset_payload_reads": 0},
        "authorized_future_reads_proposed": {
            "d6_files": [{"path": x["path"], "bytes": x["bytes"], "sha256": x["sha256"],
                          "read": 1, "hash": 1} for x in _source_read_set(index)],
            "root_stats": {"path": source_bindings["root_stats"]["file"]["path"],
                           "bytes": source_bindings["root_stats"]["file"]["bytes"],
                           "sha256": source_bindings["root_stats"]["file"]["sha256"],
                           "read": 1, "hash": 1, "npz_members": ["mean", "std"]},
            "production_assets": [{"path": rel, "bytes": rec["bytes"], "sha256": rec["sha256"],
                                   "read": 1, "hash": 1}
                                  for rel, rec in sorted(production_assets.items())],
        },
        "future_outputs_proposed": {"worker_npy": 20, "worker_json": 5, "outer_json_max": 4,
                                    "stdout_log_bytes_max": 256 << 10,
                                    "stderr_log_bytes_max": 256 << 10, "supervisor_receipt": 1,
                                    "retry": 0, "overwrite": 0},
        "closure_status": "proposed_only_not_runtime_verified",
    }
    # The current I/C matrix describes synthetic core work only. Real source,
    # asset, setup and outer-return counts remain proposed until worker review.
    package = {
        "schema": SCHEMA, "state": "metadata_only_materials_real_kernel_partial_supervision_pending",
        "executable": False, "authorized": False,
        "proposed_real_root": str(PROPOSED_REAL_ROOT),
        "historical": {"r6": meta_records["r6"], "verified_outcome": meta_records["outcome"],
                       "d6_artifacts_index": meta_records["d6_index"],
                       "synthetic_ic_acceptance": meta_records["ic_acceptance"]},
        "source": {"windows": WINDOWS, "selected_d6_files": _source_read_set(index),
                   "d6_identity": index["identity"],
                   "d6_config": pkg["contract"]["config"],
                   "d6_contract": pkg["contract"],
                   "d6_bindings": pkg["bindings"],
                   "root_stats": {"file": pkg["bindings"]["root_stats"]["file"],
                                  "headers": pkg["bindings"]["root_stats"]["headers"]},
                   "production_assets": production_assets,
                   "r6_source_binding_sha256": _sha(json.dumps(pkg["bindings"], sort_keys=True,
                                                                   separators=(",", ":")).encode()),
                   "selected_payloads_opened": False},
        "implementation": {"files": implementation},
        "call_matrix_proposal": call_matrix,
        "io_matrix_proposal": io_matrix,
        "budgets": {"wall_cap_s": 300, "work_cap_s": 240, "seal_return_reserve_s": 60,
                    "attempts": 1, "parameter_updates": 0},
        "forbidden_calls": dict(ZERO_COUNTS),
        "status": {"fresh_source_hash_verification": "not_run",
                   "checkpoint_and_batch_read": "not_run",
                   "real_worker": "numerical_kernel_implemented_supervisor_integration_pending",
                   "external_authorization": "not_granted"},
    }
    package["metadata_sha256"] = _sha(json.dumps(package, sort_keys=True, separators=(",", ":")).encode())
    draft = {
        "schema": AUTH_SCHEMA, "cohort": "w1_d6_shared_gradient_step0_v1",
        "authorized": False, "template_only": True, "includes": ["single_step0_diagnostic"],
        "attempts": 1, "checklist": dict.fromkeys(CHECKLIST), "approval_reference": None,
        "bindings": {"admission_metadata_sha256": package["metadata_sha256"],
                     "ic_acceptance": meta_records["ic_acceptance"],
                     "historical_d6_outcome": meta_records["outcome"],
                     "implementation_files": implementation},
    }
    return package, draft


def authorization_problems(auth: dict, package: dict) -> list[str]:
    problems = []
    if type(auth) is not dict or auth.get("schema") != AUTH_SCHEMA:
        return ["authorization_schema"]
    if auth.get("authorized") is not True or auth.get("template_only") is not False:
        problems.append("authorization_not_signed")
    if (auth.get("cohort") != "w1_d6_shared_gradient_step0_v1" or type(auth.get("attempts")) is not int
            or auth.get("attempts") != 1 or auth.get("includes") != ["single_step0_diagnostic"]):
        problems.append("authorization_scope_mismatch")
    if auth.get("checklist") != dict.fromkeys(CHECKLIST, True):
        problems.append("authorization_checklist_incomplete")
    bindings = auth.get("bindings")
    if type(bindings) is not dict or bindings.get("admission_metadata_sha256") != package.get("metadata_sha256"):
        problems.append("authorization_material_binding_mismatch")
    if type(bindings) is dict:
        expected_bindings = {
            "implementation_files": package.get("implementation", {}).get("files"),
            "ic_acceptance": package.get("historical", {}).get("synthetic_ic_acceptance"),
            "historical_d6_outcome": package.get("historical", {}).get("verified_outcome"),
        }
        for name, expected in expected_bindings.items():
            if bindings.get(name) != expected:
                problems.append("authorization_binding_mismatch:" + name)
    if (package.get("executable") is not True or
            package.get("status", {}).get("real_worker") != "implemented_and_reviewed_and_supervised"):
        problems.append("real_worker_or_supervisor_not_frozen")
    if not isinstance(auth.get("approval_reference"), str) or not auth["approval_reference"].strip():
        problems.append("approval_reference_missing")
    return problems
