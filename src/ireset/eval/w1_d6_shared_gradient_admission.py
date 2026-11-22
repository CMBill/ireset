"""Metadata-only preauthorization materials for the W1 D6 shared-gradient study.

This module deliberately does not open checkpoint, NPZ, source dataset, or
robot asset payloads. Its output is an unsigned review draft, never permission
to run the numerical worker.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys


SCHEMA = "w1_d6_shared_gradient_real_admission_v1"
AUTH_SCHEMA = SCHEMA + "/authorization_v1"
REVIEW_SCHEMA = SCHEMA + "/independent_review_v1"
ROOT_REL = Path("derived/audit/w1_d6_shared_gradient_step0_v1")
SOURCE_ROOT = Path("derived/experiments/omniretarget_first/w1_d6_joint_fit_lr_half_v1")
PROPOSED_REAL_ROOT = Path("derived/experiments/omniretarget_first/w1_d6_shared_gradient_step0_v1/attempt_022")
R6_REPORT = Path("derived/audit/w1_d6_joint_fit_lr_half_v1/r6/attempt_001/r6_admission.json")
OUTCOME = Path("derived/audit/w1_d6_joint_fit_lr_half_v1/execution_review/verified_outcome.json")
INDEX = SOURCE_ROOT / "d6/artifacts_index.json"
IC_ACCEPTANCE = ROOT_REL / "synthetic/attempt_006/cuda_acceptance.json"
REVIEW_DIR = ROOT_REL / "real_admission"
AUTHORIZATION_LEDGER_REL = ROOT_REL / "authorization_consumption_v1"
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
SUPERVISOR_GROUP_CLEANUP_CHECKS_MAX = 20
RUNTIME_EVIDENCE = {
    "schema": "w1_d6_joint_fit_lr_half_v1/section_6_3_runtime_v1",
    "helper": "src/ireset/eval/w1_real_fk_audit.py",
    "tensor_check_points": {
        "D6": [
            ["fk_chain_built", "cpu"],
            ["fk_chain_transferred", "cuda:0"],
            ["fk_ready", "cuda:0"],
            ["loss_ready", "cuda:0"],
            ["initial_checkpoint", "cpu"],
            ["loaded_model", "cuda:0"],
            ["loaded_conditioner", "cuda:0"],
        ],
    },
}
REAL_CALL_MATRIX = {
    "schema": SCHEMA + "/real_call_matrix_v1",
    "sources": {"d6_file_reads": 6, "root_stats_file_reads": 1, "asset_file_reads": 41,
                "source_file_hashes": 48, "d6_npz_member_reads": 60,
                "root_stats_npz_member_reads": 2, "npz_archive_opens": 4,
                "initial_checkpoint_cpu_loads": 1},
    "setup_units": {
        "runtime_identity_checks": 1, "runtime_dependency_hashes": 5,
        "runtime_binary_hashes": 1, "fk_runtime_sources": 1, "runtime_setups": 1,
        "site_record_decodes": 1, "urdf_decodes": 1, "joint_specs": 1,
        "joint_xml_bindings": 1, "joint_plain_compiles": 1,
        "tokenizer_constructions": 1, "metric_decoder_constructions": 1,
        "fk_constructions": 1, "fk_chain_builds": 1, "fk_chain_transfers": 1,
        "scene_specs": 1, "scene_xml_bindings": 1, "scene_compiles": 1,
        "scene_data_constructions": 1, "scene_resets": 1, "scene_mj_forwards": 1,
        # 86 setup/numeric checks plus pre/post guards for all 27 review reads.
        "loss_constructions": 1, "input_domain_validations": 6, "runtime_syncs": 140,
        "scene_accesses": 1, "scene_cache_misses": 1,
        "scene_plain_xml_bindings": 1, "scene_plain_compiles": 1,
    },
    "trainer_construction": {"dataset_constructions": 1, "trainer_constructions": 1,
                             "model_constructions": 1, "model_transfers": 1,
                             "conditioner_constructions": 1, "conditioner_transfers": 1,
                             "loader_constructions": 2, "loader_iterators": 0,
                             "dataset_reads": 0, "optimizer_constructions": 0},
    "numeric": {"input_tensor_transfers": 10, "trainer_predict": 1,
                "conditioner_forward": 1, "backbone_forward": 1, "loss_assembly": 1,
                "loss_decode": 2, "fk_call": 2, "paired_vjp": 7,
                "vjp_pre_sync_points": 7},
    "forbidden": {"contact_branch": 0, "optimizer_steps": 0,
                   "accumulate_micro": 0, "clip_gradients": 0,
                   "step_optimizer": 0, "parameter_updates": 0,
                   "checkpoint_writes": 0, "dataset_iterations": 0},
    "supervisor": {"authorization_file_reads": 1, "implementation_closure_hash_passes": 2,
                   "worker_launches": 1, "worker_retries": 0, "actual_wait_calls": 1,
                   "stdout_log_creates": 1, "stderr_log_creates": 1, "worker_result_reads": 1,
                   "return_observation_writes": 1, "return_observation_readbacks": 2,
                   "authorization_consumption_markers": 1, "authorization_ledger_events": 3,
                   "attempts": 1, "process_group_created": 1,
                   "group_cleanup_checks_max": SUPERVISOR_GROUP_CLEANUP_CHECKS_MAX,
                   "log_streams_drained": 2, "launch_spec_writes": 1, "return_witnesses": 1},
}


def real_call_matrix_problems(value: dict) -> list[str]:
    return [] if type(value) is dict and value == REAL_CALL_MATRIX else ["real_call_matrix_not_frozen"]


def real_io_matrix_problems(package: dict) -> list[str]:
    source = package.get("source") if type(package) is dict else None
    io = package.get("io_matrix") if type(package) is dict else None
    if type(source) is not dict or type(io) is not dict:
        return ["real_io_matrix_missing"]
    files = source.get("selected_d6_files")
    root_stats = source.get("root_stats")
    stats = root_stats.get("file") if type(root_stats) is dict else None
    assets = source.get("production_assets")
    if type(files) is not list or len(files) != 6 or type(stats) is not dict or type(assets) is not dict:
        return ["real_io_source_roster_invalid"]
    if any(type(row) is not dict or type(row.get("path")) is not str
           or type(row.get("bytes")) is not int or row["bytes"] <= 0
           or type(row.get("sha256")) is not str or len(row["sha256"]) != 64
           or any(c not in "0123456789abcdef" for c in row["sha256"])
           for row in files):
        return ["real_io_d6_roster_invalid"]
    if (type(stats.get("path")) is not str or type(stats.get("bytes")) is not int
            or stats["bytes"] <= 0 or type(stats.get("sha256")) is not str
            or len(stats["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in stats["sha256"])):
        return ["real_io_root_stats_invalid"]
    if any(type(rel) is not str or type(rec) is not dict
           or type(rec.get("bytes")) is not int or rec["bytes"] <= 0
           or type(rec.get("sha256")) is not str or len(rec["sha256"]) != 64
           or any(c not in "0123456789abcdef" for c in rec["sha256"])
           for rel, rec in assets.items()):
        return ["real_io_asset_roster_invalid"]
    expected = {
        "d6_files": [{"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"], "read": 1, "hash": 1}
                      for r in files],
        "root_stats": {"path": stats["path"], "bytes": stats["bytes"], "sha256": stats["sha256"],
                       "read": 1, "hash": 1, "npz_members": ["mean", "std"]},
        "production_assets": [{"path": rel, "bytes": rec["bytes"], "sha256": rec["sha256"],
                               "read": 1, "hash": 1} for rel, rec in sorted(assets.items())],
    }
    future = io.get("authorized_future_reads")
    if future != expected or len(assets) != 41:
        return ["real_io_source_roster_not_frozen"]
    if io.get("worker_actual") != {
        "launch_metadata_reads": 3, "ledger_reads": 2, "implementation_closure_hash_passes": 2,
        "source_file_reads": 48, "source_file_hashes": 48, "d6_npz_member_reads": 60,
        "root_stats_npz_member_reads": 2, "npz_archive_opens": 4, "independent_artifact_reads": 27,
        "independent_artifact_hashes": 20, "artifact_npy_writes": 20, "artifact_json_writes": 8,
        "d6_directory_created": 1, "checkpoint_cpu_loads": 1, "strict_model_loads": 2,
        "worker_result_writes": 1, "worker_result_readbacks": 1,
    }:
        return ["real_io_worker_counts_not_frozen"]
    outer = io.get("supervisor_actual")
    expected_outer = {key: REAL_CALL_MATRIX["supervisor"][key]
                      for key in ("authorization_file_reads", "implementation_closure_hash_passes",
                                  "worker_launches", "worker_retries", "actual_wait_calls",
                                  "stdout_log_creates", "stderr_log_creates", "worker_result_reads",
                                  "return_observation_writes", "return_observation_readbacks",
                                  "authorization_consumption_markers", "authorization_ledger_events",
                                  "attempts", "process_group_created", "log_streams_drained",
                                  "launch_spec_writes", "return_witnesses")}
    expected_outer["retries"] = expected_outer.pop("worker_retries")
    if (type(outer) is not dict
            or {key: outer.get(key) for key in expected_outer} != expected_outer
            or type(outer.get("group_cleanup_checks")) is not int
            or not 0 <= outer["group_cleanup_checks"] <= REAL_CALL_MATRIX["supervisor"]["group_cleanup_checks_max"]):
        return ["real_io_supervisor_counts_not_frozen"]
    return []


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


def _fk_buffer_source_records() -> dict:
    distribution = importlib.metadata.distribution("pytorch-kinematics")
    roster = set(map(str, distribution.files or ()))
    records = {}
    for relative in ("pytorch_kinematics/chain.py", "pytorch_kinematics/frame.py"):
        _require(relative in roster, "runtime_fk_buffer_source_missing:" + relative)
        path = Path(distribution.locate_file(relative)).resolve(strict=True)
        raw, record = _read_metadata(path, 4 << 20)
        records[relative] = {"path": str(path), "bytes": record["bytes"],
                             "sha256": record["sha256"]}
    return records


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
        "scripts/run_w1_d6_shared_gradient_worker.py",
        "scripts/review_w1_d6_shared_gradient.py",
        "tests/test_w1_d6_shared_gradient_supervisor.py",
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

    call_matrix = json.loads(json.dumps(REAL_CALL_MATRIX))
    runtime_cpu_threads = pkg["bindings"]["runtime"]["identity"].get("cpu_threads")
    _require(type(runtime_cpu_threads) is int and runtime_cpu_threads > 0,
             "r6_runtime_cpu_threads_invalid")
    worker_launch = {
        "script_rel": "scripts/run_w1_d6_shared_gradient_worker.py",
        "argv_template": [sys.executable, "-I",
                          str(root / "scripts/run_w1_d6_shared_gradient_worker.py"),
                          "--launch-spec", "{launch_spec}",
                          "--admission-snapshot", "{admission_snapshot}",
                          "--authorization-snapshot", "{authorization_snapshot}",
                          "--authorization-sha256", "{authorization_sha256}",
                          "--project-root", "{project_root}"],
        "executable": sys.executable,
        "cwd": "attempt_root",
        "env": {"LC_CTYPE": "C.UTF-8", "OMP_NUM_THREADS": str(runtime_cpu_threads),
                "MKL_NUM_THREADS": str(runtime_cpu_threads),
                "OPENBLAS_NUM_THREADS": "1"},
        "shell": False,
        "inherit_stdin": False,
        "new_process_group": True,
        "execution_mode": "real_worker_v1",
        "status": "proposed_not_executable",
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
        "authorized_future_reads": {
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
        "future_outputs_proposed": {"worker_npy": 20, "worker_json": 9, "outer_json_max": 5,
                                     "stdout_log_bytes_max": 256 << 10,
                                     "stderr_log_bytes_max": 256 << 10, "supervisor_receipt": 1,
                                     "retry": 0, "overwrite": 0},
        "worker_reads": {"launch_spec": 1, "admission_snapshot": 1, "authorization_snapshot": 1,
                          "consumption_receipt": 1, "launch_event": 1,
                           "artifact_readbacks_for_independent_review": 27,
                          "artifact_sha256_for_independent_review": 20},
        "worker_writes": {"d6_directory_create": 1, "artifact_npy": 20,
                           "artifact_json": 8, "worker_result": 1},
        "supervisor_reads": {"authorization": 1, "worker_result": 1,
                             "worker_logs": 2, "return_observation": 2,
                             "ledger_consumption_verifications": 3},
        "supervisor_writes": {"admission_snapshot": 1, "authorization_snapshot": 1,
                              "launch_spec": 1, "ledger_consumption": 1,
                              "ledger_events": 3, "stdout_log": 1, "stderr_log": 1,
                              "return_observation": 1},
        "worker_actual": {"launch_metadata_reads": 3, "ledger_reads": 2,
                          "implementation_closure_hash_passes": 2,
                          "source_file_reads": 48, "source_file_hashes": 48,
                          "d6_npz_member_reads": 60, "root_stats_npz_member_reads": 2,
                           "npz_archive_opens": 4, "independent_artifact_reads": 27,
                          "independent_artifact_hashes": 20, "artifact_npy_writes": 20,
                           "artifact_json_writes": 8, "d6_directory_created": 1,
                          "checkpoint_cpu_loads": 1, "strict_model_loads": 2,
                          "worker_result_writes": 1, "worker_result_readbacks": 1},
        "supervisor_actual": {"authorization_file_reads": 1,
                              "implementation_closure_hash_passes": 2,
                              "worker_launches": 1, "actual_wait_calls": 1,
                              "stdout_log_creates": 1, "stderr_log_creates": 1,
                              "worker_result_reads": 1, "return_observation_writes": 1,
                              "return_observation_readbacks": 2,
                              "authorization_consumption_markers": 1,
                              "authorization_ledger_events": 3,
                              "attempts": 1, "retries": 0,
                              "process_group_created": 1, "group_cleanup_checks": 0,
                              "log_streams_drained": 2, "launch_spec_writes": 1,
                              "return_witnesses": 1},
        "closure_status": "frozen_and_enforced_by_supervisor_and_worker",
    }
    d6_contract = json.loads(json.dumps(pkg["contract"]))
    runtime_evidence = json.loads(json.dumps(RUNTIME_EVIDENCE))
    runtime_evidence["fk_buffer_sources"] = _fk_buffer_source_records()
    d6_contract["runtime_evidence"] = runtime_evidence
    package = {
        "schema": SCHEMA, "state": "metadata_only_real_execution_pending_authorized_worker_cli_review",
        "executable": False, "authorized": False,
        "proposed_real_root": str(PROPOSED_REAL_ROOT),
        "authorization_consumption_ledger": str(AUTHORIZATION_LEDGER_REL),
        "historical": {"r6": meta_records["r6"], "verified_outcome": meta_records["outcome"],
                       "d6_artifacts_index": meta_records["d6_index"],
                       "synthetic_ic_acceptance": meta_records["ic_acceptance"]},
        "source": {"windows": WINDOWS, "selected_d6_files": _source_read_set(index),
                   "d6_identity": index["identity"],
                   "d6_config": pkg["contract"]["config"],
                   "d6_contract": d6_contract,
                   "d6_bindings": pkg["bindings"],
                   "root_stats": {"file": pkg["bindings"]["root_stats"]["file"],
                                  "headers": pkg["bindings"]["root_stats"]["headers"]},
                   "production_assets": production_assets,
                   "r6_source_binding_sha256": _sha(json.dumps(pkg["bindings"], sort_keys=True,
                                                                   separators=(",", ":")).encode()),
                   "selected_payloads_opened": False},
        "implementation": {"files": implementation},
        "worker_launch": worker_launch,
        "call_matrix": call_matrix,
        "io_matrix": io_matrix,
        "budgets": {"wall_cap_s": 300, "work_cap_s": 240, "seal_return_reserve_s": 60,
                    "attempts": 1, "parameter_updates": 0},
        "forbidden_calls": dict(ZERO_COUNTS),
        "status": {"fresh_source_hash_verification": "not_run",
                   "checkpoint_and_batch_read": "not_run",
                   "real_worker": "independent_worker_cli_implemented_matrix_enforced_pending_final_review",
                   "external_authorization": "not_granted"},
    }
    _require(not real_call_matrix_problems(package["call_matrix"]), "real_call_matrix_builder_drift")
    _require(not real_io_matrix_problems(package), "real_io_matrix_builder_drift")
    package["metadata_sha256"] = _sha(json.dumps(package, sort_keys=True, separators=(",", ":")).encode())
    draft = {
        "schema": AUTH_SCHEMA, "cohort": "w1_d6_shared_gradient_step0_v1",
        "authorized": False, "template_only": True, "includes": ["single_step0_diagnostic"],
        "attempts": 1, "checklist": dict.fromkeys(CHECKLIST), "approval_reference": None,
        "bindings": {"admission_metadata_sha256": package["metadata_sha256"],
                     "ic_acceptance": meta_records["ic_acceptance"],
                      "historical_d6_outcome": meta_records["outcome"],
                      "implementation_files": implementation,
                      "worker_launch": worker_launch},
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
            "worker_launch": package.get("worker_launch"),
        }
        if "independent_review" in package:
            expected_bindings["independent_review"] = package["independent_review"]
        for name, expected in expected_bindings.items():
            if bindings.get(name) != expected:
                problems.append("authorization_binding_mismatch:" + name)
    if (package.get("executable") is not True or
            package.get("status", {}).get("real_worker") != "implemented_and_reviewed_and_supervised"):
        problems.append("real_worker_or_supervisor_not_frozen")
    if not isinstance(auth.get("approval_reference"), str) or not auth["approval_reference"].strip():
        problems.append("approval_reference_missing")
    return problems


def build_reviewed_materials(project_root: Path, review_path: Path) -> tuple[dict, dict]:
    """Promote metadata-only materials only after an independent review receipt.

    This still never opens checkpoint, NPZ, dataset, or robot-asset payloads.
    The receipt proves only the worker/supervisor protocol and synthetic review;
    the eventual real run remains separately authorized and one-shot.
    """
    root = project_root.resolve(strict=True)
    package, draft = build_materials(root)
    review_path = Path(review_path).absolute()
    if review_path.resolve(strict=True) != review_path or not review_path.is_relative_to(root):
        raise AdmissionError("independent_review_path_not_in_project")
    raw, record = _read_metadata(review_path, 2 << 20)
    review = _json(raw, "independent_review_invalid")
    _require(review.get("schema") == REVIEW_SCHEMA, "independent_review_schema")
    implementation_sha = _sha(json.dumps(package["implementation"]["files"],
                                         sort_keys=True, separators=(",", ":")).encode())
    _require(review.get("implementation_closure_sha256") == implementation_sha,
             "independent_review_implementation_binding")
    _require(type(review.get("reviewed_commit")) is str and len(review["reviewed_commit"]) == 40,
             "independent_review_commit")
    _require(review.get("ready_for_executable_packet") is True,
             "independent_review_not_ready")
    _require(review.get("synthetic_protocol_passed") is True
             and review.get("supervisor_dispatch_reviewed") is True,
             "independent_review_protocol_incomplete")
    _require(review.get("real_payload_reads") == 0
             and review.get("real_worker_spawns") == 0
             and review.get("parameter_updates") == 0,
             "independent_review_scope_violation")
    tests = review.get("tests")
    _require(type(tests) is dict and type(tests.get("passed")) is int
             and tests["passed"] >= 65 and tests.get("exit_code") == 0,
             "independent_review_tests_incomplete")
    review_record = {"path": str(review_path.relative_to(root)), **record}
    package["independent_review"] = review_record
    package["state"] = "executable_ready_independent_reviewed"
    package["executable"] = True
    package["authorized"] = False
    package["worker_launch"]["status"] = "reviewed_executable"
    package["status"]["real_worker"] = "implemented_and_reviewed_and_supervised"
    package["status"]["fresh_source_hash_verification"] = "not_run_until_authorized"
    package["status"]["checkpoint_and_batch_read"] = "not_run_until_authorized"
    package["status"]["external_authorization"] = "not_granted"
    package_body = {key: value for key, value in package.items()
                    if key != "metadata_sha256"}
    package["metadata_sha256"] = _sha(json.dumps(package_body, sort_keys=True,
                                                   separators=(",", ":")).encode())
    draft["bindings"]["independent_review"] = review_record
    draft["bindings"]["admission_metadata_sha256"] = package["metadata_sha256"]
    return package, draft
