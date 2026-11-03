"""Real provider for the W1 training-update audit (execution plan Gate A).

The provider has two lifecycles:

* ``metadata`` -- reads paths, NPZ headers, JSON metadata, file byte digests
  and asset reference *text* only.  It never calls ``torch.load``, never
  decodes a checkpoint payload, never loads an NPZ array value, never
  constructs a model/optimizer and never compiles MuJoCo.
* ``materialized`` -- created inside a registered P/I/F stage after R0 sealed
  and the authorization/admission bindings were checked.  It loads the four
  frozen parent windows, their constants and the R0 noise schedule; the
  checkpoint strict load and model construction happen lazily per model in
  ``build_trainer`` under the same stage guards.

The synthetic provider implements the same reference-backend interface so the
generic ``ReferenceContext`` never calls the analytic FK for a real run.

No disk access happens at import time.
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ireset.eval.w1_real_training_update_audit import (
    ADAM_DEFAULTS,
    BOUNDARIES,
    CALL_CAPS,
    CHAIN_NOMINAL,
    CHECKPOINTS,
    COHORT_ID,
    DRAW_UNITS_PER_PATH,
    EXECUTION_ID,
    FK_PARENT,
    IMPLEMENTATION_FILES,
    LOSS_WEIGHTS,
    MICRO_COUNT,
    MICRO_SCHEDULE,
    MODEL_CONFIG,
    MODELS,
    NOISE_PROTOCOL,
    NOISE_SEED,
    OBJECTIVE_PARENT,
    PARENT_FILES,
    PREREG_DRAFT,
    STAGE_BUDGETS,
    STAGE_DIRS,
    STAGE_NOMINAL,
    TRAIN_CONFIG_PATH,
    TRAIN_CONFIG_SHA,
    UPDATES,
    WINDOW,
    WINDOW_IDS,
    WINDOW_SPECS,
    CallCounter,
    CountingFk,
    CountingLoss,
    ForbiddenLoss,
    SyntheticReferenceBackend,
    _PlaceholderDataset,
    _array_shape_problems,
    _canonical_sha256,
    _hash_array,
    _jsonable,
    _model_factory,
    _nonneg,
    _num,
    alpha_cumprod_reference,
    build_parameter_roster,
    configure_production_runtime,
    draw_noise_path,
    ALPHA_EVALUATION,
    effective_alpha_reference,
    verify_production_alpha,
    environment_facts,
    implementation_identity,
    independent_x_t,
    read_json,
    row_catalog,
    roster_problems,
    runtime_identity_facts,
    runtime_identity_problems,
    sha256_file,
    write_json,
)
from ireset.eval.w1_real_fk_audit import (
    RUNTIME_REQUIREMENTS,
    SITE_NAMES,
    _asset_closure_facts,
    effective_constants,
    independent_decode,
    mat_to_quat_wxyz_np,
)
from ireset.eval.w1_real_objective_audit import independent_encode
from ireset.model.tokens import RootStats, StateTokenizer
from ireset.paths import PROJECT_ROOT, path_for_record
from ireset.train.losses import LossAssembler
from ireset.train.trainer import Trainer, TrainConfig

PROVIDER_SCHEMA = "w1_real_training_update_provider_v1"
REFERENCE_SCHEMA = "w1_real_training_update_reference_backend_v1"
SYNTHETIC_ACCEPTANCE_SCHEMA = (
    "w1_real_training_update_synthetic_acceptance_v1")
SYNTHETIC_ACCEPTANCE_SCHEMA_V2 = (
    "w1_real_training_update_synthetic_acceptance_v2")
CALIBRATION_VERIFICATION_SCHEMA = (
    "w1_real_training_update_calibration_verification_v1")
REAL_ADMISSION_SCHEMA = "w1_real_training_update_real_admission_v1"


def acceptance_leaf_record(
        record: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ordinary child record of an acceptance v2 wrapper.

    Gate B v2 keeps the frozen v1 record under ``ordinary.record`` so both
    synthetic chains stay byte-comparable; admission binds the wrapper but
    reads roster/tolerance/implementation from this leaf.
    """
    if not isinstance(record, Mapping):
        return {}
    if record.get("schema") == SYNTHETIC_ACCEPTANCE_SCHEMA_V2:
        ordinary = record.get("ordinary")
        leaf = ordinary.get("record") if isinstance(ordinary, Mapping) else None
        return leaf if isinstance(leaf, Mapping) else {}
    return record


N_JOINTS = 29
N_SITES = len(SITE_NAMES)
STAGES = ("R0", "P", "I", "F")

REQUIRED_WINDOW_HEADERS = {
    "x0": (np.dtype(np.float32), (4, WINDOW, 38)),
    "kps_obj": (np.dtype(np.float32), (4, WINDOW, 52, 3)),
    "kps_clip": (np.dtype(np.float32), (4, WINDOW, 52, 3)),
    "box_t_clip": (np.dtype(np.float32), (4, WINDOW, 3)),
    "box_quat": (np.dtype(np.float32), (4, WINDOW, 4)),
    "contact": (np.dtype(np.float32), (4, WINDOW, 4)),
    "raw_root": (np.dtype(np.float64), (4, WINDOW, 7)),
    "raw_q": (np.dtype(np.float64), (4, WINDOW, N_JOINTS)),
    "raw_wfc": (np.dtype(np.float64), (4, 4, 4)),
    "root_mean": (np.dtype(np.float64), (3,)),
    "root_std": (np.dtype(np.float64), (3,)),
    "joint_lo": (np.dtype(np.float64), (N_JOINTS,)),
    "joint_hi": (np.dtype(np.float64), (N_JOINTS,)),
    "alpha_cumprod": (np.dtype(np.float64), (1000,)),
}

WINDOW_VALUE_KEYS = ("x0", "kps_obj", "kps_clip", "box_t_clip", "box_quat",
                     "contact", "raw_root", "raw_q", "raw_wfc", "root_mean",
                     "root_std", "joint_lo", "joint_hi", "alpha_cumprod")

CHECKPOINT_REQUIRED_KEYS = (
    "model", "conditioner", "step", "config", "root_stats",
    "parameters_finite", "checkpoint_usable", "diagnostic_only", "failure",
)
CHECKPOINT_STEP = 4000


# ---------------------------------------------------------------------------
# NPZ header inspection (no array payload decoding)
# ---------------------------------------------------------------------------


def read_npz_headers(path: Path) -> dict[str, dict]:
    """Return key/dtype/shape/order for every ``.npy`` member of an NPZ.

    Only the ``.npy`` format header is parsed; array payloads are never read.
    """
    headers: dict[str, dict] = {}
    with zipfile.ZipFile(Path(path)) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name.endswith(".npy") or info.is_dir():
                continue
            key = name[:-4]
            with archive.open(info) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = \
                        np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = \
                        np.lib.format.read_array_header_2_0(handle)
                else:  # pragma: no cover - numpy has no other writer
                    raise ValueError(f"unsupported npy version {version}")
            headers[key] = {
                "shape": [int(x) for x in shape],
                "dtype": np.dtype(dtype).name,
                "fortran_order": bool(fortran),
                "compressed": info.compress_type != zipfile.ZIP_STORED,
            }
    return headers


def window_header_problems(headers: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for key, (dtype, shape) in REQUIRED_WINDOW_HEADERS.items():
        record = (headers or {}).get(key)
        if not isinstance(record, Mapping):
            problems.append(f"header_missing:{key}")
            continue
        if record.get("dtype") != np.dtype(dtype).name:
            problems.append(f"header_dtype:{key}:{record.get('dtype')}")
        if tuple(record.get("shape") or ()) != shape:
            problems.append(f"header_shape:{key}:{record.get('shape')}")
        if record.get("fortran_order"):
            problems.append(f"header_order:{key}")
        if record.get("compressed"):
            problems.append(f"header_compressed:{key}")
    return problems


def _window_values_problems(arrays: Mapping[str, np.ndarray],
                            constants: Mapping[str, np.ndarray]) -> list[str]:
    problems: list[str] = []
    for key, (dtype, shape) in REQUIRED_WINDOW_HEADERS.items():
        value = arrays.get(key)
        if not isinstance(value, np.ndarray):
            problems.append(f"input_missing:{key}")
            continue
        if value.dtype != dtype:
            problems.append(f"input_dtype:{key}:{value.dtype}")
        if value.shape != shape:
            problems.append(f"input_shape:{key}:{value.shape}")
        if value.size == 0:
            problems.append(f"input_empty:{key}")
        elif not np.isfinite(value).all():
            problems.append(f"input_nonfinite:{key}")
    if problems:
        return problems
    x0 = np.asarray(arrays["x0"], np.float32)
    decoded = independent_decode(
        x0.astype(np.float64),
        constants["mean32"].astype(np.float64),
        constants["std32"].astype(np.float64),
        constants["lo32"].astype(np.float64),
        constants["hi32"].astype(np.float64))
    if not decoded["ok"].all():
        problems.append("input_rotation_domain")
    joint = x0[..., 9:38]
    if not ((joint >= -1.0) & (joint <= 1.0)).all():
        problems.append("input_joint_domain")
    quat = np.asarray(arrays["box_quat"], np.float64)
    norms = np.linalg.norm(quat, axis=-1)
    if not ((norms > 1e-6) & np.isfinite(norms)).all():
        problems.append("input_box_quat_norm")
    contact = np.asarray(arrays["contact"], np.float64)
    if not ((contact >= 0.0) & (contact <= 1.0)).all():
        problems.append("input_contact_domain")
    lo = np.asarray(arrays["joint_lo"], np.float64)
    hi = np.asarray(arrays["joint_hi"], np.float64)
    q = np.asarray(arrays["raw_q"], np.float64)
    if not ((q >= lo - 1e-9) & (q <= hi + 1e-9)).all():
        problems.append("input_joint_limits")
    root = np.asarray(arrays["raw_root"], np.float64)
    if not ((root[..., 3:7] ** 2).sum(-1) > 1e-12).all():
        problems.append("input_root_quat_norm")
    wfc = np.asarray(arrays["raw_wfc"], np.float64)
    for index in range(wfc.shape[0]):
        block = wfc[index, :3, :3]
        if not np.allclose(block.T @ block, np.eye(3), atol=1e-9):
            problems.append(f"input_wfc_rotation:{index}")
    return problems


# ---------------------------------------------------------------------------
# reference backends
# ---------------------------------------------------------------------------


class RealReferenceBackend:
    """MuJoCo plain-MJCF reference with per-call/Site deadline guards.

    ``unit_admission`` runs once at the entry of every ``positions`` /
    ``jacobians`` call (the caller also guards before the call); the absolute
    work deadline is re-checked before every ``mj_forward`` and ``mj_jacSite``
    so a slow per-state call cannot run past it.  Counts are incremented only
    after the corresponding call returns, so a partial failure keeps the
    completed amount and never back-fills a batch nominal.
    """

    schema = REFERENCE_SCHEMA
    mode = "real"

    def __init__(self, counter: CallCounter, *, deadline: float | None,
                 guard: Callable[[str], None] | None = None,
                 joint_lo: np.ndarray | None = None,
                 joint_hi: np.ndarray | None = None) -> None:
        import mujoco
        from ireset.embodied.g1_spec import G1Spec, JOINT_NAMES, KPT_BODIES

        self.mujoco = mujoco
        self.counter = counter
        self.deadline = None if deadline is None else float(deadline)
        self._guard = guard
        # ``G1Spec()`` already compiles once in ``__post_init__``; guard that
        # compile and the second compile before any construction continues.
        self._check("reference_backend_compile")
        spec = G1Spec()
        self._check("reference_backend_recompile")
        self.model = spec.load_plain()
        self._check("reference_backend_attach")
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 7 + N_JOINTS or self.model.nv != 6 + N_JOINTS:
            raise ValueError(
                f"locked MJCF nq/nv mismatch: {self.model.nq}/{self.model.nv}")
        self.site_ids: dict[str, int] = {}
        for name in SITE_NAMES:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"missing site {name}")
            self.site_ids[name] = int(sid)
            body_id = int(self.model.site_bodyid[sid])
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name != KPT_BODIES[name]:
                raise ValueError(f"site {name} body mismatch: {body_name}")
        for index, name in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"missing joint {name}")
            if int(self.model.jnt_qposadr[jid]) != 7 + index:
                raise ValueError(f"joint {name} qpos address")
            if int(self.model.jnt_type[jid]) != int(
                    mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"joint {name} is not a hinge")
            if joint_lo is not None and joint_hi is not None:
                lo, hi = (float(v) for v in self.model.jnt_range[jid])
                if abs(lo - float(joint_lo[index])) > 1e-9 \
                        or abs(hi - float(joint_hi[index])) > 1e-9:
                    raise ValueError(f"joint {name} limits differ")

    def _check(self, unit: str) -> None:
        if self._guard is not None:
            self._guard(unit)
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError(f"reference_deadline:{unit}")

    def _set_state(self, pos: np.ndarray, quat: np.ndarray,
                   q: np.ndarray) -> None:
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = quat
        self.data.qpos[7:7 + N_JOINTS] = q

    def positions(self, pos: np.ndarray, R: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
        q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
        self._check("reference_positions_entry")
        quat = mat_to_quat_wxyz_np(R)
        out = np.zeros((pos.shape[0], N_SITES, 3))
        for i in range(pos.shape[0]):
            self._check(f"reference_mj_forward_{i}")
            self._set_state(pos[i], quat[i], q[i])
            self.mujoco.mj_forward(self.model, self.data)
            self.counter.add("reference_mj_forward", 1)
            for s, name in enumerate(SITE_NAMES):
                out[i, s] = self.data.site_xpos[self.site_ids[name]]
        return out

    def jacobians(self, pos: np.ndarray, R: np.ndarray,
                  q: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
        q = np.asarray(q, dtype=np.float64).reshape(-1, N_JOINTS)
        self._check("reference_jacobians_entry")
        quat = mat_to_quat_wxyz_np(R)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        out = np.zeros((pos.shape[0], N_SITES, 3, self.model.nv))
        for i in range(pos.shape[0]):
            self._check(f"reference_mj_forward_{i}")
            self._set_state(pos[i], quat[i], q[i])
            self.mujoco.mj_forward(self.model, self.data)
            self.counter.add("reference_mj_forward", 1)
            for s, name in enumerate(SITE_NAMES):
                self._check(f"reference_mj_jac_site_{i}_{s}")
                self.mujoco.mj_jacSite(
                    self.model, self.data, jacp, jacr, self.site_ids[name])
                out[i, s] = jacp
                self.counter.add("reference_mj_jac_site", 1)
        Rt = np.swapaxes(R, 1, 2)
        out[..., 3:6] = np.einsum("nsck,nkj->nscj", out[..., 3:6], Rt)
        return out


# ---------------------------------------------------------------------------
# parent consumer facts
# ---------------------------------------------------------------------------


def _parent_consumer_problems(root: Path) -> tuple[dict, list[str]]:
    """Verify the parent run disk x index x ledger terminal state."""
    problems: list[str] = []
    facts: dict[str, Any] = {}
    for label, parent in (("objective", OBJECTIVE_PARENT),
                          ("fk", FK_PARENT)):
        directory = Path(root) / parent
        index_path = directory / "attempts_index.json"
        ledger_path = directory / "costs.json"
        if not index_path.is_file() or not ledger_path.is_file():
            problems.append(f"parent_consumer_missing:{label}")
            facts[label] = {"index": None, "ledger": None}
            continue
        try:
            index = read_json(index_path).get("attempts") or []
            ledger = read_json(ledger_path).get("invocations") or []
        except (OSError, ValueError, TypeError):
            problems.append(f"parent_consumer_unreadable:{label}")
            facts[label] = {"index": None, "ledger": None}
            continue
        if not index or not ledger:
            problems.append(f"parent_consumer_empty:{label}")
        index_by = {(r.get("stage"), r.get("attempt")): r for r in index
                    if isinstance(r, Mapping)}
        ledger_by = {(r.get("stage"), r.get("attempt")): r for r in ledger
                     if isinstance(r, Mapping)}
        if set(index_by) != set(ledger_by) or len(index_by) != len(index):
            problems.append(f"parent_consumer_sets:{label}")
        rows = []
        for key in sorted(index_by, key=lambda k: (str(k[0]), str(k[1]))):
            row = index_by[key]
            cost = ledger_by.get(key) or {}
            if row.get("status") != "done" or row.get("complete") is not True:
                problems.append(
                    f"parent_consumer_status:{label}:{key[0]}:{key[1]}")
            if cost.get("status") != "done" or cost.get("complete") is not True:
                problems.append(
                    f"parent_consumer_ledger:{label}:{key[0]}:{key[1]}")
            measured = row.get("measured_wall_s")
            if not _nonneg(measured) or float(measured) != float(
                    cost.get("measured_wall_s", float("nan"))):
                problems.append(
                    f"parent_consumer_cost:{label}:{key[0]}:{key[1]}")
            rows.append({"stage": key[0], "attempt": key[1],
                         "status": row.get("status"),
                         "measured_wall_s": measured})
        facts[label] = {
            "index": path_for_record(index_path),
            "ledger": path_for_record(ledger_path),
            "index_sha256": sha256_file(index_path),
            "ledger_sha256": sha256_file(ledger_path),
            "attempts": rows,
        }
    # The parent final/admission payloads must still claim their pass.
    checks = (
        ("objective_admission", OBJECTIVE_PARENT / "registration/attempt_001/"
         "admission.json", (("passed", True),)),
        ("objective_p_report", OBJECTIVE_PARENT / "production/attempt_001/"
         "p_report.json", (("complete_evidence", True),
                           ("covered_contract_passed", True))),
        ("objective_summary", OBJECTIVE_PARENT / "finalize/attempt_001/"
         "audit_summary.json", (("complete_evidence", True),
                                ("covered_contract_passed", True),
                                ("conclusion",
                                 "real_batch_covered_contract_passed"))),
        ("objective_verification", OBJECTIVE_PARENT / "finalize/attempt_001/"
         "independent_verification.json",
         (("conclusion", "real_batch_covered_contract_passed"),)),
        ("fk_final", FK_PARENT / "finalize/attempt_001/fk_audit_final.json",
         (("complete_evidence", True), ("covered_contract_passed", True),
          ("conclusion", "real_saved_fk_covered_contract_passed"))),
        ("fk_verification", FK_PARENT / "finalize/attempt_001/"
         "independent_verification.json",
         (("complete_evidence", True), ("covered_contract_passed", True))),
    )
    for name, rel, expectations in checks:
        path = Path(root) / rel
        if not path.is_file():
            problems.append(f"parent_claim_missing:{name}")
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError, TypeError):
            problems.append(f"parent_claim_unreadable:{name}")
            continue
        for field, expected in expectations:
            if payload.get(field) != expected:
                problems.append(f"parent_claim:{name}:{field}")
        for field in ("problems", "failures", "verification_problems"):
            values = payload.get(field)
            if values:
                problems.append(f"parent_claim:{name}:{field}_nonempty")
    return {"schema": "w1_real_training_update_parent_consumers_v1",
            "parents": facts,
            "objective_admission": path_for_record(
                Path(root) / OBJECTIVE_PARENT / "registration/attempt_001/"
                "admission.json"),
            "problems": problems}, problems


# ---------------------------------------------------------------------------
# synthetic acceptance / calibration verification
# ---------------------------------------------------------------------------


def calibration_verification(contract: Mapping[str, Any],
                             contract_path: Path | None,
                             contract_sha256: str | None) -> dict:
    from ireset.eval.w1_real_training_update_audit import (
        tolerance_contract_problems,
    )

    recompute = tolerance_contract_problems(contract, recompute=True)
    calibration = contract.get("calibration") or {}
    observed = calibration.get("observed") or {}
    ratios = {key: 4.0 * float((rec or {}).get("max_ratio_of_proposal", 0.0))
              for key, rec in observed.items()}
    four_x_ok = bool(ratios) and all(math.isfinite(v) and v <= 1.0
                                     for v in ratios.values())
    verification = {
        "schema": CALIBRATION_VERIFICATION_SCHEMA,
        "mode": contract.get("mode"),
        "contract_path": (path_for_record(contract_path)
                          if contract_path is not None else None),
        "contract_sha256": contract_sha256,
        "contract_schema": contract.get("schema"),
        "evidence_sha256": contract.get("evidence_sha256"),
        "fp32_contract_sha256": contract.get("fp32_contract_sha256"),
        "recompute_problems": recompute,
        "four_x_ratios": ratios,
        "four_x_ok": four_x_ok,
        "calibration_seed": calibration.get("seed"),
        "calibration_device": calibration.get("device"),
        "real_dataset_loaded": bool(calibration.get("real_dataset_loaded")),
        "real_checkpoint_loaded": bool(
            calibration.get("real_checkpoint_loaded")),
        "real_asset_loaded": bool(calibration.get("real_asset_loaded")),
        "real_model_calls": int(calibration.get("real_model_calls") or 0),
        "implementation": implementation_identity(PROJECT_ROOT),
        "environment": environment_facts(),
        "runtime_identity": runtime_identity_facts(),
        "passed": not recompute and four_x_ok,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return verification


def calibration_verification_problems(
        verification: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(verification, Mapping):
        return ["calibration_verification_missing"]
    problems: list[str] = []
    if verification.get("schema") != CALIBRATION_VERIFICATION_SCHEMA:
        problems.append("calibration_verification_schema")
    if verification.get("mode") != "synthetic_calibrated":
        problems.append("calibration_verification_mode")
    if verification.get("passed") is not True:
        problems.append("calibration_verification_not_passed")
    if verification.get("recompute_problems"):
        problems.append("calibration_verification_recompute")
    if verification.get("four_x_ok") is not True:
        problems.append("calibration_verification_four_x")
    if verification.get("real_dataset_loaded") is not False \
            or verification.get("real_checkpoint_loaded") is not False \
            or verification.get("real_asset_loaded") is not False \
            or verification.get("real_model_calls") != 0:
        problems.append("calibration_verification_real_reads")
    for key in ("contract_sha256", "evidence_sha256",
                "fp32_contract_sha256"):
        if not isinstance(verification.get(key), str):
            problems.append(f"calibration_verification_binding:{key}")
    problems += [f"calibration_{p}" for p in runtime_identity_problems(
        verification.get("runtime_identity"))]
    return problems


def _record_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_manifest(directory: Path) -> tuple[dict, list[str]]:
    """Hash every regular file below ``directory`` (sorted relative paths)."""
    directory = Path(directory)
    files: dict[str, dict] = {}
    problems: list[str] = []
    if not directory.is_dir():
        return files, [f"directory_missing:{path_for_record(directory)}"]
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(directory).as_posix()
        try:
            files[rel] = {"sha256": sha256_file(path),
                          "bytes": int(path.stat().st_size)}
        except (OSError, ValueError) as exc:
            problems.append(f"unreadable:{rel}:{type(exc).__name__}")
    return files, problems


def _manifest_problems(directory: Path, files: Any, *, label: str
                       ) -> list[str]:
    if not isinstance(files, Mapping) or not files:
        return [f"manifest_missing:{label}"]
    problems: list[str] = []
    root = Path(directory)
    for rel, record in files.items():
        if not isinstance(rel, str) or not isinstance(record, Mapping):
            problems.append(f"manifest_entry:{label}:{rel}")
            continue
        path = root / rel
        if not path.is_file():
            problems.append(f"file_missing:{label}:{rel}")
            continue
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            problems.append(f"file_sha:{label}:{rel}")
        if int(path.stat().st_size) != record.get("bytes"):
            problems.append(f"file_size:{label}:{rel}")
    return problems


#: Evidence files every sealed synthetic stage must expose.  A trimmed
#: manifest that drops one of these is refused even if the hashes agree.
STAGE_REQUIRED_FILES = {
    "R0": ("registration.json", "admission.json", "call_reservations.json"),
    "P": ("p_report.json", "p_rows.json", "p_deterministic.npz",
          "p_diffusion.npz", "call_reservations.json"),
    "I": ("i_report.json", "i_rows.json", "i_deterministic.npz",
          "i_diffusion.npz", "call_reservations.json"),
    "F": ("training_update_final.json", "worker_spec.json",
          "worker/worker_result.json", "call_reservations.json"),
}
RUN_REQUIRED_FILES = ("cohort.json", "attempts_index.json", "costs.json")
FIXTURE_REQUIRED_FILES = ("fixture.json", "roster.json", "noise.npz",
                          "windows.npz")


def synthetic_acceptance(
        result: Mapping[str, Any], *, run_root: Path,
        tolerance_contract: Mapping[str, Any],
        calibration_verification_record: Mapping[str, Any] | None,
        roster: Mapping[str, Any]) -> dict:
    """Persist the Gate B synthetic acceptance consumed by real admission."""
    stages = result.get("stages") or {}
    seals = result.get("seals") or {}
    f_stage = stages.get("F") or {}
    selected_attempts = result.get("selected_attempts") or {}
    stage_records = {}
    file_problems: list[str] = []
    for stage in ("R0", "P", "I", "F"):
        payload = stages.get(stage) or {}
        seal = seals.get(stage) or {}
        complete = payload.get("complete_evidence")
        if complete is None:
            complete = payload.get("passed")
        report = payload.get("report")
        if report is None and stage == "R0":
            report = payload.get("admission")
        attempt = selected_attempts.get(stage)
        directory = (Path(run_root) / STAGE_DIRS[stage] / str(attempt)
                     if attempt else None)
        files, problems = (_file_manifest(directory) if directory is not None
                           else ({}, [f"stage_attempt_missing:{stage}"]))
        file_problems += [f"{stage}:{p}" for p in problems]
        stage_records[stage] = {
            "attempt": attempt,
            "directory": path_for_record(directory) if directory else None,
            "files": files,
            "complete_evidence": bool(complete),
            "failures": len(payload.get("failures") or []),
            "gaps": len(payload.get("gaps") or []),
            "report": report,
            "sealed": bool(seal.get("sealed")),
            "terminal": bool(seal.get("terminal")),
            "seal_problems": list(seal.get("problems") or []),
        }
    run_files: dict[str, dict] = {}
    for name in RUN_REQUIRED_FILES:
        path = Path(run_root) / name
        if path.is_file():
            run_files[name] = {"sha256": sha256_file(path),
                               "bytes": int(path.stat().st_size)}
    fixture_files, problems = _file_manifest(Path(run_root) / "fixture")
    file_problems += [f"fixture:{p}" for p in problems]
    for stage in STAGES:
        for name in STAGE_REQUIRED_FILES[stage]:
            if name not in (stage_records[stage].get("files") or {}):
                file_problems.append(f"{stage}:required_missing:{name}")
    for name in RUN_REQUIRED_FILES:
        if name not in run_files:
            file_problems.append(f"run:required_missing:{name}")
    for name in FIXTURE_REQUIRED_FILES:
        if name not in fixture_files:
            file_problems.append(f"fixture:required_missing:{name}")
    worker = (result.get("stages") or {}).get("F", {}).get("worker_result") \
        or {}
    problems = list(file_problems)
    if result.get("complete_evidence") is not True:
        problems.append("acceptance_incomplete_evidence")
    if result.get("problems"):
        problems.append("acceptance_problems")
    if result.get("failures"):
        problems.append("acceptance_failures")
    if f_stage.get("verification_problems"):
        problems.append("acceptance_f_verification")
    if any(not rec["sealed"] or not rec["terminal"]
           for rec in stage_records.values()):
        problems.append("acceptance_stage_not_sealed")
    if any(rec["complete_evidence"] is not True
           for rec in stage_records.values()):
        problems.append("acceptance_stage_incomplete")
    if not worker or worker.get("pid") == worker.get("ppid"):
        problems.append("acceptance_worker_process")
    if not tolerance_contract.get("evidence_sha256") \
            or not tolerance_contract.get("fp32_contract_sha256"):
        problems.append("acceptance_tolerance_binding")
    for problem in calibration_verification_problems(
            calibration_verification_record):
        problems.append(f"acceptance_calibration:{problem}")
    catalog = row_catalog(roster)
    expected_rows = 14 * int(roster["n_total"]) + 168
    if catalog["n_rows"] != expected_rows:
        problems.append("acceptance_row_formula")
    selected = result.get("selected_chain_counts") or {}
    full = dict(CHAIN_NOMINAL)
    full["scene_setup_mj_forward"] = 0
    if selected != full:
        problems.append("acceptance_selected_chain_matrix")
    record = {
        "schema": SYNTHETIC_ACCEPTANCE_SCHEMA,
        "mode": "synthetic",
        "cohort_id": COHORT_ID,
        "execution_id": EXECUTION_ID,
        "run_root": path_for_record(run_root),
        "run_sha256": _canonical_sha256({
            "stages": stage_records,
            "run_files": run_files,
            "fixture_files": fixture_files,
            "conclusion": result.get("conclusion"),
            "calls": (result.get("calls") or {}).get("counts"),
            "selected_chain_counts": selected,
        }),
        "conclusion": result.get("conclusion"),
        "complete_evidence": bool(result.get("complete_evidence")),
        "covered_contract_passed": bool(
            result.get("covered_contract_passed")),
        "problems": problems,
        "passed": not problems,
        "stages": stage_records,
        "run_files": run_files,
        "fixture_files": fixture_files,
        "selected_chain_counts": selected,
        "selected_chain_training_updates": int(
            selected.get("native_adamw_steps", 0)
            + selected.get("explicit_reference_updates", 0)),
        "call_caps": dict(CALL_CAPS),
        "roster": {
            "sha256": roster["sha256"], "n_det": roster["n_det"],
            "n_diff": roster["n_diff"], "n_total": roster["n_total"],
            "n_rows": catalog["n_rows"], "row_formula": "14N+168",
            "row_keys_sha256": catalog["keys_sha256"],
            "parameter_roster": copy.deepcopy(dict(roster)),
        },
        "tolerance": {
            "path": None,
            "sha256": None,
            "mode": tolerance_contract.get("mode"),
            "evidence_sha256": tolerance_contract.get("evidence_sha256"),
            "fp32_contract_sha256": tolerance_contract.get(
                "fp32_contract_sha256"),
            "contract": copy.deepcopy(dict(tolerance_contract)),
        },
        "calibration_verification": copy.deepcopy(
            dict(calibration_verification_record or {})),
        "implementation": implementation_identity(PROJECT_ROOT),
        "environment": environment_facts(),
        "runtime_identity": runtime_identity_facts(),
        "read_scope": {
            "real_checkpoint_loaded": False,
            "real_asset_loaded": False,
            "real_dataset_loaded": False,
            "real_model_calls": 0,
        },
        "notes": [
            "synthetic acceptance covers the frozen synthetic matrix only; "
            "numeric gaps (if any) remain audit_incomplete",
            "the acceptance is metadata for real admission, not a real run",
        ],
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return record


def synthetic_acceptance_problems(record: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(record, Mapping):
        return ["acceptance_not_a_mapping"]
    if record.get("schema") == SYNTHETIC_ACCEPTANCE_SCHEMA_V2:
        return synthetic_acceptance_v2_problems(record)
    problems: list[str] = []
    if record.get("schema") != SYNTHETIC_ACCEPTANCE_SCHEMA:
        problems.append("acceptance_schema")
    if record.get("mode") != "synthetic":
        problems.append("acceptance_mode")
    if record.get("passed") is not True:
        problems.append("acceptance_not_passed")
    if record.get("problems"):
        problems.append("acceptance_problems")
    if record.get("complete_evidence") is not True:
        problems.append("acceptance_complete_evidence")
    stages = record.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(STAGES):
        problems.append("acceptance_stages")
    else:
        for stage in STAGES:
            rec = stages.get(stage) or {}
            if rec.get("sealed") is not True or rec.get("terminal") is not True:
                problems.append(f"acceptance_stage_seal:{stage}")
            if rec.get("complete_evidence") is not True:
                problems.append(f"acceptance_stage_evidence:{stage}")
            if rec.get("failures") != 0 or rec.get("seal_problems"):
                problems.append(f"acceptance_stage_failures:{stage}")
            directory = _record_path(rec.get("directory"))
            files = rec.get("files")
            if directory is None:
                problems.append(f"acceptance_stage_directory:{stage}")
                continue
            problems += [f"acceptance_stage_file:{stage}:{p}"
                         for p in _manifest_problems(
                             directory, files, label=f"stage:{stage}")]
            if not isinstance(files, Mapping):
                continue
            for name in STAGE_REQUIRED_FILES[stage]:
                if name not in files:
                    problems.append(
                        f"acceptance_stage_required:{stage}:{name}")
            report = _record_path(rec.get("report"))
            if report is None:
                problems.append(f"acceptance_stage_report:{stage}")
            else:
                if report.resolve().parent != directory.resolve():
                    problems.append(f"acceptance_stage_report_dir:{stage}")
                if report.name not in files:
                    problems.append(f"acceptance_stage_report_manifest:{stage}")
    run_root = _record_path(record.get("run_root"))
    run_files = record.get("run_files")
    fixture_files = record.get("fixture_files")
    if run_root is None:
        problems.append("acceptance_run_root")
    else:
        problems += [f"acceptance_run_file:{p}" for p in _manifest_problems(
            run_root, run_files, label="run")]
        if isinstance(run_files, Mapping):
            for name in RUN_REQUIRED_FILES:
                if name not in run_files:
                    problems.append(f"acceptance_run_required:{name}")
        problems += [f"acceptance_fixture_file:{p}"
                     for p in _manifest_problems(
                         run_root / "fixture", fixture_files,
                         label="fixture")]
        if isinstance(fixture_files, Mapping):
            for name in FIXTURE_REQUIRED_FILES:
                if name not in fixture_files:
                    problems.append(f"acceptance_fixture_required:{name}")
    problems += [f"acceptance_runtime:{p}" for p in runtime_identity_problems(
        record.get("runtime_identity"))]
    selected = record.get("selected_chain_counts") or {}
    expected = dict(CHAIN_NOMINAL)
    expected["scene_setup_mj_forward"] = 0
    if selected != expected:
        problems.append("acceptance_selected_chain_matrix")
    if record.get("selected_chain_training_updates") != 16:
        problems.append("acceptance_training_updates")
    roster = record.get("roster") or {}
    if roster.get("row_formula") != "14N+168" or not roster.get(
            "parameter_roster"):
        problems.append("acceptance_roster")
    else:
        parameter_roster = roster["parameter_roster"]
        if parameter_roster.get("sha256") != roster.get("sha256"):
            problems.append("acceptance_roster_sha")
        if row_catalog(parameter_roster)["n_rows"] != int(
                roster.get("n_rows", -1)):
            problems.append("acceptance_row_catalog")
    tolerance = record.get("tolerance") or {}
    if tolerance.get("mode") != "synthetic_calibrated":
        problems.append("acceptance_tolerance_mode")
    if not tolerance.get("evidence_sha256") \
            or not tolerance.get("fp32_contract_sha256"):
        problems.append("acceptance_tolerance_binding")
    problems += [f"acceptance_calibration:{p}" for p in
                 calibration_verification_problems(
                     record.get("calibration_verification"))]
    scope = record.get("read_scope") or {}
    for key in ("real_checkpoint_loaded", "real_asset_loaded",
                "real_dataset_loaded"):
        if scope.get(key) is not False:
            problems.append(f"acceptance_read_scope:{key}")
    if scope.get("real_model_calls") != 0:
        problems.append("acceptance_read_scope:real_model_calls")
    return problems


# ---------------------------------------------------------------------------
# metadata lifecycle
# ---------------------------------------------------------------------------


def synthetic_acceptance_v2(
        *, ordinary: Mapping[str, Any], ordinary_path: Path,
        ordinary_sha256: str, migration: Mapping[str, Any],
        migration_path: Path, migration_sha256: str,
        spec_sha256: str, snapshot_sha256: str,
        legacy_head_sha256: str) -> dict:
    """Bind the ordinary and migration synthetic chains into one Gate B.

    The two child records keep the frozen v1 acceptance schemas; this wrapper
    adds the migration identity and cohort accounting link.  It is metadata,
    not a substitute for either child run.
    """
    problems: list[str] = []
    if Path(ordinary_path).is_file():
        if sha256_file(Path(ordinary_path)) != ordinary_sha256:
            problems.append("ordinary_path_sha")
    else:
        problems.append("ordinary_path_missing")
    if Path(migration_path).is_file():
        if sha256_file(Path(migration_path)) != migration_sha256:
            problems.append("migration_path_sha")
    else:
        problems.append("migration_path_missing")
    problems += [f"ordinary:{p}" for p in
                 synthetic_acceptance_problems(ordinary)]
    problems += [f"migration:{p}" for p in
                 synthetic_acceptance_problems(migration)]
    for key in ("spec_sha256", "snapshot_sha256", "legacy_head_sha256"):
        value = locals()[key]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}",
                                                          value):
            problems.append(key)
    migration_sel = migration.get("selected_chain_training_updates")
    migration_cohort = migration.get("cohort_training_updates")
    if migration_sel != 16 or migration_cohort != 18:
        problems.append("migration_update_accounting")
    if migration.get("generation") != "generation_002":
        problems.append("migration_generation")
    record = {
        "schema": SYNTHETIC_ACCEPTANCE_SCHEMA_V2,
        "mode": "synthetic",
        "passed": not problems,
        "problems": problems,
        "ordinary": {"path": path_for_record(Path(ordinary_path)),
                     "sha256": ordinary_sha256,
                     "record": dict(ordinary)},
        "migration": {"path": path_for_record(Path(migration_path)),
                      "sha256": migration_sha256,
                      "record": dict(migration),
                      "spec_sha256": spec_sha256,
                      "snapshot_sha256": snapshot_sha256,
                      "legacy_head_sha256": legacy_head_sha256,
                      "incremental_training_updates": 16,
                      "cohort_training_updates": 18,
                      "prior_training_updates": 2},
        "runtime_identity": runtime_identity_facts(),
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return record


def synthetic_acceptance_v2_problems(
        record: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(record, Mapping):
        return ["acceptance_v2_not_a_mapping"]
    problems: list[str] = []
    if record.get("schema") != SYNTHETIC_ACCEPTANCE_SCHEMA_V2:
        problems.append("acceptance_v2_schema")
    if record.get("mode") != "synthetic":
        problems.append("acceptance_v2_mode")
    if record.get("passed") is not True or record.get("problems"):
        problems.append("acceptance_v2_not_passed")
    for name in ("ordinary", "migration"):
        part = record.get(name)
        if not isinstance(part, Mapping):
            problems.append(f"acceptance_v2_{name}")
            continue
        path = _record_path(part.get("path"))
        if path is None or not path.is_file():
            problems.append(f"acceptance_v2_{name}_path")
        elif sha256_file(path) != part.get("sha256"):
            problems.append(f"acceptance_v2_{name}_sha")
        problems += [f"acceptance_v2_{name}:{p}" for p in
                     synthetic_acceptance_problems(part.get("record"))]
    migration = record.get("migration") or {}
    if migration.get("incremental_training_updates") != 16 or \
            migration.get("cohort_training_updates") != 18 or \
            migration.get("prior_training_updates") != 2:
        problems.append("acceptance_v2_update_accounting")
    for key in ("spec_sha256", "snapshot_sha256", "legacy_head_sha256"):
        value = migration.get(key)
        if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", value):
            problems.append(f"acceptance_v2_{key}")
    return problems


def _checkpoint_config_problems(
        checkpoint_configs: Mapping[str, Any] | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(checkpoint_configs, Mapping):
        return ["metadata_checkpoint_configs_missing"]
    for model in MODELS:
        cfg = checkpoint_configs.get(model)
        if not isinstance(cfg, Mapping):
            problems.append(f"metadata_checkpoint_config:{model}")
            continue
        if cfg.get("model_type") != model:
            problems.append(f"metadata_checkpoint_model_type:{model}")
        for key in ("micro_batch", "grad_accum", "d_model", "n_layers",
                    "n_heads", "ffn_mult", "time_window",
                    "num_train_timesteps"):
            want = int(MODEL_CONFIG[key])
            if cfg.get(key) != want:
                problems.append(f"metadata_checkpoint_config:{model}:{key}")
        for key in ("lr", "weight_decay", "grad_clip", "beta_start",
                    "beta_end"):
            want = float(MODEL_CONFIG[key])
            if not _num(cfg.get(key)) or float(cfg[key]) != want:
                problems.append(f"metadata_checkpoint_config:{model}:{key}")
        if cfg.get("beta_schedule") != MODEL_CONFIG["beta_schedule"]:
            problems.append(f"metadata_checkpoint_config:{model}:beta_schedule")
        if cfg.get("steps") != CHECKPOINT_STEP:
            problems.append(f"metadata_checkpoint_steps:{model}")
        loss = cfg.get("loss") or {}
        for key, want in (("w_pos", 1.0), ("w_rot6d", 1.0), ("w_joint", 1.0),
                          ("w_geodesic", 0.5), ("w_fk", 2.0),
                          ("w_contact", 0.0), ("w_vel", 0.5)):
            if not _num(loss.get(key)) or float(loss[key]) != want:
                problems.append(f"metadata_checkpoint_loss:{model}:{key}")
    return problems


def _window_meta_problems(window_keys: Any) -> list[str]:
    if not isinstance(window_keys, list) or len(window_keys) != len(WINDOW_SPECS):
        return ["metadata_window_count"]
    problems: list[str] = []
    for index, (demo, start, end, total) in enumerate(WINDOW_SPECS):
        rec = window_keys[index]
        if not isinstance(rec, Mapping):
            problems.append(f"metadata_window:{index}")
            continue
        if rec.get("demo") != demo or int(rec.get("start", -1)) != start \
                or int(rec.get("end", -1)) != end \
                or int(rec.get("T", -1)) != total:
            problems.append(f"metadata_window:{index}")
        if rec.get("split") != "train":
            problems.append(f"metadata_window_split:{index}")
    return problems


def _state_dict_problems(module: Any, payload: Any,
                         label: str) -> list[str]:
    """Strict payload contract: exact keys, dtype, shape and finiteness."""
    if not isinstance(payload, Mapping):
        return [f"{label}_payload_not_mapping"]
    state = module.state_dict()
    problems: list[str] = []
    missing = sorted(set(state) - set(payload))
    extra = sorted(set(payload) - set(state))
    if missing:
        problems.append(f"{label}_missing:{','.join(missing[:3])}")
    if extra:
        problems.append(f"{label}_unexpected:{','.join(extra[:3])}")
    for name, target in state.items():
        value = payload.get(name)
        if not torch.is_tensor(value):
            problems.append(f"{label}_not_tensor:{name}")
            continue
        if value.dtype != target.dtype:
            problems.append(f"{label}_dtype:{name}:{value.dtype}")
        if tuple(value.shape) != tuple(target.shape):
            problems.append(f"{label}_shape:{name}")
        if not bool(torch.isfinite(value).all()):
            problems.append(f"{label}_nonfinite:{name}")
    return problems


def _copy_state_strict(module: Any, payload: Any, label: str) -> None:
    problems = _state_dict_problems(module, payload, label)
    if problems:
        raise ValueError(f"checkpoint state refused: {problems}")
    module.load_state_dict(payload, strict=True)


class RealTrainingProvider:
    """Registered real provider; never constructed without Gate A wiring."""

    mode = "real"

    def __init__(self, project_root: Path = PROJECT_ROOT,
                 config: Mapping[str, Any] | None = None, *,
                 device: str = "cuda",
                 acceptance_path: Path | None = None) -> None:
        self.root = Path(project_root)
        self.config = dict(MODEL_CONFIG if config is None else config)
        self.device = device
        self.counter: CallCounter | None = None
        self.roster: dict | None = None
        self.noise: list[dict] = []
        self.window_arrays: dict[str, dict[str, np.ndarray]] = {}
        self.tokenizer: StateTokenizer | None = None
        self.constants: dict | None = None
        self.alpha: np.ndarray | None = None
        self.reference: RealReferenceBackend | None = None
        self.problems: list[str] = []
        self.metadata_facts: dict | None = None
        self.acceptance_path: Path | None = (
            Path(acceptance_path) if acceptance_path is not None else None)
        self.acceptance: dict | None = None
        self.acceptance_sha: str | None = None
        self.source_sha256: str | None = None
        self.input_admission: dict | None = None
        self.materialized_stage: str | None = None
        self._deadline: float | None = None
        self._guard: Callable[[str], None] | None = None
        self._production_assembler: LossAssembler | None = None
        self._scene_setup_counted = False
        self._parents_cache: dict | None = None

    # --- provider identity ------------------------------------------------

    def bind_counter(self, counter: CallCounter) -> None:
        self.counter = counter
        self._rebind_reference_counter()

    def _rebind_reference_counter(self) -> None:
        if self.reference is not None and self.counter is not None:
            self.reference.counter = self.counter

    @property
    def alpha_cumprod(self) -> np.ndarray | None:
        return self.alpha

    def spec(self) -> dict:
        return {
            "schema": PROVIDER_SCHEMA,
            "mode": "real",
            "project_root": path_for_record(self.root),
            "config": dict(self.config),
            "device": self.device,
            "acceptance_path": (path_for_record(self.acceptance_path)
                                if self.acceptance_path is not None else None),
            "acceptance_sha256": self.acceptance_sha,
            "runtime_identity": runtime_identity_facts(),
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any],
                  project_root: Path = PROJECT_ROOT) -> "RealTrainingProvider":
        if spec.get("mode") != "real":
            raise ValueError("real provider spec mode mismatch")
        root = Path(project_root)
        acceptance = spec.get("acceptance_path")
        acceptance_path = None
        if acceptance:
            acceptance_path = Path(acceptance)
            if not acceptance_path.is_absolute():
                acceptance_path = root / acceptance_path
        provider = cls(root, spec.get("config") or MODEL_CONFIG,
                       device=str(spec.get("device", "cuda")),
                       acceptance_path=acceptance_path)
        if provider.acceptance_path is not None:
            # The worker must carry the same acceptance identity as its
            # parent; without this the materialize guard cannot re-check it.
            provider.load_acceptance()
            if spec.get("acceptance_sha256") \
                    and provider.acceptance_sha != spec.get("acceptance_sha256"):
                raise ValueError("synthetic acceptance sha drifted")
        return provider

    def for_stage(self, stage: str) -> "RealTrainingProvider":
        provider = RealTrainingProvider(
            self.root, self.config, device=self.device,
            acceptance_path=self.acceptance_path)
        provider.acceptance = self.acceptance
        provider.acceptance_sha = self.acceptance_sha
        provider.roster = self.roster
        if self.counter is not None:
            provider.bind_counter(self.counter)
        return provider

    # --- metadata ---------------------------------------------------------

    def load_acceptance(self) -> dict:
        if self.acceptance is not None:
            return self.acceptance
        if self.acceptance_path is None or not self.acceptance_path.is_file():
            raise ValueError("synthetic acceptance path missing")
        record = read_json(self.acceptance_path)
        self.acceptance_sha = sha256_file(self.acceptance_path)
        problems = synthetic_acceptance_problems(record)
        if problems:
            raise ValueError(f"synthetic acceptance refused: {problems}")
        self.acceptance = record
        roster_source = record
        if record.get("schema") == SYNTHETIC_ACCEPTANCE_SCHEMA_V2:
            roster_source = ((record.get("ordinary") or {}).get("record")
                             or {})
        roster = ((roster_source.get("roster") or {}).get(
            "parameter_roster"))
        if not isinstance(roster, Mapping):
            raise ValueError("synthetic acceptance roster missing")
        self.roster = copy.deepcopy(dict(roster))
        return record

    def metadata(self) -> dict:
        if self.metadata_facts is not None:
            return self.metadata_facts
        problems: list[str] = []
        try:
            self.load_acceptance()
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"metadata_acceptance:{exc}")
        parents: dict[str, dict] = {}
        byte_reads = 0
        for name, (rel, want) in PARENT_FILES.items():
            path = self.root / rel
            actual = sha256_file(path) if path.is_file() else None
            byte_reads += 1 if actual is not None else 0
            parents[name] = {"path": path_for_record(path), "expected": want,
                             "actual": actual,
                             "match": actual is not None and actual == want}
            if actual is None:
                problems.append(f"parent_missing:{name}")
            elif actual != want:
                problems.append(f"parent_drifted:{name}")
        consumers, consumer_problems = _parent_consumer_problems(self.root)
        problems += consumer_problems
        checkpoints = {}
        for model, (rel, want) in CHECKPOINTS.items():
            path = self.root / rel
            actual = sha256_file(path) if path.is_file() else None
            byte_reads += 1 if actual is not None else 0
            checkpoints[model] = {
                "path": path_for_record(path), "expected": want,
                "actual": actual,
                "bytes": int(path.stat().st_size) if path.is_file() else None,
                "match": actual is not None and actual == want,
                "step": CHECKPOINT_STEP,
            }
            if actual != want:
                problems.append(f"checkpoint:{model}")
        config_path = self.root / TRAIN_CONFIG_PATH
        config_sha = sha256_file(config_path) if config_path.is_file() else None
        byte_reads += 1 if config_sha is not None else 0
        if config_sha != TRAIN_CONFIG_SHA:
            problems.append("train_config_sha")
        metadata_registration = None
        arrays_path = self.root / PARENT_FILES["objective_arrays"][0]
        headers: dict = {}
        if arrays_path.is_file():
            try:
                headers = read_npz_headers(arrays_path)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                problems.append(f"npz_header_unreadable:{type(exc).__name__}")
        problems += window_header_problems(headers)
        try:
            registration_path = (self.root /
                                 PARENT_FILES["objective_registration"][0])
            metadata_registration = read_json(registration_path)
        except (OSError, ValueError, TypeError):
            problems.append("metadata_objective_registration_unreadable")
        window_keys = None
        checkpoint_configs = None
        if isinstance(metadata_registration, Mapping):
            window_keys = metadata_registration.get("window_keys")
            checkpoint_configs = metadata_registration.get(
                "checkpoint_configs")
            problems += _window_meta_problems(window_keys)
            problems += _checkpoint_config_problems(checkpoint_configs)
        assets = _asset_closure_facts(self.root, mode="real")
        problems += [f"asset:{p}" for p in assets.get("problems") or []]
        try:
            fk_registration = read_json(
                self.root / PARENT_FILES["fk_registration"][0])
        except (OSError, ValueError, TypeError):
            fk_registration = None
        expected_assets = (fk_registration or {}).get("locked_assets") or {}
        asset_binding = _asset_binding_problems(assets, expected_assets)
        problems += asset_binding
        runtime = configure_production_runtime()
        problems += [f"runtime:{p}" for p in runtime["problems"]]
        environment = environment_facts()
        runtime_identity = runtime_identity_facts()
        if isinstance(self.acceptance, Mapping):
            problems += [f"runtime_identity:{p}"
                         for p in runtime_identity_problems(
                             self.acceptance.get("runtime_identity"),
                             runtime_identity)]
        roster = self.roster or {}
        if roster:
            catalog = row_catalog(roster)
            if catalog["n_rows"] != 14 * int(roster["n_total"]) + 168:
                problems.append("metadata_row_formula")
        else:
            catalog = {}
        impl = implementation_identity(self.root)
        missing = [rel for rel, sha in impl["files"].items()
                   if not isinstance(sha, str)]
        if missing:
            problems.append(f"implementation_missing:{missing}")
        facts = {
            "schema": PROVIDER_SCHEMA,
            "mode": "real",
            "lifecycle": "metadata",
            "project_root": path_for_record(self.root),
            "config": dict(self.config),
            "acceptance": {
                "path": (path_for_record(self.acceptance_path)
                         if self.acceptance_path is not None else None),
                "sha256": self.acceptance_sha,
            },
            "parents": parents,
            "parent_consumers": consumers,
            "checkpoints": checkpoints,
            "train_config": {"path": path_for_record(config_path),
                             "sha256": config_sha, "expected": TRAIN_CONFIG_SHA},
            "windows": window_keys,
            "window_headers": headers,
            "checkpoint_configs": checkpoint_configs,
            "assets": assets,
            "asset_binding_problems": asset_binding,
            "roster": {
                "sha256": roster.get("sha256"),
                "n_total": roster.get("n_total"),
                "n_rows": catalog.get("n_rows"),
                "row_formula": "14N+168",
            },
            "runtime": runtime,
            "environment": environment,
            "runtime_identity": runtime_identity,
            "implementation": impl,
            "read_scope": {
                "metadata_only": True,
                "file_byte_digests": byte_reads,
                "npz_headers_read": sorted(headers),
                "npz_values_read": 0,
                "checkpoint_payload_read": 0,
                "torch_load_calls": 0,
                "model_constructor_calls": 0,
                "muJoCo_compile_calls": 0,
                "cohort_created": False,
                "registered_attempt_created": False,
            },
            "problems": problems,
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self.metadata_facts = facts
        self.problems = list(problems)
        return facts

    def metadata_problems(self) -> list[str]:
        return list(self.metadata().get("problems") or [])

    # --- materialized lifecycle ------------------------------------------

    def reset_materialization(self) -> None:
        self.noise = []
        self.window_arrays = {}
        self.tokenizer = None
        self.constants = None
        self.alpha = None
        self.reference = None
        self.source_sha256 = None
        self.input_admission = None
        self.materialized_stage = None
        self._production_assembler = None
        self._scene_setup_counted = False

    def _check(self, unit: str) -> None:
        if self._deadline is not None \
                and time.perf_counter() >= self._deadline:
            raise TimeoutError(f"provider_deadline:{unit}")
        if self._guard is not None:
            self._guard(unit)

    def _verify_source_digests(self) -> None:
        """Re-hash the frozen parent sources before decoding any value.

        Resuming from a settled R0 skips the metadata lifecycle, so the same
        byte check has to run again on every materialization.
        """
        self._check("materialize_source_digests")
        arrays_rel, arrays_sha = PARENT_FILES["objective_arrays"]
        arrays_path = self.root / arrays_rel
        actual = sha256_file(arrays_path) if arrays_path.is_file() else None
        if actual is None:
            raise ValueError("parent objective arrays missing")
        if actual != arrays_sha:
            raise ValueError(
                f"parent objective arrays drifted: {actual} != {arrays_sha}")
        config_path = self.root / TRAIN_CONFIG_PATH
        config_actual = (sha256_file(config_path)
                         if config_path.is_file() else None)
        if config_actual != TRAIN_CONFIG_SHA:
            raise ValueError(
                f"train config drifted: {config_actual} != {TRAIN_CONFIG_SHA}")
        self.source_sha256 = actual

    def _verify_asset_closure(self) -> None:
        """Re-hash the frozen locked-asset closure before any compilation.

        Resuming from a sealed R0 skips ``metadata()``, so every
        materialization re-hashes the four locked files, every resolved
        mesh/include target and the parent FK registration itself.
        """
        self._check("materialize_asset_closure")
        rel, want = PARENT_FILES["fk_registration"]
        path = self.root / rel
        actual = sha256_file(path) if path.is_file() else None
        if actual != want:
            raise ValueError("parent FK registration drifted")
        registration = read_json(path)
        expected = (registration or {}).get("locked_assets") or {}
        assets = _asset_closure_facts(self.root, mode="real")
        problems = [f"asset:{p}" for p in assets.get("problems") or []]
        problems += _asset_binding_problems(assets, expected)
        if problems:
            raise ValueError(f"asset closure refused: {problems}")

    def _load_arrays(self) -> dict[str, np.ndarray]:
        self._verify_source_digests()
        self._check("materialize_array_decode")
        path = self.root / PARENT_FILES["objective_arrays"][0]
        with np.load(path) as archive:
            arrays = {key: np.asarray(archive[key])
                      for key in WINDOW_VALUE_KEYS}
        return arrays

    def _build_tokenizer(self, arrays: Mapping[str, np.ndarray]) -> None:
        lo = np.asarray(arrays["joint_lo"], np.float64)
        hi = np.asarray(arrays["joint_hi"], np.float64)
        mean = np.asarray(arrays["root_mean"], np.float64)
        std = np.asarray(arrays["root_std"], np.float64)
        self.tokenizer = StateTokenizer(
            np.stack([lo, hi], axis=1), RootStats(mean=mean, std=std))
        self.constants = effective_constants(mean, std, lo, hi)

    def _input_admission(self, arrays: Mapping[str, np.ndarray],
                         *, counter: CallCounter) -> dict:
        problems: list[str] = []
        problems += _window_values_problems(arrays, self.constants or {})
        alpha_ref = alpha_cumprod_reference(self.config)
        alpha = np.asarray(arrays["alpha_cumprod"], np.float64)
        if alpha.shape != alpha_ref.shape:
            problems.append("input_alpha_shape")
        elif not np.array_equal(alpha, alpha_ref):
            problems.append("input_alpha_schedule")
        effective = effective_alpha_reference(self.config)
        if not (np.isfinite(effective).all()
                and (effective > 0).all() and (effective <= 1).all()):
            problems.append("input_alpha_domain")
        x0 = np.asarray(arrays["x0"], np.float32)
        root = np.asarray(arrays["raw_root"], np.float64)
        q = np.asarray(arrays["raw_q"], np.float64)
        wfc = np.asarray(arrays["raw_wfc"], np.float64)
        lo = np.asarray(arrays["joint_lo"], np.float64)
        hi = np.asarray(arrays["joint_hi"], np.float64)
        mean = np.asarray(arrays["root_mean"], np.float64)
        std = np.asarray(arrays["root_std"], np.float64)
        independent = independent_encode(root, q, wfc, lo, hi, mean, std)
        encode_error = float(np.max(np.abs(
            independent.astype(np.float32).astype(np.float64)
            - x0.astype(np.float64))))
        if encode_error > 1e-6:
            problems.append(f"input_encode_error:{encode_error:.6g}")
        production = np.stack([
            self.tokenizer.encode(root[i], q[i], wfc[i]) for i in range(4)
        ]).astype(np.float32)
        reproduction = float(np.max(np.abs(
            production.astype(np.float64) - x0.astype(np.float64))))
        if reproduction > 1e-6:
            problems.append(f"input_tokenizer_reproduction:{reproduction:.6g}")
        x_t_hashes = {}
        x_t_max = 0.0
        if self.noise:
            for unit in self.noise:
                slots = MICRO_SCHEDULE[unit["update"]][unit["micro"]]
                batch_x0 = np.stack([self.window_arrays[w]["x0"]
                                     for w in slots], axis=0)
                x_t = independent_x_t(batch_x0, unit["epsilon"].numpy(),
                                      unit["t"].numpy(), effective)
                if not np.isfinite(x_t).all():
                    problems.append(
                        f"input_x_t_nonfinite:{unit['update']}:{unit['micro']}")
                x_t_hashes[
                    f"u{unit['update']}m{unit['micro']}"] = _hash_array(x_t)
                x_t_max = max(x_t_max, float(np.max(np.abs(x_t))))
        else:
            problems.append("input_noise_schedule_missing")
        return {
            "schema": "w1_real_training_update_input_admission_v1",
            "mode": "real",
            "passed": not problems,
            "problems": problems,
            "encode_max_abs": encode_error,
            "tokenizer_reproduction_max_abs": reproduction,
            "alpha_cumprod_sha256": _hash_array(alpha),
            "alpha_effective_sha256": _hash_array(
                np.asarray(effective, np.float64)),
            "alpha_evaluation": ALPHA_EVALUATION,
            "x_t_reference_sha256": x_t_hashes,
            "x_t_reference_max_abs": x_t_max,
            "noise_units": len(self.noise),
            "window_ids": list(WINDOW_IDS),
            "source": path_for_record(
                self.root / PARENT_FILES["objective_arrays"][0]),
            "source_sha256": self.source_sha256,
            "read_scope": {"npz_values_read": len(WINDOW_VALUE_KEYS),
                           "checkpoint_payload_read": 0,
                           "model_constructor_calls": 0,
                           "forward_calls": 0},
            "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def materialize(self, *, stage: str, r0_noise: Sequence[dict] | None,
                    counter: CallCounter | None = None,
                    parent_input_admission: Mapping[str, Any] | None = None,
                    guard: Callable[[str], None] | None = None,
                    deadline: float | None = None,
                    registered: bool = False,
                    r0_registration_sha: str | None = None) -> dict:
        if stage not in ("P", "I", "F-P", "F-I"):
            raise ValueError(f"unknown materialized stage {stage}")
        self.reset_materialization()
        counter = counter if counter is not None else self.counter
        if counter is None:
            raise ValueError("materialize requires a bound counter")
        self._guard = guard
        self._deadline = None if deadline is None else float(deadline)
        self._check("materialize_start")
        if isinstance(self.acceptance, Mapping):
            identity = runtime_identity_problems(
                self.acceptance.get("runtime_identity"))
            if identity:
                raise ValueError(
                    f"runtime identity drifted since Gate B: {identity}")
        if r0_noise is None:
            raise ValueError("materialize requires the R0 noise schedule")
        self.noise = list(r0_noise)
        arrays = self._load_arrays()
        self._verify_asset_closure()
        self._check("materialize_tokenizer")
        self._build_tokenizer(arrays)
        problems = _window_values_problems(arrays, self.constants or {})
        if problems:
            self.problems = problems
            raise ValueError(f"real window input refused: {problems}")
        self.window_arrays = {}
        for index, key in enumerate(WINDOW_IDS):
            self.window_arrays[key] = {
                "x0": np.asarray(arrays["x0"][index], np.float32),
                "kps_obj": np.asarray(arrays["kps_obj"][index], np.float32),
                "kps_clip": np.asarray(arrays["kps_clip"][index], np.float32),
                "box_t_clip": np.asarray(arrays["box_t_clip"][index],
                                         np.float32),
                "box_quat": np.asarray(arrays["box_quat"][index], np.float32),
                "contact": np.asarray(arrays["contact"][index], np.float32),
                "wfc": np.asarray(arrays["raw_wfc"][index], np.float64).astype(
                    np.float32),
            }
        self.alpha = np.asarray(arrays["alpha_cumprod"], np.float64)
        self._check("materialize_input_admission")
        admission = self._input_admission(arrays, counter=counter)
        admission["r0_registration_sha256"] = r0_registration_sha
        self.input_admission = admission
        if isinstance(parent_input_admission, Mapping):
            binding = [
                "source_sha256", "alpha_cumprod_sha256",
                "alpha_effective_sha256", "alpha_evaluation",
                "x_t_reference_sha256", "noise_units",
                "r0_registration_sha256",
            ]
            for key in binding:
                if _jsonable(parent_input_admission.get(key)) != \
                        _jsonable(admission.get(key)):
                    raise ValueError(f"input admission drift: {key}")
        elif stage in ("I", "F-I"):
            raise ValueError("independent stage requires the P input admission")
        self.materialized_stage = stage
        self._check("materialize_reference_backend")
        self.reference = RealReferenceBackend(
            counter, deadline=self._deadline, guard=self._guard,
            joint_lo=np.asarray(arrays["joint_lo"], np.float64),
            joint_hi=np.asarray(arrays["joint_hi"], np.float64))
        admission["stage"] = stage
        admission["registered"] = bool(registered)
        return admission

    def load_noise_schedule(self, path: Path) -> list[dict]:
        data = np.load(path)
        schedule: list[dict] = []
        for u in UPDATES:
            for m in range(MICRO_COUNT):
                tag = f"u{u}m{m}"
                schedule.append({
                    "update": u, "micro": m,
                    "t": torch.as_tensor(data[f"{tag}__t"].copy()),
                    "epsilon": torch.as_tensor(
                        data[f"{tag}__epsilon"].copy()),
                    "generator_state_before": torch.as_tensor(
                        data[f"{tag}__state_before"].copy()),
                    "generator_state_after": torch.as_tensor(
                        data[f"{tag}__state_after"].copy()),
                })
        return schedule

    def write_noise_schedule(self, path: Path) -> str:
        arrays: dict[str, np.ndarray] = {}
        for unit in self.noise:
            tag = f"u{unit['update']}m{unit['micro']}"
            arrays[f"{tag}__t"] = unit["t"].numpy()
            arrays[f"{tag}__epsilon"] = unit["epsilon"].numpy()
            arrays[f"{tag}__state_before"] = \
                unit["generator_state_before"].numpy()
            arrays[f"{tag}__state_after"] = unit["generator_state_after"].numpy()
        np.savez(path, **arrays)
        return sha256_file(path)

    # --- stage API --------------------------------------------------------

    def micro_batch(self, update: int, micro: int) -> dict:
        slots = MICRO_SCHEDULE[update][micro]
        out: dict[str, torch.Tensor] = {}
        for name in ("x0", "kps_obj", "kps_clip", "box_t_clip", "box_quat",
                     "contact", "wfc"):
            out[name] = torch.stack(
                [torch.tensor(self.window_arrays[w][name],
                              dtype=torch.float32) for w in slots]).to(
                self.device)
        return out

    def _production_loss(self, counter: CallCounter) -> LossAssembler:
        if self._production_assembler is None:
            from ireset.geometry.distance_adapter import largebox_static_frame
            from ireset.kinematics.torch_adapter import G1TorchFK

            self._check("production_fk_construction")
            fk = G1TorchFK(device=self.device, dtype=torch.float32)

            def scene_landed() -> None:
                if not self._scene_setup_counted:
                    counter.add("scene_setup_mj_forward", 1)
                    self._scene_setup_counted = True

            def scene_check(unit: str) -> None:
                self._check(f"production_{unit}")

            # The scene compile is an independent real-asset operation; a
            # deadline crossed during FK construction must stop it here.
            self._check("production_scene_setup")
            frame = largebox_static_frame(observer=scene_landed,
                                          check=scene_check)
            self._check("production_scene_ready")
            assembler = LossAssembler(
                self.tokenizer, LOSS_WEIGHTS, self.device,
                fk=CountingFk(_GuardedFk(fk, self), counter),
                static_frame=frame)
            self._production_assembler = assembler
        return self._production_assembler

    def _load_checkpoint(self, model_type: str,
                         counter: CallCounter) -> dict:
        self._check("checkpoint_sha")
        rel, want = CHECKPOINTS[model_type]
        path = self.root / rel
        actual = sha256_file(path) if path.is_file() else None
        if actual != want:
            raise ValueError(f"checkpoint drifted: {model_type}")
        # Trusted local checkpoint whose bytes were verified above.  The
        # payload embeds numpy root statistics, so weights_only=True cannot
        # decode it; no untrusted file reaches this call.
        self._check("checkpoint_load")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        counter.add("checkpoint_loads", 1)
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint payload is not a mapping: {model_type}")
        missing = [key for key in CHECKPOINT_REQUIRED_KEYS
                   if key not in payload]
        if missing:
            raise ValueError(f"checkpoint payload missing: {missing}")
        if payload.get("step") != CHECKPOINT_STEP:
            raise ValueError(f"checkpoint step: {payload.get('step')}")
        if payload.get("parameters_finite") is not True:
            raise ValueError("checkpoint parameters are not finite")
        if payload.get("checkpoint_usable") is not True \
                or payload.get("diagnostic_only") is not False \
                or payload.get("failure") is not None:
            raise ValueError("checkpoint is not usable")
        stats = payload.get("root_stats") or {}
        mean = np.asarray(stats.get("mean"), np.float64)
        std = np.asarray(stats.get("std"), np.float64)
        if not (np.array_equal(mean, self.tokenizer.root_stats.mean)
                and np.array_equal(std, self.tokenizer.root_stats.std)):
            raise ValueError("checkpoint root statistics differ")
        identity = payload.get("identity") or {}
        if identity.get("model_type") != model_type:
            raise ValueError("checkpoint identity model_type differs")
        config = payload.get("config") or {}
        for key in ("model_type", "micro_batch", "grad_accum", "lr",
                    "weight_decay", "grad_clip", "d_model", "n_layers",
                    "n_heads", "ffn_mult", "time_window",
                    "num_train_timesteps", "beta_schedule", "beta_start",
                    "beta_end"):
            if key == "model_type":
                expected = model_type
            elif key in ("lr", "weight_decay", "grad_clip", "beta_start",
                         "beta_end"):
                expected = float(MODEL_CONFIG[key])
            else:
                expected = int(MODEL_CONFIG[key]) if isinstance(
                    MODEL_CONFIG[key], int) else MODEL_CONFIG[key]
            if config.get(key) != expected:
                raise ValueError(f"checkpoint config differs: {key}")
        return payload

    def build_trainer(self, model_type: str, *, with_optimizer: bool,
                      counter: CallCounter, loss_mode: str = "production",
                      out_dir: Path | None = None) -> Trainer:
        if self.tokenizer is None:
            raise ValueError("provider is not materialized")
        self._check("trainer_construction")
        cfg = TrainConfig(
            model_type=model_type,
            steps=len(UPDATES),
            micro_batch=int(self.config["micro_batch"]),
            grad_accum=int(self.config["grad_accum"]),
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
            grad_clip=float(self.config["grad_clip"]),
            seed=NOISE_SEED,
            num_workers=0, val_every=10 ** 9, log_every=10 ** 9,
            device=self.device, val_seed=NOISE_SEED,
            d_model=int(self.config["d_model"]),
            n_layers=int(self.config["n_layers"]),
            n_heads=int(self.config["n_heads"]),
            ffn_mult=int(self.config["ffn_mult"]),
            time_window=int(self.config["time_window"]),
            num_train_timesteps=int(self.config["num_train_timesteps"]),
            beta_schedule=str(self.config["beta_schedule"]),
            beta_start=float(self.config["beta_start"]),
            beta_end=float(self.config["beta_end"]),
            loss=LOSS_WEIGHTS,
        )
        if loss_mode == "production":
            loss = CountingLoss(self._production_loss(counter), counter)
        elif loss_mode == "forbidden":
            loss = ForbiddenLoss()
        else:
            raise ValueError(f"unknown loss_mode {loss_mode}")
        out = Path(out_dir or (self.root / "_trainer_tmp"))
        out.mkdir(parents=True, exist_ok=True)

        def construction(unit: str) -> None:
            if unit == "model":
                counter.add("model_constructions", 1)
                self._check("trainer_model_constructed")
            elif unit == "conditioner":
                counter.add("conditioner_constructions", 1)
                self._check("trainer_conditioner_constructed")
            elif unit == "optimizer" and with_optimizer:
                counter.add("native_optimizer_constructions", 1)
                self._check("trainer_optimizer_constructed")

        self._check("trainer_construction")
        with torch.random.fork_rng(devices=[]):
            trainer = Trainer(
                cfg, self.tokenizer, _PlaceholderDataset(),
                _PlaceholderDataset(), out, loss_fn=loss,
                build_optimizer=with_optimizer,
                construction_observer=construction)
        self._check("trainer_state_dict")
        payload = self._load_checkpoint(model_type, counter)
        self._check("trainer_state_copy")
        # load_state_dict(strict=True) still casts a float64 payload into the
        # float32 parameters, so every tensor is checked before the copy.
        _copy_state_strict(trainer.model, payload["model"], "model")
        _copy_state_strict(trainer.conditioner, payload["conditioner"],
                           "conditioner")
        if not all(bool(torch.isfinite(p).all())
                   for p in (*trainer.model.parameters(),
                             *trainer.conditioner.parameters())):
            raise ValueError("checkpoint contains non-finite parameters")
        if model_type == "diffusion":
            self._check("trainer_alpha_schedule")
            verify_production_alpha(trainer.model.ddpm.alphas_cumprod,
                                    self.config)
        trainer.model.train(); trainer.conditioner.train()
        return trainer

    # --- reference API ----------------------------------------------------

    def reference_positions(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise ValueError("reference backend is not materialized")
        return self.reference.positions(pos, R, q)

    def reference_jacobians(self, pos: np.ndarray, R: np.ndarray,
                            q: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise ValueError("reference backend is not materialized")
        return self.reference.jacobians(pos, R, q)


class _GuardedFk:
    """Forward the real FK and re-check the stage deadline before each batch."""

    def __init__(self, inner: Any, owner: RealTrainingProvider) -> None:
        self.inner = inner
        self.owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def site_world_positions(self, *args: Any, **kwargs: Any) -> Any:
        if self.owner._guard is not None:
            self.owner._guard("production_fk")
        if self.owner._deadline is not None and \
                time.perf_counter() >= self.owner._deadline:
            raise TimeoutError("production_fk_deadline")
        return self.inner.site_world_positions(*args, **kwargs)


def _asset_binding_problems(actual: Mapping[str, Any],
                            expected: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(expected, Mapping) or not expected.get("checked"):
        return ["asset_binding_missing"]
    if expected.get("problems"):
        problems.append("asset_binding_parent_problems")
    locked_expected = expected.get("locked") or {}
    locked_actual = actual.get("locked") or {}
    for name, rec in locked_expected.items():
        got = locked_actual.get(name) or {}
        if got.get("sha256") != rec.get("sha256"):
            problems.append(f"asset_locked_sha:{name}")
    expected_refs = expected.get("references") or {}
    actual_refs = actual.get("references") or {}
    if set(expected_refs) != set(actual_refs):
        problems.append("asset_reference_set")
    for key, rec in expected_refs.items():
        got = actual_refs.get(key) or {}
        if got.get("sha256") != rec.get("sha256"):
            problems.append(f"asset_reference_sha:{Path(key).name}")
        if got.get("tag") != rec.get("tag"):
            problems.append(f"asset_reference_tag:{Path(key).name}")
        if got.get("from") != rec.get("from"):
            problems.append(f"asset_reference_from:{Path(key).name}")
    if actual.get("closure_sha256") != expected.get("closure_sha256"):
        problems.append("asset_closure_sha")
    if int(actual.get("reference_count", -1)) != int(
            expected.get("reference_count", -2)):
        problems.append("asset_reference_count")
    return problems


# ---------------------------------------------------------------------------
# real admission package (metadata only)
# ---------------------------------------------------------------------------


def real_admission_package_problems(
        package: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(package, Mapping):
        return ["admission_not_a_mapping"]
    problems: list[str] = []
    if package.get("schema") != REAL_ADMISSION_SCHEMA:
        problems.append("admission_schema")
    if package.get("provider_mode") != "real":
        problems.append("admission_provider_mode")
    if package.get("scope") != "metadata_only":
        problems.append("admission_scope")
    if package.get("passed") is not True:
        problems.append("admission_not_passed")
    if package.get("problems"):
        problems.append("admission_has_problems")
    status = package.get("status") or {}
    if status.get("authorized") is not False:
        problems.append("admission_must_be_pre_authorization")
    if status.get("registered") is not False:
        problems.append("admission_must_not_be_registered")
    if status.get("run") != "not_run":
        problems.append("admission_must_not_be_run")
    if status.get("section_10_2_checked") is not False:
        problems.append("admission_section_10_2")
    scope = package.get("read_scope") or {}
    if scope.get("metadata_only") is not True:
        problems.append("admission_read_scope:metadata_only")
    for key in ("npz_values_read", "checkpoint_payload_read",
                "torch_load_calls", "model_constructor_calls",
                "muJoCo_compile_calls", "conditioner_forward_calls",
                "model_forward_calls", "vjp_calls", "optimizer_steps"):
        if scope.get(key) != 0:
            problems.append(f"admission_read_scope:{key}")
    for key in ("cohort_created", "registered_attempt_created"):
        if scope.get(key) is not False:
            problems.append(f"admission_read_scope:{key}")
    if scope.get("asset_closure_verified") is not True:
        problems.append("admission_asset_closure")
    for name, rec in (package.get("parent_anchors") or {}).items():
        if not isinstance(rec, Mapping) or rec.get("match") is not True:
            problems.append(f"admission_parent_anchor:{name}")
    for model, rec in (package.get("checkpoints") or {}).items():
        if not isinstance(rec, Mapping) or rec.get("match") is not True:
            problems.append(f"admission_checkpoint:{model}")
    if (package.get("config") or {}).get("sha256") != TRAIN_CONFIG_SHA:
        problems.append("admission_train_config")
    assets = package.get("assets") or {}
    if assets.get("problems"):
        problems.append("admission_assets_problems")
    if int(assets.get("reference_count", -1)) < 1:
        problems.append("admission_assets_empty")
    roster = package.get("roster") or {}
    if roster.get("row_formula") != "14N+168" or not roster.get("sha256"):
        problems.append("admission_roster")
    tolerance = package.get("tolerance") or {}
    if tolerance.get("mode") != "synthetic_calibrated" \
            or not tolerance.get("evidence_sha256"):
        problems.append("admission_tolerance_not_calibrated")
    contract = tolerance.get("contract") or {}
    if (contract.get("fp32_contract") or {}).get("valid") is not True:
        problems.append("admission_fp32")
    if contract.get("fp32_contract_sha256") != package.get(
            "fp32_contract_sha256"):
        problems.append("admission_fp32_binding")
    problems += [f"admission_calibration:{p}" for p in
                 calibration_verification_problems(
                     package.get("calibration_verification"))]
    calibration_sha = package.get("calibration_verification_sha256")
    if not isinstance(calibration_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", calibration_sha):
        problems.append("admission_calibration_binding")
    elif calibration_sha != _canonical_sha256(
            dict(package.get("calibration_verification") or {})):
        problems.append("admission_calibration_digest")
    problems += [f"admission_acceptance:{p}" for p in
                 synthetic_acceptance_problems(package.get("acceptance"))]
    impl = package.get("implementation") or {}
    for rel, sha in (impl.get("files") or {}).items():
        if not isinstance(sha, str):
            problems.append(f"admission_implementation_missing:{rel}")
    runtime = package.get("runtime")
    if not isinstance(runtime, Mapping):
        problems.append("admission_runtime_missing")
    else:
        if runtime.get("applied") is not True:
            problems.append("admission_runtime_not_applied")
        if runtime.get("observed_after_setup") != runtime.get(
                "requirements"):
            problems.append("admission_runtime_observation")
    problems += [f"admission_runtime_identity:{p}"
                 for p in runtime_identity_problems(
                     package.get("runtime_identity"))]
    chain = package.get("cohort", {}).get("full_chain_calls") or {}
    real_chain = dict(CHAIN_NOMINAL)
    if chain != real_chain:
        problems.append("admission_full_chain_matrix")
    return problems


def build_real_admission_package(        project_root: Path, *, tolerance_contract: Mapping[str, Any],
        tolerance_contract_path: Path | None,
        tolerance_contract_sha256: str | None,
        calibration_verification_record: Mapping[str, Any],
        acceptance_path: Path, acceptance: Mapping[str, Any]) -> dict:
    root = Path(project_root)
    problems: list[str] = []
    leaf = acceptance_leaf_record(acceptance)
    leaves = [leaf]
    if isinstance(acceptance, Mapping) and acceptance.get("schema") == \
            SYNTHETIC_ACCEPTANCE_SCHEMA_V2:
        migration = acceptance.get("migration") or {}
        migration_leaf = migration.get("record") \
            if isinstance(migration, Mapping) else None
        if isinstance(migration_leaf, Mapping):
            leaves.append(migration_leaf)
    provider = RealTrainingProvider(
        root, MODEL_CONFIG, device="cuda", acceptance_path=Path(acceptance_path))
    try:
        provider.acceptance = dict(acceptance)
        provider.acceptance_sha = sha256_file(Path(acceptance_path))
        provider.roster = copy.deepcopy(
            (leaf.get("roster") or {}).get("parameter_roster") or {})
    except (OSError, ValueError, TypeError):
        problems.append("admission_acceptance_unreadable")
    metadata = provider.metadata()
    problems += [f"metadata:{p}" for p in metadata.get("problems") or []]
    for problem in synthetic_acceptance_problems(acceptance):
        problems.append(f"acceptance:{problem}")
    for problem in calibration_verification_problems(
            calibration_verification_record):
        problems.append(f"calibration:{problem}")
    from ireset.eval.w1_real_training_update_audit import (
        tolerance_contract_problems,
    )
    tolerance_problems = tolerance_contract_problems(
        tolerance_contract, recompute=False)
    problems += [f"tolerance:{p}" for p in tolerance_problems]
    if tolerance_contract.get("mode") != "synthetic_calibrated":
        problems.append("tolerance_mode")
    fp32 = tolerance_contract.get("fp32_contract") or {}
    if fp32.get("valid") is not True:
        problems.append("fp32_not_valid")
    fp32_sha = tolerance_contract.get("fp32_contract_sha256")
    if fp32_sha != calibration_verification_record.get(
            "fp32_contract_sha256"):
        problems.append("fp32_calibration_binding")
    tolerance_evidence = tolerance_contract.get("evidence_sha256")
    if tolerance_evidence != calibration_verification_record.get(
            "evidence_sha256"):
        problems.append("tolerance_calibration_binding")
    for child in leaves:
        acceptance_tolerance = child.get("tolerance") or {}
        if acceptance_tolerance.get("evidence_sha256") != tolerance_evidence:
            problems.append("acceptance_tolerance_binding")
        if acceptance_tolerance.get("fp32_contract_sha256") != fp32_sha:
            problems.append("acceptance_fp32_binding")
    runtime_identity = runtime_identity_facts()
    problems += [f"runtime_identity:{p}" for p in runtime_identity_problems(
        fp32.get("runtime_identity"), runtime_identity)]
    for child in leaves:
        problems += [f"acceptance_runtime_identity:{p}"
                     for p in runtime_identity_problems(
                         child.get("runtime_identity"), runtime_identity)]
    problems += [f"calibration_runtime_identity:{p}"
                 for p in runtime_identity_problems(
                     calibration_verification_record.get("runtime_identity"),
                     runtime_identity)]
    impl = implementation_identity(root)
    missing = [rel for rel, sha in impl["files"].items()
               if not isinstance(sha, str)]
    if missing:
        problems.append(f"implementation_missing:{missing}")
    for child in leaves:
        acceptance_impl = child.get("implementation") or {}
        if acceptance_impl.get("files") != impl["files"]:
            problems.append("acceptance_implementation_drift")
    runtime = configure_production_runtime()
    problems += [f"runtime_apply:{p}" for p in runtime["problems"]]
    roster = provider.roster or {}
    catalog = row_catalog(roster) if roster else {}
    chain = dict(CHAIN_NOMINAL)
    for kind, value in chain.items():
        if value > int(CALL_CAPS[kind]):
            problems.append(f"full_chain_call_cap:{kind}")
    package = {
        "schema": REAL_ADMISSION_SCHEMA,
        "provider_mode": "real",
        "scope": "metadata_only",
        "status": {
            "mode": "pre_authorization_readonly",
            "authorized": False, "registered": False, "run": "not_run",
            "package_passed": not problems,
            "section_10_2_checked": False,
        },
        "passed": not problems,
        "problems": problems,
        "cohort": {
            "cohort_id": COHORT_ID, "execution_id": EXECUTION_ID,
            "history_root": "derived/experiments/omniretarget_first",
            "stage_budgets": {k: dict(v) for k, v in STAGE_BUDGETS.items()},
            "total_budget_s": sum(v["budget_s"]
                                  for v in STAGE_BUDGETS.values()),
            "call_caps": dict(CALL_CAPS),
            "full_chain_calls": chain,
            "scene_setup_by_mode": {
                "real": {"R0": 0, "P": 1, "I": 0, "F": 1, "chain": 2},
                "synthetic": {"R0": 0, "P": 0, "I": 0, "F": 0, "chain": 0},
            },
            "updates": {"diagnostic_parameter_update_events": 16,
                        "cohort_hard_cap": 32},
            "budget_transfer_from_old_runs": False,
        },
        "provider": {
            "schema": PROVIDER_SCHEMA,
            "mode": "real",
            "lifecycle": "metadata",
            "identity": provider.spec(),
            "metadata": metadata,
        },
        "windows": metadata.get("windows"),
        "window_headers": metadata.get("window_headers"),
        "roster": {
            "sha256": roster.get("sha256"), "n_det": roster.get("n_det"),
            "n_diff": roster.get("n_diff"),
            "n_total": roster.get("n_total"),
            "n_rows": catalog.get("n_rows"), "row_formula": "14N+168",
            "row_keys_sha256": catalog.get("keys_sha256"),
            "parameter_roster": copy.deepcopy(dict(roster)),
        },
        "tolerance": {
            "path": (path_for_record(tolerance_contract_path)
                     if tolerance_contract_path is not None else None),
            "sha256": tolerance_contract_sha256,
            "mode": tolerance_contract.get("mode"),
            "evidence_sha256": tolerance_evidence,
            "fp32_contract_sha256": fp32_sha,
            "contract": copy.deepcopy(dict(tolerance_contract)),
        },
        "fp32_contract_sha256": fp32_sha,
        "calibration_verification": copy.deepcopy(
            dict(calibration_verification_record)),
        "calibration_verification_sha256": _canonical_sha256(
            dict(calibration_verification_record)),
        "acceptance": copy.deepcopy(dict(acceptance)),
        "acceptance_path": path_for_record(Path(acceptance_path)),
        "acceptance_sha256": sha256_file(Path(acceptance_path)),
        "parent_anchors": metadata.get("parents"),
        "parent_consumers": metadata.get("parent_consumers"),
        "checkpoints": metadata.get("checkpoints"),
        "config": metadata.get("train_config"),
        "assets": metadata.get("assets"),
        "resource_caps": {"gpu_peak_gib": 22, "host_peak_gib": 32,
                          "artifacts_gib": 16, "disk_free_min_gib": 20},
        "implementation": impl,
        "environment": metadata.get("environment"),
        "runtime_identity": runtime_identity,
        "runtime": {
            "requirements": RUNTIME_REQUIREMENTS,
            "observed_after_setup": dict(runtime["after"]),
            "applied": runtime["applied"],
            "problems": list(runtime["problems"]),
        },
        "read_scope": {
            **dict(metadata.get("read_scope") or {}),
            "asset_closure_verified": bool(
                (metadata.get("assets") or {}).get("checked")),
            "conditioner_forward_calls": 0,
            "model_forward_calls": 0,
            "vjp_calls": 0,
            "optimizer_steps": 0,
        },
        "r0_obligations": {
            "parent_consumers": "parent disk x index x ledger verified above",
            "real_inputs": ("materialization loads the four parent windows, "
                            "re-encodes and writes input_admission before any "
                            "network forward"),
            "asset_closure": "locked bytes/text verified; compile deferred",
            "call_accounting": ("reservations before work; unsettled worker "
                                "leaves history unknown"),
            "f_process": "fresh spawned Python subprocess re-reads the disk",
            "runtime": "TF32 off / matmul highest / autocast off",
        },
        "coverage": {
            "objective_coverage": "partial",
            "audited": [
                "real_provider_metadata", "window_headers", "checkpoint_sha",
                "asset_closure_hashes", "parent_consumer_states",
                "synthetic_acceptance_binding",
            ],
            "not_audited": [
                "real_checkpoint_payload", "real_window_values",
                "real_forward_backward", "real_fk", "real_adamw",
            ],
        },
        "notes": [
            "metadata-only admission: no checkpoint payload, NPZ value, "
            "model/optimizer construction or MuJoCo compile happened",
            "authorization remains a separate user-provided record",
        ],
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return package


def write_real_package(out_root: Path, package: Mapping[str, Any]
                       ) -> tuple[Path, str]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "index.json"
    rows = []
    if index_path.is_file():
        try:
            data = read_json(index_path)
            rows = [r for r in (data.get("attempts") or [])
                    if isinstance(r, dict)]
        except (OSError, ValueError, TypeError):
            rows = []
    numbers = [int(r["attempt"].split("_")[1]) for r in rows
               if isinstance(r.get("attempt"), str)
               and re.fullmatch(r"attempt_\d+", r["attempt"])]
    attempt = f"attempt_{max(numbers, default=0) + 1:03d}"
    directory = out_root / attempt
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "admission_package.json"
    write_json(path, package)
    sha = sha256_file(path)
    rows.append({"attempt": attempt, "path": path_for_record(path),
                 "sha256": sha, "passed": bool(package.get("passed"))})
    write_json(index_path, {"attempts": rows})
    return path, sha
