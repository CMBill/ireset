"""Versioned, D1-only LR diagnostic. No real computation is authorized here.

The native Trainer arithmetic and the legacy constant-LR entry are unchanged.
Only this driver admits the registered schedule, persisting the common prefix
before update 501. All comparison work is CPU/file-only and F-billed.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import statistics
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from ireset.eval import w1_smallset_fit_diagnostic as fit
from ireset.eval import w1_d1_lr_protocol as protocol
from ireset.eval.w1_n0_coordinate_flags import npz_members, sha256_file
from ireset.paths import PROJECT_ROOT

SCHEDULE_ID = "d1_half_after_500_v1"
COHORT, RUN_ROOT, AUDIT_ROOT = protocol.COHORT, protocol.RUN_ROOT, protocol.AUDIT_ROOT
PLAN_PATH = "docs/plans/omniretarget_w1_d1_lr_schedule_diagnostic_plan.md"
LR_ARTIFACTS = ("lr_evidence.json", "lr_history.jsonl", "prefix_checks.json", "prefix_admission.json")
BASELINE_ROOT = "derived/experiments/omniretarget_first/attempt_002"
BASELINE_AUDIT = ("derived/audit/w1_smallset_fit_diagnostic/"
                  "real_chain_20261118_attempt_002")
BASELINE_ANCHORS = {
    "acceptance_verdict.json": "39d2bb6f3dcf66609c932ea09a4b9d3c7bccd7265dc996f5e80fe923d6b5804e",
    "d1/stage_report.json": "47c42309fc50460c20460a9c8d8f7be194bbd07772868f0c8cb4b20725d24096",
    "d1/artifacts_index.json": "1c48cc5bb0230cc4a6a0b841a20c0eac946983041eacd6c0733d82c8dfbebd4d",
    "f/f_report.json": "71e4e81366fed38840b5db8c772a3969558a38bdd79d8ddeadd2dc256ad488ac",
}
COMPLETION_SHA = "4636efd68a5cda39e37a74ac2ff58837bf7f0e6b91f32734717f5a7db50fe9f2"
CACHE_SHA = "da4bb29235dcdc2401df3ad58ebc51c7502a8187ccc3d8dacdcb0a01e5f2f810"
GATES = {"root_position_rmse_m": fit.GATE_ROOT_POS_RMSE_M,
         "root_rotation_mean_rad": fit.GATE_ROOT_ROT_MEAN_RAD,
         "joint_rmse_rad": fit.GATE_JOINT_RMSE_RAD,
         "teacher_fk_mpjpe_m": fit.GATE_FK_MPJPE_M}
LR_FIELDS = {"lr_schedule_id", "optimizer_update", "effective_lrs",
             "state_step_before", "state_step_after", "optimizer_state_sha256_before",
             "optimizer_state_sha256_after"}
RULES = {"primary_endpoint": 1000, "relative_improvement_multiplier": 0.9,
         "gates": GATES, "fk_replay_scope": fit.FK_REPLAY_SCOPE,
         "spike_gate_status": fit.SPIKE_GATE_STATUS, "auto_execute": False,
         "production_integration": False, "d6_authorized": False}


def _finite_number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def read_object(path: Path) -> dict:
    doc = json.loads(Path(path).read_bytes())
    if not isinstance(doc, dict):
        raise fit.FitDiagnosticError(f"json_not_object:{path}")
    return doc


def write_json(path: Path, doc: dict) -> None:
    """Atomic, durable publication; temporary files count toward the cap."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as out:
        json.dump(doc, out, indent=1, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _persist_npz(path, records):
    """Preserve the baseline's byte format and sync it before issuing receipts."""
    fit.write_npz_records(path, records)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def file_record(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def probe_name(idx: int, step: int) -> str:
    return f"probes/probe_{idx:04d}_step_{step:06d}.npz"


@dataclass(frozen=True)
class SchedulePlan(fit.FitPlan):
    schedule_id: str = SCHEDULE_ID
    switch_after: int = 500
    multiplier: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if (self.stage != "D1" or self.model_type != "deterministic"
                or self.schedule_id != SCHEDULE_ID
                or type(self.switch_after) is not int
                or not 0 < self.switch_after < self.steps
                or self.switch_after % self.probe_every
                or type(self.multiplier) is not float or self.multiplier != 0.5
                or (not self.synthetic and self.switch_after != 500)):
            raise fit.FitDiagnosticError("lr_schedule_plan_invalid")

    def lr_at(self, update: int) -> float:
        if type(update) is not int or not 0 <= update <= self.steps:
            raise fit.FitDiagnosticError("lr_update_number_invalid")
        return self.lr if update <= self.switch_after else self.lr * self.multiplier

    def constant_plan(self) -> fit.FitPlan:
        return fit.FitPlan(**{f.name: getattr(self, f.name) for f in fields(fit.FitPlan)})

    def schedule(self) -> dict:
        return {"id": self.schedule_id, "base_lr": self.lr,
                "switch_after": self.switch_after, "multiplier": self.multiplier,
                "switch_before_first_micro": True, "reset_optimizer": False,
                "weight_decay": self.weight_decay,
                "endpoint_effective_lr": {"initial": self.lr_at(0),
                                           "last": self.lr_at(self.steps)}}


def _validate_probe_index(doc: dict, plan: fit.FitPlan) -> None:
    probes, history = doc.get("probes"), doc.get("history_unrounded")
    if (not isinstance(probes, list) or not isinstance(history, list)
            or len(history) != plan.steps or len(probes) != plan.n_probes
            or any(not isinstance(r, dict) for r in [*probes, *history])
            or [r.get("step") for r in probes] != list(plan.probe_steps)
            or [r.get("idx") for r in probes] != list(range(plan.n_probes))
            or [r.get("step") for r in history] != list(range(1, plan.steps + 1))):
        raise fit.FitDiagnosticError("probe_or_history_coverage_invalid")
    for row in history:
        if (type(row["step"]) is not int or row.get("n_micro") != plan.grad_accum
                or row.get("n_samples") != plan.grad_accum
                or type(row.get("n_micro")) is not int
                or type(row.get("n_samples")) is not int
                or not set(fit.LOSS_WEIGHT_EXPECTED).isdisjoint(row)):
            raise fit.FitDiagnosticError("history_counts_or_fields_invalid")
        required = {"step", "n_micro", "n_samples", "pos", "rot6d", "joint",
                    "geodesic_rad2", "vel", "fk_m2", "total", "gnorm"}
        if not required <= row.keys() or any(
                not _finite_number(row[k]) for k in required):
            raise fit.FitDiagnosticError("history_value_invalid")
    for probe in probes:
        if type(probe["step"]) is not int or type(probe["idx"]) is not int:
            raise fit.FitDiagnosticError("probe_number_type_invalid")
        windows = probe.get("windows")
        if (not isinstance(windows, list) or len(windows) != 1
                or not isinstance(windows[0], dict)
                or not isinstance(windows[0].get("metrics"), dict)):
            raise fit.FitDiagnosticError("probe_single_window_required")
        metrics = windows[0]["metrics"]
        if any(not _finite_number(metrics.get(k)) or metrics[k] < 0 for k in GATES):
            raise fit.FitDiagnosticError("probe_metrics_invalid")


def freeze_baseline(root: Path, plan: SchedulePlan, *, fixture: bool = False,
                    check=lambda _point: None) -> dict:
    """JSON, byte hashes and headers only. Never torch.load/array decode."""
    root = Path(root).resolve()
    if not fixture and (root != (PROJECT_ROOT / BASELINE_ROOT).resolve()
                        or plan.synthetic):
        raise fit.FitDiagnosticError("baseline_fixed_root_required")
    records = {}

    def bind(path, expected=None):
        check("baseline_hash:" + str(path))
        rec = file_record(path)
        if expected is not None and rec["sha256"] != expected:
            raise fit.FitDiagnosticError(f"baseline_anchor_drift:{path.name}")
        records[str(path.resolve())] = rec
        return rec

    for rel, sha in BASELINE_ANCHORS.items():
        if not fixture:
            bind(root / rel, sha)
    pidx = read_object(root / "d1/probe_index.json")
    _validate_probe_index(pidx, plan.constant_plan())
    if not fit._json_value_equal(pidx.get("plan"), plan.constant_plan().as_dict()):
        raise fit.FitDiagnosticError("baseline_plan_mismatch")
    index = read_object(root / "d1/artifacts_index.json")
    required = fit._required_stage_artifacts(plan.constant_plan())
    indexed = index.get("files")
    if not isinstance(indexed, dict) or not set(required) <= indexed.keys():
        raise fit.FitDiagnosticError("baseline_artifacts_missing")
    bind(root / "d1/artifacts_index.json")
    for rel, rec in indexed.items():
        path = (root / "d1" / rel).resolve()
        if not path.is_relative_to(root / "d1") or file_record(path) != rec:
            raise fit.FitDiagnosticError(f"baseline_artifact_drift:{rel}")
        bind(path, rec["sha256"])
        if path.suffix == ".npz":
            check("baseline_npz_header:" + path.name)
            npz_members(path)
    if not fixture:
        audit = PROJECT_ROOT / BASELINE_AUDIT
        bind(audit / "completion_verification.json", COMPLETION_SHA)
        completion = read_object(audit / "completion_verification.json")
        if completion.get("passed") is not True or completion.get("problems") != []:
            raise fit.FitDiagnosticError("baseline_strict_acceptance_invalid")
        observer = read_object(audit / "chain_observer.json")
        for rel, digest in observer["files_sha256"].items():
            bind(root / rel, digest)
        for rel in ("chain_observer.json", "observer_write_observation.json",
                    "execution_finished.json", "authorization_consumption.json"):
            bind(audit / rel)
        sample = read_object(audit / "observer_write_observation.json")
        finished = read_object(audit / "execution_finished.json")
        if (observer["supervisor_return_code"] != 0 or sample["covered"] is not True
                or sample["chain_observer_sha256"] != sha256_file(audit / "chain_observer.json")
                or not finished["elapsed_after_sample_file_write_s"]
                <= completion["chain_registered_measured_s"] <= 6120):
            raise fit.FitDiagnosticError("baseline_outer_return_invalid")
        summary = read_object(root / "acceptance_summary.json")
        for key in ("admission", "authorization"):
            rec = summary[key]
            bind(PROJECT_ROOT / rec["path"], rec["sha256"])
        old_admission = read_object(PROJECT_ROOT / summary["admission"]["path"])
        cache_rec = old_admission["package"]["cache_binding"]["cache_identity"]
        bind(PROJECT_ROOT / cache_rec["path"], CACHE_SHA)
    check("baseline_complete")
    binding = {"schema": "w1_d1_lr_baseline_v1", "root": str(root),
               "fixture": fixture, "files": records,
               "artifact_files": indexed, "constant_plan": plan.constant_plan().as_dict()}
    binding["sha256"] = fit._canonical_sha256(binding)
    return binding


def baseline_problems(binding: dict, budget: fit.StageBudget | None = None,
                      counters: dict | None = None) -> list[str]:
    check = budget.check if budget else (lambda _p: None)
    problems = []
    if counters is not None:
        counters["baseline_rechecks"] = counters.get("baseline_rechecks", 0) + 1
    try:
        digest = fit._canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
        if digest != binding["sha256"]:
            problems.append("baseline_binding_digest")
        files = binding["files"]
        if not isinstance(files, dict) or not files or not isinstance(binding.get("artifact_files"), dict):
            return problems + ["baseline_files_not_a_mapping"]
        for path, rec in files.items():
            check("baseline_recheck:" + Path(path).name)
            if counters is not None:
                counters["baseline_file_hashes"] = counters.get("baseline_file_hashes", 0) + 1
                counters["baseline_bytes_hashed"] = counters.get("baseline_bytes_hashed", 0) + Path(path).stat().st_size
            if file_record(Path(path)) != rec:
                problems.append(f"baseline_file_drift:{path}")
        check("baseline_recheck_complete")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems.append(f"baseline_unreadable:{type(exc).__name__}")
    return problems


def _state_signature(opt) -> tuple:
    """No optimizer arithmetic/copy: detect replacement/in-place reset of state.

    Native steps legitimately change versions; refresh only after a completed
    native update. The schedule itself may change LR, never state or optimizer.
    """
    return (id(opt), tuple((id(p), id(entry), tuple(
        (k, id(v), v._version) if torch.is_tensor(v) else (k, v)
        for k, v in sorted(entry.items()))) for p, entry in opt.state.items()))


def _state_digest(opt) -> str:
    """Read actual moments/steps, including .data changes invisible to versions."""
    h = hashlib.sha256()
    for gi, group in enumerate(opt.param_groups):
        for pi, param in enumerate(group["params"]):
            for key, value in sorted(opt.state.get(param, {}).items()):
                h.update(f"{gi}:{pi}:{key}:".encode())
                if torch.is_tensor(value):
                    h.update(str((value.dtype, tuple(value.shape))).encode())
                    h.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    h.update(repr(value).encode())
    return h.hexdigest()


class ScheduleDriver(fit.FitDriver):
    def __init__(self, *, baseline: dict, **kwargs):
        plan = kwargs["plan"]
        if not isinstance(plan, SchedulePlan):
            raise fit.FitDiagnosticError("versioned_lr_plan_required")
        self.baseline = baseline
        self.current_lr = plan.lr
        self.lr_entry = None
        self.prefix_files = []
        self.switches = []
        self.state_signature = None
        self.lr_counts = {"lr_group_reads": 0, "history_rows_compared": 0,
                          "prefix_file_comparisons": 0, "schedule_switches": 0}
        self.cpu_evidence_counts = {}
        self.group_roster = None
        self.last_state_digest = hashlib.sha256().hexdigest()
        self.baseline_index = read_object(Path(baseline["root"]) / "d1/probe_index.json")
        identity = dict(kwargs["identity"])
        identity.update(lr_schedule=plan.schedule(), baseline_sha256=baseline["sha256"])
        kwargs["identity"] = identity
        super().__init__(**kwargs)
        self.group_roster = tuple(tuple(id(p) for p in g["params"])
                                  for g in self.trainer.opt.param_groups)
        decode = self._real_decode_clip
        def counted_decode(*args, **kw):
            # Called only AFTER the inherited deadline/runtime gates, just
            # before actually entering the production decode implementation.
            if self._guard_point is not None and self._guard_point.startswith("step_"):
                self.lr_counts["loss_production_decode_calls"] = self.lr_counts.get(
                    "loss_production_decode_calls", 0) + 1
            return decode(*args, **kw)
        self._real_decode_clip = counted_decode

    def _check_optimizer_binding(self):
        if self.group_roster is not None and tuple(tuple(id(p) for p in g["params"])
                for g in self.trainer.opt.param_groups) != self.group_roster:
            raise fit.FitDiagnosticError("optimizer_group_roster_drift")
        problems = fit._optimizer_binding_problems(
            self.trainer, self.plan, effective_lr=self.current_lr)
        self.lr_counts["lr_group_reads"] += len(self.trainer.opt.param_groups)
        if problems:
            raise fit.FitDiagnosticError(f"optimizer_binding:{problems}")

    def _check_optimizer_state(self):
        super()._check_optimizer_state()
        if self.state_signature is not None and _state_signature(self.trainer.opt) != self.state_signature:
            raise fit.FitDiagnosticError("optimizer_momentum_or_instance_reset")

    def _read_state_digest(self):
        self.lr_counts["optimizer_state_digest_reads"] = self.lr_counts.get(
            "optimizer_state_digest_reads", 0) + 1
        return _state_digest(self.trainer.opt)

    def _before_update(self, step):
        self.budget.check("lr_before_update")
        self._check_runtime_bindings()
        if self._read_state_digest() != self.last_state_digest:
            raise fit.FitDiagnosticError("optimizer_momentum_value_reset")
        if step == self.plan.switch_after + 1:
            expected = {"teacher_reference.npz", "probe_inputs.npz", *(
                probe_name(i, s) for i, s in enumerate(self.plan.probe_steps)
                if s <= self.plan.switch_after)}
            if ({r["file"] for r in self.prefix_files} != expected
                    or len(self.history) != self.plan.switch_after
                    or self.lr_counts["history_rows_compared"] != self.plan.switch_after):
                raise fit.FitDiagnosticError("baseline_prefix_mismatch:prefix_not_persisted")
            # Re-read the durable evidence at the admission boundary, not just
            # the in-memory receipts from earlier probes. No forward/FK added.
            for rel in sorted(expected):
                self.budget.check("prefix_update_admission:" + rel)
                self.lr_counts["prefix_admission_file_hashes"] = self.lr_counts.get(
                    "prefix_admission_file_hashes", 0) + 1
                if file_record(self.stage_dir / rel) != self.baseline["artifact_files"][rel]:
                    raise fit.FitDiagnosticError(f"baseline_prefix_mismatch:admission:{rel}")
            self.budget.check("prefix_history_admission")
            try:
                rows = [json.loads(line) for line in (self.stage_dir / "lr_history.jsonl").read_text().splitlines()]
            except (OSError, ValueError) as exc:
                raise fit.FitDiagnosticError("baseline_prefix_mismatch:durable_history_unreadable") from exc
            if not fit._json_value_equal(rows, self.history) or any(not fit._json_value_equal(
                    {k: v for k, v in row.items() if k not in LR_FIELDS}, want)
                    for row, want in zip(rows, self.baseline_index["history_unrounded"])):
                raise fit.FitDiagnosticError("baseline_prefix_mismatch:durable_history")
            write_json(self.stage_dir / "prefix_admission.json", {
                "before_update": step, "prefix_probe_files": len(expected) - 2,
                "history_rows": len(rows), "baseline_sha256": self.baseline["sha256"]})
            before = _state_signature(self.trainer.opt)
            state_before = self._read_state_digest()
            self.budget.check("lr_switch")
            self.current_lr = self.plan.lr_at(step)
            for group in self.trainer.opt.param_groups:
                group["lr"] = self.current_lr
            if (_state_signature(self.trainer.opt) != before
                    or self._read_state_digest() != state_before):
                raise fit.FitDiagnosticError("optimizer_reset_during_lr_switch")
            self.switches.append({"update": step, "before_first_micro": True,
                                  "state_sha256_before": state_before,
                                  "state_sha256_after": state_before})
            self.lr_counts["schedule_switches"] += 1
        if self.current_lr != self.plan.lr_at(step):
            raise fit.FitDiagnosticError("lr_switch_boundary_mismatch")
        self._check_runtime_bindings()

    def _optimizer_step_entry(self, step):
        self.budget.check("lr_step_entry")
        self._check_runtime_bindings()
        if self._completed_steps != step - 1:
            raise fit.FitDiagnosticError("lr_update_sequence_mismatch")
        if (self.current_lr != self.plan.lr_at(step)
                or self._read_state_digest() != self.last_state_digest):
            raise fit.FitDiagnosticError("lr_or_momentum_changed_inside_update")
        self.lr_entry = {"optimizer_update": step, "lr_schedule_id": SCHEDULE_ID,
                         "effective_lrs": [g["lr"] for g in self.trainer.opt.param_groups],
                         "state_step_before": self._completed_steps,
                         "state_step_after": step,
                         "optimizer_state_sha256_before": self.last_state_digest}
        # The native step is the only allowed writer until _record_update.
        self.state_signature = None

    def _record_update(self, row):
        self.state_signature = _state_signature(self.trainer.opt)
        self.last_state_digest = self._read_state_digest()
        row = {**row, **self.lr_entry,
               "optimizer_state_sha256_after": self.last_state_digest}
        if row["step"] <= self.plan.switch_after:
            original = {k: v for k, v in row.items() if k not in LR_FIELDS}
            wanted = self.baseline_index["history_unrounded"][row["step"] - 1]
            if not fit._json_value_equal(original, wanted):
                raise fit.FitDiagnosticError(f"baseline_prefix_mismatch:history:{row['step']}")
            self.lr_counts["history_rows_compared"] += 1
        super()._record_update(row)
        self.budget.check("lr_history_write")
        with (self.stage_dir / "lr_history.jsonl").open("a") as out:
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.budget.check("lr_history_written")

    def _check_prefix_file(self, rel):
        self.budget.check("baseline_prefix_file:" + rel)
        actual = file_record(self.stage_dir / rel)
        expected = self.baseline["artifact_files"].get(rel)
        self.lr_counts["prefix_file_comparisons"] += 1
        row = {"file": rel, "A": expected, "B": actual, "equal": actual == expected}
        self.prefix_files.append(row)
        write_json(self.stage_dir / "prefix_checks.json", {"files": self.prefix_files})
        self.budget.check("baseline_prefix_written:" + rel)
        if actual != expected:
            raise fit.FitDiagnosticError(f"baseline_prefix_mismatch:file:{rel}")

    def _run_probe(self, step, idx):
        super()._run_probe(step, idx)
        self.budget.check("persist_probe")
        if step == 0:
            _persist_npz(self.stage_dir / "teacher_reference.npz", self.teacher_arrays)
            self._check_prefix_file("teacher_reference.npz")
            _persist_npz(self.stage_dir / "probe_inputs.npz", {
                f"w{b}__{k}": self.probe_batch[k].detach().cpu().numpy()[b]
                for b in range(self.probe_batch["x0"].shape[0]) for k in ("x0", *fit.COND_KEYS)})
            self._check_prefix_file("probe_inputs.npz")
        rel = probe_name(idx, step)
        _persist_npz(self.stage_dir / rel, self.probe_arrays_by_idx[idx])
        if step == 0:
            # Persist the new probes directory entry as well as the file in it.
            _fsync_directory(self.stage_dir)
        if step <= self.plan.switch_after:
            self._check_prefix_file(rel)
        self.budget.check("persist_probe_complete")

    def _write_stage_artifacts(self):
        write_json(self.stage_dir / "lr_evidence.json", {
            "schema": "w1_d1_lr_evidence_v1", "schedule": self.plan.schedule(),
            "baseline_sha256": self.baseline["sha256"], "switches": self.switches,
            "counts": self.lr_counts})
        super()._write_stage_artifacts(arrays_already_persisted=True)
        problems = verify_schedule_stage(self.stage_dir, self.plan, self.baseline, self.budget,
                                         counters=self.cpu_evidence_counts)
        if problems:
            raise fit.FitDiagnosticError(f"lr_stage_evidence:{problems[:8]}")

    def run(self):
        out = super().run()
        out["lr_schedule"] = self.plan.schedule()
        out["baseline_sha256"] = self.baseline["sha256"]
        out["lr_evidence_counts"] = dict(self.lr_counts)
        out["cpu_evidence_counts"] = dict(self.cpu_evidence_counts)
        return out

    def _preflight(self):
        problems = baseline_problems(self.baseline, self.budget, self.cpu_evidence_counts)
        if problems:
            raise fit.FitDiagnosticError(f"baseline_prefix_mismatch:{problems[:3]}")
        super()._preflight()


def _candidate_artifact_problems(stage_dir, plan, report, check=lambda _point: None, counters=None):
    check("lr_saved_artifact_binding")
    index, index_sha = protocol._object_and_sha(stage_dir / "artifacts_index.json")
    files = index.get("files")
    if index.get("schema") != fit.ARTIFACTS_INDEX_SCHEMA or not isinstance(files, dict):
        return ["lr_artifact_index_files_or_schema_invalid"]
    required = {*fit._required_stage_artifacts(plan), *LR_ARTIFACTS}
    problems = [f"lr_artifact_required_missing:{rel}" for rel in sorted(required - files.keys())]
    if report is not None and report.get("artifacts_index_sha256") != index_sha:
        return problems + ["lr_artifact_index_report_mismatch"]
    for rel, expected in files.items():
        check("lr_artifact:" + rel)
        path = stage_dir / rel
        if not path.resolve().is_relative_to(stage_dir.resolve()):
            problems.append(f"lr_artifact_outside_stage:{rel}")
            continue
        if not path.is_file() or not isinstance(expected, dict):
            problems.append(f"lr_artifact_missing_or_invalid:{rel}")
            continue
        if counters is not None:
            counters["candidate_artifact_hashes"] = counters.get("candidate_artifact_hashes", 0) + 1
        if not fit._json_value_equal(file_record(path), expected):
            problems.append(f"lr_artifact_drift:{rel}")
    check("lr_saved_artifact_binding_complete")
    return problems


def verify_schedule_stage(stage_dir: Path, plan: SchedulePlan, baseline: dict,
                          budget: fit.StageBudget, *, counters: dict | None = None) -> list[str]:
    """F's independent sequence reconstruction, before any fresh model work."""
    problems = baseline_problems(baseline, budget, counters)
    try:
        budget.check("lr_saved_stage_report")
        report_path = stage_dir / "stage_report.json"
        report = read_object(report_path) if report_path.is_file() else None
        problems += _candidate_artifact_problems(stage_dir, plan, report, budget.check, counters)
        if problems:
            return problems
        budget.check("lr_saved_read")
        pidx = read_object(stage_dir / "probe_index.json")
        _validate_probe_index(pidx, plan)
        if not fit._json_value_equal(pidx.get("plan"), plan.as_dict()):
            problems.append("lr_saved_plan_mismatch")
        evidence = read_object(stage_dir / "lr_evidence.json")
        if read_object(stage_dir / "prefix_admission.json") != {
                "before_update": plan.switch_after + 1,
                "prefix_probe_files": plan.switch_after // plan.probe_every + 1,
                "history_rows": plan.switch_after, "baseline_sha256": baseline["sha256"]}:
            problems.append("prefix_admission_invalid")
        switches = evidence.get("switches")
        if (not fit._json_value_equal(evidence.get("schedule"), plan.schedule())
                or evidence.get("baseline_sha256") != baseline["sha256"]
                or not isinstance(switches, list) or len(switches) != 1
                or not isinstance(switches[0], dict)
                or switches[0].get("update") != plan.switch_after + 1
                or switches[0].get("before_first_micro") is not True
                or switches[0].get("state_sha256_before") != switches[0].get("state_sha256_after")):
            problems.append("lr_evidence_binding_mismatch")
        rows = [json.loads(line) for line in (stage_dir / "lr_history.jsonl").read_text().splitlines()]
        if not fit._json_value_equal(rows, pidx["history_unrounded"]):
            problems.append("lr_history_cross_record_mismatch")
        original = read_object(Path(baseline["root"]) / "d1/probe_index.json")
        groups = None
        previous_state = hashlib.sha256().hexdigest()
        for u, row in enumerate(pidx["history_unrounded"], 1):
            budget.check(f"lr_verify_update_{u}")
            lrs = row.get("effective_lrs")
            if not isinstance(lrs, list) or not lrs:
                problems.append(f"lr_groups_missing:{u}")
                continue
            groups = len(lrs) if groups is None else groups
            if (len(lrs) != groups or row.get("lr_schedule_id") != SCHEDULE_ID
                    or not fit._json_value_equal(row.get("optimizer_update"), u)
                    or not fit._json_value_equal(row.get("state_step_before"), u - 1)
                    or not fit._json_value_equal(row.get("state_step_after"), u)
                    or any(not fit._json_value_equal(lr, plan.lr_at(u)) for lr in lrs)):
                problems.append(f"lr_sequence_mismatch:{u}")
            state_after = row.get("optimizer_state_sha256_after")
            if (row.get("optimizer_state_sha256_before") != previous_state
                    or not isinstance(state_after, str) or len(state_after) != 64
                    or any(ch not in "0123456789abcdef" for ch in state_after)):
                problems.append(f"optimizer_state_history_disconnected:{u}")
            if u == plan.switch_after + 1 and switches and isinstance(switches[0], dict):
                if switches[0].get("state_sha256_before") != previous_state:
                    problems.append("optimizer_state_switch_not_bound")
            previous_state = state_after
            if u <= plan.switch_after and not fit._json_value_equal(
                    {k: v for k, v in row.items() if k not in LR_FIELDS},
                    original["history_unrounded"][u - 1]):
                problems.append(f"baseline_prefix_mismatch:history:{u}")
            if counters is not None:
                counters["schedule_rows_verified"] = counters.get("schedule_rows_verified", 0) + 1
                if u <= plan.switch_after:
                    counters["history_rows_compared"] = counters.get("history_rows_compared", 0) + 1
        expected_files = ["teacher_reference.npz", "probe_inputs.npz"] + [
            probe_name(i, s) for i, s in enumerate(plan.probe_steps) if s <= plan.switch_after]
        expected_receipts = []
        for rel in expected_files:
            budget.check("lr_verify_prefix:" + rel)
            actual = file_record(stage_dir / rel)
            if counters is not None:
                counters["prefix_file_comparisons"] = counters.get("prefix_file_comparisons", 0) + 1
            expected = baseline["artifact_files"][rel]
            if actual != expected:
                problems.append(f"baseline_prefix_mismatch:file:{rel}")
            expected_receipts.append({"file": rel, "A": expected, "B": actual,
                                      "equal": actual == expected})
        if not fit._json_value_equal(read_object(stage_dir / "prefix_checks.json"),
                                     {"files": expected_receipts}):
            problems.append("prefix_receipts_not_reproducible")
        budget.check("lr_verify_complete")
    except fit.BudgetExceeded:
        raise
    except (fit.FitDiagnosticError, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        problems.append(f"lr_evidence_unreadable:{type(exc).__name__}:{str(exc)[:120]}")
    return problems


def classify(a: dict, b: dict, problems: list[str]) -> dict:
    if problems:
        return {"conclusion": "not_comparable", "fit_verdict": None,
                "metrics": None, "problems": list(problems)}
    for metrics in (a, b):
        if any(not _finite_number(metrics.get(k))
               or metrics[k] < 0 for k in GATES):
            return classify({}, {}, ["comparison_metric_invalid"])
    gates = {k: b[k] <= limit for k, limit in GATES.items()}
    passed = all(gates.values())
    improved = (gates["root_position_rmse_m"] and gates["joint_rmse_rad"]
                and all(b[k] <= 0.9 * a[k] for k in
                        ("root_rotation_mean_rad", "teacher_fk_mpjpe_m")))
    return {"conclusion": ("single_window_pass" if passed else
                           "improved_but_insufficient" if improved else "no_prespecified_improvement"),
            "fit_verdict": "fit_pass" if passed else "fit_insufficient",
            "metrics": {k: {"A": a[k], "B": b[k], "B_minus_A": b[k] - a[k],
                             "B_over_A": b[k] / a[k] if a[k] else None,
                             "B_gate": gates[k],
                             "change": "regressed" if b[k] > a[k] else "unchanged"
                             if b[k] == a[k] else "improved",
                             "improved_at_least_10pct": b[k] <= 0.9 * a[k]}
                        for k in GATES}, "problems": []}


def build_comparison(*, stage_dir: Path, baseline: dict, plan: SchedulePlan,
                     f_candidate: dict, budget: fit.StageBudget,
                     counters: dict | None = None) -> dict:
    problems = verify_schedule_stage(stage_dir, plan, baseline, budget, counters=counters)
    if (f_candidate.get("complete_evidence") is not True
            or f_candidate.get("replay_passed") is not True):
        problems.append("f_replay_or_evidence_incomplete")
    out = {"schema": "w1_d1_lr_comparison_v1", "state": "pending_outer_return",
           "baseline_sha256": baseline["sha256"], "rules": RULES,
           "schedule": plan.schedule(), "paired_probes": None,
           "prefix": None, "tail": None, "final_spikes": None,
           "B_stage_report": file_record(stage_dir / "stage_report.json")}
    try:
        budget.check("comparison_read_indexes")
        a = read_object(Path(baseline["root"]) / "d1/probe_index.json")
        b = read_object(stage_dir / "probe_index.json")
        _validate_probe_index(a, plan.constant_plan())
        _validate_probe_index(b, plan)
        aa, bb = (a["probes"][-1]["windows"][0]["metrics"],
                  b["probes"][-1]["windows"][0]["metrics"])
        verdict = classify(aa, bb, problems)
        if not problems and verdict["fit_verdict"] != f_candidate.get("verdict"):
            problems.append("comparison_f_verdict_disagreement")
        if not problems:
            pairs = []
            for pa, pb in zip(a["probes"], b["probes"]):
                budget.check("comparison_pair")
                ma, mb = [p["windows"][0]["metrics"] for p in (pa, pb)]
                pairs.append({"step": pa["step"], "metrics": classify(ma, mb, [])["metrics"]})
                if counters is not None:
                    counters["paired_probe_rows"] = counters.get("paired_probe_rows", 0) + 1
            tail = {}
            for label, data in (("A", a), ("B", b)):
                tail[label] = {}
                for key in ("root_rotation_mean_rad", "teacher_fk_mpjpe_m"):
                    values = [p["windows"][0]["metrics"][key] for p in data["probes"]
                              if p["step"] >= (800 if not plan.synthetic else plan.steps * 0.8)]
                    tail[label][key] = {"median": statistics.median(values),
                                         "passing_points": sum(v <= GATES[key] for v in values),
                                         "n_points": len(values)}
            out.update(paired_probes=pairs, tail=tail,
                       prefix=read_object(stage_dir / "prefix_checks.json"),
                       final_spikes={"A": aa["spike"], "B": bb["spike"]})
        out.update(classify(aa, bb, problems))
    except fit.BudgetExceeded:
        raise
    except (fit.FitDiagnosticError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        out.update(classify({}, {}, problems + [f"comparison_unreadable:{type(exc).__name__}"]))
    budget.check("comparison_complete")
    return out


def consume_comparison(run_dir: Path, *, observer_return_code: int | None = None) -> dict:
    """A standalone comparison is NOT evidence of successful return.

    Require the independent observer's *actual*, externally observed exit code
    in addition to every disk binding. Any unknown/failure yields null metrics.
    """
    run_dir = Path(run_dir)
    try:
        if type(observer_return_code) is not int or observer_return_code != 0:
            raise fit.FitDiagnosticError("independent_actual_return_required")
        completion, _, _, _, summary, receipt = protocol.verify_observer_files(run_dir, "chain")
        protocol.verify_stage_ledger(run_dir, summary, receipt)
        f, f_sha = protocol._object_and_sha(run_dir / "f/f_report.json")
        cmp_path = run_dir / "f/comparison.json"
        comparison, cmp_sha = protocol._object_and_sha(cmp_path)
        stage_report, stage_sha = protocol._object_and_sha(run_dir / "d1/stage_report.json")
        if (summary.get("authorized_includes") != ["D1", "F"] or summary.get("d6_ran") is not False
                or (run_dir / "d6").exists()
                or f.get("complete_evidence") is not True or f.get("replay_passed") is not True
                or stage_report.get("local_complete_evidence") is not True
                or f.get("comparison") != summary.get("comparison")
                or f.get("comparison") != {"path": "comparison.json", "sha256": cmp_sha,
                                           "bytes": cmp_path.stat().st_size, "conclusion": comparison.get("conclusion")}
                or f_sha != completion["bindings"]["f/f_report.json"]
                or cmp_sha != completion["bindings"]["f/comparison.json"]
                or stage_sha != receipt["stage_artifacts"]["d1"]["report_sha256"]
                or comparison["B_stage_report"] != {"sha256": stage_sha, "bytes": (run_dir / "d1/stage_report.json").stat().st_size}
                or comparison.get("schema") != "w1_d1_lr_comparison_v1"
                or comparison.get("state") != "pending_outer_return" or comparison.get("problems") != []
                or comparison.get("conclusion") not in {"single_window_pass", "improved_but_insufficient", "no_prespecified_improvement"}
                or not isinstance(comparison.get("metrics"), dict) or set(comparison["metrics"]) != set(GATES)
                or comparison.get("fit_verdict") not in {"fit_pass", "fit_insufficient"}
                or comparison["fit_verdict"] != f.get("verdict")):
            raise fit.FitDiagnosticError("comparison_not_bound_to_complete_chain")
        admission_path = Path(summary["admission"]["path"])
        admission, admission_sha = protocol._object_and_sha(admission_path)
        if admission_sha != summary["admission"]["sha256"]:
            raise fit.FitDiagnosticError("comparison_admission_drift")
        auth_path = Path(summary["authorization"]["path"])
        auth, auth_sha = protocol._object_and_sha(auth_path)
        if (auth_sha != summary["authorization"]["sha256"] or auth.get("includes") != ["D1", "F"]
                or auth.get("authorized") is not True or auth.get("admission_package_sha256") != admission_sha):
            raise fit.FitDiagnosticError("comparison_authorization_drift")
        package = admission["package"]
        plan, baseline = SchedulePlan(**package["fit_plan"]), package["baseline"]
        if (not isinstance(baseline, dict) or type(summary.get("fixture")) is not bool
                or package.get("fixture") is not summary["fixture"] or plan.synthetic is not summary["fixture"]
                or any(d.get("baseline_sha256") != baseline.get("sha256") for d in (summary, package, comparison, stage_report))
                or any(not fit._json_value_equal(d.get("schedule"), plan.schedule()) for d in (summary, package, comparison))
                or not fit._json_value_equal(stage_report.get("plan"), plan.as_dict())):
            raise fit.FitDiagnosticError("comparison_plan_or_baseline_binding_invalid")
        if not summary["fixture"]:
            r_expected = PROJECT_ROOT / protocol.R_ROOT / "attempt_001/r_admission.json"
            if admission_path.resolve() != r_expected:
                raise fit.FitDiagnosticError("lr_r_fixed_admission_required")
            protocol.verify_r_actual_return(admission_path.parent, auth.get("r_observer_actual_return"),
                                            expected_admission_sha=admission_sha)
        # Consume actual byte evidence on BOTH sides, not just the hash of a
        # document that once pointed to those bytes. No arrays are decoded.
        problems = baseline_problems(baseline) + _candidate_artifact_problems(run_dir / "d1", plan, stage_report)
        if problems:
            return _unavailable_comparison(problems)
        return {**comparison, "state": "consumable_after_verified_actual_return"}
    except (fit.FitDiagnosticError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return _unavailable_comparison([str(exc)])


def _unavailable_comparison(problems):
    return {"schema": "w1_d1_lr_comparison_v1", "paired_probes": None,
            "prefix": None, "tail": None, "final_spikes": None,
            **classify({}, {}, problems)}
