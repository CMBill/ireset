"""W1 rotation-fusion v2 verification: register -> fixed windows -> val31 -> F.

This is an independent experiment. It never rewrites the frozen W1-D/E or
attribution artifacts; the only production-facing code is the new pure module
``ireset.model.rotation_fusion``. The candidate is the single registered
protocol ``w1_rotation_path_nlerp_v2``.

Entry points::

    register        verify anchors, budgets and SOURCE, then freeze the plan
    fixed-windows   S: 13 demos x (teacher + 2 models x V0/V1/V2 + roundtrip V0/V2)
    val31           V: 62 real sessions -> five-method 155-row matrix
    finalize        F: recompute V2 from inputs, re-measure, decide eligibility

Every attempt is single-shot: no quality retry and no automatic recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import diagnose_w1_attribution as attribution
import eval_generation as raw
import omniretarget_w1_de as driver
from ireset.data.a2_integrity import sha256_file
from ireset.model.rotation_fusion import (
    ANGULAR_SPEED_LIMIT_RADPS,
    FRAME_DT_DEFAULT,
    MIN_BLEND_NORM,
    PATH_AMBIGUITY_DOT,
    PROTOCOL,
    QUAT_NORM_TOL,
    RotationFusionRejected,
    angular_speed_radps,
    fuse_windows_v2,
)
from ireset.model.stitch import fuse_windows, window_starts

EXECUTION_ID = "w1_rotation_fusion_v1"
ACCOUNTING_SCHEMA = "pre_registered_v1"
SOURCE = ROOT / "derived/experiments/omniretarget_first/20261013_be00c50_w1_de"
ATTRIBUTION = ROOT / "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution"
PLAN = ROOT / "docs/plans/omniretarget_w1_rotation_fusion_plan.md"
KNOWN_CASE = "sub16_largebox_048"
KNOWN_FRAME = 108
KNOWN_LEGACY_FRAME = 108
DET_BAD_FRAME = 114
MODELS = ("deterministic", "diffusion")
VERSIONS = ("V0", "V1", "V2")
SEED, DDIM_STEPS = 7, 25
WINDOW, STRIDE = 60, 30
TOTAL_BUDGET_S = 7200.0
STAGE_BUDGETS = {
    "register": (600.0, 30.0),
    "fixed-windows": (1800.0, 60.0),
    "val31": (3600.0, 120.0),
    "finalize": (1200.0, 60.0),
}
STAGE_DIRS = {"register": "freeze", "fixed-windows": "fixed_windows",
              "val31": "val31", "finalize": "decisions"}
EXPECTED_CKPTS = {
    "deterministic": "7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff",
    "diffusion": "1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1",
}
PLAN_ANCHORS = {
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/execution_registration.json":
        "5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e",
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/decisions/attempt_001/w1_de_final.json":
        "7f1a9a595cf7964e99a929c83d3bcaac4aa3482a96e515228921b7b67a576737",
    "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/freeze/registration.json":
        "499257b0a983b8af1c94a3d819d573d9be5470cb97dcdf4c02a84ac03d4bab52",
    "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/diagnosis/attempt_001/attribution.json":
        "72186392cce075ce5893c68f82acaa41fd3efd45b534970696f698f5563c07ac",
    "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/decisions/diagnostic_closure.json":
        "93cac087a881e1a05c3f38d3542208e7690aa69afaaa30a626999960136c9064",
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/formal/deterministic/attempt_001/best.pt":
        EXPECTED_CKPTS["deterministic"],
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/formal/diffusion/attempt_001/best.pt":
        EXPECTED_CKPTS["diffusion"],
    "src/ireset/model/stitch.py":
        "e9fb0c55d26ba48a2d3840ba481030cfaef7502abe76efc10a220030e52fcd58",
    "src/ireset/model/generator.py":
        "9844bce25831bb636ce0b7c60d657da9fdcdb20d19a9385290bb472dce44af2d",
}
IMPLEMENTATION_FILES = (
    "src/ireset/model/rotation_fusion.py",
    "scripts/omniretarget_w1_rotation_fusion.py",
    "tests/test_rotation_fusion.py",
    "tests/test_w1_rotation_fusion.py",
    "docs/plans/omniretarget_w1_rotation_fusion_plan.md",
)
FUSION_PARAMETERS = {
    "protocol": PROTOCOL,
    "window": WINDOW, "stride": STRIDE,
    "path_ambiguity_dot": PATH_AMBIGUITY_DOT,
    "min_blend_norm": MIN_BLEND_NORM,
    "angular_speed_limit_radps": ANGULAR_SPEED_LIMIT_RADPS,
    "quaternion_norm_tolerance": 1e-5,
}


# --- JSON / paths --------------------------------------------------------------

def read_json(path: Path):
    return json.loads(Path(path).read_text())


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def write_json(path: Path, data, *, exclusive: bool = False) -> None:
    driver.atomic_json(Path(path), attribution.safe_json(data), exclusive=exclusive)


def record_path(path: Path) -> str:
    return attribution.record_path(Path(path))


def file_record(path: Path) -> dict:
    return {"path": record_path(path), "sha256": sha256_file(path)}


def full_sha(path: Path) -> str:
    return sha256_file(Path(path))


def cohort_root() -> Path:
    """Single registration scope; sibling scans cannot be reset by moving roots."""
    return (ROOT / "derived/experiments/omniretarget_first").resolve()


def guard_run(path: Path, *sources: Path) -> Path:
    """Resolve to an independent, empty derived experiment directory."""
    path = Path(path).expanduser().resolve()
    derived = (ROOT / "derived").resolve()
    if not path.is_relative_to(derived):
        raise ValueError("run root must resolve under derived/")
    if path.parent != cohort_root():
        raise ValueError("run root must be a direct child of the cohort root")
    for source in sources:
        source = Path(source).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("run root must not overlap a read-only SOURCE")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("run root must be empty; no implicit retry or overwrite")
    return path


def _hash_map(anchors: dict[str, str]) -> list[str]:
    problems = []
    for rel, want in anchors.items():
        path = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.is_file() or full_sha(path) != want:
            problems.append(f"anchor_missing_or_drifted:{rel}")
    return problems


def _select_train6(train24: list[str]) -> list[str]:
    if len(train24) != 24 or len(set(train24)) != 24:
        raise ValueError("train24 roster malformed")
    subjects = sorted({d.split("_", 1)[0] for d in train24})
    if len(subjects) != 6:
        raise ValueError("six subjects required")
    return [min(d for d in train24 if d.split("_", 1)[0] == subject)
            for subject in subjects]


def implementation_snapshot(run: Path) -> dict:
    snap = run / "freeze/implementation_snapshot"
    snap.mkdir(parents=True, exist_ok=False)
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD"],
                          capture_output=True, text=True, check=True).stdout
    (snap / "tracked_diff.patch").write_text(diff)
    untracked = {}
    for rel in subprocess.run(["git", "-C", str(ROOT), "ls-files", "--others",
                               "--exclude-standard"], capture_output=True, text=True,
                              check=True).stdout.splitlines():
        rel = rel.strip()
        if rel:
            untracked[rel] = full_sha(ROOT / rel)
    files = {}
    for rel in IMPLEMENTATION_FILES:
        src = ROOT / rel
        if not src.is_file():
            raise ValueError(f"implementation file missing:{rel}")
        dest = snap / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        files[rel] = {"sha256": full_sha(src), "snapshot": record_path(dest)}
    return {
        "head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip(),
        "tracked_diff": file_record(snap / "tracked_diff.patch"),
        "untracked": untracked,
        "files": files,
    }


def fusion_parameter_sha() -> str:
    return hashlib.sha256(json.dumps(FUSION_PARAMETERS, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def cohort_id(source_registration_sha: str) -> str:
    payload = {"source_registration_sha256": source_registration_sha,
               "protocol": PROTOCOL, "parameters": FUSION_PARAMETERS,
               "seed": SEED, "ddim_steps": DDIM_STEPS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# --- ledger, index, budgets, sealed terminal state ------------------------------

def _ledger(run: Path) -> list[dict]:
    path = run / "costs.json"
    if not path.is_file():
        return []
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("execution_id") != EXECUTION_ID:
        raise ValueError("cost ledger execution id mismatch")
    rows = doc.get("invocations")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("cost ledger malformed")
    return rows


def _write_ledger(run: Path, rows: list[dict]) -> None:
    write_json(run / "costs.json", {"execution_id": EXECUTION_ID,
                                    "invocations": rows})


def _upsert_cost(run: Path, stage: str, attempt: str, wall: float,
                 extra: dict, status: str) -> None:
    if not (isinstance(wall, (int, float)) and not isinstance(wall, bool)
            and math.isfinite(float(wall)) and float(wall) >= 0):
        raise ValueError("wall cost must be finite and nonnegative")
    rows = _ledger(run)
    hits = [i for i, row in enumerate(rows)
            if row.get("stage") == stage and row.get("attempt") == attempt]
    if len(hits) > 1:
        raise ValueError("duplicate cost ledger keys")
    row = {"stage": stage, "attempt": attempt, "status": status,
           "wall_s": round(float(wall), 1), "measured_wall_s": float(wall),
           "complete": extra.get("complete", False), **extra,
           "scope": {"stage": stage, "attempt": attempt}}
    if hits:
        row = {**rows[hits[0]], **row,
               "timestamp": rows[hits[0]].get("timestamp")}
        rows[hits[0]] = row
    else:
        rows.append(row)
    _write_ledger(run, rows)


def _index(run: Path) -> list[dict]:
    path = run / "attempts_index.json"
    if not path.is_file():
        return []
    doc = read_json(path)
    rows = doc.get("attempts") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("attempts index malformed")
    keys = [(r.get("stage"), r.get("attempt")) for r in rows]
    if any(not isinstance(s, str) or not isinstance(a, str) for s, a in keys):
        raise ValueError("attempt index keys missing")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate attempt index keys")
    return rows


def _write_index(run: Path, rows: list[dict]) -> None:
    write_json(run / "attempts_index.json", {"execution_id": EXECUTION_ID,
                                             "attempts": rows})


def _replace_index(run: Path, stage: str, attempt: str, entry: dict) -> None:
    rows = _index(run)
    hits = [i for i, row in enumerate(rows)
            if row.get("stage") == stage and row.get("attempt") == attempt]
    if len(hits) != 1:
        raise ValueError(f"index requires exactly one row for {stage}/{attempt}")
    rows[hits[0]] = {**rows[hits[0]], **entry}
    _write_index(run, rows)


def _append_index(run: Path, stage: str, attempt: str, entry: dict) -> None:
    rows = _index(run)
    if any(row.get("stage") == stage and row.get("attempt") == attempt
           for row in rows):
        raise ValueError("attempt already indexed")
    rows.append({"stage": stage, "attempt": attempt, **entry})
    _write_index(run, rows)


def _stage_index_row(run: Path, stage: str) -> dict | None:
    hits = [r for r in _index(run) if r.get("stage") == stage]
    return hits[-1] if hits else None


def _stage_ledger_rows(run: Path, stage: str) -> list[dict]:
    return [r for r in _ledger(run) if r.get("stage") == stage]


def _registration_sha(run: Path) -> str | None:
    path = run / "freeze/execution_registration.json"
    return full_sha(path) if path.is_file() else None


def _preregister_attempt(run: Path, stage: str, attempt: str, t0: float,
                         *, cohort_id: str | None = None) -> None:
    """Persist a running row before substantive work so a kill is not zero cost."""
    if cohort_id is not None and not (run / "cohort.json").is_file():
        write_json(run / "cohort.json",
                   {"execution_id": EXECUTION_ID, "cohort_id": cohort_id})
    elapsed = max(0.0, time.perf_counter() - t0)
    entry = {"status": "running", "complete": False,
             "measured_wall_s": elapsed,
             "accounting_schema": ACCOUNTING_SCHEMA,
             "execution_registration_sha256": _registration_sha(run)}
    if cohort_id is not None:
        entry["cohort_id"] = cohort_id
    hits = [r for r in _index(run)
            if r.get("stage") == stage and r.get("attempt") == attempt]
    if hits:
        _replace_index(run, stage, attempt, entry)
    else:
        _append_index(run, stage, attempt, entry)
    extra = {"complete": False, "accounting_schema": ACCOUNTING_SCHEMA}
    if cohort_id is not None:
        extra["cohort_id"] = cohort_id
    _upsert_cost(run, stage, attempt, elapsed, extra, "running")


def _disk_attempts(run: Path, stage: str) -> set[str]:
    base = run / STAGE_DIRS[stage]
    if not base.is_dir():
        return set()
    return {p.name for p in base.iterdir()
            if p.is_dir() and re.fullmatch(r"attempt_\d+", p.name)}


def stage_state(run: Path, stage: str) -> tuple[dict | None, list[str]]:
    problems = []
    try:
        row = _stage_index_row(run, stage)
        if row is None:
            return None, ["no_attempts"]
        ledger = [e for e in _stage_ledger_rows(run, stage)
                  if e.get("attempt") == row.get("attempt")]
        if len(ledger) != 1:
            return row, ["attempt_index_ledger_mismatch"]
        e = ledger[0]
        attempt_dir = run / STAGE_DIRS[stage] / row.get("attempt", "")
        if not attempt_dir.is_dir():
            problems.append("attempt_dir_missing")
        terminal = ("done", "crashed", "timeout", "incomplete")
        if row.get("status") == "running" or e.get("status") == "running":
            problems.append("attempt_running")
        elif row.get("status") == "publishing" or e.get("status") == "publishing":
            problems.append("attempt_final_check_pending")
        elif row.get("status") not in terminal or e.get("status") not in terminal:
            problems.append("attempt_status_invalid")
        elif row.get("status") != e.get("status"):
            problems.append("index_ledger_status_mismatch")
        if row.get("complete") is not e.get("complete"):
            problems.append("index_ledger_completion_mismatch")
        if row.get("execution_registration_sha256") != _registration_sha(run):
            problems.append("attempt_registration_drifted")
        for key in ("report", "summary"):
            if row.get(key):
                p = ROOT / row[key]
                if not p.is_file() or full_sha(p) != row.get(f"{key}_sha256"):
                    problems.append(f"{key}_hash_mismatch")
        artifacts = row.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            problems.append("artifacts_not_a_mapping")
        else:
            for rel, sha in artifacts.items():
                p = ROOT / rel
                if not p.is_file() or full_sha(p) != sha:
                    problems.append(f"artifact_hash_mismatch:{rel}")
        if not isinstance(e.get("complete"), bool):
            problems.append("attempt_completion_invalid")
        if not (isinstance(e.get("wall_s"), (int, float))
                and not isinstance(e.get("wall_s"), bool)
                and math.isfinite(float(e["wall_s"])) and e["wall_s"] >= 0):
            problems.append("attempt_cost_invalid")
        return row, problems
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, [f"stage_state_unreadable:{type(exc).__name__}"]


def _stage_history(run: Path, stage: str) -> float | None:
    """Cost only when disk, index and ledger agree on every live attempt."""
    try:
        ledger = _stage_ledger_rows(run, stage)
        index = [r for r in _index(run) if r.get("stage") == stage]
        disk = _disk_attempts(run, stage)
        if not ledger and not index and not disk:
            return 0.0
        ledger_by = {e.get("attempt"): e for e in ledger}
        index_by = {r.get("attempt"): r for r in index}
        if (len(ledger_by) != len(ledger) or len(index_by) != len(index)
                or None in ledger_by or None in index_by):
            return None
        if set(disk) != set(ledger_by) or set(disk) != set(index_by):
            return None
        total = 0.0
        for attempt, e in ledger_by.items():
            index_row = index_by[attempt]
            if index_row.get("status") != e.get("status"):
                return None
            if index_row.get("complete") is not e.get("complete"):
                return None
            if e.get("status") not in ("done", "crashed", "timeout", "incomplete"):
                return None
            measured = e.get("measured_wall_s")
            rounded = e.get("wall_s")
            index_measured = index_row.get("measured_wall_s")
            if not (_num_cost(measured) and _num_cost(rounded)
                    and _num_cost(index_measured)):
                return None
            # Same attempt, same cost: the index and ledger must agree on the
            # unrounded measurement, and the ledger's rounded field must be its
            # own serialization of that measurement.  Status agreement alone
            # would let one side be rewritten to zero.
            if float(index_measured) != float(measured):
                return None
            if float(rounded) != round(float(measured), 1):
                return None
            if "wall_s" in index_row:
                if not _num_cost(index_row["wall_s"]) \
                        or float(index_row["wall_s"]) != round(float(measured), 1):
                    return None
            total += float(measured)
        return total
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _num_cost(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0)


def _total_history(run: Path) -> float | None:
    total = 0.0
    for stage in STAGE_BUDGETS:
        value = _stage_history(run, stage)
        if value is None:
            return None
        total += value
    return total


def _cohort_history(run: Path, cohort: str) -> dict:
    """Same-cohort prior runs in the fixed cohort root; unknown history stops."""
    runs, by_stage, total = [], {}, 0.0
    run_resolved = Path(run).resolve()
    base = cohort_root()
    if run_resolved.is_relative_to(base):
        if run_resolved.parent != base:
            return {"known": False, "reason": "run_not_in_cohort_root",
                    "runs": runs}
        scope = base
    else:
        scope = run_resolved.parent
    if not scope.is_dir():
        return {"known": True, "runs": runs, "stage_cost_s": {}, "total_s": 0.0}
    for sibling in sorted(scope.iterdir()):
        if sibling == run_resolved or not sibling.is_dir():
            continue
        reg_path = sibling / "freeze/execution_registration.json"
        if not reg_path.is_file():
            # A pre-registered attempt killed before registration still owns
            # ledger/index/cohort evidence. Sealed attempts are charged; any
            # running or unaccounted stage keeps the cohort history unknown.
            try:
                aborted = _ledger(sibling)
            except (OSError, ValueError, TypeError):
                aborted = []
            try:
                aborted_index = _index(sibling)
            except (OSError, ValueError, TypeError):
                aborted_index = []
            try:
                marker = read_json(sibling / "cohort.json")
            except (OSError, ValueError, TypeError):
                marker = {}
            matching = (
                marker.get("cohort_id") == cohort
                or any(e.get("cohort_id") == cohort
                       and e.get("accounting_schema") == ACCOUNTING_SCHEMA
                       for e in aborted)
                or any(r.get("cohort_id") == cohort
                       and r.get("accounting_schema") == ACCOUNTING_SCHEMA
                       for r in aborted_index))
            if not matching:
                continue
            stage_costs = {}
            for stage in STAGE_BUDGETS:
                cost = _stage_history(sibling, stage)
                if cost is None:
                    return {"known": False,
                            "reason": (f"unregistered_attempt:"
                                       f"{sibling.name}:{stage}"),
                            "runs": runs}
                stage_costs[stage] = cost
                by_stage[stage] = by_stage.get(stage, 0.0) + cost
                total += cost
            runs.append({"run": record_path(sibling),
                         "stage_cost_s": stage_costs, "unregistered": True})
            continue
        try:
            reg = read_json(reg_path)
        except (OSError, ValueError):
            return {"known": False, "reason": f"unreadable_registration:{sibling.name}",
                    "runs": runs}
        if reg.get("cohort_id") != cohort:
            continue
        if reg.get("execution_id") != EXECUTION_ID:
            return {"known": False, "reason": f"foreign_execution:{sibling.name}",
                    "runs": runs}
        stage_costs = {}
        for stage in STAGE_BUDGETS:
            cost = _stage_history(sibling, stage)
            if cost is None:
                return {"known": False, "reason": f"unknown_history:{sibling.name}:{stage}",
                        "runs": runs}
            stage_costs[stage] = cost
            by_stage[stage] = by_stage.get(stage, 0.0) + cost
            total += cost
        runs.append({"run": record_path(sibling), "stage_cost_s": stage_costs})
    return {"known": True, "runs": runs, "stage_cost_s": by_stage,
            "total_s": total}


def budget_for(run: Path, stage: str, cohort: dict) -> dict:
    if not cohort.get("known"):
        raise SystemExit(f"cohort history unknown: {cohort.get('reason')}")
    local_stage = _stage_history(run, stage)
    local_total = _total_history(run)
    if local_stage is None or local_total is None:
        raise SystemExit("local cost history unknown")
    stage_budget, reserve = STAGE_BUDGETS[stage]
    prior_stage = float(cohort.get("stage_cost_s", {}).get(stage, 0.0))
    prior_total = float(cohort.get("total_s", 0.0))
    remaining_stage = stage_budget - local_stage - prior_stage
    remaining_total = TOTAL_BUDGET_S - local_total - prior_total
    remaining = min(remaining_stage, remaining_total)
    if remaining <= reserve:
        raise SystemExit(
            f"{stage} budget exhausted: local {local_stage:.1f}s + prior "
            f"{prior_stage:.1f}s of {stage_budget}s (remaining {remaining:.1f}s "
            f"<= reserve {reserve}s)")
    return {"stage": stage, "total_s": stage_budget, "reserve_s": reserve,
            "local_stage_s": local_stage, "prior_stage_s": prior_stage,
            "prior_total_s": prior_total, "remaining_s": remaining}


def _finish_stage(run: Path, stage: str, attempt: str, t0: float,
                  budget: dict, row: dict, extra: dict,
                  status: str = "done") -> bool:
    """pending -> publishing -> seal with a proof that the seal is affordable."""
    attempt_dir = run / STAGE_DIRS[stage] / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    row = {**row, "execution_registration_sha256": _registration_sha(run),
           "stage": stage, "attempt": attempt}
    for key in ("report", "summary"):
        if row.get(key):
            path = ROOT / row[key]
            if path.is_file():
                row[f"{key}_sha256"] = full_sha(path)
    writes: list[float] = []

    def timed(fn, *args):
        start = time.perf_counter()
        fn(*args)
        writes.append(time.perf_counter() - start)

    def publish(rw):
        rw = {**rw, "status": "publishing",
              "measured_wall_s": time.perf_counter() - t0}
        if _stage_index_row(run, stage) is None:
            timed(_append_index, run, stage, attempt, rw)
        else:
            timed(_replace_index, run, stage, attempt, rw)
        timed(_upsert_cost, run, stage, attempt, rw["measured_wall_s"],
              {**extra, "complete": rw.get("complete", False),
               "incomplete_reason": rw.get("incomplete_reason")}, "publishing")
        return rw

    remaining = budget["remaining_s"]
    overrun = time.perf_counter() - t0 >= remaining
    if not overrun:
        pending = {**row, "status": "running", "complete": False,
                   "measured_wall_s": time.perf_counter() - t0}
        if _stage_index_row(run, stage) is None:
            timed(_append_index, run, stage, attempt, pending)
        else:
            timed(_replace_index, run, stage, attempt, pending)
        timed(_upsert_cost, run, stage, attempt, pending["measured_wall_s"],
              {**extra, "complete": False}, "running")
        overrun = time.perf_counter() - t0 >= remaining
    if overrun:
        row.update(complete=False, incomplete_reason="budget_exceeded_after_registration")
        status = "incomplete"
    row = publish(row)
    if not overrun and time.perf_counter() - t0 >= remaining:
        overrun = True
        row.update(complete=False, incomplete_reason="budget_exceeded_after_publication")
        status = "incomplete"
        row = publish(row)

    def reserve():
        return max(driver.SEAL_TAIL_RESERVE_S, 2.0 * max(writes, default=0.0))

    def terminal(rw, st, cost, tail):
        rw = {**rw, "status": st, "measured_wall_s": cost,
              "seal_tail_reserve_s": tail}
        timed(_replace_index, run, stage, attempt, rw)
        timed(_upsert_cost, run, stage, attempt, cost,
              {**extra, "complete": rw.get("complete", False),
               "incomplete_reason": rw.get("incomplete_reason"),
               "seal_tail_reserve_s": tail}, st)
        return rw

    def revoke(reason):
        nonlocal overrun, row
        overrun = True
        tail = reserve()
        row.update(complete=False, incomplete_reason=reason)
        row = publish(row)
        return terminal(row, "incomplete", time.perf_counter() - t0 + tail, tail)

    tail = reserve()
    if not overrun and time.perf_counter() - t0 + tail >= remaining:
        overrun = True
        row.update(complete=False, incomplete_reason="budget_exceeded_before_seal")
        status = "incomplete"
        row = publish(row)
    row = terminal(row, status, time.perf_counter() - t0 + tail, tail)
    for _ in range(driver.SEAL_VERIFY_ROUNDS):
        post = time.perf_counter() - t0
        if not overrun and post >= remaining:
            row = revoke("budget_exceeded_after_seal")
            continue
        if post <= row["measured_wall_s"]:
            break
        tail = reserve()
        if not overrun and post + tail >= remaining:
            row = revoke("seal_tail_reserve_insufficient")
            continue
        row = terminal(row, status, post + tail, tail)
    else:
        row.update(complete=False, status="publishing",
                   incomplete_reason="seal_tail_reserve_unverifiable")
        row = publish(row)
        return False
    return not overrun and row.get("complete") is True


# --- registration --------------------------------------------------------------

def _expected_keys(demos: list[str], val31: list[str]) -> dict:
    s_keys = ([f"teacher:{d}" for d in demos]
              + [f"model:{d}:{m}:{v}" for d in demos for m in MODELS for v in VERSIONS]
              + [f"roundtrip:{d}:{v}" for d in demos for v in ("V0", "V2")])
    v_keys = ([f"teacher:{d}" for d in val31]
              + [f"model:{d}:{m}:{v}" for d in val31 for m in MODELS for v in ("V0", "V2")])
    return {"fixed_windows": s_keys, "val31": v_keys}


def _source_checks(reg: dict, selection: dict, att_reg: dict,
                   deadline: float) -> dict:
    from ireset.data import w1_training as w1

    problems = []
    manifest, prereg, cache_ident = raw._w1_de_inputs(
        reg, ROOT / reg["training_manifest"]["path"], ROOT / reg["cache"]["dir"],
        ROOT, deadline=deadline)
    stages = {}
    for stage in driver.STAGE_DIRS:
        _, stage_problems = driver.stage_state(SOURCE, stage, ROOT)
        stages[stage] = stage_problems
        problems += [f"source_stage:{stage}:{p}" for p in stage_problems]
    budgets = {d: driver.domain_history_cost_s(SOURCE, d) for d in driver.DOMAINS}
    for domain, value in budgets.items():
        if value is None:
            problems.append(f"source_budget_unknown:{domain}")
        elif abs(value - att_reg["source_budgets_s"][domain]) > 1e-9:
            problems.append(f"source_budget_mismatch:{domain}")
    final = read_json(SOURCE / "decisions/attempt_001/w1_de_final.json")
    if (final.get("complete") is not True
            or final.get("motion_baseline") != "complete"
            or final.get("final_interaction_method", {}).get("verdict") != "not_ready"):
        problems.append("source_final_not_closed_baseline")
    for model, want in EXPECTED_CKPTS.items():
        rec = selection["models"][model]
        if (rec.get("sha256") != want or rec.get("step") != 4000
                or full_sha(ROOT / rec["path"]) != want):
            problems.append(f"checkpoint_unregistered:{model}")
    raw_rows = read_json(SOURCE / "evaluation/raw/attempt_001/raw_eval.json")["rows"]
    if len(raw_rows) != 155 or len({r.get("key") for r in raw_rows}) != 155:
        problems.append("source_155_matrix_incomplete")
    for row in raw_rows:
        rec = row.get("product") or {}
        if not rec.get("path") or full_sha(ROOT / rec["path"]) != rec.get("sha256"):
            problems.append(f"source_155_product_drift:{row.get('key')}")
    # Attribution artifacts: all 13 scenes x 6 variants + 26 decoded windows.
    att_anchor_problems = attribution.hash_problems(att_reg["anchors"])
    problems += [f"attribution:{p}" for p in att_anchor_problems]
    scene_variants = ("teacher", "teacher_roundtrip", "deterministic", "diffusion",
                      "deterministic_all_contributors", "diffusion_all_contributors")
    windows = []
    products = []
    for demo in att_reg["demos"]:
        result_path = ATTRIBUTION / "diagnosis/attempt_001" / demo / "result.json"
        if not result_path.is_file():
            problems.append(f"attribution_scene_missing:{demo}")
            continue
        result = read_json(result_path)
        if result.get("status") != "ok" or {r.get("variant") for r in result["rows"]} != set(scene_variants):
            problems.append(f"attribution_scene_incomplete:{demo}")
        for row in result["rows"]:
            rec = row.get("product") or {}
            if not rec.get("path") or full_sha(ROOT / rec["path"]) != rec.get("sha256"):
                problems.append(f"attribution_product_drift:{demo}:{row.get('variant')}")
            else:
                products.append(rec)
        for model in MODELS:
            rec = result.get("models", {}).get(model, {}).get("windows") or {}
            if not rec.get("path") or full_sha(ROOT / rec["path"]) != rec.get("sha256"):
                problems.append(f"attribution_window_drift:{demo}:{model}")
            else:
                windows.append(rec)
    closure = read_json(ATTRIBUTION / "decisions/diagnostic_closure.json")
    if closure.get("complete") is not True:
        problems.append("attribution_closure_not_complete")
    for rel in ("src/ireset/model/generator.py", "src/ireset/model/stitch.py",
                "src/ireset/model/tokens.py", "scripts/eval_generation.py",
                "scripts/omniretarget_w1_de.py", "scripts/diagnose_w1_attribution.py"):
        if not (ROOT / rel).is_file():
            problems.append(f"implementation_dependency_missing:{rel}")
    if problems:
        raise ValueError(f"registration source checks failed: {problems}")
    return {"manifest": manifest, "prereg": prereg, "cache_identity": cache_ident,
            "stages": stages, "budgets_s": budgets,
            "checkpoint_selection": selection,
            "attribution_windows": windows, "attribution_products": products,
            "source_raw_rows": raw_rows}


def register(run: Path) -> None:
    t0 = time.perf_counter()
    run = guard_run(run, SOURCE, ATTRIBUTION)
    # All nine plan anchors + the plan itself, re-hashed from current bytes.
    anchors = dict(PLAN_ANCHORS)
    anchors[record_path(PLAN)] = full_sha(PLAN)
    problems = _hash_map(anchors)
    if problems:
        raise ValueError(problems)
    # The attribution run's 523 anchors are the read-only source closure.
    att_reg_path = ATTRIBUTION / "freeze/registration.json"
    att_reg = read_json(att_reg_path)
    if att_reg.get("execution_id") != "w1_attribution_v1":
        raise ValueError("wrong attribution execution")
    if len(att_reg["anchors"]) != 523:
        raise ValueError("attribution anchor closure is not 523 items")
    if attribution.hash_problems(att_reg["anchors"]):
        raise ValueError(attribution.hash_problems(att_reg["anchors"]))
    w1_reg_path = SOURCE / "freeze/execution_registration.json"
    w1_reg = read_json(w1_reg_path)
    selection = read_json(SOURCE / "evaluation/checkpoint_selection.json")
    train6 = _select_train6(w1_reg["rosters"]["train24"])
    val6 = list(w1_reg["rosters"]["val6"])
    demos = train6 + val6 + [KNOWN_CASE]
    if len(demos) != 13 or len(set(demos)) != 13 or KNOWN_CASE in train6 + val6:
        raise ValueError("fixed scene denominator changed")
    val31 = list(w1_reg["rosters"]["val31"])
    if len(val31) != 31 or len(set(val31)) != 31:
        raise ValueError("val31 roster changed")
    cohort = cohort_id(full_sha(w1_reg_path))
    history = _cohort_history(run, cohort)
    if not history.get("known"):
        raise SystemExit(f"same-cohort history unknown: {history.get('reason')}")
    budget = budget_for(run, "register", history)
    (run / STAGE_DIRS["register"] / "attempt_001").mkdir(parents=True, exist_ok=True)
    _preregister_attempt(run, "register", "attempt_001", t0, cohort_id=cohort)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    source = _source_checks(w1_reg, selection, att_reg, deadline)
    implementation = driver.implementation_identity(ROOT)
    environment = driver.env_record(ROOT)
    snapshot = implementation_snapshot(run)
    keys = _expected_keys(demos, val31)
    if len(keys["fixed_windows"]) != 117 or len(keys["val31"]) != 155:
        raise ValueError("registered key-set size is wrong")
    # The attribution 523-item closure is part of this registration, not just
    # a check performed inside R: F re-hashes every consumption anchor.
    anchors.update(att_reg["anchors"])
    anchors.update({
        record_path(att_reg_path): full_sha(att_reg_path),
        record_path(w1_reg_path): full_sha(w1_reg_path),
        record_path(SOURCE / "evaluation/checkpoint_selection.json"):
            full_sha(SOURCE / "evaluation/checkpoint_selection.json"),
        record_path(ROOT / w1_reg["training_manifest"]["path"]):
            full_sha(ROOT / w1_reg["training_manifest"]["path"]),
        record_path(ROOT / w1_reg["cache"]["dir"] / "cache_identity.json"):
            full_sha(ROOT / w1_reg["cache"]["dir"] / "cache_identity.json"),
        record_path(ROOT / w1_reg["cache"]["dir"] / "root_stats.npz"):
            full_sha(ROOT / w1_reg["cache"]["dir"] / "root_stats.npz"),
    })
    for rel in IMPLEMENTATION_FILES:
        anchors[rel] = full_sha(ROOT / rel)
    for rec in source["attribution_windows"] + source["attribution_products"]:
        anchors[rec["path"]] = rec["sha256"]
    for row in source["source_raw_rows"]:
        anchors[row["product"]["path"]] = row["product"]["sha256"]
    if _hash_map(anchors):
        raise ValueError(_hash_map(anchors))
    registration = {
        "execution_id": EXECUTION_ID, "cohort_id": cohort,
        "protocol": PROTOCOL, "fusion_parameters": FUSION_PARAMETERS,
        "fusion_parameter_sha256": fusion_parameter_sha(),
        "source_run": record_path(SOURCE), "attribution_run": record_path(ATTRIBUTION),
        "source_registration": file_record(w1_reg_path),
        "source_final": file_record(SOURCE / "decisions/attempt_001/w1_de_final.json"),
        "attribution_registration": file_record(att_reg_path),
        "attribution_report": file_record(
            ATTRIBUTION / "diagnosis/attempt_001/attribution.json"),
        "attribution_closure": file_record(
            ATTRIBUTION / "decisions/diagnostic_closure.json"),
        "plan": file_record(PLAN), "implementation": implementation,
        "implementation_snapshot": snapshot, "environment": environment,
        "checkpoints": {m: selection["models"][m] for m in MODELS},
        "train6": train6, "val6": val6, "known_case": KNOWN_CASE,
        "demos": demos, "val31": val31,
        "seed": SEED, "ddim_steps": DDIM_STEPS,
        "expected_keys": keys,
        "expected_counts": {"fixed_windows": 117, "val31": 155,
                            "fixed_windows_teacher": 13, "fixed_windows_model": 78,
                            "fixed_windows_roundtrip": 26,
                            "val31_teacher": 31, "val31_model": 124},
        "budgets_s": {k: v[0] for k, v in STAGE_BUDGETS.items()},
        "reserves_s": {k: v[1] for k, v in STAGE_BUDGETS.items()},
        "total_budget_s": TOTAL_BUDGET_S,
        "cohort_history": history,
        "parent_runs": history.get("runs", []),
        "prior_cohort_cost_s": history.get("total_s", 0.0),
        "source_stage_states": source["stages"],
        "source_budgets_s": source["budgets_s"],
        "source_manifest": w1_reg["training_manifest"],
        "source_cache": w1_reg["cache"],
        "validation_config": w1_reg["evaluation"]["validation_config"],
        "task_gate_tol_m": w1_reg["evaluation"]["task_gate"]["tol_m"],
        "exclude": w1_reg["evaluation"]["exclude"],
        "anchors": anchors, "anchor_count": len(anchors),
        "training_steps": 0, "production_changes": False,
        "created_wall_s": time.perf_counter() - t0,
    }
    if time.perf_counter() - t0 >= budget["remaining_s"] - budget["reserve_s"]:
        raise TimeoutError("registration exhausted its budget")
    write_json(run / "freeze/execution_registration.json", registration, exclusive=True)
    row = {"complete": True, "status": "done",
           "report": record_path(run / "freeze/execution_registration.json"),
           "artifacts": {record_path(run / "freeze/execution_registration.json"):
                         full_sha(run / "freeze/execution_registration.json")}}
    ok = _finish_stage(run, "register", "attempt_001", t0, budget, row,
                       {"anchor_count": len(anchors), "train6": train6,
                        "val6": val6, "demos": len(demos), "val31": len(val31)})
    if not ok:
        raise SystemExit("registration did not seal")
    print(json.dumps({"registered": record_path(run), "cohort": cohort,
                      "anchors": len(anchors), "train6": train6,
                      "val6": val6}, ensure_ascii=False), flush=True)


# --- shared verification -------------------------------------------------------

def verify_registration(run: Path) -> dict:
    reg_path = run / "freeze/execution_registration.json"
    reg = read_json(reg_path)
    if reg.get("execution_id") != EXECUTION_ID or reg.get("protocol") != PROTOCOL:
        raise ValueError("wrong rotation-fusion execution")
    if reg.get("fusion_parameters") != FUSION_PARAMETERS:
        raise ValueError("fusion parameters drifted from code")
    problems = _hash_map(reg["anchors"])
    if problems:
        raise ValueError(problems)
    if reg.get("implementation") != driver.implementation_identity(ROOT):
        raise ValueError("implementation identity drifted")
    if reg.get("implementation_snapshot", {}).get("files"):
        for rel, rec in reg["implementation_snapshot"]["files"].items():
            if full_sha(ROOT / rel) != rec["sha256"]:
                raise ValueError(f"implementation file drifted:{rel}")
    return reg


def verify_source_registration(reg: dict) -> tuple[dict, dict, dict, dict]:
    w1_reg = read_json(SOURCE / "freeze/execution_registration.json")
    selection = read_json(SOURCE / "evaluation/checkpoint_selection.json")
    att_reg = read_json(ATTRIBUTION / "freeze/registration.json")
    if (full_sha(SOURCE / "freeze/execution_registration.json")
            != reg["source_registration"]["sha256"]
            or full_sha(ATTRIBUTION / "freeze/registration.json")
            != reg["attribution_registration"]["sha256"]):
        raise ValueError("source registration drifted")
    return w1_reg, selection, att_reg, driver.implementation_identity(ROOT)


# --- fusion and path evidence --------------------------------------------------

def _v0_fuse(decoded, starts: list[int], T: int):
    return fuse_windows(list(decoded[0]), list(decoded[1]), list(decoded[2]),
                        starts, T, STRIDE)


def _v1_fuse(decoded, starts: list[int], T: int):
    return attribution.fuse_all_contributors(decoded, starts, T, STRIDE)


def _decoded_from_npz(path: Path) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[int]]:
    with np.load(path, allow_pickle=False) as z:
        decoded = tuple(np.array(z[f"decoded_{k}"]) for k in ("pos", "quat", "q"))
        starts = [int(v) for v in z["starts"].tolist()]
    for value, shape in zip(decoded, ((len(starts), WINDOW, 3),
                                      (len(starts), WINDOW, 4),
                                      (len(starts), WINDOW, 29))):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"decoded window artifact invalid:{path}")
    norms = np.linalg.norm(decoded[1], axis=-1)
    if np.any(np.abs(norms - 1.0) > 1e-5):
        raise ValueError("decoded quaternion norm out of tolerance")
    return decoded, starts


def _save_fusion_path(path: Path, decoded, starts: list[int], result,
                      rejection: RotationFusionRejected | None = None,
                      *, frame_dt: float = FRAME_DT_DEFAULT) -> dict:
    """Full path evidence: original/lifted quats, signs, alpha, norms, speed."""
    payload = {
        "original_pos": decoded[0], "original_quat": decoded[1], "original_q": decoded[2],
        "starts": np.asarray(starts, dtype=np.int64),
        "frame_dt": np.asarray(float(frame_dt)),
    }
    if result is not None:
        # alpha_matrix is [N,T]; inactive entries are exactly zero and the
        # active mask makes the zero-vs-inactive distinction explicit.
        T = len(result.quat)
        alpha = np.zeros((result.weights.shape[0], T), dtype=np.float64)
        active = np.zeros((result.weights.shape[0], T), dtype=bool)
        for t, windows in enumerate(result.diagnostics["frame_active_windows"]):
            for i, a in zip(windows, result.diagnostics["frame_alpha"][t]):
                alpha[i, t] = a
                active[i, t] = True
        dots = np.asarray([np.nan if v is None else v
                           for v in result.diagnostics["window_activation_dots"]])
        payload.update({
            "lifted_quat": result.lifted_quat, "weights": result.weights,
            "fused_pos_clip": result.pos, "fused_quat_clip": result.quat,
            "fused_joints_clip": result.joints,
            "blend_norm": np.asarray(result.diagnostics["blend_norm"]),
            "alpha": alpha, "active": active,
            "window_signs": np.asarray(result.diagnostics["window_signs"]),
            "window_initial_signs": np.asarray(result.diagnostics["window_initial_sign"]),
            "activation_frames": np.asarray(result.diagnostics["window_activation_frames"]),
            "activation_dots": dots,
            "lift_flips_json": np.asarray(json.dumps(result.diagnostics["window_lift_flips"])),
            "angular_speed_radps": np.asarray(result.diagnostics["angular_speed_radps"]),
            "pre_fusion_angular_speed_radps": np.asarray(
                angular_speed_radps(_v0_fuse(decoded, starts, T)[1], 1 / 30)),
        })
    else:
        payload.update({
            "prefix_pos": rejection.prefix["pos"],
            "prefix_quat": rejection.prefix["quat"],
            "prefix_joints": rejection.prefix["joints"],
            "rejection_frame": np.asarray(rejection.frame if rejection.frame is not None else -1),
        })
    np.savez_compressed(path, **payload)
    evidence = {
        "path": file_record(path),
        "rejection": None if rejection is None else {
            "code": rejection.code, "frame": rejection.frame,
            "window": rejection.window, "local_frame": rejection.local_frame,
            "evidence": rejection.evidence,
            "prefix_length": rejection.prefix["length"] if rejection.prefix else None,
            "diagnostics": rejection.diagnostics,
        },
    }
    if result is not None:
        evidence["protocol"] = result.protocol
        evidence["consumer_eligible"] = result.consumer_eligible
        evidence["fusion_temporal_pass"] = result.fusion_temporal_pass
        evidence["diagnostics"] = result.diagnostics
    return evidence


def _seam_angle_deg(speed: np.ndarray, seam_idx: np.ndarray,
                    frame_dt: float) -> float:
    if not len(seam_idx):
        return 0.0
    # speed is rad/s; the seam field is the per-frame rotation step.
    return float(np.rad2deg(float(speed[seam_idx].max()) * frame_dt))


def _measure_clip(validator, replayer, clip, teacher_scene, teacher_kpts,
                  root, joints, demo: str, task_tol_m: float, deadline: float,
                  tag: str) -> dict:
    """Independent validator + frame-aligned metrics for one full product."""
    import eval_generation as raw_mod

    cond = clip.condition
    root, joints = np.asarray(root), np.asarray(joints)
    expected = (clip.T, 7), (clip.T, 29)
    if root.shape != expected[0] or joints.shape != expected[1]:
        raise ValueError(f"{tag}: product shape invalid")
    if not (np.isfinite(root).all() and np.isfinite(joints).all()):
        return {"status": "measurement_incomplete", "measurement_gaps": ["nonfinite_product"]}
    scene = np.concatenate([root, joints, cond.target_obj_pose], axis=1)
    _deadline(deadline, f"{tag}:validate")
    report = validator.validate(scene, exclude=raw_mod.W1_DE_EXCLUDE)
    _deadline(deadline, f"{tag}:series")
    series = raw_mod.w1_de_frame_series(
        replayer, scene, cond.human_kps_world, cond.target_obj_pose, deadline=deadline)
    _deadline(deadline, f"{tag}:metrics")
    metrics = raw_mod.w1_de_row_metrics(
        demo, scene, cond, teacher_scene, teacher_kpts, series, replayer,
        validator, task_tol_m, report=report, deadline=deadline)
    measurement_gaps = list(metrics.get("measurement_gaps") or [])
    if report.schedule is None and not measurement_gaps:
        measurement_gaps.append("validator_pre_gate")
    speed = angular_speed_radps(root[:, 3:7], cond.frame_dt)
    seams = sorted({s - 1 for s in clip_starts(clip.T)[1:]} |
                   {s for s in clip_starts(clip.T)[1:]})
    seam_idx = np.asarray([i for i in seams if 0 <= i < len(speed)], dtype=int)
    pen_key = np.maximum(series["pen_key"], 0.0)
    keyframe_count = int((pen_key > 0.002).sum())
    hand_fk = {}
    for side in ("l", "r"):
        idx = raw_mod.KPT_IDS.index(f"kpt_{side}_wrist")
        hand_fk[side] = float(np.linalg.norm(
            series["robot_kpts"][:, idx] - teacher_kpts[:, idx], axis=1).mean())
    fit_deg = raw_mod.geodesic_deg(root[:, 3:7], teacher_scene[:, 3:7])
    product = {
        "root": root, "joints": joints, "target_box": cond.target_obj_pose,
        "world_from_clip": cond.world_from_clip,
        "window_starts": np.asarray(clip_starts(clip.T), dtype=np.int64),
        "task_tol_m": task_tol_m,
        "pen_key": pen_key, "box_ground": series["box_ground"],
        "gap_l": series["gap"]["l"], "gap_r": series["gap"]["r"],
        "kpt_err_w": series["kpt_err_w"], "kpt_err_b": series["kpt_err_b"],
        "robot_kpts": series["robot_kpts"],
        "teacher_fit_per_frame_m": np.linalg.norm(
            raw_mod.world_to_box_geom(series["robot_kpts"], cond.target_obj_pose[:clip.T])
            - raw_mod.world_to_box_geom(teacher_kpts, cond.target_obj_pose[:clip.T]), axis=-1).mean(1),
        "angular_speed_radps": speed,
        "root_rotation_error_deg": fit_deg,
    }
    return {
        "status": "measured" if not measurement_gaps else "measurement_incomplete",
        "metrics": metrics, "measurement_gaps": measurement_gaps,
        "product_arrays": product,
        "root_position_max_error_m": float(np.linalg.norm(
            root[:, :3] - teacher_scene[:, :3], axis=1).max()),
        "root_rotation_max_error_deg": float(fit_deg.max()),
        "joint_max_error_rad": float(np.abs(joints - clip.q).max()),
        "hand_fk_mean_m": hand_fk,
        "angular_velocity_max_radps": float(speed.max()) if len(speed) else 0.0,
        "seam_rotation_max_deg": _seam_angle_deg(speed, seam_idx, cond.frame_dt),
        "keyframe_penetration_count": keyframe_count,
    }


def _deadline(deadline: float | None, name: str) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError(f"deadline exceeded:{name}")


def clip_starts(T: int) -> list[int]:
    return window_starts(T, WINDOW, STRIDE)


def _save_product(path: Path, root, joints, metric: dict, extra: dict | None = None) -> dict:
    arrays = dict(metric["product_arrays"])
    if extra:
        arrays.update(extra)
    np.savez_compressed(path, **arrays)
    return file_record(path)


def _row_from_measurement(key: str, demo: str, group: str, variant: str,
                          metric: dict, product: Path | None,
                          path_evidence: dict | None = None,
                          binding: dict | None = None) -> dict:
    row = {
        "key": key, "demo": demo, "group": group, "variant": variant,
        "status": metric.get("status", "measurement_incomplete"),
        "metrics": metric.get("metrics"),
        "measurement_gaps": metric.get("measurement_gaps", []),
        "product": file_record(product) if product is not None else None,
        "path_evidence": path_evidence,
        "binding": binding or {},
        "root_position_max_error_m": metric.get("root_position_max_error_m"),
        "root_rotation_max_error_deg": metric.get("root_rotation_max_error_deg"),
        "joint_max_error_rad": metric.get("joint_max_error_rad"),
        "hand_fk_mean_m": metric.get("hand_fk_mean_m"),
        "angular_velocity_max_radps": metric.get("angular_velocity_max_radps"),
        "seam_rotation_max_deg": metric.get("seam_rotation_max_deg"),
        "keyframe_penetration_count": metric.get("keyframe_penetration_count"),
    }
    return row


# --- fixed-windows unit --------------------------------------------------------

def _load_fixed_context(run: Path, demo: str, *, reg: dict | None = None,
                        source: tuple | None = None):
    import eval_generation as raw_mod
    from ireset.data import w1_training as w1
    from ireset.eval.teacher_comparison import TeacherQualityReplayer
    from ireset.kinematics.mujoco_adapter import G1Kinematics
    from ireset.repair.validation import IndependentValidator, ValidationConfig

    if reg is None:
        reg = verify_registration(run)
    if source is None:
        w1_reg, selection, att_reg, _ = verify_source_registration(reg)
    else:
        w1_reg, selection, att_reg, _ = source
    if demo not in reg["demos"] and demo not in reg["val31"]:
        raise ValueError("unregistered demonstration")
    manifest = read_json(ROOT / w1_reg["training_manifest"]["path"])
    prereg = read_json(ROOT / w1_reg["preregistration"]["path"])
    hits = [r for r in manifest["rows"] if r.get("demo") == demo]
    if len(hits) != 1 or hits[0].get("admission", {}).get("motion") is not True:
        raise ValueError("demo not uniquely admitted")
    cache = ROOT / w1_reg["cache"]["dir"]
    clip = w1.load_w1_derived_clip(
        hits[0], cache / "schedules", ROOT,
        prereg["assets"]["a1_assets_sha256"], w1_reg["condition_code_sha256"])
    validator = IndependentValidator(cfg=ValidationConfig(**w1_reg["evaluation"]["validation_config"]))
    replayer = TeacherQualityReplayer(validator)
    teacher_scene = np.concatenate((clip.root, clip.q, clip.condition.target_obj_pose), axis=1)
    teacher_kpts = raw_mod.fk_keypoints(G1Kinematics(), teacher_scene)
    return reg, w1_reg, selection, att_reg, clip, validator, replayer, teacher_scene, teacher_kpts


def _original_product(att_result: dict, variant: str) -> Path:
    row = next(r for r in att_result["rows"] if r.get("variant") == variant)
    return ROOT / row["product"]["path"], row["product"]["sha256"]


def fixed_unit(run: Path, demo: str, out: Path, deadline: float) -> dict:
    t0 = time.perf_counter()
    import eval_generation as raw_mod

    (reg, w1_reg, selection, att_reg, clip, validator, replayer,
     teacher_scene, teacher_kpts) = _load_fixed_context(run, demo)
    out.mkdir(parents=True, exist_ok=False)
    group = ("train6" if demo in reg["train6"] else
             "val6" if demo in reg["val6"] else "known_case")
    T = clip.T
    starts = clip_starts(T)
    if starts != list(read_json(ATTRIBUTION / "diagnosis/attempt_001" / demo / "result.json")["starts"]):
        raise ValueError("fixed-window starts drifted")
    att_result = read_json(ATTRIBUTION / "diagnosis/attempt_001" / demo / "result.json")
    if att_result.get("status") != "ok":
        raise ValueError("attribution scene is not closed")
    task_tol = reg["task_gate_tol_m"]
    rows = []
    facts = {"demo": demo, "group": group, "T": T, "starts": starts,
             "attribution_status": att_result.get("status")}

    def measure(root, joints, tag):
        return _measure_clip(validator, replayer, clip, teacher_scene, teacher_kpts,
                             root, joints, demo, task_tol, deadline, tag)

    # 1) frozen teacher reference
    teacher_metric = measure(clip.root, clip.q, "teacher")
    teacher_product = out / "teacher.npz"
    teacher_record = _save_product(teacher_product, clip.root, clip.q, teacher_metric)
    rows.append(_row_from_measurement(
        f"teacher:{demo}", demo, group, "teacher", teacher_metric, teacher_product,
        binding={"frozen_label_root_sha256": hashlib.sha256(clip.root.tobytes()).hexdigest(),
                 "frozen_label_q_sha256": hashlib.sha256(clip.q.tobytes()).hexdigest()}))

    # 2) saved decoded windows: V0/V1/V2, no model calls
    for model in MODELS:
        rec = att_result["models"][model]["windows"]
        window_path = ROOT / rec["path"]
        if full_sha(window_path) != rec["sha256"]:
            raise ValueError(f"decoded window drift:{model}")
        decoded, loaded_starts = _decoded_from_npz(window_path)
        if loaded_starts != starts:
            raise ValueError("decoded window starts differ")
        decoded_before = tuple(a.copy() for a in decoded)
        model_facts = {"decoded_window": rec, "starts": starts}
        v0 = _v0_fuse(decoded, starts, T)
        v0_root, v0_joints = attribution.world_state(*v0[:3], clip.condition.world_from_clip)
        v0_path, v0_sha = _original_product(att_result, model)
        with np.load(v0_path, allow_pickle=False) as z:
            model_facts["v0_bitwise"] = bool(
                np.array_equal(z["root"], v0_root) and np.array_equal(z["joints"], v0_joints))
        v1 = _v1_fuse(decoded, starts, T)
        v1_root, v1_joints = attribution.world_state(*v1[:3], clip.condition.world_from_clip)
        v1_path, v1_sha = _original_product(att_result, model + "_all_contributors")
        with np.load(v1_path, allow_pickle=False) as z:
            model_facts["v1_bitwise"] = bool(
                np.array_equal(z["root"], v1_root) and np.array_equal(z["joints"], v1_joints))
        model_facts["v0_product"] = {"path": record_path(v0_path), "sha256": v0_sha}
        model_facts["v1_product"] = {"path": record_path(v1_path), "sha256": v1_sha}

        for version, fused, fused_root, fused_joints in (
                ("V0", v0, v0_root, v0_joints),
                ("V1", v1, v1_root, v1_joints)):
            metric = measure(fused_root, fused_joints, f"{model}:{version}")
            product = out / f"{model}_{version}.npz"
            _save_product(product, fused_root, fused_joints, metric)
            rows.append(_row_from_measurement(
                f"model:{demo}:{model}:{version}", demo, group, f"{model}:{version}",
                metric, product, binding=dict(model_facts)))

        try:
            v2 = fuse_windows_v2(decoded[0], decoded[1], decoded[2], starts, T,
                                 STRIDE, clip.condition.frame_dt)
        except RotationFusionRejected as exc:
            path_npz = out / f"{model}_V2_path.npz"
            path_evidence = _save_fusion_path(
                path_npz, decoded, starts, None, rejection=exc,
                frame_dt=clip.condition.frame_dt)
            model_facts.update(v2_status="algorithm_rejected",
                               v2_rejection=path_evidence["rejection"])
            rows.append(_row_from_measurement(
                f"model:{demo}:{model}:V2", demo, group, f"{model}:V2",
                {"status": "algorithm_rejected",
                 "measurement_gaps": [f"algorithm_rejected:{exc.code}"],
                 "metrics": None}, None, path_evidence=path_evidence,
                binding=dict(model_facts)))
        else:
            v2_root, v2_joints = attribution.world_state(
                v2.pos, v2.quat, v2.joints, clip.condition.world_from_clip)
            model_facts.update(
                v2_status="measured",
                v2_position_joint_bitwise=bool(
                    np.array_equal(v2.pos, v0[0]) and np.array_equal(v2.joints, v0[2])),
                v2_world_position_bitwise=bool(np.array_equal(v2_root[:, :3], v0_root[:, :3])),
                inputs_unmodified=all(np.array_equal(a, b) for a, b in zip(decoded, decoded_before)),
                blend_norm_min=float(np.min(v2.diagnostics["blend_norm"])),
                window_signs=v2.diagnostics["window_signs"],
                activation_frames=v2.diagnostics["window_activation_frames"],
                angular_speed_global_max_radps=v2.diagnostics["angular_speed_global_max_radps"],
                angular_speed_global_max_frame=v2.diagnostics["angular_speed_global_max_frame"],
                fusion_temporal_pass=v2.fusion_temporal_pass,
                consumer_eligible=v2.consumer_eligible,
            )
            path_npz = out / f"{model}_V2_path.npz"
            path_evidence = _save_fusion_path(
                path_npz, decoded, starts, v2,
                frame_dt=clip.condition.frame_dt)
            metric = measure(v2_root, v2_joints, f"{model}:V2")
            product = out / f"{model}_V2.npz"
            _save_product(product, v2_root, v2_joints, metric)
            rows.append(_row_from_measurement(
                f"model:{demo}:{model}:V2", demo, group, f"{model}:V2",
                metric, product, path_evidence=path_evidence, binding=dict(model_facts)))

    # 3) teacher encode/decode roundtrip: one oracle session per demo.
    oracle = _oracle_decoded(selection, w1_reg, clip, deadline)
    oracle_path = out / "oracle_windows.npz"
    np.savez_compressed(oracle_path,
                        **{f"decoded_{k}": v for k, v in zip(("pos", "quat", "q"), oracle[0])},
                        starts=np.asarray(oracle[1], dtype=np.int64))
    oracle_starts = oracle[1]
    if oracle_starts != starts:
        raise ValueError("oracle starts differ")
    rt0 = _v0_fuse(oracle[0], oracle_starts, T)
    rt0_root, rt0_joints = attribution.world_state(*rt0[:3], clip.condition.world_from_clip)
    rt0_path, rt0_sha = _original_product(att_result, "teacher_roundtrip")
    with np.load(rt0_path, allow_pickle=False) as z:
        rt_binding = {
            "v0_bitwise": bool(np.array_equal(z["root"], rt0_root)
                               and np.array_equal(z["joints"], rt0_joints)),
            "v0_product": {"path": record_path(rt0_path), "sha256": rt0_sha},
            "oracle_windows": file_record(oracle_path),
            "oracle_calls": 1,
        }
    rt0_metric = measure(rt0_root, rt0_joints, "roundtrip:V0")
    rt0_product = out / "roundtrip_V0.npz"
    _save_product(rt0_product, rt0_root, rt0_joints, rt0_metric)
    rows.append(_row_from_measurement(
        f"roundtrip:{demo}:V0", demo, group, "roundtrip:V0", rt0_metric,
        rt0_product, binding=rt_binding))
    try:
        rt2 = fuse_windows_v2(oracle[0][0], oracle[0][1], oracle[0][2],
                              oracle_starts, T, STRIDE, clip.condition.frame_dt)
    except RotationFusionRejected as exc:
        path_npz = out / "roundtrip_V2_path.npz"
        path_evidence = _save_fusion_path(
            path_npz, oracle[0], oracle_starts, None, rejection=exc,
            frame_dt=clip.condition.frame_dt)
        rt_binding.update(v2_status="algorithm_rejected",
                          v2_rejection=path_evidence["rejection"])
        rows.append(_row_from_measurement(
            f"roundtrip:{demo}:V2", demo, group, "roundtrip:V2",
            {"status": "algorithm_rejected",
             "measurement_gaps": [f"algorithm_rejected:{exc.code}"], "metrics": None},
            None, path_evidence=path_evidence, binding=rt_binding))
    else:
        rt2_root, rt2_joints = attribution.world_state(
            rt2.pos, rt2.quat, rt2.joints, clip.condition.world_from_clip)
        rt_binding.update(
            v2_status="measured",
            v2_position_joint_bitwise=bool(np.array_equal(rt2.pos, rt0[0])
                                           and np.array_equal(rt2.joints, rt0[2])),
            v2_world_position_bitwise=bool(np.array_equal(rt2_root[:, :3], rt0_root[:, :3])),
            fusion_temporal_pass=rt2.fusion_temporal_pass)
        path_npz = out / "roundtrip_V2_path.npz"
        path_evidence = _save_fusion_path(
            path_npz, oracle[0], oracle_starts, rt2,
            frame_dt=clip.condition.frame_dt)
        rt2_metric = measure(rt2_root, rt2_joints, "roundtrip:V2")
        rt2_product = out / "roundtrip_V2.npz"
        _save_product(rt2_product, rt2_root, rt2_joints, rt2_metric)
        rows.append(_row_from_measurement(
            f"roundtrip:{demo}:V2", demo, group, "roundtrip:V2", rt2_metric,
            rt2_product, path_evidence=path_evidence, binding=rt_binding))

    # Known-case mechanism measurements are recorded even when they are bad.
    if demo == KNOWN_CASE:
        facts["known_case"] = _known_case_facts(
            rows, out, frame_dt=clip.condition.frame_dt)

    result = {"demo": demo, "group": group, "T": T, "starts": starts,
              "rows": rows, "facts": facts, "status": "ok",
              "wall_s": time.perf_counter() - t0}
    required = [f"teacher:{demo}"] + [
        f"model:{demo}:{m}:{v}" for m in MODELS for v in VERSIONS] + [
        f"roundtrip:{demo}:{v}" for v in ("V0", "V2")]
    if [r["key"] for r in rows] != required:
        result["status"] = "incomplete"
    write_json(out / "result.json", result)
    return result


def _known_case_facts(rows: list[dict], scene_dir: Path,
                      frame_dt: float | None = None) -> dict:
    """Recompute frame speeds from the fused rotation, never from a saved array.

    Availability follows the row's recomputed status: a rejected path has no
    consumable rotation regardless of which arrays a file happens to contain.
    """
    known = {}
    for model in MODELS:
        row = next((r for r in rows
                    if r.get("key") == f"model:{KNOWN_CASE}:{model}:V2"), None)
        path = scene_dir / f"{model}_V2_path.npz"
        if row is None or not path.is_file():
            known[model] = {"available": False, "reason": "v2_row_or_path_missing"}
            continue
        rej_raw = _mapping(row.get("path_evidence")).get("rejection")
        rejection = rej_raw if isinstance(rej_raw, dict) else None
        status = row.get("status")
        if status == "algorithm_rejected":
            known[model] = {
                "available": False, "reason": "algorithm_rejected",
                "rejection": rejection,
                "rejection_frame": (rejection or {}).get("frame"),
            }
            continue
        if status != "measured":
            known[model] = {"available": False,
                            "reason": f"v2_status:{status}", "rejection": rejection}
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                if "fused_quat_clip" not in z.files:
                    known[model] = {"available": False,
                                    "reason": "fused_rotation_missing"}
                    continue
                fused_quat = np.array(z["fused_quat_clip"])
                stored_dt = (float(z["frame_dt"]) if "frame_dt" in z.files else None)
                saved_speed = (np.array(z["angular_speed_radps"])
                               if "angular_speed_radps" in z.files else None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            known[model] = {"available": False,
                            "reason": f"path_unreadable:{type(exc).__name__}"}
            continue
        dt = frame_dt if frame_dt is not None else stored_dt
        if not (_num_cost(dt) and float(dt) > 0):
            known[model] = {"available": False, "reason": "frame_dt_missing_or_invalid"}
            continue
        if fused_quat.ndim != 2 or fused_quat.shape[1] != 4 \
                or not np.issubdtype(fused_quat.dtype, np.floating) \
                or not np.isfinite(fused_quat).all():
            known[model] = {"available": False, "reason": "fused_rotation_invalid"}
            continue
        try:
            norms = np.linalg.norm(fused_quat, axis=1)
            norm_invalid = (not np.isfinite(norms).all()
                            or np.any(norms < 1e-12)
                            or np.any(np.abs(norms - 1.0) > QUAT_NORM_TOL))
        except (TypeError, ValueError, FloatingPointError):
            norm_invalid = True
        if norm_invalid:
            known[model] = {"available": False,
                            "reason": "fused_rotation_norm_invalid"}
            continue
        try:
            speed = angular_speed_radps(fused_quat, float(dt))
        except (ValueError, TypeError, FloatingPointError) as exc:
            known[model] = {
                "available": False,
                "reason": f"rotation_speed_failed:{type(exc).__name__}"}
            continue
        speed_reproducible = bool(
            saved_speed is not None and saved_speed.shape == speed.shape
            and np.array_equal(saved_speed, speed))
        dt_matches = (None if stored_dt is None
                      else bool(stored_dt == float(dt)))
        known[model] = {
            "available": True, "recomputed_from_fused_rotation": True,
            "frame_dt": float(dt), "frame_dt_recorded": stored_dt is not None,
            "frame_dt_matches_artifact": dt_matches,
            "speed_array_reproducible": speed_reproducible,
            "frame_107_108_radps": float(speed[107]) if len(speed) > 107 else None,
            "frame_113_114_radps": float(speed[113]) if len(speed) > 113 else None,
            "global_max_radps": float(speed.max()) if len(speed) else None,
        }
    return known


def _oracle_decoded(selection: dict, w1_reg: dict, clip, deadline: float):
    import torch
    from ireset.model.generator import G1Generator

    _deadline(deadline, "oracle:checkpoint")
    cache = ROOT / w1_reg["cache"]["dir"]
    generator = raw.load_w1_de_generator(
        selection["models"]["deterministic"], "cpu", DDIM_STEPS, cache,
        w1_reg["training_manifest"]["sha256"], w1_reg["cache"]["identity_sha256"],
        w1_reg["cache"]["root_stats_sha256"],
        full_sha(SOURCE / "freeze/execution_registration.json"),
        registration=w1_reg, project_root=ROOT)
    starts = clip_starts(clip.T)
    x0 = generator.tokenizer.encode(clip.root, clip.q, clip.condition.world_from_clip).astype(np.float32)
    targets = np.stack([x0[s:s + WINDOW] for s in starts])

    class FixedPrediction(torch.nn.Module):
        def forward(self, memory, frames):
            return torch.from_numpy(targets.copy())

    class NoCondition(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros((len(starts), 1, 1))

    oracle = G1Generator(FixedPrediction(), NoCondition(), generator.tokenizer,
                         device="cpu", window=WINDOW, stride=STRIDE,
                         ddim_steps=DDIM_STEPS, resample_attempts=1)
    trajectory, saved = attribution.capture_prediction(oracle, clip.condition, SEED)
    _deadline(deadline, "oracle:done")
    return saved["decoded"], starts


def _metric_gaps(metrics) -> list[str]:
    """Required fields for a measured row; missing keys are not zero."""
    if not isinstance(metrics, dict):
        return ["metrics_not_a_mapping"]
    gaps = []
    flat = ("raw_standard_pass", "robot_strict", "task", "penetration",
            "fitting", "hands", "feet", "both")
    for key in flat:
        if key not in metrics:
            gaps.append(f"missing:{key}")
    if not gaps:
        for key in ("failed_checks",):
            if key not in (metrics["robot_strict"] or {}):
                gaps.append(f"missing:robot_strict.{key}")
        for key in ("gate_pass", "world_mpjpe_p95_m", "world_mpjpe_max_m"):
            if key not in (metrics["task"] or {}):
                gaps.append(f"missing:task.{key}")
        if "max_m" not in (metrics["penetration"] or {}):
            gaps.append("missing:penetration.max_m")
        if "teacher_fit_fk_mpjpe_box_m" not in (metrics["fitting"] or {}):
            gaps.append("missing:fitting.teacher_fit_fk_mpjpe_box_m")
        for side in ("l", "r"):
            if side not in (metrics["hands"] or {}):
                gaps.append(f"missing:hands.{side}")
            if side not in (metrics["feet"] or {}):
                gaps.append(f"missing:feet.{side}")
        if "denominator_frames" not in (metrics["both"] or {}):
            gaps.append("missing:both.denominator_frames")
    return gaps


def _roundtrip_fail(row: dict | None) -> bool:
    if row is None or row.get("status") != "measured":
        return True
    return any((row.get(field) is None or row[field] > limit) for field, limit in (
        ("root_position_max_error_m", 1e-5),
        ("root_rotation_max_error_deg", 1e-3),
        ("joint_max_error_rad", 1e-5),
    ))


def _ordered_rows(rows: list[dict], expected: list[str]) -> list[dict]:
    """Emit the registered row order; a missing/duplicate key stays visible."""
    by_key = {row.get("key"): row for row in rows}
    if len(rows) != len(expected) or set(by_key) != set(expected):
        return rows
    return [by_key[key] for key in expected]


def _teacher_status_failures(rows: list[dict],
                             demos: list[str]) -> list[str]:
    by_key = {row.get("key"): row for row in rows}
    failures = []
    for demo in demos:
        key = f"teacher:{demo}"
        row = by_key.get(key)
        if row is None:
            failures.append(f"teacher_row_missing:{key}")
        elif row.get("status") != "measured":
            failures.append(
                f"teacher_row_not_measured:{key}:{row.get('status')}")
    return failures


def _fixed_gs(reg: dict, rows: list[dict], scene_facts: dict,
              synthetic_ok: bool) -> dict:
    failures, gaps = [], []
    expected = reg["expected_keys"]["fixed_windows"]
    keys = [row.get("key") for row in rows]
    if keys != expected:
        failures.append("fixed_key_matrix_mismatch")
    by_key = {row.get("key"): row for row in rows}
    failures += _teacher_status_failures(rows, reg["demos"])
    for row in rows:
        key = row.get("key")
        status = row.get("status")
        if status == "algorithm_rejected":
            if not str(key).endswith(":V2"):
                failures.append(f"non_v2_algorithm_rejection:{key}")
                continue
            rej = (_mapping(row.get("path_evidence"))).get("rejection")
            if (not isinstance(rej, dict) or not rej.get("code")
                    or rej.get("prefix_length") is None):
                gaps.append(f"rejection_evidence_missing:{key}")
            continue
        if status != "measured":
            gaps.append(f"row_not_measured:{key}")
            continue
        metrics = _mapping(row.get("metrics"))
        if not metrics or metrics.get("measurement_gaps"):
            gaps.append(f"row_measurement_incomplete:{key}")
        gaps += [f"row_metrics_missing:{key}:{g}" for g in _metric_gaps(metrics)]
        rec = row.get("product") or {}
        if not rec.get("path") or full_sha(ROOT / rec["path"]) != rec.get("sha256"):
            gaps.append(f"row_product_drift:{key}")
    for demo in reg["demos"]:
        for model in MODELS:
            for version in ("V0", "V1"):
                row = by_key.get(f"model:{demo}:{model}:{version}")
                if row is None or not _mapping(row.get("binding")).get(
                        f"{version.lower()}_bitwise"):
                    failures.append(f"replay_not_bitwise:{demo}:{model}:{version}")
            v2 = by_key.get(f"model:{demo}:{model}:V2")
            if v2 is None:
                failures.append(f"v2_row_missing:{demo}:{model}")
            elif v2.get("status") == "measured":
                binding = _mapping(v2.get("binding"))
                if not binding.get("v2_position_joint_bitwise"):
                    failures.append(f"v2_changed_positions_or_joints:{demo}:{model}")
                if not binding.get("inputs_unmodified"):
                    failures.append(f"v2_mutated_inputs:{demo}:{model}")
        v0 = by_key.get(f"roundtrip:{demo}:V0")
        v2 = by_key.get(f"roundtrip:{demo}:V2")
        if v0 is None or not _mapping(v0.get("binding")).get("v0_bitwise"):
            failures.append(f"roundtrip_replay_not_bitwise:{demo}")
        v0_fail, v2_fail = _roundtrip_fail(v0), _roundtrip_fail(v2)
        if v2_fail and not v0_fail:
            failures.append(f"roundtrip_new_failure:{demo}")
        if v2 is not None and v2.get("status") == "measured" and v0 is not None:
            for field, limit in (
                    ("root_position_max_error_m", 1e-6),
                    ("root_rotation_max_error_deg", 1e-6),
                    ("joint_max_error_rad", 1e-6)):
                a, b = v0.get(field), v2.get(field)
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    gaps.append(f"roundtrip_increment_unmeasured:{demo}:{field}")
                elif b > a + limit:
                    failures.append(f"roundtrip_increment:{demo}:{field}")
            for side in ("l", "r"):
                a = (v0.get("hand_fk_mean_m") or {}).get(side)
                b = (v2.get("hand_fk_mean_m") or {}).get(side)
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    gaps.append(f"roundtrip_hand_fk_unmeasured:{demo}:{side}")
                elif b > a + 1e-6:
                    failures.append(f"roundtrip_hand_fk_increment:{demo}:{side}")
    original_failures = sorted(demo for demo in reg["demos"]
                               if _roundtrip_fail(by_key.get(f"roundtrip:{demo}:V0")))
    if len(original_failures) != 3:
        gaps.append("original_roundtrip_negative_gate_count_changed")
    known = (scene_facts.get(KNOWN_CASE) or {}).get("known_case") or {}
    for model in MODELS:
        info = known.get(model) or {}
        if not info.get("available"):
            failures.append(f"known_case_v2_not_computable:{model}")
        elif info.get("speed_array_reproducible") is False:
            failures.append(f"known_case_speed_array_not_reproducible:{model}")
        elif info.get("frame_dt_matches_artifact") is False:
            failures.append(f"known_case_frame_dt_not_reproducible:{model}")
        elif info.get("frame_107_108_radps") is None:
            gaps.append(f"known_case_frame_missing:{model}")
        elif info["frame_107_108_radps"] > ANGULAR_SPEED_LIMIT_RADPS:
            failures.append(f"known_case_frame_108_over_speed:{model}")
    if not synthetic_ok:
        failures.append("synthetic_math_tests_failed")
    return {"passed": bool(not failures and not gaps),
            "failures": failures, "evidence_gaps": gaps,
            "facts": {
                "original_roundtrip_failures": original_failures,
                "known_case": known,
                "synthetic_tests_passed": bool(synthetic_ok),
                "roundtrip_absolute_gates": {
                    "position_m": 1e-5, "rotation_deg": 1e-3, "joints_rad": 1e-5},
                "roundtrip_increment_tolerance": {
                    "position_m": 1e-6, "rotation_deg": 1e-6,
                    "joints_rad": 1e-6, "hand_fk_m": 1e-6},
            }}


def fixed_stage(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    cohort = _cohort_history(run, reg["cohort_id"])
    budget = budget_for(run, "fixed-windows", cohort)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    attempt = run / "fixed_windows/attempt_001"
    if attempt.exists():
        raise SystemExit("fixed-windows attempt already exists; no retry")
    attempt.mkdir(parents=True, exist_ok=True)
    _preregister_attempt(run, "fixed-windows", "attempt_001", t0,
                         cohort_id=reg["cohort_id"])
    out_rows, units, scene_facts = [], [], {}
    summary = {"schema": "w1_rotation_fusion_fixed_windows_v1",
               "execution_registration": file_record(run / "freeze/execution_registration.json"),
               "protocol": PROTOCOL, "status": "running", "complete": False,
               "gs": {"passed": False, "failures": []},
               "rows": [], "units": [], "scene_facts": scene_facts,
               "evidence_gaps": []}
    write_json(attempt / "summary.json", summary, exclusive=True)
    write_json(attempt / "rows.json", out_rows, exclusive=True)
    write_json(attempt / "units.json", units, exclusive=True)
    # Synthetic math contract must pass before any real quality is read.
    try:
        tests = _run_synthetic_tests(deadline)
    except subprocess.TimeoutExpired:
        tests = {"command": "synthetic math tests", "returncode": None,
                 "stdout_tail": "", "wall_s": 0.0}
    summary["synthetic_tests"] = tests
    if tests.get("returncode") != 0:
        summary["evidence_gaps"].append("synthetic_math_tests_failed")
        summary["status"] = "incomplete"
        write_json(attempt / "summary.json", summary)
        row = {"complete": False, "status": "incomplete",
               "summary": record_path(attempt / "summary.json"),
               "artifacts": {record_path(attempt / "summary.json"):
                             full_sha(attempt / "summary.json")}}
        _finish_stage(run, "fixed-windows", "attempt_001", t0, budget, row,
                      {"tests": False}, "incomplete")
        return False
    failures = []
    for demo in reg["demos"]:
        _deadline(deadline, "fixed-windows:next-unit")
        unit_t0 = time.perf_counter()
        timeout = min(600.0, deadline - unit_t0)
        if timeout <= 1.0:
            units.append({"demo": demo, "status": "budget", "wall_s": 0.0})
            write_json(attempt / "units.json", units)
            failures.append(f"budget_exhausted_before_unit:{demo}")
            break
        unit_out = attempt / "scenes" / demo
        argv = [sys.executable, str(Path(__file__)), "fixed-window-unit",
                "--run-root", str(run), "--demo", demo, "--out", str(unit_out),
                "--deadline", str(time.perf_counter() + timeout - 2.0)]
        env = {**os.environ, **{k: "1" for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")}}
        unit = {"demo": demo, "argv": argv, "status": "running", "wall_s": 0.0}
        units.append(unit)
        write_json(attempt / "units.json", units)
        try:
            with (attempt / f"{demo}.log").open("w") as log:
                proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=timeout)
            unit.update(returncode=proc.returncode,
                        status="done" if proc.returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            unit.update(returncode=None, status="timeout")
        finally:
            unit["wall_s"] = time.perf_counter() - unit_t0
            write_json(attempt / "units.json", units)
        result_path = unit_out / "result.json"
        if result_path.is_file():
            scene = read_json(result_path)
            out_rows.extend(scene.get("rows", []))
            scene_facts[demo] = scene.get("facts", {})
            write_json(attempt / "rows.json", out_rows)
            summary["rows"] = out_rows
            summary["scene_facts"] = scene_facts
        if unit["status"] != "done":
            failures.append(f"unit_failed:{demo}:{unit['status']}")
            break
    expected = reg["expected_keys"]["fixed_windows"]
    out_rows = _ordered_rows(out_rows, expected)
    keys = [row.get("key") for row in out_rows]
    write_json(attempt / "rows.json", out_rows)
    summary.update(rows=out_rows, units=units)
    summary["counts"] = {
        "expected": len(expected), "rows": len(out_rows), "keys_unique": len(set(keys)),
        "teacher": sum(k.startswith("teacher:") for k in keys),
        "model": sum(k.startswith("model:") for k in keys),
        "roundtrip": sum(k.startswith("roundtrip:") for k in keys),
        "measured": sum(r.get("status") == "measured" for r in out_rows),
        "algorithm_rejected": sum(r.get("status") == "algorithm_rejected"
                                  for r in out_rows),
    }
    if keys != expected:
        failures.append("fixed_key_matrix_mismatch")
    failures += _teacher_status_failures(out_rows, reg["demos"])
    known_case = _known_case_facts(
        [r for r in out_rows
         if str(r.get("key", "")).startswith(f"model:{KNOWN_CASE}:")],
        attempt / "scenes" / KNOWN_CASE)
    gs = _fixed_gs(reg, out_rows, {KNOWN_CASE: {"known_case": known_case}},
                   tests.get("returncode") == 0)
    summary["gs"] = gs
    summary["gq"] = _gq_rows(reg, out_rows, reg["demos"])
    summary["algorithm_rejections"] = [
        {"key": row["key"],
         "rejection": (row.get("path_evidence") or {}).get("rejection")}
        for row in out_rows if row.get("status") == "algorithm_rejected"]
    summary["failures"] = list(dict.fromkeys(failures + gs["failures"]))
    summary["evidence_gaps"] = list(dict.fromkeys(
        list(failures) + list(gs.get("evidence_gaps", []))))
    evidence_complete = bool(not failures and keys == expected
                             and not gs.get("evidence_gaps"))
    summary["complete"] = bool(evidence_complete)
    summary["status"] = "done" if evidence_complete else "incomplete"
    summary["cost_s"] = time.perf_counter() - t0
    summary["oracle_calls"] = 13 if evidence_complete else None
    write_json(attempt / "summary.json", summary)
    row = {"complete": evidence_complete,
           "status": "done" if evidence_complete else "incomplete",
           "summary": record_path(attempt / "summary.json"),
           "artifacts": {record_path(attempt / "summary.json"):
                         full_sha(attempt / "summary.json"),
                         record_path(attempt / "rows.json"):
                         full_sha(attempt / "rows.json")}}
    ok = _finish_stage(run, "fixed-windows", "attempt_001", t0, budget, row,
                       {"gs_passed": gs["passed"], "rows": len(out_rows),
                        "tests": True},
                       "done" if evidence_complete else "incomplete")
    print(json.dumps({"fixed_windows": summary["status"], "gs": gs["passed"],
                      "rows": len(out_rows), "failures": summary["failures"]},
                     ensure_ascii=False), flush=True)
    return ok and evidence_complete


# --- GQ: conservative no-regression comparison --------------------------------

def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _compare_lower(failures, gaps, name, before, after, tol=1e-6) -> None:
    if not _num(before) or not _num(after):
        gaps.append(f"gq_metric_missing:{name}")
    elif float(after) > float(before) + tol:
        failures.append(f"gq_regression:{name}")


def _compare_count(failures, gaps, name, before, after) -> None:
    if isinstance(before, bool) or isinstance(after, bool) \
            or not isinstance(before, int) or not isinstance(after, int):
        gaps.append(f"gq_count_invalid:{name}")
    elif after < before:
        failures.append(f"gq_regression:{name}")


def _compare_count_not_increase(failures, gaps, name, before, after) -> None:
    if isinstance(before, bool) or isinstance(after, bool) \
            or not isinstance(before, int) or not isinstance(after, int):
        gaps.append(f"gq_count_invalid:{name}")
    elif after > before:
        failures.append(f"gq_regression:{name}")


def gq_compare(demo: str, model: str, v0: dict | None, v2: dict | None) -> dict:
    failures, gaps = [], []
    if v0 is None or v0.get("status") != "measured":
        gaps.append(f"gq_v0_not_measured:{demo}:{model}")
    if v2 is None or v2.get("status") != "measured":
        failures.append(f"gq_v2_not_measured:{demo}:{model}")
    if failures or gaps:
        return {"demo": demo, "model": model, "passed": False,
                "failures": failures, "evidence_gaps": gaps, "facts": {}}
    a, b = _mapping(v0.get("metrics")), _mapping(v2.get("metrics"))
    facts = {"v0_raw_standard_pass": bool(a.get("raw_standard_pass")),
             "v2_raw_standard_pass": bool(b.get("raw_standard_pass"))}
    # Hard failures: V2 may not add a validator failure that V0 did not have.
    strict0 = _mapping(a.get("robot_strict"))
    strict2 = _mapping(b.get("robot_strict"))
    if "failed_checks" not in strict0 or "failed_checks" not in strict2:
        gaps.append(f"gq_strict_checks_missing:{demo}:{model}")
    new_failed = sorted(set(strict2.get("failed_checks") or [])
                        - set(strict0.get("failed_checks") or []))
    if new_failed:
        failures.append(f"gq_new_validator_failures:{demo}:{model}:{new_failed}")
    facts["new_validator_failures"] = new_failed
    if _mapping(a.get("task")).get("gate_pass") \
            and not _mapping(b.get("task")).get("gate_pass"):
        failures.append(f"gq_new_task_failure:{demo}:{model}")
    if a.get("raw_standard_pass") and not b.get("raw_standard_pass"):
        failures.append(f"gq_new_raw_failure:{demo}:{model}")
    _compare_lower(failures, gaps, f"penetration_max:{demo}:{model}",
                   _mapping(a.get("penetration")).get("max_m"),
                   _mapping(b.get("penetration")).get("max_m"))
    _compare_count_not_increase(failures, gaps, f"keyframe_violation_count:{demo}:{model}",
                                v0.get("keyframe_penetration_count"),
                                v2.get("keyframe_penetration_count"))
    hands0, hands2 = _mapping(a.get("hands")), _mapping(b.get("hands"))
    feet0, feet2 = _mapping(a.get("feet")), _mapping(b.get("feet"))
    for side in ("l", "r"):
        h0, h2 = hands0.get(side), hands2.get(side)
        if h0 is None:
            if h2 is not None:
                gaps.append(f"gq_hand_reference_changed:{demo}:{model}:{side}")
        elif not isinstance(h0, dict):
            gaps.append(f"gq_hand_measurement_invalid:{demo}:{model}:{side}")
        elif h2 is None:
            failures.append(f"gq_lost_hand_measurement:{demo}:{model}:{side}")
        elif not isinstance(h2, dict):
            failures.append(f"gq_hand_measurement_invalid:{demo}:{model}:{side}")
        else:
            for field in ("gap_p95_m", "drift_max_m", "relative_speed_max_mps",
                          "relative_speed_p95_mps"):
                _compare_lower(failures, gaps, f"hands_{side}_{field}:{demo}:{model}",
                               h0.get(field), h2.get(field))
            for field in ("certified_frames", "certified_overlap_frames"):
                _compare_count(failures, gaps, f"hands_{side}_{field}:{demo}:{model}",
                               h0.get(field), h2.get(field))
        f0, f2 = feet0.get(side), feet2.get(side)
        if f0 is None:
            if f2 is not None:
                gaps.append(f"gq_foot_reference_changed:{demo}:{model}:{side}")
        elif not isinstance(f0, dict):
            gaps.append(f"gq_foot_measurement_invalid:{demo}:{model}:{side}")
        elif f2 is None:
            failures.append(f"gq_lost_foot_measurement:{demo}:{model}:{side}")
        elif not isinstance(f2, dict):
            failures.append(f"gq_foot_measurement_invalid:{demo}:{model}:{side}")
        else:
            for field in ("gap_p95_m", "drift_max_m", "speed_p95_mps"):
                _compare_lower(failures, gaps,
                               f"feet_{side}_{field}:{demo}:{model}",
                               f0.get(field), f2.get(field))
    both0, both2 = _mapping(a.get("both")), _mapping(b.get("both"))
    denominator0 = both0.get("denominator_frames")
    denominator2 = both2.get("denominator_frames")
    valid0 = (isinstance(denominator0, int) and not isinstance(denominator0, bool)
              and denominator0 >= 0)
    valid2 = (isinstance(denominator2, int) and not isinstance(denominator2, bool)
              and denominator2 >= 0)
    if not valid0 or not valid2:
        gaps.append(f"gq_both_denominator_invalid:{demo}:{model}")
    elif denominator0 != denominator2:
        gaps.append(f"gq_both_denominator_disagreement:{demo}:{model}")
    if valid0 and denominator0 > 0:
        for field in ("certified_overlap_frames", "longest_certified_overlap"):
            _compare_count(failures, gaps, f"both_{field}:{demo}:{model}",
                           both0.get(field), both2.get(field))
        _compare_lower(failures, gaps, f"both_valid_coverage:{demo}:{model}",
                       both0.get("valid_coverage"), both2.get("valid_coverage"))
    _compare_lower(failures, gaps, f"teacher_fit:{demo}:{model}",
                   _mapping(a.get("fitting")).get("teacher_fit_fk_mpjpe_box_m"),
                   _mapping(b.get("fitting")).get("teacher_fit_fk_mpjpe_box_m"))
    for side in ("l", "r"):
        _compare_lower(failures, gaps, f"hand_fk_{side}:{demo}:{model}",
                       _mapping(v0.get("hand_fk_mean_m")).get(side),
                       _mapping(v2.get("hand_fk_mean_m")).get(side))
    for field in ("world_mpjpe_p95_m", "world_mpjpe_max_m"):
        _compare_lower(failures, gaps, f"task_{field}:{demo}:{model}",
                       _mapping(a.get("task")).get(field),
                       _mapping(b.get("task")).get(field))
    _compare_lower(failures, gaps, f"root_angular_speed:{demo}:{model}",
                   v0.get("angular_velocity_max_radps"),
                   v2.get("angular_velocity_max_radps"))
    velocities = (strict2.get("checks") or {}).get("velocities", {})
    facts["mechanism_speed_gate_pass"] = bool(velocities.get("passed"))
    return {"demo": demo, "model": model, "passed": bool(not failures and not gaps),
            "failures": failures, "evidence_gaps": gaps, "facts": facts}


def _gq_rows(reg: dict, rows: list[dict], demos: list[str]) -> dict:
    by_key = {row.get("key"): row for row in rows}
    out = {}
    for demo in demos:
        for model in MODELS:
            out[f"{demo}:{model}"] = gq_compare(
                demo, model, by_key.get(f"model:{demo}:{model}:V0"),
                by_key.get(f"model:{demo}:{model}:V2"))
    failures = [f for v in out.values() for f in v["failures"]]
    gaps = [g for v in out.values() for g in v["evidence_gaps"]]
    return {"passed": bool(out and not failures and not gaps),
            "rows": out, "failures": failures, "evidence_gaps": gaps}


# --- val31 unit ----------------------------------------------------------------

def _original_row(demo: str, method: str, seed) -> dict:
    rows = read_json(SOURCE / "evaluation/raw/attempt_001/raw_eval.json")["rows"]
    hits = [r for r in rows if r.get("demo") == demo
            and r.get("method") == method and r.get("seed") == seed]
    if len(hits) != 1:
        raise ValueError(f"original row not unique:{demo}:{method}:{seed}")
    return hits[0]


def _load_product(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return np.array(z["root"]), np.array(z["joints"])


def val31_unit(run: Path, demo: str, out: Path, deadline: float) -> dict:
    import torch

    t0 = time.perf_counter()
    (reg, w1_reg, selection, att_reg, clip, validator, replayer,
     teacher_scene, teacher_kpts) = _load_fixed_context(run, demo)
    if demo not in reg["val31"]:
        raise ValueError("unregistered val31 demonstration")
    out.mkdir(parents=True, exist_ok=False)
    T = clip.T
    starts = clip_starts(T)
    task_tol = reg["task_gate_tol_m"]
    rows, session = [], {}

    def measure(root, joints, tag):
        return _measure_clip(validator, replayer, clip, teacher_scene, teacher_kpts,
                             root, joints, demo, task_tol, deadline, tag)

    teacher_metric = measure(clip.root, clip.q, "teacher")
    teacher_product = out / "teacher.npz"
    orig_teacher = _original_row(demo, "teacher", None)
    teacher_path = ROOT / orig_teacher["product"]["path"]
    old_root, old_joints = _load_product(teacher_path)
    teacher_binding = {
        "frozen_label_root_bitwise": bool(np.array_equal(clip.root, old_root)),
        "frozen_label_joints_bitwise": bool(np.array_equal(clip.q, old_joints)),
        "original_product": orig_teacher["product"],
    }
    _save_product(teacher_product, clip.root, clip.q, teacher_metric)
    rows.append(_row_from_measurement(
        f"teacher:{demo}", demo, "val31", "teacher", teacher_metric,
        teacher_product, binding=teacher_binding))
    if not all((teacher_binding["frozen_label_root_bitwise"],
                teacher_binding["frozen_label_joints_bitwise"])):
        raise ValueError("teacher label/product binding failed")
    for model in MODELS:
        _deadline(deadline, f"val31:{model}")
        device = w1_reg["configs"][model]["device"]
        cache = ROOT / w1_reg["cache"]["dir"]
        generator = raw.load_w1_de_generator(
            selection["models"][model], device, DDIM_STEPS, cache,
            w1_reg["training_manifest"]["sha256"], w1_reg["cache"]["identity_sha256"],
            w1_reg["cache"]["root_stats_sha256"],
            full_sha(SOURCE / "freeze/execution_registration.json"),
            registration=w1_reg, project_root=ROOT)
        trajectory, captured = attribution.capture_prediction(
            generator, clip.condition, SEED)
        decoded = captured["decoded"]
        windows_path = out / f"{model}_windows.npz"
        np.savez_compressed(
            windows_path,
            **{f"decoded_{k}": v for k, v in zip(("pos", "quat", "q"), decoded)},
            starts=np.asarray(starts, dtype=np.int64),
            normalized_prediction=np.asarray(captured["normalized_prediction"]))
        windows_rec = file_record(windows_path)
        v0 = _v0_fuse(decoded, starts, T)
        v0_root, v0_joints = attribution.world_state(
            *v0[:3], clip.condition.world_from_clip)
        condition_path = out / f"{model}_condition.npz"
        np.savez_compressed(condition_path, **captured["condition"])
        session[model] = {
            "condition": file_record(condition_path),
            "windows": windows_rec,
            "model_calls": trajectory.metadata.get("model_calls"),
            "resample_attempts_used": trajectory.metadata.get("resample_attempts_used"),
            "attempt_seeds": trajectory.metadata.get("attempt_seeds"),
            "attempts": trajectory.metadata.get("attempts"),
            "numerical_valid": trajectory.metadata.get("numerical_valid"),
            "final_invalid_windows": trajectory.metadata.get("final_invalid_windows"),
            "sampling_wall_s": trajectory.metadata.get("sampling_wall_s"),
            "generation_wall_s": trajectory.metadata.get("generation_wall_s"),
        }
        original = _original_row(demo, model, SEED)
        old_root, old_joints = _load_product(ROOT / original["product"]["path"])
        binding = {
            "decoded_window": windows_rec,
            "original_product": original["product"],
            "v0_recomputed_root_bitwise": bool(np.array_equal(trajectory.root, v0_root)),
            "v0_recomputed_joints_bitwise": bool(np.array_equal(trajectory.q, v0_joints)),
            "original_root_bitwise": bool(np.array_equal(trajectory.root, old_root)),
            "original_joints_bitwise": bool(np.array_equal(trajectory.q, old_joints)),
            "condition": file_record(condition_path),
        }
        if not all((binding["v0_recomputed_root_bitwise"],
                    binding["v0_recomputed_joints_bitwise"],
                    binding["original_root_bitwise"],
                    binding["original_joints_bitwise"])):
            raise ValueError(f"V0 binding failed:{demo}:{model}")
        v0_metric = measure(v0_root, v0_joints, f"{model}:V0")
        v0_product = out / f"{model}_V0.npz"
        _save_product(v0_product, v0_root, v0_joints, v0_metric)
        rows.append(_row_from_measurement(
            f"model:{demo}:{model}:V0", demo, "val31", f"{model}:V0",
            v0_metric, v0_product, binding=dict(binding)))
        try:
            v2 = fuse_windows_v2(decoded[0], decoded[1], decoded[2], starts, T,
                                 STRIDE, clip.condition.frame_dt)
        except RotationFusionRejected as exc:
            path_npz = out / f"{model}_V2_path.npz"
            path_evidence = _save_fusion_path(
                path_npz, decoded, starts, None, rejection=exc,
                frame_dt=clip.condition.frame_dt)
            binding.update(v2_status="algorithm_rejected",
                           v2_rejection=path_evidence["rejection"])
            rows.append(_row_from_measurement(
                f"model:{demo}:{model}:V2", demo, "val31", f"{model}:V2",
                {"status": "algorithm_rejected",
                 "measurement_gaps": [f"algorithm_rejected:{exc.code}"], "metrics": None},
                None, path_evidence=path_evidence, binding=dict(binding)))
        else:
            v2_root, v2_joints = attribution.world_state(
                v2.pos, v2.quat, v2.joints, clip.condition.world_from_clip)
            binding.update(
                v2_status="measured",
                v2_position_joint_bitwise=bool(np.array_equal(v2.pos, v0[0])
                                               and np.array_equal(v2.joints, v0[2])),
                v2_world_position_bitwise=bool(np.array_equal(
                    v2_root[:, :3], v0_root[:, :3])),
                fusion_temporal_pass=v2.fusion_temporal_pass,
                consumer_eligible=v2.consumer_eligible,
                blend_norm_min=float(np.min(v2.diagnostics["blend_norm"])),
            )
            path_npz = out / f"{model}_V2_path.npz"
            path_evidence = _save_fusion_path(
                path_npz, decoded, starts, v2,
                frame_dt=clip.condition.frame_dt)
            v2_metric = measure(v2_root, v2_joints, f"{model}:V2")
            v2_product = out / f"{model}_V2.npz"
            _save_product(v2_product, v2_root, v2_joints, v2_metric)
            rows.append(_row_from_measurement(
                f"model:{demo}:{model}:V2", demo, "val31", f"{model}:V2",
                v2_metric, v2_product, path_evidence=path_evidence,
                binding=dict(binding)))
        del generator
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    required = [f"teacher:{demo}"] + [
        f"model:{demo}:{m}:{v}" for m in MODELS for v in ("V0", "V2")]
    result = {"demo": demo, "T": T, "starts": starts, "rows": rows,
              "session": session, "status": "ok" if [r["key"] for r in rows] == required else "incomplete",
              "wall_s": time.perf_counter() - t0,
              "model_calls": sum(int(v.get("model_calls") or 0) for v in session.values())}
    write_json(out / "result.json", result)
    if result["status"] != "ok":
        raise ValueError("val31 scene row matrix incomplete")
    return result


def val31_stage(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    cohort = _cohort_history(run, reg["cohort_id"])
    budget = budget_for(run, "val31", cohort)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    attempt = run / "val31/attempt_001"
    if attempt.exists():
        raise SystemExit("val31 attempt already exists; no retry")
    attempt.mkdir(parents=True, exist_ok=True)
    _preregister_attempt(run, "val31", "attempt_001", t0,
                         cohort_id=reg["cohort_id"])
    summary = {"schema": "w1_rotation_fusion_val31_v1",
               "execution_registration": file_record(run / "freeze/execution_registration.json"),
               "protocol": PROTOCOL, "status": "running", "complete": False,
               "rows": [], "units": [], "evidence_gaps": []}
    write_json(attempt / "summary.json", summary, exclusive=True)
    fixed_row, fixed_problems = stage_state(run, "fixed-windows")
    fixed_summary = read_json(run / "fixed_windows/attempt_001/summary.json") \
        if (run / "fixed_windows/attempt_001/summary.json").is_file() else {}
    gs_passed = bool(fixed_summary.get("gs", {}).get("passed"))
    if fixed_row is None or fixed_problems or fixed_row.get("complete") is not True \
            or not gs_passed:
        summary.update(status="not_run_due_to_contract_gate", complete=True,
                       runs=0, fixed_window_problems=fixed_problems,
                       fixed_window_gs=fixed_summary.get("gs"),
                       reason="fixed-window GS not passed")
        write_json(attempt / "summary.json", summary)
        write_json(attempt / "rows.json", [], exclusive=True)
        write_json(attempt / "units.json", [], exclusive=True)
        row = {"complete": True, "status": "done",
               "summary": record_path(attempt / "summary.json"),
               "artifacts": {record_path(attempt / "summary.json"):
                             full_sha(attempt / "summary.json")}}
        return _finish_stage(run, "val31", "attempt_001", t0, budget, row,
                             {"not_run_due_to_contract_gate": True}, "done")
    out_rows, units = [], []
    failures = []
    for demo in reg["val31"]:
        _deadline(deadline, "val31:next-unit")
        unit_t0 = time.perf_counter()
        timeout = min(600.0, deadline - unit_t0)
        if timeout <= 1.0:
            failures.append(f"budget_exhausted_before_unit:{demo}")
            break
        unit_out = attempt / "demos" / demo
        argv = [sys.executable, str(Path(__file__)), "val31-unit",
                "--run-root", str(run), "--demo", demo, "--out", str(unit_out),
                "--deadline", str(time.perf_counter() + timeout - 2.0)]
        env = {**os.environ, **{k: "1" for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")}}
        unit = {"demo": demo, "argv": argv, "status": "running", "wall_s": 0.0}
        units.append(unit)
        write_json(attempt / "units.json", units)
        try:
            with (attempt / f"{demo}.log").open("w") as log:
                proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=timeout)
            unit.update(returncode=proc.returncode,
                        status="done" if proc.returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            unit.update(returncode=None, status="timeout")
        finally:
            unit["wall_s"] = time.perf_counter() - unit_t0
            write_json(attempt / "units.json", units)
        result_path = unit_out / "result.json"
        if result_path.is_file():
            scene = read_json(result_path)
            out_rows.extend(scene.get("rows", []))
            unit["model_calls"] = scene.get("model_calls")
            summary["rows"] = out_rows
            write_json(attempt / "rows.json", out_rows)
        if unit["status"] != "done":
            failures.append(f"unit_failed:{demo}:{unit['status']}")
            break
    expected = reg["expected_keys"]["val31"]
    out_rows = _ordered_rows(out_rows, expected)
    keys = [row.get("key") for row in out_rows]
    expected_keys = set(expected)
    write_json(attempt / "rows.json", out_rows)
    summary.update(rows=out_rows, units=units)
    summary["counts"] = {
        "expected": len(expected), "rows": len(out_rows), "keys_unique": len(set(keys)),
        "teacher": sum(k.startswith("teacher:") for k in keys),
        "model": sum(k.startswith("model:") for k in keys),
        "measured": sum(r.get("status") == "measured" for r in out_rows),
        "algorithm_rejected": sum(r.get("status") == "algorithm_rejected" for r in out_rows),
        "model_calls": sum(int(u.get("model_calls") or 0) for u in units),
    }
    if keys != expected:
        failures.append("val31_key_matrix_mismatch")
    failures += _teacher_status_failures(out_rows, reg["val31"])
    for row in out_rows:
        if row.get("status") == "algorithm_rejected" \
                and not str(row.get("key", "")).endswith(":V2"):
            failures.append(
                f"non_v2_algorithm_rejection:{row.get('key')}")
    for row in out_rows:
        if row.get("status") == "measured":
            summary_gaps = _metric_gaps(row.get("metrics"))
            if row.get("metrics", {}).get("measurement_gaps") or summary_gaps:
                failures.append(f"val31_row_measurement_incomplete:{row.get('key')}")
    summary["failures"] = failures
    summary["evidence_gaps"] = list(failures)
    summary["matrix"] = _val31_matrix(reg, out_rows)
    summary["gq"] = _gq_rows(reg, out_rows, reg["val31"])
    summary["algorithm_rejections"] = [
        {"key": row["key"], "rejection": (row.get("path_evidence") or {}).get("rejection")}
        for row in out_rows if row.get("status") == "algorithm_rejected"]
    evidence_complete = bool(not failures and not summary["evidence_gaps"]
                             and set(keys) == expected_keys and len(keys) == len(expected))
    summary["complete"] = evidence_complete
    summary["status"] = "done" if evidence_complete else "incomplete"
    summary["cost_s"] = time.perf_counter() - t0
    write_json(attempt / "summary.json", summary)
    row = {"complete": evidence_complete,
           "status": "done" if evidence_complete else "incomplete",
           "summary": record_path(attempt / "summary.json"),
           "artifacts": {record_path(attempt / "summary.json"):
                         full_sha(attempt / "summary.json"),
                         record_path(attempt / "rows.json"):
                         full_sha(attempt / "rows.json")}}
    ok = _finish_stage(run, "val31", "attempt_001", t0, budget, row,
                       {"rows": len(out_rows),
                        "model_calls": summary["counts"]["model_calls"]},
                       "done" if evidence_complete else "incomplete")
    print(json.dumps({"val31": summary["status"], "rows": len(out_rows),
                      "model_calls": summary["counts"]["model_calls"],
                      "gq": summary["gq"]["passed"]}, ensure_ascii=False), flush=True)
    return ok and evidence_complete


def _val31_matrix(reg: dict, rows: list[dict]) -> dict:
    by_key = {row.get("key"): row for row in rows}
    methods = {
        "Teacher raw": [f"teacher:{d}" for d in reg["val31"]],
        "Deterministic V0": [f"model:{d}:deterministic:V0" for d in reg["val31"]],
        "Diffusion V0": [f"model:{d}:diffusion:V0" for d in reg["val31"]],
        "Deterministic V2": [f"model:{d}:deterministic:V2" for d in reg["val31"]],
        "Diffusion V2": [f"model:{d}:diffusion:V2" for d in reg["val31"]],
    }
    output = {}
    for name, keys in methods.items():
        selected = [by_key.get(k) for k in keys]
        measured = [r for r in selected if r is not None and r.get("status") == "measured"]
        output[name] = {
            "expected": len(keys), "rows": sum(r is not None for r in selected),
            "n_computable": len(measured),
            "algorithm_rejected": sum(r is not None and r.get("status") == "algorithm_rejected"
                                      for r in selected),
            "raw_standard_pass": sum(bool(r.get("metrics", {}).get("raw_standard_pass"))
                                     for r in measured),
            "raw_denominator": len(keys),
            "teacher_fit_fk_mean_m_all": (
                float(np.mean([r["metrics"]["fitting"]["teacher_fit_fk_mpjpe_box_m"]
                               for r in measured]))
                if len(measured) == len(keys) else None),
            "teacher_fit_fk_mean_m_computed_only": (
                float(np.mean([r["metrics"]["fitting"]["teacher_fit_fk_mpjpe_box_m"]
                               for r in measured])) if measured else None),
            "penetration_max_m_all": (
                max(r["metrics"]["penetration"]["max_m"] for r in measured)
                if len(measured) == len(keys) else None),
            "task_gate_pass": sum(bool(r["metrics"].get("task_gate_pass")) for r in measured),
        }
    return output


# --- finalize ------------------------------------------------------------------

def _verify_rows_products(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        if row.get("status") != "measured":
            continue
        rec = row.get("product") or {}
        if not rec.get("path") or not (ROOT / rec["path"]).is_file() \
                or full_sha(ROOT / rec["path"]) != rec.get("sha256"):
            problems.append(f"product_missing_or_drifted:{row.get('key')}")
        evidence = (row.get("path_evidence") or {}).get("path")
        if evidence:
            if not (ROOT / evidence.get("path", "")).is_file() \
                    or full_sha(ROOT / evidence["path"]) != evidence.get("sha256"):
                problems.append(f"path_evidence_missing_or_drifted:{row.get('key')}")
    return problems


def _integration_ready(gs_passed: bool, fixed_gq_passed: bool,
                       val_gq_passed: bool, val_status,
                       experiment_complete: bool) -> bool:
    """The candidate is integration-eligible only after full, no-regression V."""
    return bool(gs_passed and fixed_gq_passed and val_gq_passed
                and val_status == "done" and experiment_complete)


def _load_json_or(path: Path, label: str, failures: list, default):
    try:
        return read_json(path)
    except (OSError, ValueError, TypeError):
        failures.append(f"{label}_unreadable")
        return default


def _finalize(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    cohort = _cohort_history(run, reg["cohort_id"])
    budget = budget_for(run, "finalize", cohort)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    attempt = run / "decisions/attempt_001"
    if attempt.exists():
        raise SystemExit("finalize attempt already exists; no retry")
    attempt.mkdir(parents=True, exist_ok=True)
    _preregister_attempt(run, "finalize", "attempt_001", t0,
                         cohort_id=reg["cohort_id"])
    failures, gaps = [], []
    register_row, register_problems = stage_state(run, "register")
    fixed_row, fixed_problems = stage_state(run, "fixed-windows")
    val_row, val_problems = stage_state(run, "val31")
    if register_row is None or register_problems or register_row.get("complete") is not True:
        failures.append("register_stage_incomplete")
    if fixed_row is None or fixed_problems:
        gaps += [f"fixed_windows_state:{p}" for p in fixed_problems]
    if fixed_row is not None and fixed_row.get("complete") is not True:
        failures.append("fixed_windows_stage_incomplete")
    if val_row is None or val_problems:
        gaps += [f"val31_state:{p}" for p in val_problems]
    if val_row is not None and val_row.get("complete") is not True:
        failures.append("val31_stage_incomplete")
    fixed_summary = _load_json_or(run / "fixed_windows/attempt_001/summary.json",
                                  "fixed_summary", failures, {})
    val_summary = _load_json_or(run / "val31/attempt_001/summary.json",
                                "val31_summary", failures, {})
    val_not_run = val_summary.get("status") == "not_run_due_to_contract_gate"

    fixed_section = _scene_section(run, "fixed-windows", reg)
    failures += [f"fixed:{x}" for x in fixed_section["failures"]]
    fixed_rows = fixed_section["rows"]
    fixed_rows_file = _load_json_or(run / "fixed_windows/attempt_001/rows.json",
                                    "fixed_rows_json", failures, None)
    failures += _rows_copy_mismatch("fixed_rows_json", fixed_rows_file, fixed_rows)
    failures += _rows_copy_mismatch("fixed_summary_rows",
                                    fixed_summary.get("rows"), fixed_rows)

    if val_not_run:
        val_section = {"rows": [], "meta": {},
                       "root_dir": run / "val31/attempt_001/demos"}
        val_rows = []
    else:
        val_section = _scene_section(run, "val31", reg)
        failures += [f"val31:{x}" for x in val_section["failures"]]
        val_rows = val_section["rows"]
        val_rows_file = _load_json_or(run / "val31/attempt_001/rows.json",
                                      "val31_rows_json", failures, None)
        failures += _rows_copy_mismatch("val31_rows_json", val_rows_file, val_rows)
        failures += _rows_copy_mismatch("val31_summary_rows",
                                        val_summary.get("rows"), val_rows)

    w1_reg, selection, att_reg, identity = verify_source_registration(reg)
    source = (w1_reg, selection, att_reg, identity)
    contexts = {}

    def context_for_demo(demo: str):
        if demo not in contexts:
            contexts[demo] = _load_fixed_context(run, demo, reg=reg, source=source)
        c = contexts[demo]
        return (c[5], c[6], c[4], c[7], c[8])

    def clip_for_demo(demo: str):
        return context_for_demo(demo)[2]

    product_problems = _verify_rows_products(fixed_rows) + _verify_rows_products(val_rows)
    failures += [f"product:{x}" for x in product_problems]

    frozen_replays = {"fixed_windows": _verify_frozen_replays(
        reg, "fixed-windows", run, fixed_rows, fixed_section["meta"],
        deadline, clip_for_demo)}
    if not val_not_run:
        frozen_replays["val31"] = _verify_frozen_replays(
            reg, "val31", run, val_rows, val_section["meta"],
            deadline, clip_for_demo)
    for section, report in frozen_replays.items():
        failures += [f"frozen_replay:{section}:{x}" for x in report["failures"]]
        gaps += [f"frozen_replay:{section}:{x}" for x in report["evidence_gaps"]]
        if report["checked"] != report["expected"]:
            failures.append(
                f"frozen_replay:{section}:checked_{report['checked']}_of_"
                f"{report['expected']}")

    teacher_labels = {"fixed_windows": _verify_teacher_labels(
        reg, "fixed-windows", fixed_rows, deadline, clip_for_demo)}
    if not val_not_run:
        teacher_labels["val31"] = _verify_teacher_labels(
            reg, "val31", val_rows, deadline, clip_for_demo)
    for section, report in teacher_labels.items():
        failures += [f"teacher_label:{section}:{x}" for x in report["failures"]]
        gaps += [f"teacher_label:{section}:{x}" for x in report["evidence_gaps"]]
        if report["checked"] != report["expected"]:
            failures.append(
                f"teacher_label:{section}:checked_{report['checked']}_of_"
                f"{report['expected']}")

    v2_recomputed = {"fixed_windows": _recompute_samples(
        run, "fixed-windows", reg, fixed_rows, fixed_section["meta"],
        deadline, clip_for_demo)}
    if not val_not_run:
        v2_recomputed["val31"] = _recompute_samples(
            run, "val31", reg, val_rows, val_section["meta"],
            deadline, clip_for_demo)
    for section, report in v2_recomputed.items():
        failures += [f"recompute:{section}:{x}" for x in report["failures"]]
        gaps += [f"recompute:{section}:{x}" for x in report["evidence_gaps"]]
        if report["checked"] != report["expected"]:
            failures.append(
                f"recompute:{section}:checked_{report['checked']}_of_"
                f"{report['expected']}")

    remeasure = {"fixed_windows": _remeasure_rows(
        reg, fixed_rows, deadline, context_for_demo,
        allowed_rejected=set(
            v2_recomputed["fixed_windows"].get("rejected_keys") or ()))}
    if not val_not_run:
        remeasure["val31"] = _remeasure_rows(
            reg, val_rows, deadline, context_for_demo,
            allowed_rejected=set(
                v2_recomputed["val31"].get("rejected_keys") or ()))
    for section, report in remeasure.items():
        failures += [f"remeasure:{section}:{x}" for x in report["failures"]]
        gaps += [f"remeasure:{section}:{x}" for x in report["evidence_gaps"]]
        if report["measured"] != report["expected_measured"]:
            failures.append(
                f"remeasure:{section}:measured_{report['measured']}_of_"
                f"{report['expected_measured']}")
    verified_fixed_rows = remeasure["fixed_windows"]["rows"]
    verified_val_rows = remeasure["val31"]["rows"] if not val_not_run else []

    fixed_gq = _gq_rows(reg, verified_fixed_rows, reg["demos"])
    if fixed_gq != fixed_summary.get("gq"):
        gaps.append("fixed_gq_not_reproducible")
    known_dt = None
    if KNOWN_CASE in (reg.get("demos") or []):
        known_dt = clip_for_demo(KNOWN_CASE).condition.frame_dt
    known_case = _known_case_facts(
        [r for r in verified_fixed_rows
         if str(r.get("key", "")).startswith(f"model:{KNOWN_CASE}:")],
        fixed_section["root_dir"] / KNOWN_CASE,
        frame_dt=known_dt)
    try:
        synthetic_tests = _run_synthetic_tests(deadline)
    except subprocess.TimeoutExpired:
        synthetic_tests = {"command": "synthetic math tests",
                           "returncode": None, "stdout_tail": "",
                           "wall_s": 0.0}
    if synthetic_tests.get("returncode") != 0:
        failures.append("finalize:synthetic_math_tests_failed")
    recomputed_gs = _fixed_gs(
        reg, verified_fixed_rows,
        {KNOWN_CASE: {"known_case": known_case}},
        synthetic_tests.get("returncode") == 0)
    if recomputed_gs != fixed_summary.get("gs"):
        gaps.append("fixed_gs_not_reproducible")
    if val_not_run:
        # The skip is legal only while the independently recomputed GS fails.
        if recomputed_gs.get("passed"):
            failures.append("val31_skipped_despite_gs_pass")
        if attribution.safe_json(val_summary.get("fixed_window_gs")) \
                != attribution.safe_json(recomputed_gs):
            gaps.append("val31_skip_gate_not_reproducible")
        if val_summary.get("runs") != 0 or val_summary.get("units"):
            failures.append("val31_skip_not_empty")

    if val_not_run:
        val_gq = {"passed": False, "rows": {}, "failures": ["gq_missing"],
                  "evidence_gaps": []}
    else:
        val_gq = _gq_rows(reg, verified_val_rows, reg["val31"])
        if val_gq != val_summary.get("gq"):
            gaps.append("val_gq_not_reproducible")

    # Final closure re-checks every consumption anchor AFTER all substantive
    # recomputation: no artifact may drift while F is running.
    drift = _hash_map(reg["anchors"])
    if drift:
        gaps += [f"anchor_drift_at_closure:{x}" for x in drift]
    experiment_complete = _experiment_complete(fixed_row, val_row, failures, gaps)
    integration = _integration_ready(
        bool(recomputed_gs.get("passed")), fixed_gq["passed"],
        val_gq.get("passed"), val_summary.get("status"), experiment_complete)
    blocking = list(dict.fromkeys(
        list(failures)
        + list(recomputed_gs.get("failures") or [])
        + list(fixed_gq.get("failures") or [])
        + list(val_gq.get("failures") or [])))
    remeasure_record = {
        section: {k: v for k, v in report.items() if k != "rows"}
        for section, report in remeasure.items()}
    decisions = {
        "schema": "w1_rotation_fusion_final_v1",
        "execution_registration": file_record(run / "freeze/execution_registration.json"),
        "protocol": PROTOCOL,
        "implementation_contract_passed": bool(recomputed_gs.get("passed")),
        "experiment_complete": experiment_complete,
        "integration_eligible": bool(integration and experiment_complete),
        "final_interaction_method": "not_ready",
        "blocking_findings": blocking,
        "fixed_windows": {"status": fixed_summary.get("status"),
                          "gs": recomputed_gs,
                          "gq": fixed_gq,
                          "counts": fixed_summary.get("counts")},
        "val31": {"status": val_summary.get("status"),
                  "not_run_due_to_contract_gate": val_not_run,
                  "matrix": val_summary.get("matrix"),
                  "gq": val_gq,
                  "counts": val_summary.get("counts")},
        "recompute": v2_recomputed, "remeasure": remeasure_record,
        "frozen_replays": frozen_replays,
        "teacher_labels": teacher_labels,
        "verification": {
            "fixed_scene_keys": len(fixed_rows),
            "val31_scene_keys": len(val_rows),
            "frozen_replays": {
                section: {k: v for k, v in report.items()
                          if k not in ("failures", "evidence_gaps")}
                for section, report in frozen_replays.items()},
            "teacher_labels": {
                section: {k: v for k, v in report.items()
                          if k not in ("failures", "evidence_gaps")}
                for section, report in teacher_labels.items()},
            "recompute": {
                section: {k: v for k, v in report.items()
                          if k not in ("failures", "evidence_gaps")}
                for section, report in v2_recomputed.items()},
            "remeasure": {
                section: {k: v for k, v in report.items()
                          if k not in ("rows", "failures", "evidence_gaps")}
                for section, report in remeasure.items()},
            "synthetic_tests": synthetic_tests,
            "known_case": known_case,
        },
        "failures": failures, "evidence_gaps": gaps,
        "training_steps": 0,
        "historical_anchors": {"u0": "incomplete", "ur": "no_quality_gain"},
        "next_action": {"auto_execute": False,
                        "recommendation": ("explicit versioned entry integration"
                                           if integration and experiment_complete
                                           else "keep candidate out of production; "
                                                "record local mechanism findings")},
    }
    write_json(attempt / "fusion_final.json", decisions)
    row = {"complete": experiment_complete,
           "status": "done" if experiment_complete else "incomplete",
           "summary": record_path(attempt / "fusion_final.json"),
           "artifacts": {record_path(attempt / "fusion_final.json"):
                         full_sha(attempt / "fusion_final.json")}}
    ok = _finish_stage(run, "finalize", "attempt_001", t0, budget, row,
                       {"integration_eligible": decisions["integration_eligible"],
                        "experiment_complete": experiment_complete},
                       "done" if experiment_complete else "incomplete")
    print(json.dumps({"final": record_path(attempt / "fusion_final.json"),
                      "complete": experiment_complete,
                      "integration_eligible": decisions["integration_eligible"]},
                     ensure_ascii=False), flush=True)
    return ok and experiment_complete


def _scene_section(run: Path, stage: str, reg: dict) -> dict:
    """Rebuild one sealed section from scene results, not from its summaries."""
    if stage == "fixed-windows":
        root_dir = run / "fixed_windows/attempt_001/scenes"
        demos = reg["demos"]
        expected = reg["expected_keys"]["fixed_windows"]
    else:
        root_dir = run / "val31/attempt_001/demos"
        demos = reg["val31"]
        expected = reg["expected_keys"]["val31"]
    rows, meta, failures = [], {}, []
    for demo in demos:
        result_path = root_dir / demo / "result.json"
        if not result_path.is_file():
            failures.append(f"scene_result_missing:{demo}")
            continue
        try:
            result = read_json(result_path)
        except (OSError, ValueError, TypeError):
            failures.append(f"scene_result_unreadable:{demo}")
            continue
        if not isinstance(result, dict):
            failures.append(f"scene_result_not_a_mapping:{demo}")
            continue
        if result.get("status") != "ok":
            failures.append(f"scene_not_ok:{demo}:{result.get('status')}")
            continue
        starts = result.get("starts")
        if not isinstance(starts, list) or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in starts):
            failures.append(f"scene_starts_invalid:{demo}")
            continue
        T = result.get("T")
        if not isinstance(T, int) or isinstance(T, bool) or T < 1:
            failures.append(f"scene_T_invalid:{demo}")
            continue
        scene_rows = result.get("rows")
        if (not isinstance(scene_rows, list)
                or not all(isinstance(r, dict) for r in scene_rows)):
            failures.append(f"scene_rows_invalid:{demo}")
            continue
        meta[demo] = result
        rows.extend(scene_rows)
    rows = _ordered_rows(rows, expected)
    keys = [r.get("key") for r in rows]
    if len(keys) != len(set(keys)):
        failures.append(f"{stage}_scene_keys_duplicated")
    if keys != expected:
        failures.append(f"{stage}_scene_key_matrix_mismatch")
    return {"rows": rows, "meta": meta, "failures": failures,
            "expected": expected, "root_dir": root_dir}


def _rows_copy_mismatch(label: str, rows, scene_rows: list[dict]) -> list[str]:
    if not isinstance(rows, list):
        return [f"{label}_rows_missing"]
    if attribution.safe_json(rows) != attribution.safe_json(scene_rows):
        return [f"{label}_rows_not_reproducible"]
    return []


def _metric_mismatches(row: dict, metric: dict) -> list[str]:
    names = []
    if (attribution.safe_json(row.get("metrics"))
            != attribution.safe_json(metric.get("metrics"))):
        names.append("metrics")
    for field in ("status", "measurement_gaps", "root_position_max_error_m",
                  "root_rotation_max_error_deg", "joint_max_error_rad",
                  "hand_fk_mean_m", "angular_velocity_max_radps",
                  "seam_rotation_max_deg", "keyframe_penetration_count"):
        if (attribution.safe_json(row.get(field))
                != attribution.safe_json(metric.get(field))):
            names.append(field)
    return names


def _run_synthetic_tests(deadline: float) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_rotation_fusion.py",
         "tests/test_w1_rotation_fusion.py", "-q"],
        cwd=ROOT, capture_output=True, text=True,
        timeout=max(1.0, min(300.0, deadline - started)))
    return {
        "command": "pytest tests/test_rotation_fusion.py tests/test_w1_rotation_fusion.py -q",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "wall_s": time.perf_counter() - started,
    }


def _finite_float_array(value, shape) -> bool:
    return (isinstance(value, np.ndarray) and value.shape == shape
            and np.issubdtype(value.dtype, np.floating)
            and bool(np.isfinite(value).all()))


def _path_array_problems(pos, quat, joints, T: int) -> list[str]:
    problems = []
    if not _finite_float_array(pos, (T, 3)):
        problems.append("fused_pos_clip")
    if not _finite_float_array(quat, (T, 4)):
        problems.append("fused_quat_clip")
    else:
        try:
            norms = np.linalg.norm(quat, axis=1)
            if not np.isfinite(norms).all() or np.any(norms < 1e-12) \
                    or np.any(np.abs(norms - 1.0) > QUAT_NORM_TOL):
                problems.append("fused_quat_clip")
        except (TypeError, ValueError, FloatingPointError):
            problems.append("fused_quat_clip")
    if not _finite_float_array(joints, (T, 29)):
        problems.append("fused_joints_clip")
    return problems


def _recompute_samples(run: Path, stage: str, reg: dict, rows: list[dict],
                       meta: dict, deadline: float, clip_for_demo,
                       attribution_for_demo=None) -> dict:
    """Rebuild every V2 path from decoded windows and bind it to its product."""
    source_for = attribution_for_demo or _attribution_result
    section = "fixed_windows" if stage == "fixed-windows" else "val31"
    expected_keys = [k for k in reg["expected_keys"][section] if k.endswith(":V2")]
    root_dir = run / ("fixed_windows/attempt_001/scenes"
                      if stage == "fixed-windows" else "val31/attempt_001/demos")
    failures, gaps, checked, measured, rejected = [], [], 0, 0, 0
    rejected_keys = []
    by_key = {row.get("key"): row for row in rows}
    if len(by_key) != len(rows):
        failures.append(f"{stage}_scene_keys_duplicated")
    for key in expected_keys:
        row = by_key.get(key)
        if row is None:
            failures.append(f"recompute_row_missing:{key}")
            continue
        demo = key.split(":")[1]
        result = meta.get(demo)
        if result is None:
            failures.append(f"recompute_scene_missing:{demo}")
            continue
        _deadline(deadline, f"recompute:{key}")
        evidence = _mapping(row.get("path_evidence"))
        path_rec = _mapping(evidence.get("path"))
        path = ROOT / path_rec.get("path", "")
        if not path_rec.get("path") or not path.is_file() \
                or full_sha(path) != path_rec.get("sha256"):
            failures.append(f"path_missing_or_drifted:{key}")
            continue
        scene_dir = root_dir / demo
        source_windows = None
        if stage == "fixed-windows" and key.startswith("model:"):
            model = key.split(":")[2]
            try:
                att = source_for(demo)
            except (OSError, ValueError, TypeError):
                failures.append(f"decoded_window_source_unreadable:{key}")
                continue
            source_windows = ((att.get("models") or {}).get(model) or {}).get("windows")
            if not isinstance(source_windows, dict) \
                    or _registered_artifact_path(reg, source_windows) is None:
                failures.append(f"decoded_window_not_registered:{key}")
                continue
        try:
            decoded, starts = _decoded_from_windows_for_row(
                scene_dir, row, result, source_windows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"decoded_windows_unreadable:{key}:{type(exc).__name__}")
            continue
        expected_starts = [int(v) for v in result.get("starts", [])]
        if starts != expected_starts:
            failures.append(f"recompute_starts_differ:{key}")
            continue
        clip = clip_for_demo(demo)
        try:
            with np.load(path, allow_pickle=False) as z:
                originals = {
                    name: np.array(z[name])
                    for name in ("original_pos", "original_quat", "original_q",
                                 "starts") if name in z.files}
        except (OSError, ValueError) as exc:
            failures.append(f"path_unreadable:{key}:{type(exc).__name__}")
            continue
        for name, value in (("original_pos", decoded[0]),
                            ("original_quat", decoded[1]),
                            ("original_q", decoded[2]),
                            ("starts", np.asarray(starts, dtype=np.int64))):
            if name not in originals or not np.array_equal(originals[name], value):
                failures.append(f"path_inputs_drifted:{key}:{name}")
        try:
            recomputed = fuse_windows_v2(
                decoded[0], decoded[1], decoded[2], starts, result["T"],
                STRIDE, clip.condition.frame_dt)
        except RotationFusionRejected as exc:
            saved = evidence.get("rejection") or {}
            prefix_length = (len(exc.prefix["pos"]) if exc.prefix else None)
            if row.get("status") != "algorithm_rejected":
                failures.append(f"unexpected_rejection:{key}")
            for field, want in (("code", exc.code), ("frame", exc.frame),
                                ("window", exc.window),
                                ("local_frame", exc.local_frame),
                                ("prefix_length", prefix_length)):
                if saved.get(field) != want:
                    failures.append(f"rejection_not_reproducible:{key}:{field}")
            if row.get("product"):
                failures.append(f"rejected_row_has_product:{key}")
            if exc.prefix:
                try:
                    with np.load(path, allow_pickle=False) as z:
                        for name, value in (("prefix_pos", exc.prefix["pos"]),
                                            ("prefix_quat", exc.prefix["quat"]),
                                            ("prefix_joints", exc.prefix["joints"])):
                            if name not in z.files \
                                    or not np.array_equal(z[name], value):
                                failures.append(f"rejection_prefix:{key}:{name}")
                except (OSError, ValueError):
                    failures.append(f"rejection_prefix_unreadable:{key}")
            rejected_keys.append(key)
            checked += 1
            rejected += 1
            continue
        if row.get("status") != "measured":
            failures.append(f"unexpected_recovery:{key}")
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                required = ("fused_quat_clip", "fused_pos_clip", "fused_joints_clip",
                            "lifted_quat", "weights", "blend_norm", "window_signs")
                missing = [k for k in required if k not in z.files]
                if missing:
                    gaps.append(f"path_records_missing:{key}:{missing}")
                    continue
                saved_quat = np.array(z["fused_quat_clip"])
                saved_pos = np.array(z["fused_pos_clip"])
                saved_joints = np.array(z["fused_joints_clip"])
                saved_lifted = np.array(z["lifted_quat"])
                saved_weights = np.array(z["weights"])
                saved_blend = np.array(z["blend_norm"])
                saved_signs = np.array(z["window_signs"])
        except (OSError, ValueError):
            failures.append(f"path_unreadable:{key}")
            continue
        saved_problems = _path_array_problems(
            saved_pos, saved_quat, saved_joints, result["T"])
        recomputed_problems = _path_array_problems(
            recomputed.pos, recomputed.quat, recomputed.joints, result["T"])
        failures += [f"path_array_invalid:{key}:{name}"
                     for name in saved_problems]
        failures += [f"recomputed_array_invalid:{key}:{name}"
                     for name in recomputed_problems]
        if evidence.get("fusion_temporal_pass") != recomputed.fusion_temporal_pass:
            failures.append(f"fusion_temporal_drifted:{key}")
        if evidence.get("consumer_eligible") != recomputed.consumer_eligible:
            failures.append(f"consumer_eligibility_drifted:{key}")
        if not np.array_equal(recomputed.pos, saved_pos) \
                or not np.array_equal(recomputed.joints, saved_joints):
            failures.append(f"recomputed_positions_or_joints:{key}")
        if not np.array_equal(recomputed.lifted_quat, saved_lifted):
            failures.append(f"recomputed_lift:{key}")
        if not np.array_equal(recomputed.weights, saved_weights):
            failures.append(f"recomputed_weights:{key}")
        if not np.array_equal(np.asarray(recomputed.diagnostics["blend_norm"]),
                              saved_blend):
            failures.append(f"recomputed_blend_norm:{key}")
        if not np.array_equal(np.asarray(recomputed.diagnostics["window_signs"]),
                              saved_signs):
            failures.append(f"recomputed_window_signs:{key}")
        if "fused_quat_clip" not in saved_problems \
                and "fused_quat_clip" not in recomputed_problems:
            try:
                drift = closed_geodesic_deg(recomputed.quat, saved_quat)
                if not np.all(np.isfinite(drift)) \
                        or not np.all(drift <= 1e-9):
                    failures.append(f"recomputed_rotation:{key}")
            except (ValueError, TypeError, FloatingPointError) as exc:
                failures.append(
                    f"recomputed_rotation_unreadable:{key}:"
                    f"{type(exc).__name__}")
        product_rec = _mapping(row.get("product"))
        product_path = ROOT / product_rec.get("path", "")
        if not product_rec.get("path") or not product_path.is_file() \
                or full_sha(product_path) != product_rec.get("sha256"):
            failures.append(f"product_missing_or_drifted:{key}")
        else:
            try:
                with np.load(product_path, allow_pickle=False) as z:
                    needed = ("root", "joints", "world_from_clip", "target_box",
                              "window_starts")
                    missing = [k for k in needed if k not in z.files]
                    if missing:
                        failures.append(f"product_records_missing:{key}:{missing}")
                    else:
                        product_root = np.array(z["root"])
                        product_joints = np.array(z["joints"])
                        if not np.array_equal(np.array(z["world_from_clip"]),
                                              clip.condition.world_from_clip):
                            failures.append(f"product_world_frame_drifted:{key}")
                        if not np.array_equal(np.array(z["target_box"]),
                                              clip.condition.target_obj_pose):
                            failures.append(f"product_target_drifted:{key}")
                        if not np.array_equal(
                                np.array(z["window_starts"]),
                                np.asarray(starts, dtype=np.int64)):
                            failures.append(f"product_starts_drifted:{key}")
                        world_root, world_joints = attribution.world_state(
                            recomputed.pos, recomputed.quat, recomputed.joints,
                            clip.condition.world_from_clip)
                        if not (np.array_equal(world_root, product_root)
                                and np.array_equal(world_joints, product_joints)):
                            failures.append(
                                f"product_not_bound_to_recomputed_path:{key}")
            except (OSError, ValueError):
                failures.append(f"product_unreadable:{key}")
        checked += 1
        measured += 1
    return {"failures": failures, "evidence_gaps": gaps,
            "checked": checked, "expected": len(expected_keys),
            "measured": measured, "rejected": rejected,
            "rejected_keys": rejected_keys}


def _attribution_result(demo: str) -> dict:
    return read_json(ATTRIBUTION / "diagnosis/attempt_001" / demo / "result.json")


def _recorded_file_path(rec) -> Path | None:
    if not isinstance(rec, dict):
        return None
    raw = rec.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
    if not path.is_file() or full_sha(path) != rec.get("sha256"):
        return None
    return path


def _registered_artifact_path(reg: dict, rec) -> Path | None:
    """Resolve a record only when it is the registered, still-identical bytes."""
    path = _recorded_file_path(rec)
    if path is None:
        return None
    anchor = (reg.get("anchors") or {}).get(record_path(path))
    if anchor is None or rec.get("sha256") != anchor:
        return None
    return path


def _verify_teacher_labels(reg: dict, stage: str, rows: list[dict],
                           deadline: float, clip_for_demo) -> dict:
    """Bind every teacher product bit-for-bit to the frozen clip label."""
    section = "fixed_windows" if stage == "fixed-windows" else "val31"
    demos = reg["demos"] if stage == "fixed-windows" else reg["val31"]
    expected_keys = [f"teacher:{demo}" for demo in demos]
    by_key = {row.get("key"): row for row in rows}
    failures = ([f"{section}_scene_keys_duplicated"]
                if len(by_key) != len(rows) else [])
    gaps, checked = [], 0
    for key in expected_keys:
        row = by_key.get(key)
        if row is None:
            failures.append(f"teacher_row_missing:{key}")
            continue
        if row.get("status") != "measured":
            failures.append(
                f"teacher_row_not_measured:{key}:{row.get('status')}")
            continue
        demo = key.split(":", 1)[1]
        _deadline(deadline, f"teacher-label:{key}")
        clip = clip_for_demo(demo)
        product_path = _recorded_file_path(row.get("product"))
        if product_path is None:
            failures.append(f"teacher_product_missing:{key}")
            continue
        try:
            with np.load(product_path, allow_pickle=False) as z:
                if "root" not in z.files or "joints" not in z.files:
                    raise KeyError("root/joints")
                root = np.array(z["root"])
                joints = np.array(z["joints"])
                world_from_clip = (np.array(z["world_from_clip"])
                                   if "world_from_clip" in z.files else None)
                target_box = (np.array(z["target_box"])
                              if "target_box" in z.files else None)
                starts = (np.array(z["window_starts"])
                          if "window_starts" in z.files else None)
        except (OSError, ValueError, KeyError, TypeError):
            failures.append(f"teacher_product_unreadable:{key}")
            continue
        if not np.array_equal(root, clip.root):
            failures.append(f"teacher_root_not_frozen_label:{key}")
        if not np.array_equal(joints, clip.q):
            failures.append(f"teacher_joints_not_frozen_label:{key}")
        if world_from_clip is not None and not np.array_equal(
                world_from_clip, clip.condition.world_from_clip):
            failures.append(f"teacher_world_frame_drifted:{key}")
        if target_box is not None and not np.array_equal(
                target_box, clip.condition.target_obj_pose):
            failures.append(f"teacher_target_drifted:{key}")
        if starts is not None and not np.array_equal(
                starts, np.asarray(clip_starts(clip.T), dtype=np.int64)):
            failures.append(f"teacher_starts_drifted:{key}")
        binding = _mapping(row.get("binding"))
        if stage == "fixed-windows":
            if binding.get("frozen_label_root_sha256") != \
                    hashlib.sha256(clip.root.tobytes()).hexdigest():
                failures.append(f"teacher_binding_root_drifted:{key}")
            if binding.get("frozen_label_q_sha256") != \
                    hashlib.sha256(clip.q.tobytes()).hexdigest():
                failures.append(f"teacher_binding_joints_drifted:{key}")
        else:
            if binding.get("frozen_label_root_bitwise") is not True:
                failures.append(f"teacher_binding_root_drifted:{key}")
            if binding.get("frozen_label_joints_bitwise") is not True:
                failures.append(f"teacher_binding_joints_drifted:{key}")
            if _registered_artifact_path(
                    reg, binding.get("original_product")) is None:
                failures.append(
                    f"teacher_original_product_not_registered:{key}")
        checked += 1
    return {"failures": failures, "evidence_gaps": gaps,
            "checked": checked, "expected": len(expected_keys)}


def _verify_frozen_replays(reg: dict, stage: str, run_root: Path,
                           rows: list[dict], meta: dict,
                           deadline: float, clip_for_demo,
                           attribution_for_demo=None) -> dict:
    """Bind every V0/V1 product and consumed window to its frozen source.

    The S-stage booleans are declarations.  F recomputes the replay from the
    registered windows and compares both the saved product and the recomputed
    world state with the frozen attribution/raw product, bit for bit.
    """
    section = "fixed_windows" if stage == "fixed-windows" else "val31"
    root_dir = Path(run_root) / ("fixed_windows/attempt_001/scenes"
                                 if stage == "fixed-windows"
                                 else "val31/attempt_001/demos")
    source_for = attribution_for_demo or _attribution_result
    failures, gaps, checked = [], [], 0
    by_key = {row.get("key"): row for row in rows}
    if len(by_key) != len(rows):
        failures.append(f"{section}_scene_keys_duplicated")
    keys = []
    for key in reg["expected_keys"][section]:
        if section == "fixed_windows" and (
                (key.startswith("model:") and key.endswith((":V0", ":V1")))
                or (key.startswith("roundtrip:") and key.endswith(":V0"))):
            keys.append(key)
        elif section == "val31" and key.startswith("model:") and key.endswith(":V0"):
            keys.append(key)
    for key in keys:
        row = by_key.get(key)
        if row is None:
            failures.append(f"frozen_replay_row_missing:{key}")
            continue
        if row.get("status") != "measured":
            failures.append(
                f"frozen_replay_row_not_measured:{key}:{row.get('status')}")
            continue
        demo = key.split(":")[1]
        result = meta.get(demo)
        if not isinstance(result, dict):
            gaps.append(f"frozen_replay_scene_missing:{demo}")
            continue
        _deadline(deadline, f"frozen-replay:{key}")
        source_windows = None
        version = key.split(":")[2] if key.startswith("roundtrip:") else key.split(":")[3]
        if key.startswith("model:"):
            model = key.split(":")[2]
            if section == "fixed_windows":
                try:
                    att = source_for(demo)
                except (OSError, ValueError, TypeError):
                    failures.append(f"frozen_replay_attribution_unreadable:{demo}")
                    continue
                variant = model if version == "V0" else f"{model}_all_contributors"
                hits = [r for r in (att.get("rows") or [])
                        if isinstance(r, dict) and r.get("variant") == variant]
                if len(hits) != 1:
                    failures.append(f"frozen_replay_source_row_missing:{key}")
                    continue
                source_record = hits[0].get("product") or {}
                source_windows = ((att.get("models") or {}).get(model) or {}).get("windows")
            else:
                try:
                    source_record = _original_row(demo, model, SEED).get("product") or {}
                except (OSError, ValueError, TypeError):
                    failures.append(f"frozen_replay_source_row_missing:{key}")
                    continue
        else:
            try:
                att = source_for(demo)
            except (OSError, ValueError, TypeError):
                failures.append(f"frozen_replay_attribution_unreadable:{demo}")
                continue
            hits = [r for r in (att.get("rows") or [])
                    if isinstance(r, dict) and r.get("variant") == "teacher_roundtrip"]
            if len(hits) != 1:
                failures.append(f"frozen_replay_source_row_missing:{key}")
                continue
            source_record = hits[0].get("product") or {}
        source_path = _registered_artifact_path(reg, source_record)
        if source_path is None:
            failures.append(f"frozen_source_not_registered:{key}")
            continue
        if source_windows is not None \
                and _registered_artifact_path(reg, source_windows) is None:
            failures.append(f"frozen_replay_window_not_registered:{key}")
            continue
        try:
            frozen_root, frozen_joints = _load_product(source_path)
        except (OSError, ValueError, KeyError, TypeError):
            failures.append(f"frozen_source_unreadable:{key}")
            continue
        product_path = _recorded_file_path(row.get("product"))
        if product_path is None:
            failures.append(f"frozen_replay_product_missing:{key}")
            continue
        try:
            saved_root, saved_joints = _load_product(product_path)
        except (OSError, ValueError, KeyError, TypeError):
            failures.append(f"frozen_replay_product_unreadable:{key}")
            continue
        if not (np.array_equal(saved_root, frozen_root)
                and np.array_equal(saved_joints, frozen_joints)):
            failures.append(f"frozen_replay_product_replaced:{key}")
            continue
        try:
            decoded, starts = _decoded_from_windows_for_row(
                root_dir / demo, row, result, source_windows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(
                f"frozen_replay_windows_unreadable:{key}:{type(exc).__name__}")
            continue
        expected_starts = [int(v) for v in (result.get("starts") or [])]
        if starts != expected_starts:
            failures.append(f"frozen_replay_starts_differ:{key}")
            continue
        clip = clip_for_demo(demo)
        if key.startswith("roundtrip:") or version == "V0":
            fused = _v0_fuse(decoded, starts, result["T"])
        else:
            fused = _v1_fuse(decoded, starts, result["T"])
        world_root, world_joints = attribution.world_state(
            *fused[:3], clip.condition.world_from_clip)
        if not (np.array_equal(world_root, frozen_root)
                and np.array_equal(world_joints, frozen_joints)):
            failures.append(f"frozen_replay_not_bitwise:{key}")
            continue
        checked += 1
    return {"failures": failures, "evidence_gaps": gaps,
            "checked": checked, "expected": len(keys)}


def _decoded_from_windows_for_row(scene_dir: Path, row: dict, result: dict,
                                  source_windows: dict | None = None):
    binding = _mapping(row.get("binding"))
    rec = binding.get("decoded_window")
    if rec:
        if not isinstance(rec, dict):
            raise ValueError("decoded window record is not a mapping")
        if source_windows is not None:
            if not isinstance(source_windows, dict) \
                    or rec.get("path") != source_windows.get("path") \
                    or rec.get("sha256") != source_windows.get("sha256"):
                raise ValueError("decoded window is not the registered source")
        path = ROOT / rec.get("path", "")
        if not rec.get("path") or not path.is_file() \
                or full_sha(path) != rec.get("sha256"):
            raise ValueError("decoded window artifact missing or drifted")
        decoded, starts = _decoded_from_npz(path)
        if starts != result["starts"]:
            raise ValueError("recompute starts differ")
        return decoded, starts
    if row["key"].startswith("roundtrip:"):
        oracle_rec = binding.get("oracle_windows")
        if not isinstance(oracle_rec, dict):
            raise ValueError("oracle window record missing")
        oracle = ROOT / oracle_rec.get("path", "")
        if not oracle_rec.get("path") or not oracle.is_file() \
                or full_sha(oracle) != oracle_rec.get("sha256"):
            raise ValueError("oracle window artifact missing or drifted")
        with np.load(oracle, allow_pickle=False) as z:
            decoded = tuple(np.array(z[f"decoded_{k}"]) for k in ("pos", "quat", "q"))
            starts = [int(v) for v in z["starts"].tolist()]
        return decoded, starts
    raise ValueError(f"decoded windows missing for {row['key']}")


def closed_geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    ra = Rot.from_quat(np.asarray(a)[..., [1, 2, 3, 0]])
    rb = Rot.from_quat(np.asarray(b)[..., [1, 2, 3, 0]])
    return np.rad2deg((ra.inv() * rb).magnitude())


def _remeasure_rows(reg: dict, rows: list[dict], deadline: float,
                    context_for_demo, allowed_rejected=None) -> dict:
    """Re-measure every measured row and reconcile every GQ-facing field.

    An algorithm rejection is skipped only when F independently reproduced it
    for a V2 path; every other status is a verification failure.
    """
    allowed = set(allowed_rejected or ())
    failures, gaps = [], []
    expected_measured = 0
    measured = 0
    out_rows = []
    for row in rows:
        new_row = dict(row)
        key = row.get("key")
        status = row.get("status")
        if status == "algorithm_rejected":
            if not str(key).endswith(":V2") or key not in allowed:
                failures.append(f"remeasure_unexpected_rejection:{key}")
            out_rows.append(new_row)
            continue
        if status != "measured":
            failures.append(f"remeasure_row_status_invalid:{key}:{status}")
            out_rows.append(new_row)
            continue
        expected_measured += 1
        demo = key.split(":")[1]
        rec = row.get("product") or {}
        path = ROOT / rec.get("path", "")
        if not rec.get("path") or not path.is_file() \
                or full_sha(path) != rec.get("sha256"):
            failures.append(f"remeasure_product_missing:{key}")
            out_rows.append(new_row)
            continue
        validator, replayer, clip, teacher_scene, teacher_kpts = context_for_demo(demo)
        try:
            with np.load(path, allow_pickle=False) as z:
                if "root" not in z.files or "joints" not in z.files:
                    raise KeyError("root/joints")
                root, joints = np.array(z["root"]), np.array(z["joints"])
            if root.ndim != 2 or root.shape[1] != 7 \
                    or joints.ndim != 2 or joints.shape[1] != 29:
                raise ValueError("product array shape invalid")
        except (OSError, ValueError, KeyError, TypeError):
            failures.append(f"remeasure_product_unreadable:{key}")
            out_rows.append(new_row)
            continue
        try:
            metric = _measure_clip(
                validator, replayer, clip, teacher_scene, teacher_kpts,
                root, joints, demo, reg["task_gate_tol_m"], deadline,
                f"remeasure:{key}")
        except (OSError, ValueError, KeyError, IndexError, AttributeError,
                TypeError) as exc:
            failures.append(
                f"remeasure_failed:{key}:{type(exc).__name__}")
            out_rows.append(new_row)
            continue
        if metric.get("status") != "measured":
            gaps.append(f"remeasure_incomplete:{key}")
            out_rows.append(new_row)
            continue
        for name in _metric_mismatches(row, metric):
            failures.append(f"metric_changed:{key}:{name}")
        new_row.update({
            "status": metric["status"],
            "metrics": metric.get("metrics"),
            "measurement_gaps": metric.get("measurement_gaps", []),
            "root_position_max_error_m": metric.get("root_position_max_error_m"),
            "root_rotation_max_error_deg": metric.get("root_rotation_max_error_deg"),
            "joint_max_error_rad": metric.get("joint_max_error_rad"),
            "hand_fk_mean_m": metric.get("hand_fk_mean_m"),
            "angular_velocity_max_radps": metric.get("angular_velocity_max_radps"),
            "seam_rotation_max_deg": metric.get("seam_rotation_max_deg"),
            "keyframe_penetration_count": metric.get("keyframe_penetration_count"),
        })
        measured += 1
        out_rows.append(new_row)
    return {"rows": out_rows, "failures": failures, "evidence_gaps": gaps,
            "measured": measured, "expected_measured": expected_measured}


def _experiment_complete(fixed_row: dict | None, val_row: dict | None,
                         failures: list, gaps: list) -> bool:
    return bool(fixed_row is not None and fixed_row.get("complete") is True
                and val_row is not None and val_row.get("complete") is True
                and not failures and not gaps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "register", "fixed-windows", "val31", "finalize",
        "fixed-window-unit", "val31-unit"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--demo")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.stage == "register":
        register(args.run_root)
        return 0
    run = args.run_root.expanduser().resolve()
    if run.parent != cohort_root():
        raise SystemExit("run root must be a direct child of the cohort root")
    if not (run / "freeze/execution_registration.json").is_file():
        raise SystemExit("run is not registered")
    if args.stage in ("fixed-window-unit", "val31-unit"):
        if args.demo is None or args.out is None or args.deadline is None:
            raise SystemExit("unit requires --demo --out --deadline")
        out = args.out.expanduser().resolve()
        if not out.is_relative_to(run):
            raise SystemExit("unit output must stay inside the run root")
        if args.stage == "fixed-window-unit":
            result = fixed_unit(run, args.demo, out, args.deadline)
        else:
            result = val31_unit(run, args.demo, out, args.deadline)
        return 0 if result.get("status") == "ok" else 2
    if args.stage == "fixed-windows":
        return 0 if fixed_stage(run) else 2
    if args.stage == "val31":
        return 0 if val31_stage(run) else 2
    return 0 if _finalize(run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
