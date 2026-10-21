"""W1 prediction/window consistency diagnostic (plan I -> R -> W -> P -> F).

Entry points::

    register        R: verify read-only sources, freeze the exact matrix
    saved-windows   W: 130 saved windows + 65 teacher references, no model calls
    probes          P: 520 fixed probe units (oracle / Det / 5t denoise / DDIM25)
    finalize        F: independent recomputation, fixed three-demo replay, closure

The script never trains, never changes production entry points and never
promotes a fusion candidate. All model calls are explicit and bounded; a
numerical failure is recorded as ``execution_failed`` rather than retried.
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
import torch

import diagnose_w1_attribution as attribution
import eval_generation as raw
import omniretarget_w1_de as driver
import omniretarget_w1_rotation_fusion as fusion
from ireset.data import w1_training as w1
from ireset.data.a2_integrity import sha256_file
from ireset.data.windows import TeacherWindowDataset, make_tokenizer
from ireset.eval import prediction_consistency as pc
from ireset.eval.teacher_comparison import TeacherQualityReplayer
from ireset.kinematics.mujoco_adapter import G1Kinematics
from ireset.model.tokens import RootStats
from ireset.repair.validation import IndependentValidator, ValidationConfig

EXECUTION_ID = "w1_prediction_consistency_v1"
ACCOUNTING_SCHEMA = "pre_registered_v1"
SOURCE = ROOT / "derived/experiments/omniretarget_first/20261013_be00c50_w1_de"
ATTRIBUTION = ROOT / "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution"
FUSION = ROOT / "derived/experiments/omniretarget_first/20261019_worktree_w1_rotation_fusion_r10"
PLAN = ROOT / "docs/plans/omniretarget_w1_prediction_consistency_plan.md"
KNOWN_CASE = "sub16_largebox_048"
SEED, DDIM_STEPS = 7, 25
WINDOW, STRIDE = 60, 30
TOTAL_BUDGET_S = 7200.0
STAGE_BUDGETS = {
    "register": (600.0, 60.0),
    "saved-windows": (1800.0, 120.0),
    "probes": (3000.0, 180.0),
    "finalize": (1800.0, 180.0),
}
STAGE_DIRS = {"register": "freeze", "saved-windows": "saved_windows",
              "probes": "probes", "finalize": "decisions"}
STAGE_ATTEMPT_DIR = {"register": "freeze/attempt_001",
                     "saved-windows": "saved_windows/attempt_001",
                     "probes": "probes/attempt_001",
                     "finalize": "decisions/attempt_001"}
EXPECTED_CKPTS = {
    "deterministic": "7b1efd57ceff6510d95c5fd682649c7900297030923904c3a7542f59af7b2eff",
    "diffusion": "1f9c16b124ab2d77877e93cfa39af124aa30a4f5aeb998f05350de26dcd437b1",
}
EXPECTED_T = {
    "sub10_largebox_063": 133, "sub12_largebox_047": 358,
    "sub16_largebox_000": 89, "sub3_largebox_001": 178,
    "sub7_largebox_003": 147, "sub8_largebox_005": 190,
    "sub10_largebox_100": 73, "sub12_largebox_008": 161,
    "sub16_largebox_052": 147, "sub3_largebox_018": 191,
    "sub7_largebox_032": 184, "sub8_largebox_001": 151,
    "sub16_largebox_048": 168,
}
REPLAY_DEMOS = ("sub10_largebox_063", "sub10_largebox_100", "sub16_largebox_048")
IMPLEMENTATION_FILES = (
    "src/ireset/eval/prediction_consistency.py",
    "scripts/diagnose_w1_prediction_consistency.py",
    "tests/test_w1_prediction_consistency.py",
    "docs/plans/omniretarget_w1_prediction_consistency_plan.md",
)
RUNTIME_DEPENDENCIES = (
    "src/ireset/model/generator.py", "src/ireset/model/stitch.py",
    "src/ireset/model/tokens.py", "src/ireset/model/rotation_fusion.py",
    "src/ireset/eval/prediction_consistency.py",
    "scripts/eval_generation.py", "scripts/omniretarget_w1_de.py",
    "scripts/omniretarget_w1_de_detail.py",
    "scripts/diagnose_w1_attribution.py",
    "scripts/omniretarget_w1_rotation_fusion.py",
)
PLAN_ANCHORS = {
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/freeze/execution_registration.json":
        "5561f68562189b7ba45f7193c280739e6dde5325b4df2ada82e8a9cf69ba5f7e",
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/decisions/attempt_001/w1_de_final.json":
        "7f1a9a595cf7964e99a929c83d3bcaac4aa3482a96e515228921b7b67a576737",
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/evaluation/checkpoint_selection.json":
        "a49f0ffea1d51dd103f0aa728794e7c891ae3c42fe59075519d68c452617b9af",
    "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/freeze/registration.json":
        "499257b0a983b8af1c94a3d819d573d9be5470cb97dcdf4c02a84ac03d4bab52",
    "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/diagnosis/attempt_001/attribution.json":
        "72186392cce075ce5893c68f82acaa41fd3efd45b534970696f698f5563c07ac",
    "derived/experiments/omniretarget_first/20261015_9000ac8_w1_attribution/decisions/diagnostic_closure.json":
        "93cac087a881e1a05c3f38d3542208e7690aa69afaaa30a626999960136c9064",
    "derived/experiments/omniretarget_first/20261019_worktree_w1_rotation_fusion_r10/freeze/execution_registration.json":
        "6ad7649e060eb078f6e048b946e447c5337a6a50c970db38bf33603be6df5400",
    "derived/experiments/omniretarget_first/20261019_worktree_w1_rotation_fusion_r10/decisions/attempt_001/fusion_final.json":
        "055432bf9d201408175a1b560ec9a69cdc01f3879fc9be6f7ff25247f56b9abe",
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/formal/deterministic/attempt_001/best.pt":
        EXPECTED_CKPTS["deterministic"],
    "derived/experiments/omniretarget_first/20261013_be00c50_w1_de/formal/diffusion/attempt_001/best.pt":
        EXPECTED_CKPTS["diffusion"],
}


# ---------------------------------------------------------------------------
# JSON / paths / hashes
# ---------------------------------------------------------------------------

def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, data, *, exclusive: bool = False) -> None:
    driver.atomic_json(Path(path), attribution.safe_json(data), exclusive=exclusive)


def record_path(path: Path) -> str:
    return attribution.record_path(Path(path))


def file_record(path: Path) -> dict:
    return {"path": record_path(path), "sha256": sha256_file(path)}


def full_sha(path: Path) -> str:
    return sha256_file(Path(path))


def _hash_map(anchors: dict[str, str]) -> list[str]:
    problems = []
    for rel, want in anchors.items():
        path = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.is_file() or full_sha(path) != want:
            problems.append(f"anchor_missing_or_drifted:{rel}")
    return problems


def cohort_root() -> Path:
    return (ROOT / "derived/experiments/omniretarget_first").resolve()


def guard_run(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    derived = (ROOT / "derived").resolve()
    if not path.is_relative_to(derived):
        raise ValueError("run root must resolve under derived/")
    if path.parent != cohort_root():
        raise ValueError("run root must be a direct child of the fixed cohort root")
    for source in (SOURCE, ATTRIBUTION, FUSION):
        source = source.resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("run root must not overlap a read-only source")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("run root must be empty; no implicit retry or overwrite")
    return path


def resolve_run(path: Path) -> Path:
    """An existing run root in the fixed cohort (used by non-register stages)."""
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to((ROOT / "derived").resolve()):
        raise ValueError("run root must resolve under derived/")
    if path.parent != cohort_root():
        raise ValueError("run root must be a direct child of the fixed cohort root")
    for source in (SOURCE, ATTRIBUTION, FUSION):
        source = source.resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("run root must not overlap a read-only source")
    if not path.is_dir():
        raise ValueError("run root does not exist")
    return path


def _select_train6(train24: list[str]) -> list[str]:
    if len(train24) != 24 or len(set(train24)) != 24:
        raise ValueError("train24 roster malformed")
    subjects = sorted({d.split("_", 1)[0] for d in train24})
    if len(subjects) != 6:
        raise ValueError("six subjects required")
    return [min(d for d in train24 if d.split("_", 1)[0] == subject)
            for subject in subjects]


def cohort_id(source_registration_sha: str) -> str:
    payload = {
        "execution_id": EXECUTION_ID,
        "protocol": "w1_prediction_consistency_v1",
        "source_registration_sha256": source_registration_sha,
        "attribution_registration_sha256": full_sha(ATTRIBUTION / "freeze/registration.json"),
        "checkpoints": EXPECTED_CKPTS, "seed": SEED, "ddim_steps": DDIM_STEPS,
        "noise_seed": pc.NOISE_SEED,
        "noise_protocol": pc.NOISE_PROTOCOL,
        "window": WINDOW, "stride": STRIDE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


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
        "untracked": untracked, "files": files,
    }


def historical_implementation(fus_reg: dict) -> dict:
    """Verify r10 historical bytes against its own snapshot, not current files."""
    snap = fus_reg.get("implementation_snapshot") or {}
    files = {}
    problems = []
    for rel, rec in (snap.get("files") or {}).items():
        path = ROOT / rec["snapshot"]
        if not path.is_file() or full_sha(path) != rec["sha256"]:
            problems.append(f"historical_snapshot_drift:{rel}")
        files[rel] = {"sha256": rec["sha256"], "snapshot": rec["snapshot"]}
    tracker = (snap.get("tracked_diff") or {}).get("path")
    if tracker:
        p = ROOT / tracker
        if not p.is_file() or full_sha(p) != snap["tracked_diff"]["sha256"]:
            problems.append("historical_tracked_diff_drift")
    if problems:
        raise ValueError(problems)
    return {"files": files, "tracked_diff": snap.get("tracked_diff"),
            "head": snap.get("head"), "note":
            "historical r10 bytes verified in its snapshot; current implementation "
            "identity is registered separately"}


# ---------------------------------------------------------------------------
# ledger / index / budget / sealed publication (same semantics as W1-D/E)
# ---------------------------------------------------------------------------

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


def _num_cost(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0)


def _upsert_cost(run: Path, stage: str, attempt: str, wall: float,
                 extra: dict, status: str) -> None:
    if not _num_cost(wall):
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
                         *, cohort: str | None = None) -> None:
    if cohort is not None and not (run / "cohort.json").is_file():
        write_json(run / "cohort.json", {"execution_id": EXECUTION_ID,
                                         "cohort_id": cohort})
    elapsed = max(0.0, time.perf_counter() - t0)
    entry = {"status": "running", "complete": False,
             "measured_wall_s": elapsed,
             "accounting_schema": ACCOUNTING_SCHEMA,
             "execution_registration_sha256": _registration_sha(run)}
    if cohort is not None:
        entry["cohort_id"] = cohort
    if any(r.get("stage") == stage and r.get("attempt") == attempt for r in _index(run)):
        _replace_index(run, stage, attempt, entry)
    else:
        _append_index(run, stage, attempt, entry)
    extra = {"complete": False, "accounting_schema": ACCOUNTING_SCHEMA}
    if cohort is not None:
        extra["cohort_id"] = cohort
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
        if (not _num_cost(e.get("measured_wall_s")) or not _num_cost(e.get("wall_s"))
                or abs(round(float(e["measured_wall_s"]), 1) - float(e["wall_s"])) > 1e-9):
            problems.append("attempt_cost_invalid")
        return row, problems
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, [f"stage_state_unreadable:{type(exc).__name__}"]


def _stage_history(run: Path, stage: str) -> float | None:
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
            if float(index_measured) != float(measured):
                return None
            if float(rounded) != round(float(measured), 1):
                return None
            if "wall_s" in index_row:
                if (not _num_cost(index_row["wall_s"])
                        or float(index_row["wall_s"]) != round(float(measured), 1)):
                    return None
            total += float(measured)
        return total
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _total_history(run: Path) -> float | None:
    total = 0.0
    for stage in STAGE_BUDGETS:
        value = _stage_history(run, stage)
        if value is None:
            return None
        total += value
    return total


def _cohort_history(run: Path, cohort: str) -> dict:
    runs, by_stage, total = [], {}, 0.0
    run_resolved = Path(run).resolve()
    base = cohort_root()
    if not run_resolved.parent == base:
        return {"known": False, "reason": "run_not_in_cohort_root", "runs": runs}
    if not base.is_dir():
        return {"known": True, "runs": runs, "stage_cost_s": {}, "total_s": 0.0}
    for sibling in sorted(base.iterdir()):
        if sibling == run_resolved or not sibling.is_dir():
            continue
        reg_path = sibling / "freeze/execution_registration.json"
        if not reg_path.is_file():
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
                            "reason": f"unregistered_attempt:{sibling.name}:{stage}",
                            "runs": runs}
                stage_costs[stage] = cost
                by_stage[stage] = by_stage.get(stage, 0.0) + cost
                total += cost
            runs.append({"run": record_path(sibling), "stage_cost_s": stage_costs,
                         "unregistered": True})
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
    return {"known": True, "runs": runs, "stage_cost_s": by_stage, "total_s": total}


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
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_registration")
        status = "incomplete"
    row = publish(row)
    if not overrun and time.perf_counter() - t0 >= remaining:
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_publication")
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


def _deadline(deadline: float, phase: str) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError(f"prediction_consistency_deadline:{phase}")


# ---------------------------------------------------------------------------
# source binding and context
# ---------------------------------------------------------------------------

def _verify_plan_and_sources() -> tuple[dict, dict, dict, dict, dict]:
    problems = _hash_map(PLAN_ANCHORS)
    problems += _hash_map({record_path(PLAN): full_sha(PLAN)})
    if problems:
        raise ValueError(problems)
    w1_reg = read_json(SOURCE / "freeze/execution_registration.json")
    selection = read_json(SOURCE / "evaluation/checkpoint_selection.json")
    att_reg = read_json(ATTRIBUTION / "freeze/registration.json")
    fus_reg = read_json(FUSION / "freeze/execution_registration.json")
    fus_final = read_json(FUSION / "decisions/attempt_001/fusion_final.json")
    if w1_reg.get("execution_id") != "w1_de_v1":
        raise ValueError("wrong SOURCE execution")
    if att_reg.get("execution_id") != "w1_attribution_v1":
        raise ValueError("wrong attribution execution")
    if fus_reg.get("execution_id") != "w1_rotation_fusion_v1":
        raise ValueError("wrong rotation-fusion execution")
    if (fus_final.get("experiment_complete") is not True
            or fus_final.get("final_interaction_method") != "not_ready"
            or fus_final.get("integration_eligible") is not False):
        raise ValueError("FUSION final is not the closed not-ready baseline")
    for model, want in EXPECTED_CKPTS.items():
        rec = selection["models"][model]
        if (rec.get("sha256") != want or rec.get("step") != 4000
                or full_sha(ROOT / rec["path"]) != want):
            raise ValueError(f"unregistered checkpoint:{model}")
    return w1_reg, selection, att_reg, fus_reg, fus_final


def _verify_attribution_windows(att_reg: dict) -> tuple[dict, list[dict], list[dict]]:
    closure = read_json(ATTRIBUTION / "decisions/diagnostic_closure.json")
    if closure.get("complete") is not True:
        raise ValueError("attribution closure is not complete")
    report_rec = (closure.get("artifacts") or {}).get("diagnosis") or {}
    report_path = ATTRIBUTION / "diagnosis/attempt_001/attribution.json"
    if report_rec.get("sha256") != full_sha(report_path):
        raise ValueError("attribution report is not the closure-anchored report")
    report = read_json(report_path)
    if report.get("complete") is not True or report.get("evidence_gaps"):
        raise ValueError("attribution report is not complete evidence")
    results = report.get("results") or []
    if len(results) != 13 or {r.get("demo") for r in results} != set(att_reg["demos"]):
        raise ValueError("attribution result matrix is not the registered 13 demos")
    windows = []
    scenes = []
    for result in results:
        demo = result["demo"]
        scene_path = ATTRIBUTION / "diagnosis/attempt_001" / demo / "result.json"
        if not scene_path.is_file() or read_json(scene_path) != result:
            raise ValueError(f"attribution per-demo result drifted:{demo}")
        scenes.append(file_record(scene_path))
        if result.get("status") != "ok":
            raise ValueError(f"attribution scene not ok:{demo}")
        for model in ("deterministic", "diffusion"):
            rec = ((result.get("models") or {}).get(model) or {}).get("windows") or {}
            path = ROOT / str(rec.get("path", ""))
            if not path.is_file() or full_sha(path) != rec.get("sha256"):
                raise ValueError(f"attribution window missing/drifted:{demo}:{model}")
            windows.append({"demo": demo, "model": model,
                            "path": rec["path"], "sha256": rec["sha256"]})
    if len(windows) != 26:
        raise ValueError("attribution saved-window denominator is not 26")
    return report, windows, scenes


def _verify_fusion_binding(fus_reg: dict, fus_final: dict) -> dict:
    hist = historical_implementation(fus_reg)
    # FUSION anchors include historical implementation files that may legally
    # differ from current bytes; verify every non-implementation anchor.
    skip = set(fus_reg.get("implementation_snapshot", {}).get("files", {}))
    anchors = {k: v for k, v in fus_reg.get("anchors", {}).items() if k not in skip}
    problems = _hash_map(anchors)
    if problems:
        raise ValueError([f"fusion:{p}" for p in problems])
    counts = (fus_final.get("fixed_windows") or {}).get("counts") or {}
    gs = (fus_final.get("fixed_windows") or {}).get("gs") or {}
    return {
        "registration": file_record(FUSION / "freeze/execution_registration.json"),
        "final": file_record(FUSION / "decisions/attempt_001/fusion_final.json"),
        "anchors": len(fus_reg.get("anchors", {})),
        "non_implementation_anchors": len(anchors),
        "historical_implementation": hist,
        "known_case_v2_rejection": (
            (((gs.get("facts") or {}).get("known_case") or {})
             .get("deterministic") or {}).get("rejection")),
        "fusion_counts": counts,
        "fusion_final_status": {
            "implementation_contract_passed": fus_final.get("implementation_contract_passed"),
            "experiment_complete": fus_final.get("experiment_complete"),
            "integration_eligible": fus_final.get("integration_eligible"),
            "final_interaction_method": fus_final.get("final_interaction_method"),
        },
    }


def _load_context(reg: dict, demo: str):
    w1_reg = read_json(SOURCE / "freeze/execution_registration.json")
    selection = read_json(SOURCE / "evaluation/checkpoint_selection.json")
    manifest = read_json(ROOT / w1_reg["training_manifest"]["path"])
    prereg = read_json(ROOT / w1_reg["preregistration"]["path"])
    hits = [r for r in manifest["rows"] if r.get("demo") == demo]
    if len(hits) != 1 or hits[0].get("admission", {}).get("motion") is not True:
        raise ValueError(f"demo not uniquely admitted:{demo}")
    row = hits[0]
    cache = ROOT / w1_reg["cache"]["dir"]
    clip = w1.load_w1_derived_clip(
        row, cache / "schedules", ROOT,
        prereg["assets"]["a1_assets_sha256"], w1_reg["condition_code_sha256"])
    expected_T = reg["t_by_demo"][demo]
    if clip.T != expected_T:
        raise ValueError(f"{demo}: T={clip.T} != registered {expected_T}")
    stats = RootStats.load(cache / "root_stats.npz")
    tokenizer = make_tokenizer(stats)
    validator = IndependentValidator(
        cfg=ValidationConfig(**w1_reg["evaluation"]["validation_config"]))
    replayer = TeacherQualityReplayer(validator)
    teacher_scene = np.concatenate(
        (clip.root, clip.q, clip.condition.target_obj_pose), axis=1)
    teacher_kpts = raw.fk_keypoints(G1Kinematics(), teacher_scene)
    return {"w1_reg": w1_reg, "selection": selection, "manifest": manifest,
            "prereg": prereg, "row": row, "cache": cache, "clip": clip,
            "tokenizer": tokenizer, "validator": validator,
            "replayer": replayer, "teacher_scene": teacher_scene,
            "teacher_kpts": teacher_kpts, "device": w1_reg["configs"][
                "deterministic"]["device"]}


def _load_generator(ctx, model: str, *, resample_attempts: int = 1):
    w1_reg, selection, cache = ctx["w1_reg"], ctx["selection"], ctx["cache"]
    gen = raw.load_w1_de_generator(
        selection["models"][model], w1_reg["configs"][model]["device"],
        DDIM_STEPS, cache, w1_reg["training_manifest"]["sha256"],
        w1_reg["cache"]["identity_sha256"], w1_reg["cache"]["root_stats_sha256"],
        full_sha(SOURCE / "freeze/execution_registration.json"),
        registration=w1_reg, project_root=ROOT)
    gen.resample_attempts = resample_attempts
    gen.window, gen.stride = WINDOW, STRIDE
    return gen


def _condition_arrays(clip) -> dict:
    from ireset.data.teacher import keypoints_in_source_object, world_to_clip

    cond = clip.condition
    wfc = cond.world_from_clip
    return {
        "kps_obj": keypoints_in_source_object(cond.human_kps_world, cond.source_obj_pose),
        "kps_clip": world_to_clip(cond.human_kps_world, wfc),
        "box_t_clip": world_to_clip(cond.target_obj_pose[:, 0:3], wfc),
        "box_quat": cond.target_obj_pose[:, 3:7],
        "contact": cond.contact_flags().astype(np.float32),
    }


def _window_batch(arrays: dict, starts: list[int]) -> dict:
    return {k: np.stack([np.asarray(v)[s:s + WINDOW] for s in starts])
            for k, v in arrays.items()}


def _teacher_window_geometry(ctx, start: int, deadline: float) -> dict:
    clip, cond = ctx["clip"], ctx["clip"].condition
    s = int(start)
    scene = np.concatenate((clip.root[s:s + WINDOW], clip.q[s:s + WINDOW],
                            cond.target_obj_pose[s:s + WINDOW]), axis=1)
    return pc.window_geometry_series(
        ctx["replayer"], scene, cond.human_kps_world[s:s + WINDOW],
        cond.target_obj_pose[s:s + WINDOW], deadline=deadline)


def _attribution_result(demo: str) -> dict:
    return read_json(ATTRIBUTION / "diagnosis/attempt_001" / demo / "result.json")


def _load_saved_windows(att_result: dict, model: str, starts: list[int]) -> dict:
    rec = att_result["models"][model]["windows"]
    path = ROOT / rec["path"]
    if not path.is_file() or full_sha(path) != rec["sha256"]:
        raise ValueError(f"decoded window drift:{model}")
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.array(z[k], copy=True) for k in z.files}
    if data["starts"].tolist() != list(starts):
        raise ValueError("saved window starts differ")
    for name, shape in (("decoded_pos", (len(starts), WINDOW, 3)),
                        ("decoded_quat", (len(starts), WINDOW, 4)),
                        ("decoded_q", (len(starts), WINDOW, 29)),
                        ("normalized_prediction", (len(starts), WINDOW, 38)),
                        ("window_root", (len(starts), WINDOW, 7))):
        if data.get(name, np.empty(0)).shape != shape:
            raise ValueError(f"saved window shape:{name}")
        if not np.isfinite(data[name]).all():
            raise ValueError(f"saved window nonfinite:{name}")
    data["record"] = rec
    data["path"] = path
    return data


def _condition_checks(ctx, att_result: dict, saved_by_model: dict,
                      demo: str, starts: list[int], deadline: float) -> dict:
    expected = _condition_arrays(ctx["clip"])
    dataset_rows = {}
    split = ctx["row"]["split"]
    dataset = TeacherWindowDataset(
        split, ctx["tokenizer"],
        training_manifest=ROOT / ctx["w1_reg"]["training_manifest"]["path"],
        schedule_cache_dir=ctx["cache"] / "schedules", verify_assets=True)
    for index, (ci, start) in enumerate(dataset.index):
        lc = dataset.clips[ci]
        if lc.demo == demo:
            dataset_rows[int(start)] = dataset[index]
    regular = set(pc.regular_starts(ctx["clip"].T))
    checks = {"dataset_starts": sorted(dataset_rows), "per_window": {},
              "tail_inference_only": {}, "models": sorted(saved_by_model),
              "problems": []}
    for i, start in enumerate(starts):
        s = int(start)
        row = {"start": s, "kind": "regular" if s in regular else "tail",
               "models": {}}
        for model, saved in saved_by_model.items():
            model_row = {"saved_bitwise": {}, "dataset_float32_bitwise": {}}
            for key in pc.CONDITION_KEYS:
                want = np.asarray(expected[key])[s:s + WINDOW]
                got = np.asarray(saved[f"condition_{key}"][i])
                model_row["saved_bitwise"][key] = bool(
                    got.shape == want.shape and np.array_equal(got, want))
                if s in regular:
                    item = dataset_rows.get(s)
                    if item is None:
                        model_row["dataset_float32_bitwise"][key] = False
                    else:
                        actual = item[key]
                        expect32 = torch.as_tensor(want, dtype=actual.dtype)
                        model_row["dataset_float32_bitwise"][key] = bool(
                            actual.shape == expect32.shape
                            and torch.equal(actual, expect32))
                else:
                    model_row["dataset_float32_bitwise"][key] = None
            model_row["passed"] = all(model_row["saved_bitwise"].values()) and (
                all(v for v in model_row["dataset_float32_bitwise"].values()
                    if v is not None))
            row["models"][model] = model_row
        row["passed"] = all(m["passed"] for m in row["models"].values())
        if row["kind"] == "tail":
            row["inference_only_start"] = True
            row["tail_start_matches_T_minus_window"] = s == ctx["clip"].T - WINDOW
        if not row["passed"]:
            checks["problems"].append(f"condition_binding:{s}")
        checks["per_window"][str(s)] = row
        if s not in regular:
            checks["tail_inference_only"][str(s)] = row
    if set(dataset_rows) != regular:
        checks["problems"].append("dataset_regular_denominator_mismatch")
    checks["passed"] = not checks["problems"]
    return checks


# ---------------------------------------------------------------------------
# saved-windows stage (W)
# ---------------------------------------------------------------------------

def _measure_saved_window(*, ctx, demo: str, model: str, start: int, kind: str,
                          group: str, pred_x0: np.ndarray, teacher_x0: np.ndarray,
                          decoded, saved, index: int, model_geometry: dict,
                          teacher_geometry: dict, assembler, deadline: float) -> dict:
    _deadline(deadline, "saved-window-measure")
    clip = ctx["clip"]
    out = pc.measure_window(
        demo=demo, model=model, start=int(start), T=clip.T, kind=kind,
        group=group, pred_x0=pred_x0, teacher_x0=teacher_x0, decoded=decoded,
        world_from_clip=clip.condition.world_from_clip,
        teacher_root=clip.root, teacher_joints=clip.q,
        teacher_kpts=ctx["teacher_kpts"],
        model_geometry=model_geometry, teacher_geometry=teacher_geometry,
        tokenizer=ctx["tokenizer"], assembler=assembler,
        schedule=clip.condition.schedule, frame_dt=clip.condition.frame_dt,
        task_tol_m=ctx["w1_reg"]["evaluation"]["task_gate"]["tol_m"],
        hand_gap_m=ctx["validator"].cfg.hand_gap_m,
        saved_world_root=saved["window_root"][index],
        decode_device=ctx["device"])
    if out.get("metrics") is not None:
        rebuild = pc.decode_production(pred_x0, ctx["tokenizer"],
                                       device=ctx["device"])
        saved_decoded = (saved["decoded_pos"][index], saved["decoded_quat"][index],
                         saved["decoded_q"][index])
        out["metrics"]["key"] = pc.window_key(demo, model, int(start))
        out["metrics"]["attribution_window"] = saved["record"]
        out["metrics"]["decode_consistency"] = {
            "saved_normalized_bitwise": bool(np.array_equal(
                np.asarray(pred_x0), np.asarray(saved["normalized_prediction"][index]))),
            "decoded_saved_max_abs": {
                name: float(np.abs(np.asarray(rebuild[i], dtype=np.float64)
                                   - np.asarray(saved_decoded[i], dtype=np.float64)).max())
                for i, name in enumerate(("pos", "quat", "q"))},
            "decoded_saved_bitwise": {
                name: bool(np.array_equal(np.asarray(rebuild[i]),
                                          np.asarray(saved_decoded[i])))
                for i, name in enumerate(("pos", "quat", "q"))},
        }
    return out


def _fused_facts(ctx, att_result: dict, model: str, variant: str,
                 window_penetration: np.ndarray, starts: list[int],
                 deadline: float) -> dict:
    _deadline(deadline, "fused-facts")
    row = next(r for r in att_result["rows"] if r.get("variant") == variant)
    rec = row["product"]
    path = ROOT / rec["path"]
    if not path.is_file() or full_sha(path) != rec["sha256"]:
        raise ValueError(f"fused product drift:{variant}")
    with np.load(path, allow_pickle=False) as z:
        root, joints = np.array(z["root"]), np.array(z["joints"])
        pen = np.array(z["curve__penetration_keyframe"])
    if root.shape != (ctx["clip"].T, 7) or joints.shape != (ctx["clip"].T, 29):
        raise ValueError("fused product shape")
    temporal = pc.fused_temporal_facts(
        root=root, joints=joints, starts=starts,
        frame_dt=ctx["clip"].condition.frame_dt)
    classes = pc.same_frame_penetration_classes(
        demo=ctx["clip"].condition.source_clip_id.split("_original")[0],
        model=model, starts=starts, T=ctx["clip"].T,
        window_penetration=window_penetration, fused_penetration=pen)
    return {"product": rec, "temporal": temporal, "penetration": classes}


def recompute_saved_windows(reg: dict, demo: str, *, deadline: float,
                            write: Path | None = None,
                            curve_root: Path | None = None) -> dict:
    ctx = _load_context(reg, demo)
    clip = ctx["clip"]
    starts = pc.inference_starts(clip.T)
    group = reg["groups"][demo]
    att_result = _attribution_result(demo)
    if att_result.get("status") != "ok" or att_result["starts"] != starts:
        raise ValueError(f"attribution scene not aligned:{demo}")
    teacher_x0 = pc.encode_teacher(clip, ctx["tokenizer"])
    teacher_windows = pc.teacher_windows(teacher_x0, starts)
    teacher_geom = {}
    teacher_rows = []
    for start in starts:
        geom = _teacher_window_geometry(ctx, start, deadline)
        teacher_geom[int(start)] = geom
        kind = "tail" if pc.is_tail_start(clip.T, int(start)) else "regular"
        facts = pc.geometry_window_facts(geom, geom, np.zeros((WINDOW, 52, 3)),
                                         ctx["teacher_kpts"][start:start + WINDOW],
                                         int(start), ctx["validator"].cfg)
        teacher_rows.append({
            "key": pc.teacher_key(demo, int(start)), "demo": demo,
            "start": int(start), "kind": kind, "group": group, "T": clip.T,
            "status": "measured", "geometry": facts,
            "reference": pc.reference_window_facts(clip.condition.schedule,
                                                   int(start), WINDOW),
            "teacher_kpt_error_w_m": facts["teacher_kpt_error_w_m"],
            "measurement_gaps": [],
        })
        if write is not None:
            _save_arrays(write / "teacher_curves" / f"{demo}__{start}.npz",
                         {"penetration": geom["penetration"],
                          "teacher_penetration": geom["penetration"],
                          "teacher_hand_gap_l": geom["hand_gap"]["l"],
                          "teacher_hand_gap_r": geom["hand_gap"]["r"],
                          "teacher_foot_gap_l": geom["foot_gap"]["l"],
                          "teacher_foot_gap_r": geom["foot_gap"]["r"],
                          "robot_kpts": geom["robot_kpts"]})
    saved_by_model = {model: _load_saved_windows(att_result, model, starts)
                      for model in ("deterministic", "diffusion")}
    condition_checks = _condition_checks(ctx, att_result, saved_by_model,
                                         demo, starts, deadline)
    rows, pair_rows, model_facts = [], [], {}
    curve_problems = []
    curves_dir = write / "curves" if write is not None else None
    for model in ("deterministic", "diffusion"):
        _deadline(deadline, "model-windows")
        saved = saved_by_model[model]
        decoded = (saved["decoded_pos"], saved["decoded_quat"], saved["decoded_q"])
        world_root = pc.lift_clip_windows_to_world(
            decoded[0], decoded[1], clip.condition.world_from_clip)
        teacher_root_windows = np.stack(
            [clip.root[s:s + WINDOW] for s in starts])
        root_pos_err = np.linalg.norm(
            world_root[:, :, :3] - teacher_root_windows[:, :, :3], axis=-1)
        root_rot_err = pc.quat_angle_deg(
            world_root[:, :, 3:7], teacher_root_windows[:, :, 3:7])
        assembler = pc.make_loss_assembler(
            ctx["tokenizer"], ctx["w1_reg"]["configs"][model]["loss"],
            ctx["device"])
        fk_windows = []
        window_pen = np.zeros((len(starts), WINDOW))
        for i, start in enumerate(starts):
            _deadline(deadline, "window-geometry")
            kind = "tail" if pc.is_tail_start(clip.T, int(start)) else "regular"
            scene = np.concatenate((world_root[i], decoded[2][i],
                                    clip.condition.target_obj_pose[start:start + WINDOW]), axis=1)
            geom = pc.window_geometry_series(
                ctx["replayer"], scene,
                clip.condition.human_kps_world[start:start + WINDOW],
                clip.condition.target_obj_pose[start:start + WINDOW],
                deadline=deadline)
            fk_windows.append(geom["robot_kpts"])
            window_pen[i] = geom["penetration"]
            out = _measure_saved_window(
                ctx=ctx, demo=demo, model=model, start=int(start), kind=kind,
                group=group, pred_x0=saved["normalized_prediction"][i],
                teacher_x0=teacher_windows[i], decoded=tuple(a[i] for a in decoded),
                saved=saved, index=i, model_geometry=geom,
                teacher_geometry=teacher_geom[int(start)], assembler=assembler,
                deadline=deadline)
            if out.get("metrics") is None:
                rows.append({
                    "key": pc.window_key(demo, model, int(start)),
                    "demo": demo, "model": model, "window_start": int(start),
                    "T": clip.T, "kind": kind, "group": group,
                    "status": "measurement_incomplete",
                    "measurement_gaps": out.get("measurement_gaps", []),
                })
                continue
            rows.append(out["metrics"])
            curve_path = None
            if curves_dir is not None:
                curve_path = curves_dir / f"{demo}__{model}__{start}.npz"
                _save_arrays(curve_path, out["curves"])
            elif curve_root is not None:
                curve_path = curve_root / f"{demo}__{model}__{start}.npz"
            if curve_path is not None:
                curve_problems += _compare_curve_file(
                    curve_path, out["curves"],
                    f"w_curve:{demo}:{model}:{start}")
        pairs = pc.cross_window_rows(
            demo=demo, model=model, starts=starts, T=clip.T, decoded=decoded,
            fk_windows=np.stack(fk_windows), root_position_error=root_pos_err,
            root_rotation_error_deg=root_rot_err,
            frame_dt=clip.condition.frame_dt)
        pair_rows.extend(pairs)
        fused_v0 = _fused_facts(ctx, att_result, model, model, window_pen,
                                starts, deadline)
        fused_v1 = _fused_facts(ctx, att_result, model, model + "_all_contributors",
                                window_pen, starts, deadline)
        model_facts[model] = {
            "saved_windows": saved["record"],
            "retry": att_result["models"][model].get("retry"),
            "condition_binding": {
                "checked": True,
                "model": model,
                "note": "condition arrays are model-independent; checked once per demo",
            },
            "fused_v0": fused_v0, "fused_v1": fused_v1,
            "window_internal_jumps": pc.window_internal_jumps(
                decoded, starts, clip.condition.frame_dt),
        }
    known = _known_case_windows(reg, demo, att_result, model_facts, starts,
                                clip, deadline) if demo == KNOWN_CASE else None
    result = {
        "demo": demo, "group": group, "T": clip.T, "starts": starts,
        "rows": rows, "teacher_rows": teacher_rows, "pairs": pair_rows,
        "condition_checks": condition_checks, "models": model_facts,
        "known_case": known, "status": "ok",
        "curve_problems": curve_problems,
        "teacher_keys": [r["key"] for r in teacher_rows],
        "pair_keys": [p["key"] for p in pair_rows],
        "window_keys": [r["key"] for r in rows],
    }
    expected_pairs = sum(
        len(pc.cross_window_pair_keys(demo, model, starts, clip.T))
        for model in ("deterministic", "diffusion"))
    if (len(rows) != 2 * len(starts) or len(teacher_rows) != len(starts)
            or len(pair_rows) != expected_pairs
            or any(r.get("status") != "measured" for r in rows)
            or curve_problems
            or not condition_checks["passed"]):
        result["status"] = "incomplete"
    return result


def _known_case_windows(reg, demo, att_result, model_facts, starts, clip,
                        deadline) -> dict:
    out = {"demo": demo, "fixed_frames": list(range(106, 122)),
           "v2_rejection": reg["fusion_binding"]["known_case_v2_rejection"]}
    det = _load_saved_windows(att_result, "deterministic", starts)
    if 90 in starts:
        i = starts.index(90)
        q = det["decoded_quat"][i]
        angle_deg = float(pc.quat_angle_deg(q[26], q[27]))
        out["deterministic_window_90"] = {
            "window_index": i, "local_116_117_deg": angle_deg,
            "local_116_117_radps": float(np.deg2rad(angle_deg) / clip.condition.frame_dt),
            "window_max_step_deg": float(pc.quat_angle_deg(
                q[:-1], q[1:]).max()),
        }
    for model in ("deterministic", "diffusion"):
        temporal = model_facts[model]["fused_v0"]["temporal"]
        step = np.asarray(temporal["root_rotation_step_deg"])
        speed = np.asarray(temporal["root_angular_speed_radps"])
        out[model] = {
            "frame_106_121_step_deg": [float(step[t]) for t in range(106, min(122, len(step)))],
            "frame_107_108_deg": float(step[108]) if len(step) > 108 else None,
            "frame_107_108_radps": float(speed[108]) if len(speed) > 108 else None,
            "seam_rotation_max_deg": temporal["seam_max_step_deg"],
            "global_max_radps": temporal["global_max_radps"],
        }
    return out


def _save_arrays(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def saved_windows_stage(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    cohort = _cohort_history(run, reg["cohort_id"])
    budget = budget_for(run, "saved-windows", cohort)
    attempt = _next_attempt(run, "saved-windows")
    if attempt != "attempt_001":
        raise SystemExit("saved-windows already attempted; a new attempt needs an explicit "
                         "engineering revision and a new registered implementation")
    _preregister_attempt(run, "saved-windows", attempt, t0, cohort=reg["cohort_id"])
    attempt_dir = run / STAGE_DIRS["saved-windows"] / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    report_path = attempt_dir / "summary.json"
    report = {"status": "running", "complete": False,
              "registration": file_record(run / "freeze/execution_registration.json"),
              "units": [], "rows": None, "evidence_gaps": []}
    write_json(report_path, report, exclusive=True)
    units = []
    rows, teacher_rows, pairs = [], [], []
    try:
        for demo in reg["demos"]:
            _deadline(deadline, "saved-windows-unit")
            start = time.perf_counter()
            timeout = min(900.0, deadline - start)
            if timeout <= 5:
                raise TimeoutError("no time for the next saved-window unit")
            out = attempt_dir / "units" / demo
            out.mkdir(parents=True, exist_ok=False)
            argv = [sys.executable, str(Path(__file__)), "saved-window-unit",
                    "--run-root", str(run), "--demo", demo,
                    "--out", str(out), "--deadline", str(start + timeout - 2)]
            env = {**os.environ, **{k: "1" for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")}}
            rec = {"demo": demo, "argv": argv, "status": "running", "wall_s": 0.0}
            units.append(rec)
            try:
                with (attempt_dir / "units" / f"{demo}.log").open("w") as log:
                    proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=timeout)
                rec.update(status="done" if proc.returncode == 0 else "error",
                           returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                rec.update(status="timeout", returncode=None)
            finally:
                rec["wall_s"] = time.perf_counter() - start
                write_json(attempt_dir / "units" / "units.json", units)
            p = out / "result.json"
            result = read_json(p) if p.is_file() else {
                "demo": demo, "status": "incomplete", "rows": [],
                "teacher_rows": [], "pairs": [], "window_keys": [],
                "teacher_keys": [], "pair_keys": [],
            }
            rec["result"] = file_record(p) if p.is_file() else None
            rec["result_status"] = result.get("status")
            report["units"] = units
            write_json(report_path, report)
            print(json.dumps({"demo": demo, "status": result.get("status"),
                              "wall_s": rec["wall_s"]}), flush=True)
        units_results = []
        for demo in reg["demos"]:
            p = attempt_dir / "units" / demo / "result.json"
            if not p.is_file():
                report["evidence_gaps"].append(f"unit_missing:{demo}")
                continue
            result = read_json(p)
            units_results.append(result)
            rows.extend(result.get("rows", []))
            teacher_rows.extend(result.get("teacher_rows", []))
            pairs.extend(result.get("pairs", []))
            if result.get("status") != "ok":
                report["evidence_gaps"].append(f"unit_incomplete:{demo}")
        expected = reg["window_plan"]
        row_keys = [r.get("key") for r in rows]
        teacher_keys = [r.get("key") for r in teacher_rows]
        pair_keys = [p.get("key") for p in pairs]
        if sorted(row_keys) != sorted(expected["window_keys"]):
            report["evidence_gaps"].append("window_key_set_mismatch")
        if sorted(teacher_keys) != sorted(expected["teacher_keys"]):
            report["evidence_gaps"].append("teacher_key_set_mismatch")
        if sorted(pair_keys) != sorted(expected["pair_keys"]):
            report["evidence_gaps"].append("pair_key_set_mismatch")
        if any(r.get("status") != "measured" for r in rows):
            report["evidence_gaps"].append("window_row_not_measured")
        groups = pc.aggregate_window_rows(rows, _plan_for_aggregation(reg))
        pair_summary = pc.summarize_pairs(pairs)
        teacher_inheritance = _teacher_inheritance(rows)
        known = next((u.get("known_case") for u in units_results
                      if u.get("demo") == KNOWN_CASE), None)
        report.update({
            "row_count": len(rows), "teacher_row_count": len(teacher_rows),
            "pair_row_count": len(pairs), "groups": groups,
            "pair_summary": pair_summary,
            "teacher_inheritance": teacher_inheritance,
            "known_case": known,
            "condition_binding": {
                u["demo"]: u.get("condition_checks", {}).get("passed")
                for u in units_results},
            "evidence_gaps": sorted(set(report["evidence_gaps"])),
        })
        write_json(attempt_dir / "rows.json", {"rows": rows, "teacher_rows": teacher_rows})
        write_json(attempt_dir / "pairs.json", {"pairs": pairs})
        report["rows_artifact"] = file_record(attempt_dir / "rows.json")
        report["pairs_artifact"] = file_record(attempt_dir / "pairs.json")
        report["complete"] = not report["evidence_gaps"]
        report["status"] = "done" if report["complete"] else "incomplete"
        report["source_anchors_unchanged"] = not _hash_map(reg["anchors"])
        if not report["source_anchors_unchanged"]:
            report["complete"] = False
            report["status"] = "incomplete"
            report["evidence_gaps"].append("source_anchors_drifted")
        write_json(report_path, report)
    except Exception as exc:
        report.update(status="incomplete", complete=False,
                      evidence_gaps=sorted(set(report["evidence_gaps"]
                                               + [f"stage_exception:{type(exc).__name__}:{exc}"])))
        write_json(report_path, report)
    finally:
        pass
    artifacts = {}
    for path in (attempt_dir / "rows.json", attempt_dir / "pairs.json"):
        if path.is_file():
            artifacts[record_path(path)] = full_sha(path)
    row = {"complete": report.get("complete") is True,
           "status": report.get("status", "incomplete"),
           "report": record_path(report_path), "artifacts": artifacts}
    ok = _finish_stage(run, "saved-windows", attempt, t0, budget, row,
                       {"demos": len(reg["demos"]), "rows": len(rows),
                        "pairs": len(pairs)}, row["status"])
    return ok


def _teacher_inheritance(rows: list[dict]) -> dict:
    shared = new = 0
    model_max = 0.0
    teacher_max = 0.0
    for row in rows:
        g = (row.get("geometry") or {}).get("penetration") or {}
        shared += int(g.get("shared_violation_frames", 0))
        new += int(g.get("new_violation_frames", 0))
        model_max = max(model_max, float(g.get("max_m", 0.0)))
        teacher_max = max(teacher_max, float(g.get("teacher_max_m", 0.0)))
    return {"shared_violation_frames": shared, "new_violation_frames": new,
            "model_penetration_max_m": model_max,
            "teacher_penetration_max_m": teacher_max}


def verify_registration(run: Path) -> dict:
    path = run / "freeze/execution_registration.json"
    reg = read_json(path)
    if reg.get("execution_id") != EXECUTION_ID:
        raise ValueError("wrong prediction-consistency execution")
    if reg.get("noise", {}).get("base_seed") != pc.NOISE_SEED:
        raise ValueError("noise seed drifted")
    if reg.get("t_by_demo") != EXPECTED_T:
        raise ValueError("registered T table drifted")
    if reg.get("starts_by_demo") is None:
        raise ValueError("registered starts missing")
    problems = _hash_map(reg["anchors"])
    if problems:
        raise ValueError(problems)
    if reg.get("implementation") != driver.implementation_identity(ROOT):
        raise ValueError("implementation identity drifted")
    for rel, rec in (reg.get("implementation_snapshot") or {}).get("files", {}).items():
        if full_sha(ROOT / rel) != rec["sha256"]:
            raise ValueError(f"implementation file drifted:{rel}")
    return reg


def _next_attempt(run: Path, stage: str) -> str:
    rows = [r for r in _index(run) if r.get("stage") == stage]
    if not rows:
        return "attempt_001"
    return f"attempt_{int(rows[-1]['attempt'].split('_')[1]) + 1:03d}"


def _needs_retry_guard(run: Path, stage: str) -> None:
    row, problems = stage_state(run, stage)
    if row is not None:
        raise SystemExit(f"{stage} already has an attempt ({row.get('status')}); "
                         f"no automatic retry (problems={problems})")


def _plan_for_aggregation(reg: dict) -> dict:
    plan = dict(reg["window_plan"])
    plan.update(train6=reg["train6"], val6=reg["val6"],
                known_case=reg["known_case"])
    return plan


# ---------------------------------------------------------------------------
# probes stage (P)
# ---------------------------------------------------------------------------

def _capture_session(generator, batch: dict, model: str, seed: int,
                     deadline: float) -> tuple[dict, dict]:
    _deadline(deadline, "session")
    capture = {"normalized": None}
    stats = {"network_forwards": 0, "conditioner_calls": 0}
    if model == "deterministic":
        original = generator.model.forward

        def counted(self, *args, **kwargs):
            stats["network_forwards"] += 1
            out = original(*args, **kwargs)
            capture["normalized"] = out.detach().cpu().numpy().copy()
            return out

        generator.model.forward = MethodType(counted, generator.model)
    else:
        original = generator.model.sample

        def counted(self, *args, **kwargs):
            stats["network_forwards"] += 25  # DDIM-25 exposes 25 forwards
            out = original(*args, **kwargs)
            capture["normalized"] = out.detach().cpu().numpy().copy()
            return out

        generator.model.sample = MethodType(counted, generator.model)
    try:
        stats["conditioner_calls"] += 1
        decoded, _ = generator._sample_windows(batch, seed)
    finally:
        if model == "deterministic":
            generator.model.forward = original
        else:
            generator.model.sample = original
    if capture["normalized"] is None:
        raise ValueError("session output was not captured")
    return {"normalized": capture["normalized"], "decoded": decoded}, stats


def _replay_compare(normalized: np.ndarray, decoded, saved: dict,
                    starts: list[int]) -> dict:
    n_max = float(np.abs(normalized.astype(np.float64)
                         - saved["normalized_prediction"].astype(np.float64)).max())
    d_max = {name: float(np.abs(np.asarray(decoded[i], dtype=np.float64)
                                - np.asarray(saved[key], dtype=np.float64)).max())
             for i, (name, key) in enumerate((("pos", "decoded_pos"),
                                              ("quat", "decoded_quat"),
                                              ("q", "decoded_q")))}
    rot_max = float(pc.quat_angle_deg(
        np.asarray(decoded[1], dtype=np.float64),
        np.asarray(saved["decoded_quat"], dtype=np.float64)).max())
    return {
        "normalized_bitwise": bool(np.array_equal(normalized,
                                                  saved["normalized_prediction"])),
        "normalized_max_abs": n_max,
        "decoded_bitwise": {name: bool(np.array_equal(np.asarray(decoded[i]),
                                                      np.asarray(saved[key])))
                            for i, (name, key) in enumerate((("pos", "decoded_pos"),
                                                             ("quat", "decoded_quat"),
                                                             ("q", "decoded_q")))},
        "decoded_max_abs": d_max,
        "rotation_max_deg": rot_max,
        "tolerance_pass": bool(n_max <= pc.REPLAY_TOL["normalized"]
                               and d_max["pos"] <= pc.REPLAY_TOL["position_m"]
                               and d_max["q"] <= pc.REPLAY_TOL["joints_rad"]
                               and rot_max <= pc.REPLAY_TOL["rotation_deg"]),
    }


def _measure_probe(*, ctx, demo: str, model: str | None, probe: str, start: int,
                   pred_x0: np.ndarray, teacher_x0: np.ndarray,
                   saved: dict | None, index: int | None,
                   model_geometry: dict, teacher_geometry: dict,
                   assembler, deadline: float, group: str,
                   t: int | None = None,
                   alpha: float | None = None, snr: float | None = None,
                   seed: int | None = None) -> dict:
    _deadline(deadline, "probe-measure")
    clip = ctx["clip"]
    decoded = pc.decode_production(pred_x0, ctx["tokenizer"], device=ctx["device"])
    kind = "tail" if pc.is_tail_start(clip.T, start) else "regular"
    out = pc.measure_window(
        demo=demo, model=model or probe, start=int(start), T=clip.T, kind=kind,
        group=group, pred_x0=pred_x0, teacher_x0=teacher_x0, decoded=decoded,
        world_from_clip=clip.condition.world_from_clip,
        teacher_root=clip.root, teacher_joints=clip.q,
        teacher_kpts=ctx["teacher_kpts"], model_geometry=model_geometry,
        teacher_geometry=teacher_geometry, tokenizer=ctx["tokenizer"],
        assembler=assembler, schedule=clip.condition.schedule,
        frame_dt=clip.condition.frame_dt,
        task_tol_m=ctx["w1_reg"]["evaluation"]["task_gate"]["tol_m"],
        hand_gap_m=ctx["validator"].cfg.hand_gap_m,
        saved_world_root=(saved["window_root"][index]
                          if saved is not None and index is not None else None),
        decode_device=ctx["device"])
    row = {
        "key": pc.probe_key(demo, int(start), probe, t),
        "demo": demo, "model": model, "probe": probe, "t": t,
        "start": int(start), "kind": kind, "group": group, "T": clip.T,
        "seed": seed, "alpha_cumprod": alpha, "snr": snr,
        "status": out["status"], "measurement_gaps": out["measurement_gaps"],
        "metrics": out["metrics"],
        "curves": out["curves"],
    }
    if saved is not None and index is not None:
        replay = _replay_compare(
            pred_x0, decoded,
            {"normalized_prediction": saved["normalized_prediction"][index],
             "decoded_pos": saved["decoded_pos"][index],
             "decoded_quat": saved["decoded_quat"][index],
             "decoded_q": saved["decoded_q"][index]}, [start])
        row["replay"] = replay
        if out["metrics"] is not None:
            row["metrics"]["replay"] = replay
    return row


def _failed_probe_row(*, demo: str, start: int, probe: str, t: int | None,
                      group: str, T: int, error: str) -> dict:
    kind = "tail" if pc.is_tail_start(T, start) else "regular"
    return {
        "key": pc.probe_key(demo, int(start), probe, t),
        "demo": demo, "model": None, "probe": probe, "t": t,
        "start": int(start), "kind": kind, "group": group, "T": int(T),
        "seed": None, "alpha_cumprod": None, "snr": None,
        "status": "execution_failed",
        "measurement_gaps": [f"execution_failed:{error}"],
        "metrics": None, "curves": None,
    }


def probe_unit(reg: dict, demo: str, out: Path, deadline: float,
               *, write: bool = True) -> dict:
    if write:
        out.mkdir(parents=True, exist_ok=False)
    ctx = _load_context(reg, demo)
    clip = ctx["clip"]
    starts = pc.inference_starts(clip.T)
    att_result = _attribution_result(demo)
    saved = {m: _load_saved_windows(att_result, m, starts)
             for m in ("deterministic", "diffusion")}
    teacher_x0 = pc.encode_teacher(clip, ctx["tokenizer"])
    teacher_windows = pc.teacher_windows(teacher_x0, starts)
    teacher_geom = {int(s): _teacher_window_geometry(ctx, s, deadline)
                    for s in starts}
    assembler = pc.make_loss_assembler(
        ctx["tokenizer"], ctx["w1_reg"]["configs"]["deterministic"]["loss"],
        ctx["device"])
    group = reg["groups"][demo]
    rows = []
    facts = {"demo": demo, "group": group, "T": clip.T, "starts": starts,
             "first_attempt": {}, "sessions": {}, "rng_unchanged": None}

    for model in ("deterministic", "diffusion"):
        retry = att_result["models"][model].get("retry") or {}
        facts["first_attempt"][model] = {
            "resample_attempts_used": retry.get("resample_attempts_used"),
            "resample_attempts_max": retry.get("resample_attempts_max"),
            "attempt_seeds": retry.get("attempt_seeds"),
            "first_attempt_ok": bool(retry.get("resample_attempts_used") == 1
                                     and retry.get("attempt_seeds") == [SEED]),
        }

    def teacher_geometry(start: int) -> dict:
        return teacher_geom[int(start)]

    # oracle: frozen teacher encode -> production decode, no learning model.
    for i, start in enumerate(starts):
        _deadline(deadline, "oracle")
        try:
            pred = teacher_windows[i]
            decoded = pc.decode_production(pred, ctx["tokenizer"], device=ctx["device"])
            world_root = pc.lift_clip_to_world(
                decoded[0].astype(np.float64), decoded[1].astype(np.float64),
                clip.condition.world_from_clip)
            scene = np.concatenate((world_root, decoded[2],
                                    clip.condition.target_obj_pose[start:start + WINDOW]), 1)
            geom = pc.window_geometry_series(
                ctx["replayer"], scene,
                clip.condition.human_kps_world[start:start + WINDOW],
                clip.condition.target_obj_pose[start:start + WINDOW], deadline=deadline)
            row = _measure_probe(
                ctx=ctx, demo=demo, model=None, probe="teacher_roundtrip",
                start=start, pred_x0=pred, teacher_x0=teacher_windows[i],
                saved=None, index=None, model_geometry=geom,
                teacher_geometry=teacher_geometry(start), assembler=assembler,
                deadline=deadline, group=group)
        except Exception as exc:
            row = _failed_probe_row(
                demo=demo, start=start, probe="teacher_roundtrip", t=None,
                group=group, T=clip.T, error=f"{type(exc).__name__}:{exc}")
        if row.get("metrics") is not None:
            p = row["metrics"]["physical"]
            row["oracle_absolute"] = {
                "position_max_m": p["root_position"]["max_m"],
                "joints_max_rad": p["joints"]["max_abs_rad"],
                "rotation_max_deg": p["root_rotation"]["max_deg"],
                "position_pass": p["root_position"]["max_m"] <= pc.ORACLE_TOL["position_m"],
                "joints_pass": p["joints"]["max_abs_rad"] <= pc.ORACLE_TOL["joints_rad"],
                "rotation_pass": p["root_rotation"]["max_deg"] <= pc.ORACLE_TOL["rotation_deg"],
            }
            row["metrics"]["oracle_absolute"] = row["oracle_absolute"]
        rows.append(row)

    # deterministic session: one full inference batch, one high-level call.
    det = _load_generator(ctx, "deterministic", resample_attempts=1)
    rng_before = _rng_state()
    try:
        det_batch = _window_batch(det._condition_batch(clip.condition), starts)
        det_out, det_stats = _capture_session(det, det_batch, "deterministic", SEED, deadline)
        det_error = None
    except Exception as exc:
        det_out, det_stats = None, {"network_forwards": 0, "conditioner_calls": 0}
        det_error = f"{type(exc).__name__}:{exc}"
    rng_after = _rng_state()
    facts["rng_unchanged"] = _rng_equal(rng_before, rng_after)
    facts["sessions"]["deterministic"] = {
        **det_stats, "model_calls": 1, "windows": len(starts),
        "error": det_error,
        "normalized_sha256": (pc.array_sha256(det_out["normalized"])
                              if det_out is not None else None),
    }
    for i, start in enumerate(starts):
        _deadline(deadline, "deterministic-rows")
        if det_out is None:
            rows.append(_failed_probe_row(
                demo=demo, start=start, probe="deterministic_generation", t=None,
                group=group, T=clip.T, error=det_error or "session_failed"))
            continue
        try:
            world_root = pc.lift_clip_to_world(
                det_out["decoded"][0][i].astype(np.float64),
                det_out["decoded"][1][i].astype(np.float64),
                clip.condition.world_from_clip)
            scene = np.concatenate((world_root, det_out["decoded"][2][i],
                                    clip.condition.target_obj_pose[start:start + WINDOW]), 1)
            geom = pc.window_geometry_series(
                ctx["replayer"], scene,
                clip.condition.human_kps_world[start:start + WINDOW],
                clip.condition.target_obj_pose[start:start + WINDOW], deadline=deadline)
            row = _measure_probe(
                ctx=ctx, demo=demo, model="deterministic",
                probe="deterministic_generation", start=start,
                pred_x0=det_out["normalized"][i], teacher_x0=teacher_windows[i],
                saved=saved["deterministic"], index=i, model_geometry=geom,
                teacher_geometry=teacher_geometry(start), assembler=assembler,
                deadline=deadline, group=group)
        except Exception as exc:
            row = _failed_probe_row(
                demo=demo, start=start, probe="deterministic_generation", t=None,
                group=group, T=clip.T, error=f"{type(exc).__name__}:{exc}")
        rows.append(row)
    del det
    if str(ctx["device"]).startswith("cuda"):
        torch.cuda.empty_cache()

    # denoise: same epsilon for five fixed t, one batch forward per window.
    diff = _load_generator(ctx, "diffusion", resample_attempts=1)
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=ctx["w1_reg"]["configs"]["diffusion"]["num_train_timesteps"],
        beta_schedule=ctx["w1_reg"]["configs"]["diffusion"]["beta_schedule"],
        beta_start=ctx["w1_reg"]["configs"]["diffusion"]["beta_start"],
        beta_end=ctx["w1_reg"]["configs"]["diffusion"]["beta_end"],
        prediction_type="sample", clip_sample=False)
    alpha = scheduler.alphas_cumprod.detach().cpu().numpy()
    snr = alpha / np.maximum(1.0 - alpha, 1e-12)
    denoise_inputs, denoise_outputs = {}, {}
    denoise_forwards = 0
    denoise_errors = {}
    cond = _condition_arrays(clip)
    for i, start in enumerate(starts):
        _deadline(deadline, "denoise")
        try:
            eps, seed = pc.epsilon_for_window(demo, int(start), device=ctx["device"])
            x0 = torch.as_tensor(teacher_windows[i], dtype=torch.float32,
                                 device=ctx["device"])
            x0_rep = x0[None].repeat(len(pc.DENOISE_TIMESTEPS), 1, 1)
            eps_rep = eps[None].repeat(len(pc.DENOISE_TIMESTEPS), 1, 1)
            t = torch.tensor(pc.DENOISE_TIMESTEPS, dtype=torch.long,
                             device=ctx["device"])
            with torch.no_grad():
                x_t = diff.model.add_noise(x0_rep, eps_rep, t)
                cond5 = {k: torch.as_tensor(
                    np.repeat(v[None, start:start + WINDOW],
                              len(pc.DENOISE_TIMESTEPS), 0),
                    dtype=torch.float32, device=ctx["device"])
                    for k, v in cond.items()}
                memory = diff.conditioner(cond5)
                pred = diff.model(x_t, t, memory)
            denoise_forwards += 1
            pred_np = pred.detach().cpu().numpy()
            x_t_np = x_t.detach().cpu().numpy()
            denoise_inputs[str(start)] = {
                "epsilon": eps.detach().cpu().numpy(), "x_t": x_t_np,
                "seed": seed, "t": np.asarray(pc.DENOISE_TIMESTEPS, dtype=np.int64),
                "alpha_cumprod": alpha[list(pc.DENOISE_TIMESTEPS)],
                "snr": snr[list(pc.DENOISE_TIMESTEPS)],
                "epsilon_sha256": pc.array_sha256(eps.detach().cpu().numpy()),
                "x_t_sha256": pc.array_sha256(x_t_np),
            }
            denoise_outputs[str(start)] = pred_np
        except Exception as exc:
            denoise_errors[start] = f"{type(exc).__name__}:{exc}"
            for tt in pc.DENOISE_TIMESTEPS:
                rows.append(_failed_probe_row(
                    demo=demo, start=start, probe="diffusion_denoise", t=int(tt),
                    group=group, T=clip.T, error=denoise_errors[start]))
            continue
        for ti, tt in enumerate(pc.DENOISE_TIMESTEPS):
            try:
                dec = pc.decode_production(pred_np[ti], ctx["tokenizer"],
                                           device=ctx["device"])
                world_root = pc.lift_clip_to_world(
                    dec[0].astype(np.float64), dec[1].astype(np.float64),
                    clip.condition.world_from_clip)
                scene = np.concatenate((world_root, dec[2],
                                        clip.condition.target_obj_pose[start:start + WINDOW]), 1)
                geom = pc.window_geometry_series(
                    ctx["replayer"], scene,
                    clip.condition.human_kps_world[start:start + WINDOW],
                    clip.condition.target_obj_pose[start:start + WINDOW],
                    deadline=deadline)
                row = _measure_probe(
                    ctx=ctx, demo=demo, model="diffusion",
                    probe="diffusion_denoise", start=start, t=int(tt),
                    pred_x0=pred_np[ti], teacher_x0=teacher_windows[i],
                    saved=None, index=None, model_geometry=geom,
                    teacher_geometry=teacher_geometry(start), assembler=assembler,
                    deadline=deadline, group=group, alpha=float(alpha[tt]),
                    snr=float(snr[tt]), seed=seed)
            except Exception as exc:
                row = _failed_probe_row(
                    demo=demo, start=start, probe="diffusion_denoise", t=int(tt),
                    group=group, T=clip.T, error=f"{type(exc).__name__}:{exc}")
            rows.append(row)
    facts["sessions"]["diffusion_denoise"] = {
        "windows": len(starts), "network_forwards": denoise_forwards,
        "conditioner_calls": len(starts), "same_epsilon_per_window": True,
        "timesteps": list(pc.DENOISE_TIMESTEPS),
        "failed_windows": denoise_errors,
    }

    # DDIM-25 session: full batch layout and seed 7.
    rng_before = _rng_state()
    try:
        ddim_batch = _window_batch(diff._condition_batch(clip.condition), starts)
        ddim_out, ddim_stats = _capture_session(diff, ddim_batch, "diffusion",
                                                SEED, deadline)
        ddim_error = None
    except Exception as exc:
        ddim_out, ddim_stats = None, {"network_forwards": 0,
                                      "conditioner_calls": 0}
        ddim_error = f"{type(exc).__name__}:{exc}"
    rng_after = _rng_state()
    facts["rng_unchanged"] = bool(facts["rng_unchanged"] is not False
                                  and _rng_equal(rng_before, rng_after))
    facts["sessions"]["diffusion_ddim25"] = {
        **ddim_stats, "model_calls": 1, "windows": len(starts),
        "error": ddim_error,
        "normalized_sha256": (pc.array_sha256(ddim_out["normalized"])
                              if ddim_out is not None else None),
    }
    for i, start in enumerate(starts):
        _deadline(deadline, "ddim-rows")
        if ddim_out is None:
            rows.append(_failed_probe_row(
                demo=demo, start=start, probe="diffusion_ddim25", t=None,
                group=group, T=clip.T, error=ddim_error or "session_failed"))
            continue
        try:
            world_root = pc.lift_clip_to_world(
                ddim_out["decoded"][0][i].astype(np.float64),
                ddim_out["decoded"][1][i].astype(np.float64),
                clip.condition.world_from_clip)
            scene = np.concatenate((world_root, ddim_out["decoded"][2][i],
                                    clip.condition.target_obj_pose[start:start + WINDOW]), 1)
            geom = pc.window_geometry_series(
                ctx["replayer"], scene,
                clip.condition.human_kps_world[start:start + WINDOW],
                clip.condition.target_obj_pose[start:start + WINDOW], deadline=deadline)
            row = _measure_probe(
                ctx=ctx, demo=demo, model="diffusion", probe="diffusion_ddim25",
                start=start, pred_x0=ddim_out["normalized"][i],
                teacher_x0=teacher_windows[i], saved=saved["diffusion"], index=i,
                model_geometry=geom, teacher_geometry=teacher_geometry(start),
                assembler=assembler, deadline=deadline, group=group, seed=SEED)
        except Exception as exc:
            row = _failed_probe_row(
                demo=demo, start=start, probe="diffusion_ddim25", t=None,
                group=group, T=clip.T, error=f"{type(exc).__name__}:{exc}")
        rows.append(row)
    del diff
    if str(ctx["device"]).startswith("cuda"):
        torch.cuda.empty_cache()

    expected = ([pc.probe_key(demo, s, "teacher_roundtrip") for s in starts]
                + [pc.probe_key(demo, s, "deterministic_generation") for s in starts]
                + [pc.probe_key(demo, s, "diffusion_denoise", tt)
                   for s in starts for tt in pc.DENOISE_TIMESTEPS]
                + [pc.probe_key(demo, s, "diffusion_ddim25") for s in starts])
    if [r["key"] for r in rows] != expected:
        raise ValueError("probe unit key order/set changed")
    result = {"demo": demo, "group": group, "T": clip.T, "starts": starts,
              "rows": rows, "facts": facts, "status": "ok"}
    if any(r.get("status") != "measured" for r in rows):
        result["status"] = "incomplete"
    if write:
        input_arrays = {
            "starts": np.asarray(starts, dtype=np.int64),
            "teacher_x0": teacher_windows,
            **{f"condition_{k}": np.stack(
                [np.asarray(v)[s:s + WINDOW] for s in starts])
               for k, v in cond.items()},
            **{f"denoise_{key}_{name}": value
               for key, block in denoise_inputs.items()
               for name, value in block.items()},
        }
        np.savez_compressed(out / "inputs.npz", **input_arrays)
        output_arrays = {"starts": np.asarray(starts, dtype=np.int64)}
        if det_out is not None:
            output_arrays["det_normalized"] = det_out["normalized"]
        if ddim_out is not None:
            output_arrays["ddim_normalized"] = ddim_out["normalized"]
        output_arrays.update({f"denoise_{key}": value
                              for key, value in denoise_outputs.items()})
        np.savez_compressed(out / "outputs.npz", **output_arrays)
        for row in rows:
            if row.get("curves") is None:
                continue
            _save_arrays(
                out / "curves" / f"{row['key'].replace('|', '__')}.npz",
                row["curves"])
            row.pop("curves", None)
        write_json(out / "result.json", result)
    return result


def probes_stage(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    cohort = _cohort_history(run, reg["cohort_id"])
    budget = budget_for(run, "probes", cohort)
    attempt = _next_attempt(run, "probes")
    if attempt != "attempt_001":
        raise SystemExit("probes already attempted; no automatic retry")
    _preregister_attempt(run, "probes", attempt, t0, cohort=reg["cohort_id"])
    attempt_dir = run / STAGE_DIRS["probes"] / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "units").mkdir(parents=True, exist_ok=True)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    report_path = attempt_dir / "summary.json"
    report = {"status": "running", "complete": False,
              "registration": file_record(run / "freeze/execution_registration.json"),
              "units": [], "evidence_gaps": []}
    write_json(report_path, report, exclusive=True)
    units = []
    rows = []
    try:
        for demo in reg["demos"]:
            _deadline(deadline, "probe-unit")
            start = time.perf_counter()
            timeout = min(1200.0, deadline - start)
            if timeout <= 5:
                raise TimeoutError("no time for the next probe unit")
            out = attempt_dir / "units" / demo
            argv = [sys.executable, str(Path(__file__)), "probe-unit",
                    "--run-root", str(run), "--demo", demo,
                    "--out", str(out), "--deadline", str(start + timeout - 2)]
            env = {**os.environ, **{k: "1" for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")}}
            rec = {"demo": demo, "argv": argv, "status": "running", "wall_s": 0.0}
            try:
                with (attempt_dir / "units" / f"{demo}.log").open("w") as log:
                    proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=timeout)
                rec.update(status="done" if proc.returncode == 0 else "error",
                           returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                rec.update(status="timeout", returncode=None)
            finally:
                rec["wall_s"] = time.perf_counter() - start
                units.append(rec)
                write_json(attempt_dir / "units" / "units.json", units)
            p = out / "result.json"
            result = read_json(p) if p.is_file() else {
                "demo": demo, "status": "incomplete", "rows": [], "facts": {}}
            rec["result_status"] = result.get("status")
            if result.get("status") != "ok":
                report["evidence_gaps"].append(f"unit_incomplete:{demo}")
            rows.extend(result.get("rows", []))
            report["units"] = units
            write_json(report_path, report)
            print(json.dumps({"demo": demo, "status": result.get("status"),
                              "wall_s": rec["wall_s"]}), flush=True)
        expected = reg["window_plan"]["probe_keys"]
        keys = [r.get("key") for r in rows]
        if sorted(keys) != sorted(expected):
            report["evidence_gaps"].append("probe_key_set_mismatch")
        if any(r.get("status") != "measured" for r in rows):
            report["evidence_gaps"].append("probe_row_not_measured")
        sessions = {"det_sessions": 0, "denoise_batches": 0, "ddim_sessions": 0,
                    "network_forwards": 0, "conditioner_calls": 0}
        rng_ok = True
        first_attempt = {}
        for demo in reg["demos"]:
            p = attempt_dir / "units" / demo / "result.json"
            if not p.is_file():
                continue
            result = read_json(p)
            facts = result.get("facts", {})
            first_attempt[demo] = facts.get("first_attempt")
            rng_ok = rng_ok and facts.get("rng_unchanged") is True
            s = facts.get("sessions", {})
            sessions["det_sessions"] += 1
            sessions["denoise_batches"] += int(
                (s.get("diffusion_denoise") or {}).get("windows", 0))
            sessions["ddim_sessions"] += 1
            sessions["network_forwards"] += int(
                (s.get("deterministic") or {}).get("network_forwards", 0))
            sessions["network_forwards"] += int(
                (s.get("diffusion_denoise") or {}).get("network_forwards", 0))
            sessions["network_forwards"] += int(
                (s.get("diffusion_ddim25") or {}).get("network_forwards", 0))
            sessions["conditioner_calls"] += int(
                (s.get("deterministic") or {}).get("conditioner_calls", 0))
            sessions["conditioner_calls"] += int(
                (s.get("diffusion_denoise") or {}).get("conditioner_calls", 0))
            sessions["conditioner_calls"] += int(
                (s.get("diffusion_ddim25") or {}).get("conditioner_calls", 0))
        sessions["high_level_calls"] = (sessions["det_sessions"]
                                        + sessions["denoise_batches"]
                                        + sessions["ddim_sessions"])
        expected_counts = pc.expected_ratio_counts()
        if sessions["high_level_calls"] != expected_counts["high_level_calls"]:
            report["evidence_gaps"].append("high_level_call_count")
        if sessions["network_forwards"] != expected_counts["network_forwards"]:
            report["evidence_gaps"].append("network_forward_count")
        if not rng_ok:
            report["evidence_gaps"].append("global_rng_changed")
        replay = _replay_summary(rows)
        groups = pc.aggregate_probe_rows(rows, _plan_for_aggregation(reg))
        snr = pc.denoise_snr_table(rows)
        oracle = _oracle_summary(rows)
        report.update({
            "row_count": len(rows), "sessions": sessions,
            "first_attempt": first_attempt, "replay": replay,
            "groups": groups, "denoise_snr": snr, "oracle": oracle,
            "evidence_gaps": sorted(set(report["evidence_gaps"])),
        })
        write_json(attempt_dir / "rows.json", {"rows": rows})
        report["rows_artifact"] = file_record(attempt_dir / "rows.json")
        report["complete"] = not report["evidence_gaps"]
        report["status"] = "done" if report["complete"] else "incomplete"
        report["source_anchors_unchanged"] = not _hash_map(reg["anchors"])
        if not report["source_anchors_unchanged"]:
            report["complete"] = False
            report["status"] = "incomplete"
            report["evidence_gaps"].append("source_anchors_drifted")
        write_json(report_path, report)
    except Exception as exc:
        report.update(status="incomplete", complete=False,
                      evidence_gaps=sorted(set(report["evidence_gaps"]
                                               + [f"stage_exception:{type(exc).__name__}:{exc}"])))
        write_json(report_path, report)
    artifacts = {}
    if (attempt_dir / "rows.json").is_file():
        artifacts[record_path(attempt_dir / "rows.json")] = full_sha(
            attempt_dir / "rows.json")
    row = {"complete": report.get("complete") is True,
           "status": report.get("status", "incomplete"),
           "report": record_path(report_path), "artifacts": artifacts}
    ok = _finish_stage(run, "probes", attempt, t0, budget, row,
                       {"demos": len(reg["demos"]), "rows": len(rows)},
                       row["status"])
    return ok


def _replay_summary(rows: list[dict]) -> dict:
    out = {"deterministic": {"windows": 0, "bitwise": 0, "tolerance": 0,
                             "max_normalized": 0.0, "max_position": 0.0,
                             "max_joints": 0.0, "max_rotation": 0.0},
           "diffusion": {"windows": 0, "bitwise": 0, "tolerance": 0,
                         "max_normalized": 0.0, "max_position": 0.0,
                         "max_joints": 0.0, "max_rotation": 0.0}}
    for row in rows:
        if row.get("probe") not in ("deterministic_generation", "diffusion_ddim25"):
            continue
        key = "deterministic" if row["probe"] == "deterministic_generation" else "diffusion"
        rep = row.get("replay")
        if rep is None:
            continue
        out[key]["windows"] += 1
        out[key]["bitwise"] += int(rep["normalized_bitwise"]
                                   and all(rep["decoded_bitwise"].values()))
        out[key]["tolerance"] += int(rep["tolerance_pass"])
        out[key]["max_normalized"] = max(out[key]["max_normalized"],
                                         rep["normalized_max_abs"])
        out[key]["max_position"] = max(out[key]["max_position"],
                                       rep["decoded_max_abs"]["pos"])
        out[key]["max_joints"] = max(out[key]["max_joints"],
                                     rep["decoded_max_abs"]["q"])
        out[key]["max_rotation"] = max(out[key]["max_rotation"],
                                       rep["rotation_max_deg"])
    return out


def _oracle_summary(rows: list[dict]) -> dict:
    oracle = [r for r in rows if r.get("probe") == "teacher_roundtrip"
              and r.get("oracle_absolute")]
    if not oracle:
        return {"n": 0, "all_passed": None}
    return {
        "n": len(oracle),
        "all_passed": all(r["oracle_absolute"]["position_pass"]
                          and r["oracle_absolute"]["joints_pass"]
                          and r["oracle_absolute"]["rotation_pass"] for r in oracle),
        "failed": [r["key"] for r in oracle
                   if not (r["oracle_absolute"]["position_pass"]
                           and r["oracle_absolute"]["joints_pass"]
                           and r["oracle_absolute"]["rotation_pass"])],
        "max_position_m": max(r["oracle_absolute"]["position_max_m"] for r in oracle),
        "max_joints_rad": max(r["oracle_absolute"]["joints_max_rad"] for r in oracle),
        "max_rotation_deg": max(r["oracle_absolute"]["rotation_max_deg"] for r in oracle),
        "fk_mean_m": float(np.mean([
            r["metrics"]["physical"]["fk"]["mpjpe_m"] for r in oracle])),
    }


def _rng_state() -> dict:
    cuda = []
    if torch.cuda.is_available():
        cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return {"cpu": torch.get_rng_state().clone(), "cuda": cuda}


def _rng_equal(a: dict, b: dict) -> bool:
    if not torch.equal(a["cpu"], b["cpu"]):
        return False
    if len(a["cuda"]) != len(b["cuda"]):
        return False
    return all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))


# ---------------------------------------------------------------------------
# finalize stage (F)
# ---------------------------------------------------------------------------

def finalize_stage(run: Path) -> bool:
    t0 = time.perf_counter()
    reg = verify_registration(run)
    for stage in ("register", "saved-windows", "probes"):
        row, problems = stage_state(run, stage)
        if row is None or row.get("complete") is not True or problems:
            raise SystemExit(f"cannot finalize: {stage} not complete ({row}, {problems})")
    cohort = _cohort_history(run, reg["cohort_id"])
    budget = budget_for(run, "finalize", cohort)
    attempt = _next_attempt(run, "finalize")
    if attempt != "attempt_001":
        raise SystemExit("finalize already attempted; no automatic retry")
    _preregister_attempt(run, "finalize", attempt, t0, cohort=reg["cohort_id"])
    attempt_dir = run / STAGE_DIRS["finalize"] / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    final_path = attempt_dir / "prediction_consistency_final.json"
    report = {"schema": "w1_prediction_consistency_final_v1",
              "status": "running", "complete": False,
              "registration": file_record(run / "freeze/execution_registration.json"),
              "failures": [], "evidence_gaps": []}
    write_json(final_path, report, exclusive=True)
    try:
        w_dir = run / "saved_windows/attempt_001"
        p_dir = run / "probes/attempt_001"
        w_rows = read_json(w_dir / "rows.json")
        p_rows = read_json(p_dir / "rows.json")
        w_summary = read_json(w_dir / "summary.json")
        p_summary = read_json(p_dir / "summary.json")
        failures = []
        gaps = []
        expected = reg["window_plan"]
        w_keys = [r.get("key") for r in w_rows["rows"]]
        t_keys = [r.get("key") for r in w_rows["teacher_rows"]]
        p_keys = [r.get("key") for r in p_rows["rows"]]
        if sorted(w_keys) != sorted(expected["window_keys"]):
            failures.append("w_window_key_set_mismatch")
        if sorted(t_keys) != sorted(expected["teacher_keys"]):
            failures.append("w_teacher_key_set_mismatch")
        if sorted(p_keys) != sorted(expected["probe_keys"]):
            failures.append("p_probe_key_set_mismatch")
        # Recompute W from the source windows; no model calls.
        recomputed_w = []
        for demo in reg["demos"]:
            _deadline(deadline, "f-windows")
            got = recompute_saved_windows(
                reg, demo, deadline=deadline,
                curve_root=w_dir / "units" / demo / "curves")
            recomputed_w.append(got)
            if got.get("status") != "ok":
                failures.append(f"w_recompute_incomplete:{demo}")
            failures += got.get("curve_problems", [])
            recorded = [r for r in w_rows["rows"] if r.get("demo") == demo]
            recorded_t = [r for r in w_rows["teacher_rows"] if r.get("demo") == demo]
            failures += _compare_rows(f"w:{demo}", recorded, got["rows"])
            failures += _compare_rows(f"teacher:{demo}", recorded_t, got["teacher_rows"])
            pairs_recorded = [p for p in read_json(w_dir / "pairs.json")["pairs"]
                              if p.get("demo") == demo]
            if [p.get("key") for p in pairs_recorded] != [p.get("key") for p in got["pairs"]]:
                failures.append(f"w_pair_keys:{demo}")
            for a, b in zip(pairs_recorded, got["pairs"]):
                failures += _compare_pair(a, b, f"pair:{demo}:{a.get('global_frame')}")
        # Recompute P metrics from stored inputs/outputs; verify noise replay.
        for demo in reg["demos"]:
            _deadline(deadline, "f-probes")
            failures += _recompute_probe_artifacts(reg, demo, p_dir / "units" / demo,
                                                   p_rows["rows"], deadline)
        # Fixed independent model replay: three demos, full P matrix.
        replay_dir = attempt_dir / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay_rows = []
        for demo in REPLAY_DEMOS:
            _deadline(deadline, "f-model-replay")
            result = probe_unit(reg, demo, replay_dir / demo, deadline, write=True)
            replay_rows.extend(result["rows"])
            recorded = [r for r in p_rows["rows"] if r.get("demo") == demo]
            failures += _compare_probe_rows(f"replay:{demo}", recorded,
                                            result["rows"], strict_replay=True)
        replay_counts = _replay_call_counts(replay_rows)
        if replay_counts["units"] != 88 or replay_counts["learned"] != 77 \
                or replay_counts["oracle"] != 11:
            failures.append("f_replay_matrix_size")
        # Synthetic/interface tests are part of the implementation contract.
        tests = _run_synthetic_tests(deadline)
        if tests.get("returncode") != 0:
            failures.append("implementation_synthetic_tests_failed")
        # Three-way summary recomputation.
        recomputed_window_rows = [row for unit in recomputed_w
                                  for row in unit["rows"]]
        groups = pc.aggregate_window_rows(recomputed_window_rows,
                                          _plan_for_aggregation(reg))
        pair_summary = pc.summarize_pairs(read_json(w_dir / "pairs.json")["pairs"])
        if pair_summary["n"] != expected["counts"]["cross_window_pairs"]:
            failures.append("pair_denominator")
        teacher_inheritance = _teacher_inheritance(recomputed_window_rows)
        known_case = next((unit.get("known_case") for unit in recomputed_w
                           if unit.get("demo") == KNOWN_CASE), None)
        # Summary files are claims, not evidence: compare every aggregate with
        # the independently recomputed rows, including the known-case facts.
        for label, recorded, recomputed in (
                ("w_groups", w_summary.get("groups"), groups),
                ("w_pair_summary", w_summary.get("pair_summary"), pair_summary),
                ("w_teacher_inheritance", w_summary.get("teacher_inheritance"),
                 teacher_inheritance),
                ("w_known_case", w_summary.get("known_case"), known_case),
                ("p_groups", p_summary.get("groups"),
                 pc.aggregate_probe_rows(p_rows["rows"], _plan_for_aggregation(reg))),
                ("p_denoise_snr", p_summary.get("denoise_snr"),
                 pc.denoise_snr_table(p_rows["rows"])),
                ("p_replay", p_summary.get("replay"), _replay_summary(p_rows["rows"])),
                ("p_oracle", p_summary.get("oracle"), _oracle_summary(p_rows["rows"]))):
            failures += _compare_metric_docs(recorded, recomputed, label)
        if w_summary.get("row_count") != len(w_rows["rows"]):
            failures.append("w_row_count_not_reproducible")
        if w_summary.get("teacher_row_count") != len(w_rows["teacher_rows"]):
            failures.append("w_teacher_row_count_not_reproducible")
        if w_summary.get("pair_row_count") != pair_summary["n"]:
            failures.append("w_pair_row_count_not_reproducible")
        if p_summary.get("row_count") != len(p_rows["rows"]):
            failures.append("p_row_count_not_reproducible")
        if p_summary.get("sessions") != _session_counts(p_dir / "units"):
            failures.append("p_session_counts_not_reproducible")
        oracle = _oracle_summary(p_rows["rows"])
        judgement = pc.attribution_judgement(
            w_summary={"groups": w_summary.get("groups"),
                       "pair_summary": pair_summary,
                       "teacher_inheritance": w_summary.get("teacher_inheritance")},
            p_summary={"denoise_snr": p_summary.get("denoise_snr")},
            oracle=oracle, evidence_gaps=failures + gaps)
        source_ok = not _hash_map(reg["anchors"])
        if not source_ok:
            failures.append("source_anchors_drifted")
        implementation_ok = tests.get("returncode") == 0 and not failures
        complete = not failures and not gaps and w_summary.get("complete") is True \
            and p_summary.get("complete") is True and source_ok
        report.update({
            "implementation_contract_passed": implementation_ok,
            "complete_evidence": complete,
            "diagnosis_complete": bool(implementation_ok and complete
                                       and not failures and not gaps),
            "attribution_status": judgement["status"],
            "attribution": judgement,
            "failures": sorted(set(failures)),
            "evidence_gaps": sorted(set(gaps)),
            "window_rows": len(w_rows["rows"]),
            "teacher_rows": len(w_rows["teacher_rows"]),
            "probe_rows": len(p_rows["rows"]),
            "pair_rows": pair_summary["n"],
            "groups": groups,
            "pair_summary": pair_summary,
            "oracle": oracle,
            "denoise_snr": p_summary.get("denoise_snr"),
            "replay": p_summary.get("replay"),
            "replay_counts": replay_counts,
            "synthetic_tests": tests,
            "known_case": w_summary.get("known_case"),
            "final_interaction_method": "not_ready",
            "training_steps": 0,
            "production_integration": False,
            "historical_anchors": {"u0": "incomplete", "ur": "no_quality_gain",
                                   "w1_final": "not_ready",
                                   "fusion_r10": "not_ready"},
            "next_action": _next_action(judgement, failures, gaps),
            "auto_execute": False,
        })
        report["complete"] = bool(report["diagnosis_complete"])
        report["status"] = "done" if report["complete"] else "incomplete"
        write_json(final_path, report)
        _write_tables(attempt_dir, w_rows, p_rows, report)
    except Exception as exc:
        report.update(status="incomplete", complete=False,
                      failures=sorted(set(report.get("failures", [])
                                          + [f"stage_exception:{type(exc).__name__}:{exc}"])))
        write_json(final_path, report)
    row = {"complete": report.get("complete") is True,
           "status": report.get("status", "incomplete"),
           "report": record_path(final_path),
           "artifacts": {record_path(final_path): full_sha(final_path)}}
    ok = _finish_stage(run, "finalize", attempt, t0, budget, row,
                       {"diagnosis_complete": report.get("diagnosis_complete"),
                        "attribution_status": report.get("attribution_status")},
                       row["status"])
    return ok


def _compare_rows(label: str, recorded: list[dict], recomputed: list[dict]) -> list[str]:
    problems = []
    if [r.get("key") for r in recorded] != [r.get("key") for r in recomputed]:
        problems.append(f"{label}:key_order")
        return problems
    for a, b in zip(recorded, recomputed):
        if a.get("status") != b.get("status"):
            problems.append(f"{label}:status:{a.get('key')}")
            continue
        if a.get("status") != "measured":
            continue
        problems += _compare_metric_docs(a, b, f"{label}:{a.get('key')}")
    return problems


def _compare_metric_docs(a, b, label: str) -> list[str]:
    problems = []
    if not isinstance(a, dict) or not isinstance(b, dict):
        return [f"{label}:not_mapping"]
    if set(a) != set(b):
        problems.append(f"{label}:key_set")
        return problems
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, dict) or isinstance(y, dict):
            if not isinstance(x, dict) or not isinstance(y, dict):
                problems.append(f"{label}:{key}:type")
            else:
                problems += _compare_metric_docs(x, y, f"{label}:{key}")
        elif isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)) \
                    or len(x) != len(y):
                problems.append(f"{label}:{key}:list")
                continue
            for i, (u, v) in enumerate(zip(x, y)):
                problems += _compare_metric_docs({str(i): u}, {str(i): v},
                                                 f"{label}:{key}")
        elif isinstance(x, bool) or isinstance(y, bool):
            if x is not y:
                problems.append(f"{label}:{key}:bool")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if isinstance(x, int) and isinstance(y, int) and not isinstance(x, bool):
                if int(x) != int(y):
                    problems.append(f"{label}:{key}:int")
            elif not math.isfinite(float(x)) or not math.isfinite(float(y)):
                if not (isinstance(x, float) and isinstance(y, float)
                        and math.isnan(x) and math.isnan(y)):
                    problems.append(f"{label}:{key}:nonfinite")
            else:
                tol = 1e-6 if ("m" in key or "rad" in key or "deg" in key) else 1e-9
                if abs(float(x) - float(y)) > tol:
                    problems.append(f"{label}:{key}:float")
        else:
            if x != y:
                problems.append(f"{label}:{key}:value")
    return problems


def _compare_curve_file(path: Path, recomputed: dict, label: str) -> list[str]:
    """A curve artifact is evidence only if it re-verifies and re-derives exactly."""
    if not path.is_file():
        return [f"{label}:missing"]
    try:
        with np.load(path, allow_pickle=False) as z:
            saved = {k: np.array(z[k], copy=True) for k in z.files}
    except (OSError, ValueError) as exc:
        return [f"{label}:unreadable:{type(exc).__name__}"]
    problems = [f"{label}:{p}" for p in pc.verify_window_curves(saved)]
    if set(saved) != set(recomputed):
        problems.append(f"{label}:key_set")
        return problems
    for key, value in recomputed.items():
        a = np.asarray(saved[key])
        b = np.asarray(value)
        if a.shape != b.shape:
            problems.append(f"{label}:{key}:shape")
            continue
        if a.dtype != b.dtype:
            problems.append(f"{label}:{key}:dtype")
            continue
        if a.dtype.kind in "fc":
            if not np.array_equal(a, b, equal_nan=True):
                problems.append(f"{label}:{key}:value")
        elif not np.array_equal(a, b):
            problems.append(f"{label}:{key}:value")
    return problems


def _compare_pair(a: dict, b: dict, label: str) -> list[str]:
    problems = []
    if set(a) != set(b):
        return [f"{label}:keys"]
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and not isinstance(x, bool) and not isinstance(y, bool):
            if abs(float(x) - float(y)) > 1e-6:
                problems.append(f"{label}:{key}:float")
        elif x != y:
            problems.append(f"{label}:{key}:value")
    return problems


def _compare_probe_rows(label: str, recorded: list[dict], replay: list[dict],
                        *, strict_replay: bool = False) -> list[str]:
    problems = []
    if [r.get("key") for r in recorded] != [r.get("key") for r in replay]:
        return [f"{label}:key_order"]
    for a, b in zip(recorded, replay):
        if a.get("status") != b.get("status"):
            problems.append(f"{label}:status:{a.get('key')}")
        if a.get("metrics") is not None and b.get("metrics") is not None:
            problems += _compare_metric_docs(a["metrics"], b["metrics"],
                                             f"{label}:{a.get('key')}")
        if strict_replay and a.get("probe") in ("deterministic_generation",
                                                "diffusion_ddim25"):
            ra, rb = a.get("replay") or {}, b.get("replay") or {}
            if not rb.get("tolerance_pass"):
                problems.append(f"{label}:replay_tolerance:{a.get('key')}")
    return problems


def _recompute_probe_artifacts(reg: dict, demo: str, unit: Path,
                               all_rows: list[dict], deadline: float) -> list[str]:
    """Recompute P metrics and noise inputs from the stored artifacts."""
    problems = []
    ctx = _load_context(reg, demo)
    clip = ctx["clip"]
    starts = pc.inference_starts(clip.T)
    with np.load(unit / "inputs.npz", allow_pickle=False) as z:
        inputs = {k: np.array(z[k], copy=True) for k in z.files}
    with np.load(unit / "outputs.npz", allow_pickle=False) as z:
        outputs = {k: np.array(z[k], copy=True) for k in z.files}
    teacher_x0 = inputs["teacher_x0"]
    teacher_windows = pc.teacher_windows(pc.encode_teacher(clip, ctx["tokenizer"]),
                                         starts)
    if not np.array_equal(teacher_x0, teacher_windows):
        problems.append(f"f_inputs:{demo}:teacher_x0")
    from diffusers import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=ctx["w1_reg"]["configs"]["diffusion"]["num_train_timesteps"],
        beta_schedule=ctx["w1_reg"]["configs"]["diffusion"]["beta_schedule"],
        beta_start=ctx["w1_reg"]["configs"]["diffusion"]["beta_start"],
        beta_end=ctx["w1_reg"]["configs"]["diffusion"]["beta_end"],
        prediction_type="sample", clip_sample=False)
    for i, start in enumerate(starts):
        block = {k.split(f"denoise_{start}_", 1)[1]: v
                 for k, v in inputs.items() if k.startswith(f"denoise_{start}_")}
        if not block:
            problems.append(f"f_inputs:{demo}:{start}:denoise_missing")
            continue
        seed = int(block["seed"])
        if seed != pc.noise_seed_for_window(demo, start):
            problems.append(f"f_inputs:{demo}:{start}:seed")
        eps, expected_seed = pc.epsilon_for_window(demo, start, device="cpu")
        if seed != expected_seed:
            problems.append(f"f_inputs:{demo}:{start}:seed_regeneration")
        if pc.array_sha256(eps.numpy()) != block["epsilon_sha256"]:
            problems.append(f"f_inputs:{demo}:{start}:epsilon_hash")
        x0 = torch.as_tensor(teacher_windows[i], dtype=torch.float32)
        eps_rep = torch.as_tensor(block["epsilon"])[None].repeat(5, 1, 1)
        x0_rep = x0[None].repeat(5, 1, 1)
        t = torch.tensor(block["t"], dtype=torch.long)
        x_t = scheduler.add_noise(x0_rep, eps_rep, t).numpy()
        if pc.array_sha256(x_t) != block["x_t_sha256"]:
            problems.append(f"f_inputs:{demo}:{start}:x_t_hash")
        if not np.array_equal(x_t, block["x_t"]):
            problems.append(f"f_inputs:{demo}:{start}:x_t")
        alpha = scheduler.alphas_cumprod.numpy()[block["t"]]
        if not np.allclose(alpha, block["alpha_cumprod"], rtol=0, atol=0):
            problems.append(f"f_inputs:{demo}:{start}:alpha")
    cond = _condition_arrays(clip)
    for key in pc.CONDITION_KEYS:
        expected = np.stack([np.asarray(cond[key])[s:s + WINDOW] for s in starts])
        if not np.array_equal(inputs.get(f"condition_{key}", np.empty(0)), expected):
            problems.append(f"f_inputs:{demo}:condition_{key}")
    # Metrics recompute from outputs with the W/P measurement functions.
    teacher_geom = {int(s): _teacher_window_geometry(ctx, s, deadline) for s in starts}
    assembler = pc.make_loss_assembler(
        ctx["tokenizer"], ctx["w1_reg"]["configs"]["deterministic"]["loss"],
        ctx["device"])
    recorded = {r["key"]: r for r in all_rows if r.get("demo") == demo}
    att_result = _attribution_result(demo)
    saved_windows = {m: _load_saved_windows(att_result, m, starts)
                     for m in ("deterministic", "diffusion")}

    def measure(pred: np.ndarray, probe: str, start: int, index: int,
                group: str, saved_model: str | None) -> dict:
        dec = pc.decode_production(pred, ctx["tokenizer"], device=ctx["device"])
        scene = np.concatenate((
            pc.lift_clip_to_world(dec[0].astype(np.float64),
                                  dec[1].astype(np.float64),
                                  clip.condition.world_from_clip),
            dec[2], clip.condition.target_obj_pose[start:start + WINDOW]), 1)
        geom = pc.window_geometry_series(
            ctx["replayer"], scene,
            clip.condition.human_kps_world[start:start + WINDOW],
            clip.condition.target_obj_pose[start:start + WINDOW],
            deadline=deadline)
        model = ("teacher_roundtrip" if probe == "teacher_roundtrip" else
                 "deterministic" if probe == "deterministic_generation" else
                 "diffusion")
        out = pc.decode_and_measure(
            demo=demo, model=model, start=start, T=clip.T,
            kind="tail" if pc.is_tail_start(clip.T, start) else "regular",
            group=group, pred_x0=pred, teacher_x0=teacher_windows[index],
            world_from_clip=clip.condition.world_from_clip,
            teacher_root=clip.root, teacher_joints=clip.q,
            teacher_kpts=ctx["teacher_kpts"], model_geometry=geom,
            teacher_geometry=teacher_geom[start], tokenizer=ctx["tokenizer"],
            assembler=assembler, schedule=clip.condition.schedule,
            frame_dt=clip.condition.frame_dt,
            task_tol_m=ctx["w1_reg"]["evaluation"]["task_gate"]["tol_m"],
            hand_gap_m=ctx["validator"].cfg.hand_gap_m,
            decode_device=ctx["device"],
            saved_world_root=(saved_windows[saved_model]["window_root"][index]
                              if saved_model else None))
        if out.get("metrics") is not None and saved_model is not None:
            saved = saved_windows[saved_model]
            out["metrics"]["replay"] = _replay_compare(
                pred, dec,
                {"normalized_prediction": saved["normalized_prediction"][index],
                 "decoded_pos": saved["decoded_pos"][index],
                 "decoded_quat": saved["decoded_quat"][index],
                 "decoded_q": saved["decoded_q"][index]}, [start])
        if out.get("metrics") is not None and probe == "teacher_roundtrip":
            p = out["metrics"]["physical"]
            oracle_absolute = {
                "position_max_m": p["root_position"]["max_m"],
                "joints_max_rad": p["joints"]["max_abs_rad"],
                "rotation_max_deg": p["root_rotation"]["max_deg"],
                "position_pass": p["root_position"]["max_m"] <= pc.ORACLE_TOL["position_m"],
                "joints_pass": p["joints"]["max_abs_rad"] <= pc.ORACLE_TOL["joints_rad"],
                "rotation_pass": p["root_rotation"]["max_deg"] <= pc.ORACLE_TOL["rotation_deg"],
            }
            out["metrics"]["oracle_absolute"] = oracle_absolute
        return out

    for i, start in enumerate(starts):
        for probe, pred, saved_model in (
                ("teacher_roundtrip", teacher_windows[i], None),
                ("deterministic_generation", outputs["det_normalized"][i],
                 "deterministic"),
                ("diffusion_ddim25", outputs["ddim_normalized"][i], "diffusion")):
            key = pc.probe_key(demo, start, probe)
            row = recorded.get(key)
            if row is None:
                problems.append(f"f_rows:{demo}:{key}:missing")
                continue
            out = measure(pred, probe, start, i, reg["groups"][demo], saved_model)
            if out.get("metrics") is None:
                problems.append(f"f_rows:{demo}:{key}:recompute_incomplete")
            elif row.get("metrics") is not None:
                problems += _compare_metric_docs(row["metrics"], out["metrics"],
                                                 f"f_rows:{demo}:{key}")
                problems += _compare_curve_file(
                    unit / "curves" / f"{key.replace('|', '__')}.npz",
                    out["curves"], f"f_curve:{demo}:{key}")
        for ti, tt in enumerate(pc.DENOISE_TIMESTEPS):
            key = pc.probe_key(demo, start, "diffusion_denoise", int(tt))
            row = recorded.get(key)
            if row is None:
                problems.append(f"f_rows:{demo}:{key}:missing")
                continue
            pred = outputs[f"denoise_{start}"][ti]
            out = measure(pred, "diffusion_denoise", start, i,
                          reg["groups"][demo], None)
            if out.get("metrics") is None:
                problems.append(f"f_rows:{demo}:{key}:recompute_incomplete")
            elif row.get("metrics") is not None:
                problems += _compare_metric_docs(row["metrics"], out["metrics"],
                                                 f"f_rows:{demo}:{key}")
                problems += _compare_curve_file(
                    unit / "curves" / f"{key.replace('|', '__')}.npz",
                    out["curves"], f"f_curve:{demo}:{key}")
    return problems


def _session_counts(units_dir: Path) -> dict:
    out = {"det_sessions": 0, "denoise_batches": 0, "ddim_sessions": 0,
           "network_forwards": 0, "conditioner_calls": 0, "high_level_calls": 0}
    for p in sorted(units_dir.glob("*/result.json")):
        result = read_json(p)
        s = (result.get("facts") or {}).get("sessions") or {}
        out["det_sessions"] += 1
        out["denoise_batches"] += int(
            (s.get("diffusion_denoise") or {}).get("windows", 0))
        out["ddim_sessions"] += 1
        out["network_forwards"] += int(
            (s.get("deterministic") or {}).get("network_forwards", 0))
        out["network_forwards"] += int(
            (s.get("diffusion_denoise") or {}).get("network_forwards", 0))
        out["network_forwards"] += int(
            (s.get("diffusion_ddim25") or {}).get("network_forwards", 0))
        out["conditioner_calls"] += int(
            (s.get("deterministic") or {}).get("conditioner_calls", 0))
        out["conditioner_calls"] += int(
            (s.get("diffusion_denoise") or {}).get("conditioner_calls", 0))
        out["conditioner_calls"] += int(
            (s.get("diffusion_ddim25") or {}).get("conditioner_calls", 0))
    out["high_level_calls"] = (out["det_sessions"] + out["denoise_batches"]
                               + out["ddim_sessions"])
    return out


def _replay_call_counts(rows: list[dict]) -> dict:
    units = len(rows)
    learned = sum(r.get("probe") != "teacher_roundtrip" for r in rows)
    oracle = units - learned
    return {"units": units, "learned": learned, "oracle": oracle}


def _run_synthetic_tests(deadline: float) -> dict:
    start = time.perf_counter()
    timeout = max(5.0, min(600.0, deadline - start))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_w1_prediction_consistency.py", "-q"],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        text = (proc.stdout or "") + (proc.stderr or "")
        passed = re.search(r"(\d+) passed", text)
        return {"returncode": proc.returncode,
                "passed": int(passed.group(1)) if passed else None,
                "wall_s": time.perf_counter() - start,
                "tail": text[-1000:]}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "passed": None,
                "wall_s": time.perf_counter() - start, "timeout": True}


def _next_action(judgement: dict, failures: list[str], gaps: list[str]) -> dict:
    if failures or gaps:
        return {"priority": "fix_evidence",
                "candidates": ["resolve failures/evidence gaps before interpreting"],
                "auto_execute": False}
    supported = judgement.get("supported", [])
    if "training_fit" in supported:
        candidates = ["standalone training-fit / target-implementation diagnostic "
                      "(single factor; no resume, no teacher change)"]
    elif "denoise_sampling" in supported:
        candidates = ["standalone denoising-vs-sampling single-factor diagnostic"]
    elif "cross_window" in supported:
        candidates = ["separate window-context consistency supervision and fusion-path "
                      "candidates (never one combined change)"]
    elif "coverage_context" in supported:
        candidates = ["controlled coverage/context comparison (no automatic data expansion)"]
    elif "teacher_inheritance" in supported:
        candidates = ["teacher-defect comparison on at least three train originals "
                      "(no val tuning)"]
    else:
        candidates = ["another discriminating diagnostic; no single-cause claim"]
    return {"priority": candidates[0], "candidates": candidates,
            "required_evidence": "per-example absolute magnitudes and a reproducible "
                                 "counterexample before any intervention",
            "auto_execute": False}


def _write_tables(attempt_dir: Path, w_rows: dict, p_rows: dict,
                  report: dict) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    tables = attempt_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    fields = ("key", "demo", "model", "window_start", "kind", "group",
              "fk_mpjpe_m", "root_rmse_m", "root_rot_max_deg",
              "joint_mae_rad", "penetration_max_m",
              "new_penetration_frames", "shared_penetration_frames")
    with (tables / "window_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in w_rows["rows"]:
            m = r.get("metrics") or {}
            writer.writerow({
                "key": r.get("key"), "demo": r.get("demo"),
                "model": r.get("model"), "window_start": r.get("window_start"),
                "kind": r.get("kind"), "group": r.get("group"),
                "fk_mpjpe_m": _nested(m, "physical.fk.mpjpe_m"),
                "root_rmse_m": _nested(m, "physical.root_position.rmse_m"),
                "root_rot_max_deg": _nested(m, "physical.root_rotation.max_deg"),
                "joint_mae_rad": _nested(m, "physical.joints.mae_rad"),
                "penetration_max_m": _nested(m, "geometry.penetration.max_m"),
                "new_penetration_frames": _nested(m, "geometry.penetration.new_violation_frames"),
                "shared_penetration_frames": _nested(m, "geometry.penetration.shared_violation_frames"),
            })
    with (tables / "probe_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "key", "demo", "probe", "t", "start", "status", "loss_total",
            "fk_mpjpe_m", "alpha_cumprod", "snr"))
        writer.writeheader()
        for r in p_rows["rows"]:
            m = r.get("metrics") or {}
            writer.writerow({
                "key": r.get("key"), "demo": r.get("demo"),
                "probe": r.get("probe"), "t": r.get("t"),
                "start": r.get("start"), "status": r.get("status"),
                "loss_total": _nested(m, "loss.total"),
                "fk_mpjpe_m": _nested(m, "physical.fk.mpjpe_m"),
                "alpha_cumprod": r.get("alpha_cumprod"), "snr": r.get("snr")})
    plots = attempt_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for model in ("deterministic", "diffusion"):
        selected = [r for r in w_rows["rows"] if r.get("model") == model
                    and r.get("status") == "measured"]
        axes[0].plot(range(len(selected)),
                     [_nested(r.get("metrics") or {}, "physical.fk.mpjpe_m") for r in selected],
                     marker="o", ms=2, label=model)
    axes[0].set(ylabel="Window teacher-fit FK MPJPE (m)",
                title="All saved model windows; no fusion rerun")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    for probe in ("diffusion_denoise", "diffusion_ddim25"):
        selected = [r for r in p_rows["rows"] if r.get("probe") == probe
                    and r.get("status") == "measured"]
        axes[1].plot(range(len(selected)),
                     [_nested(r.get("metrics") or {}, "physical.fk.mpjpe_m") for r in selected],
                     marker=".", ms=2, label=probe)
    axes[1].set(xlabel="Probe row index", ylabel="FK MPJPE (m)",
                title="Denoise reconstruction and actual DDIM sampling are separate")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.savefig(plots / "prediction_consistency.png", dpi=150)
    plt.close(fig)


def _nested(doc, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# registration (R)
# ---------------------------------------------------------------------------

def register(run: Path) -> None:
    """R entry: pre-register a running attempt before any substantive check."""
    t0 = time.perf_counter()
    run = guard_run(run)
    cohort = cohort_id(full_sha(SOURCE / "freeze/execution_registration.json"))
    history = _cohort_history(run, cohort)
    if not history.get("known"):
        raise SystemExit(f"same-cohort history unknown: {history.get('reason')}")
    budget = budget_for(run, "register", history)
    (run / "freeze/attempt_001").mkdir(parents=True, exist_ok=True)
    _preregister_attempt(run, "register", "attempt_001", t0, cohort=cohort)
    failure_path = run / "freeze/registration_failure.json"
    try:
        ok = _register_impl(run, t0, cohort, history, budget)
        if not ok:
            raise SystemExit("registration did not seal")
    except BaseException as exc:
        row, problems = stage_state(run, "register")
        if row is None or row.get("status") == "running":
            write_json(failure_path, {
                "status": "incomplete", "complete": False,
                "error": f"{type(exc).__name__}:{exc}",
                "execution_id": EXECUTION_ID,
            })
            _finish_stage(run, "register", "attempt_001", t0, budget,
                          {"complete": False, "status": "incomplete",
                           "report": record_path(failure_path)},
                          {"error": f"{type(exc).__name__}:{exc}"}, "incomplete")
        raise


def _register_impl(run: Path, t0: float, cohort: str, history: dict,
                   budget: dict) -> None:
    w1_reg, selection, att_reg, fus_reg, fus_final = _verify_plan_and_sources()
    att_report, att_windows, att_scenes = _verify_attribution_windows(att_reg)
    fusion_binding = _verify_fusion_binding(fus_reg, fus_final)
    train6 = _select_train6(w1_reg["rosters"]["train24"])
    val6 = list(w1_reg["rosters"]["val6"])
    demos = train6 + val6 + [KNOWN_CASE]
    if len(demos) != 13 or len(set(demos)) != 13 or KNOWN_CASE in train6 + val6:
        raise ValueError("13-scene denominator changed")
    t_by_demo = {}
    starts_by_demo = {}
    for demo in demos:
        row = next((r for r in read_json(
            ROOT / w1_reg["training_manifest"]["path"])["rows"]
            if r.get("demo") == demo), None)
        if row is None:
            raise ValueError(f"demo missing from training manifest:{demo}")
        cache = ROOT / w1_reg["cache"]["dir"]
        clip = w1.load_w1_derived_clip(
            row, cache / "schedules", ROOT,
            read_json(ROOT / w1_reg["preregistration"]["path"])["assets"]["a1_assets_sha256"],
            w1_reg["condition_code_sha256"])
        if clip.T != EXPECTED_T[demo]:
            raise ValueError(f"{demo}: T={clip.T} != plan {EXPECTED_T[demo]}")
        t_by_demo[demo] = clip.T
        starts_by_demo[demo] = pc.inference_starts(clip.T)
    plan = pc.build_window_plan(t_by_demo, train6, val6, KNOWN_CASE)
    global REG_TRAIN6, REG_VAL6
    REG_TRAIN6, REG_VAL6 = list(train6), list(val6)
    if plan["counts"]["cross_window_pairs"] <= 0:
        raise ValueError("empty cross-window pair set")
    snapshot = implementation_snapshot(run)
    barrier = t0 + budget["remaining_s"] - budget["reserve_s"]
    if time.perf_counter() >= barrier:
        raise TimeoutError("registration budget exhausted before substantive checks")
    source = fusion._source_checks(w1_reg, selection, att_reg, barrier)
    att_problems = attribution.hash_problems(att_reg["anchors"])
    if att_problems:
        raise ValueError(att_problems)
    anchors = dict(PLAN_ANCHORS)
    anchors[record_path(PLAN)] = full_sha(PLAN)
    anchors.update(w1_reg.get("source_files", {}))
    for rec in att_reg["anchors"].items():
        anchors[rec[0]] = rec[1]
    for rec in att_windows:
        anchors[rec["path"]] = rec["sha256"]
    for rec in att_scenes:
        anchors[rec["path"]] = rec["sha256"]
    for rec in source["attribution_products"]:
        anchors[rec["path"]] = rec["sha256"]
    for row in source["source_raw_rows"]:
        anchors[row["product"]["path"]] = row["product"]["sha256"]
    for key in ("registration", "final"):
        anchors[fusion_binding[key]["path"]] = fusion_binding[key]["sha256"]
    anchors.update({k: v for k, v in fus_reg["anchors"].items()
                    if k not in set(fus_reg.get("implementation_snapshot", {})
                                    .get("files", {}))})
    for rel in RUNTIME_DEPENDENCIES:
        anchors[rel] = full_sha(ROOT / rel)
    for rel in IMPLEMENTATION_FILES:
        anchors[rel] = full_sha(ROOT / rel)
    if _hash_map(anchors):
        raise ValueError(_hash_map(anchors))
    registration = {
        "execution_id": EXECUTION_ID, "cohort_id": cohort,
        "protocol": "w1_prediction_consistency_v1",
        "plan": file_record(PLAN),
        "source_run": record_path(SOURCE),
        "attribution_run": record_path(ATTRIBUTION),
        "fusion_run": record_path(FUSION),
        "source_registration": file_record(SOURCE / "freeze/execution_registration.json"),
        "source_final": file_record(SOURCE / "decisions/attempt_001/w1_de_final.json"),
        "source_selection": file_record(SOURCE / "evaluation/checkpoint_selection.json"),
        "attribution_registration": file_record(ATTRIBUTION / "freeze/registration.json"),
        "attribution_report": file_record(
            ATTRIBUTION / "diagnosis/attempt_001/attribution.json"),
        "attribution_closure": file_record(
            ATTRIBUTION / "decisions/diagnostic_closure.json"),
        "fusion_registration": fusion_binding["registration"],
        "fusion_final": fusion_binding["final"],
        "fusion_binding": fusion_binding,
        "implementation": driver.implementation_identity(ROOT),
        "implementation_snapshot": snapshot,
        "runtime_dependencies": {rel: full_sha(ROOT / rel)
                                 for rel in RUNTIME_DEPENDENCIES},
        "environment": driver.env_record(ROOT),
        "checkpoints": {m: selection["models"][m] for m in ("deterministic", "diffusion")},
        "train6": train6, "val6": val6, "known_case": KNOWN_CASE,
        "demos": demos, "groups": {d: plan["groups"][d] for d in demos},
        "t_by_demo": t_by_demo, "starts_by_demo": starts_by_demo,
        "window_plan": {
            "window": WINDOW, "stride": STRIDE, "fps": pc.FPS,
            "per_demo": plan["per_demo"], "totals": plan["totals"],
            "counts": plan["counts"],
            "teacher_keys": plan["teacher_keys"],
            "window_keys": plan["window_keys"],
            "probe_keys": plan["probe_keys"],
            "pair_keys": plan["pair_keys"],
            "regular_rule": "range(0, T-60+1, 30)",
            "tail_rule": "append T-60 when absent",
            "zero_weight_rule": "only positive-weight contributors enter pairs; "
                                "zero-weight windows keep their whole-window error row",
        },
        "probe_matrix": {
            "teacher_roundtrip": {"per_window": 1, "rows": 65},
            "deterministic_generation": {"per_window": 1, "rows": 65},
            "diffusion_denoise": {"per_window": 5, "rows": 325,
                                  "timesteps": list(pc.DENOISE_TIMESTEPS)},
            "diffusion_ddim25": {"per_window": 1, "rows": 65},
        },
        "noise": {
            "base_seed": pc.NOISE_SEED, "protocol": pc.NOISE_PROTOCOL,
            "algorithm": "sha256('w1_prediction_consistency_v1|20261021|<demo>|<start>')"
                         "[:8] big-endian mod 2**63; CPU float32 [60,38]; "
                         "same epsilon reused for all five t",
            "t0_semantics": "t=0 is the first noise level, never a no-noise proof",
        },
        "tolerances": {"replay": pc.REPLAY_TOL, "oracle": pc.ORACLE_TOL,
                       "float64_metric": pc.F64_METRIC_TOL,
                       "keyframe_penetration_m": pc.KEYFRAME_PENETRATION_M,
                       "angular_speed_limit_radps": pc.ANGULAR_SPEED_LIMIT_RADPS},
        "metric_schema": [spec.path for spec in pc.WINDOW_METRICS],
        "attribution_comparators": pc.ATTRIBUTION_COMPARATORS,
        "budgets_s": {k: v[0] for k, v in STAGE_BUDGETS.items()},
        "reserves_s": {k: v[1] for k, v in STAGE_BUDGETS.items()},
        "total_budget_s": TOTAL_BUDGET_S,
        "cohort_history": history, "parent_runs": history.get("runs", []),
        "prior_cohort_cost_s": history.get("total_s", 0.0),
        "source_stage_states": source["stages"],
        "source_budgets_s": source["budgets_s"],
        "source_manifest": w1_reg["training_manifest"],
        "source_cache": w1_reg["cache"],
        "validation_config": w1_reg["evaluation"]["validation_config"],
        "task_gate_tol_m": w1_reg["evaluation"]["task_gate"]["tol_m"],
        "attribution_windows": att_windows,
        "attribution_scenes": att_scenes,
        "anchors": anchors, "anchor_count": len(anchors),
        "training_steps": 0, "production_changes": False,
        "stop_conditions": [
            "source/implementation/path drift", "unknown or running history",
            "budget exhausted", "fixed key set incomplete",
            "numerical failure without re-run", "actual call count above registered bound",
            "disk not writable", "GPU/memory/resource failure",
        ],
        "created_wall_s": time.perf_counter() - t0,
    }
    if time.perf_counter() >= barrier:
        raise TimeoutError("registration exhausted its budget")
    write_json(run / "freeze/execution_registration.json", registration,
               exclusive=True)
    row = {"complete": True, "status": "done",
           "report": record_path(run / "freeze/execution_registration.json"),
           "artifacts": {record_path(run / "freeze/execution_registration.json"):
                         full_sha(run / "freeze/execution_registration.json")}}
    ok = _finish_stage(run, "register", "attempt_001", t0, budget, row,
                       {"anchor_count": len(anchors), "train6": train6,
                        "val6": val6, "pairs": plan["counts"]["cross_window_pairs"]})
    if not ok:
        return False
    print(json.dumps({"registered": record_path(run), "cohort": cohort,
                      "anchors": len(anchors), "counts": plan["counts"],
                      "train6": train6, "val6": val6}, ensure_ascii=False),
          flush=True)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def saved_window_unit(run: Path, demo: str, out: Path, deadline: float) -> dict:
    reg = verify_registration(run)
    result = recompute_saved_windows(reg, demo, deadline=deadline, write=out)
    write_json(out / "result.json", result)
    return result


def probe_unit_cli(run: Path, demo: str, out: Path, deadline: float) -> dict:
    reg = verify_registration(run)
    return probe_unit(reg, demo, out, deadline, write=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "saved-windows", "probes",
                                          "finalize", "saved-window-unit",
                                          "probe-unit"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--demo")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    run = guard_run(args.run_root) if args.stage == "register" else resolve_run(
        args.run_root)
    if args.stage == "register":
        register(run)
        return 0
    if args.stage == "saved-window-unit":
        if not args.demo or args.out is None or args.deadline is None:
            raise SystemExit("saved-window-unit requires --demo/--out/--deadline")
        result = saved_window_unit(run, args.demo, args.out, args.deadline)
        return 0 if result.get("status") == "ok" else 2
    if args.stage == "probe-unit":
        if not args.demo or args.out is None or args.deadline is None:
            raise SystemExit("probe-unit requires --demo/--out/--deadline")
        result = probe_unit_cli(run, args.demo, args.out, args.deadline)
        return 0 if result.get("status") == "ok" else 2
    if args.stage == "saved-windows":
        return 0 if saved_windows_stage(run) else 2
    if args.stage == "probes":
        return 0 if probes_stage(run) else 2
    return 0 if finalize_stage(run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
