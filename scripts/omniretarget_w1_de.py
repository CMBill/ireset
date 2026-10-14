"""W1-D/E execution driver (omniretarget_w1_de_plan.md).

Stage entries (plan §9): register -> train-deterministic -> train-diffusion
-> raw-eval -> detail -> finalize. ``register`` runs R0 (SOURCE fact
re-verification through the readiness checkers) + R1 (exclusive
execution-registration write). The train and raw-eval stages spawn the real
``train_generation`` / ``eval_generation`` entries as SUBPROCESSES with the
recorded argv; the driver owns the monotonic budget ledger (pre-registered
``running`` row per (stage, attempt), updated in place on every exit) and
the final deadline re-check that flips an over-budget "complete"
registration to incomplete IN PLACE (§8.2).

The SOURCE chain (20261012_21e5784_w1_a3bc_reviewfix8) and every plan
anchor below are re-HASHED at registration and at every stage entry — a
``true`` flag in an old report is never the entry condition; the anchors'
current bytes are.

Registration of a stage never claims training authorization beyond the
plan's contract: U0 ``incomplete`` / UR ``no_quality_gain`` stand; W1-D/E is
the registered weak-motion diagnostic round only.

Run:
  uv run --locked python scripts/omniretarget_w1_de.py \
    --stage register --source-run-root <SOURCE> --run-root <RUN> \
    --plan docs/plans/omniretarget_w1_de_plan.md
  uv run --locked python scripts/omniretarget_w1_de.py \
    --stage train-deterministic --run-root <RUN>   (then train-diffusion,
    raw-eval, detail, finalize)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# --- identity / budget constants (plan §2.3, §3.2, §8.1) -----------------------

W1_DE_EXECUTION_ID = "w1_de_v1"
D_BUDGET_S = 43200            # two models, cumulative GPU wall
D_RESERVE_S = 300             # final save/registration tail (§8.2)
D_MAX_STEPS = 4000            # per-model optimizer-step cap (§3.2)
E_BUDGET_S = 7200
E_RESERVE_S = 60
DETAIL_F_BUDGET_S = 3600
DETAIL_F_RESERVE_S = 60
R0R1_BUDGET_S = 7200
R0R1_RESERVE_S = 60

TASK_MPJPE_TOL_M = 0.40       # §6.4 diagnostic task working gate
MAIN_SEED = 7
APPENDIX_SEEDS = (17, 27)
METHODS = ("teacher", "deterministic", "diffusion")
MODELS = ("deterministic", "diffusion")

#: §2.1 SOURCE anchors — the bytes this round reasons over (relative to the
#: SOURCE run root); every entry re-hashes at registration and each entry.
SOURCE_ANCHORS = {
    "freeze/preregistration.json":
        "071c49d32bf822e00ee0854a86e5abf25a0df9c22f033b93732a26b8753301f5",
    "audit/attempt_001/training_manifest.json":
        "a350a8b8922ca94308de3d97787e60bf8f6a51acff6f20af9f513fd42b10e525",
    "audit/attempts_index.json":
        "55ce32790ed32bddb13aa9bd0a09f628763ce5fec737b4a2de15c369d01bb5f6",
    "cache/attempt_001/cache_identity.json":
        "90af39f123c4dc295e8f16b07b0002602c173e8d68d33ae1abe23ed611cd2e71",
    "cache/attempt_001/root_stats.npz":
        "b9d9cc049a6b58eea7ab048b886a5fd9f87755690d1f0a789cb4f113c5bd2eda",
    "cache/attempts_index.json":
        "88e40985f12aca63ce4ac0ecb22884b23bd42fb106b812d687d23484c6639b3b",
    "decisions/attempt_001/w1_readiness.json":
        "7245ced433bba50a66c6dcd175774866eb5c2062ef6970b508bd61d453c3b558",
    "costs.json":
        "43573f1bbe47a3199d0a238a0422cd0e4581b0840f47b4b59e01b00daa341aee",
    "s0/deterministic/attempt_001/s0_summary.json":
        "50a23e3c29dd614a7e03fe2abf590930c13b2daf5429ae7dbbb9f4dd5a86e06b",
    "s0/deterministic/attempt_001/last.pt":
        "716bc138f5f712c329c6c10674929f4cfa3e27c8c7f12d187e23a38c8c96a3aa",
    "s0/deterministic/attempt_001/probe_replay.pt":
        "62602cc73c2a5c0c002453ee9846815d89c1e9709429dc0ab4b1a68ec239deed",
    "s0/diffusion/attempt_001/s0_summary.json":
        "93e2298eb0f7f2e10e00d7e6cce404b1d924e5c472e8a781cbcb066a11b5e107",
    "s0/diffusion/attempt_001/last.pt":
        "c583dbcb4cfbc5f7b9d19cfa4f33424f25aca37476c23bc86e263928616efc26",
    "s0/diffusion/attempt_001/probe_replay.pt":
        "142f68772b090d9900445d8d15ff9ca20b945a8af3b450cb754352b5d9b5c842",
}

#: §2.1 project-root protocol/config anchors
PLAN_ANCHORS = {
    "docs/plans/omniretarget_first_training_plan.md":
        "ca7cd4d8db553cac0057d1392c3238f0ca0fbfcbe4748f4c5946a3e650da2e84",
    "docs/plans/omniretarget_a2_closure_plan.md":
        "28fd40e43af0f524e5c2c23dc776a5a9b1b00bd872d75ee90442b5f41a8b81b3",
    "docs/plans/omniretarget_w1_a3bc_plan.md":
        "e7247e4b76d7cdfec04e288883dbdaccd868c7908108a8f2f5171839c766fc2f",
    "configs/train/omniretarget_motion.yaml":
        "0e2cfa96849ee2de14ac84b7343289090c606f78116bb79114ff81b046b521e0",
    "configs/labels/omniretarget_motion_v1.yaml":
        "4672b3b97dd3cf0895d205171c1ba3a677a54edfefde1db21bfd3722eb086ce9",
    "uv.lock":
        "f809284e21406f7f3b51c33bf200f0e6aa567f32cce43bc15ab00797b6d69a44",
}

TRAIN_CONFIG_REL = "configs/train/omniretarget_motion.yaml"

#: stage -> (ledger stage name, attempt-directory relative to run root)
STAGES = {
    "register": "w1_de_register",
    "train-deterministic": "w1_de_train_deterministic",
    "train-diffusion": "w1_de_train_diffusion",
    "raw-eval": "w1_de_raw_eval",
    "detail": "w1_de_detail",
    "finalize": "w1_de_finalize",
}
STAGE_DIRS = {
    "w1_de_register": "preflight",
    "w1_de_train_deterministic": "formal/deterministic",
    "w1_de_train_diffusion": "formal/diffusion",
    "w1_de_raw_eval": "evaluation/raw",
    "w1_de_detail": "evaluation/detail_val6",
    "w1_de_finalize": "decisions",
}
#: budget domains (§8.1): stage names -> (budget_s, reserve_s)
DOMAINS = {
    "r0r1": (("w1_de_register",), R0R1_BUDGET_S, R0R1_RESERVE_S),
    "d": (("w1_de_train_deterministic", "w1_de_train_diffusion"),
          D_BUDGET_S, D_RESERVE_S),
    "e": (("w1_de_raw_eval",), E_BUDGET_S, E_RESERVE_S),
    "detail_f": (("w1_de_detail", "w1_de_finalize"),
                 DETAIL_F_BUDGET_S, DETAIL_F_RESERVE_S),
}
#: floor for the seal-tail reserve (§8.2): a completion must afford its own
#: final index/ledger registration; the charged reserve bounds the seal
#: writes' latency so a kill after the seal lands cannot leave an
#: undersampled done row (scaled up from observed write latencies per call).
SEAL_TAIL_RESERVE_S = 1.0
#: post-seal verification rounds: each round only continues when a write was
#: slower than the previous reserve (latencies must keep growing), so the
#: bound only matters under adversarial scheduling; exhaustion revokes.
SEAL_VERIFY_ROUNDS = 4


# --- cost ledger (§8.2/§8.3) ---------------------------------------------------

def atomic_json(path: Path, data, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                      default=str).encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def execution_lock(project_root: Path):
    lock_path = Path(project_root) / "derived/experiments/omniretarget_first/.w1_de.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another W1-D/E stage is active in this workspace") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _load_costs(run_root: Path) -> list[dict]:
    costs_path = Path(run_root) / "costs.json"
    if not costs_path.is_file():
        return []
    doc = json.loads(costs_path.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("invocations"), list):
        raise ValueError("cost ledger requires invocations list")
    if not all(isinstance(e, dict) for e in doc["invocations"]):
        raise ValueError("cost ledger contains a non-mapping row")
    return doc["invocations"]


def _upsert_cost(run_root: Path, stage: str, attempt: str, wall: float,
                 extra: dict, status: str) -> None:
    if not _valid_wall(wall):
        raise ValueError("wall cost must be finite and nonnegative")
    entries = _load_costs(run_root)
    hits = [i for i, e in enumerate(entries)
            if e.get("stage") == stage and e.get("attempt") == attempt]
    if len(hits) > 1:
        raise ValueError("duplicate cost ledger keys")
    row = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
           "execution_id": W1_DE_EXECUTION_ID,
           "stage": stage, "attempt": attempt, "status": status,
           "wall_s": round(wall, 1), "measured_wall_s": float(wall), **extra}
    model = stage.removeprefix("w1_de_train_") if stage.startswith("w1_de_train_") else None
    row["scope"] = {"stage": stage, "model": model, "attempt": attempt}
    row["model"] = model
    if hits:
        row["timestamp"] = entries[hits[0]].get("timestamp")
        entries[hits[0]] = {**entries[hits[0]], **row}
    else:
        entries.append(row)
    atomic_json(Path(run_root) / "costs.json", {"invocations": entries})


def _index_rows(stage_root: Path) -> list[dict]:
    path = Path(stage_root) / "attempts_index.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("attempts"), list):
        raise ValueError("index requires attempts list")
    rows = doc["attempts"]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("index contains a non-mapping row")
    keys = [row.get("attempt") for row in rows]
    if any(not isinstance(k, str) or not re.fullmatch(r"attempt_\d{3,}", k)
           for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("index attempt keys missing or duplicate")
    return rows


def append_stage_index(stage_root: Path, entry: dict) -> None:
    rows = _index_rows(stage_root)
    attempt = entry["attempt"]
    if any(row["attempt"] == attempt for row in rows):
        raise ValueError(f"attempt already indexed: {attempt}")
    rows.append({"timestamp": _dt.datetime.now().isoformat(timespec="seconds"), **entry})
    atomic_json(Path(stage_root) / "attempts_index.json", {"attempts": rows})


def _replace_index_row(stage_root: Path, attempt: str, entry: dict) -> None:
    rows = _index_rows(stage_root)
    hits = [i for i, row in enumerate(rows) if row["attempt"] == attempt]
    if len(hits) != 1:
        raise ValueError(f"index requires exactly one row for {attempt}")
    rows[hits[0]] = {**rows[hits[0]], **entry}
    atomic_json(Path(stage_root) / "attempts_index.json", {"attempts": rows})


def _valid_wall(w) -> bool:
    return (isinstance(w, (int, float)) and not isinstance(w, bool)
            and math.isfinite(float(w)) and float(w) >= 0)


def _local_domain_history_cost_s(run_root: Path, domain: str) -> float | None:
    total = 0.0
    try:
        entries = _load_costs(run_root)
        strict = (Path(run_root) / "freeze/execution_scope.json").is_file()
        if strict and any(e.get("stage") not in STAGE_DIRS for e in entries):
            return None
        for stage in DOMAINS[domain][0]:
            stage_root = Path(run_root) / STAGE_DIRS[stage]
            disk = {p.name for p in stage_root.glob("attempt_*") if p.is_dir()}
            rows = _index_rows(stage_root)
            index = {row["attempt"]: row for row in rows}
            ledger = {}
            for e in entries:
                if e.get("stage") != stage:
                    continue
                attempt = e.get("attempt")
                if (not isinstance(attempt, str) or attempt in ledger
                        or not _valid_wall(e.get("wall_s"))
                        or e.get("status") not in ("done", "crashed", "timeout", "incomplete")):
                    return None
                measured = e.get("measured_wall_s", e["wall_s"])
                if not _valid_wall(measured) or abs(round(measured, 1) - e["wall_s"]) > 1e-9:
                    return None
                ledger[attempt] = e
            if set(index) != disk or set(ledger) != disk:
                return None
            for attempt, e in ledger.items():
                row = index[attempt]
                if row.get("status") == "publishing":
                    # Terminal content published but the final deadline check
                    # never sealed it — non-consumable, unknown = stop.
                    return None
                if strict:
                    model = stage.removeprefix("w1_de_train_") if stage.startswith("w1_de_train_") else None
                    scope = {"stage": stage, "model": model, "attempt": attempt}
                    if e.get("scope") != scope or row.get("scope") != scope:
                        return None
                    if e.get("execution_id") != W1_DE_EXECUTION_ID:
                        return None
                    if row.get("status") != e["status"]:
                        return None
                    if not isinstance(row.get("complete"), bool) or row["complete"] is not e.get("complete"):
                        return None
                    if row["complete"] and e["status"] != "done":
                        return None
                    if row.get("measured_wall_s") != e.get("measured_wall_s"):
                        return None
                total += float(e.get("measured_wall_s", e["wall_s"]))
                if not math.isfinite(total):
                    return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return total


def _execution_history_path(project_root: Path) -> Path:
    return Path(project_root) / "derived/experiments/omniretarget_first/w1_de_execution_history.json"


def _history_roots(run_root: Path) -> list[Path]:
    from ireset.paths import PROJECT_ROOT

    history = _execution_history_path(PROJECT_ROOT)
    if not history.exists():
        if (Path(run_root) / "freeze/execution_scope.json").exists():
            raise ValueError("execution scope has no shared history")
        return [Path(run_root)]
    doc = json.loads(history.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("runs"), list):
        raise ValueError("execution history malformed")
    roots = []
    for rec in doc["runs"]:
        if not isinstance(rec, dict) or not isinstance(rec.get("run_root"), str):
            raise ValueError("execution history scope malformed")
        if rec.get("execution_id") == W1_DE_EXECUTION_ID:
            path = (Path(PROJECT_ROOT) / rec["run_root"]).resolve()
            marker = path / "freeze/execution_scope.json"
            if not marker.is_file() or json.loads(marker.read_text()) != rec:
                raise ValueError("execution history marker mismatch")
            roots.append(path)
    target = Path(run_root).resolve()
    marker = target / "freeze/execution_scope.json"
    if marker.exists() and target not in roots:
        raise ValueError("execution scope deleted from shared history")
    for marker_path in (Path(PROJECT_ROOT) / "derived/experiments/omniretarget_first").glob("*/freeze/execution_scope.json"):
        scope = json.loads(marker_path.read_text())
        if scope.get("execution_id") == W1_DE_EXECUTION_ID and marker_path.parents[1].resolve() not in roots:
            raise ValueError("execution root missing from shared history")
    if target not in roots:
        roots.append(target)
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate execution history roots")
    return roots


def _claim_execution_root(run_root: Path, project_root: Path, source_root: Path,
                          plan_rel: str) -> None:
    from ireset.data.a2_integrity import sha256_file

    root_record = str(Path(run_root).resolve())
    rec = {"execution_id": W1_DE_EXECUTION_ID, "run_root": root_record,
           "source_root": str(Path(source_root).resolve()),
           "plan_sha256": sha256_file(Path(project_root) / plan_rel)}
    marker = Path(run_root) / "freeze/execution_scope.json"
    history = _execution_history_path(project_root)
    doc = json.loads(history.read_text()) if history.exists() else {"runs": []}
    if not isinstance(doc.get("runs"), list):
        raise ValueError("execution history requires runs list")
    prior = [r for r in doc["runs"] if r.get("run_root") == root_record]
    if prior and prior != [rec]:
        raise SystemExit("execution root is already bound to different source/plan")
    if marker.exists() and json.loads(marker.read_text()) != rec:
        raise SystemExit("execution scope drifted")
    if not prior:
        doc["runs"].append(rec)
        atomic_json(history, doc)
    if not marker.exists():
        atomic_json(marker, rec, exclusive=True)


def domain_history_cost_s(run_root: Path, domain: str) -> float | None:
    total = 0.0
    try:
        for root in _history_roots(run_root):
            cost = _local_domain_history_cost_s(root, domain)
            if cost is None:
                return None
            total += cost
            if not math.isfinite(total):
                return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return total


def _registration_sha(run_root: Path) -> str | None:
    from ireset.data.a2_integrity import sha256_file

    path = Path(run_root) / "freeze/execution_registration.json"
    return sha256_file(path) if path.is_file() else None


def finish_attempt(run_root: Path, stage: str, attempt_dir: Path,
                   t0: float, budget: dict, row: dict, extra: dict,
                   status: str = "done") -> bool:
    from ireset.data.a2_integrity import sha256_file
    from ireset.paths import PROJECT_ROOT

    stage_root = Path(attempt_dir).parent
    attempt = Path(attempt_dir).name
    model = stage.removeprefix("w1_de_train_") if stage.startswith("w1_de_train_") else None
    scope = {"stage": stage, "model": model, "attempt": attempt}
    row = {**row, "attempt": attempt, "scope": scope, "status": status,
           "execution_registration_sha256": _registration_sha(run_root)}
    for key in ("report", "summary"):
        if row.get(key):
            path = Path(PROJECT_ROOT) / row[key]
            if path.is_file():
                row[f"{key}_sha256"] = sha256_file(path)
    # Four-phase publication. Phase 1: pending pre-registration (running/
    # complete=false). Phase 2: the terminal content is published under the
    # dedicated NON-CONSUMABLE status "publishing" — the deadline check has
    # not yet run over that write. Phase 3 (seal): after the publication
    # check passes, the row is sealed to the terminal status; the completion
    # must PROVE it can afford its own seal — the pre-seal check reserves a
    # tail bound (floor + 2x the write latencies observed in this call) and
    # the charged cost covers the seal writes, so a SIGKILL right after the
    # seal write lands cannot leave an undersampled completion. Phase 4: the
    # post-seal deadline check runs AFTER the last registration (plan §8.2)
    # and revokes a completion whose seal write itself crossed the deadline;
    # revocations re-publish under "publishing" and re-seal incomplete — the
    # safe direction. A SIGKILL anywhere before a done seal leaves a
    # running/publishing row (non-consumable); after it lands, the charged
    # cost bounds the residual tail by the recorded seal_tail_reserve_s.
    remaining = budget["remaining_s"]
    write_s: list[float] = []

    def _tw(fn, *args):
        start = time.perf_counter()
        fn(*args)
        write_s.append(time.perf_counter() - start)

    def _stop_docs(rw):
        for key in ("report", "summary"):
            if not rw.get(key):
                continue
            path = Path(PROJECT_ROOT) / rw[key]
            if path.is_file():
                doc = json.loads(path.read_text())
                if isinstance(doc, dict):
                    doc.update(complete=False,
                               incomplete_reason=rw["incomplete_reason"],
                               budget_stopped=True,
                               measured_wall_s=time.perf_counter() - t0)
                    if isinstance(doc.get("completeness"), dict):
                        doc["completeness"].update(complete=False, budget_stopped=True)
                    atomic_json(path, doc)
                    rw[f"{key}_sha256"] = sha256_file(path)
        return rw

    def _publish(rw):
        rw = {**rw, "status": "publishing",
              "measured_wall_s": time.perf_counter() - t0}
        if any(r["attempt"] == attempt for r in _index_rows(stage_root)):
            _tw(_replace_index_row, stage_root, attempt, rw)
        else:
            _tw(append_stage_index, stage_root, rw)
        _tw(_upsert_cost, run_root, stage, attempt, rw["measured_wall_s"],
            {**extra, "complete": rw.get("complete", False),
             "incomplete_reason": rw.get("incomplete_reason")}, "publishing")
        return rw

    overrun = time.perf_counter() - t0 >= remaining
    if not overrun:
        pending = {**row, "status": "running", "complete": False,
                   "measured_wall_s": time.perf_counter() - t0}
        if any(r["attempt"] == attempt for r in _index_rows(stage_root)):
            _tw(_replace_index_row, stage_root, attempt, pending)
        else:
            _tw(append_stage_index, stage_root, pending)
        _tw(_upsert_cost, run_root, stage, attempt, pending["measured_wall_s"],
            {**extra, "complete": False}, "running")
        # Verdict AFTER the pending registration: registration time itself
        # counts against the budget.
        overrun = time.perf_counter() - t0 >= remaining
    if overrun:
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_registration")
        row = _stop_docs(row)
        status = "incomplete"
    row = _publish(row)
    # Deadline check AFTER the terminal publication: the publication writes
    # themselves count against the budget, so a completion whose publication
    # crossed the deadline is revoked before the seal.
    if not overrun and time.perf_counter() - t0 >= remaining:
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_after_publication")
        row = _stop_docs(row)
        status = "incomplete"
        row = _publish(row)
    # Seal-tail reserve: bound the latency of the seal writes from the write
    # latencies already observed in this call (floor covers the cold case).
    reserve = max(SEAL_TAIL_RESERVE_S, 2.0 * max(write_s, default=0.0))
    if not overrun and time.perf_counter() - t0 + reserve >= remaining:
        # The completion cannot afford its own seal registration.
        overrun = True
        row.update(complete=False,
                   incomplete_reason="budget_exceeded_before_seal")
        row = _stop_docs(row)
        status = "incomplete"
        row = _publish(row)

    def _write_terminal(rw, st, cost, res):
        rw = {**rw, "status": st, "measured_wall_s": cost,
              "seal_tail_reserve_s": res}
        _tw(_replace_index_row, stage_root, attempt, rw)
        _tw(_upsert_cost, run_root, stage, attempt, cost,
            {**extra, "complete": rw.get("complete", False),
             "incomplete_reason": rw.get("incomplete_reason"),
             "seal_tail_reserve_s": res}, st)
        return rw

    def _rederive_reserve():
        return max(SEAL_TAIL_RESERVE_S, 2.0 * max(write_s, default=0.0))

    def _revoke(reason):
        # Every terminal registration — including a revocation — charges a
        # reserve covering its own final write; the verify loop below checks
        # the coverage and leaves the row unknown when it cannot be proven.
        nonlocal overrun, reserve
        overrun = True
        reserve = _rederive_reserve()
        row.update(complete=False, incomplete_reason=reason)
        _stop_docs(row)
        _publish(row)
        return _write_terminal(row, "incomplete",
                               time.perf_counter() - t0 + reserve, reserve)

    def _leave_unknown(rw):
        # Cost coverage could not be proven within the round budget: the
        # registration is left unpublished (status publishing) — non-
        # consumable, so the domain history stays unknown instead of an
        # under-counted consumable row.
        rw.setdefault("incomplete_reason", "seal_tail_reserve_unverifiable")
        _stop_docs(rw)
        return _publish(rw)

    # Terminal seal: the reserve is charged into EVERY terminal registration
    # (completions and revocations alike) so a kill after the final write
    # lands leaves measured >= actual.
    reserve = _rederive_reserve()
    row = _write_terminal(row, status, time.perf_counter() - t0 + reserve,
                          reserve)
    # Post-seal verification AFTER the last registration (plan §8.2: the
    # final index/ledger registration is itself covered). The loop must BOTH
    # enforce the deadline AND verify the charged reserve actually covered
    # the seal writes — for completions AND revoked attempts alike: an
    # undersampled registration is refreshed with the observed cost (reserve
    # re-derived from the now larger write latencies), revoked when the
    # refresh cannot be afforded, or left unknown when coverage cannot be
    # proven. An under-counted cost must never survive into scheduling.
    for _ in range(SEAL_VERIFY_ROUNDS):
        post = time.perf_counter() - t0
        if not overrun and post >= remaining:
            row = _revoke("budget_exceeded_after_seal")
            continue
        if post <= row["measured_wall_s"]:
            break  # the charged cost covers the actual elapsed time
        reserve = _rederive_reserve()
        if not overrun and post + reserve >= remaining:
            # An honest re-registration would itself cross the deadline.
            row = _revoke("seal_tail_reserve_insufficient")
            continue
        row = _write_terminal(row, status, post + reserve, reserve)
    else:
        row = _leave_unknown(row)
    return not overrun and row.get("complete", row.get("passed", False)) is True


def crash_attempt(run_root: Path, stage: str, attempt_dir: Path, t0: float,
                  budget: dict, row: dict, extra: dict, exc: BaseException) -> bool:
    return finish_attempt(run_root, stage, attempt_dir, t0, budget,
                          {**row, "complete": False, "status": "crashed",
                           "error": f"{type(exc).__name__}: {exc}"},
                          {**extra, "error": f"{type(exc).__name__}: {exc}"}, "crashed")


def stage_state(run_root: Path, stage: str, project_root: Path) -> tuple[dict | None, list[str]]:
    from ireset.data.a2_integrity import sha256_file

    problems = []
    try:
        root = Path(run_root) / STAGE_DIRS[stage]
        rows = _index_rows(root)
        if not rows:
            return None, ["no_attempts"]
        row = rows[-1]
        attempt = row["attempt"]
        ledger = [e for e in _load_costs(run_root)
                  if e.get("stage") == stage and e.get("attempt") == attempt]
        if len(ledger) != 1 or not (root / attempt).is_dir():
            return row, ["attempt_index_ledger_disk_mismatch"]
        e = ledger[0]
        terminal_statuses = ("done", "crashed", "timeout", "incomplete")
        if e.get("status") == "running" or row.get("status") == "running":
            problems.append("attempt_running")
        elif (e.get("status") == "publishing" or row.get("status") == "publishing"):
            problems.append("attempt_final_check_pending")
        elif e.get("status") not in terminal_statuses or row.get("status") not in terminal_statuses:
            problems.append("attempt_status_invalid")
        elif e["status"] != row["status"]:
            problems.append("index_ledger_status_mismatch")
        if row.get("complete") is True and (e.get("status") != "done" or e.get("complete") is not True):
            problems.append("index_ledger_completion_contradiction")
        if row.get("complete") is not e.get("complete"):
            problems.append("index_ledger_completion_mismatch")
        model = stage.removeprefix("w1_de_train_") if stage.startswith("w1_de_train_") else None
        scope = {"stage": stage, "model": model, "attempt": attempt}
        if row.get("scope") != scope or e.get("scope") != scope or e.get("execution_id") != W1_DE_EXECUTION_ID:
            problems.append("attempt_scope_mismatch")
        if row.get("measured_wall_s") != e.get("measured_wall_s"):
            problems.append("index_ledger_cost_mismatch")
        if row.get("execution_registration_sha256") != _registration_sha(run_root):
            problems.append("attempt_registration_drifted")
        for key in ("report", "summary"):
            if row.get(key):
                p = Path(project_root) / row[key]
                if not p.is_file() or sha256_file(p) != row.get(f"{key}_sha256"):
                    problems.append(f"{key}_hash_mismatch")
        artifacts = row.get("artifacts", {})
        if not isinstance(artifacts, dict):
            problems.append("attempt_artifacts_not_a_mapping")
        else:
            for rel, sha in artifacts.items():
                p = Path(project_root) / rel
                if not p.is_file() or sha256_file(p) != sha:
                    problems.append(f"artifact_hash_mismatch:{rel}")
        if (not _valid_wall(e.get("measured_wall_s")) or not _valid_wall(e.get("wall_s"))
                or abs(round(e["measured_wall_s"], 1) - e["wall_s"]) > 1e-9):
            problems.append("attempt_cost_invalid")
        if not isinstance(row.get("complete"), bool) or not isinstance(e.get("complete"), bool):
            problems.append("attempt_completion_invalid")
        return row, problems
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, [f"stage_state_unreadable:{type(exc).__name__}"]


def _budget_state(run_root: Path, domain: str) -> dict:
    """Remaining budget for a domain; raises SystemExit on unknown/exhausted
    history (never silently zero)."""
    hist = domain_history_cost_s(run_root, domain)
    if hist is None:
        raise SystemExit(
            f"{domain} cost history unknown (attempts without complete "
            "index/ledger coverage, or a leftover running row) — refusing "
            "to start without a provable remaining budget")
    _, budget, _ = DOMAINS[domain]
    remaining = budget - hist
    if remaining <= DOMAINS[domain][2]:
        raise SystemExit(
            f"{domain} budget exhausted: historical {hist:.1f}s of {budget}s "
            f"(remaining {remaining:.1f}s <= reserve "
            f"{DOMAINS[domain][2]}s — not enough for work + registration)")
    return {"total_s": budget, "historical_s": hist,
            "remaining_s": remaining, "reserve_s": DOMAINS[domain][2]}


# --- implementation identity / environment -------------------------------------

def implementation_identity(project_root: Path) -> dict:
    """Current code identity: HEAD revision, tracked-diff content hash and
    the untracked-file content map (a status line list cannot distinguish
    same-name-different-content)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(project_root), *args],
                              capture_output=True, text=True,
                              check=True).stdout
    from ireset.data import a2_integrity as integ

    untracked_files = sorted(
        l for l in git("ls-files", "--others", "--exclude-standard").splitlines()
        if l.strip())
    return {
        "revision": git("rev-parse", "HEAD").strip(),
        "status": [l for l in git("status", "--porcelain").splitlines() if l],
        "tracked_diff_sha256": hashlib.sha256(
            git("diff", "HEAD").encode()).hexdigest(),
        "untracked": {f: integ.sha256_file(project_root / f)
                      for f in untracked_files},
    }


def env_record(project_root: Path) -> dict:
    """Measured environment: interpreter, uv.lock, torch/CUDA probe,
    mujoco — the plan §2.2 item-6 record."""
    from ireset.data import a2_integrity as integ

    rec: dict = {"python": sys.version.split()[0],
                 "executable": sys.executable,
                 "uv_lock_sha256": integ.sha256_file(project_root / "uv.lock")}
    import mujoco
    import torch

    rec["mujoco"] = mujoco.__version__
    rec["torch"] = torch.__version__
    rec["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        rec["cuda_device"] = torch.cuda.get_device_name(0)
        with torch.random.fork_rng(devices=[0]):
            a = torch.randn(64, 64, device="cuda", requires_grad=True)
            (a * 2).sum().backward()
            torch.cuda.synchronize()
            rec["cuda_forward_backward_probe_ok"] = bool(
                torch.isfinite(a.grad).all().item())
    rec["cuda_runtime"] = torch.version.cuda
    rec["native_threads"] = {k: os.environ.get(k) for k in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    return rec


# --- registration (R1) ----------------------------------------------------------

def _registration_source_legs(project_root: Path, source_root: Path) -> dict:
    """Re-hash every SOURCE + plan anchor (the §2.1 tables) against the
    CURRENT bytes; any drift is a refusal reason."""
    from ireset.data import a2_integrity as integ

    out = {"source_anchors": {}, "plan_anchors": {}, "problems": []}
    for rel, want in SOURCE_ANCHORS.items():
        p = source_root / rel
        if not p.is_file():
            out["problems"].append(f"source_anchor_missing:{rel}")
        else:
            got = integ.sha256_file(p)
            out["source_anchors"][rel] = got
            if got != want:
                out["problems"].append(f"source_anchor_drifted:{rel}")
    for rel, want in PLAN_ANCHORS.items():
        p = project_root / rel
        if not p.is_file():
            out["problems"].append(f"plan_anchor_missing:{rel}")
        else:
            got = integ.sha256_file(p)
            out["plan_anchors"][rel] = got
            if got != want:
                out["problems"].append(f"plan_anchor_drifted:{rel}")
    return out


def _record_path(path: Path, project_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _expanded_source_files(project_root: Path, source_root: Path) -> dict:
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1
    from ireset.paths import OMOMO_NEW_DIR, TEACHER_TRAJ_DIR

    manifest_path = source_root / "audit/attempt_001/training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prereg = json.loads((project_root / manifest["preregistration"]["path"]).read_text())
    files = {}

    def add(path: Path, expected: str) -> None:
        path = Path(path)
        if not path.is_file() or integ.sha256_file(path) != expected:
            raise ValueError(f"source content missing or drifted: {path}")
        files[_record_path(path, project_root)] = expected

    for rec in prereg["anchors"].values():
        add(project_root / rec["path"], rec["sha256"])
    for demo, rec in prereg["inputs"].items():
        add(TEACHER_TRAJ_DIR / f"{demo}_original.npz", rec["npz"])
        add(OMOMO_NEW_DIR / f"{demo}.pt", rec["pt"])
    for row in manifest["rows"]:
        add(project_root / row["derived"]["path"], row["derived"]["sha256"])
    asset_path = project_root / prereg["assets"]["a1_assets_path"]
    assets = json.loads(asset_path.read_text())
    for section, sub in integ._FILE_SECTIONS:
        block = assets.get(section, {})
        for rel, rec in (block.get(sub, {}) if sub else block).items():
            if isinstance(rec, dict) and "sha256" in rec:
                add(project_root / rel, rec["sha256"])
    for section, key in integ._SINGLE_FILE_SECTIONS:
        block = assets.get(section, {})
        if block.get(key) and block.get(f"{key}_sha256"):
            add(project_root / block[key], block[f"{key}_sha256"])
    ident = json.loads((source_root / "cache/attempt_001/cache_identity.json").read_text())
    for rec in ident["schedule_rows"]:
        add(source_root / "cache/attempt_001/schedules" / f"{rec['demo']}__{rec['key']}.npz",
            rec["sha256"])
    for model in MODELS:
        path = source_root / "s0" / model / "attempt_001/loss_history.json"
        if path.is_file():
            add(path, integ.sha256_file(path))
    return {"files": files,
            "assets_sha256": prereg["assets"]["a1_assets_sha256"],
            "counts": {"train_originals": len(prereg["selection"]["train24"]),
                       "val_originals": len(prereg["selection"]["val31"]),
                       "train_windows": sum(row["full_windows_F60_s30"] for row in manifest["rows"] if row["split"] == "train"),
                       "val_windows": sum(row["full_windows_F60_s30"] for row in manifest["rows"] if row["split"] == "val"),
                       "train_frames": sum(row["T"] for row in manifest["rows"] if row["split"] == "train"),
                       "total_frames": sum(row["T"] for row in manifest["rows"])}}


def _require_formal_contract(cfg, meta: dict, rosters: dict, counts: dict) -> None:
    import dataclasses

    from ireset.data import w1_training as w1
    from ireset.train.trainer import TrainConfig
    from ireset.train.losses import LossWeights

    expected = dataclasses.asdict(TrainConfig(
        model_type="diffusion", steps=4000, micro_batch=4, grad_accum=4,
        lr=1e-4, weight_decay=1e-5, grad_clip=1.0, seed=20261008,
        val_seed=20261008, num_workers=4, val_every=250, log_every=25,
        device="cuda", d_model=240, n_layers=4, n_heads=6, ffn_mult=4,
        time_window=31, num_train_timesteps=1000, beta_schedule="scaled_linear",
        beta_start=1e-4, beta_end=0.02,
        loss=LossWeights(w_pos=1, w_rot6d=1, w_joint=1, w_geodesic=.5,
                         w_fk=2, w_contact=0, w_vel=.5)))
    if dataclasses.asdict(cfg) != expected:
        raise SystemExit("shortened or altered configuration cannot enter formal registration")
    contract = meta.get("formal_validation_contract", {})
    if (contract.get("loss_validation_every_optimizer_steps") != 250
            or contract.get("full_generation_every_optimizer_steps") != 1000
            or contract.get("full_generation_seed") != MAIN_SEED
            or contract.get("full_generation_expected_set") != "val31"):
        raise SystemExit("formal validation contract differs from the registered plan")
    if (len(rosters["train24"]) != 24 or len(rosters["val31"]) != 31
            or len(rosters["val6"]) != 6
            or rosters["train24_canonical_sha256"] != w1.W1_PLAN_TRAIN24_SHA
            or counts != {"train_originals": 24, "val_originals": 31,
                          "train_windows": 110, "val_windows": 145,
                          "train_frames": 4351, "total_frames": 10032}):
        raise SystemExit("formal data roster or counts differ from the accepted SOURCE")


def build_execution_registration(project_root: Path, source_root: Path,
                                 plan_rel: str, *, environment: dict | None = None) -> dict:
    """The R1 payload: plan/implementation/env identity, SOURCE anchors,
    rosters, both models' effective config, evaluation contract, budgets and
    stage commands. Every value is MEASURED here — nothing is copied from a
    prior report's claims."""
    import dataclasses

    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1

    legs = _registration_source_legs(project_root, source_root)
    manifest_path = (source_root
                     / "audit/attempt_001/training_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    prereg_path = project_root / manifest["preregistration"]["path"]
    prereg = json.loads(prereg_path.read_text())
    impl = implementation_identity(project_root)

    from train_generation import load_train_config

    config_path = project_root / TRAIN_CONFIG_REL
    cfg, meta = load_train_config(config_path)
    configs = {}
    for model in MODELS:
        mc = dataclasses.asdict(cfg)
        mc["model_type"] = model
        configs[model] = mc
    rosters = {
        "train24": list(manifest["selection"]["train24"]),
        "train24_canonical_sha256": w1.canonical_list_sha256(
            manifest["selection"]["train24"]),
        "val31": list(manifest["selection"]["val31"]),
        "val6": list(prereg["selection"]["val6"]),
    }
    from ireset.repair.validation import IndependentValidator
    from ireset.repair.config_io import load_mode_configs
    from ireset.embodied.contact_sites import infer_reference_schedule
    import inspect

    expanded = _expanded_source_files(project_root, source_root)
    _require_formal_contract(cfg, meta, rosters, expanded["counts"])
    validation_cfg = load_mode_configs("strict")[1]
    validator = IndependentValidator(cfg=validation_cfg)
    pair_record = dataclasses.asdict(validator.pairs)
    pair_record = json.loads(json.dumps(pair_record))
    reference_defaults = {key: parameter.default
                          for key, parameter in inspect.signature(infer_reference_schedule).parameters.items()
                          if parameter.default is not inspect.Parameter.empty}
    budget_domains = {
        name: {"stages": list(stages), "budget_s": budget, "reserve_s": reserve}
        for name, (stages, budget, reserve) in DOMAINS.items()}
    reg = {
        "execution_id": W1_DE_EXECUTION_ID,
        "label_protocol": w1.MOTION_PROTOCOL_VERSION,
        "source_experiment_id": w1.EXPERIMENT_ID,
        "training_purpose": w1.TRAINING_PURPOSE,
        "plan": {"path": plan_rel,
                 "sha256": integ.sha256_file(project_root / plan_rel)},
        "implementation": impl,
        "environment": (environment if environment is not None else env_record(project_root)),
        "condition_code_sha256": w1.condition_code_sha256(project_root),
        "assets_sha256": expanded["assets_sha256"],
        "source_files": expanded["files"],
        "data_counts": expanded["counts"],
        "source_run_root": {
            "path": integ.path_for_record(source_root),
            "anchors": legs["source_anchors"],
        },
        "protocol_anchors": legs["plan_anchors"],
        "training_manifest": {
            "path": integ.path_for_record(manifest_path),
            "sha256": integ.sha256_file(manifest_path)},
        "preregistration": {
            "path": integ.path_for_record(prereg_path),
            "sha256": integ.sha256_file(prereg_path)},
        "cache": {
            "dir": integ.path_for_record(source_root / "cache/attempt_001"),
            "identity_sha256": integ.sha256_file(
                source_root / "cache/attempt_001/cache_identity.json"),
            "root_stats_sha256": integ.sha256_file(
                source_root / "cache/attempt_001/root_stats.npz")},
        "rosters": rosters,
        "configs": {
            "config_file": TRAIN_CONFIG_REL,
            "config_file_sha256": integ.sha256_file(config_path),
            "deterministic": configs["deterministic"],
            "diffusion": configs["diffusion"],
            "formal_validation_contract": meta.get("formal_validation_contract"),
        },
        "evaluation": {
            "validation_mode": "strict",
            "validation_config": dataclasses.asdict(validation_cfg),
            "exclude": ["box_ground"],
            "collision_pairs": pair_record,
            "reference_schedule_defaults": reference_defaults,
            "sampling": {"window": 60, "stride": 30, "ddim_steps": 25,
                         "eta": 0.0, "resample_attempts": 3},
            "required_metrics": ["fitting", "task", "robot_strict", "penetration",
                                 "box_ground", "feet", "hands", "both",
                                 "ref_frames", "reference_intervals", "stitch", "contact_f1"],
            "task_gate": {
                "frame": "world",
                "tol_m": TASK_MPJPE_TOL_M,
                "semantics": "per-frame mean keypoint error <= tol for every "
                             "frame (equivalent to max <= tol); box-frame "
                             "series reported alongside, never a substitute",
            },
            "methods": list(METHODS),
            "main_seed": MAIN_SEED,
            "appendix_seeds": list(APPENDIX_SEEDS),
            "expected_matrix": {
                "main_keys": len(METHODS) * len(rosters["val31"]),
                "appendix_keys": len(APPENDIX_SEEDS) * len(rosters["val31"]),
            },
            "box_ground": "excluded from robot_strict (fixed task input); "
                          "measured per row (depth/ratio/position) and "
                          "declared in every aggregate",
        },
        "budgets": {"domains": budget_domains,
                    "policy": "monotonic perf_counter; remaining = budget - "
                              "three-way-reconciled historical; unknown or "
                              "running residue = stop; overrun flips the "
                              "same attempt's complete registration to "
                              "incomplete in place"},
        "rules": {
            "best_selection": "demonstration-macro teacher-fit FK MPJPE "
                              "over the FULL registered val31, complete "
                              "evidence only, strictly decreasing, ties keep "
                              "the earlier step",
            "no_qualified_best": "E consumes last.pt with "
                                 "checkpoint_role=last_diagnostic; missing "
                                 "too -> checkpoint_unavailable rows",
            "no_auto_retry": "one formal attempt per model by default; a "
                             "new attempt requires --allow-new-attempt and "
                             "still pays the same budget",
            "zero_repair": "no RepairPipeline construction or invocation "
                           "anywhere in D/E",
        },
        "historical_anchors": {
            "u0_verdict": "incomplete", "ur_verdict": "no_quality_gain",
            "note": "source-selection verdicts stand unchanged; W1-D/E is "
                    "the registered weak-motion diagnostic round",
        },
    }
    if not str(reg["environment"].get("python", "")).startswith("3.12."):
        legs["problems"].append("python_not_3_12")
    if not reg["environment"].get("cuda_available") and configs["diffusion"]["device"] == "cuda":
        legs["problems"].append("cuda_unavailable")
    if configs["diffusion"]["device"] == "cuda" and reg["environment"].get("cuda_forward_backward_probe_ok") is not True:
        legs["problems"].append("cuda_probe_failed")
    reg["anchor_problems"] = legs["problems"]
    return reg


def verify_execution_registration(run_root: Path, project_root: Path) -> list[str]:
    """Re-verify the registration against CURRENT facts at every stage entry
    (§2.3: registration binds the run; drift means a new registration, not a
    silent continuation)."""
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1

    reg_path = run_root / "freeze" / "execution_registration.json"
    if not reg_path.is_file():
        return ["execution_registration_missing"]
    problems = []
    try:
        reg = json.loads(reg_path.read_text())
        if not isinstance(reg, dict):
            return ["execution_registration_not_a_mapping"]
        source_root = project_root / reg["source_run_root"]["path"]
        plan_rel = reg["plan"]["path"]
        if not source_root.is_dir():
            return ["source_run_root_missing"]
        rebuilt = build_execution_registration(project_root, source_root, plan_rel)
        problems.extend(rebuilt.pop("anchor_problems"))
        rebuilt["stage_commands"] = _stage_commands(run_root, source_root)
        for key in sorted(set(rebuilt) | set(reg)):
            if rebuilt.get(key) != reg.get(key):
                problems.append(f"registration_{key}_not_reproducible")
        marker = run_root / "freeze/execution_scope.json"
        if not marker.is_file():
            problems.append("execution_scope_missing")
        elif run_root.resolve() not in _history_roots(run_root):
            problems.append("execution_scope_not_registered")
        preflight, preflight_problems = stage_state(run_root, STAGES["register"], project_root)
        problems.extend(f"preflight:{p}" for p in preflight_problems)
        if preflight is None or preflight.get("complete") is not True or preflight.get("passed") is not True:
            problems.append("preflight_not_complete")
        elif not preflight_problems:
            report = json.loads((project_root / preflight["report"]).read_text())
            if (not isinstance(report, dict) or report.get("passed") is not True
                    or report.get("budget_stopped") is True or report.get("reasons") != []):
                problems.append("preflight_report_not_passed")
        prereg = json.loads((project_root / reg["preregistration"]["path"]).read_text())
        from ireset.paths import TEACHER_TRAJ_DIR, OMOMO_NEW_DIR

        check = w1.verify_preregistration(prereg, project_root, TEACHER_TRAJ_DIR,
                                         OMOMO_NEW_DIR, verify_assets=False)
        problems.extend(f"preregistration:{p}" for p in check["problems"])
        manifest = json.loads((project_root / reg["training_manifest"]["path"]).read_text())
        cache = w1.verify_cache_contents(project_root / reg["cache"]["dir"],
                                         manifest, prereg, project_root)
        problems.extend(f"cache:{p}" for p in cache["problems"])
    except (OSError, ValueError, TypeError, KeyError, SystemExit) as exc:
        problems.append(f"registration_unverifiable:{type(exc).__name__}:{exc}")
    return problems


# --- best selection recompute (§5.3) --------------------------------------------

def recompute_best_selection(formal_history: list[dict],
                             expected_n: int | None = None, *,
                             expected_demos: list[str] | None = None,
                             demo_lengths: dict[str, int] | None = None,
                             every: int | None = None,
                             seed: int | None = None) -> tuple[int | None, float | None]:
    """Recompute the §5.3 best from the per-demo rows — never from the
    recorded ``complete`` flag or macro alone.

    A complete evaluation has every expected demo exactly once with status
    ``ok`` and a finite ``mpjpe_m``; its macro is the mean of those values
    and must equal the recorded macro. Best = the first evaluation whose
    recomputed macro strictly decreases (ties keep the earlier step)."""
    best_step, best_macro = None, None
    previous_step = 0
    for rec in formal_history:
        if not isinstance(rec, dict):
            continue
        step = rec.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= previous_step:
            continue
        previous_step = step
        if every is not None and step % every != 0:
            continue
        if seed is not None and rec.get("seed") != seed:
            continue
        per_demo = rec.get("per_demo")
        if not isinstance(per_demo, list) or not per_demo or not all(isinstance(r, dict) for r in per_demo):
            continue
        demos = [r.get("demo") for r in per_demo]
        if any(not isinstance(d, str) for d in demos) or len(set(demos)) != len(demos):
            continue
        n_expected = len(expected_demos) if expected_demos is not None else expected_n
        if n_expected is None:
            n_expected = rec.get("n_expected")
        if (not isinstance(n_expected, int) or isinstance(n_expected, bool)
                or n_expected <= 0 or len(per_demo) != n_expected
                or rec.get("n_expected") != n_expected):
            continue
        if expected_demos is not None and set(demos) != set(expected_demos):
            continue
        vals = []
        valid = True
        for row in per_demo:
            v = row.get("mpjpe_m")
            if (row.get("status") != "ok" or not _valid_wall(v)
                    or (expected_demos is not None and row.get("numerical_valid") is not True)
                    or row.get("numerical_valid") is False
                    or (demo_lengths is not None and (not isinstance(row.get("T"), int)
                        or isinstance(row["T"], bool) or row["T"] != demo_lengths.get(row["demo"])))):
                valid = False
                break
            vals.append(float(v))
        if not valid:
            continue
        macro = sum(vals) / len(vals)
        recorded = rec.get("macro_mpjpe_m")
        if not _valid_wall(recorded) or abs(float(recorded) - macro) > 1e-12:
            continue
        if best_macro is None or macro < best_macro:
            best_step, best_macro = step, macro
    return best_step, best_macro


def build_checkpoint_selection(run_root: Path, project_root: Path) -> dict:
    """§6.1: recompute the §5.3 selection per model from formal_history and
    the on-disk checkpoints; the evaluator consumes THIS record, never a
    glob. Returns {"models": {...}, "problems": [...]}."""
    from ireset.data import a2_integrity as integ

    reg = json.loads((run_root / "freeze/execution_registration.json").read_text())
    manifest = json.loads((project_root / reg["training_manifest"]["path"]).read_text())
    roster = reg["rosters"]["val31"]
    lengths = {r["demo"]: r["T"] for r in manifest["rows"] if r["demo"] in roster}
    every = reg["configs"]["formal_validation_contract"]["full_generation_every_optimizer_steps"]
    out = {"models": {}, "problems": [], "execution_registration_sha256": _registration_sha(run_root)}
    for model in MODELS:
        sel = {"model": model, "role": "unavailable", "reason": "checkpoint_unavailable"}
        try:
            row, state_problems = stage_state(run_root, STAGES[f"train-{model}"], project_root)
            if row is None or state_problems:
                out["problems"].extend(f"{model}:{p}" for p in state_problems)
                out["models"][model] = sel
                continue
            att = row["attempt"]
            att_dir = run_root / "formal" / model / att
            summary = json.loads((att_dir / "summary.json").read_text())
            formal_history = json.loads((att_dir / "formal_history.json").read_text())
            if formal_history != summary.get("formal_history"):
                raise ValueError("formal_history_summary_disagreement")
            best_step, best_macro = recompute_best_selection(
                formal_history, expected_demos=roster, demo_lengths=lengths,
                every=every, seed=reg["evaluation"]["main_seed"])
            entries = json.loads((att_dir / "checkpoint_index.json").read_text())
            if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
                raise ValueError("checkpoint_index_malformed")
            names = [e.get("name") for e in entries]
            if len(names) != len(set(names)):
                raise ValueError("checkpoint_index_duplicate")
            ck_index = {e["name"]: e for e in entries}
            name, step = (("best.pt", best_step) if best_step is not None
                          else ("last.pt", summary.get("steps")))
            path = att_dir / name
            index_rec = ck_index.get(name, {})
            sha = integ.sha256_file(path) if path.is_file() else None
            if sha is None or index_rec.get("sha256") != sha or index_rec.get("step") != step:
                raise ValueError(f"{name}_checkpoint_inconsistent")
            import torch

            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck.get("config") != reg["configs"][model] or ck.get("step") != step:
                raise ValueError("checkpoint_config_or_step_mismatch")
            ident = ck.get("identity", {})
            if (ident.get("model_type") != model
                    or ident.get("execution_registration", {}).get("sha256") != _registration_sha(run_root)):
                raise ValueError("checkpoint_execution_identity_mismatch")
            sel = {"model": model, "formal_attempt": att, "verdict": row.get("verdict"),
                   "steps": summary.get("steps"), "role": "best" if best_step is not None else "last",
                   "checkpoint_role": "best" if best_step is not None else "last_diagnostic",
                   "path": _record_path(path, project_root), "sha256": sha, "step": step,
                   "formal_macro_mpjpe_m": best_macro,
                   "selection_reason": "recomputed_formal_macro_strict_decrease" if best_step is not None else "no_qualified_best"}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            out["problems"].append(f"{model}:selection_unverifiable:{exc}")
            sel["reason"] = f"selection_unverifiable:{exc}"
        out["models"][model] = sel
    return out


# --- R0 preflight ---------------------------------------------------------------

def run_r0_preflight(source_root: Path, project_root: Path) -> dict:
    """§2.2 R0: re-verify the SOURCE chain's five gates through the SAME
    checkers the readiness entry used, plus the anchor legs and the
    environment record. Both the old reports' flags and this fresh
    recomputation land in the record; only the recomputation gates R1."""
    import w1_readiness as readiness

    legs = _registration_source_legs(project_root, source_root)
    record: dict = {"anchors": {**legs["source_anchors"], **legs["plan_anchors"]},
                    "anchor_problems": legs["problems"]}
    reasons: list[str] = list(legs["problems"])
    checks = {}
    try:
        cache = readiness.check_cache_binding(source_root)
        prereg = json.loads((source_root / "freeze/preregistration.json").read_text())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        cache, prereg = None, {}
        reasons.append(f"source_binding:{type(exc).__name__}:{exc}")
    calls = (("registration", readiness.check_registration, (source_root,)),
             ("a3", readiness.check_a3, (source_root,)),
             ("b", readiness.check_b, (source_root, cache)),
             ("c", readiness.check_c, (source_root, cache, prereg)))
    for name, fn, fn_args in calls:
        try:
            check = fn(*fn_args)
        except (Exception, SystemExit) as exc:
            check = {"passed": False, "reasons": [f"unverifiable:{type(exc).__name__}:{exc}"]}
        checks[name] = check
        if check.get("passed") is not True:
            reasons += [f"{name}:{r}" for r in check.get("reasons", ["failed_without_reason"])]
    # the readiness record's own flags are recorded as the PRIOR claims
    readiness_path = source_root / "decisions/attempt_001/w1_readiness.json"
    record["prior_readiness_claim"] = (json.loads(readiness_path.read_text())
                                       if readiness_path.is_file() else None)
    record["checks"] = checks
    record["environment"] = env_record(project_root)
    record["passed"] = not reasons
    record["reasons"] = reasons
    return record


# --- stage: register -------------------------------------------------------------

def _begin_attempt(run_root: Path, stage: str, extra: dict | None = None) -> Path:
    from ireset.data import a2_integrity as integ

    attempt_dir = integ.allocate_attempt(Path(run_root) / STAGE_DIRS[stage])
    model = stage.removeprefix("w1_de_train_") if stage.startswith("w1_de_train_") else None
    scope = {"stage": stage, "model": model, "attempt": attempt_dir.name}
    _upsert_cost(run_root, stage, attempt_dir.name, 0.0, extra or {}, "running")
    append_stage_index(attempt_dir.parent, {
        "attempt": attempt_dir.name, "scope": scope, "status": "running",
        "complete": False, "measured_wall_s": 0.0,
        "execution_registration_sha256": _registration_sha(run_root), **(extra or {})})
    return attempt_dir


def stage_register(args) -> int:
    from ireset.data import a2_integrity as integ
    from ireset.paths import PROJECT_ROOT, path_for_record

    t0 = time.perf_counter()
    source_root = integ.guard_output_dir(args.source_run_root)
    run_root = integ.guard_output_dir(args.run_root)
    if run_root.parent != (PROJECT_ROOT / "derived/experiments/omniretarget_first").resolve():
        raise SystemExit("run root must be a new experiment directly under derived/experiments/omniretarget_first")
    if source_root == run_root or source_root in run_root.parents or run_root in source_root.parents:
        raise SystemExit("run root must not overlap the SOURCE run root")
    if run_root.exists() and any(run_root.iterdir()) and not (run_root / "freeze/execution_scope.json").is_file():
        raise SystemExit("run root contains historical outputs without this execution scope")
    budget = _budget_state(run_root, "r0r1")
    _claim_execution_root(run_root, PROJECT_ROOT, source_root, args.plan)
    stage = STAGES["register"]
    attempt_dir = _begin_attempt(run_root, stage)
    report_path = attempt_dir / "preflight_report.json"
    row = {"report": path_for_record(report_path), "passed": False, "complete": False}
    try:
        print(f"[register] R0 SOURCE={path_for_record(source_root)}", flush=True)
        preflight = run_r0_preflight(source_root, PROJECT_ROOT)
        if time.perf_counter() - t0 >= budget["remaining_s"] - budget["reserve_s"]:
            preflight["passed"] = False
            preflight["reasons"].append("budget_deadline")
        atomic_json(report_path, preflight)
        if not preflight["passed"]:
            finish_attempt(run_root, stage, attempt_dir, t0, budget, row, {"passed": False})
            print(json.dumps({"r0_passed": False, "reasons": preflight["reasons"]}))
            return 2
        reg = build_execution_registration(PROJECT_ROOT, source_root, args.plan,
                                           environment=preflight["environment"])
        if reg.pop("anchor_problems"):
            raise ValueError("source anchors changed after preflight")
        reg["stage_commands"] = _stage_commands(run_root, source_root)
        reg_path = run_root / "freeze/execution_registration.json"
        if reg_path.exists():
            existing = json.loads(reg_path.read_text())
            if existing != reg:
                diff = sorted(k for k in set(existing) | set(reg) if existing.get(k) != reg.get(k))
                raise SystemExit(f"execution registration exists and differs ({diff}); prior bytes retained")
            reused = True
        else:
            atomic_json(reg_path, reg, exclusive=True)
            reused = False
        row.update(passed=True, complete=True, registration_reused=reused,
                   registration=path_for_record(reg_path))
        passed = finish_attempt(run_root, stage, attempt_dir, t0, budget, row,
                                {"passed": True, "registration_reused": reused})
        print(json.dumps({"r0_passed": passed, "registration_reused": reused,
                          "registration": path_for_record(reg_path), "budget": budget}), flush=True)
        return 0 if passed else 2
    except BaseException as exc:
        if not report_path.exists():
            atomic_json(report_path, {"passed": False, "reasons": [f"{type(exc).__name__}:{exc}"]})
        crash_attempt(run_root, stage, attempt_dir, t0, budget, row, {}, exc)
        raise


def _stage_commands(run_root: Path, source_root: Path) -> dict:
    """The concrete argv each stage will spawn (recorded BEFORE the runs;
    the actual spawned argv is recorded again per attempt)."""
    from ireset.paths import path_for_record

    reg = path_for_record(run_root / "freeze" / "execution_registration.json")
    manifest = path_for_record(
        source_root / "audit/attempt_001/training_manifest.json")
    cache = path_for_record(source_root / "cache/attempt_001")
    cmds = {}
    for model in MODELS:
        cmds[f"train-{model}"] = [
            "python", "scripts/train_generation.py", "--model", model,
            "--training-manifest", manifest,
            "--config", TRAIN_CONFIG_REL, "--cache-dir", cache,
            "--out", path_for_record(run_root / "formal" / model),
            "--execution-registration", reg,
            "--budget-deadline-s", "<remaining-reserve>"]
    cmds["raw-eval"] = [
        "python", "scripts/eval_generation.py", "--w1-de-raw",
        "--execution-registration", reg,
        "--checkpoint-selection",
        path_for_record(run_root / "evaluation/checkpoint_selection.json"),
        "--training-manifest", manifest, "--cache-dir", cache,
        "--out", "<raw-attempt>/raw_eval.json",
        "--products-dir", "<raw-attempt>/products",
        "--budget-deadline-s", "<remaining-reserve>"]
    return cmds


# --- stage: train ---------------------------------------------------------------

def _spawn(argv: list[str], project_root: Path, timeout_s: float,
           log_path: Path) -> tuple[int | None, str, float]:
    """Spawn a stage subprocess with a HARD timeout; the parent's measured
    elapsed (spawn -> exit) is the cost anchor (§8.2)."""
    t0 = time.perf_counter()
    if not _valid_wall(timeout_s) or timeout_s <= 0:
        return None, "timeout", time.perf_counter() - t0
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    with open(log_path, "x") as log:
        proc = subprocess.Popen(argv, cwd=str(project_root), env=env,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
            outcome = "exit"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code, outcome = None, "timeout"
    return code, outcome, time.perf_counter() - t0


def _classify_train_verdict(returncode: int | None, summary: dict | None) -> str:
    if summary is None:
        return "crashed" if returncode is None else "crashed"
    if summary.get("stop_reason") == "budget_deadline" or summary.get("budget_stopped"):
        return "budget_stopped"
    if str(summary.get("stop_reason", "")).startswith(("nonfinite", "degenerate")):
        return "nonfinite_stopped"
    if returncode != 0:
        return "crashed"
    if (summary.get("complete") is True and summary.get("stop_reason") is None
            and summary.get("steps") == summary.get("config", {}).get("steps")):
        return "completed"
    return "incomplete"


def stage_train(args, model: str) -> int:
    from ireset.data import a2_integrity as integ
    from ireset.paths import PROJECT_ROOT, path_for_record

    t0 = time.perf_counter()
    run_root = integ.guard_output_dir(args.run_root)
    problems = verify_execution_registration(run_root, PROJECT_ROOT)
    if problems:
        raise SystemExit(f"execution registration problems: {problems}")
    reg = json.loads((run_root / "freeze/execution_registration.json").read_text())
    if model == "diffusion":
        prior_det, det_problems = stage_state(run_root, STAGES["train-deterministic"], PROJECT_ROOT)
        if prior_det is None:
            raise SystemExit("train-diffusion requires terminal train-deterministic evidence")
        if det_problems or prior_det.get("public_integrity_failure"):
            raise SystemExit(f"train-deterministic public integrity unverified: {det_problems}")
    budget = _budget_state(run_root, "d")
    stage = STAGES[f"train-{model}"]
    stage_root = run_root / STAGE_DIRS[stage]
    prior = _index_rows(stage_root)
    if prior and not getattr(args, "allow_new_attempt", False):
        raise SystemExit(f"prior {stage} attempt exists; explicit --allow-new-attempt required")
    attempt_dir = _begin_attempt(run_root, stage, {"model": model})
    summary_path = attempt_dir / "summary.json"
    row = {"model": model, "summary": path_for_record(summary_path), "complete": False}
    try:
        work_remaining = budget["remaining_s"] - budget["reserve_s"] - (time.perf_counter() - t0)
        argv = [sys.executable, "scripts/train_generation.py", "--model", model,
                "--training-manifest", str(PROJECT_ROOT / reg["training_manifest"]["path"]),
                "--config", str(PROJECT_ROOT / TRAIN_CONFIG_REL),
                "--cache-dir", str(PROJECT_ROOT / reg["cache"]["dir"]),
                "--out", str(attempt_dir),
                "--execution-registration", str(run_root / "freeze/execution_registration.json"),
                "--budget-deadline-s", f"{max(work_remaining, 0.0):.3f}"]
        row["argv"] = argv
        atomic_json(attempt_dir / "driver_call.json", {"argv": argv, "budget": budget})
        print(f"[train-{model}] {attempt_dir.name} remaining={work_remaining:.1f}s", flush=True)
        code, outcome, child_elapsed = _spawn(argv, PROJECT_ROOT,
                    max(0.0, budget["remaining_s"] - budget["reserve_s"] - (time.perf_counter() - t0)),
                    attempt_dir / "driver_subprocess.log")
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            summary = {"model_type": model, "steps": 0,
                       "stop_reason": "driver_timeout" if outcome == "timeout" else "crashed_before_summary",
                       "budget_stopped": outcome == "timeout",
                       "selection_status": "no_qualified_best", "identity": None}
            atomic_json(summary_path, summary)
        verdict = _classify_train_verdict(code, summary)
        public_failure = summary.get("identity") is None or str(summary.get("stop_reason", "")).startswith("nonfinite_input")
        row.update(verdict=verdict, complete=verdict == "completed",
                   steps=summary.get("steps"), target_steps=reg["configs"][model]["steps"],
                   selection_status=summary.get("selection_status"),
                   best_formal_step=summary.get("best_formal_step"),
                   best_formal_macro_mpjpe_m=summary.get("best_formal_macro_mpjpe_m"),
                   stop_reason=summary.get("stop_reason"), subprocess_outcome=outcome,
                   returncode=code, child_wall_s=child_elapsed,
                   public_integrity_failure=public_failure,
                   artifacts={path_for_record(p): integ.sha256_file(p)
                              for p in attempt_dir.iterdir()
                              if p.is_file() and p.suffix in (".json", ".pt") and p != summary_path})
        passed = finish_attempt(run_root, stage, attempt_dir, t0, budget, row,
                                 {"verdict": verdict, "steps": summary.get("steps")},
                                 "timeout" if outcome == "timeout" else "done")
        print(json.dumps({"model": model, "verdict": verdict,
                          "steps": summary.get("steps"), "selection_status": summary.get("selection_status"),
                          "best_formal_step": summary.get("best_formal_step")}), flush=True)
        return 0 if passed else 2
    except BaseException as exc:
        if not summary_path.exists():
            atomic_json(summary_path, {"model_type": model, "steps": 0,
                                      "stop_reason": f"driver_exception:{type(exc).__name__}",
                                      "selection_status": "no_qualified_best"})
        crash_attempt(run_root, stage, attempt_dir, t0, budget,
                      {**row, "public_integrity_failure": True}, {}, exc)
        raise


# --- stage: raw-eval -------------------------------------------------------------

def stage_raw_eval(args) -> int:
    from ireset.data import a2_integrity as integ
    from ireset.paths import PROJECT_ROOT, path_for_record

    t0 = time.perf_counter()
    run_root = integ.guard_output_dir(args.run_root)
    problems = verify_execution_registration(run_root, PROJECT_ROOT)
    if problems:
        raise SystemExit(f"execution registration problems: {problems}")
    for model in MODELS:
        state, state_problems = stage_state(run_root, STAGES[f"train-{model}"], PROJECT_ROOT)
        if state is None or state_problems:
            raise SystemExit(f"raw-eval requires verified terminal {model}: {state_problems}")
    budget = _budget_state(run_root, "e")
    stage = STAGES["raw-eval"]
    prior = _index_rows(run_root / STAGE_DIRS[stage])
    if prior and not getattr(args, "allow_new_attempt", False):
        raise SystemExit("prior raw-eval attempt exists; --allow-new-attempt required")
    attempt_dir = _begin_attempt(run_root, stage)
    report_path = attempt_dir / "raw_eval.json"
    row = {"complete": False, "report": path_for_record(report_path)}
    try:
        selection = build_checkpoint_selection(run_root, PROJECT_ROOT)
        sel_path = run_root / "evaluation/checkpoint_selection.json"
        if sel_path.exists():
            if json.loads(sel_path.read_text()) != selection:
                raise ValueError("checkpoint selection changed; existing evidence retained")
        else:
            atomic_json(sel_path, selection, exclusive=True)
        reg = json.loads((run_root / "freeze/execution_registration.json").read_text())
        work_remaining = budget["remaining_s"] - budget["reserve_s"] - (time.perf_counter() - t0)
        argv = [sys.executable, "scripts/eval_generation.py", "--w1-de-raw",
                "--execution-registration", str(run_root / "freeze/execution_registration.json"),
                "--checkpoint-selection", str(sel_path),
                "--training-manifest", str(PROJECT_ROOT / reg["training_manifest"]["path"]),
                "--cache-dir", str(PROJECT_ROOT / reg["cache"]["dir"]),
                "--out", str(report_path), "--products-dir", str(attempt_dir / "products"),
                "--budget-deadline-s", f"{max(work_remaining, 0.0):.3f}"]
        row["argv"] = argv
        atomic_json(attempt_dir / "driver_call.json", {"argv": argv, "budget": budget})
        print(f"[raw-eval] {attempt_dir.name} remaining={work_remaining:.1f}s", flush=True)
        code, outcome, child_elapsed = _spawn(argv, PROJECT_ROOT,
                    max(0.0, budget["remaining_s"] - budget["reserve_s"] - (time.perf_counter() - t0)),
                    attempt_dir / "driver_subprocess.log")
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {"completeness": {"complete": False}, "rows": [],
                      "failure": "driver_timeout" if outcome == "timeout" else "crashed_before_report"}
            atomic_json(report_path, report)
        from eval_generation import w1_de_matrix_completeness

        completeness = w1_de_matrix_completeness(report.get("rows", []), reg,
                            budget_stopped=report.get("budget_stopped", False) or outcome == "timeout")
        complete = code == 0 and outcome == "exit" and completeness.get("complete") is True
        row.update(complete=complete, subprocess_outcome=outcome, returncode=code,
                   child_wall_s=child_elapsed, counts=report.get("matrix", {}),
                   completeness=completeness,
                   artifacts={path_for_record(sel_path): integ.sha256_file(sel_path)})
        passed = finish_attempt(run_root, stage, attempt_dir, t0, budget, row, {},
                                 "timeout" if outcome == "timeout" else "done")
        print(json.dumps({"raw_eval_complete": passed, "selection_problems": selection["problems"]}), flush=True)
        return 0 if passed else 2
    except BaseException as exc:
        if not report_path.exists():
            atomic_json(report_path, {"completeness": {"complete": False}, "rows": [],
                                     "failure": f"{type(exc).__name__}:{exc}"})
        crash_attempt(run_root, stage, attempt_dir, t0, budget, row, {}, exc)
        raise


# --- main ------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("register", "train-deterministic",
                             "train-diffusion", "raw-eval", "detail",
                             "finalize"))
    ap.add_argument("--source-run-root", type=Path, default=None)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--plan", type=str,
                    default="docs/plans/omniretarget_w1_de_plan.md")
    ap.add_argument("--allow-new-attempt", action="store_true",
                    help="explicit new attempt when one already exists (the "
                         "prior cost stays in the domain budget)")
    args = ap.parse_args()
    from ireset.paths import PROJECT_ROOT

    with execution_lock(PROJECT_ROOT):
        if args.stage == "register":
            if args.source_run_root is None:
                raise SystemExit("--stage register requires --source-run-root")
            return stage_register(args)
        if args.stage.startswith("train-"):
            return stage_train(args, args.stage[len("train-"):])
        if args.stage == "raw-eval":
            return stage_raw_eval(args)
        if args.stage == "detail":
            from omniretarget_w1_de_detail import stage_detail
            return stage_detail(args)
        if args.stage == "finalize":
            from omniretarget_w1_de_finalize import stage_finalize
            return stage_finalize(args)
    raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
