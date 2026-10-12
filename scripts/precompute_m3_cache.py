"""Precompute M3 caches: reference contact schedules + train root statistics.

Per NPZ: the reference-side contact schedule (hull-distance hysteresis,
plan 8.1 thresholds) is inferred once and stored compactly under
``derived/data/m3_cache/schedules/``. Then train-split clip-frame root
statistics are fitted and saved (``root_stats.npz``), and teacher
joint-limit violations are summarized (plan 7.2: record, never silently
clip).

W1 mode (omniretarget_w1_a3bc_plan §4.2): with ``--training-manifest`` and
``--out-dir`` the entry computes an IDENTITY-KEYED, manifest-bound cache
under the W1 run root — per-demo reference schedules keyed by the actual
input hashes + scene-split mapping + assets + protocol + condition-code
identity (the same file NAME never proves cache reusability), and root
statistics fitted ONLY over the original frames of the admitted train24
rows (val/test and rejected files never contribute). The cache directory is
attempt-scoped; the identity manifest binds preregistration + training
manifest + code + stats definition, and consumers refuse drifted or missing
legs rather than falling back.

Run (legacy): uv run --locked python scripts/precompute_m3_cache.py [--workers 4]
Run (W1):    uv run --locked python scripts/precompute_m3_cache.py \
               --training-manifest <run>/audit/attempt_NNN/training_manifest.json \
               --out-dir <run>/cache
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data import a2_integrity as integ
from ireset.data import w1_training as w1
from ireset.data.teacher import build_interaction_clip, schedule_to_cache, world_to_clip
from ireset.data.windows import CACHE_DIR, load_split_files
from ireset.model.tokens import RootStats
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record
from ireset.resctrl import (
    DEFAULT_CPU_BUDGET,
    apply_native_threads,
    execution_record,
    parallel_map,
    resolve_workers,
)


def process_file(job: tuple[str, bool]) -> dict:
    """One NPZ -> schedule cache row. The job is a single tuple so the
    function stays a module-level picklable task for the spawn pool
    (cpu_parallel_optimization_plan section 2)."""
    name, force = job
    out = CACHE_DIR / "schedules" / f"{name}.npz"
    clip = build_interaction_clip(name)   # infers schedule when no cache hit
    if force or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, intervals=schedule_to_cache(clip.condition.schedule))
    v = clip.validation
    return {
        "npz": name, "T": clip.T,
        "joint_limit_max_violation_rad": v["joint_limit_max_violation_rad"],
        "joint_limit_violating_entries": v["joint_limit_violating_entries"],
        "clamped": v["clamped"], "drop": v["drop"],
    }


def _w1_schedule_job(job: tuple) -> dict:
    """One W1 admitted row -> keyed schedule cache file (spawn-picklable).

    Derives the schedule from the SOURCE-scene box through the audited
    derived representation — pose AND geometry scale (review fix 1: the
    source hull is the locked largebox scaled by the fitted similarity
    scale); the output filename carries the identity key so a drifted
    input/code/asset can never silently reuse a stale cache.
    """
    (demo, derived_path, key, out_root) = job
    out = Path(out_root) / "schedules" / w1.schedule_cache_filename(demo, key)
    with np.load(derived_path, allow_pickle=False) as z:
        human = z["human_joints"]
        src_box = z["source_obj_pose"]
        fps = float(z["fps"])
        scale = float(np.asarray(z["similarity_scale"]).reshape(()))
    from ireset.embodied.contact_sites import infer_reference_schedule

    schedule = infer_reference_schedule(human, src_box, fps=fps,
                                        box_scale=scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, intervals=schedule_to_cache(schedule))
    import hashlib as _hl

    return {"demo": demo, "T": int(human.shape[0]),
            "cache": str(out.name), "status": "ok",
            "sha256": _hl.sha256(out.read_bytes()).hexdigest()}


def _worker_init(native_threads: int) -> None:
    apply_native_threads(native_threads)


def _upsert_cost(run_root: Path, stage: str, attempt: str, wall: float,
                 extra: dict, status: str) -> None:
    """Append-or-UPDATE the ledger row for one (stage, attempt).

    Exactly one row per attempt EVER exists: pre-registered ``running``
    before any work and updated in place on every exit path (round-3 review
    fix 8 mirrors the w1_s0 contract — a crashed cache attempt can never be
    left without a ledger row)."""
    costs_path = run_root / "costs.json"
    entries = []
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    row = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
           "stage": stage, "attempt": attempt, "status": status,
           "wall_s": round(wall, 1), **extra}
    for i, e in enumerate(entries):
        if e.get("stage") == stage and e.get("attempt") == attempt:
            entries[i] = {**row, "timestamp": e.get("timestamp")}
            break
    else:
        entries.append(row)
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


def _historical_cache_cost_s(out_root: Path) -> float | None:
    """Sum of PRIOR w1-cache stage walls, reconciled THREE ways.

    Round-3 review fix 8: the attempt directories ON DISK, the attempts
    index and the ledger rows keyed by attempt must agree exactly — judging
    history by the index entry count alone let a crashed worker's
    unregistered attempt_001 wash into a zero-cost history (the next call
    then started attempt_002 as if nothing had been spent). ``None`` =
    UNKNOWN history (an attempt the index/ledger cannot account for, an
    unkeyed/duplicate row, or a leftover ``running`` row) — a stop
    condition: the registered CACHE_BUDGET_S cannot be proven unexhausted
    (round-2 review fix 7).
    """
    import math as _math

    disk = {d.name for d in out_root.glob("attempt_*") if d.is_dir()}
    idx_path = out_root / integ.ATTEMPTS_INDEX
    index = ({a.get("attempt")
              for a in json.loads(idx_path.read_text())["attempts"]}
             if idx_path.is_file() else set())
    entries = []
    costs_path = out_root.parent / "costs.json"
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    ledger: dict[str, float] = {}
    total = 0.0
    for e in entries:
        if e.get("stage") != "w1_cache":
            continue
        attempt = e.get("attempt")
        if attempt is None or attempt in ledger:
            return None          # unkeyed or duplicate rows: unknown
        wall = e.get("wall_s")
        if not (isinstance(wall, (int, float)) and _math.isfinite(wall)
                and wall >= 0):
            return None
        if e.get("status") == "running":
            return None          # the exit registration never ran
        ledger[attempt] = float(wall)
        total += float(wall)
    if not disk and not ledger:
        return 0.0
    if disk != index or disk != set(ledger):
        return None              # attempt without index/ledger coverage
    return total


def _register_crashed_cache_attempt(out_root: Path, attempt_dir: Path,
                                    exc: BaseException, wall: float) -> None:
    """Honest incomplete registration for a crashed cache attempt (round-3
    review fix 8): a minimal identity, an index row and a ``crashed`` ledger
    row — the attempt stays visible and its wall stays accounted, so the
    budget history remains KNOWABLE instead of washing to zero."""
    ident_path = attempt_dir / "cache_identity.json"
    if not ident_path.is_file():
        ident_path.write_text(json.dumps({
            "protocol": w1.MOTION_PROTOCOL_VERSION,
            "experiment_id": w1.EXPERIMENT_ID,
            "attempt": attempt_dir.name,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "complete": False,
            "incomplete_reason": f"crashed:{type(exc).__name__}",
        }, indent=2, ensure_ascii=False))
    idx_path = out_root / integ.ATTEMPTS_INDEX
    indexed = ({a.get("attempt")
                for a in json.loads(idx_path.read_text())["attempts"]}
               if idx_path.is_file() else set())
    if attempt_dir.name not in indexed:
        integ.append_attempts_index(out_root, {
            "attempt": attempt_dir.name, "complete": False,
            "incomplete_reason": f"crashed:{type(exc).__name__}",
            "identity": path_for_record(ident_path),
        })
    _upsert_cost(out_root.parent, "w1_cache", attempt_dir.name, wall,
                 {"complete": False,
                  "incomplete_reason": f"crashed:{type(exc).__name__}"},
                 status="crashed")


def _replace_index_row(out_root: Path, attempt: str, entry: dict) -> None:
    """Rewrite ONE attempts-index row IN PLACE (round-5 review fix 3).

    The index is append-only by convention (exactly one row per attempt);
    the final deadline check runs AFTER the complete registration is
    written, and an overrun must flip THAT row to incomplete rather than
    append a second row for the same attempt (disk × index × ledger must
    stay one row each)."""
    idx_path = out_root / integ.ATTEMPTS_INDEX
    index = json.loads(idx_path.read_text())
    hits = [i for i, a in enumerate(index["attempts"])
            if a.get("attempt") == attempt]
    if len(hits) != 1:
        raise SystemExit(
            f"attempts index does not carry exactly one row for {attempt}: "
            "refusing to rewrite")
    i = hits[0]
    index["attempts"][i] = {**index["attempts"][i], **entry,
                            "timestamp": index["attempts"][i].get("timestamp")}
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))


def _run_w1_attempt(args, manifest: dict, prereg: dict, prereg_rec: dict,
                    manifest_path: Path, prereg_path: Path, out_root: Path,
                    attempt_dir: Path, hist: float, remaining: float,
                    deadline: float, code_sha: str, assets_sha: str,
                    t0: float) -> int:
    """One (already allocated, already ledger-registered) cache attempt."""

    def _register_incomplete(n_schedules: int, missing: list[str],
                             *, append_index: bool) -> None:
        # INCOMPLETE attempt: record what is missing (or that only the
        # budget ran out), never fabricate stats; consumers refuse this
        # attempt (the identity carries complete=False). consumed_s is
        # measured AT REGISTRATION TIME so an overrun during the final
        # writes is itself accounted (round-5 fix 3)
        identity = {
            "protocol": w1.MOTION_PROTOCOL_VERSION,
            "experiment_id": w1.EXPERIMENT_ID,
            "attempt": attempt_dir.name,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "preregistration": {"path": path_for_record(prereg_path),
                                "sha256": prereg_rec["sha256"]},
            "training_manifest": {"path": path_for_record(manifest_path),
                                  "sha256": integ.sha256_file(manifest_path)},
            "condition_code_sha256": code_sha,
            "assets_sha256": assets_sha,
            "schedule_cache_key_version": w1.SCHEDULE_KEY_VERSION,
            "complete": False,
            "incomplete_reason": "cache_budget_exhausted",
            "missing_demos": missing,
            "budget": {"total_s": w1.CACHE_BUDGET_S,
                       "historical_s": round(hist, 1),
                       "remaining_at_start_s": round(remaining, 1),
                       "consumed_s": round(
                           time.perf_counter() - (deadline - remaining), 1)},
            "execution": exec_blocks,
            "total_wall_s": round(time.time() - t0, 1),
        }
        (attempt_dir / "cache_identity.json").write_text(
            json.dumps(identity, indent=2, ensure_ascii=False))
        row = {
            "attempt": attempt_dir.name, "complete": False,
            "incomplete_reason": "cache_budget_exhausted",
            "n_schedules": n_schedules,
            "identity": path_for_record(attempt_dir / "cache_identity.json"),
        }
        if append_index:
            integ.append_attempts_index(out_root, row)
        else:
            # late overrun: the complete row was already appended — flip it
            # in place instead of growing a second row for the attempt
            _replace_index_row(out_root, attempt_dir.name, row)
        _upsert_cost(out_root.parent, "w1_cache", attempt_dir.name,
                     time.time() - t0,
                     {"attempt": attempt_dir.name,
                      "n_schedules": n_schedules, "complete": False,
                      "incomplete_reason": "cache_budget_exhausted"},
                     status="done")

    def write_incomplete(n_schedules: int, missing: list[str]) -> int:
        _register_incomplete(n_schedules, missing, append_index=True)
        print(f"[w1-cache] INCOMPLETE (budget exhausted): {n_schedules}/"
              f"{len(jobs)} schedules, missing {missing}")
        return 2

    admitted = [r for r in manifest["rows"] if r.get("admission", {}).get("motion")]
    jobs = []
    for r in admitted:
        demo = r["demo"]
        dpath = PROJECT_ROOT / r["derived"]["path"]
        if integ.sha256_file(dpath) != r["derived"]["sha256"]:
            raise SystemExit(f"{demo}: derived representation drifted since the audit")
        with np.load(dpath, allow_pickle=False) as z:
            if "similarity_scale" not in z.files:
                raise SystemExit(
                    f"{demo}: derived representation predates the geometry-scale "
                    "fix (no similarity_scale) — re-run the A3 audit first")
        key = w1.schedule_cache_key(r["inputs"]["npz_sha256"],
                                    r["inputs"]["pt_sha256"],
                                    assets_sha, code_sha)
        jobs.append((demo, str(dpath), key, str(attempt_dir)))
    print(f"[w1-cache] {len(jobs)} admitted rows -> keyed schedules "
          f"(budget: total {w1.CACHE_BUDGET_S}s, historical {hist:.1f}s, "
          f"remaining {remaining:.1f}s)")

    workers = resolve_workers(args.workers, args.cpu_budget, len(jobs))
    # the job set runs in CHUNKS with the deadline checked between chunks
    rows: list[dict] = []
    exec_blocks: list[dict] = []
    chunk = max(1, workers)
    for i in range(0, len(jobs), chunk):
        if time.perf_counter() > deadline:
            break
        part, rep = parallel_map(_w1_schedule_job, jobs[i:i + chunk],
                                 workers=workers, requested_workers=args.workers,
                                 cpu_budget=args.cpu_budget,
                                 native_threads=args.native_threads,
                                 initializer=_worker_init,
                                 initargs=(args.native_threads,))
        rows.extend(part)
        exec_blocks.append(execution_record(rep, len(part)))
    print(json.dumps(exec_blocks[0] if exec_blocks else {"n_tasks": 0}))

    # round-3 review fix 6 (deepened by round-4 fix 3 and round-5 fix 3):
    # the deadline is enforced at EVERY phase boundary — the LAST chunk's
    # return, the stats computation, the FINAL identity hashing AND the
    # registration writes (see the final re-check below).
    # ``budget_stopped`` used to depend only on missing rows, so a final
    # chunk (and the stats work after it) could overrun the registered
    # budget and still register complete=true
    budget_stopped = len(rows) < len(jobs)
    overran = time.perf_counter() > deadline
    stats = stats_facts = None
    if not budget_stopped and not overran:
        # root stats: ONLY admitted train24 rows, every ORIGINAL frame once
        # (val/test and rejected files never contribute — plan §4.2; the
        # helper filters split=="train" itself)
        stats, stats_facts = w1.fit_w1_root_stats(
            manifest["rows"], PROJECT_ROOT, attempt_dir / "schedules",
            assets_sha, code_sha)
        stats.save(attempt_dir / "root_stats.npz")
        overran = time.perf_counter() > deadline   # the stats phase counts
    if budget_stopped or overran:
        return write_incomplete(len(rows),
                                [j[0] for j in jobs[len(rows):]])

    files_blob = json.dumps(stats_facts["files"],
                            ensure_ascii=False, separators=(",", ":"))
    identity = {
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "experiment_id": w1.EXPERIMENT_ID,
        "attempt": attempt_dir.name,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "preregistration": {"path": path_for_record(prereg_path),
                            "sha256": prereg_rec["sha256"]},
        "training_manifest": {"path": path_for_record(manifest_path),
                              "sha256": integ.sha256_file(manifest_path)},
        "condition_code_sha256": code_sha,
        "assets_sha256": assets_sha,
        "schedule_cache_key_version": w1.SCHEDULE_KEY_VERSION,
        "complete": True,
        "budget": {"total_s": w1.CACHE_BUDGET_S,
                   "historical_s": round(hist, 1),
                   "remaining_at_start_s": round(remaining, 1),
                   "consumed_s": round(time.perf_counter() - (deadline - remaining), 1)},
        "root_stats": {
            "definition": "clip-frame root positions, every original frame "
                          "of the admitted train24 rows, once per frame",
            "file_sha256": integ.sha256_file(attempt_dir / "root_stats.npz"),
            "files": stats_facts["files"],
            "files_canonical_sha256": hashlib.sha256(
                files_blob.encode("utf-8")).hexdigest(),
            "frames_fitted": stats_facts["frames_fitted"],
            "n_train_rows_used": stats_facts["n_train_rows_used"],
            "std_floor": 1e-6,
            "mean": stats.mean.tolist(),
            "std": stats.std.tolist(),
        },
        # review fix 6: consumers re-hash EVERY schedule file against this
        # record — a key-correct filename with different content is caught
        "schedule_rows": [
            {"demo": row["demo"], "split": r["split"], "key": key[:16],
             "sha256": row["sha256"],
             "full_windows": r.get("full_windows_F60_s30", 0)}
            for row, r, key in zip(rows, admitted, [j[2] for j in jobs])
        ],
        "execution": exec_blocks,
        "total_wall_s": round(time.time() - t0, 1),
    }
    # round-4 review fix 3: the FINAL hashing is inside the budget — the
    # last check sat at the stats phase, so a slow manifest/identity hash
    # after it still registered complete=true with consumed_s already past
    # the budget
    if time.perf_counter() > deadline:
        return write_incomplete(len(rows), [j[0] for j in jobs[len(rows):]])
    # round-5 review fix 3: the registration WRITES are budget work too —
    # the round-4 check sat before them, so a slow identity write still
    # registered complete=true with the stale consumed snapshot while the
    # ledger recorded the real wall. The registration is now written FIRST
    # (consumed_s refreshed right before the write so it covers the final
    # hashing) and the deadline is re-checked after ALL of it; an overrun
    # rewrites the SAME attempt's identity/index/ledger rows as incomplete
    # with the elapsed measured at registration time — never a second
    # index row for one attempt
    identity["budget"]["consumed_s"] = round(
        time.perf_counter() - (deadline - remaining), 1)
    (attempt_dir / "cache_identity.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False))
    integ.append_attempts_index(out_root, {
        "attempt": attempt_dir.name,
        "complete": True,
        "n_schedules": len(rows),
        "n_train_files": stats_facts["n_train_rows_used"],
        "frames_fitted": identity["root_stats"]["frames_fitted"],
        "identity": path_for_record(attempt_dir / "cache_identity.json"),
    })
    _upsert_cost(out_root.parent, "w1_cache", attempt_dir.name,
                 time.time() - t0,
                 {"attempt": attempt_dir.name, "n_schedules": len(rows),
                  "complete": True}, status="done")
    if time.perf_counter() > deadline:
        _register_incomplete(len(rows), [], append_index=False)
        print(f"[w1-cache] INCOMPLETE (budget exhausted during final "
              f"registration): {len(rows)}/{len(jobs)} schedules")
        return 2
    print(f"[w1-cache] root stats frames={identity['root_stats']['frames_fitted']} "
          f"files={stats_facts['n_train_rows_used']} -> {path_for_record(attempt_dir)}")
    return 0


def run_w1(args) -> int:
    """W1 keyed cache: schedules + train-only root stats + identity."""
    t0 = time.time()
    manifest_path = args.training_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    prereg_rec = manifest["preregistration"]
    prereg_path = PROJECT_ROOT / prereg_rec["path"]
    if integ.sha256_file(prereg_path) != prereg_rec["sha256"]:
        raise SystemExit("training manifest not bound to its preregistration")
    prereg = json.loads(prereg_path.read_text())
    check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                      TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    if not check["passed"]:
        raise SystemExit(f"preregistration re-verification failed: {check['problems']}")

    out_root = integ.guard_output_dir(args.out_dir)
    # budget preflight BEFORE allocating the attempt (round-2 review fix 7):
    # the registered CACHE_BUDGET_S is enforced against the run-root ledger —
    # prior attempts are deducted and an unknown history refuses to start
    hist = _historical_cache_cost_s(out_root)
    if hist is None:
        raise SystemExit("cache cost history unknown (attempts exist without "
                         "ledger rows) — refusing without a provable budget")
    remaining = w1.CACHE_BUDGET_S - hist
    if remaining <= 0:
        raise SystemExit(f"cache budget already exhausted: historical "
                         f"{hist:.1f}s of {w1.CACHE_BUDGET_S}s")
    deadline = time.perf_counter() + remaining

    attempt_dir = integ.allocate_attempt(out_root)
    code_sha = w1.condition_code_sha256(PROJECT_ROOT)
    assets_sha = prereg["assets"]["a1_assets_sha256"]
    # round-3 review fix 8: the ledger row is pre-registered BEFORE any work
    # and every exit path updates it — a crashed cache stage stays accounted
    # (disk × index × ledger stay reconcilable, never an unknown zero)
    _upsert_cost(out_root.parent, "w1_cache", attempt_dir.name, 0.0,
                 {"attempt": attempt_dir.name, "n_schedules": 0},
                 status="running")
    t_attempt = time.perf_counter()
    try:
        return _run_w1_attempt(args, manifest, prereg, prereg_rec,
                               manifest_path, prereg_path, out_root,
                               attempt_dir, hist, remaining, deadline,
                               code_sha, assets_sha, t0)
    except BaseException as exc:  # noqa: BLE001 -- register, then re-raise
        _register_crashed_cache_attempt(out_root, attempt_dir, exc,
                                        time.perf_counter() - t_attempt)
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="number of processes (spawn; results in input "
                         "order). 1 = in-process serial")
    ap.add_argument("--cpu-budget", type=int, default=DEFAULT_CPU_BUDGET,
                    help="max physical cores for --workers auto")
    ap.add_argument("--native-threads", type=int, default=1,
                    help="native thread pools per worker process")
    ap.add_argument("--force", action="store_true", help="re-infer existing schedules")
    ap.add_argument("--training-manifest", type=Path, default=None,
                    help="W1 mode: the A3 training_manifest.json (keyed, "
                         "manifest-bound cache)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="W1 mode: cache root (writes attempt_NNN inside; "
                         "must resolve under derived/)")
    args = ap.parse_args()
    if args.training_manifest is not None or args.out_dir is not None:
        if args.training_manifest is None or args.out_dir is None:
            raise SystemExit("W1 mode requires BOTH --training-manifest and --out-dir")
        return run_w1(args)

    all_files = {s: load_split_files(s) for s in ("train", "val", "test")}
    jobs = [f for files in all_files.values() for f in files]
    print(f"{len(jobs)} NPZ files across splits")

    workers = resolve_workers(args.workers, args.cpu_budget, len(jobs))
    rows, rep = parallel_map(process_file, [(f, args.force) for f in jobs],
                             workers=workers, requested_workers=args.workers,
                             cpu_budget=args.cpu_budget,
                             native_threads=args.native_threads,
                             initializer=_worker_init,
                             initargs=(args.native_threads,))
    print(json.dumps(execution_record(rep, len(jobs))))

    # train-split root statistics (all frames of all train clips)
    pos = []
    for name in all_files["train"]:
        clip = build_interaction_clip(name, schedule_cache=CACHE_DIR / "schedules")
        pos.append(world_to_clip(clip.root[:, 0:3], clip.condition.world_from_clip))
    pos = np.concatenate(pos, axis=0)
    std = pos.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    np.savez(CACHE_DIR / "root_stats.npz", mean=pos.mean(0), std=std)

    viol = np.array([r["joint_limit_max_violation_rad"] for r in rows])
    manifest = {
        "n_files": len(rows),
        "splits": {s: len(f) for s, f in all_files.items()},
        "total_frames": int(sum(r["T"] for r in rows)),
        "root_stats": {"mean": pos.mean(0).tolist(), "std": std.tolist(),
                       "frames_fitted": int(pos.shape[0])},
        "joint_limits": {
            "max_violation_rad": float(viol.max()),
            "files_with_violation": int((viol > 0).sum()),
            "files_dropped": [r["npz"] for r in rows if r["drop"]],
            "files_clamped": int(sum(1 for r in rows if r["clamped"])),
        },
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["joint_limits"], indent=2))
    print(json.dumps(manifest["root_stats"], indent=2))
    print(f"cache -> {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    main()
