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


def _append_costs(run_root: Path, stage: str, wall: float, extra: dict) -> None:
    """Run-root cost ledger (review fix 12): every stage invocation is
    appended with its own measured wall — including failed attempts, which
    register BEFORE the work starts and are completed here on the way out."""
    import datetime as _dt2

    costs_path = run_root / "costs.json"
    entries = []
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    entries.append({"timestamp": _dt2.datetime.now().isoformat(timespec="seconds"),
                    "stage": stage, "wall_s": round(wall, 1), **extra})
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


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
    attempt_dir = integ.allocate_attempt(out_root)
    code_sha = w1.condition_code_sha256(PROJECT_ROOT)
    assets_sha = prereg["assets"]["a1_assets_sha256"]

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
    print(f"[w1-cache] {len(jobs)} admitted rows -> keyed schedules")

    workers = resolve_workers(args.workers, args.cpu_budget, len(jobs))
    rows, rep = parallel_map(_w1_schedule_job, jobs,
                             workers=workers, requested_workers=args.workers,
                             cpu_budget=args.cpu_budget,
                             native_threads=args.native_threads,
                             initializer=_worker_init,
                             initargs=(args.native_threads,))
    print(json.dumps(execution_record(rep, len(jobs))))

    # root stats: ONLY admitted train24 rows, every ORIGINAL frame once
    # (val/test and rejected files never contribute — plan §4.2; the helper
    # filters split=="train" itself)
    stats, stats_facts = w1.fit_w1_root_stats(
        manifest["rows"], PROJECT_ROOT, attempt_dir / "schedules",
        assets_sha, code_sha)
    stats.save(attempt_dir / "root_stats.npz")

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
        "execution": execution_record(rep, len(jobs)),
        "total_wall_s": round(time.time() - t0, 1),
    }
    (attempt_dir / "cache_identity.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False))
    integ.append_attempts_index(out_root, {
        "attempt": attempt_dir.name,
        "n_schedules": len(rows),
        "n_train_files": stats_facts["n_train_rows_used"],
        "frames_fitted": identity["root_stats"]["frames_fitted"],
        "identity": path_for_record(attempt_dir / "cache_identity.json"),
    })
    _append_costs(out_root.parent, "w1_cache", time.time() - t0,
                  {"attempt": attempt_dir.name, "n_schedules": len(rows)})
    print(f"[w1-cache] root stats frames={identity['root_stats']['frames_fitted']} "
          f"files={stats_facts['n_train_rows_used']} -> {path_for_record(attempt_dir)}")
    return 0


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
