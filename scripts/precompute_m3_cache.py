"""Precompute M3 caches: reference contact schedules + train root statistics.

Per NPZ: the reference-side contact schedule (hull-distance hysteresis,
plan 8.1 thresholds) is inferred once and stored compactly under
``derived/data/m3_cache/schedules/``. Then train-split clip-frame root
statistics are fitted and saved (``root_stats.npz``), and teacher
joint-limit violations are summarized (plan 7.2: record, never silently
clip).

Run: uv run --locked python scripts/precompute_m3_cache.py [--workers 4] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data.teacher import build_interaction_clip, schedule_to_cache
from ireset.data.windows import CACHE_DIR, load_split_files
from ireset.paths import TEACHER_TRAJ_DIR
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


def _worker_init(native_threads: int) -> None:
    apply_native_threads(native_threads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="number of processes (spawn; results in input "
                         "order). 1 = in-process serial")
    ap.add_argument("--cpu-budget", type=int, default=DEFAULT_CPU_BUDGET,
                    help="max physical cores for --workers auto")
    ap.add_argument("--native-threads", type=int, default=1,
                    help="native thread pools per worker process")
    ap.add_argument("--force", action="store_true", help="re-infer existing schedules")
    args = ap.parse_args()

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
        from ireset.data.teacher import world_to_clip

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


if __name__ == "__main__":
    main()
