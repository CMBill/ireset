"""Scheduler benchmark for the unified parallel layer
(cpu_parallel_optimization_plan section 6.4): 1/2/4/8/16 concurrency on
(a) short tasks and (b) a balanced batch, reporting COLD-start (spawn +
worker boot per batch) versus a PERSISTENT pool (many batches over one
pool) behaviour. CPU-bound synthetic tasks characterise the SCHEDULER;
the real end-to-end numbers come from the regeneration acceptance run.

Run: uv run --locked python scripts/cpu_parallel_bench.py --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.resctrl import execution_record, parallel_map, resolve_workers

import numpy as np


def _short_task(seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(60, 60))
    return float(sum((a @ a.T).trace() for _ in range(12)))


def _balanced_task(seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(120, 120))
    return float(sum((a @ a.T).trace() for _ in range(600)))


def _bench(func, tasks, workers, repeats, native_threads=1):
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        results, rep = parallel_map(func, tasks, workers=workers,
                                    native_threads=native_threads)
        wall = time.perf_counter() - t0
        out.append({"workers": workers, "n_tasks": len(tasks),
                    "wall_s": round(wall, 3),
                    "sum_task_s": round(sum(t["compute_s"] for t in rep.per_task), 3),
                    "max_task_s": round(max(t["compute_s"] for t in rep.per_task), 3),
                    "mode": rep.mode})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    curve = {"short_cold": [], "balanced_cold": [], "serial_reference": {}}
    t0 = time.perf_counter()
    serial_short, _ = parallel_map(_short_task, list(range(16)), workers=1)
    serial_bal, _ = parallel_map(_balanced_task, list(range(16)), workers=1)
    curve["serial_reference"] = {
        "short_16_s": round(time.perf_counter() - t0, 3),
        "short_results_sha": hash(json.dumps(serial_short))}

    for w in (1, 2, 4, 8, 16):
        curve["short_cold"] += _bench(_short_task, list(range(16)), w,
                                      args.repeats)
        curve["balanced_cold"] += _bench(_balanced_task, list(range(16)), w,
                                         args.repeats)

    # persistent-pool proxy: ONE pool, four consecutive batches submitted
    # back-to-back reuses the workers through parallel_map's per-call pool
    # only partially -- measure a single big batch amortisation instead
    big = list(range(64))
    t0 = time.perf_counter()
    _, rep = parallel_map(_balanced_task, big, workers=16, native_threads=1)
    curve["balanced_64_w16"] = {
        "wall_s": round(time.perf_counter() - t0, 3),
        "execution": execution_record(rep, len(big))}

    args.out.write_text(json.dumps(curve, indent=1))
    print(json.dumps({k: v for k, v in curve.items()
                      if k not in ("balanced_64_w16",)}, indent=1))
    print(f"bench -> {args.out}")


if __name__ == "__main__":
    main()
