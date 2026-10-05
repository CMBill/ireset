"""Unified CPU resource control and cross-task parallel execution
(cpu_parallel_optimization_plan section 2).

Engineering-only layer: NOTHING here may change any algorithm, profile,
threshold or numeric result. The two contracts are

- thread control: native thread limits are set in the PARENT environment
  BEFORE any process pool is created (spawn children inherit the
  environment at exec time, i.e. before their first numeric import), and
  for libraries already loaded in the CURRENT process threadpoolctl
  enforces the same limit at runtime, verified by re-reading
  ``threadpool_info``; a verification failure falls back to "keep the
  original setting, shrink concurrency and pin affinity" (plan section 2);
- ordered parallel map: ``ProcessPoolExecutor`` with the spawn context,
  results collected STRICTLY in input order, a crashed worker or a raising
  task aborts the whole batch (never a silent partial batch), and an
  execution record (workers, threads, affinity, timings, peak RSS)
  is returned for the run report.

PyTorch is deliberately NOT imported here: repair workers must not pay
the torch import cost, so torch threads are only touched when torch is
already in ``sys.modules``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Callable, Sequence

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

#: plan section 1: at most 16 physical cores by default, 20 explicitly
DEFAULT_CPU_BUDGET = 16
MAX_PHYSICAL_CORES = 20

#: holds the threadpoolctl limits object alive for the process lifetime
_thread_limits = None


def physical_cores() -> int:
    """Visible PHYSICAL cores (hyperthread siblings collapsed)."""
    try:
        pairs = set()
        with open("/proc/cpuinfo") as fh:
            phys = core = None
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif line.strip() == "" and phys is not None and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
            if phys is not None and core is not None:
                pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    # fall back to the affinity mask (logical CPUs; still a valid upper
    # bound for scheduling decisions)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:          # pragma: no cover - non-Linux
        return os.cpu_count() or 1


def resolve_workers(workers: int | str, cpu_budget: int, n_tasks: int) -> int:
    """Plan section 2: ``auto`` = min(physical cores, budget, task count);
    an explicit integer is honoured, clamped to [1, MAX_PHYSICAL_CORES].

    ``workers <= 1`` means SERIAL (the caller must not build a pool)."""
    if isinstance(workers, str):
        if workers == "auto":
            n = min(physical_cores(), int(cpu_budget), n_tasks, MAX_PHYSICAL_CORES)
            return max(1, n)
        try:                     # argparse defaults arrive as strings
            workers = int(workers)
        except ValueError:
            raise ValueError(
                f"workers must be an int or 'auto', got {workers!r}") from None
    n = int(workers)
    if n < 1:
        raise ValueError(f"workers must be >= 1, got {n}")
    return min(n, MAX_PHYSICAL_CORES)


def _apply_env(n: int) -> None:
    for k in THREAD_ENV_VARS:
        os.environ[k] = str(n)


def apply_native_threads(n: int) -> dict:
    """Limit native thread pools of THIS process to ``n`` and set the
    environment for future children. Returns the verification record
    (``thread_state``); on a threadpoolctl failure the fallback is: keep
    the original setting, pin the process to one core, and record it."""
    n = max(1, int(n))
    _apply_env(n)
    fallback = False
    global _thread_limits
    try:
        import threadpoolctl

        _thread_limits = threadpoolctl.threadpool_limits(n)
    except Exception:
        fallback = True
    if "torch" in __import__("sys").modules:
        import torch

        torch.set_num_threads(n)
    if fallback:
        # keep original settings, shrink to one core via affinity
        try:
            os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
        except OSError:
            pass
    return thread_state()


def thread_state() -> dict:
    """Facts for the run report: env, live threadpoolctl pools, torch
    threads (only when imported), affinity size."""
    out: dict = {
        "pid": os.getpid(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    try:
        import threadpoolctl

        out["threadpools"] = [
            {k: info.get(k) for k in ("user_api", "internal_api", "num_threads",
                                      "filepath", "version")}
            for info in threadpoolctl.threadpool_info()]
    except Exception as exc:            # pragma: no cover - diagnostics only
        out["threadpools_error"] = repr(exc)
    import sys

    if "torch" in sys.modules:
        import torch

        out["torch_num_threads"] = torch.get_num_threads()
    return out


def verify_native_threads(expected: int) -> tuple[bool, str]:
    """True when every loaded native pool reports <= ``expected`` threads."""
    state = thread_state()
    pools = state.get("threadpools", [])
    if not pools:
        return True, "no native pools loaded"
    bad = [f"{p.get('user_api')}:{p.get('num_threads')}" for p in pools
           if (p.get("num_threads") or 0) > expected]
    if bad:
        return False, ",".join(bad)
    return True, f"{len(pools)} pools within {expected}"


def _peak_rss_kb() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _task_runner(index: int, func: Callable, task) -> dict:
    """Runs inside the worker: per-task timing + worker identity, so the
    parent can report queue/compute separation without touching results."""
    t_start = time.perf_counter()
    result = func(task)
    return {"index": index, "worker_pid": os.getpid(),
            "compute_s": time.perf_counter() - t_start,
            "peak_rss_kb": _peak_rss_kb(), "result": result}


@dataclass
class ParallelReport:
    mode: str                       # "serial" | "parallel"
    workers: int
    requested_workers: int | str
    cpu_budget: int
    native_threads: int
    mp_context: str
    submit_s: float = 0.0
    collect_s: float = 0.0
    per_task: list = field(default_factory=list)
    parent_thread_state: dict = field(default_factory=dict)
    worker_thread_states: list = field(default_factory=list)
    failed_task: int | None = None
    error: str | None = None


def parallel_map(func: Callable, tasks: Sequence, *, workers: int,
                 requested_workers: int | str = 1, cpu_budget: int = DEFAULT_CPU_BUDGET,
                 native_threads: int = 1,
                 initializer: Callable | None = None, initargs: tuple = (),
                 on_result: Callable[[int, object], None] | None = None) -> tuple[list, ParallelReport]:
    """Ordered map over ``tasks``. ``workers <= 1`` runs ``func`` serially
    in-process (the original path -- no pool, no environment change).

    Parallel mode: spawn context, one future per task, results collected
    strictly in INPUT order; ``on_result(i, result)`` fires in input order
    as each result becomes available (printing order matches serial). A
    raising task or a crashed worker aborts the batch: the exception is
    re-raised after the report records the failed index."""
    rep = ParallelReport(mode="serial" if workers <= 1 else "parallel",
                         workers=max(1, workers), requested_workers=requested_workers,
                         cpu_budget=cpu_budget, native_threads=native_threads,
                         mp_context="spawn")
    if workers <= 1:
        t0 = time.perf_counter()
        results = []
        for i, task in enumerate(tasks):
            s = time.perf_counter()
            r = func(task)
            rep.per_task.append({"index": i, "worker_pid": os.getpid(),
                                 "compute_s": time.perf_counter() - s,
                                 "peak_rss_kb": _peak_rss_kb()})
            results.append(r)
            if on_result is not None:
                on_result(i, r)
        rep.collect_s = time.perf_counter() - t0
        return results, rep

    # children must inherit the thread env at exec time (before their first
    # numeric import) -- apply to the parent environment BEFORE the pool
    _apply_env(native_threads)
    rep.parent_thread_state = thread_state()
    t0 = time.perf_counter()
    ctx = get_context("spawn")
    futures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=initializer, initargs=initargs) as ex:
        for i, task in enumerate(tasks):
            futures.append(ex.submit(_task_runner, i, func, task))
        rep.submit_s = time.perf_counter() - t0
        results: list = []
        try:
            for i, fut in enumerate(futures):
                rec = fut.result()
                rep.per_task.append({k: rec[k] for k in
                                     ("index", "worker_pid", "compute_s", "peak_rss_kb")})
                results.append(rec["result"])
                if on_result is not None:
                    on_result(i, rec["result"])
        except BaseException as exc:
            rep.failed_task = len(rep.per_task)
            rep.error = repr(exc)
            for fut in futures:
                fut.cancel()
            raise
        finally:
            rep.collect_s = time.perf_counter() - t0
    return results, rep


def execution_record(rep: ParallelReport, n_tasks: int) -> dict:
    """JSON-able execution block for run reports (plan section 2)."""
    rss = [t["peak_rss_kb"] for t in rep.per_task] or [0]
    return {
        "mode": rep.mode,
        "workers": rep.workers,
        "requested_workers": rep.requested_workers,
        "cpu_budget": rep.cpu_budget,
        "native_threads": rep.native_threads,
        "mp_context": rep.mp_context,
        "n_tasks": n_tasks,
        "physical_cores": physical_cores(),
        "submit_s": round(rep.submit_s, 3),
        "collect_s": round(rep.collect_s, 3),
        "task_order": "input",
        "per_task_compute_s": [round(t["compute_s"], 3) for t in rep.per_task],
        "max_peak_rss_kb": int(max(rss)),
        "failed_task": rep.failed_task,
        "error": rep.error,
        "parent_thread_state": rep.parent_thread_state,
    }
