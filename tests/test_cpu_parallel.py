"""cpu_parallel_optimization_plan (docs/plans/cpu_parallel_optimization_plan.md)
boundary tests: the unified resource control module, the ordered spawn
pool, cross-task isolation / crash semantics, and the BIT-IDENTITY
contracts (real serial vs parallel regeneration; FK fast-path + distance
cache on/off).

Pinned here:

- ``resolve_workers``: auto = min(physical, budget, tasks) clamped to
  [1, 20]; explicit ints honoured and clamped; garbage rejected;
- the spawn children inherit the thread environment at exec time (before
  any numeric import they might do) and threadpoolctl caps the pools of
  the running process;
- ``parallel_map`` returns results and fires ``on_result`` STRICTLY in
  input order however the tasks finish; a raising task or a hard-killed
  worker aborts the batch (failed index recorded, never a silent partial
  batch); per-task state is isolated (no cross-task leakage through
  worker globals); duplicate tasks are both processed identically;
  seeded random input is identical serial vs parallel;
- the regeneration CLI: serial (--workers 1, the original path) vs
  parallel (--workers 2) on REAL truncated demos produces reports equal
  under the plan's comparison exclusion list (cpu_parallel_compare) --
  the same guard the frozen-R3 six-segment acceptance uses;
- FrameEvaluator state cache: the first forward on a fresh evaluator is
  never skipped even for an all-zero request (the MjData zero sentinel);
  a repeated identical request skips mj_forward; per-state distance
  memoisation hits and is invalidated by a state change; a real strict
  segment solved with the cache ON/OFF is bit-identical (trajectory
  bytes + diagnostics minus timing keys) with skips/hits > 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import cpu_parallel_compare as cmp  # noqa: E402
from ireset.resctrl import (  # noqa: E402
    MAX_PHYSICAL_CORES,
    apply_native_threads,
    parallel_map,
    resolve_workers,
    thread_state,
    verify_native_threads,
)


# -- resolve_workers ----------------------------------------------------------

def test_resolve_workers_auto_budget_and_clamp():
    n = resolve_workers("auto", 16, 40)
    assert 1 <= n <= min(16, MAX_PHYSICAL_CORES)
    # task count caps concurrency
    assert resolve_workers("auto", 16, 3) <= 3
    # explicit workers honoured, clamped to the physical cap
    assert resolve_workers(4, 16, 40) == 4
    assert resolve_workers(2, 16, 1) == 2
    assert resolve_workers(99, 16, 40) == MAX_PHYSICAL_CORES
    # serial means serial
    assert resolve_workers(1, 16, 40) == 1
    with pytest.raises(ValueError):
        resolve_workers("banana", 16, 40)
    with pytest.raises(ValueError):
        resolve_workers(0, 16, 40)


# -- thread control ------------------------------------------------------------

def test_apply_native_threads_and_verification():
    state = apply_native_threads(1)
    for k, v in state["thread_env"].items():
        assert v == "1", f"{k} not set for children: {v}"
    ok, _ = verify_native_threads(1)
    assert ok, thread_state()
    apply_native_threads(2)  # restore a permissive setting for other tests


def _child_thread_probe(task):
    import os

    return {"env": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "argv0": sys.argv[0]}


def test_spawn_children_inherit_thread_env(tmp_path):
    """The env must exist at exec time (spawn), i.e. BEFORE any numeric
    import the child performs -- recorded from inside the worker."""
    results, rep = parallel_map(_child_thread_probe, [0, 1, 2], workers=2,
                                native_threads=1)
    assert rep.mode == "parallel" and rep.mp_context == "spawn"
    for r in results:
        assert set(r["env"].values()) == {"1"}, r["env"]


# -- ordered map / crash / isolation -------------------------------------------

def _jitter_task(i: int) -> int:
    time.sleep((3 - i) * 0.08)     # EARLIER tasks sleep longer
    return i * i


def test_parallel_map_input_order_with_jitter():
    got = []
    results, _ = parallel_map(_jitter_task, list(range(4)), workers=4,
                              on_result=lambda i, r: got.append((i, r)))
    assert results == [0, 1, 4, 9]
    assert got == list(enumerate([0, 1, 4, 9])), \
        "on_result must fire in INPUT order, not completion order"


def _boom_task(i: int) -> int:
    if i == 2:
        raise RuntimeError(f"task {i} failed")
    return i


def test_parallel_map_task_crash_aborts_batch():
    seen = []
    with pytest.raises(RuntimeError):
        parallel_map(_boom_task, list(range(5)), workers=2,
                     on_result=lambda i, r: seen.append(i))
    assert seen == [0, 1], "only the completed prefix may be emitted"


def _hard_kill_task(i: int) -> int:
    if i == 1:
        os._exit(137)              # simulate a worker crash
    return i


def test_parallel_map_worker_crash_aborts_batch():
    with pytest.raises(Exception):
        parallel_map(_hard_kill_task, list(range(4)), workers=2)


_STATE = {"value": -1}


def _isolation_task(i: int) -> dict:
    seen = _STATE["value"]
    _STATE["value"] = i            # pollute for whichever task runs next
    return {"i": i, "saw": seen}


def test_cross_task_state_isolation():
    """A worker global mutated by task A must NOT leak into task B --
    the regeneration worker resets per-task state; the framework itself
    must never add shared mutable state of its own (this test pins the
    serial reference: results are a pure function of the input order)."""
    serial, _ = parallel_map(_isolation_task, list(range(6)), workers=1)
    for k, r in enumerate(serial):
        assert r == {"i": k, "saw": k - 1}


def _dup_task(x: str) -> str:
    time.sleep(0.02)
    return x.upper()


def test_duplicate_tasks_both_processed():
    tasks = ["a", "b", "a"]
    results, rep = parallel_map(_dup_task, tasks, workers=3)
    assert results == ["A", "B", "A"]
    assert len(rep.per_task) == 3


def _seeded_task(seed: int) -> list:
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.normal(size=8).tolist()


def test_seeded_random_identical_serial_vs_parallel():
    tasks = [101, 202, 303, 404, 505]
    serial, _ = parallel_map(_seeded_task, tasks, workers=1)
    par, _ = parallel_map(_seeded_task, tasks, workers=3)
    assert serial == par


# -- FrameEvaluator state cache --------------------------------------------------

def _fresh_evaluator():
    from ireset.embodied.g1_spec import G1Spec
    from ireset.geometry.collision_pairs import build_collision_pairs
    from ireset.repair.constraints import FrameEvaluator
    from ireset.repair.context import RepairConfig

    spec = G1Spec()
    ev = FrameEvaluator(spec, build_collision_pairs(spec, scene=True),
                        RepairConfig())
    return ev


def test_fk_fastpath_sentinel_skip_and_invalidation():
    ev = _fresh_evaluator()
    z = np_zeros36()
    box = np_zeros7()
    z[3] = 1.0                      # identity quaternion (a legal state)
    box[3] = 1.0
    # FIRST call on a fresh evaluator must run mj_forward even for an
    # all-zero request: fresh MjData holds zeros whose derived arrays were
    # never computed -- skipping would read an unforwarded state
    ev.forward(z, box)
    assert ev.cache_stats["fwd_skipped"] == 0
    assert float(abs(ev.data.site_xpos).sum()) > 0.0, \
        "the zero-qpos state must actually be forwarded"
    # repeated identical request -> skip
    ev.forward(z, box)
    assert ev.cache_stats["fwd_calls"] == 2
    assert ev.cache_stats["fwd_skipped"] == 1
    # distance memoisation within the state
    ga, gb = ev.hand_sites["l"][0], ev.box_geom
    d1 = ev.geom_distance(ga, gb)
    d2 = ev.geom_distance(ga, gb)
    assert d1[0] == d2[0] and (d1[1] == d2[1]).all()
    assert ev.cache_stats["dist_hits"] == 1
    # a different state clears the cache: same pair recomputed
    z2 = z.copy()
    z2[0] += 0.05
    ev.forward(z2, box)
    d3 = ev.geom_distance(ga, gb)
    assert ev.cache_stats["dist_hits"] == 1, "cache must reset with the state"
    assert d3[0] != d1[0] or not (d3[1] == d1[1]).all()
    # disabling the cache keeps the plain behaviour
    ev.state_cache = False
    ev.geom_distance(ga, gb)
    assert ev.cache_stats["dist_hits"] == 1


def np_zeros36():
    import numpy as np

    return np.zeros(36)


def np_zeros7():
    import numpy as np

    return np.zeros(7)


def _real_segment(T: int):
    from ireset.data.teacher import load_teacher_npz
    from ireset.embodied.contact_sites import infer_reference_schedule
    from ireset.paths import TEACHER_TRAJ_DIR
    from ireset.repair.context import ReferenceCondition

    scene, human, _ = load_teacher_npz(
        TEACHER_TRAJ_DIR / "sub3_largebox_000_original.npz")
    scene, human = scene[:T].copy(), human[:T].copy()
    cond = ReferenceCondition(human, scene[:, 36:43],
                              infer_reference_schedule(human, scene[:, 36:43]))
    return scene, cond


def _solve_with_cache(scene, cond, cache_on: bool):
    from ireset.repair.config_io import load_mode_configs
    from ireset.repair.pipeline import RepairPipeline

    rcfg, vcfg = load_mode_configs("strict")
    pipe = RepairPipeline(cfg=rcfg)
    pipe.validator.cfg = vcfg
    pipe.ev.state_cache = cache_on
    res = pipe.repair_trajectory(scene, cond)
    return res, dict(pipe.ev.cache_stats)


def test_real_segment_cache_bit_identity():
    """Real strict segment solved twice -- cache ON vs OFF -- must be
    bit-identical (trajectory bytes + diagnostics minus timing keys) and
    the cache must actually fire (skips and hits > 0)."""
    import numpy as np

    scene, cond = _real_segment(10)
    res_on, stats_on = _solve_with_cache(scene, cond, True)
    res_off, _ = _solve_with_cache(scene, cond, False)
    assert res_on.status == res_off.status
    assert res_on.accepted == res_off.accepted
    assert (res_on.trajectory is None) == (res_off.trajectory is None)
    if res_on.trajectory is not None:
        assert res_on.trajectory.tobytes() == res_off.trajectory.tobytes()

    def strip(d):
        return {k: v for k, v in d.items()
                if not k.endswith("_s") and k != "timing_stats"
                and k != "cost"}

    assert strip(res_on.diagnostics) == strip(res_off.diagnostics)
    assert stats_on["fwd_skipped"] > 0 and stats_on["dist_hits"] > 0


# -- real CLI: serial vs parallel ------------------------------------------------

def test_real_cli_serial_vs_parallel_bit_identical(tmp_path):
    """The plan's acceptance guard at CLI scale: real truncated demos,
    --workers 1 (original serial path) vs --workers 2 (spawn pool); the
    reports must be IDENTICAL under the exclusion list and any kept NPZ
    byte-equal (cpu_parallel_compare exit 0)."""
    proj = Path(__file__).resolve().parents[1]
    demos = ["sub3_largebox_000", "sub16_largebox_000"]
    out = {}
    for tag, workers in (("serial", "1"), ("par2", "2")):
        d = tmp_path / tag
        d.mkdir()
        cmd = [sys.executable, str(proj / "scripts" / "regenerate_teachers.py"),
               "--demos", *demos, "--frames", "12",
               "--validation-mode", "strict", "--assert-task-gate", "0.40",
               "--output-dir", str(d), "--report", str(d / "report.json"),
               "--note", "cpu-parallel boundary test", "--workers", workers]
        proc = subprocess.run(cmd, cwd=proj, capture_output=True, text=True,
                              timeout=900)
        assert proc.returncode == 0, proc.stderr[-2000:]
        out[tag] = d
    base = json.loads((out["serial"] / "report.json").read_text())
    cand = json.loads((out["par2"] / "report.json").read_text())
    assert [r["demo"] for r in cand["rows"]] == demos, \
        "parallel rows must preserve input order"
    assert cand["execution"]["mode"] == "parallel"
    assert cand["execution"]["workers"] == 2
    assert base["execution"]["mode"] == "serial"
    diffs: list[str] = []
    cmp.deep_compare(base, cand, "report", diffs,
                     _prune=cmp.REPORT_EXCLUDED | {"effective"})
    for k in set(base["effective"]) | set(cand["effective"]):
        if k == "argv":
            continue
        cmp.deep_compare(base["effective"].get(k), cand["effective"].get(k),
                         f"effective.{k}", diffs)
    assert not diffs, diffs[:20]
    for npz in sorted(out["serial"].glob("*.npz")):
        other = out["par2"] / npz.name
        assert other.is_file() and npz.read_bytes() == other.read_bytes()
