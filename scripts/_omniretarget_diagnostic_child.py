"""Child driver for the sub12 limited diagnosis (closure plan §5).

Runs INSIDE the independent upstream uv environment; the parent
(``scripts/diagnose_omniretarget_sub12.py``) launches it with an argument
list, never a shell line. The driver:

* mirrors the official entry's import resolution (checkout ``src_root`` at
  ``sys.path[0]``) so the imported retargeter IS the pinned checkout code,
  then verifies the imported file hash + ``solve_single_iteration``
  signature against the pinned expectations before anything runs;
* wraps ``InteractionMeshRetargeter.iterate`` /
  ``.solve_single_iteration`` / ``cvxpy.Problem.solve`` — the wrapped
  success path MUST call the original and return its result unchanged
  (no config, initialization, constraint, cost, solver-option or random
  state is ever touched);
* runs the EXISTING generation entry via ``runpy`` (same argv as
  ``generate_omniretarget_teachers.generate_one``);
* only when the original solve has ALREADY failed and its context is saved,
  executes the single Phase I check (§5.5): minimize ||dqa||2 over the
  original problem's constraints minus the single explicit step-size SOC,
  solved through the ORIGINAL ``Problem.solve`` (no wrapper recursion);
  the diagnostic variable values never flow back (generation already
  aborted; ``dqa.value`` is reset after rho is recorded);
* restores the wrapped functions on exit and writes a machine summary.

``--mode self-test`` builds tiny cone problems and exercises the SAME
``phase_one`` path (used by the project-side tests; needs the upstream env).

The module imports cvxpy lazily so the main environment (no cvxpy) can
still import the pure classifier ``classify_phase_one``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

#: pinned interface expectations (closure plan §5.1; verified against the
#: bccd4d7 checkout 2026-10-10; drift -> instrumentation_not_ready)
EXPECTED_SSI_PARAMS = [
    "self", "q_locked", "q_a_n_last", "q_t_last", "target_laplacian",
    "adj_list", "obj_pts_local", "foot_sticking", "w_nominal_tracking",
    "q_a_nominal", "verbose", "init_t", "frame_idx",
]

#: phase-I numeric acceptance (§5.5): the 1e-6 residual bound is THIS
#: diagnosis's verification bound, never a change to robot acceptance
RESIDUAL_TOL = 1e-6


def classify_phase_one(status: str, rho, max_violation, *,
                       r0: float = 0.2, r1: float = 0.3,
                       tol: float = 1e-6) -> str:
    """Map a Phase I outcome onto the closure plan §5.5 table.

    Pure function (no cvxpy) so the main environment can test every row of
    the table, including the ones a solver will not reliably produce
    (OPTIMAL_INACCURATE, non-finite values, residual overshoot).
    """
    if status in ("infeasible", "infeasible_inaccurate"):
        return "retained_infeasible"
    if status != "optimal":
        return "diagnostic_incomplete"
    if rho is None or max_violation is None:
        return "diagnostic_incomplete"
    if not (np.isfinite(rho) and np.isfinite(max_violation)):
        return "diagnostic_incomplete"
    if max_violation > RESIDUAL_TOL:
        return "diagnostic_incomplete"
    if rho <= r0 + tol:
        return "rho_within_original_radius"
    if rho > r1 - tol:
        return "rho_exceeds_candidate_range"
    return "supports_step03_candidate"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _jsonable(getattr(o, f.name))
                for f in dataclasses.fields(o)}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def phase_one(problem, step_size: float, orig_solve) -> dict:
    """The single Phase I check on the ORIGINAL failed problem (§5.5).

    Reuses the original constraint expressions and the original ``dqa``
    variable; requires EXACTLY ONE explicit step-size SOC; minimizes
    ||dqa||2 over every other constraint, solved through the original
    ``Problem.solve``. ``OPTIMAL`` + finite residuals <= 1e-6 is the only
    acceptable evidence; anything else is recorded, never extrapolated.
    """
    import cvxpy as cp
    from cvxpy.constraints.second_order import SOC
    rec: dict = {"phase": "one", "step_size_expected": step_size}
    socs = [c for c in problem.constraints if isinstance(c, SOC)]
    rec["n_soc_constraints"] = len(socs)
    if len(socs) != 1:
        rec["status"] = "not_single_soc"
        rec["category"] = "phase1_not_single_soc"
        return rec
    soc = socs[0]
    t_arg, x_arg = soc.args[0], soc.args[1]
    t_val = getattr(t_arg, "value", None)
    rec["soc_t"] = _jsonable(t_val)
    t_scalar = (float(np.ravel(np.asarray(t_val, dtype=float))[0])
                if t_val is not None else None)
    rec["soc_t_matches_step_size"] = bool(
        t_scalar is not None and abs(t_scalar - step_size) <= 1e-12)
    if not isinstance(x_arg, cp.Variable):
        rec["status"] = "soc_x_not_variable"
        rec["category"] = "phase1_soc_x_not_variable"
        return rec
    dqa = x_arg
    rec["dqa_shape"] = list(dqa.shape)
    kept = [c for c in problem.constraints if c is not soc]
    rec["n_kept_constraints"] = len(kept)
    rec["kept_constraint_types"] = sorted({type(c).__name__ for c in kept})
    diag = cp.Problem(cp.Minimize(cp.norm(dqa, 2)), kept)
    t0 = time.time()
    try:
        try:
            orig_solve(diag, solver=cp.CLARABEL, time_limit=55.0)
        except Exception:  # noqa: BLE001 -- solver option unsupported: retry plain
            orig_solve(diag, solver=cp.CLARABEL)
    except Exception as exc:  # noqa: BLE001 -- recorded, never raised out
        rec["status"] = "phase1_solver_error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["category"] = "phase1_solver_error"
        return rec
    rec["solve_s"] = round(time.time() - t0, 2)
    rec["status"] = str(diag.status)
    if diag.status != cp.OPTIMAL:
        dqa.value = None
        rec["category"] = classify_phase_one(rec["status"], None, None)
        return rec
    rho = float(np.linalg.norm(np.asarray(dqa.value, dtype=float)))
    worst = 0.0
    finite = True
    for c in kept:
        try:
            v = np.asarray(c.violation(), dtype=float)
        except Exception as exc:  # noqa: BLE001 -- violation not computable
            rec["violation_error"] = f"{type(exc).__name__}: {exc}"
            finite = False
            break
        if v.size and (not np.all(np.isfinite(v))):
            finite = False
            break
        worst = max(worst, float(np.max(np.abs(v))) if v.size else 0.0)
    dqa.value = None     # diagnostic values never flow anywhere
    rec["rho"] = rho if finite else None
    rec["max_violation"] = worst if finite else None
    rec["residuals_finite"] = finite
    rec["category"] = classify_phase_one("optimal" if finite else "nonfinite",
                                         rec["rho"], rec["max_violation"])
    return rec


def _save_sparse_problem(problem, out_npz: Path) -> dict:
    """Canonical Clarabel data of the original problem (§5.4), best effort."""
    import scipy.sparse as sp
    try:
        data, _chain, _inv = problem.get_problem_data("CLARABEL")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    arrays = {}
    for key in ("q", "b"):
        if key in data and data[key] is not None:
            arrays[key] = np.asarray(data[key], dtype=float)
    rec: dict = {"arrays": sorted(arrays), "matrices": {}}
    for key in ("P", "A"):
        m = data.get(key)
        if m is None:
            continue
        m = sp.csr_matrix(m)
        path = out_npz.parent / f"{out_npz.stem}_{key}.npz"
        sp.save_npz(path, m)
        rec["matrices"][key] = {"file": path.name, "shape": list(m.shape),
                                "nnz": int(m.nnz)}
    np.savez(out_npz, **arrays)
    rec["n_variables"] = int(data.get("n_var", -1))
    rec["n_constraints"] = int(data.get("n_eq", -1) + data.get("n_ineq", -1))
    return rec


def run_wrapped_generation(args) -> int:
    import cvxpy as cp          # the same module object the upstream uses

    ho_package = Path(args.holosoma_package)
    entry = ho_package / "examples" / "robot_retarget.py"
    retargeter_mod_file = (ho_package / "src" / "holosoma_retargeting" /
                           "src" / "interaction_mesh_retargeter.py")
    # mirror the official entry: checkout src_root at sys.path[0] BEFORE the
    # first holosoma_retargeting import, so the pinned checkout code is what
    # runs (and what we wrap)
    src_root = ho_package.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "driver_sha256": _sha256_file(Path(__file__).resolve()),
        "entry_path": str(entry),
        "entry_sha256": _sha256_file(entry),
        "expected_retargeter_sha256": args.retargeter_sha256,
        "expected_ssi_params": EXPECTED_SSI_PARAMS,
        "np_random_state_sha256": hashlib.sha256(
            np.asarray(np.random.get_state()[1]).tobytes()).hexdigest(),
        "argv": sys.argv[1:],
    }

    import inspect
    import runpy
    from holosoma_retargeting.src import interaction_mesh_retargeter as imr
    identity["retargeter_imported_path"] = str(Path(imr.__file__).resolve())
    identity["retargeter_imported_sha256"] = _sha256_file(
        Path(imr.__file__))
    retargeter_matches = (identity["retargeter_imported_sha256"]
                          == args.retargeter_sha256)
    ssi_params = list(inspect.signature(
        imr.InteractionMeshRetargeter.solve_single_iteration).parameters)
    signature_matches = ssi_params == EXPECTED_SSI_PARAMS
    identity["retargeter_file_matches_pinned"] = retargeter_matches
    identity["ssi_signature_matches_pinned"] = signature_matches
    if not (retargeter_matches and signature_matches):
        identity["instrumentation"] = "instrumentation_not_ready"
        (snapshot_dir / "child_summary.json").write_text(json.dumps(
            {"identity": identity, "outcome": "instrumentation_not_ready"},
            indent=2))
        print("[child] instrumentation_not_ready: imported file or "
              "signature drifted from the pinned checkout", file=sys.stderr)
        return 3

    Retargeter = imr.InteractionMeshRetargeter
    ORIG_SSI = Retargeter.solve_single_iteration
    ORIG_ITER = Retargeter.iterate
    ORIG_SOLVE = cp.Problem.solve

    call_log: list[dict] = []
    frame_rows: list[dict] = []
    problems: list = []            # refs of every Problem.solve invocation
    solve_rows: list[dict] = []
    failure: dict | None = None
    ordinal_in_frame = 0
    counter_frame = -1

    def wrapped_solve(self, *a, **kw):
        problems.append(self)
        row = {"call": len(problems)}
        try:
            return ORIG_SOLVE(self, *a, **kw)
        finally:
            row["status"] = str(self.status)
            ss = getattr(self, "solver_stats", None)
            row["solver_stats"] = {
                k: getattr(ss, k, None) for k in
                ("solver_name", "solve_time", "iterations", "status")}
            solve_rows.append(row)

    def _find_retarget_motion_frame():
        tb = sys.exc_info()[2]
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "retarget_motion":
                return tb.tb_frame
            tb = tb.tb_next
        return None

    def wrapped_ssi(self, **kwargs):
        nonlocal ordinal_in_frame, counter_frame, failure
        frame_idx = int(kwargs.get("frame_idx", 0))
        if frame_idx != counter_frame:
            ordinal_in_frame = 0
            counter_frame = frame_idx
        ordinal_in_frame += 1
        problems_before = len(problems)
        row: dict = {
            "frame_idx": frame_idx,
            "ordinal_in_frame": ordinal_in_frame,
            "init_t": bool(kwargs.get("init_t", False)),
        }
        t0 = time.time()
        try:
            result = ORIG_SSI(self, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- snapshot then re-raise
            row["exception"] = f"{type(exc).__name__}: {exc}"
            row["problem_statuses_in_call"] = [str(p.status)
                                               for p in problems[problems_before:]]
            if failure is None and "CVXPY solve failed" in str(exc):
                failure = _capture_failure(self, kwargs, row,
                                           snapshot_dir, identity)
            call_log.append(row)
            raise
        row.update(ok=True, call_s=round(time.time() - t0, 3))
        call_log.append(row)
        return result

    def wrapped_iterate(self, **kwargs):
        frame_idx = int(kwargs.get("frame_idx", 0))
        t0 = time.time()
        try:
            result = ORIG_ITER(self, **kwargs)
            frame_rows.append({
                "frame_idx": frame_idx, "ok": True,
                "n_iter_budget": int(kwargs.get("n_iter", 10)),
                "final_cost": (None if result[1] is None
                               else float(result[1])),
                "iterate_s": round(time.time() - t0, 3)})
            return result
        except Exception as exc:  # noqa: BLE001
            frame_rows.append({
                "frame_idx": frame_idx, "ok": False,
                "n_iter_budget": int(kwargs.get("n_iter", 10)),
                "exception": f"{type(exc).__name__}: {exc}"})
            raise

    def _capture_failure(retargeter, kwargs, row, snap_dir, ident) -> dict:
        """Save the original failure context + run Phase I (§5.4/§5.5)."""
        arr = {
            "q_locked": np.asarray(kwargs["q_locked"], dtype=float),
            "q_a_n_last": np.asarray(kwargs["q_a_n_last"], dtype=float),
            "q_t_last": np.asarray(kwargs["q_t_last"], dtype=float),
            "target_laplacian": np.asarray(kwargs["target_laplacian"],
                                           dtype=float),
            "obj_pts_local": np.asarray(kwargs["obj_pts_local"], dtype=float),
            "q_a_indices": np.asarray(retargeter.q_a_indices, dtype=int),
        }
        meta = dict(row)
        meta["step_size"] = float(retargeter.step_size)
        meta["foot_sticking"] = _jsonable(kwargs.get("foot_sticking"))
        meta["adj_list"] = _jsonable(kwargs.get("adj_list"))
        meta["w_nominal_tracking"] = float(
            kwargs.get("w_nominal_tracking", 0.0))
        qn = kwargs.get("q_a_nominal")
        if qn is not None:
            arr["q_a_nominal"] = np.asarray(qn, dtype=float)
        meta["q_a_nominal_available"] = qn is not None
        # original problem + its solve status (the failing problem is the
        # LAST one solved inside this call)
        failing = problems[-1] if problems else None
        meta["original_status"] = (solve_rows[-1]["status"]
                                   if solve_rows else None)
        meta["original_solver_stats"] = (solve_rows[-1]["solver_stats"]
                                         if solve_rows else None)
        meta["n_constraints_original"] = (len(failing.constraints)
                                          if failing is not None else None)
        # the frames the upstream returned before failing (via the frame
        # that raised, not a progress-bar tail)
        rm_frame = _find_retarget_motion_frame()
        prefix = None
        if rm_frame is not None:
            motions = rm_frame.f_locals.get("retargeted_motions")
            if motions:
                prefix = np.array(motions, dtype=float)[1:]
                arr["prefix_initial_q"] = np.array(motions, dtype=float)[0]
                meta["prefix_frames"] = int(prefix.shape[0])
                meta["failing_frame_idx"] = int(
                    rm_frame.f_locals.get("i", row["frame_idx"]))
        try:
            np.savez(snap_dir / "failure_context.npz", **arr)
            if prefix is not None:
                np.savez(snap_dir / "prefix_debug.npz", qpos=prefix,
                         fps=30, diagnostic_only=np.bool_(True))
            meta["prefix_layout"] = ("upstream qpos [pos(3),quat(4),"
                                     "joints(29),object pos(3),quat(4)]")
            meta["prefix_use"] = ("diagnostic only; never validated as a "
                                  "feasible trajectory, never a teacher "
                                  "label")
            if failing is not None:
                meta["canonical_data"] = _save_sparse_problem(
                    failing, snap_dir / "failure_canonical.npz")
                meta["variables"] = [
                    {"name": v.name(), "shape": list(v.shape),
                     "size": int(v.size)}
                    for v in failing.variables()]
            # Phase I — only now, with the original failure fully saved
            meta["phase_one"] = phase_one(failing, retargeter.step_size,
                                          ORIG_SOLVE)
        except Exception as exc:  # noqa: BLE001 -- snapshot errors recorded
            meta["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        (snap_dir / "failure_meta.json").write_text(json.dumps(
            {"identity": ident, "failure": meta}, indent=2,
            default=_jsonable))
        return meta

    cp.Problem.solve = wrapped_solve
    Retargeter.solve_single_iteration = wrapped_ssi
    Retargeter.iterate = wrapped_iterate
    sys.argv = ["robot_retarget.py",
                "--task-type", "object_interaction",
                "--task-name", args.demo,
                "--data-format", "smplh",
                "--data-path", args.data_path,
                "--save-dir", args.save_dir,
                "--task-config.object-name", "largebox"]
    # effective config serialized from the real config API (tyro) before
    # the entry runs — optimization/initialization/scenes included
    effective_cfg = None
    if not args.skip_config_dump:
        try:
            import tyro
            from holosoma_retargeting.config_types.retargeting import (
                RetargetingConfig)
            cfg = tyro.cli(RetargetingConfig, argv=sys.argv[1:])
            effective_cfg = _jsonable(cfg)
        except Exception as exc:  # noqa: BLE001
            effective_cfg = {"error": f"{type(exc).__name__}: {exc}"}

    outcome = "completed"
    rc = 0
    t0 = time.time()
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as exc:
        outcome = f"system_exit:{exc.code}"
        rc = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:  # noqa: BLE001 -- the failure under diagnosis
        outcome = f"{type(exc).__name__}: {exc}"
        rc = 1
    finally:
        cp.Problem.solve = ORIG_SOLVE
        Retargeter.solve_single_iteration = ORIG_SSI
        Retargeter.iterate = ORIG_ITER

    summary = {
        "demo": args.demo,
        "outcome": outcome,
        "exit_code": rc,
        "wall_s": round(time.time() - t0, 1),
        "frames_completed": (max((r["frame_idx"] for r in frame_rows
                                  if r.get("ok")), default=-1) + 1),
        "n_solve_calls": len(problems),
        "n_frames_iterated": len(frame_rows),
        "failure": failure,
        "last_two_valid_frames": [r for r in frame_rows if r.get("ok")][-2:],
        "effective_config": effective_cfg,
        "identity": identity,
        "argv_upstream": sys.argv[1:],
    }
    (snapshot_dir / "child_summary.json").write_text(json.dumps(
        summary, indent=2, default=_jsonable))
    print(f"[child] outcome={outcome} failure_captured={failure is not None}")
    return rc


def run_self_test() -> int:
    """Toy cone problems covering the §10.2 phase-one rows."""
    import cvxpy as cp
    from cvxpy.constraints.second_order import SOC

    results = []
    for name, lo in (("within", np.array([0.15, 0.0, 0.0])),
                     ("between", np.array([0.25, 0.0, 0.0])),
                     ("beyond", np.array([0.45, 0.0, 0.0]))):
        x = cp.Variable(3, name="dqa")
        cons = [x >= lo, cp.SOC(0.2, x)]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), cons)
        prob.solve(solver=cp.CLARABEL)
        rec = phase_one(prob, 0.2, cp.Problem.solve)
        rec["case"] = name
        rec["expected_rho"] = float(np.linalg.norm(lo))
        results.append(rec)
    x = cp.Variable(3, name="dqa")
    infeasible = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                            [x >= 1, x <= 0, cp.SOC(0.2, x)])
    infeasible.solve(solver=cp.CLARABEL)
    rec = phase_one(infeasible, 0.2, cp.Problem.solve)
    rec["case"] = "infeasible"
    results.append(rec)
    two_soc = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                         [x >= 0, cp.SOC(0.2, x), cp.SOC(0.3, x)])
    rec = phase_one(two_soc, 0.2, cp.Problem.solve)
    rec["case"] = "two_soc"
    results.append(rec)
    print(json.dumps({"self_test": results}, indent=2, default=_jsonable))
    ok = (results[0]["category"] == "rho_within_original_radius"
          and results[1]["category"] == "supports_step03_candidate"
          and results[2]["category"] == "rho_exceeds_candidate_range"
          and results[3]["category"] == "retained_infeasible"
          and results[4]["status"] == "not_single_soc")
    print(f"self-test {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("generation", "self-test"),
                    default="generation")
    ap.add_argument("--demo")
    ap.add_argument("--data-path")
    ap.add_argument("--save-dir")
    ap.add_argument("--snapshot-dir")
    ap.add_argument("--holosoma-package")
    ap.add_argument("--retargeter-sha256",
                    help="pinned sha256 of interaction_mesh_retargeter.py")
    ap.add_argument("--skip-config-dump", action="store_true")
    args = ap.parse_args()
    if args.mode == "self-test":
        return run_self_test()
    return run_wrapped_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
