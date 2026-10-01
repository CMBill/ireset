"""Shared forensic machinery for the train24fix2 review-fix round
(plan ``train24fix2_review_fix_plan.md`` sections 2-4).

Three pieces both forensic tools consume:

- **Context freeze/restore (R1)**: a field-complete deep copy of
  ``FrameContext`` built by iterating the dataclass FIELDS (a new field is
  picked up automatically -- the reviewed defect was a hand-written subset
  that replaced the human reference with the evaluator's robot FK), plus
  array/meta (JSON+NPZ) serialization and a bitwise field-by-field
  verifier.
- **Single-frame escalation replay (R1 section 2.2)**: the production
  frame escalation (V1 skip check with pair reflux -> prediction ->
  prev_accepted -> slack fallback, each failure refluxing discovered pairs
  through the same ``_solve_attempt`` the pipeline uses) on a FRESH
  pipeline, from a FROZEN context -- never re-derived from the candidate.
- **Full-candidate audit (R3 section 4.1)**: endpoint rows (fresh
  uncapped pair scan, all rows with values/bounds/scales), every protocol
  interpolation point (per-point pair-cache reset, passing points
  included), strict joint velocities, task MPJPE under the frozen human
  reference, and the PREFIX independent final validation -- with the
  single normalized-violation-SUM acceptance and the section-4.2
  classification vocabulary.

Nothing here is a decision path of the repair pipeline; the tools call it
on copies.  Products are versioned JSON manifests + NPZ arrays; the replay
verifier recomputes same-candidate branches from the frozen files.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ireset.repair.constraints import FrameEvaluator
from ireset.repair.pipeline import RepairPipeline, strict_joint_speeds
from ireset.repair.scp import FrameCheck, FrameContext, FrameSolver
from ireset.repair.validation import IndependentValidator

#: product format version (bump when the layout changes; replay refuses
#: mismatches)
AUDIT_FORMAT = "train24fix2_reviewfix_audit_v1"
#: acceptance: ONE normalized violation sum over all finite-bound rows
ACCEPT_VIOL = 1e-6


# ---------------------------------------------------------------------------
# R1: field-complete FrameContext freeze / restore / verify
# ---------------------------------------------------------------------------

def _freeze_value(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.ndarray):
        return v.copy()
    if issparse(v):
        return v.copy()
    if isinstance(v, dict):
        return {k: _freeze_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_freeze_value(x) for x in v)
    raise TypeError(f"FrameContext field of unhandled type {type(v)!r}")


def freeze_context(ctx: FrameContext) -> FrameContext:
    """Field-complete deep copy (arrays and nested dicts independent of the
    live context -- later FK/solve calls cannot mutate the frozen copy)."""
    kwargs = {f.name: _freeze_value(getattr(ctx, f.name))
              for f in dataclasses.fields(FrameContext)}
    return FrameContext(**kwargs)


def _values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (bool, int, float, str)):
        return type(a) is type(b) and a == b
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape
                and a.dtype == b.dtype
                and np.array_equal(a, b, equal_nan=True))
    if issparse(a):
        return (issparse(b) and a.shape == b.shape and a.dtype == b.dtype
                and a.nnz == b.nnz and (a != b).nnz == 0)
    if isinstance(a, dict):
        return (isinstance(b, dict) and set(a) == set(b)
                and all(_values_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_values_equal(x, y) for x, y in zip(a, b)))
    return a == b


def context_diff(a: FrameContext, b: FrameContext) -> list[str]:
    """Field names where two contexts differ (arrays compared bitwise).
    Iterating the dataclass fields keeps the verifier complete for future
    fields -- a silent subset can no longer hide one."""
    out = []
    for f in dataclasses.fields(FrameContext):
        if not _values_equal(getattr(a, f.name), getattr(b, f.name)):
            out.append(f.name)
    return out


def assert_snapshot_matches_live(frozen: FrameContext, live: FrameContext) -> None:
    """Self-check at capture time: the frozen copy equals the live context
    on every field (review section 2.1 -- deserialized/restored contexts
    are verified the same way)."""
    diff = context_diff(frozen, live)
    if diff:
        raise AssertionError(f"frozen context differs on fields: {diff}")


# -- JSON+NPZ serialization --------------------------------------------------

def context_arrays(ctx: FrameContext, prefix: str = "ctx") -> dict[str, np.ndarray]:
    """Every array of the context as ``{prefix}__{field}[__{key}]`` entries
    (sparse matrices as CSR data/indices/indptr/shape)."""
    out: dict[str, np.ndarray] = {}

    def put(name, v):
        if v is None:
            return
        if isinstance(v, np.ndarray):
            out[f"{prefix}__{name}"] = v
        elif issparse(v):
            c = v.tocsr()
            out[f"{prefix}__{name}__data"] = c.data
            out[f"{prefix}__{name}__indices"] = c.indices.astype(np.int64)
            out[f"{prefix}__{name}__indptr"] = c.indptr.astype(np.int64)
            out[f"{prefix}__{name}__shape"] = np.asarray(c.shape, dtype=np.int64)
        elif isinstance(v, dict):
            for k, x in v.items():
                put(f"{name}__{k}", x)

    for f in dataclasses.fields(FrameContext):
        put(f.name, getattr(ctx, f.name))
    return out


def context_meta(ctx: FrameContext) -> dict:
    """JSON-safe companion of :func:`context_arrays`: which fields are
    None / sparse / plain arrays, dict keys, and the SCALAR leaves (dict
    leaves that are arrays live in the NPZ bundle)."""
    meta: dict = {"fields_none": [], "fields_sparse": [], "fields_array": [],
                  "dicts": {}, "scalars": {}}
    for f in dataclasses.fields(FrameContext):
        name, v = f.name, getattr(ctx, f.name)
        if v is None:
            meta["fields_none"].append(name)
        elif issparse(v):
            meta["fields_sparse"].append(name)
        elif isinstance(v, np.ndarray):
            meta["fields_array"].append(name)
        elif isinstance(v, dict):
            leaves = {"keys": sorted(v.keys(), key=str), "none_keys": [],
                      "scalar_keys": []}
            for k, x in v.items():
                if x is None:
                    leaves["none_keys"].append(str(k))
                elif isinstance(x, np.ndarray):
                    pass                      # lives in the NPZ bundle
                elif isinstance(x, (bool, int, float, str)):
                    leaves["scalar_keys"].append(str(k))
                else:
                    raise TypeError(
                        f"context dict {name}[{k!r}] leaf of unhandled "
                        f"type {type(x)!r}")
            meta["dicts"][name] = leaves
        elif isinstance(v, (bool, int, float, str)):
            meta["scalars"][name] = v
        else:
            raise TypeError(f"unserializable context field {name}: {type(v)!r}")
    return meta


def context_scalar_leaves(ctx: FrameContext) -> dict:
    """``{field__key: value}`` of every SCALAR dict leaf (arrays excluded)
    -- serialized into the manifest so the replay restores them exactly."""
    out: dict = {}
    for f in dataclasses.fields(FrameContext):
        v = getattr(ctx, f.name)
        if isinstance(v, dict):
            for k, x in v.items():
                if x is not None and not isinstance(x, np.ndarray):
                    out[f"{f.name}__{k}"] = x
    return out


def context_from(arrays: dict, meta: dict, prefix: str = "ctx") -> FrameContext:
    """Rebuild the context from :func:`context_arrays` output + meta +
    scalar leaves (embedded by the caller under ``ctx_scalars``)."""
    scalars = meta.get("ctx_scalars", {})

    def get(name):
        key = f"{prefix}__{name}"
        if name in meta["fields_none"]:
            return None
        if name in meta["fields_sparse"]:
            return csr_matrix(
                (arrays[f"{key}__data"],
                 arrays[f"{key}__indices"], arrays[f"{key}__indptr"]),
                shape=tuple(arrays[f"{key}__shape"]))
        d = meta["dicts"].get(name)
        if d is not None:
            out = {}
            for k in d["keys"]:
                ks = str(k)
                if ks in d["none_keys"]:
                    out[k] = None
                elif ks in d["scalar_keys"]:
                    out[k] = scalars[f"{name}__{ks}"]
                else:
                    out[k] = arrays[f"{key}__{ks}"]
            return out
        if name in meta["scalars"]:
            return meta["scalars"][name]
        return arrays[key]

    kwargs = {f.name: get(f.name) for f in dataclasses.fields(FrameContext)}
    return FrameContext(**kwargs)


# ---------------------------------------------------------------------------
# R1 section 2.2: production-equivalent single-frame escalation replay
# ---------------------------------------------------------------------------

def escalate_frame(pipe: RepairPipeline, ctx: FrameContext, t: int,
                   z_pred: np.ndarray, z_prev: np.ndarray | None) -> dict:
    """Replay ONE frame's escalation EXACTLY as ``_run_sequential`` runs it:

    fresh pair state -> V1 skip check on the prediction (a failure feeds
    its violations back like production) -> prediction attempt ->
    prev_accepted restart (when a previous frame exists) -> slack fallback
    (when enabled), every failure refluxing its discovered pairs before
    the next attempt.  All attempts go through the pipeline's own
    ``_solve_attempt`` (same normalization, same transition backtracking,
    same attempt/check bookkeeping) on the FROZEN context -- the reference
    and anchors are never re-derived from the candidate.

    The caller owns pipeline freshness (independent evaluator/solver for
    each counterfactual branch).  Returns the skip record, every attempt
    record, and the adoption verdict with identities.
    """
    pipe.solver.probe_cfg = pipe.validator.cfg
    pipe.solver.clear_discovered_pairs()
    pipe.solver.clear_probe_pairs()
    attempts: list[dict] = []
    checks: list[dict] = []
    frame_recs: list[dict] = []

    # production order: V1-normalize the skip candidate, check, note
    skip_norm = pipe._normalize_candidate(np.asarray(z_pred, dtype=float).copy(), ctx)
    if skip_norm["ok"]:
        chk0 = pipe.frame_valid(skip_norm["z"], ctx)
    else:
        chk0 = FrameCheck(ok=False, reason=skip_norm["reason"])
    pipe._note_check(checks, chk0, "skip", t, None)
    skip_rec = {"ok": bool(chk0.ok), "reason": chk0.reason,
                "check_id": chk0.check_id,
                "velocity_normalized": bool(skip_norm.get("n_changed"))}
    if chk0.ok:
        # the skip return carries the SAME key set as the solve path (the
        # caller reads one shape; a missing key crashed the first formal
        # fail-bind of this round on a frame whose C2-off skip passes)
        return {"skipped": True, "skip": skip_rec, "adopted": True,
                "adopted_source": "skip", "adopted_z": skip_norm["z"].copy(),
                "adopted_status": "skipped",
                "last_status": "skipped", "last_term_reason": "",
                "last_check_reason": chk0.reason,
                "attempts": attempts, "checks": checks, "frame_recs": frame_recs}
    # a denied skip feeds the prediction's violations into the solve
    pipe.carry_discovered(chk0)

    plan = [("prediction", np.asarray(z_pred, dtype=float).copy(), False)]
    if z_prev is not None:
        plan.append(("prev_accepted", np.asarray(z_prev, dtype=float).copy(), False))
    if pipe.cfg.enable_slack_fallback:
        plan.append(("slack_fallback",
                     np.asarray(z_pred if z_prev is None else z_prev,
                                dtype=float).copy(), True))
    adopted = None
    last = None
    for source, z_start, force_slack in plan:
        res, chk, _bt, _bti, rec = pipe._solve_attempt(
            z_start, source, ctx, t, z_prev, len(frame_recs),
            attempts, checks, frame_recs, force_slack=force_slack)
        last = (res, chk, rec)
        if chk.ok:
            adopted = (source, res, chk, rec)
            break
        # production refluxes EVERY failed attempt's discoveries before the
        # next one starts
        pipe.carry_discovered(chk)
    res, chk, rec = last
    return {
        "skipped": False, "skip": skip_rec,
        "adopted": adopted is not None,
        "adopted_source": None if adopted is None else adopted[0],
        "adopted_z": None if adopted is None else adopted[1].z.copy(),
        "adopted_status": None if adopted is None else adopted[1].status,
        "last_status": res.status, "last_term_reason": res.term_reason,
        "last_check_reason": chk.reason,
        "attempts": attempts, "checks": checks, "frame_recs": frame_recs,
    }


# ---------------------------------------------------------------------------
# R3 section 4.1: full-candidate audit
# ---------------------------------------------------------------------------

def fresh_audit_pipe(spec, cfg, vcfg) -> RepairPipeline:
    """Independent evaluator/solver/validator per audit branch."""
    pipe = RepairPipeline(spec=spec, cfg=cfg)
    pipe.validator.cfg = vcfg
    return pipe


def _row_dump(solver: FrameSolver, val, lo, hi, scl, names) -> tuple[dict, dict]:
    """All rows with descriptors, values, bounds, scales and the single
    normalized-violation SUM acceptance (never per-family conjunctions)."""
    desc, _ids = solver._check_descriptors(list(names), [None] * len(names))
    viol = np.maximum(np.maximum(np.array(lo) - np.array(val),
                                 np.array(val) - np.array(hi)), 0.0) / np.array(scl)
    keep = list(desc)
    viol_norm = [float(v) for v in viol]
    rows = {"descriptors": keep, "n_rows": len(keep),
            "viol_sum": float(np.sum(viol_norm)),
            "accepted_sum": bool(np.sum(viol_norm) <= ACCEPT_VIOL),
            "violating": [keep[i] for i in np.nonzero(
                np.asarray(viol_norm) > 0)[0]]}
    return rows, {"val": np.asarray(val, dtype=float),
                  "lo": np.asarray(lo, dtype=float),
                  "hi": np.asarray(hi, dtype=float),
                  "scale": np.asarray(scl, dtype=float),
                  "viol": np.asarray(viol_norm, dtype=float)}


def audit_interpolation(pipe: RepairPipeline, ctx: FrameContext,
                        z: np.ndarray, z_prev: np.ndarray | None,
                        vcfg) -> dict:
    """EVERY protocol interpolation point between the previous full scene
    and the candidate, per-point pair-cache reset + fresh uncapped scan,
    collision family only (probe_mode=True, band_margin 0), PASSING points
    included -- an endpoint violation never skips the interpolation."""
    if z_prev is None or not vcfg.check_interpolated_states:
        return {"checked": False, "reason": "no_prev_or_disabled",
                "points": [], "pass": True}
    scene_a = np.zeros(43)
    scene_a[0:36] = z_prev
    scene_a[36:43] = ctx.box_pose_prev if ctx.box_pose_prev is not None else ctx.box_pose
    scene_b = np.zeros(43)
    scene_b[0:36] = z
    scene_b[36:43] = ctx.box_pose
    motion = pipe.validator._frame_motion_m(scene_a, scene_b)
    n_sub = int(np.clip(np.ceil(motion / vcfg.max_substep_motion_m),
                        1, vcfg.max_subdivisions + 1))
    points = []
    ok = True
    for i in range(1, n_sub):
        mid = pipe.validator._interpolate(scene_a, scene_b, i / n_sub)
        ctx_mid = dataclasses.replace(ctx, box_pose=mid[36:43].copy())
        pipe.solver._frame_pairs = None            # per-point rescan
        vm, lm, hm, sm, nm, _im = pipe.solver.check_rows(
            mid[0:36], ctx_mid, band_margin_m=0.0, probe_mode=True)
        n_coll = 99 + len(nm)
        viol_m = pipe.solver.violation(vm[99:n_coll], lm[99:n_coll],
                                       hm[99:n_coll], sm[99:n_coll])
        pt = {"i": i, "alpha": i / n_sub, "viol": float(viol_m),
              "pass": bool(viol_m <= ACCEPT_VIOL),
              "pairs": list(nm),
              "vals_m": [float(v) for v in vm[99:n_coll]],
              "los_m": [float(v) for v in lm[99:n_coll]],
              "his_m": [float(v) for v in hm[99:n_coll]]}
        if viol_m > ACCEPT_VIOL:
            ok = False
            pt["violating"] = [nm[k] if k < len(nm) else "?"
                               for k in range(n_coll - 99)
                               if vm[99 + k] < lm[99 + k] - 1e-9
                               or vm[99 + k] > hm[99 + k] + 1e-9]
        points.append(pt)
    return {"checked": True, "motion_m": float(motion), "n_sub": n_sub,
            "points": points, "pass": ok}


def prefix_validation(spec, vcfg, prefix: list[np.ndarray],
                      z: np.ndarray, box_poses: np.ndarray) -> dict:
    """Independent final validation of THIS PASS's accepted prefix + the
    candidate (box poses from the pass input, one per frame).  The
    conclusion covers the PREFIX ONLY -- never the remaining suffix or the
    whole segment."""
    scene = np.zeros((len(prefix) + 1, 43))
    for i, a in enumerate(prefix):
        scene[i, 0:36] = a
    scene[len(prefix), 0:36] = z
    scene[:, 36:43] = box_poses[: len(prefix) + 1]
    val = IndependentValidator(spec=spec)
    val.cfg = vcfg
    rep = val.validate(scene, exclude=("box_ground",))
    return {
        "scope": "accepted_prefix_plus_candidate",
        "passed": bool(rep.passed), "status": rep.status,
        "summary": rep.summary(),
        "checks": {k: {"passed": bool(c.passed),
                       "worst": float(c.worst), "limit": float(c.limit),
                       "location": c.location,
                       "first_failure_frame": c.first_failure_frame}
                   for k, c in rep.checks.items()},
        "failed": {k: {"location": c.location,
                       "first_failure_frame": c.first_failure_frame,
                       "evidence": c.evidence}
                   for k, c in rep.checks.items() if not c.passed},
    }


def full_candidate_audit(spec, cfg, vcfg, ctx: FrameContext, z_raw: np.ndarray,
                         *, prefix: list[np.ndarray], box_poses: np.ndarray,
                         label: str) -> tuple[dict, dict]:
    """The five checks of plan section 4.1 for ONE branch (the caller picks
    the C2 mode by ctx/cfg).  Returns ``(record, arrays)`` -- the record is
    JSON-safe, the arrays go to the NPZ bundle.  A query anomaly marks the
    branch ``evidence_incomplete`` instead of a verdict (R2 contract in the
    tools)."""
    from ireset.repair.pipeline import C2DistanceQueryError

    pipe = fresh_audit_pipe(spec, cfg, vcfg)
    z_raw = np.asarray(z_raw, dtype=float).copy()
    rec: dict = {"label": label, "format": AUDIT_FORMAT}

    # (1) input & V1 -- the actually-checked candidate is the normalized
    # one when normalization succeeds; an over-budget failure leaves the
    # RAW candidate unusable (never swap in a different state)
    finite = bool(np.isfinite(z_raw).all())
    norm = pipe._normalize_candidate(z_raw, ctx)
    z_checked = norm["z"] if norm["ok"] else z_raw
    vel_exc: list[dict] = []
    vel_ok = True
    if ctx.z_prev is not None:
        speeds = strict_joint_speeds(z_checked, ctx.z_prev, cfg)
        for j in np.nonzero(
                speeds > cfg.joint_velocity_limit_radps)[0]:
            vel_ok = False
            vel_exc.append({
                "joint": int(j), "value": float(speeds[j]),
                "limit": float(cfg.joint_velocity_limit_radps),
                "exceedance": float(speeds[j] - cfg.joint_velocity_limit_radps)})
    rec["input"] = {
        "shape_ok": z_raw.shape == (36,), "finite": finite,
        "v1": {k: (None if k == "z" else v) for k, v in norm.items()},
        "checked_candidate": "v1_normalized" if norm["ok"] and norm["n_changed"]
        else "raw" if not norm["ok"] else "raw_unchanged",
        "strict_velocity": {"ok": vel_ok, "exceedances": vel_exc},
    }

    # (2) endpoint rows: fresh uncapped scan on the restored context
    try:
        val, lo, hi, scl, names, _ids = pipe.solver.check_rows(z_checked, ctx)
    except C2DistanceQueryError as err:
        rec["verdict"] = "EVIDENCE_INCOMPLETE"
        rec["evidence_gap"] = {"cause": "c2_distance_query_anomaly",
                               "error": err.record()}
        return rec, {}
    rows, row_arrays = _row_dump(pipe.solver, val, lo, hi, scl, names)
    rows["c2_last_hand_speed"] = {s: dict(v) for s, v in
                                  pipe.solver.last_hand_speed.items()}
    rec["endpoint"] = rows
    # (3) every interpolation point (per-point rescan; passing included)
    inter = audit_interpolation(pipe, ctx, z_checked, ctx.z_prev, vcfg)
    rec["interpolation"] = {k: v for k, v in inter.items() if k != "points"}
    rec["interpolation"]["points"] = inter["points"]
    inter_ok = bool(inter["pass"])

    # (4) task under the frozen HUMAN reference (fresh evaluator state
    # cannot change it: forward is deterministic in (z, box))
    mpjpe = float(pipe.solver.evaluate(z_checked, ctx)["mpjpe"])
    task_ok = bool(mpjpe <= cfg.task_mpjpe_tol_m)
    rec["task"] = {"mpjpe_m": mpjpe, "gate_m": float(cfg.task_mpjpe_tol_m),
                   "ok": task_ok}

    # (5) prefix independent final validation (new validator)
    pref = prefix_validation(spec, vcfg, prefix, z_checked, box_poses)
    rec["prefix_validation"] = pref

    rec["branch_pass"] = bool(rows["accepted_sum"] and inter_ok
                              and vel_ok and task_ok and pref["passed"])
    # per-branch arrays are label-prefixed: two branches of one product
    # bundle must never overwrite each other's row values
    arrays = {f"rows__{label}__{k}": v for k, v in row_arrays.items()}
    arrays[f"rows__{label}__z_raw"] = z_raw
    arrays[f"rows__{label}__z_checked"] = z_checked
    return rec, arrays


# ---------------------------------------------------------------------------
# R3 section 4.2: classification
# ---------------------------------------------------------------------------

def classify_branches(current: dict | None, plus_c2: dict | None) -> str:
    """Two full branches -> the section-4.2 category.  ``current`` is the
    frozen-chain (no-C2) audit, ``plus_c2`` the same candidate with the C2
    rows on.  Incomplete evidence never counts as PASS/C2_ONLY."""
    for br in (current, plus_c2):
        if br is None or br.get("verdict") == "EVIDENCE_INCOMPLETE":
            return "EVIDENCE_INCOMPLETE"
        if br.get("input", {}).get("v1", {}).get("ok") is False:
            return "V1_UNUSABLE"
    if current["branch_pass"]:
        return "PASS" if plus_c2["branch_pass"] else "C2_ONLY"
    return "BOTH" if not plus_c2["branch_pass"] else "CURRENT_ONLY"


# ---------------------------------------------------------------------------
# R3 section 4.3: versioned products + replay verifier
# ---------------------------------------------------------------------------

def write_products(out_dir: Path, stem: str, record: dict,
                   arrays: dict) -> tuple[Path, Path]:
    """Write ``{stem}.json`` + ``{stem}.npz``; REFUSE to overwrite existing
    forensic files (the old tool's default path silently clobbered the
    frozen diag directory)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath, npath = out_dir / f"{stem}.json", out_dir / f"{stem}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise FileExistsError(
                f"refusing to overwrite existing forensic file {p} "
                f"(old products are read-only; choose a new --out)")
    record = dict(record)
    record["format"] = AUDIT_FORMAT
    record["npz"] = npath.name
    jpath.write_text(json.dumps(record, indent=1, allow_nan=False,
                                default=_json_default))
    np.savez_compressed(npath, **arrays)
    return jpath, npath


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value in forensic record: {v}")
        return v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"unserializable forensic value {type(o)!r}")


def load_products(out_dir: Path, stem: str) -> tuple[dict, dict]:
    jpath = out_dir / f"{stem}.json"
    record = json.loads(jpath.read_text())
    if record.get("format") != AUDIT_FORMAT:
        raise ValueError(f"{jpath}: format {record.get('format')!r} != "
                         f"{AUDIT_FORMAT!r}")
    with np.load(out_dir / record["npz"]) as z:
        arrays = {k: z[k] for k in z.files}
    return record, arrays


def json_safe(obj) -> str:
    """Canonical serialization used by the replay comparison (bitwise on
    the exact saved values; NaN/Inf literals are rejected upstream)."""
    return json.dumps(obj, sort_keys=True, allow_nan=False,
                      default=_json_default)
