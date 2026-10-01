"""Counterexamples for the train24fix2 implementation-review fixes
(plan ``train24fix2_review_fix_plan.md``, 2026-10-01).

Every test pins a confirmed defect of the reviewed round (or a strength
requirement the corrected conclusions must obey):

- R1: the fail-binding context rebuild replaced the HUMAN reference with the
  evaluator's current robot FK (task MPJPE of sub12@12/sub3@14 collapsed
  0.114154/0.131966 m -> 0) and dropped the hand targets / reference deltas /
  frame motion.  Pinned by the round-trip and escalation-equivalence tests.
- R2: a NaN return of the C2 post-adoption hand/box distance query was
  classified as "hand beyond the 1 m scan cap" (rows off) and the run could
  still end ACCEPTED.  Every anomaly variant must reject the run explicitly.
- R3: the same-candidate C2-off counterfactual checked endpoint rows only;
  acceptance is the single normalized-violation SUM over all finite rows.
- R4: ``status="infeasible"`` conflates budget exhaustion with structural
  infeasibility -- termination facts must be recorded, and budget-exhaustion
  evidence must not be reported as proven joint infeasibility.
- R5: the interpolation-audit cache trap must be exercised with an
  ESTABLISHED endpoint cache (see test_train24fix_review2).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.scp import FrameContext, FrameResult  # noqa: E402
from test_train24fix_review2 import _self_cond  # noqa: E402
from test_repair import standing_scene  # noqa: E402

REGEN = replace(RepairConfig(), regen_mode=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _poison_c2_queries(pipe, value, when=None):
    """Return ``geom_distance`` wrapper that returns ``value`` for EXACTLY
    the query signature ``_collect_prev_scene_dist`` issues (hand geom x box
    geom at the default 1 m range) while delegating everything else.
    ``when(state)`` gates on the frame whose collection may run (tracked via
    the read-only snapshot hook: pass index + frame index)."""
    real = pipe.ev.geom_distance
    box = pipe.ev.box_geom
    hands = set(pipe.ev._hand_geoms["l"]) | set(pipe.ev._hand_geoms["r"])
    state = {"pass": 0, "t": -1}

    def _snap(kind, data):
        if kind == "frame":
            state["pass"], state["t"] = data["pass_index"], data["t"]
    pipe.snapshot = _snap

    def wrapped(ga, gb, distmax=1.0):
        if (ga in hands and gb == box and distmax == 1.0
                and (when is None or when(state))):
            if isinstance(value, Exception):
                raise value
            return float(value), np.zeros(6)
        return real(ga, gb, distmax=distmax)
    pipe.ev.geom_distance = wrapped
    return state


def _run_standing(pipe, n=3):
    scene = standing_scene(n)
    cond = _self_cond(pipe, scene)
    return pipe.repair_trajectory(scene, cond), scene


# ---------------------------------------------------------------------------
# R2: query anomalies must reject the run (never read as "far away")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,kind", [
    (float("nan"), "nan"),
    (float("inf"), "pos_inf"),
    (float("-inf"), "neg_inf"),
    (1.2, "out_of_range"),
])
def test_query_anomaly_rejects_the_run(value, kind) -> None:
    """Every anomalous RETURN variant (NaN/±Inf/out-of-API-range) must
    reject the trajectory with ``final_stage=c2_distance_query_failed`` and
    a structured error record.  Pre-fix: NaN fell through ``d < 1.0`` as a
    miss, rows turned off, and the run could end ACCEPTED."""
    pipe = RepairPipeline(cfg=REGEN)
    _poison_c2_queries(pipe, value)
    res, _ = _run_standing(pipe, 3)
    assert not res.accepted, f"{kind} must not be classified as far-away"
    assert res.status == "rejected_frame"
    diag = res.diagnostics
    assert diag["final_stage"] == "c2_distance_query_failed"
    err = diag["c2_query_error"]
    assert err["value_kind"] == kind
    assert err["distmax"] == 1.0
    assert err["stage"] == "post_adoption_collection"
    assert err["side"] in ("l", "r")
    assert err["geom"] and isinstance(err["raw_repr"], str)
    assert err["t"] == 0            # collection runs right after t=0's adoption
    # the error belongs to the post-adoption collection, not to any check
    assert diag["reason"] == "c2_distance_query_anomaly"
    # cost facts of everything that DID run are preserved
    assert diag["cost"]["schema"] == 2 and diag["accepted_frames"] >= 1


def test_query_exception_rejects_collection() -> None:
    """A RAISED query aborts the collection with the same explicit error
    (unit level: the pipeline-level rejection is carried by the identical
    catch, proven for the value variants above)."""
    pipe = RepairPipeline(cfg=REGEN)
    scene = standing_scene(1)
    z, box = scene[0, 0:36].copy(), scene[0, 36:43].copy()
    pipe.ev.forward(z, box)
    real = pipe.ev.geom_distance
    g0 = pipe.ev._hand_geoms["l"][0]
    pipe.ev.geom_distance = (
        lambda ga, gb, distmax=1.0:
        (_ for _ in ()).throw(RuntimeError("mj_geomDistance blew up"))
        if (ga == g0 and gb == pipe.ev.box_geom and distmax == 1.0)
        else real(ga, gb, distmax=distmax))
    from ireset.repair.pipeline import C2DistanceQueryError
    with pytest.raises(C2DistanceQueryError) as ei:
        pipe._collect_prev_scene_dist(0)
    assert ei.value.value_kind == "exception"
    assert "blew up" in ei.value.record()["raw_repr"]
    assert pipe._c2_diag["query_errors"][-1]["value_kind"] == "exception"


def test_one_bad_geom_poisons_the_side() -> None:
    """An anomaly on ONE hand geom rejects even when other geoms of the same
    side return valid finite hits -- the failing geom may be the truly
    nearest one, so a partial-hit side is never 'normal'."""
    pipe = RepairPipeline(cfg=REGEN)
    real = pipe.ev.geom_distance
    box = pipe.ev.box_geom
    first = pipe.ev._hand_geoms["l"][0]
    pipe.ev.geom_distance = (
        lambda ga, gb, distmax=1.0:
        (float("nan"), np.zeros(6))
        if (ga == first and gb == box and distmax == 1.0)
        else real(ga, gb, distmax=distmax))
    res, _ = _run_standing(pipe, 2)
    assert not res.accepted
    assert res.diagnostics["final_stage"] == "c2_distance_query_failed"
    assert res.diagnostics["c2_query_error"]["value_kind"] == "nan"


def test_normal_truncation_and_real_hits_unchanged() -> None:
    """dist==distmax (normal truncation, sub16-style far hand) keeps the
    historical miss record; negative/zero real hits are kept as distances.
    Neither may reject."""
    pipe = RepairPipeline(cfg=REGEN)
    _poison_c2_queries(pipe, 1.0)              # exactly distmax: a miss
    res, _ = _run_standing(pipe, 2)
    assert res.accepted, "normal truncation must not reject"
    misses = pipe._c2_diag["query_miss"]
    assert misses and all(m["scope"] == "side_all_miss" for m in misses)

    pipe2 = RepairPipeline(cfg=REGEN)
    _poison_c2_queries(pipe2, -0.001)          # real (penetrating) hit
    res2, _ = _run_standing(pipe2, 1)
    assert res2.accepted or res2.status == "rejected_frame"
    assert pipe2._c2_diag.get("query_miss", []) == []


def test_last_frame_anomaly_rejects() -> None:
    """The collection of the FINAL frame has no successor -- its anomaly
    must still reject the run (no 'last frame is done anyway' silent pass)."""
    pipe = RepairPipeline(cfg=REGEN)
    _poison_c2_queries(pipe, float("nan"), when=lambda s: s["t"] == 2 and s["pass"] == 0)
    res, _ = _run_standing(pipe, 3)
    assert not res.accepted
    assert res.diagnostics["final_stage"] == "c2_distance_query_failed"
    assert res.diagnostics["c2_query_error"]["t"] == 2


def test_retry_pass_anomaly_rejects(monkeypatch) -> None:
    """An anomaly inside the SUFFIX-RETRY pass propagates too: the run ends
    rejected, never accepted."""
    pipe = RepairPipeline(cfg=REGEN)
    calls = {"n": 0}
    real_validate = pipe.validator.validate

    def fake_validate(traj, exclude=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # force one final-validation failure -> a suffix retry runs
            class FakeCheck:
                passed = False
                location = "fake"
                first_failure_frame = 1
                worst = 1.0
                limit = 0.0
                evidence = {}
            class FakeReport:
                passed = False
                checks = {"fake": FakeCheck()}
                status = "fake_failed"
                def summary(self):
                    return "fake_failed"
            return FakeReport()
        return real_validate(traj, exclude=exclude)
    monkeypatch.setattr(pipe.validator, "validate", fake_validate)
    _poison_c2_queries(pipe, float("nan"), when=lambda s: s["pass"] >= 1)
    res, _ = _run_standing(pipe, 3)
    assert calls["n"] >= 1
    assert not res.accepted
    assert res.diagnostics["final_stage"] == "c2_distance_query_failed"


def test_disabled_states_never_query_or_reject() -> None:
    """Standard mode and parameter-0 regen issue NO C2 distance queries: a
    poisoned query cannot reject the run there (the collection itself is
    gated off)."""
    for cfg in (RepairConfig(), replace(REGEN, regen_hand_speed_mps=0.0)):
        pipe = RepairPipeline(cfg=cfg)
        _poison_c2_queries(pipe, float("nan"))
        res, _ = _run_standing(pipe, 2)
        assert pipe._c2_diag["query_miss"] == []
        assert res.status != "rejected_frame" or \
            res.diagnostics.get("final_stage") != "c2_distance_query_failed"
        assert not (res.accepted and
                    res.diagnostics.get("final_stage") == "c2_distance_query_failed")


# ---------------------------------------------------------------------------
# R1: field-complete context freeze / restore
# ---------------------------------------------------------------------------

def _run_with_events(pipe, scene):
    """Run a trajectory capturing every frame event's ctx (live refs)."""
    events = []
    pipe.snapshot = lambda kind, data: (
        events.append((kind, data)) if kind in ("frame", "fail") else None)
    cond = _self_cond(pipe, scene)
    res = pipe.repair_trajectory(scene, cond)
    return res, cond, events


def _bump_scene(n=2):
    scene = standing_scene(n)
    limdt = 1.0 / RepairConfig().fps * 12.0
    scene[1, 7 + 26] = scene[0, 7 + 26] + limdt + 2e-6   # deny the skip
    return scene


def test_frozen_context_keeps_human_reference_and_all_c2_fields() -> None:
    """The reviewed defect class: the context was rebuilt from the
    evaluator's CURRENT robot FK.  The frozen context must carry the HUMAN
    reference, the hand targets, the reference deltas and the frame motion
    of the run's own context -- and survive later FK calls untouched."""
    import dataclasses as dc

    from train24fix2_audit import (
        assert_snapshot_matches_live, context_diff, freeze_context)
    from ireset.embodied.contact_sites import SMPLH_INDEX
    from ireset.embodied.g1_spec import KPT_TO_SMPLH

    pipe = RepairPipeline(cfg=REGEN)
    res, cond, events = _run_with_events(pipe, _bump_scene())
    assert res.accepted
    fr = next(d for k, d in events if k == "frame" and d["t"] == 1)
    ctx = fr["ctx"]
    frozen = freeze_context(ctx)
    assert_snapshot_matches_live(frozen, ctx)

    kpt_idx = [SMPLH_INDEX[v] for v in KPT_TO_SMPLH.values()]
    human_ref = cond.human_kpts_world[1][kpt_idx]
    assert np.array_equal(frozen.ref_kpts_world, human_ref)
    pipe.ev.forward(np.zeros(36) + 0.7, fr["box"])     # trash the FK state
    assert np.array_equal(frozen.ref_kpts_world, human_ref), \
        "a later FK must not be able to become the audit reference"
    assert frozen.ref_hand_kpts and frozen.ref_hand_delta
    assert frozen.frame_motion_m > 0.0
    # the defective rebuild (robot FK as reference) is DETECTABLE by the
    # field-complete verifier
    bad = FrameContext(**{**dc.asdict(frozen), "ref_kpts_world":
                          pipe.ev.site_positions(pipe.ev.kpt_ids).copy()})
    assert "ref_kpts_world" in context_diff(bad, frozen)


def test_freeze_copies_every_field_no_shared_references() -> None:
    """Every array field of the frozen context is an independent copy (a
    mutation of the live context after freezing cannot leak in), covering
    ALL dataclass fields -- a hand-picked subset is exactly the R1 defect."""
    import dataclasses as dc

    from train24fix2_audit import context_diff, freeze_context

    pipe = RepairPipeline(cfg=REGEN)
    res, _cond, events = _run_with_events(pipe, _bump_scene())
    fr = next(d for k, d in events if k == "frame" and d["t"] == 1)
    frozen = freeze_context(fr["ctx"])
    mutated = 0
    for f in dc.fields(FrameContext):
        live_v = getattr(fr["ctx"], f.name)
        if isinstance(live_v, np.ndarray):
            assert getattr(frozen, f.name) is not live_v
            live_v += 0.1234567
            mutated += 1
    assert mutated >= 7   # every top-level array field
    assert context_diff(frozen, fr["ctx"])


def test_context_json_npz_roundtrip_bitwise(tmp_path) -> None:
    """Freeze -> (arrays + JSON meta) -> restore is bitwise identical,
    including the sparse mesh Laplacian; the verifier completes over the
    dataclass fields so a future field cannot be silently dropped."""
    import json as _json

    import numpy.testing as npt

    from train24fix2_audit import (context_arrays, context_diff, context_from,
                                   context_meta, context_scalar_leaves,
                                   freeze_context)

    pipe = RepairPipeline(cfg=REGEN)
    res, _cond, events = _run_with_events(pipe, _bump_scene())
    fr = next(d for k, d in events if k == "frame" and d["t"] == 1)
    frozen = freeze_context(fr["ctx"])
    assert frozen.mesh_L is not None and frozen.delta_ref is not None
    meta = _json.loads(_json.dumps({**context_meta(frozen),
                                    "ctx_scalars": context_scalar_leaves(frozen)}))
    npz = tmp_path / "ctx.npz"
    np.savez_compressed(npz, **context_arrays(frozen))
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    restored = context_from(arrays, meta)
    assert context_diff(restored, frozen) == []
    npt.assert_array_equal(restored.mesh_L.toarray(), frozen.mesh_L.toarray())


# ---------------------------------------------------------------------------
# R1: single-frame escalation replay == production frame escalation
# ---------------------------------------------------------------------------

def test_escalation_matches_production_solve_adopted() -> None:
    """A solve-adopted frame: the replay (fresh pipeline, frozen context)
    must reproduce the production attempt chain, the adopted candidate
    BITWISE and the skip denial."""
    from train24fix2_audit import escalate_frame, freeze_context
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    scene = _bump_scene()
    res, _cond, events = _run_with_events(pipe, scene)
    assert res.diagnostics["frame_status"][1] != "skipped"
    fr = next(d for k, d in events if k == "frame" and d["t"] == 1)

    _rcfg, vcfg = load_mode_configs("strict")
    pipe2 = RepairPipeline(cfg=REGEN)
    pipe2.validator.cfg = vcfg
    esc = escalate_frame(pipe2, freeze_context(fr["ctx"]), 1,
                         fr["z_pred"], fr["z_prev"])
    assert esc["skip"]["ok"] is False                 # skip denied (budget)
    assert not esc["skipped"] and esc["adopted"]
    assert np.array_equal(esc["adopted_z"], res.trajectory[1, 0:36])
    assert len(esc["attempts"]) >= 1
    assert esc["attempts"][0]["source"] == "prediction"
    assert esc["checks"][0]["source"] == "skip"       # check ids: skip first


def test_escalation_matches_production_skip_t0() -> None:
    """t=0 (no previous frame): production skips; the replay skips on the
    same normalized candidate -- the no-prev branch executes prediction
    without prev_accepted."""
    from train24fix2_audit import escalate_frame, freeze_context
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    scene = standing_scene(2)
    res, _cond, events = _run_with_events(pipe, scene)
    assert res.diagnostics["frame_status"][0] == "skipped"
    fr = next(d for k, d in events if k == "frame" and d["t"] == 0)
    assert fr["z_prev"] is None

    _rcfg, vcfg = load_mode_configs("strict")
    pipe2 = RepairPipeline(cfg=REGEN)
    pipe2.validator.cfg = vcfg
    esc = escalate_frame(pipe2, freeze_context(fr["ctx"]), 0,
                         fr["z_pred"], None)
    assert esc["skipped"] and esc["adopted_source"] == "skip"
    assert esc["attempts"] == []


def test_escalation_matches_production_all_attempts_fail() -> None:
    """A deep-penetration frame: production burns prediction ->
    prev_accepted -> slack (each failure refluxing discovered pairs); the
    replay reproduces the three attempts and their solver exit facts, and
    the pair reflux actually happened (carried extras non-empty)."""
    from train24fix2_audit import escalate_frame, freeze_context
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    scene = standing_scene(2)          # no V1 bump: failures must come from
    box_far = np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0])   # the collision rows
    # box centered ON the root: the torso sits deep inside -- no single-frame
    # repair can satisfy the collision floor plus every other row
    root = scene[0, 0:3].copy()
    box_hit = np.array([root[0], root[1], root[2] + 0.02, 1.0, 0, 0, 0])
    scene[:, 36:43] = box_hit
    scene[0, 36:43] = box_far                        # overlap starts at t=1
    res, _cond, events = _run_with_events(pipe, scene)
    assert res.status == "rejected_frame" and res.diagnostics["t_fail"] == 1
    fr = next(d for k, d in events if k == "frame" and d["t"] == 1)

    _rcfg, vcfg = load_mode_configs("strict")
    pipe2 = RepairPipeline(cfg=REGEN)
    pipe2.validator.cfg = vcfg
    esc = escalate_frame(pipe2, freeze_context(fr["ctx"]), 1,
                         fr["z_pred"], fr["z_prev"])
    assert not esc["adopted"] and not esc["skipped"]
    sources = [a["source"] for a in esc["attempts"]]
    assert sources == ["prediction", "prev_accepted", "slack_fallback"]
    # production ran the same three attempts on this frame
    prod = res.diagnostics["cost"]["passes"][0]["attempts_detail"]
    assert [a["source"] for a in prod if a["t"] == 1] == sources
    # every attempt records its verbatim exit cause (R4 facts present)
    assert all(a.get("term_reason") for a in esc["attempts"])
    # failed checks fed pairs back into the next linearization
    assert pipe2.solver._extra_pairs, "pair reflux between attempts is live"


# ---------------------------------------------------------------------------
# R3: acceptance is ONE sum; the audit covers every check
# ---------------------------------------------------------------------------

def test_acceptance_is_the_single_sum_not_family_conjunction() -> None:
    """Two rows each violating 5e-7 of their own scale (each family
    individually within the 1e-6 budget) sum past 1e-6 and MUST reject --
    including a C2-side row joining another family.  The per-family
    conjunction reading is the reviewed mis-implementation shape."""
    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    ctx = FrameContext(
        t=1, box_pose=np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0]),
        z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=None, ref_hold={}, prev_hand_dist={})
    pipe.ev.forward(z, ctx.box_pose)
    val, lo, hi, scl, _names, _ids = pipe.solver.check_rows(z, ctx)
    viol = np.maximum(np.maximum(lo - val, val - hi), 0.0) / scl
    base = float(viol.sum())
    assert base <= 1e-6
    # one bounded row pushed 6e-7 (normalized) past its lower bound: alone
    # it stays within the budget, and a SECOND row of a different family
    # pushes the SUM over -- the conjunction reading would accept both
    k = int(np.argmax((np.isfinite(lo)) & (np.isfinite(hi))))
    val2 = val.copy()
    val2[k] = lo[k] - 6e-7 * scl[k]
    viol2 = np.maximum(np.maximum(lo - val2, val2 - hi), 0.0) / scl
    assert float(viol2.sum()) <= base + 7e-7           # still tiny per family
    assert pipe.solver.violation(val2, lo, hi, scl) <= 1e-6  # ONE row passes
    k2 = int(np.argmax((np.isfinite(lo)) & (np.isfinite(hi)) & (np.arange(len(lo)) != k)))
    val3 = val2.copy()
    val3[k2] = lo[k2] - 6e-7 * scl[k2]
    assert pipe.solver.violation(val3, lo, hi, scl) > 1e-6, \
        "the SUM must reject even when each family is individually within"


def test_audit_catches_interpolation_despite_endpoint_pass() -> None:
    """Endpoint rows pass (box far at both ends) but the box sweeps through
    the robot at an interior subdivision state: branch_pass must be False
    via the interpolation, with the interior alpha and pair recorded."""
    from train24fix2_audit import full_candidate_audit
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    pipe.ev.forward(z, np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0]))
    root = pipe.ev.data.geom_xpos[pipe.ev._hand_geoms["l"][0]]
    box_a = np.array([root[0] - 0.30, root[1], root[2], 1.0, 0, 0, 0])
    box_b = np.array([root[0] + 0.60, root[1], root[2], 1.0, 0, 0, 0])
    ctx = FrameContext(
        t=1, box_pose=box_b, z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=box_a, ref_hold={}, prev_hand_dist={},
        frame_motion_m=0.5)
    _rcfg, vcfg = load_mode_configs("strict")
    rec, _arrays = full_candidate_audit(
        pipe.spec, REGEN, vcfg, ctx, z, prefix=[z.copy()],
        box_poses=np.stack([box_a, box_b]), label="t")
    assert rec["endpoint"]["accepted_sum"] is True     # endpoint alone passes
    assert rec["interpolation"]["pass"] is False
    bad = [p for p in rec["interpolation"]["points"] if not p["pass"]]
    assert bad and 0.0 < bad[0]["alpha"] < 1.0
    assert any("largebox" in p for p in bad[0]["violating"])
    assert rec["branch_pass"] is False


def test_audit_collects_all_checks_despite_endpoint_failure() -> None:
    """An endpoint-violating candidate still gets its interpolation points,
    strict velocities, task value and prefix validation recorded (no early
    exit hides the rest)."""
    from test_band_floor_separation import _place_box

    from train24fix2_audit import full_candidate_audit
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    z = standing_scene(1)[0, 0:36].copy()
    pipe.ev.forward(z, np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0]))
    box = _place_box(pipe, z, "l", -0.03)             # deep overlap
    ctx = FrameContext(
        t=1, box_pose=box, z_pred=z.copy(), z_prev=z.copy(),
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=box, ref_hold={}, prev_hand_dist={})
    _rcfg, vcfg = load_mode_configs("strict")
    rec, _arrays = full_candidate_audit(
        pipe.spec, REGEN, vcfg, ctx, z, prefix=[z.copy()],
        box_poses=np.stack([box, box]), label="t")
    assert rec["endpoint"]["accepted_sum"] is False
    assert rec["endpoint"]["violating"]
    assert "checked" in rec["interpolation"]
    assert rec["task"]["mpjpe_m"] >= 0.0 and "ok" in rec["task"]
    assert "passed" in rec["prefix_validation"]
    assert rec["branch_pass"] is False


def test_audit_prefix_drift_failure_rows_all_pass() -> None:
    """Rows pass everywhere, yet the PREFIX fails validation: a hand held
    9.5 mm from the box (inside the validator's 10 mm entry) while the box
    slides 22 mm tangentially accumulates > 20 mm box-frame anchor drift --
    the 'static rows pass, cumulative anchor drift fails the prefix' audit
    requirement.  Regen frame rows carry no hand anchors, so nothing at
    frame level sees it."""
    from test_band_floor_separation import _place_box

    from train24fix2_audit import full_candidate_audit
    from ireset.embodied.contact_sites import SMPLH_INDEX
    from ireset.embodied.g1_spec import KPT_TO_SMPLH
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    scene = standing_scene(27)
    box0 = _place_box(pipe, scene[0, 0:36].copy(), "l", 0.0095)
    scene[:, 36:43] = box0
    for t in range(4, 27):
        scene[t, 37] -= min(0.00125 * (t - 3), 0.022)   # tangential slide
    cond = _self_cond(pipe, scene)                    # perfect tracking
    _rcfg, vcfg = load_mode_configs("strict")

    pipe.ev.forward(scene[26, 0:36], scene[26, 36:43])
    ctx = FrameContext(
        t=26, box_pose=scene[26, 36:43].copy(),
        z_pred=scene[26, 0:36].copy(), z_prev=scene[25, 0:36].copy(),
        ref_kpts_world=cond.human_kpts_world[26][
            [SMPLH_INDEX[v] for v in KPT_TO_SMPLH.values()]],
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={}, prev_hand_local={}, prev_sole_world={},
        box_pose_prev=scene[25, 36:43].copy(), ref_hold={}, prev_hand_dist={})
    rec, _arrays = full_candidate_audit(
        pipe.spec, REGEN, vcfg, ctx, scene[26, 0:36],
        prefix=[scene[i, 0:36].copy() for i in range(26)],
        box_poses=scene[:, 36:43], label="t")
    # per-frame world: rows/interpolation/velocities/task all pass
    assert rec["endpoint"]["accepted_sum"] is True, rec["endpoint"]["violating"]
    assert rec["interpolation"]["pass"] is True
    assert rec["input"]["strict_velocity"]["ok"] is True
    assert rec["task"]["ok"] is True
    # ...but the independent prefix validation fails on hand anchor drift
    assert rec["prefix_validation"]["passed"] is False, \
        "the tangential slide must break the prefix anchor check"
    assert rec["branch_pass"] is False


# ---------------------------------------------------------------------------
# R4 + classification vocabulary + product identity
# ---------------------------------------------------------------------------

def test_term_reason_recorded_on_solver_failure() -> None:
    """A failing frame records the VERBATIM loop-exit cause next to the
    historical ``status`` label: budget exhaustion / trust collapse are
    search facts, and 'infeasible' alone must not be read as proven joint
    infeasibility."""
    pipe = RepairPipeline(cfg=REGEN)
    scene = _bump_scene()
    root = scene[0, 0:3].copy()
    scene[:, 36:43] = np.array([root[0], root[1], root[2] + 0.02, 1.0, 0, 0, 0])
    scene[0, 36:43] = np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0])
    res, _cond, _events = _run_with_events(pipe, scene)
    assert res.status == "rejected_frame"
    diag = res.diagnostics
    assert diag["last_attempt"]["status"] in (
        "infeasible", "max_iters", "task_quality", "review_failed")
    assert diag["last_term_reason"] in (
        "budget_exhausted", "no_merit_step_trust_min",
        "restoration_qp_failed_trust_min"), diag["last_term_reason"]
    assert diag["last_attempt"]["term_reason"] == diag["last_term_reason"]


def test_classification_vocabulary_strength() -> None:
    """The section-4.2 table: incomplete evidence or an unusable V1 never
    counts as PASS/C2_ONLY; CURRENT_ONLY/BOTH must never be phrased as the
    C2 sole cause; verdicts come from FULL branches (endpoint AND
    interpolation AND velocity AND task AND prefix)."""
    from train24fix2_audit import classify_branches

    def branch(endpoint=True, inter=True, vel=True, task=True, prefix=True,
               v1_ok=True, incomplete=False):
        if incomplete:
            return {"verdict": "EVIDENCE_INCOMPLETE"}
        return {"input": {"v1": {"ok": v1_ok}},
                "branch_pass": endpoint and inter and vel and task and prefix}

    assert classify_branches(branch(), branch()) == "PASS"
    assert classify_branches(branch(), branch(prefix=False)) == "C2_ONLY"
    assert classify_branches(branch(task=False), branch()) == "CURRENT_ONLY"
    assert classify_branches(branch(inter=False), branch(prefix=False)) == "BOTH"
    assert classify_branches(None, branch()) == "EVIDENCE_INCOMPLETE"
    assert classify_branches(branch(incomplete=True), branch()) == \
        "EVIDENCE_INCOMPLETE"
    assert classify_branches(branch(v1_ok=False), branch()) == "V1_UNUSABLE"
    # a branch that passes the endpoint only is NOT a passing branch
    assert classify_branches(branch(inter=False), branch()) == "CURRENT_ONLY"


def test_products_refuse_overwrite_and_replay_matches(tmp_path) -> None:
    """Forensic files are never overwritten, and the saved audit recomputes
    identically from the frozen arrays (the replay cannot substitute a
    re-found candidate)."""
    from train24fix2_audit import (classify_branches, context_arrays,
                                   context_from, context_meta,
                                   context_scalar_leaves, freeze_context,
                                   full_candidate_audit, load_products,
                                   write_products)
    from ireset.repair.config_io import load_mode_configs

    pipe = RepairPipeline(cfg=REGEN)
    res, _cond, events = _run_with_events(pipe, _bump_scene())
    assert res.accepted
    fr = next(d for k, d in events if k == "frame" and d["t"] == 1)
    frozen = freeze_context(fr["ctx"])
    _rcfg, vcfg = load_mode_configs("strict")
    prefix = [res.trajectory[0, 0:36].copy()]
    boxes = np.stack([fr["box"], fr["ctx"].box_pose])
    z = res.trajectory[1, 0:36].copy()
    rec_on, arr_on = full_candidate_audit(pipe.spec, REGEN, vcfg, frozen, z,
                                          prefix=prefix, box_poses=boxes,
                                          label="c2_on")
    rec_off, _ = full_candidate_audit(
        pipe.spec, REGEN, vcfg,
        replace(freeze_context(frozen), c2_hand_enabled={}, c2_hand_reason={}),
        z, prefix=prefix, box_poses=boxes, label="c2_off")
    record = {"demo": "controlled",
              "classification": classify_branches(rec_off, rec_on),
              "audits": {"c2_on": rec_on, "c2_off": rec_off},
              "provenance": {"context_meta": {
                  **context_meta(frozen),
                  "ctx_scalars": context_scalar_leaves(frozen)}}}
    arrays = {"z_cand": z, "prefix": np.stack(prefix), "box_poses": boxes,
              **context_arrays(frozen), **arr_on}
    jp, npz = write_products(tmp_path, "audit_x", record, arrays)
    assert jp.exists() and npz.exists()
    with pytest.raises(FileExistsError):
        write_products(tmp_path, "audit_x", record, arrays)

    loaded, larr = load_products(tmp_path, "audit_x")
    ctx2 = context_from(larr, loaded["provenance"]["context_meta"])
    re_on, _ = full_candidate_audit(pipe.spec, REGEN, vcfg, ctx2,
                                    larr["z_cand"],
                                    prefix=[larr["prefix"][0]],
                                    box_poses=larr["box_poses"], label="c2_on")
    assert re_on["endpoint"]["viol_sum"] == \
        loaded["audits"]["c2_on"]["endpoint"]["viol_sum"]
    assert re_on["task"]["mpjpe_m"] == loaded["audits"]["c2_on"]["task"]["mpjpe_m"]
    assert re_on["branch_pass"] == loaded["audits"]["c2_on"]["branch_pass"]
