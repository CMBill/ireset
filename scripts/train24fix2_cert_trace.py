"""E-stage forensic freeze for the certification-edge minimal-fix round
(train24fix2 plan section 3; runs BEFORE any algorithm change).

Read-only trace over two frozen trajectories per demo:

- historical Nq = ``reviewfix_20260923_nq_strict_task040`` (the run the D-Q
  historical lower bounds come from; byte-identical to the D-Q source
  ``labelq_20260926_sixseg/rq`` -- hash-checked here),
- current authoritative rq = ``train24fix_20260927_sixseg/rq`` (dbe587a).

For every side it recomputes, with the STRICT validator's own semantics and
UNROUNDED float64 values: hand geom distance, the 3-site box-frame
differential series (per-site norms, per-axis max component), the speed-gated
series, hysteresis intervals, certified / reference / intersection masks, the
per-frame lost/gain classification of the four failing sides, the previous
FULL-SCENE hand distance vs the existing projected ``prev_hand_dist``, the
reference-interval identity, and the proposed C2 enable/release state
(section 4.2 table).  It also freezes the H/B/T reference indexes of section 5
(unrounded) and cross-checks the counts against the frozen D-Q baseline JSON.

Counterfactuals (section 3 item 4): for the ten known speed-spike frames
(current rq, adopted states) and every historical certified-in-reference
candidate of the four failing sides, it replays the CURRENT rq accepted chain
(exact post-acceptance bookkeeping: contact commits, prev_local/prev_sole,
pre-frame foot release) up to frame t, then runs the CURRENT full acceptance
check (``frame_valid``: fresh uncapped pair scan + fixed pre/post rows +
adaptive interpolated probes + V1 strict joint speed + task gate) on the
candidate, and adds the PROPOSED C2 rows on top with the proposed enable
criterion -- never the dormant 6mm projected gate.  Verdicts: PASS /
C2_ONLY / CURRENT_ONLY / BOTH / V1_UNUSABLE.  A frozen candidate failing the
counterfactual does NOT prove the segment unreachable (section 3).

Internal assertions pin section 2.1 (interval lists, counts, lost/gain frame
sets) and section 2.4 (five counterexample distance triples, sub3:r projected
gate, certified-frame component-bound statistics) BITWISE where the plan
gives full precision; any mismatch aborts the freeze.

Outputs (new files only) -> derived/data/teachers_regenerated/
train24fix2_reviewfix2_20261002/diag/ (the frozen 20260927 diag and the
superseded reviewfix 20261001 diag are read-only):
  cert_trace_provenance.json      revision/configs/assets/inputs/H-B-T index
  cert_trace_{demo}_rq.json       per-frame series + C2 state + classification
  cert_trace_counterfactuals.json full-constraint counterfactual records
  cert_trace_cf_index.json        EXPECTED candidate list for the replay (D5)
  cert_trace_assertions.json      every assertion + the section 2 tables

Run: uv run --locked python scripts/train24fix2_cert_trace.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from closed_loop_metrics import git_rev, sha256
from experiment_provenance import asset_hashes, git_state
from ireset.data.labels import continuous_runs, longest_detachment_run
from ireset.data.teacher import load_teacher_npz
from ireset.embodied.contact_sites import (
    hysteresis_intervals,
    infer_reference_schedule,
)
from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR
from ireset.repair.config_io import load_mode_configs
from ireset.repair.constraints import FrameEvaluator
from ireset.repair.context import ContactStateMachine, ReferenceCondition
from ireset.repair.pipeline import RepairPipeline
from ireset.repair.validation import IndependentValidator
from regenerate_teachers import task_mpjpe_series
from train24fix2_audit import (
    AUDIT_FORMAT,
    arrays_diff,
    attach_integrity,
    classify_branches,
    configs_from_record,
    context_from,
    deep_diff,
    freeze_context,
    full_candidate_audit,
    load_index,
    load_products,
    verify_integrity,
    write_index,
    write_products,
)

REGEN = DERIVED_DATA_DIR / "teachers_regenerated"
HIST_NQ_DIR = REGEN / "reviewfix_20260923_nq_strict_task040"
HIST_DQ_SOURCE = REGEN / "labelq_20260926_sixseg" / "rq"      # D-Q sixseg_rq
#: the E re-completion (review section 7.B.3) re-verifies the SAME frozen
#: pre-C2 authoritative chain as the original E run: the train24fix-round
#: sixseg (rev dbe587a) -- the with-C2 train24fix2 sixseg is a DIFFERENT
#: chain and belongs to the fail-binding tool, not to the E candidates
CUR_RQ_DIR = REGEN / "train24fix_20260927_sixseg" / "rq"
CUR_RV_DIR = REGEN / "train24fix_20260927_sixseg" / "rv"
CUR_NS_DIR = REGEN / "train24fix_20260927_sixseg" / "ns"
TRAIN24_KEEP_DIR = REGEN / "labelq_20260926_train24" / "regen_final"
DQ_JSON = REGEN / "train24fix_20260927_diag" / "dq_quality_baseline.json"
#: review-fix round (2026-10-01): the frozen 20260927 diag directory is
#: READ-ONLY; every product of this run goes to a fresh reviewfix directory
#: and refuses to overwrite anything that already exists there.
#: Reviewfix2 (2026-10-02): the 20261001 diag products carry the D3
#: misaligned prev-scene field and the D2/D4/D5 weak replay contract -- they
#: are SUPERSEDED by this round's directory, never overwritten
OUT_DIR = REGEN / "train24fix2_reviewfix2_20261002" / "diag"
INDEX_NAME = "cert_trace_cf_index"
PLAN_DOC = Path(__file__).resolve().parents[1] / "docs" / "plans" / \
    "train24fix2_certification_edge_plan.md"

DEMOS = ["sub10_largebox_000", "sub12_largebox_001", "sub16_largebox_000",
         "sub3_largebox_000", "sub7_largebox_000", "sub8_largebox_000"]
NO_HIST = {"sub7_largebox_000"}            # no kept historical NPZ (D-Q sixseg has 5)

V_ROW = 0.02598          # proposed regen_hand_speed_mps (section 4.1)
C2_GAP = 0.008           # enable gap on the previous full-scene distance
ACCEPT_VIOL = 1e-6       # acceptance violation-sum gate (normalized, summed)
DISTMAX = 1.0            # mj_geomDistance scan cap (real-hit: dist < distmax)

#: section 2.1 expectation table (plan; asserted exactly)
EXPECT_21 = {
    ("sub10_largebox_000", "r"): {
        "old_intervals": [[41, 52], [57, 61]], "old_counts": [15, 11],
        "new_intervals": [[35, 38], [41, 44], [45, 52]], "new_counts": [13, 7],
        "gated": [44, 58], "min_dur": [57, 59, 60], "gain": [35, 36, 37]},
    ("sub3_largebox_000", "r"): {
        "old_intervals": [[22, 26], [35, 38], [68, 79]], "old_counts": [18, 11],
        "new_intervals": [[62, 66], [71, 76], [77, 81]], "new_counts": [13, 5],
        "gated": [23, 24, 37, 70, 76], "min_dur": [22, 25, 35, 36, 68, 69],
        "gain": [62, 63, 64, 65, 79, 80]},
    ("sub8_largebox_000", "l"): {          # section 2.1 elides the other intervals
        "old_has": [[62, 68]], "new_has": [[64, 68]],
        "old_counts": [29, 7], "new_counts": [27, 7],
        "gated": [63], "min_dur": [62], "gain": []},
    ("sub8_largebox_000", "r"): {
        "old_has": [[52, 61], [64, 67]], "new_has": [[52, 60]],
        "old_counts": [41, 13], "new_counts": [37, 13],
        "gated": [60, 66], "min_dur": [64, 65], "gain": []},
}
#: section 2.4 five counterexamples on the CURRENT rq (mm, mm, m/s)
EXPECT_24 = [
    ("sub10_largebox_000", "r", 44, 1.235161, 11.309322, 0.0689921),
    ("sub10_largebox_000", "r", 58, 0.359646, 12.236363, 0.0511776),
    ("sub8_largebox_000", "l", 63, -1.035932, 10.017719, 0.0574297),
    ("sub8_largebox_000", "r", 60, -1.153275, 13.253606, 0.2237822),
    ("sub8_largebox_000", "r", 66, -1.033070, 8.541370, 0.0526074),
]
#: section 2.4 component-bound statistics on the HISTORICAL Nq (over, total)
EXPECT_COMPONENT = {
    ("sub10_largebox_000", "r"): (10, 15), ("sub3_largebox_000", "r"): (11, 18),
    ("sub8_largebox_000", "l"): (18, 29), ("sub8_largebox_000", "r"): (27, 41),
}
EXPECT_T42 = (0.04709345, 0.04185564)     # sub10:r t=42 norm / max component
SUB3R_SPIKES = [23, 24, 37, 70, 76]       # within the old projected 8mm gate

FAIL_SIDES = {("sub10_largebox_000", "r"), ("sub3_largebox_000", "r"),
              ("sub8_largebox_000", "l"), ("sub8_largebox_000", "r")}


# ---------------------------------------------------------------------------
# per-trajectory series (strict validator semantics, unrounded)
# ---------------------------------------------------------------------------

def load_cond(demo: str) -> tuple[np.ndarray, ReferenceCondition]:
    """Input scene + reference condition, exactly as the runs built them."""
    src_scene, human, _fps = load_teacher_npz(
        TEACHER_TRAJ_DIR / f"{demo}_original.npz")
    cond = ReferenceCondition(
        human, src_scene[:, 36:43],
        infer_reference_schedule(human, src_scene[:, 36:43]))
    return src_scene, cond


def ref_intervals_of(cond: ReferenceCondition, eff: str) -> list[tuple[int, int]]:
    return [(iv.start, iv.end) for iv in cond.ref_schedule.intervals
            if iv.effector == eff]


def prev_scene_distances(traj: np.ndarray) -> dict:
    """Per-side min hand-geom/box distance of the PREVIOUS FULL SCENE
    (``traj[t-1]`` robot AND box pose) and of the EXISTING projection
    (previous robot state against the CURRENT box pose -- what
    ``pipeline._frame_ctx`` feeds ``prev_hand_dist``).

    Return contract (review R2, 2026-10-01 -- same as the pipeline's
    collection): finite ``< distmax`` is a real hit (negative/zero kept);
    finite ``== distmax`` is the normal truncation sentinel (a miss, never
    a distance); NaN/±Inf/out-of-range or a raised query is an ANOMALY and
    aborts this forensic verdict (the record survives, marked
    ``evidence_incomplete`` -- an anomalous query may never back a
    classification).
    """
    from ireset.repair.pipeline import (
        C2DistanceQueryError, classify_distance_query)

    rcfg, vcfg = load_mode_configs("strict")
    validator = IndependentValidator(cfg=vcfg)
    ev = FrameEvaluator(validator.spec, validator.pairs, rcfg)
    T = traj.shape[0]
    out = {s: {"scene": [None] * T, "proj": [None] * T, "miss": []}
           for s in ("l", "r")}
    for t in range(1, T):
        for kind, box in (("scene", traj[t - 1, 36:43]),
                          ("proj", traj[t, 36:43])):
            ev.forward(traj[t - 1, 0:36], box)
            for side in ("l", "r"):
                dmin, hit = np.inf, False
                for g in ev._hand_geoms[side]:
                    try:
                        d = ev.geom_distance(g, ev.box_geom, distmax=DISTMAX)[0]
                    except Exception as exc:                 # noqa: BLE001
                        raise C2DistanceQueryError(
                            t, side, "forensic_prev_scene_query", DISTMAX,
                            "exception", repr(exc),
                            stage="forensic_prev_scene_series") from exc
                    k = classify_distance_query(d, DISTMAX)
                    if k == "hit":                     # real hit
                        hit = True
                        dmin = min(dmin, d)
                    elif k == "truncated":
                        pass
                    else:
                        raise C2DistanceQueryError(
                            t, side, "forensic_prev_scene_query", DISTMAX,
                            k, repr(d), stage="forensic_prev_scene_series")
                if hit:
                    out[side][kind][t] = float(dmin)
                else:
                    out[side]["miss"].append({"t": t, "kind": kind})
    return out


def analyze_run(npz: Path, cond: ReferenceCondition, vcfg) -> dict:
    """Full per-side series of one trajectory (validator-cache semantics)."""
    traj, _human, fps = load_teacher_npz(npz)
    validator = IndependentValidator(cfg=vcfg)
    rep = validator.validate(traj, exclude=("box_ground",))
    dt = 1.0 / vcfg.fps
    T = traj.shape[0]
    sides = {}
    for side, eff in (("l", "left_hand"), ("r", "right_hand")):
        local = validator._cache["hand_local"][side]          # [T, 3, 3]
        dist = validator._cache["hand_dist"][side]            # [T]
        disp = np.zeros((T, 3))                                # per-site norms
        h_speed = np.full(T, np.inf)                           # frame 0: inf
        max_comp = np.full(T, np.nan)
        if T > 1:
            d = np.diff(local, axis=0)                         # [T-1, 3, 3]
            disp[1:] = np.linalg.norm(d, axis=2)
            h_speed[1:] = disp[1:].max(1) / dt
            max_comp[1:] = np.abs(d).max(axis=(1, 2)) / dt
        gated_inf = h_speed >= vcfg.hand_entry_speed_mps       # speed-gated out
        gated = np.where(gated_inf, np.inf, dist)
        ivs = hysteresis_intervals(gated, vcfg.hand_enter_m, vcfg.hand_exit_m,
                                   vcfg.min_duration)
        ref_ivs = ref_intervals_of(cond, eff)
        ref = np.zeros(T, dtype=bool)
        for a, b in ref_ivs:
            ref[a:b] = True
        cert = np.zeros(T, dtype=bool)
        for a, b in ivs:
            cert[a:b] = True
        within = ref & (dist < vcfg.hand_enter_m)     # strict <, keep criterion
        sides[side] = {
            "hand_dist": [float(x) for x in dist],
            "disp_site_norm": [[float(v) for v in row] for row in disp],
            "h_speed": [None if not np.isfinite(v) else float(v)
                        for v in h_speed],
            "max_component": [None if not np.isfinite(v) else float(v)
                              for v in max_comp],
            "gated_out": [bool(v) for v in gated_inf],
            "certified": [bool(v) for v in cert],
            "reference": [bool(v) for v in ref],
            "supervision": [bool(v) for v in (cert & ref)],
            "cert_intervals": [[int(a), int(b)] for a, b in ivs],
            "ref_intervals": [[int(a), int(b)] for a, b in ref_ivs],
            "intervals_within_enter": int(within.sum()),
        }
    return {"T": int(T), "fps": float(fps),
            "validator_passed": bool(rep.passed), "units": "m / m/s",
            "sides": sides}


def intersection_runs_of(series_side: dict) -> list[tuple[int, int]]:
    sup = np.array(series_side["supervision"], dtype=bool)
    return continuous_runs(sup)


# ---------------------------------------------------------------------------
# lost/gain classification (section 2.1 categories)
# ---------------------------------------------------------------------------

def classify_side(old: dict, new: dict, vcfg) -> dict:
    """Per-frame classification of intersection changes old(Nq) -> new(rq).

    Lost frames (in old supervision, not in new), judged on the NEW series:
      speed >= entry gate -> GATED (the spike itself)
      dist <  enter       -> MIN_DUR (interval dropped by min_duration)
      otherwise           -> DIST_LOST (geometric; expected 0), with a
                             sub-attribution separating the three zero
                             categories of section 2: geometric entry/re-entry
                             denial vs detachment beyond the hysteresis exit.
    """
    T = len(old["supervision"])
    enter, exit_th, gate = (vcfg.hand_enter_m, vcfg.hand_exit_m,
                            vcfg.hand_entry_speed_mps)
    dist = np.array([np.inf if v is None else v for v in new["hand_dist"]])
    speed = np.array([np.inf if v is None else v for v in new["h_speed"]])
    sup_old = np.array(old["supervision"], dtype=bool)
    sup_new = np.array(new["supervision"], dtype=bool)
    lost = [t for t in range(T) if sup_old[t] and not sup_new[t]]
    gained = [t for t in range(T) if sup_new[t] and not sup_old[t]]
    cls = {}
    for t in lost:
        if speed[t] >= gate:
            cls[t] = "GATED"
        elif dist[t] < enter:
            cls[t] = "MIN_DUR"
        else:
            cls[t] = "DIST_LOST"
    sub = {}
    for t in (t for t in lost if cls[t] == "DIST_LOST"):
        if dist[t] > exit_th:
            sub[t] = "hysteresis_detach"
        elif any(a <= t < b for a, b in new["cert_intervals"]):
            sub[t] = "in_cert_but_not_ref"      # tripwire: ref masks are identical
        else:
            sub[t] = "entry_or_reentry_denied"
    inter_old = intersection_runs_of(old)
    inter_new = intersection_runs_of(new)
    return {
        "lost": {int(t): cls[t] for t in lost},
        "dist_lost_sub": {int(t): sub[t] for t in sub},
        "gained": [int(t) for t in gained],
        "old_intersection": [[int(a), int(b)] for a, b in inter_old],
        "new_intersection": [[int(a), int(b)] for a, b in inter_new],
        "old_counts": [int(sup_old.sum()),
                       max((b - a for a, b in inter_old), default=0)],
        "new_counts": [int(sup_new.sum()),
                       max((b - a for a, b in inter_new), default=0)],
        "lost_evidence": {int(t): {
            "dist_m": new["hand_dist"][t], "speed_mps": new["h_speed"][t],
            "max_component": new["max_component"][t]} for t in lost},
        "gain_evidence": {int(t): {
            "dist_m": new["hand_dist"][t], "speed_mps": new["h_speed"][t],
            "max_component": new["max_component"][t]} for t in gained},
    }


# ---------------------------------------------------------------------------
# proposed C2 enable state (section 4.2 table)
# ---------------------------------------------------------------------------

def c2_enable_trace(cond: ReferenceCondition, prev_scene: dict) -> dict:
    """Simulate the proposed enable/release state machine on ONE trajectory's
    previous-full-scene distances (per side).  Deterministic from the accepted
    chain and the INPUT reference schedule only (never the validator's own
    intervals)."""
    T = cond.T
    out = {}
    for side, eff in (("l", "left_hand"), ("r", "right_hand")):
        frame_iv = [None] * T
        iv_ids = {}
        for i, (a, b) in enumerate(ref_intervals_of(cond, eff)):
            iv_ids[f"{a},{b}"] = i
            for t in range(a, b):
                frame_iv[t] = i
        enabled, enabled_iv = False, None
        rows, reasons = [], []
        for t in range(T):
            iv = frame_iv[t]
            if t == 0 or prev_scene[side]["scene"][t] is None:
                # t=0 / no valid previous full scene: closed, no inheritance
                enabled, enabled_iv = False, None
                reasons.append("off_t0_or_no_prev")
            elif iv is None:
                # reference does not want hold on this side: close + release
                enabled, enabled_iv = False, None
                reasons.append("off_no_ref_released")
            else:
                if enabled and enabled_iv != iv:
                    # interval identity changed: clear, re-judge this frame
                    enabled, enabled_iv = False, None
                    reasons.append("cleared_new_interval")
                d = prev_scene[side]["scene"][t]
                if enabled:
                    reasons.append("hold")
                elif d <= C2_GAP:
                    enabled, enabled_iv = True, iv
                    reasons.append("enable")
                else:
                    reasons.append("pull_in_exempt")
            rows.append(bool(enabled))
        out[side] = {"enabled": rows, "reason": reasons,
                     "frame_interval_id": frame_iv, "interval_ids": iv_ids}
    return out


def enable_gap_frames(series: dict, c2: dict, vcfg) -> list[dict]:
    """Frames with dist < enter inside a reference interval where C2 never
    turned enabled (section 3 item 3 -- residual-risk listing)."""
    out = []
    for side in ("l", "r"):
        s, c = series["sides"][side], c2[side]
        ever = {iv for iv, en in zip(c["frame_interval_id"], c["enabled"])
                if en}
        for t in range(series["T"]):
            iv = c["frame_interval_id"][t]
            if iv is None or iv in ever:
                continue
            d = s["hand_dist"][t]
            if d is not None and d < vcfg.hand_enter_m:
                out.append({"side": side, "t": t, "dist_m": d,
                            "speed_mps": s["h_speed"][t],
                            "max_component": s["max_component"][t],
                            "ref_interval_id": iv})
    return out


# ---------------------------------------------------------------------------
# section-5 H/B/T freeze (unrounded) + D-Q cross-check
# ---------------------------------------------------------------------------

def freeze_segment(npz: Path, demo: str, cond: ReferenceCondition,
                   vcfg, rcfg) -> dict:
    """Unrounded side metrics of the section-5 core/complete quality gates."""
    traj, _human, _fps = load_teacher_npz(npz)
    validator = IndependentValidator(cfg=vcfg)
    rep = validator.validate(traj, exclude=("box_ground",))
    ev = FrameEvaluator(validator.spec, validator.pairs, rcfg)
    task = task_mpjpe_series(ev, traj, cond)
    T = traj.shape[0]
    sides = {}
    for side, eff in (("l", "left_hand"), ("r", "right_hand")):
        local = validator._cache["hand_local"][side]
        dist = validator._cache["hand_dist"][side]
        dt = 1.0 / vcfg.fps
        speed = np.full(T, np.nan)
        if T > 1:
            speed[1:] = np.linalg.norm(
                np.diff(local, axis=0), axis=2).max(1) / dt
        ref_ivs = ref_intervals_of(cond, eff)
        ref = np.zeros(T, dtype=bool)
        for a, b in ref_ivs:
            ref[a:b] = True
        ivs = [(iv.start, iv.end) for iv in rep.schedule.intervals
               if iv.effector == eff and iv.support == "object"]
        cert = np.zeros(T, dtype=bool)
        drift = []
        for a, b in ivs:
            cert[a:b] = True
            drift.append(float(np.linalg.norm(
                local[a:b] - local[a], axis=-1).max()))
        sup = cert & ref
        inter = continuous_runs(sup)
        within = ref & (dist < vcfg.hand_enter_m)
        gap_ref = dist[ref] if ref.any() else np.array([])
        fin = speed[ref][np.isfinite(speed[ref])] if ref.any() else np.array([])
        sides[side] = {
            "ref_frames": int(ref.sum()),
            "ref_intervals": [[int(a), int(b)] for a, b in ref_ivs],
            "certified_frames": int(cert.sum()),
            "certified_in_ref_frames": int(sup.sum()),
            "intersection_intervals": [[int(a), int(b)] for a, b in inter],
            "intersection_longest_run_frames": max(
                (b - a for a, b in inter), default=0),
            "certified_intervals": [[int(a), int(b)] for a, b in ivs],
            "frames_within_enter": int(within.sum()),
            "longest_detachment_run_frames": longest_detachment_run(
                within, ref_ivs),
            "max_detachment_ref_m": (float(gap_ref.max())
                                     if gap_ref.size else None),
            "gap_ref_min_m": (float(gap_ref.min()) if gap_ref.size else None),
            "speed_ref_max_mps": (float(fin.max()) if fin.size else None),
            "speed_ref_p95_mps": (float(np.percentile(fin, 95))
                                  if fin.size else None),
            "anchor_drift_certified_max_m": (float(max(drift))
                                             if drift else None),
        }
    return {"demo": demo, "npz": str(npz.resolve()), "npz_sha256": sha256(npz),
            "input_sha256": sha256(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
            "T": int(T), "validator_passed": bool(rep.passed),
            "task_mpjpe": {"p50": float(np.percentile(task, 50)),
                           "p95": float(np.percentile(task, 95)),
                           "max": float(task.max())},
            "sides": sides}


# ---------------------------------------------------------------------------
# counterfactuals: current-chain replay + full acceptance check + C2 rows
# ---------------------------------------------------------------------------

class ChainReplay:
    """Replays the CURRENT rq accepted chain's per-frame bookkeeping exactly
    as ``_run_sequential`` does (the frame's OWN pre-frame foot release from
    the prediction, then post-acceptance establishment candidates + commits,
    prev_local/prev_sole), so a ``_frame_ctx`` built at frame t matches the
    run's own context.  ``ctx_at(t)`` requires monotonically non-decreasing
    t; the pre-frame release of frame t runs BEFORE the ctx build (the run
    does the same), and is idempotent across repeated ``ctx_at(t)`` calls
    (a released anchor cannot release twice)."""

    def __init__(self, cond: ReferenceCondition, src_scene: np.ndarray,
                 rq_traj: np.ndarray, rcfg, vcfg):
        # regen_hand_speed_mps is pinned to 0.0 HERE on purpose: the frozen
        # rq chain and this counterfactual's "current check" are the PRE-C2
        # algorithm; the proposed rows are added on top by ``c2_rows_of``
        # with the trace's own enable criterion (never the deployed code).
        self.cfg = replace(rcfg, regen_mode=True, regen_hand_speed_mps=0.0)
        assert self.cfg.regen_hand_speed_mps == 0.0            # noqa: S101
        self.pipe = RepairPipeline(cfg=self.cfg)
        self.pipe.validator.cfg = vcfg
        self.vcfg = vcfg
        self.cond = cond
        self.pred = src_scene
        self.traj = rq_traj
        self.contacts = ContactStateMachine(cond.ref_schedule, rq_traj.shape[0])
        self.prev_local: dict = {}
        self.prev_sole: dict = {}
        self.t_next = 0            # first frame not yet replayed

    def _pre_frame_release(self, t: int) -> None:
        """The frame's own pre-frame anchor release (prediction-driven),
        exactly the block ``_run_sequential`` runs before ``_frame_ctx``."""
        if not self.prev_sole:
            return
        self.pipe.ev.forward(self.pred[t, 0:36], self.pred[t, 36:43])
        for side, name in (("l", "left_foot"), ("r", "right_foot")):
            if self.contacts.holding[name] and \
                    self.prev_sole.get(side) is not None:
                pts = self.pipe.ev.site_positions(
                    self.pipe.ev.sole_sites[side])
                speed = float(np.linalg.norm(
                    pts - self.prev_sole[side], axis=-1).max()) * self.cfg.fps
                if speed >= self.cfg.foot_anchor_establish_speed_mps:
                    self.contacts.release(name)

    def advance_to(self, t: int) -> None:
        while self.t_next < t:
            u = self.t_next
            self._pre_frame_release(u)
            z_u = self.traj[u, 0:36].copy()
            box_u = self.traj[u, 36:43]
            cands = self.pipe._establishment_candidates(
                z_u, box_u, self.contacts, u, self.prev_sole)
            self.contacts.commit(
                "left_hand", u, hand_local=cands["hand"].get("left_hand"))
            self.contacts.commit(
                "right_hand", u, hand_local=cands["hand"].get("right_hand"))
            self.contacts.commit(
                "left_foot", u, sole_world=cands["foot"].get("left_foot"))
            self.contacts.commit(
                "right_foot", u, sole_world=cands["foot"].get("right_foot"))
            self.pipe.ev.forward(z_u, box_u)
            self.prev_local = {"l": self.pipe.ev.hand_local("l"),
                               "r": self.pipe.ev.hand_local("r")}
            self.prev_sole = {
                "l": self.pipe.ev.site_positions(self.pipe.ev.sole_sites["l"]),
                "r": self.pipe.ev.site_positions(self.pipe.ev.sole_sites["r"])}
            self.t_next = u + 1

    def ctx_at(self, t: int):
        assert t >= self.t_next                              # noqa: S101
        self.advance_to(t)
        # frame t's OWN pre-frame release runs before its ctx (the run does
        # the same); idempotent across repeated ctx_at(t) calls
        self._pre_frame_release(t)
        z_prev = self.traj[t - 1, 0:36].copy() if t > 0 else None
        return self.pipe._frame_ctx(t, self.pred, self.cond, self.contacts,
                                    self.prev_local, self.prev_sole, z_prev)


def c2_rows_of(pipe: RepairPipeline, ctx, z: np.ndarray, c2_state: dict,
               t: int) -> dict:
    """Proposed C2 row values/bounds at a candidate state (both sides of the
    frame's enable state); the acceptance criterion is the same normalized
    violation SUM <= 1e-6 the pipeline applies to post rows."""
    lim_h = (1.0 / pipe.cfg.fps) * V_ROW
    pipe.ev.forward(z, ctx.box_pose)
    out = {}
    for side in ("l", "r"):
        if not c2_state[side]["enabled"][t]:
            out[side] = {"enabled": False,
                         "reason": c2_state[side]["reason"][t]}
            continue
        prev = ctx.prev_hand_local.get(side)
        vals = (pipe.ev.hand_local(side) - prev).reshape(-1)
        per_row = np.maximum(np.abs(vals) - lim_h, 0.0) / lim_h
        out[side] = {
            "enabled": True, "reason": c2_state[side]["reason"][t],
            "row_values_absmax_m": float(np.abs(vals).max()),
            "bound_abs_m": float(lim_h),
            "norm_speed_mps": float(np.linalg.norm(
                vals.reshape(3, 3), axis=1).max() * pipe.cfg.fps),
            "max_component_mps": float(np.abs(vals.reshape(3, 3)).max()
                                       * pipe.cfg.fps),
            "viol_sum": float(per_row.sum()),
            "n_viol_rows": int((per_row > 0).sum()),
        }
    return out


def prev_scene_flags(prev_scene: dict | None, t: int) -> dict:
    """D3 (reviewfix2): per-side previous-FULL-scene distance at frame t.
    The series' ``[t]`` entry ALREADY IS the previous scene (``traj[t-1]``
    robot AND box -- see :func:`prev_scene_distances`); reading ``[t-1]``
    shifts one more frame, and one side's value must never be written into
    both sides.  ``None`` series (no rq source) keeps both sides None."""
    if prev_scene is None:
        return {"l": None, "r": None}
    return {s: prev_scene[s]["scene"][t] for s in ("l", "r")}


def prev_scene_distances_at(traj: np.ndarray, t: int) -> dict:
    """Both sides' previous-FULL-scene hand/box distance for ONE frame,
    same query contract as :func:`prev_scene_distances` (anomaly raises
    ``C2DistanceQueryError``) -- the replay cross-check that the frozen
    per-record fields really are ``[side]["scene"][t]``."""
    from ireset.repair.pipeline import (
        C2DistanceQueryError, classify_distance_query)

    rcfg, vcfg = load_mode_configs("strict")
    validator = IndependentValidator(cfg=vcfg)
    ev = FrameEvaluator(validator.spec, validator.pairs, rcfg)
    ev.forward(traj[t - 1, 0:36], traj[t - 1, 36:43])
    out = {}
    for side in ("l", "r"):
        dmin, hit = np.inf, False
        for g in ev._hand_geoms[side]:
            try:
                d = ev.geom_distance(g, ev.box_geom, distmax=DISTMAX)[0]
            except Exception as exc:                     # noqa: BLE001
                raise C2DistanceQueryError(
                    t, side, "forensic_prev_scene_query", DISTMAX,
                    "exception", repr(exc),
                    stage="replay_prev_scene_at") from exc
            k = classify_distance_query(d, DISTMAX)
            if k == "hit":
                hit = True
                dmin = min(dmin, d)
            elif k == "truncated":
                pass
            else:
                raise C2DistanceQueryError(
                    t, side, "forensic_prev_scene_query", DISTMAX, k,
                    repr(d), stage="replay_prev_scene_at")
        out[side] = float(dmin) if hit else None
    return out


def counterfactual(replay: ChainReplay, t: int, z_cand: np.ndarray,
                   source: str, c2_state: dict, side: str,
                   prev_scene: dict | None, prev_source: dict | None,
                   rq_traj: np.ndarray, src_scene: np.ndarray,
                   spec, rcfg, vcfg) -> tuple[dict, dict]:
    """FULL two-branch counterfactual (review R3): the same candidate on
    the frozen current chain, audited with every applicable check --
    endpoint rows (fresh uncapped scan), every interpolation point, strict
    joint velocities, task under the HUMAN reference, and the accepted
    PREFIX independent final validation -- once WITHOUT C2 (the historical
    explicit ``regen_hand_speed_mps=0`` configuration, never the deployed
    default) and once with the REAL deployed C2 assembly (same previous
    sites / enable state; no parallel speed formula decides acceptance).

    Returns ``(record, arrays)``; the classification follows the section
    4.2 table (EVIDENCE_INCOMPLETE / V1_UNUSABLE / PASS / C2_ONLY /
    CURRENT_ONLY / BOTH) and the provenance carries the candidate's true
    origin (an adopted rq state or a PORTED historical Nq frame -- the
    ported origin never fabricates an attempt/check id).
    """
    from train24fix2_audit import (
        classify_branches, context_arrays, freeze_context, full_candidate_audit)

    ctx = replay.ctx_at(t)
    ctx_frozen = freeze_context(ctx)
    cfg_current = replace(rcfg, regen_mode=True, regen_hand_speed_mps=0.0)
    assert cfg_current.regen_hand_speed_mps == 0.0            # noqa: S101
    cfg_plus = replace(rcfg, regen_mode=True)                 # deployed 0.02598
    assert cfg_plus.regen_hand_speed_mps == 0.02598           # noqa: S101

    ctx_plus = replace(
        freeze_context(ctx),
        c2_hand_enabled={"l": bool(c2_state["l"]["enabled"][t]),
                         "r": bool(c2_state["r"]["enabled"][t])},
        c2_hand_reason={"l": c2_state["l"]["reason"][t],
                        "r": c2_state["r"]["reason"][t]},
        prev_hand_scene_dist=prev_scene_flags(prev_scene, t))

    prefix = [rq_traj[i, 0:36].copy() for i in range(t)]
    box_poses = src_scene[:, 36:43]
    side_flags = prev_scene_flags(prev_scene, t)
    rec = {
        "frame": int(t), "candidate_source": source, "side": side,
        "provenance": {
            "pass": 0,
            "origin": ("current_rq_adopted_state" if "current_rq" in source
                       else "ported_historical_nq_frame"),
            "note": ("ported frames carry no attempt/check identity of this "
                     "run; the source NPZ hash + frame index are the identity"),
            "c2_enable": {s: {"enabled": bool(c2_state[s]["enabled"][t]),
                              "reason": c2_state[s]["reason"][t]}
                          for s in ("l", "r")},
        },
        "box_pose_t": [float(v) for v in ctx.box_pose],
        "box_pose_prev": (None if ctx.box_pose_prev is None
                          else [float(v) for v in ctx.box_pose_prev]),
        # per-side previous-FULL-scene distance AT t (reviewfix2 D3: both
        # sides recorded separately, index [t] -- the series' [t] already
        # IS the previous scene; the selected side's value is kept under
        # prev_scene_dist_m for continuity with the first review round)
        "prev_scene_dist_both_m": dict(side_flags),
        "prev_scene_dist_m": side_flags[side],
        "prev_projected_dist_m": (None if prev_scene is None
                                  else prev_scene[side]["proj"][t]),
        "prev_scene_source": prev_source,
        # component/norm speed evidence (diagnostic only -- acceptance is
        # the deployed row assembly inside the audits below)
        "c2_speed_evidence": c2_rows_of(replay.pipe, ctx,
                                        np.asarray(z_cand, dtype=float),
                                        c2_state, t),
    }
    arrays: dict = {"z_cand": np.asarray(z_cand, dtype=float).copy(),
                    "prefix": (np.zeros((0, 36)) if not prefix
                               else np.stack(prefix)),
                    "box_poses": box_poses.copy(),
                    **context_arrays(ctx_frozen)}
    audits: dict = {}
    for label, actx, acfg in (("current", ctx_frozen, cfg_current),
                              ("plus_c2", ctx_plus, cfg_plus)):
        arec, aarr = full_candidate_audit(
            spec, acfg, vcfg, actx, z_cand, prefix=prefix,
            box_poses=box_poses, label=label)
        audits[label] = arec
        arrays.update(aarr)
    rec["audits"] = audits
    rec["classification"] = classify_branches(
        audits["current"], audits["plus_c2"])
    rec["classification_basis"] = {
        "current_branch_pass": audits["current"].get("branch_pass"),
        "plus_c2_branch_pass": audits["plus_c2"].get("branch_pass")}
    # replay inputs (section 4.3): everything needed to restore the context
    # and the C2 flags without touching the chain replay
    from train24fix2_audit import context_meta, context_scalar_leaves
    rec["replay_meta"] = {**context_meta(ctx_frozen),
                          "ctx_scalars": context_scalar_leaves(ctx_frozen)}
    rec["c2_enable_flags"] = dict(ctx_plus.c2_hand_enabled)
    rec["c2_reasons"] = dict(ctx_plus.c2_hand_reason)
    rec["prev_scene_dist_flags"] = dict(ctx_plus.prev_hand_scene_dist)
    # reviewfix2 D4: frozen effective configs of BOTH branches + the
    # integrity manifest -- the replay restores these, never the YAML
    attach_integrity(rec, cfg_plus, vcfg)
    rec["effective"]["repair_no_c2"] = asdict(cfg_current)
    return rec, arrays


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------

class Assertions:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail) -> None:
        self.rows.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: "
              f"{json.dumps(detail, default=str)[:220]}")
        if not ok:
            raise AssertionError(f"E-stage assertion failed: {name}: {detail}")

    def note(self, name: str, detail) -> None:
        self.rows.append({"name": name, "ok": None, "detail": detail})


def _interval_match(cls: dict, exp: dict) -> bool:
    old = [list(x) for x in cls["old_intersection"]]
    new = [list(x) for x in cls["new_intersection"]]
    if "old_intervals" in exp:
        return (old == [list(x) for x in exp["old_intervals"]]
                and new == [list(x) for x in exp["new_intervals"]])
    return (all(list(x) in old for x in exp["old_has"])
            and all(list(x) in new for x in exp["new_has"]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def replay_counterfactuals(out_dir: Path, replay_dir: Path) -> None:
    """Strict replay (reviewfix2 D2/D3/D4/D5): candidates come ONLY from the
    generation-time index (a replay summary can never re-enter as one, and
    an empty/missing index is an explicit failure); each product is loaded
    with its NPZ sha256 verified; the audits run under the FROZEN effective
    configs restored from the record (never the current YAML); the previous
    full-scene distance fields are CROSS-CHECKED against a fresh query of
    the hash-verified source NPZ (the D3 class -- per-side, index [t]);
    and the comparison covers the COMPLETE records (all rows, every
    interpolation point, strict velocities, task, prefix checks, c2 speed
    evidence, classification) plus the arrays BITWISE.  The summary goes to
    ``replay_dir`` -- never the products directory."""
    idx = load_index(out_dir, INDEX_NAME)
    replay_dir.mkdir(parents=True, exist_ok=True)
    by_stem = {r["stem"]: r for r in idx["records"]}
    results: dict = {}
    ok = True
    for stem in by_stem:
        record, arrays = load_products(out_dir, stem)
        entry: dict = {"stem": stem}
        results[stem] = entry
        integrity = verify_integrity(record)
        entry["integrity_ok"] = not integrity
        entry["integrity_mismatches"] = integrity[:20]
        t = record["frame"]
        cfg_plus, vcfg = configs_from_record(record)
        cfg_current, _ = configs_from_record(record, key="repair_no_c2")
        assert cfg_current.regen_hand_speed_mps == 0.0    # noqa: S101
        assert cfg_plus.regen_hand_speed_mps == 0.02598   # noqa: S101
        spec = RepairPipeline(cfg=cfg_plus).spec
        z_cand = arrays["z_cand"]
        prefix = ([arrays["prefix"][i].copy()
                   for i in range(arrays["prefix"].shape[0])]
                  if arrays["prefix"].size else [])
        box_poses = arrays["box_poses"]
        meta = record["replay_meta"]
        ctx = context_from(arrays, meta)
        ctx_plus = replace(
            freeze_context(ctx),
            c2_hand_enabled=record["c2_enable_flags"],
            c2_hand_reason=record["c2_reasons"],
            prev_hand_scene_dist=record["prev_scene_dist_flags"])

        # D3 cross-check: the frozen per-side fields must equal a FRESH
        # distance query of the previous full scene of the source NPZ
        prev_checks: dict = {}
        src = record.get("prev_scene_source")
        if src is not None and t > 0:
            npz_path = Path(src["npz"])
            digest = sha256(npz_path)
            if digest != src["sha256"]:
                prev_checks["source_npz"] = (
                    f"sha256 {digest} != recorded {src['sha256']}")
            else:
                traj, _h, _f = load_teacher_npz(npz_path)
                fresh_at = prev_scene_distances_at(traj, t)
                for s in ("l", "r"):
                    if record["prev_scene_dist_both_m"][s] != fresh_at[s]:
                        prev_checks[s] = (f"saved "
                                          f"{record['prev_scene_dist_both_m'][s]!r}"
                                          f" != fresh {fresh_at[s]!r}")
        entry["prev_scene_check"] = prev_checks or "ok"

        audits: dict = {}
        fresh_arrays: dict = {}
        for label, actx, acfg in (("current", ctx, cfg_current),
                                  ("plus_c2", ctx_plus, cfg_plus)):
            arec, aarr = full_candidate_audit(
                spec, acfg, vcfg, actx, z_cand, prefix=prefix,
                box_poses=box_poses, label=label)
            audits[label] = arec
            fresh_arrays.update(aarr)
        # diagnostic speed evidence recomputed from the frozen flags (the
        # same c2_rows_of the generation ran, state rebuilt at frame t)
        c2_state_r = {s: {"enabled": [False] * t
                          + [record["c2_enable_flags"][s]],
                          "reason": [""] * t + [record["c2_reasons"][s]]}
                      for s in ("l", "r")}
        ev_pipe = RepairPipeline(spec=spec, cfg=cfg_plus)
        speed_evidence = c2_rows_of(ev_pipe, ctx, z_cand, c2_state_r, t)
        classification = classify_branches(audits["current"], audits["plus_c2"])
        fresh_record = {
            "audits": audits,
            "classification": classification,
            "classification_basis": {
                "current_branch_pass": audits["current"].get("branch_pass"),
                "plus_c2_branch_pass": audits["plus_c2"].get("branch_pass")},
            "c2_speed_evidence": speed_evidence,
        }
        diffs = deep_diff({k: record.get(k) for k in fresh_record},
                          fresh_record)
        recomp = {k: arrays[k] for k in fresh_arrays if k in arrays}
        arr_diffs = arrays_diff(recomp, fresh_arrays)
        arr_diffs += [f"{k}: recomputed array absent from saved bundle"
                      for k in sorted(fresh_arrays) if k not in arrays]
        entry.update({
            "record_diff_paths": diffs[:40],
            "n_record_diffs": len(diffs),
            "array_diffs": arr_diffs[:40],
            "classification_recomputed": classification,
            "classification_matches": classification
            == record.get("classification"),
            "matches": not diffs and not arr_diffs,
        })
        entry["ok"] = (entry["matches"] and entry["classification_matches"]
                       and entry["integrity_ok"]
                       and prev_checks == {})
        ok = ok and entry["ok"]
        print(f"  [replay {stem}] match={entry['matches']} "
              f"class={classification} prev={entry['prev_scene_check']}")
    (replay_dir / "cert_trace_cf_replay.json").write_text(
        json.dumps({"ok": ok, "n": len(results),
                    "expected": len(idx["records"]),
                    "results": results}, indent=1, allow_nan=False,
                   default=float))
    print(f"[replay] {'ALL MATCH' if ok else 'MISMATCH'} "
          f"({len(results)} records) -> "
          f"{replay_dir / 'cert_trace_cf_replay.json'}")


def _write_guarded(path: Path, text: str) -> None:
    """Forensic outputs never overwrite (review section 4.3): the frozen
    20260927 diag directory is read-only and this run's own products are
    write-once too -- re-running means a fresh --out."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite forensic file {path}")
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-counterfactuals", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR),
                    help="output dir (default: the reviewfix2 20261002 diag "
                         "dir; the frozen 20260927 and superseded 20261001 "
                         "ones are read-only)")
    ap.add_argument("--replay", action="store_true",
                    help="recompute the same-candidate counterfactual "
                         "branches from the frozen files and compare -- "
                         "no chain replay, no re-solve")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.replay:
        replay_counterfactuals(out_dir,
                               out_dir.parent / f"{out_dir.name}_replay")
        return

    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    A = Assertions()
    rcfg, vcfg = load_mode_configs("strict")

    # ---- provenance freeze (section 3 item 1) -----------------------------
    print("== provenance freeze ==")
    prov = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "train24fix2 E-stage forensic freeze (plan section 3)",
        "this_run": {"git": git_state(), "git_rev": git_rev(),
                     "plan_sha256": sha256(PLAN_DOC)},
        "assets": asset_hashes(),
        "effective_configs": {"strict_repair": asdict(rcfg),
                              "strict_validation": asdict(vcfg)},
        "proposed_c2": {"regen_hand_speed_mps": V_ROW, "enable_gap_m": C2_GAP,
                        "accept_viol_sum": ACCEPT_VIOL,
                        "prev_distance_source": "previous accepted FULL scene"},
        "inputs": [{"demo": d,
                    "sha256": sha256(TEACHER_TRAJ_DIR / f"{d}_original.npz")}
                   for d in DEMOS],
        "dq_json_sha256": sha256(DQ_JSON),
    }
    runs = {}
    for tag, d in (("rq", CUR_RQ_DIR), ("rv", CUR_RV_DIR), ("ns", CUR_NS_DIR)):
        rep_file = next(d.glob("*_report_diagnosis6_strict.json"))
        rep = json.loads(rep_file.read_text())
        runs[tag] = {"report": str(rep_file), "git_rev": rep["git_rev"],
                     "n_kept": rep["n_kept"],
                     "code_changed_during_run":
                         rep["end"]["code_changed_during_run"],
                     "config_unchanged": rep["end"]["config_unchanged"],
                     "rows_accepted": {r["demo"]: r["accepted"]
                                       for r in rep["rows"]}}
    hist_rep = json.loads(
        (HIST_NQ_DIR / "regen_report_diagnosis6_strict.json").read_text())
    dq_source_rep = json.loads(
        (HIST_DQ_SOURCE / "regen_report_diagnosis6_strict.json").read_text())
    runs["nq_hist"] = {"git_rev": hist_rep["git_rev"],
                       "n_kept": hist_rep["n_kept"],
                       "code_changed_during_run":
                           hist_rep["end"]["code_changed_during_run"]}
    runs["dq_sixseg_source"] = {"git_rev": dq_source_rep["git_rev"],
                                "n_kept": dq_source_rep["n_kept"]}
    prov["verified_runs"] = runs
    A.check("authoritative_rev_dbe587a",
            all(runs[k]["git_rev"] == "dbe587a" for k in ("rq", "rv", "ns")),
            {k: runs[k]["git_rev"] for k in ("rq", "rv", "ns")})
    A.check("authoritative_no_drift",
            all(runs[k]["code_changed_during_run"] is False
                and runs[k]["config_unchanged"] for k in ("rq", "rv", "ns")),
            "rq/rv/ns end-state == freeze-state")
    A.check("authoritative_all_accepted",
            all(all(v for v in runs[k]["rows_accepted"].values())
                for k in ("rq", "rv", "ns")),
            {k: runs[k]["rows_accepted"] for k in ("rq", "rv", "ns")})
    A.check("historical_nq_no_drift",
            runs["nq_hist"]["code_changed_during_run"] is False,
            runs["nq_hist"])

    # H/B/T reference indexes (section 5) + hash checks
    dq = json.loads(DQ_JSON.read_text())
    dq_by_demo = {s["demo"]: s for s in dq["segments"]}
    print("== freezing H (historical sixseg) ==")
    H = []
    conds: dict[str, tuple] = {}
    for demo in [d for d in DEMOS if d not in NO_HIST]:
        A.check(f"nq_hash_matches_dq_source[{demo}]",
                sha256(HIST_NQ_DIR / f"{demo}.npz")
                == sha256(HIST_DQ_SOURCE / f"{demo}.npz"),
                sha256(HIST_NQ_DIR / f"{demo}.npz"))
        src, cond = load_cond(demo)
        conds[demo] = (src, cond)
        H.append(freeze_segment(HIST_NQ_DIR / f"{demo}.npz", demo, cond,
                                vcfg, rcfg))
    print("== freezing B (current authoritative sixseg) ==")
    B = []
    for demo in DEMOS:
        if demo not in conds:
            conds[demo] = load_cond(demo)
        src, cond = conds[demo]
        B.append(freeze_segment(CUR_RQ_DIR / f"{demo}.npz", demo, cond,
                                vcfg, rcfg))
    A.check("B_all_validator_passed",
            all(b["validator_passed"] for b in B),
            {b["demo"]: b["validator_passed"] for b in B})
    print("== freezing T (train24 keep 13) ==")
    T = []
    for npz in sorted(TRAIN24_KEEP_DIR.glob("*.npz")):
        demo = npz.stem
        if demo not in conds:
            conds[demo] = load_cond(demo)
        _src, cond = conds[demo]
        T.append(freeze_segment(npz, demo, cond, vcfg, rcfg))
    A.check("T_matches_dq_demo_set",
            {t["demo"] for t in T}
            == {d for d, s in dq_by_demo.items()
                if s["source"] == "train24_keep"},
            sorted({t["demo"] for t in T}))

    # D-Q cross-check: counts must reproduce the frozen baseline (H + T)
    for seg in H + T:
        ref_row = dq_by_demo.get(seg["demo"])
        if ref_row is None:
            continue
        side_ok = {}
        for side in ("l", "r"):
            a, b = seg["sides"][side], ref_row["sides"][side]
            side_ok[side] = all(
                a[k] == b[k] for k in
                ("ref_frames", "certified_frames", "certified_in_ref_frames",
                 "intersection_longest_run_frames", "frames_within_enter"))
        A.check(f"dq_counts_reproduced[{seg['demo']}]",
                side_ok["l"] and side_ok["r"], side_ok)

    # ---- per-run traces (section 3 items 2-3) ------------------------------
    print("== per-run traces ==")
    traces = {}                       # (demo, src) -> analysis dict
    c2_states = {}                    # (demo, src) -> c2 trace
    prev_all = {}                     # (demo, src) -> prev distances
    gaps = {}
    cls_all = {}                      # (demo, side) -> classification
    for demo in DEMOS:
        src_scene, cond = conds[demo]
        rq_traj, _, _ = load_teacher_npz(CUR_RQ_DIR / f"{demo}.npz")
        prev_rq = prev_scene_distances(rq_traj)
        prev_all[(demo, "rq")] = prev_rq
        tr_rq = analyze_run(CUR_RQ_DIR / f"{demo}.npz", cond, vcfg)
        traces[(demo, "rq")] = tr_rq
        c2_states[(demo, "rq")] = c2_enable_trace(cond, prev_rq)
        gaps[(demo, "rq")] = enable_gap_frames(tr_rq,
                                               c2_states[(demo, "rq")], vcfg)
        out = {"demo": demo, "source": "rq", "series": tr_rq,
               "prev_distances": {s: {k: prev_rq[s][k]
                                      for k in ("scene", "proj")}
                                  for s in ("l", "r")},
               "prev_query_miss": {s: prev_rq[s]["miss"] for s in ("l", "r")},
               "c2_proposed": c2_states[(demo, "rq")],
               "enable_gap_frames": gaps[(demo, "rq")]}
        if demo not in NO_HIST:
            nq_traj, _, _ = load_teacher_npz(HIST_NQ_DIR / f"{demo}.npz")
            prev_nq = prev_scene_distances(nq_traj)
            prev_all[(demo, "nq")] = prev_nq
            tr_nq = analyze_run(HIST_NQ_DIR / f"{demo}.npz", cond, vcfg)
            traces[(demo, "nq")] = tr_nq
            c2_states[(demo, "nq")] = c2_enable_trace(cond, prev_nq)
            gaps[(demo, "nq")] = enable_gap_frames(tr_nq,
                                                   c2_states[(demo, "nq")],
                                                   vcfg)
            for side in ("l", "r"):
                cls_all[(demo, side)] = classify_side(
                    tr_nq["sides"][side], tr_rq["sides"][side], vcfg)
            out.update({
                "nq_series": tr_nq,
                "nq_prev_distances": {s: {k: prev_nq[s][k]
                                          for k in ("scene", "proj")}
                                      for s in ("l", "r")},
                "nq_prev_query_miss": {s: prev_nq[s]["miss"]
                                       for s in ("l", "r")},
                "nq_c2_proposed": c2_states[(demo, "nq")],
                "nq_enable_gap_frames": gaps[(demo, "nq")],
                "classification_vs_nq": {s: cls_all[(demo, s)]
                                         for s in ("l", "r")},
            })
        _write_guarded(out_dir / f"cert_trace_{demo}_rq.json",
                       json.dumps(out, indent=1))

    # ---- section 2.1 assertions -------------------------------------------
    print("== section 2.1 assertions ==")
    for (demo, side), exp in EXPECT_21.items():
        cls = cls_all[(demo, side)]
        A.check(f"21_intervals[{demo}:{side}]", _interval_match(cls, exp),
                {"old": cls["old_intersection"], "new": cls["new_intersection"]})
        A.check(f"21_counts[{demo}:{side}]",
                cls["old_counts"] == exp["old_counts"]
                and cls["new_counts"] == exp["new_counts"],
                {"old": cls["old_counts"], "new": cls["new_counts"]})
        gated = sorted(t for t, k in cls["lost"].items() if k == "GATED")
        mind = sorted(t for t, k in cls["lost"].items() if k == "MIN_DUR")
        distl = sorted(t for t, k in cls["lost"].items() if k == "DIST_LOST")
        A.check(f"21_lost_sets[{demo}:{side}]",
                gated == exp["gated"] and mind == exp["min_dur"]
                and not distl,
                {"gated": gated, "min_dur": mind, "dist_lost": distl})
        A.check(f"21_gain[{demo}:{side}]", cls["gained"] == exp["gain"],
                {"gained": cls["gained"]})
    tot = {"GATED": 0, "MIN_DUR": 0}
    for (d, _s) in EXPECT_21:
        for k in cls_all[(d, _s)]["lost"].values():
            if k in tot:
                tot[k] += 1
    tot_gain = sum(len(cls_all[(d, _s)]["gained"]) for (d, _s) in EXPECT_21)
    A.check("21_totals_22_lost_9_gain",
            tot["GATED"] == 10 and tot["MIN_DUR"] == 12 and tot_gain == 9,
            {"gated": tot["GATED"], "min_dur": tot["MIN_DUR"],
             "gain": tot_gain})
    A.check("21_dist_lost_and_subcategories_zero",
            all(not c["dist_lost_sub"] for c in cls_all.values()),
            "entry-denial / re-entry-rejected / hysteresis-detach: 0")
    A.note("21_lost_evidence", {f"{d}:{s}": cls_all[(d, s)]["lost_evidence"]
                                for (d, s) in EXPECT_21})

    # ---- section 2.4 assertions -------------------------------------------
    print("== section 2.4 assertions ==")
    for demo, side, t, scene_mm, proj_mm, spd in EXPECT_24:
        d = prev_all[(demo, "rq")][side]["scene"][t]
        p = prev_all[(demo, "rq")][side]["proj"][t]
        s = traces[(demo, "rq")]["sides"][side]["h_speed"][t]
        A.check(f"24_scene_dist[{demo}:{side}:{t}]",
                d is not None and round(d * 1e3, 6) == scene_mm,
                {"got_mm": None if d is None else d * 1e3, "exp_mm": scene_mm})
        A.check(f"24_projected_dist[{demo}:{side}:{t}]",
                p is not None and round(p * 1e3, 6) == proj_mm,
                {"got_mm": None if p is None else p * 1e3, "exp_mm": proj_mm})
        A.check(f"24_speed[{demo}:{side}:{t}]",
                s is not None and round(s, 7) == spd,
                {"got": s, "exp": spd})
    proj3 = prev_all[("sub3_largebox_000", "rq")]["r"]["proj"]
    A.check("24_sub3r_spikes_in_old_8mm_gate",
            all(proj3[t] is not None and proj3[t] <= 0.008
                for t in SUB3R_SPIKES),
            {t: (None if proj3[t] is None else proj3[t] * 1e3)
             for t in SUB3R_SPIKES})
    for (demo, side), (n_over, n_tot) in EXPECT_COMPONENT.items():
        s = traces[(demo, "nq")]["sides"][side]
        sup_n = sum(1 for v in s["supervision"] if v)
        over = [t for t in range(len(s["supervision"]))
                if s["supervision"][t]
                and s["max_component"][t] is not None
                and s["max_component"][t] > V_ROW]
        A.check(f"24_component_over[{demo}:{side}]",
                len(over) == n_over and sup_n == n_tot,
                {"over": len(over), "total": sup_n})
    s10 = traces[("sub10_largebox_000", "nq")]["sides"]["r"]
    A.check("24_t42_norm_component",
            round(s10["h_speed"][42], 8) == EXPECT_T42[0]
            and round(s10["max_component"][42], 8) == EXPECT_T42[1],
            {"norm": s10["h_speed"][42], "comp": s10["max_component"][42]})
    for demo, side, t, *_ in EXPECT_24:
        A.check(f"c2_covers_spike[{demo}:{side}:{t}]",
                bool(c2_states[(demo, "rq")][side]["enabled"][t]),
                c2_states[(demo, "rq")][side]["reason"][t])
    for t in SUB3R_SPIKES:
        st = c2_states[("sub3_largebox_000", "rq")]["r"]
        A.check(f"c2_covers_spike[sub3:r:{t}]", bool(st["enabled"][t]),
                st["reason"][t])
    A.note("prev_query_miss_all",
           {f"{k[0]}:{k[1]}:{s}": prev_all[k][s]["miss"]
            for k in prev_all for s in ("l", "r")})

    # ---- counterfactuals (section 3 item 4; review R3 full audit) ---------
    cf_records = []
    if not args.skip_counterfactuals:
        print("== counterfactuals (full audit, review R3) ==")
        spec = RepairPipeline(cfg=replace(rcfg, regen_mode=True)).spec
        for demo in sorted({d for (d, _s) in FAIL_SIDES}):
            src_scene, cond = conds[demo]
            rq_traj, _, _ = load_teacher_npz(CUR_RQ_DIR / f"{demo}.npz")
            nq_traj = None
            replay = ChainReplay(cond, src_scene, rq_traj, rcfg, vcfg)
            c2_rq = c2_states[(demo, "rq")]
            tasks = []                      # (t, order, side, kind)
            for side in sorted(s for (d, s) in FAIL_SIDES if d == demo):
                spikes = sorted(t for t, k in
                                cls_all[(demo, side)]["lost"].items()
                                if k == "GATED")
                for t in spikes:
                    tasks.append((t, 0, side, "current_rq_adopted_spike"))
                sup = traces[(demo, "nq")]["sides"][side]["supervision"]
                for t in range(len(sup)):
                    if sup[t]:
                        tasks.append((t, 1, side,
                                      "historical_certified_candidate"))
            tasks.sort(key=lambda x: (x[0], x[1], x[2]))
            rq_npz = CUR_RQ_DIR / f"{demo}.npz"
            prev_source = {"npz": str(rq_npz), "sha256": sha256(rq_npz)}
            for t, _order, side, kind in tasks:
                if kind == "current_rq_adopted_spike":
                    z_cand = rq_traj[t, 0:36]
                    origin = {"kind": "rq_adopted",
                              "npz": str(rq_npz),
                              "npz_sha256": sha256(rq_npz),
                              "frame": int(t)}
                else:
                    if nq_traj is None:
                        nq_traj, _, _ = load_teacher_npz(
                            HIST_NQ_DIR / f"{demo}.npz")
                    z_cand = nq_traj[t, 0:36]
                    origin = {"kind": "ported_nq_frame",
                              "npz": str(HIST_NQ_DIR / f"{demo}.npz"),
                              "npz_sha256": sha256(HIST_NQ_DIR / f"{demo}.npz"),
                              "frame": int(t)}
                rec, arrays = counterfactual(
                    replay, t, z_cand, f"{kind}[{side}]", c2_rq, side,
                    prev_all[(demo, "rq")], prev_source, rq_traj, src_scene,
                    spec, rcfg, vcfg)
                rec.update({"demo": demo, "kind": kind,
                            "origin": origin})
                cf_records.append(rec)
                # one frame can be BOTH a spike and a historical certified
                # candidate -- the kind is part of the product identity
                write_products(out_dir,
                               f"cert_trace_cf_{demo}_t{t}_{side}_{kind}",
                               rec, arrays)
                print(f"  [{demo} t={t} {side} {kind}] "
                      f"{rec['classification']}")
        summary = {
            "format": AUDIT_FORMAT,
            "records": [{**{k: r[k] for k in
                            ("demo", "side", "kind", "frame",
                             "classification", "classification_basis",
                             "origin")},
                         "stem": f"cert_trace_cf_{r['demo']}_t{r['frame']}_"
                                 f"{r['side']}_{r['kind']}"}
                        for r in cf_records],
            "verdicts": {},
        }
        for r in cf_records:
            key = f"{r['kind']}:{r['classification']}"
            summary["verdicts"][key] = summary["verdicts"].get(key, 0) + 1
        _write_guarded(out_dir / "cert_trace_counterfactuals.json",
                       json.dumps(summary, indent=1, allow_nan=False))
        # reviewfix2 D5: the replay's EXPECTED candidate list -- discovery
        # never globs the products dir again (a replay summary re-entering
        # as a candidate was the second-run failure)
        write_index(out_dir, INDEX_NAME, {
            "records": summary["records"],
            "stems": [r["stem"] for r in summary["records"]]})
        A.note("counterfactual_classifications", summary["verdicts"])

    # ---- write assertion/provenance outputs --------------------------------
    prov["H_index"], prov["B_index"], prov["T_index"] = H, B, T
    _write_guarded(out_dir / "cert_trace_assertions.json", json.dumps({
        "assertions": A.rows,
        "enable_gap_summary": {f"{k[0]}:{k[1]}": len(v)
                               for k, v in gaps.items()},
        "wall_s": time.perf_counter() - t_start}, indent=1))
    _write_guarded(out_dir / "cert_trace_provenance.json",
                   json.dumps(prov, indent=1, default=str))
    print(f"-> {out_dir} ({time.perf_counter() - t_start:.1f}s, "
          f"{len(A.rows)} assertion rows, {len(cf_records)} counterfactuals)")


if __name__ == "__main__":
    main()
