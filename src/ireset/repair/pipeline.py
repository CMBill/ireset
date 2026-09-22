"""Sequential check-then-repair pipeline (plan sections 8.4/8.5).

Per frame: try to *skip* (the prediction already satisfies frame checks and
task quality), otherwise run the SCP solver; on failure escalate to a
slack-forced restoration fallback; if that fails too, reject with
diagnostics. Contact anchors are committed only from accepted states.

The accepted prefix is finally validated by the IndependentValidator. A
failing final validation triggers one suffix rollback-and-repair retry: the
prefix (its accepted states and the anchors they committed) is preserved by
replaying it as the prediction, and only the suffix is re-solved.

Output is a TrajectoryResult: accepted results carry the scene qpos;
rejected ones carry diagnostics only (never an exportable trajectory).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import trimesh

from ireset.embodied.contact_sites import HAND_TIP_JOINTS, SMPLH_INDEX, infer_reference_schedule
from ireset.embodied.g1_spec import G1Spec, KPT_TO_SMPLH
from ireset.geometry.collision_pairs import build_collision_pairs
from ireset.geometry.distance_adapter import largebox_hull
from ireset.imesh.delaunay import build_frame_mesh, laplacian_coordinates
from ireset.geometry.distance_adapter import world_to_box_geom
from ireset.kinematics.mujoco_adapter import retract_batch
from ireset.repair.constraints import FrameEvaluator, tangent_error
from ireset.repair.context import EFFECTORS, ContactStateMachine, ReferenceCondition, RepairConfig
from ireset.repair.scp import FrameCheck, FrameContext, FrameSolver, solver_band_margin
from ireset.repair.validation import IndependentValidator

KPT_IDX = [SMPLH_INDEX[v] for v in KPT_TO_SMPLH.values()]
N_SURF = 12   # object surface samples for the interaction mesh
N_GROUND = 8  # world-fixed ground points


def _surface_target(p: np.ndarray, box_pose: np.ndarray,
                    clearance: float = 0.008, max_scale: float = 3.0) -> np.ndarray:
    """Hand-tracking target projected to just OUTSIDE the box hull.

    The wrist+fingertip centroid of a grasping hand sits a couple of cm
    INSIDE the surface; tracking it directly presses the robot hand geoms
    against the -2 mm penetration floor. Bisection along (box_pos -> p)
    finds the first point with the requested hull clearance (8 mm: site
    centroid at 8 mm leaves the collision spheres a few mm of margin --
    5 mm grazed the floor through the sphere radii).
    """
    from ireset.geometry.distance_adapter import largebox_hull_surface_distance

    sdf = float(largebox_hull_surface_distance(p[None], box_pose[None])[0])
    if sdf >= clearance:
        return p
    d = p - box_pose[0:3]
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return p
    d = d / n
    lo, hi = 0.0, max_scale
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        q = box_pose[0:3] + mid * d * n
        if float(largebox_hull_surface_distance(q[None], box_pose[None])[0]) >= clearance:
            hi = mid
        else:
            lo = mid
    return box_pose[0:3] + hi * d * n


@dataclass
class TrajectoryResult:
    accepted: bool
    status: str          # accepted / rejected_frame / rejected_final /
                         # rejected_nonfinite / rejected_shape
    trajectory: np.ndarray | None = None   # scene qpos [T, 43], accepted only
    diagnostics: dict = field(default_factory=dict)


class RepairPipeline:
    def __init__(self, spec: G1Spec | None = None, cfg: RepairConfig | None = None) -> None:
        self.cfg = cfg or RepairConfig()
        self.spec = spec or G1Spec()
        self.pairs = build_collision_pairs(self.spec, scene=True)
        self.ev = FrameEvaluator(self.spec, self.pairs, self.cfg)
        self.solver = FrameSolver(self.ev, self.cfg)
        self.validator = IndependentValidator(spec=self.spec)
        self._surf_pts: np.ndarray | None = None
        self._ground_pts: np.ndarray | None = None
        #: per-frame check/probe diagnostics (regeneration debugging); only
        #: populated when ``verbose`` is set -- strictly additive, the solve
        #: path never reads it
        self.verbose: bool = False
        self.frame_log: list[dict] = []

    # -- interaction mesh precompute -------------------------------------------

    def _mesh_assets(self, condition: ReferenceCondition) -> None:
        # surface samples are demo-independent (seeded once); the ground
        # points belong to THIS condition's box trajectory and must be
        # recomputed per clip -- caching them across segments leaked demo-1
        # geometry into every later segment of a shared-pipeline run
        if self._surf_pts is None:
            pts, _ = trimesh.sample.sample_surface(largebox_hull(), N_SURF, seed=0)
            self._surf_pts = np.asarray(pts, dtype=float)
        gp = []
        for c in (condition.box_pose[0, 0:3], condition.box_pose[-1, 0:3]):
            for dx, dy in ((-0.5, -0.5), (0.5, 0.5)):
                gp.append([c[0] + dx, c[1] + dy, 0.0])
        self._ground_pts = np.asarray(gp, dtype=float)

    def _frame_mesh(self, condition: ReferenceCondition, t: int):
        """Reference-side per-frame mesh (L, delta_ref, robot-side fixed pts)."""
        self._mesh_assets(condition)
        pose = condition.box_pose[t : t + 1]
        kpt_g = world_to_box_geom(condition.human_kpts_world[t][KPT_IDX][None], pose)[0]
        ground_g = world_to_box_geom(self._ground_pts[None], pose)[0]
        X_ref = np.vstack([kpt_g, self._surf_pts, ground_g])
        mesh = build_frame_mesh(X_ref, seed=t)
        delta = laplacian_coordinates(X_ref, mesh)
        return mesh.lap, delta, np.vstack([self._surf_pts, ground_g])

    # -- frame context -----------------------------------------------------------

    def _establishment_candidates(self, z: np.ndarray, box: np.ndarray, contacts, t: int,
                                  prev_sole: dict | None = None) -> dict:
        """Anchor candidates from an ACCEPTED state, gated by geometry (plan 8.1).

        Hands must be within the establishment distance of the box *without
        deep penetration*; feet must be near the ground. Called after a frame
        is accepted (skipped or repaired).

        P0.1 L1: feet additionally pass a quasi-static gate -- a foot sliding
        faster than ``foot_anchor_establish_speed_mps`` (measured against the
        previous *accepted* sole positions) may touch (band rows apply via the
        reference schedule) but must not lock a world anchor it cannot honor.
        """
        out = {"hand": {}, "foot": {}}
        self.ev.forward(z, box)
        for side, name in (("l", "left_hand"), ("r", "right_hand")):
            # regen mode: no hand anchors at all (contact via the hard gap
            # row; drift judged by the validator at screening)
            if self.cfg.regen_mode:
                continue
            if contacts.holding[name] or not contacts.reference_wants_hold(name, t):
                continue
            dist = min(self.ev.geom_distance(g, self.ev.box_geom)[0] for g in self.ev._hand_geoms[side])
            if -self.cfg.contact_band_pen_m <= dist <= self.cfg.hand_establish_dist_m:
                out["hand"][name] = self.ev.hand_local(side)
        for side, name in (("l", "left_foot"), ("r", "right_foot")):
            if contacts.holding[name] or not contacts.reference_wants_hold(name, t):
                continue
            pts = self.ev.site_positions(self.ev.sole_sites[side])
            if prev_sole and prev_sole.get(side) is not None:
                speed = float(np.linalg.norm(pts - prev_sole[side], axis=-1).max()) * self.cfg.fps
                if speed >= self.cfg.foot_anchor_establish_speed_mps:
                    continue
            gap = min(self.ev.geom_distance(g, self.ev.ground_geom)[0] for g in self.ev.sole_geoms[side])
            if -self.cfg.contact_band_pen_m <= gap <= self.cfg.foot_establish_gap_m:
                out["foot"][name] = pts
        return out

    def _frame_ctx(self, t, prediction, condition, contacts, prev_local, prev_sole, z_prev):
        # Only anchors already LOCKED from accepted states constrain the solve
        # (plan 8.4): a first hold frame solves freely and its accepted state
        # provides the anchor afterwards.
        hand_anchor, foot_anchor = {}, {}
        for side, name in (("l", "left_hand"), ("r", "right_hand")):
            hand_anchor[side] = contacts.anchors[name] if contacts.holding[name] else None
        for side, name in (("l", "left_foot"), ("r", "right_foot")):
            foot_anchor[side] = contacts.anchors[name] if contacts.holding[name] else None
        L, delta, fixed = self._frame_mesh(condition, t)
        # Path B approach corridor input + hand-tracking target
        prev_hand_dist = {}
        ref_hand_kpts = {}
        ref_hand_delta = {}
        if self.cfg.regen_mode:
            z_ref = z_prev if z_prev is not None else prediction[t, 0:36]
            self.ev.forward(z_ref, prediction[t, 36:43])
            pose = prediction[t, 36:43]
            pose_prev = prediction[t - 1, 36:43] if t > 0 else pose
            for side in ("l", "r"):
                prev_hand_dist[side] = min(
                    self.ev.geom_distance(g, self.ev.box_geom)[0]
                    for g in self.ev._hand_geoms[side])
                ids = [SMPLH_INDEX[n] for n in HAND_TIP_JOINTS["left" if side == "l" else "right"]]
                p = condition.human_kpts_world[t][ids].mean(0)
                ref_hand_kpts[side] = _surface_target(p, pose)
                # reference box-frame displacement (F4 section 7.2): the
                # human hand centroid in the box GEOM frame (the frame
                # hand_local lives in), this frame minus the previous one --
                # computed from the INPUT (box trajectory + reference
                # keypoints), never from the robot output
                if t > 0:
                    p_prev = condition.human_kpts_world[t - 1][ids].mean(0)
                    ref_hand_delta[side] = (self._box_geom_local(p, pose)
                                            - self._box_geom_local(p_prev, pose_prev))
        return FrameContext(
            t=t, box_pose=prediction[t, 36:43].copy(),
            z_pred=prediction[t, 0:36].copy(), z_prev=z_prev,
            ref_kpts_world=condition.human_kpts_world[t][KPT_IDX],
            mesh_L=L, delta_ref=delta, mesh_fixed=fixed,
            hand_anchor=hand_anchor, foot_anchor=foot_anchor,
            prev_hand_local=prev_local, prev_sole_world=prev_sole,
            box_pose_prev=prediction[t - 1, 36:43].copy() if t > 0 else None,
            ref_hold={name: contacts.reference_wants_hold(name, t) for name in EFFECTORS},
            prev_hand_dist=prev_hand_dist,
            ref_hand_kpts=ref_hand_kpts,
            ref_hand_delta=ref_hand_delta,
            frame_motion_m=(
                self.validator._frame_motion_m(
                    np.r_[z_prev, prediction[t - 1, 36:43]],
                    np.r_[prediction[t, 0:36], prediction[t, 36:43]])
                if z_prev is not None else 0.0),
        )

    # -- frame-level checks (skip logic, plan 8.4) ---------------------------------

    def _box_geom_local(self, p_world: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
        """World point -> box GEOM frame (the frame ``ev.hand_local`` uses).

        Evaluated on the evaluator's probe data: the box frame depends only
        on the box pose, so the endpoint state stays untouched.
        """
        self.ev.probe_forward(np.r_[np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(29)],
                              box_pose)
        R, x = self.ev.probe_box_frame()
        return R.T @ (np.asarray(p_world, dtype=float) - x)

    def _hand_gaps(self) -> dict:
        """Min hand-geom/box distance per side at the CURRENT forwarded state.

        Read-only diagnostics built from the evaluator state a check already
        forwarded to; no extra FK, no re-forward (F0: verbose must not add
        checks or FK).
        """
        return {s: min(self.ev.geom_distance(g, self.ev.box_geom)[0]
                       for g in self.ev._hand_geoms[s]) for s in ("l", "r")}

    def _frame_valid(self, z: np.ndarray, ctx: FrameContext) -> FrameCheck:
        """Endpoint + transition + task-quality check for a candidate state.

        Returns the decision plus its evidence. Pairs found violating are
        recorded with their stable identities; the CALLER decides when to
        feed them back via ``carry_discovered`` (F0: checks and carry-back
        are separate actions). Verbose logging serializes this record -- it
        never triggers another check, FK, carry-back or cache change.
        """
        # fresh pair scan for acceptance decisions: the solver's per-frame
        # cache reflects the linearization state, not this candidate; the
        # check path is UNCAPPED (F1: a success may not hide behind the QP
        # row budget)
        self.solver._frame_pairs = None
        val, lo, hi, scl, names, ids = self.solver.check_rows(z, ctx)
        ep_viol = self.solver.violation(val, lo, hi, scl)
        chk = FrameCheck(
            ok=ep_viol <= 1e-6,
            viol=float(ep_viol),
            corridor={s: dict(c) for s, c in self.solver.last_corridor.items()},
            band_margin=solver_band_margin(ctx, self.cfg),
            truncated_rows=list(self.solver.last_truncated_names),
        )
        chk.hand_prev = {s: float(v) for s, v in ctx.prev_hand_dist.items()}
        chk.hand_cand = self._hand_gaps()
        if ep_viol > 1e-6:
            for k in range(len(names)):
                v, l_, h_ = val[99 + k], lo[99 + k], hi[99 + k]
                if v < l_ - 1e-9 or v > h_ + 1e-9:
                    chk.rows.append({"pair": names[k] if k < len(names) else "?",
                                     "val_m": float(v), "lo_m": float(l_),
                                     "hi_m": float(h_)})
                    chk.carried.append(names[k] if k < len(names) else "?")
                    if k < len(ids):
                        chk.discovered.append(ids[k])
            chk.reason = "state_constraints"
            return chk
        # interpolated probes between the previous accepted scene and the
        # candidate: the final validator interpolates FULL scenes (robot +
        # box), so the sequential loop must pre-check the same transitions
        # -- a box sweeping through the robot between keyframes, and the
        # previous robot state against the previous (not current) box pose
        # (P0-fix F2, 2026-09-14). Probe count mirrors the validator's
        # ADAPTIVE subdivision via its own motion metric and respects its
        # check_interpolated_states flag. Interior probes check only the
        # collision-family rows against the INTERPOLATED box pose: velocity
        # and anchor rows are defined between accepted frames over the full
        # dt, not at interior samples (same scope as the validator's probes)
        if ctx.z_prev is not None and self.validator.cfg.check_interpolated_states:
            scene_a = np.zeros(43)
            scene_a[0:36] = ctx.z_prev
            scene_a[36:43] = ctx.box_pose_prev if ctx.box_pose_prev is not None else ctx.box_pose
            scene_b = np.zeros(43)
            scene_b[0:36] = z
            scene_b[36:43] = ctx.box_pose
            vcfg = self.validator.cfg
            motion = self.validator._frame_motion_m(scene_a, scene_b)
            n_sub = int(np.clip(np.ceil(motion / vcfg.max_substep_motion_m),
                                1, vcfg.max_subdivisions + 1))
            chk.motion_m = float(motion)
            chk.probe_n = int(n_sub)
            for i in range(1, n_sub):
                mid = self.validator._interpolate(scene_a, scene_b, i / n_sub)
                ctx_mid = replace(ctx, box_pose=mid[36:43].copy())
                # rescan per probe: the endpoint candidate set cannot see
                # pairs that come into range only at interior states;
                # probes judge at the VALIDATOR floor (band_margin 0) and
                # are uncapped like the endpoint check
                self.solver._frame_pairs = None
                val_m, lo_m, hi_m, scl_m, names_m, ids_m = self.solver.check_rows(
                    mid[0:36], ctx_mid, band_margin_m=0.0,
                    probe_mode=True   # stage 2: mid-states carry no frame-end corridor upper
                )
                n_coll = 99 + len(names_m)
                viol_m = self.solver.violation(val_m[99:n_coll], lo_m[99:n_coll],
                                               hi_m[99:n_coll], scl_m[99:n_coll])
                if viol_m <= 1e-6:
                    continue
                gaps = self._hand_gaps()
                # failure snapshot only (F0: representative/failed states,
                # never every passing probe): the exact interpolated state,
                # its box pose and the interpolation fraction, so a
                # "endpoint ok, transition collides" can be replayed without
                # re-deriving the subdivision
                rec = {"i": i, "n_sub": int(n_sub), "alpha": i / n_sub,
                       "viol": float(viol_m),
                       "hand_l_mid_m": float(gaps["l"]),
                       "hand_r_mid_m": float(gaps["r"]),
                       "motion_m": float(motion),
                       "q_mid": [float(v) for v in mid[0:36]],
                       "box_pose_mid": [float(v) for v in mid[36:43]],
                       "rows": []}
                for k in range(n_coll - 99):
                    v, l_, h_ = val_m[99 + k], lo_m[99 + k], hi_m[99 + k]
                    if v < l_ - 1e-9 or v > h_ + 1e-9:
                        nm = names_m[k] if k < len(names_m) else "?"
                        rec["rows"].append({"pair": nm, "val_m": float(v),
                                            "lo_m": float(l_), "hi_m": float(h_)})
                        chk.carried.append(nm)
                        if k < len(ids_m):
                            chk.discovered.append(ids_m[k])
                            chk.probe_discovered.append(ids_m[k])
                chk.probes.append(rec)
                chk.probe_viol = max(chk.probe_viol, float(viol_m))
                chk.reason = "transition_interpolated"
                chk.ok = False           # a failing probe REJECTS the candidate
                return chk
        res = self.solver.evaluate(z, ctx)
        if res["mpjpe"] > self.cfg.task_mpjpe_tol_m:
            chk.ok = False
            chk.reason = "task_quality"
        return chk

    def frame_valid(self, z: np.ndarray, ctx: FrameContext) -> FrameCheck:
        """One-shot acceptance check (no carry-back side effect)."""
        return self._frame_valid(z, ctx)

    def carry_discovered(self, chk: FrameCheck) -> list[str]:
        """Feed a FAILED check's violating pairs into the next linearization.

        Algorithm-level action, separate from the check itself (F0); the
        solver re-resolves the geometric category from the registry (F1).
        """
        if chk.discovered:
            self.solver.note_discovered_pairs(chk.discovered)
        if chk.probe_discovered:
            self.solver.note_probe_pairs(chk.probe_discovered)
        return chk.carried

    def check_result(self, res, ctx: FrameContext, caller: str) -> FrameCheck:
        """Cached frame check for a solver result (same state -> one check).

        The outcome is stored on the result so attempt logging and the
        accept/reject decision share ONE check (F0: verbose adds no checks).
        """
        if res.frame_check is None:
            res.frame_check = self._frame_valid(res.z, ctx)
        chk = res.frame_check
        self._log_check(chk, ctx.t, caller, res)
        return chk

    def _log_check(self, chk: FrameCheck, t: int, caller: str, res=None) -> None:
        """Serialize an ALREADY computed check (no re-computation)."""
        if not self.verbose:
            return
        rec = {"t": t, "kind": "check", "caller": caller, "ok": bool(chk.ok),
               "reason": chk.reason, "viol": float(chk.viol),
               "hand_l_prev_m": float(chk.hand_prev.get("l", float("nan"))),
               "hand_l_cand_m": float(chk.hand_cand.get("l", float("nan"))),
               "hand_r_prev_m": float(chk.hand_prev.get("r", float("nan"))),
               "hand_r_cand_m": float(chk.hand_cand.get("r", float("nan"))),
               "corridor": chk.corridor, "band_margin_m": chk.band_margin}
        if chk.rows:
            rec["rows"] = chk.rows
            rec["carried"] = chk.carried
        self.frame_log.append(rec)
        for p in chk.probes:
            self.frame_log.append({"t": t, "kind": "probe", "caller": caller, **p})

    # -- public API ------------------------------------------------------------------

    def repair_trajectory(self, prediction: np.ndarray, condition: ReferenceCondition) -> TrajectoryResult:
        """End-to-end repair; ``diagnostics["total_s"]`` covers both attempts
        plus all final validations (the timing callers must report)."""
        t0 = time.perf_counter()
        self.solver.probe_cfg = self.validator.cfg
        if self.verbose:
            self.frame_log = []
        result = self._repair_trajectory(prediction, condition)
        result.diagnostics.setdefault("total_s", round(time.perf_counter() - t0, 3))
        return result

    def _repair_trajectory(self, prediction: np.ndarray, condition: ReferenceCondition) -> TrajectoryResult:
        prediction = np.asarray(prediction, dtype=float)
        if prediction.ndim != 2 or prediction.shape[1] != 43:
            return TrajectoryResult(False, "rejected_shape")
        if not np.isfinite(prediction).all():
            return TrajectoryResult(False, "rejected_nonfinite")

        result = self._run_sequential(prediction, condition)
        if not result.accepted:
            return result

        # box_ground is input-owned (fixed box trajectory; the hull of the
        # scanned bin also over-approximates its bottom) -> excluded here,
        # reported by data screening instead
        t_final = time.perf_counter()
        final = self.validator.validate(result.trajectory, exclude=("box_ground",))
        result.diagnostics["final_validation_s"] = round(time.perf_counter() - t_final, 3)
        if final.passed:
            result.diagnostics["final_validation"] = "passed"
            result.diagnostics["final_checks"] = {
                k: {"worst": round(float(v.worst), 4), "limit": float(v.limit)}
                for k, v in final.checks.items()
            }
            return result

        # suffix rollback: keep the validated prefix (states + anchors) by
        # replaying it as the prediction, re-solve only the suffix. The
        # retry start comes from the STRUCTURED first-failure frames (min
        # over failed checks); an unlocated failure retries the FULL
        # segment -- retrying only the tail would silently keep whatever
        # the missing location hides (P0-fix F3, 2026-09-14)
        f_frame = self._first_failure_frame(final)
        start = 0 if f_frame is None else max(0, f_frame - self.cfg.suffix_retry_window)
        retry_pred = np.vstack([result.trajectory[:start], prediction[start:]])
        retry = self._run_sequential(retry_pred, condition, pass_index=1)
        retry.diagnostics["first_attempt_final"] = f"{final.summary()} @t={f_frame}"
        retry.diagnostics["first_failure_frame"] = f_frame
        retry.diagnostics["suffix_retry_from"] = start
        retry.diagnostics["suffix_retries"] = 1
        retry.diagnostics["failed_checks"] = {
            k: {"location": v.location, "first_failure_frame": v.first_failure_frame}
            for k, v in final.checks.items() if not v.passed
        }
        # the first pass's cost is reported VERBATIM whichever way the retry
        # ends (a retry must never overwrite the first round's
        # attempts/QP/iteration totals, and a SUCCESSFUL retry must not drop
        # them either)
        retry.diagnostics["first_attempt"] = dict(result.diagnostics)
        if not retry.accepted:
            return TrajectoryResult(False, "rejected_final", diagnostics=retry.diagnostics)
        t_final = time.perf_counter()
        final2 = self.validator.validate(retry.trajectory, exclude=("box_ground",))
        retry.diagnostics["final_validation_s"] = round(
            retry.diagnostics.get("final_validation_s", 0.0)
            + time.perf_counter() - t_final, 3)
        if final2.passed:
            retry.diagnostics["final_validation"] = "passed_after_suffix_retry"
            retry.diagnostics["final_checks"] = {
                k: {"worst": round(float(v.worst), 4), "limit": float(v.limit)}
                for k, v in final2.checks.items()
            }
            return retry
        retry.diagnostics["final_retry_failed_checks"] = {
            k: {"location": v.location, "first_failure_frame": v.first_failure_frame}
            for k, v in final2.checks.items() if not v.passed
        }
        return TrajectoryResult(False, "rejected_final", diagnostics={
            **retry.diagnostics, "final_retry": final2.summary(),
        })

    # -- sequential core ----------------------------------------------------------------

    def _record_attempt(self, attempts: list, res, source: str, t: int,
                        pass_index: int, backtrack: str | None = None) -> None:
        """Record ONE solver attempt (always on -- the aggregate counters are
        recomputable from these events; verbose adds the same record to the
        frame log). Costs are per-attempt, so a retry can never overwrite the
        first round's totals."""
        rec = {
            "t": t, "pass": pass_index, "source": source, "backtrack": backtrack,
            "accepted": bool(res.accepted), "status": res.status,
            "iterations": int(res.iterations), "qp_calls": int(res.qp_calls),
            "slack_used": bool(res.slack_used), "trust": float(res.trust),
            "last_viol": float(res.last_viol),
            "last_qp_status": str(res.last_qp_status),
            "violated_rows": list(res.violated_rows),
            "truncated_rows": list(res.truncated_rows),
            "probe_rows": list(res.probe_rows),
            "probe_truncated": list(res.probe_truncated),
            "frame_valid": (None if res.frame_check is None
                            else bool(res.frame_check.ok)),
            "frame_valid_reason": (None if res.frame_check is None
                                   else res.frame_check.reason),
        }
        attempts.append(rec)
        if self.verbose:
            self.frame_log.append({"kind": "attempt", **rec})

    @staticmethod
    def _aggregate(attempts: list, visits: list, T: int, skipped: int,
                   backtracks: int, pass_index: int) -> dict:
        """Counters over ALL attempts (F0): totals, slack/backtrack adoption,
        processed-frame visits vs unique frames vs the original input length."""
        return {
            "trajectory_pass": pass_index,
            "attempts": len(attempts),
            "qp_calls_total": int(sum(a["qp_calls"] for a in attempts)),
            "iterations_total": int(sum(a["iterations"] for a in attempts)),
            "slack_attempts": sum(1 for a in attempts if a["slack_used"]),
            "backtracks": int(backtracks),
            "processed_frame_visits": len(visits),
            "unique_processed_frames": len(set(visits)),
            "input_frames": int(T),
            "skipped_frames": int(skipped),
            "attempts_detail": attempts,
        }

    @staticmethod
    def _timing_stats(timings: list) -> dict:
        if not timings:
            return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
        v = np.asarray(timings, dtype=float) * 1e3
        return {"mean_ms": round(float(v.mean()), 1),
                "p50_ms": round(float(np.percentile(v, 50)), 1),
                "p95_ms": round(float(np.percentile(v, 95)), 1),
                "max_ms": round(float(v.max()), 1)}

    def _run_sequential(self, prediction, condition, pass_index: int = 0) -> TrajectoryResult:
        T = prediction.shape[0]
        contacts = ContactStateMachine(condition.ref_schedule, T)
        accepted: list[np.ndarray] = []
        prev_local: dict = {}
        prev_sole: dict = {}
        frame_status: list[str] = []
        adopted_iterations: list[int] = []
        adopted_qp: list[int] = []
        qp_calls_hist: list[int] = []
        timings: list[float] = []
        slack_flags: list[bool] = []
        fallbacks = 0
        backtracks = 0
        attempts: list[dict] = []       # every solve attempt, always recorded
        visits: list[int] = []          # processed-frame visits in this pass
        skip_reasons: dict[int, str] = {}

        for t in range(T):
            t0 = time.perf_counter()
            visits.append(t)
            # stage 1: discovered-pair carry-over is a WITHIN-FRAME mechanism
            # (attempt N learns what attempt N-1's checks caught); a new
            # frame is a new state and starts from a clean candidate set
            self.solver.clear_discovered_pairs()
            self.solver.clear_probe_pairs()
            z_pred = prediction[t, 0:36].copy()
            box = prediction[t, 36:43]
            z_prev = accepted[-1].copy() if accepted else None
            # P0.1 L1: anchors live only while the foot stays quasi-static.
            # The teacher slides in bursts, so an anchor locked on a slow
            # frame must degrade to touching-without-anchor once the
            # *prediction* foot slides again (reference-driven decision,
            # never the solver's -- the repaired foot is pinned and would
            # never trigger its own release).
            if prev_sole:
                self.ev.forward(z_pred, box)
                for side, name in (("l", "left_foot"), ("r", "right_foot")):
                    if contacts.holding[name] and prev_sole.get(side) is not None:
                        pts = self.ev.site_positions(self.ev.sole_sites[side])
                        speed = float(np.linalg.norm(pts - prev_sole[side], axis=-1).max()) * self.cfg.fps
                        if speed >= self.cfg.foot_anchor_establish_speed_mps:
                            contacts.release(name)
                # symmetric L1 completion for hands: an anchor the *prediction*
                # no longer honors (hand drifted beyond the anchor tolerance
                # in box frame) degrades to unanchored -- band classification
                # itself stays anchor-driven. Regeneration mode (Path B)
                # bypasses this: the hovering warm start would release every
                # anchor it just pulled in; hand anchors there follow the
                # reference schedule alone (released at interval end)
                if not self.cfg.regen_mode:
                    for side, name in (("l", "left_hand"), ("r", "right_hand")):
                        anchor = contacts.anchors[name]
                        if contacts.holding[name] and anchor is not None:
                            hl = self.ev.hand_local(side)
                            if float(np.linalg.norm(hl - anchor, axis=-1).max()) > self.cfg.hand_anchor_tol_m:
                                contacts.release(name)
            ctx = self._frame_ctx(t, prediction, condition, contacts, prev_local, prev_sole, z_prev)

            chk = self.frame_valid(z_pred, ctx)
            self._log_check(chk, t, "skip", None)
            if chk.ok:
                z = z_pred
                frame_status.append("skipped")
                adopted_iterations.append(0)
                adopted_qp.append(0)
                qp_calls_hist.append(0)
                slack_flags.append(False)
            else:
                skip_reasons[t] = chk.reason
                # feed the prediction's violations into the solve (algorithm
                # input, not logging): the candidate set is frozen at the
                # attempt's initial state and must see what the check caught
                self.carry_discovered(chk)
                res = self.solver.solve_frame(z_pred, ctx)
                path = "repaired" if not res.slack_used else "repaired_slack"
                self._record_attempt(attempts, res, "prediction", t, pass_index)
                res, bt = self._transition_backtrack(res, ctx, z_prev)
                if bt:
                    path = bt
                    backtracks += 1
                self.check_result(res, ctx, "prediction")
                if not res.frame_check.ok:
                    self.carry_discovered(res.frame_check)
                    # multi-start restart from the previous accepted state
                    # (P0.1 L0): deep-violation frames linearize badly from
                    # z_pred but solve from the accepted predecessor
                    res_alt = self.solver.solve_frame(z_prev, ctx) if z_prev is not None else None
                    if res_alt is not None:
                        self._record_attempt(attempts, res_alt, "prev_accepted", t, pass_index)
                        res_alt, bt2 = self._transition_backtrack(res_alt, ctx, z_prev)
                        if bt2:
                            backtracks += 1
                        chk_alt = self.check_result(res_alt, ctx, "prev_accepted")
                        if chk_alt.ok:
                            res, path = res_alt, bt2 or "repaired_prev"
                        else:
                            self.carry_discovered(chk_alt)
                    if not res.frame_check.ok and self.cfg.enable_slack_fallback:
                        res2 = self.solver.solve_frame(
                            z_prev if z_prev is not None else z_pred, ctx, force_slack=True
                        )
                        self._record_attempt(attempts, res2, "slack_fallback", t, pass_index)
                        chk2 = self.check_result(res2, ctx, "slack_fallback")
                        if chk2.ok:
                            res, path = res2, "repaired_slack"
                            fallbacks += 1
                        else:
                            self.carry_discovered(chk2)
                if not res.frame_check.ok:
                    # rejection evidence: the FINAL stage and the last
                    # attempt's exit state -- never the initial skip_reason
                    # (which only describes the raw prediction)
                    timings.append(time.perf_counter() - t0)
                    return TrajectoryResult(False, "rejected_frame", diagnostics={
                        "t_fail": t,
                        "reason": res.status,
                        "final_stage": res.frame_check.reason,
                        "skip_reason": skip_reasons.get(t),
                        "reason_source": "last_solve_attempt",
                        "last_attempt": attempts[-1] if attempts else None,
                        "violated_rows": res.frame_check.rows,
                        "probe": res.frame_check.probes[0] if res.frame_check.probes else None,
                        "accepted_frames": len(accepted), "frame_status": frame_status,
                        "iterations": adopted_iterations, "timings_s": timings,
                        "qp_calls": qp_calls_hist, "fallbacks": fallbacks,
                        "slack_frames": int(sum(slack_flags)),
                        "skipped": sum(1 for st in frame_status if st == "skipped"),
                        "timing_stats": self._timing_stats(timings),
                        **self._aggregate(attempts, visits, T,
                                          sum(1 for st in frame_status if st == "skipped"),
                                          backtracks, pass_index),
                    })
                z = res.z
                frame_status.append(path)
                adopted_iterations.append(res.iterations)
                adopted_qp.append(res.qp_calls)
                qp_calls_hist.append(res.qp_calls)
                # slack is an independent fact from the chosen solve branch:
                # multi-start/repaired_prev and backtrack paths overwrite the
                # path NAME, but the flag travels with the result object
                # (P0-fix F4, 2026-09-14)
                slack_flags.append(bool(res.slack_used))

            # commit contact transitions from the accepted state only:
            # anchors lock from the geometry that just passed all checks
            cands = self._establishment_candidates(z, box, contacts, t, prev_sole)
            contacts.commit("left_hand", t, hand_local=cands["hand"].get("left_hand"))
            contacts.commit("right_hand", t, hand_local=cands["hand"].get("right_hand"))
            contacts.commit("left_foot", t, sole_world=cands["foot"].get("left_foot"))
            contacts.commit("right_foot", t, sole_world=cands["foot"].get("right_foot"))
            prev_local = {"l": self.ev.hand_local("l"), "r": self.ev.hand_local("r")}
            prev_sole = {"l": self.ev.site_positions(self.ev.sole_sites["l"]),
                         "r": self.ev.site_positions(self.ev.sole_sites["r"])}
            accepted.append(z)
            timings.append(time.perf_counter() - t0)

        trajectory = np.zeros((T, 43))
        trajectory[:, 0:36] = np.stack(accepted)
        trajectory[:, 36:43] = prediction[:, 36:43]
        return TrajectoryResult(True, "accepted", trajectory=trajectory, diagnostics={
            "frame_status": frame_status, "iterations": adopted_iterations,
            "qp_calls": adopted_qp,
            "adopted_iterations_sum": int(sum(adopted_iterations)),
            "adopted_qp_calls_sum": int(sum(adopted_qp)),
            "timings_s": timings, "fallbacks": fallbacks,
            "slack_frames": int(sum(slack_flags)),
            "skipped": sum(1 for s in frame_status if s == "skipped"),
            "timing_stats": self._timing_stats(timings),
            **self._aggregate(attempts, visits, T,
                              sum(1 for s in frame_status if s == "skipped"),
                              backtracks, pass_index),
        })

    def _transition_backtrack(self, res, ctx, z_prev) -> tuple:
        """Transition-aware step backtracking (P0.1 L0).

        The QP never sees the interpolated-third corner-cut check that
        ``frame_valid`` applies between z_prev and the candidate; when the
        solver's output fails *only* that check, shrink the step along the
        same tangent direction from z_prev and re-check. Pure solver
        improvement: no gate or tolerance changes.
        """
        if not res.accepted or z_prev is None:
            return res, None
        chk = self.check_result(res, ctx, "prediction")
        if chk.ok or chk.reason != "transition_interpolated":
            return res, None
        self.carry_discovered(chk)
        xi = tangent_error(res.z, z_prev)
        for alpha in (0.75, 0.5, 0.25):
            cand = retract_batch(self.ev.model, z_prev[None], (alpha * xi)[None])[0]
            chk_bt = self._frame_valid(cand, ctx)
            self._log_check(chk_bt, ctx.t, f"backtrack_{alpha}", res)
            if chk_bt.ok:
                res.z = cand
                res.frame_check = chk_bt
                return res, "repaired_bt"
            self.carry_discovered(chk_bt)
        return res, None

    def _first_failure_frame(self, report) -> int | None:
        """Earliest structured failure frame across FAILED checks (P0-fix F3).

        Passing checks never contribute. A failing check without a
        structured frame leaves the result None, and the caller falls back
        to a FULL-segment retry -- retrying only the tail would silently
        keep whatever violation the missing location hides. Check-dict
        insertion order must not influence the answer.
        """
        frames = [c.first_failure_frame for c in report.checks.values()
                  if not c.passed and c.first_failure_frame is not None]
        return min(frames) if frames else None
