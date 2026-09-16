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
from ireset.repair.scp import FrameContext, FrameSolver, solver_band_margin
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
        if self.cfg.regen_mode:
            z_ref = z_prev if z_prev is not None else prediction[t, 0:36]
            self.ev.forward(z_ref, prediction[t, 36:43])
            pose = prediction[t, 36:43]
            for side in ("l", "r"):
                prev_hand_dist[side] = min(
                    self.ev.geom_distance(g, self.ev.box_geom)[0]
                    for g in self.ev._hand_geoms[side])
                ids = [SMPLH_INDEX[n] for n in HAND_TIP_JOINTS["left" if side == "l" else "right"]]
                p = condition.human_kpts_world[t][ids].mean(0)
                ref_hand_kpts[side] = _surface_target(p, pose)
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
            frame_motion_m=(
                self.validator._frame_motion_m(
                    np.r_[z_prev, prediction[t - 1, 36:43]],
                    np.r_[prediction[t, 0:36], prediction[t, 36:43]])
                if z_prev is not None else 0.0),
        )

    # -- frame-level checks (skip logic, plan 8.4) ---------------------------------

    def _hand_dists_verbose(self, z: np.ndarray, box: np.ndarray) -> dict:
        self.ev.forward(z, box)
        return {s: min(self.ev.geom_distance(g, self.ev.box_geom)[0]
                       for g in self.ev._hand_geoms[s]) for s in ("l", "r")}

    def _violating_rows_verbose(self, val, lo, hi, n_rows: int) -> list[dict]:
        names = getattr(self.solver, "last_row_pairs", [])
        out = []
        for k in range(n_rows):
            v, l_, h_ = val[99 + k], lo[99 + k], hi[99 + k]
            if v < l_ - 1e-9 or v > h_ + 1e-9:
                out.append({"pair": names[k] if k < len(names) else "?",
                            "val_m": float(v), "lo_m": float(l_), "hi_m": float(h_)})
        return out

    def _carry_back_discovered_pairs(self, val, lo, hi, n_rows: int) -> list[str]:
        """Stage 1: feed pairs found violating at a check back to the solver.

        The candidate set a solve attempt used is frozen at its initial
        state; an endpoint/probe check can therefore reject on a pair the
        solve never saw. Pushing the (ga, gb, kind) identity of every
        violating row into ``note_discovered_pairs`` makes the NEXT
        linearization carry that pair regardless of its distance at the
        attempt start -- the loop can then converge on the violation it
        produced instead of failing on a blind spot. Returns pair names
        for logging.
        """
        names = getattr(self.solver, "last_row_pairs", [])
        ids = getattr(self.solver, "last_row_ids", [])
        carried = []
        discovered = []
        for k in range(n_rows):
            v, l_, h_ = val[99 + k], lo[99 + k], hi[99 + k]
            if v < l_ - 1e-9 or v > h_ + 1e-9:
                if k < len(ids):
                    discovered.append(ids[k])
                carried.append(names[k] if k < len(names) else "?")
        if discovered:
            self.solver.note_discovered_pairs(discovered)
        return carried

    def frame_valid(self, z: np.ndarray, ctx: FrameContext) -> tuple[bool, str]:
        # fresh pair scan for acceptance decisions: the solver's per-frame
        # cache reflects the linearization state, not this candidate
        self.solver._frame_pairs = None
        val, lo, hi, scl = self.solver.constraint_rows(
            z, ctx, trust=self.cfg.trust_radius_max, mode="accept"
        )
        ep_viol = self.solver.violation(val, lo, hi, scl)
        if self.verbose:
            # corridor snapshot is from THIS endpoint linearization
            # (constraint_rows -> _collision_rows refreshed it)
            cand_d = self._hand_dists_verbose(z, ctx.box_pose)
            prev_d = (self._hand_dists_verbose(
                ctx.z_prev,
                ctx.box_pose_prev if ctx.box_pose_prev is not None else ctx.box_pose)
                if ctx.z_prev is not None else {"l": float("nan"), "r": float("nan")})
            check = {"t": ctx.t, "kind": "check", "viol": float(ep_viol),
                     "hand_l_prev_m": float(prev_d["l"]), "hand_l_cand_m": float(cand_d["l"]),
                     "hand_r_prev_m": float(prev_d["r"]), "hand_r_cand_m": float(cand_d["r"]),
                     "corridor": {s: self.solver.last_corridor.get(s)
                                  for s in ("l", "r") if self.solver.last_corridor.get(s)},
                     "band_margin_m": solver_band_margin(ctx, self.cfg)}
            if ep_viol > 1e-6:
                check["rows"] = self._violating_rows_verbose(val, lo, hi, len(self.solver._last_rows))
                check["carried"] = self._carry_back_discovered_pairs(
                    val, lo, hi, len(self.solver._last_rows))
            self.frame_log.append(check)
        if ep_viol > 1e-6:
            self._carry_back_discovered_pairs(val, lo, hi, len(self.solver._last_rows))
            return False, "state_constraints"
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
            saved_pairs = self.solver._frame_pairs
            try:
                for i in range(1, n_sub):
                    mid = self.validator._interpolate(scene_a, scene_b, i / n_sub)
                    ctx_mid = replace(ctx, box_pose=mid[36:43].copy())
                    # rescan per probe: the endpoint candidate set cannot see
                    # pairs that come into range only at interior states;
                    # probes judge at the VALIDATOR floor (band_margin 0) --
                    # they approximate the final gate, which is the arbiter
                    self.solver._frame_pairs = None
                    val_m, lo_m, hi_m, scl_m = self.solver.constraint_rows(
                        mid[0:36], ctx_mid, trust=self.cfg.trust_radius_max,
                        mode="accept", band_margin_m=0.0
                    )
                    n_coll = 99 + len(self.solver._last_rows)
                    viol_m = self.solver.violation(val_m[99:n_coll], lo_m[99:n_coll],
                                                   hi_m[99:n_coll], scl_m[99:n_coll])
                    if self.verbose:
                        mid_d = self._hand_dists_verbose(mid[0:36], mid[36:43])
                        rec = {"t": ctx.t, "kind": "probe", "i": i, "n_sub": int(n_sub),
                               "viol": float(viol_m),
                               "hand_l_mid_m": float(mid_d["l"]),
                               "hand_r_mid_m": float(mid_d["r"]),
                               "motion_m": float(motion)}
                        if viol_m > 1e-6:
                            rec["rows"] = self._violating_rows_verbose(val_m, lo_m, hi_m, n_coll - 99)
                        self.frame_log.append(rec)
                    if viol_m > 1e-6:
                        # stage 1: carry the mid-state's violating pairs back
                        # even when verbose is off (convergence input, not
                        # just logging); ids/names are from THIS probe's rows
                        carried = self._carry_back_discovered_pairs(
                            val_m, lo_m, hi_m, n_coll - 99)
                        if self.verbose and carried:
                            rec["carried"] = carried
                        return False, "transition_interpolated"
            finally:
                self.solver._frame_pairs = saved_pairs
        res = self.solver.evaluate(z, ctx)
        if res["mpjpe"] > self.cfg.task_mpjpe_tol_m:
            return False, "task_quality"
        return True, "ok"

    # -- public API ------------------------------------------------------------------

    def repair_trajectory(self, prediction: np.ndarray, condition: ReferenceCondition) -> TrajectoryResult:
        """End-to-end repair; ``diagnostics["total_s"]`` covers both attempts
        plus all final validations (the timing callers must report)."""
        t0 = time.perf_counter()
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
        final = self.validator.validate(result.trajectory, exclude=("box_ground",))
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
        retry = self._run_sequential(retry_pred, condition)
        retry.diagnostics["first_attempt_final"] = f"{final.summary()} @t={f_frame}"
        retry.diagnostics["first_failure_frame"] = f_frame
        retry.diagnostics["suffix_retry_from"] = start
        retry.diagnostics["failed_checks"] = {
            k: {"location": v.location, "first_failure_frame": v.first_failure_frame}
            for k, v in final.checks.items() if not v.passed
        }
        if not retry.accepted:
            retry.diagnostics["first_attempt"] = {
                k: v for k, v in result.diagnostics.items() if k != "trajectory"
            }
            return TrajectoryResult(False, "rejected_final", diagnostics=retry.diagnostics)
        final2 = self.validator.validate(retry.trajectory, exclude=("box_ground",))
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

    def _run_sequential(self, prediction, condition) -> TrajectoryResult:
        T = prediction.shape[0]
        contacts = ContactStateMachine(condition.ref_schedule, T)
        accepted: list[np.ndarray] = []
        prev_local: dict = {}
        prev_sole: dict = {}
        frame_status: list[str] = []
        iterations: list[int] = []
        qp_calls_hist: list[int] = []
        timings: list[float] = []
        slack_flags: list[bool] = []
        fallbacks = 0

        for t in range(T):
            t0 = time.perf_counter()
            # stage 1: discovered-pair carry-over is a WITHIN-FRAME mechanism
            # (attempt N learns what attempt N-1's checks caught); a new
            # frame is a new state and starts from a clean candidate set
            self.solver.clear_discovered_pairs()
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

            # attempt-level diagnostics (stage 0, 2026-09-16): one record per
            # solver attempt with its state source and outcome, so reports can
            # be recomputed from per-sample logs (which state, which try)
            def _log_attempt(source: str, res_, bt_kind: str | None = None) -> None:
                if not self.verbose:
                    return
                self.frame_log.append({
                    "t": t, "kind": "attempt", "source": source,
                    "backtrack": bt_kind, "accepted": bool(res_.accepted)
                    if res_ is not None else None,
                    "status": getattr(res_, "status", None),
                    "iterations": getattr(res_, "iterations", None),
                    "qp_calls": getattr(res_, "qp_calls", None),
                    "slack_used": bool(getattr(res_, "slack_used", False)),
                    "merit": float(res_.merit) if res_ is not None
                    and np.isfinite(res_.merit) else None,
                    "outcome_frame_valid": self._frame_result_ok(res_, ctx)
                    if res_ is not None else None,
                })

            ok, why = self.frame_valid(z_pred, ctx)
            if self.verbose:
                self.frame_log.append({
                    "t": t, "kind": "skip_check", "source": "prediction",
                    "ok": bool(ok), "reason": why,
                })
            if ok:
                z = z_pred
                frame_status.append("skipped")
                iterations.append(0)
                slack_flags.append(False)
                qp_calls_hist.append(0)
            else:
                res = self.solver.solve_frame(z_pred, ctx)
                path = "repaired" if not res.slack_used else "repaired_slack"
                _log_attempt("prediction", res)
                res, bt = self._transition_backtrack(res, ctx, z_prev)
                if bt:
                    path = bt
                if not self._frame_result_ok(res, ctx):
                    # multi-start restart from the previous accepted state
                    # (P0.1 L0): deep-violation frames linearize badly from
                    # z_pred but solve from the accepted predecessor
                    res_alt = self.solver.solve_frame(z_prev, ctx) if z_prev is not None else None
                    if res_alt is not None:
                        _log_attempt("prev_accepted", res_alt)
                        res_alt, bt2 = self._transition_backtrack(res_alt, ctx, z_prev)
                        if self._frame_result_ok(res_alt, ctx):
                            res, path = res_alt, bt2 or "repaired_prev"
                    if not self._frame_result_ok(res, ctx) and self.cfg.enable_slack_fallback:
                        res2 = self.solver.solve_frame(
                            z_prev if z_prev is not None else z_pred, ctx, force_slack=True
                        )
                        _log_attempt("slack_fallback", res2)
                        if self._frame_result_ok(res2, ctx):
                            res, path = res2, "repaired_slack"
                            fallbacks += 1
                if not self._frame_result_ok(res, ctx):
                    timings.append(time.perf_counter() - t0)
                    return TrajectoryResult(False, "rejected_frame", diagnostics={
                        "t_fail": t, "reason": res.status, "skip_reason": why,
                        "accepted_frames": len(accepted), "frame_status": frame_status,
                        "iterations": iterations, "timings_s": timings,
                        "qp_calls": qp_calls_hist, "fallbacks": fallbacks,
                        "slack_frames": int(sum(slack_flags)),
                        "skipped": sum(1 for st in frame_status if st == "skipped"),
                    })
                z = res.z
                frame_status.append(path)
                iterations.append(res.iterations)
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
            "frame_status": frame_status, "iterations": iterations,
            "qp_calls": qp_calls_hist,
            "timings_s": timings, "fallbacks": fallbacks,
            "slack_frames": int(sum(slack_flags)),
            "skipped": sum(1 for s in frame_status if s == "skipped"),
        })

    def _frame_result_ok(self, res, ctx) -> bool:
        """A repaired state must itself pass the frame+transition checks."""
        if not res.accepted:
            return False
        ok, _why = self.frame_valid(res.z, ctx)
        return ok

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
        ok, why = self.frame_valid(res.z, ctx)
        if ok or why != "transition_interpolated":
            return res, None
        xi = tangent_error(res.z, z_prev)
        for alpha in (0.75, 0.5, 0.25):
            cand = retract_batch(self.ev.model, z_prev[None], (alpha * xi)[None])[0]
            if self.frame_valid(cand, ctx)[0]:
                res.z = cand
                return res, "repaired_bt"
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
