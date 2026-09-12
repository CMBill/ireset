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

import re
import time
from dataclasses import dataclass, field

import numpy as np
import trimesh

from ireset.embodied.contact_sites import SMPLH_INDEX, infer_reference_schedule
from ireset.embodied.g1_spec import G1Spec, KPT_TO_SMPLH
from ireset.geometry.collision_pairs import build_collision_pairs
from ireset.geometry.distance_adapter import largebox_hull
from ireset.imesh.delaunay import build_frame_mesh, laplacian_coordinates
from ireset.geometry.distance_adapter import world_to_box_geom
from ireset.repair.constraints import FrameEvaluator
from ireset.repair.context import ContactStateMachine, ReferenceCondition, RepairConfig
from ireset.repair.scp import FrameContext, FrameSolver
from ireset.repair.validation import IndependentValidator

KPT_IDX = [SMPLH_INDEX[v] for v in KPT_TO_SMPLH.values()]
N_SURF = 12   # object surface samples for the interaction mesh
N_GROUND = 8  # world-fixed ground points


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

    # -- interaction mesh precompute -------------------------------------------

    def _mesh_assets(self, condition: ReferenceCondition) -> None:
        if self._surf_pts is not None:
            return
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

    def _establishment_candidates(self, z: np.ndarray, box: np.ndarray, contacts, t: int) -> dict:
        """Anchor candidates from an ACCEPTED state, gated by geometry (plan 8.1).

        Hands must be within the establishment distance of the box *without
        deep penetration*; feet must be near the ground. Called after a frame
        is accepted (skipped or repaired).
        """
        out = {"hand": {}, "foot": {}}
        self.ev.forward(z, box)
        for side, name in (("l", "left_hand"), ("r", "right_hand")):
            if contacts.holding[name] or not contacts.reference_wants_hold(name, t):
                continue
            dist = min(self.ev.geom_distance(g, self.ev.box_geom)[0] for g in self.ev._hand_geoms[side])
            if -self.cfg.contact_band_pen_m <= dist <= self.cfg.hand_establish_dist_m:
                out["hand"][name] = self.ev.hand_local(side)
        for side, name in (("l", "left_foot"), ("r", "right_foot")):
            if contacts.holding[name] or not contacts.reference_wants_hold(name, t):
                continue
            gap = min(self.ev.geom_distance(g, self.ev.ground_geom)[0] for g in self.ev.sole_geoms[side])
            if -self.cfg.contact_band_pen_m <= gap <= self.cfg.foot_establish_gap_m:
                out["foot"][name] = self.ev.site_positions(self.ev.sole_sites[side])
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
        return FrameContext(
            t=t, box_pose=prediction[t, 36:43].copy(),
            z_pred=prediction[t, 0:36].copy(), z_prev=z_prev,
            ref_kpts_world=condition.human_kpts_world[t][KPT_IDX],
            mesh_L=L, delta_ref=delta, mesh_fixed=fixed,
            hand_anchor=hand_anchor, foot_anchor=foot_anchor,
            prev_hand_local=prev_local, prev_sole_world=prev_sole,
        )

    # -- frame-level checks (skip logic, plan 8.4) ---------------------------------

    def frame_valid(self, z: np.ndarray, ctx: FrameContext) -> tuple[bool, str]:
        # fresh pair scan for acceptance decisions: the solver's per-frame
        # cache reflects the linearization state, not this candidate
        self.solver._frame_pairs = None
        val, lo, hi, scl = self.solver.constraint_rows(
            z, ctx, trust=self.cfg.trust_radius_max, mode="accept"
        )
        if self.solver.violation(val, lo, hi, scl) > 1e-6:
            return False, "state_constraints"
        # interpolated midpoint against the previous accepted state: the
        # final validator checks interpolated states, so the sequential loop
        # must not accept corner-cutting transitions (plan 6.3)
        if ctx.z_prev is not None:
            scene_a = np.zeros(43)
            scene_a[0:36] = ctx.z_prev
            scene_a[36:43] = ctx.box_pose
            scene_b = np.zeros(43)
            scene_b[0:36] = z
            scene_b[36:43] = ctx.box_pose
            for s_frac in (1 / 3, 0.5, 2 / 3):  # match the validator's subdivision
                mid = self.validator._interpolate(scene_a, scene_b, s_frac)
                val_m, lo_m, hi_m, scl_m = self.solver.constraint_rows(
                    mid[0:36], ctx, trust=self.cfg.trust_radius_max, mode="accept"
                )
                if self.solver.violation(val_m, lo_m, hi_m, scl_m) > 1e-6:
                    return False, "transition_interpolated"
        res = self.solver.evaluate(z, ctx)
        if res["mpjpe"] > self.cfg.task_mpjpe_tol_m:
            return False, "task_quality"
        return True, "ok"

    # -- public API ------------------------------------------------------------------

    def repair_trajectory(self, prediction: np.ndarray, condition: ReferenceCondition) -> TrajectoryResult:
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
            return result

        # suffix rollback: keep the validated prefix (states + anchors) by
        # replaying it as the prediction, re-solve only the suffix
        f_frame = self._first_failure_frame(final)
        start = max(0, (f_frame if f_frame is not None else prediction.shape[0] - 1)
                    - self.cfg.suffix_retry_window)
        retry_pred = np.vstack([result.trajectory[:start], prediction[start:]])
        retry = self._run_sequential(retry_pred, condition)
        retry.diagnostics["first_attempt_final"] = f"{final.summary()} @t={f_frame}"
        retry.diagnostics["first_failure_frame"] = f_frame
        retry.diagnostics["suffix_retry_from"] = start
        if not retry.accepted:
            retry.diagnostics["first_attempt"] = {
                k: v for k, v in result.diagnostics.items() if k != "trajectory"
            }
            return TrajectoryResult(False, "rejected_final", diagnostics=retry.diagnostics)
        final2 = self.validator.validate(retry.trajectory, exclude=("box_ground",))
        if final2.passed:
            retry.diagnostics["final_validation"] = "passed_after_suffix_retry"
            return retry
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
        timings: list[float] = []
        fallbacks = 0

        for t in range(T):
            t0 = time.perf_counter()
            z_pred = prediction[t, 0:36].copy()
            box = prediction[t, 36:43]
            z_prev = accepted[-1].copy() if accepted else None
            ctx = self._frame_ctx(t, prediction, condition, contacts, prev_local, prev_sole, z_prev)

            ok, why = self.frame_valid(z_pred, ctx)
            if ok:
                z = z_pred
                frame_status.append("skipped")
                iterations.append(0)
            else:
                res = self.solver.solve_frame(z_pred, ctx)
                if not self._frame_result_ok(res, ctx):
                    if self.cfg.enable_slack_fallback:
                        res2 = self.solver.solve_frame(
                            z_prev if z_prev is not None else z_pred, ctx, force_slack=True
                        )
                        if self._frame_result_ok(res2, ctx):
                            res = res2
                            fallbacks += 1
                if not self._frame_result_ok(res, ctx):
                    timings.append(time.perf_counter() - t0)
                    return TrajectoryResult(False, "rejected_frame", diagnostics={
                        "t_fail": t, "reason": res.status, "skip_reason": why,
                        "accepted_frames": len(accepted), "frame_status": frame_status,
                        "iterations": iterations, "timings_s": timings,
                        "fallbacks": fallbacks,
                        "skipped": sum(1 for st in frame_status if st == "skipped"),
                    })
                z = res.z
                frame_status.append("repaired" if not res.slack_used else "repaired_slack")
                iterations.append(res.iterations)

            # commit contact transitions from the accepted state only:
            # anchors lock from the geometry that just passed all checks
            cands = self._establishment_candidates(z, box, contacts, t)
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
            "timings_s": timings, "fallbacks": fallbacks,
            "skipped": sum(1 for s in frame_status if s == "skipped"),
        })

    def _frame_result_ok(self, res, ctx) -> bool:
        """A repaired state must itself pass the frame+transition checks."""
        if not res.accepted:
            return False
        ok, _why = self.frame_valid(res.z, ctx)
        return ok

    def _first_failure_frame(self, report) -> int | None:
        for chk in report.checks.values():
            m = re.search(r"t=(\d+)", chk.location)
            if not chk.passed and m:
                return int(m.group(1))
        return None
