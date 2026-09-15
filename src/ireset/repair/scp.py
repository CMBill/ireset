"""SCP frame solver (plan section 8.3): OSQP linearized subproblems.

Each iteration re-linearizes residuals and constraints at the current state,
solves the damped QP in the 35-dim tangent with an L-inf trust region
(``|D^-1 xi| <= trust``), retracts, and accepts by nonlinear merit
(objective + normalized constraint violation). Infeasible QPs trigger a
slack-penalized restoration solve; restored steps are intermediate only —
a frame is accepted solely from its **nonlinear** feasibility (plan 8.3:
"中间状态不输出").

Simplification (documented): the QP is rebuilt per iteration rather than
warm-starting a persistent fixed-sparsity workspace; at this size (35 vars,
~300 rows) MuJoCo geometry dominates runtime, so reuse is a later
optimization, not a correctness item.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as spa

from ireset.kinematics.mujoco_adapter import N_TANGENT, retract_batch
from ireset.repair.constraints import (
    FrameEvaluator,
    KPT_NAMES,
    N_FOOT_ROWS,
    N_FOOT_SPEED,
    N_HAND_ROWS,
    N_HAND_SPEED,
    HAND_SPEED_OFF,
    N_ROWS,
    MAX_PAIRS,
    step_scale_matrix,
    tangent_error,
)

INF = 1e9


def cfg_noncontact(cfg) -> float:
    """Acceptance-mode lower bound for NON-contact pairs.

    Validator gate is -2mm; frame checks carry a 0.2mm safety margin so the
    validator's own interpolated probes do not graze past the gate between
    two accepted states.
    """
    return -0.0018


@dataclass
class FrameContext:
    """Per-frame solve context (built by the pipeline)."""

    t: int
    box_pose: np.ndarray                  # [7]
    z_pred: np.ndarray                    # [36] prediction to repair
    z_prev: np.ndarray | None             # [36] accepted previous frame
    ref_kpts_world: np.ndarray            # [15, 3] G1-correspondence order
    mesh_L: object | None                 # sparse [N, N] reference Laplacian
    delta_ref: np.ndarray | None          # [N, 3]
    mesh_fixed: np.ndarray | None         # [N-15, 3] object+ground pts, geom frame
    hand_anchor: dict[str, np.ndarray | None]   # per side [3,3] geom frame or None
    foot_anchor: dict[str, np.ndarray | None]   # per side [5,3] world or None
    prev_hand_local: dict[str, np.ndarray | None]
    prev_sole_world: dict[str, np.ndarray | None]
    #: previous frame's INPUT box pose [7]; None at t=0. The transition
    #: check interpolates full scenes (robot + box) between z_prev/box_prev
    #: and the candidate/current box -- box sweep motion between frames
    #: must be visible to the sequential pre-check (P0-fix F2, 2026-09-14)
    box_pose_prev: np.ndarray | None = None
    #: reference-schedule hold state per effector ("left_hand": True ...)
    #: at this frame; drives the ACTIVE hand-contact rows of regeneration
    #: mode (Path B) -- empty dict keeps standard per-pair semantics
    ref_hold: dict = field(default_factory=dict)
    #: min hand-geom/box distance at the previous ACCEPTED state (per side),
    #: for the regeneration approach corridor; missing side -> hard gap
    prev_hand_dist: dict = field(default_factory=dict)
    #: reference HUMAN hand centroid per side (world, [3]) at this frame --
    #: target of the regeneration hand-tracking objective
    ref_hand_kpts: dict = field(default_factory=dict)
    #: site motion (m) between z_prev and z_pred for this frame -- scales
    #: the solver's interpolation-dip margin (dip grows ~ quadratically
    #: with the per-frame step)
    frame_motion_m: float = 0.0


@dataclass
class FrameResult:
    accepted: bool
    z: np.ndarray
    iterations: int = 0
    skipped: bool = False
    slack_used: bool = False
    status: str = "solved"           # solved / infeasible / max_iters
    merit: float = float("inf")
    diagnostics: dict = field(default_factory=dict)


def _unit(x: np.ndarray, D: np.ndarray) -> np.ndarray:
    return x / D


def solver_band_margin(ctx: FrameContext, cfg) -> float:
    """Solver-grade band-row margin covering between-frame interpolation
    dips. Linear joint interpolation between two states with feet at the
    band floor sinks below both endpoints by ~ coef * step^2 (measured:
    0.8 mm at a 7.9 cm step, 5.6 mm at 16.6 cm). Capped, floored at the
    base 0.5 mm; accept-mode checks and the validator do NOT use it."""
    return float(min(0.006, max(0.0005, cfg.dip_margin_coef * ctx.frame_motion_m ** 2)))


class FrameSolver:
    def __init__(self, evaluator: FrameEvaluator, cfg) -> None:
        self.ev = evaluator
        self.cfg = cfg
        self.D = step_scale_matrix(cfg)
        self.Dinv2 = 1.0 / self.D**2
        self.dt = 1.0 / cfg.fps
        self._osqp = osqp.OSQP()
        self._frame_pairs: list | None = None

    # -- nonlinear evaluation ------------------------------------------------

    def evaluate(self, z: np.ndarray, ctx: FrameContext) -> dict:
        """Residuals + constraint values at z (nonlinear, MuJoCo-exact)."""
        ev, cfg = self.ev, self.cfg
        ev.forward(z, ctx.box_pose)
        out: dict = {}
        out["r_pred"] = _unit(tangent_error(z, ctx.z_pred), self.D)
        if ctx.z_prev is not None:
            out["r_temp"] = _unit(tangent_error(z, ctx.z_prev), self.D)
        else:
            out["r_temp"] = np.zeros(N_TANGENT)
        # mesh residual
        if ctx.mesh_L is not None:
            R_g, x_g = ev.box_frame()
            kpt_w = ev.site_positions(ev.kpt_ids)
            kpt_g = (R_g.T @ (kpt_w - x_g).T).T
            X = np.vstack([kpt_g, ctx.mesh_fixed])
            out["r_mesh"] = (ctx.mesh_L @ X - ctx.delta_ref).reshape(-1)
        else:
            out["r_mesh"] = None
        # task keypoint error (world)
        kpt_w = ev.site_positions(ev.kpt_ids)
        out["mpjpe"] = float(np.linalg.norm(kpt_w - ctx.ref_kpts_world, axis=-1).mean())
        # regeneration hand tracking: hand-site centroid vs the reference
        # human hand centroid (quasi-static in the box frame during grasp)
        if cfg.regen_mode and ctx.ref_hand_kpts:
            r_hand = []
            for side in ("l", "r"):
                ref = ctx.ref_hand_kpts.get(side)
                if ref is not None:
                    r_hand.append((ev.site_positions(ev.hand_sites[side]).mean(0) - ref)
                                  / cfg.regen_hand_track_scale_m)
            out["r_hand"] = (np.concatenate(r_hand) if r_hand
                             else np.zeros(0))
        else:
            out["r_hand"] = None
        return out

    def constraint_rows(
        self, z: np.ndarray, ctx: FrameContext, trust: float, mode: str = "repair",
        band_margin_m: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nonlinear constraint values and bounds [l, u] (rows as in constraints.py).

        ``mode="accept"`` uses validator-grade bounds (penetration tolerated to
        -2mm everywhere) for skip decisions; ``mode="repair"`` keeps the
        proactive +2mm non-contact clearance the solver enforces.
        """
        ev, cfg = self.ev, self.cfg
        # guarantee the evaluator is at z (callers usually forwarded already)
        ev.forward(z, ctx.box_pose)
        l = np.full(N_ROWS, -INF)
        u = np.full(N_ROWS, INF)
        val = np.zeros(N_ROWS)
        # per-row physical scales for violation normalization: a 2mm gate
        # violation must outweigh a 5cm tracking step, so one-sided rows
        # normalize by their own tolerance, not by a meter
        scale = np.ones(N_ROWS)
        scale[0:29] = np.minimum(ev.spec.joint_range[:, 1] - ev.spec.joint_range[:, 0], 0.02)

        val[0:29] = z[7:36]
        l[0:29] = ev.spec.joint_range[:, 0]
        u[0:29] = ev.spec.joint_range[:, 1]

        if ctx.z_prev is not None:
            dv = z[7:36] - ctx.z_prev[7:36]
            val[29:58] = dv
            lim = self.dt * cfg.joint_velocity_limit_radps
            l[29:58], u[29:58] = -lim, lim
            scale[29:58] = lim
            # root linear/angular velocity as NORM rows, matching the final
            # validator: per-component box rows admitted diagonal steps up to
            # sqrt(3)x the limit (fixed 2026-09-13, P0); slots 59,60,62,63
            # stay inactive to preserve the row layout
            val[58] = float(np.linalg.norm(z[0:3] - ctx.z_prev[0:3]))
            lim_t = self.dt * cfg.base_lin_velocity_limit_mps
            l[58], u[58] = -INF, lim_t
            scale[58] = lim_t
            from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz, so3_log_np

            R = quat_to_mat_wxyz(z[3:7][None])[0]
            Rp = quat_to_mat_wxyz(ctx.z_prev[3:7][None])[0]
            r_rot = so3_log_np((R @ Rp.T)[None])[0]
            val[61] = float(np.linalg.norm(r_rot))
            lim_r = self.dt * cfg.base_ang_velocity_limit_radps
            l[61], u[61] = -INF, lim_r
            scale[61] = lim_r

        val[64:99] = np.zeros(35)
        l[64:99] = -trust
        u[64:99] = trust

        # collision: active pairs recomputed at this state
        rows = self._collision_rows(ctx, band_margin_m)
        self._last_rows = rows
        for k, (dist, _J, lo, hi) in enumerate(rows):
            val[99 + k] = dist
            if mode == "accept" and hi > INF / 2:
                # non-contact rows relax to the validator penetration gate;
                # contact-band rows keep their band
                l[99 + k], u[99 + k] = cfg_noncontact(self.cfg), hi
            else:
                l[99 + k], u[99 + k] = lo, hi
            scale[99 + k] = max(min(hi - l[99 + k], self.cfg.penetration_scale_m), 1e-4)

        # hand anchors
        for si, side in enumerate(("l", "r")):
            anchor = ctx.hand_anchor.get(side)
            if anchor is None:
                continue
            local = ev.hand_local(side)
            base = 99 + MAX_PAIRS + si * 9
            val[base : base + 9] = local.reshape(-1)
            l[base : base + 9] = (anchor - cfg.hand_anchor_tol_m).reshape(-1)
            u[base : base + 9] = (anchor + cfg.hand_anchor_tol_m).reshape(-1)
            scale[base : base + 9] = cfg.hand_anchor_tol_m

        # foot anchors + support speed (per site)
        for si, side in enumerate(("l", "r")):
            anchor = ctx.foot_anchor.get(side)
            ids = ev.sole_sites[side]
            pts = ev.site_positions(ids)
            base = 99 + MAX_PAIRS + N_HAND_ROWS + si * 15
            if anchor is not None:
                val[base : base + 15] = pts.reshape(-1)
                l[base : base + 15] = (anchor - cfg.foot_anchor_tol_m).reshape(-1)
                u[base : base + 15] = (anchor + cfg.foot_anchor_tol_m).reshape(-1)
                scale[base : base + 15] = cfg.foot_anchor_tol_m
            base_s = 99 + MAX_PAIRS + N_HAND_ROWS + N_FOOT_ROWS + si * 15
            prev = ctx.prev_sole_world.get(side)
            if anchor is not None and prev is not None:
                val[base_s : base_s + 15] = (pts - prev).reshape(-1)
                lim_f = self.dt * cfg.foot_speed_mps
                l[base_s : base_s + 15] = -lim_f
                u[base_s : base_s + 15] = lim_f
                scale[base_s : base_s + 15] = lim_f

        # regen-mode hand box-frame speed rows: once a hand is IN the
        # contact band, its per-frame box-frame motion is hard-bounded
        # (the per-frame QP otherwise chatters a few mm inside the band
        # slack and the validator's quasi-static entry gate sees 2-3x the
        # reference hand speed). Bounds: +-dt * regen_hand_speed_mps,
        # measured exactly like the validator (hand_local vs previous
        # accepted hand_local). Approach frames (outside the band) are
        # exempt -- the corridor needs mm-scale closing steps.
        if cfg.regen_mode and cfg.regen_hand_speed_mps > 0.0:
            lim_h = self.dt * cfg.regen_hand_speed_mps
            for si, side in enumerate(("l", "r")):
                prev_local = ctx.prev_hand_local.get(side)
                prev_dist = ctx.prev_hand_dist.get(side)
                if (prev_local is None or prev_dist is None
                        or prev_dist > 0.006):
                    continue
                base = HAND_SPEED_OFF + si * 9
                val[base : base + 9] = (ev.hand_local(side) - prev_local).reshape(-1)
                l[base : base + 9] = -lim_h
                u[base : base + 9] = lim_h
                scale[base : base + 9] = lim_h
        return val, l, u, scale

    def _collision_rows(self, ctx: FrameContext,
                        band_margin_m: float | None = None) -> list[tuple[float, np.ndarray, float, float]]:
        """Active collision rows with contact-band classification.

        Hand geoms during hold and sole geoms during support are CONTACT
        pairs (band [-2mm, upper]); everything else keeps the positive
        non-contact clearance. A support foot touching the ground is the
        intended state, not a violation. ``band_margin_m`` overrides the
        regeneration interior margin (None -> cfg default; 0.0 gives the
        validator-grade floor for transition probes).

        The candidate pair SET is scanned once per frame and reused across
        SCP iterations (distances/Jacobians are recomputed each call at the
        current state); pairs entering range mid-frame are caught by the
        final validator and the suffix rollback.
        """
        ev, cfg = self.ev, self.cfg
        if self._frame_pairs is None:
            cands = []
            dmax = cfg.activation_margin_m + 0.01
            for ga, gb, kind in ev.pairs:
                if kind == 'box_ground':
                    # input-owned: the box trajectory is fixed (and its hull
                    # over-approximates the scanned bin bottom); not a
                    # repair constraint (matches the final validator scope)
                    continue
                dist, ft = ev.geom_distance(ga, gb, distmax=dmax)
                if dist < cfg.activation_margin_m:
                    cands.append((ga, gb, kind))
            self._frame_pairs = cands
        rows = []
        scored = []
        for ga, gb, kind in self._frame_pairs:
            dist, ft = ev.geom_distance(ga, gb, distmax=cfg.activation_margin_m + 0.01)
            scored.append((dist, ga, gb, kind, ft))
        scored.sort(key=lambda c: c[0])
        # Path B regeneration: during a reference grasp interval the
        # CLOSEST hand geom of that side carries an ACTIVE upper bound
        # instead of the geometric band classification -- the hovering
        # warm start (gap p95 33mm) must be pulled onto the box, not
        # merely allowed to touch. Only the nearest geom is bound
        # (contact := the closest surface touches, matching the
        # validator's min-over-geoms metric; binding every hand sphere
        # would demand the whole hand lie flat on the box). The row
        # bypasses the pair cap; pairs farther than the activation margin
        # are simply not candidates yet, and the selection re-runs at
        # every linearization.
        regen_active: dict[tuple[int, int], tuple[float, object, float]] = {}
        if cfg.regen_mode:
            for side, name in (("l", "left_hand"), ("r", "right_hand")):
                if not ctx.ref_hold.get(name):
                    continue
                geoms = set(ev._hand_geoms[side])
                best = None
                for dist, ga, gb, kind, ft in scored:
                    if kind == "robot_box" and ga in geoms and (best is None or dist < best[0]):
                        best = (dist, ga, gb, ft)
                if best is None:
                    continue
                # approach corridor: the bound must be reachable within one
                # frame from the previous accepted distance; once inside the
                # gap it stays at the gap (max of the two). The interpolation
                # dip margin is subtracted here with the SAME
                # ``solver_band_margin`` formula the solver band rows use
                # (F1, 2026-09-15): solutions riding the corridor sit at the
                # bound, and between-frame interpolated states dip below the
                # endpoints by ~ that margin, so the endpooint must close the
                # corridor by an extra margin for the probes to agree with
                # the rows the QP actually solved.
                prev = ctx.prev_hand_dist.get(side)
                margin = solver_band_margin(ctx, cfg)
                upper = cfg.regen_hand_contact_gap_m if prev is None else max(
                    cfg.regen_hand_contact_gap_m,
                    prev - cfg.regen_closing_rate_mps * self.dt - margin)
                regen_active[(best[1], best[2])] = (best[0], best[3], upper)        # regen mode keeps an interior margin above the -2 mm validator
        # floor on every band row: solutions sit on constraint boundaries
        # by construction, and boundary-sitting states would graze past
        # the gate at the transition probes' interpolated states. Probes
        # override this to the validator grade via ``band_margin_m=0``.
        band_lo = -cfg.contact_band_pen_m + (
            band_margin_m if band_margin_m is not None
            else (0.0005 if cfg.regen_mode else 0.0))
        added_regens: set = set()
        for (ga, gb), (dist, ft, upper) in regen_active.items():
            J = ev.distance_jacobian(ga, gb, ft, dist)
            rows.append((dist, J, band_lo, upper))
            added_regens.add((ga, gb))
        # regen rows count against the same fixed budget (99 + MAX_PAIRS):
        # the capped loop gives up that many slots so the total never
        # exceeds the QP row layout
        for dist, ga, gb, kind, ft in scored[: cfg.max_active_pairs - len(added_regens)]:
            if (ga, gb) in added_regens:
                continue
            J = ev.distance_jacobian(ga, gb, ft, dist)
            # P0.1 L1: contact bands classify PER PAIR by geometry alone -- a
            # pair closer than the band upper is a contact pair (touching
            # allowed; [2mm, band_hi] satisfies both classifications, so
            # boundary flips are harmless), farther pairs keep the noncontact
            # clearance. Freeze demands come ONLY from locked anchors: this is
            # the "touching-but-not-anchored" state, covering hovering
            # reference grasps and pitched support feet without raising any
            # threshold, and matching the validator's closest-point semantics.
            if kind == "robot_box" and dist <= cfg.contact_band_gap_m:
                rows.append((dist, J, band_lo, cfg.contact_band_gap_m))
            elif kind == "robot_ground" and dist <= cfg.foot_contact_gap_m:
                rows.append((dist, J, band_lo, cfg.foot_contact_gap_m))
            else:
                rows.append((dist, J, cfg.noncontact_clearance_m, INF))
        return rows

    def violation(self, val: np.ndarray, l: np.ndarray, u: np.ndarray,
                  scale: np.ndarray | None = None) -> float:
        """Sum of constraint violations normalized per-row by gate size."""
        if scale is None:
            scale = np.clip(u - l, 1e-3, 1.0)
        viol = np.maximum(np.maximum(l - val, val - u), 0.0) / scale
        finite = (np.abs(l) < INF / 2) | (np.abs(u) < INF / 2)
        return float(viol[finite].sum())

    def objective(self, res: dict) -> float:
        cfg = self.cfg
        f = cfg.prediction_weight * float((res["r_pred"] ** 2).sum())
        f += cfg.temporal_weight * float((res["r_temp"] ** 2).sum())
        if res["r_mesh"] is not None:
            f += cfg.laplacian_weight * float((res["r_mesh"] ** 2).sum())
        if res.get("r_hand") is not None and len(res["r_hand"]):
            f += cfg.regen_hand_track_weight * float((res["r_hand"] ** 2).sum())
        return f

    # -- linearization + QP ----------------------------------------------------

    def build_qp(self, z: np.ndarray, ctx: FrameContext, trust: float, slack: bool = False):
        """Assemble the QP at state z (caller must have forwarded to z)."""
        ev, cfg = self.ev, self.cfg

        # objective blocks
        r_pred = _unit(tangent_error(z, ctx.z_pred), self.D)
        r_temp = _unit(tangent_error(z, ctx.z_prev), self.D) if ctx.z_prev is not None else np.zeros(N_TANGENT)
        P = np.zeros((N_TANGENT, N_TANGENT))
        q = np.zeros(N_TANGENT)
        P += 2.0 * cfg.prediction_weight * np.diag(self.Dinv2)
        q += 2.0 * cfg.prediction_weight * self.Dinv2 * r_pred
        P += 2.0 * cfg.temporal_weight * np.diag(self.Dinv2)
        q += 2.0 * cfg.temporal_weight * self.Dinv2 * r_temp
        if ctx.mesh_L is not None:
            R_g, x_g = ev.box_frame()
            kpt_w = ev.site_positions(ev.kpt_ids)
            kpt_g = (R_g.T @ (kpt_w - x_g).T).T
            X = np.vstack([kpt_g, ctx.mesh_fixed])
            r_mesh = (ctx.mesh_L @ X - ctx.delta_ref).reshape(-1)
            J_kpt = np.einsum("ij,sak->sik", R_g.T, ev.site_jacobians_35(ev.kpt_ids))
            J_mesh = self._mesh_jacobian(ctx.mesh_L, J_kpt)
            P += 2.0 * cfg.laplacian_weight * (J_mesh.T @ J_mesh)
            q += 2.0 * cfg.laplacian_weight * (J_mesh.T @ r_mesh)
        P += 2.0 * cfg.regularization_weight * np.diag(self.Dinv2)

        # regeneration hand tracking (Gauss-Newton block)
        if cfg.regen_mode and ctx.ref_hand_kpts:
            J_hand_rows = []
            for side in ("l", "r"):
                if ctx.ref_hand_kpts.get(side) is not None:
                    J = ev.site_jacobians_35(ev.hand_sites[side]).mean(0)  # [3,35]
                    J_hand_rows.append(J / cfg.regen_hand_track_scale_m)
            if J_hand_rows:
                J_hand = np.vstack(J_hand_rows)
                r_hand = []
                for side in ("l", "r"):
                    ref = ctx.ref_hand_kpts.get(side)
                    if ref is not None:
                        r_hand.append(ev.site_positions(ev.hand_sites[side]).mean(0) - ref)
                r_hand = np.concatenate(r_hand) / cfg.regen_hand_track_scale_m
                P += 2.0 * cfg.regen_hand_track_weight * (J_hand.T @ J_hand)
                q += 2.0 * cfg.regen_hand_track_weight * (J_hand.T @ r_hand)

        # constraint rows
        A_rows = []
        l_rows = []
        u_rows = []

        I35 = np.eye(N_TANGENT)
        # joint limits (linearized about z: q + dq within [lo, hi])
        qz = z[7:36]
        Sel_q = I35[6:35]
        A_rows.append(Sel_q)
        l_rows.append(ev.spec.joint_range[:, 0] - qz)
        u_rows.append(ev.spec.joint_range[:, 1] - qz)

        if ctx.z_prev is not None:
            dv = qz - ctx.z_prev[7:36]
            A_rows.append(Sel_q)
            lim = self.dt * cfg.joint_velocity_limit_radps
            l_rows.append(-lim - dv); u_rows.append(lim - dv)

            # root velocity as a linearized NORM row (slot 0 of the 3-slot
            # block; slots 1,2 stay inactive): d . (dp + xi_p) <= lim at the
            # current direction d = dp/|dp|; relinearized every SCP iteration.
            # Below half the limit the row is dropped (satisfied, and the
            # direction is meaningless near zero).
            Sel_p = I35[0:3]
            dp_prev = z[0:3] - ctx.z_prev[0:3]
            lim_t = self.dt * cfg.base_lin_velocity_limit_mps
            lin_A = np.zeros((3, N_TANGENT))
            lin_lo = np.full(3, -INF)
            lin_hi = np.full(3, INF)
            n_dp = float(np.linalg.norm(dp_prev))
            if n_dp > 0.5 * lim_t and n_dp > 1e-9:
                lin_A[0] = (dp_prev / n_dp) @ Sel_p
                lin_hi[0] = lim_t - n_dp
            A_rows.append(lin_A)
            l_rows.append(lin_lo); u_rows.append(lin_hi)

            from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz, so3_log_np

            R = quat_to_mat_wxyz(z[3:7][None])[0]
            Rp = quat_to_mat_wxyz(ctx.z_prev[3:7][None])[0]
            r_rot = so3_log_np((R @ Rp.T)[None])[0]
            lim_r = self.dt * cfg.base_ang_velocity_limit_radps
            rot_A = np.zeros((3, N_TANGENT))
            rot_lo = np.full(3, -INF)
            rot_hi = np.full(3, INF)
            n_rot = float(np.linalg.norm(r_rot))
            if n_rot > 0.5 * lim_r and n_rot > 1e-9:
                rot_A[0] = (r_rot / n_rot) @ I35[3:6]
                rot_hi[0] = lim_r - n_rot
            A_rows.append(rot_A)
            l_rows.append(rot_lo); u_rows.append(rot_hi)

        # trust region (rows tracked for the slack assembly)
        Dm = np.diag(1.0 / self.D)
        trust_row_ids = list(range(sum(r.shape[0] for r in A_rows),
                                   sum(r.shape[0] for r in A_rows) + 35))
        A_rows.append(Dm)
        l_rows.append(np.full(35, -trust)); u_rows.append(np.full(35, trust))

        # collision rows for the SOLVER always carry the proactive interior
        # margin (mirrors the +2mm noncontact margin): per-frame solutions
        # sit on constraint boundaries by construction, and between-frame
        # interpolation dips past them -- the honest slack relaxation
        # (s >= 0) removed the old over-tightening that accidentally
        # provided this margin. Rebuilt here rather than reusing the
        # accept-mode rows so the margin is solver-grade in both standard
        # and regen modes.
        for dist, J, lo, hi in self._collision_rows(ctx, solver_band_margin(ctx, cfg)):
            A_rows.append(J)
            l_rows.append(np.array([lo - dist])); u_rows.append(np.array([hi - dist]) if hi < INF / 2 else np.array([INF]))

        # hand anchors
        for side in ("l", "r"):
            anchor = ctx.hand_anchor.get(side)
            if anchor is None:
                A_rows.append(np.zeros((9, N_TANGENT)))
                l_rows.append(np.full(9, -INF)); u_rows.append(np.full(9, INF))
                continue
            local = ev.hand_local(side)
            J = ev.hand_local_jac(side).reshape(9, N_TANGENT)
            A_rows.append(J)
            l_rows.append((anchor - cfg.hand_anchor_tol_m - local).reshape(-1))
            u_rows.append((anchor + cfg.hand_anchor_tol_m - local).reshape(-1))

        # foot anchors + speed
        for side in ("l", "r"):
            anchor = ctx.foot_anchor.get(side)
            ids = ev.sole_sites[side]
            pts = ev.site_positions(ids)
            J = ev.site_jacobians_35(ids).reshape(15, N_TANGENT)
            if anchor is None:
                A_rows.append(np.zeros((15, N_TANGENT)))
                l_rows.append(np.full(15, -INF)); u_rows.append(np.full(15, INF))
            else:
                A_rows.append(J)
                l_rows.append((anchor - cfg.foot_anchor_tol_m - pts).reshape(-1))
                u_rows.append((anchor + cfg.foot_anchor_tol_m - pts).reshape(-1))
            prev = ctx.prev_sole_world.get(side)
            if anchor is not None and prev is not None:
                lim_f = self.dt * cfg.foot_speed_mps
                A_rows.append(J)
                l_rows.append((-lim_f - (pts - prev)).reshape(-1))
                u_rows.append((lim_f - (pts - prev)).reshape(-1))
            else:
                A_rows.append(np.zeros((15, N_TANGENT)))
                l_rows.append(np.full(15, -INF)); u_rows.append(np.full(15, INF))

        # regen-mode hand box-frame speed rows (mirror of constraint_rows)
        if cfg.regen_mode and cfg.regen_hand_speed_mps > 0.0:
            lim_h = self.dt * cfg.regen_hand_speed_mps
            for side in ("l", "r"):
                prev_local = ctx.prev_hand_local.get(side)
                prev_dist = ctx.prev_hand_dist.get(side)
                Jh = ev.hand_local_jac(side).reshape(9, N_TANGENT)
                if (prev_local is None or prev_dist is None
                        or prev_dist > 0.006):
                    A_rows.append(np.zeros((9, N_TANGENT)))
                    l_rows.append(np.full(9, -INF)); u_rows.append(np.full(9, INF))
                else:
                    cur = ev.hand_local(side)
                    A_rows.append(Jh)
                    l_rows.append((-lim_h - (cur - prev_local)).reshape(-1))
                    u_rows.append((lim_h - (cur - prev_local)).reshape(-1))
        else:
            A_rows.append(np.zeros((N_HAND_SPEED, N_TANGENT)))
            l_rows.append(np.full(N_HAND_SPEED, -INF)); u_rows.append(np.full(N_HAND_SPEED, INF))

        # pad to the fixed layout length with inactive rows (collision block
        # may have fewer than MAX_PAIRS active pairs)
        A = np.vstack(A_rows)
        lo = np.concatenate(l_rows)
        hi = np.concatenate(u_rows)
        # pad collision to fixed MAX_PAIRS block: rows already appended variable;
        # pad total to N_ROWS with inactive rows
        n_missing = N_ROWS - A.shape[0]
        if n_missing > 0:
            A = np.vstack([A, np.zeros((n_missing, N_TANGENT))])
            lo = np.concatenate([lo, np.full(n_missing, -INF)])
            hi = np.concatenate([hi, np.full(n_missing, INF)])
        elif n_missing < 0:
            raise RuntimeError("constraint rows exceeded fixed layout")

        if slack:
            # one slack s_i >= 0 per relaxable finite row: value - s <= u and
            # value + s >= l (duplicated row); trust-region rows stay hard.
            # The s >= 0 bound is EXPLICIT (identity rows): without it the
            # +penalty linear term drives s negative (over-tightening) and
            # the QP returns destructive steps on near-infeasible systems
            # (found via the Path B corridor, 2026-09-15; slack sum was
            # -51 at a frame missing its bound by 0.9 mm).
            finite = (lo > -INF / 2) | (hi < INF / 2)
            for i in trust_row_ids:  # trust region is never relaxed
                finite[i] = False
            idx = np.where(finite)[0]
            n_s = len(idx)
            n_r = A.shape[0]
            A_new = np.zeros((n_r + 2 * n_s, N_TANGENT + n_s))
            A_new[:n_r, :N_TANGENT] = A
            for j, i in enumerate(idx):
                A_new[i, N_TANGENT + j] = -1.0
                A_new[n_r + j, :N_TANGENT] = A[i]
                A_new[n_r + j, N_TANGENT + j] = 1.0
                A_new[n_r + n_s + j, N_TANGENT + j] = 1.0      # s_j >= 0
            lo_new = np.concatenate([lo, np.full(n_s, -INF), np.zeros(n_s)])
            hi_new = np.concatenate([hi, np.full(n_s, INF), np.full(n_s, INF)])
            lo_new[idx] = -INF  # upper-side rows keep u, relaxed by -s
            lo_new[n_r:n_r + n_s] = lo[idx]  # duplicated rows enforce l + s
            P_new = np.zeros((N_TANGENT + n_s, N_TANGENT + n_s))
            P_new[:N_TANGENT, :N_TANGENT] = P
            q_new = np.concatenate([q, np.full(n_s, cfg.slack_penalty)])
            return (spa.csc_matrix(P_new), q_new, spa.csc_matrix(A_new), lo_new, hi_new), n_s

        return (spa.csc_matrix(P), q, spa.csc_matrix(A), lo, hi), 0

    def _mesh_jacobian(self, L, J_kpt) -> np.ndarray:
        """Mesh residual Jacobian: only the first 15 points vary.

        Residual rows are (point i, axis a); d r_{i,a} = sum_k L[i,k] * d x_k[a].
        With J_kpt [15, 3, 35] (geom frame), J_mesh = kron(L[:, :15], I3) @ J_kpt.
        """
        Lk = np.asarray(L[:, :15].todense()) if hasattr(L, "todense") else np.asarray(L)[:, :15]
        return np.kron(Lk, np.eye(3)) @ J_kpt.reshape(15 * 3, N_TANGENT)

    def solve_qp(self, P, q, A, lo, hi):
        prob = self._osqp
        lo = np.where(lo <= -INF / 2, -np.inf, lo)
        hi = np.where(hi >= INF / 2, np.inf, hi)
        prob.setup(P=P, q=q, A=A, l=lo, u=hi, verbose=False,
                   polish=False, eps_abs=3e-5, eps_rel=3e-5, max_iter=2500,
                   scaling=10)
        res = prob.solve()
        status = str(res.info.status)
        x = np.asarray(res.x) if res.x is not None else None
        # OSQP returns garbage (huge) x on infeasibility: never trust it
        if x is None or not np.isfinite(x).all() or np.abs(x).max() > 1e6 or "infeasible" in status:
            return None, status
        return x, status

    # -- SCP loop ------------------------------------------------------------

    def solve_frame(self, z_init: np.ndarray, ctx: FrameContext, force_slack: bool = False) -> FrameResult:
        cfg = self.cfg
        z = np.array(z_init, dtype=float)
        trust = cfg.trust_radius_init
        mu = 10.0 * max(1.0, cfg.prediction_weight + cfg.laplacian_weight + cfg.temporal_weight)
        if cfg.regen_mode:
            mu *= cfg.regen_merit_multiplier
        best_z, best_merit, best_feasible = None, float("inf"), False
        used_slack = False

        # once the tight QP proves infeasible for this frame, stay in
        # restoration mode (re-trying the tight solve each iteration doubles
        # the cost without new information)
        restoration = force_slack
        max_iters = cfg.scp_max_iters + (cfg.scp_extra_restoration_iters if force_slack else 0)
        self._frame_pairs = None  # fresh scan for this frame's first linearization
        for it in range(max_iters):
            res = self.evaluate(z, ctx)
            # feasibility/merit are judged against ACCEPTANCE bounds; the
            # proactive +2mm clearance lives only inside the QP rows
            val, lo, hi, scl = self.constraint_rows(z, ctx, trust, mode="accept")
            viol = self.violation(val, lo, hi, scl)
            merit = self.objective(res) + mu * viol
            feasible = viol <= 1e-6
            if feasible and merit < best_merit:
                best_z, best_merit, best_feasible = z.copy(), merit, True
            if feasible and res["mpjpe"] <= cfg.task_mpjpe_tol_m and self._stationary(z, ctx):
                return FrameResult(True, z, it, merit=merit)

            if restoration:
                (Ps, qs, As, los, his), _ = self.build_qp(z, ctx, trust, slack=True)
                x_s, _ = self.solve_qp(Ps, qs, As, los, his)
                if x_s is None:
                    trust *= cfg.shrink_factor
                    if trust < cfg.trust_radius_min:
                        break
                    continue
                used_slack = True
                xi = x_s[:N_TANGENT]
            else:
                (P, qv, A, lo_a, hi_a), _ = self.build_qp(z, ctx, trust, slack=False)
                xi, _status = self.solve_qp(P, qv, A, lo_a, hi_a)
                if xi is None:
                    restoration = True
                    continue
            xi = np.clip(xi, -trust * self.D, trust * self.D)

            # backtracking line search: deep-penetration linearizations can
            # mispredict, and a full step may increase the true violation
            accepted_step = False
            for step_scale in (1.0, 0.5, 0.25, 0.125):
                cand = retract_batch(self.ev.model, z[None], (step_scale * xi)[None])[0]
                res_c = self.evaluate(cand, ctx)
                val_c, lo_c, hi_c, scl_c = self.constraint_rows(cand, ctx, trust, mode="accept")
                merit_c = self.objective(res_c) + mu * self.violation(val_c, lo_c, hi_c, scl_c)
                if merit_c < merit:
                    z = cand
                    accepted_step = True
                    if merit_c < 0.5 * merit:
                        trust = min(cfg.trust_radius_max, trust * cfg.expand_factor)
                    break
            if not accepted_step:
                trust *= cfg.shrink_factor
                if trust < cfg.trust_radius_min:
                    break

        if best_feasible:
            res = self.evaluate(best_z, ctx)
            if res["mpjpe"] <= cfg.task_mpjpe_tol_m:
                return FrameResult(True, best_z, cfg.scp_max_iters, slack_used=used_slack, merit=best_merit)
            return FrameResult(False, best_z, cfg.scp_max_iters, status="task_quality", merit=best_merit)
        return FrameResult(False, z, cfg.scp_max_iters, status="infeasible", slack_used=used_slack)

    def _stationary(self, z: np.ndarray, ctx: FrameContext) -> bool:
        """Prediction-consistent stationarity: near the prediction target."""
        r = _unit(tangent_error(z, ctx.z_pred), self.D)
        return float(np.abs(r).max()) < 0.05
