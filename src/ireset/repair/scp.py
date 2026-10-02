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

Correctness contracts added 2026-09-22 (m2_m3_stage_gate_fix_plan F1/F2):

- Pair **identity** ``(ga, gb)`` and row **role** (``regen`` vs regular
  collision) are separate: the geometric category always comes from the
  static registry, so a check-discovered pair can never lose its active
  contact semantics (or its band classification) through carry-back.
- Acceptance checks (``check_rows``) are NOT capped by the QP row budget:
  a successful return may not hide a violating pair behind
  ``max_active_pairs``. Every success exit runs the same review.
- The QP linear term must equal the true objective gradient: with
  ``r = e/D`` and physical-increment variables, ``q = 2 w J_r^T r`` where
  ``J_r = diag(1/D)`` for the additive (translation/joint) blocks and the
  world-frame SO(3) left-Jacobian inverse for the rotation block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import osqp
import scipy.sparse as spa

from ireset.embodied.g1_spec import HAND_SITES
from ireset.kinematics.mujoco_adapter import N_TANGENT, retract_batch
from ireset.kinematics.rotation_adapter import (
    quat_to_mat_wxyz,
    so3_left_jacobian_inverse_np,
    so3_left_jacobian_np,
    so3_log_np,
)
from ireset.repair.profiles import OsqpSettings
from ireset.repair.constraints import (
    FrameEvaluator,
    KPT_NAMES,
    N_FIXED,
    N_FOOT_ROWS,
    N_FOOT_SPEED,
    N_HAND_ROWS,
    N_HAND_SPEED,
    N_POST_ROWS,
    HAND_SPEED_OFF,
    N_ROWS,
    MAX_PAIRS,
    step_scale_matrix,
    tangent_error,
)

INF = 1e9
_AXES = ("x", "y", "z")


def _concat_rows(*blocks) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-block ``(val, lo, hi, scale)`` tuples."""
    return tuple(np.concatenate([b[i] for b in blocks]) for i in range(4))  # type: ignore[return-value]


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
    #: reference-driven box-frame hand motion per side ([3], BOX frame):
    #: the human hand centroid's displacement from the previous frame. The
    #: grasp-stability soft target (F4 section 7.2) penalizes the robot
    #: hand's box-frame displacement DEVIATING from this -- a quasi-static
    #: reference then suppresses chatter, a moving reference is still
    #: followed (the plan forbids forcing all hand motion to zero and any
    #: robot-output-based frame selection)
    ref_hand_delta: dict = field(default_factory=dict)
    #: site motion (m) between z_prev and z_pred for this frame -- scales
    #: the solver's interpolation-dip margin (dip grows ~ quadratically
    #: with the per-frame step)
    frame_motion_m: float = 0.0
    #: C2 (train24fix2 section 4.2): per-side enable state of the regen hand
    #: box-frame speed rows at THIS frame, the reason for it, and the previous
    #: accepted FULL-SCENE hand/box distance it was derived from.  Empty or
    #: missing keys keep the rows OFF (standard mode, legacy contexts) -- the
    #: projected ``prev_hand_dist`` above belongs to the approach corridor
    #: alone and is never a C2 input.
    c2_hand_enabled: dict = field(default_factory=dict)
    c2_hand_reason: dict = field(default_factory=dict)
    prev_hand_scene_dist: dict = field(default_factory=dict)


def hand_speed_rows_active(cfg, ctx: FrameContext, side: str) -> bool:
    """THE shared enable criterion of the regen hand box-frame speed rows
    (train24fix2 section 4.2).  The value path (``_fixed_post_rows``) and
    the QP mirror (``linearize``) must answer it identically: regen mode, a
    positive bound parameter, the previous box-frame site array, and the
    frame's reference-interval enable state over the previous FULL-SCENE
    distance.  The old criterion (projected ``prev_hand_dist`` at 6mm) is
    NOT part of C2 -- it missed five of the ten frozen speed spikes.

    The criterion is form-agnostic (axis or norm_cut): the BOUND FORM is a
    profile property, never an enable input (train24fix3 plan section 7:
    enable/release semantics stay exactly the reference-interval ones)."""
    return (cfg.regen_mode and cfg.regen_hand_speed_mps > 0.0
            and ctx.prev_hand_local.get(side) is not None
            and bool(ctx.c2_hand_enabled.get(side, False)))


#: canonical site order of the C2 rows (HAND_SITES is label-keyed; the
#: row layout is index-based -- one canonical mapping, used everywhere)
_C2_SITE_LABELS = ("palm", "thumb", "pinky")


def _norm_cut_profile(c2_profile):
    """The attached profile when it defines the norm_cut bound form (else
    None -- axis rows or no profile).  A single point of truth for the
    mode switch so the value path, the QP path and the cut loop cannot
    disagree about which C2 semantics are in force."""
    if c2_profile is not None and c2_profile.c2_mode == "norm_cut":
        return c2_profile
    return None


@dataclass
class FrameResult:
    accepted: bool
    z: np.ndarray
    iterations: int = 0
    skipped: bool = False
    slack_used: bool = False
    status: str = "solved"           # solved / infeasible / max_iters / task_quality
                                     # / review_failed
    merit: float = float("inf")
    #: number of QP solves actually issued for this frame (all branches;
    #: stage 0 statistics, 2026-09-16 -- the return branches used to hardcode
    #: ``scp_max_iters``, which reported phantom iterations)
    qp_calls: int = 0
    diagnostics: dict = field(default_factory=dict)
    #: final trust radius of this attempt (F0: solver exit evidence)
    trust: float = 0.0
    #: constraint violation at the LAST evaluated state of this attempt and
    #: the collision-row names still violated there (F0: last-constraint
    #: evidence on failure, no re-derivation needed)
    last_viol: float = 0.0
    violated_rows: list = field(default_factory=list)
    #: candidate pairs dropped by the QP row budget at the last linearization
    truncated_rows: list = field(default_factory=list)
    #: OSQP exit status of this attempt's LAST QP solve (F0 evidence)
    last_qp_status: str = ""
    #: ACTUAL loop-exit cause of this attempt (review R4, 2026-10-01):
    #: converged / restoration_qp_failed_trust_min / no_merit_step_trust_min
    #: / budget_exhausted.  ``status`` keeps its historical vocabulary
    #: (infeasible/max_iters/...) -- it is a compatibility label, and
    #: "infeasible" must NOT be read back as proof of structural joint
    #: infeasibility: budget_exhausted/trust-collapse exits only show the
    #: search did not converge within the fixed budget.
    term_reason: str = ""
    #: the iteration budget this attempt ran under (base scp_max_iters, +
    #: restoration extras on forced slack) -- separates "hit the budget"
    #: from "collapsed below trust_radius_min"
    iter_budget: int = 0
    #: probe-row names active at the last linearization (F4 evidence)
    probe_rows: list = field(default_factory=list)
    probe_truncated: list = field(default_factory=list)
    #: pipeline-side acceptance check cache (scp.FrameCheck); set by the
    #: pipeline so a state is checked once and the log serializes that result
    frame_check: object | None = None


@dataclass
class FrameCheck:
    """Evidence of ONE frame+transition acceptance check (pipeline side).

    ``ok`` is the decision; everything else is serialized evidence for the
    verbose frame log -- the log never re-computes (F0: verbose must not add
    checks, FK, carry-back or cache changes). ``discovered`` holds the stable
    pair identities of violating collision rows; the *caller* decides when to
    feed them back to the solver.
    """

    ok: bool
    reason: str = "ok"                     # ok / state_constraints /
                                           # transition_interpolated / task_quality
    #: stable id of this check event, assigned by the pipeline
    #: (``pass:frame:check``); the attempt it belongs to references it (R8)
    check_id: str = ""
    viol: float = 0.0
    rows: list[dict] = field(default_factory=list)
    discovered: list[tuple] = field(default_factory=list)
    #: identity-tagged pairs found violating at an INTERPOLATED state (F4:
    #: the solver watches these with probe rows; kept separate from the
    #: endpoint violations, which the candidate set already carries)
    probe_discovered: list[tuple] = field(default_factory=list)
    carried: list[str] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    probe_n: int = 0
    probe_viol: float = 0.0
    hand_prev: dict = field(default_factory=dict)
    hand_cand: dict = field(default_factory=dict)
    corridor: dict = field(default_factory=dict)
    #: C2 evidence (train24fix2 section 4.3): per-side enable/reason and, when
    #: enabled, the row values' absmax / bound / true norm speed -- copied
    #: from the endpoint check's row assembly, never recomputed for the log
    c2: dict = field(default_factory=dict)
    band_margin: float = 0.0
    motion_m: float = 0.0
    truncated_rows: list[str] = field(default_factory=list)


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
        self.Dinv = 1.0 / self.D
        self.Dinv2 = 1.0 / self.D**2
        self.dt = 1.0 / cfg.fps
        self._osqp = osqp.OSQP()
        self._frame_pairs: list | None = None
        #: pairs discovered violating by THIS frame's checks (endpoint/probe)
        #: and carried into the next solve attempt regardless of their
        #: distance at the attempt's initial state (stage 1, 2026-09-16);
        #: identities are registry-normalized (F1, 2026-09-22)
        self._extra_pairs: list = []
        #: static registry lookup (identity -> geometric category); built once
        self._pair_kinds: dict | None = None
        #: pending partial-refresh count: each one triggers a fast-family +
        #: carried-extras rescan at the next linearization (stage 1)
        self._refresh_counter: int = 0
        #: diagnostics for the regeneration approach corridor (per side:
        #: bound geom distance + active upper at the LAST linearization);
        #: read by the pipeline's verbose frame log, not by the solver
        self.last_corridor: dict[str, dict] = {}
        #: C2 hand-speed row evidence at the LAST row assembly (per side:
        #: enabled/reason and, when on, row absmax/bound/true norm speed);
        #: populated by ``_fixed_post_rows`` from the already-forwarded
        #: evaluator state -- the log never recomputes
        self.last_hand_speed: dict[str, dict] = {}
        #: last selected nearest hand geom per side (re-selection evidence)
        self._last_regen_geom: dict[str, str] = {}
        #: geom-pair names parallel to the rows returned by the last
        #: ``_collision_rows`` call (diagnostics)
        self.last_row_pairs: list[str] = []
        #: ``(ga, gb, kind)`` parallel to the rows returned by the last
        #: ``_collision_rows`` call. Always the REGISTRY category (never a
        #: row role like ``regen``), so carry-back keeps the geometric
        #: semantics of the pair (F1).
        self.last_row_ids: list[tuple[int, int, str]] = []
        #: row role per entry of ``last_row_ids``: "regen" (active
        #: hand-contact row) or "collision" (diagnostics only)
        self.last_row_roles: list[str] = []
        #: candidate pairs silently dropped by the QP row budget at the last
        #: linearization (nearest-first keep; reported, never hidden)
        self.last_truncated: int = 0
        self.last_truncated_ids: list[tuple[int, int, str]] = []
        self.last_truncated_names: list[str] = []
        # -- transition probes (F4, 2026-09-22, plan section 7.1) -------------
        #: validation config carrying the interpolation protocol
        #: (``check_interpolated_states`` / ``max_substep_motion_m`` /
        #: ``max_subdivisions``); wired by the pipeline, None -> probes off
        self.probe_cfg = None
        #: pairs a check found violating at an INTERPOLATED state: watched in
        #: every probe row build of this frame (within-frame mechanism)
        self._probe_watch: list[tuple[int, int, str]] = []
        #: per-state cache of the probe row definitions, so the merit
        #: evaluation and the QP assembly of one iteration share one probe
        #: computation (they are always called at the same linearization state)
        self._probe_cache: dict | None = None
        self.last_probe_rows: list[str] = []
        self.last_probe_truncated: list[str] = []
        #: full probe row records of the last build (name/alpha/dist/J)
        self._probe_rows_detail: list[dict] = []
        #: cached descriptors of the fixed-pre / fixed-post blocks (static
        #: layouts; built once, R7). The post cache is mode-keyed: norm_cut
        #: swaps the C2 tail semantics (train24fix3 plan 7.2).
        self._pre_desc: list[str] | None = None
        self._post_desc: list[str] | None = None
        self._post_desc_norm: bool | None = None
        # -- train24fix3 (plan sections 4.1/5/6.1) --------------------------
        #: the attached C2 profile (None = plain config behaviour). Its
        #: solver settings apply ONLY to frames with active C2 rows
        #: (``_frame_qp_settings``); its C2 row knobs drive build_qp (the
        #: N2/N3 preconditioning and recovery-slack penalty).
        self.c2_profile = None
        #: OFF by default (None). When a dict, ``solve_frame`` records the
        #: full per-iteration evidence of every attempt into
        #: ``self.trace["attempts"]`` (plan section 5.1) and ``solve_qp``
        #: appends per-solve records to ``self.qp_trace`` (arrays to
        #: ``self.qp_trace_arrays``). Pure read-out: every guard copies
        #: what was already computed, nothing feeds back, no extra solves;
        #: with it None the control flow is identical to N0.
        self.trace: dict | None = None
        self.qp_trace: list | None = None
        self.qp_trace_arrays: list | None = None
        #: slack bookkeeping of the last ``build_qp(slack=True)`` call
        #: (trace-only; identities of the relaxed rows + their penalties)
        self.last_slack_rows: dict | None = None
        # -- train24fix3 norm_cut (plan section 7.2) -------------------------
        #: cutting planes of the CURRENT linearization only: cleared at
        #: every solve_frame entry AND at each outer iteration (a new z is a
        #: new linearization); a stale cut whose side went inactive is
        #: dropped at assembly
        self._c2_cuts: list[dict] = []
        #: per-side C2 linearization (d_j, J_j) filled by the last build_qp
        #: call on an active side; consumed by the cut loop of the SAME
        #: iteration
        self._c2_lin: dict = {}
        #: cumulative cut-loop accounting of the current solve_frame call
        #: (plan 7.2 "每轮切面求解、失败…入账"); reported in FrameResult
        #: diagnostics, never a decision input -- solve_frame resets it
        self._c2_cut_stats: dict = {
            "resolves": 0, "directions_added": 0, "per_site_cap_hit": [],
            "resolve_cap_hit": False, "nonfinite_skipped": 0,
            "final_pred_norm_max_m": None, "nonclosed": False}

    # -- pair identity -------------------------------------------------------

    def _registry_nearest_hand(self, geoms: set[int], distmax_start: float = 1.0):
        """Nearest ``(dist, ga, gb, fromto)`` over the registry's robot_box
        pairs for ``geoms``, bypassing the activation-gated candidate set.

        R2 (train24fix): starts at a 1 m query range and escalates once; only
        ``dist < distmax`` hits count (MuJoCo returns exactly ``distmax`` with
        a zero ``fromto`` on a miss -- that is a truncated sentinel, never a
        distance). Returns ``None`` when no real hit exists at either range.
        """
        ev = self.ev
        for distmax in (distmax_start, 10.0):
            best = None
            for ga, gb, kind in ev.pairs:
                if kind != "robot_box" or (ga not in geoms and gb not in geoms):
                    continue
                d, ft = ev.geom_distance(ga, gb, distmax=distmax)
                if d >= distmax:
                    continue
                if best is None or d < best[0]:
                    best = (d, ga, gb, ft)
            if best is not None:
                return best
        return None

    def pair_identity(self, ga: int, gb: int) -> tuple[int, int, str]:
        """Registry-normalized ``(ga, gb, kind)`` for a pair identity.

        The static collision registry is the ONLY source of geometric
        category (F1): row roles such as ``regen`` describe *why* a row was
        added, never *what* the pair is. The lookup accepts either order but
        returns the registry order so every consumer (nearest-hand-geom
        selection, band classification, carry-back) sees one canonical form.
        An unknown pair is an implementation error and raises instead of
        silently losing a constraint row.
        """
        if self._pair_kinds is None:
            self._pair_kinds = {(a, b): k for a, b, k in self.ev.pairs}
        key = (ga, gb)
        if key not in self._pair_kinds:
            key = (gb, ga)
            if key not in self._pair_kinds:
                raise ValueError(
                    f"pair ({ga}, {gb}) is not in the collision registry: "
                    f"{self.ev.geom_name(ga)}~{self.ev.geom_name(gb)}")
        return key[0], key[1], self._pair_kinds[key]

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
        # grasp stability (F4 section 7.2): box-frame hand displacement must
        # follow the REFERENCE's displacement -- active only while the
        # reference schedule holds this hand
        out["r_stab"] = self._stab_residual(ctx)
        return out

    def _stab_residual(self, ctx: FrameContext) -> np.ndarray | None:
        """``[3*n_active]`` grasp-stability residual at the current state.

        ``r = (hand_local(z).mean(0) - prev hand_local.mean(0) - ref_delta)/scale``
        per active side: zero when the robot hand moves exactly like the
        reference in the box frame. ``None`` when regeneration is off, the
        side is not held, or the previous accepted state is unknown.
        """
        cfg = self.cfg
        if not cfg.regen_mode or ctx.z_prev is None:
            return None
        ev = self.ev
        rows = []
        for side in ("l", "r"):
            if not ctx.ref_hold.get(f"left_hand" if side == "l" else "right_hand"):
                continue
            ref_delta = ctx.ref_hand_delta.get(side)
            prev = ctx.prev_hand_local.get(side)
            if ref_delta is None or prev is None:
                continue
            step = (ev.hand_local(side).mean(0)
                    - np.asarray(prev, dtype=float).mean(0))
            rows.append((step - np.asarray(ref_delta, dtype=float))
                        / cfg.regen_hand_stab_scale_m)
        if not rows:
            return None
        return np.concatenate(rows)

    def _stab_jacobian(self, ctx: FrameContext) -> np.ndarray:
        """``[3*n_active, 35]`` Jacobian of :meth:`_stab_residual`."""
        cfg = self.cfg
        ev = self.ev
        rows = []
        for side in ("l", "r"):
            if not ctx.ref_hold.get(f"left_hand" if side == "l" else "right_hand"):
                continue
            if ctx.ref_hand_delta.get(side) is None or ctx.prev_hand_local.get(side) is None:
                continue
            J = ev.hand_local_jac(side).mean(0)                    # [3,35]
            rows.append(J / cfg.regen_hand_stab_scale_m)
        return np.vstack(rows) if rows else np.zeros((0, N_TANGENT))

    def _fixed_pre_rows(self, z: np.ndarray, ctx: FrameContext, trust: float):
        """Rows ``[0, N_FIXED)``: joint limits, velocities, trust region."""
        ev, cfg = self.ev, self.cfg
        l = np.full(N_FIXED, -INF)
        u = np.full(N_FIXED, INF)
        val = np.zeros(N_FIXED)
        # per-row physical scales for violation normalization: a 2mm gate
        # violation must outweigh a 5cm tracking step, so one-sided rows
        # normalize by their own tolerance, not by a meter
        scale = np.ones(N_FIXED)
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
        return val, l, u, scale

    def _fixed_post_rows(self, z: np.ndarray, ctx: FrameContext):
        """Rows AFTER the collision block (block-local layout, 96 rows).

        Hand anchors, foot anchors, foot support speeds and the regen
        box-frame hand speed rows. Kept separate from the collision block so
        the QP path (budgeted) and the check path (uncapped) can share them.
        """
        ev, cfg = self.ev, self.cfg
        n = N_POST_ROWS
        l = np.full(n, -INF)
        u = np.full(n, INF)
        val = np.zeros(n)
        scale = np.ones(n)

        # hand anchors
        for si, side in enumerate(("l", "r")):
            anchor = ctx.hand_anchor.get(side)
            if anchor is None:
                continue
            local = ev.hand_local(side)
            base = si * N_HAND_ROWS // 2
            val[base : base + 9] = local.reshape(-1)
            l[base : base + 9] = (anchor - cfg.hand_anchor_tol_m).reshape(-1)
            u[base : base + 9] = (anchor + cfg.hand_anchor_tol_m).reshape(-1)
            scale[base : base + 9] = cfg.hand_anchor_tol_m

        # foot anchors + support speed (per site)
        for si, side in enumerate(("l", "r")):
            anchor = ctx.foot_anchor.get(side)
            ids = ev.sole_sites[side]
            pts = ev.site_positions(ids)
            base = N_HAND_ROWS + si * 15
            if anchor is not None:
                val[base : base + 15] = pts.reshape(-1)
                l[base : base + 15] = (anchor - cfg.foot_anchor_tol_m).reshape(-1)
                u[base : base + 15] = (anchor + cfg.foot_anchor_tol_m).reshape(-1)
                scale[base : base + 15] = cfg.foot_anchor_tol_m
            base_s = N_HAND_ROWS + N_FOOT_ROWS + si * 15
            prev = ctx.prev_sole_world.get(side)
            if anchor is not None and prev is not None:
                val[base_s : base_s + 15] = (pts - prev).reshape(-1)
                lim_f = self.dt * cfg.foot_speed_mps
                l[base_s : base_s + 15] = -lim_f
                u[base_s : base_s + 15] = lim_f
                scale[base_s : base_s + 15] = lim_f

        # regen-mode hand box-frame speed rows (C2, train24fix2 section 4):
        # once ENABLED for the current reference hold interval (the shared
        # criterion in hand_speed_rows_active -- previous accepted FULL-SCENE
        # distance <= regen_hand_contact_gap_m, see C2HandSpeedState), the
        # per-frame box-frame motion of every hand site axis is hard-bounded.
        # Bounds: +-dt * regen_hand_speed_mps, measured exactly like the
        # validator (hand_local vs previous accepted hand_local); the
        # certification entry gate (0.05) admits sqrt(3) * the bound, so a
        # row-satisfying candidate is never speed-gated out.  Approach frames
        # (interval not yet enabled) are exempt -- the corridor needs mm-scale
        # closing steps; interval end releases the rows unconditionally.
        self.last_hand_speed = {}
        if cfg.regen_mode and cfg.regen_hand_speed_mps > 0.0:
            norm_prof = _norm_cut_profile(self.c2_profile)
            lim_h = (self.dt * norm_prof.c2_norm_speed_mps if norm_prof
                     else self.dt * cfg.regen_hand_speed_mps)
            for si, side in enumerate(("l", "r")):
                active = hand_speed_rows_active(cfg, ctx, side)
                if active:
                    prev_local = ctx.prev_hand_local[side]
                    vals = (ev.hand_local(side) - prev_local).reshape(-1)
                    base = N_HAND_ROWS + N_FOOT_ROWS + N_FOOT_SPEED + si * 9
                    if norm_prof is None:
                        # axis form: all nine rows, +-lim_h per axis
                        val[base : base + 9] = vals
                        l[base : base + 9] = -lim_h
                        u[base : base + 9] = lim_h
                        scale[base : base + 9] = lim_h
                        norm_speed = float(
                            np.linalg.norm(vals.reshape(3, 3), axis=1).max()
                            / self.dt)
                    else:
                        # norm_cut form (train24fix3 plan 7.2): the REAL rows
                        # are per-site norms max(||d_j|| - B, 0)/B merged into
                        # the single violation sum. One active slot per site
                        # (3j), the other two stay inactive placeholders --
                        # the 18-slot post layout is shared with axis mode so
                        # every offset consumer keeps working; descriptors
                        # distinguish the semantics.
                        d = vals.reshape(3, 3)
                        for j in range(3):
                            r = base + 3 * j
                            val[r] = float(np.linalg.norm(d[j]))
                            u[r] = lim_h
                            scale[r] = lim_h
                        norm_speed = float(val[base : base + 9 : 3].max()
                                           / self.dt)
                    self.last_hand_speed[side] = {
                        "enabled": True,
                        "reason": ctx.c2_hand_reason.get(side),
                        "mode": "norm_cut" if norm_prof else "axis",
                        "row_absmax_m": float(np.abs(vals).max()),
                        "bound_abs_m": float(lim_h),
                        "norm_speed_mps": norm_speed,
                    }
                else:
                    self.last_hand_speed[side] = {
                        "enabled": False,
                        "reason": ctx.c2_hand_reason.get(side)}
        return val, l, u, scale

    def _collision_block(self, rows: list, mode: str):
        """``(val, lo, hi, scale)`` for a list of ``(dist, J, lo, hi)`` rows."""
        n = len(rows)
        l = np.full(n, -INF)
        u = np.full(n, INF)
        val = np.zeros(n)
        scale = np.ones(n)
        for k, (dist, _J, lo, hi) in enumerate(rows):
            val[k] = dist
            if mode == "accept" and hi > INF / 2:
                # non-contact rows relax to the validator penetration gate;
                # contact-band rows keep their band
                l[k], u[k] = cfg_noncontact(self.cfg), hi
            else:
                l[k], u[k] = lo, hi
            scale[k] = max(min(hi - l[k], self.cfg.penetration_scale_m), 1e-4)
        return val, l, u, scale

    def constraint_rows(
        self, z: np.ndarray, ctx: FrameContext, trust: float, mode: str = "repair",
        band_margin_m: float | None = None, probe_mode: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Nonlinear constraint values and bounds [l, u] (rows as in constraints.py).

        ``mode="accept"`` uses validator-grade bounds (penetration tolerated to
        -2mm everywhere) for skip decisions; ``mode="repair"`` keeps the
        proactive +2mm non-contact clearance the solver enforces.
        ``probe_mode`` marks interpolated mid-state evaluations (stage 2):
        regen corridor uppers do not bind there, only the penetration floor.

        Uses the QP row budget (``max_active_pairs``); acceptance checks use
        :meth:`check_rows`, which evaluates every candidate pair.
        """
        ev = self.ev
        # guarantee the evaluator is at z (callers usually forwarded already)
        ev.forward(z, ctx.box_pose)
        pre = self._fixed_pre_rows(z, ctx, trust)
        rows = self._collision_rows(ctx, band_margin_m, probe_mode=probe_mode,
                                    need_jacobian=False, mode=mode)
        self._last_rows = rows
        block = self._collision_block(rows, mode)
        post = self._fixed_post_rows(z, ctx)
        # transition-probe rows LAST (F4): appended after every existing
        # block so all frozen row offsets stay valid; the probe block is
        # separate from the endpoint/anchor layout and never crowds it out
        probe = self._probe_block(ctx, z)
        return _concat_rows(pre, block, post, probe)

    def check_rows(self, z: np.ndarray, ctx: FrameContext,
                   band_margin_m: float | None = None, probe_mode: bool = False):
        """Acceptance-grade rows for a FULL check (no QP row budget).

        Every candidate pair gets a row regardless of ``max_active_pairs``:
        a successful return must not hide a violating pair behind the QP row
        limit (F1). Returns ``(val, lo, hi, scale, names, ids)`` with the
        descriptors parallel to the collision block starting at ``N_FIXED``.
        """
        ev = self.ev
        ev.forward(z, ctx.box_pose)
        pre = self._fixed_pre_rows(z, ctx, self.cfg.trust_radius_max)
        rows = self._collision_rows(ctx, band_margin_m, probe_mode=probe_mode,
                                    budget=False, need_jacobian=False,
                                    mode="accept")
        self._last_rows = rows
        names, ids = list(self.last_row_pairs), list(self.last_row_ids)
        block = self._collision_block(rows, "accept")
        post = self._fixed_post_rows(z, ctx)
        val, lo, hi, scale = _concat_rows(pre, block, post)
        return val, lo, hi, scale, names, ids

    # -- transition probes (F4, 2026-09-22) -----------------------------------

    def note_probe_pairs(self, pairs: list) -> None:
        """Watch pairs a check found violating at an INTERPOLATED state.

        The transition violation is a fact about the *path* between the
        previous accepted state and the candidate; the solver can only
        repair it if its linearization carries rows for those pairs at the
        probe positions. Identities are registry-normalized (F1); unknown
        pairs raise. A NEW identity changes the row set of the current
        state, so it invalidates the probe cache immediately (R3); repeating
        an already-watched identity is idempotent.
        """
        known = {(ga, gb) for ga, gb, _k in self._probe_watch}
        n_new = 0
        for pair in pairs:
            ga, gb, kind = self.pair_identity(pair[0], pair[1])
            if (ga, gb) not in known:
                self._probe_watch.append((ga, gb, kind))
                known.add((ga, gb))
                n_new += 1
        if n_new:
            self._probe_cache = None

    def clear_probe_pairs(self) -> None:
        self._probe_watch = []
        self._probe_cache = None

    def _probe_grid(self, ctx: FrameContext, z: np.ndarray) -> tuple[list[float], float]:
        """Interior interpolation fractions for the (z_prev -> z) interval.

        Reproduces the validator's protocol (same motion metric, same
        ``max_substep_motion_m`` / ``max_subdivisions`` clipping), so probe
        rows constrain exactly the states the transition check and the final
        validator probe. Empty for the first frame or when motion needs no
        subdivision.
        """
        pcfg = self.probe_cfg
        if pcfg is None or ctx.z_prev is None or not pcfg.check_interpolated_states:
            return [], 0.0
        box_prev = ctx.box_pose_prev if ctx.box_pose_prev is not None else ctx.box_pose
        motion = self.ev.motion_between(ctx.z_prev, box_prev, z, ctx.box_pose)
        n_sub = int(np.clip(np.ceil(motion / pcfg.max_substep_motion_m), 1,
                            pcfg.max_subdivisions + 1))
        return [i / n_sub for i in range(1, n_sub)], motion

    def _probe_mid(self, ctx: FrameContext, z: np.ndarray,
                   alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated scene (robot, box) at fraction ``alpha``.

        Identical protocol to ``IndependentValidator._interpolate`` (root
        translation lerp, rotation slerp, joint lerp, box pose lerp --
        validated by a contract test), so a solver-side probe row is a
        linearization of the exact state the check evaluates.
        """
        from scipy.spatial.transform import Rotation as Rot, Slerp

        q_a = ctx.z_prev
        box_a = ctx.box_pose_prev if ctx.box_pose_prev is not None else ctx.box_pose
        q_mid = q_a.copy()
        q_mid[0:3] = (1 - alpha) * q_a[0:3] + alpha * z[0:3]
        ra = Rot.from_quat(q_a[3:7][[1, 2, 3, 0]])
        rb = Rot.from_quat(z[3:7][[1, 2, 3, 0]])
        q_mid[3:7] = Slerp([0, 1], Rot.concatenate([ra, rb]))(alpha).as_quat()[[3, 0, 1, 2]]
        q_mid[7:36] = (1 - alpha) * q_a[7:36] + alpha * z[7:36]
        box_mid = (1 - alpha) * box_a + alpha * ctx.box_pose
        return q_mid, box_mid

    def _probe_composite_jacobian(self, J_mid: np.ndarray, ctx: FrameContext,
                                  z: np.ndarray, alpha: float) -> np.ndarray:
        """Chain the mid-state distance gradient back to the endpoint increment.

        ``q_mid`` depends on the endpoint increment ``xi`` through the
        interpolation: root translation and joints carry ``alpha``, the root
        rotation carries ``alpha * J_l(alpha*D) J_l^{-1}(D)`` with
        ``D = log(R_b R_a^T)`` the world-frame relative rotation (the
        retraction is left-multiplicative and the rotation path is the slerp
        geodesic, whose world increment is exactly that combination -- FD
        verified in the contract tests). Using the endpoint Jacobian
        directly, or ``alpha`` on the rotation block, is a linearization of a
        different function (plan section 7.1).
        """
        R_a = quat_to_mat_wxyz(np.asarray(ctx.z_prev[3:7], dtype=float)[None])[0]
        R_b = quat_to_mat_wxyz(np.asarray(z[3:7], dtype=float)[None])[0]
        d_vec = so3_log_np((R_b @ R_a.T)[None])[0]
        T = (alpha * so3_left_jacobian_np((alpha * d_vec)[None])[0]
             @ so3_left_jacobian_inverse_np(d_vec[None])[0])
        M = np.zeros((N_TANGENT, N_TANGENT))
        M[0:3, 0:3] = alpha * np.eye(3)
        M[3:6, 3:6] = T
        M[6:35, 6:35] = alpha * np.eye(29)
        return J_mid @ M

    def _probe_cache_key(self, ctx: FrameContext, z: np.ndarray) -> tuple:
        """Every dependency of a probe row build (R3).

        ``z`` alone is NOT the state: the rows also depend on the previous
        accepted frame (the interpolation endpoints), the current and previous
        box poses, the watched pair identities, the candidate identities of
        the last linearization, the probe protocol and the row budget/floor.
        The old z-only key served stale rows after a carry-back added a pair,
        so the next QP solved a system without the violation the check had
        just found. Byte snapshots are compared, never live references.
        """
        pcfg = self.probe_cfg
        cfg = self.cfg
        return (
            np.asarray(z, dtype=float).tobytes(),
            None if ctx.z_prev is None
            else np.asarray(ctx.z_prev, dtype=float).tobytes(),
            np.asarray(ctx.box_pose, dtype=float).tobytes(),
            None if ctx.box_pose_prev is None
            else np.asarray(ctx.box_pose_prev, dtype=float).tobytes(),
            None if pcfg is None else (
                bool(pcfg.check_interpolated_states),
                float(pcfg.max_substep_motion_m), int(pcfg.max_subdivisions)),
            tuple((ga, gb) for ga, gb, _k in self._probe_watch),
            tuple((ga, gb) for ga, gb, _k in self.last_row_ids),
            int(cfg.max_probe_rows), float(cfg.probe_row_margin_m),
            float(cfg_noncontact(cfg)),
        )

    def _restore_probe_diagnostics(self, cache: dict) -> None:
        """Restore the diagnostics parallel to cached rows (R3): a cache hit
        must not lose the row names, alphas or truncation evidence."""
        self.last_probe_rows = list(cache["last_rows"])
        self.last_probe_truncated = list(cache["last_truncated"])

    def _probe_rows(self, ctx: FrameContext, z: np.ndarray,
                    need_jacobian: bool) -> list[dict]:
        """Probe point constraints for the current linearization state.

        Rows enforce the anti-penetration floor at every interior
        interpolation state (stage 2: no corridor upper there) for the pairs
        that are near/below at the probe -- the pairs with endpoint rows plus
        everything a check caught violating at a probe. The nonlinear path
        value is what the solver's merit and the checks use; the row's
        Jacobian is the composite chain rule above.

        Only queries that actually found a closer configuration produce rows:
        ``mj_geomDistance`` returns exactly ``distmax`` when nothing was found
        (and leaves ``fromto`` zeroed), and that search bound is not a
        distance (R1). The row budget is separate from the endpoint/anchor
        rows (``max_probe_rows``): over budget the most violated rows are kept
        and the rest are REPORTED (``last_probe_truncated``) -- the pipeline's
        uncapped probe check stays the arbiter, so a dropped row can never
        turn into a silent accept.

        The result is cached under :meth:`_probe_cache_key` (R3): a hit is only
        valid for the same state AND the same dependencies, and a value-only
        entry is never served when Jacobians are requested.
        """
        cfg = self.cfg
        key = self._probe_cache_key(ctx, z)
        cache = self._probe_cache
        if cache is not None and cache["key"] == key:
            if not need_jacobian or cache["jacobian"]:
                self._restore_probe_diagnostics(cache)
                return cache["rows"]
        alphas, motion = self._probe_grid(ctx, z)
        floor = cfg_noncontact(cfg)
        if not alphas:
            self.last_probe_rows, self.last_probe_truncated = [], []
            self._probe_cache = {"key": key, "rows": [], "jacobian": True,
                                 "last_rows": [], "last_truncated": []}
            return []
        # probe pairs: the pairs carrying endpoint rows at this linearization
        # (near at the frame end) plus the check-discovered watch list
        cand: list[tuple[int, int, str]] = []
        seen = set()
        for ga, gb, kind in self._probe_watch:
            cand.append((ga, gb, kind))
            seen.add((ga, gb))
        for ga, gb, kind in self.last_row_ids:
            if (ga, gb) not in seen:
                cand.append((ga, gb, kind))
                seen.add((ga, gb))
        rows = []
        margin = cfg.probe_row_margin_m
        dmax = floor + margin
        for alpha in alphas:
            q_mid, box_mid = self._probe_mid(ctx, z, alpha)
            self.ev.probe_forward(q_mid, box_mid)
            for ga, gb, kind in cand:
                dist, ft = self.ev.probe_geom_distance(ga, gb, distmax=dmax)
                # a truncated search returns the bound itself: not a distance,
                # no row (the old ``dist > dmax`` test let it through carrying
                # a zeroed segment and a meaningless Jacobian, R1). The test
                # is the returned DISTANCE -- a real zero-distance contact
                # collapses the segment the same way and keeps its row.
                if not dist < dmax:
                    continue
                nm = f"{self.ev.geom_name(ga)}~{self.ev.geom_name(gb)}"
                rows.append({"alpha": alpha, "dist": float(dist), "lo": floor,
                             "hi": INF, "name": nm, "id": (ga, gb, kind),
                             "J": None, "ft": ft})
        rows.sort(key=lambda r: r["dist"])
        if len(rows) > cfg.max_probe_rows:
            self.last_probe_truncated = [f"{r['name']}@a={r['alpha']:.3f}"
                                         for r in rows[cfg.max_probe_rows:]]
            rows = rows[:cfg.max_probe_rows]
        else:
            self.last_probe_truncated = []
        if need_jacobian:
            for r in rows:
                ga, gb, _k = r["id"]
                self.ev.probe_forward(*self._probe_mid(ctx, z, r["alpha"]))
                J_mid = self.ev.distance_jacobian(ga, gb, r["ft"], r["dist"], probe=True)
                r["J"] = self._probe_composite_jacobian(J_mid, ctx, z, r["alpha"])
        self.last_probe_rows = [f"{r['name']}@a={r['alpha']:.3f}" for r in rows]
        self._probe_cache = {"key": key, "rows": rows,
                             "jacobian": bool(need_jacobian),
                             "last_rows": list(self.last_probe_rows),
                             "last_truncated": list(self.last_probe_truncated)}
        return rows

    def _probe_block(self, ctx: FrameContext, z: np.ndarray) -> tuple:
        """``(val, lo, hi, scale)`` of the probe rows at the current state.

        Values are the true nonlinear probe distances (the merit must judge
        what the checks judge); bounds are the acceptance-grade penetration
        floor (the stage 2 probe semantics: floor only, no corridor upper).
        """
        rows = self._probe_rows(ctx, z, need_jacobian=False)
        self._probe_rows_detail = rows
        n = len(rows)
        val = np.array([r["dist"] for r in rows])
        lo = np.full(n, cfg_noncontact(self.cfg))
        hi = np.full(n, INF)
        scale = np.full(n, self.cfg.penetration_scale_m)
        return val, lo, hi, scale

    def _scan_pairs(self, full: bool) -> None:
        """(Re)scan the candidate pair set for this frame.

        ``full=False`` rescans only the fast families (robot_box /
        robot_ground, ~100 pairs, ~0.1 ms) plus whatever the pipeline
        discovered violating earlier in this frame — those pairs are always
        carried. ``full=True`` rescans the entire static registry
        (~1100 pairs, ~1.4 ms): used once per frame at the attempt's
        initial state, and again before a SUCCESSFUL return (stage 1,
        2026-09-16: the previous freeze-at-z_init let pairs that entered
        range mid-iteration into the accepted output unchecked by any row).
        """
        ev, cfg = self.ev, self.cfg
        dmax = cfg.activation_margin_m + 0.01
        cands: dict[tuple[int, int], tuple] = {}
        for ga, gb, kind in self._extra_pairs:
            cands[(ga, gb)] = (ga, gb, kind)
        families = ev.pairs if full else [
            p for p in ev.pairs if p[2] in ("robot_box", "robot_ground")]
        for ga, gb, kind in families:
            if kind == 'box_ground' or (ga, gb) in cands:
                continue
            dist, _ft = ev.geom_distance(ga, gb, distmax=dmax)
            if dist < cfg.activation_margin_m:
                cands[(ga, gb)] = (ga, gb, kind)
        if not full:
            # slow families (self-collision) are expensive to rescan; when
            # doing a partial refresh, keep the previously known self pairs
            # that are still within the margin (they were near at the last
            # linearization and stay candidates while the frame moves)
            for ga, gb, kind in self._frame_pairs or []:
                if kind == "self" and (ga, gb) not in cands:
                    dist, _ft = ev.geom_distance(ga, gb, distmax=dmax)
                    if dist < cfg.activation_margin_m:
                        cands[(ga, gb)] = (ga, gb, kind)
        # drop discovered pairs that have clearly left the margin again
        # (extra pairs were added unconditionally; re-check them here so the
        # set cannot grow unboundedly across attempts)
        self._frame_pairs = list(cands.values())
        if full:
            # a full scan already re-absorbed every pending refresh; leaving
            # the counter set would trigger redundant partial rescans
            self._refresh_counter = 0

    def note_discovered_pairs(self, pairs: list) -> None:
        """Carry check-discovered pairs into the candidate set (stage 1).

        Called by the pipeline when an endpoint/probe/review check finds a
        violating pair the current candidate set did not contain. ``pairs``
        carry the stable pair IDENTITY (``(ga, gb)`` or ``(ga, gb, role)``);
        the geometric category is always re-resolved from the static registry
        (F1, 2026-09-22). The old code copied the row label verbatim, so a
        pair discovered as an active ``regen`` contact row re-entered the
        candidate set labelled ``regen`` -- after which it was no longer
        eligible for the nearest-hand-geom contact selection or the
        ``robot_box`` band classification, silently dropping the contact
        requirement that had just been violated.

        Carried pairs participate in the NEXT linearization regardless of
        their distance at the initial state; they are pruned by the periodic
        rescans once genuinely far. Unknown identities raise.
        """
        known = {(ga, gb) for ga, gb, _k in self._extra_pairs}
        n_new = 0
        for pair in pairs:
            ga, gb, kind = self.pair_identity(pair[0], pair[1])
            if (ga, gb) not in known:
                self._extra_pairs.append((ga, gb, kind))
                known.add((ga, gb))
                n_new += 1
        if n_new:
            # new pairs must be rescanned into the candidate set; a partial
            # refresh (fast families + carried extras + still-near self
            # pairs) is enough and cheaper than a full registry scan
            self._frame_pairs = None
            self._refresh_counter += 1

    def clear_discovered_pairs(self) -> None:
        self._extra_pairs = []
        self._frame_pairs = None

    def _collision_rows(self, ctx: FrameContext,
                        band_margin_m: float | None = None,
                        probe_mode: bool = False,
                        budget: bool = True,
                        need_jacobian: bool = True,
                        mode: str = "repair") -> list[tuple[float, np.ndarray, float, float]]:
        """Active collision rows with contact-band classification.

        Hand geoms during hold and sole geoms during support are CONTACT
        pairs (band [-2mm, upper]); everything else keeps the positive
        non-contact clearance. A support foot touching the ground is the
        intended state, not a violation. ``band_margin_m`` explicitly
        overrides the band floor (transition probes pass 0.0 -> the
        validator-grade -2mm); ``None`` selects by ``mode``: the QP DRIVING
        floor keeps the regen interior margin (-1.5mm), while
        ``mode="accept"`` uses the acceptance-grade floor -- in regen mode
        ``cfg_noncontact`` (-1.8mm, train24fix F1: the single changed cell of
        the section-2.4 matrix; standard mode keeps the historical -2mm).
        ``probe_mode=True`` marks an INTERPOLATED mid-state evaluation: regen
        corridor rows keep only their anti-penetration floor there (the
        closing target is a frame-end bound; see the stage 2 comment below).

        ``budget=False`` skips the QP row cap: the check path evaluates
        EVERY candidate pair (F1 -- a successful return may not hide a
        violating pair behind ``max_active_pairs``). ``need_jacobian=False``
        skips the per-row distance Jacobian for pure value checks.

        Candidate pairs: full registry scan at the attempt's initial state
        (``solve_frame``), partial refresh (fast families + carried extras)
        at later linearizations, and a full rescan in the success review.
        Pairs discovered violating by endpoint/probe checks are carried via
        ``note_discovered_pairs`` with their REGISTRY category, so the
        active-contact selection and band classification survive the
        carry-back (F1, 2026-09-22).
        """
        ev, cfg = self.ev, self.cfg
        if self._frame_pairs is None:
            self._scan_pairs(full=True)
        elif self._refresh_counter > 0:
            self._refresh_counter -= 1
            self._scan_pairs(full=False)
        rows = []
        row_names = []
        row_ids = []
        row_roles = []
        self.last_corridor = {}
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
        # every linearization. Selection uses the REGISTRY category only
        # (F1), so a carried pair cannot lose its eligibility.
        regen_active: dict[tuple[int, int], tuple[float, object, float]] = {}
        if cfg.regen_mode:
            for side, name in (("l", "left_hand"), ("r", "right_hand")):
                if not ctx.ref_hold.get(name):
                    continue
                geoms = set(ev._hand_geoms[side])
                best = None
                # a candidate distance is usable ONLY below the query range:
                # a miss returns exactly distmax with a zero fromto, and a
                # carried-back far hand pair (review P1) would otherwise
                # enter the corridor row with the truncated 60 mm "distance"
                # and pass an upper the true ~100 mm distance violates
                gate = cfg.activation_margin_m + 0.01
                for dist, ga, gb, kind, ft in scored:
                    if (kind == "robot_box" and ga in geoms and dist < gate
                            and (best is None or dist < best[0])):
                        best = (dist, ga, gb, ft)
                selection_source = "candidate"
                if best is None:
                    # R2 (train24fix): the activation-gated candidate set
                    # cannot see a hovering hand -- D-R measured 5.7-11.8 cm
                    # during active reference grasps, so no corridor row
                    # existed and neither the pull nor the corridor contract
                    # ever engaged (107 deciding-check frames on sub10_001).
                    # Select the nearest hand geom from the FULL registry;
                    # only a real hit (dist < distmax, never the truncated
                    # return) is a usable distance.
                    best = self._registry_nearest_hand(geoms)
                    selection_source = "registry"
                if best is None:
                    # query failure is EXPLICIT: a synthetic always-violating
                    # row rejects the candidate instead of silently dropping
                    # the side's active row (the value is the query bound,
                    # not a measured distance)
                    prev = ctx.prev_hand_dist.get(side)
                    margin = solver_band_margin(ctx, cfg)
                    upper = cfg.regen_hand_contact_gap_m if prev is None else max(
                        cfg.regen_hand_contact_gap_m,
                        prev - cfg.regen_closing_rate_mps * self.dt - margin)
                    rows.append((10.0, np.zeros((1, N_TANGENT)), -INF, 0.0))
                    row_names.append(f"regen_query_failed[{side}]:regen")
                    row_ids.append(None)
                    row_roles.append("regen")
                    self.last_corridor[side] = {
                        "dist_m": None, "upper_m": upper, "geom": None,
                        "query_failed": True}
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
                regen_active[(best[1], best[2])] = (best[0], best[3], upper)
                sel = {"dist_m": best[0], "upper_m": upper,
                       "geom": ev.geom_name(best[1]),
                       "source": selection_source}
                prev_sel = self._last_regen_geom.get(side)
                if prev_sel is not None and prev_sel != sel["geom"]:
                    # a REAL state change re-selected the nearest hand geom;
                    # keep the reason so this cannot be confused with the old
                    # carry-back label pollution (F1)
                    sel["reselected_from"] = prev_sel
                self._last_regen_geom[side] = sel["geom"]
                self.last_corridor[side] = sel
        # regen mode keeps an interior margin above the -2 mm validator
        # floor on every band row: solutions sit on constraint boundaries
        # by construction, and boundary-sitting states would graze past the
        # gate at the transition probes' interpolated states. Probes
        # override this to the validator grade via ``band_margin_m=0``.
        # ACCEPT-mode verdicts (frame feasibility / merit / check_rows) use
        # the acceptance floor instead (train24fix F1): in regen mode
        # cfg_noncontact (-1.8mm, validator gate + the 0.2mm probe margin
        # the non-contact rows already carry) -- the driving floor stays
        # -1.5mm so the QP still keeps solutions away from the probes. The
        # independent validator's -2mm and every explicit/probe floor are
        # untouched. Standard mode is unchanged in every path.
        if band_margin_m is not None:
            band_lo = -cfg.contact_band_pen_m + band_margin_m
        elif mode == "accept" and cfg.regen_mode:
            band_lo = cfg_noncontact(cfg)
        else:
            band_lo = -cfg.contact_band_pen_m + (
                0.0005 if cfg.regen_mode else 0.0)
        added_regens: set = set()
        # Stage 2 (2026-09-16): the corridor upper bound is an END-OF-FRAME
        # closing target -- one full dt of closing from the previous accepted
        # distance. An interpolated mid-state sits at a FRACTION of the frame,
        # and binding it to the frame-end upper rejects legal approaches
        # (30mm -> 24mm closed by 5mm, the 27mm midpoint read the 24.178mm
        # frame-end bound and failed). Transition probes pass
        # ``probe_mode=True``: regen rows keep their anti-penetration floor
        # (band_lo, exactly what the validator also enforces mid-frame) and
        # drop the corridor upper. The final validator certifies entry
        # speeds/anchors at keyframes only -- it has no corridor semantics.
        for (ga, gb), (dist, ft, upper) in regen_active.items():
            J = ev.distance_jacobian(ga, gb, ft, dist) if need_jacobian else None
            hi = INF if probe_mode else upper
            rows.append((dist, J, band_lo, hi))
            row_names.append(f"{ev.geom_name(ga)}~{ev.geom_name(gb)}:regen")
            row_ids.append((ga, gb, self.pair_identity(ga, gb)[2]))
            row_roles.append("regen")
            added_regens.add((ga, gb))
        # QP row budget (99 + MAX_PAIRS layout). Pairs this frame's checks
        # caught violating (``_extra_pairs``) OUTRANK the nearest-first
        # candidate ordering -- they are the violation evidence the loop
        # must act on and may not be silently dropped by the cap (stage 1);
        # remaining slots go to the nearest candidates, and the overflow is
        # REPORTED with its identities (``last_truncated_ids``), never
        # hidden. Active regen rows are DEDUPLICATED out of the candidate
        # list first: a pair that is both an active contact row and a
        # discovered extra occupies exactly one slot (F1).
        extra_keys = {(ga, gb) for ga, gb, _k in self._extra_pairs}
        cands = [c for c in scored if (c[1], c[2]) not in added_regens]
        extras = [c for c in cands if (c[1], c[2]) in extra_keys]
        others = [c for c in cands if (c[1], c[2]) not in extra_keys]
        if budget:
            slot = cfg.max_active_pairs - len(added_regens)
            if len(extras) > slot:
                overflow = extras[slot:] + others
                chosen = extras[:slot]
            else:
                chosen = extras + others[: max(0, slot - len(extras))]
                overflow = others[max(0, slot - len(extras)):]
        else:
            chosen, overflow = extras + others, []
        self.last_truncated = len(overflow)
        self.last_truncated_ids = [(ga, gb, kind) for _, ga, gb, kind, _ in overflow]
        self.last_truncated_names = [f"{ev.geom_name(ga)}~{ev.geom_name(gb)}"
                                    for _, ga, gb, _k, _ft in overflow]
        for dist, ga, gb, kind, ft in chosen:
            J = ev.distance_jacobian(ga, gb, ft, dist) if need_jacobian else None
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
            row_names.append(f"{ev.geom_name(ga)}~{ev.geom_name(gb)}")
            row_ids.append((ga, gb, kind))
            row_roles.append("collision")
        self.last_row_pairs = row_names
        self.last_row_ids = row_ids
        self.last_row_roles = row_roles
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
        if res.get("r_stab") is not None:
            f += cfg.regen_hand_stab_weight * float((res["r_stab"] ** 2).sum())
        return f

    # -- linearization + QP ----------------------------------------------------

    def tracking_qp_blocks(self, z: np.ndarray, ctx: FrameContext):
        """Gauss-Newton blocks of the two tracking terms (F2).

        The true objective is ``f = w ||e/D||^2`` with ``r = e/D``; the QP
        variables are PHYSICAL tangent increments xi, so the residual
        Jacobian is ``J_r = dr/dxi`` and the consistent assembly is
        ``P += 2 w J_r^T J_r``, ``q += 2 w J_r^T r``.

        - translation/joint blocks: ``e`` is additive, ``J_r = diag(1/D)``,
          hence ``q = 2 w r/D = 2 w e/D^2``. (The old code used
          ``Dinv2 * r = 2 w e/D^3``: a 50x / 10x-too-large linear term at the
          default D, i.e. a QP model that disagreed with the merit the line
          search evaluates -- m2_m3_stage_gate_fix_plan F2.)
        - rotation block: the retraction is left-multiplicative
          (``R_next = Exp(dtheta) R``), so ``d r_rot/d dtheta =
          J_l^{-1}(e_rot)/D`` -- the SO(3) left-Jacobian inverse, not the
          identity (2.5% at 0.05 rad).
        """
        r_pred = _unit(tangent_error(z, ctx.z_pred), self.D)
        blocks = [(self.cfg.prediction_weight, r_pred)]
        if ctx.z_prev is not None:
            # no previous frame -> no temporal objective at all (the merit's
            # r_temp is identically zero, so a w_t J^T J term in P would be a
            # Hessian the merit does not have)
            blocks.append((self.cfg.temporal_weight,
                           _unit(tangent_error(z, ctx.z_prev), self.D)))
        out = []
        for w, r in blocks:
            J_r = np.diag(self.Dinv)
            e_rot = r[3:6] * self.D[3]                # r = e/D (uniform D_rot)
            Jinv = so3_left_jacobian_inverse_np(e_rot[None])[0]
            J_r[3:6, 3:6] = Jinv / self.D[3]
            out.append((w, J_r, r))
        return out

    def build_qp(self, z: np.ndarray, ctx: FrameContext, trust: float,
                 slack: bool = False, *, mu: float | None = None):
        """Assemble the QP at state z (caller must have forwarded to z)."""
        ev, cfg = self.ev, self.cfg

        # objective blocks
        P = np.zeros((N_TANGENT, N_TANGENT))
        q = np.zeros(N_TANGENT)
        for w, J_r, r in self.tracking_qp_blocks(z, ctx):
            P += 2.0 * w * (J_r.T @ J_r)
            q += 2.0 * w * (J_r.T @ r)
        if ctx.mesh_L is not None:
            R_g, x_g = ev.box_frame()
            kpt_w = ev.site_positions(ev.kpt_ids)
            kpt_g = (R_g.T @ (kpt_w - x_g).T).T
            X = np.vstack([kpt_g, ctx.mesh_fixed])
            r_mesh = (ctx.mesh_L @ X - ctx.delta_ref).reshape(-1)
            # NB: contract the site-point axis (see eval.hand_local_jac --
            # the old "ij,sak->sik" summed over the coordinate axis)
            J_kpt = np.einsum("ij,sjk->sik", R_g.T, ev.site_jacobians_35(ev.kpt_ids))
            J_mesh = self._mesh_jacobian(ctx.mesh_L, J_kpt)
            P += 2.0 * cfg.laplacian_weight * (J_mesh.T @ J_mesh)
            q += 2.0 * cfg.laplacian_weight * (J_mesh.T @ r_mesh)
        # step damping only: f_reg = w_reg ||D^-1 xi||^2 is NOT part of
        # ``objective`` (the line search / merit never sees it), so it lives
        # in P alone and must not be counted as a state-objective Hessian
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

        # grasp-stability soft target (F4 section 7.2): same Gauss-Newton
        # assembly as the tracking terms, so the QP model and the merit the
        # line search evaluates stay one objective
        r_stab = self._stab_residual(ctx)
        if r_stab is not None and len(r_stab):
            J_stab = self._stab_jacobian(ctx)
            P += 2.0 * cfg.regen_hand_stab_weight * (J_stab.T @ J_stab)
            q += 2.0 * cfg.regen_hand_stab_weight * (J_stab.T @ r_stab)

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

        # transition-probe rows (F4): the same linearized floor at interior
        # interpolation states, with the composite chain-rule Jacobian.
        # Separate layout/count (``max_probe_rows``), never displacing the
        # endpoint, anchor or speed rows above.
        for r in self._probe_rows(ctx, z, need_jacobian=True):
            A_rows.append(r["J"])
            l_rows.append(np.array([r["lo"] - r["dist"]]))
            u_rows.append(np.array([INF]))

        # train24fix3 (plan 6.1): actual row-count anchor of the post block.
        # The pre block's length depends on z_prev (99 with, 70 without) --
        # the C2 block start must come from the REAL assembly, never from
        # the nominal layout constant, or t=0 rows would be misattributed.
        anchor_post = sum(r.shape[0] for r in A_rows)

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

        # regen-mode hand box-frame speed rows (mirror of constraint_rows;
        # SAME shared criterion -- hand_speed_rows_active, C2 section 4.2).
        # N2/N3 (train24fix3 plan 6.1): the profile may divide these QP rows
        # by B -- a pure ROW transformation of the same linear constraint
        # set. The REAL acceptance rows stay in metres (val = d, l/u = +-B,
        # scale = B in _fixed_post_rows); the two paths align via
        # B * l_QP_scaled + d = -B (test-pinned).
        # norm_cut (plan 7.2): the three-axis +-B rows are the INITIAL
        # NECESSARY condition (ball contains box); cutting planes appended
        # below tighten toward the true per-site norm ball.
        c2_precond = (self.c2_profile is not None and
                      self.c2_profile.c2_row_precondition == "divide_by_bound")
        norm_prof = _norm_cut_profile(self.c2_profile)
        lim_h = (self.dt * norm_prof.c2_norm_speed_mps if norm_prof
                 else self.dt * cfg.regen_hand_speed_mps)
        if cfg.regen_mode and cfg.regen_hand_speed_mps > 0.0:
            self._c2_lin = {}
            for side in ("l", "r"):
                if not hand_speed_rows_active(cfg, ctx, side):
                    A_rows.append(np.zeros((9, N_TANGENT)))
                    l_rows.append(np.full(9, -INF)); u_rows.append(np.full(9, INF))
                else:
                    prev_local = ctx.prev_hand_local[side]
                    Jh = ev.hand_local_jac(side).reshape(9, N_TANGENT)
                    cur = ev.hand_local(side)
                    lo_r = -lim_h - (cur - prev_local)
                    hi_r = lim_h - (cur - prev_local)
                    if c2_precond:
                        Jh = Jh / lim_h
                        lo_r = lo_r / lim_h
                        hi_r = hi_r / lim_h
                    A_rows.append(Jh)
                    l_rows.append(lo_r.reshape(-1))
                    u_rows.append(hi_r.reshape(-1))
                    if norm_prof is not None:
                        # the linearization the cut loop consumes (plan 7.2:
                        # p_j = d_j + J_j xi at THIS state)
                        self._c2_lin[side] = {
                            "d": np.asarray(cur - prev_local, dtype=float),
                            "J": ev.hand_local_jac(side).reshape(9, N_TANGENT)}
        else:
            A_rows.append(np.zeros((N_HAND_SPEED, N_TANGENT)))
            l_rows.append(np.full(N_HAND_SPEED, -INF)); u_rows.append(np.full(N_HAND_SPEED, INF))
        c2_tail_end = sum(r.shape[0] for r in A_rows)

        # norm_cut cutting planes (plan 7.2): dynamic rows AFTER the frozen
        # post tail, in their OWN space -- they never displace collision,
        # probe or endpoint/anchor rows, and the padding target grows by
        # their count so the fixed layout check cannot fire on them.
        cut_rows_meta: list[tuple[str, int]] = []
        if norm_prof is not None and self._c2_cuts:
            for cut in self._c2_cuts:
                lin = self._c2_lin.get(cut["side"])
                if lin is None:
                    # the side went inactive between the cut's birth and now
                    # (different ctx/pass): the cut is stale, drop it
                    continue
                Jj = lin["J"][3 * cut["site"] : 3 * cut["site"] + 3]
                dj = lin["d"][cut["site"]]
                a = cut["n"] @ Jj
                nd = float(cut["n"] @ dj)
                if c2_precond:
                    a = a / lim_h
                    u_r = 1.0 - nd / lim_h
                else:
                    u_r = lim_h - nd
                A_rows.append(a.reshape(1, -1))
                l_rows.append(np.array([-INF]))
                u_rows.append(np.array([u_r]))
                cut_rows_meta.append((cut["side"], int(cut["site"])))

        # pad to the fixed layout length with inactive rows (collision block
        # may have fewer than MAX_PAIRS active pairs). The transition-probe
        # block (F4) has its OWN budgeted layout appended after the frozen
        # endpoint/anchor rows: padding target = N_ROWS + max_probe_rows
        # (+ the cut rows' own space)
        A = np.vstack(A_rows)
        lo = np.concatenate(l_rows)
        hi = np.concatenate(u_rows)
        n_missing = (N_ROWS + self.cfg.max_probe_rows + len(cut_rows_meta)
                     - A.shape[0])
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
            #
            # train24fix3 norm_cut (plan 7.2): a site's three initial axis
            # rows and ALL its cutting planes share ONE dimensionless slack
            # s_j -- plane form n'(d+J xi)/B <= 1 + s_j, priced ONCE per
            # site (mu * s_j), never per plane. Other families keep the
            # default one-slack-per-row identity mapping.
            finite = (lo > -INF / 2) | (hi < INF / 2)
            for i in trust_row_ids:  # trust region is never relaxed
                finite[i] = False
            idx = np.where(finite)[0]
            c2_lo = c2_tail_end - N_HAND_SPEED
            site_of_row: dict[int, tuple[str, int]] = {}
            if norm_prof is not None:
                for i in range(c2_lo, c2_tail_end):
                    side = "l" if i < c2_lo + 9 else "r"
                    base = c2_lo if side == "l" else c2_lo + 9
                    site_of_row[i] = (side, (i - base) // 3)
                for k, meta in enumerate(cut_rows_meta):
                    site_of_row[c2_tail_end + k] = meta
            row_slack = np.empty(len(idx), dtype=int)
            group_id: dict[tuple[str, int], int] = {}
            nxt = 0
            for k, i in enumerate(idx):
                g = site_of_row.get(int(i))
                if g is None:
                    row_slack[k] = nxt
                    nxt += 1
                else:
                    if g not in group_id:
                        group_id[g] = nxt
                        nxt += 1
                    row_slack[k] = group_id[g]
            n_s = nxt
            n_idx = len(idx)
            n_r = A.shape[0]
            A_new = np.zeros((n_r + n_idx + n_s, N_TANGENT + n_s))
            A_new[:n_r, :N_TANGENT] = A
            for k, i in enumerate(idx):
                col = N_TANGENT + row_slack[k]
                A_new[i, col] = -1.0
                A_new[n_r + k, :N_TANGENT] = A[i]
                A_new[n_r + k, col] = 1.0
            for s in range(n_s):
                A_new[n_r + n_idx + s, N_TANGENT + s] = 1.0   # s >= 0
            lo_new = np.concatenate([lo, np.full(n_idx, -INF), np.zeros(n_s)])
            hi_new = np.concatenate([hi, np.full(n_idx, INF), np.full(n_s, INF)])
            lo_new[idx] = -INF  # upper-side rows keep u, relaxed by -s
            lo_new[n_r:n_r + n_idx] = lo[idx]  # duplicated rows enforce l + s
            P_new = np.zeros((N_TANGENT + n_s, N_TANGENT + n_s))
            P_new[:N_TANGENT, :N_TANGENT] = P
            # train24fix3 N2 (plan 6.1): when the C2 rows entered the QP
            # divided by B, their recovery slacks are DIMENSIONLESS -- the
            # penalty must change units with them. scaled_10000B keeps the
            # metre-unit cost exactly: 10000 * B * s_norm == 10000 * s_m
            # (s_m = B * s_norm). Other families keep the original penalty
            # and the trust rows stay hard -- nothing else is transformed.
            # norm_cut: the shared site slacks carry the profile penalty.
            penalties = np.full(n_s, float(cfg.slack_penalty))
            if c2_precond:
                def _is_c2_row(i: int) -> bool:
                    return i in site_of_row or c2_lo <= i < c2_tail_end
                if self.c2_profile.c2_slack_penalty == "scaled_10000B":
                    scale_pen = float(cfg.slack_penalty) * lim_h
                    for k, i in enumerate(idx):
                        if _is_c2_row(int(i)):
                            penalties[row_slack[k]] = scale_pen
                elif self.c2_profile.c2_slack_penalty == "mu":
                    # N3 (plan 6.1) / S (plan 7.2): the dimensionless C2
                    # recovery penalty becomes the CURRENT merit weight --
                    # unit-consistent with the merit term's
                    # mu-per-normalized-violation. mu comes from solve_frame
                    # (incl. the regen multiplier)
                    if mu is None:
                        raise ValueError(
                            "mu-penalty profiles require the current merit "
                            "weight mu -- build_qp must be called with "
                            "solve_frame's mu")
                    for k, i in enumerate(idx):
                        if _is_c2_row(int(i)):
                            penalties[row_slack[k]] = float(mu)
            q_new = np.concatenate([q, penalties])
            if self.trace is not None:
                # train24fix3 plan 5.1: identities of the relaxed rows and
                # the penalties actually placed on their slacks (the N2/N3/S
                # profiles vary the C2 family). Read out once here -- the
                # QP rows carry no identifier arrays of their own.
                names_all = self._qp_row_names(A.shape[0], ctx)
                self.last_slack_rows = {
                    "idx": [int(i) for i in idx],
                    "names": [names_all[i] for i in idx],
                    "slack_of_row": [int(s) for s in row_slack],
                    "shared_groups": {f"{s}:{j}": g for (s, j), g
                                      in group_id.items()},
                    "penalty": [float(v) for v in q_new[N_TANGENT:]],
                    "n_s": int(n_s)}
            return (spa.csc_matrix(P_new), q_new, spa.csc_matrix(A_new), lo_new, hi_new), n_s

        return (spa.csc_matrix(P), q, spa.csc_matrix(A), lo, hi), 0

    def _qp_row_names(self, n_rows: int, ctx: FrameContext | None = None) -> list[str]:
        """Descriptors parallel to the LAST ``build_qp`` row layout (pre /
        collision / probe / post / cuts / padding). The QP block ORDER
        differs from the nonlinear value path -- the probe block sits
        BETWEEN collision and post, so the post-block relative offsets are
        NOT QP row indices (train24fix3 plan 6.1 implementation note). The
        pre block is CONDITIONAL: without ``ctx.z_prev`` the joint-velocity
        and root-velocity rows do not exist (64 rows, not 99). norm_cut:
        the 18-slot C2 tail carries the initial three-axis BOX rows (the
        real check rows are the per-site norms) and the cutting planes
        follow before the padding. Raises on desync so a silently
        mislabelled row can never enter the evidence.
        """
        pre = self._pre_row_descriptors()
        if ctx is not None and ctx.z_prev is None:
            pre = pre[:29] + pre[64:]     # joint limits + trust only
        post = list(self._post_row_descriptors())
        norm_prof = _norm_cut_profile(self.c2_profile)
        if norm_prof is not None:
            tail = [f"hand_norm_box[{side}:{label}:{ax}]"
                    for side in ("l", "r") for label in HAND_SITES[side]
                    for ax in _AXES]
            post = post[:-N_HAND_SPEED] + tail
            for cut in (self._c2_cuts or []):
                label = _C2_SITE_LABELS[cut["site"]]
                post.append(f"hand_norm_cut[{cut['side']}:{label}:"
                            f"k{cut['k']}]")
        names = (list(pre)
                 + list(self.last_row_pairs)
                 + list(self.last_probe_rows)
                 + post)
        n_missing = n_rows - len(names)
        if n_missing < 0:
            raise RuntimeError(
                f"QP row-name reconstruction desync: {len(names)} names "
                f"for {n_rows} rows")
        names += [f"pad[{i}]" for i in range(n_missing)]
        return names

    def _mesh_jacobian(self, L, J_kpt) -> np.ndarray:
        """Mesh residual Jacobian: only the first 15 points vary.

        Residual rows are (point i, axis a); d r_{i,a} = sum_k L[i,k] * d x_k[a].
        With J_kpt [15, 3, 35] (geom frame), J_mesh = kron(L[:, :15], I3) @ J_kpt.
        """
        Lk = np.asarray(L[:, :15].todense()) if hasattr(L, "todense") else np.asarray(L)[:, :15]
        return np.kron(Lk, np.eye(3)) @ J_kpt.reshape(15 * 3, N_TANGENT)

    def solve_qp(self, P, q, A, lo, hi, settings: OsqpSettings | None = None):
        """Solve one assembled QP.

        ``settings`` overrides the OSQP setup for THIS solve only; ``None``
        keeps the ``RepairConfig.osqp_*`` values (the frozen N0 settings --
        these defaults are exactly the values this method used to
        hard-code). The train24fix3 per-frame profile gating happens in
        ``solve_frame`` (``_frame_qp_settings``), which owns the frame
        context this method deliberately does not.
        """
        prob = self._osqp
        if settings is None:
            settings = OsqpSettings.from_config(self.cfg)
        lo = np.where(lo <= -INF / 2, -np.inf, lo)
        hi = np.where(hi >= INF / 2, np.inf, hi)
        prob.setup(P=P, q=q, A=A, l=lo, u=hi, verbose=False,
                   polish=bool(settings.polish),
                   eps_abs=float(settings.eps_abs),
                   eps_rel=float(settings.eps_rel),
                   max_iter=int(settings.max_iter),
                   scaling=int(settings.scaling))
        res = prob.solve()
        status = str(res.info.status)
        x = np.asarray(res.x) if res.x is not None else None
        if self.qp_trace is not None:
            # train24fix3 plan 5.1: read-only evidence of THIS solve. No
            # extra solve, no behavioural branch; the OSQP 1.1.3 info
            # fields (iter/prim_res/dual_res/obj_val/status_polish) are
            # copied as returned -- the historical pri_res/dua_res names
            # do not exist in this build.
            info = res.info
            self.qp_trace.append({
                "settings": asdict(settings),
                "status": status,
                "iter": int(info.iter),
                "prim_res": float(info.prim_res),
                "dual_res": float(info.dual_res),
                "obj_val": float(info.obj_val),
                "status_polish": int(info.status_polish),
                "x_returned": x is not None,
                "x_absmax": (float(np.abs(x).max())
                             if x is not None else None),
            })
            if self.qp_trace_arrays is not None:
                self.qp_trace_arrays.append({
                    "P": (np.asarray(P.todense()) if spa.issparse(P)
                          else np.array(P)),
                    "q": np.array(q),
                    "A": (np.asarray(A.todense()) if spa.issparse(A)
                          else np.array(A)),
                    "lo": np.array(lo), "hi": np.array(hi)})
        # OSQP returns garbage (huge) x on infeasibility: never trust it
        if x is None or not np.isfinite(x).all() or np.abs(x).max() > 1e6 or "infeasible" in status:
            return None, status
        return x, status

    def _c2_cut_refine(self, x, status, z, ctx, trust, slack, mu, settings):
        """norm_cut cutting-plane refinement (train24fix3 plan 7.2).

        After the FIRST solve of this iteration's QP, every site whose
        PREDICTED displacement d_j + J_j xi exceeds B gets one cut
        n'(d_j + J_j xi) <= B in its unit direction n = p/||p||; the QP is
        re-solved with the enlarged set. At most 8 re-solves per call and
        at most 8 directions per site (this linearization). Reaching either
        budget only RECORDS non-closure -- a cut-exhausted QP can never be
        read back as a real-norm pass (the nonlinear check rows are the
        only judge). Returns ``(x, status, n_extra_solves, stats)``;
        non-norm_cut profiles pass through with zero extra solves.
        """
        prof = _norm_cut_profile(self.c2_profile)
        if prof is None or x is None:
            return x, status, 0, None
        B = self.dt * prof.c2_norm_speed_mps
        tol = B * (1.0 + 1e-9)
        counts: dict[tuple[str, int], int] = {}
        stats = {"resolves": 0, "directions_added": 0,
                 "per_site_cap_hit": [], "resolve_cap_hit": False,
                 "nonfinite_skipped": 0, "final_pred_norm_max_m": None,
                 "nonclosed": False}
        for _round in range(8):
            xi = x[:N_TANGENT]
            added = False
            worst = 0.0
            for side in ("l", "r"):
                lin = self._c2_lin.get(side)
                if lin is None:
                    continue
                for j in range(3):
                    p = lin["d"][j] + lin["J"][3 * j : 3 * j + 3] @ xi
                    pn = float(np.linalg.norm(p))
                    worst = max(worst, pn)
                    if pn <= tol:
                        continue
                    key = (side, j)
                    if counts.get(key, 0) >= 8:
                        if key not in stats["per_site_cap_hit"]:
                            stats["per_site_cap_hit"].append(key)
                        continue
                    n_hat = p / pn
                    if not np.isfinite(n_hat).all():
                        stats["nonfinite_skipped"] += 1
                        continue
                    self._c2_cuts.append({"side": side, "site": j,
                                          "n": n_hat.copy(),
                                          "k": counts.get(key, 0)})
                    counts[key] = counts.get(key, 0) + 1
                    stats["directions_added"] += 1
                    added = True
            if worst > tol and not added:
                stats["nonclosed"] = True
            if not added:
                break
            (P, q, A, lo, hi), _ = self.build_qp(
                z, ctx, trust, slack=slack, mu=mu)
            stats["resolves"] += 1
            x, status = self.solve_qp(P, q, A, lo, hi, settings=settings)
            if x is None:
                break
        else:
            stats["resolve_cap_hit"] = True
        if x is not None:
            xi = x[:N_TANGENT]
            worst = 0.0
            for side in ("l", "r"):
                lin = self._c2_lin.get(side)
                if lin is None:
                    continue
                for j in range(3):
                    pn = float(np.linalg.norm(
                        lin["d"][j] + lin["J"][3 * j : 3 * j + 3] @ xi))
                    worst = max(worst, pn)
            stats["final_pred_norm_max_m"] = worst
            stats["nonclosed"] = worst > tol
        # merge into the frame-level account (plan 7.2: every round's cut
        # solves / failures / extra directions are accounted, never gated)
        acc = self._c2_cut_stats
        acc["resolves"] += stats["resolves"]
        acc["directions_added"] += stats["directions_added"]
        acc["nonfinite_skipped"] += stats["nonfinite_skipped"]
        acc["resolve_cap_hit"] = acc["resolve_cap_hit"] or stats["resolve_cap_hit"]
        for key in stats["per_site_cap_hit"]:
            if key not in acc["per_site_cap_hit"]:
                acc["per_site_cap_hit"].append(key)
        if stats["final_pred_norm_max_m"] is not None:
            acc["final_pred_norm_max_m"] = max(
                acc["final_pred_norm_max_m"] or 0.0,
                stats["final_pred_norm_max_m"])
        acc["nonclosed"] = acc["nonclosed"] or stats["nonclosed"]
        return x, status, stats["resolves"], stats

    def _frame_qp_settings(self, ctx: FrameContext) -> OsqpSettings | None:
        """Solver settings for THIS frame's QPs (train24fix3 plan 6.1).

        ``None`` keeps the ``RepairConfig.osqp_*`` defaults (N0). The
        attached profile's settings apply ONLY when the run is regen with
        C2 armed AND at least one side's rows are active at this frame;
        every other frame (standard mode, approach/pull-in, C2 disabled)
        solves under the N0 settings. The decision reads the frame's
        enable state ONLY -- never a failed candidate -- so it cannot
        update C2 state.
        """
        prof = self.c2_profile
        if prof is None or not self.cfg.regen_mode \
                or self.cfg.regen_hand_speed_mps <= 0.0:
            return None
        if any(hand_speed_rows_active(self.cfg, ctx, side) for side in ("l", "r")):
            return prof.solver
        return None

    @staticmethod
    def _trace_objective_parts(res: dict) -> dict:
        def _sq(r):
            if r is None:
                return None
            r = np.asarray(r)
            return float((r ** 2).sum()) if r.size else 0.0

        return {"pred_sq": _sq(res["r_pred"]), "temp_sq": _sq(res["r_temp"]),
                "mesh_sq": _sq(res["r_mesh"]), "hand_sq": _sq(res["r_hand"]),
                "stab_sq": _sq(res["r_stab"]), "mpjpe": float(res["mpjpe"])}

    def _trace_viol_blocks(self, val, lo, hi, scl) -> dict:
        """Normalized violation sum per layout block (pre / collision /
        post / probe) -- the same per-row formula as :meth:`violation`,
        split so the trace shows which family carries the residual."""
        val, lo, hi, scl = (np.asarray(a, dtype=float)
                            for a in (val, lo, hi, scl))
        n = len(val)
        n_probe = len(self.last_probe_rows)
        n_col = n - N_FIXED - N_POST_ROWS - n_probe
        v = np.maximum(np.maximum(lo - val, val - hi), 0.0) / scl
        finite = (np.abs(lo) < INF / 2) | (np.abs(hi) < INF / 2)
        v = np.where(finite, v, 0.0)
        c0 = N_FIXED + max(n_col, 0)
        return {"pre": float(v[:N_FIXED].sum()),
                "collision": float(v[N_FIXED:c0].sum()),
                "post": float(v[c0:c0 + N_POST_ROWS].sum()),
                "probe": float(v[c0 + N_POST_ROWS:].sum()),
                "n_rows": [int(N_FIXED), int(max(n_col, 0)), int(N_POST_ROWS),
                           int(n_probe)]}

    # -- success review ------------------------------------------------------

    def _pre_row_descriptors(self) -> list[str]:
        """Descriptors of the fixed-pre block, parallel to ``_fixed_pre_rows``."""
        if self._pre_desc is None:
            self._pre_desc = (
                [f"joint_limit[j{j}]" for j in range(7, 36)]
                + [f"joint_vel[j{j}]" for j in range(7, 36)]
                + ["root_lin_vel", "root_lin_vel[slot1]", "root_lin_vel[slot2]",
                   "root_ang_vel", "root_ang_vel[slot1]", "root_ang_vel[slot2]"]
                + [f"trust[k{k}]" for k in range(N_TANGENT)])
        return self._pre_desc

    def _post_row_descriptors(self) -> list[str]:
        """Descriptors of the fixed-post block, parallel to ``_fixed_post_rows``.

        norm_cut (train24fix3 plan 7.2) uses its OWN real-row semantics in
        the SAME 18-slot C2 tail: slot ``3j`` of a side carries the per-site
        NORM row, ``3j+1``/``3j+2`` are inactive placeholders -- the axis
        layout and all frozen offsets are untouched, the descriptors make
        the mode difference explicit."""
        norm_cut = _norm_cut_profile(self.c2_profile) is not None
        if self._post_desc is None or self._post_desc_norm != norm_cut:
            hand = [f"hand_anchor[{side}:{label}:{ax}]"
                    for side in ("l", "r") for label in HAND_SITES[side]
                    for ax in _AXES]
            foot_a = [f"foot_anchor[{side}:sole{i}:{ax}]"
                      for side in ("l", "r") for i in range(1, 6)
                      for ax in _AXES]
            foot_s = [f"foot_speed[{side}:sole{i}:{ax}]"
                      for side in ("l", "r") for i in range(1, 6)
                      for ax in _AXES]
            if norm_cut:
                hand_s = []
                for side in ("l", "r"):
                    for j, label in enumerate(HAND_SITES[side]):
                        hand_s.append(f"hand_norm[{side}:{label}]")
                        hand_s.append(f"hand_norm_off[{side}:{label}:slot1]")
                        hand_s.append(f"hand_norm_off[{side}:{label}:slot2]")
            else:
                hand_s = [f"hand_speed[{side}:{label}:{ax}]"
                          for side in ("l", "r") for label in HAND_SITES[side]
                          for ax in _AXES]
            self._post_desc = hand + foot_a + foot_s + hand_s
            self._post_desc_norm = norm_cut
        return self._post_desc

    def row_descriptors(self) -> tuple[list[str], list]:
        """Full-length descriptors of the LAST ``constraint_rows`` call.

        The nonlinear row order is fixed-pre, endpoint collision, fixed-post
        (hand anchors / foot anchors / foot speed / hand speed) and the probe
        block appended last -- the descriptors are built to match that order
        one-to-one, so a violating index maps to its OWN block. The old
        extraction indexed the collision+probe name list directly at offsets
        that ignored the 96 post rows, so every probe row was reported as
        whatever row happened to sit at that offset and the real probe
        violations were missed (R7).

        Returns ``(names, ids)``; ``ids`` entries are the stable pair
        identities for collision/probe rows, ``None`` for fixed rows.
        """
        pre = self._pre_row_descriptors()
        post = self._post_row_descriptors()
        names = pre + list(self.last_row_pairs) + post
        ids = [None] * len(pre) + list(self.last_row_ids) + [None] * len(post)
        for r in self._probe_rows_detail:
            names.append(f"probe:{r['name']}@a={r['alpha']:.3f}")
            ids.append(r["id"])
        return names, ids

    def _check_descriptors(self, names, ids) -> tuple[list[str], list]:
        """Descriptors parallel to a ``check_rows`` value array.

        ``check_rows`` returns fixed-pre + collision + fixed-post (no probe
        block), so the descriptors must be built in the same order.
        """
        pre, post = self._pre_row_descriptors(), self._post_row_descriptors()
        return (pre + list(names) + post,
                [None] * len(pre) + list(ids) + [None] * len(post))

    def _violating(self, val, lo, hi, names, ids=None,
                   offset: int = 0, end: int | None = None):
        """Violating rows ``[offset, end)`` as ``(names, ids, records)``.

        ``names``/``ids`` must be parallel to the FULL value array (see
        :meth:`row_descriptors`); ``end`` trims trailing blocks when the
        caller deliberately excludes them.
        """
        out_names, out_ids, recs = [], [], []
        end = len(val) if end is None else end
        for k in range(max(0, offset), min(end, len(names))):
            v, l_, h_ = val[k], lo[k], hi[k]
            if v < l_ - 1e-9 or v > h_ + 1e-9:
                nm = names[k]
                out_names.append(nm)
                if ids is not None and k < len(ids) and ids[k] is not None:
                    out_ids.append(ids[k])
                recs.append({"row": nm, "val_m": float(v),
                             "lo_m": float(l_), "hi_m": float(h_)})
        return out_names, out_ids, recs

    def success_review(self, z: np.ndarray, ctx: FrameContext) -> tuple[bool, dict]:
        """Unified review for EVERY successful return (F1).

        Re-forwards to the candidate state, rescans the FULL registry and
        checks all applicable constraints with ``check_rows`` -- uncapped by
        the QP row budget, so a success can never hide behind
        ``max_active_pairs``. Covers the early-return branch, budget
        exhaustion, trust collapse and the historical ``best_z``: the old
        code only reviewed the early return, letting ``best_feasible`` hand
        back a state whose collisions had gone stale.

        A violation invalidates the candidate and is fed back to the solver
        (``note_discovered_pairs``) so the next linearization can act on it;
        the caller decides whether to continue iterating or fail explicitly.
        """
        self._frame_pairs = None                     # full registry rescan
        try:
            val, lo, hi, scl, names, ids = self.check_rows(z, ctx)
        finally:
            # leave the cache cleared: the next linearization must rescan with
            # whatever ``note_discovered_pairs`` just added
            self._frame_pairs = None
        viol = self.violation(val, lo, hi, scl)
        info = {"viol": float(viol)}
        if viol <= 1e-6:
            return True, info
        dnames, dids = self._check_descriptors(names, ids)
        vnames, vids, recs = self._violating(val, lo, hi, dnames, dids,
                                             offset=N_FIXED,
                                             end=N_FIXED + len(names))
        info.update({"violated_rows": vnames, "rows": recs})
        if vids:
            self.note_discovered_pairs(vids)
        info["carried"] = vnames
        return False, info

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
        it_done = 0
        qp_calls = 0
        last_viol, last_names = 0.0, []
        last_qp_status = ""
        review = {}
        # train24fix3 (plan 6.1): THIS frame's OSQP settings -- the attached
        # profile's only when the frame carries active C2 rows, else None
        # (= the RepairConfig osqp_* defaults, the frozen N0 behaviour)
        settings = self._frame_qp_settings(ctx)
        # train24fix3 (plan 5.1): OFF by default. When self.trace is a dict,
        # every attempt of this frame appends a full record (iterations,
        # QP solves, line search, exit evidence); solve_qp appends to
        # self.qp_trace, which points at the CURRENT attempt's list. Pure
        # read-out: with self.trace None the control flow is unchanged.
        attempt_trace: dict | None = None
        it_rec: dict | None = None
        if self.trace is not None:
            attempt_trace = {
                "force_slack": bool(force_slack),
                "used_settings": (None if settings is None
                                  else asdict(settings)),
                "c2_sides": [s for s in ("l", "r")
                             if hand_speed_rows_active(cfg, ctx, s)],
                "mu": float(mu), "iter_budget": 0,
                "iterations": [], "qp": [], "qp_arrays": [],
                "exit": None}
            self.trace.setdefault("attempts", []).append(attempt_trace)
            self.qp_trace = attempt_trace["qp"]
            self.qp_trace_arrays = attempt_trace["qp_arrays"]
        try:
            # once the tight QP proves infeasible for this frame, stay in
            # restoration mode (re-trying the tight solve each iteration doubles
            # the cost without new information)
            restoration = force_slack
            max_iters = cfg.scp_max_iters + (cfg.scp_extra_restoration_iters if force_slack else 0)
            if attempt_trace is not None:
                attempt_trace["iter_budget"] = int(max_iters)
            # full registry scan for this frame's first linearization; carried
            # extras from previous attempts (``_extra_pairs``) are merged in. The
            # probe cache of the PREVIOUS attempt belongs to a different
            # linearization program and is dropped too (R3); the frame switch
            # drops it via ``clear_probe_pairs``.
            self._frame_pairs = None
            self._refresh_counter = 0
            self._probe_cache = None
            # train24fix3 norm_cut: cutting planes are per-LINEARIZATION --
            # a new frame (or a new pass's fresh attempt) never inherits
            # them; the accounting resets with them
            self._c2_cuts = []
            self._c2_lin = {}
            self._c2_cut_stats = {"resolves": 0, "directions_added": 0,
                                  "per_site_cap_hit": [], "resolve_cap_hit": False,
                                  "nonfinite_skipped": 0,
                                  "final_pred_norm_max_m": None,
                                  "nonclosed": False}
            # review R4: the loop-exit cause is recorded verbatim; it separates
            # "no acceptable step / trust collapsed below trust_radius_min" from
            # "budget exhausted" -- the historical status label cannot
            term_reason = ""
            for it in range(max_iters):
                it_done = it + 1
                # train24fix3 norm_cut: the previous iteration's cuts belong
                # to a different linearization point (plan 7.2 -- new state
                # clears the cache)
                self._c2_cuts = []
                res = self.evaluate(z, ctx)
                # feasibility/merit are judged against ACCEPTANCE bounds; the
                # proactive +2mm clearance lives only inside the QP rows
                val, lo, hi, scl = self.constraint_rows(z, ctx, trust, mode="accept")
                viol = self.violation(val, lo, hi, scl)
                last_viol = viol
                if viol <= 1e-6:
                    last_names = []
                else:
                    # descriptors are parallel to the FULL nonlinear value array
                    # (pre + collision + post + probe); the old extraction skipped
                    # the 96 post rows, mislabelling every probe row and losing
                    # the real probe violations (R7)
                    names_all, ids_all = self.row_descriptors()
                    if len(names_all) != len(val):
                        raise RuntimeError(
                            f"row descriptors out of sync with the value array "
                            f"({len(names_all)} vs {len(val)})")
                    last_names, _ids, _recs = self._violating(val, lo, hi, names_all, ids_all)
                merit = self.objective(res) + mu * viol
                feasible = viol <= 1e-6
                if feasible and merit < best_merit:
                    best_z, best_merit, best_feasible = z.copy(), merit, True
                if attempt_trace is not None:
                    it_rec = {
                        "it": it, "trust": float(trust),
                        "restoration": bool(restoration),
                        "z": z.copy(),
                        "merit": float(merit), "viol": float(viol),
                        "objective": self._trace_objective_parts(res),
                        "viol_blocks": self._trace_viol_blocks(val, lo, hi, scl),
                        "violated_rows": list(last_names),
                        "hand_speed": {s: dict(v) for s, v in
                                       self.last_hand_speed.items()},
                        "feasible": bool(feasible),
                        "best_merit": (float(best_merit)
                                       if best_feasible else None),
                        "qp_from": len(attempt_trace["qp"]),
                        "line_search": [], "trust_after": None,
                        "early_exit": None}
                    attempt_trace["iterations"].append(it_rec)
                if feasible and res["mpjpe"] <= cfg.task_mpjpe_tol_m and self._stationary(z, ctx):
                    # stage 1 + F1: a candidate could satisfy the budgeted rows
                    # while violating a pair they could not carry. EVERY success
                    # exit runs the uncapped full-registry review; a failure does
                    # NOT return -- it invalidates this best and feeds the
                    # discovered pairs into the next linearization.
                    ok, info = self.success_review(z, ctx)
                    review = info
                    if ok:
                        if it_rec is not None:
                            it_rec["early_exit"] = "converged_review_passed"
                            it_rec["trust_after"] = float(trust)
                        if attempt_trace is not None:
                            attempt_trace["exit"] = {
                                "accepted": True, "status": "solved",
                                "term_reason": "converged",
                                "it_done": int(it_done),
                                "qp_calls": int(qp_calls),
                                "trust": float(trust),
                                "used_slack": bool(used_slack)}
                        return self._result(True, z, it_done, used_slack, merit,
                                            qp_calls, trust, last_viol, last_names,
                                            qp_status=last_qp_status,
                                            term_reason="converged",
                                            iter_budget=max_iters)
                    if it_rec is not None:
                        it_rec["early_exit"] = "review_failed_candidate_invalidated"
                    if best_z is not None and np.array_equal(best_z, z):
                        best_z, best_merit, best_feasible = None, float("inf"), False

                if restoration:
                    (Ps, qs, As, los, his), _ = self.build_qp(
                        z, ctx, trust, slack=True, mu=mu)
                    qp_calls += 1
                    x_s, last_qp_status = self.solve_qp(Ps, qs, As, los, his,
                                                        settings=settings)
                    x_s, last_qp_status, n_cut, _cs = self._c2_cut_refine(
                        x_s, last_qp_status, z, ctx, trust, True, mu,
                        settings)
                    qp_calls += n_cut
                    if attempt_trace is not None and len(attempt_trace["qp"]):
                        qp_rec = attempt_trace["qp"][-1]
                        qp_rec["branch"] = "restoration"
                        qp_rec["rows_n"] = int(As.shape[0])
                        qp_rec["cut_resolves"] = int(n_cut)
                        if "qp_row_names" not in attempt_trace:
                            attempt_trace["qp_row_names"] = self._qp_row_names(
                                int(As.shape[0]), ctx)
                        qp_rec["slack"] = (None if x_s is None else {
                            "values": [float(v) for v in x_s[N_TANGENT:]],
                            "negative": int((np.asarray(x_s[N_TANGENT:]) < 0).sum()),
                            "nonfinite": int((~np.isfinite(
                                np.asarray(x_s[N_TANGENT:]))).sum()),
                            "rows": (dict(self.last_slack_rows)
                                     if self.last_slack_rows else None)})
                    if x_s is None:
                        trust *= cfg.shrink_factor
                        if it_rec is not None:
                            it_rec["trust_after"] = float(trust)
                            it_rec["qp_failed"] = True
                        if trust < cfg.trust_radius_min:
                            term_reason = "restoration_qp_failed_trust_min"
                            break
                        continue
                    used_slack = True
                    xi = x_s[:N_TANGENT]
                else:
                    (P, qv, A, lo_a, hi_a), _ = self.build_qp(
                    z, ctx, trust, slack=False, mu=mu)
                    qp_calls += 1
                    xi, last_qp_status = self.solve_qp(P, qv, A, lo_a, hi_a,
                                                       settings=settings)
                    xi, last_qp_status, n_cut, _cs = self._c2_cut_refine(
                        xi, last_qp_status, z, ctx, trust, False, mu,
                        settings)
                    qp_calls += n_cut
                    if attempt_trace is not None and len(attempt_trace["qp"]):
                        qp_rec = attempt_trace["qp"][-1]
                        qp_rec["branch"] = "tight"
                        qp_rec["rows_n"] = int(A.shape[0])
                        qp_rec["cut_resolves"] = int(n_cut)
                        if "qp_row_names" not in attempt_trace:
                            attempt_trace["qp_row_names"] = self._qp_row_names(
                                int(A.shape[0]), ctx)
                    if xi is None:
                        restoration = True
                        if it_rec is not None:
                            it_rec["qp_failed"] = True
                            it_rec["restoration_switch"] = True
                        continue
                xi_raw = xi
                xi = np.clip(xi, -trust * self.D, trust * self.D)
                if it_rec is not None:
                    it_rec["xi_raw"] = xi_raw.copy()
                    it_rec["xi_clipped"] = xi.copy()

                # backtracking line search: deep-penetration linearizations can
                # mispredict, and a full step may increase the true violation
                accepted_step = False
                for step_scale in (1.0, 0.5, 0.25, 0.125):
                    cand = retract_batch(self.ev.model, z[None], (step_scale * xi)[None])[0]
                    res_c = self.evaluate(cand, ctx)
                    val_c, lo_c, hi_c, scl_c = self.constraint_rows(cand, ctx, trust, mode="accept")
                    merit_c = self.objective(res_c) + mu * self.violation(val_c, lo_c, hi_c, scl_c)
                    if it_rec is not None:
                        it_rec["line_search"].append(
                            {"scale": float(step_scale),
                             "merit_c": float(merit_c),
                             "viol_c": float(self.violation(val_c, lo_c, hi_c, scl_c)),
                             "mpjpe_c": float(res_c["mpjpe"]),
                             "accepted": False})
                    if merit_c < merit:
                        z = cand
                        accepted_step = True
                        if it_rec is not None:
                            it_rec["line_search"][-1]["accepted"] = True
                        # stage 1: the step may have carried the state near pairs
                        # outside the frozen candidate set; schedule a partial
                        # refresh (fast families + carried extras) so later
                        # linearizations see them. The uncapped success review
                        # remains the correctness gate.
                        self._refresh_counter += 1
                        if merit_c < 0.5 * merit:
                            trust = min(cfg.trust_radius_max, trust * cfg.expand_factor)
                        break
                if it_rec is not None:
                    it_rec["trust_after"] = float(trust)
                    it_rec["z_next"] = z.copy()
                if not accepted_step:
                    trust *= cfg.shrink_factor
                    if it_rec is not None:
                        it_rec["trust_after"] = float(trust)
                        it_rec["step_rejected"] = True
                    if trust < cfg.trust_radius_min:
                        term_reason = "no_merit_step_trust_min"
                        break

            # budget exhausted / trust collapsed: report the ACTUAL iteration
            # count and slack usage (the old hardcoded ``scp_max_iters`` hid
            # early-exit branches and misreported slack on restoration exits;
            # stage 0, 2026-09-16). The historical best gets the SAME review as
            # the early return (F1): if the fresh scan exposes a violation, it is
            # explicitly rejected instead of being handed back unchecked.
            term_reason = term_reason or "budget_exhausted"
            if attempt_trace is not None:
                attempt_trace["exit"] = {
                    "accepted": None, "term_reason": term_reason,
                    "it_done": int(it_done), "qp_calls": int(qp_calls),
                    "trust": float(trust), "used_slack": bool(used_slack)}
            if best_feasible:
                res = self.evaluate(best_z, ctx)
                if res["mpjpe"] > cfg.task_mpjpe_tol_m:
                    if attempt_trace is not None:
                        attempt_trace["exit"].update(
                            {"accepted": False, "status": "task_quality"})
                    return self._result(False, best_z, it_done, used_slack, best_merit,
                                        qp_calls, trust, last_viol, last_names,
                                        status="task_quality", qp_status=last_qp_status,
                                        term_reason=term_reason, iter_budget=max_iters)
                ok, info = self.success_review(best_z, ctx)
                review = info
                if not ok:
                    if attempt_trace is not None:
                        attempt_trace["exit"].update(
                            {"accepted": False, "status": "review_failed"})
                    return self._result(False, best_z, it_done, used_slack, best_merit,
                                        qp_calls, trust, last_viol, last_names,
                                        status="review_failed", review=info,
                                        qp_status=last_qp_status,
                                        term_reason=term_reason, iter_budget=max_iters)
                if attempt_trace is not None:
                    attempt_trace["exit"].update(
                        {"accepted": True, "status": "solved"})
                return self._result(True, best_z, it_done, used_slack, best_merit,
                                    qp_calls, trust, last_viol, last_names,
                                    qp_status=last_qp_status,
                                    term_reason=term_reason, iter_budget=max_iters)
            if attempt_trace is not None:
                attempt_trace["exit"].update(
                    {"accepted": False, "status": "infeasible"})
            return self._result(False, z, it_done, used_slack, float("inf"), qp_calls,
                                trust, last_viol, last_names, status="infeasible",
                                review=review, qp_status=last_qp_status,
                                term_reason=term_reason, iter_budget=max_iters)
        finally:
            if attempt_trace is not None:
                # the trace pointers belong to THIS attempt only; a later
                # standalone solve_qp must not write into a finished record
                self.qp_trace = None
                self.qp_trace_arrays = None
                self.last_slack_rows = None

    def _result(self, accepted, z, it_done, used_slack, merit, qp_calls, trust,
                last_viol, last_names, status=None, review=None,
                qp_status: str = "", term_reason: str = "",
                iter_budget: int = 0) -> FrameResult:
        """FrameResult with the exit evidence F0 requires (trust, last
        constraint, QP exit status, budget truncation, review outcome) and
        the verbatim loop-exit cause (review R4 -- the status label stays
        historical)."""
        res = FrameResult(
            accepted, z, it_done, slack_used=used_slack,
            status=status or ("solved" if accepted else "max_iters"),
            merit=float(merit), qp_calls=qp_calls, trust=float(trust),
            last_viol=float(last_viol), violated_rows=list(last_names),
            truncated_rows=list(self.last_truncated_names),
            probe_rows=list(self.last_probe_rows),
            probe_truncated=list(self.last_probe_truncated),
            last_qp_status=str(qp_status), term_reason=str(term_reason),
            iter_budget=int(iter_budget))
        if review:
            res.diagnostics["review"] = review
        if self.last_truncated_ids:
            res.diagnostics["row_budget_truncated_ids"] = list(self.last_truncated_ids)
        if self._c2_cut_stats.get("resolves") or \
                self._c2_cut_stats.get("directions_added"):
            res.diagnostics["c2_cuts"] = {
                k: (list(v) if isinstance(v, list) else v)
                for k, v in self._c2_cut_stats.items()}
        return res

    def _stationary(self, z: np.ndarray, ctx: FrameContext) -> bool:
        """Prediction-consistent stationarity: near the prediction target."""
        r = _unit(tangent_error(z, ctx.z_pred), self.D)
        return float(np.abs(r).max()) < 0.05
