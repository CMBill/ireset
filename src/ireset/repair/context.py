"""Repair context: reference condition, repair configuration, contact state machine.

Implements plan sections 8.1/8.4/8.5. The contact state machine is driven by
the *reference* schedule but commits only from **accepted** states: anchors
are candidate values while solving and become locked after the frame passes
its checks. Hand anchors live in the box **geom frame** (per contact site),
foot anchors in the **world frame** (per sole site), matching the validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ireset.embodied.contact_sites import ContactSchedule
from ireset.geometry.distance_adapter import largebox_hull


@dataclass
class ReferenceCondition:
    """Minimal repair-time condition (full schema comes with M3 data layer).

    ``human_kpts_world`` is the [T, 52, 3] reference already aligned to the
    target scene (teacher-NPZ convention); ``box_pose`` is the [T, 7] target
    box trajectory (project convention) — a fixed task input the repairer
    must not modify.
    """

    human_kpts_world: np.ndarray
    box_pose: np.ndarray
    ref_schedule: ContactSchedule
    fps: float = 30.0

    @property
    def T(self) -> int:
        return self.human_kpts_world.shape[0]


@dataclass
class RepairConfig:
    # objective weights (plan 14.3 starting point)
    prediction_weight: float = 1.0
    laplacian_weight: float = 0.1
    temporal_weight: float = 0.1
    regularization_weight: float = 0.001
    # step scale matrix D (per-unit trust scaling) and trust schedule
    step_scale_root_t_m: float = 0.02
    step_scale_root_r_rad: float = 0.05
    step_scale_joint_rad: float = 0.1
    trust_radius_init: float = 1.0
    trust_radius_min: float = 1e-3
    trust_radius_max: float = 8.0
    shrink_factor: float = 0.5
    expand_factor: float = 1.8
    # P0.1 L0 (2026-09-13): 15+15 -> 40+40. Teacher failures were mostly
    # `infeasible` within the 15-iteration budget; compute-only change.
    scp_max_iters: int = 40
    scp_extra_restoration_iters: int = 40  # fallback solves get a longer budget
    # velocity bounds (shared with the validator; calibrated M2)
    joint_velocity_limit_radps: float = 12.0
    base_lin_velocity_limit_mps: float = 1.8
    base_ang_velocity_limit_radps: float = 7.0
    # collision constraint margins
    noncontact_clearance_m: float = 0.002
    contact_band_gap_m: float = 0.01          # hand-geom vs box upper band
    foot_contact_gap_m: float = 0.005         # sole-vs-ground upper band (validator gate)
    contact_band_pen_m: float = 0.002         # lower band (<= validator gate)
    activation_margin_m: float = 0.05         # pairs closer than this become active
    max_active_pairs: int = 48
    # anchor tolerances (per axis; <= validator gates)
    hand_anchor_tol_m: float = 0.01
    foot_anchor_tol_m: float = 0.01
    foot_speed_mps: float = 0.02
    # contact establishment gates (geometric, at commit time)
    hand_establish_dist_m: float = 0.012
    foot_establish_gap_m: float = 0.006
    # P0.1 L1 (2026-09-13): quasi-static gate -- sliding feet touch (band
    # rows) but never lock world anchors; aligned with the validator's
    # foot_entry_speed_mps ("support := quasi-static")
    foot_anchor_establish_speed_mps: float = 0.05
    # task quality gate: M2 working value. The teacher violates the support
    # model (slides ~1.4cm/frame), so repaired feet stay pinned while the
    # reference slides -> keypoint error grows to ~16cm; deep-tail frames with
    # sliding hands reach ~0.35. M2 working value 0.40; M3 calibrates on the
    # validation split (plan 8.4).
    task_mpjpe_tol_m: float = 0.40
    # feasibility restoration
    slack_penalty: float = 1e4
    penetration_scale_m: float = 0.002   # violation units for collision rows
    enable_slack_fallback: bool = True
    # suffix rollback
    suffix_retry_window: int = 5
    suffix_retries: int = 1
    fps: float = 30.0
    # teacher regeneration mode (Path B, p0 plan section 7): during
    # reference grasp intervals the hand-box pair carries an ACTIVE upper
    # bound regardless of current distance -- the hovering teacher warm
    # start gets pulled onto the box surface (per-pair geometric bands
    # alone never engage at 15-33mm). Hand anchors are then governed by
    # the reference schedule alone: the prediction-driven release would
    # fire every frame (the warm start itself hovers). Feet keep the
    # standard semantics (P0-fix strict 6/6 shows they already hold).
    regen_mode: bool = False
    regen_hand_contact_gap_m: float = 0.008   # < hand_enter 0.01 so the validator certifies
    # approach corridor: the active bound closes by at least this rate per
    # frame from the previous ACCEPTED distance -- a hard gap from interval
    # start is single-frame infeasible (4-5 cm of Cartesian hand motion),
    # and an unbounded reach trips the transition check
    regen_closing_rate_mps: float = 0.15
    # offline regeneration runs quality-critical solves: the default merit
    # penalty (mu ~12) lets the tracking objective outrank sub-mm violation
    # reductions and the convergence tail stalls above the 1e-6 feasibility
    # threshold; regeneration multiplies mu (standard mode unchanged)
    regen_merit_multiplier: float = 30.0
    # dedicated hand-keypoint tracking objective (regen only): the robot
    # hand-site centroid chases the REFERENCE HUMAN hand centroid, which
    # is quasi-static in the box frame during grasp. The mesh term cannot
    # take this role -- raising its weight to hold the hands breaks the
    # quasi-static foot model (it also tracks the human's foot slides)
    # interpolation-dip margin coefficient for solver band rows
    # (margin = min(6mm, max(0.5mm, coef * frame_motion^2)))
    dip_margin_coef: float = 0.3
    # transition-probe rows (F4, 2026-09-22, plan section 7.1): the solver
    # carries a linearized anti-penetration row at each interior
    # interpolation state of the frame's transition. Separate row budget --
    # probe rows never displace endpoint/anchor rows; over budget the most
    # violated rows survive and the rest are reported (the pipeline's
    # uncapped probe check remains the arbiter, so a dropped row cannot
    # become a silent accept).
    max_probe_rows: int = 12
    # a probe row is built for a pair whose probe distance is within this
    # margin above the penetration floor (and for anything violating)
    probe_row_margin_m: float = 0.003
    regen_hand_track_weight: float = 25.0
    regen_hand_track_scale_m: float = 0.02
    # grasp-stability soft target (F4, 2026-09-22, plan section 7.2): during a
    # reference grasp the robot hand's BOX-frame displacement must follow the
    # reference's displacement; the residual (step - ref_step)/scale is
    # penalized with this weight. The teacher reference itself moves (release
    # transients, grip adjustments), so the term tracks the reference's
    # motion instead of forcing zero -- a quasi-static reference then
    # suppresses solver chatter that the certification entry gate rejects.
    # Pre-registered weight comparison: off (0.0) / 0.1 / 1.0.
    regen_hand_stab_weight: float = 0.0
    regen_hand_stab_scale_m: float = 0.01
    # hard box-frame speed rows for in-band hands (C2, train24fix2 section 4):
    # per-axis bound dt * this value on the box-frame displacement of the 3
    # hand sites (3 sites x 3 axes = 9 rows per side). sqrt(3) * 0.02598 =
    # 0.04499868 < hand_entry_speed_mps 0.05, so a candidate that exactly
    # satisfies the rows cannot be speed-gated out of certification.
    # Re-enabled (was 0.0 after the 2026-09-15 negative result) with a NEW
    # enable criterion: the rows turn on only inside a reference hold
    # interval once the previous accepted FULL-SCENE hand/box distance is
    # within regen_hand_contact_gap_m, stay on for the whole interval and
    # release at interval end (C2HandSpeedState) -- the two 2026-09-15
    # failure modes were the PROJECTED prev_hand_dist 6mm gate flapping at
    # its threshold and a bound that forbade the reference's own grip
    # changes during approach; the interval-scoped enable exempts pull-in
    # and the certification-compatible bound addresses the chatter.
    regen_hand_speed_mps: float = 0.02598
    # OSQP setup (train24fix3 plan section 4.1 task 3): the values
    # ``FrameSolver.solve_qp`` historically hard-coded in ``prob.setup``,
    # now configurable. The defaults ARE the frozen N0 behaviour (polish
    # off, eps 3e-5, 2500 iterations, internal scaling 10); a train24fix3
    # profile may override them ONLY on frames whose C2 rows are active
    # (``FrameSolver._frame_qp_settings``) -- standard mode, approach
    # frames and C2-off runs always solve under these defaults.
    osqp_polish: bool = False
    osqp_eps_abs: float = 3e-5
    osqp_eps_rel: float = 3e-5
    osqp_max_iter: int = 2500
    osqp_scaling: int = 10
    # regeneration does NOT lock hand point anchors: a +-10mm box-frame
    # anchor under box rotation is repeatedly tipped out by the
    # hover-tracking objective (marginal 0.1-1mm infeasibilities). Contact
    # is maintained by the hard gap row every frame; box-frame pose
    # stability comes from keypoint tracking (the reference hand is
    # quasi-static on the box during grasp) and is JUDGED by the strict
    # validator's anchor-drift gate at screening time.


#: effectors tracked by the state machine
EFFECTORS = ("left_hand", "right_hand", "left_foot", "right_foot")


class C2HandSpeedState:
    """Per-side enable/release state of the regen hand box-frame speed rows
    (train24fix2 plan section 4.2).

    The state is a pure function of the ACCEPTED prefix and the INPUT
    reference schedule -- never of the candidate being checked, and never of
    the independent validator's own intervals (feeding supervision back as an
    input condition is what the plan forbids).  One instance per trajectory
    pass: a suffix retry or a new segment builds a new one, so cross-pass
    leakage is impossible by construction.

    Per side and frame (section 4.2 table):

    - t=0 or no valid previous full scene  -> rows off, no state inheritance;
    - reference does not want hold         -> rows off, state released;
    - interval identity changed            -> old state cleared, re-judged;
    - enabled earlier in the SAME interval -> rows on (holds through 6-8mm
      fluctuations and boundary-rounding residue by construction);
    - not yet enabled, prev full-scene distance <= gap -> enable now;
    - not yet enabled, distance > gap      -> pull-in exemption, corridor
      drives the approach.
    """

    SIDES = (("l", "left_hand"), ("r", "right_hand"))

    def __init__(self, ref_schedule: ContactSchedule, T: int, gap_m: float) -> None:
        self.gap_m = float(gap_m)
        self._frame_iv: dict[str, list] = {}
        for _side, eff in self.SIDES:
            ids = [None] * T
            ivs = [iv for iv in ref_schedule.intervals if iv.effector == eff]
            for i, iv in enumerate(ivs):
                for t in range(iv.start, iv.end):
                    ids[t] = i
            self._frame_iv[eff] = ids
        self.enabled = {eff: False for _side, eff in self.SIDES}
        self._interval = {eff: None for _side, eff in self.SIDES}

    def step(self, eff: str, t: int, ref_hold: bool,
             prev_scene_dist: float | None) -> tuple[bool, str]:
        """Enable decision for frame ``t``; commits the state (safe: it
        depends only on the accepted prefix and the reference input)."""
        iv = self._frame_iv[eff][t]
        if t == 0 or prev_scene_dist is None:
            self.enabled[eff], self._interval[eff] = False, None
            return False, "off_t0_or_no_prev"
        if not ref_hold:
            self.enabled[eff], self._interval[eff] = False, None
            return False, "off_no_ref_released"
        if self.enabled[eff] and self._interval[eff] != iv:
            self.enabled[eff], self._interval[eff] = False, None
        if self.enabled[eff]:
            return True, "hold"
        if prev_scene_dist <= self.gap_m:
            self.enabled[eff], self._interval[eff] = True, iv
            return True, "enable"
        return False, "pull_in_exempt"


class ContactStateMachine:
    """Per-effector approach/hold/release with anchors committed on acceptance."""

    def __init__(self, ref_schedule: ContactSchedule, T: int) -> None:
        self.ref_schedule = ref_schedule
        self.T = T
        self.holding: dict[str, bool] = {e: False for e in EFFECTORS}
        self.anchors: dict[str, np.ndarray | None] = {e: None for e in EFFECTORS}
        self._ref_mask = {e: ref_schedule.mask(e, T) for e in EFFECTORS}

    def reference_wants_hold(self, effector: str, t: int) -> bool:
        return bool(self._ref_mask[effector][t])

    def candidate_anchors(
        self, effector: str, hand_local: np.ndarray | None, sole_world: np.ndarray | None
    ) -> np.ndarray | None:
        """Anchors for a frame being solved: locked ones if holding, else None.

        For an effector entering hold this frame, the pipeline passes the
        *current* per-site positions as candidate anchors (satisfied by
        construction at the linearization point); they are only locked after
        the frame is accepted (``commit``).
        """
        if self.holding[effector]:
            return self.anchors[effector]
        if effector.endswith("hand"):
            if hand_local is not None and self.anchors[effector] is None:
                return hand_local  # [3 sites, 3] candidate
        else:
            if sole_world is not None and self.anchors[effector] is None:
                return sole_world  # [5, 3] candidate
        return self.anchors[effector]

    def commit(self, effector: str, t: int, hand_local=None, sole_world=None) -> None:
        """Lock anchors from an accepted state when reference (and geometry) agree."""
        wants = self.reference_wants_hold(effector, t)
        if effector.endswith("hand"):
            if wants and not self.holding[effector] and hand_local is not None:
                self.holding[effector] = True
                self.anchors[effector] = hand_local.copy()
            elif not wants and self.holding[effector]:
                self.holding[effector] = False
                self.anchors[effector] = None
        else:
            if wants and not self.holding[effector] and sole_world is not None:
                self.holding[effector] = True
                self.anchors[effector] = sole_world.copy()
            elif not wants and self.holding[effector]:
                self.holding[effector] = False
                self.anchors[effector] = None

    def release(self, effector: str) -> None:
        """Drop an anchor mid-interval (P0.1 L1): a sliding prediction foot
        degrades the contact to touching-without-anchor. The reference
        interval itself is unchanged -- only the freeze demand disappears."""
        if self.holding[effector]:
            self.holding[effector] = False
            self.anchors[effector] = None

    def rollback_to(self, t: int) -> None:
        """Release anchors established at or after frame t (suffix rollback)."""
        for e in EFFECTORS:
            if self.holding[e] and self.anchors[e] is not None:
                ivs = [iv for iv in self.ref_schedule.intervals if iv.effector == e]
                if any(iv.start >= t for iv in ivs):
                    self.holding[e] = False
                    self.anchors[e] = None
