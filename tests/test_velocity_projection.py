"""V1: bounded candidate normalization + strict joint-velocity re-check
(train24fix plan section 4.1).

Counterexamples frozen from the D-stage evidence:

- the reviewer's float case ``q_prev=0.8``: a naive ``clip`` lands on the
  boundary ``1.2000000000000002`` whose SAME-FORMULA speed is
  12.000000000000004 rad/s -- strictly over. The normalizer must return a
  state that passes ``abs(diff)/(1/fps) <= 12.0`` bit-for-bit, not by an
  epsilon;
- sub7's frozen exceedance 12.000000401362618 rad/s (joint 26): the D-V
  binding showed it comes from a SOLVER-ADOPTED candidate (tight QP, not
  only slack), needs 1.34e-8 rad -- far inside the 4e-7 rad budget;
- over-budget candidates are never partially projected: skip falls through
  to the solve escalation, solve candidates become unusable, backtrack
  alphas are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import (  # noqa: E402
    RepairPipeline,
    normalize_joint_velocity,
    strict_joint_speeds,
)
from ireset.repair.scp import FrameCheck, FrameResult  # noqa: E402
from test_repair import BOX_FAR, self_condition, standing_scene  # noqa: E402

CFG = RepairConfig()
DT = 1.0 / CFG.fps
LIMDT = DT * CFG.joint_velocity_limit_radps      # 0.4 rad
BUDGET = DT * CFG.joint_velocity_limit_radps * 1e-6   # 4e-7 rad


def _states(delta_j26: float):
    """Standing-scene pair whose frame-1 joint 26 differs by ``delta_j26``."""
    scene = standing_scene(2)
    z_prev = scene[0, 0:36].copy()
    z = scene[1, 0:36].copy()
    z[7 + 26] = z_prev[7 + 26] + delta_j26
    return z_prev, z


# --- pure normalizer --------------------------------------------------------------


def test_reviewer_float_case_naive_clip_fails_normalizer_passes() -> None:
    z_prev = np.zeros(36)
    z_prev[7 + 20] = 0.8
    z = np.zeros(36)
    z[7 + 20] = 0.8 + LIMDT + 1e-9
    naive = np.clip(z[7 + 20], 0.8 - LIMDT, 0.8 + LIMDT)
    # the failure mode itself: the clipped boundary is strictly over
    assert abs(naive - 0.8) / DT > CFG.joint_velocity_limit_radps

    info = normalize_joint_velocity(z, z_prev, CFG)
    assert info["ok"] is True
    assert info["n_changed"] == 1
    assert float(strict_joint_speeds(info["z"], z_prev, CFG).max()) \
        <= CFG.joint_velocity_limit_radps        # SAME formula, strictly
    # root pose untouched, all other joints bitwise identical
    assert info["z"][0:7].tolist() == z[0:7].tolist()
    assert np.array_equal(np.delete(info["z"], 7 + 20),
                          np.delete(z, 7 + 20))
    assert info["max_delta_rad"] <= BUDGET


def test_sub7_frozen_value_normalized_exactly() -> None:
    z_prev = np.zeros(36)
    z = np.zeros(36)
    z[7 + 26] = 12.000000401362618 * DT
    assert float(strict_joint_speeds(z, z_prev, CFG).max()) \
        == pytest.approx(12.000000401362618, rel=0, abs=1e-12)
    info = normalize_joint_velocity(z, z_prev, CFG)
    assert info["ok"] is True
    out = float(strict_joint_speeds(info["z"], z_prev, CFG).max())
    assert out <= CFG.joint_velocity_limit_radps
    assert info["max_delta_rad"] == pytest.approx(1.3378753896398088e-08,
                                                  rel=1e-3)
    assert info["max_delta_rad"] <= BUDGET


def test_budget_boundary_allowed_and_beyond_rejected() -> None:
    z_prev = np.zeros(36)
    # inside the budget is allowed. ("exactly == budget" is not a stable
    # float fixture: the boundary itself may need a nextafter inward, which
    # pushes the achieved need one ulp above -- the normalizer then refuses,
    # which is the safe side. 0.9x is unambiguous.)
    z_at = np.zeros(36)
    z_at[7 + 5] = LIMDT + BUDGET * 0.9
    assert normalize_joint_velocity(z_at, z_prev, CFG)["ok"] is True
    # clearly beyond the budget: no partial projection, no state returned
    z_over = np.zeros(36)
    z_over[7 + 5] = LIMDT + BUDGET * 1.5
    info = normalize_joint_velocity(z_over, z_prev, CFG)
    assert info["ok"] is False
    assert info["reason"] == "velocity_projection_budget"
    assert "z" not in info and info["needed_rad"] > BUDGET


def test_no_partial_projection_when_any_joint_over_budget() -> None:
    z_prev = np.zeros(36)
    z = np.zeros(36)
    z[7 + 2] = LIMDT + 1e-9            # micro: inside the budget
    z[7 + 9] = LIMDT + 5e-7            # over the budget
    info = normalize_joint_velocity(z, z_prev, CFG)
    assert info["ok"] is False and info["reason"] == "velocity_projection_budget"
    assert "z" not in info
    # both inside: both changed, both strictly legal afterwards
    z[7 + 9] = LIMDT + 1e-9
    info2 = normalize_joint_velocity(z, z_prev, CFG)
    assert info2["ok"] is True and info2["n_changed"] == 2
    assert float(strict_joint_speeds(info2["z"], z_prev, CFG).max()) \
        <= CFG.joint_velocity_limit_radps


def test_negative_direction_and_clean_passthrough() -> None:
    z_prev = np.zeros(36)
    z = np.zeros(36)
    z[7 + 2] = -LIMDT - 1e-9
    info = normalize_joint_velocity(z, z_prev, CFG)
    assert info["ok"] is True
    assert float(strict_joint_speeds(info["z"], z_prev, CFG).max()) \
        <= CFG.joint_velocity_limit_radps
    clean = np.zeros(36)
    clean[7 + 10] = 0.3
    pas = normalize_joint_velocity(clean, z_prev, CFG)
    assert pas["ok"] is True and pas["n_changed"] == 0
    assert np.array_equal(pas["z"], clean)          # bitwise untouched


# --- strict gate in _frame_valid ---------------------------------------------------


def test_frame_valid_strict_gate_rejects_micro_exceedance() -> None:
    """A micro-over joint diff passes the normalized violation sum (3.3e-8
    <= 1e-6) but MUST be rejected by the independent same-formula gate, with
    the unrounded value in the evidence row."""
    pipe = RepairPipeline(cfg=RepairConfig())
    z_prev, z_micro = _states(12.000000401362618 * DT)
    # the row-sum verdict alone would accept it:
    val, lo, hi, scl, _names, _ids = pipe.solver.check_rows(
        z_micro, _ctx(pipe, z_prev, z_micro))
    assert pipe.solver.violation(val, lo, hi, scl) <= 1e-6
    chk = pipe.frame_valid(z_micro, _ctx(pipe, z_prev, z_micro))
    assert not chk.ok and chk.reason == "state_constraints"
    row = next(r for r in chk.rows if r["row"] == "joint_vel[j33]")
    assert row["val"] == pytest.approx(12.000000401362618, rel=0, abs=1e-9)
    assert row["hi"] == CFG.joint_velocity_limit_radps


def _ctx(pipe, z_prev, z, t: int = 1, self_ref: bool = True):
    """Test context; ``self_ref=True`` derives the reference keypoints from
    the candidate's own FK so the task gate is not the failure under test."""
    from ireset.repair.scp import FrameContext

    if self_ref:
        pipe.ev.forward(z, BOX_FAR)
        ref = pipe.ev.site_positions(pipe.ev.kpt_ids).copy()
    else:
        ref = np.zeros((15, 3))
    return FrameContext(t=t, box_pose=BOX_FAR, z_pred=z.copy(), z_prev=z_prev,
                        ref_kpts_world=ref, mesh_L=None,
                        delta_ref=None, mesh_fixed=None, hand_anchor={},
                        foot_anchor={}, prev_hand_local={}, prev_sole_world={})


# --- entry coverage ----------------------------------------------------------------


def test_skip_entry_normalized_and_strictly_passing_end_to_end() -> None:
    """Two standing frames whose only difference is a micro-over joint step:
    frame 1 skips, the ADOPTED state is the normalized one, the strict final
    formula passes on the output, and the accounting counts one modified
    output frame."""
    pipe = RepairPipeline(cfg=RepairConfig())
    scene = standing_scene(2)
    scene[1, 7 + 26] = scene[0, 7 + 26] + 12.000000401362618 * DT
    cond = self_condition(pipe, scene)
    res = pipe.repair_trajectory(scene, cond)
    assert res.accepted, res.diagnostics.get("skip_reason")
    assert res.diagnostics["frame_status"][1] == "skipped"
    q0, q1 = res.trajectory[0, 7:36], res.trajectory[1, 7:36]
    assert float(np.abs(q1 - q0).max() / DT) <= CFG.joint_velocity_limit_radps
    vp = res.diagnostics["velocity_projection"]
    assert vp["checks"] >= 1 and vp["normalized"] >= 1
    assert vp["output_frames_modified"] == 1
    assert vp["max_delta_rad"] <= BUDGET
    # every other field of the adopted frame is the prediction bit-for-bit
    expect = scene[1].copy()
    expect[7 + 26] = q1[26]
    assert np.array_equal(res.trajectory[1], expect)


def test_skip_denied_over_budget_falls_through_to_solve() -> None:
    """An over-budget prediction step must not be adopted by skipping: the
    frame goes to the solve escalation (skip denied), and whatever comes out
    still faces the strict gate."""
    pipe = RepairPipeline(cfg=RepairConfig())
    scene = standing_scene(2)
    scene[1, 7 + 26] = scene[0, 7 + 26] + LIMDT + 2e-6   # needs > budget
    cond = self_condition(pipe, scene)
    res = pipe.repair_trajectory(scene, cond)
    if res.accepted:
        # the solver repaired it to a strictly legal state
        assert res.diagnostics["frame_status"][1] != "skipped"
        q0, q1 = res.trajectory[0, 7:36], res.trajectory[1, 7:36]
        assert float(np.abs(q1 - q0).max() / DT) <= CFG.joint_velocity_limit_radps
    else:
        assert res.status == "rejected_frame"
        assert res.diagnostics["frame_status"][1] != "skipped"
    assert res.diagnostics["velocity_projection"]["budget_failures"] >= 1


def test_solve_attempt_over_budget_candidate_unusable(monkeypatch) -> None:
    """A solver candidate that needs more than the budget never enters the
    acceptance checks: the attempt fails with the budget reason and the
    existing escalation handles it."""
    pipe = RepairPipeline(cfg=RepairConfig())
    z_prev, z_far = _states(LIMDT + 1e-3)
    crafted = FrameResult(accepted=True, z=z_far.copy(), status="solved",
                          qp_calls=3, iterations=3)
    crafted.frame_check = FrameCheck(ok=True)   # a stale passing check
    monkeypatch.setattr(pipe.solver, "solve_frame",
                        lambda *a, **k: crafted)
    attempts: list[dict] = []
    checks: list[dict] = []
    res, chk, bt, _bti, rec = pipe._solve_attempt(
        z_far, "prediction", _ctx(pipe, z_prev, z_far), 1, z_prev, 0,
        attempts, checks, [])
    assert bt is None
    assert not chk.ok and chk.reason == "velocity_projection_budget"
    assert rec["velocity_normalization"]["ok"] is False
    assert res.z.tolist() == z_far.tolist()     # not partially projected


def test_solve_attempt_normalized_invalidates_cached_check(monkeypatch) -> None:
    """When normalization changes ``res.z`` the cached ``frame_check`` is
    invalidated: the acceptance check re-runs on the NEW state (the crafted
    stale check object cannot validate it)."""
    pipe = RepairPipeline(cfg=RepairConfig())
    z_prev, z_micro = _states(12.000000401362618 * DT)
    crafted = FrameResult(accepted=True, z=z_micro.copy(), status="solved",
                          qp_calls=1, iterations=1)
    stale = FrameCheck(ok=True)
    crafted.frame_check = stale
    monkeypatch.setattr(pipe.solver, "solve_frame", lambda *a, **k: crafted)
    res, chk, _bt, _bti, _rec = pipe._solve_attempt(
        z_micro, "prediction", _ctx(pipe, z_prev, z_micro), 1, z_prev, 0,
        [], [], [])
    assert res.diagnostics.get("velocity_normalized") is True
    assert res.frame_check is not stale
    # the recomputed check judged the NORMALIZED state and accepted it
    assert chk.ok
    assert float(strict_joint_speeds(res.z, z_prev, CFG).max()) \
        <= CFG.joint_velocity_limit_radps


def test_backtrack_alpha_over_budget_skipped_next_tried() -> None:
    """Backtracking candidates are adoption candidates too: an alpha whose
    retracted step exceeds the budget is skipped, a legal one can be
    adopted, and a MICRO-over alpha gets normalized before adoption."""
    pipe = RepairPipeline(cfg=RepairConfig())
    z_prev, _z = _states(0.0)                 # standing frame 0, identity quat
    # xi such that alpha=0.75 lands micro-over the limit, 0.5 lands legal
    xi_joint = (LIMDT + 2e-9) / 0.75
    res = FrameResult(accepted=True, z=z_prev.copy(), status="solved")
    res.z[7 + 26] = z_prev[7 + 26] + xi_joint
    res.frame_check = FrameCheck(ok=False, reason="transition_interpolated")
    _res2, path, bt_info = pipe._transition_backtrack(
        res, _ctx(pipe, z_prev, res.z), z_prev)
    assert path == "repaired_bt"
    assert bt_info["adopted"] == 0.75
    assert 0.75 in bt_info["probed"]
    # the adopted state is the NORMALIZED 0.75 candidate: strictly legal
    assert res.diagnostics.get("velocity_normalized") is True
    assert float(strict_joint_speeds(res.z, z_prev, CFG).max()) \
        <= CFG.joint_velocity_limit_radps
    # and a step that is over the budget at EVERY alpha adopts nothing
    res_b = FrameResult(accepted=True, z=z_prev.copy(), status="solved")
    res_b.z[7 + 26] = z_prev[7 + 26] + LIMDT / 0.25 * 1.1   # every alpha over
    res_b.frame_check = FrameCheck(ok=False, reason="transition_interpolated")
    _r, path_b, info_b = pipe._transition_backtrack(
        res_b, _ctx(pipe, z_prev, res_b.z), z_prev)
    assert path_b is None
    assert info_b["adopted"] is None and len(info_b["probed"]) == 3
