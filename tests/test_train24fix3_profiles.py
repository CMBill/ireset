"""train24fix3 commit 1: profile registry, OSQP settings plumbing, per-frame
profile gating and the read-only trace hook (plan sections 4.1/5/6.1).

Failure modes this file pins:

- a registry edit with unknown fields/values must be REJECTED at load time
  (a record file must never masquerade as run settings);
- the RepairConfig ``osqp_*`` defaults must equal the N0 setup -- the
  historical ``prob.setup`` hard-codes -- so "no profile" is provably the
  frozen behaviour;
- the profile's solver settings may only reach frames with ACTIVE C2 rows;
  approach frames, C2-off runs and standard mode keep N0;
- the trace must not change ANY solver output (candidates, iterations, QP
  call counts, exit evidence) and its pointers must not survive the solve;
- v2 forensic products (pre-train24fix3 payloads) round-trip through
  ``configs_from_record`` ONLY via the registered N0 backfill; a v2 record
  already carrying the new fields is rejected as unproducible.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.profiles import (  # noqa: E402
    OsqpSettings,
    get_profile,
    load_profiles,
)
from ireset.repair.scp import FrameContext  # noqa: E402
from ireset.repair.validation import ValidationConfig  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_repair import standing_scene  # noqa: E402

REGEN_C2 = replace(RepairConfig(), regen_mode=True)
REGEN_OFF = replace(RepairConfig(), regen_mode=True, regen_hand_speed_mps=0.0)
REGISTRY = (Path(__file__).resolve().parents[1] / "configs" / "repair"
            / "train24fix3_profiles.yaml")


# --- 1. registry ------------------------------------------------------------


def test_registry_loads_and_matches_plan_matrix() -> None:
    profiles = load_profiles()
    assert sorted(profiles) == ["N0", "N1", "N2", "N3", "R1", "R2", "R3",
                               "S1", "S2"]
    n0 = profiles["N0"]
    assert n0.solver == OsqpSettings() == OsqpSettings.from_config(RepairConfig())
    n1 = profiles["N1"]
    assert (n1.solver.polish and n1.solver.eps_abs == 1e-9
            and n1.solver.eps_rel == 1e-9 and n1.solver.max_iter == 20000
            and n1.solver.scaling == 10)
    assert n1.c2_mode == "axis" and n1.c2_row_precondition == "none"
    assert n1.c2_slack_penalty == "legacy"
    n2 = profiles["N2"]
    assert n2.solver == n1.solver                      # base inheritance
    assert n2.c2_row_precondition == "divide_by_bound"
    assert n2.c2_slack_penalty == "scaled_10000B"
    n3 = profiles["N3"]
    assert n3.solver == n1.solver and n3.c2_row_precondition == "divide_by_bound"
    assert n3.c2_slack_penalty == "mu"
    s1, s2 = profiles["S1"], profiles["S2"]
    assert s1.c2_mode == "norm_cut" and s1.c2_norm_speed_mps == 0.045
    assert s1.solver == n1.solver and s1.c2_slack_penalty == "mu"
    assert s2.c2_norm_speed_mps == 0.049
    for s in (s1, s2):
        v = s.c2_norm_speed_mps
        assert v * (1 + 1e-6) < 0.05


def test_registry_rejects_unknown_and_illegal(tmp_path) -> None:
    good = REGISTRY.read_text()

    def write(text: str) -> Path:
        p = tmp_path / "profiles.yaml"
        p.write_text(text)
        return p

    with pytest.raises(ValueError, match="unknown train24fix3 profile"):
        get_profile("N9")
    bad = good.replace("  N0:", "  N0:\n    surprise_key: 1\n", 1)
    with pytest.raises(ValueError, match="unknown keys"):
        load_profiles(write(bad))
    bad = good.replace(
        "solver: {polish: false, eps_abs: 3.0e-5",
        "solver: {warm_start: 1, polish: false, eps_abs: 3.0e-5", 1)
    with pytest.raises(ValueError, match="unknown solver keys"):
        load_profiles(write(bad))
    with pytest.raises(ValueError, match="schema"):
        load_profiles(write(good.replace("train24fix3_profiles_v1", "v0")))
    bad = good.replace("  N1:", "  N1:\n    c2_norm_speed_mps: 0.045\n", 1)
    with pytest.raises(ValueError, match="axis mode"):
        load_profiles(write(bad))
    bad = good.replace("c2_norm_speed_mps: 0.049", "c2_norm_speed_mps: 0.05")
    with pytest.raises(ValueError, match="entry gate"):
        load_profiles(write(bad))
    bad = good.replace("base: N1", "base: N9")
    with pytest.raises(ValueError, match="unknown base"):
        load_profiles(write(bad))
    bad = good + "\n  X1:\n    base: X2\n  X2:\n    base: X1\n"
    with pytest.raises(ValueError, match="circular"):
        load_profiles(write(bad))
    bad = good.replace("max_iter: 2500", "max_iter: 0")
    with pytest.raises(ValueError, match="positive"):
        load_profiles(write(bad))


def test_repair_config_defaults_are_n0() -> None:
    cfg = RepairConfig()
    assert (cfg.osqp_polish is False and cfg.osqp_eps_abs == 3e-5
            and cfg.osqp_eps_rel == 3e-5 and cfg.osqp_max_iter == 2500
            and cfg.osqp_scaling == 10)


# --- 2. per-frame gating ----------------------------------------------------


def _c2_ctx(pipe, z, box, enabled=True):
    pipe.ev.forward(z, box)
    cur = pipe.ev.hand_local("l")
    return FrameContext(
        t=1, box_pose=box.copy(), z_pred=z.copy(), z_prev=None,
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={},
        prev_hand_local={"l": cur.copy()},
        prev_sole_world={}, box_pose_prev=box.copy(),
        ref_hold={"left_hand": True}, prev_hand_dist={"l": 0.012},
        frame_motion_m=0.02,
        c2_hand_enabled={"l": enabled}, c2_hand_reason={"l": "hold"},
        prev_hand_scene_dist={"l": 0.0012})


def test_frame_qp_settings_gate_on_active_c2_only() -> None:
    pipe = RepairPipeline(cfg=REGEN_C2)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.006)
    pipe.ev.forward(z, box)
    ctx_on = _c2_ctx(pipe, z, box, enabled=True)
    ctx_off = _c2_ctx(pipe, z, box, enabled=False)
    n1 = get_profile("N1")

    pipe.solver.c2_profile = None
    assert pipe.solver._frame_qp_settings(ctx_on) is None
    pipe.solver.c2_profile = n1
    assert pipe.solver._frame_qp_settings(ctx_on) == n1.solver
    assert pipe.solver._frame_qp_settings(ctx_off) is None
    pipe_off = RepairPipeline(cfg=REGEN_OFF, c2_profile=n1)
    pipe_off.ev.forward(z, box)
    assert pipe_off.solver._frame_qp_settings(ctx_on) is None
    pipe_std = RepairPipeline(cfg=RepairConfig(), c2_profile=n1)
    pipe_std.ev.forward(z, box)
    assert pipe_std.solver._frame_qp_settings(ctx_on) is None


# --- 3. solve_qp settings reach the setup and are traced ---------------------


def _qp_args():
    import scipy.sparse as spa

    n = 3
    P = spa.csc_matrix(np.eye(n))
    q = np.zeros(n)
    A = spa.csc_matrix(np.vstack([np.eye(n), -np.eye(n)]))
    lo = np.full(2 * n, -1.0)
    hi = np.full(2 * n, 1.0)
    return P, q, A, lo, hi


def test_solve_qp_default_settings_equal_n0(monkeypatch) -> None:
    import osqp

    pipe = RepairPipeline(cfg=REGEN_C2)
    seen: dict = {}
    orig = osqp.OSQP.setup

    def spy(self, **kw):
        seen.update(kw)
        return orig(self, **kw)

    monkeypatch.setattr(osqp.OSQP, "setup", spy)
    x, status = pipe.solver.solve_qp(*_qp_args())
    assert x is not None
    assert seen["polish"] is False and seen["eps_abs"] == 3e-5
    assert seen["eps_rel"] == 3e-5 and seen["max_iter"] == 2500
    assert seen["scaling"] == 10


def test_solve_qp_settings_override_reaches_setup(monkeypatch) -> None:
    import osqp

    pipe = RepairPipeline(cfg=REGEN_C2)
    seen: dict = {}
    orig = osqp.OSQP.setup

    def spy(self, **kw):
        seen.update(kw)
        return orig(self, **kw)

    monkeypatch.setattr(osqp.OSQP, "setup", spy)
    n1 = get_profile("N1")
    pipe.solver.qp_trace = []
    x, status = pipe.solver.solve_qp(*_qp_args(), settings=n1.solver)
    assert seen["polish"] is True and seen["eps_abs"] == 1e-9
    assert seen["max_iter"] == 20000
    assert x is not None
    rec = pipe.solver.qp_trace[-1]
    assert rec["settings"]["polish"] is True
    assert rec["iter"] >= 1 and "status_polish" in rec
    pipe.solver.qp_trace = None
    assert pipe.solver.qp_trace is None


# --- 4. trace purity ----------------------------------------------------------


def test_trace_on_off_purity() -> None:
    pipe = RepairPipeline(cfg=REGEN_C2)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.006)
    pipe.ev.forward(z, box)
    ctx = _c2_ctx(pipe, z, box, enabled=True)
    z0 = z.copy()
    z0[20] += 0.12                     # displaced start: real iterations

    pipe_off = RepairPipeline(cfg=REGEN_C2)
    pipe_off.ev.forward(z, box)
    res_off = pipe_off.solver.solve_frame(z0.copy(), ctx)

    pipe_on = RepairPipeline(cfg=REGEN_C2)
    pipe_on.ev.forward(z, box)
    pipe_on.solver.trace = {"attempts": []}
    res_on = pipe_on.solver.solve_frame(z0.copy(), ctx)

    assert res_on.accepted == res_off.accepted
    assert res_on.status == res_off.status
    assert res_on.term_reason == res_off.term_reason
    assert res_on.iterations == res_off.iterations
    assert res_on.qp_calls == res_off.qp_calls
    assert np.array_equal(res_on.z, res_off.z)
    assert res_on.last_qp_status == res_off.last_qp_status
    assert res_on.merit == res_off.merit

    trace = pipe_on.solver.trace
    assert trace["attempts"], "trace recorded no attempts"
    att = trace["attempts"][-1]
    assert att["exit"] is not None and att["exit"]["qp_calls"] == res_on.qp_calls
    assert len(att["iterations"]) == res_on.iterations
    assert att["c2_sides"] == ["l"]
    assert att["used_settings"] is None          # no profile attached
    if res_on.qp_calls:
        qp = att["qp"][-1]
        assert set(qp) >= {"settings", "status", "iter", "prim_res",
                           "dual_res", "obj_val", "status_polish"}
        assert att["qp_arrays"][-1]["A"].ndim == 2
    assert pipe_on.solver.qp_trace is None
    assert pipe_on.solver.qp_trace_arrays is None


# --- 5. v2 forensic payload backfill ------------------------------------------


def _v2_record(payload_repair: dict) -> dict:
    full = {k: v for k, v in asdict(RepairConfig()).items()
            if not k.startswith("osqp_")}
    full.update(payload_repair)
    return {"format": "train24fix2_reviewfix_audit_v2",
            "effective": {"repair": full,
                          "validation": asdict(ValidationConfig())}}


def test_configs_from_record_backfills_n0_only() -> None:
    from train24fix2_audit import configs_from_record

    rec = _v2_record({})
    cfg, _vcfg = configs_from_record(rec)
    assert cfg.osqp_polish is False and cfg.osqp_max_iter == 2500
    with pytest.raises(ValueError, match="v2"):
        configs_from_record(_v2_record({"osqp_polish": True}))
    # an unknown payload key is not constructible -- the round-trip check
    # rejects it (VALUE tampering is the replay deep_diff's job, as always)
    with pytest.raises(TypeError):
        bad = _v2_record({})
        bad["effective"]["repair"]["bogus_key"] = 1
        configs_from_record(bad)


def test_effective_record_carries_profile() -> None:
    from experiment_provenance import effective_record

    prof = get_profile("N1")
    rec = effective_record(
        RepairConfig(), ValidationConfig(), argv=["x"], mode="regen",
        validation_mode="strict", demos=["d"], frames=0,
        profile={"name": prof.name, "resolved": prof.as_record()})
    assert rec["profile"]["name"] == "N1"
    assert rec["profile"]["resolved"]["solver"]["polish"] is True
    rec2 = effective_record(
        RepairConfig(), ValidationConfig(), argv=["x"], mode="regen",
        validation_mode="strict", demos=["d"], frames=0)
    assert rec2.get("profile") is None
