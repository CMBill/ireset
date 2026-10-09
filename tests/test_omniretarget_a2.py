"""A2 source-comparison contracts: layouts, binding, verdict gates (§3.4).

- layout conversion is explicit per source (published [quat,pos] /
  upstream [pos,quat] / project passthrough); unknown layouts and
  degenerate quaternions are refused at load;
- binding: bit-exact human + target box + fps/T passes; every tampered
  input fails;
- evaluate_local_preferred: each frozen gate has a counterexample
  (penetration regression, task regression, foot regression, missing
  evidence -> incomplete, empty both-hands denominator -> no improvement
  score, P gap p95 zero -> no relative-drop score).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.data import omniretarget_sources as orc  # noqa: E402
from ireset.eval.teacher_comparison import (  # noqa: E402
    DemoMetrics, SideFootMetrics, SideHandMetrics, _longest_true_run, _p95,
    evaluate_local_preferred,
)

PILOT = ["subA_largebox_001", "subB_largebox_002", "subC_largebox_003",
         "subD_largebox_004", "subE_largebox_005", "subF_largebox_006"]


# --- layouts -----------------------------------------------------------------

def _raw_row(layout: str, rng) -> np.ndarray:
    q1 = rng.normal(size=4)
    q1 /= np.linalg.norm(q1)
    q2 = rng.normal(size=4)
    q2 /= np.linalg.norm(q2)
    row = np.zeros(43)
    row[7:36] = rng.uniform(-0.5, 0.5, 29)
    if layout == "published":
        row[0:4], row[4:7] = q1, [0.1, 0.2, 0.8]
        row[36:40], row[40:43] = q2, [0.3, 0.0, 0.2]
    else:  # upstream_generated / project_scene
        row[3:7], row[0:3] = q1, [0.1, 0.2, 0.8]
        row[39:43], row[36:39] = q2, [0.3, 0.0, 0.2]
    return row


def test_layout_conversions_are_source_explicit():
    rng = np.random.default_rng(3)
    pub = _raw_row("published", rng)
    proj = orc.convert_teacher_row(pub, "published")
    assert proj[0:3].tolist() == [0.1, 0.2, 0.8]          # pos first
    np.testing.assert_allclose(proj[3:7], pub[0:4])        # quat wxyz
    np.testing.assert_allclose(proj[36:39], pub[40:43])
    np.testing.assert_allclose(proj[39:43], pub[36:40])

    up = _raw_row("upstream_generated", rng)
    proj2 = orc.convert_teacher_row(up, "upstream_generated")
    np.testing.assert_array_equal(proj2, up)               # MuJoCo order == project

    passthrough = orc.convert_teacher_row(proj, "project_scene")
    np.testing.assert_array_equal(passthrough, proj)

    with pytest.raises(ValueError):
        orc.convert_teacher_row(pub, "quat_last_guessed")


def test_load_teacher_scene_rejects_degenerate_quaternion(tmp_path):
    rng = np.random.default_rng(4)
    q = np.stack([_raw_row("published", rng) for _ in range(5)])
    q[2, 36:40] = 0.0  # degenerate box quaternion
    np.savez(tmp_path / "bad.npz", qpos=q,
             human_joints=rng.normal(size=(5, 52, 3)), fps=30.0)
    with pytest.raises(ValueError, match="quaternion"):
        orc.load_teacher_scene(tmp_path / "bad.npz", "published")


# --- binding -----------------------------------------------------------------

def _scenes(rng, T=6):
    scene = np.zeros((T, 43))
    scene[:, 0:3] = rng.normal(size=(T, 3))
    qq = rng.normal(size=(T, 4))
    scene[:, 3:7] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
    scene[:, 7:36] = rng.uniform(-0.4, 0.4, (T, 29))
    qb = rng.normal(size=(T, 4))
    scene[:, 39:43] = qb / np.linalg.norm(qb, axis=1, keepdims=True)
    scene[:, 36:39] = rng.normal(size=(T, 3))
    human = rng.normal(size=(T, 52, 3))
    return scene, human


def test_check_binding_passes_bit_exact_and_fails_each_tamper():
    rng = np.random.default_rng(5)
    s, h = _scenes(rng)
    ok = orc.check_binding("d", s.copy(), h.copy(), 30.0, s.copy(), h.copy(), 30.0)
    assert ok.passed and ok.human_max_abs_diff == 0.0

    s2 = s.copy(); s2[3, 36] += 1e-6                       # target box moved
    assert not orc.check_binding("d", s2, h, 30.0, s, h, 30.0).passed
    h2 = h.copy(); h2[1, 0, 0] += 1e-6                     # reference changed
    assert not orc.check_binding("d", s, h2, 30.0, s, h, 30.0).passed
    assert not orc.check_binding("d", s[:-1], h[:-1], 30.0, s, h, 30.0).passed
    assert not orc.check_binding("d", s, h, 25.0, s, h, 30.0).passed


# --- verdict gates -----------------------------------------------------------

def _metrics(demo, *, pen=0.001, task=0.10, foot=True, both=None, valid=True):
    foot_m = {s: (SideFootMetrics(30, 0.003, 0.01, 0.01) if foot else None)
              for s in ("l", "r")}
    hand_m = {s: SideHandMetrics(30, 0.02, 0.5, 0.5, 10, 5, 0.02) for s in ("l", "r")}
    both_default = {"denominator_frames": 30, "valid_coverage": 0.40,
                    "gap_series_p95_m": 0.05}
    return DemoMetrics(
        demo=demo, T=100, fps=30.0, valid=valid, penetration_max_m=pen,
        penetration_violation_ratio=0.0, task_mpjpe_p95_m=task,
        foot=foot_m, hand=hand_m, both=(both_default if both is None else both),
    )


def _run(u_over=None, p_over=None, u_valid=None):
    p = {d: _metrics(d, **(p_over.get(d, {}) if p_over else {})) for d in PILOT}
    u = {d: _metrics(d, **(u_over.get(d, {}) if u_over else {})) for d in PILOT}
    if u_valid:
        for d, v in u_valid.items():
            u[d] = DemoMetrics(d, -1, float("nan"), valid=False, error=v)
    binding = {d: True for d in PILOT}
    return evaluate_local_preferred(PILOT, p, u, binding), p, u


def test_local_preferred_when_coverage_gains_on_four_demos():
    u_over = {d: {"both": {"denominator_frames": 30,
                           "valid_coverage": 0.55, "gap_series_p95_m": 0.05}}
              for d in PILOT[:4]}
    verdict, *_ = _run(u_over=u_over)
    assert verdict["verdict"] == "local_preferred", verdict["gates"]
    assert verdict["gates"]["g4_hand_improvement"]["n_coverage_gain_demos"] == 4


def test_gap_p95_drop_counts_and_zero_P_baseline_does_not():
    u_over = {d: {"both": {"denominator_frames": 30,
                           "valid_coverage": 0.40, "gap_series_p95_m": 0.02}}
              for d in PILOT[:3]}
    verdict, *_ = _run(u_over=u_over)   # P p95=0.05 -> 60% drop on 3 demos
    assert verdict["verdict"] == "local_preferred"

    p_over = {d: {"both": {"denominator_frames": 30,
                           "valid_coverage": 0.40, "gap_series_p95_m": 0.0}}
              for d in PILOT}
    verdict2, *_ = _run(u_over=u_over, p_over=p_over)  # P p95 == 0: no score
    assert verdict2["verdict"] == "no_quality_gain"


def test_each_regressive_gate_blocks_preference():
    pen_bad = {PILOT[0]: {"pen": 0.9}}
    v, *_ = _run(u_over=pen_bad)
    assert not v["gates"]["g3_penetration_and_feet"]["passed"]
    assert v["verdict"] == "no_quality_gain"

    task_bad = {PILOT[1]: {"task": 0.5}}
    v, *_ = _run(u_over=task_bad)
    assert not v["gates"]["g2_task_mpjpe"]["passed"]

    foot_bad = {PILOT[2]: {"foot": False}}  # U0 lost reference foot frames
    v, *_ = _run(u_over=foot_bad)
    assert not v["gates"]["g3_penetration_and_feet"]["passed"]
    assert v["gates"]["g3_penetration_and_feet"]["per_demo"][PILOT[2]]["foot"] != []


def test_p_without_reference_support_frames_skips_foot_gate():
    p_over = {PILOT[3]: {"foot": False}}
    u_over = {PILOT[3]: {"foot": False}, PILOT[0]: {
        "both": {"denominator_frames": 30, "valid_coverage": 0.55,
                 "gap_series_p95_m": 0.05}}}
    # foot gate skips demos without P reference frames; only 1 demo gains -> g4 fails
    v, *_ = _run(u_over=u_over, p_over=p_over)
    assert v["gates"]["g3_penetration_and_feet"]["per_demo"][PILOT[3]]["passed"] is True
    assert not v["gates"]["g4_hand_improvement"]["passed"]


def test_missing_or_invalid_evidence_is_incomplete_never_dropped():
    v, p, u = _run(u_valid={PILOT[2]: "generation_failed"})
    assert v["verdict"] == "incomplete"
    assert PILOT[2] in u and not u[PILOT[2]].valid     # row kept
    v2, *_ = _run()
    v2 = evaluate_local_preferred(
        PILOT, p, u, {d: (d != PILOT[1]) for d in PILOT})  # binding failed for one
    assert v2["verdict"] == "incomplete"


def test_empty_both_hands_denominator_gives_no_improvement_score():
    empty = {"denominator_frames": 0, "valid_coverage": None, "gap_series_p95_m": None}
    v, *_ = _run(u_over={d: {"both": empty} for d in PILOT},
                 p_over={d: {"both": empty} for d in PILOT})
    assert v["gates"]["g4_hand_improvement"]["n_coverage_gain_demos"] == 0
    assert v["gates"]["g4_hand_improvement"]["n_gap_drop_demos"] == 0
    assert v["verdict"] == "no_quality_gain"


def test_helpers():
    assert _longest_true_run(np.array([1, 1, 0, 1, 1, 1])) == 3
    assert _longest_true_run(np.zeros(4, dtype=bool)) == 0
    assert np.isnan(_p95(np.array([])))
    assert _p95(np.arange(100)) == pytest.approx(94.05, abs=0.1)
