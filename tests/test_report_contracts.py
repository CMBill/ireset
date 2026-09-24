"""Report-side contracts: MPJPE series (R4) and provenance (R9).

Companion to ``test_cost_contracts.py`` (R6 totals) and
``test_repair_contracts.py`` (solver evidence): these tests pin the quality
metric and the run's provenance record --

- R4: ``task_mpjpe_series`` is THE per-frame series (one FK forward per
  frame) shared by the percentile stats and the quintile growth; reading the
  evaluator without forwarding reported the last state's error for every t
  (sub10: growth -0.2242 m reported instead of ~-0.0030 m).
- R9: an experiment is identified by its EFFECTIVE config (round-tripped,
  hashed, recorded after all overrides), never by ``--note`` or a directory
  name; the standalone provenance CLI declares a snapshot, not effective
  parameters.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig, ReferenceCondition  # noqa: E402
from ireset.repair.pipeline import KPT_IDX, RepairPipeline  # noqa: E402
from ireset.repair.validation import ValidationConfig  # noqa: E402
from regenerate_teachers import task_mpjpe_series, task_mpjpe_stats  # noqa: E402
from segment_quality_report import mpjpe_growth  # noqa: E402
from experiment_provenance import (  # noqa: E402
    config_roundtrip,
    config_sha256,
    effective_record,
    provenance,
)
from test_repair import BOX_FAR, self_condition, standing_scene  # noqa: E402


@pytest.fixture(scope="module")
def pipeline() -> RepairPipeline:
    return RepairPipeline(cfg=RepairConfig())


# --- R4: per-frame MPJPE series ----------------------------------------------------


def _moving_scene_with_reference(pipeline: RepairPipeline, T: int = 8,
                                 base_m: float = 0.0, step_m: float = 0.0,
                                 axis: int = 0):
    """Standing scene whose ROOT translates ``0.02 m`` per frame, with the
    reference keypoints = the robot's own FK keypoints (possibly shifted by
    ``base_m + step_m * t`` along ``axis``).

    A moving robot is what makes the field checkable: reading the evaluator
    without forwarding leaves the LAST frame's state in place, so a per-frame
    matched reference would still report a spurious growth.
    """
    scene = standing_scene(T)
    scene[:, 0] += 0.02 * np.arange(T)
    cond = self_condition(pipeline, scene)
    human = cond.human_kpts_world.copy()
    for t in range(T):
        human[t, :, axis] += base_m + step_m * t
    return scene, ReferenceCondition(human, scene[:, 36:43].copy(),
                                     cond.ref_schedule)


def test_mpjpe_series_is_per_frame_and_shared(pipeline) -> None:
    """R4: the series forwards per frame and is the SINGLE source of both the
    percentile stats and the quintile growth (a known per-frame offset must
    reproduce the analytic ramp)."""
    ev = pipeline.ev
    T, step = 8, 0.01
    scene, cond = _moving_scene_with_reference(pipeline, T=T, step_m=step)
    # a stale evaluator state must not leak into the result
    ev.forward(scene[-1, 0:36] + np.r_[0.2, 0.0, 0.0, np.zeros(33)], BOX_FAR)
    series = task_mpjpe_series(ev, scene, cond)
    assert np.allclose(series, step * np.arange(T), atol=1e-9)

    stats = task_mpjpe_stats(ev, scene, cond)
    growth = mpjpe_growth(ev, scene, cond)
    assert stats["p95"] == pytest.approx(float(np.percentile(series, 95)), abs=1e-4)
    assert growth["p95"] == pytest.approx(stats["p95"], abs=1e-9)
    assert growth["max"] == pytest.approx(stats["max"], abs=1e-9)
    k = T // 5
    assert growth["q1_mean"] == pytest.approx(float(series[:k].mean()), abs=1e-4)
    assert growth["q5_mean"] == pytest.approx(float(series[-k:].mean()), abs=1e-4)
    assert growth["growth"] == pytest.approx(
        float(series[-k:].mean() - series[:k].mean()), abs=1e-4)
    assert growth["growth"] == pytest.approx(step * (T - k), abs=1e-4)


def test_mpjpe_growth_zero_for_per_frame_matched_reference(pipeline) -> None:
    """R4 detector: a robot whose reference is its own per-frame pose has zero
    MPJPE and zero growth. The pre-fix implementation (no forward inside the
    loop) reports the LAST state against every frame's reference, which turns
    the robot's own motion into a spurious growth."""
    ev = pipeline.ev
    T = 8
    scene, cond = _moving_scene_with_reference(pipeline, T=T)
    series = task_mpjpe_series(ev, scene, cond)
    assert np.abs(series).max() < 1e-9
    assert mpjpe_growth(ev, scene, cond)["growth"] == pytest.approx(0.0, abs=1e-9)

    # the pre-fix reading: one stale state, no per-frame forward
    ev.forward(scene[-1, 0:36], scene[-1, 36:43])
    stale = np.array([float(np.linalg.norm(
        ev.site_positions(ev.kpt_ids) - cond.human_kpts_world[t][KPT_IDX],
        axis=-1).mean()) for t in range(T)])
    k = T // 5
    stale_growth = float(stale[-k:].mean() - stale[:k].mean())
    assert not np.allclose(stale, series)
    assert abs(stale_growth) > 0.01, "the stale reading must be visibly wrong"


def test_mpjpe_single_frame_behaviour(pipeline) -> None:
    """A 1-frame clip: k = 1, growth is 0 by definition, q1 == q5."""
    scene, cond = _moving_scene_with_reference(pipeline, T=1, base_m=0.03)
    growth = mpjpe_growth(pipeline.ev, scene, cond)
    assert growth["growth"] == pytest.approx(0.0)
    assert growth["q1_mean"] == growth["q5_mean"] == pytest.approx(0.03, abs=1e-4)


# --- R9: effective configuration of the running instance ----------------------------


def load_strict():
    from ireset.repair.config_io import load_mode_configs

    return load_mode_configs("strict")


def test_effective_record_roundtrips_and_identifies_experiments(tmp_path) -> None:
    """R9: the record carries the instance's effective configs (round-tripped
    through JSON), keeps the defaults separate, and distinguishes the
    pre-registered variants by their CONFIG -- not by note or directory."""
    gate = tmp_path / "task_quality_gates.yaml"
    gate.write_text("task_quality_gates:\n  mpjpe_tol_m: 0.113\n")
    rcfg, vcfg = load_strict()
    rec = effective_record(
        replace(rcfg, regen_mode=True, regen_hand_stab_weight=0.1,
                task_mpjpe_tol_m=0.113),
        replace(vcfg, hand_gap_m=0.02),
        argv=["regenerate_teachers.py", "--mode", "regen", "--stab-weight", "0.1"],
        mode="regen", validation_mode="strict", demos=["sub10_largebox_000"],
        frames=0, note="same-note", task_gate_yaml=str(gate),
        screening={"min_hand_recall": 0.5, "min_ref_frames": 10})
    assert rec["effective_repair_config"]["regen_hand_stab_weight"] == 0.1
    assert rec["effective_repair_config"]["regen_mode"] is True
    assert rec["effective_validation_config"]["hand_gap_m"] == 0.02
    assert rec["default_repair_config"]["regen_hand_stab_weight"] == 0.0
    assert rec["is_default_repair_config"] is False
    assert rec["is_default_validation_config"] is False
    assert rec["task_gate_yaml"] == str(gate)
    assert rec["task_gate_yaml_sha256"] is not None
    assert rec["task_mpjpe_tol_m"] == pytest.approx(0.113)
    assert rec["argv"][:3] == ["regenerate_teachers.py", "--mode", "regen"]
    # round-trip: the exported config reconstructs the instance exactly
    assert config_roundtrip(RepairConfig(**rec["effective_repair_config"])) \
        == RepairConfig(**rec["effective_repair_config"])
    back = RepairConfig(**rec["effective_repair_config"])
    assert back.regen_hand_stab_weight == 0.1 and back.regen_mode is True
    assert ValidationConfig(**rec["effective_validation_config"]).hand_gap_m == 0.02

    variants = {}
    for stab in (0.0, 0.1, 1.0):
        for mode in ("regen", "standard"):
            cfg = replace(rcfg, regen_mode=(mode == "regen"),
                          regen_hand_stab_weight=stab)
            r = effective_record(cfg, vcfg, argv=[], mode=mode,
                                 validation_mode="strict", demos=[],
                                 frames=0, note="same-note")
            variants[(stab, mode)] = r["effective_config_sha256"]
    assert len(set(variants.values())) == 6, variants
    assert rec["effective_config_sha256"] not in variants.values()
    # the note must NOT be part of the identity
    other_note = effective_record(replace(rcfg, regen_mode=True,
                                          regen_hand_stab_weight=0.1),
                                  vcfg, argv=[], mode="regen",
                                  validation_mode="strict", demos=[],
                                  frames=0, note="a totally different note")
    assert other_note["effective_config_sha256"] == \
        effective_record(replace(rcfg, regen_mode=True,
                                 regen_hand_stab_weight=0.1), vcfg,
                         argv=[], mode="regen", validation_mode="strict",
                         demos=[], frames=0,
                         note="same-note")["effective_config_sha256"]
    assert config_sha256(rcfg, vcfg) == config_sha256(rcfg, vcfg)


def test_standalone_provenance_is_declared_a_snapshot() -> None:
    """R9: without a running instance the CLI must not present loader configs
    as an experiment's effective parameters."""
    snap = provenance(demos=["sub10_largebox_000"], note="cli")
    assert snap["kind"] == "snapshot"
    assert snap["configs_are_loader_defaults_not_effective"] is True
    assert snap["inputs"][0]["demo"] == "sub10_largebox_000"
    assert json.dumps(snap, default=float)  # serializable
