"""A2 sixth-round review counterexamples (independent review, 2026-10-10).

The first three tests fail on the pre-fix implementation; the last two are
positive regression guards for invariants both implementations hold:

1  the freeze leg verified EVERY consumed data dir against the FULL frozen
   list — a split retry (five demos from the original dir, the sixth from a
   retry dir holding only that demo's PT) was rejected for the five files
   the retry dir never served, although every consumed input matched the
   freeze, coverage and bindings were intact;
2  ``_both_denominator`` truncated with ``int()`` BEFORE the sign check —
   -0.5 read as a legal zero and 30.9 as 30, so P=30 / U0=30.9 "agreed"
   and the verdict could still be complete_evidence=True / local_preferred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ  # noqa: E402
from ireset.eval.teacher_comparison import (  # noqa: E402
    DemoMetrics, SideFootMetrics, SideHandMetrics, evaluate_local_preferred,
)
from ireset.paths import path_for_record  # noqa: E402

PILOT = [f"subA_largebox_{i:03d}" for i in range(1, 7)]
HANDS = ("l", "r")


# --- helpers (kept self-contained per suite convention) -----------------------


def _metrics(demo, *, pen=0.001, pen_ratio=0.0, task=0.10, hand_drift=0.02,
             foot_drift=0.01, failed=(), both=None, valid=True):
    foot_m = {s: SideFootMetrics(30, 0.003, 0.01, foot_drift, 29) for s in HANDS}
    hand_m = {s: SideHandMetrics(30, 0.02, 0.5, 0.5, 10, 5, hand_drift)
              for s in HANDS}
    both_default = {"denominator_frames": 30, "valid_coverage": 0.40,
                    "gap_series_p95_m": 0.05}
    return DemoMetrics(
        demo=demo, T=100, fps=30.0, valid=valid, penetration_max_m=pen,
        penetration_violation_ratio=pen_ratio, task_mpjpe_p95_m=task,
        foot=foot_m, hand=hand_m,
        both=(both_default if both is None else both),
        validator_failed_checks=list(failed),
        ref_support_frames={"l": 30, "r": 30}, ref_grasp_frames={"l": 30, "r": 30},
    )


def _gains(demo):
    return {"both": {"denominator_frames": 30, "valid_coverage": 0.55,
                     "gap_series_p95_m": 0.05}}


def _scenes(rng, T=5):
    scene = np.zeros((T, 43))
    scene[:, 0:3] = rng.normal(size=(T, 3)) * 0.1
    qq = rng.normal(size=(T, 4)); qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    scene[:, 3:7] = qq
    scene[:, 7:36] = rng.uniform(-0.3, 0.3, (T, 29))
    qb = rng.normal(size=(T, 4)); qb /= np.linalg.norm(qb, axis=1, keepdims=True)
    scene[:, 39:43] = qb
    scene[:, 36:39] = [0.5, 0.0, 0.2]
    return scene, rng.normal(size=(T, 52, 3))


def _write_pub_npz(rng, npz_dir, demo):
    """Published P only (for demos without U0 artifacts)."""
    scene, human = _scenes(rng)
    q = scene.copy()                       # published [qw,qx,qy,qz,x,y,z]
    q[:, 0:4], q[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
    q[:, 36:40], q[:, 40:43] = scene[:, 39:43], scene[:, 36:39]
    np.savez(npz_dir / f"{demo}_original.npz", qpos=q,
             human_joints=human, fps=30.0)


def _write_demo_artifacts(rng, npz_dir, att, demo):
    # ONE scene drives the published NPZ and both artifacts — the binding
    # check requires the human reference and box trajectory to match exactly
    scene, human = _scenes(rng)
    q = scene.copy()                       # published [qw,qx,qy,qz,x,y,z]
    q[:, 0:4], q[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
    q[:, 36:40], q[:, 40:43] = scene[:, 39:43], scene[:, 36:39]
    np.savez(npz_dir / f"{demo}_original.npz", qpos=q,
             human_joints=human, fps=30.0)
    np.savez(att / f"{demo}_original.npz", qpos=scene.copy(),
             human_joints=human, fps=30.0)
    np.savez(att / f"{demo}_original_converted.npz", qpos=scene.copy(),
             human_joints=human, fps=30.0)
    return {"demo": demo, "status": "ok", "binding": {"passed": True},
            "raw_sha256": integ.sha256_file(att / f"{demo}_original.npz")}


def _build_split_root(tmp_path, *, first_five=PILOT[:5], retry=PILOT[5]):
    """Two-attempt layout: five demos consumed at the canonical (legacy)
    dir, the sixth consumed at a retry dir holding ONLY that demo's PT."""
    rng = np.random.default_rng(11)
    npz_dir = tmp_path / "pub"; npz_dir.mkdir(parents=True)
    pt_dir = tmp_path / "pt"; pt_dir.mkdir()          # canonical: all six
    retry_dir = tmp_path / "retry_pt"; retry_dir.mkdir()
    u0 = tmp_path / "u0"
    att1 = u0 / "attempt_001"; att1.mkdir(parents=True)
    att2 = u0 / "attempt_002"; att2.mkdir()
    rows1, inputs1 = [], {}
    for demo in PILOT:      # published NPZ + canonical PT exist for ALL six
        _write_pub_npz(rng, npz_dir, demo)
        (pt_dir / f"{demo}.pt").write_bytes(f"pt-{demo}".encode())
    for demo in first_five:
        rows1.append(_write_demo_artifacts(rng, npz_dir, att1, demo))
        inputs1[demo] = {
            "npz": integ.sha256_file(npz_dir / f"{demo}_original.npz"),
            "pt": integ.sha256_file(pt_dir / f"{demo}.pt")}
    rows2, inputs2 = [], {}
    if retry is not None:
        rows2.append(_write_demo_artifacts(rng, npz_dir, att2, retry))
        (pt_dir / f"{retry}.pt").write_bytes(f"pt-{retry}".encode())
        (retry_dir / f"{retry}.pt").write_bytes(f"pt-{retry}".encode())
        inputs2[retry] = {
            "npz": integ.sha256_file(npz_dir / f"{retry}_original.npz"),
            "pt": integ.sha256_file(retry_dir / f"{retry}.pt")}
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (att1 / "generation_manifest.json").write_text(json.dumps({
        "rows": rows1, "input_sha256": inputs1,
        "identity": {"checkout": {"holosoma_revision": expected,
                                  "holosoma_dirty": False},
                     "env": {"env_python": "envy",
                             "env_uv_lock_sha256": "locksha",
                             "env_mujoco_version": "3.10.0"}}}))
    manifest2 = {
        "rows": rows2, "input_sha256": inputs2,
        "identity": {"checkout": {"holosoma_revision": expected,
                                  "holosoma_dirty": False},
                     "env": {"env_python": "envy",
                             "env_uv_lock_sha256": "locksha",
                             "env_mujoco_version": "3.10.0"}}}
    if retry is not None:
        manifest2["data_path"] = path_for_record(retry_dir)
    (att2 / "generation_manifest.json").write_text(json.dumps(manifest2))
    attempts = [{"attempt": "attempt_001"}]
    if rows2:
        attempts.append({"attempt": "attempt_002"})
    (u0 / "attempts_index.json").write_text(json.dumps({"attempts": attempts}))
    return u0, npz_dir, pt_dir, retry_dir


def _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir, *, entries=None):
    import omniretarget_teacher_comparison as cmp_mod

    freeze = tmp_path / "freeze"; freeze.mkdir(parents=True, exist_ok=True)
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": expected}}))
    if entries is None:
        entries = [
            {"demo": d,
             "npz_sha256": integ.sha256_file(npz_dir / f"{d}_original.npz"),
             "pt_sha256": integ.sha256_file(pt_dir / f"{d}.pt")}
            for d in PILOT]
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": entries}))
    monkeypatch.setattr(cmp_mod, "TEACHER_TRAJ_DIR", npz_dir)
    monkeypatch.setattr(cmp_mod, "OMOMO_NEW_DIR", pt_dir)
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: {"n_checked": 1, "mismatched": [],
                                              "missing": [], "passed": True})
    monkeypatch.setattr(integ, "measure_env_identity",
                        lambda p: {"env_python": "envy",
                                   "env_uv_lock_sha256": "locksha",
                                   "env_mujoco_version": "3.10.0",
                                   "env_lock_mujoco_version": "3.10.0",
                                   "env_probe_error": None})
    return cmp_mod.admission_record(freeze, u0, PILOT)


# --- review 1 (round 6): per-dir verification scope = consumed demos ----------


def test_split_retry_dir_verified_only_for_consumed_demos(tmp_path, monkeypatch):
    # reviewer's counterexample: five demos consumed at the canonical dir,
    # the sixth at a retry dir holding ONLY that PT — pre-fix the freeze leg
    # required the FULL frozen list at every dir and the retry dir was
    # rejected for the five files it never served
    u0, npz_dir, pt_dir, retry_dir = _build_split_root(tmp_path)
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["passed"], adm["inputs_vs_freeze"]
    assert adm["inputs_vs_freeze"]["all_match"]
    # each dir verified exactly for the demos consumed from it
    by_dir = adm["inputs_vs_freeze"]["by_dir"]
    assert set(by_dir) == {path_for_record(pt_dir), path_for_record(retry_dir)}
    assert set(by_dir[path_for_record(pt_dir)]) == set(PILOT[:5])
    assert set(by_dir[path_for_record(retry_dir)]) == {PILOT[5]}
    # the per-demo view still covers the full six-demo denominator
    assert set(adm["inputs_vs_freeze"]["by_demo"]) == set(PILOT)
    assert adm["data_paths"]["attempt_002"]["recorded_in_manifest"] is True


def test_missing_demo_pt_still_reconciled_at_canonical_dir(
        tmp_path, monkeypatch):
    # a demo with no OK artifact has no consumed dir, but its PT is still
    # reconciled at the canonical dir — the aggregate denominator stays six
    u0, npz_dir, pt_dir, retry_dir = _build_split_root(tmp_path, retry=None)
    # positive: the missing demo's PT sits at the canonical dir, frozen
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["inputs_vs_freeze"]["all_match"]
    assert set(adm["inputs_vs_freeze"]["by_demo"]) == set(PILOT)   # all six
    assert adm["inputs_vs_freeze"]["by_dir"][path_for_record(pt_dir)][PILOT[5]][
        "pt_matches"] is True

    # negative: the missing demo's PT deleted from the canonical dir — the
    # leg must still catch it (a "consumed-only" scope that silently DROPPED
    # missing demos would pass here)
    frozen_entries = [
        {"demo": d,
         "npz_sha256": integ.sha256_file(npz_dir / f"{d}_original.npz"),
         "pt_sha256": integ.sha256_file(pt_dir / f"{d}.pt")}
        for d in PILOT]
    (pt_dir / f"{PILOT[5]}.pt").unlink()
    adm2, *_ = _admission(tmp_path / "b", monkeypatch, u0, npz_dir, pt_dir,
                          entries=frozen_entries)
    assert not adm2["inputs_vs_freeze"]["all_match"]
    assert not adm2["passed"]


# --- review 2 (round 6): the denominator is an exact frame COUNT --------------


def test_fractional_both_denominator_is_invalid():
    # reviewer's counterexample: P=30, U0=30.9 — pre-fix int(30.9)=30
    # "agreed" with P and the fractional row kept voting in g4
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d) for d in PILOT}
    for d in PILOT[:3]:
        u[d].both = {"denominator_frames": 30.9, "valid_coverage": 0.55,
                     "gap_series_p95_m": 0.05}
    for d in PILOT[3:]:
        u[d].both = dict(_gains(d)["both"])
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert all(any(f"{d}:U0_both_denominator_missing_or_invalid" in g
                  for g in v["evidence_gaps"]) for d in PILOT[:3])
    # the fractional rows no longer vote in g4
    assert v["gates"]["g4_hand_improvement"]["n_coverage_gain_demos"] == 3


def test_negative_fraction_both_denominator_is_invalid():
    # -0.5 truncated to a legal zero on BOTH sides pre-fix -> "agreed"
    neg = {"denominator_frames": -0.5, "valid_coverage": None,
           "gap_series_p95_m": None}
    p = {d: _metrics(d, both=neg) for d in PILOT}
    u = {d: _metrics(d, both=neg) for d in PILOT}
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert any("P_both_denominator_missing_or_invalid" in g
               for g in v["evidence_gaps"])
    assert any("U0_both_denominator_missing_or_invalid" in g
               for g in v["evidence_gaps"])


def test_integer_both_denominators_still_reach_local_preferred():
    # positive regression guard: exact integer counts keep the full path
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d, **_gains(d)) for d in PILOT}
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["complete_evidence"] is True
    assert v["verdict"] == "local_preferred"
