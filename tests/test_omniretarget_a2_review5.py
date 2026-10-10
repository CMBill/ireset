"""A2 fifth-round review counterexamples (independent review, 2026-10-10).

Every test fails on the pre-fix implementation:

1  admission hashed BOTH PT-side input legs at the canonical OMOMO dir while
   the consumed attempts record the data path they actually consumed
   (``manifest["data_path"]``, round-4 generation entry) — a generation
   over a freeze-identical ALTERNATE dir that later drifted still passed
   admission, because both legs hashed the untouched canonical dir;
2  a both-hands ``denominator_frames`` missing from the metrics dict
   defaulted to zero — indistinguishable from "no both-grasp reference
   frames" — so a U0 row with the field deleted kept
   ``complete_evidence`` (and even ``local_preferred`` when the other
   demos supplied the gains), and the value None raised TypeError in the
   ``> 0`` comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
    """A U0 row that passes every gate: gains +15pp, nothing degrades."""
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


def _build_u0_root(tmp_path, *, demos, env_python="envy", data_path=None,
                   alt_writer=None):
    """Fake I0 layout; optionally records an alternate ``data_path``.

    ``pt_dir`` stays the canonical (freeze-matching) PT source. When
    ``data_path`` is given the manifest records it as consumed and FAITHFULLY
    hashes whatever ``alt_writer`` places there (default: freeze-identical
    copies) — the manifest leg always compares "now == recorded".
    """
    rng = np.random.default_rng(11)
    npz_dir = tmp_path / "pub"; npz_dir.mkdir(parents=True)
    pt_dir = tmp_path / "pt"; pt_dir.mkdir()
    alt = Path(data_path).expanduser().resolve() if data_path else None
    if alt is not None:
        alt.mkdir(parents=True, exist_ok=True)
        for demo in demos:
            (alt / f"{demo}.pt").write_bytes(
                (alt_writer or (lambda d: f"pt-{d}".encode()))(demo))
    u0 = tmp_path / "u0"
    att = u0 / "attempt_001"; att.mkdir(parents=True)
    rows, inputs = [], {}
    for demo in demos:
        scene, human = _scenes(rng)
        q = scene.copy()                       # published [qw,qx,qy,qz,x,y,z]
        q[:, 0:4], q[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
        q[:, 36:40], q[:, 40:43] = scene[:, 39:43], scene[:, 36:39]
        np.savez(npz_dir / f"{demo}_original.npz", qpos=q,
                 human_joints=human, fps=30.0)
        (pt_dir / f"{demo}.pt").write_bytes(f"pt-{demo}".encode())
        np.savez(att / f"{demo}_original.npz", qpos=scene.copy(),
                 human_joints=human, fps=30.0)
        np.savez(att / f"{demo}_original_converted.npz", qpos=scene.copy(),
                 human_joints=human, fps=30.0)
        rows.append({"demo": demo, "status": "ok", "binding": {"passed": True},
                     "raw_sha256": integ.sha256_file(att / f"{demo}_original.npz")})
        pt_hash = integ.sha256_file(
            (alt if alt is not None else pt_dir) / f"{demo}.pt")
        inputs[demo] = {"npz": integ.sha256_file(npz_dir / f"{demo}_original.npz"),
                        "pt": pt_hash}
    expected = integ.read_checkout_identity()["holosoma_revision"]
    manifest = {
        "rows": rows, "input_sha256": inputs,
        "identity": {
            "checkout": {"holosoma_revision": expected, "holosoma_dirty": False},
            "env": {"env_python": env_python, "env_uv_lock_sha256": "locksha",
                    "env_mujoco_version": "3.10.0"},
        }}
    if alt is not None:
        manifest["data_path"] = path_for_record(alt)
    (att / "generation_manifest.json").write_text(json.dumps(manifest))
    (u0 / "attempts_index.json").write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"}]}))
    return u0, npz_dir, pt_dir, alt


def _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir, *, entries=None):
    import omniretarget_teacher_comparison as cmp_mod

    freeze = tmp_path / "freeze"; freeze.mkdir(exist_ok=True)
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


# --- review 2 (round 5): the both-hands denominator is decidable evidence -----


def test_both_denominator_missing_blocks_local_preferred():
    # reviewer's counterexample: P has both-grasp reference frames; three U0
    # rows have their denominator field DELETED (everything else improves)
    # — pre-fix the missing key read as "no reference frames", g4 skipped
    # those demos, the other three supplied the >=3 gains and the verdict
    # was complete_evidence=True / local_preferred on deleted evidence
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d, **_gains(d)) for d in PILOT}
    for d in PILOT[3:]:
        u[d].both = {"valid_coverage": 0.55, "gap_series_p95_m": 0.05}
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert not v["complete_evidence"]
    assert all(any(f"{d}:U0_both_denominator_missing_or_invalid" in g
                  for g in v["evidence_gaps"]) for d in PILOT[3:])
    # the gates stay decidable diagnostics: g4 still sees the three gains
    assert v["gates"]["g4_hand_improvement"]["n_coverage_gain_demos"] == 3


def test_both_denominator_none_is_a_gap_not_typeerror():
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u[PILOT[0]].both = {"denominator_frames": None, "valid_coverage": 0.55,
                        "gap_series_p95_m": 0.05}
    # pre-fix: None > 0 raised TypeError inside the evidence check
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert any(f"{PILOT[0]}:U0_both_denominator_missing_or_invalid" in g
               for g in v["evidence_gaps"])


def test_both_denominator_disagreement_is_incomplete():
    # P measures 30 common frames; U0 claims ZERO for half the demos — the
    # denominator is decided by the SHARED human reference, so the two
    # disagreeing rows are contradictory evidence, not a legal "no
    # reference" case; pre-fix the zero rows were skipped and local_preferred
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d, **_gains(d)) for d in PILOT}
    for d in PILOT[3:]:
        u[d].both = {"denominator_frames": 0, "valid_coverage": None,
                     "gap_series_p95_m": None}
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert all(any(f"{d}:both_denominator_disagreement_p30_u0_0" in g
                  for g in v["evidence_gaps"]) for d in PILOT[3:])


def test_both_denominator_zero_agreed_stays_complete():
    # genuine "no both-grasp reference frames" on BOTH sides stays complete
    # (the shared reference decides it, and the two agree)
    zero = {"denominator_frames": 0, "valid_coverage": None,
            "gap_series_p95_m": None}
    p = {d: _metrics(d, both=zero) for d in PILOT}
    u = {d: _metrics(d, both=zero) for d in PILOT}
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["complete_evidence"] is True
    assert v["verdict"] == "no_quality_gain"   # g4 has no gains to count


# --- review 1 (round 5): admission anchors the RECORDED data_path -------------


def test_admission_anchors_pt_legs_at_the_recorded_data_path(
        tmp_path, monkeypatch):
    # reviewer's counterexample: generation consumed a freeze-identical
    # ALTERNATE dir (recorded in the manifest), the dir then drifted, the
    # canonical dir never moved — pre-fix both legs hashed the canonical
    # dir and admission still passed
    alt = tmp_path / "alt_pt"
    u0, npz_dir, pt_dir, alt = _build_u0_root(
        tmp_path, demos=PILOT, data_path=alt)
    assert alt is not None
    (alt / f"{PILOT[0]}.pt").write_bytes(b"drifted-after-generation")
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)

    assert not adm["passed"]
    # manifest leg: the drifted dir no longer matches what generation
    # recorded (this alone also fails pre-fix IF the canonical dir moved —
    # here it did not, so pre-fix passed)
    assert not adm["inputs_rehashed"]["all_match"]
    assert adm["inputs_rehashed"]["by_demo"][PILOT[0]]["pt_matches"] is False
    # freeze leg: the drifted dir no longer matches the A1-frozen hashes
    assert not adm["inputs_vs_freeze"]["all_match"]
    alt_label = path_for_record(alt)
    assert adm["inputs_vs_freeze"]["by_dir"][alt_label][PILOT[0]]["pt_matches"] is False
    # the reconciliation is anchored at (and records) the consumed dir
    assert adm["data_paths"]["attempt_001"] == {
        "pt_dir": alt_label, "recorded_in_manifest": True}
    assert adm["inputs_rehashed"]["by_demo"][PILOT[0]]["data_path"] == alt_label


def test_admission_accepts_freeze_identical_alternate_data_path(
        tmp_path, monkeypatch):
    # content anchoring, not path anchoring: a DIFFERENT dir whose files are
    # bit-identical to the freeze is legal and fully reconciled
    u0, npz_dir, pt_dir, alt = _build_u0_root(
        tmp_path, demos=PILOT, data_path=tmp_path / "alt_pt")
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["passed"], adm["inputs_vs_freeze"]
    assert adm["inputs_rehashed"]["all_match"]
    assert adm["inputs_vs_freeze"]["all_match"]
    assert adm["data_paths"]["attempt_001"]["recorded_in_manifest"] is True
    assert path_for_record(alt) in adm["inputs_vs_freeze"]["by_dir"]


def test_admission_legacy_manifest_falls_back_to_canonical_dir(
        tmp_path, monkeypatch):
    # pre-round-4 manifests never recorded data_path: the legs anchor the
    # canonical dir (the strongest evidence they admit) and the attempt is
    # labeled recorded_in_manifest=False — not a failure
    u0, npz_dir, pt_dir, alt = _build_u0_root(tmp_path, demos=PILOT)
    assert alt is None
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["passed"], adm
    assert adm["data_paths"]["attempt_001"] == {
        "pt_dir": path_for_record(pt_dir), "recorded_in_manifest": False}
    assert adm["inputs_rehashed"]["by_demo"][PILOT[0]]["data_path_recorded"] is False


def test_admission_freeze_leg_catches_faithfully_recorded_tampered_alt_dir(
        tmp_path, monkeypatch):
    # the data-path twin of round-4 fix 1: the alternate dir was ALREADY
    # tampered at generation time and the manifest FAITHFULLY recorded the
    # tampered hashes — the manifest leg agrees ("inputs now == recorded"),
    # the freeze leg at the recorded data path is what catches it; pre-fix
    # the freeze leg hashed the untouched canonical dir and admission passed
    alt = tmp_path / "alt_pt"
    u0, npz_dir, pt_dir, alt = _build_u0_root(
        tmp_path, demos=PILOT, data_path=alt,
        alt_writer=lambda d: f"tampered-{d}".encode())
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["inputs_rehashed"]["all_match"]      # faithfully recorded ...
    assert not adm["inputs_vs_freeze"]["all_match"]   # ... but not frozen
    assert not adm["passed"]


# --- end to end: the report reflects the anchored reconciliation --------------


class _FakeReplayer:
    def __init__(self, *a, **k):
        pass

    def measure(self, demo, scene, human, source_box, fps):
        return DemoMetrics(
            demo=demo, T=scene.shape[0], fps=fps, valid=True,
            penetration_max_m=0.001, penetration_violation_ratio=0.0,
            task_mpjpe_p95_m=0.10, foot={"l": None, "r": None},
            hand={"l": None, "r": None},
            both={"denominator_frames": 0, "valid_coverage": None,
                  "gap_series_p95_m": None},
            validator_failed_checks=[],
            ref_support_frames={"l": 0, "r": 0}, ref_grasp_frames={"l": 0, "r": 0},
        )


def test_comparison_cli_reports_incomplete_on_data_path_drift(
        tmp_path, monkeypatch):
    import omniretarget_teacher_comparison as cmp_mod

    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
    alt = tmp_path / "alt_pt"
    u0, npz_dir, pt_dir, alt = _build_u0_root(
        tmp_path, demos=PILOT, data_path=alt)
    (alt / f"{PILOT[0]}.pt").write_bytes(b"drifted-after-generation")

    freeze = tmp_path / "freeze"; freeze.mkdir()
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": expected}}))
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [
            {"demo": d,
             "npz_sha256": integ.sha256_file(npz_dir / f"{d}_original.npz"),
             "pt_sha256": integ.sha256_file(pt_dir / f"{d}.pt")}
            for d in PILOT]}))
    np.savez(freeze / "pairing_source_boxes.npz",
             **{d: np.zeros((5, 7)) for d in PILOT})

    monkeypatch.setattr(cmp_mod, "TEACHER_TRAJ_DIR", npz_dir)
    monkeypatch.setattr(cmp_mod, "OMOMO_NEW_DIR", pt_dir)
    monkeypatch.setattr(cmp_mod, "TeacherQualityReplayer", _FakeReplayer)
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: {"n_checked": 1, "mismatched": [],
                                              "missing": [], "passed": True})
    monkeypatch.setattr(integ, "measure_env_identity",
                        lambda p: {"env_python": "envy",
                                   "env_uv_lock_sha256": "locksha",
                                   "env_mujoco_version": "3.10.0",
                                   "env_lock_mujoco_version": "3.10.0",
                                   "env_probe_error": None})
    out = tmp_path / "experiments" / "a2run" / "teacher_pilot"
    monkeypatch.setattr(sys, "argv", ["prog", "--freeze", str(freeze),
                                      "--u0-dir", str(u0), "--out", str(out)])
    assert cmp_mod.main() == 0
    report = json.loads(
        (out / "cmp" / "attempt_001" / "teacher_selection.json").read_text())
    assert report["verdict"] == "incomplete"
    assert not report["admission"]["passed"]
    assert any("admission:input_hash_mismatch_vs_generation" in g
               for g in report["verdict_record"]["evidence_gaps"])
    assert any("admission:input_hash_mismatch_vs_freeze" in g
               for g in report["verdict_record"]["evidence_gaps"])
    assert report["admission"]["data_paths"]["attempt_001"][
        "recorded_in_manifest"] is True
