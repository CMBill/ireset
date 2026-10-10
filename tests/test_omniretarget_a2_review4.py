"""A2 fourth-round review counterexamples (independent review, 2026-10-09).

Every test fails on the pre-fix implementation:

1  admission reconciled the current inputs only against the generation
   manifest — inputs that disagree with the A1 FREEZE while the manifest
   faithfully recorded the on-disk (tampered) hashes still passed
   admission, especially when consuming old attempts;
2  the generation entry verified and anchored the canonical OMOMO dir but
   actually consumed ``--data-path`` — an unfrozen alternate PT dir passed
   the precheck;
3  the verdict computed the quality gates BEFORE the evidence check — a
   None task error / None foot fields / a both-hands dict without keys
   crashed with TypeError/KeyError instead of yielding incomplete, and
   serializing such a row crashed ``to_json``;
4  the env probe raised FileNotFoundError when the interpreter could not
   launch, so the comparison died before writing its incomplete report.
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


def _build_u0_root(tmp_path, *, demos, env_python="envy"):
    """Fake I0 layout: attempt, manifest, raw+converted, published P, PTs."""
    rng = np.random.default_rng(11)
    npz_dir = tmp_path / "pub"; npz_dir.mkdir(parents=True)
    pt_dir = tmp_path / "pt"; pt_dir.mkdir()
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
        inputs[demo] = {"npz": integ.sha256_file(npz_dir / f"{demo}_original.npz"),
                        "pt": integ.sha256_file(pt_dir / f"{demo}.pt")}
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (att / "generation_manifest.json").write_text(json.dumps({
        "rows": rows, "input_sha256": inputs,
        "identity": {
            "checkout": {"holosoma_revision": expected, "holosoma_dirty": False},
            "env": {"env_python": env_python, "env_uv_lock_sha256": "locksha",
                    "env_mujoco_version": "3.10.0"},
        }}))
    (u0 / "attempts_index.json").write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"}]}))
    return u0, npz_dir, pt_dir


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


# --- review 3 (round 4): deficient rows never crash the verdict ----------------


def test_deficient_rows_yield_incomplete_not_typeerror():
    p = {d: _metrics(d) for d in PILOT}

    # None task error: pre-fix TypeError in g2 (None <= float) fired before
    # the evidence check ever ran
    u = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u[PILOT[0]].task_mpjpe_p95_m = None
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert any("U0_task_mpjpe_p95_m_not_finite" in g for g in v["evidence_gaps"])
    assert v["gates"]["g2_task_mpjpe"]["per_demo"][PILOT[0]] is False

    # None foot speed: pre-fix TypeError in g3's bound arithmetic
    u2 = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u2[PILOT[1]].foot["r"].speed_p95_mps = None
    v2 = evaluate_local_preferred(PILOT, p, u2, {d: True for d in PILOT})
    assert v2["verdict"] == "incomplete"
    assert (v2["gates"]["g3_penetration_and_feet"]["per_demo"][PILOT[1]]
            ["foot"]["r"] == "foot_metric_not_finite")

    # both-hands dict missing its keys: pre-fix KeyError in g4
    u3 = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u3[PILOT[2]].both = {"denominator_frames": 30}
    v3 = evaluate_local_preferred(PILOT, p, u3, {d: True for d in PILOT})
    assert v3["verdict"] == "incomplete"
    assert any(f"{PILOT[2]}:U0_both_hands_not_finite" in g
               for g in v3["evidence_gaps"])

    # P-side violation ratio missing: pre-fix TypeError in g5's comparison
    p4 = {d: _metrics(d) for d in PILOT}
    p4[PILOT[3]].penetration_violation_ratio = None
    v4 = evaluate_local_preferred(PILOT, p4,
                                  {d: _metrics(d, **_gains(d)) for d in PILOT},
                                  {d: True for d in PILOT})
    assert v4["verdict"] == "incomplete"
    assert (v4["gates"]["g5_degradation"]["per_demo"][PILOT[3]]["degradation"]
            .get("penetration_violation_ratio_missing"))


def test_to_json_serializes_deficient_rows():
    m = _metrics(PILOT[0])
    m.task_mpjpe_p95_m = None
    m.penetration_max_m = None
    m.foot["l"].gap_p95_m = float("nan")
    m.hand["r"].drift_max_m = None          # already None-safe pre-fix
    rec = json.loads(json.dumps(m.to_json()))   # pre-fix: np.isnan(None) TypeError
    assert rec["task_mpjpe_p95_m"] is None
    assert rec["penetration_max_m"] is None
    assert rec["foot"]["l"]["gap_p95_m"] is None
    assert rec["hand"]["r"]["drift_max_m"] is None


# --- review 4 (round 4): env probe launch failure is evidence, not a crash -----


def test_measure_env_identity_records_launch_failure(tmp_path):
    missing = tmp_path / "nope" / ".venv" / "bin" / "python"
    rec = integ.measure_env_identity(missing)      # pre-fix: FileNotFoundError
    assert rec["env_mujoco_version"] is None
    assert "probe launch failed" in rec["env_probe_error"]
    assert "FileNotFoundError" in rec["env_probe_error"]


def test_comparison_reports_incomplete_when_probe_cannot_launch(
        tmp_path, monkeypatch):
    import omniretarget_teacher_comparison as cmp_mod

    # the generation manifest points at an interpreter that does not exist;
    # measure_env_identity stays REAL — admission must record the launch
    # failure and the comparison must still write its incomplete report
    u0, npz_dir, pt_dir = _build_u0_root(
        tmp_path, demos=PILOT, env_python=str(tmp_path / "nope" / ".venv"
                                              / "bin" / "python"))

    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
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
    out = tmp_path / "experiments" / "a2run" / "teacher_pilot"
    monkeypatch.setattr(sys, "argv", ["prog", "--freeze", str(freeze),
                                      "--u0-dir", str(u0), "--out", str(out)])
    assert cmp_mod.main() == 0
    report = json.loads(
        (out / "cmp" / "attempt_001" / "teacher_selection.json").read_text())
    assert report["verdict"] == "incomplete"
    assert "probe launch failed" in report["admission"]["env_mujoco"]["probe_error"]
    assert any("admission:env_" in g
               for g in report["verdict_record"]["evidence_gaps"])


# --- review 1 (round 4): admission reconciles inputs against the FREEZE --------


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


def test_admission_reconciles_inputs_against_the_freeze(tmp_path, monkeypatch):
    u0, npz_dir, pt_dir = _build_u0_root(tmp_path, demos=PILOT)
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["passed"], adm
    assert adm["inputs_vs_freeze"]["all_match"]

    # reviewer's counterexample: the on-disk inputs disagree with the A1
    # freeze while the generation manifest faithfully recorded the on-disk
    # hashes — the manifest-vs-now leg passes, chain and bindings stay
    # intact; only the freeze leg catches it
    u0b, npzb, ptb = _build_u0_root(tmp_path / "b", demos=PILOT)
    wrong = [{"demo": d, "npz_sha256": "0" * 64, "pt_sha256": "0" * 64}
             for d in PILOT]
    adm2, *_ = _admission(tmp_path / "b", monkeypatch, u0b, npzb, ptb,
                          entries=wrong)
    assert not adm2["passed"]
    assert not adm2["inputs_vs_freeze"]["all_match"]
    assert adm2["inputs_rehashed"]["all_match"]       # the other leg agrees

    # an input file that vanished is recorded, never a crash
    u0c, npzc, ptc = _build_u0_root(tmp_path / "c", demos=PILOT)
    frozen_pt = integ.sha256_file(ptc / f"{PILOT[0]}.pt")
    (ptc / f"{PILOT[0]}.pt").unlink()
    entries_c = [
        {"demo": d,
         "npz_sha256": integ.sha256_file(npzc / f"{d}_original.npz"),
         "pt_sha256": (frozen_pt if d == PILOT[0]
                       else integ.sha256_file(ptc / f"{d}.pt"))}
        for d in PILOT]
    adm3, *_ = _admission(tmp_path / "c", monkeypatch, u0c, npzc, ptc,
                          entries=entries_c)
    assert not adm3["passed"]
    assert "error" in adm3["inputs_vs_freeze"]["by_demo"]


# --- review 2 (round 4): the generation gate covers the CONSUMED data_path -----


def _freeze_with_input_hashes(tmp_path):
    freeze = tmp_path / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [
            {"demo": d, "npz_sha256": "fakehash", "pt_sha256": "fakehash"}
            for d in PILOT]}))
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit":
                      integ.read_checkout_identity()["holosoma_revision"]}}))
    return freeze


def test_generation_gate_verifies_the_actual_data_path(tmp_path, monkeypatch):
    import generate_omniretarget_teachers as gen

    alt = tmp_path / "alt_pt"; alt.mkdir()     # NOT what the freeze pinned
    for d in PILOT:
        (alt / f"{d}.pt").write_bytes(f"unfrozen-{d}".encode())

    calls = []
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
    monkeypatch.setattr(
        gen, "generate_one",
        lambda demo, env_python, attempt_dir, data_path, **kw: (
            calls.append((demo, str(data_path))),
            {"demo": demo, "argv": [], "cwd": "", "exit_code": 0, "wall_s": 0.1,
             "config": "fake", "status": "ok", "binding": {"passed": True},
             "error": None, "converted": "x"})[1])
    monkeypatch.setattr(
        integ, "sha256_file",
        lambda p: "althash" if "alt_pt" in str(p) else "fakehash")
    monkeypatch.setattr(
        integ, "measure_env_identity",
        lambda p: {"env_python": "x", "env_uv_lock_sha256": "locksha",
                   "env_mujoco_version": "3.10.0",
                   "env_lock_mujoco_version": "3.10.0", "env_probe_error": None})
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: {"n_checked": 1, "mismatched": [],
                                              "missing": [], "passed": True})
    freeze = _freeze_with_input_hashes(tmp_path)
    out_dir = tmp_path / "u0"

    # pre-fix: the precheck hashed the canonical OMOMO dir (fakehash) while
    # the run consumed --data-path alt (althash) — it passed and generated
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--freeze", str(freeze), "--out-dir",
                         str(out_dir), "--data-path", str(alt)])
    with pytest.raises(SystemExit, match="A1-frozen hashes"):
        gen.main()
    assert calls == []                 # the gate fires before any generation

    # a different DIRECTORY whose files are bit-identical to the freeze is
    # legal: the gate anchors content, not the path
    ok_dir = tmp_path / "ok_pt"; ok_dir.mkdir()
    for d in PILOT:
        (ok_dir / f"{d}.pt").write_bytes(f"pt-{d}".encode())
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--freeze", str(freeze), "--out-dir",
                         str(out_dir), "--data-path", str(ok_dir)])
    assert gen.main() == 0
    assert len(calls) == 6
    manifest = json.loads(
        (out_dir / "attempt_001" / "generation_manifest.json").read_text())
    assert manifest["data_path"]       # the consumed dir is on the record
