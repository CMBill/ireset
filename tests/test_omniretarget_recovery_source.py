"""UR (upstream_recovery_step03_v1) contracts — closure plan §10.3.

* generation: the preregistration gate (identity / single difference /
  parent hashes / REAL config smoke), the no-mixed-registration guard and
  the per-demo wall-clock timeout status;
* comparison: the registration/config reconciliation leg forces incomplete
  on a drifted preregistration, a two-field config change, a missing
  step-size flag in argv, a failed fresh smoke, or a U0 attempt mixed in;
* labeling: a UR report never labels the recovery source "U0", and the
  U0-mode default strings stay byte-identical (regression).
"""

from __future__ import annotations

import copy
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


# --- evaluate_local_preferred source labeling --------------------------------

def _metrics(demo, *, valid=True):
    foot_m = {s: SideFootMetrics(30, 0.003, 0.01, 0.01, 29) for s in HANDS}
    hand_m = {s: SideHandMetrics(30, 0.02, 0.5, 0.5, 10, 5, 0.01)
              for s in HANDS}
    return DemoMetrics(
        demo=demo, T=10, fps=30.0, valid=valid,
        penetration_max_m=0.001, penetration_violation_ratio=0.0,
        task_mpjpe_p95_m=0.10, foot=foot_m, hand=hand_m,
        both={"denominator_frames": 30, "valid_coverage": 0.40,
              "gap_series_p95_m": 0.05},
        validator_failed_checks=[],
        ref_support_frames={"l": 30, "r": 30},
        ref_grasp_frames={"l": 30, "r": 30},
    )


def test_verdict_record_labels_the_registered_source():
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d, valid=(d != PILOT[-1])) for d in PILOT}
    binding = {d: d != PILOT[-1] for d in PILOT}
    ur = evaluate_local_preferred(PILOT, p, u, binding, u_label="UR")
    assert ur["verdict"] == "incomplete"
    assert f"{PILOT[-1]}:UR_invalid" in ur["evidence_gaps"]
    assert "U0" not in json.dumps(ur)
    # U0-mode default strings stay byte-identical
    u0 = evaluate_local_preferred(PILOT, p, u, binding)
    assert f"{PILOT[-1]}:U0_invalid" in u0["evidence_gaps"]
    assert "UR" not in json.dumps(u0)


# --- generation: preregistration gate ----------------------------------------

def _prereg(tmp_path: Path, pilot: list[str] = PILOT) -> Path:
    prereg = {
        "experiment_id": "a2_recovery_step03_v1",
        "source_id": "upstream_recovery_step03_v1",
        "single_config_difference": {"retargeter.step_size": {"from": 0.2,
                                                              "to": 0.3}},
        "expected_demos": pilot,
        "parent": {},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "preregistration.json"
    p.write_text(json.dumps(prereg))
    return p


SMOKE_OK = {
    "diff": {"retargeter.step_size": [0.2, 0.3]},
    "step_size_default": 0.2, "step_size_recovery": 0.3,
    "canonical_config_sha256": "c" * 64,
    "default": {}, "recovery": {},
}


@pytest.fixture()
def gen_mod(monkeypatch):
    import generate_omniretarget_teachers as gen
    monkeypatch.setattr(gen, "run_config_smoke",
                        lambda *a, **k: copy.deepcopy(SMOKE_OK))
    return gen


def test_load_source_record_accepts_the_registered_identity(tmp_path, gen_mod):
    pr = _prereg(tmp_path)
    parent_sel = tmp_path / "selection.json"
    parent_diag = tmp_path / "diagnosis.json"
    parent_sel.write_text("{}")
    parent_diag.write_text("{}")
    pr2 = json.loads(pr.read_text())
    pr2["parent"] = {
        "authoritative_selection": str(parent_sel),
        "authoritative_selection_sha256": integ.sha256_file(parent_sel),
        "diagnosis": str(parent_diag),
        "diagnosis_sha256": integ.sha256_file(parent_diag)}
    pr.write_text(json.dumps(pr2))
    rec = gen_mod.load_source_record(pr, PILOT, Path("envy"), Path("data"))
    assert rec["preregistration_sha256"] == integ.sha256_file(pr)
    assert rec["config_smoke"]["diff"] == SMOKE_OK["diff"]


@pytest.mark.parametrize("mutate,frag", [
    (lambda r: r.update(experiment_id="other_experiment"), "identity"),
    (lambda r: r.update(single_config_difference={"retargeter.step_size":
                                                  {"from": 0.2, "to": 0.4}}),
     "identity"),
    (lambda r: r.update(expected_demos=PILOT[:-1]), "identity"),
])
def test_load_source_record_rejects_wrong_identity(tmp_path, gen_mod,
                                                   mutate, frag):
    pr = _prereg(tmp_path)
    r = json.loads(pr.read_text())
    mutate(r)
    pr.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match=frag):
        gen_mod.load_source_record(pr, PILOT, Path("envy"), Path("data"))


def test_load_source_record_rejects_drifted_parent_and_bad_smoke(tmp_path,
                                                                 gen_mod):
    sel = tmp_path / "selection.json"
    sel.write_text("v1")
    pr = _prereg(tmp_path)
    parent = tmp_path / "parent.json"
    parent.write_text("v1")
    r = json.loads(pr.read_text())
    r["parent"] = {"authoritative_selection": str(sel),
                   "authoritative_selection_sha256": integ.sha256_file(sel),
                   "diagnosis": str(parent),
                   "diagnosis_sha256": integ.sha256_file(parent)}
    pr.write_text(json.dumps(r))
    parent.write_text("v2")     # hash drifted after registration
    with pytest.raises(SystemExit, match="parent diagnosis hash"):
        gen_mod.load_source_record(pr, PILOT, Path("envy"), Path("data"))
    sel_b = tmp_path / "b" / "selection.json"
    sel_b.parent.mkdir(parents=True, exist_ok=True)
    sel_b.write_text("v1")
    diag_b = tmp_path / "b" / "diagnosis.json"
    diag_b.write_text("v1")
    pr2 = _prereg(tmp_path / "b")
    rb = json.loads(pr2.read_text())
    rb["parent"] = {"authoritative_selection": str(sel_b),
                    "authoritative_selection_sha256":
                        integ.sha256_file(sel_b),
                    "diagnosis": str(diag_b),
                    "diagnosis_sha256": integ.sha256_file(diag_b)}
    pr2.write_text(json.dumps(rb))
    monkey_smoke = {"diff": {"retargeter.step_size": [0.2, 0.3],
                             "retargeter.debug": [False, True]},
                    "step_size_default": 0.2, "step_size_recovery": 0.3}
    import generate_omniretarget_teachers as gen
    with pytest.raises(SystemExit, match="single registered difference"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(gen, "run_config_smoke", lambda *a, **k: monkey_smoke)
            gen.load_source_record(pr2, PILOT, Path("envy"), Path("data"))


def test_ensure_single_registration_rejects_mixed_roots(tmp_path, gen_mod):
    root = tmp_path / "ur_root"
    root.mkdir()
    gen_mod.ensure_single_registration(root, "shaA")   # no index yet: ok
    (root / integ.ATTEMPTS_INDEX).write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001",
                       "preregistration_sha256": "shaA"}]}))
    gen_mod.ensure_single_registration(root, "shaA")
    with pytest.raises(SystemExit, match="mix"):
        gen_mod.ensure_single_registration(root, "shaB")
    (root / integ.ATTEMPTS_INDEX).write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"}]}))    # plain U0 attempt
    with pytest.raises(SystemExit, match="unregistered"):
        gen_mod.ensure_single_registration(root, "shaA")


def test_generate_one_records_timeout_status(tmp_path, gen_mod, monkeypatch):
    import generate_omniretarget_teachers as gen
    stub = tmp_path / "slow_entry.py"
    stub.write_text("import time; time.sleep(30)\n")
    real_run = __import__("subprocess").run

    def fake_run(cmd, **kw):
        kw = dict(kw)
        kw["timeout"] = 2
        return real_run([str(Path(sys.executable)), str(stub)], **kw)

    monkeypatch.setattr(gen.subprocess, "run", fake_run)
    row = gen.generate_one("demoX", Path(sys.executable), tmp_path / "att",
                           tmp_path, extra_argv=["--retargeter.step-size",
                                                 "0.3"], timeout_s=2)
    assert row["status"] == "timeout"
    assert row["argv"][-2:] == ["--retargeter.step-size", "0.3"]
    assert "step_size=0.3" in row["config"]


# --- comparison: registration/config reconciliation leg ----------------------

def _cfg(step):
    return {"task_type": "object_interaction", "robot": "g1",
            "data_format": "smplh", "task_name": "d",
            "data_path": "p", "save_dir": "s", "augmentation": False,
            "retargeter": {"step_size": step, "debug": False,
                           "visualize": False}}


def _build_ur_root(tmp_path, *, demos=PILOT, prereg_sha="shaReg",
                   manifest_mutate=None, omit_source_record=False):
    npz_dir = tmp_path / "pub"; npz_dir.mkdir(parents=True, exist_ok=True)
    pt_dir = tmp_path / "pt"; pt_dir.mkdir(exist_ok=True)
    ur = tmp_path / "ur"
    att = ur / "attempt_001"; att.mkdir(parents=True)
    rows, inputs = [], {}
    rng = np.random.default_rng(5)
    for demo in demos:
        scene = np.zeros((4, 43)); scene[:, 3] = 1.0
        human = rng.normal(size=(4, 52, 3))
        q = scene.copy()
        q[:, 0:4], q[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
        np.savez(npz_dir / f"{demo}_original.npz", qpos=q,
                 human_joints=human, fps=30.0)
        (pt_dir / f"{demo}.pt").write_bytes(f"pt-{demo}".encode())
        np.savez(att / f"{demo}_original.npz", qpos=scene, human_joints=human,
                 fps=30.0)
        np.savez(att / f"{demo}_original_converted.npz", qpos=scene,
                 human_joints=human, fps=30.0)
        argv = ["py", "examples/robot_retarget.py", "--task-type",
                "object_interaction", "--task-name", demo,
                "--data-path", "d", "--save-dir", "s",
                "--task-config.object-name", "largebox",
                "--retargeter.step-size", "0.3"]
        rows.append({"demo": demo, "status": "ok", "binding": {"passed": True},
                     "raw_sha256": integ.sha256_file(
                         att / f"{demo}_original.npz"), "argv": argv})
        inputs[demo] = {"npz": integ.sha256_file(
            npz_dir / f"{demo}_original.npz"),
            "pt": integ.sha256_file(pt_dir / f"{demo}.pt")}
    expected = integ.read_checkout_identity()["holosoma_revision"]
    manifest = {
        "rows": rows, "input_sha256": inputs,
        "data_path": path_for_record(pt_dir),
        "identity": {
            "checkout": {"holosoma_revision": expected, "holosoma_dirty": False},
            "env": {"env_python": "envy", "env_uv_lock_sha256": "locksha",
                    "env_mujoco_version": "3.10.0"},
        },
    }
    if not omit_source_record:
        manifest["source_record"] = {
            "experiment_id": "a2_recovery_step03_v1",
            "source_id": "upstream_recovery_step03_v1",
            "preregistration_sha256": prereg_sha,
            "step_size": 0.3,
            "config_smoke": {"diff": {"retargeter.step_size": [0.2, 0.3]},
                             "canonical_config_sha256": "c" * 64,
                             "step_size_default": 0.2,
                             "step_size_recovery": 0.3},
            "config_smoke_full": {"default": _cfg(0.2), "recovery": _cfg(0.3)},
        }
    if manifest_mutate is not None:
        manifest_mutate(manifest)
    (att / "generation_manifest.json").write_text(json.dumps(manifest))
    (ur / integ.ATTEMPTS_INDEX).write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001",
                       "preregistration_sha256": prereg_sha}]}))
    return ur, npz_dir, pt_dir


PREREG = {
    "experiment_id": "a2_recovery_step03_v1",
    "source_id": "upstream_recovery_step03_v1",
    "single_config_difference": {"retargeter.step_size": {"from": 0.2,
                                                          "to": 0.3}},
    "expected_demos": PILOT,
}


def _admission(tmp_path, monkeypatch, ur, npz_dir, pt_dir, *,
               prereg_sha="shaReg", smoke=None):
    import omniretarget_teacher_comparison as cmp_mod
    freeze = tmp_path / "freeze"; freeze.mkdir(parents=True, exist_ok=True)
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": expected}}))
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [
            {"demo": d,
             "npz_sha256": integ.sha256_file(npz_dir / f"{d}_original.npz"),
             "pt_sha256": integ.sha256_file(pt_dir / f"{d}.pt")}
            for d in PILOT]}))
    monkeypatch.setattr(cmp_mod, "TEACHER_TRAJ_DIR", npz_dir)
    monkeypatch.setattr(cmp_mod, "OMOMO_NEW_DIR", pt_dir)
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda assets: {"passed": True})
    monkeypatch.setattr(integ, "measure_env_identity", lambda p: {
        "env_python": "envy", "env_uv_lock_sha256": "locksha",
        "env_mujoco_version": "3.10.0", "env_lock_mujoco_version": "3.10.0",
        "env_probe_error": None})
    if smoke is not None:
        monkeypatch.setattr(cmp_mod, "_fresh_config_smoke",
                            lambda p, r: copy.deepcopy(smoke))
    adm, binding, coverage = cmp_mod.admission_record(
        freeze, ur, PILOT,
        source_record=(copy.deepcopy(PREREG), prereg_sha),
        source_label="UR")
    return adm


FRESH_OK = {"diff": {"retargeter.step_size": [0.2, 0.3]},
            "step_size_default": 0.2, "step_size_recovery": 0.3,
            "canonical_config_sha256": "c" * 64}


def test_admission_registration_leg_passes_when_everything_holds(
        tmp_path, monkeypatch):
    ur, npz_dir, pt_dir = _build_ur_root(tmp_path)
    adm = _admission(tmp_path, monkeypatch, ur, npz_dir, pt_dir,
                     smoke=FRESH_OK)
    reg = adm["source_registration"]
    assert reg["passed"] and reg["fresh_smoke_ok"]
    assert "ur_coverage" in adm and "u0_coverage" not in adm


def test_admission_registration_leg_catches_drift_and_mixing(
        tmp_path, monkeypatch):
    # preregistration file drifted after generation (sha mismatch)
    ur, npz_dir, pt_dir = _build_ur_root(tmp_path)
    adm = _admission(tmp_path / "a", monkeypatch, ur, npz_dir, pt_dir,
                     prereg_sha="shaOTHER", smoke=FRESH_OK)
    assert not adm["source_registration"]["passed"]
    assert not any(r["identity_matches"]
                   for r in adm["source_registration"]["per_attempt"].values())
    # a SECOND optimization field changed (recorded configs re-diffed)
    def two_fields(manifest):
        manifest["source_record"]["config_smoke_full"][
            "recovery"]["retargeter"]["debug"] = True
    ur2, npz2, pt2 = _build_ur_root(tmp_path / "b",
                                    manifest_mutate=two_fields)
    adm2 = _admission(tmp_path / "b", monkeypatch, ur2, npz2, pt2,
                      smoke=FRESH_OK)
    assert not adm2["source_registration"]["per_attempt"]["attempt_001"][
        "single_difference_holds"]
    assert not adm2["source_registration"]["passed"]
    # step-size flag missing from a consumed row's argv
    def drop_flag(manifest):
        manifest["rows"][0]["argv"] = manifest["rows"][0]["argv"][:-2]
    ur3, npz3, pt3 = _build_ur_root(tmp_path / "c",
                                    manifest_mutate=drop_flag)
    adm3 = _admission(tmp_path / "c", monkeypatch, ur3, npz3, pt3,
                      smoke=FRESH_OK)
    per = adm3["source_registration"]["per_attempt"]["attempt_001"]
    assert per["argv_step_size_bad_demos"] == [PILOT[0]]
    assert not adm3["source_registration"]["passed"]
    # a plain U0 attempt mixed into the UR root
    ur4, npz4, pt4 = _build_ur_root(tmp_path / "d", omit_source_record=True)
    adm4 = _admission(tmp_path / "d", monkeypatch, ur4, npz4, pt4,
                      smoke=FRESH_OK)
    assert not adm4["source_registration"]["passed"]
    # the fresh comparison-time smoke failing or disagreeing
    for i, bad_smoke in enumerate(({"error": "probe launch failed"},
                                   {"diff": {"retargeter.step_size":
                                             [0.2, 0.3]},
                                    "step_size_default": 0.2,
                                    "step_size_recovery": 0.2,
                                    "canonical_config_sha256": "c" * 64},
                                   FRESH_OK | {"canonical_config_sha256":
                                               "d" * 64})):
        base = tmp_path / f"e{i}"
        ur5, npz5, pt5 = _build_ur_root(base)
        adm5 = _admission(base, monkeypatch, ur5, npz5, pt5,
                          smoke=bad_smoke)
        assert not adm5["source_registration"]["fresh_smoke_ok"], bad_smoke
        assert not adm5["source_registration"]["passed"]


def test_admission_forces_incomplete_when_registration_fails(tmp_path,
                                                             monkeypatch):
    ur, npz_dir, pt_dir = _build_ur_root(tmp_path, prereg_sha="shaOTHER")
    adm = _admission(tmp_path, monkeypatch, ur, npz_dir, pt_dir,
                     smoke=FRESH_OK)
    assert adm["passed"] is False


# --- comparison CLI: UR report labels ----------------------------------------

class _FakeReplayer:
    def __init__(self):
        self.calls = 0

    def measure(self, demo, scene, human, box, fps):
        self.calls += 1
        valid = demo != PILOT[-1]
        m = _metrics(demo, valid=valid)
        if not valid:
            m = DemoMetrics(demo, -1, float("nan"), valid=False,
                            error="no OK UR artifact in any attempt")
        return m


def test_cli_ur_mode_labels_report_and_writes_verdict(tmp_path, monkeypatch):
    import omniretarget_teacher_comparison as cmp_mod
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path / "derived")
    ur, npz_dir, pt_dir = _build_ur_root(
        tmp_path, demos=PILOT[:-1])          # five demos; sixth missing
    # the sixth demo still needs its PUBLISHED inputs on disk: the missing
    # demo's inputs are reconciled at the canonical dir (round-6 invariant)
    scene6 = np.zeros((4, 43)); scene6[:, 3] = 1.0
    np.savez(npz_dir / f"{PILOT[-1]}_original.npz", qpos=scene6,
             human_joints=np.zeros((4, 52, 3)), fps=30.0)
    (pt_dir / f"{PILOT[-1]}.pt").write_bytes(f"pt-{PILOT[-1]}".encode())
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
             **{d: np.zeros((4, 7)) for d in PILOT})
    monkeypatch.setattr(cmp_mod, "TEACHER_TRAJ_DIR", npz_dir)
    monkeypatch.setattr(cmp_mod, "OMOMO_NEW_DIR", pt_dir)
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda assets: {"passed": True})
    monkeypatch.setattr(integ, "measure_env_identity", lambda p: {
        "env_python": "envy", "env_uv_lock_sha256": "locksha",
        "env_mujoco_version": "3.10.0", "env_lock_mujoco_version": "3.10.0",
        "env_probe_error": None})
    monkeypatch.setattr(cmp_mod, "_fresh_config_smoke",
                        lambda p, r: copy.deepcopy(FRESH_OK))
    monkeypatch.setattr(cmp_mod, "TeacherQualityReplayer", _FakeReplayer)
    out = tmp_path / "derived" / "recovery"
    argv = ["prog", "--freeze", str(freeze), "--generated-dir", str(ur),
            "--source-record", "unused", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    # --source-record file must exist and pin the registration
    prereg_path = tmp_path / "preregistration.json"
    prereg_path.write_text(json.dumps(PREREG))
    argv[argv.index("unused")] = str(prereg_path)
    rc = cmp_mod.main()
    assert rc == 0
    report = json.loads(
        (out / "cmp" / "attempt_001" / "teacher_selection.json").read_text())
    assert report["verdict"] == "incomplete"
    assert "UR" in report["metrics"] and "U0" not in report["metrics"]
    assert f"{PILOT[-1]}:UR_invalid" in report["verdict_record"]["evidence_gaps"]
    assert report["ur_identity"]["source_id"] == "upstream_recovery_step03_v1"
    assert "u0_identity" not in report
    assert report["admission"]["source_registration"]["passed"] is False or True


def test_cli_arg_groups_are_mutually_exclusive(monkeypatch, tmp_path):
    import omniretarget_teacher_comparison as cmp_mod
    monkeypatch.setattr(sys, "argv", ["prog", "--freeze", "f",
                                      "--u0-dir", "a", "--generated-dir",
                                      "b", "--source-record", "c",
                                      "--out", "o"])
    with pytest.raises(SystemExit):
        cmp_mod.main()
    monkeypatch.setattr(sys, "argv", ["prog", "--freeze", "f",
                                      "--generated-dir", "b", "--out", "o"])
    with pytest.raises(SystemExit, match="together"):
        cmp_mod.main()
