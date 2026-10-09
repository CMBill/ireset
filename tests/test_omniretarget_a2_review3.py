"""A2 third-round review counterexamples (independent review, 2026-10-09).

Every test fails on the pre-fix implementation:

1  non-finite evidence judged complete (drift NaN unwatched, isnan lets Inf
   through, a None drift collapsed to no_quality_gain instead of
   incomplete);
2  NEW severe validator failures (joint limits / velocities) and a 0 -> 100%
   penetration violation-ratio increase never blocked local_preferred;
3  the generation entry re-recorded inputs instead of verifying them
   against the A1-frozen hashes, and the frozen asset hashes were never
   re-verified;
4  admission verified the CURRENT checkout but not the generation-time
   checkout recorded in the consumed attempt, and compared only the env
   lock sha (wrong mujoco record / drifted binary / failed probe passed);
5  robot labels in the converted NPZ could be rewritten without detection
   (no raw hash check, no raw->converted array recomputation);
6  the comparison overwrote teacher_selection.json (no attempt isolation);
7  a broken product (missing human_joints) crashed before any report was
   written instead of registering incomplete;
8  a fresh --out directory crashed the report write (no mkdir).
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


# --- helpers -----------------------------------------------------------------


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
    """u_over entry that gains both-hands coverage (feeds the g4 pass)."""
    return {"both": {"denominator_frames": 30, "valid_coverage": 0.55,
                     "gap_series_p95_m": 0.05}}


def _run(u_over=None, p_over=None):
    p = {d: _metrics(d, **(p_over.get(d, {}) if p_over else {})) for d in PILOT}
    u = {d: _metrics(d, **(u_over.get(d, {}) if u_over else {})) for d in PILOT}
    return evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT}), p, u


# --- review 1: non-finite evidence --------------------------------------------


def test_nonfinite_drift_is_incomplete_even_with_full_gains():
    # NaN hand drift on a gaining demo: g5's comparison is NaN-blind
    # (NaN > x is False) and the old gap check never looked at drift
    u = {d: _metrics(d, **_gains(d)) for d in PILOT[:4]}
    u[PILOT[0]] = _metrics(PILOT[0], **{**_gains(PILOT[0]), "hand_drift": float("nan")})
    u[PILOT[4]] = _metrics(PILOT[4])
    u[PILOT[5]] = _metrics(PILOT[5])
    v = evaluate_local_preferred(PILOT, {d: _metrics(d) for d in PILOT}, u,
                                 {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert any("hand_l_missing_with_reference_frames" in g or
               "hand_r_missing_with_reference_frames" in g
               for g in v["evidence_gaps"])

    # a MISSING (None) drift is absent evidence, not a quality verdict:
    # pre-fix this collapsed to no_quality_gain, opening the W1 path
    u2 = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u2[PILOT[0]].hand["l"].drift_max_m = None
    v2 = evaluate_local_preferred(PILOT, {d: _metrics(d) for d in PILOT}, u2,
                                  {d: True for d in PILOT})
    assert v2["verdict"] == "incomplete"


def test_infinite_task_error_on_either_side_is_incomplete():
    # P task = Inf: isnan(Inf) is False, so the old check let it through and
    # g2 even passed against an infinite bound -> local_preferred
    u_over = {d: _gains(d) for d in PILOT[:4]}
    v, *_ = _run(u_over=u_over, p_over={PILOT[1]: {"task": float("inf")}})
    assert v["verdict"] == "incomplete"
    assert any("P_task_mpjpe_p95_m_not_finite" in g for g in v["evidence_gaps"])

    v2, *_ = _run(u_over={PILOT[2]: {"task": float("inf")}})
    assert v2["verdict"] == "incomplete"
    assert any("U0_task_mpjpe_p95_m_not_finite" in g for g in v2["evidence_gaps"])


def test_nonfinite_foot_and_both_hand_fields_are_incomplete():
    foot_inf = SideFootMetrics(30, 0.003, float("inf"), 0.01, 29)
    u_none = {d: _metrics(d) for d in PILOT}
    u_none[PILOT[0]].foot["l"] = None
    v = evaluate_local_preferred(PILOT, {d: _metrics(d) for d in PILOT}, u_none,
                                 {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"

    u3 = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u3[PILOT[0]].foot["r"] = foot_inf
    v3 = evaluate_local_preferred(PILOT, {d: _metrics(d) for d in PILOT}, u3,
                                  {d: True for d in PILOT})
    assert v3["verdict"] == "incomplete"
    assert any("foot_r_missing_with_reference_frames" in g
               for g in v3["evidence_gaps"])

    # both-hands series non-finite on a positive denominator
    u4 = {d: _metrics(d, **_gains(d)) for d in PILOT}
    u4[PILOT[0]] = _metrics(PILOT[0], both={"denominator_frames": 30,
                                            "valid_coverage": 0.55,
                                            "gap_series_p95_m": float("nan")})
    v4 = evaluate_local_preferred(PILOT, {d: _metrics(d) for d in PILOT}, u4,
                                  {d: True for d in PILOT})
    assert v4["verdict"] == "incomplete"
    assert any("both_hands_not_finite" in g for g in v4["evidence_gaps"])


# --- review 2: g5 covers new hard failures and the violation ratio ------------


def test_g5_blocks_new_validator_failures_despite_all_gains():
    for severe in (["joint_limits"], ["velocities"]):
        u_over = {d: _gains(d) for d in PILOT[:4]}
        u_over[PILOT[0]] = {**_gains(PILOT[0]), "failed": severe}
        v, *_ = _run(u_over=u_over)
        assert v["verdict"] == "no_quality_gain", severe
        detail = v["gates"]["g5_degradation"]["per_demo"][PILOT[0]]["degradation"]
        assert detail["new_validator_failures"]["new"] == severe


def test_g5_blocks_penetration_violation_ratio_increase():
    # pen max within g3's bound on both sides; the RATIO mass is the tell
    u_over = {d: _gains(d) for d in PILOT[:4]}
    u_over[PILOT[0]] = {**_gains(PILOT[0]), "pen": 0.0019, "pen_ratio": 1.0}
    v, *_ = _run(u_over=u_over,
                 p_over={d: {"pen": 0.0019, "pen_ratio": 0.0} for d in PILOT})
    assert v["gates"]["g3_penetration_and_feet"]["per_demo"][PILOT[0]]["passed"]
    assert v["verdict"] == "no_quality_gain"
    detail = v["gates"]["g5_degradation"]["per_demo"][PILOT[0]]["degradation"]
    assert "penetration_violation_ratio_increased" in detail


def test_g5_allows_failures_p_already_had_and_ratio_decrease():
    # U0 failing a subset of P's failures is an improvement, not a new failure
    u_over = {d: {**_gains(d), "failed": ["joint_limits"]} for d in PILOT[:4]}
    v, *_ = _run(u_over=u_over,
                 p_over={d: {"failed": ["joint_limits", "velocities"],
                             "pen_ratio": 0.5} for d in PILOT})
    assert v["verdict"] == "local_preferred"


# --- review 3: inputs and assets verified against the A1 FREEZE ---------------


def test_verify_assets_against_freeze_detects_drift_and_missing(tmp_path):
    (tmp_path / "a.obj").write_bytes(b"asset-a")
    (tmp_path / "b.obj").write_bytes(b"asset-b")
    assets = {
        "published": {"models": {
            "a.obj": {"bytes": 7, "sha256": integ.sha256_file(tmp_path / "a.obj")}}},
        "largebox": {
            "b.obj": {"bytes": 7, "sha256": "0" * 64},          # drifted
            "gone.obj": {"bytes": 1, "sha256": "1" * 64},        # missing
        },
        "upstream": {"height_dict": "a.obj",
                     "height_dict_sha256": integ.sha256_file(tmp_path / "a.obj")},
    }
    rec = integ.verify_assets_against_freeze(assets, root=tmp_path)
    assert rec["n_checked"] == 3          # a.obj + b.obj + height_dict(a.obj)
    assert rec["mismatched"] == ["b.obj"] and rec["missing"] == ["gone.obj"]
    assert not rec["passed"]

    assets["largebox"]["b.obj"]["sha256"] = integ.sha256_file(tmp_path / "b.obj")
    del assets["largebox"]["gone.obj"]
    ok = integ.verify_assets_against_freeze(assets, root=tmp_path)
    assert ok["passed"] and ok["n_checked"] == 3


def _run_generation_argv(monkeypatch, tmp_path, freeze, out_dir):
    import generate_omniretarget_teachers as gen

    calls = []
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
    monkeypatch.setattr(
        gen, "generate_one",
        lambda demo, env_python, attempt_dir, data_path: (
            calls.append(demo),
            {"demo": demo, "argv": [], "cwd": "", "exit_code": 0, "wall_s": 0.1,
             "config": "fake", "status": "ok", "binding": {"passed": True},
             "error": None, "converted": f"{attempt_dir.name}/{demo}_c.npz"})[1])
    monkeypatch.setattr(integ, "sha256_file", lambda p: "fakehash")
    monkeypatch.setattr(
        integ, "measure_env_identity",
        lambda p: {"env_python": "x", "env_uv_lock_sha256": "locksha",
                   "env_mujoco_version": "3.10.0",
                   "env_lock_mujoco_version": "3.10.0", "env_probe_error": None})
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: {"n_checked": 1, "mismatched": [],
                                              "missing": [], "passed": True})
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--freeze", str(freeze), "--out-dir", str(out_dir)])
    return gen.main(), calls


def _freeze_with_input_hashes(tmp_path, npz_sha, pt_sha):
    freeze = tmp_path / "freeze"
    freeze.mkdir(exist_ok=True)
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [
            {"demo": d, "npz_sha256": npz_sha, "pt_sha256": pt_sha}
            for d in PILOT]}))
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit":
                      integ.read_checkout_identity()["holosoma_revision"]}}))
    return freeze


def test_generation_refuses_freeze_input_drift_before_any_call(tmp_path, monkeypatch):
    freeze = _freeze_with_input_hashes(tmp_path, "wronghash", "fakehash")
    with pytest.raises(SystemExit, match="A1-frozen hashes"):
        _run_generation_argv(monkeypatch, tmp_path, freeze, tmp_path / "u0")
    # nothing was generated: the gate fires before any teacher runs


def test_generation_refuses_asset_and_env_binary_drift(tmp_path, monkeypatch):
    import generate_omniretarget_teachers as gen

    freeze = _freeze_with_input_hashes(tmp_path, "fakehash", "fakehash")

    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
    monkeypatch.setattr(gen, "generate_one",
                        lambda *a, **k: pytest.fail("must not generate"))
    monkeypatch.setattr(integ, "sha256_file", lambda p: "fakehash")

    bad_assets = {"n_checked": 1, "mismatched": ["x.obj"], "missing": [],
                  "passed": False}
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: bad_assets)
    monkeypatch.setattr(integ, "measure_env_identity",
                        lambda p: {"env_python": "x", "env_uv_lock_sha256": "locksha",
                                   "env_mujoco_version": "3.10.0",
                                   "env_lock_mujoco_version": "3.10.0",
                                   "env_probe_error": None})
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--freeze", str(freeze), "--out-dir",
                         str(tmp_path / "u0")])
    with pytest.raises(SystemExit, match="assets drifted"):
        gen.main()

    # env binary drifted from its own lock (lock sha unchanged)
    monkeypatch.setattr(integ, "verify_assets_against_freeze",
                        lambda a, root=None: {"n_checked": 1, "mismatched": [],
                                              "missing": [], "passed": True})
    monkeypatch.setattr(integ, "measure_env_identity",
                        lambda p: {"env_python": "x", "env_uv_lock_sha256": "locksha",
                                   "env_mujoco_version": "3.15.0",
                                   "env_lock_mujoco_version": "3.10.0",
                                   "env_probe_error": None})
    with pytest.raises(SystemExit, match="drifted from its own lock"):
        gen.main()


# --- review 4: generation-time identity and the env triple check --------------


def test_parse_lock_mujoco_version():
    lock = '''version = 1
[[package]]
name = "numpy"
version = "2.3.5"

[[package]]
name = "mujoco"
version = "3.10.0"
source = "registry+https://pypi.org/simple"
'''
    assert integ.parse_lock_mujoco_version(lock) == "3.10.0"
    assert integ.parse_lock_mujoco_version('version = 1\n[[package]]\nname = "x"\n') is None


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


def _write_pub(path, scene, human):
    q = scene.copy()                       # published [qw,qx,qy,qz,x,y,z]
    q[:, 0:4], q[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
    q[:, 36:40], q[:, 40:43] = scene[:, 39:43], scene[:, 36:39]
    np.savez(path, qpos=q, human_joints=human, fps=30.0)


def _write_upstream(path, scene, human):
    np.savez(path, qpos=scene.copy(), human_joints=human, fps=30.0)


def _build_u0_root(tmp_path, *, demos, gen_revision=None, gen_mujoco="3.10.0",
                   tamper_converted=None):
    """Full fake I0 layout: attempts, manifests, raw+converted, published P."""
    rng = np.random.default_rng(11)
    npz_dir = tmp_path / "pub"; npz_dir.mkdir(parents=True)
    pt_dir = tmp_path / "pt"; pt_dir.mkdir()
    u0 = tmp_path / "u0"
    att = u0 / "attempt_001"; att.mkdir(parents=True)
    rows, inputs = [], {}
    scenes = {}
    for demo in demos:
        scene, human = _scenes(rng)
        scenes[demo] = (scene, human)
        _write_pub(npz_dir / f"{demo}_original.npz", scene, human)
        (pt_dir / f"{demo}.pt").write_bytes(f"pt-{demo}".encode())
        _write_upstream(att / f"{demo}_original.npz", scene, human)
        _write_upstream(att / f"{demo}_original_converted.npz", scene, human)
        rows.append({"demo": demo, "status": "ok", "binding": {"passed": True},
                     "raw_sha256": integ.sha256_file(att / f"{demo}_original.npz")})
        inputs[demo] = {"npz": integ.sha256_file(npz_dir / f"{demo}_original.npz"),
                        "pt": integ.sha256_file(pt_dir / f"{demo}.pt")}
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (att / "generation_manifest.json").write_text(json.dumps({
        "rows": rows, "input_sha256": inputs,
        "identity": {
            "checkout": {"holosoma_revision": gen_revision or expected,
                         "holosoma_dirty": False},
            "env": {"env_python": "envy", "env_uv_lock_sha256": "locksha",
                    "env_mujoco_version": gen_mujoco},
        }}))
    (u0 / "attempts_index.json").write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"}]}))
    if tamper_converted:
        demo = tamper_converted
        arr = np.load(att / f"{demo}_original_converted.npz")
        q = arr["qpos"].copy(); q[:, 7] += 0.2          # robot label rewrite
        np.savez(att / f"{demo}_original_converted.npz", qpos=q,
                 human_joints=arr["human_joints"], fps=30.0)
    return u0, npz_dir, pt_dir


def _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir, *, probe_mujoco="3.10.0",
               probe_lock_version="3.10.0", probe_error=None, entries=None):
    import omniretarget_teacher_comparison as cmp_mod

    freeze = tmp_path / "freeze"; freeze.mkdir(exist_ok=True)
    expected = integ.read_checkout_identity()["holosoma_revision"]
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": expected}}))
    # round-4: the comparison also reconciles current inputs against the
    # A1-frozen hashes, so the fake freeze carries per-demo input hashes
    # (default: the healthy on-disk values; counterexamples pass entries)
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
                                   "env_mujoco_version": probe_mujoco,
                                   "env_lock_mujoco_version": probe_lock_version,
                                   "env_probe_error": probe_error})
    return cmp_mod.admission_record(freeze, u0, PILOT)


def test_admission_catches_generation_checkout_and_env_drift(tmp_path, monkeypatch):
    u0, npz_dir, pt_dir = _build_u0_root(tmp_path, demos=PILOT)

    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert adm["passed"], adm

    # wrong GENERATION-time checkout recorded in the consumed manifest
    u0_bad, npz2, pt2 = _build_u0_root(tmp_path / "b", demos=PILOT,
                                       gen_revision="deadbeef")
    adm2, *_ = _admission(tmp_path / "b", monkeypatch, u0_bad, npz2, pt2)
    assert not adm2["passed"]
    assert not adm2["generation_checkout"][PILOT[0]]["matches_freeze"]

    # mujoco record disagrees with the binary actually importable now
    u0_315, npz3, pt3 = _build_u0_root(tmp_path / "c", demos=PILOT,
                                       gen_mujoco="3.15.0")
    adm3, *_ = _admission(tmp_path / "c", monkeypatch, u0_315, npz3, pt3)
    assert not adm3["passed"] and not adm3["env_mujoco"]["consistent"]

    # lock unchanged, binary drifted (probe != lock-pinned version)
    adm4, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir,
                          probe_mujoco="3.15.0", probe_lock_version="3.10.0")
    assert not adm4["passed"] and not adm4["env_mujoco"]["consistent"]

    # the probe itself failing
    adm5, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir,
                          probe_mujoco=None, probe_error="boom")
    assert not adm5["passed"] and not adm5["env_mujoco"]["consistent"]


def test_admission_catches_converted_label_rewrite(tmp_path, monkeypatch):
    u0, npz_dir, pt_dir = _build_u0_root(tmp_path, demos=PILOT,
                                         tamper_converted=PILOT[2])
    adm, binding, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, pt_dir)
    assert not adm["passed"]
    chain = adm["artifact_chain"][PILOT[2]]
    assert chain["raw_hash_matches"] and not chain["conversion_reproduced"]
    assert not chain["passed"]


# --- review 5 + 7: artifact chain, recorded failures instead of crashes --------


def test_artifact_chain_catches_each_tamper_without_raising(tmp_path):
    u0, _, _ = _build_u0_root(tmp_path, demos=[PILOT[0]])
    att = u0 / "attempt_001"
    manifest = json.loads((att / "generation_manifest.json").read_text())
    row = manifest["rows"][0]

    ok = integ.verify_artifact_chain(u0, "attempt_001", PILOT[0], row)
    assert ok["passed"] and ok["error"] is None

    # converted joint column rewritten, raw kept intact
    arr = np.load(att / f"{PILOT[0]}_original_converted.npz")
    q = arr["qpos"].copy(); q[:, 7] += 0.2
    np.savez(att / f"{PILOT[0]}_original_converted.npz", qpos=q,
             human_joints=arr["human_joints"], fps=30.0)
    conv_bad = integ.verify_artifact_chain(u0, "attempt_001", PILOT[0], row)
    assert not conv_bad["passed"] and not conv_bad["conversion_reproduced"]

    # raw file itself rewritten (recorded hash anchors it)
    raw_arr = np.load(att / f"{PILOT[0]}_original.npz")
    rq = raw_arr["qpos"].copy(); rq[:, 8] += 0.3
    np.savez(att / f"{PILOT[0]}_original.npz", qpos=rq,
             human_joints=raw_arr["human_joints"], fps=30.0)
    raw_bad = integ.verify_artifact_chain(u0, "attempt_001", PILOT[0], row)
    assert not raw_bad["passed"] and not raw_bad["raw_hash_matches"]

    # a broken product (missing human_joints) is RECORDED, never raised
    np.savez(att / f"{PILOT[0]}_original_converted.npz", qpos=rq, fps=30.0)
    broken = integ.verify_artifact_chain(u0, "attempt_001", PILOT[0], row)
    assert not broken["passed"] and broken["error"]


# --- review 6 + 8: comparison attempt isolation and output-dir creation -------


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


def test_comparison_writes_isolated_attempts_and_creates_out_dir(
        tmp_path, monkeypatch):
    import omniretarget_teacher_comparison as cmp_mod

    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)
    u0, npz_dir, pt_dir = _build_u0_root(tmp_path, demos=PILOT)

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

    # fresh, not-yet-existing nested output directory (review 8)
    out = tmp_path / "experiments" / "a2run" / "teacher_pilot"
    argv = ["prog", "--freeze", str(freeze), "--u0-dir", str(u0), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert cmp_mod.main() == 0
    first = out / "cmp" / "attempt_001" / "teacher_selection.json"
    assert first.is_file()
    rec1 = json.loads(first.read_text())
    assert rec1["verdict"] == "no_quality_gain"      # complete, g4 no gains
    assert rec1["verdict_record"]["complete_evidence"] is True
    first_bytes = first.read_bytes()

    monkeypatch.setattr(sys, "argv", argv)
    assert cmp_mod.main() == 0
    second = out / "cmp" / "attempt_002" / "teacher_selection.json"
    assert second.is_file()
    assert first.read_bytes() == first_bytes         # prior verdict untouched
    index = json.loads((out / "cmp" / "attempts_index.json").read_text())
    assert [a["attempt"] for a in index["attempts"]] == ["attempt_001", "attempt_002"]
    assert {a["verdict"] for a in index["attempts"]} == {"no_quality_gain"}
