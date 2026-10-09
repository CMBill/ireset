"""A2 review-fix counterexamples (independent review 2026-10-09, 11 items).

Every test here fails on the pre-fix implementation:

1  task MPJPE broadcast over the whole human tensor instead of frame-aligned;
2  penetration from a fixed-midpoint linear-average protocol instead of the
   validator's adaptive SLERP protocol;
3  foot speed/drift at the sole centroid (blind to foot rotation) and the
   frame-0 placeholder zero entering the statistics;
4  §3.4 item 5 absent from the verdict (hand/foot drift 100 m still
   local_preferred);
5  identity/binding trusted from constants and manifest status strings;
6  ``--demos`` retry overwriting the six-demo evidence with a one-demo
   COMPLETE manifest;
7  the output guard bypassed through the OMOMO symlink;
8  NaN / missing side measurements judged complete evidence;
9  inferred contact intervals counted as certified without interval quality;
10 relative/out-of-project paths crashing the report writes;
11 (documentation fix — sub12 progress points; no test, see ENV_FIX_NOTE).
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
from ireset.data import omniretarget_sources as orc  # noqa: E402
from ireset.embodied.contact_sites import ContactInterval  # noqa: E402
from ireset.eval.teacher_comparison import (  # noqa: E402
    DRIFT_DEGRADATION_MARGIN_M, DemoMetrics, SideFootMetrics, SideHandMetrics,
    TeacherQualityReplayer, certified_robot_mask, evaluate_local_preferred,
    foot_side_metrics, hand_side_metrics,
)
from ireset.paths import DERIVED_DIR, PROJECT_ROOT, path_for_record  # noqa: E402
from ireset.repair.validation import ValidationConfig  # noqa: E402

PILOT = [f"subA_largebox_{i:03d}" for i in range(1, 7)]


# --- helpers -----------------------------------------------------------------


def _metrics(demo, *, pen=0.001, task=0.10, foot=True, both=None, valid=True,
             hand_drift=0.02, foot_drift=0.01, ref_support=None, ref_grasp=None):
    foot_m = {s: (SideFootMetrics(30, 0.003, 0.01, foot_drift, 29) if foot else None)
              for s in ("l", "r")}
    hand_m = {s: SideHandMetrics(30, 0.02, 0.5, 0.5, 10, 5, hand_drift) for s in ("l", "r")}
    both_default = {"denominator_frames": 30, "valid_coverage": 0.40,
                    "gap_series_p95_m": 0.05}
    return DemoMetrics(
        demo=demo, T=100, fps=30.0, valid=valid, penetration_max_m=pen,
        penetration_violation_ratio=0.0, task_mpjpe_p95_m=task,
        foot=foot_m, hand=hand_m, both=(both_default if both is None else both),
        ref_support_frames=(ref_support if ref_support is not None else {"l": 30, "r": 30}),
        ref_grasp_frames=(ref_grasp if ref_grasp is not None else {"l": 30, "r": 30}),
    )


def _run(u_over=None, p_over=None):
    p = {d: _metrics(d, **(p_over.get(d, {}) if p_over else {})) for d in PILOT}
    u = {d: _metrics(d, **(u_over.get(d, {}) if u_over else {})) for d in PILOT}
    binding = {d: True for d in PILOT}
    return evaluate_local_preferred(PILOT, p, u, binding), p, u


# --- review 4: degradation gate (§3.4 item 5) --------------------------------


def test_g5_blocks_hundred_meter_drift_despite_all_other_gains():
    # every other gate passes (coverage gains on 4 demos) but U0 drifts 100 m
    u_over = {d: {"both": {"denominator_frames": 30, "valid_coverage": 0.55,
                           "gap_series_p95_m": 0.05}}
              for d in PILOT[:4]}
    u_over[PILOT[0]] = {**u_over[PILOT[0]], "hand_drift": 100.0}
    v, *_ = _run(u_over=u_over)
    assert v["gates"]["g5_degradation"]["per_demo"][PILOT[0]]["passed"] is False
    assert v["verdict"] == "no_quality_gain", (
        "§3.4 item 5: drift-type degradation must block local_preferred")

    foot_drift_case = {PILOT[1]: {"both": {"denominator_frames": 30,
                                           "valid_coverage": 0.55,
                                           "gap_series_p95_m": 0.05},
                                  "foot_drift": 100.0}}
    v2, *_ = _run(u_over=foot_drift_case)
    assert v2["verdict"] == "no_quality_gain"

    # within the registered margin is allowed (equal P/U0 drift passes)
    v3, *_ = _run(u_over={d: {"hand_drift": 0.02 + DRIFT_DEGRADATION_MARGIN_M - 1e-9,
                              "both": {"denominator_frames": 30,
                                       "valid_coverage": 0.55,
                                       "gap_series_p95_m": 0.05}}
                        for d in PILOT[:4]})
    assert v3["verdict"] == "local_preferred"


# --- review 8: evidence completeness -----------------------------------------


def test_nan_task_and_missing_side_measurements_are_incomplete_not_complete():
    u_over = {PILOT[0]: {"task": float("nan")}}
    v, *_ = _run(u_over=u_over)
    assert v["verdict"] == "incomplete"
    assert any("nan_task" in g for g in v["evidence_gaps"])

    # reference support frames exist but the U0 foot measurement is absent
    u_over2 = {PILOT[1]: {"foot": False}}
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d) for d in PILOT}
    u[PILOT[1]].foot["l"] = None
    v2 = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v2["verdict"] == "incomplete"
    assert any("U0_foot_l_missing_with_reference_frames" in g for g in v2["evidence_gaps"])


def test_missing_rows_do_not_crash_the_gates():
    p = {d: _metrics(d) for d in PILOT}
    u = {d: _metrics(d) for d in PILOT[:-1]}          # one demo has no row at all
    v = evaluate_local_preferred(PILOT, p, u, {d: True for d in PILOT})
    assert v["verdict"] == "incomplete"
    assert any("U0_row_missing" in g for g in v["evidence_gaps"])
    for g in v["gates"].values():
        assert isinstance(g["passed"], bool)           # no KeyError escaped


# --- review 3: per-site worst foot protocol -----------------------------------


def _rotating_sole_pts(T=20, r=0.05, dtheta=0.35):
    """5 sole sites rotating around a FIXED centroid: centroid speed is 0."""
    ang = np.arange(5) * 2 * np.pi / 5
    pts = np.zeros((T, 5, 3))
    for t in range(T):
        a = ang + t * dtheta
        pts[t, :, 0] = r * np.cos(a)
        pts[t, :, 1] = r * np.sin(a)
    return pts


def test_foot_speed_sees_pivoting_foot_and_excludes_frame_zero():
    fps = 30.0
    pts = _rotating_sole_pts()
    gap = np.full(pts.shape[0], 0.002)
    m = foot_side_metrics(pts, gap, [(0, pts.shape[0])], fps)
    chord = 2 * 0.05 * np.sin(0.35 / 2)               # per-site per-frame displacement
    assert m.speed_p95_mps == pytest.approx(chord * fps, rel=1e-6)
    assert m.drift_max_m > 0.03                        # per-site anchor drift seen
    assert m.n_speed_frames == pts.shape[0] - 1        # frame 0 placeholder excluded

    # two-frame interval, motion only between frames 0 and 1: the ONLY speed
    # sample is frame 1 — if the frame-0 placeholder zero were included,
    # p95([0, v]) would dilute to 0.95*v
    pts2 = np.zeros((2, 5, 3))
    pts2[1, :, 0] = 0.1
    m2 = foot_side_metrics(pts2, np.full(2, 0.002), [(0, 2)], fps)
    assert m2.speed_p95_mps == pytest.approx(0.1 * fps, rel=1e-9)
    assert m2.n_speed_frames == 1


# --- review 9: certification requires interval quality ------------------------


def _cfg():
    return ValidationConfig()


def test_slowly_drifting_grasp_is_not_certified():
    T = 30
    hand_dist = np.full(T, 0.005)                       # gap always fine
    hand_local = np.zeros((T, 3, 3))
    drift = np.linspace(0.0, 0.10, T)                   # 10 cm box-frame drift
    hand_local[:, 0, 0] = drift
    iv = ContactInterval("left_hand", 0, T, "object", anchor=hand_local[0].copy())
    mask, n_iv, n_cert = certified_robot_mask([iv], hand_dist, hand_local, _cfg())
    assert n_iv == 1 and n_cert == 0                    # anchor check fails
    assert not mask.any()

    hand_local_ok = hand_local.copy()
    hand_local_ok[:, 0, 0] = np.linspace(0.0, 0.005, T)  # 5 mm drift: inside gate
    iv_ok = ContactInterval("left_hand", 0, T, "object", anchor=hand_local_ok[0].copy())
    mask2, n_iv2, n_cert2 = certified_robot_mask([iv_ok], hand_dist, hand_local_ok, _cfg())
    assert n_cert2 == 1 and mask2.all()

    # deep penetration inside the interval also blocks certification
    hand_dist_bad = hand_dist.copy()
    hand_dist_bad[10] = 0.05
    mask3, _, n_cert3 = certified_robot_mask([iv_ok], hand_dist_bad, hand_local_ok, _cfg())
    assert n_cert3 == 0 and not mask3.any()


def test_hand_side_metrics_count_only_certified_reference_overlap():
    T = 30
    hand_dist = np.full(T, 0.005)
    hand_local = np.zeros((T, 3, 3))
    hand_local[:, 0, 0] = np.linspace(0.0, 0.10, T)     # drifts out of gate
    iv = ContactInterval("left_hand", 0, T, "object", anchor=hand_local[0].copy())
    pen = np.zeros(T)
    m = hand_side_metrics(hand_dist, hand_local, [iv], [(0, T)], _cfg(), pen)
    assert m.n_robot_intervals == 1 and m.n_robot_intervals_certified == 0
    assert m.certified_overlap_frames == 0 and m.longest_certified_overlap == 0
    assert m.drift_max_m == pytest.approx(0.10, abs=1e-6)


# --- review 1/2: protocol wiring on the real locked scene ---------------------


def _tiny_scene(T=6, joint_amp=0.25):
    scene = np.zeros((T, 43))
    scene[:, 0:3] = [0.0, 0.0, 0.8]
    scene[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
    scene[:, 7:36] = 0.0
    for t in range(T):                                   # alternating joint motion
        scene[t, 7] = joint_amp if t % 2 else -joint_amp
    scene[:, 36:39] = [3.0, 0.0, 0.2]                    # box far away: no ref intervals
    scene[:, 39:43] = [1.0, 0.0, 0.0, 0.0]
    return scene


@pytest.fixture(scope="module")
def replayer():
    return TeacherQualityReplayer()


def _human_from_fk(replayer, scene):
    """human tensor whose mapped keypoints equal the robot's per-frame FK."""
    T = scene.shape[0]
    human = np.zeros((T, 52, 3))
    for t in range(T):
        replayer._forward(scene, t)
        human[t][replayer.ref_kpt_idx] = replayer.data.site_xpos[replayer.kpt_ids]
    return human


def test_task_mpjpe_is_frame_aligned(replayer):
    scene = _tiny_scene()
    human = _human_from_fk(replayer, scene)
    source_box = np.broadcast_to(np.array([3.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
                                 (scene.shape[0], 7)).copy()
    m = replayer.measure("subX_largebox_001", scene, human, source_box, 30.0)
    assert m.valid
    assert m.task_mpjpe_p95_m == pytest.approx(0.0, abs=1e-12), (
        "per-frame-aligned keypoints match exactly")

    # the pre-fix broadcast (robot frame t vs ALL human frames) yields a
    # strictly positive mean on this alternating-pose scene — the counterexample
    K = human[:, replayer.ref_kpt_idx]                  # == robot keypoints per frame
    buggy = float(np.linalg.norm(K[:, None, :, :] - K[None, :, :, :], axis=-1)
                  .mean())
    assert buggy > 0.01


def test_penetration_max_comes_from_validator_adaptive_protocol(replayer):
    scene = _tiny_scene()
    scene[:, 2] = 0.25                                    # root pushed into the ground
    human = np.zeros((scene.shape[0], 52, 3))
    source_box = np.broadcast_to(np.array([3.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
                                 (scene.shape[0], 7)).copy()
    m = replayer.measure("subX_largebox_002", scene, human, source_box, 30.0)
    rep = replayer.validator.validate(scene, exclude=("box_ground",))
    assert m.penetration_max_m > 0.002                   # really penetrating
    assert m.penetration_max_m == float(rep.checks["penetration"].worst), (
        "reported max must be the validator's adaptive SLERP result")


# --- review 5: artifact-level binding re-verification -------------------------


def test_env_probe_keeps_venv_semantics():
    # regression: resolve() followed the .venv/bin/python symlink to the bare
    # uv-managed interpreter, breaking site-packages (probe: No module mujoco)
    env = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"
    if not env.exists():
        pytest.skip("independent upstream env not present")
    ident = integ.measure_env_identity(env)
    assert ident["env_mujoco_version"] is not None
    assert ident["env_uv_lock_sha256"] is not None


def _write_teacher_npz(path, layout, scene, human, fps):
    q = scene.copy()
    if layout == "published":            # stored [qw,qx,qy,qz,x,y,z] per freejoint
        q[:, 0:4], q[:, 4:7] = scene[:, 3:7], scene[:, 0:3]
        q[:, 36:40], q[:, 40:43] = scene[:, 39:43], scene[:, 36:39]
    np.savez(path, qpos=q, human_joints=human, fps=fps)


def test_binding_recomputed_from_artifacts_catches_tampering(tmp_path):
    rng = np.random.default_rng(7)
    T = 5
    scene = np.zeros((T, 43))
    scene[:, 0:3] = rng.normal(size=(T, 3)) * 0.1
    qq = rng.normal(size=(T, 4)); qq /= np.linalg.norm(qq, axis=1, keepdims=True)
    scene[:, 3:7] = qq
    scene[:, 7:36] = rng.uniform(-0.3, 0.3, (T, 29))
    qb = rng.normal(size=(T, 4)); qb /= np.linalg.norm(qb, axis=1, keepdims=True)
    scene[:, 39:43] = qb
    scene[:, 36:39] = [0.5, 0.0, 0.2]
    human = rng.normal(size=(T, 52, 3))

    pub = tmp_path / "subA_largebox_001_original.npz"
    conv = tmp_path / "subA_largebox_001_original_converted.npz"
    _write_teacher_npz(pub, "published", scene, human, 30.0)
    _write_teacher_npz(conv, "project_scene", scene, human, 30.0)
    ok = integ.verify_binding_from_artifacts(conv, pub)
    assert ok.passed

    tampered = np.load(conv, allow_pickle=False)
    h2 = tampered["human_joints"].copy(); h2[2, 0, 0] += 1e-6
    np.savez(conv, qpos=tampered["qpos"], human_joints=h2,
             fps=float(tampered["fps"]))
    bad = integ.verify_binding_from_artifacts(conv, pub)
    assert not bad.passed                                  # manifest status can't mask this


# --- review 6: attempt isolation and full-denominator reconciliation ----------


def _fake_freeze(tmp_path):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "frozen_lists.json").write_text(json.dumps(
        {"teacher_pilot_train6_v1": [{"demo": d} for d in PILOT]}))
    (freeze / "assets.json").write_text(json.dumps(
        {"upstream": {"expected_commit": integ.read_checkout_identity()["holosoma_revision"]}}))
    return freeze


def _run_generation(monkeypatch, tmp_path, freeze, out_dir, demos=None):
    import generate_omniretarget_teachers as gen

    calls = []
    # tmp outputs: the derived-root guard must follow the patched root
    monkeypatch.setattr(integ, "DERIVED_DIR", tmp_path)

    def fake_generate_one(demo, env_python, attempt_dir, data_path):
        calls.append((demo, attempt_dir.name))
        return {"demo": demo, "argv": [], "cwd": "", "exit_code": 0, "wall_s": 0.1,
                "config": "fake", "status": "ok", "binding": {"passed": True},
                "error": None, "converted": f"{attempt_dir.name}/{demo}_converted.npz"}

    monkeypatch.setattr(gen, "generate_one", fake_generate_one)
    monkeypatch.setattr(integ, "sha256_file", lambda p: "fakehash")
    monkeypatch.setattr(integ, "measure_env_identity",
                        lambda p: {"env_python": "x", "env_uv_lock_sha256": "locksha",
                                   "env_mujoco_version": "3.10.0", "env_probe_error": None})
    argv = ["prog", "--freeze", str(freeze), "--out-dir", str(out_dir)]
    if demos:
        argv += ["--demos", demos]
    monkeypatch.setattr(sys, "argv", argv)
    return gen.main(), calls


def test_retry_writes_new_attempt_and_keeps_full_denominator(tmp_path, monkeypatch):
    freeze = _fake_freeze(tmp_path)
    out_dir = tmp_path / "u0"
    rc1, calls1 = _run_generation(monkeypatch, tmp_path, freeze, out_dir)
    assert rc1 == 0 and len(calls1) == 6
    first_manifest = json.loads(
        (out_dir / "attempt_001" / "generation_manifest.json").read_text())
    assert first_manifest["expected_denominator"] == 6

    rc2, calls2 = _run_generation(monkeypatch, tmp_path, freeze, out_dir,
                                  demos=PILOT[0])
    assert rc2 == 1                                        # 1/6 -> INCOMPLETE, no shrink
    assert calls2[0][1] == "attempt_002"
    m1 = json.loads((out_dir / "attempt_001" / "generation_manifest.json").read_text())
    assert len(m1["rows"]) == 6                            # first evidence intact
    m2 = json.loads((out_dir / "attempt_002" / "generation_manifest.json").read_text())
    assert m2["expected_denominator"] == 6 and m2["attempt_scope"] == [PILOT[0]]
    index = json.loads((out_dir / "attempts_index.json").read_text())["attempts"]
    assert [a["attempt"] for a in index] == ["attempt_001", "attempt_002"]


def test_collect_u0_coverage_reconciles_full_pilot(tmp_path):
    root = tmp_path / "u0"
    for name, status in (("attempt_001", ["ok"] * 6), ("attempt_002", ["binding_failed"])):
        att = root / name
        att.mkdir(parents=True)
        rows = []
        for demo, st in zip(PILOT, status):
            conv = att / f"{demo}_original_converted.npz"
            if st == "ok":
                np.savez(conv, qpos=np.zeros((2, 43)))
            rows.append({"demo": demo, "status": st,
                         "binding": {"passed": st == "ok"}})
        (att / "generation_manifest.json").write_text(json.dumps(
            {"rows": rows, "input_sha256": {}}))
    index = {"attempts": [
        {"attempt": "attempt_001"}, {"attempt": "attempt_002"}]}
    (root / "attempts_index.json").write_text(json.dumps(index))

    cov = integ.collect_u0_coverage(root, PILOT)
    assert cov["missing"] == []
    assert cov["by_demo"][PILOT[0]]["attempt"] == "attempt_001"  # retry failed -> old kept

    # a demo whose converted artifact disappears has NO coverage anywhere
    (root / "attempt_001" / f"{PILOT[5]}_original_converted.npz").unlink()
    cov2 = integ.collect_u0_coverage(root, PILOT)
    assert cov2["missing"] == [PILOT[5]]
    assert PILOT[5] not in cov2["by_demo"]

    with pytest.raises(SystemExit):
        integ.collect_u0_coverage(tmp_path / "no_index", PILOT)   # legacy layout refused


def test_verify_inputs_against_manifest(tmp_path):
    (tmp_path / "d_original.npz").write_bytes(b"npz")
    (tmp_path / "d.pt").write_bytes(b"pt")
    manifest = {"input_sha256": {
        "d": {"npz": integ.sha256_file(tmp_path / "d_original.npz"),
              "pt": integ.sha256_file(tmp_path / "d.pt")}}}
    ok = integ.verify_inputs_against_manifest(manifest, tmp_path, tmp_path)
    assert ok["d"]["npz_matches"] and ok["d"]["pt_matches"]
    (tmp_path / "d.pt").write_bytes(b"pt-changed")
    bad = integ.verify_inputs_against_manifest(manifest, tmp_path, tmp_path)
    assert not bad["d"]["pt_matches"]


# --- review 7: symlink-aware output guard -------------------------------------


def test_output_guard_refuses_resource_symlink_targets():
    ok = integ.guard_output_dir(DERIVED_DIR / "experiments" / "x")
    assert str(ok).startswith(str(DERIVED_DIR))
    # the OMOMO entry is a symlink into the external source tree
    with pytest.raises(SystemExit):
        integ.guard_output_dir(PROJECT_ROOT / "resources" / "data" / "OMOMO_new" / "out")
    with pytest.raises(SystemExit):
        integ.guard_output_dir(PROJECT_ROOT / "third_party" / "holosoma" / "out")
    with pytest.raises(SystemExit):
        integ.guard_output_dir(Path("/tmp/elsewhere"))


# --- review 10: path recording never crashes ----------------------------------


def test_path_for_record_handles_relative_and_out_of_project():
    rel = path_for_record(Path("derived/experiments/x"))
    assert not rel.startswith("/") and rel.startswith("derived/")
    outside = path_for_record(Path("/tmp/some/other/project/out"))
    assert outside == "/tmp/some/other/project/out"
    through_symlink = path_for_record(
        PROJECT_ROOT / "resources" / "data" / "OMOMO_new" / "f.npz")
    assert not through_symlink.startswith("resources/")   # real target recorded
