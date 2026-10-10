"""A2 seventh-round review counterexample (independent review, 2026-10-10).

Fails on the pre-fix implementation (the last test is a positive
regression guard):

1  the MANIFEST leg re-hashed EVERY input an attempt recorded, not just
   the demos currently consumed from it — when a later attempt superseded
   the sixth demo and the superseded PT copy went missing from the older
   dir, the FileNotFoundError rejected the five demos STILL consumed from
   that attempt, although coverage, bindings, the artifact chain and the
   freeze leg all passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ  # noqa: E402
from ireset.paths import path_for_record  # noqa: E402

PILOT = [f"subA_largebox_{i:03d}" for i in range(1, 7)]


# --- helpers (kept self-contained per suite convention) -----------------------


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


def _write_demo(rng, npz_dir, att, demo):
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


def _build_supersede_root(tmp_path):
    """attempt_001 covers all six (dirA); attempt_002 supersedes the sixth
    (dirB). The sixth PT copy in dirA is then DELETED — no longer consumed
    evidence, its absence must not reject the five demos from dirA."""
    rng = np.random.default_rng(11)
    npz_dir = tmp_path / "pub"; npz_dir.mkdir(parents=True)
    dir_a = tmp_path / "pt_a"; dir_a.mkdir()
    dir_b = tmp_path / "pt_b"; dir_b.mkdir()
    u0 = tmp_path / "u0"
    att1 = u0 / "attempt_001"; att1.mkdir(parents=True)
    att2 = u0 / "attempt_002"; att2.mkdir()
    rows1, inputs1 = [], {}
    for demo in PILOT:
        rows1.append(_write_demo(rng, npz_dir, att1, demo))
        (dir_a / f"{demo}.pt").write_bytes(f"pt-{demo}".encode())
        inputs1[demo] = {
            "npz": integ.sha256_file(npz_dir / f"{demo}_original.npz"),
            "pt": integ.sha256_file(dir_a / f"{demo}.pt")}
    sixth = PILOT[5]
    rows2 = [_write_demo(rng, npz_dir, att2, sixth)]   # supersedes attempt_001
    (dir_b / f"{sixth}.pt").write_bytes(f"pt-{sixth}".encode())
    inputs2 = {sixth: {
        "npz": integ.sha256_file(npz_dir / f"{sixth}_original.npz"),
        "pt": integ.sha256_file(dir_b / f"{sixth}.pt")}}
    expected = integ.read_checkout_identity()["holosoma_revision"]
    ident = {"checkout": {"holosoma_revision": expected, "holosoma_dirty": False},
             "env": {"env_python": "envy", "env_uv_lock_sha256": "locksha",
                     "env_mujoco_version": "3.10.0"}}
    for att, rows, inputs, data_dir in ((att1, rows1, inputs1, dir_a),
                                        (att2, rows2, inputs2, dir_b)):
        (att / "generation_manifest.json").write_text(json.dumps({
            "rows": rows, "input_sha256": inputs, "identity": ident,
            "data_path": path_for_record(data_dir)}))
    (u0 / "attempts_index.json").write_text(json.dumps(
        {"attempts": [{"attempt": "attempt_001"}, {"attempt": "attempt_002"}]}))
    return u0, npz_dir, dir_a, dir_b


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
             "pt_sha256": f"pt-sha-{d}"}
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


def _hash_entries(npz_dir, tmp_path):
    """Frozen PT hashes keyed to whatever bytes the canonical dir holds —
    the freeze leg is monkeypatch-equivalent to 'all match' by matching the
    canonical dir; per-dir real PTs match by identical content."""
    return [
        {"demo": d,
         "npz_sha256": integ.sha256_file(npz_dir / f"{d}_original.npz"),
         "pt_sha256": integ.sha256_file(tmp_path / "pt_a" / f"{d}.pt")
         if (tmp_path / "pt_a" / f"{d}.pt").is_file()
         else integ.sha256_file(tmp_path / "pt_b" / f"{d}.pt")}
        for d in PILOT]


# --- review 1 (round 7): the manifest leg is scoped to CONSUMED demos --------


def test_superseded_pt_copy_absence_does_not_reject_consumed_demos(
        tmp_path, monkeypatch):
    u0, npz_dir, dir_a, dir_b = _build_supersede_root(tmp_path)
    # the sixth PT copy in dirA is no longer consumed evidence -> delete it
    (dir_a / f"{PILOT[5]}.pt").unlink()
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, dir_a,
                         entries=_hash_entries(npz_dir, tmp_path))
    # pre-fix: the manifest leg re-hashed ALL six recorded inputs of
    # attempt_001 at dirA -> FileNotFoundError rejected the five CONSUMED
    # demos too; post-fix only the consumed five are verified at dirA
    assert adm["passed"], (adm["inputs_rehashed"], adm["inputs_vs_freeze"])
    assert adm["inputs_rehashed"]["all_match"]
    by_demo = adm["inputs_rehashed"]["by_demo"]
    assert all(by_demo[d]["pt_matches"] is True for d in PILOT[:5])
    # the superseded sixth is consumed from dirB (attempt_002), verified there
    assert by_demo[PILOT[5]]["attempt"] == "attempt_002"
    assert by_demo[PILOT[5]]["pt_matches"] is True
    assert adm["data_paths"]["attempt_001"]["pt_dir"] == path_for_record(dir_a)
    assert adm["data_paths"]["attempt_002"]["pt_dir"] == path_for_record(dir_b)


def test_consumed_input_drift_is_still_caught_by_the_manifest_leg(
        tmp_path, monkeypatch):
    # scoping must not WEAKEN the check: a CONSUMED demo's input that
    # drifted from the generation record is still rejected
    u0, npz_dir, dir_a, dir_b = _build_supersede_root(tmp_path)
    (dir_a / f"{PILOT[5]}.pt").unlink()
    (dir_a / f"{PILOT[0]}.pt").write_bytes(b"drifted-consumed-input")
    adm, *_ = _admission(tmp_path, monkeypatch, u0, npz_dir, dir_a,
                         entries=_hash_entries(npz_dir, tmp_path))
    assert not adm["inputs_rehashed"]["all_match"]
    assert adm["inputs_rehashed"]["by_demo"][PILOT[0]]["pt_matches"] is False
    assert not adm["passed"]


def test_only_demos_scopes_the_recorded_hashes(monkeypatch):
    # unit contract of the integrity helper itself: only scoped demos are
    # hashed (monkeypatched hasher proves no out-of-scope file is opened)
    calls: list[str] = []

    def fake_sha(p):
        calls.append(Path(p).name)
        return "fake"

    monkeypatch.setattr(integ, "sha256_file", fake_sha)
    manifest = {"input_sha256": {d: {"npz": "x", "pt": "y"} for d in PILOT}}
    # empty scope -> nothing is ever opened
    assert integ.verify_inputs_against_manifest(
        manifest, Path("npz"), Path("pt"), only_demos=set()) == {}
    assert calls == []
    scoped = integ.verify_inputs_against_manifest(
        manifest, Path("npz"), Path("pt"), only_demos={PILOT[0]})
    assert set(scoped) == {PILOT[0]}
    assert set(calls) == {f"{PILOT[0]}_original.npz", f"{PILOT[0]}.pt"}
