"""Counterexamples for the SECOND review round of the train24fix2 forensic
tooling (reviewfix2, 2026-10-02).

Every test pins a confirmed defect of the first review-fix round:

- D1: the fail-binding identity check compared the re-run against its OWN
  snapshot (circular) -- ``bind`` now compares item-by-item against the
  ORIGINAL formal report of ``--rq-dir`` and STOPS on any mismatch.
- D2: the replays' "ALL MATCH" compared five summary values; tampering a
  row descriptor, a prefix worst value, an interpolation detail, the
  re-solve facts or the classification changed nothing.  The replay now
  compares the COMPLETE records and the arrays BITWISE, and the CLI verdict
  includes ``classification_matches``.
- D3: the counterfactual froze ONE side's ``[t-1]`` previous-full-scene
  distance into BOTH sides (the series' ``[t]`` already IS the previous
  scene) -- per-side ``[t]`` freezing plus a replay-side fresh-query
  cross-check against the hash-verified source NPZ.
- D4: the replay reloaded the CURRENT configs and ``load_products`` checked
  only a format string -- products now embed the frozen effective configs,
  the NPZ sha256 and a code/config/asset integrity manifest; the replay
  restores the frozen configs and refuses drifted inputs.
- D5: candidate discovery globbed the products directory, so a replay's own
  summary re-entered as a candidate (format None on the second run) and an
  empty directory judged ALL MATCH -- discovery is now index-driven with
  the replay output in a separate directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.config_io import load_mode_configs  # noqa: E402
from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from test_train24fix_review2 import _self_cond  # noqa: E402
from test_repair import standing_scene  # noqa: E402

REGEN = replace(RepairConfig(), regen_mode=True)
NO_C2 = replace(REGEN, regen_hand_speed_mps=0.0)


# ---------------------------------------------------------------------------
# shared scenario builders
# ---------------------------------------------------------------------------

def _failing_scene(n: int = 2) -> np.ndarray:
    """Box centered ON the root from t=1: every attempt of frame 1 fails
    (the all-attempts-fail shape of the first round's tests)."""
    scene = standing_scene(n)
    root = scene[0, 0:3].copy()
    scene[:, 36:43] = np.array([root[0], root[1], root[2] + 0.02, 1.0, 0, 0, 0])
    scene[0, 36:43] = np.array([5.0, 0.0, 0.25, 1.0, 0, 0, 0])
    return scene


def _report_row_of(this_run: dict) -> dict:
    """Formal-report row matching a run's captured identity (the genuine
    report the binding must accept)."""
    return {"accepted": False, "t_fail": this_run["t_fail"],
            "rejection": {"last_attempt_id": this_run["last_attempt_id"],
                          "last_check_id": this_run["last_check_id"],
                          "last_attempt": {"status":
                                           this_run["last_attempt_status"]}}}


def _doctored_row(row: dict) -> dict:
    """The reviewer's falsification shape: t_fail=999 plus rewritten
    attempt id / check id / attempt status."""
    bad = json.loads(json.dumps(row))
    bad["t_fail"] = 999
    bad["rejection"]["last_attempt_id"] = [9, 99, 9]
    bad["rejection"]["last_check_id"] = "p9:t99:c999"
    bad["rejection"]["last_attempt"]["status"] = "fabricated"
    return bad


def _bind_tiny(out: Path, report_row: dict) -> dict:
    """bind() on the tiny failing scenario with an INJECTED scene/cond and
    report row (no teacher-NPZ / formal-report dependency)."""
    import train24fix2_fail_bind as fb

    scene = _failing_scene()
    pipe = RepairPipeline(cfg=REGEN)
    cond = _self_cond(pipe, scene)
    return fb.bind("tiny_demo", Path("/nonexistent-rq"), out,
                   report_row=report_row, scene=scene, cond=cond)


def _genuine_row(out_probe: Path) -> dict:
    """Bind once against a row that cannot match (deterministic run) to
    LEARN the re-run's own identity, then build the genuine row from it."""
    rec = _bind_tiny(out_probe, {"accepted": False, "t_fail": -1,
                                 "rejection": {}})
    assert rec["verdict"] == "IDENTITY_MISMATCH"
    return _report_row_of(rec["identity_vs_report"]["this_run"])


# ---------------------------------------------------------------------------
# D1: identity against the ORIGINAL formal report
# ---------------------------------------------------------------------------

def test_identity_vs_report_flags_each_doctored_field() -> None:
    """Each of the four report items flags its own doctored variant -- the
    reviewer's falsification (t_fail=999 + changed attempt/check/status)
    must not pass item by item."""
    import train24fix2_fail_bind as fb

    captured = {"t": 44, "last_attempt_id": [0, 44, 2],
                "chk": {"check_id": "p0:t44:c62"},
                "res_fields": {"status": "infeasible"}}
    row = _report_row_of({"t_fail": 44, "last_attempt_id": [0, 44, 2],
                          "last_check_id": "p0:t44:c62",
                          "last_attempt_status": "infeasible"})
    assert all(fb.identity_vs_report(row, captured)["match"].values())

    for field, doctor in (("t_fail", lambda r: r.update(t_fail=999)),
                          ("last_attempt_id",
                           lambda r: r["rejection"].update(
                               last_attempt_id=[9, 99, 9])),
                          ("last_check_id",
                           lambda r: r["rejection"].update(
                               last_check_id="p9:t99:c999")),
                          ("last_attempt_status",
                           lambda r: r["rejection"]["last_attempt"].update(
                               status="fabricated"))):
        bad = _report_row_of({"t_fail": 44, "last_attempt_id": [0, 44, 2],
                              "last_check_id": "p0:t44:c62",
                              "last_attempt_status": "infeasible"})
        doctor(bad)
        match = fb.identity_vs_report(bad, captured)["match"]
        assert match[field] is False, field
        assert not all(match.values()), field


def test_bind_stops_on_doctored_report_and_binds_on_genuine(tmp_path) -> None:
    """A doctored report row STOPS the binding (no audits frozen, explicit
    failed items); the genuine row of the very same deterministic re-run
    proceeds to the full audits."""
    genuine = _genuine_row(tmp_path / "probe")

    rec_bad = _bind_tiny(tmp_path / "bad", _doctored_row(genuine))
    assert rec_bad["verdict"] == "IDENTITY_MISMATCH"
    assert "audits" not in rec_bad
    assert rec_bad["identity_failed_items"], "failed items are named"
    assert not all(rec_bad["identity_vs_report"]["match"].values())

    rec_good = _bind_tiny(tmp_path / "good", genuine)
    assert "audits" in rec_good and "c2_on" in rec_good["audits"]
    assert "classification" in rec_good
    assert all(rec_good["identity_vs_report"]["match"].values())


# ---------------------------------------------------------------------------
# D2 + D4 + D5: strict replay of the fail-binding products
# ---------------------------------------------------------------------------

def _make_bind_products(tmp_path: Path) -> Path:
    import train24fix2_fail_bind as fb

    out = tmp_path / "products"
    genuine = _genuine_row(tmp_path / "probe")
    rec = _bind_tiny(out, genuine)
    assert "audits" in rec
    fb.write_index(out, fb.INDEX_NAME,
                   {"demos": ["tiny_demo"], "stems": ["fail_bind2_tiny_demo"],
                    "rq_dir": "injected", "report_sha256": "injected"})
    return out


def _rewrite_record(out: Path, stem: str, mutate) -> None:
    jpath = out / f"{stem}.json"
    record = json.loads(jpath.read_text())
    mutate(record)
    jpath.write_text(json.dumps(record, indent=1, allow_nan=False,
                                default=float))


def test_replay_full_evidence_detects_tampering(tmp_path, monkeypatch) -> None:
    """The replay recomputes the branches, the re-solve escalation AND its
    candidate audit under the FROZEN configs and compares the COMPLETE
    records + arrays; every tampering class of the review (descriptor,
    prefix worst, interpolation detail, classification, re-solve facts,
    array values) flips the verdict, while the pristine products match.
    Current-YAML drift (task gate 0.20) changes nothing -- frozen only."""
    import train24fix2_fail_bind as fb

    out = _make_bind_products(tmp_path)
    stem = "fail_bind2_tiny_demo"
    replay_dir = tmp_path / "replay"
    pristine = (out / f"{stem}.json").read_text()

    # D4: current config drift must not reach the replay computation
    real = load_mode_configs

    def drifted(mode):
        rcfg, vcfg = real(mode)
        return replace(rcfg, task_mpjpe_tol_m=0.20), vcfg
    monkeypatch.setattr("ireset.repair.config_io.load_mode_configs", drifted)

    summary = fb.replay(out, replay_dir)
    assert summary["ok"] is True, summary["results"][stem]
    assert summary["results"][stem]["classification_matches"] is True
    # D5: the summary lives in the SEPARATE replay dir, never in products
    assert (replay_dir / "fail_bind2_replay.json").exists()
    assert not (out / "fail_bind2_replay.json").exists()

    def expect_mismatch(mutate, needle=None):
        (out / f"{stem}.json").write_text(pristine)       # untamper first
        _rewrite_record(out, stem, mutate)
        s = fb.replay(out, replay_dir / "run2")
        assert s["ok"] is False, "tampering must break ALL MATCH"
        entry = s["results"][stem]
        assert not (entry["matches"] and entry["classification_matches"]), \
            entry
        if needle:
            joined = " ".join(entry["record_diff_paths"])
            assert needle in joined, entry["record_diff_paths"]

    expect_mismatch(
        lambda r: r["audits"]["c2_on"]["endpoint"]["descriptors"].__setitem__(
            0, "fabricated_row"),
        needle="descriptors")
    expect_mismatch(
        lambda r: next(iter(r["audits"]["c2_on"]["prefix_validation"]
                            ["checks"].values())).__setitem__("worst", 123.0),
        needle="prefix_validation")
    pts = json.loads((out / f"{stem}.json").read_text())[
        "audits"]["c2_on"]["interpolation"]["points"]
    if pts:
        expect_mismatch(
            lambda r: r["audits"]["c2_on"]["interpolation"]["points"][0]
            ["vals_m"].__setitem__(0, pts[0]["vals_m"][0] + 0.5),
            needle="interpolation")
    expect_mismatch(
        lambda r: r["resolve_without_c2"].__setitem__(
            "last_term_reason", "fabricated"),
        needle="resolve_without_c2")
    expect_mismatch(lambda r: r.__setitem__("classification", "PASS"))

    # frozen-config tamper: the replay COMPUTES with the stored values, so
    # a rewritten task gate must diverge (proof the frozen configs are the
    # replay's only config source -- the tiny frame's C2 rows are off, so
    # the gate is the bound that actually enters this audit)
    (out / f"{stem}.json").write_text(pristine)

    def tamper_cfg(r):
        r["effective"]["repair"]["task_mpjpe_tol_m"] = 1e-12
    _rewrite_record(out, stem, tamper_cfg)
    s = fb.replay(out, replay_dir / "run3")
    assert s["ok"] is False

    # array tampering WITH a fixed-up hash: load succeeds, the bitwise
    # array comparison catches the changed row values
    def tamper_arrays():
        (out / f"{stem}.json").write_text(pristine)
        record, arrays = fb.load_products(out, stem)
        arrays["rows__c2_on__val"] = arrays["rows__c2_on__val"] + 0.25
        np.savez_compressed(out / record["npz"], **arrays)
        digest = hashlib.sha256(
            (out / record["npz"]).read_bytes()).hexdigest()

        def fix_hash(r):
            r["npz_sha256"] = digest
        _rewrite_record(out, stem, fix_hash)
    tamper_arrays()
    s = fb.replay(out, replay_dir / "run4")
    entry = s["results"][stem]
    assert s["ok"] is False
    assert any("rows__c2_on__val" in d for d in entry["array_diffs"]), \
        entry["array_diffs"]


def test_replay_ignores_decoys_and_repeats(tmp_path) -> None:
    """D5: a summary-shaped JSON (no product format) in the products dir is
    ignored -- discovery comes from the index, and a SECOND replay run is
    stable because its own output lives elsewhere."""
    import train24fix2_fail_bind as fb

    out = _make_bind_products(tmp_path)
    (out / "fail_bind2_replay.json").write_text(          # the old trap file
        json.dumps({"ok": True, "results": {}}))
    (out / "fail_bind2_decoy.json").write_text('{"not": "a product"}')
    s1 = fb.replay(out, tmp_path / "r1")
    assert s1["ok"] is True and s1["n"] == 1
    s2 = fb.replay(out, tmp_path / "r2")
    assert s2["ok"] is True and s2["n"] == 1


def test_replay_refuses_missing_or_empty_index(tmp_path) -> None:
    """D5: no index / empty index are explicit failures -- the glob-based
    replay judged an empty directory ALL MATCH."""
    import train24fix2_fail_bind as fb

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        fb.replay(empty, tmp_path / "r1")
    out = _make_bind_products(tmp_path)
    fb.write_index(out, "fail_bind2_index_other", {"stems": ["x"]})
    idx = json.loads((out / f"{fb.INDEX_NAME}.json").read_text())
    idx["stems"] = []
    (out / f"{fb.INDEX_NAME}.json").write_text(json.dumps(idx))
    with pytest.raises(ValueError):
        fb.replay(out, tmp_path / "r2")


# ---------------------------------------------------------------------------
# D4: product integrity
# ---------------------------------------------------------------------------

def test_load_products_refuses_tampered_npz(tmp_path) -> None:
    """A rewritten NPZ (even with the arrays' shapes intact) is refused --
    the old loader only checked the format string and loaded it happily."""
    from train24fix2_audit import load_products, write_products

    write_products(tmp_path, "p", {"k": 1}, {"a": np.arange(4)})
    npz_path = tmp_path / "p.npz"
    np.savez_compressed(npz_path, a=np.arange(4) + 1)      # silent tamper
    with pytest.raises(ValueError, match="sha256"):
        load_products(tmp_path, "p")


def test_verify_integrity_detects_drift(tmp_path) -> None:
    """A doctored/missing integrity manifest entry is reported explicitly;
    the intact manifest of a fresh product verifies clean."""
    from train24fix2_audit import (integrity_manifest, verify_integrity,
                                   write_products)

    write_products(tmp_path, "p", {}, {"a": np.arange(2)})
    record = {"integrity": {"files": integrity_manifest()}}
    assert verify_integrity(record) == []
    doctored = {"integrity": {"files": {
        k: ("0" * 64 if i == 0 else v)
        for i, (k, v) in enumerate(sorted(record["integrity"]["files"]
                                          .items()))}}}
    drift = verify_integrity(doctored)
    assert drift and "changed since generation" in drift[0]
    assert verify_integrity({"integrity": {"files": {}}}) == \
        ["record carries no integrity manifest"]


# ---------------------------------------------------------------------------
# D3: per-side previous-full-scene distance at frame t
# ---------------------------------------------------------------------------

def test_prev_scene_flags_per_side_at_t() -> None:
    """The flags read ``[side]["scene"][t]`` -- one side's value never
    lands in the other side, and ``[t-1]`` (the reviewed off-by-one) or
    the other side's entry are both detectably wrong."""
    from train24fix2_cert_trace import prev_scene_flags

    series = {"l": {"scene": [None, 0.30, 0.11, 0.12]},
              "r": {"scene": [None, 0.31, 0.22, 0.23]}}
    flags = prev_scene_flags(series, 2)
    assert flags == {"l": 0.11, "r": 0.22}
    assert flags["l"] != flags["r"]
    # the reviewed defects, made explicit
    assert flags["r"] != series["r"]["scene"][1]      # [t-1] shift
    assert flags["l"] != series["r"]["scene"][2]      # one side into both
    assert prev_scene_flags(None, 3) == {"l": None, "r": None}


def _build_cf_product(tmp_path: Path) -> tuple[Path, str, dict]:
    """Tiny counterfactual product + index (shared by the reviewfix2 D3 and
    the reviewfix3 G2 replay tests): a 3-frame pre-C2 chain with the box
    9.5 mm from the left hand, one cf record at t=1 with a teacher-layout
    source NPZ, written under ``tmp_path/cf``."""
    from test_band_floor_separation import _place_box
    from train24fix2_audit import write_products
    import train24fix2_cert_trace as ct

    rcfg, vcfg = load_mode_configs("strict")
    cfg0 = replace(REGEN, regen_hand_speed_mps=0.0)        # pre-C2 chain
    pipe = RepairPipeline(cfg=cfg0)
    scene = standing_scene(3)
    scene[:, 36:43] = _place_box(pipe, scene[0, 0:36].copy(), "l", 0.0095)
    cond = _self_cond(pipe, scene)
    res = pipe.repair_trajectory(scene, cond)
    # the chain replay only needs A trajectory (it replays bookkeeping, not
    # solving); prefer the run's own output when it accepted
    rq_traj = res.trajectory if res.accepted else scene.copy()

    prev = ct.prev_scene_distances(rq_traj)
    t = 1
    chain = ct.ChainReplay(cond, scene, rq_traj, rcfg, vcfg)
    c2_state = {s: {"enabled": [False, True], "reason": ["off", "enable"]}
                for s in ("l", "r")}
    # teacher-layout NPZ so the replay's fresh-query cross-check can load
    # the source trajectory back
    q = np.zeros((rq_traj.shape[0], 43))
    q[:, 4:7] = rq_traj[:, 0:3]
    q[:, 0:4] = rq_traj[:, 3:7]
    q[:, 7:36] = rq_traj[:, 7:36]
    q[:, 40:43] = rq_traj[:, 36:39]
    q[:, 36:40] = rq_traj[:, 39:43]
    src_npz = tmp_path / "tiny_rq.npz"
    np.savez(src_npz, qpos=q,
             human_joints=np.zeros((rq_traj.shape[0], 52, 3)), fps=30.0)
    prev_source = {"npz": str(src_npz), "sha256": ct.sha256(src_npz)}

    rec, arrays = ct.counterfactual(
        chain, t, rq_traj[t, 0:36].copy(), "current_rq_adopted_spike[l]",
        c2_state, "l", prev, prev_source, rq_traj, scene,
        pipe.spec, rcfg, vcfg)
    out = tmp_path / "cf"
    stem = "cert_trace_cf_tiny_t1_l_spike"
    write_products(out, stem, rec, arrays)
    ct.write_index(out, ct.INDEX_NAME, {"records": [
        {"demo": "tiny", "side": "l", "kind": "spike", "frame": t,
         "stem": stem}], "stems": [stem]})
    return out, stem, {"prev": prev, "t": t, "rec": rec}


def test_counterfactual_per_side_freeze_and_replay_crosscheck(tmp_path) -> None:
    """End-to-end D3: a real counterfactual on a tiny accepted chain
    freezes BOTH sides' ``[t]`` distances; the strict replay reproduces
    everything from the frozen product (ALL MATCH), and tampering the
    per-side distance field is caught by the fresh-query cross-check
    against the hash-verified source NPZ."""
    import train24fix2_cert_trace as ct

    out, stem, info = _build_cf_product(tmp_path)
    prev, t, rec = info["prev"], info["t"], info["rec"]
    both = rec["prev_scene_dist_both_m"]
    assert prev["l"]["scene"][t] is not None
    # the per-frame helper agrees with the full-series computation
    assert ct.prev_scene_distances_at(_load_src_traj(rec), t) == \
        {s: prev[s]["scene"][t] for s in ("l", "r")}
    assert both["l"] == prev["l"]["scene"][t]
    assert both["r"] == prev["r"]["scene"][t]
    assert rec["prev_scene_dist_flags"] == both          # ctx flags too
    assert rec["prev_scene_dist_m"] == both["l"]         # selected side

    replay_dir = tmp_path / "cf_replay"
    ct.replay_counterfactuals(out, replay_dir)
    s1 = json.loads((replay_dir / "cert_trace_cf_replay.json").read_text())
    assert s1["ok"] is True, s1["results"][stem]

    # D3 tamper: shift ONE side's frozen distance -- the fresh query of the
    # source NPZ must flag exactly that side
    def tamper(r):
        r["prev_scene_dist_both_m"]["l"] = \
            r["prev_scene_dist_both_m"]["l"] + 1e-6
    jpath = out / f"{stem}.json"
    record = json.loads(jpath.read_text())
    tamper(record)
    jpath.write_text(json.dumps(record, indent=1, allow_nan=False,
                                default=float))
    ct.replay_counterfactuals(out, replay_dir / "run2")
    s2 = json.loads((replay_dir / "run2" / "cert_trace_cf_replay.json")
                    .read_text())
    entry = s2["results"][stem]
    assert s2["ok"] is False
    assert any(k.endswith(":l") for k in entry["prev_scene_check"]), \
        entry["prev_scene_check"]


def _load_src_traj(rec: dict) -> np.ndarray:
    from ireset.data.teacher import load_teacher_npz
    traj, _h, _f = load_teacher_npz(Path(rec["prev_scene_source"]["npz"]))
    return traj
