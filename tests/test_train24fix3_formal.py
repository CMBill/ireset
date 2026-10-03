"""train24fix3 continuation unit 2: the FULL §8.1 quality gate, the formal
sixseg/train24 driver stages and the §12 subtask state machine
(plan sections 8.2/9/11.1 unit 2 / 12).

Pinned here:

- every §8.1 row carries an explicit verdict: coverage against the FROZEN
  demo list (dropped/duplicated report rows fail), NPZ/report consistency
  (kept set == accepted set == metrics set), independent strict of every
  kept segment, per-side core/complete/task cells, the demo-level
  both-hands overlap + usable windows (the dataset's own 60/30 rule),
  sub16's C protections, the mechanism family (spikes/enable from the cf
  summary + GATED-zero reclassification of the four frozen failing sides
  vs H), the integrity family (R9/profile/row coherence/argv) and the
  label family (graded_labels_v1 manifest + label NPZ contract);
- ``full`` mode makes evidence-requiring families MANDATORY: a missing cf
  summary / label manifest / profile is an explicit FAILURE, never a pass;
- the formal stages refuse to start without the SELECTION (winner=None,
  no summary, --profile != winner); the standard-mode control carries no
  C2 evidence and matches B's own ns reference counts;
- the §12 state machine: resume reuses ONLY complete+same-identity+still-
  hash-verified attempts, half-written attempts stay as failed evidence,
  a rerun goes to a NEW attempt directory, an identity change forces a
  rerun, and a stage whose subtasks disagree never writes a PASS summary
  from partial files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import train24fix3_convergence as drv  # noqa: E402
import train24fix3_quality_gate as qg  # noqa: E402

DIAG6 = ["sub10_largebox_000", "sub12_largebox_001", "sub16_largebox_000",
         "sub3_largebox_000", "sub7_largebox_000", "sub8_largebox_000"]
#: the fabricated runs keep sub16 + the four frozen failing sides' demos
KEPT_DEFAULT = ("sub16_largebox_000", "sub10_largebox_000",
                "sub3_largebox_000", "sub8_largebox_000")


# ---------------------------------------------------------------------------
# §8.1 gate: verdict logic on fabricated metric sources (fast contracts)
# ---------------------------------------------------------------------------


def _fake_seg(demo: str, *, T: int = 120, cert_ref: int = 20,
              longest: int = 8, within: int = 40, task: float = 0.1,
              strict: bool = True, gated: int = 0) -> dict:
    sides = {}
    for side in ("l", "r"):
        sup = np.zeros(T, dtype=bool)
        sup[10:10 + cert_ref] = True
        speed = np.full(T, 0.01)
        if gated:
            # frames the NEW run LOSES vs the reference at speed >= the
            # entry gate -- the reclassification's GATED class
            sup[10:10 + gated] = False
            speed[10:10 + gated] = 1.0
        sides[side] = {
            "ref_frames": 74, "ref_intervals": [[8, 80]],
            "certified_frames": cert_ref + 4,
            "certified_in_ref_frames": int(sup.sum()),
            "intersection_intervals": [[10, 10 + longest]],
            "intersection_longest_run_frames": longest,
            "certified_intervals": [[9, 9 + cert_ref + 4]],
            "frames_within_enter": within,
            "longest_detachment_run_frames": 2,
            "max_detachment_ref_m": 0.02, "gap_ref_min_m": 0.001,
            "speed_ref_max_mps": 0.05, "speed_ref_p95_mps": 0.02,
            "anchor_drift_certified_max_m": 0.005,
            "series": {"hand_dist": [0.005] * T,
                       "h_speed": [float(s) for s in speed],
                       "max_component": [0.0] * T,
                       "supervision": sup.tolist(),
                       "cert_intervals": [[9, 9 + cert_ref + 4]]},
        }
    return {"demo": demo, "npz": f"{demo}.npz", "npz_sha256": "0" * 16,
            "input_sha256": "0" * 16, "T": T, "validator_passed": strict,
            "task_mpjpe": {"p50": task, "p95": task, "max": task},
            "sides": sides}


def _fake_report(run_dir: Path, demos=DIAG6, accepted=KEPT_DEFAULT,
                 dup=False, drop=None) -> Path:
    rows = []
    for d in demos:
        if drop and d == drop:
            continue
        rows.append({"demo": d, "accepted": d in accepted,
                     "status": "accepted" if d in accepted else "rejected_frame",
                     "T": 120, "fps": 30.0, "skip_reason": None,
                     "slack_frames": 0, "t_fail": None})
    if dup:
        rows.append(dict(rows[0]))
    rep = run_dir / "regen_report_diagnosis6_strict.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({
        "mode": "regen", "n_demos": len(rows) - (1 if dup else 0),
        "n_kept": len(accepted), "rows": rows,
        "effective": {"argv": ["x"], "mode": "regen"},
        "end": {"code_changed_during_run": False},
        "profile": {"name": "S1"}}))
    return rep


@pytest.fixture()
def fake_metrics(monkeypatch, tmp_path):
    """Fabricated per-directory metric sources + a label manifest; the
    VERDICT LOGIC runs real (only freeze_segment's recomputation is
    replaced -- its arithmetic is pinned by the real-data smoke below)."""
    new = {d: _fake_seg(d) for d in KEPT_DEFAULT}
    refs = {
        "H": {d: _fake_seg(d, cert_ref=15, longest=6, within=45)
              for d in DIAG6 if d != "sub7_largebox_029"},
        "B": {d: _fake_seg(d, cert_ref=14, longest=7, within=44)
              for d in DIAG6},
        "C": {"sub16_largebox_000": _fake_seg("sub16_largebox_000",
                                              cert_ref=15, longest=15)},
        "T": {d: _fake_seg(d, cert_ref=10, longest=5, within=40)
              for d in DIAG6[:3]},
    }
    state = {"new": new, "refs": refs}

    def fake_metrics_of_run(run_dir, *, with_series=False):
        key = Path(run_dir).name
        if key in refs:
            return refs[key]
        for name, d in (("rq", "new"),):
            if key.startswith(name):
                return state["new"]
        return refs["H"] if "labelq" in str(run_dir) else state["new"]

    monkeypatch.setattr(qg, "metrics_of_run", fake_metrics_of_run)
    monkeypatch.setattr(qg, "REFERENCES",
                        {"H": Path("/H"), "B": Path("/B"), "C": Path("/C"),
                         "T": Path("/T")})
    return state


def _fake_label_manifest(tmp_path, demos=KEPT_DEFAULT) -> Path:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DIAG6:
        row = {"demo": d, "accepted": d in demos, "status":
               "accepted" if d in demos else "rejected_frame",
               "T": 120, "group": "train",
               "label_eligibility": {"motion_fk": d in demos},
               "strict_recheck_passed": d in demos}
        if d in demos:
            path = labels_dir / f"{d}.npz"
            prox = np.zeros((120, 2), dtype=bool)
            sup = np.zeros((120, 2), dtype=bool)
            ids = np.full((120, 2), -1, dtype=np.int64)
            sup[10:30, 0] = True
            ids[10:30, 0] = 0
            np.savez(path, hand_proximity_mask=prox,
                     hand_contact_supervision_mask=sup,
                     hand_supervision_interval_id=ids,
                     meta_json=json.dumps(
                         {"label_protocol": "graded_labels_v1", "demo": d,
                          "T": 120}))
            import hashlib
            row["label_npz"] = {
                "path": str(path), "sha256":
                    hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
        rows.append(row)
    manifest = labels_dir / "label_manifest.json"
    manifest.write_text(json.dumps(
        {"label_protocol": "graded_labels_v1", "rows": rows}))
    return manifest


def test_gate_full_mode_every_family_has_verdict(tmp_path, fake_metrics):
    run_dir = tmp_path / "rq"
    _fake_report(run_dir)
    labels = _fake_label_manifest(tmp_path)
    rep = qg.gate_report(run_dir, expected_demos=DIAG6, profile="S1",
                         cf_summary={"spikes_ok": True, "enable_ok": True},
                         label_manifest=labels, full=True)
    v = rep["verdicts"]
    for key in ("coverage_demos_ok", "npz_consistency_ok",
                "strict_all_kept_ok", "H_all_sides_ok", "B_all_sides_ok",
                "C_all_sides_ok", "T_all_sides_ok",
                "C_sub16_protections_ok", "mechanism_spikes_ok",
                "mechanism_enable_ok", "mechanism_gated_zero_ok",
                "integrity_r9_clean", "integrity_profile_matches",
                "integrity_rows_coherent", "labels_manifest_ok"):
        assert key in v, f"verdict {key} missing"
        assert v[key] is not None, f"verdict {key} is None (missing gate)"
    # both-hands + usable-window cells exist as demo-level rows
    both_rows = [r for r in rep["rows"] if r.get("side") == "both"]
    assert both_rows and "usable_windows" in both_rows[0]["cells"]
    assert "both_supervision_frames" in both_rows[0]["cells"]


def test_gate_full_mode_missing_evidence_fails(tmp_path, fake_metrics):
    run_dir = tmp_path / "rq"
    _fake_report(run_dir)
    rep = qg.gate_report(run_dir, full=True)      # NO cf / labels / profile
    v = rep["verdicts"]
    assert v["mechanism_spikes_ok"] is False      # 缺 gate 不是通过
    assert v["mechanism_enable_ok"] is False
    assert v["labels_manifest_ok"] is False


def test_gate_rejects_dropped_and_duplicated_rows(tmp_path, fake_metrics):
    base = tmp_path / "rq"
    _fake_report(base)
    ok_rep = qg.gate_report(base, expected_demos=DIAG6)
    assert ok_rep["verdicts"]["coverage_demos_ok"] is True

    dropped = tmp_path / "rq_dropped"
    _fake_report(dropped, drop="sub7_largebox_000")
    v = qg.gate_report(dropped, expected_demos=DIAG6)["verdicts"]
    assert v["coverage_demos_ok"] is False        # 报告删失败 demo 不能缩分母

    dup = tmp_path / "rq_dup"
    _fake_report(dup, dup=True)
    assert qg.gate_report(dup, expected_demos=DIAG6
                          )["verdicts"]["coverage_demos_ok"] is False


def test_gate_kept_npz_and_strict_consistency(tmp_path, fake_metrics):
    run_dir = tmp_path / "rq"
    # the accepted set disagrees with the metrics (kept NPZ) set: sub12 is
    # accepted but has no kept NPZ
    _fake_report(run_dir, accepted=KEPT_DEFAULT + ("sub12_largebox_001",))
    v = qg.gate_report(run_dir)["verdicts"]
    assert v["npz_consistency_ok"] is False

    # a kept segment that fails the independent strict revalidation
    run2 = tmp_path / "rq2"
    _fake_report(run2)
    fake_metrics["new"]["sub16_largebox_000"]["validator_passed"] = False
    v2 = qg.gate_report(run2)["verdicts"]
    assert v2["strict_all_kept_ok"] is False
    fake_metrics["new"]["sub16_largebox_000"]["validator_passed"] = True


def test_gate_mechanism_gated_zero_and_integrity(tmp_path, fake_metrics):
    run_dir = tmp_path / "rq"
    _fake_report(run_dir)
    rep = qg.gate_report(run_dir, cf_summary={"spikes_ok": False,
                                              "enable_ok": True})
    v = rep["verdicts"]
    assert v["mechanism_spikes_ok"] is False
    assert v["mechanism_gated_zero_ok"] is True
    # a speed-gated supervision break vs H flips the mechanism verdict
    fake_metrics["new"]["sub8_largebox_000"] = _fake_seg(
        "sub8_largebox_000", gated=2)
    v2 = qg.gate_report(run_dir)["verdicts"]
    assert v2["mechanism_gated_zero_ok"] is False
    assert v2["mechanism_gated_frames_total"] == 4  # both sides x 2 frames
    # integrity: dirty worktree / wrong profile / incoherent rows
    rep_path = run_dir / "regen_report_diagnosis6_strict.json"
    doc = json.loads(rep_path.read_text())
    doc["end"]["code_changed_during_run"] = True
    doc["profile"]["name"] = "S2"
    # a REJECTED row (sub12) claiming acceptance
    row12 = next(r for r in doc["rows"] if r["demo"] == "sub12_largebox_001")
    row12["status"] = "accepted"
    rep_path.write_text(json.dumps(doc))
    v3 = qg.gate_report(run_dir, profile="S1")["verdicts"]
    assert v3["integrity_r9_clean"] is False
    assert v3["integrity_profile_matches"] is False
    assert v3["integrity_rows_coherent"] is False


def test_gate_labels_contract_rejects_tampering(tmp_path, fake_metrics):
    run_dir = tmp_path / "rq"
    _fake_report(run_dir)
    labels = _fake_label_manifest(tmp_path)
    assert qg.gate_report(run_dir, label_manifest=labels,
                          full=True)["verdicts"]["labels_manifest_ok"] is True
    # protocol bump
    doc = json.loads(labels.read_text())
    doc["label_protocol"] = "graded_labels_v2"
    labels.write_text(json.dumps(doc))
    assert qg.gate_report(run_dir, label_manifest=labels,
                          full=True)["verdicts"]["labels_manifest_ok"] is False
    # supervision mask disagrees with the interval ids
    labels = _fake_label_manifest(tmp_path / "t2")
    with np.load(labels.parent / "sub16_largebox_000.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["hand_contact_supervision_mask"] = np.zeros((120, 2), dtype=bool)
    np.savez(labels.parent / "sub16_largebox_000.npz", **arrays)
    doc = json.loads(labels.read_text())
    import hashlib
    p = labels.parent / "sub16_largebox_000.npz"
    row16 = next(r for r in doc["rows"]
                 if r["demo"] == "sub16_largebox_000")
    row16["label_npz"]["sha256"] = \
        hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    labels.write_text(json.dumps(doc))
    assert qg.gate_report(run_dir, label_manifest=labels,
                          full=True)["verdicts"]["labels_manifest_ok"] is False


def test_gate_real_data_smoke_c_run():
    """One REAL end-to-end computation: the C production run (1 kept, the
    current 1/6 state) under the full verdict logic -- the per-side core
    cells reproduce the frozen D-Q failing-side semantics and the new
    families carry real values (H comparisons on the kept sub16 only)."""
    c_run = drv.TEACHERS / "train24fix2_20260927_sixseg" / "rq"
    rep = qg.gate_report(c_run)
    v = rep["verdicts"]
    assert v["npz_consistency_ok"] is True
    assert v["n_kept"] == 1
    # every H/B demo but sub16 is missing from the new run -> kept FAILs
    assert v["H_all_sides_ok"] is False
    assert v["mechanism_gated_zero_ok"] is not None
    sub16 = [r for r in rep["rows"]
             if r["demo"] == "sub16_largebox_000"
             and r["reference"] == "C" and r["side"] == "r"]
    assert sub16 and sub16[0]["ok"] is True   # C's own kept stays >= itself


def test_gate_real_data_smoke_b_run_four_failing_sides():
    """The unit-2 acceptance smoke: the B (train24fix D-Q) run vs H
    reproduces the frozen four failing sides EXACTLY -- sub10:r 13<15,
    sub3:r 13<18, sub8:l 27<29, sub8:r 37<41 -- under the new full-gate
    code (the verdict logic may grow, the frozen regressions must not
    move)."""
    b_run = drv.TEACHERS / "train24fix_20260927_sixseg" / "rq"
    rep = qg.gate_report(b_run)
    expect = {("sub10_largebox_000", "r"): (13, 15),
              ("sub3_largebox_000", "r"): (13, 18),
              ("sub8_largebox_000", "l"): (27, 29),
              ("sub8_largebox_000", "r"): (37, 41)}
    seen = {}
    for row in rep["rows"]:
        if row["reference"] != "H" or row.get("side") not in ("l", "r"):
            continue
        key = (row["demo"], row["side"])
        if key in expect:
            cell = row["cells"]["certified_in_ref_frames"]
            seen[key] = (cell["new"], cell["ref"], cell["ok"])
    for key, (new, ref) in expect.items():
        assert key in seen, f"row {key} missing from the H comparison"
        got = seen[key]
        assert got[0] == new and got[1] == ref, (key, got, (new, ref))
        assert got[2] is False, f"{key}: the frozen regression must FAIL"


# ---------------------------------------------------------------------------
# §12 subtask state machine
# ---------------------------------------------------------------------------


def test_subtask_registry_resume_and_identity(tmp_path):
    reg = drv.SubtaskRegistry(tmp_path)
    ident = {"profile": "S1", "git_rev": "x"}
    att = reg.begin("rq1", tmp_path, ident, 6)
    out = att / "report.json"
    out.write_text("{}")
    rec = reg.complete("rq1", [out], 6)
    assert rec["ok"] is True
    # complete + same identity + outputs intact -> reusable
    again = drv.SubtaskRegistry(tmp_path)
    v = again.verified("rq1", ident)
    assert v is not None and v["processed"] == 6
    # a changed identity is NOT reusable (same candidate requires the
    # frozen code/profile/worktree)
    assert again.verified("rq1", {**ident, "git_rev": "y"}) is None
    # a tampered output is NOT reusable
    out.write_text("{} ")
    assert drv.SubtaskRegistry(tmp_path).verified("rq1", ident) is None
    out.write_text("{}")
    assert drv.SubtaskRegistry(tmp_path).verified("rq1", ident) is not None
    # a FAILED attempt stays as evidence and never verifies
    att2 = reg.begin("rq1", tmp_path, ident, 6)
    reg.fail("rq1", "boom")
    assert json.loads((tmp_path / "subtasks.json").read_text())
    assert att2.name == "attempt_002"
    assert drv.SubtaskRegistry(tmp_path).verified("rq1", ident)["attempt"] == 1


# ---------------------------------------------------------------------------
# formal stages: refusal paths + the full flow with fabricated runners
# ---------------------------------------------------------------------------


def _fabricated_selection(root: Path, winner: str = "S1",
                          selected: bool = True) -> None:
    (root / "norm").mkdir(parents=True, exist_ok=True)
    (root / "norm" / "norm_summary.json").write_text(json.dumps(
        {"stage": "norm", "ok": True, "winner": winner}))
    (root / "norm" / winner).mkdir(parents=True, exist_ok=True)
    (root / "norm" / winner / "profile_summary.json").write_text(json.dumps(
        {"selected": selected, "spikes_ok": True, "enable_ok": True}))
    # the replay subtask needs ff/cf products -- a marker index suffices
    # because replay_recovery itself is monkeypatched in the flow tests
    (root / "norm" / winner / "ff").mkdir(exist_ok=True)
    (root / "norm" / winner / "cf").mkdir(exist_ok=True)


def test_selection_refusal_paths(tmp_path):
    with pytest.raises(RuntimeError, match="no algorithm-stage summary"):
        drv._selection_winner(tmp_path)
    (tmp_path / "norm").mkdir()
    (tmp_path / "norm" / "norm_summary.json").write_text(
        json.dumps({"stage": "norm", "ok": True, "winner": None}))
    with pytest.raises(RuntimeError, match="winner=None"):
        drv._selection_winner(tmp_path)
    _fabricated_selection(tmp_path)
    assert drv._selection_winner(tmp_path)[0] == "S1"
    with pytest.raises(SystemExit, match="错误选定"):
        drv.stage_sixseg(tmp_path, profile="S2")
    # the winner's own record must say selected
    _fabricated_selection(tmp_path / "t2", selected=False)
    with pytest.raises(SystemExit, match="selected"):
        drv.stage_sixseg(tmp_path / "t2", profile="S1")
    # train24 needs the formal sixseg PASS first
    with pytest.raises(RuntimeError, match="sixseg"):
        drv.stage_train24(tmp_path, profile="S1")


def _fake_run_writer(rq: Path, profile_name, mode: str, set_name: str,
                     kept) -> dict:
    """Write a fabricated run (report + kept NPZs).  Standard mode mirrors
    B's own ns reference rows so the ns-control comparison is meaningful."""
    rq.mkdir(parents=True, exist_ok=True)
    prefix = "regen_report" if mode == "regen" else "standard_report"
    rep = rq / f"{prefix}_{set_name}_strict.json"
    if mode == "standard":
        b_ns = drv.TEACHERS / "train24fix_20260927_sixseg" / "ns" / \
            f"standard_report_{set_name}_strict.json"
        doc = json.loads(b_ns.read_text())
    else:
        demos = drv.registered_set_of(set_name)
        doc = {
            "mode": mode, "n_demos": len(demos), "n_kept": len(kept),
            "rows": [{"demo": d, "accepted": d in kept,
                      "status": "accepted" if d in kept else "rejected_frame",
                      "T": 120, "skip_reason": None, "slack_frames": 0,
                      "t_fail": None} for d in demos],
            "effective": {"argv": ["x"], "mode": mode},
            "end": {"code_changed_during_run": False},
            "profile": {"name": profile_name}}
    rep.write_text(json.dumps(doc))
    for r in doc["rows"]:
        if r["accepted"]:
            (rq / f"{r['demo']}.npz").write_bytes(r["demo"].encode())
    return {"rq_dir": str(rq), "report_path": str(rep),
            "n_demos": doc["n_demos"], "n_kept": doc["n_kept"]}


def _patch_flow(monkeypatch, *, kept_ratio=1.0, gate_ok=True,
                fail_on=None, ns_c2=False):
    """Fabricated runners for the formal stages: runs write report+NPZs,
    labels/gate/replay write their products; ``fail_on`` names a call
    ordinal whose runner raises (the crash/resume path)."""
    calls = {"order": []}

    def fake_run(profile_name, rq, *, verbose=False, mode="regen",
                 set_name="diagnosis6", note=None):
        calls["order"].append(("run", profile_name, mode, str(rq)))
        if fail_on and len(calls["order"]) == fail_on:
            raise RuntimeError("fabricated crash")
        demos = drv.registered_set_of(set_name)
        kept = [d for d in demos if demos.index(d) < max(
            1, int(len(demos) * kept_ratio))]
        return _fake_run_writer(rq, profile_name, mode, set_name, kept)

    def fake_labels(run_dir, out_dir):
        calls["order"].append(("labels", str(out_dir)))
        if fail_on and len(calls["order"]) == fail_on:
            raise RuntimeError("fabricated crash")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "label_manifest.json").write_text("{}")
        return out_dir / "label_manifest.json"

    def fake_replay(prof_dir, out_dir, **kw):
        calls["order"].append(("replay", str(out_dir)))
        if fail_on and len(calls["order"]) == fail_on:
            raise RuntimeError("fabricated crash")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "recovery_replay.json").write_text(
            json.dumps({"ok": True}))
        return {"ok": True}

    def fake_write_gate(att_dir, run_dir, profile, set_name, *,
                        cf_summary=None, labels=None):
        att_dir.mkdir(parents=True, exist_ok=True)
        if fail_on and "gate" not in calls and len(calls["order"]) >= 4:
            calls["gate"] = True
            raise RuntimeError("fabricated crash")
        (att_dir / "quality_gate.json").write_text(
            json.dumps({"ok": gate_ok, "verdicts": {"fabricated": gate_ok}}))
        return att_dir / "quality_gate.json"

    monkeypatch.setattr(drv, "_run_sixseg", fake_run)
    monkeypatch.setattr(drv, "_run_labels", fake_labels)
    monkeypatch.setattr(drv, "replay_recovery", fake_replay)
    monkeypatch.setattr(drv, "_write_gate", fake_write_gate)
    return calls


def _fake_metrics_light(monkeypatch):
    """The train24 stage recomputes run metrics for usable windows -- a
    fabricated cheap source (the real arithmetic is covered above)."""

    def fake(run_dir, *, with_series=False):
        return {p.stem: {"T": 120, "validator_passed": True,
                         "task_mpjpe": {"p95": 0.1, "max": 0.1},
                         "sides": {s: {"certified_in_ref_frames": 1,
                                       "intersection_longest_run_frames": 1,
                                       "frames_within_enter": 1,
                                       "longest_detachment_run_frames": 0,
                                       "max_detachment_ref_m": 0.0,
                                       "anchor_drift_certified_max_m": 0.0,
                                       "certified_intervals": [[1, 2]],
                                       "intersection_intervals": [[1, 2]]}
                                  for s in ("l", "r")}}
                for p in Path(run_dir).glob("*.npz")}

    monkeypatch.setattr("train24fix3_quality_gate.metrics_of_run", fake)


def test_stage_sixseg_full_flow_and_resume(tmp_path, monkeypatch):
    _fabricated_selection(tmp_path)
    calls = _patch_flow(monkeypatch)
    _fake_metrics_light(monkeypatch)
    drv.stage_sixseg(tmp_path, profile="S1")
    summary = json.loads((tmp_path / "sixseg" / "sixseg_summary.json"
                          ).read_text())
    assert summary["ok"] is True and summary["selected_profile"] == "S1"
    sub = json.loads((tmp_path / "sixseg" / "subtasks.json").read_text())
    assert set(sub["subtasks"]) == {"rq1", "rq2", "rv", "ns", "labels",
                                    "quality_gate", "replay"}
    for name, atts in sub["subtasks"].items():
        assert atts[-1]["status"] == "complete" and atts[-1]["ok"] is True
    ns_att = Path(sub["subtasks"]["ns"][-1]["dir"])
    ns_rep = json.loads(
        (ns_att / "standard_report_diagnosis6_strict.json").read_text())
    assert ns_rep["effective"]["mode"] == "standard"

    # a mismatched run set never produces a PASS summary: rv fabricated
    # with a different kept set -> triple consistency fails
    def runner2(profile_name, rq, *, verbose=False, mode="regen",
                set_name="diagnosis6", note=None):
        calls["order"].append(("run", profile_name, mode, str(rq)))
        demos = drv.registered_set_of(set_name)
        kept = demos if not verbose else demos[:-1]   # rv keeps 5, not 6
        return _fake_run_writer(rq, profile_name, mode, set_name, kept)

    _patch_flow(monkeypatch, gate_ok=True)          # restore label/replay
    monkeypatch.setattr(drv, "_run_sixseg", runner2)
    root3 = tmp_path / "r3"
    _fabricated_selection(root3)
    drv.stage_sixseg(root3, profile="S1")
    summary3 = json.loads((root3 / "sixseg" / "sixseg_summary.json"
                           ).read_text())
    assert summary3["ok"] is False
    assert any(c["name"] == "triple_run_consistency" and not c["ok"]
               for c in summary3["checks"])


def test_stage_sixseg_crash_resume_reuses_verified(tmp_path, monkeypatch):
    _fabricated_selection(tmp_path)
    _patch_flow(monkeypatch, fail_on=3)             # crash on rv
    with pytest.raises(RuntimeError, match="fabricated crash"):
        drv.stage_sixseg(tmp_path, profile="S1")
    assert not (tmp_path / "sixseg" / "sixseg_summary.json").exists()
    sub = json.loads((tmp_path / "sixseg" / "subtasks.json").read_text())
    statuses = {k: v[-1]["status"] for k, v in sub["subtasks"].items()}
    assert statuses["rq1"] == "complete" and statuses["rv"] == "failed"

    _patch_flow(monkeypatch)                        # no failure now
    drv.stage_sixseg(tmp_path, profile="S1")
    sub2 = json.loads((tmp_path / "sixseg" / "subtasks.json").read_text())
    # rq1/rq2 were REUSED (still attempt_001, no new attempts); rv reran
    assert sub2["subtasks"]["rq1"][-1]["dir"] == \
        sub["subtasks"]["rq1"][-1]["dir"]
    assert len(sub2["subtasks"]["rq1"]) == 1
    assert len(sub2["subtasks"]["rv"]) == 2
    summary = json.loads((tmp_path / "sixseg" / "sixseg_summary.json"
                          ).read_text())
    assert summary["runs"]["rq1"]["reused"] is True
    assert summary["runs"]["rv"]["reused"] is False


def test_stage_train24_flow(tmp_path, monkeypatch):
    _fabricated_selection(tmp_path)
    (tmp_path / "sixseg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sixseg" / "sixseg_summary.json").write_text(json.dumps(
        {"stage": "sixseg", "ok": True, "selected_profile": "S1"}))
    _patch_flow(monkeypatch)
    _fake_metrics_light(monkeypatch)
    drv.stage_train24(tmp_path, profile="S1")
    summary = json.loads((tmp_path / "train24" / "train24_summary.json"
                          ).read_text())
    # all 24 kept by the fabricated runner -> quantity gate passes
    assert summary["ok"] is True
    assert summary["n_kept"] >= 18
    assert summary["quality_gate"]["ok"] is True
    # a mismatched frozen profile is an explicit failure
    fresh = tmp_path / "fresh"
    _fabricated_selection(fresh)
    (fresh / "sixseg").mkdir(exist_ok=True)
    (fresh / "sixseg" / "sixseg_summary.json").write_text(json.dumps(
        {"stage": "sixseg", "ok": True, "selected_profile": "S1"}))
    with pytest.raises(SystemExit, match="frozen profile"):
        drv.stage_train24(fresh, profile="S2")
