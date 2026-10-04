"""train24fix3 continuation unit 3: the evidence re-verification stage
(plan 4.4 item 6 / 13) and the actual-failure-prefix / anchor-drift stage
(plan 5.3 / 13).

Pinned here:

- the HISTORICAL source of record is real and frozen: the A-D root's norm
  summaries verify (winner=None, 0/6, R9 clean, registry/resolved equal to
  the live profiles), the old cf directories carry 113 per-item
  classifications with exactly the 41/72 (S1) and 57/56 (S2) distributions;
- ``stage_evidence`` runs every §12 subtask (ff 5 / cf 113 / replay 118 x2
  per profile), re-emits the mechanism verdicts from the stage summaries
  (resume reuses them without re-solving), keeps the replay outputs out of
  the products directories, and compares item-by-item against the frozen
  targets -- a classification difference is LOCATED (per-stem diff list,
  shifted distribution) and does NOT silently pass as "identical", but it
  is not forced back either;
- budget control: subtasks not started before the deadline are recorded
  ``not_run_budget`` and the stage ends INCOMPLETE (never a partial pass);
  a failing subtask is recorded and the stage fails;
- ``stage_failure_prefix`` refuses to start unless evidence fully passed,
  assembles exactly 10 solve bindings (S1 x6 + S2 x4) + 2 anchor bindings
  from the REAL reports, reuses verified attempts, and a binding whose
  re-run diverges from the original report is a recorded FAILURE (never
  written as evidence);
- the compact whole-frame QP serialization keeps row/col identities for
  EVERY QP and full numeric matrices only for the final QP;
- ``freeze_segment(with_series=True)`` adds the per-interval cumulative
  anchor-drift curves, consistent with the frozen per-interval maximum.
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

HIST = drv.RECOVERY_SRC_ROOT


# ---------------------------------------------------------------------------
# the historical source of record (real data, read-only)
# ---------------------------------------------------------------------------


def test_historical_norm_evidence_real():
    gate = drv.Gate()
    hist = drv._historical_norm_evidence(gate)
    assert gate.ok, [c for c in gate.checks if not c["ok"]]
    assert hist["norm_summary"]["winner"] in (None, False)
    for p, dist in (("S1", {"PASS": 41, "C2_ONLY": 72}),
                    ("S2", {"PASS": 57, "C2_ONLY": 56})):
        prof = hist["profiles"][p]
        assert len(prof["ff_rows"]) == 5
        assert all(r["adopted"] for r in prof["ff_rows"])
        assert prof["cf_distribution"] == dist
        # the whole historical cf directory is content-registered
        # (json count includes cf_index.json; the npz products do not)
        n_npz = sum(1 for k in prof["cf_dir"]["manifest"]
                    if k.startswith("cf_") and k.endswith(".npz"))
        assert n_npz == 113
        assert len(prof["cf_dir"]["manifest"]) == 227


def test_cf_hist_map_real_41_72_and_57_56():
    for p, dist in (("S1", {"PASS": 41, "C2_ONLY": 72}),
                    ("S2", {"PASS": 57, "C2_ONLY": 56})):
        m = drv._cf_hist_map(p)
        assert len(m) == 113
        counts: dict[str, int] = {}
        for v in m.values():
            counts[v] = counts.get(v, 0) + 1
        assert counts == dist


# ---------------------------------------------------------------------------
# compact QP serialization for a whole-frame escalation
# ---------------------------------------------------------------------------


def test_fp_trace_arrays_compact_identities_all_matrices_last_only():
    trace = {"attempts": [
        {"iterations": [{"z": np.zeros(36), "xi_raw": np.ones(36)}],
         "qp_arrays": [
             {"row_names": ["r0", "r1"], "col_names": ["xi0"],
              "P": np.eye(1), "q": np.zeros(1), "A": np.ones((2, 1)),
              "lo": np.zeros(2), "hi": np.ones(2)},
             {"row_names": ["r0", "r1", "cut0"], "col_names": ["xi0"],
              "P": np.eye(1), "q": np.zeros(1), "A": np.ones((3, 1)),
              "lo": np.zeros(3), "hi": np.ones(3)}]},
        {"iterations": [],
         "qp_arrays": [
             {"row_names": ["r0"], "col_names": ["xi0", "s0"],
              "P": np.eye(2), "q": np.zeros(2), "A": np.ones((1, 2)),
              "lo": np.zeros(1), "hi": np.ones(1)}]}]}
    arr = drv._fp_trace_arrays(trace)
    # row/col identities for EVERY QP (Q3's one-to-one binding)
    for ai, qi, rows in ((0, 0, 2), (0, 1, 3), (1, 0, 1)):
        assert arr[f"a{ai}_qp{qi}_row_names"].shape == (rows,)
    assert "a0_it0_z" in arr and "a0_it0_xi_raw" in arr
    # full numeric matrices ONLY for the final attempt's last QP
    assert "a1_qp0_P" in arr and "a1_qp0_A" in arr
    assert "a0_qp0_P" not in arr and "a0_qp1_A" not in arr
    assert "a0_qp0_q" not in arr and "a0_qp1_lo" not in arr


# ---------------------------------------------------------------------------
# stage_evidence: flow / resume / located differences / budget / failure
# ---------------------------------------------------------------------------


def _fabricated_ff_run(hist):
    def run(prof, att):
        rows = [{k: r[k] for k in ("stem", "demo", "t", "adopted",
                                   "adopted_source", "last_status",
                                   "last_term_reason")}
                for r in hist["profiles"][prof.name]["ff_rows"]]
        for r in rows:
            r["outcome"] = "adopted_audit_pass"
        (att / "ff").mkdir(parents=True, exist_ok=True)
        (att / "ff" / "ff_index.json").write_text(json.dumps(
            {"format": drv.RECOVERY_FORMAT,
             "stems": [f"ff_{r['stem']}" for r in rows],
             "records": [{"stem": f"ff_{r['stem']}"} for r in rows]}))
        drv.write_guarded(att / "ff_stage_summary.json", json.dumps({
            "n": len(rows),
            "counts": {"n_adopted": 5, "n_full_audit_pass": 5,
                       "n_adopted_audit_fail": 0,
                       "n_evidence_incomplete": 0, "n_not_adopted": 0},
            "rows": rows}, indent=1))
        return len(rows)
    return run


def _fabricated_cf_run(hist, flips: dict[str, str] | None = None,
                       flips_profile: str = "S1"):
    def run(prof, att, gate):
        stems = sorted(drv._cf_hist_map(prof.name))
        classes = {s: drv._cf_hist_map(prof.name)[s] for s in stems}
        if flips is not None and prof.name == flips_profile:
            for s, v in flips.items():
                classes[s] = v
        dist: dict[str, int] = {}
        for v in classes.values():
            dist[v] = dist.get(v, 0) + 1
        spikes = [{"stem": f"cf_spk{i}", "rejected": True, "mode": {"r": "norm_cut"}}
                  for i in range(10)]
        (att / "cf").mkdir(parents=True, exist_ok=True)
        (att / "cf" / "cf_index.json").write_text(json.dumps(
            {"format": drv.RECOVERY_FORMAT,
             "stems": [f"cf_{s}" for s in stems],
             "records": [{"stem": f"cf_{s}"} for s in stems]}))
        drv.write_guarded(att / "cf_stage_summary.json", json.dumps({
            "n": len(stems), "distribution": dist, "spikes_ok": True,
            "enable_ok": True, "classifications": classes,
            "spikes": spikes}, indent=1))
        return len(stems)
    return run


def _fabricated_replay():
    def replay(prof_dir, out_dir, *, kinds=("ff", "cf"),
               live_manifest=None, ff_dir=None, cf_dir=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = {"ok": True, "kinds": {
            "ff": {"ok": True, "n": 5, "expected": 5, "entries": {}},
            "cf": {"ok": True, "n": 113, "expected": 113, "entries": {}}}}
        (out_dir / "recovery_replay.json").write_text(json.dumps(doc))
        # the products directories must stay clean (checked by the stage)
        return doc
    return replay


def _patch_evidence(monkeypatch, *, flips=None, cf_raises=False):
    hist = drv._historical_norm_evidence(drv.Gate())   # real frozen targets
    monkeypatch.setattr(drv, "_baseline_registration",
                        lambda gate, out: {"references": {},
                                           "source_manifest_products": 249,
                                           "untracked": []})
    monkeypatch.setattr(drv, "_ff_run", _fabricated_ff_run(hist))
    if cf_raises:
        def raising(prof, att, gate):
            raise RuntimeError("fabricated cf failure")
        monkeypatch.setattr(drv, "_cf_run", raising)
    else:
        monkeypatch.setattr(drv, "_cf_run", _fabricated_cf_run(hist, flips))
    monkeypatch.setattr(drv, "replay_recovery", _fabricated_replay())


def test_stage_evidence_flow_and_resume(tmp_path, monkeypatch):
    _patch_evidence(monkeypatch)
    drv.stage_evidence(tmp_path)
    s1 = json.loads(drv.stage_summary_path(tmp_path, "evidence").read_text())
    assert s1["ok"] and s1["status"] == "COMPLETE"
    assert sorted(s1["subtasks"]) == [
        "cf_S1", "cf_S2", "ff_S1", "ff_S2", "replay_S1_run1",
        "replay_S1_run2", "replay_S2_run1", "replay_S2_run2"]
    assert [s1["subtasks"][n]["processed"] for n in (
        "ff_S1", "cf_S1", "replay_S1_run1", "replay_S2_run2")] == \
        [5, 113, 118, 118]
    assert not s1["not_run_budget"] and not s1["subtask_failures"]
    for p, dist in (("S1", {"PASS": 41, "C2_ONLY": 72}),
                    ("S2", {"PASS": 57, "C2_ONLY": 56})):
        assert s1["per_profile"][p]["cf"]["distribution"] == dist
        assert s1["per_profile"][p]["cf"]["per_item_identical"] is True
        assert s1["per_profile"][p]["ff"]["historical_identical"] is True
        assert all(v["ok"] for r in s1["per_profile"][p]["replays"].values()
                   for v in r.values())
    # the mechanism verdicts are re-emitted from the stage summaries
    names = {c["name"] for c in s1["checks"]}
    assert {"S1_spikes_all_rejected_by_norm", "S2_enable_state_norm_rows",
            "S1_replay_run1_run2_stable", "S2_products_unpolluted"} <= names
    # every verdict green
    assert all(c["ok"] for c in s1["checks"])
    # resume: same command reuses every verified subtask, still passes
    drv.stage_evidence(tmp_path)
    s2 = json.loads(drv.stage_summary_path(tmp_path, "evidence").read_text())
    assert s2["ok"] and s2["status"] == "COMPLETE"
    assert all(v["reused"] for v in s2["subtasks"].values())


def test_stage_evidence_locates_cf_differences_without_forcing(tmp_path, monkeypatch):
    hist = drv._historical_norm_evidence(drv.Gate())
    flips = {s: "PASS" for s in sorted(drv._cf_hist_map("S1"))[:2]
             if drv._cf_hist_map("S1")[s] == "C2_ONLY"}
    assert len(flips) == 2
    _patch_evidence(monkeypatch, flips=flips)
    drv.stage_evidence(tmp_path)
    s = json.loads(drv.stage_summary_path(tmp_path, "evidence").read_text())
    # hard gates still hold (replays/counts/mechanism), but the difference
    # is located: per-stem list + shifted distribution, identical=False
    assert s["ok"] and s["status"] == "COMPLETE"
    cf = s["per_profile"]["S1"]["cf"]
    assert cf["per_item_identical"] is False
    assert len(cf["diffs"]) == 2 and all(d[1:] == ["C2_ONLY", "PASS"]
                                         for d in cf["diffs"])
    assert cf["distribution"] == {"PASS": 43, "C2_ONLY": 70}
    assert cf["historical_distribution"] == {"PASS": 41, "C2_ONLY": 72}
    # S2 untouched
    assert s["per_profile"]["S2"]["cf"]["per_item_identical"] is True


def test_stage_evidence_budget_incomplete(tmp_path, monkeypatch):
    _patch_evidence(monkeypatch)
    drv.stage_evidence(tmp_path, budget_min=1e-6)     # deadline already gone
    s = json.loads(drv.stage_summary_path(tmp_path, "evidence").read_text())
    assert s["status"] == "INCOMPLETE" and not s["ok"]
    assert len(s["not_run_budget"]) == 8
    assert "ff" not in s["per_profile"]["S1"]


def test_stage_evidence_subtask_failure_recorded(tmp_path, monkeypatch):
    _patch_evidence(monkeypatch, cf_raises=True)
    drv.stage_evidence(tmp_path)
    s = json.loads(drv.stage_summary_path(tmp_path, "evidence").read_text())
    assert s["status"] == "INCOMPLETE" and not s["ok"]
    assert set(s["subtask_failures"]) == {"cf_S1", "cf_S2"}
    assert "fabricated cf failure" in s["subtask_failures"]["cf_S1"]
    # ff subtasks still ran and registered
    assert "ff_S1" in s["subtasks"] and s["subtasks"]["ff_S1"]["processed"] == 5


# ---------------------------------------------------------------------------
# stage_failure_prefix: prerequisite / real specs / reuse / mismatch
# ---------------------------------------------------------------------------


def _fake_evidence_summary(root: Path, *, ok=True, status="COMPLETE"):
    p = drv.stage_summary_path(root, "evidence")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"stage": "evidence", "ok": ok, "status": status}))


def test_stage_failure_prefix_refuses_without_passed_evidence(tmp_path):
    # missing summary: load_stage_summary's explicit prerequisite failure
    with pytest.raises(RuntimeError, match="summary missing"):
        drv.stage_failure_prefix(tmp_path)
    # present but not ok: the same prerequisite contract
    _fake_evidence_summary(tmp_path, ok=False)
    with pytest.raises(RuntimeError, match="did not pass"):
        drv.stage_failure_prefix(tmp_path)
    # passed but INCOMPLETE (budget stop): the stage's own refusal
    _fake_evidence_summary(tmp_path, ok=True, status="INCOMPLETE")
    with pytest.raises(SystemExit, match="must fully PASS"):
        drv.stage_failure_prefix(tmp_path)


def _real_specs():
    specs = {"solve": [], "anchor": []}
    for p in ("S1", "S2"):
        rep = json.loads((HIST / "norm" / p / "sixseg" / "rq"
                          / "regen_report_diagnosis6_strict.json").read_text())
        for row in rep["rows"]:
            if row["final_stage"] == "state_constraints":
                specs["solve"].append((p, row["demo"], row["t_fail"]))
            elif p == "S2" and row["final_stage"] == "final_validation":
                specs["anchor"].append((p, row["demo"], None))
    return specs


def test_stage_failure_prefix_real_spec_counts():
    specs = _real_specs()
    assert len(specs["solve"]) == 10          # S1 x6 + S2 x4
    assert [(p, d) for p, d, _ in specs["solve"] if p == "S1"] == [
        ("S1", "sub10_largebox_000"), ("S1", "sub12_largebox_001"),
        ("S1", "sub16_largebox_000"), ("S1", "sub3_largebox_000"),
        ("S1", "sub7_largebox_000"), ("S1", "sub8_largebox_000")]
    assert [(p, d) for p, d, _ in specs["solve"] if p == "S2"] == [
        ("S2", "sub10_largebox_000"), ("S2", "sub12_largebox_001"),
        ("S2", "sub7_largebox_000"), ("S2", "sub8_largebox_000")]
    assert [(p, d) for p, d, _ in specs["anchor"]] == [
        ("S2", "sub16_largebox_000"), ("S2", "sub3_largebox_000")]


def _fake_bindings(monkeypatch, *, fail_demo=None):
    def solve(profile, demo, row, rep_path, att):
        if demo == fail_demo:
            raise RuntimeError("identity mismatch vs the original report -- "
                               "binding STOPPED")
        att.mkdir(parents=True, exist_ok=True)
        (att / f"bind_{demo}.json").write_text(json.dumps(
            {"demo": demo, "identity": {"match": True}}))
        (att / f"bind_{demo}.npz").write_bytes(b"npz")
        (att / f"esc_{demo}.json").write_text("{}")

    def anchor(profile, demo, row, rep_path, att):
        att.mkdir(parents=True, exist_ok=True)
        (att / f"anchor_{demo}.json").write_text(json.dumps(
            {"demo": demo, "identity": {"match": True}}))
        (att / f"anchor_{demo}.npz").write_bytes(b"npz")
        for pi in (0, 1):
            (att / f"anchor_{demo}_pass{pi}_scene.npz").write_bytes(b"npz")

    monkeypatch.setattr(drv, "_bind_solve_failure", solve)
    monkeypatch.setattr(drv, "_bind_anchor_failure", anchor)


def test_stage_failure_prefix_flow_and_reuse(tmp_path, monkeypatch):
    _fake_evidence_summary(tmp_path)
    _fake_bindings(monkeypatch)
    drv.stage_failure_prefix(tmp_path)
    s1 = json.loads(drv.stage_summary_path(tmp_path, "failure_prefix").read_text())
    assert s1["ok"] and s1["status"] == "COMPLETE"
    assert s1["n_solve_bindings"] == 10 and s1["n_anchor_bindings"] == 2
    assert len(s1["subtasks"]) == 12
    assert all(c["ok"] for c in s1["checks"])
    # the anchor outputs registered include the two pass scenes
    reg = json.loads((tmp_path / "failure_prefix" / "subtasks.json").read_text())
    anchor_att = next(v[-1] for k, v in reg["subtasks"].items()
                      if k.startswith("anchor_"))
    outs = list(anchor_att["outputs"])
    assert any("pass0_scene.npz" in o for o in outs)
    assert any("pass1_scene.npz" in o for o in outs)
    # resume reuses every verified binding
    drv.stage_failure_prefix(tmp_path)
    s2 = json.loads(drv.stage_summary_path(tmp_path, "failure_prefix").read_text())
    assert s2["ok"] and all(v["reused"] for v in s2["subtasks"].values())


def test_stage_failure_prefix_mismatch_fails_binding(tmp_path, monkeypatch):
    _fake_evidence_summary(tmp_path)
    _fake_bindings(monkeypatch, fail_demo="sub3_largebox_000")
    drv.stage_failure_prefix(tmp_path)
    s = json.loads(drv.stage_summary_path(tmp_path, "failure_prefix").read_text())
    assert not s["ok"] and s["status"] == "INCOMPLETE"
    failed = {k for k in s["binding_failures"]}
    # S2's sub3 is an ANCHOR binding (final_validation), not a solve one
    assert failed == {"solve_S1_sub3_largebox_000"}
    # the failed attempt stays on disk as evidence, no binding product
    reg = json.loads((tmp_path / "failure_prefix" / "subtasks.json").read_text())
    att = reg["subtasks"]["solve_S1_sub3_largebox_000"][-1]
    assert att["status"] == "failed"
    assert not (Path(att["dir"]) / f"bind_sub3_largebox_000.json").exists()
    # the other 11 bindings still succeeded
    assert sum(1 for c in s["checks"] if c["name"].startswith("bound_")
               and c["ok"]) == 11


# ---------------------------------------------------------------------------
# the anchor binding's data flow (regression for the 2026-10-03 crash) and
# the plan-12 scoped reuse across a revision change
# ---------------------------------------------------------------------------


def test_anchor_binding_writes_loadable_scenes(tmp_path, monkeypatch):
    """The final_fail trajectory IS the [T,43] validator scene: the binding's
    per-pass NPZs must round-trip through ``load_teacher_npz`` (the crash
    hstacked the box onto it -> [T,50] -> every binding failed post-solve)."""
    import train24fix2_cert_trace as ct
    from ireset.data.teacher import load_teacher_npz
    from ireset.repair.config_io import load_mode_configs

    T = 20
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(T, 43))
    human = rng.normal(size=(T, 52, 3))
    loc, fff = "right_hand anchor t=14", 14

    class _Chk:
        passed = False
        location = loc
        first_failure_frame = fff

    class _Final:
        passed = False
        checks = {"hand_grasp": _Chk()}

    class _Ctx:
        c2_hand_enabled = {"l": True, "r": True}
        hand_anchor = {"l": None, "r": None}

    class _Res:
        accepted = False
        diagnostics = {
            "final_stage": "final_validation", "t_fail": None,
            "suffix_retry_from": 9,
            "failed_checks": {"hand_grasp": {"location": loc,
                                             "first_failure_frame": fff}},
            "final_retry_failed_checks": {"hand_grasp": {"location": loc,
                                                          "first_failure_frame": fff}},
            "cost": {"passes": [
                {"trajectory_pass": 0, "status": "accepted", "complete": True,
                 "t_fail": None, "accepted_frames": T, "attempts": 5,
                 "qp_calls_total": 100, "iterations_total": 40},
                {"trajectory_pass": 1, "status": "accepted", "complete": True,
                 "t_fail": None, "accepted_frames": T, "attempts": 4,
                 "qp_calls_total": 80, "iterations_total": 30}]},
            "c2_hand_speed": {"frames_on": {"l": T, "r": T}}}

    class _Pipe:
        def __init__(self):
            self.snapshot = None

        def repair_trajectory(self, scene, cond):
            for pi in (0, 1):
                self.snapshot("final_fail", {
                    "pass_index": pi, "trajectory": traj.copy(),
                    "final": _Final(), "first_failure_frame": fff})
                self.snapshot("frame", {
                    "t": 1, "pass_index": pi, "ctx": _Ctx(),
                    "z_pred": None, "z_prev": None, "box": None,
                    "prediction": None})
            return _Res()

    rcfg, vcfg = load_mode_configs("strict")

    class _Prof:
        name = "S2"

        def as_record(self):
            return {"name": "S2"}

    def fake_rerun(prof, demo):
        return traj, human, 30.0, "cond", rcfg, vcfg, _Prof(), _Pipe()

    def fake_freeze(npz, demo, cond, vcfg, rcfg, *, with_series=False):
        return {"task_mpjpe": {"p50": 0.1, "p95": 0.11, "max": 0.2},
                "sides": {side: {
                    "certified_intervals": [[4, 16]],
                    "anchor_drift_certified_max_m": 0.05,
                    "speed_ref_max_mps": 0.03,
                    "series": {"h_speed": [None] + [0.01] * (T - 1),
                               "anchor_drift_curves": {"4-16": [0.0] * 12}}}
                    for side in ("l", "r")}}

    monkeypatch.setattr(drv, "_rerun_segment", fake_rerun)
    monkeypatch.setattr(ct, "freeze_segment", fake_freeze)
    monkeypatch.setattr(ct, "ref_intervals_of",
                        lambda cond, eff: [(2, 18)])

    row = {"final_stage": "final_validation", "t_fail": None,
           "suffix_retry": {"from_frame": 9,
                            "first_attempt_failed_checks": {
                                "hand_grasp": {"location": loc,
                                               "first_failure_frame": fff}}},
           "rejection": {"violated_rows": [f"hand_grasp: {loc}"],
                         "validator_failed_checks": {
                             "hand_grasp": {"location": loc,
                                            "first_failure_frame": fff}}},
           "cost": {"passes": _Res.diagnostics["cost"]["passes"]},
           "c2_hand_speed": {"frames_on": {"l": T, "r": T}}}
    att = tmp_path / "att"
    att.mkdir()
    rep = tmp_path / "report.json"
    rep.write_text("{}")                       # hashed into the source record
    prod = drv._bind_anchor_failure("S2", "sub16_largebox_000", row,
                                     rep, att)
    assert prod["identity"]["match"] is True
    assert len(prod["passes"]) == 2
    assert prod["hand_anchor_verdict"]["any_nonempty"] is False
    # THE regression: both pass scenes load as teacher qpos [T, 43]
    for pi in (0, 1):
        scene, _h, _f = load_teacher_npz(
            att / f"anchor_sub16_largebox_000_pass{pi}_scene.npz")
        assert scene.shape == (T, 43)
    z = np.load(att / "anchor_sub16_largebox_000.npz")
    assert z["pass0_trajectory"].shape == (T, 43)
    # the per-side analysis consumed the fabricated freeze output
    s = prod["passes"][0]["sides"]["r"]
    assert s["ref_intervals"] == [[2, 18]]
    assert s["certified_intervals"] == [[4, 16]]
    assert s["anchor_establishment_frames"] == [4]
    assert s["interval_start_differences"]["first_c2_enable"] == 1
    assert s["hand_anchor_ever_nonempty"] is False


def test_scope_change_reruns_only_affected_kind(tmp_path, monkeypatch):
    """Plan 12's 'rerun only the AFFECTED tasks': a source change in one
    binding kind's tooling forces that kind's attempts to rerun while the
    other kind's verified attempts are reused across the revision change."""
    _fake_evidence_summary(tmp_path)
    _fake_bindings(monkeypatch)
    real = drv._binding_scope_sha
    drv.stage_failure_prefix(tmp_path)
    s1 = json.loads(drv.stage_summary_path(tmp_path, "failure_prefix").read_text())
    assert s1["ok"]
    reg1 = json.loads((tmp_path / "failure_prefix" / "subtasks.json").read_text())

    # simulate a commit that only touched the ANCHOR tooling
    monkeypatch.setattr(drv, "_binding_scope_sha",
                        lambda kind: real(kind) if kind == "solve" else "changed")
    drv.stage_failure_prefix(tmp_path)
    s2 = json.loads(drv.stage_summary_path(tmp_path, "failure_prefix").read_text())
    assert s2["ok"] and s2["status"] == "COMPLETE"
    reg2 = json.loads((tmp_path / "failure_prefix" / "subtasks.json").read_text())
    for k, v in reg2["subtasks"].items():
        if k.startswith("solve_"):
            assert len(v) == 1, "unaffected kind must be reused verbatim"
            assert v[0]["dir"] == reg1["subtasks"][k][0]["dir"]
        else:
            assert len(v) == 2, "affected kind must rerun into attempt_002"
            assert v[-1]["identity"]["code_scope"]["sha256"] == "changed"
    assert sum(1 for x in s2["subtasks"].values() if x["reused"]) == 10


def test_reusable_attempt_bootstraps_only_audited_pre_scope_attempts(tmp_path):
    """Attempts recorded before the scoped-reuse commit carry git_rev but no
    code fingerprint: they reuse ONLY through the audited bootstrap map
    (the fixing commit's diff provably left their kind's functions
    byte-identical) -- an unaudited rev never bootstraps."""
    reg = drv.SubtaskRegistry(tmp_path)
    sem = {"profile": "solve:S1", "registry_sha256": "r", "set": "sub10",
           "task_gate": 0.4, "report_sha256": "rep", "t_fail": 60,
           "final_stage": "state_constraints"}
    ident = {**sem, "code_scope": {"kind": "solve",
                                    "sha256": drv._binding_scope_sha("solve")}}
    att_dir = reg.begin("solve_S1_sub10", tmp_path,
                        {**sem, "git_rev": "258802c", "git": {}}, 1)
    out = att_dir / "o.json"
    out.write_text("{}")
    reg.complete("solve_S1_sub10", [out], 1)
    # the bootstrap constant must BE the live solve scope (258802c's solve
    # functions are byte-identical to the fixing commit's)
    assert drv._BOOTSTRAP_SCOPES[("258802c", "solve")] \
        == drv._binding_scope_sha("solve")
    assert drv._reusable_attempt(reg, "solve_S1_sub10", ident) is not None
    # an unaudited pre-scope rev never bootstraps (reload the registry --
    # the edit went to disk, not to the in-memory doc)
    doc = json.loads((tmp_path / "subtasks.json").read_text())
    doc["subtasks"]["solve_S1_sub10"][0]["identity"]["git_rev"] = "deadbee"
    (tmp_path / "subtasks.json").write_text(json.dumps(doc))
    reg2 = drv.SubtaskRegistry(tmp_path)
    assert drv._reusable_attempt(reg2, "solve_S1_sub10", ident) is None
    # and a scope change still refuses even the bootstrapped rev
    doc["subtasks"]["solve_S1_sub10"][0]["identity"]["git_rev"] = "258802c"
    (tmp_path / "subtasks.json").write_text(json.dumps(doc))
    reg3 = drv.SubtaskRegistry(tmp_path)
    changed = {**ident, "code_scope": {"kind": "solve", "sha256": "changed"}}
    assert drv._reusable_attempt(reg3, "solve_S1_sub10", changed) is None


# ---------------------------------------------------------------------------
# stage_recovery: the R-candidate development comparison driver
# ---------------------------------------------------------------------------


def test_stage_recovery_flow_and_refusal(tmp_path, monkeypatch):
    """The R driver refuses without the Q/P pre-gates and runs the
    registered R candidates in order (only R1 exists today: ff + cf + one
    quiet sixseg + the full gate; a passing candidate wins, later ones are
    recorded not_executed)."""
    with pytest.raises(RuntimeError, match="summary missing"):
        drv.stage_recovery(tmp_path)
    _fake_evidence_summary(tmp_path)
    # failure-prefix summary still missing
    with pytest.raises(RuntimeError, match="failure_prefix.*summary missing"):
        drv.stage_recovery(tmp_path)
    # failure-prefix present but not ok
    fp = drv.stage_summary_path(tmp_path, "failure_prefix")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({"stage": "failure_prefix", "ok": False,
                              "status": "COMPLETE"}))
    with pytest.raises(RuntimeError, match="did not pass"):
        drv.stage_recovery(tmp_path)
    fp.write_text(json.dumps({"stage": "failure_prefix", "ok": True,
                              "status": "COMPLETE"}))

    calls: list[str] = []

    def fake_ff(prof, diag, pdir):
        calls.append(f"ff:{prof.name}")
        rows = [{"stem": f"fb{i}", "demo": "d", "t": i, "adopted": True,
                 "adopted_source": "prediction", "last_status": "solved",
                 "last_term_reason": "converged", "outcome":
                     "adopted_audit_pass"} for i in range(5)]
        (pdir / "ff").mkdir(parents=True, exist_ok=True)
        return {"rows": rows, "n_adopted": 5, "n_full_audit_pass": 5,
                "n_adopted_audit_fail": 0, "n_evidence_incomplete": 0,
                "n_not_adopted": 0}

    def fake_cf(prof, diag, pdir, gate):
        calls.append(f"cf:{prof.name}")
        return {"rows": [], "distribution": {"PASS": 113}, "n": 113,
                "spikes": [], "spikes_ok": True, "enable_ok": True}

    def fake_six(profile_name, rq, **kw):
        calls.append(f"sixseg:{profile_name}")
        rq.mkdir(parents=True, exist_ok=True)
        return {"rq_dir": str(rq), "wall_s": 1.0, "n_demos": 6,
                "n_kept": 6, "kept": ["d1"], "code_changed_during_run": False,
                "report": {}}

    import train24fix3_quality_gate as qg_mod
    monkeypatch.setattr(drv, "_profile_frozen_frames", fake_ff)
    monkeypatch.setattr(drv, "_norm_counterfactuals", fake_cf)
    monkeypatch.setattr(drv, "_run_sixseg", fake_six)
    gate_calls: list[str] = []

    def fake_gate(run_dir, **kw):
        gate_calls.append(kw.get("profile"))
        return {"ok": True, "verdicts": {"everything": True}}
    monkeypatch.setattr(qg_mod, "gate_report", fake_gate)

    drv.stage_recovery(tmp_path)
    s = json.loads(drv.stage_summary_path(tmp_path, "recovery").read_text())
    assert s["ok"] and s["winner"] == "R1"
    # R2/R3 are registered but the fixed order stops at the winner
    assert s["not_executed"] == ["R2", "R3"]
    assert calls == ["ff:R1", "cf:R1", "sixseg:R1"]
    assert gate_calls == ["R1"]
    # the gate call carried the frozen demo list + the cf mechanism facts
    psum = json.loads((tmp_path / "recovery" / "R1"
                       / "profile_summary.json").read_text())
    assert psum["selected"] is True and psum["sixseg"]["n_kept"] == 6


def test_stage_recovery_reuses_prior_complete_candidate(tmp_path, monkeypatch):
    """A prior COMPLETE comparison of the same resolved profile (products
    intact, R9 clean) is reused verbatim -- a rerun after registering a
    later candidate must not redo the earlier candidate's compute, and the
    refuse-overwrite convention would otherwise turn the rerun into an
    error.  A drifted profile record forces that candidate's rerun."""
    _fake_evidence_summary(tmp_path)
    fp = drv.stage_summary_path(tmp_path, "failure_prefix")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps({"stage": "failure_prefix", "ok": True,
                              "status": "COMPLETE"}))
    calls: list[str] = []

    def fake_ff(prof, diag, pdir):
        calls.append(f"ff:{prof.name}")
        return {"rows": [], "n_adopted": 0, "n_full_audit_pass": 0,
                "n_adopted_audit_fail": 0, "n_evidence_incomplete": 0,
                "n_not_adopted": 0}

    def fake_cf(prof, diag, pdir, gate):
        calls.append(f"cf:{prof.name}")
        return {"rows": [], "distribution": {}, "n": 113, "spikes": [],
                "spikes_ok": True, "enable_ok": True}

    def fake_six(profile_name, rq, **kw):
        calls.append(f"sixseg:{profile_name}")
        rq.mkdir(parents=True, exist_ok=True)
        rep = rq / "regen_report_diagnosis6_strict.json"
        rep.write_text(json.dumps({
            "n_kept": 1, "rows": [{"demo": "sub16_largebox_000",
                                   "accepted": True}],
            "end": {"code_changed_during_run": False}}))
        return {"rq_dir": str(rq), "wall_s": 1.0, "n_demos": 6,
                "n_kept": 1, "kept": ["sub16_largebox_000"],
                "code_changed_during_run": False, "report": {}}

    import train24fix3_quality_gate as qg_mod
    monkeypatch.setattr(drv, "_profile_frozen_frames", fake_ff)
    monkeypatch.setattr(drv, "_norm_counterfactuals", fake_cf)
    monkeypatch.setattr(drv, "_run_sixseg", fake_six)
    monkeypatch.setattr(qg_mod, "gate_report",
                        lambda run_dir, **kw: {"ok": False, "verdicts": {}})
    # first run: R1 executed (failed its gate), then R2
    drv.stage_recovery(tmp_path)
    assert calls == ["ff:R1", "cf:R1", "sixseg:R1",
                     "ff:R2", "cf:R2", "sixseg:R2",
                     "ff:R3", "cf:R3", "sixseg:R3"]
    # second run: every prior comparison reused, nothing re-executed
    calls.clear()
    drv.stage_recovery(tmp_path)
    s2 = json.loads(drv.stage_summary_path(tmp_path, "recovery").read_text())
    assert calls == []
    assert all(s2["comparisons"][n]["reused"] is True
               for n in ("R1", "R2", "R3"))
    assert s2["winner"] is None
    # a drifted profile record: the prior comparison neither verifies for
    # reuse NOR may be overwritten -- an explicit refusal, never a silent
    # redo (retiring prior products is a registered decision)
    ps = json.loads((tmp_path / "recovery" / "R2"
                     / "profile_summary.json").read_text())
    ps["profile"]["c2_norm_speed_mps"] = 0.049
    (tmp_path / "recovery" / "R2" / "profile_summary.json").write_text(
        json.dumps(ps))
    with pytest.raises(SystemExit, match="does not verify for reuse"):
        drv.stage_recovery(tmp_path)
    assert calls == []


# ---------------------------------------------------------------------------
# freeze_segment with_series: the cumulative anchor-drift curves
# ---------------------------------------------------------------------------


def test_freeze_segment_drift_curves_consistent_with_max():
    from ireset.repair.config_io import load_mode_configs
    from train24fix2_cert_trace import load_cond, freeze_segment

    demo = "sub8_largebox_000"        # the B reference's richest intervals
    npz = drv.TEACHERS / "train24fix_20260927_sixseg" / "rq" / f"{demo}.npz"
    if not npz.is_file():
        pytest.skip("B rq reference not present")
    _src, cond = load_cond(demo)
    rcfg, vcfg = load_mode_configs("strict")
    m = freeze_segment(npz, demo, cond, vcfg, rcfg, with_series=True)
    found = False
    for side in ("l", "r"):
        series = m["sides"][side].get("series") or {}
        curves = series.get("anchor_drift_curves") or {}
        ivs = [tuple(iv) for iv in m["sides"][side]["certified_intervals"]]
        assert sorted(tuple(int(x) for x in k.split("-")) for k in curves) \
            == sorted(ivs)
        for key, curve in curves.items():
            a, b = (int(x) for x in key.split("-"))
            assert len(curve) == b - a
            assert all(v >= 0.0 for v in curve)
            assert curve[0] == pytest.approx(0.0, abs=1e-12)
        if curves:
            found = True
            worst = max(max(c) for c in curves.values())
            assert worst == pytest.approx(
                m["sides"][side]["anchor_drift_certified_max_m"],
                rel=0, abs=1e-15)
    assert found, "sub8 must carry certified intervals on both sides"
