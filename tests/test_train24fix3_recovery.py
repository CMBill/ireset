"""train24fix3 continuation pre-stage Q: evidence-chain fixes Q1-Q4
(plan sections 4.3-4.4 / 11.2).

Counterexamples pinned here:

- Q1: a legal norm displacement audited WITHOUT the profile is rejected by
  the axis interpreter while the SAME candidate passes with the frozen
  profile attached -- the frozen-frame audit must carry the profile, and
  its arrays are frozen (never discarded); adopted vs full-audit-pass are
  counted separately;
- Q2: the recovery loader refuses deleted integrity, doctored live scp
  digests, deleted OSQP/profile fields, wrong source hashes and tampered
  NPZ; ``configs_from_record`` backfills ONLY v2 (a new-format record
  missing solver fields is refused); the integrity manifest covers this
  round's driver/quality-gate/registry; the strict replay recomputes BOTH
  counterfactual branches, re-classifies, re-queries the source distances
  fresh, matches n==expected, is stable on a second run and never writes
  into the products directory;
- Q3: every ``prob.solve()`` is one-to-one with its own metadata -- the
  branch/outer-iter/cut-round identity, the ACTUAL matrix shape and the
  per-QP row-name/slack arrays (initial QP + cut re-solves + restoration),
  with a monotonic QP id; trace-off keeps old N/S behaviour bit-identical;
- Q4: written array keys carry exactly ONE ``rows__`` prefix; the v1
  double-prefix layout is decoded explicitly (with a version marker) and a
  mixed layout is corruption, not a rename.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.pipeline import RepairPipeline  # noqa: E402
from ireset.repair.profiles import get_profile  # noqa: E402
from ireset.repair.scp import N_TANGENT  # noqa: E402
from ireset.repair.validation import ValidationConfig  # noqa: E402
from test_band_floor_separation import _place_box  # noqa: E402
from test_hand_speed_row import REGEN_C2  # noqa: E402
from test_repair import standing_scene  # noqa: E402

import train24fix3_convergence as drv  # noqa: E402
import train24fix2_audit as A  # noqa: E402

S1 = get_profile("S1")
DT = 1.0 / 30.0
B_S1 = DT * 0.045
#: a legal-for-S1 displacement: per-site NORM inside dt*V, one AXIS over
#: the old dt*0.02598 axis bound (the Q1 falsification input)
LEGAL_NORM_D = np.array([[0.0013, 0.0, 0.0], [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])


def _ctx_with_prev(pipe, z, box, d_target):
    pipe.ev.forward(z, box)
    cur = pipe.ev.hand_local("l")
    prev = cur - np.asarray(d_target, dtype=float)
    return A.FrameContext(
        t=1, box_pose=box.copy(), z_pred=z.copy(), z_prev=None,
        ref_kpts_world=pipe.ev.site_positions(pipe.ev.kpt_ids).copy(),
        mesh_L=None, delta_ref=None, mesh_fixed=None,
        hand_anchor={}, foot_anchor={},
        prev_hand_local={"l": prev.copy()},
        prev_sole_world={}, box_pose_prev=box.copy(),
        ref_hold={"left_hand": True}, prev_hand_dist={"l": 0.012},
        frame_motion_m=0.02,
        c2_hand_enabled={"l": True}, c2_hand_reason={"l": "hold"},
        prev_hand_scene_dist={"l": 0.0012})


def _setup(profile=None):
    pipe = RepairPipeline(cfg=REGEN_C2, c2_profile=profile)
    z = standing_scene(1)[0, 0:36].copy()
    box = _place_box(pipe, z, "l", 0.006)
    return pipe, z, box


def _frozen_meta(ctx):
    return {**A.context_meta(ctx),
            "ctx_scalars": A.context_scalar_leaves(ctx)}


# ---------------------------------------------------------------------------
# Q1: the audit judges the SAME semantics the candidate ran under
# ---------------------------------------------------------------------------


def test_q1_legal_norm_candidate_axis_fails_norm_audit_passes() -> None:
    """The Q1 falsification: one candidate, two audits.  Without the
    profile the endpoint rows are the old AXIS rows and the legal-norm
    displacement is rejected; with the frozen S1 profile attached the real
    norm rows accept it.  The pre-Q1 ``_profile_frozen_frames`` called the
    audit WITHOUT the profile -- every norm adoption was judged (and its
    arrays dropped) by the axis interpreter."""
    pipe, z, box = _setup()
    ctx = _ctx_with_prev(pipe, z, box, LEGAL_NORM_D)

    def audit(profile):
        return A.full_candidate_audit(
            pipe.spec, REGEN_C2, ValidationConfig(), ctx, z.copy(),
            prefix=[], box_poses=np.asarray([box]), label="probe",
            c2_profile=profile)

    axis_rec, axis_arr = audit(None)
    norm_rec, norm_arr = audit(S1)
    assert "hand_speed[l:palm:x]" in axis_rec["endpoint"]["violating"]
    assert norm_rec["endpoint"]["viol_sum"] == 0.0
    assert "hand_norm[l:" in " ".join(
        norm_rec["endpoint"]["descriptors"])
    # the arrays are part of the evidence (the pre-Q1 code discarded them)
    for arr in (axis_arr, norm_arr):
        assert set(arr) == {f"rows__probe__{k}" for k in
                            ("val", "lo", "hi", "scale", "viol", "z_raw",
                             "z_checked")}
        assert arr["rows__probe__val"].ndim == 1


def test_q1_frozen_frames_pass_profile_save_products_and_count(tmp_path,
                                                               monkeypatch):
    """End-to-end driver pass: the frozen-frame audit receives the frozen
    profile (spied), the product freezes the audit arrays + context +
    prefix under the recovery schema, the index is dual-view, and the
    counts list adopted / full-audit-pass / audit-fail / evidence gaps
    separately."""
    pipe, z, box = _setup(S1)
    ctx = _ctx_with_prev(pipe, z, box, LEGAL_NORM_D)
    diag = tmp_path / "diag"
    stem = "fake_fb_stem"
    fb_record = {
        "demo": "fake_demo", "t_fail": 1, "pass_index": 0,
        "provenance": {"context_meta": _frozen_meta(ctx)},
        "last_attempt": {
            "attempt_id": [0, 1, 0],
            "chk": {"ok": False, "reason": "fabricated"},
            "solver_fields": {"status": "infeasible", "term_reason": "old",
                              "iterations": 7, "qp_calls": 9}},
        "effective": {"repair": asdict(replace(RepairConfig(),
                                               regen_mode=True)),
                      "validation": asdict(ValidationConfig())},
    }
    A.write_products(diag, stem, fb_record, {
        **A.context_arrays(ctx), "prefix": np.zeros((0, 36)),
        "box_poses": np.asarray([box])})
    A.write_index(diag, "fail_bind2_index", {"stems": [stem]})

    seen: list[dict] = []
    real_audit = A.full_candidate_audit

    def spy(spec, cfg, vcfg, actx, z_raw, **kw):
        seen.append(dict(kw))
        return real_audit(spec, cfg, vcfg, actx, z_raw, **kw)

    monkeypatch.setattr(A, "full_candidate_audit", spy)
    pdir = tmp_path / "prof"
    ff = drv._profile_frozen_frames(S1, diag, pdir)

    # Q1: the frozen profile reached the audit (the pre-Q3 call omitted it)
    assert seen and seen[0]["c2_profile"] is S1
    row = ff["rows"][0]
    assert row["adopted"] and row["outcome"] in (
        "adopted_audit_pass", "adopted_audit_fail",
        "adopted_evidence_incomplete")
    assert ff["n_adopted"] == 1
    # adopted and full-audit-pass are INDEPENDENT counts
    assert ff["n_full_audit_pass"] == (1
                                       if row["outcome"] == "adopted_audit_pass"
                                       else 0)
    assert ff["n_full_audit_pass"] + ff["n_adopted_audit_fail"] \
        + ff["n_evidence_incomplete"] == ff["n_adopted"]

    # the product froze the audit arrays (single prefix), the context, the
    # candidate and the prefix -- and loads through the verifying loader
    name = f"ff_{stem}"
    record, arrays = drv.load_recovery_products(pdir / "ff", name)
    assert record["kind"] == "frozen_frame"
    assert any(k.startswith("rows__profile_adopted__") for k in arrays)
    assert "ctx__z_pred" in arrays and "adopted_z" in arrays
    idx = drv.load_recovery_index(pdir / "ff", name="ff_index")
    assert idx["stems"] == [name]
    with pytest.raises(FileNotFoundError):
        drv.load_recovery_index(pdir, name="ff_index")   # wrong level


# ---------------------------------------------------------------------------
# Q2: configs_from_record backfill is v2-only; integrity coverage
# ---------------------------------------------------------------------------


def _recovery_eff(drop=()):
    repair = {k: v for k, v in asdict(replace(RepairConfig(),
                                              regen_mode=True)).items()
              if k not in drop}
    return {"repair": repair, "validation": asdict(ValidationConfig())}


def test_q2_configs_from_record_backfills_v2_only() -> None:
    v2 = {"format": A.AUDIT_FORMAT, "effective": _recovery_eff(
        drop=("osqp_polish", "osqp_max_iter"))}
    cfg, _ = A.configs_from_record(v2)          # v2: registered N0 backfill
    assert cfg.osqp_polish is False and cfg.osqp_max_iter == 2500
    for fmt in (drv.RECOVERY_FORMAT, drv.NORM_FORMAT, "anything_else"):
        rec = {"format": fmt, "effective": _recovery_eff(
            drop=("osqp_eps_abs",))}
        with pytest.raises(ValueError, match="backfill is v2-only"):
            A.configs_from_record(rec)
        full = {"format": fmt, "effective": _recovery_eff()}
        cfg2, _ = A.configs_from_record(full)   # complete: accepted verbatim
        assert cfg2.osqp_eps_abs == RepairConfig().osqp_eps_abs


def test_q2_integrity_manifest_covers_fix3_chain() -> None:
    m = A.integrity_manifest()
    for frag in ("scripts/train24fix3_convergence.py",
                 "scripts/train24fix3_quality_gate.py",
                 "configs/repair/train24fix3_profiles.yaml",
                 "src/ireset/repair/profiles.py"):
        assert frag in m, f"manifest misses {frag}"


# ---------------------------------------------------------------------------
# Q2: a fabricated-source fixture shared by the loader/replay tests
# ---------------------------------------------------------------------------


def _fabricated_cf_source(tmp_path):
    """One fabricated F-style counterfactual source product on disk: a real
    frozen context (the norm fixture), a real candidate, distance views
    that EQUAL a fresh query of a REAL teacher NPZ, and v2 effective
    configs (so the real configs_from_record/backfill path runs)."""
    from closed_loop_metrics import sha256 as sha256_16
    from train24fix2_cert_trace import (load_teacher_npz,
                                        prev_scene_distances_at)

    pipe, z, box = _setup()
    ctx = _ctx_with_prev(pipe, z, box, LEGAL_NORM_D)
    src_npz = sorted((Path(drv.PROJECT_ROOT)
                      / "derived/data/teachers_regenerated"
                      / "train24fix_20260927_sixseg/rq").glob("*.npz"))[0]
    traj, _h, _fps = load_teacher_npz(src_npz)
    t_src = 1
    fresh = prev_scene_distances_at(traj, t_src)
    diag = tmp_path / "diag"
    stem = "cert_trace_cf_fake_t1_l"
    record = {
        "demo": "fake_demo", "frame": t_src, "side": "l",
        "candidate_source": "fabricated[l]",
        "replay_meta": _frozen_meta(ctx),
        "c2_enable_flags": {"l": True, "r": False},
        "c2_reasons": {"l": "hold", "r": "off"},
        "prev_scene_dist_flags": dict(fresh),
        "prev_scene_dist_both_m": dict(fresh),
        "prev_scene_dist_m": fresh["l"],
        "prev_scene_source": {"npz": str(src_npz),
                              "sha256": sha256_16(src_npz)},
        "classification": "FROZEN",
        "effective": {
            "repair": asdict(replace(RepairConfig(), regen_mode=True)),
            "validation": asdict(ValidationConfig()),
            "repair_no_c2": asdict(replace(
                RepairConfig(), regen_mode=True,
                regen_hand_speed_mps=0.0))},
    }
    A.write_products(diag, stem, record, {
        **A.context_arrays(ctx), "z_cand": z.copy(),
        "prefix": np.zeros((0, 36)), "box_poses": np.asarray([box])})
    A.write_index(diag, "cert_trace_cf_index", {"stems": [stem]})
    return diag, stem, ctx, z, box


def _generate_cf_products(tmp_path):
    diag, stem, ctx, z, box = _fabricated_cf_source(tmp_path)
    pdir = tmp_path / "prof"
    gate = drv.Gate()
    cf = drv._norm_counterfactuals(S1, diag, pdir, gate)
    assert cf["n"] == 1
    return pdir, diag, stem, cf


def test_q2_products_single_prefix_and_full_freeze(tmp_path) -> None:
    pdir, diag, stem, cf = _generate_cf_products(tmp_path)
    record, arrays = drv.load_recovery_products(pdir / "cf", f"cf_{stem}")
    # Q4: exactly ONE rows__ namespace level, matching the audit keys
    rows_keys = [k for k in arrays if k.startswith("rows__")]
    assert rows_keys, "audit arrays were dropped (the Q1 defect)"
    assert any(k.startswith("rows__current__") for k in rows_keys)
    assert any(k.startswith("rows__plus_c2_norm__") for k in rows_keys)
    assert not any(k.count("rows__") > 1 for k in arrays)
    # the freeze is COMPLETE: both audits, both configs, source, profile,
    # the frozen context and the candidate
    assert set(record["audits"]) == {"current", "plus_c2_norm"}
    assert record["classification"] == cf["rows"][0]["classification"]
    assert record["frozen_classification"] == "FROZEN"
    assert set(record["effective"]) == {"repair", "validation",
                                        "repair_no_c2"}
    assert record["source"]["stem"] == stem
    assert record["profile"]["name"] == "S1"
    assert "ctx__z_pred" in arrays and "z_cand" in arrays


def test_q2_loader_rejects_tampering(tmp_path) -> None:
    pdir, diag, stem, _cf = _generate_cf_products(tmp_path)
    prod_dir = pdir / "cf"
    name = f"cf_{stem}"
    drv.load_recovery_products(prod_dir, name)      # untampered: loads

    def load_mutated(mutate, live=None):
        import shutil
        d = tmp_path / f"mut_{mutate.__name__}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(prod_dir, d)
        jpath = d / f"{name}.json"
        rec = json.loads(jpath.read_text())
        mutate(rec, d, name)
        jpath.write_text(json.dumps(rec, indent=1, allow_nan=False,
                                    default=float))
        return drv.load_recovery_products(d, name, live_manifest=live)

    fresh = A.integrity_manifest()

    def del_integrity(rec, *_):
        rec.pop("integrity")

    def del_osqp(rec, *_):
        rec["effective"]["repair"].pop("osqp_polish")

    def del_profile(rec, *_):
        rec.pop("profile")

    def bad_source_hash(rec, *_):
        rec["source"]["json_sha256"] = "0" * 64

    def del_source(rec, *_):
        rec.pop("source")

    def rewrite_npz(rec, d, n):
        import numpy as np
        with np.load(d / f"{n}.npz") as bz:
            arrays = {k: bz[k] for k in bz.files}
        arrays["rows__plus_c2_norm__viol"] = arrays[
            "rows__plus_c2_norm__viol"] + 1.0
        np.savez(d / f"{n}.npz", **arrays)     # hash NOT rewritten

    for mutate, exc in ((del_integrity, ValueError),
                        (del_osqp, ValueError),
                        (del_profile, ValueError),
                        (bad_source_hash, ValueError),
                        (del_source, ValueError),
                        (rewrite_npz, ValueError)):
        with pytest.raises(exc):
            load_mutated(mutate)
    # a doctored LIVE side (scp.py changed since generation) is refused
    with pytest.raises(ValueError, match="integrity drift"):
        load_mutated(lambda rec, *_: None,
                     live={**fresh, "src/ireset/repair/scp.py": "0" * 64})


def test_q2_strict_replay_two_branches_n_expected_stable(tmp_path) -> None:
    pdir, diag, stem, cf = _generate_cf_products(tmp_path)
    prod_dir = pdir / "cf"
    before = sorted(p.name for p in prod_dir.iterdir())
    out1 = tmp_path / "replay" / "run1"
    res1 = drv.replay_recovery(pdir, out1, kinds=("cf",))
    kind = res1["kinds"]["cf"]
    assert res1["ok"] and kind["ok"]
    assert kind["n"] == kind["expected"] == 1      # n == expected is a gate
    entry = kind["entries"][f"cf_{stem}"]
    assert entry["matches"] and entry["classification_matches"]
    assert entry["prev_scene_check"] == "ok"       # fresh four-view query
    # the replay recomputed BOTH branches (never the frozen old summary)
    assert entry["classification_recomputed"] == record_class(pdir, stem)
    # second run is stable and NEVER writes into the products directory
    out2 = tmp_path / "replay" / "run2"
    res2 = drv.replay_recovery(pdir, out2, kinds=("cf",))
    assert res2["ok"] == res1["ok"]
    assert sorted(p.name for p in prod_dir.iterdir()) == before
    assert (out1 / "recovery_replay.json").is_file()

    # tamper: classification flipped in the JSON -> replay MISMATCH
    jpath = prod_dir / f"cf_{stem}.json"
    rec = json.loads(jpath.read_text())
    rec["classification"] = "PASS" if rec["classification"] != "PASS" \
        else "C2_ONLY"
    jpath.write_text(json.dumps(rec, indent=1, allow_nan=False,
                                default=float))
    res3 = drv.replay_recovery(pdir, tmp_path / "replay" / "run3",
                               kinds=("cf",))
    assert not res3["ok"]
    assert not res3["kinds"]["cf"]["entries"][f"cf_{stem}"]["ok"]

    # tamper: a row ARRAY changed with the NPZ self-hash REWRITTEN to match
    # -- the loader passes, the replay's bitwise arrays_diff catches it
    rec = json.loads(jpath.read_text())
    npath = prod_dir / f"cf_{stem}.npz"
    with np.load(npath) as bz:
        arrays = {k: bz[k] for k in bz.files}
    arrays["rows__plus_c2_norm__viol"] = \
        arrays["rows__plus_c2_norm__viol"] + 1e-9
    np.savez(npath, **arrays)
    rec["npz_sha256"] = drv.sha256_full(npath)
    jpath.write_text(json.dumps(rec, indent=1, allow_nan=False,
                                default=float))
    drv.load_recovery_products(prod_dir, f"cf_{stem}")   # hash consistent
    res4 = drv.replay_recovery(pdir, tmp_path / "replay" / "run4",
                               kinds=("cf",))
    assert not res4["ok"]
    entry4 = res4["kinds"]["cf"]["entries"][f"cf_{stem}"]
    assert entry4["array_diffs"]

    # tamper: the distance source removed -> an EXPLICIT failure, no skip
    rec = json.loads(jpath.read_text())
    rec["source"].pop("prev_scene_source")
    jpath.write_text(json.dumps(rec, indent=1, allow_nan=False,
                                default=float))
    res5 = drv.replay_recovery(pdir, tmp_path / "replay" / "run5",
                               kinds=("cf",))
    entry5 = res5["kinds"]["cf"]["entries"][f"cf_{stem}"]
    assert not entry5["ok"]
    assert entry5["prev_scene_check"] != "ok"


def record_class(pdir, stem) -> str:
    return json.loads((pdir / "cf" / f"cf_{stem}.json").read_text())[
        "classification"]


def test_q2_frozen_frame_replay_escalation(tmp_path) -> None:
    """The frozen-frame replay re-runs the FULL escalation chain from the
    frozen context and matches identity + candidate + audit; a doctored
    term_reason is a MISMATCH."""
    pipe, z, box = _setup(S1)
    ctx = _ctx_with_prev(pipe, z, box, LEGAL_NORM_D)
    diag = tmp_path / "diag"
    stem = "fake_ff_stem"
    fb_record = {
        "demo": "fake_demo", "t_fail": 1, "pass_index": 0,
        "provenance": {"context_meta": _frozen_meta(ctx)},
        "last_attempt": {"attempt_id": [0, 1, 0],
                         "chk": {"ok": False, "reason": "fabricated"},
                         "solver_fields": {"status": "infeasible",
                                           "term_reason": "old"}},
        "effective": {"repair": asdict(replace(RepairConfig(),
                                               regen_mode=True)),
                      "validation": asdict(ValidationConfig())},
    }
    A.write_products(diag, stem, fb_record, {
        **A.context_arrays(ctx), "prefix": np.zeros((0, 36)),
        "box_poses": np.asarray([box])})
    A.write_index(diag, "fail_bind2_index", {"stems": [stem]})
    pdir = tmp_path / "prof"
    drv._profile_frozen_frames(S1, diag, pdir)
    out = tmp_path / "replay"
    res = drv.replay_recovery(pdir, out, kinds=("ff",))
    assert res["ok"] and res["kinds"]["ff"]["n"] == 1
    entry = res["kinds"]["ff"]["entries"][f"ff_{stem}"]
    assert entry["ok"] and entry["escalation_matches"]

    jpath = pdir / "ff" / f"ff_{stem}.json"
    rec = json.loads(jpath.read_text())
    rec["escalation"]["last_term_reason"] = "doctored"
    jpath.write_text(json.dumps(rec, indent=1, allow_nan=False,
                                default=float))
    res2 = drv.replay_recovery(pdir, out / "run2", kinds=("ff",))
    assert not res2["kinds"]["ff"]["entries"][f"ff_{stem}"]["ok"]


def test_q2_recovery_index_rejects_bad_views(tmp_path) -> None:
    pdir = tmp_path / "cf"
    pdir.mkdir(parents=True)
    good = {"format": drv.RECOVERY_FORMAT, "stems": ["cf_a"],
            "records": [{"stem": "cf_a"}]}
    (pdir / "cf_index.json").write_text(json.dumps(good))
    assert drv.load_recovery_index(pdir)["stems"] == ["cf_a"]
    for mutate in ("empty_stems", "empty_records", "mismatch", "dup",
                   "bad_format"):
        d = tmp_path / f"cf_{mutate}"
        d.mkdir(parents=True)
        doc = json.loads(json.dumps(good))
        if mutate == "empty_stems":
            doc["stems"] = []
        elif mutate == "empty_records":
            doc["records"] = []
        elif mutate == "mismatch":
            doc["records"] = [{"stem": "cf_a"}, {"stem": "cf_b"}]
            doc["stems"] = ["cf_a", "cf_b"]
            doc["records"] = [{"stem": "cf_a"}]
        elif mutate == "dup":
            doc["stems"] = ["cf_a", "cf_a"]
            doc["records"] = [{"stem": "cf_a"}, {"stem": "cf_a"}]
        elif mutate == "bad_format":
            doc["format"] = drv.NORM_FORMAT
        (d / "cf_index.json").write_text(json.dumps(doc))
        with pytest.raises(ValueError):
            drv.load_recovery_index(d)
    with pytest.raises(FileNotFoundError):
        drv.load_recovery_index(tmp_path / "nowhere")


# ---------------------------------------------------------------------------
# Q4: one namespace level; the historical double prefix decodes explicitly
# ---------------------------------------------------------------------------


def test_q4_v1_double_prefix_explicit_decode() -> None:
    v1 = {f"rows__plus_c2_norm__rows__plus_c2_norm__{k}":
          np.arange(3, dtype=float) for k in ("val", "lo", "hi")}
    decoded, marker = drv.decode_norm_v1_arrays(v1)
    assert marker == "v1_double_prefix"
    assert set(decoded) == {f"rows__plus_c2_norm__{k}"
                            for k in ("val", "lo", "hi")}
    # a mixed layout (single-prefix key in a v1 bundle) is corruption
    with pytest.raises(ValueError, match="mixed layout"):
        drv.decode_norm_v1_arrays({**v1, "rows__current__val":
                                   np.zeros(3)})
    with pytest.raises(ValueError, match="no v1"):
        drv.decode_norm_v1_arrays({"rows__plus_c2_norm__val":
                                   np.zeros(3)})


def test_q4_writer_refuses_recovery_format_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError):
        drv._write_norm_products(tmp_path / "x", "s",
                                 {"format": drv.RECOVERY_FORMAT},
                                 {})          # cf_ writer is the v1 layout


# ---------------------------------------------------------------------------
# Q3: every QP one-to-one with its own metadata and matrix
# ---------------------------------------------------------------------------


def _trace_pointers(pipe, attempt):
    pipe.solver.trace = {"attempts": [attempt]}
    pipe.solver.qp_trace = attempt["qp"]
    pipe.solver.qp_trace_arrays = attempt["qp_arrays"]


def test_q3_cut_refine_qp_metadata_one_to_one() -> None:
    """Initial QP + cut re-solves: each record carries branch/outer-iter/
    cut_round, the ACTUAL row count of ITS matrix, the layout flag, and
    per-QP row-name arrays whose length equals that matrix's rows; the QP
    ids are monotonic and the JSON count equals the arrays count."""
    pipe, z, box = _setup(S1)
    # every axis over the bound: any box-feasible corner has norm > B, so
    # the cut loop MUST add directions and re-solve
    d = np.array([[0.0022, 0.0022, 0.0022]] * 3)
    ctx = _ctx_with_prev(pipe, z, box, d)
    pipe.ev.forward(z, box)
    attempt = {"iterations": [], "qp": [], "qp_arrays": []}
    _trace_pointers(pipe, attempt)
    (P, q, A, lo, hi), _n = pipe.solver.build_qp(
        z, ctx, trust=8.0, slack=True, mu=360.0)
    x0, status = pipe.solver.solve_qp(
        P, q, A, lo, hi, settings=S1.solver,
        qp_identity={"branch": "restoration", "outer_iter": 3,
                     "cut_round": 0})
    assert x0 is not None, status
    x1, _st, n_extra, stats = pipe.solver._c2_cut_refine(
        x0, status, z, ctx, 8.0, True, 360.0, S1.solver,
        branch="restoration", outer_iter=3)
    assert stats["resolves"] == n_extra and n_extra >= 1

    recs, arrs = attempt["qp"], attempt["qp_arrays"]
    assert len(recs) == len(arrs) == 1 + n_extra
    for i, (rec, arr) in enumerate(zip(recs, arrs)):
        assert rec["qp_id"] == i
        assert rec["branch"] == "restoration" and rec["outer_iter"] == 3
        assert rec["cut_round"] == (0 if i == 0 else i)
        assert rec["rows_n"] == arr["A"].shape[0]
        assert rec["cols_n"] == arr["A"].shape[1]
        assert rec["layout_in_arrays"] is True
        assert len(arr["row_names"]) == rec["rows_n"]
        assert len(arr["col_names"]) == rec["cols_n"]
        assert rec["n_s"] > 0 and rec["slack_rows"] is not None
        if rec["cut_round"] >= 1:
            assert rec["n_cuts"] >= rec["cut_round"]
    # the dynamic layout grew against the FIRST assembly (the pre-Q3 code
    # stamped the initial row count on every record)
    assert len({r["rows_n"] for r in recs}) > 1 or recs[0]["n_cuts"] > 0
    # slack copies + nonneg rows are NAMED
    names = list(arrs[-1]["row_names"])
    assert any(n.endswith("::slack_lo") for n in names)
    assert any(n.endswith("::nonneg") for n in names)


def test_q3_solve_frame_metadata_and_counts() -> None:
    """A full solve_frame under S1: every QP has a complete identity, the
    per-QP row names match their matrix, slack maps ride only restoration
    QPs, the JSON count equals the arrays count equals FrameResult.qp_calls
    (incl. ``z_prev=None`` conditional pre layout), and the attempt-level
    single row-name cache is GONE."""
    pipe, z, box = _setup(S1)
    ctx = _ctx_with_prev(pipe, z, box, LEGAL_NORM_D)
    z0 = z.copy()
    z0[20] += 0.12                       # displaced start: real iterations
    pipe.solver.trace = {"attempts": []}    # solve_frame owns the attempt
    res = pipe.solver.solve_frame(z0.copy(), ctx)
    assert res.qp_calls > 0
    attempt = pipe.solver.trace["attempts"][-1]
    assert "qp_row_names" not in attempt
    assert len(attempt["qp"]) == len(attempt["qp_arrays"]) == res.qp_calls
    saw_restoration = False
    for i, (rec, arr) in enumerate(zip(attempt["qp"], attempt["qp_arrays"])):
        assert rec["qp_id"] == i
        assert rec["branch"] in ("tight", "restoration")
        assert rec["outer_iter"] is not None and rec["cut_round"] >= 0
        assert rec["rows_n"] == arr["A"].shape[0]
        assert rec["layout_in_arrays"] is True
        assert len(arr["row_names"]) == rec["rows_n"]
        assert rec["z_prev"] is False        # the conditional pre layout
        if rec["branch"] == "restoration":
            saw_restoration = True
            assert rec["n_s"] > 0 and rec["slack_rows"] is not None
        else:
            assert rec["n_s"] == 0 and rec["slack_rows"] is None
        if rec["cut_round"] >= 1:
            assert rec["n_cuts"] >= 1
    # iterations carry the cut accounting of THIS iteration (Q3 moved it
    # from the post-hoc last-record patch into the iteration record); the
    # one legitimate exception is the converged early exit, which returns
    # before any QP of that iteration runs
    for it in attempt["iterations"]:
        assert "c2_cut_stats" in it or it.get("early_exit"), it
        if it.get("restoration"):
            assert "restoration_slack" in it
    assert saw_restoration or attempt["qp"][0]["branch"] == "tight"


@pytest.mark.parametrize("prof_name", ["N2", "N3"])
def test_q3_trace_off_keeps_candidate_bitwise(prof_name) -> None:
    """Trace OFF under the old numeric profiles: same candidate, same
    counts, same exit evidence -- the Q3 metadata work is read-only."""
    prof = get_profile(prof_name)
    outs = []
    for trace_on in (False, True):
        pipe, z, box = _setup(prof)
        ctx = _ctx_with_prev(pipe, z, box, LEGAL_NORM_D)
        z0 = z.copy()
        z0[20] += 0.12
        if trace_on:
            attempt = {"iterations": [], "qp": [], "qp_arrays": []}
            _trace_pointers(pipe, attempt)
        else:
            pipe.solver.trace = None
        res = pipe.solver.solve_frame(z0.copy(), ctx)
        outs.append(res)
    a, b = outs
    assert a.accepted == b.accepted and a.status == b.status
    assert a.term_reason == b.term_reason
    assert a.iterations == b.iterations and a.qp_calls == b.qp_calls
    assert np.array_equal(a.z, b.z)
    assert a.last_qp_status == b.last_qp_status
    assert a.merit == b.merit
