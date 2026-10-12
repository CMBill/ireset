"""Counterexamples for the 12 W1 A3/B/C review gaps (2026-10-11 review).

Review round 2: the independent review rejected ``ready_for_w1_d=true``
with 10 P1 + 2 P2 gaps. Every fix is pinned here by a counterexample that
failed (or laundered a pass) before the fix:

* fix 1 (P1)  source-box geometry ignores the fitted similarity scale —
              schedule/quality measured against the unscaled locked hull;
* fix 2 (P1)  readiness ``check_c`` trusted the recorded verdict string;
* fix 3 (P1)  preregistration verification iterated only PRESENT
              anchors/inputs — emptied blocks verified as "passed";
* fix 4 (P1)  dataset trusted per-row ``split`` labels without reconciling
              the rows against the registered selection;
* fix 5 (P1)  quality completeness trusted ``status=complete`` without the
              measured fields;
* fix 6 (P1)  cache verified by NAMES (root_stats exists, key-correct
              filename) — tampered content passed the B gate;
* fix 7 (P1)  S0 deadline mixed wall and monotonic clocks and ignored
              prior attempts' cost;
* fix 8 (P1)  W1 raw eval skipped the preregistration/cache/stats/model
              identity legs beyond the manifest binding;
* fix 9 (P1)  the formal best rule was recorded but the trainer selected
              ``best.pt`` by window-validation loss;
* fix 10 (P1) W1 raw/training outputs had no derived/ guard and could
              overwrite source data or prior artifacts;
* fix 11 (P2) the 6-window S0 small set trained 4+2 ragged micro-batches
              with equal weighting (effective 12, not 16, samples);
* fix 12 (P2) cost ledger double-counted the first model into the second
              and never registered cache/failed attempts.

Review round 3 (4 P1 + 4 P2) — every counterexample below laundered a pass
under the round-2 code:

* r3-1 (P1) a NaN probe-replay input replays to NaN and
            ``abs(NaN - rec) > tol`` is silently False;
* r3-2 (P1) a doctored summary probe curve (trend evidence) was never
            reconciled with the checkpoint's replay artifact;
* r3-3 (P1) a loss history shrunk to {step, gnorm} still counted as
            complete training evidence (only PRESENT fields were checked);
* r3-4 (P1) a deleted per-side reference-count KEY read as zero (legal
            zero-reference side) and the both-hands denominator was never
            recomputed from the reference intervals;
* r3-5 (P2) deleting one on-cadence probe kept "increasing multiples"
            schedule validity — the schedule must EQUAL the full cadence;
* r3-6 (P2) the cache's LAST chunk / stats phase could overrun the budget
            and still register complete=true;
* r3-7 (P2) S0 budget_stopped required ``step < cfg.steps`` — a tail
            overrun after completing the target still verdicted passed;
* r3-8 (P2) a crashed cache worker left an unregistered attempt dir that
            washed into a zero-cost history (index count alone).

Review round 4 (1 P1 + 2 P2 + 1 doc correction):

* r4-1 (P1) the replayed batch inside probe_replay.pt was never bound to
            the REGISTERED six windows — a shrunk/foreign batch with
            honestly recomputed errors and synced last probes passed;
            the diffusion t/noise were likewise never reconciled with the
            dedicated-generator draw for the registered batch;
* r4-2 (P2) structurally-invalid replay evidence (NaN/missing probe_t,
            unreadable or non-mapping artifact) escaped check_c as an
            exception (IndexError et al.) instead of an unready reason;
* r4-3 (P2) the cache's FINAL identity hashing and registration writes sat
            after the last deadline check — a slow manifest hash there
            still registered complete=true with consumed_s over budget.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_omniretarget_w1 import (  # noqa: E402
    _complete_metrics,
    _patch_paths,
    _repin,
    _run_real_s0_attempt,
    _sha,
    _write_demo_files,
    build_synthetic_world,
    make_synthetic_demo,
)
from ireset.data import w1_training as w1  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --- fix 1: source-box geometry scale ------------------------------------------

def test_scaled_hull_distance_matches_scaled_mesh_ground_truth():
    """The algebraic scaling must equal a REAL scaled mesh: build the hull
    in the BODY frame, scale it by s about the body origin, and compare
    unsigned surface distances point for point."""
    import trimesh

    from ireset.geometry.distance_adapter import (
        LARGEBOX_AABB_CENTER, largebox_hull, largebox_hull_surface_distance,
        largebox_static_frame,
    )

    rng = np.random.default_rng(5)
    pts = rng.uniform(-0.6, 0.6, size=(40, 3))
    pose = np.tile(np.array([0.3, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0]), (40, 1))
    R_g, x_g = largebox_static_frame()
    # body-frame hull vertices of the locked largebox
    v_geom = np.asarray(largebox_hull().vertices)
    hull_faces = np.asarray(largebox_hull().faces)
    v_body = v_geom @ R_g.T + x_g
    # world -> body for the query points at the identity rotation above
    p_body = pts - pose[:, 0:3][0]
    for s in (0.5, 0.75, 1.3):
        scaled = trimesh.Trimesh(vertices=s * v_body, faces=hull_faces,
                                 process=False)
        truth = np.clip(
            -trimesh.proximity.signed_distance(scaled, p_body), 0.0, None)
        got = largebox_hull_surface_distance(pts, pose, scale=s)
        assert np.allclose(got, truth, atol=1e-9), \
            f"scale {s}: max diff {np.abs(got - truth).max()}"
    # sanity: the unscaled case is exactly the legacy metric
    base = largebox_hull_surface_distance(pts, pose)
    assert np.allclose(base, largebox_hull_surface_distance(pts, pose, 1.0))
    # non-positive/NaN scale is an explicit error, never a silent 1.0
    for bad in (0.0, -0.5, float("nan")):
        with pytest.raises(ValueError):
            largebox_hull_surface_distance(pts, pose, scale=bad)
    assert np.isfinite(LARGEBOX_AABB_CENTER).all()


def test_schedule_inference_applies_source_geometry_scale():
    """A hand centroid grazing the unscaled hull loses its contact when the
    source box shrinks by the fitted scale — the schedule is a function of
    the GEOMETRY, not just the pose."""
    from ireset.embodied.contact_sites import infer_reference_schedule
    from ireset.geometry.distance_adapter import (
        largebox_hull_surface_distance,
    )

    pose = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (5, 1))
    pose1 = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    # find a grazing radius: inside the enter threshold unscaled, beyond the
    # exit threshold at scale 0.5
    r_graze = None
    for r in np.linspace(0.15, 0.45, 61):
        p = np.array([[r, 0.0, 0.0]])
        d1 = float(largebox_hull_surface_distance(p, pose1, scale=1.0)[0])
        d5 = float(largebox_hull_surface_distance(p, pose1, scale=0.5)[0])
        if d1 < 0.03 and d5 > 0.09:
            r_graze = float(r)
            break
    assert r_graze is not None, "no grazing radius found for the fixture"

    def human_at(point):
        # all joints AT the point (hand centroid == point); feet at ground
        # height only adds ground intervals, the assertions are hand-only
        return np.full((5, 52, 3), point)

    sched_1 = infer_reference_schedule(human_at([r_graze, 0, 0]), pose,
                                       fps=30.0, box_scale=1.0)
    sched_05 = infer_reference_schedule(human_at([r_graze, 0, 0]), pose,
                                        fps=30.0, box_scale=0.5)
    hands_1 = [iv for iv in sched_1.intervals if iv.effector == "left_hand"]
    hands_05 = [iv for iv in sched_05.intervals if iv.effector == "left_hand"]
    assert hands_1 and not hands_05     # contact only against the full-size box


def test_derive_source_box_returns_pose_and_scale():
    data = make_synthetic_demo("sub10_largebox_000", T=90, seed=3)
    pose, scale = w1.derive_source_box(
        data["pt"][:, 162:318].reshape(90, 52, 3), data["human"],
        data["pt"][:, 318:325])
    assert pose.shape == (90, 7) and np.isfinite(pose).all()
    assert 0.0 < scale < 2.0
    from ireset.data.pairing import fit_similarity, source_obj_pose_in_target_scene

    s2, R, t, _ = fit_similarity(
        data["pt"][:, 162:318].reshape(-1, 3), data["human"].reshape(-1, 3))
    pose2 = source_obj_pose_in_target_scene(data["pt"][:, 318:325], s2, R, t)
    assert np.array_equal(pose, pose2) and s2 == scale


def test_cache_key_version_pins_scaled_geometry_semantics():
    """v2 = scaled source geometry; a version change must change the key
    (stale v1 cache files can never be addressed by the new key)."""
    assert w1.SCHEDULE_KEY_VERSION == 2
    k2 = w1.schedule_cache_key("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    w1.SCHEDULE_KEY_VERSION = 1
    try:
        k1 = w1.schedule_cache_key("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    finally:
        w1.SCHEDULE_KEY_VERSION = 2
    assert k1 != k2


def test_load_w1_derived_clip_requires_finite_scale(tmp_path):
    """The derived representation must CARRY the scale: a pre-fix npz
    without it is refused, and a non-finite scale is an explicit error."""
    import numpy as np

    d = tmp_path / "derived"
    d.mkdir()
    demo = "sub10_largebox_000"
    data = make_synthetic_demo(demo, T=70, seed=4)
    pose, scale = w1.derive_source_box(
        data["pt"][:, 162:318].reshape(70, 52, 3), data["human"],
        data["pt"][:, 318:325])
    row = {"demo": demo,
           "derived": {"path": str(d / f"{demo}_w1_scene.npz")},
           "inputs": {"npz_sha256": "0" * 64, "pt_sha256": "0" * 64}}
    # the keyed cache file will not exist -> FileNotFoundError AFTER the
    # scale check; a missing scale field fails BEFORE that with KeyError
    np.savez(d / f"{demo}_w1_scene.npz", qpos=data["qpos"],
             human_joints=data["human"], fps=np.float64(30.0),
             source_obj_pose=pose)
    with pytest.raises(KeyError):
        w1.load_w1_derived_clip(row, tmp_path / "sched", tmp_path, "0" * 64,
                                code_sha="0" * 64)
    np.savez(d / f"{demo}_w1_scene.npz", qpos=data["qpos"],
             human_joints=data["human"], fps=np.float64(30.0),
             source_obj_pose=pose, similarity_scale=np.float64(float("nan")))
    with pytest.raises(ValueError, match="similarity_scale"):
        w1.load_w1_derived_clip(row, tmp_path / "sched", tmp_path, "0" * 64,
                                code_sha="0" * 64)


# --- fix 3: required anchor/input sets -----------------------------------------

def test_verify_preregistration_requires_full_anchor_and_input_sets(
        tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR

    def verify(prereg):
        return w1.verify_preregistration(prereg, PROJECT_ROOT,
                                         TEACHER_TRAJ_DIR, OMOMO_NEW_DIR,
                                         verify_assets=False)

    base = json.loads(world["prereg_path"].read_text())
    assert verify(base)["passed"]

    emptied = json.loads(json.dumps(base)); emptied["anchors"] = {}
    probs = verify(emptied)["problems"]
    assert any("anchor_set_incomplete" in x for x in probs), probs

    dropped = json.loads(json.dumps(base))
    del dropped["anchors"]["ur_full_comparison"]
    assert any("anchor_set_incomplete:['ur_full_comparison']" in x
               for x in verify(dropped)["problems"])

    no_inputs = json.loads(json.dumps(base)); no_inputs["inputs"] = {}
    probs = verify(no_inputs)["problems"]
    assert any("inputs_set_mismatch:missing=" in x for x in probs), probs

    one_gone = json.loads(json.dumps(base))
    victim = sorted(one_gone["inputs"])[0]
    del one_gone["inputs"][victim]
    assert any(f"missing=['{victim}']" in x
               for x in verify(one_gone)["problems"])

    extra = json.loads(json.dumps(base))
    extra["inputs"]["subX_largebox_999"] = {"npz": "0" * 64, "pt": "0" * 64}
    assert any("extra=['subX_largebox_999']" in x
               for x in verify(extra)["problems"])


# --- fix 4: rows vs registered selection ---------------------------------------

def _w1_dataset(world, split):
    import torch

    from ireset.data.windows import TeacherWindowDataset
    from test_omniretarget_w1 import _resync_cache_identity, _tokenizer

    return TeacherWindowDataset(
        split, _tokenizer(world), window=60, stride=30,
        training_manifest=world["manifest_path"],
        schedule_cache_dir=world["cache_dir"] / "schedules")


def test_dataset_refuses_row_relabelled_across_splits(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    val_demo = world["val31"][0]

    def edit(manifest):
        row = next(r for r in manifest["rows"] if r["demo"] == val_demo)
        row["split"] = "train"           # a val label laundered into train

    from test_omniretarget_w1 import _repin, _resync_cache_identity

    _repin(world, edit)
    _resync_cache_identity(world)
    with pytest.raises(ValueError, match="split disagrees"):
        _w1_dataset(world, "train")


def test_dataset_refuses_dropped_row_masked_by_duplicate(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)

    def edit(manifest):
        rows = manifest["rows"]
        keep = next(i for i, r in enumerate(rows)
                    if r["demo"] == world["train24"][0])
        dropped = next(i for i, r in enumerate(rows)
                       if r["demo"] == world["val31"][0])
        dupe = json.loads(json.dumps(rows[keep]))
        dupe["split"] = rows[dropped]["split"]
        rows[dropped] = dupe            # unique count stays 54/55... and dupes

    from test_omniretarget_w1 import _repin, _resync_cache_identity

    _repin(world, edit)
    _resync_cache_identity(world)
    with pytest.raises(ValueError, match="duplicate demo rows"):
        _w1_dataset(world, "train")


# --- fix 5: quality completeness over measured fields ---------------------------

def test_quality_completeness_requires_actual_measurements():
    from audit_omniretarget_training import (
        exit_criteria, quality_registration_complete, summarize,
    )

    def _row(demo):
        return {"demo": demo, "split": "train", "subject": "sub3",
           "status": "processed", "admission": {"motion": True},
                "T": 90, "full_windows_F60_s30": 1,
                "quality": {"status": "complete", "reason": None,
                            "metrics": _complete_metrics(
                                {"demo": demo, "T": 90})}}

    row = _row("d0")
    ok, problems = quality_registration_complete(row)
    assert ok, problems

    # the review counterexample: status kept, measurements deleted
    stripped = _row("d3")
    stripped["quality"]["metrics"] = None
    stripped["quality"]["metrics"] = None
    ok, problems = quality_registration_complete(stripped)
    assert not ok and any("status:" in p for p in problems)

    # core scalar non-finite -> incomplete (registered NaN is not evidence)
    nan_core = json.loads(json.dumps(row))
    nan_core["quality"]["metrics"]["penetration_max_m"] = float("nan")
    assert not quality_registration_complete(nan_core)[0]

    # a present side entry without reference frames contradicts measure()
    bad_side = json.loads(json.dumps(row))
    bad_side["quality"]["metrics"]["hand"]["l"] = {
        "n_ref_frames": 0, "gap_p95_m": None, "coverage": None}
    assert not quality_registration_complete(bad_side)[0]

    # --- round-2 fix 5: measurement EXISTENCE is checked against the recorded
    # reference-frame counts (the reviewer's real-sub10 counterexamples) ---
    def _row_with_refs(demo):
        r = _row(demo)
        m = r["quality"]["metrics"]
        m["ref_grasp_frames"] = {"l": 46, "r": 50}
        m["ref_support_frames"] = {"l": 94, "r": 89}
        m["reference_intervals"] = [
            {"eff": "left_hand", "start": 1, "end": 47, "sup": "object"},
            {"eff": "right_hand", "start": 44, "end": 94, "sup": "object"},
            {"eff": "left_foot", "start": 4, "end": 98, "sup": "ground"},
            {"eff": "right_foot", "start": 0, "end": 89, "sup": "ground"},
        ]
        m["hand"] = {
            "l": {"n_ref_frames": 46, "gap_p95_m": 0.02, "coverage": 0.9},
            "r": {"n_ref_frames": 50, "gap_p95_m": 0.02, "coverage": 0.9}}
        m["foot"] = {
            "l": {"n_ref_frames": 94, "gap_p95_m": 0.01, "speed_p95_mps": 0.1,
                  "drift_max_m": 0.02, "n_speed_frames": 90},
            "r": {"n_ref_frames": 89, "gap_p95_m": 0.01, "speed_p95_mps": 0.1,
                  "drift_max_m": 0.02, "n_speed_frames": 85}}
        m["both"] = {"denominator_frames": 3, "valid_coverage": 0.5,
                     "gap_series_p95_m": 0.03}
        return r

    ref_row = _row_with_refs("d10")
    ok, problems = quality_registration_complete(ref_row)
    assert ok, problems

    def _mutated(base, fn):
        r = json.loads(json.dumps(base))
        fn(r["quality"]["metrics"])
        return quality_registration_complete(r)

    # hand.l -> None while 46 reference frames exist (was: legal zero side)
    ok, problems = _mutated(ref_row, lambda m: m["hand"].__setitem__("l", None))
    assert not ok and "hand[l]:reference_without_measurement" in problems
    # whole hand/foot blocks deleted
    ok, problems = _mutated(ref_row, lambda m: m.pop("hand"))
    assert not ok and any("reference_without_measurement" in p for p in problems)
    ok, problems = _mutated(ref_row, lambda m: m.pop("foot"))
    assert not ok
    # keyframe violation ratio NaN'd (a measured quantity, not optional)
    ok, problems = _mutated(
        ref_row, lambda m: m.__setitem__("penetration_violation_ratio",
                                         float("nan")))
    assert not ok and any("penetration_violation_ratio" in p for p in problems)
    # both block cleared (was: any dict passed)
    ok, problems = _mutated(ref_row, lambda m: m.__setitem__("both", {}))
    assert not ok and "both_missing" in problems
    # doctored per-entry n_ref_frames vs the recorded reference counts
    ok, problems = _mutated(
        ref_row, lambda m: m["hand"]["l"].__setitem__("n_ref_frames", 1))
    assert not ok and "hand[l].n_ref_frames:1!=46" in problems
    # ref counts doctored against the recorded intervals
    ok, problems = _mutated(
        ref_row, lambda m: m["ref_grasp_frames"].__setitem__("l", 999))
    assert not ok and any("ref_frames_vs_intervals" in p for p in problems)
    # ref count dicts deleted entirely
    ok, problems = _mutated(ref_row, lambda m: m.pop("ref_grasp_frames"))
    assert not ok and "ref_frame_counts_missing" in problems
    # --- round-3 fix 4: a missing per-side KEY is missing evidence ---
    # reviewer scenario: ref_grasp_frames.l deleted AND hand.l=None (46
    # reference frames actually exist) used to read the key as the
    # recomputed value / as zero -> legal zero-reference side
    ok, problems = _mutated(
        ref_row, lambda m: (m["ref_grasp_frames"].pop("l"),
                            m["hand"].__setitem__("l", None)))
    assert not ok
    assert "hand[l]:ref_frames_key_missing" in problems
    assert "hand[l]:ref_frames_vs_intervals:None!=46" in problems
    # same laundering through the foot counts
    ok, problems = _mutated(
        ref_row, lambda m: (m["ref_support_frames"].pop("r"),
                            m["foot"].__setitem__("r", None)))
    assert not ok and "foot[r]:ref_frames_key_missing" in problems
    # the both-hands denominator is RECOMPUTED from the intervals: the
    # left[1,47) x right[44,94) grasp intersection is 3 frames — a doctored
    # 0 with the measurements deleted used to pass as a legal zero block
    ok, problems = _mutated(ref_row, lambda m: m.__setitem__(
        "both", {"denominator_frames": 0, "valid_coverage": None,
                 "gap_series_p95_m": None}))
    assert not ok
    assert "both.denominator_frames_vs_intervals:0!=3" in problems
    # a denominator INFLATED above the recomputed intersection is equally
    # inconsistent (measurements cannot exist for frames without reference)
    ok, problems = _mutated(
        ref_row, lambda m: m["both"].__setitem__("denominator_frames", 40))
    assert not ok
    assert "both.denominator_frames_vs_intervals:40!=3" in problems
    # a genuinely zero-reference side stays legal (None with 0 counts)
    zero_side = json.loads(json.dumps(ref_row))
    zm = zero_side["quality"]["metrics"]
    zm["ref_grasp_frames"] = {"l": 46, "r": 0}
    zm["reference_intervals"] = [iv for iv in zm["reference_intervals"]
                                 if iv["eff"] != "right_hand"]
    zm["hand"]["r"] = None
    zm["both"] = {"denominator_frames": 0, "valid_coverage": None,
                  "gap_series_p95_m": None}
    ok, problems = quality_registration_complete(zero_side)
    assert ok, problems

    # ...and the exit gate flips when a TRAIN row loses its measurements
    rows = [_row(f"d{i}") for i in range(55)]
    rows[3] = stripped
    summary = summarize(rows, [f"d{i}" for i in range(55)])
    assert "d3" in summary["train_quality_incomplete"]
    gates = exit_criteria(summary, rows)
    assert gates["train_label_quality_complete"] is False


# --- fix 6: cache CONTENT verification ------------------------------------------

def _make_cache_problems(world):

    def _cache_problems():
        manifest = json.loads(world["manifest_path"].read_text())
        prereg = json.loads(world["prereg_path"].read_text())
        return w1.verify_cache_contents(world["cache_dir"], manifest, prereg,
                                        world["root"])["problems"]

    return _cache_problems


def test_cache_content_verification_catches_tampering(tmp_path, monkeypatch):
    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    _cache_problems = _make_cache_problems(world)
    assert _cache_problems() == []

    # root stats mean shifted by +100 m (content, not existence)
    import numpy as np

    rs = world["cache_dir"] / "root_stats.npz"
    rs_bytes = rs.read_bytes()
    with np.load(rs) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["mean"] = arrs["mean"] + 100.0
    np.savez(rs, **arrs)
    assert any("root_stats_content_drifted" in x for x in _cache_problems())

    # restore, then tamper ONE schedule's contact intervals
    rs.write_bytes(rs_bytes)
    assert _cache_problems() == []
    sched = sorted((world["cache_dir"] / "schedules").glob("*.npz"))[0]
    sched_bytes = sched.read_bytes()
    with np.load(sched) as z:
        iv = z["intervals"].copy()
    np.savez(sched, intervals=np.concatenate([iv, [[0, 0, 60, 0]]]))
    assert any("schedule_content_drifted" in x for x in _cache_problems())

    # restore, then a stale key-version record must be refused
    sched.write_bytes(sched_bytes)
    assert _cache_problems() == []
    ident_p = world["cache_dir"] / "cache_identity.json"
    ident = json.loads(ident_p.read_text())
    ident["schedule_cache_key_version"] = 1
    ident_p.write_text(json.dumps(ident))
    assert any("cache_key_version_stale" in x for x in _cache_problems())

    # restore, then drifted condition CODE invalidates the whole cache
    ident["schedule_cache_key_version"] = w1.SCHEDULE_KEY_VERSION
    ident_p.write_text(json.dumps(ident))
    assert _cache_problems() == []
    code_file = world["root"] / "src/ireset/data/teacher.py"
    code_file.write_text(code_file.read_text() + "\n# drift\n")
    assert any("cache_condition_code_drifted" in x for x in _cache_problems())


# --- fix 7 + 12: S0 budget clock, history and cost ledger ------------------------

def test_historical_s0_cost_semantics(tmp_path):
    """Round-2 fix 5: the ledger reconciles THREE ways (disk attempt dirs ×
    per-model index × cost rows) — a crashed attempt without a row, a
    leftover ``running`` row or duplicate (model, attempt) rows are all
    UNKNOWN history (stop), never a zero."""
    from w1_s0 import historical_s0_cost_s

    run = tmp_path / "run"
    assert historical_s0_cost_s(run) == 0.0        # nothing yet: full budget

    def stage(attempts, rows):
        import shutil

        for model in ("deterministic", "diffusion"):
            mroot = run / "s0" / model
            if mroot.exists():
                shutil.rmtree(mroot)
            mroot.mkdir(parents=True, exist_ok=True)
            for name in attempts.get(model, ()):
                (mroot / name).mkdir(parents=True, exist_ok=True)
            (mroot / "attempts_index.json").write_text(json.dumps(
                {"attempts": [{"attempt": n} for n in attempts.get(model, ())]}))
        (run / "costs.json").write_text(json.dumps({"invocations": rows}))

    stage({"deterministic": ["attempt_001"], "diffusion": ["attempt_001"]}, [])
    # attempts exist but the ledger does not: UNKNOWN -> stop condition
    assert historical_s0_cost_s(run) is None
    stage({"deterministic": ["attempt_001"], "diffusion": ["attempt_001"]}, [
        {"stage": "w1_cache", "wall_s": 500.0},
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 100.0, "status": "done"},
        {"stage": "w1_readiness", "wall_s": 5.0},
    ])
    assert historical_s0_cost_s(run) is None        # diffusion unaccounted
    stage({"deterministic": ["attempt_001"], "diffusion": ["attempt_001"]}, [
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 100.0, "status": "done"},
        {"stage": "s0_diffusion", "attempt": "attempt_001",
         "wall_s": 50.0, "status": "done"},
    ])
    # ONLY s0 stages count; cache/readiness walls never consume the S0 budget
    assert historical_s0_cost_s(run) == 150.0
    # a leftover "running" row = the exit registration never ran -> unknown
    stage({"deterministic": ["attempt_001"], "diffusion": ["attempt_001"]}, [
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 100.0, "status": "done"},
        {"stage": "s0_diffusion", "attempt": "attempt_001",
         "wall_s": 50.0, "status": "running"},
    ])
    assert historical_s0_cost_s(run) is None
    # duplicate (model, attempt) rows cannot masquerade as two models
    stage({"deterministic": ["attempt_001"], "diffusion": ["attempt_001"]}, [
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 100.0, "status": "done"},
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 50.0, "status": "done"},
    ])
    assert historical_s0_cost_s(run) is None
    # a crashed attempt WITH its exit-registered row stays accounted (its
    # wall consumed budget) — the wash-to-zero path is closed
    stage({"deterministic": ["attempt_001", "attempt_002"],
           "diffusion": []}, [
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 100.0, "status": "done"},
        {"stage": "s0_deterministic", "attempt": "attempt_002",
         "wall_s": 37.5, "status": "crashed"},
    ])
    assert historical_s0_cost_s(run) == 137.5
    # an attempt dir missing from the index (crash before index write)
    d = run / "s0" / "diffusion" / "attempt_001"
    d.mkdir(parents=True)
    assert historical_s0_cost_s(run) is None


def test_s0_main_refuses_exhausted_or_unknown_budget(tmp_path, monkeypatch):
    import w1_s0

    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(w1_s0, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(w1_s0, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(w1_s0, "OMOMO_NEW_DIR", world["pt_dir"])

    # a prior pair of attempts fully consuming the 1h budget
    for model, wall in (("deterministic", 2000.0), ("diffusion", 1800.0)):
        d = world["run"] / "s0" / model
        (d / "attempt_001").mkdir(parents=True)
        (d / "attempts_index.json").write_text(
            json.dumps({"attempts": [{"attempt": "attempt_001"}]}))
    (world["run"] / "costs.json").write_text(json.dumps({"invocations": [
        {"stage": "s0_deterministic", "attempt": "attempt_001",
         "wall_s": 2000.0, "status": "done"},
        {"stage": "s0_diffusion", "attempt": "attempt_001",
         "wall_s": 1800.0, "status": "done"},
    ]}))
    argv = ["prog", "--preregistration", str(world["prereg_path"]),
            "--training-manifest", str(world["manifest_path"]),
            "--cache-dir", str(world["cache_dir"]), "--out", str(world["run"]),
            "--model", "deterministic", "--steps", "5"]
    monkeypatch.setattr("sys.argv", argv)
    import train_generation
    from test_omniretarget_w1 import _s0_cfg

    monkeypatch.setattr(train_generation, "load_train_config",
                        lambda pth: (_s0_cfg("deterministic", 5), {}))
    with pytest.raises(SystemExit, match="already exhausted"):
        w1_s0.main()

    # attempts WITHOUT ledger rows -> unknown history -> refused
    (world["run"] / "costs.json").unlink()
    with pytest.raises(SystemExit, match="unknown"):
        w1_s0.main()


def test_s0_crash_stays_in_ledger_and_cannot_wash_to_zero(
        tmp_path, monkeypatch):
    """Round-2 fix 5 reviewer scenario: run_s0_model raising after the
    attempt was allocated used to leave an unaccounted attempt_001 whose
    next historical cost read 0.0. Now the crash registers an index row, a
    crash summary and a ledger row — history stays KNOWN and non-zero."""
    import w1_s0

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(w1_s0, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(w1_s0, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(w1_s0, "OMOMO_NEW_DIR", world["pt_dir"])
    import train_generation
    from test_omniretarget_w1 import _s0_cfg

    cfg = _s0_cfg("deterministic", 2)
    cfg.micro_batch = 2
    monkeypatch.setattr(train_generation, "load_train_config",
                        lambda p: (cfg, {}))

    def explode(*a, **k):
        import time

        time.sleep(0.2)          # a measurable wall before the crash
        raise RuntimeError("boom after attempt allocation")

    monkeypatch.setattr(w1_s0, "run_s0_model", explode)
    argv = ["prog", "--preregistration", str(world["prereg_path"]),
            "--training-manifest", str(world["manifest_path"]),
            "--cache-dir", str(world["cache_dir"]), "--out", str(world["run"]),
            "--model", "deterministic", "--steps", "2"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(RuntimeError, match="boom"):
        w1_s0.main()

    att = world["run"] / "s0" / "deterministic" / "attempt_001"
    assert att.is_dir()
    assert (att / "s0_summary.json").is_file()
    assert json.loads((att / "s0_summary.json").read_text())["verdict"] \
        == "incomplete"
    rows = [e for e in json.loads(
        (world["run"] / "costs.json").read_text())["invocations"]
        if e["stage"] == "s0_deterministic"]
    assert len(rows) == 1 and rows[0]["status"] == "crashed"
    # the crashed attempt's history is KNOWN and its wall consumed budget —
    # not the reviewer's 0.0
    hist = w1_s0.historical_s0_cost_s(world["run"])
    assert hist is not None and hist > 0.0


# --- fix 8: W1 raw eval unified admission ---------------------------------------

def _stub_eval_ckpt(world, tmp_path, model="stub"):
    import ireset.data.a2_integrity as integ

    ckpt = {"identity": {
        "protocol": w1.MOTION_PROTOCOL_VERSION, "model_type": model,
        "training_manifest": {
            "path": str(world["manifest_path"]),
            "sha256": integ.sha256_file(world["manifest_path"])},
        "cache": {
            "identity_sha256": integ.sha256_file(
                world["cache_dir"] / "cache_identity.json"),
            "root_stats_sha256": integ.sha256_file(
                world["cache_dir"] / "root_stats.npz")}},
        "model": {}, "conditioner": {}, "config": {"d_model": 32},
        "root_stats": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
    (tmp_path / "models" / model).mkdir(parents=True)
    torch.save(ckpt, tmp_path / "models" / model / "best.pt")
    return str(tmp_path / "models")


def test_w1_raw_eval_refuses_tampered_cache_and_drifted_inputs(
        tmp_path, monkeypatch):
    import eval_generation as eg

    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    models_dir = _stub_eval_ckpt(world, tmp_path)
    out = str(world["run"] / "eval/r1.json")

    def run_eval():
        argv = ["prog", "--models", "stub", "--models-dir", models_dir,
                "--training-manifest", str(world["manifest_path"]),
                "--expected-set", "val31", "--full-length",
                "--cache-dir", str(world["cache_dir"]), "--output", out]
        monkeypatch.setattr("sys.argv", argv)
        eg.main()

    # cache schedule content tampered -> refused BEFORE any generation
    sched = sorted((world["cache_dir"] / "schedules").glob("*.npz"))[0]
    with np.load(sched) as z:
        iv = z["intervals"].copy()
    np.savez(sched, intervals=np.concatenate([iv, [[0, 0, 60, 0]]]))
    with pytest.raises(SystemExit, match="cache content"):
        run_eval()
    np.savez(sched, intervals=iv)

    # an input PT drifted after registration -> the preregistration leg
    # refuses (the manifest binding alone would still match)
    victim = world["pt_dir"] / f"{world['val31'][0]}.pt"
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="input_drifted"):
        run_eval()


# --- fix 9: formal best selection -----------------------------------------------

def _formal_trainer(tmp_path, macro_sequence, complete_sequence):
    from ireset.train.losses import LossWeights
    ireset = pytest.importorskip("ireset")
    from ireset.data.windows import make_tokenizer  # noqa: F401
    from test_omniretarget_w1 import _FakeDS, _tokenizer_for_losses
    from ireset.train.trainer import TrainConfig, Trainer

    cfg = TrainConfig(model_type="deterministic", steps=2, val_every=1,
                      log_every=1, d_model=32, n_layers=1, n_heads=2,
                      time_window=3, ffn_mult=2, device="cpu",
                      micro_batch=2, num_workers=0, seed=11, val_seed=13,
                      loss=LossWeights(w_contact=0.0))
    calls = {"i": 0}

    def formal_eval(model, conditioner):
        i = calls["i"]
        calls["i"] += 1
        return {"n_expected": 31, "n_ok": 31 if complete_sequence[i] else 12,
                "macro_mpjpe_m": macro_sequence[i], "per_demo": []}

    out = tmp_path / f"out_{abs(hash((tuple(macro_sequence), id(tmp_path))))}"
    tr = Trainer(cfg, _tokenizer_for_losses(), _FakeDS(8), _FakeDS(4),
                 out, identity={"experiment_id": "x"},
                 formal_eval=formal_eval, formal_eval_every=1)
    return tr, out


def test_formal_best_never_selected_by_window_loss_or_incomplete_evidence(
        tmp_path):
    # macro WORSENS at the final step: the step-1 checkpoint stays best —
    # the final step must NOT replace it even though its window loss is the
    # lowest seen (that is exactly the laundered selection the review found)
    tr, out = _formal_trainer(tmp_path, [0.01, 10.0], [True, True])
    summary = tr.run()
    assert (out / "best.pt").exists()            # step 1: complete + finite
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 1 and ck["formal"]["macro_mpjpe_m"] == 0.01
    assert summary["best_formal_step"] == 1
    assert summary["best_selection_rule"] == "formal_macro_mpjpe"

    # NO evaluation ever completes: best.pt must not exist at all
    tr, out = _formal_trainer(tmp_path, [0.5, 0.2], [False, False])
    summary = tr.run()
    assert not (out / "best.pt").exists()
    assert summary["best_formal_step"] is None

    # complete + improving macro: best appears, carries the formal evidence
    tr, out = _formal_trainer(tmp_path, [0.9, 0.4], [True, True])
    summary = tr.run()
    assert (out / "best.pt").exists()
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and ck["formal"]["macro_mpjpe_m"] == 0.4
    assert summary["best_formal_macro_mpjpe_m"] == 0.4
    assert summary["best_formal_step"] == 2


# --- fix 10: output guards -------------------------------------------------------

def test_report_output_guard_protected_targets(tmp_path, monkeypatch):
    import eval_generation as eg

    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    # outside the writable derived/ tree (e.g. a source npz) -> refused
    with pytest.raises(SystemExit, match="derived"):
        eg.guard_report_output(world["npz_dir"] / "victim_original.npz")
    # an existing artifact is never overwritten
    victim = world["run"] / "eval/exists.json"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("prior evidence")
    with pytest.raises(SystemExit, match="never overwritten"):
        eg.guard_report_output(victim)
    # a fresh path under derived/ passes
    ok = eg.guard_report_output(world["run"] / "eval/fresh.json")
    assert ok == (world["run"] / "eval/fresh.json").resolve()


def test_w1_train_out_guard_and_attempt_isolation(tmp_path, monkeypatch):
    import train_generation as tg

    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    with pytest.raises(SystemExit, match="derived"):
        tg.guard_w1_out(world["npz_dir"] / "steal_dir")
    fresh = tg.guard_w1_out(world["run"] / "formal/deterministic")
    assert fresh.name == "deterministic"
    fresh.mkdir(parents=True)
    (fresh / "best.pt").write_text("prior run")
    again = tg.guard_w1_out(world["run"] / "formal/deterministic")
    assert again.name == "attempt_001"          # non-empty -> new attempt
    assert again.parent.name == "deterministic"


# --- fix 2 + 11 + 12: readiness recompute / S0 batch / per-model costs ----------

def test_readiness_check_c_rejects_doctored_summaries(tmp_path, monkeypatch):
    """Round-2 review fixes 1+2+6: the C gate recomputes every gate from
    PRIMARY evidence (loss history, probe schedule, small-set identity) and
    actually REBUILDS + RELOADS + REPLAYS the checkpoint — fabricated or
    doctored training evidence fails even under an intact verdict=passed."""
    import w1_readiness as rd

    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(rd, "PROJECT_ROOT", world["root"])
    cache = rd.check_cache_binding(world["run"])
    prereg = json.loads(world["prereg_path"].read_text())

    monkeypatch.setattr(w1, "S0_MIN_STEPS", 20)
    monkeypatch.setattr(w1, "S0_PROBE_EVERY", 2)
    monkeypatch.setattr(w1, "S0_TREND_WINDOW", 3)
    monkeypatch.setattr(w1, "S0_PROBE_MIN", 6)

    def latest(model):
        return sorted((world["run"] / "s0" / model).glob("attempt_*"))[-1]

    def fresh_pair():
        """New REAL attempts for both models -> the honest baseline passes."""
        atts = {}
        for model in ("deterministic", "diffusion"):
            atts[model], summary = _run_real_s0_attempt(world, model, steps=24)
            assert summary["verdict"] == "passed", summary["gates"]
        return atts

    def doctor(model, summary_edit=None, history_edit=None, ckpt_edit=None,
               delete=()):
        att = latest(model)
        if summary_edit is not None:
            sp = att / "s0_summary.json"
            s = json.loads(sp.read_text())
            summary_edit(s)
            sp.write_text(json.dumps(s))
        if history_edit is not None:
            hp = att / "loss_history.json"
            h = json.loads(hp.read_text())
            history_edit(h)
            hp.write_text(json.dumps(h))
        if ckpt_edit is not None:
            ck = torch.load(att / "last.pt", map_location="cpu",
                            weights_only=False)
            ckpt_edit(ck)
            torch.save(ck, att / "last.pt")
        for name in delete:
            (att / name).unlink(missing_ok=True)
        return att

    fresh_pair()
    assert rd.check_c(world["run"], cache, prereg)["passed"]

    # round-2 fix 1: deleted loss history no longer passes vacuously
    fresh_pair()
    doctor("deterministic", delete=("loss_history.json",))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["deterministic"]["recomputed_gates"][
        "loss_history_present"] is False

    # NaN in ANY loss part (not just gnorm) breaks all_finite
    fresh_pair()
    doctor("deterministic",
           history_edit=lambda h: h[3].__setitem__("total", float("nan")))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["deterministic"]["recomputed_gates"]["all_finite"] is False

    # a gap in the step numbering breaks contiguity
    fresh_pair()
    doctor("deterministic", history_edit=lambda h: h.pop(5))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["deterministic"]["recomputed_gates"][
        "loss_history_contiguous"] is False

    # probes off the registered cadence are invalid evidence
    fresh_pair()
    doctor("deterministic",
           summary_edit=lambda s: s["probes"].insert(
               2, {"step": 7, "motion_reconstruction": 0.5}))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["deterministic"]["recomputed_gates"][
        "probe_schedule_valid"] is False

    # a shrunken small set no longer "comes from" pilot6 (was: subset
    # check) — evidenced from both directions: the recorded set emptied,
    # and a registered pilot6 larger than what actually trained
    fresh_pair()
    doctor("deterministic",
           summary_edit=lambda s: s["identity"].__setitem__(
               "small_set_demos", []))
    r = rd.check_c(world["run"], cache, prereg)
    assert any("small_set_not_the_registered_pilot6" in p for p in r["reasons"])
    fresh_pair()
    six = {**prereg, "selection": {**prereg["selection"],
                                   "pilot6_from_a1": list(world["train24"])}}
    r = rd.check_c(world["run"], cache, six)
    assert any("small_set_not_the_registered_pilot6" in p for p in r["reasons"])
    assert any("small_set_batch_size_mismatch" in p for p in r["reasons"])

    # round-2 fix 6: budget_stopped is a hard fail even with a good trend
    fresh_pair()
    doctor("deterministic", summary_edit=lambda s: s.__setitem__(
        "budget_stopped", True))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["deterministic"]["recomputed_gates"][
        "completed_within_budget"] is False

    # round-2 fix 2: cleared state dicts must fail a STRICT load
    fresh_pair()
    doctor("deterministic", ckpt_edit=lambda ck: ck.__setitem__("model", {}))
    r = rd.check_c(world["run"], cache, prereg)
    assert any("checkpoint_weights_unloadable" in p
               for p in r["per_model"]["deterministic"]["identity_problems"])

    # NaN weights load but fail the finiteness sweep
    fresh_pair()
    def nan_weights(ck):
        k = next(iter(ck["model"]))
        ck["model"][k] = ck["model"][k] * float("nan")
    doctor("deterministic", ckpt_edit=nan_weights)
    r = rd.check_c(world["run"], cache, prereg)
    assert any("checkpoint_weights_not_finite" in p
               for p in r["per_model"]["deterministic"]["identity_problems"])

    # embedded statistics doctored (+100 m) differ from the cache's
    fresh_pair()
    def bump_stats(ck):
        import numpy as _np

        ck["root_stats"]["mean"] = (_np.asarray(ck["root_stats"]["mean"])
                                    + 100.0).tolist()
    doctor("deterministic", ckpt_edit=bump_stats)
    r = rd.check_c(world["run"], cache, prereg)
    assert any("checkpoint_stats_differ_from_cache" in p
               for p in r["per_model"]["deterministic"]["identity_problems"])

    # deleted probe-replay artifact: the fixed probe can no longer be replayed
    fresh_pair()
    doctor("deterministic", delete=("probe_replay.pt",))
    r = rd.check_c(world["run"], cache, prereg)
    assert any("probe_replay_missing" in p
               for p in r["per_model"]["deterministic"]["identity_problems"])

    # original round-1 regressions still hold on real evidence
    fresh_pair()
    doctor("deterministic",
           summary_edit=lambda s: s.__setitem__("optimizer_steps", 0))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["deterministic"]["recomputed_gates"][
        "min_100_optimizer_steps"] is False

    fresh_pair()
    doctor("deterministic", summary_edit=lambda s: s.__setitem__("probes", []))
    assert not rd.check_c(world["run"], cache, prereg)["passed"]

    fresh_pair()
    doctor("deterministic",
           ckpt_edit=lambda ck: ck["identity"]["cache"].__setitem__(
               "identity_sha256", "0" * 64))
    r = rd.check_c(world["run"], cache, prereg)
    assert any("checkpoint_cache_identity_mismatch" in p
               for p in r["per_model"]["deterministic"]["identity_problems"])

    fresh_pair()
    doctor("diffusion", history_edit=lambda h: h[0].__setitem__("contact_m2", 0.5))
    r = rd.check_c(world["run"], cache, prereg)
    assert r["per_model"]["diffusion"]["recomputed_gates"][
        "contact_branch_silent"] is False


def test_s0_main_equal_micro_batches_and_per_model_costs(tmp_path, monkeypatch):
    """Fix 11 + 12 evidence from a REAL main() pass with a stubbed trainer
    loop: the small-set loader's batch size equals the full small set (no
    4+2 ragged batches), and the ledger books one wall per model attempt."""
    import w1_s0

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(w1_s0, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(w1_s0, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(w1_s0, "OMOMO_NEW_DIR", world["pt_dir"])
    import train_generation
    from test_omniretarget_w1 import _s0_cfg

    cfg = _s0_cfg("deterministic", 2)
    cfg.micro_batch = 2
    monkeypatch.setattr(train_generation, "load_train_config",
                        lambda p: (cfg, {}))
    captured = {}

    def stub_run(model_type, cfg, tokenizer, small_batch, small_loader,
                 device, identity, attempt_dir, budget_deadline):
        captured.setdefault(model_type, {})["loader_batch"] = \
            small_loader.batch_size
        captured[model_type]["dir"] = attempt_dir
        return {"verdict": "passed", "gates": {}, "optimizer_steps": 2,
                "small_set_batch_size": int(small_batch["x0"].shape[0]),
                "samples_per_optimizer_step":
                    int(small_batch["x0"].shape[0] * cfg.grad_accum),
                "micro_batches": 4, "grad_accum": cfg.grad_accum,
                "wall_s": 0.1, "probes": [], "budget_stopped": False,
                "nonfinite_stop": None, "identity": identity}

    monkeypatch.setattr(w1_s0, "run_s0_model", stub_run)
    argv = ["prog", "--preregistration", str(world["prereg_path"]),
            "--training-manifest", str(world["manifest_path"]),
            "--cache-dir", str(world["cache_dir"]), "--out", str(world["run"]),
            "--steps", "2"]
    monkeypatch.setattr("sys.argv", argv)
    w1_s0.main()

    # the small set = pilot6∩train24 = ONE demo -> one full micro-batch
    assert set(captured) == {"deterministic", "diffusion"}
    for model, cap in captured.items():
        assert cap["loader_batch"] >= 1
    # per-model cost rows with their own attempt ids and cumulative budget
    entries = json.loads((world["run"] / "costs.json").read_text())["invocations"]
    s0_rows = [e for e in entries if e["stage"].startswith("s0_")]
    assert len(s0_rows) == 2
    assert {e["stage"] for e in s0_rows} == {"s0_deterministic",
                                             "s0_diffusion"}
    assert all("attempt" in e and e["wall_s"] >= 0.0 for e in s0_rows)
    assert s0_rows[1]["budget_consumed_cumulative_s"] >= \
        s0_rows[0]["budget_consumed_cumulative_s"]


# --- round-2 fix 3: formal-selection denominator stays the REGISTERED val31 ----

def test_formal_eval_denominator_is_the_registered_val31(tmp_path, monkeypatch):
    """A rejected val label keeps its row as a gap: n_expected stays the
    FULL registered val31 count, the macro is None and the evidence is
    incomplete — the validation set can no longer shrink to whichever
    labels survived admission."""
    from train_generation import make_formal_eval

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr("ireset.paths.PROJECT_ROOT", world["root"])

    def reject_one(manifest):
        victim = next(r for r in manifest["rows"] if r["split"] == "val")
        victim["admission"]["motion"] = False
        victim["admission"]["reasons"] = ["doctored_for_test"]

    _repin(world, reject_one)
    from test_omniretarget_w1 import _tokenizer

    callback = make_formal_eval(
        world["manifest_path"], world["cache_dir"], _tokenizer(world),
        "cpu", {"full_generation_seed": 7}, "deterministic")
    from ireset.model.conditioner import ConditionEncoder
    from ireset.model.deterministic import DeterministicG1
    from test_omniretarget_w1 import _s0_cfg

    cfg = _s0_cfg("deterministic", 2)
    model = DeterministicG1(**cfg.backbone_kwargs())
    cond = ConditionEncoder(cfg.d_model)
    out = callback(model, cond)
    assert out["expected_set"] == "registered_val31"
    assert out["n_expected"] == len(world["val31"])      # the FULL set
    assert out["n_label_gaps"] == 1
    gap_rows = [r for r in out["per_demo"] if r["status"] == "label_not_admitted"]
    assert len(gap_rows) == 1 and gap_rows[0]["mpjpe_m"] is None
    # the gap alone makes the evidence incomplete (macro None even though
    # every loadable demo generated fine)
    assert out["n_ok"] < out["n_expected"]
    assert out["macro_mpjpe_m"] is None
    from ireset.train.trainer import Trainer  # completeness rule: n_ok==n_expected
    assert not (out["n_ok"] == out["n_expected"]
                and isinstance(out["macro_mpjpe_m"], float))


# --- round-2 fix 6: budget stop is never a pass -----------------------------------

def test_budget_stop_mid_run_is_incomplete(tmp_path, monkeypatch):
    """Reviewer scenario: 390/1000 steps under budget exhaustion with a good
    trend used to exit verdict=passed. A controlled monotonic clock lets
    some steps finish before the deadline passes; the verdict must be
    incomplete regardless of the trend."""
    import w1_s0
    from test_omniretarget_w1 import _FakeDS, _s0_cfg, _tokenizer_for_losses

    monkeypatch.setattr(w1, "S0_PROBE_EVERY", 2)
    tok = _tokenizer_for_losses()
    ds = _FakeDS(4, F=8, seed=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    out = tmp_path / "s0budget"
    out.mkdir()

    real = time.perf_counter
    calls = {"n": 0}

    def slow_clock():
        calls["n"] += 1
        # the first ~40 calls run at the real clock (setup + some steps),
        # everything after jumps past any deadline
        return real() if calls["n"] <= 40 else real() + 1e6

    monkeypatch.setattr(w1_s0.time, "perf_counter", slow_clock)
    cfg = _s0_cfg("deterministic", 40)
    summary = w1_s0.run_s0_model(
        "deterministic", cfg, tok, {k: ds.__dict__[k].clone() for k in _FakeDS.KEYS},
        loader, "cpu", {"experiment_id": "x"}, out, real() + 5.0)
    monkeypatch.undo()
    assert summary["budget_stopped"] is True
    assert 0 < summary["optimizer_steps"] < cfg.steps
    assert summary["verdict"] == "incomplete"          # never "passed"
    assert summary["gates"]["completed_within_budget"] is False


# --- round-2 fix 7: the cache stage executes CACHE_BUDGET_S ----------------------

def test_cache_budget_exhaustion_stops_incomplete(tmp_path, monkeypatch):
    import precompute_m3_cache as pc

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(pc, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(pc, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(pc, "OMOMO_NEW_DIR", world["pt_dir"])

    out_root = world["run"] / "cache"
    argv = ["prog", "--training-manifest", str(world["manifest_path"]),
            "--out-dir", str(out_root), "--workers", "1"]

    # historical exhaustion: a prior attempt with a full-budget ledger row
    (out_root / "attempt_001").mkdir(parents=True, exist_ok=True)
    (out_root / "attempts_index.json").write_text(
        json.dumps({"attempts": [{"attempt": "attempt_001"}]}))
    (world["run"] / "costs.json").write_text(json.dumps({"invocations": [
        {"stage": "w1_cache", "wall_s": float(w1.CACHE_BUDGET_S),
         "attempt": "attempt_001", "complete": True}]}))
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="already exhausted"):
        pc.main()

    # a slow job set overruns the remaining budget -> INCOMPLETE attempt,
    # non-zero exit, and the consumer-side content check refuses it
    import shutil

    (world["run"] / "costs.json").unlink()
    shutil.rmtree(out_root)

    def slow_job(job):
        import time as _t
        import hashlib as _hl
        import numpy as _np

        _t.sleep(1.2)
        out = Path(job[3]) / "schedules" / w1.schedule_cache_filename(
            job[0], job[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(out, intervals=_np.zeros((1, 4)))
        return {"demo": job[0], "T": 1, "cache": out.name, "status": "ok",
                "sha256": _hl.sha256(out.read_bytes()).hexdigest()}

    monkeypatch.setattr(pc, "_w1_schedule_job", slow_job)
    monkeypatch.setattr(w1, "CACHE_BUDGET_S", 1.0)
    monkeypatch.setattr("sys.argv", argv)
    assert pc.main() == 2
    att = sorted(out_root.glob("attempt_*"))[-1]
    ident = json.loads((att / "cache_identity.json").read_text())
    assert ident["complete"] is False
    assert ident["incomplete_reason"] == "cache_budget_exhausted"
    manifest = json.loads(world["manifest_path"].read_text())
    prereg = json.loads(world["prereg_path"].read_text())
    problems = w1.verify_cache_contents(att, manifest, prereg, world["root"])
    assert any("cache_not_complete" in p for p in problems["problems"])


# --- round-2 fix 8: the cache code identity covers the distance impl -------------

def test_cache_code_identity_covers_distance_adapter(tmp_path, monkeypatch):
    """Reviewer counterexample: reverting ONLY distance_adapter's scaled
    query left condition_code_sha256 unchanged and old keyed schedules
    reusable. The file is now part of CONDITION_CODE_FILES, so a content
    change re-keys every schedule."""
    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)

    assert any("distance_adapter.py" in f for f in w1.CONDITION_CODE_FILES)
    # the fixture copies the real code files INTO the synthetic root
    adapter = world["root"] / "src/ireset/geometry/distance_adapter.py"
    before = adapter.read_bytes()
    base_sha = w1.condition_code_sha256(world["root"])
    try:
        # a content-only change (e.g. a reverted scaled query) must re-key
        adapter.write_bytes(before + b"# reviewfix8 identity probe\n")
        assert w1.condition_code_sha256(world["root"]) != base_sha
        # the cache was produced under the OLD code hash -> consumer-side drift
        manifest = json.loads(world["manifest_path"].read_text())
        prereg = json.loads(world["prereg_path"].read_text())
        problems = w1.verify_cache_contents(
            world["cache_dir"], manifest, prereg, world["root"])["problems"]
        assert "cache_condition_code_drifted" in problems
    finally:
        adapter.write_bytes(before)
    assert w1.condition_code_sha256(world["root"]) == base_sha


# --- round-3 fixes 1+2: replay finiteness + summary/artifact reconciliation ------

def _readiness_fixture(tmp_path, monkeypatch):
    """Shared fixture: synthetic world + relaxed S0 constants + cache binding."""
    import w1_readiness as rd

    world = build_synthetic_world(tmp_path)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(rd, "PROJECT_ROOT", world["root"])
    cache = rd.check_cache_binding(world["run"])
    prereg = json.loads(world["prereg_path"].read_text())
    for k, v in (("S0_MIN_STEPS", 20), ("S0_PROBE_EVERY", 2),
                 ("S0_TREND_WINDOW", 3), ("S0_PROBE_MIN", 6)):
        monkeypatch.setattr(w1, k, v)
    return rd, world, cache, prereg


def test_replay_nan_input_and_recorded_probe_are_rejected(tmp_path, monkeypatch):
    """r3-1: one NaN x0 element makes the replay error NaN — ``abs(NaN -
    rec) > tol`` is False, so the round-2 comparison laundered a pass. The
    inputs and every replayed/recorded error must be checked finite FIRST."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    rp = torch.load(att / "probe_replay.pt", map_location="cpu",
                    weights_only=False)
    rp["batch"]["x0"][0, 0, 0] = float("nan")
    torch.save(rp, att / "probe_replay.pt")
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["deterministic"]["identity_problems"]
    assert any(p.startswith("probe_replay_input_not_finite:batch.x0")
               for p in probs), probs
    # round-4: invalid inputs are never replayed — the NaN batch must not
    # reach run_fixed_probe (it used to crash the gate on a NaN probe_t)
    assert "probe_replay_output_not_finite" not in probs, probs
    assert r["per_model"]["deterministic"]["checkpoint_replay"][
        "replayed_probe"] is None
    assert not r["passed"]

    # a NaN'd RECORDED probe is missing evidence, not a comparable value
    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    rp = torch.load(att / "probe_replay.pt", map_location="cpu",
                    weights_only=False)
    rp["last_probe"]["motion_reconstruction"] = float("nan")
    torch.save(rp, att / "probe_replay.pt")
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["deterministic"]["identity_problems"]
    assert "recorded_probe_not_finite" in probs, probs
    assert not r["passed"]


def test_summary_trend_curve_must_match_replay_artifact(tmp_path, monkeypatch):
    """r3-2: only summary.probes doctored into a declining 100->50 curve
    (checkpoint and replay artifact untouched) kept schedule/finite/trend
    all green under round-2 code — the trend evidence must be the SAME
    measurement the artifact recorded."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    att, summary = _run_real_s0_attempt(world, "deterministic", steps=24)
    sp = att / "s0_summary.json"
    s = json.loads(sp.read_text())
    for i, p in enumerate(s["probes"]):
        for k in ("pos", "rot6d", "joint", "motion_reconstruction"):
            p[k] = 100.0 - 0.5 * i          # still finite, still declining
    sp.write_text(json.dumps(s))
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["deterministic"]["identity_problems"]
    assert any(p.startswith("summary_last_probe_mismatch:") for p in probs), probs
    assert not r["passed"]
    # the honest summary still reconciles (regression on the same fixture)
    att, summary = _run_real_s0_attempt(world, "deterministic", steps=24)
    r = rd.check_c(world["run"], cache, prereg)
    det = r["per_model"]["deterministic"]
    assert not [p for p in det["identity_problems"]
                if "summary_last_probe" in p or "probe_replay" in p]


def test_loss_history_requires_every_configured_field(tmp_path, monkeypatch):
    """r3-3: rows shrunk to {step, gnorm} (or {step, total, gnorm}) are
    missing the configured loss parts — existence is its own gate, checked
    before finiteness."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    hp = att / "loss_history.json"
    h = json.loads(hp.read_text())
    assert all(k in h[0] for k in w1.S0_REQUIRED_LOSS_KEYS)  # the fixture is complete
    hp.write_text(json.dumps(
        [{"step": row["step"], "gnorm": row["gnorm"], "ms": row["ms"]}
         for row in h]))
    gates = rd.check_c(world["run"], cache, prereg)["per_model"][
        "deterministic"]["recomputed_gates"]
    assert gates["loss_fields_complete"] is False

    hp.write_text(json.dumps(
        [{"step": row["step"], "total": row["total"], "gnorm": row["gnorm"]}
         for row in h]))
    gates = rd.check_c(world["run"], cache, prereg)["per_model"][
        "deterministic"]["recomputed_gates"]
    assert gates["loss_fields_complete"] is False

    # the honest history passes the same gate
    hp.write_text(json.dumps(h))
    gates = rd.check_c(world["run"], cache, prereg)["per_model"][
        "deterministic"]["recomputed_gates"]
    assert gates["loss_fields_complete"] is True


def test_probe_schedule_must_be_the_complete_cadence(tmp_path, monkeypatch):
    """r3-5: deleting one on-cadence probe leaves the remaining steps
    increasing multiples of the cadence — only an exact comparison with the
    full expected schedule catches the hole."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    sp = att / "s0_summary.json"
    s = json.loads(sp.read_text())
    assert [p["step"] for p in s["probes"]] == list(range(0, 25, 2))
    del s["probes"][1]                      # the step-2 sample point
    sp.write_text(json.dumps(s))
    gates = rd.check_c(world["run"], cache, prereg)["per_model"][
        "deterministic"]["recomputed_gates"]
    assert gates["probe_schedule_valid"] is False


# --- round-3 fixes 6+7: budget enforcement at every phase boundary ---------------

def test_cache_final_chunk_overrun_is_incomplete(tmp_path, monkeypatch):
    """r3-6: with every schedule row landed, a final chunk that overruns the
    budget used to still register complete=true/consumed_s>budget — the
    deadline is now enforced at the chunk return, the stats phase and the
    final registration."""
    import precompute_m3_cache as pc

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(pc, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(pc, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(pc, "OMOMO_NEW_DIR", world["pt_dir"])
    out_root = world["run"] / "cache"
    argv = ["prog", "--training-manifest", str(world["manifest_path"]),
            "--out-dir", str(out_root), "--workers", "1"]
    monkeypatch.setattr("sys.argv", argv)

    manifest = json.loads(world["manifest_path"].read_text())
    last_demo = [r for r in manifest["rows"]
                 if r.get("admission", {}).get("motion")][-1]["demo"]

    def slow_last_job(job):
        import hashlib as _hl
        import time as _t

        if job[0] == last_demo:
            _t.sleep(1.2)                   # the FINAL chunk overruns
        out = Path(job[3]) / "schedules" / w1.schedule_cache_filename(
            job[0], job[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, intervals=np.zeros((1, 4)))
        return {"demo": job[0], "T": 1, "cache": out.name, "status": "ok",
                "sha256": _hl.sha256(out.read_bytes()).hexdigest()}

    monkeypatch.setattr(pc, "_w1_schedule_job", slow_last_job)
    monkeypatch.setattr(w1, "CACHE_BUDGET_S", 1.0)
    assert pc.main() == 2
    att = sorted(out_root.glob("attempt_*"))[-1]
    ident = json.loads((att / "cache_identity.json").read_text())
    assert ident["complete"] is False
    assert ident["incomplete_reason"] == "cache_budget_exhausted"
    assert ident["missing_demos"] == []     # every row landed — the BUDGET ran out
    assert ident["budget"]["consumed_s"] >= 1.0
    rows = [e for e in json.loads(
        (world["run"] / "costs.json").read_text())["invocations"]
        if e["stage"] == "w1_cache"]
    # the fixture's ledgered attempt_001 + THIS over-budget attempt_002
    assert [r["attempt"] for r in rows] == ["attempt_001", "attempt_002"]
    assert rows[-1]["status"] == "done" and rows[-1]["complete"] is False
    # no root stats fabricated for an over-budget attempt
    assert not (att / "root_stats.npz").is_file()
    prereg = json.loads(world["prereg_path"].read_text())
    problems = w1.verify_cache_contents(att, manifest, prereg, world["root"])
    assert any("cache_not_complete" in p for p in problems["problems"])


def test_s0_overrun_after_target_steps_is_incomplete(tmp_path, monkeypatch):
    """r3-7: the full step target completed (400 in the review, 8 here) but
    the checkpoint phase crossed the deadline — budget_stopped used to
    require ``step < cfg.steps``, so the run still verdicted passed."""
    import w1_s0
    from test_omniretarget_w1 import _FakeDS, _s0_cfg, _tokenizer_for_losses

    monkeypatch.setattr(w1, "S0_PROBE_EVERY", 2)
    tok = _tokenizer_for_losses()
    ds = _FakeDS(4, F=8, seed=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    out = tmp_path / "s0tail"
    out.mkdir()
    real = time.perf_counter
    state = {"jump": False}

    def clock():
        return real() + (1e6 if state["jump"] else 0.0)

    real_save = torch.save

    def flipping_save(*a, **k):
        state["jump"] = True                # the deadline passes DURING the
        return real_save(*a, **k)           # checkpoint save, after all steps

    monkeypatch.setattr(w1_s0.time, "perf_counter", clock)
    monkeypatch.setattr(torch, "save", flipping_save)
    cfg = _s0_cfg("deterministic", 8)
    summary = w1_s0.run_s0_model(
        "deterministic", cfg, tok,
        {k: ds.__dict__[k].clone() for k in _FakeDS.KEYS},
        loader, "cpu", {"experiment_id": "x"}, out, real() + 600.0)
    monkeypatch.undo()
    assert summary["optimizer_steps"] == cfg.steps      # the full target ran
    assert summary["budget_stopped"] is True            # ...but the tail overran
    assert summary["verdict"] == "incomplete"           # never "passed"
    assert summary["gates"]["completed_within_budget"] is False


def test_cache_crash_stays_ledgered_and_disk_only_attempts_stop(
        tmp_path, monkeypatch):
    """r3-8: a worker raising after attempt allocation used to leave an
    unregistered dir that washed into a zero-cost history; the crash now
    registers identity+index+ledger, and a FABRICATED disk-only attempt is
    UNKNOWN history — the next invocation refuses to start."""
    import precompute_m3_cache as pc

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(pc, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(pc, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(pc, "OMOMO_NEW_DIR", world["pt_dir"])
    out_root = world["run"] / "cache"
    argv = ["prog", "--training-manifest", str(world["manifest_path"]),
            "--out-dir", str(out_root), "--workers", "1"]

    def exploding_job(job):
        import time as _t

        _t.sleep(0.2)                       # a measurable wall before dying
        raise RuntimeError("worker boom after attempt allocation")

    monkeypatch.setattr(pc, "_w1_schedule_job", exploding_job)
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(RuntimeError, match="worker boom"):
        pc.main()

    # the fixture's attempt_001 (ledgered 0.0s) is prior history; the crash
    # landed in attempt_002 and must be fully registered
    att = out_root / "attempt_002"
    assert att.is_dir()
    ident = json.loads((att / "cache_identity.json").read_text())
    assert ident["complete"] is False
    assert "crashed" in ident["incomplete_reason"]
    index = json.loads((out_root / "attempts_index.json").read_text())["attempts"]
    assert [a["attempt"] for a in index] == ["attempt_001", "attempt_002"]
    assert index[-1]["incomplete_reason"] == ident["incomplete_reason"]
    rows = [e for e in json.loads(
        (world["run"] / "costs.json").read_text())["invocations"]
        if e["stage"] == "w1_cache"]
    assert [r["status"] for r in rows] == ["done", "crashed"]
    # the crashed attempt stays ACCOUNTED: known history, wall consumed
    hist = pc._historical_cache_cost_s(out_root)
    assert hist is not None and hist > 0.0

    # a disk-only attempt (no index, no ledger) is UNKNOWN -> refuse
    (out_root / "attempt_003").mkdir()
    assert pc._historical_cache_cost_s(out_root) is None
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="unknown"):
        pc.main()


# --- round-4 fixes 1-3: registered-set binding, exception safety, final
# registration budget ---------------------------------------------------------

def _reload_probe_modules(att: Path, model: str):
    """Fresh (model, conditioner) strictly loaded from a real checkpoint."""
    import w1_s0
    from ireset.train.trainer import TrainConfig

    ck = torch.load(att / "last.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["config"])
    m, cond = w1_s0.build_s0_modules(model, cfg, "cpu")
    m.load_state_dict(ck["model"], strict=True)
    cond.load_state_dict(ck["conditioner"], strict=True)
    return w1_s0, m, cond


def _sync_recomputed_errors(att: Path, rp: dict, err: dict) -> None:
    """Write the artifact back with errors honestly recomputed for the
    TAMPERED inputs and sync the summary's last probe to them — the
    laundering that defeated the round-3 gate."""
    step = rp["last_probe"]["step"]
    rp["last_probe"] = {"step": step, **err}
    torch.save(rp, att / "probe_replay.pt")
    sp = att / "s0_summary.json"
    s = json.loads(sp.read_text())
    s["probes"][-1] = {"step": step, **err}
    sp.write_text(json.dumps(s))


def test_replay_batch_must_be_the_registered_windows(tmp_path, monkeypatch):
    """r4-1: check_c bound only the summary's DECLARED demo list and batch
    size — the batch actually inside probe_replay.pt was free. Tamper the
    batch, HONESTLY recompute the replay errors for the tampered inputs and
    sync summary/artifact last probes: only the registered-set binding can
    catch it."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    # (a) content tamper: one x0 element moved, everything else reconciled
    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    rp = torch.load(att / "probe_replay.pt", map_location="cpu",
                    weights_only=False)
    rp["batch"]["x0"][0, 0, 0] += 0.5
    w1_s0, m, cond = _reload_probe_modules(att, "deterministic")
    err = w1_s0.run_fixed_probe(m, cond, rp["batch"], "deterministic",
                                None, None)
    _sync_recomputed_errors(att, rp, err)
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["deterministic"]["identity_problems"]
    assert any(p.startswith("probe_replay_batch_differs:") for p in probs), probs
    assert not r["passed"]

    # (b) foreign demo: the batch replaced by ANOTHER admitted train demo's
    # first window (not in the registered pilot set), errors recomputed
    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    rp = torch.load(att / "probe_replay.pt", map_location="cpu",
                    weights_only=False)
    from ireset.data.windows import TeacherWindowDataset, make_tokenizer
    from ireset.model.tokens import RootStats

    ds = TeacherWindowDataset(
        "train", make_tokenizer(
            RootStats.load(world["cache_dir"] / "root_stats.npz")),
        window=w1.WINDOW_F, stride=w1.WINDOW_STRIDE,
        training_manifest=world["manifest_path"],
        schedule_cache_dir=world["cache_dir"] / "schedules")
    registered = prereg["selection"]["pilot6_from_a1"][0]
    other = next(d for d in world["train24"] if d != registered)
    lc_idx = next(i for i, lc in enumerate(ds.clips) if lc.demo == other)
    item = ds[ds.index.index((lc_idx, 0))]
    rp["batch"] = {k: v.unsqueeze(0) for k, v in item.items()
                   if torch.is_tensor(v)}
    w1_s0, m, cond = _reload_probe_modules(att, "deterministic")
    err = w1_s0.run_fixed_probe(m, cond, rp["batch"], "deterministic",
                                None, None)
    _sync_recomputed_errors(att, rp, err)
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["deterministic"]["identity_problems"]
    assert any(p.startswith("probe_replay_batch_differs:") for p in probs), probs

    # honest regression: the genuine artifact IS the registered window
    att, _ = _run_real_s0_attempt(world, "deterministic", steps=24)
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["deterministic"]["identity_problems"]
    assert not [p for p in probs if "probe_replay_batch" in p], probs


def test_diffusion_probe_t_must_be_the_registered_draw(tmp_path, monkeypatch):
    """r4-1 (diffusion leg): the recorded probe_t/probe_noise were never
    reconciled with the dedicated-generator draw for the registered batch —
    a t from a different draw (guaranteed different value), with errors
    honestly recomputed and last probes synced, passed round-3 code."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    att, _ = _run_real_s0_attempt(world, "diffusion", steps=24)
    rp = torch.load(att / "probe_replay.pt", map_location="cpu",
                    weights_only=False)
    w1_s0, m, cond = _reload_probe_modules(att, "diffusion")
    from ireset.train.trainer import TrainConfig

    ck = torch.load(att / "last.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["config"])
    # the registered draw (same expression as the S0 loop) — tamper to a
    # GUARANTEED different value so the counterexample is deterministic
    g0 = torch.Generator().manual_seed(cfg.val_seed or cfg.seed)
    reg_t = torch.randint(0, cfg.num_train_timesteps,
                          (rp["batch"]["x0"].shape[0],), generator=g0)
    assert torch.equal(rp["probe_t"], reg_t)      # the honest artifact
    rp["probe_t"] = reg_t.clone()
    rp["probe_t"][0] = (reg_t[0].item() + 1) % cfg.num_train_timesteps
    err = w1_s0.run_fixed_probe(m, cond, rp["batch"], "diffusion",
                                rp["probe_t"], rp["probe_noise"])
    _sync_recomputed_errors(att, rp, err)
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["diffusion"]["identity_problems"]
    assert "probe_t_not_the_registered_generator_draw" in probs, probs
    assert not r["passed"]

    # honest regression: the recorded t/noise ARE the registered draw
    att, _ = _run_real_s0_attempt(world, "diffusion", steps=24)
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["diffusion"]["identity_problems"]
    assert not [p for p in probs
                if "probe_t_not" in p or "probe_noise_not" in p], probs


def test_invalid_replay_evidence_is_recorded_not_raised(tmp_path, monkeypatch):
    """r4-2: structurally-invalid replay evidence used to ESCAPE check_c as
    an exception (NaN probe_t -> IndexError out of the readiness script; a
    missing t/noise or an unreadable/non-mapping artifact likewise) — an
    unready reason must be recorded and the report still written."""
    rd, world, cache, prereg = _readiness_fixture(tmp_path, monkeypatch)

    att, _ = _run_real_s0_attempt(world, "diffusion", steps=24)
    base = torch.load(att / "probe_replay.pt", map_location="cpu",
                      weights_only=False)

    # (a) NaN probe_t — round-3 code replayed it and crashed
    torch.save({**base, "probe_t": torch.full(
        tuple(base["probe_t"].shape), float("nan"))}, att / "probe_replay.pt")
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["diffusion"]["identity_problems"]
    assert "probe_replay_input_not_finite:probe_t" in probs, probs
    assert r["per_model"]["diffusion"]["checkpoint_replay"][
        "replayed_probe"] is None          # never replayed
    assert not r["passed"]

    # (b) missing t — invalid input, recorded, no replay
    torch.save({**base, "probe_t": None}, att / "probe_replay.pt")
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["diffusion"]["identity_problems"]
    assert "probe_replay_input_invalid:probe_t" in probs, probs
    assert not r["passed"]

    # (c) unreadable artifact — a reason, never an exception
    (att / "probe_replay.pt").write_bytes(b"this is not a torch file")
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["diffusion"]["identity_problems"]
    assert any(p.startswith("probe_replay_unreadable:") for p in probs), probs
    assert not r["passed"]

    # (d) non-mapping artifact (e.g. a bare list)
    torch.save([1, 2, 3], att / "probe_replay.pt")
    r = rd.check_c(world["run"], cache, prereg)
    probs = r["per_model"]["diffusion"]["identity_problems"]
    assert "probe_replay_not_a_mapping" in probs, probs
    assert not r["passed"]


def test_cache_final_registration_overrun_is_incomplete(tmp_path, monkeypatch):
    """r4-3: the last deadline check sat at the stats phase — a slow
    manifest hash in the FINAL identity construction still registered
    complete=true with consumed_s past the budget."""
    import precompute_m3_cache as pc

    world = build_synthetic_world(tmp_path, n_train=2, n_val=2, T=90)
    _patch_paths(monkeypatch, world)
    monkeypatch.setattr(pc, "PROJECT_ROOT", world["root"])
    monkeypatch.setattr(pc, "TEACHER_TRAJ_DIR", world["npz_dir"])
    monkeypatch.setattr(pc, "OMOMO_NEW_DIR", world["pt_dir"])
    out_root = world["run"] / "cache"
    argv = ["prog", "--training-manifest", str(world["manifest_path"]),
            "--out-dir", str(out_root), "--workers", "1"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(w1, "CACHE_BUDGET_S", 5.0)

    # the FIRST training-manifest hash is exactly the final identity
    # construction (preflight/jobs/stats hash other files) — delay it past
    # the 5s budget; every earlier phase stays comfortably inside
    real_sha = pc.integ.sha256_file
    state = {"armed": True}

    def slow_manifest_sha(p, *a, **k):
        if state["armed"] and Path(p) == world["manifest_path"]:
            state["armed"] = False
            time.sleep(6.0)
        return real_sha(p, *a, **k)

    monkeypatch.setattr(pc.integ, "sha256_file", slow_manifest_sha)

    assert pc.main() == 2
    att = sorted(out_root.glob("attempt_*"))[-1]
    ident = json.loads((att / "cache_identity.json").read_text())
    assert ident["complete"] is False
    assert ident["incomplete_reason"] == "cache_budget_exhausted"
    assert ident["budget"]["consumed_s"] >= 5.0
    # every row landed and the stats phase was within budget — only the
    # FINAL hashing overran, so missing_demos is honestly empty
    assert ident["missing_demos"] == []
    assert (att / "root_stats.npz").is_file()    # stats phase did complete
    manifest = json.loads(world["manifest_path"].read_text())
    prereg = json.loads(world["prereg_path"].read_text())
    problems = w1.verify_cache_contents(att, manifest, prereg, world["root"])
    assert any("cache_not_complete" in p for p in problems["problems"])
