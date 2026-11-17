"""R-phase metadata admission + real-data entry counterexamples (plan v0.4).

The fixture below rebuilds the SOURCE structure (preregistration, training
manifest, keyed cache, derived scenes) around six train + two val demos so
every admission refusal, drift, deadline and billing rule is exercised
without touching the real SOURCE.  Real-window materialization is tested in
fixture mode on CPU only (clearly labelled, no production equivalence).
"""

import importlib.util
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.data import w1_training as w1  # noqa: E402
from ireset.data.teacher import SMPLH_INDEX  # noqa: E402
from ireset.data.w1_training import canonical_list_sha256, select_train24  # noqa: E402
from ireset.eval import w1_smallset_fit_r_admission as radm  # noqa: E402
from ireset.eval.w1_smallset_fit_diagnostic import (  # noqa: E402
    FitDiagnosticError, FitPlan, StageBudget, expected_call_matrix,
    materialize_for_fit, MetadataOnlyReader)
from ireset.eval.w1_real_training_update_audit import runtime_identity_facts  # noqa: E402
from ireset.paths import PROJECT_ROOT  # noqa: E402

PILOT6 = ["sub10_largebox_097", "sub12_largebox_082", "sub16_largebox_000",
          "sub3_largebox_027", "sub7_largebox_022", "sub8_largebox_035"]
VAL_DEMOS = ["sub10_largebox_002", "sub10_largebox_022"]
T_FIX = 120
FPS = 30.0
ANCHOR_KEYS = ("c2_final_decision", "ur_full_comparison", "ur_preregistration",
               "a1_frozen_lists", "a1_assets", "a1_pairing_source_boxes",
               "split_train", "split_val", "split_manifest")


def _sha(path: Path) -> str:
    return radm.sha256_file(path)


def _demo_scene(rng: np.random.Generator, demo: str) -> dict:
    """A published-layout derived scene within the real joint limits."""
    from ireset.data.teacher import _spec

    limits = _spec().joint_range
    t = np.arange(T_FIX)
    yaw = 0.05 * np.sin(t / 9.0)
    quat = np.stack([np.cos(yaw / 2), np.zeros(T_FIX), np.zeros(T_FIX),
                     np.sin(yaw / 2)], axis=1)
    pos = np.stack([0.6 + 0.02 * np.cos(t / 20), 0.2 + 0.02 * np.sin(t / 20),
                    0.79 + 0.005 * np.sin(t / 15)], axis=1)
    frac = 0.5 + 0.1 * np.sin(t[:, None] / 11.0 + np.arange(29)[None] * 0.7)
    joints = limits[:, 0][None] + frac * (limits[:, 1] - limits[:, 0])[None]
    qpos = np.concatenate([quat, pos, joints,
                           quat[:, :4] * 0.0 + np.array([0.0, 0.0, 0.7071, 0.7071]),
                           np.tile([0.4, 0.05, 0.35], (T_FIX, 1))], axis=1)
    hips = np.array([0.0, -0.10, 0.90])
    shoulders = np.array([0.0, 0.20, 1.35])
    human = np.tile(hips, (T_FIX, 52, 1))
    human += rng.normal(0, 0.005, human.shape)
    for idx in (SMPLH_INDEX["L_Hip"], SMPLH_INDEX["R_Hip"]):
        human[:, idx] = hips
    for idx in (SMPLH_INDEX["L_Shoulder"], SMPLH_INDEX["R_Shoulder"]):
        human[:, idx] = shoulders
    human[:, SMPLH_INDEX["L_Hip"]] += np.array([0.03, 0.0, 0.0])
    human[:, SMPLH_INDEX["R_Hip"]] += np.array([-0.03, 0.0, 0.0])
    box_pose = np.concatenate([pos, quat], axis=1)
    src_pose = box_pose.copy()
    src_pose[:, 2] += 0.06
    return {"qpos": qpos, "human_joints": human, "fps": np.float64(FPS),
            "source_obj_pose": src_pose, "target_obj_pose": box_pose,
            "similarity_scale": np.float64(0.7995)}


def _fixture_project(tmp: Path) -> dict:
    """Structurally identical SOURCE tree; code assets are symlinked views
    of the real project so condition-code identity is the REAL code."""
    proj = tmp / "proj"
    (proj / "derived/data/split").mkdir(parents=True)
    src_root = proj / "derived/experiments/fixture_w1_source"
    (src_root / "freeze").mkdir(parents=True)
    (src_root / "audit/attempt_001/derived").mkdir(parents=True)
    cache_dir = src_root / "cache/attempt_001"
    (cache_dir / "schedules").mkdir(parents=True)
    npz_dir = proj / "fixture_data/robot-object"
    pt_dir = proj / "fixture_data/omomo"
    npz_dir.mkdir(parents=True)
    pt_dir.mkdir(parents=True)
    for rel in ("src", "scripts", "tests"):
        (proj / rel).symlink_to(ROOT / rel, target_is_directory=True)
    for name in ("pyproject.toml", "uv.lock"):
        (proj / name).symlink_to(ROOT / name)
    (proj / "configs/train").mkdir(parents=True)
    shutil.copy(ROOT / "configs/train/omniretarget_motion.yaml",
                proj / "configs/train/omniretarget_motion.yaml")

    rng = np.random.default_rng(20261117)
    train = list(PILOT6)
    rows, inputs, window_files = [], {}, {}
    code_sha = w1.condition_code_sha256(ROOT)
    (proj / "derived/robots").mkdir(parents=True)
    asset_payload = proj / "derived/robots/fixture_asset.txt"
    asset_payload.write_text("fixture asset bytes\n")
    (proj / "derived/anchors").mkdir(parents=True, exist_ok=True)
    assets_path = proj / "derived/anchors/a1_assets.json"
    assets_path.write_text(json.dumps({
        "locked_g1": {str(asset_payload.relative_to(proj)):
                      {"sha256": radm.sha256_file(asset_payload),
                       "bytes": asset_payload.stat().st_size}}}))
    assets_sha = _sha(assets_path)
    schedules = {}
    for demo in train + VAL_DEMOS:
        npz_file = npz_dir / f"{demo}_original.npz"
        npz_file.write_bytes(f"teacher {demo}\n".encode())
        pt_file = pt_dir / f"{demo}.pt"
        pt_file.write_bytes(f"omomo {demo}\n".encode())
        npz_sha, pt_sha = _sha(npz_file), _sha(pt_file)
        inputs[demo] = {"npz": npz_sha, "pt": pt_sha}
        scene = _demo_scene(rng, demo)
        scene["meta_json"] = np.array(json.dumps({
            "demo": demo, "split": "train" if demo in train else "val",
            "protocol": w1.MOTION_PROTOCOL_VERSION, "npz_sha256": npz_sha,
            "pt_sha256": pt_sha, "similarity_scale": 0.7995,
            "layout": "published_verbatim"}))
        derived_path = src_root / f"audit/attempt_001/derived/{demo}_w1_scene.npz"
        np.savez(derived_path, **scene)
        key = w1.schedule_cache_key(npz_sha, pt_sha, assets_sha, code_sha)
        sched_file = cache_dir / "schedules" / w1.schedule_cache_filename(demo, key)
        np.savez(sched_file, intervals=np.array(
            [[0, 5, 55, 0], [1, 5, 55, 0], [2, 0, 40, 1], [3, 0, 40, 1]],
            dtype=np.int64))
        schedules[demo] = {"key": key[:16], "sha256": _sha(sched_file),
                           "full_windows": (T_FIX - 60) // 30 + 1}
        rows.append({
            "demo": demo, "subject": demo.split("_")[0], "variant": "original",
            "split": "train" if demo in train else "val",
            "list_ids": ["teacher_production_train24_v1"],
            "published_revision": w1.PUBLISHED_REVISION, "error": None,
            "admission": {"motion": True, "reasons": [],
                          "facts": {"T": T_FIX, "fps": FPS}},
            "review": None, "pairing": {"t_match": True},
            "scene_split": {"offset_pure_z": True}, "T": T_FIX, "fps": FPS,
            "duration_s": T_FIX / FPS,
            "full_windows_F60_s30": (T_FIX - 60) // 30 + 1,
            "short_trajectory": False,
            "inputs": {"npz_sha256": npz_sha, "pt_sha256": pt_sha},
            "derived": {"path": str(derived_path.relative_to(proj)),
                        "sha256": _sha(derived_path)},
            "quality": {"status": "complete", "metrics": {"penetration_max_m": 0.02}},
            "status": "done"})
    train_json = proj / "derived/data/split/train.json"
    train_json.write_text(json.dumps({"demos": train}))
    val_json = proj / "derived/data/split/val.json"
    val_json.write_text(json.dumps({"demos": VAL_DEMOS}))
    train24 = select_train24(train)
    assert sorted(train24) == sorted(train)

    anchors = {}
    anchor_payloads = {
        "split_train": train_json, "split_val": val_json,
        "a1_assets": assets_path}
    for key in ANCHOR_KEYS:
        p = anchor_payloads.get(key)
        if p is None:
            p = proj / f"derived/anchors/{key}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"anchor": key}))
        anchors[key] = {"path": str(p.relative_to(proj)), "sha256": _sha(p)}

    frozen_lists = {
        "selection_rules": {"teacher_pilot_train6_v1": "pilot"},
        "teacher_pilot_train6_v1": [{"demo": d} for d in PILOT6],
        "val_originals_v1": [{"demo": d} for d in VAL_DEMOS],
        "val6_pilot_v1": [{"demo": VAL_DEMOS[0]}]}
    lists_path = proj / "derived/anchors/a1_frozen_lists_full.json"
    lists_path.write_text(json.dumps(frozen_lists))
    anchors["a1_frozen_lists"] = {"path": str(lists_path.relative_to(proj)),
                                  "sha256": _sha(lists_path)}

    prereg = {
        "stage": "fixture W0 preregistration", "timestamp": "2026-11-17T00:00:00",
        "identity": {"experiment_id": w1.EXPERIMENT_ID,
                     "protocol": w1.MOTION_PROTOCOL_VERSION,
                     "training_purpose": w1.TRAINING_PURPOSE,
                     "source_type": w1.SOURCE_TYPE,
                     "published_revision": w1.PUBLISHED_REVISION,
                     "run_id": "fixture"},
        "repo_identity_at_registration": {"git_rev": "fixture"},
        "environment_measured": {},
        "plan": {"path": "derived/anchors/plan.json",
                 "sha256": anchors["c2_final_decision"]["sha256"]},
        "anchors": anchors,
        "entry_evidence": {"note": "fixture"},
        "selection": {
            "train24_list_id": "teacher_production_train24_v1",
            "train24_selection_rule": "per subject first 4 by hash",
            "train24": train24,
            "train24_canonical_sha256": canonical_list_sha256(train24),
            "val31_list_id": w1.VAL31_LIST_ID, "val31": VAL_DEMOS,
            "val6_list_id": w1.VAL6_LIST_ID, "val6": [VAL_DEMOS[0]],
            "pilot6_from_a1": PILOT6},
        "inputs": {demo: {"npz": rec["npz"], "pt": rec["pt"]}
                   for demo, rec in inputs.items()},
        "assets": {"a1_assets_path": str(assets_path.relative_to(proj)),
                   "a1_assets_sha256": assets_sha,
                   "closure_verification": {"fixture": True}},
        "labels_protocol": {"w_contact": 0.0},
        "budgets_and_stops": {}}
    plan_anchor = proj / "derived/anchors/plan.json"
    plan_anchor.write_text(json.dumps({"plan": "fixture"}))
    prereg["plan"]["sha256"] = _sha(plan_anchor)
    prereg_path = src_root / "freeze/preregistration.json"
    prereg_path.write_text(json.dumps(prereg, indent=1))

    stats_mean = np.array([0.0112, 0.0740, 0.7255])
    stats_std = np.array([0.3932, 0.4046, 0.0610])
    root_stats_path = cache_dir / "root_stats.npz"
    np.savez(root_stats_path, mean=stats_mean, std=stats_std)

    manifest = {
        "manifest_name": w1.MANIFEST_NAME, "protocol": w1.MOTION_PROTOCOL_VERSION,
        "experiment_id": w1.EXPERIMENT_ID, "training_purpose": w1.TRAINING_PURPOSE,
        "source_type": w1.SOURCE_TYPE,
        "published_revision": w1.PUBLISHED_REVISION, "attempt": "attempt_001",
        "timestamp": "2026-11-17T00:00:00",
        "preregistration": {"path": str(prereg_path.relative_to(proj)),
                            "sha256": _sha(prereg_path)},
        "identity": {"git_rev": "fixture", "rows_schema": "fixture"},
        "selection": {"train24": train24,
                      "train24_canonical_sha256": canonical_list_sha256(train24),
                      "val31": VAL_DEMOS, "val6": [VAL_DEMOS[0]]},
        "counts": {}, "exit_criteria": {}, "rows": rows, "total_wall_s": 1.0}
    manifest_path = src_root / "audit/attempt_001/training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))

    cache_identity = {
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "experiment_id": w1.EXPERIMENT_ID, "attempt": "attempt_001",
        "timestamp": "2026-11-17T00:00:00",
        "preregistration": {"path": str(prereg_path.relative_to(proj)),
                            "sha256": _sha(prereg_path)},
        "training_manifest": {"path": str(manifest_path.relative_to(proj)),
                              "sha256": _sha(manifest_path)},
        "condition_code_sha256": code_sha, "assets_sha256": assets_sha,
        "schedule_cache_key_version": 2, "complete": True,
        "budget": {"total_s": 7200, "historical_s": 0, "consumed_s": 1},
        "root_stats": {"definition": "train24 original frames",
                       "file_sha256": _sha(root_stats_path),
                       "files": train, "files_canonical_sha256":
                           canonical_list_sha256(train),
                       "frames_fitted": T_FIX * len(train),
                       "n_train_rows_used": len(train), "std_floor": 1e-6,
                       "mean": stats_mean.tolist(), "std": stats_std.tolist()},
        "schedule_rows": [{"demo": d, "split": "train" if d in train else "val",
                           **schedules[d]} for d in train + VAL_DEMOS],
        "execution": [], "total_wall_s": 1.0}
    cache_identity_path = cache_dir / "cache_identity.json"
    cache_identity_path.write_text(json.dumps(cache_identity, indent=1))

    return {"proj": proj, "source_root": src_root, "npz_dir": npz_dir,
            "pt_dir": pt_dir, "cache_dir": cache_dir,
            "preregistration": prereg, "manifest": manifest,
            "cache_identity": cache_identity, "train24": train24,
            "plan_anchor": plan_anchor}


def _runtime_probe():
    ref = json.loads((ROOT / radm.T6_REGISTRATION_REL).read_text())
    return dict(ref["runtime_identity"])


def _disk_probe():
    return lambda _p: 64 * (1 << 30)


def _stub_git(_root):
    return {"head": "fixture", "dirty": False, "status_sha256": "0" * 64,
            "tracked_diff_sha256": "0" * 64}


def _run_r(tmp: Path, fixture: dict, *, mutate=None, monkeypatch=None,
           runtime_probe=None, disk_probe=None, wall_cap_s=600.0) -> dict:
    if mutate is not None:
        mutate(fixture)
    out = tmp / f"r_out_{len(list(tmp.iterdir())) if tmp.exists() else 0}"
    budget = StageBudget(stage="R", wall_cap_s=wall_cap_s, reserve_s=10.0)
    result = radm.run_r_admission(
        out_dir=out, budget=budget, project_root=fixture["proj"],
        source_root=fixture["source_root"], teacher_npz_dir=fixture["npz_dir"],
        omomo_pt_dir=fixture["pt_dir"],
        runtime_probe=runtime_probe or _runtime_probe,
        disk_probe=disk_probe or _disk_probe(),
        git_identity=_stub_git)
    return result


def _authorize(fixture: dict, admission_path: Path, tmp: Path,
               includes=("D1", "D6", "F")) -> Path:
    doc = json.loads(Path(admission_path).read_text())
    package = doc["package"]
    payload = {
        "schema": radm.AUTHORIZATION_SCHEMA,
        "authorized": True,
        "cohort_id": radm.COHORT_ID,
        "checklist": {k: True for k in radm.AUTHORIZATION_CHECKLIST},
        "admission_package_sha256": radm.admission_package_sha256(admission_path),
        "includes": list(includes),
        "budget_caps": {k: [v["wall_cap_s"], v["reserve_s"]]
                        for k, v in package["budget"]["caps"].items()},
        "call_matrix_sha256": package["call_matrix"]["sha256"],
        "implementation_files_sha256":
            package["code_identity"]["implementation_files_sha256"],
        "note": "fixture authorization"}
    path = tmp / "authorization.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


@contextmanager
def _mutate_manifest(fixture, fn):
    manifest_path = (fixture["source_root"] / "audit/attempt_001"
                     / "training_manifest.json")
    ident_path = fixture["cache_dir"] / "cache_identity.json"
    saved_manifest, saved_ident = manifest_path.read_text(), ident_path.read_text()
    try:
        doc = json.loads(saved_manifest)
        fn(doc)
        manifest_path.write_text(json.dumps(doc, indent=1))
        ident = json.loads(saved_ident)
        ident["training_manifest"]["sha256"] = radm.sha256_file(manifest_path)
        ident_path.write_text(json.dumps(ident, indent=1))
        yield doc
    finally:
        manifest_path.write_text(saved_manifest)
        ident_path.write_text(saved_ident)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory) -> dict:
    return _fixture_project(tmp_path_factory.mktemp("r_fixture"))


@pytest.fixture(scope="module")
def admission(fixture, tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("r_admit")
    result = _run_r(tmp, fixture)
    assert result["summary_doc"]["admission_passed"] is True, \
        result["summary_doc"]["problems"]
    return {"result": result, "path": Path(result["summary"]),
            "doc": result["summary_doc"], "tmp": tmp}


# --------------------------------------------------------------------------
# R admission: happy path, zero declarations, selection semantics
# --------------------------------------------------------------------------

def test_admission_passes_with_six_windows_and_zero_declarations(admission):
    doc = admission["doc"]
    package = doc["package"]
    assert doc["admission_passed"] is True and doc["problems"] == []
    assert [w["demo"] for w in package["windows"]] == sorted(PILOT6)
    assert package["d1_window_id"] == "sub10_largebox_097:0:60"
    zero = doc["zero_declarations"]
    assert zero["array_value_reads"] == 0 and zero["checkpoint_reads"] == 0
    assert zero["cohort_created"] is False and zero["authorization_issued"] is False
    assert doc["seal"]["status"] == "sealed_within_budget"
    assert package["windows"][0]["derived_members"]["qpos"] == [[T_FIX, 43], "float64"]
    assert package["call_matrix"]["totals"]["window_consumptions_total"] == 28000
    assert package["call_matrix"]["totals"]["conditioner_forward_calls"] == 8206


def test_r_metadata_guard_refuses_arrays_and_checkpoints(tmp_path):
    reader = MetadataOnlyReader()
    p = tmp_path / "x.json"
    p.write_text("{}")
    assert reader.read_json(p) == {}
    with pytest.raises(FitDiagnosticError, match="r_metadata_only_array_read"):
        reader.load_arrays(p)
    with pytest.raises(FitDiagnosticError, match="r_metadata_only_checkpoint"):
        reader.torch_load(p)


def test_selection_reads_only_registration_keys(fixture):
    """Mutating quality/pairing numbers must not change the selection."""
    manifest = json.loads(json.dumps(fixture["manifest"]))
    for row in manifest["rows"]:
        row["quality"]["metrics"]["penetration_max_m"] = 9.9
        row["pairing"]["t_match"] = False
        row["scene_split"]["offset_pure_z"] = False
    windows, problems = radm.select_windows(
        fixture["preregistration"], manifest,
        {"teacher_pilot_train6_v1": [{"demo": d} for d in PILOT6]})
    assert problems == []
    assert [w.demo for w in windows] == sorted(PILOT6)
    assert windows[0].window_id == "sub10_largebox_097:0:60"


# --------------------------------------------------------------------------
# R admission refusals (§3.2/§8)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mutation,problem", [
    ("missing_demo", "r_pilot_demo_missing"),
    ("val_split", "r_pilot_split_not_train"),
    ("not_in_train24", "r_manifest_train24_mismatch_vs_preregistration"),
    ("not_admitted", "r_pilot_not_admitted"),
    ("short_T", "r_pilot_window_too_short"),
    ("no_full_window", "r_pilot_no_full_window"),
])
def test_window_selection_refusals(fixture, tmp_path, mutation, problem):
    def mutate_doc(doc):
        row = next(r for r in doc["rows"] if r["demo"] == PILOT6[-1])
        if mutation == "missing_demo":
            doc["rows"] = [r for r in doc["rows"] if r["demo"] != PILOT6[-1]]
        elif mutation == "val_split":
            row["split"] = "val"
        elif mutation == "not_in_train24":
            sel = doc["selection"]
            sel["train24"] = [d for d in sel["train24"] if d != PILOT6[-1]]
        elif mutation == "not_admitted":
            row["admission"]["motion"] = False
        elif mutation == "short_T":
            row["T"] = 30
            row["full_windows_F60_s30"] = 0
        elif mutation == "no_full_window":
            row["full_windows_F60_s30"] = 0
    with _mutate_manifest(fixture, mutate_doc):
        result = _run_r(tmp_path, fixture)
    assert result["summary_doc"]["admission_passed"] is False
    assert any(problem in p for p in result["summary_doc"]["problems"])


def test_pilot_demo_dropped_from_prereg_train24_refused(fixture, tmp_path):
    ident_path = fixture["cache_dir"] / "cache_identity.json"
    prereg_path = fixture["source_root"] / "freeze/preregistration.json"
    saved_prereg, saved_ident = prereg_path.read_text(), ident_path.read_text()
    try:
        doc = json.loads(saved_prereg)
        doc["selection"]["train24"] = [d for d in doc["selection"]["train24"]
                                       if d != PILOT6[-1]]
        doc["selection"]["train24_canonical_sha256"] = canonical_list_sha256(
            doc["selection"]["train24"])
        prereg_path.write_text(json.dumps(doc, indent=1))
        ident = json.loads(saved_ident)
        ident["preregistration"]["sha256"] = radm.sha256_file(prereg_path)
        ident_path.write_text(json.dumps(ident, indent=1))
        result = _run_r(tmp_path, fixture)
        assert result["summary_doc"]["admission_passed"] is False
        assert any("r_pilot_not_in_train24" in p
                   for p in result["summary_doc"]["problems"])
    finally:
        prereg_path.write_text(saved_prereg)
        ident_path.write_text(saved_ident)


def test_pilot6_shape_and_frozen_list_binding_refused(fixture, tmp_path):
    ident_path = fixture["cache_dir"] / "cache_identity.json"
    prereg_path = fixture["source_root"] / "freeze/preregistration.json"
    saved_prereg, saved_ident = prereg_path.read_text(), ident_path.read_text()
    try:
        doc = json.loads(saved_prereg)
        doc["selection"]["pilot6_from_a1"] = PILOT6[:5]
        prereg_path.write_text(json.dumps(doc, indent=1))
        ident = json.loads(saved_ident)
        ident["preregistration"]["sha256"] = radm.sha256_file(prereg_path)
        ident_path.write_text(json.dumps(ident, indent=1))
        result = _run_r(tmp_path, fixture)
        assert result["summary_doc"]["admission_passed"] is False
        problems = result["summary_doc"]["problems"]
        assert any("r_pilot6_invalid" in p for p in problems)

        doc["selection"]["pilot6_from_a1"] = PILOT6[:-1] + ["sub9_largebox_999"]
        prereg_path.write_text(json.dumps(doc, indent=1))
        ident["preregistration"]["sha256"] = radm.sha256_file(prereg_path)
        ident_path.write_text(json.dumps(ident, indent=1))
        result = _run_r(tmp_path, fixture)
        assert any("r_pilot6_not_bound_to_a1_frozen_lists" in p
                   for p in result["summary_doc"]["problems"])
    finally:
        prereg_path.write_text(saved_prereg)
        ident_path.write_text(saved_ident)


def test_binding_breaks_are_refused(fixture, tmp_path):
    manifest_path = (fixture["source_root"] / "audit/attempt_001"
                     / "training_manifest.json")
    ident_path = fixture["cache_dir"] / "cache_identity.json"
    saved_manifest, saved_ident = manifest_path.read_text(), ident_path.read_text()
    try:
        manifest_path.write_text(saved_manifest + "tampered")
        result = _run_r(tmp_path, fixture)
        assert any("r_training_manifest_binding_broken" in p
                   for p in result["summary_doc"]["problems"])
        manifest_path.write_text(saved_manifest)
        ident = json.loads(saved_ident)
        ident["training_manifest"]["sha256"] = "0" * 64
        ident_path.write_text(json.dumps(ident, indent=1))
        result = _run_r(tmp_path, fixture)
        assert result["summary_doc"]["admission_passed"] is False
        assert any("r_training_manifest_binding_broken" in p
                   for p in result["summary_doc"]["problems"])
    finally:
        manifest_path.write_text(saved_manifest)
        ident_path.write_text(saved_ident)


@pytest.mark.parametrize("target,label", [
    ("teacher_npz", "r_preregistration_reverify_failed"),
    ("derived", "r_window_derived_sha_mismatch"),
    ("schedule", "r_window_schedule_sha_mismatch"),
    ("root_stats", "r_cache_reverify_failed"),
])
def test_source_drift_refusals(fixture, tmp_path, target, label):
    if target == "teacher_npz":
        p = fixture["npz_dir"] / f"{PILOT6[0]}_original.npz"
    elif target == "derived":
        p = (fixture["source_root"] / "audit/attempt_001/derived"
             / f"{PILOT6[0]}_w1_scene.npz")
    elif target == "schedule":
        rec = next(r for r in fixture["cache_identity"]["schedule_rows"]
                   if r["demo"] == PILOT6[0])
        p = fixture["cache_dir"] / "schedules" / \
            w1.schedule_cache_filename(PILOT6[0], rec["key"])
    else:
        p = fixture["cache_dir"] / "root_stats.npz"
    saved = p.read_bytes()
    try:
        p.write_bytes(saved + b"drift")
        result = _run_r(tmp_path, fixture)
        assert result["summary_doc"]["admission_passed"] is False
        assert any(label in q for q in result["summary_doc"]["problems"])
    finally:
        p.write_bytes(saved)


def test_condition_code_drift_refused(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(w1, "condition_code_sha256",
                        lambda root, hash_file=None: "0" * 64)
    result = _run_r(tmp_path, fixture)
    assert result["summary_doc"]["admission_passed"] is False
    assert any("cache_condition_code_drifted" in p
               for p in result["summary_doc"]["problems"])


def test_runtime_identity_drift_stops_admission(fixture, tmp_path):
    probe = _runtime_probe()
    probe["torch"] = "0.0.0+drift"
    result = _run_r(tmp_path, fixture, runtime_probe=lambda: probe)
    assert result["summary_doc"]["admission_passed"] is False
    assert any("r_runtime_identity:runtime_identity:torch" in p
               for p in result["summary_doc"]["problems"])


def test_disk_margin_refused(fixture, tmp_path):
    result = _run_r(tmp_path, fixture, disk_probe=lambda _p: 1 << 30)
    assert result["summary_doc"]["admission_passed"] is False
    assert any("r_disk_free_insufficient" in p
               for p in result["summary_doc"]["problems"])


def test_config_drift_refused(fixture, tmp_path):
    yaml_path = fixture["proj"] / "configs/train/omniretarget_motion.yaml"
    saved = yaml_path.read_text()
    try:
        yaml_path.write_text(saved.replace("w_fk: 2.0", "w_fk: 9.0"))
        result = _run_r(tmp_path, fixture)
        assert result["summary_doc"]["admission_passed"] is False
        assert any("r_config:config_loss_w_fk" in p
                   for p in result["summary_doc"]["problems"])
    finally:
        yaml_path.write_text(saved)


def test_budget_exhaustion_withdraws_the_seal(fixture, tmp_path):
    result = _run_r(tmp_path, fixture, wall_cap_s=0.001)
    doc = result["summary_doc"]
    assert doc["seal"]["status"] == "budget_exceeded_after_seal"
    assert doc["admission_passed"] is False
    assert doc["label_verdict"] == "budget_exceeded_after_seal"
    assert doc["seal"]["measured_wall_s_at_summary"] >= 0.001


def test_load_admission_refuses_unsealed_or_failed_documents(fixture, admission,
                                                             tmp_path):
    radm.load_admission(admission["path"])
    bad = tmp_path / "bad.json"
    doc = json.loads(admission["path"].read_text())
    doc["admission_passed"] = False
    bad.write_text(json.dumps(doc))
    with pytest.raises(FitDiagnosticError, match="admission_not_passed"):
        radm.load_admission(bad)
    doc["admission_passed"] = True
    doc["seal"]["status"] = "sealing"
    bad.write_text(json.dumps(doc))
    with pytest.raises(FitDiagnosticError, match="admission_seal_not_consumable"):
        radm.load_admission(bad)


# --------------------------------------------------------------------------
# admission recheck + §8.3 authorization gate
# --------------------------------------------------------------------------

def test_admission_recheck_passes_and_detects_drift(fixture, admission,
                                                    tmp_path):
    assert radm.admission_recheck_problems(
        admission["doc"], project_root=fixture["proj"],
        source_root=fixture["source_root"], teacher_npz_dir=fixture["npz_dir"],
        omomo_pt_dir=fixture["pt_dir"], runtime_probe=_runtime_probe,
        disk_probe=_disk_probe()) == []
    sched_rec = next(r for r in fixture["cache_identity"]["schedule_rows"]
                     if r["demo"] == PILOT6[1])
    sched_path = fixture["cache_dir"] / "schedules" / \
        w1.schedule_cache_filename(PILOT6[1], sched_rec["key"])
    saved = sched_path.read_bytes()
    try:
        sched_path.write_bytes(saved + b"drift")
        problems = radm.admission_recheck_problems(
            admission["doc"], project_root=fixture["proj"],
            source_root=fixture["source_root"],
            teacher_npz_dir=fixture["npz_dir"], omomo_pt_dir=fixture["pt_dir"],
            runtime_probe=_runtime_probe, disk_probe=_disk_probe())
        assert any("admission_window_file_drifted" in p for p in problems)
        assert any("admission_cache_drifted" in p for p in problems)
    finally:
        sched_path.write_bytes(saved)
    drifted = _runtime_probe()
    drifted["mujoco"] = "9.9.9"
    problems = radm.admission_recheck_problems(
        admission["doc"], project_root=fixture["proj"],
        source_root=fixture["source_root"], teacher_npz_dir=fixture["npz_dir"],
        omomo_pt_dir=fixture["pt_dir"], runtime_probe=lambda: drifted,
        disk_probe=_disk_probe())
    assert any("admission_runtime_identity" in p for p in problems)


def test_authorization_gate(admission, tmp_path, fixture):
    auth = _authorize(fixture, admission["path"], tmp_path)
    assert radm.authorization_problems(
        json.loads(auth.read_text()), admission["doc"], admission["path"]) == []

    doc = json.loads(auth.read_text())
    def mutated(**changes):
        bad = dict(doc)
        bad.update(changes)
        return bad
    cases = [
        ({"schema": "other"}, "authorization_schema_invalid"),
        ({"authorized": False}, "authorization_not_authorized"),
        ({"checklist": {}}, "authorization_checklist_invalid"),
        ({"admission_package_sha256": "0" * 64},
         "authorization_admission_sha_mismatch"),
        ({"cohort_id": "other_cohort"}, "authorization_cohort_mismatch"),
        ({"includes": ["D6", "F"]}, "authorization_stage_includes_invalid"),
        ({"budget_caps": {}}, "authorization_budget_caps_mismatch"),
        ({"call_matrix_sha256": "0" * 64},
         "authorization_call_matrix_sha_mismatch"),
        ({"implementation_files_sha256": "0" * 64},
         "authorization_implementation_sha_mismatch"),
    ]
    for changes, problem in cases:
        assert problem in radm.authorization_problems(
            mutated(**changes), admission["doc"], admission["path"])
    checklist_bad = dict(doc["checklist"])
    checklist_bad[radm.AUTHORIZATION_CHECKLIST[0]] = False
    assert any("authorization_checklist_not_true" in p
               for p in radm.authorization_problems(
                   mutated(checklist=checklist_bad), admission["doc"],
                   admission["path"]))
    assert "authorization_implementation_sha_mismatch" in \
        radm.authorization_problems(doc, admission["doc"], admission["path"],
                                    current_impl_sha="1" * 64)


# --------------------------------------------------------------------------
# billed-window materialization (§3.2)
# --------------------------------------------------------------------------

def _mini_world(fixture, admission, tmp, stage="D1", **kwargs):
    budget = StageBudget(stage=stage, wall_cap_s=600.0, reserve_s=10.0)
    return budget, radm.materialize_real_world(
        admission=admission["doc"], stage=stage, budget=budget, device="cpu",
        fixture=True, project_root=fixture["proj"],
        source_root=fixture["source_root"], teacher_npz_dir=fixture["npz_dir"],
        omomo_pt_dir=fixture["pt_dir"], **kwargs)


def test_materialization_requires_a_started_fit_window(admission, fixture):
    world = object()
    with pytest.raises(FitDiagnosticError, match="without_billed_interval"):
        materialize_for_fit(world, None)
    r_budget = StageBudget(stage="R", wall_cap_s=600.0, reserve_s=10.0)
    with pytest.raises(FitDiagnosticError, match="materialization_stage_not_fit"):
        materialize_for_fit(world, r_budget)
    f_budget = StageBudget(stage="F", wall_cap_s=600.0, reserve_s=10.0)
    with pytest.raises(FitDiagnosticError, match="materialization_stage_not_fit"):
        materialize_for_fit(world, f_budget)
    spent = StageBudget(stage="D1", wall_cap_s=1.0, reserve_s=0.5,
                        t0=0.0, clock=lambda: 2.0)
    with pytest.raises(FitDiagnosticError, match="materialize_for_fit|deadline"):
        materialize_for_fit(world, spent)


def test_fixture_d1_world_materialization(admission, fixture, tmp_path):
    budget, world = _mini_world(fixture, admission, tmp_path, stage="D1")
    assert world.registration_for("D1").windows[0].window_id == \
        "sub10_largebox_097:0:60"
    assert len(world.items) == 1
    item = world.items[0]
    assert item["x0"].shape == (60, 38) and item["x0"].dtype == torch.float32
    assert torch.isfinite(item["x0"]).all()
    probe = world.view_for("D1")["probe_batch"]
    assert probe["x0"].shape == (1, 60, 38)
    counters = world.counters
    assert counters["registered_window_reads"] == 1
    assert counters["window_x0_encodes"] == 1
    x0_sha = world.facts["windows"][0]["x0_sha256"]
    # a second materialization of the same window is bitwise identical
    _, world2 = _mini_world(fixture, admission, tmp_path, stage="D1")
    assert world2.facts["windows"][0]["x0_sha256"] == x0_sha


def test_fixture_d6_world_and_shared_window_binding(admission, fixture,
                                                    tmp_path):
    _, d1_world = _mini_world(fixture, admission, tmp_path, stage="D1")
    inputs = tmp_path / "probe_inputs.npz"
    np.savez(inputs, **{"w0__x0": d1_world.items[0]["x0"].numpy()})
    budget = StageBudget(stage="D6", wall_cap_s=600.0, reserve_s=10.0)
    world = radm.materialize_real_world(
        admission=admission["doc"], stage="D6", budget=budget, device="cpu",
        fixture=True, project_root=fixture["proj"],
        source_root=fixture["source_root"], teacher_npz_dir=fixture["npz_dir"],
        omomo_pt_dir=fixture["pt_dir"], d1_probe_inputs=inputs)
    assert len(world.items) == 6
    assert world.facts["d1_shared_window_x0_sha256"] == \
        world.facts["windows"][0]["x0_sha256"]
    assert world.counters["registered_window_reads"] == 6
    # a tampered D1 saved window must refuse the D6 materialization
    tampered = tmp_path / "tampered_inputs.npz"
    x0 = d1_world.items[0]["x0"].numpy().copy()
    x0[0, 0] += 1e-3
    np.savez(tampered, **{"w0__x0": x0})
    with pytest.raises(FitDiagnosticError, match="d1_d6_shared_window_x0_mismatch"):
        radm.materialize_real_world(
            admission=admission["doc"], stage="D6", budget=budget, device="cpu",
            fixture=True, project_root=fixture["proj"],
            source_root=fixture["source_root"],
            teacher_npz_dir=fixture["npz_dir"],
            omomo_pt_dir=fixture["pt_dir"], d1_probe_inputs=tampered)


def test_materialization_refuses_source_drift(admission, fixture, tmp_path):
    sched_rec = next(r for r in fixture["cache_identity"]["schedule_rows"]
                     if r["demo"] == PILOT6[2])
    sched_path = fixture["cache_dir"] / "schedules" / \
        w1.schedule_cache_filename(PILOT6[2], sched_rec["key"])
    saved = sched_path.read_bytes()
    try:
        sched_path.write_bytes(saved + b"drift")
        with pytest.raises(FitDiagnosticError, match="materialize_admission_drifted"):
            _mini_world(fixture, admission, tmp_path, stage="D1")
    finally:
        sched_path.write_bytes(saved)


def test_real_world_refuses_fk_injection(admission, fixture, tmp_path):
    budget = StageBudget(stage="D1", wall_cap_s=600.0, reserve_s=10.0)
    with pytest.raises(FitDiagnosticError,
                       match="real_world_production_fk_injection_forbidden"):
        radm.materialize_real_world(
            admission=admission["doc"], stage="D1", budget=budget,
            device="cpu", fixture=False, fk=object(),
            project_root=fixture["proj"], source_root=fixture["source_root"],
            teacher_npz_dir=fixture["npz_dir"], omomo_pt_dir=fixture["pt_dir"])


def test_apparatus_window_guard(admission, fixture, tmp_path):
    with pytest.raises(FitDiagnosticError, match="materialization_stage_not_fit"):
        radm.materialize_fit_apparatus(
            admission=admission["doc"], device="cpu", fixture=True,
            budget=StageBudget(stage="R", wall_cap_s=600.0, reserve_s=10.0),
            project_root=fixture["proj"], source_root=fixture["source_root"],
            teacher_npz_dir=fixture["npz_dir"], omomo_pt_dir=fixture["pt_dir"])
    budget = StageBudget(stage="F", wall_cap_s=600.0, reserve_s=10.0)
    apparatus = radm.materialize_fit_apparatus(
        admission=admission["doc"], budget=budget, device="cpu", fixture=True,
        project_root=fixture["proj"], source_root=fixture["source_root"],
        teacher_npz_dir=fixture["npz_dir"], omomo_pt_dir=fixture["pt_dir"])
    x0 = np.zeros((60, 38), dtype=np.float32)
    x0[:, 3] = 1.0  # identity 6D: [1,0,0, 0,1,0]
    x0[:, 7] = 1.0
    decoded = apparatus.decoder.decode(x0)
    assert decoded["pos"].shape == (60, 3)
    assert decoded["R"].shape == (60, 3, 3) and decoded["q"].shape == (60, 29)
    assert apparatus.counters["root_stats_loads"] == 1


# --------------------------------------------------------------------------
# call matrix consistency
# --------------------------------------------------------------------------

def test_frozen_call_matrix_matches_plan_and_driver():
    matrix = radm.frozen_call_matrix()
    d1 = expected_call_matrix(FitPlan("D1"), 1)
    d6 = expected_call_matrix(FitPlan("D6"), 6)
    for key, want in d1.items():
        assert matrix["D1"][key] == want, key
    for key, want in d6.items():
        assert matrix["D6"][key] == want, key
    totals = matrix["totals"]
    assert totals["training_updates"] == 2000
    assert totals["training_micros"] == 8000
    assert totals["window_consumptions_total"] == 28000
    assert totals["probe_calls"] == 202
    assert totals["conditioner_forward_calls"] == 8206
    assert totals["backbone_forward_calls"] == 8206
    f = matrix["F"]
    assert f["adamw_step_calls"] == 0 and f["backward_calls"] == 0
    assert f["training_updates"] == 0
    assert f["checkpoint_loads"] == 4 and f["fresh_probe_calls"] == 4
    assert f["model_constructions"] == 2 and f["conditioner_constructions"] == 2


# --------------------------------------------------------------------------
# supervisor (real-fit outer return supervision)
# --------------------------------------------------------------------------

_supervisor_spec = importlib.util.spec_from_file_location(
    "w1_real_fit_supervisor",
    ROOT / "scripts/run_w1_smallset_fit_real_supervisor.py")


def _load_supervisor():
    spec = importlib.util.spec_from_file_location(
        "w1_real_fit_supervisor",
        ROOT / "scripts/run_w1_smallset_fit_real_supervisor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seal_and_proof(path: Path, t0: float, cap: float, registered: float,
                    elapsed: float, extra: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"seal": {"status": "sealed_within_budget", "terminal": True,
                    "t0_perf_counter": t0, "wall_cap_s": cap,
                    "measured_wall_s_at_summary": registered},
           "budget": {"t0_perf_counter": t0, "wall_cap_s": cap,
                      "measured_wall_s_at_summary": registered}}
    doc.update(extra)
    path.write_text(json.dumps(doc))
    proof = {"schema": "n0_return_observation_v1", "covered": True,
             "elapsed_s": elapsed, "registered_measured_s": registered,
             "wall_cap_s": cap,
             "summary_sha256": _load_supervisor()._sha(path)}
    (path.parent / "return_observation.json").write_text(json.dumps(proof))
    return doc


def _populate_r(out: Path, command, mutation):
    t0 = float.fromhex(command[command.index("--outer-t0-hex") + 1])
    launch_id = command[command.index("--launch-id") + 1]
    package = {"schema": "w1_smallset_fit_admission_package_v1",
               "cohort_id": radm.COHORT_ID, "windows": [{"demo": d} for d in PILOT6],
               "d1_window_id": "sub10_largebox_097:0:60",
               "call_matrix": {"sha256": "a" * 64},
               "runtime_binding": {"requirements": {}},
               "code_identity": {"implementation_files_sha256": "b" * 64}}
    zero = {"array_value_reads": 0, "checkpoint_reads": 0,
            "cohort_created": False}
    extra = {"schema": "w1_smallset_fit_r_admission_v1", "kind": "r_admission",
             "admission_passed": True, "problems": [], "package": package,
             "zero_declarations": zero}
    if mutation == "r_not_passed":
        extra["admission_passed"] = False
        extra["problems"] = ["injected"]
    if mutation == "r_zero_declarations_broken":
        zero["array_value_reads"] = 3
    if mutation == "r_windows_incomplete":
        package["windows"] = package["windows"][:5]
    _seal_and_proof(out / "r_admission.json", t0, 120.0, 20.0, 4.0, extra)
    summary = json.loads((out / "r_admission.json").read_text())
    summary["outer_launch"] = {"launch_id": launch_id, "t0_perf_counter": t0,
                               "cli_entry_perf_counter": t0 + 1.0,
                               "supervisor_declared": True}
    summary["label_verdict"] = "pending_outer_return"
    (out / "r_admission.json").write_text(json.dumps(summary))
    proof = json.loads((out / "return_observation.json").read_text())
    proof["summary_sha256"] = _load_supervisor()._sha(out / "r_admission.json")
    (out / "return_observation.json").write_text(json.dumps(proof))


@pytest.mark.parametrize("mutation,expected", [
    (None, None),
    ("r_not_passed", "r_admission_incomplete"),
    ("r_zero_declarations_broken", "r_admission_incomplete"),
    ("r_windows_incomplete", "r_admission_incomplete"),
    ("late_wait", "actual_wait_not_covered"),
    ("nonzero", "worker_exit:1"),
    ("source_drift", "implementation_changed_during_run"),
])
def test_r_supervisor_requires_actual_wait_and_valid_summary(
        tmp_path, monkeypatch, capsys, mutation, expected):
    supervisor = _load_supervisor()
    ticks = {"now": 105.0}
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: ticks["now"])
    hashes = iter([{"source": "before"},
                   {"source": "after" if mutation == "source_drift" else "before"}])
    monkeypatch.setattr(supervisor, "_source_hashes", lambda: next(hashes))
    real_write = supervisor._write_json

    def controlled_write(path, obj):
        real_write(path, obj)

    class Child:
        pid = 1234

        def __init__(self, command, **kwargs):
            assert command[:3] == ["uv", "run", "--locked"]
            assert "--mode" in command and "r-admit" in command
            _populate_r(tmp_path / "attempt_001", command, mutation)

        def wait(self, timeout=None):
            ticks["now"] = 121.0 if mutation == "late_wait" else 110.0
            return 1 if mutation == "nonzero" else 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", Child)
    args = SimpleNamespace(mode="r-admit", output_root=tmp_path,
                           wall_cap_s=120.0, reserve_s=10.0,
                           admission=None, authorization=None, impl_sha=None,
                           fixture=False)
    rc = supervisor.run(args, t0=100.0)
    out = tmp_path / "attempt_001"
    verdict = json.loads((out / "acceptance_verdict.json").read_text())
    if expected is None:
        assert rc == 0 and verdict["verdict"] == "candidate_complete"
        assert verdict["problems"] == []
        receipt = json.loads((out / "outer_return_observation.json").read_text())
        assert receipt["wait_elapsed_s"] == 10.0
        assert receipt["in_process_elapsed_s"] <= receipt["wait_elapsed_s"] <= 20.0
    else:
        assert rc == 3 and verdict["verdict"] == "incomplete"
        assert expected in verdict["problems"]
