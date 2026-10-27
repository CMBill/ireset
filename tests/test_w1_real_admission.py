"""Tests for the read-only W1 real-objective admission package.

The fixture is a synthetic metadata tree: no real dataset, cache, checkpoint or
production history is read.  Every test either proves a frozen field or a
refusal; none may create a cohort, a registered run or a model call.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ireset.eval.w1_real_admission as admission
import ireset.eval.w1_real_objective_audit as audit
from ireset.data import w1_training as w1


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _demos() -> list[str]:
    return [f"sub{subj}_demo_{i:02d}"
            for subj in ("A", "B", "C", "D", "E", "F")
            for i in range(8)]


@pytest.fixture(scope="module")
def calibrated_contract() -> dict:
    contract = audit.synthetic_calibration(seed=20261026)
    assert audit.tolerance_contract_problems(contract) == []
    return contract


@pytest.fixture()
def project(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    root = tmp_path / "proj"
    root.mkdir()
    w1_run = Path("derived/experiments/omniretarget_first/w1_run")
    source = Path("derived/experiments/omniretarget_first/source")
    attr = Path("derived/experiments/omniretarget_first/attr")

    config = root / "configs/train/omniretarget_motion.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model_type: deterministic\n")

    for rel in ("src/ireset/eval/w1_real_admission.py",
                "scripts/w1_real_objective_admission.py",
                "tests/test_w1_real_admission.py"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# fixture {rel}\n")

    demos = _demos()
    _write_json(root / "derived/data/split/train.json", {"demos": demos})
    train24 = w1.select_train24(demos)

    prereg = root / source / "audit/attempt_001/preregistration.json"
    _write_json(prereg, {"assets": {"a1_assets_sha256": "a" * 64}})

    rows = []
    for demo in train24:
        rows.append({
            "demo": demo, "T": 133, "split": "train",
            "admission": {"motion": True},
            "inputs": {
                "npz_sha256": hashlib.sha256(f"{demo}|npz".encode()).hexdigest(),
                "pt_sha256": hashlib.sha256(f"{demo}|pt".encode()).hexdigest(),
            },
            "derived": {
                "path": f"{source}/audit/attempt_001/derived/{demo}.npz",
                "sha256": hashlib.sha256(f"{demo}|derived".encode()).hexdigest(),
            },
        })
    manifest = {
        "manifest_name": w1.MANIFEST_NAME,
        "protocol": w1.MOTION_PROTOCOL_VERSION,
        "selection": {
            "train24": train24,
            "train24_canonical_sha256": w1.canonical_list_sha256(train24),
        },
        "preregistration": {
            "path": str(prereg.relative_to(root)), "sha256": _sha(prereg),
        },
        "rows": rows,
    }
    manifest_path = root / source / "audit/attempt_001/training_manifest.json"
    _write_json(manifest_path, manifest)

    cache_dir = root / source / "cache/attempt_001"
    cache_identity = cache_dir / "cache_identity.json"
    root_stats = cache_dir / "root_stats.npz"
    _write_json(cache_identity, {"cache": "identity"})
    root_stats.parent.mkdir(parents=True, exist_ok=True)
    root_stats.write_bytes(b"root-stats")

    reg_path = root / w1_run / "freeze/execution_registration.json"
    _write_json(reg_path, {
        "training_manifest": {
            "path": str(manifest_path.relative_to(root)),
            "sha256": _sha(manifest_path),
        },
        "cache": {
            "dir": str(cache_dir.relative_to(root)),
            "identity_sha256": _sha(cache_identity),
            "root_stats_sha256": _sha(root_stats),
        },
        "configs": {
            "config_file": "configs/train/omniretarget_motion.yaml",
            "config_file_sha256": _sha(config),
        },
    })

    w1_final = root / w1_run / "decisions/attempt_001/w1_de_final.json"
    _write_json(w1_final, {"conclusion": "not_ready"})
    attr_reg = root / attr / "freeze/execution_registration.json"
    _write_json(attr_reg, {"registration": "attr"})
    attr_final = root / attr / "decisions/attempt_001/prediction_consistency_final.json"
    _write_json(attr_final, {"conclusion": "mixed_evidence"})
    source_manifest = root / source / "audit/attempt_001/training_manifest.json"

    checkpoints = {}
    for model in ("deterministic", "diffusion"):
        p = root / w1_run / f"formal/{model}/attempt_001/best.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"{model}-checkpoint".encode())
        checkpoints[model] = (p.relative_to(root), _sha(p), 4000)

    parent_anchors = {
        "w1_execution_registration": (
            str(reg_path.relative_to(root)), _sha(reg_path)),
        "w1_final": (str(w1_final.relative_to(root)), _sha(w1_final)),
        "prediction_consistency_registration": (
            str(attr_reg.relative_to(root)), _sha(attr_reg)),
        "prediction_consistency_final": (
            str(attr_final.relative_to(root)), _sha(attr_final)),
    }
    source_anchors = {
        "source_training_manifest": (
            str(source_manifest.relative_to(root)), _sha(source_manifest)),
    }

    monkeypatch.setattr(audit, "W1_RUN", w1_run)
    monkeypatch.setattr(audit, "PARENT_ANCHORS", parent_anchors)
    monkeypatch.setattr(audit, "SOURCE_ANCHORS", source_anchors)
    monkeypatch.setattr(audit, "EXPECTED_CHECKPOINTS", checkpoints)
    monkeypatch.setattr(audit, "environment_facts", lambda: {"python": "test"})
    monkeypatch.setattr(audit, "implementation_identity",
                        lambda _root: {"git": {"commit": "test"}, "files": {}})
    monkeypatch.setattr(w1, "condition_code_sha256", lambda _root: "c" * 64)

    return SimpleNamespace(
        root=root, w1_run=w1_run, manifest_path=manifest_path,
        reg_path=reg_path, manifest=manifest, train24=train24,
        parent_anchors=parent_anchors, source_anchors=source_anchors,
        checkpoints=checkpoints,
        config=config, cache_identity=cache_identity,
    )


def _build(project: SimpleNamespace, contract: dict, **kwargs) -> dict:
    return admission.build_admission_package(
        project.root, tolerance_contract=contract,
        tolerance_contract_path=kwargs.get("path"),
        tolerance_contract_sha256=kwargs.get("sha"))


def _rewrite_manifest(project: SimpleNamespace, manifest: dict) -> None:
    _write_json(project.manifest_path, manifest)
    reg = json.loads(project.reg_path.read_text())
    reg["training_manifest"]["sha256"] = _sha(project.manifest_path)
    _write_json(project.reg_path, reg)
    project.parent_anchors["w1_execution_registration"] = (
        str(project.reg_path.relative_to(project.root)), _sha(project.reg_path))
    project.source_anchors["source_training_manifest"] = (
        str(project.manifest_path.relative_to(project.root)),
        _sha(project.manifest_path))


def test_package_freezes_matrix_and_reads_metadata_only(
        project: SimpleNamespace, calibrated_contract: dict, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("torch.load must not run for an admission package")

    monkeypatch.setattr(torch, "load", forbidden)
    package = _build(project, calibrated_contract,
                     path=Path("derived/audit/tolerance_contract.json"),
                     sha="f" * 64)
    assert package["passed"], package["problems"]
    assert package["status"]["run"] == "not_run"
    assert package["status"]["authorized"] is False
    assert package["status"]["section_8_checked"] is False
    assert package["read_scope"]["metadata_only"] is True
    assert package["read_scope"]["checkpoint_torch_load"] is False
    assert package["read_scope"]["model_forward_calls"] == 0
    assert package["read_scope"]["vjp_calls"] == 0

    candidates = package["window_selection"]["candidates"]
    assert len(candidates) == 4
    demos = sorted({c["demo"] for c in candidates})
    assert demos == sorted(project.train24)[:2]
    for demo in demos:
        starts = [c["start"] for c in candidates if c["demo"] == demo]
        assert starts == [0, 30]
        assert all(c["end"] == c["start"] + 60 for c in candidates
                   if c["demo"] == demo)
        assert all(c["split"] == "train" and c["T"] == 133
                   for c in candidates if c["demo"] == demo)

    assert package["required_rows"]["objective_count"] == 98
    assert package["required_rows"]["gradient_count"] == 14
    assert package["required_rows"]["expected_objective_count"] == 98
    assert package["required_rows"]["expected_gradient_count"] == 14
    assert package["window_selection"]["subjects"] == ["subA"]
    assert package["window_selection"]["same_subject"] is True
    assert any("same subject" in note or "from subject subA" in note
               for note in package["review_notes"])
    assert package["noise"]["sha256"] == audit._hash_array(
        audit.generate_noise(audit.NOISE_SEED, (4, 60, 38)))
    assert package["alpha"]["sha256"] == audit._hash_array(
        audit.independent_alpha_cumprod(1000, "scaled_linear", 1e-4, 0.02))
    assert package["cohort"]["call_caps"] == audit.CALL_CAPS
    assert package["cohort"]["total_budget_s"] == 900.0
    assert package["cohort"]["stage_budgets"]["R0"]["attempts"] == 2
    assert package["call_matrix"]["R0"] == {
        "conditioner": 0, "model": 0, "production_vjp": 0,
        "independent_vjp": 0}
    assert package["call_matrix"]["total_batched_vjp_cap"] == 8
    assert package["fixed_assets"]["checkpoints"]["deterministic"]["match"] is True
    assert package["fixed_assets"]["checkpoints"]["deterministic"]["step"] == 4000
    assert package["fixed_assets"]["checkpoints"]["deterministic"]["step_verified"] is False
    assert package["fixed_assets"]["manifest"]["match"] is True
    assert package["fixed_assets"]["config"]["match"] is True
    assert package["tolerance_contract"]["evidence_sha256"] == \
        calibrated_contract["evidence_sha256"]
    assert package["tolerance_contract"]["sha256"] == "f" * 64
    assert package["generator"]["schema"] == "w1_real_admission_generator_v1"
    assert all(package["generator"]["files"].values())
    assert package["generator"]["calibration_contract_sha256"] == "f" * 64
    assert package["authorization"]["authorized"] is False
    assert set(package["authorization"]["checklist"]) == \
        set(audit.AUTHORIZATION_REQUIRED)
    assert all(v is None for v in package["authorization"]["checklist"].values())
    assert package["authorization"]["template"]["authorized"] is False
    assert package["boundaries"] == audit.BOUNDARIES
    assert package["coverage"]["objective_coverage"] == "partial"
    assert not (project.root / project.w1_run / "cohort.json").exists()
    assert not (project.root / project.w1_run / "registration").exists()


def test_package_refuses_default_or_widened_contract(project: SimpleNamespace):
    with pytest.raises(ValueError, match="completed synthetic calibration"):
        _build(project, audit.build_tolerance_contract())
    widened = copy.deepcopy(audit.synthetic_calibration(seed=123))
    widened["tolerances"]["loss_float32"]["atol"] *= 10.0
    with pytest.raises(ValueError, match="tolerance contract refused"):
        _build(project, widened)


def test_checkpoint_drift_is_reported(project: SimpleNamespace,
                                      calibrated_contract: dict):
    ckpt = project.root / project.checkpoints["deterministic"][0]
    ckpt.write_bytes(b"tampered")
    package = _build(project, calibrated_contract)
    assert package["passed"] is False
    assert "checkpoint_drifted:deterministic" in package["problems"]
    assert package["fixed_assets"]["checkpoints"]["deterministic"]["match"] is False


def test_manifest_drift_is_reported(project: SimpleNamespace,
                                    calibrated_contract: dict):
    project.manifest_path.write_text(project.manifest_path.read_text() + "\n")
    package = _build(project, calibrated_contract)
    assert package["passed"] is False
    assert "training_manifest_hash_drifted" in package["problems"]


def test_motion_exclusion_follows_the_frozen_order(
        project: SimpleNamespace, calibrated_contract: dict):
    eligible = sorted(project.train24)
    manifest = copy.deepcopy(project.manifest)
    for row in manifest["rows"]:
        if row["demo"] == eligible[0]:
            row["admission"]["motion"] = False
    _rewrite_manifest(project, manifest)
    package = _build(project, calibrated_contract)
    assert package["passed"], package["problems"]
    demos = [c["demo"] for c in package["window_selection"]["candidates"]]
    assert demos == [eligible[1], eligible[1], eligible[2], eligible[2]]


def test_one_eligible_demo_is_incomplete(project: SimpleNamespace,
                                         calibrated_contract: dict):
    eligible = sorted(project.train24)
    manifest = copy.deepcopy(project.manifest)
    for row in manifest["rows"]:
        row["admission"]["motion"] = row["demo"] == eligible[0]
    _rewrite_manifest(project, manifest)
    package = _build(project, calibrated_contract)
    assert package["passed"] is False
    assert any("fewer_than_2_train_demos" in p for p in package["problems"])
    assert any("window_candidate_count:2" in p for p in package["problems"])


def test_authorization_template_is_not_authorization():
    template = admission.authorization_template()
    assert template["template_only"] is True
    problems = audit.authorization_problems(template)
    assert "not_authorized" in problems
    assert any(p.startswith("authorization_checklist:") for p in problems)
    authorized = copy.deepcopy(template)
    authorized["template_only"] = False
    authorized["authorized"] = True
    authorized["granted_by"] = "unit-test"
    authorized["checklist"] = {k: True for k in audit.AUTHORIZATION_REQUIRED}
    assert audit.authorization_problems(authorized) == []


def test_write_package_never_overwrites(project: SimpleNamespace,
                                        calibrated_contract: dict,
                                        tmp_path: Path):
    package = _build(project, calibrated_contract)
    p1, s1 = admission.write_package(tmp_path / "packages", package)
    p2, s2 = admission.write_package(tmp_path / "packages", package)
    assert p1.parent.name == "attempt_001"
    assert p2.parent.name == "attempt_002"
    assert s1 != s2
    assert json.loads(p1.read_text())["attempt"] == "attempt_001"
    assert json.loads(p2.read_text())["attempt"] == "attempt_002"
