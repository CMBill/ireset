"""Supplementary real-material admission tamper checks.

This file is intentionally outside ``IMPLEMENTATION_FILES`` so it can test the
frozen M5 bytes without changing the implementation identity.  It skips when
the current M5 attempt is not present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ireset.eval.w1_real_training_update_audit as A
from ireset.paths import PROJECT_ROOT


MATERIALS = (PROJECT_ROOT / "derived/audit/w1_real_training_update_audit/"
             "gradient_numerics_retest/materials/attempt_009")


def _package() -> dict:
    path = MATERIALS / "attempt_001/retest_admission.json"
    if not path.is_file():
        pytest.skip("frozen retest admission is not present")
    return json.loads(path.read_text())


def test_frozen_admission_is_deeply_consumable():
    assert A.retest_admission_problems(
        _package(), project_root=PROJECT_ROOT, strict=True) == []


def test_draft_binds_real_admission_and_preflight_checks_them():
    draft = json.loads((MATERIALS / "authorization_draft.json").read_text())
    package = json.loads(
        (MATERIALS / "attempt_001/retest_admission.json").read_text())
    real = json.loads(
        (PROJECT_ROOT / "derived/audit/w1_real_training_update_audit/"
         "admission_real/attempt_018/admission_package.json").read_text())
    assert draft["calibration_verification_sha256"] == real[
        "calibration_verification_sha256"]
    assert draft["tolerance_evidence_sha256"] == real["tolerance"][
        "evidence_sha256"]
    assert draft["fp32_contract_sha256"] == real["fp32_contract_sha256"]
    assert draft["admission_package_sha256"] == A.sha256_file(
        PROJECT_ROOT / "derived/audit/w1_real_training_update_audit/"
        "admission_real/attempt_018/admission_package.json")
    assert draft["acceptance_sha256"] == package["retest_acceptance"][
        "sha256"]
    assert json.loads((MATERIALS / "preflight_verification.json").read_text())[
        "passed"] is True


def test_snapshot_spec_and_budget_summary_drift_are_rejected():
    record = _package()
    for key, expected in (
            ("snapshot", "retest_admission_snapshot_sha"),
            ("spec", "retest_admission_spec_sha"),
            ("budget_revision", "retest_admission_budget_revision_sha")):
        broken = json.loads(json.dumps(record))
        broken[key]["sha256"] = "0" * 64
        problems = A.retest_admission_problems(
            broken, project_root=PROJECT_ROOT, strict=True)
        assert any(expected in p for p in problems), (key, problems)


def test_protected_entry_change_and_missing_child_are_rejected(tmp_path):
    record = _package()
    manifest = json.loads(
        (MATERIALS / "protected_manifest.json").read_text())
    first = next(iter(manifest["entries"]))
    manifest["entries"][first]["sha256"] = "0" * 64
    manifest["manifest_sha256"] = A.canonical_sha256(manifest["entries"])
    changed = tmp_path / "protected_manifest.json"
    A.write_json(changed, manifest)
    broken = json.loads(json.dumps(record))
    broken["protected_manifest"]["path"] = A.path_for_record(changed)
    broken["protected_manifest"]["sha256"] = A.sha256_file(changed)
    problems = A.retest_admission_problems(
        broken, project_root=PROJECT_ROOT, strict=True)
    assert any("protected_manifest_changed" in p for p in problems)
    missing = json.loads(json.dumps(record))
    missing.pop("real_admission", None)
    assert any("real_admission_missing" in p
               for p in A.retest_admission_problems(
                   missing, project_root=PROJECT_ROOT, strict=True))
