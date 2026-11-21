import json
import importlib.util
from pathlib import Path

import pytest

from ireset.eval import w1_d6_shared_gradient_admission as admission


def test_metadata_package_binds_only_existing_records_and_source_roster(monkeypatch):
    seen = []
    real_read = admission._read_metadata

    def record(path, limit=64 << 20):
        path = Path(path)
        assert path.name != "initial.pt"
        assert path.name != "probe_inputs.npz"
        assert "/resources/data/" not in str(path)
        assert "/resources/robots/" not in str(path)
        seen.append(path)
        return real_read(path, limit)

    monkeypatch.setattr(admission, "_read_metadata", record)
    package, draft = admission.build_materials(Path(__file__).resolve().parents[1])

    assert package["executable"] is False
    assert package["authorized"] is False
    assert package["status"]["checkpoint_and_batch_read"] == "not_run"
    assert package["forbidden_calls"] == dict.fromkeys(admission.FORBIDDEN_NOW, 0)
    assert [Path(x["path"]).name for x in package["source"]["selected_d6_files"]] == [
        "initial.pt", "probe_inputs.npz", "payload.json", "optimizer_roster.json",
        "history.jsonl", "probe_index.jsonl",
    ]
    assert package["source"]["selected_payloads_opened"] is False
    assert draft["authorized"] is False and draft["template_only"] is True
    assert draft["checklist"] == dict.fromkeys(admission.CHECKLIST)
    assert len(seen) == 4 + len(package["implementation"]["files"])


def test_authorization_is_rejected_until_real_worker_is_frozen():
    package, draft = admission.build_materials(Path(__file__).resolve().parents[1])
    assert "authorization_not_signed" in admission.authorization_problems(draft, package)
    assert "real_worker_or_supervisor_not_frozen" in admission.authorization_problems(draft, package)
    forged = {**draft, "authorized": True, "template_only": False,
              "attempts": True, "checklist": dict.fromkeys(admission.CHECKLIST, True),
              "approval_reference": "review", "bindings": draft["bindings"]}
    assert "authorization_scope_mismatch" in admission.authorization_problems(forged, package)


def test_supervisor_refuses_before_opening_authorization_or_spawning_worker(capsys):
    path = Path(__file__).resolve().parents[1] / "scripts/supervise_w1_d6_shared_gradient.py"
    spec = importlib.util.spec_from_file_location("shared_gradient_supervisor", path)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)

    assert supervisor.main(["--authorization", "/not/opened", "--authorization-sha256", "0" * 64, "--run"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["worker_spawned"] is False
    assert result["authorization_path_opened"] is False
    assert result["checkpoint_or_batch_reads"] == 0
    assert result["parameter_updates"] == 0
