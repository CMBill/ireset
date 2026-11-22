import json
import importlib.util
from pathlib import Path
import subprocess
import sys

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
    assert len(seen) == 6 + len(package["implementation"]["files"])
    runtime_evidence = package["source"]["d6_contract"]["runtime_evidence"]
    assert {key: value for key, value in runtime_evidence.items()
            if key != "fk_buffer_sources"} == admission.RUNTIME_EVIDENCE
    assert set(runtime_evidence["fk_buffer_sources"]) == {
        "pytorch_kinematics/chain.py", "pytorch_kinematics/frame.py"}
    assert all(record["bytes"] > 0 and len(record["sha256"]) == 64
               for record in runtime_evidence["fk_buffer_sources"].values())


def test_pinned_worker_environment_includes_python_locale_normalization():
    package, _ = admission.build_materials(Path(__file__).resolve().parents[1])
    env = package["worker_launch"]["env"]
    actual = subprocess.check_output(
        [sys.executable, "-I", "-c",
         "import json,os,torch; print(json.dumps({'env':dict(os.environ),'cpu_threads':torch.get_num_threads()},sort_keys=True))"],
        env=env, text=True)
    observed = json.loads(actual)
    assert observed["env"] == env
    assert env["LC_CTYPE"] == "C.UTF-8"
    assert env["OMP_NUM_THREADS"] == str(
        package["source"]["d6_bindings"]["runtime"]["identity"]["cpu_threads"])
    assert observed["cpu_threads"] == package["source"]["d6_bindings"]["runtime"]["identity"]["cpu_threads"]


def test_real_attempt_root_advances_past_consumed_failed_claim():
    package, _ = admission.build_materials(Path(__file__).resolve().parents[1])
    assert package["proposed_real_root"].endswith("/attempt_011")
    assert admission.PROPOSED_REAL_ROOT.name == "attempt_011"
    root = Path(__file__).resolve().parents[1] / admission.PROPOSED_REAL_ROOT
    if root.exists():
        assert (root / "worker_failure.json").is_file()


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


def test_frozen_real_call_and_io_matrices_reject_single_count_drift():
    package, _ = admission.build_materials(Path(__file__).resolve().parents[1])
    assert admission.real_call_matrix_problems(package["call_matrix"]) == []
    assert package["call_matrix"]["setup_units"]["scene_accesses"] == 1
    assert package["call_matrix"]["setup_units"]["scene_cache_misses"] == 1
    assert package["call_matrix"]["setup_units"]["scene_plain_xml_bindings"] == 1
    assert package["call_matrix"]["setup_units"]["scene_plain_compiles"] == 1
    assert admission.real_io_matrix_problems(package) == []
    changed_calls = json.loads(json.dumps(package["call_matrix"]))
    changed_calls["numeric"]["fk_call"] += 1
    assert admission.real_call_matrix_problems(changed_calls) == ["real_call_matrix_not_frozen"]
    changed_io = json.loads(json.dumps(package))
    changed_io["io_matrix"]["worker_actual"]["d6_npz_member_reads"] -= 1
    assert admission.real_io_matrix_problems(changed_io) == ["real_io_worker_counts_not_frozen"]
    changed_asset = json.loads(json.dumps(package))
    changed_asset["io_matrix"]["authorized_future_reads"]["production_assets"].pop()
    assert admission.real_io_matrix_problems(changed_asset) == ["real_io_source_roster_not_frozen"]
