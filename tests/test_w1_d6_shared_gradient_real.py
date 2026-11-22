import io
import builtins
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch
from torch import nn

from ireset.eval import w1_d6_joint_fit_evidence as evidence
from ireset.eval import w1_d6_shared_gradient as shared
from ireset.eval import w1_d6_shared_gradient_real as real


def test_worker_cli_rejects_bad_launch_before_importing_numeric_core(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_w1_d6_shared_gradient_worker.py"
    spec = importlib.util.spec_from_file_location("d6_worker_cli_preflight_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    attempted_numeric_imports = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"ireset.eval.w1_d6_shared_gradient_real", "ireset.eval.w1_d6_shared_gradient"}:
            attempted_numeric_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(OSError):
        cli.execute_worker(
            launch_spec_path=tmp_path / "missing-launch.json",
            admission_snapshot_path=tmp_path / "admission.json",
            authorization_snapshot_path=tmp_path / "authorization.json",
            authorization_sha256="0" * 64,
            project_root=tmp_path,
        )
    assert attempted_numeric_imports == []


def test_worker_preimport_closure_rejects_symlinked_parent(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_w1_d6_shared_gradient_worker.py"
    spec = importlib.util.spec_from_file_location("d6_worker_cli_symlink_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    (external / "module.py").write_text("payload\n")
    (project / "src").symlink_to(external, target_is_directory=True)
    record = (external / "module.py").read_bytes()
    roster = {"src/module.py": {"bytes": len(record), "sha256": cli._sha(record)}}
    with pytest.raises(OSError):
        cli._implementation_closure(project, roster)


def test_worker_cli_rejects_unsigned_snapshot_before_numeric_core_import(tmp_path, monkeypatch):
    source_root = Path(__file__).resolve().parents[1]
    script = source_root / "scripts/run_w1_d6_shared_gradient_worker.py"
    spec = importlib.util.spec_from_file_location("d6_worker_cli_unsigned_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    project = tmp_path / "project"
    attempt = project / "derived/run/attempt_001"
    package_path = project / "src/ireset/eval/w1_d6_shared_gradient_admission.py"
    worker_path = project / "scripts/run_w1_d6_shared_gradient_worker.py"
    package_path.parent.mkdir(parents=True)
    worker_path.parent.mkdir(parents=True)
    attempt.mkdir(parents=True)
    (project / "src/ireset").mkdir(exist_ok=True)
    (project / "src/ireset/__init__.py").write_text("")
    (project / "src/ireset/eval/__init__.py").write_text("")
    package_path.write_bytes((source_root / "src/ireset/eval/w1_d6_shared_gradient_admission.py").read_bytes())
    worker_path.write_bytes(script.read_bytes())
    cli.__file__ = str(worker_path)
    code_path = project / "contract.txt"
    code_path.write_text("pinned metadata only\n")
    files = {}
    for rel, path in (("contract.txt", code_path),
                      ("scripts/run_w1_d6_shared_gradient_worker.py", worker_path)):
        raw = path.read_bytes()
        files[rel] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    worker_launch = {"script_rel": "scripts/run_w1_d6_shared_gradient_worker.py",
                     "executable": sys.executable, "env": dict(os.environ),
                     "status": "reviewed_executable", "execution_mode": "real_worker_v1"}
    package = {"schema": "w1_d6_shared_gradient_real_admission_v1",
               "executable": True, "proposed_real_root": "derived/run/attempt_001",
               "authorization_consumption_ledger": "derived/ledger",
               "implementation": {"files": files}, "worker_launch": worker_launch,
               "historical": {"synthetic_ic_acceptance": {}, "verified_outcome": {}},
               "status": {"real_worker": "implemented_and_reviewed_and_supervised"}}
    package["metadata_sha256"] = cli._canonical_sha(package)
    admission_raw = (json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n").encode()
    auth = {"schema": "w1_d6_shared_gradient_real_admission_v1/authorization_v1",
            "cohort": "w1_d6_shared_gradient_step0_v1", "authorized": False,
            "template_only": True, "includes": ["single_step0_diagnostic"], "attempts": 1,
            "checklist": dict.fromkeys(real.admission.CHECKLIST),
            "approval_reference": None,
            "bindings": {"admission_metadata_sha256": package["metadata_sha256"],
                         "implementation_files": files,
                         "ic_acceptance": {}, "historical_d6_outcome": {},
                         "worker_launch": worker_launch}}
    authorization_raw = (json.dumps(auth, sort_keys=True, separators=(",", ":")) + "\n").encode()
    t0 = time.perf_counter() - .01
    launch_spec = {"schema": "w1_d6_shared_gradient_real_admission_v1/worker_launch_spec_v1",
                   "execution_mode": "real_worker_v1", "launch_id": "a" * 64,
                   "authorization_sha256": hashlib.sha256(authorization_raw).hexdigest(),
                   "authorization_record": {"bytes": len(authorization_raw),
                                            "sha256": hashlib.sha256(authorization_raw).hexdigest()},
                   "admission_metadata_sha256": package["metadata_sha256"],
                   "admission_record": {"bytes": len(admission_raw),
                                        "sha256": hashlib.sha256(admission_raw).hexdigest()},
                   "implementation_pre": files,
                   "implementation_pre_sha256": cli._canonical_sha(files),
                   "authorization_ledger": {"path": "a" * 64 + "/consumed.json", "sha256": "b" * 64},
                   "attempt_root": str(attempt), "project_root": str(project),
                   "t0": t0, "work_deadline": t0 + 240,
                   "hard_deadline": t0 + 300, "parameter_updates": 0}
    (attempt / "worker_launch.json").write_text(json.dumps(launch_spec))
    (attempt / "admission_snapshot.json").write_bytes(admission_raw)
    (attempt / "authorization_snapshot.json").write_bytes(authorization_raw)

    attempted_numeric_imports = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ireset.eval.w1_d6_shared_gradient_real":
            attempted_numeric_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.chdir(attempt)
    with pytest.raises(ValueError, match="worker_authorization_rejected"):
        cli.execute_worker(
            launch_spec_path=attempt / "worker_launch.json",
            admission_snapshot_path=attempt / "admission_snapshot.json",
            authorization_snapshot_path=attempt / "authorization_snapshot.json",
            authorization_sha256=launch_spec["authorization_sha256"],
            project_root=project,
        )
    assert attempted_numeric_imports == []
    assert not (attempt / "d6").exists()


def test_worker_cli_persists_early_failure_beside_supervisor_claim(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/run_w1_d6_shared_gradient_worker.py"
    spec = importlib.util.spec_from_file_location("d6_worker_cli_failure_receipt_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for name in ("worker_launch.json", "admission_snapshot.json", "authorization_snapshot.json"):
        (tmp_path / name).write_text("{}\n")
    monkeypatch.chdir(tmp_path)

    def fail_preflight(**_kwargs):
        raise ValueError("worker_environment_not_package_pinned")

    monkeypatch.setattr(cli, "execute_worker", fail_preflight)
    rc = cli.main(["--launch-spec", str(tmp_path / "worker_launch.json"),
                   "--admission-snapshot", str(tmp_path / "admission_snapshot.json"),
                   "--authorization-snapshot", str(tmp_path / "authorization_snapshot.json"),
                   "--authorization-sha256", "a" * 64, "--project-root", str(tmp_path)])
    assert rc == 2
    failure = json.loads((tmp_path / "worker_failure.json").read_text())
    assert failure["failure_stage"] == "worker_cli_preflight_or_dispatch"
    assert failure["error"] == "worker_environment_not_package_pinned"
    assert failure["parameter_updates"] == 0
    assert json.loads(capsys.readouterr().err)["failure_artifact"]["sha256"] == real._sha(
        (tmp_path / "worker_failure.json").read_bytes())


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _fixture_blobs():
    fields = {}
    arrays = {}
    for slot, window_id in enumerate(shared.WINDOWS):
        fields[window_id] = {}
        for key, (shape, dtype_name) in evidence.INPUT_TABLE.items():
            dtype = np.dtype(dtype_name)
            value = np.zeros(shape, dtype=dtype)
            if key == "box_quat":
                value[:, 0] = 1
            elif key == "valid":
                value[:] = 1
            elif key == "wfc":
                value[3, 3] = 1
            elif key == "clip_idx":
                value[...] = slot
            arrays[f"w{slot}__{key}"] = value
            if key in evidence.BATCH_KEYS:
                fields[window_id][key] = evidence.array_record(value)
    payload = {"schema": real.protocol.SCHEMA + "/payload",
               "identity": {"launch": "bound-d6"},
               "windows": [{"window_id": window_id, "fields": fields[window_id]}
                           for window_id in shared.WINDOWS]}
    positions = list(real.PERMUTATION)
    slots = [{"batch_pos": b, "canonical_slot": slot, "window_id": shared.WINDOWS[slot]}
             for b, slot in enumerate(positions)]
    first_micro = evidence.consumption(payload, positions, kind="train", step=1, micro=1, event=2)
    micros = [first_micro] + [{"kind": "train", "step": 1, "micro": i + 1,
                               "event": 2 + i, "slots": slots}
                              for i in range(1, 4)]
    history = [{"step": 1, "micros": micros}]
    probes = [{"idx": i, "step": i * 10,
               "windows": ([{"window_id": window_id, "batch_pos": j}
                            for j, window_id in enumerate(shared.WINDOWS)] if i == 0 else [])}
              for i in range(101)]
    roster = {"groups": [[{"name": "model.x", "shape": [1], "dtype": "torch.float32"}],
                          [{"name": "conditioner.x", "shape": [1], "dtype": "torch.float32"}]]}
    archive = io.BytesIO()
    np.savez(archive, **arrays)
    raw = {
        "d6/initial.pt": b"checkpoint bytes placeholder",
        "d6/probe_inputs.npz": archive.getvalue(),
        "d6/payload.json": _json(payload),
        "d6/optimizer_roster.json": _json(roster),
        "d6/history.jsonl": _json(history[0]),
        "d6/probe_index.jsonl": b"".join(_json(row) for row in probes),
    }
    records = {name: {"path": name, "bytes": len(value), "sha256": real._sha(value)}
               for name, value in raw.items()}
    return raw, records


def test_reconstructs_registered_first_micro_from_one_bound_snapshot():
    blobs, records = _fixture_blobs()
    items, batch, _, _ = real.reconstruct_batch(blobs, records, expected_identity={"launch": "bound-d6"})
    assert len(items) == 6
    assert batch["x0"].shape == (6, 60, 38)
    assert batch["clip_idx"].tolist() == list(real.PERMUTATION)
    assert batch["start"].tolist() == [0] * 6
    assert set(batch) == set(evidence.INPUT_TABLE)
    with pytest.raises(real.RealWorkerError, match="payload_d6_identity"):
        real.reconstruct_batch(blobs, records, expected_identity={"launch": "other-d6"})


@pytest.mark.parametrize("key,mutate,reason", [
    ("valid", lambda value: value.__setitem__(0, 0), "payload_valid_domain"),
    ("contact", lambda value: value.__setitem__(0, 2), "payload_contact_domain"),
    ("box_quat", lambda value: value.__setitem__((0, 0), 0.), "payload_box_quaternion_domain"),
    ("wfc", lambda value: value.__setitem__((3, 3), 0.), "payload_world_from_clip_domain"),
])
def test_reconstruct_rejects_input_domain_values(key, mutate, reason):
    blobs, records = _fixture_blobs()
    archive = np.load(io.BytesIO(blobs["d6/probe_inputs.npz"]), allow_pickle=False)
    values = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(values[f"w0__{key}"])
    updated = io.BytesIO()
    np.savez(updated, **values)
    blobs["d6/probe_inputs.npz"] = updated.getvalue()
    records["d6/probe_inputs.npz"] = {"path": "d6/probe_inputs.npz",
                                       "bytes": len(blobs["d6/probe_inputs.npz"]),
                                       "sha256": real._sha(blobs["d6/probe_inputs.npz"])}
    payload = json.loads(blobs["d6/payload.json"])
    payload["windows"][0]["fields"][key] = evidence.array_record(values[f"w0__{key}"])
    blobs["d6/payload.json"] = _json(payload)
    records["d6/payload.json"] = {"path": "d6/payload.json",
                                   "bytes": len(blobs["d6/payload.json"]),
                                   "sha256": real._sha(blobs["d6/payload.json"])}
    history = json.loads(blobs["d6/history.jsonl"])
    history["micros"][0] = evidence.consumption(
        payload, list(real.PERMUTATION), kind="train", step=1, micro=1, event=2)
    blobs["d6/history.jsonl"] = _json(history)
    records["d6/history.jsonl"] = {"path": "d6/history.jsonl",
                                    "bytes": len(blobs["d6/history.jsonl"]),
                                    "sha256": real._sha(blobs["d6/history.jsonl"])}
    with pytest.raises(real.RealWorkerError, match=reason):
        real.reconstruct_batch(blobs, records, expected_identity={"launch": "bound-d6"})


def test_rejects_source_byte_drift_before_npz_decode():
    blobs, records = _fixture_blobs()
    blobs["d6/probe_inputs.npz"] = blobs["d6/probe_inputs.npz"][:-1]
    with pytest.raises(real.RealWorkerError, match="approved_input_snapshot_drift"):
        real.reconstruct_batch(blobs, records, expected_identity={"launch": "bound-d6"})


def test_bound_snapshot_checks_digest_and_calls_opener_once():
    raw = b"synthetic-source-bytes"
    calls = []
    spec = {"path": "fixture/source.bin", "bytes": len(raw), "sha256": real._sha(raw)}

    def opener(rel, size):
        calls.append((str(rel), size))
        return raw

    loaded, rec = real.read_bound_snapshot(spec, opener)
    assert loaded == raw and rec["sha256"] == spec["sha256"]
    assert calls == [("fixture/source.bin", len(raw))]
    bad = {**spec, "sha256": "0" * 64}
    with pytest.raises(real.RealWorkerError, match="source_sha256_mismatch"):
        real.read_bound_snapshot(bad, opener)


def test_regular_source_open_rejects_symlink_components(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"source")
    assert real.open_regular_source(tmp_path, Path("target.bin"), 32) == b"source"
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(OSError):
        real.open_regular_source(tmp_path, Path("link.bin"), 32)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"secret")
    nested = tmp_path / "nested"
    nested.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        real.open_regular_source(tmp_path, Path("nested/secret.bin"), 32)


def test_authorized_source_bundle_snapshots_each_approved_file_once(tmp_path):
    d6_rows = []
    expected = {}
    for rel in real.INPUT_RECORDS:
        raw = ("fixture:" + rel).encode()
        source_rel = str(real.admission.SOURCE_ROOT / rel)
        path = tmp_path / source_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        d6_rows.append({"path": source_rel, "bytes": len(raw), "sha256": real._sha(raw)})
        expected[source_rel] = raw
    stats_raw = b"fixture-root-stats"
    stats_rel = "fixture/root_stats.npz"
    (tmp_path / stats_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / stats_rel).write_bytes(stats_raw)
    expected[stats_rel] = stats_raw
    assets = {}
    for index in range(41):
        rel = f"fixture/assets/{index}.bin"
        raw = f"asset:{index}".encode()
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        expected[rel] = raw
        assets[rel] = {"bytes": len(raw), "sha256": real._sha(raw)}
    package = {"source": {"selected_d6_files": d6_rows,
                           "root_stats": {"file": {"path": stats_rel, "bytes": len(stats_raw),
                                                     "sha256": real._sha(stats_raw)},
                                          "headers": {}},
                            "d6_bindings": {"production_assets": {"files": assets}}}}

    class Budget:
        def __init__(self):
            self.points = []

        def check(self, point):
            self.points.append(point)

    budget = Budget()
    io_meter = {"d6_file_reads": 0, "root_stats_file_reads": 0,
                "asset_file_reads": 0, "source_file_reads": 0,
                "source_file_hashes": 0}
    snapshots, records = real.read_authorized_source_bundle(
        package, tmp_path, budget, io_meter=io_meter)
    assert snapshots == expected
    assert len(records) == 48
    assert io_meter == {"d6_file_reads": 6, "root_stats_file_reads": 1,
                        "asset_file_reads": 41, "source_file_reads": 48,
                        "source_file_hashes": 48}
    assert len(budget.points) == 96
    assert all(count == 2 for count in Counter(point.rsplit(":", 1)[0]
                                               for point in budget.points).values())


def test_prepare_snapshot_inputs_binds_probe_endpoint_and_root_stats():
    blobs, _ = _fixture_blobs()
    identity = {"launch": "bound-d6"}
    config = __import__("dataclasses").asdict(real.d6real.joint.config(real.evidence.plan_for(False)))
    model, conditioner = nn.Linear(2, 1), nn.Linear(2, 1)
    endpoint = real.state_digest(model.state_dict(), conditioner.state_dict())
    source_stats = {"mean": np.array([0.1, 0.2, 0.9], dtype="<f8"),
                    "std": np.array([0.5, 0.5, 0.3], dtype="<f8")}
    checkpoint_stats = {key: value.astype("<f4") for key, value in source_stats.items()}
    checkpoint_doc = {"step": 0, "val": {"step": 0}, "identity": identity, "config": config,
                      "checkpoint_usable": True, "parameters_finite": True, "diagnostic_only": False,
                      "failure": None, "formal": None, "root_stats": checkpoint_stats,
                      "model": model.state_dict(), "conditioner": conditioner.state_dict()}
    checkpoint = io.BytesIO()
    torch.save(checkpoint_doc, checkpoint)
    blobs["d6/initial.pt"] = checkpoint.getvalue()
    probe_rows = [json.loads(line) for line in blobs["d6/probe_index.jsonl"].splitlines()]
    probe_rows[0]["endpoint_state_sha256"] = endpoint
    blobs["d6/probe_index.jsonl"] = b"".join(_json(row) for row in probe_rows)
    stat_npz = io.BytesIO()
    np.savez(stat_npz, **source_stats)
    d6_prefix = "fixture/d6/"
    snapshots, records, selected = {}, {}, []
    for rel, raw in blobs.items():
        source_path = str(real.admission.SOURCE_ROOT / rel)
        snapshots[source_path] = raw
        records[source_path] = {"path": source_path, "bytes": len(raw), "sha256": real._sha(raw)}
        selected.append(records[source_path])
    stats_path = "fixture/root_stats.npz"
    snapshots[stats_path] = stat_npz.getvalue()
    records[stats_path] = {"path": stats_path, "bytes": len(stat_npz.getvalue()),
                           "sha256": real._sha(stat_npz.getvalue())}
    headers = {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in source_stats.items()}
    package = {"source": {"selected_d6_files": selected, "d6_identity": identity,
                           "d6_config": {key: config[key] for key in ("model_type", "seed", "d_model")},
                           "root_stats": {"file": records[stats_path], "headers": headers}}}
    prepared = real.prepare_snapshot_inputs(package, snapshots, records)
    assert prepared.batch["clip_idx"].tolist() == list(real.PERMUTATION)
    assert prepared.expected_endpoint_sha256 == endpoint
    assert prepared.root_stats["std"].tolist() == [0.5, 0.5, 0.3]
    victim = selected[2]["path"]
    snapshots[victim] = b"changed payload bytes"
    records[victim] = {"path": victim, "bytes": len(snapshots[victim]), "sha256": real._sha(snapshots[victim])}
    with pytest.raises(real.RealWorkerError, match="admission_d6_pinned_snapshot_mismatch"):
        real.prepare_snapshot_inputs(package, snapshots, records)


def test_json_and_root_stats_reject_ambiguous_or_malformed_bytes():
    with pytest.raises(real.RealWorkerError, match="duplicate_key"):
        real._json_object(b'{"a":1,"a":2}', "fixture_json")
    with pytest.raises(real.RealWorkerError, match="fixture_json"):
        real._json_object(b'{"a":NaN}', "fixture_json")
    valid = io.BytesIO()
    np.savez(valid, mean=np.zeros(3, dtype="<f4"), std=np.ones(3, dtype="<f4"))
    headers = {"mean": {"shape": [3], "dtype": "<f4"},
               "std": {"shape": [3], "dtype": "<f4"}}
    values = real.decode_root_stats(valid.getvalue(), headers)
    assert values["mean"].tolist() == [0.0, 0.0, 0.0]
    bad_std = io.BytesIO()
    np.savez(bad_std, mean=np.zeros(3, dtype="<f4"), std=np.zeros(3, dtype="<f4"))
    with pytest.raises(real.RealWorkerError, match="root_stats_std_nonpositive"):
        real.decode_root_stats(bad_std.getvalue(), headers)


def test_root_stats_accepts_training_float64_source_for_float32_checkpoint():
    archive = io.BytesIO()
    np.savez(archive, mean=np.array([0.1, 0.2, 0.9], dtype="<f8"),
             std=np.array([0.5, 0.5, 0.3], dtype="<f8"))
    headers = {"mean": {"shape": [3], "dtype": "<f8"},
               "std": {"shape": [3], "dtype": "<f8"}}
    values = real.decode_root_stats(archive.getvalue(), headers)
    assert values["mean"].dtype == np.dtype("<f8")
    assert values["std"].dtype == np.dtype("<f8")
    assert np.array_equal(values["mean"].astype("<f4"), np.array([0.1, 0.2, 0.9], dtype="<f4"))


def test_worker_budget_binds_d6_runtime_stage():
    worker_path = Path(__file__).resolve().parents[1] / "scripts/run_w1_d6_shared_gradient_worker.py"
    spec = importlib.util.spec_from_file_location("w1_d6_worker_budget", worker_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    budget = module.WorkerBudget(1.0, 2.0, clock=lambda: 1.0)
    assert budget.stage == "D6"
    assert budget.work_deadline == budget.deadline == 2.0


def test_conditioner_receives_only_registered_condition_fields():
    full_batch = {key: object() for key in shared.INPUT_KEYS}
    condition = real.conditioner_only_input(full_batch)
    assert set(condition) == set(shared.COND_KEYS)
    assert all(condition[key] is full_batch[key] for key in shared.COND_KEYS)
    with pytest.raises(real.RealWorkerError, match="real_conditioner_input_roster"):
        real.conditioner_only_input({key: value for key, value in full_batch.items() if key != "contact"})


def test_step0_runtime_report_binds_each_field_with_specific_failure():
    flags = {key: value for key, value in real.d6real.r6.REQ.items() if key != "device"}
    expected = {"identity": {"cpu_threads": 20}, "gpu": {"uuid": "gpu-0"},
                "observed_before_setup": dict(flags)}
    policy = {"schema": real.d6real.r6.RUNTIME_REPORT, "helper": "runtime_helper.py",
              "tensor_check_points": {"D6": [["fk_ready", "cuda:0"]]}}
    ready = {"flags": flags, "current_device": real.d6real.DEVICE, "cpu_threads": 20}
    report = {"schema": policy["schema"], "stage": "D6", "kind": "production",
              "device": real.d6real.DEVICE, "source": policy["helper"],
              "identity": expected["identity"], "gpu": expected["gpu"],
              "before": dict(flags),
              "setup": {"schema": "w1_real_fk_runtime_v1",
                        "requirements": {k: flags[k] for k in real.d6real.r6.AUDITED_FLAGS},
                        "before": {k: flags[k] for k in real.d6real.r6.AUDITED_FLAGS},
                        "after": {k: flags[k] for k in real.d6real.r6.AUDITED_FLAGS},
                        "applied": True, "problems": []},
              "ready": ready, "endpoint": ready,
              "tensor_checks": [{"point": "fk_ready", "device": "cuda:0",
                                 "float_dtype": flags["default_dtype"], "float_tensors": 1, "tensors": 1}]}
    real.verify_step0_runtime_report("D6", report, expected, policy)
    broken = dict(report, source="other.py")
    with pytest.raises(real.RealWorkerError, match="step0_runtime_report_binding:source"):
        real.verify_step0_runtime_report("D6", broken, expected, policy)


def test_vjp_pair_receipts_bind_request_and_response_entries():
    requests = [{"call_id": i, "request_sha256": f"request-{i}"} for i in range(7)]
    responses = [{"call_id": i, "response_sha256": f"response-{i}"} for i in range(7)]
    pairs = real.vjp_pair_receipts(requests, responses)
    assert pairs == [{"call_id": i, "request_sha256": f"request-{i}",
                      "response_sha256": f"response-{i}"} for i in range(7)]
    with pytest.raises(real.RealWorkerError, match="real_vjp_receipt_count"):
        real.vjp_pair_receipts(requests, responses[:-1])


def test_outcome_array_roster_excludes_late_json_receipts():
    records = {"h_0.npy": {"sha256": "h"}, "g_0.npy": {"sha256": "g"},
               "loss_pos.npy": {"sha256": "p"}, "counts.json": {"sha256": "c"}}
    selected = {name: record for name, record in records.items()
                if name.startswith("h_") or name.startswith("g_") or name.startswith("loss_")}
    assert set(selected) == {"h_0.npy", "g_0.npy", "loss_pos.npy"}


def test_setup_meter_separates_entry_only_scene_observations_from_completed_units():
    class Budget:
        def check(self, _point):
            pass

    meter = real._WorkerMeter(Budget(), {
        "scene_accesses": 1,
        "scene_cache_misses": 1,
        "scene_compiles": 1,
    })
    meter.enter("scene_accesses")
    meter.enter("scene_cache_misses")
    meter.unit("scene_compiles", lambda: "compiled")

    receipt = meter.verify()
    assert receipt["entered"] == {"scene_accesses": 1, "scene_cache_misses": 1,
                                  "scene_compiles": 1}
    assert receipt["completed"] == {"scene_compiles": 1}


def test_rejects_duplicate_or_extra_npz_members():
    blobs, records = _fixture_blobs()
    arrays = {f"w{s}__{key}": np.zeros(shape, dtype=np.dtype(dtype))
              for s in range(6) for key, (shape, dtype) in evidence.INPUT_TABLE.items()}
    arrays["unregistered"] = np.zeros((1,), dtype=np.float32)
    archive = io.BytesIO()
    np.savez(archive, **arrays)
    blobs["d6/probe_inputs.npz"] = archive.getvalue()
    records["d6/probe_inputs.npz"] = {"path": "d6/probe_inputs.npz", "bytes": len(archive.getvalue()),
                                      "sha256": real._sha(archive.getvalue())}
    with pytest.raises(real.RealWorkerError, match="probe_archive_member_count"):
        real.reconstruct_batch(blobs, records, expected_identity={"launch": "bound-d6"})


def test_checkpoint_preflight_and_endpoint_bind_before_strict_load():
    model, conditioner = nn.Linear(2, 1), nn.Linear(2, 1)
    checkpoint = {"step": 0, "val": {"step": 0}, "identity": {"id": "d6"},
                  "config": {"seed": 20261008}, "checkpoint_usable": True,
                  "parameters_finite": True, "diagnostic_only": False, "failure": None, "formal": None,
                  "root_stats": {"mean": np.zeros(3, dtype=np.float32), "std": np.ones(3, dtype=np.float32)},
                  "model": model.state_dict(), "conditioner": conditioner.state_dict()}
    raw = io.BytesIO()
    torch.save(checkpoint, raw)
    loaded = real.checkpoint_snapshot(raw.getvalue())
    endpoint = real.state_digest(loaded["model"], loaded["conditioner"])
    real.validate_initial_checkpoint(loaded, expected_config=checkpoint["config"],
                                     expected_identity=checkpoint["identity"],
                                     expected_endpoint_sha256=endpoint, expected_root_stats=checkpoint["root_stats"],
                                     model=model, conditioner=conditioner)
    with torch.no_grad():
        model.weight.add_(1)
        conditioner.bias.sub_(1)
    receipt = real.strict_load_initial_checkpoint(loaded, model=model, conditioner=conditioner,
                                                  expected_endpoint_sha256=endpoint,
                                                  budget=real.NullBudget(), runtime_guard=_NullRuntime())
    assert receipt["strict_model_loads"] == 2
    assert real.state_digest(model.state_dict(), conditioner.state_dict()) == endpoint
    with pytest.raises(real.RealWorkerError, match="checkpoint_initial_probe_endpoint_mismatch"):
        real.validate_initial_checkpoint(loaded, expected_config=checkpoint["config"],
                                         expected_identity=checkpoint["identity"],
                                         expected_endpoint_sha256="0" * 64,
                                         expected_root_stats=checkpoint["root_stats"],
                                         model=model, conditioner=conditioner)


def test_checkpoint_preflight_rejects_non_float32_state_before_strict_load():
    model, conditioner = nn.Linear(2, 1), nn.Linear(2, 1)
    bad_model = {key: value.double() for key, value in model.state_dict().items()}
    checkpoint = {"step": 0, "val": {"step": 0}, "model": bad_model,
                  "conditioner": conditioner.state_dict(), "root_stats": {"mean": np.zeros(3), "std": np.ones(3)},
                  "identity": {}, "config": {}, "checkpoint_usable": True,
                  "parameters_finite": True, "diagnostic_only": False, "failure": None, "formal": None}
    raw = io.BytesIO()
    torch.save(checkpoint, raw)
    with pytest.raises(real.RealWorkerError, match="initial_checkpoint_tensor_contract:model"):
        real.checkpoint_snapshot(raw.getvalue())


def test_unsigned_admission_cannot_mint_real_worker_permit(tmp_path):
    from ireset.eval import w1_d6_shared_gradient_admission as admission

    package, draft = admission.build_materials(Path(__file__).resolve().parents[1])
    raw = json.dumps(draft, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(real.RealWorkerError, match="real_worker_authorization_rejected"):
        real.authorize_worker(draft, package, authorization_raw=raw,
                              authorization_sha256=real._sha(raw),
                              project_root=Path(__file__).resolve().parents[1])


def test_numeric_kernel_rejects_missing_permit_before_touching_components(tmp_path):
    class Sentinel:
        @property
        def opt(self):
            raise AssertionError("must reject permit before inspecting trainer")

    with pytest.raises(real.RealWorkerError, match="real_worker_permit_required"):
        real.run_numeric_kernel(Sentinel(), {}, None, permit=None, expected_roster=[], trainer_receipt=None,
                                artifact_dir=tmp_path / "artifacts", budget=None, runtime_guard=None,
                                setup_report={})


def _make_review_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "PARAMETER_TENSORS", 2)
    monkeypatch.setattr(shared, "PARAMETER_COUNT", 2)
    analysis = {"decision": "attribution_insufficient", "groups": {}, "h_checks": []}
    monkeypatch.setattr(shared, "analyze_gradients", lambda _bundle: analysis)
    monkeypatch.setattr(shared, "verify_vjp_records", lambda requests, responses, roster: list(range(7)))
    root = tmp_path / "artifacts"
    store = shared.ArtifactStore(root)
    roster = [{"name": "model.weight", "shape": [1], "dtype": "torch.float32", "numel": 1},
              {"name": "conditioner.weight", "shape": [1], "dtype": "torch.float32", "numel": 1}]
    responses, requests, pairs = [], [], []
    for idx in range(7):
        h = np.zeros((6, 60, 38), dtype="<f4")
        g = np.asarray([idx, -idx], dtype="<f4")
        store.write_array(f"h_{idx}.npy", h)
        store.write_array(f"g_{idx}.npy", g)
        response = {"h": shared.array_record(h), "g_flat": shared.array_record(g),
                    "g": [shared.array_record(np.asarray([g[0]], dtype="<f4")),
                          shared.array_record(np.asarray([g[1]], dtype="<f4"))],
                    "call_id": idx}
        requests.append({"call_id": idx, "request_sha256": str(idx) * 64})
        response.update(request_sha256=requests[-1]["request_sha256"], response_sha256=f"{idx:064x}")
        responses.append(response)
        pairs.append({"call_id": idx, "request_sha256": requests[-1]["request_sha256"],
                      "response_sha256": response["response_sha256"]})
    loss_arrays = {
        "pos": np.zeros((6, 60, 3), dtype="<f4"),
        "rot6d": np.zeros((6, 60, 6), dtype="<f4"),
        "joint": np.zeros((6, 60, 29), dtype="<f4"),
        "geodesic_rad2": np.zeros((6, 60), dtype="<f4"),
        "vel": np.zeros((6, 59, 38), dtype="<f4"),
        "fk_m2": np.zeros((6, 60, 15), dtype="<f4"),
    }
    receipts = {"loss_" + key + ".npy": store.write_array("loss_" + key + ".npy", arr)
                for key, arr in loss_arrays.items()}
    independent = shared.independent_loss_scalars(loss_arrays)
    production = {"per_window": [0.] * 6, "terms": {**{key: 0. for key in shared.LOSS_NAMES}, "total": 0.},
                  "weights": dict(shared.WEIGHTS), "batch": 0.}
    loss_records = {"production": production, "independent": independent,
                    "loss_arrays": receipts,
                    "array_records": {key: shared.array_record(arr) for key, arr in loss_arrays.items()}}
    vjp_records = {"requests": requests, "responses": responses, "roster": roster,
                   "roster_sha256": shared.sha256(shared.canonical_json(roster))}
    vjp_counts = {"vjp_entered": 7, "vjp_completed": 7, "vjp_pre_sync_points": 7}
    outcome = {"schema": real.SCHEMA, "status": "candidate_pending_external_return",
               "analysis": analysis, "files": dict(store.records), "vjp_pairs": pairs,
               "vjp_counts": vjp_counts, "parameter_updates": 0, "optimizer_construction": 0,
               "contact_branch": 0,
               "forbidden_update_calls": {name: 0 for name in ("accumulate_micro", "clip_gradients", "step_optimizer")},
               "trainer_construction": {"optimizer_construction": 0, "loader_iterator_creations": 0,
                                         "dataset_iterations": 0},
               "runtime_report": {"stage": "D6", "kind": "production"},
               "runtime_sync_count": 1,
               "rng_unchanged": True, "parameter_state_unchanged": True,
                "independent_loss_scalars": independent, "loss_scalars": production,
                "loss_arrays": receipts, "counts": {"vjp": 7}}
    store.write_json("launch.json", {"schema": real.SCHEMA + "/launch_artifact",
                                      "launch_id": "fixture", "authorization_sha256": "a" * 64,
                                      "admission_metadata_sha256": "b" * 64})
    store.write_json("inputs.json", {"schema": real.SCHEMA + "/inputs_artifact",
                                     "windows": list(shared.WINDOWS), "source_records": {}})
    store.write_json("counts.json", {"schema": real.SCHEMA + "/counts_artifact",
                                     "counts": outcome["counts"], "vjp": vjp_counts,
                                     "runtime_sync_count": 1})
    store.write_json("vjp_records.json", vjp_records)
    store.write_json("loss_scalars.json", loss_records)
    store.write_json("call_matrix.json", {"counts": outcome["counts"], "vjp": vjp_counts,
                                           "runtime_sync_count": 1,
                                           "parameter_updates": 0, "contact_branch": 0})
    store.write_json("outcome.json", outcome)
    return root


def test_real_artifact_readback_is_independent_and_binds_all_outputs(tmp_path, monkeypatch):
    root = _make_review_fixture(tmp_path, monkeypatch)
    report = real.review_real_artifacts(root)
    assert report["passed"] is True and report["vjp_pair_count"] == 7
    with (root / "g_0.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(real.RealWorkerError, match="real_array_file_binding"):
        real.review_real_artifacts(root)


class _NullRuntime:
    def check_unit(self, _point):
        return None
