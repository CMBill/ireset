import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ireset.eval import w1_d6_joint_fit_evidence as evidence
from ireset.eval import w1_d6_shared_gradient as shared
from ireset.eval import w1_d6_shared_gradient_real as real


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
                           "production_assets": {"files": assets}}}

    class Budget:
        def __init__(self):
            self.points = []

        def check(self, point):
            self.points.append(point)

    budget = Budget()
    snapshots, records = real.read_authorized_source_bundle(package, tmp_path, budget)
    assert snapshots == expected
    assert len(records) == 48
    assert len(budget.points) == 96
    assert all(count == 2 for count in Counter(point.rsplit(":", 1)[0]
                                               for point in budget.points).values())


def test_prepare_snapshot_inputs_binds_probe_endpoint_and_root_stats():
    blobs, _ = _fixture_blobs()
    identity = {"launch": "bound-d6"}
    config = __import__("dataclasses").asdict(real.d6real.joint.config(real.evidence.plan_for(False)))
    model, conditioner = nn.Linear(2, 1), nn.Linear(2, 1)
    endpoint = real.state_digest(model.state_dict(), conditioner.state_dict())
    stats = {"mean": np.zeros(3, dtype="<f4"), "std": np.ones(3, dtype="<f4")}
    checkpoint_doc = {"step": 0, "val": {"step": 0}, "identity": identity, "config": config,
                      "checkpoint_usable": True, "parameters_finite": True, "diagnostic_only": False,
                      "failure": None, "formal": None, "root_stats": stats,
                      "model": model.state_dict(), "conditioner": conditioner.state_dict()}
    checkpoint = io.BytesIO()
    torch.save(checkpoint_doc, checkpoint)
    blobs["d6/initial.pt"] = checkpoint.getvalue()
    probe_rows = [json.loads(line) for line in blobs["d6/probe_index.jsonl"].splitlines()]
    probe_rows[0]["endpoint_state_sha256"] = endpoint
    blobs["d6/probe_index.jsonl"] = b"".join(_json(row) for row in probe_rows)
    stat_npz = io.BytesIO()
    np.savez(stat_npz, **stats)
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
    headers = {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in stats.items()}
    package = {"source": {"selected_d6_files": selected, "d6_identity": identity,
                           "d6_config": {key: config[key] for key in ("model_type", "seed", "d_model")},
                           "root_stats": {"file": records[stats_path], "headers": headers}}}
    prepared = real.prepare_snapshot_inputs(package, snapshots, records)
    assert prepared.batch["clip_idx"].tolist() == list(real.PERMUTATION)
    assert prepared.expected_endpoint_sha256 == endpoint
    assert prepared.root_stats["std"].tolist() == [1.0, 1.0, 1.0]
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
