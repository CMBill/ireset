"""Authorized-snapshot adapter and numerical kernel for the D6 study.

There is intentionally no CLI here. The outer supervisor must produce a
reviewed in-memory permit and exact byte snapshots before calling these
functions. Tests use synthetic archives only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import zipfile

import numpy as np
import torch

from ireset.eval import w1_d6_joint_fit_evidence as evidence
from ireset.eval import w1_d6_joint_fit_real as d6real
from ireset.eval import w1_d6_joint_fit_protocol as protocol
from ireset.eval import w1_d6_shared_gradient as shared
from ireset.eval import w1_d6_shared_gradient_admission as admission
from ireset.eval import w1_smallset_fit_diagnostic as fit


SCHEMA = "w1_d6_shared_gradient_real_worker_v1"
INPUT_RECORDS = ("d6/initial.pt", "d6/probe_inputs.npz", "d6/payload.json",
                 "d6/optimizer_roster.json", "d6/history.jsonl", "d6/probe_index.jsonl")
PERMUTATION = (1, 0, 4, 3, 2, 5)  # first native micro: batch position -> registered slot
NPZ_MEMBER_COUNT = len(evidence.INPUT_TABLE) * protocol.N
NPZ_MEMBER_BYTES_CAP = 4 << 20
ROOT_STATS_CAP = 1 << 20
_PERMIT_TOKEN = object()
_TRAINER_TOKEN = object()


class RealWorkerError(RuntimeError):
    pass


class NullBudget:
    def check(self, _point: str) -> None:
        return None


@dataclass(frozen=True)
class RealWorkerPermit:
    root: Path
    launch_id: str
    authorization_sha256: str
    admission_metadata_sha256: str
    _token: object


@dataclass(frozen=True)
class PreparedInputs:
    items: list[dict]
    batch: dict
    expected_roster: list
    payload_manifest: dict
    initial_checkpoint: dict
    root_stats: dict
    expected_identity: dict
    expected_config: dict
    expected_endpoint_sha256: str


@dataclass(frozen=True)
class ReadonlyTrainerReceipt:
    trainer_id: int
    model_id: int
    conditioner_id: int
    dataset_id: int
    construction: dict
    _token: object


def authorize_worker(auth: dict, package: dict, *, authorization_raw: bytes,
                     authorization_sha256: str, project_root: Path, budget=None) -> RealWorkerPermit:
    """Turn a separately verified, executable package into an in-process permit."""
    require(type(authorization_raw) is bytes and admission._sha(authorization_raw) == authorization_sha256
            and _json_object(authorization_raw, "authorization_json_invalid") == auth,
            "authorization_file_hash")
    package_body = {k: v for k, v in package.items() if k != "metadata_sha256"}
    canonical_package = json.dumps(package_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(admission._sha(canonical_package) == package.get("metadata_sha256"),
            "admission_metadata_hash")
    problems = admission.authorization_problems(auth, package)
    require(not problems, "real_worker_authorization_rejected:" + ",".join(problems))
    root = Path(project_root).resolve(strict=True)
    budget = budget or NullBudget()
    file_roster = package.get("implementation", {}).get("files")
    require(type(file_roster) is dict and len(file_roster) >= 1, "worker_implementation_roster")
    for rel, expected in file_roster.items():
        budget.check("worker_implementation_hash:" + rel + ":pre")
        rel_path = Path(rel)
        require(not rel_path.is_absolute() and ".." not in rel_path.parts,
                "worker_implementation_path_invalid")
        raw = open_regular_source(root, rel_path, 32 << 20)
        budget.check("worker_implementation_hash:" + rel + ":post")
        observed = {"path": rel, "bytes": len(raw), "sha256": _sha(raw)}
        require(observed == {"path": rel, "bytes": expected.get("bytes"), "sha256": expected.get("sha256")},
                "worker_implementation_drift:" + rel)
    require(type(auth.get("bindings")) is dict and type(auth.get("launch_id")) is str
            and len(auth["launch_id"]) == 64, "real_worker_authorization_fields")
    require(package.get("proposed_real_root") == str(admission.PROPOSED_REAL_ROOT),
            "worker_real_root_not_pinned")
    return RealWorkerPermit(root / admission.PROPOSED_REAL_ROOT, auth["launch_id"],
                            authorization_sha256, package["metadata_sha256"], _PERMIT_TOKEN)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RealWorkerError(code)


def conditioner_only_input(batch_like: dict) -> dict:
    require(type(batch_like) is dict and set(shared.COND_KEYS) <= set(batch_like),
            "real_conditioner_input_roster")
    condition = {key: batch_like[key] for key in shared.COND_KEYS}
    require(set(condition) == set(shared.COND_KEYS), "real_conditioner_input_roster")
    return condition


def verify_step0_runtime_report(stage, doc, expected, policy) -> None:
    require(type(doc) is dict and doc.get("schema") == policy.get("schema")
            and doc.get("stage") == stage == "D6"
            and doc.get("kind") == "production" and doc.get("device") == d6real.DEVICE,
            "step0_runtime_report_header")
    flags = {key: value for key, value in d6real.r6.REQ.items() if key != "device"}
    checks = (
        ("source", doc.get("source"), policy.get("helper")),
        ("identity", doc.get("identity"), expected.get("identity")),
        ("gpu", doc.get("gpu"), expected.get("gpu")),
        ("before", doc.get("before"), expected.get("observed_before_setup")),
        ("setup", doc.get("setup"), {
            "schema": "w1_real_fk_runtime_v1",
            "requirements": {key: flags[key] for key in d6real.r6.AUDITED_FLAGS},
            "before": {key: doc["before"][key] for key in d6real.r6.AUDITED_FLAGS},
            "after": {key: flags[key] for key in d6real.r6.AUDITED_FLAGS},
            "applied": True, "problems": [],
        } if type(doc.get("before")) is dict and set(d6real.r6.AUDITED_FLAGS) <= set(doc["before"]) else None),
        ("ready", doc.get("ready"), {"flags": flags, "current_device": d6real.DEVICE,
                                      "cpu_threads": expected.get("identity", {}).get("cpu_threads")}),
        ("endpoint", doc.get("endpoint"), {"flags": flags, "current_device": d6real.DEVICE,
                                            "cpu_threads": expected.get("identity", {}).get("cpu_threads")}),
        ("tensor_check_points",
         [[row.get("point"), row.get("device")] for row in doc.get("tensor_checks", [])]
         if type(doc.get("tensor_checks")) is list else None,
         policy.get("tensor_check_points", {}).get("D6")),
    )
    for field, got, want in checks:
        try:
            d6real.p.equal(got, want)
        except Exception as exc:
            raise RealWorkerError("step0_runtime_report_binding:" + field) from exc
    require(all(type(row) is dict and row.get("float_dtype") == flags["default_dtype"]
                and type(row.get("float_tensors")) is int and type(row.get("tensors")) is int
                and 0 < row["float_tensors"] <= row["tensors"] for row in doc["tensor_checks"]),
            "step0_runtime_tensor_check_facts")


def vjp_pair_receipts(requests: list, responses: list) -> list[dict]:
    require(type(requests) is list and type(responses) is list and len(requests) == len(responses) == 7,
            "real_vjp_receipt_count")
    receipts = []
    for request, response in zip(requests, responses, strict=True):
        require(type(request) is dict and type(response) is dict
                and type(request.get("call_id")) is int
                and request.get("call_id") == response.get("call_id")
                and type(request.get("request_sha256")) is str
                and type(response.get("response_sha256")) is str,
                "real_vjp_receipt_binding")
        receipts.append({"call_id": request["call_id"],
                         "request_sha256": request["request_sha256"],
                         "response_sha256": response["response_sha256"]})
    return receipts


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_bound_snapshot(spec: dict, opener) -> tuple[bytes, dict]:
    """Read one already-authorized path exactly once, rejecting path races."""
    require(type(spec) is dict and type(spec.get("path")) is str, "source_spec_invalid")
    rel = Path(spec["path"])
    require(not rel.is_absolute() and ".." not in rel.parts, "source_path_not_relative")
    require(type(spec.get("bytes")) is int and 0 < spec["bytes"] <= 1 << 30
            and type(spec.get("sha256")) is str and len(spec["sha256"]) == 64,
            "source_record_invalid")
    raw = opener(rel, spec["bytes"])
    require(type(raw) is bytes and len(raw) == spec["bytes"], "source_byte_count_mismatch")
    record = {"path": spec["path"], "bytes": len(raw), "sha256": _sha(raw)}
    require(record["sha256"] == spec["sha256"], "source_sha256_mismatch")
    return raw, record


def open_regular_source(project_root: Path, relative: Path, size_cap: int) -> bytes:
    """Read a bounded project file without following symlinks in any component."""
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts,
            "source_path_not_relative")
    require(type(size_cap) is int and 0 < size_cap <= 1 << 30, "source_size_cap_invalid")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    root_fd = os.open(Path(project_root), directory_flags)
    current_fd = root_fd
    owned_fds = []
    try:
        for part in relative.parts[:-1]:
            current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            owned_fds.append(current_fd)
        fd = os.open(relative.parts[-1], file_flags, dir_fd=current_fd)
        owned_fds.append(fd)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= size_cap,
                "source_file_type_or_size")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            require(chunk, "source_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "source_changed_during_read")
        return b"".join(chunks)
    finally:
        for fd in reversed(owned_fds):
            os.close(fd)
        os.close(root_fd)


def read_authorized_source_bundle(package: dict, project_root: Path, budget, *, io_meter=None) -> tuple[dict, dict]:
    """Take the exact authorized snapshots once; never re-open through helpers."""
    source = package.get("source")
    require(type(source) is dict, "admission_source_block")
    d6 = source.get("selected_d6_files")
    stats = source.get("root_stats")
    bindings = source.get("d6_bindings")
    require(type(d6) is list and [row.get("path") for row in d6] ==
            [str(admission.SOURCE_ROOT / rel) for rel in INPUT_RECORDS], "admission_d6_source_roster")
    require(type(stats) is dict and type(stats.get("file")) is dict and type(stats.get("headers")) is dict,
            "admission_root_stats_binding")
    production_assets = bindings.get("production_assets") if type(bindings) is dict else None
    asset_files = production_assets.get("files") if type(production_assets) is dict else None
    require(type(asset_files) is dict and len(asset_files) == 41, "admission_asset_source_roster")
    d6_caps = {"initial.pt": 32 << 20, "probe_inputs.npz": 16 << 20,
               "payload.json": 2 << 20, "optimizer_roster.json": 2 << 20,
               "history.jsonl": 32 << 20, "probe_index.jsonl": 4 << 20}
    specs = [(row, "selected_d6_snapshot:" + row["path"], d6_caps[Path(row["path"]).name])
             for row in d6]
    specs.append((stats["file"], "root_stats_snapshot:" + stats["file"]["path"], ROOT_STATS_CAP))
    specs.extend(({"path": rel, **rec}, "asset_snapshot:" + rel, 32 << 20)
                 for rel, rec in sorted(asset_files.items()))
    require(len({row["path"] for row, _, _ in specs}) == len(specs), "admission_source_paths_duplicate")
    snapshots, records = {}, {}
    root = Path(project_root).resolve(strict=True)
    for spec, label, cap in specs:
        budget.check(label + ":pre")
        require(type(spec.get("bytes")) is int and spec["bytes"] <= cap,
                "authorized_source_size_cap:" + spec["path"])
        raw, record = read_bound_snapshot(
            spec, lambda rel, _requested_size: open_regular_source(root, rel, cap),
        )
        budget.check(label + ":post")
        if io_meter is not None:
            if label.startswith("selected_d6_snapshot:"):
                io_meter["d6_file_reads"] += 1
            elif label.startswith("root_stats_snapshot:"):
                io_meter["root_stats_file_reads"] += 1
            elif label.startswith("asset_snapshot:"):
                io_meter["asset_file_reads"] += 1
            io_meter["source_file_reads"] += 1
            io_meter["source_file_hashes"] += 1
        snapshots[spec["path"]] = raw
        records[spec["path"]] = record
    require(len(snapshots) == 48, "authorized_source_snapshot_count")
    return snapshots, records


def prepare_snapshot_inputs(package: dict, snapshots: dict, records: dict, *, io_meter=None) -> PreparedInputs:
    """Bind source records, reconstruct the first micro, and preflight step 0."""
    source = package.get("source")
    require(type(source) is dict and type(source.get("d6_identity")) is dict
            and type(source.get("d6_config")) is dict, "admission_d6_identity_config")
    expected_config = asdict(d6real.joint.config(evidence.plan_for(False)))
    require(set(source["d6_config"]).issubset(expected_config)
            and all(expected_config[key] == value for key, value in source["d6_config"].items()),
            "admission_d6_config_not_reproducible")
    selected = source.get("selected_d6_files")
    require(type(selected) is list and len(selected) == len(INPUT_RECORDS), "admission_d6_roster")
    blobs, refs = {}, {}
    for spec in selected:
        name = Path(spec["path"]).name
        rel = "d6/" + name
        expected_path = str(admission.SOURCE_ROOT / rel)
        require(spec.get("path") == expected_path, "admission_d6_path_mismatch:" + rel)
        require(rel not in blobs and spec["path"] in snapshots and spec["path"] in records,
                "admission_d6_snapshot_missing:" + rel)
        observed = records[spec["path"]]
        require(observed.get("bytes") == spec.get("bytes")
                and observed.get("sha256") == spec.get("sha256")
                and len(snapshots[spec["path"]]) == spec.get("bytes")
                and _sha(snapshots[spec["path"]]) == spec.get("sha256"),
                "admission_d6_pinned_snapshot_mismatch:" + rel)
        blobs[rel] = snapshots[spec["path"]]
        refs[rel] = {"path": rel, "bytes": spec["bytes"], "sha256": spec["sha256"]}
    require(set(blobs) == set(INPUT_RECORDS), "admission_d6_snapshot_coverage")
    items, batch, roster, payload = reconstruct_batch(blobs, refs, expected_identity=source["d6_identity"],
                                                       io_meter=io_meter)
    stats_spec = source["root_stats"]["file"]
    require(stats_spec["path"] in snapshots and stats_spec["path"] in records,
            "admission_root_stats_snapshot_missing")
    require(records[stats_spec["path"]]["sha256"] == stats_spec["sha256"]
            and records[stats_spec["path"]]["bytes"] == stats_spec["bytes"]
            and len(snapshots[stats_spec["path"]]) == stats_spec["bytes"]
            and _sha(snapshots[stats_spec["path"]]) == stats_spec["sha256"],
            "admission_root_stats_binding")
    stats = decode_root_stats(snapshots[stats_spec["path"]], source["root_stats"]["headers"],
                              io_meter=io_meter)
    if io_meter is not None:
        io_meter["checkpoint_cpu_loads"] += 1
    checkpoint = checkpoint_snapshot(blobs["d6/initial.pt"])
    require(checkpoint.get("identity") == source["d6_identity"], "checkpoint_d6_identity")
    require(checkpoint.get("config") == expected_config, "checkpoint_config_mismatch")
    require(checkpoint.get("step") == 0 and checkpoint.get("val") == {"step": 0}
            and checkpoint.get("checkpoint_usable") is True and checkpoint.get("parameters_finite") is True
            and checkpoint.get("diagnostic_only") is False and checkpoint.get("failure") is None
            and checkpoint.get("formal") is None, "checkpoint_not_usable")
    for key in ("mean", "std"):
        actual = checkpoint.get("root_stats", {}).get(key)
        try:
            actual_raw = np.asarray(actual)
            expected_raw = np.asarray(stats[key])
            require(np.issubdtype(actual_raw.dtype, np.number)
                    and actual_raw.dtype != np.dtype(bool)
                    and np.issubdtype(expected_raw.dtype, np.number),
                    "checkpoint_root_stats_type:" + key)
            actual_f32 = np.ascontiguousarray(actual_raw, dtype=np.float32)
            expected_f32 = np.ascontiguousarray(expected_raw, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealWorkerError("checkpoint_root_stats_unreadable:" + key) from exc
        require(np.isfinite(actual_f32).all()
                and actual_f32.shape == expected_f32.shape
                and np.array_equal(actual_f32.view(np.uint32), expected_f32.view(np.uint32)),
                "checkpoint_root_stats_mismatch:" + key)
    probe_first = _jsonl_first(blobs["d6/probe_index.jsonl"], "probe_index_invalid")
    endpoint = probe_first.get("endpoint_state_sha256")
    require(type(endpoint) is str and len(endpoint) == 64 and
            all(c in "0123456789abcdef" for c in endpoint), "initial_endpoint_sha_invalid")
    return PreparedInputs(items, batch, roster, payload, checkpoint, stats, source["d6_identity"],
                          expected_config, endpoint)


def _json_object(raw: bytes, code: str) -> dict:
    def no_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RealWorkerError(code + "_duplicate_key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=no_duplicate_keys,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealWorkerError(code) from exc
    except ValueError as exc:
        raise RealWorkerError(code) from exc
    require(type(value) is dict, code + "_not_object")
    return value


def _jsonl_first(raw: bytes, code: str) -> dict:
    require(type(raw) is bytes, code + "_bytes_invalid")
    first = next((line for line in raw.splitlines() if line.strip()), None)
    require(first is not None, code + "_empty")
    return _json_object(first, code + "_first_record_invalid")


def _archive_members(raw: bytes, expected: set[str], *, io_meter=None) -> dict[str, np.ndarray]:
    require(type(raw) is bytes and len(raw) <= NPZ_MEMBER_BYTES_CAP * NPZ_MEMBER_COUNT,
            "probe_archive_byte_cap")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            if io_meter is not None:
                io_meter["npz_archive_opens"] += 1
            infos = archive.infolist()
            require(len(infos) == NPZ_MEMBER_COUNT, "probe_archive_member_count")
            names = [info.filename[:-4] if info.filename.endswith(".npy") else "" for info in infos]
            require(len(set(names)) == len(names) and set(names) == expected, "probe_archive_member_roster")
            require(all(0 < info.file_size <= NPZ_MEMBER_BYTES_CAP for info in infos), "probe_member_size_cap")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if io_meter is not None:
                io_meter["npz_archive_opens"] += 1
            require(set(archive.files) == expected and len(archive.files) == NPZ_MEMBER_COUNT,
                    "probe_archive_loaded_roster")
            arrays = {}
            for name in sorted(expected):
                arr = archive[name]
                if io_meter is not None:
                    io_meter["d6_npz_member_reads"] += 1
                require(arr.dtype == np.dtype("<f4") or arr.dtype == np.dtype("<i8"),
                        "probe_member_dtype:" + name)
                require(arr.flags.c_contiguous and np.isfinite(arr).all(), "probe_member_domain:" + name)
                arrays[name] = arr
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        if isinstance(exc, RealWorkerError):
            raise
        raise RealWorkerError("probe_archive_unreadable") from exc
    return arrays


def _expected_npz_roster() -> set[str]:
    return {f"w{slot}__{key}" for slot in range(protocol.N) for key in evidence.INPUT_TABLE}


def decode_root_stats(raw: bytes, headers: dict, *, io_meter=None) -> dict[str, np.ndarray]:
    """Decode pinned FP32/FP64 statistics from one already-hashed NPZ blob."""
    require(type(raw) is bytes and 0 < len(raw) <= ROOT_STATS_CAP, "root_stats_archive_size")
    require(type(headers) is dict and set(headers) == {"mean", "std"}, "root_stats_header_roster")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            if io_meter is not None:
                io_meter["npz_archive_opens"] += 1
            infos = archive.infolist()
            require(len(infos) == 2 and {i.filename for i in infos} == {"mean.npy", "std.npy"},
                    "root_stats_archive_members")
            require(all(0 < i.file_size <= 4096 for i in infos), "root_stats_member_size")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if io_meter is not None:
                io_meter["npz_archive_opens"] += 1
            values = {}
            for key in ("mean", "std"):
                value = archive[key]
                if io_meter is not None:
                    io_meter["root_stats_npz_member_reads"] += 1
                header = headers[key]
                require(type(header) is dict and list(value.shape) == header.get("shape")
                        and header.get("dtype") in {"<f4", "<f8"}
                        and value.dtype.str == header.get("dtype")
                        and value.dtype in (np.dtype("<f4"), np.dtype("<f8"))
                        and value.flags.c_contiguous and np.isfinite(value).all(),
                        "root_stats_value_contract:" + key)
                values[key] = np.array(value, copy=True)
            require((values["std"] > 0).all(), "root_stats_std_nonpositive")
    except RealWorkerError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        raise RealWorkerError("root_stats_archive_unreadable") from exc
    return values


def _first_micro_positions(row: dict, payload: dict) -> list[int]:
    require(type(row.get("step")) is int and row.get("step") == 1
            and type(row.get("micros")) is list and len(row["micros"]) == 4,
            "history_first_update_contract")
    micro = row["micros"][0]
    require(type(micro) is dict and micro.get("kind") == "train"
            and type(micro.get("micro")) is int and micro.get("micro") == 1
            and type(micro.get("event")) is int and micro.get("event") == 2,
            "history_first_micro_contract")
    slots = micro.get("slots")
    require(type(slots) is list and len(slots) == protocol.N, "history_first_micro_slots")
    windows = payload.get("windows")
    require(type(windows) is list and len(windows) == protocol.N, "payload_window_roster")
    by_id = {w.get("window_id"): i for i, w in enumerate(windows) if type(w) is dict}
    positions = []
    for batch_pos, slot in enumerate(slots):
        require(type(slot) is dict and type(slot.get("batch_pos")) is int
                and slot.get("batch_pos") == batch_pos and type(slot.get("canonical_slot")) is int,
                "history_batch_position")
        canonical = by_id.get(slot.get("window_id"))
        require(canonical is not None and canonical == slot.get("canonical_slot"), "history_window_binding")
        positions.append(canonical)
    require(tuple(positions) == PERMUTATION and sorted(positions) == list(range(protocol.N)),
            "history_first_micro_permutation")
    return positions


def _validate_reconstructed_domain(arrays: dict[str, np.ndarray], slot: int) -> None:
    """Reject source-domain violations before any production module work."""
    require(set(arrays) == set(evidence.BATCH_KEYS), "payload_key_set")
    for key, value in arrays.items():
        require(isinstance(value, np.ndarray) and value.dtype == np.float32
                and np.isfinite(value).all(), f"payload_shape_dtype_domain:{slot}:{key}")
    require(np.isin(arrays["contact"], (0., 1.)).all(),
            f"payload_contact_domain:{slot}")
    require(np.array_equal(arrays["valid"], np.ones_like(arrays["valid"])),
            f"payload_valid_domain:{slot}")
    require(np.allclose(np.linalg.norm(arrays["box_quat"], axis=-1), 1., rtol=0, atol=1e-5),
            f"payload_box_quaternion_domain:{slot}")
    require(np.array_equal(arrays["wfc"][3], np.asarray([0., 0., 0., 1.], dtype=np.float32)),
            f"payload_world_from_clip_domain:{slot}")


def reconstruct_batch(blobs: dict[str, bytes], records: dict[str, dict], *,
                      expected_identity: dict, io_meter=None) -> tuple[list[dict], dict, list, dict]:
    """Decode only the six approved records and rebuild the first micro batch."""
    require(set(blobs) == set(INPUT_RECORDS), "approved_input_snapshot_roster")
    require(set(records) == set(INPUT_RECORDS), "approved_input_record_roster")
    for rel, raw in blobs.items():
        require(type(raw) is bytes and _sha(raw) == records[rel].get("sha256")
                and len(raw) == records[rel].get("bytes"), "approved_input_snapshot_drift:" + rel)
    payload = _json_object(blobs["d6/payload.json"], "payload_invalid")
    roster_doc = _json_object(blobs["d6/optimizer_roster.json"], "optimizer_roster_invalid")
    history_first = _jsonl_first(blobs["d6/history.jsonl"], "history_invalid")
    probe_first = _jsonl_first(blobs["d6/probe_index.jsonl"], "probe_index_invalid")
    require(type(probe_first.get("idx")) is int and probe_first.get("idx") == 0
            and type(probe_first.get("step")) is int and probe_first.get("step") == 0,
            "probe_index_initial_endpoint")
    require(payload.get("schema") == protocol.SCHEMA + "/payload", "payload_schema")
    groups = roster_doc.get("groups")
    evidence.validate_roster(groups)
    initial_probe = probe_first
    require(type(initial_probe.get("windows")) is list and
            all(type(w) is dict and type(w.get("batch_pos")) is int for w in initial_probe["windows"])
            and [(w.get("window_id"), w.get("batch_pos")) for w in initial_probe["windows"]] ==
            [(window_id, pos) for pos, window_id in enumerate(shared.WINDOWS)],
            "probe_index_registered_windows")
    windows = payload.get("windows")
    require(payload.get("identity") == expected_identity, "payload_d6_identity")
    require(type(windows) is list and [w.get("window_id") for w in windows] == list(shared.WINDOWS),
            "payload_registered_window_order")
    positions = _first_micro_positions(history_first, payload)
    expected_consumption = evidence.consumption(payload, positions, kind="train", step=1,
                                                micro=1, event=2)
    protocol.equal(history_first["micros"][0], expected_consumption)
    arrays = _archive_members(blobs["d6/probe_inputs.npz"], _expected_npz_roster(), io_meter=io_meter)
    items = []
    field_names = tuple(evidence.BATCH_KEYS)
    for slot, window in enumerate(windows):
        fields = window.get("fields")
        require(type(fields) is dict and set(fields) == set(field_names), "payload_field_roster")
        item = {}
        for key in evidence.INPUT_TABLE:
            arr = arrays[f"w{slot}__{key}"]
            shape, dtype = evidence.INPUT_TABLE[key]
            require(tuple(arr.shape) == tuple(shape) and arr.dtype == np.dtype(dtype),
                    f"probe_input_contract:{slot}:{key}")
            if key in field_names:
                require(evidence.array_record(arr) == fields[key], f"payload_field_binding:{slot}:{key}")
            item[key] = torch.from_numpy(np.array(arr, copy=True))
        _validate_reconstructed_domain(
            {key: item[key].numpy() for key in evidence.BATCH_KEYS}, slot)
        require(int(item["clip_idx"]) == slot and int(item["start"]) == 0,
                "probe_input_slot_identity")
        items.append(item)
    ordered_items = [items[i] for i in positions]
    batch = torch.utils.data.default_collate(ordered_items)
    require(set(batch) == set(evidence.INPUT_TABLE), "real_batch_field_roster")
    require(batch["clip_idx"].tolist() == list(PERMUTATION) and batch["start"].tolist() == [0] * protocol.N,
            "real_batch_first_micro_order")
    roster_doc.get("groups")
    return items, batch, groups, payload


def checkpoint_snapshot(raw: bytes, *, torch_module=torch):
    """Restricted CPU deserialization; structural checks happen before strict copy."""
    require(type(raw) is bytes and 0 < len(raw) <= 32 << 20, "initial_checkpoint_size")
    try:
        from numpy._core.multiarray import _reconstruct

        safe_numpy_types = [_reconstruct, np.ndarray, np.dtype,
                            type(np.dtype("float32")), type(np.dtype("float64"))]
        with torch_module.serialization.safe_globals(safe_numpy_types):
            checkpoint = torch_module.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RealWorkerError("initial_checkpoint_unreadable") from exc
    require(type(checkpoint) is dict and type(checkpoint.get("step")) is int and checkpoint.get("step") == 0
            and checkpoint.get("val") == {"step": 0} and isinstance(checkpoint.get("model"), dict)
            and isinstance(checkpoint.get("conditioner"), dict), "initial_checkpoint_contract")
    for side in ("model", "conditioner"):
        state = checkpoint[side]
        require(state and all(torch_module.is_tensor(v) and v.device.type == "cpu"
                              and (not v.is_floating_point() or (v.dtype is torch_module.float32
                                                               and bool(torch_module.isfinite(v).all())))
                              for v in state.values()), "initial_checkpoint_tensor_contract:" + side)
    return checkpoint


def state_digest(model_state: dict, conditioner_state: dict, budget=None) -> str:
    return evidence.endpoint_state_digest(model_state, conditioner_state, budget or NullBudget())


def validate_initial_checkpoint(checkpoint: dict, *, expected_config: dict, expected_identity: dict,
                                expected_endpoint_sha256: str, expected_root_stats: dict,
                                model, conditioner) -> None:
    require(checkpoint.get("identity") == expected_identity, "checkpoint_d6_identity")
    require(checkpoint.get("config") == expected_config, "checkpoint_config_mismatch")
    require(checkpoint.get("checkpoint_usable") is True and checkpoint.get("parameters_finite") is True
            and checkpoint.get("diagnostic_only") is False
            and checkpoint.get("failure") is None and checkpoint.get("formal") is None,
            "checkpoint_not_usable")
    root_stats = checkpoint.get("root_stats")
    require(type(root_stats) is dict and set(root_stats) == {"mean", "std"}, "checkpoint_root_stats_roster")
    for key in ("mean", "std"):
        try:
            actual_raw = np.asarray(root_stats[key])
            expected_raw = np.asarray(expected_root_stats[key])
            require(np.issubdtype(actual_raw.dtype, np.number) and actual_raw.dtype != np.dtype(bool)
                    and np.issubdtype(expected_raw.dtype, np.number), "checkpoint_root_stats_type:" + key)
            actual = np.ascontiguousarray(actual_raw, dtype=np.float32)
            expected = np.ascontiguousarray(expected_raw, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RealWorkerError("checkpoint_root_stats_unreadable:" + key) from exc
        require(np.isfinite(actual).all(), "checkpoint_root_stats_nonfinite:" + key)
        require(actual.shape == expected.shape and np.array_equal(actual.view(np.uint32), expected.view(np.uint32)),
                "checkpoint_root_stats_mismatch:" + key)
    destinations = (("model", model.state_dict()), ("conditioner", conditioner.state_dict()))
    for side, destination in destinations:
        state = checkpoint[side]
        require(state.keys() == destination.keys(), "checkpoint_state_key_roster:" + side)
        for name, target in destination.items():
            source = state[name]
            require(source.shape == target.shape and source.dtype == target.dtype,
                    "checkpoint_tensor_shape_dtype:" + side + ":" + name)
    digest = state_digest(checkpoint["model"], checkpoint["conditioner"])
    require(digest == expected_endpoint_sha256, "checkpoint_initial_probe_endpoint_mismatch")


def strict_load_initial_checkpoint(checkpoint: dict, *, model, conditioner,
                                   expected_endpoint_sha256: str, budget,
                                   runtime_guard) -> dict:
    """Strictly copy the already-preflighted step-0 state into fresh modules."""
    loaded = {}
    strict_loads = 0
    for side, module in (("model", model), ("conditioner", conditioner)):
        runtime_guard.check_unit("initial_strict_load_" + side + ":pre")
        budget.check("initial_strict_load_" + side + ":pre")
        live = module.state_dict()
        source = checkpoint[side]
        require(live.keys() == source.keys(), "checkpoint_state_key_roster:" + side)
        for name, target in live.items():
            value = source[name]
            require(value.shape == target.shape and value.dtype == target.dtype
                    and value.device.type == "cpu", "checkpoint_tensor_contract:" + side + ":" + name)
        result = module.load_state_dict(source, strict=True)
        strict_loads += 1
        require(not result.missing_keys and not result.unexpected_keys,
                "checkpoint_strict_load_result:" + side)
        budget.check("initial_strict_load_" + side + ":post")
        runtime_guard.check_unit("initial_strict_load_" + side + ":post")
        loaded[side] = module.state_dict()
    digest = state_digest(loaded["model"], loaded["conditioner"], budget)
    require(digest == expected_endpoint_sha256, "loaded_initial_probe_endpoint_mismatch")
    return {"step": 0, "endpoint_state_sha256": digest, "strict_model_loads": strict_loads,
            "missing_keys": [], "unexpected_keys": []}


def build_readonly_trainer(prepared: PreparedInputs, *, permit: RealWorkerPermit,
                           tokenizer, loss, run_dir: Path, plan, budget, runtime_guard,
                           construction_entry, construction_check):
    """Construct the exact production Trainer without optimizer or iteration."""
    require(isinstance(permit, RealWorkerPermit) and permit._token is _PERMIT_TOKEN
            and Path(run_dir) == permit.root / "d6", "real_worker_permit_required")
    run_dir = Path(run_dir)
    require(run_dir.is_dir() and not run_dir.is_symlink(), "real_worker_d6_directory_unowned")
    budget.check("trainer_construct:pre")
    runtime_guard.check_unit("trainer_construct:pre")
    cfg = shared.TrainConfig(**{k: getattr(plan, k) for k in fit.CHECKPOINT_CONFIG_FIELDS},
                             num_workers=0, device="cuda:0",
                             loss=shared.LossWeights(w_contact=0.))
    cfg_doc = asdict(cfg)
    require(type(prepared.expected_config) is dict
            and set(prepared.expected_config).issubset(cfg_doc)
            and all(cfg_doc[key] == value for key, value in prepared.expected_config.items()),
            "real_trainer_config_mismatch")
    dataset = shared.NoReadRegisteredWindowDataset(prepared.items, window_keys=None)
    dataset_constructions = 1
    allowed = ("model", "model_transfer", "conditioner", "conditioner_transfer",
               "train_loader", "val_loader")
    entered, returned = [], []

    def enter(name):
        require(name in allowed and name not in entered, "real_trainer_construction_extra:" + name)
        runtime_guard.check_unit("trainer_construct_" + name + ":pre")
        budget.check("trainer_construct_" + name + ":pre")
        construction_entry(name)
        entered.append(name)

    def check(name, module):
        budget.check("trainer_construct_" + name + ":post")
        runtime_guard.check_unit("trainer_construct_" + name + ":post")
        construction_check(name, module)
        returned.append(name)

    trainer_constructions = 1
    trainer = shared.Trainer(cfg, tokenizer, dataset, dataset, run_dir,
                             budget_deadline=budget.work_deadline, loss_fn=loss,
                             build_optimizer=False, construction_entry=enter,
                             construction_check=check)
    require(entered == list(allowed) and returned == ["model", "model_transfer", "conditioner", "conditioner_transfer"],
            "real_trainer_construction_matrix")
    budget.check("trainer_construct:post")
    runtime_guard.check_unit("trainer_construct:post")
    require(trainer.opt is None and trainer.train_loader.dataset is dataset
            and trainer.val_loader.dataset is dataset and trainer.train_loader.num_workers == 0
            and trainer.val_loader.num_workers == 0 and trainer.train_loader.batch_size == 6
            and trainer.val_loader.batch_size == 6 and trainer.train_loader._iterator is None
            and trainer.val_loader._iterator is None,
            "real_trainer_readonly_loader_contract")
    trainer.model.train()
    trainer.conditioner.train()
    construction = {"construction_entered": entered, "construction_returned": returned,
                    "dataset_constructions": dataset_constructions,
                    "trainer_constructions": trainer_constructions,
                    "optimizer_construction": 0, "loader_constructions": 2,
                    "loader_iterator_creations": 0, "dataset_iterations": 0}
    receipt = ReadonlyTrainerReceipt(id(trainer), id(trainer.model), id(trainer.conditioner),
                                     id(dataset), construction, _TRAINER_TOKEN)
    return trainer, receipt


class _WorkerMeter:
    ENTRY_ONLY = frozenset({"scene_accesses", "scene_cache_misses", "scene_cache_hits"})
    BUDGET_METERED = frozenset({"fk_runtime_sources"})

    def __init__(self, budget, expected):
        self.budget = budget
        self.expected = dict(expected)
        self.entered = {}
        self.completed = {}

    def enter(self, name):
        self.budget.check(name + ":pre")
        require(name in self.expected, "worker_setup_call_not_registered:" + name)
        count = self.entered.get(name, 0) + 1
        require(count <= self.expected[name], "worker_setup_call_extra:" + name)
        self.entered[name] = count

    def unit(self, name, fn, /, *args, point=None, **kwargs):
        label = point or name
        self.enter(name)
        result = fn(*args, **kwargs)
        self.budget.check(label + ":post")
        self.completed[name] = self.completed.get(name, 0) + 1
        return result

    def verify(self):
        metered_expected = {name: count for name, count in self.expected.items()
                            if name not in self.BUDGET_METERED}
        expected_completed = {name: count for name, count in metered_expected.items()
                              if name not in self.ENTRY_ONLY}
        external = {name: count for name, count in self.expected.items()
                    if name in self.BUDGET_METERED}
        budget_calls = getattr(self.budget, "units", {})
        require(self.entered == metered_expected and self.completed == expected_completed
                and all(budget_calls.get(name, 0) == count for name, count in external.items()),
                "worker_setup_call_matrix_mismatch")
        return {"expected": self.expected, "entered": {**self.entered, **external},
                "completed": dict(self.completed)}


class _GuardedArtifactStore(shared.ArtifactStore):
    def __init__(self, root, *, budget, runtime_guard, **kwargs):
        self.budget, self.runtime_guard = budget, runtime_guard
        self.io_counts = {"npy_writes": 0, "json_writes": 0}
        super().__init__(root, **kwargs)

    def write_array(self, name, value):
        self.budget.check("artifact_write:" + name + ":pre")
        record = super().write_array(name, value)
        self.io_counts["npy_writes"] += 1
        self.budget.check("artifact_write:" + name + ":post")
        return record

    def write_json(self, name, value):
        self.budget.check("artifact_write:" + name + ":pre")
        record = super().write_json(name, value)
        self.io_counts["json_writes"] += 1
        self.budget.check("artifact_write:" + name + ":post")
        return record


class _NullRuntime:
    def check_unit(self, _point):
        return None


@dataclass(frozen=True)
class WorkerComponents:
    trainer: object
    batch: dict
    loss: object
    runtime: object
    trainer_receipt: ReadonlyTrainerReceipt
    setup_report: dict
    setup_meter: object


def setup_production_worker(package: dict, snapshots: dict, records: dict, *,
                            prepared: PreparedInputs, permit: RealWorkerPermit, run_dir: Path,
                            budget, construction_entry, construction_check, setup_meter) -> WorkerComponents:
    """Create production apparatus/modules, strict-load step 0, and transfer one batch."""
    require(isinstance(permit, RealWorkerPermit) and permit._token is _PERMIT_TOKEN,
            "real_worker_permit_required")
    plan = evidence.plan_for(False)
    registration = evidence.registration_for(False)
    cfg = d6real.joint.config(plan)
    require(asdict(cfg) == prepared.expected_config, "real_plan_config_not_bound")
    require([window.window_id for window in registration.windows] == list(shared.WINDOWS),
            "real_registration_window_order")
    app, _spec, runtime, _meter = build_production_apparatus(package, snapshots, records,
                                                              prepared=prepared, budget=budget,
                                                              meter=setup_meter)
    for slot, item in enumerate(prepared.items):
        arrays = {key: item[key].detach().cpu().numpy() for key in evidence.BATCH_KEYS}
        setup_meter.unit("input_domain_validations", evidence.valid_window, arrays, app.decoder,
                         point="input_domain_validation:" + str(slot))
    trainer, receipt = build_readonly_trainer(prepared, permit=permit, tokenizer=app.tokenizer,
                                               loss=app.loss_fn, run_dir=run_dir, plan=plan,
                                               budget=budget, runtime_guard=runtime,
                                               construction_entry=construction_entry,
                                               construction_check=construction_check)
    runtime.inspect("initial_checkpoint", d6real.tensors.checkpoint_facts,
                    prepared.initial_checkpoint, "cpu")
    validate_initial_checkpoint(prepared.initial_checkpoint, expected_config=asdict(trainer.cfg),
                                expected_identity=prepared.expected_identity,
                                expected_endpoint_sha256=prepared.expected_endpoint_sha256,
                                expected_root_stats=prepared.root_stats,
                                model=trainer.model, conditioner=trainer.conditioner)
    loaded = strict_load_initial_checkpoint(prepared.initial_checkpoint, model=trainer.model,
                                           conditioner=trainer.conditioner,
                                           expected_endpoint_sha256=prepared.expected_endpoint_sha256,
                                           budget=budget, runtime_guard=runtime)
    runtime.inspect("loaded_model", d6real.tensors.module_facts, trainer.model, "cuda:0")
    runtime.inspect("loaded_conditioner", d6real.tensors.module_facts, trainer.conditioner, "cuda:0")
    runtime.check_unit("loaded_endpoint")
    batch = {}
    for key, value in prepared.batch.items():
        runtime.check_unit("batch_transfer:" + key + ":pre")
        budget.check("batch_transfer:" + key + ":pre")
        require(torch.is_tensor(value) and value.device.type == "cpu", "real_cpu_batch_contract:" + key)
        moved = value.to("cuda:0", copy=True)
        torch.cuda.synchronize("cuda:0")
        budget.check("batch_transfer:" + key + ":post")
        runtime.check_unit("batch_transfer:" + key + ":post")
        batch[key] = moved
    require(batch["clip_idx"].tolist() == list(PERMUTATION)
            and batch["start"].tolist() == [0] * protocol.N, "real_transferred_batch_order")
    for key in evidence.BATCH_KEYS:
        host = batch[key].detach().cpu().numpy()
        for batch_pos, canonical_slot in enumerate(PERMUTATION):
            require(evidence.array_record(host[batch_pos]) ==
                    prepared.payload_manifest["windows"][canonical_slot]["fields"][key],
                    "real_transferred_batch_payload:" + key)
    setup_report = {"app_runtime": runtime.facts(), "strict_load": loaded,
                    "trainer_construction": receipt.construction,
                    "setup_calls": {"expected": setup_meter.expected,
                                    "entered": dict(setup_meter.entered),
                                    "completed": dict(setup_meter.completed)},
                    "input_tensor_transfers": len(batch), "optimizer_construction": 0,
                    "parameter_updates": 0, "loader_iterations": 0}
    return WorkerComponents(trainer, batch, app.loss_fn, runtime, receipt, setup_report, setup_meter)


def build_production_apparatus(package: dict, snapshots: dict, records: dict, *,
                               prepared: PreparedInputs, budget, meter=None):
    """Build the existing D6 production FK/loss stack from the verified bytes."""
    source = package.get("source")
    require(type(source) is dict and type(source.get("d6_bindings")) is dict
            and type(source.get("d6_contract")) is dict, "admission_d6_production_bindings")
    bindings = source["d6_bindings"]
    assets = bindings.get("production_assets", {}).get("files")
    require(type(assets) is dict and len(assets) == 41, "admission_production_asset_roster")
    raw_assets = {}
    for rel in assets:
        require(rel in snapshots and rel in records, "production_asset_snapshot_missing:" + rel)
        raw_assets[rel] = snapshots[rel]
    root_spec = bindings["root_stats"]["file"]
    require(root_spec["path"] in records, "root_stats_record_missing")
    root_record = records[root_spec["path"]]
    meter = meter or _WorkerMeter(budget, package["call_matrix"]["setup_units"])
    runtime = d6real.Runtime(meter, meter, bindings["runtime"], source["d6_contract"]["runtime_evidence"])
    app, spec = d6real.apparatus(bindings, meter, runtime, preloaded_assets=raw_assets,
                                 preloaded_root_stats=(prepared.root_stats, root_record))
    return app, spec, runtime, meter


class _VJPMeter:
    """Narrow meter adapter for the shared seven-call capture contract."""
    def __init__(self, budget, runtime_guard, synchronize):
        self.budget, self.runtime_guard, self.synchronize = budget, runtime_guard, synchronize
        self.entered = 0
        self.completed = 0
        self.sync_points = 0

    def synchronize_at(self, point):
        self.budget.check(point + ":pre")
        self.runtime_guard.check_unit(point + ":pre_runtime")
        self.synchronize()
        self.sync_points += 1
        self.budget.check(point + ":post")
        self.runtime_guard.check_unit(point + ":post_runtime")

    def unit(self, name, fn, /, *args, **kwargs):
        require(name == "vjp" and self.entered < 7, "real_vjp_call_matrix_extra")
        self.budget.check("vjp:pre")
        self.runtime_guard.check_unit("vjp:pre_runtime")
        self.entered += 1
        value = fn(*args, **kwargs)
        self.synchronize()
        self.budget.check("vjp:post")
        self.runtime_guard.check_unit("vjp:post_runtime")
        self.completed += 1
        return value

    def verify(self):
        require(self.entered == self.completed == 7 and self.sync_points == 7,
                "real_vjp_call_matrix_incomplete")
        return {"vjp_entered": self.entered, "vjp_completed": self.completed,
                "vjp_pre_sync_points": self.sync_points}


def run_numeric_kernel(trainer, batch, loss, *, permit: RealWorkerPermit, artifact_dir: Path,
                       expected_roster: list, trainer_receipt: ReadonlyTrainerReceipt,
                       budget, runtime_guard, setup_report: dict,
                       synchronize=None, failure_context: dict | None = None,
                       launch_artifact: dict | None = None,
                       inputs_artifact: dict | None = None) -> dict:
    """Run the one predict/loss/seven-VJP core after caller's source/setup gates."""
    require(isinstance(permit, RealWorkerPermit) and permit._token is _PERMIT_TOKEN
            and Path(artifact_dir) == permit.root / "artifacts", "real_worker_permit_required")
    require(trainer.opt is None and trainer.model.training and trainer.conditioner.training,
            "real_trainer_not_readonly_training_mode")
    require(isinstance(trainer_receipt, ReadonlyTrainerReceipt)
            and trainer_receipt._token is _TRAINER_TOKEN
            and trainer_receipt.trainer_id == id(trainer)
            and trainer_receipt.model_id == id(trainer.model)
            and trainer_receipt.conditioner_id == id(trainer.conditioner),
            "real_trainer_receipt_invalid")
    require(batch["x0"].device.type == "cuda" and batch["x0"].dtype is torch.float32,
            "real_batch_device_dtype")
    evidence.validate_roster(expected_roster)
    sync = synchronize or (lambda: torch.cuda.synchronize(batch["x0"].device))
    store = _GuardedArtifactStore(artifact_dir, budget=budget, runtime_guard=runtime_guard)
    calls = {key: 0 for key in ("trainer_predict", "conditioner_forward", "backbone_forward",
                                "loss_assembly", "loss_decode", "fk_call")}
    handles = []
    update_calls = {name: 0 for name in ("accumulate_micro", "clip_gradients", "step_optimizer")}
    original_update_methods = {}
    for name in update_calls:
        original = getattr(trainer, name)
        original_update_methods[name] = original

        def reject_update(*_args, _name=name, **_kwargs):
            update_calls[_name] += 1
            raise RealWorkerError("parameter_update_forbidden:" + _name)

        setattr(trainer, name, reject_update)
    original_conditioner = trainer.conditioner.forward
    original_decode = loss.decode_clip
    fk = loss.fk
    original_fk = fk.site_world_positions

    def enter(key):
        budget.check(key + ":pre")
        calls[key] += 1
        require(calls[key] == 1 if key in ("trainer_predict", "conditioner_forward", "backbone_forward", "loss_assembly")
                else calls[key] <= 2, "real_numeric_call_extra:" + key)

    def condition_only(batch_like):
        return original_conditioner(conditioner_only_input(batch_like))

    trainer.conditioner.forward = condition_only
    def pre(key):
        def hook(_module, _inputs):
            runtime_guard.check_unit(key + ":pre_runtime")
            enter(key)
        return hook
    def post(key):
        def hook(_module, _inputs, output):
            sync()
            budget.check(key + ":post")
            runtime_guard.check_unit(key + ":post_runtime")
            return output
        return hook
    handles.append(trainer.conditioner.register_forward_pre_hook(pre("conditioner_forward")))
    handles.append(trainer.conditioner.register_forward_hook(post("conditioner_forward")))
    handles.append(trainer.model.register_forward_pre_hook(pre("backbone_forward")))
    handles.append(trainer.model.register_forward_hook(post("backbone_forward")))

    def counted_decode(*args, **kwargs):
        runtime_guard.check_unit("loss_decode:pre_runtime")
        enter("loss_decode")
        value = original_decode(*args, **kwargs)
        sync()
        budget.check("loss_decode:post")
        runtime_guard.check_unit("loss_decode:post_runtime")
        return value
    loss.decode_clip = counted_decode
    def counted_fk(*args, **kwargs):
        runtime_guard.check_unit("fk_call:pre_runtime")
        enter("fk_call")
        value = original_fk(*args, **kwargs)
        sync()
        budget.check("fk_call:post")
        runtime_guard.check_unit("fk_call:post_runtime")
        return value
    fk.site_world_positions = counted_fk

    try:
        store.write_json("launch.json", launch_artifact or {
            "schema": SCHEMA + "/launch_artifact", "launch_id": permit.launch_id,
            "authorization_sha256": permit.authorization_sha256,
            "admission_metadata_sha256": permit.admission_metadata_sha256,
        })
        store.write_json("inputs.json", inputs_artifact or {"schema": SCHEMA + "/inputs_artifact"})
        runtime_guard.check_unit("shared_gradient_before_predict")
        before_state = shared.parameter_state_digest(trainer)
        before_rng = shared._rng_snapshot()
        enter("trainer_predict")
        _, prediction = trainer.predict(batch)
        sync()
        budget.check("trainer_predict:post")
        runtime_guard.check_unit("trainer_predict:post_runtime")
        require(shared.parameter_state_digest(trainer) == before_state, "real_params_changed_in_predict")
        require(shared._rng_equal(before_rng, shared._rng_snapshot()), "real_rng_changed_in_predict")
        observer = {}
        enter("loss_assembly")
        total, _ = loss(prediction, batch, observer)
        sync()
        budget.check("loss_assembly:post")
        runtime_guard.check_unit("loss_assembly:post_runtime")
        require(observer.get("total") is total and observer.get("contact_branch_executed") is False,
                "real_loss_observer_or_contact_branch")
        batch_loss, per_window, values = shared._loss_items(observer, protocol.N)
        arrays = {key: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
                  for key, value in values.items() if key != "scalars"}
        replay = shared.independent_loss_scalars(arrays)
        production_scalars = values["scalars"]
        for got, want in zip(production_scalars["per_window"], replay["per_window"], strict=True):
            shared.require(abs(got - want) <= shared.LOSS_TOL * max(abs(got), abs(want)),
                           "real_loss_cpu_replay")
        vjp_meter = _VJPMeter(budget, runtime_guard, sync)
        vjp = shared.capture_vjps(trainer, batch, observer, batch_loss, per_window,
                                  store=store, synchronize=sync, unit_meter=vjp_meter)
        vjp_counts = vjp_meter.verify()
        expected_parameters = [{**entry, "numel": int(np.prod(entry["shape"], dtype=np.int64))}
                               for group in expected_roster for entry in group]
        actual_parameters = [{"name": name, "shape": list(parameter.shape),
                              "dtype": str(parameter.dtype), "numel": parameter.numel()}
                             for name, parameter in shared._named_parameters(trainer)]
        require(actual_parameters == expected_parameters, "real_parameter_roster_mismatch")
        require(shared.parameter_state_digest(trainer) == before_state, "real_params_changed_during_kernel")
        require(shared._rng_equal(before_rng, shared._rng_snapshot()), "real_rng_changed_during_kernel")
        require(calls == {"trainer_predict": 1, "conditioner_forward": 1, "backbone_forward": 1,
                          "loss_assembly": 1, "loss_decode": 2, "fk_call": 2},
                "real_numeric_call_matrix_mismatch")
        require(update_calls == {"accumulate_micro": 0, "clip_gradients": 0, "step_optimizer": 0},
                "real_parameter_update_path_entered")
        analysis = shared.analyze_gradients(vjp)
        shared.require(analysis["parameter_gate_passed"], "real_parameter_additivity_gate")
        shared.require(all(parameter.grad is None for parameter in
                           (*trainer.model.parameters(), *trainer.conditioner.parameters())),
                       "real_parameter_grad_field_written")
        runtime_guard.finish(verifier=verify_step0_runtime_report)
        runtime_sync_counts = runtime_guard.guards.entered.get("runtime_syncs", 0)
        loss_receipts = shared._write_losses(store, values)
        counts_artifact = {"schema": SCHEMA + "/counts_artifact", "counts": calls,
                           "vjp": vjp_counts, "runtime_sync_count": runtime_sync_counts,
                           "parameter_updates": 0, "contact_branch": 0}
        store.write_json("counts.json", counts_artifact)
        outcome_files = {name: record for name, record in store.records.items()
                         if name.startswith("h_") or name.startswith("g_")
                         or name.startswith("loss_")}
        outcome = {"schema": SCHEMA, "status": "candidate_pending_external_return",
                   "analysis": analysis, "loss_scalars": production_scalars,
                   "independent_loss_scalars": replay, "loss_arrays": loss_receipts,
                   "vjp_roster_sha256": vjp["roster_sha256"],
                    "vjp_pairs": vjp_pair_receipts(vjp["requests"], vjp["responses"]),
                   "vjp_counts": vjp_counts, "parameter_state_sha256": before_state,
                   "parameter_state_unchanged": True, "rng_unchanged": vjp["rng_unchanged"],
                    "parameter_updates": 0, "optimizer_construction": 0,
                    "contact_branch": 0, "forbidden_update_calls": update_calls,
                    "trainer_construction": trainer_receipt.construction,
                    "setup_report": setup_report, "runtime_report": runtime_guard.facts(),
                    "runtime_sync_count": runtime_sync_counts,
                     "counts": calls, "files": outcome_files}
        store.write_json("vjp_records.json", {"requests": vjp["requests"], "responses": vjp["responses"],
                                                "roster": vjp["roster"], "roster_sha256": vjp["roster_sha256"],
                                                "file_records": vjp["file_records"]})
        store.write_json("loss_scalars.json", {"production": production_scalars, "independent": replay,
                                                "loss_arrays": loss_receipts,
                                                "array_records": {k: shared.array_record(v) for k, v in arrays.items()}})
        store.write_json("call_matrix.json", {"counts": calls, "vjp": vjp_counts,
                                               "runtime_sync_count": runtime_sync_counts,
                                               "parameter_updates": 0, "contact_branch": 0})
        store.write_json("outcome.json", outcome)
        review = review_real_artifacts(store.root, analysis, budget=budget, runtime_guard=runtime_guard)
        store.write_json("independent_review.json", review)
        return {"outcome": outcome, "independent_review": review,
                "counts": calls, "vjp": vjp_counts, "artifact_peak_bytes": store.peak_bytes,
                "artifact_io": dict(store.io_counts), "artifact_readback_io": review["io"]}
    except BaseException:
        if failure_context is not None:
            failure_context["numeric"] = {
                "counts": dict(calls),
                "forbidden_update_calls": dict(update_calls),
                "artifact_io": dict(store.io_counts),
                "artifact_files": dict(store.records),
                "budget": budget.facts(),
            }
            runtime_guard = locals().get("runtime_guard")
            if runtime_guard is not None:
                try:
                    failure_context["numeric"]["runtime_report"] = runtime_guard.facts()
                except Exception as report_exc:
                    failure_context["numeric"]["runtime_report_error"] = type(report_exc).__name__
            meter = locals().get("vjp_meter")
            if meter is not None:
                failure_context["numeric"]["vjp"] = {
                    "entered": meter.entered,
                    "completed": meter.completed,
                    "sync_points": meter.sync_points,
                }
        raise
    finally:
        for handle in handles:
            handle.remove()
        trainer.conditioner.forward = original_conditioner
        loss.decode_clip = original_decode
        fk.site_world_positions = original_fk
        for name, original in original_update_methods.items():
            setattr(trainer, name, original)


def _verify_runtime_sync_reconciliation(setup_calls: dict, outcome_count: int,
                                       review_io: dict) -> None:
    entered = setup_calls.get("entered") if type(setup_calls) is dict else None
    reads = review_io.get("artifact_reads") if type(review_io) is dict else None
    require(type(entered) is dict and type(entered.get("runtime_syncs")) is int
            and type(outcome_count) is int and type(reads) is int and reads >= 0
            and entered["runtime_syncs"] == outcome_count + 2 * reads,
            "runtime_guard_sync_count_disagreement")


def run_authorized_worker(*, authorization_raw: bytes, authorization_sha256: str,
                          package: dict, project_root: Path, work_budget,
                          runtime_construction_entry, runtime_construction_check,
                          failure_context: dict | None = None) -> dict:
    """In-process real worker. Only a supervisor-issued external permit may enter."""
    permit = authorize_worker(package=package, auth=_json_object(authorization_raw, "authorization_invalid"),
                              authorization_raw=authorization_raw,
                              authorization_sha256=authorization_sha256,
                              project_root=project_root, budget=work_budget)
    require(permit._token is _PERMIT_TOKEN and permit.root == Path(project_root).resolve() /
            admission.PROPOSED_REAL_ROOT, "worker_permit_path_binding")
    work_budget.check("worker_source_snapshot_entry")
    source_io = {"d6_file_reads": 0, "root_stats_file_reads": 0,
                 "asset_file_reads": 0, "source_file_reads": 0,
                 "source_file_hashes": 0, "d6_npz_member_reads": 0,
                 "root_stats_npz_member_reads": 0, "npz_archive_opens": 0,
                 "checkpoint_cpu_loads": 0,
                 }
    if failure_context is not None:
        failure_context["source_io"] = source_io
    snapshots, records = read_authorized_source_bundle(package, project_root, work_budget,
                                                       io_meter=source_io)
    prepared = prepare_snapshot_inputs(package, snapshots, records, io_meter=source_io)
    work_budget.check("worker_production_setup_entry")
    matrix = package.get("call_matrix")
    require(type(matrix) is dict and matrix.get("schema") == admission.SCHEMA + "/real_call_matrix_v1",
            "real_call_matrix_missing")
    setup_meter = _WorkerMeter(work_budget, matrix.get("setup_units", {}))
    if failure_context is not None:
        failure_context["setup_calls"] = setup_meter.entered
    components = setup_production_worker(package, snapshots, records, prepared=prepared,
                                         permit=permit, run_dir=permit.root / "d6",
                                         budget=work_budget,
                                         construction_entry=runtime_construction_entry,
                                         construction_check=runtime_construction_check,
                                         setup_meter=setup_meter)
    work_budget.check("worker_numerics_entry")
    result = run_numeric_kernel(components.trainer, components.batch, components.loss,
                                permit=permit, artifact_dir=permit.root / "artifacts",
                                expected_roster=prepared.expected_roster,
                                trainer_receipt=components.trainer_receipt,
                                budget=work_budget, runtime_guard=components.runtime,
                                setup_report=components.setup_report,
                                failure_context=failure_context,
                                launch_artifact={
                                    "schema": SCHEMA + "/launch_artifact",
                                    "launch_id": permit.launch_id,
                                    "authorization_sha256": permit.authorization_sha256,
                                    "admission_metadata_sha256": permit.admission_metadata_sha256,
                                },
                                inputs_artifact={
                                    "schema": SCHEMA + "/inputs_artifact",
                                    "identity": prepared.expected_identity,
                                    "windows": prepared.payload_manifest.get("windows"),
                                    "source_records": records,
                                })
    result["source_io"] = source_io
    require(result["outcome"]["status"] == "candidate_pending_external_return",
            "worker_candidate_state")
    construction = components.setup_report["trainer_construction"]
    setup_calls = components.setup_meter.verify()
    _verify_runtime_sync_reconciliation(setup_calls, result["outcome"].get("runtime_sync_count"),
                                        result["artifact_readback_io"])
    construction_entered = construction["construction_entered"]
    construction_returned = construction["construction_returned"]
    trainer_calls = {
        "dataset_constructions": construction["dataset_constructions"],
        "trainer_constructions": construction["trainer_constructions"],
        "model_constructions": construction_entered.count("model"),
        "model_transfers": construction_entered.count("model_transfer"),
        "conditioner_constructions": construction_entered.count("conditioner"),
        "conditioner_transfers": construction_entered.count("conditioner_transfer"),
        "loader_constructions": construction_entered.count("train_loader") + construction_entered.count("val_loader"),
        "loader_iterators": construction["loader_iterator_creations"],
        "dataset_reads": construction["dataset_iterations"],
        "optimizer_constructions": construction["optimizer_construction"],
    }
    source_counts = {"d6_file_reads": source_io["d6_file_reads"],
                     "root_stats_file_reads": source_io["root_stats_file_reads"],
                     "asset_file_reads": source_io["asset_file_reads"],
                     "source_file_hashes": source_io["source_file_hashes"],
                     "d6_npz_member_reads": source_io["d6_npz_member_reads"],
                     "root_stats_npz_member_reads": source_io["root_stats_npz_member_reads"],
                     "npz_archive_opens": source_io["npz_archive_opens"],
                     "initial_checkpoint_cpu_loads": source_io["checkpoint_cpu_loads"]}
    actual_calls = {
        "schema": matrix["schema"],
        "sources": source_counts,
        "setup_units": setup_calls["entered"],
        "trainer_construction": trainer_calls,
        "numeric": {"input_tensor_transfers": components.setup_report["input_tensor_transfers"],
                    "trainer_predict": result["counts"]["trainer_predict"],
                    "conditioner_forward": result["counts"]["conditioner_forward"],
                    "backbone_forward": result["counts"]["backbone_forward"],
                    "loss_assembly": result["counts"]["loss_assembly"],
                    "loss_decode": result["counts"]["loss_decode"],
                    "fk_call": result["counts"]["fk_call"],
                    "paired_vjp": result["vjp"]["vjp_entered"],
                    "vjp_pre_sync_points": result["vjp"]["vjp_pre_sync_points"]},
        "forbidden": {"contact_branch": result["outcome"]["contact_branch"],
                      "optimizer_steps": sum(result["outcome"]["forbidden_update_calls"].values()),
                      "accumulate_micro": result["outcome"]["forbidden_update_calls"]["accumulate_micro"],
                      "clip_gradients": result["outcome"]["forbidden_update_calls"]["clip_gradients"],
                      "step_optimizer": result["outcome"]["forbidden_update_calls"]["step_optimizer"],
                      "parameter_updates": result["outcome"]["parameter_updates"],
                      "checkpoint_writes": 0,
                      "dataset_iterations": construction["dataset_iterations"]},
    }
    require(actual_calls == {key: matrix[key] for key in actual_calls},
            "real_worker_call_matrix_mismatch")
    io_matrix = real_worker_io_matrix(package, records, result)
    return {"schema": SCHEMA + "/worker_result", "status": "candidate_pending_external_return",
            "launch_id": permit.launch_id, "authorization_sha256": permit.authorization_sha256,
            "admission_metadata_sha256": permit.admission_metadata_sha256,
            "outcome": result["outcome"], "independent_review": result["independent_review"],
            "counts": result["counts"], "vjp": result["vjp"],
             "artifact_peak_bytes": result["artifact_peak_bytes"],
             "artifact_io": result["artifact_io"],
             "source_records": worker_result_source_records(io_matrix),
             "call_matrix": actual_calls,
             "io_matrix": io_matrix,
             "budget": work_budget.facts(), "parameter_updates": 0,
             "worker_pid": os.getpid()}


def real_worker_io_matrix(package: dict, source_records: dict, result: dict) -> dict:
    """Summarize only worker-owned reads/writes for supervisor equality checks."""
    io = package.get("io_matrix")
    require(type(io) is dict, "real_io_matrix_missing")
    expected_paths = {row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                      for row in io["authorized_future_reads"]["d6_files"]}
    root = io["authorized_future_reads"]["root_stats"]
    expected_paths[root["path"]] = {"bytes": root["bytes"], "sha256": root["sha256"]}
    for row in io["authorized_future_reads"]["production_assets"]:
        expected_paths[row["path"]] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    require(set(source_records) == set(expected_paths), "real_worker_source_record_roster")
    normalized = {path: {"bytes": rec["bytes"], "sha256": rec["sha256"]}
                  for path, rec in source_records.items()}
    require(normalized == expected_paths, "real_worker_source_records_drift")
    artifact_io = result["artifact_io"]
    require(artifact_io == {"npy_writes": 20, "json_writes": 8}, "real_worker_artifact_io_matrix")
    files = result["outcome"].get("files")
    require(type(files) is dict and len(files) == 20
            and all(name.endswith(".npy") for name in files),
            "real_worker_artifact_file_matrix")
    readback = result["artifact_readback_io"]
    require(readback == {"artifact_reads": 27, "file_sha256_verified": 20},
            "real_worker_artifact_read_matrix")
    source_io = result["source_io"]
    require(source_io == {"d6_file_reads": 6, "root_stats_file_reads": 1,
                          "asset_file_reads": 41, "source_file_reads": 48,
                          "source_file_hashes": 48, "d6_npz_member_reads": 60,
                          "root_stats_npz_member_reads": 2, "npz_archive_opens": 4,
                          "checkpoint_cpu_loads": 1,
                          },
            "real_worker_source_member_matrix")
    return {"source_records": normalized, **source_io,
            "independent_artifact_reads": readback["artifact_reads"],
            "independent_artifact_hashes": readback["file_sha256_verified"],
            "artifact_files": files, "artifact_io": artifact_io,
            "strict_model_loads": result["outcome"]["setup_report"]["strict_load"]["strict_model_loads"]}


def worker_result_source_records(io_matrix: dict) -> dict:
    require(type(io_matrix) is dict and type(io_matrix.get("source_records")) is dict,
            "real_worker_source_projection_invalid")
    return io_matrix["source_records"]


def review_real_artifacts(root: Path, expected_analysis: dict | None = None, *,
                          budget=None, runtime_guard=None) -> dict:
    """Independent CPU replay of the real-kernel candidate's saved arrays."""
    root = Path(root)
    budget = budget or NullBudget()
    runtime_guard = runtime_guard or _NullRuntime()

    io_counts = {"artifact_reads": 0, "file_sha256_verified": 0}

    def read(rel, cap):
        budget.check("independent_read:" + rel + ":pre")
        runtime_guard.check_unit("independent_read:" + rel + ":pre")
        raw = open_regular_source(root, Path(rel), cap)
        io_counts["artifact_reads"] += 1
        budget.check("independent_read:" + rel + ":post")
        runtime_guard.check_unit("independent_read:" + rel + ":post")
        return raw

    outcome_raw = read("outcome.json", 8 << 20)
    outcome = _json_object(outcome_raw, "real_outcome_invalid")
    require(outcome.get("schema") == SCHEMA and outcome.get("status") == "candidate_pending_external_return",
            "real_outcome_not_candidate")
    files = outcome.get("files")
    expected_names = {f"{role}_{i}.npy" for i in range(7) for role in ("h", "g")}
    expected_names.update(f"loss_{key}.npy" for key in shared.LOSS_NAMES)
    require(type(files) is dict and set(files) == expected_names, "real_outcome_array_roster")
    arrays = {}
    for name, rec in files.items():
        require(Path(name).name == name and type(rec) is dict, "real_array_record_invalid")
        size_cap = shared.PARAMETER_COUNT * 4 + (16 << 10) if name.startswith("g_") else 2 << 20
        raw = read(name, size_cap)
        require(len(raw) == rec.get("bytes") and _sha(raw) == rec.get("sha256"), "real_array_file_binding")
        io_counts["file_sha256_verified"] += 1
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        require(type(rec.get("shape")) is list and list(arr.shape) == rec["shape"]
                and rec.get("dtype") == "<f4" and arr.dtype == np.dtype("<f4")
                and np.isfinite(arr).all(), "real_array_dtype_shape_or_domain")
        arrays[name] = arr
    vjp = _json_object(read("vjp_records.json", 8 << 20), "real_vjp_records_invalid")
    loss = _json_object(read("loss_scalars.json", 2 << 20), "real_loss_records_invalid")
    roster = vjp.get("roster")
    require(type(roster) is list and len(roster) == shared.PARAMETER_TENSORS
            and sum(r.get("numel", 0) for r in roster) == shared.PARAMETER_COUNT,
            "real_parameter_roster")
    require(vjp.get("roster_sha256") == shared.sha256(shared.canonical_json(roster)),
            "real_parameter_roster_sha")
    pairs = shared.verify_vjp_records(vjp.get("requests"), vjp.get("responses"), roster)
    require(outcome.get("vjp_pairs") == [
        {"call_id": request["call_id"], "request_sha256": request["request_sha256"],
         "response_sha256": response["response_sha256"]}
        for request, response in zip(vjp["requests"], vjp["responses"], strict=True)],
        "real_vjp_pair_binding")
    outputs = {}
    for idx, response in enumerate(vjp["responses"]):
        role = "batch" if idx == 0 else f"window_{idx - 1}"
        h = arrays[f"h_{idx}.npy"]
        g = arrays[f"g_{idx}.npy"]
        require(shared.array_record(h) == response["h"] and shared.array_record(g) == response["g_flat"],
                "real_vjp_array_record")
        offset = 0
        for tensor, param in zip(response["g"], roster, strict=True):
            size = param["numel"]
            part = g[offset:offset + size].reshape(param["shape"])
            require(shared.array_record(part) == {k: tensor[k] for k in ("shape", "dtype", "bytes", "sha256")},
                    "real_named_gradient_binding")
            offset += size
        require(offset == g.size, "real_flat_gradient_length")
        outputs[role] = {"h": h, "g": g}
    analysis = shared.analyze_gradients({"outputs": outputs, "roster": roster})
    require(analysis == outcome.get("analysis"), "real_gradient_analysis_drift")
    if expected_analysis is not None:
        require(analysis == expected_analysis, "real_gradient_analysis_not_live_result")
    loss_arrays = {}
    for key in shared.LOSS_NAMES:
        name = f"loss_{key}.npy"
        arr = arrays[name]
        expected_shape = {"pos": (6, 60, 3), "rot6d": (6, 60, 6), "joint": (6, 60, 29),
                          "geodesic_rad2": (6, 60), "vel": (6, 59, 38), "fk_m2": (6, 60, 15)}[key]
        require(arr.shape == expected_shape and arr.dtype == np.dtype("<f4") and (arr >= 0).all(),
                "real_loss_array_contract:" + key)
        loss_arrays[key] = arr
    scalars = shared.independent_loss_scalars(loss_arrays)
    require(scalars == loss.get("independent") == outcome.get("independent_loss_scalars"),
            "real_loss_scalar_readback")
    require(loss.get("loss_arrays") == outcome.get("loss_arrays"), "real_loss_receipt_binding")
    array_records = loss.get("array_records")
    require(type(array_records) is dict and set(array_records) == set(shared.LOSS_NAMES),
            "real_loss_array_record_roster")
    for key, arr in loss_arrays.items():
        require(shared.array_record(arr) == array_records.get(key), "real_loss_array_record:" + key)
    production = loss.get("production")
    require(production == outcome.get("loss_scalars") and type(production) is dict
            and len(production.get("per_window", [])) == 6
            and all(abs(got - want) <= shared.LOSS_TOL * max(abs(got), abs(want))
                    for got, want in zip(production["per_window"], scalars["per_window"], strict=True)),
            "real_production_loss_scalar_binding")
    terms = production.get("terms")
    require(type(terms) is dict and set(terms) == {*shared.LOSS_NAMES, "total"}
            and production.get("weights") == shared.WEIGHTS,
            "real_production_loss_term_roster")
    for key in shared.LOSS_NAMES:
        got, want = terms[key], scalars["batch_terms"][key]
        require(type(got) in (int, float) and np.isfinite(got)
                and abs(got - want) <= shared.LOSS_TOL * max(abs(got), abs(want)),
                "real_production_loss_term_mismatch:" + key)
    got_total = production.get("batch")
    want_total = scalars["batch_terms"]["total"]
    require(type(got_total) in (int, float) and np.isfinite(got_total)
            and abs(got_total - want_total) <= shared.LOSS_TOL * max(abs(got_total), abs(want_total)),
            "real_production_batch_loss_mismatch")
    launch = _json_object(read("launch.json", 256 << 10), "real_launch_artifact_invalid")
    inputs = _json_object(read("inputs.json", 8 << 20), "real_inputs_artifact_invalid")
    counts = _json_object(read("counts.json", 256 << 10), "real_counts_artifact_invalid")
    require(launch.get("schema") == SCHEMA + "/launch_artifact"
            and type(launch.get("launch_id")) is str
            and type(launch.get("authorization_sha256")) is str
            and type(launch.get("admission_metadata_sha256")) is str,
            "real_launch_artifact_contract")
    require(inputs.get("schema") == SCHEMA + "/inputs_artifact"
            and type(inputs.get("windows")) is list and len(inputs["windows"]) == protocol.N
            and type(inputs.get("source_records")) is dict,
            "real_inputs_artifact_contract")
    call_matrix = _json_object(read("call_matrix.json", 256 << 10), "real_call_matrix_invalid")
    forbidden_updates = {name: 0 for name in ("accumulate_micro", "clip_gradients", "step_optimizer")}
    require(outcome.get("vjp_counts") == {"vjp_entered": 7, "vjp_completed": 7, "vjp_pre_sync_points": 7}
            and outcome.get("parameter_updates") == outcome.get("optimizer_construction") == 0
            and outcome.get("forbidden_update_calls") == forbidden_updates
            and outcome.get("trainer_construction", {}).get("optimizer_construction") == 0
            and outcome.get("trainer_construction", {}).get("loader_iterator_creations") == 0
            and outcome.get("trainer_construction", {}).get("dataset_iterations") == 0
            and outcome.get("runtime_report", {}).get("stage") == "D6"
            and outcome.get("runtime_report", {}).get("kind") == "production"
            and type(outcome.get("runtime_sync_count")) is int
            and outcome["runtime_sync_count"] >= 1
            and outcome.get("contact_branch") == 0 and outcome.get("rng_unchanged") is True
            and outcome.get("parameter_state_unchanged") is True
            and call_matrix.get("counts") == outcome.get("counts")
            and call_matrix.get("vjp") == outcome.get("vjp_counts")
            and call_matrix.get("runtime_sync_count") == outcome.get("runtime_sync_count")
            and counts.get("counts") == outcome.get("counts")
            and counts.get("vjp") == outcome.get("vjp_counts")
            and counts.get("runtime_sync_count") == outcome.get("runtime_sync_count")
            and call_matrix.get("parameter_updates") == 0 and call_matrix.get("contact_branch") == 0,
            "real_candidate_forbidden_or_call_counts")
    return {"schema": SCHEMA + "/independent_review", "passed": True,
            "decision": analysis["decision"], "groups": analysis["groups"],
            "h_checks": analysis["h_checks"], "vjp_pair_count": len(pairs),
            "array_files": len(arrays), "parameter_updates": 0, "io": io_counts}
