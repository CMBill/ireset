"""Synthetic I/C primitives for the fixed-six-window shared-gradient draft.

The real archive/checkpoint path is intentionally absent.  This module can
exercise the production deterministic network, conditioner, Trainer.predict
and LossAssembler on synthetic inputs; it cannot read a real source or run a
real diagnostic.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import default_collate

from ireset.model.conditioner import ConditionEncoder
from ireset.model.deterministic import DeterministicG1
from ireset.model.tokens import RootStats, StateTokenizer
from ireset.eval.w1_smallset_fit_r_admission import RegisteredWindowDataset
from ireset.train.losses import LossAssembler, LossWeights
from ireset.train.trainer import TrainConfig, Trainer

SCHEMA = "w1_d6_shared_gradient_synthetic_v1"
WINDOWS = (
    "sub10_largebox_097:0:60",
    "sub12_largebox_082:0:60",
    "sub16_largebox_000:0:60",
    "sub3_largebox_027:0:60",
    "sub7_largebox_022:0:60",
    "sub8_largebox_035:0:60",
)
WINDOW = 60
PARAMETER_COUNT = 3_739_210
PARAMETER_TENSORS = 130
COND_KEYS = ("kps_obj", "kps_clip", "box_t_clip", "box_quat", "contact")
INPUT_KEYS = ("x0", *COND_KEYS, "wfc", "valid", "clip_idx", "start")
MICRO_CANONICAL_SLOTS = (1, 0, 4, 3, 2, 5)  # batch position -> registered window slot
BATCH_SLOT_BY_CANONICAL = tuple(MICRO_CANONICAL_SLOTS.index(i) for i in range(6))
LOSS_NAMES = ("pos", "rot6d", "joint", "geodesic_rad2", "vel", "fk_m2")
WEIGHTS = {"pos": 1., "rot6d": 1., "joint": 1., "geodesic_rad2": .5, "vel": .5, "fk_m2": 2.}
LOSS_TOL = 2e-6
H_RTOL = 2e-4
PARAMETER_RHO_MAX = 1e-3
SYNTHETIC_ACCEPTANCE_RHO = 2.5e-4
MIN_COMMON_NORM = 1e-10
CORE_SYNC_POINTS = 32  # 25 metered-unit completions + 7 pre-VJP barriers
CORE_ARTIFACT_IO = {"npy_writes": 20, "json_writes": 5,
                    "artifact_reads": 24, "file_sha256_writes": 25,
                    "file_sha256_verified": 23}
IC_IMPLEMENTATION_FILES = (
    "src/ireset/eval/w1_d6_shared_gradient.py",
    "scripts/verify_w1_d6_shared_gradient.py",
    "tests/test_w1_d6_shared_gradient.py",
    "src/ireset/train/trainer.py",
    "src/ireset/train/losses.py",
    "src/ireset/model/conditioner.py",
    "src/ireset/model/deterministic.py",
    "src/ireset/model/backbone.py",
    "src/ireset/model/tokens.py",
    "src/ireset/geometry/distance_adapter.py",
    "src/ireset/kinematics/rotation_adapter.py",
    "src/ireset/eval/w1_smallset_fit_r_admission.py",
    "src/ireset/embodied/g1_spec.py",
    "src/ireset/diffusion/generator.py",
    "pyproject.toml",
    "uv.lock",
)
CORE_CALL_MATRIX = {
    "registered_dataset_construct": 1, "trainer_construct": 1,
    "model_construct": 1, "model_device_transfer": 1,
    "conditioner_construct": 1, "conditioner_device_transfer": 1,
    "default_collate": 1,
    "batch_device_transfer": 1, "train_dataloader_construct": 1,
    "val_dataloader_construct": 1, "train_iterator_create": 0,
    "val_iterator_create": 0,
    "trainer_predict": 1, "conditioner_forward": 1, "backbone_forward": 1,
    "loss_assembly": 1, "loss_decode": 2, "fk_call": 2, "vjp": 7,
    "optimizer_construction": 0, "parameter_updates": 0,
    "contact_branch": 0, "dataset_iterations": 0,
}


class SharedGradientError(RuntimeError):
    pass


def require(condition: bool, code: str):
    if not condition:
        raise SharedGradientError(code)


class UnitMeter:
    """Exact single-attempt call caps and pre/post monotonic deadline gates."""
    def __init__(self, expected: dict[str, int], *, deadline_s: float = 1800.,
                 clock: Callable[[], float] = time.perf_counter,
                 synchronize: Callable[[], None] | None = None,
                 expected_sync_points: int | None = None):
        self.expected = dict(expected)
        self.entered = {name: 0 for name in expected}
        self.completed = {name: 0 for name in expected}
        self.events = []
        self.clock, self.start, self.deadline_s = clock, clock(), deadline_s
        self.synchronize = synchronize or (lambda: None)
        self.expected_sync_points = expected_sync_points
        self.sync_points = 0
        self.sync_events = []

    def check(self, point: str):
        elapsed = self.clock() - self.start
        require(elapsed < self.deadline_s, "deadline:" + point)

    def enter(self, name: str):
        self.check(name + ":pre")
        require(name in self.expected and self.entered[name] < self.expected[name], "call_matrix_extra:" + name)
        self.entered[name] += 1
        self.events.append({"event": "enter", "unit": name, "ordinal": self.entered[name]})

    def finish(self, name: str):
        self.synchronize_at(name + ":post")
        self.check(name + ":post")
        require(self.completed[name] < self.entered[name], "call_matrix_unentered_completion:" + name)
        self.completed[name] += 1
        self.events.append({"event": "return", "unit": name, "ordinal": self.completed[name]})

    def synchronize_at(self, point: str):
        self.check(point + ":pre_sync")
        self.synchronize()
        self.sync_points += 1
        self.sync_events.append({"point": point, "ordinal": self.sync_points})
        self.check(point + ":post_sync")

    def unit(self, name: str, fn: Callable, /, *args, **kwargs):
        self.enter(name)
        result = fn(*args, **kwargs)
        self.finish(name)
        return result

    def verify_complete(self):
        require(self.entered == self.expected and self.completed == self.expected,
                "call_matrix_incomplete")
        if self.expected_sync_points is not None:
            require(self.sync_points == self.expected_sync_points, "cuda_sync_matrix_incomplete")
        return {"expected": self.expected, "entered": self.entered, "completed": self.completed,
                "events": self.events, "sync_points": self.sync_points,
                "sync_events": self.sync_events, "elapsed_s": self.clock() - self.start}


def canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _array_bytes(value: np.ndarray) -> bytes:
    arr = np.asarray(value)
    require(arr.dtype.kind == "f" and arr.dtype.itemsize == 4 and arr.dtype.byteorder in ("<", "="),
            "array_must_be_little_endian_float32")
    require(np.isfinite(arr).all(), "array_nonfinite")
    return np.ascontiguousarray(arr, dtype="<f4").tobytes(order="C")


def array_record(value: np.ndarray) -> dict:
    arr = np.asarray(value)
    raw = _array_bytes(arr)
    return {"shape": list(arr.shape), "dtype": "<f4", "bytes": len(raw), "sha256": sha256(raw)}


def _tensor_digest(tensor: torch.Tensor) -> tuple[dict, np.ndarray]:
    require(torch.is_tensor(tensor) and tensor.layout == torch.strided and tensor.dtype is torch.float32,
            "vjp_tensor_contract")
    require(bool(torch.isfinite(tensor).all()), "vjp_tensor_nonfinite")
    host = tensor.detach().to(device="cpu").contiguous().numpy().astype("<f4", copy=False)
    return array_record(host), host


def _named_parameters(trainer: Trainer):
    rows = [(f"model.{name}", value) for name, value in trainer.model.named_parameters()]
    rows.extend((f"conditioner.{name}", value) for name, value in trainer.conditioner.named_parameters())
    require(len(rows) == PARAMETER_TENSORS, "parameter_tensor_count")
    require(sum(value.numel() for _, value in rows) == PARAMETER_COUNT, "parameter_coordinate_count")
    require(len({name for name, _ in rows}) == len(rows), "parameter_name_duplicate")
    for name, value in rows:
        require(value.requires_grad and value.dtype is torch.float32, f"parameter_contract:{name}")
    return rows


def parameter_state_digest(trainer: Trainer) -> str:
    h = hashlib.sha256()
    for side, module in (("model", trainer.model), ("conditioner", trainer.conditioner)):
        for name, value in module.state_dict().items():
            host = value.detach().to("cpu").contiguous().numpy()
            h.update(canonical_json({"side": side, "name": name, "shape": list(host.shape),
                                     "dtype": host.dtype.str}))
            h.update(host.tobytes(order="C"))
    return h.hexdigest()


def _rng_snapshot() -> dict:
    return {"torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "numpy": np.random.get_state()}


def _rng_equal(left: dict, right: dict) -> bool:
    a, b = left["numpy"], right["numpy"]
    return (torch.equal(left["torch_cpu"], right["torch_cpu"])
            and len(left["torch_cuda"]) == len(right["torch_cuda"])
            and all(torch.equal(x, y) for x, y in zip(left["torch_cuda"], right["torch_cuda"], strict=True))
            and a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:])


def _loss_items(observer: dict, batch_size: int) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    elems = observer.get("term_elements")
    fk = observer.get("fk_elements")
    require(isinstance(elems, dict) and set(elems) == set(LOSS_NAMES[:-1]), "loss_element_roster")
    require(torch.is_tensor(fk) and tuple(fk.shape) == (batch_size, WINDOW, 15), "fk_element_shape")
    values = {**elems, "fk_m2": fk}
    expected_shapes = {"pos": (batch_size, WINDOW, 3), "rot6d": (batch_size, WINDOW, 6),
                       "joint": (batch_size, WINDOW, 29), "geodesic_rad2": (batch_size, WINDOW),
                       "vel": (batch_size, WINDOW - 1, 38), "fk_m2": (batch_size, WINDOW, 15)}
    for key, shape in expected_shapes.items():
        value = values[key]
        require(torch.is_tensor(value) and tuple(value.shape) == shape and value.dtype is torch.float32,
                f"loss_element_contract:{key}")
        require(bool(torch.isfinite(value).all()) and bool((value >= 0).all()), f"loss_element_domain:{key}")
    weights = WEIGHTS
    per_batch = []
    for slot in range(batch_size):
        part = weights["pos"] * values["pos"][slot].mean()
        part = part + weights["rot6d"] * values["rot6d"][slot].mean()
        part = part + weights["joint"] * values["joint"][slot].mean()
        part = part + weights["geodesic_rad2"] * values["geodesic_rad2"][slot].mean()
        part = part + weights["vel"] * values["vel"][slot].mean()
        part = part + weights["fk_m2"] * values["fk_m2"][slot].mean()
        per_batch.append(part)
    per_window = [per_batch[BATCH_SLOT_BY_CANONICAL[i]] for i in range(6)]
    mean = sum(per_window[1:], per_window[0]) / batch_size
    batch_loss = observer.get("total")
    require(torch.is_tensor(batch_loss) and batch_loss.ndim == 0 and batch_loss is observer.get("term_tensors", {}).get("total"),
            "batch_loss_identity")
    lhs, rhs = float(batch_loss.detach().cpu()), float(mean.detach().cpu())
    require(abs(lhs - rhs) <= LOSS_TOL * max(abs(lhs), abs(rhs)), "batch_loss_per_window_mean_mismatch")
    scalars = {"batch": lhs, "per_window": [float(x.detach().cpu()) for x in per_window],
               "weights": dict(weights), "terms": {k: float(v.detach().cpu()) for k, v in observer["term_tensors"].items()}}
    return batch_loss, per_window, {**values, "scalars": scalars}


def independent_loss_scalars(values: dict[str, np.ndarray], weights: dict = WEIGHTS) -> dict:
    """Float64 CPU replay of production element denominators and batch slots."""
    shapes = {"pos": (6, 60, 3), "rot6d": (6, 60, 6), "joint": (6, 60, 29),
              "geodesic_rad2": (6, 60), "vel": (6, 59, 38), "fk_m2": (6, 60, 15)}
    require(set(values) == set(shapes), "independent_loss_element_roster")
    for key, shape in shapes.items():
        arr = np.asarray(values[key])
        require(arr.shape == shape and arr.dtype == np.dtype("<f4") and np.isfinite(arr).all()
                and (arr >= 0).all(), "independent_loss_element_contract:" + key)
    denoms = {"pos": 180., "rot6d": 360., "joint": 1740., "geodesic_rad2": 60.,
              "vel": 2242., "fk_m2": 900.}
    per_batch, terms_batch = [], {key: [] for key in LOSS_NAMES}
    for slot in range(6):
        for key in LOSS_NAMES:
            terms_batch[key].append(float(np.sum(values[key][slot].astype(np.float64), dtype=np.float64)
                                          / denoms[key]))
        total = weights["pos"] * terms_batch["pos"][slot]
        total = total + weights["rot6d"] * terms_batch["rot6d"][slot]
        total = total + weights["joint"] * terms_batch["joint"][slot]
        total = total + weights["geodesic_rad2"] * terms_batch["geodesic_rad2"][slot]
        total = total + weights["vel"] * terms_batch["vel"][slot]
        total = total + weights["fk_m2"] * terms_batch["fk_m2"][slot]
        per_batch.append(total)
    per_canonical = [per_batch[BATCH_SLOT_BY_CANONICAL[i]] for i in range(6)]
    terms = {key: [rows[BATCH_SLOT_BY_CANONICAL[i]] for i in range(6)] for key, rows in terms_batch.items()}
    aggregate = {key: float(np.mean(np.asarray(rows, dtype=np.float64), dtype=np.float64))
                 for key, rows in terms.items()}
    aggregate["total"] = float(np.mean(np.asarray(per_canonical, dtype=np.float64), dtype=np.float64))
    return {"per_window": per_canonical, "batch_terms": aggregate}


class ArtifactStore:
    """Atomic one-array-at-a-time NPY writer with an explicit peak-byte cap."""

    def __init__(self, root: Path, *, cap_bytes: int = 128 * 1024**2,
                 reserve_bytes: int = 1024**2, header_cap: int = 4096):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.cap_bytes, self.reserve_bytes, self.header_cap = cap_bytes, reserve_bytes, header_cap
        self.records: dict[str, dict] = {}
        self.peak_bytes = 0
        self.write_count = 0
        self.file_sha256_count = 0

    def write_array(self, name: str, value: np.ndarray) -> dict:
        require(name.endswith(".npy") and Path(name).name == name and name not in self.records,
                "artifact_name_or_duplicate")
        arr = np.ascontiguousarray(value, dtype="<f4")
        raw = _npy_bytes(arr)
        require(len(raw) <= arr.nbytes + self.header_cap, "npy_header_cap")
        existing = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        projected = existing + 2 * len(raw) + self.reserve_bytes
        require(projected <= self.cap_bytes, "artifact_space_cap")
        tmp = self.root / (name + ".tmp")
        require(not tmp.exists(), "artifact_temp_exists")
        with tmp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.peak_bytes = max(self.peak_bytes, existing + len(raw) + tmp.stat().st_size + self.reserve_bytes)
        os.replace(tmp, self.root / name)
        _fsync_dir(self.root)
        rec = {"bytes": len(raw), "sha256": sha256(raw), "shape": list(arr.shape), "dtype": "<f4"}
        self.write_count += 1
        self.file_sha256_count += 1
        self.records[name] = rec
        return rec

    def write_json(self, name: str, value) -> dict:
        require(name.endswith(".json") and Path(name).name == name and name not in self.records,
                "json_artifact_name")
        raw = canonical_json(value)
        existing = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        require(existing + len(raw) + 8192 <= self.cap_bytes, "artifact_space_cap")
        tmp = self.root / (name + ".tmp")
        require(not tmp.exists(), "artifact_temp_exists")
        with tmp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.root / name)
        _fsync_dir(self.root)
        rec = {"bytes": len(raw), "sha256": sha256(raw)}
        self.write_count += 1
        self.file_sha256_count += 1
        self.records[name] = rec
        self.peak_bytes = max(self.peak_bytes, sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()))
        return rec


def _npy_bytes(array: np.ndarray) -> bytes:
    import io
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array, dtype="<f4"), allow_pickle=False)
    return stream.getvalue()


def _fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SyntheticDataset(torch.utils.data.Dataset):
    """The loader identity is constructed, but no dataset iteration is valid."""
    def __len__(self):
        return len(WINDOWS)

    def __getitem__(self, index):
        raise SharedGradientError("synthetic_dataset_must_not_be_iterated")


class NoReadRegisteredWindowDataset(RegisteredWindowDataset):
    """Registered-window API with a hard guard against loader consumption."""
    def __getitem__(self, index):
        raise SharedGradientError("registered_dataset_read_forbidden")


class SyntheticFK:
    """Differentiable fixture with production LossAssembler call semantics."""
    def __init__(self, device: str, seed: int):
        rng = np.random.default_rng(seed)
        self.offsets = torch.as_tensor(rng.normal(0., .08, (15, 3)), dtype=torch.float32, device=device)
        self.q_map = torch.as_tensor(rng.normal(0., .001, (29, 15, 3)), dtype=torch.float32, device=device)
        self.calls = 0

    def site_world_positions(self, pos, quat, q, sites):
        require(list(sites) == list(__import__("ireset.embodied.g1_spec", fromlist=["KPT_TO_SMPLH"]).KPT_TO_SMPLH),
                "synthetic_fk_site_roster")
        self.calls += 1
        rotation = _quat_to_matrix(quat)
        rotated = torch.einsum("bij,sj->bsi", rotation, self.offsets)
        articulated = torch.einsum("bj,jsd->bsd", q, self.q_map)
        return pos[:, None, :] + rotated + articulated


def _quat_to_matrix(quat):
    from ireset.model.tokens import quat_to_mat_torch
    return quat_to_mat_torch(quat)


def synthetic_items(seed: int, *, group: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    identity = np.array([1., 0., 0., 0.], dtype=np.float32)
    x0 = np.zeros((WINDOW, 38), dtype=np.float32)
    x0[:, 0:3] = rng.normal(0., .4, (WINDOW, 3)).astype(np.float32)
    x0[:, 3:9] = np.array([1., 0., 0., 0., 1., 0.], dtype=np.float32)
    x0[:, 9:38] = rng.normal(0., .4, (WINDOW, 29)).astype(np.float32)
    base = {
        "kps_obj": rng.normal(0., .08, (WINDOW, 52, 3)).astype(np.float32),
        "kps_clip": rng.normal(0., .09, (WINDOW, 52, 3)).astype(np.float32),
        "box_t_clip": rng.normal(0., .05, (WINDOW, 3)).astype(np.float32),
    }
    items = []
    for slot in range(6):
        x0_slot = x0.copy()
        x0_slot[:, 0:3] += np.asarray([slot * .025, -slot * .012, slot * .008], dtype=np.float32)
        x0_slot[:, 9:38] += np.float32((slot - 2.5) * .035)
        kps_obj = base["kps_obj"].copy()
        kps_obj += np.float32((slot - 2.5) * .015)
        kps_clip = base["kps_clip"].copy()
        kps_clip += np.float32((slot - 2.5) * .01)
        item = {
            "x0": torch.from_numpy(x0_slot),
            "kps_obj": torch.from_numpy(kps_obj),
            "kps_clip": torch.from_numpy(kps_clip),
            "box_t_clip": torch.from_numpy(base["box_t_clip"].copy()),
            "box_quat": torch.from_numpy(np.tile(identity, (WINDOW, 1))),
            "contact": torch.zeros((WINDOW, 4), dtype=torch.float32),
            "wfc": torch.eye(4, dtype=torch.float32),
            "valid": torch.ones(WINDOW, dtype=torch.float32),
            "clip_idx": slot,
            "start": 0,
        }
        # Keep the fixture key visible in the row identity without changing
        # the production batch fields.
        item["_fixture_group"] = group
        items.append(item)
    return items


def _batch_for_trainer(items: list[dict], device: str, meter: UnitMeter) -> dict:
    require(len(items) == 6 and all(set(row) == set(INPUT_KEYS) | {"_fixture_group"} for row in items),
            "synthetic_input_roster")
    clean = [{k: value for k, value in items[slot].items() if k in INPUT_KEYS}
             for slot in MICRO_CANONICAL_SLOTS]
    batch = meter.unit("default_collate", default_collate, clean)
    require(set(batch) == set(INPUT_KEYS), "collated_input_roster")
    require(device == "cpu", "synthetic_collate_must_start_on_cpu")
    require(tuple(batch["x0"].shape) == (6, WINDOW, 38), "collated_x0_shape")
    require(batch["x0"].dtype is torch.float32, "collated_x0_dtype")
    require(bool(torch.isfinite(batch["x0"]).all()), "collated_x0_finite")
    return batch


def _runtime_setup(device: str):
    require(device == "cuda:0" and torch.cuda.is_available(), "synthetic_gpu_required")
    require(torch.cuda.device_count() >= 1, "synthetic_cuda_device_missing")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    require(not torch.is_autocast_enabled("cuda") and not torch.is_autocast_enabled("cpu"), "synthetic_autocast_enabled")
    return {"device": "cuda:0", "dtype": "float32", "torch": torch.__version__,
             "numpy": np.__version__, "threads": torch.get_num_threads(),
             "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
             "tf32_cudnn": torch.backends.cudnn.allow_tf32,
             "autocast_cuda": torch.is_autocast_enabled("cuda"),
             "autocast_cpu": torch.is_autocast_enabled("cpu"),
             "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
             "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
             "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}


def _make_trainer(seed: int, device: str, fk: SyntheticFK, output: Path,
                  dataset: RegisteredWindowDataset, meter: UnitMeter):
    root_stats = RootStats(mean=np.array([.12, -.07, .48], dtype=np.float32),
                           std=np.array([.28, .31, .22], dtype=np.float32))
    joint_range = np.stack((np.full(29, -2.8), np.full(29, 2.8)), axis=1)
    tokenizer = StateTokenizer(joint_range, root_stats)
    loss = LossAssembler(tokenizer, LossWeights(w_pos=1., w_rot6d=1., w_joint=1., w_geodesic=.5,
                                               w_fk=2., w_contact=0., w_vel=.5), device,
                         use_fk=True, fk=fk, static_frame=(np.eye(3), np.zeros(3)))
    cfg = TrainConfig(model_type="deterministic", steps=1, micro_batch=6, grad_accum=1,
                      seed=seed, num_workers=0, device=device, d_model=240, n_layers=4,
                      n_heads=6, ffn_mult=4, time_window=31,
                      loss=LossWeights(w_pos=1., w_rot6d=1., w_joint=1., w_geodesic=.5,
                                       w_fk=2., w_contact=0., w_vel=.5))
    def enter(unit: str):
        names = {"model": "model_construct", "model_transfer": "model_device_transfer",
                 "conditioner": "conditioner_construct", "conditioner_transfer": "conditioner_device_transfer",
                 "train_loader": "train_dataloader_construct", "val_loader": "val_dataloader_construct"}
        meter.enter(names[unit])

    def checked(unit: str, _module):
        names = {"model": "model_construct", "model_transfer": "model_device_transfer",
                 "conditioner": "conditioner_construct", "conditioner_transfer": "conditioner_device_transfer"}
        meter.finish(names[unit])

    def constructed(unit: str):
        names = {"train_loader": "train_dataloader_construct", "val_loader": "val_dataloader_construct"}
        if unit in names:
            meter.finish(names[unit])

    trainer = meter.unit("trainer_construct", Trainer, cfg, tokenizer, dataset, dataset, output,
                         loss_fn=loss, build_optimizer=False, construction_entry=enter,
                         construction_check=checked, construction_observer=constructed)
    require(trainer.opt is None, "synthetic_optimizer_constructed")
    trainer.model.train()
    trainer.conditioner.train()
    return trainer, loss


def _install_runtime_counts(trainer: Trainer, meter: UnitMeter):
    handles = []
    original_forward = trainer.conditioner.forward

    def condition_only_forward(data):
        require(isinstance(data, dict) and set(COND_KEYS) <= set(data), "conditioner_input_missing_condition")
        condition = {key: data[key] for key in COND_KEYS}
        require(set(condition) == set(COND_KEYS), "conditioner_received_noncondition_input")
        return original_forward(condition)

    trainer.conditioner.forward = condition_only_forward

    def before(name):
        def hook(_module, _inputs):
            meter.enter(name)
        return hook

    def after(name):
        def hook(_module, _inputs, output):
            meter.finish(name)
            return output
        return hook

    handles.append(trainer.conditioner.register_forward_pre_hook(before("conditioner_forward")))
    handles.append(trainer.conditioner.register_forward_hook(after("conditioner_forward")))
    handles.append(trainer.model.register_forward_pre_hook(before("backbone_forward")))
    handles.append(trainer.model.register_forward_hook(after("backbone_forward")))
    return handles, original_forward


def _npy_write(store: ArtifactStore | None, name: str, array: np.ndarray):
    return None if store is None else store.write_array(name, array)


def capture_vjps(trainer: Trainer, batch: dict, observer: dict, batch_loss: torch.Tensor,
                 per_window: list[torch.Tensor], *, store: ArtifactStore | None = None,
                 synchronize: Callable[[], None] | None = None,
                 unit_meter: UnitMeter | None = None) -> dict:
    """Execute the seven paired autograd calls and bind positional returns."""
    params = _named_parameters(trainer)
    names = [name for name, _ in params]
    tensors = [value for _, value in params]
    roster = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()}
              for name, value in params]
    roster_sha = sha256(canonical_json(roster))
    frozen_ids = tuple(id(value) for value in tensors)
    frozen_versions = tuple(value._version for value in tensors)
    before_state = parameter_state_digest(trainer)
    before_rng = _rng_snapshot()
    before_grad = tuple(value.grad for value in tensors)
    sync = synchronize or (lambda: torch.cuda.synchronize("cuda:0"))
    outputs, requests, responses = {}, [], []
    file_records = {}
    scalars = [("batch", None, batch_loss)] + [(f"window_{i}", i, value) for i, value in enumerate(per_window)]
    try:
        for index, (role, slot, scalar) in enumerate(scalars):
            require(tuple(id(value) for value in tensors) == frozen_ids, "parameter_identity_changed_before_vjp")
            require(tuple(value._version for value in tensors) == frozen_versions, "parameter_version_changed_before_vjp")
            require(scalar.ndim == 0 and scalar.dtype is torch.float32 and bool(torch.isfinite(scalar)),
                    f"vjp_scalar_contract:{role}")
            if unit_meter is not None:
                unit_meter.synchronize_at(f"vjp_{index}:pre")
            else:
                sync()
            batch_slot = None if slot is None else BATCH_SLOT_BY_CANONICAL[slot]
            request = {"schema": SCHEMA + "/vjp_request", "call_id": index,
                       "role": role, "window_id": None if slot is None else WINDOWS[slot],
                       "batch_slot": batch_slot, "scalar_tensor_id": id(scalar), "roster_sha256": roster_sha,
                       "parameters": roster, "inputs": ["x0_pred", *names], "retain_graph": index < 6}
            request_sha = sha256(canonical_json(request))
            # The first input is the prediction, then every named trainable
            # parameter in optimizer-roster order. autograd.grad never fills .grad.
            if unit_meter is not None:
                got = unit_meter.unit("vjp", torch.autograd.grad, scalar,
                                      (observer["x0_pred"], *tensors), retain_graph=index < 6,
                                      create_graph=False, allow_unused=False, materialize_grads=False)
            else:
                got = torch.autograd.grad(scalar, (observer["x0_pred"], *tensors),
                                          retain_graph=index < 6, create_graph=False,
                                          allow_unused=False, materialize_grads=False)
                sync()
            h_rec, h = _tensor_digest(got[0])
            g_rows = []
            flat_parts = []
            for name, grad in zip(names, got[1:], strict=True):
                require(grad.shape == dict((r["name"], tuple(r["shape"])) for r in roster)[name],
                        f"gradient_shape:{role}:{name}")
                rec, host = _tensor_digest(grad)
                g_rows.append({"name": name, **rec})
                flat_parts.append(host.reshape(-1))
            g = np.concatenate(flat_parts).astype("<f4", copy=False)
            response = {"schema": SCHEMA + "/vjp_response", "call_id": index,
                        "request_sha256": request_sha, "role": role,
                        "h": h_rec, "g": g_rows, "g_flat": array_record(g)}
            response_sha = sha256(canonical_json(response))
            requests.append({**request, "request_sha256": request_sha})
            responses.append({**response, "response_sha256": response_sha})
            outputs[role] = {"h": h, "g": g}
            h_file = _npy_write(store, f"h_{index}.npy", h)
            g_file = _npy_write(store, f"g_{index}.npy", g)
            review = None
            if store is not None:
                file_records[f"h_{index}.npy"] = h_file
                file_records[f"g_{index}.npy"] = g_file
            del got, h, g, flat_parts
        require(tuple(id(value) for value in tensors) == frozen_ids, "parameter_identity_changed_after_vjp")
        require(tuple(value._version for value in tensors) == frozen_versions, "parameter_version_changed_after_vjp")
        require(parameter_state_digest(trainer) == before_state, "parameter_or_buffer_changed_during_vjp")
        require(_rng_equal(before_rng, _rng_snapshot()), "rng_changed_during_vjp")
        require(all(value.grad is prior is None for value, prior in zip(tensors, before_grad, strict=True)),
                "parameter_grad_field_written")
        return {"roster": roster, "roster_sha256": roster_sha, "requests": requests,
                "responses": responses, "outputs": outputs,
                "file_records": file_records,
                "parameter_state_sha256": before_state, "rng_unchanged": True,
                "parameter_grad_none": True}
    except Exception:
        # No hidden retry: caller owns the one-shot fixture/attempt and records
        # the exact entered prefix through this explicit exception attribute.
        raise


def _tree_norm_sq(rows: Iterable[np.ndarray]) -> float:
    total = np.float64(0.)
    for row in rows:
        flat = np.asarray(row, dtype=np.float64).ravel(order="C")
        total = np.float64(total + np.sum(flat * flat, dtype=np.float64))
    return float(total)


def _tree_dot(left: Iterable[np.ndarray], right: Iterable[np.ndarray]) -> float:
    total = np.float64(0.)
    for a, b in zip(left, right, strict=True):
        x, y = np.asarray(a, dtype=np.float64).ravel(order="C"), np.asarray(b, dtype=np.float64).ravel(order="C")
        require(x.shape == y.shape, "dot_roster_shape_mismatch")
        total = np.float64(total + np.sum(x * y, dtype=np.float64))
    return float(total)


def _rho(batch: np.ndarray, mean: np.ndarray, windows: list[np.ndarray]) -> tuple[float, float, float]:
    numerator = math.sqrt(max(0., _tree_norm_sq([batch - mean])))
    denominator = sum(math.sqrt(max(0., _tree_norm_sq([g]))) for g in windows) / 6.
    return numerator, denominator, numerator / denominator if denominator else (0. if numerator == 0 else math.inf)


def projection_metrics(g_batch: np.ndarray, g_windows: list[np.ndarray], e_all: float,
                      *, minimum_norm: float = MIN_COMMON_NORM) -> dict:
    """Prespecified common-window projection and robust threshold decision."""
    require(len(g_windows) == 6 and math.isfinite(float(e_all)) and e_all >= 0.,
            "projection_input_contract")
    common = np.asarray(g_windows[0], dtype=np.float64)
    common_norm = math.sqrt(max(0., _tree_norm_sq([common])))
    if common_norm <= minimum_norm:
        return {"common_norm": common_norm, "own": None, "other_i": None,
                "other_ratio": None, "batch_ratio": None, "margin": None,
                "batch_mean_difference": None, "direction_interval": None,
                "direction": None, "decision": "attribution_insufficient",
                "reason": "common_gradient_too_small", "cosines": None}
    own = _tree_dot([common], [common]) / 6.
    others = [_tree_dot([common], [np.asarray(g_windows[i], dtype=np.float64)]) / 6.
              for i in range(1, 6)]
    other_ratio = sum(others) / own
    batch_ratio = _tree_dot([common], [np.asarray(g_batch, dtype=np.float64)]) / own
    margin = common_norm * e_all / own
    interval = [1. + other_ratio - margin, 1. + other_ratio + margin]
    if other_ratio + margin <= -.25:
        decision, reason = "candidate", "prespecified_cancellation_gate_passed"
    elif other_ratio - margin > -.25:
        decision, reason = "attribution_insufficient", "cancellation_below_gate"
    else:
        decision, reason = "attribution_insufficient", "threshold_boundary_unresolved"
    direction = "down" if interval[0] > 0 else "up" if interval[1] < 0 else "undetermined"
    return {"common_norm": common_norm, "own": own, "other_i": others,
            "other_ratio": other_ratio, "batch_ratio": batch_ratio, "margin": margin,
            "batch_mean_difference": batch_ratio - (1. + other_ratio),
            "direction_interval": interval, "direction": direction, "decision": decision,
            "reason": reason, "cosines": [_cosine(common, np.asarray(g_windows[i], dtype=np.float64))
                                            for i in range(1, 6)]}


def analyze_gradients(bundle: dict) -> dict:
    """Independent float64 reduction, all window orderings explicit."""
    outputs = bundle["outputs"]
    require(set(outputs) == {"batch", *(f"window_{i}" for i in range(6))}, "vjp_role_roster")
    h_batch = outputs["batch"]["h"]
    gs = [outputs[f"window_{i}"]["g"] for i in range(6)]
    hs = [outputs[f"window_{i}"]["h"] for i in range(6)]
    g_batch = outputs["batch"]["g"]
    require(tuple(h_batch.shape) == (6, WINDOW, 38), "h_batch_shape")
    h_checks = []
    for i, h in enumerate(hs):
        require(tuple(h.shape) == (6, WINDOW, 38), f"h_shape:{i}")
        require(h.dtype == np.dtype("<f4") and np.isfinite(h).all(), f"h_dtype_or_finite:{i}")
        active_slot = BATCH_SLOT_BY_CANONICAL[i]
        for other in range(6):
            if other != active_slot:
                require(np.count_nonzero(h[other]) == 0, f"h_off_slot_nonzero:{i}:{other}")
        if np.count_nonzero(h) == 0:
            require(np.count_nonzero(gs[i]) == 0, f"zero_h_nonzero_g:{i}")
        expected = h[active_slot].astype(np.float64) / 6.
        diff = np.linalg.norm(h_batch[active_slot].astype(np.float64) - expected)
        bound = H_RTOL * np.linalg.norm(expected)
        passed = diff <= bound if bound else np.array_equal(h_batch[active_slot], expected)
        require(passed, f"h_batch_slot_mismatch:{i}:{diff:.9g}:{bound:.9g}")
        h_checks.append({"canonical_window": i, "batch_slot": active_slot,
                         "difference_l2": float(diff), "bound_l2": float(bound),
                         "gate_fraction": float(diff / bound) if bound else 0.,
                         "passed": bool(passed)})
    g_mean_parts = []
    for tensor_idx in range(len(bundle["roster"])):
        offset = sum(row["numel"] for row in bundle["roster"][:tensor_idx])
        end = offset + bundle["roster"][tensor_idx]["numel"]
        acc = np.zeros(end - offset, dtype=np.float64)
        for i in range(6):
            acc += gs[i][offset:end].astype(np.float64)
        g_mean_parts.append((acc / 6.).astype(np.float64))
    require(g_batch.dtype == np.dtype("<f4") and np.isfinite(g_batch).all(), "g_batch_dtype_or_finite")
    require(all(g.dtype == np.dtype("<f4") and np.isfinite(g).all() for g in gs),
            "g_window_dtype_or_finite")
    mean = np.concatenate(g_mean_parts)
    rows = []
    split = _side_split(bundle)
    boundaries = [0]
    for row in bundle["roster"]:
        boundaries.append(boundaries[-1] + row["numel"])

    def add_group(name: str, ids: list[int]):
        require(ids and ids == list(range(ids[0], ids[-1] + 1)), "parameter_group_not_contiguous")
        sl = slice(boundaries[ids[0]], boundaries[ids[-1] + 1])
        bpart, mpart = g_batch[sl].astype(np.float64), mean[sl]
        wparts = [g[sl].astype(np.float64) for g in gs]
        num, den, rho = _rho(bpart, mpart, wparts)
        rows.append({"group": name, "norm_batch_minus_mean": num,
                     "mean_window_norm_sum": den, "rho": rho,
                     "passed": rho <= PARAMETER_RHO_MAX})

    add_group("model", list(range(split)))
    add_group("conditioner", list(range(split, len(bundle["roster"]))))
    add_group("joint", list(range(len(bundle["roster"]))))
    for i, item in enumerate(bundle["roster"]):
        add_group("parameter:" + item["name"], [i])
    common = gs[0].astype(np.float64)
    common_norm = math.sqrt(max(0., _tree_norm_sq([common])))
    if common_norm <= MIN_COMMON_NORM:
        projection = {key: None for key in ("own", "other_i", "other_ratio", "batch_ratio", "margin",
                                             "batch_mean_difference", "direction_interval", "direction")}
        projection["common_norm"] = common_norm
    else:
        own = _tree_dot([common], [common]) / 6.
        others = [_tree_dot([common], [gs[i].astype(np.float64)]) / 6. for i in range(1, 6)]
        other_ratio = sum(others) / own
        batch_ratio = _tree_dot([common], [g_batch.astype(np.float64)]) / own
        joint_row = next(r for r in rows if r["group"] == "joint")
        margin = common_norm * (PARAMETER_RHO_MAX * joint_row["mean_window_norm_sum"]) / own
        interval = [1. + other_ratio - margin, 1. + other_ratio + margin]
        direction = "down" if interval[0] > 0 else "up" if interval[1] < 0 else "undetermined"
        projection = {"common_norm": common_norm, "own": own, "other_i": others,
                      "other_ratio": other_ratio, "batch_ratio": batch_ratio,
                      "margin": margin, "batch_mean_difference": batch_ratio - (1. + other_ratio),
                      "direction_interval": interval, "direction": direction,
                      "cosines": [_cosine(common, gs[i].astype(np.float64)) for i in range(1, 6)]}
        module_projection = {}
        for name, indices in (("model", range(0, split)),
                              ("conditioner", range(split, len(bundle["roster"])) )):
            lo, hi = boundaries[indices.start], boundaries[indices.stop]
            base = gs[0][lo:hi].astype(np.float64)
            module_projection[name] = [_tree_dot([base], [gs[i][lo:hi].astype(np.float64)]) / 6.
                                       for i in range(1, 6)]
        projection["module_other_i"] = module_projection
        projection["window_gradient_norms"] = [math.sqrt(max(0., _tree_norm_sq([g.astype(np.float64)])))
                                                for g in gs]
        projection["batch_direction_derivative_ratio"] = batch_ratio / 6.
    if projection["other_ratio"] is None:
        decision, reason = "attribution_insufficient", "common_gradient_too_small"
    elif projection["other_ratio"] + projection["margin"] <= -.25:
        decision, reason = "candidate", "prespecified_cancellation_gate_passed"
    elif projection["other_ratio"] - projection["margin"] > -.25:
        decision, reason = "attribution_insufficient", "cancellation_below_gate"
    else:
        decision, reason = "attribution_insufficient", "threshold_boundary_unresolved"
    return {"schema": SCHEMA + "/analysis", "groups": rows, "h_checks": h_checks,
            "max_h_gate_fraction": max(row["gate_fraction"] for row in h_checks),
            "projection": projection,
            "decision": decision, "reason": reason,
            "parameter_gate_passed": all(row["passed"] for row in rows)}


def _side_split(bundle):
    matches = [i for i, row in enumerate(bundle["roster"]) if row["name"].startswith("conditioner.")]
    require(matches and matches == list(range(matches[0], len(bundle["roster"]))), "parameter_group_order")
    return matches[0]


def _cosine(left, right):
    n1, n2 = math.sqrt(max(0., _tree_norm_sq([left]))), math.sqrt(max(0., _tree_norm_sq([right])))
    return _tree_dot([left], [right]) / (n1 * n2) if n1 and n2 else None


def verify_vjp_records(requests: list[dict], responses: list[dict], roster: list[dict]) -> list[dict]:
    require(len(requests) == len(responses) == 7, "vjp_record_count")
    pairs = []
    for i, (request_doc, response_doc) in enumerate(zip(requests, responses, strict=True)):
        request = dict(request_doc)
        response = dict(response_doc)
        request_sha = request.pop("request_sha256", None)
        response_sha = response.pop("response_sha256", None)
        role = "batch" if i == 0 else f"window_{i - 1}"
        slot = None if i == 0 else BATCH_SLOT_BY_CANONICAL[i - 1]
        require(request.get("call_id") == response.get("call_id") == i
                and request.get("role") == response.get("role") == role
                and request.get("batch_slot") == slot, "vjp_record_order_or_slot")
        require(request.get("parameters") == roster and request.get("inputs") == ["x0_pred", *[r["name"] for r in roster]],
                "vjp_record_roster_binding")
        require(request_sha == sha256(canonical_json(request)), "vjp_request_sha")
        require(response.get("request_sha256") == request_sha and response_sha == sha256(canonical_json(response)),
                "vjp_response_request_binding")
        require([row.get("name") for row in response.get("g", [])] == [row["name"] for row in roster],
                "vjp_gradient_name_order")
        for row, param in zip(response["g"], roster, strict=True):
            require(row["shape"] == param["shape"] and row["dtype"] == "<f4"
                    and row["bytes"] == param["numel"] * 4, "vjp_gradient_tensor_contract")
        pairs.append({"call_id": i, "request_sha256": request_sha,
                      "response_sha256": response_sha})
    return pairs


def _write_losses(store: ArtifactStore | None, values: dict):
    records = {}
    for key, value in values.items():
        if key == "scalars":
            continue
        host = np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
        records[f"loss_{key}.npy"] = {"file": _npy_write(store, f"loss_{key}.npy", host),
                                        "array": array_record(host)}
    return records


def run_synthetic_case(*, seed: int, input_seed: int, group: int, output: Path | None = None) -> dict:
    """One full D6-sized seven-VJP synthetic case; never reads a real source."""
    runtime = _runtime_setup("cuda:0")
    device = runtime["device"]
    fk = SyntheticFK(device, input_seed + 19)
    with tempfile.TemporaryDirectory(prefix="w1-shared-gradient-") as temp:
        items = synthetic_items(input_seed, group=group)
        sync = lambda: torch.cuda.synchronize("cuda:0")
        meter = UnitMeter(CORE_CALL_MATRIX, synchronize=sync,
                          expected_sync_points=CORE_SYNC_POINTS)
        dataset = meter.unit("registered_dataset_construct", NoReadRegisteredWindowDataset,
                             items, list(WINDOWS))
        trainer, loss = _make_trainer(seed, device, fk, Path(temp) / "trainer", dataset, meter)
        require(trainer.opt is None, "optimizer_construction_nonzero")
        trainer.accumulate_micro = lambda *_a, **_k: (_ for _ in ()).throw(
            SharedGradientError("parameter_update_forbidden:accumulate_micro"))
        trainer.clip_gradients = lambda *_a, **_k: (_ for _ in ()).throw(
            SharedGradientError("parameter_update_forbidden:clip_gradients"))
        trainer.step_optimizer = lambda *_a, **_k: (_ for _ in ()).throw(
            SharedGradientError("parameter_update_forbidden:step_optimizer"))
        cpu_batch = _batch_for_trainer(items, "cpu", meter)
        batch = meter.unit("batch_device_transfer", trainer._to_device, cpu_batch)
        handles, original_conditioner_forward = _install_runtime_counts(trainer, meter)
        original_decode = loss.decode_clip
        loss.decode_clip = lambda *a, **kw: meter.unit("loss_decode", original_decode, *a, **kw)
        original_fk = fk.site_world_positions
        def counted_fk(*args, **kwargs):
            return meter.unit("fk_call", original_fk, *args, **kwargs)
        fk.site_world_positions = counted_fk
        store = ArtifactStore(Path(output) / f"case_{group:02d}_seed_{seed}") if output is not None else None
        try:
            torch.cuda.reset_peak_memory_stats(0)
            observer = {}
            before_state = parameter_state_digest(trainer)
            before_forward_rng = _rng_snapshot()
            memory, prediction = meter.unit("trainer_predict", trainer.predict, batch)
            production_loss, _ = meter.unit("loss_assembly", loss, prediction, batch, observer)
            require(observer.get("contact_branch_executed") is False, "contact_branch_was_run")
            require(observer.get("total") is production_loss, "loss_observer_tensor_identity")
            require(parameter_state_digest(trainer) == before_state, "parameters_changed_during_forward")
            require(_rng_equal(before_forward_rng, _rng_snapshot()), "rng_changed_during_forward")
            batch_loss, per_window, value_arrays = _loss_items(observer, 6)
            cpu_values = {key: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
                          for key, value in value_arrays.items() if key != "scalars"}
            replay = independent_loss_scalars(cpu_values)
            for i, (actual, expected) in enumerate(zip(value_arrays["scalars"]["per_window"],
                                                         replay["per_window"], strict=True)):
                require(abs(actual - expected) <= LOSS_TOL * max(abs(actual), abs(expected)),
                        f"independent_window_loss_mismatch:{i}")
            batch_actual = value_arrays["scalars"]["batch"]
            expected_batch = replay["batch_terms"]["total"]
            require(abs(batch_actual - expected_batch) <= LOSS_TOL * max(abs(batch_actual), abs(expected_batch)),
                    "independent_batch_loss_mismatch")
            bundle = capture_vjps(trainer, batch, observer, batch_loss, per_window,
                                  store=store, synchronize=sync, unit_meter=meter)
            require(len(bundle["requests"]) == 7, "vjp_call_matrix")
            call_matrix = meter.verify_complete()
            analysis = analyze_gradients(bundle)
            require(all(t.grad is None for t in (*trainer.model.parameters(), *trainer.conditioner.parameters())),
                    "parameter_grad_field_written")
            require(analysis["parameter_gate_passed"], "synthetic_parameter_additivity_gate")
            require(all(row["rho"] < SYNTHETIC_ACCEPTANCE_RHO for row in analysis["groups"][:3]),
                    "synthetic_parameter_25pct_margin")
            loss_receipts = _write_losses(store, value_arrays)
            counts_doc = {"expected": CORE_CALL_MATRIX, "entered": call_matrix["entered"],
                          "completed": call_matrix["completed"], "dataset_reads": 0,
                          "optimizer_construction": 0, "parameter_updates": 0,
                          "contact_branch": 0, "cuda_sync_points": call_matrix["sync_points"],
                          "expected_cuda_sync_points": CORE_SYNC_POINTS}
            if store is not None:
                store.write_json("vjp_records.json", {"requests": bundle["requests"],
                                                       "responses": bundle["responses"],
                                                       "roster": bundle["roster"],
                                                       "roster_sha256": bundle["roster_sha256"],
                                                       "file_records": bundle["file_records"]})
                store.write_json("loss_scalars.json", {"production": value_arrays["scalars"],
                                                         "independent": replay,
                                                         "loss_arrays": loss_receipts,
                                                         "array_records": {k: array_record(v) for k, v in cpu_values.items()}})
                store.write_json("call_matrix.json", {**counts_doc, "events": call_matrix["events"],
                                                       "sync_events": call_matrix["sync_events"],
                                                       "artifact_io_expected": CORE_ARTIFACT_IO})
                outcome = {"schema": SCHEMA, "status": "synthetic_pass",
                           "analysis": analysis, "counts": counts_doc,
                           "files": store.records}
                store.write_json("outcome.json", outcome)
                review = independently_review_artifacts(store.root, analysis)
                require(review["io"]["artifact_reads"] == CORE_ARTIFACT_IO["artifact_reads"]
                        and review["io"]["file_sha256_verified"] == CORE_ARTIFACT_IO["file_sha256_verified"],
                        "artifact_readback_io_matrix_mismatch")
                store.write_json("independent_review.json", review)
                require(store.write_count == CORE_ARTIFACT_IO["npy_writes"] + CORE_ARTIFACT_IO["json_writes"]
                        and store.file_sha256_count == CORE_ARTIFACT_IO["file_sha256_writes"],
                        "artifact_write_io_matrix_mismatch")
            return {"schema": SCHEMA, "seed": seed, "input_seed": input_seed, "group": group,
                    "runtime": runtime, "analysis": analysis, "counts": counts_doc,
                     "call_matrix": call_matrix,
                     "independent_review": review,
                     "independent_review_sha256": None if review is None else sha256(canonical_json(review)),
                     "independent_review_bytes": 0 if review is None else len(canonical_json(review)),
                     "max_cuda_bytes": torch.cuda.max_memory_allocated(0),
                    "parameter_state_unchanged": True, "vjp_rng_unchanged": bundle["rng_unchanged"],
                     "independent_loss_scalars": replay,
                     "artifact_files": [] if store is None else sorted(store.records),
                     "artifact_peak_bytes": 0 if store is None else store.peak_bytes,
                     "artifact_io": None if store is None else {
                         "npy_writes": 20, "json_writes": store.write_count - 20,
                         "file_sha256_writes": store.file_sha256_count,
                         "artifact_reads": review["io"]["artifact_reads"],
                         "file_sha256_verified": review["io"]["file_sha256_verified"]}}
        finally:
            for handle in handles:
                handle.remove()
            trainer.conditioner.forward = original_conditioner_forward
            if store is not None:
                del store


def independently_review_artifacts(root: Path, expected_analysis: dict | None = None) -> dict:
    """Re-read the one saved synthetic case and recompute its CPU verdict."""
    root = Path(root)
    snapshots: dict[str, bytes] = {}
    file_hash_count = 0

    def read_once(name: str) -> bytes:
        require(Path(name).name == name, "independent_artifact_path")
        if name not in snapshots:
            snapshots[name] = (root / name).read_bytes()
        return snapshots[name]

    outcome = json.loads(read_once("outcome.json"))
    require(outcome.get("schema") == SCHEMA and outcome.get("status") == "synthetic_pass",
            "independent_outcome_not_complete")
    require(isinstance(outcome.get("files"), dict), "independent_outcome_file_roster")
    for name, receipt in outcome["files"].items():
        require(Path(name).name == name and isinstance(receipt, dict), "independent_outcome_file_entry")
        raw = read_once(name)
        require(len(raw) == receipt.get("bytes") and sha256(raw) == receipt.get("sha256"),
                "independent_outcome_file_binding:" + name)
        file_hash_count += 1
    require(outcome.get("analysis") == expected_analysis, "independent_outcome_analysis_binding")
    vjp = json.loads(read_once("vjp_records.json"))
    loss = json.loads(read_once("loss_scalars.json"))
    roster = vjp["roster"]
    require(len(roster) == PARAMETER_TENSORS and sum(row["numel"] for row in roster) == PARAMETER_COUNT,
            "independent_parameter_roster")
    require(vjp["roster_sha256"] == sha256(canonical_json(roster)), "independent_roster_sha")
    expected_npy = {f"h_{i}.npy" for i in range(7)} | {f"g_{i}.npy" for i in range(7)} \
        | {f"loss_{name}.npy" for name in LOSS_NAMES}
    npy_paths = {path.name for path in root.iterdir() if path.suffix == ".npy"}
    require(npy_paths == expected_npy, "independent_npy_file_set")
    require(set(vjp["file_records"]) == {f"h_{i}.npy" for i in range(7)}
            | {f"g_{i}.npy" for i in range(7)}, "independent_vjp_file_roster")
    require(len(vjp["requests"]) == len(vjp["responses"]) == 7, "independent_vjp_count")
    pairs = verify_vjp_records(vjp["requests"], vjp["responses"], roster)
    arrays: dict[str, np.ndarray] = {}
    for i, (request, response) in enumerate(zip(vjp["requests"], vjp["responses"], strict=True)):
        request = dict(request)
        response = dict(response)
        request_sha = request.pop("request_sha256")
        response_sha = response.pop("response_sha256")
        role = "batch" if i == 0 else f"window_{i - 1}"
        require(request["call_id"] == response["call_id"] == i and request["role"] == response["role"] == role,
                "independent_vjp_order")
        require(request_sha == sha256(canonical_json(request)), "independent_request_binding")
        require(response["request_sha256"] == request_sha and response_sha == sha256(canonical_json(response)),
                "independent_response_binding")
        for prefix, metadata in (("h", response["h"]), ("g", response["g_flat"])):
            name = f"{prefix}_{i}.npy"
            raw = read_once(name)
            file_rec = vjp["file_records"][name]
            require(file_rec == outcome["files"].get(name),
                    f"independent_npy_file_binding:{name}")
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            require(array.dtype == np.dtype("<f4") and list(array.shape) == metadata["shape"],
                    f"independent_npy_shape_dtype:{name}")
            require(array_record(array) == metadata, f"independent_vjp_array_binding:{name}")
            arrays[name] = array
        flat = arrays[f"g_{i}.npy"]
        offset = 0
        for row, param in zip(response["g"], roster, strict=True):
            size = param["numel"]
            tensor_values = flat[offset:offset + size].reshape(param["shape"])
            require(array_record(tensor_values) == {key: row[key] for key in ("shape", "dtype", "bytes", "sha256")},
                    f"independent_named_gradient_bytes:{i}:{param['name']}")
            offset += size
        require(offset == flat.size, f"independent_flat_gradient_length:{i}")
    loss_values = {}
    for key in LOSS_NAMES:
        name = f"loss_{key}.npy"
        raw = read_once(name)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        receipt = loss["loss_arrays"][name]
        require(receipt["file"] == outcome["files"].get(name),
                f"independent_loss_file_sha:{key}")
        require(arr.dtype == np.dtype("<f4") and array_record(arr) == receipt["array"],
                f"independent_loss_array_binding:{key}")
        require(receipt["array"] == loss["array_records"][key], f"independent_loss_array_record:{key}")
        loss_values[key] = arr
    scalar_replay = independent_loss_scalars(loss_values)
    require(canonical_json(scalar_replay) == canonical_json(loss["independent"]),
            "independent_loss_scalar_replay")
    production = loss.get("production")
    require(isinstance(production, dict) and len(production.get("per_window", [])) == 6
            and isinstance(production.get("terms"), dict), "independent_production_scalars_missing")
    for i, (actual, expected) in enumerate(zip(production["per_window"], scalar_replay["per_window"], strict=True)):
        require(abs(actual - expected) <= LOSS_TOL * max(abs(actual), abs(expected)),
                f"independent_production_window_scalar:{i}")
    for name in (*LOSS_NAMES, "total"):
        actual = production["batch"] if name == "total" else production["terms"].get(name)
        expected = scalar_replay["batch_terms"].get(name)
        require(isinstance(actual, (int, float)) and isinstance(expected, (int, float))
                and abs(actual - expected) <= LOSS_TOL * max(abs(actual), abs(expected)),
                "independent_production_batch_scalar:" + name)
    call_doc = json.loads(read_once("call_matrix.json"))
    require(call_doc.get("expected") == CORE_CALL_MATRIX
            and call_doc.get("entered") == CORE_CALL_MATRIX
            and call_doc.get("completed") == CORE_CALL_MATRIX
            and call_doc.get("cuda_sync_points") == CORE_SYNC_POINTS
            and call_doc.get("expected_cuda_sync_points") == CORE_SYNC_POINTS
            and len(call_doc.get("sync_events", [])) == CORE_SYNC_POINTS
            and call_doc.get("artifact_io_expected") == CORE_ARTIFACT_IO,
            "independent_call_matrix_mismatch")
    require(len(snapshots) == CORE_ARTIFACT_IO["artifact_reads"]
            and file_hash_count == CORE_ARTIFACT_IO["file_sha256_verified"],
            "independent_artifact_read_count")
    bundle = {"roster": roster, "outputs": {
        "batch": {"h": arrays["h_0.npy"], "g": arrays["g_0.npy"]},
        **{f"window_{i}": {"h": arrays[f"h_{i + 1}.npy"], "g": arrays[f"g_{i + 1}.npy"]}
           for i in range(6)}}}
    analysis = analyze_gradients(bundle)
    if expected_analysis is not None:
        require(canonical_json(analysis) == canonical_json(expected_analysis), "independent_analysis_mismatch")
    return {"schema": SCHEMA + "/independent_review", "passed": True,
            "vjp_pairs": pairs, "array_files": len(expected_npy),
            "io": {"artifact_reads": len(snapshots), "file_sha256_verified": file_hash_count,
                   "expected_artifact_reads": CORE_ARTIFACT_IO["artifact_reads"],
                   "expected_file_sha256_verified": CORE_ARTIFACT_IO["file_sha256_verified"]},
            "projection": analysis["projection"], "groups": analysis["groups"],
            "decision": analysis["decision"], "parameter_gate_passed": analysis["parameter_gate_passed"]}


def require_real_authorization(authorization=None):
    """Hard refusal before path resolution, hashing, file reads or model work."""
    require(authorization is None, "real_execution_requires_new_frozen_authorization")
    raise SharedGradientError("real_execution_not_implemented_or_authorized")


def real_entry(*args, authorization=None, **kwargs):
    # Deliberately do not inspect args/kwargs: they may contain source paths.
    return require_real_authorization(authorization)


def run_cuda_synthetic_acceptance(output: Path) -> dict:
    """Frozen 9-input × 2-initialization CUDA/float32 synthetic acceptance."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    implementation_before = _implementation_identity()
    model_seeds = (20261008, 20261122)
    summaries = []
    for seed in model_seeds:
        for group in range(9):
            if seed == model_seeds[0] and group == 0:
                row = run_synthetic_case(seed=seed, input_seed=20261130 + group,
                                         group=group, output=root)
                row["case_artifact_mode"] = "retained"
                summaries.append(row)
                del row
                torch.cuda.empty_cache()
                continue
            with tempfile.TemporaryDirectory(prefix=f"case_{seed}_{group}_", dir=root) as scratch:
                row = run_synthetic_case(seed=seed, input_seed=20261130 + group,
                                         group=group, output=Path(scratch))
                row["case_artifact_mode"] = "independently_reviewed_then_removed"
                summaries.append(row)
                del row
                torch.cuda.empty_cache()
    maxima = {name: max(row["analysis"]["groups"][index]["rho"] for row in summaries)
              for index, name in enumerate(("model", "conditioner", "joint"))}
    require(all(value < SYNTHETIC_ACCEPTANCE_RHO for value in maxima.values()),
            "cuda_synthetic_25pct_acceptance_margin")
    require(all(row["analysis"]["parameter_gate_passed"] for row in summaries),
            "cuda_synthetic_parameter_gate")
    implementation_after = _implementation_identity()
    require(implementation_after == implementation_before, "implementation_changed_during_acceptance")
    result = {"schema": SCHEMA + "/cuda_synthetic_acceptance", "passed": True,
              "model_seeds": list(model_seeds), "input_groups_per_seed": 9,
              "cases": len(summaries), "max_group_rho": maxima,
              "synthetic_acceptance_rho": SYNTHETIC_ACCEPTANCE_RHO,
              "call_matrix": CORE_CALL_MATRIX,
              "cuda_sync_points_per_case": CORE_SYNC_POINTS,
              "artifact_io_per_case": CORE_ARTIFACT_IO,
              "implementation_identity": implementation_before,
              "implementation_identity_reads": 2 * len(IC_IMPLEMENTATION_FILES),
              "runtime": summaries[0]["runtime"],
              "peak_cuda_allocated_bytes": max(row["max_cuda_bytes"] for row in summaries),
              "peak_case_artifact_bytes": max(row["artifact_peak_bytes"] for row in summaries),
              "saved_full_case": "case_00_seed_20261008",
              "real_archive_reads": 0, "checkpoint_loads": 0,
              "production_asset_reads": 0, "parameter_updates": 0,
              "cases_summary": [{"seed": row["seed"], "input_seed": row["input_seed"],
                                  "group_rho": {item["group"]: item["rho"]
                                                for item in row["analysis"]["groups"][:3]},
                                  "max_h_gate_fraction": row["analysis"]["max_h_gate_fraction"],
                                  "all_parameter_gates_passed": row["analysis"]["parameter_gate_passed"],
                                  "independent_review_passed": bool(row["independent_review"]
                                                                    and row["independent_review"]["passed"]),
                                  "independent_review_sha256": row["independent_review_sha256"],
                                  "independent_review_bytes": row["independent_review_bytes"],
                                  "observed_call_matrix": row["call_matrix"]["completed"],
                                  "observed_cuda_sync_points": row["call_matrix"]["sync_points"],
                                  "artifact_peak_bytes": row["artifact_peak_bytes"],
                                  "artifact_io": row["artifact_io"],
                                  "artifact_mode": row["case_artifact_mode"]}
                                 for row in summaries]}
    raw = canonical_json(result)
    path = root / "cuda_acceptance.json"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(root)
    return {**result, "acceptance_file": {"bytes": len(raw), "sha256": sha256(raw)}}


def _implementation_identity() -> dict:
    project_root = Path(__file__).resolve().parents[3]
    rows = []
    for relpath in IC_IMPLEMENTATION_FILES:
        path = project_root / relpath
        raw = path.read_bytes()
        rows.append({"path": relpath, "bytes": len(raw), "sha256": sha256(raw)})
    return {"files": rows, "sha256": sha256(canonical_json(rows))}


def run_cpu_synthetic_acceptance() -> dict:
    """Hand-check the projection decision table without constructing a model."""
    checks = []
    g0 = np.array([6., 0.])
    orthogonal = [g0, *[np.array([0., 1.]) for _ in range(5)]]
    facts = projection_metrics(np.mean(orthogonal, axis=0), orthogonal, 0.)
    require(facts["own"] == 6. and facts["other_ratio"] == 0.
            and facts["batch_ratio"] == 1., "cpu_control_orthogonal")
    checks.append("orthogonal")
    cancel = [np.array([6., 0.]), np.array([-3., 0.]), *[np.array([0., 1.]) for _ in range(4)]]
    facts = projection_metrics(np.mean(cancel, axis=0), cancel, 0.)
    require(facts["other_ratio"] == -.5 and facts["batch_ratio"] == .5
            and facts["decision"] == "candidate", "cpu_control_cancellation")
    checks.append("cancellation")
    reverse = [np.array([6., 0.]), np.array([-12., 0.]), *[np.array([0., 1.]) for _ in range(4)]]
    facts = projection_metrics(np.mean(reverse, axis=0), reverse, 0.)
    require(facts["batch_ratio"] == -1. and facts["direction"] == "up", "cpu_control_reversal")
    checks.append("direction_reversal")
    threshold = [np.array([6., 0.]), np.array([-1.5, 0.]), *[np.array([0., 1.]) for _ in range(4)]]
    edge = projection_metrics(np.mean(threshold, axis=0), threshold, 0.)
    require(edge["other_ratio"] == -.25 and edge["decision"] == "candidate",
            "cpu_control_threshold")
    checks.append("threshold")
    zero = [np.zeros(2) for _ in range(6)]
    require(projection_metrics(np.zeros(2), zero, 0.)["other_ratio"] is None,
            "cpu_control_zero_common")
    checks.append("zero_norm")
    windows = [np.array([1., 0.]), np.array([0., 100.]), *[np.array([0., -10.]) for _ in range(4)]]
    original_mean = np.mean(windows, axis=0)
    swapped = windows.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    before = projection_metrics(original_mean, windows, 0.)
    after = projection_metrics(original_mean, swapped, 0.)
    require(np.array_equal(original_mean, np.mean(swapped, axis=0))
            and before["other_ratio"] != after["other_ratio"], "cpu_control_order_binding")
    checks.append("paired_order")
    return {"schema": SCHEMA + "/cpu_acceptance", "passed": True,
            "checks": checks, "check_count": len(checks),
            "real_archive_reads": 0, "checkpoint_loads": 0,
            "model_constructions": 0, "parameter_updates": 0}
