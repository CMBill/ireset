"""Experiment provenance snapshot (m2_m3_stage_gate_fix_plan sections 3/10,
extended by the review-fix plan section 5.2 / R9).

A run's report must be reproducible and comparable without /tmp probes:
this module records the exact code revision (plus a hash of any dirty-tree
diff AND the content of files git does not track), the effective configs of
the RUNNING instance, input and asset hashes, the dependency lock, thread
limits and the number of parallel workers a run used.

Two different things are deliberately separated (R9):

- ``provenance()`` -- a standalone SNAPSHOT (CLI): it declares the configs
  that would be loaded for the registered demos. It never claims to be some
  experiment's effective parameters, because no instance exists.
- ``effective_record()`` -- the effective configuration of ONE run instance,
  captured AFTER all CLI/YAML overrides and BEFORE the first solve, with the
  default dataclasses recorded separately (a default must never masquerade
  as an effective value).

CLI: write a standalone snapshot JSON for a registered demo set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from closed_loop_metrics import git_rev, sha256  # noqa: E402
from ireset.paths import DERIVED_ROBOTS_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR  # noqa: E402
from ireset.repair.config_io import MODES, load_mode_configs  # noqa: E402
from ireset.repair.context import RepairConfig  # noqa: E402
from ireset.repair.validation import ValidationConfig  # noqa: E402

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

#: directories whose untracked/modified contents must be recoverable from a
#: dirty-run record (code that can change behaviour)
_CODE_DIRS = ("src/", "scripts/", "configs/", "tests/")


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=PROJECT_ROOT).stdout


def canonical_json(obj) -> str:
    """Deterministic JSON (sorted keys, compact) for hashing configs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_sha256(*cfgs) -> str:
    """Hash of a tuple of effective configs (field-by-field, canonical)."""
    return hashlib.sha256(
        canonical_json([asdict(c) for c in cfgs]).encode()).hexdigest()[:16]


def config_roundtrip(cfg):
    """``asdict -> JSON -> constructor`` round-trip of an effective config.

    An exported config that cannot be reconstructed is not a usable record;
    this raises instead of silently writing a lossy snapshot (R9 regression).
    """
    payload = json.loads(json.dumps(asdict(cfg)))
    back = type(cfg)(**payload)
    if back != cfg:
        raise ValueError(f"config round-trip mismatch for {type(cfg).__name__}: "
                         f"{payload}")
    return back


def git_state() -> dict:
    status = _git("status", "--porcelain")
    return {
        "rev": git_rev(),
        "dirty": bool(status.strip()),
        "status_sha256": _sha16(status),
        "diff_sha256": _sha16(_git("diff", "HEAD")),
    }


def effective_record(rcfg, vcfg, *, argv, mode: str, validation_mode: str,
                     demos: list[str], frames: int, note: str | None = None,
                     task_gate_yaml: str | None = None,
                     screening: dict | None = None,
                     profile: dict | None = None) -> dict:
    """Effective configuration of ONE run instance, after every override.

    Round-trips both configs through JSON before exporting them, records the
    task-gate YAML path AND hash, and keeps the default dataclasses under
    explicit ``default_*`` keys plus an ``is_default_*`` flag -- a default
    value must never be presented as the instance's effective parameter (R9).
    ``profile`` (train24fix3) freezes the resolved profile and the registry
    hash alongside the argv, so a run's solver-behaviour changes are
    attributable to an exact registry revision.
    """
    config_roundtrip(rcfg)
    config_roundtrip(vcfg)
    gate_path = Path(task_gate_yaml) if task_gate_yaml else None
    rec = {
        "argv": [str(a) for a in argv],
        "mode": mode,
        "validation_mode": validation_mode,
        "demos": list(demos),
        "frames": int(frames),
        "note": note,
        "screening": dict(screening or {}),
        "task_gate_yaml": None if gate_path is None else str(gate_path),
        "task_gate_yaml_sha256": (None if gate_path is None or not gate_path.exists()
                                  else sha256(gate_path)),
        "task_mpjpe_tol_m": float(rcfg.task_mpjpe_tol_m),
        "effective_repair_config": asdict(rcfg),
        "effective_validation_config": asdict(vcfg),
        "effective_config_sha256": config_sha256(rcfg, vcfg),
        "default_repair_config": asdict(RepairConfig()),
        "default_validation_config": asdict(ValidationConfig()),
        "is_default_repair_config": RepairConfig() == rcfg,
        "is_default_validation_config": ValidationConfig() == vcfg,
        "profile": profile,
    }
    return rec


def config_unchanged(rec: dict, rcfg, vcfg) -> bool:
    """True when the running configs still equal the record's effective ones.

    Called at the END of a run: a parameter mutated mid-run must be flagged,
    not silently reported as the startup value (R9).
    """
    return (asdict(rcfg) == rec["effective_repair_config"]
            and asdict(vcfg) == rec["effective_validation_config"])


def dirty_evidence(out_dir: Path) -> dict:
    """Persist what a dirty working tree ACTUALLY contains (R9).

    A diff hash alone cannot reconstruct the run: the tracked diff is written
    as a real patch and every untracked/modified file under the code
    directories is copied into ``<out_dir>/dirty_files/`` with its sha256. A
    clean tree records ``dirty: false`` and nothing else.
    """
    status = _git("status", "--porcelain")
    out = {"dirty": bool(status.strip()), "status": status,
           "diff_sha256": _sha16(_git("diff", "HEAD")), "files": []}
    if not out["dirty"]:
        return out
    stamp = time.strftime("%Y%m%d_%H%M%S")
    patch = out_dir / f"dirty_diff_{stamp}.patch"
    patch.write_text(_git("diff", "HEAD"))
    out["patch_path"] = str(patch)
    out["patch_sha256"] = sha256(patch)
    copy_root = out_dir / "dirty_files"
    for line in status.splitlines():
        code, _, rel = line[:2], line[2:3], line[3:].strip()
        if not rel.startswith(_CODE_DIRS) or code.strip() == "D":
            continue
        src = PROJECT_ROOT / rel
        if not src.is_file():
            continue
        dst = copy_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out["files"].append({"path": rel, "status": code.strip(),
                             "sha256": sha256(src), "copy": str(dst)})
    return out


def effective_configs() -> dict:
    out = {}
    for mode in MODES:
        rcfg, vcfg = load_mode_configs(mode)
        out[mode] = {"repair": asdict(rcfg), "validation": asdict(vcfg)}
    return out


def asset_hashes() -> dict:
    files = sorted(p for p in DERIVED_ROBOTS_DIR.rglob("*")
                   if p.is_file() and "/assets/" in str(p))
    files += [DERIVED_ROBOTS_DIR / n for n in
              ("g1_29dof_locked.xml", "g1_29dof_locked_w_largebox.xml",
               "g1_29dof_locked.urdf", "lock_record.json")]
    return {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in files if p.exists()}


def _npz_frames(path: Path) -> tuple[int, float]:
    import numpy as np

    with np.load(path) as d:
        return int(d["qpos"].shape[0]), float(d["fps"])


def input_hashes(demos: list[str]) -> list[dict]:
    rows = []
    for demo in demos:
        src = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
        T, fps = _npz_frames(src)
        rows.append({"demo": demo, "path": str(src), "sha256": sha256(src),
                     "T": T, "fps": fps})
    return rows


def env_snapshot() -> dict:
    import mujoco
    import numpy
    import osqp
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "mujoco": mujoco.__version__,
        "osqp": osqp.__version__,
        "torch": torch.__version__,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
    }


def provenance(demos: list[str] | None = None, note: str | None = None) -> dict:
    """Standalone SNAPSHOT for a demo set (no running instance).

    ``configs`` here are the configs that WOULD be loaded for the requested
    modes -- a snapshot of the loaders, NOT the effective parameters of an
    experiment. Run reports must use :func:`effective_record` instead (R9);
    the ``kind`` field makes the distinction machine-readable.
    """
    out = {"kind": "snapshot", "note": note, "git": git_state(),
           "configs": effective_configs(),
           "configs_are_loader_defaults_not_effective": True,
           "assets": asset_hashes(), "env": env_snapshot(),
           "uv_lock_sha256": sha256(PROJECT_ROOT / "uv.lock")}
    if demos:
        out["inputs"] = input_hashes(demos)
    return out


def main() -> None:
    from regenerate_teachers import registered_set

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="diagnosis6")
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    demos = args.demos or registered_set(args.set)
    snap = provenance(demos, note=args.note)
    snap["set"] = args.set
    snap["demos"] = demos
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=1, default=float))
    print(f"provenance -> {out} ({len(demos)} demos, rev {snap['git']['rev']}"
          f"{' dirty' if snap['git']['dirty'] else ''})")


if __name__ == "__main__":
    main()
