"""Experiment provenance snapshot (m2_m3_stage_gate_fix_plan sections 3/10).

A run's report must be reproducible and comparable without /tmp probes:
this module records the exact code revision (plus a hash of any dirty-tree
diff), the effective strict/relaxed configs, input and asset hashes, the
dependency lock, thread-limit environment and the number of parallel
workers a run used.

CLI: write a standalone snapshot JSON for a registered demo set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from closed_loop_metrics import git_rev, sha256  # noqa: E402
from ireset.paths import DERIVED_ROBOTS_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR  # noqa: E402
from ireset.repair.config_io import MODES, load_mode_configs  # noqa: E402

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=PROJECT_ROOT).stdout


def git_state() -> dict:
    status = _git("status", "--porcelain")
    return {
        "rev": git_rev(),
        "dirty": bool(status.strip()),
        "status_sha256": _sha16(status),
        "diff_sha256": _sha16(_git("diff", "HEAD")),
    }


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
    out = {"note": note, "git": git_state(), "configs": effective_configs(),
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
