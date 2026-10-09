"""I0 thin entry: generate U0 teachers with the pinned upstream checkout.

Plan omniretarget_first_training_plan §3.3/§10.1: drive the independently
managed upstream environment (``derived/upstream/holosoma_env``) over the
frozen pilot list, with the reference-only upstream initialization (no
oracle warm-start), then verify each output against the published teacher
(bit-exact human reference + target box trajectory + fps/T binding) and
convert the checkout's MuJoCo [pos,quat] layout into the project scene
convention.

Read-only over resources/ and third_party/ (input hashes re-verified after
the run). Raw upstream outputs are kept next to the converted copies. All
expected demos keep a result row; failures never shrink the denominator.

Run: uv run --locked python scripts/generate_omniretarget_teachers.py \
       --freeze derived/experiments/omniretarget_first/<a1_run>/freeze \
       --out-dir derived/experiments/omniretarget_first/<a2_run>/teacher_pilot/u0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data import omniretarget_sources as orc
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR

HOLOSOMA_PACKAGE = (
    PROJECT_ROOT / "third_party" / "holosoma" / "src" / "holosoma_retargeting"
    / "holosoma_retargeting"
)
DEFAULT_ENV_PYTHON = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"


def pilot_demos_from_freeze(freeze_dir: Path) -> list[str]:
    lists = json.loads((freeze_dir / "frozen_lists.json").read_text())
    return [e["demo"] for e in lists["teacher_pilot_train6_v1"]]


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_one(
    demo: str,
    env_python: Path,
    out_dir: Path,
    data_path: Path,
) -> dict:
    """Run the upstream retargeting for one demo; -> per-demo result row."""
    t0 = time.time()
    cmd = [
        str(env_python), "examples/robot_retarget.py",
        "--task-type", "object_interaction",
        "--task-name", demo,
        "--data-format", "smplh",
        "--data-path", str(data_path),
        "--save-dir", str(out_dir),
        "--task-config.object-name", "largebox",
    ]
    proc = subprocess.run(
        cmd, cwd=HOLOSOMA_PACKAGE, capture_output=True, text=True
    )
    row = {
        "demo": demo,
        "argv": cmd,
        "cwd": str(HOLOSOMA_PACKAGE),
        "exit_code": proc.returncode,
        "wall_s": round(time.time() - t0, 2),
        "config": "upstream defaults: joint_limits/obj_non_penetration/foot_sticking on, "
                  "visualize/debug off, augmentation off (reference-only init)",
        "status": None, "binding": None, "error": None,
    }
    if proc.returncode != 0:
        row["status"] = "generation_failed"
        row["error"] = (proc.stderr or proc.stdout)[-2000:]
        return row
    try:
        raw_path = out_dir / f"{demo}_original.npz"
        pub_path = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
        scene_u0, human_u0, fps_u0 = orc.load_teacher_scene(raw_path, "upstream_generated")
        scene_p, human_p, fps_p = orc.load_teacher_scene(pub_path, "published")
        binding = orc.check_binding(demo, scene_u0, human_u0, fps_u0, scene_p, human_p, fps_p)
        conv_path = out_dir / f"{demo}_original_converted.npz"
        np.savez(
            conv_path,
            qpos=scene_u0,
            human_joints=human_u0,
            fps=fps_u0,
            source_type="upstream_generated",
            layout="project_scene_v1",
            provenance=json.dumps({
                "demo": demo, "raw_file": raw_path.name, "raw_sha256": _sha(raw_path),
                "published_sha256": _sha(pub_path),
                "binding": binding.to_json_dict(),
            }),
        )
        row["status"] = "ok" if binding.passed else "binding_failed"
        row["binding"] = binding.to_json_dict()
        row["raw_sha256"] = _sha(raw_path)
        row["converted"] = str(conv_path.relative_to(PROJECT_ROOT))
        row["T"] = int(scene_u0.shape[0])
    except Exception as exc:
        row["status"] = "postcheck_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True,
                    help="A1 freeze directory (frozen_lists.json)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--env-python", type=Path, default=DEFAULT_ENV_PYTHON)
    ap.add_argument("--data-path", type=Path, default=OMOMO_NEW_DIR)
    ap.add_argument("--demos", type=str, default=None,
                    help="comma-separated override (dev retry of specific demos); "
                         "the frozen pilot list remains the expected denominator")
    args = ap.parse_args()

    demos = pilot_demos_from_freeze(args.freeze)
    if args.demos:
        wanted = [d.strip() for d in args.demos.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in demos]
        if unknown:
            raise SystemExit(f"--demos outside frozen pilot list: {unknown}")
        demos = wanted
    out_dir = args.out_dir.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    # The upstream subprocess runs with cwd inside the pinned read-only
    # checkout; a relative --save-dir would scatter outputs there (found in
    # the first dev run). Refuse anything under the read-only roots.
    for forbidden in (PROJECT_ROOT / "third_party", PROJECT_ROOT / "resources"):
        if out_dir == forbidden or forbidden in out_dir.parents:
            raise SystemExit(f"--out-dir must not live under {forbidden.name}/: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input immutability: hash the bound inputs before and after generation.
    inputs = {}
    for demo in demos:
        for key, p in (("npz", TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
                       ("pt", OMOMO_NEW_DIR / f"{demo}.pt")):
            inputs[f"{demo}:{key}"] = _sha(p)
    env_lock_sha = _sha(args.env_python.parents[2] / "uv.lock")

    rows = [generate_one(d, args.env_python, out_dir, data_path) for d in demos]

    for demo in demos:
        for key, p in (("npz", TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
                       ("pt", OMOMO_NEW_DIR / f"{demo}.pt")):
            if inputs[f"{demo}:{key}"] != _sha(p):
                for row in rows:
                    if row["demo"] == demo and row["status"] == "ok":
                        row["status"] = "input_mutated"

    report = {
        "stage": "I0 U0 generation (plan §3.3/§10.1)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "freeze": str(args.freeze),
        "env_python": str(args.env_python),
        "env_uv_lock_sha256": env_lock_sha,
        "holosoma_checkout": "bccd4d7451640a2800ddc77e469d911a84f91994",
        "expected": len(demos),
        "n_ok": sum(r["status"] == "ok" for r in rows),
        "n_binding_failed": sum(r["status"] == "binding_failed" for r in rows),
        "n_generation_failed": sum(r["status"] == "generation_failed" for r in rows),
        "n_postcheck_failed": sum(r["status"] == "postcheck_failed" for r in rows),
        "input_mutated": any(r["status"] == "input_mutated" for r in rows),
        "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
        "rows": rows,
    }
    out = out_dir / "generation_manifest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in report if not k.startswith("rows")}, indent=2))
    print(f"manifest -> {out}")
    ok = report["n_ok"] == report["expected"] and not report["input_mutated"]
    print(f"generation: {'COMPLETE' if ok else 'INCOMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
