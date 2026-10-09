"""I0 thin entry: generate U0 teachers with the pinned upstream checkout.

Plan omniretarget_first_training_plan §3.3/§10.1: drive the independently
managed upstream environment (``derived/upstream/holosoma_env``) over the
frozen pilot list, with the reference-only upstream initialization (no
oracle warm-start), then verify each output against the published teacher
(bit-exact human reference + target box trajectory + fps/T binding) and
convert the checkout's MuJoCo [pos,quat] layout into the project scene
convention.

Run-integrity rules (A2 review fixes 5/6/7/10, 2026-10-09):

* the upstream checkout identity is MEASURED (git rev + dirty state) and
  must match the A1-frozen expected commit before anything runs; the env
  lock hash and mujoco version are measured and recorded;
* every invocation writes its own ``attempt_NNN`` directory and appends to
  ``attempts_index.json`` — a ``--demos`` retry never overwrites prior
  evidence, and ``expected_denominator`` stays the FULL frozen pilot list
  so a partial retry reports INCOMPLETE, never a shrunken COMPLETE;
* input hashes are recorded in the manifest (the comparison re-verifies
  them against the on-disk files) in addition to the before/after
  immutability check inside this run; round-3: the on-disk inputs and all
  A1-frozen asset files are additionally verified against the FREEZE hashes
  and any mismatch stops the run before generation (review 3); round-4:
  the PT-side verification and the immutability anchors cover the ACTUAL
  ``--data-path`` consumed by this run, and the consumed data_path is
  recorded in the manifest (review 2);
* the env binary must match its own lock's pinned mujoco version (a
  re-resolved environment behind an unchanged lock hash is refused);
* the converted artifact's sha256 is recorded alongside the raw's, so the
  comparison can verify the full raw->converted chain (review 5);
* output directories must fully resolve under ``derived/`` (symlink targets
  under resources/ are refused).

Read-only over resources/ and third_party/. Raw upstream outputs are kept
next to the converted copies. All frozen pilot demos keep a reconciled
result; failures never shrink the denominator.

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

from ireset.data import a2_integrity as integ
from ireset.data import omniretarget_sources as orc
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record

HOLOSOMA_PACKAGE = (
    PROJECT_ROOT / "third_party" / "holosoma" / "src" / "holosoma_retargeting"
    / "holosoma_retargeting"
)
DEFAULT_ENV_PYTHON = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"


def pilot_demos_from_freeze(freeze_dir: Path) -> list[str]:
    lists = json.loads((freeze_dir / "frozen_lists.json").read_text())
    return [e["demo"] for e in lists["teacher_pilot_train6_v1"]]


def generate_one(
    demo: str,
    env_python: Path,
    attempt_dir: Path,
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
        "--save-dir", str(attempt_dir),
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
        raw_path = attempt_dir / f"{demo}_original.npz"
        pub_path = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
        scene_u0, human_u0, fps_u0 = orc.load_teacher_scene(raw_path, "upstream_generated")
        scene_p, human_p, fps_p = orc.load_teacher_scene(pub_path, "published")
        binding = orc.check_binding(demo, scene_u0, human_u0, fps_u0, scene_p, human_p, fps_p)
        conv_path = attempt_dir / f"{demo}_original_converted.npz"
        np.savez(
            conv_path,
            qpos=scene_u0,
            human_joints=human_u0,
            fps=fps_u0,
            source_type="upstream_generated",
            layout="project_scene_v1",
            provenance=json.dumps({
                "demo": demo, "raw_file": raw_path.name,
                "raw_sha256": integ.sha256_file(raw_path),
                "published_sha256": integ.sha256_file(pub_path),
                "binding": binding.to_json_dict(),
            }),
        )
        row["status"] = "ok" if binding.passed else "binding_failed"
        row["binding"] = binding.to_json_dict()
        row["raw_sha256"] = integ.sha256_file(raw_path)
        row["converted_sha256"] = integ.sha256_file(conv_path)
        row["converted"] = path_for_record(conv_path)
        row["T"] = int(scene_u0.shape[0])
    except Exception as exc:
        row["status"] = "postcheck_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True,
                    help="A1 freeze directory (frozen_lists.json + assets.json)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="U0 root directory (each run creates attempt_NNN inside)")
    ap.add_argument("--env-python", type=Path, default=DEFAULT_ENV_PYTHON)
    ap.add_argument("--data-path", type=Path, default=OMOMO_NEW_DIR)
    ap.add_argument("--demos", type=str, default=None,
                    help="comma-separated retry scope (subset of the frozen pilot "
                         "list); writes a NEW attempt, never overwrites")
    args = ap.parse_args()

    freeze = args.freeze.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    out_root = integ.guard_output_dir(args.out_dir)

    pilot = pilot_demos_from_freeze(freeze)
    scope = pilot
    if args.demos:
        wanted = [d.strip() for d in args.demos.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in pilot]
        if unknown:
            raise SystemExit(f"--demos outside frozen pilot list: {unknown}")
        scope = wanted

    # Identity gate BEFORE any generation: measured checkout must equal the
    # A1-frozen commit and be clean; env identity is recorded, not assumed.
    freeze_assets = json.loads((freeze / "assets.json").read_text())
    checkout = integ.verify_checkout_against_freeze(freeze_assets)
    if not checkout["passed"]:
        raise SystemExit(
            f"upstream checkout identity check failed vs A1 freeze: {checkout}")
    env = integ.measure_env_identity(args.env_python)
    if env["env_mujoco_version"] is None:
        raise SystemExit(f"env probe failed: {env['env_probe_error']}")
    if env["env_lock_mujoco_version"] != env["env_mujoco_version"]:
        raise SystemExit(
            "env binary drifted from its own lock: probed mujoco "
            f"{env['env_mujoco_version']} vs lock-pinned "
            f"{env['env_lock_mujoco_version']}")

    # Round-3 review fix 3: the inputs and the frozen assets are verified
    # against the A1 FREEZE (not merely re-recorded) — a mutated input or a
    # drifted asset must stop the run before any teacher is generated.
    # Round-4 review fix 2: the PT side is verified at the ACTUAL --data-path
    # this run consumes, not at the canonical directory — passing an
    # alternate source dir must not pass a check anchored elsewhere.
    frozen_entries = json.loads(
        (freeze / "frozen_lists.json").read_text())["teacher_pilot_train6_v1"]
    freeze_inputs = integ.verify_inputs_against_freeze(
        [e for e in frozen_entries if e["demo"] in scope],
        TEACHER_TRAJ_DIR, data_path)
    bad_inputs = [d for d, r in freeze_inputs.items()
                  if not (r["npz_matches"] and r["pt_matches"])]
    if bad_inputs:
        raise SystemExit(
            f"inputs no longer match the A1-frozen hashes: {bad_inputs}")
    freeze_assets_check = integ.verify_assets_against_freeze(freeze_assets)
    if not freeze_assets_check["passed"]:
        raise SystemExit(
            f"A1-frozen assets drifted: {freeze_assets_check}")

    attempt_dir = integ.allocate_attempt(out_root)
    attempt = attempt_dir.name

    # Input immutability anchors: recorded in the manifest (the comparison
    # re-verifies them later) AND checked before/after this run. The PT side
    # anchors the ACTUAL consumed data_path (round-4 review fix 2).
    inputs = {}
    for demo in scope:
        inputs[demo] = {
            "npz": integ.sha256_file(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
            "pt": integ.sha256_file(data_path / f"{demo}.pt"),
        }

    rows = [generate_one(d, args.env_python, attempt_dir, data_path) for d in scope]

    input_mutated = False
    for demo in scope:
        now = {
            "npz": integ.sha256_file(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
            "pt": integ.sha256_file(data_path / f"{demo}.pt"),
        }
        if now != inputs[demo]:
            input_mutated = True
            for row in rows:
                if row["demo"] == demo and row["status"] == "ok":
                    row["status"] = "input_mutated"

    n_ok = sum(r["status"] == "ok" for r in rows)
    report = {
        "stage": "I0 U0 generation (plan §3.3/§10.1)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "freeze": path_for_record(freeze),
        "data_path": path_for_record(data_path),   # the dir ACTUALLY consumed
        "attempt": attempt,
        "attempt_scope": scope,
        "expected_denominator": len(pilot),
        "identity": {
            "checkout": checkout,
            "env": env,
            "freeze_expected_commit": freeze_assets["upstream"]["expected_commit"],
        },
        "freeze_verification": {
            "inputs_vs_freeze": freeze_inputs,
            "assets_vs_freeze": freeze_assets_check,
        },
        "input_sha256": inputs,
        "n_ok": n_ok,
        "n_binding_failed": sum(r["status"] == "binding_failed" for r in rows),
        "n_generation_failed": sum(r["status"] == "generation_failed" for r in rows),
        "n_postcheck_failed": sum(r["status"] == "postcheck_failed" for r in rows),
        "input_mutated": input_mutated,
        "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
        "rows": rows,
    }
    out = attempt_dir / "generation_manifest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    integ.append_attempts_index(out_root, {
        "attempt": attempt,
        "scope": scope,
        "expected_denominator": len(pilot),
        "n_ok": n_ok,
        "complete": (n_ok == len(pilot)) and not input_mutated,
        "manifest": integ.manifest_path_record(out_root, attempt),
    })
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    print(f"manifest -> {path_for_record(out)}")
    ok = n_ok == len(pilot) and not input_mutated
    print(f"generation ({attempt}, scope {len(scope)}/{len(pilot)}): "
          f"{'COMPLETE' if ok else 'INCOMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
