"""A2 teacher source comparison: published P vs locally generated U0.

Plan omniretarget_first_training_plan §3.3-§3.4/§8 (A2): replay both
sources for the frozen pilot six on the same locked validator geometry and
reference intervals (inferred from the A1 source-scene pairing), apply the
frozen ``local_preferred`` criteria, and write ``teacher_selection.json``.

Admission is re-verified from the artifacts, never taken from manifest
status strings (A2 review fix 5, 2026-10-09): the upstream checkout is
measured again and compared against the A1 freeze; the env lock hash must
equal the one recorded at generation; every consumed input file is hashed
against the generation manifest; and the U0-vs-P binding is recomputed
bit-exactly from the converted and published NPZs. Any mismatch or any
frozen pilot demo without an OK artifact forces ``incomplete``.

P rows come from the published NPZ; U0 rows from the newest OK attempt
artifact per demo. Every expected demo keeps a row on both sides;
missing/invalid evidence forces ``incomplete`` (never a silent drop).
Fixed-input ``box_ground`` conflicts are recorded separately (§7.3). U1 is
not pre-registered in this round -> ``not_applicable`` (§3.3).

Run: uv run --locked python scripts/omniretarget_teacher_comparison.py \
       --freeze <a1_run>/freeze --u0-dir <a2_run>/teacher_pilot/u0 \
       --out <a2_run>/teacher_pilot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data import a2_integrity as integ
from ireset.data import omniretarget_sources as orc
from ireset.eval.teacher_comparison import (
    DemoMetrics, TeacherQualityReplayer, evaluate_local_preferred,
)
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record


def load_source_boxes(freeze_dir: Path) -> dict[str, np.ndarray]:
    npz = np.load(freeze_dir / "pairing_source_boxes.npz")
    return {k: np.asarray(v) for k, v in npz.items()}


def measure_source(
    replayer: TeacherQualityReplayer,
    demos: list[str],
    scene_loader,
    source_boxes: dict[str, np.ndarray],
) -> dict[str, DemoMetrics]:
    out: dict[str, DemoMetrics] = {}
    for demo in demos:
        try:
            scene, human, fps = scene_loader(demo)
            if demo not in source_boxes:
                raise ValueError("no A1 source-scene box for this demo")
            out[demo] = replayer.measure(demo, scene, human, source_boxes[demo], fps)
        except Exception as exc:
            out[demo] = DemoMetrics(demo, -1, float("nan"), valid=False,
                                    error=f"{type(exc).__name__}: {exc}")
        r = out[demo]
        pen = f"{r.penetration_max_m:.4f}" if np.isfinite(r.penetration_max_m) else "n/a"
        task = f"{r.task_mpjpe_p95_m:.4f}" if np.isfinite(r.task_mpjpe_p95_m) else "n/a"
        print(f"    {demo}: valid={r.valid} pen_max={pen} task_p95={task} "
              f"both={r.both.get('valid_coverage')}/{r.both.get('gap_series_p95_m')}")
    return out


def admission_record(freeze: Path, u0_root: Path, pilot: list[str],
                     env_python: Path | None = None) -> dict:
    """Artifact-level admission: identity, inputs, bindings (review fix 5)."""
    freeze_assets = json.loads((freeze / "assets.json").read_text())
    checkout = integ.verify_checkout_against_freeze(freeze_assets)

    coverage = integ.collect_u0_coverage(u0_root, pilot)
    inputs_ok = True
    inputs_detail: dict[str, dict] = {}
    env_lock_shas: set[str] = set()
    env_pythons: set[str] = set()
    binding: dict[str, dict] = {}
    binding_passed: dict[str, bool] = {}
    for demo, entry in sorted(coverage["by_demo"].items()):
        manifest = integ.load_attempt_manifest(u0_root, entry["attempt"])
        env_lock_shas.add(manifest.get("identity", {}).get("env", {}).get("env_uv_lock_sha256"))
        env_pythons.add(manifest.get("identity", {}).get("env", {}).get("env_python"))
        # inputs re-hashed NOW against what generation recorded
        demo_inputs = integ.verify_inputs_against_manifest(
            manifest, TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
        row = demo_inputs.get(demo, {})
        inputs_detail[demo] = {
            "attempt": entry["attempt"],
            "npz_matches": row.get("npz_matches"),
            "pt_matches": row.get("pt_matches"),
        }
        if not (row.get("npz_matches") and row.get("pt_matches")):
            inputs_ok = False
        # binding recomputed from the artifacts on disk
        check = integ.verify_binding_from_artifacts(
            Path(entry["converted"]),
            TEACHER_TRAJ_DIR / f"{demo}_original.npz",
        )
        binding[demo] = {
            "recomputed": check.to_json_dict(),
            "manifest_status": entry["manifest_status"],
            "manifest_binding_passed": entry["manifest_binding_passed"],
            "manifest_agrees": (entry["manifest_status"] == "ok"
                                and entry["manifest_binding_passed"]),
        }
        binding_passed[demo] = check.passed
    for demo in coverage["missing"]:
        binding_passed[demo] = False
        binding[demo] = {"recomputed": None, "note": "no OK artifact in any attempt"}

    # env probed NOW at the python the generation actually used (falling
    # back to the canonical independent env), compared against what the
    # generation manifests recorded
    default_env = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"
    env_python = Path(env_python) if env_python else Path(next(
        (p for p in env_pythons if p), str(default_env)))
    env_now = integ.measure_env_identity(env_python)
    env_lock_consistent = (
        len(env_lock_shas) <= 1
        and None not in env_lock_shas
        and env_now["env_uv_lock_sha256"] in env_lock_shas
    )
    return {
        "checkout_vs_freeze": checkout,
        "env_lock": {
            "measured_now": env_now["env_uv_lock_sha256"],
            "recorded_at_generation": sorted(x for x in env_lock_shas if x),
            "consistent": env_lock_consistent,
        },
        "inputs_rehashed": {"all_match": inputs_ok, "by_demo": inputs_detail},
        "binding_recomputed": binding,
        "u0_coverage": {"missing": coverage["missing"],
                        "by_demo": {d: e["attempt"] for d, e in coverage["by_demo"].items()}},
        "passed": (checkout["passed"] and env_lock_consistent and inputs_ok
                   and all(binding_passed.values())),
    }, binding_passed, coverage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True, help="A1 freeze dir")
    ap.add_argument("--u0-dir", type=Path, required=True,
                    help="I0 U0 root dir (attempts_index.json)")
    ap.add_argument("--out", type=Path, required=True, help="teacher_pilot output dir")
    args = ap.parse_args()
    t0 = time.time()

    freeze = args.freeze.expanduser().resolve()
    u0_root = args.u0_dir.expanduser().resolve()
    out_dir = integ.guard_output_dir(args.out)

    lists = json.loads((freeze / "frozen_lists.json").read_text())
    pilot = [e["demo"] for e in lists["teacher_pilot_train6_v1"]]
    source_boxes = load_source_boxes(freeze)

    admission, binding_passed, coverage = admission_record(freeze, u0_root, pilot)

    def load_p(demo):
        return orc.load_teacher_scene(TEACHER_TRAJ_DIR / f"{demo}_original.npz", "published")

    u0_by_demo = coverage["by_demo"]

    def load_u0(demo):
        return orc.load_teacher_scene(
            Path(u0_by_demo[demo]["converted"]), "project_scene")

    print(f"[a2] replaying P (published) on {len(pilot)} demos ...")
    replayer = TeacherQualityReplayer()
    p_metrics = measure_source(replayer, pilot, load_p, source_boxes)
    print("[a2] replaying U0 (generated) ...")
    u_metrics = {}
    for demo in pilot:
        if demo in u0_by_demo:
            u_metrics.update(
                measure_source(replayer, [demo], load_u0, source_boxes))
        else:
            u_metrics[demo] = DemoMetrics(
                demo, -1, float("nan"), valid=False,
                error="no OK U0 artifact in any attempt")

    verdict = evaluate_local_preferred(pilot, p_metrics, u_metrics, binding_passed)
    # Admission failures are verdict-level: a wrong checkout, a drifted env
    # lock, mutated inputs or a failed binding recheck must NEVER surface as
    # local_preferred with complete evidence (review fix 5).
    if not admission["passed"]:
        reasons = []
        if not admission["checkout_vs_freeze"]["passed"]:
            reasons.append("checkout_identity_mismatch")
        if not admission["env_lock"]["consistent"]:
            reasons.append("env_lock_drift_vs_generation")
        if not admission["inputs_rehashed"]["all_match"]:
            reasons.append("input_hash_mismatch_vs_generation")
        reasons += [f"{d}:binding_recheck_failed"
                    for d, ok in binding_passed.items() if not ok]
        verdict["evidence_gaps"] = list(verdict["evidence_gaps"]) + [
            f"admission:{r}" for r in reasons]
        verdict["complete_evidence"] = False
        verdict["verdict"] = "incomplete"
    selection = {
        "stage": "A2 teacher source comparison (plan §3.3-§3.4)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pilot": pilot,
        "p_revision": "135259130158f1cb1799e4ab9b6566c938285fc9",
        "admission": admission,
        "u0_identity": {
            "holosoma_checkout": admission["checkout_vs_freeze"]["holosoma_revision"],
            "env_uv_lock_sha256": admission["env_lock"]["measured_now"],
            "env_mujoco_version": None,   # filled from the generation manifest below
            "generation_attempts": [a["attempt"] for a in integ.load_attempts(u0_root)],
        },
        "u1": "not_applicable (not pre-registered this round, §3.3)",
        "metrics": {
            "P": {d: p_metrics[d].to_json() for d in pilot},
            "U0": {d: u_metrics[d].to_json() for d in pilot},
        },
        "binding": binding_passed,
        "verdict_record": verdict,
        "verdict": verdict["verdict"],
        "residual_risks": (
            "residual defects below reference levels are allowed; local_preferred "
            "authorizes approximate motion labels for the small batch only (§3.4); "
            "an improvement in one metric family is never summarized as reliable grasp"
        ),
        "cost_s": round(time.time() - t0, 1),
    }
    # fill measured env facts from the newest attempt manifest
    attempts = integ.load_attempts(u0_root)
    if attempts:
        manifest = integ.load_attempt_manifest(u0_root, attempts[-1]["attempt"])
        selection["u0_identity"]["env_mujoco_version"] = (
            manifest.get("identity", {}).get("env", {}).get("env_mujoco_version"))

    out = out_dir / "teacher_selection.json"
    out.write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    print(json.dumps({k: verdict[k] for k in ("complete_evidence", "verdict")}, indent=2))
    for name, g in verdict["gates"].items():
        print(f"  {name}: {'PASS' if g['passed'] else 'FAIL'}")
    if verdict["evidence_gaps"]:
        print(f"  evidence gaps: {verdict['evidence_gaps'][:6]}"
              f"{' ...' if len(verdict['evidence_gaps']) > 6 else ''}")
    print(f"teacher_selection -> {path_for_record(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
