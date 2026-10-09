"""A2 teacher source comparison: published P vs locally generated U0.

Plan omniretarget_first_training_plan §3.3-§3.4/§8 (A2): replay both
sources for the frozen pilot six on the same locked validator geometry and
reference intervals (inferred from the A1 source-scene pairing), apply the
frozen ``local_preferred`` criteria, and write ``teacher_selection.json``.

P rows come from the published NPZ; U0 rows from the I0 generation
manifest. Every expected demo keeps a row on both sides; missing/invalid
evidence forces ``incomplete`` (never a silent drop). Fixed-input
``box_ground`` conflicts are recorded separately (§7.3). U1 is
not pre-registered in this round -> ``not_applicable`` (§3.3).

Run: uv run --locked python scripts/omniretarget_teacher_comparison.py \
       --freeze <a1_run>/freeze --u0-dir <a2_run>/teacher_pilot/u0 \
       --out <a2_run>/teacher_pilot
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
from ireset.eval.teacher_comparison import (
    DemoMetrics, TeacherQualityReplayer, evaluate_local_preferred,
)
from ireset.paths import PROJECT_ROOT, TEACHER_TRAJ_DIR


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
        print(f"    {demo}: valid={r.valid} pen_max={r.penetration_max_m:.4f} "
              f"task_p95={r.task_mpjpe_p95_m:.4f} "
              f"both={r.both.get('valid_coverage')}/{r.both.get('gap_series_p95_m')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True, help="A1 freeze dir")
    ap.add_argument("--u0-dir", type=Path, required=True,
                    help="I0 generation output dir (generation_manifest.json)")
    ap.add_argument("--out", type=Path, required=True, help="teacher_pilot output dir")
    args = ap.parse_args()
    t0 = time.time()

    lists = json.loads((args.freeze / "frozen_lists.json").read_text())
    pilot = [e["demo"] for e in lists["teacher_pilot_train6_v1"]]
    gen = json.loads((args.u0_dir / "generation_manifest.json").read_text())
    source_boxes = load_source_boxes(args.freeze)
    binding = {r["demo"]: (r["status"] == "ok") for r in gen["rows"]}

    def load_p(demo):
        return orc.load_teacher_scene(TEACHER_TRAJ_DIR / f"{demo}_original.npz", "published")

    def load_u0(demo):
        return orc.load_teacher_scene(
            args.u0_dir / f"{demo}_original_converted.npz", "project_scene"
        )

    print(f"[a2] replaying P (published) on {len(pilot)} demos ...")
    replayer = TeacherQualityReplayer()
    p_metrics = measure_source(replayer, pilot, load_p, source_boxes)
    print("[a2] replaying U0 (generated) ...")
    u_metrics = measure_source(replayer, pilot, load_u0, source_boxes)

    verdict = evaluate_local_preferred(pilot, p_metrics, u_metrics, binding)
    selection = {
        "stage": "A2 teacher source comparison (plan §3.3-§3.4)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pilot": pilot,
        "p_revision": "135259130158f1cb1799e4ab9b6566c938285fc9",
        "u0_identity": {
            "holosoma_checkout": gen.get("holosoma_checkout"),
            "env_uv_lock_sha256": gen.get("env_uv_lock_sha256"),
            "generation_manifest": str((args.u0_dir / "generation_manifest.json").relative_to(PROJECT_ROOT)),
            "config": gen["rows"][0]["config"] if gen["rows"] else None,
        },
        "u1": "not_applicable (not pre-registered this round, §3.3)",
        "metrics": {
            "P": {d: p_metrics[d].to_json() for d in pilot},
            "U0": {d: u_metrics[d].to_json() for d in pilot},
        },
        "binding": binding,
        "verdict_record": verdict,
        "verdict": verdict["verdict"],
        "residual_risks": (
            "residual defects below reference levels are allowed; local_preferred "
            "authorizes approximate motion labels for the small batch only (§3.4)"
        ),
        "cost_s": round(time.time() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "teacher_selection.json"
    out.write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    print(json.dumps({k: verdict[k] for k in ("complete_evidence", "verdict")}, indent=2))
    for name, g in verdict["gates"].items():
        print(f"  {name}: {'PASS' if g['passed'] else 'FAIL'}")
    print(f"teacher_selection -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
