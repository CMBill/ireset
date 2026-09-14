"""P0.0.2 row-level repair failure diagnosis (p0 plan section 3).

For each registered teacher segment: run the sequential repairer with a
capture wrapper around ``FrameSolver.solve_frame``. On rejection, classify
the violated accept-mode constraint rows at the failing frame's *prediction*
(fresh pair scan) into the taxonomy families of
``docs/p0_relaxed_validation_plan.md`` section 1.3, plus drift / task
diagnostics, so every intervention step (L0/L1/L2) has a measured go/no-go.

``--geometry-census`` measures teacher contact geometry directly (no repair):
sole-ground gap inside reference support intervals; hand-geom/box distance and
per-interval hand anchor drift inside reference grasp intervals. These
distributions set the L2 paired relaxed values.

Run: uv run --locked python scripts/diagnose_repair.py [--validation-mode strict]
        [--geometry-census] [--output PATH] [--demos N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from audit_data import convert_scene_qpos
from ireset.embodied.contact_sites import infer_reference_schedule
from ireset.paths import DERIVED_DATA_DIR, TEACHER_TRAJ_DIR
from ireset.repair.config_io import MODES, load_mode_configs
from ireset.repair.context import ReferenceCondition
from ireset.repair.pipeline import RepairPipeline

OUT = DERIVED_DATA_DIR / "validation"
SCREEN = DERIVED_DATA_DIR / "screening" / "velocity_screen.json"


def load(demo: str, frames: int):
    npz = np.load(TEACHER_TRAJ_DIR / f"{demo}_original.npz", allow_pickle=True)
    scene = np.stack([convert_scene_qpos(r) for r in npz["qpos"]])[:frames]
    human = npz["human_joints"].astype(np.float64)[:frames]
    cond = ReferenceCondition(human, scene[:, 36:43], infer_reference_schedule(human, scene[:, 36:43]))
    return scene, cond


def registered_demos(n_diag: int) -> list[str]:
    if SCREEN.exists():
        return json.loads(SCREEN.read_text())["registered"]["p0_diagnosis_6"][:n_diag]
    from collections import defaultdict

    by_subj: dict[str, list[str]] = defaultdict(list)
    for p in sorted(TEACHER_TRAJ_DIR.glob("*_original.npz")):
        by_subj[p.name.split("_")[0]].append(p.name[: -len("_original.npz")])
    return [d for _, ds in sorted(by_subj.items()) for d in ds[:n_diag]]


def classify_rows(pipe: RepairPipeline, z: np.ndarray, ctx) -> dict:
    """Violated accept-mode rows at z, classified by taxonomy family."""
    cfg = pipe.cfg
    pipe.solver._frame_pairs = None   # fresh pair scan at this candidate
    val, lo, hi, scl = pipe.solver.constraint_rows(z, ctx, trust=cfg.trust_radius_max, mode="accept")
    rows = pipe.solver._last_rows
    viol = np.maximum(lo - val, val - hi) / scl
    bad = np.where(viol > 1e-6)[0]
    maxp = cfg.max_active_pairs
    kinds, worst = Counter(), {}
    for i in bad:
        if i < 29:
            k = "joint_limit"
        elif i < 58:
            k = "joint_vel"
        elif i < 61:
            k = "base_lin_vel"
        elif i < 64:
            k = "base_ang_vel"
        elif i < 99:
            k = "trust"
        elif i - 99 < maxp:
            r = i - 99
            if r < len(rows) and np.isfinite(rows[r][3]):
                band_hi = rows[r][3]
                k = ("foot_contact_band" if abs(band_hi - cfg.foot_contact_gap_m)
                     < abs(band_hi - cfg.contact_band_gap_m) else "hand_contact_band")
            else:
                k = "noncontact_clearance"
        elif i - 99 - maxp < 18:
            k = "hand_anchor"
        elif i - 99 - maxp < 48:
            k = "foot_anchor"
        else:
            k = "foot_speed"
        kinds[k] += 1
        worst[k] = max(worst.get(k, 0.0), float(viol[i]))
    return {"kinds": dict(kinds), "worst_viol": {k: round(v, 1) for k, v in worst.items()},
            "n_violated": int(len(bad))}


def diagnose(pipe: RepairPipeline, scene: np.ndarray, cond: ReferenceCondition) -> dict:
    caps = []
    orig = pipe.solver.solve_frame

    def cap(z0, ctx, **kw):
        r = orig(z0, ctx, **kw)
        caps.append((ctx.t, kw.get("force_slack", False), ctx, z0.copy(), r))
        return r

    pipe.solver.solve_frame = cap
    t0 = time.perf_counter()
    res = pipe.repair_trajectory(scene, cond)
    wall = time.perf_counter() - t0
    pipe.solver.solve_frame = orig

    d = res.diagnostics
    fs = d.get("frame_status", [])
    row = {
        "accepted": bool(res.accepted), "status": res.status,
        "t_fail": d.get("t_fail"), "skip_reason": d.get("skip_reason"),
        "accepted_frames": d.get("accepted_frames"),
        "n_frames": len(fs) or (scene.shape[0] if res.accepted else None),
        "skipped": sum(1 for s in fs if s == "skipped"),
        "slack": sum(1 for s in fs if "slack" in s),
        "multistart": sum(1 for s in fs if s == "repaired_prev"),
        "wall_s": round(wall, 1),
        "final": d.get("final_validation", d.get("first_attempt_final", ""))[:90],
    }
    if not res.accepted and d.get("t_fail") is not None:
        idx = [i for i, (t, sl, *_) in enumerate(caps) if t == d["t_fail"] and not sl]
        if idx:
            _, _, ctx, z0, r = caps[idx[0]]
            cls = classify_rows(pipe, z0, ctx)
            row["failing_frame"] = {
                "solver_status": r.status, "taxonomy": cls,
                "pred_mpjpe": round(float(pipe.solver.evaluate(z0, ctx)["mpjpe"]), 3),
                "pred_root_gap_m": (round(float(np.linalg.norm(z0[0:3] - ctx.z_prev[0:3])), 3)
                                    if ctx.z_prev is not None else None),
                "hand_anchor": "".join("L" if ctx.hand_anchor.get(s) is not None else "-" for s in ("l", "r")),
                "foot_anchor": "".join("L" if ctx.foot_anchor.get(s) is not None else "-" for s in ("l", "r")),
            }
    return row


def geometry_census(pipe: RepairPipeline, scene: np.ndarray, cond: ReferenceCondition) -> dict:
    """Teacher contact geometry inside reference intervals (no repair)."""
    ev, cfg = pipe.ev, pipe.cfg
    T = scene.shape[0]
    gap, hdist, hlocal = {}, {}, {}
    for t in range(T):
        ev.forward(scene[t, 0:36], scene[t, 36:43])
        gap[t] = {s: min(ev.geom_distance(g, ev.ground_geom)[0] for g in ev.sole_geoms[s]) for s in ("l", "r")}
        hdist[t] = {s: min(ev.geom_distance(g, ev.box_geom)[0] for g in ev._hand_geoms[s]) for s in ("l", "r")}
        hlocal[t] = {s: ev.hand_local(s).copy() for s in ("l", "r")}
    support_gap, grasp_gap, grasp_drift = [], [], []
    for iv in cond.ref_schedule.intervals:
        side = iv.effector[0] == "left" and "l" or "r"
        if iv.support == "ground":
            support_gap += [gap[t][side] for t in range(iv.start, iv.end)]
        else:
            grasp_gap += [hdist[t][side] for t in range(iv.start, iv.end)]
            base = hlocal[iv.start][side]
            grasp_drift += [float(np.linalg.norm(hlocal[t][side] - base, axis=-1).max())
                            for t in range(iv.start, iv.end)]
    out = {}
    if support_gap:
        a = np.abs(support_gap)
        out["support_abs_gap_m"] = {p: round(float(np.percentile(a, p)), 4) for p in (50, 90, 95)}
        out["support_abs_gap_m"]["max"] = round(float(a.max()), 4)
        out["support_frames_over_5mm"] = int((a > 0.005).sum())
        out["n_support_frames"] = int(len(a))
    if grasp_gap:
        a = np.abs(grasp_gap)
        out["grasp_abs_gap_m"] = {p: round(float(np.percentile(a, p)), 4) for p in (50, 90, 95)}
        out["grasp_abs_gap_m"]["max"] = round(float(a.max()), 4)
        out["grasp_frames_over_1cm"] = int((a > 0.01).sum())
        out["n_grasp_frames"] = int(len(a))
    if grasp_drift:
        out["grasp_anchor_drift_m"] = {p: round(float(np.percentile(grasp_drift, p)), 4) for p in (50, 90, 95)}
        out["grasp_anchor_drift_m"]["max"] = round(float(max(grasp_drift)), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=6)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--validation-mode", choices=MODES, default="strict")
    ap.add_argument("--geometry-census", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    demos = registered_demos(args.demos)
    rcfg, _vcfg = load_mode_configs(args.validation_mode)
    pipe = RepairPipeline(cfg=rcfg)
    pipe.validator.cfg = _vcfg
    rows, census, agg = [], {}, Counter()
    for demo in demos:
        scene, cond = load(demo, args.frames)
        rows.append({"demo": demo, **diagnose(pipe, scene, cond)})
        r = rows[-1]
        tax = r.get("failing_frame", {}).get("taxonomy", {}).get("kinds", {})
        agg.update(tax)
        if args.geometry_census:
            census[demo] = geometry_census(pipe, scene, cond)
        print(f"{demo}: accepted={r['accepted']} t_fail={r['t_fail']} "
              f"skip={r['skipped']} slack={r['slack']} kinds={tax or '-'}")

    report = {
        "date": str(date.today()), "mode": args.validation_mode,
        "demos": demos, "rows": rows,
        "aggregate_taxonomy_kinds": dict(agg),
        "acceptance": round(float(np.mean([r["accepted"] for r in rows])), 3),
    }
    if args.geometry_census:
        report["geometry_census"] = census
    out = Path(args.output) if args.output else OUT / f"diagnose_{args.validation_mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\naggregate taxonomy: {dict(agg)}")
    print(f"acceptance: {report['acceptance']}  ->  {out}")


if __name__ == "__main__":
    main()
