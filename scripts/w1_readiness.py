"""W1 readiness decision: machine-readable A3/B/C completion record.

W1 plan §10. Reads the run root's ACTUAL artifacts and recomputes every
gate from facts (never from prose or exit codes):

* ``registration_complete`` — freeze/preregistration.json exists and its
  anchors/lists/inputs/assets legs re-verify against disk (the C2/UR entry
  evidence is re-checked through the same legs the W0 entry used);
* ``a3_complete`` — the newest audit attempt's exit criteria recomputed
  from its rows (55 final rows, train quality registration complete, six
  subjects with usable windows);
* ``b_complete`` — the cache identity closes against the preregistration
  and training manifest on disk, AND both W1 datasets actually construct
  (the startup verification legs run for real) with the train window count
  >= micro_batch and one real batch fetch;
* ``c_complete`` — both models' newest S0 attempts are ``passed`` on every
  gate (>=100 real optimizer steps, all-finite, contact branch silent,
  >=40 fixed probes, fit trend, reload reproducibility);
* ``ready_for_w1_d`` — all of the above (each reason recorded; a missing
  value and an evidenced zero stay distinct).

Exit 0 means the readiness RECORD was written from complete inputs; it is
never a claim that training may start — when ``ready_for_w1_d`` is false
the reasons are in the record and the exit code is 2.

Run: uv run --locked python scripts/w1_readiness.py --run-root <run_root>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ
from ireset.data import w1_training as w1
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record


def latest_attempt(root: Path) -> dict | None:
    index = root / integ.ATTEMPTS_INDEX
    if not index.is_file():
        return None
    attempts = json.loads(index.read_text())["attempts"]
    return attempts[-1] if attempts else None


def check_registration(run_root: Path) -> dict:
    from w1_preregister import verify_c2_decision, verify_ur_report

    reasons: list[str] = []
    prereg_path = run_root / "freeze" / "preregistration.json"
    if not prereg_path.is_file():
        return {"passed": False, "reasons": ["preregistration_missing"]}
    prereg = json.loads(prereg_path.read_text())
    check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                      TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    reasons += [f"preregistration:{p}" for p in check["problems"]]
    paths = w1.anchor_paths(PROJECT_ROOT)
    c2 = verify_c2_decision(paths["c2_final_decision"])
    ur = verify_ur_report(paths["ur_full_comparison"],
                          paths["ur_preregistration"])
    reasons += [f"c2:{p}" for p in c2["problems"]]
    reasons += [f"ur:{p}" for p in ur["problems"]]
    return {"passed": not reasons, "reasons": reasons,
            "preregistration": path_for_record(prereg_path)}


def check_a3(run_root: Path) -> dict:
    from audit_omniretarget_training import (
        BOOL_EXIT_GATES, exit_criteria, exit_ok, summarize,
    )

    att = latest_attempt(run_root / "audit")
    if att is None:
        return {"passed": False, "reasons": ["a3_no_attempts"]}
    manifest = json.loads((PROJECT_ROOT / att["manifest"]).read_text())
    scope = list(manifest["selection"]["train24"]) + list(manifest["selection"]["val31"])
    summary = summarize(manifest["rows"], scope)
    gates = exit_criteria(summary, manifest["rows"])
    # only the boolean gates count as unmet reasons (``missing_rows`` is the
    # evidence list, not itself a verdict)
    reasons = [k for k in BOOL_EXIT_GATES if gates.get(k) is not True]
    return {"passed": exit_ok(gates) and not reasons, "reasons": reasons,
            "exit_criteria": gates, "counts": {
                k: v for k, v in summary.items()
                if not isinstance(v, (dict, list))},
            "manifest": att["manifest"]}


def check_cache_binding(run_root: Path) -> dict | None:
    att = latest_attempt(run_root / "cache")
    if att is None:
        return None
    ident_path = PROJECT_ROOT / att["identity"]
    return {"attempt": att["attempt"], "identity": ident_path,
            "dir": ident_path.parent,
            "record": json.loads(ident_path.read_text())}


def check_b(run_root: Path, cache: dict | None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ireset.data.windows import TeacherWindowDataset, make_tokenizer
    from ireset.model.tokens import RootStats

    reasons: list[str] = []
    if cache is None:
        return {"passed": False, "reasons": ["cache_no_attempts"]}
    rec = cache["record"]
    cache_dir = cache["dir"]
    manifest_path = PROJECT_ROOT / rec["training_manifest"]["path"]
    if integ.sha256_file(manifest_path) != rec["training_manifest"]["sha256"]:
        reasons.append("cache_manifest_binding_drifted")
    prereg_path = run_root / "freeze" / "preregistration.json"
    if integ.sha256_file(prereg_path) != rec["preregistration"]["sha256"]:
        reasons.append("cache_prereg_binding_drifted")
    # review fix 6: CONTENT re-verification — root stats and every keyed
    # schedule re-hashed against the identity record, keys re-derived from
    # the current code/assets, stats identity reconciled with the manifest
    manifest = json.loads(manifest_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    content = w1.verify_cache_contents(cache_dir, manifest, prereg, PROJECT_ROOT)
    reasons += [f"cache_content:{p}" for p in content["problems"]]
    root_stats = cache_dir / "root_stats.npz"
    if not root_stats.is_file():
        reasons.append("root_stats_missing")
    if reasons:
        return {"passed": False, "reasons": reasons}

    # REAL construction of both datasets: every startup verification leg
    # runs (protocol/prereg/selection/inputs/derived/keyed-cache/assets)
    stats = RootStats.load(root_stats)
    tokenizer = make_tokenizer(stats)
    out = {}
    try:
        for split in ("train", "val"):
            ds = TeacherWindowDataset(
                split, tokenizer, window=w1.WINDOW_F, stride=w1.WINDOW_STRIDE,
                training_manifest=manifest_path,
                schedule_cache_dir=cache_dir / "schedules")
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            out[split] = {"n_windows": len(ds), "n_clips": len(ds.clips),
                          "excluded": {k: len(v) for k, v in ds.excluded.items()},
                          "frame_count": ds.frame_count(),
                          "real_batch": tuple(batch["x0"].shape)}
        if out["train"]["n_windows"] < 4 or out["train"]["real_batch"][0] != 4:
            reasons.append("train_micro_batch_not_formable")
    except Exception as exc:  # noqa: BLE001 -- recorded as the reason
        reasons.append(f"dataset_construction_failed:{type(exc).__name__}: {exc}")
    return {"passed": not reasons, "reasons": reasons,
            "datasets": out, "cache_attempt": cache["attempt"]}


def _recompute_s0_verdict(summary: dict, history: list[dict]) -> tuple[dict, list[str]]:
    """Recompute EVERY S0 gate from the raw facts (review fix 2) — the
    recorded ``verdict`` string is never trusted by itself."""
    import math as _math

    import numpy as np

    probes = summary.get("probes") or []
    reasons: list[str] = []
    steps = summary.get("optimizer_steps")
    if not isinstance(steps, int):
        reasons.append("optimizer_steps_not_an_int")
        steps = -1
    gates = {}
    gates["min_100_optimizer_steps"] = steps >= w1.S0_MIN_STEPS
    gates["max_1000_optimizer_steps"] = steps <= w1.S0_MAX_STEPS
    gates["micro_batches_real"] = (summary.get("micro_batches", 0)
                                   >= steps * summary.get("grad_accum", 0) > 0)
    gates["all_finite"] = (summary.get("nonfinite_stop") is None
                           and all(
                               isinstance(h.get("gnorm"), (int, float))
                               and _math.isfinite(h["gnorm"]) for h in history))
    # contact branch silence recomputed from the per-step loss parts
    contact_keys = ("contact_m2", "contact")
    gates["contact_branch_silent"] = not any(
        anyk in h for h in history for anyk in contact_keys)
    errs = [p.get("motion_reconstruction") for p in probes]
    gates["probe_count_ge_40"] = len(errs) >= w1.S0_PROBE_MIN
    if len(errs) >= 2 * w1.S0_TREND_WINDOW and all(
            isinstance(e, (int, float)) and _math.isfinite(e) for e in errs):
        first = float(np.mean(errs[:w1.S0_TREND_WINDOW]))
        last = float(np.mean(errs[-w1.S0_TREND_WINDOW:]))
        gates["trend_fit_demonstrated"] = last < first
        gates["first20_mean"], gates["last20_mean"] = first, last
    else:
        gates["trend_fit_demonstrated"] = False
        gates["first20_mean"] = gates["last20_mean"] = None
    return gates, reasons


def check_c(run_root: Path, cache: dict | None, prereg: dict) -> dict:
    """Recompute both models' S0 gates from their recorded facts AND bind
    the checkpoints to the CURRENT manifest/cache identity (review fix 2:
    a doctored ``verdict=passed`` with zero steps, deleted probes, a
    missing checkpoint or a foreign-identity checkpoint must all fail)."""
    import torch

    reasons: list[str] = []
    per_model = {}
    manifest_sha = cache["record"]["training_manifest"]["sha256"] if cache else None
    cache_ident_sha = root_stats_sha = None
    if cache is not None:
        cache_ident_sha = integ.sha256_file(cache["identity"])
        root_stats_sha = integ.sha256_file(cache["dir"] / "root_stats.npz")
    for model in ("deterministic", "diffusion"):
        att = latest_attempt(run_root / "s0" / model)
        if att is None:
            reasons.append(f"s0_{model}_no_attempts")
            per_model[model] = None
            continue
        att_dir = (PROJECT_ROOT / att["summary"]).parent
        summary = json.loads((PROJECT_ROOT / att["summary"]).read_text())
        history_path = att_dir / "loss_history.json"
        history = (json.loads(history_path.read_text())
                   if history_path.is_file() else [])
        gates, gate_reasons = _recompute_s0_verdict(summary, history)
        reasons += [f"s0_{model}:{r}" for r in gate_reasons]
        # checkpoint evidence: exists, reloadable, bound to CURRENT identity
        ckpt_path = att_dir / "last.pt"
        ident_problems = []
        if not ckpt_path.is_file():
            ident_problems.append("checkpoint_missing")
        else:
            try:
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                ident_problems.append(f"checkpoint_unreadable:{type(exc).__name__}")
                ck = None
            if ck is not None:
                ci = ck.get("identity") or {}
                if ci.get("training_manifest", {}).get("sha256") != manifest_sha:
                    ident_problems.append("checkpoint_manifest_identity_mismatch")
                if ci.get("cache", {}).get("identity_sha256") != cache_ident_sha:
                    ident_problems.append("checkpoint_cache_identity_mismatch")
                if ci.get("cache", {}).get("root_stats_sha256") != root_stats_sha:
                    ident_problems.append("checkpoint_root_stats_mismatch")
                if ck.get("step") != summary.get("optimizer_steps"):
                    ident_problems.append("checkpoint_step_mismatch")
        reasons += [f"s0_{model}:{p}" for p in ident_problems]
        # small set must come from the CURRENT registration
        pilot6 = list(prereg.get("selection", {}).get("pilot6_from_a1", []))
        small = summary.get("identity", {}).get("small_set_demos") or []
        if not small or not set(small) <= set(pilot6):
            reasons.append(f"s0_{model}:small_set_not_from_registered_pilot6")
        reload_ok = summary.get("gates", {}).get("reload_reproducible")
        if reload_ok is not True:
            reasons.append(f"s0_{model}:reload_reproducible_not_true")
        recomputed_pass = (all(gates.values()) and not ident_problems
                           and reload_ok is True)
        if summary.get("verdict") != "passed":
            reasons.append(f"s0_{model}_{summary.get('verdict')}")
        elif not recomputed_pass:
            reasons.append(f"s0_{model}:recorded_verdict_not_reproducible")
        per_model[model] = {
            "verdict": summary.get("verdict"),
            "recomputed_gates": {k: bool(v) if isinstance(v, bool) else v
                                 for k, v in gates.items()},
            "identity_problems": ident_problems,
            "optimizer_steps": summary.get("optimizer_steps"),
            "small_set_demos": small,
        }
    return {"passed": not reasons, "reasons": reasons, "per_model": per_model}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    run_root = integ.guard_output_dir(args.run_root)

    registration = check_registration(run_root)
    a3 = check_a3(run_root)
    cache = check_cache_binding(run_root)
    b = check_b(run_root, cache)
    prereg_path = run_root / "freeze" / "preregistration.json"
    prereg = (json.loads(prereg_path.read_text())
              if prereg_path.is_file() else {})
    c = check_c(run_root, cache, prereg)

    ready = registration["passed"] and a3["passed"] and b["passed"] and c["passed"]
    record = {
        "stage": "W1 A3/B/C readiness (W1 plan §10)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "registration_complete": registration["passed"],
        "registration_reasons": registration["reasons"],
        "a3_complete": a3["passed"],
        "a3_reasons": a3["reasons"],
        "a3_exit_criteria": a3.get("exit_criteria"),
        "b_complete": b["passed"],
        "b_reasons": b["reasons"],
        "b_datasets": b.get("datasets"),
        "c_complete": c["passed"],
        "c_reasons": c["reasons"],
        "c_per_model": c["per_model"],
        "ready_for_w1_d": ready,
        "unmet": ([] if ready else
                  (registration["reasons"] + a3["reasons"]
                   + b["reasons"] + c["reasons"])),
        "note": "exit 0 = readiness record written; ready_for_w1_d is the "
                "training-entry gate, NOT a feasibility claim — U0 "
                "incomplete / UR no_quality_gain / final interaction goals "
                "all stand unchanged",
        "wall_s": round(time.time() - t0, 1),
    }
    decisions = run_root / "decisions"
    attempt_dir = integ.allocate_attempt(decisions)
    out = attempt_dir / "w1_readiness.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    integ.append_attempts_index(decisions, {
        "attempt": attempt_dir.name,
        "ready_for_w1_d": ready,
        "report": path_for_record(out),
    })
    _append_costs(run_root, "w1_readiness", time.time() - t0,
                  {"ready_for_w1_d": ready})
    print(json.dumps({k: record[k] for k in
                      ("registration_complete", "a3_complete", "b_complete",
                       "c_complete", "ready_for_w1_d", "unmet")}, indent=2))
    print(f"readiness -> {path_for_record(out)}")
    return 0 if ready else 2


def _append_costs(run_root: Path, stage: str, wall: float, extra: dict) -> None:
    costs_path = run_root / "costs.json"
    entries = []
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    entries.append({"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "stage": stage, "wall_s": round(wall, 1), **extra})
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
