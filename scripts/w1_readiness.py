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
    from audit_omniretarget_training import exit_criteria, summarize

    att = latest_attempt(run_root / "audit")
    if att is None:
        return {"passed": False, "reasons": ["a3_no_attempts"]}
    manifest = json.loads((PROJECT_ROOT / att["manifest"]).read_text())
    scope = list(manifest["selection"]["train24"]) + list(manifest["selection"]["val31"])
    summary = summarize(manifest["rows"], scope)
    gates = exit_criteria(summary, manifest["rows"])
    # only the boolean gates count as unmet reasons (``missing_rows`` is the
    # evidence list, not itself a verdict)
    bool_gates = ("all_candidates_processed", "train_label_quality_complete",
                  "six_subjects_usable_train_windows")
    reasons = [k for k in bool_gates if gates.get(k) is not True]
    return {"passed": not reasons, "reasons": reasons,
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
    manifest_path = PROJECT_ROOT / rec["training_manifest"]["path"]
    if integ.sha256_file(manifest_path) != rec["training_manifest"]["sha256"]:
        reasons.append("cache_manifest_binding_drifted")
    prereg_path = run_root / "freeze" / "preregistration.json"
    if integ.sha256_file(prereg_path) != rec["preregistration"]["sha256"]:
        reasons.append("cache_prereg_binding_drifted")
    root_stats = cache["dir"] / "root_stats.npz"
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
                schedule_cache_dir=cache["dir"] / "schedules")
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


def check_c(run_root: Path) -> dict:
    reasons: list[str] = []
    per_model = {}
    for model in ("deterministic", "diffusion"):
        att = latest_attempt(run_root / "s0" / model)
        if att is None:
            reasons.append(f"s0_{model}_no_attempts")
            per_model[model] = None
            continue
        summary = json.loads((PROJECT_ROOT / att["summary"]).read_text())
        per_model[model] = {"verdict": summary["verdict"],
                            "gates": summary["gates"],
                            "optimizer_steps": summary["optimizer_steps"]}
        if summary["verdict"] != "passed":
            reasons.append(f"s0_{model}_{summary['verdict']}")
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
    c = check_c(run_root)

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
