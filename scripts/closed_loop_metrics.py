"""Closed-loop evaluation bookkeeping (P0-fix F4, p0_fix_plan section 5).

Pure functions shared by eval_generation.py / eval_repair.py:

- paired per-demo task-MPJPE comparison against the teacher baseline on the
  COMMON accepted set only (plan 10.1 formula); n=0 is undefined, never 0;
- success-weighted hand contact F1 over the pre-registered full set
  (rejected -> 0), with the accepted-set raw value and the dual-empty
  undefined policy kept explicit (never silently filled with 1);
- P0.3 relative conditions (acceptance / paired MPJPE + 2 cm / success
  weighted F1 >= 0.95x baseline); any missing or undefined input leaves the
  condition unpassed;
- product metadata (code revision, mode, task gate, inputs, seeds) with
  reuse validation so stale baselines cannot be silently mixed in.

No FK/geometry here on purpose: callers pass per-sample rows that already
carry computed numbers, so every rule is unit-testable without MuJoCo.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MPJPE_MARGIN_M = 0.02      # plan 10.1: paired MPJPE <= baseline + 2 cm
F1_RATIO = 0.95            # plan 10.1: success-weighted F1 >= 0.95 x baseline


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# -- paired comparison -----------------------------------------------------------


def paired_task_mpjpe(method_rows: list[dict], baseline_rows: list[dict],
                      key: str = "task_mpjpe_box_m") -> dict:
    """Per-demo comparison on demos BOTH sides accepted (plan 10.1).

    A method's total success count is never the pairing denominator. Rows
    missing the metric on an accepted common demo are counted as
    ``n_metric_missing`` and excluded from the mean -- and the caller's
    condition check treats that as evidence loss, not as a pass.
    """
    base = {r["demo"]: r for r in baseline_rows if r.get("accepted")}
    demos, m_vals, b_vals, missing = [], [], [], 0
    for r in method_rows:
        if not r.get("accepted") or r["demo"] not in base:
            continue
        m, b = r.get(key), base[r["demo"]].get(key)
        if m is None or b is None:
            missing += 1
            continue
        demos.append(r["demo"])
        m_vals.append(float(m))
        b_vals.append(float(b))
    if not demos:
        return {"n": 0, "paired_mean_m": None, "baseline_mean_m": None,
                "diff_mean_m": None, "demos": [], "n_metric_missing": missing}
    d = np.asarray(m_vals) - np.asarray(b_vals)
    return {"n": len(demos), "paired_mean_m": float(np.mean(m_vals)),
            "baseline_mean_m": float(np.mean(b_vals)),
            "diff_mean_m": float(np.mean(d)), "demos": sorted(demos),
            "n_metric_missing": missing}


# -- success-weighted hand F1 ----------------------------------------------------


def hand_f1_report(rows: list[dict], sides=("hand_f1_left", "hand_f1_right")) -> dict:
    """Accepted-set raw and full-set success-weighted hand F1, per side.

    Caliber: per side, demo-macro. Raw = mean over accepted samples whose
    F1 is DEFINED (dual-empty stays undefined, excluded from the raw mean
    and reported via n_defined/n_undefined); success-weighted = mean over
    ALL rows with rejected and undefined samples scoring 0. hand_macro is
    the mean of the two sides' success-weighted values.
    """
    out = {"n_rows": len(rows)}
    weighted = []
    for side in sides:
        accepted_defined = [r[side] for r in rows
                            if r.get("accepted") and r.get(side) is not None]
        n_acc = sum(1 for r in rows if r.get("accepted"))
        n_rej = len(rows) - n_acc
        if accepted_defined or n_rej:
            # rejected rows legitimately score 0 (rejection is evidence);
            # accepted-but-undefined rows also score 0 in the weighted mean
            # but are tracked via n_undefined -- when there is NEITHER a
            # defined sample NOR a rejection, the side carries no evidence
            # at all and stays undefined instead of silently reading 0.0
            w = float(np.mean([
                r.get(side) if r.get("accepted") and r.get(side) is not None else 0.0
                for r in rows]))
        else:
            w = None
        weighted.append(w)
        out[side] = {
            "raw_mean": float(np.mean(accepted_defined)) if accepted_defined else None,
            "n_accepted": n_acc,
            "n_defined": len(accepted_defined),
            "n_undefined": n_acc - len(accepted_defined),
            "success_weighted": w,
        }
    out["hand_macro_success_weighted"] = (
        float(np.mean(weighted)) if all(w is not None for w in weighted) else None)
    return out


# -- P0.3 relative conditions ----------------------------------------------------


def summarize_method(rows: list[dict], baseline_rows: list[dict],
                     margin_m: float = MPJPE_MARGIN_M, f1_ratio: float = F1_RATIO) -> dict:
    """Acceptance, paired MPJPE and success-weighted hand F1 vs the teacher
    baseline, plus the three P0.3 conditions. Any missing or undefined
    metric leaves its condition unpassed (None), never silently True."""
    def acceptance(rs):
        return float(np.mean([bool(r.get("accepted")) for r in rs])) if rs else None

    acc, acc_b = acceptance(rows), acceptance(baseline_rows)
    paired = paired_task_mpjpe(rows, baseline_rows)
    f1 = hand_f1_report(rows)["hand_macro_success_weighted"]
    f1_b = hand_f1_report(baseline_rows)["hand_macro_success_weighted"]

    c_acc = None if acc is None or acc_b is None else bool(acc >= acc_b)
    c_mpjpe = (None if paired["diff_mean_m"] is None
               else bool(paired["diff_mean_m"] <= margin_m))
    if c_mpjpe is not None and paired["n_metric_missing"]:
        c_mpjpe = None   # lost evidence on common accepted demos
    c_f1 = None if f1 is None or f1_b is None else bool(f1 >= f1_ratio * f1_b)
    return {
        "n": len(rows),
        "n_accepted": sum(1 for r in rows if r.get("accepted")),
        "acceptance": acc,
        "teacher_baseline_acceptance": acc_b,
        "paired_task_mpjpe": paired,
        "success_weighted_hand_macro_f1": f1,
        "teacher_baseline_success_weighted_hand_macro_f1": f1_b,
        "conditions": {
            "acceptance_ge_baseline": c_acc,
            "paired_mpjpe_le_baseline_plus_2cm": c_mpjpe,
            "success_weighted_f1_ge_0.95x_baseline": c_f1,
        },
        "all_conditions_met": (all(c is True for c in (c_acc, c_mpjpe, c_f1))
                               if None not in (c_acc, c_mpjpe, c_f1) else None),
    }


# -- product metadata ------------------------------------------------------------


def metadata_block(mode: str, task_gate: float, files: list[str], seed: int,
                   models: dict | None = None, extra: dict | None = None) -> dict:
    return {
        "git_rev": git_rev(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "task_mpjpe_tol_m": float(task_gate),
        "files": sorted(files),
        "seed": int(seed),
        "models": models or {},
        **({"extra": extra} if extra else {}),
    }


def metadata_compatible(stored: dict, mode: str, task_gate: float,
                        files: list[str], seed: int, models: dict | None = None) -> tuple[bool, str]:
    """Exact-match reuse rule: code revision, mode, gate, inputs, seed and
    model checksums must all agree; otherwise the stored product must be
    recomputed, never quietly mixed in."""
    models = models or {}
    checks = [
        ("git_rev", stored.get("git_rev"), git_rev()),
        ("mode", stored.get("mode"), mode),
        ("task_mpjpe_tol_m", float(stored.get("task_mpjpe_tol_m", -1)), float(task_gate)),
        ("files", sorted(stored.get("files", [])), sorted(files)),
        ("seed", stored.get("seed"), int(seed)),
        ("models", stored.get("models", {}), models),
    ]
    for name, a, b in checks:
        if a != b:
            return False, f"{name} mismatch: stored={a!r} current={b!r}"
    return True, "ok"
