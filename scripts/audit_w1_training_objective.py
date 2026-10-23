"""Synthetic contract audit for the W1 training objective (A0-A5).

Default mode is **synthetic only**: analytic states, a synthetic FK stand-in and
minimal loop stand-ins.  There is deliberately no option to consume a real
training manifest, keyed cache, checkpoint, demonstration list or research
budget.  Passing any of those is an error, not a silent fallback: real-data
audits and single-factor training experiments require a separate, versioned
pre-registration (plan section 5).

The old diagnostic entry points are not imported or executed here.

Run:
  uv run --locked python scripts/audit_w1_training_objective.py \
    --out derived/audit/w1_training_objective_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ireset.eval.training_objective_audit import (  # noqa: E402
    AUDIT_SCHEMA,
    run_synthetic_audit,
)

_EXIT = {"synthetic_contract_passed": 0,
         "implementation_mismatch": 2,
         "audit_incomplete": 3}

#: research inputs that must never enter this synthetic entry
_FORBIDDEN = (
    "training_manifest", "cache_dir", "checkpoint", "execution_registration",
    "preregistration", "budget_deadline_s", "demos", "steps", "seed",
    "model", "source_dir", "data_path", "max_files", "max_windows",
)


def _guard_out(out: Path) -> Path:
    resolved = Path(out).expanduser().resolve()
    experiments = (ROOT / "derived" / "experiments").resolve()
    if resolved == experiments or experiments in resolved.parents:
        raise SystemExit(
            f"refusing to write synthetic audit artifacts into a research "
            f"experiment root: {resolved}")
    for name in ("resources", "third_party", "src", "scripts", "tests"):
        forbidden = (ROOT / name).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise SystemExit(f"refusing to write audit artifacts under {name}/: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        # never overwrite a previous audit; allocate a new attempt
        n = 1
        while (resolved / f"attempt_{n:03d}").exists():
            n += 1
        resolved = resolved / f"attempt_{n:03d}"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "derived/audit/w1_training_objective_audit",
                    help="artifact directory (a new attempt_NNN is allocated if non-empty)")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/train/omniretarget_motion.yaml",
                    help="text config whose effective values are audited")
    # Defined so the rejection is explicit and testable, never a silent no-op.
    ap.add_argument("--training-manifest", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--execution-registration", type=Path, default=None)
    ap.add_argument("--preregistration", type=Path, default=None)
    ap.add_argument("--budget-deadline-s", type=float, default=None)
    ap.add_argument("--demos", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--source-dir", type=Path, default=None)
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    supplied = [name for name in _FORBIDDEN if getattr(args, name) is not None]
    if supplied:
        raise SystemExit(
            "synthetic audit refuses real/research inputs: "
            + ", ".join(f"--{n.replace('_', '-')}" for n in supplied)
            + " — a real audit needs a separate pre-registration")

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    out = _guard_out(args.out)
    print(f"[audit] synthetic only; config={config_path} out={out}", flush=True)
    summary = run_synthetic_audit(out, ROOT, config_path)
    print(f"[audit] schema={AUDIT_SCHEMA}")
    print(f"[audit] conclusion={summary['conclusion']}")
    print(f"[audit] rows={summary['counts']['rows']} "
          f"failures={summary['counts']['failures']} gaps={summary['counts']['gaps']}")
    print(f"[audit] artifact={out}")
    return _EXIT.get(summary["conclusion"], 3)


if __name__ == "__main__":
    raise SystemExit(main())
