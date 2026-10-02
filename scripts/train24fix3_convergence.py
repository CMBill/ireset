"""train24fix3 convergence-recovery driver (plan sections 4/5/6/7/8/9/13).

One experiment root per invocation series
(``derived/data/teachers_regenerated/train24fix3_<date>_<shortrev>/``); every
stage writes ``<root>/<stage>/<stage>_summary.json`` and the driver REFUSES to
run a stage whose prerequisite summary is missing or not ok -- a stage is
never inferred from "the directory exists".

Stages:

- ``baseline`` : F re-verification (fb 5 + cf 113 strict replays into NEW
  directories -- the historical ``diag_replay`` stays untouched), the 249
  product source-manifest liveness check, H/B/C/T reference registration and
  the N0 identity freeze (code/env/assets/effective configs + the content of
  every untracked workspace file, plan section 12).  Must run BEFORE any
  algorithm change: the F products' integrity manifest is compared against
  the live tree.
- ``trace``    : five frozen frames' full escalation-chain trace (plan 5).
- ``numeric``  : N1 -> N2 -> N3 fixed comparison (plan 6).
- ``norm``     : S1 -> S2 (plan 7; only when numeric found no winner).
- ``sixseg``   : formal stage-E acceptance for the SELECTED profile (plan 8).
- ``train24``  : stage F for the FROZEN profile (plan 9).

Run: uv run --locked python scripts/train24fix3_convergence.py --stage baseline
        --out derived/data/teachers_regenerated/train24fix3_20261002_<rev>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hashlib  # noqa: E402

from closed_loop_metrics import git_rev  # noqa: E402
from experiment_provenance import (  # noqa: E402
    asset_hashes,
    env_snapshot,
    git_state,
)


def sha256_full(path: Path) -> str:
    """FULL-length digest. The F source manifest and the integrity chain use
    64-hex digests; ``closed_loop_metrics.sha256`` is the 16-hex truncated
    variant -- comparing them across tools is a documented false-positive
    source (2026-10-02 engineering note), so this driver hashes full-length
    everywhere it compares or registers file content."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
from ireset.paths import DERIVED_DATA_DIR, PROJECT_ROOT  # noqa: E402
from ireset.repair.config_io import load_mode_configs  # noqa: E402
from train24fix2_audit import (  # noqa: E402
    integrity_manifest,
    load_index,
)
from dataclasses import asdict  # noqa: E402

TEACHERS = DERIVED_DATA_DIR / "teachers_regenerated"
#: plan section 3.2 reference registry (read-only)
F_ROOT = TEACHERS / "train24fix2_reviewfix3_20261002"
REFERENCES = {
    "H": {"root": TEACHERS / "labelq_20260926_sixseg", "subdirs": ("rq", "rv", "ns"),
          "expect": {"rq_npz": 5}},
    "B": {"root": TEACHERS / "train24fix_20260927_sixseg",
          "subdirs": ("rq", "rv", "ns"), "expect": {"rq_npz": 6}},
    "C": {"root": TEACHERS / "train24fix2_20260927_sixseg",
          "subdirs": ("rq", "rv", "ns"), "expect": {"rq_npz": 1}},
    "T": {"root": TEACHERS / "labelq_20260926_train24" / "regen_final",
          "subdirs": (), "expect": {"npz": 13}},
}


class Gate:
    """Named checks; the stage summary carries every verdict explicitly."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, ok: bool, detail) -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[{'ok ' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def write_guarded(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def stage_summary_path(root: Path, stage: str) -> Path:
    return root / stage / f"{stage}_summary.json"


def load_stage_summary(root: Path, stage: str) -> dict:
    path = stage_summary_path(root, stage)
    if not path.exists():
        raise RuntimeError(
            f"stage '{stage}' summary missing ({path}) -- prerequisites are "
            f"never inferred from directory existence (plan section 13)")
    summary = json.loads(path.read_text())
    if not summary.get("ok"):
        raise RuntimeError(f"stage '{stage}' did not pass (ok=false) -- "
                           f"the dependent stage refuses to start")
    return summary


# ---------------------------------------------------------------------------
# stage A: baseline (plan section 4)
# ---------------------------------------------------------------------------

def _reference_registration(gate: Gate) -> dict:
    """H/B/C/T: full-length sha256 of every file, plus expected counts."""
    out = {}
    for name, spec in REFERENCES.items():
        root = spec["root"]
        gate.check(f"ref_{name}_exists", root.is_dir(), str(root))
        files = {}
        if root.is_dir():
            for sub in spec["subdirs"] or (None,):
                base = root if sub is None else root / sub
                if not base.is_dir():
                    continue
                for p in sorted(base.iterdir()):
                    if p.is_file():
                        files[str(p.relative_to(root))] = sha256_full(p)
        counts = {
            "rq_npz": sum(1 for k in files if k.startswith("rq/") and k.endswith(".npz"))
            if "rq" in spec["subdirs"] else None,
            "npz": sum(1 for k in files if k.endswith(".npz")),
        }
        expect = spec["expect"]
        exp_checks = {k: v for k, v in expect.items()}
        ok = all(counts[k] == v for k, v in exp_checks.items())
        gate.check(f"ref_{name}_counts", ok,
                   {k: [counts[k], v] for k, v in exp_checks.items()})
        out[name] = {"root": str(root), "n_files": len(files), "files": files}
    return out


def _untracked_registration(out_dir: Path) -> list[dict]:
    """Plan section 12: every untracked workspace file gets a full content
    hash AND a recoverable copy -- ``git_state``'s status hash covers the
    status text only, and ``dirty_evidence`` copies just the code dirs."""
    import subprocess

    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT).stdout
    rows = []
    for line in status.splitlines():
        code, rel = line[:2], line[3:].strip()
        if code.strip() == "??":
            src = PROJECT_ROOT / rel
            if src.is_file():
                dst = out_dir / "untracked" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                rows.append({"path": rel, "status": "??",
                             "sha256": sha256_full(src), "copy": str(dst)})
            elif src.is_dir():
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        rel2 = p.relative_to(PROJECT_ROOT)
                        dst = out_dir / "untracked" / rel2
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dst)
                        rows.append({"path": str(rel2), "status": "??",
                                     "sha256": sha256_full(p), "copy": str(dst)})
    return rows


def stage_baseline(root: Path) -> None:
    from train24fix2_cert_trace import replay_counterfactuals
    from train24fix2_fail_bind import replay as fb_replay

    out = root / "baseline"
    gate = Gate()
    diag = F_ROOT / "diag"

    # 1. index dual views + actual counts (plan 4.1 task 1)
    fb_idx = load_index(diag, "fail_bind2_index")
    cf_idx = load_index(diag, "cert_trace_cf_index")
    gate.check("fb_index_count", len(fb_idx["stems"]) == 5,
               f"stems={len(fb_idx['stems'])}")
    gate.check("cf_index_count", len(cf_idx["stems"]) == 113,
               f"stems={len(cf_idx['stems'])} records={len(cf_idx.get('records', []))}")

    # 2. strict replays into NEW directories (the historical diag_replay of
    # reviewfix3 is read-only; the CLI default output path is NOT used)
    fb_dir = out / "fb_replay"
    fb_summary = fb_replay(diag, fb_dir)
    gate.check("fb_replay_all_match",
               fb_summary["ok"] and fb_summary["n"] == fb_summary["expected"] == 5,
               f"ok={fb_summary['ok']} n={fb_summary['n']}/{fb_summary['expected']}")
    cf_dir = out / "cf_replay"
    replay_counterfactuals(diag, cf_dir)
    cf_summary = json.loads((cf_dir / "cert_trace_cf_replay.json").read_text())
    gate.check("cf_replay_all_match",
               cf_summary["ok"] and cf_summary["n"] == cf_summary["expected"] == 113,
               f"ok={cf_summary['ok']} n={cf_summary['n']}/{cf_summary['expected']}")

    # 3. source manifest: all 249 products still hash-match (liveness)
    sm = json.loads((F_ROOT / "source_manifest.json").read_text())
    drifted = [p["path"] for p in sm["products"]
               if sha256_full(F_ROOT / p["path"]) != p["sha256"]]
    gate.check("source_manifest_liveness",
               not drifted and len(sm["products"]) == 249,
               f"products={len(sm['products'])} drifted={drifted[:5]}")

    # 4. H/B/C/T reference registration
    refs = _reference_registration(gate)

    # 5. N0 identity freeze (plan 4.1 task 2 + section 12)
    rcfg, vcfg = load_mode_configs("strict")
    n0 = {
        "purpose": "N0 = the frozen pre-train24fix3 behaviour; the numeric "
                   "profiles are judged against THIS identity",
        "git": git_state(),
        "git_rev_full": git_rev(),
        "env": env_snapshot(),
        "assets": asset_hashes(),
        "uv_lock_sha256": sha256_full(PROJECT_ROOT / "uv.lock"),
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "effective_strict": {"repair": asdict(rcfg), "validation": asdict(vcfg)},
        "integrity_manifest": integrity_manifest(),
    }
    untracked = _untracked_registration(out)
    n0["untracked_files"] = untracked
    gate.check("n0_tree_state_recorded", True,
               f"dirty={n0['git']['dirty']} untracked={len(untracked)} "
               f"({[u['path'] for u in untracked]})")

    write_guarded(out / "n0_identity.json", json.dumps(n0, indent=1, default=float))
    summary = {
        "stage": "baseline", "ok": gate.ok,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "f_root": str(F_ROOT),
        "fb_replay_dir": str(fb_dir), "cf_replay_dir": str(cf_dir),
        "references": refs, "checks": gate.checks,
        "next": "trace (requires algorithm commit + profiles registry; "
                "plan sections 4.1 task 3-5 and 5)",
    }
    write_guarded(stage_summary_path(root, "baseline"),
                  json.dumps(summary, indent=1, default=float))
    print(f"baseline -> {'PASS' if gate.ok else 'FAIL'} "
          f"({stage_summary_path(root, 'baseline')})")


# ---------------------------------------------------------------------------
# stage B: trace (plan section 5) -- implemented with the trace hook commit
# ---------------------------------------------------------------------------

def stage_trace(root: Path) -> None:
    load_stage_summary(root, "baseline")
    raise SystemExit(
        "stage trace requires the trace-hook implementation commit "
        "(plan section 5); not yet available in this revision")


def stage_numeric(root: Path) -> None:
    load_stage_summary(root, "baseline")
    load_stage_summary(root, "trace")
    raise SystemExit(
        "stage numeric requires the N1/N2/N3 implementation commits "
        "(plan section 6); not yet available in this revision")


def stage_norm(root: Path) -> None:
    numeric = load_stage_summary(root, "numeric")
    if numeric.get("winner"):
        raise SystemExit(
            "stage norm is NOT to run: a numeric profile satisfied the full "
            "six-segment development gates (plan section 6 exit)")
    raise SystemExit(
        "stage norm requires the norm_cut implementation commit "
        "(plan section 7); not yet available in this revision")


def stage_sixseg(root: Path) -> None:
    raise SystemExit("stage sixseg requires the selection summary (plan "
                     "section 8); not yet implemented in this revision")


def stage_train24(root: Path) -> None:
    raise SystemExit("stage train24 requires the formal sixseg pass (plan "
                     "section 9); not yet implemented in this revision")


STAGES = {
    "baseline": stage_baseline,
    "trace": stage_trace,
    "numeric": stage_numeric,
    "norm": stage_norm,
    "sixseg": stage_sixseg,
    "train24": stage_train24,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--out", required=True,
                    help="experiment root (train24fix3_<date>_<shortrev>)")
    ap.add_argument("--profile", default=None,
                    help="profile name for sixseg/train24 (frozen before the "
                         "run starts; never used for candidate search)")
    args = ap.parse_args()
    if args.stage in ("sixseg", "train24") and not args.profile:
        raise SystemExit(f"--profile is required for --stage {args.stage}")
    root = Path(args.out).resolve()
    STAGES[args.stage](root)


if __name__ == "__main__":
    main()
