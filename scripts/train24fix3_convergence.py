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
    arrays_diff,
    attach_integrity,
    classify_branches,
    configs_from_record,
    context_arrays,
    context_from,
    deep_diff,
    freeze_context,
    full_candidate_audit,
    integrity_manifest,
    load_index,
    load_products,
    verify_integrity,
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
# stage B: trace (plan section 5)
# ---------------------------------------------------------------------------

def _thread_env() -> dict:
    env = dict(os.environ)
    env.update({k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                 "VECLIB_MAXIMUM_THREADS")})
    return env


def _trace_jsonable(trace: dict) -> dict:
    """Trace minus the heavy arrays (those go to the NPZ bundle)."""
    out = {"n_attempts": len(trace.get("attempts", []))}
    atts = []
    for att in trace.get("attempts", []):
        a = {k: v for k, v in att.items() if k != "qp_arrays"}
        iters = [{k: v for k, v in it.items()
                  if k not in ("z", "z_next", "xi_raw", "xi_clipped")}
                 for it in att.get("iterations", [])]
        a["iterations"] = iters
        atts.append(a)
    out["attempts"] = atts
    return out


def _trace_arrays(trace: dict) -> dict:
    import numpy as np

    arr: dict = {}
    for ai, att in enumerate(trace.get("attempts", [])):
        for ii, it in enumerate(att.get("iterations", [])):
            for key in ("z", "z_next", "xi_raw", "xi_clipped"):
                v = it.get(key)
                if v is not None:
                    arr[f"a{ai}_it{ii}_{key}"] = np.asarray(v, dtype=float)
        for qi, qa in enumerate(att.get("qp_arrays", [])):
            # Q3: row/col name arrays are per-QP evidence -- a dynamic
            # assembly (cuts added, slack layout) carries its OWN
            # descriptors, never the attempt-level cached set
            for key in ("P", "q", "A", "lo", "hi", "row_names", "col_names"):
                if key in qa:
                    arr[f"a{ai}_qp{qi}_{key}"] = np.asarray(qa[key])
    return arr


def stage_trace(root: Path) -> None:
    """Plan section 5: the five frozen frames' full escalation chains under
    the N0 behaviour, with the complete per-iteration trace.  Identity
    reproduction (same failure, same exit) is the gate; the trace itself
    is the deliverable."""
    import numpy as np
    from ireset.repair.pipeline import RepairPipeline
    from train24fix2_audit import (
        configs_from_record,
        context_from,
        escalate_frame,
        load_products,
    )

    load_stage_summary(root, "baseline")
    out = root / "trace"
    gate = Gate()
    diag = F_ROOT / "diag"
    idx = load_index(diag, "fail_bind2_index")
    summary_rows = []
    for stem in idx["stems"]:
        record, arrays = load_products(diag, stem)
        cfg, vcfg = configs_from_record(record)
        spec = RepairPipeline(cfg=cfg).spec
        ctx = context_from(arrays, record["provenance"]["context_meta"])
        t = record["t_fail"]
        pipe = RepairPipeline(spec=spec, cfg=cfg)
        pipe.validator.cfg = vcfg
        pipe.solver.trace = {"attempts": []}
        t0 = time.perf_counter()
        esc = escalate_frame(pipe, ctx, t, ctx.z_pred, ctx.z_prev)
        wall = time.perf_counter() - t0
        trace = pipe.solver.trace
        frozen = record["last_attempt"]["solver_fields"]
        atts = trace.get("attempts", [])
        ident = {
            "t_fail": t,
            "skipped": bool(esc["skipped"]),
            "adopted": bool(esc["adopted"]),
            "esc_last_status": esc["last_status"],
            "esc_last_term": esc["last_term_reason"],
            "frozen_status": frozen["status"],
            "frozen_term": frozen["term_reason"],
            "frozen_iterations": frozen["iterations"],
            "frozen_qp_calls": frozen["qp_calls"],
            "replay_last_iterations": (atts[-1]["exit"]["it_done"]
                                       if atts and atts[-1].get("exit")
                                       else None),
            "replay_last_qp_calls": (atts[-1]["exit"]["qp_calls"]
                                     if atts and atts[-1].get("exit")
                                     else None),
        }
        matches = (not esc["skipped"] and not esc["adopted"]
                   and esc["last_status"] == frozen["status"]
                   and esc["last_term_reason"] == frozen["term_reason"]
                   and ident["replay_last_iterations"] == frozen["iterations"]
                   and ident["replay_last_qp_calls"] == frozen["qp_calls"])
        gate.check(f"identity_{stem}", matches, ident)
        write_guarded(out / f"trace_{stem}.json", json.dumps(
            {"stem": stem, "demo": record["demo"], "wall_s": round(wall, 2),
             "identity": ident, "identity_matches": matches,
             "trace": _trace_jsonable(trace)}, indent=1, default=float))
        np.savez(out / f"trace_{stem}.npz", **_trace_arrays(trace))
        summary_rows.append(
            {"stem": stem, "demo": record["demo"], "t": t,
             "identity_matches": matches, "n_attempts": len(atts),
             "n_iterations_total": sum(len(a["iterations"]) for a in atts),
             "n_qp_total": sum(len(a["qp"]) for a in atts),
             "wall_s": round(wall, 2)})
        print(f"[trace] {stem}: identity={matches} "
              f"attempts={len(atts)} qp={summary_rows[-1]['n_qp_total']}")
    summary = {"stage": "trace", "ok": gate.ok,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "rows": summary_rows, "checks": gate.checks,
               "next": "numeric (N1 -> N2 -> N3, plan section 6)"}
    write_guarded(stage_summary_path(root, "trace"),
                  json.dumps(summary, indent=1, default=float))
    print(f"trace -> {'PASS' if gate.ok else 'FAIL'} "
          f"({stage_summary_path(root, 'trace')})")


# ---------------------------------------------------------------------------
# stage C: numeric candidates (plan section 6)
# ---------------------------------------------------------------------------

def _profile_frozen_frames(prof, diag: Path, pdir: Path | None = None) -> dict:
    """Five frozen frames under ONE profile (plan 6.2 item 1).  Q1 fixes
    two defects of the first implementation: the adopted candidate's full
    audit now runs WITH THE SAME FROZEN PROFILE (the pre-Q1 call audited
    norm_cut adoptions through the axis interpreter), and the audit's
    arrays are KEPT -- frozen into the recovery product, never discarded.
    Outcomes are listed separately (legal-solve / adopted-but-audit-fail /
    evidence-incomplete / not-adopted), never merged into one number.
    ``pdir`` (the profile directory) enables product freezing for the
    strict replay (plan 4.4)."""
    import numpy as np
    from ireset.repair.pipeline import RepairPipeline
    from ireset.repair.profiles import profiles_yaml_sha256
    from train24fix2_audit import (
        configs_from_record,
        context_from,
        escalate_frame,
        full_candidate_audit,
        load_products,
    )

    idx = load_index(diag, "fail_bind2_index")
    rows = []
    audits: dict[str, dict] = {}
    ff_dir = pdir / "ff" if pdir is not None else None
    for stem in idx["stems"]:
        record, arrays = load_products(diag, stem)
        cfg, vcfg = configs_from_record(record)
        spec = RepairPipeline(cfg=cfg).spec
        ctx = context_from(arrays, record["provenance"]["context_meta"])
        t = record["t_fail"]
        pipe = RepairPipeline(spec=spec, cfg=cfg, c2_profile=prof)
        pipe.validator.cfg = vcfg
        t0 = time.perf_counter()
        esc = escalate_frame(pipe, ctx, t, ctx.z_pred, ctx.z_prev)
        wall = time.perf_counter() - t0
        frozen = record["last_attempt"]["solver_fields"]
        row = {"stem": stem, "demo": record["demo"], "t": t,
               "skipped": bool(esc["skipped"]),
               "adopted": bool(esc["adopted"]),
               "adopted_source": esc["adopted_source"],
               "last_status": esc["last_status"],
               "last_term_reason": esc["last_term_reason"],
               "n_attempts": len(esc["attempts"]),
               "wall_s": round(wall, 2),
               "frozen_status": frozen["status"],
               "frozen_term": frozen["term_reason"]}
        prefix = ([arrays["prefix"][i].copy()
                   for i in range(arrays["prefix"].shape[0])]
                  if arrays["prefix"].size else [])
        if esc["adopted"]:
            # Q1: the audit judges the SAME semantics the candidate ran
            # under -- the frozen profile goes in, and the returned arrays
            # (previously discarded) are frozen with the record
            arec, aarr = full_candidate_audit(
                spec, cfg, vcfg, ctx, esc["adopted_z"],
                prefix=prefix, box_poses=arrays["box_poses"],
                label="profile_adopted", c2_profile=prof)
            audits[stem] = arec
            row["adopted_audit_branch_pass"] = arec.get("branch_pass")
            if arec.get("verdict") == "EVIDENCE_INCOMPLETE":
                row["outcome"] = "adopted_evidence_incomplete"
            elif arec.get("branch_pass"):
                row["outcome"] = "adopted_audit_pass"
            else:
                row["outcome"] = "adopted_audit_fail"
            if ff_dir is not None:
                prod = {
                    "format": RECOVERY_FORMAT, "kind": "frozen_frame",
                    "stem": f"ff_{stem}",
                    "profile": {"name": prof.name,
                                "resolved": prof.as_record(),
                                "registry_sha256": profiles_yaml_sha256()},
                    "input": {
                        "demo": record["demo"], "t": t,
                        "pass_index": record.get("pass_index"),
                        "attempt_id": record["last_attempt"].get("attempt_id"),
                        "frozen_last_status": frozen["status"],
                        "frozen_last_term": frozen["term_reason"]},
                    "source": {
                        "diag_root": str(diag), "stem": stem,
                        "json_sha256": sha256_full(diag / f"{stem}.json"),
                        "npz_sha256": sha256_full(diag / f"{stem}.npz")},
                    "escalation": {
                        k: row[k] for k in
                        ("skipped", "adopted", "adopted_source",
                         "last_status", "last_term_reason", "n_attempts")},
                    "effective": {"repair": asdict(cfg),
                                  "validation": asdict(vcfg)},
                    "audit": arec,
                    "replay_meta": record["provenance"]["context_meta"],
                }
                attach_integrity(prod, cfg, vcfg)
                _write_products(ff_dir, f"ff_{stem}", prod, {
                    **context_arrays(ctx), **aarr,
                    "adopted_z": np.asarray(esc["adopted_z"], dtype=float),
                    "prefix": np.asarray(prefix, dtype=float).reshape(
                        -1, 36),
                    "box_poses": np.asarray(arrays["box_poses"],
                                            dtype=float)})
        else:
            row["outcome"] = "skipped" if esc["skipped"] else "not_adopted"
        rows.append(row)
        print(f"  [{prof.name}] {stem}: adopted={row['adopted']} "
              f"outcome={row['outcome']} term={row['last_term_reason']} "
              f"({row['wall_s']}s)")
    if ff_dir is not None and rows:
        write_guarded(ff_dir / "ff_index.json", json.dumps(
            {"format": RECOVERY_FORMAT,
             "stems": [f"ff_{r['stem']}" for r in rows],
             "records": [{"stem": f"ff_{r['stem']}"} for r in rows]},
            indent=1))
    counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in (
        "adopted_audit_pass", "adopted_audit_fail",
        "adopted_evidence_incomplete", "not_adopted", "skipped")}
    return {"rows": rows, "audits": audits,
            "n_adopted": sum(1 for r in rows if r["adopted"]),
            "n_full_audit_pass": counts["adopted_audit_pass"],
            "n_adopted_audit_fail": counts["adopted_audit_fail"],
            "n_evidence_incomplete": counts["adopted_evidence_incomplete"],
            "n_not_adopted": counts["not_adopted"] + counts["skipped"]}


def _run_sixseg(profile_name: str | None, rq: Path, *, verbose: bool = False,
                mode: str = "regen", set_name: str = "diagnosis6",
                note: str | None = None) -> dict:
    """One full run under the profile (plan 6.2 item 2 / 8.2-9).  ``None``
    profile = the STANDARD-MODE control (ns): no ``--profile`` flag reaches
    the subprocess (standard mode has no profile gating at all).  The
    report file name follows the tool's own mode convention
    (``regen_report_``/``standard_report_``)."""
    import subprocess

    rq.mkdir(parents=True, exist_ok=True)
    prefix = "regen_report" if mode == "regen" else "standard_report"
    report = rq / f"{prefix}_{set_name}_strict.json"
    cmd = ["uv", "run", "--locked", "python",
           "scripts/regenerate_teachers.py",
           "--set", set_name, "--mode", mode,
           "--validation-mode", "strict",
           "--assert-task-gate", "0.40",
           "--output-dir", str(rq),
           "--report", str(report),
           "--note", note or (
               f"train24fix3 formal run (profile {profile_name}, mode {mode})")]
    if profile_name is not None:
        cmd += ["--profile", profile_name]
    if verbose:
        cmd += ["--verbose"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_thread_env())
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"sixseg subprocess failed for {profile_name}")
    rep = json.loads(report.read_text())
    return {"rq_dir": str(rq), "report_path": str(report),
            "wall_s": round(wall, 1),
            "n_demos": rep["n_demos"], "n_kept": rep["n_kept"],
            "kept": [r["demo"] for r in rep["rows"] if r["accepted"]],
            "code_changed_during_run":
                rep.get("end", {}).get("code_changed_during_run"),
            "argv": rep.get("effective", {}).get("argv"),
            "profile": rep.get("profile"),
            "mode": rep.get("effective", {}).get("mode"),
            "report": rep}


def _run_labels(run_dir: Path, out_dir: Path) -> Path:
    """Export this run's graded-label manifest (plan 8.1 label row)."""
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "--locked", "python", "scripts/export_label_manifest.py",
           "--run", str(run_dir), "--out", str(out_dir),
           "--validation-mode", "strict",
           "--note", "train24fix3 formal stage label export"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_thread_env())
    if proc.returncode != 0:
        raise RuntimeError(f"label export failed for {run_dir}")
    manifest = out_dir / "label_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"label manifest missing after export: {manifest}")
    return manifest


# ---------------------------------------------------------------------------
# plan section 12: the subtask state machine (formal stages E/F)
# ---------------------------------------------------------------------------

SUBTASK_FORMAT = "train24fix3_subtasks_v1"


class SubtaskRegistry:
    """§12 continuation state machine.  Every subtask (a formal stage's
    runs / labels / gate / replay) is registered with its expected count,
    input identity (profile + registry hash + git rev + worktree state),
    output hashes, start/end, processed count and status.  ``verified``
    returns a COMPLETE attempt ONLY when the identity matches AND every
    registered output still hash-verifies -- resume reuses exactly that;
    half-written attempts stay on disk as failed evidence and a rerun goes
    to a NEW attempt directory (never an overwrite)."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "subtasks.json"
        if self.path.exists():
            self.doc = json.loads(self.path.read_text())
            if self.doc.get("format") != SUBTASK_FORMAT:
                raise ValueError(f"{self.path}: format "
                                 f"{self.doc.get('format')!r} != "
                                 f"{SUBTASK_FORMAT!r}")
        else:
            self.doc = {"format": SUBTASK_FORMAT, "subtasks": {}}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.doc, indent=1, default=float))

    def attempts(self, name: str) -> list[dict]:
        return self.doc["subtasks"].get(name, [])

    def _outputs_ok(self, att: dict) -> bool:
        root = Path(att["root"])
        outs = att.get("outputs") or {}
        if not outs:
            return False
        for rel, digest in outs.items():
            p = root / rel
            if not p.is_file() or sha256_full(p) != digest:
                return False
        return True

    def verified(self, name: str, identity: dict) -> dict | None:
        """The attempt ``--resume`` may reuse: complete, same identity, and
        still hash-verified on disk.  Anything else is not reusable."""
        for att in self.attempts(name):
            if att.get("status") != "complete" or att.get("ok") is not True:
                continue
            if att.get("identity") != identity:
                continue
            if not self._outputs_ok(att):
                continue
            return att
        return None

    def begin(self, name: str, root: Path, identity: dict,
              expected: int) -> Path:
        n = len(self.attempts(name)) + 1
        att_dir = Path(root) / name / f"attempt_{n:03d}"
        att_dir.mkdir(parents=True, exist_ok=False)   # never overwrite
        self.doc["subtasks"].setdefault(name, []).append({
            "attempt": n, "dir": str(att_dir), "root": str(root),
            "identity": identity, "expected": int(expected),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running"})
        self._write()
        return att_dir

    def complete(self, name: str, outputs: list[Path],
                 processed: int) -> dict:
        att = self.attempts(name)[-1]
        root = Path(att["root"])
        att["outputs"] = {str(p.relative_to(root)): sha256_full(p)
                          for p in outputs}
        att["processed"] = int(processed)
        att["ok"] = bool(int(processed) == att["expected"])
        att["status"] = "complete"
        att["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write()
        return att

    def fail(self, name: str, reason: str) -> None:
        if not self.attempts(name):
            return
        att = self.attempts(name)[-1]
        att["status"] = "failed"
        att["reason"] = reason
        att["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write()


def _stage_identity(profile_name: str | None, set_name: str) -> dict:
    """Per-candidate frozen identity (§12): profile + registry content hash
    + revision + worktree state.  A resume/重复运行 of the same candidate
    requires THIS to be unchanged."""
    from ireset.repair.profiles import profiles_yaml_sha256

    return {"profile": profile_name, "registry_sha256": profiles_yaml_sha256(),
            "git_rev": git_rev(), "git": git_state(),
            "set": set_name, "task_gate": 0.40}


def _subtask(reg: SubtaskRegistry, name: str, root: Path, identity: dict,
             expected: int, runner, outputs_of):
    """Run one subtask through the state machine.  ``runner(attempt_dir)``
    returns the processed count; ``outputs_of(attempt_dir)`` lists the
    files whose hashes register.  A verified same-identity completion is
    REUSED (recorded as such); any failure marks the attempt failed and
    propagates -- the half-written directory stays as evidence."""
    prior = reg.verified(name, identity)
    if prior is not None:
        return {"name": name, "reused": True, "dir": prior["dir"],
                "processed": prior["processed"], "ok": prior["ok"]}
    att_dir = reg.begin(name, root, identity, expected)
    try:
        processed = runner(att_dir)
        att = reg.complete(name, outputs_of(att_dir), processed)
        return {"name": name, "reused": False, "dir": str(att_dir),
                "processed": processed, "ok": att["ok"]}
    except Exception as err:  # noqa: BLE001 -- the attempt is evidence
        reg.fail(name, repr(err))
        raise


def _selection_winner(root: Path) -> tuple[str, dict]:
    """The selected profile from the ALGORITHM stages' own summaries (never
    inferred from directory existence): norm first (S branch), numeric
    otherwise; winner=None refuses the formal stages."""
    for stage, label in (("norm", "S"), ("numeric", "N")):
        path = stage_summary_path(root, stage)
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        if not summary.get("ok"):
            raise RuntimeError(
                f"stage '{stage}' summary is not ok -- the formal stage "
                f"refuses to start from a failed algorithm stage")
        winner = summary.get("winner")
        if winner is None:
            raise RuntimeError(
                f"stage '{stage}' recorded winner=None ({label} branch "
                f"ended without a selection) -- the formal stages require "
                f"a SELECTED profile (plan 7.3/8)")
        return winner, summary
    raise RuntimeError(
        "no algorithm-stage summary found (norm/numeric) -- the formal "
        "stages never start without one (prerequisites are never inferred "
        "from directory existence)")


def _reports_consistent(reports: list[dict]) -> tuple[bool, list[str]]:
    """§8.2 item 2: three-run consistency on the ALGORITHM-SEMANTIC fields
    (states, kept set, per-row counters); wall times and machine load are
    compared nowhere here."""
    diffs: list[str] = []
    base = reports[0]
    for i, other in enumerate(reports[1:], start=2):
        for k in ("n_demos", "n_kept", "mode"):
            if base.get(k) != other.get(k):
                diffs.append(f"run{i}.{k}: {base.get(k)!r} != {other.get(k)!r}")
        brow = {r["demo"]: r for r in base["rows"]}
        orow = {r["demo"]: r for r in other["rows"]}
        if set(brow) != set(orow):
            diffs.append(f"run{i}: demo sets differ "
                         f"{sorted(set(brow) ^ set(orow))[:6]}")
        for demo in sorted(set(brow) & set(orow)):
            for k in ("accepted", "status", "t_fail", "skip_reason",
                      "slack_frames"):
                if brow[demo].get(k) != orow[demo].get(k):
                    diffs.append(f"run{i}.{demo}.{k}: "
                                 f"{brow[demo].get(k)!r} != "
                                 f"{orow[demo].get(k)!r}")
    return not diffs, diffs


def _npz_bytes_equal(dirs: list[Path], kept: list[str]) -> tuple[bool, list[str]]:
    """Kept trajectory NPZs byte-identical across the compared runs."""
    diffs: list[str] = []
    base = Path(dirs[0])
    for demo in kept:
        hashes = [sha256_full(d / f"{demo}.npz")
                  for d in dirs if (d / f"{demo}.npz").is_file()]
        if len(hashes) != len(dirs) or len(set(hashes)) != 1:
            diffs.append(f"{demo}.npz: {len(set(hashes))} distinct hashes "
                         f"across {len(dirs)} runs")
    return not diffs, diffs


def stage_numeric(root: Path) -> None:
    """Plan section 6: N1 -> N2 -> N3 in order; the FIRST profile meeting
    the full six-segment development gates is selected and the remaining
    candidates are recorded as not-needed-to-execute (a started comparison
    always completes its full sixseg)."""
    from ireset.repair.profiles import load_profiles
    from train24fix3_quality_gate import gate_report

    load_stage_summary(root, "baseline")
    load_stage_summary(root, "trace")
    out = root / "numeric"
    diag = F_ROOT / "diag"
    gate = Gate()
    profiles = load_profiles()
    winner = None
    comparisons: dict[str, dict] = {}
    not_executed: list[str] = []
    for name in ("N1", "N2", "N3"):
        if winner is not None:
            not_executed.append(name)
            continue
        prof = profiles[name]
        pdir = out / name
        ff = _profile_frozen_frames(prof, diag, pdir)
        six = _run_sixseg(name, pdir / "sixseg" / "rq")
        qg = gate_report(Path(six["rq_dir"]))
        base_ok = (six["n_kept"] == six["n_demos"]
                   and six["code_changed_during_run"] is False)
        cand_ok = base_ok and qg["ok"]
        comparisons[name] = {
            "profile": prof.as_record(),
            "frozen_frames": {k: ff[k] for k in (
                "rows", "n_adopted", "n_full_audit_pass",
                "n_adopted_audit_fail", "n_evidence_incomplete",
                "n_not_adopted")},
            "frozen_frame_products": str(pdir / "ff"),
            "sixseg": {k: six[k] for k in
                       ("rq_dir", "wall_s", "n_demos", "n_kept", "kept",
                        "code_changed_during_run", "profile")},
            "quality_gate": {"verdicts": qg["verdicts"], "rows": qg["rows"]},
            "base_gate_ok": base_ok, "quality_gate_ok": qg["ok"],
            "selected": bool(cand_ok)}
        write_guarded(pdir / "profile_summary.json", json.dumps(
            comparisons[name], indent=1, default=float))
        gate.check(f"{name}_base_gate", base_ok,
                   f"keep {six['n_kept']}/{six['n_demos']}, "
                   f"r9={six['code_changed_during_run']}")
        gate.check(f"{name}_quality_gate", qg["ok"], qg["verdicts"])
        if cand_ok:
            winner = name
    # the stage COMPLETED even without a winner; the winner field decides
    # the next branch (plan 6.2 exit) -- ok is never massaged to pass
    summary = {"stage": "numeric", "ok": True,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "winner": winner, "not_executed": not_executed,
               "comparisons": comparisons,
               "checks": gate.checks,
               "next": ("sixseg (formal stage E, selected profile)"
                        if winner else "norm (S1 -> S2, plan section 7)")}
    write_guarded(stage_summary_path(root, "numeric"),
                  json.dumps(summary, indent=1, default=float))
    print(f"numeric -> winner={winner} "
          f"({stage_summary_path(root, 'numeric')})")


# ---------------------------------------------------------------------------
# stage D: norm candidates (plan section 7)
# ---------------------------------------------------------------------------

#: product format of the norm-stage counterfactual products (plan 7.2:
#: new modes use a NEW format; the v2 forensic products stay v2-decoded).
#: HISTORICAL schema (2026-10-02/03 S1+S2 runs): its records lack the
#: replay inputs (no frozen context/prefix/source/current branch) and its
#: NPZ array keys carry the double ``rows__`` prefix (Q4) -- products in
#: this format stay byte-preserved, load through this path only and are at
#: most "historical summary / evidence incomplete" (plan 4.4 item 1).
NORM_FORMAT = "train24fix3_norm_products_v1"
#: the COMPLETE diagnostic schema of the continuation round (plan 4.4):
#: identity, frozen FrameContext, candidate, accepted prefix + box
#: sequence, BOTH branch audits, source hashes, full effective configs,
#: resolved profile and the live integrity manifest.  New-format records
#: get NO config backfill (Q2) and their array keys keep exactly the
#: audit-returned single ``rows__<label>__*`` prefix (Q4).
RECOVERY_FORMAT = "train24fix3_recovery_audit_v1"
#: version marker of the dynamic per-QP trace products (plan 4.4 item 1 /
#: 5.3 -- the Q3 one-to-one solve<->metadata<->matrix binding)
QP_TRACE_FORMAT = "train24fix3_qp_trace_v2"
_V1_AUDIT_LABEL = "plus_c2_norm"
_V1_DOUBLE_PREFIX = f"rows__{_V1_AUDIT_LABEL}__rows__{_V1_AUDIT_LABEL}__"


def _write_products(pdir: Path, name: str, record: dict,
                    arrays: dict) -> None:
    """Versioned JSON + NPZ under one product ``name``; refuses overwrite;
    the NPZ is hashed INTO the JSON after writing (self-hash)."""
    import numpy as np

    pdir.mkdir(parents=True, exist_ok=True)
    jpath, npath = pdir / f"{name}.json", pdir / f"{name}.npz"
    if jpath.exists() or npath.exists():
        raise FileExistsError(f"refusing to overwrite {jpath}")
    np.savez(npath, **arrays)
    record["npz_sha256"] = sha256_full(npath)
    jpath.write_text(json.dumps(record, indent=1, allow_nan=False,
                                default=float))


def _write_norm_products(pdir: Path, stem: str, record: dict,
                         arrays: dict) -> None:
    """The historical v1 norm writer (kept verbatim for the S1/S2 layout;
    new code writes :data:`RECOVERY_FORMAT` via ``_write_products``)."""
    if record.get("format") not in (None, NORM_FORMAT):
        raise ValueError(f"the cf_ writer produces the v1 layout only "
                         f"(format {NORM_FORMAT!r}); got "
                         f"{record.get('format')!r}")
    _write_products(pdir, f"cf_{stem}", record, arrays)


def _profile_from_record(record: dict):
    """Rebuild the frozen profile from a recovery product's RESOLVED values
    (Q2).  The registry YAML may have moved on between generation and
    replay (R1-R3 add entries) -- the replay uses the product's own
    expansion, never the live registry."""
    from ireset.repair.profiles import C2Profile, OsqpSettings

    prof = record.get("profile") or {}
    resolved = prof.get("resolved")
    if not prof.get("name") or not isinstance(resolved, dict):
        raise ValueError("record carries no resolved profile -- the replay "
                         "cannot reconstruct the audited C2 semantics")
    try:
        solver = OsqpSettings(**resolved["solver"])
        kwargs = {k: v for k, v in resolved.items() if k != "solver"}
        return C2Profile(solver=solver, **kwargs)
    except (KeyError, TypeError) as err:
        raise ValueError(f"resolved profile does not reconstruct: "
                         f"{err}") from err


def load_norm_products(pdir: Path, stem: str) -> tuple[dict, dict]:
    """Load one v3 norm product: format checked, NPZ hash verified against
    the JSON's self-hash (the HISTORICAL schema -- complete verification
    and strict replay exist only for :data:`RECOVERY_FORMAT` records)."""
    import numpy as np

    jpath = pdir / f"cf_{stem}.json"
    if not jpath.exists():
        raise FileNotFoundError(f"{jpath}: norm product missing")
    record = json.loads(jpath.read_text())
    if record.get("format") != NORM_FORMAT:
        raise ValueError(f"{jpath}: format {record.get('format')!r} != "
                         f"{NORM_FORMAT!r} -- old products are decoded by "
                         f"the v2 path, never this one")
    npath = pdir / f"cf_{stem}.npz"
    digest = sha256_full(npath)
    if digest != record.get("npz_sha256"):
        raise ValueError(f"{npath}: sha256 {digest} != recorded "
                         f"{record.get('npz_sha256')}")
    with np.load(npath) as data:
        arrays = {k: data[k] for k in data.files}
    return record, arrays


def decode_norm_v1_arrays(arrays: dict) -> tuple[dict, str]:
    """Explicit decode of the HISTORICAL v1 double-prefixed array keys
    (Q4): ``rows__plus_c2_norm__rows__plus_c2_norm__*`` maps to the
    canonical single-prefix names.  Returns ``(canonical_arrays,
    layout_marker)``; a mixed or foreign layout is an explicit error --
    v1 bundles are decoded AS v1, never silently renamed, and the marker
    forbids claiming cross-version array identity."""
    out: dict = {}
    for k, v in arrays.items():
        if k.startswith(_V1_DOUBLE_PREFIX):
            out[f"rows__{_V1_AUDIT_LABEL}__"
                + k[len(_V1_DOUBLE_PREFIX):]] = v
    if not out:
        raise ValueError("no v1 double-prefixed arrays found -- a bundle "
                         "without any is not a v1 layout")
    for k in arrays:
        if not k.startswith(_V1_DOUBLE_PREFIX):
            raise ValueError(f"v1 array layout carries non-double-prefixed "
                             f"key {k!r} -- a mixed layout is corruption, "
                             f"not a v1 bundle")
    return out, "v1_double_prefix"


def load_recovery_products(pdir: Path, name: str, *,
                           live_manifest: dict | None = None
                           ) -> tuple[dict, dict]:
    """Load + VERIFY one recovery product (plan 4.4 / Q2).  Beyond the v1
    checks (format, NPZ self-hash) this loader requires: complete frozen
    effective configs that round-trip (new formats get NO backfill -- a
    missing solver field is a corrupt/foreign record), a reconstructable
    resolved profile, LIVE source files whose full-length hashes still
    match the registration, and a fresh-disk integrity check."""
    import numpy as np

    jpath = pdir / f"{name}.json"
    if not jpath.exists():
        raise FileNotFoundError(f"{jpath}: recovery product missing")
    record = json.loads(jpath.read_text())
    if record.get("format") != RECOVERY_FORMAT:
        raise ValueError(f"{jpath}: format {record.get('format')!r} != "
                         f"{RECOVERY_FORMAT!r}")
    npath = pdir / f"{name}.npz"
    digest = sha256_full(npath)
    if digest != record.get("npz_sha256"):
        raise ValueError(f"{npath}: sha256 {digest} != recorded "
                         f"{record.get('npz_sha256')}")
    try:
        configs_from_record(record)
        if record.get("kind") == "counterfactual":
            configs_from_record(record, key="repair_no_c2")
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{jpath}: frozen effective configs incomplete "
                         f"or not round-trippable: {err}") from err
    _profile_from_record(record)
    src = record.get("source")
    if not isinstance(src, dict) or not all(
            src.get(k) for k in ("diag_root", "stem", "json_sha256",
                                 "npz_sha256")):
        raise ValueError(f"{jpath}: source registration missing or "
                         f"incomplete -- provenance is mandatory, a "
                         f"sourceless record is never loaded")
    for key, suffix in (("json_sha256", ".json"), ("npz_sha256", ".npz")):
        path = Path(src["diag_root"]) / f"{src['stem']}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"{jpath}: registered source {path} "
                                    f"is missing")
        live = sha256_full(path)
        if live != src[key]:
            raise ValueError(f"{path}: source {key} {live} != recorded "
                             f"{src[key]} (tampered or wrong file)")
    drift = verify_integrity(record, live_manifest=live_manifest)
    if drift:
        raise ValueError(f"{jpath}: integrity drift since generation: "
                         f"{drift[:5]}")
    with np.load(npath) as data:
        arrays = {k: data[k] for k in data.files}
    return record, arrays


def _recovery_index(pdir: Path, name: str) -> dict:
    """Dual-view recovery index (plan 4.4 item 5): both views non-empty,
    duplicate-free and set-equal; wrong/foreign format rejected.  Q2's
    UNIFIED directory convention: ``pdir`` IS the products directory (the
    one holding ``<name>.json`` and the product files) -- no loader
    appends a subdirectory to its argument."""
    path = pdir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: recovery index missing -- "
                                f"refusing to replay an empty candidate set")
    idx = json.loads(path.read_text())
    if idx.get("format") != RECOVERY_FORMAT:
        raise ValueError(f"{path}: format {idx.get('format')!r} != "
                         f"{RECOVERY_FORMAT!r}")
    stems = idx.get("stems")
    if not isinstance(stems, list) or not stems:
        raise ValueError(f"{path}: index lists no candidates")
    if len(set(stems)) != len(stems):
        raise ValueError(f"{path}: duplicate stems in index")
    records = idx.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: index records empty while stems list "
                         f"{len(stems)}")
    rec_stems = [r.get("stem") if isinstance(r, dict) else None
                 for r in records]
    if any(x is None for x in rec_stems) or \
            len(set(rec_stems)) != len(rec_stems) or \
            set(rec_stems) != set(stems):
        raise ValueError(f"{path}: index records and stems disagree")
    return idx


def load_recovery_index(pdir: Path, name: str = "cf_index") -> dict:
    """The counterfactual products' recovery index (see
    :func:`_recovery_index`)."""
    return _recovery_index(pdir, name)


def load_norm_index(pdir: Path) -> dict:
    """The norm stage's dual-view index under the G1 contract (non-empty,
    duplicate-free, records/stems set-equal, correct format).  Q2 unified
    convention: ``pdir`` is the products directory itself (matching
    :func:`load_norm_products`); the pre-Q3 code appended ``cf/`` to a
    PARENT argument here -- one loader must not invent path levels
    another loader does not have."""
    path = pdir / "cf_index.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: norm index missing -- refusing "
                                f"to replay an empty candidate set")
    idx = json.loads(path.read_text())
    if idx.get("format") != NORM_FORMAT:
        raise ValueError(f"{path}: format {idx.get('format')!r} != "
                         f"{NORM_FORMAT!r}")
    stems = idx.get("stems")
    if not isinstance(stems, list) or not stems:
        raise ValueError(f"{path}: index lists no candidates")
    if len(set(stems)) != len(stems):
        raise ValueError(f"{path}: duplicate stems in index")
    records = idx.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: index records empty while stems list "
                         f"{len(stems)}")
    rec_stems = [r.get("stem") if isinstance(r, dict) else None
                 for r in records]
    if any(x is None for x in rec_stems) or \
            len(set(rec_stems)) != len(rec_stems) or \
            set(rec_stems) != set(stems):
        raise ValueError(f"{path}: index records and stems disagree")
    return idx


def _norm_counterfactuals(prof, diag: Path, pdir: Path, gate: Gate) -> dict:
    """113 existing counterfactuals re-audited under the S profile's REAL
    norm rows (plan 7.3).  Q2/Q4 rewrite: BOTH branches are recomputed
    under the current code -- the current (C2-off) branch with the frozen
    ``repair_no_c2`` config, the C2-on branch with the profile attached --
    and the classification comes from THESE two audits (never the frozen
    chain's old summary).  The complete inputs (identity, context,
    candidate, prefix, box sequence, both audits, source hashes, effective
    configs, resolved profile, integrity) freeze into recovery products;
    array keys keep the audit-returned SINGLE prefix (Q4 -- the first
    implementation re-prefixed already-prefixed keys).  The TEN spike
    candidates must ALL be rejected by the real norm constraint;
    historical certified candidates are classified as they come."""
    import numpy as np
    from dataclasses import asdict as _asdict
    from dataclasses import replace as _replace

    from ireset.repair.pipeline import RepairPipeline
    from ireset.repair.profiles import profiles_yaml_sha256

    idx = load_index(diag, "cert_trace_cf_index")
    out = pdir / "cf"
    rows = []
    spikes = []
    for stem in idx["stems"]:
        record, arrays = load_products(diag, stem)
        cfg, vcfg = configs_from_record(record)
        cfg_current, _ = configs_from_record(record, key="repair_no_c2")
        spec = RepairPipeline(cfg=cfg).spec
        ctx = context_from(arrays, record["replay_meta"])
        ctx_plus = _replace(
            freeze_context(ctx),
            c2_hand_enabled=record["c2_enable_flags"],
            c2_hand_reason=record["c2_reasons"],
            prev_hand_scene_dist=record["prev_scene_dist_flags"])
        prefix = ([arrays["prefix"][i].copy()
                   for i in range(arrays["prefix"].shape[0])]
                  if arrays["prefix"].size else [])
        cur_rec, cur_arr = full_candidate_audit(
            spec, cfg_current, vcfg, ctx, arrays["z_cand"],
            prefix=prefix, box_poses=arrays["box_poses"], label="current")
        arec, aarr = full_candidate_audit(
            spec, cfg, vcfg, ctx_plus, arrays["z_cand"],
            prefix=prefix, box_poses=arrays["box_poses"],
            label=_V1_AUDIT_LABEL, c2_profile=prof)
        classification = classify_branches(cur_rec, arec)
        enabled_modes = {side: (v.get("mode") if v.get("enabled") else None)
                         for side, v in
                         (arec.get("endpoint", {}).get("c2_last_hand_speed")
                          or {}).items()}
        row = {"stem": stem,
               "classification": classification,
               "plus_branch_pass": arec.get("branch_pass"),
               "current_branch_pass": cur_rec.get("branch_pass"),
               "current_branch_recomputed": True,
               "frozen_classification": record.get("classification"),
               "enabled_row_modes": enabled_modes,
               "violating": (arec.get("endpoint", {}).get("violating") or [])[:6]}
        rows.append(row)
        is_spike = "current_rq_adopted_spike" in stem
        if is_spike:
            spikes.append({"stem": stem,
                           "rejected": not arec.get("branch_pass"),
                           "mode": enabled_modes})
        prod = {"format": RECOVERY_FORMAT, "kind": "counterfactual",
                "stem": f"cf_{stem}",
                "profile": {"name": prof.name,
                            "resolved": prof.as_record(),
                            "registry_sha256": profiles_yaml_sha256()},
                "input": {"demo": record["demo"], "t": record["frame"],
                          "side": record["side"],
                          "candidate_source": record["candidate_source"]},
                "source": {"diag_root": str(diag), "stem": stem,
                           "json_sha256": sha256_full(diag / f"{stem}.json"),
                           "npz_sha256": sha256_full(diag / f"{stem}.npz"),
                           "prev_scene_source":
                               record.get("prev_scene_source")},
                "distance_views": {
                    "both_m": record["prev_scene_dist_both_m"],
                    "selected_m": record["prev_scene_dist_m"],
                    "selected_side": record["side"],
                    "flags": record["prev_scene_dist_flags"]},
                "c2_enable_flags": record["c2_enable_flags"],
                "c2_reasons": record["c2_reasons"],
                "effective": {"repair": _asdict(cfg),
                              "validation": _asdict(vcfg),
                              "repair_no_c2": _asdict(cfg_current)},
                "audits": {"current": cur_rec, _V1_AUDIT_LABEL: arec},
                "classification": classification,
                "frozen_classification": record.get("classification"),
                "replay_meta": record["replay_meta"]}
        attach_integrity(prod, cfg, vcfg)
        # Q4: the audit-returned keys go out AS-IS -- exactly one
        # ``rows__<label>__`` namespace level, never a second one
        _write_products(out, f"cf_{stem}", prod, {
            **context_arrays(ctx), **cur_arr, **aarr,
            "z_cand": np.asarray(arrays["z_cand"], dtype=float),
            "prefix": np.asarray(prefix, dtype=float).reshape(-1, 36),
            "box_poses": np.asarray(arrays["box_poses"], dtype=float)})
        print(f"  [{prof.name}] {stem}: class={classification} "
              f"pass={arec.get('branch_pass')}")
    # dual-view index (non-empty, unique, set-equal -- the G1 contract);
    # stems are the PRODUCT names (cf_<stem>), exactly what a replay loads
    write_guarded(out / "cf_index.json", json.dumps(
        {"format": RECOVERY_FORMAT,
         "stems": [f"cf_{r['stem']}" for r in rows],
         "records": [{"stem": f"cf_{r['stem']}"} for r in rows]}, indent=1))
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["classification"]] = dist.get(r["classification"], 0) + 1
    spikes_ok = (len(spikes) == 10
                 and all(sp["rejected"] for sp in spikes)
                 and all(m == "norm_cut"
                         for sp in spikes for m in sp["mode"].values()
                         if m is not None))
    enable_ok = all(m == "norm_cut" for r in rows
                    for m in r["enabled_row_modes"].values() if m is not None)
    gate.check(f"{prof.name}_spikes_all_rejected_by_norm", spikes_ok,
               f"n={len(spikes)} rejected={sum(1 for s in spikes if s['rejected'])}")
    gate.check(f"{prof.name}_enable_state_norm_rows", enable_ok,
               "every enabled side audited under norm_cut rows")
    return {"rows": rows, "distribution": dist,
            "n": len(rows), "spikes": spikes,
            "spikes_ok": spikes_ok, "enable_ok": enable_ok}


def _prev_scene_cross_check(record: dict) -> dict:
    """Four-view fresh query of the previous full-scene distance against
    the hash-verified source NPZ (plan 4.4 item 3; the reviewfix3 G2
    contract).  The source's own ``sha256`` is the historical 16-hex
    domain -- compared with the SAME function, never against a full
    digest.  A missing source is an explicit failure, never a skip."""
    from closed_loop_metrics import sha256 as sha256_16
    from train24fix2_cert_trace import load_teacher_npz, prev_scene_distances_at

    views = record.get("distance_views") or {}
    t = record["input"]["t"]
    checks: dict = {}
    if t == 0:
        if any(views.get(k, {}).get(s) is not None
               for k in ("both_m", "flags") for s in ("l", "r")) or \
                views.get("selected_m") is not None:
            checks["t0"] = "t=0 must freeze None distances"
        return checks
    src = (record.get("source") or {}).get("prev_scene_source")
    if not isinstance(src, dict) or "npz" not in src or "sha256" not in src:
        checks["source_npz"] = ("prev_scene_source missing or incomplete -- "
                                "the fresh-query cross-check is mandatory, "
                                "never skippable")
        return checks
    npz = Path(src["npz"])
    if not npz.is_file():
        checks["source_npz"] = f"{npz}: source NPZ missing"
        return checks
    digest = sha256_16(npz)
    if digest != src["sha256"]:
        checks["source_npz"] = (f"sha256 (16-hex domain) {digest} != "
                                f"recorded {src['sha256']}")
        return checks
    traj, _h, _f = load_teacher_npz(npz)
    fresh_at = prev_scene_distances_at(traj, t)
    for s in ("l", "r"):
        for view in ("both_m", "flags"):
            if views[view][s] != fresh_at[s]:
                checks[f"{view}:{s}"] = (f"saved {views[view][s]!r} != "
                                         f"fresh {fresh_at[s]!r}")
    side = views["selected_side"]
    if views["selected_m"] != fresh_at[side]:
        checks[f"selected:{side}"] = (f"saved {views['selected_m']!r} != "
                                      f"fresh {fresh_at[side]!r}")
    return checks


def replay_recovery(prof_dir: Path, out_dir: Path, *,
                    kinds: tuple[str, ...] = ("ff", "cf"),
                    live_manifest: dict | None = None) -> dict:
    """Strict replay of one profile's recovery products (plan 4.4 item 4,
    Q2).  Index-driven over both kinds; every product is loaded through
    the verifying loader; the frozen inputs rebuild the computation --
    counterfactuals recompute BOTH branch audits and re-classify (the old
    current-branch summary is never borrowed), frozen frames re-run the
    full escalation chain -- and EVERYTHING semantic is compared
    (``deep_diff`` over complete records, ``arrays_diff`` bitwise).  The
    previous-scene distances are re-queried FRESH from the hash-verified
    source NPZ across all four frozen views.  Output goes ONLY to
    ``out_dir`` (the products directory is never written); the processed
    count is part of the verdict (``n == expected``)."""
    import numpy as np
    from dataclasses import replace as _replace

    from ireset.repair.pipeline import RepairPipeline
    from train24fix2_audit import escalate_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    if live_manifest is None:
        live_manifest = integrity_manifest()
    results: dict = {"ok": True, "kinds": {}}
    for kind in kinds:
        pdir = prof_dir / kind
        idx = _recovery_index(pdir, f"{kind}_index")
        entries: dict = {}
        ok = True
        for name in idx["stems"]:
            record, arrays = load_recovery_products(
                pdir, name, live_manifest=live_manifest)
            entry: dict = {"stem": name}
            entries[name] = entry
            cfg, vcfg = configs_from_record(record)
            prof = _profile_from_record(record)
            spec = RepairPipeline(cfg=cfg).spec
            ctx = context_from(arrays, record["replay_meta"])
            prefix = ([arrays["prefix"][i].copy()
                       for i in range(arrays["prefix"].shape[0])]
                      if arrays["prefix"].size else [])
            box_poses = arrays["box_poses"]
            diffs: list[str] = []
            arr_diffs: list[str] = []
            fresh_arrays: dict = {}
            if record["kind"] == "counterfactual":
                cfg_current, _ = configs_from_record(record,
                                                     key="repair_no_c2")
                ctx_plus = _replace(
                    freeze_context(ctx),
                    c2_hand_enabled=record["c2_enable_flags"],
                    c2_hand_reason=record["c2_reasons"],
                    prev_hand_scene_dist=record["distance_views"]["flags"])
                cur_rec, cur_arr = full_candidate_audit(
                    spec, cfg_current, vcfg, ctx, arrays["z_cand"],
                    prefix=prefix, box_poses=box_poses, label="current")
                arec, aarr = full_candidate_audit(
                    spec, cfg, vcfg, ctx_plus, arrays["z_cand"],
                    prefix=prefix, box_poses=box_poses,
                    label=_V1_AUDIT_LABEL, c2_profile=prof)
                classification = classify_branches(cur_rec, arec)
                fresh = {"audits": {"current": cur_rec,
                                    _V1_AUDIT_LABEL: arec},
                         "classification": classification}
                diffs = deep_diff({k: record.get(k) for k in fresh}, fresh)
                fresh_arrays = {**cur_arr, **aarr}
                entry["classification_recomputed"] = classification
                entry["classification_matches"] = \
                    classification == record.get("classification")
                prev_checks = _prev_scene_cross_check(record)
                entry["prev_scene_check"] = prev_checks or "ok"
            else:      # frozen_frame: the full escalation chain re-runs
                pipe = RepairPipeline(spec=spec, cfg=cfg, c2_profile=prof)
                pipe.validator.cfg = vcfg
                esc = escalate_frame(pipe, ctx, record["input"]["t"],
                                     ctx.z_pred, ctx.z_prev)
                esc_summary = {k: esc[k] for k in (
                    "skipped", "adopted", "adopted_source", "last_status",
                    "last_term_reason")}
                esc_summary["n_attempts"] = len(esc["attempts"])
                diffs = deep_diff(record.get("escalation"), esc_summary)
                if esc["adopted"]:
                    z = np.asarray(esc["adopted_z"], dtype=float)
                    if not np.array_equal(arrays["adopted_z"], z):
                        arr_diffs.append("adopted_z: values differ")
                    arec, aarr = full_candidate_audit(
                        spec, cfg, vcfg, ctx, z, prefix=prefix,
                        box_poses=box_poses, label="profile_adopted",
                        c2_profile=prof)
                    diffs += [f"audit.{d}"
                              for d in deep_diff(record.get("audit"), arec)]
                    fresh_arrays = aarr
                entry["escalation_matches"] = not diffs
            recomp = {k: arrays[k] for k in fresh_arrays if k in arrays}
            arr_diffs += arrays_diff(recomp, fresh_arrays)
            arr_diffs += [f"{k}: recomputed array absent from saved bundle"
                          for k in sorted(fresh_arrays) if k not in arrays]
            entry.update({
                "record_diff_paths": diffs[:40],
                "n_record_diffs": len(diffs),
                "array_diffs": arr_diffs[:40],
                "matches": not diffs and not arr_diffs,
            })
            entry["ok"] = (entry["matches"]
                           and entry.get("classification_matches", True)
                           and entry.get("prev_scene_check", "ok") == "ok")
            ok = ok and entry["ok"]
            print(f"  [replay {kind}] {name}: ok={entry['ok']} "
                  f"diffs={len(diffs)} arrays={len(arr_diffs)}")
        # the processed count is part of the verdict (an index/products
        # disagreement must never shrink the replay to "0 records MATCH")
        if len(entries) != len(idx["stems"]):
            ok = False
        results["kinds"][kind] = {"ok": ok, "n": len(entries),
                                  "expected": len(idx["stems"]),
                                  "entries": entries}
        results["ok"] = results["ok"] and ok
    (out_dir / "recovery_replay.json").write_text(
        json.dumps(results, indent=1, allow_nan=False, default=float))
    print(f"[replay] {'ALL MATCH' if results['ok'] else 'MISMATCH'} "
          f"({', '.join(f'{k}: {v['n']}/{v['expected']}' for k, v in results['kinds'].items())}) "
          f"-> {out_dir / 'recovery_replay.json'}")
    return results


def stage_norm(root: Path) -> None:
    """Plan section 7: S1 -> S2 (numeric found no winner; the branch is
    pre-registered).  Each S profile: five frozen frames, the 113 existing
    counterfactuals under the REAL norm rows (all ten spikes must be
    rejected), one full diagnosis6, and the full quality gates."""
    from ireset.repair.profiles import load_profiles
    from train24fix3_quality_gate import gate_report

    numeric = load_stage_summary(root, "numeric")
    if numeric.get("winner"):
        raise SystemExit(
            "stage norm is NOT to run: a numeric profile satisfied the "
            "development gates (plan section 6 exit -- norm is the branch "
            "for 'no numeric winner' only)")
    out = root / "norm"
    diag = F_ROOT / "diag"
    gate = Gate()
    profiles = load_profiles()
    winner = None
    comparisons: dict[str, dict] = {}
    not_executed: list[str] = []
    for name in ("S1", "S2"):
        if winner is not None:
            not_executed.append(name)
            continue
        prof = profiles[name]
        pdir = out / name
        ff = _profile_frozen_frames(prof, diag, pdir)
        cf = _norm_counterfactuals(prof, diag, pdir, gate)
        six = _run_sixseg(name, pdir / "sixseg" / "rq")
        qg = gate_report(Path(six["rq_dir"]))
        base_ok = (six["n_kept"] == six["n_demos"]
                   and six["code_changed_during_run"] is False)
        cand_ok = base_ok and qg["ok"] and cf["spikes_ok"] and cf["enable_ok"]
        comparisons[name] = {
            "profile": prof.as_record(),
            "frozen_frames": {k: ff[k] for k in (
                "rows", "n_adopted", "n_full_audit_pass",
                "n_adopted_audit_fail", "n_evidence_incomplete",
                "n_not_adopted")},
            "frozen_frame_products": str(pdir / "ff"),
            "counterfactuals": {k: cf[k] for k in
                                ("distribution", "n", "spikes_ok",
                                 "enable_ok")},
            "counterfactual_products": str(pdir / "cf"),
            "sixseg": {k: six[k] for k in
                       ("rq_dir", "wall_s", "n_demos", "n_kept", "kept",
                        "code_changed_during_run")},
            "quality_gate": {"verdicts": qg["verdicts"]},
            "base_gate_ok": base_ok, "quality_gate_ok": qg["ok"],
            "selected": bool(cand_ok)}
        write_guarded(pdir / "profile_summary.json", json.dumps(
            comparisons[name], indent=1, default=float))
        gate.check(f"{name}_base_gate", base_ok,
                   f"keep {six['n_kept']}/{six['n_demos']}, "
                   f"r9={six['code_changed_during_run']}")
        gate.check(f"{name}_quality_gate", qg["ok"], qg["verdicts"])
        if cand_ok:
            winner = name
    summary = {"stage": "norm", "ok": True,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "numeric_winner_was": numeric.get("winner"),
               "winner": winner, "not_executed": not_executed,
               "comparisons": comparisons, "checks": gate.checks,
               "next": ("sixseg (formal stage E, selected profile)"
                        if winner else
                        "algorithm candidate comparison ENDED (plan 7.3): "
                        "deliver the failure specification and a new "
                        "minimal problem spec; the next strategy change "
                        "needs a NEW pre-registered plan")}
    write_guarded(stage_summary_path(root, "norm"),
                  json.dumps(summary, indent=1, default=float))
    print(f"norm -> winner={winner} ({stage_summary_path(root, 'norm')})")


# ---------------------------------------------------------------------------
# later stages (plan sections 8/9) -- formal E/F through the subtask machine
# ---------------------------------------------------------------------------


def _formal_runs(reg: SubtaskRegistry, out: Path,
                 profile_name: str | None, set_name: str,
                 runs: list[tuple[str, dict]]) -> dict[str, dict]:
    """The formal stage's execution runs as §12 subtasks.  ``runs`` pairs a
    subtask name with the ``_run_sixseg`` kwargs (a ``profile_name`` entry
    overrides the stage's profile -- the ns control passes None); each
    attempt's processed count is the run's demo count and its outputs are
    the report + every kept NPZ (a partial write can never verify)."""
    expected = len(registered_set_of(set_name))
    identity = _stage_identity(profile_name, set_name)
    results: dict[str, dict] = {}
    for name, kwargs in runs:
        run_profile = kwargs.pop("profile_name", profile_name)

        def runner(att_dir: Path, _kw=kwargs, _prof=run_profile) -> int:
            info = _run_sixseg(_prof, att_dir, **_kw)
            return int(info["n_demos"])

        def outputs_of(att_dir: Path, _kw=kwargs) -> list[Path]:
            prefix = ("regen_report" if _kw.get("mode", "regen") == "regen"
                      else "standard_report")
            rep = att_dir / f"{prefix}_{set_name}_strict.json"
            doc = json.loads(rep.read_text())
            outs = [rep] + [att_dir / f"{r['demo']}.npz"
                            for r in doc["rows"] if r["accepted"]]
            missing = [str(p) for p in outs if not p.is_file()]
            if missing:
                raise RuntimeError(f"run outputs missing: {missing[:3]}")
            return outs

        results[name] = _subtask(reg, name, out, identity, expected,
                                 runner, outputs_of)
        print(f"  [subtask {name}] reused={results[name]['reused']} "
              f"processed={results[name]['processed']}/{expected}")
    return results


def registered_set_of(set_name: str) -> list[str]:
    """The FROZEN demo list a formal run must cover exactly (plan 8.2 --
    coverage is judged against this, never against what the report kept)."""
    from regenerate_teachers import registered_set

    demos = registered_set(set_name)
    if not demos:
        raise RuntimeError(f"registered set {set_name!r} is empty")
    return demos


def stage_sixseg(root: Path, profile: str | None = None) -> None:
    """Plan section 8: the FORMAL six-segment acceptance of the SELECTED
    profile -- two quiet runs + one verbose + the standard-mode control,
    all as §12 subtasks; triple consistency, labels, the FULL §8.1 gate
    (every row an explicit verdict, missing evidence a failure) and the
    strict replay of the selection's frozen evidence.  The verdict is
    machine-written only when everything completed; any missing subtask
    leaves the stage INCOMPLETE (never a pass from partial files)."""
    from ireset.repair.profiles import profiles_yaml_sha256
    from train24fix3_quality_gate import gate_report

    winner, selection = _selection_winner(root)
    if profile is not None and profile != winner:
        raise SystemExit(
            f"--profile {profile!r} != the selected profile {winner!r} -- "
            f"the formal stage runs the SELECTION only (plan 8.2: 错误选"
            f"定 profile 显式失败)")
    # cross-check against the selection's own per-profile record
    sel_path = root / ("norm" if selection.get("stage") == "norm"
                       else "numeric") / winner / "profile_summary.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        if sel.get("selected") is not True:
            raise SystemExit(
                f"{sel_path}: the winner's own record says selected="
                f"{sel.get('selected')!r} -- refusing to formalize")
    out = root / "sixseg"
    out.mkdir(parents=True, exist_ok=True)
    reg = SubtaskRegistry(out)
    gate = Gate()
    set_name = "diagnosis6"
    expected_demos = registered_set_of(set_name)
    expected_n = len(expected_demos)

    runs = _formal_runs(reg, out, winner, set_name, [
        ("rq1", {}), ("rq2", {}), ("rv", {"verbose": True}),
        ("ns", {"mode": "standard", "profile_name": None}),
    ])
    rq1_dir = Path(runs["rq1"]["dir"])
    rq2_dir = Path(runs["rq2"]["dir"])
    rv_dir = Path(runs["rv"]["dir"])
    ns_dir = Path(runs["ns"]["dir"])

    reports = []
    for name, d in (("rq1", rq1_dir), ("rq2", rq2_dir), ("rv", rv_dir)):
        reports.append(json.loads(
            (Path(runs[name]["dir"]) / "regen_report_diagnosis6_strict.json"
             ).read_text()))
    consistent, diffs = _reports_consistent(reports)
    kept = [r["demo"] for r in reports[0]["rows"] if r["accepted"]]
    npz_ok, npz_diffs = _npz_bytes_equal([rq1_dir, rq2_dir, rv_dir], kept)
    gate.check("triple_run_consistency", consistent, diffs[:8])
    gate.check("triple_npz_bytes_equal", npz_ok, npz_diffs[:8])

    # the standard-mode control: no C2 evidence at all + algorithm counts
    # equal to B's own ns reference (§8.2 item 3)
    ns_rep = json.loads(
        (ns_dir / "standard_report_diagnosis6_strict.json").read_text())
    ns_problems: list[str] = []
    if ns_rep.get("effective", {}).get("mode") != "standard":
        ns_problems.append(f"mode={ns_rep.get('effective', {}).get('mode')!r}")
    for r in ns_rep["rows"]:
        if r.get("c2_hand_speed"):
            ns_problems.append(f"{r['demo']}: C2 evidence in standard mode")
    b_ns = REFERENCES["B"]["root"] / "ns" / "standard_report_diagnosis6_strict.json"
    if b_ns.is_file():
        b_rep = json.loads(b_ns.read_text())
        b_ok, b_diffs = _reports_consistent([b_rep, ns_rep])
        ns_problems += [f"vs B/ns: {d}" for d in b_diffs]
    else:
        ns_problems.append("B/ns reference report missing")
    gate.check("ns_standard_control", not ns_problems, ns_problems[:8])

    # labels + the FULL §8.1 gate (evidence-requiring families mandatory)
    identity = _stage_identity(winner, set_name)
    labels = _subtask(
        reg, "labels", out, identity, expected_n,
        lambda att: (_run_labels(rq1_dir, att), expected_n)[1],
        lambda att: [att / "label_manifest.json"])
    gate.check("labels_subtask_complete", labels["ok"],
               {"reused": labels["reused"]})
    qg = _subtask(
        reg, "quality_gate", out, identity, 1,
        lambda att: (_write_gate(att, rq1_dir, winner, set_name,
                                 cf_summary=sel_path if sel_path.exists()
                                 else None,
                                 labels=Path(labels["dir"])
                                 / "label_manifest.json"), 1)[1],
        lambda att: [att / "quality_gate.json"])
    qg_doc = json.loads((Path(qg["dir"]) / "quality_gate.json").read_text())
    gate.check("quality_gate_full", qg_doc.get("ok") is True,
               qg_doc.get("verdicts"))

    # the selection's frozen evidence replays TWICE into independent dirs
    replay = _subtask(
        reg, "replay", out, identity, 2,
        lambda att: (replay_recovery(root / ("norm" if selection.get("stage")
                                             == "norm" else "numeric") / winner,
                                     att / "run1"),
                     replay_recovery(root / ("norm" if selection.get("stage")
                                             == "norm" else "numeric") / winner,
                                     att / "run2"), 2)[2],
        lambda att: [att / "run1" / "recovery_replay.json",
                     att / "run2" / "recovery_replay.json"])
    replay_ok = all(
        json.loads((Path(replay["dir"]) / f"run{i}" / "recovery_replay.json"
                    ).read_text())["ok"] for i in (1, 2))
    gate.check("selection_evidence_replay", replay_ok,
               {"reused": replay["reused"]})

    summary = {"stage": "sixseg", "ok": gate.ok,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "selected_profile": winner,
               "registry_sha256": profiles_yaml_sha256(),
               "expected_set": set_name,
               "expected_demos": sorted(expected_demos),
               "runs": {k: {kk: v[kk] for kk in ("reused", "dir", "processed",
                                                 "ok")}
                        for k, v in runs.items()},
               "labels": {k: labels[k] for k in ("reused", "dir", "ok")},
               "quality_gate": {"dir": qg["dir"], "ok": qg_doc.get("ok"),
                                "verdicts": qg_doc.get("verdicts")},
               "replay": {"dir": replay["dir"], "ok": replay_ok},
               "checks": gate.checks,
               "next": ("train24 (plan section 9)" if gate.ok else
                        "formal sixseg FAILED -- no train24, no freeze")}
    write_guarded(stage_summary_path(root, "sixseg"),
                  json.dumps(summary, indent=1, default=float))
    print(f"sixseg -> {'PASS' if gate.ok else 'FAIL'} "
          f"({stage_summary_path(root, 'sixseg')})")


def _write_gate(att_dir: Path, run_dir: Path, profile: str, set_name: str,
                *, cf_summary, labels) -> Path:
    """The FULL gate of one formal run, persisted as a subtask output."""
    from train24fix3_quality_gate import gate_report

    cf = (json.loads(Path(cf_summary).read_text())
          if cf_summary and Path(cf_summary).is_file() else None)
    if cf is None:
        raise RuntimeError(
            "the selection's profile_summary.json is REQUIRED evidence for "
            "the formal gate (mechanism family) -- missing is a failure")
    report = gate_report(run_dir, expected_demos=registered_set_of(set_name),
                         profile=profile, cf_summary=cf,
                         label_manifest=labels, full=True)
    att_dir.mkdir(parents=True, exist_ok=True)
    out = att_dir / "quality_gate.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=1, default=float))
    return out


def stage_train24(root: Path, profile: str | None = None) -> None:
    """Plan section 9: the frozen train24 rerun -- three complete clean runs
    of the calibration24_train_v1 list as §12 subtasks, labels + the FULL
    gate, the ≥18/24 quantity gate, per-subject usable windows and the T
    keeps' preservation.  Requires the formal sixseg PASS; the profile is
    locked from that stage's own summary."""
    from ireset.repair.profiles import profiles_yaml_sha256
    from train24fix3_quality_gate import gate_report

    sixseg = load_stage_summary(root, "sixseg")
    winner = sixseg.get("selected_profile")
    if not winner:
        raise SystemExit("the sixseg summary carries no selected profile -- "
                         "stage train24 refuses to start")
    if profile is not None and profile != winner:
        raise SystemExit(
            f"--profile {profile!r} != the frozen profile {winner!r} "
            f"(plan 9: the profile is locked before any train24 read)")
    out = root / "train24"
    out.mkdir(parents=True, exist_ok=True)
    reg = SubtaskRegistry(out)
    gate = Gate()
    set_name = "calibration24_train_v1"
    expected_demos = registered_set_of(set_name)
    expected_n = len(expected_demos)

    runs = _formal_runs(reg, out, winner, set_name, [
        ("run1", {"set_name": set_name}), ("run2", {"set_name": set_name}),
        ("run3", {"set_name": set_name})])
    run_dirs = [Path(runs[n]["dir"]) for n in ("run1", "run2", "run3")]
    reports = [json.loads((d / f"regen_report_{set_name}_strict.json"
                           ).read_text()) for d in run_dirs]
    consistent, diffs = _reports_consistent(reports)
    gate.check("triple_run_consistency", consistent, diffs[:8])
    kept = [r["demo"] for r in reports[0]["rows"] if r["accepted"]]
    npz_ok, npz_diffs = _npz_bytes_equal(run_dirs, kept)
    gate.check("triple_npz_bytes_equal", npz_ok, npz_diffs[:8])

    identity = _stage_identity(winner, set_name)
    labels = _subtask(
        reg, "labels", out, identity, expected_n,
        lambda att: (_run_labels(run_dirs[0], att), expected_n)[1],
        lambda att: [att / "label_manifest.json"])
    gate.check("labels_subtask_complete", labels["ok"],
               {"reused": labels["reused"]})
    # the mechanism family's cf evidence is the SELECTION's own profile
    # summary (norm/numeric -- the sixseg stage writes its own record but
    # never a second profile summary)
    sel_cf = None
    for base in ("norm", "numeric"):
        cand = root / base / winner / "profile_summary.json"
        if cand.is_file():
            sel_cf = cand
            break
    qg = _subtask(
        reg, "quality_gate", out, identity, 1,
        lambda att: (_write_gate(att, run_dirs[0], winner, set_name,
                                 cf_summary=sel_cf,
                                 labels=Path(labels["dir"])
                                 / "label_manifest.json"), 1)[1],
        lambda att: [att / "quality_gate.json"])
    qg_doc = json.loads((Path(qg["dir"]) / "quality_gate.json").read_text())
    gate.check("quality_gate_full", qg_doc.get("ok") is True,
               qg_doc.get("verdicts"))

    # quantity gate: >= 18/24 keep, kept demos all in the frozen list
    n_kept = len(kept)
    foreign = sorted(set(kept) - set(expected_demos))
    gate.check("train24_quantity_ge_18", n_kept >= 18 and not foreign,
               {"n_kept": n_kept, "foreign_kept": foreign[:5]})

    # every subject of the frozen list has at least one KEPT demo with at
    # least one usable 60-frame window (plan 9 item 4)
    from train24fix3_quality_gate import _usable_windows, metrics_of_run
    metrics = metrics_of_run(run_dirs[0])
    subjects = {d.split("_")[0] for d in expected_demos}
    per_subject: dict[str, int] = {}
    for demo in kept:
        seg = metrics.get(demo)
        if seg is not None:
            per_subject[demo.split("_")[0]] = per_subject.get(
                demo.split("_")[0], 0) + _usable_windows(seg)
    missing_subjects = sorted(s for s in subjects
                              if per_subject.get(s, 0) < 1)
    gate.check("train24_subjects_have_usable_windows",
               not missing_subjects,
               {"missing": missing_subjects,
                "windows": {k: per_subject[k] for k in sorted(per_subject)}})

    # the original T keeps stay strict+keep (plan 9 item 5): every T
    # reference demo inside this run's scope must still be kept
    t_dir = REFERENCES["T"]["root"]
    t_kept = {p.stem for p in Path(t_dir).glob("*.npz")}
    lost_t = sorted(d for d in t_kept
                    if d in set(expected_demos) and d not in kept)
    gate.check("train24_T_kept_preserved", not lost_t, lost_t[:6])

    summary = {"stage": "train24", "ok": gate.ok,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "frozen_profile": winner,
               "registry_sha256": profiles_yaml_sha256(),
               "expected_set": set_name,
               "expected_demos": sorted(expected_demos),
               "n_kept": n_kept, "kept": sorted(kept),
               "runs": {k: {kk: v[kk] for kk in ("reused", "dir", "processed",
                                                 "ok")}
                        for k, v in runs.items()},
               "quality_gate": {"dir": qg["dir"], "ok": qg_doc.get("ok"),
                                "verdicts": qg_doc.get("verdicts")},
               "per_subject_windows": per_subject,
               "checks": gate.checks,
               "next": ("production handoff (plan section 10)" if gate.ok
                        else "train24 FAILED -- no production handoff")}
    write_guarded(stage_summary_path(root, "train24"),
                  json.dumps(summary, indent=1, default=float))
    print(f"train24 -> {'PASS' if gate.ok else 'FAIL'} "
          f"({stage_summary_path(root, 'train24')})")


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
                    help="profile name for sixseg/train24 (cross-checked "
                         "against the SELECTION summary before anything "
                         "runs -- a mismatched name is an explicit failure)")
    args = ap.parse_args()
    if args.stage in ("sixseg", "train24") and not args.profile:
        raise SystemExit(f"--profile is required for --stage {args.stage}")
    root = Path(args.out).resolve()
    if args.stage in ("sixseg", "train24"):
        STAGES[args.stage](root, profile=args.profile)
    else:
        STAGES[args.stage](root)


if __name__ == "__main__":
    main()
