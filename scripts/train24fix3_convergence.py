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
    attach_integrity,
    classify_branches,
    configs_from_record,
    context_from,
    full_candidate_audit,
    integrity_manifest,
    load_index,
    load_products,
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
            for key in ("P", "q", "A", "lo", "hi"):
                arr[f"a{ai}_qp{qi}_{key}"] = np.asarray(qa[key], dtype=float)
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

def _profile_frozen_frames(prof, diag: Path) -> dict:
    """Five frozen frames under ONE profile: the full escalation; an
    adopted candidate additionally gets the complete C2-on audit (plan
    6.2 item 1 -- legal-solve / failure / timeout / insufficient evidence
    are listed separately, never merged)."""
    from ireset.repair.pipeline import RepairPipeline
    from train24fix2_audit import (
        configs_from_record,
        context_from,
        escalate_frame,
        full_candidate_audit,
        load_products,
    )

    idx = load_index(diag, "fail_bind2_index")
    rows = []
    audits = {}
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
        row = {"stem": stem, "demo": record["demo"], "t": t,
               "skipped": bool(esc["skipped"]),
               "adopted": bool(esc["adopted"]),
               "adopted_source": esc["adopted_source"],
               "last_status": esc["last_status"],
               "last_term_reason": esc["last_term_reason"],
               "n_attempts": len(esc["attempts"]),
               "wall_s": round(wall, 2)}
        if esc["adopted"]:
            prefix = ([arrays["prefix"][i].copy()
                       for i in range(arrays["prefix"].shape[0])]
                      if arrays["prefix"].size else [])
            arec, _aarr = full_candidate_audit(
                spec, cfg, vcfg, ctx, esc["adopted_z"],
                prefix=prefix, box_poses=arrays["box_poses"],
                label="profile_adopted")
            audits[stem] = arec
            row["adopted_audit_branch_pass"] = arec.get("branch_pass")
        rows.append(row)
        print(f"  [{prof.name}] {stem}: adopted={row['adopted']} "
              f"term={row['last_term_reason']} ({row['wall_s']}s)")
    return {"rows": rows,
            "n_adopted": sum(1 for r in rows if r["adopted"]),
            "audits": audits}


def _run_sixseg(profile_name: str, rq: Path) -> dict:
    """One full diagnosis6 quiet run under the profile (plan 6.2 item 2)."""
    import subprocess

    rq.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "--locked", "python",
           "scripts/regenerate_teachers.py",
           "--set", "diagnosis6", "--mode", "regen",
           "--validation-mode", "strict", "--profile", profile_name,
           "--assert-task-gate", "0.40",
           "--output-dir", str(rq),
           "--report", str(rq / "regen_report_diagnosis6_strict.json"),
           "--note", f"train24fix3 numeric candidate {profile_name}"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_thread_env())
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"sixseg subprocess failed for {profile_name}")
    report = json.loads(
        (rq / "regen_report_diagnosis6_strict.json").read_text())
    return {"rq_dir": str(rq), "wall_s": round(wall, 1),
            "n_demos": report["n_demos"], "n_kept": report["n_kept"],
            "kept": [r["demo"] for r in report["rows"] if r["accepted"]],
            "code_changed_during_run":
                report.get("end", {}).get("code_changed_during_run"),
            "argv": report.get("effective", {}).get("argv"),
            "profile": report.get("profile")}


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
        ff = _profile_frozen_frames(prof, diag)
        six = _run_sixseg(name, pdir / "sixseg" / "rq")
        qg = gate_report(Path(six["rq_dir"]))
        base_ok = (six["n_kept"] == six["n_demos"]
                   and six["code_changed_during_run"] is False)
        cand_ok = base_ok and qg["ok"]
        comparisons[name] = {
            "profile": prof.as_record(),
            "frozen_frames": {k: ff[k] for k in ("rows", "n_adopted")},
            "frozen_frame_adopted_audits": ff["audits"],
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
               "comparisons": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "frozen_frame_adopted_audits"}
                               for k, v in comparisons.items()},
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
#: new modes use a NEW format; the v2 forensic products stay v2-decoded)
NORM_FORMAT = "train24fix3_norm_products_v1"


def _write_norm_products(pdir: Path, stem: str, record: dict,
                         arrays: dict) -> None:
    """Versioned JSON + NPZ, refusing overwrite; the NPZ is hashed INTO the
    JSON after writing (self-hash), the effective configs + resolved
    profile + integrity manifest are embedded so a replay never reads the
    current YAML."""
    import numpy as np

    jpath, npath = pdir / f"cf_{stem}.json", pdir / f"cf_{stem}.npz"
    if jpath.exists() or npath.exists():
        raise FileExistsError(f"refusing to overwrite {jpath}")
    np.savez(npath, **arrays)
    record["npz_sha256"] = sha256_full(npath)
    jpath.write_text(json.dumps(record, indent=1, allow_nan=False,
                                default=float))


def load_norm_products(pdir: Path, stem: str) -> tuple[dict, dict]:
    """Load one v3 norm product: format checked, NPZ hash verified against
    the JSON's self-hash, integrity verified against the live tree (v3
    records REQUIRE the full effective fields -- no backfill)."""
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


def load_norm_index(pdir: Path) -> dict:
    """The norm stage's dual-view index under the G1 contract (non-empty,
    duplicate-free, records/stems set-equal, correct format)."""
    path = pdir / "cf" / "cf_index.json"
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
    norm rows (plan 7.3).  Only the plus-C2 branch is recomputed -- audits
    solve nothing, so the current (C2-off) branch is profile-independent
    and the frozen one stands.  The TEN spike candidates must ALL be
    rejected by the real norm constraint; historical certified candidates
    are classified as they come (no 13/90 distribution is required)."""
    from dataclasses import asdict as _asdict
    from dataclasses import replace as _replace

    from ireset.repair.pipeline import RepairPipeline
    from train24fix2_audit import freeze_context

    from ireset.repair.profiles import profiles_yaml_sha256

    idx = load_index(diag, "cert_trace_cf_index")
    out = pdir / "cf"
    rows = []
    spikes = []
    for stem in idx["stems"]:
        record, arrays = load_products(diag, stem)
        cfg, vcfg = configs_from_record(record)
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
        arec, aarr = full_candidate_audit(
            spec, cfg, vcfg, ctx_plus, arrays["z_cand"],
            prefix=prefix, box_poses=arrays["box_poses"],
            label="plus_c2_norm", c2_profile=prof)
        classification = classify_branches(record["audits"]["current"], arec)
        enabled_modes = {side: (v.get("mode") if v.get("enabled") else None)
                         for side, v in
                         (arec.get("endpoint", {}).get("c2_last_hand_speed")
                          or {}).items()}
        row = {"stem": stem,
               "classification": classification,
               "plus_branch_pass": arec.get("branch_pass"),
               "frozen_classification": record.get("classification"),
               "enabled_row_modes": enabled_modes,
               "violating": (arec.get("endpoint", {}).get("violating") or [])[:6]}
        rows.append(row)
        is_spike = "current_rq_adopted_spike" in stem
        if is_spike:
            spikes.append({"stem": stem,
                           "rejected": not arec.get("branch_pass"),
                           "mode": enabled_modes})
        prod = {"format": NORM_FORMAT, "stem": stem, "profile": prof.name,
                "profile_resolved": prof.as_record(),
                "registry_sha256": profiles_yaml_sha256(),
                "effective": {"repair": _asdict(cfg),
                              "validation": _asdict(vcfg)},
                "audits": {"plus_c2_norm": arec},
                "classification": classification,
                "frozen_classification": record.get("classification")}
        attach_integrity(prod, cfg, vcfg)
        _write_norm_products(out, stem, prod,
                             {f"rows__plus_c2_norm__{k}": v
                              for k, v in aarr.items()})
        print(f"  [{prof.name}] {stem}: class={classification} "
              f"pass={arec.get('branch_pass')}")
    # dual-view index (non-empty, unique, set-equal -- the G1 contract)
    write_guarded(out / "cf_index.json", json.dumps(
        {"format": NORM_FORMAT, "stems": [r["stem"] for r in rows],
         "records": [{"stem": r["stem"]} for r in rows]}, indent=1))
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
        ff = _profile_frozen_frames(prof, diag)
        cf = _norm_counterfactuals(prof, diag, pdir, gate)
        six = _run_sixseg(name, pdir / "sixseg" / "rq")
        qg = gate_report(Path(six["rq_dir"]))
        base_ok = (six["n_kept"] == six["n_demos"]
                   and six["code_changed_during_run"] is False)
        cand_ok = base_ok and qg["ok"] and cf["spikes_ok"] and cf["enable_ok"]
        comparisons[name] = {
            "profile": prof.as_record(),
            "frozen_frames": {k: ff[k] for k in ("rows", "n_adopted")},
            "counterfactuals": {k: cf[k] for k in
                                ("distribution", "n", "spikes_ok",
                                 "enable_ok")},
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
# later stages (plan sections 8/9)
# ---------------------------------------------------------------------------


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
