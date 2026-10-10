"""A2 closure entry: C0 evidence re-verification + C1 U0 stop-line registration.

Closure plan ``omniretarget_a2_closure_plan.md`` §3-§4/§9. This entry only
ORGANIZES evidence and decisions — it reuses the existing comparison entry
for the fresh comparison (never re-implements the quality gates), reuses
``a2_integrity`` for every integrity leg, and writes:

* ``baseline/snapshots/attempt_NNN/baseline_evidence.json`` — C0: plan-§1.1
  anchor hashes, run identity (start+end), fresh comparison report +
  bit-comparison vs the authoritative report, known-vs-new gap split,
  consumption-map checks. Attempt-scoped and NEVER overwritten (review fix
  1: the first round's fixed-path snapshot was rewritten by every later
  call, dangling the D0/G0 parent hash references);
* ``baseline/snapshots/attempt_NNN/evidence_files.json``    — the file
  inventory (paths, sizes, full SHA-256) of everything this closure reasons
  over, with sub12's teacher recorded as explicitly MISSING (no empty NPZ,
  no fake hash);
* ``decisions/attempt_NNN/a2_closure.{json,md}`` — C1: the stop-line record
  whose statuses stay separated (baseline verdict incomplete / training
  not_ready / registration_status ≠ source verdict).

Exit 0 means the REGISTRATION completed from consistent evidence — it is
never a teacher-source pass. Unexpected evidence damage exits non-zero with
the reasons on disk. ``--diagnosis``/``--recovery-report`` merge already
existing, hash-pinned D0/UR results; this entry never launches diagnosis,
recovery generation or training.

Run: uv run --locked python scripts/omniretarget_a2_closure.py \
       --freeze derived/experiments/omniretarget_first/20261009_13906c1_a1/freeze \
       --selection derived/experiments/omniretarget_first/20261010_cec0287_a2_reviewfix7/teacher_pilot/cmp/attempt_001/teacher_selection.json \
       --u0-dir derived/experiments/omniretarget_first/20261009_123bc6f_a2_reviewfix/teacher_pilot/u0 \
       --out derived/experiments/omniretarget_first/<run_id>
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ
from ireset.eval import a2_closure as a2c
from ireset.paths import PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record

#: closure plan §1.1 — the declared SHA-256 of the anchors this closure is
#: allowed to reason over (measured 2026-10-10; the closure re-hashes them)
PLAN_ANCHORS = {
    "authoritative_selection": "bc03d085535c462c70864fabf265d399b3a18838d36587138d3f2c67d2f3c683",
    "a1_frozen_lists": "395094a30d69158ad08bf5b576f07c541c052187d5bf242f1cc7fde05e353d9d",
    "a1_assets": "2f72c9ed0f1832b0c1d80e7d1f2e82382785c56ae83556a63a46ec6789c7ad79",
    "a1_pairing_source_boxes": "39fce035ab63569bd69a4ddf61e1e6cd4fb70d4d1b2a12abf7051bd55ce3d09d",
    "upstream_uv_lock": "0fe6c3679e32925371a95b87a1953c72baeaab7571e91a19ef2f53887c4073b0",
}

#: behavioral code of this round (evidence_files "behavior_code" section)
BEHAVIOR_CODE = [
    "scripts/omniretarget_a2_closure.py",
    "scripts/omniretarget_teacher_comparison.py",
    "scripts/generate_omniretarget_teachers.py",
    "scripts/diagnose_omniretarget_sub12.py",
    "scripts/_omniretarget_diagnostic_child.py",
    "src/ireset/eval/a2_closure.py",
    "src/ireset/eval/teacher_comparison.py",
    "src/ireset/data/a2_integrity.py",
    "src/ireset/data/omniretarget_sources.py",
    "src/ireset/paths.py",
    "tests/test_omniretarget_a2_closure.py",
    "docs/plans/omniretarget_a2_closure_plan.md",
    "docs/plans/omniretarget_first_training_plan.md",
]


def measure_repo_identity() -> dict:
    """Measured main-repo identity: rev, dirty state, untracked file hashes
    AND the tracked-content diff hash (review fix 10, 2026-10-10: the status
    line list only says a file is modified — the same "M file" line with two
    different contents compared identity-equal; the diff hash pins the
    CONTENT of the dirty tree)."""
    import hashlib as _hashlib
    def run(*a: str) -> str:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *a],
                              capture_output=True, text=True,
                              check=True).stdout
    status = run("status", "--porcelain", "-uall")
    untracked = sorted(line[3:] for line in status.splitlines()
                       if line.startswith("??"))
    untracked_hashes = {
        rel: (integ.sha256_file(PROJECT_ROOT / rel)
              if (PROJECT_ROOT / rel).is_file() else "missing")
        for rel in untracked}
    tracked_diff = run("diff", "HEAD")
    return {
        "git_rev": run("rev-parse", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "status_lines": status.strip().splitlines()[:20],
        "untracked_files": untracked_hashes,
        "tracked_diff_sha256": _hashlib.sha256(
            tracked_diff.encode()).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff.encode()),
    }


def run_fresh_comparison(freeze: Path, u0_root: Path, baseline_dir: Path,
                         timeout_s: int = 2400) -> Path:
    """Fresh comparison attempt via the EXISTING comparison entry (§3.1)."""
    baseline_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,
           str(PROJECT_ROOT / "scripts" / "omniretarget_teacher_comparison.py"),
           "--freeze", str(freeze), "--u0-dir", str(u0_root),
           "--out", str(baseline_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"fresh comparison failed ({proc.returncode}):\n"
                         f"{(proc.stderr or proc.stdout)[-4000:]}")
    attempts = sorted((baseline_dir / "cmp").glob("attempt_*/teacher_selection.json"))
    if not attempts:
        raise SystemExit("fresh comparison wrote no teacher_selection.json")
    return attempts[-1]


def _file_record(path: Path) -> dict:
    path = Path(path)
    return {"path": path_for_record(path), "bytes": path.stat().st_size,
            "sha256": integ.sha256_file(path)}


def evidence_inventory(freeze: Path, selection: Path, u0_root: Path,
                       fresh_report: Path, pilot: list[str],
                       data_path: Path) -> dict:
    """§3.3 file inventory: full hashes for small evidence files, path+hash
    references for the big raw inputs; the missing sub12 teacher is an
    EXPLICIT missing entry (never an empty NPZ or a zero hash)."""
    inv: dict[str, dict] = {}
    inv["authoritative_selection"] = _file_record(selection)
    inv["fresh_selection"] = _file_record(fresh_report)
    inv["a1_freeze"] = {p.name: _file_record(p)
                        for p in sorted(freeze.iterdir()) if p.is_file()}
    inv["upstream_env"] = {
        "uv_lock": _file_record(
            PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / "uv.lock")}
    attempts = integ.load_attempts(u0_root)
    u0: dict[str, dict] = {"attempts_index": _file_record(u0_root / integ.ATTEMPTS_INDEX)}
    for att in attempts:
        name = att["attempt"]
        u0[name] = {"manifest": _file_record(
            u0_root / name / "generation_manifest.json")}
        for demo in pilot:
            raw = u0_root / name / f"{demo}_original.npz"
            conv = u0_root / name / f"{demo}_original_converted.npz"
            entry = u0[name].setdefault(demo, {})
            if raw.is_file():
                entry["raw"] = _file_record(raw)
            if conv.is_file():
                entry["converted"] = _file_record(conv)
            if not entry:
                entry["missing"] = True
                entry["note"] = "no artifact for this demo in this attempt"
    u0["sub12_teacher"] = {"missing": True,
                           "note": "no OK U0 artifact in any attempt "
                                   "(both default runs failed at CVXPY "
                                   "infeasible; see generation manifests)"}
    inv["u0_evidence"] = u0
    inv["published_teachers"] = {d: _file_record(
        TEACHER_TRAJ_DIR / f"{d}_original.npz") for d in pilot}
    inv["source_pts"] = {d: _file_record(Path(data_path) / f"{d}.pt")
                         for d in pilot}
    inv["behavior_code"] = {rel: _file_record(PROJECT_ROOT / rel)
                            for rel in BEHAVIOR_CODE}
    return inv


def sub12_cost_basis(u0_root: Path, demo: str) -> dict:
    """The registered cost/robustness basis for stopping U0 expansion: the
    sub12 rows of every retained attempt manifest (statuses, walls, error
    tails) — read from the artifacts, not from prose."""
    rows = []
    for att in integ.load_attempts(u0_root):
        manifest = integ.load_attempt_manifest(u0_root, att["attempt"])
        row = integ.manifest_row_for(manifest, demo)
        if row is not None:
            rows.append({
                "attempt": att["attempt"], "status": row.get("status"),
                "wall_s": row.get("wall_s"),
                "exit_code": row.get("exit_code"),
                "error_tail": (row.get("error") or "")[-300:],
            })
    return {"failed_runs": rows,
            "note": "deterministic CVXPY infeasible under the default "
                    "upstream configuration (both retained attempts)"}


def load_merge_extra(path: Path | None) -> dict | None:
    """--diagnosis / --recovery-report merge: load + hash-pin an existing
    result file; this entry never launches the work itself. The status is
    read wherever the producer put it (diagnosis: ``verdict.status``;
    comparison report: top-level ``verdict`` string)."""
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"merge input missing: {p}")
    data = json.loads(p.read_text())
    status = "merged"
    if isinstance(data, dict):
        v = data.get("verdict")
        if isinstance(v, str):
            status = v
        elif isinstance(v, dict) and isinstance(v.get("status"), str):
            status = v["status"]
        elif isinstance(data.get("status"), str):
            status = data["status"]
    return {"status": status, "source": path_for_record(p),
            "sha256": integ.sha256_file(p)}


def verify_merge_chain(diagnosis_path: Path | None,
                       recovery_path: Path | None,
                       prereg_path: Path | None,
                       anchors: dict, pilot: list[str]) -> dict:
    """C2 merge evidence-chain verification (review fix 6, 2026-10-10).

    Registering a merged UR verdict used to record only the CURRENT file
    hashes — a report with ``admission.passed=false`` / incomplete evidence
    / an empty denominator still registered as long as the top-level verdict
    said ``no_quality_gain``. Now every merged piece is verified against the
    chain that produced it:

    * preregistration: the recovery report's ``ur_identity`` must pin THIS
      file's hash, and the file must still hash to the diagnosis hash it
      registered (the registered parent stays intact on disk);
    * the merged diagnosis is either the registered parent itself or a
      post-fix reconfirmation whose verdict/failing frame/ordinal/rho AGREE
      with the registered one;
    * the recovery report passes the structural report invariants on the UR
      metrics side, and a ``no_quality_gain``/``local_preferred`` verdict is
      accepted only with complete evidence, a passed admission and an empty
      missing list;
    * the preregistration's authoritative-selection anchor equals the
      anchor-measured baseline report.

    Any problem blocks the closure registration (the reasons land in the
    decision record and the exit code goes non-zero).
    """
    from diagnose_omniretarget_sub12 import verify_diagnosis_record
    problems: list[str] = []
    rec: dict = {"checked": []}

    diag_data: dict | None = None
    if diagnosis_path is not None:
        diag_data = json.loads(Path(diagnosis_path).read_text())
        rec["diagnosis_verification"] = verify_diagnosis_record(diag_data)
        rec["checked"].append("diagnosis")
        if not rec["diagnosis_verification"]["verified"]:
            problems += [f"diagnosis:{p}" for p in
                         rec["diagnosis_verification"]["problems"]]

    if recovery_path is not None:
        rec["checked"].append("recovery")
        report = json.loads(Path(recovery_path).read_text())
        prereg = None
        if prereg_path is None:
            problems.append("recovery:preregistration_not_provided")
        else:
            pp = Path(prereg_path).expanduser().resolve()
            if not pp.is_file():
                problems.append("recovery:preregistration_missing")
            else:
                prereg_sha = integ.sha256_file(pp)
                prereg = json.loads(pp.read_text())
                rec["preregistration"] = {"path": path_for_record(pp),
                                          "sha256": prereg_sha}
                if report.get("ur_identity", {}).get(
                        "preregistration_sha256") != prereg_sha:
                    problems.append("recovery:report_not_bound_to_prereg")
                parent = prereg.get("parent", {})
                raw = parent.get("diagnosis")
                dpath = None
                if raw:
                    dp = Path(raw).expanduser()
                    dpath = dp if dp.is_absolute() else PROJECT_ROOT / dp
                registered_intact = (
                    dpath is not None and dpath.is_file()
                    and integ.sha256_file(dpath)
                    == parent.get("diagnosis_sha256"))
                rec["registered_parent_intact"] = bool(registered_intact)
                if not registered_intact:
                    problems.append("recovery:registered_diagnosis_drifted")
                sel_anchor = anchors["anchors"]["authoritative_selection"][
                    "measured_sha256"]
                if parent.get("authoritative_selection_sha256") != sel_anchor:
                    problems.append("recovery:prereg_anchor_mismatch")
                # the merged diagnosis: the registered parent itself, or a
                # reconfirmation that agrees with it
                if diag_data is not None and registered_intact:
                    reg_diag = json.loads(dpath.read_text())
                    merged_sha = integ.sha256_file(Path(diagnosis_path))
                    rec["merged_is_registered_diagnosis"] = (
                        merged_sha == parent.get("diagnosis_sha256"))
                    if not rec["merged_is_registered_diagnosis"]:
                        reg_v, merged_v = (reg_diag.get("verdict", {}),
                                           diag_data.get("verdict", {}))
                        agree = (reg_v.get("status")
                                 == merged_v.get("status")
                                 == "supports_step03_candidate")
                        for key in ("failing_frame_run1", "ordinal_run1"):
                            if reg_v.get(key) != merged_v.get(key):
                                agree = False
                                problems.append(
                                    f"diagnosis:reconfirmation_{key}_differs")
                        r0, r1 = reg_v.get("rho_run1"), merged_v.get("rho_run1")
                        if not (isinstance(r0, (int, float))
                                and isinstance(r1, (int, float))
                                and abs(r0 - r1) <= 1e-6):
                            agree = False
                            problems.append(
                                "diagnosis:reconfirmation_rho_differs")
                        rec["reconfirmation_agrees"] = bool(agree)
                        if not agree and not any(
                                p.startswith("diagnosis:reconfirmation")
                                for p in problems):
                            problems.append(
                                "diagnosis:reconfirmation_verdict_differs")
        inv = a2c.check_report_invariants(report, pilot, u_side="UR")
        rec["report_invariants"] = inv
        if not inv["passed"]:
            problems += [f"recovery:{v}" for v in inv["violations"]]
        verdict = report.get("verdict")
        admission_passed = bool(report.get("admission", {}).get("passed"))
        complete = bool(report.get("verdict_record", {}).get(
            "complete_evidence"))
        missing = list(report.get("admission", {}).get("ur_coverage", {})
                       .get("missing", []))
        rec["recovery_report_facts"] = {
            "verdict": verdict, "admission_passed": admission_passed,
            "complete_evidence": complete, "missing": missing}
        if verdict in ("no_quality_gain", "local_preferred"):
            if not complete:
                problems.append(f"recovery:{verdict}_without_complete_evidence")
            if not admission_passed:
                problems.append(f"recovery:{verdict}_with_failed_admission")
            if missing:
                problems.append(f"recovery:{verdict}_with_missing_demos:"
                                f"{missing}")
    rec["problems"] = problems
    rec["passed"] = not problems
    return rec


def run_closure(args, comparison_runner=run_fresh_comparison) -> int:
    t0 = time.time()
    freeze = args.freeze.expanduser().resolve()
    selection = args.selection.expanduser().resolve()
    u0_root = args.u0_dir.expanduser().resolve()
    out_root = integ.guard_output_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    identity_start = measure_repo_identity()
    checkout = integ.verify_checkout_against_freeze(
        json.loads((freeze / "assets.json").read_text()))

    # ---- C0: plan anchors ---------------------------------------------
    anchors = a2c.verify_plan_anchors(
        {"authoritative_selection": selection,
         "a1_frozen_lists": freeze / "frozen_lists.json",
         "a1_assets": freeze / "assets.json",
         "a1_pairing_source_boxes": freeze / "pairing_source_boxes.npz",
         "upstream_uv_lock": PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / "uv.lock"},
        PLAN_ANCHORS, integ.sha256_file)
    authoritative = json.loads(selection.read_text())
    pilot = authoritative["pilot"]
    # the known-missing set is the BASELINE-REGISTERED one (§3.1: known
    # missing is limited to what the authoritative report registered) —
    # never a free-floating constant that could mask a newly missing demo
    baseline_missing = list(authoritative.get("admission", {})
                            .get("u0_coverage", {}).get("missing", []))

    # ---- C0: fresh comparison via the existing entry -------------------
    baseline_dir = out_root / "baseline"
    print(f"[closure] fresh comparison -> {path_for_record(baseline_dir)}")
    fresh_path = comparison_runner(freeze, u0_root, baseline_dir)
    fresh = json.loads(fresh_path.read_text())
    fresh_cmp = a2c.compare_fresh_to_authoritative(authoritative, fresh)
    gaps = a2c.split_gaps(fresh["verdict_record"]["evidence_gaps"],
                          known_missing=baseline_missing)
    invariants = a2c.check_report_invariants(fresh, pilot)
    coverage = integ.collect_u0_coverage(u0_root, pilot)
    attempts_index = integ.load_attempts(u0_root)
    # review fix 9: retention verified on the manifest ROWS — the attempt
    # scopes the index declares must still have their rows (deleting one of
    # sub12's two registered failure rows shrinks the cost basis)
    attempt_scopes = {a["attempt"]: list(a.get("scope") or [])
                      for a in attempts_index}
    manifest_rows = {}
    for att in attempt_scopes:
        try:
            manifest_rows[att] = integ.load_attempt_manifest(
                u0_root, att).get("rows", [])
        except Exception:  # noqa: BLE001 -- recorded as a consumption gap
            manifest_rows[att] = None
    consumption = a2c.check_consumption(
        attempts_index, coverage, pilot, known_missing=baseline_missing,
        attempt_scopes=attempt_scopes, manifest_rows=manifest_rows)

    identity_end = measure_repo_identity()
    identity_stable = (identity_start == identity_end)

    baseline_evidence = {
        "stage": "C0 baseline evidence (closure plan §3)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "plan_anchors": anchors,
        "identity": {"start": identity_start, "end": identity_end,
                     "stable_during_run": identity_stable},
        "upstream_checkout_vs_freeze": checkout,
        "fresh_comparison": {
            "report": path_for_record(fresh_path),
            "report_sha256": integ.sha256_file(fresh_path),
            "authoritative_sha256": anchors["anchors"][
                "authoritative_selection"]["measured_sha256"],
            "matches_authoritative": fresh_cmp["matches"],
            "differences": fresh_cmp["differences"],
            "excluded_fields": fresh_cmp["excluded_fields"],
            "admission_passed": fresh["admission"]["passed"],
        },
        "known_gaps": gaps["known"],
        "unexpected_gaps": gaps["unexpected"],
        "report_invariants": invariants,
        "consumption": consumption,
        "passed": (anchors["passed"] and fresh_cmp["matches"]
                   and not gaps["unexpected"] and invariants["passed"]
                   and consumption["passed"] and checkout["passed"]
                   and identity_stable),
    }
    # review fix 1: the C0 snapshot is attempt-scoped and IMMUTABLE — later
    # invocations never rewrite it, so a D0/G0/C2 parent hash stays resolvable
    # (the first round's fixed-path snapshot was overwritten by every later
    # call and the D0 parent reference 83287703… silently dangled)
    snapshot_dir = integ.allocate_attempt(out_root / "baseline" / "snapshots")
    baseline_path = snapshot_dir / "baseline_evidence.json"
    baseline_path.write_text(json.dumps(baseline_evidence, indent=2,
                                        ensure_ascii=False))

    data_path, data_path_recorded = consumed_data_path(u0_root)
    inv = evidence_inventory(freeze, selection, u0_root, fresh_path, pilot,
                             data_path)
    inv["consumed_data_path"] = {
        "pt_dir": path_for_record(data_path),
        "recorded_in_manifest": data_path_recorded,
        "note": "legacy manifests predate the round-4 data_path field; the "
                "canonical-dir fallback is the anchor the comparison's "
                "recorded_in_manifest=false legs reconciled against (§3.2)",
    }
    (snapshot_dir / "evidence_files.json").write_text(
        json.dumps(inv, indent=2, ensure_ascii=False))

    # ---- C2 merge evidence-chain verification (review fix 6) ------------
    merge_verification = None
    if args.diagnosis is not None or args.recovery_report is not None:
        if args.recovery_report is not None and args.preregistration is None:
            raise SystemExit("--recovery-report requires --preregistration "
                             "(the merged verdict is registered only against "
                             "its own preregistration chain)")
        merge_verification = verify_merge_chain(
            args.diagnosis, args.recovery_report, args.preregistration,
            anchors, pilot)

    # ---- C1: stop-line registration ------------------------------------
    quality = a2c.summarize_quality_evidence(authoritative)
    record = a2c.build_closure_record(
        baseline={"report_path": path_for_record(selection),
                  "report_sha256": anchors["anchors"][
                      "authoritative_selection"]["measured_sha256"],
                  "verdict": authoritative["verdict"],
                  "pilot": pilot,
                  "measurable": quality["measurable_demos"],
                  "missing": list(coverage["missing"])},
        verification={
            "baseline_evidence": path_for_record(baseline_path),
            "baseline_evidence_sha256": integ.sha256_file(baseline_path),
            "anchors_passed": anchors["passed"],
            "fresh_matches": fresh_cmp["matches"],
            "fresh_differences": fresh_cmp["differences"],
            "known_gaps": gaps["known"],
            "unexpected_gaps": gaps["unexpected"],
            "report_violations": invariants["violations"],
            "consumption_violations": consumption["violations"],
            "attempts_retained": consumption["attempts_retained"],
            "identity_stable": identity_stable,
            "checkout_passed": checkout["passed"],
        },
        quality_evidence=quality,
        cost_basis=sub12_cost_basis(u0_root, coverage["missing"][0]
                                    if coverage["missing"] else
                                    a2c.KNOWN_MISSING_DEFAULT[0]),
        diagnosis=load_merge_extra(args.diagnosis),
        recovery=load_merge_extra(args.recovery_report),
    )
    if merge_verification is not None:
        record["merge_verification"] = merge_verification
        if not merge_verification["passed"]:
            record["registration_status"]["complete"] = False
            record["registration_status"]["blocking_reasons"] += [
                f"merge_chain:{p}" for p in merge_verification["problems"]]
    decisions = out_root / "decisions"
    attempt_dir = integ.allocate_attempt(decisions)
    json_path = attempt_dir / "a2_closure.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    (attempt_dir / "a2_closure.md").write_text(a2c.render_markdown(record))
    integ.append_attempts_index(decisions, {
        "attempt": attempt_dir.name,
        "decision": "stop_u0_expansion",
        "baseline_verdict": record["baseline"]["verdict"],
        "registration_complete": record["registration_status"]["complete"],
        "report": path_for_record(json_path),
    })

    # review fix 11: costs accumulate per invocation (the fixed-path file
    # kept only the LATEST call's wall time)
    costs_path = out_root / "costs.json"
    if costs_path.is_file():
        prev = json.loads(costs_path.read_text())
        entries = (prev.get("invocations") if isinstance(prev, dict) else None)
        if entries is None and "stage_wall_s" in prev:
            entries = [prev]     # legacy single-shot shape
    else:
        entries = []
    entries = entries + [{"timestamp": _dt.datetime.now().isoformat(
        timespec="seconds"), "stage_wall_s": round(time.time() - t0, 1)}]
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))

    print(a2c.render_markdown(record))
    print(f"a2_closure -> {path_for_record(json_path)}")
    reasons = record["registration_status"]["blocking_reasons"]
    if not record["registration_status"]["complete"]:
        print(f"REGISTRATION BLOCKED: {reasons}", file=sys.stderr)
        return 2
    print("registration complete (exit 0 = registration bookkeeping, NOT a "
          "teacher-source pass; baseline_verdict=incomplete, training=not_ready)")
    return 0


def consumed_data_path(u0_root: Path) -> tuple[Path, bool]:
    """The PT dir the consumed attempts recorded (§3.2).

    Legacy manifests (pre round-4) never recorded ``data_path``; the
    canonical dir is the fallback and the caller labels it — its hashes
    prove the CURRENT canonical bytes equal the generation records, they
    are not a claim that the historical consumption path was recorded.
    """
    from ireset.paths import OMOMO_NEW_DIR
    attempts = integ.load_attempts(u0_root)
    if attempts:
        manifest = integ.load_attempt_manifest(u0_root, attempts[-1]["attempt"])
        recorded = manifest.get("data_path")
        if recorded:
            p = Path(recorded).expanduser()
            return (p if p.is_absolute() else PROJECT_ROOT / p), True
    return OMOMO_NEW_DIR, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True,
                    help="authoritative teacher_selection.json (plan §1.1)")
    ap.add_argument("--u0-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="closure run root (must resolve under derived/)")
    ap.add_argument("--diagnosis", type=Path, default=None,
                    help="optional: merge an existing D0 diagnosis result "
                         "(verified against the registration chain)")
    ap.add_argument("--recovery-report", type=Path, default=None,
                    help="optional: merge an existing UR result — requires "
                         "--preregistration; the chain is verified, not "
                         "just hash-recorded")
    ap.add_argument("--preregistration", type=Path, default=None,
                    help="UR preregistration.json backing --recovery-report")
    args = ap.parse_args()
    return run_closure(args)


if __name__ == "__main__":
    raise SystemExit(main())
