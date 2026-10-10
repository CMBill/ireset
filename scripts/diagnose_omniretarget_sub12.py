"""D0: sub12 limited fault diagnosis (closure plan §5) + G0 registration.

Parent runs in the MAIN environment: it verifies the inputs against the C0
baseline evidence, launches the wrapped-generation child inside the
independent upstream uv environment (argument list, never a shell line),
saves FULL stdout/stderr, and compares:

* control: ``sub16_largebox_000`` wrapped default generation must produce
  arrays bit-equal to the registered raw artifact — otherwise the wrapping
  is not equivalent and NO sub12 attribution is allowed (§5.3 step 1);
* two sub12 runs from frame 0 with the original default initialization:
  the ACTUAL failing ``frame_idx`` and in-frame solver ordinal are captured
  by the child (the historical 171/347 progress-bar number is never parsed
  back out of a log tail);
* Phase I (computed inside the failing child, on the original problem):
  rho = min ||dqa||2 over the original constraints minus the single
  step-size SOC; the §5.5 table decides whether the single 0.3 radius
  candidate may be registered (G0).

Subcommand ``register-recovery`` (G0, §6.2): writes the exclusive-create
``preregistration.json`` for ``a2_recovery_step03_v1`` — allowed ONLY when
the referenced diagnosis summary has status ``supports_step03_candidate``
and no UR products exist without prior registration.

Run: uv run --locked python scripts/diagnose_omniretarget_sub12.py \
       --baseline-evidence <run>/baseline/baseline_evidence.json \
       --out <run>/diagnosis
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from _omniretarget_diagnostic_child import classify_phase_one
from ireset.data import a2_integrity as integ
from ireset.paths import (OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR,
                          path_for_record)

HOLOSOMA_PACKAGE = (
    PROJECT_ROOT / "third_party" / "holosoma" / "src" / "holosoma_retargeting"
    / "holosoma_retargeting"
)
DEFAULT_ENV_PYTHON = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"
CHILD_DRIVER = PROJECT_ROOT / "scripts" / "_omniretarget_diagnostic_child.py"
RETARGETER_FILE = HOLOSOMA_PACKAGE / "src" / "interaction_mesh_retargeter.py"

#: §5.5 two-run consistency: identical key arrays + rho agreement
RHO_AGREEMENT = 1e-6

#: Required snapshot evidence (round-3 review fix 3, 2026-10-10): a §5.5
#: verdict needs ALL of these present on BOTH runs before any equality
#: counts — a key missing from both sides used to read as ``both_missing``
#: == equal, so deleting ``q_a_indices`` / the canonical objective ``c`` /
#: the P and A dumps from BOTH snapshots still yielded
#: ``supports_step03_candidate`` over absent evidence. ``q_a_nominal``
#: stays OPTIONAL (first-round snapshots predate it); pre-fix snapshots
#: whose canonical dump only saved ``b`` are historical defect evidence and
#: fail the required set by design.
REQUIRED_CTX_ARRAYS = ("q_locked", "q_a_n_last", "q_t_last",
                       "target_laplacian", "obj_pts_local", "q_a_indices")
OPTIONAL_CTX_ARRAYS = ("q_a_nominal",)
REQUIRED_CANONICAL_KEYS = ("b", "c")
REQUIRED_ARTIFACTS = ("failure_context.npz", "failure_canonical.npz",
                      "failure_canonical_P.npz", "failure_canonical_A.npz")


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _recorded_path(recorded: str | Path) -> Path:
    """evidence_files records repo-relative paths inside the project and
    absolute paths outside; undo exactly that."""
    p = Path(recorded).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _hash_or_none(path: Path) -> str | None:
    return integ.sha256_file(path) if Path(path).is_file() else None


def compare_control(control_raw: Path, registered_raw: Path,
                    registered_sha: str | None) -> dict:
    """§5.3 step 1: wrapped generation must reproduce the registered raw."""
    rec: dict = {"control_raw": path_for_record(control_raw),
                 "registered_raw": path_for_record(registered_raw)}
    if registered_sha is not None:
        rec["registered_sha_matches"] = (
            _hash_or_none(registered_raw) == registered_sha)
        if not rec["registered_sha_matches"]:
            rec.update(equal=False, reason="registered_raw_drifted")
            return rec
    if not control_raw.is_file():
        rec.update(equal=False, reason="control_raw_missing")
        return rec
    a = np.load(control_raw)
    b = np.load(registered_raw)
    try:
        equal = (np.array_equal(a["qpos"], b["qpos"])
                 and np.array_equal(a["human_joints"], b["human_joints"])
                 and float(a["fps"]) == float(b["fps"]))
    except KeyError as exc:
        rec.update(equal=False, reason=f"missing_key:{exc}")
        return rec
    rec.update(equal=bool(equal))
    return rec


def _arrays_equal(npz_a: Path, npz_b: Path, keys: list[str]) -> dict:
    """Two-run npz comparison over ``keys``.

    A key missing on BOTH sides is consistent evidence (``both_missing``,
    counted as equal) — this is only legitimate for OPTIONAL keys
    (``q_a_nominal``); REQUIRED keys are existence-checked by
    ``decide_diagnosis`` BEFORE this comparison runs, so a required key
    absent from both snapshots never reaches the ``both_missing == equal``
    branch. Missing on ONE side is always a difference.
    """
    out: dict = {}
    if not (npz_a.is_file() and npz_b.is_file()):
        return {"all_equal": False, "missing": True}
    a, b = np.load(npz_a), np.load(npz_b)
    for k in keys:
        in_a, in_b = k in a, k in b
        if in_a and in_b:
            out[k] = bool(np.array_equal(a[k], b[k]))
        elif in_a or in_b:
            out[k] = False
        else:
            out[k] = "both_missing"
    return {"all_equal": all(v is True or v == "both_missing"
                             for v in out.values()), "per_key": out}


def _snapshot_artifact_hashes(snapshot: Path) -> dict:
    """File-level SHA-256 of the deterministic failure artifacts (review fix
    5): the context npz and the canonical Clarabel dumps. failure_meta.json
    is excluded — it embeds solve timings, so two healthy runs never hash
    it identically; its deterministic fields are compared separately."""
    out: dict = {}
    for name in ("failure_context.npz", "failure_canonical.npz",
                 "failure_canonical_P.npz", "failure_canonical_A.npz"):
        p = Path(snapshot) / name
        out[name] = _hash_or_none(p)
    return out


def _compare_canonical(run_a: Path, run_b: Path) -> dict:
    """Compare the canonical Clarabel objective/rhs vectors of the two runs
    (``c`` for conic data; first-round dumps only saved ``b`` — the
    objective-serializer defect — and stay comparable on the keys they have)."""
    pa, pb = Path(run_a) / "failure_canonical.npz", Path(run_b) / "failure_canonical.npz"
    if not (pa.is_file() and pb.is_file()):
        return {"equal": False, "missing": True}
    a, b = np.load(pa), np.load(pb)
    out: dict = {}
    for k in ("c", "q", "b"):
        in_a, in_b = k in a, k in b
        if in_a and in_b:
            out[k] = bool(np.array_equal(a[k], b[k]))
        elif in_a or in_b:
            out[k] = False
        else:
            out[k] = "both_missing"
    return {"equal": all(v is True or v == "both_missing"
                         for v in out.values()), "per_key": out}


def decide_diagnosis(control: dict, runs: list[dict]) -> dict:
    """Aggregate the §5.5 verdict. Pure; counterexample-testable.

    Review fix 5 (2026-10-10): two-run consistency is verified on the FULL
    captured evidence — the five context arrays AND ``q_a_indices`` /
    ``q_a_nominal`` (variable-column mapping), the canonical Clarabel
    objective/rhs vectors, the file-level SHA-256 of the deterministic
    snapshot artifacts, and the problem shape (SOC t, dqa shape, kept
    constraint count). Editing the second run's canonical ``b`` or column
    mapping previously still yielded ``supports_step03_candidate``.

    Round-3 review fix 3 (2026-10-10): the REQUIRED evidence (context
    arrays incl. ``q_a_indices``, canonical ``b`` AND ``c``, the P/A dumps)
    must EXIST on both runs before any equality counts — ``both_missing``
    used to read as equal, so deleting required keys from BOTH snapshots
    still produced ``supports_step03_candidate`` over absent evidence.
    """
    if not control.get("equal", False):
        return {"status": "control_not_equivalent",
                "reason": control.get("reason"),
                "note": "wrapping not proven equivalent; no sub12 "
                        "attribution allowed (§5.3 step 1)"}
    failed, problems = [], []
    for run in runs:
        s = run.get("summary") or {}
        if run.get("timed_out"):
            problems.append(f"{run['label']}:timeout")
            continue
        if run.get("launch_error"):
            problems.append(f"{run['label']}:launch_failed")
            continue
        if not s:
            problems.append(f"{run['label']}:no_child_summary")
            continue
        failure = s.get("failure")
        if not failure:
            if "instrumentation" in str(s.get("outcome")):
                problems.append(f"{run['label']}:instrumentation_not_ready")
            else:
                problems.append(f"{run['label']}:unexpected_success")
            continue
        failed.append(run)
    if problems:
        return {"status": "diagnostic_incomplete",
                "problems": problems,
                "note": "a run neither failed at the original solver nor "
                        "produced usable evidence (§11 stop)"}
    # both runs failed with captured context: required evidence FIRST
    # (round-3 review fix 3) — equality over absent evidence is not
    # evidence; a both-missing key is only legal for the OPTIONAL set
    snap0, snap1 = Path(failed[0]["snapshot"]), Path(failed[1]["snapshot"])
    missing_required: list[str] = []
    for run, snap in zip(failed, (snap0, snap1)):
        for name in REQUIRED_ARTIFACTS:
            if not (snap / name).is_file():
                missing_required.append(f"{run['label']}:{name}")
        if (snap / "failure_context.npz").is_file():
            with np.load(snap / "failure_context.npz") as z:
                for k in REQUIRED_CTX_ARRAYS:
                    if k not in z.files:
                        missing_required.append(
                            f"{run['label']}:failure_context:{k}")
        if (snap / "failure_canonical.npz").is_file():
            with np.load(snap / "failure_canonical.npz") as z:
                for k in REQUIRED_CANONICAL_KEYS:
                    if k not in z.files:
                        missing_required.append(
                            f"{run['label']}:failure_canonical:{k}")
    if missing_required:
        return {"status": "diagnostic_incomplete",
                "problems": [f"missing_required_evidence:{m}"
                             for m in missing_required],
                "note": "a §5.5 verdict requires the full captured evidence "
                        "on both runs; a key missing from BOTH sides is "
                        "absent evidence, not consistent evidence"}
    # determinism first
    f0, f1 = (r["summary"]["failure"] for r in failed)
    same_frame = f0.get("failing_frame_idx", f0.get("frame_idx")) == \
        f1.get("failing_frame_idx", f1.get("frame_idx"))
    same_ordinal = f0.get("ordinal_in_frame") == f1.get("ordinal_in_frame")
    ctx_equal = _arrays_equal(
        snap0 / "failure_context.npz", snap1 / "failure_context.npz",
        list(REQUIRED_CTX_ARRAYS) + list(OPTIONAL_CTX_ARRAYS))
    canonical_equal = _compare_canonical(snap0, snap1)
    h0, h1 = _snapshot_artifact_hashes(snap0), _snapshot_artifact_hashes(snap1)
    hashes_equal = all(
        (h0[k] is None and h1[k] is None) or h0[k] == h1[k] for k in h0)
    p0 = f0.get("phase_one", {}) or {}
    p1 = f1.get("phase_one", {}) or {}
    cats = [p.get("category") or f"phase1_{p.get('status', 'missing')}"
            for p in (p0, p1)]
    rho0, rho1 = p0.get("rho"), p1.get("rho")
    rho_agree = (rho0 is not None and rho1 is not None
                 and abs(rho0 - rho1) <= RHO_AGREEMENT)
    shape_keys = ("soc_t", "dqa_shape", "n_kept_constraints",
                  "n_soc_constraints", "soc_t_matches_step_size")
    shape_equal = all(p0.get(k) == p1.get(k) for k in shape_keys)
    rec: dict = {
        "failing_frame_run1": f0.get("failing_frame_idx", f0.get("frame_idx")),
        "failing_frame_run2": f1.get("failing_frame_idx", f1.get("frame_idx")),
        "ordinal_run1": f0.get("ordinal_in_frame"),
        "ordinal_run2": f1.get("ordinal_in_frame"),
        "same_frame": bool(same_frame), "same_ordinal": bool(same_ordinal),
        "key_arrays_equal": ctx_equal,
        "canonical_vectors_equal": canonical_equal,
        "artifact_sha256": {"run1": h0, "run2": h1,
                            "equal": bool(hashes_equal)},
        "problem_shape": {k: [p0.get(k), p1.get(k)] for k in shape_keys},
        "problem_shape_equal": bool(shape_equal),
        "phase1_categories": cats,
        "rho_run1": rho0, "rho_run2": rho1, "rho_agreement": bool(rho_agree),
    }
    consistent = (same_frame and same_ordinal and ctx_equal.get("all_equal")
                  and canonical_equal.get("equal") and hashes_equal
                  and shape_equal)
    if not consistent:
        rec.update(status="nondeterministic",
                   note="two runs disagree on the failure context or its "
                        "captured evidence; no recovery candidate "
                        "(§5.3 step 3)")
        return rec
    if all(c == "supports_step03_candidate" for c in cats) and rho_agree:
        rec.update(status="supports_step03_candidate",
                   note="both failing linearizations need a step larger "
                        "than 0.2 and within 0.3; register the single "
                        "step03 candidate before reading any new labels")
        return rec
    if all(c == "supports_step03_candidate" for c in cats) and not rho_agree:
        rec.update(status="nondeterministic",
                   note="identical failure contexts but Phase I rho "
                        f"disagrees beyond {RHO_AGREEMENT}; no candidate "
                        "(§5.5 two-run consistency)")
        return rec
    # terminal non-candidate rows of the §5.5 table
    if "rho_exceeds_candidate_range" in cats:
        terminal = "rho_exceeds_candidate_range"
    elif "retained_infeasible" in cats:
        terminal = "retained_infeasible"
    elif "rho_within_original_radius" in cats:
        terminal = "rho_within_original_radius"
    else:
        terminal = "diagnostic_incomplete"
    rec.update(status=f"no_recovery_candidate:{terminal}",
               note="finite-recovery hypothesis not supported; U0 stop line "
                    "stands (§6.1)")
    return rec


def verify_diagnosis_record(diag: dict) -> dict:
    """Verify a diagnosis_run.json AS merge/registration evidence (review
    fixes 5/6, 2026-10-10; round-3 review fix 2): the recorded verdict must
    be backed by the recorded facts, the parent baseline-evidence hash it
    registered against must still resolve to the file on disk, and — new —
    the verdict must be RE-DERIVABLE from the run records plus the snapshot
    artifacts ON DISK. The recorded pass flags alone prove nothing: editing
    run 2's canonical ``b`` while keeping the recorded hashes previously
    passed both G0 and C2, because nothing re-read the artifacts.

    The re-derivation rebuilds the two run records (label / snapshot /
    failure) and re-runs :func:`decide_diagnosis` — which re-hashes the
    snapshot files, re-compares the captured arrays, re-checks the required
    evidence and re-dispatches the §5.5 table — and additionally compares
    the RECORDED per-run artifact hashes against the files on disk now.
    Used by G0 ``register-recovery`` and by the closure C2 merge
    verification.
    """
    problems: list[str] = []
    verdict = diag.get("verdict", {})
    status = verdict.get("status")
    control = (diag.get("control") or {}).get("comparison") or {}
    if not control.get("equal"):
        problems.append("control_not_equivalent")
    parent = diag.get("baseline") or {}
    parent_path = parent.get("baseline_evidence")
    if parent_path:
        pp = Path(parent_path)
        pp = pp if pp.is_absolute() else PROJECT_ROOT / pp
        if not pp.is_file():
            problems.append("parent_baseline_missing")
        elif integ.sha256_file(pp) != parent.get("baseline_evidence_sha256"):
            problems.append("parent_baseline_drifted")
    else:
        problems.append("parent_baseline_not_recorded")
    rederived_info: dict = {}
    if status == "supports_step03_candidate":
        for key, val in (("same_frame", True), ("same_ordinal", True),
                         ("rho_agreement", True)):
            if verdict.get(key) is not val:
                problems.append(f"{key}_not_holds")
        if not (verdict.get("key_arrays_equal") or {}).get("all_equal"):
            problems.append("key_arrays_not_equal")
        if not (verdict.get("canonical_vectors_equal") or {}).get("equal"):
            problems.append("canonical_vectors_missing_or_unequal")
        if not (verdict.get("artifact_sha256") or {}).get("equal"):
            problems.append("artifact_sha256_missing_or_unequal")
        if not verdict.get("problem_shape_equal"):
            problems.append("problem_shape_missing_or_unequal")
        cats = verdict.get("phase1_categories") or []
        if not cats or not all(c == "supports_step03_candidate" for c in cats):
            problems.append("phase1_categories_not_supporting")
        rho = verdict.get("rho_run1")
        if not (isinstance(rho, (int, float))
                and 0.2 + 1e-6 < rho <= 0.3 - 1e-6):
            problems.append("rho_outside_candidate_window")
        # round-3 review fix 2: re-derive from the run records + the
        # snapshot artifacts on disk — the flags above are claims, not
        # evidence
        runs = diag.get("runs") or []
        if len(runs) != 2:
            problems.append("runs_not_recorded_for_reverification")
        elif any(not r.get("snapshot") for r in runs):
            problems.append("snapshot_not_recorded_for_reverification")
        else:
            rebuilt = [{"label": r.get("label"),
                        "timed_out": r.get("timed_out"),
                        "launch_error": r.get("launch_error"),
                        "snapshot": r.get("snapshot"),
                        "summary": {"failure": r.get("failure"),
                                    "outcome": r.get("outcome")}}
                       for r in runs]
            rederived = decide_diagnosis(control, rebuilt)
            rederived_info = {
                "status": rederived.get("status"),
                "failing_frame_run1": rederived.get("failing_frame_run1"),
                "ordinal_run1": rederived.get("ordinal_run1"),
                "rho_run1": rederived.get("rho_run1"),
            }
            if rederived.get("status") != status:
                problems.append("status_not_reproducible_from_artifacts")
            for key in ("failing_frame_run1", "ordinal_run1"):
                if rederived.get(key) != verdict.get(key):
                    problems.append(f"{key}_not_reproducible")
            r0, r1 = rederived.get("rho_run1"), verdict.get("rho_run1")
            if not (isinstance(r0, (int, float))
                    and isinstance(r1, (int, float))
                    and abs(r0 - r1) <= RHO_AGREEMENT):
                problems.append("rho_not_reproducible")
            # the recorded per-run artifact hashes vs the files on disk NOW
            rec_hashes = verdict.get("artifact_sha256") or {}
            for side, run in zip(("run1", "run2"), runs):
                now = _snapshot_artifact_hashes(
                    _recorded_path(run["snapshot"]))
                if rec_hashes.get(side) != now:
                    problems.append(
                        f"artifact_hashes_drifted_vs_disk:{side}")
    elif status == "not_run":
        problems.append("diagnosis_not_run")
    return {"verified": not problems, "problems": problems,
            "status": status,
            "rederived": rederived_info,
            "facts": {k: verdict.get(k) for k in
                      ("failing_frame_run1", "ordinal_run1", "rho_run1",
                       "rho_run2")}}


def run_child(env_python: Path, demo: str, data_path: Path, save_dir: Path,
              snapshot_dir: Path, retargeter_sha: str,
              timeout_s: int) -> dict:
    """Launch the wrapped generation child; FULL logs; group-kill timeout.

    The child runs in its own session (``start_new_session``), so a timeout
    kills the WHOLE process group and the partially-captured stdout/stderr
    is still saved — no orphan solver survives into a later attempt.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(env_python), str(CHILD_DRIVER),
           "--mode", "generation", "--demo", demo,
           "--data-path", str(data_path),
           "--save-dir", str(save_dir),
           "--snapshot-dir", str(snapshot_dir),
           "--holosoma-package", str(HOLOSOMA_PACKAGE),
           "--retargeter-sha256", retargeter_sha]
    t0 = time.time()
    rec: dict = {"label": demo, "argv": cmd, "snapshot": str(snapshot_dir),
                 "save_dir": str(save_dir), "timeout_s": timeout_s}
    try:
        proc = subprocess.Popen(cmd, cwd=str(HOLOSOMA_PACKAGE),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
    except OSError as exc:
        rec.update(launch_error=f"{type(exc).__name__}: {exc}",
                   timed_out=False)
        return rec
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # child leads its group
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
    (snapshot_dir / "child.stdout.txt").write_text(out or "")
    (snapshot_dir / "child.stderr.txt").write_text(err or "")
    rec.update(exit_code=None if timed_out else proc.returncode,
               wall_s=round(time.time() - t0, 1), timed_out=timed_out)
    if timed_out:
        (snapshot_dir / "TIMEOUT.txt").write_text(
            f"timeout after {timeout_s}s; process group killed; partial "
            f"logs saved\n")
        return rec
    summary_path = snapshot_dir / "child_summary.json"
    rec["summary"] = _load_json(summary_path) if summary_path.is_file() else None
    return rec


def diagnose(args) -> int:
    t0 = time.time()
    out_root = integ.guard_output_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- baseline evidence + anchors -----------------------------------
    baseline_evidence_path = Path(args.baseline_evidence).expanduser().resolve()
    if not baseline_evidence_path.is_file():
        raise SystemExit(f"baseline evidence missing: {baseline_evidence_path}")
    baseline = _load_json(baseline_evidence_path)
    evidence_files_path = baseline_evidence_path.parent / "evidence_files.json"
    if not evidence_files_path.is_file():
        raise SystemExit(f"evidence inventory missing: {evidence_files_path}")
    ev = _load_json(evidence_files_path)
    selection_path = _recorded_path(ev["authoritative_selection"]["path"])
    selection_now = _hash_or_none(selection_path)
    selection_ok = (selection_now == ev["authoritative_selection"]["sha256"]
                    == baseline["plan_anchors"]["anchors"][
                        "authoritative_selection"]["measured_sha256"])
    freeze_dir = _recorded_path(
        ev["a1_freeze"]["frozen_lists.json"]["path"]).parent

    # ---- FULL pre-admission (review fix 4, 2026-10-10) ------------------
    # a failed C0, drifted assets, missing/tampered inputs at the ACTUAL
    # --data-path, or an env that drifted from what the consumed attempts
    # recorded must block BEFORE any child process is launched — the first
    # round only checked selection/checkout/env-probe and ran three
    # subprocesses against evidence it had never re-admitted.
    blocked: list[str] = []
    pre: dict = {}
    if not selection_ok:
        blocked.append("authoritative_selection_drifted")
    if baseline.get("passed") is not True:
        blocked.append("baseline_c0_not_passed")
    if not baseline.get("identity", {}).get("stable_during_run", False):
        blocked.append("baseline_identity_unstable")
    checkout = integ.verify_checkout_against_freeze(
        _load_json(freeze_dir / "assets.json"))
    pre["checkout"] = {"passed": checkout["passed"]}
    if not checkout["passed"]:
        blocked.append("upstream_checkout_mismatch")
    try:
        assets_check = integ.verify_assets_against_freeze(
            _load_json(freeze_dir / "assets.json"))
        pre["assets"] = {"passed": assets_check["passed"]}
        if not assets_check["passed"]:
            blocked.append("frozen_assets_drifted")
    except Exception as exc:  # noqa: BLE001 -- recorded, blocked
        pre["assets"] = {"error": f"{type(exc).__name__}: {exc}"}
        blocked.append("frozen_assets_check_failed")
    env = integ.measure_env_identity(args.env_python)
    env_ok = (env["env_mujoco_version"] is not None
              and env["env_mujoco_version"] == env["env_lock_mujoco_version"])
    pre["env_probe"] = env
    if not env_ok:
        blocked.append(f"env_probe_failed:{env.get('env_probe_error')
                                             or env['env_mujoco_version']}")
    data_path = Path(args.data_path).expanduser().resolve()
    scope_demos = sorted({args.control_demo, args.demo})
    try:
        frozen_entries = [e for e in _load_json(
            freeze_dir / "frozen_lists.json")["teacher_pilot_train6_v1"]
            if e["demo"] in scope_demos]
        inputs_check = integ.verify_inputs_against_freeze(
            frozen_entries, TEACHER_TRAJ_DIR, data_path)
        pre["inputs_vs_freeze"] = inputs_check
        bad = [d for d, r in inputs_check.items()
               if not (r.get("npz_matches") and r.get("pt_matches"))]
        if bad:
            blocked.append(f"inputs_vs_freeze_failed:{bad}")
    except Exception as exc:  # noqa: BLE001 -- recorded, blocked
        pre["inputs_vs_freeze"] = {"error": f"{type(exc).__name__}: {exc}"}
        blocked.append(f"inputs_vs_freeze_error:{type(exc).__name__}")
    # the env the consumed U0 attempts recorded vs the env measured now
    try:
        u0_root = _recorded_path(
            ev["u0_evidence"]["attempts_index"]["path"]).parent
        recorded_envs = []
        for att in integ.load_attempts(u0_root):
            ident = integ.load_attempt_manifest(
                u0_root, att["attempt"]).get("identity", {}).get("env", {})
            recorded_envs.append(ident)
        pre["env_recorded_at_generation"] = recorded_envs
        lock_shas = {e.get("env_uv_lock_sha256") for e in recorded_envs}
        versions = {e.get("env_mujoco_version") for e in recorded_envs}
        env_consistent = (
            len(lock_shas) == 1 and None not in lock_shas
            and env["env_uv_lock_sha256"] in lock_shas
            and len(versions) == 1 and None not in versions
            and env["env_mujoco_version"] in versions)
        if not env_consistent:
            blocked.append("env_drifted_vs_generation")
    except Exception as exc:  # noqa: BLE001 -- recorded, blocked
        pre["env_recorded_at_generation"] = {
            "error": f"{type(exc).__name__}: {exc}"}
        blocked.append("generation_env_record_unavailable")

    idx_common = {"purpose": "sub12 limited diagnosis",
                  "control_demo": args.control_demo, "demo": args.demo,
                  "baseline_evidence": path_for_record(baseline_evidence_path),
                  "baseline_evidence_sha256":
                      integ.sha256_file(baseline_evidence_path)}
    if blocked:
        # a blocked diagnosis still allocates its own attempt and writes an
        # explicit record (never a crash with no report on disk); ZERO
        # child processes are launched on blocked evidence
        attempt_dir = integ.allocate_attempt(out_root)
        summary = {"status": "diagnostic_incomplete", "blocked": blocked,
                   "preconditions": pre,
                   "timestamp": _dt.datetime.now().isoformat(timespec="seconds")}
        (attempt_dir / "diagnosis_run.json").write_text(json.dumps(
            summary, indent=2, ensure_ascii=False))
        integ.append_attempts_index(out_root, {
            **idx_common, "attempt": attempt_dir.name,
            "status": summary["status"],
            "report": path_for_record(attempt_dir / "diagnosis_run.json")})
        print(json.dumps(summary, indent=2))
        return 2

    retargeter_sha = integ.sha256_file(RETARGETER_FILE)

    # §9 structure (review fix 4): the control equivalence is CONFIRMED
    # before ANY sub12 subprocess is launched — attempt_001 = control only;
    # a non-equivalent control ends the diagnosis there with zero sub12
    # calls; attempt_002 = sub12 run 1; attempt_003 = sub12 run 2 (the
    # concluding verdict lives there).
    def _final_summary(verdict, control_cmp, runs, attempts):
        return {
            "stage": "D0 sub12 limited diagnosis (closure plan §5)",
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "attempts": attempts,
            "baseline": {
                "baseline_evidence": path_for_record(baseline_evidence_path),
                "baseline_evidence_sha256":
                    integ.sha256_file(baseline_evidence_path),
                "evidence_files": path_for_record(evidence_files_path),
                "authoritative_selection": path_for_record(selection_path),
                "authoritative_selection_sha256": selection_now,
                "authoritative_selection_still_pinned": selection_ok,
            },
            "preconditions": pre,
            "identity": {"checkout": checkout, "env": env,
                         "retargeter_sha256": retargeter_sha,
                         "child_driver_sha256": integ.sha256_file(CHILD_DRIVER),
                         "data_path": path_for_record(data_path)},
            "control": control_cmp,
            "runs": runs,
            "verdict": verdict,
            "wall_s": round(time.time() - t0, 1),
        }

    att1 = integ.allocate_attempt(out_root)
    control = run_child(args.env_python, args.control_demo, data_path,
                        att1 / "control", att1 / "control", retargeter_sha,
                        args.run_timeout_s)
    reg_raw_info = (ev.get("u0_evidence", {}).get("attempt_001", {})
                    .get(args.control_demo, {}).get("raw", {}))
    control_cmp = {"run": {k: v for k, v in control.items()
                           if k != "summary"}}
    if reg_raw_info:
        control_cmp["comparison"] = compare_control(
            att1 / "control" / f"{args.control_demo}_original.npz",
            _recorded_path(reg_raw_info["path"]), reg_raw_info.get("sha256"))
    else:
        control_cmp["comparison"] = {"equal": False,
                                     "reason": "no registered raw artifact"}
    integ.append_attempts_index(out_root, {
        **idx_common, "attempt": att1.name, "purpose": "sub16 control",
        "control_equal": control_cmp["comparison"].get("equal"),
        "report": path_for_record(att1 / "control" / "child_summary.json")})
    if not control_cmp["comparison"].get("equal"):
        verdict = decide_diagnosis(control_cmp["comparison"], [])
        out_json = att1 / "diagnosis_run.json"
        out_json.write_text(json.dumps(
            _final_summary(verdict, control_cmp, [],
                           {"control": att1.name, "runs": []}),
            indent=2, ensure_ascii=False, default=str))
        integ.append_attempts_index(out_root, {
            **idx_common, "attempt": att1.name, "purpose": "sub16 control",
            "status": verdict["status"],
            "report": path_for_record(out_json)})
        print(json.dumps(verdict, indent=2, ensure_ascii=False, default=str))
        print(f"diagnosis -> {path_for_record(out_json)}")
        return 2

    att2 = integ.allocate_attempt(out_root)
    run1 = run_child(args.env_python, args.demo, data_path,
                     att2 / "sub12_run1", att2 / "sub12_run1",
                     retargeter_sha, args.run_timeout_s)
    att3 = integ.allocate_attempt(out_root)
    run2 = run_child(args.env_python, args.demo, data_path,
                     att3 / "sub12_run2", att3 / "sub12_run2",
                     retargeter_sha, args.run_timeout_s)

    verdict = decide_diagnosis(control_cmp["comparison"], [run1, run2])
    runs = [{k: v for k, v in r.items() if k != "summary"}
            | {"failure": (r.get("summary") or {}).get("failure"),
               "outcome": (r.get("summary") or {}).get("outcome"),
               "frames_completed": (r.get("summary") or {})
               .get("frames_completed")}
            for r in (run1, run2)]
    summary = _final_summary(
        verdict, control_cmp, runs,
        {"control": att1.name, "run1": att2.name, "run2": att3.name})
    out_json = att3 / "diagnosis_run.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                   default=str))
    integ.append_attempts_index(out_root, {
        **idx_common, "attempt": att2.name, "purpose": "sub12 diagnosis 1",
        "run1_outcome": (run1.get("summary") or {}).get("outcome"),
        "report": path_for_record(att2 / "sub12_run1" / "child_summary.json")})
    integ.append_attempts_index(out_root, {
        **idx_common, "attempt": att3.name, "purpose": "sub12 diagnosis 2",
        "status": verdict["status"],
        "report": path_for_record(out_json)})
    print(json.dumps(verdict, indent=2, ensure_ascii=False, default=str))
    print(f"diagnosis -> {path_for_record(out_json)}")
    definitive = not verdict["status"].startswith("diagnostic_incomplete")
    return 0 if definitive else 2


# ---- G0: recovery preregistration (§6.2) ----------------------------------

PREREG_FIELDS = {
    "experiment_id": "a2_recovery_step03_v1",
    "source_id": "upstream_recovery_step03_v1",
    "single_config_difference": {"retargeter.step_size": {"from": 0.2,
                                                          "to": 0.3}},
    "held_constant": [
        "holosoma checkout (bccd4d7 pinned by the A1 freeze)",
        "upstream environment (lock + mujoco 3.10.0 triple check)",
        "A1 frozen assets and pilot six, full lengths, 30 Hz",
        "initialization (reference-only, augmentation off)",
        "foot sticking / non-penetration / joint limits semantics",
        "object/human/fps/T binding acceptance",
        "g1-g5 gates and the §3.4 verdict rules of the parent plan",
    ],
    "candidate_count": 1,
    "retry_policy": ("one normal run per demo; at most one same-config "
                     "retry for engineering failures or new-failure "
                     "verification; reasons and both costs recorded"),
    "stop_conditions": [
        "missing evidence or budget exhaustion -> incomplete",
        "no improvement -> stop; no additional candidates",
    ],
}


def register_recovery(args) -> int:
    """G0: exclusive-create preregistration.json (§6.2/§9).

    Review fix 5 (2026-10-10): the referenced diagnosis is VERIFIED as
    evidence, not just status-read — the recorded verdict must be backed by
    the recorded facts (control equivalence, two-run consistency over the
    extended evidence, rho in the candidate window) and its parent baseline
    hash must still resolve. A status string alone no longer registers a
    candidate.
    """
    diag_path = Path(args.from_diagnosis).expanduser().resolve()
    if not diag_path.is_file():
        raise SystemExit(f"diagnosis summary missing: {diag_path}")
    diag = _load_json(diag_path)
    if diag.get("verdict", {}).get("status") != "supports_step03_candidate":
        raise SystemExit(
            "refusing to register: diagnosis status is "
            f"{diag.get('verdict', {}).get('status')!r}; the single 0.3 "
            "candidate is registrable only from supports_step03_candidate")
    verification = verify_diagnosis_record(diag)
    if not verification["verified"]:
        raise SystemExit(
            "refusing to register: the diagnosis record does not verify as "
            f"evidence: {verification['problems']}")
    out_root = integ.guard_output_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    gen_root = out_root / "generation"
    if gen_root.exists() and any(gen_root.iterdir()):
        raise SystemExit(
            "refusing to register: UR products already exist without a "
            "prior registration (§6.2 no retroactive registration)")
    ev = _load_json(Path(args.baseline_evidence).expanduser().resolve().parent
                    / "evidence_files.json")
    freeze_dir = _recorded_path(
        ev["a1_freeze"]["frozen_lists.json"]["path"]).parent
    pilot = [e["demo"] for e in _load_json(
        freeze_dir / "frozen_lists.json")["teacher_pilot_train6_v1"]]
    prereg = {
        **PREREG_FIELDS,
        "registered_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "parent": {
            "authoritative_selection": ev["authoritative_selection"]["path"],
            "authoritative_selection_sha256":
                ev["authoritative_selection"]["sha256"],
            "diagnosis": path_for_record(diag_path),
            "diagnosis_sha256": integ.sha256_file(diag_path),
        },
        "diagnosis_verification": {
            "verified": verification["verified"],
            "problems": verification["problems"],
            "facts": verification["facts"],
        },
        "expected_demos": pilot,
        "expected_denominator": len(pilot),
        "recovery_dir": path_for_record(out_root),
    }
    target = out_root / "preregistration.json"
    try:
        with open(target, "x") as f:
            f.write(json.dumps(prereg, indent=2, ensure_ascii=False))
    except FileExistsError:
        raise SystemExit(f"preregistration already exists: {target} "
                         "(never overwritten; write a NEW experiment)")
    integ.append_attempts_index(out_root, {
        "attempt": "preregistration",
        "experiment_id": prereg["experiment_id"],
        "sha256": integ.sha256_file(target),
        "path": path_for_record(target)})
    print(f"preregistration -> {path_for_record(target)}")
    print(f"sha256 = {integ.sha256_file(target)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="run the D0 limited diagnosis")
    d.add_argument("--baseline-evidence", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True,
                   help="diagnosis dir (attempt_NNN inside)")
    d.add_argument("--env-python", type=Path, default=DEFAULT_ENV_PYTHON)
    d.add_argument("--demo", default="sub12_largebox_082")
    d.add_argument("--control-demo", default="sub16_largebox_000")
    d.add_argument("--data-path", type=Path, default=OMOMO_NEW_DIR)
    d.add_argument("--run-timeout-s", type=int, default=900,
                   help="per-generation budget (§11: <=15 min)")

    r = sub.add_parser("register-recovery", help="G0 preregistration")
    r.add_argument("--from-diagnosis", type=Path, required=True,
                   help="diagnosis_run.json with the final verdict")
    r.add_argument("--baseline-evidence", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True,
                   help="recovery dir (preregistration.json inside)")

    args = ap.parse_args()
    if args.cmd == "diagnose":
        return diagnose(args)
    return register_recovery(args)


if __name__ == "__main__":
    raise SystemExit(main())
