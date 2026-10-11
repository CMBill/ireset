"""I0 thin entry: generate U0 teachers with the pinned upstream checkout.

Plan omniretarget_first_training_plan §3.3/§10.1: drive the independently
managed upstream environment (``derived/upstream/holosoma_env``) over the
frozen pilot list, with the reference-only upstream initialization (no
oracle warm-start), then verify each output against the published teacher
(bit-exact human reference + target box trajectory + fps/T binding) and
convert the checkout's MuJoCo [pos,quat] layout into the project scene
convention.

Run-integrity rules (A2 review fixes 5/6/7/10, 2026-10-09):

* the upstream checkout identity is MEASURED (git rev + dirty state) and
  must match the A1-frozen expected commit before anything runs; the env
  lock hash and mujoco version are measured and recorded;
* every invocation writes its own ``attempt_NNN`` directory and appends to
  ``attempts_index.json`` — a ``--demos`` retry never overwrites prior
  evidence, and ``expected_denominator`` stays the FULL frozen pilot list
  so a partial retry reports INCOMPLETE, never a shrunken COMPLETE;
* input hashes are recorded in the manifest (the comparison re-verifies
  them against the on-disk files) in addition to the before/after
  immutability check inside this run; round-3: the on-disk inputs and all
  A1-frozen asset files are additionally verified against the FREEZE hashes
  and any mismatch stops the run before generation (review 3); round-4:
  the PT-side verification and the immutability anchors cover the ACTUAL
  ``--data-path`` consumed by this run, and the consumed data_path is
  recorded in the manifest (review 2);
* the env binary must match its own lock's pinned mujoco version (a
  re-resolved environment behind an unchanged lock hash is refused);
* the converted artifact's sha256 is recorded alongside the raw's, so the
  comparison can verify the full raw->converted chain (review 5);
* output directories must fully resolve under ``derived/`` (symlink targets
  under resources/ are refused).

Read-only over resources/ and third_party/. Raw upstream outputs are kept
next to the converted copies. All frozen pilot demos keep a reconciled
result; failures never shrink the denominator.

UR mode (closure plan §7.1-§7.2, 2026-10-10): ``--source-record`` selects
the preregistered recovery source ``upstream_recovery_step03_v1`` — the
SINGLE registered difference ``retargeter.step_size: 0.2 -> 0.3`` is
verified through a REAL upstream config parse (tyro smoke; a flag string
appearing in argv proves nothing — tyro 1.x renamed ``argv=`` to ``args=``
and silently ignores the old kwarg), every attempt in the root must belong
to the same registration, per-demo generation gets a wall-clock timeout,
and the manifest records the registration identity + the effective-config
diff. Without ``--source-record`` the entry keeps the exact U0 semantics.

Run: uv run --locked python scripts/generate_omniretarget_teachers.py \
       --freeze derived/experiments/omniretarget_first/<a1_run>/freeze \
       --out-dir derived/experiments/omniretarget_first/<a2_run>/teacher_pilot/u0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ireset.data import a2_integrity as integ
from ireset.data import omniretarget_sources as orc
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record

HOLOSOMA_PACKAGE = (
    PROJECT_ROOT / "third_party" / "holosoma" / "src" / "holosoma_retargeting"
    / "holosoma_retargeting"
)
DEFAULT_ENV_PYTHON = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"
CHILD_DRIVER = PROJECT_ROOT / "scripts" / "_omniretarget_diagnostic_child.py"
UR_STEP_SIZE = 0.3
#: §11 budget: a single generation may not exceed 30 min
GENERATION_TIMEOUT_S = 1800
#: §6.2 retry policy made structural (review fix 11, 2026-10-10; round-3
#: review fix 4): ONE normal run plus AT MOST one justified retry = at most
#: TWO actual subprocess invocations per demo in this root — counted by
#: ACTUAL CALLS (any non-skipped row), never by failures alone: one failed
#: run plus one success previously authorized a third call. Skipped rows
#: (``skipped_*``) record no call and never count.
MAX_INVOCATIONS_PER_DEMO = 2
#: §11 budget: the whole UR experiment may not exceed 3h, and that budget
#: COVERS generation + the final comparison ("预留终检/比较时间") — the
#: generation loop may only spend what remains after reserving the
#: comparison time
UR_TOTAL_BUDGET_S = 3 * 3600
UR_COMPARISON_RESERVE_S = 600
UR_GENERATION_BUDGET_S = UR_TOTAL_BUDGET_S - UR_COMPARISON_RESERVE_S


def _history_manifest_problems(manifest: object, entry: object) -> list[str]:
    """Round-5/6 review fix: structural validation of a REGISTERED attempt's
    manifest. Parsing successfully is not evidence — ``{}`` (no rows, no
    cost) and ``rows`` without ``total_wall_s`` both used to read as zero
    invocations and zero cumulative cost, making the §6.2 cap and the UR
    budget unknowable-in-practice while claimed zero. Required evidence =
    exactly what the gates consume: a finite non-negative-number
    ``total_wall_s``, a list ``rows``, and per row the ``demo``/``status``
    keys (a row without ``demo`` cannot be attributed to the scope — and
    would crash the counting loop anyway).

    Round-6 fix 1: the recorded cost must also be non-negative and
    RECONCILE with the per-row evidence — a ``total_wall_s`` of −600 used
    to RESTORE budget, and a 9900s row sum flattened to a 0 total re-
    approved six fresh 300s calls past the 3h budget. Genuine writers
    record ``total_wall_s = round(sum(row.wall_s), 1)`` — one rounding, so
    the reconciliation bound is 0.05 (real roots: 0.04/0.05/0.05). An
    all-skipped attempt's ``0.0`` total against zero-valued rows is
    evidence-backed zero cost and stays legal.

    Round-6 fix 2: the rows must reconcile with the REGISTERED scope
    (index entry ``scope`` and manifest ``attempt_scope`` — genuine
    attempts record exactly one row per scope demo): deleting a call row
    used to undercount a demo's invocations and re-open a third call.
    """
    att = entry.get("attempt") if isinstance(entry, dict) else None
    att = att if isinstance(att, str) else "unknown_attempt"
    if not isinstance(manifest, dict):
        return [f"{att}:manifest_not_object"]
    problems: list[str] = []
    total = manifest.get("total_wall_s")
    total_ok = (not isinstance(total, bool)
                and isinstance(total, (int, float))
                and math.isfinite(total) and total >= 0)
    if not total_ok:
        problems.append(f"{att}:total_wall_s_missing_negative_or_not_finite")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        problems.append(f"{att}:rows_missing_or_not_list")
        return problems
    row_sum = 0.0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"{att}:row_{i}_not_object")
            continue
        if not isinstance(row.get("demo"), str):
            problems.append(f"{att}:row_{i}_demo_missing_or_not_str")
        if "status" not in row:
            problems.append(f"{att}:row_{i}_status_missing")
        wall = row.get("wall_s")
        if (isinstance(wall, bool) or not isinstance(wall, (int, float))
                or not math.isfinite(wall) or wall < 0):
            problems.append(
                f"{att}:row_{i}_wall_s_missing_negative_or_not_finite")
        else:
            row_sum += float(wall)
    if total_ok and abs(float(total) - row_sum) > 0.05 + 1e-6:
        problems.append(f"{att}:total_wall_s_mismatch_row_sum:"
                        f"{total}:vs:{round(row_sum, 2)}")
    scopes: list[tuple[str, list]] = []
    if isinstance(entry, dict) and isinstance(entry.get("scope"), list):
        scopes.append(("index_scope", entry["scope"]))
    if isinstance(manifest.get("attempt_scope"), list):
        scopes.append(("manifest_scope", manifest["attempt_scope"]))
    if not scopes:
        problems.append(f"{att}:scope_missing_everywhere")
    else:
        row_demos = sorted(row.get("demo") for row in rows
                           if isinstance(row, dict)
                           and isinstance(row.get("demo"), str))
        for name, sc in scopes:
            scope_demos = sorted(str(d) for d in sc)
            if scope_demos != row_demos:
                problems.append(f"{att}:rows_do_not_match_{name}:"
                                f"{scope_demos}:vs:{row_demos}")
    return problems


def _prior_run_facts(out_root: Path,
                     scope: list[str]) -> tuple[dict, float, list[str]]:
    """Actual-invocation history and cumulative wall time already recorded
    in this root — the retry cap and the UR budget gate are enforced ACROSS
    invocations, not per call (the first round happily launched a third
    same-config attempt after two recorded failures).

    Round-3 review fix 4: the cap counts REAL calls (every row whose status
    is not ``skipped*`` — ok, generation_failed, binding_failed, timeout,
    postcheck_failed, input_mutated, and unknown/None statuses
    conservatively), never failures alone, and each entry carries the prior
    status so the retry justification (a retry is legitimate only after a
    FAILED run) is auditable in the skipped row.

    Round-4 review fix 4: an attempt REGISTERED in the index whose manifest
    cannot be read is NOT silently skipped history — the invocation count
    and the cumulative budget are unknowable over unreadable rows, so the
    error is returned and the caller must stop BEFORE any generation
    (unreadable history must never be treated as zero history).

    Round-5 review fix: a manifest that PARSES but lacks the required
    structure is the same unknowable history — ``{}`` or rows-without-
    ``total_wall_s`` must not slip through the defaults as zero calls and
    zero cost (``_history_manifest_problems``).

    Round-6 review fixes: the cost must also be non-negative and reconcile
    with the per-row ``wall_s`` sum (a negative total used to restore
    budget; a flattened total re-approved calls past 3h), and the rows must
    reconcile with the REGISTERED scope (index ``scope`` / manifest
    ``attempt_scope``) — a deleted call row used to undercount a demo's
    invocations.
    """
    prior_calls: dict[str, list[dict]] = {}
    prior_wall_s = 0.0
    history_errors: list[str] = []
    if not (out_root / integ.ATTEMPTS_INDEX).is_file():
        return prior_calls, prior_wall_s, history_errors
    for att in integ.load_attempts(out_root):
        try:
            manifest = integ.load_attempt_manifest(out_root, att["attempt"])
        except Exception as exc:  # noqa: BLE001 -- history unreadable
            history_errors.append(
                f"{att.get('attempt')}:{type(exc).__name__}")
            continue
        problems = _history_manifest_problems(manifest, att)
        if problems:
            history_errors.extend(problems)
            continue
        prior_wall_s += float(manifest["total_wall_s"])
        for row in manifest["rows"]:
            if row.get("demo") not in scope:
                continue
            status = row.get("status")
            if str(status).startswith("skipped"):
                continue          # no subprocess was launched for this row
            prior_calls.setdefault(row["demo"], []).append(
                {"attempt": att["attempt"], "status": status})
    return prior_calls, prior_wall_s, history_errors


def pilot_demos_from_freeze(freeze_dir: Path) -> list[str]:
    lists = json.loads((freeze_dir / "frozen_lists.json").read_text())
    return [e["demo"] for e in lists["teacher_pilot_train6_v1"]]


def run_config_smoke(env_python: Path, demo: str, data_path: Path,
                     step_size: float) -> dict:
    """Real upstream config parse via the child driver (§7.1): tyro must
    actually APPLY ``--retargeter.step-size`` — a flag string appearing in
    argv proves nothing (tyro 1.x silently ignores the legacy ``argv=``
    kwarg, caught live by this smoke on 2026-10-10)."""
    cmd = [str(env_python), str(CHILD_DRIVER), "--mode", "config-smoke",
           "--demo", demo, "--data-path", str(data_path),
           "--save-dir", str(data_path.parent / "_smoke_no_write"),
           "--holosoma-package", str(HOLOSOMA_PACKAGE),
           "--step-size", str(step_size)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise SystemExit(f"config smoke failed: {(proc.stderr or proc.stdout)[-1500:]}")
    return json.loads(proc.stdout)


def load_source_record(path: Path, pilot: list[str], env_python: Path,
                       data_path: Path) -> dict:
    """Load + verify the UR preregistration BEFORE any generation (§6.2/§7.1).

    Identity, the single registered difference, the parent hashes and the
    REAL config-parse diff are all verified here; a drifted preregistration
    or a config smoke that does not reproduce exactly the one registered
    difference refuses to run.
    """
    prereg = json.loads(Path(path).read_text())
    if (prereg.get("experiment_id") != "a2_recovery_step03_v1"
            or prereg.get("source_id") != "upstream_recovery_step03_v1"
            or prereg.get("single_config_difference")
            != {"retargeter.step_size": {"from": 0.2, "to": UR_STEP_SIZE}}
            or list(prereg.get("expected_demos", [])) != pilot):
        raise SystemExit(
            f"preregistration identity mismatch vs the frozen pilot: {path}")
    parent = prereg.get("parent", {})
    for key in ("authoritative_selection", "diagnosis"):
        rel, want = parent.get(key), parent.get(f"{key}_sha256")
        p = Path(rel) if rel else None
        if not p or not p.is_file() or integ.sha256_file(p) != want:
            raise SystemExit(f"preregistration parent {key} hash mismatch: {rel}")
    smoke = run_config_smoke(env_python, pilot[0], data_path, UR_STEP_SIZE)
    if (smoke["diff"] != {"retargeter.step_size": [0.2, UR_STEP_SIZE]}
            or smoke["step_size_default"] != 0.2
            or smoke["step_size_recovery"] != UR_STEP_SIZE):
        raise SystemExit(
            "config smoke does not reproduce the single registered "
            f"difference: {smoke['diff']}")
    return {"prereg": prereg,
            "preregistration": path_for_record(Path(path)),
            "preregistration_sha256": integ.sha256_file(Path(path)),
            "config_smoke": {k: smoke[k] for k in
                             ("diff", "step_size_default", "step_size_recovery",
                              "canonical_config_sha256")},
            "config_smoke_full": {k: smoke[k] for k in ("default", "recovery")}}


def ensure_single_registration(out_root: Path, prereg_sha: str) -> None:
    """§7.2: every attempt already in this root must belong to the SAME
    registration — a UR root cannot mix registrations (and never mixes in
    plain U0 attempts, which carry no preregistration sha)."""
    if not (out_root / integ.ATTEMPTS_INDEX).is_file():
        return
    for att in integ.load_attempts(out_root):
        prev = att.get("preregistration_sha256")
        if prev is None or prev != prereg_sha:
            raise SystemExit(
                f"attempt {att.get('attempt')} does not belong to this "
                "registration; a UR root cannot mix registrations or "
                "include unregistered attempts")


def generate_one(
    demo: str,
    env_python: Path,
    attempt_dir: Path,
    data_path: Path,
    extra_argv: list[str] | None = None,
    timeout_s: int | None = GENERATION_TIMEOUT_S,
) -> dict:
    """Run the upstream retargeting for one demo; -> per-demo result row.

    ``timeout_s=None`` keeps the §11 single-generation default (the budget
    gate caps this at the remaining UR budget when it passes a value).
    """
    if timeout_s is None:
        timeout_s = GENERATION_TIMEOUT_S
    t0 = time.time()
    cmd = [
        str(env_python), "examples/robot_retarget.py",
        "--task-type", "object_interaction",
        "--task-name", demo,
        "--data-format", "smplh",
        "--data-path", str(data_path),
        "--save-dir", str(attempt_dir),
        "--task-config.object-name", "largebox",
    ] + list(extra_argv or [])
    try:
        proc = subprocess.run(
            cmd, cwd=HOLOSOMA_PACKAGE, capture_output=True, text=True,
            timeout=timeout_s,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code, timed_out = None, True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    row = {
        "demo": demo,
        "argv": cmd,
        "cwd": str(HOLOSOMA_PACKAGE),
        "exit_code": exit_code,
        "wall_s": round(time.time() - t0, 2),
        "config": "upstream defaults: joint_limits/obj_non_penetration/foot_sticking on, "
                  "visualize/debug off, augmentation off (reference-only init)"
                  + (f"; UR override: retargeter.step_size={UR_STEP_SIZE}"
                     if extra_argv else ""),
        "status": None, "binding": None, "error": None,
    }
    if timed_out:
        row["status"] = "timeout"
        row["error"] = (stderr or stdout)[-2000:]
        return row
    if exit_code != 0:
        row["status"] = "generation_failed"
        row["error"] = (stderr or stdout)[-2000:]
        return row
    try:
        raw_path = attempt_dir / f"{demo}_original.npz"
        pub_path = TEACHER_TRAJ_DIR / f"{demo}_original.npz"
        scene_u0, human_u0, fps_u0 = orc.load_teacher_scene(raw_path, "upstream_generated")
        scene_p, human_p, fps_p = orc.load_teacher_scene(pub_path, "published")
        binding = orc.check_binding(demo, scene_u0, human_u0, fps_u0, scene_p, human_p, fps_p)
        conv_path = attempt_dir / f"{demo}_original_converted.npz"
        np.savez(
            conv_path,
            qpos=scene_u0,
            human_joints=human_u0,
            fps=fps_u0,
            source_type="upstream_generated",
            layout="project_scene_v1",
            provenance=json.dumps({
                "demo": demo, "raw_file": raw_path.name,
                "raw_sha256": integ.sha256_file(raw_path),
                "published_sha256": integ.sha256_file(pub_path),
                "binding": binding.to_json_dict(),
            }),
        )
        row["status"] = "ok" if binding.passed else "binding_failed"
        row["binding"] = binding.to_json_dict()
        row["raw_sha256"] = integ.sha256_file(raw_path)
        row["converted_sha256"] = integ.sha256_file(conv_path)
        row["converted"] = path_for_record(conv_path)
        row["T"] = int(scene_u0.shape[0])
    except Exception as exc:
        row["status"] = "postcheck_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True,
                    help="A1 freeze directory (frozen_lists.json + assets.json)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="U0 root directory (each run creates attempt_NNN inside)")
    ap.add_argument("--env-python", type=Path, default=DEFAULT_ENV_PYTHON)
    ap.add_argument("--data-path", type=Path, default=OMOMO_NEW_DIR)
    ap.add_argument("--demos", type=str, default=None,
                    help="comma-separated retry scope (subset of the frozen pilot "
                         "list); writes a NEW attempt, never overwrites")
    ap.add_argument("--source-record", type=Path, default=None,
                    help="UR preregistration.json (closure plan §7.1); the "
                         "single registered difference "
                         "retargeter.step_size 0.2->0.3 is applied and "
                         "verified; omit for the original U0 semantics")
    args = ap.parse_args()

    freeze = args.freeze.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    out_root = integ.guard_output_dir(args.out_dir)

    pilot = pilot_demos_from_freeze(freeze)
    scope = pilot
    if args.demos:
        wanted = [d.strip() for d in args.demos.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in pilot]
        if unknown:
            raise SystemExit(f"--demos outside frozen pilot list: {unknown}")
        # round-4 review fix 3: one demo twice in a single invocation would
        # bypass the per-demo invocation cap (the cap history is read once
        # before the loop, so `--demos A,A` after a failure used to launch
        # TWO more calls = three actual invocations); the sanctioned retry
        # path is re-invoking this entry, whose history the cap reads
        dups = sorted({d for d in wanted if wanted.count(d) > 1})
        if dups:
            raise SystemExit(
                f"duplicate demo in --demos: {dups} — each demo may appear "
                "once per invocation; the retry path is re-running the "
                "entry (the invocation cap counts ACTUAL calls across the "
                "root, §6.2)")
        scope = wanted

    # Identity gate BEFORE any generation: measured checkout must equal the
    # A1-frozen commit and be clean; env identity is recorded, not assumed.
    freeze_assets = json.loads((freeze / "assets.json").read_text())
    checkout = integ.verify_checkout_against_freeze(freeze_assets)
    if not checkout["passed"]:
        raise SystemExit(
            f"upstream checkout identity check failed vs A1 freeze: {checkout}")
    env = integ.measure_env_identity(args.env_python)
    if env["env_mujoco_version"] is None:
        raise SystemExit(f"env probe failed: {env['env_probe_error']}")
    if env["env_lock_mujoco_version"] != env["env_mujoco_version"]:
        raise SystemExit(
            "env binary drifted from its own lock: probed mujoco "
            f"{env['env_mujoco_version']} vs lock-pinned "
            f"{env['env_lock_mujoco_version']}")

    # Round-3 review fix 3: the inputs and the frozen assets are verified
    # against the A1 FREEZE (not merely re-recorded) — a mutated input or a
    # drifted asset must stop the run before any teacher is generated.
    # Round-4 review fix 2: the PT side is verified at the ACTUAL --data-path
    # this run consumes, not at the canonical directory — passing an
    # alternate source dir must not pass a check anchored elsewhere.
    frozen_entries = json.loads(
        (freeze / "frozen_lists.json").read_text())["teacher_pilot_train6_v1"]
    freeze_inputs = integ.verify_inputs_against_freeze(
        [e for e in frozen_entries if e["demo"] in scope],
        TEACHER_TRAJ_DIR, data_path)
    bad_inputs = [d for d, r in freeze_inputs.items()
                  if not (r["npz_matches"] and r["pt_matches"])]
    if bad_inputs:
        raise SystemExit(
            f"inputs no longer match the A1-frozen hashes: {bad_inputs}")
    freeze_assets_check = integ.verify_assets_against_freeze(freeze_assets)
    if not freeze_assets_check["passed"]:
        raise SystemExit(
            f"A1-frozen assets drifted: {freeze_assets_check}")

    # UR mode: the preregistration is verified (identity, single difference,
    # parent hashes, REAL config-parse smoke) before anything runs, and an
    # existing attempts index in this root must belong to the SAME
    # registration — a mixed root is refused (§7.1/§7.2).
    source_record = None
    if args.source_record is not None:
        source_record = load_source_record(args.source_record, pilot,
                                           args.env_python, data_path)
        ensure_single_registration(out_root,
                                   source_record["preregistration_sha256"])

    # review fix 11 + round-3 review fixes 4/5 + round-4/5 review fixes: the
    # invocation history is read BEFORE this run allocates its attempt, and
    # history that cannot be read OR parses but lacks the required
    # structure (``{}``, rows without total_wall_s) STOPS the run — both
    # make the invocation count and the UR cumulative budget unknowable,
    # and generating over them would silently restart the root's past as
    # zero.
    prior_calls, prior_wall_s, history_errors = _prior_run_facts(
        out_root, scope)
    if history_errors:
        raise SystemExit(
            f"history_incomplete: prior attempt manifest(s) unreadable or "
            f"structurally deficient under {out_root}: {history_errors} — "
            "the per-demo invocation cap (§6.2) and the UR cumulative "
            "budget are unknowable over such history; refusing to generate "
            "(unreadable or deficient history is not zero history)")

    attempt_dir = integ.allocate_attempt(out_root)
    attempt = attempt_dir.name

    # Input immutability anchors: recorded in the manifest (the comparison
    # re-verifies them later) AND checked before/after this run. The PT side
    # anchors the ACTUAL consumed data_path (round-4 review fix 2).
    inputs = {}
    for demo in scope:
        inputs[demo] = {
            "npz": integ.sha256_file(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
            "pt": integ.sha256_file(data_path / f"{demo}.pt"),
        }

    extra_argv = (["--retargeter.step-size", str(UR_STEP_SIZE)]
                  if source_record else None)
    # review fix 11 + round-3 review fixes 4/5: per-demo invocation cap
    # across the root (one normal run + at most one JUSTIFIED retry — a
    # retry after a SUCCESS has no justification) and the UR cumulative
    # budget with a comparison reserve; both enforced BEFORE the subprocess
    # call, with the reason in the row. The budget stops at EXHAUSTION
    # (>=, not >) and caps the CURRENT call's timeout at the remaining
    # budget — a 5s-remaining call used to launch with the default 1800s
    # timeout and blow the 3h experiment budget.
    rows = []
    spent = 0.0
    for d in scope:
        prior = prior_calls.get(d, [])
        if len(prior) >= MAX_INVOCATIONS_PER_DEMO:
            rows.append({
                "demo": d, "argv": None, "cwd": str(HOLOSOMA_PACKAGE),
                "exit_code": None, "wall_s": 0.0,
                "status": "skipped_retry_cap",
                "prior_invocations": prior,
                "reason": f"{MAX_INVOCATIONS_PER_DEMO} actual generation "
                          "invocations already recorded in this root "
                          "(§6.2: one normal run + at most one justified "
                          "retry) — no further attempt without a new "
                          "registration",
            })
            continue
        if (len(prior) == 1 and prior[0]["status"] == "ok"):
            rows.append({
                "demo": d, "argv": None, "cwd": str(HOLOSOMA_PACKAGE),
                "exit_code": None, "wall_s": 0.0,
                "status": "skipped_retry_cap",
                "prior_invocations": prior,
                "reason": "the prior run of this demo already SUCCEEDED; a "
                          "retry is justified only after a failed run "
                          "(§6.2) — regenerate via a new registration, not "
                          "a same-config re-run",
            })
            continue
        call_timeout = None
        if source_record:
            cumulative = prior_wall_s + spent
            remaining = UR_GENERATION_BUDGET_S - cumulative
            if remaining <= 0:
                rows.append({
                    "demo": d, "argv": None, "cwd": str(HOLOSOMA_PACKAGE),
                    "exit_code": None, "wall_s": 0.0,
                    "status": "skipped_budget",
                    "cumulative_wall_s": round(cumulative, 1),
                    "reason": "UR cumulative budget (§11, 3h incl. "
                              f"{UR_COMPARISON_RESERVE_S}s comparison "
                              "reserve) exhausted before this demo",
                })
                continue
            call_timeout = int(min(GENERATION_TIMEOUT_S, remaining))
        row = generate_one(d, args.env_python, attempt_dir, data_path,
                           extra_argv=extra_argv, timeout_s=call_timeout)
        if call_timeout is not None:
            row["budget_timeout_s"] = call_timeout
            row["budget_remaining_s"] = round(
                UR_GENERATION_BUDGET_S - prior_wall_s - spent, 1)
        rows.append(row)
        spent += float(row.get("wall_s") or 0.0)
        # round-4 review fix 3: count THIS run's actual invocations too —
        # the cap must hold across the root AND within one invocation
        # (defense in depth behind the --demos duplicate rejection)
        prior_calls.setdefault(d, []).append(
            {"attempt": attempt, "status": row.get("status")})

    input_mutated = False
    for demo in scope:
        now = {
            "npz": integ.sha256_file(TEACHER_TRAJ_DIR / f"{demo}_original.npz"),
            "pt": integ.sha256_file(data_path / f"{demo}.pt"),
        }
        if now != inputs[demo]:
            input_mutated = True
            for row in rows:
                if row["demo"] == demo and row["status"] == "ok":
                    row["status"] = "input_mutated"

    n_ok = sum(r["status"] == "ok" for r in rows)
    report = {
        "stage": "I0 U0 generation (plan §3.3/§10.1)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "freeze": path_for_record(freeze),
        "data_path": path_for_record(data_path),   # the dir ACTUALLY consumed
        "attempt": attempt,
        "attempt_scope": scope,
        "expected_denominator": len(pilot),
        "identity": {
            "checkout": checkout,
            "env": env,
            "freeze_expected_commit": freeze_assets["upstream"]["expected_commit"],
        },
        "freeze_verification": {
            "inputs_vs_freeze": freeze_inputs,
            "assets_vs_freeze": freeze_assets_check,
        },
        "input_sha256": inputs,
        "n_ok": n_ok,
        "n_binding_failed": sum(r["status"] == "binding_failed" for r in rows),
        "n_generation_failed": sum(r["status"] == "generation_failed" for r in rows),
        "n_postcheck_failed": sum(r["status"] == "postcheck_failed" for r in rows),
        "input_mutated": input_mutated,
        "n_timeout": sum(r["status"] == "timeout" for r in rows),
        "n_skipped_retry_cap": sum(r["status"] == "skipped_retry_cap"
                                   for r in rows),
        "n_skipped_budget": sum(r["status"] == "skipped_budget" for r in rows),
        "prior_wall_s_in_root": round(prior_wall_s, 1),
        "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
        "rows": rows,
    }
    if source_record:
        report["source_record"] = {
            "experiment_id": source_record["prereg"]["experiment_id"],
            "source_id": source_record["prereg"]["source_id"],
            "preregistration": source_record["preregistration"],
            "preregistration_sha256": source_record["preregistration_sha256"],
            "step_size": UR_STEP_SIZE,
            "config_smoke": source_record["config_smoke"],
            # FULL parsed configs: the comparison RE-COMPUTES the single-
            # difference claim from these instead of trusting the summary
            "config_smoke_full": source_record["config_smoke_full"],
        }
        report["stage"] = ("UR recovery generation (closure plan §7, "
                           + source_record["prereg"]["experiment_id"] + ")")
    out = attempt_dir / "generation_manifest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    integ.append_attempts_index(out_root, {
        "attempt": attempt,
        "scope": scope,
        "expected_denominator": len(pilot),
        "n_ok": n_ok,
        "complete": (n_ok == len(pilot)) and not input_mutated,
        "manifest": integ.manifest_path_record(out_root, attempt),
        **({"preregistration_sha256":
            source_record["preregistration_sha256"],
            "experiment_id": source_record["prereg"]["experiment_id"]}
           if source_record else {}),
    })
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))
    print(f"manifest -> {path_for_record(out)}")
    ok = n_ok == len(pilot) and not input_mutated
    print(f"generation ({attempt}, scope {len(scope)}/{len(pilot)}): "
          f"{'COMPLETE' if ok else 'INCOMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
