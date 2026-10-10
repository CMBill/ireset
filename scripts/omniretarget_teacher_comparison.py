"""A2 teacher source comparison: published P vs locally generated U0.

Plan omniretarget_first_training_plan §3.3-§3.4/§8 (A2): replay both
sources for the frozen pilot six on the same locked validator geometry and
reference intervals (inferred from the A1 source-scene pairing), apply the
frozen ``local_preferred`` criteria, and write ``teacher_selection.json``.

Admission is re-verified from the artifacts, never taken from manifest
status strings (A2 review fix 5, 2026-10-09): the upstream checkout is
measured again and compared against the A1 freeze; the env lock hash must
equal the one recorded at generation; every consumed input file is hashed
against the generation manifest; and the U0-vs-P binding is recomputed
bit-exactly from the converted and published NPZs. Any mismatch or any
frozen pilot demo without an OK artifact forces ``incomplete``.

Round-3 additions (2026-10-09): the A1-frozen ASSET hashes are re-verified;
the GENERATION-time checkout recorded in each consumed attempt manifest is
checked against the freeze; the env check adds the lock-pinned mujoco
version and the actually-importable binary to the lock-sha comparison; the
raw->converted artifact chain is verified bit-exactly per demo (robot
labels cannot be rewritten between generation and comparison); per-demo
artifact defects are recorded instead of raised; and every invocation
writes its own ``cmp/attempt_NNN/teacher_selection.json`` — prior verdict
files are never overwritten, and a fresh ``--out`` directory is created.

Round-4 additions (2026-10-09): the input reconciliation is THREE-WAY —
current inputs vs the generation manifest (existing) AND current inputs vs
the A1-FROZEN hashes — so a tampered input the manifest faithfully recorded
still fails admission; an env probe that cannot launch (missing
interpreter) is recorded, never raised; and the verdict's evidence check
runs before the quality gates, so deficient rows yield ``incomplete``
instead of a TypeError/KeyError mid-computation.

Round-5 additions (2026-10-10): BOTH PT-side legs of the input
reconciliation (manifest and freeze) anchor the data path each consumed
attempt RECORDED consuming (manifest ``data_path``, round-4 generation
entry) — a run over a freeze-identical alternate dir that later drifted
previously passed admission because both legs hashed the untouched
canonical OMOMO dir. Legacy manifests (pre round-4) never recorded the
path; they fall back to the canonical dir and are labeled
``recorded_in_manifest: false``. The both-hands denominator is verified
for existence, validity and P/U0 agreement (missing/None is deleted
evidence → ``incomplete``, never "zero reference frames").

Round-6 additions (2026-10-10): the freeze leg verifies each data dir
ONLY for the demos actually consumed from it (a split retry dir holding
one retry PT is not rejected for files it never served; demos without an
OK artifact are reconciled at the canonical dir), and the aggregate
denominator must still cover the full pilot. The both-hands denominator
must be a non-negative INTEGER — ``int()`` truncation previously read
-0.5 as a legal zero and 30.9 as 30.

Round-7 additions (2026-10-10): the MANIFEST leg is scoped the same way
— only the demos currently consumed from an attempt are re-hashed, so a
superseded demo's PT copy missing from the older dir does not reject the
demos still consumed from it.

P rows come from the published NPZ; U0 rows from the newest OK attempt
artifact per demo. Every expected demo keeps a row on both sides;
missing/invalid evidence forces ``incomplete`` (never a silent drop).
Fixed-input ``box_ground`` conflicts are recorded separately (§7.3). U1 is
not pre-registered in this round -> ``not_applicable`` (§3.3).

UR mode (closure plan §7.3, 2026-10-10): ``--generated-dir`` +
``--source-record`` (mutually exclusive with ``--u0-dir``) compares P vs
the preregistered recovery source UR — the report labels the source UR
(metrics key, evidence gaps, gate fields), never "U0". Admission gains a
registration/config reconciliation leg: every consumed attempt's manifest
must pin the CURRENT preregistration hash, the recorded full parsed
configs must re-diff to EXACTLY the single registered difference, every
consumed row's argv must carry the step-size override, and a FRESH
upstream config smoke must reproduce step_size 0.3 + the canonical config
hash (argv/config strings in the manifest are never trusted alone).

Run: uv run --locked python scripts/omniretarget_teacher_comparison.py \
       --freeze <a1_run>/freeze --u0-dir <a2_run>/teacher_pilot/u0 \
       --out <a2_run>/teacher_pilot
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

import numpy as np

from _omniretarget_diagnostic_child import (  # main-env-safe (lazy cvxpy)
    CONFIG_RUN_METADATA_KEYS, _flatten,
)

from ireset.data import a2_integrity as integ
from ireset.data import omniretarget_sources as orc
from ireset.eval.teacher_comparison import (
    DemoMetrics, TeacherQualityReplayer, evaluate_local_preferred,
)
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record


def load_source_boxes(freeze_dir: Path) -> dict[str, np.ndarray]:
    npz = np.load(freeze_dir / "pairing_source_boxes.npz")
    return {k: np.asarray(v) for k, v in npz.items()}


def measure_source(
    replayer: TeacherQualityReplayer,
    demos: list[str],
    scene_loader,
    source_boxes: dict[str, np.ndarray],
) -> dict[str, DemoMetrics]:
    out: dict[str, DemoMetrics] = {}
    for demo in demos:
        try:
            scene, human, fps = scene_loader(demo)
            if demo not in source_boxes:
                raise ValueError("no A1 source-scene box for this demo")
            out[demo] = replayer.measure(demo, scene, human, source_boxes[demo], fps)
        except Exception as exc:
            out[demo] = DemoMetrics(demo, -1, float("nan"), valid=False,
                                    error=f"{type(exc).__name__}: {exc}")
        r = out[demo]
        pen = f"{r.penetration_max_m:.4f}" if np.isfinite(r.penetration_max_m) else "n/a"
        task = f"{r.task_mpjpe_p95_m:.4f}" if np.isfinite(r.task_mpjpe_p95_m) else "n/a"
        print(f"    {demo}: valid={r.valid} pen_max={pen} task_p95={task} "
              f"both={r.both.get('valid_coverage')}/{r.both.get('gap_series_p95_m')}")
    return out


def _pt_dir_from_record(recorded: str) -> Path:
    """The PT dir a generation manifest recorded consuming.

    ``path_for_record`` writes repo-relative paths for dirs inside the
    project and absolute paths otherwise; undo exactly that (never trust
    the recorded string to be absolute).
    """
    p = Path(recorded).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _recompute_config_diff(full: dict) -> dict:
    """Re-diff the manifest's FULL parsed configs (never trust the summary)."""
    fd = {k: v for k, v in _flatten(full.get("default", {})).items()
          if k not in CONFIG_RUN_METADATA_KEYS}
    fr = {k: v for k, v in _flatten(full.get("recovery", {})).items()
          if k not in CONFIG_RUN_METADATA_KEYS}
    return {k: [fd.get(k), fr.get(k)]
            for k in sorted(set(fd) | set(fr)) if fd.get(k) != fr.get(k)}


#: per-row argv metadata masked when comparing two rows' command lines
_ARGV_META_FLAGS = ("--task-name", "--save-dir")

#: the value-taking flags a generation row must parse cleanly — a row whose
#: command line is TRUNCATED (a flag dangling at the end without its value)
#: must fail registration as a parse error, never crash the comparison
#: (round-3 review fix 6, 2026-10-10)
_ARGV_VALUE_FLAGS = ("--task-type", "--task-name", "--data-format",
                     "--data-path", "--save-dir",
                     "--task-config.object-name", "--retargeter.step-size")


def _argv_flag_value(argv: list[str], flag: str) -> str | None:
    """Value of ``flag`` in a recorded argv — ``None`` when the flag is
    absent OR dangles at the end without a value. A truncated command line
    previously raised IndexError straight out of admission and no report
    was written at all."""
    try:
        i = argv.index(flag)
    except ValueError:
        return None
    return argv[i + 1] if i + 1 < len(argv) else None


def _argv_parse_error(argv: list[str]) -> str | None:
    """Structural parse check of a recorded command line: every known
    value-taking flag must actually carry a value."""
    for flag in _ARGV_VALUE_FLAGS:
        if flag in argv and _argv_flag_value(argv, flag) is None:
            return f"dangling_flag:{flag}"
    return None


def _normalized_argv(argv: list[str]) -> list[str]:
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in _ARGV_META_FLAGS:
            skip = True
            continue
        out.append(tok)
    return out


def _canonical_base_argv(recorded: list[str]) -> list[str]:
    """The canonical U0 generation command with this row's metadata — the
    default side of the argv-bound smoke (review fix 2).

    The recorded argv's own values are reused for task-name/data-path/
    save-dir (per-demo run metadata, stripped from the canonical hash), so
    the base differs from the recorded command ONLY when the recorded
    command carries extra flags or non-template values — exactly what the
    diff must expose. A dangling metadata flag reads as an empty value
    (the row's own parse error is reported separately, never a crash).
    """
    def val(flag: str, default: str = "") -> str:
        v = _argv_flag_value(recorded, flag)
        return default if v is None else v
    return ["--task-type", "object_interaction",
            "--task-name", val("--task-name"),
            "--data-format", "smplh",
            "--data-path", val("--data-path"),
            "--save-dir", val("--save-dir"),
            "--task-config.object-name", "largebox"]


def _argv_config_smoke(env_python: Path, argv: list[str]) -> dict:
    """Comparison-time config smoke over the argv the generation manifest
    ACTUALLY recorded (review fix 2, 2026-10-10).

    The first round smoked a freshly CONSTRUCTED argv while only checking
    that the step-size string appears in the consumed argv — adding
    ``--retargeter.penetration-tolerance 0.1`` to the real command changed
    the parsed constraints yet passed admission, and editing BOTH manifest
    configs consistently never tripped the canonical-hash check (nothing
    re-derived the hash from the actual command). The smoke now parses the
    recorded argv against the CANONICAL base command: the parsed diff must
    be exactly the single registered difference and the canonical hash must
    equal the recorded one — an extra flag shows on the recorded side only
    and fails the single-difference leg (deriving the default from the
    recorded argv itself would hide it on both sides).
    """
    child = Path(__file__).resolve().parent / "_omniretarget_diagnostic_child.py"
    cmd = [str(env_python), str(child), "--mode", "config-smoke",
           "--holosoma-package",
           "third_party/holosoma/src/holosoma_retargeting/holosoma_retargeting",
           "--argv-json", json.dumps(list(argv)),
           "--base-argv-json", json.dumps(_canonical_base_argv(list(argv)))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return {"error": (proc.stderr or proc.stdout)[-500:]}
        smoke = json.loads(proc.stdout)
        return {k: smoke[k] for k in ("diff", "step_size_default",
                                      "step_size_recovery",
                                      "canonical_config_sha256")}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _source_registration_record(gen_root: Path, coverage: dict,
                                prereg: dict, prereg_sha_now: str,
                                env_python: Path | None = None) -> dict:
    """Per-consumed-attempt registration reconciliation (§7.3): every
    attempt backing a consumed demo must belong to THIS preregistration,
    with a single-field config diff, consistent step-size argv across the
    consumed rows, and — review fix 2 — a REAL config parse of every
    consumed row's recorded argv whose canonical hash matches the recorded
    one (the verification is bound to the actual generation command, never
    to a stand-in argv or a flag-string presence check)."""
    step_to = prereg["single_config_difference"]["retargeter.step_size"]["to"]
    single_diff = {"retargeter.step_size": [0.2, step_to]}
    by_attempt: dict[str, set] = {}
    for demo, entry in coverage["by_demo"].items():
        by_attempt.setdefault(entry["attempt"], set()).add(demo)
    per_attempt: dict[str, dict] = {}
    for attempt, demos in sorted(by_attempt.items()):
        manifest = integ.load_attempt_manifest(gen_root, attempt)
        sr = manifest.get("source_record", {})
        rec = {
            "consumed_demos": sorted(demos),
            "experiment_id": sr.get("experiment_id"),
            "source_id": sr.get("source_id"),
            "identity_matches": (
                sr.get("preregistration_sha256") == prereg_sha_now
                and sr.get("experiment_id") == prereg["experiment_id"]
                and sr.get("source_id") == prereg["source_id"]),
            "step_size": sr.get("step_size"),
        }
        full = sr.get("config_smoke_full") or {}
        rec["recomputed_config_diff"] = _recompute_config_diff(full)
        rec["single_difference_holds"] = (
            rec["recomputed_config_diff"] == single_diff)
        rec["recorded_canonical_config_sha256"] = (
            sr.get("config_smoke", {}).get("canonical_config_sha256"))
        bad_argv = []
        parse_errors: dict[str, str] = {}
        normalized: dict[str, list[str]] = {}
        for demo in demos:
            row = integ.manifest_row_for(manifest, demo) or {}
            argv = list(row.get("argv") or [])
            normalized[demo] = _normalized_argv(argv)
            # round-3 review fix 6: a truncated command line (a flag
            # dangling at the end without its value) is a per-row PARSE
            # ERROR recorded in the reconciliation — never an IndexError
            # that crashes the comparison before any report is written
            err = _argv_parse_error(argv)
            if err is not None:
                parse_errors[demo] = err
                bad_argv.append(demo)
                continue
            step_val = _argv_flag_value(argv, "--retargeter.step-size")
            if step_val != str(step_to):
                bad_argv.append(demo)
        rec["argv_step_size_ok"] = not bad_argv
        rec["argv_step_size_bad_demos"] = bad_argv
        rec["argv_parse_errors"] = parse_errors
        rec["argv_consistent_across_rows"] = (
            len({tuple(v) for v in normalized.values()}) <= 1)
        # the real gate: parse each consumed row's ACTUAL recorded argv and
        # require the registered single difference + the recorded hash
        argv_smoke: dict[str, dict] = {}
        if env_python is not None:
            for demo in sorted(demos):
                if demo in parse_errors:
                    # a structurally unparseable row never reaches the
                    # child (the real tyro parse would only fail the same
                    # way); the parse error itself is the recorded reason
                    argv_smoke[demo] = {
                        "error": f"argv_parse_error:{parse_errors[demo]}"}
                    continue
                row = integ.manifest_row_for(manifest, demo) or {}
                smoke = _argv_config_smoke(env_python, list(row.get("argv")
                                                            or []))
                smoke["matches_registration"] = (
                    "error" not in smoke
                    and smoke.get("diff") == single_diff
                    and smoke.get("step_size_default") == 0.2
                    and smoke.get("step_size_recovery") == step_to
                    and smoke.get("canonical_config_sha256")
                    == rec["recorded_canonical_config_sha256"])
                argv_smoke[demo] = smoke
        rec["argv_smoke"] = argv_smoke
        rec["argv_smoke_ok"] = bool(argv_smoke) and all(
            s.get("matches_registration") for s in argv_smoke.values())
        per_attempt[attempt] = rec
    passed = all(
        r["identity_matches"] and r["single_difference_holds"]
        and r["argv_step_size_ok"] and r["argv_consistent_across_rows"]
        and (not r["argv_smoke"] or r["argv_smoke_ok"])
        and r["step_size"] == 0.3
        for r in per_attempt.values()) and bool(per_attempt)
    return {"preregistration_sha256_now": prereg_sha_now,
            "per_attempt": per_attempt, "passed": passed}


def _assert_plain_u0_root(u0_root: Path) -> None:
    """``--u0-dir`` must never consume a UR generation root (review fix 3,
    2026-10-10): the mode is chosen by CLI flags only, and a UR directory
    passed there previously sailed through admission with the registration
    leg entirely skipped, producing a full verdict labeled "U0". Any
    registration marker in the attempts index, or any manifest carrying a
    ``source_record``, refuses the run."""
    for att in integ.load_attempts(u0_root):
        if att.get("preregistration_sha256") or att.get("experiment_id"):
            raise SystemExit(
                f"attempt {att.get('attempt')} carries a recovery "
                "registration; a UR generation root must be compared via "
                "--generated-dir/--source-record, never --u0-dir")
    for att in integ.load_attempts(u0_root):
        try:
            manifest = integ.load_attempt_manifest(u0_root, att["attempt"])
        except Exception:  # noqa: BLE001 -- unresolvable manifest: refuse
            raise SystemExit(
                f"attempt {att.get('attempt')} manifest unreadable; a UR "
                "root cannot be consumed in --u0-dir mode") from None
        if manifest.get("source_record") is not None:
            raise SystemExit(
                f"attempt {att.get('attempt')} manifest records a recovery "
                "source_record; a UR generation root must be compared via "
                "--generated-dir/--source-record, never --u0-dir")


def admission_record(freeze: Path, u0_root: Path, pilot: list[str],
                     env_python: Path | None = None,
                     source_record: tuple[dict, str] | None = None,
                     source_label: str = "U0") -> dict:
    """Artifact-level admission: identity, inputs, artifact chain, bindings.

    Round-3 extensions (2026-10-09): the GENERATION-time checkout recorded
    in each consumed attempt manifest is verified against the A1 freeze (a
    wrong generation checkout behind a correct current checkout is caught);
    the env check is three-way — lock sha recorded at generation vs lock on
    disk now, mujoco version recorded at generation vs lock-pinned version
    vs the version actually importable now (an unchanged lock with a drifted
    binary, or a failed probe, fails admission); and the raw->converted
    artifact chain is re-verified per demo (raw hash vs the generation
    manifest, conversion recomputed bit-exactly), so robot labels cannot be
    rewritten between generation and comparison. Every per-demo artifact
    defect is RECORDED (never raised) and forces ``incomplete`` — a broken
    NPZ must not crash the report (review fix 7).

    Round-5 review fix 1 (2026-10-10): both PT-side legs (manifest and
    freeze) anchor the data path each consumed attempt RECORDED consuming
    (``manifest["data_path"]``, written by the round-4 generation entry).
    Pre-fix, both hashed the canonical OMOMO dir, so a generation over a
    freeze-identical ALTERNATE dir that later drifted passed admission.
    Legacy manifests (pre round-4) never recorded the path; for those the
    legs anchor the canonical dir and the attempt is labeled
    ``recorded_in_manifest: false`` — the strongest evidence a legacy
    manifest admits (its ``input_sha256`` still pins the consumed bytes
    against the canonical-dir check).
    """
    freeze_assets = json.loads((freeze / "assets.json").read_text())
    checkout = integ.verify_checkout_against_freeze(freeze_assets)
    assets_check = integ.verify_assets_against_freeze(freeze_assets)

    coverage = integ.collect_u0_coverage(u0_root, pilot)
    inputs_ok = True
    inputs_detail: dict[str, dict] = {}
    data_paths: dict[str, dict] = {}   # attempt -> the PT dir it consumed
    pt_dirs: dict[str, Path] = {}      # label -> dir, for the freeze leg
    dir_demos: dict[str, set] = {}     # label -> demos CONSUMED at that dir
    # attempt -> demos whose CURRENT evidence comes from this attempt; a
    # later attempt supersedes the older one per demo, so an attempt may
    # have recorded inputs that are no longer consumed from it
    attempt_consumed: dict[str, set] = {}
    for d, e in coverage["by_demo"].items():
        attempt_consumed.setdefault(e["attempt"], set()).add(d)
    env_lock_shas: set[str] = set()
    env_versions: set[str] = set()
    env_pythons: set[str] = set()
    gen_checkout: dict[str, dict] = {}
    chain: dict[str, dict] = {}
    binding: dict[str, dict] = {}
    binding_passed: dict[str, bool] = {}
    expected_commit = freeze_assets["upstream"]["expected_commit"]
    for demo, entry in sorted(coverage["by_demo"].items()):
        attempt = entry["attempt"]
        manifest = integ.load_attempt_manifest(u0_root, attempt)
        ident = manifest.get("identity", {})
        env_lock_shas.add(ident.get("env", {}).get("env_uv_lock_sha256"))
        env_versions.add(ident.get("env", {}).get("env_mujoco_version"))
        env_pythons.add(ident.get("env", {}).get("env_python"))
        recorded_rev = ident.get("checkout", {}).get("holosoma_revision")
        recorded_clean = not ident.get("checkout", {}).get("holosoma_dirty", True)
        gen_checkout[demo] = {
            "attempt": attempt,
            "recorded_revision": recorded_rev,
            "matches_freeze": recorded_rev == expected_commit,
            "recorded_clean": recorded_clean,
        }
        # round-5 review fix 1: the PT side anchors the data path THIS
        # attempt recorded consuming; legacy manifests fall back to the
        # canonical dir and are labeled as such. Round-6 fix 1 tracks WHICH
        # demos each dir was consumed for — the freeze leg verifies a dir
        # only for the demos actually consumed from it, so a split retry
        # dir holding a single retry PT is not rejected for the five files
        # it never served.
        recorded_dp = manifest.get("data_path")
        pt_dir = (_pt_dir_from_record(str(recorded_dp)) if recorded_dp
                  else OMOMO_NEW_DIR)
        if attempt not in data_paths:
            label = path_for_record(pt_dir)
            data_paths[attempt] = {
                "pt_dir": label,
                "recorded_in_manifest": bool(recorded_dp),
            }
            pt_dirs.setdefault(label, pt_dir)
        dir_demos.setdefault(data_paths[attempt]["pt_dir"], set()).add(demo)
        # inputs re-hashed NOW at the recorded data path against what
        # generation recorded, restricted to the demos CONSUMED from this
        # attempt (round-7 review fix): a superseded demo's PT copy in this
        # dir is no longer consumed evidence — its absence must not reject
        # the demos that are. A missing CONSUMED file stays an admission
        # failure for this demo, not a crash.
        try:
            demo_inputs = integ.verify_inputs_against_manifest(
                manifest, TEACHER_TRAJ_DIR, pt_dir,
                only_demos=attempt_consumed[attempt])
            row = demo_inputs.get(demo, {})
            inputs_detail[demo] = {
                "attempt": attempt,
                "data_path": data_paths[attempt]["pt_dir"],
                "data_path_recorded": bool(recorded_dp),
                "npz_matches": row.get("npz_matches"),
                "pt_matches": row.get("pt_matches"),
            }
            if not (row.get("npz_matches") and row.get("pt_matches")):
                inputs_ok = False
        except Exception as exc:  # noqa: BLE001 -- recorded admission failure
            inputs_detail[demo] = {"attempt": attempt,
                                   "data_path": data_paths[attempt]["pt_dir"],
                                   "data_path_recorded": bool(recorded_dp),
                                   "error": f"{type(exc).__name__}: {exc}"}
            inputs_ok = False
        # raw -> converted chain from the artifacts (review fix 5)
        chain[demo] = integ.verify_artifact_chain(
            u0_root, attempt, demo, integ.manifest_row_for(manifest, demo) or {})
        # binding recomputed from the artifacts on disk
        try:
            check = integ.verify_binding_from_artifacts(
                Path(entry["converted"]),
                TEACHER_TRAJ_DIR / f"{demo}_original.npz",
            )
            binding[demo] = {
                "recomputed": check.to_json_dict(),
                "manifest_status": entry["manifest_status"],
                "manifest_binding_passed": entry["manifest_binding_passed"],
                "manifest_agrees": (entry["manifest_status"] == "ok"
                                    and entry["manifest_binding_passed"]),
            }
            binding_passed[demo] = check.passed
        except Exception as exc:  # noqa: BLE001 -- recorded admission failure
            binding_passed[demo] = False
            binding[demo] = {"recomputed": None,
                             "error": f"{type(exc).__name__}: {exc}"}
    for demo in coverage["missing"]:
        binding_passed[demo] = False
        binding[demo] = {"recomputed": None, "note": "no OK artifact in any attempt"}

    # Round-4 review fix 1 + round-5 fix 1: the third leg of the input
    # reconciliation — current inputs vs the A1-FROZEN hashes — so a
    # tampered input the generation manifest faithfully recorded still
    # fails admission, hashed at EVERY data path the consumed attempts
    # actually recorded (plus the canonical dir for legacy manifests): the
    # manifest comparison only proves "inputs now == inputs at generation".
    # Round-6 review fix 1: each dir is verified ONLY for the demos it was
    # actually consumed for — a split retry dir holding one retry PT must
    # not be rejected for files it never served — while demos with no OK
    # artifact are reconciled at the canonical dir so the leg still covers
    # the FULL frozen denominator.
    try:
        frozen_entries = json.loads(
            (freeze / "frozen_lists.json").read_text())["teacher_pilot_train6_v1"]
        if coverage["missing"]:
            fallback = path_for_record(OMOMO_NEW_DIR)
            pt_dirs.setdefault(fallback, OMOMO_NEW_DIR)
            dir_demos.setdefault(fallback, set()).update(coverage["missing"])
        by_dir: dict[str, dict] = {}
        for label, d in sorted(pt_dirs.items()):
            entries = [e for e in frozen_entries
                       if e["demo"] in dir_demos.get(label, ())]
            try:
                by_dir[label] = integ.verify_inputs_against_freeze(
                    entries, TEACHER_TRAJ_DIR, d)
            except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                by_dir[label] = {"error": f"{type(exc).__name__}: {exc}"}

        def _dir_ok(res: dict) -> bool:
            return (bool(res) and "error" not in res
                    and all(r.get("npz_matches") and r.get("pt_matches")
                            for r in res.values()))

        # the aggregate denominator stays the FULL pilot: every demo's PT
        # must actually have been verified somewhere
        verified = set()
        for res in by_dir.values():
            if "error" not in res:
                verified.update(res)
        freeze_inputs_ok = (bool(by_dir)
                            and all(_dir_ok(r) for r in by_dir.values())
                            and verified == set(pilot))
        # per-demo view: each demo was verified at exactly one dir
        merged: dict = {}
        errs = []
        for label in sorted(by_dir):
            res = by_dir[label]
            if "error" in res:
                errs.append(f"{label}: {res['error']}")
            else:
                merged.update(res)
        if errs:
            merged = {"error": "; ".join(errs), **merged}
        inputs_vs_freeze = {"all_match": freeze_inputs_ok,
                            "by_demo": merged, "by_dir": by_dir}
    except Exception as exc:  # noqa: BLE001 -- recorded admission failure
        inputs_vs_freeze = {"all_match": False,
                            "error": f"{type(exc).__name__}: {exc}"}
        freeze_inputs_ok = False

    # env probed NOW at the python the generation actually used (falling
    # back to the canonical independent env), compared against what the
    # generation manifests recorded AND against the version the env's own
    # lock pins (round-3 review fix 4)
    default_env = PROJECT_ROOT / "derived" / "upstream" / "holosoma_env" / ".venv" / "bin" / "python"
    env_python = Path(env_python) if env_python else Path(next(
        (p for p in env_pythons if p), str(default_env)))
    env_now = integ.measure_env_identity(env_python)
    recorded_locks = sorted(x for x in env_lock_shas if x)
    recorded_versions = sorted(x for x in env_versions if x)
    env_lock_consistent = (
        len(env_lock_shas) <= 1
        and None not in env_lock_shas
        and env_now["env_uv_lock_sha256"] in env_lock_shas
    )
    env_version_consistent = (
        len(env_versions) == 1
        and None not in env_versions
        and env_now["env_mujoco_version"] is not None
        and env_now["env_mujoco_version"] in env_versions
        and env_now["env_lock_mujoco_version"] == env_now["env_mujoco_version"]
    )
    gen_checkout_ok = all(
        r["matches_freeze"] and r["recorded_clean"] for r in gen_checkout.values())
    chain_ok = all(r["passed"] for r in chain.values()) if chain else False
    source_reg = None
    if source_record is not None:
        prereg, prereg_sha_now = source_record
        source_reg = _source_registration_record(
            u0_root, coverage, prereg, prereg_sha_now, env_python=env_python)
        # review fix 2: the registration leg is only as strong as the
        # argv-bound smokes — at least one consumed row's actual recorded
        # command must have been re-parsed in the CURRENT environment
        smoked = any(bool(r["argv_smoke"])
                     for r in source_reg["per_attempt"].values())
        source_reg["argv_smoke_any"] = bool(smoked)
        if not smoked:
            source_reg["passed"] = False
    admission = {
        "checkout_vs_freeze": checkout,
        "assets_vs_freeze": assets_check,
        "generation_checkout": gen_checkout,
        "env_lock": {
            "measured_now": env_now["env_uv_lock_sha256"],
            "recorded_at_generation": recorded_locks,
            "consistent": env_lock_consistent,
        },
        "env_mujoco": {
            "probed_now": env_now["env_mujoco_version"],
            "lock_pinned": env_now["env_lock_mujoco_version"],
            "recorded_at_generation": recorded_versions,
            "probe_error": env_now["env_probe_error"],
            "consistent": env_version_consistent,
        },
        "inputs_rehashed": {"all_match": inputs_ok, "by_demo": inputs_detail},
        "inputs_vs_freeze": inputs_vs_freeze,
        "data_paths": data_paths,
        "artifact_chain": chain,
        "binding_recomputed": binding,
        f"{source_label.lower()}_coverage": {
            "missing": coverage["missing"],
            "by_demo": {d: e["attempt"] for d, e in coverage["by_demo"].items()}},
        "passed": (checkout["passed"] and assets_check["passed"]
                   and freeze_inputs_ok and gen_checkout_ok
                   and env_lock_consistent and env_version_consistent
                   and inputs_ok and chain_ok
                   and all(binding_passed.values())
                   and (source_reg is None or source_reg["passed"])),
    }
    # U0 mode keeps its admission BYTE-IDENTICAL to the frozen baseline
    # reports — the registration leg exists only in UR mode (§10.3)
    if source_reg is not None:
        admission["source_registration"] = source_reg
    return admission, binding_passed, coverage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True, help="A1 freeze dir")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--u0-dir", type=Path,
                     help="I0 U0 root dir (attempts_index.json)")
    src.add_argument("--generated-dir", type=Path,
                     help="UR generation root (attempts_index.json); "
                          "requires --source-record")
    ap.add_argument("--source-record", type=Path, default=None,
                    help="UR preregistration.json (closure plan §7.3)")
    ap.add_argument("--out", type=Path, required=True, help="teacher_pilot output dir")
    args = ap.parse_args()
    if bool(args.generated_dir) != bool(args.source_record):
        raise SystemExit("--generated-dir and --source-record must be used "
                         "together (and never with --u0-dir)")
    t0 = time.time()

    freeze = args.freeze.expanduser().resolve()
    u0_root = (args.u0_dir or args.generated_dir).expanduser().resolve()
    if args.u0_dir is not None:
        # review fix 3: refuse a UR generation root in plain U0 mode BEFORE
        # anything else runs — the registration leg would be silently
        # skipped and the report would label the recovery source "U0"
        _assert_plain_u0_root(u0_root)
    out_dir = integ.guard_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)   # round-3 fix 8: a fresh
    # --out path is legal under derived/ and must not crash the report write

    lists = json.loads((freeze / "frozen_lists.json").read_text())
    pilot = [e["demo"] for e in lists["teacher_pilot_train6_v1"]]
    source_boxes = load_source_boxes(freeze)

    source_record = None
    source_label = "U0"
    if args.source_record is not None:
        prereg = json.loads(args.source_record.expanduser().resolve().read_text())
        if (prereg.get("experiment_id") != "a2_recovery_step03_v1"
                or list(prereg.get("expected_demos", [])) != pilot):
            raise SystemExit("preregistration identity mismatch vs the "
                             f"frozen pilot: {args.source_record}")
        source_record = (prereg,
                         integ.sha256_file(args.source_record.expanduser().resolve()))
        source_label = "UR"

    admission, binding_passed, coverage = admission_record(
        freeze, u0_root, pilot, source_record=source_record,
        source_label=source_label)

    def load_p(demo):
        return orc.load_teacher_scene(TEACHER_TRAJ_DIR / f"{demo}_original.npz", "published")

    u0_by_demo = coverage["by_demo"]

    def load_u0(demo):
        return orc.load_teacher_scene(
            Path(u0_by_demo[demo]["converted"]), "project_scene")

    print(f"[a2] replaying P (published) on {len(pilot)} demos ...")
    replayer = TeacherQualityReplayer()
    p_metrics = measure_source(replayer, pilot, load_p, source_boxes)
    print(f"[a2] replaying {source_label} (generated) ...")
    u_metrics = {}
    for demo in pilot:
        if demo in u0_by_demo:
            u_metrics.update(
                measure_source(replayer, [demo], load_u0, source_boxes))
        else:
            u_metrics[demo] = DemoMetrics(
                demo, -1, float("nan"), valid=False,
                error=f"no OK {source_label} artifact in any attempt")

    verdict = evaluate_local_preferred(pilot, p_metrics, u_metrics,
                                       binding_passed, u_label=source_label)
    # Admission failures are verdict-level: a wrong checkout, a drifted env
    # (lock or binary), mutated inputs or assets, a broken artifact chain or
    # a failed binding recheck must NEVER surface as local_preferred with
    # complete evidence (review fixes 5 and 3/4).
    if not admission["passed"]:
        reasons = []
        if not admission["checkout_vs_freeze"]["passed"]:
            reasons.append("checkout_identity_mismatch")
        if not admission["assets_vs_freeze"]["passed"]:
            reasons.append("frozen_assets_drifted")
        reasons += [f"generation_checkout:{d}"
                    for d, r in admission["generation_checkout"].items()
                    if not (r["matches_freeze"] and r["recorded_clean"])]
        if not admission["env_lock"]["consistent"]:
            reasons.append("env_lock_drift_vs_generation")
        if not admission["env_mujoco"]["consistent"]:
            reasons.append("env_mujoco_version_inconsistent")
        if not admission["inputs_rehashed"]["all_match"]:
            reasons.append("input_hash_mismatch_vs_generation")
        if not admission["inputs_vs_freeze"]["all_match"]:
            reasons.append("input_hash_mismatch_vs_freeze")
        reasons += [f"artifact_chain_failed:{d}"
                    for d, r in admission["artifact_chain"].items() if not r["passed"]]
        reasons += [f"{d}:binding_recheck_failed"
                    for d, ok in binding_passed.items() if not ok]
        # review fix 12 (2026-10-10): a failed registration/config
        # reconciliation leg must surface in evidence_gaps — the UR verdict
        # previously went incomplete+complete_evidence=false with an EMPTY
        # gap list, so nothing named WHY
        source_reg = admission.get("source_registration")
        if source_reg is not None and not source_reg.get("passed"):
            reasons.append("source_registration_reconciliation_failed")
            for att, r in source_reg.get("per_attempt", {}).items():
                if not r.get("identity_matches"):
                    reasons.append(f"source_registration:{att}:identity")
                if not r.get("single_difference_holds"):
                    reasons.append(f"source_registration:{att}:config_diff")
                if not r.get("argv_consistent_across_rows"):
                    reasons.append(
                        f"source_registration:{att}:argv_inconsistent")
                for demo, err in (r.get("argv_parse_errors") or {}).items():
                    reasons.append(
                        f"source_registration:{att}:argv_parse:{demo}:{err}")
                if r.get("argv_smoke") and not r.get("argv_smoke_ok"):
                    reasons.append(f"source_registration:{att}:argv_smoke")
        verdict["evidence_gaps"] = list(verdict["evidence_gaps"]) + [
            f"admission:{r}" for r in reasons]
        verdict["complete_evidence"] = False
        verdict["verdict"] = "incomplete"
    selection = {
        "stage": (f"A2 teacher source comparison (plan §3.3-§3.4; "
                  f"{source_label} = {source_record[0]['source_id']})"
                  if source_record else
                  "A2 teacher source comparison (plan §3.3-§3.4)"),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pilot": pilot,
        "p_revision": "135259130158f1cb1799e4ab9b6566c938285fc9",
        "admission": admission,
        "u0_identity": {
            "holosoma_checkout": admission["checkout_vs_freeze"]["holosoma_revision"],
            "env_uv_lock_sha256": admission["env_lock"]["measured_now"],
            "env_mujoco_version": None,   # filled from the generation manifest below
            "generation_attempts": [a["attempt"] for a in integ.load_attempts(u0_root)],
        },
        "comparison_attempt": None,       # filled below (isolated attempts)
        "u1": "not_applicable (not pre-registered this round, §3.3)",
        "metrics": {
            "P": {d: p_metrics[d].to_json() for d in pilot},
            source_label: {d: u_metrics[d].to_json() for d in pilot},
        },
        "binding": binding_passed,
        "verdict_record": verdict,
        "verdict": verdict["verdict"],
        "residual_risks": (
            "residual defects below reference levels are allowed; local_preferred "
            "authorizes approximate motion labels for the small batch only (§3.4); "
            "an improvement in one metric family is never summarized as reliable grasp"
        ),
        "cost_s": round(time.time() - t0, 1),
    }
    # fill measured env facts from the newest attempt manifest
    attempts = integ.load_attempts(u0_root)
    if attempts:
        manifest = integ.load_attempt_manifest(u0_root, attempts[-1]["attempt"])
        selection["u0_identity"]["env_mujoco_version"] = (
            manifest.get("identity", {}).get("env", {}).get("env_mujoco_version"))
    if source_record is not None:
        # the UR report must not label the recovery source "U0": move the
        # identity block to its registered name and pin the registration
        selection["ur_identity"] = selection.pop("u0_identity")
        selection["ur_identity"].update({
            "experiment_id": source_record[0]["experiment_id"],
            "source_id": source_record[0]["source_id"],
            "preregistration_sha256": source_record[1],
            "step_size": source_record[0][
                "single_config_difference"]["retargeter.step_size"]["to"],
        })

    # Round-3 review fix 6: the comparison writes its own attempt directory
    # — an older teacher_selection.json is never overwritten, and an
    # interrupted run leaves the prior verdict untouched.
    cmp_root = out_dir / "cmp"
    attempt_dir = integ.allocate_attempt(cmp_root)
    selection["comparison_attempt"] = attempt_dir.name
    out = attempt_dir / "teacher_selection.json"
    out.write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    integ.append_attempts_index(cmp_root, {
        "attempt": attempt_dir.name,
        "verdict": selection["verdict"],
        "complete_evidence": selection["verdict_record"]["complete_evidence"],
        "report": path_for_record(out),
    })
    print(json.dumps({k: verdict[k] for k in ("complete_evidence", "verdict")}, indent=2))
    for name, g in verdict["gates"].items():
        print(f"  {name}: {'PASS' if g['passed'] else 'FAIL'}")
    if verdict["evidence_gaps"]:
        print(f"  evidence gaps: {verdict['evidence_gaps'][:6]}"
              f"{' ...' if len(verdict['evidence_gaps']) > 6 else ''}")
    print(f"teacher_selection -> {path_for_record(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
