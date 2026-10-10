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

P rows come from the published NPZ; U0 rows from the newest OK attempt
artifact per demo. Every expected demo keeps a row on both sides;
missing/invalid evidence forces ``incomplete`` (never a silent drop).
Fixed-input ``box_ground`` conflicts are recorded separately (§7.3). U1 is
not pre-registered in this round -> ``not_applicable`` (§3.3).

Run: uv run --locked python scripts/omniretarget_teacher_comparison.py \
       --freeze <a1_run>/freeze --u0-dir <a2_run>/teacher_pilot/u0 \
       --out <a2_run>/teacher_pilot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

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


def admission_record(freeze: Path, u0_root: Path, pilot: list[str],
                     env_python: Path | None = None) -> dict:
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
        # canonical dir and are labeled as such
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
        # inputs re-hashed NOW at the recorded data path against what
        # generation recorded; a missing file is an admission failure for
        # this demo, not a crash
        try:
            demo_inputs = integ.verify_inputs_against_manifest(
                manifest, TEACHER_TRAJ_DIR, pt_dir)
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
    try:
        frozen_entries = json.loads(
            (freeze / "frozen_lists.json").read_text())["teacher_pilot_train6_v1"]
        by_dir: dict[str, dict] = {}
        for label, d in sorted(pt_dirs.items()):
            try:
                by_dir[label] = integ.verify_inputs_against_freeze(
                    frozen_entries, TEACHER_TRAJ_DIR, d)
            except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                by_dir[label] = {"error": f"{type(exc).__name__}: {exc}"}

        def _dir_ok(res: dict) -> bool:
            return (bool(res) and "error" not in res
                    and all(r.get("npz_matches") and r.get("pt_matches")
                            for r in res.values()))

        freeze_inputs_ok = bool(by_dir) and all(_dir_ok(r) for r in by_dir.values())
        if len(by_dir) == 1:
            merged = next(iter(by_dir.values()))
        else:
            bad = [r for r in by_dir.values() if "error" in r]
            if bad:
                merged = {"error": "; ".join(r["error"] for r in bad)}
            else:
                merged = {
                    demo: {"npz_matches": all(
                               r.get(demo, {}).get("npz_matches")
                               for r in by_dir.values()),
                           "pt_matches": all(
                               r.get(demo, {}).get("pt_matches")
                               for r in by_dir.values())}
                    for demo in sorted({d for r in by_dir.values() for d in r})
                }
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
    return {
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
        "u0_coverage": {"missing": coverage["missing"],
                        "by_demo": {d: e["attempt"] for d, e in coverage["by_demo"].items()}},
        "passed": (checkout["passed"] and assets_check["passed"]
                   and freeze_inputs_ok and gen_checkout_ok
                   and env_lock_consistent and env_version_consistent
                   and inputs_ok and chain_ok
                   and all(binding_passed.values())),
    }, binding_passed, coverage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", type=Path, required=True, help="A1 freeze dir")
    ap.add_argument("--u0-dir", type=Path, required=True,
                    help="I0 U0 root dir (attempts_index.json)")
    ap.add_argument("--out", type=Path, required=True, help="teacher_pilot output dir")
    args = ap.parse_args()
    t0 = time.time()

    freeze = args.freeze.expanduser().resolve()
    u0_root = args.u0_dir.expanduser().resolve()
    out_dir = integ.guard_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)   # round-3 fix 8: a fresh
    # --out path is legal under derived/ and must not crash the report write

    lists = json.loads((freeze / "frozen_lists.json").read_text())
    pilot = [e["demo"] for e in lists["teacher_pilot_train6_v1"]]
    source_boxes = load_source_boxes(freeze)

    admission, binding_passed, coverage = admission_record(freeze, u0_root, pilot)

    def load_p(demo):
        return orc.load_teacher_scene(TEACHER_TRAJ_DIR / f"{demo}_original.npz", "published")

    u0_by_demo = coverage["by_demo"]

    def load_u0(demo):
        return orc.load_teacher_scene(
            Path(u0_by_demo[demo]["converted"]), "project_scene")

    print(f"[a2] replaying P (published) on {len(pilot)} demos ...")
    replayer = TeacherQualityReplayer()
    p_metrics = measure_source(replayer, pilot, load_p, source_boxes)
    print("[a2] replaying U0 (generated) ...")
    u_metrics = {}
    for demo in pilot:
        if demo in u0_by_demo:
            u_metrics.update(
                measure_source(replayer, [demo], load_u0, source_boxes))
        else:
            u_metrics[demo] = DemoMetrics(
                demo, -1, float("nan"), valid=False,
                error="no OK U0 artifact in any attempt")

    verdict = evaluate_local_preferred(pilot, p_metrics, u_metrics, binding_passed)
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
        verdict["evidence_gaps"] = list(verdict["evidence_gaps"]) + [
            f"admission:{r}" for r in reasons]
        verdict["complete_evidence"] = False
        verdict["verdict"] = "incomplete"
    selection = {
        "stage": "A2 teacher source comparison (plan §3.3-§3.4)",
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
            "U0": {d: u_metrics[d].to_json() for d in pilot},
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
