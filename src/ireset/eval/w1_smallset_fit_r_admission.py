"""W1 R-phase metadata admission and the real-data fit entry (plan v0.4).

R (§3.2/§6.3/§8) freezes the SOURCE binding, the six registered training
windows and the D1 window, the runtime metadata against the T6
registration, the code/config identity, the budgets and the exact call
matrix -- metadata only: JSON reads, file byte hashes and NPZ headers.
Array values, checkpoints, tokenizers, training modules and FK are NOT
constructed or decoded here.

Array materialization (§3.2) happens ONLY inside a started, not-exceeded
D1/D6/F billing interval (:func:`materialize_real_world` and the shared
:func:`materialize_fit_apparatus`); the real chain (:func:`run_real_chain`)
reuses the audited D1/D6 machinery (``FitDriver`` over the shared
``Trainer`` update units) and the F verification unchanged, and adds the
§8.3 authorization gate, the per-stage §8.1 budgets and the real SOURCE
world.

Nothing here starts a real stage by itself: the chain entry refuses to run
without a sealed admission package and a §8.3 authorization whose
checklist is entirely true.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ireset.data import w1_training as w1
from ireset.data.teacher import (build_interaction_clip, keypoints_in_source_object,
                                 world_to_clip)
from ireset.embodied.g1_spec import G1Spec
from ireset.eval.w1_n0_coordinate_flags import (
    _write_summary_sealed, npz_members, sha256_file, verify_return_coverage)
from ireset.eval.w1_real_fk_audit import RUNTIME_REQUIREMENTS
from ireset.eval.w1_real_training_update_audit import (
    runtime_identity_facts, runtime_identity_problems)
from ireset.eval.w1_smallset_fit_diagnostic import (
    BUDGET_CAPS, COND_KEYS, FIT_STEPS, F_WITHDRAW_ON_EXCEED, FitDiagnosticError,
    FitDriver, FitPlan, FitRegistration, I_WITHDRAW_ON_EXCEED, J_JOINTS,
    LossWeights, MetricDecoder, MetadataOnlyReader, PLAN_D_MODEL, PLAN_FFN_MULT,
    PLAN_GRAD_CLIP, PLAN_LR, PLAN_N_HEADS, PLAN_N_LAYERS, PLAN_SEED,
    PLAN_TIME_WINDOW, PLAN_VAL_SEED, PLAN_WEIGHT_DECAY, PROBE_EVERY, RootStats,
    SPIKE_GATE_STATUS, StageBudget, StateTokenizer, T_WINDOW, Trainer,
    TrainConfig, WindowKey, _observe_returned_report, budget_for_stage,
    canonical_window_order, expected_call_matrix, finish_stage,
    frozen_plan_defaults_problems, materialize_for_fit, resolve_handoff,
    run_f_verification, write_acceptance_outputs)
from ireset.kinematics.torch_adapter import G1TorchFK
from ireset.model.tokens import RootStats as _RootStats  # noqa: F401
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, path_for_record
from ireset.train.losses import LossAssembler

# --------------------------------------------------------------------------
# frozen constants (plan §3.2/§8)
# --------------------------------------------------------------------------

R_ADMISSION_SCHEMA = "w1_smallset_fit_r_admission_v1"
ADMISSION_PACKAGE_SCHEMA = "w1_smallset_fit_admission_package_v1"
AUTHORIZATION_SCHEMA = "w1_smallset_fit_real_authorization_v1"

#: §3.2: the ONLY read-only SOURCE for the fit diagnostic.
SOURCE_ROOT_REL = ("derived/experiments/omniretarget_first/"
                   "20261012_21e5784_w1_a3bc_reviewfix8/w1_reviewfix8")
#: §6.3: the T6-registered runtime metadata R compares against.
T6_REGISTRATION_REL = ("derived/experiments/omniretarget_first/"
                       "20261102_2704709_w1_real_training_update_audit/"
                       "registration/attempt_003/registration.json")
#: §8.3: the new cohort is only CREATED after the frozen materials and the
#: explicit authorization exist; R only declares the identity.
COHORT_ID = "w1_patched_smallset_fit_v1"

R_ARTIFACT_CAP_BYTES = 2 * (1 << 30)
R_DISK_MARGIN_BYTES = 5 * (1 << 30)
R_REQUIRED_DISK_FREE_BYTES = R_ARTIFACT_CAP_BYTES + R_DISK_MARGIN_BYTES

R_TOTALS = {"n0_n1_total_s": 1800.0, "fit_stages_total_s": 6120.0,
            "grand_total_s": 7920.0}

R_WITHDRAW_ON_EXCEED = {"admission_passed": False,
                        "label_verdict": "budget_exceeded_after_seal"}
CHAIN_WITHDRAW_ON_EXCEED = {"kind": "real_fit_chain",
                            "label_verdict": "budget_exceeded_after_seal"}

#: §8.3 checklist: every entry must be exactly ``True`` in the authorization.
AUTHORIZATION_CHECKLIST = (
    "research_question_new_limited_fit_capability",
    "numerics_closure_and_legacy_gaps_reviewed",
    "final_bindings_windows_runtime_call_matrix_reviewed",
    "cohort_budget_attempt_and_stop_rules_declared",
    "checkpoint_and_f_replay_nonzero_calls_acknowledged",
    "ddim_val_test_and_production_boundaries_acknowledged",
)

#: The executable surface the real chain binds (T6 pattern: file bytes, not
#: git state).  Documentation commits do not invalidate the code identity.
R_IMPLEMENTATION_FILES = (
    "src/ireset/eval/w1_smallset_fit_diagnostic.py",
    "src/ireset/eval/w1_smallset_fit_r_admission.py",
    "scripts/run_w1_smallset_fit_real.py",
    "scripts/run_w1_smallset_fit_real_supervisor.py",
    "tests/test_w1_smallset_fit_diagnostic.py",
    "tests/test_w1_smallset_fit_r_admission.py",
    "src/ireset/train/trainer.py",
    "src/ireset/train/losses.py",
    "src/ireset/model/backbone.py",
    "src/ireset/model/conditioner.py",
    "src/ireset/model/deterministic.py",
    "src/ireset/model/tokens.py",
    "src/ireset/diffusion/generator.py",
    "src/ireset/kinematics/torch_adapter.py",
    "src/ireset/kinematics/rotation_adapter.py",
    "src/ireset/geometry/distance_adapter.py",
    "src/ireset/embodied/g1_spec.py",
    "src/ireset/eval/w1_real_fk_audit.py",
    "src/ireset/data/windows.py",
    "src/ireset/data/w1_training.py",
    "src/ireset/data/teacher.py",
    "src/ireset/paths.py",
    "configs/train/omniretarget_motion.yaml",
    "pyproject.toml",
    "uv.lock",
)

#: The config values the diagnostic REUSES from the production config file;
#: steps/micro_batch are diagnostic-frozen (1000 / 1 or 6) and deliberately
#: NOT the yaml's formal-training values.
CONFIG_REUSED_FIELDS = {
    "lr": PLAN_LR, "weight_decay": PLAN_WEIGHT_DECAY,
    "grad_clip": PLAN_GRAD_CLIP, "seed": PLAN_SEED, "val_seed": PLAN_VAL_SEED,
    "d_model": PLAN_D_MODEL, "n_layers": PLAN_N_LAYERS,
    "n_heads": PLAN_N_HEADS, "ffn_mult": PLAN_FFN_MULT,
    "time_window": PLAN_TIME_WINDOW,
}


def canonical_sha256(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------
# exact call matrix (plan §8.2)
# --------------------------------------------------------------------------

def _world_construction_counts() -> dict:
    """Construction classes of ONE billed-world materialization.

    ``largebox_scene_access`` is exact per stage (the first access compiles
    the locked scene, later accesses hit the process cache; the compile-
    vs-hit fact is recorded honestly at run time)."""
    return {"root_stats_loads": 1, "tokenizer_builds": 1,
            "loss_assembler_constructions": 1, "g1_fk_constructions": 1,
            "g1_spec_constructions": 2, "largebox_scene_access": 1}


def frozen_call_matrix() -> dict:
    """The §8.2 exact per-stage call matrix for the authorized real chain.

    Training classes are EXACT (shared definition with the driver's
    run-time enforcement); F's saved-quantity recompute classes are caps
    that are exact when both stages complete (the actual counts are
    recorded and judged at run time)."""
    d1, d6 = FitPlan("D1"), FitPlan("D6")
    d1_matrix = {**expected_call_matrix(d1, 1), **_world_construction_counts(),
                 "registered_window_reads": 1, "window_x0_encodes": 1,
                 "model_constructions": 1, "conditioner_constructions": 1,
                 "optimizer_constructions": 1, "checkpoint_loads": 0}
    d6_matrix = {**expected_call_matrix(d6, 6), **_world_construction_counts(),
                 "registered_window_reads": 6, "window_x0_encodes": 6,
                 "model_constructions": 1, "conditioner_constructions": 1,
                 "optimizer_constructions": 1, "checkpoint_loads": 0}
    f_matrix = {**_world_construction_counts(),
                "registered_window_reads": 0, "window_x0_encodes": 0,
                "model_constructions": 2, "conditioner_constructions": 2,
                "optimizer_constructions": 0, "checkpoint_loads": 4,
                "fresh_probe_calls": 4, "probe_calls": 4,
                "conditioner_forward_calls": 4, "backbone_forward_calls": 4,
                "production_fk_batch_calls": 14,
                "loss_calls": 0, "backward_calls": 0, "clip_calls": 0,
                "adamw_step_calls": 0, "training_updates": 0,
                "window_consumptions_total": 0,
                "production_decode_calls": 728, "metric_decode_calls": 742}
    totals = {
        "training_updates": d1_matrix["adamw_step_calls"]
        + d6_matrix["adamw_step_calls"],
        "training_micros": d1_matrix["loss_calls"] + d6_matrix["loss_calls"],
        "window_consumptions_total": (d1_matrix["window_consumptions_total"]
                                      + d6_matrix["window_consumptions_total"]),
        "probe_calls": d1_matrix["probe_calls"] + d6_matrix["probe_calls"],
        "conditioner_forward_calls": (d1_matrix["conditioner_forward_calls"]
                                      + d6_matrix["conditioner_forward_calls"]
                                      + f_matrix["conditioner_forward_calls"]),
        "backbone_forward_calls": (d1_matrix["backbone_forward_calls"]
                                   + d6_matrix["backbone_forward_calls"]
                                   + f_matrix["backbone_forward_calls"]),
    }
    matrix = {"D1": d1_matrix, "D6": d6_matrix, "F": f_matrix,
              "totals": totals}
    matrix["sha256"] = canonical_sha256(
        {k: v for k, v in matrix.items() if k != "sha256"})
    return matrix


# --------------------------------------------------------------------------
# R metadata admission (plan §3.2/§6.3/§8)
# --------------------------------------------------------------------------

def _file_record(path: Path) -> dict:
    return {"path": path_for_record(Path(path)), "sha256": sha256_file(Path(path))}


def _npz_header_record(path: Path) -> dict:
    members = npz_members(Path(path))
    return {"sha256": sha256_file(Path(path)),
            "members": {k: [list(v[0]), v[1]] for k, v in members.items()}}


def _git_identity(root: Path) -> dict:
    import subprocess

    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD"],
                              capture_output=True, check=True).stdout
        return {"head": head, "dirty": bool(status.strip()),
                "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None, "status_sha256": None,
                "tracked_diff_sha256": None}


def implementation_identity(project_root: Path) -> dict:
    root = Path(project_root)
    files = {}
    for rel in R_IMPLEMENTATION_FILES:
        p = root / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    missing = sorted(rel for rel, sha in files.items() if sha is None)
    return {"schema": "w1_smallset_fit_implementation_v1", "files": files,
            "missing_files": missing,
            "implementation_files_sha256": canonical_sha256({"files": files})}


def config_cross_check_problems(project_root: Path) -> list[str]:
    """The yaml values the diagnostic reuses must equal the frozen plan."""
    import yaml

    path = Path(project_root) / "configs/train/omniretarget_motion.yaml"
    cfg = yaml.safe_load(path.read_text())
    problems = []
    for key, want in CONFIG_REUSED_FIELDS.items():
        got = cfg.get(key)
        if got != want:
            problems.append(f"config_{key}:{got!r}:want{want!r}")
    loss = cfg.get("loss") or {}
    from ireset.eval.w1_smallset_fit_diagnostic import LOSS_WEIGHT_EXPECTED
    for key, want in LOSS_WEIGHT_EXPECTED.items():
        if loss.get(key) != want:
            problems.append(f"config_loss_{key}:{loss.get(key)!r}:want{want!r}")
    return problems


def select_windows(prereg: dict, manifest: dict,
                   frozen_lists: dict) -> tuple[list[WindowKey], list[str]]:
    """§3.2 window selection: ``pilot6 ∩ train24``, one start=0 window per
    demo.  Reads ONLY registration keys (split / admission.motion / T /
    full_windows) -- never fit, collision or contact results, never
    val/test demos."""
    problems: list[str] = []
    pilot6 = (prereg.get("selection") or {}).get("pilot6_from_a1")
    if not isinstance(pilot6, list) or len(pilot6) != 6 \
            or len(set(pilot6)) != 6 or not all(isinstance(d, str) for d in pilot6):
        problems.append(f"r_pilot6_invalid:{type(pilot6).__name__}")
        return [], problems
    listed = [e["demo"] for e in (frozen_lists or {}).get(
        "teacher_pilot_train6_v1", [])]
    if sorted(listed) != sorted(pilot6):
        problems.append("r_pilot6_not_bound_to_a1_frozen_lists")
    train24 = (prereg.get("selection") or {}).get("train24")
    if (manifest.get("selection") or {}).get("train24") != train24:
        problems.append("r_manifest_train24_mismatch_vs_preregistration")
    if isinstance(train24, list) and w1.canonical_list_sha256(list(train24)) != \
            (prereg.get("selection") or {}).get("train24_canonical_sha256"):
        problems.append("r_train24_canonical_sha_mismatch")
    rows = {r.get("demo"): r for r in manifest.get("rows", [])
            if isinstance(r, dict)}
    selected: list[WindowKey] = []
    for demo in pilot6:
        row = rows.get(demo)
        if row is None:
            problems.append(f"r_pilot_demo_missing:{demo}")
            continue
        if row.get("split") != "train":
            problems.append(
                f"r_pilot_split_not_train:{demo}:{row.get('split')!r}")
        if isinstance(train24, list) and demo not in set(train24):
            problems.append(f"r_pilot_not_in_train24:{demo}")
        adm = row.get("admission") or {}
        if adm.get("motion") is not True:
            problems.append(f"r_pilot_not_admitted:{demo}")
        t_val = row.get("T")
        if not isinstance(t_val, int) or isinstance(t_val, bool) \
                or t_val < T_WINDOW:
            problems.append(f"r_pilot_window_too_short:{demo}:{t_val!r}")
        if not (row.get("full_windows_F60_s30") or 0) >= 1:
            problems.append(f"r_pilot_no_full_window:{demo}")
        selected.append(WindowKey(demo, 0, T_WINDOW))
    if problems:
        return [], problems
    windows = list(canonical_window_order(selected))
    if len(windows) != 6:
        problems.append(f"r_window_selection_failed:{len(windows)}")
        return [], problems
    return windows, problems


def _resolve_cache_root(source_root: Path, cache_root: Path | None) -> tuple[Path, Path]:
    """(bound cache attempt dir, the SOURCE's own registered cache dir)."""
    source_cache = Path(source_root) / "cache/attempt_001"
    return (Path(cache_root) if cache_root is not None else source_cache,
            source_cache)


def _migration_record_problems(root: Path, source_cache: Path,
                               cache_root: Path, bound_identity: dict,
                               ) -> list[str]:
    """§3.2 revision: a cache that REPLACES the SOURCE's registered one is
    never a default substitution -- it must carry a migration record proving
    bitwise value equality against the registered cache (the rebuild writes
    it from the recorded reference identity SHA)."""
    if cache_root == source_cache:
        return []
    rec = bound_identity.get("migration_check")
    if not isinstance(rec, dict):
        return ["r_cache_migration_record_missing"]
    if rec.get("passed") is not True:
        return ["r_cache_migration_not_passed"]
    reference = rec.get("reference_cache") or {}
    registered_identity = source_cache / "cache_identity.json"
    if not registered_identity.is_file() or \
            reference.get("sha256") != sha256_file(registered_identity) or \
            reference.get("path") != path_for_record(registered_identity):
        return ["r_cache_migration_reference_mismatch"]
    if not isinstance(rec.get("per_demo"), dict) or not rec["per_demo"]:
        return ["r_cache_migration_per_demo_missing"]
    bad = [d for d, v in rec["per_demo"].items()
           if not isinstance(v, dict) or v.get("bitwise_equal") is not True]
    if bad:
        return [f"r_cache_migration_demo_not_equal:{sorted(bad)[:4]}"]
    return []


def run_r_admission(*, out_dir: Path, budget: StageBudget,
                    project_root: Path | None = None,
                    source_root: Path | None = None,
                    teacher_npz_dir: Path | None = None,
                    omomo_pt_dir: Path | None = None,
                    t6_registration: Path | None = None,
                    reader: MetadataOnlyReader | None = None,
                    runtime_probe: Callable[[], dict] | None = None,
                    disk_probe: Callable[[Path], int] | None = None,
                    git_identity: Callable[[Path], dict] | None = None,
                    cache_root: Path | None = None,
                    ) -> dict:
    """One R window: metadata admission under the §8.1 R budget (120s/15s).

    Returns the sealed on-disk document; ``admission_passed`` is True only
    when every binding leg verified.  No array value, checkpoint,
    tokenizer, training module or FK is touched -- the reader refuses them
    structurally and the zero declarations are written into the sealed
    summary.  ``cache_root`` binds a REPLACEMENT cache (plan §3.2 revision,
    §41): the SOURCE's registered cache stays the identity-chain root, the
    bound cache must be consumable under the CURRENT code and carry a valid
    migration record."""
    reader = reader or MetadataOnlyReader()
    root = Path(project_root or PROJECT_ROOT)
    source_root = Path(source_root or root / SOURCE_ROOT_REL)
    cache_root, source_cache = _resolve_cache_root(source_root, cache_root)
    # the T6 registration is a REAL project artifact, independent of any
    # fixture root the caller passes for tests
    t6_registration = Path(t6_registration or PROJECT_ROOT / T6_REGISTRATION_REL)
    out_dir = Path(out_dir)
    # the supervisor reserves the attempt directory (launch record + logs
    # land there first); only the R artifacts themselves must not exist yet
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("r_admission.json", "return_observation.json"):
        if (out_dir / name).exists():
            raise FitDiagnosticError(f"r_output_reserved_unavailable:{name}")
    facts: dict = {"schema": R_ADMISSION_SCHEMA, "kind": "r_admission",
                   "cohort_id": COHORT_ID, "stage": "R",
                   "source_root": path_for_record(source_root)}
    problems: list[str] = []
    package: dict = {"schema": ADMISSION_PACKAGE_SCHEMA, "cohort_id": COHORT_ID}
    try:
        budget.check("r_entry")

        # -- SOURCE identity chain (cache_identity -> prereg/manifest) ------
        budget.check("r_source_binding")
        source_identity_path = source_cache / "cache_identity.json"
        source_chain_identity = reader.read_json(source_identity_path)
        source_binding: dict = {"registered_cache_identity":
                                _file_record(source_identity_path)}
        # the preregistration/manifest binding chain starts at the SOURCE's
        # own registered cache identity, even when a replacement cache is
        # bound for schedules/root_stats (plan §41)
        cache_identity_path = cache_root / "cache_identity.json"
        cache_identity = reader.read_json(cache_identity_path)
        prereg_rec = source_chain_identity.get("preregistration") or {}
        manifest_rec = source_chain_identity.get("training_manifest") or {}
        prereg_path = root / str(prereg_rec.get("path", ""))
        manifest_path = root / str(manifest_rec.get("path", ""))
        prereg: dict = {}
        manifest: dict = {}
        if not prereg_path.is_file() or sha256_file(prereg_path) != \
                prereg_rec.get("sha256"):
            problems.append("r_preregistration_binding_broken")
        else:
            prereg = reader.read_json(prereg_path)
            source_binding["preregistration"] = _file_record(prereg_path)
        if not manifest_path.is_file() or sha256_file(manifest_path) != \
                manifest_rec.get("sha256"):
            problems.append("r_training_manifest_binding_broken")
        else:
            manifest = reader.read_json(manifest_path)
            source_binding["training_manifest"] = _file_record(manifest_path)

        # -- audited registration + cache re-verification (byte hashes) -----
        budget.check("r_registration_reverify")
        npz_dir = Path(teacher_npz_dir
                       or root / "resources/data/omniretarget/robot-object")
        pt_dir = Path(omomo_pt_dir or OMOMO_NEW_DIR)
        prereg_check = w1.verify_preregistration(prereg, root, npz_dir, pt_dir,
                                                 verify_assets=True)
        if not prereg_check["passed"]:
            problems.append(f"r_preregistration_reverify_failed:"
                            f"{prereg_check['problems'][:6]}")
        cache_check = w1.verify_cache_contents(cache_root, manifest, prereg, root)
        if not cache_check["passed"]:
            problems.append(
                f"r_cache_reverify_failed:{cache_check['problems'][:6]}")
        migration_problems = _migration_record_problems(
            root, source_cache, cache_root, cache_identity)
        problems += migration_problems
        source_binding["preregistration_verification"] = {
            "passed": prereg_check["passed"],
            "problems": prereg_check["problems"][:10]}
        source_binding["cache_verification"] = {
            "passed": cache_check["passed"],
            "problems": cache_check["problems"][:10]}
        source_binding["cache_migration_problems"] = migration_problems
        a1_assets_path = root / str((prereg.get("assets") or {})
                                    .get("a1_assets_path", ""))
        if a1_assets_path.is_file():
            source_binding["a1_assets"] = _file_record(a1_assets_path)

        # -- window selection + per-window binding ---------------------------
        budget.check("r_window_selection")
        lists_rec = (prereg.get("anchors") or {}).get("a1_frozen_lists") or {}
        lists_path = root / str(lists_rec.get("path", ""))
        frozen_lists = reader.read_json(lists_path) if lists_path.is_file() else {}
        if lists_path.is_file():
            source_binding["a1_frozen_lists"] = _file_record(lists_path)
        windows, sel_problems = select_windows(prereg, manifest, frozen_lists)
        problems += sel_problems
        rows = {r.get("demo"): r for r in manifest.get("rows", [])
                if isinstance(r, dict)}
        schedule_rows = {rec.get("demo"): rec
                         for rec in cache_identity.get("schedule_rows", [])}
        window_bindings = []
        for w in windows:
            budget.check(f"r_window_binding_{w.demo}")
            row = rows.get(w.demo) or {}
            inputs = row.get("inputs") or {}
            derived = row.get("derived") or {}
            sched = schedule_rows.get(w.demo) or {}
            rec = {"demo": w.demo, "start": w.start, "end": w.end,
                   "window_id": w.window_id, "T": row.get("T"),
                   "split": row.get("split"), "fps": row.get("fps"),
                   "full_windows": row.get("full_windows_F60_s30"),
                   "npz_sha256": inputs.get("npz_sha256"),
                   "pt_sha256": inputs.get("pt_sha256"),
                   "derived_path": derived.get("path"),
                   "derived_sha256": derived.get("sha256"),
                   "schedule_key": sched.get("key"),
                   "schedule_sha256": sched.get("sha256"),
                   "schedule_full_windows": sched.get("full_windows")}
            derived_path = root / str(derived.get("path", ""))
            if derived_path.is_file():
                rec["derived_members"] = npz_members(derived_path)
                if sha256_file(derived_path) != derived.get("sha256"):
                    problems.append(f"r_window_derived_sha_mismatch:{w.demo}")
            else:
                problems.append(f"r_window_derived_missing:{w.demo}")
            sched_path = cache_root / "schedules" / w1.schedule_cache_filename(
                w.demo, sched.get("key") or "")
            if sched_path.is_file():
                rec["schedule_members"] = npz_members(sched_path)
                if sha256_file(sched_path) != sched.get("sha256"):
                    problems.append(f"r_window_schedule_sha_mismatch:{w.demo}")
            else:
                problems.append(f"r_window_schedule_missing:{w.demo}")
            window_bindings.append(rec)
        package["windows"] = window_bindings
        package["d1_window_id"] = windows[0].window_id if windows else None
        package["selection_rule"] = (
            "pilot6 (SOURCE preregistration selection.pilot6_from_a1, "
            "cross-checked against the A1 frozen teacher_pilot_train6_v1) "
            "intersected with the registered train24 manifest rows; one "
            "start=0 end=60 window per demo; D1 = canonical lexicographic "
            "first. Selection reads only registration keys (split / "
            "admission.motion / T / full_windows) -- never fit, collision "
            "or contact results, never val/test demos.")
        package["source_binding"] = source_binding
        package["cache_binding"] = {
            "path": path_for_record(cache_root),
            "cache_identity": _file_record(cache_identity_path),
            "is_source_registered_cache": cache_root == source_cache,
            "condition_code_sha256": cache_identity.get("condition_code_sha256"),
            "schedule_key_version": cache_identity.get("schedule_cache_key_version"),
            "migration_check": cache_identity.get("migration_check"),
            "migration_problems": migration_problems}
        rs_rec = cache_identity.get("root_stats") or {}
        rs_path = cache_root / "root_stats.npz"
        if rs_path.is_file():
            package["root_stats"] = _npz_header_record(rs_path)
            if sha256_file(rs_path) != rs_rec.get("file_sha256"):
                problems.append("r_root_stats_sha_mismatch")

        # -- §6.3 runtime metadata (T6 comparison, requirements binding) ----
        budget.check("r_runtime_binding")
        t6_doc = reader.read_json(t6_registration)
        reference = t6_doc.get("runtime_identity") or {}
        current = (runtime_probe or runtime_identity_facts)()
        identity_problems = runtime_identity_problems(reference, current)
        problems += [f"r_runtime_identity:{p}" for p in identity_problems]
        import ireset.eval.w1_real_fk_audit as fk_audit
        package["runtime_binding"] = {
            "t6_registration": _file_record(t6_registration),
            "reference_identity": dict(reference),
            "current_identity": dict(current),
            "identity_problems": identity_problems,
            "requirements": dict(RUNTIME_REQUIREMENTS),
            "requirements_module": {
                "name": "ireset.eval.w1_real_fk_audit",
                "source_sha256": sha256_file(Path(fk_audit.__file__))},
            "apply_entry": ("configure_production_runtime() must be applied "
                            "and re-verified inside EACH D1/D6/F billing "
                            "window before any materialization, construction "
                            "or forward; CUDA/float32 only"),
        }

        # -- code/config identity + budget + matrix + disk ------------------
        budget.check("r_code_identity")
        git = (git_identity or _git_identity)(root)
        impl = implementation_identity(root)
        problems += [f"r_implementation_file_missing:{rel}"
                     for rel in impl["missing_files"]]
        cfg_problems = config_cross_check_problems(root)
        problems += [f"r_config:{p}" for p in cfg_problems]
        plan_problems = frozen_plan_defaults_problems()
        problems += [f"r_frozen_plan_defaults:{p}" for p in plan_problems]
        package["code_identity"] = {
            "implementation_files": impl["files"],
            "implementation_files_sha256": impl["implementation_files_sha256"],
            "missing_files": impl["missing_files"],
            "git": git,
            "condition_code_sha256": w1.condition_code_sha256(root),
            "config_yaml": _file_record(
                root / "configs/train/omniretarget_motion.yaml"),
            "config_cross_check_problems": cfg_problems,
            "frozen_plan_defaults_problems": plan_problems,
        }
        budget.check("r_budget_matrix")
        package["budget"] = {
            "caps": {stage: {"wall_cap_s": cap, "reserve_s": reserve,
                             "work_window_s": cap - reserve}
                     for stage, (cap, reserve) in BUDGET_CAPS.items()},
            "totals": dict(R_TOTALS)}
        package["call_matrix"] = frozen_call_matrix()
        budget.check("r_disk_check")
        free = (disk_probe or (lambda p: shutil.disk_usage(p).free))(root)
        disk_ok = free >= R_REQUIRED_DISK_FREE_BYTES
        if not disk_ok:
            problems.append(f"r_disk_free_insufficient:{free}:"
                            f"want>={R_REQUIRED_DISK_FREE_BYTES}")
        package["disk"] = {"free_bytes": free,
                           "required_bytes": R_REQUIRED_DISK_FREE_BYTES,
                           "artifact_cap_bytes": R_ARTIFACT_CAP_BYTES,
                           "passed": disk_ok}
        package["authorization_template"] = {
            "schema": AUTHORIZATION_SCHEMA, "authorized": False,
            "template_only": True,
            "checklist": {k: None for k in AUTHORIZATION_CHECKLIST},
            "bindings_required": [
                "admission_package_sha256 (the sealed r_admission.json bytes)",
                "cohort_id", "includes (subset of D1/D6/F, D1 required)",
                "budget_caps (equal to package budget.caps)",
                "call_matrix_sha256 (equal to package call_matrix.sha256)",
                "implementation_files_sha256 (equal to package code_identity)"]}
    except FitDiagnosticError as exc:
        problems.append(f"r_refused:{exc}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"r_error:{type(exc).__name__}:{str(exc)[:200]}")
    facts["package"] = package
    facts["problems"] = problems
    facts["admission_passed"] = not problems
    facts["metadata_reads"] = {"json_reads": len(reader.reads),
                               "read_log_tail": reader.reads[-12:]}
    facts["zero_declarations"] = {
        "array_value_reads": 0, "checkpoint_reads": 0, "torch_load_calls": 0,
        "tokenizer_constructs": 0, "model_constructs": 0, "fk_constructs": 0,
        "forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
        "cohort_created": False, "attempt_created": False,
        "claim_created": False, "authorization_issued": False}
    facts["budget"] = budget.facts()
    path = out_dir / "r_admission.json"
    seal = _write_summary_sealed(path, facts, budget,
                                 withdraw=dict(R_WITHDRAW_ON_EXCEED),
                                 clock=budget.clock)
    coverage = verify_return_coverage(path, budget,
                                      withdraw=dict(R_WITHDRAW_ON_EXCEED),
                                      clock=budget.clock)
    doc = coverage.get("final_doc")
    if doc is None:
        doc = json.loads(path.read_text())
    return {"summary": str(path), "seal": seal, "return_coverage": coverage,
            "summary_doc": doc}


# --------------------------------------------------------------------------
# admission consumption (D1/D6/F entries) + §8.3 authorization gate
# --------------------------------------------------------------------------

def load_admission(path: Path) -> dict:
    """Load a sealed admission document; refuse anything not consumable."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("schema") != R_ADMISSION_SCHEMA:
        raise FitDiagnosticError("admission_schema_invalid")
    seal = doc.get("seal") or {}
    if seal.get("status") != "sealed_within_budget" \
            or seal.get("terminal") is not True:
        raise FitDiagnosticError("admission_seal_not_consumable")
    if doc.get("admission_passed") is not True:
        raise FitDiagnosticError(
            f"admission_not_passed:{(doc.get('problems') or [])[:4]}")
    package = doc.get("package")
    if not isinstance(package, dict) or \
            package.get("schema") != ADMISSION_PACKAGE_SCHEMA:
        raise FitDiagnosticError("admission_package_invalid")
    return doc


def admission_package_sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _registered_dirs(root: Path, source_root: Path | None,
                     teacher_npz_dir: Path | None, omomo_pt_dir: Path | None):
    source_root = Path(source_root or root / SOURCE_ROOT_REL)
    npz_dir = Path(teacher_npz_dir
                   or root / "resources/data/omniretarget/robot-object")
    pt_dir = Path(omomo_pt_dir or OMOMO_NEW_DIR)
    return source_root, npz_dir, pt_dir


def admission_recheck_problems(admission: dict, *, project_root: Path | None = None,
                               source_root: Path | None = None,
                               teacher_npz_dir: Path | None = None,
                               omomo_pt_dir: Path | None = None,
                               runtime_probe: Callable[[], dict] | None = None,
                               disk_probe: Callable[[Path], int] | None = None,
                               impl_sha: str | None = None,
                               ) -> list[str]:
    """Re-verify EVERY frozen binding against current disk (D1/D6/F entry).

    Source drift, a drifted implementation identity, a runtime identity
    drift or a lost disk margin each refuse the stage before any
    materialization (plan §3.2/§6.3/§8.2)."""
    root = Path(project_root or PROJECT_ROOT)
    package = admission.get("package") or {}
    source_root, npz_dir, pt_dir = _registered_dirs(
        root, source_root or package.get("source_root"), teacher_npz_dir,
        omomo_pt_dir)
    problems: list[str] = []
    binding = package.get("source_binding") or {}
    for key in ("registered_cache_identity", "preregistration",
                "training_manifest", "a1_frozen_lists", "a1_assets"):
        rec = binding.get(key)
        if not isinstance(rec, dict) or not rec.get("path"):
            continue
        p = root / str(rec["path"])
        if not p.is_file() or sha256_file(p) != rec.get("sha256"):
            problems.append(f"admission_source_drifted:{key}")
    # the prereg/manifest chain starts at the SOURCE's registered cache
    # identity; the BOUND cache (package.cache_binding) is what is
    # re-verified and consumed for schedules/root_stats (plan §41)
    source_cache = source_root / "cache/attempt_001"
    source_identity_path = source_cache / "cache_identity.json"
    source_chain_identity = json.loads(source_identity_path.read_text()) \
        if source_identity_path.is_file() else {}
    cache_rec = package.get("cache_binding") or {}
    cache_root = Path(cache_rec.get("path")
                      or source_root / "cache/attempt_001")
    if not (root / str(cache_rec.get("path") or "")).is_dir() \
            and cache_rec.get("path"):
        cache_root = root / str(cache_rec["path"])
    cache_identity_path = cache_root / "cache_identity.json"
    cache_identity = json.loads(cache_identity_path.read_text()) \
        if cache_identity_path.is_file() else {}
    cache_id_rec = cache_rec.get("cache_identity") or {}
    if cache_id_rec.get("sha256"):
        if not cache_identity_path.is_file() or \
                sha256_file(cache_identity_path) != cache_id_rec["sha256"]:
            problems.append("admission_cache_identity_drifted")
    prereg_path = root / str((source_chain_identity.get("preregistration") or {})
                             .get("path", ""))
    manifest_path = root / str((source_chain_identity.get("training_manifest")
                                or {}).get("path", ""))
    prereg = json.loads(prereg_path.read_text()) if prereg_path.is_file() else {}
    manifest = json.loads(manifest_path.read_text()) \
        if manifest_path.is_file() else {}
    prereg_check = w1.verify_preregistration(prereg, root, npz_dir, pt_dir,
                                             verify_assets=True)
    if not prereg_check["passed"]:
        problems.append(f"admission_preregistration_drifted:"
                        f"{prereg_check['problems'][:4]}")
    cache_check = w1.verify_cache_contents(cache_root, manifest, prereg, root)
    if not cache_check["passed"]:
        problems.append(
            f"admission_cache_drifted:{cache_check['problems'][:4]}")
    if cache_rec.get("path") and cache_root.resolve() != source_cache.resolve():
        problems += [f"admission_cache_migration:{p}" for p in
                     _migration_record_problems(root, source_cache,
                                                cache_root, cache_identity)]
    rows = {r.get("demo"): r for r in manifest.get("rows", [])
            if isinstance(r, dict)}
    schedule_rows = {rec.get("demo"): rec
                     for rec in cache_identity.get("schedule_rows", [])}
    for rec in package.get("windows") or []:
        demo = rec.get("demo")
        row = rows.get(demo) or {}
        inputs = row.get("inputs") or {}
        derived = row.get("derived") or {}
        sched_path = cache_root / "schedules" / w1.schedule_cache_filename(
            demo, (schedule_rows.get(demo) or {}).get("key") or "")
        for label, p, want in (
                ("teacher_npz", Path(npz_dir) / f"{demo}_original.npz",
                 inputs.get("npz_sha256")),
                ("teacher_pt", Path(pt_dir) / f"{demo}.pt",
                 inputs.get("pt_sha256")),
                ("derived", root / str(derived.get("path", "")),
                 derived.get("sha256")),
                ("schedule", sched_path, rec.get("schedule_sha256"))):
            if not p.is_file() or (want and sha256_file(p) != want):
                problems.append(f"admission_window_file_drifted:{demo}:{label}")
    rb = package.get("runtime_binding") or {}
    current = (runtime_probe or runtime_identity_facts)()
    problems += [f"admission_runtime_identity:{p}" for p in
                 runtime_identity_problems(rb.get("reference_identity") or {},
                                           current)]
    import ireset.eval.w1_real_fk_audit as fk_audit
    mod_rec = rb.get("requirements_module") or {}
    if mod_rec.get("source_sha256") and Path(fk_audit.__file__).is_file() \
            and sha256_file(Path(fk_audit.__file__)) != mod_rec["source_sha256"]:
        problems.append("admission_runtime_requirements_module_drifted")
    code = package.get("code_identity") or {}
    if code.get("implementation_files_sha256"):
        if impl_sha is None:
            impl_sha = implementation_identity(root)["implementation_files_sha256"]
        if impl_sha != code["implementation_files_sha256"]:
            problems.append("admission_implementation_drifted")
    if code.get("condition_code_sha256") and \
            code["condition_code_sha256"] != w1.condition_code_sha256(root):
        problems.append("admission_condition_code_drifted")
    yaml_rec = code.get("config_yaml") or {}
    if yaml_rec.get("path"):
        p = root / str(yaml_rec["path"])
        if not p.is_file() or sha256_file(p) != yaml_rec.get("sha256"):
            problems.append("admission_config_yaml_drifted")
    budget_doc = package.get("budget") or {}
    if {k: (v["wall_cap_s"], v["reserve_s"]) for k, v in
            (budget_doc.get("caps") or {}).items()} != \
            {k: v for k, v in BUDGET_CAPS.items()}:
        problems.append("admission_budget_caps_drifted")
    matrix = package.get("call_matrix") or {}
    if matrix.get("sha256") and matrix["sha256"] != frozen_call_matrix()["sha256"]:
        problems.append("admission_call_matrix_drifted")
    free = (disk_probe or (lambda p: shutil.disk_usage(p).free))(root)
    if free < R_REQUIRED_DISK_FREE_BYTES:
        problems.append(f"admission_disk_free_insufficient:{free}")
    return problems


def authorization_problems(doc, admission: dict, admission_path: Path,
                           *, current_impl_sha: str | None = None,
                           project_root: Path | None = None) -> list[str]:
    """§8.3: the explicit authorization must be entirely true and bound to
    the sealed admission package and the CURRENT implementation bytes."""
    root = Path(project_root or PROJECT_ROOT)
    if not isinstance(doc, dict) or doc.get("schema") != AUTHORIZATION_SCHEMA:
        return ["authorization_schema_invalid"]
    problems: list[str] = []
    if doc.get("authorized") is not True:
        problems.append("authorization_not_authorized")
    checklist = doc.get("checklist")
    if not isinstance(checklist, dict) or \
            set(checklist) != set(AUTHORIZATION_CHECKLIST):
        problems.append("authorization_checklist_invalid")
    else:
        for key in AUTHORIZATION_CHECKLIST:
            if checklist[key] is not True:
                problems.append(f"authorization_checklist_not_true:{key}")
    package = admission.get("package") or {}
    if doc.get("admission_package_sha256") != admission_package_sha256(admission_path):
        problems.append("authorization_admission_sha_mismatch")
    if doc.get("cohort_id") != package.get("cohort_id"):
        problems.append("authorization_cohort_mismatch")
    includes = doc.get("includes")
    if not isinstance(includes, list) or not includes or \
            not set(includes) <= {"D1", "D6", "F"} or "D1" not in includes:
        problems.append("authorization_stage_includes_invalid")
    caps = {k: [v["wall_cap_s"], v["reserve_s"]] for k, v in
            ((package.get("budget") or {}).get("caps") or {}).items()}
    if doc.get("budget_caps") != caps:
        problems.append("authorization_budget_caps_mismatch")
    matrix_sha = (package.get("call_matrix") or {}).get("sha256")
    if doc.get("call_matrix_sha256") != matrix_sha:
        problems.append("authorization_call_matrix_sha_mismatch")
    impl_sha = (package.get("code_identity") or {}).get(
        "implementation_files_sha256")
    if current_impl_sha is None:
        current_impl_sha = implementation_identity(root)[
            "implementation_files_sha256"]
    if doc.get("implementation_files_sha256") != impl_sha or \
            current_impl_sha != impl_sha:
        problems.append("authorization_implementation_sha_mismatch")
    return problems


# --------------------------------------------------------------------------
# billed-window materialization (§3.2): apparatus + registered windows
# --------------------------------------------------------------------------

def _guard_apparatus_window(billing: StageBudget | None) -> None:
    """Root-statistics/tokenizer/decoder/FK construction is legal only
    inside a started, not-exceeded D1/D6/F billing interval (§3.2/§7.5)."""
    if billing is None:
        raise FitDiagnosticError("materialization_without_billed_interval")
    if billing.stage not in ("D1", "D6", "F"):
        raise FitDiagnosticError(
            f"materialization_stage_not_fit:{billing.stage}")
    if billing.exceeded:
        raise FitDiagnosticError("materialization_after_budget_exceeded")
    billing.check("materialize_apparatus")


@dataclass
class FitApparatus:
    """The shared verification/training apparatus of one billed window."""

    device: str
    tokenizer: StateTokenizer
    decoder: MetricDecoder
    loss_fn: LossAssembler
    counters: dict
    fixture: bool

    def facts(self) -> dict:
        return {"device": self.device, "fixture": self.fixture,
                "counters": dict(self.counters)}


def _bound_cache_root(root: Path, package: dict,
                      source_root: Path | None) -> Path:
    """The cache attempt dir the admission package froze (plan §41)."""
    rec = package.get("cache_binding") or {}
    p = rec.get("path")
    if not p:
        src = Path(source_root or root / SOURCE_ROOT_REL)
        return src / "cache/attempt_001"
    c = Path(p)
    return c if c.is_absolute() else root / c


def materialize_fit_apparatus(*, admission: dict, budget: StageBudget,
                              device: str = "cuda", fixture: bool = False,
                              fk=None, static_frame=None,
                              project_root: Path | None = None,
                              source_root: Path | None = None,
                              teacher_npz_dir: Path | None = None,
                              omomo_pt_dir: Path | None = None,
                              runtime_probe: Callable[[], dict] | None = None,
                              ) -> FitApparatus:
    """Root statistics, joint range, tokenizer, metric decoder and the
    production loss/FK -- each a guarded unit of the CURRENT billed window."""
    _guard_apparatus_window(budget)
    root = Path(project_root or PROJECT_ROOT)
    counters: dict = {}
    budget.check("materialize_apparatus_entry")
    problems = admission_recheck_problems(
        admission, project_root=root, source_root=source_root,
        teacher_npz_dir=teacher_npz_dir, omomo_pt_dir=omomo_pt_dir,
        runtime_probe=runtime_probe)
    if problems:
        raise FitDiagnosticError(f"materialize_admission_drifted:{problems[:6]}")
    cache_root = _bound_cache_root(root, admission.get("package") or {},
                                   source_root)
    stats = _read_root_stats(cache_root / "root_stats.npz", budget, counters)
    budget.check("materialize_joint_range")
    if fixture:
        joint_range = np.stack([-2.0 + 0.02 * np.arange(J_JOINTS),
                                2.0 - 0.02 * np.arange(J_JOINTS)], axis=1)
    else:
        counters["g1_spec_constructions"] = \
            counters.get("g1_spec_constructions", 0) + 1
        joint_range = G1Spec().joint_range.astype(np.float64)
    tokenizer = StateTokenizer(joint_range, stats)
    counters["tokenizer_builds"] = counters.get("tokenizer_builds", 0) + 1
    decoder = MetricDecoder(stats.mean, stats.std, tokenizer.lo, tokenizer.hi)
    loss_fn = _build_loss(budget, counters, device, tokenizer,
                          fixture=fixture, fk=fk, static_frame=static_frame)
    return FitApparatus(device=device, tokenizer=tokenizer, decoder=decoder,
                        loss_fn=loss_fn, counters=counters, fixture=fixture)


def _read_root_stats(path: Path, budget: StageBudget,
                     counters: dict) -> RootStats:
    budget.check("materialize_root_stats_read")
    counters["root_stats_loads"] = counters.get("root_stats_loads", 0) + 1
    with np.load(path, allow_pickle=False) as z:
        mean = np.asarray(z["mean"], dtype=np.float64)
        std = np.asarray(z["std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,):
        raise FitDiagnosticError("root_stats_shape_invalid")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()
            and (std > 0).all()):
        raise FitDiagnosticError("root_stats_values_invalid")
    return RootStats(mean=mean, std=std)


def _build_loss(budget: StageBudget, counters: dict, device: str,
                tokenizer: StateTokenizer, *, fixture: bool,
                fk=None, static_frame=None) -> LossAssembler:
    """The production loss/FK as guarded construction units.

    Real path: the audited ``G1TorchFK`` and the locked-scene static frame
    are constructed explicitly (compile-or-cache-hit recorded honestly);
    injection is forbidden.  Fixture path (tests only): labelled stubs,
    never a production equivalence claim."""
    from ireset.geometry import distance_adapter as _da
    from ireset.geometry.distance_adapter import largebox_static_frame

    budget.check("materialize_loss_construct")
    counters["loss_assembler_constructions"] = \
        counters.get("loss_assembler_constructions", 0) + 1
    if fixture:
        from ireset.eval.w1_smallset_fit_diagnostic import SyntheticFK
        fk = fk if fk is not None else SyntheticFK(seed=0)
        static_frame = static_frame if static_frame is not None \
            else (np.eye(3), np.zeros(3))
    else:
        if fk is not None or static_frame is not None:
            raise FitDiagnosticError(
                "real_world_production_fk_injection_forbidden")
        budget.check("materialize_g1_fk_construct")
        counters["g1_fk_constructions"] = \
            counters.get("g1_fk_constructions", 0) + 1
        fk = G1TorchFK(device=device, dtype=torch.float32)
        budget.check("materialize_scene_access")

        def scene_check(point: str) -> None:
            budget.check(f"materialize_scene_{point}")

        counters["largebox_scene_access"] = \
            counters.get("largebox_scene_access", 0) + 1
        counters["largebox_scene_compiled"] = _da._STATIC_FRAME is None
        static_frame = largebox_static_frame(check=scene_check)
    return LossAssembler(tokenizer, LossWeights(w_contact=0.0), device,
                         use_fk=True, fk=fk, static_frame=static_frame)


class RegisteredWindowDataset(torch.utils.data.Dataset):
    """The registered windows only, in canonical order (§3.2: exactly the
    six admitted train demos' first windows; D1 takes the canonical first)."""

    def __init__(self, items: list[dict],
                 window_keys: list[WindowKey] | None = None) -> None:
        self.items = list(items)
        self.window_keys = list(window_keys) if window_keys is not None else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


@dataclass
class RealFitWorld:
    """The real SOURCE world for one fit stage: registered windows on top
    of the billed apparatus.  Every array read, clip build and compile
    happened inside the stage's billed window; ``facts`` is what the stage
    report seals."""

    stage: str
    device: str
    tokenizer: StateTokenizer
    registration_d1: FitRegistration
    registration_d6: FitRegistration
    items: list[dict]
    decoder: MetricDecoder
    loss_fn: LossAssembler
    counters: dict
    facts: dict
    fixture: bool = False

    def registration_for(self, stage: str) -> FitRegistration:
        if stage == "D1":
            return self.registration_d1
        if stage == "D6":
            return self.registration_d6
        raise FitDiagnosticError(f"unknown stage {stage!r}")

    def view_for(self, stage: str) -> dict:
        reg = self.registration_for(stage)
        items = self.items[:len(reg.windows)]
        ds = RegisteredWindowDataset(items, window_keys=list(reg.windows))
        probe_batch = {k: torch.stack([it[k] for it in items])
                       for k in ("x0", *COND_KEYS)}
        return {"ds": ds, "probe_batch": probe_batch, "registration": reg}

    def build_trainer(self, plan: FitPlan, out_dir: Path,
                      budget_deadline: float | None,
                      identity: dict | None = None,
                      train_ds=None,
                      construction_observer=None) -> Trainer:
        cfg = TrainConfig(
            model_type=plan.model_type, steps=plan.steps,
            micro_batch=plan.micro_batch, grad_accum=plan.grad_accum,
            lr=plan.lr, weight_decay=plan.weight_decay,
            grad_clip=plan.grad_clip, seed=plan.seed, num_workers=0,
            val_seed=plan.val_seed, d_model=plan.d_model,
            n_layers=plan.n_layers, n_heads=plan.n_heads,
            ffn_mult=plan.ffn_mult, time_window=plan.time_window,
            device=self.device, loss=LossWeights(w_contact=0.0))
        ds = train_ds if train_ds is not None else self.view_for(plan.stage)["ds"]
        return Trainer(cfg, self.tokenizer, ds, ds, Path(out_dir),
                       identity=identity, budget_deadline=budget_deadline,
                       loss_fn=self.loss_fn,
                       construction_observer=construction_observer)


def _window_item(clip, tokenizer: StateTokenizer, start: int, fi: int) -> dict:
    """One registered window item with EXACTLY the production dataset
    semantics (``TeacherWindowDataset.__getitem__`` at start=0)."""
    cond = clip.condition
    root, q = clip.root[start:start + T_WINDOW], clip.q[start:start + T_WINDOW]
    x0 = tokenizer.encode(root, q, cond.world_from_clip)
    box_clip = world_to_clip(cond.target_obj_pose[start:start + T_WINDOW, 0:3],
                             cond.world_from_clip)
    kps_obj = keypoints_in_source_object(
        cond.human_kps_world, cond.source_obj_pose)[start:start + T_WINDOW]
    kps_clip = world_to_clip(cond.human_kps_world,
                             cond.world_from_clip)[start:start + T_WINDOW]
    return {
        "x0": torch.as_tensor(x0, dtype=torch.float32),
        "kps_obj": torch.as_tensor(kps_obj, dtype=torch.float32),
        "kps_clip": torch.as_tensor(kps_clip, dtype=torch.float32),
        "box_t_clip": torch.as_tensor(box_clip, dtype=torch.float32),
        "box_quat": torch.as_tensor(
            cond.target_obj_pose[start:start + T_WINDOW, 3:7],
            dtype=torch.float32),
        "contact": torch.as_tensor(cond.contact_flags()[start:start + T_WINDOW],
                                   dtype=torch.float32),
        "wfc": torch.as_tensor(cond.world_from_clip, dtype=torch.float32),
        "valid": torch.ones(T_WINDOW, dtype=torch.float32),
        "clip_idx": fi, "start": start,
    }


def registrations_from_package(package: dict) -> tuple[FitRegistration,
                                                        FitRegistration]:
    """The frozen D1/D6 registrations derived from the admission package
    (available before any world materialization, §7.4 item 3)."""
    windows = [WindowKey(rec["demo"], rec["start"], rec["end"])
               for rec in package["windows"]]
    d6_registration = FitRegistration(
        stage="D6", windows=canonical_window_order(windows),
        splits=tuple(["train"] * len(windows)))
    d1_registration = FitRegistration(
        stage="D1", windows=(d6_registration.windows[0],), splits=("train",))
    if len(d6_registration.windows) != 6:
        raise FitDiagnosticError("materialize_window_count_invalid")
    return d1_registration, d6_registration


def materialize_real_world(*, admission: dict, stage: str,
                           budget: StageBudget, device: str = "cuda",
                           fixture: bool = False, fk=None,
                           static_frame=None,
                           project_root: Path | None = None,
                           source_root: Path | None = None,
                           teacher_npz_dir: Path | None = None,
                           omomo_pt_dir: Path | None = None,
                           d1_probe_inputs: Path | None = None,
                           apparatus: FitApparatus | None = None,
                           ) -> RealFitWorld:
    """Materialize the registered windows inside the CURRENT billed window.

    §3.2: the admission package is re-verified against disk first; every
    registered-window clip is a guarded unit of THIS stage.  ``apparatus``
    may be supplied when the caller already materialized it in the SAME
    billed window (e.g. the D6 handoff check, §7.4 item 3) -- it is then
    reused instead of rebuilt.  A failure raises; the stage records it as
    incomplete."""
    materialize_for_fit({"stage": stage}, budget)
    if not fixture and (fk is not None or static_frame is not None):
        raise FitDiagnosticError("real_world_production_fk_injection_forbidden")
    root = Path(project_root or PROJECT_ROOT)
    budget.check("materialize_entry")
    if apparatus is None:
        apparatus = materialize_fit_apparatus(
            admission=admission, budget=budget, device=device, fixture=fixture,
            fk=fk, static_frame=static_frame, project_root=root,
            source_root=source_root, teacher_npz_dir=teacher_npz_dir,
            omomo_pt_dir=omomo_pt_dir)
    counters = apparatus.counters
    package = admission["package"]
    d1_registration, d6_registration = registrations_from_package(package)
    want = 1 if stage == "D1" else 6
    stage_windows = d6_registration.windows[:want]

    cache_root = _bound_cache_root(root, package, source_root)
    cache_identity = json.loads(
        (cache_root / "cache_identity.json").read_text())
    manifest_path = root / str((cache_identity.get("training_manifest") or {})
                               .get("path", ""))
    manifest = json.loads(manifest_path.read_text())
    rows = {r.get("demo"): r for r in manifest.get("rows", [])}
    schedule_rows = {rec.get("demo"): rec
                     for rec in cache_identity.get("schedule_rows", [])}
    prereg_path = root / str((cache_identity.get("preregistration") or {})
                             .get("path", ""))
    assets_sha = (json.loads(prereg_path.read_text()).get("assets") or {}
                  ).get("a1_assets_sha256")
    code_sha = w1.condition_code_sha256(root)
    items: list[dict] = []
    window_facts: list[dict] = []
    for fi, w in enumerate(stage_windows):
        budget.check(f"materialize_window_{fi}_{w.demo}")
        counters["registered_window_reads"] = \
            counters.get("registered_window_reads", 0) + 1
        row = rows.get(w.demo) or {}
        inputs = row.get("inputs") or {}
        cache_file = cache_root / "schedules" / w1.schedule_cache_filename(
            w.demo, w1.schedule_cache_key(inputs.get("npz_sha256", ""),
                                          inputs.get("pt_sha256", ""),
                                          assets_sha or "", code_sha))
        if not cache_file.is_file():
            raise FitDiagnosticError(
                f"materialize_schedule_cache_missing:{w.demo}:{cache_file.name}")
        # load_w1_derived_clip takes the SCHEDULES dir (the keyed-cache
        # layout used by every W1 consumer)
        clip = w1.load_w1_derived_clip(row, cache_root / "schedules", root,
                                       assets_sha or "", code_sha)
        if clip.validation.get("drop") or clip.validation.get("clamped"):
            raise FitDiagnosticError(
                f"materialize_window_invalid:{w.demo}:{clip.validation}")
        if clip.T != row.get("T"):
            raise FitDiagnosticError(
                f"materialize_window_T_mismatch:{w.demo}:{clip.T}")
        item = _window_item(clip, apparatus.tokenizer, w.start, fi)
        x0_np = item["x0"].numpy()
        if x0_np.shape != (T_WINDOW, 38) or not np.isfinite(x0_np).all():
            raise FitDiagnosticError(f"materialize_window_x0_invalid:{w.demo}")
        counters["window_x0_encodes"] = counters.get("window_x0_encodes", 0) + 1
        window_facts.append({
            "window_id": w.window_id, "demo": w.demo,
            "x0_sha256": hashlib.sha256(
                np.ascontiguousarray(x0_np).tobytes()).hexdigest(),
            "derived_sha256": (row.get("derived") or {}).get("sha256"),
            "schedule_key": (schedule_rows.get(w.demo) or {}).get("key"),
            "schedule_cache_file": cache_file.name})
        items.append(item)
    probe_batch = {k: torch.stack([it[k] for it in items])
                   for k in ("x0", *COND_KEYS)}

    # §3.2 binding: the shared D1 window must materialize bitwise-identically
    d1_shared = None
    if stage == "D6" and d1_probe_inputs is not None \
            and Path(d1_probe_inputs).is_file():
        budget.check("materialize_d1_shared_window_binding")
        with np.load(d1_probe_inputs, allow_pickle=False) as z:
            if "w0__x0" in z:
                d1_shared = hashlib.sha256(
                    np.ascontiguousarray(z["w0__x0"]).tobytes()).hexdigest()
        if d1_shared is not None and d1_shared != window_facts[0]["x0_sha256"]:
            raise FitDiagnosticError("d1_d6_shared_window_x0_mismatch")

    facts = {"stage": stage, "device": device, "fixture": fixture,
             "windows": window_facts, "counters": dict(counters),
             "d1_shared_window_x0_sha256": d1_shared,
             "scene_compiled": counters.get("largebox_scene_compiled"),
             "probe_batch_shapes": {k: list(v.shape)
                                    for k, v in probe_batch.items()}}
    return RealFitWorld(stage=stage, device=device,
                        tokenizer=apparatus.tokenizer,
                        registration_d1=d1_registration,
                        registration_d6=d6_registration, items=items,
                        decoder=apparatus.decoder, loss_fn=apparatus.loss_fn,
                        counters=counters, facts=facts, fixture=fixture)


# --------------------------------------------------------------------------
# real chain orchestration (§7 sequencing, §8.1 per-stage budgets)
# --------------------------------------------------------------------------

def real_build_modules_factory(device: str, plan: FitPlan):
    def factory():
        if plan.model_type == "diffusion":
            from ireset.diffusion.generator import DiffusionG1
            model = DiffusionG1(
                num_train_timesteps=1000, d_model=plan.d_model,
                n_layers=plan.n_layers, n_heads=plan.n_heads,
                ffn_mult=plan.ffn_mult, time_window=plan.time_window).to(device)
        else:
            from ireset.model.deterministic import DeterministicG1
            model = DeterministicG1(
                d_model=plan.d_model, n_layers=plan.n_layers,
                n_heads=plan.n_heads, ffn_mult=plan.ffn_mult,
                time_window=plan.time_window).to(device)
        from ireset.model.conditioner import ConditionEncoder
        conditioner = ConditionEncoder(plan.d_model).to(device)
        return model, conditioner
    return factory


class ProductionRuntimeGuardShim:
    """Dispatch to the audited production guard, or to the labelled
    synthetic guard on the fixture path (no production equivalence claim)."""

    def __init__(self, fixture: bool) -> None:
        from ireset.eval.w1_smallset_fit_diagnostic import (
            ProductionRuntimeGuard, SyntheticRuntimeGuard)

        self._inner = (SyntheticRuntimeGuard() if fixture
                       else ProductionRuntimeGuard())
        self.expected_device = getattr(self._inner, "expected_device",
                                       "cpu" if fixture else "cuda")
        self.expected_dtype = getattr(self._inner, "expected_dtype",
                                      torch.float32)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _stage_identity(admission: dict, plan: FitPlan,
                    registration: FitRegistration, stage_dir: Path) -> dict:
    from ireset.eval.w1_smallset_fit_diagnostic import _stage_identity_binding

    binding = _stage_identity_binding(plan, registration)
    return {"module": "ireset.eval.w1_smallset_fit_r_admission",
            "kind": "real_fit", "stage": plan.stage,
            "cohort_id": admission.get("cohort_id") or COHORT_ID,
            "admission_package_sha256":
                (admission.get("package") or {}).get("_sha256"),
            "stage_dir": path_for_record(Path(stage_dir)),
            **binding}


def run_real_chain(*, admission_path: Path, authorization_path: Path,
                   run_root: Path, chain_t0: float | None = None,
                   clock: Callable[[], float] = time.perf_counter,
                   device: str = "cuda", fixture: bool = False,
                   fk=None, static_frame=None,
                   project_root: Path | None = None,
                   source_root: Path | None = None,
                   teacher_npz_dir: Path | None = None,
                   omomo_pt_dir: Path | None = None,
                   impl_sha: str | None = None,
                   ) -> dict:
    """Authorized real chain: D1 -> conditional D6 -> F under §8.1 budgets.

    Each stage gets its OWN budget whose t0 starts when control reaches the
    stage (no inter-window gap); the acceptance summary is sealed against a
    chain budget whose cap equals the §8.1 fit total.  The authorization
    gate, the admission re-check and the runtime application run inside
    D1's already-started window."""
    root = Path(project_root or PROJECT_ROOT)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    chain_t0 = chain_t0 if chain_t0 is not None else clock()
    chain_budget = StageBudget(stage="chain",
                               wall_cap_s=R_TOTALS["fit_stages_total_s"],
                               reserve_s=60.0, t0=chain_t0, clock=clock)
    facts: dict = {"schema": "w1_smallset_fit_real_run_v1",
                   "kind": "real_fit_chain", "cohort_id": COHORT_ID,
                   "device": device, "fixture": fixture,
                   "admission": {"path": str(admission_path),
                                 "sha256": admission_package_sha256(admission_path)},
                   "authorization": {"path": str(authorization_path),
                                     "sha256": sha256_file(Path(authorization_path))}}
    problems: list[str] = []
    try:
        admission = load_admission(admission_path)
        admission["package"]["_sha256"] = admission_package_sha256(admission_path)
        authorization = json.loads(Path(authorization_path).read_text())
        budget_d1 = budget_for_stage("D1", t0=chain_t0, clock=clock)
        runtime_guard = ProductionRuntimeGuardShim(fixture)
        auth = {"includes": list(authorization.get("includes") or []),
                "cohort_id": authorization.get("cohort_id")}
        d1_plan = FitPlan("D1")
        d1_dir = run_root / "d1"
        budget_d1.check("chain_entry")
        problems += authorization_problems(
            authorization, admission, admission_path,
            current_impl_sha=impl_sha, project_root=root)
        if problems:
            raise FitDiagnosticError(f"authorization_refused:{problems}")
        budget_d1.check("chain_admission_recheck")
        drift = admission_recheck_problems(
            admission, project_root=root, source_root=source_root,
            teacher_npz_dir=teacher_npz_dir, omomo_pt_dir=omomo_pt_dir,
            impl_sha=impl_sha)
        if drift:
            raise FitDiagnosticError(f"admission_drifted:{drift[:6]}")
        d1_world = materialize_real_world(
            admission=admission, stage="D1", budget=budget_d1, device=device,
            fixture=fixture, fk=fk, static_frame=static_frame,
            project_root=root, source_root=source_root,
            teacher_npz_dir=teacher_npz_dir, omomo_pt_dir=omomo_pt_dir)
        budget_d1.check("d1_world_ready")
        d1_identity = _stage_identity(admission, d1_plan,
                                      d1_world.registration_for("D1"), d1_dir)
        d1_view = d1_world.view_for("D1")
        trainer = d1_world.build_trainer(
            d1_plan, d1_dir, budget_deadline=budget_d1.work_deadline,
            identity=d1_identity)
        driver = FitDriver(
            trainer=trainer, plan=d1_plan,
            registration=d1_view["registration"],
            loss_fn=d1_world.loss_fn, decoder=d1_world.decoder,
            probe_batch=d1_view["probe_batch"], budget=budget_d1,
            runtime_guard=runtime_guard, identity=d1_identity,
            stage_dir=d1_dir,
            training_gen=torch.Generator().manual_seed(d1_plan.seed),
            dataset_window_keys=d1_view["ds"].window_keys,
            authorization_includes_d6="D6" in auth["includes"])
        summary = driver.run()
        summary["materialization"] = d1_world.facts
        finished = finish_stage(d1_dir, summary, budget_d1)
        d1_returned = clock() - budget_d1.t0
        d1_cov = _observe_returned_report(d1_dir / "stage_report.json",
                                          budget_d1, I_WITHDRAW_ON_EXCEED)
        finished["return_coverage"] = d1_cov
        doc = finished["summary_doc"]
        facts["d1"] = {
            "local_complete_evidence": doc.get("local_complete_evidence"),
            "local_mean_fit_passed": doc.get("local_mean_fit_passed"),
            "local_conclusion": doc.get("local_conclusion"),
            "seal": (doc.get("seal") or {}).get("status"),
            "return_coverage": d1_cov.get("covered"),
            "stage_returned_elapsed_s": d1_returned,
            "call_matrix_problems": summary["call_matrix"]["problems"],
            "materialization_counters": d1_world.facts["counters"]}
        d6_plan = d6_world = d6_identity = None
        d6_dir = run_root / "d6"
        d1_registration = d1_world.registration_for("D1")
        d6_ran = False
        facts["authorized_includes"] = list(auth["includes"])
        if facts["d1"]["local_complete_evidence"] is not True \
                or facts["d1"]["return_coverage"] is not True:
            facts["d1_handoff"] = {"handoff_ready": None,
                                   "skipped_reason": "d1_incomplete"}
            facts["d6"] = {"skipped_reason": "d1_incomplete"}
        elif facts["d1"]["local_mean_fit_passed"] is not True:
            # §7.2: a complete negative local result is never retried with
            # D6; the local fact goes to F for final verification.
            facts["d1_handoff"] = {"handoff_ready": False,
                                   "skipped_reason":
                                       "d1_mean_fit_gate_not_passed"}
            facts["d6"] = {"skipped_reason": "d1_mean_fit_gate_not_passed"}
        elif "D6" not in auth["includes"]:
            facts["d1_handoff"] = {"handoff_ready": False,
                                   "skipped_reason": "d6_not_authorized"}
            facts["d6"] = {"skipped_reason": "d6_not_authorized"}
        else:
            # D6's OWN window starts here.  §7.4 item 3: the handoff
            # re-verification (and the verification apparatus it requires)
            # is D6-billed and precedes ANY training materialization.
            budget_d6 = budget_for_stage("D6", t0=clock(), clock=clock)
            d6_plan = FitPlan("D6")
            d6_apparatus = materialize_fit_apparatus(
                admission=admission, budget=budget_d6, device=device,
                fixture=fixture, fk=fk, static_frame=static_frame,
                project_root=root, source_root=source_root,
                teacher_npz_dir=teacher_npz_dir, omomo_pt_dir=omomo_pt_dir)
            budget_d6.check("d6_apparatus_ready")
            handoff = resolve_handoff(
                d1_dir, d1_plan, d6_apparatus.decoder, d1_registration, auth,
                budget_d6, expected_identity=d1_identity,
                production_decoder=d6_apparatus.loss_fn)
            facts["d1_handoff"] = {"handoff_ready": handoff["handoff_ready"],
                                   "problems": handoff["problems"][:5]}
            if handoff["handoff_ready"]:
                budget_d6.check("d6_handoff_resolved")
                d6_world = materialize_real_world(
                    admission=admission, stage="D6", budget=budget_d6,
                    device=device, fixture=fixture, fk=fk,
                    static_frame=static_frame, project_root=root,
                    source_root=source_root, teacher_npz_dir=teacher_npz_dir,
                    omomo_pt_dir=omomo_pt_dir,
                    d1_probe_inputs=d1_dir / "probe_inputs.npz",
                    apparatus=d6_apparatus)
                budget_d6.check("d6_world_ready")
                d6_identity = _stage_identity(admission, d6_plan,
                                              d6_world.registration_for("D6"),
                                              d6_dir)
                d6_view = d6_world.view_for("D6")
                trainer6 = d6_world.build_trainer(
                    d6_plan, d6_dir, budget_deadline=budget_d6.work_deadline,
                    identity=d6_identity)
                driver6 = FitDriver(
                    trainer=trainer6, plan=d6_plan,
                    registration=d6_view["registration"],
                    loss_fn=d6_world.loss_fn, decoder=d6_world.decoder,
                    probe_batch=d6_view["probe_batch"], budget=budget_d6,
                    runtime_guard=runtime_guard, identity=d6_identity,
                    stage_dir=d6_dir,
                    training_gen=torch.Generator().manual_seed(d6_plan.seed),
                    dataset_window_keys=d6_view["ds"].window_keys,
                    authorization_includes_d6=True)
                d6_summary = driver6.run()
                d6_summary["materialization"] = d6_world.facts
                finished6 = finish_stage(d6_dir, d6_summary, budget_d6)
                d6_returned = clock() - budget_d6.t0
                d6_cov = _observe_returned_report(
                    d6_dir / "stage_report.json", budget_d6,
                    I_WITHDRAW_ON_EXCEED)
                finished6["return_coverage"] = d6_cov
                d6_doc = finished6["summary_doc"]
                facts["d6"] = {
                    "local_complete_evidence": d6_doc.get("local_complete_evidence"),
                    "local_mean_fit_passed": d6_doc.get("local_mean_fit_passed"),
                    "local_conclusion": d6_doc.get("local_conclusion"),
                    "seal": (d6_doc.get("seal") or {}).get("status"),
                    "return_coverage": d6_cov.get("covered"),
                    "stage_returned_elapsed_s": d6_returned,
                    "call_matrix_problems": d6_summary["call_matrix"]["problems"],
                    "materialization_counters": d6_world.facts["counters"]}
                d6_ran = True
            else:
                facts["d6"] = {"skipped_reason": "handoff_not_ready",
                               "handoff_problems": handoff["problems"][:6]}
        facts["d6_ran"] = d6_ran
        # F verifies whatever completed locally; a complete negative result
        # is verifiable too (plan §7.3).
        f_specs = []
        if facts["d1"].get("local_complete_evidence") is True:
            f_specs.append({"stage_dir": d1_dir, "plan": d1_plan,
                            "registration": d1_registration,
                            "decoder": d1_world.decoder,
                            "loss_fn": d1_world.loss_fn,
                            "identity": d1_identity})
        if d6_ran and (facts.get("d6") or {}).get("local_complete_evidence") \
                is True and facts["d6"].get("return_coverage") is True:
            f_specs.append({"stage_dir": d6_dir, "plan": d6_plan,
                            "registration": d6_world.registration_for("D6"),
                            "decoder": d6_world.decoder,
                            "loss_fn": d6_world.loss_fn,
                            "identity": d6_identity})
        if f_specs:
            budget_f = budget_for_stage("F", t0=clock(), clock=clock)
            f_apparatus = materialize_fit_apparatus(
                admission=admission, budget=budget_f, device=device,
                fixture=fixture, fk=fk, static_frame=static_frame,
                project_root=root, source_root=source_root,
                teacher_npz_dir=teacher_npz_dir, omomo_pt_dir=omomo_pt_dir)
            budget_f.check("f_apparatus_ready")
            f_out = run_f_verification(
                stage_specs=f_specs, out_dir=run_root / "f",
                budget=budget_f, runtime_guard=runtime_guard,
                build_modules=real_build_modules_factory(device, d1_plan),
                authorization=auth)
            f_returned = budget_f.clock() - budget_f.t0
            f_cov = _observe_returned_report(run_root / "f" / "f_report.json",
                                             budget_f, F_WITHDRAW_ON_EXCEED)
            f_out["return_coverage"] = f_cov
            f_out["summary_doc"] = f_cov["final_doc"]
            f_doc = f_out["summary_doc"]
            facts["f"] = {k: f_doc.get(k)
                          for k in ("complete_evidence", "replay_passed",
                                    "mean_fit_passed", "verdict")}
            facts["f"]["return_coverage"] = f_cov.get("covered")
            facts["f"]["stage_returned_elapsed_s"] = f_returned
            facts["f"]["stages"] = [
                {k: s.get(k) for k in ("stage", "replay_enabled",
                                       "replay_passed", "skip_reason",
                                       "replay_problems", "mean_fit_passed")}
                for s in (f_doc.get("stages") or [])]
            facts["f"]["materialization_counters"] = f_apparatus.facts()["counters"]
    except FitDiagnosticError as exc:
        problems.append(f"chain_refused:{exc}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"chain_error:{type(exc).__name__}:{str(exc)[:200]}")
    facts["problems"] = problems
    facts["chain_budget"] = chain_budget.facts()
    outputs = write_acceptance_outputs(run_root, facts, chain_budget,
                                       withdraw=dict(CHAIN_WITHDRAW_ON_EXCEED))
    outputs["facts"] = facts
    return outputs
