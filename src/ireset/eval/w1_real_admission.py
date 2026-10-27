"""Read-only admission package for the W1 real training-objective audit.

The package freezes what an authorized run would be asked to execute: fixed
asset identities, the four candidate window keys, timesteps/noise, array and
row schemas, cohort budgets, attempt and call limits, and the still-unchecked
authorization checklist.  Everything here is derived from metadata files.

No window arrays, reference schedules, cache contents or checkpoints are
loaded; no model/conditioner forward or VJP runs; no cohort or registered run
is created.  Drift is reported in ``problems`` and must stop authorization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from ireset.eval import w1_real_objective_audit as audit

SCHEMA = "w1_real_objective_admission_package_v1"
AUTHORIZATION_SCHEMA = audit.AUTHORIZATION_SCHEMA
SPLIT_TRAIN = Path("derived/data/split/train.json")

_KNOWN_LIMITATIONS = [
    "fk_m2 covers only aggregation from production FK points; real FK geometry, "
    "decode-to-FK correctness and derivatives remain not_audited",
    "network parameter gradients, accumulation, clipping, AdamW, optimizer "
    "updates, model selection and DDIM sampling are not_audited; R3=not_run",
    "the independent reference and the F verifier share one formula "
    "implementation; F recomputes raw arrays but is not a second formula",
    "near-pi float32 so3_log precision: values outside the independent rotation "
    "domain are gaps, never passes",
    "float32 term overrides for geodesic_rad2/total/L5 are precision allowances, "
    "not structural relaxations",
    "real CUDA determinism and RNG consumption are not_run; F requires bitwise "
    "production-gradient replay and records mismatch rather than widening",
    "the synthetic calibration proves the contract was not deleted, replaced or "
    "widened; it does not prove real float32/float64 differences are below the "
    "frozen thresholds",
]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _file_hash(path: Path) -> str | None:
    p = Path(path)
    return audit.sha256_file(p) if p.is_file() else None


def _safe_read(path: Path, problems: list[str], tag: str) -> Any:
    try:
        return _read_json(path)
    except FileNotFoundError:
        problems.append(f"{tag}_missing")
    except (OSError, ValueError) as exc:
        problems.append(f"{tag}_unreadable:{type(exc).__name__}")
    return None


def authorization_template() -> dict:
    """A reviewable schema whose ``authorized=false`` the real entry refuses."""
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "template_only": True,
        "authorized": False,
        "cohort_id": audit.COHORT_ID,
        "execution_id": audit.EXECUTION_ID,
        "granted_by": None,
        "granted_wall": None,
        "training_updates": 0,
        "parameter_fd": 0,
        "ddim_calls": 0,
        "checklist": {key: None for key in audit.AUTHORIZATION_REQUIRED},
        "budgets": {
            stage: {"budget_s": spec["budget_s"], "attempts": spec["attempts"]}
            for stage, spec in audit.STAGE_BUDGETS.items()
        },
        "notes": (
            "Template only. The user must create a separate authorization record "
            "with authorized=true and every checklist item true; schema must "
            "match; training_updates/parameter_fd/ddim_calls must be JSON "
            "integer 0; budget_s/attempts must exactly equal the frozen values; "
            "section 8 also requires the pending admission items to be closed "
            "first."
        ),
    }


def _anchor_facts(root: Path, problems: list[str]) -> dict:
    anchors: dict[str, dict] = {}
    for group_name, group in (("parent", audit.PARENT_ANCHORS),
                              ("source", audit.SOURCE_ANCHORS)):
        for name, (rel, want) in group.items():
            p = root / rel
            actual = _file_hash(p)
            anchors[name] = {
                "group": group_name,
                "path": audit.path_for_record(p),
                "expected_sha256": want,
                "actual_sha256": actual,
                "match": actual is not None and actual == want,
            }
            if actual is None:
                problems.append(f"anchor_missing:{name}")
            elif want is not None and actual != want:
                problems.append(f"anchor_drifted:{name}")
    return anchors


def _select_candidate_windows(manifest: Mapping[str, Any], *,
                              assets_sha: str, code_sha: str
                              ) -> tuple[list[audit.WindowKey], list[str], dict]:
    from ireset.data import w1_training as w1

    problems: list[str] = []
    rows = {
        str(r["demo"]): r for r in manifest.get("rows", [])
        if isinstance(r, Mapping) and "demo" in r
    }
    selection = manifest.get("selection")
    selection = selection if isinstance(selection, Mapping) else {}
    train24 = [d for d in selection.get("train24", []) if d in rows]
    eligible = {d: rows[d] for d in train24 if rows[d].get("split") == "train"}
    selected: list[tuple[str, list[int]]] = []
    for demo in sorted(eligible):
        row = eligible[demo]
        admission = row.get("admission")
        admission = admission if isinstance(admission, Mapping) else {}
        if not admission.get("motion"):
            continue
        total = int(row.get("T", 0))
        starts = list(range(0, total - audit.WINDOW + 1, audit.STRIDE))
        if len(starts) >= audit.WINDOWS_PER_DEMO:
            selected.append((demo, starts[:audit.WINDOWS_PER_DEMO]))
        if len(selected) == audit.DEMO_COUNT:
            break
    if len(selected) != audit.DEMO_COUNT:
        problems.append(
            f"fewer_than_{audit.DEMO_COUNT}_train_demos_with_two_full_windows:"
            f"{len(selected)}")
    keys: list[audit.WindowKey] = []
    facts: dict[str, dict] = {}
    for demo, starts in selected:
        row = eligible[demo]
        admission = row.get("admission")
        admission = admission if isinstance(admission, Mapping) else {}
        try:
            cache_key = w1.schedule_cache_key(
                row["inputs"]["npz_sha256"], row["inputs"]["pt_sha256"],
                assets_sha, code_sha)
            input_sha = audit.sha256_bytes(json.dumps(
                {"npz": row["inputs"]["npz_sha256"],
                 "pt": row["inputs"]["pt_sha256"],
                 "derived": row["derived"]["sha256"]},
                sort_keys=True, separators=(",", ":")).encode())
            derived_sha = str(row["derived"]["sha256"])
        except (KeyError, TypeError) as exc:
            problems.append(
                f"window_metadata_incomplete:{demo}:{type(exc).__name__}")
            continue
        facts[demo] = {
            "T": int(row.get("T", 0)), "starts": starts,
            "full_windows_available": len(
                range(0, int(row.get("T", 0)) - audit.WINDOW + 1, audit.STRIDE)),
            "motion_admitted": bool(admission.get("motion")),
            "cache_key": cache_key, "input_sha": input_sha,
            "derived_sha": derived_sha,
        }
        for start in starts:
            keys.append(audit.WindowKey(
                demo=demo, start=int(start), end=int(start + audit.WINDOW),
                T=int(row.get("T", 0)), cache_key=cache_key, split="train",
                input_sha=input_sha, derived_sha=derived_sha))
    if len(keys) != audit.WINDOW_COUNT:
        problems.append(f"window_candidate_count:{len(keys)}")
    return keys, problems, facts


def _stage_call_matrix() -> dict:
    zero = {"conditioner": 0, "model": 0, "production_vjp": 0,
            "independent_vjp": 0}
    one = {"conditioner": 1, "model": 1, "production_vjp": 1,
           "independent_vjp": 1}
    return {
        "R0": dict(zero),
        "P": {"deterministic": dict(one), "diffusion": dict(one),
              "total": {"conditioner": 2, "model": 2,
                        "production_vjp": 2, "independent_vjp": 2}},
        "F": {"deterministic": dict(one), "diffusion": dict(one),
              "total": {"conditioner": 2, "model": 2,
                        "production_vjp": 2, "independent_vjp": 2}},
        "cohort_caps": dict(audit.CALL_CAPS),
        "total_batched_vjp_cap": (audit.CALL_CAPS["production_vjp"]
                                  + audit.CALL_CAPS["independent_vjp"]),
    }


def build_admission_package(project_root: Path, *,
                            tolerance_contract: Mapping[str, Any],
                            tolerance_contract_path: Path | None = None,
                            tolerance_contract_sha256: str | None = None
                            ) -> dict:
    """Freeze the reviewable package; never executes real audit stages."""
    root = Path(project_root)
    contract_problems = audit.tolerance_contract_problems(tolerance_contract)
    if contract_problems:
        raise ValueError(f"tolerance contract refused: {contract_problems}")
    if (tolerance_contract.get("mode") != "synthetic_calibrated"
            or not tolerance_contract.get("evidence_sha256")):
        raise ValueError(
            "admission package requires a completed synthetic calibration contract")

    problems: list[str] = []
    anchors = _anchor_facts(root, problems)

    reg_path = root / audit.W1_RUN / "freeze/execution_registration.json"
    registration = _safe_read(reg_path, problems, "w1_registration")
    manifest = None
    prereg = None
    cache_facts: dict = {}
    config_facts: dict = {}
    checkpoint_facts: dict = {}
    manifest_facts: dict = {}
    train24_facts: dict = {}

    if isinstance(registration, Mapping):
        manifest_rec = registration.get("training_manifest")
        manifest_rec = manifest_rec if isinstance(manifest_rec, Mapping) else {}
        manifest_rel = manifest_rec.get("path")
        if not manifest_rel:
            problems.append("w1_registration_manifest_path_missing")
        else:
            manifest_path = root / manifest_rel
            declared = manifest_rec.get("sha256")
            actual = _file_hash(manifest_path)
            manifest_facts = {
                "path": audit.path_for_record(manifest_path),
                "declared_sha256": declared, "actual_sha256": actual,
                "match": actual is not None and actual == declared,
            }
            if actual is None:
                problems.append("training_manifest_missing")
            elif actual != declared:
                problems.append("training_manifest_hash_drifted")
            manifest = _safe_read(manifest_path, problems, "training_manifest")

        cache_rec = registration.get("cache")
        cache_rec = cache_rec if isinstance(cache_rec, Mapping) else {}
        cache_dir = root / cache_rec["dir"] if cache_rec.get("dir") else None
        for name, filename in (("identity", "cache_identity.json"),
                               ("root_stats", "root_stats.npz")):
            if not cache_rec.get("dir"):
                problems.append(f"cache_{name}_dir_missing")
                continue
            p = cache_dir / filename
            declared = (cache_rec.get("identity_sha256") if name == "identity"
                        else cache_rec.get("root_stats_sha256"))
            actual = _file_hash(p)
            cache_facts[name] = {
                "path": audit.path_for_record(p),
                "declared_sha256": declared, "actual_sha256": actual,
                "match": actual is not None and actual == declared,
            }
            if actual is None:
                problems.append(f"cache_{name}_missing")
            elif actual != declared:
                problems.append(f"cache_{name}_hash_drifted")

        configs_rec = registration.get("configs")
        configs_rec = configs_rec if isinstance(configs_rec, Mapping) else {}
        config_path = root / audit.TRAIN_CONFIG
        declared = configs_rec.get("config_file_sha256")
        actual = _file_hash(config_path)
        config_facts = {
            "path": audit.path_for_record(config_path),
            "declared_sha256": declared, "actual_sha256": actual,
            "match": actual is not None and actual == declared,
        }
        if actual is None:
            problems.append("train_config_missing")
        elif declared is not None and actual != declared:
            problems.append("train_config_drifted")

    for model, (rel, want, step) in audit.EXPECTED_CHECKPOINTS.items():
        p = root / rel
        actual = _file_hash(p)
        checkpoint_facts[model] = {
            "path": audit.path_for_record(p),
            "expected_sha256": want, "actual_sha256": actual,
            "step": int(step), "match": actual is not None and actual == want,
            "step_verified": False,
            "note": "step/usable/finite require the R0 strict torch.load",
        }
        if actual is None:
            problems.append(f"checkpoint_missing:{model}")
        elif actual != want:
            problems.append(f"checkpoint_drifted:{model}")

    assets_sha = ""
    if isinstance(manifest, Mapping):
        prereg_rec = manifest.get("preregistration")
        prereg_rec = prereg_rec if isinstance(prereg_rec, Mapping) else {}
        prereg_path = root / prereg_rec.get("path", "")
        declared = prereg_rec.get("sha256")
        actual = _file_hash(prereg_path)
        prereg_facts = {
            "path": audit.path_for_record(prereg_path),
            "declared_sha256": declared, "actual_sha256": actual,
            "match": actual is not None and actual == declared,
        }
        if actual is None:
            problems.append("preregistration_missing")
        elif declared is not None and actual != declared:
            problems.append("preregistration_hash_drifted")
        prereg = _safe_read(prereg_path, problems, "preregistration")
        if isinstance(prereg, Mapping):
            assets_sha = (prereg.get("assets") or {}).get("a1_assets_sha256") or ""
        prereg_facts["a1_assets_sha256"] = assets_sha or None
        prereg_facts["asset_closure_verified"] = False
        prereg_facts["asset_closure_note"] = "deferred to R0 under registration"

        split_path = root / SPLIT_TRAIN
        split_actual = _file_hash(split_path)
        split = _safe_read(split_path, problems, "split_train")
        manifest_selection = manifest.get("selection")
        manifest_selection = (manifest_selection
                              if isinstance(manifest_selection, Mapping) else {})
        train24_declared = list(manifest_selection.get("train24", []))
        train24_recomputed: list[str] = []
        canonical_declared = manifest_selection.get("train24_canonical_sha256")
        canonical_recomputed = None
        if isinstance(split, Mapping) and isinstance(split.get("demos"), list):
            try:
                from ireset.data import w1_training as w1
                train24_recomputed = w1.select_train24(list(split["demos"]))
                canonical_recomputed = w1.canonical_list_sha256(train24_recomputed)
            except Exception as exc:  # noqa: BLE001 -- recorded, not crashed
                problems.append(
                    f"train24_recompute_failed:{type(exc).__name__}")
        if train24_recomputed != train24_declared:
            problems.append("train24_not_reproducible")
        if (canonical_declared is not None
                and canonical_declared != canonical_recomputed):
            problems.append("train24_canonical_sha_mismatch")
        train24_facts = {
            "source": {
                "path": audit.path_for_record(split_path),
                "sha256": split_actual,
            },
            "declared": train24_declared,
            "recomputed": train24_recomputed,
            "match": train24_recomputed == train24_declared,
            "canonical_sha256_declared": canonical_declared,
            "canonical_sha256_recomputed": canonical_recomputed,
            "rule": "per subject first four by SHA256(prefix+demo); canonical order",
        }
    else:
        prereg_facts = {}
        train24_facts = {}

    keys: list[audit.WindowKey] = []
    selection_facts: dict = {}
    if isinstance(manifest, Mapping):
        try:
            from ireset.data import w1_training as w1
            code_sha = w1.condition_code_sha256(root)
        except Exception as exc:  # noqa: BLE001 -- recorded, not crashed
            problems.append(f"condition_code_hash_unavailable:{type(exc).__name__}")
            code_sha = ""
        if assets_sha and code_sha:
            keys, selection_problems, selection_facts = \
                _select_candidate_windows(
                    manifest, assets_sha=assets_sha, code_sha=code_sha)
            problems += selection_problems
        else:
            if not assets_sha:
                problems.append("prereg_assets_sha_missing")
            problems.append("window_candidates_skipped")

    noise = audit.generate_noise(
        audit.NOISE_SEED, (audit.WINDOW_COUNT, audit.WINDOW, 38))
    alpha = audit.independent_alpha_cumprod(1000, "scaled_linear", 1e-4, 0.02)
    weights = audit._audit_weights().as_dict()
    objective_keys = audit.required_objective_row_keys(keys)
    gradient_keys = audit.required_gradient_row_keys()
    environment = audit.environment_facts()
    subjects = sorted({str(k.demo).split("_")[0] for k in keys})
    review_notes: list[str] = []
    if keys and len(subjects) == 1:
        review_notes.append(
            f"both candidate demos are from subject {subjects[0]} under the "
            "frozen lexicographic rule; no cross-subject coverage is claimed")
    if not bool(environment.get("deterministic_algorithms")):
        review_notes.append(
            "torch deterministic algorithms are off at package time; R0 "
            "records the actual setting and F registers a mismatch rather "
            "than widening if bitwise replay cannot be reproduced")
    if len(keys) != audit.WINDOW_COUNT:
        review_notes.append(
            f"candidate window count is {len(keys)}, not {audit.WINDOW_COUNT}; "
            "the package is not executable")

    template = authorization_template()
    package = {
        "schema": SCHEMA,
        "status": {
            "mode": "pre_authorization_readonly",
            "authorized": False,
            "registered": False,
            "run": "not_run",
            "package_passed": not problems,
            "section_8_checked": False,
        },
        "passed": not problems,
        "problems": problems,
        "cohort": {
            "cohort_id": audit.COHORT_ID,
            "execution_id": audit.EXECUTION_ID,
            "audit_schema": audit.AUDIT_SCHEMA,
            "accounting_schema": audit.ACCOUNTING_SCHEMA,
            "history_root": str(audit.HISTORY_ROOT),
            "history_scan": "all registered attempts under the fixed root",
            "budget_transfer_from_old_runs": False,
            "stage_budgets": {k: dict(v) for k, v in audit.STAGE_BUDGETS.items()},
            "total_budget_s": audit.TOTAL_BUDGET_S,
            "seal_tail_reserve_s": audit.SEAL_TAIL_RESERVE_S,
            "seal_verify_rounds": audit.SEAL_VERIFY_ROUNDS,
            "call_caps": dict(audit.CALL_CAPS),
            "full_chain_calls": dict(audit.FULL_CHAIN_CALLS),
        },
        "call_matrix": _stage_call_matrix(),
        "tolerance_contract": {
            "path": (audit.path_for_record(tolerance_contract_path)
                     if tolerance_contract_path is not None else None),
            "sha256": tolerance_contract_sha256,
            "mode": tolerance_contract.get("mode"),
            "evidence_sha256": tolerance_contract.get("evidence_sha256"),
            "contract": dict(tolerance_contract),
        },
        "fixed_assets": {
            "anchors": anchors,
            "registration": {
                "path": audit.path_for_record(reg_path),
                "sha256": _file_hash(reg_path),
            },
            "manifest": manifest_facts,
            "cache": cache_facts,
            "preregistration": prereg_facts,
            "config": config_facts,
            "checkpoints": checkpoint_facts,
        },
        "window_selection": {
            "rule": ("frozen train24 metadata; demo name lexicographic; motion "
                     "admitted; first two full T=60 windows per demo"),
            "train24": train24_facts,
            "candidates": [k.as_dict() for k in keys],
            "per_demo": selection_facts,
            "count": len(keys),
            "subjects": subjects,
            "same_subject": bool(keys) and len(subjects) == 1,
            "status": "candidate_pending_R0_registration",
            "substitution_after_values": False,
        },
        "timesteps": [int(t) for t in audit.DIFFUSION_TIMESTEPS],
        "noise": {
            "seed": audit.NOISE_SEED, "device": "cpu",
            "shape": [audit.WINDOW_COUNT, audit.WINDOW, 38],
            "dtype": "float32", "draws": 1, "protocol": audit.NOISE_PROTOCOL,
            "sha256": audit._hash_array(noise),
        },
        "alpha": {
            "schedule": "scaled_linear", "beta_start": 1e-4, "beta_end": 0.02,
            "num_train_timesteps": 1000, "sha256": audit._hash_array(alpha),
        },
        "models": list(audit.MODELS),
        "weights": weights,
        "effective_config": {
            "weights": weights, "micro_batch": audit.WINDOW_COUNT,
            "grad_accum": 4, "window": audit.WINDOW, "stride": audit.STRIDE,
            "models": list(audit.MODELS),
            "contact_supervision": "disabled",
            "audit_scope": "pre-accumulation micro objective; no optimizer step",
        },
        "array_schema": audit._array_schema(),
        "gradient_schema": audit._gradient_schema(),
        "required_rows": {
            "objective_count": len(objective_keys),
            "gradient_count": len(gradient_keys),
            "objective_keys": objective_keys,
            "gradient_keys": gradient_keys,
            "expected_objective_count": 98,
            "expected_gradient_count": 14,
        },
        "artifacts": {
            "registration": ["registration.json", "window_keys.json",
                             "tolerance_contract.json", "admission.json",
                             "r0_report.json"],
            "production": ["p_report.json", "objective_real_rows.json",
                           "objective_real_arrays.npz",
                           "gradient_real_rows.json",
                           "gradient_real_arrays.npz", "call_trace.json",
                           "rng_state_facts.json"],
            "finalize": ["audit_summary.json", "independent_verification.json"],
        },
        "coverage": {
            "objective_coverage": "partial",
            "audited": [
                "clip_encode", "root_standardization", "joint_mapping",
                "pos", "rot6d", "joint", "geodesic_rad2", "vel",
                "fk_m2_aggregation_only", "total_arithmetic",
                "output_space_gradients_five_terms_and_L5",
                "contact_branch_off",
            ],
            "not_audited": [
                "real_fk_geometry", "fk_derivative", "network_parameter_gradient",
                "accumulation", "gradient_clip", "adamw", "optimizer_update",
                "model_selection", "ddim_sampling",
            ],
        },
        "boundaries": dict(audit.BOUNDARIES),
        "known_limitations": _KNOWN_LIMITATIONS,
        "authorization": {
            "required": True,
            "authorized": False,
            "authorization_sha256": None,
            "required_fields": list(audit.AUTHORIZATION_REQUIRED),
            "checklist": {k: None for k in audit.AUTHORIZATION_REQUIRED},
            "budgets_expected": {
                stage: {"budget_s": spec["budget_s"],
                        "attempts": spec["attempts"]}
                for stage, spec in audit.STAGE_BUDGETS.items()
            },
            "template": template,
            "entry": (
                "uv run --locked python scripts/audit_w1_real_objective.py "
                "--real --out derived/experiments/omniretarget_first/<run> "
                "--authorization <authorization.json> "
                "--tolerance-contract derived/audit/w1_real_objective_audit/"
                "tolerance_contract.json"
            ),
            "refusals": [
                "schema is not w1_real_objective_authorization_v1",
                "authorized is not exactly true",
                "any checklist item is not exactly true",
                "training_updates/parameter_fd/ddim_calls are not JSON integer 0",
                "budget_s is not the exact frozen JSON number",
                "attempts is not the exact frozen JSON integer",
                "budget stages are not exactly R0/P/F",
                "tolerance contract is not a completed synthetic calibration",
                "run root is not a direct child of the fixed history root",
            ],
        },
        "implementation": audit.implementation_identity(root),
        "generator": {
            "schema": "w1_real_admission_generator_v1",
            "files": {
                rel: _file_hash(root / rel) for rel in (
                    "src/ireset/eval/w1_real_admission.py",
                    "scripts/w1_real_objective_admission.py",
                    "tests/test_w1_real_admission.py",
                )
            },
            "calibration_contract_sha256": tolerance_contract_sha256,
            "reads": "metadata files only; no arrays, schedules, checkpoints or models",
        },
        "environment": environment,
        "review_notes": review_notes,
        "read_scope": {
            "metadata_only": True,
            "window_values_loaded": False,
            "schedules_read": False,
            "cache_contents_loaded": False,
            "checkpoint_torch_load": False,
            "conditioner_forward_calls": 0,
            "model_forward_calls": 0,
            "vjp_calls": 0,
            "asset_closure_verified": False,
            "asset_closure_deferred_to": "R0 under registration",
            "cohort_created": False,
            "registered_attempt_created": False,
            "note": ("file hashes and frozen JSON metadata only; no array, "
                     "schedule, checkpoint or model was opened"),
        },
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return package


def write_package(out_root: Path, package: Mapping[str, Any]
                  ) -> tuple[Path, str]:
    """Write into a fresh attempt directory; never overwrite a prior package."""
    out = Path(out_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = 1
    while (out / f"attempt_{index:03d}").exists():
        index += 1
    attempt = out / f"attempt_{index:03d}"
    attempt.mkdir()
    body = dict(package)
    body["attempt"] = attempt.name
    path = attempt / "admission_package.json"
    audit.write_json(path, body)
    return path, audit.sha256_file(path)
