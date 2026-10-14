"""Independent W1-D/E closure from registered histories and product bytes.

Training completion, evaluation completeness, negative motion outcomes and
final interaction readiness remain separate. Partial/crashed evidence is
reported as incomplete; it never becomes an empty successful comparison.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import omniretarget_w1_de as driver  # noqa: E402
from omniretarget_w1_de_detail import (  # noqa: E402
    _begin_attempt, _check_deadline, _expected_identity, _file_record, _finish,
    _json_safe, _load_clip, _load_indexed_report, _read_json, _record_path,
    _representative_cases, _run_unit,
)

REPLAY_TOL = 1e-5


def _finite(value, *, nonnegative=False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not nonnegative or value >= 0))


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _compare_evidence(recorded, fresh, prefix="", *, atol=1e-6, rtol=1e-6) -> list[str]:
    """Unknown/nonfinite scalars never participate in a tolerance comparison."""
    path = prefix or "evidence"
    if isinstance(fresh, dict):
        if not isinstance(recorded, dict) or set(recorded) != set(fresh):
            return [f"{path}:fields_differ"]
        return [p for k, v in fresh.items() for p in
                _compare_evidence(recorded[k], v, f"{path}.{k}", atol=atol, rtol=rtol)]
    if isinstance(fresh, list):
        if not isinstance(recorded, list) or len(recorded) != len(fresh):
            return [f"{path}:length_or_type_differ"]
        return [p for i, v in enumerate(fresh) for p in
                _compare_evidence(recorded[i], v, f"{path}[{i}]", atol=atol, rtol=rtol)]
    if isinstance(fresh, bool) or fresh is None or isinstance(fresh, str):
        return [] if type(recorded) is type(fresh) and recorded == fresh else [f"{path}:value_differ"]
    if isinstance(fresh, int):
        return [] if type(recorded) is int and recorded == fresh else [f"{path}:count_differ"]
    if isinstance(fresh, float):
        if not _finite(recorded) or not _finite(fresh):
            return [f"{path}:not_finite"]
        return [] if abs(recorded - fresh) <= atol + rtol * abs(fresh) else [f"{path}:not_reproducible"]
    return [] if recorded == fresh else [f"{path}:value_differ"]


def _cadence(steps: int, every: int, target: int) -> list[int]:
    if not _count(steps) or not _count(every) or every == 0:
        raise ValueError("invalid_validation_cadence")
    scheduled = list(range(every, steps + 1, every))
    if steps == target and steps > 0 and steps not in scheduled:
        scheduled.append(steps)
    return scheduled


def _training_evidence(run: Path, project: Path, reg: dict, model: str,
                       *, deadline=None) -> dict:
    """Check the actual steps, all histories, configurations and checkpoint bytes."""
    import eval_generation as raw
    from ireset.data import a2_integrity as integ
    from ireset.data import w1_training as w1

    row, problems = driver.stage_state(run, driver.STAGES[f"train-{model}"], project)
    out = {"attempt": (row or {}).get("attempt"), "complete": False,
           "evidence_complete": False, "problems": list(problems),
           "steps": None, "target_steps": reg["configs"][model].get("steps"),
           "selection_status": "unknown", "loss_first_last": {"first": None, "last": None}}
    if row is None:
        return out
    attempt = run / driver.STAGE_DIRS[driver.STAGES[f"train-{model}"]] / row["attempt"]
    summary = {}
    try:
        _check_deadline(deadline)
        summary = _read_json(attempt / "summary.json")
        out.update(steps=summary.get("steps"), stop_reason=summary.get("stop_reason"),
                   verdict=row.get("verdict"), recorded_complete=row.get("complete"))
        history = _read_json(attempt / "loss_history.json", list)
        val = _read_json(attempt / "val_history.json", list)
        formal = _read_json(attempt / "formal_history.json", list)
        checkpoints = _read_json(attempt / "checkpoint_index.json", list)
        required = ["summary.json", "loss_history.json", "val_history.json",
                    "formal_history.json", "checkpoint_index.json"]
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict):
            out["problems"].append("training_artifact_hashes_missing")
            artifacts = {}
        out["artifacts"] = {}
        for name in required:
            path = attempt / name
            record = _file_record(path, project)
            out["artifacts"][name] = record
            if name == "summary.json":
                if (not isinstance(row.get("summary"), str)
                        or (project / row["summary"]).resolve() != path.resolve()
                        or row.get("summary_sha256") != record["sha256"]):
                    out["problems"].append("training_summary_unregistered")
            elif artifacts.get(record["path"]) != record["sha256"]:
                out["problems"].append(f"training_artifact_unregistered:{name}")
        config = reg["configs"][model]
        if summary.get("config") != config:
            out["problems"].append("effective_config_differs_from_registration")
        if summary.get("identity", {}).get("config") != config:
            out["problems"].append("summary_identity_config_mismatch")
        if summary.get("debug_subset") is not None:
            out["problems"].append("formal_training_debug_subset")
        steps, target = summary.get("steps"), config["steps"]
        if not _count(steps) or steps > target or target <= 0:
            raise ValueError("actual_step_count_invalid")
        if summary.get("val_history") != val or summary.get("formal_history") != formal:
            out["problems"].append("summary_history_serializations_disagree")
        if summary.get("checkpoint_index") != checkpoints:
            out["problems"].append("summary_checkpoint_index_disagrees")
        if any(not isinstance(h, dict) for h in history) or [h.get("step") for h in history] != list(range(1, steps + 1)):
            out["problems"].append("loss_step_history_incomplete")
        parts = set(w1.S0_REQUIRED_LOSS_KEYS)
        for index, h in enumerate(history):
            if not isinstance(h, dict):
                continue
            if any(k not in h or not _finite(h[k]) for k in parts | {"gnorm", "lr", "ms"}):
                out["problems"].append(f"loss_fields_missing_or_nonfinite:{index + 1}")
            if (h.get("n_micro") != config["grad_accum"]
                    or h.get("n_samples") != config["micro_batch"] * config["grad_accum"]):
                out["problems"].append(f"optimizer_microbatch_counts:{index + 1}")
            if h.get("lr") != config["lr"] or not _finite(h.get("ms"), nonnegative=True):
                out["problems"].append(f"optimizer_lr_or_time:{index + 1}")
            if "contact" in h:
                out["problems"].append(f"contact_branch_not_silent:{index + 1}")
        manifest = _read_json(project / reg["training_manifest"]["path"])
        manifest_rows = manifest.get("rows", [])
        roster = reg["rosters"]
        if any(not isinstance(r, dict) for r in manifest_rows):
            raise ValueError("manifest_rows_malformed")
        ids = [r.get("demo") for r in manifest_rows]
        if len(ids) != len(set(ids)) or set(ids) != set(roster["train24"]) | set(roster["val31"]):
            out["problems"].append("manifest_registered_roster_mismatch")
        windows, lengths = {}, {}
        for split, roster_key in (("train", "train24"), ("val", "val31")):
            chosen = [r for r in manifest_rows if r.get("demo") in roster[roster_key]]
            if any(r.get("split") != split or r.get("admission", {}).get("motion") is not True for r in chosen):
                out["problems"].append(f"{split}_dataset_membership_or_admission")
            windows[split] = sum(w1.full_windows(r["T"]) for r in chosen
                                 if r.get("admission", {}).get("motion") is True)
            lengths.update({r["demo"]: r["T"] for r in chosen})
        if summary.get("n_train_windows") != windows["train"] or summary.get("n_val_windows") != windows["val"]:
            out["problems"].append("train_or_val_window_count_mismatch")
        loader = summary.get("train_loader")
        if not isinstance(loader, dict):
            raise ValueError("train_loader_evidence_missing")
        n_batches = windows["train"] // config["micro_batch"]
        n_micro = sum(h.get("n_micro", 0) for h in history if isinstance(h, dict) and _count(h.get("n_micro")))
        n_samples = sum(h.get("n_samples", 0) for h in history if isinstance(h, dict) and _count(h.get("n_samples")))
        want_loader = {"n_windows": windows["train"], "micro_batch": config["micro_batch"],
                       "batches_per_epoch": n_batches,
                       "drop_last_discarded_per_epoch": windows["train"] - n_batches * config["micro_batch"]}
        for k, v in want_loader.items():
            if loader.get(k) != v:
                out["problems"].append(f"train_loader:{k}")
        if summary.get("stop_reason") is None:
            if loader.get("micro_batches_total") != n_micro or loader.get("samples_total") != n_samples:
                out["problems"].append("train_loader_completed_counts_disagree")
            expected_epochs = max(1, (n_micro + n_batches - 1) // n_batches) if n_batches else 0
            if loader.get("epochs_started") != expected_epochs:
                out["problems"].append("train_loader_epochs_started")
        contract = reg["configs"]["formal_validation_contract"]
        expected_val_steps = _cadence(steps, config["val_every"], target)
        expected_formal_steps = _cadence(steps, contract["full_generation_every_optimizer_steps"], target)
        if any(not isinstance(e, dict) for e in val) or [e.get("step") for e in val] != expected_val_steps:
            out["problems"].append("val_schedule_incomplete")
        for e in val:
            if not isinstance(e, dict):
                continue
            if (any(k not in e or not _finite(e[k]) for k in parts)
                    or e.get("n_windows_evaluated") != windows["val"]
                    or e.get("n_val_windows_total") != windows["val"]
                    or e.get("n_batches") != math.ceil(windows["val"] / config["micro_batch"])
                    or e.get("complete") is not True):
                out["problems"].append(f"val_evidence_incomplete:{e.get('step')}")
        if any(not isinstance(e, dict) for e in formal) or [e.get("step") for e in formal] != expected_formal_steps:
            out["problems"].append("formal_schedule_incomplete")
        formal_evals = []
        for e in formal:
            if not isinstance(e, dict):
                continue
            rows = e.get("per_demo")
            valid_rows = isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
            ids = [r.get("demo") for r in rows] if valid_rows else []
            identity_ok = (valid_rows and len(ids) == len(roster["val31"])
                           and len(ids) == len(set(ids)) and set(ids) == set(roster["val31"])
                           and e.get("n_expected") == len(roster["val31"])
                           and e.get("seed") == reg["evaluation"]["main_seed"]
                           and e.get("model_type") == model)
            vals = [r["mpjpe_m"] for r in rows
                    if r.get("status") == "ok" and r.get("numerical_valid") is True
                    and r.get("T") == lengths.get(r.get("demo"))
                    and _finite(r.get("mpjpe_m"), nonnegative=True)] if valid_rows else []
            full = bool(identity_ok and len(vals) == len(roster["val31"]))
            macro = sum(vals) / len(vals) if full else None
            if not identity_ok:
                out["problems"].append(f"formal_matrix_identity:{e.get('step')}")
            if (e.get("complete") is not full or e.get("n_ok") != len(vals)
                    or (macro is None and e.get("macro_mpjpe_m") is not None)
                    or (macro is not None and (not _finite(e.get("macro_mpjpe_m"))
                        or abs(e["macro_mpjpe_m"] - macro) > 1e-12))):
                out["problems"].append(f"formal_macro_not_reproducible:{e.get('step')}")
            formal_evals.append({"step": e.get("step"), "complete": full,
                                 "macro": macro, "n_defined": len(vals)})
        best_step, best_macro = driver.recompute_best_selection(
            formal, expected_demos=roster["val31"], demo_lengths=lengths,
            every=contract["full_generation_every_optimizer_steps"],
            seed=reg["evaluation"]["main_seed"])
        status = "best_by_formal_macro" if best_step is not None else "no_qualified_best"
        if (summary.get("best_formal_step") != best_step or summary.get("selection_status") != status
                or summary.get("best_formal_macro_mpjpe_m") != best_macro):
            out["problems"].append("summary_selection_not_reproducible")
        names = [r.get("name") for r in checkpoints if isinstance(r, dict)]
        want_names = {"last.pt"} | ({"best.pt"} if best_step is not None else set())
        if len(names) != len(checkpoints) or len(names) != len(set(names)) or set(names) != want_names:
            out["problems"].append("checkpoint_index_missing_duplicate_or_extra")
        ck_records = {}
        for entry in checkpoints:
            if not isinstance(entry, dict) or entry.get("name") not in want_names:
                continue
            _check_deadline(deadline)
            name = entry["name"]
            path = attempt / name
            want_step = best_step if name == "best.pt" else steps
            sha = integ.sha256_file(path)
            if sha != entry.get("sha256") or entry.get("step") != want_step:
                out["problems"].append(f"checkpoint_index_mismatch:{name}")
            if artifacts.get(_record_path(path, project)) != sha:
                out["problems"].append(f"checkpoint_unregistered:{name}")
            selection = {"path": str(path), "sha256": sha, "step": want_step, "model": model,
                         "role": "best" if name == "best.pt" else "last",
                         "checkpoint_role": "best" if name == "best.pt" else "last_diagnostic",
                         "formal_attempt": attempt.name}
            try:
                generator = raw.load_w1_de_generator(
                    selection, "cpu", 25, project / reg["cache"]["dir"],
                    reg["training_manifest"]["sha256"], reg["cache"]["identity_sha256"],
                    reg["cache"]["root_stats_sha256"], integ.sha256_file(run / "freeze/execution_registration.json"),
                    registration=reg, project_root=project)
                del generator
            except (Exception, SystemExit) as exc:
                out["problems"].append(f"checkpoint_deep_load:{name}:{type(exc).__name__}:{exc}")
            ck_records[name] = {"path": _record_path(path, project), "sha256": sha, "step": want_step}
        totals = [h["total"] for h in history if isinstance(h, dict) and _finite(h.get("total"))]
        out.update(selection_status=status, best_formal_step=best_step,
                   best_formal_macro_mpjpe_m=best_macro, checkpoints=ck_records,
                   formal_evals=formal_evals, dataset_windows=windows,
                   optimizer_steps_recorded=len(history), micro_batches_recorded=n_micro,
                   samples_recorded=n_samples, loss_first_last={
                       "first": sum(totals[:100]) / len(totals[:100]) if totals else None,
                       "last": sum(totals[-100:]) / len(totals[-100:]) if totals else None})
        evidence_complete = not out["problems"]
        out["evidence_complete"] = evidence_complete
        out["complete"] = bool(evidence_complete and steps == target
                               and summary.get("stop_reason") is None
                               and summary.get("complete") is True
                               and summary.get("budget_stopped") is False
                               and not summary.get("failure") and not summary.get("preservation_errors")
                               and row.get("complete") is True and row.get("status") == "done")
    except TimeoutError:
        raise
    except (Exception, SystemExit) as exc:
        out["problems"].append(f"training_evidence_unreadable:{type(exc).__name__}:{exc}")
    return _json_safe(out)


def _replay_arrays(root, joints, stored_root, stored_joints, target, stored_target,
                   *, T: int, tol: float = REPLAY_TOL) -> dict:
    import numpy as np

    shapes = (np.shape(root) == np.shape(stored_root) == (T, 7)
              and np.shape(joints) == np.shape(stored_joints) == (T, 29)
              and np.shape(target) == np.shape(stored_target) == (T, 7))
    finite = shapes and all(np.isfinite(a).all() for a in
                           (root, joints, stored_root, stored_joints, target, stored_target))
    out = {"shape_valid": bool(shapes), "all_finite": bool(finite),
           "within_tolerance": False, "max_root_abs_diff": None,
           "max_joint_abs_diff": None, "box_binding_bitwise": False}
    if not finite or not _finite(tol, nonnegative=True):
        return out
    dr = float(np.abs(root - stored_root).max())
    dq = float(np.abs(joints - stored_joints).max())
    box = bool(np.array_equal(target, stored_target))
    out.update(max_root_abs_diff=dr, max_joint_abs_diff=dq,
               box_binding_bitwise=box, within_tolerance=bool(dr <= tol and dq <= tol and box))
    return out


def _replay_negative_output(exc, arrays: dict, stored_metadata: dict, *,
                            method: str, seed: int, T: int, tol=REPLAY_TOL) -> dict:
    """Reproduce a structured rejection without treating NaNs as toleranced values."""
    import numpy as np
    import torch
    import eval_generation as raw
    from ireset.model.generator import GenerationNumericalError
    from ireset.model.stitch import window_starts

    out = {"verified": False, "failure_arrays_match": False,
           "retry_facts_match": False, "numerical_rejection_reproduced": False}
    if not isinstance(exc, GenerationNumericalError) or not _finite(tol, nonnegative=True):
        return out
    metadata = exc.metadata
    starts = window_starts(T, 60, 30)
    if (metadata.get("starts") != starts or arrays["window_starts"].tolist() != starts
            or metadata.get("numerical_error") != stored_metadata.get("numerical_error")):
        return out
    retry, old_retry = raw._w1_de_retry(metadata), raw._w1_de_retry(stored_metadata)
    if (raw._w1_de_retry_problems(retry, method, seed, len(starts))
            or raw._w1_de_retry_problems(old_retry, method, seed, len(starts))):
        return out
    def facts(value):
        if isinstance(value, dict):
            return {k: facts(v) for k, v in value.items() if k != "wall_s"}
        if isinstance(value, list):
            return [facts(v) for v in value]
        return value
    out["retry_facts_match"] = not _compare_evidence(facts(retry), facts(old_retry))
    output = exc.output
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    if isinstance(output, np.ndarray):
        fresh = {"normalized_windows": output}
    elif isinstance(output, (tuple, list)) and len(output) in (2, 3):
        names = ("root", "joints") if len(output) == 2 else ("decoded_pos", "decoded_quat", "decoded_joints")
        fresh = {k: np.asarray(v) for k, v in zip(names, output)}
    else:
        return out
    expected = {k for k in ("normalized_windows", "root", "joints", "decoded_pos",
                            "decoded_quat", "decoded_joints") if k in arrays}
    if set(fresh) != expected or not expected:
        return out
    for key, a in fresh.items():
        b = arrays[key]
        if a.shape != b.shape or not a.size:
            return out
        masks = (np.isfinite, np.isnan, np.isposinf, np.isneginf)
        if any(not np.array_equal(fn(a), fn(b)) for fn in masks):
            return out
        finite = np.isfinite(a)
        if finite.any() and np.abs(a[finite] - b[finite]).max() > tol:
            return out
    out["failure_arrays_match"] = True
    if "normalized_windows" in fresh or "decoded_pos" in fresh:
        rejected, numerical = raw._w1_de_window_rejection(fresh, metadata, T)
        old_rejected, old_numerical = raw._w1_de_window_rejection(arrays, stored_metadata, T)
        reproduced = (rejected and old_rejected and numerical == old_numerical
                      and numerical.get("final_invalid_indices_match_output") is not False)
    else:
        a, q = fresh["root"], fresh["joints"]
        reproduced = (a.shape != (T, 7) or q.shape != (T, 29)
                      or not np.isfinite(a).all() or not np.isfinite(q).all()
                      or not np.all(np.abs(np.linalg.norm(a[:, 3:7], axis=-1) - 1) <= 1e-5))
    out["numerical_rejection_reproduced"] = bool(reproduced)
    out["verified"] = bool(reproduced and out["retry_facts_match"])
    return out


def _checkpoint_replay_unit(payload: dict) -> dict:
    import numpy as np
    import torch
    import eval_generation as raw
    from ireset.data import a2_integrity as integ

    project, deadline = Path(payload["project_root"]), payload.get("deadline")
    reg_path = Path(payload["registration_path"])
    reg = _read_json(reg_path)
    if integ.sha256_file(reg_path) != payload["registration_sha256"]:
        raise ValueError("registration_hash_mismatch")
    clip, manifest_row = _load_clip(reg, project, payload["demo"])
    cond, T = clip.condition.truncate(clip.T), clip.T
    model, selection = payload["model"], payload["selection"]
    row = payload["row"]
    expected_key = f"{model}|{payload['demo']}|{reg['evaluation']['main_seed']}"
    if row.get("key") != expected_key or row.get("matrix") != "main":
        raise ValueError("replay_main_key_mismatch")
    if selection.get("role") == "unavailable":
        return {"status": "checkpoint_unavailable", "verified": False}
    path = project / selection["path"]
    if integ.sha256_file(path) != selection["sha256"]:
        raise ValueError("checkpoint_sha_mismatch")
    product_path = project / row["product"]["path"]
    verification = raw.verify_w1_de_product(
        product_path, row, reg, cond, project_root=project,
        expected_identity=_expected_identity(
            reg, {"models": {model: selection}}, row, cond, manifest_row, project,
            payload["registration_sha256"]))
    if verification.get("passed") is not True:
        raise ValueError(f"replay_product_invalid:{verification.get('problems')}")
    arrays = verification["arrays"]
    _check_deadline(deadline)
    device = reg["configs"][model].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"status": "replay_device_unavailable", "verified": False}
    generator = raw.load_w1_de_generator(
        {**selection, "model": model}, device, 25, project / reg["cache"]["dir"],
        reg["training_manifest"]["sha256"], reg["cache"]["identity_sha256"],
        reg["cache"]["root_stats_sha256"], payload["registration_sha256"],
        registration=reg, project_root=project)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    try:
        traj = generator.sample(cond, seed=reg["evaluation"]["main_seed"])
    except Exception as exc:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        _check_deadline(deadline)
        metadata, evidence = getattr(exc, "metadata", None), getattr(exc, "evidence", None)
        negative = _replay_negative_output(
            exc, arrays, verification["metadata"], method=model,
            seed=reg["evaluation"]["main_seed"], T=T)
        stable = (integ.sha256_file(path) == selection["sha256"]
                  and integ.sha256_file(product_path) == row["product"]["sha256"])
        return {"status": "numerical_rejected" if isinstance(metadata, dict) else "replay_failed",
                **negative, "verified": bool(negative["verified"] and stable
                                              and row.get("status") == "numerical_rejected"),
                "reason": f"{type(exc).__name__}:{exc}", "metadata": metadata,
                "evidence": evidence, "source_hashes_stable": stable,
                "full_length_finite_replay": False,
                "note": "matching negative arrays and retry facts; finite pose replay unavailable"}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _check_deadline(deadline)
    checks = _replay_arrays(traj.root, traj.q, arrays["root"], arrays["joints"],
                            cond.target_obj_pose, arrays["target_box"], T=T)
    starts = np.asarray(traj.metadata.get("starts", []))
    starts_same = bool(np.array_equal(starts, arrays["window_starts"]))
    covers = bool(checks["shape_valid"] and bool(arrays["covers_last_frame"]))
    still_hashed = (integ.sha256_file(path) == selection["sha256"]
                    and integ.sha256_file(product_path) == row["product"]["sha256"])
    return {"status": "replayed", "T": T, "step": selection["step"],
            "role": selection["checkpoint_role"], "seed": reg["evaluation"]["main_seed"],
            **checks, "window_starts_match": starts_same, "covers_last_frame": covers,
            "source_hashes_stable": still_hashed,
            "verified": bool(checks["within_tolerance"] and starts_same and covers and still_hashed)}


def _replay_selected_checkpoints(run: Path, project: Path, reg: dict,
                                 raw_report: dict, selection: dict, *,
                                 attempt_dir: Path, deadline: float) -> dict:
    val6 = reg.get("rosters", {}).get("val6")
    if not isinstance(val6, list) or not val6:
        return {"models": {}, "complete": False, "problems": ["registered_val6_empty"]}
    # Identity order, not best fit or quality, fixes the full-clip replay demo.
    demo = sorted(val6)[0]
    out = {"demo": demo, "models": {}, "complete": False, "problems": []}
    from ireset.data.a2_integrity import sha256_file

    reg_path = run / "freeze/execution_registration.json"
    for model in driver.MODELS:
        hits = [r for r in raw_report.get("rows", []) if isinstance(r, dict)
                and r.get("key") == f"{model}|{demo}|{reg['evaluation']['main_seed']}"]
        selected = selection.get("models", {}).get(model)
        if len(hits) != 1 or not isinstance(selected, dict) or selected.get("role") == "unavailable":
            out["models"][model] = {"status": "checkpoint_or_main_product_unavailable", "verified": False}
            continue
        try:
            out["models"][model] = _run_unit("checkpoint", {
                "project_root": str(project), "registration_path": str(reg_path),
                "registration_sha256": sha256_file(reg_path), "demo": demo,
                "model": model, "selection": selected, "row": hits[0]},
                deadline, attempt_dir, f"checkpoint__{model}")
        except TimeoutError as exc:
            out["models"][model] = {"status": "timeout", "verified": False, "error": str(exc)}
        except (Exception, SystemExit) as exc:
            out["models"][model] = {"status": "replay_failed", "verified": False,
                                     "error": f"{type(exc).__name__}:{exc}"}
    out["complete"] = (set(out["models"]) == set(driver.MODELS)
                       and all(out["models"][m].get("verified") is True for m in driver.MODELS))
    return out


def _verify_detail(report: dict, reg: dict, raw_report_path: Path | None,
                   raw_rows: list[dict], project: Path) -> dict:
    import numpy as np
    import eval_generation as raw
    from ireset.data.a2_integrity import sha256_file

    expected = [r for r in raw.w1_de_expected_matrix(reg)
                if r["matrix"] == "main" and r["demo"] in reg["rosters"]["val6"]]
    rows, problems = report.get("rows"), []
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        return {"complete": False, "problems": ["detail_rows_missing_or_malformed"],
                "expected": len(expected), "processed": 0}
    keys = [r.get("key") for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {r["key"] for r in expected}:
        problems.append("detail_matrix_missing_duplicate_or_foreign")
    if report.get("val6") != reg["rosters"]["val6"]:
        problems.append("detail_val6_roster_mismatch")
    if raw_report_path is None or not raw_report_path.is_file():
        problems.append("detail_raw_report_missing")
    else:
        source = report.get("raw_eval_report")
        if (not isinstance(source, dict) or source.get("sha256") != sha256_file(raw_report_path)
                or (project / source.get("path", "")).resolve() != raw_report_path.resolve()):
            problems.append("detail_raw_report_binding_mismatch")
    raw_by_key = {r.get("key"): r for r in raw_rows if isinstance(r, dict)}
    processed = 0
    for row in rows:
        key, source = row.get("key"), raw_by_key.get(row.get("key"))
        if not source or row.get("scene_product") != source.get("product"):
            problems.append(f"{key}:detail_source_product_mismatch")
            continue
        if row.get("status") not in ("ok", "numerical_rejected") or row.get("consistent_with_raw") is not True:
            problems.append(f"{key}:detail_row_unverified")
            continue
        if row.get("raw_standard_pass") is not source.get("raw_standard_pass"):
            problems.append(f"{key}:detail_pass_flag_drifted")
        if raw.compare_w1_de_row_metrics(row.get("recomputed", {}), source):
            problems.append(f"{key}:detail_metrics_drifted")
        required = ("reference",) if row.get("status") == "numerical_rejected" else ("curves", "reference")
        for field in required:
            rec = row.get(field)
            if not isinstance(rec, dict) or not isinstance(rec.get("path"), str):
                problems.append(f"{key}:{field}_record_missing")
                continue
            path = project / rec["path"]
            if not path.is_file() or sha256_file(path) != rec.get("sha256"):
                problems.append(f"{key}:{field}_hash_mismatch")
                continue
            try:
                with np.load(path, allow_pickle=False) as z:
                    if field == "reference":
                        human, box = z["human_kpts_world"], z["target_box"]
                        if (human.shape != (row["T"], 52, 3) or box.shape != (row["T"], 7)
                                or not np.isfinite(human).all() or not np.isfinite(box).all()):
                            raise ValueError("reference_shape_or_finite")
                    else:
                        if z["task_world"].shape != (row["T"],) or not np.isfinite(z["task_world"]).all():
                            raise ValueError("task_curve_shape_or_finite")
                        if not np.array_equal(z["both_certified_overlap"],
                                              z["both_reference"] & z["hand_certified_l"] & z["hand_certified_r"]):
                            raise ValueError("both_intersection")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                problems.append(f"{key}:{field}_invalid:{type(exc).__name__}")
        processed += 1
    cases = _representative_cases(rows, reg["rosters"]["val6"])
    recorded = report.get("representative_cases")
    if not isinstance(recorded, dict) or set(recorded) != set(cases):
        problems.append("representative_case_records_missing")
    else:
        for name, case in cases.items():
            rec = recorded[name]
            if not isinstance(rec, dict) or rec.get("demo") != case["demo"]:
                problems.append(f"{name}:not_dictionary_first")
            elif case["demo"] is not None:
                gif = rec.get("gif")
                if (rec.get("complete") is not True or not isinstance(gif, dict)
                        or not (project / gif.get("path", "")).is_file()
                        or sha256_file(project / gif["path"]) != gif.get("sha256")):
                    problems.append(f"{name}:render_missing_or_hash_mismatch")
            elif not rec.get("reason"):
                problems.append(f"{name}:absent_class_reason_missing")
    if report.get("complete") is not True or report.get("budget_stopped"):
        problems.append("detail_registered_incomplete")
    return {"complete": bool(expected and processed == len(expected) and not problems),
            "expected": len(expected), "processed": processed, "problems": problems}


def _route_judgment(training: dict, evaluation: dict, *, main_seed=7) -> dict:
    groups = evaluation.get("per_method", {})
    evidence = []
    candidates = []
    for model in driver.MODELS:
        group = groups.get(f"{model}|{main_seed}", {})
        counts = group.get("counts", {})
        fit = group.get("macros", {}).get("teacher_fit_fk_mpjpe_box_m", {})
        evidence.append({"model": model, "counts": counts, "teacher_fit": fit,
                         "train_loss": training.get("models", {}).get(model, {}).get("loss_first_last")})
        if counts.get("numerical_rejected", 0):
            candidates.append("learning/normalization/condition/inference implementation")
        if counts.get("strict_rejected", 0) or counts.get("task_rejected", 0):
            candidates.append("prediction/stitching/condition and known teacher defects")
    complete = evaluation.get("evaluation_complete") is True and training.get("training_complete") is True
    return {"primary_attribution": "attribution insufficient",
            "attribution_status": "insufficient_evidence",
            "candidate_explanations": sorted(set(candidates)), "evidence": evidence,
            "label_principal_bottleneck_supported": False,
            "distinct_train_examples_with_paired_defect_evidence": 0,
            "reason": "no >=3 different registered train demonstrations with paired Teacher/model defect and error-scale evidence",
            "next_candidate": ("inspect paired failure families and prediction errors before registering the next experiment"
                               if complete else "resolve registered evidence gaps before choosing the next experiment"),
            "auto_execute": False}


def stage_finalize(args) -> int:
    import eval_generation as raw
    from ireset.data import a2_integrity as integ
    from ireset.paths import PROJECT_ROOT

    t0, project = time.perf_counter(), Path(PROJECT_ROOT)
    run = integ.guard_output_dir(Path(args.run_root))
    reg_path = run / "freeze/execution_registration.json"
    reg = _read_json(reg_path)
    entry_problems = driver.verify_execution_registration(run, project)
    budget = driver._budget_state(run, "detail_f")
    deadline = t0 + budget["remaining_s"] - budget["reserve_s"]
    attempt, output = _begin_attempt(run, "finalize", "w1_de_final.json", project)
    record = {"stage": "W1-D/E independent closure", "complete": False,
              "problems": list(entry_problems), "budget_stopped": False,
              "motion_baseline": "incomplete"}
    artifacts = {}
    try:
        matrix = raw.w1_de_expected_matrix(reg)
        raw_att, raw_path, raw_report, raw_problems = _load_indexed_report(run, "raw-eval", project)
        detail_att, detail_path, detail_report, detail_problems = _load_indexed_report(run, "detail", project)
        training = {"models": {}}
        for model in driver.MODELS:
            try:
                training["models"][model] = _training_evidence(run, project, reg, model, deadline=deadline)
            except TimeoutError as exc:
                training["models"][model] = {"complete": False, "evidence_complete": False, "problems": [str(exc)]}
                record["budget_stopped"] = True
        training["training_complete"] = (set(training["models"]) == set(driver.MODELS)
                                         and all(training["models"][m].get("complete") is True for m in driver.MODELS))
        record["training"] = training
        selection_path = run / "evaluation/checkpoint_selection.json"
        selection = {"models": {}, "problems": ["selection_missing"]}
        selection_problems = []
        try:
            selection = _read_json(selection_path)
            fresh = driver.build_checkpoint_selection(run, project)
            if fresh != selection:
                selection_problems.append("checkpoint_selection_drifted_since_raw_eval")
            selection_problems.extend(fresh.get("problems", []))
            if set(selection.get("models", {})) != set(driver.MODELS):
                selection_problems.append("checkpoint_selection_models_missing_or_extra")
            artifacts[_record_path(selection_path, project)] = integ.sha256_file(selection_path)
        except (Exception, SystemExit) as exc:
            selection_problems.append(f"selection_unverifiable:{type(exc).__name__}:{exc}")
        problems = [f"raw-eval:{p}" for p in raw_problems] + selection_problems
        source_rows = raw_report.get("rows")
        if not isinstance(source_rows, list):
            source_rows = []
            problems.append("raw_rows_missing_or_malformed")
        rows = [dict(r) for r in source_rows if isinstance(r, dict)]
        if len(rows) != len(source_rows):
            problems.append("raw_nonmapping_rows")
        recorded_completeness = raw.w1_de_matrix_completeness(
            rows, reg, budget_stopped=raw_report.get("budget_stopped", False))
        by_key = {r.get("key"): r for r in rows}
        verified = {}
        if raw_path is not None and raw_path.is_file() and selection_path.is_file() and not entry_problems:
            for demo in reg["rosters"]["val31"]:
                try:
                    result = _run_unit("measure", {
                        "project_root": str(project), "demo": demo,
                        "registration_path": str(reg_path), "registration_sha256": integ.sha256_file(reg_path),
                        "raw_report_path": str(raw_path), "raw_report_sha256": integ.sha256_file(raw_path),
                        "selection_path": str(selection_path), "selection_sha256": integ.sha256_file(selection_path),
                        "keys": [r["key"] for r in matrix if r["demo"] == demo],
                        "attempt_dir": str(attempt), "detail": False},
                        deadline, attempt, f"final_measure__{demo}")
                    for rec in result["rows"]:
                        key = rec["key"]
                        verified[key] = rec
                        if rec.get("consistent_with_raw") is True and rec.get("status") in ("ok", "numerical_rejected"):
                            by_key[key] = rec["recomputed"]
                        else:
                            problems.extend(f"{key}:{p}" for p in rec.get("problems", ["product_unverified"]))
                            if key in by_key:
                                by_key[key].update(status="evidence_incomplete", product_verified=False)
                except TimeoutError as exc:
                    record["budget_stopped"] = True
                    problems.append(str(exc))
                    break
                except (Exception, SystemExit) as exc:
                    problems.append(f"{demo}:product_replay_failed:{type(exc).__name__}:{exc}")
        # Preserve duplicate/unexpected rows for exact matrix auditing rather
        # than laundering them through the keyed recomputation lookup.
        recomputed_rows = [by_key.get(r.get("key"), r) for r in rows]
        for wanted in matrix:
            if wanted["key"] not in verified:
                problems.append(f"{wanted['key']}:independent_product_verification_missing")
        completeness = raw.w1_de_matrix_completeness(
            recomputed_rows, reg, budget_stopped=record["budget_stopped"]
            or raw_report.get("completeness", {}).get("budget_stopped", False))
        aggregates = raw.w1_de_aggregate_rows(recomputed_rows, reg)
        problems.extend(_compare_evidence(raw_report.get("per_method"), aggregates, "per_method"))
        problems.extend(_compare_evidence(raw_report.get("completeness"), recorded_completeness, "completeness"))
        if raw_report.get("execution_registration", {}).get("sha256") != integ.sha256_file(reg_path):
            problems.append("raw_registration_identity_mismatch")
        if raw_report.get("training_manifest") != reg["training_manifest"]:
            problems.append("raw_manifest_identity_mismatch")
        if raw_report.get("validation_config") != reg["evaluation"]["validation_config"]:
            problems.append("raw_validation_config_mismatch")
        if raw_report.get("repair_pipeline_invocations") != 0:
            problems.append("raw_zero_repair_evidence_missing")
        evaluation_complete = bool(completeness["complete"] and not problems and not entry_problems)
        evaluation = {"raw_attempt": (raw_att or {}).get("attempt"),
                      "per_method": aggregates, "completeness": completeness,
                      "recorded_completeness_recomputed": recorded_completeness,
                      "recompute_problems": sorted(set(problems)),
                      "evaluation_complete": evaluation_complete,
                      "main": {"expected": completeness["expected_main_keys"],
                               "complete": completeness["main_complete"] and not problems},
                      "multi_sampling": {"expected": completeness["expected_appendix_keys"],
                                         "complete": completeness["appendix_complete"] and not problems},
                      "product_verification": verified}
        record["evaluation"] = evaluation
        detail = _verify_detail(detail_report, reg, raw_path, recomputed_rows, project)
        detail["problems"].extend(detail_problems)
        detail["complete"] = detail["complete"] and not detail_problems
        detail["attempt"] = (detail_att or {}).get("attempt")
        record["detail"] = detail
        replay = _replay_selected_checkpoints(run, project, reg, raw_report, selection,
                                               attempt_dir=attempt, deadline=deadline)
        record["checkpoint_replay"] = replay
        closure_complete = bool(training["training_complete"] and evaluation_complete
                                and detail["complete"] and replay["complete"]
                                and not record["budget_stopped"] and not entry_problems)
        numerical_failures = sum(g["counts"]["numerical_rejected"] for g in aggregates.values())
        motion = "failed" if closure_complete and numerical_failures else "complete" if closure_complete else "incomplete"
        unmet = ["sustained two-hand carry and main-plan final task acceptance remain unproven"]
        for model in driver.MODELS:
            name = f"{model}|{reg['evaluation']['main_seed']}"
            count = aggregates[name]["counts"]
            if count["raw_standard_pass"] != count["expected"]:
                unmet.append(f"{name}:raw_standard_pass {count['raw_standard_pass']}/{count['expected']}")
        historical = reg.get("historical_anchors")
        if (not isinstance(historical, dict) or historical.get("u0_verdict") != "incomplete"
                or historical.get("ur_verdict") != "no_quality_gain"):
            record["problems"].append("historical_U0_UR_state_mismatch")
            closure_complete, motion = False, "incomplete"
        for path in (raw_path, detail_path):
            if path is not None and path.is_file():
                artifacts[_record_path(path, project)] = integ.sha256_file(path)
        record.update(complete=closure_complete, motion_baseline=motion,
                      final_interaction_method={"verdict": "not_ready" if evaluation_complete else "evidence_insufficient",
                                                "unmet": unmet,
                                                "basis": "diagnostic raw robot/task checks and full both-reference evidence; no final carrying claim"},
                      next_action=_route_judgment(training, evaluation, main_seed=reg["evaluation"]["main_seed"]),
                      historical_anchors=historical, checkpoint_selection=selection,
                      registration=_file_record(reg_path, project), artifacts=artifacts)
        finished = _finish(run, "finalize", attempt, output, record, t0, budget, project, artifacts)
    except BaseException as exc:
        _finish(run, "finalize", attempt, output, record, t0, budget, project, artifacts, exc)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 2
    print(json.dumps({"complete": finished, "motion_baseline": record["motion_baseline"],
                      "training_complete": record["training"]["training_complete"],
                      "evaluation_complete": record["evaluation"]["evaluation_complete"],
                      "detail_complete": record["detail"]["complete"],
                      "checkpoint_replay_ok": record["checkpoint_replay"]["complete"],
                      "final_interaction_method": record["final_interaction_method"]["verdict"]}, indent=2))
    return 0 if finished else 2
