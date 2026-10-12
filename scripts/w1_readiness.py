"""W1 readiness decision: machine-readable A3/B/C completion record.

W1 plan §10. Reads the run root's ACTUAL artifacts and recomputes every
gate from facts (never from prose or exit codes):

* ``registration_complete`` — freeze/preregistration.json exists and its
  anchors/lists/inputs/assets legs re-verify against disk (the C2/UR entry
  evidence is re-checked through the same legs the W0 entry used);
* ``a3_complete`` — the newest audit attempt's exit criteria recomputed
  from its rows (55 final rows, train quality registration complete, six
  subjects with usable windows);
* ``b_complete`` — the cache identity closes against the preregistration
  and training manifest on disk, AND both W1 datasets actually construct
  (the startup verification legs run for real) with the train window count
  >= micro_batch and one real batch fetch;
* ``c_complete`` — both models' newest S0 attempts are ``passed`` on every
  gate (>=100 real optimizer steps, all-finite, contact branch silent,
  >=40 fixed probes, fit trend, reload reproducibility);
* ``ready_for_w1_d`` — all of the above (each reason recorded; a missing
  value and an evidenced zero stay distinct).

Exit 0 means the readiness RECORD was written from complete inputs; it is
never a claim that training may start — when ``ready_for_w1_d`` is false
the reasons are in the record and the exit code is 2.

Run: uv run --locked python scripts/w1_readiness.py --run-root <run_root>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ireset.data import a2_integrity as integ
from ireset.data import w1_training as w1
from ireset.paths import OMOMO_NEW_DIR, PROJECT_ROOT, TEACHER_TRAJ_DIR, path_for_record


def latest_attempt(root: Path) -> dict | None:
    index = root / integ.ATTEMPTS_INDEX
    if not index.is_file():
        return None
    attempts = json.loads(index.read_text())["attempts"]
    return attempts[-1] if attempts else None


def check_registration(run_root: Path) -> dict:
    from w1_preregister import verify_c2_decision, verify_ur_report

    reasons: list[str] = []
    prereg_path = run_root / "freeze" / "preregistration.json"
    if not prereg_path.is_file():
        return {"passed": False, "reasons": ["preregistration_missing"]}
    prereg = json.loads(prereg_path.read_text())
    check = w1.verify_preregistration(prereg, PROJECT_ROOT,
                                      TEACHER_TRAJ_DIR, OMOMO_NEW_DIR)
    reasons += [f"preregistration:{p}" for p in check["problems"]]
    paths = w1.anchor_paths(PROJECT_ROOT)
    c2 = verify_c2_decision(paths["c2_final_decision"])
    ur = verify_ur_report(paths["ur_full_comparison"],
                          paths["ur_preregistration"])
    reasons += [f"c2:{p}" for p in c2["problems"]]
    reasons += [f"ur:{p}" for p in ur["problems"]]
    return {"passed": not reasons, "reasons": reasons,
            "preregistration": path_for_record(prereg_path)}


def check_a3(run_root: Path) -> dict:
    from audit_omniretarget_training import (
        BOOL_EXIT_GATES, exit_criteria, exit_ok, summarize,
    )

    att = latest_attempt(run_root / "audit")
    if att is None:
        return {"passed": False, "reasons": ["a3_no_attempts"]}
    manifest = json.loads((PROJECT_ROOT / att["manifest"]).read_text())
    scope = list(manifest["selection"]["train24"]) + list(manifest["selection"]["val31"])
    summary = summarize(manifest["rows"], scope)
    gates = exit_criteria(summary, manifest["rows"])
    # only the boolean gates count as unmet reasons (``missing_rows`` is the
    # evidence list, not itself a verdict)
    reasons = [k for k in BOOL_EXIT_GATES if gates.get(k) is not True]
    return {"passed": exit_ok(gates) and not reasons, "reasons": reasons,
            "exit_criteria": gates, "counts": {
                k: v for k, v in summary.items()
                if not isinstance(v, (dict, list))},
            "manifest": att["manifest"]}


def check_cache_binding(run_root: Path) -> dict | None:
    att = latest_attempt(run_root / "cache")
    if att is None:
        return None
    ident_path = PROJECT_ROOT / att["identity"]
    return {"attempt": att["attempt"], "identity": ident_path,
            "dir": ident_path.parent,
            "record": json.loads(ident_path.read_text())}


def check_b(run_root: Path, cache: dict | None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ireset.data.windows import TeacherWindowDataset, make_tokenizer
    from ireset.model.tokens import RootStats

    reasons: list[str] = []
    if cache is None:
        return {"passed": False, "reasons": ["cache_no_attempts"]}
    rec = cache["record"]
    cache_dir = cache["dir"]
    manifest_path = PROJECT_ROOT / rec["training_manifest"]["path"]
    if integ.sha256_file(manifest_path) != rec["training_manifest"]["sha256"]:
        reasons.append("cache_manifest_binding_drifted")
    prereg_path = run_root / "freeze" / "preregistration.json"
    if integ.sha256_file(prereg_path) != rec["preregistration"]["sha256"]:
        reasons.append("cache_prereg_binding_drifted")
    # review fix 6: CONTENT re-verification — root stats and every keyed
    # schedule re-hashed against the identity record, keys re-derived from
    # the current code/assets, stats identity reconciled with the manifest
    manifest = json.loads(manifest_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    content = w1.verify_cache_contents(cache_dir, manifest, prereg, PROJECT_ROOT)
    reasons += [f"cache_content:{p}" for p in content["problems"]]
    root_stats = cache_dir / "root_stats.npz"
    if not root_stats.is_file():
        reasons.append("root_stats_missing")
    if reasons:
        return {"passed": False, "reasons": reasons}

    # REAL construction of both datasets: every startup verification leg
    # runs (protocol/prereg/selection/inputs/derived/keyed-cache/assets)
    stats = RootStats.load(root_stats)
    tokenizer = make_tokenizer(stats)
    out = {}
    try:
        for split in ("train", "val"):
            ds = TeacherWindowDataset(
                split, tokenizer, window=w1.WINDOW_F, stride=w1.WINDOW_STRIDE,
                training_manifest=manifest_path,
                schedule_cache_dir=cache_dir / "schedules")
            loader = DataLoader(ds, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            out[split] = {"n_windows": len(ds), "n_clips": len(ds.clips),
                          "excluded": {k: len(v) for k, v in ds.excluded.items()},
                          "frame_count": ds.frame_count(),
                          "real_batch": tuple(batch["x0"].shape)}
        if out["train"]["n_windows"] < 4 or out["train"]["real_batch"][0] != 4:
            reasons.append("train_micro_batch_not_formable")
    except Exception as exc:  # noqa: BLE001 -- recorded as the reason
        reasons.append(f"dataset_construction_failed:{type(exc).__name__}: {exc}")
    return {"passed": not reasons, "reasons": reasons,
            "datasets": out, "cache_attempt": cache["attempt"]}


def _recompute_s0_verdict(summary: dict, history: list[dict],
                          history_present: bool) -> tuple[dict, list[str]]:
    """Recompute EVERY S0 gate from the raw facts (review fix 2, deepened
    rounds 2-3) — the recorded ``verdict`` string is never trusted by itself,
    and neither are summary COUNTS: the loss history must exist, be
    contiguous 1..N with EVERY configured loss part present and finite, the
    probes must equal the complete registered schedule, and a budget stop is
    a hard fail."""
    import math as _math

    import numpy as np

    probes = summary.get("probes") or []
    reasons: list[str] = []
    steps = summary.get("optimizer_steps")
    if not isinstance(steps, int) or isinstance(steps, bool):
        reasons.append("optimizer_steps_not_an_int")
        steps = -1
    gates = {}
    gates["min_100_optimizer_steps"] = steps >= w1.S0_MIN_STEPS
    gates["max_1000_optimizer_steps"] = steps <= w1.S0_MAX_STEPS
    gates["micro_batches_real"] = (summary.get("micro_batches", 0)
                                   >= steps * summary.get("grad_accum", 0) > 0)
    # the loss history is PRIMARY evidence: it must exist and cover every
    # completed optimizer step exactly once (round-2 review fix 1 — an
    # empty/deleted history was accepted by all([]) vacuously)
    gates["loss_history_present"] = history_present
    gates["loss_history_contiguous"] = (
        history_present and len(history) == steps
        and [h.get("step") for h in history] == list(range(1, steps + 1)))
    # round-3 review fix 3: every configured loss part must EXIST in every
    # row — checking only the fields that are present let a history shrunk
    # to {step, gnorm} count as complete training evidence
    required = (*w1.S0_REQUIRED_LOSS_KEYS, "gnorm")
    gates["loss_fields_complete"] = bool(history) and all(
        all(k in h for k in required) for h in history)
    # finiteness over EVERY measured value in each row, not just gnorm
    def _row_finite(h: dict) -> bool:
        for k, v in h.items():
            if k in ("step", "ms"):
                continue
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and _math.isfinite(float(v))):
                return False
        return "gnorm" in h
    gates["all_finite"] = (summary.get("nonfinite_stop") is None
                           and bool(history)
                           and all(_row_finite(h) for h in history))
    # contact branch silence recomputed from the per-step loss parts
    contact_keys = ("contact_m2", "contact")
    gates["contact_branch_silent"] = not any(
        anyk in h for h in history for anyk in contact_keys)
    errs = [p.get("motion_reconstruction") for p in probes]
    gates["probe_count_ge_40"] = len(errs) >= w1.S0_PROBE_MIN
    gates["probe_errors_finite"] = bool(errs) and all(
        isinstance(e, (int, float)) and not isinstance(e, bool)
        and _math.isfinite(e) for e in errs)
    # probe timing (round-3 review fix 5): the recorded steps must EQUAL the
    # complete registered schedule up to the completed steps — increasing
    # multiples of the cadence did not prove that no sample point was
    # deleted (e.g. dropping the step-10 probe kept the gate green)
    psteps = [p.get("step") for p in probes]
    expected_psteps = ([0] + list(range(w1.S0_PROBE_EVERY, steps + 1,
                                        w1.S0_PROBE_EVERY)))
    gates["probe_schedule_valid"] = psteps == expected_psteps
    if gates["probe_errors_finite"] and len(errs) >= 2 * w1.S0_TREND_WINDOW:
        first = float(np.mean(errs[:w1.S0_TREND_WINDOW]))
        last = float(np.mean(errs[-w1.S0_TREND_WINDOW:]))
        gates["trend_fit_demonstrated"] = last < first
        gates["first20_mean"], gates["last20_mean"] = first, last
    else:
        gates["trend_fit_demonstrated"] = False
        gates["first20_mean"] = gates["last20_mean"] = None
    # round-2 review fix 6: a budget-stopped run is never a pass
    gates["completed_within_budget"] = summary.get("budget_stopped") is not True
    return gates, reasons


#: the fixed-probe error fields that must be finite and RECONCILED between
#: the summary's trend evidence and the replay artifact (round-3 fix 2)
PROBE_ERROR_FIELDS = ("pos", "rot6d", "joint", "motion_reconstruction")


def _rebuild_small_probe_batch(cache_dir: Path, manifest_path: Path,
                               pilot6: list[str]) -> tuple[dict | None, list[str]]:
    """Rebuild the small-set probe inputs from the CURRENT registered list,
    training manifest and keyed cache (round-4 review fix 1).

    The check_c gate used to bind only the summary's DECLARED demo list and
    batch size — the batch actually inside ``probe_replay.pt`` was never
    bound to them, so a batch shrunk to one (or foreign) demo with honestly
    recomputed errors and synced last probes still passed. The dataset
    slicing is deterministic (pure cached-schedule reads), so exact equality
    with the rebuilt windows is the correct criterion."""
    import torch

    from ireset.data.windows import TeacherWindowDataset, make_tokenizer
    from ireset.model.tokens import RootStats

    problems: list[str] = []
    manifest = json.loads(manifest_path.read_text())
    row_by_demo = {r["demo"]: r for r in manifest["rows"]}
    ds = TeacherWindowDataset(
        "train", make_tokenizer(RootStats.load(cache_dir / "root_stats.npz")),
        window=w1.WINDOW_F, stride=w1.WINDOW_STRIDE,
        training_manifest=manifest_path,
        schedule_cache_dir=cache_dir / "schedules")
    items = []
    for demo in pilot6:
        row = row_by_demo.get(demo)
        if row is None or not row.get("admission", {}).get("motion"):
            problems.append(f"probe_batch_demo_not_admitted:{demo}")
            continue
        lc_idx = next((i for i, lc in enumerate(ds.clips) if lc.demo == demo),
                      None)
        if lc_idx is None:
            problems.append(f"probe_batch_demo_not_in_dataset:{demo}")
            continue
        items.append(ds[ds.index.index((lc_idx, 0))])
    if problems or len(items) != len(pilot6):
        return None, problems
    return ({k: torch.stack([it[k] for it in items])
             for k in items[0] if torch.is_tensor(items[0][k])}, [])


def _verify_checkpoint_deep(ckpt_path: Path, att_dir: Path, cache_dir: Path,
                            model: str, summary: dict,
                            expected_batch: dict | None) -> tuple[list[str], dict | None]:
    """Actually REBUILD the modules and REPLAY the fixed probe (round-2
    review fix 2: torch.load only proved the container readable — cleared
    state dicts, NaN weights and doctored embedded statistics all passed).

    Catches: cleared/foreign weights (strict load fails or replay diverges),
    NaN weights (finiteness over every parameter), embedded root statistics
    differing from the cache, config/model-type mismatch, a missing or
    foreign probe-replay artifact.

    Round-3 review fixes 1+2: the replay INPUTS and OUTPUTS must be finite
    BEFORE any tolerance comparison (``abs(NaN - rec) > tol`` is silently
    False, so a NaN input laundered a pass), and the summary's LAST probe —
    the value the trend gate was computed from — must equal the artifact's
    recorded last probe (step and every error field), so a doctored summary
    curve cannot coexist with a genuine checkpoint.

    Round-4 review fixes 1+2: the replayed batch must BE the registered
    six windows rebuilt from the current manifest/cache (bitwise), the
    diffusion t/noise must be the dedicated-generator draw for that batch
    (regenerated from the checkpoint config and compared bitwise), and an
    unreadable/non-mapping/structurally-invalid replay artifact is RECORDED
    as an unready reason instead of crashing the gate — invalid inputs are
    never replayed (a NaN probe_t reached run_fixed_probe and raised
    IndexError out of the readiness script)."""
    import math as _math

    import numpy as _np
    import torch

    import w1_s0

    problems: list[str] = []
    replay_path = att_dir / "probe_replay.pt"
    if not replay_path.is_file():
        return ["probe_replay_missing"], None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # round-4 fix 2: an unreadable or non-mapping replay artifact is a
    # REASON, never an exception escaping the gate
    try:
        rp = torch.load(replay_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return [f"probe_replay_unreadable:{type(exc).__name__}"], None
    if not isinstance(rp, dict):
        return ["probe_replay_not_a_mapping"], None
    if rp.get("model_type") != model or rp.get("step") != ck.get("step"):
        problems.append("probe_replay_identity_mismatch")
    try:
        from ireset.train.trainer import TrainConfig

        cfg = TrainConfig(**ck["config"])
    except Exception as exc:  # noqa: BLE001
        return problems + [f"checkpoint_config_unusable:{type(exc).__name__}"], None
    if cfg.model_type != model:
        problems.append("checkpoint_model_type_mismatch")
    # embedded statistics must EQUAL the cache root statistics (bitwise)
    with _np.load(cache_dir / "root_stats.npz") as rs:
        stats_match = (
            _np.array_equal(_np.asarray(ck["root_stats"]["mean"]), rs["mean"])
            and _np.array_equal(_np.asarray(ck["root_stats"]["std"]), rs["std"]))
    if not stats_match:
        problems.append("checkpoint_stats_differ_from_cache")
    # round-4 fix 1: bind the ACTUAL replay batch to the registered set —
    # the summary's declared list + batch size left the replay inputs free
    # (one-demo/foreign batches with honestly recomputed errors passed)
    batch = rp.get("batch")
    if expected_batch is None:
        problems.append("probe_replay_batch_rebuild_failed")
    elif not isinstance(batch, dict) or set(batch) != set(expected_batch):
        problems.append("probe_replay_batch_keys_mismatch")
    else:
        for k, v in expected_batch.items():
            got = batch.get(k)
            if not (torch.is_tensor(got) and got.dtype == v.dtype
                    and got.shape == v.shape and torch.equal(got, v)):
                problems.append(f"probe_replay_batch_differs:{k}")
                break
    # round-4 fix 1 (diffusion): the recorded t/noise must be the
    # dedicated-generator draw for the registered batch — same expression
    # the S0 loop used (val_seed or seed; randint THEN randn on one stream).
    # A non-tensor value fails the binding too (the structural gate below
    # records it separately — this comparison must never raise)
    if model == "diffusion" and expected_batch is not None:
        if rp.get("val_seed") != cfg.val_seed:
            problems.append("probe_replay_val_seed_mismatch")
        seed = cfg.val_seed or cfg.seed
        if seed is None:
            problems.append("probe_replay_generator_seed_unavailable")
        else:
            g = torch.Generator().manual_seed(int(seed))
            shape = expected_batch["x0"].shape
            exp_t = torch.randint(0, int(cfg.num_train_timesteps),
                                  (shape[0],), generator=g)
            exp_noise = torch.randn(shape, generator=g)
            if not (torch.is_tensor(rp.get("probe_t"))
                    and torch.equal(rp["probe_t"], exp_t)):
                problems.append("probe_t_not_the_registered_generator_draw")
            if not (torch.is_tensor(rp.get("probe_noise"))
                    and torch.equal(rp["probe_noise"], exp_noise)):
                problems.append("probe_noise_not_the_registered_generator_draw")
    try:
        m, cond = w1_s0.build_s0_modules(model, cfg, "cpu")
        m.load_state_dict(ck["model"], strict=True)
        cond.load_state_dict(ck["conditioner"], strict=True)
    except Exception as exc:  # noqa: BLE001
        return problems + [f"checkpoint_weights_unloadable:{type(exc).__name__}"], None
    if not all(bool(torch.isfinite(p).all()) for p in
               [*m.parameters(), *cond.parameters()]):
        return problems + ["checkpoint_weights_not_finite"], None
    # round-4 fix 2: STRUCTURAL validation of the replay inputs — a missing
    # t/noise or a non-tensor batch value used to reach run_fixed_probe and
    # raise, crashing the readiness script before any report was written
    inputs_valid = True
    if not (isinstance(batch, dict) and batch and "x0" in batch
            and all(torch.is_tensor(v) for v in batch.values())):
        problems.append("probe_replay_input_invalid:batch")
        inputs_valid = False
    if model == "diffusion":
        for name in ("probe_t", "probe_noise"):
            if not torch.is_tensor(rp.get(name)):
                problems.append(f"probe_replay_input_invalid:{name}")
                inputs_valid = False
        if (inputs_valid and isinstance(batch, dict)
                and torch.is_tensor(batch.get("x0"))):
            if tuple(rp["probe_t"].shape) != (batch["x0"].shape[0],):
                problems.append("probe_replay_input_invalid:probe_t_shape")
                inputs_valid = False
            if tuple(rp["probe_noise"].shape) != tuple(batch["x0"].shape):
                problems.append("probe_replay_input_invalid:probe_noise_shape")
                inputs_valid = False
    # round-3 fix 1: finite replay INPUTS — a NaN x0/t/noise replays to NaN
    # and the tolerance comparison below can never reject it
    finite_inputs = True
    for name, tensor in [("probe_t", rp.get("probe_t")),
                         ("probe_noise", rp.get("probe_noise")),
                         *((f"batch.{k}", v)
                           for k, v in (batch or {}).items())]:
        if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
            problems.append(f"probe_replay_input_not_finite:{name}")
            finite_inputs = False
    recorded = rp.get("last_probe") or {}
    rec = recorded.get("motion_reconstruction")
    detail = {"replayed_probe": None, "recorded_last_probe": recorded}
    recorded_finite = isinstance(recorded, dict) and all(
        isinstance(recorded.get(k), (int, float))
        and _math.isfinite(float(recorded.get(k)))
        for k in PROBE_ERROR_FIELDS)
    if not recorded_finite:
        problems.append("recorded_probe_not_finite")
    # round-3 fix 2: the summary's last probe IS the trend evidence — it
    # must be the SAME measurement the artifact recorded (step + every
    # error field; both serializations round-trip Python floats exactly)
    s_probes = summary.get("probes") or []
    s_last = s_probes[-1] if s_probes else None
    if not isinstance(s_last, dict):
        problems.append("summary_last_probe_missing")
    elif recorded_finite:
        if s_last.get("step") != recorded.get("step"):
            problems.append("summary_last_probe_mismatch:step")
        for k in PROBE_ERROR_FIELDS:
            if s_last.get(k) != recorded.get(k):
                problems.append(f"summary_last_probe_mismatch:{k}")
    # round-4 fix 2: invalid or non-finite inputs are never replayed — the
    # round-3 flow proceeded to run_fixed_probe and a NaN probe_t escaped
    # as an IndexError; the unready reasons are recorded instead
    if not (inputs_valid and finite_inputs):
        return problems, detail
    try:
        err = w1_s0.run_fixed_probe(m, cond, rp["batch"], model,
                                    rp.get("probe_t"), rp.get("probe_noise"))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"probe_replay_failed:{type(exc).__name__}")
        return problems, detail
    detail["replayed_probe"] = err
    # round-3 fix 1: every replayed error must be finite BEFORE comparing
    outputs_finite = all(
        isinstance(v, (int, float)) and _math.isfinite(float(v))
        for v in err.values())
    if not outputs_finite:
        problems.append("probe_replay_output_not_finite")
    if recorded_finite and outputs_finite:
        if (abs(err["motion_reconstruction"] - float(rec))
                > 1e-3 + 1e-3 * abs(float(rec))):
            problems.append(
                f"probe_replay_mismatch:{err['motion_reconstruction']:.3e}"
                f"!={float(rec):.3e}")
    return problems, detail


def check_c(run_root: Path, cache: dict | None, prereg: dict) -> dict:
    """Recompute both models' S0 gates from their recorded facts AND bind
    the checkpoints to the CURRENT manifest/cache identity (review fix 2:
    a doctored ``verdict=passed`` with zero steps, deleted probes, a
    missing checkpoint or a foreign-identity checkpoint must all fail)."""
    import torch

    reasons: list[str] = []
    per_model = {}
    manifest_sha = cache["record"]["training_manifest"]["sha256"] if cache else None
    cache_ident_sha = root_stats_sha = None
    expected_batch: dict | None = None
    rebuild_problems: list[str] = []
    if cache is not None:
        cache_ident_sha = integ.sha256_file(cache["identity"])
        root_stats_sha = integ.sha256_file(cache["dir"] / "root_stats.npz")
        # round-4 review fix 1: rebuild the registered six first windows ONCE
        # from the current manifest/cache — every model's replay batch is
        # bound to them bitwise (the declared list alone proved nothing)
        pilot6 = list(prereg.get("selection", {}).get("pilot6_from_a1", []))
        expected_batch, rebuild_problems = _rebuild_small_probe_batch(
            cache["dir"], PROJECT_ROOT / cache["record"]["training_manifest"]["path"],
            pilot6)
    for model in ("deterministic", "diffusion"):
        att = latest_attempt(run_root / "s0" / model)
        if att is None:
            reasons.append(f"s0_{model}_no_attempts")
            per_model[model] = None
            continue
        att_dir = (PROJECT_ROOT / att["summary"]).parent
        summary = json.loads((PROJECT_ROOT / att["summary"]).read_text())
        history_path = att_dir / "loss_history.json"
        history = (json.loads(history_path.read_text())
                   if history_path.is_file() else [])
        gates, gate_reasons = _recompute_s0_verdict(
            summary, history, history_path.is_file())
        reasons += [f"s0_{model}:{r}" for r in gate_reasons]
        # checkpoint evidence: exists, reloadable, bound to CURRENT identity
        ckpt_path = att_dir / "last.pt"
        ident_problems = []
        deep_detail = None
        if not ckpt_path.is_file():
            ident_problems.append("checkpoint_missing")
        else:
            try:
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                ident_problems.append(f"checkpoint_unreadable:{type(exc).__name__}")
                ck = None
            if ck is not None:
                ci = ck.get("identity") or {}
                if ci.get("training_manifest", {}).get("sha256") != manifest_sha:
                    ident_problems.append("checkpoint_manifest_identity_mismatch")
                if ci.get("cache", {}).get("identity_sha256") != cache_ident_sha:
                    ident_problems.append("checkpoint_cache_identity_mismatch")
                if ci.get("cache", {}).get("root_stats_sha256") != root_stats_sha:
                    ident_problems.append("checkpoint_root_stats_mismatch")
                if ck.get("step") != summary.get("optimizer_steps"):
                    ident_problems.append("checkpoint_step_mismatch")
                if cache is not None:
                    deep_problems, deep_detail = _verify_checkpoint_deep(
                        ckpt_path, att_dir, cache["dir"], model, summary,
                        expected_batch)
                    ident_problems += deep_problems
                    # a failed rebuild (unadmitted/missing pilot demo) is
                    # itself unready evidence, reported per model
                    ident_problems += [f"probe_batch_rebuild:{p}"
                                       for p in rebuild_problems]
        reasons += [f"s0_{model}:{p}" for p in ident_problems]
        # the small set must COVER the registered pilot6 (round-2 review fix
        # 1: a shrunken one-demo small set still "came from" pilot6 under
        # the subset check) and the recorded batch size must match it
        pilot6 = list(prereg.get("selection", {}).get("pilot6_from_a1", []))
        small = summary.get("identity", {}).get("small_set_demos") or []
        if sorted(small) != sorted(pilot6):
            reasons.append(f"s0_{model}:small_set_not_the_registered_pilot6")
        if summary.get("small_set_batch_size") != len(pilot6):
            reasons.append(f"s0_{model}:small_set_batch_size_mismatch")
        reload_ok = summary.get("gates", {}).get("reload_reproducible")
        if reload_ok is not True:
            reasons.append(f"s0_{model}:reload_reproducible_not_true")
        bool_gates = {k: v for k, v in gates.items() if isinstance(v, bool)}
        recomputed_pass = (all(bool_gates.values()) and not ident_problems
                           and reload_ok is True)
        if summary.get("verdict") != "passed":
            reasons.append(f"s0_{model}_{summary.get('verdict')}")
        elif not recomputed_pass:
            reasons.append(f"s0_{model}:recorded_verdict_not_reproducible")
        per_model[model] = {
            "verdict": summary.get("verdict"),
            "recomputed_gates": {k: bool(v) if isinstance(v, bool) else v
                                 for k, v in gates.items()},
            "identity_problems": ident_problems,
            "optimizer_steps": summary.get("optimizer_steps"),
            "small_set_demos": small,
            "checkpoint_replay": deep_detail,
        }
    return {"passed": not reasons, "reasons": reasons, "per_model": per_model}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    run_root = integ.guard_output_dir(args.run_root)

    registration = check_registration(run_root)
    a3 = check_a3(run_root)
    cache = check_cache_binding(run_root)
    b = check_b(run_root, cache)
    prereg_path = run_root / "freeze" / "preregistration.json"
    prereg = (json.loads(prereg_path.read_text())
              if prereg_path.is_file() else {})
    c = check_c(run_root, cache, prereg)

    ready = registration["passed"] and a3["passed"] and b["passed"] and c["passed"]
    record = {
        "stage": "W1 A3/B/C readiness (W1 plan §10)",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "registration_complete": registration["passed"],
        "registration_reasons": registration["reasons"],
        "a3_complete": a3["passed"],
        "a3_reasons": a3["reasons"],
        "a3_exit_criteria": a3.get("exit_criteria"),
        "b_complete": b["passed"],
        "b_reasons": b["reasons"],
        "b_datasets": b.get("datasets"),
        "c_complete": c["passed"],
        "c_reasons": c["reasons"],
        "c_per_model": c["per_model"],
        "ready_for_w1_d": ready,
        "unmet": ([] if ready else
                  (registration["reasons"] + a3["reasons"]
                   + b["reasons"] + c["reasons"])),
        "note": "exit 0 = readiness record written; ready_for_w1_d is the "
                "training-entry gate, NOT a feasibility claim — U0 "
                "incomplete / UR no_quality_gain / final interaction goals "
                "all stand unchanged",
        "wall_s": round(time.time() - t0, 1),
    }
    decisions = run_root / "decisions"
    attempt_dir = integ.allocate_attempt(decisions)
    out = attempt_dir / "w1_readiness.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    integ.append_attempts_index(decisions, {
        "attempt": attempt_dir.name,
        "ready_for_w1_d": ready,
        "report": path_for_record(out),
    })
    _append_costs(run_root, "w1_readiness", time.time() - t0,
                  {"ready_for_w1_d": ready})
    print(json.dumps({k: record[k] for k in
                      ("registration_complete", "a3_complete", "b_complete",
                       "c_complete", "ready_for_w1_d", "unmet")}, indent=2))
    print(f"readiness -> {path_for_record(out)}")
    return 0 if ready else 2


def _append_costs(run_root: Path, stage: str, wall: float, extra: dict) -> None:
    costs_path = run_root / "costs.json"
    entries = []
    if costs_path.is_file():
        entries = json.loads(costs_path.read_text()).get("invocations", [])
    entries.append({"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "stage": stage, "wall_s": round(wall, 1), **extra})
    costs_path.write_text(json.dumps({"invocations": entries}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
