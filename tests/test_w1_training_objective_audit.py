"""Synthetic audit tests for the W1 training objective (plan A0-A5).

All fixtures are analytic; no dataset, checkpoint, MuJoCo scene or G1 URDF is
read.  The FK stand-in is explicitly the audit's synthetic backend.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ireset.eval.training_objective_audit as audit
import ireset.train.losses as losses_mod
from ireset.eval.training_objective_audit import (
    INITIAL_PARAMS,
    REQUIRED_ARTIFACTS,
    REQUIRED_AUDIT_ROWS,
    W1_SYNTHETIC_WEIGHTS,
    build_synthetic_assembler,
    build_synthetic_world,
    gradient_fact,
    independent_macro,
    make_synthetic_batch,
    parity_problems,
    run_a1,
    run_a2,
    run_synthetic_audit,
    verify_audit_artifacts,
)


@pytest.fixture(scope="module")
def audit_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("w1_objective_audit")
    summary = run_synthetic_audit(out, ROOT)
    assert summary["conclusion"] == "synthetic_contract_passed", summary
    return out


def _load(name: str, d: Path):
    return json.loads((d / name).read_text())


def _rows(d: Path) -> dict[str, dict]:
    return {r["id"]: r for r in _load("objective_rows.json", d)["rows"]}


def test_full_audit_passes_and_writes_artifacts(audit_dir: Path):
    summary = _load("audit_summary.json", audit_dir)
    assert summary["producer_conclusion"] == "synthetic_contract_passed"
    assert summary["conclusion"] == "synthetic_contract_passed"
    assert summary["failures"] == [] and summary["gaps"] == []
    assert summary["scope"]["real_dataset_loaded"] is False
    assert summary["scope"]["real_checkpoint_loaded"] is False
    assert summary["scope"]["real_model_calls"] == 0
    assert summary["boundaries"]["final_interaction_method"] == "not_ready"
    assert summary["boundaries"]["auto_execute"] is False
    for name in REQUIRED_ARTIFACTS:
        assert (audit_dir / name).is_file(), name
    verification = _load("independent_verification.json", audit_dir)
    assert verification["conclusion"] == "synthetic_contract_passed"
    assert verification["problems"] == []
    # every required row is present under its group
    rows = _rows(audit_dir)
    for group, ids in REQUIRED_AUDIT_ROWS.items():
        for rid in ids:
            assert rows[rid]["group"] == group
            assert rows[rid]["status"] == "synthetic_verified", rid


def test_no_real_asset_access(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the synthetic audit touched a real asset")

    monkeypatch.setattr(losses_mod, "G1TorchFK", boom)
    monkeypatch.setattr(losses_mod, "largebox_static_frame", boom)
    world = build_synthetic_world()
    rows_a1, _ = run_a1(world)
    rows_a2, _ = run_a2(world)
    assert all(r["status"] == "synthetic_verified" for r in rows_a1 + rows_a2)
    # the synthetic constructor is the only assembler path used
    asm = build_synthetic_assembler(world.tokenizer, W1_SYNTHETIC_WEIGHTS, torch.float64)
    assert asm.fk is not None and isinstance(asm.fk, audit.SyntheticFK)


def test_encode_column_isolation_and_decode():
    world = build_synthetic_world()
    rows = _rows_from(run_a1(world)[0])
    assert np.allclose(rows["encode.numpy_columns_and_transform"]["actual"],
                       rows["encode.torch_vs_numpy"]["actual"], atol=1e-12, rtol=0)
    iso = rows["encode.column_isolation"]["actual"]
    assert iso["root_pos_changed_cols"] == [0, 1, 2]
    assert iso["joint_changed_cols"] == [12]
    decode = rows["decode.clip_vs_world"]["actual"]
    assert decode["pos_max_err"] == 0.0
    assert decode["rot_max_err"] < 3e-8
    assert decode["quat_abs_dot_min"] > 0.99999999


def _rows_from(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def test_loss_reductions_weights_and_vel_grouping():
    rows = _rows_from(run_a1(build_synthetic_world())[0])
    g = rows["loss.group_means_not_overall_mse"]
    assert g["status"] == "synthetic_verified"
    assert g["actual"]["gap"] > 1e-6
    assert abs(g["evidence"]["implementation_total_minus_group_sum"]) < 1e-12
    vel = rows["loss.vel_reduction_38_channels"]
    assert vel["status"] == "synthetic_verified"
    assert vel["evidence"]["grouped_differs"] is True
    batch = rows["loss.batch_permutation_invariance"]
    assert batch["status"] == "synthetic_verified"
    assert batch["actual"]["unpermuted"][0] > 1e-6
    assert batch["actual"]["permuted"][0] > 1e-6
    assert all(v > 0 for v in batch["actual"]["independent_mean"][:3])
    repeat = rows["loss.repeated_batch_invariance"]
    assert repeat["status"] == "synthetic_verified"
    assert repeat["actual"]["repeated"][0] > 1e-6
    total = rows["loss.total_weighted_sum"]
    assert total["status"] == "synthetic_verified"
    w = W1_SYNTHETIC_WEIGHTS
    terms = total["evidence"]["terms"]
    manual = (w.w_pos * terms["pos"] + w.w_rot6d * terms["rot6d"]
              + w.w_joint * terms["joint"] + w.w_geodesic * terms["geodesic_rad2"]
              + w.w_vel * terms["vel"] + w.w_fk * terms["fk_m2"])
    assert abs(manual - total["actual"]) < 1e-12
    # w_contact=0 is the W1 config, not the dataclass default 0.5
    assert W1_SYNTHETIC_WEIGHTS.w_contact == 0.0


def test_negative_gates_are_reported_not_passed_as_errors():
    rows = _rows_from(run_a1(build_synthetic_world())[0])
    near_pi = rows["domain.near_pi_rotation"]
    assert near_pi["status"] == "synthetic_verified"
    assert near_pi["limitation"] == "near_pi_precision_negative_gate_retained"
    assert near_pi["actual"]["geodesic_rad2"] > 0
    assert near_pi["actual"]["relative_error_vs_independent"] > 0.5
    parallel = rows["domain.parallel_6d"]
    assert parallel["status"] == "synthetic_verified"
    assert parallel["actual"]["formal_trainer_rejects"] is True
    assert parallel["actual"]["generator_validity_rejects"] is True
    assert parallel["actual"]["tokenizer_decode_raise"] is not None
    deg = rows["decode.degenerate_rotation_flagged"]
    assert deg["actual"]["tokenizer_ok"] is False
    assert deg["actual"]["formal_trainer_rejects"] is True


def test_contact_branch_silent_at_weight_zero():
    rows = _rows_from(run_a1(build_synthetic_world())[0])
    silent = rows["contact.w_contact_zero_silent"]
    assert silent["status"] == "synthetic_verified"
    assert silent["actual"]["hand_fk_calls"] == 0
    assert silent["actual"]["parts"] == ["fk_m2", "geodesic_rad2", "joint",
                                         "pos", "rot6d", "total", "vel"]
    positive = rows["contact.w_contact_positive_branch"]
    assert positive["status"] == "synthetic_verified"
    assert positive["actual"]["hand_fk_called"] is True
    assert positive["actual"]["contact_m2_contact_flags"] is not None


def test_gradient_sum_fd_and_target_branch():
    rows = _rows_from(run_a2(build_synthetic_world())[0])
    total = rows["grad.total_sum"]
    assert total["status"] == "synthetic_verified"
    assert total["evidence"]["max_abs_error"] < 1e-15
    fd = rows["grad.finite_difference_float64"]
    assert fd["status"] == "synthetic_verified"
    for term, rec in fd["actual"].items():
        assert rec["ok"] is True, term
        assert rec["cosine"] is not None and abs(rec["cosine"]) <= 1.0 + 1e-12
    target = rows["grad.fk_target_no_grad"]
    assert target["actual"]["call_requires_grad"] == [True, False]
    zero = rows["grad.zero_term_null_reason"]
    assert zero["actual"]["cosine"] is None
    assert zero["actual"]["reason"] == "prediction_gradient_zero"


def test_detached_prediction_is_detected():
    world = build_synthetic_world()
    asm = build_synthetic_assembler(world.tokenizer, W1_SYNTHETIC_WEIGHTS, torch.float64)
    batch = make_synthetic_batch(world, dtype=torch.float64)
    pred = torch.tensor(audit._make_pred(world, "generic"), dtype=torch.float64)[None]

    normal = gradient_fact(lambda p: asm(p, batch)[0], pred)
    assert normal["finite"] is True and normal["nonzero"] is True
    assert normal["error"] is None

    detached = gradient_fact(lambda p: asm(p.detach(), batch)[0], pred)
    assert detached["nonzero"] is False
    assert detached["error"] is not None

    zero_w = audit.build_synthetic_assembler(
        world.tokenizer,
        audit.LossWeights(w_pos=0, w_rot6d=0, w_joint=0, w_geodesic=0,
                          w_fk=0, w_vel=0, w_contact=0),
        torch.float64)
    zero = gradient_fact(lambda p: zero_w(p, batch)[0], pred)
    assert zero["finite"] is True and zero["nonzero"] is False


def test_parity_predicate_catches_tampering(audit_dir: Path):
    runs = _load("update_parity.json", audit_dir)["runs"]
    assert len(runs) == 4
    assert all(not r["problems"] for r in runs)
    assert all(r["n_micro_s0"] == 4 and r["n_micro_formal"] == 4 for r in runs)
    z = np.load(audit_dir / "update_arrays.npz")
    prefix = "deterministic_no_clip__"
    names = ["model.bias", "conditioner.scale"]
    for n in names:
        assert f"{prefix}s0_param_{n}" in z.files
        assert f"{prefix}formal_param_{n}" in z.files

    def make_res(prefix_: str = prefix, model_type: str = "deterministic") -> dict:
        params = {n: torch.tensor(z[f"{prefix_}s0_param_{n}"]) for n in names}
        tr_params = {n: torch.tensor(z[f"{prefix_}formal_param_{n}"]) for n in names}
        grads = {n: torch.tensor(z[f"{prefix_}s0_grad_{n}"]) for n in names}
        tr_grads = {n: torch.tensor(z[f"{prefix_}formal_grad_{n}"]) for n in names}
        state = {n: {"step": int(z[f"{prefix_}s0_step_{n}"]),
                     "exp_avg": torch.tensor(z[f"{prefix_}s0_exp_avg_{n}"]),
                     "exp_avg_sq": torch.tensor(z[f"{prefix_}s0_exp_avg_sq_{n}"])}
                 for n in names}
        tr_state = {n: {"step": int(z[f"{prefix_}formal_step_{n}"]),
                        "exp_avg": torch.tensor(z[f"{prefix_}formal_exp_avg_{n}"]),
                        "exp_avg_sq": torch.tensor(z[f"{prefix_}formal_exp_avg_sq_{n}"])}
                    for n in names}
        clip = {"pre": [torch.tensor(z[f"{prefix_}s0_preclip_0"]),
                        torch.tensor(z[f"{prefix_}s0_preclip_1"])],
                "post": [torch.tensor(z[f"{prefix_}s0_postclip_0"]),
                         torch.tensor(z[f"{prefix_}s0_postclip_1"])],
                "gnorm": float(z[f"{prefix_}s0_gnorm"])}
        tr_clip = {"pre": [torch.tensor(z[f"{prefix_}formal_preclip_0"]),
                           torch.tensor(z[f"{prefix_}formal_preclip_1"])],
                   "post": [torch.tensor(z[f"{prefix_}formal_postclip_0"]),
                            torch.tensor(z[f"{prefix_}formal_postclip_1"])],
                   "gnorm": float(z[f"{prefix_}formal_gnorm"])}
        res = {
            "s0_params": params, "tr_params": tr_params,
            "s0_grads": grads, "tr_grads": tr_grads,
            "s0_preds": [torch.tensor(v) for v in z[f"{prefix_}s0_preds"]],
            "tr_preds": [torch.tensor(v) for v in z[f"{prefix_}formal_preds"]],
            "s0_parts": [dict(zip(audit.LOSS_PART_KEYS, v)) for v in z[f"{prefix_}s0_parts"]],
            "tr_parts": [dict(zip(audit.LOSS_PART_KEYS, v)) for v in z[f"{prefix_}formal_parts"]],
            "s0_clip": clip, "tr_clip": tr_clip,
            "s0_state": state, "tr_state": tr_state,
            "s0_micro_calls": 4, "tr_micro_calls": 4,
            "s0_records": [], "tr_records": [],
        }
        if model_type == "diffusion":
            res["s0_records"] = [
                {"t": torch.tensor(z[f"{prefix_}s0_t"][i]),
                 "noise": torch.tensor(z[f"{prefix_}s0_noise"][i]),
                 "x_t": torch.tensor(z[f"{prefix_}s0_x_t"][i])}
                for i in range(4)]
            res["tr_records"] = [
                {"t": torch.tensor(z[f"{prefix_}formal_t"][i]),
                 "noise": torch.tensor(z[f"{prefix_}formal_noise"][i]),
                 "x_t": torch.tensor(z[f"{prefix_}formal_x_t"][i])}
                for i in range(4)]
        return res

    res = make_res()
    assert parity_problems(res) == []
    # conditioner missing from the optimizer / never used
    res["s0_grads"]["conditioner.scale"] = None
    res["tr_grads"]["conditioner.scale"] = None
    assert any("gradient_zero_or_nonfinite:conditioner.scale" in p
               for p in parity_problems(res))
    # no micro-batch actually consumed
    res = make_res()
    res["s0_micro_calls"] = 0
    assert "micro_calls_s0:0" in parity_problems(res)
    # parameters not updated
    res = make_res()
    res["s0_params"]["model.bias"] = torch.full_like(res["s0_params"]["model.bias"], 0.01)
    res["tr_params"]["model.bias"] = torch.full_like(res["tr_params"]["model.bias"], 0.01)
    assert "parameter_unchanged:model.bias" in parity_problems(res)
    # pre-clip gradients are a separate contract from post-clip gradients
    res = make_res()
    res["tr_clip"]["pre"][0] = res["tr_clip"]["pre"][0] + 0.01
    assert "preclip_mismatch:0" in parity_problems(res)
    res = make_res()
    res["s0_clip"]["gnorm"] = 1.5
    assert "preclip_gnorm_mismatch" in parity_problems(res)
    # forward output / per-term loss must also match
    res = make_res()
    res["tr_preds"][0] = res["tr_preds"][0] + 1.0
    assert "forward_output_mismatch" in parity_problems(res)
    res = make_res()
    res["tr_parts"][0]["pos"] = 1.0
    assert "loss_part_mismatch:pos" in parity_problems(res)
    # diffusion actual x_t must be the same on both sides
    res = make_res("diffusion_no_clip__", "diffusion")
    assert parity_problems(res) == []
    res["tr_records"][0]["x_t"] = res["tr_records"][0]["x_t"] + 1.0
    assert "t_epsilon_or_x_t_mismatch" in parity_problems(res)


def test_sampling_facts(audit_dir: Path):
    s = _load("sampling_trace.json", audit_dir)
    assert max(s["x_t_closed_form"]["max_abs_errors"]) <= 1e-6
    assert s["t0"]["noise_fraction"] > 0
    assert s["ddim"]["prediction_type"] == "sample"
    assert s["ddim"]["clip_sample"] is False
    assert s["ddim"]["thresholding"] is False
    rows = _rows(audit_dir)
    assert rows["sampling.ddim_single_step"]["status"] == "synthetic_verified"
    full = rows["sampling.ddim_full_loop"]
    assert full["status"] == "synthetic_verified"
    assert full["actual"]["sample_calls"] == 1
    assert full["actual"]["forward_calls"] == 5
    assert full["actual"]["scheduler_step_calls"] == 5
    assert full["actual"]["timesteps_visited"] == full["actual"]["scheduler_timesteps"]
    assert len(set(full["actual"]["timesteps_visited"])) == 5
    assert full["actual"]["reference_final_max_abs_error"] <= 1e-5


def test_data_consumption_traces(audit_dir: Path):
    s = _load("sampling_trace.json", audit_dir)["consumption"]
    assert s["s0"]["samples"] == 24
    assert s["s0"]["unique"] == 6
    assert all(v == 4 for v in s["s0"]["counts"].values())
    f = s["formal"]
    assert len(f["log"]) == 32
    assert all(len(d) == 2 for d in f["dropped_per_epoch"])
    assert sorted(f["coverage"]) == list(range(10))
    rows = _rows(audit_dir)
    assert rows["consumption.formal_4x4_drop_last"]["actual"]["samples_total"] == 32
    assert rows["consumption.formal_4x4_drop_last"]["actual"]["micro_batches"] == 8
    eq = rows["update.equal_size_big_batch"]
    assert eq["status"] == "synthetic_verified" and eq["actual"]["max_abs"] <= 1e-6


def test_rng_facts(audit_dir: Path):
    r = _load("rng_state_facts.json", audit_dir)
    assert r["facts"]["validation_restored"] is True
    assert r["facts"]["dataloader_dedicated_unchanged"] is True
    rows = _rows(audit_dir)
    assert rows["rng.validation_restores_streams"]["status"] == "synthetic_verified"
    assert rows["rng.module_modes_restored"]["status"] == "synthetic_verified"


def test_macro_micro_selection_and_best(audit_dir: Path):
    rows = _rows(audit_dir)
    rank = rows["aggregate.macro_vs_micro_ranking"]["actual"]
    assert rank["macro"]["method1"] < rank["macro"]["method2"]
    assert rank["micro"]["method1"] > rank["micro"]["method2"]
    sel = rows["selection.macro_from_raw_rows"]["actual"]
    assert sel["good"]["trainer"]["complete"] is True
    assert sel["good"]["trainer"]["macro_mpjpe_m"] == pytest.approx(0.11)
    for name, rec in sel["variants"].items():
        assert rec["trainer_complete"] is False, name
        assert rec["independent_complete"] is False, name
    best = rows["selection.best_uses_macro_complete_only"]["actual"]
    assert best["complete_run"]["best_formal_step"] == 2
    assert best["incomplete_run"]["best_formal_step"] == 1
    assert rows["aggregate.window_micro_average"]["evidence"]["max_abs_error"] < 1e-9


def test_independent_macro_counterexamples():
    expected = ["A", "B"]
    lengths = {"A": 10, "B": 20}
    good = [{"demo": "A", "T": 10, "status": "ok", "numerical_valid": True, "mpjpe_m": 0.1},
            {"demo": "B", "T": 20, "status": "ok", "numerical_valid": True, "mpjpe_m": 0.2}]
    assert independent_macro(good, expected, lengths)["macro"] == pytest.approx(0.15)
    variants = []
    missing = [good[0]]
    dup = [good[0], dict(good[0])]
    bool_metric = [good[0], dict(good[1], mpjpe_m=True)]
    nan_metric = [good[0], dict(good[1], mpjpe_m=float("nan"))]
    wrong_T = [good[0], dict(good[1], T=21)]
    missing_key = [good[0], {k: v for k, v in good[1].items() if k != "mpjpe_m"}]
    variants = [missing, dup, bool_metric, nan_metric, wrong_T, missing_key]
    for rs in variants:
        res = independent_macro(rs, expected, lengths)
        assert res["complete"] is False
        assert res["macro"] is None
        assert res["gaps"]


def test_determinism_and_rounding_limitation(audit_dir: Path):
    rows = _rows(audit_dir)
    det = rows["representation.determinism_bits"]
    assert det["actual"]["bitwise_equal"] is True
    unrounded = rows["representation.unrounded_artifacts"]
    assert unrounded["actual"]["has_beyond_6_decimals"] is True
    rounding = rows["representation.loss_history_rounding_limitation"]
    assert rounding["actual"]["all_core_fields_rounded_6"] is True
    assert rounding["limitation"] == "rounded_history_not_usable_for_exact_reconciliation"


# --- independent verifier attacks -------------------------------------------


def _copy_audit(audit_dir: Path, dst: Path) -> Path:
    shutil.copytree(audit_dir, dst)
    return dst


def test_verifier_detects_tampered_objective(audit_dir: Path, tmp_path: Path):
    d = _copy_audit(audit_dir, tmp_path / "tamper_objective")
    obj = _load("objective_rows.json", d)
    for r in obj["rows"]:
        if r["id"] == "loss.total_weighted_sum":
            r["actual"] = float(r["actual"]) + 0.1
    (d / "objective_rows.json").write_text(json.dumps(obj))
    v = verify_audit_artifacts(d)
    assert v["conclusion"] != "synthetic_contract_passed"
    assert any("objective_recompute_mismatch" in p for p in v["problems"])


def test_verifier_detects_missing_row_and_corrupt_artifact(audit_dir: Path, tmp_path: Path):
    d = _copy_audit(audit_dir, tmp_path / "tamper_missing")
    obj = _load("objective_rows.json", d)
    obj["rows"] = [r for r in obj["rows"] if r["id"] != "grad.vel"]
    (d / "objective_rows.json").write_text(json.dumps(obj))
    v = verify_audit_artifacts(d)
    assert any("missing_row:A2:grad.vel" in p for p in v["problems"])
    assert v["conclusion"] == "audit_incomplete"

    d2 = _copy_audit(audit_dir, tmp_path / "tamper_corrupt")
    (d2 / "gradient_rows.json").write_text("{not json")
    v2 = verify_audit_artifacts(d2)
    assert any("unreadable_artifact:gradient_rows.json" in p for p in v2["problems"])
    assert v2["conclusion"] == "audit_incomplete"


def test_verifier_detects_tampered_update_arrays(audit_dir: Path, tmp_path: Path):
    d = _copy_audit(audit_dir, tmp_path / "tamper_update")
    z = dict(np.load(d / "update_arrays.npz"))
    key = "deterministic_no_clip__formal_param_model.bias"
    z[key] = z[key] + np.float32(0.01)
    np.savez(d / "update_arrays.npz", **z)
    v = verify_audit_artifacts(d)
    assert any("update_replay_mismatch" in p and "model.bias" in p for p in v["problems"])
    assert v["conclusion"] == "implementation_mismatch"


def test_verifier_detects_tampered_gradient_array(audit_dir: Path, tmp_path: Path):
    d = _copy_audit(audit_dir, tmp_path / "tamper_gradient")
    z = dict(np.load(d / "gradient_arrays.npz"))
    z["grad_vel"] = z["grad_vel"] + 0.1
    np.savez(d / "gradient_arrays.npz", **z)
    v = verify_audit_artifacts(d)
    assert any("gradient_recompute_mismatch" in p or "gradient_saved_sum_mismatch" in p for p in v["problems"])


def test_verifier_does_not_crash_on_missing_artifact(audit_dir: Path, tmp_path: Path):
    d = _copy_audit(audit_dir, tmp_path / "tamper_missing_artifact")
    (d / "static_contract.json").unlink()
    v = verify_audit_artifacts(d)
    assert v["conclusion"] == "audit_incomplete"
    assert any("missing_artifact:static_contract.json" in p for p in v["problems"])


# --- CLI guard --------------------------------------------------------------


def test_cli_rejects_real_inputs(monkeypatch, tmp_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_cli", ROOT / "scripts/audit_w1_training_objective.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    values = {"--training-manifest": "m.json", "--cache-dir": "cache",
              "--checkpoint": "last.pt", "--execution-registration": "reg.json",
              "--demos": "a", "--budget-deadline-s": "10", "--steps": "1",
              "--seed": "7", "--model": "diffusion"}
    for flag, value in values.items():
        with pytest.raises(SystemExit) as exc:
            monkeypatch.setattr(sys, "argv",
                                ["audit_w1_training_objective.py", "--out", str(tmp_path / "out"),
                                 flag, value])
            cli.main()
        assert "refuses real/research inputs" in str(exc.value)
    with pytest.raises(SystemExit):
        cli._guard_out(ROOT / "derived/experiments/omniretarget_first")


def test_cli_runs_synthetic_only(tmp_path: Path):
    out = tmp_path / "cli_out"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_w1_training_objective.py"),
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    summary_files = list(out.glob("audit_summary.json")) + list(out.glob("attempt_*/audit_summary.json"))
    assert summary_files, proc.stdout
    summary = json.loads(summary_files[0].read_text())
    assert summary["conclusion"] == "synthetic_contract_passed"
    assert summary["scope"]["real_dataset_loaded"] is False


def test_empty_roster_is_not_a_vacuous_pass():
    res = independent_macro([], [], None)
    assert res["complete"] is False and res["macro"] is None
    assert "invalid_registered_roster" in res["gaps"]
    res2 = independent_macro([], ["A"], {"A": 10})
    assert res2["complete"] is False and res2["macro"] is None
    assert any(g.startswith("demo_row_count:A") for g in res2["gaps"])


def test_missing_grad_accum_division_is_detected():
    world = build_synthetic_world()
    cfg = audit._stub_cfg("deterministic", steps=1, micro_batch=4,
                          grad_accum=2, grad_clip=1e9)
    model = audit._StubModel("deterministic")
    cond = audit._StubConditioner()
    with torch.no_grad():
        cond.scale.fill_(1.5)
    asm = build_synthetic_assembler(world.tokenizer, cfg.loss, torch.float32, use_fk=False)
    items = audit._make_items(world, 6)
    for it in items[4:]:
        it["x0"][..., 0] += 3.0
    micro = [audit._ListDataset(items[:4]), audit._ListDataset(items[4:])]

    def batches(ds):
        return {k: torch.stack([x[k] for x in ds.items])
                for k in ds.items[0] if torch.is_tensor(ds.items[0][k])}

    def forward_losses():
        return [asm(model(cond(batches(ds)), world.T), batches(ds))[0] for ds in micro]

    model.zero_grad(set_to_none=True); cond.zero_grad(set_to_none=True)
    for loss in forward_losses():
        (loss / cfg.grad_accum).backward()
    g_div = model.bias.grad.detach().clone()
    model.zero_grad(set_to_none=True); cond.zero_grad(set_to_none=True)
    for loss in forward_losses():
        loss.backward()          # deliberately missing the 1/grad_accum factor
    g_nodiv = model.bias.grad.detach().clone()
    assert float((g_div - g_nodiv).abs().max()) > 1e-6
    # the audit's recorded fact is the divided path; the contaminated path is
    # distinguishable, so a regression cannot hide behind rounded logs
    assert float(g_div.norm()) > 0


def test_conditioner_not_in_optimizer_is_detected():
    world = build_synthetic_world()
    z = {"conditioner.scale": torch.tensor(1.0), "model.bias": torch.tensor(0.0099)}
    res = {
        "s0_params": {k: v.clone() for k, v in z.items()},
        "tr_params": {k: v.clone() for k, v in z.items()},
        "s0_grads": {k: torch.ones_like(v) for k, v in z.items()},
        "tr_grads": {k: torch.ones_like(v) for k, v in z.items()},
        "s0_state": {k: {"step": 1, "exp_avg": torch.zeros_like(v),
                         "exp_avg_sq": torch.zeros_like(v)} for k, v in z.items()},
        "tr_state": {k: {"step": 1, "exp_avg": torch.zeros_like(v),
                         "exp_avg_sq": torch.zeros_like(v)} for k, v in z.items()},
        "s0_micro_calls": 4, "tr_micro_calls": 4,
        "s0_records": [], "tr_records": [],
    }
    res["s0_grads"]["conditioner.scale"] = None
    res["tr_grads"]["conditioner.scale"] = None
    problems = parity_problems(res)
    assert "gradient_zero_or_nonfinite:conditioner.scale" in problems
    # a completely omitted conditioner parameter (never updated) is also caught
    res2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in res.items()}
    res2["s0_grads"]["conditioner.scale"] = torch.ones(())
    res2["tr_grads"]["conditioner.scale"] = torch.ones(())
    res2["s0_params"]["conditioner.scale"] = torch.tensor(1.0)
    res2["tr_params"]["conditioner.scale"] = torch.tensor(1.0)
    assert "parameter_unchanged:conditioner.scale" in parity_problems(res2)


def test_contact_branch_does_not_execute_at_zero_weight(monkeypatch):
    world = build_synthetic_world()
    fk = audit.SyntheticFK(torch.float64)
    asm = build_synthetic_assembler(
        world.tokenizer,
        audit.LossWeights(w_pos=1, w_rot6d=1, w_joint=1, w_geodesic=.5,
                          w_fk=2, w_contact=0, w_vel=.5),
        torch.float64, fk=fk)
    batch = make_synthetic_batch(world, dtype=torch.float64)
    pred = torch.tensor(audit._make_pred(world, "generic"), dtype=torch.float64)[None]
    sdf_calls = []
    real_sdf = losses_mod.point_box_sdf_torch

    def spy(*a, **k):
        sdf_calls.append(1)
        return real_sdf(*a, **k)

    monkeypatch.setattr(losses_mod, "point_box_sdf_torch", spy)
    _, parts = asm(pred, batch)
    assert "contact_m2" not in parts
    assert sdf_calls == []
    assert all(tuple(c) != tuple(audit.HAND_SITE_NAMES) for c in fk.calls)


def _tampered(audit_dir: Path, tmp_path: Path, name: str, mutate) -> dict:
    d = _copy_audit(audit_dir, tmp_path / name)
    mutate(d)
    return verify_audit_artifacts(d)


def test_verifier_detects_a5_evidence_attacks(audit_dir: Path, tmp_path: Path):
    def empty_parity(d):
        (d / "update_parity.json").write_text("{}")

    def best_999(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "selection.best_uses_macro_complete_only":
                r["actual"]["complete_run"]["best_formal_step"] = 999
        (d / "objective_rows.json").write_text(json.dumps(o))

    def nan_total(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "loss.total_weighted_sum":
                r["actual"] = float("nan")
        (d / "objective_rows.json").write_text(json.dumps(o))

    def nan_grad(d):
        z = dict(np.load(d / "gradient_arrays.npz"))
        z["grad_vel"] = z["grad_vel"].copy()
        z["grad_vel"].flat[0] = np.nan
        np.savez(d / "gradient_arrays.npz", **z)

    def double_grads(d):
        z = dict(np.load(d / "gradient_arrays.npz"))
        for k in list(z):
            if k.startswith("grad_"):
                z[k] = z[k] * 2
        np.savez(d / "gradient_arrays.npz", **z)

    def root_shift(d):
        z = dict(np.load(d / "synthetic_inputs.npz"))
        z["root"] = z["root"].copy()
        z["root"][:, 0] += 100.0
        np.savez(d / "synthetic_inputs.npz", **z)

    def wfk_2000(d):
        s = _load("static_contract.json", d)
        s["effective_config"]["loss"]["w_fk"] = 2000.0
        (d / "static_contract.json").write_text(json.dumps(s))

    def xt_miswire(d):
        z = dict(np.load(d / "update_arrays.npz"))
        for k in list(z):
            if k.endswith("__formal_x_t"):
                run = k[: -len("formal_x_t")]
                z[k] = z[run + "micro_x0"].copy()
        np.savez(d / "update_arrays.npz", **z)

    def objective_list(d):
        (d / "objective_rows.json").write_text("[]")

    def empty_errors(d):
        s = _load("sampling_trace.json", d)
        s["x_t_closed_form"]["max_abs_errors"] = []
        (d / "sampling_trace.json").write_text(json.dumps(s))

    def ddim_not_run(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "sampling.ddim_full_loop":
                r["actual"]["sample_calls"] = 0
        (d / "objective_rows.json").write_text(json.dumps(o))

    def empty_gradient(d):
        np.savez(d / "gradient_arrays.npz")
        (d / "gradient_rows.json").write_text("{}")

    def grad_status(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "grad.vel":
                r["status"] = "runtime_not_checked"
        (d / "objective_rows.json").write_text(json.dumps(o))

    def grad_nan_row(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "grad.vel":
                r["actual"]["grad_norm"] = float("nan")
        (d / "objective_rows.json").write_text(json.dumps(o))

    def fd_bad(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "grad.finite_difference_float64":
                r["actual"]["vel"]["rel_error"] = 1.0
                r["actual"]["vel"]["ok"] = False
        (d / "objective_rows.json").write_text(json.dumps(o))

    def one_run(d):
        z = dict(np.load(d / "update_arrays.npz"))
        z = {k: v for k, v in z.items() if k.startswith("deterministic_no_clip__")}
        np.savez(d / "update_arrays.npz", **z)
        p = _load("update_parity.json", d)
        p["runs"] = [r for r in p["runs"] if r["run_id"] == "deterministic_no_clip"]
        (d / "update_parity.json").write_text(json.dumps(p))

    def noise_zero(d):
        z = dict(np.load(d / "update_arrays.npz"))
        for k in list(z):
            if k.endswith("__s0_noise") or k.endswith("__formal_noise"):
                z[k] = np.zeros_like(z[k])
        np.savez(d / "update_arrays.npz", **z)

    def noise_inf(d):
        z = dict(np.load(d / "update_arrays.npz"))
        for k in list(z):
            if k.endswith("__s0_noise") or k.endswith("__formal_noise"):
                z[k] = np.full_like(z[k], np.inf)
        np.savez(d / "update_arrays.npz", **z)

    def static_short(d):
        s = _load("static_contract.json", d)
        s["checks"] = s["checks"][:1]
        (d / "static_contract.json").write_text(json.dumps(s))

    def gap_add(d):
        s = _load("audit_summary.json", d)
        s["gaps"] = ["missing_row:A1:fake"]
        s["counts"]["gaps"] = 1
        (d / "audit_summary.json").write_text(json.dumps(s))

    def permute_bad(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "loss.batch_permutation_invariance":
                r["actual"]["permuted"][0] += 0.1
        (d / "objective_rows.json").write_text(json.dumps(o))

    def ddim_zero(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "sampling.ddim_full_loop":
                r["actual"]["scheduler_step_calls"] = 0
                r["actual"]["timesteps_visited"] = [0, 0, 0, 0, 0]
        (d / "objective_rows.json").write_text(json.dumps(o))

    def clip_arrays(d):
        z = dict(np.load(d / "update_arrays.npz"))
        for k in list(z):
            for with_run, no_run in (("deterministic_with_clip__", "deterministic_no_clip__"),
                                     ("diffusion_with_clip__", "diffusion_no_clip__")):
                if k.startswith(with_run):
                    z[k] = z[no_run + k[len(with_run):]].copy()
        np.savez(d / "update_arrays.npz", **z)

    def scalar_3d(d):
        z = dict(np.load(d / "update_arrays.npz"))
        for k in list(z):
            if k.endswith("initial_bias") or k.endswith("initial_scale"):
                z[k] = np.full((3, 1, 1), float(z[k]), dtype=np.float32)
        np.savez(d / "update_arrays.npz", **z)

    def fd_raw(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "grad.finite_difference_float64":
                r["actual"]["vel"]["fd"] = 1.0
        (d / "objective_rows.json").write_text(json.dumps(o))

    def fd_delete(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] == "grad.finite_difference_float64":
                for term in r["actual"]:
                    r["actual"][term].pop("fd", None)
                    r["actual"][term].pop("analytic", None)
        (d / "objective_rows.json").write_text(json.dumps(o))

    def batch_bad(d):
        o = _load("objective_rows.json", d)
        for r in o["rows"]:
            if r["id"] in ("loss.batch_permutation_invariance",
                           "loss.repeated_batch_invariance"):
                for key in ("unpermuted", "permuted", "repeated", "independent_mean"):
                    if isinstance(r["actual"].get(key), list):
                        r["actual"][key][0] += 0.1
        (d / "objective_rows.json").write_text(json.dumps(o))

    def grad_detail_bad(d):
        g = _load("gradient_rows.json", d)
        for r in g["rows"]:
            if r["id"] == "grad.vel":
                r["actual"]["grad_norm"] = 123.0
                r["actual"]["weighted_grad_norm"] = 456.0
        (d / "gradient_rows.json").write_text(json.dumps(g))

    def summary_empty(d):
        (d / "audit_summary.json").write_text("{}")

    def config_bad(d):
        s = _load("static_contract.json", d)
        s["effective_config"]["seed"] = 999
        s["effective_config"]["grad_clip"] = 999.0
        (d / "static_contract.json").write_text(json.dumps(s))

    expected = {
        "empty_parity": ("audit_incomplete", "structure_empty:update_parity.runs"),
        "best_999": ("implementation_mismatch", "selection_row_mismatch"),
        "nan_total": ("implementation_mismatch", "nonfinite:objective"),
        "nan_grad": ("implementation_mismatch", "nonfinite:gradient_arrays"),
        "double_grads": ("implementation_mismatch", "gradient_recompute_mismatch"),
        "root_shift": ("implementation_mismatch", "input_x0_not_bound"),
        "wfk_2000": ("implementation_mismatch", "static_effective_loss_mismatch"),
        "xt_miswire": ("implementation_mismatch", "update_replay_mismatch"),
        "objective_list": ("audit_incomplete", "structure_not_mapping:objective_rows"),
        "empty_errors": ("audit_incomplete", "sampling_x_t_errors_empty"),
        "ddim_not_run": ("implementation_mismatch", "sampling_full_loop_not_executed"),
        "empty_gradient": ("audit_incomplete", "gradient_rows"),
        "grad_status": ("audit_incomplete", "required_row_not_verified:A2:grad.vel"),
        "grad_nan_row": ("implementation_mismatch", "nonfinite:objective_rows"),
        "fd_bad": ("implementation_mismatch", "finite_difference_row_not_verified"),
        "one_run": ("audit_incomplete", "update_run_set_mismatch"),
        "noise_zero": ("implementation_mismatch", "noise_to_x_t"),
        "noise_inf": ("implementation_mismatch", "nonfinite:diffusion"),
        "static_short": ("audit_incomplete", "static_check_set_mismatch"),
        "gap_add": ("audit_incomplete", "summary_gap_list_mismatch"),
        "permute_bad": ("implementation_mismatch", "batch_invariance_mismatch"),
        "ddim_zero": ("implementation_mismatch", "sampling_full_loop_forward_or_step_count"),
        "clip_arrays": ("implementation_mismatch", "grad_clip_identity"),
        "scalar_3d": ("audit_incomplete", "structure_shape"),
        "fd_raw": ("implementation_mismatch", "finite_difference_recompute_mismatch"),
        "fd_delete": ("implementation_mismatch", "finite_difference_recompute_mismatch"),
        "batch_bad": ("implementation_mismatch", "batch_invariance_mismatch"),
        "grad_detail_bad": ("implementation_mismatch", "gradient_detail_mismatch"),
        "summary_empty": ("audit_incomplete", "structure_empty:audit_summary"),
        "config_bad": ("implementation_mismatch", "static_effective_config_mismatch"),
    }
    cases = [("empty_parity", empty_parity), ("best_999", best_999),
             ("nan_total", nan_total), ("nan_grad", nan_grad),
             ("double_grads", double_grads), ("root_shift", root_shift),
             ("wfk_2000", wfk_2000), ("xt_miswire", xt_miswire),
             ("objective_list", objective_list), ("empty_errors", empty_errors),
             ("ddim_not_run", ddim_not_run), ("empty_gradient", empty_gradient),
             ("grad_status", grad_status), ("grad_nan_row", grad_nan_row),
             ("fd_bad", fd_bad), ("one_run", one_run), ("noise_zero", noise_zero),
             ("noise_inf", noise_inf), ("static_short", static_short),
             ("gap_add", gap_add), ("permute_bad", permute_bad),
             ("ddim_zero", ddim_zero), ("clip_arrays", clip_arrays),
             ("scalar_3d", scalar_3d), ("fd_raw", fd_raw), ("fd_delete", fd_delete),
             ("batch_bad", batch_bad), ("grad_detail_bad", grad_detail_bad),
             ("summary_empty", summary_empty), ("config_bad", config_bad)]
    for name, mutate in cases:
        v = _tampered(audit_dir, tmp_path, f"attack_{name}", mutate)
        want_conclusion, want_problem = expected[name]
        assert v["conclusion"] == want_conclusion, (name, v["problems"])
        assert any(want_problem in p for p in v["problems"]), (name, v["problems"])


def test_batch_invariance_is_not_zero_loss_and_detects_batch_sum():
    world = build_synthetic_world()
    weights = audit.W1_SYNTHETIC_WEIGHTS
    target = world.x0
    pred = target.copy()
    pred[2:5, 0:3] += 0.02
    a = build_synthetic_assembler(world.tokenizer, weights, torch.float64)
    batch = make_synthetic_batch(world, B=3, dtype=torch.float64,
                                 x0=np.stack([target, target, target]))
    _, parts = a(torch.tensor(np.stack([pred, pred, pred]), dtype=torch.float64), batch)
    expected = float(np.mean([
        audit._independent_terms(pred, target, world, weights)["total"]
        for _ in range(3)]))
    assert parts["total"] > 1e-6
    assert abs(parts["total"] - expected) < 1e-12
    # a batch-size-summing loss would report 3x the mean; the audit's independent
    # expected mean catches it instead of comparing the buggy value to itself
    assert abs(parts["total"] * 3 - expected) > 1e-6


def test_update_arrays_contain_all_four_runs_for_independent_replay(audit_dir: Path):
    z = np.load(audit_dir / "update_arrays.npz")
    for run_id in ("deterministic_no_clip", "deterministic_with_clip",
                   "diffusion_no_clip", "diffusion_with_clip"):
        for suffix in ("micro_x0", "s0_preds", "formal_preds", "s0_parts",
                       "formal_parts", "s0_preclip_0", "formal_preclip_0",
                       "s0_postclip_0", "formal_postclip_0", "s0_gnorm",
                       "formal_gnorm", "s0_param_model.bias",
                       "formal_param_model.bias", "s0_grad_model.bias",
                       "formal_grad_model.bias", "s0_exp_avg_model.bias",
                       "formal_exp_avg_model.bias"):
            assert f"{run_id}__{suffix}" in z.files, (run_id, suffix)
    # diffusion runs must carry the real t/epsilon/x_t inputs
    for run_id in ("diffusion_no_clip", "diffusion_with_clip"):
        for suffix in ("s0_t", "formal_t", "s0_noise", "formal_noise",
                       "s0_x_t", "formal_x_t"):
            assert f"{run_id}__{suffix}" in z.files, (run_id, suffix)


def test_ddim_full_loop_is_executed(monkeypatch, tmp_path: Path):
    from ireset.diffusion.generator import DiffusionG1

    def boom(*_a, **_k):
        raise RuntimeError("sampling loop disabled")

    monkeypatch.setattr(DiffusionG1, "sample", boom)
    rows, _, _ = audit.run_a3(build_synthetic_world(), tmp_path)
    row = next(r for r in rows if r["id"] == "sampling.ddim_full_loop")
    assert row["status"] == "contract_mismatch"
    assert row["actual"]["sample_error"] is not None


def test_ddim_schedule_and_terminal_update_are_bound(monkeypatch, tmp_path: Path):
    from diffusers import DDIMScheduler
    from ireset.diffusion.generator import DiffusionG1

    # (a) a sampled schedule [4,3,2,1,0] must not match the frozen table
    def wrong_schedule(self, memory, F, steps=25, eta=0.0, generator=None):
        self.ddim.timesteps = torch.tensor([4, 3, 2, 1, 0])
        x = torch.randn(memory.shape[0], F, 38, generator=generator)
        for tv in self.ddim.timesteps:
            x0 = self.forward(x, tv.expand(memory.shape[0]), memory)
            x = self.ddim.step(x0, tv, x, eta=eta).prev_sample
        return x

    monkeypatch.setattr(DiffusionG1, "sample", wrong_schedule)
    rows, _, _ = audit.run_a3(build_synthetic_world(), tmp_path / "schedule")
    row = next(r for r in rows if r["id"] == "sampling.ddim_full_loop")
    assert row["status"] == "contract_mismatch"
    assert row["actual"]["timesteps_visited"] != row["actual"]["scheduler_timesteps"]


def test_ddim_terminal_step_is_bound(monkeypatch, tmp_path: Path):
    from diffusers import DDIMScheduler
    from ireset.diffusion.generator import DiffusionG1

    class Out:
        def __init__(self, prev_sample):
            self.prev_sample = prev_sample

    real_step = DDIMScheduler.step

    def broken_step(self, model_output, timestep, sample, eta=0.0, **kwargs):
        if int(timestep.reshape(-1)[0]) == 0:
            return Out(sample)          # deliberately return the old state
        return real_step(self, model_output, timestep, sample, eta=eta, **kwargs)

    monkeypatch.setattr(DDIMScheduler, "step", broken_step)
    rows, _, _ = audit.run_a3(build_synthetic_world(), tmp_path / "terminal")
    row = next(r for r in rows if r["id"] == "sampling.ddim_full_loop")
    assert row["status"] == "contract_mismatch"
    assert row["actual"]["reference_final_max_abs_error"] > 1e-5
