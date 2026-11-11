"""I-phase small-set fit diagnostic tests (plan v0.4 §6).

Pure-synthetic and offline: synthetic tokenizer/joint range/teacher windows
and a synthetic 15-site FK stub.  No real data, checkpoint, asset compile or
CUDA path is touched; nothing here claims production equivalence.

The heavy test is the full §6.2/v0.3/v0.4 counterexample battery (115 cases)
shared with the synthetic acceptance CLI.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.eval import w1_smallset_fit_diagnostic as diag


@pytest.fixture(scope="session", autouse=True)
def _single_thread_torch():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


# --------------------------------------------------------------------------
# frozen plan / registration
# --------------------------------------------------------------------------

def test_frozen_plan_defaults_match_plan_constants():
    assert diag.frozen_plan_defaults_problems() == []
    d1, d6 = diag.FitPlan("D1"), diag.FitPlan("D6")
    assert (d1.micro_batch, d6.micro_batch) == (1, 6)
    assert d1.grad_accum == 4 and d1.steps == 1000 and d1.probe_every == 10
    assert d1.n_probes == 101
    assert d1.probe_steps[0] == 0 and d1.probe_steps[-1] == 1000
    # non-synthetic plans may not deviate from the frozen values
    with pytest.raises(diag.FitDiagnosticError):
        diag.FitPlan("D1", steps=500)
    # step count must align with the probe schedule
    with pytest.raises(diag.FitDiagnosticError):
        diag.FitPlan("D1", steps=999, synthetic=True)


def test_registration_validation():
    w6 = diag.make_synthetic_world(seed=7, n_windows=6, d_model=16)
    regs = w6.registration_for("D6")
    assert regs.problems() == []
    d1 = w6.registration_for("D1")
    assert d1.problems() == []
    assert len(d1.windows) == 1 and d1.windows[0] == regs.windows[0]
    # D6 needs exactly six windows
    short = diag.FitRegistration("D6", windows=regs.windows[:5],
                                 splits=("train",) * 5)
    assert any("window_count" in p for p in short.problems())
    dup = diag.FitRegistration("D6", windows=regs.windows[:5] + (regs.windows[0],),
                               splits=("train",) * 6)
    assert any("duplicate_window" in p for p in dup.problems())
    mixed = diag.FitRegistration("D6", windows=regs.windows,
                                 splits=("train",) * 5 + ("val",))
    assert any("split_not_train" in p for p in mixed.problems())
    span = diag.FitRegistration("D1", windows=(diag.WindowKey("d", 10, 70),),
                                splits=("train",))
    assert any("window_span" in p for p in span.problems())


# --------------------------------------------------------------------------
# metric_decode_v1 and §7.1/§7.2 metrics
# --------------------------------------------------------------------------

def test_metric_decoder_exact_promotion_and_binding():
    joint_range = np.stack([-2.0 + 0.02 * np.arange(29),
                            2.0 - 0.02 * np.arange(29)], axis=1)
    mean = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.3], dtype=np.float32)
    dec = diag.MetricDecoder(mean, std, joint_range[:, 0], joint_range[:, 1])
    # constants are exact float32 promotions of the production-effective values
    assert dec.mean.dtype == np.float64
    assert np.array_equal(dec.mean.astype(np.float32), mean)
    assert np.array_equal(dec.lo.astype(np.float32),
                          joint_range[:, 0].astype(np.float32))
    # identity 6D decodes to the exact identity rotation; zero joints decode
    # to the exact mid-range; pos 0 decodes to the exact mean
    x0 = np.zeros((diag.T_WINDOW, 38), dtype=np.float32)
    x0[:, 3] = 1.0
    x0[:, 7] = 1.0
    out = dec.decode(x0)
    assert np.array_equal(out["pos"], np.tile(mean.astype(np.float64),
                                              (diag.T_WINDOW, 1)))
    assert np.allclose(out["R"], np.eye(3), atol=0)
    mid = (dec.lo + dec.hi) / 2.0
    assert np.allclose(out["q"], np.tile(mid, (diag.T_WINDOW, 1)), atol=0)
    # legality: dtype / shape / finiteness / rot6d degeneracy are hard refusals
    assert dec.verify_x0(x0.astype(np.float64)) != []
    bad = x0.copy()
    bad[0, 3] = 0.0
    bad[0, 7] = 0.0
    assert any("rotation6d" in p for p in dec.verify_x0(bad))
    with pytest.raises(diag.FitDiagnosticError):
        dec.decode(bad)


def test_window_metric_formulas_units_and_gates():
    tgt = {"pos": np.zeros((diag.T_WINDOW, 3)),
           "R": np.tile(np.eye(3), (diag.T_WINDOW, 1, 1)),
           "q": np.zeros((diag.T_WINDOW, diag.J_JOINTS))}
    pred = {"pos": np.zeros((diag.T_WINDOW, 3)),
            "R": np.tile(np.eye(3), (diag.T_WINDOW, 1, 1)),
            "q": np.zeros((diag.T_WINDOW, diag.J_JOINTS))}
    pred["pos"][:, 0] = 0.03
    pred["q"][:, 0] = 0.04
    fk_p = np.zeros((diag.T_WINDOW, diag.S_SITES, 3))
    fk_p[:, 0, 0] = 0.02
    m = diag.compute_window_metrics(pred, tgt, fk_p, np.zeros_like(fk_p))
    # §7.1 formulas: RMSE over frames of the norm; joint RMSE over T×J;
    # MPJPE = mean per-site distance (NOT the loss's mean-squared form)
    assert abs(m["root_position_rmse_m"] - 0.03) < 1e-12
    assert m["root_position_rmse_m"] != pytest.approx(
        float(np.sqrt(0.03 ** 2 / 3.0)))
    assert abs(m["joint_rmse_rad"] - 0.04 / math.sqrt(diag.J_JOINTS)) < 1e-12
    assert abs(m["teacher_fk_mpjpe_m"] - 0.02 / diag.S_SITES) < 1e-12
    assert m["root_rotation_mean_rad"] == 0.0
    assert m["gates"]["root_position_rmse_m"] is False
    assert m["gates"]["joint_rmse_rad"] is True
    assert m["window_mean_fit_passed"] is False
    # rotation geodesic: 1-degree relative yaw
    ang = math.radians(1.0)
    R = np.array([[math.cos(ang), -math.sin(ang), 0.0],
                  [math.sin(ang), math.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    pred2 = {"pos": tgt["pos"].copy(), "R": np.tile(R, (diag.T_WINDOW, 1, 1)),
             "q": tgt["q"].copy()}
    m2 = diag.compute_window_metrics(pred2, tgt, fk_p, np.zeros_like(fk_p))
    assert abs(m2["root_rotation_mean_rad"] - math.radians(1.0)) < 1e-9
    assert m2["gates"]["root_rotation_mean_rad"] is True


def test_spike_visibility_single_frame_60deg():
    # the plan's own example: one 60deg frame in a quiet window passes the
    # 1-degree MEAN gate while p95/max keep the spike fully visible
    angles = np.zeros(diag.T_WINDOW)
    angles[17] = math.radians(60.0)
    assert float(np.mean(angles)) <= diag.GATE_ROOT_ROT_MEAN_RAD + 1e-12
    assert diag._max_first(angles) == (math.radians(60.0), (17,))
    stats = diag.window_spike_stats(angles)
    assert stats["max"] == math.radians(60.0)
    assert stats["worst_coordinate"] == [17]
    # ties resolve to the smallest frame, then joint/site order
    tie = np.zeros((4, 5))
    tie[1, 2] = tie[3, 0] = 5.0
    assert diag._max_first(tie) == (5.0, (1, 2))
    assert diag.SPIKE_GATE_STATUS == "not_evaluated"


def test_aggregate_local_verdicts_macro_never_substitutes_conjunction():
    windows = [{"window_id": f"w{i}",
                "metrics": {"window_mean_fit_passed": True,
                            "root_position_rmse_m": 0.001}}
               for i in range(5)]
    windows.append({"window_id": "w5",
                    "metrics": {"window_mean_fit_passed": False,
                                "root_position_rmse_m": 0.001}})
    agg = diag.aggregate_local_verdicts(windows)
    assert agg["local_mean_fit_passed"] is False
    assert agg["macro_root_position_rmse_m"] is not None
    assert agg["macro_is_not_a_gate"] is True


def test_local_report_validator_blocks_final_namespace():
    ok = {"schema": diag.STAGE_REPORT_SCHEMA, "local_complete_evidence": True,
          "local_mean_fit_passed": True, "local_conclusion": "complete_fit",
          "spike_gate_status": "not_evaluated",
          "final_verification_status": "pending_F"}
    assert diag.local_report_problems(ok) == []
    for key in ("complete_evidence", "mean_fit_passed", "replay_passed"):
        bad = {**ok, key: True}
        assert any("final_namespace_field" in p
                   for p in diag.local_report_problems(bad))
    bad = {**ok, "final_verification_status": "final_F"}
    assert diag.local_report_problems(bad) != []
    bad = {**ok, "local_complete_evidence": False, "local_mean_fit_passed": True}
    assert any("without_complete_evidence" in p
               for p in diag.local_report_problems(bad))


# --------------------------------------------------------------------------
# attempt ledger / R guards
# --------------------------------------------------------------------------

def test_attempt_ledger_running_row_is_unknown_history():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "run"
        # a run root with neither attempts nor rows proves an empty history
        assert diag.AttemptLedger(Path(td) / "fresh", "D1").prior_state() == \
            (0.0, [])
        (root / "D1" / "attempt_001").mkdir(parents=True)
        ledger = diag.AttemptLedger(root, "D1")
        ledger.begin("attempt_001")  # a SIGKILL leaves this row running
        prior, problems = ledger.prior_state()
        assert prior is None and any("running_row" in p for p in problems)
        with pytest.raises(diag.FitDiagnosticError):
            diag.remaining_budget_or_refuse(root, "D1",
                                            diag.StageBudget("D1", 120.0, 15.0))


def test_failed_attempt_does_not_reset_budget():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "run"
        (root / "D1" / "attempt_001").mkdir(parents=True)
        ledger = diag.AttemptLedger(root, "D1")
        ledger.begin("attempt_001")
        ledger.finish("attempt_001", 110.0, "done", {"verdict": "incomplete"})
        assert ledger.prior_state() == (110.0, [])
        with pytest.raises(diag.FitDiagnosticError, match="budget_exhausted"):
            diag.remaining_budget_or_refuse(
                root, "D1", diag.StageBudget("D1", 120.0, 15.0))


def test_r_metadata_guard_and_unbilled_materialization():
    reader = diag.MetadataOnlyReader()
    with pytest.raises(diag.FitDiagnosticError,
                       match="r_metadata_only_array_read_forbidden"):
        reader.load_arrays("any.npz")
    with pytest.raises(diag.FitDiagnosticError,
                       match="r_metadata_only_checkpoint_read_forbidden"):
        reader.torch_load("any.pt")
    with pytest.raises(diag.FitDiagnosticError,
                       match="materialization_without_billed_interval"):
        diag.materialize_for_fit(object(), None)
    with pytest.raises(diag.FitDiagnosticError,
                       match="materialization_stage_not_fit"):
        diag.materialize_for_fit(object(), diag.StageBudget("R", 10.0, 2.0))


def test_runtime_guards_parameter_contract():
    module = torch.nn.Linear(3, 3)
    assert diag.parameter_contract_problems({"m": module}, "cpu",
                                            torch.float32) == []
    module.register_parameter(
        "bad", torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)))
    problems = diag.parameter_contract_problems({"m": module}, "cpu",
                                                torch.float32)
    assert any("param_dtype" in p for p in problems)
    problems = diag.parameter_contract_problems({"m": module}, "cuda",
                                                torch.float32)
    assert any("param_device" in p for p in problems)


# --------------------------------------------------------------------------
# end-to-end synthetic stage / F replay
# --------------------------------------------------------------------------

def _mini(tmp, stage="D1", n_windows=1, steps=2, probe_every=1, seed=99,
          finish=True):
    world = diag.make_synthetic_world(seed=seed, n_windows=n_windows, d_model=16)
    plan = diag.FitPlan(stage, steps=steps, probe_every=probe_every,
                        synthetic=True, d_model=16, n_layers=1, n_heads=4,
                        ffn_mult=2, lr=0.02)
    view = world.view_for(stage)
    budget = diag.StageBudget(stage=stage, wall_cap_s=600.0, reserve_s=10.0)
    trainer = world.build_trainer(plan, Path(tmp) / stage.lower(),
                                  budget_deadline=budget.work_deadline,
                                  train_ds=view["ds"])
    driver = diag.FitDriver(
        trainer=trainer, plan=plan, registration=view["registration"],
        loss_fn=world.loss_fn, decoder=world.decoder,
        probe_batch=view["probe_batch"], budget=budget,
        runtime_guard=diag.SyntheticRuntimeGuard(), identity={"synthetic": True},
        stage_dir=Path(tmp) / stage.lower(),
        training_gen=torch.Generator().manual_seed(plan.seed),
        dataset_window_keys=view["ds"].window_keys,
        authorization_includes_d6=True)
    summary = driver.run()
    finished = diag.finish_stage(Path(tmp) / stage.lower(), summary, budget) \
        if finish else None
    return world, plan, driver, summary, Path(tmp) / stage.lower(), budget, finished


def test_d6_driver_window_consumption_and_probe_multiset():
    with tempfile.TemporaryDirectory() as td:
        world, plan, driver, summary, sdir, budget, finished = _mini(
            td, stage="D6", n_windows=6, steps=1, probe_every=1)
        assert summary["local_complete_evidence"] is True, summary.get("failure")
        assert summary["call_matrix"]["problems"] == []
        counters = summary["counters"]
        # one step = grad_accum(4) micros x 6 windows = 24 window consumptions
        assert counters["window_consumptions_total"] == 24
        assert counters["adamw_step_calls"] == 1
        assert counters["backward_calls"] == 4
        # two probes x 6 registered windows = 12 coverage rows
        cov = summary["coverage"]
        assert cov["expected_rows"] == 12 and cov["actual_rows"] == 12
        assert cov["missing"] == [] and cov["duplicated"] == []
        # every probe is exactly one conditioner + one backbone forward
        pidx = json.loads((sdir / "probe_index.json").read_text())
        assert len(pidx["probes"]) == 2
        for entry in pidx["probes"]:
            assert entry["forward_counts"] == {"conditioner": 1, "backbone": 1}
            assert [w["window_id"] for w in entry["windows"]] == \
                list(world.registration_for("D6").window_ids)


def test_mini_stage_local_verification_and_f_bitwise_replay():
    with tempfile.TemporaryDirectory() as td:
        world, plan, driver, summary, sdir, budget, finished = _mini(td)
        assert summary["local_complete_evidence"] is True, summary.get("failure")
        assert summary["call_matrix"]["problems"] == []
        assert summary["spike_gate_status"] == "not_evaluated"
        assert summary["final_verification_status"] == "pending_F"
        assert finished["seal"]["status"] == "sealed_within_budget"
        assert finished["return_coverage"]["covered"] is True
        doc = finished["summary_doc"]
        assert diag.local_report_problems(
            {k: v for k, v in doc.items() if k != "seal"}) == []
        # F: fresh modules + strict loads replay BOTH endpoints bitwise
        f_out = diag.run_f_verification(
            stage_specs=[{"stage_dir": sdir, "plan": plan,
                          "registration": world.registration_for("D1"),
                          "decoder": world.decoder, "loss_fn": world.loss_fn,
                          "identity": diag._synthetic_stage_identity(
                              plan, world.registration_for("D1"))}],
            out_dir=Path(td) / "f", budget=diag.budget_for_stage("F"),
            runtime_guard=diag.SyntheticRuntimeGuard(),
            build_modules=diag.build_modules_factory(world, plan),
            authorization=None)
        fdoc = f_out["summary_doc"]
        s0 = fdoc["stages"][0]
        assert s0["replay_enabled"] is True and s0["replay_passed"] is True
        assert s0["replay_problems"] == []
        # exactly two fresh probes: initial + last
        assert s0["fresh_counters"]["probe_calls"] == 2
        assert s0["fresh_counters"]["conditioner_forward_calls"] == 2
        assert fdoc["complete_evidence"] is True
        assert fdoc["tolerance_policy"] == "bitwise_no_tolerance"
        # the negative local result is verifiable too (mean gate fails at
        # 2 steps): mean_fit_passed is the honest false, not null
        assert fdoc["mean_fit_passed"] is False
        assert fdoc["verdict"] == "fit_insufficient"


def test_handoff_requires_gate_authorization_and_seal():
    with tempfile.TemporaryDirectory() as td:
        world, plan, driver, summary, sdir, budget, finished = _mini(
            td, steps=2, probe_every=1)
        # local mean gate fails at 2 steps -> no handoff even with D6 auth
        handoff = diag.resolve_handoff(sdir, plan, world.decoder,
                                       world.registration_for("D1"),
                                       {"includes": ["D6"]},
                                       diag.StageBudget("D6", 60.0, 5.0),
                                       expected_identity=diag._synthetic_stage_identity(
                                           plan, world.registration_for("D1")))
        assert handoff["handoff_ready"] is False
        assert any("d1_mean_fit_gate_not_passed" in p
                   for p in handoff["problems"])
        # a passing D1 without the D6 authorization is refused as well
        pidx = json.loads((sdir / "probe_index.json").read_text())
        assert pidx["schema"] == diag.PROBE_INDEX_SCHEMA
        auth_missing = diag.resolve_handoff(sdir, plan, world.decoder,
                                            world.registration_for("D1"),
                                            {"includes": []},
                                            diag.StageBudget("D6", 60.0, 5.0),
                                            expected_identity=diag._synthetic_stage_identity(
                                                plan, world.registration_for("D1")))
        assert any("authorization_does_not_include_d6" in p
                   for p in auth_missing["problems"])
        # the D1 report itself never publishes final-namespace fields
        doc = json.loads((sdir / "stage_report.json").read_text())
        assert doc["final_verification_status"] == "pending_F"
        assert "complete_evidence" not in doc
        assert "mean_fit_passed" not in doc


def test_budget_deadline_stops_the_driver_honestly():
    with tempfile.TemporaryDirectory() as td:
        world = diag.make_synthetic_world(seed=11, n_windows=1, d_model=16)
        plan = diag.FitPlan("D1", steps=4, probe_every=2, synthetic=True,
                            d_model=16, n_layers=1, n_heads=4, ffn_mult=2,
                            lr=0.02)
        view = world.view_for("D1")

        class TimeMachine:
            def __init__(self):
                self.t = 0.0

            def __call__(self):
                return self.t

        clock = TimeMachine()
        budget = diag.StageBudget(stage="D1", wall_cap_s=100.0, reserve_s=10.0,
                                  t0=0.0, clock=clock)
        trainer = world.build_trainer(plan, Path(td) / "d1", None)
        driver = diag.FitDriver(
            trainer=trainer, plan=plan, registration=view["registration"],
            loss_fn=world.loss_fn, decoder=world.decoder,
            probe_batch=view["probe_batch"], budget=budget,
            runtime_guard=diag.SyntheticRuntimeGuard(), identity={},
            stage_dir=Path(td) / "d1",
            training_gen=torch.Generator().manual_seed(plan.seed))
        # advance the clock past the work deadline before the loop starts:
        # the next unit check must refuse (no further forward/backward/step)
        clock.t = budget.work_deadline + 1.0
        summary = driver.run()
        assert summary["local_complete_evidence"] is False
        assert summary["local_mean_fit_passed"] is None
        assert summary["local_conclusion"] == "incomplete"
        failure = summary.get("failure") or {}
        assert "deadline" in failure.get("message", "")
        # the last completed unit boundary is visible in the counters
        assert summary["call_matrix"]["problems"] == ["stage_failed"]


# --------------------------------------------------------------------------
# the full §6.2/v0.3/v0.4 counterexample battery
# --------------------------------------------------------------------------

def test_counterexample_battery_all_caught():
    budget = diag.StageBudget(stage="battery", wall_cap_s=1800.0, reserve_s=60.0)
    results = diag.run_counterexample_battery(budget)
    assert set(results) == set(diag.COUNTEREXAMPLE_IDS)
    failures = {cid: r for cid, r in results.items() if not r["caught"]}
    assert failures == {}, json.dumps(failures, indent=1)[:2000]


def test_cli_output_and_reread_inside_return_coverage(tmp_path):
    """P1 (round-2 + round-4 review): the flushed stdout AND the CLI's own
    final summary re-read must both happen BEFORE the final coverage sample.

    The hook is PRECISE (round-4 review): it delays only reads of the
    acceptance summary issued by the CLI itself — identified by the caller
    frame, so the internal reads inside ``verify_return_coverage`` /
    ``_write_summary_sealed`` / ``write_acceptance_outputs`` are never
    delayed.  Under this exact hook the current CLI must exit 3 while the
    reverted ordering (coverage sample BEFORE the final read — the round-2
    bug) still exits 0, which is what this test locks out.
    """
    import importlib.util
    from unittest.mock import patch

    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" /
           "verify_w1_smallset_fit_diagnostic.py").read_text()
    # revert the round-3 ordering: final read AFTER the coverage sample
    reverted_src = src.replace(
        '    disk = json.loads(Path(paths["summary"]).read_text())\n'
        '    cov = diag.verify_return_coverage(Path(paths["summary"]), budget,\n'
        '                                      withdraw=dict(WITHDRAW_ON_EXCEED))\n',
        '    cov = diag.verify_return_coverage(Path(paths["summary"]), budget,\n'
        '                                      withdraw=dict(WITHDRAW_ON_EXCEED))\n')
    assert reverted_src != src
    reverted_src = reverted_src.replace(
        '        return 3\n'
        '    # P1 (round-2 review): the necessary final disk read happened BEFORE the\n'
        '    # coverage sample above (flushed stdout → re-read → sample); after the\n'
        '    # passing sample there is NO further I/O — the exit decision uses the\n'
        '    # already-read disk state.\n',
        '        return 3\n'
        '    disk = json.loads(Path(paths["summary"]).read_text())\n')
    assert reverted_src.index("cov = diag.verify_return_coverage") \
        < reverted_src.index('disk = json.loads')
    assert src.index('disk = json.loads') \
        < src.index("cov = diag.verify_return_coverage")

    fake_facts = {"frozen_plan_defaults_problems": [],
                  "all_counterexamples_caught": True,
                  "d1": {"local_complete_evidence": True,
                         "call_matrix_problems": [], "report_problems": []},
                  "f": {"replay_passed": True}}

    def run_cli(name: str, source: str) -> dict:
        cli_path = tmp_path / f"{name}.py"
        cli_path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, cli_path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cdir = tmp_path / name
        ct = cli._T0
        ticks = {"t": ct + 1.0}
        printed = {"done": False}

        def fake_time():
            return ticks["t"]

        real_read_text = Path.read_text
        internal_readers = ("verify_return_coverage", "_write_summary_sealed",
                            "write_acceptance_outputs")

        def precise_slow_read(path, *a, **kw):
            value = real_read_text(path, *a, **kw)
            if path.name == "acceptance_summary.json" and printed["done"]:
                caller = sys._getframe(1).f_code.co_name
                if caller not in internal_readers:
                    ticks["t"] = ct + 1000.0  # the CLI's own re-read blocks
            return value

        def flag_print(*a, **kw):
            printed["done"] = True

        with patch.object(sys, "argv", ["verify", "--output-root", str(cdir),
                                        "--wall-cap-s", "100",
                                        "--reserve-s", "10"]), \
             patch.object(cli.diag, "run_synthetic_acceptance",
                          lambda *a, **kw: json.loads(json.dumps(fake_facts))), \
             patch.object(time, "perf_counter", fake_time), \
             patch.object(Path, "read_text", precise_slow_read), \
             patch("builtins.print", flag_print):
            rc = cli.main()
        cs = json.loads((cdir / "attempt_001" /
                         "acceptance_summary.json").read_text())
        co = json.loads((cdir / "attempt_001" /
                         "return_observation.json").read_text())
        return {"rc": rc, "seal": cs["seal"]["status"], "covered": co["covered"],
                "measured": cs["seal"]["measured_wall_s_at_summary"]}

    current = run_cli("cli_current", src)
    assert current["rc"] == 3
    assert current["seal"] == "budget_exceeded_after_seal"
    assert current["covered"] is False
    # the withdrawal's registered cost covers the 1000s tail (costs only rise)
    assert current["measured"] >= 1000.0
    # the reverted ordering exits 0 under the SAME precise hook: the delay
    # lands on an unaccounted post-sample read.  If the CLI ever regressed to
    # that ordering, the assertions above would flip and this test fails.
    reverted = run_cli("cli_reverted", reverted_src)
    assert reverted["rc"] == 0
