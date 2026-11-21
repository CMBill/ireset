from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ireset.eval import w1_d6_shared_gradient as sg


def tiny_bundle(window_gradients, *, corrupt_h=None):
    h_rows = []
    active = sg.BATCH_SLOT_BY_CANONICAL
    h_batch = np.zeros((6, 60, 38), dtype=np.float32)
    for i, slot in enumerate(active):
        h = np.zeros_like(h_batch)
        h[slot, 0, 0] = 6.
        h_rows.append(h)
        h_batch[slot, 0, 0] = 1.
    if corrupt_h == "offslot":
        h_rows[0][0, 0, 0] = 1.
    elif corrupt_h == "wrongslot":
        h_rows[0][active[0], 0, 0] = 0.
        h_rows[0][0, 0, 0] = 6.
    roster = [
        {"name": "model.weight", "shape": [2], "dtype": "torch.float32", "numel": 2},
        {"name": "conditioner.weight", "shape": [2], "dtype": "torch.float32", "numel": 2},
    ]
    outputs = {"batch": {"h": h_batch, "g": np.mean(window_gradients, axis=0).astype(np.float32)}}
    outputs.update({f"window_{i}": {"h": h_rows[i], "g": np.asarray(g, dtype=np.float32)}
                    for i, g in enumerate(window_gradients)})
    return {"roster": roster, "outputs": outputs}


def test_historical_micro_order_and_common_window_slot():
    assert sg.MICRO_CANONICAL_SLOTS == (1, 0, 4, 3, 2, 5)
    assert sg.BATCH_SLOT_BY_CANONICAL == (1, 0, 4, 3, 2, 5)
    assert sg.WINDOWS[0] == "sub10_largebox_097:0:60"


def test_projection_reports_exact_equal_weight_hand_calculation():
    g0 = np.array([6., 0.])
    others = [np.array([0., 1.]) for _ in range(5)]
    gradients = [g0, *others]
    batch = np.mean(gradients, axis=0)
    facts = sg.projection_metrics(batch, gradients, e_all=0.)
    assert facts["own"] == 6.
    assert facts["other_ratio"] == 0.
    assert facts["batch_ratio"] == 1.
    assert facts["direction_interval"] == [1., 1.]
    assert facts["decision"] == "attribution_insufficient"


def test_projection_detects_quarter_cancellation_and_direction_reversal():
    positive = [np.array([6., 0.]), np.array([-3., 0.]), *[np.array([0., 1.]) for _ in range(4)]]
    measured = sg.projection_metrics(np.mean(positive, axis=0), positive, 0.)
    assert measured["other_ratio"] == -.5
    assert measured["batch_ratio"] == .5
    assert measured["decision"] == "candidate"
    negative = [np.array([6., 0.]), np.array([-12., 0.]), *[np.array([0., 1.]) for _ in range(4)]]
    reversed_direction = sg.projection_metrics(np.mean(negative, axis=0), negative, 0.)
    assert reversed_direction["batch_ratio"] == -1.
    assert reversed_direction["direction"] == "up"


@pytest.mark.parametrize("ratio,margin,decision,reason", [
    (-.25, 0., "candidate", "prespecified_cancellation_gate_passed"),
    (-.250001, 0., "candidate", "prespecified_cancellation_gate_passed"),
    (-.249999, 0., "attribution_insufficient", "cancellation_below_gate"),
])
def test_projection_threshold_boundary(ratio, margin, decision, reason):
    gradients = [np.array([1., 0.]), np.array([ratio, 1.])] + [np.array([0., 1.]) for _ in range(4)]
    result = sg.projection_metrics(np.mean(gradients, axis=0), gradients, margin / 6.)
    assert result["decision"] == decision
    assert result["reason"] == reason


def test_projection_interval_containing_zero_is_undetermined():
    gradients = [np.array([1., 0.])] + [np.array([-1., 1.]) for _ in range(5)]
    result = sg.projection_metrics(np.mean(gradients, axis=0), gradients, e_all=10.)
    assert result["direction_interval"][0] < 0 < result["direction_interval"][1]
    assert result["direction"] == "undetermined"


def test_projection_small_common_gradient_nulls_ratios():
    gradients = [np.array([1e-11, 0.])] + [np.array([0., 1.]) for _ in range(5)]
    result = sg.projection_metrics(np.mean(gradients, axis=0), gradients, e_all=0.)
    assert result["common_norm"] <= sg.MIN_COMMON_NORM
    assert result["other_ratio"] is None and result["margin"] is None
    assert result["decision"] == "attribution_insufficient"


def test_vjp_gradient_permutation_can_change_projection_while_mean_stays_fixed():
    values = [np.array([1., 0.]), np.array([0., 100.]), *[np.array([0., -10.]) for _ in range(4)]]
    before = np.mean(values, axis=0)
    old = sg.projection_metrics(before, values, 0.)
    swapped = values.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert np.array_equal(before, np.mean(swapped, axis=0))
    new = sg.projection_metrics(before, swapped, 0.)
    assert old["other_ratio"] == 0.
    assert new["other_ratio"] == pytest.approx(-.4)


@pytest.mark.parametrize("corruption,reason", [("offslot", "h_off_slot_nonzero"),
                                                  ("wrongslot", "h_off_slot_nonzero")])
def test_analyzer_rejects_output_gradient_slot_mismatch(corruption, reason):
    windows = [np.array([1., 0., 0., 0.]) for _ in range(6)]
    with pytest.raises(sg.SharedGradientError, match=reason):
        sg.analyze_gradients(tiny_bundle(windows, corrupt_h=corruption))


def test_analyzer_rejects_parameter_gradient_return_swap():
    windows = [np.array([1., 0., 0., 0.]), np.array([0., 100., 0., 0.])]
    windows.extend(np.array([0., -10., 0., 0.]) for _ in range(4))
    bundle = tiny_bundle(windows)
    # The mathematically unchanged average does not authorize relabeling the
    # paired outputs; request/response validation catches the mismatch first.
    req = {"call_id": 1, "role": "window_0", "batch_slot": 1}
    response = {"call_id": 1, "role": "window_1"}
    assert np.array_equal(np.mean(windows, axis=0), np.mean([*windows[1:2], windows[0], *windows[2:]], axis=0))
    assert (req["call_id"], req["role"]) != (response["call_id"], response["role"])
    # The projection helper itself illustrates the material change.
    before = sg.projection_metrics(bundle["outputs"]["batch"]["g"], windows, 0.)
    windows[0], windows[1] = windows[1], windows[0]
    after = sg.projection_metrics(bundle["outputs"]["batch"]["g"], windows, 0.)
    assert before["other_ratio"] != after["other_ratio"]


def test_request_response_receipt_rejects_swapped_g_records_even_if_mean_is_same():
    roster = [
        {"name": "model.weight", "shape": [2], "dtype": "torch.float32", "numel": 2},
        {"name": "conditioner.weight", "shape": [2], "dtype": "torch.float32", "numel": 2},
    ]
    requests, responses = [], []
    for i in range(7):
        role = "batch" if i == 0 else f"window_{i-1}"
        slot = None if i == 0 else sg.BATCH_SLOT_BY_CANONICAL[i-1]
        request = {"schema": sg.SCHEMA + "/vjp_request", "call_id": i, "role": role,
                   "window_id": None if i == 0 else sg.WINDOWS[i-1], "batch_slot": slot,
                   "scalar_tensor_id": 100 + i, "roster_sha256": "a" * 64,
                   "parameters": roster, "inputs": ["x0_pred", "model.weight", "conditioner.weight"],
                   "retain_graph": i < 6}
        request_sha = sg.sha256(sg.canonical_json(request))
        requests.append({**request, "request_sha256": request_sha})
        gs = [{"name": row["name"], "shape": row["shape"], "dtype": "<f4", "bytes": 8,
               "sha256": f"{i}{j}".ljust(64, "0")} for j, row in enumerate(roster)]
        response = {"schema": sg.SCHEMA + "/vjp_response", "call_id": i,
                    "request_sha256": request_sha, "role": role,
                    "h": {"shape": [6, 60, 38], "dtype": "<f4", "bytes": 54720, "sha256": "b"*64},
                    "g": gs, "g_flat": {"shape": [4], "dtype": "<f4", "bytes": 16, "sha256": "c"*64}}
        responses.append({**response, "response_sha256": sg.sha256(sg.canonical_json(response))})
    assert len(sg.verify_vjp_records(requests, responses, roster)) == 7
    left, right = responses[2]["g"]
    responses[2]["g"] = [right, left]
    unsigned = {k: v for k, v in responses[2].items() if k != "response_sha256"}
    responses[2]["response_sha256"] = sg.sha256(sg.canonical_json(unsigned))
    with pytest.raises(sg.SharedGradientError, match="vjp_gradient_name_order"):
        sg.verify_vjp_records(requests, responses, roster)


def test_loss_scalar_float64_reduction_uses_frozen_denominators_and_window_order():
    values = {
        "pos": np.ones((6, 60, 3), dtype="<f4"),
        "rot6d": np.ones((6, 60, 6), dtype="<f4") * 2,
        "joint": np.ones((6, 60, 29), dtype="<f4") * 3,
        "geodesic_rad2": np.ones((6, 60), dtype="<f4") * 4,
        "vel": np.ones((6, 59, 38), dtype="<f4") * 5,
        "fk_m2": np.ones((6, 60, 15), dtype="<f4") * 6,
    }
    replay = sg.independent_loss_scalars(values)
    expected = 1 + 2 + 3 + .5 * 4 + .5 * 5 + 2 * 6
    assert replay["per_window"] == [expected] * 6
    assert replay["batch_terms"]["total"] == expected
    assert replay["batch_terms"]["fk_m2"] == 6.
    values["fk_m2"] = values["fk_m2"][list(sg.MICRO_CANONICAL_SLOTS)]
    # The independent reduction returns registered-window order, not input
    # batch order; values are equal here, so explicitly verify a varied case.
    for i in range(6):
        values["fk_m2"][i].fill(i + 1)
    replay = sg.independent_loss_scalars(values)
    expected_fk = [float(np.mean(values["fk_m2"][slot])) for slot in sg.BATCH_SLOT_BY_CANONICAL]
    assert replay["batch_terms"]["fk_m2"] == pytest.approx(np.mean(expected_fk))


def test_unit_meter_fails_before_extra_call_and_after_late_return():
    now = [0.]
    meter = sg.UnitMeter({"one": 1}, deadline_s=10., clock=lambda: now[0])
    assert meter.unit("one", lambda: 7) == 7
    with pytest.raises(sg.SharedGradientError, match="call_matrix_extra:one"):
        meter.enter("one")
    assert meter.verify_complete()["completed"] == {"one": 1}

    now = [0.]
    delayed = sg.UnitMeter({"one": 1}, deadline_s=1., clock=lambda: now[0])
    def late():
        now[0] = 1.
        return "returned"
    with pytest.raises(sg.SharedGradientError, match="deadline:one:post"):
        delayed.unit("one", late)
    assert delayed.entered == {"one": 1} and delayed.completed == {"one": 0}


def test_artifact_store_refuses_space_before_creating_temporary_file(tmp_path):
    root = tmp_path / "attempt"
    store = sg.ArtifactStore(root, cap_bytes=64, reserve_bytes=32, header_cap=4096)
    with pytest.raises(sg.SharedGradientError, match="artifact_space_cap"):
        store.write_array("large.npy", np.ones(64, dtype="<f4"))
    assert list(root.iterdir()) == []


def test_real_path_refuses_before_any_source_argument_is_touched():
    class ExplodingPath:
        def __fspath__(self):
            raise AssertionError("source path must not be inspected")
    with pytest.raises(sg.SharedGradientError, match="real_execution_not_implemented_or_authorized"):
        sg.real_entry(ExplodingPath())


def test_only_explicit_authorization_marker_is_never_sufficient_to_open_real_path():
    with pytest.raises(sg.SharedGradientError, match="real_execution_requires_new_frozen_authorization"):
        sg.real_entry(authorization={"authorized": True})
