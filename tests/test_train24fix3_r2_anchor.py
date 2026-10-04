"""train24fix3 continuation unit 5: the R2 fixed-interval regen hand
anchors (plan sections 7.4/7.6).

Pinned here:

- the registry: R2 = R1 + ``regen_anchor_policy: fixed_interval``; the
  closed field set rejects unknown policies and any anchor policy without
  the R1 recovery policy on the norm_cut mode;
- the state semantics (plan 7.6 items 2/3): the interval's FIRST fully
  adopted C2-active frame freezes its box-frame coordinates; the anchor is
  numerically invariant within the interval (later adoptions never
  re-derive it); the interval's first C2-active frame itself solves
  without an anchor (no out-of-interval borrowing); t=0 never anchors; a
  release fires on interval identity change -- including the renumbering
  under an unchanged hold mask -- and a later interval re-establishes;
  left/right are independent; only ``note_accepted`` writes state;
- the wiring (item 4): ``_frame_ctx`` passes the interval-fixed anchor
  into the existing ``FrameContext.hand_anchor`` for R2/R3 only; the ctx
  arrays freeze it (the audit/replay path sees the same rows); on a real
  truncated segment the anchor appears in the ctx of frames AFTER the
  establishing adoption, identical across those frames, with the run
  diagnostics carrying the established/released events;
- purity: no anchor policy (R1/S1/N-series/standard) builds NO state --
  the run diagnostics carry ``policy: None`` and zero anchored frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from ireset.embodied.contact_sites import (  # noqa: E402
    ContactInterval,
    ContactSchedule,
)
from ireset.repair.context import RegenHandAnchorState  # noqa: E402
from ireset.repair.profiles import get_profile, load_profiles  # noqa: E402

R2 = get_profile("R2")


def _sched(*specs) -> ContactSchedule:
    """spec: (effector, start, end) -- enumerated in order, so ADJACENT
    intervals of one effector carry DIFFERENT ids (the renumbering-under-
    unchanged-hold case)."""
    s = ContactSchedule()
    for eff, a, b in specs:
        s.intervals.append(ContactInterval(eff, a, b, "object"))
    return s


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_r2_registry_and_closed_fields(tmp_path):
    import yaml

    from ireset.repair.profiles import PROFILES_YAML

    ps = load_profiles()
    r2, r1 = ps["R2"], ps["R1"]
    assert r2.regen_anchor_policy == "fixed_interval"
    assert r2.recovery_policy == r1.recovery_policy == "feasibility_first"
    assert r2.c2_norm_speed_mps == r1.c2_norm_speed_mps
    assert r1.regen_anchor_policy is None
    assert all(p.regen_anchor_policy is None for n, p in ps.items()
               if n not in ("R2", "R3"))
    # R3 = R2 + the registered S2 bound; that numeric bound is its ONLY
    # delta (the finite matrix's last candidate)
    r3 = ps["R3"]
    assert r3.regen_anchor_policy == "fixed_interval"
    assert r3.c2_norm_speed_mps == 0.049 == ps["S2"].c2_norm_speed_mps
    assert r3.solver == r2.solver and r3.c2_slack_penalty == r2.c2_slack_penalty

    base = yaml.safe_load(PROFILES_YAML.read_text())["profiles"]
    cases = {
        "bad_value": {"base": "R2", "regen_anchor_policy": "per_frame"},
        "without_recovery": {
            "base": "R2", "recovery_policy": None,
            "regen_anchor_policy": "fixed_interval"},
        "axis_mode": {
            "base": "R2", "c2_mode": "axis", "c2_norm_speed_mps": None,
            "regen_anchor_policy": "fixed_interval"},
    }
    for name, extra in cases.items():
        doc = dict(base)
        doc["X"] = extra
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump({"schema": "train24fix3_profiles_v1",
                                     "profiles": doc}))
        with pytest.raises(ValueError):
            load_profiles(p)


# ---------------------------------------------------------------------------
# state semantics (deterministic, no solver)
# ---------------------------------------------------------------------------


def test_anchor_lifecycle_invariant_within_interval_release_on_renumber():
    # left hand: two ADJACENT intervals (same hold, different ids); right
    # hand: one long interval -- sides are independent
    sched = _sched(("left_hand", 2, 6), ("left_hand", 6, 10),
                   ("right_hand", 2, 10))
    st = RegenHandAnchorState(sched, 12)
    A = np.full((3, 3), 0.11)
    B = np.full((3, 3), 0.22)

    # interval 0, first C2-active frame: NO anchor yet (nothing borrowed)
    assert st.frame_anchor("l", 2, c2_enabled=True) is None
    st.note_accepted("l", 2, True, A)                 # establishes from A
    ev = [e for e in st.events if e["event"] == "established"]
    assert ev == [{"t": 2, "side": "l", "event": "established",
                   "reason": "first_adopted_c2_frame", "interval": 0,
                   "source_t": 2}]
    # later frames of the interval see the SAME frozen array; a later
    # adoption with DIFFERENT coordinates must not re-derive it
    got = st.frame_anchor("l", 3, c2_enabled=True)
    assert np.array_equal(got, A)
    st.note_accepted("l", 3, True, B)
    assert np.array_equal(st.frame_anchor("l", 4, c2_enabled=True), A)
    # a C2-off frame sees no anchor but does not release
    assert st.frame_anchor("l", 4, c2_enabled=False) is None
    assert np.array_equal(st.frame_anchor("l", 5, c2_enabled=True), A)
    # outside every interval: nothing (and no state change)
    assert st.frame_anchor("l", 10, c2_enabled=True) is None
    # interval renumbering under the SAME hold: release + re-establish
    assert st.frame_anchor("l", 6, c2_enabled=True) is None
    assert any(e["event"] == "released" and e["reason"] == "interval_changed"
               for e in st.events)
    st.note_accepted("l", 6, True, B)
    assert np.array_equal(st.frame_anchor("l", 7, c2_enabled=True), B)
    # the right side ran its own independent lifecycle
    assert st.frame_anchor("r", 2, c2_enabled=True) is None
    st.note_accepted("r", 4, True, A)
    assert np.array_equal(st.frame_anchor("r", 8, c2_enabled=True), A)
    assert st.source_t == {"left_hand": 6, "right_hand": 4}


def test_anchor_t0_and_unadopted_first_frame():
    sched = _sched(("left_hand", 0, 5))
    st = RegenHandAnchorState(sched, 5)
    # t=0 never anchors -- not even if (contrary to C2 semantics) a caller
    # offered an adoption there
    st.note_accepted("l", 0, True, np.ones((3, 3)))
    assert st.frame_anchor("l", 0, c2_enabled=True) is None
    assert st.anchor["left_hand"] is None
    # a C2-inactive adoption establishes nothing
    st.note_accepted("l", 2, False, np.ones((3, 3)))
    assert st.anchor["left_hand"] is None
    # an adoption outside every interval establishes nothing
    sched2 = _sched(("left_hand", 3, 5))
    st2 = RegenHandAnchorState(sched2, 5)
    st2.note_accepted("l", 1, True, np.ones((3, 3)))
    assert st2.anchor["left_hand"] is None


# ---------------------------------------------------------------------------
# wiring: _frame_ctx hand-off, ctx freezing, a real truncated segment
# ---------------------------------------------------------------------------


def _pipe(profile):
    from test_hand_speed_row import REGEN_C2
    from ireset.repair.pipeline import RepairPipeline

    return RepairPipeline(cfg=REGEN_C2, c2_profile=profile)


def _cond(T):
    from ireset.repair.context import ReferenceCondition

    box = np.tile(np.array([0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]), (T, 1))
    return ReferenceCondition(np.zeros((T, 52, 3)), box,
                              _sched(("left_hand", 0, T)))


def _pred(T):
    """Zero scene with VALID identity quaternions (root AND box)."""
    p = np.zeros((T, 43))
    p[:, 3:7] = (1.0, 0.0, 0.0, 0.0)
    p[:, 39:43] = (1.0, 0.0, 0.0, 0.0)
    return p


def test_frame_ctx_regen_anchor_replaces_contact_read():
    from ireset.repair.context import ContactStateMachine

    pipe = _pipe(R2)
    T = 3
    contacts = ContactStateMachine(_sched(("left_hand", 0, 3)), T)
    cond = _cond(T)
    ctx = pipe._frame_ctx(
        1, _pred(T), cond, contacts,
        {}, {}, None, regen_hand_anchor={"l": np.full((3, 3), 0.5),
                                         "r": None})
    assert np.array_equal(ctx.hand_anchor["l"], np.full((3, 3), 0.5))
    assert ctx.hand_anchor["r"] is None
    # without the R2 dict the original contact-machine read is untouched
    ctx2 = pipe._frame_ctx(1, _pred(T), cond,
                           contacts, {}, {}, None)
    assert ctx2.hand_anchor == {"l": None, "r": None}


def test_ctx_arrays_freeze_the_regen_anchor():
    from train24fix2_audit import context_arrays
    from ireset.repair.context import ContactStateMachine

    pipe = _pipe(R2)
    T = 2
    contacts = ContactStateMachine(_sched(("left_hand", 0, 2)), T)
    anchor = np.arange(9, dtype=float).reshape(3, 3)
    ctx = pipe._frame_ctx(1, _pred(T), _cond(T),
                          contacts, {}, {}, None,
                          regen_hand_anchor={"l": anchor, "r": None})
    arr = context_arrays(ctx)
    assert "ctx__hand_anchor__l" in arr
    assert np.array_equal(arr["ctx__hand_anchor__l"], anchor)
    assert "ctx__hand_anchor__r" not in arr          # None stays unset


def test_r2_real_truncated_segment_establishes_and_holds():
    """A real 4-frame sub3 prefix (deep-contact opening, C2 enabled from
    t=1 on both sides): the anchor freezes at the first adopted C2-active
    frame and every LATER frame's ctx carries the IDENTICAL array; the run
    diagnostics record the policy and the established events."""
    from ireset.data.teacher import load_teacher_npz
    from ireset.embodied.contact_sites import infer_reference_schedule
    from ireset.paths import TEACHER_TRAJ_DIR
    from ireset.repair.config_io import load_mode_configs
    from ireset.repair.context import ReferenceCondition
    from test_hand_speed_row import REGEN_C2

    T = 4
    scene, human, _fps = load_teacher_npz(
        TEACHER_TRAJ_DIR / "sub3_largebox_000_original.npz")
    scene, human = scene[:T].copy(), human[:T].copy()
    cond = ReferenceCondition(human, scene[:, 36:43],
                              infer_reference_schedule(human, scene[:, 36:43]))
    rcfg, vcfg = load_mode_configs("strict")
    pipe_r2 = _pipe(R2)
    pipe_r2.validator.cfg = vcfg
    frames: list[dict] = []
    pipe_r2.snapshot = lambda kind, data: (
        frames.append({"kind": kind, "t": data.get("t"),
                       "anchor": {s: (None if (data["ctx"].hand_anchor.get(s)
                                               is None)
                                       else np.array(data["ctx"].hand_anchor[s],
                                                     dtype=float))
                                  for s in ("l", "r")},
                       "enabled": dict(data["ctx"].c2_hand_enabled)})
        if kind == "frame" else None)
    res = pipe_r2.repair_trajectory(scene, cond)
    diag = res.diagnostics["regen_anchor"]
    assert diag["policy"] == "fixed_interval"
    assert diag["established"], "the deep-contact opening must anchor early"
    est = diag["established"][0]
    assert est["source_t"] >= 1 and est["side"] in ("l", "r")
    # every ctx AFTER the establishing adoption carries the IDENTICAL array
    anchored = [f for f in frames if f["anchor"]["l"] is not None
                or f["anchor"]["r"] is not None]
    assert anchored
    for side in ("l", "r"):
        vals = [f["anchor"][side] for f in anchored
                if f["anchor"][side] is not None]
        if vals:
            assert all(np.array_equal(v, vals[0]) for v in vals), \
                "the interval anchor must be numerically invariant"
    assert diag["frames_anchored"] == {
        "l": sum(1 for f in frames if f["anchor"]["l"] is not None),
        "r": sum(1 for f in frames if f["anchor"]["r"] is not None)}

    # purity: the SAME prefix under R1 (no anchor policy) builds no state
    pipe_r1 = _pipe(get_profile("R1"))
    pipe_r1.validator.cfg = vcfg
    res1 = pipe_r1.repair_trajectory(scene, cond)
    d1 = res1.diagnostics["regen_anchor"]
    assert d1["policy"] is None and d1["frames_anchored"] == {"l": 0, "r": 0}
    assert d1["established"] == [] and d1["released"] == []
