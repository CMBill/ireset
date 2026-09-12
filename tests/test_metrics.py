"""Contact F1 metrics tests: undefinedness policy (Fix 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.embodied.contact_sites import ContactInterval, ContactSchedule
from ireset.eval.metrics import contact_f1_all, contact_f1_success_weighted, contact_prf


def sched(*specs: tuple[str, int, int]) -> ContactSchedule:
    s = ContactSchedule()
    for eff, a, b in specs:
        s.intervals.append(ContactInterval(eff, a, b, "object"))
    return s


def test_double_empty_returns_none() -> None:
    """Neither side claims contact: no evidence, never a free 1.0."""
    T = 10
    assert contact_prf(ContactSchedule(), ContactSchedule(), "left_hand", T) is None


def test_macro_skips_undefined() -> None:
    T = 10
    pred = sched(("left_hand", 0, 5))
    ref = sched(("left_hand", 0, 5))          # perfect on the only defined effector
    out = contact_f1_all(pred, ref, T)
    assert out["left_hand"]["f1"] == 1.0
    assert out["right_hand"] is None
    assert out["n_defined"] == 1
    assert out["macro_f1"] == 1.0


def test_all_undefined_macro_is_none() -> None:
    out = contact_f1_all(ContactSchedule(), ContactSchedule(), 10)
    assert out["macro_f1"] is None and out["n_defined"] == 0


def test_one_sided_empty_scores_zero() -> None:
    """Reference has contact, prediction none: F1 must be 0.0, not 1.0."""
    T = 10
    pred = ContactSchedule()
    ref = sched(("left_hand", 2, 8))
    prf = contact_prf(pred, ref, "left_hand", T)
    assert prf is not None and prf[2] == 0.0


def test_partial_overlap() -> None:
    T = 10
    pred = sched(("left_hand", 0, 4))
    ref = sched(("left_hand", 2, 8))
    p, r, f1 = contact_prf(pred, ref, "left_hand", T)
    assert p == 0.5 and abs(r - 1 / 3) < 1e-12 and abs(f1 - 0.4) < 1e-12


def test_success_weighted_scoring() -> None:
    rows = [0.9, None, 0.7]
    assert abs(contact_f1_success_weighted(rows) - (0.9 + 0.0 + 0.7) / 3) < 1e-12
    assert contact_f1_success_weighted([None, None]) == 0.0
