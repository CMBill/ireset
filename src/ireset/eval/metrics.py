"""Evaluation metrics (plan section 10.1): contact F1 and friends."""

from __future__ import annotations

import numpy as np

from ireset.embodied.contact_sites import ContactSchedule

EFFECTORS = ("left_hand", "right_hand", "left_foot", "right_foot")


def contact_prf(
    predicted: ContactSchedule, reference: ContactSchedule, effector: str, T: int
) -> tuple[float, float, float]:
    """Frame-level precision/recall/F1 for one effector."""
    p = predicted.mask(effector, T)
    r = reference.mask(effector, T)
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def contact_f1_all(predicted: ContactSchedule, reference: ContactSchedule, T: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for eff in EFFECTORS:
        _, _, f1 = contact_prf(predicted, reference, eff, T)
        out[f"{eff}_f1"] = f1
    out["macro_f1"] = float(np.mean([out[f"{e}_f1"] for e in EFFECTORS]))
    return out
