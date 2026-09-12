"""Evaluation metrics (plan section 10.1): contact F1 and friends.

Undefinedness policy: when neither side claims contact for an effector
(tp = fp = fn = 0) the score carries no evidence. Scoring it 1.0 would
reward a scheduler for producing no intervals at all; scoring it 0 would
punish a correct "genuinely no contact". Such effectors are excluded from
the macro average and reported via ``n_defined``.
"""

from __future__ import annotations

import numpy as np

from ireset.embodied.contact_sites import ContactSchedule

EFFECTORS = ("left_hand", "right_hand", "left_foot", "right_foot")


def contact_prf(
    predicted: ContactSchedule, reference: ContactSchedule, effector: str, T: int
) -> tuple[float, float, float] | None:
    """Frame-level precision/recall/F1; None when neither side claims contact."""
    p = predicted.mask(effector, T)
    r = reference.mask(effector, T)
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def contact_f1_all(predicted: ContactSchedule, reference: ContactSchedule, T: int) -> dict:
    """Per-effector P/R/F1 (None when undefined), ``n_defined`` and macro F1.

    ``macro_f1`` is None when every effector is undefined (no evidence at
    all); it is never silently filled with 0 or 1.
    """
    out: dict = {}
    defined: list[float] = []
    for eff in EFFECTORS:
        prf = contact_prf(predicted, reference, eff, T)
        if prf is None:
            out[eff] = None
        else:
            out[eff] = {"precision": prf[0], "recall": prf[1], "f1": prf[2]}
            defined.append(prf[2])
    out["n_defined"] = len(defined)
    out["macro_f1"] = float(np.mean(defined)) if defined else None
    return out


def contact_f1_success_weighted(rows: list[float | None]) -> float:
    """Demo-macro F1 over the FULL test set; rejected/undefined clips score 0.

    Plan section 10.1: report both the raw value on accepted samples and the
    success-weighted demo-macro value over the whole test set — the relative
    gates of section 1.3 use the latter.
    """
    return float(np.mean([0.0 if r is None else r for r in rows]))
