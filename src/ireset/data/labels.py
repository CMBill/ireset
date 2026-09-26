"""Graded teacher labels: per-frame masks, interval ids and count metrics.

Label-quality plan (m2_m3_label_quality_train24_plan.md) stages A/B: every
report number and every training-side supervision decision derives from the
SAME per-frame series, so a certified-total can never stand in for the
certified-and-in-reference intersection again (sub10 left: 3 certified
frames, 0 of them inside the reference grasp window).

Frozen protocol (``configs/labels/graded_labels_v1.yaml``):

- ``hand_proximity_mask[t, side]``  = ref_contact(t) AND dist(t) < hand_enter_m
  (strict ``<`` -- the keep criterion's operator, not ``<=``);
- ``hand_contact_supervision_mask[t, side]`` = ref_contact(t) AND certified(t);
- ``hand_supervision_interval_id[t, side]`` numbers the CONTINUOUS pieces of
  that intersection, restarting at every reference gap AND every certified
  interval boundary (-1 outside), so adjacent frames form a grasp-timing pair
  only when both are valid and share an id.

Certification itself stays the validator's (quasi-static entry gates); these
functions only consume its per-frame evidence. Zero reference frames is
UNDEFINED (None) for ratios; zero certification is an explicit 0.
"""

from __future__ import annotations

import numpy as np

#: protocol identity stamped into label NPZs and quality manifests; the
#: dataset entry refuses manifests carrying anything else
LABEL_PROTOCOL_VERSION = "graded_labels_v1"

#: fixed column order of every [T, 2] label array
LABEL_SIDES = ("l", "r")
LABEL_EFFECTORS = ("left_hand", "right_hand")


def continuous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs of True in a 1-D boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(int)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def intersection_interval_ids(certified_intervals: list[tuple[int, int]],
                              ref_mask: np.ndarray, T: int) -> np.ndarray:
    """Ids of the continuous pieces of certified AND reference frames.

    Numbering restarts at every reference gap and at every certified
    interval boundary: pieces from different certified intervals never
    share an id even when they touch. Frames outside get -1.
    """
    ids = np.full(T, -1, dtype=np.int64)
    ref = np.asarray(ref_mask, dtype=bool)
    nxt = 0
    for a, b in sorted(certified_intervals):
        for s, e in continuous_runs(ref[a:b]):
            ids[a + s:a + e] = nxt
            nxt += 1
    return ids


def side_label_arrays(dist: np.ndarray, ref_mask: np.ndarray,
                      certified_intervals: list[tuple[int, int]],
                      enter_m: float) -> dict[str, np.ndarray]:
    """Per-side proximity/supervision masks + interval ids from series.

    ``dist`` is the per-frame closest hand-geom-to-box distance of this
    side; proximity uses the keep criterion's strict ``<``.
    """
    T = int(ref_mask.shape[0])
    ref = np.asarray(ref_mask, dtype=bool)
    prox = ref & (np.asarray(dist, dtype=float) < enter_m)
    cert = np.zeros(T, dtype=bool)
    for a, b in certified_intervals:
        cert[a:b] = True
    sup = cert & ref
    ids = intersection_interval_ids(certified_intervals, ref, T)
    return {"proximity": prox, "supervision": sup, "interval_id": ids,
            "certified": cert}


def longest_run_frames(mask: np.ndarray) -> int:
    runs = continuous_runs(mask)
    return max((e - s for s, e in runs), default=0)


def crossings_within_intervals(within_mask: np.ndarray,
                               intervals: list[tuple[int, int]]) -> int:
    """Threshold crossings counted ONLY between adjacent frames inside the
    SAME interval -- a gap between two reference intervals is not a crossing
    (the plan bans substituting concatenated-series crossings)."""
    within = np.asarray(within_mask, dtype=bool)
    n = 0
    for a, b in intervals:
        if b - a > 1:
            n += int(np.count_nonzero(within[a:b - 1] != within[a + 1:b]))
    return n


def longest_detachment_run(within_mask: np.ndarray,
                           intervals: list[tuple[int, int]]) -> int:
    """Longest run of consecutive NOT-within frames inside ONE interval."""
    within = np.asarray(within_mask, dtype=bool)
    best = 0
    for a, b in intervals:
        best = max(best, longest_run_frames(~within[a:b]))
    return best


def grasp_pair_valid(interval_id: np.ndarray) -> np.ndarray:
    """[T] bool: adjacent frames t, t+1 may form a grasp-timing pair.

    True at t only when BOTH frames carry the same non-negative interval id
    -- a reference gap or a certified-interval boundary breaks the pair
    (plan section 3.2). Position 0 is always False (no predecessor).
    """
    ids = np.asarray(interval_id)
    ok = np.zeros(ids.shape[0], dtype=bool)
    if ids.shape[0] > 1:
        a, b = ids[:-1], ids[1:]
        ok[:-1] = (a >= 0) & (a == b)
    return ok


def both_hands_metrics(per_side: dict[str, np.ndarray]) -> dict:
    """Simultaneous-certification counts, listed separately from per-side
    numbers (never substitutable for them)."""
    cert_l, cert_r = per_side["l"]["certified"], per_side["r"]["certified"]
    sup_l, sup_r = per_side["l"]["supervision"], per_side["r"]["supervision"]
    both_cert = cert_l & cert_r
    both_sup = sup_l & sup_r
    runs = continuous_runs(both_sup)
    return {
        "both_certified_frames": int(both_cert.sum()),
        "both_supervision_frames": int(both_sup.sum()),
        "both_longest_supervision_run_frames": max(
            (e - s for s, e in runs), default=0),
    }
