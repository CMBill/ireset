"""Windowed generation and stitching (plan section 7.4).

Windows of F=60 at stride 30 share one clip coordinate system (no per-window
world reset). In overlaps, positions and joint angles take a deterministic
linear crossfade; rotations blend by quaternion SLERP. The fused trajectory
is a *candidate*: the whole segment then goes through sequential repair.

Boundary diagnostics report joint/base velocity at window seams together
with interior percentiles, so seam failures cannot hide behind means.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ireset.kinematics.rotation_adapter import wxyz_to_xyzw, xyzw_to_wxyz


def window_starts(T: int, F: int, stride: int) -> list[int]:
    starts = list(range(0, max(T - F, 0) + 1, stride))
    if not starts:
        return [0]
    if starts[-1] != T - F and T > F:
        starts.append(T - F)
    return starts


def window_weight(i: int, n_windows: int, F: int, stride: int) -> np.ndarray:
    """Crossfade weight [F] for window ``i`` (ramp-in/out over one stride)."""
    w = np.ones(F)
    if i > 0:
        w[:stride] = np.linspace(0.0, 1.0, stride)
    if i < n_windows - 1 and F > stride:
        w[F - stride :] = np.linspace(1.0, 0.0, stride)
    return w


def fuse_windows(
    pos: list[np.ndarray],      # [Nw][F, 3]
    quat: list[np.ndarray],     # [Nw][F, 4] wxyz (clip frame)
    q: list[np.ndarray],        # [Nw][F, 29]
    starts: list[int],
    T: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Weighted fusion -> (pos [T,3], quat [T,4], q [T,29], diagnostics)."""
    F = pos[0].shape[0]
    Nw = len(pos)
    wsum = np.zeros(T)
    pos_out, q_out = np.zeros((T, 3)), np.zeros((T, 29))
    quat_out = np.zeros((T, 4))
    quat_acc: dict[int, list[tuple[np.ndarray, float]]] = {}

    for i, s in enumerate(starts):
        w = window_weight(i, Nw, F, stride)
        seg = slice(s, s + F)
        wsum[seg] += w
        pos_out[seg] += pos[i] * w[:, None]
        q_out[seg] += q[i] * w[:, None]
        for j, t in enumerate(range(s, s + F)):
            quat_acc.setdefault(t, []).append((quat[i][j], w[j]))

    for t, entries in quat_acc.items():
        if len(entries) == 1:
            quat_out[t] = entries[0][0]
        else:
            (qa, wa), (qb, wb) = entries[0], entries[-1]
            alpha = float(wb / max(wa + wb, 1e-12))
            s = Slerp([0.0, 1.0], Rot.from_quat(np.stack([wxyz_to_xyzw(qa), wxyz_to_xyzw(qb)])))
            quat_out[t] = xyzw_to_wxyz(s([alpha])[0].as_quat())

    pos_out /= np.maximum(wsum, 1e-12)[:, None]
    q_out /= np.maximum(wsum, 1e-12)[:, None]
    quat_out /= np.linalg.norm(quat_out, axis=1, keepdims=True)

    diag = boundary_diagnostics(pos_out, quat_out, q_out, starts, stride)
    return pos_out, quat_out, q_out, diag


def boundary_diagnostics(
    pos: np.ndarray, quat: np.ndarray, q: np.ndarray, starts: list[int], stride: int, fps: float = 30.0
) -> dict:
    """Velocity spikes at window seams vs interior (p50/p95/max, rad/s & m/s)."""
    T = q.shape[0]
    if T < 2:
        return {"seam_joint_speed": [], "interior_joint_speed_p50": 0.0}
    jq = np.linalg.norm(np.diff(q, axis=0), axis=1) * fps
    jp = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    seams = {t for s in starts[1:] for t in (s - 1, s)}
    idx = np.array(sorted(seams & set(range(T - 1))), dtype=int)
    interior = np.setdiff1d(np.arange(T - 1), idx)
    pct = lambda a, p: float(np.percentile(a, p)) if len(a) else 0.0
    return {
        "n_seam_steps": int(len(idx)),
        "joint_speed_seam_max": float(jq[idx].max()) if len(idx) else 0.0,
        "joint_speed_seam_p95": pct(jq[idx], 95),
        "joint_speed_interior_p50": pct(jq[interior], 50),
        "joint_speed_interior_p95": pct(jq[interior], 95),
        "joint_speed_interior_max": float(jq[interior].max()) if len(interior) else 0.0,
        "root_speed_seam_max": float(jp[idx].max()) if len(idx) else 0.0,
        "root_speed_interior_p95": pct(jp[interior], 95),
    }
