"""Path-consistent all-contributor rotation fusion (protocol v2).

The legacy stitcher keeps positions/joints but fuses rotations from only the
first and last positive-weight slot, using a per-frame shortest-arc SLERP. That
drops a middle contributor in three-window overlaps and can switch branches
when consecutive windows lie on opposite sides of a relative pi rotation.

This module implements one preregistered candidate,
``w1_rotation_path_nlerp_v2``:

1. lift each window's quaternion representation continuously (deterministic
   first sign, then negate whenever the adjacent dot product is negative);
2. activate each later window at its first positive-weight frame by choosing
   one overall sign against the previous *fused* frame, then keep it fixed;
3. blend every positive-weight contributor with normalized linear quaternion
   interpolation, rejecting ill-conditioned sums.

Positions and joints are delegated bit-for-bit to the original
:func:`ireset.model.stitch.fuse_windows`.  This module never mutates the
caller's decoded arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ireset.model.stitch import fuse_windows, window_weight

PROTOCOL = "w1_rotation_path_nlerp_v2"
QUAT_NORM_TOL = 1e-5
PATH_AMBIGUITY_DOT = 1e-6
MIN_BLEND_NORM = 0.5
ANGULAR_SPEED_LIMIT_RADPS = 7.0
FRAME_DT_DEFAULT = 1.0 / 30.0


class RotationFusionRejected(ValueError):
    """The candidate cannot be computed for this segment.

    ``prefix`` holds the already fused frames before ``frame`` (arrays with
    length ``frame``), so a caller can persist honest partial evidence.  This
    is not a quality judgement: rejections here mean the registered math
    contract cannot produce a complete output.
    """

    def __init__(self, code: str, *, frame: int | None = None,
                 window: int | None = None, local_frame: int | None = None,
                 evidence: dict | None = None,
                 prefix: dict | None = None,
                 diagnostics: dict | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.frame = frame
        self.window = window
        self.local_frame = local_frame
        self.evidence = dict(evidence or {})
        self.prefix = prefix
        self.diagnostics = dict(diagnostics or {})


@dataclass
class RotationFusionResult:
    pos: np.ndarray
    quat: np.ndarray
    joints: np.ndarray
    lifted_quat: np.ndarray
    weights: np.ndarray
    diagnostics: dict
    numerical_valid: bool = True
    fusion_math_valid: bool = True
    fusion_temporal_pass: bool = True
    consumer_eligible: bool = True
    protocol: str = PROTOCOL
    extra: dict = field(default_factory=dict)


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RotationFusionRejected(f"invalid_{name}", evidence={name: repr(value)})
    value = int(value)
    if value < 1:
        raise RotationFusionRejected(f"invalid_{name}", evidence={name: value})
    return value


def _finite_scalar(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RotationFusionRejected(f"invalid_{name}", evidence={name: repr(value)})
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise RotationFusionRejected(f"invalid_{name}", evidence={name: value})
    return value


def _starts_vector(starts) -> np.ndarray:
    raw = np.asarray(starts)
    if raw.ndim != 1 or raw.size == 0:
        raise RotationFusionRejected("invalid_starts", evidence={"shape": list(raw.shape)})
    if raw.dtype == bool or not np.issubdtype(raw.dtype, np.integer):
        raise RotationFusionRejected("invalid_starts", evidence={"dtype": str(raw.dtype)})
    values = [int(v) for v in raw.tolist()]
    if any(v < 0 for v in values):
        raise RotationFusionRejected("invalid_starts", evidence={"starts": values})
    if len(set(values)) != len(values) or any(b <= a for a, b in zip(values, values[1:])):
        raise RotationFusionRejected("invalid_starts", evidence={"starts": values})
    return np.asarray(values, dtype=np.int64)


def _array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise RotationFusionRejected(f"invalid_{name}_shape",
                                     evidence={"shape": list(arr.shape),
                                               "expected": list(shape)})
    if not np.issubdtype(arr.dtype, np.floating):
        raise RotationFusionRejected(f"invalid_{name}_dtype",
                                     evidence={"dtype": str(arr.dtype)})
    if not np.isfinite(arr).all():
        raise RotationFusionRejected(f"nonfinite_{name}")
    return arr


def _copy_arrays(pos, quat, joints, starts, T: int, stride: int):
    F = int(np.asarray(pos).shape[1]) if np.asarray(pos).ndim == 3 else -1
    if F < 1:
        raise RotationFusionRejected("invalid_pos_shape")
    starts_arr = _starts_vector(starts)
    N = len(starts_arr)
    if starts_arr[0] != 0:
        raise RotationFusionRejected("starts_do_not_begin_at_zero",
                                     evidence={"starts": starts_arr.tolist()})
    if starts_arr[-1] != T - F:
        raise RotationFusionRejected("starts_do_not_cover_last_frame",
                                     evidence={"starts": starts_arr.tolist(), "T": T, "F": F})
    if any(b > a + F for a, b in zip(starts_arr.tolist(), starts_arr.tolist()[1:])):
        raise RotationFusionRejected("uncovered_frame",
                                     evidence={"starts": starts_arr.tolist(), "F": F})
    pos_arr = _array("pos", pos, (N, F, 3))
    quat_arr = _array("quat", quat, (N, F, 4))
    joints_arr = _array("joints", joints, (N, F, 29))
    norms = np.linalg.norm(quat_arr.astype(np.float64), axis=-1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise RotationFusionRejected("degenerate_quaternion")
    if np.any(np.abs(norms - 1.0) > QUAT_NORM_TOL):
        raise RotationFusionRejected("quaternion_norm_out_of_tolerance",
                                     evidence={"max_abs_error": float(np.abs(norms - 1).max())})
    return pos_arr, quat_arr, joints_arr, starts_arr, F, N


def lift_window_signs(quat: np.ndarray) -> tuple[np.ndarray, dict]:
    """Continuous representation lift for one window.

    The first sign is deterministic (largest absolute component positive;
    ties resolve w/x/y/z).  Later frames are negated when the adjacent dot is
    negative.  An ambiguous adjacent pair (``abs(dot) <= 1e-6``) is reported
    instead of guessed.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RotationFusionRejected("degenerate_quaternion")
    q /= norms[:, None]
    first_index = int(np.argmax(np.abs(q[0])))
    initial_sign = 1.0 if q[0, first_index] >= 0.0 else -1.0
    q[0] *= initial_sign
    flips: list[int] = []
    first_ambiguous = None
    for j in range(1, len(q)):
        dot = float(np.dot(q[j - 1], q[j]))
        if abs(dot) <= PATH_AMBIGUITY_DOT:
            first_ambiguous = {"local_frame": int(j), "dot": dot}
            break
        if dot < 0:
            q[j] = -q[j]
            flips.append(int(j))
    info = {
        "initial_sign": float(initial_sign),
        "flip_frames": flips,
        "ambiguous": first_ambiguous,
    }
    if first_ambiguous is not None:
        return q, info
    return q, info


def angular_speed_radps(quat: np.ndarray, frame_dt: float) -> np.ndarray:
    """SO(3) angular speed for adjacent frames; entry k is t=k -> k+1."""
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternion array must be [T,4]")
    if len(q) < 2:
        return np.zeros(0, dtype=np.float64)
    rot = Rot.from_quat(q[:, [1, 2, 3, 0]])
    relative = rot[:-1].inv() * rot[1:]
    return relative.magnitude() / float(frame_dt)


def _seam_indices(starts: np.ndarray, T: int) -> np.ndarray:
    seams = {int(s) - 1 for s in starts[1:]} | {int(s) for s in starts[1:]}
    return np.asarray(sorted(t for t in seams if 0 <= t < T - 1), dtype=np.int64)


def _worst(series: np.ndarray, indices: np.ndarray | None = None):
    if indices is not None:
        if not len(indices):
            return None, None
        local = series[indices]
        pos = int(np.argmax(local))
        return float(local[pos]), int(indices[pos]) + 1
    if not len(series):
        return None, None
    pos = int(np.argmax(series))
    return float(series[pos]), pos + 1


def _json_index(values) -> list:
    return [int(v) for v in values]


def _prefix(pos, quat, joints, length: int) -> dict:
    return {
        "pos": np.array(pos[:length], copy=True),
        "quat": np.array(quat[:length], copy=True),
        "joints": np.array(joints[:length], copy=True),
        "length": int(length),
    }


def fuse_windows_v2(
    pos,
    quat,
    joints,
    starts,
    T: int,
    stride: int,
    frame_dt: float = FRAME_DT_DEFAULT,
    *,
    angular_speed_limit_radps: float = ANGULAR_SPEED_LIMIT_RADPS,
) -> RotationFusionResult:
    """Fuse decoded windows under the registered v2 rotation path contract.

    ``pos``/``quat``/``joints`` are ``[N,F,3]/[N,F,4]/[N,F,29]`` decoded
    windows in the clip frame, with ``quat`` in wxyz order.  Returns arrays for
    the full segment.  A temporal violation is reported through
    ``fusion_temporal_pass=False`` and ``consumer_eligible=False``; the raw
    candidate is still returned.  A path/blend degeneracy raises
    :class:`RotationFusionRejected` with the computed prefix.
    """
    T = _positive_int("T", T)
    stride = _positive_int("stride", stride)
    frame_dt = _finite_scalar("frame_dt", frame_dt)
    limit = _finite_scalar("angular_speed_limit_radps", angular_speed_limit_radps)
    pos_arr, quat_arr, joints_arr, starts_arr, F, N = _copy_arrays(
        pos, quat, joints, starts, T, stride)
    if T < F:
        raise RotationFusionRejected("clip_shorter_than_window",
                                     evidence={"T": T, "F": F})
    if stride > F:
        raise RotationFusionRejected("stride_exceeds_window",
                                     evidence={"stride": stride, "F": F})

    weights = np.stack([window_weight(i, N, F, stride) for i in range(N)])  # [N,F]
    if not np.isfinite(weights).all():
        raise RotationFusionRejected("nonfinite_window_weights")
    weights_by_frame = np.zeros((N, T), dtype=np.float64)
    for i, start in enumerate(starts_arr.tolist()):
        weights_by_frame[i, start:start + F] = weights[i]
    weight_sum = weights_by_frame.sum(axis=0, dtype=np.float64)
    if np.any(~np.isfinite(weight_sum)) or np.any(weight_sum <= 0):
        raise RotationFusionRejected("no_positive_weight",
                                     evidence={"frames": np.flatnonzero(weight_sum <= 0).tolist()})

    # Positions and joints are exactly the original fusion output.
    pos_out, quat_legacy, joints_out, legacy_diag = fuse_windows(
        list(pos_arr), list(quat_arr), list(joints_arr), starts_arr.tolist(), T, stride)

    lifted = np.empty((N, F, 4), dtype=np.float64)
    lift_info: list[dict] = []
    first_lift_ambiguous = None
    for i in range(N):
        lifted[i], info = lift_window_signs(quat_arr[i])
        lift_info.append(info)
        if info["ambiguous"] is not None:
            local = int(info["ambiguous"]["local_frame"])
            global_frame = int(starts_arr[i]) + local
            candidate = {"window": i, "local_frame": local, "frame": global_frame,
                         "dot": float(info["ambiguous"]["dot"])}
            if first_lift_ambiguous is None or (global_frame, i) < (
                    first_lift_ambiguous["frame"], first_lift_ambiguous["window"]):
                first_lift_ambiguous = candidate

    activation_frames = []
    for i in range(N):
        positive = np.flatnonzero(weights[i] > 0)
        if not len(positive):
            raise RotationFusionRejected("window_never_active", window=i)
        activation_frames.append(int(starts_arr[i]) + int(positive[0]))
    if activation_frames[0] != 0:
        raise RotationFusionRejected("first_window_not_active_at_zero",
                                     evidence={"activation_frame": activation_frames[0]})

    signs = np.zeros(N, dtype=np.float64)
    activation_dots: list[float | None] = [None] * N
    quat_out = np.empty((T, 4), dtype=np.float64)
    blend_norm = np.full(T, np.nan, dtype=np.float64)
    frame_active: list[list[int]] = [[] for _ in range(T)]
    frame_alpha: list[list[float]] = [[] for _ in range(T)]
    activated = [False] * N

    for t in range(T):
        if first_lift_ambiguous is not None and t >= first_lift_ambiguous["frame"]:
            diagnostics = {
                "protocol": PROTOCOL, "starts": starts_arr.tolist(), "T": T, "F": F,
                "stride": stride, "first_rejection": "window_lift_ambiguous",
                "window": first_lift_ambiguous["window"],
                "local_frame": first_lift_ambiguous["local_frame"],
                "frame": first_lift_ambiguous["frame"],
                "lift": lift_info,
            }
            raise RotationFusionRejected(
                "window_lift_ambiguous", frame=t,
                window=first_lift_ambiguous["window"],
                local_frame=first_lift_ambiguous["local_frame"],
                evidence=first_lift_ambiguous,
                prefix=_prefix(pos_out, quat_out, joints_out, t),
                diagnostics=diagnostics)

        newly = [i for i in range(N) if activation_frames[i] == t and not activated[i]]
        if t == 0:
            if newly != [0]:
                raise RotationFusionRejected("first_window_activation_mismatch",
                                             frame=t, evidence={"newly": newly})
            signs[0] = 1.0
            activated[0] = True
        elif newly:
            previous = quat_out[t - 1]
            local_dots = {}
            for i in newly:
                local = t - int(starts_arr[i])
                dot = float(np.dot(lifted[i, local], previous))
                local_dots[i] = dot
                if abs(dot) <= PATH_AMBIGUITY_DOT:
                    diagnostics = {
                        "protocol": PROTOCOL, "starts": starts_arr.tolist(), "T": T,
                        "F": F, "stride": stride,
                        "first_rejection": "window_activation_ambiguous",
                        "window": i, "frame": t, "dot": dot,
                        "activation_dots": activation_dots,
                    }
                    raise RotationFusionRejected(
                        "window_activation_ambiguous", frame=t, window=i,
                        local_frame=t - int(starts_arr[i]),
                        evidence={"dot": dot}, prefix=_prefix(pos_out, quat_out, joints_out, t),
                        diagnostics=diagnostics)
            for i in newly:  # start order; all against the same previous fused frame
                signs[i] = 1.0 if local_dots[i] > 0 else -1.0
                activation_dots[i] = float(signs[i] * local_dots[i])
                lifted[i] *= signs[i]
                activated[i] = True

        total = float(weight_sum[t])
        u = np.zeros(4, dtype=np.float64)
        for i in range(N):
            w = float(weights_by_frame[i, t])
            if w <= 0:
                continue
            local = t - int(starts_arr[i])
            alpha = w / total
            u += alpha * lifted[i, local]
            frame_active[t].append(i)
            frame_alpha[t].append(float(alpha))
        norm = float(np.linalg.norm(u))
        if not np.isfinite(norm) or not np.isfinite(u).all():
            diagnostics = {
                "protocol": PROTOCOL, "starts": starts_arr.tolist(), "T": T,
                "F": F, "stride": stride, "first_rejection": "nonfinite_blend",
                "frame": t, "weight_sum": total,
            }
            raise RotationFusionRejected(
                "nonfinite_blend", frame=t,
                prefix=_prefix(pos_out, quat_out, joints_out, t), diagnostics=diagnostics)
        if norm < MIN_BLEND_NORM:
            diagnostics = {
                "protocol": PROTOCOL, "starts": starts_arr.tolist(), "T": T,
                "F": F, "stride": stride,
                "first_rejection": "quaternion_blend_ill_conditioned",
                "frame": t, "blend_norm": norm,
                "active_windows": frame_active[t], "alpha": frame_alpha[t],
                "signs": signs.tolist(),
            }
            raise RotationFusionRejected(
                "quaternion_blend_ill_conditioned", frame=t,
                evidence={"blend_norm": norm}, prefix=_prefix(pos_out, quat_out, joints_out, t),
                diagnostics=diagnostics)
        blend_norm[t] = norm
        quat_out[t] = u / norm

    if not (np.isfinite(quat_out).all() and np.isfinite(blend_norm).all()):
        raise RotationFusionRejected("nonfinite_fused_rotation")

    speed = angular_speed_radps(quat_out, frame_dt)
    pre_speed = angular_speed_radps(quat_legacy, frame_dt)
    window_speed = np.stack([angular_speed_radps(lifted[i], frame_dt) for i in range(N)])
    seams = _seam_indices(starts_arr, T)
    nonseam = np.setdiff1d(np.arange(max(T - 1, 0)), seams)
    speed_max, speed_frame = _worst(speed)
    seam_max, seam_frame = _worst(speed, seams)
    nonseam_max, nonseam_frame = _worst(speed, nonseam)
    pre_max, pre_frame = _worst(pre_speed)
    pre_seam_max, pre_seam_frame = _worst(pre_speed, seams)
    pre_nonseam_max, pre_nonseam_frame = _worst(pre_speed, nonseam)
    violations = np.flatnonzero(speed > limit) if len(speed) else np.asarray([], dtype=np.int64)
    temporal_pass = bool(not len(violations))
    window_worst = []
    for i in range(N):
        maximum, frame = _worst(window_speed[i])
        window_worst.append({"window": i, "max_radps": maximum,
                             "frame": frame,
                             "global_frame": None if frame is None else int(starts_arr[i]) + frame})

    diagnostics = {
        "protocol": PROTOCOL,
        "T": T, "F": F, "stride": stride, "n_windows": N,
        "frame_dt": frame_dt,
        "starts": starts_arr.tolist(),
        "window_weights": weights.tolist(),
        "window_initial_sign": [float(x["initial_sign"]) for x in lift_info],
        "window_lift_flips": [list(x["flip_frames"]) for x in lift_info],
        "window_lift_ambiguous": [x["ambiguous"] for x in lift_info],
        "window_signs": signs.tolist(),
        "window_activation_frames": activation_frames,
        "window_activation_dots": activation_dots,
        "frame_active_windows": frame_active,
        "frame_alpha": frame_alpha,
        "frame_weight_sum": [float(x) for x in weight_sum],
        "blend_norm": [float(x) for x in blend_norm],
        "blend_norm_min": float(blend_norm.min()),
        "blend_norm_min_frame": int(np.argmin(blend_norm)),
        "path_ambiguity_dot": PATH_AMBIGUITY_DOT,
        "min_blend_norm": MIN_BLEND_NORM,
        "quaternion_norm_tolerance": QUAT_NORM_TOL,
        "zero_weight_slots_ignored": int((weights == 0).sum()),
        "angular_speed_radps": [float(x) for x in speed],
        "angular_speed_global_max_radps": speed_max,
        "angular_speed_global_max_frame": speed_frame,
        "angular_speed_seam_max_radps": seam_max,
        "angular_speed_seam_max_frame": seam_frame,
        "angular_speed_nonseam_max_radps": nonseam_max,
        "angular_speed_nonseam_max_frame": nonseam_frame,
        "seam_steps": _json_index(seams),
        "pre_fusion_angular_speed_max_radps": pre_max,
        "pre_fusion_angular_speed_max_frame": pre_frame,
        "pre_fusion_seam_max_radps": pre_seam_max,
        "pre_fusion_seam_max_frame": pre_seam_frame,
        "pre_fusion_nonseam_max_radps": pre_nonseam_max,
        "pre_fusion_nonseam_max_frame": pre_nonseam_frame,
        "window_angular_speed_max": window_worst,
        "angular_speed_limit_radps": limit,
        "first_temporal_violation_frame": int(violations[0]) + 1 if len(violations) else None,
        "temporal_violation_frames": (violations + 1).tolist(),
        "fusion_temporal_pass": temporal_pass,
        "legacy_boundary": legacy_diag,
    }
    return RotationFusionResult(
        pos=pos_out, quat=quat_out, joints=joints_out, lifted_quat=lifted,
        weights=weights, diagnostics=diagnostics,
        numerical_valid=True, fusion_math_valid=True,
        fusion_temporal_pass=temporal_pass,
        consumer_eligible=bool(temporal_pass),
    )
