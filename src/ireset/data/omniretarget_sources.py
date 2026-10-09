"""OmniRetarget teacher source loading, layout conversion and binding checks.

Two NPZ layouts exist in this project (A2/I0 smoke finding, 2026-10-09):

* ``published`` (dataset revision 1352591): qpos [T, 43] with both
  freejoints stored ``[qw, qx, qy, qz, x, y, z]`` (root 0:7, box 36:43).
* ``upstream_generated`` (pinned checkout bccd4d7): the same 43 columns in
  MuJoCo order ``[x, y, z, qw, qx, qy, z]`` (root 0:7, box 36:43).

Both convert here to the project scene convention (root 0:3 pos / 3:7 quat
wxyz, joints 7:36, box 36:39 pos / 39:43 quat wxyz). Source identity is
explicit; nothing is auto-detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ireset.data import pairing

LAYOUTS = ("published", "upstream_generated", "project_scene")


def convert_teacher_row(row: np.ndarray, layout: str) -> np.ndarray:
    """Raw teacher row [43] -> project scene row [x,y,z,qw,qx,qy,qz] joints box."""
    out = np.zeros(43)
    if layout == "published":            # [qw,qx,qy,qz,x,y,z] both freejoints
        out[0:3], out[3:7] = row[4:7], row[0:4]
        out[36:39], out[39:43] = row[40:43], row[36:40]
    elif layout == "upstream_generated":  # [x,y,z,qw,qx,qy,qz] both freejoints
        out[0:3], out[3:7] = row[0:3], row[3:7]
        out[36:39], out[39:43] = row[36:39], row[39:43]
    elif layout == "project_scene":       # already converted (I0 output copy)
        out = np.asarray(row, dtype=float).copy()
    else:
        raise ValueError(f"unknown teacher layout {layout!r} (expected one of {LAYOUTS})")
    out[7:36] = row[7:36]
    return out


def load_teacher_scene(
    path: Path, layout: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (scene [T,43] project convention, human [T,52,3], fps).

    Structural checks follow plan §4.2: qpos exactly [T,43], human exactly
    [T,52,3], fps finite, quaternion norms within the 1e-4 bound.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown teacher layout {layout!r}")
    q_raw, human, fps = pairing.load_teacher_npz_raw(path)
    if not np.isfinite(fps):
        raise ValueError(f"{path.name}: fps not finite")
    scene = np.stack([convert_teacher_row(r, layout) for r in q_raw])
    for name, sl in (("root", (slice(0, 3), slice(3, 7))), ("box", (slice(36, 39), slice(39, 43)))):
        pos, quat = scene[:, sl[0]], scene[:, sl[1]]
        nerr = float(np.abs(np.linalg.norm(quat, axis=1) - 1.0).max())
        if nerr > pairing.GATE_QUAT_NORM_ERR:
            raise ValueError(f"{path.name}: {name} quaternion norm error {nerr} > 1e-4")
        if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
            raise ValueError(f"{path.name}: non-finite {name} pose")
    return scene, human, fps


@dataclass(frozen=True)
class BindingCheck:
    """U0-vs-P binding evidence for one generated demo (plan §3.3)."""

    demo: str
    human_max_abs_diff: float
    box_pos_max_abs_diff_m: float
    box_quat_abs_dot_min: float
    t_match: bool
    fps_match: bool
    passed: bool

    def to_json_dict(self) -> dict:
        return {
            "demo": self.demo,
            "human_max_abs_diff": self.human_max_abs_diff,
            "box_pos_max_abs_diff_m": self.box_pos_max_abs_diff_m,
            "box_quat_abs_dot_min": self.box_quat_abs_dot_min,
            "t_match": self.t_match,
            "fps_match": self.fps_match,
            "passed": self.passed,
        }


def check_binding(
    demo: str,
    generated_scene: np.ndarray,
    generated_human: np.ndarray,
    generated_fps: float,
    published_scene: np.ndarray,
    published_human: np.ndarray,
    published_fps: float,
    pos_tol: float = 1e-9,
    quat_dot_tol: float = 1.0 - 1e-6,
) -> BindingCheck:
    """Same source reference + same target box trajectory + same fps/T.

    The smoke run established human/target-box are bit-exact for the pinned
    checkout (quat |dot| >= 1 - 6e-8 from solver round-off); the defaults
    allow only float round-off. A failure means the upstream pipeline
    changed the bound inputs (rescaled/moved the target box) and the result
    may not enter the paired quality comparison.
    """
    gs, ps = np.asarray(generated_scene), np.asarray(published_scene)
    gh, ph = np.asarray(generated_human), np.asarray(published_human)
    t_match = gs.shape == ps.shape and gh.shape == ph.shape
    fps_match = float(generated_fps) == float(published_fps)
    if not t_match:
        return BindingCheck(demo, float("inf"), float("inf"), 0.0, False, fps_match, False)
    human_diff = float(np.abs(gh - ph).max())
    box_pos_diff = float(np.abs(gs[:, 36:39] - ps[:, 36:39]).max())
    dot_min = float(np.abs((gs[:, 39:43] * ps[:, 39:43]).sum(1)).min())
    passed = (
        human_diff <= pos_tol
        and box_pos_diff <= pos_tol
        and dot_min >= quat_dot_tol
        and t_match
        and fps_match
    )
    return BindingCheck(demo, human_diff, box_pos_diff, dot_min, t_match, fps_match, passed)
