"""Collision pair registry for the locked G1 scene.

Self-collision: non-adjacent body pairs that both carry collision geoms.
Adjacency is defined as tree distance <= ``max_tree_distance`` (2), plus an
explicit allow-list for designed overlaps (e.g. the two ankle intermediate
balls inside the knee link region, feet vs their own leg). The allow-list
documents *why* each excluded pair is excluded; everything else is checked.

Environment: robot-vs-ground, robot-vs-box, box-vs-ground (all collision
geoms on both sides).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ireset.embodied.g1_spec import G1Spec


def _body_depths_and_parents(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    parent = m.body_parentid.copy()
    depth = np.zeros(m.nbody, dtype=int)
    for b in range(m.nbody):
        depth[b] = depth[parent[b]] + 1 if b else 0
    return depth, parent


def _tree_distance(m: mujoco.MjModel, a: int, b: int, depth: np.ndarray, parent: np.ndarray) -> int:
    x, y, d = a, b, 0
    while depth[x] > depth[y]:
        x = parent[x]; d += 1
    while depth[y] > depth[x]:
        y = parent[y]; d += 1
    while x != y:
        x = parent[x]; y = parent[y]; d += 2
    return d


#: Designed-in overlaps that must not be flagged (documented exclusions).
SELF_COLLISION_ALLOW: dict[tuple[str, str], str] = {
    ("left_ankle_intermediate_1_link", "left_ankle_pitch_link"): "ankle ball inside pitch link region",
    ("right_ankle_intermediate_1_link", "right_ankle_pitch_link"): "ankle ball inside pitch link region",
    ("left_ankle_intermediate_1_link", "left_knee_link"): "ankle ball adjacent to knee",
    ("right_ankle_intermediate_1_link", "right_knee_link"): "ankle ball adjacent to knee",
    ("torso_link", "pelvis_contour_link"): "waist overlap in nominal configuration",
}


@dataclass
class CollisionPairs:
    self_pairs: list[tuple[int, int]] = field(default_factory=list)     # geom id pairs
    robot_ground: list[tuple[int, int]] = field(default_factory=list)
    robot_box: list[tuple[int, int]] = field(default_factory=list)
    box_ground: tuple[int, int] | None = None
    excluded_adjacent: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_pairs(self) -> list[tuple[int, int, str]]:
        out = [(a, b, "self") for a, b in self.self_pairs]
        out += [(a, b, "robot_ground") for a, b in self.robot_ground]
        out += [(a, b, "robot_box") for a, b in self.robot_box]
        if self.box_ground is not None:
            out.append((*self.box_ground, "box_ground"))
        return out


def build_collision_pairs(spec: G1Spec, scene: bool = True, max_tree_distance: int = 2) -> CollisionPairs:
    m = spec.load_scene() if scene else spec.load_plain()
    geom_body = m.geom_bodyid
    body_name = lambda b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731

    collision_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    ground_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "largebox")

    robot_geoms = [
        g for g in collision_geoms
        if g not in (ground_geom, box_geom) and geom_body[g] != 0
    ]

    depth, parent = _body_depths_and_parents(m)
    pairs = CollisionPairs()

    for i, ga in enumerate(robot_geoms):
        for gb in robot_geoms[i + 1 :]:
            ba, bb = geom_body[ga], geom_body[gb]
            if ba == bb:
                continue
            na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ba)
            nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb)
            key = tuple(sorted((na, nb)))
            if key in SELF_COLLISION_ALLOW:
                continue
            if _tree_distance(m, ba, bb, depth, parent) <= max_tree_distance:
                pairs.excluded_adjacent.append(key)
                continue
            pairs.self_pairs.append((ga, gb))

    if scene and box_geom >= 0:
        pairs.robot_ground = [(g, ground_geom) for g in robot_geoms]
        pairs.robot_box = [(g, box_geom) for g in robot_geoms]
        pairs.box_ground = (box_geom, ground_geom)
    return pairs
