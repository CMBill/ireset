"""Interaction-mesh tests: rigid invariance, semantic layout, degeneracy.

Plan section 5.3 acceptance: Laplacian coordinates are world-translation
invariant and world-rotation equivariant; object-frame deltas are invariant
under a rigid transform of the whole scene (points + object pose + ground),
since everything is consistently re-expressed in the object frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.imesh.delaunay import build_frame_mesh, laplacian_coordinates, mesh_energy
from ireset.imesh.transform import (
    object_to_world,
    world_to_object,
    world_to_object_torch,
)
from ireset.kinematics.rotation_adapter import (
    mat_to_quat_wxyz,
    quat_to_mat_wxyz,
    so3_exp_np,
)


def _make_scene(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference keypoints world [K,3], object pose [7], ground [E,3]."""
    rng = np.random.default_rng(seed)
    kpts = rng.uniform(-0.5, 0.5, (15, 3))
    pose = np.concatenate([rng.uniform(-1, 1, 3), mat_to_quat_wxyz(so3_exp_np(rng.standard_normal(3) * 0.5))])[None]
    ground = np.stack(np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), [0.0]), -1).reshape(-1, 3)
    return kpts, pose, ground


def _assemble(kpts_world: np.ndarray, pose: np.ndarray, ground_world: np.ndarray, surf_local: np.ndarray) -> np.ndarray:
    K, P = kpts_world.shape[0], surf_local.shape[0]
    pts = np.empty((K + P + ground_world.shape[0], 3))
    pts[:K] = world_to_object(kpts_world[None], pose)[0]
    pts[K : K + P] = surf_local
    pts[K + P :] = world_to_object(ground_world[None], pose)[0]
    return pts


def test_object_frame_roundtrip() -> None:
    kpts, pose, _ = _make_scene(0)
    obj = world_to_object(kpts[None], pose)[0]
    back = object_to_world(obj[None], pose)[0]
    assert np.abs(back - kpts).max() < 1e-10


def test_torch_transform_matches_numpy_and_grads() -> None:
    kpts, pose, _ = _make_scene(1)
    ref = world_to_object(kpts[None], pose)[0]
    pos_t = torch.tensor(pose[0, 0:3])[None].requires_grad_(True)          # [1,3]
    quat_t = torch.tensor(pose[0, 3:7])[None].requires_grad_(True)         # [1,4]
    pts_t = torch.tensor(kpts, requires_grad=True)                          # [K,3]
    out = world_to_object_torch(pts_t[None], pos_t, quat_t)[0]
    assert np.abs(out.detach().numpy() - ref).max() < 1e-12
    out.sum().backward()
    assert torch.isfinite(pos_t.grad).all() and torch.isfinite(pts_t.grad).all()


def test_laplacian_operator_is_translation_invariant_rotation_equivariant() -> None:
    """Operator-level properties in a fixed frame (plan section 5.3)."""
    kpts, pose, _ = _make_scene(2)
    surf = np.array([[0.2, 0.1, 0.15], [-0.2, -0.1, 0.15], [0.0, 0.0, -0.2]])
    pts = np.vstack([kpts, surf])  # plain world-frame cloud
    mesh = build_frame_mesh(pts, seed=0)
    delta = laplacian_coordinates(pts, mesh)

    # translation: identical deltas
    assert np.abs(laplacian_coordinates(pts + np.array([1.0, -2.0, 0.5]), mesh) - delta).max() < 1e-10

    # rotation: deltas rotate with the points (same topology reused)
    R = so3_exp_np(np.array([0.1, -0.2, 0.3]))
    assert np.abs(laplacian_coordinates(pts @ R.T, mesh) - delta @ R.T).max() < 1e-9


def test_rigid_scene_transform_invariance() -> None:
    """Whole-scene rigid motion (object included) leaves object-frame deltas unchanged.

    This is the property the plan requires for the clip/object-frame condition:
    all geometry (kpts, object pose, ground) transformed consistently.
    """
    kpts, pose, ground = _make_scene(3)
    surf = np.array([[0.2, 0.1, 0.15], [-0.2, -0.1, 0.15], [0.0, 0.0, -0.2], [0.1, -0.2, 0.2]])
    pts = _assemble(kpts, pose, ground, surf)
    mesh = build_frame_mesh(pts, seed=0)
    delta = laplacian_coordinates(pts, mesh)

    R = so3_exp_np(np.array([0.2, 0.3, -0.1]))
    t = np.array([0.7, -0.3, 0.2])
    R_obj = quat_to_mat_wxyz(pose[0, 3:7])
    kpts2 = kpts @ R.T + t
    pose2 = pose.copy()
    pose2[0, 0:3] = pose[0, 0:3] @ R.T + t
    pose2[0, 3:7] = mat_to_quat_wxyz(R @ R_obj)
    ground2 = ground @ R.T + t
    pts2 = _assemble(kpts2, pose2, ground2, surf)
    assert np.abs(laplacian_coordinates(pts2, mesh) - delta).max() < 1e-9


def test_energy_zero_for_matching_robot_points() -> None:
    kpts, pose, ground = _make_scene(4)
    surf = np.array([[0.2, 0.1, 0.15], [-0.2, -0.1, 0.15], [0.0, 0.0, -0.2]])
    pts = _assemble(kpts, pose, ground, surf)
    mesh = build_frame_mesh(pts, seed=0)
    delta = laplacian_coordinates(pts, mesh)
    assert mesh_energy(pts, delta, mesh) == 0.0
    assert mesh_energy(pts + 0.01, delta, mesh) > 0.0


def test_degenerate_duplicate_points_fall_back() -> None:
    rng = np.random.default_rng(5)
    pts = rng.uniform(-0.5, 0.5, (20, 3))
    pts[7] = pts[3]  # exact duplicate
    mesh = build_frame_mesh(pts, seed=0)
    assert mesh.used_fallback
    delta = laplacian_coordinates(pts, mesh)
    assert np.isfinite(delta).all()
