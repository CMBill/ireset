"""Geometry layer tests: collision pair registry, analytic vs MuJoCo geometry.

Key verified facts (M1):

- The largebox geom frame equals the *stored mesh* frame; world points map
  via ``p_world = R_g @ p_stored + x_g`` with (R_g, x_g) from the current
  ``d.geom_xmat/geom_xpos``. MuJoCo hull distances and the analytic AABB
  both live in this frame.
- The scanned bin's convex hull deviates from the analytic AABB (up to
  ~0.14 m); the AABB is a training proxy, never an acceptance oracle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ireset.embodied.g1_spec import G1Spec
from ireset.geometry.collision_pairs import SELF_COLLISION_ALLOW, build_collision_pairs
from ireset.geometry.distance_adapter import (
    LARGEBOX_AABB_CENTER,
    LARGEBOX_HALF_EXTENTS,
    largebox_static_frame,
    point_box_sdf_numpy,
    point_box_sdf_torch,
    point_box_surface_distance_numpy,
    point_box_surface_distance_torch,
    point_plane_distance_torch,
    world_to_box_geom,
)


@pytest.fixture(scope="module")
def spec() -> G1Spec:
    return G1Spec()


def stored_box_data(spec: G1Spec):
    """(verts, hull-mesh, box-geom-id) from the compiled scene."""
    m = spec.load_scene()
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "largebox_mesh")
    verts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    hull = trimesh.Trimesh(vertices=verts, process=False).convex_hull
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
    return m, verts, hull, box


def test_self_collision_pairs_exclude_adjacent(spec: G1Spec) -> None:
    pairs = build_collision_pairs(spec, scene=True)
    m = spec.load_scene()
    body_names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    boxg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
    assert len(pairs.self_pairs) > 0
    for ga, gb in pairs.self_pairs:
        assert m.geom_bodyid[ga] != m.geom_bodyid[gb]
        assert ga not in (ground, boxg) and gb not in (ground, boxg)
    for a, b in SELF_COLLISION_ALLOW:
        assert a in body_names and b in body_names, f"stale allow entry {a}/{b}"
    assert pairs.box_ground is not None
    assert len(pairs.robot_box) > 0 and len(pairs.robot_ground) > 0


def test_stored_aabb_constants_match_compiled_mesh(spec: G1Spec) -> None:
    _, verts, _, _ = stored_box_data(spec)
    center = (verts.min(0) + verts.max(0)) / 2
    half = (verts.max(0) - verts.min(0)) / 2
    assert np.abs(center - LARGEBOX_AABB_CENTER).max() < 1e-4
    assert np.abs(half - LARGEBOX_HALF_EXTENTS).max() < 1e-5


def test_world_to_box_geom_matches_mujoco_hull(spec: G1Spec) -> None:
    """Probe spheres at analytic points must match MuJoCo hull distances.

    Residual is bounded by the sphere-probe approximation (2r) — the frame
    mapping itself must be exact.
    """
    m, verts, hull, _ = stored_box_data(spec)
    probe = mujoco.MjSpec.from_file(str(spec.scene_xml))
    probe.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0], contype=0, conaffinity=0)
    m2 = probe.compile()
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetData(m2, d2)
    box2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
    qadr2 = m2.jnt_qposadr[m2.body_jntadr[m2.geom_bodyid[box2]]]
    rng = np.random.default_rng(11)
    fromto = np.zeros(6)
    sph = [g for g in range(m2.ngeom) if m2.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE][0]
    max_gap = 0.0
    for _ in range(8):
        box_q = np.concatenate([rng.uniform(-1, 1, 3), rng.standard_normal(4)])
        box_q[3:7] /= np.linalg.norm(box_q[3:7])
        d2.qpos[qadr2 : qadr2 + 7] = box_q
        mujoco.mj_forward(m2, d2)
        R_g = d2.geom_xmat[box2].reshape(3, 3)
        x_g = d2.geom_xpos[box2]
        for _ in range(10):
            p_geom = rng.uniform(-0.55, 0.55, 3)
            if np.linalg.norm(p_geom) < 0.4:
                continue
            p_world = R_g @ p_geom + x_g
            d2.geom_xpos[sph] = p_world
            dist_mj = mujoco.mj_geomDistance(m2, d2, sph, box2, 10.0, fromto) - 1e-4
            dist_hull = -trimesh.proximity.signed_distance(hull, [p_geom])[0]
            max_gap = max(max_gap, abs(dist_mj - dist_hull))
    assert max_gap < 5e-4, f"frame mapping gap {max_gap}"

    # our world_to_box_geom must invert the same composition
    from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz

    box_q = np.array([0.1, -0.2, 0.35, 0.5, 0.5, 0.5, 0.5])
    R_b = quat_to_mat_wxyz(box_q[3:7][None])[0]
    t_b = box_q[0:3]
    R_gs, x_gs = largebox_static_frame()
    pts = rng.uniform(-0.6, 0.6, (5, 3))
    mapped = world_to_box_geom(pts[None], box_q[None])[0]
    direct = (R_gs.T @ ((R_b.T @ (pts - t_b).T).T - x_gs).T).T
    assert np.abs(mapped - direct).max() < 1e-10


def test_box_geom_world_roundtrip() -> None:
    """box_geom_to_world must invert world_to_box_geom (review round 3 fix).

    The mesh-recentering rotation R_g is non-symmetric, so a flipped
    geom->body direction does not cancel in the round trip (the previous
    ``"ji"`` einsum left a 0.26 m residual on box-surface points).
    """
    from ireset.geometry.distance_adapter import box_geom_to_world

    rng = np.random.default_rng(7)
    R_g, _ = largebox_static_frame()
    assert not np.allclose(R_g, R_g.T), "guard: recentering rotation must stay non-symmetric"
    for _ in range(4):
        box_q = np.concatenate([rng.uniform(-1, 1, 3), rng.standard_normal(4)])
        box_q[3:7] /= np.linalg.norm(box_q[3:7])
        pose = box_q[None].repeat(9, 0)
        pts = rng.uniform(-0.5, 0.5, (9, 3))
        back = world_to_box_geom(box_geom_to_world(pts, pose), pose)
        assert np.abs(back.reshape(9, 3) - pts).max() < 1e-10


def test_aabb_is_documented_proxy_not_oracle(spec: G1Spec) -> None:
    """Quantify the AABB-vs-hull approximation gap (rounded scanned bin)."""
    _, verts, hull, _ = stored_box_data(spec)
    rng = np.random.default_rng(12)
    gaps = []
    for p in rng.uniform(-0.7, 0.7, (120, 3)):
        if np.linalg.norm(p) <= 0.42:
            continue
        d_hull = -trimesh.proximity.signed_distance(hull, [p])[0]
        d_aabb = point_box_surface_distance_numpy(p[None])[0]
        gaps.append(abs(d_hull - d_aabb))
    assert np.mean(gaps) < 0.1 and max(gaps) < 0.2  # documented ~0.07 / ~0.14


def test_analytic_sdf_surface_properties() -> None:
    rng = np.random.default_rng(13)
    face_pts = []
    for axis in range(3):
        for sign in (+1, -1):
            for _ in range(20):
                q = rng.uniform(-1, 1, 3)
                q[axis] = sign
                face_pts.append(q * LARGEBOX_HALF_EXTENTS + LARGEBOX_AABB_CENTER)
    face_pts = np.array(face_pts)
    assert np.abs(point_box_sdf_numpy(face_pts)).max() < 1e-12
    assert np.abs(point_box_surface_distance_numpy(face_pts)).max() < 1e-12
    sdf_t = point_box_sdf_torch(torch.tensor(face_pts))
    assert np.abs(sdf_t.numpy()).max() < 1e-12
    inside = LARGEBOX_AABB_CENTER + rng.uniform(-0.8, 0.8, 3) * LARGEBOX_HALF_EXTENTS
    assert point_box_surface_distance_numpy(inside) == 0.0
    assert point_box_sdf_numpy(inside) < 0


def test_point_box_distance_gradient() -> None:
    he = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64)
    p = torch.tensor([[0.3, 0.1, 0.0]], dtype=torch.float64, requires_grad=True)
    d = point_box_surface_distance_torch(p, he, torch.zeros(3, dtype=torch.float64))
    d.sum().backward()
    assert torch.isfinite(p.grad).all()
    assert torch.allclose(p.grad[0].norm(), torch.tensor(1.0, dtype=torch.float64), atol=1e-9)


def test_plane_distance() -> None:
    p = torch.tensor([[0.1, 0.2, 0.5], [0.0, 0.0, -0.1]], dtype=torch.float64)
    assert torch.allclose(point_plane_distance_torch(p), torch.tensor([0.5, -0.1], dtype=torch.float64))
