"""Differentiable distance helpers (analytic box AABB / ground plane).

Verified conventions (tests/test_geometry.py, M1):

- MuJoCo compiles the largebox mesh **recentered**: the geom frame is the
  stored-mesh frame, related to the OBJ file by the compile-time
  ``mesh_pos``/``mesh_quat``. World-frame distances must go through the
  *geom frame*, obtainable from ``d.geom_xpos/geom_xmat`` (or composed
  statically, see :func:`largebox_static_frame`).
- The scanned largebox is a rounded, non-watertight bin: its convex hull
  (the collision geometry) deviates from the analytic AABB by up to
  ~0.14 m (mean ~0.07 m outside). The AABB is therefore a *training
  proxy* only; acceptance/validation always uses MuJoCo hull queries.
- ``point_box_*`` functions operate in the **geom (stored mesh) frame**;
  use :func:`world_to_box_geom` to map world points there given the box
  body pose ``[x,y,z,qw,qx,qy,qz]``.
"""

from __future__ import annotations

import numpy as np
import torch

#: AABB of the stored largebox vertices (geom frame), from the compiled
#: locked scene (tests re-derive and guard these values).
LARGEBOX_AABB_CENTER = np.array([2.810e-3, 3.0e-5, 7.030e-3])
LARGEBOX_HALF_EXTENTS = np.array([0.22132, 0.21167, 0.20298])


def largebox_static_frame() -> tuple[np.ndarray, np.ndarray]:
    """Static geom frame (R, x) of the largebox in its **body** frame.

    Cached after first compile of the locked scene: with the box body at
    identity, ``geom_xmat/xpos`` give T_body_geom.
    """
    global _STATIC_FRAME
    if _STATIC_FRAME is None:
        import mujoco

        from ireset.embodied.g1_spec import G1Spec

        m = G1Spec().load_scene()
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
        bid = m.geom_bodyid[box]
        qadr = m.jnt_qposadr[m.body_jntadr[bid]]
        d.qpos[qadr : qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        _STATIC_FRAME = (d.geom_xmat[box].reshape(3, 3).copy(), d.geom_xpos[box].copy())
    return _STATIC_FRAME


_STATIC_FRAME: tuple[np.ndarray, np.ndarray] | None = None


def world_to_box_geom(points_world: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
    """World points [..., 3] -> largebox geom frame, per-frame pose [T, 7]."""
    from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz

    R_g, x_g = largebox_static_frame()
    R_b = quat_to_mat_wxyz(np.atleast_2d(box_pose)[:, 3:7])     # [T,3,3]
    t_b = np.atleast_2d(box_pose)[:, 0:3]                        # [T,3]
    pts = np.atleast_2d(points_world).reshape(box_pose.shape[0], -1, 3)
    p_body = np.einsum("tji,tnj->tni", R_b, pts - t_b[:, None, :])  # R_b^T (p - t)
    return np.einsum("ji,tnj->tni", R_g, p_body - x_g[None, None, :])  # R_g^T (p_body - x_g)


def box_geom_to_world(points_geom: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
    from ireset.kinematics.rotation_adapter import quat_to_mat_wxyz

    R_g, x_g = largebox_static_frame()
    R_b = quat_to_mat_wxyz(np.atleast_2d(box_pose)[:, 3:7])
    t_b = np.atleast_2d(box_pose)[:, 0:3]
    pts = np.atleast_2d(points_geom).reshape(box_pose.shape[0], -1, 3)
    p_body = np.einsum("ij,tnj->tni", R_g, pts) + x_g[None, None, :]   # R_g p + x_g (geom->body)
    return np.einsum("tij,tnj->tni", R_b, p_body) + t_b[:, None, :]


# --- acceptance-grade hull geometry ------------------------------------------

_HULL: "trimesh.Trimesh | None" = None


def largebox_hull():
    """Convex hull of the compiled largebox mesh (the collision geometry).

    This is the acceptance-grade geometry; the analytic AABB in this module
    is a differentiable *training* proxy only (mean ~1.8cm error at a true
    2cm gap — exactly the regime contact inference decides in).
    """
    global _HULL
    if _HULL is None:
        import mujoco
        import trimesh

        from ireset.embodied.g1_spec import G1Spec

        m = G1Spec().load_scene()
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "largebox_mesh")
        verts = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        _HULL = trimesh.Trimesh(vertices=verts.astype(np.float64), process=False).convex_hull
    return _HULL


def largebox_hull_surface_distance(points_world: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
    """Unsigned hull surface distance (0 inside), [T]; batched over frames.

    ``points_world`` [T, 3], ``box_pose`` [T, 7] project convention. This is
    the reference-side metric for contact scheduling and F1 scoring.
    """
    import trimesh

    geom = world_to_box_geom(points_world, box_pose).reshape(-1, 3)
    return np.clip(-trimesh.proximity.signed_distance(largebox_hull(), geom), 0.0, None)


# --- analytic AABB distances (geom frame) ------------------------------------


def point_box_surface_distance_numpy(
    p: np.ndarray,
    half_extents: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Unsigned AABB surface distance (0 inside); geom frame."""
    he = LARGEBOX_HALF_EXTENTS if half_extents is None else half_extents
    c = LARGEBOX_AABB_CENTER if center is None else center
    q = np.abs(p - c) - he
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1)


def point_box_sdf_numpy(p: np.ndarray, half_extents=None, center=None) -> np.ndarray:
    he = LARGEBOX_HALF_EXTENTS if half_extents is None else half_extents
    c = LARGEBOX_AABB_CENTER if center is None else center
    q = np.abs(p - c) - he
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def point_box_surface_distance_torch(
    p: torch.Tensor,
    half_extents: torch.Tensor | None = None,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable unsigned AABB surface distance (0 inside); geom frame.

    Inputs may be world-frame points already mapped to the geom frame, or a
    world pose can be applied upstream with :func:`world_to_box_geom`.
    """
    he = half_extents if half_extents is not None else torch.tensor(LARGEBOX_HALF_EXTENTS, dtype=p.dtype, device=p.device)
    c = center if center is not None else torch.tensor(LARGEBOX_AABB_CENTER, dtype=p.dtype, device=p.device)
    q = (p - c).abs() - he
    return torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)


def point_box_sdf_torch(p: torch.Tensor, half_extents=None, center=None) -> torch.Tensor:
    he = half_extents if half_extents is not None else torch.tensor(LARGEBOX_HALF_EXTENTS, dtype=p.dtype, device=p.device)
    c = center if center is not None else torch.tensor(LARGEBOX_AABB_CENTER, dtype=p.dtype, device=p.device)
    q = (p - c).abs() - he
    outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(torch.amax(q, dim=-1), max=0.0)
    return outside + inside


def point_plane_distance_torch(p: torch.Tensor, plane: torch.Tensor | None = None) -> torch.Tensor:
    """Height over the ground plane z=0 (positive up); plane [a,b,c,d] optional."""
    if plane is None:
        return p[..., 2]
    return (p * plane[:3]).sum(-1) + plane[3]
