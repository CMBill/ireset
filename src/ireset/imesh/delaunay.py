"""Per-frame interaction mesh: Delaunay topology + uniform Laplacian.

Plan section 5.3. Point layout (semantic index order, identical for the
reference and robot sides):

    [0:K)     K anatomical keypoints (human reference / G1 FK), in object frame
    [K:K+P)   fixed object surface samples, local coordinates (constant)
    [K+P:K+P+E) ground points, world-fixed then transformed into the object
              frame per frame (NOT glued to the object)

Per frame, a Delaunay triangulation is built **on the reference-side
points**; the resulting sparse uniform Laplacian ``L`` (``L x_i = x_i -
mean of Delaunay neighbors``) is reused unchanged for the robot-side points
and for all SCP iterations of that frame. Degenerate inputs (duplicates,
coplanarity) fall back to joggled input ``QJ`` with a fixed seed; failures
are reported, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix


@dataclass
class FrameMesh:
    """Topology + operator for one frame; points are supplied at evaluation."""

    tri: np.ndarray              # [M, 4] tetrahedra (3D Delaunay) or [M, 3] if 2D
    lap: csr_matrix             # [N, N] uniform Laplacian
    n_points: int
    used_fallback: bool          # True if QJ joggling was needed
    seed: int


def build_laplacian_from_delaunay(tri: np.ndarray, n_points: int) -> csr_matrix:
    """Uniform-weight Laplacian: L x = x - mean_neighbors(x)."""
    edges = set()
    simplex = tri.shape[1]
    for s in range(simplex):
        for t in range(s + 1, simplex):
            pairs = tri[:, [s, t]]
            for a, b in pairs:
                if a != b:
                    edges.add((int(a), int(b)))
    rows, cols, vals = [], [], []
    neighbor_lists: list[list[int]] = [[] for _ in range(n_points)]
    for a, b in edges:
        neighbor_lists[a].append(b)
        neighbor_lists[b].append(a)
    for i, nbrs in enumerate(neighbor_lists):
        rows.append(i); cols.append(i); vals.append(1.0)
        if nbrs:
            for j in nbrs:
                rows.append(i); cols.append(j); vals.append(-1.0 / len(nbrs))
    return csr_matrix((vals, (rows, cols)), shape=(n_points, n_points))


def build_frame_mesh(
    points_obj: np.ndarray,
    seed: int = 0,
    degenerate_eps: float = 1e-9,
) -> FrameMesh:
    """Build the per-frame mesh from reference-side object-frame points [N,3].

    Points closer than ``degenerate_eps`` are perturbed by a seeded,
    scale-dependent joggle (max 1e-6 of the point-cloud scale) before
    triangulation so Delaunay does not fail on duplicates/coplanar sets.
    """
    pts = np.asarray(points_obj, dtype=float).copy()
    n = pts.shape[0]
    scale = max(float(np.abs(pts).max()), 1e-6)

    rng = np.random.default_rng(seed)
    used_fallback = False
    # break exact duplicates first (Delaunay requirement)
    min_dist = _min_pair_distance(pts)
    if min_dist < degenerate_eps:
        pts += rng.uniform(-1e-6, 1e-6, pts.shape) * scale
        used_fallback = True
    try:
        tri = Delaunay(pts).simplices
    except Exception:
        # QJ: joggle input to escape degeneracy, deterministic via seed offset
        pts2 = pts + rng.uniform(-1e-5, 1e-5, pts.shape) * scale
        tri = Delaunay(pts2, qhull_options="QJ").simplices
        used_fallback = True
    return FrameMesh(tri=tri, lap=build_laplacian_from_delaunay(tri, n), n_points=n,
                     used_fallback=used_fallback, seed=seed)


def _min_pair_distance(pts: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(d[:, 1].min())


def laplacian_coordinates(points: np.ndarray, mesh: FrameMesh) -> np.ndarray:
    """delta = L @ points, [N, 3]."""
    return (mesh.lap @ np.asarray(points, dtype=float)).reshape(mesh.n_points, 3)


def mesh_energy(points_obj_robot: np.ndarray, delta_ref: np.ndarray, mesh: FrameMesh) -> float:
    """E_L = mean(||delta_robot - delta_ref||^2) (plan section 5.3)."""
    d = laplacian_coordinates(points_obj_robot, mesh) - delta_ref
    return float((d ** 2).sum(-1).mean())
