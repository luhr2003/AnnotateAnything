from __future__ import annotations

"""
region_proposal.py

Object-side anchor proposal for category-based grasp generation.

Design:
- Pure library module: no SimulationApp import, no Isaac Sim app startup.
- Reads object geometry from USD using pxr.
- Builds an assembled object mesh from all Mesh prims under a root prim.
- Samples surface points from the assembled mesh.
- Computes shared descriptors.
- Proposes category-specific anchors.

Current status:
- cat1 (edge_rim): implemented
- cat2 (bottom_band): implemented
- cat3 (side_wrap): implemented
- cat4 (convex_hold): scaffolded

Intended usage from main code:
    anchors = propose_anchors_from_usd(
        usd_path=...,
        category="cat1",
        region_cfg=...,
        num_surface_points=3000,
        root_prim_path=None,
        seed=0,
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import argparse
import math

import numpy as np

from pxr import Gf, Usd, UsdGeom


Category = Literal["cat1", "cat2", "cat3", "cat4"]


# ============================================================
# Data structures
# ============================================================

@dataclass
class AssembledMesh:
    vertices: np.ndarray          # (V, 3)
    faces: np.ndarray             # (F, 3), triangle indices into vertices
    mesh_prim_paths: List[str]    # source mesh prim paths


@dataclass
class SurfaceSamples:
    points: np.ndarray            # (N, 3)
    normals: np.ndarray           # (N, 3)
    face_indices: np.ndarray      # (N,)
    triangle_vertices: np.ndarray # (N, 3, 3)
    bbox_min: np.ndarray          # (3,)
    bbox_max: np.ndarray          # (3,)


@dataclass
class Anchor:
    category: str
    point: np.ndarray                     # (3,)
    normal: np.ndarray                    # (3,)
    score: float
    frame_R: np.ndarray                   # (3, 3), columns are local x/y/z axes
    support_indices: np.ndarray           # indices into sampled points
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Small utilities
# ============================================================

def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _gf_matrix_to_np(mat: Gf.Matrix4d) -> np.ndarray:
    # USD uses row-vector convention (p' = p @ M), so translation is at mat[3][0:3].
    # _transform_points uses column-vector convention (p' = M @ p), so we transpose
    # to move the translation into the last column where it belongs.
    out = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = mat[i][j]
    return out.T


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    # points: (N,3), T: (4,4)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([points, ones], axis=1)
    out = (T @ homog.T).T
    return out[:, :3]


def _fan_triangulate(face_counts: Sequence[int], face_indices: Sequence[int]) -> np.ndarray:
    """
    Convert arbitrary polygon faces to triangle faces using fan triangulation.
    """
    face_counts = list(face_counts)
    face_indices = list(face_indices)

    tris: List[Tuple[int, int, int]] = []
    cursor = 0
    for c in face_counts:
        if c < 3:
            cursor += c
            continue
        poly = face_indices[cursor : cursor + c]
        cursor += c
        for i in range(1, c - 1):
            tris.append((poly[0], poly[i], poly[i + 1]))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def _triangle_areas_and_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]  # (F,3,3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    nz = np.linalg.norm(cross, axis=1) > 1e-12
    normals[nz] = cross[nz] / np.linalg.norm(cross[nz], axis=1, keepdims=True)
    return areas, normals


def _sample_points_on_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Area-weighted triangle sampling.
    Returns:
        points: (N,3)
        face_indices: (N,)
        tri_vertices: (N,3,3)
    """
    rng = np.random.default_rng(seed)

    areas, _ = _triangle_areas_and_normals(vertices, faces)
    total_area = areas.sum()
    if total_area <= 0:
        raise ValueError("Mesh has zero total triangle area; cannot sample surface points.")

    probs = areas / total_area
    chosen_faces = rng.choice(len(faces), size=num_points, replace=True, p=probs)

    tri = vertices[faces[chosen_faces]]  # (N,3,3)

    # Barycentric sampling
    u = rng.random(num_points)
    v = rng.random(num_points)
    sqrt_u = np.sqrt(u)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v

    points = (
        tri[:, 0] * w0[:, None] +
        tri[:, 1] * w1[:, None] +
        tri[:, 2] * w2[:, None]
    )

    return points, chosen_faces, tri


def _triangle_face_normals(triangles: np.ndarray) -> np.ndarray:
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = np.zeros_like(cross)
    valid = norms[:, 0] > 1e-12
    normals[valid] = cross[valid] / norms[valid]
    return normals


def _pairwise_knn(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    kNN using scipy cKDTree — O(N log N), works for large N.
    Returns:
        knn_idx: (N, k)
        knn_dist: (N, k)
    """
    from scipy.spatial import cKDTree
    k = min(k, max(1, len(points) - 1))
    tree = cKDTree(points)
    # k+1 because the query returns the point itself as the nearest neighbour
    dists, indices = tree.query(points, k=k + 1, workers=-1)
    # Strip self (always at column 0, dist=0)
    return indices[:, 1:], dists[:, 1:]


def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    """
    Monotonic chain convex hull.
    Returns hull vertices in CCW order.
    """
    pts = np.unique(points_xy, axis=0)
    if len(pts) <= 1:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def _point_to_segment_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / ab2)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _distance_to_convex_hull_2d(points_xy: np.ndarray, hull_xy: np.ndarray) -> np.ndarray:
    if len(hull_xy) == 0:
        return np.full(len(points_xy), np.inf, dtype=np.float64)
    if len(hull_xy) == 1:
        return np.linalg.norm(points_xy - hull_xy[0], axis=1)

    dists = np.full(len(points_xy), np.inf, dtype=np.float64)
    for i in range(len(hull_xy)):
        a = hull_xy[i]
        b = hull_xy[(i + 1) % len(hull_xy)]
        seg_d = np.array([_point_to_segment_distance_2d(p, a, b) for p in points_xy])
        dists = np.minimum(dists, seg_d)
    return dists


def _nms_keep(points: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    """
    Greedy non-maximum suppression by 3D distance.
    """
    order = np.argsort(-scores)
    keep: List[int] = []

    for idx in order:
        if len(keep) == 0:
            keep.append(int(idx))
            continue
        d = np.linalg.norm(points[keep] - points[idx], axis=1)
        if np.all(d > radius):
            keep.append(int(idx))

    return np.asarray(keep, dtype=np.int64)


def _best_scoring_band_mask(values: np.ndarray, scores: np.ndarray, band_width: float) -> np.ndarray:
    """
    Return a boolean mask for the highest-scoring 1D band of fixed width.
    """
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    if len(values) == 1 or band_width <= 0.0:
        return np.ones(len(values), dtype=bool)

    order = np.argsort(values)
    v = values[order]
    s = scores[order]

    prefix = np.concatenate([[0.0], np.cumsum(s)])
    best_i = 0
    best_j = 1
    best_score = -np.inf
    j = 0
    for i in range(len(v)):
        while j < len(v) and (v[j] - v[i]) <= band_width:
            j += 1
        window_score = prefix[j] - prefix[i]
        if window_score > best_score:
            best_score = window_score
            best_i = i
            best_j = j

    keep_sorted = np.zeros(len(values), dtype=bool)
    keep_sorted[best_i:best_j] = True
    keep = np.zeros(len(values), dtype=bool)
    keep[order] = keep_sorted
    return keep


def _connected_components_from_mask(
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    mask: np.ndarray,
    radius: float,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Build connected components over the masked points using a symmetric kNN graph.
    """
    labels = np.full(len(mask), -1, dtype=np.int64)
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return [], labels

    adjacency: List[List[int]] = [[] for _ in range(len(mask))]
    for idx in valid_idx:
        for nbr, dist in zip(knn_idx[idx], knn_dist[idx]):
            nbr_i = int(nbr)
            if dist > radius or not mask[nbr_i]:
                continue
            adjacency[idx].append(nbr_i)
            adjacency[nbr_i].append(int(idx))

    components: List[np.ndarray] = []
    comp_id = 0
    for root in valid_idx:
        if labels[root] >= 0:
            continue
        stack = [int(root)]
        labels[root] = comp_id
        comp_members: List[int] = []
        while stack:
            cur = stack.pop()
            comp_members.append(cur)
            for nbr in adjacency[cur]:
                if labels[nbr] >= 0:
                    continue
                labels[nbr] = comp_id
                stack.append(nbr)
        components.append(np.asarray(comp_members, dtype=np.int64))
        comp_id += 1

    return components, labels


def _component_principal_extent_xy(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0

    pts_xy = np.asarray(points[:, :2], dtype=np.float64)
    centered = pts_xy - pts_xy.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(pts_xy))
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    proj = pts_xy @ axis
    return float(proj.max() - proj.min())


def _compute_local_frame_cat1(
    anchor_point: np.ndarray,
    neighbor_points: np.ndarray,
    anchor_normal: np.ndarray,
    object_center_xy: np.ndarray,
) -> np.ndarray:
    """
    Build a local frame for cat1.
    x-axis: local tangent (prefer horizontal)
    z-axis: global up
    y-axis: z × x

    If PCA tangent is unstable, fall back to an outward/perimeter-based tangent.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    centered = neighbor_points - neighbor_points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(neighbor_points))
    evals, evecs = np.linalg.eigh(cov)
    tangent = evecs[:, np.argmax(evals)]

    tangent_xy = tangent.copy()
    tangent_xy[2] = 0.0
    tangent_xy = _safe_normalize(tangent_xy)

    if np.linalg.norm(tangent_xy) < 1e-6:
        outward_xy = anchor_point[:2] - object_center_xy
        outward_xy = _safe_normalize(np.array([outward_xy[0], outward_xy[1], 0.0]))
        tangent_xy = np.array([-outward_xy[1], outward_xy[0], 0.0], dtype=np.float64)
        tangent_xy = _safe_normalize(tangent_xy)

    x_axis = tangent_xy
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    if np.linalg.norm(y_axis) < 1e-6:
        # Fall back to normal-derived axis
        y_axis = _safe_normalize(np.cross(anchor_normal, x_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = _safe_normalize(np.cross(y_axis, z_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R


def _compute_local_frame_cat2(
    anchor_point: np.ndarray,
    neighbor_points: np.ndarray,
    object_center_xy: np.ndarray,
) -> np.ndarray:
    """
    Build a local frame for cat2.
    x-axis: lower-edge tangent in XY
    y-axis: outward horizontal direction from object center
    z-axis: global up
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    outward_xy = anchor_point[:2] - object_center_xy
    y_axis = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
    y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    centered = neighbor_points - neighbor_points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(neighbor_points))
    evals, evecs = np.linalg.eigh(cov)
    tangent = evecs[:, np.argmax(evals)]
    tangent_xy = tangent.copy()
    tangent_xy[2] = 0.0
    tangent_xy = _safe_normalize(tangent_xy)
    if np.linalg.norm(tangent_xy) < 1e-6:
        tangent_xy = np.array([-y_axis[1], y_axis[0], 0.0], dtype=np.float64)
        tangent_xy = _safe_normalize(tangent_xy)

    if float(np.dot(tangent_xy, y_axis)) > 0.85:
        tangent_xy = np.array([-y_axis[1], y_axis[0], 0.0], dtype=np.float64)
        tangent_xy = _safe_normalize(tangent_xy)

    x_axis = tangent_xy
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))
    if float(np.dot(y_axis[:2], outward_xy)) < 0.0:
        x_axis *= -1.0
        y_axis *= -1.0
    x_axis = _safe_normalize(np.cross(y_axis, z_axis))

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _compute_local_frame_cat4(
    anchor_point: np.ndarray,
    anchor_normal: np.ndarray,
    object_center_xy: np.ndarray,
) -> np.ndarray:
    """
    Build a local frame for cat4.
    x-axis: horizontal tangent around the convex side
    y-axis: outward radial direction from the object center
    z-axis: global up

    cat4 is intended for BODex-style single-hand convex holds driven from the
    direct USD mesh, so the anchor frame should represent a stable side patch
    rather than an edge transition.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    outward_xy = anchor_point[:2] - object_center_xy
    y_axis = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
    y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = _safe_normalize(_project_to_xy(anchor_normal))
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = _safe_normalize(np.cross(z_axis, y_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = _safe_normalize(np.cross(x_axis, z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _compute_local_frame_cat3(
    anchor_point: np.ndarray,
    anchor_normal: np.ndarray,
    object_center_xy: np.ndarray,
) -> np.ndarray:
    """
    Build a geometric local frame for cat3.
    x-axis: horizontal tangent around the side band
    y-axis: outward radial direction from the object center
    z-axis: global up

    This geometric frame is used for anchor ranking and opposite-side pairing.
    Later, contact resolution re-orients cat3 into the hand-oriented
    "fingers down / palm inward" frame used for side holding.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    outward_xy = anchor_point[:2] - object_center_xy
    y_axis = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
    y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = _safe_normalize(_project_to_xy(anchor_normal))
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = _safe_normalize(np.cross(z_axis, y_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = _safe_normalize(np.cross(x_axis, z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _summarize_cat2_anchor_distribution(
    anchors: Sequence[Anchor],
    samples: SurfaceSamples,
) -> Dict[str, Any]:
    if len(anchors) < 3:
        return {
            "cat2_anchor_distribution_mode": "default",
        }

    xy = np.asarray([anchor.point[:2] for anchor in anchors], dtype=np.float64)
    z = np.asarray([anchor.point[2] for anchor in anchors], dtype=np.float64)
    center_xy = np.mean(xy, axis=0)
    centered_xy = xy - center_xy[None, :]
    cov = centered_xy.T @ centered_xy / max(1, len(anchors))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    major_axis_xy = _safe_normalize(evecs[:, order[-1]])
    if np.linalg.norm(major_axis_xy) < 1e-8:
        major_axis_xy = np.array([1.0, 0.0], dtype=np.float64)

    mean_tangent_xy = np.mean(
        np.asarray([anchor.frame_R[:2, 0] for anchor in anchors], dtype=np.float64),
        axis=0,
    )
    if np.linalg.norm(mean_tangent_xy) > 1e-8 and float(np.dot(major_axis_xy, mean_tangent_xy)) < 0.0:
        major_axis_xy *= -1.0

    side_axis_xy = np.array([-major_axis_xy[1], major_axis_xy[0]], dtype=np.float64)
    side_axis_xy = _safe_normalize(side_axis_xy)
    if np.linalg.norm(side_axis_xy) < 1e-8:
        side_axis_xy = np.array([0.0, 1.0], dtype=np.float64)

    existing_side_xy = np.mean(
        np.asarray([anchor.frame_R[:2, 1] for anchor in anchors], dtype=np.float64),
        axis=0,
    )
    if np.linalg.norm(existing_side_xy) < 1e-8:
        bbox_center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
        existing_side_xy = np.mean(xy - bbox_center_xy[None, :], axis=0)
    if np.linalg.norm(existing_side_xy) > 1e-8 and float(np.dot(side_axis_xy, existing_side_xy)) < 0.0:
        side_axis_xy *= -1.0

    major_proj = centered_xy @ major_axis_xy
    side_proj = centered_xy @ side_axis_xy
    major_span = float(np.ptp(major_proj))
    side_span = float(np.ptp(side_proj))
    major_std = float(np.std(major_proj))
    side_std = float(np.std(side_proj))
    z_span = float(np.ptp(z))
    side_alignment_mean = float(
        np.mean(
            np.abs(
                np.asarray([anchor.frame_R[:2, 1] for anchor in anchors], dtype=np.float64)
                @ side_axis_xy
            )
        )
    )

    line_like = bool(
        len(anchors) >= 3
        and major_span >= 0.06
        and side_span <= 0.04
        and side_span <= max(0.012, 0.30 * max(major_span, 1e-8))
        and z_span <= 0.05
        and side_alignment_mean >= 0.55
    )

    return {
        "cat2_anchor_distribution_mode": "bottom_rim_line" if line_like else "default",
        "cat2_line_center_xy": [float(x) for x in center_xy.tolist()],
        "cat2_line_axis_xy": [float(x) for x in major_axis_xy.tolist()],
        "cat2_line_side_axis_xy": [float(x) for x in side_axis_xy.tolist()],
        "cat2_line_major_span": major_span,
        "cat2_line_side_span": side_span,
        "cat2_line_major_std": major_std,
        "cat2_line_side_std": side_std,
        "cat2_line_z_span": z_span,
        "cat2_line_side_alignment_mean": side_alignment_mean,
    }


# ============================================================
# USD assembly
# ============================================================

def load_assembled_mesh_from_usd(
    usd_path: str | Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[Sequence[str]] = None,
) -> AssembledMesh:
    """
    Assemble all Mesh prims under a root prim into one mesh in world frame.

    Keeping proposal geometry in world coordinates makes anchor generation
    consistent with the SDF/object-query path, which also bakes authored USD
    transforms such as scale into the sampled geometry.

    root_prim_path:
        - if provided, traverse under that prim
        - otherwise use default prim if available
        - otherwise use the first child under pseudo-root

    exclude_prim_paths:
        - optional absolute prim paths to skip
    """
    usd_path = Path(usd_path).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    if root_prim_path is not None:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            raise ValueError(f"Invalid root prim path: {root_prim_path}")
    else:
        root_prim = stage.GetDefaultPrim()
        if not root_prim or not root_prim.IsValid():
            children = [p for p in stage.GetPseudoRoot().GetChildren()]
            if len(children) == 0:
                raise RuntimeError("USD stage has no traversable root children.")
            root_prim = children[0]

    exclude_set = set(exclude_prim_paths or [])

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    assembled_vertices: List[np.ndarray] = []
    assembled_faces: List[np.ndarray] = []
    prim_paths: List[str] = []

    vertex_offset = 0

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.GetPath().pathString in exclude_set:
            continue

        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr().Get()
        fvc_attr = mesh.GetFaceVertexCountsAttr().Get()
        fvi_attr = mesh.GetFaceVertexIndicesAttr().Get()

        if pts_attr is None or fvc_attr is None or fvi_attr is None:
            continue

        vertices_local = np.asarray(pts_attr, dtype=np.float64)
        faces_local = _fan_triangulate(fvc_attr, fvi_attr)
        if len(vertices_local) == 0 or len(faces_local) == 0:
            continue

        mesh_world = _gf_matrix_to_np(xform_cache.GetLocalToWorldTransform(prim))
        vertices_world = _transform_points(vertices_local, mesh_world)

        assembled_vertices.append(vertices_world)
        assembled_faces.append(faces_local + vertex_offset)
        prim_paths.append(prim.GetPath().pathString)

        vertex_offset += vertices_world.shape[0]

    if len(assembled_vertices) == 0 or len(assembled_faces) == 0:
        raise RuntimeError("No valid Mesh prims found for assembly.")

    vertices = np.concatenate(assembled_vertices, axis=0)
    faces = np.concatenate(assembled_faces, axis=0)

    # Release the standalone pxr stage so PhysX/Fabric don't warn about a
    # leaked stage reference at Isaac Sim shutdown.
    del stage

    return AssembledMesh(
        vertices=vertices,
        faces=faces,
        mesh_prim_paths=prim_paths,
    )


# ============================================================
# Surface sampling and descriptor computation
# ============================================================

def sample_surface_from_assembled_mesh(
    mesh: AssembledMesh,
    num_points: int = 3000,
    seed: int = 0,
) -> SurfaceSamples:
    points, face_indices, tri_vertices = _sample_points_on_triangles(
        mesh.vertices,
        mesh.faces,
        num_points=num_points,
        seed=seed,
    )
    normals = _triangle_face_normals(tri_vertices)
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)

    return SurfaceSamples(
        points=points,
        normals=normals,
        face_indices=face_indices,
        triangle_vertices=tri_vertices,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def _filter_samples_top_ratio(samples: SurfaceSamples, top_ratio: float) -> SurfaceSamples:
    """
    Keep only the points in the top `top_ratio` fraction of the z-axis range.
    The original bounding box is preserved so that z_ratio values stay
    consistent with the full-mesh coordinate system.

    E.g. top_ratio=0.30 keeps the top 30 % of the object height — enough to
    cover a rim while discarding the bulk of the body below.
    """
    z_threshold = samples.bbox_max[2] - top_ratio * (samples.bbox_max[2] - samples.bbox_min[2])
    mask = samples.points[:, 2] >= z_threshold
    if not np.any(mask):
        return samples  # nothing survives — return unfiltered
    return SurfaceSamples(
        points=samples.points[mask],
        normals=samples.normals[mask],
        face_indices=samples.face_indices[mask],
        triangle_vertices=samples.triangle_vertices[mask],
        bbox_min=samples.bbox_min,   # intentionally keep original bbox
        bbox_max=samples.bbox_max,
    )


def _filter_samples_bottom_ratio(samples: SurfaceSamples, bottom_ratio: float) -> SurfaceSamples:
    """
    Keep only the points in the bottom `bottom_ratio` fraction of the z-axis range.
    The original bounding box is preserved so that z_ratio values stay consistent
    with the full-mesh coordinate system.
    """
    z_threshold = samples.bbox_min[2] + bottom_ratio * (samples.bbox_max[2] - samples.bbox_min[2])
    mask = samples.points[:, 2] <= z_threshold
    if not np.any(mask):
        return samples
    return SurfaceSamples(
        points=samples.points[mask],
        normals=samples.normals[mask],
        face_indices=samples.face_indices[mask],
        triangle_vertices=samples.triangle_vertices[mask],
        bbox_min=samples.bbox_min,
        bbox_max=samples.bbox_max,
    )


def _filter_samples_exclude_bottom_clearance(samples: SurfaceSamples, clearance: float) -> SurfaceSamples:
    """
    Keep all points except those within `clearance` meters of the global minimum z.
    The original bounding box is preserved so descriptor coordinates stay consistent
    with the full object frame.
    """
    z_threshold = samples.bbox_min[2] + clearance
    mask = samples.points[:, 2] >= z_threshold
    if not np.any(mask):
        return samples
    return SurfaceSamples(
        points=samples.points[mask],
        normals=samples.normals[mask],
        face_indices=samples.face_indices[mask],
        triangle_vertices=samples.triangle_vertices[mask],
        bbox_min=samples.bbox_min,
        bbox_max=samples.bbox_max,
    )


def compute_shared_descriptors(
    samples: SurfaceSamples,
    k_neighbors: int = 32,
    perimeter_band_width: Optional[float] = None,
    access_probe_length: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute shared descriptors for all categories.

    Returned arrays are length N:
        z_ratio
        normal_variation
        perimeter_score
        accessibility_score
        support_asymmetry
        local_width
        side_score
        up_score
        down_score
        knn_idx  (N, k)
        knn_dist (N, k)
    """
    points = samples.points
    normals = samples.normals
    N = points.shape[0]

    knn_idx, knn_dist = _pairwise_knn(points, k=k_neighbors)

    bbox_min = samples.bbox_min
    bbox_max = samples.bbox_max
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-8)
    z_ratio = (points[:, 2] - bbox_min[2]) / bbox_size[2]

    # Normal variation = mean angular difference to neighbors
    neighbor_normals = normals[knn_idx]  # (N,k,3)
    dots = np.sum(neighbor_normals * normals[:, None, :], axis=2)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # radians
    normal_variation = angles.mean(axis=1)

    # Perimeter score from XY convex hull distance
    points_xy = points[:, :2]
    hull_xy = _convex_hull_2d(points_xy)
    dist_to_hull = _distance_to_convex_hull_2d(points_xy, hull_xy)

    if perimeter_band_width is None:
        xy_extent = np.linalg.norm((bbox_max - bbox_min)[:2])
        perimeter_band_width = max(0.01, 0.08 * xy_extent)

    perimeter_score = 1.0 - np.clip(dist_to_hull / max(perimeter_band_width, 1e-6), 0.0, 1.0)

    # Accessibility and support asymmetry:
    # Use outward radial direction in XY from object center.
    xy_center = 0.5 * (bbox_min[:2] + bbox_max[:2])
    vec_xy = points_xy - xy_center[None, :]
    vec_xy_norm = np.linalg.norm(vec_xy, axis=1, keepdims=True)
    outward_xy = np.zeros((N, 2), dtype=np.float64)
    valid = vec_xy_norm[:, 0] > 1e-8
    outward_xy[valid] = vec_xy[valid] / vec_xy_norm[valid]

    if access_probe_length is None:
        access_probe_length = max(0.02, 0.10 * np.linalg.norm(bbox_max - bbox_min))

    support_asymmetry = np.zeros(N, dtype=np.float64)
    accessibility_score = np.zeros(N, dtype=np.float64)
    local_width = np.zeros(N, dtype=np.float64)
    normal_variation_axis = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        nbr = knn_idx[i]
        nbr_pts = points[nbr]
        nbr_nrm = normals[nbr]                         # (k, 3)
        offsets = nbr_pts - points[i][None, :]

        out_dir3 = np.array([outward_xy[i, 0], outward_xy[i, 1], 0.0], dtype=np.float64)
        if np.linalg.norm(out_dir3) < 1e-8:
            out_dir3 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        proj = offsets @ out_dir3
        inward = np.sum(proj < 0.0)
        outward_cnt = np.sum(proj > 0.0)
        total = max(1, inward + outward_cnt)

        # High when there is more local support inward than outward.
        support_asymmetry[i] = (inward - outward_cnt) / total

        # Accessibility high if few neighbors occupy the outward probe direction.
        outward_mask = (proj > 0.0) & (proj < access_probe_length)
        accessibility_score[i] = 1.0 - (np.sum(outward_mask) / max(1, len(nbr)))

        # Local width estimate along outward/inward direction.
        if len(proj) > 0:
            local_width[i] = float(np.percentile(proj, 95) - np.percentile(proj, 5))
        else:
            local_width[i] = 0.0

        # Principal axis of normal variation in the neighborhood.
        # This is the direction normals change most across the k neighbors.
        # - At a rim (top edge): normals transition from horizontal (wall) to
        #   vertical (top face), so this axis lies between outward-XY and +Z
        #   and has a significant |z| component.
        # - At a vertical wall edge: normals rotate within the XY plane, so
        #   the axis is nearly horizontal (|z| ≈ 0).
        # Eigenvectors are sign-ambiguous, so always use abs when comparing.
        mn = nbr_nrm.mean(axis=0)
        centered_n = nbr_nrm - mn[None, :]
        cov_n = centered_n.T @ centered_n
        evals, evecs = np.linalg.eigh(cov_n)
        normal_variation_axis[i] = evecs[:, np.argmax(evals)]

    up_score = np.clip(normals[:, 2], 0.0, 1.0)
    down_score = np.clip(-normals[:, 2], 0.0, 1.0)
    side_score = 1.0 - np.abs(normals[:, 2])

    return {
        "z_ratio": z_ratio,
        "normal_variation": normal_variation,
        "dist_to_hull": dist_to_hull,
        "perimeter_score": perimeter_score,
        "accessibility_score": np.clip(accessibility_score, 0.0, 1.0),
        "support_asymmetry": np.clip((support_asymmetry + 1.0) * 0.5, 0.0, 1.0),  # map [-1,1] -> [0,1]
        "local_width": local_width,
        "normal_variation_axis": normal_variation_axis,
        "side_score": side_score,
        "up_score": up_score,
        "down_score": down_score,
        "knn_idx": knn_idx,
        "knn_dist": knn_dist,
    }


# ============================================================
# Category-specific proposal
# ============================================================

def _cat1_default_params(region_cfg: Dict[str, Any], samples: SurfaceSamples) -> Dict[str, Any]:
    """
    Convert config fields into explicit numeric parameters for cat1.
    """
    bbox_min = samples.bbox_min
    bbox_max = samples.bbox_max
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    xy_diag = float(np.linalg.norm((bbox_max - bbox_min)[:2]))

    top_band_ratio = float(region_cfg.get("top_band_ratio", 0.20))
    top_band_min = 1.0 - top_band_ratio

    pre_filter = region_cfg.get("pre_filter_top_ratio", None)
    if pre_filter is not None and float(pre_filter) < top_band_ratio:
        import warnings
        warnings.warn(
            f"cat1: pre_filter_top_ratio ({pre_filter}) < top_band_ratio ({top_band_ratio}). "
            "The pre-filter is tighter than the top-band gate — the z_ratio gate will never "
            "reject any surviving point. Either raise pre_filter_top_ratio or lower top_band_ratio.",
            stacklevel=3,
        )

    out = {
        "top_band_min": region_cfg.get("top_band_min", top_band_min),
        "min_segment_length": float(region_cfg.get("min_segment_length", 0.05)),
        "max_regions": int(region_cfg.get("max_regions", 20)),
        "perimeter_band_width": float(region_cfg.get("perimeter_band_width", max(0.01, 0.08 * xy_diag))),
        "access_probe_length": float(region_cfg.get("access_probe_length", max(0.02, 0.10 * diag))),
        "edge_normal_var_thresh": float(region_cfg.get("edge_normal_var_thresh", math.radians(15.0))),
        "grasp_width_min": float(region_cfg.get("grasp_width_min", 0.0)),
        "grasp_width_max": float(region_cfg.get("grasp_width_max", 0.08)),
        "anchor_nms_radius": float(region_cfg.get("anchor_nms_radius", max(0.015, 0.04 * diag))),
        "k_neighbors": int(region_cfg.get("k_neighbors", 32)),
        # Optional normal-direction filter: keep anchors whose surface normal is within
        # normal_filter_tol_deg of normal_filter_dir.  None = disabled.
        "normal_filter_dir": region_cfg.get("normal_filter_dir", None),
        "normal_filter_tol_deg": float(region_cfg.get("normal_filter_tol_deg", 30.0)),
        "weights": {
            "top": float(region_cfg.get("w_top", 1.0)),
            "edge": float(region_cfg.get("w_edge", 0.4)),
            "balance": float(region_cfg.get("w_balance", 1.5)),
            "perimeter": float(region_cfg.get("w_perimeter", 0.8)),
            "access": float(region_cfg.get("w_access", 0.8)),
            "width": float(region_cfg.get("w_width", 0.6)),
        },
    }
    return out


def _cat2_default_params(region_cfg: Dict[str, Any], samples: SurfaceSamples) -> Dict[str, Any]:
    """
    Convert config fields into explicit numeric parameters for cat2.
    cat2 mirrors cat1 anchor selection, but on the bottom band with -Z normals.
    """
    bbox_min = samples.bbox_min
    bbox_max = samples.bbox_max
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    xy_diag = float(np.linalg.norm((bbox_max - bbox_min)[:2]))

    out = {
        "bottom_clearance_from_min_z": float(region_cfg.get("bottom_clearance_from_min_z", 0.06)),
        "max_regions": int(region_cfg.get("max_regions", 20)),
        "perimeter_band_width": float(region_cfg.get("perimeter_band_width", max(0.01, 0.08 * xy_diag))),
        "access_probe_length": float(region_cfg.get("access_probe_length", max(0.02, 0.10 * diag))),
        "edge_normal_var_thresh": float(region_cfg.get("edge_normal_var_thresh", math.radians(15.0))),
        "grasp_width_min": float(region_cfg.get("grasp_width_min", 0.0)),
        "grasp_width_max": float(region_cfg.get("grasp_width_max", 0.08)),
        "anchor_nms_radius": float(region_cfg.get("anchor_nms_radius", max(0.015, 0.04 * diag))),
        "k_neighbors": int(region_cfg.get("k_neighbors", 32)),
        "normal_max_z": float(region_cfg.get("normal_max_z", 0.10)),
        "perimeter_min": float(region_cfg.get("perimeter_min", 0.1)),
        "accessibility_min": float(region_cfg.get("accessibility_min", 0.05)),
        "down_score_min": float(region_cfg.get("down_score_min", 0.0)),
        "z_outlier_band_width": float(region_cfg.get("z_outlier_band_width", 0.04)),
        "weights": {
            "down": float(region_cfg.get("w_down", 1.0)),
            "edge": float(region_cfg.get("w_edge", 0.4)),
            "balance": float(region_cfg.get("w_balance", 1.5)),
            "perimeter": float(region_cfg.get("w_perimeter", 0.8)),
            "access": float(region_cfg.get("w_access", 0.8)),
            "width": float(region_cfg.get("w_width", 0.6)),
        },
    }
    return out


def _cat3_default_params(region_cfg: Dict[str, Any], samples: SurfaceSamples) -> Dict[str, Any]:
    """
    Convert config fields into explicit numeric parameters for cat3.
    cat3 is a bimanual side-hold category, so we want stable side-band anchors
    that can later be paired on opposite sides at similar heights.
    """
    bbox_min = samples.bbox_min
    bbox_max = samples.bbox_max
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    xy_diag = float(np.linalg.norm((bbox_max - bbox_min)[:2]))

    default_width_max = max(0.10, min(0.22, 0.35 * xy_diag))
    out = {
        "side_height_range_ratio": list(region_cfg.get("side_height_range_ratio", [0.25, 0.75])),
        "max_regions": int(region_cfg.get("max_regions", 24)),
        "perimeter_band_width": float(region_cfg.get("perimeter_band_width", max(0.01, 0.08 * xy_diag))),
        "access_probe_length": float(region_cfg.get("access_probe_length", max(0.02, 0.10 * diag))),
        "smooth_angle_scale": float(region_cfg.get("smooth_angle_scale", math.radians(22.0))),
        "grasp_width_min": float(region_cfg.get("grasp_width_min", 0.015)),
        "grasp_width_max": float(region_cfg.get("grasp_width_max", default_width_max)),
        "anchor_nms_radius": float(region_cfg.get("anchor_nms_radius", max(0.018, 0.05 * diag))),
        "k_neighbors": int(region_cfg.get("k_neighbors", 48)),
        "side_score_min": float(region_cfg.get("side_score_min", 0.55)),
        "perimeter_min": float(region_cfg.get("perimeter_min", 0.05)),
        "accessibility_min": float(region_cfg.get("accessibility_min", 0.10)),
        "wrap_balance_min": float(region_cfg.get("wrap_balance_min", 0.35)),
        "uniform_azimuth_bins": int(region_cfg.get("uniform_azimuth_bins", 12)),
        "uniform_height_bins": int(region_cfg.get("uniform_height_bins", 4)),
        "weights": {
            "side": float(region_cfg.get("w_side", 1.20)),
            "smooth": float(region_cfg.get("w_smooth", 0.75)),
            "balance": float(region_cfg.get("w_balance", 1.00)),
            "perimeter": float(region_cfg.get("w_perimeter", 0.55)),
            "access": float(region_cfg.get("w_access", 1.00)),
            "width": float(region_cfg.get("w_width", 0.45)),
            "height": float(region_cfg.get("w_height", 0.45)),
        },
    }
    return out


def _cat4_default_params(region_cfg: Dict[str, Any], samples: SurfaceSamples) -> Dict[str, Any]:
    """
    Convert config fields into explicit numeric parameters for cat4.
    cat4 is a direct-USD convex-side hold scaffold inspired by the BODex-style
    one-hand convex grasp setting, without requiring convex piece assets.
    """
    bbox_min = samples.bbox_min
    bbox_max = samples.bbox_max
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    xy_diag = float(np.linalg.norm((bbox_max - bbox_min)[:2]))

    out = {
        "side_height_range_ratio": list(region_cfg.get("side_height_range_ratio", [0.18, 0.82])),
        "top_height_range_ratio": list(region_cfg.get("top_height_range_ratio", [0.72, 1.00])),
        "max_regions": int(region_cfg.get("max_regions", 40)),
        "perimeter_band_width": float(region_cfg.get("perimeter_band_width", max(0.01, 0.08 * xy_diag))),
        "access_probe_length": float(region_cfg.get("access_probe_length", max(0.02, 0.10 * diag))),
        "smooth_angle_scale": float(region_cfg.get("smooth_angle_scale", math.radians(25.0))),
        "grasp_width_min": float(region_cfg.get("grasp_width_min", 0.01)),
        "grasp_width_max": float(region_cfg.get("grasp_width_max", 0.16)),
        "anchor_nms_radius": float(region_cfg.get("anchor_nms_radius", max(0.018, 0.05 * diag))),
        "k_neighbors": int(region_cfg.get("k_neighbors", 48)),
        "top_grasp_enabled": bool(region_cfg.get("top_grasp_enabled", True)),
        "side_score_min": float(region_cfg.get("side_score_min", 0.45)),
        "top_normal_min_z": float(region_cfg.get("top_normal_min_z", 0.55)),
        "perimeter_min": float(region_cfg.get("perimeter_min", 0.05)),
        "accessibility_min": float(region_cfg.get("accessibility_min", 0.10)),
        "top_accessibility_min": float(region_cfg.get("top_accessibility_min", 0.05)),
        "support_min": float(region_cfg.get("support_min", 0.52)),
        "top_support_min": float(region_cfg.get("top_support_min", 0.35)),
        "uniform_azimuth_bins": int(region_cfg.get("uniform_azimuth_bins", 12)),
        "uniform_height_bins": int(region_cfg.get("uniform_height_bins", 3)),
        "top_uniform_azimuth_bins": int(region_cfg.get("top_uniform_azimuth_bins", 8)),
        "top_region_fraction": float(region_cfg.get("top_region_fraction", 0.35)),
        "min_top_regions_if_available": int(region_cfg.get("min_top_regions_if_available", 4)),
        "weights": {
            "side": float(region_cfg.get("w_side", 1.2)),
            "smooth": float(region_cfg.get("w_smooth", 0.9)),
            "convex": float(region_cfg.get("w_convex", 0.9)),
            "perimeter": float(region_cfg.get("w_perimeter", 0.8)),
            "access": float(region_cfg.get("w_access", 1.0)),
            "width": float(region_cfg.get("w_width", 0.5)),
            "height": float(region_cfg.get("w_height", 0.4)),
        },
        "top_weights": {
            "up": float(region_cfg.get("w_top_up", 1.25)),
            "smooth": float(region_cfg.get("w_top_smooth", 0.8)),
            "convex": float(region_cfg.get("w_top_convex", 0.7)),
            "access": float(region_cfg.get("w_top_access", 0.8)),
            "width": float(region_cfg.get("w_top_width", 0.5)),
            "height": float(region_cfg.get("w_top_height", 0.9)),
        },
    }
    return out


def _cat4_uniform_candidate_order(
    keep_indices: np.ndarray,
    score: np.ndarray,
    samples: SurfaceSamples,
    *,
    azimuth_bins: int,
    height_bins: int,
) -> np.ndarray:
    if len(keep_indices) <= 1:
        return keep_indices

    azimuth_bins = max(1, int(azimuth_bins))
    height_bins = max(1, int(height_bins))
    center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    z_min = float(samples.bbox_min[2])
    z_span = max(float(samples.bbox_max[2] - samples.bbox_min[2]), 1e-8)

    buckets: Dict[tuple[int, int], List[int]] = {}
    for idx in keep_indices.tolist():
        point = np.asarray(samples.points[int(idx)], dtype=np.float64)
        rel_xy = point[:2] - center_xy
        azimuth = float(np.arctan2(rel_xy[1], rel_xy[0]))
        if azimuth < 0.0:
            azimuth += 2.0 * np.pi
        az_bin = min(azimuth_bins - 1, int(np.floor(azimuth_bins * azimuth / (2.0 * np.pi))))
        z_ratio = float(np.clip((point[2] - z_min) / z_span, 0.0, 1.0 - 1e-9))
        h_bin = min(height_bins - 1, int(np.floor(height_bins * z_ratio)))
        buckets.setdefault((az_bin, h_bin), []).append(int(idx))

    for bucket_indices in buckets.values():
        bucket_indices.sort(key=lambda i: float(score[int(i)]), reverse=True)

    ordered_keys = sorted(buckets)
    ordered: List[int] = []
    while True:
        added = False
        for key in ordered_keys:
            bucket = buckets.get(key, [])
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added = True
        if not added:
            break
    return np.asarray(ordered, dtype=np.int64)


def _cat4_anchor_bin_key(
    point: np.ndarray,
    samples: SurfaceSamples,
    *,
    azimuth_bins: int,
    height_bins: int,
) -> tuple[int, int]:
    azimuth_bins = max(1, int(azimuth_bins))
    height_bins = max(1, int(height_bins))
    center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    z_min = float(samples.bbox_min[2])
    z_span = max(float(samples.bbox_max[2] - samples.bbox_min[2]), 1e-8)

    point = np.asarray(point, dtype=np.float64)
    rel_xy = point[:2] - center_xy
    azimuth = float(np.arctan2(rel_xy[1], rel_xy[0]))
    if azimuth < 0.0:
        azimuth += 2.0 * np.pi
    az_bin = min(azimuth_bins - 1, int(np.floor(azimuth_bins * azimuth / (2.0 * np.pi))))
    z_ratio = float(np.clip((point[2] - z_min) / z_span, 0.0, 1.0 - 1e-9))
    h_bin = min(height_bins - 1, int(np.floor(height_bins * z_ratio)))
    return az_bin, h_bin


def propose_cat1_anchors(
    samples: SurfaceSamples,
    descriptors: Dict[str, np.ndarray],
    region_cfg: Dict[str, Any],
) -> List[Anchor]:
    """
    cat1 = edge_rim

    Strategy:
    - top-band filtering
    - edge sharpness from neighborhood normal variation
    - rim/boundary proxy from support asymmetry + perimeter proximity
    - accessibility
    - local grasp width range
    - diversity pruning by NMS
    """
    params = _cat1_default_params(region_cfg, samples)

    pts = samples.points
    nrm = samples.normals
    z_ratio = descriptors["z_ratio"]
    normal_variation = descriptors["normal_variation"]
    perimeter_score = descriptors["perimeter_score"]
    accessibility_score = descriptors["accessibility_score"]
    support_asymmetry = descriptors["support_asymmetry"]
    local_width = descriptors["local_width"]
    knn_idx = descriptors["knn_idx"]

    top_mask = z_ratio >= params["top_band_min"]
    edge_score = np.clip(normal_variation / max(params["edge_normal_var_thresh"], 1e-6), 0.0, 1.0)

    # balance_score peaks when a point has roughly equal inward and outward neighbours
    # in the radial direction (support_asymmetry ≈ 0.5 after normalisation to [0,1]).
    # outer edge of rim → sa ≈ 1.0 → balance ≈ 0
    # mid-rim           → sa ≈ 0.5 → balance ≈ 1   ← desired
    # inner edge of rim → sa ≈ 0.0 → balance ≈ 0
    balance_score = 1.0 - np.abs(2.0 * support_asymmetry - 1.0)

    wmin = params["grasp_width_min"]
    wmax = params["grasp_width_max"]
    within_width = (local_width >= wmin) & (local_width <= wmax)

    width_score = np.zeros_like(local_width)
    width_score[within_width] = 1.0

    # Own-normal direction filter: keep only points whose surface normal is within
    # normal_filter_tol_deg of normal_filter_dir.  Applied as a hard gate over all
    # points before NMS so that off-direction points never compete for slots.
    fdir_raw = params["normal_filter_dir"]
    if fdir_raw is not None:
        fdir = _safe_normalize(np.array(fdir_raw, dtype=np.float64))
        cos_tol = math.cos(math.radians(params["normal_filter_tol_deg"]))
        normal_mask = (nrm @ fdir) >= cos_tol   # dot with own normal, no abs
    else:
        normal_mask = np.ones(len(pts), dtype=bool)

    # balance_score is primary: places anchors at the mid-rim rather than the
    # outer/inner rim edges where edge_score peaks.
    # edge_score kept as a secondary signal to bias within the rim zone.
    w = params["weights"]
    score = (
        w["top"] * top_mask.astype(np.float64) +
        w["edge"] * edge_score +
        w["balance"] * balance_score +
        w["perimeter"] * perimeter_score +
        w["access"] * accessibility_score +
        w["width"] * width_score
    )

    # Hard gate: top band + perimeter + accessibility + width + own-normal direction.
    hard_mask = (
        top_mask
        & (perimeter_score > 0.1)
        & (accessibility_score > 0.05)
        & within_width
        & normal_mask
    )
    score = score * hard_mask.astype(np.float64)

    valid_idx = np.where(score > 0.0)[0]
    if len(valid_idx) == 0:
        return []

    keep_idx = _nms_keep(
        points=pts[valid_idx],
        scores=score[valid_idx],
        radius=params["anchor_nms_radius"],
    )
    chosen = valid_idx[keep_idx]

    # Keep only top max_regions
    chosen = chosen[np.argsort(-score[chosen])]
    chosen = chosen[: params["max_regions"]]

    xy_center = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])

    anchors: List[Anchor] = []
    for idx in chosen:
        nbr_idx = knn_idx[idx]
        nbr_pts = pts[nbr_idx]

        frame_R = _compute_local_frame_cat2(
            anchor_point=pts[idx],
            neighbor_points=nbr_pts,
            object_center_xy=xy_center,
        )

        anchors.append(
            Anchor(
                category="cat1",
                point=pts[idx].copy(),
                normal=nrm[idx].copy(),
                score=float(score[idx]),
                frame_R=frame_R,
                support_indices=nbr_idx.copy(),
                metadata={
                    "z_ratio": float(z_ratio[idx]),
                    "edge_score": float(edge_score[idx]),
                    "balance_score": float(balance_score[idx]),
                    "perimeter_score": float(perimeter_score[idx]),
                    "accessibility_score": float(accessibility_score[idx]),
                    "local_width": float(local_width[idx]),
                },
            )
        )

    distribution_meta = _summarize_cat2_anchor_distribution(anchors, samples)
    if distribution_meta.get("cat2_anchor_distribution_mode") == "bottom_rim_line":
        line_axis_xy = np.asarray(distribution_meta["cat2_line_axis_xy"], dtype=np.float64)
        side_axis_xy = np.asarray(distribution_meta["cat2_line_side_axis_xy"], dtype=np.float64)
        line_center_xy = np.asarray(distribution_meta["cat2_line_center_xy"], dtype=np.float64)

        shared_x_axis = np.array([line_axis_xy[0], line_axis_xy[1], 0.0], dtype=np.float64)
        shared_y_axis = np.array([side_axis_xy[0], side_axis_xy[1], 0.0], dtype=np.float64)
        shared_z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        shared_x_axis = _safe_normalize(shared_x_axis)
        shared_y_axis = _safe_normalize(shared_y_axis)
        shared_x_axis = _safe_normalize(np.cross(shared_y_axis, shared_z_axis))
        shared_frame_R = np.stack([shared_x_axis, shared_y_axis, shared_z_axis], axis=1)

        for anchor in anchors:
            anchor.frame_R = shared_frame_R.copy()
            centered_xy = np.asarray(anchor.point[:2], dtype=np.float64) - line_center_xy
            anchor.metadata["cat2_line_coord"] = float(np.dot(centered_xy, line_axis_xy))
            anchor.metadata["cat2_line_side_coord"] = float(np.dot(centered_xy, side_axis_xy))

    for anchor in anchors:
        anchor.metadata.update(distribution_meta)

    return anchors


def propose_cat2_anchors(
    samples: SurfaceSamples,
    descriptors: Dict[str, np.ndarray],
    region_cfg: Dict[str, Any],
) -> List[Anchor]:
    """
    cat2 = bottom_band

    Mirror cat1 anchor proposal on the downward side:
    - use all points except the very bottom strip
    - edge sharpness from neighborhood normal variation
    - rim/boundary proxy from support asymmetry + perimeter proximity
    - accessibility
    - local grasp width range
    - allow XY and -Z normals, reject +Z
    """
    params = _cat2_default_params(region_cfg, samples)

    pts = samples.points
    nrm = samples.normals
    z_ratio = descriptors["z_ratio"]
    normal_variation = descriptors["normal_variation"]
    down_score = descriptors["down_score"]
    perimeter_score = descriptors["perimeter_score"]
    accessibility_score = descriptors["accessibility_score"]
    support_asymmetry = descriptors["support_asymmetry"]
    local_width = descriptors["local_width"]
    knn_idx = descriptors["knn_idx"]

    z_from_bottom = pts[:, 2] - samples.bbox_min[2]
    clearance_mask = z_from_bottom >= params["bottom_clearance_from_min_z"]
    edge_score = np.clip(normal_variation / max(params["edge_normal_var_thresh"], 1e-6), 0.0, 1.0)
    balance_score = 1.0 - np.abs(2.0 * support_asymmetry - 1.0)
    down_mask = down_score >= params["down_score_min"]

    wmin = params["grasp_width_min"]
    wmax = params["grasp_width_max"]
    within_width = (local_width >= wmin) & (local_width <= wmax)

    width_score = np.zeros_like(local_width)
    width_score[within_width] = 1.0

    normal_mask = nrm[:, 2] <= params["normal_max_z"]

    w = params["weights"]
    score = (
        w["down"] * down_score
        + w["edge"] * edge_score
        + w["balance"] * balance_score
        + w["perimeter"] * perimeter_score
        + w["access"] * accessibility_score
        + w["width"] * width_score
    )

    hard_mask = (
        clearance_mask
        & down_mask
        & (perimeter_score > params["perimeter_min"])
        & (accessibility_score > params["accessibility_min"])
        & within_width
        & normal_mask
    )
    score = score * hard_mask.astype(np.float64)

    valid_idx = np.where(score > 0.0)[0]
    if len(valid_idx) == 0:
        return []

    keep_idx = _nms_keep(
        points=pts[valid_idx],
        scores=score[valid_idx],
        radius=params["anchor_nms_radius"],
    )
    chosen = valid_idx[keep_idx]
    chosen = chosen[np.argsort(-score[chosen])]

    if len(chosen) > 1:
        chosen_z = z_from_bottom[chosen]
        chosen_score = score[chosen]
        z_keep = _best_scoring_band_mask(chosen_z, chosen_score, params["z_outlier_band_width"])
        if np.any(z_keep):
            chosen = chosen[z_keep]
            chosen = chosen[np.argsort(-score[chosen])]

    chosen = chosen[: params["max_regions"]]

    if len(chosen) > 0:
        chosen_z = z_from_bottom[chosen]
        z_band_center = float(np.median(chosen_z))
        z_band_min = float(np.min(chosen_z))
        z_band_max = float(np.max(chosen_z))
    else:
        z_band_center = 0.0
        z_band_min = 0.0
        z_band_max = 0.0

    xy_center = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    anchors: List[Anchor] = []
    for idx in chosen:
        nbr_idx = knn_idx[idx]
        nbr_pts = pts[nbr_idx]

        frame_R = _compute_local_frame_cat2(
            anchor_point=pts[idx],
            neighbor_points=nbr_pts,
            object_center_xy=xy_center,
        )

        anchors.append(
            Anchor(
                category="cat2",
                point=pts[idx].copy(),
                normal=nrm[idx].copy(),
                score=float(score[idx]),
                frame_R=frame_R,
                support_indices=nbr_idx.copy(),
                metadata={
                    "z_ratio": float(z_ratio[idx]),
                    "z_from_bottom": float(z_from_bottom[idx]),
                    "bottom_clearance_from_min_z": float(params["bottom_clearance_from_min_z"]),
                    "z_outlier_band_width": float(params["z_outlier_band_width"]),
                    "z_band_center": z_band_center,
                    "z_band_min": z_band_min,
                    "z_band_max": z_band_max,
                    "normal_z": float(nrm[idx, 2]),
                    "normal_max_z": float(params["normal_max_z"]),
                    "down_score": float(down_score[idx]),
                    "edge_score": float(edge_score[idx]),
                    "balance_score": float(balance_score[idx]),
                    "perimeter_score": float(perimeter_score[idx]),
                    "accessibility_score": float(accessibility_score[idx]),
                    "local_width": float(local_width[idx]),
                },
            )
        )

    return anchors


def propose_cat3_anchors(
    samples: SurfaceSamples,
    descriptors: Dict[str, np.ndarray],
    region_cfg: Dict[str, Any],
) -> List[Anchor]:
    """
    cat3 = side_wrap
    """
    params = _cat3_default_params(region_cfg, samples)

    pts = samples.points
    nrm = samples.normals
    z_ratio = descriptors["z_ratio"]
    normal_variation = descriptors["normal_variation"]
    perimeter_score = descriptors["perimeter_score"]
    accessibility_score = descriptors["accessibility_score"]
    support_asymmetry = descriptors["support_asymmetry"]
    local_width = descriptors["local_width"]
    side_score = descriptors["side_score"]
    knn_idx = descriptors["knn_idx"]

    side_height_lo, side_height_hi = [float(x) for x in params["side_height_range_ratio"]]
    if side_height_lo > side_height_hi:
        side_height_lo, side_height_hi = side_height_hi, side_height_lo

    side_height_mask = (z_ratio >= side_height_lo) & (z_ratio <= side_height_hi)
    width_mask = (local_width >= params["grasp_width_min"]) & (local_width <= params["grasp_width_max"])
    side_mask = side_score >= params["side_score_min"]
    perimeter_mask = perimeter_score >= params["perimeter_min"]
    access_mask = accessibility_score >= params["accessibility_min"]

    smooth_angle_scale = max(float(params["smooth_angle_scale"]), 1e-6)
    smooth_score = 1.0 - np.clip(normal_variation / smooth_angle_scale, 0.0, 1.0)
    wrap_balance_score = 1.0 - np.abs(2.0 * support_asymmetry - 1.0)
    wrap_balance_mask = wrap_balance_score >= params["wrap_balance_min"]

    side_height_mid = 0.5 * (side_height_lo + side_height_hi)
    side_height_half = max(0.5 * (side_height_hi - side_height_lo), 1e-6)
    side_height_score = 1.0 - np.clip(np.abs(z_ratio - side_height_mid) / side_height_half, 0.0, 1.0)

    width_center = 0.5 * (params["grasp_width_min"] + params["grasp_width_max"])
    width_half = max(0.5 * (params["grasp_width_max"] - params["grasp_width_min"]), 1e-6)
    width_score = 1.0 - np.clip(np.abs(local_width - width_center) / width_half, 0.0, 1.0)

    w = params["weights"]
    score = (
        w["side"] * side_score
        + w["smooth"] * smooth_score
        + w["balance"] * wrap_balance_score
        + w["perimeter"] * perimeter_score
        + w["access"] * accessibility_score
        + w["width"] * width_score
        + w["height"] * side_height_score
    )

    hard_mask = side_height_mask & width_mask & side_mask & perimeter_mask & access_mask & wrap_balance_mask
    if not np.any(hard_mask):
        hard_mask = side_height_mask & width_mask & side_mask & perimeter_mask & access_mask
    if not np.any(hard_mask):
        hard_mask = side_height_mask & width_mask & side_mask & perimeter_mask
    if not np.any(hard_mask):
        hard_mask = side_height_mask & side_mask

    valid_idx = np.where(hard_mask & (score > 0.0))[0]
    if len(valid_idx) == 0:
        return []

    keep_idx = _nms_keep(
        points=pts[valid_idx],
        scores=score[valid_idx],
        radius=params["anchor_nms_radius"],
    )
    chosen = valid_idx[keep_idx]
    chosen = _cat4_uniform_candidate_order(
        chosen,
        score,
        samples,
        azimuth_bins=params["uniform_azimuth_bins"],
        height_bins=params["uniform_height_bins"],
    )
    chosen = chosen[: params["max_regions"]]

    xy_center = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    anchors: List[Anchor] = []
    for idx in chosen:
        nbr_idx = knn_idx[idx]
        frame_R = _compute_local_frame_cat3(
            anchor_point=pts[idx],
            anchor_normal=nrm[idx],
            object_center_xy=xy_center,
        )
        rel_xy = np.asarray(pts[idx][:2], dtype=np.float64) - xy_center
        azimuth = float(np.arctan2(rel_xy[1], rel_xy[0]))

        anchors.append(
            Anchor(
                category="cat3",
                point=pts[idx].copy(),
                normal=nrm[idx].copy(),
                score=float(score[idx]),
                frame_R=frame_R,
                support_indices=nbr_idx.copy(),
                metadata={
                    "z_ratio": float(z_ratio[idx]),
                    "normal_variation": float(normal_variation[idx]),
                    "perimeter_score": float(perimeter_score[idx]),
                    "accessibility_score": float(accessibility_score[idx]),
                    "support_asymmetry": float(support_asymmetry[idx]),
                    "wrap_balance_score": float(wrap_balance_score[idx]),
                    "local_width": float(local_width[idx]),
                    "side_score": float(side_score[idx]),
                    "side_height_score": float(side_height_score[idx]),
                    "cat3_azimuth_rad": azimuth,
                },
            )
        )

    return anchors


def propose_cat4_anchors(
    samples: SurfaceSamples,
    descriptors: Dict[str, np.ndarray],
    region_cfg: Dict[str, Any],
) -> List[Anchor]:
    """
    cat4 = convex_hold

    Strategy:
    - use the direct USD mesh, not convex piece assets
    - generate both side-band and top-cap anchor families
    - keep side anchors distributed around the object body
    - keep top anchors distributed around the top cap for small-object top grasps
    - attach a local frame plus grasp-mode metadata for later contact / seed generation
    """
    params = _cat4_default_params(region_cfg, samples)

    pts = samples.points
    nrm = samples.normals
    z_ratio = descriptors["z_ratio"]
    normal_variation = descriptors["normal_variation"]
    perimeter_score = descriptors["perimeter_score"]
    accessibility_score = descriptors["accessibility_score"]
    support_asymmetry = descriptors["support_asymmetry"]
    local_width = descriptors["local_width"]
    side_score = descriptors["side_score"]
    knn_idx = descriptors["knn_idx"]

    side_height_lo, side_height_hi = [float(x) for x in params["side_height_range_ratio"]]
    if side_height_lo > side_height_hi:
        side_height_lo, side_height_hi = side_height_hi, side_height_lo

    top_height_lo, top_height_hi = [float(x) for x in params["top_height_range_ratio"]]
    if top_height_lo > top_height_hi:
        top_height_lo, top_height_hi = top_height_hi, top_height_lo

    side_height_mask = (z_ratio >= side_height_lo) & (z_ratio <= side_height_hi)
    top_height_mask = (z_ratio >= top_height_lo) & (z_ratio <= top_height_hi)
    width_mask = (local_width >= params["grasp_width_min"]) & (local_width <= params["grasp_width_max"])
    side_mask = side_score >= params["side_score_min"]
    perimeter_mask = perimeter_score >= params["perimeter_min"]
    access_mask = accessibility_score >= params["accessibility_min"]
    support_mask = support_asymmetry >= params["support_min"]

    smooth_angle_scale = max(float(params["smooth_angle_scale"]), 1e-6)
    smooth_score = 1.0 - np.clip(normal_variation / smooth_angle_scale, 0.0, 1.0)
    convex_score = np.clip(0.5 * perimeter_score + 0.5 * support_asymmetry, 0.0, 1.0)
    side_height_mid = 0.5 * (side_height_lo + side_height_hi)
    side_height_half = max(0.5 * (side_height_hi - side_height_lo), 1e-6)
    side_height_score = 1.0 - np.clip(np.abs(z_ratio - side_height_mid) / side_height_half, 0.0, 1.0)
    top_height_score = np.clip((z_ratio - top_height_lo) / max(top_height_hi - top_height_lo, 1e-6), 0.0, 1.0)

    width_center = 0.5 * (params["grasp_width_min"] + params["grasp_width_max"])
    width_half = max(0.5 * (params["grasp_width_max"] - params["grasp_width_min"]), 1e-6)
    width_score = 1.0 - np.clip(np.abs(local_width - width_center) / width_half, 0.0, 1.0)

    side_weights = params["weights"]
    side_candidate_score = (
        side_weights["side"] * side_score
        + side_weights["smooth"] * smooth_score
        + side_weights["convex"] * convex_score
        + side_weights["perimeter"] * perimeter_score
        + side_weights["access"] * accessibility_score
        + side_weights["width"] * width_score
        + side_weights["height"] * side_height_score
    )

    side_candidate_mask = side_height_mask & width_mask & side_mask & perimeter_mask & access_mask & support_mask
    if not np.any(side_candidate_mask):
        side_candidate_mask = side_height_mask & width_mask & side_mask & perimeter_mask
    if not np.any(side_candidate_mask):
        side_candidate_mask = side_height_mask & side_mask

    side_candidate_indices = np.flatnonzero(side_candidate_mask)
    side_keep_indices = np.zeros((0,), dtype=np.int64)
    if len(side_candidate_indices) > 0:
        side_keep_local = _nms_keep(
            pts[side_candidate_indices],
            side_candidate_score[side_candidate_indices],
            radius=params["anchor_nms_radius"],
        )
        side_keep_indices = side_candidate_indices[side_keep_local]
        side_keep_indices = side_keep_indices[np.argsort(-side_candidate_score[side_keep_indices])]
        side_keep_indices = _cat4_uniform_candidate_order(
            side_keep_indices,
            side_candidate_score,
            samples,
            azimuth_bins=params["uniform_azimuth_bins"],
            height_bins=params["uniform_height_bins"],
        )

    top_keep_indices = np.zeros((0,), dtype=np.int64)
    top_candidate_score = np.zeros_like(side_candidate_score)
    top_normal_score = np.clip(nrm[:, 2], 0.0, 1.0)
    if bool(params["top_grasp_enabled"]):
        top_weights = params["top_weights"]
        top_candidate_score = (
            top_weights["up"] * top_normal_score
            + top_weights["smooth"] * smooth_score
            + top_weights["convex"] * convex_score
            + top_weights["access"] * accessibility_score
            + top_weights["width"] * width_score
            + top_weights["height"] * top_height_score
        )

        top_normal_mask = top_normal_score >= params["top_normal_min_z"]
        top_access_mask = accessibility_score >= params["top_accessibility_min"]
        top_support_mask = support_asymmetry >= params["top_support_min"]
        top_candidate_mask = top_height_mask & width_mask & top_normal_mask & top_access_mask & top_support_mask
        if not np.any(top_candidate_mask):
            top_candidate_mask = top_height_mask & width_mask & top_normal_mask & top_access_mask
        if not np.any(top_candidate_mask):
            top_candidate_mask = top_height_mask & top_normal_mask

        top_candidate_indices = np.flatnonzero(top_candidate_mask)
        if len(top_candidate_indices) > 0:
            top_keep_local = _nms_keep(
                pts[top_candidate_indices],
                top_candidate_score[top_candidate_indices],
                radius=params["anchor_nms_radius"],
            )
            top_keep_indices = top_candidate_indices[top_keep_local]
            top_keep_indices = top_keep_indices[np.argsort(-top_candidate_score[top_keep_indices])]
            top_keep_indices = _cat4_uniform_candidate_order(
                top_keep_indices,
                top_candidate_score,
                samples,
                azimuth_bins=params["top_uniform_azimuth_bins"],
                height_bins=1,
            )

    if len(side_keep_indices) == 0 and len(top_keep_indices) == 0:
        return []

    max_regions = max(1, int(params["max_regions"]))
    min_top_regions = max(0, int(params["min_top_regions_if_available"]))
    top_target = 0
    if len(top_keep_indices) > 0:
        top_target = int(round(max_regions * float(params["top_region_fraction"])))
        top_target = max(min_top_regions, top_target)
        top_target = min(max_regions, len(top_keep_indices), max(1, top_target))
    side_target = min(len(side_keep_indices), max_regions - top_target)
    chosen_side = side_keep_indices[:side_target].tolist()
    chosen_top = top_keep_indices[:top_target].tolist()

    remaining = max_regions - (len(chosen_side) + len(chosen_top))
    if remaining > 0:
        leftover: List[tuple[float, int, str]] = []
        for idx in side_keep_indices[len(chosen_side):].tolist():
            leftover.append((float(side_candidate_score[int(idx)]), int(idx), "side"))
        for idx in top_keep_indices[len(chosen_top):].tolist():
            leftover.append((float(top_candidate_score[int(idx)]), int(idx), "top"))
        leftover.sort(key=lambda item: item[0], reverse=True)
        for _, idx, mode in leftover[:remaining]:
            if mode == "top":
                chosen_top.append(int(idx))
            else:
                chosen_side.append(int(idx))

    keep_entries = [("side", int(idx)) for idx in chosen_side] + [("top", int(idx)) for idx in chosen_top]

    chosen_bin_counts: Dict[tuple[str, int, int], int] = {}
    for mode, idx in keep_entries:
        bin_key = _cat4_anchor_bin_key(
            pts[int(idx)],
            samples,
            azimuth_bins=(
                params["top_uniform_azimuth_bins"] if mode == "top" else params["uniform_azimuth_bins"]
            ),
            height_bins=(1 if mode == "top" else params["uniform_height_bins"]),
        )
        chosen_bin_counts[(mode, int(bin_key[0]), int(bin_key[1]))] = (
            chosen_bin_counts.get((mode, int(bin_key[0]), int(bin_key[1])), 0) + 1
        )

    occupied_az_bins = sorted({int(key[1]) for key in chosen_bin_counts if key[0] == "side"})
    occupied_h_bins = sorted({int(key[2]) for key in chosen_bin_counts if key[0] == "side"})
    occupied_top_az_bins = sorted({int(key[1]) for key in chosen_bin_counts if key[0] == "top"})
    distribution_summary = {
        "cat4_uniform_azimuth_bins": int(params["uniform_azimuth_bins"]),
        "cat4_uniform_height_bins": int(params["uniform_height_bins"]),
        "cat4_top_uniform_azimuth_bins": int(params["top_uniform_azimuth_bins"]),
        "cat4_top_grasp_enabled": bool(params["top_grasp_enabled"]),
        "cat4_num_selected_anchors": int(len(keep_entries)),
        "cat4_num_selected_side_anchors": int(len(chosen_side)),
        "cat4_num_selected_top_anchors": int(len(chosen_top)),
        "cat4_num_occupied_azimuth_bins": int(len(occupied_az_bins)),
        "cat4_num_occupied_height_bins": int(len(occupied_h_bins)),
        "cat4_occupied_azimuth_bins": occupied_az_bins,
        "cat4_occupied_height_bins": occupied_h_bins,
        "cat4_num_occupied_top_azimuth_bins": int(len(occupied_top_az_bins)),
        "cat4_occupied_top_azimuth_bins": occupied_top_az_bins,
    }

    object_center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    anchors: List[Anchor] = []
    for mode, idx in keep_entries:
        nbr = knn_idx[idx]
        az_bin, h_bin = _cat4_anchor_bin_key(
            pts[idx],
            samples,
            azimuth_bins=(
                params["top_uniform_azimuth_bins"] if mode == "top" else params["uniform_azimuth_bins"]
            ),
            height_bins=(1 if mode == "top" else params["uniform_height_bins"]),
        )
        frame_R = _compute_local_frame_cat4(
            anchor_point=pts[idx],
            anchor_normal=nrm[idx],
            object_center_xy=object_center_xy,
        )
        anchor_score = float(top_candidate_score[idx]) if mode == "top" else float(side_candidate_score[idx])
        anchors.append(
            Anchor(
                category="cat4",
                point=pts[idx],
                normal=nrm[idx],
                score=anchor_score,
                frame_R=frame_R,
                support_indices=np.asarray(nbr, dtype=np.int64),
                metadata={
                    "z_ratio": float(z_ratio[idx]),
                    "anchor_azimuth_deg": float(
                        np.rad2deg(
                            np.arctan2(
                                float(pts[idx][1] - object_center_xy[1]),
                                float(pts[idx][0] - object_center_xy[0]),
                            )
                        )
                    ),
                    "anchor_azimuth_bin": int(az_bin),
                    "anchor_height_bin": int(h_bin),
                    "anchor_bin_count": int(chosen_bin_counts.get((mode, az_bin, h_bin), 0)),
                    "normal_variation": float(normal_variation[idx]),
                    "perimeter_score": float(perimeter_score[idx]),
                    "accessibility_score": float(accessibility_score[idx]),
                    "support_asymmetry": float(support_asymmetry[idx]),
                    "local_width": float(local_width[idx]),
                    "side_score": float(side_score[idx]),
                    "top_normal_score": float(top_normal_score[idx]),
                    "smooth_score": float(smooth_score[idx]),
                    "convex_score": float(convex_score[idx]),
                    "height_score": float(top_height_score[idx] if mode == "top" else side_height_score[idx]),
                    "cat4_grasp_mode": mode,
                    "anchor_source": (
                        "direct_usd_convex_top" if mode == "top" else "direct_usd_convex_side"
                    ),
                    **distribution_summary,
                },
            )
        )

    return anchors


# ============================================================
# Public proposal entry points
# ============================================================

def propose_anchors_from_assembled_mesh(
    mesh: AssembledMesh,
    category: Category,
    region_cfg: Dict[str, Any],
    num_surface_points: int = 3000,
    seed: int = 0,
) -> List[Anchor]:
    samples = sample_surface_from_assembled_mesh(
        mesh=mesh,
        num_points=num_surface_points,
        seed=seed,
    )

    # Optional: discard the lower portion of the object before running the
    # expensive kNN / descriptor computation.  This lets you sample very
    # densely (e.g. 100 000 points) and then focus effort on the rim band.
    pre_filter_top = region_cfg.get("pre_filter_top_ratio", None)
    pre_filter_bottom = region_cfg.get("pre_filter_bottom_ratio", None)
    pre_filter_bottom_clearance = region_cfg.get("pre_filter_bottom_clearance", None)
    if pre_filter_top is not None:
        samples = _filter_samples_top_ratio(samples, float(pre_filter_top))
    elif pre_filter_bottom is not None:
        samples = _filter_samples_bottom_ratio(samples, float(pre_filter_bottom))
    elif pre_filter_bottom_clearance is not None:
        samples = _filter_samples_exclude_bottom_clearance(samples, float(pre_filter_bottom_clearance))

    # Descriptor params
    k_neighbors = int(region_cfg.get("k_neighbors", 32))
    perimeter_band_width = region_cfg.get("perimeter_band_width", None)
    access_probe_length = region_cfg.get("access_probe_length", None)

    descriptors = compute_shared_descriptors(
        samples=samples,
        k_neighbors=k_neighbors,
        perimeter_band_width=perimeter_band_width,
        access_probe_length=access_probe_length,
    )

    return propose_anchors_from_samples(
        samples=samples,
        descriptors=descriptors,
        category=category,
        region_cfg=region_cfg,
    )


def propose_anchors_from_samples(
    *,
    samples: SurfaceSamples,
    descriptors: Dict[str, np.ndarray],
    category: Category,
    region_cfg: Dict[str, Any],
) -> List[Anchor]:
    if category == "cat1":
        return propose_cat1_anchors(samples, descriptors, region_cfg)
    if category == "cat2":
        return propose_cat2_anchors(samples, descriptors, region_cfg)
    if category == "cat3":
        return propose_cat3_anchors(samples, descriptors, region_cfg)
    if category == "cat4":
        return propose_cat4_anchors(samples, descriptors, region_cfg)

    raise ValueError(f"Unsupported category: {category}")


def propose_anchors_from_usd(
    usd_path: str | Path,
    category: Category,
    region_cfg: Dict[str, Any],
    num_surface_points: int = 3000,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Anchor]:
    mesh = load_assembled_mesh_from_usd(
        usd_path=usd_path,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    return propose_anchors_from_assembled_mesh(
        mesh=mesh,
        category=category,
        region_cfg=region_cfg,
        num_surface_points=num_surface_points,
        seed=seed,
    )


# ============================================================
# Standalone debug CLI
# ============================================================

def _default_region_cfg_for_category(category: Category) -> Dict[str, Any]:
    if category == "cat1":
        return {
            "mode": "top_rim_or_edge",
            "top_band_ratio": 0.20,
            "prefer_boundary_points": True,
            "prefer_high_normal_variation": True,
            "prefer_outer_perimeter": True,
            "require_accessibility": True,
            "min_segment_length": 0.05,
            "max_regions": 20,
            # Optional tunables:
            # "edge_normal_var_thresh": math.radians(15.0),
            # "grasp_width_min": 0.005,
            # "grasp_width_max": 0.08,
            # "anchor_nms_radius": 0.03,
        }
    if category == "cat2":
        return {
            "mode": "bottom_band",
            "bottom_clearance_from_min_z": 0.06,
            "pre_filter_bottom_clearance": 0.06,
            "max_regions": 20,
            "anchor_nms_radius": 0.01,
            "k_neighbors": 64,
            "grasp_width_min": 0.004,
            "grasp_width_max": 0.08,
            "accessibility_min": 0.05,
            "perimeter_min": 0.10,
            "down_score_min": 0.0,
            "z_outlier_band_width": 0.04,
            "normal_max_z": 0.10,
        }
    if category == "cat3":
        return {
            "mode": "side_band",
            "side_height_range_ratio": [0.25, 0.75],
            "require_wrap_suitability": True,
            "require_accessibility": True,
            "min_segment_area": 0.01,
            "max_regions": 20,
            "anchor_nms_radius": 0.02,
            "k_neighbors": 48,
            "grasp_width_min": 0.015,
            "grasp_width_max": 0.20,
            "side_score_min": 0.55,
            "perimeter_min": 0.05,
            "accessibility_min": 0.10,
            "wrap_balance_min": 0.35,
            "uniform_azimuth_bins": 12,
            "uniform_height_bins": 4,
        }
    if category == "cat4":
        return {
            "mode": "convex_hold",
            "side_height_range_ratio": [0.18, 0.82],
            "top_height_range_ratio": [0.72, 1.00],
            "max_regions": 40,
            "anchor_nms_radius": 0.02,
            "k_neighbors": 48,
            "grasp_width_min": 0.01,
            "grasp_width_max": 0.16,
            "top_grasp_enabled": True,
            "side_score_min": 0.45,
            "top_normal_min_z": 0.55,
            "perimeter_min": 0.05,
            "accessibility_min": 0.10,
            "top_accessibility_min": 0.05,
            "support_min": 0.52,
            "top_support_min": 0.35,
        }
    raise ValueError(category)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Debug object-side anchor proposal from USD.")
    parser.add_argument("--object_usd", type=Path, required=True)
    parser.add_argument("--category", type=str, choices=["cat1", "cat2", "cat3", "cat4"], required=True)
    parser.add_argument("--root_prim_path", type=str, default=None)
    parser.add_argument("--num_surface_points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = _default_region_cfg_for_category(args.category)  # placeholder config
    anchors = propose_anchors_from_usd(
        usd_path=args.object_usd,
        category=args.category,  # type: ignore[arg-type]
        region_cfg=cfg,
        num_surface_points=args.num_surface_points,
        root_prim_path=args.root_prim_path,
        seed=args.seed,
    )

    print(f"Found {len(anchors)} anchors for {args.category}")
    for i, a in enumerate(anchors[:10]):
        print(f"[{i}] score={a.score:.4f}, point={a.point.tolist()}, meta={a.metadata}")


if __name__ == "__main__":
    _main()
