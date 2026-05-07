"""
find_rim_anchors.py
===================
Geometry-only EEF pair finder for bi-gripper grasp generation.

Pipeline
--------
1. load_mesh_from_usd  — load object mesh from USD
2. find_rim_anchors    — sample surface, score, and filter candidate EEF points on the rim
3. select_opposite_pairs — project anchors to XY by dropping Z, classify each
                           anchor by which XY bbox edge family it is closest to,
                           then pair opposite-side anchors within that edge family

Public API
----------
    pairs = get_eef_pairs(vertices, faces, ...)
    # pairs : List[Tuple[anchor_a, anchor_b]]

Each anchor dict contains:
    position       : [x, y, z]
    normal         : [nx, ny, nz]  surface normal  (= eef_z axis)
    edge_tangent   : [tx, ty, tz]  local rim direction estimated from XY neighbours
    eef_R          : 3×3 rotation matrix, columns = [eef_x, eef_y, eef_z]
    eef_quaternion : [w, x, y, z]
    score          : float
    metadata       : {z_ratio, edge_score, balance_score,
                      perimeter_score, accessibility_score, local_width}

EEF frame axes
--------------
    x — local rim tangent estimated from the projected XY neighbourhood
    y — gripper finger direction, perpendicular to the rim edge
    z — surface normal; the runtime later flips this so gripper local z approaches along −normal

Usage
-----
    python find_rim_anchors.py --object_usd /path/to/Object.usd --visualize --headless
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from isaacsim import SimulationApp


# ============================================================
# Global constants — tune here, applies to all objects
# ============================================================

# Surface sampling
NUM_SURFACE_POINTS      = 100000
SEED                    = 0

# Rim filter geometry
TOP_BAND_RATIO          = 0.20   # fraction of object height defining the rim zone (from top)
PRE_FILTER_TOP_RATIO    = 0.30   # discard bottom (1-ratio) fraction before kNN
NORMAL_FILTER_TOL_DEG   = 60.0   # surface normal must be within this many degrees of +z
K_NEIGHBORS             = 64
GRASP_WIDTH_MAX         = 0.06   # max rim width the gripper can straddle (metres)

# Scoring weights
W_TOP                   = 1.0
W_EDGE                  = 0.4
W_BALANCE               = 1.5
W_PERIMETER             = 0.8
W_ACCESS                = 0.8

# Anchor output
NMS_RADIUS              = 0.005  # non-maximum suppression radius (metres)
MAX_ANCHORS             = 500

# Pair selection
N_CANDIDATES            = 10      # opposite-side candidates paired with each anchor
MAX_PAIRS               = 3000

# CLI summary
TOP_K                   = 10


# ============================================================
# Geometry utilities
# ============================================================

def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n >= eps else v * 0.0


def _gf_matrix_to_np(mat) -> np.ndarray:
    """USD GfMatrix4d → 4×4 numpy (column-vector convention)."""
    out = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = mat[i][j]
    return out.T   # USD row-vector → column-vector


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float64)
    homog = np.concatenate([points, ones], axis=1)
    return (T @ homog.T).T[:, :3]


def _fan_triangulate(face_counts: Sequence[int], face_indices: Sequence[int]) -> np.ndarray:
    tris: List[Tuple[int, int, int]] = []
    cursor = 0
    for c in face_counts:
        if c >= 3:
            poly = face_indices[cursor: cursor + c]
            for i in range(1, c - 1):
                tris.append((poly[0], poly[i], poly[i + 1]))
        cursor += c
    return np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _triangle_face_normals(triangles: np.ndarray) -> np.ndarray:
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e1, e2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    out = np.zeros_like(cross)
    valid = norms[:, 0] > 1e-12
    out[valid] = cross[valid] / norms[valid]
    return out


def _sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface sampling. Returns (points, normals)."""
    rng = np.random.default_rng(seed)
    areas = _triangle_areas(vertices, faces)
    total = areas.sum()
    if total <= 0:
        raise ValueError("Mesh has zero surface area.")
    probs = areas / total
    chosen = rng.choice(len(faces), size=num_points, replace=True, p=probs)
    tri = vertices[faces[chosen]]   # (N, 3, 3)
    u = rng.random(num_points)
    v = rng.random(num_points)
    su = np.sqrt(u)
    pts = (tri[:, 0] * (1 - su)[:, None]
           + tri[:, 1] * (su * (1 - v))[:, None]
           + tri[:, 2] * (su * v)[:, None])
    nrm = _triangle_face_normals(tri)
    return pts, nrm


def _knn(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree
    k = min(k, max(1, len(points) - 1))
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=k + 1, workers=-1)
    return idx[:, 1:], dists[:, 1:]


def _convex_hull_2d(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.unique(pts_xy, axis=0)
    if len(pts) <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _dist_to_hull(pts_xy: np.ndarray, hull: np.ndarray) -> np.ndarray:
    if len(hull) == 0:
        return np.full(len(pts_xy), np.inf)
    if len(hull) == 1:
        return np.linalg.norm(pts_xy - hull[0], axis=1)

    def seg_dist(p, a, b):
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return float(np.linalg.norm(p - a))
        t = np.clip(float(np.dot(p - a, ab) / ab2), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))

    dists = np.full(len(pts_xy), np.inf)
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        d = np.array([seg_dist(p, a, b) for p in pts_xy])
        dists = np.minimum(dists, d)
    return dists


def _nms(points: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep: List[int] = []
    for idx in order:
        if not keep or np.all(np.linalg.norm(points[keep] - points[idx], axis=1) > radius):
            keep.append(int(idx))
    return np.asarray(keep, dtype=np.int64)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Returns [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _estimate_edge_tangent_xy(local_xy: np.ndarray, fallback_xy: np.ndarray) -> np.ndarray:
    """Estimate an undirected rim tangent from a local XY neighbourhood."""
    if local_xy.ndim != 2 or local_xy.shape[0] == 0:
        tangent_xy = np.array(fallback_xy, dtype=np.float64)
    else:
        centered = local_xy - local_xy.mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        if np.linalg.norm(cov) < 1e-12:
            tangent_xy = np.array(fallback_xy, dtype=np.float64)
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            tangent_xy = eigvecs[:, int(np.argmax(eigvals))]
            if np.dot(tangent_xy, fallback_xy) < 0.0:
                tangent_xy = -tangent_xy

    tangent = np.array([float(tangent_xy[0]), float(tangent_xy[1]), 0.0], dtype=np.float64)
    tangent = _safe_normalize(tangent)
    if np.linalg.norm(tangent) < 1e-8:
        tangent = _safe_normalize(np.array([float(fallback_xy[0]), float(fallback_xy[1]), 0.0], dtype=np.float64))
    return tangent


def _frame_from_normal_and_tangent(normal: np.ndarray,
                                   tangent: np.ndarray,
                                   fallback_radial: np.ndarray) -> np.ndarray:
    """Build an orthonormal EEF frame from a surface normal and edge tangent.

    The stored anchor frame uses the local X axis along the rim tangent.
    bi_gripper_grasp_gen.py later flips the frame 180 deg about local X, which
    preserves X and makes the gripper local Y axis perpendicular to the edge.
    """
    eef_z = _safe_normalize(normal)
    eef_x = tangent - np.dot(tangent, eef_z) * eef_z
    eef_x = _safe_normalize(eef_x)

    if np.linalg.norm(eef_x) < 1e-8:
        fallback_x = fallback_radial - np.dot(fallback_radial, eef_z) * eef_z
        fallback_x = _safe_normalize(fallback_x)
        if np.linalg.norm(fallback_x) < 1e-8:
            helper = np.array([1.0, 0.0, 0.0]) if abs(eef_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            fallback_x = _safe_normalize(np.cross(helper, eef_z))
        eef_x = fallback_x

    eef_y = _safe_normalize(np.cross(eef_z, eef_x))
    eef_x = _safe_normalize(np.cross(eef_y, eef_z))
    return np.stack([eef_x, eef_y, eef_z], axis=1)


# ============================================================
# USD mesh assembly
# ============================================================

def load_mesh_from_usd(
    usd_path: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble all Mesh prims under root_prim into one mesh in object-local frame.
    Returns (vertices, faces).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    if root_prim_path:
        root = stage.GetPrimAtPath(root_prim_path)
        if not root.IsValid():
            raise ValueError(f"Invalid root prim: {root_prim_path}")
    else:
        root = stage.GetDefaultPrim()
        if not root or not root.IsValid():
            children = list(stage.GetPseudoRoot().GetChildren())
            if not children:
                raise RuntimeError("No traversable root in USD.")
            root = children[0]

    exclude = set(exclude_prim_paths or [])
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_w = _gf_matrix_to_np(xfc.GetLocalToWorldTransform(root))
    root_w_inv = np.linalg.inv(root_w)

    all_verts, all_faces = [], []
    offset = 0

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.GetPath().pathString in exclude:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        fvi = mesh.GetFaceVertexIndicesAttr().Get()
        if pts is None or fvc is None or fvi is None:
            continue
        verts = np.asarray(pts, dtype=np.float64)
        tris = _fan_triangulate(fvc, fvi)
        if len(verts) == 0 or len(tris) == 0:
            continue
        mesh_w = _gf_matrix_to_np(xfc.GetLocalToWorldTransform(prim))
        verts = _transform_points(_transform_points(verts, mesh_w), root_w_inv)
        all_verts.append(verts)
        all_faces.append(tris + offset)
        offset += len(verts)

    del stage

    if not all_verts:
        raise RuntimeError("No valid Mesh prims found.")
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


# ============================================================
# Step 1 — rim anchor candidates
# ============================================================

def find_rim_anchors(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_surface_points: int = NUM_SURFACE_POINTS,
    seed: int = SEED,
    top_band_ratio: float = TOP_BAND_RATIO,
    pre_filter_top_ratio: float = PRE_FILTER_TOP_RATIO,
    normal_filter_tol_deg: float = NORMAL_FILTER_TOL_DEG,
    k_neighbors: int = K_NEIGHBORS,
    grasp_width_max: float = GRASP_WIDTH_MAX,
    w_top: float = W_TOP,
    w_edge: float = W_EDGE,
    w_balance: float = W_BALANCE,
    w_perimeter: float = W_PERIMETER,
    w_access: float = W_ACCESS,
    nms_radius: float = NMS_RADIUS,
    max_anchors: int = MAX_ANCHORS,
) -> List[Dict[str, Any]]:
    """Sample and score EEF anchor candidates on the object rim."""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-8)

    pts, nrm = _sample_surface(vertices, faces, num_surface_points, seed)

    z_thresh = bbox_max[2] - pre_filter_top_ratio * bbox_size[2]
    mask = pts[:, 2] >= z_thresh
    if not np.any(mask):
        mask = np.ones(len(pts), dtype=bool)
    pts, nrm = pts[mask], nrm[mask]

    N = len(pts)
    if N < k_neighbors + 1:
        return []

    z_ratio = (pts[:, 2] - bbox_min[2]) / bbox_size[2]
    top_band_min = 1.0 - top_band_ratio

    knn_idx, _ = _knn(pts, k_neighbors)

    # Edge score — normal variation among neighbours
    nbr_nrm = nrm[knn_idx]
    dots = np.clip(np.sum(nbr_nrm * nrm[:, None, :], axis=2), -1.0, 1.0)
    edge_score = np.clip(np.arccos(dots).mean(axis=1) / max(math.radians(15.0), 1e-6), 0.0, 1.0)

    # Perimeter score — proximity to XY convex hull
    pts_xy = pts[:, :2]
    hull_xy = _convex_hull_2d(pts_xy)
    band_w = max(0.01, 0.08 * float(np.linalg.norm((bbox_max - bbox_min)[:2])))
    perimeter_score = 1.0 - np.clip(_dist_to_hull(pts_xy, hull_xy) / band_w, 0.0, 1.0)

    # Support asymmetry and accessibility
    xy_center = 0.5 * (bbox_min[:2] + bbox_max[:2])
    vec_xy = pts_xy - xy_center
    vec_norm = np.linalg.norm(vec_xy, axis=1, keepdims=True)
    outward_xy = np.where(vec_norm > 1e-8, vec_xy / vec_norm, np.array([[1.0, 0.0]]))

    access_len = max(0.02, 0.10 * float(np.linalg.norm(bbox_max - bbox_min)))
    support_asym = np.zeros(N)
    accessibility = np.zeros(N)
    local_width = np.zeros(N)

    for i in range(N):
        offsets = pts[knn_idx[i]] - pts[i]
        out3 = np.array([outward_xy[i, 0], outward_xy[i, 1], 0.0])
        proj = offsets @ out3
        inward = np.sum(proj < 0.0)
        outward_cnt = np.sum(proj > 0.0)
        support_asym[i] = (inward - outward_cnt) / max(1, inward + outward_cnt)
        accessibility[i] = 1.0 - np.sum((proj > 0.0) & (proj < access_len)) / max(1, len(proj))
        local_width[i] = float(np.percentile(proj, 95) - np.percentile(proj, 5))

    sa_norm = np.clip((support_asym + 1.0) * 0.5, 0.0, 1.0)
    balance_score = 1.0 - np.abs(2.0 * sa_norm - 1.0)
    accessibility_score = np.clip(accessibility, 0.0, 1.0)

    cos_tol = math.cos(math.radians(normal_filter_tol_deg))
    top_mask = z_ratio >= top_band_min
    hard_mask = (
        top_mask
        & (perimeter_score > 0.1)
        & (accessibility_score > 0.05)
        & (local_width <= grasp_width_max)
        & (nrm[:, 2] >= cos_tol)
    )

    score = (
        w_top * top_mask.astype(float)
        + w_edge * edge_score
        + w_balance * balance_score
        + w_perimeter * perimeter_score
        + w_access * accessibility_score
    ) * hard_mask.astype(float)

    valid = np.where(score > 0.0)[0]
    if len(valid) == 0:
        return []

    keep = _nms(pts[valid], score[valid], nms_radius)
    chosen = valid[keep]
    chosen = chosen[np.argsort(-score[chosen])][:max_anchors]

    anchors = []
    for rank, idx in enumerate(chosen):
        surf_n = nrm[idx]
        radial = np.array([outward_xy[idx, 0], outward_xy[idx, 1], 0.0], dtype=np.float64)
        fallback_tangent_xy = np.array([-outward_xy[idx, 1], outward_xy[idx, 0]], dtype=np.float64)
        local_xy = np.vstack([pts_xy[idx], pts_xy[knn_idx[idx]]])
        edge_tangent = _estimate_edge_tangent_xy(local_xy, fallback_tangent_xy)
        R = _frame_from_normal_and_tangent(surf_n, edge_tangent, radial)

        anchors.append({
            "rank": rank,
            "score": float(score[idx]),
            "position": pts[idx].tolist(),
            "normal": nrm[idx].tolist(),
            "edge_tangent": edge_tangent.tolist(),
            "eef_R": R.tolist(),
            "eef_quaternion": _rotation_matrix_to_quaternion(R).tolist(),
            "metadata": {
                "z_ratio": float(z_ratio[idx]),
                "edge_score": float(edge_score[idx]),
                "balance_score": float(balance_score[idx]),
                "perimeter_score": float(perimeter_score[idx]),
                "accessibility_score": float(accessibility_score[idx]),
                "local_width": float(local_width[idx]),
            },
        })

    return anchors


# ============================================================
# Step 2 — opposite pair selection
# ============================================================

def select_opposite_pairs(
    anchors: List[Dict[str, Any]],
    center_xy: Optional[np.ndarray] = None,
    bbox_min_xy: Optional[np.ndarray] = None,
    bbox_max_xy: Optional[np.ndarray] = None,
    n_candidates: int = N_CANDIDATES,
    max_pairs: int = MAX_PAIRS,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Pair anchors by edge family in the mapped XY plane.

    For each anchor A:
    - drop Z and classify the XY point by the nearest XY bbox edge family
    - preserve the along-edge coordinate (x for x-parallel edges, y for y-parallel)
    - among candidates on the opposite side of the object centre, prefer the one
      with the most similar along-edge coordinate and the largest separation in
      the across-edge coordinate

    Returns deduplicated (anchor_a, anchor_b) pairs ordered by primary-anchor rank
    and edge-opposition quality.
    """
    if len(anchors) < 2:
        return []

    xy = np.array([a["position"][:2] for a in anchors], dtype=np.float64)

    if center_xy is None:
        center_xy = xy.mean(axis=0)
    if bbox_min_xy is None:
        bbox_min_xy = np.min(xy, axis=0)
    if bbox_max_xy is None:
        bbox_max_xy = np.max(xy, axis=0)
    bbox_min_xy = np.asarray(bbox_min_xy, dtype=np.float64)
    bbox_max_xy = np.asarray(bbox_max_xy, dtype=np.float64)

    scores = np.array([a["score"] for a in anchors], dtype=np.float64)
    tangent_xy = np.array([a["edge_tangent"][:2] for a in anchors], dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent_xy, axis=1, keepdims=True)
    tangent_xy = tangent_xy / np.where(tangent_norm > 1e-8, tangent_norm, 1.0)

    anchor_bbox_min_xy = np.min(xy, axis=0)
    anchor_bbox_max_xy = np.max(xy, axis=0)
    projected_xy = xy.copy()
    projected_along_axis_idx = np.zeros(len(anchors), dtype=np.int64)
    for idx in range(len(anchors)):
        dist_to_vertical_edges = min(
            abs(xy[idx, 0] - anchor_bbox_min_xy[0]),
            abs(xy[idx, 0] - anchor_bbox_max_xy[0]),
        )
        dist_to_horizontal_edges = min(
            abs(xy[idx, 1] - anchor_bbox_min_xy[1]),
            abs(xy[idx, 1] - anchor_bbox_max_xy[1]),
        )
        if dist_to_horizontal_edges <= dist_to_vertical_edges:
            projected_along_axis_idx[idx] = 0
        else:
            projected_along_axis_idx[idx] = 1

    seen: set = set()
    pair_list: List[Tuple[int, int, float, float, float, float]] = []

    for i in range(len(anchors)):
        candidates = np.arange(len(anchors), dtype=np.int64)
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        edge_axis = tangent_xy[i]
        along_axis_idx = int(projected_along_axis_idx[i])
        across_axis_idx = 1 - along_axis_idx
        along_i = float(projected_xy[i, along_axis_idx])
        across_i = float(projected_xy[i, across_axis_idx])
        across_center = float(center_xy[across_axis_idx])
        candidate_family = projected_along_axis_idx[candidates]
        same_family_mask = candidate_family == along_axis_idx
        if np.any(same_family_mask):
            candidates = candidates[same_family_mask]
        if len(candidates) == 0:
            continue
        tangent_alignment = np.abs(tangent_xy[candidates] @ edge_axis)
        along_diff = np.abs(projected_xy[candidates, along_axis_idx] - along_i)
        across_sep = np.abs(projected_xy[candidates, across_axis_idx] - across_i)
        across_balance = np.abs(
            (projected_xy[candidates, across_axis_idx] - across_center) + (across_i - across_center)
        )
        opposite_mask = (
            (projected_xy[candidates, across_axis_idx] - across_center) * (across_i - across_center) < 0.0
        )
        if np.any(opposite_mask):
            candidates = candidates[opposite_mask]
            tangent_alignment = tangent_alignment[opposite_mask]
            along_diff = along_diff[opposite_mask]
            across_sep = across_sep[opposite_mask]
            across_balance = across_balance[opposite_mask]
        if len(candidates) == 0:
            continue
        order = np.lexsort(
            (
                -scores[candidates],
                1.0 - tangent_alignment,
                across_balance,
                -across_sep,
                along_diff,
            )
        )
        top = candidates[order[:n_candidates]]
        for j in top:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen.add(key)
                pair_list.append(
                    (
                        i,
                        int(j),
                        float(abs(projected_xy[int(j), along_axis_idx] - along_i)),
                        float(abs(projected_xy[int(j), across_axis_idx] - across_i)),
                        float(
                            abs((projected_xy[int(j), across_axis_idx] - across_center) + (across_i - across_center))
                        ),
                        float(np.abs(np.dot(tangent_xy[i], tangent_xy[int(j)]))),
                        scores[i] + scores[int(j)],
                    )
                )

    pair_list.sort(
        key=lambda x: (
            int(x[0]),
            float(x[2]),
            -float(x[3]),
            float(x[4]),
            -float(x[5]),
            -float(x[6]),
        )
    )
    return [(anchors[i], anchors[j]) for i, j, _, _, _, _, _ in pair_list[:max_pairs]]


# ============================================================
# Public entry point
# ============================================================

def get_eef_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    # anchor params
    num_surface_points: int = NUM_SURFACE_POINTS,
    seed: int = SEED,
    top_band_ratio: float = TOP_BAND_RATIO,
    pre_filter_top_ratio: float = PRE_FILTER_TOP_RATIO,
    normal_filter_tol_deg: float = NORMAL_FILTER_TOL_DEG,
    k_neighbors: int = K_NEIGHBORS,
    grasp_width_max: float = GRASP_WIDTH_MAX,
    nms_radius: float = NMS_RADIUS,
    max_anchors: int = MAX_ANCHORS,
    # pair params
    center_xy: Optional[np.ndarray] = None,
    n_candidates: int = N_CANDIDATES,
    max_pairs: int = MAX_PAIRS,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Full pipeline: mesh → EEF pairs for bi-gripper pose generation.

    Returns a list of (anchor_a, anchor_b) tuples, one anchor per gripper arm,
    ordered by primary-anchor rank and mirrored-pair quality.
    """
    anchors = find_rim_anchors(
        vertices=vertices,
        faces=faces,
        num_surface_points=num_surface_points,
        seed=seed,
        top_band_ratio=top_band_ratio,
        pre_filter_top_ratio=pre_filter_top_ratio,
        normal_filter_tol_deg=normal_filter_tol_deg,
        k_neighbors=k_neighbors,
        grasp_width_max=grasp_width_max,
        nms_radius=nms_radius,
        max_anchors=max_anchors,
    )
    bbox_min_xy = np.min(vertices[:, :2], axis=0)
    bbox_max_xy = np.max(vertices[:, :2], axis=0)
    return select_opposite_pairs(
        anchors,
        center_xy=center_xy,
        bbox_min_xy=bbox_min_xy,
        bbox_max_xy=bbox_max_xy,
        n_candidates=n_candidates,
        max_pairs=max_pairs,
    )


# ============================================================
# Visualisation (Isaac Sim)
# ============================================================

def visualize(
    object_usd: Path,
    pts_display: np.ndarray,
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> None:
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Sdf, Vt, UsdGeom

    stage = omni.usd.get_context().get_stage()

    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")

    for p in ["/World/Object", "/World/DebugSurface", "/World/DebugPairs"]:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            stage.RemovePrim(Sdf.Path(p))

    add_reference_to_stage(usd_path=str(object_usd.resolve()), prim_path="/World/Object")

    # Surface point cloud
    max_pts = 3000
    disp = pts_display if len(pts_display) <= max_pts else pts_display[
        np.random.default_rng(0).choice(len(pts_display), max_pts, replace=False)
    ]
    surf = UsdGeom.Points.Define(stage, "/World/DebugSurface")
    surf.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p.tolist()) for p in disp]))
    surf.GetWidthsAttr().Set(Vt.FloatArray([0.004] * len(disp)))
    surf.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.4, 0.8, 1.0)]))

    # Pairs — anchor_a in red, anchor_b in green
    UsdGeom.Xform.Define(stage, "/World/DebugPairs")
    for k, (a, b) in enumerate(pairs):
        for label, anchor, colour in [("a", a, (1.0, 0.2, 0.2)), ("b", b, (0.2, 1.0, 0.2))]:
            sp = UsdGeom.Sphere.Define(stage, f"/World/DebugPairs/pair_{k:04d}_{label}")
            sp.CreateRadiusAttr(0.008)
            UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*anchor["position"]))
            sp.CreateDisplayColorAttr().Set([Gf.Vec3f(*colour)])

    print(f"Visualised {len(disp)} surface points and {len(pairs)} EEF pairs.")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    global simulation_app

    parser = argparse.ArgumentParser(description="EEF pair finder for bi-gripper grasp generation.")
    parser.add_argument("--object_usd", type=Path, required=True)
    parser.add_argument("--root_prim_path", type=str, default=None)
    parser.add_argument("--exclude_prim_paths", nargs="*", default=None)

    parser.add_argument("--num_surface_points", type=int, default=NUM_SURFACE_POINTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top_band_ratio", type=float, default=TOP_BAND_RATIO)
    parser.add_argument("--pre_filter_top_ratio", type=float, default=PRE_FILTER_TOP_RATIO)
    parser.add_argument("--normal_filter_tol_deg", type=float, default=NORMAL_FILTER_TOL_DEG)
    parser.add_argument("--k_neighbors", type=int, default=K_NEIGHBORS)
    parser.add_argument("--grasp_width_max", type=float, default=GRASP_WIDTH_MAX)
    parser.add_argument("--nms_radius", type=float, default=NMS_RADIUS)
    parser.add_argument("--max_anchors", type=int, default=MAX_ANCHORS)
    parser.add_argument("--n_candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--max_pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--top_k", type=int, default=TOP_K)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": args.headless})

    try:
        import omni.kit.app

        object_usd = args.object_usd.resolve()
        vertices, faces = load_mesh_from_usd(
            usd_path=object_usd,
            root_prim_path=args.root_prim_path,
            exclude_prim_paths=args.exclude_prim_paths,
        )
        print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces.")

        pairs = get_eef_pairs(
            vertices=vertices,
            faces=faces,
            num_surface_points=args.num_surface_points,
            seed=args.seed,
            top_band_ratio=args.top_band_ratio,
            pre_filter_top_ratio=args.pre_filter_top_ratio,
            normal_filter_tol_deg=args.normal_filter_tol_deg,
            k_neighbors=args.k_neighbors,
            grasp_width_max=args.grasp_width_max,
            nms_radius=args.nms_radius,
            max_anchors=args.max_anchors,
            n_candidates=args.n_candidates,
            max_pairs=args.max_pairs,
        )
        print(f"Found {len(pairs)} EEF pairs.")

        summary = {
            "object_usd": str(object_usd),
            "num_vertices": int(len(vertices)),
            "num_faces": int(len(faces)),
            "num_pairs": len(pairs),
            "pairs_top_k": [
                {"anchor_a": a, "anchor_b": b} for a, b in pairs[: args.top_k]
            ],
        }
        print(json.dumps(summary, indent=2))

        if args.visualize:
            pts_display, _ = _sample_surface(vertices, faces, min(args.num_surface_points, 100_000), args.seed)
            bbox_max, bbox_min = vertices.max(axis=0), vertices.min(axis=0)
            z_thresh = bbox_max[2] - args.pre_filter_top_ratio * (bbox_max[2] - bbox_min[2])
            pts_display = pts_display[pts_display[:, 2] >= z_thresh]

            visualize(object_usd, pts_display, pairs)

            app = omni.kit.app.get_app()
            print("Visualisation ready. Close the window to exit.")
            while simulation_app.is_running():
                app.update()

    finally:
        pass


if __name__ == "__main__":
    main()
