"""
bi_gripper_pipeline.py
======================
End-to-end bi-gripper grasp generation pipeline in a single file.

Pipeline
--------
1. load_mesh_from_usd  — assemble all Mesh prims into one numpy mesh
2. sample_surface      — area-weighted surface point sampling
3. find_rim_anchors    — score and filter EEF candidates on the object rim
4. select_opposite_pairs — classify anchors in XY by the nearest bbox edge family
                           and pair opposite-side anchors within that edge family
5. Physics validation  — Isaac Sim grid-cloner parallel evaluation:
       step-back → approach (both grippers) → close (both) → lift (both) → Z-rise check
6. Save results        — JSON to <object_dir>/Annotation/bi_gripper_grasps.json

EEF frame convention
---------------------
    x — radial outward projected onto the plane ⊥ to surface normal  (finger axis)
    y — rim tangent  cross(z, x)
    z — surface normal; gripper approaches along −z

Output JSON format
------------------
{
  "type": "<object_type>",
  "bottom_center": [x, y, z],
  "functional_grasp": {
    "body": [
      {"left": [x, y, z, qw, qx, qy, qz],
       "right": [x, y, z, qw, qx, qy, qz]},
      ...
    ]
  }
}

Usage
-----
    python bi_gripper_pipeline.py --object_usd /path/to/Object.usd --object_type Bin
"""

# ── argparse must come before SimulationApp ────────────────────────────────
import argparse
_parser = argparse.ArgumentParser(description="Bi-gripper grasp generation pipeline.")
_parser.add_argument("--object_usd",          type=str, required=True)
_parser.add_argument("--object_type",         type=str, default="Bin")
_parser.add_argument("--root_prim_path",      type=str, default=None)
_parser.add_argument("--exclude_prim_paths",  nargs="*", default=None)
_args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# ── standard imports ───────────────────────────────────────────────────────
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from isaacsim.core.cloner import GridCloner
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import get_timeline_interface
import isaacsim.replicator.grasping.transform_utils as transform_utils

timeline = get_timeline_interface()


# ============================================================
# Global constants — tune here, applies to all objects
# ============================================================

# Gripper asset
_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent

def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()

GRIPPER_USD = _path_from_env("BIGRIPPER_GRIPPER_USD", _WORKSPACE_ROOT / "Flying_hand_probe_pro.usd")

# ── Surface sampling ──────────────────────────────────────────────────────
NUM_SURFACE_POINTS     = 100_000
SEED                   = 0

# ── Rim filter ────────────────────────────────────────────────────────────
TOP_BAND_RATIO         = 0.20    # fraction of object height defining rim zone (from top)
PRE_FILTER_TOP_RATIO   = 0.30    # discard bottom (1-ratio) fraction before kNN
NORMAL_FILTER_TOL_DEG  = 60.0    # surface normal must be within this many °  of +z
K_NEIGHBORS            = 64
GRASP_WIDTH_MAX        = 0.06    # max rim width the gripper can straddle (m)

# ── Scoring weights ───────────────────────────────────────────────────────
W_TOP        = 1.0
W_EDGE       = 0.4
W_BALANCE    = 1.5
W_PERIMETER  = 0.8
W_ACCESS     = 0.8

# ── Anchor output ─────────────────────────────────────────────────────────
NMS_RADIUS   = 0.005   # non-maximum suppression radius (m)
MAX_ANCHORS  = 500

# ── Pair selection ────────────────────────────────────────────────────────
N_CANDIDATES = 10       # opposite-side candidates paired with each anchor
MAX_PAIRS    = 3000

# ── Grid cloner ───────────────────────────────────────────────────────────
NUM_COPIES      = 200
CLONE_SPACING   = 3.0
GROUND_Z        = -10.0
ENV_BASE_PATH   = "/World/Envs"
ENV_ROOT_PREFIX = f"{ENV_BASE_PATH}/env"
PHYSICS_SCENE_PATH = "/World/physicsScene"

# ── Approach ──────────────────────────────────────────────────────────────
APPROACH_DISTANCE = 0.20   # step back along surface normal (m)
APPROACH_STEPS    = 80

# ── Close ─────────────────────────────────────────────────────────────────
CLOSE_STEPS   = 64
FINGER_OPEN   = 0.04
FINGER_CLOSE  = 0.0

# ── Lift ──────────────────────────────────────────────────────────────────
LIFT_DISTANCE   = 0.15   # m
LIFT_STEPS      = 80
LIFT_SUCCESS_Z  = 0.04   # object must rise at least this much (m) to count

# ── Physics material ──────────────────────────────────────────────────────
HIGH_STATIC_FRICTION  = 2.0
HIGH_DYNAMIC_FRICTION = 2.0


# ============================================================
# USD path helpers
# ============================================================
def env_path(i):   return f"{ENV_ROOT_PREFIX}_{i}"
def obj_wrap(i):   return f"{env_path(i)}/Object"
def obj_ref(i):    return f"{env_path(i)}/Object/ref"
def gripl_wrap(i): return f"{env_path(i)}/GripperL"
def gripl_ref(i):  return f"{env_path(i)}/GripperL/ref"
def gripr_wrap(i): return f"{env_path(i)}/GripperR"
def gripr_ref(i):  return f"{env_path(i)}/GripperR/ref"

def _finger_joints(grip_ref_path: str) -> List[str]:
    return [
        f"{grip_ref_path}/panda_hand/panda_finger_joint1",
        f"{grip_ref_path}/panda_hand/panda_finger_joint2",
    ]


# ============================================================
# ── STAGE 1: Geometry utilities (pure numpy / pxr) ──────────
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
    return out.T


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float64)
    return (T @ np.concatenate([points, ones], axis=1).T).T[:, :3]


def _fan_triangulate(face_counts: Sequence[int],
                     face_indices: Sequence[int]) -> np.ndarray:
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


def _sample_surface(vertices: np.ndarray, faces: np.ndarray,
                    num_points: int, seed: int = 0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface sampling. Returns (points, normals)."""
    rng = np.random.default_rng(seed)
    areas = _triangle_areas(vertices, faces)
    total = areas.sum()
    if total <= 0:
        raise ValueError("Mesh has zero surface area.")
    probs = areas / total
    chosen = rng.choice(len(faces), size=num_points, replace=True, p=probs)
    tri = vertices[faces[chosen]]
    u, v = rng.random(num_points), rng.random(num_points)
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
        if not keep or np.all(
                np.linalg.norm(points[keep] - points[idx], axis=1) > radius):
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


# ============================================================
# ── STAGE 2: USD mesh assembly ───────────────────────────────
# ============================================================

def load_mesh_from_usd(
    usd_path: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble all Mesh prims under root_prim into one mesh in object-local frame.
    Returns (vertices, faces). Uses pure pxr — no omni.usd context needed."""
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

    all_verts, all_faces, offset = [], [], 0
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
        tris  = _fan_triangulate(fvc, fvi)
        if len(verts) == 0 or len(tris) == 0:
            continue
        mesh_w = _gf_matrix_to_np(xfc.GetLocalToWorldTransform(prim))
        verts  = _transform_points(_transform_points(verts, mesh_w), root_w_inv)
        all_verts.append(verts)
        all_faces.append(tris + offset)
        offset += len(verts)

    del stage
    if not all_verts:
        raise RuntimeError("No valid Mesh prims found.")
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


# ============================================================
# ── STAGE 3: Rim anchor candidates ──────────────────────────
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
    bbox_min  = vertices.min(axis=0)
    bbox_max  = vertices.max(axis=0)
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

    z_ratio    = (pts[:, 2] - bbox_min[2]) / bbox_size[2]
    top_band_min = 1.0 - top_band_ratio
    knn_idx, _ = _knn(pts, k_neighbors)

    # Edge score — normal variation among neighbours
    nbr_nrm    = nrm[knn_idx]
    dots       = np.clip(np.sum(nbr_nrm * nrm[:, None, :], axis=2), -1.0, 1.0)
    edge_score = np.clip(
        np.arccos(dots).mean(axis=1) / max(math.radians(15.0), 1e-6), 0.0, 1.0)

    # Perimeter score — proximity to XY convex hull
    pts_xy        = pts[:, :2]
    hull_xy       = _convex_hull_2d(pts_xy)
    band_w        = max(0.01, 0.08 * float(np.linalg.norm((bbox_max - bbox_min)[:2])))
    perimeter_score = 1.0 - np.clip(_dist_to_hull(pts_xy, hull_xy) / band_w, 0.0, 1.0)

    # Support asymmetry and accessibility
    xy_center  = 0.5 * (bbox_min[:2] + bbox_max[:2])
    vec_xy     = pts_xy - xy_center
    vec_norm   = np.linalg.norm(vec_xy, axis=1, keepdims=True)
    outward_xy = np.where(vec_norm > 1e-8, vec_xy / vec_norm, np.array([[1.0, 0.0]]))
    access_len = max(0.02, 0.10 * float(np.linalg.norm(bbox_max - bbox_min)))

    support_asym  = np.zeros(N)
    accessibility = np.zeros(N)
    local_width   = np.zeros(N)
    for i in range(N):
        offsets    = pts[knn_idx[i]] - pts[i]
        out3       = np.array([outward_xy[i, 0], outward_xy[i, 1], 0.0])
        proj       = offsets @ out3
        inward     = np.sum(proj < 0.0)
        outward_cnt = np.sum(proj > 0.0)
        support_asym[i]  = (inward - outward_cnt) / max(1, inward + outward_cnt)
        accessibility[i] = 1.0 - np.sum(
            (proj > 0.0) & (proj < access_len)) / max(1, len(proj))
        local_width[i]   = float(np.percentile(proj, 95) - np.percentile(proj, 5))

    sa_norm            = np.clip((support_asym + 1.0) * 0.5, 0.0, 1.0)
    balance_score      = 1.0 - np.abs(2.0 * sa_norm - 1.0)
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
        w_top       * top_mask.astype(float)
        + w_edge    * edge_score
        + w_balance * balance_score
        + w_perimeter * perimeter_score
        + w_access  * accessibility_score
    ) * hard_mask.astype(float)

    valid = np.where(score > 0.0)[0]
    if len(valid) == 0:
        return []

    keep   = _nms(pts[valid], score[valid], nms_radius)
    chosen = valid[keep]
    chosen = chosen[np.argsort(-score[chosen])][:max_anchors]

    anchors = []
    for rank, idx in enumerate(chosen):
        surf_n = nrm[idx]
        eef_z  = _safe_normalize(surf_n)
        radial = np.array([outward_xy[idx, 0], outward_xy[idx, 1], 0.0])
        eef_x  = _safe_normalize(radial - np.dot(radial, eef_z) * eef_z)
        eef_y  = _safe_normalize(np.cross(eef_z, eef_x))
        eef_x  = _safe_normalize(np.cross(eef_y, eef_z))
        R      = np.stack([eef_x, eef_y, eef_z], axis=1)
        anchors.append({
            "rank":           rank,
            "score":          float(score[idx]),
            "position":       pts[idx].tolist(),
            "normal":         nrm[idx].tolist(),
            "eef_R":          R.tolist(),
            "eef_quaternion": _rotation_matrix_to_quaternion(R).tolist(),
            "metadata": {
                "z_ratio":             float(z_ratio[idx]),
                "edge_score":          float(edge_score[idx]),
                "balance_score":       float(balance_score[idx]),
                "perimeter_score":     float(perimeter_score[idx]),
                "accessibility_score": float(accessibility_score[idx]),
                "local_width":         float(local_width[idx]),
            },
        })
    return anchors


# ============================================================
# ── STAGE 4: Opposite-side pair selection ───────────────────
# ============================================================

def select_opposite_pairs(
    anchors: List[Dict[str, Any]],
    center_xy: Optional[np.ndarray] = None,
    bbox_min_xy: Optional[np.ndarray] = None,
    bbox_max_xy: Optional[np.ndarray] = None,
    n_candidates: int = N_CANDIDATES,
    max_pairs: int = MAX_PAIRS,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Pair anchors by edge family in XY.

    For x-parallel edges, preserve x and choose the opposite-side anchor with the
    largest y separation; for y-parallel edges, preserve y and choose the opposite-
    side anchor with the largest x separation. Edge family is decided from the
    anchor's XY projection against the anchor-cloud bbox, not from a radial
    centre projection. Returns deduplicated pairs ordered by primary-anchor rank
    and edge-opposition quality."""
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
# ── STAGE 5: Physics validation helpers ─────────────────────
# ============================================================

async def _step_sim(n: int = 1):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def _setup_physics_scene(stage):
    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")
    if not prim.HasAPI(UsdPhysics.Scene):
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    else:
        scene = UsdPhysics.Scene(prim)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx = PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(65536)
    physx.CreateGpuTotalAggregatePairsCapacityAttr().Set(65536)


def _apply_convex_decomp(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    count = 0
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexDecomposition)
        count += 1
    print(f"[INFO]   Convex decomp applied to {count} meshes under {root_path}")


def _create_physics_material(stage, mat_path: str) -> UsdShade.Material:
    parent = "/".join(mat_path.split("/")[:-1])
    if parent and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent)
    mat = UsdShade.Material.Define(stage, mat_path)
    pm  = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(HIGH_STATIC_FRICTION)
    pm.CreateDynamicFrictionAttr().Set(HIGH_DYNAMIC_FRICTION)
    pm.CreateRestitutionAttr().Set(0.0)
    return mat


def _bind_physics_material(stage, root_path: str, mat: UsdShade.Material):
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not p.IsA(UsdGeom.Mesh):
            continue
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(p)
        api = UsdShade.MaterialBindingAPI(p)
        try:
            api.Bind(mat, materialPurpose="physics")
        except Exception:
            api.Bind(mat)


def _make_object_rigid(stage, wrapper_path: str):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(True)
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb = PhysxSchema.PhysxRigidBodyAPI(prim)
    rb.CreateDisableGravityAttr().Set(True)
    rb.CreateContactSlopCoefficientAttr().Set(2.0)


def _make_kinematic_gripper(stage, wrapper_path: str):
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    rb.CreateKinematicEnabledAttr().Set(True)


def _set_gripper_pose(stage, wrapper_path: str,
                      world_pos: np.ndarray, quat_wxyz: np.ndarray,
                      env_world_pos: np.ndarray):
    """Set kinematic gripper pose (world → env-local, GridCloner is translation-only)."""
    local = world_pos - env_world_pos
    prim  = stage.GetPrimAtPath(wrapper_path)
    if not prim.IsValid():
        return
    xf   = UsdGeom.Xformable(prim)
    ops  = xf.GetOrderedXformOps()
    t_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    r_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient),    None)
    if t_op is None: t_op = xf.AddTranslateOp()
    if r_op is None: r_op = xf.AddOrientOp()
    t_op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))
    w, x, y, z = (float(v) for v in quat_wxyz)
    r_op.Set(Gf.Quatd(w, x, y, z))


def _set_obj_gravity(stage, wrapper_path: str, disable: bool):
    prim = stage.GetPrimAtPath(wrapper_path)
    if prim.IsValid():
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(disable)


def _reset_obj_pose(stage, wrapper_path: str):
    """Teleport object back to env-local origin and zero velocities."""
    prim = stage.GetPrimAtPath(wrapper_path)
    if not prim.IsValid():
        return
    xf   = UsdGeom.Xformable(prim)
    ops  = xf.GetOrderedXformOps()
    t_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if t_op is None: t_op = xf.AddTranslateOp()
    t_op.Set(Gf.Vec3d(0, 0, 0))
    r_op = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if r_op is not None:
        r_op.Set(Gf.Quatd(1, 0, 0, 0))
    for attr_name, sdf_type in [
        ("physics:velocity",        Sdf.ValueTypeNames.Vector3f),
        ("physics:angularVelocity", Sdf.ValueTypeNames.Vector3f),
    ]:
        a = prim.GetAttribute(attr_name)
        if not (a and a.IsValid()):
            a = prim.CreateAttribute(attr_name, sdf_type)
        a.Set(Gf.Vec3f(0, 0, 0))


def _apply_finger_target(stage, grip_ref_path: str, target: float):
    for jpath in _finger_joints(grip_ref_path):
        p = stage.GetPrimAtPath(jpath)
        if not p.IsValid():
            continue
        for prop in p.GetProperties():
            name = prop.GetName()
            if "drive" in name and "targetPosition" in name:
                prop.Set(float(target))


# ============================================================
# ── STAGE 5: Physics validation (async) ─────────────────────
# ============================================================

async def _validate_pairs(
    pairs:          List[Tuple[Dict, Dict]],
    object_usd_path: Path,
) -> List[Tuple[Dict, Dict]]:
    """Run grid-cloner parallel physics validation. Returns list of valid pairs."""
    ctx   = omni.usd.get_context()
    stage_existing = ctx.get_stage()
    if stage_existing:
        await ctx.close_stage_async()
        await _step_sim(5)

    await ctx.new_stage_async()
    stage = ctx.get_stage()

    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)

    UsdGeom.Scope.Define(stage, ENV_BASE_PATH)
    UsdGeom.Xform.Define(stage, env_path(0))

    # ── Object in env_0 ──
    UsdGeom.Xform.Define(stage, obj_wrap(0))
    add_reference_to_stage(str(object_usd_path), obj_ref(0))
    await _step_sim(2)

    ref_prim = stage.GetPrimAtPath(obj_ref(0))
    if ref_prim.IsValid():
        for p in Usd.PrimRange(ref_prim):
            for api_cls in [UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxRigidBodyAPI]:
                if p.HasAPI(api_cls):
                    p.RemoveAPI(api_cls)

    _apply_convex_decomp(stage, obj_ref(0))
    _make_object_rigid(stage, obj_wrap(0))

    # ── Two grippers in env_0 ──
    for wrap_fn, ref_fn in [(gripl_wrap, gripl_ref), (gripr_wrap, gripr_ref)]:
        UsdGeom.Xform.Define(stage, wrap_fn(0))
        add_reference_to_stage(str(GRIPPER_USD), ref_fn(0))
    await _step_sim(2)

    _make_kinematic_gripper(stage, gripl_wrap(0))
    _make_kinematic_gripper(stage, gripr_wrap(0))

    # ── Physics scene + ground + material ──
    _setup_physics_scene(stage)
    GroundPlane(prim_path="/World/GroundPlane", z_position=GROUND_Z)
    mat = _create_physics_material(stage, "/World/PhysicsMat/HF")
    _bind_physics_material(stage, obj_ref(0), mat)
    await _step_sim(2)

    # ── Make refs instanceable, then clone ──
    stage.GetPrimAtPath(obj_ref(0)).SetInstanceable(True)
    stage.GetPrimAtPath(gripl_ref(0)).SetInstanceable(True)
    stage.GetPrimAtPath(gripr_ref(0)).SetInstanceable(True)
    await _step_sim(2)

    print(f"[INFO] Cloning {NUM_COPIES} envs (spacing={CLONE_SPACING} m)...")
    cloner = GridCloner(spacing=CLONE_SPACING)
    cloner.define_base_env(ENV_BASE_PATH)
    env_paths_list = cloner.generate_paths(ENV_ROOT_PREFIX, NUM_COPIES)
    cloner.clone(source_prim_path=env_path(0), prim_paths=env_paths_list)
    await _step_sim(4)
    print("[INFO] Cloning done.")

    env_world_pos = []
    for k in range(NUM_COPIES):
        p, _ = transform_utils.get_prim_world_pose(stage.GetPrimAtPath(env_path(k)))
        env_world_pos.append(np.array([float(p[0]), float(p[1]), float(p[2])]))

    obj_prims = [stage.GetPrimAtPath(obj_wrap(k)) for k in range(NUM_COPIES)]

    timeline.play()
    await _step_sim(5)
    print("[INFO] Simulation running.")

    valid_pairs  = []
    batch_size   = NUM_COPIES

    for base in range(0, len(pairs), batch_size):
        batch = pairs[base: base + batch_size]
        K     = len(batch)
        print(f"[INFO] Batch {base // batch_size + 1}: "
              f"pairs {base + 1}..{base + K} / {len(pairs)}")

        anch_a = [batch[k][0] for k in range(K)]
        anch_b = [batch[k][1] for k in range(K)]

        pos_a  = [np.array(a["position"],       dtype=np.float64) for a in anch_a]
        pos_b  = [np.array(a["position"],       dtype=np.float64) for a in anch_b]
        nrm_a  = [np.array(a["normal"],         dtype=np.float64) for a in anch_a]
        nrm_b  = [np.array(a["normal"],         dtype=np.float64) for a in anch_b]
        quat_a = [np.array(a["eef_quaternion"], dtype=np.float64) for a in anch_a]
        quat_b = [np.array(a["eef_quaternion"], dtype=np.float64) for a in anch_b]

        start_a = [pos_a[k] + APPROACH_DISTANCE * nrm_a[k] for k in range(K)]
        start_b = [pos_b[k] + APPROACH_DISTANCE * nrm_b[k] for k in range(K)]

        # ── Reset envs ──
        for k in range(K):
            _reset_obj_pose(stage, obj_wrap(k))
            _set_obj_gravity(stage, obj_wrap(k), disable=True)
            _set_gripper_pose(stage, gripl_wrap(k), start_a[k], quat_a[k], env_world_pos[k])
            _set_gripper_pose(stage, gripr_wrap(k), start_b[k], quat_b[k], env_world_pos[k])
            _apply_finger_target(stage, gripl_ref(k), FINGER_OPEN)
            _apply_finger_target(stage, gripr_ref(k), FINGER_OPEN)
        await _step_sim(5)

        z_pre = []
        for k in range(K):
            p, _ = transform_utils.get_prim_world_pose(obj_prims[k])
            z_pre.append(float(p[2]))

        # ── Approach: interpolate both grippers to contact ──
        for t in range(APPROACH_STEPS + 1):
            alpha = t / float(APPROACH_STEPS)
            for k in range(K):
                ca = start_a[k] * (1 - alpha) + pos_a[k] * alpha
                cb = start_b[k] * (1 - alpha) + pos_b[k] * alpha
                _set_gripper_pose(stage, gripl_wrap(k), ca, quat_a[k], env_world_pos[k])
                _set_gripper_pose(stage, gripr_wrap(k), cb, quat_b[k], env_world_pos[k])
            await omni.kit.app.get_app().next_update_async()
        await _step_sim(5)

        # Gate: reject if approach disturbed object
        active = []
        for k in range(K):
            p, _ = transform_utils.get_prim_world_pose(obj_prims[k])
            active.append(abs(float(p[2]) - z_pre[k]) <= 0.03)

        # ── Close both fingers simultaneously ──
        for _ in range(CLOSE_STEPS):
            for k in range(K):
                if not active[k]:
                    continue
                _apply_finger_target(stage, gripl_ref(k), FINGER_CLOSE)
                _apply_finger_target(stage, gripr_ref(k), FINGER_CLOSE)
            await omni.kit.app.get_app().next_update_async()

        z_contact = []
        for k in range(K):
            p, _ = transform_utils.get_prim_world_pose(obj_prims[k])
            z_contact.append(float(p[2]) if active[k] else None)

        # Enable gravity for lift test
        for k in range(K):
            if active[k]:
                _set_obj_gravity(stage, obj_wrap(k), disable=False)
        await _step_sim(3)

        # ── Lift: both grippers move straight up simultaneously ──
        lift_a = [pos_a[k] + np.array([0.0, 0.0, LIFT_DISTANCE]) for k in range(K)]
        lift_b = [pos_b[k] + np.array([0.0, 0.0, LIFT_DISTANCE]) for k in range(K)]

        for t in range(LIFT_STEPS + 1):
            alpha = t / float(LIFT_STEPS)
            for k in range(K):
                if not active[k]:
                    continue
                ca = pos_a[k] * (1 - alpha) + lift_a[k] * alpha
                cb = pos_b[k] * (1 - alpha) + lift_b[k] * alpha
                _set_gripper_pose(stage, gripl_wrap(k), ca, quat_a[k], env_world_pos[k])
                _set_gripper_pose(stage, gripr_wrap(k), cb, quat_b[k], env_world_pos[k])
                _apply_finger_target(stage, gripl_ref(k), FINGER_CLOSE)
                _apply_finger_target(stage, gripr_ref(k), FINGER_CLOSE)
            await omni.kit.app.get_app().next_update_async()
        await _step_sim(5)

        # ── Check success: object Z rise ──
        for k in range(K):
            if not active[k] or z_contact[k] is None:
                continue
            p, _ = transform_utils.get_prim_world_pose(obj_prims[k])
            if (float(p[2]) - z_contact[k]) >= LIFT_SUCCESS_Z:
                valid_pairs.append((anch_a[k], anch_b[k]))

        for k in range(K):
            _set_obj_gravity(stage, obj_wrap(k), disable=True)

    timeline.stop()
    return valid_pairs


# ============================================================
# ── STAGE 6: Full pipeline ───────────────────────────────────
# ============================================================

async def run_pipeline(object_usd_path: Path, object_type: str,
                       root_prim_path: Optional[str],
                       exclude_prim_paths: Optional[List[str]]):
    # ── 1. Load mesh ──────────────────────────────────────────
    print("[INFO] ── Stage 1: load mesh ──")
    vertices, faces = load_mesh_from_usd(
        object_usd_path,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )
    print(f"[INFO] Mesh: {len(vertices)} verts, {len(faces)} faces")

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bottom_center = [
        float((bbox_min[0] + bbox_max[0]) / 2),
        float((bbox_min[1] + bbox_max[1]) / 2),
        float(bbox_min[2]),
    ]

    # ── 2–3. Sample surface + find anchors ────────────────────
    print("[INFO] ── Stage 2–3: surface sampling + rim anchors ──")
    anchors = find_rim_anchors(vertices, faces)
    print(f"[INFO] {len(anchors)} rim anchors found")
    if not anchors:
        print("[WARN] No anchors found — exiting.")
        return

    # ── 4. EEF pair selection ─────────────────────────────────
    print("[INFO] ── Stage 4: opposite-side pair selection ──")
    pairs = select_opposite_pairs(anchors)
    print(f"[INFO] {len(pairs)} candidate EEF pairs")
    if not pairs:
        print("[WARN] No pairs found — exiting.")
        return

    # ── 5. Physics validation ─────────────────────────────────
    print("[INFO] ── Stage 5: physics validation ──")
    valid_pairs = await _validate_pairs(pairs, object_usd_path)
    print(f"[INFO] {len(valid_pairs)} / {len(pairs)} pairs validated successfully")

    # ── 6. Save results ───────────────────────────────────────
    print("[INFO] ── Stage 6: save results ──")
    annotation_dir = object_usd_path.parent / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    body_list = [
        {
            "left":  list(a["position"]) + list(a["eef_quaternion"]),
            "right": list(b["position"]) + list(b["eef_quaternion"]),
        }
        for a, b in valid_pairs
    ]

    result = {
        "type":            object_type,
        "bottom_center":   bottom_center,
        "functional_grasp": {"body": body_list},
    }

    out_path = annotation_dir / "bi_gripper_grasps.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[INFO] Saved {len(body_list)} grasp poses → {out_path}")


# ============================================================
# Entry point
# ============================================================
def main():
    object_usd = Path(_args.object_usd).resolve()

    async def _run():
        await run_pipeline(
            object_usd_path    = object_usd,
            object_type        = _args.object_type,
            root_prim_path     = _args.root_prim_path,
            exclude_prim_paths = _args.exclude_prim_paths,
        )

    try:
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(_run())
        while not task.done():
            simulation_app.update()
        if task.exception():
            raise task.exception()
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
