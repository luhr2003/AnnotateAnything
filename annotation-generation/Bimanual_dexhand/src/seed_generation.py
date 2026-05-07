from __future__ import annotations

"""
seed_generation.py

Current role:
- Convert a resolved semantic contact template into rough single-hand pose
  seeds for later optimization.
- Produce wrist pose + joint posture seeds using:
  - the local contact frame from contact resolution
  - hand frame conventions
  - category approach families
  - configured posture presets and perturbation ranges

Intentional current simplifications:
- Wrist seeds are geometry-guided heuristics, not FK- or IK-correct poses.
- The module does not yet use exact hand kinematics to back-project semantic
  sphere targets into the wrist frame.
- Side-specific seed adjustments from pose_seeds.yaml are applied as lightweight
  pose offsets, not as a full kinematic retargeting pass.

Future work:
- Add hand-FK-aware wrist initialization so semantic sphere locations are
  matched more faithfully before optimization.
- Add optional seed rejection using SDF collision once the object query layer
  is wired into the pipeline.
- Tune category-specific approach offsets and perturbation scales from empirical
  results.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from src.contact_resolution import ContactResolutionResult, ContactTarget
from src.hand_kinematics import HandKinematicsModel, load_hand_kinematics_model, make_hand_pose
from src.types_config import ResolvedHandRuntimeConfig


@dataclass
class PoseSeed:
    category: str
    contact_template: str
    posture_name: str
    approach_name: str

    wrist_position: np.ndarray
    wrist_quaternion_xyzw: np.ndarray
    wrist_rotation: np.ndarray

    joint_positions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedGenerationResult:
    category: str
    contact_template: str
    contact_result: ContactResolutionResult
    seeds: List[PoseSeed]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _horizontal_axis(v: np.ndarray) -> np.ndarray:
    axis = np.asarray(v, dtype=np.float64).copy()
    axis[2] = 0.0
    return _safe_normalize(axis)


def _semantic_tags(name: str, role_tags: Sequence[str]) -> set[str]:
    tags = {str(t).lower() for t in role_tags}
    tags.update(name.replace("-", "_").split("_"))
    return tags


def _is_thumb_target(target: ContactTarget) -> bool:
    return "thumb" in _semantic_tags(target.name, target.role_tags)


def _is_palm_target(target: ContactTarget) -> bool:
    return "palm" in _semantic_tags(target.name, target.role_tags)


def _target_digit_rank(target: ContactTarget) -> int:
    tags = _semantic_tags(target.name, target.role_tags)
    for idx, digit in enumerate(("index", "middle", "ring", "little")):
        if digit in tags:
            return idx
    return 99


def _cat2_digit_representative_targets(contact_result: ContactResolutionResult) -> List[ContactTarget]:
    finger_targets = [target for target in contact_result.active_targets if not _is_thumb_target(target)]
    if len(finger_targets) <= 1:
        return finger_targets

    targets_by_digit: Dict[int, List[ContactTarget]] = {}
    for target in finger_targets:
        digit_rank = _target_digit_rank(target)
        targets_by_digit.setdefault(digit_rank, []).append(target)

    representatives: List[ContactTarget] = []
    for digit_rank in sorted(targets_by_digit):
        digit_targets = sorted(
            targets_by_digit[digit_rank],
            key=lambda target: (
                0 if "pad" in _semantic_tags(target.name, target.role_tags) else 1,
                0 if "tip" in _semantic_tags(target.name, target.role_tags) else 1,
                target.name,
            ),
        )
        representatives.append(digit_targets[0])
    return representatives


def _weighted_centroid(targets: Sequence[ContactTarget]) -> Optional[np.ndarray]:
    if len(targets) == 0:
        return None
    weights = np.asarray(
        [max(float(t.weight), 1e-6) for t in targets],
        dtype=np.float64,
    )
    points = np.asarray([t.target_point for t in targets], dtype=np.float64)
    return np.average(points, axis=0, weights=weights)


def _estimate_contact_span(contact_result: ContactResolutionResult) -> float:
    pts = np.asarray([t.target_point for t in contact_result.active_targets], dtype=np.float64)
    if len(pts) <= 1:
        return float(contact_result.metadata.get("patch_radius", 0.03))

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(np.max(dist))


def _estimate_palm_depth(runtime_cfg: ResolvedHandRuntimeConfig) -> float:
    palm_link = runtime_cfg.hand.root.palm_link
    palm_spheres = runtime_cfg.collision.spheres_by_link.get(palm_link, [])
    if len(palm_spheres) == 0:
        return 0.05

    extent = 0.0
    for s in palm_spheres:
        center = np.asarray(s.center, dtype=np.float64)
        extent = max(extent, float(np.linalg.norm(center) + s.radius))
    return max(extent, 0.03)


def _orthonormal_basis_from_semantics(
    finger_forward_local: np.ndarray,
    thumb_side_local: np.ndarray,
    palm_normal_local: np.ndarray,
) -> np.ndarray:
    x_axis = _safe_normalize(np.asarray(finger_forward_local, dtype=np.float64))
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    y_axis = np.asarray(thumb_side_local, dtype=np.float64)
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        fallback = np.asarray(palm_normal_local, dtype=np.float64)
        fallback = fallback - x_axis * float(np.dot(fallback, x_axis))
        y_axis = _safe_normalize(fallback)
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    z_axis = _safe_normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    palm_normal_local = _safe_normalize(np.asarray(palm_normal_local, dtype=np.float64))
    if np.linalg.norm(palm_normal_local) > 1e-8 and float(np.dot(z_axis, palm_normal_local)) < 0.0:
        y_axis *= -1.0
        z_axis *= -1.0

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _build_hand_semantic_basis(runtime_cfg: ResolvedHandRuntimeConfig) -> np.ndarray:
    frame_cfg = runtime_cfg.hand.frame_convention
    return _orthonormal_basis_from_semantics(
        finger_forward_local=np.asarray(frame_cfg.finger_forward_local, dtype=np.float64),
        thumb_side_local=np.asarray(frame_cfg.thumb_opposition_local, dtype=np.float64),
        palm_normal_local=np.asarray(frame_cfg.palm_normal_local, dtype=np.float64),
    )


def _derive_thumb_side_axis(contact_result: ContactResolutionResult) -> np.ndarray:
    thumb_targets = [t for t in contact_result.active_targets if _is_thumb_target(t)]
    finger_targets = [t for t in contact_result.active_targets if not _is_thumb_target(t)]

    thumb_center = _weighted_centroid(thumb_targets)
    finger_center = _weighted_centroid(finger_targets)

    if thumb_center is not None and finger_center is not None:
        axis = _safe_normalize(thumb_center - finger_center)
        if np.linalg.norm(axis) > 1e-8:
            return axis

    axis = _safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
    if np.linalg.norm(axis) < 1e-8:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return axis


def _cat4_grasp_mode(contact_result: ContactResolutionResult) -> str:
    mode = str(contact_result.anchor.metadata.get("cat4_grasp_mode", "side")).lower()
    return "top" if mode == "top" else "side"


def _desired_cat4_forward_world(contact_result: ContactResolutionResult) -> np.ndarray:
    finger_targets = [
        target
        for target in contact_result.active_targets
        if (not _is_thumb_target(target)) and (not _is_palm_target(target))
    ]
    if len(finger_targets) > 0:
        by_digit: Dict[int, Dict[str, np.ndarray]] = {}
        for target in finger_targets:
            digit_rank = _target_digit_rank(target)
            if digit_rank >= 99:
                continue
            tags = _semantic_tags(target.name, target.role_tags)
            kind = "tip" if "tip" in tags else "pad" if "pad" in tags else None
            if kind is None:
                continue
            by_digit.setdefault(digit_rank, {})[kind] = np.asarray(target.reference_point, dtype=np.float64)

        forward_dirs: List[np.ndarray] = []
        for digit_rank in sorted(by_digit):
            digit_points = by_digit[digit_rank]
            if "pad" not in digit_points or "tip" not in digit_points:
                continue
            forward_dir = _safe_normalize(digit_points["tip"] - digit_points["pad"])
            if np.linalg.norm(forward_dir) > 1e-8:
                forward_dirs.append(forward_dir)
        if forward_dirs:
            desired = _safe_normalize(np.mean(np.asarray(forward_dirs, dtype=np.float64), axis=0))
            if np.linalg.norm(desired) > 1e-8:
                return desired

    desired = _safe_normalize(np.asarray(contact_result.frame_R[:, 0], dtype=np.float64))
    if np.linalg.norm(desired) < 1e-8:
        desired = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return desired


def _desired_cat4_thumb_world(contact_result: ContactResolutionResult) -> np.ndarray:
    thumb_targets = [target for target in contact_result.active_targets if _is_thumb_target(target)]
    if len(thumb_targets) > 0:
        normals = np.asarray(
            [np.asarray(t.target_normal, dtype=np.float64) for t in thumb_targets],
            dtype=np.float64,
        )
        weights = np.asarray(
            [max(float(t.weight), 1e-6) for t in thumb_targets],
            dtype=np.float64,
        )
        desired = _safe_normalize(np.average(normals, axis=0, weights=weights))
        if np.linalg.norm(desired) > 1e-8:
            return desired

    desired = _derive_thumb_side_axis(contact_result)
    if np.linalg.norm(desired) < 1e-8:
        desired = _safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
    if np.linalg.norm(desired) < 1e-8:
        desired = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return desired


def _desired_cat4_palm_world(contact_result: ContactResolutionResult) -> np.ndarray:
    palm_targets = [target for target in contact_result.active_targets if _is_palm_target(target)]
    if len(palm_targets) > 0:
        normals = np.asarray([np.asarray(t.target_normal, dtype=np.float64) for t in palm_targets], dtype=np.float64)
        weights = np.asarray([max(float(t.weight), 1e-6) for t in palm_targets], dtype=np.float64)
        desired = _safe_normalize(np.average(normals, axis=0, weights=weights))
        if np.linalg.norm(desired) > 1e-8:
            return desired

    if _cat4_grasp_mode(contact_result) == "top":
        desired = -_safe_normalize(np.asarray(contact_result.frame_R[:, 2], dtype=np.float64))
    else:
        desired = -_safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
    if np.linalg.norm(desired) < 1e-8:
        desired = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return desired


def _derive_seed_world_basis(contact_result: ContactResolutionResult) -> np.ndarray:
    world_x = _safe_normalize(np.asarray(contact_result.frame_R[:, 0], dtype=np.float64))
    if np.linalg.norm(world_x) < 1e-8:
        world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    world_y = _derive_thumb_side_axis(contact_result)
    world_y = world_y - world_x * float(np.dot(world_y, world_x))
    world_y = _safe_normalize(world_y)
    if np.linalg.norm(world_y) < 1e-8:
        world_y = _safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
    if np.linalg.norm(world_y) < 1e-8:
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    world_z = _safe_normalize(np.cross(world_x, world_y))
    if np.linalg.norm(world_z) < 1e-8:
        world_z = np.asarray(contact_result.frame_R[:, 2], dtype=np.float64)
        world_z = _safe_normalize(world_z)
    if np.linalg.norm(world_z) < 1e-8:
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if float(np.dot(world_z, contact_result.frame_R[:, 2])) < 0.0:
        world_y *= -1.0
        world_z *= -1.0

    world_y = _safe_normalize(np.cross(world_z, world_x))
    return np.stack([world_x, world_y, world_z], axis=1)


def _append_vector_constraint(
    local_vectors: List[np.ndarray],
    world_vectors: List[np.ndarray],
    weights: List[float],
    local_vec: np.ndarray,
    world_vec: np.ndarray,
    *,
    weight: float,
    colinear_threshold: float = 0.98,
) -> None:
    local_dir = _safe_normalize(np.asarray(local_vec, dtype=np.float64))
    world_dir = _safe_normalize(np.asarray(world_vec, dtype=np.float64))
    if np.linalg.norm(local_dir) < 1e-8 or np.linalg.norm(world_dir) < 1e-8:
        return
    for existing_local in local_vectors:
        if abs(float(np.dot(local_dir, existing_local))) >= colinear_threshold:
            return
    local_vectors.append(local_dir)
    world_vectors.append(world_dir)
    weights.append(float(weight))


def _solve_rotation_from_constraints(
    local_vectors: Sequence[np.ndarray],
    world_vectors: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    if len(local_vectors) == 0 or len(world_vectors) == 0 or len(local_vectors) != len(world_vectors):
        return np.eye(3, dtype=np.float64)
    if len(local_vectors) == 1:
        local_dir = _safe_normalize(np.asarray(local_vectors[0], dtype=np.float64))
        world_dir = _safe_normalize(np.asarray(world_vectors[0], dtype=np.float64))
        if np.linalg.norm(local_dir) < 1e-8 or np.linalg.norm(world_dir) < 1e-8:
            return np.eye(3, dtype=np.float64)
        cross = np.cross(local_dir, world_dir)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(local_dir, world_dir), -1.0, 1.0))
        if cross_norm < 1e-8:
            if dot >= 0.0:
                return np.eye(3, dtype=np.float64)
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(axis, local_dir))) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = _safe_normalize(np.cross(local_dir, axis))
            return Rotation.from_rotvec(np.pi * axis).as_matrix()
        axis = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)
        return Rotation.from_rotvec(axis * angle).as_matrix()

    rotation, _ = Rotation.align_vectors(
        np.asarray(world_vectors, dtype=np.float64),
        np.asarray(local_vectors, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
    )
    return rotation.as_matrix()


def _desired_cat1_palm_world(contact_result: ContactResolutionResult) -> np.ndarray:
    down_world = -_safe_normalize(np.asarray(contact_result.frame_R[:, 2], dtype=np.float64))
    if np.linalg.norm(down_world) < 1e-8:
        down_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    if _cat1_contact_mode(contact_result) != "tip":
        return down_world

    inward_blend = float(contact_result.metadata.get("cat1_seed_palm_inward_blend", 0.0))
    inward_blend = float(np.clip(inward_blend, 0.0, 0.85))
    inward_world = -_safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
    if np.linalg.norm(inward_world) < 1e-8 or inward_blend <= 1e-8:
        return down_world

    desired = _safe_normalize((1.0 - inward_blend) * down_world + inward_blend * inward_world)
    if np.linalg.norm(desired) < 1e-8:
        return down_world
    return desired


def _cat1_contact_mode(contact_result: ContactResolutionResult) -> str:
    return str(contact_result.metadata.get("cat1_contact_mode", "pad")).lower()


def _desired_cat1_forward_world(contact_result: ContactResolutionResult) -> np.ndarray:
    if _cat1_contact_mode(contact_result) == "tip":
        desired = _safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
        if np.linalg.norm(desired) < 1e-8:
            desired = _safe_normalize(np.asarray(contact_result.frame_R[:, 0], dtype=np.float64))
        if np.linalg.norm(desired) < 1e-8:
            desired = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return desired

    desired = -_derive_thumb_side_axis(contact_result)
    if np.linalg.norm(desired) < 1e-8:
        desired = _safe_normalize(np.asarray(contact_result.frame_R[:, 0], dtype=np.float64))
    if np.linalg.norm(desired) < 1e-8:
        desired = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return desired


def _cat1_seed_rotation(runtime_cfg: ResolvedHandRuntimeConfig, contact_result: ContactResolutionResult) -> np.ndarray:
    frame_cfg = runtime_cfg.hand.frame_convention
    local_vectors: List[np.ndarray] = []
    world_vectors: List[np.ndarray] = []
    weights: List[float] = []

    forward_local = np.asarray(frame_cfg.finger_forward_local, dtype=np.float64)
    palm_local = np.asarray(frame_cfg.palm_normal_local, dtype=np.float64)
    thumb_local = np.asarray(frame_cfg.thumb_opposition_local, dtype=np.float64)

    world_forward = _desired_cat1_forward_world(contact_result)
    palm_weight = float(contact_result.metadata.get("cat1_seed_palm_alignment_weight", 4.0))
    thumb_weight = float(contact_result.metadata.get("cat1_seed_thumb_alignment_weight", 1.0))

    desired_palm_world = _desired_cat1_palm_world(contact_result)
    desired_palm_world = desired_palm_world - world_forward * float(np.dot(desired_palm_world, world_forward))
    desired_palm_world = _safe_normalize(desired_palm_world)
    if np.linalg.norm(desired_palm_world) < 1e-8:
        desired_palm_world = _desired_cat1_palm_world(contact_result)

    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        forward_local,
        world_forward,
        weight=3.0,
    )
    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        palm_local,
        desired_palm_world,
        weight=palm_weight,
    )

    if _cat1_contact_mode(contact_result) == "tip":
        thumb_world = -world_forward
    else:
        thumb_world = _derive_thumb_side_axis(contact_result)
    thumb_world = thumb_world - world_forward * float(np.dot(thumb_world, world_forward))
    thumb_world = _safe_normalize(thumb_world)
    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        thumb_local,
        thumb_world,
        weight=thumb_weight,
    )

    rotation = _solve_rotation_from_constraints(local_vectors, world_vectors, weights)

    palm_world = _safe_normalize(rotation @ _safe_normalize(palm_local))
    desired_full_palm_world = _desired_cat1_palm_world(contact_result)
    if np.linalg.norm(palm_world) > 1e-8 and np.linalg.norm(desired_full_palm_world) > 1e-8:
        if float(np.dot(palm_world, desired_full_palm_world)) < 0.0:
            flip_rotation = Rotation.from_rotvec(np.pi * world_forward).as_matrix()
            candidate_rotation = flip_rotation @ rotation
            candidate_palm_world = _safe_normalize(candidate_rotation @ _safe_normalize(palm_local))
            if float(np.dot(candidate_palm_world, desired_full_palm_world)) > float(
                np.dot(palm_world, desired_full_palm_world)
            ):
                rotation = candidate_rotation

    return rotation


def _cat2_seed_rotation(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    branch_sign: int = 1,
    local_forward_axis: Optional[np.ndarray] = None,
    local_span_axis: Optional[np.ndarray] = None,
    local_palm_axis: Optional[np.ndarray] = None,
    local_thumb_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    frame_cfg = runtime_cfg.hand.frame_convention
    local_vectors: List[np.ndarray] = []
    world_vectors: List[np.ndarray] = []
    weights: List[float] = []

    forward_local = np.asarray(
        local_forward_axis if local_forward_axis is not None else frame_cfg.finger_forward_local,
        dtype=np.float64,
    )
    palm_local = np.asarray(
        local_palm_axis if local_palm_axis is not None else frame_cfg.palm_normal_local,
        dtype=np.float64,
    )
    thumb_local = np.asarray(
        local_thumb_axis if local_thumb_axis is not None else frame_cfg.thumb_opposition_local,
        dtype=np.float64,
    )

    world_up, edge_world, world_forward = _cat2_seed_alignment_axes(
        contact_result,
        branch_sign=branch_sign,
    )
    support_world = -world_forward

    span_local = (
        np.asarray(local_span_axis, dtype=np.float64)
        if local_span_axis is not None
        else _safe_normalize(np.cross(palm_local, forward_local))
    )
    span_local = _safe_normalize(span_local)
    if np.linalg.norm(span_local) < 1e-8:
        span_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        palm_local,
        world_up,
        weight=4.0,
    )
    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        span_local,
        edge_world,
        weight=3.0,
    )
    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        thumb_local,
        support_world,
        weight=2.0,
    )
    if local_thumb_axis is None:
        _append_vector_constraint(
            local_vectors,
            world_vectors,
            weights,
            forward_local,
            world_forward,
            weight=1.5,
        )

    rotation = _solve_rotation_from_constraints(local_vectors, world_vectors, weights)

    palm_world = _safe_normalize(rotation @ _safe_normalize(palm_local))
    if np.linalg.norm(palm_world) > 1e-8 and float(np.dot(palm_world, world_up)) < 0.0:
        flip_rotation = Rotation.from_rotvec(np.pi * world_forward).as_matrix()
        candidate_rotation = flip_rotation @ rotation
        candidate_palm_world = _safe_normalize(candidate_rotation @ _safe_normalize(palm_local))
        if float(np.dot(candidate_palm_world, world_up)) > float(np.dot(palm_world, world_up)):
            rotation = candidate_rotation

    thumb_world = _safe_normalize(rotation @ _safe_normalize(thumb_local))
    if np.linalg.norm(thumb_world) > 1e-8 and float(np.dot(thumb_world, support_world)) < 0.0:
        flip_rotation = Rotation.from_rotvec(np.pi * world_up).as_matrix()
        candidate_rotation = flip_rotation @ rotation
        candidate_thumb_world = _safe_normalize(candidate_rotation @ _safe_normalize(thumb_local))
        if float(np.dot(candidate_thumb_world, support_world)) > float(np.dot(thumb_world, support_world)):
            rotation = candidate_rotation

    return rotation


def _cat4_seed_rotation(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    local_forward_axis: Optional[np.ndarray] = None,
    local_span_axis: Optional[np.ndarray] = None,
    local_palm_axis: Optional[np.ndarray] = None,
    local_thumb_axis: Optional[np.ndarray] = None,
    forward_sign: int = 1,
) -> np.ndarray:
    frame_cfg = runtime_cfg.hand.frame_convention
    local_vectors: List[np.ndarray] = []
    world_vectors: List[np.ndarray] = []
    weights: List[float] = []

    forward_local = np.asarray(
        local_forward_axis if local_forward_axis is not None else frame_cfg.finger_forward_local,
        dtype=np.float64,
    )
    palm_local = np.asarray(
        local_palm_axis if local_palm_axis is not None else frame_cfg.palm_normal_local,
        dtype=np.float64,
    )
    thumb_local = np.asarray(
        local_thumb_axis if local_thumb_axis is not None else frame_cfg.thumb_opposition_local,
        dtype=np.float64,
    )

    world_forward = _desired_cat4_forward_world(contact_result)
    if int(forward_sign) < 0:
        world_forward *= -1.0
    world_thumb_side = _desired_cat4_thumb_world(contact_result)
    world_up = _safe_normalize(np.asarray(contact_result.frame_R[:, 2], dtype=np.float64))
    if np.linalg.norm(world_up) < 1e-8:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_palm = _desired_cat4_palm_world(contact_result)
    span_local = (
        _safe_normalize(np.asarray(local_span_axis, dtype=np.float64))
        if local_span_axis is not None
        else None
    )

    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        palm_local,
        world_palm,
        weight=2.6,
    )
    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        forward_local,
        world_forward,
        weight=3.0,
    )
    _append_vector_constraint(
        local_vectors,
        world_vectors,
        weights,
        thumb_local,
        world_thumb_side,
        weight=3.2,
    )
    if span_local is not None and np.linalg.norm(span_local) > 1e-8:
        _append_vector_constraint(
            local_vectors,
            world_vectors,
            weights,
            span_local,
            world_up,
            weight=1.4,
        )

    rotation = _solve_rotation_from_constraints(local_vectors, world_vectors, weights)

    thumb_world = _safe_normalize(rotation @ _safe_normalize(thumb_local))
    if np.linalg.norm(thumb_world) > 1e-8 and float(np.dot(thumb_world, world_thumb_side)) < 0.0:
        flip_rotation = Rotation.from_rotvec(np.pi * world_up).as_matrix()
        candidate_rotation = flip_rotation @ rotation
        candidate_thumb_world = _safe_normalize(candidate_rotation @ _safe_normalize(thumb_local))
        if float(np.dot(candidate_thumb_world, world_thumb_side)) > float(np.dot(thumb_world, world_thumb_side)):
            rotation = candidate_rotation

    palm_world = _safe_normalize(rotation @ _safe_normalize(palm_local))
    if np.linalg.norm(palm_world) > 1e-8 and float(np.dot(palm_world, world_palm)) < 0.0:
        flip_rotation = Rotation.from_rotvec(np.pi * world_forward).as_matrix()
        candidate_rotation = flip_rotation @ rotation
        candidate_palm_world = _safe_normalize(candidate_rotation @ _safe_normalize(palm_local))
        if float(np.dot(candidate_palm_world, world_palm)) > float(np.dot(palm_world, world_palm)):
            rotation = candidate_rotation

    return rotation


def _cat4_thumb_outside_metrics(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    joint_positions: Dict[str, float],
) -> tuple[float, Dict[str, Any]]:
    active_point_names = [target.name for target in contact_result.active_targets]
    hand_pose = make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=None,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=active_point_names)

    thumb_points: List[np.ndarray] = []
    finger_points: List[np.ndarray] = []
    for target in contact_result.active_targets:
        sphere_state = semantic_states.get(target.name)
        if sphere_state is None or _is_palm_target(target):
            continue
        center_world = np.asarray(sphere_state.center_world, dtype=np.float64)
        if _is_thumb_target(target):
            thumb_points.append(center_world)
        else:
            finger_points.append(center_world)

    outward_axis = np.asarray(contact_result.frame_R[:, 1], dtype=np.float64)
    outward_axis[2] = 0.0
    outward_axis = _safe_normalize(outward_axis)
    if np.linalg.norm(outward_axis) < 1e-8:
        outward_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if len(thumb_points) == 0 or len(finger_points) == 0:
        return 0.0, {
            "cat4_thumb_outside_margin": 0.0,
            "cat4_thumb_centroid": None,
            "cat4_finger_centroid": None,
        }

    thumb_centroid = np.mean(np.asarray(thumb_points, dtype=np.float64), axis=0)
    finger_centroid = np.mean(np.asarray(finger_points, dtype=np.float64), axis=0)
    thumb_outside_margin = float(np.dot(thumb_centroid - finger_centroid, outward_axis))
    return thumb_outside_margin, {
        "cat4_thumb_outside_margin": thumb_outside_margin,
        "cat4_thumb_centroid": thumb_centroid.tolist(),
        "cat4_finger_centroid": finger_centroid.tolist(),
    }


def _cat4_finger_forward_alignment(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    joint_positions: Dict[str, float],
) -> tuple[float, Dict[str, Any]]:
    point_names = [
        target.name for target in contact_result.active_targets
        if (not _is_thumb_target(target)) and (not _is_palm_target(target))
    ]
    if len(point_names) == 0:
        return 0.0, {"cat4_forward_alignment_mean_dot": 0.0}

    hand_pose = make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=None,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=point_names)

    by_digit_actual: Dict[int, Dict[str, np.ndarray]] = {}
    by_digit_desired: Dict[int, Dict[str, np.ndarray]] = {}
    for target in contact_result.active_targets:
        if _is_thumb_target(target) or _is_palm_target(target):
            continue
        digit_rank = _target_digit_rank(target)
        if digit_rank >= 99:
            continue
        tags = _semantic_tags(target.name, target.role_tags)
        kind = "tip" if "tip" in tags else "pad" if "pad" in tags else None
        if kind is None:
            continue
        state = semantic_states.get(target.name)
        if state is None:
            continue
        by_digit_actual.setdefault(digit_rank, {})[kind] = np.asarray(state.center_world, dtype=np.float64)
        by_digit_desired.setdefault(digit_rank, {})[kind] = np.asarray(target.reference_point, dtype=np.float64)

    dots: List[float] = []
    for digit_rank in sorted(by_digit_actual):
        actual = by_digit_actual[digit_rank]
        desired = by_digit_desired.get(digit_rank, {})
        if "pad" not in actual or "tip" not in actual or "pad" not in desired or "tip" not in desired:
            continue
        actual_dir = _safe_normalize(actual["tip"] - actual["pad"])
        desired_dir = _safe_normalize(desired["tip"] - desired["pad"])
        if np.linalg.norm(actual_dir) < 1e-8 or np.linalg.norm(desired_dir) < 1e-8:
            continue
        dots.append(float(np.dot(actual_dir, desired_dir)))

    mean_dot = float(np.mean(np.asarray(dots, dtype=np.float64))) if dots else 0.0
    return mean_dot, {
        "cat4_forward_alignment_mean_dot": mean_dot,
        "cat4_forward_alignment_num_digits": int(len(dots)),
    }


def _choose_cat4_seed_branch(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    approach_name: str,
    joint_positions: Dict[str, float],
    semantic_local_axes: Optional[Dict[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    def _adjust_cat4_seed_outward(
        wrist_position: np.ndarray,
        wrist_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, Dict[str, Any]]:
        outward_axis = np.asarray(contact_result.frame_R[:, 1], dtype=np.float64).copy()
        outward_axis[2] = 0.0
        outward_axis = _safe_normalize(outward_axis)
        if np.linalg.norm(outward_axis) < 1e-8:
            outward_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        patch_radius = float(contact_result.metadata.get("patch_radius", 0.03))
        desired_min_margin = float(np.clip(0.06 * patch_radius, 0.0005, 0.003))
        step = float(np.clip(0.10 * patch_radius, 0.001, 0.003))
        max_shift = float(np.clip(0.22 * patch_radius, 0.003, 0.008))

        adjusted_position = np.asarray(wrist_position, dtype=np.float64).copy()
        total_shift = 0.0
        thumb_margin, thumb_meta = _cat4_thumb_outside_metrics(
            runtime_cfg=runtime_cfg,
            hand_model=hand_model,
            contact_result=contact_result,
            wrist_position=adjusted_position,
            wrist_rotation=wrist_rotation,
            joint_positions=joint_positions,
        )
        while thumb_margin < desired_min_margin and total_shift + 1e-9 < max_shift:
            delta = min(step, max_shift - total_shift)
            adjusted_position = adjusted_position + outward_axis * float(delta)
            total_shift += float(delta)
            thumb_margin, thumb_meta = _cat4_thumb_outside_metrics(
                runtime_cfg=runtime_cfg,
                hand_model=hand_model,
                contact_result=contact_result,
                wrist_position=adjusted_position,
                wrist_rotation=wrist_rotation,
                joint_positions=joint_positions,
            )
        return adjusted_position, thumb_margin, {
            "cat4_seed_outward_shift": float(total_shift),
            "cat4_seed_desired_thumb_outside_margin": float(desired_min_margin),
            **thumb_meta,
        }

    base_position, base_rotation, base_meta = _base_wrist_pose_for_seed(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        approach_name=approach_name,
        semantic_local_axes=semantic_local_axes,
        cat4_forward_sign=1,
    )
    flipped_position, flipped_rotation, flipped_meta = _base_wrist_pose_for_seed(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        approach_name=approach_name,
        semantic_local_axes=semantic_local_axes,
        cat4_forward_sign=-1,
    )

    base_position, base_thumb_margin, base_thumb_meta = _adjust_cat4_seed_outward(
        base_position,
        base_rotation,
    )
    base_forward_dot, base_forward_meta = _cat4_finger_forward_alignment(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        wrist_position=base_position,
        wrist_rotation=base_rotation,
        joint_positions=joint_positions,
    )

    flipped_position, flipped_thumb_margin, flipped_thumb_meta = _adjust_cat4_seed_outward(
        flipped_position,
        flipped_rotation,
    )
    flipped_forward_dot, flipped_forward_meta = _cat4_finger_forward_alignment(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        wrist_position=flipped_position,
        wrist_rotation=flipped_rotation,
        joint_positions=joint_positions,
    )

    base_rank = (
        1 if base_thumb_margin > 0.0 else 0,
        base_forward_dot,
        base_thumb_margin,
    )
    flipped_rank = (
        1 if flipped_thumb_margin > 0.0 else 0,
        flipped_forward_dot,
        flipped_thumb_margin,
    )

    if flipped_rank > base_rank:
        return flipped_position, flipped_rotation, {
            "cat4_orientation_branch": "flipped_tangent",
            "cat4_orientation_base_forward_dot": base_forward_dot,
            "cat4_orientation_selected_forward_dot": flipped_forward_dot,
            "cat4_orientation_base_thumb_outside_margin": base_thumb_margin,
            "cat4_orientation_selected_thumb_outside_margin": flipped_thumb_margin,
            **flipped_meta,
            **flipped_thumb_meta,
            **flipped_forward_meta,
        }

    return base_position, base_rotation, {
        "cat4_orientation_branch": "original_tangent",
        "cat4_orientation_base_forward_dot": base_forward_dot,
        "cat4_orientation_selected_forward_dot": base_forward_dot,
        "cat4_orientation_base_thumb_outside_margin": base_thumb_margin,
        "cat4_orientation_selected_thumb_outside_margin": base_thumb_margin,
        **base_meta,
        **base_thumb_meta,
        **base_forward_meta,
    }


def _rotation_matrix_from_rpy_deg(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [np.deg2rad(float(x)) for x in rpy_deg]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        diag = np.diag(m)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-8))
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-8))
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-8))
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    return _safe_normalize(q)


def _quaternion_wxyz_from_xyzw(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _sample_range(
    rng: np.random.Generator,
    low_high: Sequence[float],
    enabled: bool,
) -> float:
    low = float(low_high[0])
    high = float(low_high[1])
    if not enabled or abs(high - low) < 1e-12:
        return 0.5 * (low + high)
    return float(rng.uniform(low, high))


def _clip_joint_positions(
    joint_positions: Dict[str, float],
    runtime_cfg: ResolvedHandRuntimeConfig,
) -> Dict[str, float]:
    clipped: Dict[str, float] = {}
    for joint_name, value in joint_positions.items():
        lo_hi = runtime_cfg.hand.joints.limits.get(joint_name, None)
        if lo_hi is None:
            clipped[joint_name] = float(value)
        else:
            clipped[joint_name] = float(np.clip(value, lo_hi[0], lo_hi[1]))
    return clipped


def _resolve_posture_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    posture_names_override: Optional[Sequence[str]] = None,
) -> List[str]:
    posture_names = list(posture_names_override or runtime_cfg.seed_cfg.joint_seed_postures)
    valid = [name for name in posture_names if name in runtime_cfg.hand.default_postures]
    if len(valid) == 0:
        valid = ["neutral"] if "neutral" in runtime_cfg.hand.default_postures else []
    if len(valid) == 0:
        raise ValueError("No valid seed posture names found in hand.default_postures.")
    return valid


def _active_contact_centroid(contact_result: ContactResolutionResult) -> np.ndarray:
    centroid = _weighted_centroid(contact_result.active_targets)
    if centroid is not None:
        return centroid
    return np.asarray(contact_result.anchor.point, dtype=np.float64)


def _cat1_seed_reference_point(contact_result: ContactResolutionResult) -> np.ndarray:
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    if np.linalg.norm(frame_R) < 1e-8:
        return anchor_point

    finger_targets = [target for target in contact_result.active_targets if not _is_thumb_target(target)]
    support_targets = finger_targets if finger_targets else list(contact_result.active_targets)
    if len(support_targets) == 0:
        return anchor_point

    local_x_values: List[float] = []
    weights: List[float] = []
    for target in support_targets:
        local_point = frame_R.T @ (np.asarray(target.target_point, dtype=np.float64) - anchor_point)
        local_x_values.append(float(local_point[0]))
        weights.append(max(float(target.weight), 1e-6))

    if len(local_x_values) == 0:
        return anchor_point

    reference_local = np.array(
        [
            float(np.average(np.asarray(local_x_values, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64))),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return anchor_point + frame_R @ reference_local


def _cat2_seed_alignment_axes(
    contact_result: ContactResolutionResult,
    branch_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    branch_sign = -1 if int(branch_sign) < 0 else 1
    line_mode = str(contact_result.anchor.metadata.get("cat2_anchor_distribution_mode", "default")) == "bottom_rim_line"

    if line_mode:
        edge_world = _safe_normalize(
            np.array(
                list(contact_result.anchor.metadata.get("cat2_line_axis_xy", [1.0, 0.0])) + [0.0],
                dtype=np.float64,
            )
        )
        support_world = _safe_normalize(
            np.array(
                list(contact_result.anchor.metadata.get("cat2_line_side_axis_xy", [0.0, 1.0])) + [0.0],
                dtype=np.float64,
            )
        )
    else:
        edge_world = _horizontal_axis(contact_result.frame_R[:, 0])
        support_world = _horizontal_axis(contact_result.frame_R[:, 1])

    if np.linalg.norm(edge_world) < 1e-8 and np.linalg.norm(support_world) < 1e-8:
        edge_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        support_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif np.linalg.norm(edge_world) < 1e-8:
        edge_world = _safe_normalize(np.cross(support_world, world_up))
    elif np.linalg.norm(support_world) < 1e-8:
        support_world = _safe_normalize(np.cross(world_up, edge_world))

    edge_world = edge_world - support_world * float(np.dot(edge_world, support_world))
    edge_world = _safe_normalize(edge_world)
    if np.linalg.norm(edge_world) < 1e-8:
        edge_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    support_world = _safe_normalize(np.cross(world_up, edge_world))
    support_hint = _horizontal_axis(contact_result.frame_R[:, 1])
    if np.linalg.norm(support_hint) > 1e-8 and float(np.dot(support_world, support_hint)) < 0.0:
        edge_world *= -1.0
        support_world *= -1.0

    world_forward = -support_world

    if branch_sign < 0:
        edge_world *= -1.0
        world_forward *= -1.0

    return world_up, edge_world, world_forward


def _cat2_anchor_between_fingers_point(contact_result: ContactResolutionResult) -> tuple[np.ndarray, Dict[str, Any]]:
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    finger_targets = _cat2_digit_representative_targets(contact_result)
    if len(finger_targets) == 0 or np.linalg.norm(frame_R) < 1e-8:
        return anchor_point, {
            "cat2_anchor_between_fingers_point": anchor_point.tolist(),
            "cat2_finger_span_local_x_range": [0.0, 0.0],
            "cat2_finger_span_local_x_mid": 0.0,
        }

    local_x_values: List[float] = []
    for target in finger_targets:
        local_point = frame_R.T @ (np.asarray(target.reference_point, dtype=np.float64) - anchor_point)
        local_x_values.append(float(local_point[0]))

    if len(local_x_values) == 0:
        return anchor_point, {
            "cat2_anchor_between_fingers_point": anchor_point.tolist(),
            "cat2_finger_span_local_x_range": [0.0, 0.0],
            "cat2_finger_span_local_x_mid": 0.0,
        }

    x_min = float(min(local_x_values))
    x_max = float(max(local_x_values))
    x_mid = 0.5 * (x_min + x_max)
    midpoint = anchor_point + frame_R @ np.array([x_mid, 0.0, 0.0], dtype=np.float64)
    return midpoint, {
        "cat2_anchor_between_fingers_point": midpoint.tolist(),
        "cat2_finger_span_local_x_range": [x_min, x_max],
        "cat2_finger_span_local_x_mid": x_mid,
    }


def _cat2_seed_reference_point(
    contact_result: ContactResolutionResult,
    branch_sign: int = 1,
) -> tuple[np.ndarray, Dict[str, Any]]:
    finger_targets = [target for target in contact_result.active_targets if not _is_thumb_target(target)]
    finger_centroid = _weighted_centroid(finger_targets)
    if finger_centroid is None:
        finger_centroid = _active_contact_centroid(contact_result)

    anchor_between_fingers_point, between_meta = _cat2_anchor_between_fingers_point(contact_result)

    _, _, world_forward = _cat2_seed_alignment_axes(contact_result, branch_sign=branch_sign)
    line_mode = str(contact_result.anchor.metadata.get("cat2_anchor_distribution_mode", "default")) == "bottom_rim_line"
    finger_under_anchor_offset = float(
        np.clip(
            (0.18 if line_mode else 0.35) * float(contact_result.metadata.get("patch_radius", 0.03)),
            0.004 if line_mode else 0.008,
            0.009 if line_mode else 0.014,
        )
    )
    reference_point = np.asarray(anchor_between_fingers_point, dtype=np.float64) + world_forward * finger_under_anchor_offset
    return reference_point, {
        "cat2_finger_centroid": np.asarray(finger_centroid, dtype=np.float64).tolist(),
        **between_meta,
        "cat2_anchor_distribution_mode": contact_result.anchor.metadata.get("cat2_anchor_distribution_mode"),
        "cat2_forward_axis": world_forward.tolist(),
        "cat2_finger_under_anchor_offset": finger_under_anchor_offset,
    }


def _cat4_seed_reference_point(
    contact_result: ContactResolutionResult,
) -> tuple[np.ndarray, Dict[str, Any]]:
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    if np.linalg.norm(frame_R) < 1e-8:
        return anchor_point, {
            "cat4_seed_reference_point": anchor_point.tolist(),
            "cat4_reference_mode": "anchor_fallback",
        }

    support_targets = [
        target
        for target in contact_result.active_targets
        if (not _is_thumb_target(target)) and (not _is_palm_target(target))
    ]
    if len(support_targets) == 0:
        support_targets = [target for target in contact_result.active_targets if not _is_palm_target(target)]
    if len(support_targets) == 0:
        return anchor_point, {
            "cat4_seed_reference_point": anchor_point.tolist(),
            "cat4_reference_mode": "anchor_only",
        }

    local_points = np.asarray(
        [
            frame_R.T @ (np.asarray(target.reference_point, dtype=np.float64) - anchor_point)
            for target in support_targets
        ],
        dtype=np.float64,
    )
    weights = np.asarray([max(float(target.weight), 1e-6) for target in support_targets], dtype=np.float64)

    patch_radius = float(contact_result.metadata.get("patch_radius", 0.03))
    reference_local = np.array(
        [
            float(np.average(local_points[:, 0], weights=weights)),
            float(-np.clip(0.10 * patch_radius, 0.002, 0.006)),
            float(np.average(local_points[:, 2], weights=weights)),
        ],
        dtype=np.float64,
    )
    reference_point = anchor_point + frame_R @ reference_local
    return reference_point, {
        "cat4_seed_reference_point": reference_point.tolist(),
        "cat4_seed_reference_local": reference_local.tolist(),
        "cat4_reference_mode": "finger_group_centered",
    }


def _cat2_alignment_point_name(runtime_cfg: ResolvedHandRuntimeConfig, point_name: str) -> str:
    _ = runtime_cfg
    return point_name


def _cat2_hand_local_semantic_axes(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    joint_positions: Dict[str, float],
    point_names_override: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    point_names = list(point_names_override or runtime_cfg.contact_usage.active_points)
    if len(point_names) == 0:
        return {}

    hand_pose = make_hand_pose(
        wrist_position=np.zeros(3, dtype=np.float64),
        wrist_quaternion_xyzw=None,
        wrist_rotation=np.eye(3, dtype=np.float64),
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=point_names)

    digit_pad: Dict[int, np.ndarray] = {}
    digit_tip: Dict[int, np.ndarray] = {}
    digit_centers: Dict[int, List[np.ndarray]] = {}
    thumb_points: List[np.ndarray] = []
    palm_points: List[np.ndarray] = []
    finger_points: List[np.ndarray] = []

    for point_name in point_names:
        sphere_state = semantic_states.get(point_name)
        if sphere_state is None:
            continue
        point_world = np.asarray(sphere_state.center_world, dtype=np.float64)
        tags = _semantic_tags(point_name, [])
        if "thumb" in tags:
            thumb_points.append(point_world)
            continue
        if "palm" in tags:
            palm_points.append(point_world)
            continue

        digit_rank = 99
        for idx, digit in enumerate(("index", "middle", "ring", "little")):
            if digit in tags:
                digit_rank = idx
                break
        digit_centers.setdefault(digit_rank, []).append(point_world)
        finger_points.append(point_world)
        if "pad" in tags and digit_rank not in digit_pad:
            digit_pad[digit_rank] = point_world
        if "tip" in tags and digit_rank not in digit_tip:
            digit_tip[digit_rank] = point_world

    local_axes: Dict[str, np.ndarray] = {}

    forward_dirs: List[np.ndarray] = []
    for digit_rank, pad_center in digit_pad.items():
        tip_center = digit_tip.get(digit_rank)
        if tip_center is None:
            continue
        forward_dir = _safe_normalize(np.asarray(tip_center, dtype=np.float64) - np.asarray(pad_center, dtype=np.float64))
        if np.linalg.norm(forward_dir) > 1e-8:
            forward_dirs.append(forward_dir)
    if forward_dirs:
        local_axes["forward"] = _safe_normalize(np.mean(np.asarray(forward_dirs, dtype=np.float64), axis=0))

    ordered_digits = sorted(digit_centers)
    if len(ordered_digits) >= 2:
        low_digit = np.mean(np.asarray(digit_centers[ordered_digits[0]], dtype=np.float64), axis=0)
        high_digit = np.mean(np.asarray(digit_centers[ordered_digits[-1]], dtype=np.float64), axis=0)
        span_axis = _safe_normalize(low_digit - high_digit)
        if np.linalg.norm(span_axis) > 1e-8:
            local_axes["span"] = span_axis

    cfg_palm = _safe_normalize(np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64))
    if palm_points and finger_points:
        palm_axis = _safe_normalize(
            np.mean(np.asarray(palm_points, dtype=np.float64), axis=0)
            - np.mean(np.asarray(finger_points, dtype=np.float64), axis=0)
        )
        if np.linalg.norm(palm_axis) > 1e-8 and np.linalg.norm(cfg_palm) > 1e-8 and float(np.dot(palm_axis, cfg_palm)) < 0.0:
            palm_axis *= -1.0
        if np.linalg.norm(palm_axis) > 1e-8:
            local_axes["palm"] = palm_axis
    elif "forward" in local_axes and "span" in local_axes:
        palm_axis = _safe_normalize(np.cross(local_axes["span"], local_axes["forward"]))
        if np.linalg.norm(palm_axis) > 1e-8 and np.linalg.norm(cfg_palm) > 1e-8 and float(np.dot(palm_axis, cfg_palm)) < 0.0:
            palm_axis *= -1.0
        if np.linalg.norm(palm_axis) > 1e-8:
            local_axes["palm"] = palm_axis

    if thumb_points and finger_points:
        thumb_axis = _safe_normalize(
            np.mean(np.asarray(thumb_points, dtype=np.float64), axis=0)
            - np.mean(np.asarray(finger_points, dtype=np.float64), axis=0)
        )
        cfg_thumb = _safe_normalize(np.asarray(runtime_cfg.hand.frame_convention.thumb_opposition_local, dtype=np.float64))
        if np.linalg.norm(thumb_axis) > 1e-8 and np.linalg.norm(cfg_thumb) > 1e-8 and float(np.dot(thumb_axis, cfg_thumb)) < 0.0:
            thumb_axis *= -1.0
        if np.linalg.norm(thumb_axis) > 1e-8:
            local_axes["thumb"] = thumb_axis

    return local_axes


def _cat2_hand_local_thumb_axis(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    joint_positions: Dict[str, float],
) -> Optional[np.ndarray]:
    point_names = list(runtime_cfg.contact_usage.active_points)
    if len(point_names) == 0:
        return None

    hand_pose = make_hand_pose(
        wrist_position=np.zeros(3, dtype=np.float64),
        wrist_quaternion_xyzw=None,
        wrist_rotation=np.eye(3, dtype=np.float64),
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=point_names)

    thumb_points: List[np.ndarray] = []
    finger_points: List[np.ndarray] = []
    for point_name in point_names:
        sphere_state = semantic_states.get(point_name)
        if sphere_state is None:
            continue
        center_world = np.asarray(sphere_state.center_world, dtype=np.float64)
        if "thumb" in _semantic_tags(point_name, []):
            thumb_points.append(center_world)
        else:
            finger_points.append(center_world)

    if len(thumb_points) == 0 or len(finger_points) == 0:
        return None

    thumb_centroid = np.mean(np.asarray(thumb_points, dtype=np.float64), axis=0)
    finger_centroid = np.mean(np.asarray(finger_points, dtype=np.float64), axis=0)
    thumb_axis = _safe_normalize(thumb_centroid - finger_centroid)
    if np.linalg.norm(thumb_axis) < 1e-8:
        return None
    return thumb_axis


def _cat2_thumb_outside_metrics(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    joint_positions: Dict[str, float],
) -> tuple[float, Dict[str, Any]]:
    active_point_names = [target.name for target in contact_result.active_targets]
    hand_pose = make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=None,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=active_point_names)

    thumb_points: List[np.ndarray] = []
    finger_points: List[np.ndarray] = []
    for target in contact_result.active_targets:
        sphere_state = semantic_states.get(target.name)
        if sphere_state is None:
            continue
        center_world = np.asarray(sphere_state.center_world, dtype=np.float64)
        if _is_thumb_target(target):
            thumb_points.append(center_world)
        else:
            finger_points.append(center_world)

    outward_axis = np.asarray(contact_result.frame_R[:, 1], dtype=np.float64)
    outward_axis[2] = 0.0
    outward_axis = _safe_normalize(outward_axis)
    if np.linalg.norm(outward_axis) < 1e-8:
        outward_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if len(thumb_points) == 0 or len(finger_points) == 0:
        return 0.0, {
            "thumb_outside_margin": 0.0,
            "thumb_centroid": None,
            "finger_centroid": None,
            "outward_axis": outward_axis.tolist(),
        }

    thumb_centroid = np.mean(np.asarray(thumb_points, dtype=np.float64), axis=0)
    finger_centroid = np.mean(np.asarray(finger_points, dtype=np.float64), axis=0)
    thumb_outside_margin = float(np.dot(thumb_centroid - finger_centroid, outward_axis))
    return thumb_outside_margin, {
        "thumb_outside_margin": thumb_outside_margin,
        "thumb_centroid": thumb_centroid.tolist(),
        "finger_centroid": finger_centroid.tolist(),
        "outward_axis": outward_axis.tolist(),
    }


def _align_cat2_seed_wrist_to_finger_targets(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    joint_positions: Dict[str, float],
    branch_sign: int = 1,
) -> tuple[np.ndarray, Dict[str, Any]]:
    finger_targets = [target for target in contact_result.active_targets if not _is_thumb_target(target)]
    if len(finger_targets) == 0:
        return wrist_position, {}

    point_names = [_cat2_alignment_point_name(runtime_cfg, target.name) for target in finger_targets]
    hand_pose = make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=None,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=point_names)
    world_up, _, world_forward = _cat2_seed_alignment_axes(contact_result, branch_sign=branch_sign)
    anchor_between_fingers_point, between_meta = _cat2_anchor_between_fingers_point(contact_result)
    patch_radius = float(contact_result.metadata.get("patch_radius", 0.03))
    line_mode = str(contact_result.anchor.metadata.get("cat2_anchor_distribution_mode", "default")) == "bottom_rim_line"
    lower_offset = float(
        np.clip(
            (0.05 if line_mode else 0.08) * patch_radius,
            0.0010 if line_mode else 0.0015,
            0.0025 if line_mode else 0.0035,
        )
    )
    deeper_offset = float(
        np.clip(
            (0.04 if line_mode else 0.08) * patch_radius,
            0.0010 if line_mode else 0.0015,
            0.0025 if line_mode else 0.0040,
        )
    )

    actual_points: List[np.ndarray] = []
    desired_points: List[np.ndarray] = []
    weights: List[float] = []
    proxy_names: List[str] = []
    for target in finger_targets:
        proxy_name = _cat2_alignment_point_name(runtime_cfg, target.name)
        sphere_state = semantic_states.get(proxy_name)
        if sphere_state is None:
            continue
        actual_points.append(np.asarray(sphere_state.center_world, dtype=np.float64))
        desired_points.append(
            np.asarray(target.target_point, dtype=np.float64)
            - world_up * lower_offset
            + world_forward * deeper_offset
        )
        weights.append(max(float(target.weight), 1e-6))
        proxy_names.append(proxy_name)

    if len(actual_points) == 0:
        return wrist_position, {}

    actual_centroid = np.average(np.asarray(actual_points, dtype=np.float64), axis=0, weights=np.asarray(weights))
    desired_points_arr = np.asarray(desired_points, dtype=np.float64)
    desired_centroid = np.average(desired_points_arr, axis=0, weights=np.asarray(weights))
    desired_anchor_center = (
        np.asarray(anchor_between_fingers_point, dtype=np.float64)
        - world_up * lower_offset
        + world_forward * deeper_offset
    )
    centroid_shift = desired_anchor_center - desired_centroid
    desired_points_arr = desired_points_arr + centroid_shift[None, :]
    desired_centroid = desired_centroid + centroid_shift
    translation_delta = desired_centroid - actual_centroid
    aligned_wrist_position = wrist_position + translation_delta
    seed_downshift = float(np.clip(0.12 * patch_radius, 0.0025, 0.0050))
    aligned_wrist_position = aligned_wrist_position - world_up * seed_downshift

    return aligned_wrist_position, {
        "cat2_finger_target_centroid": desired_centroid.tolist(),
        "cat2_finger_current_centroid": actual_centroid.tolist(),
        "cat2_finger_alignment_delta": translation_delta.tolist(),
        "cat2_alignment_proxy_points": proxy_names,
        "cat2_alignment_anchor_center": desired_anchor_center.tolist(),
        "cat2_alignment_centroid_shift": centroid_shift.tolist(),
        "cat2_alignment_lower_offset": lower_offset,
        "cat2_alignment_deeper_offset": deeper_offset,
        "cat2_seed_downshift": seed_downshift,
        **between_meta,
    }


def _cat4_pregrasp_desired_center(target: ContactTarget) -> np.ndarray:
    normal = _safe_normalize(np.asarray(target.target_normal, dtype=np.float64))
    if np.linalg.norm(normal) < 1e-8:
        return np.asarray(target.reference_point, dtype=np.float64)
    sphere_radius = 0.0 if target.source_sphere is None else float(target.source_sphere.radius)
    return (
        np.asarray(target.reference_point, dtype=np.float64)
        + normal * (sphere_radius + float(target.desired_clearance_pregrasp))
    )


def _align_cat4_seed_wrist_to_targets(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    joint_positions: Dict[str, float],
) -> tuple[np.ndarray, Dict[str, Any]]:
    active_targets = list(contact_result.active_targets)
    if len(active_targets) == 0:
        return wrist_position, {}

    point_names = [target.name for target in active_targets]
    hand_pose = make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=None,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )
    semantic_states = hand_model.semantic_points_world(runtime_cfg, hand_pose, point_names=point_names)

    actual_points: List[np.ndarray] = []
    desired_points: List[np.ndarray] = []
    weights: List[float] = []
    palm_actual: List[np.ndarray] = []
    aligned_names: List[str] = []

    for target in active_targets:
        sphere_state = semantic_states.get(target.name)
        if sphere_state is None:
            continue
        actual = np.asarray(sphere_state.center_world, dtype=np.float64)
        desired = _cat4_pregrasp_desired_center(target)
        weight = max(float(target.weight), 1e-6)
        if _is_palm_target(target):
            palm_actual.append(actual)
        actual_points.append(actual)
        desired_points.append(desired)
        weights.append(weight)
        aligned_names.append(target.name)

    if len(actual_points) == 0:
        return wrist_position, {}

    actual_arr = np.asarray(actual_points, dtype=np.float64)
    desired_arr = np.asarray(desired_points, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    translation_delta = np.average(desired_arr - actual_arr, axis=0, weights=weights_arr)

    aligned_wrist_position = np.asarray(wrist_position, dtype=np.float64) + translation_delta
    return aligned_wrist_position, {
        "cat4_alignment_delta": np.asarray(translation_delta, dtype=np.float64).tolist(),
        "cat4_alignment_actual_centroid": np.average(actual_arr, axis=0, weights=weights_arr).tolist(),
        "cat4_alignment_desired_centroid": np.average(desired_arr, axis=0, weights=weights_arr).tolist(),
        "cat4_alignment_target_names": aligned_names,
        "cat4_alignment_used_palm_targets": int(len(palm_actual)),
    }


def _choose_cat2_seed_branch(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    joint_positions: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    aligned_position, aligned_meta = _align_cat2_seed_wrist_to_finger_targets(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        wrist_position=wrist_position,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )
    base_margin, base_metric_meta = _cat2_thumb_outside_metrics(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        wrist_position=aligned_position,
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    flip_rotation = Rotation.from_rotvec(np.pi * world_up).as_matrix() @ wrist_rotation
    flipped_position, flipped_align_meta = _align_cat2_seed_wrist_to_finger_targets(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        wrist_position=wrist_position,
        wrist_rotation=flip_rotation,
        joint_positions=joint_positions,
    )
    flipped_margin, flipped_metric_meta = _cat2_thumb_outside_metrics(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        wrist_position=flipped_position,
        wrist_rotation=flip_rotation,
        joint_positions=joint_positions,
    )

    if flipped_margin > base_margin + 1e-6:
        return flipped_position, flip_rotation, {
            "cat2_orientation_branch": "flipped_about_world_up",
            "cat2_orientation_base_thumb_outside_margin": base_margin,
            "cat2_orientation_selected_thumb_outside_margin": flipped_margin,
            **flipped_metric_meta,
            **flipped_align_meta,
        }

    return aligned_position, wrist_rotation, {
        "cat2_orientation_branch": "original",
        "cat2_orientation_base_thumb_outside_margin": base_margin,
        "cat2_orientation_selected_thumb_outside_margin": base_margin,
        **base_metric_meta,
        **aligned_meta,
    }


def _approach_offset_local(
    contact_result: ContactResolutionResult,
    approach_name: str,
    palm_depth: float,
    contact_span: float,
    pregrasp_clearance: float,
) -> np.ndarray:
    patch_radius = float(contact_result.metadata.get("patch_radius", max(0.03, 0.5 * contact_span)))
    side_step = max(0.25 * contact_span, 0.30 * patch_radius)
    top_step = max(0.10 * contact_span, 0.20 * patch_radius)
    below_step = max(0.20 * contact_span, 0.35 * patch_radius)
    forward_step = max(0.10 * contact_span, 0.12 * patch_radius)

    if contact_result.category == "cat2":
        below_step *= 0.55
        forward_step *= 0.60
    elif contact_result.category == "cat4":
        side_step *= 0.35
        top_step *= 0.40
        below_step *= 0.50
        forward_step *= 0.45

    if approach_name == "outside_in":
        return np.array([0.0, +side_step, +0.5 * pregrasp_clearance], dtype=np.float64)
    if approach_name == "slightly_top_down":
        return np.array([0.0, +0.85 * side_step, +top_step], dtype=np.float64)
    if approach_name == "from_below":
        return np.array([0.0, 0.0, -below_step], dtype=np.float64)
    if approach_name == "below_and_forward":
        return np.array([+forward_step, 0.0, -below_step], dtype=np.float64)
    if approach_name == "side_in":
        return np.array([0.0, +0.60 * side_step, 0.0], dtype=np.float64)
    if approach_name == "side_wrap":
        return np.array([0.0, +0.40 * side_step, 0.0], dtype=np.float64)
    if approach_name == "top_down":
        return np.array([0.0, 0.0, +top_step], dtype=np.float64)

    _ = palm_depth
    return np.zeros(3, dtype=np.float64)


def _cat4_seed_approach_names(
    contact_result: ContactResolutionResult,
    approach_names: Sequence[str],
) -> List[str]:
    mode = _cat4_grasp_mode(contact_result)
    names = [str(name) for name in approach_names]
    if mode == "top":
        preferred = [name for name in names if name in {"top_down", "slightly_top_down"}]
        return preferred or ["top_down"]
    preferred = [name for name in names if name in {"side_in", "side_wrap"}]
    return preferred or names


def _base_wrist_pose_for_seed(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    approach_name: str,
    cat2_branch_sign: int = 1,
    semantic_local_axes: Optional[Dict[str, np.ndarray]] = None,
    cat4_forward_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    basis_mode = "semantic_thumb_side"
    if contact_result.category == "cat1":
        basis_mode = "vector_constraints_cat1"
        wrist_rotation = _cat1_seed_rotation(runtime_cfg, contact_result)
    elif contact_result.category == "cat2":
        basis_mode = "vector_constraints_cat2"
        wrist_rotation = _cat2_seed_rotation(
            runtime_cfg,
            contact_result,
            branch_sign=cat2_branch_sign,
            local_forward_axis=None if semantic_local_axes is None else semantic_local_axes.get("forward"),
            local_span_axis=None if semantic_local_axes is None else semantic_local_axes.get("span"),
            local_thumb_axis=None if semantic_local_axes is None else semantic_local_axes.get("thumb"),
        )
    elif contact_result.category == "cat4":
        basis_mode = "vector_constraints_cat4"
        wrist_rotation = _cat4_seed_rotation(
            runtime_cfg,
            contact_result,
            local_forward_axis=None if semantic_local_axes is None else semantic_local_axes.get("forward"),
            local_span_axis=None if semantic_local_axes is None else semantic_local_axes.get("span"),
            local_palm_axis=None if semantic_local_axes is None else semantic_local_axes.get("palm"),
            local_thumb_axis=None if semantic_local_axes is None else semantic_local_axes.get("thumb"),
            forward_sign=cat4_forward_sign,
        )
    else:
        hand_basis_local = _build_hand_semantic_basis(runtime_cfg)
        world_basis = _derive_seed_world_basis(contact_result)
        wrist_rotation = world_basis @ hand_basis_local.T

    active_centroid = _active_contact_centroid(contact_result)
    seed_reference_mode = "active_centroid"
    seed_reference_point = active_centroid
    category_reference_meta: Dict[str, Any] = {}
    if contact_result.category == "cat1":
        seed_reference_mode = "anchor_centered_cat1"
        seed_reference_point = _cat1_seed_reference_point(contact_result)
    elif contact_result.category == "cat2":
        seed_reference_mode = "finger_under_anchor_cat2"
        seed_reference_point, category_reference_meta = _cat2_seed_reference_point(
            contact_result,
            branch_sign=cat2_branch_sign,
        )
    elif contact_result.category == "cat4":
        seed_reference_mode = "convex_hold_cat4"
        seed_reference_point, category_reference_meta = _cat4_seed_reference_point(contact_result)
    contact_span = _estimate_contact_span(contact_result)
    palm_depth = _estimate_palm_depth(runtime_cfg)
    pregrasp_clearance = max(
        [t.desired_clearance_pregrasp for t in contact_result.active_targets] or [0.01]
    )
    if contact_result.category == "cat1":
        palm_depth *= float(contact_result.metadata.get("cat1_seed_palm_depth_scale", 1.0))
        pregrasp_clearance += float(contact_result.metadata.get("cat1_seed_pregrasp_clearance_extra", 0.0))
    elif contact_result.category == "cat2":
        line_mode = str(contact_result.anchor.metadata.get("cat2_anchor_distribution_mode", "default")) == "bottom_rim_line"
        palm_depth *= 0.62 if line_mode else 0.68
        pregrasp_clearance *= 0.60 if line_mode else 0.70
    elif contact_result.category == "cat4":
        palm_depth *= 0.58
        pregrasp_clearance *= 0.45

    palm_normal_local = _safe_normalize(
        np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )
    palm_normal_world = _safe_normalize(wrist_rotation @ palm_normal_local)
    if np.linalg.norm(palm_normal_world) < 1e-8:
        if contact_result.category == "cat1":
            palm_normal_world = _desired_cat1_palm_world(contact_result)
        elif contact_result.category == "cat4":
            if _cat4_grasp_mode(contact_result) == "top":
                palm_normal_world = -_safe_normalize(np.asarray(contact_result.frame_R[:, 2], dtype=np.float64))
            else:
                palm_normal_world = -_safe_normalize(np.asarray(contact_result.frame_R[:, 1], dtype=np.float64))
        else:
            palm_normal_world = _safe_normalize(np.asarray(contact_result.frame_R[:, 2], dtype=np.float64))
    if np.linalg.norm(palm_normal_world) < 1e-8:
        palm_normal_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    local_offset = _approach_offset_local(
        contact_result=contact_result,
        approach_name=approach_name,
        palm_depth=palm_depth,
        contact_span=contact_span,
        pregrasp_clearance=pregrasp_clearance,
    )

    wrist_position = (
        seed_reference_point
        - palm_normal_world * (palm_depth + pregrasp_clearance)
        + contact_result.frame_R @ local_offset
    )
    cat2_wrist_retreat = None
    if contact_result.category == "cat2":
        _, _, cat2_forward_world = _cat2_seed_alignment_axes(
            contact_result,
            branch_sign=cat2_branch_sign,
        )
        line_mode = str(contact_result.anchor.metadata.get("cat2_anchor_distribution_mode", "default")) == "bottom_rim_line"
        cat2_wrist_retreat = float(
            np.clip(
                (0.55 if line_mode else 0.85) * float(contact_result.metadata.get("patch_radius", max(0.03, 0.5 * contact_span))),
                0.010 if line_mode else 0.018,
                0.018 if line_mode else 0.030,
            )
        )
        wrist_position = wrist_position - cat2_forward_world * cat2_wrist_retreat

    seed_adjustment_local = np.asarray(
        runtime_cfg.seed_adjustment.position_offset_local,
        dtype=np.float64,
    )
    seed_adjustment_rpy_deg = np.asarray(
        runtime_cfg.seed_adjustment.wrist_rpy_offset_deg,
        dtype=np.float64,
    )
    wrist_rotation = wrist_rotation @ _rotation_matrix_from_rpy_deg(seed_adjustment_rpy_deg)
    wrist_position = wrist_position + contact_result.frame_R @ seed_adjustment_local

    metadata = {
        "active_centroid": active_centroid.tolist(),
        "seed_reference_point": np.asarray(seed_reference_point, dtype=np.float64).tolist(),
        "seed_reference_mode": seed_reference_mode,
        "contact_span": float(contact_span),
        "palm_depth": float(palm_depth),
        "pregrasp_clearance": float(pregrasp_clearance),
        "approach_offset_local": local_offset.tolist(),
        "thumb_side_axis": _derive_thumb_side_axis(contact_result).tolist(),
        "cat1_forward_axis": (
            _desired_cat1_forward_world(contact_result).tolist()
            if contact_result.category == "cat1"
            else None
        ),
        "cat2_seed_palm_depth_scale": (0.68 if contact_result.category == "cat2" else None),
        "cat2_seed_pregrasp_clearance_scale": (0.70 if contact_result.category == "cat2" else None),
        "cat2_seed_below_step_scale": (0.55 if contact_result.category == "cat2" else None),
        "cat2_seed_forward_step_scale": (0.60 if contact_result.category == "cat2" else None),
        "cat2_wrist_retreat_along_fingers": cat2_wrist_retreat,
        "cat4_seed_palm_depth_scale": (0.82 if contact_result.category == "cat4" else None),
        "cat4_seed_pregrasp_clearance_scale": (0.82 if contact_result.category == "cat4" else None),
        "cat4_forward_sign": (int(cat4_forward_sign) if contact_result.category == "cat4" else None),
        "cat4_grasp_mode": (_cat4_grasp_mode(contact_result) if contact_result.category == "cat4" else None),
        **category_reference_meta,
        "seed_basis_mode": basis_mode,
        "seed_adjustment_local": seed_adjustment_local.tolist(),
        "seed_adjustment_rpy_deg": seed_adjustment_rpy_deg.tolist(),
    }
    return wrist_position, wrist_rotation, metadata


def generate_pose_seeds(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    num_seeds: Optional[int] = None,
    seed: int = 0,
    posture_names_override: Optional[Sequence[str]] = None,
    approach_names_override: Optional[Sequence[str]] = None,
    semantic_hand_model: Optional[HandKinematicsModel] = None,
) -> SeedGenerationResult:
    """
    Generate rough wrist pose + joint posture seeds for one resolved contact
    template.

    The first cycle over (posture x approach) combinations is deterministic and
    unperturbed. Additional cycles draw translation / rotation perturbations from
    the configured wrist_perturbation ranges.
    """
    rng = np.random.default_rng(seed)

    if posture_names_override is None and contact_result.category == "cat1":
        adaptive_postures = contact_result.metadata.get("cat1_seed_posture_names")
        if isinstance(adaptive_postures, (list, tuple)):
            posture_names_override = [str(name) for name in adaptive_postures]

    posture_names = _resolve_posture_names(runtime_cfg, posture_names_override)
    approach_names = list(approach_names_override or runtime_cfg.seed_cfg.approach_family)
    if contact_result.category == "cat4":
        approach_names = _cat4_seed_approach_names(contact_result, approach_names)
    if len(approach_names) == 0:
        approach_names = ["default"]

    if num_seeds is None:
        num_seeds = int(runtime_cfg.seed_cfg.default_num_seeds_per_region)
    num_seeds = max(1, int(num_seeds))

    combos = list(product(posture_names, approach_names))
    if len(combos) == 0:
        raise ValueError("No posture/approach combinations available for seed generation.")

    perturb_cfg = runtime_cfg.seed_cfg.wrist_perturbation
    seeds: List[PoseSeed] = []
    if contact_result.category in {"cat2", "cat4"} and semantic_hand_model is None:
        semantic_hand_model = load_hand_kinematics_model(runtime_cfg)
    if contact_result.category == "cat4":
        semantic_axis_point_names = tuple(target.name for target in contact_result.active_targets)
    else:
        semantic_axis_point_names = tuple(
            target.name
            for target in contact_result.active_targets
            if not _is_palm_target(target)
        )
    semantic_local_axes_by_posture: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
    base_pose_cache: Dict[tuple[str, str], tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = {}

    for i in range(num_seeds):
        posture_name, approach_name = combos[i % len(combos)]
        is_canonical = i < len(combos)

        joint_positions = _clip_joint_positions(
            dict(runtime_cfg.hand.default_postures[posture_name]),
            runtime_cfg,
        )
        semantic_local_axes = semantic_local_axes_by_posture.get(posture_name)
        if posture_name not in semantic_local_axes_by_posture:
            semantic_local_axes = None
            if contact_result.category in {"cat2", "cat4"} and semantic_hand_model is not None:
                semantic_local_axes = _cat2_hand_local_semantic_axes(
                    runtime_cfg=runtime_cfg,
                    hand_model=semantic_hand_model,
                    joint_positions=joint_positions,
                    point_names_override=semantic_axis_point_names,
                )
            semantic_local_axes_by_posture[posture_name] = semantic_local_axes

        combo_key = (posture_name, approach_name)
        if combo_key in base_pose_cache:
            cached_position, cached_rotation, cached_meta = base_pose_cache[combo_key]
            base_position = np.asarray(cached_position, dtype=np.float64).copy()
            base_rotation = np.asarray(cached_rotation, dtype=np.float64).copy()
            base_meta = dict(cached_meta)
        else:
            if contact_result.category == "cat4" and semantic_hand_model is not None:
                base_position, base_rotation, base_meta = _choose_cat4_seed_branch(
                    runtime_cfg=runtime_cfg,
                    hand_model=semantic_hand_model,
                    contact_result=contact_result,
                    approach_name=approach_name,
                    joint_positions=joint_positions,
                    semantic_local_axes=semantic_local_axes,
                )
            else:
                base_position, base_rotation, base_meta = _base_wrist_pose_for_seed(
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                    approach_name=approach_name,
                    semantic_local_axes=semantic_local_axes,
                )
            base_pose_cache[combo_key] = (
                np.asarray(base_position, dtype=np.float64).copy(),
                np.asarray(base_rotation, dtype=np.float64).copy(),
                dict(base_meta),
            )

        xyz_local = np.array(
            [
                _sample_range(rng, perturb_cfg.xyz_offset_local["x"], enabled=not is_canonical),
                _sample_range(rng, perturb_cfg.xyz_offset_local["y"], enabled=not is_canonical),
                _sample_range(rng, perturb_cfg.xyz_offset_local["z"], enabled=not is_canonical),
            ],
            dtype=np.float64,
        )
        rpy_deg = np.array(
            [
                _sample_range(rng, perturb_cfg.rpy_offset_deg["roll"], enabled=not is_canonical),
                _sample_range(rng, perturb_cfg.rpy_offset_deg["pitch"], enabled=not is_canonical),
                _sample_range(rng, perturb_cfg.rpy_offset_deg["yaw"], enabled=not is_canonical),
            ],
            dtype=np.float64,
        )

        rot_delta = _rotation_matrix_from_rpy_deg(rpy_deg)
        wrist_rotation = base_rotation @ rot_delta
        wrist_position = base_position + contact_result.frame_R @ xyz_local

        alignment_meta: Dict[str, Any] = {}
        if contact_result.category == "cat2" and semantic_hand_model is not None:
            wrist_position, alignment_meta = _align_cat2_seed_wrist_to_finger_targets(
                runtime_cfg=runtime_cfg,
                hand_model=semantic_hand_model,
                contact_result=contact_result,
                wrist_position=wrist_position,
                wrist_rotation=wrist_rotation,
                joint_positions=joint_positions,
            )
        elif contact_result.category == "cat4" and semantic_hand_model is not None:
            wrist_position, alignment_meta = _align_cat4_seed_wrist_to_targets(
                runtime_cfg=runtime_cfg,
                hand_model=semantic_hand_model,
                contact_result=contact_result,
                wrist_position=wrist_position,
                wrist_rotation=wrist_rotation,
                joint_positions=joint_positions,
            )

        seeds.append(
            PoseSeed(
                category=contact_result.category,
                contact_template=contact_result.contact_template,
                posture_name=posture_name,
                approach_name=approach_name,
                wrist_position=wrist_position,
                wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(wrist_rotation),
                wrist_rotation=wrist_rotation,
                joint_positions=joint_positions,
                metadata={
                    "seed_index": int(i),
                    "is_canonical": bool(is_canonical),
                    "xyz_offset_local": xyz_local.tolist(),
                    "rpy_offset_deg": rpy_deg.tolist(),
                    **alignment_meta,
                },
            )
        )

    return SeedGenerationResult(
        category=contact_result.category,
        contact_template=contact_result.contact_template,
        contact_result=contact_result,
        seeds=seeds,
        metadata={
            "num_seeds": int(len(seeds)),
            "posture_names": posture_names,
            "approach_names": approach_names,
            "seed_rng": int(seed),
        },
    )


def generate_pose_seeds_for_contacts(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_results: Sequence[ContactResolutionResult],
    num_seeds_per_contact: Optional[int] = None,
    seed: int = 0,
    posture_names_override: Optional[Sequence[str]] = None,
    approach_names_override: Optional[Sequence[str]] = None,
    semantic_hand_model: Optional[HandKinematicsModel] = None,
) -> List[SeedGenerationResult]:
    results: List[SeedGenerationResult] = []
    for i, contact_result in enumerate(contact_results):
        results.append(
            generate_pose_seeds(
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
                num_seeds=num_seeds_per_contact,
                seed=seed + i,
                posture_names_override=posture_names_override,
                approach_names_override=approach_names_override,
                semantic_hand_model=semantic_hand_model,
            )
        )
    return results


def summarize_seed_generation(result: SeedGenerationResult, top_k: int = 5) -> Dict[str, Any]:
    packed: List[Dict[str, Any]] = []
    for seed in result.seeds[:top_k]:
        quat_xyzw = np.asarray(seed.wrist_quaternion_xyzw, dtype=np.float64)
        packed.append(
            {
                "posture_name": seed.posture_name,
                "approach_name": seed.approach_name,
                "wrist_position": [float(x) for x in seed.wrist_position.tolist()],
                "wrist_quaternion_wxyz": [float(x) for x in _quaternion_wxyz_from_xyzw(quat_xyzw).tolist()],
                "wrist_quaternion_xyzw": [float(x) for x in quat_xyzw.tolist()],
                "joint_positions": {k: float(v) for k, v in seed.joint_positions.items()},
                "metadata": seed.metadata,
            }
        )

    return {
        "category": result.category,
        "contact_template": result.contact_template,
        "num_seeds": int(len(result.seeds)),
        "metadata": result.metadata,
        "seeds_top_k": packed,
    }
