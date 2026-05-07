from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from src.contact_resolution import ContactResolutionResult, ContactTarget, OppositionConstraint
from src.hand_kinematics import (
    CollisionSphereState,
    HandKinematicsModel,
    HandPose,
    load_hand_kinematics_model,
    make_hand_pose,
)
from src.object_query import ObjectQueryBackend
from src.seed_generation import PoseSeed
from src.types_config import ResolvedHandRuntimeConfig, SemanticPoint


StageName = Literal["pregrasp", "grasp", "squeeze"]


DEFAULT_OPTIMIZER_CFG: Dict[str, Any] = {
    "sdf": {
        "voxel_size": 0.005,
        "padding_voxels": 8,
        "force_rebuild": False,
    },
    "solver": {
        "max_iterations": 140,
        "ftol": 1e-8,
        "gtol": 1e-8,
        "eps": 1e-4,
        "maxls": 80,
    },
    "stage_init": {
        "open_posture_name": "open_hand",
        "close_posture_name": "rim_pad_close",
        "pregrasp_alpha": 0.20,
        "pregrasp_alignment_mode": "translation_only",
        "pregrasp_roll_refine_enabled": True,
        "pregrasp_roll_max_deg": 35.0,
        "pregrasp_roll_num_steps": 25,
        "pregrasp_roll_regularization": 0.01,
        "pregrasp_roll_thumb_weight": 2.0,
        "pregrasp_pitch_refine_enabled": True,
        "pregrasp_pitch_max_deg": 20.0,
        "pregrasp_pitch_num_steps": 17,
        "pregrasp_pitch_regularization": 0.015,
        "pregrasp_pitch_thumb_weight": 1.5,
        "pregrasp_alignment_max_translation": 0.05,
        "pregrasp_alignment_max_rotation_deg": 35.0,
        "grasp_refine_enabled": True,
        "grasp_roll_max_deg": 12.0,
        "grasp_pitch_max_deg": 8.0,
        "grasp_alignment_mode": "rigid_clamped",
        "grasp_alignment_max_translation": 0.045,
        "grasp_alignment_max_rotation_deg": 10.0,
        "squeeze_refine_enabled": True,
        "squeeze_roll_max_deg": 10.0,
        "squeeze_pitch_max_deg": 8.0,
        "squeeze_alignment_mode": "rigid_clamped",
        "squeeze_alignment_max_translation": 0.04,
        "squeeze_alignment_max_rotation_deg": 8.0,
        "pregrasp_preserve_seed_palm_hemisphere": True,
        "pregrasp_min_palm_alignment_dot": 0.0,
        "grasp_alpha": 0.64,
        "squeeze_alpha": 0.90,
        "grasp_approach_scale": 0.95,
        "squeeze_approach_scale": 0.80,
        "grasp_shift_min": 0.008,
        "squeeze_shift_min": 0.005,
        "grasp_wrist_blend_alpha": 0.35,
        "grasp_joint_blend_alpha": 0.75,
        "squeeze_wrist_blend_alpha": 0.15,
        "squeeze_joint_blend_alpha": 0.88,
    },
    "collision": {
        "non_active_margin": 0.004,
        "palm_margin": 0.010,
        "self_collision_margin": 0.002,
    },
    "contact_model": {
        "cat1_sample_blend": {
            "pregrasp": 0.10,
            "grasp": 0.30,
            "squeeze": 0.55,
        },
        "cat2_sample_blend": {
            "pregrasp": 0.10,
            "grasp": 0.22,
            "squeeze": 0.32,
        },
        "cat3_sample_blend": {
            "pregrasp": 0.10,
            "grasp": 0.35,
            "squeeze": 0.55,
        },
        "cat2_local_axis_weights": {
            "pregrasp": [0.55, 0.30, 0.70],
            "grasp": [0.85, 0.40, 0.95],
            "squeeze": [1.00, 0.50, 1.10],
        },
        "cat3_local_axis_weights": {
            "pregrasp": [0.50, 0.85, 1.00],
            "grasp": [0.70, 1.00, 1.15],
            "squeeze": [0.85, 1.05, 1.25],
        },
        "cat3_global_penetration": {
            "enabled": True,
            "pregrasp_weight": 20000.0,
            "grasp_weight": 40000.0,
            "squeeze_weight": 80000.0,
        },
        "cat3_bottom_clearance": {
            "enabled": True,
            "pregrasp_weight": 80000.0,
            "grasp_weight": 140000.0,
            "squeeze_weight": 140000.0,
            "margin": 0.0,
            "tolerance": 0.0,
        },
        "cat3_frame_alignment": {
            "enabled": True,
            "grasp_weight": 10.0,
            "squeeze_weight": 18.0,
            "palm_multiplier": 1.2,
        },
        "cat3_fast_pose_search": {
            "enabled": True,
            "snap_rotation_to_contact_frame": True,
            "direct_target_translation": True,
            "direct_alignment_max_translation": 0.090,
            "contact_joint_posture": "cat3_palm_first",
            "grasp_joint_close_blend": 0.0,
            "squeeze_base_posture": "cat3_side_press",
            "squeeze_joint_close_blend": 0.0,
            "squeeze_toward_limit_close": False,
            "squeeze_collision_evaluation": False,
            "refine_toward_base_steps": 8,
            "approach_offsets": [
                -0.060,
                -0.052,
                -0.044,
                -0.042,
                -0.034,
                -0.026,
                -0.018,
                -0.010,
                -0.004,
                0.0,
                0.004,
                0.010,
                0.018,
                0.026,
                0.034,
                0.042,
                0.044,
                0.052,
                0.060,
            ],
            "side_offsets": [0.0],
            "vertical_offsets": [-0.006, 0.0, 0.006],
            "pregrasp_from_contact": True,
            "pregrasp_retreat_offsets": [0.045, 0.060, 0.030, 0.075, 0.015, 0.090, 0.0],
            "pregrasp_vertical_offsets": [0.0, 0.006, 0.012, 0.024],
            "pregrasp_preferred_retreat": 0.045,
            "pregrasp_min_retreat": 0.020,
            "pregrasp_preferred_vertical_offset": 0.0,
            "pregrasp_min_vertical_offset": 0.0,
            "pregrasp_max_evaluations": 8,
            "max_pose_candidates": 57,
        },
        "cat4_sample_blend": {
            "pregrasp": 0.08,
            "grasp": 0.24,
            "squeeze": 0.40,
        },
        "cat4_local_axis_weights": {
            "pregrasp": [0.20, 1.00, 0.30],
            "grasp": [0.45, 1.00, 0.55],
            "squeeze": [0.60, 1.05, 0.75],
        },
        "cat4_thumb_outside": {
            "enabled": True,
            "thumb_blend_multiplier": 0.35,
            "grasp_weight": 24.0,
            "squeeze_weight": 38.0,
            "y_slack": 0.001,
        },
        "cat1_local_axis_weights": {
            "pregrasp": [0.10, 1.00, 0.45],
            "grasp": [0.18, 1.00, 0.65],
            "squeeze": [0.28, 1.00, 0.80],
        },
        "cat1_tip_support": {
            "enabled": True,
            "grasp_weight": 0.8,
            "squeeze_weight": 8.0,
            "grasp_clearance": 0.003,
            "squeeze_clearance": -0.002,
            "grasp_tolerance": 0.004,
            "squeeze_tolerance": 0.003,
            "grasp_required_contacts": 0,
            "squeeze_required_contacts": 1,
        },
        "cat4_global_penetration": {
            "enabled": True,
            "grasp_weight": 320.0,
            "squeeze_weight": 680.0,
        },
    },
    "final_contact_seek": {
        "enabled": True,
        "max_attempts": 8,
        "wrist_step": 0.004,
        "joint_close_step": 0.25,
        "limit_close_blend": 0.35,
        "active_clearance_multiplier": 8.0,
        "active_target_position_multiplier": 0.05,
        "active_normal_alignment_multiplier": 0.05,
        "opposition_multiplier": 0.10,
        "wrist_translation_prior_multiplier": 0.10,
        "wrist_rotation_prior_multiplier": 0.15,
        "joint_prior_multiplier": 0.05,
        "palm_collision_multiplier": 0.85,
        "non_active_collision_multiplier": 0.85,
        "self_collision_multiplier": 0.90,
        "max_iterations": 220,
        "max_translation_from_reference": 0.045,
        "max_translation_from_coarse": 0.060,
        "max_rotation_from_reference_deg": 22.0,
        "max_rotation_from_coarse_deg": 30.0,
        "min_reference_palm_alignment_dot": 0.35,
        "max_active_target_position_factor": 4.0,
        "max_active_target_position_slack": 0.020,
    },
    "stage_weights": {
        "pregrasp": {
            "active_clearance": 6.0,
            "active_target_position": 2.5,
            "active_normal_alignment": 0.5,
            "thumb_active_multiplier": 1.0,
            "opposition": 0.5,
            "non_active_collision": 12.0,
            "palm_collision": 16.0,
            "self_collision": 4.0,
            "wrist_translation_prior": 3.0,
            "wrist_rotation_prior": 3.0,
            "joint_prior": 2.5,
            "avoid_clearance": 1.0,
        },
        "grasp": {
            "active_clearance": 12.0,
            "active_target_position": 8.0,
            "active_normal_alignment": 1.5,
            "thumb_active_multiplier": 1.8,
            "opposition": 3.0,
            "non_active_collision": 10.0,
            "palm_collision": 18.0,
            "self_collision": 5.0,
            "wrist_translation_prior": 1.25,
            "wrist_rotation_prior": 1.5,
            "joint_prior": 0.9,
            "avoid_clearance": 2.5,
        },
        "squeeze": {
            "active_clearance": 18.0,
            "active_target_position": 12.0,
            "active_normal_alignment": 2.25,
            "thumb_active_multiplier": 2.8,
            "opposition": 3.5,
            "non_active_collision": 9.0,
            "palm_collision": 20.0,
            "self_collision": 5.0,
            "wrist_translation_prior": 1.0,
            "wrist_rotation_prior": 1.2,
            "joint_prior": 0.25,
            "avoid_clearance": 3.0,
        },
    },
    "stage_completion": {
        "grasp": {
            "active_contact_tolerance": 0.010,
            "thumb_contact_tolerance": 0.008,
        },
        "squeeze": {
            "active_contact_tolerance": 0.006,
            "thumb_contact_tolerance": 0.004,
        },
    },
}


@dataclass
class CostBreakdown:
    terms: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(sum(self.terms.values()))


@dataclass
class StagePoseResult:
    stage_name: StageName
    hand_pose: HandPose
    success: bool
    cost: float
    message: str
    num_iterations: int
    cost_breakdown: CostBreakdown
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StagedGraspResult:
    category: str
    contact_template: str
    seed: PoseSeed
    stage_results: Dict[StageName, StagePoseResult]
    total_cost: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def sort_staged_grasp_results(
    contact_result: ContactResolutionResult,
    results: Sequence[StagedGraspResult],
) -> List[StagedGraspResult]:
    result_list = list(results)
    if contact_result.category in {"cat3", "cat4"}:
        return sorted(
            result_list,
            key=lambda item: (
                0 if item.success else 1,
                *_stage_result_rank_key(item.stage_results["squeeze"]),
                item.total_cost,
            ),
        )
    return sorted(result_list, key=lambda item: (0 if item.success else 1, item.total_cost))


def _deep_merge(base: Dict[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result = dict(base)
    if override is None:
        return result
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _merged_optimizer_cfg(runtime_cfg: ResolvedHandRuntimeConfig) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULT_OPTIMIZER_CFG, runtime_cfg.optimizer_cfg.get("global", {}))
    cfg = _deep_merge(cfg, runtime_cfg.optimizer_cfg.get("category", {}))
    return cfg


def _rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()


def _quaternion_wxyz_from_xyzw(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _clearance_for_stage(target: ContactTarget, stage_name: StageName) -> float:
    if stage_name == "pregrasp":
        return float(target.desired_clearance_pregrasp)
    if stage_name == "grasp":
        return float(target.desired_clearance_grasp)
    if stage_name == "squeeze":
        return float(target.desired_clearance_squeeze)
    raise ValueError(f"Unsupported stage: {stage_name}")


def _stage_cfg_scalar(
    stage_values: Mapping[str, Any],
    stage_name: StageName,
    default: float,
) -> float:
    value = stage_values.get(stage_name, default)
    return float(value)


def _stage_cfg_vec3(
    stage_values: Mapping[str, Any],
    stage_name: StageName,
    default: Sequence[float],
) -> np.ndarray:
    value = stage_values.get(stage_name, default)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        return np.asarray(default, dtype=np.float64)
    return arr


def _scaled_stage_weights(
    base_weights: Mapping[str, float],
    multipliers: Mapping[str, float],
) -> Dict[str, float]:
    scaled = {key: float(value) for key, value in base_weights.items()}
    for key, multiplier in multipliers.items():
        if key in scaled:
            scaled[key] = float(scaled[key]) * float(multiplier)
    return scaled


def _metadata_float(metadata: Mapping[str, Any], key: str, default: float) -> float:
    value = metadata.get(key, None)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metadata_str(metadata: Mapping[str, Any], key: str, default: str) -> str:
    value = metadata.get(key, None)
    if value is None:
        return str(default)
    text = str(value)
    return text if text else str(default)


def _resolve_posture(runtime_cfg: ResolvedHandRuntimeConfig, posture_name: str) -> Dict[str, float]:
    return {
        joint_name: float(value)
        for joint_name, value in runtime_cfg.hand.default_postures.get(posture_name, {}).items()
    }


def _derive_limit_close_pose(
    runtime_cfg: ResolvedHandRuntimeConfig,
    open_pose: Mapping[str, float],
    close_pose: Mapping[str, float],
    joint_names: Sequence[str],
) -> Dict[str, float]:
    limit_pose: Dict[str, float] = {}
    for joint_name in joint_names:
        open_value = float(open_pose.get(joint_name, 0.0))
        close_value = float(close_pose.get(joint_name, open_value))
        limits = runtime_cfg.hand.joints.limits.get(joint_name)
        if limits is None:
            limit_pose[joint_name] = close_value
            continue
        lo, hi = float(limits[0]), float(limits[1])
        if close_value > open_value + 1e-6:
            limit_pose[joint_name] = hi
        elif close_value < open_value - 1e-6:
            limit_pose[joint_name] = lo
        else:
            limit_pose[joint_name] = close_value
    return limit_pose


def _blend_joint_positions(
    pose_a: Mapping[str, float],
    pose_b: Mapping[str, float],
    alpha: float,
    joint_names: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    alpha = float(alpha)
    for joint_name in joint_names:
        a = float(pose_a.get(joint_name, 0.0))
        b = float(pose_b.get(joint_name, a))
        out[joint_name] = (1.0 - alpha) * a + alpha * b
    return out


def _blend_joint_subset_toward_pose(
    base_pose: Mapping[str, float],
    target_pose: Mapping[str, float],
    alpha: float,
    joint_names: Sequence[str],
) -> Dict[str, float]:
    out = {joint_name: float(value) for joint_name, value in base_pose.items()}
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return out
    for joint_name in joint_names:
        current = float(base_pose.get(joint_name, 0.0))
        target = float(target_pose.get(joint_name, current))
        out[joint_name] = (1.0 - alpha) * current + alpha * target
    return out


def _fit_rigid_alignment(
    source_points_world: np.ndarray,
    target_points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source_points_world, dtype=np.float64)
    dst = np.asarray(target_points_world, dtype=np.float64)
    if len(src) == 0 or len(dst) == 0 or len(src) != len(dst):
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    if len(src) == 1:
        return np.eye(3, dtype=np.float64), dst[0] - src[0]

    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)
    src_centered = src - src_centroid[None, :]
    dst_centered = dst - dst_centroid[None, :]
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    return R, t


def _is_thumb_target(target: ContactTarget) -> bool:
    if "thumb" in str(target.name).lower():
        return True
    return any("thumb" in str(tag).lower() for tag in target.role_tags)


def _is_palm_target(target: ContactTarget) -> bool:
    if "palm" in str(target.name).lower():
        return True
    return any("palm" in str(tag).lower() for tag in target.role_tags)


def _is_tip_target(target: ContactTarget) -> bool:
    if "tip" in str(target.name).lower():
        return True
    return any("tip" in str(tag).lower() for tag in target.role_tags)


def _declared_tip_links(runtime_cfg: ResolvedHandRuntimeConfig) -> set[str]:
    tips = runtime_cfg.hand.links.tips
    return {
        link_name
        for link_name in (tips.thumb, tips.index, tips.middle, tips.ring, tips.little)
        if str(link_name)
    }


def _is_declared_tip_target(
    runtime_cfg: ResolvedHandRuntimeConfig,
    target: ContactTarget,
) -> bool:
    return str(target.source_link) in _declared_tip_links(runtime_cfg) or _is_tip_target(target)


def _target_probe_local_direction(
    runtime_cfg: ResolvedHandRuntimeConfig,
    target: ContactTarget,
) -> np.ndarray:
    sphere_list = runtime_cfg.collision.spheres_by_link.get(str(target.source_link), [])
    sphere_index = int(target.source_sphere_index)
    current_center = None
    prev_center = None
    next_center = None
    if 0 <= sphere_index < len(sphere_list):
        current_center = np.asarray(sphere_list[sphere_index].center, dtype=np.float64)
    if 0 <= sphere_index - 1 < len(sphere_list):
        prev_center = np.asarray(sphere_list[sphere_index - 1].center, dtype=np.float64)
    if 0 <= sphere_index + 1 < len(sphere_list):
        next_center = np.asarray(sphere_list[sphere_index + 1].center, dtype=np.float64)

    local_dir = np.zeros(3, dtype=np.float64)
    if prev_center is not None and next_center is not None:
        local_dir = next_center - prev_center
    elif current_center is not None and next_center is not None:
        local_dir = next_center - current_center
    elif current_center is not None and prev_center is not None:
        local_dir = current_center - prev_center

    local_dir = _safe_normalize(local_dir)
    if np.linalg.norm(local_dir) > 1e-8:
        return local_dir

    fallback = (
        runtime_cfg.hand.frame_convention.thumb_opposition_local
        if _is_thumb_target(target)
        else runtime_cfg.hand.frame_convention.finger_forward_local
    )
    return _safe_normalize(np.asarray(fallback, dtype=np.float64))


def _target_probe_world(
    runtime_cfg: ResolvedHandRuntimeConfig,
    link_transforms: Mapping[str, np.ndarray],
    sphere_state: CollisionSphereState,
    target: ContactTarget,
    *,
    local_dir_override: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    link_T = link_transforms.get(str(target.source_link))
    if link_T is None:
        return None
    if local_dir_override is not None:
        local_dir = _safe_normalize(np.asarray(local_dir_override, dtype=np.float64))
    else:
        local_dir = _target_probe_local_direction(runtime_cfg, target)
    if _is_thumb_target(target):
        root_T = link_transforms.get(str(runtime_cfg.hand.root.wrist_link))
        if root_T is not None:
            thumb_dir_hand = _safe_normalize(
                np.asarray(runtime_cfg.hand.frame_convention.thumb_opposition_local, dtype=np.float64)
            )
            if np.linalg.norm(thumb_dir_hand) > 1e-8:
                thumb_dir_world = _safe_normalize(
                    np.asarray(root_T[:3, :3], dtype=np.float64) @ thumb_dir_hand
                )
                thumb_dir_local = _safe_normalize(
                    np.asarray(link_T[:3, :3], dtype=np.float64).T @ thumb_dir_world
                )
                if np.linalg.norm(thumb_dir_local) > 1e-8:
                    local_dir = thumb_dir_local if local_dir_override is None else _safe_normalize(
                        0.45 * thumb_dir_local + 0.55 * local_dir
                    )
    if np.linalg.norm(local_dir) < 1e-8:
        return None
    world_dir = _safe_normalize(np.asarray(link_T[:3, :3], dtype=np.float64) @ local_dir)
    if np.linalg.norm(world_dir) < 1e-8:
        return None

    # For declared tip links, use the frontmost surface point over the whole link
    # instead of the semantic point's source sphere. On dex3 the semantic tip
    # sphere is slightly proximal to the tiny distal cap sphere, which creates a
    # persistent visual gap even when the grasp is otherwise correct.
    sphere_list = runtime_cfg.collision.spheres_by_link.get(str(target.source_link), [])
    if sphere_list:
        local_surface_candidates = []
        for sphere in sphere_list:
            center_local = np.asarray(sphere.center, dtype=np.float64)
            radius = float(sphere.radius)
            local_surface_candidates.append(center_local + local_dir * radius)
        local_surface_candidates_arr = np.asarray(local_surface_candidates, dtype=np.float64)
        front_idx = int(np.argmax(local_surface_candidates_arr @ local_dir))
        probe_local = local_surface_candidates_arr[front_idx]
        probe_world = (
            np.asarray(link_T[:3, :3], dtype=np.float64) @ probe_local
            + np.asarray(link_T[:3, 3], dtype=np.float64)
        )
    else:
        probe_world = np.asarray(sphere_state.center_world, dtype=np.float64) + world_dir * float(sphere_state.radius)
    return probe_world, world_dir


def _query_cat4_tip_probe_metrics(
    runtime_cfg: ResolvedHandRuntimeConfig,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    hand_pose: HandPose,
    contact_result: ContactResolutionResult,
    semantic_lookup: Mapping[str, CollisionSphereState],
    stage_name: StageName,
) -> Dict[str, Dict[str, Any]]:
    if contact_result.category != "cat4" or stage_name not in {"grasp", "squeeze"}:
        return {}

    link_transforms = hand_model.forward_link_transforms(hand_pose)
    items: List[tuple[str, np.ndarray, np.ndarray]] = []
    for target in contact_result.active_targets:
        if _is_palm_target(target) or not _is_declared_tip_target(runtime_cfg, target):
            continue
        sphere_state = semantic_lookup.get(target.name)
        if sphere_state is None:
            continue
        local_dir_override = None
        if _is_thumb_target(target):
            link_T = link_transforms.get(str(target.source_link))
            if link_T is not None:
                target_world = _cat4_stage_target_point_world(
                    runtime_cfg,
                    contact_result,
                    target,
                    stage_name,
                )
                desired_world_dir = _safe_normalize(
                    np.asarray(target_world, dtype=np.float64)
                    - np.asarray(sphere_state.center_world, dtype=np.float64)
                )
                if np.linalg.norm(desired_world_dir) > 1e-8:
                    local_dir_override = _safe_normalize(
                        np.asarray(link_T[:3, :3], dtype=np.float64).T @ desired_world_dir
                    )
        probe = _target_probe_world(
            runtime_cfg,
            link_transforms,
            sphere_state,
            target,
            local_dir_override=local_dir_override,
        )
        if probe is None:
            continue
        probe_world, probe_dir_world = probe
        items.append((target.name, probe_world, probe_dir_world))

    if not items:
        return {}

    probe_points = np.asarray([item[1] for item in items], dtype=np.float64)
    sdf_vals = np.asarray(object_query.signed_distance(probe_points), dtype=np.float64)
    grads = np.asarray(object_query.gradient(probe_points), dtype=np.float64)

    metrics: Dict[str, Dict[str, Any]] = {}
    for idx, (target_name, probe_world, probe_dir_world) in enumerate(items):
        metrics[target_name] = {
            "probe_world": np.asarray(probe_world, dtype=np.float64),
            "probe_dir_world": np.asarray(probe_dir_world, dtype=np.float64),
            "sdf": float(sdf_vals[idx]),
            "grad": np.asarray(grads[idx], dtype=np.float64),
            "clearance": float(sdf_vals[idx]),
        }
    return metrics


def _clamp_rotation_matrix(rotation_matrix: np.ndarray, max_rotation_deg: Optional[float]) -> np.ndarray:
    if max_rotation_deg is None:
        return np.asarray(rotation_matrix, dtype=np.float64)
    max_rotation_rad = np.deg2rad(float(max_rotation_deg))
    rot = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64))
    rotvec = rot.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle <= max_rotation_rad or angle < 1e-8:
        return rot.as_matrix()
    return Rotation.from_rotvec(rotvec * (max_rotation_rad / angle)).as_matrix()


def _rotate_points_about_pivot(
    points_world: np.ndarray,
    rotation_matrix: np.ndarray,
    pivot_world: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    pivot = np.asarray(pivot_world, dtype=np.float64)
    return (np.asarray(rotation_matrix, dtype=np.float64) @ (pts - pivot[None, :]).T).T + pivot[None, :]


def _limit_delta_rotation_by_palm_hemisphere(
    *,
    base_rotation: np.ndarray,
    delta_rotation: np.ndarray,
    palm_normal_local: np.ndarray,
    reference_palm_world: np.ndarray,
    min_alignment_dot: float,
) -> np.ndarray:
    palm_local = _safe_normalize(np.asarray(palm_normal_local, dtype=np.float64))
    reference_world = _safe_normalize(np.asarray(reference_palm_world, dtype=np.float64))
    if np.linalg.norm(palm_local) < 1e-8 or np.linalg.norm(reference_world) < 1e-8:
        return np.asarray(delta_rotation, dtype=np.float64)

    base_rot = np.asarray(base_rotation, dtype=np.float64)
    candidate_delta = np.asarray(delta_rotation, dtype=np.float64)
    candidate_palm_world = _safe_normalize((candidate_delta @ base_rot) @ palm_local)
    if float(np.dot(candidate_palm_world, reference_world)) >= float(min_alignment_dot):
        return candidate_delta

    delta_rotvec = Rotation.from_matrix(candidate_delta).as_rotvec()
    angle = float(np.linalg.norm(delta_rotvec))
    if angle < 1e-8:
        return candidate_delta

    lo = 0.0
    hi = 1.0
    best_alpha = 0.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        test_delta = Rotation.from_rotvec(delta_rotvec * mid).as_matrix()
        test_palm_world = _safe_normalize((test_delta @ base_rot) @ palm_local)
        if float(np.dot(test_palm_world, reference_world)) >= float(min_alignment_dot):
            best_alpha = mid
            lo = mid
        else:
            hi = mid

    return Rotation.from_rotvec(delta_rotvec * best_alpha).as_matrix()


def _align_pose_to_stage_targets(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    hand_pose: HandPose,
    stage_name: StageName,
    *,
    alignment_mode: str = "full_rigid",
    max_translation: Optional[float] = None,
    max_rotation_deg: Optional[float] = None,
    preserve_palm_hemisphere: bool = False,
    reference_palm_world: Optional[np.ndarray] = None,
    min_palm_alignment_dot: float = 0.0,
) -> HandPose:
    point_names = [target.name for target in contact_result.active_targets]
    if len(point_names) == 0:
        return hand_pose

    semantic_states = hand_model.semantic_points_world(
        runtime_cfg,
        hand_pose,
        point_names=point_names,
    )

    actual_centers: List[np.ndarray] = []
    desired_centers: List[np.ndarray] = []
    for target in contact_result.active_targets:
        sphere_state = semantic_states.get(target.name)
        if sphere_state is None:
            continue
        actual_centers.append(np.asarray(sphere_state.center_world, dtype=np.float64))
        desired_centers.append(
            _target_desired_center(
                target,
                sphere_state,
                stage_name,
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
            )
        )

    if len(actual_centers) == 0:
        return hand_pose

    actual_centers_arr = np.asarray(actual_centers, dtype=np.float64)
    desired_centers_arr = np.asarray(desired_centers, dtype=np.float64)
    wrist_position = np.asarray(hand_pose.wrist_position, dtype=np.float64)
    wrist_rotation = hand_pose.rotation_matrix()
    palm_normal_local = _safe_normalize(
        np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )

    if alignment_mode == "none":
        delta_R = np.eye(3, dtype=np.float64)
        delta_t = np.zeros(3, dtype=np.float64)
    elif alignment_mode == "translation_only":
        delta_R = np.eye(3, dtype=np.float64)
        delta_t = np.mean(desired_centers_arr - actual_centers_arr, axis=0)
    elif alignment_mode == "rigid_clamped":
        full_delta_R, _ = _fit_rigid_alignment(
            actual_centers_arr,
            desired_centers_arr,
        )
        delta_R = _clamp_rotation_matrix(full_delta_R, max_rotation_deg)
        rotated_actual = (
            delta_R @ (actual_centers_arr - wrist_position[None, :]).T
        ).T + wrist_position[None, :]
        delta_t = np.mean(desired_centers_arr - rotated_actual, axis=0)
    elif alignment_mode == "full_rigid":
        delta_R, _ = _fit_rigid_alignment(
            actual_centers_arr,
            desired_centers_arr,
        )
    else:
        raise ValueError(f"Unsupported alignment mode: {alignment_mode}")

    if preserve_palm_hemisphere and reference_palm_world is not None:
        delta_R = _limit_delta_rotation_by_palm_hemisphere(
            base_rotation=wrist_rotation,
            delta_rotation=delta_R,
            palm_normal_local=palm_normal_local,
            reference_palm_world=np.asarray(reference_palm_world, dtype=np.float64),
            min_alignment_dot=float(min_palm_alignment_dot),
        )

    if alignment_mode in {"rigid_clamped", "full_rigid"}:
        rotated_actual = _rotate_points_about_pivot(
            actual_centers_arr,
            delta_R,
            wrist_position,
        )
        delta_t = np.mean(desired_centers_arr - rotated_actual, axis=0)

    if max_translation is not None:
        max_translation = float(max_translation)
        delta_t_norm = float(np.linalg.norm(delta_t))
        if delta_t_norm > max_translation and delta_t_norm > 1e-8:
            delta_t = delta_t * (max_translation / delta_t_norm)

    wrist_rotation = delta_R @ wrist_rotation
    wrist_position = wrist_position + delta_t
    return make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(wrist_rotation),
        wrist_rotation=wrist_rotation,
        joint_positions=hand_pose.joint_positions,
    )


def _refine_pose_roll_about_forward_axis(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    hand_pose: HandPose,
    stage_name: StageName,
    *,
    max_roll_deg: float,
    num_steps: int,
    regularization: float,
    thumb_weight: float,
) -> HandPose:
    point_names = [target.name for target in contact_result.active_targets]
    if len(point_names) == 0:
        return hand_pose

    forward_local = _safe_normalize(
        np.asarray(runtime_cfg.hand.frame_convention.finger_forward_local, dtype=np.float64)
    )
    if np.linalg.norm(forward_local) < 1e-8:
        return hand_pose

    base_rotation = hand_pose.rotation_matrix()
    wrist_position = np.asarray(hand_pose.wrist_position, dtype=np.float64)
    forward_world = _safe_normalize(base_rotation @ forward_local)
    if np.linalg.norm(forward_world) < 1e-8:
        return hand_pose

    max_roll_rad = np.deg2rad(float(max_roll_deg))
    num_steps = max(3, int(num_steps))
    candidate_angles = np.linspace(-max_roll_rad, max_roll_rad, num_steps, dtype=np.float64)

    best_angle = 0.0
    best_cost = float("inf")
    for angle in candidate_angles:
        delta_rotation = Rotation.from_rotvec(forward_world * float(angle)).as_matrix()
        candidate_rotation = delta_rotation @ base_rotation
        candidate_pose = make_hand_pose(
            wrist_position=wrist_position,
            wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(candidate_rotation),
            wrist_rotation=candidate_rotation,
            joint_positions=hand_pose.joint_positions,
        )
        semantic_states = hand_model.semantic_points_world(
            runtime_cfg,
            candidate_pose,
            point_names=point_names,
        )

        actual_centers: List[np.ndarray] = []
        desired_centers: List[np.ndarray] = []
        weights: List[float] = []
        for target in contact_result.active_targets:
            sphere_state = semantic_states.get(target.name)
            if sphere_state is None:
                continue
            actual_centers.append(np.asarray(sphere_state.center_world, dtype=np.float64))
            desired_centers.append(
                _target_desired_center(
                    target,
                    sphere_state,
                    stage_name,
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                )
            )
            weight = max(float(target.weight), 1e-6)
            if _is_thumb_target(target):
                weight *= float(thumb_weight)
            weights.append(weight)

        if len(actual_centers) == 0:
            continue

        actual_arr = np.asarray(actual_centers, dtype=np.float64)
        desired_arr = np.asarray(desired_centers, dtype=np.float64)
        weight_arr = np.asarray(weights, dtype=np.float64)
        delta_t = np.average(desired_arr - actual_arr, axis=0, weights=weight_arr)
        residual = desired_arr - (actual_arr + delta_t[None, :])
        cost = float(np.sum(weight_arr[:, None] * (residual ** 2)))
        cost += float(regularization) * float(angle ** 2)
        if cost < best_cost:
            best_cost = cost
            best_angle = float(angle)

    if abs(best_angle) < 1e-10:
        return hand_pose

    refined_rotation = Rotation.from_rotvec(forward_world * best_angle).as_matrix() @ base_rotation
    return make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(refined_rotation),
        wrist_rotation=refined_rotation,
        joint_positions=hand_pose.joint_positions,
    )


def _refine_pose_pitch_about_side_axis(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    hand_pose: HandPose,
    stage_name: StageName,
    *,
    max_pitch_deg: float,
    num_steps: int,
    regularization: float,
    thumb_weight: float,
) -> HandPose:
    point_names = [target.name for target in contact_result.active_targets]
    if len(point_names) == 0:
        return hand_pose

    side_local = _safe_normalize(
        np.asarray(runtime_cfg.hand.frame_convention.thumb_opposition_local, dtype=np.float64)
    )
    if np.linalg.norm(side_local) < 1e-8:
        side_local = _safe_normalize(
            np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
        )
    if np.linalg.norm(side_local) < 1e-8:
        return hand_pose

    base_rotation = hand_pose.rotation_matrix()
    wrist_position = np.asarray(hand_pose.wrist_position, dtype=np.float64)
    side_world = _safe_normalize(base_rotation @ side_local)
    if np.linalg.norm(side_world) < 1e-8:
        return hand_pose

    max_pitch_rad = np.deg2rad(float(max_pitch_deg))
    num_steps = max(3, int(num_steps))
    candidate_angles = np.linspace(-max_pitch_rad, max_pitch_rad, num_steps, dtype=np.float64)

    best_angle = 0.0
    best_cost = float("inf")
    for angle in candidate_angles:
        delta_rotation = Rotation.from_rotvec(side_world * float(angle)).as_matrix()
        candidate_rotation = delta_rotation @ base_rotation
        candidate_pose = make_hand_pose(
            wrist_position=wrist_position,
            wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(candidate_rotation),
            wrist_rotation=candidate_rotation,
            joint_positions=hand_pose.joint_positions,
        )
        semantic_states = hand_model.semantic_points_world(
            runtime_cfg,
            candidate_pose,
            point_names=point_names,
        )

        actual_centers: List[np.ndarray] = []
        desired_centers: List[np.ndarray] = []
        weights: List[float] = []
        for target in contact_result.active_targets:
            sphere_state = semantic_states.get(target.name)
            if sphere_state is None:
                continue
            actual_centers.append(np.asarray(sphere_state.center_world, dtype=np.float64))
            desired_centers.append(
                _target_desired_center(
                    target,
                    sphere_state,
                    stage_name,
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                )
            )
            weight = max(float(target.weight), 1e-6)
            if _is_thumb_target(target):
                weight *= float(thumb_weight)
            weights.append(weight)

        if len(actual_centers) == 0:
            continue

        actual_arr = np.asarray(actual_centers, dtype=np.float64)
        desired_arr = np.asarray(desired_centers, dtype=np.float64)
        weight_arr = np.asarray(weights, dtype=np.float64)
        delta_t = np.average(desired_arr - actual_arr, axis=0, weights=weight_arr)
        residual = desired_arr - (actual_arr + delta_t[None, :])
        cost = float(np.sum(weight_arr[:, None] * (residual ** 2)))
        cost += float(regularization) * float(angle ** 2)
        if cost < best_cost:
            best_cost = cost
            best_angle = float(angle)

    if abs(best_angle) < 1e-10:
        return hand_pose

    refined_rotation = Rotation.from_rotvec(side_world * best_angle).as_matrix() @ base_rotation
    return make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(refined_rotation),
        wrist_rotation=refined_rotation,
        joint_positions=hand_pose.joint_positions,
    )


def _blend_pose_hint(
    previous_pose: HandPose,
    stage_hint_pose: HandPose,
    *,
    wrist_alpha: float,
    joint_alpha: float,
    joint_names: Sequence[str],
) -> HandPose:
    wrist_position = (
        (1.0 - float(wrist_alpha)) * np.asarray(previous_pose.wrist_position, dtype=np.float64)
        + float(wrist_alpha) * np.asarray(stage_hint_pose.wrist_position, dtype=np.float64)
    )
    prev_rot = Rotation.from_matrix(previous_pose.rotation_matrix())
    hint_rot = Rotation.from_matrix(stage_hint_pose.rotation_matrix())
    delta_rotvec = (prev_rot.inv() * hint_rot).as_rotvec()
    wrist_rotation = (prev_rot * Rotation.from_rotvec(float(wrist_alpha) * delta_rotvec)).as_matrix()
    joint_positions = _blend_joint_positions(
        previous_pose.joint_positions,
        stage_hint_pose.joint_positions,
        joint_alpha,
        joint_names,
    )
    return make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(wrist_rotation),
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )


def _stage_target_refine_pose(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model: HandKinematicsModel,
    contact_result: ContactResolutionResult,
    hand_pose: HandPose,
    *,
    stage_name: StageName,
    roll_max_deg: float,
    pitch_max_deg: float,
    alignment_mode: str,
    alignment_max_translation: float,
    alignment_max_rotation_deg: float,
    thumb_weight: float,
    roll_num_steps: int = 11,
    pitch_num_steps: int = 11,
) -> HandPose:
    refined_pose = _refine_pose_roll_about_forward_axis(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        hand_pose=hand_pose,
        stage_name=stage_name,
        max_roll_deg=float(roll_max_deg),
        num_steps=max(3, int(roll_num_steps)),
        regularization=0.01,
        thumb_weight=float(thumb_weight),
    )
    refined_pose = _refine_pose_pitch_about_side_axis(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        hand_pose=refined_pose,
        stage_name=stage_name,
        max_pitch_deg=float(pitch_max_deg),
        num_steps=max(3, int(pitch_num_steps)),
        regularization=0.01,
        thumb_weight=float(thumb_weight),
    )
    reference_palm_world = _safe_normalize(
        refined_pose.rotation_matrix()
        @ np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )
    return _align_pose_to_stage_targets(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        hand_pose=refined_pose,
        stage_name=stage_name,
        alignment_mode=str(alignment_mode),
        max_translation=float(alignment_max_translation),
        max_rotation_deg=float(alignment_max_rotation_deg),
        preserve_palm_hemisphere=True,
        reference_palm_world=reference_palm_world,
        min_palm_alignment_dot=0.2,
    )


def _weighted_mean_active_normal(contact_result: ContactResolutionResult) -> np.ndarray:
    if not contact_result.active_targets:
        return _safe_normalize(contact_result.frame_R[:, 1])
    weights = np.asarray(
        [max(float(target.weight), 1e-6) for target in contact_result.active_targets],
        dtype=np.float64,
    )
    normals = np.asarray(
        [_safe_normalize(np.asarray(target.target_normal, dtype=np.float64)) for target in contact_result.active_targets],
        dtype=np.float64,
    )
    mean_normal = np.average(normals, axis=0, weights=weights)
    return _safe_normalize(mean_normal)


def _approach_direction_for_pose(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    hand_pose: HandPose,
) -> np.ndarray:
    palm_world = _safe_normalize(
        hand_pose.rotation_matrix()
        @ np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )
    approach_normal = _weighted_mean_active_normal(contact_result)

    approach_direction = np.zeros(3, dtype=np.float64)
    if str(contact_result.metadata.get("cat1_contact_mode", "pad")).lower() == "tip":
        palm_weight = float(contact_result.metadata.get("cat1_approach_palm_weight", 0.35))
        palm_weight = float(np.clip(palm_weight, 0.0, 1.0))
        normal_weight = 1.0 - palm_weight
        if np.linalg.norm(palm_world) > 1e-8:
            approach_direction += palm_weight * (-palm_world)
        if np.linalg.norm(approach_normal) > 1e-8:
            approach_direction += normal_weight * (-approach_normal)
    elif np.linalg.norm(palm_world) > 1e-8:
        approach_direction = -palm_world

    if np.linalg.norm(approach_direction) < 1e-8:
        if np.linalg.norm(approach_normal) > 1e-8:
            approach_direction = -approach_normal
    if np.linalg.norm(approach_direction) < 1e-8:
        approach_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return _safe_normalize(approach_direction)


def build_cat1_stage_initializations(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seed: PoseSeed,
) -> Dict[StageName, HandPose]:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    stage_init_cfg = cfg["stage_init"]
    joint_names = list(runtime_cfg.hand.joints.controllable)
    hand_model = load_hand_kinematics_model(runtime_cfg)
    pregrasp_alignment_mode = str(stage_init_cfg.get("pregrasp_alignment_mode", "translation_only"))
    pregrasp_roll_refine_enabled = bool(stage_init_cfg.get("pregrasp_roll_refine_enabled", True))
    pregrasp_roll_max_deg = float(stage_init_cfg.get("pregrasp_roll_max_deg", 35.0))
    pregrasp_roll_num_steps = int(stage_init_cfg.get("pregrasp_roll_num_steps", 25))
    pregrasp_roll_regularization = float(stage_init_cfg.get("pregrasp_roll_regularization", 0.01))
    pregrasp_roll_thumb_weight = float(stage_init_cfg.get("pregrasp_roll_thumb_weight", 2.0))
    pregrasp_pitch_refine_enabled = bool(stage_init_cfg.get("pregrasp_pitch_refine_enabled", True))
    pregrasp_pitch_max_deg = float(stage_init_cfg.get("pregrasp_pitch_max_deg", 20.0))
    pregrasp_pitch_num_steps = int(stage_init_cfg.get("pregrasp_pitch_num_steps", 17))
    pregrasp_pitch_regularization = float(stage_init_cfg.get("pregrasp_pitch_regularization", 0.015))
    pregrasp_pitch_thumb_weight = float(stage_init_cfg.get("pregrasp_pitch_thumb_weight", 1.5))
    pregrasp_alignment_max_translation = stage_init_cfg.get("pregrasp_alignment_max_translation", 0.05)
    pregrasp_alignment_max_rotation_deg = stage_init_cfg.get("pregrasp_alignment_max_rotation_deg", 35.0)
    grasp_refine_enabled = bool(stage_init_cfg.get("grasp_refine_enabled", True))
    grasp_roll_max_deg = float(stage_init_cfg.get("grasp_roll_max_deg", 12.0))
    grasp_pitch_max_deg = float(stage_init_cfg.get("grasp_pitch_max_deg", 8.0))
    grasp_alignment_mode = str(stage_init_cfg.get("grasp_alignment_mode", "rigid_clamped"))
    grasp_alignment_max_translation = float(stage_init_cfg.get("grasp_alignment_max_translation", 0.035))
    grasp_alignment_max_rotation_deg = float(stage_init_cfg.get("grasp_alignment_max_rotation_deg", 10.0))
    grasp_refine_num_steps = int(stage_init_cfg.get("grasp_refine_num_steps", 11))
    squeeze_refine_enabled = bool(stage_init_cfg.get("squeeze_refine_enabled", True))
    squeeze_roll_max_deg = float(stage_init_cfg.get("squeeze_roll_max_deg", 10.0))
    squeeze_pitch_max_deg = float(stage_init_cfg.get("squeeze_pitch_max_deg", 8.0))
    squeeze_alignment_mode = str(stage_init_cfg.get("squeeze_alignment_mode", "rigid_clamped"))
    squeeze_alignment_max_translation = float(stage_init_cfg.get("squeeze_alignment_max_translation", 0.03))
    squeeze_alignment_max_rotation_deg = float(stage_init_cfg.get("squeeze_alignment_max_rotation_deg", 8.0))
    squeeze_refine_num_steps = int(stage_init_cfg.get("squeeze_refine_num_steps", 11))
    pregrasp_preserve_seed_palm_hemisphere = bool(
        stage_init_cfg.get("pregrasp_preserve_seed_palm_hemisphere", True)
    )
    pregrasp_min_palm_alignment_dot = float(
        stage_init_cfg.get("pregrasp_min_palm_alignment_dot", 0.0)
    )

    open_pose = _resolve_posture(runtime_cfg, str(stage_init_cfg["open_posture_name"]))
    close_posture_name = _metadata_str(
        contact_result.metadata,
        "cat1_close_posture_name",
        str(stage_init_cfg["close_posture_name"]),
    )
    close_pose = _resolve_posture(runtime_cfg, close_posture_name)
    if not open_pose:
        open_pose = dict(seed.joint_positions)
    if not close_pose:
        close_pose = dict(seed.joint_positions)

    joint_pregrasp = _blend_joint_positions(
        open_pose,
        close_pose,
        float(stage_init_cfg["pregrasp_alpha"]),
        joint_names,
    )
    joint_grasp = _blend_joint_positions(
        open_pose,
        close_pose,
        float(stage_init_cfg["grasp_alpha"]),
        joint_names,
    )
    joint_squeeze = _blend_joint_positions(
        open_pose,
        close_pose,
        float(stage_init_cfg["squeeze_alpha"]),
        joint_names,
    )

    if contact_result.category == "cat4":
        seed_joint_pose = dict(seed.joint_positions)
        limit_close_pose = _derive_limit_close_pose(
            runtime_cfg,
            open_pose,
            close_pose,
            joint_names,
        )
        joint_groups = runtime_cfg.hand.joints.groups
        if isinstance(joint_groups, Mapping):
            thumb_joint_names = list(joint_groups.get("thumb", []))
        else:
            thumb_joint_names = list(getattr(joint_groups, "thumb", []))
        grasp_alpha = float(stage_init_cfg.get("grasp_alpha", 0.62))
        squeeze_alpha = float(stage_init_cfg.get("squeeze_alpha", 0.86))
        squeeze_alpha *= float(stage_init_cfg.get("cat4_squeeze_close_multiplier", 1.0))
        grasp_alpha = float(np.clip(grasp_alpha, 0.0, 1.0))
        squeeze_alpha = float(np.clip(squeeze_alpha, 0.0, 1.0))
        grasp_joint_pose = _blend_joint_positions(
            seed_joint_pose,
            limit_close_pose,
            grasp_alpha,
            joint_names,
        )
        squeeze_joint_pose = _blend_joint_positions(
            seed_joint_pose,
            limit_close_pose,
            squeeze_alpha,
            joint_names,
        )
        grasp_joint_pose = _blend_joint_subset_toward_pose(
            grasp_joint_pose,
            limit_close_pose,
            float(stage_init_cfg.get("cat4_grasp_thumb_close_boost", 0.0)),
            thumb_joint_names,
        )
        squeeze_joint_pose = _blend_joint_subset_toward_pose(
            squeeze_joint_pose,
            limit_close_pose,
            float(stage_init_cfg.get("cat4_squeeze_thumb_close_boost", 0.0)),
            thumb_joint_names,
        )
        pregrasp_pose = make_hand_pose(
            wrist_position=seed.wrist_position,
            wrist_quaternion_xyzw=seed.wrist_quaternion_xyzw,
            wrist_rotation=seed.wrist_rotation,
            joint_positions=seed_joint_pose,
        )
        grasp_pose = make_hand_pose(
            wrist_position=seed.wrist_position,
            wrist_quaternion_xyzw=seed.wrist_quaternion_xyzw,
            wrist_rotation=seed.wrist_rotation,
            joint_positions=grasp_joint_pose,
        )
        squeeze_pose = make_hand_pose(
            wrist_position=seed.wrist_position,
            wrist_quaternion_xyzw=seed.wrist_quaternion_xyzw,
            wrist_rotation=seed.wrist_rotation,
            joint_positions=squeeze_joint_pose,
        )
        return {
            "pregrasp": pregrasp_pose,
            "grasp": grasp_pose,
            "squeeze": squeeze_pose,
        }

    clearance_pre = max(
        [_clearance_for_stage(target, "pregrasp") for target in contact_result.active_targets] or [0.01]
    )
    clearance_grasp = max(
        [_clearance_for_stage(target, "grasp") for target in contact_result.active_targets] or [0.003]
    )
    clearance_squeeze = max(
        [_clearance_for_stage(target, "squeeze") for target in contact_result.active_targets] or [0.0]
    )

    grasp_shift = max(
        max(clearance_pre - clearance_grasp, 0.0),
        float(stage_init_cfg.get("grasp_shift_min", 0.012)),
    ) * float(stage_init_cfg["grasp_approach_scale"])
    squeeze_shift = max(
        max(clearance_grasp - clearance_squeeze, 0.0),
        float(stage_init_cfg.get("squeeze_shift_min", 0.007)),
    ) * float(stage_init_cfg["squeeze_approach_scale"])
    grasp_shift *= _metadata_float(contact_result.metadata, "cat1_grasp_shift_scale", 1.0)
    squeeze_shift *= _metadata_float(contact_result.metadata, "cat1_squeeze_shift_scale", 1.0)

    cat3_fast_cfg = cfg.get("contact_model", {}).get("cat3_fast_pose_search", {})
    if contact_result.category == "cat3" and bool(cat3_fast_cfg.get("enabled", True)):
        seed_joint_pose = dict(seed.joint_positions)
        contact_posture_name = str(cat3_fast_cfg.get("contact_joint_posture", "") or "")
        contact_joint_pose = (
            dict(runtime_cfg.hand.default_postures[contact_posture_name])
            if contact_posture_name in runtime_cfg.hand.default_postures
            else dict(seed_joint_pose)
        )
        squeeze_base_name = str(cat3_fast_cfg.get("squeeze_base_posture", "") or "")
        squeeze_base_pose = (
            dict(runtime_cfg.hand.default_postures[squeeze_base_name])
            if squeeze_base_name in runtime_cfg.hand.default_postures
            else dict(seed_joint_pose)
        )
        use_legacy_cat3_squeeze = False
        if (
            squeeze_base_name == "cat3_side_press"
            and squeeze_base_name not in runtime_cfg.hand.default_postures
        ):
            use_legacy_cat3_squeeze = True
            if "cat3_fingers_down" in runtime_cfg.hand.default_postures:
                squeeze_base_pose = dict(runtime_cfg.hand.default_postures["cat3_fingers_down"])
        limit_close_pose = _derive_limit_close_pose(
            runtime_cfg,
            open_pose,
            close_pose,
            joint_names,
        )
        grasp_joint_pose = _blend_joint_positions(
            contact_joint_pose,
            close_pose,
            float(np.clip(float(cat3_fast_cfg.get("grasp_joint_close_blend", 0.0)), 0.0, 1.0)),
            joint_names,
        )
        squeeze_close_blend = float(
            np.clip(float(cat3_fast_cfg.get("squeeze_joint_close_blend", 0.0)), 0.0, 1.0)
        )
        if use_legacy_cat3_squeeze:
            squeeze_close_blend = max(squeeze_close_blend, 0.15)
        squeeze_target_pose = (
            limit_close_pose
            if (
                bool(cat3_fast_cfg.get("squeeze_toward_limit_close", True))
                or use_legacy_cat3_squeeze
            )
            else close_pose
        )
        squeeze_joint_pose = _blend_joint_positions(
            squeeze_base_pose,
            squeeze_target_pose,
            squeeze_close_blend,
            joint_names,
        )

        pregrasp_pose = make_hand_pose(
            wrist_position=seed.wrist_position,
            wrist_quaternion_xyzw=seed.wrist_quaternion_xyzw,
            wrist_rotation=seed.wrist_rotation,
            joint_positions=seed_joint_pose,
        )

        def direct_cat3_stage_pose(stage_name: StageName, joint_pose: Mapping[str, float]) -> HandPose:
            stage_pose = make_hand_pose(
                wrist_position=seed.wrist_position,
                wrist_quaternion_xyzw=seed.wrist_quaternion_xyzw,
                wrist_rotation=seed.wrist_rotation,
                joint_positions=joint_pose,
            )
            if bool(cat3_fast_cfg.get("direct_target_translation", True)):
                stage_pose = _align_pose_to_stage_targets(
                    runtime_cfg=runtime_cfg,
                    hand_model=hand_model,
                    contact_result=contact_result,
                    hand_pose=stage_pose,
                    stage_name=stage_name,
                    alignment_mode="translation_only",
                    max_translation=float(cat3_fast_cfg.get("direct_alignment_max_translation", 0.090)),
                )
            return stage_pose

        grasp_pose = direct_cat3_stage_pose("grasp", grasp_joint_pose)
        squeeze_pose = make_hand_pose(
            wrist_position=grasp_pose.wrist_position,
            wrist_quaternion_xyzw=grasp_pose.wrist_quaternion_xyzw,
            wrist_rotation=grasp_pose.rotation_matrix(),
            joint_positions=squeeze_joint_pose,
        )

        return {
            "pregrasp": pregrasp_pose,
            "grasp": grasp_pose,
            "squeeze": squeeze_pose,
        }

    pregrasp_pose = make_hand_pose(
        wrist_position=seed.wrist_position,
        wrist_quaternion_xyzw=seed.wrist_quaternion_xyzw,
        wrist_rotation=seed.wrist_rotation,
        joint_positions=joint_pregrasp,
    )
    if pregrasp_roll_refine_enabled:
        pregrasp_pose = _refine_pose_roll_about_forward_axis(
            runtime_cfg=runtime_cfg,
            hand_model=hand_model,
            contact_result=contact_result,
            hand_pose=pregrasp_pose,
            stage_name="pregrasp",
            max_roll_deg=pregrasp_roll_max_deg,
            num_steps=pregrasp_roll_num_steps,
            regularization=pregrasp_roll_regularization,
            thumb_weight=pregrasp_roll_thumb_weight,
        )
    if pregrasp_pitch_refine_enabled:
        pregrasp_pose = _refine_pose_pitch_about_side_axis(
            runtime_cfg=runtime_cfg,
            hand_model=hand_model,
            contact_result=contact_result,
            hand_pose=pregrasp_pose,
            stage_name="pregrasp",
            max_pitch_deg=pregrasp_pitch_max_deg,
            num_steps=pregrasp_pitch_num_steps,
            regularization=pregrasp_pitch_regularization,
            thumb_weight=pregrasp_pitch_thumb_weight,
        )
    reference_palm_world = _safe_normalize(
        pregrasp_pose.rotation_matrix()
        @ np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )
    pregrasp_pose = _align_pose_to_stage_targets(
        runtime_cfg=runtime_cfg,
        hand_model=hand_model,
        contact_result=contact_result,
        hand_pose=pregrasp_pose,
        stage_name="pregrasp",
        alignment_mode=pregrasp_alignment_mode,
        max_translation=pregrasp_alignment_max_translation,
        max_rotation_deg=pregrasp_alignment_max_rotation_deg,
        preserve_palm_hemisphere=pregrasp_preserve_seed_palm_hemisphere,
        reference_palm_world=reference_palm_world,
        min_palm_alignment_dot=pregrasp_min_palm_alignment_dot,
    )

    approach_direction = _approach_direction_for_pose(
        runtime_cfg,
        contact_result,
        pregrasp_pose,
    )

    grasp_pose = make_hand_pose(
        wrist_position=np.asarray(pregrasp_pose.wrist_position, dtype=np.float64) + approach_direction * grasp_shift,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(pregrasp_pose.rotation_matrix()),
        wrist_rotation=pregrasp_pose.rotation_matrix(),
        joint_positions=joint_grasp,
    )
    if grasp_refine_enabled:
        grasp_pose = _stage_target_refine_pose(
            runtime_cfg=runtime_cfg,
            hand_model=hand_model,
            contact_result=contact_result,
            hand_pose=grasp_pose,
            stage_name="grasp",
            roll_max_deg=grasp_roll_max_deg,
            pitch_max_deg=grasp_pitch_max_deg,
            alignment_mode=grasp_alignment_mode,
            alignment_max_translation=grasp_alignment_max_translation,
            alignment_max_rotation_deg=grasp_alignment_max_rotation_deg,
            thumb_weight=float(cfg["stage_weights"]["grasp"].get("thumb_active_multiplier", 1.0)),
            roll_num_steps=grasp_refine_num_steps,
            pitch_num_steps=grasp_refine_num_steps,
        )

    squeeze_pose = make_hand_pose(
        wrist_position=np.asarray(grasp_pose.wrist_position, dtype=np.float64) + approach_direction * squeeze_shift,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(grasp_pose.rotation_matrix()),
        wrist_rotation=grasp_pose.rotation_matrix(),
        joint_positions=joint_squeeze,
    )
    if squeeze_refine_enabled:
        squeeze_pose = _stage_target_refine_pose(
            runtime_cfg=runtime_cfg,
            hand_model=hand_model,
            contact_result=contact_result,
            hand_pose=squeeze_pose,
            stage_name="squeeze",
            roll_max_deg=squeeze_roll_max_deg,
            pitch_max_deg=squeeze_pitch_max_deg,
            alignment_mode=squeeze_alignment_mode,
            alignment_max_translation=squeeze_alignment_max_translation,
            alignment_max_rotation_deg=squeeze_alignment_max_rotation_deg,
            thumb_weight=float(cfg["stage_weights"]["squeeze"].get("thumb_active_multiplier", 1.0)),
            roll_num_steps=squeeze_refine_num_steps,
            pitch_num_steps=squeeze_refine_num_steps,
        )

    return {
        "pregrasp": pregrasp_pose,
        "grasp": grasp_pose,
        "squeeze": squeeze_pose,
    }


def _pack_pose_variables(
    runtime_cfg: ResolvedHandRuntimeConfig,
    init_pose: HandPose,
) -> tuple[np.ndarray, np.ndarray, List[tuple[Optional[float], Optional[float]]]]:
    joint_names = list(runtime_cfg.hand.joints.controllable)
    x0 = np.zeros(6 + len(joint_names), dtype=np.float64)
    x0[:3] = np.asarray(init_pose.wrist_position, dtype=np.float64)
    x0[3:6] = 0.0
    for idx, joint_name in enumerate(joint_names):
        x0[6 + idx] = float(init_pose.joint_positions.get(joint_name, 0.0))

    base_rotation = init_pose.rotation_matrix()
    bounds: List[tuple[Optional[float], Optional[float]]] = [(None, None)] * 6
    for joint_name in joint_names:
        lo_hi = runtime_cfg.hand.joints.limits.get(joint_name)
        if lo_hi is None:
            bounds.append((None, None))
        else:
            bounds.append((float(lo_hi[0]), float(lo_hi[1])))
    return x0, base_rotation, bounds


def _apply_stage_motion_bounds(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    stage_name: StageName,
    x0: np.ndarray,
    bounds: Sequence[tuple[Optional[float], Optional[float]]],
) -> List[tuple[Optional[float], Optional[float]]]:
    bounded = list(bounds)
    cfg = _merged_optimizer_cfg(runtime_cfg)
    motion_cfg = cfg.get("stage_motion_bounds", {})
    stage_cfg = motion_cfg.get(stage_name, {})
    if not stage_cfg:
        return bounded

    translation_limit = stage_cfg.get("translation")
    if translation_limit is not None:
        translation_limit = max(float(translation_limit), 0.0)
        for idx in range(3):
            center = float(x0[idx])
            bounded[idx] = (center - translation_limit, center + translation_limit)

    rotation_deg_limit = stage_cfg.get("rotation_deg")
    if rotation_deg_limit is not None:
        rotation_limit = max(np.deg2rad(float(rotation_deg_limit)), 0.0)
        for idx in range(3, 6):
            bounded[idx] = (-rotation_limit, rotation_limit)

    joint_delta_limit = stage_cfg.get("joint_delta")
    if joint_delta_limit is not None:
        joint_delta_limit = max(float(joint_delta_limit), 0.0)
        for idx in range(6, len(x0)):
            center = float(x0[idx])
            lower, upper = bounded[idx]
            new_lower = center - joint_delta_limit
            new_upper = center + joint_delta_limit
            if lower is not None:
                new_lower = max(float(lower), new_lower)
            if upper is not None:
                new_upper = min(float(upper), new_upper)
            bounded[idx] = (new_lower, new_upper)

    if contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"}:
        joint_names = list(runtime_cfg.hand.joints.controllable)
        stage_init_cfg = cfg.get("stage_init", {})
        open_pose = _resolve_posture(runtime_cfg, str(stage_init_cfg.get("open_posture_name", "open_hand")))
        close_pose = _resolve_posture(runtime_cfg, str(stage_init_cfg.get("close_posture_name", "light_close")))
        opening_slack = max(float(stage_cfg.get("joint_opening_slack", 0.02)), 0.0)
        for joint_idx, joint_name in enumerate(joint_names, start=6):
            open_value = float(open_pose.get(joint_name, 0.0))
            close_value = float(close_pose.get(joint_name, open_value))
            close_delta = close_value - open_value
            if abs(close_delta) <= 1e-6:
                continue
            lower, upper = bounded[joint_idx]
            center = float(x0[joint_idx])
            if close_delta > 0.0:
                new_lower = center - opening_slack
                if lower is not None:
                    new_lower = max(float(lower), new_lower)
                bounded[joint_idx] = (new_lower, upper)
            else:
                new_upper = center + opening_slack
                if upper is not None:
                    new_upper = min(float(upper), new_upper)
                bounded[joint_idx] = (lower, new_upper)

    return bounded


def _unpack_pose_variables(
    runtime_cfg: ResolvedHandRuntimeConfig,
    *,
    x: np.ndarray,
    base_rotation: np.ndarray,
) -> HandPose:
    joint_names = list(runtime_cfg.hand.joints.controllable)
    wrist_position = np.asarray(x[:3], dtype=np.float64)
    rot_delta = np.asarray(x[3:6], dtype=np.float64)
    wrist_rotation = base_rotation @ Rotation.from_rotvec(rot_delta).as_matrix()
    joint_positions = {
        joint_name: float(x[6 + idx])
        for idx, joint_name in enumerate(joint_names)
    }
    return make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(wrist_rotation),
        wrist_rotation=wrist_rotation,
        joint_positions=joint_positions,
    )


def _cat1_stage_target_local_point(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    sample_blend_cfg = contact_model_cfg.get("cat1_sample_blend", {})
    blend_alpha = _stage_cfg_scalar(
        sample_blend_cfg,
        stage_name,
        {"pregrasp": 0.10, "grasp": 0.30, "squeeze": 0.55}[stage_name],
    )
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))

    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    reference_local = frame_R.T @ (
        np.asarray(target.reference_point, dtype=np.float64) - anchor_point
    )
    sample_local = frame_R.T @ (
        np.asarray(target.target_point, dtype=np.float64) - anchor_point
    )
    return (1.0 - blend_alpha) * reference_local + blend_alpha * sample_local


def _cat1_stage_target_point_world(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    return anchor_point + frame_R @ _cat1_stage_target_local_point(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )


def _cat2_stage_target_point_world(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    sample_blend_cfg = contact_model_cfg.get("cat2_sample_blend", {})
    blend_alpha = _stage_cfg_scalar(
        sample_blend_cfg,
        stage_name,
        {"pregrasp": 0.10, "grasp": 0.22, "squeeze": 0.32}[stage_name],
    )
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))

    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)

    reference_local = frame_R.T @ (
        np.asarray(target.reference_point, dtype=np.float64) - anchor_point
    )
    sample_local = frame_R.T @ (
        np.asarray(target.target_point, dtype=np.float64) - anchor_point
    )
    blended_local = (1.0 - blend_alpha) * reference_local + blend_alpha * sample_local

    if _is_thumb_target(target):
        return anchor_point + frame_R @ blended_local

    finger_targets = [t for t in contact_result.active_targets if not _is_thumb_target(t)]
    if len(finger_targets) <= 1:
        return anchor_point + frame_R @ blended_local

    ref_x_values: List[float] = []
    blended_x_values: List[float] = []
    for finger_target in finger_targets:
        ref_local = frame_R.T @ (
            np.asarray(finger_target.reference_point, dtype=np.float64) - anchor_point
        )
        sample_local_i = frame_R.T @ (
            np.asarray(finger_target.target_point, dtype=np.float64) - anchor_point
        )
        blended_local_i = (1.0 - blend_alpha) * ref_local + blend_alpha * sample_local_i
        ref_x_values.append(float(ref_local[0]))
        blended_x_values.append(float(blended_local_i[0]))

    x_shift = float(np.mean(np.asarray(ref_x_values, dtype=np.float64)) - np.mean(np.asarray(blended_x_values, dtype=np.float64)))
    adjusted_local = blended_local.copy()
    adjusted_local[0] += x_shift
    return anchor_point + frame_R @ adjusted_local


def _cat3_stage_target_local_point(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    sample_blend_cfg = contact_model_cfg.get("cat3_sample_blend", {})
    blend_alpha = _stage_cfg_scalar(
        sample_blend_cfg,
        stage_name,
        {"pregrasp": 0.10, "grasp": 0.35, "squeeze": 0.55}[stage_name],
    )
    if _is_palm_target(target):
        blend_alpha = 0.0
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))

    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    reference_local = frame_R.T @ (
        np.asarray(target.reference_point, dtype=np.float64) - anchor_point
    )
    sample_local = frame_R.T @ (
        np.asarray(target.target_point, dtype=np.float64) - anchor_point
    )
    return (1.0 - blend_alpha) * reference_local + blend_alpha * sample_local


def _cat3_stage_target_point_world(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    return anchor_point + frame_R @ _cat3_stage_target_local_point(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )


def _cat4_stage_target_local_point(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    sample_blend_cfg = contact_model_cfg.get("cat4_sample_blend", {})
    blend_alpha = _stage_cfg_scalar(
        sample_blend_cfg,
        stage_name,
        {"pregrasp": 0.10, "grasp": 0.90, "squeeze": 1.00}[stage_name],
    )
    if _is_declared_tip_target(runtime_cfg, target):
        tip_blend_cfg = contact_model_cfg.get("cat4_tip_sample_blend", {})
        tip_blend_min = _stage_cfg_scalar(
            tip_blend_cfg,
            stage_name,
            {"pregrasp": 0.20, "grasp": 0.55, "squeeze": 0.85}[stage_name],
        )
        blend_alpha = max(float(blend_alpha), float(tip_blend_min))
    if _is_palm_target(target):
        blend_alpha = 0.0
    elif (
        _is_thumb_target(target)
        and not _is_declared_tip_target(runtime_cfg, target)
        and bool(contact_model_cfg.get("cat4_thumb_outside", {}).get("enabled", False))
    ):
        thumb_outside_cfg = contact_model_cfg.get("cat4_thumb_outside", {})
        blend_alpha *= float(thumb_outside_cfg.get("thumb_blend_multiplier", 0.35))
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))

    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    reference_local = frame_R.T @ (
        np.asarray(target.reference_point, dtype=np.float64) - anchor_point
    )
    sample_local = frame_R.T @ (
        np.asarray(target.target_point, dtype=np.float64) - anchor_point
    )
    return (1.0 - blend_alpha) * reference_local + blend_alpha * sample_local


def _cat4_stage_target_point_world(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
) -> np.ndarray:
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    return anchor_point + frame_R @ _cat4_stage_target_local_point(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )


def _surface_contact_point_world(
    center_world: np.ndarray,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> Optional[np.ndarray]:
    grad = _safe_normalize(np.asarray(gradient_world, dtype=np.float64))
    if np.linalg.norm(grad) < 1e-8 or not np.isfinite(float(signed_distance)):
        return None
    return np.asarray(center_world, dtype=np.float64) - grad * float(signed_distance)


def _cat1_stage_position_error(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    sphere_state: CollisionSphereState,
    stage_name: StageName,
    *,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> float:
    surface_point_world = _surface_contact_point_world(
        sphere_state.center_world,
        signed_distance,
        gradient_world,
    )
    if surface_point_world is None:
        desired_center = _target_desired_center(
            target,
            sphere_state,
            stage_name,
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
        )
        pos_err = np.asarray(sphere_state.center_world, dtype=np.float64) - desired_center
        return float(np.dot(pos_err, pos_err))

    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    axis_weights = _stage_cfg_vec3(
        contact_model_cfg.get("cat1_local_axis_weights", {}),
        stage_name,
        {"pregrasp": [0.10, 1.00, 0.45], "grasp": [0.18, 1.00, 0.65], "squeeze": [0.28, 1.00, 0.80]}[stage_name],
    )
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    local_surface_point = frame_R.T @ (surface_point_world - anchor_point)
    local_target = _cat1_stage_target_local_point(runtime_cfg, contact_result, target, stage_name)
    local_error = (local_surface_point - local_target) * axis_weights
    return float(np.dot(local_error, local_error))


def _cat2_stage_position_error(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    sphere_state: CollisionSphereState,
    stage_name: StageName,
    *,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> float:
    surface_point_world = _surface_contact_point_world(
        sphere_state.center_world,
        signed_distance,
        gradient_world,
    )
    if surface_point_world is None:
        desired_center = _target_desired_center(
            target,
            sphere_state,
            stage_name,
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
        )
        pos_err = np.asarray(sphere_state.center_world, dtype=np.float64) - desired_center
        return float(np.dot(pos_err, pos_err))

    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    axis_weights = _stage_cfg_vec3(
        contact_model_cfg.get("cat2_local_axis_weights", {}),
        stage_name,
        {"pregrasp": [0.55, 0.30, 0.70], "grasp": [0.85, 0.40, 0.95], "squeeze": [1.00, 0.50, 1.10]}[stage_name],
    )
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    local_surface_point = frame_R.T @ (surface_point_world - anchor_point)
    local_target_world = _cat2_stage_target_point_world(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )
    local_target = frame_R.T @ (local_target_world - anchor_point)
    local_error = (local_surface_point - local_target) * axis_weights
    return float(np.dot(local_error, local_error))


def _cat2_thumb_side_guard_cost(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    sphere_state: CollisionSphereState,
    stage_name: StageName,
    *,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> float:
    if contact_result.category != "cat2" or not _is_thumb_target(target):
        return 0.0

    cfg = _merged_optimizer_cfg(runtime_cfg)
    guard_cfg = cfg.get("contact_model", {}).get("cat2_thumb_side_guard", {})
    if not bool(guard_cfg.get("enabled", True)):
        return 0.0

    stage_weight = _stage_cfg_scalar(
        guard_cfg,
        stage_name,
        {"pregrasp": 120.0, "grasp": 320.0, "squeeze": 420.0}[stage_name],
    )
    if stage_weight <= 0.0:
        return 0.0

    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    probe_world = _surface_contact_point_world(
        np.asarray(sphere_state.center_world, dtype=np.float64),
        signed_distance,
        gradient_world,
    )
    if probe_world is None:
        probe_world = np.asarray(sphere_state.center_world, dtype=np.float64)

    local_probe = frame_R.T @ (np.asarray(probe_world, dtype=np.float64) - anchor_point)
    local_reference = frame_R.T @ (
        np.asarray(target.reference_point, dtype=np.float64) - anchor_point
    )
    local_target = frame_R.T @ (
        np.asarray(target.target_point, dtype=np.float64) - anchor_point
    )

    patch_radius = float(contact_result.metadata.get("patch_radius", 0.03))
    side_extent = max(float(local_reference[1]), float(local_target[1]), 0.0)
    min_side_distance = float(guard_cfg.get("min_side_distance", 0.006))
    side_fraction = float(guard_cfg.get("side_fraction", 0.60))
    patch_radius_fraction = float(guard_cfg.get("patch_radius_fraction", 0.18))
    min_side = max(
        min_side_distance,
        side_fraction * side_extent,
        patch_radius_fraction * patch_radius,
    )

    min_local_z = float(guard_cfg.get("min_local_z", 0.0))
    z_weight_scale = float(guard_cfg.get("z_weight_scale", 0.75))

    side_deficit = max(0.0, min_side - float(local_probe[1]))
    z_deficit = max(0.0, min_local_z - float(local_probe[2]))
    return float(stage_weight) * (side_deficit**2 + z_weight_scale * z_deficit**2)


def _cat4_stage_position_error(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    sphere_state: CollisionSphereState,
    stage_name: StageName,
    *,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> float:
    surface_point_world = _surface_contact_point_world(
        sphere_state.center_world,
        signed_distance,
        gradient_world,
    )
    if surface_point_world is None:
        desired_center = _target_desired_center(
            target,
            sphere_state,
            stage_name,
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
        )
        pos_err = np.asarray(sphere_state.center_world, dtype=np.float64) - desired_center
        return float(np.dot(pos_err, pos_err))

    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    axis_weights = _stage_cfg_vec3(
        contact_model_cfg.get("cat4_local_axis_weights", {}),
        stage_name,
        {"pregrasp": [0.20, 1.00, 0.30], "grasp": [0.45, 1.00, 0.55], "squeeze": [0.60, 1.05, 0.75]}[stage_name],
    )
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    local_surface_point = frame_R.T @ (surface_point_world - anchor_point)
    local_target_world = _cat4_stage_target_point_world(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )
    local_target = frame_R.T @ (local_target_world - anchor_point)
    local_error = (local_surface_point - local_target) * axis_weights
    return float(np.dot(local_error, local_error))


def _cat3_stage_position_error(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    sphere_state: CollisionSphereState,
    stage_name: StageName,
    *,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> float:
    surface_point_world = _surface_contact_point_world(
        sphere_state.center_world,
        signed_distance,
        gradient_world,
    )
    if surface_point_world is None:
        desired_center = _target_desired_center(
            target,
            sphere_state,
            stage_name,
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
        )
        pos_err = np.asarray(sphere_state.center_world, dtype=np.float64) - desired_center
        return float(np.dot(pos_err, pos_err))

    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    axis_weights = _stage_cfg_vec3(
        contact_model_cfg.get("cat3_local_axis_weights", {}),
        stage_name,
        {"pregrasp": [0.50, 0.85, 1.00], "grasp": [0.70, 1.00, 1.15], "squeeze": [0.85, 1.05, 1.25]}[stage_name],
    )
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    local_surface_point = frame_R.T @ (surface_point_world - anchor_point)
    local_target_world = _cat3_stage_target_point_world(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )
    local_target = frame_R.T @ (local_target_world - anchor_point)
    local_error = (local_surface_point - local_target) * axis_weights
    return float(np.dot(local_error, local_error))


def _cat4_tip_stage_position_error(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    target: ContactTarget,
    stage_name: StageName,
    *,
    probe_world: np.ndarray,
    signed_distance: float,
    gradient_world: np.ndarray,
) -> float:
    surface_point_world = _surface_contact_point_world(
        probe_world,
        signed_distance,
        gradient_world,
    )
    if surface_point_world is None:
        target_world = _cat4_stage_target_point_world(
            runtime_cfg,
            contact_result,
            target,
            stage_name,
        )
        desired_probe = target_world + _safe_normalize(np.asarray(target.target_normal, dtype=np.float64)) * float(
            _clearance_for_stage(target, stage_name)
        )
        pos_err = np.asarray(probe_world, dtype=np.float64) - desired_probe
        return float(np.dot(pos_err, pos_err))

    cfg = _merged_optimizer_cfg(runtime_cfg)
    contact_model_cfg = cfg.get("contact_model", {})
    axis_weights = _stage_cfg_vec3(
        contact_model_cfg.get("cat4_local_axis_weights", {}),
        stage_name,
        {"pregrasp": [0.20, 1.00, 0.30], "grasp": [0.45, 1.00, 0.55], "squeeze": [0.60, 1.05, 0.75]}[stage_name],
    )
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
    local_surface_point = frame_R.T @ (surface_point_world - anchor_point)
    local_target = _cat4_stage_target_local_point(
        runtime_cfg,
        contact_result,
        target,
        stage_name,
    )
    local_error = (local_surface_point - local_target) * axis_weights
    return float(np.dot(local_error, local_error))


def _target_desired_center(
    target: ContactTarget,
    sphere_state: CollisionSphereState,
    stage_name: StageName,
    *,
    runtime_cfg: Optional[ResolvedHandRuntimeConfig] = None,
    contact_result: Optional[ContactResolutionResult] = None,
) -> np.ndarray:
    normal = _safe_normalize(np.asarray(target.target_normal, dtype=np.float64))
    target_clearance = _clearance_for_stage(target, stage_name)
    target_point = np.asarray(target.target_point, dtype=np.float64)
    if runtime_cfg is not None and contact_result is not None and contact_result.category == "cat1":
        target_point = _cat1_stage_target_point_world(
            runtime_cfg,
            contact_result,
            target,
            stage_name,
        )
    elif runtime_cfg is not None and contact_result is not None and contact_result.category == "cat2":
        target_point = _cat2_stage_target_point_world(
            runtime_cfg,
            contact_result,
            target,
            stage_name,
        )
    elif runtime_cfg is not None and contact_result is not None and contact_result.category == "cat3":
        target_point = _cat3_stage_target_point_world(
            runtime_cfg,
            contact_result,
            target,
            stage_name,
        )
    elif runtime_cfg is not None and contact_result is not None and contact_result.category == "cat4":
        target_point = _cat4_stage_target_point_world(
            runtime_cfg,
            contact_result,
            target,
            stage_name,
        )
    return target_point + normal * (sphere_state.radius + target_clearance)


def _contact_completion_tolerance(
    target: ContactTarget,
    completion_cfg: Mapping[str, float],
) -> float:
    if _is_thumb_target(target):
        return float(completion_cfg.get("thumb_contact_tolerance", completion_cfg.get("active_contact_tolerance", 0.0)))
    return float(completion_cfg.get("active_contact_tolerance", 0.0))


def _semantic_point_tags(
    semantic_point: SemanticPoint,
    point_name: str,
) -> set[str]:
    tags = set(semantic_point.role_tags)
    tags.update(point_name.replace("-", "_").split("_"))
    return {tag.lower() for tag in tags}


def _digit_rank(
    semantic_point: SemanticPoint,
    point_name: str,
) -> int:
    tags = _semantic_point_tags(semantic_point, point_name)
    for idx, digit in enumerate(("index", "middle", "ring", "little")):
        if digit in tags:
            return idx
    return 99


def _is_thumb_semantic_point(
    semantic_point: SemanticPoint,
    point_name: str,
) -> bool:
    return "thumb" in _semantic_point_tags(semantic_point, point_name)


def _cat1_tip_support_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
) -> List[str]:
    if contact_result.category != "cat1":
        return []
    if str(contact_result.metadata.get("cat1_contact_mode", "pad")).lower() != "tip":
        return []
    active_names = {target.name for target in contact_result.active_targets}
    support_candidates_by_digit: Dict[int, List[str]] = {}
    for point_name, semantic_point in runtime_cfg.semantic_points.items():
        if point_name in active_names:
            continue
        tags = _semantic_point_tags(semantic_point, point_name)
        if "contact" not in tags or "pad" not in tags or "avoid" in tags or "palm" in tags:
            continue
        if _is_thumb_semantic_point(semantic_point, point_name):
            continue
        digit_rank = _digit_rank(semantic_point, point_name)
        if digit_rank >= 99:
            continue
        support_candidates_by_digit.setdefault(digit_rank, []).append(point_name)

    selected_names: List[str] = []
    for digit_rank in sorted(support_candidates_by_digit):
        digit_points = sorted(
            support_candidates_by_digit[digit_rank],
            key=lambda name: (0 if "main" in name.replace("-", "_").split("_") else 1, name),
        )
        if digit_points:
            selected_names.append(digit_points[0])
    return selected_names


def _cat2_palm_support_targets(contact_result: ContactResolutionResult) -> List[ContactTarget]:
    if contact_result.category != "cat2":
        return []
    return [target for target in contact_result.avoid_targets if "palm" in str(target.name).lower()]


def _cat2_finger_support_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
) -> List[str]:
    if contact_result.category != "cat2":
        return []
    active_names = {target.name for target in contact_result.active_targets}
    active_keys: set[tuple[str, int]] = set()
    active_digit_ranks: set[int] = set()
    for target in contact_result.active_targets:
        semantic_point = runtime_cfg.semantic_points.get(target.name)
        if semantic_point is None or _is_thumb_target(target):
            continue
        active_keys.add((semantic_point.source_link, int(semantic_point.source_sphere_index)))
        digit_rank = _digit_rank(semantic_point, target.name)
        if digit_rank < 99:
            active_digit_ranks.add(digit_rank)
    support_candidates_by_digit: Dict[int, List[str]] = {}
    for point_name, semantic_point in runtime_cfg.semantic_points.items():
        if point_name in active_names:
            continue
        tags = _semantic_point_tags(semantic_point, point_name)
        if "contact" not in tags or "avoid" in tags or "palm" in tags:
            continue
        if _is_thumb_semantic_point(semantic_point, point_name):
            continue
        if (semantic_point.source_link, int(semantic_point.source_sphere_index)) in active_keys:
            continue
        digit_rank = _digit_rank(semantic_point, point_name)
        if digit_rank >= 99 or (active_digit_ranks and digit_rank not in active_digit_ranks):
            continue
        support_candidates_by_digit.setdefault(digit_rank, []).append(point_name)

    selected_names: List[str] = []

    def _support_priority(name: str) -> tuple[int, int, str]:
        semantic_point = runtime_cfg.semantic_points[name]
        tags = _semantic_point_tags(semantic_point, name)
        normalized_name = name.replace("-", "_").split("_")
        if "tip" in tags or "tip" in normalized_name:
            kind_rank = 0
        elif "side" in tags or "side" in normalized_name:
            kind_rank = 1
        elif "pad" in tags or "pad" in normalized_name:
            kind_rank = 2
        else:
            kind_rank = 3
        main_rank = 0 if "main" in normalized_name else 1
        return (kind_rank, main_rank, name)

    for digit_rank in sorted(support_candidates_by_digit):
        digit_points = sorted(
            support_candidates_by_digit[digit_rank],
            key=_support_priority,
        )
        if digit_points:
            selected_names.append(digit_points[0])
    return selected_names


def _build_sphere_lookup(
    hand_model: HandKinematicsModel,
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_pose: HandPose,
) -> tuple[Dict[tuple[str, int], CollisionSphereState], Dict[str, CollisionSphereState]]:
    sphere_lookup = hand_model.collision_sphere_map(runtime_cfg, hand_pose)
    semantic_lookup: Dict[str, CollisionSphereState] = {}
    for point_name, semantic_point in runtime_cfg.semantic_points.items():
        key = (semantic_point.source_link, semantic_point.source_sphere_index)
        if key in sphere_lookup:
            semantic_lookup[point_name] = sphere_lookup[key]
    return sphere_lookup, semantic_lookup


def _query_sphere_sdf_metrics(
    object_query: ObjectQueryBackend,
    sphere_lookup: Mapping[tuple[str, int], CollisionSphereState],
) -> Dict[tuple[str, int], Dict[str, Any]]:
    if not sphere_lookup:
        return {}

    items = list(sphere_lookup.items())
    centers = np.asarray(
        [np.asarray(sphere_state.center_world, dtype=np.float64) for _, sphere_state in items],
        dtype=np.float64,
    )
    sdf_vals = np.asarray(object_query.signed_distance(centers), dtype=np.float64)
    grads = np.asarray(object_query.gradient(centers), dtype=np.float64)

    metrics: Dict[tuple[str, int], Dict[str, Any]] = {}
    for idx, (key, sphere_state) in enumerate(items):
        sdf_val = float(sdf_vals[idx])
        grad = np.asarray(grads[idx], dtype=np.float64)
        metrics[key] = {
            "sdf": sdf_val,
            "grad": grad,
            "clearance": float(sdf_val - sphere_state.radius),
        }
    return metrics


def _object_bbox_from_query(object_query: ObjectQueryBackend) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    metadata = getattr(getattr(object_query, "sdf", None), "metadata", {}) or {}
    bbox_min_raw = metadata.get("bbox_min")
    bbox_max_raw = metadata.get("bbox_max")
    if bbox_min_raw is None or bbox_max_raw is None:
        return None, None
    try:
        bbox_min = np.asarray(bbox_min_raw, dtype=np.float64).reshape(3)
        bbox_max = np.asarray(bbox_max_raw, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None, None
    if not (np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max))):
        return None, None
    return bbox_min, bbox_max


def _hand_bottom_clearance_metrics(
    object_query: ObjectQueryBackend,
    sphere_lookup: Mapping[tuple[str, int], CollisionSphereState],
    *,
    margin: float = 0.0,
    tolerance: float = 0.0,
    bbox_min_override: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    bbox_min = None
    if bbox_min_override is not None:
        try:
            bbox_min = np.asarray(bbox_min_override, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            bbox_min = None
        if bbox_min is not None and not np.all(np.isfinite(bbox_min)):
            bbox_min = None
    if bbox_min is None:
        bbox_min, _ = _object_bbox_from_query(object_query)
    if bbox_min is None or not sphere_lookup:
        return {
            "enabled": False,
            "passes_bottom_clearance": True,
            "max_bottom_violation": 0.0,
            "sum_bottom_violation_sq": 0.0,
        }

    object_bottom_z = float(bbox_min[2])
    required_bottom_z = object_bottom_z + float(margin)
    min_hand_bottom_z = float("inf")
    max_violation = 0.0
    sum_violation_sq = 0.0
    worst_link = ""
    worst_sphere_index = -1
    num_violating = 0

    for (link_name, sphere_index), sphere_state in sphere_lookup.items():
        center_world = np.asarray(sphere_state.center_world, dtype=np.float64)
        sphere_bottom_z = float(center_world[2] - float(sphere_state.radius))
        if sphere_bottom_z < min_hand_bottom_z:
            min_hand_bottom_z = sphere_bottom_z
        violation = max(0.0, required_bottom_z - sphere_bottom_z)
        if violation > 0.0:
            num_violating += 1
            sum_violation_sq += float(violation * violation)
        if violation > max_violation:
            max_violation = float(violation)
            worst_link = str(link_name)
            worst_sphere_index = int(sphere_index)

    return {
        "enabled": True,
        "passes_bottom_clearance": bool(max_violation <= float(tolerance)),
        "max_bottom_violation": float(max_violation),
        "sum_bottom_violation_sq": float(sum_violation_sq),
        "min_hand_bottom_z": float(min_hand_bottom_z),
        "object_bottom_z": float(object_bottom_z),
        "required_bottom_z": float(required_bottom_z),
        "configured_bottom_margin": float(margin),
        "configured_bottom_tolerance": float(tolerance),
        "worst_bottom_link": worst_link,
        "worst_bottom_sphere_index": int(worst_sphere_index),
        "num_bottom_violating_spheres": int(num_violating),
    }


def _cat3_bottom_clearance_cfg(runtime_cfg: ResolvedHandRuntimeConfig) -> Mapping[str, Any]:
    return (
        _merged_optimizer_cfg(runtime_cfg)
        .get("contact_model", {})
        .get("cat3_bottom_clearance", {})
    )


def _annotate_completion_with_bottom_metrics(
    completion: Dict[str, Any],
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
) -> None:
    if not metrics.get("enabled", False):
        return

    if prefix:
        for key, value in metrics.items():
            if key == "enabled":
                continue
            completion[f"{prefix}_{key}"] = value

    current_violation = float(completion.get("max_bottom_violation", 0.0))
    new_violation = float(metrics.get("max_bottom_violation", 0.0))
    if new_violation >= current_violation:
        completion["max_bottom_violation"] = float(new_violation)
        completion["min_hand_bottom_z"] = float(metrics.get("min_hand_bottom_z", np.inf))
        completion["object_bottom_z"] = float(metrics.get("object_bottom_z", np.nan))
        completion["required_bottom_z"] = float(metrics.get("required_bottom_z", np.nan))
        completion["configured_bottom_margin"] = float(metrics.get("configured_bottom_margin", 0.0))
        completion["configured_bottom_tolerance"] = float(metrics.get("configured_bottom_tolerance", 0.0))
        completion["worst_bottom_link"] = str(metrics.get("worst_bottom_link", ""))
        completion["worst_bottom_sphere_index"] = int(metrics.get("worst_bottom_sphere_index", -1))
        completion["num_bottom_violating_spheres"] = int(metrics.get("num_bottom_violating_spheres", 0))
    completion["passes_bottom_clearance"] = bool(
        completion.get("passes_bottom_clearance", True)
        and bool(metrics.get("passes_bottom_clearance", True))
    )


def _self_collision_pairs(runtime_cfg: ResolvedHandRuntimeConfig) -> List[tuple[tuple[str, int], tuple[str, int], float]]:
    buffers = runtime_cfg.collision.self_collision_buffer
    ignore_pairs: set[tuple[str, str]] = set()
    for link_a, ignored_links in runtime_cfg.collision.self_collision_ignore.items():
        for link_b in ignored_links:
            ignore_pairs.add((link_a, link_b))
            ignore_pairs.add((link_b, link_a))

    pairs: List[tuple[tuple[str, int], tuple[str, int], float]] = []
    links = list(runtime_cfg.collision.spheres_by_link.keys())
    for i, link_a in enumerate(links):
        spheres_a = runtime_cfg.collision.spheres_by_link.get(link_a, [])
        for link_b in links[i:]:
            if link_a == link_b or (link_a, link_b) in ignore_pairs:
                continue
            spheres_b = runtime_cfg.collision.spheres_by_link.get(link_b, [])
            buffer_margin = float(buffers.get(link_a, 0.0)) + float(buffers.get(link_b, 0.0))
            for idx_a in range(len(spheres_a)):
                for idx_b in range(len(spheres_b)):
                    pairs.append(((link_a, idx_a), (link_b, idx_b), buffer_margin))
    return pairs


def _accumulate_stage_cost(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    hand_pose: HandPose,
    init_pose: HandPose,
    stage_name: StageName,
    weights: Mapping[str, float],
    self_collision_pairs: Sequence[tuple[tuple[str, int], tuple[str, int], float]],
    reference_semantic_lookup: Optional[Mapping[str, CollisionSphereState]] = None,
) -> CostBreakdown:
    breakdown = CostBreakdown()
    cfg = _merged_optimizer_cfg(runtime_cfg)
    completion_cfg = cfg.get("stage_completion", {}).get(stage_name, {})
    contact_model_cfg = cfg.get("contact_model", {})
    sphere_lookup, semantic_lookup = _build_sphere_lookup(hand_model, runtime_cfg, hand_pose)
    sphere_sdf_metrics = _query_sphere_sdf_metrics(object_query, sphere_lookup)
    cat4_tip_probe_metrics = _query_cat4_tip_probe_metrics(
        runtime_cfg=runtime_cfg,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=hand_pose,
        contact_result=contact_result,
        semantic_lookup=semantic_lookup,
        stage_name=stage_name,
    )

    def add_cost(name: str, value: float) -> None:
        breakdown.terms[name] = breakdown.terms.get(name, 0.0) + float(value)

    active_keys = {
        (target.source_link, target.source_sphere_index)
        for target in contact_result.active_targets
    }
    if contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"}:
        active_keys = {
            (target.source_link, target.source_sphere_index)
            for target in contact_result.active_targets
            if not _is_palm_target(target)
        }
    support_keys: set[tuple[str, int]] = set()
    tip_support_cfg = contact_model_cfg.get("cat1_tip_support", {})
    use_tip_support = (
        stage_name in {"grasp", "squeeze"}
        and bool(tip_support_cfg.get("enabled", True))
        and contact_result.category == "cat1"
        and str(contact_result.metadata.get("cat1_contact_mode", "pad")).lower() == "tip"
    )
    if use_tip_support:
        tip_smallness = float(contact_result.metadata.get("cat1_tip_smallness", 0.0))
        stage_support_weight = float(tip_support_cfg.get(f"{stage_name}_weight", 0.0))
        stage_support_clearance = float(tip_support_cfg.get(f"{stage_name}_clearance", 0.0))
        support_weight_scale = 0.6 + 0.8 * float(np.clip(tip_smallness, 0.0, 1.0))
        for point_name in _cat1_tip_support_point_names(runtime_cfg, contact_result):
            semantic_point = runtime_cfg.semantic_points.get(point_name)
            sphere_state = semantic_lookup.get(point_name)
            if semantic_point is None or sphere_state is None:
                continue
            key = (semantic_point.source_link, semantic_point.source_sphere_index)
            support_keys.add(key)
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            support_weight = stage_support_weight * support_weight_scale
            if "thumb" in point_name:
                support_weight *= 0.8
            deficit = max(0.0, clearance - stage_support_clearance)
            add_cost(
                "tip_support_clearance",
                float(support_weight) * deficit**2,
            )

    cat2_palm_support_cfg = contact_model_cfg.get("cat2_palm_support", {})
    use_cat2_palm_support = (
        stage_name in {"grasp", "squeeze"}
        and bool(cat2_palm_support_cfg.get("enabled", True))
        and contact_result.category == "cat2"
    )
    if use_cat2_palm_support:
        stage_support_weight = float(cat2_palm_support_cfg.get(f"{stage_name}_weight", 0.0))
        stage_position_weight = float(cat2_palm_support_cfg.get(f"{stage_name}_position_weight", 0.0))
        for target in _cat2_palm_support_targets(contact_result):
            semantic_point = runtime_cfg.semantic_points.get(target.name)
            sphere_state = semantic_lookup.get(target.name)
            if semantic_point is None or sphere_state is None:
                continue
            key = (semantic_point.source_link, semantic_point.source_sphere_index)
            support_keys.add(key)
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            target_clearance = _clearance_for_stage(target, stage_name)
            deficit = max(0.0, clearance - target_clearance)
            add_cost(
                "cat2_palm_support_clearance",
                float(stage_support_weight) * float(target.weight) * deficit**2,
            )
            desired_center = _target_desired_center(
                target,
                sphere_state,
                stage_name,
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
            )
            pos_err = np.asarray(sphere_state.center_world, dtype=np.float64) - desired_center
            add_cost(
                "cat2_palm_support_position",
                float(stage_position_weight) * float(target.weight) * float(np.dot(pos_err, pos_err)),
            )

    cat2_finger_support_cfg = contact_model_cfg.get("cat2_finger_support", {})
    use_cat2_finger_support = (
        stage_name in {"grasp", "squeeze"}
        and bool(cat2_finger_support_cfg.get("enabled", True))
        and contact_result.category == "cat2"
    )
    if use_cat2_finger_support:
        stage_support_weight = float(cat2_finger_support_cfg.get(f"{stage_name}_weight", 0.0))
        stage_support_clearance = float(cat2_finger_support_cfg.get(f"{stage_name}_clearance", 0.0))
        for point_name in _cat2_finger_support_point_names(runtime_cfg, contact_result):
            semantic_point = runtime_cfg.semantic_points.get(point_name)
            sphere_state = semantic_lookup.get(point_name)
            if semantic_point is None or sphere_state is None:
                continue
            key = (semantic_point.source_link, semantic_point.source_sphere_index)
            support_keys.add(key)
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            deficit = max(0.0, clearance - stage_support_clearance)
            add_cost(
                "cat2_finger_support_clearance",
                float(stage_support_weight) * deficit**2,
            )

    cat2_active_stability_cfg = contact_model_cfg.get("cat2_active_stability", {})
    use_cat2_active_stability = (
        stage_name in {"grasp", "squeeze"}
        and bool(cat2_active_stability_cfg.get("enabled", True))
        and contact_result.category == "cat2"
        and reference_semantic_lookup is not None
    )
    if use_cat2_active_stability:
        stage_stability_weight = float(cat2_active_stability_cfg.get(f"{stage_name}_weight", 0.0))
        thumb_multiplier = float(cat2_active_stability_cfg.get("thumb_multiplier", 0.35))
        for target in contact_result.active_targets:
            sphere_state = semantic_lookup.get(target.name)
            reference_state = reference_semantic_lookup.get(target.name)
            if sphere_state is None or reference_state is None:
                continue
            target_weight = max(float(target.weight), 1e-6)
            if _is_thumb_target(target):
                target_weight *= thumb_multiplier
            delta = (
                np.asarray(sphere_state.center_world, dtype=np.float64)
                - np.asarray(reference_state.center_world, dtype=np.float64)
            )
            add_cost(
                "cat2_active_stability",
                float(stage_stability_weight) * target_weight * float(np.dot(delta, delta)),
            )

    cat2_global_penetration_cfg = contact_model_cfg.get("cat2_global_penetration", {})
    use_cat2_global_penetration = (
        stage_name in {"grasp", "squeeze"}
        and bool(cat2_global_penetration_cfg.get("enabled", True))
        and contact_result.category == "cat2"
    )
    if use_cat2_global_penetration:
        stage_penetration_weight = float(cat2_global_penetration_cfg.get(f"{stage_name}_weight", 0.0))
        for key, sphere_state in sphere_lookup.items():
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            penetration = max(0.0, -clearance)
            if penetration <= 0.0:
                continue
            add_cost(
                "cat2_global_penetration",
                float(stage_penetration_weight) * penetration**2,
            )

    cat3_global_penetration_cfg = contact_model_cfg.get("cat3_global_penetration", {})
    use_cat3_global_penetration = (
        stage_name in {"pregrasp", "grasp", "squeeze"}
        and bool(cat3_global_penetration_cfg.get("enabled", True))
        and contact_result.category == "cat3"
    )
    if use_cat3_global_penetration:
        stage_penetration_weight = float(cat3_global_penetration_cfg.get(f"{stage_name}_weight", 0.0))
        for key, sphere_state in sphere_lookup.items():
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            penetration = max(0.0, -clearance)
            if penetration <= 0.0:
                continue
            add_cost(
                "cat3_global_penetration",
                float(stage_penetration_weight) * penetration**2,
            )

    cat3_bottom_cfg = contact_model_cfg.get("cat3_bottom_clearance", {})
    use_cat3_bottom_clearance = (
        stage_name in {"pregrasp", "grasp", "squeeze"}
        and bool(cat3_bottom_cfg.get("enabled", True))
        and contact_result.category == "cat3"
    )
    if use_cat3_bottom_clearance:
        bottom_metrics = _hand_bottom_clearance_metrics(
            object_query,
            sphere_lookup,
            margin=float(cat3_bottom_cfg.get("margin", 0.0)),
            tolerance=float(cat3_bottom_cfg.get("tolerance", 0.0)),
            bbox_min_override=contact_result.metadata.get("object_bbox_min"),
        )
        if bottom_metrics.get("enabled", False):
            stage_bottom_weight = float(cat3_bottom_cfg.get(f"{stage_name}_weight", 0.0))
            add_cost(
                "cat3_bottom_clearance",
                stage_bottom_weight * float(bottom_metrics.get("sum_bottom_violation_sq", 0.0)),
            )

    cat4_global_penetration_cfg = contact_model_cfg.get("cat4_global_penetration", {})
    use_cat4_global_penetration = (
        stage_name in {"grasp", "squeeze"}
        and bool(cat4_global_penetration_cfg.get("enabled", True))
        and contact_result.category == "cat4"
    )
    if use_cat4_global_penetration:
        stage_penetration_weight = float(cat4_global_penetration_cfg.get(f"{stage_name}_weight", 0.0))
        for key, sphere_state in sphere_lookup.items():
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            penetration = max(0.0, -clearance)
            if penetration <= 0.0:
                continue
            add_cost(
                "cat4_global_penetration",
                float(stage_penetration_weight) * penetration**2,
            )

    for target in contact_result.active_targets:
        sphere_state = semantic_lookup.get(target.name)
        if sphere_state is None:
            continue
        if contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"} and _is_palm_target(target):
            continue
        target_weight = max(float(target.weight), 1e-6)
        if contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"}:
            if _is_declared_tip_target(runtime_cfg, target):
                tip_priority_cfg = contact_model_cfg.get("cat4_tip_priority", {})
                target_weight *= float(
                    tip_priority_cfg.get(
                        f"{stage_name}_multiplier",
                        1.8 if stage_name == "grasp" else 2.6,
                    )
                )
            elif not _is_palm_target(target):
                tip_priority_cfg = contact_model_cfg.get("cat4_tip_priority", {})
                target_weight *= float(
                    tip_priority_cfg.get(
                        f"{stage_name}_non_tip_multiplier",
                        0.70 if stage_name == "grasp" else 0.55,
                    )
                )
        if _is_thumb_target(target):
            target_weight *= float(weights.get("thumb_active_multiplier", 1.0))
        key = (target.source_link, target.source_sphere_index)
        sdf_val = float(sphere_sdf_metrics[key]["sdf"])
        grad = _safe_normalize(np.asarray(sphere_sdf_metrics[key]["grad"], dtype=np.float64))
        clearance = float(sphere_sdf_metrics[key]["clearance"])
        tip_probe_metrics = None
        if contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"} and _is_declared_tip_target(runtime_cfg, target):
            tip_probe_metrics = cat4_tip_probe_metrics.get(target.name)
        if tip_probe_metrics is not None:
            probe_sdf_val = float(tip_probe_metrics["sdf"])
            probe_grad = _safe_normalize(np.asarray(tip_probe_metrics["grad"], dtype=np.float64))
            probe_clearance = float(tip_probe_metrics["clearance"])
            sdf_val = probe_sdf_val
            grad = probe_grad
            clearance = probe_clearance
        target_clearance = _clearance_for_stage(target, stage_name)

        if contact_result.category == "cat1":
            pos_err_sq = _cat1_stage_position_error(
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
                target=target,
                sphere_state=sphere_state,
                stage_name=stage_name,
                signed_distance=sdf_val,
                gradient_world=grad,
            )
        elif contact_result.category == "cat2":
            pos_err_sq = _cat2_stage_position_error(
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
                target=target,
                sphere_state=sphere_state,
                stage_name=stage_name,
                signed_distance=sdf_val,
                gradient_world=grad,
            )
        elif contact_result.category == "cat3":
            pos_err_sq = _cat3_stage_position_error(
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
                target=target,
                sphere_state=sphere_state,
                stage_name=stage_name,
                signed_distance=sdf_val,
                gradient_world=grad,
            )
        elif contact_result.category == "cat4":
            if tip_probe_metrics is not None:
                pos_err_sq = _cat4_tip_stage_position_error(
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                    target=target,
                    stage_name=stage_name,
                    probe_world=np.asarray(tip_probe_metrics["probe_world"], dtype=np.float64),
                    signed_distance=sdf_val,
                    gradient_world=grad,
                )
            else:
                pos_err_sq = _cat4_stage_position_error(
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                    target=target,
                    sphere_state=sphere_state,
                    stage_name=stage_name,
                    signed_distance=sdf_val,
                    gradient_world=grad,
                )
        else:
            desired_center = _target_desired_center(
                target,
                sphere_state,
                stage_name,
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
            )
            pos_err = np.asarray(sphere_state.center_world, dtype=np.float64) - desired_center
            pos_err_sq = float(np.dot(pos_err, pos_err))
        add_cost(
            "active_clearance",
            float(weights["active_clearance"]) * target_weight * (clearance - target_clearance) ** 2,
        )
        use_active_target_position = True
        if contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"}:
            # For cat4, palm is only soft support geometry. The actual closing
            # behavior should be driven by thumb/finger contacts around the
            # object, similar to the BODex convex hold setup.
            use_active_target_position = not _is_palm_target(target)
        if use_active_target_position:
            add_cost(
                "active_target_position",
                float(weights["active_target_position"]) * target_weight * pos_err_sq,
            )
        if contact_result.category == "cat2" and _is_thumb_target(target):
            add_cost(
                "cat2_thumb_side_guard",
                _cat2_thumb_side_guard_cost(
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                    target=target,
                    sphere_state=sphere_state,
                    stage_name=stage_name,
                    signed_distance=sdf_val,
                    gradient_world=grad,
                ),
            )
        if contact_result.category == "cat4" and _is_thumb_target(target):
            thumb_outside_cfg = contact_model_cfg.get("cat4_thumb_outside", {})
            if bool(thumb_outside_cfg.get("enabled", True)) and stage_name in {"grasp", "squeeze"}:
                surface_point_world = _surface_contact_point_world(
                    sphere_state.center_world,
                    sdf_val,
                    grad,
                )
                ref_world = (
                    np.asarray(surface_point_world, dtype=np.float64)
                    if surface_point_world is not None
                    else np.asarray(sphere_state.center_world, dtype=np.float64)
                )
                frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
                anchor_point = np.asarray(contact_result.anchor.point, dtype=np.float64)
                local_surface = frame_R.T @ (ref_world - anchor_point)
                local_target = _cat4_stage_target_local_point(
                    runtime_cfg,
                    contact_result,
                    target,
                    stage_name,
                )
                y_gap = float(
                    local_target[1]
                    - local_surface[1]
                    - float(thumb_outside_cfg.get("y_slack", 0.001))
                )
                if y_gap > 0.0:
                    add_cost(
                        "cat4_thumb_outside",
                        float(thumb_outside_cfg.get(f"{stage_name}_weight", 0.0)) * target_weight * y_gap**2,
                    )
        if np.linalg.norm(grad) > 1e-8:
            normal_align = 1.0 - float(
                np.dot(
                    grad,
                    _safe_normalize(np.asarray(target.target_normal, dtype=np.float64)),
                )
            )
            add_cost(
                "active_normal_alignment",
                float(weights["active_normal_alignment"]) * target_weight * normal_align**2,
            )

    for target in contact_result.avoid_targets:
        if use_cat2_palm_support and "palm" in str(target.name).lower():
            continue
        sphere_state = semantic_lookup.get(target.name)
        if sphere_state is None:
            continue
        key = (target.source_link, target.source_sphere_index)
        clearance = float(sphere_sdf_metrics[key]["clearance"])
        target_clearance = _clearance_for_stage(target, stage_name)
        deficit = max(0.0, target_clearance - clearance)
        add_cost(
            "avoid_clearance",
            float(weights["avoid_clearance"]) * float(target.weight) * deficit**2,
        )

    cat3_frame_alignment_cfg = contact_model_cfg.get("cat3_frame_alignment", {})
    use_cat3_frame_alignment = (
        stage_name in {"grasp", "squeeze"}
        and bool(cat3_frame_alignment_cfg.get("enabled", True))
        and contact_result.category == "cat3"
    )
    if use_cat3_frame_alignment:
        stage_alignment_weight = float(cat3_frame_alignment_cfg.get(f"{stage_name}_weight", 0.0))
        palm_multiplier = float(cat3_frame_alignment_cfg.get("palm_multiplier", 1.0))
        frame_cfg = runtime_cfg.hand.frame_convention
        rotation = hand_pose.rotation_matrix()
        finger_world = _safe_normalize(
            rotation @ np.asarray(frame_cfg.finger_forward_local, dtype=np.float64)
        )
        palm_world = _safe_normalize(
            rotation @ np.asarray(frame_cfg.palm_normal_local, dtype=np.float64)
        )
        desired_finger_world = _safe_normalize(np.asarray(contact_result.frame_R[:, 0], dtype=np.float64))
        desired_palm_world = _safe_normalize(np.asarray(contact_result.frame_R[:, 2], dtype=np.float64))
        if np.linalg.norm(finger_world) > 1e-8 and np.linalg.norm(desired_finger_world) > 1e-8:
            finger_err = 1.0 - float(np.dot(finger_world, desired_finger_world))
            add_cost("cat3_finger_frame_alignment", stage_alignment_weight * finger_err**2)
        if np.linalg.norm(palm_world) > 1e-8 and np.linalg.norm(desired_palm_world) > 1e-8:
            palm_err = 1.0 - float(np.dot(palm_world, desired_palm_world))
            add_cost("cat3_palm_frame_alignment", stage_alignment_weight * palm_multiplier * palm_err**2)

    non_active_margin = float(_merged_optimizer_cfg(runtime_cfg)["collision"]["non_active_margin"])
    palm_margin = float(_merged_optimizer_cfg(runtime_cfg)["collision"]["palm_margin"])
    palm_link = runtime_cfg.hand.root.palm_link
    for key, sphere_state in sphere_lookup.items():
        if key in active_keys or key in support_keys:
            continue
        clearance = float(sphere_sdf_metrics[key]["clearance"])
        margin = palm_margin if sphere_state.link_name == palm_link else non_active_margin
        deficit = max(0.0, margin - clearance)
        if deficit <= 0.0:
            continue
        if sphere_state.link_name == palm_link:
            add_cost("palm_collision", float(weights["palm_collision"]) * deficit**2)
        else:
            add_cost("non_active_collision", float(weights["non_active_collision"]) * deficit**2)

    for opp in contact_result.opposition_constraints:
        state_a = semantic_lookup.get(opp.point_a)
        state_b = semantic_lookup.get(opp.point_b)
        if state_a is None or state_b is None:
            continue
        pair_dir = _safe_normalize(state_b.center_world - state_a.center_world)
        desired_axis = _safe_normalize(np.asarray(opp.desired_axis_world, dtype=np.float64))
        if np.linalg.norm(pair_dir) < 1e-8 or np.linalg.norm(desired_axis) < 1e-8:
            continue
        align_err = 1.0 - float(np.dot(pair_dir, desired_axis))
        opposition_weight = float(weights["opposition"])
        if contact_result.category == "cat1" and str(contact_result.metadata.get("cat1_contact_mode", "pad")).lower() == "tip":
            tip_smallness = float(np.clip(contact_result.metadata.get("cat1_tip_smallness", 0.0), 0.0, 1.0))
            if stage_name == "grasp":
                opposition_weight *= max(0.25, 0.60 - 0.20 * tip_smallness)
            elif stage_name == "squeeze":
                opposition_weight *= max(0.15, 0.30 - 0.10 * tip_smallness)
        add_cost("opposition", opposition_weight * float(opp.weight) * align_err**2)

    self_collision_margin = float(cfg["collision"]["self_collision_margin"])
    for key_a, key_b, buffer_margin in self_collision_pairs:
        sphere_a = sphere_lookup.get(key_a)
        sphere_b = sphere_lookup.get(key_b)
        if sphere_a is None or sphere_b is None:
            continue
        d = float(np.linalg.norm(sphere_a.center_world - sphere_b.center_world))
        min_sep = sphere_a.radius + sphere_b.radius + buffer_margin + self_collision_margin
        deficit = max(0.0, min_sep - d)
        if deficit > 0.0:
            add_cost("self_collision", float(weights["self_collision"]) * deficit**2)

    pos_delta = np.asarray(hand_pose.wrist_position, dtype=np.float64) - np.asarray(init_pose.wrist_position, dtype=np.float64)
    add_cost(
        "wrist_translation_prior",
        float(weights["wrist_translation_prior"]) * float(np.dot(pos_delta, pos_delta)),
    )

    rot_delta = Rotation.from_matrix(init_pose.rotation_matrix().T @ hand_pose.rotation_matrix()).as_rotvec()
    add_cost(
        "wrist_rotation_prior",
        float(weights["wrist_rotation_prior"]) * float(np.dot(rot_delta, rot_delta)),
    )

    joint_delta_sq = 0.0
    for joint_name in runtime_cfg.hand.joints.controllable:
        dq = float(hand_pose.joint_positions.get(joint_name, 0.0)) - float(init_pose.joint_positions.get(joint_name, 0.0))
        joint_delta_sq += dq * dq
    add_cost("joint_prior", float(weights["joint_prior"]) * joint_delta_sq)

    return breakdown


def _evaluate_stage_completion(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    hand_pose: HandPose,
    stage_name: StageName,
) -> Dict[str, Any]:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    completion_cfg = cfg.get("stage_completion", {}).get(stage_name, {})
    if stage_name == "pregrasp" or not completion_cfg:
        completion = {
            "passes_completion": True,
            "passes_global_penetration": True,
            "passes_bottom_clearance": True,
            "max_active_excess": 0.0,
            "max_thumb_excess": 0.0,
            "max_palm_excess": 0.0,
            "max_finger_excess": 0.0,
            "max_global_penetration": 0.0,
            "configured_max_global_penetration": float("inf"),
            "worst_penetrating_link": "",
            "worst_penetrating_sphere_index": -1,
            "num_active_contacts_evaluated": 0,
            "num_active_contacts_satisfied": 0,
            "num_palm_contacts_evaluated": 0,
            "num_palm_contacts_satisfied": 0,
            "num_finger_contacts_evaluated": 0,
            "num_finger_contacts_satisfied": 0,
        }
        if contact_result.category == "cat3" and stage_name == "pregrasp":
            sphere_lookup, _ = _build_sphere_lookup(hand_model, runtime_cfg, hand_pose)
            sphere_sdf_metrics = _query_sphere_sdf_metrics(object_query, sphere_lookup)
            global_cfg = cfg.get("contact_model", {}).get("cat3_global_penetration", {})
            max_allowed_global_penetration = float(
                completion_cfg.get(
                    "max_global_penetration",
                    global_cfg.get("pregrasp_max_penetration", 0.00025),
                )
            )
            max_global_penetration = 0.0
            worst_penetrating_link = ""
            worst_penetrating_sphere_index = -1
            for (link_name, sphere_index), sphere_state in sphere_lookup.items():
                clearance = float(sphere_sdf_metrics[(link_name, sphere_index)]["clearance"])
                penetration = max(0.0, -clearance)
                if penetration > max_global_penetration:
                    max_global_penetration = float(penetration)
                    worst_penetrating_link = str(link_name)
                    worst_penetrating_sphere_index = int(sphere_index)
            passes_global_penetration = bool(max_global_penetration <= max_allowed_global_penetration)

            bottom_metrics: Dict[str, Any] = {}
            bottom_cfg = _cat3_bottom_clearance_cfg(runtime_cfg)
            if bool(bottom_cfg.get("enabled", True)):
                bottom_metrics = _hand_bottom_clearance_metrics(
                    object_query,
                    sphere_lookup,
                    margin=float(bottom_cfg.get("margin", 0.0)),
                    tolerance=float(bottom_cfg.get("tolerance", 0.0)),
                    bbox_min_override=contact_result.metadata.get("object_bbox_min"),
                )
            completion.update(
                {
                    "passes_global_penetration": bool(passes_global_penetration),
                    "max_global_penetration": float(max_global_penetration),
                    "configured_max_global_penetration": float(max_allowed_global_penetration),
                    "worst_penetrating_link": worst_penetrating_link,
                    "worst_penetrating_sphere_index": int(worst_penetrating_sphere_index),
                }
            )
            _annotate_completion_with_bottom_metrics(completion, bottom_metrics)
            completion["passes_completion"] = bool(
                passes_global_penetration
                and completion.get("passes_bottom_clearance", True)
            )
        return completion

    sphere_lookup, semantic_lookup = _build_sphere_lookup(hand_model, runtime_cfg, hand_pose)
    sphere_sdf_metrics = _query_sphere_sdf_metrics(object_query, sphere_lookup)
    cat4_tip_probe_metrics = _query_cat4_tip_probe_metrics(
        runtime_cfg=runtime_cfg,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=hand_pose,
        contact_result=contact_result,
        semantic_lookup=semantic_lookup,
        stage_name=stage_name,
    )
    active_excesses: Dict[str, float] = {}
    thumb_excesses: Dict[str, float] = {}
    palm_excesses: Dict[str, float] = {}
    finger_excesses: Dict[str, float] = {}
    support_excesses: Dict[str, float] = {}
    completion_active_targets = list(contact_result.active_targets)
    if contact_result.category == "cat4":
        if stage_name == "squeeze":
            tip_targets = [
                target
                for target in completion_active_targets
                if (not _is_palm_target(target)) and _is_declared_tip_target(runtime_cfg, target)
            ]
            completion_active_targets = tip_targets or [
                target
                for target in completion_active_targets
                if not _is_palm_target(target)
            ]
        else:
            completion_active_targets = [
                target
                for target in completion_active_targets
                if not _is_palm_target(target)
            ]

    for target in completion_active_targets:
        sphere_state = semantic_lookup.get(target.name)
        if sphere_state is None:
            continue
        key = (target.source_link, target.source_sphere_index)
        clearance = float(sphere_sdf_metrics[key]["clearance"])
        if (
            contact_result.category == "cat4"
            and stage_name in {"grasp", "squeeze"}
            and _is_declared_tip_target(runtime_cfg, target)
            and target.name in cat4_tip_probe_metrics
        ):
            clearance = float(cat4_tip_probe_metrics[target.name]["clearance"])
        target_clearance = _clearance_for_stage(target, stage_name)
        completion_tol = _contact_completion_tolerance(target, completion_cfg)
        excess = max(0.0, clearance - (target_clearance + completion_tol))
        active_excesses[target.name] = excess
        if _is_thumb_target(target):
            thumb_excesses[target.name] = excess
        elif _is_palm_target(target):
            palm_excesses[target.name] = excess
        else:
            finger_excesses[target.name] = excess

    tip_support_cfg = _merged_optimizer_cfg(runtime_cfg).get("contact_model", {}).get("cat1_tip_support", {})
    support_required_contacts = 0
    if (
        stage_name in {"grasp", "squeeze"}
        and bool(tip_support_cfg.get("enabled", True))
        and contact_result.category == "cat1"
        and str(contact_result.metadata.get("cat1_contact_mode", "pad")).lower() == "tip"
    ):
        support_clearance = float(tip_support_cfg.get(f"{stage_name}_clearance", 0.0))
        support_tolerance = float(tip_support_cfg.get(f"{stage_name}_tolerance", 0.0))
        support_required_contacts = int(max(0, tip_support_cfg.get(f"{stage_name}_required_contacts", 0)))
        for point_name in _cat1_tip_support_point_names(runtime_cfg, contact_result):
            sphere_state = semantic_lookup.get(point_name)
            if sphere_state is None:
                continue
            semantic_point = runtime_cfg.semantic_points.get(point_name)
            if semantic_point is None:
                continue
            key = (semantic_point.source_link, semantic_point.source_sphere_index)
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            excess = max(0.0, clearance - (support_clearance + support_tolerance))
            support_excesses[point_name] = excess

    cat2_finger_support_cfg = _merged_optimizer_cfg(runtime_cfg).get("contact_model", {}).get("cat2_finger_support", {})
    if (
        stage_name in {"grasp", "squeeze"}
        and bool(cat2_finger_support_cfg.get("enabled", True))
        and contact_result.category == "cat2"
    ):
        support_clearance = float(cat2_finger_support_cfg.get(f"{stage_name}_clearance", 0.0))
        support_tolerance = float(cat2_finger_support_cfg.get(f"{stage_name}_tolerance", 0.0))
        finger_support_names = _cat2_finger_support_point_names(runtime_cfg, contact_result)
        required_raw = int(cat2_finger_support_cfg.get(f"{stage_name}_required_contacts", 0))
        if required_raw < 0:
            support_required_contacts = len(finger_support_names)
        else:
            support_required_contacts = int(max(0, required_raw))
        for point_name in finger_support_names:
            sphere_state = semantic_lookup.get(point_name)
            if sphere_state is None:
                continue
            semantic_point = runtime_cfg.semantic_points.get(point_name)
            if semantic_point is None:
                continue
            key = (semantic_point.source_link, semantic_point.source_sphere_index)
            clearance = float(sphere_sdf_metrics[key]["clearance"])
            excess = max(0.0, clearance - (support_clearance + support_tolerance))
            support_excesses[point_name] = excess

    num_active = len(completion_active_targets)
    num_eval = len(active_excesses)
    num_satisfied = sum(1 for excess in active_excesses.values() if excess <= 1e-8)
    num_support_eval = len(support_excesses)
    num_support_satisfied = sum(1 for excess in support_excesses.values() if excess <= 1e-8)
    max_allowed_global_penetration = float(completion_cfg.get("max_global_penetration", np.inf))
    max_global_penetration = 0.0
    worst_penetrating_link = ""
    worst_penetrating_sphere_index = -1
    for (link_name, sphere_index), sphere_state in sphere_lookup.items():
        clearance = float(sphere_sdf_metrics[(link_name, sphere_index)]["clearance"])
        penetration = max(0.0, -clearance)
        if penetration > max_global_penetration:
            max_global_penetration = float(penetration)
            worst_penetrating_link = str(link_name)
            worst_penetrating_sphere_index = int(sphere_index)
    passes_global_penetration = bool(max_global_penetration <= max_allowed_global_penetration)
    passes_bottom_clearance = True
    bottom_metrics: Dict[str, Any] = {}
    if contact_result.category == "cat3" and stage_name in {"grasp", "squeeze"}:
        bottom_cfg = _cat3_bottom_clearance_cfg(runtime_cfg)
        if bool(bottom_cfg.get("enabled", True)):
            bottom_metrics = _hand_bottom_clearance_metrics(
                object_query,
                sphere_lookup,
                margin=float(bottom_cfg.get("margin", 0.0)),
                tolerance=float(bottom_cfg.get("tolerance", 0.0)),
                bbox_min_override=contact_result.metadata.get("object_bbox_min"),
            )
            passes_bottom_clearance = bool(bottom_metrics.get("passes_bottom_clearance", True))
    support_passes = (
        support_required_contacts <= 0
        or (
            num_support_eval >= support_required_contacts
            and num_support_satisfied >= support_required_contacts
        )
    )
    passes_completion = (
        num_eval == num_active
        and num_active > 0
        and all(excess <= 1e-8 for excess in active_excesses.values())
        and (len(thumb_excesses) == 0 or all(excess <= 1e-8 for excess in thumb_excesses.values()))
        and support_passes
        and passes_global_penetration
        and passes_bottom_clearance
    )

    completion = {
        "passes_completion": bool(passes_completion),
        "support_passes_completion": bool(support_passes),
        "passes_global_penetration": bool(passes_global_penetration),
        "passes_bottom_clearance": bool(passes_bottom_clearance),
        "support_required_contacts": int(support_required_contacts),
        "max_active_excess": float(max(active_excesses.values()) if active_excesses else 0.0),
        "max_thumb_excess": float(max(thumb_excesses.values()) if thumb_excesses else 0.0),
        "max_palm_excess": float(max(palm_excesses.values()) if palm_excesses else 0.0),
        "max_finger_excess": float(max(finger_excesses.values()) if finger_excesses else 0.0),
        "max_support_excess": float(max(support_excesses.values()) if support_excesses else 0.0),
        "max_global_penetration": float(max_global_penetration),
        "configured_max_global_penetration": float(max_allowed_global_penetration),
        "worst_penetrating_link": worst_penetrating_link,
        "worst_penetrating_sphere_index": int(worst_penetrating_sphere_index),
        "num_active_contacts_evaluated": int(num_eval),
        "num_active_contacts_satisfied": int(num_satisfied),
        "num_palm_contacts_evaluated": int(len(palm_excesses)),
        "num_palm_contacts_satisfied": int(sum(1 for excess in palm_excesses.values() if excess <= 1e-8)),
        "num_finger_contacts_evaluated": int(len(finger_excesses)),
        "num_finger_contacts_satisfied": int(sum(1 for excess in finger_excesses.values() if excess <= 1e-8)),
        "num_support_contacts_evaluated": int(num_support_eval),
        "num_support_contacts_satisfied": int(num_support_satisfied),
    }
    _annotate_completion_with_bottom_metrics(completion, bottom_metrics)
    return completion


def optimize_cat1_stage(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    init_pose: HandPose,
    *,
    stage_name: StageName,
    prior_pose: Optional[HandPose] = None,
    weights_override: Optional[Mapping[str, float]] = None,
    solver_override: Optional[Mapping[str, Any]] = None,
) -> StagePoseResult:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    solver_cfg = dict(cfg["solver"])
    if solver_override:
        solver_cfg.update(dict(solver_override))
    weights = dict(cfg["stage_weights"][stage_name])
    if weights_override:
        weights.update({key: float(value) for key, value in weights_override.items()})
    self_collision_pairs = _self_collision_pairs(runtime_cfg)
    use_stage_init_reference = bool(
        contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"}
    )
    reference_pose = init_pose if use_stage_init_reference else (prior_pose or init_pose)
    reference_semantic_lookup = hand_model.semantic_points_world(
        runtime_cfg,
        reference_pose,
        point_names=[target.name for target in contact_result.active_targets],
    )

    x0, base_rotation, bounds = _pack_pose_variables(runtime_cfg, init_pose)
    bounds = _apply_stage_motion_bounds(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        stage_name=stage_name,
        x0=x0,
        bounds=bounds,
    )

    def objective(x: np.ndarray) -> float:
        hand_pose = _unpack_pose_variables(runtime_cfg, x=x, base_rotation=base_rotation)
        breakdown = _accumulate_stage_cost(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=hand_pose,
            init_pose=reference_pose,
            stage_name=stage_name,
            weights=weights,
            self_collision_pairs=self_collision_pairs,
            reference_semantic_lookup=reference_semantic_lookup,
        )
        return breakdown.total

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": int(solver_cfg["max_iterations"]),
            "ftol": float(solver_cfg["ftol"]),
            "gtol": float(solver_cfg["gtol"]),
            "eps": float(solver_cfg["eps"]),
            "maxls": int(solver_cfg.get("maxls", 50)),
        },
    )

    final_pose = _unpack_pose_variables(runtime_cfg, x=np.asarray(result.x, dtype=np.float64), base_rotation=base_rotation)
    final_breakdown = _accumulate_stage_cost(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=final_pose,
        init_pose=reference_pose,
        stage_name=stage_name,
        weights=weights,
        self_collision_pairs=self_collision_pairs,
        reference_semantic_lookup=reference_semantic_lookup,
    )
    completion_metrics = _evaluate_stage_completion(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=final_pose,
        stage_name=stage_name,
    )
    solver_status = int(getattr(result, "status", 0) or 0)
    solver_success = bool(result.success)
    completion_pass = bool(completion_metrics.get("passes_completion", True))
    if stage_name == "pregrasp":
        stage_success = solver_success and (
            completion_pass if contact_result.category == "cat3" else True
        )
    else:
        stage_success = completion_pass and (solver_success or solver_status == 1)
    message = str(result.message)
    if stage_name != "pregrasp" and not completion_pass:
        message = f"{message} | contact completion failed"

    return StagePoseResult(
        stage_name=stage_name,
        hand_pose=final_pose,
        success=bool(stage_success),
        cost=float(final_breakdown.total),
        message=message,
        num_iterations=int(getattr(result, "nit", 0) or 0),
        cost_breakdown=final_breakdown,
        metadata={
            "optimizer_status": solver_status,
            "initial_cost": float(objective(x0)),
            "final_cost": float(final_breakdown.total),
            "used_separate_prior": bool(prior_pose is not None and not use_stage_init_reference),
            "solver_success": solver_success,
            "completion": completion_metrics,
        },
    )


def evaluate_stage_pose(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    hand_pose: HandPose,
    *,
    stage_name: StageName,
    prior_pose: Optional[HandPose] = None,
) -> StagePoseResult:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    weights = dict(cfg["stage_weights"][stage_name])
    self_collision_pairs = _self_collision_pairs(runtime_cfg)
    use_stage_init_reference = bool(
        contact_result.category == "cat4" and stage_name in {"grasp", "squeeze"}
    )
    reference_pose = hand_pose if use_stage_init_reference else (prior_pose or hand_pose)
    reference_semantic_lookup = hand_model.semantic_points_world(
        runtime_cfg,
        reference_pose,
        point_names=[target.name for target in contact_result.active_targets],
    )
    breakdown = _accumulate_stage_cost(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=hand_pose,
        init_pose=reference_pose,
        stage_name=stage_name,
        weights=weights,
        self_collision_pairs=self_collision_pairs,
        reference_semantic_lookup=reference_semantic_lookup,
    )
    completion_metrics = _evaluate_stage_completion(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=hand_pose,
        stage_name=stage_name,
    )
    stage_success = (
        bool(completion_metrics.get("passes_completion", True))
        if (stage_name == "pregrasp" and contact_result.category == "cat3")
        else (True if stage_name == "pregrasp" else bool(completion_metrics.get("passes_completion", True)))
    )
    return StagePoseResult(
        stage_name=stage_name,
        hand_pose=hand_pose,
        success=stage_success,
        cost=float(breakdown.total),
        message="SKIPPED: used stage init pose",
        num_iterations=0,
        cost_breakdown=breakdown,
        metadata={
            "optimizer_status": 0,
            "initial_cost": float(breakdown.total),
            "final_cost": float(breakdown.total),
            "used_separate_prior": False,
            "solver_success": True,
            "solver_skipped": True,
            "completion": completion_metrics,
        },
    )


def _stage_result_rank_key(
    stage_result: StagePoseResult,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    final_seek = stage_result.metadata.get("final_contact_seek", {})
    branch_fail = 0.0 if bool(final_seek.get("branch_valid", True)) else 1.0
    completion = stage_result.metadata.get("completion", {})
    completion_fail = 0.0 if bool(completion.get("passes_completion", False)) else 1.0
    global_penetration = float(completion.get("max_global_penetration", 0.0))
    max_excess = float(completion.get("max_active_excess", 0.0))
    support_required = int(completion.get("support_required_contacts", 0))
    support_satisfied = int(completion.get("num_support_contacts_satisfied", 0))
    support_deficit = float(max(0, support_required - support_satisfied))
    max_support_excess = float(completion.get("max_support_excess", 0.0))
    active_target_position = float(stage_result.cost_breakdown.terms.get("active_target_position", np.inf))
    tip_support_clearance = float(stage_result.cost_breakdown.terms.get("tip_support_clearance", np.inf))
    translation_from_reference = float(final_seek.get("translation_from_reference", np.inf))
    active_normal_alignment = float(stage_result.cost_breakdown.terms.get("active_normal_alignment", np.inf))
    return (
        branch_fail,
        completion_fail,
        global_penetration,
        max_excess,
        support_deficit,
        max_support_excess,
        active_target_position,
        tip_support_clearance,
        translation_from_reference,
        active_normal_alignment,
        float(stage_result.cost),
    )


def _pose_rotation_delta_deg(reference_pose: HandPose, candidate_pose: HandPose) -> float:
    delta_rot = Rotation.from_matrix(
        reference_pose.rotation_matrix().T @ candidate_pose.rotation_matrix()
    )
    return float(np.rad2deg(np.linalg.norm(delta_rot.as_rotvec())))


def _palm_alignment_dot(runtime_cfg: ResolvedHandRuntimeConfig, pose_a: HandPose, pose_b: HandPose) -> float:
    palm_local = _safe_normalize(
        np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )
    if np.linalg.norm(palm_local) < 1e-8:
        return 1.0
    palm_a = _safe_normalize(pose_a.rotation_matrix() @ palm_local)
    palm_b = _safe_normalize(pose_b.rotation_matrix() @ palm_local)
    if np.linalg.norm(palm_a) < 1e-8 or np.linalg.norm(palm_b) < 1e-8:
        return 1.0
    return float(np.dot(palm_a, palm_b))


def _annotate_final_contact_seek_metrics(
    runtime_cfg: ResolvedHandRuntimeConfig,
    seek_cfg: Mapping[str, Any],
    coarse_pose: HandPose,
    reference_pose: HandPose,
    reference_result: StagePoseResult,
    candidate_result: StagePoseResult,
) -> None:
    candidate_pose = candidate_result.hand_pose
    translation_from_reference = float(
        np.linalg.norm(
            np.asarray(candidate_pose.wrist_position, dtype=np.float64)
            - np.asarray(reference_pose.wrist_position, dtype=np.float64)
        )
    )
    translation_from_coarse = float(
        np.linalg.norm(
            np.asarray(candidate_pose.wrist_position, dtype=np.float64)
            - np.asarray(coarse_pose.wrist_position, dtype=np.float64)
        )
    )
    rotation_from_reference_deg = _pose_rotation_delta_deg(reference_pose, candidate_pose)
    rotation_from_coarse_deg = _pose_rotation_delta_deg(coarse_pose, candidate_pose)
    palm_alignment_dot = _palm_alignment_dot(runtime_cfg, reference_pose, candidate_pose)
    reference_target_position = float(
        reference_result.cost_breakdown.terms.get("active_target_position", 0.0)
    )
    max_target_position = reference_target_position * float(
        seek_cfg.get("max_active_target_position_factor", 4.0)
    ) + float(seek_cfg.get("max_active_target_position_slack", 0.020))
    candidate_target_position = float(
        candidate_result.cost_breakdown.terms.get("active_target_position", 0.0)
    )
    branch_valid = (
        translation_from_reference
        <= float(seek_cfg.get("max_translation_from_reference", 0.045))
        and translation_from_coarse
        <= float(seek_cfg.get("max_translation_from_coarse", 0.060))
        and rotation_from_reference_deg
        <= float(seek_cfg.get("max_rotation_from_reference_deg", 22.0))
        and rotation_from_coarse_deg
        <= float(seek_cfg.get("max_rotation_from_coarse_deg", 30.0))
        and palm_alignment_dot
        >= float(seek_cfg.get("min_reference_palm_alignment_dot", 0.35))
        and candidate_target_position <= max_target_position
    )
    candidate_result.metadata["final_contact_seek"] = {
        "branch_valid": bool(branch_valid),
        "translation_from_reference": translation_from_reference,
        "translation_from_coarse": translation_from_coarse,
        "rotation_from_reference_deg": rotation_from_reference_deg,
        "rotation_from_coarse_deg": rotation_from_coarse_deg,
        "reference_palm_alignment_dot": palm_alignment_dot,
        "reference_active_target_position": reference_target_position,
        "max_allowed_active_target_position": max_target_position,
        "candidate_active_target_position": candidate_target_position,
    }


def _contact_seek_final_stage(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    coarse_pose: HandPose,
    current_result: StagePoseResult,
    final_close_pose: Mapping[str, float],
    joint_names: Sequence[str],
) -> StagePoseResult:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    seek_cfg = cfg.get("final_contact_seek", {})
    if not bool(seek_cfg.get("enabled", True)):
        return current_result
    if bool(current_result.success):
        return current_result
    if contact_result.category == "cat4":
        # For cat4, the seed already places the hand in the right wrap regime.
        # The generic final contact-seek pass tends to over-push the hand inward
        # and is the main source of late-stage penetration regressions.
        return current_result

    max_attempts = max(0, int(seek_cfg.get("max_attempts", 4)))
    if max_attempts == 0:
        return current_result
    stop_after_first_success = bool(seek_cfg.get("stop_after_first_success", False))

    base_weights = cfg["stage_weights"]["squeeze"]
    seek_weights = _scaled_stage_weights(
        base_weights,
        {
            "active_clearance": float(seek_cfg.get("active_clearance_multiplier", 2.8))
            * _metadata_float(contact_result.metadata, "cat1_final_seek_clearance_multiplier_scale", 1.0),
            "active_target_position": float(seek_cfg.get("active_target_position_multiplier", 0.45)),
            "active_normal_alignment": float(seek_cfg.get("active_normal_alignment_multiplier", 0.70)),
            "opposition": float(seek_cfg.get("opposition_multiplier", 1.10)),
            "wrist_translation_prior": float(seek_cfg.get("wrist_translation_prior_multiplier", 0.20)),
            "wrist_rotation_prior": float(seek_cfg.get("wrist_rotation_prior_multiplier", 0.25)),
            "joint_prior": float(seek_cfg.get("joint_prior_multiplier", 0.15)),
            "palm_collision": float(seek_cfg.get("palm_collision_multiplier", 0.85)),
            "non_active_collision": float(seek_cfg.get("non_active_collision_multiplier", 0.85)),
            "self_collision": float(seek_cfg.get("self_collision_multiplier", 0.90)),
        },
    )
    solver_override = {
        "max_iterations": int(seek_cfg.get("max_iterations", cfg["solver"]["max_iterations"])),
    }

    best_result = current_result
    reference_pose = current_result.hand_pose
    reference_result = current_result
    _annotate_final_contact_seek_metrics(
        runtime_cfg=runtime_cfg,
        seek_cfg=seek_cfg,
        coarse_pose=coarse_pose,
        reference_pose=reference_pose,
        reference_result=reference_result,
        candidate_result=best_result,
    )
    best_rank = _stage_result_rank_key(best_result)

    wrist_step = float(seek_cfg.get("wrist_step", 0.006)) * _metadata_float(
        contact_result.metadata, "cat1_final_seek_wrist_step_scale", 1.0
    )
    joint_close_step = float(seek_cfg.get("joint_close_step", 0.08)) * _metadata_float(
        contact_result.metadata, "cat1_final_seek_joint_close_step_scale", 1.0
    )
    approach_direction = _approach_direction_for_pose(
        runtime_cfg,
        contact_result,
        reference_pose,
    )

    for attempt in range(max_attempts):
        seek_joint_alpha = float(np.clip(joint_close_step * (attempt + 1), 0.0, 1.0))
        closer_joint_positions = _blend_joint_positions(
            reference_pose.joint_positions,
            final_close_pose,
            seek_joint_alpha,
            joint_names,
        )
        seek_pose = make_hand_pose(
            wrist_position=np.asarray(reference_pose.wrist_position, dtype=np.float64)
            + approach_direction * wrist_step * float(attempt + 1),
            wrist_quaternion_xyzw=np.asarray(reference_pose.wrist_quaternion_xyzw, dtype=np.float64),
            wrist_rotation=reference_pose.rotation_matrix(),
            joint_positions=closer_joint_positions,
        )
        candidate_result = optimize_cat1_stage(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            init_pose=seek_pose,
            stage_name="squeeze",
            prior_pose=reference_pose,
            weights_override=seek_weights,
            solver_override=solver_override,
        )
        _annotate_final_contact_seek_metrics(
            runtime_cfg=runtime_cfg,
            seek_cfg=seek_cfg,
            coarse_pose=coarse_pose,
            reference_pose=reference_pose,
            reference_result=reference_result,
            candidate_result=candidate_result,
        )
        candidate_rank = _stage_result_rank_key(candidate_result)
        if candidate_rank < best_rank:
            best_result = candidate_result
            best_rank = candidate_rank
        if stop_after_first_success and bool(best_result.success):
            break

    return best_result


def _cfg_float_list(raw_value: Any, default: Sequence[float]) -> List[float]:
    if raw_value is None:
        return [float(v) for v in default]
    if isinstance(raw_value, (int, float)):
        return [float(raw_value)]
    try:
        return [float(v) for v in raw_value]
    except (TypeError, ValueError):
        return [float(v) for v in default]


def _pose_with_wrist_position(base_pose: HandPose, wrist_position: np.ndarray) -> HandPose:
    wrist_rotation = base_pose.rotation_matrix()
    return make_hand_pose(
        wrist_position=wrist_position,
        wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(wrist_rotation),
        wrist_rotation=wrist_rotation,
        joint_positions=base_pose.joint_positions,
    )


def _cat3_fast_pose_candidates(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    base_pose: HandPose,
    search_cfg: Mapping[str, Any],
) -> List[tuple[HandPose, Dict[str, float]]]:
    palm_local = _safe_normalize(
        np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
    )
    approach_axis = _safe_normalize(base_pose.rotation_matrix() @ palm_local)
    if np.linalg.norm(approach_axis) < 1e-8:
        approach_axis = _approach_direction_for_pose(runtime_cfg, contact_result, base_pose)
    if np.linalg.norm(approach_axis) < 1e-8:
        approach_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    side_axis = _safe_normalize(frame_R[:, 1])
    if np.linalg.norm(side_axis) < 1e-8:
        side_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vertical_axis = _safe_normalize(frame_R[:, 0])
    if np.linalg.norm(vertical_axis) < 1e-8:
        vertical_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    approach_offsets = _cfg_float_list(
        search_cfg.get("approach_offsets"),
        [
            -0.060,
            -0.052,
            -0.044,
            -0.042,
            -0.034,
            -0.026,
            -0.018,
            -0.010,
            -0.004,
            0.0,
            0.004,
            0.010,
            0.018,
            0.026,
            0.034,
            0.042,
            0.044,
            0.052,
            0.060,
        ],
    )
    side_offsets = _cfg_float_list(search_cfg.get("side_offsets"), [0.0])
    vertical_offsets = _cfg_float_list(search_cfg.get("vertical_offsets"), [-0.006, 0.0, 0.006])
    palm_roll_offsets_deg = _cfg_float_list(search_cfg.get("palm_roll_offsets_deg"), [0.0])
    max_candidates = max(1, int(search_cfg.get("max_pose_candidates", 57)))
    priority_approach_count = max(1, int(search_cfg.get("priority_approach_count", 7)))

    approach_offsets_sorted = sorted(approach_offsets, key=lambda value: (abs(value), value))
    side_offsets_sorted = sorted(side_offsets, key=lambda value: (abs(value), value))
    palm_roll_offsets_sorted = sorted(palm_roll_offsets_deg, key=lambda value: (abs(value), value))
    priority_approach_offsets = approach_offsets_sorted[: min(len(approach_offsets_sorted), priority_approach_count)]

    offset_specs: List[tuple[float, float, float, float, float]] = []

    def add_spec(approach_offset: float, side_offset: float, vertical_offset: float, palm_roll_deg: float) -> None:
        norm = float(
            np.linalg.norm(
                approach_axis * approach_offset
                + side_axis * side_offset
                + vertical_axis * vertical_offset
            )
        )
        offset_specs.append(
            (
                norm,
                float(approach_offset),
                float(side_offset),
                float(vertical_offset),
                float(palm_roll_deg),
            )
        )

    add_spec(0.0, 0.0, 0.0, 0.0)
    for vertical_offset in vertical_offsets:
        for approach_offset in priority_approach_offsets:
            for side_offset in side_offsets_sorted:
                for palm_roll_deg in palm_roll_offsets_sorted:
                    add_spec(approach_offset, side_offset, vertical_offset, palm_roll_deg)

    for approach_offset in approach_offsets:
        for side_offset in side_offsets:
            for vertical_offset in vertical_offsets:
                for palm_roll_deg in palm_roll_offsets_deg:
                    add_spec(approach_offset, side_offset, vertical_offset, palm_roll_deg)

    candidates: List[tuple[HandPose, Dict[str, float]]] = []
    seen: set[tuple[float, float, float, float]] = set()
    base_position = np.asarray(base_pose.wrist_position, dtype=np.float64)
    base_rotation = base_pose.rotation_matrix()
    for _, approach_offset, side_offset, vertical_offset, palm_roll_deg in offset_specs:
        key = (
            round(approach_offset, 6),
            round(side_offset, 6),
            round(vertical_offset, 6),
            round(palm_roll_deg, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        wrist_position = (
            base_position
            + approach_axis * approach_offset
            + side_axis * side_offset
            + vertical_axis * vertical_offset
        )
        wrist_rotation = base_rotation
        if abs(float(palm_roll_deg)) > 1e-8:
            delta_rotation = Rotation.from_rotvec(
                approach_axis * float(np.deg2rad(float(palm_roll_deg)))
            ).as_matrix()
            wrist_rotation = delta_rotation @ base_rotation
        candidates.append(
            (
                make_hand_pose(
                    wrist_position=wrist_position,
                    wrist_quaternion_xyzw=_rotation_matrix_to_quaternion_xyzw(wrist_rotation),
                    wrist_rotation=wrist_rotation,
                    joint_positions=base_pose.joint_positions,
                ),
                {
                    "approach_offset": float(approach_offset),
                    "side_offset": float(side_offset),
                    "vertical_offset": float(vertical_offset),
                    "palm_roll_deg": float(palm_roll_deg),
                },
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def _cat3_fast_stage_rank_key(stage_result: StagePoseResult) -> tuple[float, ...]:
    completion = stage_result.metadata.get("completion", {})
    max_bottom_violation = float(completion.get("max_bottom_violation", 0.0))
    max_global_penetration = float(completion.get("max_global_penetration", 0.0))
    allowed_global_penetration = float(completion.get("configured_max_global_penetration", 0.0))
    penetration_excess = max(0.0, max_global_penetration - allowed_global_penetration)
    max_active_excess = float(completion.get("max_active_excess", 0.0))
    max_palm_excess = float(completion.get("max_palm_excess", max_active_excess))
    max_finger_excess = float(completion.get("max_finger_excess", max_active_excess))
    max_thumb_excess = float(completion.get("max_thumb_excess", max_active_excess))
    active_target_position = float(stage_result.cost_breakdown.terms.get("active_target_position", np.inf))
    active_clearance = float(stage_result.cost_breakdown.terms.get("active_clearance", np.inf))
    completion_fail = 0.0 if bool(completion.get("passes_completion", False)) else 1.0
    return (
        max_bottom_violation,
        penetration_excess,
        max_palm_excess,
        max_finger_excess,
        max_thumb_excess,
        max_active_excess,
        active_target_position,
        max_global_penetration,
        active_clearance,
        completion_fail,
        float(stage_result.cost),
    )


def _stage_result_max_global_penetration(stage_result: StagePoseResult) -> float:
    completion = stage_result.metadata.get("completion", {})
    return float(completion.get("max_global_penetration", 0.0))


def _stage_result_allowed_global_penetration(stage_result: StagePoseResult) -> float:
    completion = stage_result.metadata.get("completion", {})
    return float(completion.get("configured_max_global_penetration", 0.0))


def _stage_result_max_bottom_violation(stage_result: StagePoseResult) -> float:
    completion = stage_result.metadata.get("completion", {})
    return float(completion.get("max_bottom_violation", 0.0))


def _stage_result_allowed_bottom_violation(stage_result: StagePoseResult) -> float:
    completion = stage_result.metadata.get("completion", {})
    return float(completion.get("configured_bottom_tolerance", 0.0))


def _cat3_semantic_basis_from_config(runtime_cfg: ResolvedHandRuntimeConfig) -> np.ndarray:
    frame_cfg = runtime_cfg.hand.frame_convention
    x_axis = _safe_normalize(np.asarray(frame_cfg.finger_forward_local, dtype=np.float64))
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    y_axis = np.asarray(frame_cfg.thumb_opposition_local, dtype=np.float64)
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        palm_fallback = np.asarray(frame_cfg.palm_normal_local, dtype=np.float64)
        palm_fallback = palm_fallback - x_axis * float(np.dot(palm_fallback, x_axis))
        y_axis = _safe_normalize(palm_fallback)
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    z_axis = _safe_normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    palm_local = _safe_normalize(np.asarray(frame_cfg.palm_normal_local, dtype=np.float64))
    if np.linalg.norm(palm_local) > 1e-8 and float(np.dot(z_axis, palm_local)) < 0.0:
        y_axis *= -1.0
        z_axis *= -1.0

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _cat3_seed_snapped_to_contact_frame(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seed: PoseSeed,
) -> PoseSeed:
    hand_basis_local = _cat3_semantic_basis_from_config(runtime_cfg)
    world_basis = np.asarray(contact_result.frame_R, dtype=np.float64)
    if world_basis.shape != (3, 3):
        return seed
    wrist_rotation = world_basis @ hand_basis_local.T
    if not np.all(np.isfinite(wrist_rotation)):
        return seed
    wrist_quaternion_xyzw = Rotation.from_matrix(wrist_rotation).as_quat()
    return PoseSeed(
        category=seed.category,
        contact_template=seed.contact_template,
        posture_name=seed.posture_name,
        approach_name=seed.approach_name,
        wrist_position=np.asarray(seed.wrist_position, dtype=np.float64).copy(),
        wrist_quaternion_xyzw=np.asarray(wrist_quaternion_xyzw, dtype=np.float64),
        wrist_rotation=wrist_rotation,
        joint_positions=dict(seed.joint_positions),
        metadata={
            **dict(seed.metadata),
            "cat3_rotation_snapped_to_contact_frame": True,
            "cat3_original_wrist_quaternion_xyzw": np.asarray(
                seed.wrist_quaternion_xyzw,
                dtype=np.float64,
            ).tolist(),
        },
    )


def _refine_cat3_stage_toward_base(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    safe_result: StagePoseResult,
    base_pose: HandPose,
    *,
    stage_name: StageName,
    prior_pose: Optional[HandPose],
    num_steps: int,
) -> tuple[StagePoseResult, int]:
    if int(num_steps) <= 0:
        return safe_result, 0

    allowed_penetration = _stage_result_allowed_global_penetration(safe_result)
    if _stage_result_max_global_penetration(safe_result) > allowed_penetration:
        return safe_result, 0
    allowed_bottom_violation = _stage_result_allowed_bottom_violation(safe_result)
    if _stage_result_max_bottom_violation(safe_result) > allowed_bottom_violation:
        return safe_result, 0

    safe_position = np.asarray(safe_result.hand_pose.wrist_position, dtype=np.float64)
    target_position = np.asarray(base_pose.wrist_position, dtype=np.float64)
    if float(np.linalg.norm(target_position - safe_position)) < 1e-8:
        return safe_result, 0

    target_result = evaluate_stage_pose(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=base_pose,
        stage_name=stage_name,
        prior_pose=prior_pose,
    )
    num_evaluated = 1
    if (
        _stage_result_max_global_penetration(target_result) <= allowed_penetration
        and _stage_result_max_bottom_violation(target_result) <= allowed_bottom_violation
    ):
        if _cat3_fast_stage_rank_key(target_result) < _cat3_fast_stage_rank_key(safe_result):
            return target_result, num_evaluated
        return safe_result, num_evaluated

    best_safe_result = safe_result
    low_position = safe_position
    high_position = target_position
    for _ in range(int(num_steps)):
        mid_position = 0.5 * (low_position + high_position)
        mid_pose = _pose_with_wrist_position(base_pose, mid_position)
        mid_result = evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=mid_pose,
            stage_name=stage_name,
            prior_pose=prior_pose,
        )
        num_evaluated += 1
        if (
            _stage_result_max_global_penetration(mid_result) <= allowed_penetration
            and _stage_result_max_bottom_violation(mid_result) <= allowed_bottom_violation
        ):
            best_safe_result = mid_result
            low_position = mid_position
        else:
            high_position = mid_position

    return best_safe_result, num_evaluated


def _annotate_cat3_joint_pose_bottom_clearance(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    stage_result: StagePoseResult,
    joint_positions: Mapping[str, float],
    *,
    prefix: str,
) -> None:
    bottom_cfg = _cat3_bottom_clearance_cfg(runtime_cfg)
    if not bool(bottom_cfg.get("enabled", True)):
        return

    base_pose = stage_result.hand_pose
    check_pose = make_hand_pose(
        wrist_position=np.asarray(base_pose.wrist_position, dtype=np.float64),
        wrist_quaternion_xyzw=np.asarray(base_pose.wrist_quaternion_xyzw, dtype=np.float64),
        wrist_rotation=base_pose.rotation_matrix(),
        joint_positions=joint_positions,
    )
    sphere_lookup, _ = _build_sphere_lookup(hand_model, runtime_cfg, check_pose)
    bottom_metrics = _hand_bottom_clearance_metrics(
        object_query,
        sphere_lookup,
        margin=float(bottom_cfg.get("margin", 0.0)),
        tolerance=float(bottom_cfg.get("tolerance", 0.0)),
        bbox_min_override=contact_result.metadata.get("object_bbox_min"),
    )
    completion = dict(stage_result.metadata.get("completion", {}))
    _annotate_completion_with_bottom_metrics(
        completion,
        bottom_metrics,
        prefix=prefix,
    )
    stage_result.metadata["completion"] = completion


def _evaluate_cat3_fast_stage(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    base_pose: HandPose,
    *,
    stage_name: StageName,
    prior_pose: Optional[HandPose] = None,
    search_cfg: Mapping[str, Any],
    bottom_check_joint_positions: Optional[Mapping[str, float]] = None,
) -> StagePoseResult:
    candidates = _cat3_fast_pose_candidates(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        base_pose=base_pose,
        search_cfg=search_cfg,
    )
    best_result: Optional[StagePoseResult] = None
    best_rank: Optional[tuple[float, ...]] = None
    best_candidate_index = 0
    best_offsets: Dict[str, float] = {}
    num_refine_evaluations = 0

    for candidate_index, (candidate_pose, offsets) in enumerate(candidates):
        candidate_result = evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=candidate_pose,
            stage_name=stage_name,
            prior_pose=prior_pose,
        )
        if (
            stage_name == "grasp"
            and bottom_check_joint_positions is not None
            and bool(search_cfg.get("include_joint_only_squeeze_bottom_clearance", True))
        ):
            _annotate_cat3_joint_pose_bottom_clearance(
                runtime_cfg,
                contact_result,
                object_query,
                hand_model,
                candidate_result,
                bottom_check_joint_positions,
                prefix="joint_only_squeeze",
            )
        candidate_rank = _cat3_fast_stage_rank_key(candidate_result)
        if best_result is None or best_rank is None or candidate_rank < best_rank:
            best_result = candidate_result
            best_rank = candidate_rank
            best_candidate_index = int(candidate_index)
            best_offsets = dict(offsets)

    if best_result is None:
        best_result = evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=base_pose,
            stage_name=stage_name,
            prior_pose=prior_pose,
        )
        if (
            stage_name == "grasp"
            and bottom_check_joint_positions is not None
            and bool(search_cfg.get("include_joint_only_squeeze_bottom_clearance", True))
        ):
            _annotate_cat3_joint_pose_bottom_clearance(
                runtime_cfg,
                contact_result,
                object_query,
                hand_model,
                best_result,
                bottom_check_joint_positions,
                prefix="joint_only_squeeze",
            )

    refine_steps = int(search_cfg.get("refine_toward_base_steps", 8))
    pre_refine_result = best_result
    refine_reverted_for_bottom_clearance = False
    best_result, num_refine_evaluations = _refine_cat3_stage_toward_base(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        safe_result=best_result,
        base_pose=base_pose,
        stage_name=stage_name,
        prior_pose=prior_pose,
        num_steps=refine_steps,
    )
    if (
        stage_name == "grasp"
        and bottom_check_joint_positions is not None
        and bool(search_cfg.get("include_joint_only_squeeze_bottom_clearance", True))
    ):
        _annotate_cat3_joint_pose_bottom_clearance(
            runtime_cfg,
            contact_result,
            object_query,
            hand_model,
            best_result,
            bottom_check_joint_positions,
            prefix="joint_only_squeeze",
        )
        allowed_bottom_violation = _stage_result_allowed_bottom_violation(best_result)
        if (
            _stage_result_max_bottom_violation(best_result) > allowed_bottom_violation
            and _stage_result_max_bottom_violation(pre_refine_result) <= allowed_bottom_violation
        ):
            best_result = pre_refine_result
            refine_reverted_for_bottom_clearance = True

    best_result.metadata["cat3_fast_pose_search"] = {
        "enabled": True,
        "num_candidates": int(len(candidates)),
        "num_refine_evaluations": int(num_refine_evaluations),
        "selected_candidate_index": int(best_candidate_index),
        "selected_offsets": best_offsets,
        "refine_reverted_for_bottom_clearance": bool(refine_reverted_for_bottom_clearance),
        "rotation_locked_to_seed": True,
    }
    return best_result


def _cat3_pregrasp_from_contact_rank_key(
    stage_result: StagePoseResult,
    *,
    retreat_offset: float,
    vertical_offset: float,
    preferred_retreat: float,
    min_retreat: float,
    preferred_vertical_offset: float,
    min_vertical_offset: float,
) -> tuple[float, ...]:
    completion = stage_result.metadata.get("completion", {})
    max_bottom_violation = float(completion.get("max_bottom_violation", 0.0))
    max_global_penetration = float(completion.get("max_global_penetration", 0.0))
    allowed_global_penetration = float(completion.get("configured_max_global_penetration", 0.0))
    penetration_excess = max(0.0, max_global_penetration - allowed_global_penetration)
    completion_fail = 0.0 if bool(completion.get("passes_completion", False)) else 1.0
    retreat_shortfall = max(0.0, float(min_retreat) - float(retreat_offset))
    vertical_shortfall = max(0.0, float(min_vertical_offset) - float(vertical_offset))
    preferred_retreat_error = abs(float(retreat_offset) - float(preferred_retreat))
    preferred_vertical_error = abs(float(vertical_offset) - float(preferred_vertical_offset))
    collision_cost = (
        float(stage_result.cost_breakdown.terms.get("non_active_collision", 0.0))
        + float(stage_result.cost_breakdown.terms.get("palm_collision", 0.0))
        + float(stage_result.cost_breakdown.terms.get("self_collision", 0.0))
    )
    return (
        max_bottom_violation,
        penetration_excess,
        completion_fail,
        vertical_shortfall,
        retreat_shortfall,
        preferred_retreat_error,
        preferred_vertical_error,
        collision_cost,
        float(stage_result.cost),
    )


def _cat3_pregrasp_from_contact_stage(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    grasp_result: StagePoseResult,
    base_pregrasp_pose: HandPose,
    search_cfg: Mapping[str, Any],
) -> StagePoseResult:
    if not bool(search_cfg.get("pregrasp_from_contact", True)):
        return evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=base_pregrasp_pose,
            stage_name="pregrasp",
        )

    contact_pose = grasp_result.hand_pose
    frame_R = np.asarray(contact_result.frame_R, dtype=np.float64)
    approach_direction = _safe_normalize(np.array([frame_R[0, 2], frame_R[1, 2], 0.0], dtype=np.float64))
    if np.linalg.norm(approach_direction) < 1e-8:
        palm_world = _safe_normalize(
            contact_pose.rotation_matrix()
            @ np.asarray(runtime_cfg.hand.frame_convention.palm_normal_local, dtype=np.float64)
        )
        approach_direction = _safe_normalize(np.array([palm_world[0], palm_world[1], 0.0], dtype=np.float64))
    if np.linalg.norm(approach_direction) < 1e-8:
        target_points = [
            np.asarray(target.target_point, dtype=np.float64)
            for target in contact_result.active_targets
        ]
        if target_points:
            target_centroid = np.mean(np.asarray(target_points, dtype=np.float64), axis=0)
            approach_direction = _safe_normalize(
                np.asarray(contact_result.anchor.point, dtype=np.float64) - target_centroid
            )
            approach_direction = _safe_normalize(
                np.array([approach_direction[0], approach_direction[1], 0.0], dtype=np.float64)
            )
    if np.linalg.norm(approach_direction) < 1e-8:
        approach_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    retreat_offsets = _cfg_float_list(
        search_cfg.get("pregrasp_retreat_offsets"),
        [0.045, 0.060, 0.030, 0.075, 0.015, 0.090, 0.0],
    )
    vertical_offsets = _cfg_float_list(
        search_cfg.get("pregrasp_vertical_offsets"),
        [0.0, 0.006, 0.012, 0.024],
    )
    preferred_retreat = float(search_cfg.get("pregrasp_preferred_retreat", 0.045))
    min_retreat = float(search_cfg.get("pregrasp_min_retreat", 0.020))
    preferred_vertical_offset = float(search_cfg.get("pregrasp_preferred_vertical_offset", 0.0))
    min_vertical_offset = float(search_cfg.get("pregrasp_min_vertical_offset", 0.0))
    max_evaluations = max(1, int(search_cfg.get("pregrasp_max_evaluations", 8)))

    base_position = np.asarray(contact_pose.wrist_position, dtype=np.float64)
    base_rotation = contact_pose.rotation_matrix()
    base_quat = _rotation_matrix_to_quaternion_xyzw(base_rotation)
    pregrasp_joint_positions = dict(contact_pose.joint_positions)

    contact_sphere_lookup, _ = _build_sphere_lookup(hand_model, runtime_cfg, contact_pose)
    bottom_cfg = _cat3_bottom_clearance_cfg(runtime_cfg)
    bottom_metrics = _hand_bottom_clearance_metrics(
        object_query,
        contact_sphere_lookup,
        margin=float(bottom_cfg.get("margin", 0.0)),
        tolerance=float(bottom_cfg.get("tolerance", 0.0)),
        bbox_min_override=contact_result.metadata.get("object_bbox_min"),
    )
    min_contact_bottom_z = float(bottom_metrics.get("min_hand_bottom_z", np.inf))
    required_bottom_z = float(bottom_metrics.get("required_bottom_z", -np.inf))

    candidate_specs: List[tuple[tuple[float, ...], float, float, float]] = []
    seen: set[tuple[float, float]] = set()
    for retreat_offset in retreat_offsets:
        for vertical_offset in vertical_offsets:
            retreat_offset = float(retreat_offset)
            vertical_offset = float(vertical_offset)
            key = (round(retreat_offset, 6), round(vertical_offset, 6))
            if key in seen:
                continue
            seen.add(key)
            delta_z = float((-approach_direction * retreat_offset + up_axis * vertical_offset)[2])
            predicted_min_bottom_z = min_contact_bottom_z + delta_z
            predicted_violation = (
                max(0.0, required_bottom_z - predicted_min_bottom_z)
                if np.isfinite(min_contact_bottom_z) and np.isfinite(required_bottom_z)
                else 0.0
            )
            retreat_shortfall = max(0.0, min_retreat - retreat_offset)
            vertical_shortfall = max(0.0, min_vertical_offset - vertical_offset)
            candidate_specs.append(
                (
                    (
                        float(predicted_violation),
                        float(vertical_shortfall),
                        float(retreat_shortfall),
                        abs(retreat_offset - preferred_retreat),
                        abs(vertical_offset - preferred_vertical_offset),
                    ),
                    retreat_offset,
                    vertical_offset,
                    float(predicted_min_bottom_z),
                )
            )
    candidate_specs.sort(key=lambda item: item[0])

    best_result: Optional[StagePoseResult] = None
    best_rank: Optional[tuple[float, ...]] = None
    best_offsets: Dict[str, float] = {}
    num_evaluated = 0

    for _, retreat_offset, vertical_offset, predicted_min_bottom_z in candidate_specs[:max_evaluations]:
        wrist_position = (
            base_position
            - approach_direction * retreat_offset
            + up_axis * vertical_offset
        )
        candidate_pose = make_hand_pose(
            wrist_position=wrist_position,
            wrist_quaternion_xyzw=base_quat,
            wrist_rotation=base_rotation,
            joint_positions=pregrasp_joint_positions,
        )
        candidate_result = evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=candidate_pose,
            stage_name="pregrasp",
            prior_pose=None,
        )
        num_evaluated += 1
        candidate_rank = _cat3_pregrasp_from_contact_rank_key(
            candidate_result,
            retreat_offset=retreat_offset,
            vertical_offset=vertical_offset,
            preferred_retreat=preferred_retreat,
            min_retreat=min_retreat,
            preferred_vertical_offset=preferred_vertical_offset,
            min_vertical_offset=min_vertical_offset,
        )
        if best_result is None or best_rank is None or candidate_rank < best_rank:
            best_result = candidate_result
            best_rank = candidate_rank
            best_offsets = {
                "retreat_offset": float(retreat_offset),
                "vertical_offset": float(vertical_offset),
                "predicted_min_hand_bottom_z": float(predicted_min_bottom_z),
            }
        if bool(candidate_result.metadata.get("completion", {}).get("passes_completion", False)):
            break

    if best_result is None:
        best_result = evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=base_pregrasp_pose,
            stage_name="pregrasp",
        )

    best_result.metadata["cat3_pregrasp_from_contact"] = {
        "enabled": True,
        "num_candidates": int(len(candidate_specs)),
        "num_evaluated": int(num_evaluated),
        "selected_offsets": best_offsets,
        "joint_source": "contact_stage",
        "rotation_source": "contact_stage",
    }
    return best_result


def _cat3_post_contact_squeeze_result(
    grasp_result: StagePoseResult,
    squeeze_joint_positions: Mapping[str, float],
    *,
    runtime_cfg: Optional[ResolvedHandRuntimeConfig] = None,
    contact_result: Optional[ContactResolutionResult] = None,
    object_query: Optional[ObjectQueryBackend] = None,
    hand_model: Optional[HandKinematicsModel] = None,
) -> StagePoseResult:
    grasp_pose = grasp_result.hand_pose
    squeeze_pose = make_hand_pose(
        wrist_position=np.asarray(grasp_pose.wrist_position, dtype=np.float64),
        wrist_quaternion_xyzw=grasp_pose.wrist_quaternion_xyzw,
        wrist_rotation=grasp_pose.rotation_matrix(),
        joint_positions=squeeze_joint_positions,
    )
    grasp_completion = dict(grasp_result.metadata.get("completion", {}))
    if runtime_cfg is not None and object_query is not None and hand_model is not None:
        bottom_cfg = _cat3_bottom_clearance_cfg(runtime_cfg)
        if bool(bottom_cfg.get("enabled", True)):
            sphere_lookup, _ = _build_sphere_lookup(hand_model, runtime_cfg, squeeze_pose)
            bottom_metrics = _hand_bottom_clearance_metrics(
                object_query,
                sphere_lookup,
                margin=float(bottom_cfg.get("margin", 0.0)),
                tolerance=float(bottom_cfg.get("tolerance", 0.0)),
                bbox_min_override=(
                    None
                    if contact_result is None
                    else contact_result.metadata.get("object_bbox_min")
                ),
            )
            _annotate_completion_with_bottom_metrics(
                grasp_completion,
                bottom_metrics,
                prefix="joint_only_squeeze",
            )
    return StagePoseResult(
        stage_name="squeeze",
        hand_pose=squeeze_pose,
        success=True,
        cost=0.0,
        message="SKIPPED: post-contact joint-only squeeze; collision and wrist motion are not evaluated",
        num_iterations=0,
        cost_breakdown=CostBreakdown(),
        metadata={
            "optimizer_status": 0,
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "used_separate_prior": False,
            "solver_success": True,
            "solver_skipped": True,
            "completion": grasp_completion,
            "cat3_post_contact_squeeze": {
                "enabled": True,
                "wrist_locked_to_grasp": True,
                "collision_evaluated": False,
                "movement_evaluated": False,
                "source_stage": "grasp",
            },
        },
    )


def _optimize_cat3_fast_staged_grasp(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seed: PoseSeed,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
) -> StagedGraspResult:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    stage_init_cfg = cfg["stage_init"]
    search_cfg = cfg.get("contact_model", {}).get("cat3_fast_pose_search", {})
    if bool(search_cfg.get("snap_rotation_to_contact_frame", True)):
        seed = _cat3_seed_snapped_to_contact_frame(runtime_cfg, contact_result, seed)
    init_poses = build_cat1_stage_initializations(runtime_cfg, contact_result, seed)

    seed_pregrasp_result = evaluate_stage_pose(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        hand_pose=init_poses["pregrasp"],
        stage_name="pregrasp",
    )
    grasp_result = _evaluate_cat3_fast_stage(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        base_pose=init_poses["grasp"],
        stage_name="grasp",
        prior_pose=seed_pregrasp_result.hand_pose,
        search_cfg=search_cfg,
        bottom_check_joint_positions=init_poses["squeeze"].joint_positions,
    )
    pregrasp_result = _cat3_pregrasp_from_contact_stage(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        grasp_result=grasp_result,
        base_pregrasp_pose=init_poses["pregrasp"],
        search_cfg=search_cfg,
    )
    if bool(search_cfg.get("squeeze_collision_evaluation", False)):
        squeeze_result = _evaluate_cat3_fast_stage(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            base_pose=init_poses["squeeze"],
            stage_name="squeeze",
            prior_pose=grasp_result.hand_pose,
            search_cfg=search_cfg,
        )
        squeeze_role = "translation_grid_final"
    else:
        squeeze_result = _cat3_post_contact_squeeze_result(
            grasp_result,
            init_poses["squeeze"].joint_positions,
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
        )
        squeeze_role = "post_contact_joint_only"

    stage_results: Dict[StageName, StagePoseResult] = {
        "pregrasp": pregrasp_result,
        "grasp": grasp_result,
        "squeeze": squeeze_result,
    }
    total_cost = float(sum(stage.cost for stage in stage_results.values()))
    success = all(stage.success for stage in stage_results.values())
    return StagedGraspResult(
        category=contact_result.category,
        contact_template=contact_result.contact_template,
        seed=seed,
        stage_results=stage_results,
        total_cost=total_cost,
        success=success,
        metadata={
            "seed_posture_name": seed.posture_name,
            "seed_approach_name": seed.approach_name,
            "stage_init_blend": {
                "grasp_wrist_blend_alpha": float(stage_init_cfg.get("grasp_wrist_blend_alpha", 0.50)),
                "grasp_joint_blend_alpha": 0.0,
                "squeeze_wrist_blend_alpha": float(stage_init_cfg.get("squeeze_wrist_blend_alpha", 0.35)),
                "squeeze_joint_blend_alpha": 0.0,
            },
            "stage_roles": {
                "pregrasp": "seed",
                "grasp": "translation_grid",
                "squeeze": squeeze_role,
            },
            "cat3_fast_pose_search": {
                "enabled": True,
                "rotation_locked_to_seed": True,
                "squeeze_collision_evaluation": bool(search_cfg.get("squeeze_collision_evaluation", False)),
                "max_pose_candidates": int(search_cfg.get("max_pose_candidates", 57)),
            },
        },
    )


def optimize_cat1_staged_grasp(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seed: PoseSeed,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
) -> StagedGraspResult:
    cfg = _merged_optimizer_cfg(runtime_cfg)
    cat3_fast_cfg = cfg.get("contact_model", {}).get("cat3_fast_pose_search", {})
    if contact_result.category == "cat3" and bool(cat3_fast_cfg.get("enabled", True)):
        return _optimize_cat3_fast_staged_grasp(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            seed=seed,
            object_query=object_query,
            hand_model=hand_model,
        )

    init_poses = build_cat1_stage_initializations(runtime_cfg, contact_result, seed)
    stage_init_cfg = cfg["stage_init"]
    joint_names = list(runtime_cfg.hand.joints.controllable)
    open_pose = _resolve_posture(runtime_cfg, str(stage_init_cfg["open_posture_name"]))
    close_posture_name = _metadata_str(
        contact_result.metadata,
        "cat1_close_posture_name",
        str(stage_init_cfg["close_posture_name"]),
    )
    close_pose = _resolve_posture(runtime_cfg, close_posture_name)
    if not open_pose:
        open_pose = dict(seed.joint_positions)
    if not close_pose:
        close_pose = dict(seed.joint_positions)
    limit_close_pose = _derive_limit_close_pose(
        runtime_cfg,
        open_pose,
        close_pose,
        joint_names,
    )
    final_contact_seek_cfg = cfg.get("final_contact_seek", {})
    final_close_pose = _blend_joint_positions(
        close_pose,
        limit_close_pose,
        _metadata_float(
            contact_result.metadata,
            "cat1_limit_close_blend",
            float(final_contact_seek_cfg.get("limit_close_blend", 0.35)),
        ),
        joint_names,
    )

    if contact_result.category in {"cat3", "cat4"}:
        # Cat3/cat4 seeds already serve as the intended pregrasp regime;
        # evaluating them directly avoids one full solver pass per candidate.
        pregrasp_result = evaluate_stage_pose(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            hand_pose=init_poses["pregrasp"],
            stage_name="pregrasp",
        )
    else:
        pregrasp_result = optimize_cat1_stage(
            runtime_cfg=runtime_cfg,
            contact_result=contact_result,
            object_query=object_query,
            hand_model=hand_model,
            init_pose=init_poses["pregrasp"],
            stage_name="pregrasp",
        )

    grasp_start_pose = _blend_pose_hint(
        pregrasp_result.hand_pose,
        init_poses["grasp"],
        wrist_alpha=float(stage_init_cfg.get("grasp_wrist_blend_alpha", 0.50)),
        joint_alpha=float(stage_init_cfg.get("grasp_joint_blend_alpha", 0.70)),
        joint_names=joint_names,
    )
    grasp_result = optimize_cat1_stage(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        init_pose=grasp_start_pose,
        stage_name="grasp",
    )

    squeeze_start_pose = _blend_pose_hint(
        grasp_result.hand_pose,
        init_poses["squeeze"],
        wrist_alpha=float(stage_init_cfg.get("squeeze_wrist_blend_alpha", 0.35)),
        joint_alpha=float(stage_init_cfg.get("squeeze_joint_blend_alpha", 0.75)),
        joint_names=joint_names,
    )
    squeeze_result = optimize_cat1_stage(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        init_pose=squeeze_start_pose,
        stage_name="squeeze",
        prior_pose=grasp_result.hand_pose,
    )
    squeeze_result = _contact_seek_final_stage(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        object_query=object_query,
        hand_model=hand_model,
        coarse_pose=grasp_result.hand_pose,
        current_result=squeeze_result,
        final_close_pose=final_close_pose,
        joint_names=joint_names,
    )

    stage_results: Dict[StageName, StagePoseResult] = {
        "pregrasp": pregrasp_result,
        "grasp": grasp_result,
        "squeeze": squeeze_result,
    }
    total_cost = float(sum(stage.cost for stage in stage_results.values()))
    success = all(stage.success for stage in stage_results.values())
    return StagedGraspResult(
        category=contact_result.category,
        contact_template=contact_result.contact_template,
        seed=seed,
        stage_results=stage_results,
        total_cost=total_cost,
        success=success,
        metadata={
            "seed_posture_name": seed.posture_name,
            "seed_approach_name": seed.approach_name,
            "stage_init_blend": {
                "grasp_wrist_blend_alpha": float(stage_init_cfg.get("grasp_wrist_blend_alpha", 0.50)),
                "grasp_joint_blend_alpha": float(stage_init_cfg.get("grasp_joint_blend_alpha", 0.70)),
                "squeeze_wrist_blend_alpha": float(stage_init_cfg.get("squeeze_wrist_blend_alpha", 0.35)),
                "squeeze_joint_blend_alpha": float(stage_init_cfg.get("squeeze_joint_blend_alpha", 0.75)),
            },
            "stage_roles": {
                "pregrasp": "find",
                "grasp": "coarse",
                "squeeze": "final",
            },
        },
    )


def optimize_cat1_staged_grasps(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seeds: Sequence[PoseSeed],
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    max_workers: Optional[int] = None,
) -> List[StagedGraspResult]:
    seed_list = list(seeds)
    if not seed_list:
        return []

    requested_max_workers = 0 if max_workers is None else int(max_workers)
    if requested_max_workers <= 0:
        effective_max_workers = min(len(seed_list), max(1, os.cpu_count() or 1), 8)
    else:
        effective_max_workers = min(len(seed_list), max(1, requested_max_workers))

    if effective_max_workers <= 1:
        results = [
            optimize_cat1_staged_grasp(
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
                seed=seed,
                object_query=object_query,
                hand_model=hand_model,
            )
            for seed in seed_list
        ]
    else:
        results_by_index: Dict[int, StagedGraspResult] = {}
        with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
            future_to_index = {
                executor.submit(
                    optimize_cat1_staged_grasp,
                    runtime_cfg=runtime_cfg,
                    contact_result=contact_result,
                    seed=seed,
                    object_query=object_query,
                    hand_model=hand_model,
                ): idx
                for idx, seed in enumerate(seed_list)
            }
            for future in as_completed(future_to_index):
                results_by_index[future_to_index[future]] = future.result()
        results = [results_by_index[idx] for idx in range(len(seed_list))]
    return sort_staged_grasp_results(contact_result, results)


def optimize_staged_grasp(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seed: PoseSeed,
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
) -> StagedGraspResult:
    return optimize_cat1_staged_grasp(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        seed=seed,
        object_query=object_query,
        hand_model=hand_model,
    )


def optimize_staged_grasps(
    runtime_cfg: ResolvedHandRuntimeConfig,
    contact_result: ContactResolutionResult,
    seeds: Sequence[PoseSeed],
    object_query: ObjectQueryBackend,
    hand_model: HandKinematicsModel,
    max_workers: Optional[int] = None,
) -> List[StagedGraspResult]:
    return optimize_cat1_staged_grasps(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        seeds=seeds,
        object_query=object_query,
        hand_model=hand_model,
        max_workers=max_workers,
    )


def summarize_staged_grasp_result(result: StagedGraspResult) -> Dict[str, Any]:
    packed_stages: Dict[str, Any] = {}
    for stage_name, stage_result in result.stage_results.items():
        quat_xyzw = _rotation_matrix_to_quaternion_xyzw(stage_result.hand_pose.rotation_matrix())
        quat_wxyz = _quaternion_wxyz_from_xyzw(quat_xyzw)
        packed_stages[stage_name] = {
            "success": bool(stage_result.success),
            "cost": float(stage_result.cost),
            "message": stage_result.message,
            "num_iterations": int(stage_result.num_iterations),
            "wrist_position": [
                float(x) for x in np.asarray(stage_result.hand_pose.wrist_position, dtype=np.float64).tolist()
            ],
            "wrist_quaternion_wxyz": [
                float(x) for x in quat_wxyz.tolist()
            ],
            "wrist_quaternion_xyzw": [
                float(x) for x in quat_xyzw.tolist()
            ],
            "joint_positions": {
                joint_name: float(value)
                for joint_name, value in stage_result.hand_pose.joint_positions.items()
            },
            "cost_breakdown": {
                name: float(value) for name, value in stage_result.cost_breakdown.terms.items()
            },
            "metadata": stage_result.metadata,
        }

    return {
        "category": result.category,
        "contact_template": result.contact_template,
        "seed_posture_name": result.seed.posture_name,
        "seed_approach_name": result.seed.approach_name,
        "success": bool(result.success),
        "total_cost": float(result.total_cost),
        "stages": packed_stages,
        "metadata": result.metadata,
    }
