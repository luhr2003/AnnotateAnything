from __future__ import annotations

import multiprocessing as mp
import os
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config_loader import load_all_for_hand
from src.contact_resolution import (
    resolve_contact_templates_for_anchors,
    summarize_contact_resolution,
)
from src.hand_kinematics import load_hand_kinematics_model
from src.object_query import (
    default_sdf_cache_path,
    load_or_build_sdf_from_usd,
    make_object_query_from_sdf,
)
from src.optimizer_core import optimize_staged_grasps, summarize_staged_grasp_result
from src.pipeline_single_hand import SingleHandProposalResult, run_single_hand_region_proposal
from src.region_proposal import Anchor
from src.seed_generation import generate_pose_seeds_for_contacts, summarize_seed_generation
from src.staged_grasp_pipeline import (
    ContactSeedOptimizationBundle,
)
from src.types_config import Category, ResolvedHandRuntimeConfig, Side


@dataclass
class BimanualAnchorPair:
    primary_anchor: Anchor
    opposite_anchor: Anchor
    primary_anchor_rank: int
    opposite_anchor_rank: int
    pair_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BimanualContactSeedOptimizationBundle:
    primary_side: Side
    opposite_side: Side
    anchor_pair: BimanualAnchorPair
    bundles_by_side: Dict[Side, ContactSeedOptimizationBundle]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BimanualStagedPipelineResult:
    proposal_result: SingleHandProposalResult
    runtime_cfg_by_side: Dict[Side, ResolvedHandRuntimeConfig]
    pair_bundles: List[BimanualContactSeedOptimizationBundle]
    evaluated_pair_bundles: List[BimanualContactSeedOptimizationBundle]
    sdf_cache_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


def _opposite_side(side: Side) -> Side:
    return "left" if side == "right" else "right"


_PROCESS_SIDE_BUNDLE_CONTEXT: Dict[str, Any] = {}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _runtime_cfg_with_optimizer_overrides(
    runtime_cfg: ResolvedHandRuntimeConfig,
    optimizer_cfg_overrides: Optional[Dict[str, Any]],
) -> ResolvedHandRuntimeConfig:
    if not optimizer_cfg_overrides:
        return runtime_cfg
    cloned = deepcopy(runtime_cfg)
    optimizer_cfg = deepcopy(cloned.optimizer_cfg or {})
    for scope in ("global", "category"):
        override_scope = optimizer_cfg_overrides.get(scope)
        if isinstance(override_scope, dict):
            optimizer_cfg[scope] = _deep_merge_dict(
                optimizer_cfg.get(scope, {}),
                override_scope,
            )
    cloned.optimizer_cfg = optimizer_cfg
    return cloned


def _init_process_side_bundle_worker(
    samples,
    descriptors,
    object_query,
    hand_model_by_side,
    num_seeds_per_contact: Optional[int],
    top_k_seeds_per_contact: int,
    top_k_optimized_per_side: Optional[int],
    contact_cfg_overrides: Optional[Dict[str, Any]],
) -> None:
    global _PROCESS_SIDE_BUNDLE_CONTEXT
    _PROCESS_SIDE_BUNDLE_CONTEXT = {
        "samples": samples,
        "descriptors": descriptors,
        "object_query": object_query,
        "hand_model_by_side": hand_model_by_side,
        "num_seeds_per_contact": num_seeds_per_contact,
        "top_k_seeds_per_contact": top_k_seeds_per_contact,
        "top_k_optimized_per_side": top_k_optimized_per_side,
        "contact_cfg_overrides": contact_cfg_overrides,
    }


def _compute_side_task_process(task: Dict[str, Any]) -> tuple[tuple[str, int], ContactSeedOptimizationBundle]:
    context = _PROCESS_SIDE_BUNDLE_CONTEXT
    side = task["side"]
    bundle = _build_side_bundle(
        runtime_cfg=task["runtime_cfg"],
        anchor=task["anchor"],
        samples=context["samples"],
        descriptors=context["descriptors"],
        object_query=context["object_query"],
        hand_model=context["hand_model_by_side"][side],
        seed=task["seed"],
        num_seeds_per_contact=context["num_seeds_per_contact"],
        top_k_seeds_per_contact=context["top_k_seeds_per_contact"],
        top_k_optimized_per_side=context["top_k_optimized_per_side"],
        contact_cfg_overrides=context["contact_cfg_overrides"],
    )
    return task["cache_key"], bundle


def _cat2_line_distribution_from_anchors(anchors: List[Anchor]) -> Optional[Dict[str, Any]]:
    if len(anchors) < 3:
        return None
    mode = str(anchors[0].metadata.get("cat2_anchor_distribution_mode", "default"))
    if mode != "bottom_rim_line":
        return None
    if "cat2_line_center_xy" not in anchors[0].metadata or "cat2_line_axis_xy" not in anchors[0].metadata:
        return None
    return {
        "mode": mode,
        "center_xy": np.asarray(anchors[0].metadata.get("cat2_line_center_xy", [0.0, 0.0]), dtype=np.float64),
        "line_axis_xy": np.asarray(anchors[0].metadata.get("cat2_line_axis_xy", [1.0, 0.0]), dtype=np.float64),
        "side_axis_xy": np.asarray(anchors[0].metadata.get("cat2_line_side_axis_xy", [0.0, 1.0]), dtype=np.float64),
        "major_span": float(anchors[0].metadata.get("cat2_line_major_span", 0.0)),
        "side_span": float(anchors[0].metadata.get("cat2_line_side_span", 0.0)),
    }


def _select_cat2_line_symmetric_pairs(
    anchors: List[Anchor],
    *,
    primary_anchor_limit: int,
    n_candidates: int,
    max_pairs: int,
    line_center_xy: np.ndarray,
    line_axis_xy: np.ndarray,
    line_side_axis_xy: np.ndarray,
    min_tangent_parallel_dot: float = 0.60,
) -> List[BimanualAnchorPair]:
    if len(anchors) < 2:
        return []

    anchors_sorted = sorted(anchors, key=lambda item: item.score, reverse=True)
    xy = np.asarray([anchor.point[:2] for anchor in anchors_sorted], dtype=np.float64)
    centered_xy = xy - np.asarray(line_center_xy, dtype=np.float64)[None, :]
    line_axis_xy = np.asarray(line_axis_xy, dtype=np.float64)
    line_side_axis_xy = np.asarray(line_side_axis_xy, dtype=np.float64)
    line_axis_xy = line_axis_xy / max(np.linalg.norm(line_axis_xy), 1e-8)
    line_side_axis_xy = line_side_axis_xy / max(np.linalg.norm(line_side_axis_xy), 1e-8)

    line_coord = centered_xy @ line_axis_xy
    side_coord = centered_xy @ line_side_axis_xy
    scores = np.asarray([anchor.score for anchor in anchors_sorted], dtype=np.float64)
    tangent_xy = np.asarray([anchor.frame_R[:2, 0] for anchor in anchors_sorted], dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent_xy, axis=1, keepdims=True)
    tangent_xy = tangent_xy / np.where(tangent_norm > 1e-8, tangent_norm, 1.0)
    z_vals = np.asarray([anchor.point[2] for anchor in anchors_sorted], dtype=np.float64)
    local_widths = np.asarray(
        [float(anchor.metadata.get("local_width", 0.0)) for anchor in anchors_sorted],
        dtype=np.float64,
    )

    primary_limit = min(len(anchors_sorted), max(1, int(primary_anchor_limit)))
    n_candidates = max(1, int(n_candidates))
    max_pairs = max(1, int(max_pairs))
    min_tangent_parallel_dot = float(np.clip(min_tangent_parallel_dot, 0.0, 1.0))

    seen: set[tuple[int, int]] = set()
    pair_list: List[BimanualAnchorPair] = []
    for i in range(primary_limit):
        candidates = np.arange(len(anchors_sorted), dtype=np.int64)
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue

        symmetry_target = -float(line_coord[i])
        symmetry_error = np.abs(line_coord[candidates] - symmetry_target)
        side_difference = np.abs(side_coord[candidates] - side_coord[i])
        span_distance = np.abs(line_coord[candidates] - line_coord[i])
        tangent_alignment = np.abs(tangent_xy[candidates] @ tangent_xy[i])
        z_diff = np.abs(z_vals[candidates] - z_vals[i])
        width_diff = np.abs(local_widths[candidates] - local_widths[i])

        opposite_side_mask = line_coord[candidates] * line_coord[i] <= 0.0
        if np.any(opposite_side_mask):
            candidates = candidates[opposite_side_mask]
            symmetry_error = symmetry_error[opposite_side_mask]
            side_difference = side_difference[opposite_side_mask]
            span_distance = span_distance[opposite_side_mask]
            tangent_alignment = tangent_alignment[opposite_side_mask]
            z_diff = z_diff[opposite_side_mask]
            width_diff = width_diff[opposite_side_mask]

        if np.any(tangent_alignment >= min_tangent_parallel_dot):
            tangent_mask = tangent_alignment >= min_tangent_parallel_dot
            candidates = candidates[tangent_mask]
            symmetry_error = symmetry_error[tangent_mask]
            side_difference = side_difference[tangent_mask]
            span_distance = span_distance[tangent_mask]
            tangent_alignment = tangent_alignment[tangent_mask]
            z_diff = z_diff[tangent_mask]
            width_diff = width_diff[tangent_mask]

        if len(candidates) == 0:
            continue

        rank_order = np.lexsort(
            (
                -scores[candidates],
                z_diff,
                width_diff,
                1.0 - tangent_alignment,
                -span_distance,
                side_difference,
                symmetry_error,
            )
        )
        ranked = candidates[rank_order[:n_candidates]]
        for j in ranked:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            tangent_parallel_dot = float(np.abs(np.dot(tangent_xy[i], tangent_xy[int(j)])))
            pair_list.append(
                BimanualAnchorPair(
                    primary_anchor=anchors_sorted[i],
                    opposite_anchor=anchors_sorted[int(j)],
                    primary_anchor_rank=int(i),
                    opposite_anchor_rank=int(j),
                    pair_score=float(scores[i] + scores[int(j)]),
                    metadata={
                        "pairing_geometry_mode": "cat2_line_symmetric",
                        "center_xy": [float(x) for x in np.asarray(line_center_xy, dtype=np.float64).tolist()],
                        "line_axis_xy": [float(x) for x in line_axis_xy.tolist()],
                        "line_side_axis_xy": [float(x) for x in line_side_axis_xy.tolist()],
                        "primary_projected_xy": [float(x) for x in xy[i].tolist()],
                        "opposite_projected_xy": [float(x) for x in xy[int(j)].tolist()],
                        "line_coord_axis": "line",
                        "across_axis": "line",
                        "along_axis": "side",
                        "primary_line_coord": float(line_coord[i]),
                        "opposite_line_coord": float(line_coord[int(j)]),
                        "line_symmetry_error": float(abs(line_coord[int(j)] + line_coord[i])),
                        "line_side_difference": float(abs(side_coord[int(j)] - side_coord[i])),
                        "across_difference": float(abs(line_coord[int(j)] - line_coord[i])),
                        "along_difference": float(abs(side_coord[int(j)] - side_coord[i])),
                        "tangent_parallel_dot": tangent_parallel_dot,
                        "local_width_difference": float(abs(local_widths[i] - local_widths[int(j)])),
                        "z_difference": float(abs(z_vals[i] - z_vals[int(j)])),
                        "min_tangent_parallel_dot": min_tangent_parallel_dot,
                    },
                )
            )

    pair_list.sort(
        key=lambda item: (
            int(item.primary_anchor_rank),
            float(item.metadata.get("line_symmetry_error", np.inf)),
            float(item.metadata.get("line_side_difference", np.inf)),
            -float(item.metadata.get("across_difference", 0.0)),
            -float(item.metadata.get("tangent_parallel_dot", 0.0)),
            float(item.metadata.get("local_width_difference", np.inf)),
            float(item.metadata.get("z_difference", np.inf)),
            -float(item.pair_score),
        )
    )
    return pair_list[:max_pairs]


def _select_opposite_anchor_pairs(
    anchors: List[Anchor],
    *,
    primary_anchor_limit: int,
    n_candidates: int,
    max_pairs: int,
    center_xy: Optional[np.ndarray] = None,
    bbox_min_xy: Optional[np.ndarray] = None,
    bbox_max_xy: Optional[np.ndarray] = None,
    min_tangent_parallel_dot: float = 0.75,
    prefer_similar_z: bool = False,
    max_z_pair_difference: Optional[float] = None,
) -> List[BimanualAnchorPair]:
    if len(anchors) < 2:
        return []

    anchors_sorted = sorted(anchors, key=lambda item: item.score, reverse=True)
    xy = np.asarray([anchor.point[:2] for anchor in anchors_sorted], dtype=np.float64)
    if center_xy is None:
        center_xy = xy.mean(axis=0)
    center_xy = np.asarray(center_xy, dtype=np.float64)
    if bbox_min_xy is None:
        bbox_min_xy = np.min(xy, axis=0)
    if bbox_max_xy is None:
        bbox_max_xy = np.max(xy, axis=0)
    bbox_min_xy = np.asarray(bbox_min_xy, dtype=np.float64)
    bbox_max_xy = np.asarray(bbox_max_xy, dtype=np.float64)
    anchor_bbox_min_xy = np.min(xy, axis=0)
    anchor_bbox_max_xy = np.max(xy, axis=0)

    scores = np.asarray([anchor.score for anchor in anchors_sorted], dtype=np.float64)
    tangent_xy = np.asarray([anchor.frame_R[:2, 0] for anchor in anchors_sorted], dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent_xy, axis=1, keepdims=True)
    tangent_xy = tangent_xy / np.where(tangent_norm > 1e-8, tangent_norm, 1.0)
    z_vals = np.asarray([anchor.point[2] for anchor in anchors_sorted], dtype=np.float64)
    local_widths = np.asarray(
        [float(anchor.metadata.get("local_width", 0.0)) for anchor in anchors_sorted],
        dtype=np.float64,
    )

    primary_limit = min(len(anchors_sorted), max(1, int(primary_anchor_limit)))
    n_candidates = max(1, int(n_candidates))
    max_pairs = max(1, int(max_pairs))
    min_tangent_parallel_dot = float(np.clip(min_tangent_parallel_dot, 0.0, 1.0))

    projected_xy = xy.copy()
    projected_along_axis_idx = np.zeros(len(anchors_sorted), dtype=np.int64)
    projected_edge_source: List[str] = []
    for idx in range(len(anchors_sorted)):
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
            projected_edge_source.append("xy_projected_anchor_near_horizontal_edge")
        else:
            projected_along_axis_idx[idx] = 1
            projected_edge_source.append("xy_projected_anchor_near_vertical_edge")

    seen: set[tuple[int, int]] = set()
    pair_list: List[BimanualAnchorPair] = []
    for i in range(primary_limit):
        candidates = np.arange(len(anchors_sorted), dtype=np.int64)
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        edge_axis = tangent_xy[i]
        along_axis_idx = int(projected_along_axis_idx[i])
        edge_family_source = projected_edge_source[i]
        across_axis_idx = 1 - along_axis_idx
        along_axis_name = "x" if along_axis_idx == 0 else "y"
        across_axis_name = "y" if along_axis_idx == 0 else "x"
        along_i = float(projected_xy[i, along_axis_idx])
        across_i = float(projected_xy[i, across_axis_idx])
        across_center = float(center_xy[across_axis_idx])

        candidate_tangent_xy = tangent_xy[candidates]
        candidate_along_axis_idx = projected_along_axis_idx[candidates]
        same_edge_family_mask = candidate_along_axis_idx == along_axis_idx
        if np.any(same_edge_family_mask):
            candidates = candidates[same_edge_family_mask]
            candidate_tangent_xy = candidate_tangent_xy[same_edge_family_mask]
        if len(candidates) == 0:
            continue

        along_candidates = projected_xy[candidates, along_axis_idx]
        across_candidates = projected_xy[candidates, across_axis_idx]
        tangent_alignment = np.abs(tangent_xy[candidates] @ edge_axis)
        z_diff = np.abs(z_vals[candidates] - z_vals[i])
        width_diff = np.abs(local_widths[candidates] - local_widths[i])
        along_diff = np.abs(along_candidates - along_i)
        across_balance = np.abs(
            (across_candidates - across_center) + (across_i - across_center)
        )
        across_separation = np.abs(across_candidates - across_i)
        opposite_across_mask = (
            (across_candidates - across_center) * (across_i - across_center) < 0.0
        )
        if np.any(opposite_across_mask):
            candidates = candidates[opposite_across_mask]
            tangent_alignment = tangent_alignment[opposite_across_mask]
            z_diff = z_diff[opposite_across_mask]
            width_diff = width_diff[opposite_across_mask]
            along_diff = along_diff[opposite_across_mask]
            across_balance = across_balance[opposite_across_mask]
            across_separation = across_separation[opposite_across_mask]
        if np.any(tangent_alignment >= min_tangent_parallel_dot):
            filtered_mask = tangent_alignment >= min_tangent_parallel_dot
            candidates = candidates[filtered_mask]
            tangent_alignment = tangent_alignment[filtered_mask]
            z_diff = z_diff[filtered_mask]
            width_diff = width_diff[filtered_mask]
            along_diff = along_diff[filtered_mask]
            across_balance = across_balance[filtered_mask]
            across_separation = across_separation[filtered_mask]
        if max_z_pair_difference is not None and len(candidates) > 0:
            same_z_mask = z_diff <= float(max_z_pair_difference)
            if np.any(same_z_mask):
                candidates = candidates[same_z_mask]
                tangent_alignment = tangent_alignment[same_z_mask]
                z_diff = z_diff[same_z_mask]
                width_diff = width_diff[same_z_mask]
                along_diff = along_diff[same_z_mask]
                across_balance = across_balance[same_z_mask]
                across_separation = across_separation[same_z_mask]
        if prefer_similar_z:
            rank_order = np.lexsort(
                (
                    -scores[candidates],
                    width_diff,
                    1.0 - tangent_alignment,
                    across_balance,
                    -across_separation,
                    z_diff,
                    along_diff,
                )
            )
        else:
            rank_order = np.lexsort(
                (
                    -scores[candidates],
                    z_diff,
                    width_diff,
                    1.0 - tangent_alignment,
                    across_balance,
                    -across_separation,
                    along_diff,
                )
            )
        ranked = candidates[rank_order[:n_candidates]]
        for j in ranked:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            tangent_parallel_dot = float(np.abs(np.dot(tangent_xy[i], tangent_xy[int(j)])))
            along_j = float(projected_xy[int(j), along_axis_idx])
            across_j = float(projected_xy[int(j), across_axis_idx])
            along_difference = abs(along_j - along_i)
            across_difference = abs(across_j - across_i)
            across_opposite_balance = abs((across_j - across_center) + (across_i - across_center))
            pair_list.append(
                BimanualAnchorPair(
                    primary_anchor=anchors_sorted[i],
                    opposite_anchor=anchors_sorted[int(j)],
                    primary_anchor_rank=int(i),
                    opposite_anchor_rank=int(j),
                    pair_score=float(scores[i] + scores[int(j)]),
                    metadata={
                        "center_xy": [float(x) for x in center_xy.tolist()],
                        "bbox_min_xy": [float(x) for x in bbox_min_xy.tolist()],
                        "bbox_max_xy": [float(x) for x in bbox_max_xy.tolist()],
                        "anchor_bbox_min_xy": [float(x) for x in anchor_bbox_min_xy.tolist()],
                        "anchor_bbox_max_xy": [float(x) for x in anchor_bbox_max_xy.tolist()],
                        "edge_axis_xy": [float(x) for x in edge_axis.tolist()],
                        "primary_projected_xy": [float(x) for x in projected_xy[i].tolist()],
                        "opposite_projected_xy": [float(x) for x in projected_xy[int(j)].tolist()],
                        "edge_family": f"{along_axis_name}_parallel",
                        "edge_family_source": edge_family_source,
                        "along_axis": along_axis_name,
                        "across_axis": across_axis_name,
                        "along_difference": float(along_difference),
                        "across_difference": float(across_difference),
                        "across_opposite_balance": float(across_opposite_balance),
                        "tangent_parallel_dot": tangent_parallel_dot,
                        "local_width_difference": float(abs(local_widths[i] - local_widths[int(j)])),
                        "z_difference": float(abs(z_vals[i] - z_vals[int(j)])),
                        "min_tangent_parallel_dot": min_tangent_parallel_dot,
                    },
                )
            )

    if prefer_similar_z:
        pair_list.sort(
            key=lambda item: (
                int(item.primary_anchor_rank),
                float(item.metadata.get("along_difference", np.inf)),
                float(item.metadata.get("z_difference", np.inf)),
                -float(item.metadata.get("across_difference", 0.0)),
                float(item.metadata.get("across_opposite_balance", np.inf)),
                -float(item.metadata.get("tangent_parallel_dot", 0.0)),
                float(item.metadata.get("local_width_difference", np.inf)),
                -float(item.pair_score),
            )
        )
    else:
        pair_list.sort(
            key=lambda item: (
                int(item.primary_anchor_rank),
                float(item.metadata.get("along_difference", np.inf)),
                -float(item.metadata.get("across_difference", 0.0)),
                float(item.metadata.get("across_opposite_balance", np.inf)),
                -float(item.metadata.get("tangent_parallel_dot", 0.0)),
                float(item.metadata.get("local_width_difference", np.inf)),
                float(item.metadata.get("z_difference", np.inf)),
                -float(item.pair_score),
            )
        )
    return pair_list[:max_pairs]


def _select_cat3_parallel_side_anchor_pairs(
    anchors: List[Anchor],
    *,
    primary_anchor_limit: int,
    n_candidates: int,
    max_pairs: int,
    center_xy: np.ndarray,
    bbox_min_xy: np.ndarray,
    bbox_max_xy: np.ndarray,
    max_z_pair_difference: Optional[float] = None,
) -> List[BimanualAnchorPair]:
    if len(anchors) < 2:
        return []

    anchors_sorted = sorted(anchors, key=lambda item: item.score, reverse=True)
    xy = np.asarray([anchor.point[:2] for anchor in anchors_sorted], dtype=np.float64)
    center_xy = np.asarray(center_xy, dtype=np.float64)
    bbox_min_xy = np.asarray(bbox_min_xy, dtype=np.float64)
    bbox_max_xy = np.asarray(bbox_max_xy, dtype=np.float64)
    scores = np.asarray([anchor.score for anchor in anchors_sorted], dtype=np.float64)
    z_vals = np.asarray([anchor.point[2] for anchor in anchors_sorted], dtype=np.float64)
    normals_xy = np.asarray([anchor.normal[:2] for anchor in anchors_sorted], dtype=np.float64)
    normal_norm = np.linalg.norm(normals_xy, axis=1, keepdims=True)
    normals_xy = normals_xy / np.where(normal_norm > 1e-8, normal_norm, 1.0)
    local_widths = np.asarray(
        [float(anchor.metadata.get("local_width", 0.0)) for anchor in anchors_sorted],
        dtype=np.float64,
    )

    bbox_span_xy = bbox_max_xy - bbox_min_xy
    if float(bbox_span_xy[0]) <= float(bbox_span_xy[1]):
        across_axis_idx = 0
        along_axis_idx = 1
    else:
        across_axis_idx = 1
        along_axis_idx = 0
    across_axis_name = "x" if across_axis_idx == 0 else "y"
    along_axis_name = "y" if across_axis_idx == 0 else "x"
    across_axis_xy = np.array([1.0, 0.0], dtype=np.float64) if across_axis_idx == 0 else np.array([0.0, 1.0], dtype=np.float64)
    along_axis_xy = np.array([1.0, 0.0], dtype=np.float64) if along_axis_idx == 0 else np.array([0.0, 1.0], dtype=np.float64)
    across_coord = xy[:, across_axis_idx] - float(center_xy[across_axis_idx])
    along_coord = xy[:, along_axis_idx]
    side_extent = max(
        0.5 * float(bbox_span_xy[across_axis_idx]),
        float(np.max(np.abs(across_coord))) if len(across_coord) else 0.0,
        1e-6,
    )
    side_band_threshold = 0.35 * side_extent
    # Opposite cat3 anchors should lie on the same local cross-section. A large
    # along-side offset creates diagonal palm directions and unstable squeezes.
    cross_section_along_limit = max(
        0.015,
        min(
            0.090,
            0.18 * float(max(bbox_span_xy[along_axis_idx], 1e-6)),
            0.30 * float(max(bbox_span_xy[across_axis_idx], 1e-6)),
        ),
    )

    primary_limit = min(len(anchors_sorted), max(1, int(primary_anchor_limit)))
    # Cat3 should create a small neighborhood of paired side holds, not a direct
    # farthest-point match. Keep the user-facing candidate count bounded here.
    n_candidates = min(max(1, int(n_candidates)), 5)
    max_pairs = max(1, int(max_pairs))
    seen: set[tuple[int, int]] = set()
    pair_list: List[BimanualAnchorPair] = []

    for i in range(primary_limit):
        candidates = np.arange(len(anchors_sorted), dtype=np.int64)
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue

        primary_side_value = float(across_coord[i])
        if abs(primary_side_value) < side_band_threshold:
            continue
        primary_side_sign = 1.0 if primary_side_value >= 0.0 else -1.0
        candidate_side_values = across_coord[candidates]
        opposite_side_mask = candidate_side_values * primary_side_value < 0.0
        if np.any(opposite_side_mask):
            candidates = candidates[opposite_side_mask]
            candidate_side_values = candidate_side_values[opposite_side_mask]
        if len(candidates) == 0:
            continue

        candidate_side_band_mask = np.abs(candidate_side_values) >= side_band_threshold
        if np.any(candidate_side_band_mask):
            candidates = candidates[candidate_side_band_mask]
            candidate_side_values = candidate_side_values[candidate_side_band_mask]
        if len(candidates) == 0:
            continue

        along_diff = np.abs(along_coord[candidates] - along_coord[i])
        z_diff = np.abs(z_vals[candidates] - z_vals[i])
        across_balance = np.abs(candidate_side_values + primary_side_value)
        across_separation = np.abs(candidate_side_values - primary_side_value)
        width_diff = np.abs(local_widths[candidates] - local_widths[i])
        normal_opposition = -(normals_xy[candidates] @ normals_xy[i])
        finite_normal_mask = np.linalg.norm(normals_xy[candidates], axis=1) > 1e-8
        if np.any(finite_normal_mask & (normal_opposition >= 0.35)):
            normal_mask = finite_normal_mask & (normal_opposition >= 0.35)
            candidates = candidates[normal_mask]
            candidate_side_values = candidate_side_values[normal_mask]
            along_diff = along_diff[normal_mask]
            z_diff = z_diff[normal_mask]
            across_balance = across_balance[normal_mask]
            across_separation = across_separation[normal_mask]
            width_diff = width_diff[normal_mask]
            normal_opposition = normal_opposition[normal_mask]

        same_cross_section_mask = along_diff <= cross_section_along_limit
        if np.any(same_cross_section_mask):
            candidates = candidates[same_cross_section_mask]
            candidate_side_values = candidate_side_values[same_cross_section_mask]
            along_diff = along_diff[same_cross_section_mask]
            z_diff = z_diff[same_cross_section_mask]
            across_balance = across_balance[same_cross_section_mask]
            across_separation = across_separation[same_cross_section_mask]
            width_diff = width_diff[same_cross_section_mask]
            normal_opposition = normal_opposition[same_cross_section_mask]
        else:
            continue

        if max_z_pair_difference is not None and len(candidates) > 0:
            same_z_mask = z_diff <= float(max_z_pair_difference)
            if np.any(same_z_mask):
                candidates = candidates[same_z_mask]
                candidate_side_values = candidate_side_values[same_z_mask]
                along_diff = along_diff[same_z_mask]
                z_diff = z_diff[same_z_mask]
                across_balance = across_balance[same_z_mask]
                across_separation = across_separation[same_z_mask]
                width_diff = width_diff[same_z_mask]
                normal_opposition = normal_opposition[same_z_mask]

        rank_order = np.lexsort(
            (
                -scores[candidates],
                width_diff,
                -normal_opposition,
                across_balance,
                -across_separation,
                z_diff,
                along_diff,
            )
        )
        ranked = candidates[rank_order[:n_candidates]]
        for j in ranked:
            key = (min(i, int(j)), max(i, int(j)))
            if key in seen:
                continue
            seen.add(key)
            xy_distance = float(np.linalg.norm(xy[int(j)] - xy[i]))
            anchor_axis_xy = xy[int(j)] - xy[i]
            anchor_axis = np.array([anchor_axis_xy[0], anchor_axis_xy[1], 0.0], dtype=np.float64)
            anchor_axis_norm = float(np.linalg.norm(anchor_axis))
            if anchor_axis_norm > 1e-8:
                anchor_axis = anchor_axis / anchor_axis_norm
            else:
                anchor_axis = np.array([-primary_side_sign * across_axis_xy[0], -primary_side_sign * across_axis_xy[1], 0.0], dtype=np.float64)
            pair_axis = np.array(
                [
                    -primary_side_sign * across_axis_xy[0],
                    -primary_side_sign * across_axis_xy[1],
                    0.0,
                ],
                dtype=np.float64,
            )
            along_difference = float(abs(along_coord[int(j)] - along_coord[i]))
            across_difference = float(abs(across_coord[int(j)] - across_coord[i]))
            across_opposite_balance = float(abs(across_coord[int(j)] + across_coord[i]))
            normal_opposition_ij = float(-(normals_xy[i] @ normals_xy[int(j)]))
            primary_anchor = replace(
                anchors_sorted[i],
                metadata={
                    **dict(anchors_sorted[i].metadata),
                    "cat3_pair_inward_axis": [float(x) for x in pair_axis.tolist()],
                },
            )
            opposite_anchor = replace(
                anchors_sorted[int(j)],
                metadata={
                    **dict(anchors_sorted[int(j)].metadata),
                    "cat3_pair_inward_axis": [float(x) for x in (-pair_axis).tolist()],
                },
            )
            pair_list.append(
                BimanualAnchorPair(
                    primary_anchor=primary_anchor,
                    opposite_anchor=opposite_anchor,
                    primary_anchor_rank=int(i),
                    opposite_anchor_rank=int(j),
                    pair_score=float(scores[i] + scores[int(j)]),
                    metadata={
                        "pairing_geometry_mode": "cat3_parallel_side_closest",
                        "center_xy": [float(x) for x in center_xy.tolist()],
                        "bbox_min_xy": [float(x) for x in bbox_min_xy.tolist()],
                        "bbox_max_xy": [float(x) for x in bbox_max_xy.tolist()],
                        "primary_projected_xy": [float(x) for x in xy[i].tolist()],
                        "opposite_projected_xy": [float(x) for x in xy[int(j)].tolist()],
                        "cat3_side_family": f"{across_axis_name}_opposing_parallel_sides",
                        "cat3_side_pair_mode": "closest_opposite_side_neighbors",
                        "cat3_opposite_neighbor_limit": int(n_candidates),
                        "cat3_across_axis": across_axis_name,
                        "cat3_along_axis": along_axis_name,
                        "cat3_across_axis_xy": [float(x) for x in across_axis_xy.tolist()],
                        "cat3_along_axis_xy": [float(x) for x in along_axis_xy.tolist()],
                        "cat3_cross_section_along_limit": float(cross_section_along_limit),
                        "cat3_orientation_axis_mode": "fixed_across_side_normal",
                        "primary_side_coord": float(across_coord[i]),
                        "opposite_side_coord": float(across_coord[int(j)]),
                        "primary_side_sign": float(primary_side_sign),
                        "opposite_side_sign": float(1.0 if across_coord[int(j)] >= 0.0 else -1.0),
                        "cat3_pair_axis": [float(x) for x in pair_axis.tolist()],
                        "cat3_pair_anchor_axis": [float(x) for x in anchor_axis.tolist()],
                        "xy_distance": xy_distance,
                        "along_difference": along_difference,
                        "across_difference": across_difference,
                        "across_opposite_balance": across_opposite_balance,
                        "normal_opposition_dot": normal_opposition_ij,
                        "local_width_difference": float(abs(local_widths[i] - local_widths[int(j)])),
                        "z_difference": float(abs(z_vals[i] - z_vals[int(j)])),
                    },
                )
            )

    pair_list.sort(
        key=lambda item: (
            float(item.metadata.get("along_difference", np.inf)),
            float(item.metadata.get("z_difference", np.inf)),
            -float(item.metadata.get("across_difference", 0.0)),
            int(item.primary_anchor_rank),
            float(item.metadata.get("across_opposite_balance", np.inf)),
            -float(item.metadata.get("normal_opposition_dot", 0.0)),
            float(item.metadata.get("local_width_difference", np.inf)),
            -float(item.pair_score),
        )
    )
    return pair_list[:max_pairs]


def _expand_ordered_anchor_pairs(
    anchor_pairs: List[BimanualAnchorPair],
    *,
    include_swapped: bool = True,
) -> tuple[List[BimanualAnchorPair], Dict[int, List[int]]]:
    expanded_pairs: List[BimanualAnchorPair] = []
    ordered_indices_by_geometric: Dict[int, List[int]] = {}

    for geometric_pair_rank, pair in enumerate(anchor_pairs):
        ordered_indices_by_geometric[geometric_pair_rank] = []

        original_meta = dict(pair.metadata)
        original_meta.update(
            {
                "pair_assignment_mode": "original",
                "geometric_pair_rank": int(geometric_pair_rank),
                "geometric_primary_anchor_rank": int(pair.primary_anchor_rank),
                "geometric_opposite_anchor_rank": int(pair.opposite_anchor_rank),
            }
        )
        ordered_indices_by_geometric[geometric_pair_rank].append(len(expanded_pairs))
        expanded_pairs.append(
            BimanualAnchorPair(
                primary_anchor=pair.primary_anchor,
                opposite_anchor=pair.opposite_anchor,
                primary_anchor_rank=int(pair.primary_anchor_rank),
                opposite_anchor_rank=int(pair.opposite_anchor_rank),
                pair_score=float(pair.pair_score),
                metadata=original_meta,
            )
        )

        if include_swapped and int(pair.primary_anchor_rank) != int(pair.opposite_anchor_rank):
            swapped_meta = dict(pair.metadata)
            swapped_meta.update(
                {
                    "pair_assignment_mode": "swapped",
                    "geometric_pair_rank": int(geometric_pair_rank),
                    "geometric_primary_anchor_rank": int(pair.primary_anchor_rank),
                    "geometric_opposite_anchor_rank": int(pair.opposite_anchor_rank),
                }
            )
            ordered_indices_by_geometric[geometric_pair_rank].append(len(expanded_pairs))
            expanded_pairs.append(
                BimanualAnchorPair(
                    primary_anchor=pair.opposite_anchor,
                    opposite_anchor=pair.primary_anchor,
                    primary_anchor_rank=int(pair.opposite_anchor_rank),
                    opposite_anchor_rank=int(pair.primary_anchor_rank),
                    pair_score=float(pair.pair_score),
                    metadata=swapped_meta,
                )
            )

    return expanded_pairs, ordered_indices_by_geometric


def _build_side_bundle(
    *,
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    samples,
    descriptors,
    object_query,
    hand_model,
    seed: int,
    num_seeds_per_contact: Optional[int],
    top_k_seeds_per_contact: int,
    top_k_optimized_per_side: Optional[int],
    contact_cfg_overrides: Optional[Dict[str, Any]],
) -> ContactSeedOptimizationBundle:
    contact_results = resolve_contact_templates_for_anchors(
        runtime_cfg=runtime_cfg,
        anchors=[anchor],
        samples=samples,
        descriptors=descriptors,
        contact_cfg_overrides=contact_cfg_overrides,
    )
    if not contact_results:
        raise RuntimeError("Failed to resolve a contact template for the selected anchor.")

    seed_results = generate_pose_seeds_for_contacts(
        runtime_cfg=runtime_cfg,
        contact_results=contact_results,
        num_seeds_per_contact=num_seeds_per_contact,
        seed=seed,
        semantic_hand_model=hand_model,
    )
    if not seed_results:
        raise RuntimeError("Failed to generate seeds for the selected contact template.")

    contact_result = contact_results[0]
    seed_result = seed_results[0]
    optimize_limit = (
        max(1, int(top_k_optimized_per_side))
        if top_k_optimized_per_side is not None
        else max(1, int(top_k_seeds_per_contact))
    )
    optimized_results = optimize_staged_grasps(
        runtime_cfg=runtime_cfg,
        contact_result=contact_result,
        seeds=seed_result.seeds[:optimize_limit],
        object_query=object_query,
        hand_model=hand_model,
        max_workers=1,
    )
    return ContactSeedOptimizationBundle(
        contact_result=contact_result,
        seed_result=seed_result,
        optimized_results=optimized_results,
    )


def _bundle_cache_key(
    *,
    side: Side,
    anchor_rank: int,
) -> tuple[str, int]:
    return (str(side), int(anchor_rank))


def _cache_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _cache_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _cache_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_cache_jsonable(item) for item in value]
    return value


def _anchor_cache_descriptor(anchor: Anchor) -> Dict[str, Any]:
    support_indices = np.asarray(anchor.support_indices, dtype=np.int64)
    return {
        "category": str(anchor.category),
        "point": np.asarray(anchor.point, dtype=np.float64).round(8).tolist(),
        "normal": np.asarray(anchor.normal, dtype=np.float64).round(8).tolist(),
        "score": round(float(anchor.score), 8),
        "frame_R": np.asarray(anchor.frame_R, dtype=np.float64).round(8).tolist(),
        "support_indices_sha256": hashlib.sha256(support_indices.tobytes()).hexdigest(),
        "metadata": _cache_jsonable(anchor.metadata),
    }


def _file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _side_bundle_cache_file(
    *,
    cache_root: Path,
    side: Side,
    category: Category,
    object_usd: Path,
    hand_dir: Path,
    global_config_dir: Path,
    anchor: Anchor,
    seed: int,
    num_seeds_per_contact: Optional[int],
    top_k_seeds_per_contact: int,
    top_k_optimized_per_side: Optional[int],
    contact_cfg_overrides: Optional[Dict[str, Any]],
    optimizer_cfg_overrides: Optional[Dict[str, Any]],
) -> Path:
    hand_cfg_dir = (hand_dir / "config").resolve()
    signature_paths: List[Path] = []
    signature_paths.extend(sorted(hand_cfg_dir.glob("*.yaml")))
    signature_paths.extend(sorted(global_config_dir.resolve().glob("*.yaml")))
    module_dir = Path(__file__).resolve().parent
    signature_paths.extend(
        [
            module_dir / "contact_resolution.py",
            module_dir / "seed_generation.py",
            module_dir / "optimizer_core.py",
            module_dir / "staged_grasp_pipeline.py",
            module_dir / "pipeline_bimanual.py",
        ]
    )
    signature_paths = [path for path in signature_paths if path.exists()]

    payload = {
        "cache_version": 1,
        "side": str(side),
        "category": str(category),
        "object_usd": str(object_usd.resolve()),
        "object_usd_signature": _file_signature(object_usd.resolve()),
        "hand_dir": str(hand_dir.resolve()),
        "global_config_dir": str(global_config_dir.resolve()),
        "anchor": _anchor_cache_descriptor(anchor),
        "seed": int(seed),
        "num_seeds_per_contact": None if num_seeds_per_contact is None else int(num_seeds_per_contact),
        "top_k_seeds_per_contact": int(top_k_seeds_per_contact),
        "top_k_optimized_per_side": (
            None if top_k_optimized_per_side is None else int(top_k_optimized_per_side)
        ),
        "contact_cfg_overrides": _cache_jsonable(contact_cfg_overrides or {}),
        "optimizer_cfg_overrides": _cache_jsonable(optimizer_cfg_overrides or {}),
        "signatures": [_file_signature(path) for path in sorted(signature_paths)],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return cache_root / str(category) / str(side) / f"{digest}.pkl"


def _load_cached_side_bundle(cache_file: Path) -> Optional[ContactSeedOptimizationBundle]:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, ContactSeedOptimizationBundle):
            return bundle
    except Exception:
        return None
    return None


def _save_cached_side_bundle(cache_file: Path, bundle: ContactSeedOptimizationBundle) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_file)


def _hand_pose_aabb(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model,
    hand_pose,
    *,
    padding: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    spheres = hand_model.collision_spheres_world(runtime_cfg, hand_pose)
    if not spheres:
        center = np.asarray(hand_pose.wrist_position, dtype=np.float64)
        return center.copy(), center.copy()
    padding = float(max(0.0, padding))
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for sphere in spheres:
        radius = float(sphere.radius) + padding
        center = np.asarray(sphere.center_world, dtype=np.float64)
        mins.append(center - radius)
        maxs.append(center + radius)
    return np.min(np.asarray(mins, dtype=np.float64), axis=0), np.max(np.asarray(maxs, dtype=np.float64), axis=0)


def _hand_pose_spheres(
    runtime_cfg: ResolvedHandRuntimeConfig,
    hand_model,
    hand_pose,
):
    return hand_model.collision_spheres_world(runtime_cfg, hand_pose)


def _aabb_overlap_metadata(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> Dict[str, Any]:
    min_a = np.asarray(min_a, dtype=np.float64)
    max_a = np.asarray(max_a, dtype=np.float64)
    min_b = np.asarray(min_b, dtype=np.float64)
    max_b = np.asarray(max_b, dtype=np.float64)
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    positive_overlap = np.maximum(overlap, 0.0)
    separation = np.maximum(-overlap, 0.0)
    intersects = bool(np.all(overlap > 0.0))
    return {
        "intersects": intersects,
        "overlap_xyz": [float(x) for x in positive_overlap.tolist()],
        "overlap_volume": float(np.prod(positive_overlap)),
        "separation_xyz": [float(x) for x in separation.tolist()],
        "separation_norm": float(np.linalg.norm(separation)),
    }


def _sphere_overlap_metadata(
    primary_spheres,
    opposite_spheres,
    *,
    sphere_padding: float = 0.0,
) -> Dict[str, Any]:
    sphere_padding = float(sphere_padding)
    min_signed_distance = np.inf
    max_penetration = 0.0
    overlap_count = 0

    if not primary_spheres or not opposite_spheres:
        return {
            "intersects": False,
            "min_signed_distance": float("inf"),
            "max_penetration": 0.0,
            "num_overlapping_pairs": 0,
            "sphere_padding": sphere_padding,
        }

    for primary_sphere in primary_spheres:
        primary_center = np.asarray(primary_sphere.center_world, dtype=np.float64)
        primary_radius = float(primary_sphere.radius) + sphere_padding
        for opposite_sphere in opposite_spheres:
            opposite_center = np.asarray(opposite_sphere.center_world, dtype=np.float64)
            opposite_radius = float(opposite_sphere.radius) + sphere_padding
            center_distance = float(np.linalg.norm(primary_center - opposite_center))
            signed_distance = center_distance - (primary_radius + opposite_radius)
            min_signed_distance = min(min_signed_distance, signed_distance)
            if signed_distance < 0.0:
                overlap_count += 1
                max_penetration = max(max_penetration, -signed_distance)

    if not np.isfinite(min_signed_distance):
        min_signed_distance = float("inf")

    return {
        "intersects": bool(overlap_count > 0),
        "min_signed_distance": float(min_signed_distance),
        "max_penetration": float(max_penetration),
        "num_overlapping_pairs": int(overlap_count),
        "sphere_padding": sphere_padding,
    }


def _safe_normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return np.zeros_like(arr)
    return arr / norm


def _semantic_world_direction(runtime_cfg: ResolvedHandRuntimeConfig, hand_pose, local_axis: Any) -> np.ndarray:
    local_dir = _safe_normalize_vector(np.asarray(local_axis, dtype=np.float64))
    if np.linalg.norm(local_dir) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return _safe_normalize_vector(hand_pose.rotation_matrix() @ local_dir)


def _evaluate_hand_pair_semantic_alignment(
    *,
    primary_runtime_cfg: ResolvedHandRuntimeConfig,
    primary_result,
    opposite_runtime_cfg: ResolvedHandRuntimeConfig,
    opposite_result,
    stage_name: str,
) -> Dict[str, Any]:
    primary_pose = primary_result.stage_results[stage_name].hand_pose
    opposite_pose = opposite_result.stage_results[stage_name].hand_pose
    primary_frame = primary_runtime_cfg.hand.frame_convention
    opposite_frame = opposite_runtime_cfg.hand.frame_convention

    primary_finger = _semantic_world_direction(
        primary_runtime_cfg,
        primary_pose,
        primary_frame.finger_forward_local,
    )
    opposite_finger = _semantic_world_direction(
        opposite_runtime_cfg,
        opposite_pose,
        opposite_frame.finger_forward_local,
    )
    primary_palm = _semantic_world_direction(
        primary_runtime_cfg,
        primary_pose,
        primary_frame.palm_normal_local,
    )
    opposite_palm = _semantic_world_direction(
        opposite_runtime_cfg,
        opposite_pose,
        opposite_frame.palm_normal_local,
    )

    finger_dot = float(np.dot(primary_finger, opposite_finger))
    palm_dot = float(np.dot(primary_palm, opposite_palm))
    return {
        "stage_name": str(stage_name),
        "primary_finger_forward_world": [float(x) for x in primary_finger.tolist()],
        "opposite_finger_forward_world": [float(x) for x in opposite_finger.tolist()],
        "finger_forward_dot": finger_dot,
        "primary_palm_normal_world": [float(x) for x in primary_palm.tolist()],
        "opposite_palm_normal_world": [float(x) for x in opposite_palm.tolist()],
        "palm_normal_dot": palm_dot,
    }


def _evaluate_hand_pair_overlap(
    *,
    primary_runtime_cfg: ResolvedHandRuntimeConfig,
    primary_hand_model,
    primary_result,
    opposite_runtime_cfg: ResolvedHandRuntimeConfig,
    opposite_hand_model,
    opposite_result,
    stage_name: str = "squeeze",
    aabb_padding: float = 0.003,
    sphere_padding: float = 0.0,
) -> Dict[str, Any]:
    primary_stage = primary_result.stage_results[stage_name]
    opposite_stage = opposite_result.stage_results[stage_name]
    primary_pose = primary_stage.hand_pose
    opposite_pose = opposite_stage.hand_pose
    primary_spheres = _hand_pose_spheres(primary_runtime_cfg, primary_hand_model, primary_pose)
    opposite_spheres = _hand_pose_spheres(opposite_runtime_cfg, opposite_hand_model, opposite_pose)
    primary_min, primary_max = _hand_pose_aabb(
        primary_runtime_cfg,
        primary_hand_model,
        primary_pose,
        padding=aabb_padding,
    )
    opposite_min, opposite_max = _hand_pose_aabb(
        opposite_runtime_cfg,
        opposite_hand_model,
        opposite_pose,
        padding=aabb_padding,
    )
    aabb_overlap = _aabb_overlap_metadata(primary_min, primary_max, opposite_min, opposite_max)
    sphere_overlap = _sphere_overlap_metadata(
        primary_spheres,
        opposite_spheres,
        sphere_padding=sphere_padding,
    )
    overlap_info = dict(aabb_overlap)
    overlap_info.update(
        {
            "intersects": bool(aabb_overlap["intersects"]),
            "aabb_intersects": bool(aabb_overlap["intersects"]),
            "sphere_intersects": bool(sphere_overlap["intersects"]),
            "sphere_overlap": sphere_overlap,
            "overlap_filter_mode": "aabb",
            "stage_name": str(stage_name),
            "aabb_padding": float(aabb_padding),
            "primary_aabb_min": [float(x) for x in primary_min.tolist()],
            "primary_aabb_max": [float(x) for x in primary_max.tolist()],
            "opposite_aabb_min": [float(x) for x in opposite_min.tolist()],
            "opposite_aabb_max": [float(x) for x in opposite_max.tolist()],
            "num_primary_spheres": int(len(primary_spheres)),
            "num_opposite_spheres": int(len(opposite_spheres)),
            "wrist_distance": float(
                np.linalg.norm(
                    np.asarray(primary_pose.wrist_position, dtype=np.float64)
                    - np.asarray(opposite_pose.wrist_position, dtype=np.float64)
                )
            ),
        }
    )
    return overlap_info


def _select_valid_result_pairs(
    *,
    evaluated_result_pairs: List[Dict[str, Any]],
    min_wrist_distance: float = 0.10,
    require_both_success: bool = False,
    require_both_stage_success: bool = False,
    min_finger_forward_dot: Optional[float] = None,
    max_palm_normal_dot: Optional[float] = None,
    max_global_penetration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    def passes_semantic_alignment(item: Dict[str, Any]) -> bool:
        alignment = item.get("semantic_alignment") or {}
        if min_finger_forward_dot is not None:
            if float(alignment.get("finger_forward_dot", -1.0)) < float(min_finger_forward_dot):
                return False
        if max_palm_normal_dot is not None:
            if float(alignment.get("palm_normal_dot", 1.0)) > float(max_palm_normal_dot):
                return False
        return True

    def passes_global_penetration(item: Dict[str, Any]) -> bool:
        if max_global_penetration is None:
            return True
        return float(item.get("max_global_penetration", np.inf)) <= float(max_global_penetration)

    valid_pairs = [
        item
        for item in evaluated_result_pairs
        if (
            not bool(item.get("hand_overlap", {}).get("intersects", False))
            and float(item.get("hand_overlap", {}).get("wrist_distance", 0.0)) >= float(min_wrist_distance)
            and (not bool(require_both_success) or bool(item.get("both_success", False)))
            and (
                not bool(require_both_stage_success)
                or bool(item.get("both_stage_success", item.get("both_success", False)))
            )
            and passes_semantic_alignment(item)
            and passes_global_penetration(item)
        )
    ]
    valid_pairs.sort(
        key=lambda item: (
            0 if item["both_success"] else 1,
            -int(item["num_successful_hands"]),
            -float((item.get("semantic_alignment") or {}).get("finger_forward_dot", 0.0)),
            float((item.get("semantic_alignment") or {}).get("palm_normal_dot", 1.0)),
            float(item.get("max_global_penetration", np.inf)),
            float(item["total_cost"]),
            -float(item["hand_overlap"].get("wrist_distance", 0.0)),
            -float(item["hand_overlap"].get("separation_norm", 0.0)),
        )
    )
    return valid_pairs


def _enumerate_result_pairs(
    *,
    primary_side: Side,
    primary_runtime_cfg: ResolvedHandRuntimeConfig,
    primary_hand_model,
    primary_results,
    opposite_side: Side,
    opposite_runtime_cfg: ResolvedHandRuntimeConfig,
    opposite_hand_model,
    opposite_results,
    stage_name: str = "squeeze",
    aabb_padding: float = 0.003,
    sphere_padding: float = 0.0,
    max_result_pair_checks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    result_pairs: List[Dict[str, Any]] = []
    candidate_indices: List[tuple[int, int]] = [
        (primary_idx, opposite_idx)
        for primary_idx in range(len(primary_results))
        for opposite_idx in range(len(opposite_results))
    ]
    candidate_indices.sort(key=lambda item: (item[0] + item[1], item[0], item[1]))
    if max_result_pair_checks is not None:
        candidate_indices = candidate_indices[: max(1, int(max_result_pair_checks))]

    for primary_idx, opposite_idx in candidate_indices:
        primary_result = primary_results[primary_idx]
        opposite_result = opposite_results[opposite_idx]
        overlap_info = _evaluate_hand_pair_overlap(
            primary_runtime_cfg=primary_runtime_cfg,
            primary_hand_model=primary_hand_model,
            primary_result=primary_result,
            opposite_runtime_cfg=opposite_runtime_cfg,
            opposite_hand_model=opposite_hand_model,
            opposite_result=opposite_result,
            stage_name=stage_name,
            aabb_padding=aabb_padding,
            sphere_padding=sphere_padding,
        )
        semantic_alignment = None
        if str(primary_result.category).lower() == "cat3" or str(opposite_result.category).lower() == "cat3":
            semantic_alignment = _evaluate_hand_pair_semantic_alignment(
                primary_runtime_cfg=primary_runtime_cfg,
                primary_result=primary_result,
                opposite_runtime_cfg=opposite_runtime_cfg,
                opposite_result=opposite_result,
                stage_name=stage_name,
            )
        primary_stage_success = bool(primary_result.stage_results[stage_name].success)
        opposite_stage_success = bool(opposite_result.stage_results[stage_name].success)
        primary_completion = dict(
            primary_result.stage_results[stage_name].metadata.get("completion", {})
        )
        opposite_completion = dict(
            opposite_result.stage_results[stage_name].metadata.get("completion", {})
        )
        max_pair_global_penetration = max(
            float(primary_completion.get("max_global_penetration", 0.0)),
            float(opposite_completion.get("max_global_penetration", 0.0)),
        )
        result_pairs.append(
            {
                "result_rank_by_side": {
                    primary_side: int(primary_idx),
                    opposite_side: int(opposite_idx),
                },
                "total_cost": float(primary_result.total_cost + opposite_result.total_cost),
                "num_successful_hands": int(bool(primary_result.success)) + int(bool(opposite_result.success)),
                "both_success": bool(primary_result.success and opposite_result.success),
                "num_successful_stage_hands": int(primary_stage_success) + int(opposite_stage_success),
                "both_stage_success": bool(primary_stage_success and opposite_stage_success),
                "stage_completion": {
                    primary_side: primary_completion,
                    opposite_side: opposite_completion,
                },
                "max_global_penetration": float(max_pair_global_penetration),
                "hand_overlap": overlap_info,
                "semantic_alignment": semantic_alignment,
            }
        )
    result_pairs.sort(
        key=lambda item: (
            0 if not bool(item["hand_overlap"].get("intersects", False)) else 1,
            0 if item["both_success"] else 1,
            -int(item["num_successful_hands"]),
            -float((item.get("semantic_alignment") or {}).get("finger_forward_dot", 0.0)),
            float((item.get("semantic_alignment") or {}).get("palm_normal_dot", 1.0)),
            float(item.get("max_global_penetration", np.inf)),
            float(item["total_cost"]),
            float(item["hand_overlap"].get("overlap_volume", 0.0)),
            -float(item["hand_overlap"].get("separation_norm", 0.0)),
        )
    )
    return result_pairs


def run_bimanual_staged(
    *,
    hand_dir: Path,
    global_config_dir: Path,
    primary_side: Side,
    category: Category,
    object_usd: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
    num_surface_points: int = 100000,
    seed: int = 0,
    top_k_anchors: int = 5,
    num_opposite_candidates: int = 5,
    top_k_anchor_pairs: Optional[int] = None,
    num_seeds_per_contact: Optional[int] = None,
    top_k_seeds_per_contact: int = 3,
    top_k_optimized_per_side: Optional[int] = None,
    max_workers: Optional[int] = None,
    sdf_cache_path: Optional[Path] = None,
    force_rebuild_sdf: bool = False,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
    optimizer_cfg_overrides: Optional[Dict[str, Any]] = None,
    pair_selection_mode: str = "ranked",
    max_result_pair_checks: Optional[int] = None,
    parallel_backend: str = "thread",
) -> BimanualStagedPipelineResult:
    hand_dir = hand_dir.resolve()
    global_config_dir = global_config_dir.resolve()
    object_usd = object_usd.resolve()
    proposal_result = run_single_hand_region_proposal(
        hand_dir=hand_dir,
        global_config_dir=global_config_dir,
        side=primary_side,
        category=category,
        object_usd=object_usd,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        num_surface_points=num_surface_points,
        seed=seed,
    )

    primary_runtime_cfg = _runtime_cfg_with_optimizer_overrides(
        proposal_result.runtime_cfg,
        optimizer_cfg_overrides,
    )
    proposal_result.runtime_cfg = primary_runtime_cfg
    opposite_side = _opposite_side(primary_side)
    opposite_runtime_cfg = _runtime_cfg_with_optimizer_overrides(
        load_all_for_hand(
        hand_dir=hand_dir.resolve(),
        global_config_dir=global_config_dir.resolve(),
        side=opposite_side,
        category=category,
        ),
        optimizer_cfg_overrides,
    )

    optimizer_global_cfg = primary_runtime_cfg.optimizer_cfg.get("global", {})
    sdf_cfg = optimizer_global_cfg.get("sdf", {})
    voxel_size = float(sdf_cfg.get("voxel_size", 0.005))
    padding_voxels = int(sdf_cfg.get("padding_voxels", 8))
    force_rebuild_sdf = bool(force_rebuild_sdf or sdf_cfg.get("force_rebuild", False))

    sdf = load_or_build_sdf_from_usd(
        usd_path=object_usd,
        cache_path=sdf_cache_path,
        voxel_size=voxel_size,
        padding_voxels=padding_voxels,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        force_rebuild=force_rebuild_sdf,
    )
    effective_sdf_cache_path = Path(
        sdf.metadata.get(
            "cache_path",
            str(sdf_cache_path.resolve() if sdf_cache_path is not None else default_sdf_cache_path(object_usd)),
        )
    ).resolve()
    object_query = make_object_query_from_sdf(sdf)
    hand_model_by_side = {
        primary_side: load_hand_kinematics_model(primary_runtime_cfg),
        opposite_side: load_hand_kinematics_model(opposite_runtime_cfg),
    }

    pair_bundles: List[BimanualContactSeedOptimizationBundle] = []
    evaluated_pair_bundles: List[BimanualContactSeedOptimizationBundle] = []
    pair_limit = (
        top_k_anchor_pairs
        if top_k_anchor_pairs is not None
        else (
            min(8, max(1, int(top_k_anchors)) * max(1, int(num_opposite_candidates)))
            if category == "cat3"
            else max(1, int(top_k_anchors)) * max(1, int(num_opposite_candidates))
        )
    )
    anchor_pool_limit = max(1, int(top_k_anchors))
    if category == "cat3":
        anchor_pool_limit = len(proposal_result.anchors)
    anchor_pool = list(proposal_result.anchors[:anchor_pool_limit])
    side_bundle_cache: Dict[tuple[str, int], ContactSeedOptimizationBundle] = {}
    side_bundle_cache_hits = 0
    side_bundle_cache_misses = 0
    persistent_cache_hits = 0
    persistent_cache_misses = 0
    persistent_cache_root = global_config_dir.parent / ".cache" / "bimanual_side_bundles"
    object_center_xy = 0.5 * (
        np.asarray(proposal_result.samples.bbox_min[:2], dtype=np.float64)
        + np.asarray(proposal_result.samples.bbox_max[:2], dtype=np.float64)
    )
    cat2_line_distribution = (
        _cat2_line_distribution_from_anchors(anchor_pool)
        if category == "cat2"
        else None
    )
    if cat2_line_distribution is not None:
        all_geometric_anchor_pairs = _select_cat2_line_symmetric_pairs(
            anchor_pool,
            primary_anchor_limit=len(anchor_pool),
            n_candidates=num_opposite_candidates,
            max_pairs=max(1, len(anchor_pool) * max(1, int(num_opposite_candidates))),
            line_center_xy=np.asarray(cat2_line_distribution["center_xy"], dtype=np.float64),
            line_axis_xy=np.asarray(cat2_line_distribution["line_axis_xy"], dtype=np.float64),
            line_side_axis_xy=np.asarray(cat2_line_distribution["side_axis_xy"], dtype=np.float64),
        )
    elif category == "cat3":
        object_height = float(proposal_result.samples.bbox_max[2] - proposal_result.samples.bbox_min[2])
        cat3_z_pair_tolerance = float(np.clip(0.08 * object_height, 0.012, 0.04))
        all_geometric_anchor_pairs = _select_cat3_parallel_side_anchor_pairs(
            anchor_pool,
            primary_anchor_limit=len(anchor_pool),
            n_candidates=num_opposite_candidates,
            max_pairs=max(1, len(anchor_pool) * max(1, int(num_opposite_candidates))),
            center_xy=object_center_xy,
            bbox_min_xy=np.asarray(proposal_result.samples.bbox_min[:2], dtype=np.float64),
            bbox_max_xy=np.asarray(proposal_result.samples.bbox_max[:2], dtype=np.float64),
            max_z_pair_difference=cat3_z_pair_tolerance,
        )
    else:
        all_geometric_anchor_pairs = _select_opposite_anchor_pairs(
            anchor_pool,
            primary_anchor_limit=len(anchor_pool),
            n_candidates=num_opposite_candidates,
            max_pairs=max(1, len(anchor_pool) * max(1, int(num_opposite_candidates))),
            center_xy=object_center_xy,
            bbox_min_xy=np.asarray(proposal_result.samples.bbox_min[:2], dtype=np.float64),
            bbox_max_xy=np.asarray(proposal_result.samples.bbox_max[:2], dtype=np.float64),
            prefer_similar_z=(category == "cat1"),
            max_z_pair_difference=None,
        )
    all_anchor_pairs, ordered_indices_by_geometric = _expand_ordered_anchor_pairs(
        all_geometric_anchor_pairs,
        include_swapped=True,
    )
    if pair_selection_mode == "farthest_raw":
        if category == "cat3":
            selected_geometric_pair_indices = sorted(
                range(len(all_geometric_anchor_pairs)),
                key=lambda idx: (
                    float(all_geometric_anchor_pairs[idx].metadata.get("along_difference", np.inf)),
                    float(all_geometric_anchor_pairs[idx].metadata.get("z_difference", np.inf)),
                    -float(all_geometric_anchor_pairs[idx].metadata.get("across_difference", 0.0)),
                    float(all_geometric_anchor_pairs[idx].metadata.get("across_opposite_balance", np.inf)),
                    -float(all_geometric_anchor_pairs[idx].metadata.get("normal_opposition_dot", 0.0)),
                ),
            )[: max(1, int(pair_limit))]
        else:
            selected_geometric_pair_indices = sorted(
                range(len(all_geometric_anchor_pairs)),
                key=lambda idx: float(all_geometric_anchor_pairs[idx].metadata.get("across_difference", 0.0)),
                reverse=True,
            )[: max(1, int(pair_limit))]
    else:
        selected_geometric_pair_indices = list(
            range(min(len(all_geometric_anchor_pairs), max(1, int(pair_limit))))
        )
    selected_pair_indices: List[int] = []
    for geometric_pair_idx in selected_geometric_pair_indices:
        selected_pair_indices.extend(ordered_indices_by_geometric.get(int(geometric_pair_idx), []))
    anchor_pairs = [(int(pair_idx), all_anchor_pairs[int(pair_idx)]) for pair_idx in selected_pair_indices]
    total_side_bundle_requests = 2 * len(anchor_pairs)

    unique_side_tasks: Dict[tuple[str, int], Dict[str, Any]] = {}
    for pair_rank, anchor_pair in anchor_pairs:
        for side, pair_anchor, pair_anchor_rank, seed_offset in (
            (primary_side, anchor_pair.primary_anchor, anchor_pair.primary_anchor_rank, 0),
            (opposite_side, anchor_pair.opposite_anchor, anchor_pair.opposite_anchor_rank, 10_000),
        ):
            cache_anchor_rank = (
                int(pair_rank) * 100_000 + int(pair_anchor_rank)
                if category == "cat3"
                else int(pair_anchor_rank)
            )
            cache_key = _bundle_cache_key(side=side, anchor_rank=cache_anchor_rank)
            if cache_key in unique_side_tasks:
                continue
            bundle_seed = seed + seed_offset + pair_rank
            unique_side_tasks[cache_key] = {
                "cache_key": cache_key,
                "side": side,
                "runtime_cfg": primary_runtime_cfg if side == primary_side else opposite_runtime_cfg,
                "anchor": pair_anchor,
                "seed": int(bundle_seed),
                "cache_file": _side_bundle_cache_file(
                    cache_root=persistent_cache_root,
                    side=side,
                    category=category,
                    object_usd=object_usd,
                    hand_dir=hand_dir,
                    global_config_dir=global_config_dir,
                    anchor=pair_anchor,
                    seed=bundle_seed,
                    num_seeds_per_contact=num_seeds_per_contact,
                    top_k_seeds_per_contact=top_k_seeds_per_contact,
                    top_k_optimized_per_side=top_k_optimized_per_side,
                    contact_cfg_overrides=contact_cfg_overrides,
                    optimizer_cfg_overrides=optimizer_cfg_overrides,
                ),
            }

    num_unique_side_bundles = len(unique_side_tasks)
    pending_side_tasks: List[Dict[str, Any]] = []
    for cache_key, task in unique_side_tasks.items():
        cached_bundle = _load_cached_side_bundle(task["cache_file"])
        if cached_bundle is not None:
            persistent_cache_hits += 1
            side_bundle_cache[cache_key] = cached_bundle
        else:
            persistent_cache_misses += 1
            pending_side_tasks.append(task)

    requested_max_workers = 0 if max_workers is None else int(max_workers)
    requested_parallel_backend = str(parallel_backend).lower()
    if pending_side_tasks:
        if requested_max_workers <= 0:
            effective_max_workers = min(len(pending_side_tasks), max(1, os.cpu_count() or 1), 4)
        else:
            effective_max_workers = min(len(pending_side_tasks), max(1, requested_max_workers))
    else:
        effective_max_workers = 0

    def _compute_side_task(task: Dict[str, Any]) -> tuple[tuple[str, int], ContactSeedOptimizationBundle]:
        side = task["side"]
        bundle = _build_side_bundle(
            runtime_cfg=task["runtime_cfg"],
            anchor=task["anchor"],
            samples=proposal_result.samples,
            descriptors=proposal_result.descriptors,
            object_query=object_query,
            hand_model=hand_model_by_side[side],
            seed=task["seed"],
            num_seeds_per_contact=num_seeds_per_contact,
            top_k_seeds_per_contact=top_k_seeds_per_contact,
            top_k_optimized_per_side=top_k_optimized_per_side,
            contact_cfg_overrides=contact_cfg_overrides,
        )
        return task["cache_key"], bundle

    effective_parallel_backend = "serial" if effective_max_workers <= 1 else requested_parallel_backend
    parallel_backend_fallback_reason = None
    effective_process_start_method = None

    if pending_side_tasks:
        if effective_max_workers <= 1:
            for task in pending_side_tasks:
                cache_key, side_bundle = _compute_side_task(task)
                _save_cached_side_bundle(task["cache_file"], side_bundle)
                side_bundle_cache[cache_key] = side_bundle
        else:
            if requested_parallel_backend == "process":
                try:
                    start_methods = mp.get_all_start_methods()
                    start_method = "fork" if "fork" in start_methods else "spawn"
                    effective_process_start_method = start_method
                    mp_context = mp.get_context(start_method)
                    with ProcessPoolExecutor(
                        max_workers=effective_max_workers,
                        mp_context=mp_context,
                        initializer=_init_process_side_bundle_worker,
                        initargs=(
                            proposal_result.samples,
                            proposal_result.descriptors,
                            object_query,
                            hand_model_by_side,
                            num_seeds_per_contact,
                            top_k_seeds_per_contact,
                            top_k_optimized_per_side,
                            contact_cfg_overrides,
                        ),
                    ) as executor:
                        future_to_task = {
                            executor.submit(_compute_side_task_process, task): task
                            for task in pending_side_tasks
                        }
                        for future in as_completed(future_to_task):
                            task = future_to_task[future]
                            cache_key, side_bundle = future.result()
                            _save_cached_side_bundle(task["cache_file"], side_bundle)
                            side_bundle_cache[cache_key] = side_bundle
                except Exception as exc:
                    effective_parallel_backend = "thread"
                    parallel_backend_fallback_reason = f"{type(exc).__name__}: {exc}"
                    with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
                        future_to_task = {
                            executor.submit(_compute_side_task, task): task
                            for task in pending_side_tasks
                        }
                        for future in as_completed(future_to_task):
                            task = future_to_task[future]
                            cache_key, side_bundle = future.result()
                            _save_cached_side_bundle(task["cache_file"], side_bundle)
                            side_bundle_cache[cache_key] = side_bundle
            else:
                with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
                    future_to_task = {
                        executor.submit(_compute_side_task, task): task
                        for task in pending_side_tasks
                    }
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        cache_key, side_bundle = future.result()
                        _save_cached_side_bundle(task["cache_file"], side_bundle)
                        side_bundle_cache[cache_key] = side_bundle

    side_bundle_cache_hits = max(0, total_side_bundle_requests - num_unique_side_bundles)
    side_bundle_cache_misses = max(0, num_unique_side_bundles - persistent_cache_hits)

    num_pairs_before_overlap_filter = 0
    num_pairs_rejected_by_overlap = 0
    candidate_pairs_debug: List[Dict[str, Any]] = [
        {
            "pair_rank": int(pair_rank),
            "pair_score": float(anchor_pair.pair_score),
            "primary_anchor_rank": int(anchor_pair.primary_anchor_rank),
            "opposite_anchor_rank": int(anchor_pair.opposite_anchor_rank),
            "pair_metadata": anchor_pair.metadata,
            "primary_point": [float(x) for x in anchor_pair.primary_anchor.point.tolist()],
            "opposite_point": [float(x) for x in anchor_pair.opposite_anchor.point.tolist()],
            "pair_evaluated": False,
            "passed_overlap_filter": None,
            "num_evaluated_result_pairs": 0,
            "num_valid_result_pairs": 0,
            "best_result_pair": None,
            "selected_valid_result_pair": None,
        }
        for pair_rank, anchor_pair in enumerate(all_anchor_pairs)
    ]

    for pair_rank, anchor_pair in anchor_pairs:
        num_pairs_before_overlap_filter += 1
        side_bundles: Dict[Side, ContactSeedOptimizationBundle] = {}
        for side, _pair_anchor, pair_anchor_rank, _seed_offset in (
            (primary_side, anchor_pair.primary_anchor, anchor_pair.primary_anchor_rank, 0),
            (opposite_side, anchor_pair.opposite_anchor, anchor_pair.opposite_anchor_rank, 10_000),
        ):
            cache_anchor_rank = (
                int(pair_rank) * 100_000 + int(pair_anchor_rank)
                if category == "cat3"
                else int(pair_anchor_rank)
            )
            cache_key = _bundle_cache_key(side=side, anchor_rank=cache_anchor_rank)
            side_bundle = side_bundle_cache.get(cache_key)
            if side_bundle is None:
                raise RuntimeError(f"Missing side bundle for {cache_key}.")
            side_bundles[side] = side_bundle

        if len(side_bundles) != 2:
            continue

        primary_bundle = side_bundles[primary_side]
        opposite_bundle = side_bundles[opposite_side]
        evaluated_result_pairs: List[Dict[str, Any]] = []
        valid_result_pairs: List[Dict[str, Any]] = []
        overlap_metadata: Optional[Dict[str, Any]] = None
        pair_evaluation_stage = "grasp" if category == "cat3" else "squeeze"
        if primary_bundle.optimized_results and opposite_bundle.optimized_results:
            evaluated_result_pairs = _enumerate_result_pairs(
                primary_side=primary_side,
                primary_runtime_cfg=primary_runtime_cfg,
                primary_hand_model=hand_model_by_side[primary_side],
                primary_results=primary_bundle.optimized_results,
                opposite_side=opposite_side,
                opposite_runtime_cfg=opposite_runtime_cfg,
                opposite_hand_model=hand_model_by_side[opposite_side],
                opposite_results=opposite_bundle.optimized_results,
                stage_name=pair_evaluation_stage,
                max_result_pair_checks=max_result_pair_checks,
            )
            valid_result_pairs = _select_valid_result_pairs(
                evaluated_result_pairs=evaluated_result_pairs,
                min_wrist_distance=0.10,
                min_finger_forward_dot=0.85 if category == "cat3" else None,
                max_palm_normal_dot=-0.85 if category == "cat3" else None,
                max_global_penetration=0.00025 if category == "cat3" else None,
            )
            if valid_result_pairs:
                overlap_metadata = valid_result_pairs[0]["hand_overlap"]
        best_result_pair = evaluated_result_pairs[0] if evaluated_result_pairs else None
        candidate_pairs_debug[pair_rank].update(
            {
                "pair_evaluated": True,
                "passed_overlap_filter": bool(valid_result_pairs),
                "num_evaluated_result_pairs": int(len(evaluated_result_pairs)),
                "num_valid_result_pairs": int(len(valid_result_pairs)),
                "best_result_pair": best_result_pair,
                "selected_valid_result_pair": valid_result_pairs[0] if valid_result_pairs else None,
            }
        )
        selected_result_pair = valid_result_pairs[0] if valid_result_pairs else best_result_pair
        evaluated_pair_bundles.append(
            BimanualContactSeedOptimizationBundle(
                primary_side=primary_side,
                opposite_side=opposite_side,
                anchor_pair=anchor_pair,
                bundles_by_side=side_bundles,
                metadata={
                    "pair_rank": int(pair_rank),
                    "hand_overlap": selected_result_pair["hand_overlap"] if selected_result_pair is not None else None,
                    "passes_overlap_filter": bool(valid_result_pairs),
                    "selected_result_pair_rank": 0,
                    "pair_evaluation_stage": pair_evaluation_stage,
                    "selected_result_rank_by_side": (
                        selected_result_pair["result_rank_by_side"] if selected_result_pair is not None else {}
                    ),
                    "best_result_pair": best_result_pair,
                    "valid_result_pairs": valid_result_pairs,
                },
            )
        )
        if not valid_result_pairs:
            num_pairs_rejected_by_overlap += 1
            continue

        pair_bundles.append(
            BimanualContactSeedOptimizationBundle(
                primary_side=primary_side,
                opposite_side=opposite_side,
                anchor_pair=anchor_pair,
                bundles_by_side=side_bundles,
                metadata={
                    "pair_rank": int(pair_rank),
                    "hand_overlap": overlap_metadata,
                    "selected_result_pair_rank": 0,
                    "pair_evaluation_stage": pair_evaluation_stage,
                    "selected_result_rank_by_side": valid_result_pairs[0]["result_rank_by_side"],
                    "valid_result_pairs": valid_result_pairs,
                },
            )
        )

    return BimanualStagedPipelineResult(
        proposal_result=proposal_result,
        runtime_cfg_by_side={
            primary_side: primary_runtime_cfg,
            opposite_side: opposite_runtime_cfg,
        },
        pair_bundles=pair_bundles,
        evaluated_pair_bundles=evaluated_pair_bundles,
        sdf_cache_path=effective_sdf_cache_path,
        metadata={
            "category": category,
            "primary_side": primary_side,
            "opposite_side": opposite_side,
            "voxel_size": voxel_size,
            "padding_voxels": padding_voxels,
            "force_rebuild_sdf": force_rebuild_sdf,
            "top_k_anchors": int(top_k_anchors),
            "num_anchors_in_pair_pool": int(len(anchor_pool)),
            "num_opposite_candidates": int(num_opposite_candidates),
            "top_k_anchor_pairs": int(pair_limit),
            "top_k_seeds_per_contact": int(top_k_seeds_per_contact),
            "top_k_optimized_per_side": (
                None if top_k_optimized_per_side is None else int(top_k_optimized_per_side)
            ),
            "optimizer_cfg_overrides": _cache_jsonable(optimizer_cfg_overrides or {}),
            "max_result_pair_checks": (
                None if max_result_pair_checks is None else int(max_result_pair_checks)
            ),
            "pairing_center_xy": [float(x) for x in object_center_xy.tolist()],
            "cat2_anchor_distribution_mode": (
                cat2_line_distribution.get("mode")
                if cat2_line_distribution is not None
                else "default"
            ),
            "ordered_pair_assignments_enabled": True,
            "num_pairs_before_overlap_filter": int(num_pairs_before_overlap_filter),
            "num_pairs_after_overlap_filter": int(len(pair_bundles)),
            "num_pairs_rejected_by_overlap": int(num_pairs_rejected_by_overlap),
            "num_candidate_pairs_total": int(len(all_anchor_pairs)),
            "num_geometric_candidate_pairs_total": int(len(all_geometric_anchor_pairs)),
            "num_pairs_selected_for_optimization": int(len(anchor_pairs)),
            "num_geometric_pairs_selected": int(len(selected_geometric_pair_indices)),
            "pair_selection_mode": pair_selection_mode,
            "pair_evaluation_stage": ("grasp" if category == "cat3" else "squeeze"),
            "pair_limit_effective": int(pair_limit),
            "side_bundle_cache": {
                "num_entries": int(len(side_bundle_cache)),
                "num_hits": int(side_bundle_cache_hits),
                "num_misses": int(side_bundle_cache_misses),
            },
            "parallelism": {
                "requested_max_workers": int(requested_max_workers),
                "effective_max_workers": int(effective_max_workers),
                "requested_parallel_backend": requested_parallel_backend,
                "effective_parallel_backend": effective_parallel_backend,
                "effective_process_start_method": effective_process_start_method,
                "parallel_backend_fallback_reason": parallel_backend_fallback_reason,
                "num_unique_side_bundles": int(num_unique_side_bundles),
                "num_total_side_bundle_requests": int(total_side_bundle_requests),
            },
            "persistent_side_bundle_cache": {
                "cache_root": str(persistent_cache_root),
                "num_hits": int(persistent_cache_hits),
                "num_misses": int(persistent_cache_misses),
            },
            "candidate_pairs_debug": candidate_pairs_debug,
        },
    )


def run_bimanual_staged_cat1(
    *,
    hand_dir: Path,
    global_config_dir: Path,
    primary_side: Side,
    object_usd: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
    num_surface_points: int = 100000,
    seed: int = 0,
    top_k_anchors: int = 5,
    num_opposite_candidates: int = 5,
    top_k_anchor_pairs: Optional[int] = None,
    num_seeds_per_contact: Optional[int] = None,
    top_k_seeds_per_contact: int = 3,
    top_k_optimized_per_side: Optional[int] = None,
    max_workers: Optional[int] = None,
    sdf_cache_path: Optional[Path] = None,
    force_rebuild_sdf: bool = False,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
    pair_selection_mode: str = "ranked",
    max_result_pair_checks: Optional[int] = None,
) -> BimanualStagedPipelineResult:
    return run_bimanual_staged(
        hand_dir=hand_dir,
        global_config_dir=global_config_dir,
        primary_side=primary_side,
        category="cat1",
        object_usd=object_usd,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        num_surface_points=num_surface_points,
        seed=seed,
        top_k_anchors=top_k_anchors,
        num_opposite_candidates=num_opposite_candidates,
        top_k_anchor_pairs=top_k_anchor_pairs,
        num_seeds_per_contact=num_seeds_per_contact,
        top_k_seeds_per_contact=top_k_seeds_per_contact,
        top_k_optimized_per_side=top_k_optimized_per_side,
        max_workers=max_workers,
        sdf_cache_path=sdf_cache_path,
        force_rebuild_sdf=force_rebuild_sdf,
        contact_cfg_overrides=contact_cfg_overrides,
        pair_selection_mode=pair_selection_mode,
        max_result_pair_checks=max_result_pair_checks,
    )


def summarize_bimanual_staged_pipeline(
    result: BimanualStagedPipelineResult,
    *,
    top_k_optimized_per_contact: int = 1,
) -> Dict[str, Any]:
    primary_side = result.metadata.get("primary_side", result.proposal_result.side)
    opposite_side = result.metadata.get("opposite_side", _opposite_side(result.proposal_result.side))

    pair_summaries: List[Dict[str, Any]] = []
    for bundle in result.pair_bundles:
        valid_result_pairs = list(bundle.metadata.get("valid_result_pairs", []))
        selected_result_pair_rank = int(bundle.metadata.get("selected_result_pair_rank", 0))
        selected_result_pair = (
            valid_result_pairs[max(0, min(selected_result_pair_rank, len(valid_result_pairs) - 1))]
            if valid_result_pairs
            else None
        )
        hands_summary: Dict[str, Any] = {}
        for side in (primary_side, opposite_side):
            hand_bundle = bundle.bundles_by_side[side]
            preferred_indices: List[int] = []
            if selected_result_pair is not None:
                preferred_indices.append(int(selected_result_pair["result_rank_by_side"][side]))
            preferred_indices.extend(
                idx
                for idx in range(len(hand_bundle.optimized_results))
                if idx not in preferred_indices
            )
            hands_summary[side] = {
                "contact": summarize_contact_resolution(hand_bundle.contact_result),
                "seed_generation": summarize_seed_generation(
                    hand_bundle.seed_result,
                    top_k=top_k_optimized_per_contact,
                ),
                "optimized": [
                    summarize_staged_grasp_result(hand_bundle.optimized_results[idx])
                    for idx in preferred_indices[: max(1, int(top_k_optimized_per_contact))]
                ],
            }

        pair_summaries.append(
            {
                "pair_rank": int(bundle.metadata.get("pair_rank", len(pair_summaries))),
                "pair_score": float(bundle.anchor_pair.pair_score),
                "primary_anchor_rank": int(bundle.anchor_pair.primary_anchor_rank),
                "opposite_anchor_rank": int(bundle.anchor_pair.opposite_anchor_rank),
                "pair_metadata": bundle.anchor_pair.metadata,
                "bundle_metadata": bundle.metadata,
                "selected_result_pair": selected_result_pair,
                "anchors": {
                    primary_side: {
                        "score": float(bundle.anchor_pair.primary_anchor.score),
                        "point": [float(x) for x in bundle.anchor_pair.primary_anchor.point.tolist()],
                        "normal": [float(x) for x in bundle.anchor_pair.primary_anchor.normal.tolist()],
                        "metadata": bundle.anchor_pair.primary_anchor.metadata,
                    },
                    opposite_side: {
                        "score": float(bundle.anchor_pair.opposite_anchor.score),
                        "point": [float(x) for x in bundle.anchor_pair.opposite_anchor.point.tolist()],
                        "normal": [float(x) for x in bundle.anchor_pair.opposite_anchor.normal.tolist()],
                        "metadata": bundle.anchor_pair.opposite_anchor.metadata,
                    },
                },
                "hands": hands_summary,
            }
        )

    return {
        "category": result.proposal_result.category,
        "object_usd": str(result.proposal_result.object_usd),
        "primary_side": primary_side,
        "opposite_side": opposite_side,
        "num_anchors_total": int(len(result.proposal_result.anchors)),
        "num_anchor_pairs": int(len(result.pair_bundles)),
        "sdf_cache_path": str(result.sdf_cache_path),
        "metadata": result.metadata,
        "candidate_pairs": result.metadata.get("candidate_pairs_debug", []),
        "pairs": pair_summaries,
    }
