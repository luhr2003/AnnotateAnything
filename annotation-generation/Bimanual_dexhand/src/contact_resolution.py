from __future__ import annotations

"""
contact_resolution.py

Current role:
- Convert one object-side anchor into an optimization-ready semantic contact
  template.
- Assign active semantic contact points to rough target regions on the object.
- Provide a local contact frame, opposition structure, and stage clearances for
  later seed generation and optimization.

Intentional current simplifications:
- Surface targets are selected from sampled surface points, not exact mesh
  nearest-point queries and not SDF-backed contact queries.
- Palm handling defaults to collision-only behavior; palm avoid targets are
  optional soft biases, not part of the default contact template.
- Category contact layouts are still encoded in Python heuristics rather than
  fully externalized in config.

Future work:
- Replace sample-based target picking with mesh / SDF query backends while
  keeping the output dataclasses stable.
- Move more category-specific finger-role / side-assignment rules into config so
  the same logic transfers more easily across hand models.
- Continue tuning how optional ring / little contacts are weighted once more
  hand variants are exercised.
- Tune per-category offsets, weights, and stage clearances from empirical grasp
  results.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from src.types_config import (
    CollisionSphere,
    ResolvedHandRuntimeConfig,
    SemanticPoint,
)

if TYPE_CHECKING:
    from src.region_proposal import Anchor, SurfaceSamples


TargetRole = Literal["active", "avoid"]


@dataclass
class ContactTarget:
    name: str
    role: TargetRole
    source_link: str
    source_sphere_index: int
    source_sphere: CollisionSphere
    role_tags: List[str]

    target_point: np.ndarray
    target_normal: np.ndarray
    reference_point: np.ndarray

    desired_clearance_pregrasp: float
    desired_clearance_grasp: float
    desired_clearance_squeeze: float
    weight: float

    candidate_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OppositionConstraint:
    point_a: str
    point_b: str
    desired_axis_world: np.ndarray
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactResolutionResult:
    category: str
    contact_template: str
    anchor: Anchor
    frame_R: np.ndarray
    object_center: np.ndarray

    active_targets: List[ContactTarget]
    avoid_targets: List[ContactTarget]
    opposition_constraints: List[OppositionConstraint]

    patch_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _TargetPreference:
    name: str
    role: TargetRole
    preferred_local_point: np.ndarray
    preferred_local_normal: np.ndarray
    desired_clearances: Tuple[float, float, float]
    weight: float

    prefer_side_surface: bool = False
    prefer_top_surface: bool = False
    prefer_bottom_surface: bool = False
    prefer_horizontal_normal: bool = False

    side_axis: Optional[int] = None
    side_sign: float = 0.0
    side_bias_weight: float = 0.5
    side_bias_scale: float = 0.01
    min_local_separation: float = 0.0
    search_radius_scale: float = 1.0


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _project_to_xy(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).copy()
    if out.shape[0] >= 3:
        out[2] = 0.0
    return out


def _ensure_right_handed_frame(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    x_axis = _safe_normalize(x_axis)
    y_axis = _safe_normalize(y_axis)
    z_axis = _safe_normalize(z_axis)

    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    y_axis = _safe_normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = _safe_normalize(np.cross(y_axis, z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _lookup_semantic_source(
    runtime_cfg: ResolvedHandRuntimeConfig,
    point_name: str,
) -> tuple[SemanticPoint, CollisionSphere]:
    semantic_point = runtime_cfg.semantic_points[point_name]
    sphere_list = runtime_cfg.collision.spheres_by_link[semantic_point.source_link]
    sphere = sphere_list[semantic_point.source_sphere_index]
    return semantic_point, sphere


def _build_patch_indices(
    anchor: Anchor,
    samples: SurfaceSamples,
    radius: float,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    support = np.asarray(anchor.support_indices, dtype=np.int64)
    if cKDTree is None or len(samples.points) == 0:
        return np.unique(support)

    tree = cKDTree(samples.points)
    ball_idx = tree.query_ball_point(anchor.point, r=radius)
    if len(ball_idx) == 0:
        return np.unique(support)
    merged = np.concatenate(
        [support, np.asarray(ball_idx, dtype=np.int64)],
        axis=0,
    )
    return np.unique(merged)


def _build_cat1_rim_band_indices(
    anchor: Anchor,
    samples: SurfaceSamples,
    frame_R: np.ndarray,
    patch_radius: float,
) -> np.ndarray:
    local_points = (samples.points - anchor.point[None, :]) @ frame_R
    local_normals = samples.normals @ frame_R

    local_width = float(anchor.metadata.get("local_width", max(0.02, 0.6 * patch_radius)))
    object_diag = float(np.linalg.norm(samples.bbox_max - samples.bbox_min))

    x_half_extent = max(2.75 * patch_radius, 1.4 * local_width, 0.08 * object_diag)
    y_half_extent = max(1.30 * local_width, 0.80 * patch_radius, 0.018)
    z_upper = max(0.15 * patch_radius, 0.010)
    z_lower = max(0.90 * local_width, 1.65 * patch_radius, 0.020)

    band_mask = (
        (np.abs(local_points[:, 0]) <= x_half_extent)
        & (np.abs(local_points[:, 1]) <= y_half_extent)
        & (local_points[:, 2] <= z_upper)
        & (local_points[:, 2] >= -z_lower)
    )

    # Keep side-wall and near-rim top candidates, rather than a tiny ball around the anchor.
    normal_mask = (np.abs(local_normals[:, 1]) >= 0.20) | (local_normals[:, 2] >= 0.35)
    band_indices = np.flatnonzero(band_mask & normal_mask).astype(np.int64)
    if len(band_indices) == 0:
        band_indices = np.flatnonzero(band_mask).astype(np.int64)

    base_indices = _build_patch_indices(anchor, samples, radius=patch_radius)
    if len(band_indices) == 0:
        return base_indices
    if len(base_indices) == 0:
        return np.unique(band_indices)
    return np.unique(np.concatenate([base_indices, band_indices], axis=0))


def _estimate_active_contact_span(runtime_cfg: ResolvedHandRuntimeConfig) -> float:
    return _estimate_active_contact_span_for_names(runtime_cfg, runtime_cfg.contact_usage.active_points)


def _estimate_active_contact_span_for_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    point_names: Sequence[str],
) -> float:
    radii: List[float] = []
    for point_name, _ in _available_usage_points(
        runtime_cfg,
        point_names,
    )[0]:
        _, sphere = _lookup_semantic_source(runtime_cfg, point_name)
        radii.append(float(sphere.radius))

    if len(radii) == 0:
        return 0.02

    radii_arr = np.asarray(radii, dtype=np.float64)
    return float(max(4.0 * radii_arr.mean(), 2.5 * radii_arr.max()))


def _estimate_support_patch_scale(anchor: Anchor, samples: SurfaceSamples) -> float:
    support = np.asarray(anchor.support_indices, dtype=np.int64)
    if len(support) == 0:
        return 0.0

    support_points = samples.points[support]
    d = np.linalg.norm(support_points - anchor.point[None, :], axis=1)
    if len(d) == 0:
        return 0.0
    return float(np.percentile(d, 85))


def _estimate_patch_radius(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    samples: SurfaceSamples,
    overrides: Dict[str, Any],
) -> tuple[float, Dict[str, float | str]]:
    active_point_names_for_span: List[str] = list(runtime_cfg.contact_usage.active_points)
    if anchor.category == "cat1":
        pad_point_names, _ = _cat1_mode_point_names(runtime_cfg, "pad")
        tip_point_names, _ = _cat1_mode_point_names(runtime_cfg, "tip")
        active_point_names_for_span = sorted(set(pad_point_names + tip_point_names))
    elif anchor.category == "cat2":
        active_point_names_for_span, _ = _cat2_mode_point_names(runtime_cfg)
    elif anchor.category == "cat4":
        active_point_names_for_span, _ = _cat4_mode_point_names(runtime_cfg)

    active_contact_span = _estimate_active_contact_span_for_names(
        runtime_cfg,
        active_point_names_for_span,
    )
    if "patch_radius" in overrides:
        patch_radius = float(overrides["patch_radius"])
        return patch_radius, {
            "patch_radius_source": "override",
            "active_contact_span": float(active_contact_span),
            "feature_scale": float(anchor.metadata.get("local_width", 0.0) or 0.0),
            "support_scale": float(_estimate_support_patch_scale(anchor, samples)),
        }

    object_diag = float(np.linalg.norm(samples.bbox_max - samples.bbox_min))
    support_scale = _estimate_support_patch_scale(anchor, samples)

    feature_scale_candidates: List[float] = []
    for key in ("local_width", "grasp_width", "feature_width", "radius_estimate"):
        value = anchor.metadata.get(key, None)
        if value is None:
            continue
        value_f = float(value)
        if value_f > 0.0:
            feature_scale_candidates.append(value_f)

    if support_scale > 0.0:
        feature_scale_candidates.append(2.0 * support_scale)

    feature_scale = max(feature_scale_candidates) if feature_scale_candidates else 0.0

    lower_bound = max(0.008, 1.5 * active_contact_span / 4.0, 1.5 * max(0.0, support_scale))
    nominal = max(1.1 * active_contact_span, 0.6 * feature_scale)
    upper_bound = max(0.04, 2.5 * active_contact_span, 0.20 * object_diag)

    patch_radius = float(np.clip(nominal, lower_bound, upper_bound))
    return patch_radius, {
        "patch_radius_source": "local_grasp_scale",
        "active_contact_span": float(active_contact_span),
        "feature_scale": float(feature_scale),
        "support_scale": float(support_scale),
    }


def _semantic_point_tags(semantic_point: SemanticPoint, point_name: str) -> set[str]:
    tags = set(semantic_point.role_tags)
    tokens = point_name.replace("-", "_").split("_")
    tags.update(tokens)
    return {t.lower() for t in tags}


def _digit_rank(semantic_point: SemanticPoint, point_name: str) -> int:
    tags = _semantic_point_tags(semantic_point, point_name)
    for i, digit in enumerate(("index", "middle", "ring", "little")):
        if digit in tags:
            return i
    return 99


def _is_thumb_point(semantic_point: SemanticPoint, point_name: str) -> bool:
    tags = _semantic_point_tags(semantic_point, point_name)
    return "thumb" in tags


def _declared_tip_links(runtime_cfg: ResolvedHandRuntimeConfig) -> set[str]:
    tips = runtime_cfg.hand.links.tips
    return {
        link_name
        for link_name in (tips.thumb, tips.index, tips.middle, tips.ring, tips.little)
        if str(link_name)
    }


def _is_declared_tip_semantic_point(
    runtime_cfg: ResolvedHandRuntimeConfig,
    semantic_point: SemanticPoint,
    point_name: str,
) -> bool:
    if semantic_point.source_link in _declared_tip_links(runtime_cfg):
        return True
    return "tip" in _semantic_point_tags(semantic_point, point_name)


def _is_palm_point(semantic_point: SemanticPoint, point_name: str) -> bool:
    tags = _semantic_point_tags(semantic_point, point_name)
    return "palm" in tags


def _available_usage_points(
    runtime_cfg: ResolvedHandRuntimeConfig,
    point_names: Sequence[str],
) -> tuple[List[tuple[str, SemanticPoint]], List[str]]:
    available: List[tuple[str, SemanticPoint]] = []
    missing: List[str] = []

    for point_name in point_names:
        semantic_point = runtime_cfg.semantic_points.get(point_name)
        if semantic_point is None:
            missing.append(point_name)
            continue
        available.append((point_name, semantic_point))

    return available, missing


def _available_named_points(
    runtime_cfg: ResolvedHandRuntimeConfig,
    point_names: Sequence[str],
) -> List[str]:
    return [point_name for point_name in point_names if point_name in runtime_cfg.semantic_points]


def _sort_thumb_points(
    points: Sequence[tuple[str, SemanticPoint]],
    *,
    prefer_pad: bool,
) -> List[tuple[str, SemanticPoint]]:
    primary_tag = "pad" if prefer_pad else "tip"
    secondary_tag = "tip" if prefer_pad else "pad"
    return sorted(
        points,
        key=lambda item: (
            0 if primary_tag in _semantic_point_tags(item[1], item[0]) else 1,
            0 if secondary_tag in _semantic_point_tags(item[1], item[0]) else 1,
            item[0],
        ),
    )


def _split_active_thumb_and_finger_points(
    runtime_cfg: ResolvedHandRuntimeConfig,
    *,
    prefer_thumb_pad: bool,
) -> tuple[List[tuple[str, SemanticPoint]], List[tuple[str, SemanticPoint]], List[str]]:
    return _split_named_thumb_and_finger_points(
        runtime_cfg,
        runtime_cfg.contact_usage.active_points,
        prefer_thumb_pad=prefer_thumb_pad,
    )


def _split_named_thumb_and_finger_points(
    runtime_cfg: ResolvedHandRuntimeConfig,
    point_names: Sequence[str],
    *,
    prefer_thumb_pad: bool,
) -> tuple[List[tuple[str, SemanticPoint]], List[tuple[str, SemanticPoint]], List[str]]:
    available_points, missing_points = _available_usage_points(
        runtime_cfg,
        point_names,
    )

    thumb_points: List[tuple[str, SemanticPoint]] = []
    finger_points: List[tuple[str, SemanticPoint]] = []
    for point_name, semantic_point in available_points:
        if _is_thumb_point(semantic_point, point_name):
            thumb_points.append((point_name, semantic_point))
        else:
            finger_points.append((point_name, semantic_point))

    thumb_points = _sort_thumb_points(thumb_points, prefer_pad=prefer_thumb_pad)
    finger_points.sort(key=lambda item: (_digit_rank(item[1], item[0]), item[0]))
    return thumb_points, finger_points, missing_points


def _finger_priority_weight(digit_rank: int) -> float:
    return 1.0 if digit_rank < 99 else 0.9


def _estimate_contact_span_for_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    point_names: Sequence[str],
) -> float:
    radii: List[float] = []
    for point_name in _available_named_points(runtime_cfg, point_names):
        _, sphere = _lookup_semantic_source(runtime_cfg, point_name)
        radii.append(float(sphere.radius))

    if len(radii) == 0:
        return _estimate_active_contact_span(runtime_cfg)

    radii_arr = np.asarray(radii, dtype=np.float64)
    return float(max(4.0 * radii_arr.mean(), 2.5 * radii_arr.max()))


def _cat1_candidate_points(
    runtime_cfg: ResolvedHandRuntimeConfig,
) -> List[tuple[str, SemanticPoint]]:
    candidates: List[tuple[str, SemanticPoint]] = []
    for point_name, semantic_point in runtime_cfg.semantic_points.items():
        tags = _semantic_point_tags(semantic_point, point_name)
        if "contact" not in tags or "avoid" in tags or "palm" in tags:
            continue
        if _is_thumb_point(semantic_point, point_name) or _digit_rank(semantic_point, point_name) < 99:
            candidates.append((point_name, semantic_point))
    return candidates


def _cat1_point_preference_key(
    point_name: str,
    semantic_point: SemanticPoint,
    *,
    mode: str,
    is_thumb: bool,
) -> tuple[int, int, int, str]:
    tags = _semantic_point_tags(semantic_point, point_name)
    if mode == "pad":
        preferred_order = ("pad", "tip", "side") if is_thumb else ("pad", "side", "tip")
    else:
        preferred_order = ("tip", "pad", "side")

    surface_rank = len(preferred_order)
    for idx, tag in enumerate(preferred_order):
        if tag in tags:
            surface_rank = idx
            break

    main_rank = 0 if "main" in tags else 1
    contact_rank = 0 if "contact" in tags else 1
    return (surface_rank, main_rank, contact_rank, point_name)


def _legacy_cat1_mode_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    mode: str,
) -> tuple[List[str], List[tuple[str, str]]]:
    candidates = _cat1_candidate_points(runtime_cfg)
    thumb_candidates = [
        (point_name, semantic_point)
        for point_name, semantic_point in candidates
        if _is_thumb_point(semantic_point, point_name)
    ]
    finger_candidates = [
        (point_name, semantic_point)
        for point_name, semantic_point in candidates
        if not _is_thumb_point(semantic_point, point_name)
    ]

    selected_names: List[str] = []
    if thumb_candidates:
        thumb_candidates.sort(
            key=lambda item: _cat1_point_preference_key(
                item[0],
                item[1],
                mode=mode,
                is_thumb=True,
            )
        )
        selected_names.append(thumb_candidates[0][0])

    used_digits: set[int] = set()
    for point_name, semantic_point in sorted(
        finger_candidates,
        key=lambda item: _cat1_point_preference_key(
            item[0],
            item[1],
            mode=mode,
            is_thumb=False,
        ),
    ):
        digit_rank = _digit_rank(semantic_point, point_name)
        if digit_rank < 99:
            if digit_rank in used_digits:
                continue
            used_digits.add(digit_rank)
        selected_names.append(point_name)

    opposition_pairs: List[tuple[str, str]] = []
    if len(selected_names) >= 2:
        thumb_name = selected_names[0]
        for finger_name in selected_names[1:]:
            opposition_pairs.append((thumb_name, finger_name))
    return selected_names, opposition_pairs


def _cat2_point_preference_key(
    point_name: str,
    semantic_point: SemanticPoint,
    *,
    is_thumb: bool,
) -> tuple[int, int, int, str]:
    tags = _semantic_point_tags(semantic_point, point_name)
    if is_thumb:
        preferred_order = ("pad", "tip", "side")
    else:
        preferred_order = ("pad", "tip", "side")

    surface_rank = len(preferred_order)
    for idx, tag in enumerate(preferred_order):
        if tag in tags:
            surface_rank = idx
            break

    main_rank = 0 if "main" in tags else 1
    contact_rank = 0 if "contact" in tags else 1
    return (surface_rank, main_rank, contact_rank, point_name)


def _cat2_mode_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
) -> tuple[List[str], List[tuple[str, str]]]:
    candidates = _cat1_candidate_points(runtime_cfg)
    thumb_candidates = [
        (point_name, semantic_point)
        for point_name, semantic_point in candidates
        if _is_thumb_point(semantic_point, point_name)
    ]
    finger_candidates: Dict[int, List[tuple[str, SemanticPoint]]] = {}
    for point_name, semantic_point in candidates:
        if _is_thumb_point(semantic_point, point_name):
            continue
        digit_rank = _digit_rank(semantic_point, point_name)
        if digit_rank >= 99:
            continue
        finger_candidates.setdefault(digit_rank, []).append((point_name, semantic_point))

    selected_names: List[str] = []
    opposition_pairs: List[tuple[str, str]] = []

    thumb_name: Optional[str] = None
    if thumb_candidates:
        thumb_candidates.sort(
            key=lambda item: _cat2_point_preference_key(
                item[0],
                item[1],
                is_thumb=True,
            )
        )
        thumb_name = thumb_candidates[0][0]
        selected_names.append(thumb_name)

    for digit_rank in sorted(finger_candidates):
        digit_points = sorted(
            finger_candidates[digit_rank],
            key=lambda item: _cat2_point_preference_key(
                item[0],
                item[1],
                is_thumb=False,
            ),
        )
        chosen_for_digit: List[str] = []
        seen_kinds: set[str] = set()
        for point_name, semantic_point in digit_points:
            tags = _semantic_point_tags(semantic_point, point_name)
            kind = "other"
            if "pad" in tags:
                kind = "pad"
            elif "tip" in tags:
                kind = "tip"
            elif "side" in tags:
                kind = "side"
            if kind in seen_kinds:
                continue
            chosen_for_digit.append(point_name)
            seen_kinds.add(kind)
            if {"pad", "tip"} <= seen_kinds:
                break
        if not chosen_for_digit and digit_points:
            chosen_for_digit.append(digit_points[0][0])
        selected_names.extend(chosen_for_digit)
        if thumb_name is not None and chosen_for_digit:
            opposition_pairs.append((thumb_name, chosen_for_digit[0]))

    return selected_names, opposition_pairs


def _cat4_candidate_points(
    runtime_cfg: ResolvedHandRuntimeConfig,
) -> List[tuple[str, SemanticPoint]]:
    candidates: List[tuple[str, SemanticPoint]] = []
    for point_name, semantic_point in runtime_cfg.semantic_points.items():
        tags = _semantic_point_tags(semantic_point, point_name)
        if "contact" not in tags or "avoid" in tags:
            continue
        if _is_thumb_point(semantic_point, point_name):
            candidates.append((point_name, semantic_point))
        elif _is_palm_point(semantic_point, point_name):
            candidates.append((point_name, semantic_point))
        elif _digit_rank(semantic_point, point_name) < 99:
            candidates.append((point_name, semantic_point))
    return candidates


def _cat4_mode_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
) -> tuple[List[str], List[tuple[str, str]]]:
    candidates = _cat4_candidate_points(runtime_cfg)
    palm_candidates = [
        (point_name, semantic_point)
        for point_name, semantic_point in candidates
        if _is_palm_point(semantic_point, point_name)
    ]
    thumb_candidates = [
        (point_name, semantic_point)
        for point_name, semantic_point in candidates
        if _is_thumb_point(semantic_point, point_name)
    ]
    finger_candidates: Dict[int, List[tuple[str, SemanticPoint]]] = {}
    for point_name, semantic_point in candidates:
        if _is_thumb_point(semantic_point, point_name) or _is_palm_point(semantic_point, point_name):
            continue
        digit_rank = _digit_rank(semantic_point, point_name)
        if digit_rank >= 99:
            continue
        finger_candidates.setdefault(digit_rank, []).append((point_name, semantic_point))

    selected_names: List[str] = []
    opposition_pairs: List[tuple[str, str]] = []

    if palm_candidates:
        palm_candidates.sort(
            key=lambda item: (
                0 if "support" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
                item[0],
            )
        )
        selected_names.append(palm_candidates[0][0])

    thumb_name: Optional[str] = None
    if thumb_candidates:
        thumb_candidates.sort(
            key=lambda item: (
                0 if "pad" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "tip" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
                item[0],
            )
        )
        thumb_name = thumb_candidates[0][0]
        selected_names.append(thumb_name)

    for digit_rank in sorted(finger_candidates):
        digit_points = sorted(
            finger_candidates[digit_rank],
            key=lambda item: (
                0 if "pad" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "tip" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "side" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
                item[0],
            ),
        )
        chosen_for_digit: List[str] = []
        seen_kinds: set[str] = set()
        for point_name, semantic_point in digit_points:
            tags = _semantic_point_tags(semantic_point, point_name)
            kind = "other"
            if "pad" in tags:
                kind = "pad"
            elif "tip" in tags:
                kind = "tip"
            elif "side" in tags:
                kind = "side"
            if kind in seen_kinds:
                continue
            chosen_for_digit.append(point_name)
            seen_kinds.add(kind)
            if {"pad", "tip"} <= seen_kinds:
                break
        if not chosen_for_digit and digit_points:
            chosen_for_digit.append(digit_points[0][0])
        selected_names.extend(chosen_for_digit)
        if thumb_name is not None and chosen_for_digit:
            finger_opp_name = next(
                (
                    point_name
                    for point_name, semantic_point in digit_points
                    if point_name in chosen_for_digit
                    and "tip" in _semantic_point_tags(semantic_point, point_name)
                ),
                chosen_for_digit[0],
            )
            opposition_pairs.append((thumb_name, finger_opp_name))

    return selected_names, opposition_pairs


def _cat1_mode_point_names(
    runtime_cfg: ResolvedHandRuntimeConfig,
    mode: str,
) -> tuple[List[str], List[tuple[str, str]]]:
    candidates = _cat1_candidate_points(runtime_cfg)
    thumb_candidates = [
        (point_name, semantic_point)
        for point_name, semantic_point in candidates
        if _is_thumb_point(semantic_point, point_name)
    ]
    finger_candidates: Dict[int, List[tuple[str, SemanticPoint]]] = {}
    for point_name, semantic_point in candidates:
        if _is_thumb_point(semantic_point, point_name):
            continue
        digit_rank = _digit_rank(semantic_point, point_name)
        if digit_rank >= 99:
            continue
        finger_candidates.setdefault(digit_rank, []).append((point_name, semantic_point))

    selected_names: List[str] = []
    if thumb_candidates:
        thumb_candidates.sort(
            key=lambda item: _cat1_point_preference_key(
                item[0],
                item[1],
                mode=mode,
                is_thumb=True,
            )
        )
        selected_names.append(thumb_candidates[0][0])

    for digit_rank in sorted(finger_candidates):
        digit_points = sorted(
            finger_candidates[digit_rank],
            key=lambda item: _cat1_point_preference_key(
                item[0],
                item[1],
                mode=mode,
                is_thumb=False,
            ),
        )
        if digit_points:
            selected_names.append(digit_points[0][0])

    if len(selected_names) < 2:
        return _legacy_cat1_mode_point_names(runtime_cfg, mode)

    opposition_pairs: List[tuple[str, str]] = []
    if len(selected_names) >= 2:
        thumb_name = selected_names[0]
        for finger_name in selected_names[1:]:
            opposition_pairs.append((thumb_name, finger_name))
    return selected_names, opposition_pairs


def _select_cat1_contact_mode(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    patch_radius: float,
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    contact_params = dict(runtime_cfg.category_cfg.contact_logic.params)
    contact_params.update(dict(overrides))

    explicit_mode = str(contact_params.get("cat1_contact_mode", "auto")).lower()
    pad_point_names, pad_pairs = _cat1_mode_point_names(runtime_cfg, "pad")
    tip_point_names, tip_pairs = _cat1_mode_point_names(runtime_cfg, "tip")

    local_width = float(anchor.metadata.get("local_width", max(0.02, 0.6 * patch_radius)))
    pad_contact_span = _estimate_contact_span_for_point_names(runtime_cfg, pad_point_names)
    tip_contact_span = _estimate_contact_span_for_point_names(runtime_cfg, tip_point_names)
    width_to_pad_span = local_width / max(pad_contact_span, 1e-8)
    pad_thumb_count = sum(
        1
        for point_name in pad_point_names
        if (
            runtime_cfg.semantic_points.get(point_name) is not None
            and _is_thumb_point(runtime_cfg.semantic_points[point_name], point_name)
        )
    )
    pad_finger_count = max(0, len(pad_point_names) - pad_thumb_count)

    pad_min_width_ratio = float(contact_params.get("pad_contact_min_width_ratio", 0.55))
    pad_min_local_width = float(contact_params.get("pad_contact_min_local_width", 0.024))

    if explicit_mode in {"tip", "pad"}:
        mode = explicit_mode
        mode_source = "override"
    else:
        use_pad_mode = (
            pad_thumb_count >= 1
            and pad_finger_count >= 1
            and local_width >= pad_min_local_width
            and width_to_pad_span >= pad_min_width_ratio
        )
        mode = "pad" if use_pad_mode else "tip"
        mode_source = "adaptive_local_width"

    active_point_names, opposition_pairs = (
        (pad_point_names, pad_pairs) if mode == "pad" else (tip_point_names, tip_pairs)
    )
    if mode == "pad":
        seed_posture_names = ["light_close", "rim_pad_close"]
        close_posture_name = "rim_pad_close"
        limit_close_blend = 0.35
        seed_palm_depth_scale = 1.0
        seed_pregrasp_clearance_extra = 0.0
        grasp_shift_scale = 1.0
        squeeze_shift_scale = 1.0
        seed_palm_alignment_weight = 4.0
        seed_thumb_alignment_weight = 1.0
        seed_palm_inward_blend = 0.0
        approach_palm_weight = 1.0
        tip_smallness = 0.0
        final_seek_wrist_step_scale = 1.0
        final_seek_joint_close_step_scale = 1.0
        final_seek_clearance_multiplier_scale = 1.0
        tip_grasp_clearance = 0.003
        tip_squeeze_clearance = -0.0015
    else:
        tip_smallness = float(np.clip((0.012 - local_width) / 0.008, 0.0, 1.0))
        seed_posture_names = ["light_close", "medium_close"]
        use_tip_hook_close = tip_smallness >= 0.35
        close_posture_name = "rim_tip_hook_close" if use_tip_hook_close else "medium_close"
        limit_close_blend = (0.14 + 0.06 * tip_smallness) if use_tip_hook_close else 0.06
        seed_palm_depth_scale = 1.28
        seed_pregrasp_clearance_extra = max(0.004, 0.75 * local_width)
        grasp_shift_scale = 0.52
        squeeze_shift_scale = 0.32
        seed_palm_alignment_weight = 1.40 - 0.40 * tip_smallness
        seed_thumb_alignment_weight = 0.60 - 0.10 * tip_smallness
        seed_palm_inward_blend = 0.55 + 0.15 * tip_smallness
        approach_palm_weight = 0.28 - 0.10 * tip_smallness
        final_seek_wrist_step_scale = 0.90 - 0.20 * tip_smallness
        final_seek_joint_close_step_scale = 1.20 + 0.25 * tip_smallness
        final_seek_clearance_multiplier_scale = 1.10 + 0.35 * tip_smallness
        tip_grasp_clearance = 0.0020 - 0.0010 * tip_smallness
        tip_squeeze_clearance = -0.0020 - 0.0020 * tip_smallness

    return {
        "mode": mode,
        "mode_source": mode_source,
        "active_point_names": active_point_names,
        "opposition_pairs": opposition_pairs,
        "seed_posture_names": seed_posture_names,
        "close_posture_name": close_posture_name,
        "limit_close_blend": limit_close_blend,
        "seed_palm_depth_scale": seed_palm_depth_scale,
        "seed_pregrasp_clearance_extra": seed_pregrasp_clearance_extra,
        "grasp_shift_scale": grasp_shift_scale,
        "squeeze_shift_scale": squeeze_shift_scale,
        "seed_palm_alignment_weight": seed_palm_alignment_weight,
        "seed_thumb_alignment_weight": seed_thumb_alignment_weight,
        "seed_palm_inward_blend": seed_palm_inward_blend,
        "approach_palm_weight": approach_palm_weight,
        "tip_grasp_clearance": tip_grasp_clearance,
        "tip_squeeze_clearance": tip_squeeze_clearance,
        "final_seek_wrist_step_scale": final_seek_wrist_step_scale,
        "final_seek_joint_close_step_scale": final_seek_joint_close_step_scale,
        "final_seek_clearance_multiplier_scale": final_seek_clearance_multiplier_scale,
        "local_width": local_width,
        "pad_contact_span": pad_contact_span,
        "tip_contact_span": tip_contact_span,
        "tip_smallness": tip_smallness,
        "use_tip_hook_close": use_tip_hook_close if mode == "tip" else False,
        "width_to_pad_span": width_to_pad_span,
        "pad_min_width_ratio": pad_min_width_ratio,
        "pad_min_local_width": pad_min_local_width,
    }


def _normalize_opposition_pair(
    point_a: str,
    point_b: str,
    points_by_name: Dict[str, SemanticPoint],
) -> tuple[str, str]:
    semantic_a = points_by_name.get(point_a)
    semantic_b = points_by_name.get(point_b)
    if semantic_a is None or semantic_b is None:
        return point_a, point_b

    a_is_thumb = _is_thumb_point(semantic_a, point_a)
    b_is_thumb = _is_thumb_point(semantic_b, point_b)
    if (not a_is_thumb) and b_is_thumb:
        return point_b, point_a
    return point_a, point_b


def _build_opposition_constraints(
    runtime_cfg: ResolvedHandRuntimeConfig,
    desired_axis_world: np.ndarray,
    thumb_points: Sequence[tuple[str, SemanticPoint]],
    finger_points: Sequence[tuple[str, SemanticPoint]],
    *,
    opposition_pairs_override: Optional[Sequence[tuple[str, str]]] = None,
    ensure_thumb_to_all_fingers: bool,
) -> List[OppositionConstraint]:
    points_by_name = {name: semantic_point for name, semantic_point in thumb_points}
    points_by_name.update({name: semantic_point for name, semantic_point in finger_points})

    available_names = set(points_by_name)
    opposition: List[OppositionConstraint] = []
    seen_pairs: set[tuple[str, str]] = set()

    configured_pairs = list(opposition_pairs_override or runtime_cfg.contact_usage.opposition_pairs)
    for point_a, point_b in configured_pairs:
        if point_a not in available_names or point_b not in available_names:
            continue

        pair = _normalize_opposition_pair(point_a, point_b, points_by_name)
        if pair in seen_pairs:
            continue

        opposition.append(
            OppositionConstraint(
                point_a=pair[0],
                point_b=pair[1],
                desired_axis_world=np.asarray(desired_axis_world, dtype=np.float64).copy(),
                weight=1.0,
                metadata={"source": "config"},
            )
        )
        seen_pairs.add(pair)

    if ensure_thumb_to_all_fingers and thumb_points and finger_points:
        primary_thumb = thumb_points[0][0]
        for finger_name, _ in finger_points:
            pair = (primary_thumb, finger_name)
            if pair in seen_pairs:
                continue

            opposition.append(
                OppositionConstraint(
                    point_a=pair[0],
                    point_b=pair[1],
                    desired_axis_world=np.asarray(desired_axis_world, dtype=np.float64).copy(),
                    weight=1.0,
                    metadata={
                        "source": "derived",
                        "primary_thumb": primary_thumb,
                    },
                )
            )
            seen_pairs.add(pair)

    return opposition


def _candidate_score(
    local_point: np.ndarray,
    local_normal: np.ndarray,
    pref: _TargetPreference,
    position_sigma: float,
) -> float:
    delta = local_point - pref.preferred_local_point
    pos_score = float(np.exp(-np.dot(delta, delta) / max(position_sigma**2, 1e-8)))

    pref_normal = _safe_normalize(pref.preferred_local_normal)
    if np.linalg.norm(pref_normal) < 1e-8:
        normal_score = 0.5
    else:
        normal_score = 0.5 * (1.0 + float(np.dot(local_normal, pref_normal)))

    score = 3.0 * pos_score + 2.0 * normal_score

    if pref.side_axis is not None:
        side_coord = float(local_point[pref.side_axis])
        side_scale = max(pref.side_bias_scale, 1e-4)
        side_score = pref.side_sign * np.tanh(side_coord / side_scale)
        score += pref.side_bias_weight * float(side_score)

    if pref.prefer_side_surface:
        score += 1.5 * (1.0 - abs(float(local_normal[2])))
    if pref.prefer_horizontal_normal:
        score += 1.0 * (1.0 - abs(float(local_normal[2])))
    if pref.prefer_top_surface:
        score += 1.5 * max(float(local_normal[2]), 0.0)
    if pref.prefer_bottom_surface:
        score += 1.5 * max(float(-local_normal[2]), 0.0)

    return score


def _pick_surface_candidate(
    pref: _TargetPreference,
    patch_indices: np.ndarray,
    anchor: Anchor,
    frame_R: np.ndarray,
    samples: SurfaceSamples,
    reserved_indices: Sequence[int],
    patch_radius: float,
) -> Optional[int]:
    if len(patch_indices) == 0:
        return None

    local_points = (samples.points[patch_indices] - anchor.point[None, :]) @ frame_R
    local_normals = samples.normals[patch_indices] @ frame_R

    candidate_mask = np.ones(len(patch_indices), dtype=bool)
    if pref.side_axis is not None:
        preferred_side_extent = abs(float(pref.preferred_local_point[pref.side_axis]))
        side_margin = max(0.25 * preferred_side_extent, 1.5e-3)
        signed_side_coord = pref.side_sign * local_points[:, pref.side_axis]
        side_valid = signed_side_coord >= side_margin
        if np.any(side_valid):
            candidate_mask &= side_valid
    if pref.prefer_horizontal_normal:
        horizontal_valid = np.abs(local_normals[:, 2]) <= 0.35
        if np.any(candidate_mask & horizontal_valid):
            candidate_mask &= horizontal_valid
    if pref.prefer_top_surface:
        top_valid = local_normals[:, 2] >= 0.35
        if np.any(candidate_mask & top_valid):
            candidate_mask &= top_valid
    if pref.prefer_bottom_surface:
        bottom_valid = local_normals[:, 2] <= -0.35
        if np.any(candidate_mask & bottom_valid):
            candidate_mask &= bottom_valid
    pref_normal = _safe_normalize(pref.preferred_local_normal)
    if np.linalg.norm(pref_normal) > 1e-8:
        normal_alignment = local_normals @ pref_normal
        normal_valid = normal_alignment >= 0.15
        if np.any(candidate_mask & normal_valid):
            candidate_mask &= normal_valid

    position_sigma = max(0.25 * patch_radius * pref.search_radius_scale, 1e-3)
    scores = np.array(
        [
            _candidate_score(local_points[i], local_normals[i], pref, position_sigma)
            for i in range(len(patch_indices))
        ],
        dtype=np.float64,
    )

    if pref.min_local_separation > 0.0 and reserved_indices:
        reserved_points = samples.points[np.asarray(list(reserved_indices), dtype=np.int64)]
        d = np.linalg.norm(
            samples.points[patch_indices][:, None, :] - reserved_points[None, :, :],
            axis=2,
        )
        separation_valid = ~np.any(d < pref.min_local_separation, axis=1)
        if np.any(candidate_mask & separation_valid):
            candidate_mask &= separation_valid
        else:
            scores[~separation_valid] -= 4.0

    scores[~candidate_mask] = -np.inf

    best_local = int(np.argmax(scores))
    if not np.isfinite(scores[best_local]):
        return None
    return int(patch_indices[best_local])


def _resolve_sample_target_normal(
    sample_normal_world: np.ndarray,
    frame_R: np.ndarray,
    pref: _TargetPreference,
) -> np.ndarray:
    sample_normal_world = _safe_normalize(np.asarray(sample_normal_world, dtype=np.float64))
    pref_normal_world = _safe_normalize(frame_R @ pref.preferred_local_normal)
    if np.linalg.norm(pref_normal_world) < 1e-8:
        return sample_normal_world

    if pref.prefer_top_surface:
        if float(sample_normal_world[2]) >= 0.6:
            return sample_normal_world
        return pref_normal_world

    if pref.prefer_bottom_surface:
        if float(sample_normal_world[2]) <= -0.6:
            return sample_normal_world
        return pref_normal_world

    if pref.prefer_side_surface or pref.prefer_horizontal_normal:
        local_normal = sample_normal_world @ frame_R
        alignment = float(np.dot(sample_normal_world, pref_normal_world))
        if abs(float(local_normal[2])) > 0.20 or alignment < 0.65:
            return pref_normal_world

    return sample_normal_world


def _orient_cat1_frame(anchor: Anchor, samples: SurfaceSamples) -> np.ndarray:
    frame_R = np.asarray(anchor.frame_R, dtype=np.float64).copy()

    x_axis = _safe_normalize(_project_to_xy(frame_R[:, 0]))
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    object_center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    outward_xy = anchor.point[:2] - object_center_xy
    if np.linalg.norm(outward_xy) > 1e-8:
        outward_axis = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
        outward_axis = _safe_normalize(outward_axis)
        if float(np.dot(y_axis, outward_axis)) < 0.0:
            x_axis *= -1.0
            y_axis *= -1.0

    return _ensure_right_handed_frame(x_axis, y_axis, z_axis)


def _orient_cat2_frame(anchor: Anchor, samples: SurfaceSamples) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _safe_normalize(_project_to_xy(anchor.frame_R[:, 0]))
    y_axis = _safe_normalize(_project_to_xy(anchor.frame_R[:, 1]))

    if np.linalg.norm(x_axis) < 1e-8 and np.linalg.norm(y_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif np.linalg.norm(x_axis) < 1e-8:
        x_axis = _safe_normalize(np.cross(y_axis, z_axis))
    elif np.linalg.norm(y_axis) < 1e-8:
        y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    x_axis = _safe_normalize(x_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    y_axis = y_axis - z_axis * float(np.dot(y_axis, z_axis))
    y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
    y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = _safe_normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    object_center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    outward_xy = anchor.point[:2] - object_center_xy
    if np.linalg.norm(outward_xy) > 1e-8:
        outward_axis = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
        outward_axis = _safe_normalize(outward_axis)
        if float(np.dot(y_axis, outward_axis)) < 0.0:
            x_axis *= -1.0
            y_axis *= -1.0

    return _ensure_right_handed_frame(x_axis, y_axis, z_axis)


def _cat2_anchor_distribution_mode(anchor: Anchor) -> str:
    return str(anchor.metadata.get("cat2_anchor_distribution_mode", "default"))


def _cat4_grasp_mode(anchor: Anchor) -> str:
    mode = str(anchor.metadata.get("cat4_grasp_mode", "side")).lower()
    return "top" if mode == "top" else "side"


def _orient_cat3_frame(anchor: Anchor, samples: SurfaceSamples) -> np.ndarray:
    object_center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    outward_xy = anchor.point[:2] - object_center_xy
    outward = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
    outward = _safe_normalize(outward)
    if np.linalg.norm(outward) < 1e-8:
        outward = _project_to_xy(anchor.normal)
        outward = _safe_normalize(outward)
    if np.linalg.norm(outward) < 1e-8:
        outward = _project_to_xy(anchor.frame_R[:, 1])
        outward = _safe_normalize(outward)
    if np.linalg.norm(outward) < 1e-8:
        outward = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    pair_inward_axis = anchor.metadata.get("cat3_pair_inward_axis")
    pair_inward = None
    if pair_inward_axis is not None:
        pair_inward = np.asarray(pair_inward_axis, dtype=np.float64).reshape(-1)
        if pair_inward.shape[0] >= 3:
            pair_inward = np.array([pair_inward[0], pair_inward[1], 0.0], dtype=np.float64)
            pair_inward = _safe_normalize(pair_inward)
        else:
            pair_inward = None

    # cat3 should seed a vertical side hold:
    # - fingers point downward
    # - palm faces inward toward the opposing hand/object
    # - thumb/finger opposition spans the lateral side axis
    x_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    z_axis = pair_inward if pair_inward is not None and np.linalg.norm(pair_inward) > 1e-8 else -outward
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = _project_to_xy(anchor.frame_R[:, 0])
        y_axis = _safe_normalize(y_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = _safe_normalize(np.cross(x_axis, y_axis))
    return _ensure_right_handed_frame(x_axis, y_axis, z_axis)


def _orient_cat4_frame(anchor: Anchor, samples: SurfaceSamples) -> np.ndarray:
    frame_R = np.asarray(anchor.frame_R, dtype=np.float64).copy()

    x_axis = _safe_normalize(_project_to_xy(frame_R[:, 0]))
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = _safe_normalize(np.cross(z_axis, x_axis))

    object_center_xy = 0.5 * (samples.bbox_min[:2] + samples.bbox_max[:2])
    outward_xy = anchor.point[:2] - object_center_xy
    if np.linalg.norm(outward_xy) > 1e-8:
        outward_axis = np.array([outward_xy[0], outward_xy[1], 0.0], dtype=np.float64)
        outward_axis = _safe_normalize(outward_axis)
        if float(np.dot(y_axis, outward_axis)) < 0.0:
            x_axis *= -1.0
            y_axis *= -1.0

    return _ensure_right_handed_frame(x_axis, y_axis, z_axis)


def _cat1_preferences(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    patch_radius: float,
    mode_info: Mapping[str, Any],
    use_palm_avoid_targets: bool = False,
) -> tuple[List[_TargetPreference], List[_TargetPreference], List[OppositionConstraint]]:
    contact_mode = str(mode_info.get("mode", "pad")).lower()
    local_width = float(anchor.metadata.get("local_width", max(0.02, 0.6 * patch_radius)))
    if contact_mode == "tip":
        rim_half_width = np.clip(0.35 * local_width, 0.0015, 0.012)
        finger_spread_total = np.clip(0.85 * local_width, 0.006, 0.024)
        tip_hold_drop = np.clip(0.18 * local_width, 0.0015, 0.006)
    else:
        rim_half_width = np.clip(0.35 * local_width, 0.006, 0.03)
        finger_spread_total = np.clip(0.85 * local_width, 0.016, 0.065)
        tip_hold_drop = np.clip(0.04 * local_width, 0.0005, 0.0035)
    thumb_wall_drop = np.clip(0.08 * local_width, 0.001, 0.008)
    finger_wall_drop = np.clip(0.22 * local_width, 0.004, 0.018)

    thumb_points, finger_points, _ = _split_named_thumb_and_finger_points(
        runtime_cfg,
        mode_info["active_point_names"],
        prefer_thumb_pad=False,
    )

    active_prefs: List[_TargetPreference] = []
    tip_grasp_clearance = float(mode_info.get("tip_grasp_clearance", 0.003))
    tip_squeeze_clearance = float(mode_info.get("tip_squeeze_clearance", -0.0015))

    if len(thumb_points) > 0:
        thumb_x_offsets = np.linspace(-0.18, 0.18, len(thumb_points)) * finger_spread_total
        for idx, (point_name, semantic_point) in enumerate(thumb_points):
            tags = _semantic_point_tags(semantic_point, point_name)
            thumb_weight = 1.30 if idx == 0 else 1.05
            thumb_clearances = (0.016, 0.002, -0.0015)
            thumb_is_tip = "tip" in tags and "pad" not in tags
            if thumb_is_tip:
                thumb_clearances = (0.016, tip_grasp_clearance, tip_squeeze_clearance)
                preferred_local_point = np.array(
                    [float(thumb_x_offsets[idx]), -0.85 * rim_half_width, -tip_hold_drop],
                    dtype=np.float64,
                )
                preferred_local_normal = _safe_normalize(np.array([0.0, -1.0, 0.30], dtype=np.float64))
                prefer_side_surface = True
                prefer_horizontal_normal = False
                prefer_top_surface = False
                side_bias_weight = 0.20
                search_radius_scale = 0.95
            else:
                preferred_local_point = np.array(
                    [float(thumb_x_offsets[idx]), -rim_half_width, -thumb_wall_drop],
                    dtype=np.float64,
                )
                preferred_local_normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                prefer_side_surface = True
                prefer_horizontal_normal = True
                prefer_top_surface = False
                side_bias_weight = 0.55
                search_radius_scale = 1.15 if "pad" in tags else 1.0
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=preferred_local_point,
                    preferred_local_normal=preferred_local_normal,
                    desired_clearances=thumb_clearances,
                    weight=thumb_weight,
                    prefer_side_surface=prefer_side_surface,
                    prefer_top_surface=prefer_top_surface,
                    prefer_horizontal_normal=prefer_horizontal_normal,
                    side_axis=1,
                    side_sign=-1.0,
                    side_bias_weight=side_bias_weight,
                    side_bias_scale=max(0.35 * rim_half_width, 0.006),
                    min_local_separation=0.010,
                    search_radius_scale=search_radius_scale,
                )
            )

    if len(finger_points) > 0:
        if len(finger_points) == 1:
            finger_x_offsets = np.array([0.0], dtype=np.float64)
        else:
            finger_x_offsets = np.linspace(
                +0.55 * finger_spread_total,
                -0.55 * finger_spread_total,
                len(finger_points),
            )

        for idx, (point_name, semantic_point) in enumerate(finger_points):
            tags = _semantic_point_tags(semantic_point, point_name)
            digit_rank = _digit_rank(semantic_point, point_name)
            finger_is_tip = "tip" in tags and "side" not in tags
            finger_clearances = (0.015, 0.003, -0.0015)
            if finger_is_tip:
                finger_clearances = (0.015, tip_grasp_clearance, tip_squeeze_clearance)
                preferred_local_point = np.array(
                    [float(finger_x_offsets[idx]), +0.85 * rim_half_width, -tip_hold_drop],
                    dtype=np.float64,
                )
                preferred_local_normal = _safe_normalize(np.array([0.0, +1.0, 0.30], dtype=np.float64))
                prefer_side_surface = True
                prefer_horizontal_normal = False
                prefer_top_surface = False
                side_bias_weight = 0.20
                search_radius_scale = 0.95
            else:
                preferred_local_point = np.array(
                    [float(finger_x_offsets[idx]), +rim_half_width, -finger_wall_drop],
                    dtype=np.float64,
                )
                preferred_local_normal = np.array([0.0, +1.0, 0.0], dtype=np.float64)
                prefer_side_surface = True
                prefer_horizontal_normal = True
                prefer_top_surface = False
                side_bias_weight = 0.45
                search_radius_scale = 1.18 if "side" in tags else 1.05
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=preferred_local_point,
                    preferred_local_normal=preferred_local_normal,
                    desired_clearances=finger_clearances,
                    weight=_finger_priority_weight(digit_rank),
                    prefer_side_surface=prefer_side_surface,
                    prefer_top_surface=prefer_top_surface,
                    prefer_horizontal_normal=prefer_horizontal_normal,
                    side_axis=1,
                    side_sign=+1.0,
                    side_bias_weight=side_bias_weight,
                    side_bias_scale=max(0.35 * rim_half_width, 0.006),
                    min_local_separation=0.012,
                    search_radius_scale=search_radius_scale,
                )
            )

    avoid_prefs: List[_TargetPreference] = []
    if use_palm_avoid_targets:
        available_avoid_names = {
            name
            for name, _ in _available_usage_points(
                runtime_cfg,
                runtime_cfg.contact_usage.avoid_points,
            )[0]
        }
        if "palm_avoid_center" in available_avoid_names:
            avoid_prefs.append(
                _TargetPreference(
                    name="palm_avoid_center",
                    role="avoid",
                    preferred_local_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    desired_clearances=(0.015, 0.010, 0.006),
                    weight=0.35,
                    prefer_top_surface=True,
                    side_axis=None,
                )
            )
        if "palm_avoid_lower" in available_avoid_names:
            avoid_prefs.append(
                _TargetPreference(
                    name="palm_avoid_lower",
                    role="avoid",
                    preferred_local_point=np.array([0.0, 0.0, -0.02], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    desired_clearances=(0.015, 0.010, 0.006),
                    weight=0.35,
                    prefer_side_surface=True,
                    side_axis=None,
                )
            )

    frame_R = np.asarray(anchor.frame_R, dtype=np.float64)
    opp_axis = _safe_normalize(frame_R[:, 1])
    if np.linalg.norm(opp_axis) < 1e-8:
        opp_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    opposition = _build_opposition_constraints(
        runtime_cfg=runtime_cfg,
        desired_axis_world=opp_axis,
        thumb_points=thumb_points,
        finger_points=finger_points,
        opposition_pairs_override=mode_info["opposition_pairs"],
        ensure_thumb_to_all_fingers=True,
    )
    return active_prefs, avoid_prefs, opposition


def _cat2_preferences(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    patch_radius: float,
    use_palm_avoid_targets: bool = False,
) -> tuple[List[_TargetPreference], List[_TargetPreference], List[OppositionConstraint]]:
    line_mode = _cat2_anchor_distribution_mode(anchor) == "bottom_rim_line"
    support_depth = np.clip((0.12 if line_mode else 0.20) * patch_radius, 0.003, 0.015)
    finger_spread_total = np.clip((0.85 if line_mode else 0.70) * patch_radius, 0.012, 0.06)
    thumb_support_y = (0.40 if line_mode else 0.50) * patch_radius
    finger_local_y = -np.clip(0.12 * patch_radius, 0.0015, 0.0060) if line_mode else 0.0
    active_point_names, opposition_pairs_override = _cat2_mode_point_names(runtime_cfg)
    thumb_points, finger_points, _ = _split_named_thumb_and_finger_points(
        runtime_cfg,
        active_point_names,
        prefer_thumb_pad=True,
    )

    active_prefs: List[_TargetPreference] = []

    if len(thumb_points) > 0:
        thumb_x_offsets = np.linspace(-0.12, 0.12, len(thumb_points)) * finger_spread_total
        for idx, (point_name, semantic_point) in enumerate(thumb_points):
            tags = _semantic_point_tags(semantic_point, point_name)
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=np.array(
                        [float(thumb_x_offsets[idx]), +thumb_support_y, +support_depth],
                        dtype=np.float64,
                    ),
                    preferred_local_normal=np.array([0.0, +1.0, 0.0], dtype=np.float64),
                    desired_clearances=(0.018, 0.007, 0.0035) if line_mode else (0.020, 0.008, 0.004),
                    weight=0.55 if idx == 0 else 0.45,
                    prefer_horizontal_normal=True,
                    side_axis=1,
                    side_sign=+1.0,
                    min_local_separation=0.012,
                    search_radius_scale=1.0 if "pad" in tags else 1.1,
                )
            )

    if len(finger_points) > 0:
        finger_points_by_digit: Dict[int, List[tuple[str, SemanticPoint]]] = {}
        for point_name, semantic_point in finger_points:
            digit_rank = _digit_rank(semantic_point, point_name)
            finger_points_by_digit.setdefault(digit_rank, []).append((point_name, semantic_point))

        ordered_digits = sorted(finger_points_by_digit)
        if len(ordered_digits) == 1:
            digit_x_offsets = np.array([0.0], dtype=np.float64)
        else:
            digit_x_offsets = np.linspace(
                +0.5 * finger_spread_total,
                -0.5 * finger_spread_total,
                len(ordered_digits),
            )

        for digit_idx, digit_rank in enumerate(ordered_digits):
            digit_points = sorted(
                finger_points_by_digit[digit_rank],
                key=lambda item: (
                    0 if "pad" in _semantic_point_tags(item[1], item[0]) else 1,
                    0 if "tip" in _semantic_point_tags(item[1], item[0]) else 1,
                    item[0],
                ),
            )
            digit_x = float(digit_x_offsets[digit_idx])
            for point_name, semantic_point in digit_points:
                tags = _semantic_point_tags(semantic_point, point_name)
                is_pad = "pad" in tags
                local_z = -np.clip((0.05 if is_pad and line_mode else 0.08 if is_pad else 0.015 if line_mode else 0.03) * patch_radius, 0.001, 0.004)
                desired_clearances = (
                    (0.012, 0.0015, 0.0000) if is_pad else (0.013, 0.0020, 0.0005)
                ) if line_mode else (
                    (0.014, 0.0020, 0.0000) if is_pad else (0.015, 0.0025, 0.0005)
                )
                weight_scale = 1.00 if is_pad else 0.82
                active_prefs.append(
                    _TargetPreference(
                        name=point_name,
                        role="active",
                        preferred_local_point=np.array(
                            [digit_x, float(finger_local_y), float(local_z)],
                            dtype=np.float64,
                        ),
                        preferred_local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                        desired_clearances=desired_clearances,
                        weight=_finger_priority_weight(digit_rank) * weight_scale,
                        prefer_bottom_surface=True,
                        min_local_separation=0.010,
                        search_radius_scale=0.95 if is_pad else 0.90,
                    )
                )

    avoid_prefs: List[_TargetPreference] = []
    if use_palm_avoid_targets:
        available_avoid_names = {
            name
            for name, _ in _available_usage_points(
                runtime_cfg,
                runtime_cfg.contact_usage.avoid_points,
            )[0]
        }
        if "palm_avoid_center" in available_avoid_names:
            avoid_prefs.append(
                _TargetPreference(
                    name="palm_avoid_center",
                    role="avoid",
                    preferred_local_point=np.array([0.0, +0.45 * patch_radius, +0.02], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, +1.0, 0.0], dtype=np.float64),
                    desired_clearances=(0.015, 0.010, 0.006),
                    weight=0.35,
                    prefer_horizontal_normal=True,
                )
            )
        if "palm_avoid_lower" in available_avoid_names:
            avoid_prefs.append(
                _TargetPreference(
                    name="palm_avoid_lower",
                    role="avoid",
                    preferred_local_point=np.array([0.0, +0.45 * patch_radius, -0.01], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, +1.0, 0.0], dtype=np.float64),
                    desired_clearances=(0.015, 0.010, 0.006),
                    weight=0.35,
                    prefer_horizontal_normal=True,
                )
            )

    opposition_finger_points: List[tuple[str, SemanticPoint]] = []
    finger_points_by_digit: Dict[int, List[tuple[str, SemanticPoint]]] = {}
    for point_name, semantic_point in finger_points:
        finger_points_by_digit.setdefault(_digit_rank(semantic_point, point_name), []).append((point_name, semantic_point))
    for digit_rank in sorted(finger_points_by_digit):
        digit_points = sorted(
            finger_points_by_digit[digit_rank],
            key=lambda item: (
                0 if "pad" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
                item[0],
            ),
        )
        if digit_points:
            opposition_finger_points.append(digit_points[0])

    opposition = _build_opposition_constraints(
        runtime_cfg=runtime_cfg,
        desired_axis_world=np.asarray(anchor.frame_R[:, 1], dtype=np.float64),
        thumb_points=thumb_points,
        finger_points=opposition_finger_points or finger_points,
        opposition_pairs_override=opposition_pairs_override,
        ensure_thumb_to_all_fingers=True,
    )
    return active_prefs, avoid_prefs, opposition


def _cat3_preferences(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    patch_radius: float,
    use_palm_avoid_targets: bool = False,
) -> tuple[List[_TargetPreference], List[_TargetPreference], List[OppositionConstraint]]:
    side_offset = np.clip(0.55 * patch_radius, 0.010, 0.04)
    finger_drop = np.clip(0.40 * patch_radius, 0.008, 0.028)
    thumb_raise = np.clip(0.14 * patch_radius, 0.002, 0.012)
    available_points, _ = _available_usage_points(
        runtime_cfg,
        runtime_cfg.contact_usage.active_points,
    )

    thumb_points: List[tuple[str, SemanticPoint]] = []
    palm_points: List[tuple[str, SemanticPoint]] = []
    finger_points: List[tuple[str, SemanticPoint]] = []
    for point_name, semantic_point in available_points:
        if _is_thumb_point(semantic_point, point_name):
            thumb_points.append((point_name, semantic_point))
        elif _is_palm_point(semantic_point, point_name):
            palm_points.append((point_name, semantic_point))
        elif _digit_rank(semantic_point, point_name) < 99:
            finger_points.append((point_name, semantic_point))

    thumb_points = _sort_thumb_points(thumb_points, prefer_pad=True)
    palm_points = sorted(
        palm_points,
        key=lambda item: (
            0 if "support" in _semantic_point_tags(item[1], item[0]) else 1,
            0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
            item[0],
        ),
    )
    finger_points = sorted(
        finger_points,
        key=lambda item: (
            _digit_rank(item[1], item[0]),
            0 if "pad" in _semantic_point_tags(item[1], item[0]) else 1,
            0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
            item[0],
        ),
    )

    active_prefs: List[_TargetPreference] = []

    if len(palm_points) > 0:
        point_name, _semantic_point = palm_points[0]
        active_prefs.append(
            _TargetPreference(
                name=point_name,
                role="active",
                preferred_local_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                preferred_local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                desired_clearances=(0.018, 0.004, 0.0),
                weight=1.15,
                prefer_horizontal_normal=True,
                side_axis=1,
                side_sign=0.0,
                min_local_separation=0.012,
                search_radius_scale=1.25,
            )
        )

    if len(thumb_points) > 0:
        if len(thumb_points) == 1:
            thumb_x_offsets = np.array([-0.5 * thumb_raise], dtype=np.float64)
        else:
            thumb_x_offsets = np.linspace(-thumb_raise, 0.0, len(thumb_points))

        for idx, (point_name, semantic_point) in enumerate(thumb_points):
            tags = _semantic_point_tags(semantic_point, point_name)
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=np.array([float(thumb_x_offsets[idx]), +side_offset, 0.0], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    desired_clearances=(0.015, 0.003, 0.0),
                    weight=1.2 if idx == 0 else 1.0,
                    prefer_horizontal_normal=True,
                    side_axis=1,
                    side_sign=+1.0,
                    min_local_separation=0.012,
                    search_radius_scale=1.0 if "pad" in tags else 1.1,
                )
            )

    if len(finger_points) > 0:
        if len(finger_points) == 1:
            finger_x_offsets = np.array([0.45 * finger_drop], dtype=np.float64)
        else:
            finger_x_offsets = np.linspace(0.15 * finger_drop, finger_drop, len(finger_points))

        for idx, (point_name, semantic_point) in enumerate(finger_points):
            digit_rank = _digit_rank(semantic_point, point_name)
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=np.array([float(finger_x_offsets[idx]), -side_offset, 0.0], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    desired_clearances=(0.015, 0.003, 0.0),
                    weight=_finger_priority_weight(digit_rank),
                    prefer_horizontal_normal=True,
                    side_axis=1,
                    side_sign=-1.0,
                    min_local_separation=0.012,
                )
            )

    avoid_prefs: List[_TargetPreference] = []
    if use_palm_avoid_targets:
        available_avoid_names = {
            name
            for name, _ in _available_usage_points(
                runtime_cfg,
                runtime_cfg.contact_usage.avoid_points,
            )[0]
        }
        if "palm_avoid_center" in available_avoid_names:
            avoid_prefs.append(
                _TargetPreference(
                    name="palm_avoid_center",
                    role="avoid",
                    preferred_local_point=np.array([0.20 * finger_drop, 0.0, 0.0], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    desired_clearances=(0.012, 0.008, 0.004),
                    weight=0.40,
                    prefer_horizontal_normal=True,
                )
            )
        if "palm_avoid_lower" in available_avoid_names:
            avoid_prefs.append(
                _TargetPreference(
                    name="palm_avoid_lower",
                    role="avoid",
                    preferred_local_point=np.array([0.55 * finger_drop, 0.0, 0.0], dtype=np.float64),
                    preferred_local_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                    desired_clearances=(0.012, 0.008, 0.004),
                    weight=0.40,
                    prefer_horizontal_normal=True,
                )
            )

    opp_axis = -_safe_normalize(anchor.frame_R[:, 1])
    if np.linalg.norm(opp_axis) < 1e-8:
        opp_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    opposition = _build_opposition_constraints(
        runtime_cfg=runtime_cfg,
        desired_axis_world=opp_axis,
        thumb_points=thumb_points,
        finger_points=finger_points,
        ensure_thumb_to_all_fingers=True,
    )
    return active_prefs, avoid_prefs, opposition


def _cat4_preferences(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    patch_radius: float,
) -> tuple[List[_TargetPreference], List[_TargetPreference], List[OppositionConstraint]]:
    grasp_mode = _cat4_grasp_mode(anchor)
    wrap_offset = np.clip(0.60 * patch_radius, 0.012, 0.05)
    vertical_spread = np.clip(0.38 * patch_radius, 0.010, 0.035)
    tangent_stagger = np.clip(0.12 * patch_radius, 0.0015, 0.006)

    active_point_names, opposition_pairs_override = _cat4_mode_point_names(runtime_cfg)
    available_points, _ = _available_usage_points(runtime_cfg, active_point_names)

    thumb_points: List[tuple[str, SemanticPoint]] = []
    palm_points: List[tuple[str, SemanticPoint]] = []
    finger_points_by_digit: Dict[int, List[tuple[str, SemanticPoint]]] = {}
    for point_name, semantic_point in available_points:
        if _is_thumb_point(semantic_point, point_name):
            thumb_points.append((point_name, semantic_point))
        elif _is_palm_point(semantic_point, point_name):
            palm_points.append((point_name, semantic_point))
        else:
            digit_rank = _digit_rank(semantic_point, point_name)
            if digit_rank < 99:
                finger_points_by_digit.setdefault(digit_rank, []).append((point_name, semantic_point))

    thumb_points = _sort_thumb_points(
        thumb_points,
        prefer_pad=(grasp_mode == "side"),
    )
    finger_rep_points: List[tuple[str, SemanticPoint]] = []
    active_prefs: List[_TargetPreference] = []

    if grasp_mode == "side" and len(palm_points) > 0:
        palm_points = sorted(
            palm_points,
            key=lambda item: (
                0 if "support" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
                item[0],
            ),
        )
        point_name, _semantic_point = palm_points[0]
        active_prefs.append(
            _TargetPreference(
                name=point_name,
                role="active",
                preferred_local_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                preferred_local_normal=np.array([0.0, +1.0, 0.0], dtype=np.float64),
                desired_clearances=(0.024, 0.014, 0.008),
                weight=0.40,
                prefer_horizontal_normal=True,
                side_axis=1,
                side_sign=+1.0,
                min_local_separation=0.012,
                search_radius_scale=1.15,
            )
        )

    if len(thumb_points) > 0:
        if len(thumb_points) == 1:
            thumb_z_offsets = np.array([0.0], dtype=np.float64)
        else:
            thumb_z_offsets = np.linspace(+0.35 * vertical_spread, -0.35 * vertical_spread, len(thumb_points))

        for idx, (point_name, semantic_point) in enumerate(thumb_points):
            tags = _semantic_point_tags(semantic_point, point_name)
            is_tip = _is_declared_tip_semantic_point(runtime_cfg, semantic_point, point_name) and "pad" not in tags
            if grasp_mode == "top":
                preferred_local_point = np.array(
                    [0.0, +0.84 * wrap_offset, float(thumb_z_offsets[idx] - 0.52 * vertical_spread)],
                    dtype=np.float64,
                ) if is_tip else np.array(
                    [0.0, +1.08 * wrap_offset, float(thumb_z_offsets[idx] - 0.34 * vertical_spread)],
                    dtype=np.float64,
                )
                preferred_local_normal = _safe_normalize(
                    np.array([0.0, +1.0, +0.22], dtype=np.float64)
                ) if is_tip else _safe_normalize(np.array([0.0, +1.0, +0.12], dtype=np.float64))
                desired_clearances = (0.016, 0.0030, -0.0015) if is_tip else (0.021, 0.0080, 0.0025)
                weight = (1.60 if is_tip else 0.66) if idx == 0 else (1.30 if is_tip else 0.56)
                prefer_side_surface = False
                prefer_horizontal_normal = False
                search_radius_scale = 1.10 if is_tip else 1.18
            else:
                preferred_local_point = np.array(
                    [0.0, +0.88 * wrap_offset, float(thumb_z_offsets[idx] - 0.18 * vertical_spread)],
                    dtype=np.float64,
                ) if is_tip else np.array(
                    [0.0, +1.28 * wrap_offset, float(thumb_z_offsets[idx])],
                    dtype=np.float64,
                )
                preferred_local_normal = _safe_normalize(
                    np.array([0.0, +1.0, -0.20], dtype=np.float64)
                ) if is_tip else np.array([0.0, +1.0, 0.0], dtype=np.float64)
                desired_clearances = (0.016, 0.0035, -0.0015) if is_tip else (0.022, 0.0100, 0.0040)
                weight = (1.65 if is_tip else 0.78) if idx == 0 else (1.35 if is_tip else 0.62)
                prefer_side_surface = True
                prefer_horizontal_normal = True
                search_radius_scale = 0.92 if is_tip else 1.12
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=preferred_local_point,
                    preferred_local_normal=preferred_local_normal,
                    desired_clearances=desired_clearances,
                    weight=weight,
                    prefer_side_surface=prefer_side_surface,
                    prefer_horizontal_normal=prefer_horizontal_normal,
                    side_axis=1,
                    side_sign=+1.0,
                    min_local_separation=0.012,
                    search_radius_scale=search_radius_scale,
                )
            )

    ordered_digits = sorted(finger_points_by_digit)
    if len(ordered_digits) == 1:
        digit_z_offsets = np.array([0.0], dtype=np.float64)
    elif len(ordered_digits) > 1:
        digit_z_offsets = np.linspace(+vertical_spread, -vertical_spread, len(ordered_digits))
    else:
        digit_z_offsets = np.zeros(0, dtype=np.float64)

    for digit_idx, digit_rank in enumerate(ordered_digits):
        digit_points = sorted(
            finger_points_by_digit[digit_rank],
            key=lambda item: (
                0 if "pad" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "tip" in _semantic_point_tags(item[1], item[0]) else 1,
                0 if "main" in _semantic_point_tags(item[1], item[0]) else 1,
                item[0],
            ),
        )
        if digit_points:
            finger_rep_points.append(digit_points[0])
        digit_z = float(digit_z_offsets[digit_idx]) if len(digit_z_offsets) > digit_idx else 0.0
        for point_name, semantic_point in digit_points:
            tags = _semantic_point_tags(semantic_point, point_name)
            is_tip = _is_declared_tip_semantic_point(runtime_cfg, semantic_point, point_name) and "pad" not in tags
            if grasp_mode == "top":
                local_x = +0.70 * tangent_stagger if is_tip else -0.60 * tangent_stagger
                local_y = -0.92 * wrap_offset if is_tip else -0.76 * wrap_offset
                local_z = digit_z - 0.34 * vertical_spread
                preferred_local_normal = _safe_normalize(np.array([0.0, -1.0, +0.20], dtype=np.float64))
                desired_clearances = (0.018, 0.0055, 0.0000) if is_tip else (0.020, 0.0080, 0.0020)
                prefer_horizontal_normal = False
                search_radius_scale = 1.08 if is_tip else 1.14
            else:
                local_x = +tangent_stagger if is_tip else -tangent_stagger
                local_y = -1.05 * wrap_offset if is_tip else -wrap_offset
                local_z = digit_z
                preferred_local_normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                desired_clearances = (0.018, 0.006, 0.0000) if is_tip else (0.020, 0.010, 0.0030)
                prefer_horizontal_normal = True
                search_radius_scale = 0.95 if is_tip else 1.0
            active_prefs.append(
                _TargetPreference(
                    name=point_name,
                    role="active",
                    preferred_local_point=np.array([float(local_x), float(local_y), float(local_z)], dtype=np.float64),
                    preferred_local_normal=preferred_local_normal,
                    desired_clearances=desired_clearances,
                    weight=_finger_priority_weight(digit_rank) * (1.30 if is_tip else 0.90),
                    prefer_horizontal_normal=prefer_horizontal_normal,
                    side_axis=1,
                    side_sign=-1.0,
                    min_local_separation=0.012,
                    search_radius_scale=search_radius_scale,
                )
            )

    opp_axis = _safe_normalize(anchor.frame_R[:, 1])
    if np.linalg.norm(opp_axis) < 1e-8:
        opp_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    opposition = _build_opposition_constraints(
        runtime_cfg=runtime_cfg,
        desired_axis_world=opp_axis,
        thumb_points=thumb_points,
        finger_points=finger_rep_points,
        opposition_pairs_override=opposition_pairs_override,
        ensure_thumb_to_all_fingers=True,
    )
    return active_prefs, [], opposition


def _make_targets(
    runtime_cfg: ResolvedHandRuntimeConfig,
    prefs: List[_TargetPreference],
    patch_indices: np.ndarray,
    anchor: Anchor,
    frame_R: np.ndarray,
    samples: SurfaceSamples,
    patch_radius: float,
) -> List[ContactTarget]:
    reserved_indices: List[int] = []
    results: List[ContactTarget] = []

    for pref in prefs:
        semantic_point, source_sphere = _lookup_semantic_source(runtime_cfg, pref.name)
        idx = _pick_surface_candidate(
            pref=pref,
            patch_indices=patch_indices,
            anchor=anchor,
            frame_R=frame_R,
            samples=samples,
            reserved_indices=reserved_indices,
            patch_radius=patch_radius,
        )

        if idx is None:
            target_point = anchor.point + frame_R @ pref.preferred_local_point
            target_normal = _safe_normalize(frame_R @ pref.preferred_local_normal)
            ref_point = target_point.copy()
            metadata = {
                "selection": "fallback_from_anchor_frame",
            }
        else:
            duplicate_or_too_close = idx in reserved_indices
            if (not duplicate_or_too_close) and pref.min_local_separation > 0.0 and reserved_indices:
                reserved_points = samples.points[np.asarray(list(reserved_indices), dtype=np.int64)]
                d = np.linalg.norm(reserved_points - samples.points[idx][None, :], axis=1)
                duplicate_or_too_close = bool(np.any(d < pref.min_local_separation))

            if duplicate_or_too_close:
                target_point = anchor.point + frame_R @ pref.preferred_local_point
                target_normal = _safe_normalize(frame_R @ pref.preferred_local_normal)
                ref_point = target_point.copy()
                metadata = {
                    "selection": "fallback_due_to_duplicate_sample",
                    "conflict_candidate_index": int(idx),
                }
                idx = None
            else:
                target_point = samples.points[idx].copy()
                target_normal = _resolve_sample_target_normal(samples.normals[idx], frame_R, pref)
                ref_point = anchor.point + frame_R @ pref.preferred_local_point
                metadata = {
                    "selection": "surface_sample",
                    "local_point": ((target_point - anchor.point) @ frame_R).tolist(),
                    "normal_source": (
                        "preferred_local_normal"
                        if not np.allclose(target_normal, _safe_normalize(samples.normals[idx].copy()))
                        else "sample_normal"
                    ),
                }
                reserved_indices.append(idx)

        c_pre, c_grasp, c_squeeze = pref.desired_clearances
        results.append(
            ContactTarget(
                name=pref.name,
                role=pref.role,
                source_link=semantic_point.source_link,
                source_sphere_index=semantic_point.source_sphere_index,
                source_sphere=source_sphere,
                role_tags=list(semantic_point.role_tags),
                target_point=target_point,
                target_normal=target_normal,
                reference_point=ref_point,
                desired_clearance_pregrasp=float(c_pre),
                desired_clearance_grasp=float(c_grasp),
                desired_clearance_squeeze=float(c_squeeze),
                weight=float(pref.weight),
                candidate_index=idx,
                metadata=metadata,
            )
        )

    return results


def resolve_contact_template(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchor: Anchor,
    samples: SurfaceSamples,
    descriptors: Optional[Dict[str, np.ndarray]] = None,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
) -> ContactResolutionResult:
    """
    Resolve one object-side anchor into a contact template that later stages can
    use for seed generation and optimization.

    Design intent:
    - Keep the interface shared across categories.
    - Let category-specific logic differ mostly in local-frame interpretation and
      preferred target locations for semantic contact spheres.
    - Use sampled surface points as a lightweight stand-in for final mesh / SDF
      queries. Later, this module can swap the candidate picker to SDF + mesh
      projection without changing the output structure.
    """
    _ = descriptors
    overrides = dict(contact_cfg_overrides or {})
    use_palm_avoid_targets = bool(
        overrides.get("use_palm_avoid_targets", anchor.category == "cat2")
    )
    available_active_points, missing_active_points = _available_usage_points(
        runtime_cfg,
        runtime_cfg.contact_usage.active_points,
    )
    available_avoid_points, missing_avoid_points = _available_usage_points(
        runtime_cfg,
        runtime_cfg.contact_usage.avoid_points,
    )
    patch_radius, patch_meta = _estimate_patch_radius(
        runtime_cfg=runtime_cfg,
        anchor=anchor,
        samples=samples,
        overrides=overrides,
    )

    cat1_mode_info: Optional[Dict[str, Any]] = None
    if anchor.category == "cat1":
        cat1_mode_info = _select_cat1_contact_mode(
            runtime_cfg=runtime_cfg,
            anchor=anchor,
            patch_radius=patch_radius,
            overrides=overrides,
        )
        available_active_points, missing_active_points = _available_usage_points(
            runtime_cfg,
            cat1_mode_info["active_point_names"],
        )
        frame_R = _orient_cat1_frame(anchor, samples)
        active_prefs, avoid_prefs, opposition = _cat1_preferences(
            runtime_cfg,
            anchor,
            patch_radius,
            cat1_mode_info,
            use_palm_avoid_targets=use_palm_avoid_targets,
        )
    elif anchor.category == "cat2":
        cat2_active_point_names, _ = _cat2_mode_point_names(runtime_cfg)
        available_active_points, missing_active_points = _available_usage_points(
            runtime_cfg,
            cat2_active_point_names,
        )
        frame_R = _orient_cat2_frame(anchor, samples)
        active_prefs, avoid_prefs, opposition = _cat2_preferences(
            runtime_cfg,
            anchor,
            patch_radius,
            use_palm_avoid_targets=use_palm_avoid_targets,
        )
    elif anchor.category == "cat3":
        frame_R = _orient_cat3_frame(anchor, samples)
        active_prefs, avoid_prefs, opposition = _cat3_preferences(
            runtime_cfg,
            anchor,
            patch_radius,
            use_palm_avoid_targets=use_palm_avoid_targets,
        )
    elif anchor.category == "cat4":
        cat4_active_point_names, _ = _cat4_mode_point_names(runtime_cfg)
        available_active_points, missing_active_points = _available_usage_points(
            runtime_cfg,
            cat4_active_point_names,
        )
        frame_R = _orient_cat4_frame(anchor, samples)
        active_prefs, avoid_prefs, opposition = _cat4_preferences(
            runtime_cfg,
            anchor,
            patch_radius,
        )
    else:
        raise ValueError(f"Unsupported category: {anchor.category}")

    if anchor.category == "cat1":
        patch_indices = _build_cat1_rim_band_indices(
            anchor,
            samples,
            frame_R,
            patch_radius,
        )
    else:
        patch_indices = _build_patch_indices(anchor, samples, radius=patch_radius)
    active_targets = _make_targets(
        runtime_cfg=runtime_cfg,
        prefs=active_prefs,
        patch_indices=patch_indices,
        anchor=anchor,
        frame_R=frame_R,
        samples=samples,
        patch_radius=patch_radius,
    )
    avoid_targets = _make_targets(
        runtime_cfg=runtime_cfg,
        prefs=avoid_prefs,
        patch_indices=patch_indices,
        anchor=anchor,
        frame_R=frame_R,
        samples=samples,
        patch_radius=patch_radius,
    )

    desired_opp_axis = _safe_normalize(frame_R[:, 1])
    if anchor.category == "cat4":
        desired_opp_axis *= -1.0
    if np.linalg.norm(desired_opp_axis) < 1e-8:
        desired_opp_axis = _safe_normalize(frame_R[:, 0])
    for opp in opposition:
        opp.desired_axis_world = desired_opp_axis.copy()

    object_center = 0.5 * (samples.bbox_min + samples.bbox_max)
    available_active_names = [name for name, _ in available_active_points]
    configured_opposition_pairs = [
        [point_a, point_b]
        for point_a, point_b in runtime_cfg.contact_usage.opposition_pairs
    ]
    if cat1_mode_info is not None:
        available_active_names = list(cat1_mode_info["active_point_names"])
        configured_opposition_pairs = [
            [point_a, point_b] for point_a, point_b in cat1_mode_info["opposition_pairs"]
        ]
    return ContactResolutionResult(
        category=anchor.category,
        contact_template=runtime_cfg.category_cfg.contact_logic.contact_template,
        anchor=anchor,
        frame_R=frame_R,
        object_center=object_center,
        active_targets=active_targets,
        avoid_targets=avoid_targets,
        opposition_constraints=opposition,
        patch_indices=patch_indices,
        metadata={
            "anchor_score": float(anchor.score),
            "cat4_grasp_mode": anchor.metadata.get("cat4_grasp_mode"),
            "cat2_anchor_distribution_mode": anchor.metadata.get("cat2_anchor_distribution_mode"),
            "cat2_line_axis_xy": anchor.metadata.get("cat2_line_axis_xy"),
            "cat2_line_side_axis_xy": anchor.metadata.get("cat2_line_side_axis_xy"),
            "cat2_line_center_xy": anchor.metadata.get("cat2_line_center_xy"),
            "cat2_line_coord": anchor.metadata.get("cat2_line_coord"),
            "cat2_line_side_coord": anchor.metadata.get("cat2_line_side_coord"),
            "patch_radius": float(patch_radius),
            "object_bbox_min": [float(x) for x in samples.bbox_min.tolist()],
            "object_bbox_max": [float(x) for x in samples.bbox_max.tolist()],
            "num_patch_points": int(len(patch_indices)),
            "palm_policy": (
                "soft_clearance_targets" if use_palm_avoid_targets else "collision_only"
            ),
            "available_active_points": available_active_names,
            "missing_active_points": missing_active_points,
            "available_avoid_points": [name for name, _ in available_avoid_points],
            "missing_avoid_points": missing_avoid_points,
            "configured_opposition_pairs": configured_opposition_pairs,
            "resolved_opposition_pairs": [
                [opp.point_a, opp.point_b]
                for opp in opposition
            ],
            **(
                {
                    "cat1_contact_mode": cat1_mode_info.get("mode"),
                    "cat1_contact_mode_source": cat1_mode_info.get("mode_source"),
                    "cat1_seed_posture_names": cat1_mode_info.get("seed_posture_names"),
                    "cat1_close_posture_name": cat1_mode_info.get("close_posture_name"),
                    "cat1_limit_close_blend": cat1_mode_info.get("limit_close_blend"),
                    "cat1_seed_palm_depth_scale": cat1_mode_info.get("seed_palm_depth_scale"),
                    "cat1_seed_pregrasp_clearance_extra": cat1_mode_info.get("seed_pregrasp_clearance_extra"),
                    "cat1_grasp_shift_scale": cat1_mode_info.get("grasp_shift_scale"),
                    "cat1_squeeze_shift_scale": cat1_mode_info.get("squeeze_shift_scale"),
                    "cat1_seed_palm_alignment_weight": cat1_mode_info.get("seed_palm_alignment_weight"),
                    "cat1_seed_thumb_alignment_weight": cat1_mode_info.get("seed_thumb_alignment_weight"),
                    "cat1_seed_palm_inward_blend": cat1_mode_info.get("seed_palm_inward_blend"),
                    "cat1_approach_palm_weight": cat1_mode_info.get("approach_palm_weight"),
                    "cat1_tip_grasp_clearance": cat1_mode_info.get("tip_grasp_clearance"),
                    "cat1_tip_squeeze_clearance": cat1_mode_info.get("tip_squeeze_clearance"),
                    "cat1_final_seek_wrist_step_scale": cat1_mode_info.get("final_seek_wrist_step_scale"),
                    "cat1_final_seek_joint_close_step_scale": cat1_mode_info.get("final_seek_joint_close_step_scale"),
                    "cat1_final_seek_clearance_multiplier_scale": cat1_mode_info.get("final_seek_clearance_multiplier_scale"),
                    "cat1_local_width": cat1_mode_info.get("local_width"),
                    "cat1_pad_contact_span": cat1_mode_info.get("pad_contact_span"),
                    "cat1_tip_contact_span": cat1_mode_info.get("tip_contact_span"),
                    "cat1_tip_smallness": cat1_mode_info.get("tip_smallness"),
                    "cat1_width_to_pad_span": cat1_mode_info.get("width_to_pad_span"),
                    "cat1_pad_min_width_ratio": cat1_mode_info.get("pad_min_width_ratio"),
                    "cat1_pad_min_local_width": cat1_mode_info.get("pad_min_local_width"),
                }
                if cat1_mode_info
                else {}
            ),
            **patch_meta,
        },
    )


def resolve_contact_templates_for_anchors(
    runtime_cfg: ResolvedHandRuntimeConfig,
    anchors: Sequence[Anchor],
    samples: SurfaceSamples,
    descriptors: Optional[Dict[str, np.ndarray]] = None,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
) -> List[ContactResolutionResult]:
    return [
        resolve_contact_template(
            runtime_cfg=runtime_cfg,
            anchor=anchor,
            samples=samples,
            descriptors=descriptors,
            contact_cfg_overrides=contact_cfg_overrides,
        )
        for anchor in anchors
    ]


def summarize_contact_resolution(result: ContactResolutionResult) -> Dict[str, Any]:
    def pack_target(t: ContactTarget) -> Dict[str, Any]:
        return {
            "name": t.name,
            "role": t.role,
            "source_link": t.source_link,
            "source_sphere_index": int(t.source_sphere_index),
            "sphere_radius": float(t.source_sphere.radius),
            "target_point": [float(x) for x in t.target_point.tolist()],
            "target_normal": [float(x) for x in t.target_normal.tolist()],
            "reference_point": [float(x) for x in t.reference_point.tolist()],
            "desired_clearance_pregrasp": float(t.desired_clearance_pregrasp),
            "desired_clearance_grasp": float(t.desired_clearance_grasp),
            "desired_clearance_squeeze": float(t.desired_clearance_squeeze),
            "candidate_index": None if t.candidate_index is None else int(t.candidate_index),
            "metadata": t.metadata,
        }

    return {
        "category": result.category,
        "contact_template": result.contact_template,
        "anchor_score": float(result.anchor.score),
        "anchor_point": [float(x) for x in result.anchor.point.tolist()],
        "num_patch_points": int(len(result.patch_indices)),
        "active_targets": [pack_target(t) for t in result.active_targets],
        "avoid_targets": [pack_target(t) for t in result.avoid_targets],
        "opposition_constraints": [
            {
                "point_a": c.point_a,
                "point_b": c.point_b,
                "desired_axis_world": [float(x) for x in c.desired_axis_world.tolist()],
                "weight": float(c.weight),
                "metadata": c.metadata,
            }
            for c in result.opposition_constraints
        ],
        "metadata": result.metadata,
    }
