from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config_loader import load_all_for_hand
from src.types_config import Category, Side, ResolvedHandRuntimeConfig
from src.region_proposal import (
    Anchor,
    AssembledMesh,
    SurfaceSamples,
    _filter_samples_bottom_ratio,
    _filter_samples_exclude_bottom_clearance,
    _filter_samples_top_ratio,
    compute_shared_descriptors,
    load_assembled_mesh_from_usd,
    propose_anchors_from_samples,
    sample_surface_from_assembled_mesh,
)


@dataclass
class SingleHandProposalResult:
    side: Side
    category: Category
    object_usd: Path

    runtime_cfg: ResolvedHandRuntimeConfig
    mesh: AssembledMesh
    samples: SurfaceSamples
    descriptors: Dict[str, np.ndarray]
    anchors: List[Anchor]


def order_anchors_for_consumption(
    anchors: List[Anchor],
    *,
    category: Optional[Category] = None,
) -> List[Anchor]:
    ordered = sorted(anchors, key=lambda anchor: anchor.score, reverse=True)
    if not ordered:
        return ordered

    resolved_category = category or ordered[0].category
    if resolved_category != "cat4":
        return ordered

    side_anchors = [
        anchor
        for anchor in ordered
        if str(anchor.metadata.get("cat4_grasp_mode", "side")) != "top"
    ]
    top_anchors = [
        anchor
        for anchor in ordered
        if str(anchor.metadata.get("cat4_grasp_mode", "side")) == "top"
    ]
    if not side_anchors or not top_anchors:
        return ordered

    result: List[Anchor] = []
    side_queue = list(side_anchors)
    top_queue = list(top_anchors)
    next_mode = "top" if top_queue[0].score > side_queue[0].score else "side"
    while side_queue or top_queue:
        if next_mode == "side":
            if side_queue:
                result.append(side_queue.pop(0))
            next_mode = "top" if top_queue else "side"
        else:
            if top_queue:
                result.append(top_queue.pop(0))
            next_mode = "side" if side_queue else "top"
    return result


def _extract_region_cfg(runtime_cfg: ResolvedHandRuntimeConfig) -> Dict[str, Any]:
    """
    Convert loaded category config into the dict expected by region_proposal.py.
    """
    cfg: Dict[str, Any] = {}
    cfg["mode"] = runtime_cfg.category_cfg.region_proposal.mode
    cfg.update(runtime_cfg.category_cfg.region_proposal.params)
    return cfg


def run_single_hand_region_proposal(
    hand_dir: Path,
    global_config_dir: Path,
    side: Side,
    category: Category,
    object_usd: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[List[str]] = None,
    num_surface_points: int = 100000,
    seed: int = 0,
    region_cfg_overrides: Optional[Dict[str, Any]] = None,
) -> SingleHandProposalResult:
    """
    Current stage:
    - load configs
    - assemble object mesh from USD
    - sample object surface
    - compute shared descriptors
    - propose category anchors

    Later this can be extended with:
    - contact resolution
    - seed generation
    - coarse/fine/final optimization
    """
    hand_dir = hand_dir.resolve()
    global_config_dir = global_config_dir.resolve()
    object_usd = object_usd.resolve()

    runtime_cfg = load_all_for_hand(
        hand_dir=hand_dir,
        global_config_dir=global_config_dir,
        side=side,
        category=category,
    )

    region_cfg = _extract_region_cfg(runtime_cfg)
    if region_cfg_overrides:
        region_cfg.update(region_cfg_overrides)

    mesh = load_assembled_mesh_from_usd(
        usd_path=object_usd,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
    )

    samples = sample_surface_from_assembled_mesh(
        mesh=mesh,
        num_points=num_surface_points,
        seed=seed,
    )

    # Mirror the pre-filter used inside propose_anchors_from_assembled_mesh so
    # that result.samples (used for visualisation) shows only the rim band and
    # descriptors are computed on the same reduced set.
    pre_filter_top = region_cfg.get("pre_filter_top_ratio", None)
    pre_filter_bottom = region_cfg.get("pre_filter_bottom_ratio", None)
    pre_filter_bottom_clearance = region_cfg.get("pre_filter_bottom_clearance", None)
    if pre_filter_top is not None:
        samples = _filter_samples_top_ratio(samples, float(pre_filter_top))
    elif pre_filter_bottom is not None:
        samples = _filter_samples_bottom_ratio(samples, float(pre_filter_bottom))
    elif pre_filter_bottom_clearance is not None:
        samples = _filter_samples_exclude_bottom_clearance(samples, float(pre_filter_bottom_clearance))

    k_neighbors = int(region_cfg.get("k_neighbors", 32))
    perimeter_band_width = region_cfg.get("perimeter_band_width", None)
    access_probe_length = region_cfg.get("access_probe_length", None)

    descriptors = compute_shared_descriptors(
        samples=samples,
        k_neighbors=k_neighbors,
        perimeter_band_width=perimeter_band_width,
        access_probe_length=access_probe_length,
    )

    anchors = propose_anchors_from_samples(
        samples=samples,
        descriptors=descriptors,
        category=category,
        region_cfg=region_cfg,
    )

    return SingleHandProposalResult(
        side=side,
        category=category,
        object_usd=object_usd,
        runtime_cfg=runtime_cfg,
        mesh=mesh,
        samples=samples,
        descriptors=descriptors,
        anchors=anchors,
    )


def summarize_single_hand_result(
    result: SingleHandProposalResult,
    top_k: int = 10,
) -> Dict[str, Any]:
    anchors_sorted = order_anchors_for_consumption(
        result.anchors,
        category=result.category,
    )[:top_k]

    summary = {
        "side": result.side,
        "category": result.category,
        "object_usd": str(result.object_usd),
        "num_vertices": int(result.mesh.vertices.shape[0]),
        "num_faces": int(result.mesh.faces.shape[0]),
        "num_surface_points": int(result.samples.points.shape[0]),
        "num_anchors": int(len(result.anchors)),
        "anchors_top_k": [],
    }

    for i, a in enumerate(anchors_sorted):
        summary["anchors_top_k"].append(
            {
                "rank": i,
                "score": float(a.score),
                "point": [float(x) for x in a.point.tolist()],
                "normal": [float(x) for x in a.normal.tolist()],
                "metadata": {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in a.metadata.items()
                },
            }
        )

    return summary
