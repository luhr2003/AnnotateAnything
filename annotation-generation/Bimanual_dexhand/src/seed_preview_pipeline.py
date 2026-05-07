from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.contact_resolution import (
    ContactResolutionResult,
    resolve_contact_template,
    summarize_contact_resolution,
)
from src.seed_generation import (
    PoseSeed,
    SeedGenerationResult,
    generate_pose_seeds,
    summarize_seed_generation,
)
from src.types_config import Category, Side

if TYPE_CHECKING:
    from src.pipeline_single_hand import SingleHandProposalResult


def _quat_wxyz_from_xyzw(quaternion_xyzw):
    q = [float(x) for x in quaternion_xyzw]
    return [q[3], q[0], q[1], q[2]]


@dataclass
class SingleHandSeedPreviewResult:
    proposal_result: SingleHandProposalResult
    contact_result: ContactResolutionResult
    seed_result: SeedGenerationResult
    selected_seed: PoseSeed
    anchor_rank: int
    seed_rank: int


def run_single_hand_seed_preview(
    *,
    hand_dir: Path,
    global_config_dir: Path,
    side: Side,
    category: Category,
    object_usd: Path,
    root_prim_path: Optional[str] = None,
    exclude_prim_paths: Optional[list[str]] = None,
    num_surface_points: int = 100000,
    seed: int = 0,
    anchor_rank: int = 0,
    num_seeds_per_contact: Optional[int] = None,
    seed_rank: int = 0,
    contact_cfg_overrides: Optional[Dict[str, Any]] = None,
) -> SingleHandSeedPreviewResult:
    from src.pipeline_single_hand import order_anchors_for_consumption, run_single_hand_region_proposal

    proposal_result = run_single_hand_region_proposal(
        hand_dir=hand_dir,
        global_config_dir=global_config_dir,
        side=side,
        category=category,
        object_usd=object_usd,
        root_prim_path=root_prim_path,
        exclude_prim_paths=exclude_prim_paths,
        num_surface_points=num_surface_points,
        seed=seed,
    )

    anchors_sorted = order_anchors_for_consumption(
        proposal_result.anchors,
        category=proposal_result.category,
    )
    if not anchors_sorted:
        raise RuntimeError("No anchors were generated for preview.")
    anchor_rank = int(max(0, min(anchor_rank, len(anchors_sorted) - 1)))
    anchor = anchors_sorted[anchor_rank]

    contact_result = resolve_contact_template(
        runtime_cfg=proposal_result.runtime_cfg,
        anchor=anchor,
        samples=proposal_result.samples,
        descriptors=proposal_result.descriptors,
        contact_cfg_overrides=contact_cfg_overrides,
    )

    seed_result = generate_pose_seeds(
        runtime_cfg=proposal_result.runtime_cfg,
        contact_result=contact_result,
        num_seeds=num_seeds_per_contact,
        seed=seed,
    )
    if not seed_result.seeds:
        raise RuntimeError("No pose seeds were generated for preview.")
    seed_rank = int(max(0, min(seed_rank, len(seed_result.seeds) - 1)))
    selected_seed = seed_result.seeds[seed_rank]

    return SingleHandSeedPreviewResult(
        proposal_result=proposal_result,
        contact_result=contact_result,
        seed_result=seed_result,
        selected_seed=selected_seed,
        anchor_rank=anchor_rank,
        seed_rank=seed_rank,
    )


def summarize_single_hand_seed_preview(
    result: SingleHandSeedPreviewResult,
) -> Dict[str, Any]:
    return {
        "side": result.proposal_result.side,
        "category": result.proposal_result.category,
        "object_usd": str(result.proposal_result.object_usd),
        "anchor_rank": int(result.anchor_rank),
        "seed_rank": int(result.seed_rank),
        "contact": summarize_contact_resolution(result.contact_result),
        "seed_generation": summarize_seed_generation(result.seed_result, top_k=result.seed_rank + 1),
        "selected_seed": {
            "posture_name": result.selected_seed.posture_name,
            "approach_name": result.selected_seed.approach_name,
            "wrist_position": [float(x) for x in result.selected_seed.wrist_position.tolist()],
            "wrist_quaternion_wxyz": _quat_wxyz_from_xyzw(
                result.selected_seed.wrist_quaternion_xyzw.tolist()
            ),
            "wrist_quaternion_xyzw": [
                float(x) for x in result.selected_seed.wrist_quaternion_xyzw.tolist()
            ],
            "joint_positions": {
                joint_name: float(value)
                for joint_name, value in result.selected_seed.joint_positions.items()
            },
            "metadata": result.selected_seed.metadata,
        },
    }
