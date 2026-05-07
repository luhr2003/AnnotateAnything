from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


Side = Literal["left", "right"]
Category = Literal["cat1", "cat2", "cat3", "cat4"]


@dataclass
class AssetConfig:
    usd_path: Path
    urdf_path: Optional[Path] = None


@dataclass
class RootConfig:
    wrist_link: str
    palm_link: str


@dataclass
class FrameConvention:
    palm_normal_local: List[float]
    finger_forward_local: List[float]
    thumb_opposition_local: List[float]


@dataclass
class LinkTips:
    thumb: str
    index: str
    middle: str
    ring: str = ""
    little: str = ""


@dataclass
class LinksConfig:
    tips: LinkTips
    contact_candidate_links: List[str]
    collision_link_names: List[str]


@dataclass
class JointConfig:
    controllable: List[str]
    groups: Dict[str, List[str]]
    limits: Dict[str, Tuple[float, float]]


@dataclass
class HandConfig:
    side: Side
    asset: AssetConfig
    root: RootConfig
    frame_convention: FrameConvention
    links: LinksConfig
    joints: JointConfig
    default_postures: Dict[str, Dict[str, float]]


@dataclass
class CollisionSphere:
    center: List[float]
    radius: float


@dataclass
class CollisionConfig:
    default_joint_positions: Dict[str, float]
    self_collision_ignore: Dict[str, List[str]]
    self_collision_buffer: Dict[str, float]
    spheres_by_link: Dict[str, List[CollisionSphere]]


@dataclass
class SemanticPoint:
    source_type: str
    source_link: str
    source_sphere_index: int
    role_tags: List[str]


@dataclass
class CategoryContactUsage:
    active_points: List[str]
    avoid_points: List[str]
    opposition_pairs: List[Tuple[str, str]]


@dataclass
class ContactPointsConfig:
    left_points: Dict[str, SemanticPoint]
    right_points: Dict[str, SemanticPoint]
    categories: Dict[Category, CategoryContactUsage]


@dataclass
class WristPerturbation:
    rpy_offset_deg: Dict[str, List[float]]
    xyz_offset_local: Dict[str, List[float]]


@dataclass
class SeedAdjustment:
    wrist_rpy_offset_deg: List[float]
    position_offset_local: List[float]


@dataclass
class CategorySeedConfig:
    seed_template: str
    joint_seed_postures: List[str]
    default_num_seeds_per_region: int
    approach_family: List[str]
    wrist_perturbation: WristPerturbation


@dataclass
class PoseSeedsConfig:
    available_postures: List[str]
    left_seed_adjustment: SeedAdjustment
    right_seed_adjustment: SeedAdjustment
    categories: Dict[Category, CategorySeedConfig]


@dataclass
class RegionProposalConfig:
    mode: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContactLogicConfig:
    contact_template: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedLogicConfig:
    seed_template: str
    default_num_seeds_per_region: int
    approach_family: List[str]


@dataclass
class OptimizationLogicConfig:
    objective_template: str


@dataclass
class FilteringLogicConfig:
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryConfig:
    region_proposal: RegionProposalConfig
    contact_logic: ContactLogicConfig
    seed_logic: SeedLogicConfig
    optimization_logic: OptimizationLogicConfig
    filtering: FilteringLogicConfig


@dataclass
class OptimizerConfig:
    global_cfg: Dict[str, Any]
    by_category: Dict[Category, Dict[str, Any]]


@dataclass
class ResolvedHandRuntimeConfig:
    side: Side
    hand: HandConfig
    collision: CollisionConfig
    semantic_points: Dict[str, SemanticPoint]
    contact_usage: CategoryContactUsage
    seed_cfg: CategorySeedConfig
    seed_adjustment: SeedAdjustment
    category_cfg: CategoryConfig
    optimizer_cfg: Dict[str, Any]
