from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.types_config import (
    AssetConfig,
    Category,
    CategoryConfig,
    CategoryContactUsage,
    CategorySeedConfig,
    CollisionConfig,
    CollisionSphere,
    ContactLogicConfig,
    ContactPointsConfig,
    FilteringLogicConfig,
    FrameConvention,
    HandConfig,
    JointConfig,
    LinkTips,
    LinksConfig,
    OptimizationLogicConfig,
    OptimizerConfig,
    PoseSeedsConfig,
    RegionProposalConfig,
    ResolvedHandRuntimeConfig,
    RootConfig,
    SeedAdjustment,
    SeedLogicConfig,
    SemanticPoint,
    Side,
    WristPerturbation,
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return (base_dir / value).resolve()


def load_hand_config(path: Path) -> HandConfig:
    raw = _read_yaml(path)
    base_dir = path.parent

    asset = AssetConfig(
        usd_path=(base_dir / raw["asset"]["usd_path"]).resolve(),
        urdf_path=_resolve_optional_path(base_dir, raw["asset"].get("urdf_path")),
    )

    root = RootConfig(
        wrist_link=raw["root"]["wrist_link"],
        palm_link=raw["root"]["palm_link"],
    )

    frame = FrameConvention(
        palm_normal_local=raw["frame_convention"]["palm_normal_local"],
        finger_forward_local=raw["frame_convention"]["finger_forward_local"],
        thumb_opposition_local=raw["frame_convention"]["thumb_opposition_local"],
    )

    tips_raw = raw["links"]["tips"]
    tips = LinkTips(
        thumb=tips_raw.get("thumb", ""),
        index=tips_raw.get("index", ""),
        middle=tips_raw.get("middle", ""),
        ring=tips_raw.get("ring", ""),
        little=tips_raw.get("little", ""),
    )

    links = LinksConfig(
        tips=tips,
        contact_candidate_links=raw["links"]["contact_candidate_links"],
        collision_link_names=raw["links"]["collision_link_names"],
    )

    limits: Dict[str, Tuple[float, float]] = {
        joint: (float(v[0]), float(v[1]))
        for joint, v in raw["joints"]["limits"].items()
    }

    joints = JointConfig(
        controllable=raw["joints"]["controllable"],
        groups=raw["joints"]["groups"],
        limits=limits,
    )

    return HandConfig(
        side=raw["side"],
        asset=asset,
        root=root,
        frame_convention=frame,
        links=links,
        joints=joints,
        default_postures=raw["default_postures"],
    )


def load_collision_config(path: Path) -> CollisionConfig:
    raw = _read_yaml(path)

    ignore = raw["self_collision"]["ignore"]
    buffer = raw["self_collision"].get("buffer", {})

    spheres_by_link: Dict[str, List[CollisionSphere]] = {}
    sphere_block = raw["geometry"]["collision_spheres"]["spheres"]
    for link_name, sphere_list in sphere_block.items():
        spheres_by_link[link_name] = [
            CollisionSphere(center=s["center"], radius=float(s["radius"]))
            for s in sphere_list
        ]

    return CollisionConfig(
        default_joint_positions=raw["default_joint_positions"],
        self_collision_ignore=ignore,
        self_collision_buffer=buffer,
        spheres_by_link=spheres_by_link,
    )


def load_contact_points_config(path: Path) -> ContactPointsConfig:
    raw = _read_yaml(path)

    def parse_points(block: Dict[str, Any]) -> Dict[str, SemanticPoint]:
        return {
            name: SemanticPoint(
                source_type=v["source_type"],
                source_link=v["source_link"],
                source_sphere_index=int(v["source_sphere_index"]),
                role_tags=v.get("role_tags", []),
            )
            for name, v in block.items()
        }

    left_points = parse_points(raw["left_points"])
    right_points = parse_points(raw["right_points"])

    categories: Dict[Category, CategoryContactUsage] = {}
    for cat in ("cat1", "cat2", "cat3", "cat4"):
        categories[cat] = CategoryContactUsage(
            active_points=raw[cat]["active_points"],
            avoid_points=raw[cat]["avoid_points"],
            opposition_pairs=[tuple(x) for x in raw[cat]["opposition_pairs"]],
        )

    return ContactPointsConfig(
        left_points=left_points,
        right_points=right_points,
        categories=categories,
    )


def load_pose_seeds_config(path: Path) -> PoseSeedsConfig:
    raw = _read_yaml(path)

    def parse_adjustment(block: Dict[str, Any]) -> SeedAdjustment:
        return SeedAdjustment(
            wrist_rpy_offset_deg=block["wrist_rpy_offset_deg"],
            position_offset_local=block["position_offset_local"],
        )

    def parse_cat_seed(block: Dict[str, Any]) -> CategorySeedConfig:
        wp = WristPerturbation(
            rpy_offset_deg=block["wrist_perturbation"]["rpy_offset_deg"],
            xyz_offset_local=block["wrist_perturbation"]["xyz_offset_local"],
        )
        return CategorySeedConfig(
            seed_template=block["seed_template"],
            joint_seed_postures=block["joint_seed_postures"],
            default_num_seeds_per_region=int(block["default_num_seeds_per_region"]),
            approach_family=block["approach_family"],
            wrist_perturbation=wp,
        )

    categories: Dict[Category, CategorySeedConfig] = {}
    for cat in ("cat1", "cat2", "cat3", "cat4"):
        categories[cat] = parse_cat_seed(raw[cat])

    return PoseSeedsConfig(
        available_postures=raw["shared_posture_names"]["available"],
        left_seed_adjustment=parse_adjustment(raw["left_seed_adjustment"]),
        right_seed_adjustment=parse_adjustment(raw["right_seed_adjustment"]),
        categories=categories,
    )


def load_category_config(path: Path) -> Dict[Category, CategoryConfig]:
    raw = _read_yaml(path)
    result: Dict[Category, CategoryConfig] = {}

    for cat in ("cat1", "cat2", "cat3", "cat4"):
        cat_raw = raw[cat]

        region_mode = cat_raw["region_proposal"]["mode"]
        region_params = {
            k: v for k, v in cat_raw["region_proposal"].items() if k != "mode"
        }

        contact_template = cat_raw["contact_logic"]["contact_template"]
        contact_params = {
            k: v for k, v in cat_raw["contact_logic"].items() if k != "contact_template"
        }

        seed_template = cat_raw["seed_logic"]["seed_template"]

        result[cat] = CategoryConfig(
            region_proposal=RegionProposalConfig(
                mode=region_mode,
                params=region_params,
            ),
            contact_logic=ContactLogicConfig(
                contact_template=contact_template,
                params=contact_params,
            ),
            seed_logic=SeedLogicConfig(
                seed_template=seed_template,
                default_num_seeds_per_region=int(
                    cat_raw["seed_logic"]["default_num_seeds_per_region"]
                ),
                approach_family=cat_raw["seed_logic"]["approach_family"],
            ),
            optimization_logic=OptimizationLogicConfig(
                objective_template=cat_raw["optimization_logic"]["objective_template"]
            ),
            filtering=FilteringLogicConfig(
                params=cat_raw["filtering"]
            ),
        )

    return result


def load_optimizer_config(path: Path) -> OptimizerConfig:
    raw = _read_yaml(path)
    by_category: Dict[Category, Dict[str, Any]] = {}
    for cat in ("cat1", "cat2", "cat3", "cat4"):
        by_category[cat] = raw.get(cat, {})
    return OptimizerConfig(global_cfg=raw.get("global", {}), by_category=by_category)


def load_all_for_hand(
    hand_dir: Path,
    global_config_dir: Path,
    side: Side,
    category: Category,
) -> ResolvedHandRuntimeConfig:
    side_hand_file = hand_dir / "config" / f"{side}_hand.yaml"
    side_collision_file = hand_dir / "collision" / f"{side}_collision.yaml"
    contact_file = hand_dir / "config" / "contact_points.yaml"
    seeds_file = hand_dir / "config" / "pose_seeds.yaml"
    category_file = global_config_dir / "category_config.yaml"
    optimizer_file = global_config_dir / "optimizer.yaml"

    hand = load_hand_config(side_hand_file)
    collision = load_collision_config(side_collision_file)
    contact_cfg = load_contact_points_config(contact_file)
    seeds_cfg = load_pose_seeds_config(seeds_file)
    category_cfg_map = load_category_config(category_file)
    optimizer_cfg = load_optimizer_config(optimizer_file)

    semantic_points = (
        contact_cfg.left_points if side == "left" else contact_cfg.right_points
    )

    return ResolvedHandRuntimeConfig(
        side=side,
        hand=hand,
        collision=collision,
        semantic_points=semantic_points,
        contact_usage=contact_cfg.categories[category],
        seed_cfg=seeds_cfg.categories[category],
        seed_adjustment=(
            seeds_cfg.left_seed_adjustment
            if side == "left"
            else seeds_cfg.right_seed_adjustment
        ),
        category_cfg=category_cfg_map[category],
        optimizer_cfg={
            "global": optimizer_cfg.global_cfg,
            "category": optimizer_cfg.by_category[category],
        },
    )
