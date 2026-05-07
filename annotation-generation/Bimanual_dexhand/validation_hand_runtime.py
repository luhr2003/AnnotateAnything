from __future__ import annotations

import math
from typing import Callable

from pxr import PhysxSchema, Usd, UsdPhysics


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def resolve_authored_hand_structure(
    stage,
    container_path: str,
    *,
    palm_link_name: str,
    controllable_joint_names: list[str],
    cache: dict | None = None,
    logger: Callable[[str], None] | None = None,
):
    """
    Resolve the palm prim and controllable-joint prims from an authored hand USD.

    This mirrors the newer dex3_1 validation flow where we preserve the authored
    collision geometry and drive the authored joints directly instead of routing
    through the older generic collision/setup path.
    """
    cache = cache if cache is not None else {}
    cache_key = (
        stage.GetRootLayer().identifier,
        container_path,
        palm_link_name,
        tuple(controllable_joint_names),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        palm_path = cached.get("palm_path")
        ordered_joint_paths = cached.get("ordered_joint_paths", [])
        if palm_path and stage.GetPrimAtPath(palm_path).IsValid():
            if all(stage.GetPrimAtPath(path).IsValid() for path in ordered_joint_paths):
                return cached

    container_prim = stage.GetPrimAtPath(container_path)
    if not container_prim.IsValid():
        _log(logger, f"[WARN] Invalid authored hand container path: {container_path}")
        return None

    palm_path = None
    joint_paths_by_name = {}
    required_joint_names = set(controllable_joint_names)

    for prim in Usd.PrimRange(container_prim):
        if not prim.IsValid():
            continue
        name = prim.GetName()
        if palm_path is None and name == palm_link_name:
            palm_path = prim.GetPath().pathString
        if prim.IsA(UsdPhysics.Joint) and name in required_joint_names and name not in joint_paths_by_name:
            joint_paths_by_name[name] = prim.GetPath().pathString

    missing = [name for name in controllable_joint_names if name not in joint_paths_by_name]
    if palm_path is None:
        _log(logger, f"[WARN] Could not find palm link '{palm_link_name}' under {container_path}")
        return None
    if missing:
        _log(logger, f"[WARN] Missing authored hand joints under {container_path}: {missing}")
        return None

    structure = {
        "palm_path": palm_path,
        "joint_paths_by_name": joint_paths_by_name,
        "ordered_joint_paths": [joint_paths_by_name[name] for name in controllable_joint_names],
    }
    cache[cache_key] = structure
    return structure


def configure_authored_hand_joint_drives(
    stage,
    container_path: str,
    *,
    palm_link_name: str,
    controllable_joint_names: list[str],
    joint_stiffness: float,
    joint_damping: float,
    joint_max_force: float,
    joint_armature: float,
    joint_velocity_limit: float,
    cache: dict | None = None,
    logger: Callable[[str], None] | None = None,
):
    structure = resolve_authored_hand_structure(
        stage,
        container_path,
        palm_link_name=palm_link_name,
        controllable_joint_names=controllable_joint_names,
        cache=cache,
        logger=logger,
    )
    if structure is None:
        return None

    joint_count = 0
    for joint_path in structure["ordered_joint_paths"]:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            continue

        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            drive_kind = "angular"
            stiffness = float(joint_stiffness) * math.pi / 180.0
            damping = float(joint_damping) * math.pi / 180.0
            max_velocity = float(joint_velocity_limit) * 180.0 / math.pi
        elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
            drive_kind = "linear"
            stiffness = float(joint_stiffness)
            damping = float(joint_damping)
            max_velocity = float(joint_velocity_limit)
        else:
            continue

        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_kind)
        (drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()).Set(stiffness)
        (drive.GetDampingAttr() or drive.CreateDampingAttr()).Set(damping)
        (drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()).Set(float(joint_max_force))
        (drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()).Set(0.0)
        (drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr()).Set(0.0)

        if not joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
            PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        physx_joint = PhysxSchema.PhysxJointAPI(joint_prim)
        (physx_joint.GetMaxJointVelocityAttr() or physx_joint.CreateMaxJointVelocityAttr()).Set(max_velocity)
        (physx_joint.GetArmatureAttr() or physx_joint.CreateArmatureAttr()).Set(float(joint_armature))
        joint_count += 1

    _log(logger, f"Configured authored joint drives for {joint_count} joints under {container_path}")
    return structure


def set_authored_hand_joint_targets(
    stage,
    container_path: str,
    joint_positions_rad: dict[str, float],
    *,
    palm_link_name: str,
    controllable_joint_names: list[str],
    set_joint_drive_target_fn,
    cache: dict | None = None,
    logger: Callable[[str], None] | None = None,
):
    structure = resolve_authored_hand_structure(
        stage,
        container_path,
        palm_link_name=palm_link_name,
        controllable_joint_names=controllable_joint_names,
        cache=cache,
        logger=logger,
    )
    if structure is None:
        return False

    for joint_name in controllable_joint_names:
        if joint_name not in joint_positions_rad:
            continue
        joint_path = structure["joint_paths_by_name"][joint_name]
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            continue

        target_value = float(joint_positions_rad[joint_name])
        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            target_value *= 57.29577951308232

        if not set_joint_drive_target_fn(joint_prim, target_value):
            _log(logger, f"[WARN] Failed to set authored joint target for {joint_path}")

    return True
