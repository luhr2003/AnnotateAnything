from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from simulation_app_runtime import create_simulation_app, cleanup_simulation_runtime_cache

np = None
Rotation = None


def _import_runtime_math() -> None:
    global np
    global Rotation

    import numpy as _np
    from scipy.spatial.transform import Rotation as _Rotation

    np = _np
    Rotation = _Rotation


def _effective_num_surface_points(category: str, requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    if str(category).lower() == "cat4":
        return 20000
    return 100000


def _clear_prim_if_exists(stage, prim_path: str) -> None:
    from pxr import Sdf

    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def _ensure_world(stage) -> None:
    from pxr import UsdGeom

    world = stage.GetPrimAtPath("/World")
    if not world or not world.IsValid():
        UsdGeom.Xform.Define(stage, "/World")


def _set_translate_orient_ops(xformable, position: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()

    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        translate_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    else:
        translate_op.Set(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        orient_op.Set(Gf.Quatd(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    else:
        orient_op.Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))


def _quat_xyzw_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_quat()


def _quat_wxyz_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    q = _quat_xyzw_from_rotation_matrix(rotation_matrix)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _rotation_matrix_from_quat_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64)).as_matrix()


def _gf_matrix_to_np(mat) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = mat[i][j]
    return out.T


def _make_transform(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_matrix_from_quat_xyzw(quaternion_xyzw)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _add_reference(stage, usd_path: Path, prim_path: str):
    from isaacsim.core.utils.stage import add_reference_to_stage

    return add_reference_to_stage(
        usd_path=str(Path(usd_path).resolve()),
        prim_path=prim_path,
    )


def _set_joint_drive_target(joint_prim, target_position: float) -> bool:
    from pxr import UsdPhysics

    if not joint_prim.IsValid():
        return False
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        drive_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        drive_kind = "linear"
    else:
        return False

    try:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_kind)
        target_attr = drive.GetTargetPositionAttr()
        if not target_attr or not target_attr.IsValid():
            target_attr = drive.CreateTargetPositionAttr()
        target_attr.Set(float(target_position))
        return True
    except Exception:
        return False


def _set_joint_state_position(joint_prim, target_position: float) -> bool:
    from pxr import PhysxSchema, UsdPhysics

    if not joint_prim.IsValid():
        return False
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        state_kind = "angular"
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        state_kind = "linear"
    else:
        return False

    try:
        joint_state = PhysxSchema.JointStateAPI.Apply(joint_prim, state_kind)
        position_attr = joint_state.GetPositionAttr()
        if not position_attr or not position_attr.IsValid():
            position_attr = joint_state.CreatePositionAttr()
        velocity_attr = joint_state.GetVelocityAttr()
        if not velocity_attr or not velocity_attr.IsValid():
            velocity_attr = joint_state.CreateVelocityAttr()
        position_attr.Set(float(target_position))
        velocity_attr.Set(0.0)
        return True
    except Exception:
        return False


def _set_joint_target_fallback(joint_prim, target_position: float) -> bool:
    matched = False
    for prop in joint_prim.GetProperties():
        name = prop.GetName()
        if "drive" in name and "targetPosition" in name:
            prop.Set(float(target_position))
            matched = True
    return matched


def _set_hand_joint_targets(stage, hand_prim_path: str, joint_targets_rad: Dict[str, float]) -> Dict[str, Iterable[str]]:
    from pxr import Usd, UsdPhysics

    hand_prim = stage.GetPrimAtPath(hand_prim_path)
    if not hand_prim or not hand_prim.IsValid():
        raise RuntimeError(f"Invalid hand prim path: {hand_prim_path}")

    joint_map = {}
    for prim in Usd.PrimRange(hand_prim):
        if prim.IsA(UsdPhysics.Joint):
            joint_map[prim.GetName()] = prim

    updated = []
    state_updated = []
    missing = []
    for joint_name, target_rad in joint_targets_rad.items():
        joint_prim = joint_map.get(joint_name)
        if joint_prim is None:
            missing.append(joint_name)
            continue

        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            target_value = float(np.rad2deg(target_rad))
        else:
            target_value = float(target_rad)

        if not _set_joint_drive_target(joint_prim, target_value):
            if not _set_joint_target_fallback(joint_prim, target_value):
                missing.append(joint_name)
                continue
        if _set_joint_state_position(joint_prim, target_value):
            state_updated.append(joint_name)
        updated.append(joint_name)

    return {
        "updated_joint_names": updated,
        "state_updated_joint_names": state_updated,
        "missing_joint_names": missing,
    }


def _find_descendant_prim_by_name(root_prim, name: str):
    from pxr import Usd

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return prim
    return None


def _set_hand_container_from_wrist_pose(
    stage,
    *,
    hand_prim_path: str,
    wrist_link_name: str,
    wrist_position: np.ndarray,
    wrist_quaternion_xyzw: np.ndarray,
) -> Dict[str, str]:
    from pxr import UsdGeom

    hand_prim = stage.GetPrimAtPath(hand_prim_path)
    if not hand_prim or not hand_prim.IsValid():
        raise RuntimeError(f"Invalid hand prim path: {hand_prim_path}")

    wrist_prim = _find_descendant_prim_by_name(hand_prim, wrist_link_name)
    if wrist_prim is None or not wrist_prim.IsValid():
        _set_translate_orient_ops(
            UsdGeom.Xformable(hand_prim),
            wrist_position,
            wrist_quaternion_xyzw,
        )
        return {
            "hand_prim_path": hand_prim_path,
            "wrist_prim_path": "",
            "alignment_mode": "direct_root_pose_fallback",
        }

    hand_world = _gf_matrix_to_np(UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(0.0))
    wrist_world = _gf_matrix_to_np(UsdGeom.Xformable(wrist_prim).ComputeLocalToWorldTransform(0.0))
    hand_to_wrist = np.linalg.inv(hand_world) @ wrist_world

    desired_wrist_world = _make_transform(wrist_position, wrist_quaternion_xyzw)
    desired_hand_world = desired_wrist_world @ np.linalg.inv(hand_to_wrist)

    desired_hand_quat_xyzw = _quat_xyzw_from_rotation_matrix(desired_hand_world[:3, :3])
    _set_translate_orient_ops(
        UsdGeom.Xformable(hand_prim),
        desired_hand_world[:3, 3],
        desired_hand_quat_xyzw,
    )
    return {
        "hand_prim_path": hand_prim_path,
        "wrist_prim_path": str(wrist_prim.GetPath()),
        "alignment_mode": "wrist_link_aligned",
    }


def _visualize_selected_anchor(stage, anchor, prefix: str) -> None:
    from pxr import Gf, UsdGeom

    prim_path = f"{prefix}/Anchor"
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(0.01)
    _set_translate_orient_ops(
        UsdGeom.Xformable(sphere.GetPrim()),
        np.asarray(anchor.point, dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])


def _visualize_surface_subsample(stage, proposal_result, prefix: str, *, max_display: int = 2000) -> None:
    from pxr import Gf, Vt, UsdGeom

    pts = proposal_result.samples.points
    if len(pts) > max_display:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=max_display, replace=False)
        pts = pts[idx]

    points_prim = UsdGeom.Points.Define(stage, f"{prefix}/Surface")
    points_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
    )
    points_prim.GetWidthsAttr().Set(Vt.FloatArray([0.0035] * len(pts)))
    points_prim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.4, 0.8, 1.0)]))


def _visualize_stage_targets(stage, prefix: str, runtime_cfg, contact_result, hand_pose, stage_name: str) -> Dict[str, float]:
    from pxr import Gf, Vt, UsdGeom

    from src.hand_kinematics import load_hand_kinematics_model
    from src.optimizer_core import _target_desired_center

    hand_model = load_hand_kinematics_model(runtime_cfg)
    semantic_states = hand_model.semantic_points_world(
        runtime_cfg,
        hand_pose,
        point_names=[t.name for t in contact_result.active_targets],
    )

    target_points = [np.asarray(t.target_point, dtype=np.float64) for t in contact_result.active_targets]
    desired_centers = []
    actual_centers = []
    for target in contact_result.active_targets:
        sphere_state = semantic_states.get(target.name)
        if sphere_state is None:
            continue
        desired_centers.append(
            _target_desired_center(
                target,
                sphere_state,
                stage_name,
                runtime_cfg=runtime_cfg,
                contact_result=contact_result,
            )
        )
        actual_centers.append(np.asarray(sphere_state.center_world, dtype=np.float64))

    target_prim = UsdGeom.Points.Define(stage, f"{prefix}/TargetPoints")
    target_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in target_points])
    )
    target_prim.GetWidthsAttr().Set(Vt.FloatArray([0.01] * len(target_points)))
    target_prim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.2, 1.0, 0.2)]))

    desired_prim = UsdGeom.Points.Define(stage, f"{prefix}/DesiredCenters")
    desired_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in desired_centers])
    )
    desired_prim.GetWidthsAttr().Set(Vt.FloatArray([0.009] * len(desired_centers)))
    desired_prim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.1, 1.0, 1.0)]))

    actual_prim = UsdGeom.Points.Define(stage, f"{prefix}/ActualCenters")
    actual_prim.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in actual_centers])
    )
    actual_prim.GetWidthsAttr().Set(Vt.FloatArray([0.009] * len(actual_centers)))
    actual_prim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.8, 0.1)]))

    metrics = {
        "num_active_targets_visualized": float(len(actual_centers)),
        "mean_center_error": 0.0,
        "max_center_error": 0.0,
    }
    if len(actual_centers) == len(desired_centers) and len(actual_centers) > 0:
        errors = np.linalg.norm(
            np.asarray(actual_centers, dtype=np.float64)
            - np.asarray(desired_centers, dtype=np.float64),
            axis=1,
        )
        metrics["mean_center_error"] = float(np.mean(errors))
        metrics["max_center_error"] = float(np.max(errors))
    return metrics


def _joint_degrees(joint_positions_rad: Dict[str, float]) -> Dict[str, float]:
    return {
        joint_name: float(np.rad2deg(value))
        for joint_name, value in joint_positions_rad.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview an optimized staged single-hand pose in Isaac Sim and print its joint targets."
    )
    parser.add_argument("--hand_dir", type=Path, required=True)
    parser.add_argument("--config_dir", type=Path, required=True)
    parser.add_argument("--object_usd", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--category", choices=["cat1", "cat2", "cat3", "cat4"], required=True)

    parser.add_argument("--root_prim_path", type=str, default=None)
    parser.add_argument("--exclude_prim_paths", type=str, nargs="*", default=None)
    parser.add_argument("--num_surface_points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k_anchors", type=int, default=5)
    parser.add_argument("--num_seeds_per_contact", type=int, default=None)
    parser.add_argument("--top_k_seeds_per_contact", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=0)
    parser.add_argument("--bundle_rank", type=int, default=0)
    parser.add_argument("--result_rank", type=int, default=0)
    parser.add_argument("--stage", choices=["pregrasp", "grasp", "squeeze"], default="grasp")
    parser.add_argument("--sdf_cache_path", type=Path, default=None)
    parser.add_argument("--force_rebuild_sdf", action="store_true")
    parser.add_argument("--object_prim_path", type=str, default="/World/Object")
    parser.add_argument("--hand_prim_path", type=str, default="/World/HandStagePose")
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save_json", type=Path, default=None)

    args = parser.parse_args()
    simulation_app = create_simulation_app(headless=args.headless, script_name="run_single_hand_staged_pose_preview")

    try:
        _import_runtime_math()
        import omni.kit.app
        import omni.usd
        from pxr import UsdGeom

        from src.optimizer_core import summarize_staged_grasp_result
        from src.staged_grasp_pipeline import run_single_hand_staged

        num_seeds_per_contact = (
            args.num_seeds_per_contact
            if args.num_seeds_per_contact is not None
            else max(1, int(args.top_k_seeds_per_contact))
        )
        num_surface_points = _effective_num_surface_points(args.category, args.num_surface_points)

        pipeline_result = run_single_hand_staged(
            hand_dir=args.hand_dir,
            global_config_dir=args.config_dir,
            side=args.side,
            category=args.category,
            object_usd=args.object_usd,
            root_prim_path=args.root_prim_path,
            exclude_prim_paths=args.exclude_prim_paths,
            num_surface_points=num_surface_points,
            seed=args.seed,
            top_k_anchors=args.top_k_anchors,
            num_seeds_per_contact=num_seeds_per_contact,
            top_k_seeds_per_contact=args.top_k_seeds_per_contact,
            max_workers=args.max_workers,
            sdf_cache_path=args.sdf_cache_path,
            force_rebuild_sdf=args.force_rebuild_sdf,
        )

        if not pipeline_result.bundles:
            raise RuntimeError("No optimized contact bundles were produced.")

        bundle_rank = int(max(0, min(args.bundle_rank, len(pipeline_result.bundles) - 1)))
        bundle = pipeline_result.bundles[bundle_rank]
        if not bundle.optimized_results:
            raise RuntimeError(f"Selected bundle {bundle_rank} has no optimized staged results.")

        result_rank = int(max(0, min(args.result_rank, len(bundle.optimized_results) - 1)))
        staged_result = bundle.optimized_results[result_rank]
        stage_result = staged_result.stage_results[args.stage]
        hand_pose = stage_result.hand_pose

        summary = {
            "side": pipeline_result.proposal_result.side,
            "category": pipeline_result.proposal_result.category,
            "object_usd": str(pipeline_result.proposal_result.object_usd),
            "sdf_cache_path": str(pipeline_result.sdf_cache_path),
            "pipeline_metadata": pipeline_result.metadata,
            "bundle_rank": bundle_rank,
            "result_rank": result_rank,
            "selected_stage": args.stage,
            "contact_anchor_score": float(bundle.contact_result.metadata.get("anchor_score", 0.0)),
            "contact_mode": bundle.contact_result.metadata.get("cat1_contact_mode"),
            "contact_mode_source": bundle.contact_result.metadata.get("cat1_contact_mode_source"),
            "contact_local_width": bundle.contact_result.metadata.get("cat1_local_width"),
            "selected_result": summarize_staged_grasp_result(staged_result),
            "selected_stage_pose": {
                "success": bool(stage_result.success),
                "cost": float(stage_result.cost),
                "message": stage_result.message,
                "wrist_position": [float(x) for x in np.asarray(hand_pose.wrist_position, dtype=np.float64).tolist()],
                "wrist_quaternion_wxyz": [
                    float(x) for x in _quat_wxyz_from_rotation_matrix(hand_pose.rotation_matrix()).tolist()
                ],
                "wrist_quaternion_xyzw": [
                    float(x) for x in _quat_xyzw_from_rotation_matrix(hand_pose.rotation_matrix()).tolist()
                ],
                "joint_positions_rad": {
                    joint_name: float(value) for joint_name, value in hand_pose.joint_positions.items()
                },
                "joint_positions_deg": _joint_degrees(hand_pose.joint_positions),
            },
        }

        stage = omni.usd.get_context().get_stage()
        _ensure_world(stage)
        _clear_prim_if_exists(stage, args.object_prim_path)
        _clear_prim_if_exists(stage, args.hand_prim_path)
        _clear_prim_if_exists(stage, "/World/DebugStagePreview")
        UsdGeom.Xform.Define(stage, "/World/DebugStagePreview")

        _add_reference(stage, args.object_usd, args.object_prim_path)
        _add_reference(stage, pipeline_result.proposal_result.runtime_cfg.hand.asset.usd_path, args.hand_prim_path)

        hand_alignment_info = _set_hand_container_from_wrist_pose(
            stage,
            hand_prim_path=args.hand_prim_path,
            wrist_link_name=pipeline_result.proposal_result.runtime_cfg.hand.root.wrist_link,
            wrist_position=hand_pose.wrist_position,
            wrist_quaternion_xyzw=_quat_xyzw_from_rotation_matrix(hand_pose.rotation_matrix()),
        )
        joint_update_info = _set_hand_joint_targets(
            stage,
            args.hand_prim_path,
            hand_pose.joint_positions,
        )
        summary["hand_alignment_info"] = hand_alignment_info
        summary["joint_update_info"] = joint_update_info

        debug_prefix = "/World/DebugStagePreview"
        _visualize_surface_subsample(stage, pipeline_result.proposal_result, debug_prefix)
        _visualize_selected_anchor(stage, bundle.contact_result.anchor, debug_prefix)
        summary["stage_alignment_metrics"] = _visualize_stage_targets(
            stage,
            debug_prefix,
            pipeline_result.proposal_result.runtime_cfg,
            bundle.contact_result,
            hand_pose,
            args.stage,
        )

        print(json.dumps(summary, indent=2))

        if args.save_json is not None:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            with args.save_json.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        app = omni.kit.app.get_app()
        for _ in range(max(0, int(args.settle_steps))):
            app.update()

        if not args.headless:
            print("Stage pose preview ready in Isaac Sim GUI. Close the app window to exit.")
            while simulation_app.is_running():
                app.update()
    finally:
        simulation_app.close()
        cleanup_simulation_runtime_cache()


if __name__ == "__main__":
    main()
