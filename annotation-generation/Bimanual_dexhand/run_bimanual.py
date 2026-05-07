from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
import shutil
import sys
import traceback

from simulation_app_runtime import create_simulation_app, cleanup_simulation_runtime_cache

np = None
_add_reference = None
_clear_prim_if_exists = None
_ensure_world = None
_quat_xyzw_from_rotation_matrix = None
_set_hand_container_from_wrist_pose = None
_set_hand_joint_targets = None
_set_translate_orient_ops = None
_visualize_stage_targets = None
_visualize_surface_subsample = None


def _project_cache_root() -> Path:
    return (Path(__file__).resolve().parent / ".cache").resolve()


def _clear_project_cache() -> None:
    cache_root = _project_cache_root()
    if not cache_root.exists():
        return
    shutil.rmtree(cache_root, ignore_errors=True)


def _create_simulation_app(*, headless: bool):
    return create_simulation_app(headless=headless, script_name="run_bimanual")


def _import_runtime_helpers() -> None:
    global np
    global _add_reference
    global _clear_prim_if_exists
    global _ensure_world
    global _quat_xyzw_from_rotation_matrix
    global _set_hand_container_from_wrist_pose
    global _set_hand_joint_targets
    global _set_translate_orient_ops
    global _visualize_stage_targets
    global _visualize_surface_subsample

    import numpy as _np
    import run_single_hand_staged_pose_preview as _runtime_preview

    _runtime_preview._import_runtime_math()

    _runtime_add_reference = _runtime_preview._add_reference
    _runtime_clear_prim_if_exists = _runtime_preview._clear_prim_if_exists
    _runtime_ensure_world = _runtime_preview._ensure_world
    _runtime_quat_xyzw_from_rotation_matrix = _runtime_preview._quat_xyzw_from_rotation_matrix
    _runtime_set_hand_container_from_wrist_pose = _runtime_preview._set_hand_container_from_wrist_pose
    _runtime_set_hand_joint_targets = _runtime_preview._set_hand_joint_targets
    _runtime_set_translate_orient_ops = _runtime_preview._set_translate_orient_ops
    _runtime_visualize_stage_targets = _runtime_preview._visualize_stage_targets
    _runtime_visualize_surface_subsample = _runtime_preview._visualize_surface_subsample

    np = _np
    _add_reference = _runtime_add_reference
    _clear_prim_if_exists = _runtime_clear_prim_if_exists
    _ensure_world = _runtime_ensure_world
    _quat_xyzw_from_rotation_matrix = _runtime_quat_xyzw_from_rotation_matrix
    _set_hand_container_from_wrist_pose = _runtime_set_hand_container_from_wrist_pose
    _set_hand_joint_targets = _runtime_set_hand_joint_targets
    _set_translate_orient_ops = _runtime_set_translate_orient_ops
    _visualize_stage_targets = _runtime_visualize_stage_targets
    _visualize_surface_subsample = _runtime_visualize_surface_subsample


def _quat_wxyz_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    q = _quat_xyzw_from_rotation_matrix(rotation_matrix)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _effective_num_surface_points(category: str, requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    if str(category).lower() == "cat3":
        return 20000
    if str(category).lower() == "cat4":
        return 20000
    return 30000


def _effective_top_k_optimized_per_side(requested: int | None, top_k_seeds_per_contact: int) -> int:
    if requested is not None:
        return max(1, int(requested))
    return min(max(1, int(top_k_seeds_per_contact)), 4)


def _effective_max_result_pair_checks(requested: int | None, top_k_optimized_per_side: int) -> int:
    if requested is not None:
        return max(1, int(requested))
    return min(max(1, int(top_k_optimized_per_side)) ** 2, 24)


def _effective_export_speed_profile(requested: str | None, save_format: str) -> str:
    if requested is not None:
        return str(requested)
    return "safe" if str(save_format).lower() in {"validation", "both"} else "standard"


def _effective_parallel_backend(requested: str, *, headless: bool, save_format: str) -> str:
    resolved = str(requested).lower()
    if resolved != "auto":
        return resolved
    if bool(headless) and str(save_format).lower() in {"validation", "both"}:
        return "process"
    return "thread"


VALIDATION_CAT1_USE_FINE_POSE_FOR_COARSE = True
# Cat1/cat2 validation export uses a slightly tighter final close than raw
# squeeze, while still respecting per-joint limits.
VALIDATION_FINAL_JOINT_EXTRAPOLATION_CATEGORIES = {"cat1", "cat2"}
VALIDATION_FINAL_JOINT_EXTRAPOLATION_GAIN = 0.40


def _optimizer_overrides_for_export_speed_profile(profile: str, category: str) -> dict | None:
    resolved = str(profile).lower()
    if resolved == "standard":
        return None
    if resolved != "safe":
        raise ValueError(f"Unsupported export speed profile: {profile}")
    overrides = {
        "global": {
            "solver": {
                "max_iterations": 80,
                "ftol": 1.0e-6,
                "gtol": 1.0e-6,
                "maxls": 25,
            }
        },
        "category": {
            "stage_init": {
                "pregrasp_roll_num_steps": 9,
                "pregrasp_pitch_num_steps": 7,
                "grasp_refine_num_steps": 5,
                "squeeze_refine_num_steps": 5,
            },
            "final_contact_seek": {
                "max_attempts": 2,
                "max_iterations": 80,
                "stop_after_first_success": True,
            },
        },
    }
    if str(category).lower() == "cat2":
        # Cat2 remains heavier than cat1 because its support/stability terms and
        # contact-seek pass are more expensive. Keep export behavior safe, but
        # trim the heaviest refinement work further for validation-style runs.
        overrides["global"]["solver"]["max_iterations"] = 70
        overrides["category"]["stage_init"]["grasp_refine_num_steps"] = 3
        overrides["category"]["stage_init"]["squeeze_refine_num_steps"] = 3
        overrides["category"]["final_contact_seek"]["max_attempts"] = 1
        overrides["category"]["final_contact_seek"]["max_iterations"] = 50
    elif str(category).lower() == "cat3":
        # Cat3 side-push contacts are cheaper than edge/bottom grasps, so use
        # the fast validation profile without affecting cat1/cat2/cat4.
        overrides["global"]["solver"]["max_iterations"] = 60
        overrides["category"]["stage_init"]["grasp_refine_num_steps"] = 3
        overrides["category"]["stage_init"]["squeeze_refine_num_steps"] = 3
        overrides["category"]["final_contact_seek"]["max_attempts"] = 1
        overrides["category"]["final_contact_seek"]["max_iterations"] = 45
    return overrides


def _object_type_label(object_usd: Path) -> str:
    object_usd = object_usd.resolve()
    if object_usd.stem.lower() == "object":
        if object_usd.parent.parent.name:
            return object_usd.parent.parent.name
        if object_usd.parent.name:
            return object_usd.parent.name
    return object_usd.stem or "object"


def _bottom_center_from_result(result) -> list[float]:
    bbox_min = np.asarray(result.proposal_result.samples.bbox_min, dtype=np.float64)
    bbox_max = np.asarray(result.proposal_result.samples.bbox_max, dtype=np.float64)
    return [
        float(0.5 * (bbox_min[0] + bbox_max[0])),
        float(0.5 * (bbox_min[1] + bbox_max[1])),
        float(bbox_min[2]),
    ]


def _pack_legacy_stage(stage_result, joint_order: list[str]) -> dict:
    hand_pose = stage_result.hand_pose
    quat_wxyz = _quat_wxyz_from_rotation_matrix(hand_pose.rotation_matrix())
    return {
        "position": [
            float(x) for x in np.asarray(hand_pose.wrist_position, dtype=np.float64).tolist()
        ],
        "orientation": [
            float(x) for x in quat_wxyz.tolist()
        ],
        "joints": [
            float(hand_pose.joint_positions.get(joint_name, 0.0))
            for joint_name in joint_order
        ],
    }


def _stage_components(stage_result):
    hand_pose = stage_result.hand_pose
    quat_wxyz = _quat_wxyz_from_rotation_matrix(hand_pose.rotation_matrix())
    return (
        np.asarray(hand_pose.wrist_position, dtype=np.float64),
        np.asarray(quat_wxyz, dtype=np.float64),
        {str(k): float(v) for k, v in hand_pose.joint_positions.items()},
    )


def _pack_legacy_components(
    wrist_position: np.ndarray,
    wrist_quaternion_wxyz: np.ndarray,
    joint_positions: dict[str, float],
    joint_order: list[str],
) -> dict:
    return {
        "position": [float(x) for x in np.asarray(wrist_position, dtype=np.float64).tolist()],
        "orientation": [float(x) for x in np.asarray(wrist_quaternion_wxyz, dtype=np.float64).tolist()],
        "joints": [
            float(joint_positions.get(joint_name, 0.0))
            for joint_name in joint_order
        ],
    }


def _extrapolate_joint_positions(
    *,
    from_joints: dict[str, float],
    to_joints: dict[str, float],
    joint_limits: dict[str, tuple[float, float]],
    gain: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    all_names = set(from_joints.keys()) | set(to_joints.keys())
    for joint_name in all_names:
        start = float(from_joints.get(joint_name, 0.0))
        end = float(to_joints.get(joint_name, start))
        value = end + float(gain) * (end - start)
        limits = joint_limits.get(joint_name)
        if limits is not None:
            lo, hi = float(limits[0]), float(limits[1])
            value = float(np.clip(value, lo, hi))
        out[joint_name] = float(value)
    return out


def _pack_legacy_hand_grasp(staged_result, runtime_cfg, category: str) -> dict:
    joint_order = list(runtime_cfg.hand.joints.controllable)

    pre_pos, pre_quat, pre_joints = _stage_components(staged_result.stage_results["pregrasp"])
    grasp_pos, grasp_quat, grasp_joints = _stage_components(staged_result.stage_results["grasp"])
    squeeze_pos, squeeze_quat, squeeze_joints = _stage_components(staged_result.stage_results["squeeze"])

    coarse_pos = pre_pos
    coarse_quat = pre_quat
    coarse_joints = pre_joints
    final_joints = squeeze_joints

    category_key = str(category).lower()
    if category_key == "cat1":
        if VALIDATION_CAT1_USE_FINE_POSE_FOR_COARSE:
            coarse_pos = grasp_pos
            coarse_quat = grasp_quat
    if category_key in VALIDATION_FINAL_JOINT_EXTRAPOLATION_CATEGORIES:
        final_joints = _extrapolate_joint_positions(
            from_joints=grasp_joints,
            to_joints=squeeze_joints,
            joint_limits=runtime_cfg.hand.joints.limits,
            gain=VALIDATION_FINAL_JOINT_EXTRAPOLATION_GAIN,
        )
    elif category_key == "cat3":
        squeeze_pos = grasp_pos
        squeeze_quat = grasp_quat

    return {
        "coarse_grasp": _pack_legacy_components(coarse_pos, coarse_quat, coarse_joints, joint_order),
        "fine_grasp": _pack_legacy_components(grasp_pos, grasp_quat, grasp_joints, joint_order),
        "final_grasp": _pack_legacy_components(squeeze_pos, squeeze_quat, final_joints, joint_order),
    }


def _bimanual_result_to_validation_payload(result) -> dict:
    body = []
    category = str(result.proposal_result.category)
    for pair_bundle in result.pair_bundles:
        pair_entry = {}
        for side in ("left", "right"):
            side_bundle = pair_bundle.bundles_by_side.get(side)
            runtime_cfg = result.runtime_cfg_by_side.get(side)
            if side_bundle is None or runtime_cfg is None or not side_bundle.optimized_results:
                continue
            selected_rank = int(pair_bundle.metadata.get("selected_result_rank_by_side", {}).get(side, 0))
            selected_rank = max(0, min(selected_rank, len(side_bundle.optimized_results) - 1))
            staged_result = side_bundle.optimized_results[selected_rank]
            pair_entry[f"{side}_hand"] = _pack_legacy_hand_grasp(staged_result, runtime_cfg, category)
        if "left_hand" in pair_entry and "right_hand" in pair_entry:
            body.append(pair_entry)

    return {
        "type": _object_type_label(result.proposal_result.object_usd),
        "bottom_center": _bottom_center_from_result(result),
        "functional_grasp": {
            "body": body,
        },
        "grasp": {},
    }


def _visualize_selected_anchor(stage, anchor, prim_path: str, color_rgb: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(0.01)
    _set_translate_orient_ops(
        UsdGeom.Xformable(sphere.GetPrim()),
        np.asarray(anchor.point, dtype=np.float64),
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))])


def _candidate_pair_color(
    pair_index: int,
    total_pairs: int,
    passed: bool,
    *,
    evaluated: bool = True,
) -> tuple[float, float, float]:
    total_pairs = max(1, int(total_pairs))
    hue = (float(pair_index % total_pairs) / float(total_pairs)) % 1.0
    if not evaluated:
        saturation = 0.25
        value = 0.85
    else:
        saturation = 0.8 if passed else 0.45
        value = 1.0 if passed else 0.8
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _visualize_candidate_pairs(stage, candidate_pairs, prim_path: str) -> None:
    from pxr import Gf, UsdGeom, Vt

    UsdGeom.Xform.Define(stage, prim_path)
    total_pairs = len(candidate_pairs)
    for idx, pair_info in enumerate(candidate_pairs):
        evaluated = bool(pair_info.get("pair_evaluated", False))
        passed = bool(pair_info.get("passed_overlap_filter", False))
        color = _candidate_pair_color(idx, total_pairs, passed, evaluated=evaluated)
        primary_point = np.asarray(pair_info["primary_point"], dtype=np.float64)
        opposite_point = np.asarray(pair_info["opposite_point"], dtype=np.float64)
        pair_group = f"{prim_path}/Pair_{idx:03d}"
        UsdGeom.Xform.Define(stage, pair_group)

        for label, point in (("Primary", primary_point), ("Opposite", opposite_point)):
            sphere = UsdGeom.Sphere.Define(stage, f"{pair_group}/{label}")
            sphere.CreateRadiusAttr(0.0075 if evaluated and passed else 0.0065)
            _set_translate_orient_ops(
                UsdGeom.Xformable(sphere.GetPrim()),
                point,
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
            sphere.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
            )

        curve = UsdGeom.BasisCurves.Define(stage, f"{pair_group}/Link")
        curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr().Set(Vt.IntArray([2]))
        curve.CreatePointsAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(float(primary_point[0]), float(primary_point[1]), float(primary_point[2])),
                    Gf.Vec3f(float(opposite_point[0]), float(opposite_point[1]), float(opposite_point[2])),
                ]
            )
        )
        curve.CreateWidthsAttr().Set(Vt.FloatArray([0.0035 if passed else 0.0025, 0.0035 if passed else 0.0025]))
        curve.CreateDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        )

        midpoint = 0.5 * (primary_point + opposite_point)
        midpoint[2] += 0.01 + 0.003 * float(idx)
        midpoint_marker = UsdGeom.Sphere.Define(stage, f"{pair_group}/Midpoint")
        midpoint_marker.CreateRadiusAttr(0.004)
        _set_translate_orient_ops(
            UsdGeom.Xformable(midpoint_marker.GetPrim()),
            midpoint,
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        midpoint_marker.CreateDisplayColorAttr().Set(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )


def _visualize_candidate_pair_focus(stage, pair_info, prim_path: str) -> None:
    from pxr import Gf, UsdGeom, Vt

    pair_rank = int(pair_info.get("pair_rank", 0))
    evaluated = bool(pair_info.get("pair_evaluated", False))
    passed = bool(pair_info.get("passed_overlap_filter", False))
    color = _candidate_pair_color(pair_rank, max(1, pair_rank + 1), passed, evaluated=evaluated)
    primary_point = np.asarray(pair_info["primary_point"], dtype=np.float64)
    opposite_point = np.asarray(pair_info["opposite_point"], dtype=np.float64)
    UsdGeom.Xform.Define(stage, prim_path)

    for label, point in (("Primary", primary_point), ("Opposite", opposite_point)):
        sphere = UsdGeom.Sphere.Define(stage, f"{prim_path}/{label}")
        sphere.CreateRadiusAttr(0.011)
        _set_translate_orient_ops(
            UsdGeom.Xformable(sphere.GetPrim()),
            point,
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        sphere.CreateDisplayColorAttr().Set(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )

    curve = UsdGeom.BasisCurves.Define(stage, f"{prim_path}/Link")
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr().Set(Vt.IntArray([2]))
    curve.CreatePointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(float(primary_point[0]), float(primary_point[1]), float(primary_point[2])),
                Gf.Vec3f(float(opposite_point[0]), float(opposite_point[1]), float(opposite_point[2])),
            ]
        )
    )
    curve.CreateWidthsAttr().Set(Vt.FloatArray([0.006, 0.006]))
    curve.CreateDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    )


def _visualize_all_anchors(stage, anchors, prim_path: str) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, prim_path)
    anchors_sorted = sorted(anchors, key=lambda item: item.score, reverse=True)
    if not anchors_sorted:
        return

    max_score = max(float(anchor.score) for anchor in anchors_sorted)
    min_score = min(float(anchor.score) for anchor in anchors_sorted)
    score_span = max(max_score - min_score, 1e-8)

    for idx, anchor in enumerate(anchors_sorted):
        prim = UsdGeom.Sphere.Define(stage, f"{prim_path}/Anchor_{idx:03d}")
        prim.CreateRadiusAttr(0.0045)
        _set_translate_orient_ops(
            UsdGeom.Xformable(prim.GetPrim()),
            np.asarray(anchor.point, dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        t = (float(anchor.score) - min_score) / score_span
        prim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(t), 0.25, float(1.0 - t))])


def _visualize_paired_anchor_overlays(stage, candidate_pairs, prim_path: str) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, prim_path)
    total_pairs = len(candidate_pairs)
    for idx, pair_info in enumerate(candidate_pairs):
        pair_rank = int(pair_info.get("pair_rank", idx))
        evaluated = bool(pair_info.get("pair_evaluated", False))
        passed = bool(pair_info.get("passed_overlap_filter", False))
        color = _candidate_pair_color(pair_rank, total_pairs, passed, evaluated=evaluated)
        z_lift = 0.0015 * float(idx)
        radius = 0.0085 if evaluated and passed else 0.0075
        pair_group = f"{prim_path}/Pair_{idx:03d}"
        UsdGeom.Xform.Define(stage, pair_group)

        for label, point_key in (("Primary", "primary_point"), ("Opposite", "opposite_point")):
            point = np.asarray(pair_info[point_key], dtype=np.float64).copy()
            point[2] += z_lift
            sphere = UsdGeom.Sphere.Define(stage, f"{pair_group}/{label}")
            sphere.CreateRadiusAttr(radius)
            _set_translate_orient_ops(
                UsdGeom.Xformable(sphere.GetPrim()),
                point,
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            )
            sphere.CreateDisplayColorAttr().Set(
                [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
            )


def _pair_across_difference(pair_info) -> float:
    return float(pair_info.get("pair_metadata", {}).get("across_difference", float("-inf")))


def _bundle_across_difference(bundle) -> float:
    return float(bundle.anchor_pair.metadata.get("across_difference", float("-inf")))


def _print_pair_count_summary(summary: dict) -> None:
    metadata = summary.get("metadata", {})
    pair_visualization = summary.get("pair_visualization", {})
    cache_info = metadata.get("side_bundle_cache", {})
    persistent_cache_info = metadata.get("persistent_side_bundle_cache", {})
    parallel_info = metadata.get("parallelism", {})

    rows = [
        ("Candidate Pairs", metadata.get("num_candidate_pairs_total", 0)),
        ("Selected For Eval", metadata.get("num_pairs_selected_for_optimization", 0)),
        ("Evaluated Pairs", metadata.get("num_pairs_before_overlap_filter", 0)),
        ("Surviving Pairs", metadata.get("num_pairs_after_overlap_filter", 0)),
        ("Rejected By Overlap", metadata.get("num_pairs_rejected_by_overlap", 0)),
        ("Visualized Pairs", pair_visualization.get("num_visualized_pairs", 0)),
    ]
    if cache_info:
        rows.extend(
            [
                ("Cache Entries", cache_info.get("num_entries", 0)),
                ("Cache Hits", cache_info.get("num_hits", 0)),
                ("Cache Misses", cache_info.get("num_misses", 0)),
            ]
        )
    if persistent_cache_info:
        rows.extend(
            [
                ("Disk Cache Hits", persistent_cache_info.get("num_hits", 0)),
                ("Disk Cache Misses", persistent_cache_info.get("num_misses", 0)),
            ]
        )
    if parallel_info:
        rows.extend(
            [
                ("Unique Side Bundles", parallel_info.get("num_unique_side_bundles", 0)),
                ("Total Side Requests", parallel_info.get("num_total_side_bundle_requests", 0)),
                ("Parallel Backend", parallel_info.get("effective_parallel_backend", "unknown")),
                ("Parallel Workers", parallel_info.get("effective_max_workers", 0)),
            ]
        )

    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"

    print("\nPair Count Summary")
    print(border)
    print(f"| {'Metric'.ljust(label_width)} | {'Count'.rjust(value_width)} |")
    print(border)
    for label, value in rows:
        print(f"| {label.ljust(label_width)} | {str(value).rjust(value_width)} |")
    print(border)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bimanual staged grasp generation by pairing a primary anchor with an opposite-side anchor."
    )
    parser.add_argument("--hand_dir", type=Path, required=True)
    parser.add_argument("--config_dir", type=Path, required=True)
    parser.add_argument("--object_usd", type=Path, required=True)
    parser.add_argument("--primary_side", choices=["left", "right"], default="right")
    parser.add_argument("--category", choices=["cat1", "cat2", "cat3", "cat4"], default="cat1")

    parser.add_argument("--root_prim_path", type=str, default=None)
    parser.add_argument("--exclude_prim_paths", type=str, nargs="*", default=None)

    parser.add_argument("--num_surface_points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k_anchors", type=int, default=5)
    parser.add_argument("--num_opposite_candidates", type=int, default=5)
    parser.add_argument("--top_k_anchor_pairs", type=int, default=None)
    parser.add_argument("--num_seeds_per_contact", type=int, default=None)
    parser.add_argument("--top_k_seeds_per_contact", type=int, default=3)
    parser.add_argument(
        "--top_k_optimized_per_side",
        type=int,
        default=None,
        help="How many ranked seeds per side bundle to run through staged optimization. "
             "Defaults to min(top_k_seeds_per_contact, 4).",
    )
    parser.add_argument(
        "--max_result_pair_checks",
        type=int,
        default=None,
        help="Maximum left/right optimized-result combinations to overlap-check per ordered pair. "
             "Defaults to min(top_k_optimized_per_side^2, 24).",
    )
    parser.add_argument("--max_workers", type=int, default=0)
    parser.add_argument(
        "--parallel_backend",
        choices=["auto", "thread", "process"],
        default="auto",
        help="Backend for side-bundle parallelism. 'auto' prefers processes for headless validation exports.",
    )
    parser.add_argument("--pair_rank", type=int, default=0)
    parser.add_argument("--candidate_pair_rank", type=int, default=None)
    parser.add_argument("--pair_visualization", choices=["filtered", "all"], default="filtered")
    parser.add_argument(
        "--pair_selector",
        choices=["ranked", "farthest", "farthest_filtered"],
        default="farthest",
    )
    parser.add_argument("--result_rank", type=int, default=0)
    parser.add_argument("--seed_rank", type=int, default=0)
    parser.add_argument("--pose_source", choices=["staged", "seed"], default="staged")
    parser.add_argument("--stage", choices=["pregrasp", "grasp", "squeeze"], default="squeeze")

    parser.add_argument("--sdf_cache_path", type=Path, default=None)
    parser.add_argument("--force_rebuild_sdf", action="store_true")
    parser.add_argument("--object_prim_path", type=str, default="/World/Object")
    parser.add_argument("--left_hand_prim_path", type=str, default="/World/LeftHandStagePose")
    parser.add_argument("--right_hand_prim_path", type=str, default="/World/RightHandStagePose")
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show_debug_overlays", action="store_true")
    parser.add_argument("--save_json", type=Path, default=None)
    parser.add_argument(
        "--save_format",
        choices=["summary", "validation", "both"],
        default="summary",
        help="JSON format to save: preview summary, physics_validation schema, or both.",
    )
    parser.add_argument(
        "--export_speed_profile",
        choices=["standard", "safe"],
        default=None,
        help="Optimization profile for export runs. Defaults to 'safe' for validation exports and "
             "'standard' otherwise.",
    )

    args = parser.parse_args()

    simulation_app = _create_simulation_app(headless=args.headless)

    try:
        _import_runtime_helpers()

        import omni.kit.app
        import omni.usd
        from pxr import UsdGeom

        from src.pipeline_bimanual import run_bimanual_staged, summarize_bimanual_staged_pipeline
        from src.optimizer_core import summarize_staged_grasp_result

        num_surface_points = _effective_num_surface_points(args.category, args.num_surface_points)
        top_k_optimized_per_side = _effective_top_k_optimized_per_side(
            args.top_k_optimized_per_side,
            args.top_k_seeds_per_contact,
        )
        max_result_pair_checks = _effective_max_result_pair_checks(
            args.max_result_pair_checks,
            top_k_optimized_per_side,
        )
        export_speed_profile = _effective_export_speed_profile(
            args.export_speed_profile,
            args.save_format,
        )
        if args.export_speed_profile is None and str(args.category).lower() == "cat3":
            export_speed_profile = "safe"
        optimizer_cfg_overrides = _optimizer_overrides_for_export_speed_profile(
            export_speed_profile,
            args.category,
        )
        parallel_backend = _effective_parallel_backend(
            args.parallel_backend,
            headless=bool(args.headless),
            save_format=args.save_format,
        )

        result = run_bimanual_staged(
            hand_dir=args.hand_dir,
            global_config_dir=args.config_dir,
            primary_side=args.primary_side,
            category=args.category,
            object_usd=args.object_usd,
            root_prim_path=args.root_prim_path,
            exclude_prim_paths=args.exclude_prim_paths,
            num_surface_points=num_surface_points,
            seed=args.seed,
            top_k_anchors=args.top_k_anchors,
            num_opposite_candidates=args.num_opposite_candidates,
            top_k_anchor_pairs=args.top_k_anchor_pairs,
            num_seeds_per_contact=args.num_seeds_per_contact,
            top_k_seeds_per_contact=args.top_k_seeds_per_contact,
            top_k_optimized_per_side=top_k_optimized_per_side,
            max_workers=args.max_workers,
            sdf_cache_path=args.sdf_cache_path,
            force_rebuild_sdf=args.force_rebuild_sdf,
            optimizer_cfg_overrides=optimizer_cfg_overrides,
            pair_selection_mode="farthest_raw" if args.pair_selector == "farthest" else "ranked",
            max_result_pair_checks=max_result_pair_checks,
            parallel_backend=parallel_backend,
        )
        summary = summarize_bimanual_staged_pipeline(result)
        result.metadata["export_speed_profile"] = export_speed_profile
        result.metadata["parallel_backend"] = parallel_backend
        primary_side = result.metadata["primary_side"]
        opposite_side = result.metadata["opposite_side"]
        all_candidate_pairs = list(result.metadata.get("candidate_pairs_debug", []))
        if args.pair_visualization == "all":
            candidate_pairs = all_candidate_pairs
        else:
            candidate_pairs = [
                pair_info for pair_info in all_candidate_pairs
                if bool(pair_info.get("passed_overlap_filter", False))
            ]
        summary["pair_visualization"] = {
            "mode": args.pair_visualization,
            "num_visualized_pairs": int(len(candidate_pairs)),
            "num_all_candidate_pairs": int(len(all_candidate_pairs)),
        }
        summary["pair_selector"] = args.pair_selector
        summary["show_debug_overlays"] = bool(args.show_debug_overlays)
        summary["pose_source"] = args.pose_source
        summary["export_speed_profile"] = export_speed_profile
        summary["parallel_backend"] = parallel_backend

        stage = omni.usd.get_context().get_stage()
        _ensure_world(stage)
        _clear_prim_if_exists(stage, args.object_prim_path)
        _clear_prim_if_exists(stage, args.left_hand_prim_path)
        _clear_prim_if_exists(stage, args.right_hand_prim_path)
        _clear_prim_if_exists(stage, "/World/DebugBimanualPreview")
        _clear_prim_if_exists(stage, "/World/SelectedBimanualAnchors")

        _add_reference(stage, args.object_usd, args.object_prim_path)
        if args.show_debug_overlays:
            UsdGeom.Xform.Define(stage, "/World/DebugBimanualPreview")
            _visualize_surface_subsample(stage, result.proposal_result, "/World/DebugBimanualPreview")
            _visualize_all_anchors(stage, result.proposal_result.anchors, "/World/DebugBimanualPreview/AllAnchors")
            _visualize_paired_anchor_overlays(
                stage,
                candidate_pairs,
                "/World/DebugBimanualPreview/PairedAnchorOverlays",
            )
            _visualize_candidate_pairs(stage, candidate_pairs, "/World/DebugBimanualPreview/CandidatePairs")

        selected_candidate_pair = None
        pair_bundle = None
        if args.pair_selector == "farthest" and result.evaluated_pair_bundles:
            pair_bundle = max(result.evaluated_pair_bundles, key=_bundle_across_difference)
        elif result.pair_bundles:
            if args.pair_selector == "farthest_filtered":
                pair_bundle = max(result.pair_bundles, key=_bundle_across_difference)
            else:
                pair_rank = int(max(0, min(args.pair_rank, len(result.pair_bundles) - 1)))
                pair_bundle = result.pair_bundles[pair_rank]

        candidate_pair_source = all_candidate_pairs if args.pair_selector == "farthest" else candidate_pairs
        if candidate_pair_source:
            if pair_bundle is not None:
                target_pair_rank = int(pair_bundle.metadata.get("pair_rank", -1))
                selected_candidate_pair = next(
                    (
                        pair_info
                        for pair_info in candidate_pair_source
                        if int(pair_info.get("pair_rank", -1)) == target_pair_rank
                    ),
                    None,
                )
            if selected_candidate_pair is None:
                if args.pair_selector == "farthest":
                    selected_candidate_pair = max(candidate_pair_source, key=_pair_across_difference)
                elif args.pair_selector == "farthest_filtered":
                    selected_candidate_pair = max(candidate_pair_source, key=_pair_across_difference)
                else:
                    candidate_pair_rank = args.candidate_pair_rank
                    if candidate_pair_rank is None:
                        candidate_pair_rank = args.pair_rank
                    candidate_pair_rank = int(max(0, min(candidate_pair_rank, len(candidate_pair_source) - 1)))
                    selected_candidate_pair = candidate_pair_source[candidate_pair_rank]
            if selected_candidate_pair is not None:
                if args.show_debug_overlays:
                    _visualize_candidate_pair_focus(
                        stage,
                        selected_candidate_pair,
                        "/World/DebugBimanualPreview/SelectedCandidatePair",
                    )
                summary["selected_candidate_pair"] = selected_candidate_pair

        if pair_bundle is not None:
            pair_rank = int(pair_bundle.metadata.get("pair_rank", 0))
            valid_result_pairs = list(pair_bundle.metadata.get("valid_result_pairs", []))
            if valid_result_pairs:
                selected_result_pair_rank = int(max(0, min(args.result_rank, max(0, len(valid_result_pairs) - 1))))
                selected_result_pair = valid_result_pairs[selected_result_pair_rank]
            else:
                selected_result_pair_rank = 0
                best_result_pair = pair_bundle.metadata.get("best_result_pair")
                selected_result_pair = (
                    best_result_pair
                    if best_result_pair is not None
                    else {
                        "result_rank_by_side": {
                            primary_side: int(args.result_rank),
                            opposite_side: int(args.result_rank),
                        }
                    }
                )

            selected_pair_summary = {
                "pair_rank": pair_rank,
                "result_pair_rank": selected_result_pair_rank,
                "selected_stage": (args.stage if args.pose_source == "staged" else None),
                "selected_pose_source": args.pose_source,
                "pair_score": float(pair_bundle.anchor_pair.pair_score),
                "primary_anchor_rank": int(pair_bundle.anchor_pair.primary_anchor_rank),
                "opposite_anchor_rank": int(pair_bundle.anchor_pair.opposite_anchor_rank),
                "pair_metadata": pair_bundle.anchor_pair.metadata,
                "bundle_metadata": pair_bundle.metadata,
                "selected_result_pair": selected_result_pair,
                "passes_overlap_filter": bool(pair_bundle.metadata.get("passes_overlap_filter", bool(valid_result_pairs))),
                "hands": {},
            }

            for side in (primary_side, opposite_side):
                hand_bundle = pair_bundle.bundles_by_side[side]
                side_summary = {
                    "contact_anchor_score": float(hand_bundle.contact_result.metadata.get("anchor_score", 0.0)),
                    "contact_category": hand_bundle.contact_result.category,
                    "contact_template": hand_bundle.contact_result.contact_template,
                    "contact_mode": hand_bundle.contact_result.metadata.get("cat1_contact_mode"),
                    "contact_mode_source": hand_bundle.contact_result.metadata.get("cat1_contact_mode_source"),
                    "contact_local_width": hand_bundle.contact_result.metadata.get("cat1_local_width"),
                }

                if args.pose_source == "seed":
                    if not hand_bundle.seed_result.seeds:
                        continue
                    seed_rank = int(max(0, min(args.seed_rank, len(hand_bundle.seed_result.seeds) - 1)))
                    selected_seed = hand_bundle.seed_result.seeds[seed_rank]
                    side_summary["selected_seed"] = {
                        "seed_rank": seed_rank,
                        "posture_name": selected_seed.posture_name,
                        "approach_name": selected_seed.approach_name,
                        "wrist_position": [
                            float(x) for x in np.asarray(selected_seed.wrist_position, dtype=np.float64).tolist()
                        ],
                        "wrist_quaternion_wxyz": [
                            float(x)
                            for x in np.asarray(
                                [
                                    selected_seed.wrist_quaternion_xyzw[3],
                                    selected_seed.wrist_quaternion_xyzw[0],
                                    selected_seed.wrist_quaternion_xyzw[1],
                                    selected_seed.wrist_quaternion_xyzw[2],
                                ],
                                dtype=np.float64,
                            ).tolist()
                        ],
                        "wrist_quaternion_xyzw": [
                            float(x) for x in np.asarray(selected_seed.wrist_quaternion_xyzw, dtype=np.float64).tolist()
                        ],
                        "joint_positions_rad": {
                            joint_name: float(value) for joint_name, value in selected_seed.joint_positions.items()
                        },
                        "joint_positions_deg": {
                            joint_name: float(np.rad2deg(value)) for joint_name, value in selected_seed.joint_positions.items()
                        },
                        "metadata": selected_seed.metadata,
                    }
                else:
                    if not hand_bundle.optimized_results:
                        continue
                    preferred_rank = int(selected_result_pair["result_rank_by_side"][side])
                    result_rank = int(max(0, min(preferred_rank, len(hand_bundle.optimized_results) - 1)))
                    staged_result = hand_bundle.optimized_results[result_rank]
                    stage_result = staged_result.stage_results[args.stage]
                    hand_pose = stage_result.hand_pose
                    side_summary["selected_result"] = summarize_staged_grasp_result(staged_result)
                    side_summary["selected_stage_pose"] = {
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
                        "joint_positions_deg": {
                            joint_name: float(np.rad2deg(value)) for joint_name, value in hand_pose.joint_positions.items()
                        },
                    }

                selected_pair_summary["hands"][side] = side_summary

            summary["selected_pair"] = selected_pair_summary

            anchor_colors = {
                primary_side: (1.0, 0.2, 0.2),
                opposite_side: (0.2, 1.0, 0.2),
            }
            pair_anchors = {
                primary_side: pair_bundle.anchor_pair.primary_anchor,
                opposite_side: pair_bundle.anchor_pair.opposite_anchor,
            }
            hand_prim_paths = {
                "left": args.left_hand_prim_path,
                "right": args.right_hand_prim_path,
            }
            hand_alignment_info = {}
            joint_update_info = {}
            stage_alignment_metrics = {}

            for side in (primary_side, opposite_side):
                hand_bundle = pair_bundle.bundles_by_side[side]
                runtime_cfg = result.runtime_cfg_by_side[side]
                hand_prim_path = hand_prim_paths[side]

                if args.pose_source == "seed":
                    if not hand_bundle.seed_result.seeds:
                        continue
                    seed_rank = int(max(0, min(args.seed_rank, len(hand_bundle.seed_result.seeds) - 1)))
                    selected_seed = hand_bundle.seed_result.seeds[seed_rank]
                    wrist_position = np.asarray(selected_seed.wrist_position, dtype=np.float64)
                    wrist_quaternion_xyzw = np.asarray(selected_seed.wrist_quaternion_xyzw, dtype=np.float64)
                    joint_positions = selected_seed.joint_positions
                else:
                    if not hand_bundle.optimized_results:
                        continue
                    preferred_rank = int(selected_result_pair["result_rank_by_side"][side])
                    result_rank = int(max(0, min(preferred_rank, len(hand_bundle.optimized_results) - 1)))
                    staged_result = hand_bundle.optimized_results[result_rank]
                    stage_result = staged_result.stage_results[args.stage]
                    hand_pose = stage_result.hand_pose
                    wrist_position = hand_pose.wrist_position
                    wrist_quaternion_xyzw = _quat_xyzw_from_rotation_matrix(hand_pose.rotation_matrix())
                    joint_positions = hand_pose.joint_positions

                _add_reference(stage, runtime_cfg.hand.asset.usd_path, hand_prim_path)
                hand_alignment_info[side] = _set_hand_container_from_wrist_pose(
                    stage,
                    hand_prim_path=hand_prim_path,
                    wrist_link_name=runtime_cfg.hand.root.wrist_link,
                    wrist_position=wrist_position,
                    wrist_quaternion_xyzw=wrist_quaternion_xyzw,
                )
                joint_update_info[side] = _set_hand_joint_targets(
                    stage,
                    hand_prim_path,
                    joint_positions,
                )
                _visualize_selected_anchor(
                    stage,
                    pair_anchors[side],
                    f"/World/SelectedBimanualAnchors/{side.capitalize()}Anchor",
                    anchor_colors[side],
                )
                if args.show_debug_overlays and args.pose_source == "staged":
                    stage_alignment_metrics[side] = _visualize_stage_targets(
                        stage,
                        f"/World/DebugBimanualPreview/{side.capitalize()}Targets",
                        runtime_cfg,
                        hand_bundle.contact_result,
                        hand_pose,
                        args.stage,
                    )

            summary["hand_alignment_info"] = hand_alignment_info
            summary["joint_update_info"] = joint_update_info
            if args.show_debug_overlays:
                summary["stage_alignment_metrics"] = stage_alignment_metrics

        save_format = str(args.save_format).lower()
        quiet_validation_export = (
            bool(args.headless)
            and args.save_json is not None
            and save_format == "validation"
        )
        if not quiet_validation_export:
            print(json.dumps(summary, indent=2), flush=True)
            _print_pair_count_summary(summary)

        if args.save_json is not None:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            if save_format in {"summary", "both"}:
                with args.save_json.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            if save_format in {"validation", "both"}:
                validation_payload = _bimanual_result_to_validation_payload(result)
                validation_path = (
                    args.save_json
                    if save_format == "validation"
                    else args.save_json.with_name(f"{args.save_json.stem}_validation{args.save_json.suffix}")
                )
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                with validation_path.open("w", encoding="utf-8") as f:
                    json.dump(validation_payload, f, indent=2)
                if save_format == "validation":
                    num_grasps = len(validation_payload.get("functional_grasp", {}).get("body", []))
                    print(f"Saved {num_grasps} physics-validation grasps to: {validation_path}")
                else:
                    print(f"Saved summary JSON to: {args.save_json}")
                    print(f"Saved physics-validation JSON to: {validation_path}")

        app = omni.kit.app.get_app()
        for _ in range(max(0, int(args.settle_steps))):
            app.update()

        if not args.headless:
            print("Bimanual preview ready in Isaac Sim GUI. Close the app window to exit.")
            while simulation_app.is_running():
                app.update()
    except BaseException as exc:
        print(
            f"[run_bimanual] fatal path reached: {type(exc).__name__}: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        if isinstance(exc, SystemExit):
            raise
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
        _clear_project_cache()
        cleanup_simulation_runtime_cache()


if __name__ == "__main__":
    main()
