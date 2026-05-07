from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from simulation_app_runtime import create_simulation_app, cleanup_simulation_runtime_cache


def _effective_num_surface_points(category: str, requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    if str(category).lower() == "cat4":
        return 20000
    return 100000


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


def _optimizer_overrides_for_export_speed_profile(profile: str) -> dict | None:
    resolved = str(profile).lower()
    if resolved == "standard":
        return None
    if resolved != "safe":
        raise ValueError(f"Unsupported export speed profile: {profile}")
    return {
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


def _object_label(object_usd: Path) -> str:
    object_usd = object_usd.resolve()
    if object_usd.stem.lower() == "object" and object_usd.parent.name:
        return object_usd.parent.name
    return object_usd.stem or "object"


def _default_output_json_path(
    *,
    category: str,
    side: str,
    object_usd: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "outputs" / "single_hand_staged" / str(category)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}_{side}_{_object_label(object_usd)}_{timestamp}.json"
    return out_dir / filename


def _default_validation_json_path(
    *,
    category: str,
    side: str,
    object_usd: Path,
) -> Path:
    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / "outputs" / "physics_validation_inputs" / str(category)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}_{side}_{_object_label(object_usd)}_{timestamp}_grasps.json"
    return out_dir / filename


def _object_type_label(object_usd: Path) -> str:
    object_usd = object_usd.resolve()
    if object_usd.stem.lower() == "object":
        if object_usd.parent.parent.name:
            return object_usd.parent.parent.name
        if object_usd.parent.name:
            return object_usd.parent.name
    return object_usd.stem or "object"


def _validation_object_type_label(category: str, object_usd: Path) -> str:
    if str(category).lower() == "cat4":
        return "small_obj"
    return _object_type_label(object_usd)


def _bottom_center_from_result(result) -> list[float]:
    bbox_min = result.proposal_result.samples.bbox_min
    bbox_max = result.proposal_result.samples.bbox_max
    return [
        float(0.5 * (float(bbox_min[0]) + float(bbox_max[0]))),
        float(0.5 * (float(bbox_min[1]) + float(bbox_max[1]))),
        float(bbox_min[2]),
    ]


def _pack_legacy_stage(stage_summary: dict, joint_order: list[str]) -> dict:
    return {
        "position": [float(x) for x in stage_summary["wrist_position"]],
        "orientation": [float(x) for x in stage_summary["wrist_quaternion_wxyz"]],
        "joints": [
            float(stage_summary.get("joint_positions", {}).get(joint_name, 0.0))
            for joint_name in joint_order
        ],
    }


def _summary_to_physics_validation_input(
    summary: dict,
    *,
    object_type: str,
    bottom_center: list[float],
    joint_order: list[str],
) -> dict:
    body = []
    for bundle in summary.get("bundles", []):
        for optimized in bundle.get("optimized", []):
            stages = optimized.get("stages", {})
            if not all(stage_name in stages for stage_name in ("pregrasp", "grasp", "squeeze")):
                continue
            body.append(
                {
                    "coarse_grasp": _pack_legacy_stage(stages["pregrasp"], joint_order),
                    "fine_grasp": _pack_legacy_stage(stages["grasp"], joint_order),
                    "final_grasp": _pack_legacy_stage(stages["squeeze"], joint_order),
                }
            )

    return {
        "type": str(object_type),
        "bottom_center": [float(x) for x in bottom_center],
        "functional_grasp": {
            "body": body,
        },
        "grasp": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run headless single-hand staged grasp generation and save JSON summaries."
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
    parser.add_argument("--top_k_optimized_per_contact", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=0)
    parser.add_argument(
        "--parallel_backend",
        choices=["auto", "thread", "process"],
        default="auto",
        help="Backend for candidate optimization parallelism. 'auto' prefers processes for headless validation exports.",
    )

    parser.add_argument("--sdf_cache_path", type=Path, default=None)
    parser.add_argument("--force_rebuild_sdf", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save_json", type=Path, default=None)
    parser.add_argument(
        "--save_format",
        choices=["summary", "validation", "both"],
        default="summary",
        help="JSON format to save: staged summary, physics_validation input, or both.",
    )
    parser.add_argument(
        "--export_speed_profile",
        choices=["standard", "safe"],
        default=None,
        help="Optimization profile for export runs. Defaults to 'safe' for validation exports and "
             "'standard' otherwise.",
    )

    args = parser.parse_args()

    simulation_app = create_simulation_app(headless=args.headless, script_name="run_single_hand_staged")

    try:
        from src.staged_grasp_pipeline import (
            run_single_hand_staged,
            summarize_single_hand_staged_pipeline,
        )

        num_surface_points = _effective_num_surface_points(args.category, args.num_surface_points)
        export_speed_profile = _effective_export_speed_profile(
            args.export_speed_profile,
            args.save_format,
        )
        optimizer_cfg_overrides = _optimizer_overrides_for_export_speed_profile(
            export_speed_profile,
        )
        parallel_backend = _effective_parallel_backend(
            args.parallel_backend,
            headless=bool(args.headless),
            save_format=args.save_format,
        )
        top_k_optimized_per_contact = (
            None if int(args.top_k_optimized_per_contact) <= 0 else int(args.top_k_optimized_per_contact)
        )

        result = run_single_hand_staged(
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
            num_seeds_per_contact=args.num_seeds_per_contact,
            top_k_seeds_per_contact=args.top_k_seeds_per_contact,
            top_k_optimized_per_contact=top_k_optimized_per_contact,
            max_workers=(None if int(args.max_workers) <= 0 else int(args.max_workers)),
            sdf_cache_path=args.sdf_cache_path,
            force_rebuild_sdf=args.force_rebuild_sdf,
            optimizer_cfg_overrides=optimizer_cfg_overrides,
            parallel_backend=parallel_backend,
        )
        summary = summarize_single_hand_staged_pipeline(
            result,
            top_k_optimized_per_contact=top_k_optimized_per_contact,
        )
        summary["export_speed_profile"] = export_speed_profile
        summary["parallel_backend"] = parallel_backend

        save_format = str(args.save_format).lower()
        save_summary = save_format in {"summary", "both"}
        save_validation = save_format in {"validation", "both"}

        save_json_path = args.save_json
        validation_json_path = None
        if save_summary:
            if save_json_path is None and args.headless:
                save_json_path = _default_output_json_path(
                    category=args.category,
                    side=args.side,
                    object_usd=args.object_usd,
                )
        elif save_validation:
            validation_json_path = args.save_json

        if save_validation and validation_json_path is None and args.headless:
            validation_json_path = _default_validation_json_path(
                category=args.category,
                side=args.side,
                object_usd=args.object_usd,
            )

        compact_summary = {
            "side": summary["side"],
            "category": summary["category"],
            "object_usd": summary["object_usd"],
            "num_anchors_total": summary["num_anchors_total"],
            "num_contact_bundles": summary["num_contact_bundles"],
            "metadata": summary["metadata"],
            "export_speed_profile": export_speed_profile,
            "parallel_backend": parallel_backend,
        }

        if save_summary and save_json_path is not None:
            save_json_path.parent.mkdir(parents=True, exist_ok=True)
            with save_json_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            compact_summary["saved_summary_json"] = str(save_json_path)

        if save_validation and validation_json_path is not None:
            validation_payload = _summary_to_physics_validation_input(
                summary,
                object_type=_validation_object_type_label(args.category, args.object_usd),
                bottom_center=_bottom_center_from_result(result),
                joint_order=list(result.proposal_result.runtime_cfg.hand.joints.controllable),
            )
            validation_json_path.parent.mkdir(parents=True, exist_ok=True)
            with validation_json_path.open("w", encoding="utf-8") as f:
                json.dump(validation_payload, f, indent=2)
            compact_summary["saved_validation_json"] = str(validation_json_path)

        if save_summary or save_validation:
            print(json.dumps(compact_summary, indent=2))
        else:
            print(json.dumps(summary, indent=2))
    finally:
        simulation_app.close()
        cleanup_simulation_runtime_cache()


if __name__ == "__main__":
    main()
