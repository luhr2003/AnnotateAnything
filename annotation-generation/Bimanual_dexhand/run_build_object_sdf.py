from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulation_app_runtime import create_simulation_app, cleanup_simulation_runtime_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or load a cached object SDF from a USD file or a live Isaac Sim stage prim."
    )
    parser.add_argument("--object_usd", type=Path, default=None)
    parser.add_argument("--cache_path", type=Path, default=None)
    parser.add_argument("--root_prim_path", type=str, default="/World/Object")
    parser.add_argument("--exclude_prim_paths", type=str, nargs="*", default=None)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--padding_voxels", type=int, default=8)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--use_stage", action="store_true")
    parser.add_argument("--load_into_stage", action="store_true")
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    simulation_app = create_simulation_app(headless=args.headless, script_name="run_build_object_sdf")
    try:
        from src.object_query import (
            default_sdf_cache_path,
            ensure_object_reference_in_stage,
            load_or_build_sdf_from_stage,
            load_or_build_sdf_from_usd,
        )

        if args.use_stage:
            if args.load_into_stage:
                if args.object_usd is None:
                    raise ValueError("--object_usd is required when --load_into_stage is set.")
                ensure_object_reference_in_stage(
                    args.object_usd,
                    prim_path=args.root_prim_path,
                )
            cache_path = args.cache_path
            if cache_path is None:
                if args.object_usd is None:
                    raise ValueError("--cache_path is required for stage-only SDF generation.")
                cache_path = default_sdf_cache_path(args.object_usd)
            sdf = load_or_build_sdf_from_stage(
                root_prim_path=args.root_prim_path,
                cache_path=cache_path,
                voxel_size=args.voxel_size,
                padding_voxels=args.padding_voxels,
                exclude_prim_paths=args.exclude_prim_paths,
                force_rebuild=args.force_rebuild,
            )
        else:
            if args.object_usd is None:
                raise ValueError("--object_usd is required unless --use_stage is set.")
            sdf = load_or_build_sdf_from_usd(
                usd_path=args.object_usd,
                cache_path=args.cache_path,
                voxel_size=args.voxel_size,
                padding_voxels=args.padding_voxels,
                root_prim_path=args.root_prim_path,
                exclude_prim_paths=args.exclude_prim_paths,
                force_rebuild=args.force_rebuild,
            )

        summary = {
            "cache_path": str(sdf.metadata.get("cache_path", "")),
            "voxel_size": float(sdf.voxel_size),
            "shape": [int(x) for x in sdf.shape],
            "metadata": sdf.metadata,
        }
        print(json.dumps(summary, indent=2))
    finally:
        simulation_app.close()
        cleanup_simulation_runtime_cache()


if __name__ == "__main__":
    main()
