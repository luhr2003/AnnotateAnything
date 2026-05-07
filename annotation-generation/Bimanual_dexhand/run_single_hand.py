from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulation_app_runtime import create_simulation_app, cleanup_simulation_runtime_cache

np = None


def _import_runtime_numpy() -> None:
    global np
    import numpy as _np

    np = _np


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


def _visualize_anchors_in_stage(result, object_usd: Path, root_object_prim: str = "/World/Object") -> None:
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Vt, UsdGeom

    stage = omni.usd.get_context().get_stage()

    # Ensure /World exists
    world = stage.GetPrimAtPath("/World")
    if not world or not world.IsValid():
        UsdGeom.Xform.Define(stage, "/World")

    # Clear previous prims
    _clear_prim_if_exists(stage, root_object_prim)
    _clear_prim_if_exists(stage, "/World/DebugSurface")
    _clear_prim_if_exists(stage, "/World/DebugAnchors")

    # Add object reference
    add_reference_to_stage(usd_path=str(object_usd.resolve()), prim_path=root_object_prim)

    # --- Surface point cloud (subsampled for display) ---
    pts = result.samples.points  # (N, 3)
    max_display = 2000
    if len(pts) > max_display:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), size=max_display, replace=False)
        pts_display = pts[idx]
    else:
        pts_display = pts
    surface_points = UsdGeom.Points.Define(stage, "/World/DebugSurface")
    surface_points.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts_display]))
    surface_points.GetWidthsAttr().Set(Vt.FloatArray([0.004] * len(pts_display)))
    surface_points.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.4, 0.8, 1.0)]))  # light blue
    print(f"Visualized {len(pts_display)}/{len(pts)} surface points under /World/DebugSurface")

    # --- Anchor spheres ---
    UsdGeom.Xform.Define(stage, "/World/DebugAnchors")

    anchors_sorted = sorted(result.anchors, key=lambda a: a.score, reverse=True)
    if not anchors_sorted:
        print("No anchors to visualize.")
        print(f"Object loaded at {root_object_prim}")
        return

    max_score = max(a.score for a in anchors_sorted)
    min_score = min(a.score for a in anchors_sorted)
    score_span = max(max_score - min_score, 1e-8)

    for i, anchor in enumerate(anchors_sorted):
        prim_path = f"/World/DebugAnchors/anchor_{i:03d}"
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.CreateRadiusAttr(0.008)

        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(anchor.point[0]), float(anchor.point[1]), float(anchor.point[2]))
        )

        t = (anchor.score - min_score) / score_span
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(float(t), 0.2, float(1.0 - t))])

    print(f"Visualized {len(anchors_sorted)} anchors under /World/DebugAnchors")
    print(f"Object loaded at {root_object_prim}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-hand region proposal for grasp generation.")
    parser.add_argument("--hand_dir", type=Path, required=True)
    parser.add_argument("--config_dir", type=Path, required=True)
    parser.add_argument("--object_usd", type=Path, required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--category", choices=["cat1", "cat2", "cat3", "cat4"], required=True)

    parser.add_argument("--root_prim_path", type=str, default=None)
    parser.add_argument("--exclude_prim_paths", type=str, nargs="*", default=None)

    parser.add_argument("--num_surface_points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save_json", type=Path, default=None)

    args = parser.parse_args()

    simulation_app = create_simulation_app(headless=args.headless, script_name="run_single_hand")

    try:
        _import_runtime_numpy()
        import omni.kit.app

        from src.pipeline_single_hand import (
            run_single_hand_region_proposal,
            summarize_single_hand_result,
        )
        num_surface_points = _effective_num_surface_points(args.category, args.num_surface_points)

        result = run_single_hand_region_proposal(
            hand_dir=args.hand_dir,
            global_config_dir=args.config_dir,
            side=args.side,
            category=args.category,
            object_usd=args.object_usd,
            root_prim_path=args.root_prim_path,
            exclude_prim_paths=args.exclude_prim_paths,
            num_surface_points=num_surface_points,
            seed=args.seed,
        )

        summary = summarize_single_hand_result(result, top_k=args.top_k)
        print(json.dumps(summary, indent=2))

        if args.save_json is not None:
            args.save_json.parent.mkdir(parents=True, exist_ok=True)
            with args.save_json.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved summary to: {args.save_json}")

        if args.visualize:
            _visualize_anchors_in_stage(result, args.object_usd)

            # Keep GUI alive
            app = omni.kit.app.get_app()
            print("Visualization ready in Isaac Sim GUI. Close the app window to exit.")
            while simulation_app.is_running():
                app.update()

    finally:
        simulation_app.close()
        cleanup_simulation_runtime_cache()


if __name__ == "__main__":
    main()
