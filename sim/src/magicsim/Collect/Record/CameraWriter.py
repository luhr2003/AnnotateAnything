import os
from typing import Any, Dict

import numpy as np
import magicsim.Collect.Record.io_functions as F
from omni.replicator.core.scripts.writers_default.tools import (
    colorize_distance,
    colorize_normals,
)


def write_annotator_step(
    backend,
    annotator_name: str,
    payload: Dict[str, Any],
    annotator_dir_rel: str,
    step_idx: int,
):
    """Write a single annotator output for one camera and one step.

    This mirrors the behavior of `MyWriter.write`'s per-annotator write functions,
    but uses a simplified naming scheme:

        <annotator_dir_rel>/step_{step_idx:04d}.<ext>
    """
    # Ensure folder exists on disk (backend uses output_dir as root)
    # The caller is responsible for creating parent folders if needed.
    base_prefix = os.path.join(annotator_dir_rel, f"step_{step_idx:04d}")

    # Normalize payload format to match TiledCaptureManager
    if not isinstance(payload, dict):
        payload = {"data": payload}

    # ------------------------------------------------------------------
    # RGB
    # ------------------------------------------------------------------
    if annotator_name.startswith("rgb"):
        data = payload["data"]
        img_path = base_prefix + ".png"
        backend.schedule(F.write_image, data=data, path=img_path)
        return

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------
    if annotator_name.startswith("normals"):
        normals_data = payload["data"]
        file_path = base_prefix + ".png"
        colorized_normals_data = colorize_normals(normals_data)
        backend.schedule(F.write_image, data=colorized_normals_data, path=file_path)
        return

    # ------------------------------------------------------------------
    # Depth-like data (follow MyWriter._write_distance_to_camera / _write_distance_to_image_plane)
    # ------------------------------------------------------------------
    if annotator_name.startswith("distance_to_camera"):
        dist_to_cam_data = payload["data"]
        # Ensure data is numpy array with float type
        if not isinstance(dist_to_cam_data, np.ndarray):
            if hasattr(dist_to_cam_data, "cpu"):
                dist_to_cam_data = dist_to_cam_data.cpu().numpy()
            elif hasattr(dist_to_cam_data, "numpy"):
                dist_to_cam_data = dist_to_cam_data.numpy()
            else:
                dist_to_cam_data = np.array(dist_to_cam_data)
        # Convert to float if needed (np.isnan requires float types)
        if dist_to_cam_data.dtype not in [np.float32, np.float64]:
            dist_to_cam_data = dist_to_cam_data.astype(np.float32)
        # Save raw depth as .npy
        npy_path = base_prefix + ".npy"
        backend.schedule(F.write_np, data=dist_to_cam_data, path=npy_path)

        # Also save a colorized PNG (same logic as MyWriter)
        png_path = base_prefix + ".png"
        valid_data = dist_to_cam_data[
            (dist_to_cam_data != -np.inf)
            & (dist_to_cam_data != np.inf)
            & ~np.isnan(dist_to_cam_data)
        ]
        if len(valid_data) > 0:
            near = np.min(valid_data)
            far = np.max(valid_data)
            backend.schedule(
                F.write_image,
                data=colorize_distance(dist_to_cam_data, near=near, far=far),
                path=png_path,
            )
        else:
            blank_image = np.zeros((*dist_to_cam_data.shape[:2], 4), dtype=np.uint8)
            backend.schedule(F.write_image, data=blank_image, path=png_path)
        return

    if annotator_name.startswith("distance_to_image_plane"):
        dis_to_img_plane_data = payload["data"]
        # Ensure data is numpy array with float type
        if not isinstance(dis_to_img_plane_data, np.ndarray):
            if hasattr(dis_to_img_plane_data, "cpu"):
                dis_to_img_plane_data = dis_to_img_plane_data.cpu().numpy()
            elif hasattr(dis_to_img_plane_data, "numpy"):
                dis_to_img_plane_data = dis_to_img_plane_data.numpy()
            else:
                dis_to_img_plane_data = np.array(dis_to_img_plane_data)
        # Convert to float if needed (np.isnan requires float types)
        if dis_to_img_plane_data.dtype not in [np.float32, np.float64]:
            dis_to_img_plane_data = dis_to_img_plane_data.astype(np.float32)
        # Save raw depth as .npy
        npy_path = base_prefix + ".npy"
        backend.schedule(F.write_np, data=dis_to_img_plane_data, path=npy_path)

        # Also save a colorized PNG (same logic as MyWriter)
        png_path = base_prefix + ".png"
        valid_data = dis_to_img_plane_data[
            (dis_to_img_plane_data != -np.inf)
            & (dis_to_img_plane_data != np.inf)
            & ~np.isnan(dis_to_img_plane_data)
        ]
        if len(valid_data) > 0:
            near = np.min(valid_data)
            far = np.max(valid_data)
            backend.schedule(
                F.write_image,
                data=colorize_distance(dis_to_img_plane_data, near=near, far=far),
                path=png_path,
            )
        else:
            blank_image = np.zeros(
                (*dis_to_img_plane_data.shape[:2], 4), dtype=np.uint8
            )
            backend.schedule(F.write_image, data=blank_image, path=png_path)
        return

    # ------------------------------------------------------------------
    # Semantic segmentation
    # ------------------------------------------------------------------
    if annotator_name.startswith("semantic_segmentation"):
        semantic_seg_data = payload["data"]
        height, width = semantic_seg_data.shape[:2]

        png_path = base_prefix + ".png"
        # Follow MyWriter: handle both colorized (uint8) and raw (uint32) formats
        # If uint32, convert to uint8 RGBA by viewing as bytes; if uint8, use as-is
        if semantic_seg_data.dtype == np.uint32:
            semantic_seg_rgba = semantic_seg_data.view(np.uint8).reshape(
                height, width, -1
            )
        elif semantic_seg_data.dtype == np.uint8 and len(semantic_seg_data.shape) == 2:
            semantic_seg_rgba = semantic_seg_data.view(np.uint8).reshape(
                height, width, -1
            )
        else:
            semantic_seg_rgba = semantic_seg_data

        backend.schedule(F.write_image, data=semantic_seg_rgba, path=png_path)

        if "idToLabels" in payload:
            labels_path = base_prefix + "_labels.json"
            backend.schedule(
                F.write_json,
                data={str(k): v for k, v in payload["idToLabels"].items()},
                path=labels_path,
            )
        return

    # ------------------------------------------------------------------
    # Instance-id segmentation (support both fast and non-fast names)
    # ------------------------------------------------------------------
    if annotator_name.startswith("instance_id_segmentation"):
        instance_seg_data = payload["data"]
        height, width = instance_seg_data.shape[:2]

        png_path = base_prefix + ".png"
        # Follow MyWriter: handle both colorized (uint8) and raw (uint32) formats
        if instance_seg_data.dtype == np.uint32:
            instance_seg_rgba = instance_seg_data.view(np.uint8).reshape(
                height, width, -1
            )
        elif instance_seg_data.dtype == np.uint8 and len(instance_seg_data.shape) == 2:
            instance_seg_rgba = instance_seg_data.view(np.uint8).reshape(
                height, width, -1
            )
        else:
            instance_seg_rgba = instance_seg_data

        backend.schedule(F.write_image, data=instance_seg_rgba, path=png_path)

        if "idToLabels" in payload:
            labels_path = base_prefix + "_mapping.json"
            backend.schedule(
                F.write_json,
                data={str(k): v for k, v in payload["idToLabels"].items()},
                path=labels_path,
            )
        return

    # ------------------------------------------------------------------
    # Instance segmentation (support both fast and non-fast names)
    # ------------------------------------------------------------------
    if annotator_name.startswith("instance_segmentation"):
        instance_seg_data = payload["data"]
        height, width = instance_seg_data.shape[:2]

        png_path = base_prefix + ".png"
        # Follow MyWriter: handle both colorized (uint8) and raw (uint32) formats
        if instance_seg_data.dtype == np.uint32:
            instance_seg_rgba = instance_seg_data.view(np.uint8).reshape(
                height, width, -1
            )
        elif instance_seg_data.dtype == np.uint8 and len(instance_seg_data.shape) == 2:
            instance_seg_rgba = instance_seg_data.view(np.uint8).reshape(
                height, width, -1
            )
        else:
            instance_seg_rgba = instance_seg_data

        backend.schedule(F.write_image, data=instance_seg_rgba, path=png_path)

        if "idToLabels" in payload:
            labels_path = base_prefix + "_mapping.json"
            backend.schedule(
                F.write_json,
                data={str(k): v for k, v in payload["idToLabels"].items()},
                path=labels_path,
            )
        if "idToSemantics" in payload:
            sem_path = base_prefix + "_semantics_mapping.json"
            backend.schedule(
                F.write_json,
                data={str(k): v for k, v in payload["idToSemantics"].items()},
                path=sem_path,
            )
        return

    # ------------------------------------------------------------------
    # Bounding boxes (2D / 3D)
    # ------------------------------------------------------------------
    if annotator_name.startswith("bounding_box"):
        bbox_data = payload["data"]
        id_to_labels = payload.get("idToLabels", {})
        prim_paths = payload.get("primPaths", {})

        npy_path = base_prefix + ".npy"
        backend.schedule(F.write_np, data=bbox_data, path=npy_path)

        labels_path = base_prefix + "_labels.json"
        backend.schedule(F.write_json, data=id_to_labels, path=labels_path)

        prim_paths_path = base_prefix + "_prim_paths.json"
        backend.schedule(F.write_json, data=prim_paths, path=prim_paths_path)
        return

    # ------------------------------------------------------------------
    # Camera params
    # ------------------------------------------------------------------
    if annotator_name.startswith("camera_params"):
        camera_data = payload
        serializable_data = {}

        for key, val in camera_data.items():
            if isinstance(val, np.ndarray):
                serializable_data[key] = val.tolist()
            else:
                serializable_data[key] = val

        file_path = base_prefix + ".json"
        backend.schedule(F.write_json, data=serializable_data, path=file_path)
        return

    # ------------------------------------------------------------------
    # Pointcloud (simple version following MyWriter)
    # ------------------------------------------------------------------
    if annotator_name.startswith("pointcloud"):
        pointcloud_data = payload["data"]
        pointcloud_rgb = payload.get("pointRgb", None)
        pointcloud_normals = payload.get("pointNormals", None)
        pointcloud_semantic = payload.get("pointSemantic", None)
        pointcloud_instance = payload.get("pointInstance", None)

        pc_path = base_prefix + ".npy"
        backend.schedule(F.write_np, data=pointcloud_data, path=pc_path)

        if pointcloud_rgb is not None:
            rgb_path = base_prefix + "_rgb.npy"
            backend.schedule(F.write_np, data=pointcloud_rgb, path=rgb_path)
        if pointcloud_normals is not None:
            normals_path = base_prefix + "_normals.npy"
            backend.schedule(F.write_np, data=pointcloud_normals, path=normals_path)
        if pointcloud_semantic is not None:
            semantic_path = base_prefix + "_semantic.npy"
            backend.schedule(F.write_np, data=pointcloud_semantic, path=semantic_path)
        if pointcloud_instance is not None:
            instance_path = base_prefix + "_instance.npy"
            backend.schedule(F.write_np, data=pointcloud_instance, path=instance_path)
        return

    # ------------------------------------------------------------------
    # Skeleton data
    # ------------------------------------------------------------------
    if annotator_name.startswith("skeleton_data"):
        # For now, store raw payload as JSON-serializable as much as possible
        # (full MyWriter skeleton export is quite involved and usually not required here).
        def _to_serializable(obj):
            if hasattr(obj, "cpu"):
                obj = obj.cpu().numpy()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        file_path = base_prefix + ".json"
        backend.schedule(F.write_json, data=_to_serializable(payload), path=file_path)
        return

    # ------------------------------------------------------------------
    # Fallback: save raw data as npy
    # ------------------------------------------------------------------
    data = payload.get("data", payload)
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    npy_path = base_prefix + ".npy"
    backend.schedule(F.write_np, data=data, path=npy_path)
