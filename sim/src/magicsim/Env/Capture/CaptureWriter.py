"""
CaptureWriter: Utility functions for writing CaptureManager.step() output to disk.

This module provides functions to save capture data from CaptureManager.step()
to disk with a simple folder structure: path/env_id/cam_id/annotator_name/
"""

import os
from typing import Any, Dict, List, Optional
from omni.replicator.core.scripts.backends import BackendDispatch
from magicsim.Collect.Record.CameraWriter import write_annotator_step


def convert_annotator_data_to_payload(
    annotator_name: str, anno_data: Any
) -> Dict[str, Any]:
    """Convert annotator.get_data() output to payload format matching TiledCaptureManager.

    This handles the case where anno_data may have {"data": ..., "info": {...}} structure
    and needs to be flattened based on annotator type.

    Args:
        annotator_name: Name of the annotator (e.g., "rgb", "distance_to_camera")
        anno_data: Raw data from annotator.get_data()

    Returns:
        Payload dictionary in format expected by write_annotator_step
    """
    # If not a dict, wrap it
    if not isinstance(anno_data, dict):
        return {"data": anno_data}

    # Check if it has the nested "info" structure
    if "info" in anno_data and "data" in anno_data:
        info = anno_data["info"]
        data = anno_data["data"]

        # Convert data to numpy if it's a torch tensor
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        elif hasattr(data, "numpy"):
            data = data.numpy()

        # Build payload based on annotator type (matching TiledCaptureManager logic)
        if annotator_name == "rgb":
            return {"data": data}

        elif annotator_name == "normals":
            return {"data": data}

        elif annotator_name.startswith("semantic_segmentation"):
            return {
                "data": data,
                "idToLabels": info.get("idToLabels", {}),
            }

        elif annotator_name.startswith("instance_id_segmentation"):
            return {
                "data": data,
                "idToLabels": info.get("idToLabels", {}),
            }

        elif annotator_name.startswith("instance_segmentation"):
            return {
                "data": data,
                "idToLabels": info.get("idToLabels", {}),
                "idToSemantics": info.get("idToSemantics", {}),
            }

        elif annotator_name.startswith("bounding_box"):
            return {
                "data": data,
                "idToLabels": info.get("idToLabels", {}),
                "primPaths": info.get("primPaths", {}),
            }

        elif annotator_name == "camera_params":
            # camera_params returns the whole dict
            return anno_data

        elif annotator_name == "pointcloud" or annotator_name == "skeleton_data":
            # pointcloud and skeleton_data return the whole dict
            return anno_data

        else:
            # Fallback: just use data field
            return {"data": data}

    else:
        # Already in flattened format (from TiledCaptureManager) or simple dict
        # Ensure data is numpy if it's a tensor
        if "data" in anno_data:
            data = anno_data["data"]
            if hasattr(data, "cpu"):
                anno_data["data"] = data.cpu().numpy()
            elif hasattr(data, "numpy"):
                anno_data["data"] = data.numpy()
        return anno_data


def write_capture_data(
    capture_data: List[Dict[str, List[Any]]],
    step_idx: int,
    path: str,
    env_ids: Optional[List[int]] = None,
):
    """Write CaptureManager.step() output to disk.

    Creates folder structure: path/env_id/cam_id/annotator_name/
    and saves files as step_{step_idx:04d}.* (e.g., step_0000.png, step_0000.npy)

    Args:
        capture_data: Output from CaptureManager.step()
            Format: List[Dict[str, List[Any]]]
            - Outer list indexed by cam_id
            - Dict keys are annotator names
            - Inner list contains data for each env_id in order
            - Format: data[cam_id][annotator_name][env_index] = annotator_data
        step_idx: Step index for file naming (e.g., 0, 1, 2, ...)
        path: Base path where data will be saved
        env_ids: Optional list of environment IDs to save. If None, saves all environments
            found in the data. Defaults to None.

    Example:
        >>> capture_data = capture_manager.step()
        >>> write_capture_data(capture_data, step_idx=0, path="/path/to/output")
        # Creates:
        # /path/to/output/0/cam_0/rgb/step_0000.png
        # /path/to/output/0/cam_0/distance_to_camera/step_0000.npy
        # /path/to/output/0/cam_0/distance_to_camera/step_0000.png
        # /path/to/output/1/cam_0/rgb/step_0000.png
        # ...
    """
    if not capture_data:
        return

    # Determine number of cameras and environments
    num_cams = len(capture_data)
    if num_cams == 0:
        return

    # Get annotator names from first camera (all cameras should have same annotators)
    first_cam_data = capture_data[0]
    if not first_cam_data:
        return

    # Determine number of environments from first annotator's list length
    first_annotator_name = next(iter(first_cam_data.keys()))
    num_envs = len(first_cam_data[first_annotator_name])

    # If env_ids not specified, save all environments
    if env_ids is None:
        env_ids = list(range(num_envs))

    # Create backend for saving (paths are relative to base path)
    backend = BackendDispatch(output_dir=path)

    # Iterate over each camera
    for cam_id in range(num_cams):
        if cam_id >= len(capture_data):
            continue

        cam_data = capture_data[cam_id]
        if not cam_data:
            continue

        # Iterate over each annotator
        for annotator_name, env_list in cam_data.items():
            if not isinstance(env_list, list):
                continue

            # Iterate over each environment
            for env_id in env_ids:
                if env_id >= len(env_list):
                    continue

                anno_data = env_list[env_id]
                if anno_data is None:
                    continue

                # Create directory structure: path/env_id/cam_id/annotator_name/
                annotator_dir_rel = os.path.join(
                    str(env_id), f"cam_{cam_id}", annotator_name
                )
                annotator_dir_abs = os.path.join(path, annotator_dir_rel)
                os.makedirs(annotator_dir_abs, exist_ok=True)

                # Convert annotator data to payload format
                payload = convert_annotator_data_to_payload(annotator_name, anno_data)

                # Write using shared helper function
                write_annotator_step(
                    backend=backend,
                    annotator_name=annotator_name,
                    payload=payload,
                    annotator_dir_rel=annotator_dir_rel,
                    step_idx=step_idx,
                )
