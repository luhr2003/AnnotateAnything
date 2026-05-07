"""Utility functions for RecordManager.

This module contains helper functions that can be used by RecordManager
for data conversion, validation, and processing.
"""

from typing import Any, Dict
import torch
import numpy as np


def to_serializable(value):
    """Convert data (including torch / numpy) to JSON-serializable types."""
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    if isinstance(value, np.ndarray):
        # Convert to list and round floats
        result = value.tolist()
        # Recursively round floats in the list
        return to_serializable(result)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    # Round float values to 4 decimal places
    if isinstance(value, (float, np.floating, np.integer)):
        # Convert numpy types to Python native types and round floats
        if isinstance(value, (float, np.floating)):
            return round(float(value), 4)
        else:
            # For integers, just convert to Python int
            return int(value)
    return value


def extract_env_from_dict(data: Dict[str, Any], env_id: int) -> Dict[str, Any]:
    """Recursively extract single env data from batched dict."""
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[env_id]
        elif isinstance(value, dict):
            result[key] = extract_env_from_dict(value, env_id)
        elif isinstance(value, list):
            # Handle lists (may contain tensors, dicts or other structures)
            per_env_list = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    # List of batched tensors -> take env slice
                    per_env_list.append(item[env_id])
                elif isinstance(item, dict):
                    # List of dicts -> recursively extract per-env data
                    per_env_list.append(extract_env_from_dict(item, env_id))
                else:
                    # Other types are kept as-is
                    per_env_list.append(item)
            result[key] = per_env_list
        else:
            result[key] = value
    return result


def check_dict_values_length(
    data: Dict[str, Any], expected_length: int, path: str = ""
) -> None:
    """Recursively check that all innermost values in a dictionary have the expected length.

    Args:
        data: Dictionary to check (can be nested)
        expected_length: Expected length (num_envs)
        path: Current path in the dictionary (for error messages)
    """
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, torch.Tensor):
            # Check tensor first dimension
            if value.ndim > 0:
                actual_length = value.shape[0]
                assert actual_length == expected_length, (
                    f"Value at path '{current_path}' has length {actual_length}, "
                    f"expected {expected_length}. Shape: {value.shape}"
                )
            # Scalar tensors are allowed (they don't have a first dimension)
        elif isinstance(value, dict):
            # Recursively check nested dictionaries
            check_dict_values_length(value, expected_length, current_path)
        elif isinstance(value, (list, tuple)):
            # For lists/tuples, check each item
            # Note: The list itself may not have length num_envs (e.g., list of observation managers)
            # but the items inside (if dicts or tensors) should have num_envs length
            for i, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    # Check tensor first dimension
                    if item.ndim > 0:
                        actual_item_length = item.shape[0]
                        assert actual_item_length == expected_length, (
                            f"Value at path '{current_path}[{i}]' has length {actual_item_length}, "
                            f"expected {expected_length}. Shape: {item.shape}"
                        )
                elif isinstance(item, dict):
                    # Recursively check nested dicts in list
                    check_dict_values_length(
                        item, expected_length, f"{current_path}[{i}]"
                    )
                # Other types in list are allowed (e.g., None, strings, etc.)
        # Other types (None, int, float, str, etc.) are allowed and skipped


def is_image_annotator(annotator_name: str) -> bool:
    """Check if annotator outputs image data."""
    image_annotators = [
        "rgb",
        "normals",
        "distance_to_camera",
        "distance_to_image_plane",
        "semantic_segmentation",
        "instance_id_segmentation",
        "instance_segmentation",
    ]
    return any(annotator_name.startswith(prefix) for prefix in image_annotators)


def is_json_annotator(annotator_name: str) -> bool:
    """Check if annotator outputs JSON data."""
    json_annotators = ["camera_params", "skeleton_data"]
    return any(annotator_name.startswith(prefix) for prefix in json_annotators)


def is_numpy_annotator(annotator_name: str) -> bool:
    """Check if annotator outputs numpy data."""
    numpy_annotators = ["bounding_box", "pointcloud"]
    return any(annotator_name.startswith(prefix) for prefix in numpy_annotators)


def extract_image_data(annotator_name: str, payload: Dict[str, Any]):
    """Extract image data from payload for video encoding.

    Returns RGB (3-channel) uint8 numpy array with shape (H, W, 3).
    """
    from omni.replicator.core.scripts.writers_default.tools import (
        colorize_distance,
        colorize_normals,
    )

    data = payload.get("data")
    if data is None:
        return None

    # Convert to numpy if needed
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    elif hasattr(data, "numpy"):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    # Handle different annotator types
    if annotator_name.startswith("rgb"):
        # RGB: ensure uint8 and correct shape
        if data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.floating):
                data = np.clip(data * 255.0, 0, 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)
        # Ensure RGB format (3 channels)
        if len(data.shape) == 2:
            data = np.stack([data] * 3, axis=-1)
        elif len(data.shape) == 3:
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            elif data.shape[2] == 4:
                data = data[:, :, :3]  # RGBA to RGB
            elif data.shape[2] != 3:
                raise ValueError(f"Unexpected RGB shape: {data.shape}")
        return data

    elif annotator_name.startswith("normals"):
        # Normals: colorize
        img = colorize_normals(data)
        # Ensure RGB format
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # RGBA to RGB
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.uint8)

    elif annotator_name.startswith("distance_to_camera") or annotator_name.startswith(
        "distance_to_image_plane"
    ):
        # Distance: colorize
        if data.dtype not in [np.float32, np.float64]:
            data = data.astype(np.float32)
        valid_data = data[(data != -np.inf) & (data != np.inf) & ~np.isnan(data)]
        if len(valid_data) > 0:
            near = np.min(valid_data)
            far = np.max(valid_data)
            img = colorize_distance(data, near=near, far=far)
        else:
            img = np.zeros((*data.shape[:2], 4), dtype=np.uint8)
        # Ensure RGB format
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # RGBA to RGB
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.uint8)

    elif (
        annotator_name.startswith("semantic_segmentation")
        or annotator_name.startswith("instance_id_segmentation")
        or annotator_name.startswith("instance_segmentation")
    ):
        # Segmentation: handle uint32/uint8 formats
        height, width = data.shape[:2]
        if data.dtype == np.uint32:
            img = data.view(np.uint8).reshape(height, width, -1)
        elif data.dtype == np.uint8 and len(data.shape) == 2:
            img = data.view(np.uint8).reshape(height, width, -1)
        else:
            img = data
        # Ensure RGB format
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]  # RGBA to RGB
            elif img.shape[2] != 3:
                # If unexpected channels, convert to grayscale then RGB
                if img.shape[2] > 3:
                    img = img[:, :, :3]
                else:
                    img = np.repeat(img[:, :, 0:1], 3, axis=2)
        return img.astype(np.uint8)

    return None


def extract_json_data(annotator_name: str, payload: Dict[str, Any]):
    """Extract JSON data from payload."""
    if annotator_name.startswith("camera_params"):
        # Camera params: convert numpy arrays to lists
        serializable_data = {}
        for key, val in payload.items():
            if isinstance(val, np.ndarray):
                serializable_data[key] = val.tolist()
            else:
                serializable_data[key] = val
        return serializable_data

    elif annotator_name.startswith("skeleton_data"):
        # Skeleton data: convert to serializable
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

        return _to_serializable(payload)

    return None


def extract_json_metadata(annotator_name: str, payload: Dict[str, Any]):
    """Extract JSON metadata from image annotators (e.g., labels for segmentation)."""
    metadata = {}
    if (
        annotator_name.startswith("semantic_segmentation")
        or annotator_name.startswith("instance_id_segmentation")
        or annotator_name.startswith("instance_segmentation")
    ):
        if "idToLabels" in payload:
            metadata["idToLabels"] = payload["idToLabels"]
        if "idToSemantics" in payload:
            metadata["idToSemantics"] = payload["idToSemantics"]
    return metadata if metadata else None


def extract_numpy_data(annotator_name: str, payload: Dict[str, Any]):
    """Extract numpy data from payload."""
    if annotator_name.startswith("bounding_box"):
        # Bounding box: return data and metadata
        return {
            "data": payload.get("data"),
            "idToLabels": payload.get("idToLabels", {}),
            "primPaths": payload.get("primPaths", {}),
        }

    elif annotator_name.startswith("pointcloud"):
        # Pointcloud: return all pointcloud data
        return {
            "data": payload.get("data"),
            "pointRgb": payload.get("pointRgb"),
            "pointNormals": payload.get("pointNormals"),
            "pointSemantic": payload.get("pointSemantic"),
            "pointInstance": payload.get("pointInstance"),
        }

    return None


def convert_annotator_data_to_payload(
    annotator_name: str, anno_data: Any
) -> Dict[str, Any]:
    """Convert annotator.get_data() output to payload format matching TiledCaptureManager.

    This handles the case where anno_data may have {"data": ..., "info": {...}} structure
    and needs to be flattened based on annotator type.
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
