from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from magicsim.Env.Utils.rotations import euler_to_quat

VOXEL_SIZE_M_DEFAULT = 0.05  # Default voxel size in meters
_PIL = None
_imageio = None
try:
    from PIL import Image as _PIL_Image  # type: ignore

    _PIL = _PIL_Image
except Exception:
    try:
        import imageio.v2 as _imageio  # type: ignore
    except Exception:
        _imageio = None


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in world meters."""

    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]

    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self.max_xyz[0] - self.min_xyz[0],
            self.max_xyz[1] - self.min_xyz[1],
            self.max_xyz[2] - self.min_xyz[2],
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_xyz[0] + self.max_xyz[0]) / 2.0,
            (self.min_xyz[1] + self.max_xyz[1]) / 2.0,
            (self.min_xyz[2] + self.max_xyz[2]) / 2.0,
        )


class AnnotationLoader:
    """
    Base loader: discovers files and caches JSON.

    Expected directory for one processed house (basename = file stem):
        <house_dir>/
          basename.json
          basename.png              # single floor map
          scene_slices/             # stack of slice PNGs
            basename_slice000.png
            ...
          room_boundaries/
            <arbitrary_room_jsons>.json  # any filenames supported
            room001/                    # room-specific PNGs
    """

    def __init__(self, house_dir: Union[str, Path]):
        self.house_dir = Path(house_dir)
        if not self.house_dir.exists():
            raise FileNotFoundError(f"House folder not found: {self.house_dir}")

        # pick a house-level json at the root
        top_jsons = [p for p in self.house_dir.glob("*.json") if p.is_file()]
        if not top_jsons:
            raise FileNotFoundError(
                f"No house boundary JSON found in {self.house_dir}. "
                f"Expected something like '<basename>.json' at the folder root."
            )
        if len(top_jsons) > 1:
            # Prefer one matching the folder name; else shortest stem
            prefer = [p for p in top_jsons if p.stem == self.house_dir.name]
            self.house_json = (
                prefer[0] if prefer else sorted(top_jsons, key=lambda p: len(p.stem))[0]
            )
        else:
            self.house_json = top_jsons[0]

        self._house_meta: Optional[Dict[str, Any]] = None
        self._room_index: Dict[int, Path] = {}
        self._room_meta_cache: Dict[int, Dict[str, Any]] = {}
        self._room_png_index: Dict[int, List[Path]] = {}  # Track room PNG files
        self._scene_slices: List[Path] = []
        self._floor_png: Optional[Path] = None

        self._discover_files()

    def _discover_files(self) -> None:
        # house json
        self._house_meta = json.loads(Path(self.house_json).read_text())

        # floor map
        floor = self.house_dir / f"{self.house_json.stem}.png"
        if floor.exists():
            self._floor_png = floor

        # slices
        slices_dir = self.house_dir / "scene_slices"
        if slices_dir.exists():
            self._scene_slices = sorted(
                slices_dir.glob(f"{self.house_json.stem}_slice*.png")
            )

        # rooms (permissive indexing over ANY *.json in room_boundaries/)
        room_dir = self.house_dir / "room_boundaries"
        if room_dir.exists():
            seen_ids = set()
            next_auto_id = 0
            for p in sorted(room_dir.glob("*.json")):
                rid: Optional[int] = None

                # try to read a declared id in JSON (id / room_id / roomId / rid)
                try:
                    data_obj = json.loads(p.read_text())
                    for key in ("id", "room_id", "roomId", "rid"):
                        if key in data_obj and isinstance(data_obj[key], (int, str)):
                            try:
                                rid = int(data_obj[key])
                                break
                            except Exception:
                                pass
                except Exception:
                    pass

                # permissive filename patterns
                if rid is None:
                    name = p.name
                    for pattern in [
                        r"_room(\d+)\.json$",  # ..._room12.json
                        r"room[_-]?(\d+)\.json$",  # room_12.json / room-12.json
                        r"(\d+)\.json$",  # 12.json
                    ]:
                        m = re.search(pattern, name, flags=re.IGNORECASE)
                        if m:
                            rid = int(m.group(1))
                            break

                # final fallback: assign stable incremental id
                if rid is None:
                    while next_auto_id in seen_ids:
                        next_auto_id += 1
                    rid = next_auto_id
                    next_auto_id += 1

                # avoid collisions
                if rid in seen_ids:
                    while next_auto_id in seen_ids:
                        next_auto_id += 1
                    rid = next_auto_id
                    next_auto_id += 1

                seen_ids.add(rid)
                self._room_index[rid] = p

                # Discover room PNG files in subdirectories
                # Look for room001/, room002/, etc.
                room_subdir = room_dir / f"room{rid:03d}"
                if room_subdir.exists() and room_subdir.is_dir():
                    room_pngs = sorted(room_subdir.glob("*.png"))
                    if room_pngs:
                        self._room_png_index[rid] = room_pngs

    # ---------- helpers & properties ----------

    @property
    def house_meta(self) -> Dict[str, Any]:
        return dict(self._house_meta) if self._house_meta is not None else {}

    @property
    def voxel_size(self) -> Optional[float]:
        # probe any room json
        for rid in self._room_index:
            meta = self.get_room_by_id(rid)
            vs = meta.get("voxel_size_m")
            if isinstance(vs, (int, float)):
                return float(vs)
        return None

    @property
    def z_range(self) -> Optional[Tuple[float, float]]:
        h = self.house_meta
        try:
            return float(h["min"][2]), float(h["max"][2])
        except Exception:
            return None

    def _warn(self, msg: str) -> None:
        print(f"[Room][WARN] {msg}")

    # convenience for external callers
    def list_room_ids(self) -> List[int]:
        """Return sorted list of discovered room IDs."""
        return sorted(self._room_index.keys())

    def list_room_files(self) -> List[Tuple[int, Path]]:
        """Return [(room_id, path), ...] sorted by room_id."""
        return [(rid, self._room_index[rid]) for rid in self.list_room_ids()]

    def get_room_by_id(self, room_id: int) -> Dict[str, Any]:
        """Return the raw JSON dict for a room id (after permissive indexing)."""
        if room_id in self._room_meta_cache:
            return dict(self._room_meta_cache[room_id])
        p = self._room_index.get(room_id)
        if p is None:
            raise KeyError(f"Room id not found: {room_id}")
        data = json.loads(p.read_text())
        self._room_meta_cache[room_id] = data
        return dict(data)


class Room(AnnotationLoader, SingleGeometryPrim):
    """
    Accessor for house-level annotations & per-room geometry.

    Methods:
    - get_house_omap(height=0.5) -> np.ndarray (N,3): (x, y, label) in house/world frame
    - get_house_omap_img(height=0.5) -> np.ndarray (raw image array)
    - get_room_bb_world(room_id) -> BoundingBox relative to world
    - get_house_bb() -> BoundingBox
    - get_room_num() -> int
    - get_room_boundary(room_id) -> np.ndarray (N,2): (x, y) in room frame
    - get_room_bb_world(room_id) -> BoundingBox relative to world
    - get_room_bb_local(room_id) -> BoundingBox relative to room center
    - get_room_omap(room_id, height=0.5) -> np.ndarray (N,3): (x, y, label) in room frame
    - get_room_omap_img(room_id, height=0.5) -> np.ndarray (raw image array)
    """

    def __init__(
        self,
        annotation_dir: Union[str, Path] = None,
        prim_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        room_config: Optional[DictConfig] = None,
        instance_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Room object with both annotation loading and geometry creation support.

        Args:
            annotation_dir: Directory path for room annotation data
            prim_path: Prim path in Isaac Sim (optional)
            usd_path: USD asset file path (optional, if None will wrap existing prim)
            room_config: Top-level room configuration (optional)
            instance_config: Instance configuration dictionary (optional)
        """
        # Initialize SingleGeometryPrim if prim_path is provided
        if prim_path is not None:
            # Load USD asset if usd_path is provided
            if usd_path is not None:
                room_prim = add_reference_to_stage(
                    usd_path=usd_path, prim_path=prim_path
                )
                if not room_prim or not room_prim.IsValid():
                    raise RuntimeError(
                        f"Failed to load USD from {usd_path} to {prim_path}"
                    )

            # Initialize SingleGeometryPrim
            collision = room_config.get("collision", True) if room_config else True
            SingleGeometryPrim.__init__(
                self,
                prim_path=prim_path,
                name=prim_path.split("/")[-1],
                collision=collision,
            )

            # Set pose (position, orientation, scale)
            if instance_config is None and room_config is not None:
                # Parse instance_config from room_config
                path_parts = prim_path.split("/")
                instance_name = path_parts[-1]
                category_name = instance_name.rsplit("_", 1)[0]
                category_config = room_config.get(category_name, {})
                instance_config = category_config.get(instance_name, {})

            if instance_config:
                pos = instance_config.get("pos", [0.0, 0.0, 0])
                ori_euler = instance_config.get("ori", [0.0, 0.0, 0.0])
                ori_quat = euler_to_quat(ori_euler)
                self.set_local_pose(translation=pos, orientation=ori_quat)
                scale = instance_config.get("scale", [1.0, 1.0, 1.0])
                self.set_local_scale(scale)

        # Initialize AnnotationLoader
        if annotation_dir is not None:
            AnnotationLoader.__init__(self, annotation_dir)
            print(
                f"Annotation loader initialized with annotation directory: {annotation_dir}"
            )

    def reset(self):
        """
        A reset interface reserved for the Room class. Currently, hard resets are handled
        by the SceneManager by creating a new instance.
        Soft reset logic (e.g., changing only colors or lights) could be added here in the future.
        """
        pass

    # ---- house ----
    def get_house_omap_img(
        self,
        height: Optional[float] = None,
    ) -> np.ndarray:
        """
        Returns raw image array from the occupancy map PNG.
        If multiple z-slices exist, chooses the closest to requested height in meters;
        otherwise returns the single floor PNG and warns if a height was requested.
        """
        # Preferred: scene slices exist
        if self._scene_slices:
            zr = self.z_range
            vs = self.voxel_size
            if not zr or vs is None:
                self._warn("Missing z-range or voxel_size; selecting middle slice.")
                idx = len(self._scene_slices) // 2
                chosen = self._scene_slices[idx]
                return _read_png_array(chosen)

            zmin, zmax = zr
            start_z = zmin + vs
            target_z = (zmin + zmax) / 2.0 if height is None else float(height)
            idx = int(round((target_z - start_z) / max(vs, 1e-9)))
            idx = max(0, min(idx, len(self._scene_slices) - 1))
            chosen = self._scene_slices[idx]
            z_sel = start_z + idx * vs

            if height is not None and abs(z_sel - target_z) > 1e-6:
                self._warn(
                    f"Requested z={target_z:.3f}m, using closest slice at z={z_sel:.3f}m."
                )

            return _read_png_array(chosen)

        # Fallback: single floor map
        if self._floor_png and self._floor_png.exists():
            if height is not None:
                self._warn(
                    "Only a single floor map exists; returning it regardless of requested height."
                )
            return _read_png_array(self._floor_png)

        raise FileNotFoundError("No occupancy map images found.")

    def get_house_omap(
        self,
        height: Optional[float] = None,
    ) -> np.ndarray:
        """
        Returns occupancy map as labeled array.

        Args:
            height: Optional height in meters for z-slice selection

        Returns:
            np.ndarray:
                A (N, 3) float32 array where each row is (x, y, label).
                - x, y are house/world coordinates in meters, derived from the scene center.
                - label is occupancy:
                    1  = occupied
                    0  = free
                    -1  = unknown
        """
        img = self.get_house_omap_img(height=height)
        labels = omap_to_npy(img, save_path=None)
        r = self.house_meta
        ref_origin = r.get("footprint_xy")
        if ref_origin is None or "xmin_ymax" not in ref_origin:
            raise RuntimeError("House JSON missing 'footprint_xy[\"xmin_ymax\"]'.")
        x_ref = ref_origin["xmin_ymax"][0]
        y_ref = ref_origin["xmin_ymax"][1]
        H, W = labels.shape
        pts = np.empty((H * W, 3), dtype=np.float32)
        idx = 0

        for y_prime in range(H):
            for x_prime in range(W):
                # Convert pixel coordinates to local coordinates
                x = x_prime * VOXEL_SIZE_M_DEFAULT + x_ref
                y = -y_prime * VOXEL_SIZE_M_DEFAULT + y_ref

                # Get the label at this pixel
                label = int(labels[y_prime, x_prime])

                pts[idx, 0] = x
                pts[idx, 1] = y
                pts[idx, 2] = label
                idx += 1

        return pts

    def get_house_bb(self) -> BoundingBox:
        h = self.house_meta
        try:
            return BoundingBox(tuple(h["min"]), tuple(h["max"]))
        except Exception as e:
            raise RuntimeError(f"Malformed house JSON (missing 'min'/'max'): {e}")

    # ---- rooms ----
    def get_room_num(self) -> int:
        """Return the number of room JSON files (number of rooms)."""
        return len(self._room_index)

    def get_room_boundary(self, room_id: int) -> np.ndarray:
        """
        Get room boundary coordinates from precomputed boundary_coord field.

        Args:
            room_id: The room ID

        Returns:
            np.ndarray: Array of shape (N, 2) with (x, y) coordinates in world meters.
        """
        r = self.get_room_by_id(room_id)
        boundary_coord = r.get("boundary_coord")

        if boundary_coord is None:
            raise RuntimeError(
                f"Room {room_id} JSON missing 'boundary_coord' field. "
                f"Please run augment_room_json.py to add precomputed fields."
            )

        return np.asarray(boundary_coord, dtype=float)

    def get_room_bb(self, room_id: int) -> BoundingBox:
        """Get room bounding box in house coordinates."""
        r = self.get_room_by_id(room_id)

        # Use precomputed boundary_coord
        xyz_bounds = r.get("xyz_bounds")
        if xyz_bounds is None:
            raise RuntimeError(f"Room {room_id} JSON missing 'xyz_bounds' field. ")

        x_min = xyz_bounds["x_min"]
        x_max = xyz_bounds["x_max"]
        y_min = xyz_bounds["y_min"]
        y_max = xyz_bounds["y_max"]
        z_min = xyz_bounds["z_min"]
        z_max = xyz_bounds["z_max"]

        return BoundingBox((x_min, y_min, z_min), (x_max, y_max, z_max))

    def get_room_omap_img(
        self, room_id: int, height: Optional[float] = None
    ) -> np.ndarray:
        """
        Get room occupancy map as raw image array from room-specific PNG files.

        Args:
            room_id: The room ID
            height: Optional height in meters for z-slice selection

        Returns:
            np.ndarray: Raw image array (RGB/RGBA/grayscale) for the room
        """
        # Check if room has its own PNG files
        room_pngs = self._room_png_index.get(room_id, [])

        if not room_pngs:
            raise FileNotFoundError(f"No PNG files found for room {room_id}")

        # If only one PNG, return it (with warning if height was requested)
        if len(room_pngs) == 1:
            if height is not None:
                self._warn(
                    f"Room {room_id} has only one PNG; returning it regardless of requested height."
                )
            return _read_png_array(room_pngs[0])

        # Multiple PNGs - select based on height
        r = self.get_room_by_id(room_id)
        vs = r.get("voxel_size_m")
        zr = self.z_range

        if not zr or vs is None:
            self._warn(
                f"Missing z-range or voxel_size for room {room_id}; selecting middle PNG."
            )
            idx = len(room_pngs) // 2
            return _read_png_array(room_pngs[idx])

        zmin, zmax = zr
        start_z = zmin + vs
        target_z = (zmin + zmax) / 2.0 if height is None else float(height)
        idx = int(round((target_z - start_z) / max(vs, 1e-9)))
        idx = 16
        chosen = room_pngs[idx]
        z_sel = start_z + idx * vs

        if height is not None and abs(z_sel - target_z) > 1e-6:
            self._warn(
                f"Room {room_id}: Requested z={target_z:.3f}m, using closest slice at z={z_sel:.3f}m."
            )

        return _read_png_array(chosen)

    def get_room_omap(self, room_id: int, height: Optional[float] = None) -> np.ndarray:
        """
        Get room occupancy map as a point array in room-centered coordinates.

        Args:
            room_id: The room ID
            height: Optional height in meters for z-slice selection

        Returns:
            np.ndarray:
                A (N, 3) float32 array where each row is (x, y, label).
                - x, y are room-centered coordinates in meters.
                - label is occupancy:
                    1  = occupied
                    0  = free
                    -1  = unknown
        """
        # Get the raw image
        img = self.get_room_omap_img(room_id, height=height)
        labels = omap_to_npy(img, save_path=None)

        # Get room metadata
        r = self.get_room_by_id(room_id)
        pixel_bound = r.get("bbox_pixels")
        reference_origin = r.get("origin_xy_m")
        world_center = r.get("center_in_world")
        voxel_size_m = r.get("voxel_size_m")

        if (
            pixel_bound is None
            or reference_origin is None
            or world_center is None
            or voxel_size_m is None
        ):
            raise RuntimeError(
                f"Room {room_id} JSON missing required fields: bbox_pixels, origin_xy_m, center_in_world, voxel_size_m"
            )

        x_start = pixel_bound["x_start"]
        y_start = pixel_bound["y_start"]

        # Build the coordinate mapping
        H, W = labels.shape
        pts = np.empty((H * W, 3), dtype=np.float32)  # (x, y, label)
        idx = 0

        for y_prime in range(H):
            for x_prime in range(W):
                # Convert pixel coordinates to local coordinates
                x = (
                    (x_prime + x_start) * voxel_size_m
                    + reference_origin[0]
                    - world_center[0]
                )
                y = (
                    -(y_prime + y_start) * voxel_size_m
                    + reference_origin[1]
                    - world_center[1]
                )

                # Get the label at this pixel
                label = int(labels[y_prime, x_prime])

                pts[idx, 0] = x
                pts[idx, 1] = y
                pts[idx, 2] = label
                idx += 1

        return pts

    def get_room_free_point(
        self,
        room_id: int,
        height: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get free points in a room where a square region of specified radius around each point
        is completely free (no occupied or unknown cells).

        Args:
            room_id: The room ID
            height: Optional height in meters for z-slice selection
            radius: Optional radius in meters. If specified (e.g., 2.0), checks that a
                    radius*2 x radius*2 square (e.g., 2m x 2m) centered on each free point
                    contains only free cells. The actual pixel count is computed from voxel_size.

        Returns:
            np.ndarray:
                A (N, 2) float32 array where each row is (x, y) in room-centered coordinates.
                Only includes free points where the surrounding square region (if radius is specified)
                is completely free.
        """
        # Get the raw image and labels
        img = self.get_room_omap_img(room_id, height=height)
        labels = omap_to_npy(img, save_path=None)

        # Get room metadata
        r = self.get_room_by_id(room_id)
        pixel_bound = r.get("bbox_pixels")
        reference_origin = r.get("origin_xy_m")
        world_center = r.get("center_in_world")
        voxel_size_m = r.get("voxel_size_m")

        if (
            pixel_bound is None
            or reference_origin is None
            or world_center is None
            or voxel_size_m is None
        ):
            raise RuntimeError(
                f"Room {room_id} JSON missing required fields: bbox_pixels, origin_xy_m, center_in_world, voxel_size_m"
            )

        x_start = pixel_bound["x_start"]
        y_start = pixel_bound["y_start"]
        H, W = labels.shape

        # Convert radius from meters to pixels (half-width of the square)
        if radius is not None:
            radius_pixels = int(np.ceil(radius / max(voxel_size_m, 1e-9)))
        else:
            radius_pixels = 0

        # Collect valid free points
        free_points = []

        for y_prime in range(H):
            for x_prime in range(W):
                # Check if this pixel is free
                if labels[y_prime, x_prime] != 0:
                    continue

                # If radius is specified, check the surrounding square region
                if radius_pixels > 0:
                    # Define the square region bounds
                    x_min = max(0, x_prime - radius_pixels)
                    x_max = min(W, x_prime + radius_pixels + 1)
                    y_min = max(0, y_prime - radius_pixels)
                    y_max = min(H, y_prime + radius_pixels + 1)

                    # Extract the region
                    region = labels[y_min:y_max, x_min:x_max]

                    # Check if all cells in the region are free (label == 0)
                    if not np.all(region == 0):
                        continue

                # Convert pixel coordinates to room-centered coordinates
                x = (
                    (x_prime + x_start) * voxel_size_m
                    + reference_origin[0]
                    - world_center[0]
                )
                y = (
                    -(y_prime + y_start) * voxel_size_m
                    + reference_origin[1]
                    - world_center[1]
                )

                free_points.append([x, y])

        if not free_points:
            return np.empty((0, 2), dtype=np.float32)

        return np.array(free_points, dtype=np.float32)


# ---------- image reading helper ----------


def _read_png_array(path: Union[str, Path]) -> np.ndarray:
    """Read a PNG file into a numpy array using PIL or imageio, with a clear error if neither available."""
    p = Path(path)
    if _PIL is not None:
        with _PIL.open(p) as im:
            return np.array(im)
    if _imageio is not None:
        return _imageio.imread(p)
    raise ImportError(
        "No image backend available. Please install Pillow (`pip install pillow`) or imageio (`pip install imageio`)."
    )


def omap_to_npy(omap: np.ndarray, save_path: str | None = None) -> np.ndarray:
    """
    Convert a raw RGBA/RGB/gray occupancy PNG (NumPy array) to a 2D label map
    and optionally save it as a .npy file.

    Label convention:
        1 = occupied
        0 = free / unoccupied
       -1 = unknown
    """
    a = omap

    # --- Convert to grayscale ---
    if a.ndim == 3:  # RGB or RGBA
        rgb = a[..., :3].astype(np.float32)
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
            np.uint8
        )
    else:
        g = a.astype(np.uint8)

    labels = np.full(g.shape, -1, dtype=np.int8)

    # --- Case 1: occupancy values 4/5/6 (Isaac/Omni convention) ---
    if np.isin(g, [4, 5, 6]).any():
        labels[g == 4] = 1  # occupied
        labels[g == 5] = 0  # free
        labels[g == 6] = -1  # unknown
    else:
        # --- Case 2: color intensity (black/white/gray) ---
        labels[g <= 10] = 1  # black → occupied
        labels[g >= 245] = 0  # white → free

    # --- Save to npy if requested ---
    if save_path is not None:
        p = Path(save_path)
        if p.suffix.lower() != ".npy":
            p = p.with_suffix(".npy")
        p.parent.mkdir(parents=True, exist_ok=True)

        np.save(p, labels)
        reloaded = np.load(p)
        if reloaded.shape != labels.shape or reloaded.dtype != labels.dtype:
            raise IOError("Save verification failed (shape/dtype mismatch).")
        abs_path = str(p.resolve())
        print(f"[INFO] Saved occupancy map to {abs_path}")

    return labels
