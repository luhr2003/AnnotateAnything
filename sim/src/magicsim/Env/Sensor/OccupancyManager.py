from __future__ import annotations


from magicsim.Env.Environment.Isaac import IsaacRLEnv
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Union

import numpy as np
import torch

import omni
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.core.utils.extensions import enable_extension
from skimage.draw import line as sk_line

enable_extension("isaacsim.asset.gen.omap")


@dataclass
class OMapValues:
    occupied: float = 0.0
    unoccupied: float = 1.0
    unknown: float = 2.0


class OccupancyManager:
    """
    Occupancy map manager for multi-environment scanning.

    Coordinate System:
    ------------------
    - origin: Scan center in LOCAL coordinates [x, y, z] (relative to env_origin)
    - boundary: Scan boundary in LOCAL coordinates [x_min, x_max, y_min, y_max, z_min, z_max]
    - Internally converts to WORLD coordinates for scanning: world = env_origin + local
    """

    def __init__(
        self,
        num_envs: int,
        device: Union[str, torch.device],
        cell_size: float,
        values: Iterable[float] = (0.0, 1.0, 2.0),
    ) -> None:
        if omni is None or _omap is None:
            raise RuntimeError(
                "Isaac Sim environment not detected. Import and run inside Isaac Sim."
            )

        self.num_envs = num_envs
        self.cell_size = float(cell_size)
        self.values = OMapValues(*values)
        # Verbose prints (set transform / occupied / free counts) — off by default.
        # Toggle via ``occupancy_manager.verbose = True`` (e.g. from Dwb when debug=True).
        self.verbose: bool = False

        self._gen: Optional[_omap.Generator] = None
        self._physx = None
        self.grid: List[Optional[np.ndarray]] = [None] * num_envs
        self.boundary: List[Optional[np.ndarray]] = [None] * num_envs
        self.origin: List[Optional[np.ndarray]] = [None] * num_envs
        self.grid_w: List[int] = [0] * num_envs
        self.grid_h: List[int] = [0] * num_envs

        self.sim: Optional[IsaacRLEnv] = None
        self.env_origins: Optional[torch.Tensor] = None

    def _normalize_env_ids(self, env_ids: Optional[List[int]]) -> List[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        return env_ids

    def _normalize_env_ids_with_single(
        self, env_ids: Optional[List[int]]
    ) -> Tuple[List[int], bool]:
        if env_ids is None:
            return list(range(self.num_envs)), False
        return env_ids, len(env_ids) == 1

    def _normalize_value(self, value, env_id: int):
        arr = np.array(value, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        idx = env_id if env_id < len(arr) else 0
        return np.array(arr[idx], dtype=np.float32)

    def _update_grid_dimensions(self, env_id: int) -> None:
        """Update grid width and height based on boundary and cell_size."""
        if self.boundary[env_id] is not None:
            self.grid_w[env_id] = int(
                (self.boundary[env_id][1] - self.boundary[env_id][0]) / self.cell_size
            )
            self.grid_h[env_id] = int(
                (self.boundary[env_id][3] - self.boundary[env_id][2]) / self.cell_size
            )

    def initialize(self, sim: IsaacRLEnv) -> None:
        self.sim = sim
        self.env_origins = sim.scene.env_origins
        self._physx = omni.physx.get_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()

        self._gen = _omap.Generator(self._physx, stage_id)
        self._gen.update_settings(
            self.cell_size,
            self.values.occupied,
            self.values.unoccupied,
            self.values.unknown,
        )

    def update_settings(
        self,
        cell_size: Optional[float] = None,
        values: Optional[Iterable[float]] = None,
        env_id: Optional[int] = None,
    ) -> None:
        if cell_size is not None:
            self.cell_size = float(cell_size)
        if values is not None:
            self.values = OMapValues(*values)

        if self._gen is not None:
            self._gen.update_settings(
                self.cell_size,
                self.values.occupied,
                self.values.unoccupied,
                self.values.unknown,
            )

        if env_id is not None:
            self._update_grid_dimensions(env_id)
            self.grid[env_id] = None

    def _set_transform_internal(self, env_id: int) -> None:
        if self._gen is None:
            return
        min_bounds = [
            float(self.boundary[env_id][0]),
            float(self.boundary[env_id][2]),
            float(self.boundary[env_id][4]),
        ]
        max_bounds = [
            float(self.boundary[env_id][1]),
            float(self.boundary[env_id][3]),
            float(self.boundary[env_id][5]),
        ]
        if self.verbose:
            print(
                f"Set transform with origin {self.origin[env_id].tolist()}, min_bounds {min_bounds}, max_bounds {max_bounds}"
            )
        self._gen.set_transform(self.origin[env_id].tolist(), min_bounds, max_bounds)

    def generate(
        self,
        origin,
        boundary,
        type,
        env_ids=None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate occupancy map

        Args:
            origin: Scan center in LOCAL coordinates [x, y, z] (required)
            boundary: Scan boundary in LOCAL coordinates [x_min, x_max, y_min, y_max, z_min, z_max]
            type: Occupancy map type ("2d" or "3d")
            env_ids: Environment IDs to generate for

        Returns:
            Occupancy map array or list of arrays
        """
        env_ids_normalized, single_env = self._normalize_env_ids_with_single(env_ids)
        env_ids = env_ids_normalized

        if self._gen is None:
            raise RuntimeError("Generator not initialized. Call initialize() first.")

        if self.env_origins is None:
            raise RuntimeError("env_origins not initialized. Call initialize() first.")

        self.update_settings(
            cell_size=self.cell_size,
            values=(self.values.occupied, self.values.unoccupied, self.values.unknown),
        )

        results = []
        for env_id in env_ids:
            if origin is None:
                raise ValueError(
                    f"origin must be provided for env {env_id}. Specify scan center in LOCAL coordinates."
                )

            if boundary is None:
                raise ValueError(
                    f"boundary must be provided for env {env_id}. Cannot be None."
                )

            # Convert LOCAL coordinates to WORLD coordinates
            origin_local = self._normalize_value(origin, env_id)
            boundary_val = self._normalize_value(boundary, env_id)

            env_origin = self.env_origins[env_id].cpu().numpy()
            origin_world = env_origin + origin_local

            # Set transform with world coordinates
            self.origin[env_id] = origin_world
            self.boundary[env_id] = boundary_val
            self._update_grid_dimensions(env_id)
            self._set_transform_internal(env_id)
            self.grid[env_id] = None

            self.update_settings(env_id=env_id)

            if type == "2d":
                self._gen.generate2d()
            elif type == "3d":
                self._gen.generate3d()
            else:
                raise ValueError("type must be '2d' or '3d'")

            occupied_points = self._gen.get_occupied_positions()
            free_points = self._gen.get_free_positions()
            if self.verbose:
                print(f"Get {len(occupied_points)} occupied points")
                print(f"Get {len(free_points)} free points")
            cells = np.array([list(p) for p in occupied_points], dtype=np.float32)

            grid = np.zeros((self.grid_h[env_id], self.grid_w[env_id]), dtype=np.uint8)

            if cells.size > 0:
                # Calculate scan area center (should equal env_origin if boundary is symmetric)
                scan_center_x = self.origin[env_id][0]
                scan_center_y = self.origin[env_id][1]

                grid_center_x = self.grid_w[env_id] // 2
                grid_center_y = self.grid_h[env_id] // 2

                # Map world coordinates to grid with scan center at grid center
                xi = (
                    ((cells[:, 0] - scan_center_x) / self.cell_size) + grid_center_x
                ).astype(int)
                yi = (
                    ((cells[:, 1] - scan_center_y) / self.cell_size) + grid_center_y
                ).astype(int)

                inb = (
                    (xi >= 0)
                    & (xi < self.grid_w[env_id])
                    & (yi >= 0)
                    & (yi < self.grid_h[env_id])
                )

                grid[yi[inb], xi[inb]] = 1

            self.grid[env_id] = grid
            results.append(grid)

        # vis_grid = (1 - results[0]) * 255
        # vis_grid = vis_grid.astype(np.uint8)
        # cv2.imshow("Occupancy Map", vis_grid)
        # cv2.waitKey(-1)

        return results[0] if single_env else results

    def save_grid_npy(
        self,
        path: str,
        env_id: Optional[int] = None,
    ) -> None:
        if env_id is None:
            for i in range(self.num_envs):
                if self.grid[i] is not None:
                    np.save(f"{path}_env_{i}.npy", self.grid[i].astype(np.uint8))
        else:
            if self.grid[env_id] is None:
                raise RuntimeError(f"No grid to save for env {env_id}")
            np.save(path, self.grid[env_id].astype(np.uint8))

    def _world_to_grid_xy(
        self, xy_world: Iterable[float], env_id: int
    ) -> Tuple[int, int]:
        """Convert world XY coordinates to grid indices for a specific environment

        Args:
            xy_world: World coordinates [x, y]
            env_id: Environment ID

        Returns:
            Grid indices (gx, gy)
        """
        if self.boundary[env_id] is None:
            raise RuntimeError(f"Boundary not set for env {env_id}")

        env_origin = self.env_origins[env_id].cpu().numpy()
        x_min_abs = float(env_origin[0] + self.boundary[env_id][0])
        x_max_abs = float(env_origin[0] + self.boundary[env_id][1])
        y_min_abs = float(env_origin[1] + self.boundary[env_id][2])
        y_max_abs = float(env_origin[1] + self.boundary[env_id][3])

        # Calculate scan center
        scan_center_x = (x_min_abs + x_max_abs) / 2
        scan_center_y = (y_min_abs + y_max_abs) / 2

        grid_center_x = self.grid_w[env_id] // 2
        grid_center_y = self.grid_h[env_id] // 2

        # Map world to grid
        gx = int(((xy_world[0] - scan_center_x) / self.cell_size) + grid_center_x)
        gy = int(((xy_world[1] - scan_center_y) / self.cell_size) + grid_center_y)

        return gx, gy

    def is_escaped(self, xy_local: Iterable[float], env_id: int) -> bool:
        """Check if a local position is outside the navigable area

        Args:
            xy_local: Local coordinates [x, y] (relative to env_origin)
            env_id: Environment ID

        Returns:
            True if position is occupied (not navigable) or outside grid bounds
        """
        if self.grid[env_id] is None:
            raise RuntimeError(
                f"Grid not built for env {env_id}; call generate() first."
            )

        # Convert LOCAL to WORLD
        env_origin = self.env_origins[env_id].cpu().numpy()
        xy_world = [env_origin[0] + xy_local[0], env_origin[1] + xy_local[1]]

        gx, gy = self._world_to_grid_xy(xy_world, env_id)
        h, w = self.grid[env_id].shape

        # Outside grid bounds = escaped
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            return True

        # Occupied cell (grid == 1) = escaped (not navigable)
        # Free cell (grid == 0) = inside (navigable)
        return self.grid[env_id][gy, gx] == 1

    def is_collided(
        self, src_xy_local: Iterable[float], dst_xy_local: Iterable[float], env_id: int
    ) -> bool:
        """Check if a line between two local positions collides with obstacles

        Args:
            src_xy_local: Source local coordinates [x, y] (relative to env_origin)
            dst_xy_local: Destination local coordinates [x, y] (relative to env_origin)
            env_id: Environment ID

        Returns:
            True if line collides with obstacles
        """
        if self.grid[env_id] is None:
            raise RuntimeError(
                f"Grid not built for env {env_id}; call generate() first."
            )

        # Convert LOCAL to WORLD
        env_origin = self.env_origins[env_id].cpu().numpy()
        src_xy_world = [
            env_origin[0] + src_xy_local[0],
            env_origin[1] + src_xy_local[1],
        ]
        dst_xy_world = [
            env_origin[0] + dst_xy_local[0],
            env_origin[1] + dst_xy_local[1],
        ]

        x1, y1 = self._world_to_grid_xy(src_xy_world, env_id)
        x2, y2 = self._world_to_grid_xy(dst_xy_world, env_id)
        rr, cc = sk_line(y1, x1, y2, x2)
        h, w = self.grid[env_id].shape
        mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[mask], cc[mask]
        if rr.size == 0:
            return True
        return np.any(self.grid[env_id][rr, cc] == 1)
