# navigation_manager.py
# Scene-centric NavigationManager for multi-agent navigation without character animation system

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING

import carb
import numpy as np

if TYPE_CHECKING:

    class Float3:
        x: float
        y: float
        z: float
else:
    from carb import Float3 as Float3

Vec3 = Tuple[float, float, float]
PointLike = Union[Float3, Vec3, np.ndarray]


def _f3(p: PointLike) -> Float3:
    """Convert any point-like (tuple, list, np.ndarray) to Float3"""
    if isinstance(p, Float3):
        return p
    if isinstance(p, np.ndarray):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
    else:
        x, y, z = p
    return Float3(float(x), float(y), float(z))


def _np_array(p: Float3) -> np.ndarray:
    """Convert Float3 to numpy array"""
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def _sub(a: Float3, b: Float3) -> Float3:
    return Float3(a.x - b.x, a.y - b.y, a.z - b.z)


def _len3(v: Float3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _dist3(a: Float3, b: Float3) -> float:
    return _len3(_sub(a, b))


class NavMeshManager:
    def __init__(
        self,
        navmesh,
    ):
        """
        Initialize NavigationManager.

        Args:
            usd_path: Kept for API compatibility; ignored (stage must already be loaded)
            navmesh_enabled: Whether to use navmesh for pathfinding
            dynamic_avoidance_enabled: Whether to enable dynamic obstacle avoidance
            agent_name: Unique identifier for this agent
        """
        self.navmesh = navmesh

    def _validate_navmesh_point_2d(self, p: Float3) -> bool:
        """Check if point is on navmesh (2D XY check)."""

        if hasattr(self.navmesh, "validate_point_2d"):
            return bool(self.navmesh.validate_point_2d([p.x, p.y, 0.0]))
        if hasattr(self.navmesh, "is_point_on_navmesh"):
            return bool(self.navmesh.is_point_on_navmesh([p.x, p.y, 0.0]))
        print("Navmesh does not support 2D point validation.")
        return True

    def generate_path(
        self,
        start_point: PointLike,
        coords: Sequence[PointLike],
        return_path_object: bool = False,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], object]]:
        """
        Generate path through waypoints using navmesh.

        Args:
            start_point: Starting position as numpy array, tuple, or Float3
            coords: List of waypoints as numpy arrays, tuples, or Float3
            return_path_object: If True, also return the last NavMesh path object

        Returns:
            If return_path_object=False: List of path points as numpy arrays [x, y, z]
            If return_path_object=True: Tuple of (path_points, navmesh_path_object)
        """

        # Convert inputs to Float3 for navmesh API
        pts = [_f3(p) for p in coords]
        pts.insert(0, _f3(start_point))
        prev_point = pts[0]
        path_float3: List[Float3] = []
        last_path_object = None

        for point in pts[1:]:
            try:
                generated_path = self.navmesh.query_shortest_path(prev_point, point)
                last_path_object = generated_path  # Save the last path object
            except Exception as e:
                carb.log_error(
                    f"[NavMeshManager] query_shortest_path failed {prev_point} -> {point}: {e}"
                )
                generated_path = None

            if generated_path is None:
                carb.log_error(
                    f"There is no valid path between point position: {prev_point} and position: {point}"
                )
                if return_path_object:
                    return None, None
                return None

            try:
                seg_points = list(generated_path.get_points())
            except Exception:
                seg_points = []
                try:
                    for q in generated_path:
                        if isinstance(q, Float3):
                            seg_points.append(q)
                        else:
                            seg_points.append(
                                Float3(float(q[0]), float(q[1]), float(q[2]))
                            )
                except Exception:
                    seg_points = []

            if not path_float3:
                path_float3.extend(seg_points)
            else:
                if seg_points and _dist3(path_float3[-1], seg_points[0]) < 1e-6:
                    path_float3.extend(seg_points[1:])
                else:
                    path_float3.extend(seg_points)

            prev_point = point

        # Convert Float3 path to numpy arrays
        path_np = [_np_array(p) for p in path_float3]

        if return_path_object:
            return path_np, last_path_object
        return path_np
