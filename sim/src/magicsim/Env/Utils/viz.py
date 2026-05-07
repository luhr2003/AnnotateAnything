from pxr import Gf
from isaacsim.util.debug_draw import _debug_draw

import torch


def _normalize_grasp_poses(
    grasp_poses,
) -> list[tuple["Gf.Vec3d", "Gf.Quatd"]]:
    """Normalize grasp_poses from various formats to list[tuple[Gf.Vec3d, Gf.Quatd]].

    Supported formats:
        - list[tuple[Gf.Vec3d, Gf.Quatd]]: original format, returned as-is.
        - torch.Tensor of shape (N, 7): each row is [x, y, z, qw, qx, qy, qz].
        - list[torch.Tensor]: each element is a 7-dim tensor [x, y, z, qw, qx, qy, qz].
    """
    if isinstance(grasp_poses, torch.Tensor):
        if grasp_poses.numel() == 0:
            return []
    elif not grasp_poses:
        return []

    # --- torch.Tensor (N, 7) ---
    if isinstance(grasp_poses, torch.Tensor):
        assert grasp_poses.ndim == 2 and grasp_poses.shape[1] == 7, (
            f"Expected tensor of shape (N, 7), got {grasp_poses.shape}"
        )
        poses = grasp_poses.detach().cpu().tolist()
        return [
            (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6]))
            for p in poses
        ]

    # --- list ---
    assert isinstance(grasp_poses, (list, tuple)), (
        f"grasp_poses must be a list or torch.Tensor, got {type(grasp_poses)}"
    )

    first = grasp_poses[0]

    # list[torch.Tensor], each tensor is (7,)
    if isinstance(first, torch.Tensor):
        for i, t in enumerate(grasp_poses):
            assert isinstance(t, torch.Tensor) and t.shape == (7,), (
                f"Element {i} must be a 7-dim tensor, got shape {t.shape}"
            )
        poses = [t.detach().cpu().tolist() for t in grasp_poses]
        return [
            (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6]))
            for p in poses
        ]

    # list[tuple[Gf.Vec3d, Gf.Quatd]] – original format, pass through
    return grasp_poses


def draw_grasp_samples_as_axes(
    grasp_poses: "list[tuple[Gf.Vec3d, Gf.Quatd]] | list[torch.Tensor] | torch.Tensor",
    axis_length: float = 0.03,
    line_thickness: float = 2,
    line_opacity: float = 0.5,
    clear_existing: bool = True,
):
    """Draw grasp samples as oriented frames (axes) in the viewport.

    Args:
        grasp_poses: Grasp poses in one of the following formats:
            - list[tuple[Gf.Vec3d, Gf.Quatd]]: position and orientation tuples.
            - list[torch.Tensor]: list of 7-dim tensors [x, y, z, qw, qx, qy, qz].
            - torch.Tensor: tensor of shape (N, 7) with [x, y, z, qw, qx, qy, qz].
        axis_length: Length of each axis line.
        line_thickness: Thickness of the axis lines.
        line_opacity: Opacity of the axis lines.
        clear_existing: Whether to clear existing samples before drawing.
    """
    grasp_poses = _normalize_grasp_poses(grasp_poses)
    draw_iface = _debug_draw.acquire_debug_draw_interface()
    if clear_existing:
        draw_iface.clear_lines()
    # Axis colors: X=Red, Y=Green, Z=Blue
    x_color = [1.0, 0.0, 0.0, line_opacity]
    y_color = [0.0, 1.0, 0.0, line_opacity]
    z_color = [0.0, 0.0, 1.0, line_opacity]
    start_points = []
    end_points = []
    colors = []
    thicknesses = []
    for location, quat in grasp_poses:
        origin = [location[0], location[1], location[2]]
        # X axis
        x_axis = quat.Transform(Gf.Vec3d(1, 0, 0)) * axis_length
        x_end = [origin[0] + x_axis[0], origin[1] + x_axis[1], origin[2] + x_axis[2]]
        start_points.append(origin)
        end_points.append(x_end)
        colors.append(x_color)
        thicknesses.append(line_thickness)
        # Y axis
        y_axis = quat.Transform(Gf.Vec3d(0, 1, 0)) * axis_length
        y_end = [origin[0] + y_axis[0], origin[1] + y_axis[1], origin[2] + y_axis[2]]
        start_points.append(origin)
        end_points.append(y_end)
        colors.append(y_color)
        thicknesses.append(line_thickness)
        # Z axis
        z_axis = quat.Transform(Gf.Vec3d(0, 0, 1)) * axis_length
        z_end = [origin[0] + z_axis[0], origin[1] + z_axis[1], origin[2] + z_axis[2]]
        start_points.append(origin)
        end_points.append(z_end)
        colors.append(z_color)
        thicknesses.append(line_thickness)
    draw_iface.draw_lines(start_points, end_points, colors, thicknesses)


def draw_waypoints(
    points: "list[list[float]] | list[tuple[float, float, float]]",
    point_size: float = 6.0,
    color: tuple[float, float, float, float] = (1.0, 0.8, 0.1, 0.6),
    clear_existing: bool = True,
    offset: "tuple[float, float, float] | list[float] | None" = None,
):
    """Draw waypoint positions as points in the viewport."""
    draw_iface = _debug_draw.acquire_debug_draw_interface()
    if clear_existing:
        draw_iface.clear_points()
    if not points:
        return
    if offset is not None:
        points = [
            [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]] for p in points
        ]
    colors = [list(color)] * len(points)
    sizes = [point_size] * len(points)
    draw_iface.draw_points(points, colors, sizes)
