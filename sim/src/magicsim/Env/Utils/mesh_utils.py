"""Utilities for computing world-space bounding box from USD prims."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def get_world_bbox_half_extents(prim) -> Optional[Tuple[float, float, float]]:
    """
    根据物体的世界包围盒计算半轴长 (half-extents)。

    使用 USD UsdGeom.Imageable.ComputeWorldBound 得到世界坐标系下的 AABB，
    返回 (half_x, half_y, half_z)，单位与场景一致（米）。

    Args:
        prim: USD Prim（需为 UsdGeom.Imageable，如 RigidObject.prim / GeometryObject.prim）

    Returns:
        世界 AABB 的半轴长 (half_x, half_y, half_z)，失败时返回 None。
    """
    try:
        from pxr import Usd, UsdGeom

        imageable = UsdGeom.Imageable(prim)
        time = Usd.TimeCode.Default()
        bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
        bound_range = bound.ComputeAlignedBox()
        min_pt = bound_range.GetMin()
        max_pt = bound_range.GetMax()
        half = (
            (max_pt[0] - min_pt[0]) * 0.5,
            (max_pt[1] - min_pt[1]) * 0.5,
            (max_pt[2] - min_pt[2]) * 0.5,
        )
        return half
    except Exception:
        return None


def get_local_bbox_half_extents(prim) -> Optional[Tuple[float, float, float]]:
    """
    Prim 自身局部坐标系下的轴对齐包围盒半轴长 (half_x, half_y, half_z)。

    使用 ``UsdGeom.Imageable.ComputeUntransformedBound``，与物体在世界中的朝向无关，
    因此刚体绕任意轴旋转时半轴长不变（随网格与 prim 尺度固定）。

    Args:
        prim: USD Prim（需为 UsdGeom.Imageable）

    Returns:
        局部 AABB 半轴长，失败时返回 None。
    """
    try:
        from pxr import Usd, UsdGeom

        imageable = UsdGeom.Imageable(prim)
        time = Usd.TimeCode.Default()
        bound = imageable.ComputeUntransformedBound(time, UsdGeom.Tokens.default_)
        bound_range = bound.ComputeAlignedBox()
        if bound_range.IsEmpty():
            return None
        min_pt = bound_range.GetMin()
        max_pt = bound_range.GetMax()
        half = (
            (max_pt[0] - min_pt[0]) * 0.5,
            (max_pt[1] - min_pt[1]) * 0.5,
            (max_pt[2] - min_pt[2]) * 0.5,
        )
        return half
    except Exception:
        return None


def get_local_bbox_min_max(prim) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Prim 局部坐标系下的轴对齐包围盒 [min, max]（与 ``get_local_bbox_half_extents`` 同源）。

    Returns:
        (min_xyz, max_xyz) 各为 shape (3,) 的 float64，失败时 None。
    """
    try:
        from pxr import Usd, UsdGeom

        imageable = UsdGeom.Imageable(prim)
        time = Usd.TimeCode.Default()
        bound = imageable.ComputeUntransformedBound(time, UsdGeom.Tokens.default_)
        bound_range = bound.ComputeAlignedBox()
        if bound_range.IsEmpty():
            return None
        mn = bound_range.GetMin()
        mx = bound_range.GetMax()
        return (
            np.array([mn[0], mn[1], mn[2]], dtype=np.float64),
            np.array([mx[0], mx[1], mx[2]], dtype=np.float64),
        )
    except Exception:
        return None


def ray_aabb_entry_face_center_outward(
    origin: np.ndarray,
    direction: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    eps: float = 1e-7,
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """
    射线 ``origin + t * direction``（``t >= 0``）从**盒外**首次进入轴对齐盒时：

    - ``t_enter = max(t_near)``（slab 法）
    - 进入面由 ``argmax(t_near)`` 确定：``d[k] > 0`` 为 ``x_k = min`` 面，否则为 ``max`` 面
    - 返回该矩形面的几何中心 ``face_center``、以及**朝盒外且指向射线起点**的单位法向 ``n_out``

    坐标系与 ``box_min`` / ``box_max`` 一致（物体局部系）。起点在盒内时返回 None，由调用方回退。

    Returns:
        ``(t_enter, face_center, n_out)``；不相交或退化时返回 None。
    """
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = np.linalg.norm(d)
    if dn < eps:
        return None
    d = d / dn
    bmin = np.asarray(box_min, dtype=np.float64).reshape(3)
    bmax = np.asarray(box_max, dtype=np.float64).reshape(3)

    t1 = np.zeros(3)
    t2 = np.zeros(3)
    for i in range(3):
        if abs(d[i]) < eps:
            if o[i] < bmin[i] - eps or o[i] > bmax[i] + eps:
                return None
            t1[i] = -np.inf
            t2[i] = np.inf
        else:
            ta = (bmin[i] - o[i]) / d[i]
            tb = (bmax[i] - o[i]) / d[i]
            t1[i] = min(ta, tb)
            t2[i] = max(ta, tb)

    t_enter = float(np.max(t1))
    t_exit = float(np.min(t2))
    if t_enter > t_exit + eps:
        return None
    # 仅处理从外部进入：t_enter 为沿射线首次碰到盒子的正参数
    if t_enter < eps:
        return None

    k = int(np.argmax(t1))
    if d[k] > 0:
        face_is_max = False
    else:
        face_is_max = True

    fc = np.array(
        [
            0.5 * (bmin[0] + bmax[0]),
            0.5 * (bmin[1] + bmax[1]),
            0.5 * (bmin[2] + bmax[2]),
        ],
        dtype=np.float64,
    )
    if face_is_max:
        fc[k] = bmax[k]
        n = np.zeros(3, dtype=np.float64)
        n[k] = 1.0
    else:
        fc[k] = bmin[k]
        n = np.zeros(3, dtype=np.float64)
        n[k] = -1.0

    if np.dot(n, o - fc) < 0:
        n = -n

    return t_enter, fc, n


def compute_reach_offset_from_bbox(
    half_extents: Tuple[float, float, float],
    push_direction: np.ndarray,
    margin: float,
) -> np.ndarray:
    """
    根据物体 bbox 半轴长和推动方向，计算 reach 目标相对物体中心的偏移，
    使接近点位于物体 bbox 外侧，避免碰撞。

    约定：reach 目标 = 物体中心 + offset；offset 沿 push_direction 反方向，
    距离 = bbox 在该方向的“半径”+ margin。

    对世界系 AABB，沿单位方向 d 的“半径”为 dot(half_extents, |d|)。

    Args:
        half_extents: (half_x, half_y, half_z) 世界 AABB 半轴长
        push_direction: (3,) 单位向量，推动方向（世界系）
        margin: 额外安全距离（米）

    Returns:
        offset: (3,) 世界系偏移向量
    """
    half = np.array(half_extents, dtype=np.float64)
    d = np.asarray(push_direction, dtype=np.float64).reshape(3)
    d = d / (np.linalg.norm(d) + 1e-9)
    radius_along_d = float(np.dot(half, np.abs(d)))
    offset = -d * (radius_along_d + margin)
    return offset
