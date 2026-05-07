from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from src.types_config import CollisionSphere, ResolvedHandRuntimeConfig


@dataclass
class HandPose:
    wrist_position: np.ndarray
    wrist_quaternion_xyzw: Optional[np.ndarray] = None
    wrist_rotation: Optional[np.ndarray] = None
    joint_positions: Dict[str, float] = field(default_factory=dict)

    def rotation_matrix(self) -> np.ndarray:
        if self.wrist_rotation is not None:
            return np.asarray(self.wrist_rotation, dtype=np.float64)
        quat = np.asarray(self.wrist_quaternion_xyzw, dtype=np.float64)
        return Rotation.from_quat(quat).as_matrix()


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclass
class CollisionSphereState:
    link_name: str
    sphere_index: int
    radius: float
    center_world: np.ndarray


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _rotation_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def _make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ hom.T).T
    return out[:, :3]


class HandKinematicsModel:
    """
    Lightweight URDF FK model for floating dexterous hands.

    The palm/root pose is provided externally through HandPose; URDF joints are
    only used for the articulated finger chain below that root.
    """

    def __init__(
        self,
        *,
        root_link: str,
        joints_by_name: Dict[str, JointSpec],
        child_joints_by_parent: Dict[str, List[str]],
    ) -> None:
        self.root_link = root_link
        self.joints_by_name = joints_by_name
        self.child_joints_by_parent = child_joints_by_parent

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path,
        root_link: str,
    ) -> "HandKinematicsModel":
        tree = ET.parse(str(urdf_path))
        robot = tree.getroot()

        joints_by_name: Dict[str, JointSpec] = {}
        child_joints_by_parent: Dict[str, List[str]] = {}

        for joint_elem in robot.findall("joint"):
            joint_type = joint_elem.attrib.get("type", "fixed")
            name = joint_elem.attrib["name"]

            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                continue

            parent_link = parent_elem.attrib["link"]
            child_link = child_elem.attrib["link"]

            origin_elem = joint_elem.find("origin")
            xyz = np.zeros(3, dtype=np.float64)
            rpy = np.zeros(3, dtype=np.float64)
            if origin_elem is not None:
                if "xyz" in origin_elem.attrib:
                    xyz = np.asarray(
                        [float(x) for x in origin_elem.attrib["xyz"].split()],
                        dtype=np.float64,
                    )
                if "rpy" in origin_elem.attrib:
                    rpy = np.asarray(
                        [float(x) for x in origin_elem.attrib["rpy"].split()],
                        dtype=np.float64,
                    )

            axis_elem = joint_elem.find("axis")
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if axis_elem is not None and "xyz" in axis_elem.attrib:
                axis = np.asarray(
                    [float(x) for x in axis_elem.attrib["xyz"].split()],
                    dtype=np.float64,
                )
            axis = _safe_normalize(axis)

            joint = JointSpec(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
            )
            joints_by_name[name] = joint
            child_joints_by_parent.setdefault(parent_link, []).append(name)

        return cls(
            root_link=root_link,
            joints_by_name=joints_by_name,
            child_joints_by_parent=child_joints_by_parent,
        )

    def _joint_transform(
        self,
        joint: JointSpec,
        joint_positions: Mapping[str, float],
    ) -> np.ndarray:
        T_origin = _make_transform(
            _rotation_from_rpy(joint.origin_rpy),
            joint.origin_xyz,
        )
        if joint.joint_type in ("fixed", "floating"):
            return T_origin
        if joint.joint_type in ("revolute", "continuous"):
            angle = float(joint_positions.get(joint.name, 0.0))
            R_axis = Rotation.from_rotvec(joint.axis * angle).as_matrix()
            return T_origin @ _make_transform(R_axis, np.zeros(3, dtype=np.float64))
        raise ValueError(f"Unsupported joint type: {joint.joint_type} for {joint.name}")

    def forward_link_transforms(
        self,
        hand_pose: HandPose,
    ) -> Dict[str, np.ndarray]:
        root_T = _make_transform(
            hand_pose.rotation_matrix(),
            hand_pose.wrist_position,
        )
        result: Dict[str, np.ndarray] = {self.root_link: root_T}

        stack: List[str] = [self.root_link]
        while stack:
            parent_link = stack.pop()
            parent_T = result[parent_link]
            for joint_name in self.child_joints_by_parent.get(parent_link, []):
                joint = self.joints_by_name[joint_name]
                child_T = parent_T @ self._joint_transform(joint, hand_pose.joint_positions)
                result[joint.child_link] = child_T
                stack.append(joint.child_link)

        return result

    def collision_spheres_world(
        self,
        runtime_cfg: ResolvedHandRuntimeConfig,
        hand_pose: HandPose,
    ) -> List[CollisionSphereState]:
        link_transforms = self.forward_link_transforms(hand_pose)
        spheres_world: List[CollisionSphereState] = []
        for link_name, sphere_list in runtime_cfg.collision.spheres_by_link.items():
            link_T = link_transforms.get(link_name)
            if link_T is None:
                continue
            for sphere_index, sphere in enumerate(sphere_list):
                center_world = _transform_points(link_T, np.asarray(sphere.center, dtype=np.float64))[0]
                spheres_world.append(
                    CollisionSphereState(
                        link_name=link_name,
                        sphere_index=sphere_index,
                        radius=float(sphere.radius),
                        center_world=center_world,
                    )
                )
        return spheres_world

    def collision_sphere_map(
        self,
        runtime_cfg: ResolvedHandRuntimeConfig,
        hand_pose: HandPose,
    ) -> Dict[tuple[str, int], CollisionSphereState]:
        spheres = self.collision_spheres_world(runtime_cfg, hand_pose)
        return {(s.link_name, s.sphere_index): s for s in spheres}

    def semantic_points_world(
        self,
        runtime_cfg: ResolvedHandRuntimeConfig,
        hand_pose: HandPose,
        point_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, CollisionSphereState]:
        sphere_map = self.collision_sphere_map(runtime_cfg, hand_pose)
        names = point_names if point_names is not None else runtime_cfg.semantic_points.keys()
        result: Dict[str, CollisionSphereState] = {}
        for point_name in names:
            semantic_point = runtime_cfg.semantic_points.get(point_name)
            if semantic_point is None:
                continue
            key = (semantic_point.source_link, semantic_point.source_sphere_index)
            sphere_state = sphere_map.get(key)
            if sphere_state is not None:
                result[point_name] = sphere_state
        return result


def make_hand_pose(
    wrist_position: Sequence[float],
    wrist_quaternion_xyzw: Optional[Sequence[float]],
    wrist_rotation: Optional[np.ndarray],
    joint_positions: Mapping[str, float],
) -> HandPose:
    quat_arr = None
    rot_arr = None
    if wrist_quaternion_xyzw is not None:
        quat_arr = np.asarray(wrist_quaternion_xyzw, dtype=np.float64)
    if wrist_rotation is not None:
        rot_arr = np.asarray(wrist_rotation, dtype=np.float64)
    return HandPose(
        wrist_position=np.asarray(wrist_position, dtype=np.float64),
        wrist_quaternion_xyzw=quat_arr,
        wrist_rotation=rot_arr,
        joint_positions={k: float(v) for k, v in joint_positions.items()},
    )


def load_hand_kinematics_model(
    runtime_cfg: ResolvedHandRuntimeConfig,
) -> HandKinematicsModel:
    urdf_path = runtime_cfg.hand.asset.urdf_path
    if urdf_path is None:
        raise ValueError("Hand asset URDF path is required to build the kinematics model.")
    return HandKinematicsModel.from_urdf(
        urdf_path=urdf_path,
        root_link=runtime_cfg.hand.root.wrist_link,
    )
