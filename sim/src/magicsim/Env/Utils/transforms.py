import math

import torch
import numpy as np
from pxr import Gf
from scipy.spatial.transform import Rotation as R

# internal global constants
_POLE_LIMIT = 1.0 - 1e-6


def quat_inverse(q):
    """
    Inverse of unit quaternion (wxyz)
    """
    w, x, y, z = q
    return torch.tensor([w, -x, -y, -z], dtype=q.dtype)


def matrix_to_euler_angles(
    mat: np.ndarray, degrees: bool = False, extrinsic: bool = True
) -> np.ndarray:
    """Convert rotation matrix to Euler XYZ extrinsic or intrinsic angles.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.
        degrees (bool, optional): Whether returned angles should be in degrees.
        extrinsic (bool, optional): True if the rotation matrix follows the extrinsic matrix
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic matrix conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    if extrinsic:
        if mat[2, 0] > _POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[2, 0] < -_POLE_LIMIT:
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = -np.arcsin(mat[2, 0])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])
    else:
        if mat[0, 2] > _POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])

        if mat[0, 2] < -_POLE_LIMIT:
            roll = np.arctan2(mat[1, 0], mat[1, 1])
            pitch = -np.pi / 2
            yaw = 0.0
            return np.array([roll, pitch, yaw])
        roll = -math.atan2(mat[1, 2], mat[2, 2])
        pitch = math.asin(mat[0, 2])
        yaw = -math.atan2(mat[0, 1], mat[0, 0])

        if degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
        return np.array([roll, pitch, yaw])


def euler_to_rot_matrix(
    euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True
) -> np.ndarray:
    """Convert Euler XYZ or ZYX angles to rotation matrix.

    Args:
        euler_angles (np.ndarray): Euler angles.
        degrees (bool, optional): Whether passed angles are in degrees.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray:  A 3x3 rotation matrix in its extrinsic or intrinsic form depends on the extrinsic argument.
    """
    if extrinsic:
        yaw, pitch, roll = euler_angles
    else:
        roll, pitch, yaw = euler_angles
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    if extrinsic:
        return np.array(
            [
                [(cp * cr), ((cr * sp * sy) - (cy * sr)), ((cr * cy * sp) + (sr * sy))],
                [(cp * sr), ((cy * cr) + (sr * sp * sy)), ((cy * sp * sr) - (cr * sy))],
                [-sp, (cp * sy), (cy * cp)],
            ]
        )
    else:
        return np.array(
            [
                [(cp * cy), (-cp * sy), sp],
                [
                    ((cy * sr * sp) + (cr * sy)),
                    ((cr * cy) - (sr * sp * sy)),
                    (-cp * sr),
                ],
                [
                    ((-cr * cy * sp) + (sr * sy)),
                    ((cy * sr) + (cr * sp * sy)),
                    (cr * cp),
                ],
            ]
        )


def rot_matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Quaternion.

    Args:
        mat (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    if mat.shape == (3, 3):
        tmp = np.eye(4)
        tmp[0:3, 0:3] = mat
        mat = tmp

    q = np.empty((4,), dtype=np.float64)
    t = np.trace(mat)
    if t > mat[3, 3]:
        q[0] = t
        q[3] = mat[1, 0] - mat[0, 1]
        q[2] = mat[0, 2] - mat[2, 0]
        q[1] = mat[2, 1] - mat[1, 2]
    else:
        i, j, k = 0, 1, 2
        if mat[1, 1] > mat[0, 0]:
            i, j, k = 1, 2, 0
        if mat[2, 2] > mat[i, i]:
            i, j, k = 2, 0, 1
        t = mat[i, i] - (mat[j, j] + mat[k, k]) + mat[3, 3]
        q[i + 1] = t
        q[j + 1] = mat[i, j] + mat[j, i]
        q[k + 1] = mat[k, i] + mat[i, k]
        q[0] = mat[k, j] - mat[j, k]
    q *= 0.5 / np.sqrt(t * mat[3, 3])
    return q


def quat_to_euler_angles(
    quat: np.ndarray, degrees: bool = False, extrinsic: bool = True
) -> np.ndarray:
    """Convert input quaternion to Euler XYZ or ZYX angles.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).
        degrees (bool, optional): Whether returned angles should be in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.


    Returns:
        np.ndarray: Euler XYZ angles (intrinsic form) if extrinsic is False and Euler XYZ angles (extrinsic form) if extrinsic is True.
    """
    return matrix_to_euler_angles(
        quat_to_rot_matrix(quat), degrees=degrees, extrinsic=extrinsic
    )


def euler_angles_to_quat(
    euler_angles: np.ndarray, degrees: bool = False, extrinsic: bool = True
) -> np.ndarray:
    """Convert Euler angles to quaternion.

    Args:
        euler_angles (np.ndarray):  Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    mat = np.array(
        euler_to_rot_matrix(euler_angles, degrees=degrees, extrinsic=extrinsic)
    )
    return rot_matrix_to_quat(mat)


def rotate_point_cloud(point_cloud, roll, pitch, yaw):
    """
    Rotate a point cloud using Euler angles (roll, pitch, yaw).

    Parameters:
    - point_cloud: A numpy array of shape (N, 3) representing the point cloud.
    - roll: Roll angle in radians.
    - pitch: Pitch angle in radians.
    - yaw: Yaw angle in radians.

    Returns:
    - rotated_point_cloud: The rotated point cloud as a numpy array.
    """

    # Convert Euler angles to rotation matrix
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    R = np.dot(Rz, np.dot(Ry, Rx))

    # Apply rotation to each point in the point cloud
    rotated_point_cloud = np.dot(point_cloud, R.T)

    return rotated_point_cloud


def get_pose_world(trans_rel, rot_rel, robot_pos, robot_rot):
    if rot_rel is not None:
        rot = robot_rot @ rot_rel
    else:
        rot = None

    if trans_rel is not None:
        trans = robot_rot @ trans_rel + robot_pos
    else:
        trans = None

    return trans, rot


def get_pose_relat(trans, rot, robot_pos, robot_rot):
    inv_rob_rot = robot_rot.T

    if trans is not None:
        trans_rel = inv_rob_rot @ (trans - robot_pos)
    else:
        trans_rel = None

    if rot is not None:
        rot_rel = inv_rob_rot @ rot
    else:
        rot_rel = None

    return trans_rel, rot_rel


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([w, x, y, z], axis=-1).reshape(shape)

    return quat


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return np.concatenate((a[:, 0:1], -a[:, 1:]), axis=-1).reshape(shape)


def quat_diff_rad(a, b):
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    return 2.0 * np.arcsin(np.clip(np.linalg.norm(mul[:, 1:], axis=-1), 0, 1))


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert input quaternion to rotation matrix.

    Args:
        quat (np.ndarray): Input quaternion (w, x, y, z).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # might need to be normalized
    rotm = Gf.Matrix3f(Gf.Quatf(*quat.tolist())).GetTranspose()
    return np.array(rotm)


_FLOAT_EPS = np.finfo(np.float32).eps
_EPS4 = _FLOAT_EPS * 4.0


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    return euler_angles_to_quat(matrix_to_euler_angles(mat))


def world_to_robot_frame(eef_pose, robot_base_pose):
    """
    Converts the end-effector orientation from the world frame to the robot base frame.

    Parameters:
        eef_ori (array-like): 4-element list/tuple/np.ndarray representing a quaternion in [w, x, y, z] format.
                              This is the orientation of the end-effector in the world coordinate system.
        robot_base_ori (array-like): 4-element list/tuple/np.ndarray representing a quaternion in [w, x, y, z] format.
                                     This is the orientation of the robot base in the world coordinate system.

    Returns:
        np.ndarray: The orientation of the end-effector in the robot base coordinate system, in [x, y, z, w] format.
    """
    assert len(eef_pose) == 7, (
        "End-effector pose must be a list or tuple of length 7 (position + orientation)"
    )
    assert len(robot_base_pose) == 7, (
        "Robot base pose must be a list or tuple of length 7 (position + orientation)"
    )

    eef_pos = eef_pose[:3]  # Extract position from end-effector pose
    eef_ori = eef_pose[3:]  # Extract orientation from end-effector pose
    robot_base_pos = robot_base_pose[:3]  # Extract position from robot base pose
    robot_base_ori = robot_base_pose[3:]  # Extract orientation from robot base pose

    # Convert input quaternions to scipy Rotation objects
    eef_rot = R.from_quat(eef_ori)  # End-effector rotation in world frame
    base_rot = R.from_quat(robot_base_ori)  # Robot base rotation in world frame

    # Compute the inverse of the robot base rotation (transforms from world to robot frame)
    base_inv_rot = base_rot.inv()

    # Apply the transformation: rotate EEF orientation into robot base frame
    eef_in_robot = base_inv_rot * eef_rot

    eef_in_robot_pos = eef_pos - np.array(
        robot_base_pos
    )  # Translate EEF position into robot base frame

    # Convert the result back to quaternion and reorder to [w, x, y, z]
    quat_xyzw = eef_in_robot.as_quat()  # [x, y, z, w]
    quat_wxyz = np.roll(quat_xyzw, 1)  # [w, x, y, z]

    # Return the resulting orientation as a quaternion in [x, y, z, w] format
    return np.concatenate((eef_in_robot_pos, quat_wxyz))
