# python
import math
import typing
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

# omniverse
from pxr import Gf

# internal global constants
_POLE_LIMIT = 1.0 - 1e-6


def rot_matrices_to_quats(rotation_matrices: torch.Tensor, device=None) -> torch.Tensor:
    """Vectorized version of converting rotation matrices to quaternions

    Args:
        rotation_matrices (torch.Tensor): N Rotation matrices with shape (N, 3, 3) or (3, 3)

    Returns:
        torch.Tensor: quaternion representation of the rotation matrices (N, 4) or (4,) - scalar first
    """
    rot = Rotation.from_matrix(rotation_matrices.cpu().numpy())
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    result = torch.from_numpy(np.asarray(result, dtype=np.float32)).float().to(device)
    return result


def quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert input quaternion to rotation matrix.

    Args:
        quat (torch.Tensor): (..., 4) with (w, x, y, z)

    Returns:
        torch.Tensor: (..., 3, 3)
    """

    if not isinstance(quat, torch.Tensor):
        quat = torch.tensor(quat, dtype=torch.float32)
    q = quat.clone()
    nq = torch.sum(q * q, dim=-1)
    small = nq < 1e-10
    scale = torch.sqrt(2.0 / torch.clamp(nq, min=1e-10))
    q = q * scale.unsqueeze(-1)

    # Outer product: (..., 4, 4)
    q_outer = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (..., 4, 4)

    R = torch.stack(
        [
            torch.stack(
                [
                    1.0 - q_outer[..., 2, 2] - q_outer[..., 3, 3],
                    q_outer[..., 1, 2] - q_outer[..., 3, 0],
                    q_outer[..., 1, 3] + q_outer[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    q_outer[..., 1, 2] + q_outer[..., 3, 0],
                    1.0 - q_outer[..., 1, 1] - q_outer[..., 3, 3],
                    q_outer[..., 2, 3] - q_outer[..., 1, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    q_outer[..., 1, 3] - q_outer[..., 2, 0],
                    q_outer[..., 2, 3] + q_outer[..., 1, 0],
                    1.0 - q_outer[..., 1, 1] - q_outer[..., 2, 2],
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    # Handle near-zero norm case
    R = torch.where(
        small.unsqueeze(-1).unsqueeze(-1),
        torch.eye(3, dtype=q.dtype, device=q.device),
        R,
    )
    return R


def matrix_to_euler_angles(
    mat: torch.Tensor, degrees: bool = False, extrinsic: bool = True
) -> torch.Tensor:
    """
    Convert rotation matrix to Euler XYZ angles.
    """

    if not isinstance(mat, torch.Tensor):
        mat = torch.tensor(mat, dtype=torch.float32)

    if extrinsic:
        # Check for gimbal lock
        pole_mask_p = mat[..., 2, 0] > _POLE_LIMIT
        pole_mask_n = mat[..., 2, 0] < -_POLE_LIMIT

        roll = torch.atan2(mat[..., 2, 1], mat[..., 2, 2])
        pitch = -torch.asin(torch.clamp(mat[..., 2, 0], -1.0, 1.0))
        yaw = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])

        # Handle poles
        roll_p = torch.atan2(mat[..., 0, 1], mat[..., 0, 2])
        roll_n = roll_p.clone()
        pitch_p = -torch.pi / 2
        pitch_n = torch.pi / 2
        yaw_p = torch.zeros_like(yaw)
        yaw_n = yaw_p.clone()

        roll = torch.where(
            pole_mask_p | pole_mask_n, torch.where(pole_mask_p, roll_p, roll_n), roll
        )
        pitch = torch.where(
            pole_mask_p, pitch_p, torch.where(pole_mask_n, pitch_n, pitch)
        )
        yaw = torch.where(pole_mask_p | pole_mask_n, yaw_p, yaw)
    else:
        pole_mask_p = mat[..., 0, 2] > _POLE_LIMIT
        pole_mask_n = mat[..., 0, 2] < -_POLE_LIMIT

        roll = -torch.atan2(mat[..., 1, 2], mat[..., 2, 2])
        pitch = torch.asin(torch.clamp(mat[..., 0, 2], -1.0, 1.0))
        yaw = -torch.atan2(mat[..., 0, 1], mat[..., 0, 0])

        roll_p = torch.atan2(mat[..., 1, 0], mat[..., 1, 1])
        roll_n = roll_p.clone()
        pitch_p = torch.pi / 2
        pitch_n = -torch.pi / 2
        yaw_p = torch.zeros_like(yaw)
        yaw_n = yaw_p.clone()

        roll = torch.where(
            pole_mask_p | pole_mask_n, torch.where(pole_mask_p, roll_p, roll_n), roll
        )
        pitch = torch.where(
            pole_mask_p, pitch_p, torch.where(pole_mask_n, pitch_n, pitch)
        )
        yaw = torch.where(pole_mask_p | pole_mask_n, yaw_p, yaw)

    angles = torch.stack([roll, pitch, yaw], dim=-1)

    if degrees:
        angles = angles * (180.0 / torch.pi)

    return angles


def euler_to_rot_matrix(
    euler_angles: torch.Tensor, degrees: bool = False, extrinsic: bool = True
) -> torch.Tensor:
    """
    Convert Euler angles to rotation matrix.
    """
    if not isinstance(euler_angles, torch.Tensor):
        euler_angles = torch.tensor(euler_angles, dtype=torch.float32)  # or float64

    angles = euler_angles.clone()
    if degrees:
        angles = angles * (torch.pi / 180.0)

    if extrinsic:
        yaw, pitch, roll = angles.unbind(dim=-1)
    else:
        roll, pitch, yaw = angles.unbind(dim=-1)

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    if extrinsic:
        mat = torch.stack(
            [
                torch.stack(
                    [cp * cr, cr * sp * sy - cy * sr, cr * cy * sp + sr * sy], dim=-1
                ),
                torch.stack(
                    [cp * sr, cy * cr + sr * sp * sy, cy * sp * sr - cr * sy], dim=-1
                ),
                torch.stack([-sp, cp * sy, cy * cp], dim=-1),
            ],
            dim=-2,
        )
    else:
        mat = torch.stack(
            [
                torch.stack([cp * cy, -cp * sy, sp], dim=-1),
                torch.stack(
                    [cy * sr * sp + cr * sy, cr * cy - sr * sp * sy, -cp * sr], dim=-1
                ),
                torch.stack(
                    [-cr * cy * sp + sr * sy, cy * sr + cr * sp * sy, cr * cp], dim=-1
                ),
            ],
            dim=-2,
        )

    return mat  # shape: (..., 3, 3)


def quat_to_euler_angles(
    quat: torch.Tensor, degrees: bool = False, extrinsic: bool = True
) -> torch.Tensor:
    """
    Convert quaternion to Euler angles.
    """
    if not isinstance(quat, torch.Tensor):
        quat = torch.tensor(quat, dtype=torch.float32)
    mat = quat_to_rot_matrix(quat)
    return matrix_to_euler_angles(mat, degrees=degrees, extrinsic=extrinsic)


def euler_angles_to_quat(
    euler_angles: torch.Tensor,
    degrees: bool = False,
    extrinsic: bool = False,
    device=None,
) -> torch.Tensor:
    """Vectorized version of converting euler angles to quaternion (scalar first)

    Args:
        euler_angles (typing.Union[np.ndarray, torch.Tensor]): euler angles with shape (N, 3)
        degrees (bool, optional): True if degrees, False if radians. Defaults to False.
        extrinsic (bool, optional): True if the euler angles follows the extrinsic angles
                   convention (equivalent to ZYX ordering but returned in the reverse) and False if it follows
                   the intrinsic angles conventions (equivalent to XYZ ordering).
                   Defaults to True.

    Returns:
        typing.Union[np.ndarray, torch.Tensor]: quaternions representation of the angles (N, 4) - scalar first.
    """
    if not isinstance(euler_angles, torch.Tensor):
        euler_angles = torch.tensor(euler_angles, dtype=torch.float32)
    if extrinsic:
        order = "xyz"
    else:
        order = "XYZ"
    # TODO: implement a torch version
    rot = Rotation.from_euler(order, euler_angles.cpu().numpy(), degrees=degrees)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    result = torch.from_numpy(np.asarray(result, dtype=np.float32)).float().to(device)
    return result


def lookat_to_quatf(camera: Gf.Vec3f, target: Gf.Vec3f, up: Gf.Vec3f) -> Gf.Quatf:
    """[summary]

    Args:
        camera (Gf.Vec3f): [description]
        target (Gf.Vec3f): [description]
        up (Gf.Vec3f): [description]

    Returns:
        Gf.Quatf: Pxr quaternion object.
    """
    F = (target - camera).GetNormalized()
    R = Gf.Cross(up, F).GetNormalized()
    U = Gf.Cross(F, R)

    q = Gf.Quatf()
    trace = R[0] + U[1] + F[2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = Gf.Quatf(
            0.25 / s, Gf.Vec3f((U[2] - F[1]) * s, (F[0] - R[2]) * s, (R[1] - U[0]) * s)
        )
    else:
        if R[0] > U[1] and R[0] > F[2]:
            s = 2.0 * math.sqrt(1.0 + R[0] - U[1] - F[2])
            q = Gf.Quatf(
                (U[2] - F[1]) / s,
                Gf.Vec3f(0.25 * s, (U[0] + R[1]) / s, (F[0] + R[2]) / s),
            )
        elif U[1] > F[2]:
            s = 2.0 * math.sqrt(1.0 + U[1] - R[0] - F[2])
            q = Gf.Quatf(
                (F[0] - R[2]) / s,
                Gf.Vec3f((U[0] + R[1]) / s, 0.25 * s, (F[1] + U[2]) / s),
            )
        else:
            s = 2.0 * math.sqrt(1.0 + F[2] - R[0] - U[1])
            q = Gf.Quatf(
                (R[1] - U[0]) / s,
                Gf.Vec3f((F[0] + R[2]) / s, (F[1] + U[2]) / s, 0.25 * s),
            )
    return q


def gf_quat_to_np_array(
    orientation: typing.Union[Gf.Quatd, Gf.Quatf, Gf.Quaternion],
) -> np.ndarray:
    """Converts a pxr Quaternion type to a numpy array following [w, x, y, z] convention.

    Args:
        orientation (typing.Union[Gf.Quatd, Gf.Quatf, Gf.Quaternion]): Input quaternion object.

    Returns:
        np.ndarray: A (4,) quaternion array in (w, x, y, z).
    """
    quat = np.zeros(4)
    quat[1:] = orientation.GetImaginary()
    quat[0] = orientation.GetReal()
    return quat


def gf_rotation_to_np_array(orientation: Gf.Rotation) -> np.ndarray:
    """Converts a pxr Rotation type to a numpy array following [w, x, y, z] convention.

    Args:
        orientation (Gf.Rotation): Pxr rotation object.

    Returns:
        np.ndarray: A (4,) quaternion array in (w, x, y, z).
    """
    return gf_quat_to_np_array(orientation.GetQuat())


def euler_to_quat(euler: list[float]) -> list[float]:
    """Convert Euler angles (degrees) to quaternion in (w, x, y, z) format.

    Args:
        euler: Euler angles in (x, y, z) order (degrees)

    Returns:
        Quaternion in (w, x, y, z) format
    """
    rot = R.from_euler("xyz", euler, degrees=True)
    quat = rot.as_quat()  # Returns (x, y, z, w)
    return [quat[3], quat[0], quat[1], quat[2]]  # Convert to (w, x, y, z)
