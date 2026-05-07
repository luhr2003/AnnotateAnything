from typing import Dict, List, Tuple, Union
import torch
import numpy as np
from numpy.typing import ndarray
from magicsim.Env.Sensor.CameraManager import Camera
from magicsim.Env.Utils.rotations import (
    euler_angles_to_quat,
    quat_to_rot_matrix,
)


def execute_action(self, env_id: int, cur_camera: Camera, action: Dict):
    """
    Execute a predefined action for a specific environment.
    This method should be called when executing predefined actions for cameras.

    Args:
        env_id: The index of the environment to execute the action on.
        action: The action to be executed, which could be a dictionary containing function names and parameters.
    """
    cam_id = self.cameras[env_id].index(cur_camera)
    ((func_name, func_param),) = action.items()

    origin_pose = self.camera_poses[env_id][cam_id]
    if func_name == "move":
        noise = self.action_noise[env_id][cam_id]
        pos, ori = self.action_with_noise(func_param=func_param, noise=noise)
        self.move(env_id, cur_camera, pos, ori)  # func_param: tuple(translate, rotate)
    elif func_name == "move_to":
        noise = self.action_noise[env_id][cam_id]
        pos, ori = self.action_with_noise(func_param=func_param, noise=noise)
        self.move_to(
            env_id, cur_camera, pos, ori
        )  # func_param: tuple(position, orientation)
    elif func_name == "look_at":
        self.look_at(env_id, cur_camera, func_param)  # func_param: list[target_paths]
    elif func_name == "randomize_camera_pose":
        self.randomize_camera_pose(
            env_id, cur_camera, func_param[0], func_param[1], func_param[2]
        )  # func_param: tuple(pos_range, rot_range, type)
    elif func_name == "go":
        noise = self.action_noise[env_id][cam_id]
        pos, ori = self.action_with_noise(func_param=func_param, noise=noise)
        self.go(
            env_id, cur_camera, pos, ori
        )  # func_param: tuple(go_distance, go_rotation)
    elif func_name == "randomize_camera_pose_centered":
        self.randomize_camera_pose_centered(
            env_id,
            cur_camera,
            func_param,
        )  # func_param: tuple(pos_limit, max_rot_range_degrees, min_rot_range_degrees, type)
    else:
        raise ValueError(f"Invalid action in env:{env_id}, cam:{cam_id}")
    return None


def move_to(
    self,
    env_id: int,
    camera: Camera,
    position: List[float] | torch.Tensor = None,
    orientation: List[float] | torch.Tensor = None,
):
    """
    Move the camera to a specific position and orientation.

    Args:
        env_id: The index of the environment to move the camera for.
        camera: The camera to be moved.
        position: A list of position values [x, y, z].
        orientation: A list of orientation values [roll, pitch, yaw] in degrees.
    """
    # Implementation for moving the camera to a specific position and orientation
    cam_id = self.cameras[env_id].index(camera)
    cur_camera_xform = self.cameras_xform[env_id][cam_id]
    if position is not None:
        if len(position) == 3:
            if not isinstance(position, torch.Tensor):
                new_position = torch.tensor(position)
            else:
                new_position = position
        else:
            raise ValueError("Length of new position should be 3.")
    else:
        pass
    if orientation is not None:
        if len(orientation) == 3:
            if not isinstance(orientation, torch.Tensor):
                orientation = torch.tensor(orientation)
            new_quat = euler_angles_to_quat(orientation, degrees=True).reshape(4)

        else:
            raise ValueError(
                "Length of new orientation should be 3, in form of [roll, pitch, yaw] in degrees."
            )
    else:
        pass
    cur_camera_xform.set_local_pose(new_position, new_quat)
    return


def move(
    self,
    env_id: int,
    camera: Camera,
    translate: List[float] | torch.Tensor = None,
    rotate: List[float] | torch.Tensor = None,
):
    """
    Translate and rotate the camera based on the provided values.

    Args:
        env_id: The index of the environment to move the camera for.
        translate: A list of translation values [x, y, z].
        rotate: A list of rotation values [roll, pitch, yaw].
    """
    # Implementation for moving the camera
    cam_id = self.cameras[env_id].index(camera)
    cur_camera_xform = self.cameras_xform[env_id][cam_id]
    cur_position, cur_quat = cur_camera_xform.get_local_pose()
    cur_position = cur_position.cpu()
    cur_quat = cur_quat.cpu()
    if translate is not None:
        if not isinstance(translate, torch.Tensor):
            translate = torch.tensor(translate, dtype=torch.float32)
        if len(translate) == 3:
            new_position = cur_position.reshape(3) + translate
        else:
            raise ValueError("Length of translation should be 3.")
    else:
        new_position = cur_position
    if rotate is not None:
        if len(rotate) == 3:
            rotate1 = rotate.clone()
            rotate1[0], rotate1[1] = rotate[1], rotate[0]
            new_quat = self.quat_multiply(
                cur_quat.reshape(4),
                euler_angles_to_quat(rotate1, degrees=True, extrinsic=False),
            ).reshape(4)

        else:
            raise ValueError(
                "Length of rotation should be 3, in form of [roll, pitch, yaw] in degrees."
            )
    else:
        new_quat = cur_quat

    cur_camera_xform.set_local_pose(new_position, new_quat)
    return


def look_at(self, env_id: int, camera: Camera, target_path: str):
    """
    Make the camera look at a specific target.

    Args:
        env_id: The index of the environment to make the camera look at the target.
        camera: The camera to be used for looking at the target.
        target_path: The path to the target object in the scene.
    """
    # Implementation for making the camera look at a target
    pass


def quat_multiply(self, q1: torch.tensor, q2: torch.tensor) -> torch.tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.tensor([w, x, y, z])


def go(
    self,
    env_id: int,
    camera: Camera,
    go_distance: List[float] | torch.Tensor = None,
    go_rotation: List[float] | torch.Tensor = None,
):
    """
    This function moves the designated camera in its object coordinates. +x moves the camera forward, +y moves left, +z moves up. Pay attention to the difference between self.go and self.move

    Args:
        env_id (int): The env_id of the camera need to move
        camera (Camera): The camera need to move
        go_distance (List[float] | np.ndarray, optional): [dx, dy, dz] in the camera coordinates. None -> do not move
        go_rotation (List[float] | np.ndarray, optional): Euler angels [roll, pitch, yaw] in the camera coordinates. None -> do not rotate

    """
    if go_distance is None:
        go_distance = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        if not isinstance(go_distance, torch.Tensor):
            go_distance = torch.tensor(go_distance, dtype=torch.float32)
        else:
            go_distance = go_distance.to(torch.float32)
        go_distance /= 9
    if go_rotation is None:
        go_rotation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        if not isinstance(go_rotation, torch.Tensor):
            go_rotation = torch.tensor(go_rotation, dtype=torch.float32)
        else:
            go_rotation = go_rotation.to(torch.float32)
    cam_id = self.cameras[env_id].index(camera)
    cur_camera_xform = self.cameras_xform[env_id][cam_id]
    cur_pos, cur_quat = cur_camera_xform.get_local_pose()
    cur_pos = cur_pos.reshape(3).cpu()
    cur_quat = cur_quat.reshape(4).cpu()
    R = quat_to_rot_matrix(cur_quat)

    world_delta = go_distance @ R
    world_delta1 = world_delta.clone()
    world_delta1[0], world_delta1[1] = (
        world_delta[1],
        world_delta[0],
    )
    # This is becuase the definition of camera coordinates, the input here use FLU (x: front, y:left, z:up), but the matrix and quats use (x:left, y:front, z: up) For rotation, it's similar.
    new_pos = cur_pos + world_delta1
    go_rotation1 = go_rotation.clone()
    go_rotation1[0], go_rotation1[1] = go_rotation[1], go_rotation[0]
    dq = euler_angles_to_quat(go_rotation1, degrees=True)
    new_quat = self.quat_multiply(cur_quat, dq)
    cur_camera_xform.set_local_pose(new_pos, new_quat)
    return


def randomize_camera_pose(
    self,
    env_id: int,
    camera: Camera,
    pos_range: List[float] | torch.Tensor = None,
    rot_range: List[float] | torch.Tensor = None,
    type: str = "uniform",
):
    """
    Randomize the camera pose for a specific environment.

    Args:
        env_id: The index of the environment to randomize the camera pose for.
        camera: The camera to be randomized.
        pos_range: A list of position ranges [min_x, max_x, min_y, max_y, min_z, max_z].
        rot_range: A list of rotation ranges [min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw].
    """
    # Implementation for randomizing the camera pose
    cam_id = self.cameras[env_id].index(camera)
    cur_camera_xform = self.cameras_xform[env_id][cam_id]
    if pos_range is not None:
        try:
            pos_range = torch.tensor(pos_range).reshape(
                6,
            )
            if type == "uniform":
                random_pos = torch.tensor(
                    np.random.uniform(pos_range[::2], pos_range[1::2]).reshape(3)
                )
            elif type == "normal":
                random_pos = torch.tensor(
                    np.random.normal(
                        pos_range.reshape(3, 2).mean(dim=1),
                        pos_range.reshape(3, 2).std(dim=1),
                    ).reshape(3)
                )
            else:
                raise ValueError("Type should be uniform or normal")
        except ValueError:
            raise ValueError(
                "Position range should be [min_x, max_x, min_y, max_y, min_z, max_z]"
            )
    else:
        random_pos = torch.zeros(3)
    if rot_range is not None:
        try:
            rot_range = torch.tensor(rot_range).reshape(
                6,
            )
            if type == "uniform":
                random_rot = torch.tensor(
                    np.random.uniform(rot_range[::2], rot_range[1::2]).reshape(3)
                )
                random_rot = euler_angles_to_quat(random_rot, degrees=True)
            elif type == "normal":
                random_rot = torch.tensor(
                    np.random.normal(
                        rot_range.to(torch.float32).reshape(3, 2).mean(dim=1),
                        rot_range.to(torch.float32).reshape(3, 2).std(dim=1),
                    ).reshape(3)
                )
                random_rot = euler_angles_to_quat(random_rot, degrees=True)
            else:
                raise ValueError("Type should be uniform or normal")
        except ValueError:
            raise ValueError(
                "Rotation range should be [min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw]"
            )
    else:
        random_rot = euler_angles_to_quat(torch.zeros(3), degrees=True)
    cur_camera_xform.set_local_pose(random_pos, random_rot)
    return


def calculate_target_orientation(
    self, direction_to_center: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Calculate target orientation (roll, pitch, yaw) from a direction vector.

    Args:
        direction_to_center: 3D vector pointing from camera to target center

    Returns:
        Tuple[float, float, float]: target_roll, target_pitch, target_yaw in degrees
    """
    # Normalize the direction vector
    direction = direction_to_center / torch.norm(direction_to_center)

    # Extract components
    x, y, z = direction[0].item(), direction[1].item(), direction[2].item()

    # Calculate pitch (rotation around x-axis)
    # pitch = arcsin(-z) for standard camera conventions
    pitch = torch.asin(torch.tensor(-z, dtype=torch.float32)) * 180.0 / torch.pi

    # Calculate yaw (rotation around z-axis)
    # yaw = atan2(y, x) for standard camera conventions
    yaw = (
        torch.atan2(
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
        )
        * 180.0
        / torch.pi
    )

    # Roll is typically set to 0 for "look-at" behavior (camera remains level)
    # Unless you have specific requirements for roll orientation
    roll = 0.0

    return roll, pitch.item(), yaw.item()


def randomize_camera_pose_centered(
    self,
    env_id: int,
    camera: "Camera",
    pos_limit: Union[List[float], torch.Tensor],
    max_rot_range_degrees: float = 90.0,  # Max deviation from 'look-at-center' direction
    min_rot_range_degrees: float = 5.0,  # Min deviation (at the boundary)
    type: str = "uniform",
):
    """
    Randomizes the camera pose for a specific environment. The rotation range
    is calculated based on the camera's new position relative to the center
    of the defined position box (pos_limit).

    Args:
        env_id: The index of the environment to randomize the camera pose for.
        camera: The camera object to be randomized.
        pos_limit: A list of position ranges [min_x, max_x, min_y, max_y, min_z, max_z].
        max_rot_range_degrees: The maximum rotational freedom (at the center point).
        min_rot_range_degrees: The minimum rotational freedom (at the boundary).
        type: The type of randomization (e.g., "uniform").
    """

    cam_id = self.cameras[env_id].index(camera)
    cur_camera_xform = self.cameras_xform[env_id][cam_id]

    # Ensure pos_limit is a torch.Tensor for calculations
    if not isinstance(pos_limit, torch.Tensor):
        pos_limit = torch.as_tensor(pos_limit, dtype=torch.float32)

    # 1. Calculate the Center Point from pos_limit
    # pos_limit is [min_x, max_x, min_y, max_y, min_z, max_z]
    mins = pos_limit[0::2]  # [min_x, min_y, min_z]
    maxs = pos_limit[1::2]  # [max_x, max_x, max_z]

    # Center = (min + max) / 2
    center_point = (mins + maxs) / 2.0

    # 2. Randomize Camera Position (P_new)
    # Generate a uniform random position P_new within the defined box
    new_pos = torch.rand(3) * (maxs - mins) + mins

    # 3. Calculate Distance and Normalized Distance

    # Calculate Euclidean distance from new_pos to center_point
    dist_vector = new_pos - center_point
    dist_to_center = torch.norm(dist_vector)

    # Determine the maximum possible distance (from center to a corner)
    # for normalization (dist_norm).
    half_diag_vector = (maxs - mins) / 2.0
    max_dist = torch.norm(half_diag_vector)

    # Normalize distance: 0 at center, 1 at max_dist (corner)
    normalized_dist = torch.clamp(dist_to_center / max_dist, 0.0, 1.0)

    # 4. Determine Rotation Range based on Normalized Distance

    # Linear interpolation:
    # Near center (d_norm=0) -> range = max_rot_range_degrees
    # Near boundary (d_norm=1) -> range = min_rot_range_degrees
    rot_range_half = (
        max_rot_range_degrees
        - (max_rot_range_degrees - min_rot_range_degrees) * normalized_dist
    )

    # 5. Calculate the Target (Look-at-Center) Rotation

    # The direction the camera MUST look towards at the boundary
    direction_to_center = center_point - new_pos

    # --- System-Dependent Conversion ---
    # Convert this vector to a set of target Euler angles (roll, pitch, yaw)
    # or a target Quaternion. This conversion (vector_to_euler or look_at_to_quat)
    # is crucial and system-dependent (e.g., how "up" is defined).
    #
    # target_euler = self.vector_to_euler(direction_to_center) # e.g., [r_t, p_t, y_t]

    # Placeholder: Assume we get target angles [roll_t, pitch_t, yaw_t]
    # For simplicity in this example, let's assume we have them.
    # Replace this with your actual conversion logic:
    target_roll, target_pitch, target_yaw = self.calculate_target_orientation(
        -direction_to_center
    )

    # 6. Final Rotation Randomization

    # Randomly sample the final angles within the calculated range [target_angle +/- rot_range_half]

    # Sample deviation for each angle
    roll_dev = torch.rand(1) * (2 * rot_range_half) - rot_range_half
    pitch_dev = torch.rand(1) * (2 * rot_range_half) - rot_range_half
    yaw_dev = torch.rand(1) * (2 * rot_range_half) - rot_range_half

    new_roll = 0
    new_pitch = 0
    new_yaw = target_yaw + 90

    # 7. Update Camera Pose (Final step, system-dependent)

    new_rot = euler_angles_to_quat(
        torch.tensor([new_roll, new_pitch, new_yaw]), degrees=True
    )
    cur_camera_xform.set_local_pose(new_pos, new_rot)


def get_camera_xform_pose(self) -> List[List[tuple[ndarray, ndarray]]]:
    """
    Get the current positions of all cameras in all envs.

    Returns:
        List[List[ndarray, ndarray]]: For every env, for every camera, return its position and orientation. Position [x, y, z], Orientation quat [w, x, y, z]
    """
    return self.camera_poses
