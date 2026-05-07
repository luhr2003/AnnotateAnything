import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import torch
import math


def p(*a):
    print(*a, flush=True)


def quat_from_axis_angle(axis, angle):
    """Create quaternion (w, x, y, z) from axis-angle representation."""
    half_angle = angle / 2
    s = math.sin(half_angle)
    return [math.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s]


def interpolate_pose(pose1, pose2, t):
    """Linear interpolation between two poses (position + quaternion)."""
    return [p1 + t * (p2 - p1) for p1, p2 in zip(pose1, pose2)]


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="humanoid_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # ========== Define EEF Trajectory Waypoints (Pelvis Frame) ==========
    # Coordinate System (Relative to Pelvis):
    # x: Forward (+), Backward (-)
    # y: Left (+), Right (-)
    # z: Up (+), Down (-)
    # Note: Pelvis is roughly at z=0.
    # Shoulders are roughly at z=+0.3 to +0.4
    # Hips/Thighs are roughly at z=-0.1 to -0.2

    # Format: [x, y, z, qw, qx, qy, qz]
    # Quaternion (w, x, y, z) - identity is [1, 0, 0, 0]

    # --- Left Arm Waypoints (y should be positive) ---
    left_waypoints = [
        # 1. Home position (relaxed at side/hip)
        [0.2, 0.3, -0.1, 1.0, 0.0, 0.0, 0.0],
        # 2. Reach Forward (Chest height)
        [0.5, 0.25, 0.2, 0.707, 0.0, 0.707, 0.0],  # Palm inward
        # 3. High Reach (Above head, slightly forward)
        [0.3, 0.3, 0.6, 0.707, 0.0, 0.707, 0.0],
        # 4. T-Pose Left (Shoulder height, out to side)
        [0.1, 0.6, 0.35, 0.707, 0.707, 0.0, 0.0],  # Palm down
        # 5. Cross Body (Reach to right hip)
        [0.3, -0.1, -0.1, 0.5, 0.5, 0.5, 0.5],
        # 6. Deep Forward/Down (Picking from ground in front)
        [0.5, 0.2, -0.3, 0.0, 0.707, 0.0, 0.707],  # Palm down
        # 7. Wide High Embrace (YMCA 'Y' shape)
        [0.2, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5],
        # 8. Back to Home
        [0.2, 0.3, -0.1, 1.0, 0.0, 0.0, 0.0],
    ]

    # --- Right Arm Waypoints (Mirrored: y is negative) ---
    right_waypoints = [
        # 1. Home position
        [0.2, -0.3, -0.1, 1.0, 0.0, 0.0, 0.0],
        # 2. Reach Forward
        [0.5, -0.25, 0.2, 0.707, 0.0, 0.707, 0.0],
        # 3. High Reach
        [0.3, -0.3, 0.6, 0.707, 0.0, 0.707, 0.0],
        # 4. T-Pose Right
        [0.1, -0.6, 0.35, 0.707, -0.707, 0.0, 0.0],
        # 5. Cross Body (Reach to left hip)
        [0.3, 0.1, -0.1, 0.5, -0.5, 0.5, -0.5],
        # 6. Deep Forward/Down
        [0.5, -0.2, -0.3, 0.0, 0.707, 0.0, 0.707],
        # 7. Wide High Embrace
        [0.2, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        # 8. Back to Home
        [0.2, -0.3, -0.1, 1.0, 0.0, 0.0, 0.0],
    ]

    # ========== Trajectory Execution Parameters ==========
    # Increased steps for smoother, slower motion to be visible
    steps_per_waypoint = 20  # ~1 second per segment at 60Hz

    current_waypoint_idx = 0
    interpolation_step = 0
    num_waypoints = len(left_waypoints)

    step_count = 0
    loop_count = 0

    print("\n" + "=" * 60)
    print("Starting PinkIK Trajectory Test (Local/Pelvis Frame)")
    print(f"Total waypoints: {num_waypoints}")
    print(f"Steps per waypoint: {steps_per_waypoint}")
    print("=" * 60 + "\n")

    while True:
        # Get target waypoints
        next_waypoint_idx = (current_waypoint_idx + 1) % num_waypoints
        target_left = left_waypoints[next_waypoint_idx]
        target_right = right_waypoints[next_waypoint_idx]
        start_left = left_waypoints[current_waypoint_idx]
        start_right = right_waypoints[current_waypoint_idx]

        # Calculate interpolation factor (0 to 1)
        t = interpolation_step / steps_per_waypoint
        # Smooth interpolation using ease-in-out
        t_smooth = 0.5 * (1 - math.cos(t * math.pi))

        # Interpolate poses
        left_pose = interpolate_pose(start_left, target_left, t_smooth)
        right_pose = interpolate_pose(start_right, target_right, t_smooth)

        # Build action: Body (7) + Left Arm Pose (7) + Right Arm Pose (7) + Hand (14)
        action_list = (
            [0.3, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0] + left_pose + right_pose + [0.0] * 14
        )
        # action_list = [0.0] * 15 + left_pose + right_pose + [0.0] * 14

        action = torch.tensor(action_list, device=env.device, dtype=torch.float32)
        action = action.unsqueeze(0).repeat(env.num_envs, 1)

        env.step(action=action)

        # Update interpolation
        interpolation_step += 1
        step_count += 1

        # Check if we've reached the target waypoint
        if interpolation_step >= steps_per_waypoint:
            interpolation_step = 0
            current_waypoint_idx = next_waypoint_idx

            print(
                f"[Loop {loop_count + 1}] Reached waypoint {current_waypoint_idx + 1}/{num_waypoints}: "
                f"L_pos=({target_left[0]:.2f}, {target_left[1]:.2f}, {target_left[2]:.2f}) "
            )

            # Check if we completed a full loop
            if current_waypoint_idx == 0:
                loop_count += 1
                print(
                    f"\n>>> Completed trajectory loop {loop_count} ({step_count} total steps) <<<\n"
                )

    env.close()


if __name__ == "__main__":
    main()
