"""Open-loop debug for MobileOpenDrawer.

Fully mirrors :mod:`Task.TableTop.Env.Test.DebugOpenDrawerEnv`:
  1. reset + let sim settle
  2. pull the world-frame trajectory off the articulation annotation
  3. feed it to ``env.step`` phase-by-phase (approach → grasp → close → pull → release)

The only things that differ from the tabletop version:
  * action is ``[base(3), arm_pose(7), gripper(1)] = 11D`` (ridgebackFranka layout)
  * the base stays at zeros so the arm is the only mover — this isolates the
    arm-tracking path; the cabinet is placed close enough that the arm can
    reach it without a mobile-base correction (we override
    ``Scene.robot.RidgebackFranka.common.initial_pos_range`` so the robot
    spawns right in front of the cabinet).

Expected joint on the spawned drawer: ``joint_2`` (top drawer, same as the
tabletop debug).
"""

import torch
from magicsim.Task.MobileManip.Env.MobileOpenDrawerEnv import MobileOpenDrawerEnv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


# Arrival thresholds for waypoint tracking (match DebugOpenDrawerEnv)
EEF_POS_THRESHOLD = 0.03
EEF_ROT_THRESHOLD = 0.1
WAYPOINT_MAX_STEPS = 20

# Mirror the tabletop DebugOpenDrawerEnv layout: drawer at x=1.0 y=0 with
# yaw=0 (handle faces -x), robot at origin facing +x. Relative distance and
# orientation are identical to the fixed-base Franka in tabletop, only the
# z stack is shifted (ridgeback on floor instead of Franka-on-desk).
ROBOT_INITIAL_POS_RANGE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ROBOT_INITIAL_ORI_RANGE = [0, 0, 0, 0, 0, 0]  # face +x toward the cabinet


def _build_action(
    pose7: list[float] | torch.Tensor,
    gripper: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Pack ``[base(8)=NaN, arm_pose(7), gripper(1)]`` into a single-env action.

    The ridgebackFranka planner's base input is an 8-dim
    ``[x, y, z, qw, qx, qy, qz, lock_flag]`` vector consumed by
    ``RidgebackFrankaPControllerHelper.preprocess``. An all-NaN row is
    interpreted as ``lock_flag=-1`` (lock_skip) and the target pose falls
    back to the current base pose — i.e. base holds still, which is what
    this open-loop arm-only debug wants.

    ``gripper=None`` pads 0.0 (open). Always returns shape ``[1, 16]``.
    """
    if isinstance(pose7, torch.Tensor):
        pose = pose7.detach().cpu().tolist()
    else:
        pose = list(pose7)
    g = 0.0 if gripper is None else float(gripper)
    nan = float("nan")
    row = [nan] * 8 + pose[:7] + [g]  # 8 + 7 + 1 = 16
    return torch.tensor(row, device=device, dtype=torch.float32).unsqueeze(0)


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="mobile_open_drawer_env"
)
def main(cfg: DictConfig):
    # Force the robot closer to the cabinet before env instantiation.
    OmegaConf.set_struct(cfg, False)
    cfg.Scene.robot.RidgebackFranka.common.initial_pos_range = ROBOT_INITIAL_POS_RANGE
    cfg.Scene.robot.RidgebackFranka.common.initial_ori_range = ROBOT_INITIAL_ORI_RANGE
    OmegaConf.set_struct(cfg, True)
    print(cfg)

    logger = Logger("Env", log)
    env: MobileOpenDrawerEnv = gym.make(
        "MobileOpenDrawerEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    # Let the simulation settle
    for i in range(40):
        env.sim_step()

    # Get the articulation object (env 0, first articulation item)
    articulation_obj = env.scene.scene_manager.articulation_objects[0][
        "articulation_items"
    ][0]

    # Get trajectory poses (transformed to world coordinates)
    traj_data = articulation_obj.get_trajectory_poses(
        annotation_name="open_by_handle_trajectory",
        joint_name="joint_2",
        transform_to_world=True,
    )
    print(f"Trajectory data keys: {list(traj_data.keys())}")

    # Support both "trajectories" and "grasp_trajectories" keys
    trajs = traj_data.get("trajectories") or traj_data.get("grasp_trajectories")
    joint_trajs = trajs["joint_2"]
    first_traj_key = sorted(joint_trajs.keys())[0]
    trajectory = joint_trajs[first_traj_key]  # Tensor (N, 7)
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
    else:
        trajectory = trajectory.detach().clone().float()

    print(f"Using traj key: {first_traj_key}, shape: {trajectory.shape}")
    print(f"  First waypoint (handle): {[f'{v:.4f}' for v in trajectory[0].tolist()]}")
    print(f"  Last waypoint (pulled):  {[f'{v:.4f}' for v in trajectory[-1].tolist()]}")

    # Visualize the trajectory waypoints as axes
    draw_grasp_samples_as_axes(
        grasp_poses=trajectory,
        axis_length=0.03,
        line_thickness=3,
        line_opacity=0.8,
        clear_existing=True,
    )

    # Handle pose = first waypoint of trajectory
    handle_pose = trajectory[0]
    handle_list = handle_pose.cpu().numpy().tolist()

    # Print EEF pose for sanity check
    eef_pose = env.get_eef_pose(env_ids=[0])
    print(f"  EEF pose:    {[f'{v:.4f}' for v in eef_pose[0].tolist()]}")
    dist = torch.norm(eef_pose[0, :3] - handle_pose[:3].to(eef_pose.device)).item()
    print(f"  Distance EEF -> handle: {dist:.3f}m")

    # ---- Phase 1: Approach (pre-grasp, gripper open) ----
    print("\n[DebugMobileOpenDrawer] Phase 1/5: Approach (pre-grasp, 100 steps)")
    grasp_pos = torch.tensor(handle_list[:3], device=env.device)
    grasp_quat = torch.tensor(handle_list[3:7], device=env.device)
    rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))
    grasp_direction = rot_matrix[0, :, 2]
    grasp_direction_normalized = grasp_direction / torch.norm(grasp_direction)
    pre_grasp_pos = grasp_pos - grasp_direction_normalized * 0.15
    pre_grasp_pose = pre_grasp_pos.cpu().tolist() + handle_list[3:7]

    for i in range(100):
        env.step(action=_build_action(pre_grasp_pose, gripper=0.0, device=env.device))

    # ---- Phase 2: Move to handle (gripper open) ----
    print("[DebugMobileOpenDrawer] Phase 2/5: Move to handle (60 steps)")
    for i in range(60):
        env.step(action=_build_action(handle_list, gripper=0.0, device=env.device))

    # ---- Phase 3: Close gripper ----
    print("[DebugMobileOpenDrawer] Phase 3/5: Close gripper (80 steps)")
    for i in range(80):
        env.step(action=_build_action(handle_list, gripper=1.0, device=env.device))

    # ---- Phase 4: Pull along trajectory (gripper closed) ----
    num_waypoints = trajectory.shape[0]
    print(
        f"[DebugMobileOpenDrawer] Phase 4/5: Pull trajectory "
        f"({num_waypoints} waypoints, max {WAYPOINT_MAX_STEPS} steps/wp)"
    )

    total_pull_steps = 0
    pos_diff = quat_diff = float("inf")
    for wp_idx in range(num_waypoints):
        waypoint = trajectory[wp_idx]
        wp_pose = waypoint.cpu().numpy().tolist()

        arrived = False
        for step in range(WAYPOINT_MAX_STEPS):
            env.step(action=_build_action(wp_pose, gripper=1.0, device=env.device))
            total_pull_steps += 1

            # Arrival check
            eef_pose = env.get_eef_pose(env_ids=[0])
            eef_pos = eef_pose[0, :3]
            eef_quat = eef_pose[0, 3:7]
            target_pos = waypoint[:3].to(eef_pos.device)
            target_quat = waypoint[3:7].to(eef_quat.device)

            pos_diff = torch.linalg.norm(eef_pos - target_pos).item()
            quat_diff = torch.min(
                torch.norm(eef_quat - target_quat),
                torch.norm(eef_quat + target_quat),
            ).item()

            if pos_diff < EEF_POS_THRESHOLD and quat_diff < EEF_ROT_THRESHOLD:
                arrived = True
                break

        if wp_idx % max(1, num_waypoints // 5) == 0:
            print(
                f"  wp {wp_idx}/{num_waypoints}: "
                f"arrived={arrived}, steps={step + 1}, "
                f"pos_diff={pos_diff:.4f}, rot_diff={quat_diff:.4f}"
            )

    print(f"  Pull complete: {total_pull_steps} total steps")

    # ---- Phase 5: Release gripper ----
    print("[DebugMobileOpenDrawer] Phase 5/5: Release gripper (20 steps)")
    last_wp = trajectory[-1].cpu().numpy().tolist()
    for i in range(20):
        env.step(action=_build_action(last_wp, gripper=0.0, device=env.device))

    print("\n[DebugMobileOpenDrawer] Done! Simulation alive for inspection.")
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
