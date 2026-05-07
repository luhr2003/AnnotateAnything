import torch
from magicsim.Task.TableTop.Env.OpenDrawerEnv import OpenDrawerEnv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


# Arrival thresholds for waypoint tracking
EEF_POS_THRESHOLD = 0.03
EEF_ROT_THRESHOLD = 0.1
WAYPOINT_MAX_STEPS = 20


@hydra.main(version_base=None, config_path="../../Conf", config_name="open_drawer_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: OpenDrawerEnv = gym.make(
        "OpenDrawerEnv-V0", config=cfg, cli_args=None, logger=logger
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
    action = handle_pose.cpu().numpy().tolist()

    # Print EEF pose for sanity check
    eef_pose = env.get_eef_pose(env_ids=[0])
    print(f"  EEF pose:    {[f'{v:.4f}' for v in eef_pose[0].tolist()]}")
    dist = torch.norm(eef_pose[0, :3] - handle_pose[:3].to(eef_pose.device)).item()
    print(f"  Distance EEF -> handle: {dist:.3f}m")

    # ---- Phase 1: Approach (pre-grasp) ----
    print("\n[DebugOpenDrawer] Phase 1/5: Approach (pre-grasp, 100 steps)")
    for i in range(100):
        pre_action = action.copy()
        grasp_pos = torch.tensor(pre_action[:3], device=env.device)
        grasp_quat = torch.tensor(pre_action[3:7], device=env.device)

        rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))
        grasp_direction = rot_matrix[0, :, 2]
        grasp_direction_normalized = grasp_direction / torch.norm(grasp_direction)

        offset = grasp_direction_normalized * 0.15
        new_pos = grasp_pos - offset
        pre_action[:3] = new_pos.cpu().tolist()
        pre_action = torch.tensor(pre_action, device=env.device).unsqueeze(0)
        env.step(action=pre_action)

    # ---- Phase 2: Move to handle ----
    print("[DebugOpenDrawer] Phase 2/5: Move to handle (60 steps)")
    for i in range(60):
        grasp_action = torch.tensor(action, device=env.device).unsqueeze(0)
        env.step(action=grasp_action)

    # ---- Phase 3: Close gripper ----
    print("[DebugOpenDrawer] Phase 3/5: Close gripper (20 steps)")
    for i in range(80):
        close_action = action.copy()
        close_action.append(1)  # gripper close
        close_action = torch.tensor(close_action, device=env.device).unsqueeze(0)
        env.step(action=close_action)

    # ---- Phase 4: Pull along trajectory ----
    num_waypoints = trajectory.shape[0]
    print(
        f"[DebugOpenDrawer] Phase 4/5: Pull trajectory "
        f"({num_waypoints} waypoints, max {WAYPOINT_MAX_STEPS} steps/wp)"
    )

    total_pull_steps = 0
    for wp_idx in range(num_waypoints):
        waypoint = trajectory[wp_idx]
        wp_action_list = waypoint.cpu().numpy().tolist()
        wp_action_list.append(1)  # keep gripper closed

        arrived = False
        for step in range(WAYPOINT_MAX_STEPS):
            wp_tensor = torch.tensor(wp_action_list, device=env.device).unsqueeze(0)
            env.step(action=wp_tensor)
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
    print("[DebugOpenDrawer] Phase 5/5: Release gripper (20 steps)")
    last_wp = trajectory[-1].cpu().numpy().tolist()
    for i in range(20):
        release_action = last_wp.copy()
        release_action.append(0)  # gripper open
        release_action = torch.tensor(release_action, device=env.device).unsqueeze(0)
        env.step(action=release_action)

    print("\n[DebugOpenDrawer] Done! Simulation alive for inspection.")
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
