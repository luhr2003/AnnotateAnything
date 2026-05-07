import torch
from magicsim.Task.TableTop.Env.CloseDrawerEnv import CloseDrawerEnv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log

# Annotation: close_by_push_trajectory.json
CLOSE_BY_PUSH_ANNOTATION = "close_by_push_trajectory"

# Arrival thresholds for waypoint tracking
EEF_POS_THRESHOLD = 0.03
EEF_ROT_THRESHOLD = 0.1
WAYPOINT_MAX_STEPS = 20


@hydra.main(version_base=None, config_path="../../Conf", config_name="close_drawer_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: CloseDrawerEnv = gym.make(
        "CloseDrawerEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    # Let the simulation settle
    for i in range(40):
        env.sim_step()

    # Get the articulation object (env 0, first articulation item)
    articulation_obj = env.scene.scene_manager.articulation_objects[0][
        "articulation_items"
    ][0]

    # Set initial joint positions from annotation (drawers open for close_by_push)
    traj_annotation = articulation_obj.get_annotation(CLOSE_BY_PUSH_ANNOTATION)
    if traj_annotation and "initial_joint_angles" in traj_annotation:
        init_angles = traj_annotation["initial_joint_angles"]
        positions = [
            init_angles["joint_0"] / 3,
            init_angles["joint_1"] / 3,
            init_angles["joint_2"] / 3,
        ]
        articulation_obj.set_current_joint_positions(positions)

    # Get trajectory poses (close_by_push: first=push start, last=push end)
    traj_data = articulation_obj.get_trajectory_poses(
        annotation_name=CLOSE_BY_PUSH_ANNOTATION,
        joint_name="joint_2",  # use one drawer for demo
        transform_to_world=True,
    )
    trajs = traj_data.get("trajectories") or traj_data.get("grasp_trajectories")
    joint_trajs = trajs["joint_2"]
    first_traj_key = sorted(joint_trajs.keys())[0]
    trajectory = joint_trajs[first_traj_key]
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
    else:
        trajectory = trajectory.detach().clone().float()

    print(f"Using traj key: {first_traj_key}, shape: {trajectory.shape}")
    print(
        f"  First waypoint (push start): {[f'{v:.4f}' for v in trajectory[0].tolist()]}"
    )
    print(
        f"  Last waypoint (push end):    {[f'{v:.4f}' for v in trajectory[-1].tolist()]}"
    )

    # Visualize the trajectory waypoints as axes
    draw_grasp_samples_as_axes(
        grasp_poses=trajectory,
        axis_length=0.03,
        line_thickness=3,
        line_opacity=0.8,
        clear_existing=True,
    )

    # Push pose = first waypoint of trajectory (where to place hand to push)
    push_pose = trajectory[0]
    action = push_pose.cpu().numpy().tolist()

    eef_pose = env.get_eef_pose(env_ids=[0])
    print(f"  EEF pose:    {[f'{v:.4f}' for v in eef_pose[0].tolist()]}")
    dist = torch.norm(eef_pose[0, :3] - push_pose[:3].to(eef_pose.device)).item()
    print(f"  Distance EEF -> push start: {dist:.3f}m")

    # ---- Phase 1: Pregrasp (approach to push position) ----
    print("\n[DebugCloseDrawer] Phase 1/3: Pregrasp (approach, 100 steps)")
    for i in range(100):
        pre_action = action.copy()
        push_pos = torch.tensor(pre_action[:3], device=env.device)
        push_quat = torch.tensor(pre_action[3:7], device=env.device)

        rot_matrix = quat_to_rot_matrix(push_quat.unsqueeze(0))
        push_direction = rot_matrix[0, :, 2]
        push_direction_normalized = push_direction / torch.norm(push_direction)

        offset = push_direction_normalized * 0.15
        new_pos = push_pos - offset
        pre_action[:3] = new_pos.cpu().tolist()
        pre_action = torch.tensor(pre_action, device=env.device).unsqueeze(0)
        env.step(action=pre_action)

    # ---- Phase 2: Move to push position + Close gripper ----
    print(
        "[DebugCloseDrawer] Phase 2/3: Move to push position & close gripper (80 steps)"
    )
    for i in range(80):
        close_action = action.copy()
        close_action.append(1)  # gripper close
        close_action = torch.tensor(close_action, device=env.device).unsqueeze(0)
        env.step(action=close_action)

    # ---- Phase 3: Push along trajectory ----
    num_waypoints = trajectory.shape[0]
    print(
        f"[DebugCloseDrawer] Phase 3/3: Push trajectory "
        f"({num_waypoints} waypoints, max {WAYPOINT_MAX_STEPS} steps/wp)"
    )

    total_push_steps = 0
    for wp_idx in range(num_waypoints):
        waypoint = trajectory[wp_idx]
        wp_action_list = waypoint.cpu().numpy().tolist()
        wp_action_list.append(1)  # keep gripper closed

        arrived = False
        for step in range(WAYPOINT_MAX_STEPS):
            wp_tensor = torch.tensor(wp_action_list, device=env.device).unsqueeze(0)
            env.step(action=wp_tensor)
            total_push_steps += 1

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

    print(f"  Push complete: {total_push_steps} total steps")

    print("\n[DebugCloseDrawer] Done! Simulation alive for inspection.")
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
