"""Open-loop debug driver for DexOpenDrawerEnv.

Mirrors :mod:`DebugOpenDrawerEnv` but commands the 19-D
(pose + 12 finger joints) action expected by the Franka + Xhand robot. The
trajectory is loaded from the ``xhand_open_by_handle_trajectory`` annotation
on the Drawer/7120 articulation (source data:
``~/sharpa_bin+xhand_open_by_handle/xhand_open_by_handle``).

Each waypoint is ``[x, y, z, qw, qx, qy, qz, j0..j11]`` (rad).

Trajectory selection mirrors what the close-loop ``DexOpenDrawer`` atomic
skill does: stack the first waypoint of every ``joint_<i>/<traj_id>`` as a
candidate handle pose, submit them as a goalset to the curobo IK server,
and pick the trajectory whose ``goalset_index`` came back from IK. Falls
back to the first sorted key if IK times out / fails for every candidate.
"""

# Env class import must come first so omni gets registered before any
# Planner / IsaacLab imports.
from magicsim.Task.TableTop.Env.DexOpenDrawerEnv import DexOpenDrawerEnv

import time
import torch
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from loguru import logger as log

from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes, draw_waypoints
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest


# Arrival thresholds for waypoint tracking
EEF_POS_THRESHOLD = 0.03
EEF_ROT_THRESHOLD = 0.1
WAYPOINT_MAX_STEPS = 20

ARM_POSE_DIM = 7
FINGER_JOINT_DIM = 12
DEX_ACTION_DIM = ARM_POSE_DIM + FINGER_JOINT_DIM

# -1 → load all joints' trajectories (let IK pick); specific id → only that joint.
TARGET_JOINT_ID = -1
IK_TIMEOUT_SEC = 10.0


def _split_waypoint(waypoint: torch.Tensor) -> tuple[list[float], list[float]]:
    pose = waypoint[:ARM_POSE_DIM].cpu().tolist()
    joints = waypoint[ARM_POSE_DIM:DEX_ACTION_DIM].cpu().tolist()
    return pose, joints


def _pack_single_arm_goalset(
    arm_poses: torch.Tensor, hand_id: int, eef_num: int
) -> torch.Tensor:
    """Same packing as :meth:`AtomicSkill.pack_single_arm_goalset`.

    ``arm_poses``: ``(G, 7)``. Single-EEF robot returns ``(1, G, 7)``;
    multi-EEF places the active hand's poses into a NaN-filled
    ``(1, G, eef_num*7)`` tensor (NaN rows disable the inactive tool).
    """
    if arm_poses.ndim == 2:
        arm_poses = arm_poses.unsqueeze(0)
    if eef_num == 1:
        return arm_poses.contiguous()
    target = torch.full(
        (1, arm_poses.shape[1], eef_num, 7),
        float("nan"),
        device=arm_poses.device,
        dtype=arm_poses.dtype,
    )
    target[:, :, hand_id, :] = arm_poses
    return target.reshape(1, arm_poses.shape[1], eef_num * 7).contiguous()


def _select_trajectory_via_ik(
    env: DexOpenDrawerEnv,
    trajectories: dict,
    robot_id: int = 0,
    hand_id: int = 0,
) -> tuple[str, torch.Tensor]:
    """Submit candidate handle poses (first waypoint of each trajectory)
    to the IK server as a goalset and return ``(key, [N, 19] tensor)``
    chosen by ``goalset_index``. Falls back to the first sorted key if IK
    fails for every candidate.
    """
    keys = sorted(trajectories.keys())
    candidates_19d = torch.stack([trajectories[k][0] for k in keys], dim=0)
    candidates_7d = candidates_19d[:, :ARM_POSE_DIM].to(env.device).contiguous()

    planner_manager = getattr(env.scene, "planner_manager", None)
    if planner_manager is None or not getattr(planner_manager, "ik_server", None):
        print("[DebugDexOpenDrawer] No IK server, falling back to first key.")
        return keys[0], trajectories[keys[0]]

    ik_dict = planner_manager.ik_server
    robot_name_list = list(env.scene.robot_manager.robots.keys())
    robot_name = (
        robot_name_list[robot_id]
        if 0 <= robot_id < len(robot_name_list)
        else next(iter(ik_dict.keys()))
    )
    if robot_name not in ik_dict:
        print(f"[DebugDexOpenDrawer] No IK server for '{robot_name}', falling back.")
        return keys[0], trajectories[keys[0]]
    ik_server = ik_dict[robot_name]

    # Refresh world obstacles so IK collision checks reflect the spawned drawer.
    planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        env_ids=[0],
        obstacle_ignore_path_list=["articulation_items"],
    )

    robot_state = list(
        env.scene.robot_manager.get_robot_state(noise_flag=False)[0].values()
    )[0]
    robot_states_dict = {
        "base_pos": robot_state["base_pos"],
        "base_quat": robot_state["base_quat"],
        "joint_pos": robot_state["joint_pos"],
        "joint_vel": robot_state["joint_vel"],
    }

    eef_num = int(getattr(ik_server, "eef_num", 1))
    target = _pack_single_arm_goalset(candidates_7d, hand_id, eef_num)
    is_dual_ik = getattr(ik_server, "dual_mode", False)
    if is_dual_ik:
        req = DualIKPlanRequest(
            env_ids=[0],
            target_pos=target,
            robot_states=robot_states_dict,
            mode="goalset",
            lock_base=False,
        )
    else:
        req = IKPlanRequest(
            env_ids=[0],
            target_pos=target,
            robot_states=robot_states_dict,
            mode="goalset",
        )
    print(
        f"[DebugDexOpenDrawer] Submitting goalset of {candidates_7d.shape[0]} "
        f"candidate handle poses to IK ({robot_name}, dual={is_dual_ik})..."
    )

    fut = ik_server.submit_ik(req)
    deadline = time.monotonic() + IK_TIMEOUT_SEC
    while not fut.done() and time.monotonic() < deadline:
        env.sim_step()
    if not fut.done():
        print("[DebugDexOpenDrawer] IK future timed out, falling back.")
        return keys[0], trajectories[keys[0]]

    success_list, goalset_index_list, _ = fut.result()
    selected_idx = (
        int(goalset_index_list[0])
        if goalset_index_list and len(goalset_index_list) >= 1
        else -1
    )
    if selected_idx < 0 or not (success_list and bool(success_list[0])):
        print(
            f"[DebugDexOpenDrawer] IK FAILED (success={success_list}, "
            f"goalset_index={goalset_index_list}); falling back to {keys[0]}."
        )
        return keys[0], trajectories[keys[0]]

    selected_key = keys[selected_idx]
    print(
        f"[DebugDexOpenDrawer] IK selected idx={selected_idx} "
        f"key={selected_key} out of {len(keys)} candidates."
    )
    return selected_key, trajectories[selected_key]


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="dex_open_drawer_env"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: DexOpenDrawerEnv = gym.make(
        "DexOpenDrawerEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    for _ in range(40):
        env.sim_step()

    trajectories = env.get_drawer_trajectories(
        env_id=0,
        annotation_name="xhand_open_by_handle_trajectory",
        joint_id=TARGET_JOINT_ID,
    )
    print(
        f"Loaded {len(trajectories)} candidate trajectories: "
        f"{list(trajectories.keys())}"
    )
    if not trajectories:
        print("[DebugDexOpenDrawer] No trajectories found, idling.")
        while True:
            env.sim_step()

    # Visualise candidate handle poses (first waypoint of each trajectory).
    candidate_first_pose = torch.stack(
        [trajectories[k][0, :ARM_POSE_DIM] for k in sorted(trajectories.keys())],
        dim=0,
    )
    draw_waypoints(
        candidate_first_pose[:, :3].cpu().tolist(),
        point_size=12.0,
        color=(1.0, 1.0, 0.0, 0.8),
        clear_existing=True,
    )

    selected_key, trajectory = _select_trajectory_via_ik(
        env, trajectories, robot_id=0, hand_id=0
    )
    trajectory = trajectory.to(env.device)
    print(f"Using {selected_key}, shape: {trajectory.shape}")
    print(
        f"  First waypoint pose: "
        f"{[f'{v:.4f}' for v in trajectory[0, :ARM_POSE_DIM].tolist()]}"
    )
    print(
        f"  Last waypoint pose:  "
        f"{[f'{v:.4f}' for v in trajectory[-1, :ARM_POSE_DIM].tolist()]}"
    )

    draw_grasp_samples_as_axes(
        grasp_poses=trajectory[:, :ARM_POSE_DIM],
        axis_length=0.03,
        line_thickness=3,
        line_opacity=0.8,
        clear_existing=False,
    )

    handle_pose, handle_joints = _split_waypoint(trajectory[0])
    last_pose, _ = _split_waypoint(trajectory[-1])
    open_hand = [0.0] * FINGER_JOINT_DIM

    eef_pose = env.get_eef_pose(env_ids=[0])
    print(f"  EEF pose:    {[f'{v:.4f}' for v in eef_pose[0].tolist()]}")
    handle_pos_t = torch.tensor(handle_pose[:3], device=eef_pose.device)
    print(
        f"  Distance EEF -> handle: "
        f"{torch.norm(eef_pose[0, :3] - handle_pos_t).item():.3f}m"
    )

    # ---- Phase 1: Approach (pre-grasp, hand open) ----
    print("\n[DebugDexOpenDrawer] Phase 1/5: Approach (pre-grasp, 100 steps)")
    grasp_pos = torch.tensor(handle_pose[:3], device=env.device)
    grasp_quat = torch.tensor(handle_pose[3:7], device=env.device)
    rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))
    grasp_dir = rot_matrix[0, :, 2]
    grasp_dir = grasp_dir / torch.norm(grasp_dir)
    pre_pos = (grasp_pos - grasp_dir * 0.15).cpu().tolist()
    pre_action_list = pre_pos + handle_pose[3:7] + open_hand
    pre_action = torch.tensor(pre_action_list, device=env.device).unsqueeze(0)
    for _ in range(100):
        env.step(action=pre_action)

    # ---- Phase 2: Move to handle (hand still open) ----
    print("[DebugDexOpenDrawer] Phase 2/5: Move to handle (60 steps)")
    move_action = torch.tensor(handle_pose + open_hand, device=env.device).unsqueeze(0)
    for _ in range(60):
        env.step(action=move_action)

    # ---- Phase 3: Close fingers (use first-waypoint joint targets) ----
    print("[DebugDexOpenDrawer] Phase 3/5: Close fingers (80 steps)")
    close_action = torch.tensor(
        handle_pose + handle_joints, device=env.device
    ).unsqueeze(0)
    for _ in range(80):
        env.step(action=close_action)

    # ---- Phase 4: Pull along trajectory (full 19-D waypoints) ----
    num_waypoints = trajectory.shape[0]
    print(
        f"[DebugDexOpenDrawer] Phase 4/5: Pull trajectory "
        f"({num_waypoints} waypoints, max {WAYPOINT_MAX_STEPS} steps/wp)"
    )
    total_pull_steps = 0
    pos_diff = quat_diff = float("inf")
    step = 0
    for wp_idx in range(num_waypoints):
        waypoint = trajectory[wp_idx]
        wp_action = waypoint.unsqueeze(0).to(env.device)

        arrived = False
        for step in range(WAYPOINT_MAX_STEPS):
            env.step(action=wp_action)
            total_pull_steps += 1

            eef_pose = env.get_eef_pose(env_ids=[0])
            target_pos = waypoint[:3].to(eef_pose.device)
            target_quat = waypoint[3:7].to(eef_pose.device)
            pos_diff = torch.linalg.norm(eef_pose[0, :3] - target_pos).item()
            quat_diff = torch.min(
                torch.norm(eef_pose[0, 3:7] - target_quat),
                torch.norm(eef_pose[0, 3:7] + target_quat),
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

    # ---- Phase 5: Release fingers ----
    print("[DebugDexOpenDrawer] Phase 5/5: Release fingers (20 steps)")
    release_action = torch.tensor(last_pose + open_hand, device=env.device).unsqueeze(0)
    for _ in range(20):
        env.step(action=release_action)

    print("\n[DebugDexOpenDrawer] Done! Simulation alive for inspection.")
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
