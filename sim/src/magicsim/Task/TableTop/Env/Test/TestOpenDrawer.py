import os
import json
import torch
from magicsim.Task.TableTop.Env.OpenDrawerEnv import OpenDrawerEnv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from magicsim.Collect.Record.utils import (
    to_serializable,
    is_image_annotator,
    extract_image_data,
    convert_annotator_data_to_payload,
)
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


# Arrival thresholds for waypoint tracking
EEF_POS_THRESHOLD = 0.03
EEF_ROT_THRESHOLD = 0.1
WAYPOINT_MAX_STEPS = 20

# Target joint to open
TARGET_JOINT = "joint_2"

# Output directory for saved data
OUTPUT_DIR = "open_drawer_output"


class EpisodeRecorder:
    """Buffers per-step data and saves in the TestOutput format."""

    def __init__(self, output_dir: str, env_id: int = 0):
        self.output_dir = output_dir
        self.env_id = env_id
        self.action_buffer = []
        self.collect_buffer = []
        self.obs_buffer = []
        self.camera_frames = {}  # {cam_id: {annotator_name: [frame_array, ...]}}

    def record_step(self, action, env, phase, wp_idx=None):
        """Record one step of data."""
        # Get observations
        policy_obs = env.get_policy_obs()
        privilege_obs = env.get_privilege_obs()

        # Get state (includes robot_state, scene_state, camera_state)
        state = env.get_state()

        # Build action entry
        action_list = action if isinstance(action, list) else action.tolist()
        raw_action = {
            "arm_action": action_list[:7],
            "eef_action": action_list[7:] if len(action_list) > 7 else [0.0],
        }
        action_entry = {
            "command": action_list,
            "robot_action": action_list,
            "raw_action": raw_action,
        }
        self.action_buffer.append(action_entry)

        # Build collect entry (planner/skill metadata)
        collect_entry = {
            "phase": phase,
        }
        if wp_idx is not None:
            collect_entry["waypoint_idx"] = wp_idx
        self.collect_buffer.append(collect_entry)

        # Build obs entry
        # Extract camera_info for video frames, store paths in obs
        camera_info = policy_obs.get("camera_info", {})
        self._buffer_camera_frames(camera_info)

        obs_entry = {
            "obs": {
                "policy_obs": {
                    "robot_state": to_serializable(policy_obs.get("robot_state")),
                    "camera_info": {},  # will be replaced with paths at save time
                },
                "privilege_obs": to_serializable(privilege_obs),
            },
            "reward": None,
            "terminated": None,
            "truncated": None,
            "info": {
                "state": to_serializable(state),
            },
        }
        self.obs_buffer.append(obs_entry)

    def _buffer_camera_frames(self, camera_info):
        """Buffer camera image frames for video encoding."""
        if not camera_info:
            return
        for cam_id, cam_data in enumerate(camera_info):
            if cam_id not in self.camera_frames:
                self.camera_frames[cam_id] = {}
            for annotator_name, env_list in cam_data.items():
                if not isinstance(env_list, list) or len(env_list) == 0:
                    continue
                anno_data = (
                    env_list[self.env_id]
                    if len(env_list) > self.env_id
                    else env_list[0]
                )
                if anno_data is None:
                    continue
                payload = convert_annotator_data_to_payload(annotator_name, anno_data)
                if is_image_annotator(annotator_name):
                    img = extract_image_data(annotator_name, payload)
                    if img is not None:
                        if annotator_name not in self.camera_frames[cam_id]:
                            self.camera_frames[cam_id][annotator_name] = []
                        self.camera_frames[cam_id][annotator_name].append(img)

    def save(self, trajectory_id: int = 0):
        """Save all buffered data to disk in the TestOutput format."""
        traj_dir = os.path.join(self.output_dir, str(trajectory_id))
        action_dir = os.path.join(traj_dir, "action")
        collect_dir = os.path.join(traj_dir, "collect")
        env_info_dir = os.path.join(traj_dir, "env", "info")
        camera_base_dir = os.path.join(traj_dir, "env", "camera")
        os.makedirs(action_dir, exist_ok=True)
        os.makedirs(collect_dir, exist_ok=True)
        os.makedirs(env_info_dir, exist_ok=True)
        os.makedirs(camera_base_dir, exist_ok=True)

        # Save action_merged.json
        action_merged = {
            f"frame_{i:04d}": to_serializable(a)
            for i, a in enumerate(self.action_buffer)
        }
        with open(os.path.join(action_dir, "action_merged.json"), "w") as f:
            json.dump(action_merged, f, indent=4)

        # Save collect_merged.json
        collect_merged = {
            f"frame_{i:04d}": to_serializable(c)
            for i, c in enumerate(self.collect_buffer)
        }
        with open(os.path.join(collect_dir, "collect_merged.json"), "w") as f:
            json.dump(collect_merged, f, indent=4)

        # Save camera videos and build camera_info paths
        camera_paths = {}
        for cam_id, annotators in self.camera_frames.items():
            cam_dir = os.path.join(camera_base_dir, f"cam_{cam_id}")
            for annotator_name, frames in annotators.items():
                if not frames:
                    continue
                anno_dir = os.path.join(cam_dir, annotator_name)
                os.makedirs(anno_dir, exist_ok=True)
                video_rel_path = os.path.join(
                    "env",
                    "camera",
                    f"cam_{cam_id}",
                    annotator_name,
                    f"{annotator_name}.mp4",
                )
                # Write video
                try:
                    import cv2

                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_abs_path = os.path.join(traj_dir, video_rel_path)
                    writer = cv2.VideoWriter(video_abs_path, fourcc, 30.0, (w, h))
                    for frame in frames:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(bgr)
                    writer.release()
                except ImportError:
                    # Fallback: save individual PNGs
                    from PIL import Image

                    for idx, frame in enumerate(frames):
                        Image.fromarray(frame).save(
                            os.path.join(anno_dir, f"step_{idx:04d}.png")
                        )
                    video_rel_path = os.path.join(
                        "env", "camera", f"cam_{cam_id}", annotator_name
                    )

                if annotator_name not in camera_paths:
                    camera_paths[annotator_name] = [None] * (cam_id + 1)
                while len(camera_paths[annotator_name]) <= cam_id:
                    camera_paths[annotator_name].append(None)
                camera_paths[annotator_name][cam_id] = video_rel_path

        # Inject camera_info paths into obs entries and save env_merged.json
        env_merged = {}
        for i, obs_entry in enumerate(self.obs_buffer):
            obs_entry["obs"]["policy_obs"]["camera_info"] = camera_paths
            env_merged[f"frame_{i:04d}"] = obs_entry
        with open(
            os.path.join(env_info_dir, f"env_{self.env_id}_merged.json"), "w"
        ) as f:
            json.dump(env_merged, f, indent=4)

        print(
            f"Saved trajectory {trajectory_id} ({len(self.action_buffer)} steps) to {traj_dir}"
        )


@hydra.main(version_base=None, config_path="../../Conf", config_name="open_drawer_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: OpenDrawerEnv = gym.make(
        "OpenDrawerEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    # Create output directory
    output_dir = os.path.join(os.getcwd(), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output data to: {output_dir}")

    recorder = EpisodeRecorder(output_dir, env_id=0)

    # Let the simulation settle
    for i in range(40):
        env.sim_step()

    # Get the articulation object (env 0, first articulation item)
    articulation_obj = env.scene.scene_manager.articulation_objects[0][
        "articulation_items"
    ][0]

    # Get trajectory poses (transformed to world coordinates) for all joints
    traj_data = articulation_obj.get_trajectory_poses(
        annotation_name="open_by_handle_trajectory",
        joint_name=None,
        transform_to_world=True,
    )
    print(f"Trajectory data keys: {list(traj_data.keys())}")

    # Support both "trajectories" and "grasp_trajectories" keys
    trajs = traj_data.get("trajectories") or traj_data.get("grasp_trajectories")

    # Visualize trajectories for all joints
    for joint_idx, joint_name in enumerate(["joint_0", "joint_1", "joint_2"]):
        if joint_name not in trajs:
            print(f"Warning: {joint_name} not in trajectories, skipping")
            continue
        joint_trajs = trajs[joint_name]
        first_traj_key = sorted(joint_trajs.keys())[0]
        first_traj = joint_trajs[first_traj_key]
        if not isinstance(first_traj, torch.Tensor):
            first_traj = torch.tensor(first_traj, dtype=torch.float32)
        print(
            f"{joint_name} trajectory key: {first_traj_key}, shape: {first_traj.shape}"
        )

        draw_grasp_samples_as_axes(
            grasp_poses=first_traj,
            axis_length=0.03,
            line_thickness=3,
            line_opacity=0.8,
            clear_existing=(joint_idx == 0),
        )

    # Extract the target joint trajectory for execution
    if TARGET_JOINT not in trajs:
        print(f"Error: {TARGET_JOINT} not in trajectories, cannot open drawer")
        while True:
            env.sim_step()
        return

    joint_trajs = trajs[TARGET_JOINT]
    first_traj_key = sorted(joint_trajs.keys())[0]
    trajectory = joint_trajs[first_traj_key]
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
    else:
        trajectory = trajectory.detach().clone().float()

    print(
        f"\nUsing {TARGET_JOINT} traj key: {first_traj_key}, shape: {trajectory.shape}"
    )
    print(f"  First waypoint (handle): {[f'{v:.4f}' for v in trajectory[0].tolist()]}")
    print(f"  Last waypoint (pulled):  {[f'{v:.4f}' for v in trajectory[-1].tolist()]}")

    # Handle pose = first waypoint of trajectory
    handle_pose = trajectory[0]
    action = handle_pose.cpu().numpy().tolist()

    # Print EEF pose for sanity check
    eef_pose = env.get_eef_pose(env_ids=[0])
    print(f"  EEF pose:    {[f'{v:.4f}' for v in eef_pose[0].tolist()]}")
    dist = torch.norm(eef_pose[0, :3] - handle_pose[:3].to(eef_pose.device)).item()
    print(f"  Distance EEF -> handle: {dist:.3f}m")

    # ---- Phase 1: Approach (pre-grasp) ----
    print("\n[OpenDrawer] Phase 1/5: Approach (pre-grasp, 100 steps)")
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
        pre_action_tensor = torch.tensor(pre_action, device=env.device).unsqueeze(0)
        env.step(action=pre_action_tensor)
        recorder.record_step(pre_action, env, "approach")

    # ---- Phase 2: Move to handle ----
    print("[OpenDrawer] Phase 2/5: Move to handle (60 steps)")
    for i in range(60):
        grasp_action = torch.tensor(action, device=env.device).unsqueeze(0)
        env.step(action=grasp_action)
        recorder.record_step(action, env, "move_to_handle")

    # ---- Phase 3: Close gripper ----
    print("[OpenDrawer] Phase 3/5: Close gripper (80 steps)")
    for i in range(80):
        close_action = action.copy()
        close_action.append(1)  # gripper close
        close_action_tensor = torch.tensor(close_action, device=env.device).unsqueeze(0)
        env.step(action=close_action_tensor)
        recorder.record_step(close_action, env, "close_gripper")

    # ---- Phase 4: Pull along trajectory ----
    num_waypoints = trajectory.shape[0]
    print(
        f"[OpenDrawer] Phase 4/5: Pull trajectory "
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
            recorder.record_step(wp_action_list, env, "pull", wp_idx=wp_idx)
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
    print("[OpenDrawer] Phase 5/5: Release gripper (20 steps)")
    last_wp = trajectory[-1].cpu().numpy().tolist()
    for i in range(20):
        release_action = last_wp.copy()
        release_action.append(0)  # gripper open
        release_action_tensor = torch.tensor(
            release_action, device=env.device
        ).unsqueeze(0)
        env.step(action=release_action_tensor)
        recorder.record_step(release_action, env, "release")

    # Save all buffered data
    recorder.save(trajectory_id=0)

    print("\n[OpenDrawer] Done! Simulation alive for inspection.")
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
