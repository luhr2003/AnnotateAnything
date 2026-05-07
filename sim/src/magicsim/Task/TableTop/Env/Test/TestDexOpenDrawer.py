"""Open-loop dex open-drawer with episode recording.

Mirrors :mod:`TestOpenDrawer` (which records parallel-gripper open-drawer
demonstrations) but operates on the 19-D Franka + Xhand action space and
loads the ``xhand_open_by_handle_trajectory`` annotation from Drawer/7120
(source: ``~/sharpa_bin+xhand_open_by_handle/xhand_open_by_handle``).
"""

# Env class import must come first so omni gets registered before any
# Planner / IsaacLab imports.
from magicsim.Task.TableTop.Env.DexOpenDrawerEnv import DexOpenDrawerEnv

import os
import json
import torch
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from loguru import logger as log

from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from magicsim.Env.Utils.file import Logger
from magicsim.Collect.Record.utils import (
    to_serializable,
    is_image_annotator,
    extract_image_data,
    convert_annotator_data_to_payload,
)


EEF_POS_THRESHOLD = 0.03
EEF_ROT_THRESHOLD = 0.1
WAYPOINT_MAX_STEPS = 20

ARM_POSE_DIM = 7
FINGER_JOINT_DIM = 12
DEX_ACTION_DIM = ARM_POSE_DIM + FINGER_JOINT_DIM

TARGET_JOINT_ID = 1
OUTPUT_DIR = "dex_open_drawer_output"


class EpisodeRecorder:
    """Buffers per-step data and saves in the TestOutput format."""

    def __init__(self, output_dir: str, env_id: int = 0):
        self.output_dir = output_dir
        self.env_id = env_id
        self.action_buffer = []
        self.collect_buffer = []
        self.obs_buffer = []
        self.camera_frames = {}

    def record_step(self, action, env, phase, wp_idx=None):
        policy_obs = env.get_policy_obs()
        privilege_obs = env.get_privilege_obs()
        state = env.get_state()

        action_list = action if isinstance(action, list) else action.tolist()
        raw_action = {
            "arm_action": action_list[:ARM_POSE_DIM],
            "eef_action": action_list[ARM_POSE_DIM:DEX_ACTION_DIM]
            if len(action_list) > ARM_POSE_DIM
            else [0.0] * FINGER_JOINT_DIM,
        }
        self.action_buffer.append(
            {
                "command": action_list,
                "robot_action": action_list,
                "raw_action": raw_action,
            }
        )

        collect_entry = {"phase": phase}
        if wp_idx is not None:
            collect_entry["waypoint_idx"] = wp_idx
        self.collect_buffer.append(collect_entry)

        camera_info = policy_obs.get("camera_info", {})
        self._buffer_camera_frames(camera_info)

        self.obs_buffer.append(
            {
                "obs": {
                    "policy_obs": {
                        "robot_state": to_serializable(policy_obs.get("robot_state")),
                        "camera_info": {},
                    },
                    "privilege_obs": to_serializable(privilege_obs),
                },
                "reward": None,
                "terminated": None,
                "truncated": None,
                "info": {"state": to_serializable(state)},
            }
        )

    def _buffer_camera_frames(self, camera_info):
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
                        self.camera_frames[cam_id].setdefault(
                            annotator_name, []
                        ).append(img)

    def save(self, trajectory_id: int = 0):
        traj_dir = os.path.join(self.output_dir, str(trajectory_id))
        action_dir = os.path.join(traj_dir, "action")
        collect_dir = os.path.join(traj_dir, "collect")
        env_info_dir = os.path.join(traj_dir, "env", "info")
        camera_base_dir = os.path.join(traj_dir, "env", "camera")
        os.makedirs(action_dir, exist_ok=True)
        os.makedirs(collect_dir, exist_ok=True)
        os.makedirs(env_info_dir, exist_ok=True)
        os.makedirs(camera_base_dir, exist_ok=True)

        action_merged = {
            f"frame_{i:04d}": to_serializable(a)
            for i, a in enumerate(self.action_buffer)
        }
        with open(os.path.join(action_dir, "action_merged.json"), "w") as f:
            json.dump(action_merged, f, indent=4)

        collect_merged = {
            f"frame_{i:04d}": to_serializable(c)
            for i, c in enumerate(self.collect_buffer)
        }
        with open(os.path.join(collect_dir, "collect_merged.json"), "w") as f:
            json.dump(collect_merged, f, indent=4)

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
                    from PIL import Image

                    for idx, frame in enumerate(frames):
                        Image.fromarray(frame).save(
                            os.path.join(anno_dir, f"step_{idx:04d}.png")
                        )
                    video_rel_path = os.path.join(
                        "env", "camera", f"cam_{cam_id}", annotator_name
                    )

                camera_paths.setdefault(annotator_name, [None] * (cam_id + 1))
                while len(camera_paths[annotator_name]) <= cam_id:
                    camera_paths[annotator_name].append(None)
                camera_paths[annotator_name][cam_id] = video_rel_path

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


def _split_waypoint(waypoint: torch.Tensor) -> tuple[list[float], list[float]]:
    pose = waypoint[:ARM_POSE_DIM].cpu().tolist()
    joints = waypoint[ARM_POSE_DIM:DEX_ACTION_DIM].cpu().tolist()
    return pose, joints


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

    output_dir = os.path.join(os.getcwd(), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output data to: {output_dir}")
    recorder = EpisodeRecorder(output_dir, env_id=0)

    for _ in range(40):
        env.sim_step()

    trajectories = env.get_drawer_trajectories(
        env_id=0,
        annotation_name="xhand_open_by_handle_trajectory",
        joint_id=TARGET_JOINT_ID,
    )
    if not trajectories:
        print("[TestDexOpenDrawer] No trajectories found, idling.")
        while True:
            env.sim_step()

    selected_key = sorted(trajectories.keys())[0]
    trajectory = trajectories[selected_key].to(env.device)
    print(f"Using {selected_key}, shape: {trajectory.shape}")

    draw_grasp_samples_as_axes(
        grasp_poses=trajectory[:, :ARM_POSE_DIM],
        axis_length=0.03,
        line_thickness=3,
        line_opacity=0.8,
        clear_existing=True,
    )

    handle_pose, handle_joints = _split_waypoint(trajectory[0])
    last_pose, _ = _split_waypoint(trajectory[-1])
    open_hand = [0.0] * FINGER_JOINT_DIM

    eef_pose = env.get_eef_pose(env_ids=[0])
    handle_pos_t = torch.tensor(handle_pose[:3], device=eef_pose.device)
    print(
        f"  Distance EEF -> handle: "
        f"{torch.norm(eef_pose[0, :3] - handle_pos_t).item():.3f}m"
    )

    # ---- Phase 1: Approach ----
    print("\n[TestDexOpenDrawer] Phase 1/5: Approach (pre-grasp, 100 steps)")
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
        recorder.record_step(pre_action_list, env, "approach")

    # ---- Phase 2: Move to handle ----
    print("[TestDexOpenDrawer] Phase 2/5: Move to handle (60 steps)")
    move_action_list = handle_pose + open_hand
    move_action = torch.tensor(move_action_list, device=env.device).unsqueeze(0)
    for _ in range(60):
        env.step(action=move_action)
        recorder.record_step(move_action_list, env, "move_to_handle")

    # ---- Phase 3: Close fingers ----
    print("[TestDexOpenDrawer] Phase 3/5: Close fingers (80 steps)")
    close_action_list = handle_pose + handle_joints
    close_action = torch.tensor(close_action_list, device=env.device).unsqueeze(0)
    for _ in range(80):
        env.step(action=close_action)
        recorder.record_step(close_action_list, env, "close_gripper")

    # ---- Phase 4: Pull along trajectory ----
    num_waypoints = trajectory.shape[0]
    print(
        f"[TestDexOpenDrawer] Phase 4/5: Pull "
        f"({num_waypoints} wps, max {WAYPOINT_MAX_STEPS} steps/wp)"
    )
    total_pull_steps = 0
    pos_diff = quat_diff = float("inf")
    step = 0
    for wp_idx in range(num_waypoints):
        waypoint = trajectory[wp_idx]
        wp_action_list = waypoint.cpu().tolist()
        wp_action = waypoint.unsqueeze(0).to(env.device)

        arrived = False
        for step in range(WAYPOINT_MAX_STEPS):
            env.step(action=wp_action)
            recorder.record_step(wp_action_list, env, "pull", wp_idx=wp_idx)
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
                f"  wp {wp_idx}/{num_waypoints}: arrived={arrived}, "
                f"steps={step + 1}, pos_diff={pos_diff:.4f}, "
                f"rot_diff={quat_diff:.4f}"
            )
    print(f"  Pull complete: {total_pull_steps} total steps")

    # ---- Phase 5: Release ----
    print("[TestDexOpenDrawer] Phase 5/5: Release fingers (20 steps)")
    release_action_list = last_pose + open_hand
    release_action = torch.tensor(release_action_list, device=env.device).unsqueeze(0)
    for _ in range(20):
        env.step(action=release_action)
        recorder.record_step(release_action_list, env, "release")

    recorder.save(trajectory_id=0)
    print("\n[TestDexOpenDrawer] Done! Simulation alive for inspection.")
    while True:
        env.sim_step()


if __name__ == "__main__":
    main()
