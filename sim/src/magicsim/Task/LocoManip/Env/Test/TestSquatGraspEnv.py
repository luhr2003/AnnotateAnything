"""
Test script for SquatGraspEnv: bottle on floor, robot squats to grasp.

Pipeline: env registration -> config (squat_grasp_env) -> scene (bottle on floor)
         -> termination (lift success / fall truncated) -> grasp execution
"""

from typing import List
import torch
from magicsim.Task.LocoManip.Env.SquatGraspEnv import SquatGraspEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from magicsim.Env.Utils.rotations import quat_to_rot_matrix

from pxr import Gf

# Visualization settings
AXIS_LENGTH = 0.05
LINE_THICKNESS = 3
LINE_OPACITY = 0.8


def visualize_grasp_pose(grasp_pose: List[torch.Tensor]):
    grasp_pose = [p.cpu().numpy().tolist() for p in grasp_pose]
    grasp_pose_list = [
        (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6]))
        for p in grasp_pose
    ]
    draw_grasp_samples_as_axes(
        grasp_poses=grasp_pose_list,
        axis_length=AXIS_LENGTH,
        line_thickness=LINE_THICKNESS,
        line_opacity=LINE_OPACITY,
        clear_existing=True,
    )


def _flatten_hand_grasp_dict(grasp_dict: dict) -> list:
    """Flatten functional_grasp/grasp dict to list of candidates."""
    if not grasp_dict or not isinstance(grasp_dict, dict):
        return []
    out = []
    for top_key in ("functional_grasp", "grasp"):
        parts = grasp_dict.get(top_key, {})
        if not isinstance(parts, dict):
            continue
        for part_list in parts.values():
            if isinstance(part_list, list):
                out.extend(part_list)
    return out


def compute_pose_along_grasp_direction(grasp_pose, offset_distance, backward=True):
    """Compute a pose by moving along the grasp direction."""
    device = grasp_pose.device
    if not isinstance(grasp_pose, torch.Tensor):
        grasp_pose = torch.tensor(grasp_pose, device=device, dtype=torch.float32)
    else:
        grasp_pose = grasp_pose.to(device=device)

    grasp_pos = grasp_pose[:3]
    grasp_quat = grasp_pose[3:7]
    rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))
    grasp_direction = rot_matrix[0, :, 1]
    grasp_direction_normalized = grasp_direction / torch.norm(grasp_direction)
    offset = grasp_direction_normalized * offset_distance

    if backward:
        new_pos = grasp_pos - offset
    else:
        new_pos = grasp_pos + offset

    return torch.cat([new_pos, grasp_quat], dim=0)


@hydra.main(version_base=None, config_path="../../Conf", config_name="squat_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SquatGraspEnv = gym.make(
        "SquatGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    device = env.device
    dtype = torch.float32
    p_controller = torch.full((15,), torch.nan, device=device, dtype=dtype)
    left_arm_ik = torch.tensor(
        [-0.6, 0.15, 0.85, 1, 0, 0, 0], device=device, dtype=dtype
    )
    left_hand_joints = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
    )
    right_arm_ik = torch.tensor(
        [-0.6, -0.15, 0.85, 1, 0, 0, 0], device=device, dtype=dtype
    )

    # Initial stand / approach
    for i in range(100):
        action = torch.cat(
            [
                p_controller,
                left_arm_ik,
                right_arm_ik,
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=0,
        )
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        env.step(action=action)

    stage_list = ["coarse_grasp", "fine_grasp", "final_grasp"]
    dexgrasp_list = env.get_grasp_pose(hand_type="dex3_1")
    env_dict = dexgrasp_list[0] if dexgrasp_list else None
    candidates = _flatten_hand_grasp_dict(env_dict) if env_dict else []
    if not candidates:
        log.error(
            "No dex3_1 grasp candidates. Ensure object has dex3_1_grasp_pose annotation."
        )
        return

    candidate = candidates[0]
    stage_data = {}
    for stage in stage_list:
        if stage not in candidate:
            continue
        joints = candidate[stage]["joints"]
        if isinstance(joints, torch.Tensor) and joints.numel() > 7:
            joints = joints.flatten()[:7]
        stage_data[stage] = {
            "right_pos": candidate[stage]["position"],
            "right_ori": candidate[stage]["orientation"],
            "right_hand_joints": joints,
        }

    coarse_stage_data = stage_data.get("coarse_grasp")
    if not coarse_stage_data:
        log.error("coarse_grasp stage not found in annotation.")
        return

    coarse_pos = coarse_stage_data["right_pos"]
    coarse_ori = coarse_stage_data["right_ori"]

    # Pregrasp: offset along grasp direction (squat down toward floor bottle)
    pregrasp_pos = compute_pose_along_grasp_direction(
        torch.cat([coarse_pos, coarse_ori], dim=0), 0.1
    )
    print("current stage: pregrasp (squat toward bottle on floor)")
    visualize_grasp_pose([pregrasp_pos])
    for i in range(300):
        right_arm_ik = pregrasp_pos
        action = torch.cat(
            [
                p_controller,
                right_arm_ik,
                left_arm_ik,
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=0,
        )
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        obs, reward, terminated, truncated, info, pending = env.step(action=action)
        if terminated.any() or truncated.any():
            log.info(f"terminated={terminated}, truncated={truncated}")
            break

    print("current stage: grasp")
    visualize_grasp_pose([torch.cat([coarse_pos, coarse_ori], dim=0)])
    for i in range(150):
        right_arm_ik = torch.cat([coarse_pos, coarse_ori], dim=0)
        action = torch.cat(
            [
                p_controller,
                right_arm_ik,
                left_arm_ik,
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=0,
        )
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        obs, reward, terminated, truncated, info, pending = env.step(action=action)
        if terminated.any() or truncated.any():
            log.info(f"terminated={terminated}, truncated={truncated}")
            break

    for stage in stage_list:
        if stage not in stage_data:
            continue
        print(f"current stage: {stage}")
        right_pos = stage_data[stage]["right_pos"]
        right_ori = stage_data[stage]["right_ori"]
        right_hand_joints = stage_data[stage]["right_hand_joints"]
        visualize_grasp_pose([torch.cat([right_pos, right_ori], dim=0)])

        for i in range(80):
            right_arm_ik = torch.cat([right_pos, right_ori], dim=0)
            action = torch.cat(
                [
                    p_controller,
                    right_arm_ik,
                    left_arm_ik,
                    left_hand_joints,
                    right_hand_joints,
                ],
                dim=0,
            )
            action = action.unsqueeze(0).repeat(env.num_envs, 1)
            obs, reward, terminated, truncated, info, pending = env.step(action=action)
            if terminated.any():
                log.info(f"Success: terminated at stage {stage}")
                break
            if truncated.any():
                log.warning(f"Truncated (object fell): truncated at stage {stage}")
                break

    final_stage_data = stage_data.get("final_grasp")
    if not final_stage_data:
        log.error("final_grasp stage not found in annotation.")
        return

    final_pos = final_stage_data["right_pos"]
    final_ori = final_stage_data["right_ori"]
    final_hand_joints = final_stage_data["right_hand_joints"]

    retrieval_pos = compute_pose_along_grasp_direction(
        torch.cat([final_pos, final_ori], dim=0), 0.2
    )
    print("current stage: retrieval (lift bottle from floor)")

    visualize_grasp_pose([retrieval_pos])
    for i in range(100):
        right_arm_ik = retrieval_pos
        action = torch.cat(
            [
                p_controller,
                right_arm_ik,
                left_arm_ik,
                left_hand_joints,
                final_hand_joints,
            ],
            dim=0,
        )
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        obs, reward, terminated, truncated, info, pending = env.step(action=action)
        if terminated.any():
            log.info("Success: object lifted (termination)")
            break
        if truncated.any():
            log.warning("Truncated: object fell through floor")
            break

    while True:
        env.step(action=action)


if __name__ == "__main__":
    main()
