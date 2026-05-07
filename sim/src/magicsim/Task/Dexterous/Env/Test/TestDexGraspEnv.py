"""
Test DexGraspEnv: direct env test (no AutoCollect).
Uses Dexterous dex_grasp scene with single_franka_xhand.
Adapted from TestLocoGraspEnv for XHand (single arm + 12-DOF hand).
"""

from typing import List
import torch
from magicsim.Task.Dexterous.Env.DexGraspEnv import DexGraspEnv
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
    """Flatten functional_grasp/grasp dict to list of candidates (each has coarse/fine/final)."""
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
    """
    Compute a pose by moving along the grasp direction.

    Args:
        grasp_pose: torch.Tensor of 7 elements [x, y, z, qw, qx, qy, qz]
        offset_distance: Distance to move along grasp direction
        backward: If True, move backward (subtract); if False, move forward (add)

    Returns:
        torch.Tensor: New pose [x, y, z, qw, qx, qy, qz]
    """
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


@hydra.main(version_base=None, config_path="../../Conf", config_name="dex_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: DexGraspEnv = gym.make(
        "DexGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    device = env.device
    dtype = torch.float32

    hand_joints_open = torch.zeros(12, device=device, dtype=dtype)

    for i in range(50):
        env.sim_step()

    # Get xhand grasp poses (dict per env: functional_grasp/grasp with parts)
    grasp_list = env.get_grasp_pose(hand_type="xhand")
    if not grasp_list or grasp_list[0] is None:
        log.error(
            "No xhand grasp poses found. Ensure object has xhand_grasp_pose annotation."
        )
        return

    candidates = _flatten_hand_grasp_dict(grasp_list[0])
    if not candidates:
        log.error("No xhand grasp candidates in annotation.")
        return

    # Use first candidate for env 0
    candidate = candidates[0]
    stage_list = ["coarse_grasp", "fine_grasp", "final_grasp"]
    stage_data = {}
    for stage in stage_list:
        if stage not in candidate:
            continue
        stage_data[stage] = {
            "pos": candidate[stage]["position"],
            "ori": candidate[stage]["orientation"],
            "joints": candidate[stage]["joints"],
        }

    # Pregrasp: move to coarse_grasp position, offset along grasp direction
    coarse_stage_data = stage_data["coarse_grasp"]
    coarse_pos = coarse_stage_data["pos"]
    coarse_ori = coarse_stage_data["ori"]

    pregrasp_pos = compute_pose_along_grasp_direction(
        torch.cat([coarse_pos, coarse_ori], dim=0), 0.1
    )
    print("current stage: pregrasp")
    visualize_grasp_pose([pregrasp_pos])
    for i in range(100):
        arm_ik = pregrasp_pos
        action = torch.cat([arm_ik, hand_joints_open], dim=0)
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        env.step(action=action)

    # Approach grasp position
    print("current stage: grasp")
    visualize_grasp_pose([torch.cat([coarse_pos, coarse_ori], dim=0)])
    for i in range(100):
        arm_ik = torch.cat([coarse_pos, coarse_ori], dim=0)
        action = torch.cat([arm_ik, hand_joints_open], dim=0)
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        env.step(action=action)

    # Execute grasp stages: coarse_grasp, fine_grasp, final_grasp
    for stage in stage_list:
        if stage not in stage_data:
            continue
        print(f"current stage: {stage}")
        pos = stage_data[stage]["pos"]
        ori = stage_data[stage]["ori"]
        hand_joints = stage_data[stage]["joints"]
        visualize_grasp_pose([torch.cat([pos, ori], dim=0)])

        for i in range(50):
            arm_ik = torch.cat([pos, ori], dim=0)
            action = torch.cat([arm_ik, hand_joints], dim=0)
            action = action.unsqueeze(0).repeat(env.num_envs, 1)
            env.step(action=action)

    # Retrieval: lift object
    final_stage_data = stage_data["final_grasp"]
    final_pos = final_stage_data["pos"]
    final_ori = final_stage_data["ori"]
    final_hand_joints = final_stage_data["joints"]

    retrieval_pos = final_pos.clone()
    retrieval_pos[2] += 0.2
    retrieval_pos = torch.cat([retrieval_pos, final_ori], dim=0)
    print("current stage: retrieval")

    visualize_grasp_pose([retrieval_pos])
    for i in range(100):
        arm_ik = retrieval_pos
        action = torch.cat([arm_ik, final_hand_joints], dim=0)
        action = action.unsqueeze(0).repeat(env.num_envs, 1)
        env.step(action=action)

    while True:
        env.step(action=action)


if __name__ == "__main__":
    main()
