"""
MobileGraspEnv: sim + monitor target object pose + IK goalset + visualize grasp poses.

Same logic as TestLocoGraspPose but adapted for ridgebackFranka (parallel gripper).
Records object_ref on first observation, runs IK goalset + viz once.
Re-runs goalset when |p-p_ref|>1e-3 and |p-p_prev|<1e-4 (object moved then stabilised).
"""

from typing import List, Optional, Tuple
import torch
from magicsim.Task.MobileManip.Env.MobileGraspEnv import MobileGraspEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest

from pxr import Gf

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


def _resolve_robot_name(env: MobileGraspEnv) -> str:
    robot_manager = getattr(env.scene, "robot_manager", None)
    if robot_manager is None:
        raise RuntimeError("robot_manager not available.")
    robot_dict = getattr(robot_manager, "robots", None)
    if isinstance(robot_dict, dict) and robot_dict:
        return next(iter(robot_dict.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_robot_state_dict(env: MobileGraspEnv) -> dict:
    robot_states = env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
    if isinstance(robot_states, dict):
        name = _resolve_robot_name(env)
        robot_state = robot_states.get(name, next(iter(robot_states.values())))
    else:
        robot_state = robot_states
    return {
        "base_pos": robot_state["base_pos"],
        "base_quat": robot_state["base_quat"],
        "joint_pos": robot_state["joint_pos"],
        "joint_vel": robot_state["joint_vel"],
    }


def _get_ik_server(env: MobileGraspEnv):
    """Get IK server for the single arm (hand_id=0)."""
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None)
    if not ik_dict:
        return None
    robot_name = _resolve_robot_name(env)
    per_robot = ik_dict.get(robot_name)
    if not isinstance(per_robot, dict) or 0 not in per_robot:
        return None
    return per_robot[0]


def _ik_select_best(
    env: MobileGraspEnv, poses: torch.Tensor, env_id: int = 0
) -> Tuple[bool, int]:
    """Submit goalset to IK and return (success, selected_index)."""
    ik_server = _get_ik_server(env)
    if ik_server is None:
        log.warning("No IK server; using candidate index 0.")
        return False, 0
    rs = _get_robot_state_dict(env)
    target = poses.unsqueeze(0).to(env.device)  # [1, G, 7]
    n_goals = int(target.shape[1])
    is_dual = getattr(ik_server, "dual_mode", False)
    log.info(
        "[ik goalset] submit env_ids={} dual_ik={} n_goals={} target.shape={}",
        [env_id],
        is_dual,
        n_goals,
        tuple(target.shape),
    )
    if is_dual:
        req = DualIKPlanRequest(
            env_ids=[env_id],
            target_pos=target,
            robot_states=rs,
            mode="goalset",
            lock_base=False,
        )
        print("req: ", req)
    else:
        req = IKPlanRequest(
            env_ids=[env_id],
            target_pos=target,
            robot_states=rs,
            mode="goalset",
        )
    fut = ik_server.submit_ik(req)
    try:
        success_list, goalset_index_list, returned_env_ids = fut.result(timeout=120.0)
    except Exception as ex:
        log.warning("[ik goalset] result exception: {}", ex)
        return False, 0
    log.info(
        "[ik goalset] result success_list={} goalset_index_list={} returned_env_ids={}",
        success_list,
        goalset_index_list,
        returned_env_ids,
    )
    if not returned_env_ids or int(returned_env_ids[0]) != int(env_id):
        return False, 0
    selected_idx = -1
    if goalset_index_list is not None and len(goalset_index_list) >= 1:
        selected_idx = int(goalset_index_list[0])
    ok = len(success_list) >= 1 and bool(success_list[0])
    if not ok or selected_idx < 0:
        return False, 0
    return True, selected_idx


def _combine_poses_from_dict(parts_dict: dict) -> Optional[torch.Tensor]:
    """Combine [N, 7] pose tensors from all parts in a dict."""
    if not parts_dict or not isinstance(parts_dict, dict):
        return None
    tensor_list = []
    for poses in parts_dict.values():
        if poses is not None and isinstance(poses, torch.Tensor) and poses.numel() > 0:
            if poses.ndim == 1:
                poses = poses.unsqueeze(0)
            if poses.shape[-1] == 7:
                tensor_list.append(poses)
    if not tensor_list:
        return None
    return torch.cat(tensor_list, dim=0)


def _get_target_object_pose_7d(env: MobileGraspEnv) -> Optional[torch.Tensor]:
    """Target object pose [7] in world frame."""
    obj_name = getattr(env, "target_obj_name", None)
    poses = env.get_object_pose()
    if obj_name is None or obj_name not in poses:
        for k in poses:
            if k != "simple_desk":
                obj_name = k
                break
    if obj_name is None or obj_name not in poses:
        return None
    return poses[obj_name][0].to(env.device)


def _object_moved_from_reference(
    obj_now: torch.Tensor, obj_ref: torch.Tensor, pos_eps: float = 1e-3
) -> bool:
    if obj_now is None or obj_ref is None:
        return False
    return bool(torch.norm(obj_now[:3] - obj_ref[:3]) > pos_eps)


def _object_stable_vs_prev(
    obj_now: torch.Tensor, obj_prev: Optional[torch.Tensor], pos_eps: float = 1e-4
) -> bool:
    if obj_now is None or obj_prev is None:
        return False
    return bool(torch.norm(obj_now[:3] - obj_prev[:3]) < pos_eps)


def should_regenerate_grasp(
    env: MobileGraspEnv,
    obj_ref: Optional[torch.Tensor],
    obj_prev: Optional[torch.Tensor],
) -> bool:
    obj_now = _get_target_object_pose_7d(env)
    if obj_now is None or obj_ref is None:
        return False
    if not _object_moved_from_reference(obj_now, obj_ref):
        return False
    if not _object_stable_vs_prev(obj_now, obj_prev):
        return False
    return True


def apply_goalset_and_visualize(env: MobileGraspEnv, device: torch.device) -> bool:
    """Get grasp poses (parallel gripper), run IK goalset, visualize best."""
    grasp_list = env.get_grasp_pose(env_ids=[0])
    env_dict = grasp_list[0] if grasp_list else None
    if env_dict is None:
        log.error("No grasp annotations found.")
        return False

    # Collect all pose tensors from functional_grasp / grasp
    functional_dict = env_dict.get("functional_grasp", {})
    grasp_dict = env_dict.get("grasp", {})
    all_poses = _combine_poses_from_dict(functional_dict)
    if all_poses is None:
        all_poses = _combine_poses_from_dict(grasp_dict)
    if all_poses is None:
        # Try combining both
        merged = {}
        if functional_dict:
            merged.update(functional_dict)
        if grasp_dict:
            merged.update(grasp_dict)
        all_poses = _combine_poses_from_dict(merged)
    if all_poses is None or all_poses.shape[0] == 0:
        log.error("No grasp poses available.")
        return False

    all_poses = all_poses.to(device)
    ok, idx = _ik_select_best(env, all_poses)
    if not ok or idx < 0 or idx >= all_poses.shape[0]:
        log.warning("IK goalset failed; using candidate 0.")
        idx = 0

    selected_pose = all_poses[idx]
    log.info("[viz] selected grasp pose idx={} pose={}", idx, selected_pose.tolist())
    visualize_grasp_pose([selected_pose])
    return True


@hydra.main(version_base=None, config_path="../../Conf", config_name="mobile_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: MobileGraspEnv = gym.make(
        "MobileGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    device = env.device
    dtype = torch.float32

    # RidgebackFranka planner dims: base p_controller(8) + arm ik_pink(7) + eef binary(1) = 16
    # Use NaN for base p_controller (no actuation), current EEF for arm (hold), 0 for gripper (open)
    total_action_dim = env.scene.planner_manager.total_action_dim
    eef_pose = env.get_eef_pose()[0]  # [7]
    base_dim = total_action_dim - 7 - 1  # base = total - arm(7) - eef(1)
    p_controller_nan = torch.full((base_dim,), torch.nan, device=device, dtype=dtype)
    gripper_open = torch.zeros(1, device=device, dtype=dtype)
    neutral_action = (
        torch.cat([p_controller_nan, eef_pose, gripper_open], dim=0)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )

    obj_ref: Optional[torch.Tensor] = None
    obj_prev: Optional[torch.Tensor] = None

    # Let physics settle
    for _ in range(50):
        env.step(action=neutral_action)

    # Update obstacles (ignore target object for IK planning)
    obj_name = getattr(env, "target_obj_name", "apple")
    print("obj_name: ", obj_name)
    env.scene.planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        obstacle_ignore_path_list=[obj_name],
        env_ids=[0],
    )

    while True:
        env.step(action=neutral_action)
        obj_now = _get_target_object_pose_7d(env)
        if obj_now is None:
            continue

        if obj_ref is None:
            obj_ref = obj_now.clone()
            obj_prev = obj_now.clone()
            apply_goalset_and_visualize(env, device)
            continue

        if should_regenerate_grasp(env, obj_ref, obj_prev):
            if apply_goalset_and_visualize(env, device):
                obj_ref = obj_now.clone()

        obj_prev = obj_now.clone()


if __name__ == "__main__":
    main()
