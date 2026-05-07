"""
LocoGraspEnv：仅仿真 + 监测目标物体位姿（与 TestSquatGraspPose 同逻辑，环境为桌面 LocoGrasp）。

首次记录 object_ref 后会立刻做一次 IK goalset + 可视化。
之后仅当 |p−p_ref|>1e-3 且 |p−p_prev|<1e-4 时再次 goalset。
"""

from typing import List, Optional, Tuple
import torch
from magicsim.Task.LocoManip.Env.LocoGraspEnv import LocoGraspEnv
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


def _to_tensor(x, dtype=torch.float32, device=None):
    if isinstance(x, torch.Tensor):
        out = x.detach().clone().to(dtype=dtype)
        return out.to(device) if device is not None else out
    return torch.tensor(x, dtype=dtype, device=device)


def _extract_coarse_poses_from_candidates(candidates: list) -> Optional[torch.Tensor]:
    poses = []
    for c in candidates:
        coarse = c.get("coarse_grasp")
        if coarse is None:
            continue
        pos = _to_tensor(coarse["position"])
        ori = _to_tensor(coarse["orientation"])
        poses.append(torch.cat([pos.flatten()[:3], ori.flatten()[:4]], dim=0))
    if not poses:
        return None
    return torch.stack(poses, dim=0)


def _resolve_robot_name(env: LocoGraspEnv) -> str:
    robot_manager = getattr(env.scene, "robot_manager", None)
    if robot_manager is None:
        raise RuntimeError("robot_manager not available.")
    robot_dict = getattr(robot_manager, "robots", None)
    if isinstance(robot_dict, dict) and robot_dict:
        return next(iter(robot_dict.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_robot_state_dict(env: LocoGraspEnv) -> dict:
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


def _get_ik_server_right_hand(env: LocoGraspEnv):
    """Resolve the robot's IK server.

    Post-MERGE_LEFT_RIGHT §1–§8: ``planner_manager.ik_server`` is a flat
    ``{robot_name: IKServer}`` mapping (one server per robot). The old
    per-hand nested dict (``ik_server[robot_name][hand_id]``) is gone.
    """
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None)
    if not ik_dict:
        return None
    robot_name = _resolve_robot_name(env)
    return ik_dict.get(robot_name)


def _ik_select_best_coarse(
    env: LocoGraspEnv, coarse_poses: torch.Tensor, env_id: int = 0
) -> Tuple[bool, int]:
    ik_server = _get_ik_server_right_hand(env)
    if ik_server is None:
        log.warning("No IK server for right hand; using candidate index 0.")
        return False, 0
    rs = _get_robot_state_dict(env)
    target = coarse_poses.unsqueeze(0).to(env.device)
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


def _get_target_object_pose_7d(env: LocoGraspEnv) -> Optional[torch.Tensor]:
    """Grasp target object pose [7] in world frame (same pattern as squat test)."""
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
    env: LocoGraspEnv,
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


def apply_goalset_and_visualize(env: LocoGraspEnv, device: torch.device) -> bool:
    dexgrasp_list = env.get_grasp_pose(hand_type="dex3_1")
    env_dict = dexgrasp_list[0] if dexgrasp_list else None
    candidates = env._extract_grasp_candidates(env_dict, True, None) if env_dict else []
    if not candidates:
        log.error("No dex3_1 grasp candidates.")
        return False
    coarse_poses = _extract_coarse_poses_from_candidates(candidates)
    if coarse_poses is None or coarse_poses.shape[0] == 0:
        log.error("No coarse_grasp poses in candidates.")
        return False
    coarse_poses = coarse_poses.to(device)
    ok, idx = _ik_select_best_coarse(env, coarse_poses)
    if not ok or idx < 0 or idx >= len(candidates):
        log.warning("IK goalset failed; using candidate 0.")
        idx = 0
    coarse = candidates[idx]["coarse_grasp"]
    pos = coarse["position"]
    ori = coarse["orientation"]
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, device=device, dtype=torch.float32)
    if not isinstance(ori, torch.Tensor):
        ori = torch.tensor(ori, device=device, dtype=torch.float32)
    pose7 = torch.cat([pos.flatten()[:3], ori.flatten()[:4]], dim=0)
    visualize_grasp_pose([pose7])
    return True


@hydra.main(version_base=None, config_path="../../Conf", config_name="loco_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: LocoGraspEnv = gym.make(
        "LocoGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    device = env.device
    dtype = torch.float32
    p_controller = torch.full((15,), torch.nan, device=device, dtype=dtype)
    left_arm_ik = torch.tensor(
        [-0.6, 0.15, 0.85, 1, 0, 0, 0], device=device, dtype=dtype
    )
    right_arm_ik = torch.tensor(
        [-0.6, -0.15, 0.85, 1, 0, 0, 0], device=device, dtype=dtype
    )
    open_hand = torch.zeros(14, device=device, dtype=dtype)
    neutral_action = (
        torch.cat([p_controller, right_arm_ik, left_arm_ik, open_hand], dim=0)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )

    obj_ref: Optional[torch.Tensor] = None
    obj_prev: Optional[torch.Tensor] = None
    for _ in range(50):
        env.step(action=neutral_action)

    env.scene.planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        obstacle_ignore_path_list=["bottle"],
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
