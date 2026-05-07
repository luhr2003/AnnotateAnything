"""
TestBiDexGraspPose: visualize the paired sharpa grasp pose chosen by IK.

Counterpart to :file:`TestLocoGraspPose.py` for the bimanual dex case. The
robot stays parked (only ``sim_step`` is used so implicit actuators hold the
authored initial pose) — this script does NOT command motion. It just:

  1. loads the paired ``sharpa_dex_grasp_pose.json`` annotation through
     :meth:`BiDexGraspEnv.get_bimanual_grasp_pose`,
  2. packs the ``coarse_grasp`` poses as a ``(1, G, 14)`` goalset
     (``[right_7, left_7]`` per row) and submits it to the robot's paired
     dual IK,
  3. draws the right + left coarse_grasp axes for the selected paired
     candidate (single shared index that satisfies both arms).

When the bin moves (settling, etc.) the goalset is re-submitted at the new
object pose so the drawn axes track world-frame ground truth.
"""

from typing import List, Optional, Tuple

import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from loguru import logger as log

# Env class must come before any Planner / IsaacLab / omni-touching import,
# else the test driver dies with ``ModuleNotFoundError: No module named 'omni'``.
from magicsim.Task.Dexterous.Env.BiDexGraspEnv import BiDexGraspEnv
from magicsim.Env.Utils.file import Logger
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


def _flatten_pair_grasp_dict(grasp_dict: dict) -> list:
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


def _pair_pose7(side: dict) -> Optional[torch.Tensor]:
    coarse = side.get("coarse_grasp")
    if coarse is None:
        return None
    pos = _to_tensor(coarse["position"]).flatten()[:3]
    ori = _to_tensor(coarse["orientation"]).flatten()[:4]
    return torch.cat([pos, ori], dim=0)


def _extract_paired_coarse_goalset(candidates: list) -> Optional[torch.Tensor]:
    rows = []
    for c in candidates:
        if not (isinstance(c, dict) and "left_hand" in c and "right_hand" in c):
            continue
        right = _pair_pose7(c["right_hand"])
        left = _pair_pose7(c["left_hand"])
        if right is None or left is None:
            continue
        rows.append(torch.cat([right, left], dim=0))  # (14,)
    if not rows:
        return None
    return torch.stack(rows, dim=0)  # (G, 14)


def _resolve_robot_name(env: BiDexGraspEnv) -> str:
    robot_manager = getattr(env.scene, "robot_manager", None)
    if robot_manager is None:
        raise RuntimeError("robot_manager not available.")
    robot_dict = getattr(robot_manager, "robots", None)
    if isinstance(robot_dict, dict) and robot_dict:
        return next(iter(robot_dict.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_robot_state_dict(env: BiDexGraspEnv) -> dict:
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


def _get_ik_server(env: BiDexGraspEnv):
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None)
    if not ik_dict:
        return None
    return ik_dict.get(_resolve_robot_name(env))


def _ik_select_paired(
    env: BiDexGraspEnv, paired_goalset: torch.Tensor, env_id: int = 0
) -> Tuple[bool, int]:
    """Submit ``(1, G, 14)`` paired goalset; return ``(ok, selected_idx)``."""
    ik_server = _get_ik_server(env)
    if ik_server is None:
        log.warning("No IK server resolvable; using paired candidate 0.")
        return False, 0
    rs = _get_robot_state_dict(env)
    target = paired_goalset.unsqueeze(0).to(env.device).contiguous()
    n_goals = int(target.shape[1])
    is_dual = bool(getattr(ik_server, "dual_mode", False))
    log.info(
        "[bi-ik goalset] submit env_ids={} dual_ik={} n_goals={} target.shape={}",
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
        log.warning("[bi-ik goalset] result exception: {}", ex)
        return False, 0
    log.info(
        "[bi-ik goalset] result success_list={} goalset_index_list={} returned_env_ids={}",
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


def _get_target_object_pose_7d(env: BiDexGraspEnv) -> Optional[torch.Tensor]:
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
    env: BiDexGraspEnv,
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


def apply_goalset_and_visualize(env: BiDexGraspEnv) -> bool:
    grasp_list = env.get_bimanual_grasp_pose(hand_type="sharpa")
    env_dict = grasp_list[0] if grasp_list else None
    candidates = _flatten_pair_grasp_dict(env_dict) if env_dict else []
    if not candidates:
        log.error("No paired sharpa grasp candidates.")
        return False
    paired_goalset = _extract_paired_coarse_goalset(candidates)
    if paired_goalset is None or paired_goalset.shape[0] == 0:
        log.error("No paired coarse_grasp goalset rows.")
        return False
    paired_goalset = paired_goalset.to(env.device)
    ok, idx = _ik_select_paired(env, paired_goalset)
    if not ok or idx < 0 or idx >= len(candidates):
        log.warning("Paired IK goalset failed; using paired candidate 0.")
        idx = 0
    chosen = candidates[idx]
    r_pose = _pair_pose7(chosen["right_hand"]).to(env.device)
    l_pose = _pair_pose7(chosen["left_hand"]).to(env.device)
    log.info(
        "[bi-ik goalset] selected idx={}/{} R=({:+.3f},{:+.3f},{:+.3f}) "
        "L=({:+.3f},{:+.3f},{:+.3f})",
        idx,
        len(candidates),
        float(r_pose[0]),
        float(r_pose[1]),
        float(r_pose[2]),
        float(l_pose[0]),
        float(l_pose[1]),
        float(l_pose[2]),
    )
    visualize_grasp_pose([r_pose, l_pose])
    return True


@hydra.main(version_base=None, config_path="../../Conf", config_name="bi_dex_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: BiDexGraspEnv = gym.make(
        "BiDexGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # Let physics settle (no env.step → robot held by implicit actuators at
    # the authored initial pose, so we never need to construct a neutral
    # action that matches the vega1pSharpa action layout).
    for _ in range(50):
        env.sim_step()

    env.scene.planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        obstacle_ignore_path_list=["bin"],
        env_ids=[0],
    )

    obj_ref: Optional[torch.Tensor] = None
    obj_prev: Optional[torch.Tensor] = None
    while True:
        env.sim_step()
        obj_now = _get_target_object_pose_7d(env)
        if obj_now is None:
            continue

        if obj_ref is None:
            obj_ref = obj_now.clone()
            obj_prev = obj_now.clone()
            apply_goalset_and_visualize(env)
            continue

        if should_regenerate_grasp(env, obj_ref, obj_prev):
            if apply_goalset_and_visualize(env):
                obj_ref = obj_now.clone()

        obj_prev = obj_now.clone()


if __name__ == "__main__":
    main()
