"""
Test BiDexGraspEnv: openloop direct env test (no AutoCollect / atomic skill).

Selects a paired Sharpa grasp candidate by submitting the
``coarse_grasp`` poses as a paired ``(1, G, 14)`` goalset to the robot's
dual IK (one shared index ``g`` must satisfy both arms), then steps both
arms + both hands through pre / coarse / fine / final / retrieval phases
on that selected pair. Mirrors :file:`TestDexGraspEnv.py` for paired
bimanual annotations.

Robot: ``single_vega1pSharpa_parked`` (parked dual-arm Vega + Sharpa hands;
14-DOF arms + 2 × 22-DOF Sharpa fingers + 3-DOF holonomic base).
"""

from typing import List, Optional, Tuple

# Env class must precede any Planner / IsaacLab / omni-touching import so
# the SimulationApp boots before pxr.PhysxSchema is loaded.
from magicsim.Task.Dexterous.Env.BiDexGraspEnv import BiDexGraspEnv

import torch
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

# Per-hand finger DOF count for Sharpa (matches
# ``sharpa_dex_grasp_pose.json`` joint arrays).
HAND_JOINT_DIM = 22


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


def _flatten_pair_grasp_dict(grasp_dict: dict) -> list:
    """Flatten functional_grasp/grasp into a list of paired candidates."""
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


def compute_pair_pregrasp(
    r_pose: torch.Tensor, l_pose: torch.Tensor, offset_distance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair pregrasp = each hand moves AWAY from its partner along the
    inter-hand line (xyz only, orientation untouched).

    Sharpa annotations don't encode a clean per-hand outward axis in the
    local quat (both R/L map their local +Y to the same bin-y direction),
    so we derive the back-off direction geometrically from the pair's xyz.
    In the parked-vega scene with bin yaw=90° this projects onto world Y,
    matching the user's "y 轴" intuition.
    """
    r_pos, r_quat = r_pose[:3], r_pose[3:7]
    l_pos, l_quat = l_pose[:3], l_pose[3:7]
    inter = r_pos - l_pos
    norm = torch.norm(inter)
    if float(norm) < 1e-6:
        return r_pose.clone(), l_pose.clone()
    unit = inter / norm
    pre_r = torch.cat([r_pos + offset_distance * unit, r_quat], dim=0)
    pre_l = torch.cat([l_pos - offset_distance * unit, l_quat], dim=0)
    return pre_r, pre_l


def _phase_pose7(phase: dict) -> torch.Tensor:
    pos = phase["position"]
    ori = phase["orientation"]
    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos, dtype=torch.float32)
    if not isinstance(ori, torch.Tensor):
        ori = torch.as_tensor(ori, dtype=torch.float32)
    return torch.cat([pos.flatten()[:3], ori.flatten()[:4]], dim=0)


def _phase_joints(phase: dict, dim: int, device, dtype) -> torch.Tensor:
    j = phase.get("joints")
    if j is None:
        return torch.zeros(dim, dtype=dtype, device=device)
    if not isinstance(j, torch.Tensor):
        j = torch.as_tensor(j, dtype=dtype, device=device)
    j = j.flatten().to(device=device, dtype=dtype)
    if j.numel() >= dim:
        return j[:dim].contiguous()
    pad = torch.zeros(dim - j.numel(), dtype=dtype, device=device)
    return torch.cat([j, pad], dim=0)


# ----------------------------------------------------------------------
# Paired (right + left) IK goalset selection
# ----------------------------------------------------------------------


def _extract_paired_coarse_goalset(candidates: list) -> Optional[torch.Tensor]:
    """Build ``(G, 14)`` = ``[right_7, left_7]`` per row from paired candidates."""
    rows = []
    for c in candidates:
        if not (isinstance(c, dict) and "left_hand" in c and "right_hand" in c):
            continue
        right_cg = c["right_hand"].get("coarse_grasp")
        left_cg = c["left_hand"].get("coarse_grasp")
        if right_cg is None or left_cg is None:
            continue
        rows.append(torch.cat([_phase_pose7(right_cg), _phase_pose7(left_cg)], dim=0))
    if not rows:
        return None
    return torch.stack(rows, dim=0)


def _resolve_robot_name(env: BiDexGraspEnv) -> str:
    rm = getattr(env.scene, "robot_manager", None)
    if rm is not None and isinstance(getattr(rm, "robots", None), dict) and rm.robots:
        return next(iter(rm.robots.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_robot_state_dict(env: BiDexGraspEnv) -> dict:
    states = env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
    if isinstance(states, dict):
        rs = states.get(_resolve_robot_name(env), next(iter(states.values())))
    else:
        rs = states
    return {
        "base_pos": rs["base_pos"],
        "base_quat": rs["base_quat"],
        "joint_pos": rs["joint_pos"],
        "joint_vel": rs["joint_vel"],
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
    """Submit the paired ``(1, G, 14)`` coarse_grasp goalset to dual IK; return
    ``(ok, selected_idx)``. The ``g`` returned satisfies BOTH arms (paired).
    """
    ik_server = _get_ik_server(env)
    if ik_server is None:
        log.warning("[bi-ik] no IK server resolvable; falling back to candidate 0.")
        return False, 0
    rs = _get_robot_state_dict(env)
    target = paired_goalset.unsqueeze(0).to(env.device).contiguous()  # (1, G, 14)
    n_goals = int(target.shape[1])
    is_dual = bool(getattr(ik_server, "dual_mode", False))
    log.info(
        "[bi-ik] submit env_ids={} dual_ik={} n_goals={} target.shape={}",
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
        success_list, idx_list, ret_envs = fut.result(timeout=120.0)
    except Exception as ex:
        log.warning("[bi-ik] result exception: {}", ex)
        return False, 0
    log.info(
        "[bi-ik] result success_list={} idx_list={} ret_envs={}",
        success_list,
        idx_list,
        ret_envs,
    )
    if not ret_envs or int(ret_envs[0]) != int(env_id):
        return False, 0
    selected_idx = int(idx_list[0]) if idx_list and len(idx_list) >= 1 else -1
    ok = len(success_list) >= 1 and bool(success_list[0])
    if not ok or selected_idx < 0:
        return False, 0
    return True, selected_idx


@hydra.main(version_base=None, config_path="../../Conf", config_name="bi_dex_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: BiDexGraspEnv = gym.make(
        "BiDexGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    device = env.device
    dtype = torch.float32
    hand_zeros = torch.zeros(2 * HAND_JOINT_DIM, device=device, dtype=dtype)
    # Vega1pSharpa parked uses the p_controller base channel: 15-dim
    # G1-style ``[base(7), arm_center(7), lock_flag(1)]`` block. All-NaN
    # → :class:`Vega1pSharpaPControllerHelper` resolves it to ``lock_skip``
    # with the last valid base pose, i.e. the parked base stays put.
    base_nan = torch.full((15,), torch.nan, device=device, dtype=dtype)

    def _build_action(arm14: torch.Tensor, hand44: torch.Tensor) -> torch.Tensor:
        # Layout: ``[base_p_controller(15), right_arm(7), left_arm(7), hands(44)] = 73``.
        return (
            torch.cat([base_nan, arm14, hand44], dim=0)
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
        )

    for _ in range(50):
        env.sim_step()

    # Register the table (and any other ``/dynamic`` scene props) as IK
    # obstacles so paired goalset selection rejects candidates that would
    # need either arm to cross through the desk top — without this call
    # IK will happily pick a "reachable" index that physically clips
    # through the table or swaps left/right by routing arms across the
    # body. Ignore the bin itself (the grasp target).
    env.scene.planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        obstacle_ignore_path_list=["bin"],
        env_ids=[0],
    )

    grasp_list = env.get_bimanual_grasp_pose(hand_type="sharpa")
    if not grasp_list or grasp_list[0] is None:
        log.error("No paired sharpa grasp poses loaded — check the asset JSON.")
        return

    candidates = _flatten_pair_grasp_dict(grasp_list[0])
    if not candidates:
        log.error("No paired grasp candidates in the annotation.")
        return

    # Paired IK goalset solve: pick a single ``g`` that satisfies BOTH arms
    # at the coarse_grasp pose. Fall back to candidate 0 only if IK fails.
    paired_goalset = _extract_paired_coarse_goalset(candidates)
    if paired_goalset is None or paired_goalset.shape[0] == 0:
        log.error("No paired coarse_grasp goalset rows.")
        return
    paired_goalset = paired_goalset.to(device)
    ok, selected_idx = _ik_select_paired(env, paired_goalset)
    if not ok or selected_idx < 0 or selected_idx >= len(candidates):
        log.warning(
            "[bi-ik] paired goalset failed; falling back to candidate 0 "
            "(ok={}, idx={}, n_candidates={}).",
            ok,
            selected_idx,
            len(candidates),
        )
        selected_idx = 0
    log.info(
        "[bi-ik] selected paired idx={}/{}",
        selected_idx,
        len(candidates),
    )

    pair = candidates[selected_idx]
    right = pair["right_hand"]
    left = pair["left_hand"]
    stage_list = ["coarse_grasp", "fine_grasp", "final_grasp"]

    r_coarse = _phase_pose7(right["coarse_grasp"]).to(device)
    l_coarse = _phase_pose7(left["coarse_grasp"]).to(device)

    # Inter-hand-axis pregrasp: each hand 0.25m AWAY from its partner.
    pre_r, pre_l = compute_pair_pregrasp(r_coarse, l_coarse, 0.25)

    print("current stage: pregrasp")
    visualize_grasp_pose([pre_r, pre_l])
    for _ in range(100):
        arm = torch.cat([pre_r, pre_l], dim=0)
        action = _build_action(arm, hand_zeros)
        env.step(action=action)

    for stage in stage_list:
        if stage not in right or stage not in left:
            continue
        print(f"current stage: {stage}")
        r_pose = _phase_pose7(right[stage]).to(device)
        l_pose = _phase_pose7(left[stage]).to(device)
        r_joints = _phase_joints(right[stage], HAND_JOINT_DIM, device, dtype)
        l_joints = _phase_joints(left[stage], HAND_JOINT_DIM, device, dtype)
        visualize_grasp_pose([r_pose, l_pose])
        for _ in range(100):
            arm = torch.cat([r_pose, l_pose], dim=0)
            hand = torch.cat([r_joints, l_joints], dim=0)
            action = _build_action(arm, hand)
            env.step(action=action)

    # Retrieval: lift both arms together, hands stay closed.
    r_final = _phase_pose7(right["final_grasp"]).to(device).clone()
    l_final = _phase_pose7(left["final_grasp"]).to(device).clone()
    r_final[2] += 0.2
    l_final[2] += 0.2
    r_joints = _phase_joints(right["final_grasp"], HAND_JOINT_DIM, device, dtype)
    l_joints = _phase_joints(left["final_grasp"], HAND_JOINT_DIM, device, dtype)
    print("current stage: retrieval")
    visualize_grasp_pose([r_final, l_final])
    for _ in range(150):
        arm = torch.cat([r_final, l_final], dim=0)
        hand = torch.cat([r_joints, l_joints], dim=0)
        action = _build_action(arm, hand)
        env.step(action=action)

    while True:
        env.step(action=action)


if __name__ == "__main__":
    main()
