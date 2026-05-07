"""
Open-loop bimanual bin grasp on :class:`LocoBiGraspEnv`.

Mirrors :file:`TestLocoGraspEnv.py` but drives BOTH arms in sync to grasp the
bin via the ``dex3_1_bimanual_grasp_pose`` annotation. Paired IK picks a
single candidate that is simultaneously reachable by the right and left arm.

Action layout per step (single env slice; broadcast to ``num_envs``)::

    [p_controller(15), right_arm_ik(7), left_arm_ik(7), left_hand(7), right_hand(7)]

The robot is pinned in front of the bin via ``single_g1_bi_grasp.yaml`` (no
mobile base is driven here — ``p_controller`` is kept all-NaN).
"""

from typing import List, Tuple
from magicsim.Task.LocoManip.Env.LocoBiGraspEnv import LocoBiGraspEnv
import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig
import gymnasium as gym
from pxr import Gf

from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes


AXIS_LENGTH = 0.05
LINE_THICKNESS = 3
LINE_OPACITY = 0.8


def visualize_grasp_poses(poses: List[torch.Tensor]):
    poses_cpu = [p.cpu().numpy().tolist() for p in poses]
    samples = [
        (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6]))
        for p in poses_cpu
    ]
    draw_grasp_samples_as_axes(
        grasp_poses=samples,
        axis_length=AXIS_LENGTH,
        line_thickness=LINE_THICKNESS,
        line_opacity=LINE_OPACITY,
        clear_existing=True,
    )


def _to_t(x, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _pose7(side_phase: dict, device: torch.device) -> torch.Tensor:
    pos = _to_t(side_phase["position"], device).flatten()[:3]
    ori = _to_t(side_phase["orientation"], device).flatten()[:4]
    return torch.cat([pos, ori], dim=0)


def _joints7(side_phase: dict, device: torch.device) -> torch.Tensor:
    j = _to_t(side_phase["joints"], device).flatten()
    if j.numel() >= 7:
        return j[:7]
    pad = torch.zeros(7 - j.numel(), device=device, dtype=j.dtype)
    return torch.cat([j, pad], dim=0)


def _resolve_robot_name(env: LocoBiGraspEnv) -> str:
    robot_manager = getattr(env.scene, "robot_manager", None)
    if robot_manager is None:
        raise RuntimeError("robot_manager not available.")
    robots = getattr(robot_manager, "robots", None)
    if isinstance(robots, dict) and robots:
        return next(iter(robots.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_ik_server(env: LocoBiGraspEnv):
    """Post-MERGE_LEFT_RIGHT §1–§8: one flat server per robot."""
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None)
    if not ik_dict:
        return None
    return ik_dict.get(_resolve_robot_name(env))


def _robot_state_dict(env: LocoBiGraspEnv) -> dict:
    states = env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
    if isinstance(states, dict):
        name = _resolve_robot_name(env)
        state = states.get(name, next(iter(states.values())))
    else:
        state = states
    return {
        "base_pos": state["base_pos"],
        "base_quat": state["base_quat"],
        "joint_pos": state["joint_pos"],
        "joint_vel": state["joint_vel"],
    }


def _flatten_pairs(bi_dict: dict) -> list:
    """Flatten ``{functional_grasp, grasp} -> parts -> pair list`` into a flat list."""
    if not isinstance(bi_dict, dict):
        return []
    out = []
    for top_key in ("functional_grasp", "grasp"):
        parts = bi_dict.get(top_key, {})
        if not isinstance(parts, dict):
            continue
        for pair_list in parts.values():
            if isinstance(pair_list, list):
                out.extend(pair_list)
    return [
        p for p in out if isinstance(p, dict) and "left_hand" in p and "right_hand" in p
    ]


def _pack_paired_goalset(
    right_coarse: torch.Tensor, left_coarse: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """(G, 7) + (G, 7) -> (1, G, 14) with slot 0=right, slot 1=left."""
    G = right_coarse.shape[0]
    target = torch.empty((1, G, 14), device=device, dtype=torch.float32)
    target[0, :, :7] = right_coarse.to(device)
    target[0, :, 7:] = left_coarse.to(device)
    return target


def _ik_select_paired(
    env: LocoBiGraspEnv, pairs: list, env_id: int = 0
) -> Tuple[bool, int]:
    ik_server = _get_ik_server(env)
    if ik_server is None:
        log.warning("IK server unavailable; falling back to pair[0].")
        return False, 0

    dev = env.device
    rights, lefts = [], []
    for pair in pairs:
        rights.append(_pose7(pair["right_hand"]["coarse_grasp"], dev))
        lefts.append(_pose7(pair["left_hand"]["coarse_grasp"], dev))
    r_stack = torch.stack(rights, dim=0)  # (G, 7)
    l_stack = torch.stack(lefts, dim=0)
    target = _pack_paired_goalset(r_stack, l_stack, dev)
    is_dual = getattr(ik_server, "dual_mode", False)
    log.info(
        "[ik paired] submit dual={} n_pairs={} target.shape={}",
        is_dual,
        int(target.shape[1]),
        tuple(target.shape),
    )
    rs = _robot_state_dict(env)
    if is_dual:
        # lock_base=True: solve with the current base pose pinned. Open-loop
        # test never drives the base (p_controller is all-NaN), so base must
        # NOT enter the IK search — otherwise curobo can return a candidate
        # that's only reachable by moving the base ~100m away (joint[3] saw
        # -145 in standalone testing). In sim the base stays put, Pink IK
        # then can't reach the target with arms alone, and the soft
        # orientation cost (2.0 vs position 8.0) lets the wrist settle in a
        # palm-up pose to satisfy position only.
        req = DualIKPlanRequest(
            env_ids=[env_id],
            target_pos=target,
            robot_states=rs,
            mode="goalset",
            lock_base=True,
        )
    else:
        req = IKPlanRequest(
            env_ids=[env_id], target_pos=target, robot_states=rs, mode="goalset"
        )
    fut = ik_server.submit_ik(req)
    try:
        success, idx_list, ret_env_ids = fut.result(timeout=120.0)
    except Exception as ex:
        log.warning("[ik paired] exception: {}", ex)
        return False, 0
    log.info(
        "[ik paired] success={} idx={} ret_envs={}", success, idx_list, ret_env_ids
    )
    if not ret_env_ids or int(ret_env_ids[0]) != int(env_id):
        return False, 0
    selected = int(idx_list[0]) if idx_list and len(idx_list) >= 1 else -1
    ok = bool(success[0]) if success and len(success) >= 1 else False
    if not ok or selected < 0 or selected >= len(pairs):
        return False, 0
    return True, selected


def _pose_along_grasp_direction(
    pose7: torch.Tensor, offset: float, backward: bool = True
) -> torch.Tensor:
    device = pose7.device
    pos = pose7[:3]
    quat = pose7[3:7]
    rot = quat_to_rot_matrix(quat.unsqueeze(0))
    approach = rot[0, :, 1]
    approach = approach / torch.norm(approach)
    off = approach * offset
    new_pos = pos - off if backward else pos + off
    return torch.cat([new_pos, quat], dim=0).to(device)


def _pose_upward(pose7: torch.Tensor, offset: float) -> torch.Tensor:
    new_pos = pose7[:3].clone()
    new_pos[2] += offset
    return torch.cat([new_pos, pose7[3:7]], dim=0)


def _build_action(
    device: torch.device,
    right_arm: torch.Tensor,
    left_arm: torch.Tensor,
    right_hand: torch.Tensor,
    left_hand: torch.Tensor,
) -> torch.Tensor:
    """Action layout (43D): ``[p_ctrl(15), right_arm(7), left_arm(7), right_hand(7), left_hand(7)]``.

    Hand order matches ``G1.eef_action['joint_pos']`` (G1.py:847) whose joint
    groups are right first then left. ``TestLocoGraspEnv.py`` has left/right
    flipped — it closes only the opposite hand which usually hits joint
    limits and silently clamps, giving the appearance of an inert hand.
    """
    p_controller = torch.full((15,), torch.nan, device=device, dtype=torch.float32)
    return torch.cat([p_controller, right_arm, left_arm, right_hand, left_hand], dim=0)


def _step_n(env: LocoBiGraspEnv, action_1d: torch.Tensor, steps: int):
    batched = action_1d.unsqueeze(0).repeat(env.num_envs, 1)
    for _ in range(steps):
        env.step(action=batched)


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="loco_bi_grasp_env"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: LocoBiGraspEnv = gym.make(
        "LocoBiGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    device = env.device
    dtype = torch.float32

    # Neutral pose: arms parked in front of the chest, hands open.
    neutral_right = torch.tensor(
        [-0.5, -0.20, 0.95, 1, 0, 0, 0], device=device, dtype=dtype
    )
    neutral_left = torch.tensor(
        [-0.5, 0.20, 0.95, 1, 0, 0, 0], device=device, dtype=dtype
    )
    open_hand = torch.zeros(7, device=device, dtype=dtype)

    # ---- 1. Settle to neutral pose ----
    neutral_action = _build_action(
        device, neutral_right, neutral_left, open_hand, open_hand
    )
    _step_n(env, neutral_action, 150)

    # env.scene.planner_manager.update_obstacles(
    #     obstacle_avoidance_path_list=["dynamic"],
    #     obstacle_ignore_path_list=["bin"],
    #     env_ids=[0],
    # )

    # ---- 2. Load paired annotation + IK-select one pair ----
    bi_list = env.get_bimanual_grasp_pose(hand_type="dex3_1")
    bi_dict = bi_list[0] if bi_list else None
    pairs = _flatten_pairs(bi_dict) if bi_dict else []
    if not pairs:
        log.error(
            "No dex3_1_bimanual grasp pairs. Check Bin Annotation/dex3_1_bimanual_grasp_pose.json."
        )
        return
    ok, idx = _ik_select_paired(env, pairs)
    if not ok:
        log.warning("Paired IK failed; falling back to pair[0].")
        idx = 0
    chosen = pairs[idx]
    log.info("Using paired candidate idx={} of {}", idx, len(pairs))

    # ---- 3. Extract phase poses and joints for both sides ----
    right = chosen["right_hand"]
    left = chosen["left_hand"]
    r_coarse = _pose7(right["coarse_grasp"], device)
    l_coarse = _pose7(left["coarse_grasp"], device)
    r_final = _pose7(right["final_grasp"], device)
    l_final = _pose7(left["final_grasp"], device)
    # fine_grasp uses the final pose for the ARM but keeps the hand open —
    # this is the "precise positioning" stage where the wrist is already at
    # the closing pose but fingers haven't curled yet. Annotation's
    # fine_grasp pos sometimes differs from final, but for this test we
    # want the arm to settle once and only the hand to actuate.
    r_fine = r_final
    l_fine = l_final

    r_j_coarse = (
        _joints7(right["coarse_grasp"], device)
        if "joints" in right["coarse_grasp"]
        else open_hand
    )
    l_j_coarse = (
        _joints7(left["coarse_grasp"], device)
        if "joints" in left["coarse_grasp"]
        else open_hand
    )
    r_j_final = _joints7(right["final_grasp"], device)
    l_j_final = _joints7(left["final_grasp"], device)

    # Pre-grasp: 10cm straight up along world z (short approach from above),
    # same quat as coarse. Per-side grasp-direction offset gave bad geometry
    # on the bin's side-grasps, so keep it simple here.
    r_pre = _pose_upward(r_coarse, 0.1)
    l_pre = _pose_upward(l_coarse, 0.1)
    r_retrieval = _pose_upward(r_final, 0.2)
    l_retrieval = _pose_upward(l_final, 0.2)

    # ---- 4. Run phases: pre → coarse → fine → final → retrieval ----
    # Phase tuple order: (name, right_arm, left_arm, right_hand, left_hand, steps)
    phases = [
        ("pre_grasp", r_pre, l_pre, open_hand, open_hand, 500),
        ("coarse_grasp", r_coarse, l_coarse, open_hand, open_hand, 400),
        ("fine_grasp", r_fine, l_fine, open_hand, open_hand, 300),
        ("final_grasp", r_final, l_final, r_j_final, l_j_final, 300),
        ("retrieval", r_retrieval, l_retrieval, r_j_final, l_j_final, 200),
    ]
    for name, rp, lp, rh, lh, n_steps in phases:
        print(f"current phase: {name}")
        # Print one row of the actual command going to env.step. Layout:
        #   [p_ctrl(15) NaN, right_arm(7)=[xyz,wxyz], left_arm(7)=[xyz,wxyz],
        #    right_hand(7) joint_pos, left_hand(7) joint_pos]
        rp_np = rp.detach().cpu().numpy().tolist()
        lp_np = lp.detach().cpu().numpy().tolist()
        rh_np = rh.detach().cpu().numpy().tolist()
        lh_np = lh.detach().cpu().numpy().tolist()
        print(
            f"  right_arm xyz=[{rp_np[0]:+.3f},{rp_np[1]:+.3f},{rp_np[2]:+.3f}] "
            f"wxyz=[{rp_np[3]:+.3f},{rp_np[4]:+.3f},{rp_np[5]:+.3f},{rp_np[6]:+.3f}]"
        )
        print(
            f"  left_arm  xyz=[{lp_np[0]:+.3f},{lp_np[1]:+.3f},{lp_np[2]:+.3f}] "
            f"wxyz=[{lp_np[3]:+.3f},{lp_np[4]:+.3f},{lp_np[5]:+.3f},{lp_np[6]:+.3f}]"
        )
        print(f"  right_hand joints={[round(v, 3) for v in rh_np]}")
        print(f"  left_hand  joints={[round(v, 3) for v in lh_np]}")
        visualize_grasp_poses([rp, lp])
        action = _build_action(device, rp, lp, rh, lh)
        _step_n(env, action, n_steps)

    # Hold final state.
    hold = _build_action(device, r_retrieval, l_retrieval, r_j_final, l_j_final)
    batched_hold = hold.unsqueeze(0).repeat(env.num_envs, 1)
    while True:
        env.step(action=batched_hold)


if __name__ == "__main__":
    main()
