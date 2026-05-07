"""Open-loop bimanual basket grasp on :class:`BiGraspEnv`.

Closely follows :file:`Task/LocoManip/Env/Test/TestLocoBiGraspEnv.py`:
load the paired annotation, pack every candidate into a goalset, let
paired IK pick the single pair both arms can solve, then drive open-loop
through pre → grasp → close → lift.

Why "let IK pick" instead of swapping JSON keys
-----------------------------------------------
``Assets/Object/basket/basket_*/Annotation/bi_gripper_grasp_pose.json``
labels its candidates ``left`` / ``right`` non-deterministically — during
labeling the basket was sometimes yawed, so a candidate's ``right`` key
can sit at either world +y or world -y. Out of 534 pairs in basket_1, the
JSON's ``right`` key has y<0 in 274 of them and y>0 in 260; there is no
side-stable convention. So we don't yaw the basket, we don't swap by
hand, we just pack every candidate's ``right`` into goalset slot 0 (right
arm at world y<0) and ``left`` into slot 1 — curobo paired IK fails any
candidate where the labels happen to be flipped and succeeds on one
where they line up.

Action layout (16D, right-first per :file:`Env/Robot/Cfg/DualManipulator/DualFranka.py:10`)::

    [right_arm(7), left_arm(7), right_grip(1), left_grip(1)]
"""

from typing import List, Tuple
from magicsim.Task.TableTop.Env.BiGraspEnv import BiGraspEnv  # noqa: F401 (gym register)
import gymnasium as gym
import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig
from pxr import Gf

from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes


PRE_OFFSET_Z = 0.15
LIFT_OFFSET_Z = 0.30

AXIS_LENGTH = 0.06
LINE_THICKNESS = 2
LINE_OPACITY = 0.9


def visualize_grasp_poses(poses: List[torch.Tensor]):
    """Draw axes at each pose. Pose is 7D ``[x, y, z, qw, qx, qy, qz]``."""
    poses_cpu = [p.detach().cpu().numpy().tolist() for p in poses]
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


def _shift_up(pose7: torch.Tensor, dz: float) -> torch.Tensor:
    out = pose7.clone()
    out[2] += dz
    return out


def _resolve_robot_name(env: BiGraspEnv) -> str:
    rm = getattr(env.scene, "robot_manager", None)
    if rm is None:
        raise RuntimeError("robot_manager not available.")
    robots = getattr(rm, "robots", None)
    if isinstance(robots, dict) and robots:
        return next(iter(robots.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_ik_server(env: BiGraspEnv):
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None)
    if not ik_dict:
        return None
    return ik_dict.get(_resolve_robot_name(env))


def _robot_state_dict(env: BiGraspEnv) -> dict:
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


def _pack_paired_goalset(
    rights: torch.Tensor, lefts: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """(G, 7) + (G, 7) → (1, G, 14) with slot 0=right, slot 1=left."""
    G = rights.shape[0]
    target = torch.empty((1, G, 14), device=device, dtype=torch.float32)
    target[0, :, :7] = rights.to(device)
    target[0, :, 7:] = lefts.to(device)
    return target


def _ik_select_paired(
    env: BiGraspEnv, pairs: List[dict], env_id: int = 0
) -> Tuple[bool, int]:
    """Submit all (right, left) candidates, let paired IK pick a reachable pair."""
    ik_server = _get_ik_server(env)
    if ik_server is None:
        log.warning("IK server unavailable; falling back to pair[0].")
        return False, 0

    dev = env.device
    rights = torch.stack([p["right"].to(dev) for p in pairs], dim=0)  # (G, 7)
    lefts = torch.stack([p["left"].to(dev) for p in pairs], dim=0)
    target = _pack_paired_goalset(rights, lefts, dev)
    is_dual = getattr(ik_server, "dual_mode", False)
    log.info(
        "[ik paired] submit dual={} n_pairs={} target.shape={}",
        is_dual,
        int(target.shape[1]),
        tuple(target.shape),
    )
    rs = _robot_state_dict(env)
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


def _build_16d_action(
    right_arm_pose: torch.Tensor,
    left_arm_pose: torch.Tensor,
    right_grip: float,
    left_grip: float,
    device: torch.device,
) -> torch.Tensor:
    grip = torch.tensor([right_grip, left_grip], device=device, dtype=torch.float32)
    row = torch.cat(
        [
            right_arm_pose.to(device).flatten()[:7],
            left_arm_pose.to(device).flatten()[:7],
            grip,
        ],
        dim=0,
    )
    assert row.numel() == 16, f"expected 16D, got {row.numel()}"
    return row


def _step_n(env: BiGraspEnv, action_1d: torch.Tensor, steps: int):
    batched = action_1d.unsqueeze(0).repeat(env.num_envs, 1)
    for _ in range(steps):
        env.step(action=batched)


def _print_phase(name: str, r_arm: torch.Tensor, l_arm: torch.Tensor, grip: float):
    right_pose = r_arm.detach().cpu().numpy().tolist()
    left_pose = l_arm.detach().cpu().numpy().tolist()
    print(f"current phase: {name}")
    print(
        f"  right_arm xyz=[{right_pose[0]:+.3f},{right_pose[1]:+.3f},{right_pose[2]:+.3f}] "
        f"wxyz=[{right_pose[3]:+.3f},{right_pose[4]:+.3f},{right_pose[5]:+.3f},{right_pose[6]:+.3f}]"
    )
    print(
        f"  left_arm  xyz=[{left_pose[0]:+.3f},{left_pose[1]:+.3f},{left_pose[2]:+.3f}] "
        f"wxyz=[{left_pose[3]:+.3f},{left_pose[4]:+.3f},{left_pose[5]:+.3f},{left_pose[6]:+.3f}]"
    )
    print(f"  grip={grip}")


@hydra.main(version_base=None, config_path="../../Conf", config_name="bi_grasp_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: BiGraspEnv = gym.make(
        "BiGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    device = env.device

    # ---- 1. Settle ----
    print("[bi_grasp] settling scene (50 steps)...")
    for _ in range(50):
        env.step(action=None)

    # No obstacle avoidance for this open-loop test: curobo's default
    # state is empty obstacle set, which is exactly what we want — the
    # IK only needs to solve kinematics for the basket's annotated grasp
    # poses; collisions with the basket / table are handled by the
    # physics sim. Calling ``update_obstacles`` with the basket ignored
    # would also work but is needless plumbing here.

    # ---- DEBUG: dump scene geometry so we can sanity-check IK reach ----
    obj_pose = env.get_object_pose().get(cfg.target_obj_name, None)
    if obj_pose is not None:
        bp = obj_pose[0].cpu().numpy().tolist()
        log.info(
            "[debug] basket world pose: pos=({:+.3f},{:+.3f},{:+.3f}) "
            "quat=({:+.3f},{:+.3f},{:+.3f},{:+.3f})",
            bp[0],
            bp[1],
            bp[2],
            bp[3],
            bp[4],
            bp[5],
            bp[6],
        )
    rs = _robot_state_dict(env)
    bp_t = rs["base_pos"]
    bpos = bp_t[0].cpu().numpy().tolist() if hasattr(bp_t, "cpu") else list(bp_t[0])
    log.info(
        "[debug] dual_franka base_link world pos=({:+.3f},{:+.3f},{:+.3f}); "
        "R_link0≈({:+.3f},{:+.3f},{:+.3f}) L_link0≈({:+.3f},{:+.3f},{:+.3f}); "
        "Franka reach≈0.855m",
        bpos[0],
        bpos[1],
        bpos[2],
        bpos[0],
        bpos[1] - 0.9,
        bpos[2],
        bpos[0],
        bpos[1] + 0.9,
        bpos[2],
    )

    # ---- 2. Load paired annotation + paired IK selection ----
    pairs = env.get_bigripper_pairs_flat(
        env_id=0,
        obj_name=cfg.target_obj_name,
        functional_grasp=True,
        part="body",
    )
    if not pairs:
        log.error(
            "No bi_gripper grasp pairs. Check Annotation/bi_gripper_grasp_pose.json."
        )
        return
    log.info("Loaded {} bigripper pairs from annotation.", len(pairs))

    ok, idx = _ik_select_paired(env, pairs)
    if not ok:
        log.warning("Paired IK failed; falling back to pair[0].")
        idx = 0
    chosen = pairs[idx]
    # Slot order matches the goalset we submitted: right=slot 0, left=slot 1.
    r_arm_grasp = chosen["right"].to(device)
    l_arm_grasp = chosen["left"].to(device)
    log.info(
        "Using paired candidate idx={} of {}; right_arm.y={:+.3f}, left_arm.y={:+.3f}",
        idx,
        len(pairs),
        float(r_arm_grasp[1].item()),
        float(l_arm_grasp[1].item()),
    )

    # ---- 3. Per-phase poses ----
    r_pre = _shift_up(r_arm_grasp, PRE_OFFSET_Z)
    l_pre = _shift_up(l_arm_grasp, PRE_OFFSET_Z)
    r_lift = _shift_up(r_arm_grasp, LIFT_OFFSET_Z)
    l_lift = _shift_up(l_arm_grasp, LIFT_OFFSET_Z)

    # (name, right_arm, left_arm, grip, n_steps)
    phases = [
        ("pre_grasp", r_pre, l_pre, 0.0, 200),
        ("grasp", r_arm_grasp, l_arm_grasp, 0.0, 200),
        ("close_gripper", r_arm_grasp, l_arm_grasp, 1.0, 100),
        ("lift", r_lift, l_lift, 1.0, 250),
    ]

    for name, rp, lp, grip, n_steps in phases:
        _print_phase(name, rp, lp, grip)
        visualize_grasp_poses([rp, lp])
        action = _build_16d_action(rp, lp, grip, grip, device)
        _step_n(env, action, n_steps)

    # ---- 4. Hold lifted state ----
    print("[bi_grasp] holding lifted state; idling...")
    hold = _build_16d_action(r_lift, l_lift, 1.0, 1.0, device)
    while True:
        env.step(action=hold.unsqueeze(0).repeat(env.num_envs, 1))


if __name__ == "__main__":
    main()
