"""Open-loop bimanual handover on :class:`HandoverEnv`.

Mirrors :file:`Test/TestBiGraspEnv.py` in shape: load grasp annotation,
let IK pick reachable poses, then drive the env step-by-step with 16D
actions (right_arm + left_arm + right_grip + left_grip) feeding
``ik_dual_diff``. No AutoCollect / atomic-skill machinery — the goal of
this driver is to verify the geometric pipeline (grasp pool, handover
candidate generation, paired IK) works in isolation.

Two IK solves happen, exactly like in the closed-loop atomic skill:

    1. Single-arm IK over the full grasp pool with right slot NaN-disabled
       — picks a reachable left grasp on the mug.
    2. Paired bimanual IK over a goalset of
       (handover_mug_pose × right_local_grasp) cross-product. Slot 0 holds
       the right-arm world target derived from each (mug_pose, right_local);
       slot 1 holds the matching left-arm world target = mug_pose ⊙
       left_grasp_local — i.e. the left wrist must stay grasping the mug at
       the proposed handover orientation. Curobo argmins jointly so the
       chosen ``g`` is reachable for BOTH arms simultaneously.

Phases (16D action; grip 0=open, 1=close):

    settle           - env.step(None) × 50
    left_pre_grasp   - L=grasp - 0.15·z_local; R=hold; gL=0
    left_grasp       - L=grasp;                R=hold; gL=0
    left_close       - L=grasp;                R=hold; gL=1
    left_lift        - L=grasp + 0.15·z_world; R=hold; gL=1
    right_pre_grasp  - L=handover_eef; R=right_grasp - 0.15·z_local; gR=0 gL=1
    right_grasp      - L=handover_eef; R=right_grasp;                 gR=0 gL=1
    right_close      - L=handover_eef; R=right_grasp;                 gR=1 gL=1
    left_open        - L=handover_eef; R=right_grasp;                 gR=1 gL=0
    left_retract     - L=handover_eef - 0.15·z_local; R=right_grasp;  gR=1 gL=0
    hold             - last action repeated forever
"""

# Env import MUST come first — it triggers the Isaac Sim app bootstrap.
# Importing IK / IsaacLab modules before this raises
# ``ModuleNotFoundError: No module named 'omni'`` at import time.
from magicsim.Task.TableTop.Env.HandoverEnv import HandoverEnv  # noqa: F401 (gym register)

from typing import Tuple

import gymnasium as gym
import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig
from pxr import Gf

from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Scene.Object.Rigid import RigidObject
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes


PRE_OFFSET_Z = 0.15
LIFT_OFFSET_Z = 0.15
RIGHT_PRE_OFFSET_Z = 0.15
LEFT_RETRACT_OFFSET_Z = 0.15

# Open-loop smoke test hard-codes "left grabs BODY, right grabs HANDLE"
# for mug 8684 (body 46 candidates, handle 22). Right pool is the FULL
# handle set (no sub-sample). Mug-pose generator is shrunk to keep G
# manageable: yaw=8, pitch=3, roll=3, y/z=3 → 432 mug poses,
# G = 432 × 22 = 9504 (under default ik_max_goalset 10000).
SMOKE_N_YAWS = 8
SMOKE_PITCH_DEGS = (-30.0, 0.0, 30.0)
SMOKE_ROLL_DEGS = (-30.0, 0.0, 30.0)

AXIS_LENGTH = 0.06
LINE_THICKNESS = 2
LINE_OPACITY = 0.9


# ---------------------------------------------------------------- viz
def _viz_poses(poses):
    samples = []
    for p in poses:
        p = p.detach().cpu().numpy().tolist()
        samples.append((Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6])))
    draw_grasp_samples_as_axes(
        grasp_poses=samples,
        axis_length=AXIS_LENGTH,
        line_thickness=LINE_THICKNESS,
        line_opacity=LINE_OPACITY,
        clear_existing=True,
    )


# ---------------------------------------------------------------- robot helpers
def _resolve_robot_name(env: HandoverEnv) -> str:
    rm = getattr(env.scene, "robot_manager", None)
    if rm is None:
        raise RuntimeError("robot_manager not available.")
    robots = getattr(rm, "robots", None)
    if isinstance(robots, dict) and robots:
        return next(iter(robots.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_ik_server(env: HandoverEnv):
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None)
    if not ik_dict:
        return None
    return ik_dict.get(_resolve_robot_name(env))


def _robot_state_dict(env: HandoverEnv) -> dict:
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


def _get_eef_pose(env: HandoverEnv, slot: int) -> torch.Tensor:
    """Read current eef pose 7-vec for slot 0=right, 1=left."""
    states = env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
    name = _resolve_robot_name(env)
    rs = states[name] if isinstance(states, dict) else states
    pos = rs["eef_pos"]
    quat = rs["eef_quat"]
    if pos.dim() == 3:
        return torch.cat([pos[0, slot], quat[0, slot]], dim=0).to(env.device)
    return torch.cat([pos[0], quat[0]], dim=0).to(env.device)


# ---------------------------------------------------------------- IK
def _submit_goalset(env, target: torch.Tensor) -> Tuple[bool, int]:
    """Submit a goalset IK request, return (success, selected_idx)."""
    ik_server = _get_ik_server(env)
    if ik_server is None:
        return False, -1
    is_dual = bool(getattr(ik_server, "dual_mode", False))
    rs = _robot_state_dict(env)
    if is_dual:
        req = DualIKPlanRequest(
            env_ids=[0],
            target_pos=target,
            robot_states=rs,
            mode="goalset",
            lock_base=False,
        )
    else:
        req = IKPlanRequest(
            env_ids=[0], target_pos=target, robot_states=rs, mode="goalset"
        )
    fut = ik_server.submit_ik(req)
    try:
        success, idx_list, ret_envs = fut.result(timeout=180.0)
    except Exception as ex:
        log.warning("[ik] exception: {}", ex)
        return False, -1
    if not ret_envs or int(ret_envs[0]) != 0:
        return False, -1
    ok = bool(success[0]) if success else False
    idx = int(idx_list[0]) if idx_list else -1
    log.info("[ik] success={} idx={} G={}", ok, idx, int(target.shape[1]))
    return ok, idx


def _select_left_grasp(env: HandoverEnv, world_pool: torch.Tensor) -> int:
    """Single-arm IK: pack right slot NaN, left slot = grasp pool. Returns idx."""
    device = env.device
    G = world_pool.shape[0]
    target = torch.full((1, G, 14), float("nan"), device=device, dtype=torch.float32)
    target[0, :, 7:] = world_pool  # slot 1 = left
    ok, idx = _submit_goalset(env, target)
    if not ok or idx < 0:
        return -1
    return idx


def _select_handover(
    env: HandoverEnv,
    rights_world: torch.Tensor,
    lefts_world: torch.Tensor,
) -> int:
    """Paired IK over (right, left) candidates already in world frame."""
    device = env.device
    G = rights_world.shape[0]
    target = torch.empty((1, G, 14), device=device, dtype=torch.float32)
    target[0, :, :7] = rights_world  # slot 0 = right
    target[0, :, 7:] = lefts_world  # slot 1 = left
    ok, idx = _submit_goalset(env, target)
    if not ok or idx < 0:
        return -1
    return idx


# ---------------------------------------------------------------- geometry
def _shift_along_local_z(
    pose7: torch.Tensor, offset: float, backward: bool = True
) -> torch.Tensor:
    rot = quat_to_rot_matrix(pose7[3:7].unsqueeze(0))[0]
    approach = rot[:, 2]
    approach = approach / torch.norm(approach)
    delta = approach * offset
    new_pos = pose7[:3] - delta if backward else pose7[:3] + delta
    return torch.cat([new_pos, pose7[3:7]], dim=0)


def _shift_world_z(pose7: torch.Tensor, dz: float) -> torch.Tensor:
    out = pose7.clone()
    out[2] += dz
    return out


# ---------------------------------------------------------------- action / step
def _build_16d(
    right_arm: torch.Tensor,
    left_arm: torch.Tensor,
    right_grip: float,
    left_grip: float,
    device: torch.device,
) -> torch.Tensor:
    grip = torch.tensor([right_grip, left_grip], device=device, dtype=torch.float32)
    row = torch.cat(
        [right_arm.to(device).flatten()[:7], left_arm.to(device).flatten()[:7], grip],
        dim=0,
    )
    assert row.numel() == 16, f"expected 16D, got {row.numel()}"
    return row


def _step_n(env: HandoverEnv, action_1d: torch.Tensor, steps: int):
    batched = action_1d.unsqueeze(0).repeat(env.num_envs, 1)
    for _ in range(steps):
        env.step(action=batched)


def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two unit quaternions ``[w,x,y,z]``."""
    q0 = q0 / q0.norm().clamp_min(1e-8)
    q1 = q1 / q1.norm().clamp_min(1e-8)
    dot = (q0 * q1).sum()
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Near-parallel — fall back to lerp + normalize.
        out = (1.0 - alpha) * q0 + alpha * q1
        return out / out.norm().clamp_min(1e-8)
    theta_0 = torch.acos(dot.clamp(-1.0, 1.0))
    theta = theta_0 * alpha
    s0 = torch.cos(theta) - dot * torch.sin(theta) / torch.sin(theta_0)
    s1 = torch.sin(theta) / torch.sin(theta_0)
    return s0 * q0 + s1 * q1


def _interp_pose(p0: torch.Tensor, p1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate two 7-vec poses ``[x,y,z,qw,qx,qy,qz]``: lerp pos, slerp quat."""
    pos = (1.0 - alpha) * p0[:3] + alpha * p1[:3]
    quat = _slerp(p0[3:7], p1[3:7], alpha)
    return torch.cat([pos, quat], dim=0)


def _interp_step_n(
    env: HandoverEnv,
    start_r: torch.Tensor,
    start_l: torch.Tensor,
    end_r: torch.Tensor,
    end_l: torch.Tensor,
    gr: float,
    gl: float,
    steps: int,
    device: torch.device,
):
    """Drive both wrists smoothly from (start_r, start_l) → (end_r, end_l).

    ``ik_dual_diff`` solves a single-tick DLS IK to drive the wrist toward
    the commanded pose, with no built-in trajectory smoothing. Sending a
    distant target straight away makes the controller saturate joint
    velocities and the closed-finger contact slips off rigid objects (mug
    flies out). Linearly interpolating the commanded pose over ``steps``
    ticks tames the controller — joint velocities stay bounded, contact
    forces stay within friction cone.
    """
    for i in range(steps):
        alpha = (i + 1) / steps
        right = _interp_pose(start_r, end_r, alpha)
        left = _interp_pose(start_l, end_l, alpha)
        action = _build_16d(right, left, gr, gl, device)
        env.step(action=action.unsqueeze(0).repeat(env.num_envs, 1))


def _print_phase(
    name: str, right: torch.Tensor, left: torch.Tensor, gr: float, gl: float
):
    rp = right.detach().cpu().numpy().tolist()
    lp = left.detach().cpu().numpy().tolist()
    print(f"current phase: {name}")
    print(
        f"  R xyz=[{rp[0]:+.3f},{rp[1]:+.3f},{rp[2]:+.3f}] "
        f"wxyz=[{rp[3]:+.3f},{rp[4]:+.3f},{rp[5]:+.3f},{rp[6]:+.3f}] grip={gr}"
    )
    print(
        f"  L xyz=[{lp[0]:+.3f},{lp[1]:+.3f},{lp[2]:+.3f}] "
        f"wxyz=[{lp[3]:+.3f},{lp[4]:+.3f},{lp[5]:+.3f},{lp[6]:+.3f}] grip={gl}"
    )


# ---------------------------------------------------------------- main
@hydra.main(version_base=None, config_path="../../Conf", config_name="handover_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: HandoverEnv = gym.make(
        "HandoverEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    device = env.device

    # ---- 1. Settle ----
    print("[handover] settling scene (50 steps)...")
    for _ in range(50):
        env.step(action=None)

    obj_world = env.get_object_world_pose(env_id=0, obj_name=cfg.target_obj_name)
    if obj_world is None:
        log.error("Object {} not found in scene.", cfg.target_obj_name)
        return
    log.info(
        "[debug] {} world pose: pos=({:+.3f},{:+.3f},{:+.3f}) "
        "quat=({:+.3f},{:+.3f},{:+.3f},{:+.3f})",
        cfg.target_obj_name,
        *obj_world.cpu().tolist(),
    )

    # ---- 2. Grasp pool + LEFT IK (BODY only) ----
    # Hard-coded for this test: left arm only grasps the BODY part
    # (annotations: ``grasp.body``, 46 candidates for mug 8684).
    body_pool_local = env.get_grasp_pool(
        env_id=0,
        obj_name=cfg.target_obj_name,
        transform_to_world=False,
        part="body",
    )
    body_pool_world = env.get_grasp_pool(
        env_id=0,
        obj_name=cfg.target_obj_name,
        transform_to_world=True,
        part="body",
    )
    if body_pool_local is None or body_pool_local.shape[0] == 0:
        log.error("No body grasp candidates for {}.", cfg.target_obj_name)
        return
    log.info("Loaded BODY grasp pool: {} candidates.", body_pool_local.shape[0])

    left_idx = _select_left_grasp(env, body_pool_world)
    if left_idx < 0:
        log.error("Left-arm IK found no reachable BODY grasp.")
        return
    left_grasp_local = body_pool_local[left_idx].clone()
    left_grasp_world = body_pool_world[left_idx].clone()
    log.info(
        "Left BODY grasp idx={}/{} world=({:+.3f},{:+.3f},{:+.3f})",
        left_idx,
        body_pool_world.shape[0],
        *left_grasp_world[:3].cpu().tolist(),
    )

    # ---- 3. Hold-pose for the right arm during left-only phases ----
    right_hold = _get_eef_pose(env, slot=0)
    log.info(
        "Right hold pose (initial): xyz=({:+.3f},{:+.3f},{:+.3f})",
        *right_hold[:3].cpu().tolist(),
    )

    # ---- 4. Left phases ----
    left_pre = _shift_along_local_z(left_grasp_world, PRE_OFFSET_Z, backward=True)
    left_lift = _shift_world_z(left_grasp_world, LIFT_OFFSET_Z)

    # Track the LAST commanded wrist pose for both arms so each phase
    # can interpolate smoothly from the previous commanded target.
    cur_r = right_hold.clone()
    cur_l = _get_eef_pose(env, slot=1)

    left_phases = [
        ("left_pre_grasp", right_hold, left_pre, 0.0, 0.0, 200),
        ("left_grasp", right_hold, left_grasp_world, 0.0, 0.0, 200),
        ("left_close", right_hold, left_grasp_world, 0.0, 1.0, 100),
        ("left_lift", right_hold, left_lift, 0.0, 1.0, 250),
    ]
    for name, right, left, gr, gl, n in left_phases:
        _print_phase(name, right, left, gr, gl)
        _viz_poses([right, left])
        _interp_step_n(env, cur_r, cur_l, right, left, gr, gl, n, device)
        cur_r, cur_l = right, left

    # ---- 5. Handover candidate generation + PAIRED IK ----
    # Smaller mug-pose sweep + full HANDLE pool for right arm (22
    # candidates for mug 8684). Hard-coded "right grabs handle" matches
    # the user's natural handover semantic.
    mug_poses = env.generate_handover_mug_poses(
        n_yaws=SMOKE_N_YAWS,
        pitch_degs=SMOKE_PITCH_DEGS,
        roll_degs=SMOKE_ROLL_DEGS,
    ).to(device)
    n_m = mug_poses.shape[0]
    handle_pool_local = env.get_grasp_pool(
        env_id=0,
        obj_name=cfg.target_obj_name,
        transform_to_world=False,
        part="handle",
    )
    if handle_pool_local is None or handle_pool_local.shape[0] == 0:
        log.error("No handle grasp candidates for {}.", cfg.target_obj_name)
        return
    right_local = handle_pool_local.to(device)
    n_r = right_local.shape[0]
    log.info(
        "Handover IK: n_mug_poses={} n_right_local={} (full HANDLE pool) G={}",
        n_m,
        n_r,
        n_m * n_r,
    )

    rights_world = torch.empty((n_m * n_r, 7), device=device, dtype=torch.float32)
    lefts_world = torch.empty((n_m * n_r, 7), device=device, dtype=torch.float32)
    for mi in range(n_m):
        mug_pos = mug_poses[mi, :3]
        mug_quat = mug_poses[mi, 3:7]
        left_world_for_mug = RigidObject.transform_pose_to_world(
            left_grasp_local, mug_pos, mug_quat
        )
        for ri in range(n_r):
            row = mi * n_r + ri
            rights_world[row] = RigidObject.transform_pose_to_world(
                right_local[ri], mug_pos, mug_quat
            )
            lefts_world[row] = left_world_for_mug

    handover_idx = _select_handover(env, rights_world, lefts_world)
    if handover_idx < 0:
        log.error(
            "Paired handover IK failed across all G={} candidates. "
            "Try widening generate_handover_mug_poses (more yaws / pitches / "
            "y_offsets) or moving handover_center.",
            n_m * n_r,
        )
        # Hold the lifted state so the user can inspect.
        hold = _build_16d(right_hold, left_lift, 0.0, 1.0, device)
        while True:
            env.step(action=hold.unsqueeze(0).repeat(env.num_envs, 1))

    chosen_mug_idx = handover_idx // n_r
    chosen_right_idx = handover_idx % n_r
    chosen_right_local = right_local[chosen_right_idx].clone()
    # Planned values from the paired IK selection — use these for the
    # FIRST handover phase (right_pre_grasp) so the left arm actually
    # swings from its lift pose to the IK-chosen handover pose. Refresh
    # only kicks in on subsequent phases (after the left has moved).
    planned_right_grasp = rights_world[handover_idx].clone()
    planned_left_handover = lefts_world[handover_idx].clone()
    planned_right_pre = _shift_along_local_z(
        planned_right_grasp, RIGHT_PRE_OFFSET_Z, backward=True
    )
    planned_left_retract = _shift_along_local_z(
        planned_left_handover, LEFT_RETRACT_OFFSET_Z, backward=True
    )
    log.info(
        "Handover idx={}/{} (mug_pose={}, right_local={}) "
        "R_planned=({:+.3f},{:+.3f},{:+.3f}) L_planned=({:+.3f},{:+.3f},{:+.3f})",
        handover_idx,
        n_m * n_r,
        chosen_mug_idx,
        chosen_right_idx,
        *planned_right_grasp[:3].cpu().tolist(),
        *planned_left_handover[:3].cpu().tolist(),
    )

    def _refresh_right_targets():
        """Re-derive right grasp + left hold from CURRENT mug world pose.

        Why: between phase transitions the mug drifts a few mm/cm — left
        wrist never reaches its target exactly, so the mug rigidly held
        by the left gripper ends up at ``left_eef_actual ⊙
        inverse(left_grasp_local)`` rather than the planned handover
        pose. Recomputing right's grasp from the LATEST mug pose makes
        right close on the actual mug, not on a stale plan.
        """
        mug_now = env.get_object_world_pose(env_id=0, obj_name=cfg.target_obj_name)
        if mug_now is None:
            log.warning("[refresh] mug pose unavailable; keeping stale targets")
            return None, None, None, None
        mug_pos = mug_now[:3]
        mug_quat = mug_now[3:7]
        right_grasp_w = RigidObject.transform_pose_to_world(
            chosen_right_local, mug_pos, mug_quat
        )
        left_hold_w = RigidObject.transform_pose_to_world(
            left_grasp_local, mug_pos, mug_quat
        )
        right_pre_w = _shift_along_local_z(
            right_grasp_w, RIGHT_PRE_OFFSET_Z, backward=True
        )
        left_retract_w = _shift_along_local_z(
            left_hold_w, LEFT_RETRACT_OFFSET_Z, backward=True
        )
        log.info(
            "[refresh] mug=({:+.3f},{:+.3f},{:+.3f}) "
            "R_grasp=({:+.3f},{:+.3f},{:+.3f}) L_hold=({:+.3f},{:+.3f},{:+.3f})",
            *mug_pos.cpu().tolist(),
            *right_grasp_w[:3].cpu().tolist(),
            *left_hold_w[:3].cpu().tolist(),
        )
        return right_grasp_w, right_pre_w, left_hold_w, left_retract_w

    # ---- 6. Right phases + release ----
    # IMPORTANT: ``right_pre_grasp`` MUST use the PLANNED values, not the
    # refresh. Refresh derives left_target from the CURRENT mug pose,
    # which is still at the lift position — telling the left arm to
    # "stay there" instead of swinging to handover. Refresh only makes
    # sense AFTER left has moved (i.e., from right_grasp onward).
    #
    # Per-phase config: (name, "planned" or "refresh", which target,
    # right_grip, left_grip, sim_steps).
    # right_pre_grasp swings the LEFT arm a long way (lift → handover);
    # bumping its step count gives the interpolation extra ticks so
    # joint velocities stay bounded and the closed-finger contact
    # holds the mug. Same for left_retract which moves left away from
    # the mug after release.
    right_phases = [
        ("right_pre_grasp", "planned", "pre", 0.0, 1.0, 400),
        ("right_grasp", "refresh", "grasp", 0.0, 1.0, 200),
        ("right_close", "refresh", "grasp", 1.0, 1.0, 100),
        ("left_open", "refresh", "grasp", 1.0, 0.0, 100),
        ("left_retract", "refresh", "retract", 1.0, 0.0, 250),
    ]
    last_right = last_left = None
    for name, source, which, gr, gl, n in right_phases:
        if source == "planned":
            rgw, rpw, lhw, lretw = (
                planned_right_grasp,
                planned_right_pre,
                planned_left_handover,
                planned_left_retract,
            )
        else:
            rgw, rpw, lhw, lretw = _refresh_right_targets()
            if rgw is None:
                rgw, rpw, lhw, lretw = (
                    planned_right_grasp,
                    planned_right_pre,
                    planned_left_handover,
                    planned_left_retract,
                )
        if which == "pre":
            r_target, l_target = rpw, lhw
        elif which == "grasp":
            r_target, l_target = rgw, lhw
        else:  # retract
            r_target, l_target = rgw, lretw
        last_right, last_left = r_target, l_target
        _print_phase(name, r_target, l_target, gr, gl)
        _viz_poses([r_target, l_target])
        _interp_step_n(env, cur_r, cur_l, r_target, l_target, gr, gl, n, device)
        cur_r, cur_l = r_target, l_target

    # ---- 7. Hold final state ----
    print("[handover] holding final state; idling...")
    final = _build_16d(last_right, last_left, 1.0, 0.0, device)
    while True:
        env.step(action=final.unsqueeze(0).repeat(env.num_envs, 1))


if __name__ == "__main__":
    main()
