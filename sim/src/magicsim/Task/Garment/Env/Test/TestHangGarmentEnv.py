"""Open-loop hang-garment driver on :class:`HangGarmentEnv`.

Mirrors :mod:`TestFoldEnv` / :mod:`TestFlingEnv` in shape: settle, refresh
keypoints, **visualize the grasp + lift waypoints** in the viewport, then
walk fixed-step phases that interpolate the wrist pose between waypoints.
Single-arm, action is 8D ``[pos(3), quat(4), grip(1)]`` consumed by the
franka cfg's ``ik_curobo`` arm action + ``binary`` gripper action.

Phase plan
----------
Grab the GARMENT by the midpoint of its ``top_left`` / ``top_right``
shoulders (the natural "collar" / hanger neck), lift, then carry toward
the rack and stop close to it (user spec — only the gesture matters; no
drop/release).

    settle      env.step(None) × 50  (cloth + rack settle)
    pre_grasp   neck + 0.15z, grip=0
    grasp       neck,         grip=0
    close       neck,         grip=1   (post-hook: garment.gravity_scale = 0)
    lift        neck + 0.15z, grip=1
    approach    near_hanger,  grip=1
    hold        last action — idle so the result stays on screen

Visualization (overrides any markers from :meth:`Garment.visualize_keypoint`)
- Red point  : grasp (neck midpoint)
- Blue point : lift (neck midpoint + 0.15 z)
- Green point: approach target (slightly back from the rack apex)
- Axes are drawn at each waypoint with the commanded ``[0, 1, 0, 0]`` quat.
"""

# Env import MUST come first — it triggers the Isaac Sim app bootstrap.
# Importing IsaacLab / omni / curobo modules before this raises
# ``ModuleNotFoundError: No module named 'omni'`` at import time.
from magicsim.Task.Garment.Env.HangGarmentEnv import HangGarmentEnv  # noqa: F401

import gymnasium as gym
import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig

from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes, draw_waypoints


SETTLE_STEPS = 50
# Aligned with TestFlingEnv + TestFoldEnv + atomic_skill/default.yaml:
#   GRIPPER_LENGTH       = 0.2     panda_hand → fingertip
#   PRE_REACH_DISTANCE   = 0.06    wrist 6 cm above the grasp wrist (Fold)
#   INSERTION_DEPTH      = 0.022   fingertip 22 mm below kp at grasp (Fling /
#                                  atomic_skill yaml — TestFoldEnv's 0.015 is
#                                  the open-loop outlier)
#   LIFT_HEIGHT          = 0.10    fingertip 10 cm above kp at lift (Fold;
#                                  Fling uses 0.28 for dynamic swing — too high
#                                  for a hang carry)
#
# Wrist-shift grasp → lift = LIFT_HEIGHT + INSERTION_DEPTH = 0.122.
PRE_OFFSET_Z = 0.06
LIFT_OFFSET_Z = 0.122

# Approach the rack apex but stop SHORT — never lower onto it. Wrist
# pulls back along world -x by APPROACH_PULLBACK_X and stays
# APPROACH_HOVER_Z above the apex (= same fingertip clearance as
# Fold's lift_height).
APPROACH_PULLBACK_X = 0.10
APPROACH_HOVER_Z = 0.10

WAYPOINT_POINT_SIZE = 14.0
WAYPOINT_AXIS_LENGTH = 0.06
WAYPOINT_LINE_THICKNESS = 2
WAYPOINT_LINE_OPACITY = 0.9


def _shift_z(pose7: torch.Tensor, dz: float) -> torch.Tensor:
    out = pose7.clone()
    out[2] += dz
    return out


def _shift_x(pose7: torch.Tensor, dx: float) -> torch.Tensor:
    out = pose7.clone()
    out[0] += dx
    return out


def _build_action(
    pose7: torch.Tensor, grip: float, device: torch.device
) -> torch.Tensor:
    grip_t = torch.tensor([grip], device=device, dtype=torch.float32)
    row = torch.cat([pose7.to(device).flatten()[:7], grip_t], dim=0)
    assert row.numel() == 8, f"expected 8D, got {row.numel()}"
    return row


def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: float) -> torch.Tensor:
    q0 = q0 / q0.norm().clamp_min(1e-8)
    q1 = q1 / q1.norm().clamp_min(1e-8)
    dot = (q0 * q1).sum()
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = (1.0 - alpha) * q0 + alpha * q1
        return out / out.norm().clamp_min(1e-8)
    theta_0 = torch.acos(dot.clamp(-1.0, 1.0))
    theta = theta_0 * alpha
    s0 = torch.cos(theta) - dot * torch.sin(theta) / torch.sin(theta_0)
    s1 = torch.sin(theta) / torch.sin(theta_0)
    return s0 * q0 + s1 * q1


def _interp_pose(p0: torch.Tensor, p1: torch.Tensor, alpha: float) -> torch.Tensor:
    pos = (1.0 - alpha) * p0[:3] + alpha * p1[:3]
    quat = _slerp(p0[3:7], p1[3:7], alpha)
    return torch.cat([pos, quat], dim=0)


def _interp_step_n(
    env: HangGarmentEnv,
    start: torch.Tensor,
    end: torch.Tensor,
    grip: float,
    steps: int,
    device: torch.device,
):
    for i in range(steps):
        alpha = (i + 1) / steps
        pose = _interp_pose(start, end, alpha)
        action = _build_action(pose, grip, device)
        env.step(action=action.unsqueeze(0).repeat(env.num_envs, 1))


def _print_phase(name: str, pose7: torch.Tensor, grip: float):
    p = pose7.detach().cpu().numpy().tolist()
    print(f"current phase: {name}")
    print(
        f"  eef xyz=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}] "
        f"wxyz=[{p[3]:+.3f},{p[4]:+.3f},{p[5]:+.3f},{p[6]:+.3f}] grip={grip}"
    )


def _collect_garments(env):
    scene_mgr = env.scene.scene_manager
    garments = []
    for env_id in range(env.num_envs):
        for _cat, glist in scene_mgr.garment_objects[env_id].items():
            garments.extend(glist)
    return garments


def _visualize_waypoints(
    pick_kp: torch.Tensor,
    grasp: torch.Tensor,
    lift: torch.Tensor,
    approach: torch.Tensor,
) -> None:
    """Show pick keypoint + wrist waypoints in the viewport.

    Mirrors TestFoldEnv: ``draw_waypoints`` paints the cloth-anchored
    pick keypoint red so it stays visible against the cloth, and the
    panda_hand wrist targets get colored markers (yellow grasp, blue
    lift, green approach). ``draw_grasp_samples_as_axes`` adds the
    wrist quaternion triad on top so the [0, 1, 0, 0] top-down
    orientation is visible too.
    """
    pick_pt = pick_kp[:3].detach().cpu().numpy().tolist()
    grasp_pt = grasp[:3].detach().cpu().numpy().tolist()
    lift_pt = lift[:3].detach().cpu().numpy().tolist()
    approach_pt = approach[:3].detach().cpu().numpy().tolist()
    draw_waypoints(
        [pick_pt],
        point_size=WAYPOINT_POINT_SIZE,
        color=(1.0, 0.0, 0.0, 1.0),  # red — raw pick keypoint on cloth
        clear_existing=True,
    )
    draw_waypoints(
        [grasp_pt],
        point_size=WAYPOINT_POINT_SIZE,
        color=(1.0, 0.85, 0.1, 1.0),  # yellow — wrist grasp (offset above pick)
        clear_existing=False,
    )
    draw_waypoints(
        [lift_pt],
        point_size=WAYPOINT_POINT_SIZE,
        color=(0.0, 0.4, 1.0, 1.0),  # blue — wrist lift
        clear_existing=False,
    )
    draw_waypoints(
        [approach_pt],
        point_size=WAYPOINT_POINT_SIZE,
        color=(0.0, 1.0, 0.3, 1.0),  # green — wrist approach (near hanger)
        clear_existing=False,
    )
    pose_tensor = torch.stack([grasp, lift, approach], dim=0).detach().cpu()
    draw_grasp_samples_as_axes(
        pose_tensor,
        axis_length=WAYPOINT_AXIS_LENGTH,
        line_thickness=WAYPOINT_LINE_THICKNESS,
        line_opacity=WAYPOINT_LINE_OPACITY,
        clear_existing=True,
    )


@hydra.main(version_base=None, config_path="../../Conf", config_name="hang_garment_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: HangGarmentEnv = gym.make(
        "HangGarmentEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    device = env.device

    # ---- Settle ----
    print(f"[hang_garment] settling scene ({SETTLE_STEPS} steps)...")
    for _ in range(SETTLE_STEPS):
        env.step(action=None)

    # ---- Keypoints ----
    garments = _collect_garments(env)
    if not garments:
        log.error("No garment found in scene; aborting.")
        return
    g = garments[0]
    g.update_keypoint()
    g.visualize_keypoint()

    pick_kp = env.get_garment_top_keypoint(env_id=0)
    grasp = env.get_garment_top_grasp(env_id=0)
    if pick_kp is None or grasp is None:
        log.error(
            "Could not derive top grasp — garment is missing top_left/top_right "
            "keypoints. Got: {}",
            list(env.get_keypoint_positions(0).keys()),
        )
        return
    log.info(
        "[hang_garment] pick keypoint xyz=({:+.3f},{:+.3f},{:+.3f}) "
        "→ wrist grasp xyz=({:+.3f},{:+.3f},{:+.3f}) "
        "(gripper_len={:.3f}, insertion={:.3f})",
        *pick_kp.cpu().tolist(),
        *grasp[:3].cpu().tolist(),
        env.gripper_length,
        env.insertion_depth,
    )

    hanger_top = env.get_hanger_top_pose(env_id=0)
    if hanger_top is None:
        log.error("Could not read hanger pose — rack missing from geometry_objects.")
        return
    log.info(
        "[hang_garment] hanger top wrist xyz=({:+.3f},{:+.3f},{:+.3f})",
        *hanger_top[:3].cpu().tolist(),
    )

    # ---- Waypoints (all are panda_hand WRIST poses; ik_abs commands wrist) ----
    pre_grasp = _shift_z(grasp, PRE_OFFSET_Z)
    lift = _shift_z(grasp, LIFT_OFFSET_Z)
    # "Stop short" — pull back along world -x and stay above the apex so
    # the gripper never collides with the rack. Same y as the apex so
    # the gesture clearly walks toward the hanger.
    approach = _shift_z(_shift_x(hanger_top, -APPROACH_PULLBACK_X), APPROACH_HOVER_Z)

    # ---- Visualize raw pick + planned wrist waypoints ----
    _visualize_waypoints(pick_kp, grasp, lift, approach)

    # Track last commanded wrist pose so each phase interpolates from
    # there — same pattern as TestFoldEnv (avoids ik_diff saturation when
    # wrist commands jump far in a single tick).
    cur = pre_grasp.clone()
    # Phase plan: (name, target, grip, steps, action_after_phase).
    # Step counts aligned with TestFoldEnv (pre_reach=80, reach=80,
    # close=80, lift=200) — the previous 200/200/100 was just slow.
    phases = [
        ("pre_grasp", pre_grasp, 0.0, 80, None),
        ("grasp", grasp, 0.0, 80, None),
        (
            "close",
            grasp,
            1.0,
            80,
            lambda: env.set_garment_gravity_scale(0.0),
        ),
        ("lift", lift, 1.0, 200, None),
        ("approach", approach, 1.0, 240, None),
    ]
    for name, target, grip, n_steps, after in phases:
        _print_phase(name, target, grip)
        _interp_step_n(env, cur, target, grip, n_steps, device)
        cur = target.clone()
        if after is not None:
            print(f"[hang_garment] post-{name} hook firing.")
            after()

    # ---- Hold — stop near hanger (user: "靠近hanger停住就行了") ----
    print("[hang_garment] reached approach target; holding.")
    final = _build_action(cur, 1.0, device)
    while True:
        env.step(action=final.unsqueeze(0).repeat(env.num_envs, 1))


if __name__ == "__main__":
    main()
