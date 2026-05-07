"""Open-loop Fling test on FlingEnv (no MoveL, no curobo).

Loads the dual_franka + garment scene, settles the cloth, picks the sleeve
keypoints (``top_left`` / ``top_right``), and drives the two arms phase by
phase by pushing a 16D world-frame action tensor (14D dual arm + 2D binary
eef) straight into ``env.step``. The robot's ``ik_dual_diff`` action does
the world→arm-base transform internally using
``R_panda_link0`` / ``L_panda_link0`` as ref bodies, so we supply poses in
world frame — no motion planning, no IK server in the loop.

Phase plan mirrors :mod:`magicsim.Collect.AtomicSkill.Fling`::

    reach → close_gripper → lift → fling_forward → drop → open_gripper

but timing is open-loop (fixed step counts per phase, no convergence
checks). Useful for eyeballing frame conventions / gripper length /
``grasp_quat`` choices without spinning up the GlobalPlanner stack.
"""

from magicsim.Task.Garment.Env.FlingEnv import FlingEnv
import math
from typing import List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from loguru import logger as log

from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes, draw_waypoints  # noqa: F401 (gym register)


# ----- open-loop tuning knobs (match Collect/Conf/atomic_skill/default.yaml) -----

# Fingertip targets (added to keypoint z). ``GRIPPER_LENGTH`` is added on
# top of each so that the ``panda_hand`` wrist lands gripper-length above
# the fingertip target.
REACH_Z_OFFSET = -0.022  # fingertip pressed below keypoint so fingers sink in
LIFT_HEIGHT = 0.28  # fingertip height above pick keypoint after grasp
FLING_DISTANCE = -0.10  # X translation during fling (world): toward neckline (-X)
FLING_APEX = 0.1  # extra height at fling apex
DROP_HEIGHT = 0.05  # fingertip height above pick keypoint at drop
# Drop continues past the fling end-point along -X (further toward the
# neckline) and downward — the arms lay the cloth out on the table
# instead of releasing in mid-air above the fling target.
DROP_DISTANCE = -0.20  # X translation at drop (world): more forward than fling
GRIPPER_LENGTH = 0.2  # panda_hand → fingertip (added to every phase's z)

# Shift each sleeve keypoint toward the other sleeve (i.e. toward the
# garment midline) by this much before commanding the arms. Raw
# ``top_left`` / ``top_right`` sit right at the sleeve tip / seam edge,
# where the two fingers can close on empty air; pulling the grasp point
# inward lands the fingers on fabric body instead of the hem.
INWARD_SHIFT = 0.05  # meters toward midpoint along the LR segment

# Extra X-axis nudge of the grasp anchor along world +X — i.e. AWAY
# from the neckline (which is at -X), toward the bottom hem of the
# shirt. Applied as ``kp[0] += shift``.
GRASP_KP_X_SHIFT = 0.02

# Standard single-Franka home-pose panda_hand world quat (gripper forward).
GRASP_QUAT_STD = [0.0, 1.0, 0.0, 0.0]
LEFT_ARM_YAW_DEG = -90.0  # L_panda_link0 yaw in world (dual_franka.urdf)
RIGHT_ARM_YAW_DEG = 90.0  # R_panda_link0 yaw in world

# Per-phase garment gravity scaling. Cloth stays quasi-suspended through
# the entire dynamic motion (lift → fling → drop) so the fabric trails
# the gripper smoothly along the diagonal-down drop trajectory; gravity
# is restored at open_gripper so the cloth settles onto the table.
GRAVITY_SCALE_DEFAULT = 1.0
GRAVITY_SCALE_FLING = 0.05
PHASES_WITH_LOW_GRAVITY = {"lift_up", "fling_forward", "drop"}


# ----- quaternion helpers --------------------------------------------------


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _rotate_world_quat_by_yaw(q_std: List[float], yaw_deg: float) -> List[float]:
    """``q_world = Rz(yaw_deg) ⊗ q_std`` — same convention as Fling skill."""
    half = math.radians(yaw_deg) * 0.5
    q_yaw = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return _quat_mul(q_yaw, list(q_std))


# ----- garment helpers -----------------------------------------------------


def _collect_garments(env):
    scene_mgr = env.scene.scene_manager
    garments = []
    for env_id in range(env.num_envs):
        for _cat, glist in scene_mgr.garment_objects[env_id].items():
            garments.extend(glist)
    return garments


def _pick_sleeve_keypoints(
    garment, left_name: str = "top_left", right_name: str = "top_right"
) -> Tuple[np.ndarray, np.ndarray]:
    garment.update_keypoint()
    garment.visualize_keypoint()
    kp = garment.get_keypoint()
    if left_name not in kp or right_name not in kp:
        raise RuntimeError(
            f"keypoints {left_name}/{right_name} missing; got {list(kp.keys())}"
        )
    # Garment labels (``top_left`` / ``top_right``) follow the garment's
    # own body frame and are NOT a safe match for the ROBOT's L/R arms.
    # dual_franka has L_panda_link0 at world +Y and R_panda_link0 at -Y,
    # so assign the sleeve keypoint with the larger world Y to the left
    # arm and the smaller one to the right arm.
    kp_a = np.asarray(kp[left_name], dtype=np.float32)
    kp_b = np.asarray(kp[right_name], dtype=np.float32)
    if kp_a[1] >= kp_b[1]:
        left_arm_kp, right_arm_kp = kp_a, kp_b
        left_label, right_label = left_name, right_name
    else:
        left_arm_kp, right_arm_kp = kp_b, kp_a
        left_label, right_label = right_name, left_name
    print(
        f"[open-loop] arm ↔ garment-kp (by world Y): "
        f"L_arm (+Y) ← '{left_label}' @ y={left_arm_kp[1]:.3f}; "
        f"R_arm (-Y) ← '{right_label}' @ y={right_arm_kp[1]:.3f}"
    )
    return left_arm_kp, right_arm_kp


# ----- action builders -----------------------------------------------------


def _build_16d_action(
    right_xyz: np.ndarray,
    left_xyz: np.ndarray,
    right_quat: List[float],
    left_quat: List[float],
    gripper_close: float,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    """Assemble a (num_envs, 16) tensor: [right_7, left_7, grip_r, grip_l]."""
    right_pose = list(right_xyz.tolist()) + list(right_quat)
    left_pose = list(left_xyz.tolist()) + list(left_quat)
    arm_16d = right_pose + left_pose + [float(gripper_close), float(gripper_close)]
    assert len(arm_16d) == 16, f"expected 16 dims, got {len(arm_16d)}"
    row = torch.tensor(arm_16d, dtype=torch.float32, device=device)
    return row.unsqueeze(0).repeat(num_envs, 1)


def _draw_phase_viz(
    left_kp: np.ndarray,
    right_kp: np.ndarray,
    right_xyz: np.ndarray,
    left_xyz: np.ndarray,
    right_quat: List[float],
    left_quat: List[float],
    phase_color: Tuple[float, float, float, float],
):
    """Red pick keypoints + phase-colored target points + target axes."""
    draw_waypoints(
        [left_kp.tolist(), right_kp.tolist()],
        point_size=14.0,
        color=(1.0, 0.0, 0.0, 1.0),
        clear_existing=True,
    )
    draw_waypoints(
        [right_xyz.tolist(), left_xyz.tolist()],
        point_size=10.0,
        color=phase_color,
        clear_existing=False,
    )
    pose_tensor = torch.tensor(
        [
            list(right_xyz.tolist()) + list(right_quat),
            list(left_xyz.tolist()) + list(left_quat),
        ],
        dtype=torch.float32,
    )
    draw_grasp_samples_as_axes(
        pose_tensor,
        axis_length=0.06,
        line_thickness=2,
        line_opacity=0.9,
        clear_existing=True,
    )


# ----- main ----------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../Conf", config_name="fling_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: FlingEnv = gym.make("FlingEnv-V0", config=cfg, cli_args=None, logger=logger)
    env.reset()

    # Settle the cloth onto the table before snapping keypoints.
    print("[open-loop] settling garment (50 steps)...")
    for _ in range(50):
        env.step(action=None)

    garments = _collect_garments(env)
    if not garments:
        raise RuntimeError("no garments found in scene")
    left_kp_raw, right_kp_raw = _pick_sleeve_keypoints(garments[0])
    print(
        f"[open-loop] sleeve keypoints (raw): left={left_kp_raw.tolist()} "
        f"right={right_kp_raw.tolist()}"
    )

    # Nudge each sleeve keypoint toward the midline so the fingers land on
    # fabric body instead of the sleeve-tip seam.
    midpoint = (left_kp_raw + right_kp_raw) * 0.5
    lr_vec = right_kp_raw - left_kp_raw
    lr_dist = float(np.linalg.norm(lr_vec))
    if lr_dist > 1e-4 and INWARD_SHIFT > 0:
        step = INWARD_SHIFT / lr_dist  # fraction of LR segment
        left_kp = left_kp_raw + lr_vec * step
        right_kp = right_kp_raw - lr_vec * step
    else:
        left_kp, right_kp = left_kp_raw.copy(), right_kp_raw.copy()
    # Push each grasp anchor along world +X (away from the neckline,
    # toward the shirt's bottom hem).
    if GRASP_KP_X_SHIFT != 0.0:
        left_kp[0] += GRASP_KP_X_SHIFT
        right_kp[0] += GRASP_KP_X_SHIFT
    print(
        f"[open-loop] grasp points (shifted {INWARD_SHIFT:.3f} m inward LR, "
        f"x {GRASP_KP_X_SHIFT:+.3f} m): "
        f"left={left_kp.tolist()} right={right_kp.tolist()}"
    )

    # Per-arm world-frame grasp orientations.
    q_left = _rotate_world_quat_by_yaw(GRASP_QUAT_STD, LEFT_ARM_YAW_DEG)
    q_right = _rotate_world_quat_by_yaw(GRASP_QUAT_STD, RIGHT_ARM_YAW_DEG)
    print(f"[open-loop] q_left={q_left}")
    print(f"[open-loop] q_right={q_right}")

    gl = GRIPPER_LENGTH
    reach_off = np.array([0.0, 0.0, REACH_Z_OFFSET + gl], dtype=np.float32)
    lift_off = np.array([0.0, 0.0, LIFT_HEIGHT + gl], dtype=np.float32)
    fling_off = np.array(
        [FLING_DISTANCE, 0.0, LIFT_HEIGHT + FLING_APEX + gl], dtype=np.float32
    )
    drop_off = np.array([DROP_DISTANCE, 0.0, DROP_HEIGHT + gl], dtype=np.float32)

    # Phase = (name, left_target, right_target, gripper, n_steps, color, interp_from_xyz)
    # interp_from_xyz: optional (left_start, right_start) — when set, the
    # target xyz linearly interpolates from (start) to (target) across
    # n_steps so the arms ramp into the goal instead of yanking.
    LIFT_FROM = (left_kp + reach_off, right_kp + reach_off)
    FLING_FROM = (left_kp + lift_off, right_kp + lift_off)
    DROP_FROM = (left_kp + fling_off, right_kp + fling_off)
    phases = [
        # Gripper convention (MagicSim MultipleBinaryJointAction,
        # src/magicsim/Env/Robot/mdp/actions.py:2239):
        #   action >= 1 → close_command_expr  (fingers 0.00)
        #   action <  1 → open_command_expr   (fingers 0.04)
        (
            "reach",
            left_kp + reach_off,
            right_kp + reach_off,
            0.0,
            60,
            (1.0, 0.8, 0.1, 0.9),
            None,
        ),
        (
            "close_gripper",
            left_kp + reach_off,
            right_kp + reach_off,
            1.0,
            80,
            (1.0, 0.8, 0.1, 0.9),
            None,
        ),
        (
            "lift_up",
            left_kp + lift_off,
            right_kp + lift_off,
            1.0,
            240,
            (0.2, 0.9, 0.2, 0.9),
            LIFT_FROM,
        ),
        (
            "fling_forward",
            left_kp + fling_off,
            right_kp + fling_off,
            1.0,
            120,
            (0.2, 0.6, 1.0, 0.9),
            FLING_FROM,
        ),
        (
            "drop",
            left_kp + drop_off,
            right_kp + drop_off,
            1.0,
            120,
            (0.9, 0.4, 0.9, 0.9),
            DROP_FROM,
        ),
        (
            "open_gripper",
            left_kp + drop_off,
            right_kp + drop_off,
            0.0,
            20,
            (0.9, 0.4, 0.9, 0.9),
            None,
        ),
    ]

    # Cache garment particle materials so we can flip gravity_scale per phase.
    particle_materials = []
    for g in garments:
        pm = getattr(g, "particle_material", None)
        if pm is not None:
            particle_materials.append(pm)
    print(
        f"[open-loop] cached {len(particle_materials)} particle_material(s) "
        "for dynamic gravity_scale"
    )

    current_grav_scale = None

    def _set_gravity_scale(value: float):
        nonlocal current_grav_scale
        if current_grav_scale == value:
            return
        for pm in particle_materials:
            pm.set_gravity_scale(float(value))
        print(f"[open-loop]   gravity_scale -> {value}")
        current_grav_scale = value

    for name, left_xyz, right_xyz, grip, n_steps, color, interp_from in phases:
        print(
            f"[open-loop] phase={name} right_xyz={right_xyz.tolist()} "
            f"left_xyz={left_xyz.tolist()} gripper={grip} steps={n_steps} "
            f"interp={'yes' if interp_from is not None else 'no'}"
        )
        _set_gravity_scale(
            GRAVITY_SCALE_FLING
            if name in PHASES_WITH_LOW_GRAVITY
            else GRAVITY_SCALE_DEFAULT
        )
        _draw_phase_viz(left_kp, right_kp, right_xyz, left_xyz, q_right, q_left, color)
        if interp_from is None:
            action = _build_16d_action(
                right_xyz=right_xyz,
                left_xyz=left_xyz,
                right_quat=q_right,
                left_quat=q_left,
                gripper_close=grip,
                num_envs=env.num_envs,
                device=env.device,
            )
            for _ in range(n_steps):
                env.step(action=action)
        else:
            left_start, right_start = interp_from
            for k in range(n_steps):
                # alpha 0→1 across the phase; clamp to 1.0 on the final step.
                alpha = (k + 1) / max(1, n_steps)
                cur_left = left_start + (left_xyz - left_start) * alpha
                cur_right = right_start + (right_xyz - right_start) * alpha
                action = _build_16d_action(
                    right_xyz=cur_right,
                    left_xyz=cur_left,
                    right_quat=q_right,
                    left_quat=q_left,
                    gripper_close=grip,
                    num_envs=env.num_envs,
                    device=env.device,
                )
                env.step(action=action)

    print("[open-loop] all phases done; idling...")
    while True:
        env.step(action=None)


if __name__ == "__main__":
    main()
