"""Bimanual Fold test on GarmentFoldEnv (open-loop, no MoveL / curobo).

Structurally mirrors :mod:`TestFlingEnv`: top-level tuning knobs, world-Y
arm assignment, and inter-phase linear interpolation. Fold-specific bits
are the phase plan (sleeve fold → bottom-to-shoulder fold) and the place
geometry (reflection across the shoulder→bottom line for the sleeve
placement, shoulder targets for the bottom placement).

The robot's ``ik_dual_diff`` action does the world→arm-base transform
internally using ``R_panda_link0`` / ``L_panda_link0`` as ref bodies, so
poses are supplied in world frame — no motion planning, no IK server.

Phase plan mirrors :mod:`magicsim.Collect.AtomicSkill.Fold`::

    sleeve : reach → close → lift → move → drop → open → retract
    bottom : reach → close → lift → move → drop → open → retract

Timing is open-loop: fixed step counts per phase, no convergence checks.
"""

from magicsim.Task.Garment.Env.GarmentFoldEnv import GarmentFoldEnv
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

# Fingertip targets (added to keypoint z). ``GRIPPER_LENGTH`` is added
# on top of each so that the panda_hand wrist lands gripper-length above
# the fingertip target.
LIFT_HEIGHT = 0.1  # fingertip height above pick keypoint after grasp
DROP_HEIGHT = 0.03  # fingertip height above place target when releasing
RETRACT_HEIGHT = 0.15  # fingertip clearance above place point after release
GRIPPER_LENGTH = 0.2  # panda_hand wrist → fingertip distance, applied along the per-arm approach direction

# Insertion knobs. With a slanted gripper, the fingertip slides in along
# the approach direction (gripper +Z) instead of pressing straight down.
# ``PRE_REACH_DISTANCE`` is how far back along the approach the gripper
# starts; ``INSERTION_DEPTH`` is how far past the keypoint the fingertip
# pushes once the slide-in finishes (positive = into the cloth).
PRE_REACH_DISTANCE = 0.06
INSERTION_DEPTH = 0.015

# Keypoint adjustments before commanding the arms (mirror Fling's knobs).
INWARD_SHIFT = 0.05  # pull each pick along LR segment toward midline (m)
GRASP_KP_X_SHIFT = 0.02  # push each pick along world +X (m)

# When the sleeve reflection / bottom-to-shoulder fold would drive the two
# wrists across the body midline (or too close to each other), the place
# targets are clamped so each arm stays on its own side of the midline
# with at least ``MIN_LR_SEPARATION`` total Y-gap. Prevents the two
# panda_hands from colliding around the body center.
MIN_LR_SEPARATION = 0.20  # meters along world Y between L and R targets

# Standard single-Franka home-pose panda_hand world quat (gripper forward).
GRASP_QUAT_STD = [0.0, 1.0, 0.0, 0.0]
LEFT_ARM_YAW_DEG = -90.0  # L_panda_link0 yaw in world (dual_franka.urdf)
RIGHT_ARM_YAW_DEG = 90.0  # R_panda_link0 yaw in world

# Slant the gripper instead of pointing straight down. ``TILT_DEG`` is an
# Rx rotation in each arm's ROOT-LINK frame composed BEFORE the
# top-down quat, so positive values tip the approach direction forward
# along that arm's reach (out toward its keypoint side). Applied to every
# phase — reach / lift / move / drop / retract — so the wrist orientation
# stays consistent throughout the fold.
TILT_DEG = 45.0  # degrees

# Settle steps before keypoint detection.
SETTLE_STEPS = 50

# ----- garment physics overrides (test-only; yaml stays untouched) -----
# After ``env.reset()`` builds the cloth with the values from
# ``garment_fold_env.yaml`` (stretch_stiffness=1e12, adhesion=10, etc.),
# this test overrides the relevant knobs in-place via Isaac Sim setters
# / USD attribute writes so we can iterate without touching the
# checked-in scene yaml. Set any entry to ``None`` to leave the yaml
# value alone for that field.
GARMENT_PHYSICS_OVERRIDES = {
    # ParticleMaterial
    "adhesion": 0.0,  # was 10 — table no longer holds the cloth during lift
    "particle_adhesion_scale": 0.5,  # was 3.0
    # garment_config (cloth springs + mass) — written via USD attributes
    # on the cloth prim because SingleClothPrim doesn't expose setters
    # for all of these.
    "particle_mass": 1e-3,  # was 5e-11
    "stretch_stiffness": 1e4,  # was 1e12 — let the cloth stretch under tension
    "bend_stiffness": 1e2,  # was 1e3
    "shear_stiffness": 1e2,  # was 1e3
}


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


def _tilt_arm_local(q_arm_local: List[float], tilt_deg: float) -> List[float]:
    """Slant an arm-local quat: ``Rx(tilt_deg) ⊗ q_arm_local``.

    Composed BEFORE ``_rotate_world_quat_by_yaw``. Positive ``tilt_deg``
    tips the approach direction forward toward the arm's reach (the
    panda_hand's down-pointing Z axis rotates toward arm-local +Y).
    """
    half = math.radians(tilt_deg) * 0.5
    q_tilt = [math.cos(half), math.sin(half), 0.0, 0.0]  # Rx(tilt_deg)
    return _quat_mul(q_tilt, list(q_arm_local))


def _approach_dir_from_quat(q: List[float]) -> np.ndarray:
    """Rotate (0,0,1) by world quat (wxyz) → unit gripper +Z (approach).

    For Franka's ``panda_hand``, +Z points from wrist to between the
    fingertips, i.e. the approach direction. With straight top-down
    grasp this is (0,0,-1); with a 45° forward tilt it picks up a
    horizontal component along the arm's reach side.
    """
    w, x, y, z = q
    return np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float32,
    )


# ----- geometry helpers ----------------------------------------------------


def _reflect_across_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reflect ``p`` across the line through ``a`` and ``b`` in 3D."""
    d = b - a
    dd = float(np.dot(d, d))
    if dd < 1e-9:
        return 2.0 * a - p
    t = float(np.dot(p - a, d)) / dd
    foot = a + t * d
    return 2.0 * foot - p


def _shift_inward(left_xyz: np.ndarray, right_xyz: np.ndarray, dist: float):
    """Pull two opposite-side picks toward each other by ``dist`` meters."""
    lr = right_xyz - left_xyz
    lr_dist = float(np.linalg.norm(lr))
    if lr_dist < 1e-4 or dist <= 0.0:
        return left_xyz.copy(), right_xyz.copy()
    step = dist / lr_dist
    return left_xyz + lr * step, right_xyz - lr * step


def _clamp_place_to_sides(
    L_place: np.ndarray,
    R_place: np.ndarray,
    midline_y: float,
    min_separation: float,
):
    """Keep L_place at +Y side and R_place at -Y side of ``midline_y``.

    L_place must have y ≥ ``midline_y + min_separation/2``;
    R_place must have y ≤ ``midline_y - min_separation/2``. Anything
    that would cross past the midline (e.g. an over-aggressive sleeve
    reflection) gets snapped back to the half-separation boundary on
    its own side. XZ are left untouched.
    """
    half = float(min_separation) * 0.5
    L_out = L_place.copy()
    R_out = R_place.copy()
    L_min_y = float(midline_y) + half
    R_max_y = float(midline_y) - half
    if L_out[1] < L_min_y:
        L_out[1] = L_min_y
    if R_out[1] > R_max_y:
        R_out[1] = R_max_y
    return L_out, R_out


# ----- garment helpers -----------------------------------------------------


def _collect_garments(env):
    scene_mgr = env.scene.scene_manager
    garments = []
    for env_id in range(env.num_envs):
        for _cat, glist in scene_mgr.garment_objects[env_id].items():
            garments.extend(glist)
    return garments


def _override_garment_physics(garment, overrides: dict) -> None:
    """Apply runtime physics tweaks to a Garment after env.reset().

    The yaml-loaded values are baked into the cloth + particle material
    at construction time. This rewrites the live PhysX state via Isaac
    Sim's ``ParticleMaterial`` setters and ``SingleClothPrim`` per-cloth
    scalar helpers. Keys with value ``None`` are skipped (yaml wins).
    """
    pm = getattr(garment, "particle_material", None)

    def _try(label, fn):
        try:
            fn()
            print(f"[open-loop][physics-override] {label} = ok")
        except Exception as exc:
            print(f"[open-loop][physics-override] {label} FAILED: {exc}")

    # ---- ParticleMaterial scalars ----
    if pm is not None:
        if overrides.get("adhesion") is not None:
            _try(
                f"adhesion → {overrides['adhesion']}",
                lambda: pm.set_adhesion(float(overrides["adhesion"])),
            )
        if overrides.get("particle_adhesion_scale") is not None:
            _try(
                f"particle_adhesion_scale → {overrides['particle_adhesion_scale']}",
                lambda: pm.set_particle_adhesion_scale(
                    float(overrides["particle_adhesion_scale"])
                ),
            )

    # ---- Cloth spring stiffnesses (per-cloth scalar) ----
    if overrides.get("stretch_stiffness") is not None:
        _try(
            f"stretch_stiffness → {overrides['stretch_stiffness']}",
            lambda: garment.set_cloth_stretch_stiffness(
                float(overrides["stretch_stiffness"])
            ),
        )
    if overrides.get("bend_stiffness") is not None:
        _try(
            f"bend_stiffness → {overrides['bend_stiffness']}",
            lambda: garment.set_cloth_bend_stiffness(
                float(overrides["bend_stiffness"])
            ),
        )
    if overrides.get("shear_stiffness") is not None:
        _try(
            f"shear_stiffness → {overrides['shear_stiffness']}",
            lambda: garment.set_cloth_shear_stiffness(
                float(overrides["shear_stiffness"])
            ),
        )

    # ---- Particle mass (SingleClothPrim has no scalar setter; go
    # through the underlying multi-prim view with a (1, N) tensor) ----
    if overrides.get("particle_mass") is not None:
        target = float(overrides["particle_mass"])

        def _set_mass():
            view = getattr(garment, "_cloth_prim_view", None)
            if view is None:
                raise RuntimeError("garment._cloth_prim_view is None")
            pts = garment.prim.GetAttribute("points").Get()
            if pts is None or len(pts) == 0:
                raise RuntimeError("garment prim has no 'points' attribute")
            n = len(pts)
            masses = np.full((1, n), target, dtype=np.float32)
            view.set_particle_masses(masses)

        _try(f"particle_mass → {target} (per-particle, N={'?'} → broadcast)", _set_mass)


def _resolve_garment_side_mapping(
    kp: dict,
) -> Tuple[str, dict]:
    """Decide which garment side belongs to which robot arm.

    Garment labels (``*_left`` / ``*_right``) follow the cloth's own body
    frame and are NOT a safe match for the ROBOT's L/R arms. dual_franka
    has L_panda_link0 at world +Y and R_panda_link0 at -Y, so assign the
    keypoint with the larger world Y to the left arm. Decision is made
    once from ``top_*`` and applied consistently to bottom + shoulder.
    """
    tl = np.asarray(kp["top_left"], dtype=np.float32)
    tr = np.asarray(kp["top_right"], dtype=np.float32)
    if tl[1] >= tr[1]:
        # garment "left" side aligns with robot L arm
        labels_for_arm = {
            "L": ("top_left", "bottom_left", "left_shoulder"),
            "R": ("top_right", "bottom_right", "right_shoulder"),
        }
        decision = "garment_left → L_arm"
    else:
        labels_for_arm = {
            "L": ("top_right", "bottom_right", "right_shoulder"),
            "R": ("top_left", "bottom_left", "left_shoulder"),
        }
        decision = "garment_right → L_arm (flipped)"
    return decision, labels_for_arm


# ----- action / viz builders -----------------------------------------------


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
    pick_left: np.ndarray,
    pick_right: np.ndarray,
    right_xyz: np.ndarray,
    left_xyz: np.ndarray,
    right_quat: List[float],
    left_quat: List[float],
    phase_color: Tuple[float, float, float, float],
):
    """Red current-stage pick keypoints + phase-colored target points + axes."""
    draw_waypoints(
        [pick_left.tolist(), pick_right.tolist()],
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


@hydra.main(version_base=None, config_path="../../Conf", config_name="garment_fold_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: GarmentFoldEnv = gym.make(
        "GarmentFoldEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # ---- runtime physics overrides (test-only; yaml stays untouched) ----
    # Apply BEFORE the settle loop so the cloth comes to rest under the
    # tweaked params (lower stretch_stiffness, no table adhesion, sane
    # particle mass).
    pre_settle_garments = _collect_garments(env)
    if pre_settle_garments:
        print(
            f"[open-loop] applying physics overrides to {len(pre_settle_garments)} "
            f"garment(s): {GARMENT_PHYSICS_OVERRIDES}"
        )
        for g_pre in pre_settle_garments:
            _override_garment_physics(g_pre, GARMENT_PHYSICS_OVERRIDES)

    # ---- settle ----
    print(f"[open-loop] settling garment ({SETTLE_STEPS} steps)...")
    for _ in range(SETTLE_STEPS):
        env.step(action=None)

    # ---- collect garment + keypoints ----
    garments = _collect_garments(env)
    if not garments:
        raise RuntimeError("no garments found in scene")
    g = garments[0]
    g.update_keypoint()
    g.visualize_keypoint()
    kp = g.get_keypoint()
    required = (
        "top_left",
        "top_right",
        "left_shoulder",
        "right_shoulder",
        "bottom_left",
        "bottom_right",
    )
    missing = [k for k in required if k not in kp]
    if missing:
        raise RuntimeError(
            f"[open-loop] missing keypoints {missing}; got {list(kp.keys())}"
        )

    # ---- arm ↔ garment-side assignment (by world Y of top_*) ----
    decision, labels_for_arm = _resolve_garment_side_mapping(kp)
    L_top, L_bot, L_shoulder_lbl = labels_for_arm["L"]
    R_top, R_bot, R_shoulder_lbl = labels_for_arm["R"]
    L_pick_sleeve = np.asarray(kp[L_top], dtype=np.float32)
    R_pick_sleeve = np.asarray(kp[R_top], dtype=np.float32)
    L_pick_bottom = np.asarray(kp[L_bot], dtype=np.float32)
    R_pick_bottom = np.asarray(kp[R_bot], dtype=np.float32)
    L_shoulder = np.asarray(kp[L_shoulder_lbl], dtype=np.float32)
    R_shoulder = np.asarray(kp[R_shoulder_lbl], dtype=np.float32)
    print(f"[open-loop] {decision}")
    print(
        f"[open-loop] sleeve picks: L←'{L_top}' y={L_pick_sleeve[1]:.3f}; "
        f"R←'{R_top}' y={R_pick_sleeve[1]:.3f}"
    )
    print(
        f"[open-loop] bottom picks: L←'{L_bot}' y={L_pick_bottom[1]:.3f}; "
        f"R←'{R_bot}' y={R_pick_bottom[1]:.3f}"
    )

    # ---- place targets ----
    # Sleeve: reflect each pick across that side's shoulder→bottom line.
    L_place_sleeve = _reflect_across_line(L_pick_sleeve, L_shoulder, L_pick_bottom)
    R_place_sleeve = _reflect_across_line(R_pick_sleeve, R_shoulder, R_pick_bottom)
    # Bottom: place onto the same-side shoulder.
    L_place_bottom = L_shoulder.copy()
    R_place_bottom = R_shoulder.copy()

    # ---- clamp places so the two arms don't collide near the body midline ----
    sleeve_midline_y = 0.5 * (L_pick_sleeve[1] + R_pick_sleeve[1])
    bottom_midline_y = 0.5 * (L_pick_bottom[1] + R_pick_bottom[1])
    L_place_sleeve_raw, R_place_sleeve_raw = (
        L_place_sleeve.copy(),
        R_place_sleeve.copy(),
    )
    L_place_bottom_raw, R_place_bottom_raw = (
        L_place_bottom.copy(),
        R_place_bottom.copy(),
    )
    L_place_sleeve, R_place_sleeve = _clamp_place_to_sides(
        L_place_sleeve, R_place_sleeve, sleeve_midline_y, MIN_LR_SEPARATION
    )
    L_place_bottom, R_place_bottom = _clamp_place_to_sides(
        L_place_bottom, R_place_bottom, bottom_midline_y, MIN_LR_SEPARATION
    )
    print(
        f"[open-loop] sleeve places (raw → clamped) min_sep={MIN_LR_SEPARATION:.3f}: "
        f"L y {L_place_sleeve_raw[1]:.3f} → {L_place_sleeve[1]:.3f}; "
        f"R y {R_place_sleeve_raw[1]:.3f} → {R_place_sleeve[1]:.3f}"
    )
    print(
        f"[open-loop] bottom places (raw → clamped): "
        f"L y {L_place_bottom_raw[1]:.3f} → {L_place_bottom[1]:.3f}; "
        f"R y {R_place_bottom_raw[1]:.3f} → {R_place_bottom[1]:.3f}"
    )

    # ---- inward shift on picks (sleeve + bottom, independently) ----
    L_pick_sleeve_s, R_pick_sleeve_s = _shift_inward(
        L_pick_sleeve, R_pick_sleeve, INWARD_SHIFT
    )
    L_pick_bottom_s, R_pick_bottom_s = _shift_inward(
        L_pick_bottom, R_pick_bottom, INWARD_SHIFT
    )

    # ---- world +X nudge on picks (Fling-style grasp_kp_x_shift) ----
    if GRASP_KP_X_SHIFT != 0.0:
        for arr in (
            L_pick_sleeve_s,
            R_pick_sleeve_s,
            L_pick_bottom_s,
            R_pick_bottom_s,
        ):
            arr[0] += GRASP_KP_X_SHIFT
    print(
        f"[open-loop] picks (inward {INWARD_SHIFT:.3f} m + x_shift {GRASP_KP_X_SHIFT:+.3f} m): "
        f"sleeve L={L_pick_sleeve_s.tolist()} R={R_pick_sleeve_s.tolist()}; "
        f"bottom L={L_pick_bottom_s.tolist()} R={R_pick_bottom_s.tolist()}"
    )

    # ---- per-arm world-frame grasp orientations ----
    # arm-local top-down composed with TILT_DEG slant, then lifted into
    # world by each arm's root-link yaw.
    arm_local_q = _tilt_arm_local(GRASP_QUAT_STD, TILT_DEG)
    q_left = _rotate_world_quat_by_yaw(arm_local_q, LEFT_ARM_YAW_DEG)
    q_right = _rotate_world_quat_by_yaw(arm_local_q, RIGHT_ARM_YAW_DEG)
    print(f"[open-loop] tilt_deg={TILT_DEG} arm_local_q(post-tilt)={arm_local_q}")
    print(f"[open-loop] q_left={q_left}")
    print(f"[open-loop] q_right={q_right}")

    gl = GRIPPER_LENGTH
    appR = _approach_dir_from_quat(q_right)
    appL = _approach_dir_from_quat(q_left)
    print(f"[open-loop] approach_dir right={appR.tolist()} left={appL.tolist()}")

    z_lift = np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)
    z_drop = np.array([0.0, 0.0, DROP_HEIGHT], dtype=np.float32)
    z_retract = np.array([0.0, 0.0, RETRACT_HEIGHT], dtype=np.float32)

    def wrist(L_finger, R_finger):
        """fingertip → wrist via per-arm approach direction.

        ``wrist = fingertip - gripper_length * approach_dir``. Accounts
        for the gripper tilt: with TILT_DEG ≠ 0, the gripper_length is
        no longer purely vertical.
        """
        return L_finger - gl * appL, R_finger - gl * appR

    # ---- phase tables ----
    # Each entry: (name, left_target, right_target, gripper, n_steps,
    #              color, interp_from_xyz, viz_pick_pair)
    # interp_from_xyz: optional (left_start, right_start) — when set, the
    # target xyz linearly interpolates from start to target across n_steps.
    # viz_pick_pair: which (left, right) to draw as the red pick markers.
    color_pick_sleeve = (1.0, 0.8, 0.1, 0.9)  # yellow
    color_carry_sleeve = (0.2, 0.9, 0.2, 0.9)  # green
    color_place_sleeve = (0.9, 0.4, 0.9, 0.9)  # magenta
    color_pick_bottom = (1.0, 0.6, 0.1, 0.9)  # orange
    color_carry_bottom = (0.1, 0.8, 0.6, 0.9)  # teal
    color_place_bottom = (0.4, 0.4, 1.0, 0.9)  # blue

    sleeve_pair = (L_pick_sleeve, R_pick_sleeve)
    bottom_pair = (L_pick_bottom, R_pick_bottom)

    # ---- per-phase wrist targets ----
    # Sleeve stage: fingertip at pick (insertion along approach), then
    # straight up / over to place / down / release / retract.
    L_wrist_pre_sleeve, R_wrist_pre_sleeve = wrist(
        L_pick_sleeve_s + (INSERTION_DEPTH - PRE_REACH_DISTANCE) * appL,
        R_pick_sleeve_s + (INSERTION_DEPTH - PRE_REACH_DISTANCE) * appR,
    )
    L_wrist_reach_sleeve, R_wrist_reach_sleeve = wrist(
        L_pick_sleeve_s + INSERTION_DEPTH * appL,
        R_pick_sleeve_s + INSERTION_DEPTH * appR,
    )
    L_wrist_lift_sleeve, R_wrist_lift_sleeve = wrist(
        L_pick_sleeve_s + z_lift,
        R_pick_sleeve_s + z_lift,
    )
    L_wrist_move_sleeve, R_wrist_move_sleeve = wrist(
        L_place_sleeve + z_lift,
        R_place_sleeve + z_lift,
    )
    L_wrist_drop_sleeve, R_wrist_drop_sleeve = wrist(
        L_place_sleeve + z_drop,
        R_place_sleeve + z_drop,
    )
    L_wrist_retract_sleeve, R_wrist_retract_sleeve = wrist(
        L_place_sleeve + z_retract,
        R_place_sleeve + z_retract,
    )

    # Bottom stage analogues.
    L_wrist_pre_bottom, R_wrist_pre_bottom = wrist(
        L_pick_bottom_s + (INSERTION_DEPTH - PRE_REACH_DISTANCE) * appL,
        R_pick_bottom_s + (INSERTION_DEPTH - PRE_REACH_DISTANCE) * appR,
    )
    L_wrist_reach_bottom, R_wrist_reach_bottom = wrist(
        L_pick_bottom_s + INSERTION_DEPTH * appL,
        R_pick_bottom_s + INSERTION_DEPTH * appR,
    )
    L_wrist_lift_bottom, R_wrist_lift_bottom = wrist(
        L_pick_bottom_s + z_lift,
        R_pick_bottom_s + z_lift,
    )
    L_wrist_move_bottom, R_wrist_move_bottom = wrist(
        L_place_bottom + z_lift,
        R_place_bottom + z_lift,
    )
    L_wrist_drop_bottom, R_wrist_drop_bottom = wrist(
        L_place_bottom + z_drop,
        R_place_bottom + z_drop,
    )
    L_wrist_retract_bottom, R_wrist_retract_bottom = wrist(
        L_place_bottom + z_retract,
        R_place_bottom + z_retract,
    )

    # Gripper convention (MagicSim MultipleBinaryJointAction):
    #   action >= 1 → close (fingers 0.00)   action < 1 → open (fingers 0.04)
    phases = [
        # ---- sleeve fold ----
        # pre_reach: hover back along the approach direction so the
        # next step is a pure slide-in along approach (insertion feel).
        (
            "pre_reach_sleeve",
            L_wrist_pre_sleeve,
            R_wrist_pre_sleeve,
            0.0,
            80,
            color_pick_sleeve,
            None,
            sleeve_pair,
        ),
        # reach: slide forward along approach into the cloth.
        (
            "reach_sleeve",
            L_wrist_reach_sleeve,
            R_wrist_reach_sleeve,
            0.0,
            80,
            color_pick_sleeve,
            (L_wrist_pre_sleeve, R_wrist_pre_sleeve),
            sleeve_pair,
        ),
        (
            "close_gripper_sleeve",
            L_wrist_reach_sleeve,
            R_wrist_reach_sleeve,
            1.0,
            80,
            color_pick_sleeve,
            None,
            sleeve_pair,
        ),
        (
            "lift_sleeve",
            L_wrist_lift_sleeve,
            R_wrist_lift_sleeve,
            1.0,
            200,
            color_carry_sleeve,
            (L_wrist_reach_sleeve, R_wrist_reach_sleeve),
            sleeve_pair,
        ),
        (
            "move_sleeve",
            L_wrist_move_sleeve,
            R_wrist_move_sleeve,
            1.0,
            240,
            color_carry_sleeve,
            (L_wrist_lift_sleeve, R_wrist_lift_sleeve),
            sleeve_pair,
        ),
        (
            "drop_sleeve",
            L_wrist_drop_sleeve,
            R_wrist_drop_sleeve,
            1.0,
            120,
            color_place_sleeve,
            (L_wrist_move_sleeve, R_wrist_move_sleeve),
            sleeve_pair,
        ),
        (
            "open_gripper_sleeve",
            L_wrist_drop_sleeve,
            R_wrist_drop_sleeve,
            0.0,
            40,
            color_place_sleeve,
            None,
            sleeve_pair,
        ),
        (
            "retract_sleeve",
            L_wrist_retract_sleeve,
            R_wrist_retract_sleeve,
            0.0,
            120,
            color_place_sleeve,
            (L_wrist_drop_sleeve, R_wrist_drop_sleeve),
            sleeve_pair,
        ),
        # ---- bottom-to-shoulder fold ----
        (
            "pre_reach_bottom",
            L_wrist_pre_bottom,
            R_wrist_pre_bottom,
            0.0,
            80,
            color_pick_bottom,
            None,
            bottom_pair,
        ),
        (
            "reach_bottom",
            L_wrist_reach_bottom,
            R_wrist_reach_bottom,
            0.0,
            80,
            color_pick_bottom,
            (L_wrist_pre_bottom, R_wrist_pre_bottom),
            bottom_pair,
        ),
        (
            "close_gripper_bottom",
            L_wrist_reach_bottom,
            R_wrist_reach_bottom,
            1.0,
            80,
            color_pick_bottom,
            None,
            bottom_pair,
        ),
        (
            "lift_bottom",
            L_wrist_lift_bottom,
            R_wrist_lift_bottom,
            1.0,
            200,
            color_carry_bottom,
            (L_wrist_reach_bottom, R_wrist_reach_bottom),
            bottom_pair,
        ),
        (
            "move_bottom",
            L_wrist_move_bottom,
            R_wrist_move_bottom,
            1.0,
            240,
            color_carry_bottom,
            (L_wrist_lift_bottom, R_wrist_lift_bottom),
            bottom_pair,
        ),
        (
            "drop_bottom",
            L_wrist_drop_bottom,
            R_wrist_drop_bottom,
            1.0,
            120,
            color_place_bottom,
            (L_wrist_move_bottom, R_wrist_move_bottom),
            bottom_pair,
        ),
        (
            "open_gripper_bottom",
            L_wrist_drop_bottom,
            R_wrist_drop_bottom,
            0.0,
            40,
            color_place_bottom,
            None,
            bottom_pair,
        ),
        (
            "retract_bottom",
            L_wrist_retract_bottom,
            R_wrist_retract_bottom,
            0.0,
            120,
            color_place_bottom,
            (L_wrist_drop_bottom, R_wrist_drop_bottom),
            bottom_pair,
        ),
    ]

    # ---- run phases ----
    for (
        name,
        left_xyz,
        right_xyz,
        grip,
        n_steps,
        color,
        interp_from,
        pick_pair,
    ) in phases:
        print(
            f"[open-loop] phase={name} right_xyz={right_xyz.tolist()} "
            f"left_xyz={left_xyz.tolist()} gripper={grip} steps={n_steps} "
            f"interp={'yes' if interp_from is not None else 'no'}"
        )
        _draw_phase_viz(
            pick_pair[0],
            pick_pair[1],
            right_xyz,
            left_xyz,
            q_right,
            q_left,
            color,
        )
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
