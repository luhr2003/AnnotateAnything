"""Keypoint-drag fling experiment (no robot motion).

Drives the cloth alone by editing only the two sleeve-keypoint vertex
positions through the lift → fling → drop trajectory. Each frame::

    mp = garment.get_current_mesh_points()    # full vertex array, world-space
    mp[idx_left]  = lerped_left_target
    mp[idx_right] = lerped_right_target
    garment.set_current_mesh_points(mp, ...)  # write the whole array back
    env.step(action=None)                     # advance physics one tick

Other vertices are copied through unchanged so the cloth can deform
freely under gravity / spring constraints, while the two pinned indices
follow our scripted trajectory. Robot stays at home pose throughout.

Phase plan (kp-driven xyz, world frame; same shape as TestFlingEnv):

    lift_up        : kp_z += LIFT_HEIGHT                      (interp LIFT_STEPS)
    fling_forward  : x += FLING_DISTANCE,  z += FLING_APEX    (interp FLING_STEPS)
    drop           : x += DROP_DISTANCE,   z = DROP_HEIGHT    (interp DROP_STEPS)
    settle         : stop touching kp; sim only               (DROP_SETTLE_STEPS)
"""

from typing import Tuple

from magicsim.Task.Garment.Env.FlingEnv import FlingEnv  # noqa: F401 (gym register)
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig
from loguru import logger as log

from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_waypoints


# ----- trajectory knobs (mirror TestFlingEnv) ----------------------------

LIFT_HEIGHT = 0.28  # world-z rise during lift_up (relative to start)
FLING_DISTANCE = -0.10  # world-x translation during fling_forward
FLING_APEX = 0.10  # extra world-z at fling apex (on top of LIFT_HEIGHT)
DROP_DISTANCE = -0.20  # world-x at drop (more forward than fling)
DROP_HEIGHT = 0.05  # world-z at drop relative to start kp z

LIFT_STEPS = 240
FLING_STEPS = 120
DROP_STEPS = 120
DROP_SETTLE_STEPS = 200

# Number of ``env.step(action=None)`` calls to run after every
# ``set_current_mesh_points`` write. 0 = no sim between sets (back-to-
# back overrides; useful to confirm whether the write actually sticks).
# >0 = give the cloth constraint solver time to propagate the pinned
# vertex displacement.
SIM_STEPS_PER_SET = 0


# ----- helpers ------------------------------------------------------------


def _collect_garments(env):
    scene_mgr = env.scene.scene_manager
    out = []
    for env_id in range(env.num_envs):
        for _cat, glist in scene_mgr.garment_objects[env_id].items():
            out.extend(glist)
    return out


def _resolve_arm_kp_indices(
    garment, name_a: str = "top_left", name_b: str = "top_right"
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Return ``(idx_left_arm, idx_right_arm, pos_left, pos_right)`` with
    L/R assignment by world Y (larger Y → left arm)."""
    garment.update_keypoint()
    indices = getattr(garment, "_keypoint_indices", None)
    if not indices:
        raise RuntimeError("garment has no _keypoint_indices after update_keypoint()")
    if name_a not in indices or name_b not in indices:
        raise RuntimeError(
            f"keypoints {name_a}/{name_b} missing; got {list(indices.keys())}"
        )
    idx_a = int(indices[name_a])
    idx_b = int(indices[name_b])
    mp, _, _, _ = garment.get_current_mesh_points()
    mp = np.asarray(mp, dtype=np.float32)
    pos_a = mp[idx_a].copy()
    pos_b = mp[idx_b].copy()
    if pos_a[1] >= pos_b[1]:
        idx_left, idx_right = idx_a, idx_b
        pos_left, pos_right = pos_a, pos_b
        l_label, r_label = name_a, name_b
    else:
        idx_left, idx_right = idx_b, idx_a
        pos_left, pos_right = pos_b, pos_a
        l_label, r_label = name_b, name_a
    print(
        f"[kp-drag] L_arm ← '{l_label}' idx={idx_left} pos={pos_left.tolist()}\n"
        f"[kp-drag] R_arm ← '{r_label}' idx={idx_right} pos={pos_right.tolist()}"
    )
    return idx_left, idx_right, pos_left, pos_right


def _draw_marker(left_xyz: np.ndarray, right_xyz: np.ndarray, color):
    draw_waypoints(
        [left_xyz.tolist(), right_xyz.tolist()],
        point_size=14.0,
        color=color,
        clear_existing=True,
    )


def _step_and_set(
    env,
    garment,
    indices: Tuple[int, int],
    target_left_world: np.ndarray,
    target_right_world: np.ndarray,
):
    """Read full mesh, override the two kp indices, write back, then
    advance physics ``SIM_STEPS_PER_SET`` times so the constraint
    solver propagates the pinned-vertex displacement to the rest of
    the cloth before the next override.

    Targets are supplied in WORLD frame. CPU/GPU dispatch:
        - GPU: ``set_current_mesh_points`` takes WORLD vertices via
          ``_cloth_prim_view.set_world_positions``. We read world,
          modify world, write world.
        - CPU: ``set_current_mesh_points`` writes the USD ``points``
          attr in LOCAL frame and then sets the prim's world pose.
          We need to express the kp targets in the prim's local frame
          before writing.
    """
    transformed_world, mesh_local, pos_world, ori_world = (
        garment.get_current_mesh_points()
    )
    is_gpu = pos_world is None  # GPU path returns None for these
    if is_gpu:
        mp = np.asarray(transformed_world, dtype=np.float32).copy()
        mp[indices[0]] = target_left_world
        mp[indices[1]] = target_right_world
        garment.set_current_mesh_points(mp, None, None)
    else:
        # Convert world-frame targets into the prim's local frame.
        pw_np = (
            pos_world.detach().cpu().numpy()
            if hasattr(pos_world, "detach")
            else np.asarray(pos_world)
        )
        ow_np = (
            ori_world.detach().cpu().numpy()
            if hasattr(ori_world, "detach")
            else np.asarray(ori_world)
        )
        try:
            sw_np = garment.get_world_scale()
            sw_np = (
                sw_np.detach().cpu().numpy()
                if hasattr(sw_np, "detach")
                else np.asarray(sw_np)
            )
        except Exception:
            sw_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        local_l = _world_to_local_point(target_left_world, pw_np, ow_np, sw_np)
        local_r = _world_to_local_point(target_right_world, pw_np, ow_np, sw_np)
        mp_local = np.asarray(mesh_local, dtype=np.float32).copy()
        mp_local[indices[0]] = local_l
        mp_local[indices[1]] = local_r
        garment.set_current_mesh_points(mp_local, pw_np, ow_np)
    for _ in range(int(1)):
        env.step(action=None)


def _quat_rotate_inverse(quat_wxyz: np.ndarray, vec3: np.ndarray) -> np.ndarray:
    """Rotate ``vec3`` by the inverse of ``quat_wxyz`` (= conjugate)."""
    w, x, y, z = quat_wxyz
    # q_inv = (w, -x, -y, -z); apply via standard quat-vec rotation.
    qx, qy, qz = -x, -y, -z
    qw = w
    vx, vy, vz = vec3
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + qw*t + cross(q.xyz, t)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return np.array([rx, ry, rz], dtype=np.float32)


def _world_to_local_point(
    p_world: np.ndarray,
    pos_world: np.ndarray,
    ori_world_wxyz: np.ndarray,
    scale_world: np.ndarray,
) -> np.ndarray:
    """Inverse of ``transform_points``: world → local."""
    rel = np.asarray(p_world, dtype=np.float32) - np.asarray(
        pos_world, dtype=np.float32
    )
    rotated = _quat_rotate_inverse(np.asarray(ori_world_wxyz, dtype=np.float32), rel)
    sw = np.asarray(scale_world, dtype=np.float32)
    safe = np.where(np.abs(sw) < 1e-8, 1.0, sw)
    return (rotated / safe).astype(np.float32)


def _lerp_phase(
    env,
    garment,
    indices: Tuple[int, int],
    start_l: np.ndarray,
    start_r: np.ndarray,
    end_l: np.ndarray,
    end_r: np.ndarray,
    n_steps: int,
    label: str,
    color,
    log_every: int = 30,
):
    print(f"[kp-drag] phase={label} steps={n_steps}")
    for k in range(n_steps):
        alpha = (k + 1) / max(1, n_steps)
        cur_l = start_l + alpha * (end_l - start_l)
        cur_r = start_r + alpha * (end_r - start_r)
        if k == 0 or (k + 1) % log_every == 0:
            _draw_marker(cur_l, cur_r, color)
            print(
                f"  {label} step {k + 1}/{n_steps} alpha={alpha:.2f} "
                f"L={cur_l.tolist()} R={cur_r.tolist()}"
            )
        _step_and_set(env, garment, indices, cur_l, cur_r)


# ----- main ---------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../Conf", config_name="fling_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: FlingEnv = gym.make("FlingEnv-V0", config=cfg, cli_args=None, logger=logger)
    env.reset()

    print("[kp-drag] settling cloth (50 steps)...")
    for _ in range(50):
        env.step(action=None)

    garments = _collect_garments(env)
    if not garments:
        raise RuntimeError("no garments found in scene")
    garment = garments[0]

    idx_l, idx_r, start_l, start_r = _resolve_arm_kp_indices(garment)

    # Endpoints relative to the captured kp start.
    lift_l = start_l + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)
    lift_r = start_r + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32)

    fling_l = lift_l + np.array([FLING_DISTANCE, 0.0, FLING_APEX], dtype=np.float32)
    fling_r = lift_r + np.array([FLING_DISTANCE, 0.0, FLING_APEX], dtype=np.float32)

    drop_l = start_l + np.array([DROP_DISTANCE, 0.0, DROP_HEIGHT], dtype=np.float32)
    drop_r = start_r + np.array([DROP_DISTANCE, 0.0, DROP_HEIGHT], dtype=np.float32)

    print(
        f"[kp-drag] targets:\n"
        f"  lift  L={lift_l.tolist()}  R={lift_r.tolist()}\n"
        f"  fling L={fling_l.tolist()} R={fling_r.tolist()}\n"
        f"  drop  L={drop_l.tolist()}  R={drop_r.tolist()}"
    )

    _lerp_phase(
        env,
        garment,
        (idx_l, idx_r),
        start_l,
        start_r,
        lift_l,
        lift_r,
        LIFT_STEPS,
        "lift_up",
        (0.2, 0.9, 0.2, 0.9),
        log_every=40,
    )
    _lerp_phase(
        env,
        garment,
        (idx_l, idx_r),
        lift_l,
        lift_r,
        fling_l,
        fling_r,
        FLING_STEPS,
        "fling_forward",
        (0.2, 0.6, 1.0, 0.9),
        log_every=30,
    )
    _lerp_phase(
        env,
        garment,
        (idx_l, idx_r),
        fling_l,
        fling_r,
        drop_l,
        drop_r,
        DROP_STEPS,
        "drop",
        (0.9, 0.4, 0.9, 0.9),
        log_every=30,
    )

    print(
        f"[kp-drag] phase=settle (no kp override, sim only) steps={DROP_SETTLE_STEPS}"
    )
    for _ in range(DROP_SETTLE_STEPS):
        env.step(action=None)

    print("[kp-drag] all phases done; idling...")
    while True:
        env.step(action=None)


if __name__ == "__main__":
    main()
