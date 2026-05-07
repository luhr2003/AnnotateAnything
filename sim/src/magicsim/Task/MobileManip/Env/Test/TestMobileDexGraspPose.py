"""
MobileDexGraspEnv: sim + watch bottle pose + IK goalset + visualize the
selected sharpa grasp pose. NO arm motion — base / arm action stay at
NaN (p-controller lock_skip → parked-base hold + pink IK live FK
fallback) so the bottle settles undisturbed and we can see whether the
IK actually picks a sane candidate after the seed_config anchoring fix.

Mirrors :file:`TestMobileGraspPose.py` (parallel gripper) but adapted
for the sharpa dex annotation:
  * candidates are dicts with ``coarse_grasp / fine_grasp / final_grasp``,
    not flat 7-D poses; we use ``coarse_grasp`` for the goalset.
  * IK server is :class:`DualIKServer` with ``eef_num=2``; we pack the
    right-arm-only goalset via the same NaN-on-inactive-slot trick the
    skill code uses (``pack_single_arm_goalset``).
  * Re-runs goalset when the bottle moves > 1mm from reference AND has
    stabilised < 0.1mm vs the previous tick (object knocked, then
    settled).
"""

from typing import List, Optional, Tuple
import torch
from magicsim.Task.MobileManip.Env.MobileDexGraspEnv import MobileDexGraspEnv
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

TARGET_OBJ_NAME = "bottle"
SETTLE_STEPS = 50
HAND_ID = 0  # right wrist (R_ee)


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


def _resolve_robot_name(env: MobileDexGraspEnv) -> str:
    robot_manager = getattr(env.scene, "robot_manager", None)
    if robot_manager is None:
        raise RuntimeError("robot_manager not available.")
    robot_dict = getattr(robot_manager, "robots", None)
    if isinstance(robot_dict, dict) and robot_dict:
        return next(iter(robot_dict.keys()))
    raise RuntimeError("Unable to resolve robot_name.")


def _get_robot_state_dict(env: MobileDexGraspEnv) -> dict:
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


def _get_ik_server(env: MobileDexGraspEnv):
    """Single-server-per-robot layout (MERGE_LEFT_RIGHT.md §1–§8)."""
    pm = getattr(env.scene, "planner_manager", None)
    if pm is None:
        return None
    ik_dict = getattr(pm, "ik_server", None) or {}
    if not ik_dict:
        return None
    robot_name = _resolve_robot_name(env)
    return ik_dict.get(robot_name)


def _term_dim(space) -> int:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"Cannot determine dim for space {space!r}")
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[-1])


def _build_neutral_action(env: MobileDexGraspEnv) -> dict:
    """Dict-of-dict action that keeps the parked vega from moving:
    base = NaN (p-controller lock_skip), arm = NaN (pink IK FK fallback),
    eef = 0 (sharpa fingers open). One row per env."""
    device = env.device
    n = env.num_envs
    actions: dict = {}
    pm = env.scene.planner_manager
    for robot_name, robot_space in pm.single_action_space.spaces.items():
        per_robot: dict = {}
        for term_name, term_space in robot_space.spaces.items():
            dim = _term_dim(term_space)
            if term_name == "eef_action":
                vec = torch.zeros((n, dim), device=device, dtype=torch.float32)
            else:
                vec = torch.full(
                    (n, dim), float("nan"), device=device, dtype=torch.float32
                )
            per_robot[term_name] = vec
        actions[robot_name] = per_robot
    return actions


def _flat_candidates(env_dict: dict) -> list:
    """Flatten functional_grasp + grasp parts the same way DexGrasp +
    TestMobileDexGrasp do, so ``selected_idx`` shares the same
    ordering."""
    flat: list = []
    for top_key in ("functional_grasp", "grasp"):
        parts = env_dict.get(top_key, {})
        if not isinstance(parts, dict):
            continue
        for part_list in parts.values():
            if isinstance(part_list, list):
                flat.extend(part_list)
    return flat


def _phase_poses(candidates: list, phase: str, device) -> Optional[torch.Tensor]:
    """Extract a 7-D pose from each candidate's ``phase`` (``coarse_grasp``
    / ``fine_grasp`` / ``final_grasp``) and stack to ``(G, 7)``."""
    poses = []
    for c in candidates:
        ph = c.get(phase)
        if ph is None:
            continue
        pos = ph["position"]
        ori = ph["orientation"]
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos, dtype=torch.float32, device=device)
        if not isinstance(ori, torch.Tensor):
            ori = torch.tensor(ori, dtype=torch.float32, device=device)
        poses.append(
            torch.cat([pos.flatten()[:3], ori.flatten()[:4]], dim=0).to(device)
        )
    if not poses:
        return None
    return torch.stack(poses, dim=0)


def _pack_single_arm_goalset(
    arm_poses: torch.Tensor, hand_id: int, eef_num: int
) -> torch.Tensor:
    """Mirror :meth:`AtomicSkill.pack_single_arm_goalset`. For ``eef_num=2``
    builds ``(1, G, 14)`` with the active hand's poses in slot ``hand_id``
    and NaN in the other slot (Server flips that slot's
    :class:`ToolPoseCriteria` to ``disabled()`` per env)."""
    if arm_poses.ndim == 2:
        arm_poses = arm_poses.unsqueeze(0)
    N, G, _ = arm_poses.shape
    if eef_num == 1:
        return arm_poses.contiguous()
    target = torch.full(
        (N, G, eef_num, 7),
        float("nan"),
        device=arm_poses.device,
        dtype=arm_poses.dtype,
    )
    target[:, :, hand_id, :] = arm_poses
    return target.reshape(N, G, eef_num * 7).contiguous()


def _ik_select_best(
    env: MobileDexGraspEnv,
    arm_poses: torch.Tensor,
    env_id: int = 0,
) -> Tuple[bool, int]:
    """Submit a goalset IK on the right arm (free base) and return
    (success, selected_index). Returns ``(False, 0)`` if the server
    times out or fails."""
    ik_server = _get_ik_server(env)
    if ik_server is None:
        log.warning("[ik goalset] no IK server (planner.ik.enable=true?).")
        return False, 0
    rs = _get_robot_state_dict(env)
    eef_num = int(getattr(ik_server, "eef_num", 1))
    is_dual = getattr(ik_server, "dual_mode", False)
    target = _pack_single_arm_goalset(
        arm_poses.to(env.device), hand_id=HAND_ID, eef_num=eef_num
    )
    log.info(
        "[ik goalset] submit env_ids={} dual={} eef_num={} G={} hand_id={}",
        [env_id],
        is_dual,
        eef_num,
        arm_poses.shape[0],
        HAND_ID,
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
        "[ik goalset] success={} idx={} env_ids={}",
        success_list,
        goalset_index_list,
        returned_env_ids,
    )
    if not returned_env_ids or int(returned_env_ids[0]) != int(env_id):
        return False, 0
    if not success_list or not bool(success_list[0]):
        return False, 0
    selected_idx = -1
    if goalset_index_list is not None and len(goalset_index_list) >= 1:
        selected_idx = int(goalset_index_list[0])
    if selected_idx < 0:
        return False, 0
    return True, selected_idx


def _get_target_object_pose_7d(env: MobileDexGraspEnv) -> Optional[torch.Tensor]:
    """Target object pose [7] in world frame. Picks ``target_obj_name``
    if set, otherwise the first non-desk rigid object."""
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
    env: MobileDexGraspEnv,
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


# Which annotation phase to use as the IK goalset target.
# ``final_grasp`` = the closed-hand end-pose; usually a tight wrap
# directly on the bottle surface, which constrains the IK far more than
# the open-hand pre-roll ``coarse_grasp`` and tends to pick a more
# physically sensible candidate. Switch back to ``"coarse_grasp"`` if
# the fingers' contact pose is too restrictive for the right arm to
# reach.
_GOALSET_PHASE = "final_grasp"


def apply_goalset_and_visualize(env: MobileDexGraspEnv, device: torch.device) -> bool:
    """Pull all sharpa ``_GOALSET_PHASE`` poses, run IK goalset, viz the
    selected pose. Returns True iff a candidate was selected."""
    grasp_list = env.get_grasp_pose(env_ids=[0], hand_type="sharpa")
    env_dict = grasp_list[0] if grasp_list else None
    if env_dict is None:
        log.error("[goalset] no sharpa annotation on target.")
        return False
    candidates = _flat_candidates(env_dict)
    arm_poses = _phase_poses(candidates, _GOALSET_PHASE, device)
    if arm_poses is None or arm_poses.shape[0] == 0:
        log.error("[goalset] no {} poses available.", _GOALSET_PHASE)
        return False

    ok, idx = _ik_select_best(env, arm_poses)
    if not ok or idx < 0 or idx >= arm_poses.shape[0]:
        log.warning("[goalset] IK failed; falling back to candidate 0.")
        idx = 0

    selected_pose = arm_poses[idx]
    log.info(
        "[viz] phase={} selected idx={}/{} pose={}",
        _GOALSET_PHASE,
        idx,
        arm_poses.shape[0],
        [f"{v:+.3f}" for v in selected_pose.tolist()],
    )
    visualize_grasp_pose([selected_pose])
    return True


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="mobile_dex_grasp_env"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: MobileDexGraspEnv = gym.make(
        "MobileDexGraspEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    device = env.device

    neutral_action = _build_neutral_action(env)

    obj_ref: Optional[torch.Tensor] = None
    obj_prev: Optional[torch.Tensor] = None

    # Settle the bottle on the desk.
    for _ in range(SETTLE_STEPS):
        env.step(action=neutral_action)

    # Push obstacles to the IK / motiongen worlds, ignoring the target.
    env.scene.planner_manager.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        obstacle_ignore_path_list=[TARGET_OBJ_NAME],
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
