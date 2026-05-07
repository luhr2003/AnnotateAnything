"""
Open-loop dex-grasp drive on Vega 1P Sharpa, modeled on the
:class:`DexGrasp` atomic skill but without GlobalPlanner / phase
truncation feedback. The full grasp is replayed phase-by-phase with a
fixed step budget per phase:

  1. settle the scene,
  2. flatten all sharpa coarse_grasp candidates → ``(G, 7)`` goalset,
  3. push the world to the IK server (ignore the bottle, same as
     ``DexGrasp.reset``),
  4. submit one goalset IK on the right arm (free base, lock_base=False),
  5. commit the chosen candidate's coarse / fine / final poses + joints,
  6. derive ``pre_grasp`` (back along approach axis, 0.05m) and
     ``retrieval`` (z-up, 0.20m) — same offsets DexGrasp uses,
  7. open-loop drive R_ee through ``[pre, coarse, fine, final, retrieval]``
     via pink IK, fingers open during arm reach, fully closed once we
     enter ``final_grasp`` and held closed through ``retrieval``.

Per-phase action layout (mirrors ``TestMobileDualReachEnv``):
``base_action`` = NaN (p-controller lock_skip → parked-base hold),
``arm_action`` = ``[right(7) | left(7)]`` with left = NaN (pink IK
falls back to live L_ee FK), ``eef_action`` = open / close vector
across the 44 sharpa fingers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import gymnasium as gym
import hydra
from omegaconf import DictConfig
from loguru import logger as log

from magicsim.Task.MobileManip.Env.MobileDexGraspEnv import MobileDexGraspEnv
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest

from pxr import Gf


SETTLE_STEPS = 50
TARGET_OBJ_NAME = "bottle"

PRE_GRASP_OFFSET = 0.05  # m, back along approach axis (DexGrasp default)
RETRIEVAL_OFFSET = 0.20  # m, z-up (DexGrasp default)
APPROACH_AXIS = 1  # axis 1 = y of the grasp frame (DexGrasp:611)

# Step budget per phase — open-loop replacement for the GlobalPlanner
# ``finished`` signal that DexGrasp normally relies on. Tuned generously
# so pink IK has time to converge before phase advances.
PHASE_STEPS: Dict[str, int] = {
    "pre_grasp": 50,
    "coarse_grasp": 50,
    "fine_grasp": 50,
    "final_grasp": 50,
    "retrieval": 50,
}

# Sharpa hand: 22 right + 22 left = 44 finger DOFs. Vega's eef term is
# wired with ``joint_names=_R_HAND_JOINT_NAMES + _L_HAND_JOINT_NAMES`` and
# ``preserve_order=True`` (vega1psharpa.py), so ``eef_action[0:22]`` is
# the right hand in the annotation's ``OUTPUT_JOINT_ORDER`` —
# ``final_grasp.joints`` (22-vec, right) drops in directly.
HAND_DIM_PER = 22
HAND_DIM_TOTAL = 2 * HAND_DIM_PER

_POSE_DIM = 7
_DUAL_POSE_DIM = 2 * _POSE_DIM

AXIS_LENGTH = 0.05
LINE_THICKNESS = 3
LINE_OPACITY = 0.8


def _viz(poses: list[torch.Tensor]) -> None:
    poses_l = [p.cpu().numpy().tolist() for p in poses]
    pose_pairs = [
        (Gf.Vec3d(p[0], p[1], p[2]), Gf.Quatd(p[3], p[4], p[5], p[6])) for p in poses_l
    ]
    draw_grasp_samples_as_axes(
        grasp_poses=pose_pairs,
        axis_length=AXIS_LENGTH,
        line_thickness=LINE_THICKNESS,
        line_opacity=LINE_OPACITY,
        clear_existing=True,
    )


def _term_dim(space) -> int:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"Cannot determine dim for space {space!r}")
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[-1])


def _flat_candidates(env_dict: dict) -> list:
    """Same flat candidate ordering DexGrasp uses (functional → grasp)."""
    flat: list = []
    for top_key in ("functional_grasp", "grasp"):
        parts = env_dict.get(top_key, {})
        if not isinstance(parts, dict):
            continue
        for part_list in parts.values():
            if isinstance(part_list, list):
                flat.extend(part_list)
    return flat


def _phase_to_pose7(phase: dict | None, device) -> Optional[torch.Tensor]:
    if phase is None:
        return None
    pos = phase["position"]
    ori = phase["orientation"]
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, dtype=torch.float32, device=device)
    if not isinstance(ori, torch.Tensor):
        ori = torch.tensor(ori, dtype=torch.float32, device=device)
    return torch.cat([pos.flatten()[:3], ori.flatten()[:4]], dim=0).to(device)


def _phase_joints(
    phase: dict | None, dim: int, device, dtype=torch.float32
) -> torch.Tensor:
    """Read the ``joints`` array off a phase dict (coarse/fine/final), pad
    or truncate to ``dim``. Mirrors ``TestBiDexGraspEnv._phase_joints``."""
    if phase is None:
        return torch.zeros(dim, dtype=dtype, device=device)
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


def _pack_right_hand_action(right_joints: torch.Tensor, device) -> torch.Tensor:
    """Pack right-hand 22-vec into the 44-dim sharpa eef vector.
    ``eef_action[0:22] = right (OUTPUT_JOINT_ORDER)``,
    ``eef_action[22:44] = left = 0`` (open / unused on this task)."""
    left = torch.zeros(HAND_DIM_PER, dtype=right_joints.dtype, device=device)
    return torch.cat([right_joints.to(device), left], dim=0)


# Which annotation phase to feed into the goalset IK selector. The
# IK only uses this to pick ``selected_idx``; the per-phase driving
# loop afterwards still pulls coarse / fine / final from
# ``candidates[selected_idx]`` regardless. ``final_grasp`` is the
# closed-hand contact pose — usually a tighter geometric constraint
# than ``coarse_grasp``, so the goalset reachability filter prunes
# more "physically silly" candidates (palm reversed, side flipped).
# Switch back to ``"coarse_grasp"`` if ``final_grasp`` is too
# restrictive for the right arm to reach.
_GOALSET_PHASE = "final_grasp"


def _phase_poses(candidates: list, phase: str, device) -> Optional[torch.Tensor]:
    """Stack the 7-D pose of each candidate's ``phase`` into ``(G, 7)``."""
    poses = []
    for c in candidates:
        p = _phase_to_pose7(c.get(phase), device)
        if p is not None:
            poses.append(p)
    if not poses:
        return None
    return torch.stack(poses, dim=0)


def _compute_pre_grasp(coarse_pose: torch.Tensor, offset: float) -> torch.Tensor:
    """Pre-grasp = back off along the grasp frame's approach axis (y).
    Mirrors ``DexGrasp._compute_pose_along_grasp_direction`` with
    ``backward=True``."""
    pos = coarse_pose[:3]
    quat = coarse_pose[3:7]
    R = quat_to_rot_matrix(quat.unsqueeze(0))[0]
    approach = R[:, APPROACH_AXIS]
    approach = approach / approach.norm().clamp_min(1e-8)
    return torch.cat([pos - approach * offset, quat], dim=0)


def _compute_retrieval(final_pose: torch.Tensor, offset: float) -> torch.Tensor:
    """Retrieval = lift z-up while keeping the final orientation.
    Mirrors ``DexGrasp._compute_pose_upward``."""
    pos = final_pose[:3].clone()
    pos[2] += offset
    return torch.cat([pos, final_pose[3:7]], dim=0)


def _pack_single_arm_goalset(
    arm_poses: torch.Tensor, hand_id: int, eef_num: int
) -> torch.Tensor:
    """Mirror :meth:`AtomicSkill.pack_single_arm_goalset` for non-skill callers."""
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


def _solve_goalset(
    env: MobileDexGraspEnv,
    ik_server,
    robot_name: str,
    env_id: int,
    hand_id: int,
    arm_poses: torch.Tensor,
) -> int:
    """Submit one goalset IK (free base) and block on the result."""
    robot_state = env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
    if isinstance(robot_state, dict):
        robot_state = next(iter(robot_state.values()))
    robot_states_dict = {
        "base_pos": robot_state["base_pos"],
        "base_quat": robot_state["base_quat"],
        "joint_pos": robot_state["joint_pos"],
        "joint_vel": robot_state["joint_vel"],
    }
    eef_num = int(getattr(ik_server, "eef_num", 1))
    is_dual = getattr(ik_server, "dual_mode", False)
    target = _pack_single_arm_goalset(arm_poses, hand_id=hand_id, eef_num=eef_num)
    # ``lock_base=True`` matches the actual sim behaviour: the parked
    # vega's base stays at (-1, 0, 0.1) for the whole open-loop replay
    # (p-controller gets all-NaN → lock_skip every step). Submitting
    # with lock_base=False let cuRobo solve as if the base could drift,
    # so it picked candidates only reachable when the base shifts in y
    # — and pink IK then had to contort the right arm to reach a +y
    # / "wrong-side-of-bottle" target from the actual fixed base.
    # locked_solver also ignores the dummy_base joints in cspace, so
    # the IK only varies torso + arms.
    log.info(
        "[goalset submit] robot={} env_id={} hand_id={} eef_num={} dual={} mode=goalset lock_base=True G={}",
        robot_name,
        env_id,
        hand_id,
        eef_num,
        is_dual,
        arm_poses.shape[0],
    )
    if is_dual:
        req = DualIKPlanRequest(
            env_ids=[env_id],
            target_pos=target,
            robot_states=robot_states_dict,
            mode="goalset",
            lock_base=False,
        )
    else:
        req = IKPlanRequest(
            env_ids=[env_id],
            target_pos=target,
            robot_states=robot_states_dict,
            mode="goalset",
        )
    fut = ik_server.submit_ik(req)
    success_list, goalset_index_list, returned_env_ids = fut.result()
    log.info(
        "[goalset result] success={} idx={} env_ids={}",
        success_list,
        goalset_index_list,
        returned_env_ids,
    )
    if not returned_env_ids or int(returned_env_ids[0]) != env_id:
        return -1
    if not success_list or not bool(success_list[0]):
        return -1
    if goalset_index_list is None or len(goalset_index_list) < 1:
        return -1
    return int(goalset_index_list[0])


def _build_phase_action(
    env: MobileDexGraspEnv,
    right_pose_world: torch.Tensor,
    hand_action: torch.Tensor,
) -> Dict[str, Any]:
    """Per-step pink IK action: ``[right(7) | NaN(7)]`` arm, NaN base,
    ``hand_action`` (44-vec) across the sharpa fingers. The eef term's
    width is read from ``single_action_space``; pad with 0 / truncate if
    it doesn't match ``hand_action`` exactly."""
    device = env.device
    n = env.num_envs
    actions: dict = {}
    planner_manager = env.scene.planner_manager
    for robot_name, robot_space in planner_manager.single_action_space.spaces.items():
        per_robot: dict = {}
        for term_name, term_space in robot_space.spaces.items():
            dim = _term_dim(term_space)
            if term_name == "base_action":
                vec = torch.full(
                    (n, dim), float("nan"), device=device, dtype=torch.float32
                )
            elif term_name == "arm_action":
                if dim == _DUAL_POSE_DIM:
                    arm = torch.full(
                        (dim,), float("nan"), device=device, dtype=torch.float32
                    )
                    arm[:_POSE_DIM] = right_pose_world.to(
                        device=device, dtype=torch.float32
                    )
                    vec = arm.unsqueeze(0).repeat(n, 1)
                else:
                    vec = torch.full(
                        (n, dim), float("nan"), device=device, dtype=torch.float32
                    )
            elif term_name == "eef_action":
                ha = hand_action.to(device=device, dtype=torch.float32).flatten()
                if ha.numel() == dim:
                    eef = ha
                elif ha.numel() > dim:
                    eef = ha[:dim].contiguous()
                else:
                    pad = torch.zeros(
                        dim - ha.numel(), device=device, dtype=torch.float32
                    )
                    eef = torch.cat([ha, pad], dim=0)
                vec = eef.unsqueeze(0).repeat(n, 1)
            else:
                vec = torch.full(
                    (n, dim), float("nan"), device=device, dtype=torch.float32
                )
            per_robot[term_name] = vec
        actions[robot_name] = per_robot
    return actions


def _log_pose(tag: str, pose: torch.Tensor) -> None:
    R = quat_to_rot_matrix(pose[3:7].unsqueeze(0))[0]
    palm = R[:, 0] / R[:, 0].norm().clamp_min(1e-8)
    fingers = R[:, 2] / R[:, 2].norm().clamp_min(1e-8)
    log.info(
        "[{}] xyz=({:+.3f},{:+.3f},{:+.3f}) quat=({:+.3f},{:+.3f},{:+.3f},{:+.3f}) "
        "palm=({:+.2f},{:+.2f},{:+.2f}) fingers=({:+.2f},{:+.2f},{:+.2f})",
        tag,
        *pose[:3].tolist(),
        *pose[3:7].tolist(),
        *palm.tolist(),
        *fingers.tolist(),
    )


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
    env_id = 0
    hand_id = 0  # right wrist (R_ee)

    # 1. Settle bottle on desk before sampling the annotation.
    for _ in range(SETTLE_STEPS):
        env.sim_step()

    # 2. Pull all sharpa candidates and extract the chosen-phase poses
    # (default ``final_grasp``) for the goalset IK input.
    grasp_list = env.get_grasp_pose(env_ids=[env_id], hand_type="sharpa")
    env_dict = grasp_list[0] if grasp_list else None
    if env_dict is None:
        log.error("No sharpa annotation on target object.")
        return
    candidates = _flat_candidates(env_dict)
    goalset_poses = _phase_poses(candidates, _GOALSET_PHASE, device)
    if goalset_poses is None or goalset_poses.shape[0] == 0:
        log.error("No {} poses available.", _GOALSET_PHASE)
        return

    pm = env.scene.planner_manager
    ik_dict = getattr(pm, "ik_server", {}) or {}
    if not ik_dict:
        log.error(
            "IKServer not available — set planner.ik.enable: true in "
            "single_vega1pSharpa.yaml."
        )
        return
    robot_name, ik_server = next(iter(ik_dict.items()))

    # 3. Push obstacles to IK / motiongen world, ignoring the target bottle.
    pm.update_obstacles(
        obstacle_avoidance_path_list=["dynamic"],
        # obstacle_ignore_path_list=[TARGET_OBJ_NAME],
        env_ids=[env_id],
    )

    # 4. Goalset IK → cuRobo picks the reachable candidate (filter on
    # ``_GOALSET_PHASE``; idx is then re-used to pull all five phase
    # poses from ``candidates[selected_idx]`` in the loop below).
    selected_idx = _solve_goalset(
        env, ik_server, robot_name, env_id, hand_id, goalset_poses
    )
    if selected_idx < 0:
        log.error("IK goalset found no reachable {} candidate.", _GOALSET_PHASE)
        return
    log.info(
        "[ik goalset] phase={} selected idx={}/{} (robot={})",
        _GOALSET_PHASE,
        selected_idx,
        goalset_poses.shape[0],
        robot_name,
    )

    hand_open = torch.zeros(HAND_DIM_TOTAL, dtype=torch.float32, device=device)

    # 5. Open-loop phase loop with REACTIVE refresh.
    # On entry to every phase, re-fetch the grasp annotation (which is
    # transformed into world frame using the bottle's CURRENT pose) and
    # pull ``candidates[selected_idx]`` again. Mirrors DexGrasp's
    # ``_update_poses_from_object_pose`` reactive flag — handles bottles
    # that drift / settle between phases.
    # Phase → arm target pose / hand action mapping (matches DexGrasp.step):
    #   pre_grasp    : pre = back-off along approach axis from coarse  |  hand open
    #   coarse_grasp : coarse                                          |  hand open
    #   fine_grasp   : final (precision-position with hand still open) |  hand open
    #   final_grasp  : final                                           |  hand close
    #   retrieval    : retrieval = z-up offset from final              |  hand close
    phase_schedule = [
        "pre_grasp",
        "coarse_grasp",
        "fine_grasp",
        "final_grasp",
        "retrieval",
    ]

    target_pose = None
    hand_action = hand_open
    for phase_name in phase_schedule:
        # ---- refresh from current bottle pose ----
        grasp_list = env.get_grasp_pose(env_ids=[env_id], hand_type="sharpa")
        env_dict = grasp_list[0] if grasp_list else None
        if env_dict is None:
            log.error("[{}] grasp annotation lookup failed", phase_name)
            return
        candidates = _flat_candidates(env_dict)
        if selected_idx >= len(candidates):
            log.error(
                "[{}] selected_idx={} out of range (n={})",
                phase_name,
                selected_idx,
                len(candidates),
            )
            return
        chosen = candidates[selected_idx]
        coarse_w = _phase_to_pose7(chosen.get("coarse_grasp"), device)
        final_w = _phase_to_pose7(chosen.get("final_grasp"), device)
        if coarse_w is None or final_w is None:
            log.error("[{}] candidate missing coarse / final pose", phase_name)
            return
        final_joints = _phase_joints(chosen.get("final_grasp"), HAND_DIM_PER, device)
        pre_grasp_w = _compute_pre_grasp(coarse_w, PRE_GRASP_OFFSET)
        retrieval_w = _compute_retrieval(final_w, RETRIEVAL_OFFSET)
        hand_final = _pack_right_hand_action(final_joints, device)

        if phase_name == "pre_grasp":
            target_pose, hand_action = pre_grasp_w, hand_open
        elif phase_name == "coarse_grasp":
            target_pose, hand_action = coarse_w, hand_open
        elif phase_name == "fine_grasp":
            target_pose, hand_action = final_w, hand_open
        elif phase_name == "final_grasp":
            target_pose, hand_action = final_w, hand_final
        elif phase_name == "retrieval":
            target_pose, hand_action = retrieval_w, hand_final

        steps = PHASE_STEPS[phase_name]
        log.info(
            "[phase] {} (refresh) → {} steps (hand={})",
            phase_name,
            steps,
            "close" if hand_action is hand_final else "open",
        )
        _log_pose(phase_name, target_pose)
        _viz([target_pose])
        for _ in range(steps):
            action = _build_phase_action(env, target_pose, hand_action)
            env.step(action=action)

    log.info("[phase] done — holding last retrieval pose.")
    while True:
        action = _build_phase_action(env, target_pose, hand_action)
        env.step(action=action)


if __name__ == "__main__":
    main()
