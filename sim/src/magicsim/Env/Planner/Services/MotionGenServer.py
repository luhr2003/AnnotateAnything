"""cuRobo 2.0 MotionGenServer — in-process motion planning service.

v2 port of the v1 ``MotionGenServer`` (see
``CUROBO_V2_01_CURRENT_INTERFACES.md §1.3``). Architecture preserved 1:1:
pool of ``_MotionGenInstance`` worker threads, microbatch window per
instance, merge + dedup + chunk loop with per-slot scene loads. The
solver backend switches from ``curobo.wrap.reacher.motion_gen.MotionGen``
(``plan_batch_env`` / ``plan_single``) to
``curobo.batch_motion_planner.BatchMotionPlanner`` wrapped by the local
``VariableBatchPlanner`` (``batch_planning_utils.py``).

Biggest behaviour deltas from v1 (see ``CUROBO_V2_02_MIGRATION_PLAN.md`` §0.2):
- **``force_single_plan`` branch deleted**. v2 ``BatchMotionPlanner.plan_pose``
  with a multi-tool-frame YAML handles ``B > 1, L > 1`` natively; no more
  per-env ``plan_single`` loop.
- ``result.get_paths()`` → iterate ``result.interpolated_trajectory[slot]`` +
  ``trim_joint_state_trajectory(traj, 0, last_tstep[slot])``.
- ``motion_gen.get_full_js(traj)`` → ``planner.kinematics.get_full_js(traj)``.
- ``mode="position"`` post-step uses
  ``planner.compute_kinematics(JointState).tool_poses[name]`` in cfg
  ``info_links`` order (v1 ``kin_model.get_state(q).link_poses[name]``).

Dynamic batch size (see ``CUROBO_V2_03_DYNAMIC_BATCH.md`` §4):
- ``BatchMotionPlanner.plan_pose`` requires exactly ``max_batch_size``.
  We wrap with ``VariableBatchPlanner`` which pads
  ``current_state`` + ``goal`` with retract-state / retract-FK dummy
  problems and slices the result back. The wrapper also reloads
  ``Scene()`` into pad slots on B shrink.

MagicSim info_links contract (see ``ServiceMigrate.md`` §3):
- YAML ``tool_frames`` = union of tracked arms + FK-only info links.
- Server cfg ``tracked_tool_frames`` → tracked subset. Defaults to
  ``planner.tool_frames`` for backwards compatibility.
- Server cfg ``info_links`` → ordering of FK output in ``mode="position"``.
- At init: ``planner.update_tool_pose_criteria({...})`` with
  ``track_...`` for tracked frames, ``ToolPoseCriteria.disabled()`` for
  the rest. Propagates to both IK and TrajOpt stages.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from magicsim.Env.Robot.RobotManager import RobotManager
from magicsim.Env.Planner.Utils import quat_mul, quat_inv
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Planner.Services import _normalize_planner_devices
from magicsim.Env.Planner.Services.batch_planning_utils import VariableBatchPlanner

# cuRobo v2
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.state.state_joint_ops import stack_joint_states
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

from magicsim.Env.Planner.Services.nan_preprocessing import (
    detect_nan_disable_single,
    log_joint_state_submit,
    log_scene_slot_load,
)


def _ensure_base_quat_4d(base_quat: torch.Tensor) -> torch.Tensor:
    """Ensure base_quat is (..., 4). If last dim is 1, treat as yaw (rad) and convert to quat (w,x,y,z)."""
    if base_quat.shape[-1] == 4:
        return base_quat
    if base_quat.shape[-1] == 1:
        yaw = base_quat.squeeze(-1)
        half = yaw * 0.5
        w = half.cos().unsqueeze(-1)
        z = half.sin().unsqueeze(-1)
        x = torch.zeros_like(w)
        y = torch.zeros_like(w)
        return torch.cat([w, x, y, z], dim=-1)
    raise ValueError(
        f"base_quat must have last dim 4 or 1, got shape {base_quat.shape}"
    )


@dataclass
class MotionGenPlanRequest:
    """One planning request.

    ``robot_states`` tensors are expected to cover all envs; we slice
    by ``env_id`` inside ``_preprocess_request``.

    ``target_pos`` accepted shapes:
      - 2-D ``(N, eef_num * 7)`` — caller-side flat layout
      - 3-D ``(N, eef_num, 7)`` — canonical per-EEF layout

    where ``eef_num == len(tracked_tool_frames)`` and each 7-vec is
    ``[x, y, z, qw, qx, qy, qz]``. Position-only ``(..., 3)`` shapes are
    rejected — a NaN-xyz row disables the tool per env (see
    :func:`nan_preprocessing.detect_nan_disable_single`).
    """

    env_ids: List[int]
    target_pos: torch.Tensor
    robot_states: Dict[str, torch.Tensor]


class _MotionGenInstance:
    """Single BatchMotionPlanner instance with its own worker thread + queue."""

    def __init__(
        self,
        instance_id: int,
        robot_cfg: dict,
        batch_size: int,
        robot_dof_name: List[str],
        robot_dof_name_active: List[str],
        robot_lock_joints: Optional[List[str]],
        robot_ignore_joints: dict,
        robot_manager: RobotManager,
        robot_name: str,
        device: torch.device,
        planner_device: torch.device,
        world_cfg_list: List[dict],
        world_lock: threading.Lock,
        microbatch_wait_s: float,
        debug: bool,
        mode: str,
        info_links: Optional[List[str]],
        robot_add_joints: dict,
        extra_fk_link: Optional[List[str]],
        track_xyz_weight: Tuple[float, float, float],
        track_rpy_weight: Tuple[float, float, float],
        max_attempts: int = 2,
        enable_graph_attempt: int = 0,
        relative_to_world_frame: bool = True,
    ):
        self.instance_id = instance_id
        self.robot_cfg = robot_cfg
        self.batch_size = batch_size
        self.robot_dof_name = robot_dof_name
        self.robot_dof_name_active = robot_dof_name_active
        self.robot_lock_joints = robot_lock_joints
        self.robot_ignore_joints = robot_ignore_joints
        self.robot_add_joints = robot_add_joints or {}
        # ``self.eef_num`` is owned by the outer :class:`MotionGenServer` —
        # PlannerManager sets it from ``motiongen.eef_num`` in the YAML
        # right after construction.
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.device = device
        self.planner_device = torch.device(planner_device)
        self.world_cfg_list = world_cfg_list
        self._world_lock = world_lock
        self.microbatch_wait_s = microbatch_wait_s
        self._debug = debug
        self.mode = mode
        self.max_attempts = int(max_attempts)
        self.enable_graph_attempt = int(enable_graph_attempt)
        self.relative_to_world_frame = relative_to_world_frame
        self._track_xyz_weight = list(track_xyz_weight)
        self._track_rpy_weight = list(track_rpy_weight)

        # Build the v2 batch motion planner. multi_env=True + max_batch_size=N
        # allocates N per-env collision worlds, same model as v1.
        device_cfg = DeviceCfg(device=self.planner_device, dtype=torch.float32)
        # For multi_env=True we must seed the solver with a list of
        # ``batch_size`` initial SCENE DICTS (not Scene objects — the cfg
        # factory iterates and calls ``SceneCfg.create(x)`` per element,
        # which expects a dict). Empty dicts are fine; per-slot obstacles
        # come in via ``load_collision_model(..., env_idx=slot)`` at each
        # chunk. Pattern from
        # ``curobo/examples/isaacsim/dynamic_batch_motion_gen_reacher.py:246-249``.
        initial_scenes = [{} for _ in range(batch_size)]
        # Seed counts match v1 MagicSim (``num_trajopt_seeds=4``) and
        # curobo v2's own ``MotionPlannerCfg.create`` default. The
        # ``batch_motion_gen_reacher.py`` example uses 12 but that's a
        # benchmark setting — in production 4 is a ~3× solve-time win
        # with negligible success-rate drop for our goal distributions.
        # ``num_ik_seeds=32`` is already the cfg default; passed explicitly
        # so the config is self-documenting.
        # Motion planning is single-pose only — fix ``max_goalset=1`` here
        # and don't expose it as a Server kwarg. Goalset planning would
        # add a kernel variant we don't ship.
        planner_cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            device_cfg=device_cfg,
            scene_model=initial_scenes,
            max_batch_size=batch_size,
            multi_env=True,
            max_goalset=1,
            collision_cache={"cuboid": 10, "mesh": 500},
            num_trajopt_seeds=4,
            num_ik_seeds=32,
            # See IKServer for why use_cuda_graph=False: variable num_goalset
            # per call + per-chunk load_collision_model both invalidate captured
            # graphs ("CUDA graph reset is not available" / illegal memory
            # access on replay).
            use_cuda_graph=False,
            self_collision_check=True,
            optimizer_collision_activation_distance=0.025,
        )
        planner = BatchMotionPlanner(planner_cfg)
        # Warmup MUST run on the worker thread (see __init__'s
        # _warmup_done_event comment below). Main-thread warmup loads
        # warp kernel modules into main's CUDA context; the worker's
        # first plan_pose then reloads them in its own context, shuffling
        # GPU pages and invalidating any CUDA graph captured between
        # construction and first solve.
        self._planner = planner
        self._vbp = VariableBatchPlanner(planner)

        # Resolve tracked + info link lists from merged tool_frames.
        # After PlannerManager merge, ``planner.tool_frames`` = YAML tracked
        # frames ++ extra_fk_link (order preserved). tracked = complement.
        yaml_tool_frames = list(planner.tool_frames)
        fk_only = list(extra_fk_link or [])
        missing = set(fk_only) - set(yaml_tool_frames)
        if missing:
            raise ValueError(
                f"extra_fk_link {fk_only} contains frames not present in merged "
                f"tool_frames={yaml_tool_frames}: {missing}. "
                f"PlannerManager must merge extra_fk_link into kinematics.tool_frames."
            )
        self._extra_fk_link: List[str] = fk_only
        fk_set = set(fk_only)
        self._tracked_tool_frames: List[str] = [
            f for f in yaml_tool_frames if f not in fk_set
        ]

        # info_links → ordering of position-mode FK export. Default = the
        # original YAML tool_frames order (= tracked, without extras).
        if info_links is None or len(info_links) == 0:
            self._info_links: List[str] = list(self._tracked_tool_frames)
        else:
            missing_i = set(info_links) - set(yaml_tool_frames)
            if missing_i:
                raise ValueError(
                    f"info_links {info_links} contains frames not in merged "
                    f"tool_frames={yaml_tool_frames}: {missing_i}. "
                    f"Add them to kinematics.tool_frames or extra_fk_link."
                )
            self._info_links = list(info_links)

        # Criteria: track for tracked frames, disabled() for extra_fk_link.
        # ``update_tool_pose_criteria`` propagates to BOTH the IK stage and
        # the TrajOpt stage of MotionPlanner (motion_planner_batch.py:617-619
        # → motion_planner.py:630-632), so extra_fk_link frames contribute
        # ZERO cost to both IK seeding and trajectory optimization.
        criteria: Dict[str, ToolPoseCriteria] = {}
        tracked_set = set(self._tracked_tool_frames)
        for frame in yaml_tool_frames:
            if frame in tracked_set:
                criteria[frame] = ToolPoseCriteria.track_position_and_orientation(
                    xyz=self._track_xyz_weight,
                    rpy=self._track_rpy_weight,
                )
            else:
                criteria[frame] = ToolPoseCriteria.disabled()
        planner.update_tool_pose_criteria(criteria)

        # Cache criteria templates for the per-env runtime disable path
        # (NaN preprocessing in ``_plan_one_batch``).
        self._track_criteria_template = ToolPoseCriteria.track_position_and_orientation(
            xyz=self._track_xyz_weight,
            rpy=self._track_rpy_weight,
        )
        self._disabled_criteria_template = ToolPoseCriteria.disabled()

        # Per-slot disable-mask cache. Init state matches the broadcast
        # ``update_tool_pose_criteria`` above (all slots = all-track =
        # all-False). ``_plan_one_batch`` only re-writes per-env criteria
        # when the new mask differs from this cache — saves B CUDA buffer
        # writes + syncs per solve when the caller passes fully-tracked
        # targets (the locomotion / WBC tick path).
        self._persisted_disable_per_slot: torch.Tensor = torch.zeros(
            (self.batch_size, len(self._tracked_tool_frames)),
            dtype=torch.bool,
            device=self.planner_device,
        )

        self._curobo_joint_names = list(planner.kinematics.joint_names)

        # Worker thread / queue for this instance.
        self._queue: List[Tuple[concurrent.futures.Future, MotionGenPlanRequest]] = []
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._shutdown = False

        # See IKServer __init__ comment: the worker's first CUDA op
        # initializes a per-thread CUDA context and shuffles GPU memory,
        # which invalidates CUDA graphs captured by the main thread (e.g.
        # the IsaacLab action-IK graph). Block __init__ until the worker
        # has touched CUDA at least once.
        self._warmup_done_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._warmup_done_event.wait()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        with self._queue_cv:
            self._shutdown = True
            self._queue_cv.notify_all()
        if join:
            self._worker.join(timeout=timeout_s)

    def submit(self, fut: concurrent.futures.Future, req: MotionGenPlanRequest):
        with self._queue_cv:
            self._queue.append((fut, req))
            self._queue_cv.notify()

    def queue_size(self) -> int:
        with self._queue_cv:
            return len(self._queue)

    def is_idle(self) -> bool:
        with self._queue_cv:
            return len(self._queue) == 0

    # ------------------------------------------------------------------
    # Frame + joint-state helpers
    # ------------------------------------------------------------------

    def _world_to_robot_frame(
        self,
        target_pos: torch.Tensor,
        robot_base_pose: torch.Tensor,
        robot_base_quat: torch.Tensor,
        env_ids: List[int],
    ) -> torch.Tensor:
        """[N, eef_num, 7] world-frame targets → robot-base-frame targets."""
        device = robot_base_pose.device
        target_tensor = target_pos.to(device=device, dtype=torch.float32)
        if target_tensor.ndim != 3 or target_tensor.shape[2] != 7:
            raise ValueError(
                f"Expected target_pos [N,eef,7], got {target_tensor.shape}"
            )

        n, eef_num, _ = target_tensor.shape
        flat = target_tensor.reshape(-1, 7)
        target_positions = flat[:, :3]
        target_quats = flat[:, 3:]

        env_ids_tensor = torch.tensor(env_ids, device=device, dtype=torch.long)
        base_positions = robot_base_pose[env_ids_tensor]
        base_quats = robot_base_quat[env_ids_tensor]
        base_positions = base_positions.repeat_interleave(eef_num, dim=0)
        base_quats = base_quats.repeat_interleave(eef_num, dim=0)

        pos_relative = target_positions - base_positions
        base_rot_matrices = quat_to_rot_matrix(base_quats)
        base_rot_inv = base_rot_matrices.transpose(-2, -1)
        pos_relative_rotated = torch.bmm(
            base_rot_inv, pos_relative.unsqueeze(-1)
        ).squeeze(-1)

        base_quats_inv = quat_inv(base_quats)
        quats_relative = quat_mul(base_quats_inv, target_quats)

        out = torch.cat([pos_relative_rotated, quats_relative], dim=1).reshape(
            n, eef_num, 7
        )
        return out.to(device=device, dtype=torch.float32)

    def _preprocess_request(
        self,
        req: MotionGenPlanRequest,
    ) -> Tuple[
        List[int],
        torch.Tensor,  # target_pos in robot frame [N, eef_num, 7]
        JointState,  # current_state in cuRobo joint order [N, dof]
        torch.Tensor,  # base_pos [N, 3]
        torch.Tensor,  # base_quat [N, 4]
        torch.Tensor,  # joint_pos (sim order) [N, num_joints]
    ]:
        env_ids = [int(x) for x in req.env_ids]
        target_pos = req.target_pos

        # Caller contract: ``target_pos`` is either 2-D ``(N, eef_num * 7)``
        # or 3-D ``(N, eef_num, 7)`` — canonicalize to 3-D. Position-only
        # shapes (``..., 3``) are NOT accepted; callers that want to pass
        # position-only should pack as ``[x, y, z, w=1, x=0, y=0, z=0]`` or
        # use quaternion-NaN (which the NaN-disable path already handles).
        if target_pos.ndim == 2:
            flat = target_pos.shape[1]
            assert flat % 7 == 0, (
                f"MotionGenPlanRequest.target_pos 2-D last dim must be a "
                f"multiple of 7 (one 7-vec per EEF); got {flat}. "
                f"Position-only (..., 3) shapes are not accepted."
            )
            target_pos = target_pos.view(target_pos.shape[0], flat // 7, 7)
        elif target_pos.ndim == 3:
            assert target_pos.shape[-1] == 7, (
                f"MotionGenPlanRequest.target_pos 3-D last dim must be 7; "
                f"got shape {tuple(target_pos.shape)}. Position-only "
                f"(N, eef_num, 3) shapes are not accepted."
            )
        else:
            raise AssertionError(
                f"MotionGenPlanRequest.target_pos must be 2-D "
                f"(N, eef_num*7) or 3-D (N, eef_num, 7); got "
                f"{tuple(target_pos.shape)}"
            )

        robot_base_quat = _ensure_base_quat_4d(req.robot_states["base_quat"])
        if self.relative_to_world_frame:
            target_pos = self._world_to_robot_frame(
                target_pos,
                req.robot_states["base_pos"],
                robot_base_quat,
                env_ids,
            )

        robot_joint_poses = req.robot_states["joint_pos"]
        robot_joint_vels = req.robot_states["joint_vel"]
        robot_base_pos = req.robot_states["base_pos"]

        base_pos_list: List[torch.Tensor] = []
        base_quat_list: List[torch.Tensor] = []
        joint_pos_list: List[torch.Tensor] = []
        batch_full_js: Optional[JointState] = None

        for env_id in env_ids:
            env_id = int(env_id)
            robot_joint_pose = robot_joint_poses[env_id]
            robot_joint_vel = robot_joint_vels[env_id]
            base_pos_list.append(robot_base_pos[env_id].clone())
            base_quat_list.append(robot_base_quat[env_id].clone())
            joint_pos_list.append(robot_joint_pose.clone())

            # Build active-joint JointState in the v1 "active dof" order.
            active_pos: List[float] = []
            active_vel: List[float] = []
            for x in self.robot_dof_name:
                if x in (self.robot_lock_joints or []) or x in self.robot_ignore_joints:
                    continue
                if x in self.robot_add_joints:
                    active_pos.append(robot_joint_pose[self.robot_add_joints[x]].item())
                    active_vel.append(robot_joint_vel[self.robot_add_joints[x]].item())
                    continue
                joint_ids, _ = self.robot_manager.robots[self.robot_name].find_joints(x)
                if len(joint_ids) == 0:
                    raise ValueError(f"Joint {x} not found in robot_joint_pose")
                joint_idx_int = int(joint_ids[0])
                if 0 <= joint_idx_int < robot_joint_pose.shape[0]:
                    active_pos.append(robot_joint_pose[joint_idx_int].item())
                    active_vel.append(robot_joint_vel[joint_idx_int].item())

            active_pos_t = torch.tensor(
                active_pos, device=self.planner_device, dtype=torch.float32
            ).view(1, -1)
            active_vel_t = torch.tensor(
                active_vel, device=self.planner_device, dtype=torch.float32
            ).view(1, -1)
            cur_js = JointState(
                position=active_pos_t,
                velocity=active_vel_t * 0.0,
                acceleration=active_pos_t * 0.0,
                jerk=active_pos_t * 0.0,
                joint_names=self.robot_dof_name_active,
            )
            # v2: ``JointState.stack()`` is deprecated; use the functional form.
            batch_full_js = (
                cur_js
                if batch_full_js is None
                else stack_joint_states(batch_full_js, cur_js)
            )

        # Reorder to cuRobo joint order.
        # v2 renamed ``get_ordered_joint_state`` → ``reorder``.
        batch_full_js = batch_full_js.reorder(self._curobo_joint_names)

        base_pos = torch.stack(base_pos_list, dim=0)
        base_quat = torch.stack(base_quat_list, dim=0)
        joint_pos = torch.stack(joint_pos_list, dim=0)
        return env_ids, target_pos, batch_full_js, base_pos, base_quat, joint_pos

    # ------------------------------------------------------------------
    # Position-mode FK export (info_links order)
    # ------------------------------------------------------------------

    def _trajectory_to_eef_pose(
        self,
        action_traj: torch.Tensor,  # [T, 2 * active_dof]
        base_pos: torch.Tensor,  # [3]
        base_quat: torch.Tensor,  # [4]
        joint_pos: torch.Tensor,  # [num_joints] (sim order; unused here)
        action_joint_names: List[str],
    ) -> torch.Tensor:
        """Convert a joint-space trajectory to EEF pose trajectory.

        Returns ``[T, len(info_links) * 7]``. World frame when
        ``relative_to_world_frame=True``; arm-ref frame otherwise.
        """
        action_dim = action_traj.shape[1] // 2
        action_joint_pos = action_traj[:, :action_dim]
        action_name_to_idx = {name: idx for idx, name in enumerate(action_joint_names)}

        # Reorder active joint positions to cuRobo joint order.
        per_step: List[List[float]] = []
        for step_idx in range(action_joint_pos.shape[0]):
            ordered: List[float] = []
            for name in self._curobo_joint_names:
                if name in action_name_to_idx:
                    j = action_name_to_idx[name]
                    ordered.append(action_joint_pos[step_idx, j].item())
                elif name in (self.robot_lock_joints or []):
                    ordered.append(0.0)
                elif name in self.robot_ignore_joints:
                    ordered.append(float(self.robot_ignore_joints[name]))
                else:
                    raise ValueError(
                        f"Joint {name} not found in action_joint_names and no lock/ignore "
                        f"entry provided"
                    )
            per_step.append(ordered)

        q = torch.tensor(per_step, device=self.planner_device, dtype=torch.float32)

        # FK in one shot using compute_kinematics.
        js = JointState.from_position(q, joint_names=self._curobo_joint_names)
        kin = self._planner.compute_kinematics(js)

        # Broadcast base pose over the trajectory.
        T = q.shape[0]
        base_pos_w = (
            base_pos.to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(T, 1)
        )
        base_quat_w = _ensure_base_quat_4d(
            base_quat.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        ).squeeze(0)
        base_quat_w = base_quat_w.unsqueeze(0).repeat(T, 1)
        base_rot = quat_to_rot_matrix(base_quat_w)

        poses_dict = (
            kin.tool_poses.to_dict()
        )  # {name: Pose(position=(T,3), quaternion=(T,4))}
        out_per_link: List[torch.Tensor] = []
        for link in self._info_links:
            if link not in poses_dict:
                raise ValueError(
                    f"info_link {link!r} not present in planner tool_poses "
                    f"({list(poses_dict.keys())}). "
                    f"Declare it in kinematics.tool_frames."
                )
            p = poses_dict[link]
            ee_pos_local = p.position.to(device=self.device, dtype=torch.float32)
            ee_quat_local = p.quaternion.to(device=self.device, dtype=torch.float32)

            if self.relative_to_world_frame:
                ee_pos_world = base_pos_w + torch.bmm(
                    base_rot, ee_pos_local.unsqueeze(-1)
                ).squeeze(-1)
                ee_quat_world = quat_mul(base_quat_w, ee_quat_local)
                out_per_link.append(torch.cat([ee_pos_world, ee_quat_world], dim=1))
            else:
                out_per_link.append(torch.cat([ee_pos_local, ee_quat_local], dim=1))

        return torch.cat(out_per_link, dim=1)  # [T, L * 7]

    # ------------------------------------------------------------------
    # Plan one batch via VariableBatchPlanner.plan_pose
    # ------------------------------------------------------------------

    def _build_common_js_names(self, cmd_plan: JointState) -> List[str]:
        common: List[str] = []
        for x in self.robot_dof_name:
            if x in (self.robot_lock_joints or []) or x in (
                self.robot_ignore_joints or []
            ):
                continue
            if x in cmd_plan.joint_names:
                common.append(x)
        return common

    def _plan_one_batch(
        self,
        batch_env_ids: List[int],
        batch_target_pos: torch.Tensor,  # [B, eef_num, 7] (robot frame)
        batch_full_js: JointState,  # [B, dof] cuRobo order
        batch_base_pos: torch.Tensor,  # [B, 3]
        batch_base_quat: torch.Tensor,  # [B, 4]
        batch_joint_pos: torch.Tensor,  # [B, num_joints] (sim order)
    ) -> Tuple[List[Optional[torch.Tensor]], List[bool]]:
        """Plan for ``len(batch_env_ids)`` envs. ``VariableBatchPlanner``
        handles pad-to-max_batch_size + result slicing back."""
        B = len(batch_env_ids)
        assert B == batch_target_pos.shape[0] == batch_full_js.position.shape[0]
        assert batch_target_pos.shape[1] == len(self._tracked_tool_frames), (
            f"target_pos eef_num={batch_target_pos.shape[1]} != "
            f"len(tracked_tool_frames)={len(self._tracked_tool_frames)}"
        )

        # Real-slot per-env scene loads.
        with self._world_lock:
            for slot, env_id in enumerate(batch_env_ids):
                env_id_int = int(env_id)
                if env_id_int < 0 or env_id_int >= len(self.world_cfg_list):
                    raise ValueError(
                        f"env_id {env_id_int} out of range [0, {len(self.world_cfg_list)})"
                    )
                scene_cfg = self.world_cfg_list[env_id_int]
                if isinstance(scene_cfg, Scene):
                    scene = scene_cfg
                elif isinstance(scene_cfg, dict) and scene_cfg:
                    scene = Scene.create(scene_cfg)
                else:
                    scene = Scene()
                self._planner.scene_collision_checker.load_collision_model(
                    scene,
                    env_idx=int(slot),
                )
                if self._debug:
                    log_scene_slot_load(scene, int(slot), env_id_int, tag="MG")
            # VariableBatchPlanner handles pad-slot reload on B shrink.

        # ----- NaN preprocessing → per-env disable + FK substitution -----
        # Per (env, tool) any-NaN xyz row ⇒ disable that tool for that env
        # this solve. The disabled tool's filler value is the current FK
        # (cost weight is zero on disabled rows; value is inert).
        target_dev = batch_target_pos.to(self.planner_device)
        disable_mask = detect_nan_disable_single(target_dev)  # (B, eef_num) bool

        # Compute current FK once — used both as NaN substitution filler
        # AND as the info-only frame filler below.
        fk_kin = self._planner.compute_kinematics(batch_full_js)
        fk_pose_per_frame: Dict[str, Pose] = {}
        for frame in self._planner.tool_frames:
            fk_pose_per_frame[frame] = fk_kin.tool_poses.get_link_pose(
                frame,
                make_contiguous=True,
            )

        # Substitute NaN (env, tool) rows with FK values BEFORE building
        # GoalToolPose (curobo asserts no-NaN goal tensors).
        target_subbed = target_dev.clone()
        if bool(disable_mask.any().item()):
            for li, frame in enumerate(self._tracked_tool_frames):
                row_mask = disable_mask[:, li]
                if not bool(row_mask.any().item()):
                    continue
                p = fk_pose_per_frame[frame]
                fk_pos = p.position.view(-1, 3)
                fk_quat = p.quaternion.view(-1, 4)
                target_subbed[row_mask, li, :3] = fk_pos[row_mask]
                target_subbed[row_mask, li, 3:] = fk_quat[row_mask]

        # Per-env runtime criteria — write only the slots whose disable
        # pattern CHANGED vs ``self._persisted_disable_per_slot``. The
        # common case (locomotion / WBC ticks) sends fully-tracked
        # targets every solve, so ``disable_mask`` matches the cache and
        # this loop becomes a single CPU comparison + zero CUDA writes.
        # NOTE: ``env_idx`` is the SOLVER SLOT INDEX (0..max_batch_size-1).
        cached_dev = self._persisted_disable_per_slot[:B]
        if cached_dev.device != disable_mask.device:
            cached_dev = cached_dev.to(disable_mask.device)
        slot_changed = (cached_dev != disable_mask).any(dim=-1)  # (B,) bool
        if bool(slot_changed.any().item()):
            changed_idx = slot_changed.nonzero(as_tuple=True)[0].tolist()
            for b in changed_idx:
                crit: Dict[str, ToolPoseCriteria] = {}
                for li, frame in enumerate(self._tracked_tool_frames):
                    crit[frame] = (
                        self._disabled_criteria_template
                        if bool(disable_mask[b, li])
                        else self._track_criteria_template
                    )
                self._planner.update_tool_pose_criteria_per_env(b, crit)
            # Persist new state so the next solve's diff is correct.
            self._persisted_disable_per_slot[:B] = disable_mask.to(
                self._persisted_disable_per_slot.device
            )

        # Build GoalToolPose covering all tool_frames.
        pose_dict: Dict[str, Pose] = {}
        for i, frame in enumerate(self._tracked_tool_frames):
            pose_dict[frame] = Pose(
                position=target_subbed[:, i, :3].contiguous(),
                quaternion=target_subbed[:, i, 3:].contiguous(),
            )

        # Non-tracked / info-only frames — reuse cached FK (no extra call).
        needs_fk = [f for f in self._planner.tool_frames if f not in pose_dict]
        for frame in needs_fk:
            pose_dict[frame] = fk_pose_per_frame[frame]

        goal = GoalToolPose.from_poses(
            pose_dict,
            ordered_tool_frames=list(self._planner.tool_frames),
            num_goalset=1,
        )

        actions: List[Optional[torch.Tensor]] = [None] * B
        success: List[bool] = [False] * B
        common_js_names_list: List[Optional[List[str]]] = [None] * B

        if self._debug:
            log_joint_state_submit(batch_full_js, batch_env_ids, tag="MG")
        t0 = time.time()
        result = self._vbp.plan_pose(
            goal,
            batch_full_js,
            max_attempts=self.max_attempts,
            enable_graph_attempt=self.enable_graph_attempt,
        )
        plan_dt = time.time() - t0

        if result is None:
            # ``plan_pose`` returns None when no IK seed survived — most
            # commonly: target unreachable from the initial joint state,
            # all collision-checked seeds in self-collision, or world
            # collision blocking every IK candidate.
            tgt = batch_target_pos.detach().cpu()
            base_p = batch_base_pos.detach().cpu()
            base_q = batch_base_quat.detach().cpu()
            print(
                f"[MotionGenServer][FAIL] Instance {self.instance_id} "
                f"plan_pose returned None — no IK seed survived "
                f"(self-collision / world-collision / unreachable). "
                f"envs={batch_env_ids} dt={plan_dt:.4f}s"
            )
            for b, env_id in enumerate(batch_env_ids):
                print(
                    f"[MotionGenServer][FAIL]   env={env_id} slot={b}"
                    f" base_pos={base_p[b].tolist()}"
                    f" base_quat={base_q[b].tolist()}"
                    f" target=\n{tgt[b].tolist()}"
                )
            return actions, success

        pos_err_t = result.position_error
        rot_err_t = result.rotation_error
        pos_err = (
            float(pos_err_t.max().item()) if pos_err_t is not None else float("nan")
        )
        rot_err = (
            float(rot_err_t.max().item()) if rot_err_t is not None else float("nan")
        )
        succ_t = result.success
        succ_any = succ_t.any(dim=-1) if succ_t is not None else None

        if not bool(succ_t.any().item()):
            tgt = batch_target_pos.detach().cpu()
            print(
                f"[MotionGenServer][FAIL] Instance {self.instance_id} "
                f"plan_pose envs={batch_env_ids} ALL FAILED "
                f"pos_err_max={pos_err:.4f}m rot_err_max={rot_err:.4f}rad "
                f"dt={plan_dt:.4f}s "
                f"(thresholds: pos<=0.005m rot<=0.05rad). Common causes: "
                f"target out of workspace, target inside obstacle, "
                f"paired-IK blocking convergence on this G."
            )
            for b, env_id in enumerate(batch_env_ids):
                pe = (
                    float(pos_err_t[b].max().item())
                    if pos_err_t is not None
                    else float("nan")
                )
                re_ = (
                    float(rot_err_t[b].max().item())
                    if rot_err_t is not None
                    else float("nan")
                )
                print(
                    f"[MotionGenServer][FAIL]   env={env_id} slot={b}"
                    f" pos_err={pe:.4f}m rot_err={re_:.4f}rad"
                    f" target=\n{tgt[b].tolist()}"
                )
            return actions, success

        print(
            f"[MotionGenServer][debug] Instance {self.instance_id} plan_pose "
            f"envs={batch_env_ids} success={succ_any} pos_err_max={pos_err:.4f}m "
            f"rot_err_max={rot_err:.4f}rad dt={plan_dt:.4f}s"
        )

        interp = result.interpolated_trajectory  # (B, 1, max_H, dof_full) wrapper
        last = result.interpolated_last_tstep  # (B, 1)

        if interp is None or last is None:
            return actions, success

        for slot in range(B):
            if not bool(result.success[slot].any().item()):
                continue
            traj = interp[slot].squeeze(0) if hasattr(interp, "__getitem__") else None
            if traj is None:
                continue
            valid_h = int(last[slot].item())
            cmd_plan = trim_joint_state_trajectory(traj, 0, valid_h)
            common_js_names = self._build_common_js_names(cmd_plan)
            cmd_plan = cmd_plan.reorder(common_js_names)

            position = cmd_plan.position.to(device=self.device, dtype=torch.float32)
            velocity = (
                cmd_plan.velocity.to(device=self.device, dtype=torch.float32)
                if cmd_plan.velocity is not None
                else torch.zeros_like(position)
            )
            actions[slot] = torch.cat([position, velocity], dim=1)
            success[slot] = True
            common_js_names_list[slot] = common_js_names

        # mode="position" — re-FK the trajectory into an [T, len(info_links)*7] tensor.
        if self.mode == "position":
            for slot in range(B):
                if actions[slot] is None:
                    continue
                if common_js_names_list[slot] is None:
                    raise RuntimeError(
                        "Missing common_js_names for position mode conversion"
                    )
                actions[slot] = self._trajectory_to_eef_pose(
                    actions[slot],
                    batch_base_pos[slot],
                    batch_base_quat[slot],
                    batch_joint_pos[slot],
                    common_js_names_list[slot],
                )

        return actions, success

    # ------------------------------------------------------------------
    # Worker loop — unchanged v1 structure
    # ------------------------------------------------------------------

    def _worker_loop(self):
        # Full planner warmup on this thread — JIT-compiles all warp
        # kernels into the worker's CUDA context so that subsequent
        # main-thread CUDA-graph captures (e.g. curobo_ik_actions) see a
        # stable warp/CUDA module layout. See __init__ comment.
        try:
            self._planner.warmup(enable_graph=False, num_warmup_iterations=3)
            torch.cuda.synchronize(self.planner_device)
        finally:
            self._warmup_done_event.set()

        while True:
            with self._queue_cv:
                while not self._shutdown and len(self._queue) == 0:
                    self._queue_cv.wait()
                if self._shutdown:
                    return

                batch: List[Tuple[concurrent.futures.Future, MotionGenPlanRequest]] = []
                batch.append(self._queue.pop(0))
                if self.microbatch_wait_s > 0.0:
                    start = time.time()
                    while True:
                        remaining = self.microbatch_wait_s - (time.time() - start)
                        if remaining <= 0:
                            break
                        if len(self._queue) == 0:
                            self._queue_cv.wait(timeout=remaining)
                            if len(self._queue) == 0:
                                break
                        batch.append(self._queue.pop(0))
                else:
                    while len(self._queue) > 0:
                        batch.append(self._queue.pop(0))
                        if len(batch) == self.batch_size:
                            break

            preprocessed: List[
                Tuple[
                    concurrent.futures.Future,
                    List[int],
                    torch.Tensor,
                    JointState,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]
            ] = []

            try:
                t_req0 = time.time()
                for fut, req in batch:
                    try:
                        env_ids, tgt, js, bp, bq, jp = self._preprocess_request(req)
                        preprocessed.append((fut, env_ids, tgt, js, bp, bq, jp))
                    except Exception as ex:
                        # Log before fanning the exception out to the future so
                        # caller-side stack traces don't lose the server-side
                        # frames. (Previously only the future carried the
                        # traceback; callers' ``except Exception:`` branches
                        # swallowed it silently.)
                        tb = traceback.format_exc()
                        print(
                            f"[MotionGenServer][ERROR] Instance {self.instance_id} "
                            f"preprocessing failed for request envs={req.env_ids}: "
                            f"{type(ex).__name__}: {ex}\n{tb}"
                        )
                        # Pass the original exception through — don't wrap,
                        # so callers that do ``except <SpecificError>:`` still work.
                        fut.set_exception(ex)

                if len(preprocessed) == 0:
                    continue

                merged_env_ids: List[int] = []
                merged_targets: List[torch.Tensor] = []
                merged_js_list: List[JointState] = []
                merged_base_pos_list: List[torch.Tensor] = []
                merged_base_quat_list: List[torch.Tensor] = []
                merged_joint_pos_list: List[torch.Tensor] = []
                per_req_indices: Dict[concurrent.futures.Future, List[int]] = {}
                per_req_env_ids: Dict[concurrent.futures.Future, List[int]] = {}

                seen_env = set()
                for fut, env_ids, tgt, js, bp, bq, jp in preprocessed:
                    if tgt.ndim != 3:
                        raise ValueError(
                            f"target_pos must be [N,eef,7], got {tgt.shape}"
                        )
                    if tgt.shape[0] != len(env_ids):
                        raise ValueError("target_pos batch size mismatch with env_ids")
                    if js.position.shape[0] != len(env_ids):
                        raise ValueError(
                            f"batch_full_js batch size {js.position.shape[0]} "
                            f"mismatch with env_ids length {len(env_ids)}"
                        )

                    idx_list: List[int] = []
                    for j, env_id in enumerate(env_ids):
                        if env_id in seen_env:
                            existing = merged_env_ids.index(env_id)
                            idx_list.append(existing)
                            if self._debug:
                                print(
                                    f"[MotionGenServer][debug] Instance {self.instance_id} "
                                    f"Env {env_id} already in microbatch, reusing "
                                    f"result from index {existing}"
                                )
                            continue
                        seen_env.add(env_id)
                        merged_env_ids.append(env_id)
                        merged_targets.append(tgt[j])
                        merged_js_list.append(js[j : j + 1])
                        merged_base_pos_list.append(bp[j])
                        merged_base_quat_list.append(bq[j])
                        merged_joint_pos_list.append(jp[j])
                        idx_list.append(len(merged_env_ids) - 1)
                    per_req_indices[fut] = idx_list
                    per_req_env_ids[fut] = env_ids

                if len(merged_env_ids) == 0:
                    for fut, *_ in preprocessed:
                        fut.set_result(([], [], []))
                    continue

                merged_target_tensor = torch.stack(merged_targets, dim=0)
                merged_base_pos = torch.stack(merged_base_pos_list, dim=0)
                merged_base_quat = torch.stack(merged_base_quat_list, dim=0)
                merged_joint_pos = torch.stack(merged_joint_pos_list, dim=0)
                merged_full_js = merged_js_list[0]
                for j in merged_js_list[1:]:
                    merged_full_js = stack_joint_states(merged_full_js, j)

                merged_actions: List[Optional[torch.Tensor]] = [None] * len(
                    merged_env_ids
                )
                merged_success: List[bool] = [False] * len(merged_env_ids)

                for s in range(0, len(merged_env_ids), self.batch_size):
                    e = min(s + self.batch_size, len(merged_env_ids))
                    batch_env_ids = merged_env_ids[s:e]
                    a, ok = self._plan_one_batch(
                        batch_env_ids,
                        merged_target_tensor[s:e],
                        merged_full_js[s:e],
                        merged_base_pos[s:e],
                        merged_base_quat[s:e],
                        merged_joint_pos[s:e],
                    )
                    assert len(a) == len(ok) == len(batch_env_ids)
                    merged_actions[s:e] = a
                    merged_success[s:e] = ok

                for fut, orig_env_ids, _, _, _, _, _ in preprocessed:
                    idx_list = per_req_indices.get(fut)
                    stored_env_ids = per_req_env_ids.get(fut)
                    if idx_list is None or stored_env_ids is None:
                        raise ValueError(f"Missing mapping for future {id(fut)}")
                    actions_out = [merged_actions[ii] for ii in idx_list]
                    success_out = [merged_success[ii] for ii in idx_list]
                    assert len(actions_out) == len(success_out) == len(stored_env_ids)
                    fut.set_result((actions_out, success_out, stored_env_ids))
                    if self._debug:
                        print(
                            f"[MotionGenServer][debug] Instance {self.instance_id} "
                            f"Replied to future {id(fut)}: env_ids={stored_env_ids}, "
                            f"results={len(actions_out)}"
                        )

                if self._debug:
                    print(
                        f"[MotionGenServer][debug] Instance {self.instance_id} served "
                        f"{len(preprocessed)} request(s) merged_envs="
                        f"{len(merged_env_ids)} dt_total={time.time() - t_req0:.4f}s"
                    )
            except Exception as ex:
                # Outer worker failure — log full traceback AND fan out the
                # same exception to every future we took off the queue so
                # callers' ``future.result()`` raises instead of hanging.
                print(
                    f"[MotionGenServer][ERROR] Instance {self.instance_id} "
                    f"worker exception — {type(ex).__name__}: {ex}\n"
                    f"{traceback.format_exc()}"
                )
                for fut, *_ in preprocessed:
                    if not fut.done():
                        fut.set_exception(ex)
                for fut, req in batch:
                    if not fut.done():
                        fut.set_exception(ex)


class MotionGenServer:
    """In-process cuRobo 2.0 motion-gen service with instance pool + microbatching."""

    dual_mode: bool = False

    def __init__(
        self,
        robot_manager: RobotManager,
        robot_cfg: dict,
        robot_name: str,
        robot_dof_name: List[str],
        robot_dof_name_active: List[str],
        robot_lock_joints: Optional[List[str]],
        robot_ignore_joints: dict,
        robot_add_joints: dict,
        device: torch.device,
        batch_size: int = 2,
        microbatch_wait_ms: float = 200.0,
        num_instances: int = 2,
        debug: bool = False,
        mode: str = "joint",
        info_links: Optional[List[str]] = None,
        extra_fk_link: Optional[List[str]] = None,
        track_xyz_weight: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        track_rpy_weight: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        max_attempts: int = 2,
        enable_graph_attempt: int = 0,
        relative_to_world_frame: bool = True,
        planner_devices: Optional[Union[str, torch.device, Sequence]] = None,
    ):
        self.robot_manager = robot_manager
        self.robot_cfg = robot_cfg
        self.robot_name = robot_name
        self.device = device

        self.planner_devices: List[torch.device] = _normalize_planner_devices(
            planner_devices, int(num_instances)
        )

        self.robot_dof_name = robot_dof_name
        self.robot_dof_name_active = robot_dof_name_active
        self.robot_lock_joints = robot_lock_joints
        self.robot_ignore_joints = robot_ignore_joints or {}
        self.robot_add_joints = robot_add_joints or {}

        self.batch_size = min(int(batch_size), int(robot_manager.num_envs))
        self.microbatch_wait_s = max(float(microbatch_wait_ms) / 1000.0, 0.0)
        if batch_size == 1:
            self.microbatch_wait_s = 0.0
        self.num_instances = num_instances
        self._debug = debug
        self.mode = mode
        self.relative_to_world_frame = relative_to_world_frame
        # info_links / extra_fk_link default falls through to instance resolution.
        self.info_links = info_links
        self.extra_fk_link = list(extra_fk_link or [])

        # World cfg cache (List[dict]); shared across instances.
        self.world_cfg_list: List[dict] = [{} for _ in range(robot_manager.num_envs)]
        self._world_lock = threading.Lock()

        # Round-robin counter for load balancing when all instances are idle.
        self._next_instance_idx = 0
        self._instance_idx_lock = threading.Lock()

        self.instances: List[_MotionGenInstance] = []
        for i in range(self.num_instances):
            instance = _MotionGenInstance(
                instance_id=i,
                robot_cfg=robot_cfg,
                batch_size=self.batch_size,
                robot_dof_name=robot_dof_name,
                robot_dof_name_active=robot_dof_name_active,
                robot_lock_joints=robot_lock_joints,
                robot_ignore_joints=robot_ignore_joints,
                robot_manager=robot_manager,
                robot_name=robot_name,
                device=device,
                planner_device=self.planner_devices[i],
                world_cfg_list=self.world_cfg_list,
                world_lock=self._world_lock,
                microbatch_wait_s=self.microbatch_wait_s,
                debug=debug,
                mode=mode,
                info_links=info_links,
                robot_add_joints=self.robot_add_joints,
                extra_fk_link=self.extra_fk_link,
                track_xyz_weight=track_xyz_weight,
                track_rpy_weight=track_rpy_weight,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
                relative_to_world_frame=self.relative_to_world_frame,
            )
            self.instances.append(instance)

        print(
            f"[MotionGenServer] Initialized with {self.num_instances} instance(s), "
            f"batch_size={self.batch_size}, microbatch_wait_ms={microbatch_wait_ms}, "
            f"planner_devices={[str(d) for d in self.planner_devices]}"
        )

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        for instance in self.instances:
            instance.shutdown(join=join, timeout_s=timeout_s)

    def update_world(self, env_ids: List[int], world_cfgs: List[Union[Scene, dict]]):
        """Update cached scene defs for env_ids (shared across instances).

        Accepts :class:`curobo.scene.Scene` instances or raw SceneCfg dicts.
        """
        if len(env_ids) != len(world_cfgs):
            raise ValueError("env_ids and world_cfgs length mismatch")
        with self._world_lock:
            for env_id, cfg in zip(env_ids, world_cfgs):
                self.world_cfg_list[int(env_id)] = cfg if cfg is not None else {}

    def _select_instance(self) -> _MotionGenInstance:
        idle_instances = [inst for inst in self.instances if inst.is_idle()]
        if len(idle_instances) > 0:
            if len(idle_instances) == len(self.instances):
                with self._instance_idx_lock:
                    selected_idx = self._next_instance_idx
                    self._next_instance_idx = (self._next_instance_idx + 1) % len(
                        self.instances
                    )
                selected = self.instances[selected_idx]
                if self._debug:
                    print(
                        f"[MotionGenServer][debug] All instances idle, using round-robin: "
                        f"selecting instance {selected.instance_id}"
                    )
                return selected
            least_loaded_idle = min(idle_instances, key=lambda inst: inst.queue_size())
            if self._debug and len(idle_instances) > 1:
                print(
                    f"[MotionGenServer][debug] Multiple idle instances "
                    f"({len(idle_instances)}), selecting instance "
                    f"{least_loaded_idle.instance_id} "
                    f"(queue_size={least_loaded_idle.queue_size()})"
                )
            return least_loaded_idle

        least_loaded = min(self.instances, key=lambda inst: inst.queue_size())
        if self._debug:
            print(
                f"[MotionGenServer][debug] All instances busy, selecting least-loaded "
                f"instance {least_loaded.instance_id} "
                f"(queue_size={least_loaded.queue_size()})"
            )
        return least_loaded

    def submit_plan(self, req: MotionGenPlanRequest):
        """Submit plan request; returns a Future with
        ``(actions, success, env_ids)``."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        instance = self._select_instance()
        instance.submit(fut, req)
        if self._debug:
            print(
                f"[MotionGenServer][debug] Submitted request envs={req.env_ids} "
                f"to instance {instance.instance_id} "
                f"(queue_size={instance.queue_size()})"
            )
        return fut

    def get_robot_spheres(self, joint_pos: torch.Tensor):
        """Return robot collision spheres for the given joint positions.

        In v2 this routes through ``planner.kinematics`` directly (no
        standalone kin_model instance). ``joint_pos`` is in sim joint
        order; reorder to cuRobo active-joint order, then fetch spheres.
        """
        if not torch.is_tensor(joint_pos):
            joint_pos = torch.as_tensor(
                joint_pos, device=self.device, dtype=torch.float32
            )
        else:
            joint_pos = joint_pos.to(self.device, dtype=torch.float32)
        if joint_pos.ndim != 1:
            joint_pos = joint_pos.view(-1)

        # Reorder to cuRobo active-joint order (same logic as v1).
        active_pos: List[float] = []
        for joint_name in self.robot_dof_name_active:
            if (
                joint_name in (self.robot_lock_joints or [])
                or joint_name in self.robot_ignore_joints
            ):
                continue
            if joint_name in self.robot_add_joints:
                active_pos.append(joint_pos[self.robot_add_joints[joint_name]].item())
                continue
            joint_ids, _ = self.robot_manager.robots[self.robot_name].find_joints(
                joint_name
            )
            if len(joint_ids) == 0:
                raise ValueError(f"Joint {joint_name} not found in joint_pos")
            joint_idx_int = int(joint_ids[0])
            if 0 <= joint_idx_int < joint_pos.shape[0]:
                active_pos.append(joint_pos[joint_idx_int].item())

        planner = self.instances[0]._planner
        kinematics = planner.kinematics
        active_pos_t = torch.tensor(
            active_pos, device=self.planner_devices[0], dtype=torch.float32
        ).view(1, -1)
        cur_js = JointState(
            position=active_pos_t,
            velocity=active_pos_t * 0.0,
            acceleration=active_pos_t * 0.0,
            jerk=active_pos_t * 0.0,
            joint_names=self.robot_dof_name_active,
        )
        ordered = cur_js.reorder(list(kinematics.joint_names))
        return kinematics.get_robot_as_spheres(ordered.position.contiguous())
