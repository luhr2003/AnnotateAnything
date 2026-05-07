"""cuRobo 2.0 DualMotionGenServer — locked-base ⊕ free-base motion planning for mobile robots.

v2 port of ``DualMotionGenServer`` (v1 layout in
``CUROBO_V2_01_CURRENT_INTERFACES.md §1.4``). Same Dual* semantic as
:class:`DualIKServer` — the name refers to the *solver pair*
(locked YAML + free YAML), not to arm count. Multi-tool-frame handling
is identical to :class:`MotionGenServer`.

Per-request ``lock_base`` picks which :class:`BatchMotionPlanner` runs;
each planner is wrapped with :class:`VariableBatchPlanner` for dynamic
per-call ``B``. ``force_single_plan`` branch from v1 is removed
(``CUROBO_V2_02_MIGRATION_PLAN.md §0.2``).
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


@dataclass
class DualMotionGenPlanRequest:
    """One dual motion-gen request.

    ``lock_base=True``  → locked-base planner; targets transformed to
                          robot base frame.
    ``lock_base=False`` → free-base planner (virtual base joints); targets
                          in world frame.

    ``target_pos`` accepted shapes (same as
    :class:`MotionGenServer.MotionGenPlanRequest`):
      - 2-D ``(N, eef_num * 7)``
      - 3-D ``(N, eef_num, 7)``

    Position-only ``(..., 3)`` shapes are rejected.
    """

    env_ids: List[int]
    target_pos: torch.Tensor
    robot_states: Dict[str, torch.Tensor]
    lock_base: bool = False


class _DualMotionGenInstance:
    """Dual motion-gen instance — locked + free :class:`BatchMotionPlanner` pair."""

    def __init__(
        self,
        instance_id: int,
        robot_cfg_locked: dict,
        robot_cfg_free: dict,
        batch_size: int,
        robot_dof_name: List[str],
        robot_dof_name_active: List[str],
        robot_lock_joints: Optional[List[str]],
        robot_ignore_joints: dict,
        robot_manager: RobotManager,
        robot_name: str,
        device: torch.device,
        planner_device: torch.device,
        world_cfg_list_locked: List[dict],
        world_cfg_list_free: List[dict],
        world_lock: threading.Lock,
        microbatch_wait_s: float,
        debug: bool,
        mode: str,
        info_links: Optional[List[str]],
        robot_add_joints: dict,
        base_joint_names: List[str],
        extra_fk_link: Optional[List[str]],
        track_xyz_weight: Tuple[float, float, float],
        track_rpy_weight: Tuple[float, float, float],
        left_arm_joints: Optional[List[str]] = None,
        right_arm_joints: Optional[List[str]] = None,
        pin_inactive_arm: bool = False,
        max_attempts: int = 2,
        enable_graph_attempt: int = 0,
    ):
        self.instance_id = instance_id
        self.robot_cfg_locked = robot_cfg_locked
        self.robot_cfg_free = robot_cfg_free
        self.batch_size = batch_size
        self.robot_dof_name = robot_dof_name
        self.base_joint_names = list(base_joint_names or [])

        # FREE-mode parameters (caller-provided).
        self.robot_dof_name_active = robot_dof_name_active
        self.robot_lock_joints = robot_lock_joints

        # LOCKED-mode parameters derived from FREE ∓ base joints.
        self.robot_dof_name_active_locked = [
            name for name in robot_dof_name_active if name not in self.base_joint_names
        ]
        self.robot_lock_joints_locked = list(robot_lock_joints or []) + list(
            self.base_joint_names
        )

        self.robot_ignore_joints = robot_ignore_joints or {}
        self.robot_add_joints = robot_add_joints or {}
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.device = device
        self.planner_device = torch.device(planner_device)
        self.world_cfg_list_locked = world_cfg_list_locked
        self.world_cfg_list_free = world_cfg_list_free
        self._world_lock = world_lock
        self.microbatch_wait_s = microbatch_wait_s
        self._debug = debug
        self.mode = mode
        self._track_xyz_weight = list(track_xyz_weight)
        self._track_rpy_weight = list(track_rpy_weight)
        self.max_attempts = int(max_attempts)
        self.enable_graph_attempt = int(enable_graph_attempt)

        # ``pin_inactive_arm`` mode — see DualIKServer for the
        # contract. When True and one tool slot is fully NaN'd, the FK
        # seed for that arm's joints is overridden with the LIVE sim
        # joint pos AND the criterion stays as track. trajopt is then
        # forced to keep that arm at its current values throughout the
        # planned trajectory instead of swinging it 90°+ to balance.
        self._pin_inactive_arm = bool(pin_inactive_arm)
        self._left_arm_joints = list(left_arm_joints or [])
        self._right_arm_joints = list(right_arm_joints or [])

        # Build both planners + wrappers.
        device_cfg = DeviceCfg(device=self.planner_device, dtype=torch.float32)
        # Per-env scene slots need a list of ``batch_size`` initial
        # SCENE DICTS (MotionPlannerCfg.create calls ``SceneCfg.create(x)``
        # per element which expects a dict, not a Scene instance). Empty
        # dicts are fine — see MotionGenServer comment.
        initial_scenes = [{} for _ in range(batch_size)]

        # Config tuned to match v2's ``batch_motion_gen_reacher.py`` /
        # ``dynamic_batch_motion_gen_reacher.py`` examples. See MotionGenServer
        # for the seed-count rationale.
        print(
            f"[DualMotionGenInstance {instance_id}] Creating LOCKED base MotionPlanner..."
        )
        # Motion planning is single-pose only — see MotionGenServer.
        locked_cfg = MotionPlannerCfg.create(
            robot=robot_cfg_locked,
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
        self.planner_locked = BatchMotionPlanner(locked_cfg)
        # Warmup deferred to worker thread — see __init__'s
        # _warmup_done_event comment.
        self._vbp_locked = VariableBatchPlanner(self.planner_locked)

        print(
            f"[DualMotionGenInstance {instance_id}] Creating FREE base MotionPlanner..."
        )
        free_cfg = MotionPlannerCfg.create(
            robot=robot_cfg_free,
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
        self.planner_free = BatchMotionPlanner(free_cfg)
        # Warmup deferred to worker thread — see __init__'s
        # _warmup_done_event comment.
        self._vbp_free = VariableBatchPlanner(self.planner_free)

        # Resolve tracked + info link lists per planner + apply criteria.
        # PlannerManager merges extra_fk_link into each YAML's tool_frames
        # before constructing robot_cfg_{locked,free}; here we derive tracked
        # = planner.tool_frames \\ extra_fk_link and apply disabled() to the
        # extras so IK + TrajOpt both zero-cost them.
        self._extra_fk_link: List[str] = list(extra_fk_link or [])
        self._tracked_locked = self._resolve_and_apply_criteria(
            self.planner_locked,
            self._extra_fk_link,
        )
        self._tracked_free = self._resolve_and_apply_criteria(
            self.planner_free,
            self._extra_fk_link,
        )
        self._info_links_locked = self._resolve_info_links(
            self.planner_locked,
            info_links,
            self._tracked_locked,
        )
        self._info_links_free = self._resolve_info_links(
            self.planner_free,
            info_links,
            self._tracked_free,
        )

        self._curobo_joint_names_locked = list(
            self.planner_locked.kinematics.joint_names
        )
        self._curobo_joint_names_free = list(self.planner_free.kinematics.joint_names)

        # Cache criteria templates for the per-env runtime disable path
        # (NaN preprocessing in ``_plan_one_batch``). Same template object
        # is fed to either planner — ToolPoseCriteria has no solver bind.
        self._track_criteria_template = ToolPoseCriteria.track_position_and_orientation(
            xyz=self._track_xyz_weight,
            rpy=self._track_rpy_weight,
        )
        self._disabled_criteria_template = ToolPoseCriteria.disabled()

        # Per-slot disable-mask cache, ONE per solver. Init = all-False
        # (matches the broadcast init criteria in ``_resolve_and_apply_criteria``).
        # ``_plan_one_batch`` only re-writes per-env criteria for slots
        # whose disable pattern changed — saves N CUDA buffer writes per
        # solve when targets are fully tracked (locomotion / WBC ticks).
        # Tracked-frame counts may differ between locked/free solvers, so
        # one cache per side. Sized to ``batch_size`` (= max_batch_size).
        self._persisted_disable_locked: torch.Tensor = torch.zeros(
            (self.batch_size, len(self._tracked_locked)),
            dtype=torch.bool,
            device=self.planner_device,
        )
        self._persisted_disable_free: torch.Tensor = torch.zeros(
            (self.batch_size, len(self._tracked_free)),
            dtype=torch.bool,
            device=self.planner_device,
        )

        # Worker thread / queue.
        self._queue: List[
            Tuple[concurrent.futures.Future, DualMotionGenPlanRequest]
        ] = []
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._shutdown = False

        # See IKServer __init__: worker's first CUDA op must happen before
        # any main-thread CUDA-graph capture, otherwise the graph reads
        # stale memory after the worker shuffles GPU pages on context init.
        self._warmup_done_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._warmup_done_event.wait()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _resolve_and_apply_criteria(
        self,
        planner: BatchMotionPlanner,
        extra_fk_link: List[str],
    ) -> List[str]:
        """Apply per-frame ToolPoseCriteria and return the tracked list.

        After PlannerManager's merge, ``planner.tool_frames`` = YAML tracked
        frames ++ extra_fk_link (dedup). Tracked = complement. ``extra_fk_link``
        entries not in THIS planner's tool_frames are silently skipped (locked
        and free YAMLs may declare different sets).
        """
        yaml_tool_frames = list(planner.tool_frames)
        fk_set = set(extra_fk_link) & set(yaml_tool_frames)
        tracked = [f for f in yaml_tool_frames if f not in fk_set]

        criteria: Dict[str, ToolPoseCriteria] = {}
        tracked_set = set(tracked)
        for frame in yaml_tool_frames:
            if frame in tracked_set:
                criteria[frame] = ToolPoseCriteria.track_position_and_orientation(
                    xyz=self._track_xyz_weight,
                    rpy=self._track_rpy_weight,
                )
            else:
                criteria[frame] = ToolPoseCriteria.disabled()
        # update_tool_pose_criteria propagates to BOTH IK and TrajOpt stages,
        # so extra_fk_link frames contribute ZERO cost to motion planning.
        planner.update_tool_pose_criteria(criteria)
        return tracked

    def _resolve_info_links(
        self,
        planner: BatchMotionPlanner,
        info_links: Optional[List[str]],
        tracked: List[str],
    ) -> List[str]:
        """Resolve info_links for this planner.

        Default (YAML omitted info_links): return the tracked list (= original
        YAML tool_frames order, without extras). Explicit info_links may
        include extras (their FK will be read post-solve).
        """
        yaml_tool_frames = list(planner.tool_frames)
        if info_links is None or len(info_links) == 0:
            return list(tracked)
        return [f for f in info_links if f in set(yaml_tool_frames)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        with self._queue_cv:
            self._shutdown = True
            self._queue_cv.notify_all()
        if join:
            self._worker.join(timeout=timeout_s)

    def submit(self, fut: concurrent.futures.Future, req: DualMotionGenPlanRequest):
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
    # Frame + joint state helpers (mirror DualIKInstance / MotionGenInstance)
    # ------------------------------------------------------------------

    def _world_to_robot_frame(
        self,
        target_pos: torch.Tensor,
        robot_base_pose: torch.Tensor,
        robot_base_quat: torch.Tensor,
        env_ids: List[int],
    ) -> torch.Tensor:
        """[N, eef_num, 7] or [N, 7] world → robot-base frame (for lock_base=True)."""
        device = robot_base_pose.device
        target_tensor = target_pos.to(device=device, dtype=torch.float32)

        if target_tensor.ndim == 2:
            target_tensor = target_tensor.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        n, num_eef, _ = target_tensor.shape

        env_ids_tensor = torch.tensor(env_ids, device=device, dtype=torch.long)
        base_positions = robot_base_pose[env_ids_tensor]
        base_quats = robot_base_quat[env_ids_tensor]

        base_positions = base_positions.unsqueeze(1).expand(-1, num_eef, -1)
        base_quats = base_quats.unsqueeze(1).expand(-1, num_eef, -1)

        target_positions = target_tensor[:, :, :3]
        target_quats = target_tensor[:, :, 3:]

        pos_relative = target_positions - base_positions
        base_rot_matrices = quat_to_rot_matrix(base_quats.reshape(-1, 4))
        base_rot_inv = base_rot_matrices.transpose(-2, -1)

        pos_relative_flat = pos_relative.reshape(-1, 3, 1)
        pos_local = (
            torch.bmm(base_rot_inv, pos_relative_flat)
            .squeeze(-1)
            .reshape(n, num_eef, 3)
        )
        base_quats_inv = quat_inv(base_quats.reshape(-1, 4))
        quats_local = quat_mul(base_quats_inv, target_quats.reshape(-1, 4)).reshape(
            n, num_eef, 4
        )

        result = torch.cat([pos_local, quats_local], dim=-1)
        if squeeze_output:
            result = result.squeeze(1)
        return result

    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _arm_joints_for_tool(self, frame: str) -> List[str]:
        """Pick which arm-joint group corresponds to a tool frame.

        Heuristic: lower-cased frame name containing ``right`` or
        starting with ``r_`` → ``right_arm_joints``; ``left`` /
        ``l_`` → ``left_arm_joints``. Returns ``[]`` if neither side
        is declared in the YAML so callers can fall through to the
        default behaviour. Mirror of ``_DualIKInstance._arm_joints_for_tool``.
        """
        n = frame.lower()
        if n.startswith("r_") or "right" in n:
            return self._right_arm_joints
        if n.startswith("l_") or "left" in n:
            return self._left_arm_joints
        return []

    def _build_pinned_fk_state(
        self,
        current_state: JointState,
        planner,
        tracked: List[str],
        disable_mask: torch.Tensor,
        batch_env_ids: List[int],
    ) -> Tuple[JointState, torch.Tensor]:
        """Same contract as :meth:`_DualIKInstance._build_pinned_fk_state`.
        Override the per-arm joint columns of the FK seed with the LIVE
        sim joint pos for any tool slot the caller marked disabled, so
        the FK-substituted target reflects the arm's actual current
        pose. Returns ``(fk_state, effective_disable)``;
        ``effective_disable`` is all-False when pin_inactive_arm is on
        so trajopt keeps tracking those slots.
        """
        device = current_state.position.device
        sim_robot = self.robot_manager.robots[self.robot_name]
        sim_joint_pos = sim_robot.data.joint_pos
        sim_joint_names = list(sim_robot.joint_names)
        sim_jname_to_idx = {n: i for i, n in enumerate(sim_joint_names)}
        env_ids_t = torch.tensor(batch_env_ids, device=device, dtype=torch.long)
        live_jp = sim_joint_pos.to(device).index_select(0, env_ids_t)

        curobo_jnames = list(planner.kinematics.joint_names)
        curobo_jname_to_idx = {n: i for i, n in enumerate(curobo_jnames)}

        new_pos = current_state.position.clone()
        for li, frame in enumerate(tracked):
            slot_disabled = disable_mask[:, li]
            if not bool(slot_disabled.any().item()):
                continue
            arm_joints = self._arm_joints_for_tool(frame)
            if not arm_joints:
                continue
            for jname in arm_joints:
                curobo_idx = curobo_jname_to_idx.get(jname)
                sim_idx = sim_jname_to_idx.get(jname)
                if curobo_idx is None or sim_idx is None:
                    continue
                if sim_idx >= live_jp.shape[1]:
                    continue
                new_pos[slot_disabled, curobo_idx] = live_jp[slot_disabled, sim_idx].to(
                    new_pos.dtype
                )

        fk_state = JointState(
            position=new_pos,
            velocity=torch.zeros_like(new_pos),
            acceleration=torch.zeros_like(new_pos),
            jerk=None,
            joint_names=curobo_jnames,
        )
        effective_disable = torch.zeros_like(disable_mask)
        return fk_state, effective_disable

    def _build_joint_state(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        lock_base: bool,
    ) -> JointState:
        """Build a JointState on the planner device in cuRobo joint order."""
        if lock_base:
            planner = self.planner_locked
            active_list = self.robot_dof_name_active_locked
            lock_list = self.robot_lock_joints_locked
            add_joints: Dict[str, int] = {}
        else:
            planner = self.planner_free
            active_list = self.robot_dof_name_active
            lock_list = self.robot_lock_joints
            add_joints = self.robot_add_joints

        joint_names = planner.kinematics.joint_names
        batch = base_pos.shape[0]

        js_positions: List[torch.Tensor] = []
        for joint_name in joint_names:
            if joint_name in self.robot_ignore_joints:
                js_positions.append(
                    torch.full(
                        (batch,),
                        float(self.robot_ignore_joints[joint_name]),
                        device=base_pos.device,
                        dtype=base_pos.dtype,
                    )
                )
            elif joint_name in (lock_list or []):
                js_positions.append(
                    torch.zeros(batch, device=base_pos.device, dtype=base_pos.dtype)
                )
            elif joint_name in add_joints:
                joint_idx = add_joints[joint_name]
                js_positions.append(joint_pos[:, joint_idx])
            else:
                joint_ids, _ = self.robot_manager.robots[self.robot_name].find_joints(
                    joint_name
                )
                if len(joint_ids) > 0:
                    joint_idx = int(joint_ids[0])
                    if joint_idx < joint_pos.shape[1]:
                        js_positions.append(joint_pos[:, joint_idx])
                    else:
                        js_positions.append(
                            torch.zeros(
                                batch, device=base_pos.device, dtype=base_pos.dtype
                            )
                        )
                else:
                    js_positions.append(
                        torch.zeros(batch, device=base_pos.device, dtype=base_pos.dtype)
                    )

        js_pos = torch.stack(js_positions, dim=1).to(
            device=self.planner_device, dtype=torch.float32
        )
        return JointState(
            position=js_pos,
            velocity=torch.zeros_like(js_pos),
            acceleration=torch.zeros_like(js_pos),
            jerk=torch.zeros_like(js_pos),
            joint_names=list(joint_names),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_request(
        self,
        req: DualMotionGenPlanRequest,
    ) -> Tuple[
        List[int],
        torch.Tensor,  # target_pos (frame depends on lock_base)
        torch.Tensor,  # base_pos
        torch.Tensor,  # base_quat
        torch.Tensor,  # joint_pos (sim order)
        JointState,  # current_state in cuRobo joint order
        bool,  # lock_base
    ]:
        env_ids = [int(x) for x in req.env_ids]
        target_pos = req.target_pos
        lock_base = req.lock_base

        # Caller contract: 2-D ``(N, eef_num * 7)`` or 3-D ``(N, eef_num, 7)``.
        # Position-only ``(..., 3)`` shapes are NOT accepted — see
        # ``MotionGenServer._preprocess_request`` for the full rationale.
        if target_pos.ndim == 2:
            flat = target_pos.shape[1]
            assert flat % 7 == 0, (
                f"DualMotionGenPlanRequest.target_pos 2-D last dim must "
                f"be a multiple of 7; got {flat}. Position-only shapes "
                f"are not accepted."
            )
            target_pos = target_pos.view(target_pos.shape[0], flat // 7, 7)
        elif target_pos.ndim == 3:
            assert target_pos.shape[-1] == 7, (
                f"DualMotionGenPlanRequest.target_pos 3-D last dim must "
                f"be 7; got shape {tuple(target_pos.shape)}. Position-only "
                f"(N, eef_num, 3) shapes are not accepted."
            )
        else:
            raise AssertionError(
                f"DualMotionGenPlanRequest.target_pos must be 2-D "
                f"(N, eef_num*7) or 3-D (N, eef_num, 7); got "
                f"{tuple(target_pos.shape)}"
            )

        base_pos = req.robot_states["base_pos"]
        base_quat = req.robot_states["base_quat"]
        joint_pos = req.robot_states["joint_pos"]

        env_ids_tensor = torch.tensor(env_ids, device=base_pos.device, dtype=torch.long)
        base_pos_envs = base_pos[env_ids_tensor]
        base_quat_envs = base_quat[env_ids_tensor]
        joint_pos_envs = joint_pos[env_ids_tensor]

        if lock_base:
            target_pos = self._world_to_robot_frame(
                target_pos, base_pos, base_quat, env_ids
            )

        js = self._build_joint_state(
            base_pos_envs, base_quat_envs, joint_pos_envs, lock_base
        )
        return (
            env_ids,
            target_pos,
            base_pos_envs,
            base_quat_envs,
            joint_pos_envs,
            js,
            lock_base,
        )

    # ------------------------------------------------------------------
    # FK export (position-mode)
    # ------------------------------------------------------------------

    def _build_common_js_names(
        self, cmd_plan: JointState, lock_base: bool
    ) -> List[str]:
        robot_lock_joints = (
            self.robot_lock_joints_locked if lock_base else self.robot_lock_joints
        )
        common: List[str] = []
        for x in self.robot_dof_name:
            if x in (robot_lock_joints or []) or x in (self.robot_ignore_joints or []):
                continue
            if x in cmd_plan.joint_names:
                common.append(x)
        return common

    def _trajectory_to_eef_pose(
        self,
        action_traj: torch.Tensor,  # [T, 2 * active_dof]
        base_pos: torch.Tensor,  # [3]
        base_quat: torch.Tensor,  # [4]
        joint_pos: torch.Tensor,  # [num_joints] (unused)
        action_joint_names: List[str],
        lock_base: bool,
        disabled_frames: Optional[List[str]] = None,
        env_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert joint-space trajectory to EEF pose trajectory.

        For ``lock_base=True``: planner kinematics is rooted at the robot base;
            FK returns robot-frame poses, we transform back to world frame.
        For ``lock_base=False``: planner kinematics includes virtual base joints;
            FK is already in world frame.
        Returns ``[T, len(info_links) * 7]``.

        ``pin_inactive_arm`` mode (vega): if ``disabled_frames`` is
        non-empty and the YAML declared ``left_arm_joints`` /
        ``right_arm_joints``, the corresponding arm columns of ``q``
        are overridden with the LIVE sim joint pos at ``env_id``
        before FK. The motion-gen plan itself is unchanged (the
        disabled tool was disabled during the solve as usual); we
        just rewrite the inactive arm's trajectory to "stay at sim
        values" so the EEF readout doesn't carry the planner's
        wandering L_arm. Useful for downstream consumers that drive
        each EEF directly off this readout (e.g., pink IK action term).
        """
        planner = self.planner_locked if lock_base else self.planner_free
        info_links = self._info_links_locked if lock_base else self._info_links_free
        curobo_jnames = (
            self._curobo_joint_names_locked
            if lock_base
            else self._curobo_joint_names_free
        )
        lock_list = (
            self.robot_lock_joints_locked if lock_base else self.robot_lock_joints
        )

        action_dim = action_traj.shape[1] // 2
        action_joint_pos = action_traj[:, :action_dim]
        action_name_to_idx = {name: idx for idx, name in enumerate(action_joint_names)}

        per_step: List[List[float]] = []
        for step_idx in range(action_joint_pos.shape[0]):
            ordered: List[float] = []
            for name in curobo_jnames:
                if name in action_name_to_idx:
                    j = action_name_to_idx[name]
                    ordered.append(action_joint_pos[step_idx, j].item())
                elif name in (lock_list or []):
                    ordered.append(0.0)
                elif name in self.robot_ignore_joints:
                    ordered.append(float(self.robot_ignore_joints[name]))
                else:
                    raise ValueError(
                        f"Joint {name} not found in action_joint_names and no "
                        f"lock/ignore entry"
                    )
            per_step.append(ordered)

        q = torch.tensor(per_step, device=self.planner_device, dtype=torch.float32)

        # Pin-inactive-arm override: rewrite the disabled side's arm
        # joint columns of every waypoint with sim's CURRENT joint pos
        # for the relevant env. Active during ``pin_inactive_arm=True``
        # in YAML; no-op otherwise.
        if (
            self._pin_inactive_arm
            and disabled_frames
            and env_id is not None
            and (self._left_arm_joints or self._right_arm_joints)
        ):
            sim_robot = self.robot_manager.robots[self.robot_name]
            sim_jp_env = sim_robot.data.joint_pos[env_id].to(
                device=q.device, dtype=q.dtype
            )
            sim_jname_to_idx = {n: i for i, n in enumerate(list(sim_robot.joint_names))}
            curobo_jname_to_idx = {n: i for i, n in enumerate(curobo_jnames)}
            for frame in disabled_frames:
                for jname in self._arm_joints_for_tool(frame):
                    cidx = curobo_jname_to_idx.get(jname)
                    sidx = sim_jname_to_idx.get(jname)
                    if cidx is None or sidx is None:
                        continue
                    if sidx >= sim_jp_env.shape[0]:
                        continue
                    q[:, cidx] = sim_jp_env[sidx]
        js = JointState.from_position(q, joint_names=curobo_jnames)
        kin = planner.compute_kinematics(js)

        T = q.shape[0]
        base_pos_w = (
            base_pos.to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(T, 1)
        )
        base_quat_w = (
            base_quat.to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(T, 1)
        )
        base_rot = quat_to_rot_matrix(base_quat_w)

        poses_dict = kin.tool_poses.to_dict()
        out_per_link: List[torch.Tensor] = []
        for link in info_links:
            if link not in poses_dict:
                raise ValueError(
                    f"info_link {link!r} not present in planner tool_poses "
                    f"({list(poses_dict.keys())})."
                )
            p = poses_dict[link]
            pos_local = p.position.to(device=self.device, dtype=torch.float32)
            quat_local = p.quaternion.to(device=self.device, dtype=torch.float32)
            if lock_base:
                pos_world = base_pos_w + torch.bmm(
                    base_rot, pos_local.unsqueeze(-1)
                ).squeeze(-1)
                quat_world = quat_mul(base_quat_w, quat_local)
                out_per_link.append(torch.cat([pos_world, quat_world], dim=1))
            else:
                # Planner kin chain already rooted at world (virtual base joints).
                out_per_link.append(torch.cat([pos_local, quat_local], dim=1))
        return torch.cat(out_per_link, dim=1)

    # ------------------------------------------------------------------
    # Plan one batch
    # ------------------------------------------------------------------

    def _plan_one_batch(
        self,
        batch_env_ids: List[int],
        batch_target_pos: torch.Tensor,  # [B, eef_num, 7]
        batch_full_js: JointState,
        batch_base_pos: torch.Tensor,
        batch_base_quat: torch.Tensor,
        batch_joint_pos: torch.Tensor,
        lock_base: bool,
    ) -> Tuple[List[Optional[torch.Tensor]], List[bool]]:
        planner = self.planner_locked if lock_base else self.planner_free
        vbp = self._vbp_locked if lock_base else self._vbp_free
        tracked = self._tracked_locked if lock_base else self._tracked_free
        world_cfg_list = (
            self.world_cfg_list_locked if lock_base else self.world_cfg_list_free
        )

        B = len(batch_env_ids)
        assert B == batch_target_pos.shape[0] == batch_full_js.position.shape[0]
        assert batch_target_pos.shape[1] == len(tracked), (
            f"target_pos eef_num={batch_target_pos.shape[1]} != "
            f"len(tracked_tool_frames)={len(tracked)}"
        )

        with self._world_lock:
            for slot, env_id in enumerate(batch_env_ids):
                env_id_int = int(env_id)
                if env_id_int < 0 or env_id_int >= len(world_cfg_list):
                    raise ValueError(
                        f"env_id {env_id_int} out of range [0, {len(world_cfg_list)})"
                    )
                scene_cfg = world_cfg_list[env_id_int]
                if isinstance(scene_cfg, Scene):
                    scene = scene_cfg
                elif isinstance(scene_cfg, dict) and scene_cfg:
                    scene = Scene.create(scene_cfg)
                else:
                    scene = Scene()
                planner.scene_collision_checker.load_collision_model(
                    scene,
                    env_idx=int(slot),
                )
                if self._debug:
                    log_scene_slot_load(
                        scene,
                        int(slot),
                        env_id_int,
                        tag=f"DualMG-{'locked' if lock_base else 'free'}",
                    )

        # ----- NaN preprocessing → per-env disable + FK substitution -----
        # Same logic as MotionGenServer — see that file for full rationale.
        target_dev = batch_target_pos.to(self.planner_device)
        disable_mask = detect_nan_disable_single(target_dev)  # (B, eef_num)

        fk_kin = planner.compute_kinematics(batch_full_js)
        fk_pose_per_frame: Dict[str, Pose] = {}
        for frame in planner.tool_frames:
            fk_pose_per_frame[frame] = fk_kin.tool_poses.get_link_pose(
                frame,
                make_contiguous=True,
            )

        target_subbed = target_dev.clone()
        if bool(disable_mask.any().item()):
            for li, frame in enumerate(tracked):
                row_mask = disable_mask[:, li]
                if not bool(row_mask.any().item()):
                    continue
                p = fk_pose_per_frame[frame]
                fk_pos = p.position.view(-1, 3)
                fk_quat = p.quaternion.view(-1, 4)
                target_subbed[row_mask, li, :3] = fk_pos[row_mask]
                target_subbed[row_mask, li, 3:] = fk_quat[row_mask]

        # Per-env runtime criteria — see MotionGenServer for the cache
        # rationale. ``env_idx`` is the SOLVER SLOT INDEX (0..max_batch_size-1).
        # Per-solver cache (locked or free); the other solver's slot rows
        # are untouched.
        persisted = (
            self._persisted_disable_locked
            if lock_base
            else self._persisted_disable_free
        )
        cached_dev = persisted[:B]
        if cached_dev.device != disable_mask.device:
            cached_dev = cached_dev.to(disable_mask.device)
        slot_changed = (cached_dev != disable_mask).any(dim=-1)  # (B,) bool
        if not bool(slot_changed.any().item()):
            changed_idx: List[int] = []
        else:
            changed_idx = slot_changed.nonzero(as_tuple=True)[0].tolist()
        for b in changed_idx:
            crit: Dict[str, ToolPoseCriteria] = {}
            for li, frame in enumerate(tracked):
                crit[frame] = (
                    self._disabled_criteria_template
                    if bool(disable_mask[b, li])
                    else self._track_criteria_template
                )
            planner.update_tool_pose_criteria_per_env(b, crit)
        if changed_idx:
            persisted[:B] = disable_mask.to(persisted.device)

        pose_dict: Dict[str, Pose] = {}
        for i, frame in enumerate(tracked):
            pose_dict[frame] = Pose(
                position=target_subbed[:, i, :3].contiguous(),
                quaternion=target_subbed[:, i, 3:].contiguous(),
            )
        # Non-tracked / info-only frames — reuse cached FK (no extra call).
        needs_fk = [f for f in planner.tool_frames if f not in pose_dict]
        for frame in needs_fk:
            pose_dict[frame] = fk_pose_per_frame[frame]
        goal = GoalToolPose.from_poses(
            pose_dict,
            ordered_tool_frames=list(planner.tool_frames),
            num_goalset=1,
        )

        actions: List[Optional[torch.Tensor]] = [None] * B
        success: List[bool] = [False] * B
        common_js_names_list: List[Optional[List[str]]] = [None] * B

        mode_label = "locked" if lock_base else "free"
        if self._debug:
            log_joint_state_submit(
                batch_full_js,
                batch_env_ids,
                tag=f"DualMG-{mode_label}",
            )
        t0 = time.time()
        result = vbp.plan_pose(
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
            # collision blocking every IK candidate. Dump enough state to
            # let the caller diagnose without enabling full debug.
            tgt_world = batch_target_pos.detach().cpu()
            base_p = batch_base_pos.detach().cpu()
            base_q = batch_base_quat.detach().cpu()
            print(
                f"[DualMotionGenServer][FAIL] Instance {self.instance_id} "
                f"[{mode_label}] plan_pose returned None — no IK seed survived "
                f"(self-collision / world-collision / unreachable). "
                f"envs={batch_env_ids} dt={plan_dt:.4f}s"
            )
            for b, env_id in enumerate(batch_env_ids):
                print(
                    f"[DualMotionGenServer][FAIL]   env={env_id} slot={b}"
                    f" base_pos={base_p[b].tolist()}"
                    f" base_quat={base_q[b].tolist()}"
                    f" target=\n{tgt_world[b].tolist()}"
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
            # All envs failed — print the per-env reason summary so callers
            # can correlate which envs missed and by how much.
            tgt_world = batch_target_pos.detach().cpu()
            print(
                f"[DualMotionGenServer][FAIL] Instance {self.instance_id} "
                f"[{mode_label}] plan_pose envs={batch_env_ids} ALL FAILED "
                f"pos_err_max={pos_err:.4f}m rot_err_max={rot_err:.4f}rad "
                f"dt={plan_dt:.4f}s "
                f"(thresholds: pos<=0.005m rot<=0.05rad — see solver cfg). "
                f"Common causes: target out of workspace, target inside "
                f"obstacle, paired-IK blocking convergence on this G."
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
                    f"[DualMotionGenServer][FAIL]   env={env_id} slot={b}"
                    f" pos_err={pe:.4f}m rot_err={re_:.4f}rad"
                    f" target=\n{tgt_world[b].tolist()}"
                )
            return actions, success

        print(
            f"[DualMotionGenServer][debug] Instance {self.instance_id} [{mode_label}] "
            f"plan_pose envs={batch_env_ids} success={succ_any} "
            f"pos_err_max={pos_err:.4f}m rot_err_max={rot_err:.4f}rad dt={plan_dt:.4f}s"
        )

        interp = result.interpolated_trajectory
        last = result.interpolated_last_tstep
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
            common_js_names = self._build_common_js_names(cmd_plan, lock_base)
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

        if self.mode == "position":
            for slot in range(B):
                if actions[slot] is None:
                    continue
                if common_js_names_list[slot] is None:
                    raise RuntimeError(
                        "Missing common_js_names for position mode conversion"
                    )
                # ``disabled_frames`` for this slot — list of tool-frame
                # names whose target was NaN'd by the caller. Consumed
                # by ``_trajectory_to_eef_pose`` only when
                # ``pin_inactive_arm`` is on; otherwise unused.
                disabled_frames = [
                    tracked[li]
                    for li in range(len(tracked))
                    if bool(disable_mask[slot, li].item())
                ]
                actions[slot] = self._trajectory_to_eef_pose(
                    actions[slot],
                    batch_base_pos[slot],
                    batch_base_quat[slot],
                    batch_joint_pos[slot],
                    common_js_names_list[slot],
                    lock_base,
                    disabled_frames=disabled_frames,
                    env_id=int(batch_env_ids[slot]),
                )
        return actions, success

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self):
        # Full planner warmup on this thread for both locked and free
        # solvers — JITs all warp kernels into the worker's CUDA context
        # so subsequent main-thread CUDA-graph captures see a stable
        # module layout. See __init__ comment.
        try:
            self.planner_locked.warmup(enable_graph=False, num_warmup_iterations=3)
            self.planner_free.warmup(enable_graph=False, num_warmup_iterations=3)
            torch.cuda.synchronize(self.planner_device)
        finally:
            self._warmup_done_event.set()

        while True:
            with self._queue_cv:
                while not self._shutdown and len(self._queue) == 0:
                    self._queue_cv.wait()
                if self._shutdown:
                    return

                batch: List[
                    Tuple[concurrent.futures.Future, DualMotionGenPlanRequest]
                ] = []
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
                        if len(batch) >= self.batch_size * 2:
                            break
                else:
                    while len(self._queue) > 0 and len(batch) < self.batch_size * 2:
                        batch.append(self._queue.pop(0))

            preprocessed: List[Tuple] = []
            try:
                t_req0 = time.time()
                for fut, req in batch:
                    try:
                        result = self._preprocess_request(req)
                        preprocessed.append((fut,) + result)
                    except Exception as ex:
                        tb = traceback.format_exc()
                        print(
                            f"[DualMotionGenServer][ERROR] Instance {self.instance_id} "
                            f"preprocessing failed for request envs={req.env_ids} "
                            f"lock_base={req.lock_base}: "
                            f"{type(ex).__name__}: {ex}\n{tb}"
                        )
                        fut.set_exception(ex)

                if len(preprocessed) == 0:
                    continue

                # Group by lock_base.
                locked_reqs = [r for r in preprocessed if r[7]]
                free_reqs = [r for r in preprocessed if not r[7]]

                for requests, lb in [(locked_reqs, True), (free_reqs, False)]:
                    if len(requests) == 0:
                        continue
                    self._process_request_group(requests, lb)

                if self._debug:
                    print(
                        f"[DualMotionGenServer][debug] Instance {self.instance_id} served "
                        f"{len(preprocessed)} request(s) "
                        f"(locked={len(locked_reqs)}, free={len(free_reqs)}) "
                        f"dt_total={time.time() - t_req0:.4f}s"
                    )
            except Exception as ex:
                print(
                    f"[DualMotionGenServer][ERROR] Instance {self.instance_id} "
                    f"worker exception — {type(ex).__name__}: {ex}\n"
                    f"{traceback.format_exc()}"
                )
                for item in preprocessed:
                    if not item[0].done():
                        item[0].set_exception(ex)
                for fut, req in batch:
                    if not fut.done():
                        fut.set_exception(ex)

    def _process_request_group(self, requests: List[Tuple], lock_base: bool) -> None:
        """Merge requests of the same lock_base, chunk, plan, fan out."""
        merged_env_ids: List[int] = []
        merged_targets: List[torch.Tensor] = []
        merged_base_pos_list: List[torch.Tensor] = []
        merged_base_quat_list: List[torch.Tensor] = []
        merged_joint_pos_list: List[torch.Tensor] = []
        merged_js_list: List[JointState] = []
        per_req_indices: Dict[concurrent.futures.Future, List[int]] = {}
        per_req_env_ids: Dict[concurrent.futures.Future, List[int]] = {}

        seen_env = set()
        for item in requests:
            fut, env_ids, tgt, bp, bq, jp, js, _ = item
            per_req_env_ids[fut] = env_ids
            idx_list: List[int] = []
            for j, env_id in enumerate(env_ids):
                if env_id in seen_env:
                    existing = merged_env_ids.index(env_id)
                    idx_list.append(existing)
                    continue
                seen_env.add(env_id)
                merged_env_ids.append(env_id)
                merged_targets.append(tgt[j])
                merged_base_pos_list.append(bp[j])
                merged_base_quat_list.append(bq[j])
                merged_joint_pos_list.append(jp[j])
                merged_js_list.append(js[j : j + 1])
                idx_list.append(len(merged_env_ids) - 1)
            per_req_indices[fut] = idx_list

        if len(merged_env_ids) == 0:
            for item in requests:
                item[0].set_result(([], [], []))
            return

        merged_target_tensor = torch.stack(merged_targets, dim=0)
        merged_base_pos = torch.stack(merged_base_pos_list, dim=0)
        merged_base_quat = torch.stack(merged_base_quat_list, dim=0)
        merged_joint_pos = torch.stack(merged_joint_pos_list, dim=0)
        # v2: ``JointState.stack()`` is deprecated; use the functional form.
        merged_full_js = merged_js_list[0]
        for j in merged_js_list[1:]:
            merged_full_js = stack_joint_states(merged_full_js, j)

        merged_actions: List[Optional[torch.Tensor]] = [None] * len(merged_env_ids)
        merged_success: List[bool] = [False] * len(merged_env_ids)

        for s in range(0, len(merged_env_ids), self.batch_size):
            e = min(s + self.batch_size, len(merged_env_ids))
            bids = merged_env_ids[s:e]
            a, ok = self._plan_one_batch(
                bids,
                merged_target_tensor[s:e],
                merged_full_js[s:e],
                merged_base_pos[s:e],
                merged_base_quat[s:e],
                merged_joint_pos[s:e],
                lock_base,
            )
            merged_actions[s:e] = a
            merged_success[s:e] = ok

        for item in requests:
            fut = item[0]
            idx_list = per_req_indices.get(fut)
            stored_env_ids = per_req_env_ids.get(fut)
            if idx_list is None or stored_env_ids is None:
                raise ValueError(f"Missing mapping for future {id(fut)}")
            actions_out = [merged_actions[ii] for ii in idx_list]
            success_out = [merged_success[ii] for ii in idx_list]
            fut.set_result((actions_out, success_out, stored_env_ids))


class DualMotionGenServer:
    """cuRobo 2.0 dual motion-gen service — locked-base ⊕ free-base pair."""

    dual_mode: bool = True

    def __init__(
        self,
        robot_manager: RobotManager,
        robot_cfg_locked: dict,
        robot_cfg_free: dict,
        robot_name: str,
        robot_dof_name: List[str],
        robot_dof_name_active: List[str],
        robot_lock_joints: Optional[List[str]],
        robot_ignore_joints: dict,
        robot_add_joints: dict,
        device: torch.device,
        base_joint_names: List[str],
        batch_size: int = 2,
        microbatch_wait_ms: float = 200.0,
        num_instances: int = 2,
        debug: bool = False,
        mode: str = "joint",
        info_links: Optional[List[str]] = None,
        extra_fk_link: Optional[List[str]] = None,
        track_xyz_weight: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        track_rpy_weight: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        left_arm_joints: Optional[List[str]] = None,
        right_arm_joints: Optional[List[str]] = None,
        pin_inactive_arm: bool = False,
        max_attempts: int = 2,
        enable_graph_attempt: int = 0,
        planner_devices: Optional[Union[str, torch.device, Sequence]] = None,
    ):
        self.robot_manager = robot_manager
        self.robot_cfg_locked = robot_cfg_locked
        self.robot_cfg_free = robot_cfg_free
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
        self.base_joint_names = list(base_joint_names or [])

        self.batch_size = min(int(batch_size), int(robot_manager.num_envs))
        self.microbatch_wait_s = max(float(microbatch_wait_ms) / 1000.0, 0.0)
        if batch_size == 1:
            self.microbatch_wait_s = 0.0
        self.num_instances = num_instances
        self._debug = debug
        self.mode = mode
        self.info_links = info_links
        self.extra_fk_link = list(extra_fk_link or [])

        # Separate world cfg caches (locked / free).
        self.world_cfg_list_locked: List[dict] = [
            {} for _ in range(robot_manager.num_envs)
        ]
        self.world_cfg_list_free: List[dict] = [
            {} for _ in range(robot_manager.num_envs)
        ]
        self._world_lock = threading.Lock()

        self._next_instance_idx = 0
        self._instance_idx_lock = threading.Lock()

        self.instances: List[_DualMotionGenInstance] = []
        for i in range(self.num_instances):
            instance = _DualMotionGenInstance(
                instance_id=i,
                robot_cfg_locked=robot_cfg_locked,
                robot_cfg_free=robot_cfg_free,
                batch_size=self.batch_size,
                robot_dof_name=robot_dof_name,
                robot_dof_name_active=robot_dof_name_active,
                robot_lock_joints=robot_lock_joints,
                robot_ignore_joints=robot_ignore_joints,
                robot_manager=robot_manager,
                robot_name=robot_name,
                device=device,
                planner_device=self.planner_devices[i],
                world_cfg_list_locked=self.world_cfg_list_locked,
                world_cfg_list_free=self.world_cfg_list_free,
                world_lock=self._world_lock,
                microbatch_wait_s=self.microbatch_wait_s,
                debug=debug,
                mode=mode,
                info_links=info_links,
                robot_add_joints=self.robot_add_joints,
                base_joint_names=self.base_joint_names,
                extra_fk_link=self.extra_fk_link,
                track_xyz_weight=track_xyz_weight,
                track_rpy_weight=track_rpy_weight,
                left_arm_joints=left_arm_joints,
                right_arm_joints=right_arm_joints,
                pin_inactive_arm=pin_inactive_arm,
                max_attempts=max_attempts,
                enable_graph_attempt=enable_graph_attempt,
            )
            self.instances.append(instance)

        print(
            f"[DualMotionGenServer] Initialized with {self.num_instances} instance(s), "
            f"batch_size={self.batch_size}, microbatch_wait_ms={microbatch_wait_ms}, "
            f"planner_devices={[str(d) for d in self.planner_devices]}"
        )

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        for instance in self.instances:
            instance.shutdown(join=join, timeout_s=timeout_s)

    def update_world(
        self,
        env_ids: List[int],
        world_cfgs: List[dict],
        relative_to_world_frame: bool = True,
    ):
        """Update one mode's world cache.

        ``relative_to_world_frame=True``  → free-base cache.
        ``relative_to_world_frame=False`` → locked-base cache.
        """
        if len(env_ids) != len(world_cfgs):
            raise ValueError("env_ids and world_cfgs length mismatch")
        with self._world_lock:
            if relative_to_world_frame:
                for env_id, cfg in zip(env_ids, world_cfgs):
                    self.world_cfg_list_free[int(env_id)] = cfg or {}
            else:
                for env_id, cfg in zip(env_ids, world_cfgs):
                    self.world_cfg_list_locked[int(env_id)] = cfg or {}

    def update_world_dual(
        self,
        env_ids: List[int],
        world_cfgs_locked: List[dict],
        world_cfgs_free: List[dict],
    ):
        if len(env_ids) != len(world_cfgs_locked) or len(env_ids) != len(
            world_cfgs_free
        ):
            raise ValueError("env_ids and world_cfgs length mismatch")
        with self._world_lock:
            for env_id, cfg_l, cfg_f in zip(
                env_ids, world_cfgs_locked, world_cfgs_free
            ):
                self.world_cfg_list_locked[int(env_id)] = cfg_l or {}
                self.world_cfg_list_free[int(env_id)] = cfg_f or {}

    def _select_instance(self) -> _DualMotionGenInstance:
        idle = [inst for inst in self.instances if inst.is_idle()]
        if idle:
            if len(idle) == len(self.instances):
                with self._instance_idx_lock:
                    selected_idx = self._next_instance_idx
                    self._next_instance_idx = (self._next_instance_idx + 1) % len(
                        self.instances
                    )
                return self.instances[selected_idx]
            return min(idle, key=lambda inst: inst.queue_size())
        return min(self.instances, key=lambda inst: inst.queue_size())

    def submit_plan(self, req: DualMotionGenPlanRequest):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        instance = self._select_instance()
        instance.submit(fut, req)
        if self._debug:
            lock_str = "LOCKED" if req.lock_base else "FREE"
            print(
                f"[DualMotionGenServer][debug] Submitted request lock_base={lock_str} "
                f"envs={req.env_ids} to instance {instance.instance_id}"
            )
        return fut

    def get_robot_spheres(self, joint_pos: torch.Tensor):
        """Return robot collision spheres — uses the FREE-mode planner's kinematics."""
        if not torch.is_tensor(joint_pos):
            joint_pos = torch.as_tensor(
                joint_pos, device=self.device, dtype=torch.float32
            )
        else:
            joint_pos = joint_pos.to(self.device, dtype=torch.float32)
        if joint_pos.ndim != 1:
            joint_pos = joint_pos.view(-1)

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

        planner = self.instances[0].planner_free
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
