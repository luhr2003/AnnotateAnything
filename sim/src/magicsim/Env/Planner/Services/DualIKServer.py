"""cuRobo 2.0 DualIKServer — locked-base ⊕ free-base IK for mobile robots.

v2 port of the v1 ``DualIKServer`` (see
``CUROBO_V2_01_CURRENT_INTERFACES.md §1.2``). The "Dual" name refers to
the *solver pair* — one :class:`InverseKinematics` built from the
locked-base YAML (targets transformed to robot base frame), one built
from the free-base YAML (virtual ``base_x/y/h/z`` joints; targets stay
in world frame). ``lock_base: bool`` on each :class:`DualIKPlanRequest`
picks which solver runs.

Arm count is orthogonal to the Dual* distinction — multi-tool-frame
handling (``tool_frames``, ``tracked_tool_frames``, ``info_links``,
``ToolPoseCriteria``) is identical to :class:`IKServer`.

Same architecture as ``IKServer``: pool of ``_DualIKInstance`` worker
threads, microbatch window, chunk loop with per-slot scene loads. Group
key adds ``lock_base`` → ``(G, lock_base)``.

See ``ServiceMigrate.md`` §3 for the info_links semantics and the
``ToolPoseCriteria.disabled()`` plumbing.
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

# cuRobo v2
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.solver.solver_core_cfg import enable_paired_tool_pose
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

from magicsim.Env.Planner.Services.nan_preprocessing import (
    detect_nan_and_pad_goalset,
    log_joint_state_submit,
    log_scene_slot_load,
)

# Re-use the same seed_config builder + noise std as the single-server
# IK path so both backends apply identical anchoring semantics.
from magicsim.Env.Planner.Services.IKServer import _build_seed_config


@dataclass
class DualIKPlanRequest:
    """One dual IK solving request.

    ``lock_base=True``  → locked solver; targets transformed from world to
                          robot base frame.
    ``lock_base=False`` → free solver; targets stay in world frame; the
                          virtual ``base_x/y/h/z`` joints are seeded from
                          ``robot_states.base_pos / base_quat``.

    ``target_pos`` shape contract — same as
    :class:`IKServer.IKPlanRequest`. ``L`` is the picked solver's tracked
    frame count (``tracked_locked`` or ``tracked_free``):

      single  : 2-D ``(N, L * 7)``      OR  3-D ``(N, L, 7)``
      goalset : 3-D ``(N, G, L * 7)``   OR  4-D ``(N, G, L, 7)``

    NaN-as-disable per (env, tool) — see :class:`IKPlanRequest` docstring.
    """

    env_ids: List[int]
    target_pos: torch.Tensor
    robot_states: Dict[str, torch.Tensor]
    mode: str  # "single" or "goalset" — shape dispatch only (see IKPlanRequest)
    lock_base: bool = True


class _DualIKInstance:
    """Dual IK solver instance — owns a locked-base and a free-base :class:`InverseKinematics`."""

    def __init__(
        self,
        instance_id: int,
        robot_cfg_locked: dict,
        robot_cfg_free: dict,
        batch_size: int,
        robot_manager: RobotManager,
        robot_name: str,
        device: torch.device,
        planner_device: torch.device,
        world_cfg_list_locked: List[dict],
        world_cfg_list_free: List[dict],
        world_lock: threading.Lock,
        num_seeds_locked: int,
        num_seeds_free: int,
        position_threshold: float,
        rotation_threshold: float,
        max_goalset: int,
        microbatch_wait_s: float,
        base_joint_names: List[str],
        robot_add_joints: dict,
        robot_ignore_joints: Optional[dict],
        robot_lock_joints: Optional[list],
        robot_dof_name_active: Optional[List[str]],
        extra_fk_link: Optional[List[str]],
        info_links: Optional[List[str]],
        track_xyz_weight: Tuple[float, float, float],
        track_rpy_weight: Tuple[float, float, float],
        left_arm_joints: Optional[List[str]] = None,
        right_arm_joints: Optional[List[str]] = None,
        pin_inactive_arm: bool = False,
        debug: bool = False,
        paired: bool = True,
    ):
        self.instance_id = instance_id
        self.batch_size = batch_size
        # Paired-goalset semantics — applied to BOTH the locked and free
        # solvers so they share the same kernel variant. With L=1 (single
        # tracked frame) paired silently reduces to unpaired argmin.
        self._paired = bool(paired)
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.device = device
        self.planner_device = torch.device(planner_device)
        self.world_cfg_list_locked = world_cfg_list_locked
        self.world_cfg_list_free = world_cfg_list_free
        self._world_lock = world_lock
        self.num_seeds_locked = num_seeds_locked
        self.num_seeds_free = num_seeds_free
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_goalset = int(max(1, max_goalset))
        self.microbatch_wait_s = microbatch_wait_s
        self.base_joint_names = list(base_joint_names or [])
        self.robot_add_joints = dict(robot_add_joints or {})
        self._debug = debug
        self._track_xyz_weight = list(track_xyz_weight)
        self._track_rpy_weight = list(track_rpy_weight)

        # ``pin_inactive_arm`` mode: when caller marks one tool slot
        # disabled (full NaN target) the server overrides the FK seed
        # for that arm's joints with the LIVE sim joint pos AND keeps
        # the criteria as track (instead of switching to disabled).
        # The optimizer is then told to drive that tool to the seed-FK
        # pose, effectively pinning the inactive arm at sim's current
        # joint values for the duration of the solve. Only effective
        # if the YAML supplies ``left_arm_joints`` / ``right_arm_joints``
        # so we know which curobo cspace columns to override.
        self._pin_inactive_arm = bool(pin_inactive_arm)
        self._left_arm_joints = list(left_arm_joints or [])
        self._right_arm_joints = list(right_arm_joints or [])

        # Derived joint sets: LOCKED = FREE \ base_joint_names.
        self.robot_ignore_joints = robot_ignore_joints or {}
        self.robot_lock_joints = robot_lock_joints
        self.robot_dof_name_active = robot_dof_name_active or []
        self.robot_dof_name_active_locked = [
            name
            for name in (robot_dof_name_active or [])
            if name not in self.base_joint_names
        ]
        self.robot_lock_joints_locked = list(robot_lock_joints or []) + list(
            self.base_joint_names
        )

        # Build both solvers.
        device_cfg = DeviceCfg(device=self.planner_device, dtype=torch.float32)

        # v2 quirk: see IKServer for the template + override pattern that
        # forces per-env slot allocation on the scene-collision cfg.
        print(f"[DualIKInstance {instance_id}] Creating LOCKED base IK solver...")
        locked_cfg = InverseKinematicsCfg.create(
            robot=robot_cfg_locked,
            device_cfg=device_cfg,
            scene_model={},
            num_seeds=num_seeds_locked,
            position_tolerance=position_threshold,
            orientation_tolerance=rotation_threshold,
            self_collision_check=True,
            use_cuda_graph=False,
            collision_cache={"cuboid": 10, "mesh": 500},
            max_batch_size=batch_size,
            multi_env=True,
            max_goalset=self.max_goalset,
        )
        locked_cfg.core_cfg.scene_collision_cfg.scene_model = [
            Scene() for _ in range(batch_size)
        ]
        locked_cfg.core_cfg.scene_collision_cfg.num_envs = batch_size
        if self._paired:
            enable_paired_tool_pose(locked_cfg.core_cfg)
        self.ik_solver_locked = InverseKinematics(locked_cfg)

        print(f"[DualIKInstance {instance_id}] Creating FREE base IK solver...")
        free_cfg = InverseKinematicsCfg.create(
            robot=robot_cfg_free,
            device_cfg=device_cfg,
            scene_model={},
            num_seeds=num_seeds_free,
            position_tolerance=position_threshold,
            orientation_tolerance=rotation_threshold,
            self_collision_check=True,
            use_cuda_graph=False,
            collision_cache={"cuboid": 10, "mesh": 500},
            max_batch_size=batch_size,
            multi_env=True,
            max_goalset=self.max_goalset,
        )
        free_cfg.core_cfg.scene_collision_cfg.scene_model = [
            Scene() for _ in range(batch_size)
        ]
        free_cfg.core_cfg.scene_collision_cfg.num_envs = batch_size
        if self._paired:
            enable_paired_tool_pose(free_cfg.core_cfg)
        self.ik_solver_free = InverseKinematics(free_cfg)

        # Resolve + validate extra_fk_link / info_links per solver (tool_frames
        # differ between locked and free YAMLs so we validate per-solver).
        self._extra_fk_link = list(extra_fk_link or [])
        self._tracked_locked = self._resolve_and_apply_criteria(
            self.ik_solver_locked,
            self._extra_fk_link,
        )
        self._tracked_free = self._resolve_and_apply_criteria(
            self.ik_solver_free,
            self._extra_fk_link,
        )
        self._info_links_locked = self._resolve_info_links(
            self.ik_solver_locked,
            info_links,
        )
        self._info_links_free = self._resolve_info_links(
            self.ik_solver_free,
            info_links,
        )

        # Joint-name mapping for each solver.
        self._curobo_joint_names_locked = list(self.ik_solver_locked.joint_names)
        self._curobo_joint_names_free = list(self.ik_solver_free.joint_names)

        # Cache criteria templates for the per-env runtime disable path
        # (NaN preprocessing in ``_solve_one_batch``). Shared across both
        # solvers — the same ToolPoseCriteria object can be passed to either.
        self._track_criteria_template = ToolPoseCriteria.track_position_and_orientation(
            xyz=self._track_xyz_weight,
            rpy=self._track_rpy_weight,
        )
        self._disabled_criteria_template = ToolPoseCriteria.disabled()

        # Per-slot disable-mask cache, ONE per solver — see MotionGenServer
        # for the rationale. Init = all-False (matches broadcast init).
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

        # Pad-slot hygiene — per-solver.
        self._last_B_locked: Optional[int] = None
        self._last_B_free: Optional[int] = None

        # Worker thread / queue.
        self._queue: List[Tuple[concurrent.futures.Future, DualIKPlanRequest]] = []
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._shutdown = False

        # Warmup MUST run on the worker thread to initialize its per-thread
        # CUDA context up front. If the worker's first CUDA op happens
        # later (during a real solve), it shuffles GPU memory and
        # invalidates any CUDA graph already captured by the main thread
        # (e.g. the IsaacLab ``curobo_ik_actions`` graph). __init__ blocks
        # on _warmup_done_event until both locked + free are warmed.
        self._warmup_done_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._warmup_done_event.wait()

    def _warmup_solver(
        self,
        solver: InverseKinematics,
        joint_names: List[str],
        tag: str,
    ) -> None:
        """Prime ``solver`` with one retract-pose solve at max B and G."""
        try:
            default_js_1 = solver.default_joint_state.clone().unsqueeze(0)
            kin = solver.compute_kinematics(default_js_1)
            tool_frames = list(solver.tool_frames)
            retract_poses: Dict[str, Pose] = {}
            for frame in tool_frames:
                retract_poses[frame] = kin.tool_poses.get_link_pose(
                    frame,
                    make_contiguous=True,
                )
            dof = len(joint_names)

            def _run(num_goalset: int) -> None:
                B = int(self.batch_size)
                pose_dict: Dict[str, Pose] = {}
                for frame, p in retract_poses.items():
                    pos = p.position.view(1, 3).expand(B, 3).contiguous()
                    quat = p.quaternion.view(1, 4).expand(B, 4).contiguous()
                    if num_goalset > 1:
                        pos = (
                            pos.unsqueeze(1)
                            .expand(B, num_goalset, 3)
                            .reshape(B * num_goalset, 3)
                            .contiguous()
                        )
                        quat = (
                            quat.unsqueeze(1)
                            .expand(B, num_goalset, 4)
                            .reshape(B * num_goalset, 4)
                            .contiguous()
                        )
                    pose_dict[frame] = Pose(position=pos, quaternion=quat)
                goal = GoalToolPose.from_poses(
                    pose_dict,
                    ordered_tool_frames=tool_frames,
                    num_goalset=num_goalset,
                )
                state = JointState(
                    position=default_js_1.position.expand(B, -1)
                    .contiguous()
                    .to(self.planner_device),
                    velocity=torch.zeros(B, dof, device=self.planner_device),
                    acceleration=torch.zeros(B, dof, device=self.planner_device),
                    jerk=None,
                    joint_names=joint_names,
                )
                solver.solve_pose(goal, current_state=state)

            _run(num_goalset=1)
            if self.max_goalset > 1:
                _run(num_goalset=self.max_goalset)
            if self._debug:
                print(
                    f"[DualIKServer][debug] Instance {self.instance_id} warmup "
                    f"{tag} done (B={self.batch_size}, G={{1, {self.max_goalset}}})."
                )
        except Exception as e:
            print(
                f"[DualIKServer][ERROR] Instance {self.instance_id} warmup "
                f"{tag} failed: {e!r}"
            )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _resolve_and_apply_criteria(
        self,
        solver: InverseKinematics,
        extra_fk_link: List[str],
    ) -> List[str]:
        """Apply per-frame ToolPoseCriteria and return the tracked list.

        After PlannerManager's merge, ``solver.tool_frames`` = YAML tracked
        tool_frames ++ extra_fk_link (dedup). ``tracked`` = complement of
        extra_fk_link. Any ``extra_fk_link`` entry not present in this
        solver's ``tool_frames`` is silently skipped (locked and free YAMLs
        may declare different tool_frame sets; a frame only relevant to one
        is fine as long as it's been merged into THAT YAML).
        """
        yaml_tool_frames = list(solver.tool_frames)
        fk_set = set(extra_fk_link) & set(yaml_tool_frames)
        tracked: List[str] = [f for f in yaml_tool_frames if f not in fk_set]

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
        solver.update_tool_pose_criteria(criteria)
        return tracked

    def _resolve_info_links(
        self, solver: InverseKinematics, info_links: Optional[List[str]]
    ) -> List[str]:
        yaml_tool_frames = list(solver.tool_frames)
        if info_links is None or len(info_links) == 0:
            return list(yaml_tool_frames)
        missing = set(info_links) - set(yaml_tool_frames)
        if missing:
            # Per-solver validation: info_link must exist in THIS solver's
            # tool_frames (after merge). Silent-drop would hide YAML errors.
            return [f for f in info_links if f in set(yaml_tool_frames)]
        return list(info_links)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        with self._queue_cv:
            self._shutdown = True
            self._queue_cv.notify_all()
        if join:
            self._worker.join(timeout=timeout_s)

    def submit(self, fut: concurrent.futures.Future, req: DualIKPlanRequest):
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
    # Frame + joint state helpers
    # ------------------------------------------------------------------

    def _world_to_robot_frame(
        self,
        target_pos: torch.Tensor,
        robot_base_pose: torch.Tensor,
        robot_base_quat: torch.Tensor,
        env_ids: List[int],
    ) -> torch.Tensor:
        device = robot_base_pose.device
        target_tensor = target_pos.to(device=device, dtype=torch.float32)

        original_ndim = target_tensor.ndim
        if original_ndim == 2:
            n = target_tensor.shape[0]
            target_tensor = target_tensor.unsqueeze(1)
            collapse = True
        elif original_ndim == 3:
            n, g = target_tensor.shape[0], target_tensor.shape[1]
            collapse = False
        else:
            raise ValueError(
                f"Expected target_pos [N,7] or [N,G,7], got {target_tensor.shape}"
            )

        flat = target_tensor.reshape(-1, 7)
        target_positions = flat[:, :3]
        target_quats = flat[:, 3:]

        env_ids_tensor = torch.tensor(env_ids, device=device, dtype=torch.long)
        base_positions = robot_base_pose[env_ids_tensor]
        base_quats = robot_base_quat[env_ids_tensor]

        if not collapse:
            base_positions = base_positions.repeat_interleave(g, dim=0)
            base_quats = base_quats.repeat_interleave(g, dim=0)

        pos_relative = target_positions - base_positions
        base_rot_matrices = quat_to_rot_matrix(base_quats)
        base_rot_inv = base_rot_matrices.transpose(-2, -1)
        pos_relative_rotated = torch.bmm(
            base_rot_inv, pos_relative.unsqueeze(-1)
        ).squeeze(-1)

        base_quats_inv = quat_inv(base_quats)
        quats_relative = quat_mul(base_quats_inv, target_quats)

        out = torch.cat([pos_relative_rotated, quats_relative], dim=1)
        if collapse:
            out = out.reshape(n, 7)
        else:
            out = out.reshape(n, g, 7)
        return out.to(device=device, dtype=torch.float32)

    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _arm_joints_for_tool(self, frame: str) -> List[str]:
        """Pick which arm-joint group corresponds to a tool frame.

        Heuristic: lower-cased frame name containing ``right`` or
        starting with ``r_`` → ``right_arm_joints``; ``left`` or
        ``l_`` → ``left_arm_joints``. Returns ``[]`` if neither side
        was declared in the YAML (caller falls back to default
        behaviour).
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
        ik: InverseKinematics,
        tracked: List[str],
        fully_nan: torch.Tensor,
        batch_env_ids: List[int],
    ) -> Tuple[JointState, torch.Tensor]:
        """Build a per-env JointState whose disabled-arm joints are
        overridden with the LIVE sim joint pos. Used so the
        FK-substituted target for the disabled tool slot reflects the
        arm's actual current pose, not the (possibly stale) snapshot
        in ``current_state``.

        Returns:
            (fk_state, effective_disable). ``fk_state.position`` has
            been overridden in place; ``effective_disable`` is the
            criteria-disable mask: when ``pin_inactive_arm=True`` we
            keep tracking on the disabled slot, so this is all-False
            even though ``fully_nan`` flagged some slots disabled.
        """
        device = current_state.position.device
        # Pull live sim joint pos for these env ids.
        sim_robot = self.robot_manager.robots[self.robot_name]
        sim_joint_pos = sim_robot.data.joint_pos
        sim_joint_names = list(sim_robot.joint_names)
        sim_jname_to_idx = {n: i for i, n in enumerate(sim_joint_names)}
        env_ids_t = torch.tensor(batch_env_ids, device=device, dtype=torch.long)
        live_jp = sim_joint_pos.to(device).index_select(0, env_ids_t)

        curobo_jnames = list(ik.joint_names)
        curobo_jname_to_idx = {n: i for i, n in enumerate(curobo_jnames)}

        new_pos = current_state.position.clone()
        for li, frame in enumerate(tracked):
            slot_disabled = fully_nan[:, li]
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
        # All slots stay tracked when pinning is active.
        effective_disable = torch.zeros_like(fully_nan)
        return fk_state, effective_disable

    def _build_joint_state(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        lock_base: bool,
    ) -> torch.Tensor:
        """Build joint positions in cuRobo joint order for the chosen solver.

        LOCKED mode: uses locked YAML joint names (no virtual base joints).
        FREE mode: uses free YAML joint names (includes virtual base joints);
            caller must provide ``joint_pos`` with base columns
            ``[base_x, base_y, base_h=yaw, base_z]`` appended at indices
            ``-4..-1`` before this call.
        """
        if lock_base:
            ik = self.ik_solver_locked
            active_list = self.robot_dof_name_active_locked
            lock_list = self.robot_lock_joints_locked
            add_joints: Dict[str, int] = {}
        else:
            ik = self.ik_solver_free
            active_list = self.robot_dof_name_active
            lock_list = self.robot_lock_joints
            add_joints = self.robot_add_joints

        joint_names = ik.joint_names
        batch = base_pos.shape[0]

        js_positions: List[torch.Tensor] = []
        for joint_name in joint_names:
            if joint_name in self.robot_ignore_joints:
                locked_val = self.robot_ignore_joints[joint_name]
                js_positions.append(
                    torch.full(
                        (batch,),
                        float(locked_val),
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
                # Real sim joint name → pull the live value from joint_pos.
                # Mirrors :meth:`IKServer._build_full_joint_pos`. The prior
                # "v1 parity" zero fallback meant LOCKED mode (which forces
                # ``add_joints={}`` above) and any FREE-mode joint not
                # explicitly registered in YAML ``add_joints`` got seeded
                # at zero — slow / wrong convergence.
                joint_ids, _ = self.robot_manager.robots[self.robot_name].find_joints(
                    joint_name
                )
                if len(joint_ids) > 0 and int(joint_ids[0]) < joint_pos.shape[1]:
                    js_positions.append(joint_pos[:, int(joint_ids[0])])
                else:
                    js_positions.append(
                        torch.zeros(batch, device=base_pos.device, dtype=base_pos.dtype)
                    )
        return torch.stack(js_positions, dim=1)

    def _preprocess_request(
        self,
        req: DualIKPlanRequest,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor, bool]:
        """Canonicalize ``target_pos`` to ``(N, G, L, 7)`` based on
        ``req.mode``, world→robot if locked, build seed JointState.

        Returns: ``env_ids, target_pos_canonical, joint_pos_solver, lock_base``.
        Mode is consumed here; downstream is mode-agnostic.
        """
        env_ids = [int(x) for x in req.env_ids]
        target_pos = req.target_pos
        mode = req.mode
        lock_base = req.lock_base
        tracked = self._tracked_locked if lock_base else self._tracked_free
        L = len(tracked)
        N = target_pos.shape[0]

        # ---- mode-driven shape canonicalization → (N, G, L, 7) -----------
        if mode == "single":
            if target_pos.ndim == 2:
                if target_pos.shape[1] != L * 7:
                    raise ValueError(
                        f"single 2-D target_pos last dim must be L*7={L * 7}; "
                        f"got {tuple(target_pos.shape)}"
                    )
                target_pos = target_pos.view(N, 1, L, 7)
            elif target_pos.ndim == 3:
                if target_pos.shape[1] != L or target_pos.shape[2] != 7:
                    raise ValueError(
                        f"single 3-D target_pos must be (N, L={L}, 7); got "
                        f"{tuple(target_pos.shape)}"
                    )
                target_pos = target_pos.unsqueeze(1)
            else:
                raise ValueError(
                    f"single target_pos must be 2-D (N, L*7) or 3-D (N, L, 7); "
                    f"got {tuple(target_pos.shape)}"
                )
        elif mode == "goalset":
            if target_pos.ndim == 3:
                if target_pos.shape[2] != L * 7:
                    raise ValueError(
                        f"goalset 3-D target_pos last dim must be L*7={L * 7}; "
                        f"got {tuple(target_pos.shape)}"
                    )
                target_pos = target_pos.view(N, target_pos.shape[1], L, 7)
            elif target_pos.ndim == 4:
                if target_pos.shape[2] != L or target_pos.shape[3] != 7:
                    raise ValueError(
                        f"goalset 4-D target_pos must be (N, G, L={L}, 7); got "
                        f"{tuple(target_pos.shape)}"
                    )
            else:
                raise ValueError(
                    f"goalset target_pos must be 3-D (N, G, L*7) or 4-D "
                    f"(N, G, L, 7); got {tuple(target_pos.shape)}"
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        base_pos = req.robot_states["base_pos"]
        base_quat = req.robot_states["base_quat"]
        joint_pos = req.robot_states["joint_pos"]

        env_ids_tensor = torch.tensor(env_ids, device=base_pos.device, dtype=torch.long)
        base_pos_envs = base_pos[env_ids_tensor]
        base_quat_envs = base_quat[env_ids_tensor]
        joint_pos_envs = joint_pos[env_ids_tensor]

        if lock_base:
            shape_in = target_pos.shape
            K = shape_in[1] * shape_in[2]
            flat3 = target_pos.reshape(N, K, 7)
            flat3 = self._world_to_robot_frame(
                flat3,
                base_pos,
                base_quat,
                env_ids,
            )
            target_pos = flat3.reshape(shape_in)
            joint_pos_for_build = joint_pos_envs
        else:
            yaw = self._quat_to_yaw(base_quat_envs).unsqueeze(1)
            joint_pos_for_build = torch.cat(
                [
                    joint_pos_envs,
                    base_pos_envs[:, 0:1],
                    base_pos_envs[:, 1:2],
                    yaw.to(dtype=joint_pos_envs.dtype),
                    base_pos_envs[:, 2:3],
                ],
                dim=1,
            )

        joint_pos_solver = self._build_joint_state(
            base_pos_envs, base_quat_envs, joint_pos_for_build, lock_base
        )
        return env_ids, target_pos, joint_pos_solver, lock_base

    # ------------------------------------------------------------------
    # Solver call
    # ------------------------------------------------------------------

    def _solve_one_batch(
        self,
        batch_env_ids: List[int],
        batch_target_pos: torch.Tensor,
        lock_base: bool,
        batch_joint_pos: torch.Tensor,
    ) -> Tuple[List[bool], List[int]]:
        ik = self.ik_solver_locked if lock_base else self.ik_solver_free
        tracked = self._tracked_locked if lock_base else self._tracked_free
        curobo_jnames = (
            self._curobo_joint_names_locked
            if lock_base
            else self._curobo_joint_names_free
        )
        world_cfg_list = (
            self.world_cfg_list_locked if lock_base else self.world_cfg_list_free
        )

        B_actual = len(batch_env_ids)
        assert B_actual == batch_target_pos.shape[0]

        # Per-slot scene reload + pad hygiene.
        last_B = self._last_B_locked if lock_base else self._last_B_free
        with self._world_lock:
            for slot, env_id in enumerate(batch_env_ids):
                env_id_int = int(env_id)
                if env_id_int < 0 or env_id_int >= len(world_cfg_list):
                    raise ValueError(
                        f"env_id {env_id_int} out of range [0, {len(world_cfg_list)}) "
                        f"at slot {slot}"
                    )
                scene_cfg = world_cfg_list[env_id_int]
                if isinstance(scene_cfg, Scene):
                    scene = scene_cfg
                elif isinstance(scene_cfg, dict) and scene_cfg:
                    scene = Scene.create(scene_cfg)
                else:
                    scene = Scene()
                ik.scene_collision_checker.load_collision_model(
                    scene,
                    env_idx=int(slot),
                )
                if self._debug:
                    log_scene_slot_load(
                        scene,
                        int(slot),
                        env_id_int,
                        tag=f"DualIK-{'locked' if lock_base else 'free'}",
                    )
            if last_B is None or last_B > B_actual or last_B == self.batch_size:
                for pad in range(B_actual, self.batch_size):
                    ik.scene_collision_checker.load_collision_model(
                        Scene(),
                        env_idx=pad,
                    )
            if lock_base:
                self._last_B_locked = B_actual
            else:
                self._last_B_free = B_actual

        # Build current_state. Leave ``jerk=None`` — see IKServer for the
        # reason (upstream ``_pad_batch_inputs`` doesn't pad jerk, which
        # breaks dynamic-B solves when ``from_position`` sets it to zeros).
        current_state_pos = batch_joint_pos.to(self.planner_device).contiguous()
        current_state = JointState(
            position=current_state_pos,
            velocity=torch.zeros_like(current_state_pos),
            acceleration=torch.zeros_like(current_state_pos),
            jerk=None,
            joint_names=curobo_jnames,
        )

        # ----- Canonical-tensor pipeline (mirrors IKServer) -----------------
        # Input arrives canonicalized by ``_preprocess_request``:
        #   - single:  (B, L, 7)         → unsqueeze G=1  →  (B, L, 1, 7)
        #   - goalset: (B, G, L, 7)      → permute G↔L     →  (B, L, G, 7)
        # Internal helper ``detect_nan_and_pad_goalset`` operates on
        # (B, L, G, 7).
        pose_dict: Dict[str, Pose] = {}
        num_tracked = len(tracked)

        # Compute current FK ONCE — reused for NaN substitution filler AND
        # info-only (extra_fk_link) frame filler.
        fk_kin = ik.compute_kinematics(current_state)
        fk_pose_per_frame: Dict[str, Pose] = {}
        for frame in ik.tool_frames:
            fk_pose_per_frame[frame] = fk_kin.tool_poses.get_link_pose(
                frame,
                make_contiguous=True,
            )
        fk_filler = torch.zeros(
            (B_actual, num_tracked, 7),
            device=self.planner_device,
            dtype=batch_target_pos.dtype,
        )
        for li, f in enumerate(tracked):
            p = fk_pose_per_frame[f]
            fk_filler[:, li, :3] = p.position.view(-1, 3)
            fk_filler[:, li, 3:] = p.quaternion.view(-1, 4)

        # Caller-canonical (B, G, L, 7) → helper-canonical (B, L, G, 7).
        target_dev = batch_target_pos.to(self.planner_device)
        assert target_dev.ndim == 4 and target_dev.shape[2] == num_tracked, (
            f"_solve_one_batch expects canonical (B, G, L={num_tracked}, 7); "
            f"got {tuple(target_dev.shape)}"
        )
        num_goalset = target_dev.shape[1]
        canonical = target_dev.permute(0, 2, 1, 3).contiguous()  # (B, L, G, 7)

        padded, fully_nan, _real_count = detect_nan_and_pad_goalset(
            canonical,
            tracked,
            paired=self._paired,
        )
        if bool(fully_nan.any().item()):
            fk_g = fk_filler.unsqueeze(2).expand(B_actual, num_tracked, num_goalset, 7)
            full_mask = fully_nan.unsqueeze(-1).unsqueeze(-1).expand_as(padded)
            padded = torch.where(full_mask, fk_g, padded)

        # Per-env runtime criteria — write every (slot, tracked_frame)
        # per solve. NOTE: ``env_idx`` is the SOLVER SLOT INDEX. Routes to
        # whichever solver this request hit (locked OR free); the other
        # solver's per-env criteria rows stay as last set. Per-solver
        # cache makes the writes a no-op when the disable pattern matches
        # what's already on the GPU buffer (common case).
        persisted = (
            self._persisted_disable_locked
            if lock_base
            else self._persisted_disable_free
        )
        cached_dev = persisted[:B_actual]
        if cached_dev.device != fully_nan.device:
            cached_dev = cached_dev.to(fully_nan.device)
        slot_changed = (cached_dev != fully_nan).any(dim=-1)  # (B_actual,) bool
        if bool(slot_changed.any().item()):
            changed_idx = slot_changed.nonzero(as_tuple=True)[0].tolist()
            for b in changed_idx:
                crit: Dict[str, ToolPoseCriteria] = {}
                for li, frame in enumerate(tracked):
                    crit[frame] = (
                        self._disabled_criteria_template
                        if bool(fully_nan[b, li])
                        else self._track_criteria_template
                    )
                ik.update_tool_pose_criteria_per_env(b, crit)
            persisted[:B_actual] = fully_nan.to(persisted.device)

        for li, frame in enumerate(tracked):
            pose_dict[frame] = Pose(
                position=padded[:, li, :, :3]
                .reshape(B_actual * num_goalset, 3)
                .contiguous(),
                quaternion=padded[:, li, :, 3:]
                .reshape(B_actual * num_goalset, 4)
                .contiguous(),
            )

        # Non-tracked / info-only frames — reuse cached FK (no extra call).
        needs_fk = [f for f in ik.tool_frames if f not in pose_dict]
        for frame in needs_fk:
            p = fk_pose_per_frame[frame]
            if num_goalset > 1:
                pos_g = (
                    p.position.unsqueeze(1)
                    .expand(B_actual, num_goalset, 3)
                    .reshape(B_actual * num_goalset, 3)
                )
                quat_g = (
                    p.quaternion.unsqueeze(1)
                    .expand(B_actual, num_goalset, 4)
                    .reshape(B_actual * num_goalset, 4)
                )
                pose_dict[frame] = Pose(
                    position=pos_g.contiguous(),
                    quaternion=quat_g.contiguous(),
                )
            else:
                pose_dict[frame] = p

        goal = GoalToolPose.from_poses(
            pose_dict,
            ordered_tool_frames=list(ik.tool_frames),
            num_goalset=num_goalset,
        )

        if self._debug:
            log_joint_state_submit(
                current_state,
                batch_env_ids,
                tag=f"DualIK-{'locked' if lock_base else 'free'}",
            )
        # Same seed-anchoring trick as :class:`IKServer`. Default cuRobo
        # path only seeds 1 of N from current_state and fills the other
        # N-1 with random configs — for goalset solves the candidate
        # picked is whichever the random seeds happened to converge on,
        # not the one closest to the robot's current pose. Replicate
        # current_state across all ``num_seeds_{locked,free}`` and add
        # ~6° per-joint noise on seed[1:] so multiple seeds explore
        # nearby reachable goalset entries while staying anchored.
        n_seeds = self.num_seeds_locked if lock_base else self.num_seeds_free
        seed_config = _build_seed_config(current_state.position, n_seeds)
        _t0 = time.time()
        result = ik.solve_pose(
            goal, current_state=current_state, seed_config=seed_config
        )
        _dt = time.time() - _t0

        success_t = result.success
        if success_t.dim() == 2:
            success_t = success_t[:, 0]
        success = [bool(success_t[i].item()) for i in range(B_actual)]

        # Always populate goalset_index — single is just G=1 (returns 0
        # for every successful env). Failed envs get -1.
        goalset_index: List[int] = [-1] * B_actual
        gi = result.goalset_index
        if gi is not None:
            if gi.dim() == 3:
                gi_flat = gi[:, 0, 0] if gi.shape[2] >= 1 else gi[:, 0]
            elif gi.dim() == 2:
                gi_flat = gi[:, 0]
            else:
                gi_flat = gi
            for i in range(B_actual):
                if i < gi_flat.shape[0] and success[i]:
                    goalset_index[i] = int(gi_flat[i].item())

        mode_label = "locked" if lock_base else "free"
        pos_err = (
            float(result.position_error.max().item())
            if result.position_error is not None
            else float("nan")
        )
        rot_err = (
            float(result.rotation_error.max().item())
            if result.rotation_error is not None
            else float("nan")
        )
        print(
            f"[DualIKServer][debug] Instance {self.instance_id} [{mode_label}] "
            f"Solved batch {batch_env_ids} (G={num_goalset}) success={success} "
            f"goalset_index={goalset_index} pos_err_max={pos_err:.4f}m "
            f"rot_err_max={rot_err:.4f}rad dt={_dt:.4f}s"
        )

        return success, goalset_index

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self):
        # Warm up FIRST on this thread — see __init__ comment.
        try:
            self._warmup_solver(
                self.ik_solver_locked,
                self._curobo_joint_names_locked,
                tag="locked",
            )
            self._warmup_solver(
                self.ik_solver_free,
                self._curobo_joint_names_free,
                tag="free",
            )
        finally:
            self._warmup_done_event.set()

        while True:
            with self._queue_cv:
                while not self._shutdown and len(self._queue) == 0:
                    self._queue_cv.wait()
                if self._shutdown:
                    return

                batch: List[Tuple[concurrent.futures.Future, DualIKPlanRequest]] = []
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
                        if len(batch) == self.batch_size * 2:
                            break
                else:
                    while len(self._queue) > 0:
                        batch.append(self._queue.pop(0))
                        if len(batch) == self.batch_size * 2:
                            break

            preprocessed: List[
                Tuple[
                    concurrent.futures.Future,
                    List[int],
                    torch.Tensor,
                    torch.Tensor,
                    bool,
                ]
            ] = []

            try:
                t_req0 = time.time()
                for fut, req in batch:
                    try:
                        env_ids, tgt, jp, lb = self._preprocess_request(req)
                        preprocessed.append((fut, env_ids, tgt, jp, lb))
                    except Exception as ex:
                        tb = traceback.format_exc()
                        print(
                            f"[DualIKServer][ERROR] Instance {self.instance_id} "
                            f"preprocessing failed for request envs={req.env_ids} "
                            f"lock_base={req.lock_base}: "
                            f"{type(ex).__name__}: {ex}\n{tb}"
                        )
                        fut.set_exception(ex)

                if len(preprocessed) == 0:
                    continue

                # Group by (G, lock_base). Shapes are canonical post
                # ``_preprocess_request``: every request is (N, G, L, 7);
                # single is just G=1. L is fixed per (Server, lock_base)
                # so it doesn't enter the group key.
                groups: Dict[
                    Tuple[int, bool],
                    List[
                        Tuple[
                            concurrent.futures.Future,
                            List[int],
                            torch.Tensor,
                            torch.Tensor,
                        ]
                    ],
                ] = {}
                per_req_env_ids: Dict[concurrent.futures.Future, List[int]] = {}

                for fut, env_ids, tgt, jp, lb in preprocessed:
                    per_req_env_ids[fut] = env_ids
                    key = (int(tgt.shape[1]), lb)
                    groups.setdefault(key, []).append((fut, env_ids, tgt, jp))

                for (G, lb), group_requests in groups.items():
                    merged_env_ids: List[int] = []
                    merged_targets: List[torch.Tensor] = []
                    merged_joint_pos: List[torch.Tensor] = []
                    per_req_indices: Dict[concurrent.futures.Future, List[int]] = {}

                    seen_env = set()
                    for fut, env_ids, tgt, jp in group_requests:
                        idx_list: List[int] = []
                        for j, env_id in enumerate(env_ids):
                            if env_id in seen_env:
                                existing = merged_env_ids.index(env_id)
                                idx_list.append(existing)
                                continue
                            seen_env.add(env_id)
                            merged_env_ids.append(env_id)
                            merged_joint_pos.append(jp[j])
                            merged_targets.append(tgt[j])
                            idx_list.append(len(merged_env_ids) - 1)
                        per_req_indices[fut] = idx_list

                    if len(merged_env_ids) == 0:
                        continue

                    merged_target_tensor = torch.stack(merged_targets, dim=0)
                    merged_joint_pos_tensor = torch.stack(merged_joint_pos, dim=0)

                    merged_success: List[bool] = [False] * len(merged_env_ids)
                    merged_goalset_index: List[int] = [-1] * len(merged_env_ids)

                    for s in range(0, len(merged_env_ids), self.batch_size):
                        e = min(s + self.batch_size, len(merged_env_ids))
                        bids = merged_env_ids[s:e]
                        success_chunk, gi_chunk = self._solve_one_batch(
                            bids,
                            merged_target_tensor[s:e],
                            lb,
                            batch_joint_pos=merged_joint_pos_tensor[s:e],
                        )
                        merged_success[s:e] = success_chunk
                        merged_goalset_index[s:e] = gi_chunk

                    for fut, orig_env_ids, _, _ in group_requests:
                        idx_list = per_req_indices.get(fut)
                        stored_env_ids = per_req_env_ids.get(fut)
                        if idx_list is None or stored_env_ids is None:
                            raise ValueError(f"Missing mapping for future {id(fut)}")
                        success_out = [merged_success[ii] for ii in idx_list]
                        goalset_out = [merged_goalset_index[ii] for ii in idx_list]
                        fut.set_result((success_out, goalset_out, stored_env_ids))

                if self._debug:
                    print(
                        f"[DualIKServer][debug] Instance {self.instance_id} served "
                        f"{len(preprocessed)} request(s) dt_total="
                        f"{time.time() - t_req0:.4f}s"
                    )
            except Exception as ex:
                print(
                    f"[DualIKServer][ERROR] Instance {self.instance_id} "
                    f"worker exception — {type(ex).__name__}: {ex}\n"
                    f"{traceback.format_exc()}"
                )
                for fut, *_ in preprocessed:
                    if not fut.done():
                        fut.set_exception(ex)
                for fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(ex)


class DualIKServer:
    """cuRobo 2.0 dual IK service — locked-base ⊕ free-base pair.

    - Two :class:`InverseKinematics` instances per worker
      (locked / free), selected per-request by ``lock_base``.
    - Separate world-cfg caches (locked / free), because obstacle
      transforms differ between robot-frame and world-frame.
    - ``dual_mode = True`` — PlannerManager duck-types this.
    """

    dual_mode: bool = True

    def __init__(
        self,
        robot_manager: RobotManager,
        robot_cfg_locked: dict,
        robot_cfg_free: dict,
        robot_name: str,
        device: torch.device,
        batch_size: int = 2,
        microbatch_wait_ms: float = 200.0,
        num_instances: int = 2,
        num_seeds_locked: int = 20,
        num_seeds_free: int = 50,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        max_goalset: int = 10000,
        base_joint_names: Optional[List[str]] = None,
        robot_add_joints: Optional[dict] = None,
        robot_ignore_joints: Optional[dict] = None,
        robot_lock_joints: Optional[list] = None,
        robot_dof_name_active: Optional[List[str]] = None,
        extra_fk_link: Optional[List[str]] = None,
        info_links: Optional[List[str]] = None,
        track_xyz_weight: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        track_rpy_weight: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        left_arm_joints: Optional[List[str]] = None,
        right_arm_joints: Optional[List[str]] = None,
        pin_inactive_arm: bool = False,
        debug: bool = False,
        planner_devices: Optional[Union[str, torch.device, Sequence]] = None,
        paired: bool = True,
    ):
        self.robot_manager = robot_manager
        self.robot_cfg_locked = robot_cfg_locked
        self.robot_cfg_free = robot_cfg_free
        self.robot_name = robot_name
        self.device = device
        self.batch_size = min(int(batch_size), int(robot_manager.num_envs))
        self.microbatch_wait_s = max(float(microbatch_wait_ms) / 1000.0, 0.0)
        self.num_instances = num_instances
        self.num_seeds_locked = num_seeds_locked
        self.num_seeds_free = num_seeds_free
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_goalset = int(max(1, max_goalset))
        self.base_joint_names = list(base_joint_names or [])
        self.robot_add_joints = dict(robot_add_joints or {})
        self._debug = debug
        self._paired = bool(paired)

        self.planner_devices: List[torch.device] = _normalize_planner_devices(
            planner_devices, self.num_instances
        )

        # Separate world cfg caches for locked (robot frame) and free (world frame) modes.
        self.world_cfg_list_locked: List[dict] = [
            {} for _ in range(robot_manager.num_envs)
        ]
        self.world_cfg_list_free: List[dict] = [
            {} for _ in range(robot_manager.num_envs)
        ]
        self._world_lock = threading.Lock()

        self.instances: List[_DualIKInstance] = []
        for i in range(self.num_instances):
            instance = _DualIKInstance(
                instance_id=i,
                robot_cfg_locked=robot_cfg_locked,
                robot_cfg_free=robot_cfg_free,
                batch_size=self.batch_size,
                robot_manager=robot_manager,
                robot_name=robot_name,
                device=device,
                planner_device=self.planner_devices[i],
                world_cfg_list_locked=self.world_cfg_list_locked,
                world_cfg_list_free=self.world_cfg_list_free,
                world_lock=self._world_lock,
                num_seeds_locked=num_seeds_locked,
                num_seeds_free=num_seeds_free,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                max_goalset=self.max_goalset,
                microbatch_wait_s=self.microbatch_wait_s,
                base_joint_names=self.base_joint_names,
                robot_add_joints=self.robot_add_joints,
                robot_ignore_joints=robot_ignore_joints,
                robot_lock_joints=robot_lock_joints,
                robot_dof_name_active=robot_dof_name_active,
                extra_fk_link=extra_fk_link,
                info_links=info_links,
                track_xyz_weight=track_xyz_weight,
                track_rpy_weight=track_rpy_weight,
                left_arm_joints=left_arm_joints,
                right_arm_joints=right_arm_joints,
                pin_inactive_arm=pin_inactive_arm,
                debug=debug,
                paired=self._paired,
            )
            self.instances.append(instance)

        print(
            f"[DualIKServer] Initialized with {self.num_instances} instance(s), "
            f"batch_size={self.batch_size}, microbatch_wait_ms={microbatch_wait_ms}, "
            f"num_seeds_locked={num_seeds_locked}, num_seeds_free={num_seeds_free}, "
            f"paired={self._paired}, "
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
        """Update one mode's cache.

        ``relative_to_world_frame=True`` → free-base cache (world frame).
        ``relative_to_world_frame=False`` → locked-base cache (robot frame).
        """
        if len(env_ids) != len(world_cfgs):
            raise ValueError("env_ids and world_cfgs length mismatch")
        with self._world_lock:
            target = (
                self.world_cfg_list_free
                if relative_to_world_frame
                else self.world_cfg_list_locked
            )
            for env_id, cfg in zip(env_ids, world_cfgs):
                target[int(env_id)] = cfg or {}

    def update_world_dual(
        self,
        env_ids: List[int],
        world_cfgs_locked: List[dict],
        world_cfgs_free: List[dict],
    ):
        """Update both caches in one call."""
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

    def _select_instance(self) -> _DualIKInstance:
        idle = [inst for inst in self.instances if inst.is_idle()]
        if idle:
            return min(idle, key=lambda inst: inst.queue_size())
        return min(self.instances, key=lambda inst: inst.queue_size())

    def submit_ik(self, req: DualIKPlanRequest) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        instance = self._select_instance()
        instance.submit(fut, req)
        if self._debug:
            print(
                f"[DualIKServer][debug] Submitted request mode={req.mode} "
                f"lock_base={req.lock_base} envs={req.env_ids} "
                f"to instance {instance.instance_id} "
                f"(queue_size={instance.queue_size()})"
            )
        return fut
