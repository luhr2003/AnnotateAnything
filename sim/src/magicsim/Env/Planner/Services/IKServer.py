"""cuRobo 2.0 IKServer — in-process IK service with instance pool and microbatching.

v2 port of the v1 ``IKServer`` (see
``CUROBO_V2_01_CURRENT_INTERFACES.md §1.1``). Architecture preserved
1:1 — pool of ``_IKInstance`` worker threads, microbatch window per
instance, grouping by ``(mode, G_or_eef_count)``, chunk loop with
per-slot scene loads, dedup per env within a microbatch. The solver
internals switch from
``curobo.wrap.reacher.ik_solver.IKSolver.solve_batch_env[_goalset]``
to ``curobo.inverse_kinematics.InverseKinematics.solve_pose(GoalToolPose,
current_state)``.

v1 → v2 API deltas (see ``CUROBO_V2_02_MIGRATION_PLAN.md`` §4):
- ``IKSolverConfig.load_from_robot_config(robot, WorldConfig(), n_collision_envs=N, ...)``
  → ``InverseKinematicsCfg.create(robot=..., scene_model=None, max_batch_size=N, multi_env=True, max_goalset=G, ...)``.
- ``world_coll_checker.load_collision_model(WorldConfig, env_idx=slot)``
  → ``scene_collision_checker.load_collision_model(Scene.create(dict), env_idx=slot)``.
- ``solve_batch_env(pose, retract_config, seed_config, link_poses={...})``
  → ``solve_pose(GoalToolPose.from_poses({tool_frame: Pose, ...}, ordered_tool_frames=ik.tool_frames, num_goalset=G), current_state=JointState.from_position(...))``.
- v1 goalset / extra-link side channel (``link_poses`` kwarg) folds into
  the single ``GoalToolPose`` dict.
- ``world_cfg_list: List[WorldConfig]`` → ``List[dict]`` (SceneCfg dicts);
  Servers convert to ``Scene`` at chunk time.

MagicSim YAML contract (see ``ServiceMigrate.md`` §3):
- YAML ``robot_cfg.kinematics.tool_frames``  = TRACKED frames (drive IK cost).
- YAML top-level ``extra_fk_link``           = FK-only frames (``ToolPoseCriteria.disabled()``);
  PlannerManager merges these into ``kinematics.tool_frames`` before handing
  ``robot_cfg`` to cuRobo and forwards the ``extra_fk_link`` list here.
- YAML top-level ``info_links``              = ordering for FK readout
  (validated ⊆ merged ``ik.tool_frames``; IKServer doesn't produce trajectories
  so info_links is carried for API parity — used when callers read FK poses
  post-solve via ``compute_kinematics``).
- At init: ``ik.update_tool_pose_criteria({...})`` — ``track_...`` for tracked,
  ``ToolPoseCriteria.disabled()`` for every frame in ``extra_fk_link``.
- At solve: caller target_pos covers only tracked frames (``ik.tool_frames \
  extra_fk_link``, in that order); disabled frames get current-FK pose filler
  before ``GoalToolPose.from_poses``.

Dynamic batch size (see ``CUROBO_V2_03_DYNAMIC_BATCH.md`` §2):
- ``InverseKinematics.solve_pose`` auto-pads B to ``max_batch_size`` and
  slices the result back. We still maintain pad-slot hygiene: when
  ``B_actual`` shrinks, reload ``Scene()`` into pad slots
  ``[B_actual..max_batch_size-1]`` so they don't evaluate against stale
  real scenes.

Debug lines preserved verbatim except ``result.status`` (which has no v2
equivalent — see ``ServiceMigrate.md`` §4.3); the replacement reports
``pos_err_max`` / ``rot_err_max``.
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


# Std (radians) for the per-joint Gaussian perturbation applied to seeds
# 1..N-1 in :func:`_build_seed_config`. ~0.10 rad ≈ 5.7°. Large enough to
# break LM symmetry and let different seeds find different reachable
# goalset entries; small enough that all seeds stay close to the live
# joint state (the whole point of the seed_config injection).
_SEED_CONFIG_NOISE_STD = 0.10


def _build_seed_config(current_position: torch.Tensor, num_seeds: int) -> torch.Tensor:
    """Replicate ``current_position`` across ``num_seeds`` with small noise.

    Forces the solver to anchor every IK seed at the live joint state
    rather than letting (num_seeds - 1) random seeds wander. seed[0] is
    kept exact (precise anchor); seed[1..N-1] gets per-joint Gaussian
    noise with std :data:`_SEED_CONFIG_NOISE_STD`.

    Args:
        current_position: ``(B, dof)`` joint positions in the solver's
            cuRobo joint order.
        num_seeds: Solver's IK seed count (``self.num_seeds`` /
            ``self.num_seeds_locked|free``).

    Returns:
        ``(B, num_seeds, dof)`` seed config the cuRobo
        :func:`solve_pose` accepts via its ``seed_config=`` kwarg —
        :class:`SeedManager.prepare_action_seeds` skips random fill when
        ``n_provided_seeds == num_seeds``.
    """
    B, dof = current_position.shape
    seed = current_position.unsqueeze(1).expand(B, num_seeds, dof).contiguous().clone()
    if num_seeds > 1:
        noise = (
            torch.randn(
                B,
                num_seeds - 1,
                dof,
                device=current_position.device,
                dtype=current_position.dtype,
            )
            * _SEED_CONFIG_NOISE_STD
        )
        seed[:, 1:] = seed[:, 1:] + noise
    return seed


@dataclass
class IKPlanRequest:
    """One IK solving request.

    ``target_pos`` tensors are for *all* envs and indexed by ``env_id``.
    ``L`` always equals ``len(tracked_tool_frames)`` and frames appear
    in the Server's ``tracked_tool_frames`` cfg order. Per (env, tool)
    all-NaN xyz across the G axis ⇒ that tool is disabled for that env
    this solve only.

    Accepted ``target_pos`` shapes (``mode`` disambiguates — for ``L=1``
    the 3-D forms `(N, L, 7)` and `(N, G, L*7)` collide on the wire, so
    callers must declare intent):

      mode="single"   → 2-D ``(N, L * 7)`` or 3-D ``(N, L, 7)``    [G=1]
      mode="goalset"  → 3-D ``(N, G, L * 7)`` or 4-D ``(N, G, L, 7)``

    Internally both modes canonicalize to ``(N, G, L, 7)`` and run the
    same kernel — single is just G=1. Downstream grouping and
    ``_solve_one_batch`` are mode-agnostic.

    Per (env, tool) NaN handling:
      - ALL G slots NaN ⇒ disable that tool for that env.
      - Some NaN ⇒ mod-pad NaN slots from this tool's real candidates
        (``g_pad → real[g % real_count]``).
      - ``paired=True`` requires equal active-tool real counts per env;
        ``paired=False`` is unconstrained.

    Position-only ``(..., 3)`` shapes are not accepted.
    """

    env_ids: List[int]
    target_pos: torch.Tensor
    robot_states: Dict[str, torch.Tensor]
    mode: str  # "single" or "goalset" — shape dispatch only (see above)


@dataclass
class IKPlanResult:
    """Result from IK solving.

    ``success[i]`` indicates env_ids[i] has a valid solution.
    ``goalset_index[i]`` is the selected goal index in [0, G) — always
    populated; trivially 0 for ``G=1`` requests; -1 on failed envs.
    """

    success: List[bool]
    goalset_index: List[int]
    env_ids: List[int]


class _IKInstance:
    """Single IK solver instance with its own worker thread and queue."""

    def __init__(
        self,
        instance_id: int,
        robot_cfg: dict,
        batch_size: int,
        robot_manager: RobotManager,
        robot_name: str,
        device: torch.device,
        planner_device: torch.device,
        world_cfg_list: List[dict],
        world_lock: threading.Lock,
        num_seeds: int,
        position_threshold: float,
        rotation_threshold: float,
        max_goalset: int,
        microbatch_wait_s: float,
        extra_fk_link: Optional[List[str]],
        info_links: Optional[List[str]],
        track_xyz_weight: Tuple[float, float, float],
        track_rpy_weight: Tuple[float, float, float],
        debug: bool = False,
        relative_to_world_frame: bool = True,
        paired: bool = True,
    ):
        self.instance_id = instance_id
        self.batch_size = batch_size
        # Paired-goalset semantics: every tool in the goalset must share a
        # single ``g_idx`` (used for bimanual rigid grasps where slot ``g``
        # on the right and slot ``g`` on the left are jointly reachable).
        # Single-frame robots fall through to plain unpaired argmin —
        # ``enable_paired_tool_pose`` is safe with L=1.
        self._paired = bool(paired)
        self.robot_manager = robot_manager
        self.robot_name = robot_name
        self.device = device
        self.planner_device = torch.device(planner_device)
        self.world_cfg_list = world_cfg_list
        self._world_lock = world_lock
        self.num_seeds = num_seeds
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_goalset = int(max(1, max_goalset))
        self.microbatch_wait_s = microbatch_wait_s
        self._debug = debug
        self.relative_to_world_frame = relative_to_world_frame
        self._track_xyz_weight = list(track_xyz_weight)
        self._track_rpy_weight = list(track_rpy_weight)

        # Robot-cfg bookkeeping for joint-order mapping + locked joint padding.
        self.cspace_joint_names = (
            robot_cfg.get("kinematics", {}).get("cspace", {}).get("joint_names", None)
        )
        self.lock_joints = robot_cfg.get("kinematics", {}).get("lock_joints", {}) or {}

        # Build the v2 IK solver. ``scene_model`` is a per-env list so the
        # scene-collision cache is allocated ``(batch_size, N_cuboids)``;
        # per-chunk scenes come in via
        # ``scene_collision_checker.load_collision_model(Scene, env_idx=slot)``.
        #
        # ``use_cuda_graph=False`` is required for our access pattern because:
        #   (1) production calls ``solve_pose`` at BOTH ``num_goalset=1`` (single
        #       mode) and ``num_goalset=max_goalset`` (goalset mode). Once a
        #       graph is captured for one shape cuRobo raises "CUDA graph reset
        #       is not available." on the next call with a different shape.
        #   (2) per-chunk ``load_collision_model(..., env_idx=slot)`` can rebind
        #       references the captured graph had pinned, leading to illegal
        #       memory access on replay.
        # ``ik_batched_env.py`` uses ``True`` safely because it only loads
        # scenes at init and uses a fixed ``num_goalset=1``.
        device_cfg = DeviceCfg(device=self.planner_device, dtype=torch.float32)
        # v2 quirk: ``InverseKinematicsCfg.create(multi_env=True, max_batch_size=N)``
        # only wires ``num_envs=N`` on the kinematics side; the scene-collision cfg
        # built from a singular ``scene_model`` is shaped ``(1, N_cuboids)`` and a
        # runtime ``load_collision_model(env_idx=k>0)`` trips an index error. Pass
        # a template here, then override to a per-env list before constructing
        # the solver so SceneData allocates ``(N, N_cuboids)`` slots. Pattern
        # taken from ``curobo/examples/isaacsim/ik_batched_env.py:276-307``.
        ik_cfg = InverseKinematicsCfg.create(
            robot=robot_cfg,
            device_cfg=device_cfg,
            scene_model={},
            num_seeds=num_seeds,
            position_tolerance=position_threshold,
            orientation_tolerance=rotation_threshold,
            self_collision_check=True,
            use_cuda_graph=False,
            collision_cache={"cuboid": 10, "mesh": 500},
            max_batch_size=batch_size,
            multi_env=True,
            max_goalset=self.max_goalset,
        )
        ik_cfg.core_cfg.scene_collision_cfg.scene_model = [
            Scene() for _ in range(batch_size)
        ]
        ik_cfg.core_cfg.scene_collision_cfg.num_envs = batch_size
        # Paired flag must be applied to the cfg BEFORE the solver consumes
        # it. ``enable_paired_tool_pose`` walks every rollout's
        # tool_pose_cfg and flips ``paired = True``; the solver picks the
        # paired warp kernel at construction. Requires ``per_env=True``,
        # which the factory auto-enabled via ``multi_env=True`` above.
        if self._paired:
            enable_paired_tool_pose(ik_cfg.core_cfg)
        self.ik_solver = InverseKinematics(ik_cfg)

        # Resolve tracked + info-link lists from the merged tool_frames.
        # After PlannerManager's merge, ``ik.tool_frames`` == YAML tracked
        # tool_frames ++ YAML top-level extra_fk_link (dedup, order preserved).
        # ``tracked`` is the complement of extra_fk_link — the set that
        # receives caller target poses.
        yaml_tool_frames = list(self.ik_solver.tool_frames)
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

        if info_links is None:
            # Default to the merged order so readout covers everything.
            self._info_links: List[str] = list(yaml_tool_frames)
        else:
            missing = set(info_links) - set(yaml_tool_frames)
            if missing:
                raise ValueError(
                    f"info_links {info_links} contains frames not in merged "
                    f"tool_frames={yaml_tool_frames}: {missing}. "
                    f"Add them to kinematics.tool_frames or extra_fk_link in the YAML."
                )
            self._info_links = list(info_links)

        # Per-frame ToolPoseCriteria: track for tracked frames, disable for
        # extra_fk_link. Zero-weight frames contribute nothing to cost/gradient
        # at every stage (seed LM + main optimizer) — see ``ServiceMigrate.md`` §3.5.
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
        self.ik_solver.update_tool_pose_criteria(criteria)

        # Cache criteria templates for the per-env runtime disable path
        # (NaN preprocessing in ``_solve_one_batch``). Building these every
        # solve would burn ~B*L cheap allocations; cache once at init.
        self._track_criteria_template = ToolPoseCriteria.track_position_and_orientation(
            xyz=self._track_xyz_weight,
            rpy=self._track_rpy_weight,
        )
        self._disabled_criteria_template = ToolPoseCriteria.disabled()

        # Per-slot disable-mask cache. Init = all-False (matches the
        # broadcast init criteria above). ``_solve_one_batch`` only
        # re-writes per-slot criteria for slots whose disable pattern
        # changed — saves N CUDA buffer writes per solve when the caller
        # passes fully-tracked targets (the common case).
        self._persisted_disable_per_slot: torch.Tensor = torch.zeros(
            (self.batch_size, len(self._tracked_tool_frames)),
            dtype=torch.bool,
            device=self.planner_device,
        )

        # Joint-order mapping: cuRobo order ↔ IsaacLab local controlled-joint order.
        curobo_jnames = list(self.ik_solver.joint_names)
        self._dof = len(curobo_jnames)
        self._curobo_joint_names = curobo_jnames

        # Sim-side joint reference — the ordered list of sim joints that
        # ``_build_full_joint_pos`` below will read from.
        self._ee_link_name = yaml_tool_frames[0] if yaml_tool_frames else None
        self._tool_frames = yaml_tool_frames  # for downstream reference / logs

        # Pad-slot hygiene: track last real batch size.
        self._last_B: Optional[int] = None

        # Worker thread / queue for this instance.
        self._queue: List[Tuple[concurrent.futures.Future, IKPlanRequest]] = []
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)
        self._shutdown = False

        # Warm up MUST run in the worker thread, not main: PyTorch's CUDA
        # context is per-thread, so the worker's first CUDA op (later, on a
        # real solve) would initialize a fresh per-thread context and
        # shuffle GPU memory pages — invalidating any CUDA graph captured
        # by the main thread in the meantime (e.g. ``curobo_ik_actions``'s
        # action-IK graph). Doing the warmup INSIDE the worker initializes
        # its CUDA context up front; we block __init__ on a ready Event so
        # downstream construction (and subsequent main-thread graph
        # captures) sees a stable warp/CUDA module layout.
        self._warmup_done_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._warmup_done_event.wait()

    def _warmup(self) -> None:
        """Run one ``solve_pose`` at the max tensor shape to prime kernels.

        Uses the robot's retract configuration as both the current state and
        the target pose (FK-consistent → every seed already satisfies the
        goal so the solve is trivial). Runs at num_goalset=1 and again at
        ``max_goalset`` if that path will actually be exercised, so the
        larger goalset tensor allocation happens here rather than on the
        first real goalset request.
        """
        try:
            default_js_1 = self.ik_solver.default_joint_state.clone().unsqueeze(0)
            kin = self.ik_solver.compute_kinematics(default_js_1)
            yaml_tool_frames = list(self.ik_solver.tool_frames)
            # Build retract pose per tool frame (same shape as a real solve).
            retract_poses: Dict[str, Pose] = {}
            for frame in yaml_tool_frames:
                retract_poses[frame] = kin.tool_poses.get_link_pose(
                    frame,
                    make_contiguous=True,
                )

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
                    ordered_tool_frames=yaml_tool_frames,
                    num_goalset=num_goalset,
                )
                state = JointState(
                    position=default_js_1.position.expand(B, -1)
                    .contiguous()
                    .to(self.planner_device),
                    velocity=torch.zeros(B, self._dof, device=self.planner_device),
                    acceleration=torch.zeros(B, self._dof, device=self.planner_device),
                    jerk=None,
                    joint_names=self._curobo_joint_names,
                )
                self.ik_solver.solve_pose(goal, current_state=state)

            _run(num_goalset=1)
            if self.max_goalset > 1:
                _run(num_goalset=self.max_goalset)
            if self._debug:
                print(
                    f"[IKServer][debug] Instance {self.instance_id} warmup "
                    f"done (B={self.batch_size}, G={{1, {self.max_goalset}}})."
                )
        except Exception as e:
            print(f"[IKServer][ERROR] Instance {self.instance_id} warmup failed: {e!r}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        with self._queue_cv:
            self._shutdown = True
            self._queue_cv.notify_all()
        if join:
            self._worker.join(timeout=timeout_s)

    def submit(self, fut: concurrent.futures.Future, req: IKPlanRequest):
        """Submit a request to this instance's queue."""
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
    # Frame helpers
    # ------------------------------------------------------------------

    def _world_to_robot_frame(
        self,
        target_pos: torch.Tensor,
        robot_base_pose: torch.Tensor,
        robot_base_quat: torch.Tensor,
        env_ids: List[int],
    ) -> torch.Tensor:
        """Transform target poses from world frame to robot base frame.

        Args:
            target_pos: [N, 7] for single mode, [N, E, 7] / [N, G, 7] otherwise.
            robot_base_pose: [num_envs, 3]
            robot_base_quat: [num_envs, 4]
            env_ids: List of environment IDs

        Returns:
            Transformed target_pos in robot frame, same shape as input.
        """
        device = robot_base_pose.device
        target_tensor = target_pos.to(device=device, dtype=torch.float32)

        original_ndim = target_tensor.ndim
        if original_ndim == 2:
            n = target_tensor.shape[0]
            target_tensor = target_tensor.unsqueeze(1)  # [N, 1, 7]
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
        base_positions = robot_base_pose[env_ids_tensor]  # [N, 3]
        base_quats = robot_base_quat[env_ids_tensor]  # [N, 4]

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

    def _build_full_joint_pos(self, joint_pos_envs: torch.Tensor) -> torch.Tensor:
        """Build joint_pos aligned to cuRobo joint order, padding locked joints."""
        target_joint_names = self._curobo_joint_names
        if not target_joint_names:
            return joint_pos_envs

        sim_joint_names: Optional[List[str]] = None
        if self.robot_manager.robots.get(self.robot_name) is not None:
            sim_joint_names = self.robot_manager.robots[self.robot_name].joint_names
        elif len(self.robot_manager.robots) > 0:
            sim_joint_names = next(iter(self.robot_manager.robots.values())).joint_names

        if sim_joint_names is None:
            return joint_pos_envs

        sim_index = {name: idx for idx, name in enumerate(sim_joint_names)}
        if isinstance(self.lock_joints, dict):
            lock_joint_values = self.lock_joints
        else:
            lock_joint_values = {name: 0.0 for name in self.lock_joints}

        device = joint_pos_envs.device
        dtype = joint_pos_envs.dtype
        batch = joint_pos_envs.shape[0]
        full_cols = []
        for name in target_joint_names:
            idx = sim_index.get(name, None)
            if idx is not None and idx < joint_pos_envs.shape[1]:
                full_cols.append(joint_pos_envs[:, idx : idx + 1])
            elif name in lock_joint_values:
                value = float(lock_joint_values[name])
                full_cols.append(
                    torch.full((batch, 1), value, device=device, dtype=dtype)
                )
            else:
                raise ValueError(
                    f"Missing joint '{name}' in robot state and lock_joints"
                )
        return torch.cat(full_cols, dim=1)

    def _preprocess_request(
        self,
        req: IKPlanRequest,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Preprocess: canonicalize ``target_pos`` to ``(N, G, L, 7)``
        based on ``req.mode``, transform world→robot if needed, extract
        seed JointState.

        Returns: ``env_ids, target_pos_canonical, joint_pos_curobo_order``.
        Mode is consumed here; downstream is mode-agnostic.
        """
        env_ids = [int(x) for x in req.env_ids]
        target_pos = req.target_pos
        mode = req.mode
        L = len(self._tracked_tool_frames)
        N = target_pos.shape[0]

        # ---- mode-driven shape canonicalization → (N, G, L, 7) -----------
        if mode == "single":
            if target_pos.ndim == 2:
                if target_pos.shape[1] != L * 7:
                    raise ValueError(
                        f"single 2-D target_pos last dim must be L*7={L * 7}; "
                        f"got {tuple(target_pos.shape)}. Position-only shapes "
                        f"are not accepted."
                    )
                target_pos = target_pos.view(N, 1, L, 7)
            elif target_pos.ndim == 3:
                if target_pos.shape[1] != L or target_pos.shape[2] != 7:
                    raise ValueError(
                        f"single 3-D target_pos must be (N, L={L}, 7); got "
                        f"{tuple(target_pos.shape)}"
                    )
                target_pos = target_pos.unsqueeze(1)  # (N, 1, L, 7)
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
                        f"got {tuple(target_pos.shape)}. Position-only shapes "
                        f"are not accepted."
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
        # target_pos is now canonical (N, G, L, 7).

        # ---- World → robot-frame transform (per-pose) -------------------
        # Flatten middle dims so the (N, K, 7) helper operates uniformly.
        if self.relative_to_world_frame:
            shape_in = target_pos.shape
            K = shape_in[1] * shape_in[2]
            flat3 = target_pos.reshape(N, K, 7)
            flat3 = self._world_to_robot_frame(
                flat3,
                req.robot_states["base_pos"],
                req.robot_states["base_quat"],
                env_ids,
            )
            target_pos = flat3.reshape(shape_in)

        joint_pos = req.robot_states["joint_pos"]
        env_ids_tensor = torch.tensor(
            env_ids, device=joint_pos.device, dtype=torch.long
        )
        joint_pos_envs = joint_pos[env_ids_tensor]
        joint_pos_envs = self._build_full_joint_pos(joint_pos_envs)
        return env_ids, target_pos, joint_pos_envs

    # ------------------------------------------------------------------
    # Solver call
    # ------------------------------------------------------------------

    def _solve_one_batch(
        self,
        batch_env_ids: List[int],
        batch_target_pos: torch.Tensor,
        batch_joint_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[List[bool], List[int]]:
        """Solve IK for one chunk; returns aligned lists for ``batch_env_ids``.

        Args:
            batch_env_ids: env IDs for this chunk (length B_actual).
            batch_target_pos: canonical ``(B, G, L, 7)``. ``G=1`` for the
                single-pose case. See :class:`IKPlanRequest` for the
                input shapes that get canonicalized to this layout.
            batch_joint_pos: ``[B, dof]`` in cuRobo joint order (seed / current_state).

        Returns:
            (success: List[bool], goalset_index: Optional[List[int]])
        """
        B_actual = len(batch_env_ids)
        assert B_actual == batch_target_pos.shape[0], (
            f"Input alignment mismatch: env_ids={B_actual}, "
            f"targets={batch_target_pos.shape[0]}"
        )

        # ----- Per-slot scene reload --------------------------------------
        # slot i -> batch_env_ids[i] -> world_cfg_list[batch_env_ids[i]] -> env_idx=i
        with self._world_lock:
            for slot, env_id in enumerate(batch_env_ids):
                env_id_int = int(env_id)
                if env_id_int < 0 or env_id_int >= len(self.world_cfg_list):
                    raise ValueError(
                        f"env_id {env_id_int} out of range [0, {len(self.world_cfg_list)}) "
                        f"at slot {slot} in batch_env_ids={batch_env_ids}"
                    )
                scene_cfg = self.world_cfg_list[env_id_int]
                if isinstance(scene_cfg, Scene):
                    scene = scene_cfg
                elif isinstance(scene_cfg, dict) and scene_cfg:
                    scene = Scene.create(scene_cfg)
                else:
                    scene = Scene()
                self.ik_solver.scene_collision_checker.load_collision_model(
                    scene,
                    env_idx=int(slot),
                )
                if self._debug:
                    log_scene_slot_load(scene, int(slot), env_id_int, tag="IK")
            # Pad-slot hygiene: only reload empty scenes when B shrinks.
            if (
                self._last_B is None
                or self._last_B > B_actual
                or self._last_B == self.batch_size
            ):
                for pad in range(B_actual, self.batch_size):
                    self.ik_solver.scene_collision_checker.load_collision_model(
                        Scene(),
                        env_idx=pad,
                    )
            self._last_B = B_actual

        # ----- Build current_state (seed) ---------------------------------
        if batch_joint_pos is None:
            raise ValueError("batch_joint_pos is required for v2 solve_pose")
        current_state_pos = batch_joint_pos.to(self.planner_device).contiguous()
        # NOTE: we construct JointState manually (not via ``from_position``)
        # and leave ``jerk=None``. Upstream v2's ``_pad_batch_inputs``
        # (``solver_ik.py``) pads position/velocity/acceleration but not
        # jerk — so ``from_position`` (which sets ``jerk=zeros(B, dof)``)
        # triggers a shape-mismatch in ``manager_goal.update_goal_buffer``
        # whenever ``batch_size < max_batch_size``. With ``jerk=None`` the
        # JIT copy in ``state_joint_jit_helpers`` skips the jerk branch.
        current_state = JointState(
            position=current_state_pos,
            velocity=torch.zeros_like(current_state_pos),
            acceleration=torch.zeros_like(current_state_pos),
            jerk=None,
            joint_names=self._curobo_joint_names,
        )

        # ----- Canonical-tensor pipeline -----------------------------------
        # Input arrives already canonicalized by ``_preprocess_request``:
        #   - single:  (B, L, 7)         → unsqueeze G=1  →  (B, L, 1, 7)
        #   - goalset: (B, G, L, 7)      → permute G↔L     →  (B, L, G, 7)
        # The internal helper ``detect_nan_and_pad_goalset`` operates on
        # (B, L, G, 7) — mode-uniform.
        #
        # NaN semantics (per env, per tracked tool):
        #   - ALL G slots NaN ⇒ tool disabled this solve (criterion weight
        #     zeroed via ``update_tool_pose_criteria_per_env``).
        #   - Some NaN ⇒ helper mod-pads the NaN slots from the tool's own
        #     real candidates.
        #   - paired=True: per env, active-tool real_counts must be equal
        #     across active tools (helper raises otherwise).
        # Disabled-tool slots are value-substituted with current FK so
        # curobo never sees NaN goals (criteria weight is zero anyway).
        pose_dict: Dict[str, Pose] = {}
        tracked = self._tracked_tool_frames
        num_tracked = len(tracked)

        # Compute current FK ONCE — reused for NaN substitution filler AND
        # info-only (extra_fk_link) frame filler at the end of this method.
        fk_kin = self.ik_solver.compute_kinematics(current_state)
        fk_pose_per_frame: Dict[str, Pose] = {}
        for frame in self.ik_solver.tool_frames:
            fk_pose_per_frame[frame] = fk_kin.tool_poses.get_link_pose(
                frame,
                make_contiguous=True,
            )
        # (B, L_tracked, 7) FK filler for tracked frames.
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

        # Run NaN preprocessing on the canonical tensor. Returns:
        #   padded (B, L, G, 7): NaN slots inside partially-NaN tools
        #     replaced by mod-cycled real candidates.
        #   fully_nan (B, L) bool: True where every (env, tool) slot is NaN
        #     ⇒ disable that tool for that env this solve.
        padded, fully_nan, _real_count = detect_nan_and_pad_goalset(
            canonical,
            tracked,
            paired=self._paired,
        )
        # Substitute fully-NaN tools with FK filler broadcast across G.
        if bool(fully_nan.any().item()):
            fk_g = fk_filler.unsqueeze(2).expand(B_actual, num_tracked, num_goalset, 7)
            full_mask = fully_nan.unsqueeze(-1).unsqueeze(-1).expand_as(padded)
            padded = torch.where(full_mask, fk_g, padded)

        # Per-env runtime criteria update — every solve writes a fresh
        # row per (slot, tracked_frame). No state carries between solves.
        # NOTE: ``update_tool_pose_criteria_per_env(env_idx, ...)`` takes
        # the SOLVER SLOT INDEX (0..max_batch_size-1), NOT the game-level
        # ``batch_env_ids[b]``. This matches the same indexing the scene
        # loader uses (``load_collision_model(scene, env_idx=slot)``) so
        # slot ``b`` consistently refers to: scene[batch_env_ids[b]],
        # current_state[b], target[b], result.success[b], criteria row b.
        # Pad slots [B_actual..max_batch_size) keep their previous criteria
        # rows — that's fine because their target rows are also pad rows
        # (auto-padded by the solver) and their results are sliced out.
        # Diff against the persisted cache and only write the slots whose
        # disable pattern actually changed (skips B CUDA writes + syncs
        # when the caller passes fully-tracked targets — common case).
        cached_dev = self._persisted_disable_per_slot[:B_actual]
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
                self.ik_solver.update_tool_pose_criteria_per_env(b, crit)
            self._persisted_disable_per_slot[:B_actual] = fully_nan.to(
                self._persisted_disable_per_slot.device
            )

        # Build pose_dict from canonical (B, L, G, 7) — single-frame goalset
        # uses G=1 reshape; multi-frame goalset uses G>1 reshape.
        for li, frame in enumerate(tracked):
            pose_dict[frame] = Pose(
                position=padded[:, li, :, :3]
                .reshape(B_actual * num_goalset, 3)
                .contiguous(),
                quaternion=padded[:, li, :, 3:]
                .reshape(B_actual * num_goalset, 4)
                .contiguous(),
            )

        # Fill non-tracked (info-only / extra_fk_link) frames with current FK
        # so GoalToolPose is shape-complete. Zero criterion weight → value
        # does not affect the solve. Reuses ``fk_pose_per_frame`` already
        # computed above; no extra FK call.
        needs_fk = [f for f in self.ik_solver.tool_frames if f not in pose_dict]
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
            ordered_tool_frames=list(self.ik_solver.tool_frames),
            num_goalset=num_goalset,
        )

        # ----- Solve -------------------------------------------------------
        if self._debug:
            log_joint_state_submit(current_state, batch_env_ids, tag="IK")
        # Anchor every IK seed to the live joint state. The default cuRobo
        # path only seeds 1 of N from current_state and fills the rest
        # with random configs — for goalset solves this means the picked
        # candidate is whichever the random seeds happened to converge
        # on, NOT the one closest to the robot's current pose. Replicate
        # current_state across all ``num_seeds`` and add a small
        # per-joint perturbation (~6°) on seed[1:] to keep enough
        # diversity to escape local minima while staying close to the
        # current configuration. seed[0] is left exact as an anchor.
        # solver_ik.solve_pose passes seed_config straight through to
        # ``prepare_action_seeds``; with ``n_provided_seeds == num_seeds``
        # no random fill happens (manager_seed.py:123-124).
        seed_config = _build_seed_config(current_state.position, self.num_seeds)
        _t0 = time.time()
        result = self.ik_solver.solve_pose(
            goal, current_state=current_state, seed_config=seed_config
        )
        _dt = time.time() - _t0

        # ----- Extract success + goalset_index ----------------------------
        # Single + goalset are unified at the v2 API level: ``solve_pose``
        # always returns ``goalset_index`` (trivially 0 for G=1). Failed
        # envs report -1.
        success_t = result.success
        if success_t.dim() == 2:
            success_t = success_t[:, 0]
        success = [bool(success_t[i].item()) for i in range(B_actual)]

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

        # ----- Debug line (pos/rot error replaces v1 status) --------------
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
            f"[IKServer][debug] Instance {self.instance_id} Solved batch "
            f"{batch_env_ids} (G={num_goalset}) success={success} "
            f"goalset_index={goalset_index} pos_err_max={pos_err:.4f}m "
            f"rot_err_max={rot_err:.4f}rad dt={_dt:.4f}s"
        )
        # Per-goal diagnostic when whole batch fails — distinguishes
        # IK convergence misses from collision-rejected solutions.
        if self._debug and not any(success):
            try:
                pe = result.position_error
                re_ = result.rotation_error
                sc = result.success
                if pe is not None and pe.dim() >= 2:
                    pos_per_goal = pe[0].flatten().detach().cpu().tolist()
                    rot_per_goal = (
                        re_[0].flatten().detach().cpu().tolist()
                        if re_ is not None and re_.dim() >= 2
                        else [float("nan")] * len(pos_per_goal)
                    )
                    succ_per_goal = (
                        sc[0].flatten().detach().cpu().tolist()
                        if sc.dim() >= 2
                        else [bool(sc[0].item())]
                    )
                    pos_thr = float(
                        getattr(self.ik_solver.config, "position_tolerance", 0.005)
                    )
                    rot_thr = float(
                        getattr(self.ik_solver.config, "orientation_tolerance", 0.05)
                    )
                    n_within_pose = sum(
                        1
                        for p, r in zip(pos_per_goal, rot_per_goal)
                        if p <= pos_thr and r <= rot_thr
                    )
                    print(
                        f"[IKServer][debug-pergoal] within_pose_thresh="
                        f"{n_within_pose}/{len(pos_per_goal)} "
                        f"(pos_thr={pos_thr}, rot_thr={rot_thr}) — "
                        f"succ={succ_per_goal[:8]}{'...' if len(succ_per_goal) > 8 else ''} "
                        f"pos_err={[f'{p:.4f}' for p in pos_per_goal[:8]]} "
                        f"rot_err={[f'{r:.4f}' for r in rot_per_goal[:8]]}"
                    )
                    if n_within_pose > 0:
                        print(
                            "[IKServer][debug-pergoal] => candidates HIT pose "
                            "threshold but success=False → collision check "
                            "rejecting (self or world)."
                        )
            except Exception as _ex:  # noqa: BLE001
                print(f"[IKServer][debug-pergoal] dump failed: {_ex}")

        assert len(success) == B_actual and len(goalset_index) == B_actual, (
            f"Result alignment mismatch: success={len(success)}, "
            f"goalset_index={len(goalset_index)}, env_ids={B_actual}"
        )

        return success, goalset_index

    # ------------------------------------------------------------------
    # Worker loop (unchanged v1 structure)
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Worker loop — pulls requests from queue, microbatches, solves, fans out."""
        # Warm up FIRST — this initializes this thread's per-thread CUDA
        # context and JITs all warp/cuRobo kernels we'll need. See the
        # __init__ comment above for why this must run on the worker
        # thread, not main. __init__ blocks on _warmup_done_event until we
        # signal here.
        try:
            self._warmup()
        finally:
            self._warmup_done_event.set()

        while True:
            with self._queue_cv:
                while not self._shutdown and len(self._queue) == 0:
                    self._queue_cv.wait()
                if self._shutdown:
                    return

                # Take everything currently queued; optionally wait a little for more.
                batch: List[Tuple[concurrent.futures.Future, IKPlanRequest]] = []
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
                ]
            ] = []

            try:
                t_req0 = time.time()
                for fut, req in batch:
                    try:
                        env_ids, tgt, jp = self._preprocess_request(req)
                        preprocessed.append((fut, env_ids, tgt, jp))
                    except Exception as ex:
                        tb = traceback.format_exc()
                        print(
                            f"[IKServer][ERROR] Instance {self.instance_id} "
                            f"preprocessing failed for request envs={req.env_ids}: "
                            f"{type(ex).__name__}: {ex}\n{tb}"
                        )
                        fut.set_exception(ex)

                if len(preprocessed) == 0:
                    continue

                # Group by G only. After ``_preprocess_request`` every
                # request is canonical (N, G, L, 7) — single is just
                # G=1. L is fixed per Server (doesn't enter the key); G
                # varies because the kernel JITs per ``num_goalset``.
                groups: Dict[
                    int,
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

                for fut, env_ids, tgt, jp in preprocessed:
                    per_req_env_ids[fut] = env_ids
                    if len(env_ids) != tgt.shape[0]:
                        raise ValueError(
                            f"env_ids length {len(env_ids)} mismatch with "
                            f"target_pos batch {tgt.shape[0]}"
                        )
                    G = int(tgt.shape[1])
                    groups.setdefault(G, []).append((fut, env_ids, tgt, jp))

                for G, group_requests in groups.items():
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
                                if self._debug:
                                    print(
                                        f"[IKServer][debug] Instance {self.instance_id} "
                                        f"Env {env_id} already in microbatch, reusing "
                                        f"result from index {existing}"
                                    )
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
                    merged_joint_pos_tensor = (
                        torch.stack(merged_joint_pos, dim=0)
                        if merged_joint_pos
                        else None
                    )

                    merged_success: List[bool] = [False] * len(merged_env_ids)
                    merged_goalset_index: List[int] = [-1] * len(merged_env_ids)

                    for s in range(0, len(merged_env_ids), self.batch_size):
                        e = min(s + self.batch_size, len(merged_env_ids))
                        batch_env_ids = merged_env_ids[s:e]
                        batch_target_pos = merged_target_tensor[s:e]
                        batch_joint_pos = (
                            merged_joint_pos_tensor[s:e]
                            if merged_joint_pos_tensor is not None
                            else None
                        )
                        assert len(batch_env_ids) == batch_target_pos.shape[0]
                        success_chunk, gi_chunk = self._solve_one_batch(
                            batch_env_ids,
                            batch_target_pos,
                            batch_joint_pos=batch_joint_pos,
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
                        assert len(success_out) == len(stored_env_ids)
                        assert len(goalset_out) == len(stored_env_ids)
                        fut.set_result((success_out, goalset_out, stored_env_ids))
                        if self._debug:
                            print(
                                f"[IKServer][debug] Instance {self.instance_id} "
                                f"Replied to future {id(fut)}: env_ids={stored_env_ids}, "
                                f"G={G}, results={len(success_out)}"
                            )

                if self._debug:
                    print(
                        f"[IKServer][debug] Instance {self.instance_id} served "
                        f"{len(preprocessed)} request(s) dt_total="
                        f"{time.time() - t_req0:.4f}s"
                    )
            except Exception as ex:
                print(
                    f"[IKServer][ERROR] Instance {self.instance_id} "
                    f"worker exception — {type(ex).__name__}: {ex}\n"
                    f"{traceback.format_exc()}"
                )
                for fut, *_ in preprocessed:
                    if not fut.done():
                        fut.set_exception(ex)
                for fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(ex)


class IKServer:
    """In-process cuRobo 2.0 IK service with instance pool and microbatching.

    Architecture preserved from v1:
    - Pool of ``_IKInstance`` workers (one thread each).
    - Per-request load balancing (idle first, else least-loaded).
    - Microbatch window merges multiple requests into one solve.
    - Per-env scene dict cache, updated via ``update_world(env_ids, world_cfgs)``.
    - Each request gets its own Future with ``(success, goalset_index, env_ids)``.

    New in v2:
    - YAML-driven ``extra_fk_link`` + ``info_links`` contract (see module
      docstring). Caller (PlannerManager) merges ``extra_fk_link`` into
      ``robot_cfg.kinematics.tool_frames`` before handing it to cuRobo, then
      forwards the list here so the Server can disable those frames.
    - World cfg cache is ``List[dict]`` (SceneCfg dicts) — converted to
      ``Scene`` per chunk.
    - Pad-slot hygiene on B shrink.
    """

    dual_mode: bool = False

    def __init__(
        self,
        robot_manager: RobotManager,
        robot_cfg: dict,
        robot_name: str,
        device: torch.device,
        batch_size: int = 2,
        microbatch_wait_ms: float = 200.0,
        num_instances: int = 2,
        num_seeds: int = 20,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        max_goalset: int = 10000,
        extra_fk_link: Optional[List[str]] = None,
        info_links: Optional[List[str]] = None,
        track_xyz_weight: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        track_rpy_weight: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        debug: bool = False,
        relative_to_world_frame: bool = True,
        planner_devices: Optional[Union[str, torch.device, Sequence]] = None,
        paired: bool = True,
    ):
        self.robot_manager = robot_manager
        self.robot_cfg = robot_cfg
        self.robot_name = robot_name
        self.device = device

        self.batch_size = min(int(batch_size), int(robot_manager.num_envs))
        self.microbatch_wait_s = max(float(microbatch_wait_ms) / 1000.0, 0.0)
        self.num_instances = num_instances
        self.num_seeds = num_seeds
        self.max_goalset = int(max(1, max_goalset))
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self._debug = debug
        self.relative_to_world_frame = relative_to_world_frame
        self._paired = bool(paired)

        # Resolve per-instance planner devices (one torch.device per instance).
        self.planner_devices: List[torch.device] = _normalize_planner_devices(
            planner_devices, self.num_instances
        )

        # World cfg cache — indexed by env_id; shared across instances.
        # v2: stores SceneCfg dicts; servers convert dict → Scene at chunk time.
        self.world_cfg_list: List[dict] = [{} for _ in range(robot_manager.num_envs)]
        self._world_lock = threading.Lock()

        self.instances: List[_IKInstance] = []
        for i in range(self.num_instances):
            instance = _IKInstance(
                instance_id=i,
                robot_cfg=robot_cfg,
                batch_size=self.batch_size,
                robot_manager=robot_manager,
                robot_name=robot_name,
                device=device,
                planner_device=self.planner_devices[i],
                world_cfg_list=self.world_cfg_list,
                world_lock=self._world_lock,
                num_seeds=num_seeds,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                max_goalset=self.max_goalset,
                microbatch_wait_s=self.microbatch_wait_s,
                extra_fk_link=extra_fk_link,
                info_links=info_links,
                track_xyz_weight=track_xyz_weight,
                track_rpy_weight=track_rpy_weight,
                debug=debug,
                relative_to_world_frame=self.relative_to_world_frame,
                paired=self._paired,
            )
            self.instances.append(instance)

        print(
            f"[IKServer] Initialized with {self.num_instances} instance(s), "
            f"batch_size={self.batch_size}, microbatch_wait_ms={microbatch_wait_ms}, "
            f"num_seeds={num_seeds}, max_goalset={self.max_goalset}, "
            f"paired={self._paired}, "
            f"planner_devices={[str(d) for d in self.planner_devices]}"
        )

    def shutdown(self, join: bool = False, timeout_s: Optional[float] = None):
        for instance in self.instances:
            instance.shutdown(join=join, timeout_s=timeout_s)

    def update_world(self, env_ids: List[int], world_cfgs: List[Union[Scene, dict]]):
        """Update cached scene defs for env_ids (shared across instances).

        In v2 we accept either :class:`curobo.scene.Scene` instances or
        raw SceneCfg dicts; ``_solve_one_batch`` dispatches at chunk time.
        """
        if len(env_ids) != len(world_cfgs):
            raise ValueError("env_ids and world_cfgs length mismatch")
        with self._world_lock:
            for env_id, cfg in zip(env_ids, world_cfgs):
                self.world_cfg_list[int(env_id)] = cfg if cfg is not None else {}

    def _select_instance(self) -> _IKInstance:
        idle_instances = [inst for inst in self.instances if inst.is_idle()]
        if len(idle_instances) > 0:
            least_loaded_idle = min(idle_instances, key=lambda inst: inst.queue_size())
            if self._debug and len(idle_instances) > 1:
                print(
                    f"[IKServer][debug] Multiple idle instances ({len(idle_instances)}), "
                    f"selecting instance {least_loaded_idle.instance_id} "
                    f"(queue_size={least_loaded_idle.queue_size()})"
                )
            return least_loaded_idle

        least_loaded = min(self.instances, key=lambda inst: inst.queue_size())
        if self._debug:
            print(
                f"[IKServer][debug] All instances busy, selecting least-loaded "
                f"instance {least_loaded.instance_id} "
                f"(queue_size={least_loaded.queue_size()})"
            )
        return least_loaded

    def submit_ik(self, req: IKPlanRequest):
        """Submit an IK request; returns a Future with
        ``(success, goalset_index, env_ids)``."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        instance = self._select_instance()
        instance.submit(fut, req)
        if self._debug:
            print(
                f"[IKServer][debug] Submitted request envs={req.env_ids} "
                f"to instance {instance.instance_id} "
                f"(queue_size={instance.queue_size()})"
            )
        return fut
