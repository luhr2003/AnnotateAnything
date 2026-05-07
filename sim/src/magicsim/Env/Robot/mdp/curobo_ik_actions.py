"""cuRobo 2.0 batched IK action term — unified single/multi tool-frame.

Overview
--------
Every tick we feed the whole ``num_envs`` batch into a **single**
:class:`curobo.inverse_kinematics.InverseKinematics` call. The robot YAML
declares ``tool_frames: [...]`` (length ``L``); we track every one of them
at once. Set ``L = 1`` for single-arm robots; ``L = 2`` for dual-arm (or
any humanoid multi-EE stack that lists multiple tool frames in one YAML).

This action term **does not** attach a world/scene collision model to the
solver (``scene_model=None``). Only the robot YAML's self-collision +
joint-limit constraints apply. Scene collision avoidance stays in the
Service layer (see ``CUROBO_V2_02_MIGRATION_PLAN.md``).

Input contract per env: a ``7 * L`` vector — ``L`` consecutive 7-tuples
``[x, y, z, qw, qx, qy, qz]`` in env-origin-relative world frame, one per
entry of ``cfg.tool_frames`` (in the same order). Output: joint position
targets for the joints listed in ``cfg.joint_names``.

v1 → v2 deltas (see ``CUROBO_V2_02_MIGRATION_PLAN.md``)
-------------------------------------------------------
* ``IKSolver`` + ``IKSolverConfig.load_from_robot_config(... world_model, ...)``
  → :class:`InverseKinematics` + :meth:`InverseKinematicsCfg.create`
  with ``scene_model=None``.
* ``solve_batch(Pose(pos, quat), seed_config, retract_config,
  link_poses=...)`` → one ``solve_pose(GoalToolPose, current_state=JointState)``
  call. The per-link dict and the retract/seed tensors are both folded in.
* The per-arm reference-frame / Jacobian-rotation logic that the v1 dual
  variant needed (``subtract_frame_transforms`` / ``_arm_jacobian_in_ref``)
  goes away — the dual-arm YAML encodes both kinematic chains from the
  shared articulation root, so we feed world-frame targets directly.

Device model
------------
Same as v1: two device domains that may differ.

* ``self.device``         — IsaacLab simulation device.
* ``self._curobo_device`` — solver device (``cfg.curobo_device``; falls
  back to sim device, then to any CUDA device if sim is CPU — cuRobo 2.0
  still requires CUDA for fused kinematics).

Every tensor that crosses the boundary is moved explicitly. Buffers that
live on the solver side (``_seed_buf`` and the preallocated zero goal
tensors) are allocated on ``_curobo_device``.

Decimation
----------
When ``L == 1`` the inter-decimation Jacobian diff-IK (the v1 behaviour) is
preserved. When ``L > 1`` the diff-IK fallback is silently clamped to
``decimation = 1`` and a warning is logged — multi-task diff-IK (stacked
Jacobian for ``L`` frames) is a follow-up; the v2 batched IK at
``use_cuda_graph=True`` is usually fast enough to fire every step.

NaN handling (preserved from v1)
--------------------------------
Identical per-slice semantics to the v1 single-arm / dual-arm variants:

* If a row's 7-slot for tool frame ``i`` is NaN, we fall back to the last
  valid action recorded for that env + frame (``_last_valid_action``).
* If there is no last valid action yet (first tick), we fall back to the
  current EEF pose for that frame (``body_link_state_w`` - env origin).
* Buffers stay shape-consistent; we never short-circuit the solver call.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from gymnasium import spaces

from isaaclab.assets.articulation import Articulation
from magicsim.Env.Robot.mdp.action_manager import ActionTerm
from magicsim.Env.Robot.mdp.differential_ik import DifferentialIKController
from magicsim.Env.Robot.mdp.differential_ik_cfg import DifferentialIKControllerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import curobo_ik_cfg


class CuroboIKAction(ActionTerm):
    """cuRobo 2.0 batched IK, ``L`` tool frames in a single solve.

    Input per env: ``7 * L`` floats, one 7-tuple ``(x, y, z, qw, qx, qy, qz)``
    per tool frame (env-origin-relative world frame), ordered to match
    ``cfg.tool_frames`` when set, else the YAML's ``kinematics.tool_frames``.
    """

    cfg: curobo_ik_cfg.CuroboIKActionCfg
    _asset: Articulation

    def __init__(self, cfg: curobo_ik_cfg.CuroboIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env
        self._step_count = 0

        self._curobo_device = self._resolve_curobo_device(cfg.curobo_device)

        self._initialize_joint_info()
        self._initialize_curobo()  # sets self._tool_frames, self._L
        self._initialize_body_indices()  # uses self._tool_frames
        self._initialize_diff_ik()
        self._initialize_buffers()
        self._warmup_solver()

        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_curobo_device(raw: str | None) -> str:
        """cuRobo 2.0 requires CUDA. Resolve ``raw`` to a concrete CUDA device string."""
        if raw is None:
            raw = (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        dev = torch.device(raw)
        if dev.type == "cuda":
            return str(dev)
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        raise RuntimeError(
            "CuRobo IK requires CUDA (fused kinematics). No CUDA device is available."
        )

    def _initialize_joint_info(self) -> None:
        """Resolve controlled-joint IDs / Jacobian-column offsets."""
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)

        # PhysX jacobian column offsets (same logic as DifferentialInverseKinematicsAction).
        if self._asset.is_fixed_base:
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

    def _initialize_curobo(self) -> None:
        """Build :class:`InverseKinematics` and resolve ``tool_frames`` + joint-order maps."""
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg

        device_cfg = DeviceCfg(
            device=torch.device(self._curobo_device),
            dtype=torch.float32,
        )

        ik_cfg_kwargs = dict(
            robot=self.cfg.robot_cfg_file,
            device_cfg=device_cfg,
            num_seeds=self.cfg.num_seeds,
            position_tolerance=self.cfg.position_threshold,
            orientation_tolerance=self.cfg.rotation_threshold,
            self_collision_check=self.cfg.self_collision_check,
            scene_model=None,  # no world collision
            max_batch_size=self.num_envs,  # one problem per env
            multi_env=False,  # shared (empty) world
            max_goalset=1,
            # Disabled: running curobo IK in two concurrent instances on
            # the same device (this action-side IK + the IKServer /
            # MotionGenServer worker threads) races CUDA-graph replay
            # against non-graph allocations on every physics step. The
            # graph's pool-isolation isn't strong enough to tolerate
            # concurrent regular allocations — the graph eventually
            # reads/writes pages the worker has repurposed, yielding
            # CUDA_ERROR_ILLEGAL_ADDRESS. Warmup-ordering alone doesn't
            # fix this (it only addresses init-time races). To reclaim
            # this perf, route action-IK through the IKServer queue
            # instead of keeping two concurrent solver instances.
            use_cuda_graph=False,
        )
        ik_cfg = InverseKinematicsCfg.create(**ik_cfg_kwargs)
        self._ik = InverseKinematics(ik_cfg)

        # ``cfg.tool_frames`` (when set) must be a subset of the YAML's
        # ``kinematics.tool_frames``. When it is None (the default for
        # single-arm cfgs) we just use the YAML's list verbatim.
        cfg_tool_frames = list(self.cfg.tool_frames) if self.cfg.tool_frames else None
        if cfg_tool_frames is not None:
            missing = set(cfg_tool_frames) - set(self._ik.tool_frames)
            if missing:
                raise ValueError(
                    f"cfg.tool_frames {cfg_tool_frames} contains frames not declared in "
                    f"robot YAML {self.cfg.robot_cfg_file} "
                    f"(YAML tool_frames={list(self._ik.tool_frames)})."
                )
            self._tool_frames: list[str] = cfg_tool_frames
        else:
            self._tool_frames = list(self._ik.tool_frames)

        self._L = len(self._tool_frames)
        if self._L == 0:
            raise ValueError(
                f"Robot YAML {self.cfg.robot_cfg_file} declares no tool_frames "
                f"and none were provided via cfg.tool_frames."
            )

        # Joint-order map between cuRobo's kinematics.joint_names and our
        # controlled-joint subset (self._joint_names).
        curobo_jnames = list(self._ik.joint_names)
        self._dof = len(curobo_jnames)
        name_to_local = {n: i for i, n in enumerate(self._joint_names)}
        self._curobo_to_local: list[int] = []
        for jname in curobo_jnames:
            if jname not in name_to_local:
                raise RuntimeError(
                    f"cuRobo joint '{jname}' not found among controlled joints "
                    f"{self._joint_names}. Check joint_names in CuroboIKActionCfg."
                )
            self._curobo_to_local.append(name_to_local[jname])
        self._local_to_curobo = [0] * self._num_joints
        for curobo_idx, local_idx in enumerate(self._curobo_to_local):
            self._local_to_curobo[local_idx] = curobo_idx

    def _initialize_body_indices(self) -> None:
        """Body indices for each tool frame (used for NaN fallback / FK on sim side)."""
        self._eef_body_idx: list[int] = [
            self._asset.data.body_names.index(name) for name in self._tool_frames
        ]

    def _initialize_diff_ik(self) -> None:
        """Inter-decimation diff-IK — enabled only for ``L == 1`` with a method set.

        When ``cfg.diff_ik_method is None`` (or ``L > 1`` which has no
        multi-task implementation yet) we hold the last cuRobo solution
        between ticks. ``_effective_decimation`` stays at ``cfg.decimation``
        so cuRobo fires at the user-specified cadence; intermediate ticks
        just leave ``_processed_actions`` untouched.
        """
        self._diff_ik: DifferentialIKController | None = None
        self._effective_decimation = int(max(1, self.cfg.decimation))

        method = self.cfg.diff_ik_method
        if method is None:
            return

        if self._L != 1:
            warnings.warn(
                f"[CuroboIKAction] L={self._L} tool frames with diff_ik_method={method!r}: "
                "multi-task differential IK is not implemented yet — inter-decimation "
                "refinement is disabled (last cuRobo solution held). cuRobo still fires "
                "every 'decimation' ticks.",
                UserWarning,
                stacklevel=2,
            )
            return

        diff_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=method,
        )
        self._diff_ik = DifferentialIKController(
            cfg=diff_cfg, num_envs=self.num_envs, device=self.device
        )

    def _initialize_buffers(self) -> None:
        """Allocate sim- and solver-side buffers."""
        # Sim-side buffers (live on self.device).
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._last_valid_action = torch.full(
            (self.num_envs, self.action_dim), float("nan"), device=self.device
        )
        # Desired EEF pose retained across decimation ticks so diff-IK always
        # has a target. Stored as (N, L, 7) env-origin-relative.
        self._desired_eef_pose = torch.zeros(
            self.num_envs, self._L, 7, device=self.device
        )
        self._processed_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

        # Solver-side buffer used as the IK seed / initial current_state.
        self._seed_buf = torch.zeros(
            self.num_envs, self._dof, device=self._curobo_device
        )

    def _warmup_solver(self) -> None:
        """Pre-compile CUDA graph with the exact call signature used at runtime."""
        from curobo.types import GoalToolPose, JointState, Pose

        dummy_pos = torch.zeros(self.num_envs, 3, device=self._curobo_device)
        dummy_quat = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self._curobo_device)
            .expand(self.num_envs, -1)
            .contiguous()
        )
        pose = Pose(position=dummy_pos, quaternion=dummy_quat)
        dummy_goal = GoalToolPose.from_poses(
            {frame: pose for frame in self._tool_frames},
            ordered_tool_frames=list(self._ik.tool_frames),
            num_goalset=1,
        )
        dummy_state = JointState.from_position(
            self._seed_buf, joint_names=list(self._ik.joint_names)
        )
        self._ik.solve_pose(dummy_goal, current_state=dummy_state)

    # ------------------------------------------------------------------
    # Joint-order helpers
    # ------------------------------------------------------------------

    def _to_curobo_order(self, joint_pos_local: torch.Tensor) -> torch.Tensor:
        """``(N, num_joints)`` local order → ``(N, dof)`` cuRobo order.

        ``local[:, _curobo_to_local]`` produces a tensor whose column c is
        ``local[_curobo_to_local[c]]`` = local's value for the joint at
        cuRobo idx c — that's cuRobo-ordered output. The previous version
        used ``_local_to_curobo`` here, which is the inverse permutation
        and silently mis-mapped joints whenever cspace order ≠ DOF order
        (manifests on robots where IsaacLab's ``find_joints`` interleaves
        left/right arms — e.g. vega — but is invisible on robots whose
        DOF order matches the cuRobo cspace, so the bug went unnoticed
        on dual_piper / dual_arx_x5 / etc.).
        """
        return joint_pos_local[:, self._curobo_to_local]

    def _to_local_order(self, joint_pos_curobo: torch.Tensor) -> torch.Tensor:
        """``(N, dof)`` cuRobo order → ``(N, num_joints)`` local order.

        Inverse of :meth:`_to_curobo_order` — uses ``_local_to_curobo``."""
        return joint_pos_curobo[:, self._local_to_curobo]

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 7 * self._L

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def process_actions(self, actions: torch.Tensor, env_ids: torch.Tensor) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int64)

        valid_actions = self._resolve_nan_actions(actions, env_ids)

        self._raw_actions[env_ids] = valid_actions
        self._last_valid_action[env_ids] = valid_actions.clone()

        # Fill the full-env desired pose tensor, patching any rows that never
        # received a valid action with the current EEF pose.
        all_actions = self._last_valid_action.clone()
        nan_rows = torch.isnan(all_actions).any(dim=1)
        if nan_rows.any():
            nan_ids = torch.where(nan_rows)[0]
            all_actions[nan_ids] = self._get_current_eef_pose(nan_ids)

        self._desired_eef_pose.copy_(all_actions.view(self.num_envs, self._L, 7))

        self._step_count += 1
        # First tick always fires cuRobo, otherwise every ``decimation`` ticks.
        # Between ticks ``_solve_differential_ik`` runs (its default no-ops
        # when no diff-IK is configured; subclasses can override for
        # per-arm diff-IK).
        fire_curobo = self._step_count == 1 or (
            self._step_count % self._effective_decimation == 0
        )
        if fire_curobo:
            self._solve_curobo_batch()
        else:
            self._solve_differential_ik()

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._last_valid_action.fill_(float("nan"))
            self._desired_eef_pose.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._last_valid_action[env_ids] = float("nan")
            self._desired_eef_pose[env_ids] = 0.0
        if self._diff_ik is not None:
            self._diff_ik.reset(env_ids)

    # ------------------------------------------------------------------
    # Solver implementations
    # ------------------------------------------------------------------

    def _world_to_base_pose(
        self, pos_world: torch.Tensor, quat_world: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-express ``(pos_world, quat_world)`` shape ``(N, L, 3/4)`` in the
        cuRobo solver's base-link frame. Used only when
        ``cfg.world_to_base_frame`` is True.

        The base-link pose is read from ``body_link_state_w[base_link_idx]``
        when ``cfg.base_link_name`` is set — this is the live world pose of
        the link the cuRobo yaml declares as ``kinematics.base_link``,
        which on a mobile manipulator (vega) is *not* the articulation
        root (``vega_1p_mobile``) but a child link below the dummy_base
        virtual joints (``vega_1p_base``). Falls back to
        ``root_pos_w / root_quat_w`` when ``base_link_name`` is None.

        Quaternion convention is (w, x, y, z), matching the rest of cuRobo
        and IsaacLab.
        """
        N, L, _ = pos_world.shape
        base_link_name = getattr(self.cfg, "base_link_name", None)
        # IsaacLab ``body_link_state_w[..., :7]`` is in absolute world
        # frame. For vega's fixed-base articulation (PhysicsFixedJoint
        # anchoring vega_1p_mobile), the articulation root's *actual*
        # world pose is the USD-baked anchor — ``initial_pos_range`` and
        # ``write_root_pose_to_sim`` do NOT move that anchor, so the
        # spawn is NOT at ``initial_pos_range`` despite the cfg saying so.
        # The link's world pose comes straight from ``body_link_state_w``.
        if base_link_name is not None:
            if not hasattr(self, "_world_to_base_link_idx"):
                self._world_to_base_link_idx = self._asset.data.body_names.index(
                    base_link_name
                )
            link_state = self._asset.data.body_link_state_w[
                :, self._world_to_base_link_idx, :7
            ]  # (N, 7) — already world frame
            root_pos = link_state[:, :3].to(
                device=pos_world.device, dtype=pos_world.dtype
            )
            root_quat = link_state[:, 3:7].to(
                device=pos_world.device, dtype=quat_world.dtype
            )
        else:
            root_pos = self._asset.data.root_pos_w.to(
                device=pos_world.device, dtype=pos_world.dtype
            )
            root_quat = self._asset.data.root_quat_w.to(
                device=pos_world.device, dtype=quat_world.dtype
            )

        if not hasattr(self, "_world_to_base_logged"):
            self._world_to_base_logged = True
            print(
                f"[CuroboIKAction._world_to_base_pose] base_link_name={base_link_name!r} "
                f"base_world_pos[0]={root_pos[0].tolist()} "
                f"base_world_quat[0]={root_quat[0].tolist()} "
                f"input[0,0]_pos={pos_world[0, 0].tolist()} input[0,0]_quat={quat_world[0, 0].tolist()}"
            )

        # Inverse rotation = conjugate for unit quats. (w, -x, -y, -z).
        rq_inv = root_quat.clone()
        rq_inv[:, 1:] = -rq_inv[:, 1:]

        # Position: p_base = R(rq_inv) @ (p_world - root_pos)
        delta = pos_world - root_pos.unsqueeze(1)  # (N, L, 3)
        # Build (N, 3, 3) rotation matrix from quat.
        w, x, y, z = rq_inv.unbind(-1)
        R_inv = torch.stack(
            [
                torch.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    dim=-1,
                ),
            ],
            dim=-2,
        )  # (N, 3, 3)
        pos_base = torch.einsum("nij,nlj->nli", R_inv, delta)

        # Quaternion: q_base = rq_inv ⊗ q_world. Hamilton product, batched.
        rq_inv_e = rq_inv.unsqueeze(1).expand(N, L, 4).reshape(N * L, 4)
        q_w_flat = quat_world.reshape(N * L, 4)
        w1, x1, y1, z1 = rq_inv_e.unbind(-1)
        w2, x2, y2, z2 = q_w_flat.unbind(-1)
        quat_base = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        ).reshape(N, L, 4)
        return pos_base, quat_base

    def _solve_curobo_batch(self) -> None:
        """Single ``(num_envs, L)`` solve via :meth:`InverseKinematics.solve_pose`.

        Frame contract: cuRobo's kinematics model is rooted at the robot
        URDF's root frame. By default the action assumes each env's robot
        lives at ``env_origin`` in world coords, so
        ``env-origin-relative == robot-base frame`` and
        ``self._desired_eef_pose`` is fed directly.

        When ``cfg.world_to_base_frame`` is True, the input is interpreted
        as world-frame and re-expressed in the **live** articulation root
        frame (``root_pos_w`` / ``root_quat_w``) so the action works for
        robots spawned away from ``env_origin`` (mobile manipulators with
        a non-zero parked base, etc.). The diff-IK fallback below still
        uses world frame because IsaacLab's Jacobian + ``body_pos_w`` both
        live there.
        """
        from curobo.types import GoalToolPose, JointState, Pose

        base_pos = self._desired_eef_pose[..., :3]  # (N, L, 3)
        base_quat = self._desired_eef_pose[..., 3:7]  # (N, L, 4)

        # Optional world → robot-base transform. Reads the live root pose
        # so a parked mobile base (or any non-env_origin spawn) doesn't
        # break the cuRobo target frame.
        if getattr(self.cfg, "world_to_base_frame", False):
            base_pos, base_quat = self._world_to_base_pose(base_pos, base_quat)

        # Build one Pose per tool frame on the solver device.
        pose_dict: dict[str, Pose] = {}
        for li, frame in enumerate(self._tool_frames):
            pose_dict[frame] = Pose(
                position=base_pos[:, li, :].to(self._curobo_device).contiguous(),
                quaternion=base_quat[:, li, :].to(self._curobo_device).contiguous(),
            )

        goal = GoalToolPose.from_poses(
            pose_dict,
            ordered_tool_frames=list(self._ik.tool_frames),
            num_goalset=1,
        )

        # current_state: current joint positions in cuRobo joint order.
        cur_local = self._asset.data.joint_pos[:, self._joint_ids]  # sim device
        cur_curobo_order = self._to_curobo_order(cur_local)  # sim device
        self._seed_buf.copy_(cur_curobo_order)  # → solver device
        current_state = JointState.from_position(
            self._seed_buf, joint_names=list(self._ik.joint_names)
        )

        result = self._ik.solve_pose(goal, current_state=current_state)

        # result.js_solution.position: (N, return_seeds, dof) or (N, dof).
        js_pos = result.js_solution.position
        if js_pos.dim() == 3:
            js_pos = js_pos[:, 0, :]
        success = result.success
        if success.dim() == 2:
            success = success[:, 0]

        js_pos_sim = self._to_local_order(js_pos).to(self.device)
        success_sim = success.to(self.device)

        if self.cfg.fallback_to_current_on_fail:
            fail_ids = torch.where(~success_sim)[0]
            if fail_ids.numel() > 0:
                js_pos_sim[fail_ids] = self._asset.data.joint_pos[fail_ids][
                    :, self._joint_ids
                ]

        self._processed_actions.copy_(js_pos_sim)

    def _solve_differential_ik(self) -> None:
        """Single-step Jacobian IK toward ``self._desired_eef_pose``.

        Only the ``L == 1`` path is implemented here. The ``L > 1`` path is
        either held (no refinement) or handled by a subclass — see
        :class:`DualCuroboIKAction` for per-arm diff-IK.
        """
        if self._diff_ik is None:
            return

        # All tensors on sim device — no cross-device transfers.
        eef_idx = self._eef_body_idx[0]
        ee_pos_w = self._asset.data.body_pos_w[:, eef_idx]  # (N, 3)
        ee_quat_w = self._asset.data.body_quat_w[:, eef_idx]  # (N, 4)

        env_origins = self._env.scene.env_origins
        des_pos_w = self._desired_eef_pose[:, 0, :3] + env_origins
        des_quat_w = self._desired_eef_pose[:, 0, 3:7]

        if self._asset.is_fixed_base:
            jacobi_body_idx = eef_idx - 1
        else:
            jacobi_body_idx = eef_idx

        jacobian = self._asset.root_physx_view.get_jacobians()[
            :, jacobi_body_idx, :, self._jacobi_joint_ids
        ]  # (N, 6, num_joints)

        cur_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        self._diff_ik.set_command(command=torch.cat([des_pos_w, des_quat_w], dim=1))
        new_joint_pos = self._diff_ik.compute(
            ee_pos=ee_pos_w,
            ee_quat=ee_quat_w,
            jacobian=jacobian,
            joint_pos=cur_joint_pos,
        )
        self._processed_actions.copy_(new_joint_pos)

    # ------------------------------------------------------------------
    # NaN / fallback helpers
    # ------------------------------------------------------------------

    def _resolve_nan_actions(
        self, actions: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Per-slice NaN fallback: last-valid slice, else current EEF pose.

        Matches the v1 semantics — each tool-frame 7-slot falls back
        independently so one arm's NaN doesn't invalidate the other.
        """
        valid = actions.clone()
        per_frame_view = valid.view(-1, self._L, 7)

        # For each frame slice: find NaN rows, pull last-valid; if last-valid
        # is also NaN for that slice, read the current EEF pose.
        for li in range(self._L):
            slot = per_frame_view[:, li, :]
            nan_mask = torch.isnan(slot).any(dim=1)
            if not nan_mask.any():
                continue

            nan_local = torch.where(nan_mask)[0]
            nan_global = env_ids[nan_local]

            last_full = self._last_valid_action[nan_global].view(-1, self._L, 7)
            last_slot = last_full[:, li, :]
            has_last = (~torch.isnan(last_slot)).all(dim=1)

            need_current = ~has_last
            if need_current.any():
                cur_ids = nan_global[need_current]
                cur_pose = self._get_current_eef_pose_single(cur_ids, li)
                last_slot[need_current] = cur_pose

            per_frame_view[nan_local, li, :] = last_slot

        return valid

    def _get_current_eef_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        """``(len(env_ids), 7 * L)`` — current EEF poses across all tool frames.

        Env-origin-relative ``[x, y, z, qw, qx, qy, qz]`` per frame, ordered
        exactly like ``self._tool_frames``.
        """
        parts = [
            self._get_current_eef_pose_single(env_ids, li) for li in range(self._L)
        ]
        return torch.cat(parts, dim=1)

    def _get_current_eef_pose_single(
        self, env_ids: torch.Tensor, frame_idx: int
    ) -> torch.Tensor:
        """Current pose for one tool frame, env-origin-relative."""
        body_idx = self._eef_body_idx[frame_idx]
        state = self._asset.data.body_link_state_w[env_ids, body_idx, :7]
        pos = state[:, :3] - self._env.scene.env_origins[env_ids]
        return torch.cat([pos, state[:, 3:7]], dim=1)


# ======================================================================
# DualCuroboIKAction — v2 two-independent-arms variant
# ======================================================================


class DualCuroboIKAction(CuroboIKAction):
    """Same multi-tool-frame cuRobo batched solve as :class:`CuroboIKAction`,
    plus **per-arm** differential-IK between decimation ticks.

    Rationale: for DualSO101 / DualPiper / DualArxX5 the two arms have
    **disjoint joint sets** (R_joint.* vs L_joint.*). The cuRobo batch
    (``L = 2``) handles both EEFs in one solve — unchanged from the unified
    class. What differs is the inter-decimation refinement: a single shared
    Jacobian wouldn't drive both arms' targets coherently, so we run **two**
    :class:`DifferentialIKController`s, each sliced to its own arm's
    Jacobian columns and joint subset. With ``decimation > 1`` and
    ``diff_ik_method`` set, this gives you useful tracking between the
    cuRobo ticks (which the unified class can't do at ``L > 1``).

    Input per env remains ``14 = 7 * 2`` — right first, left second — in
    env-origin-relative world frame. cuRobo consumes it verbatim (the
    YAML's multi-tool-frame kinematics is rooted at the shared
    articulation base, so env-relative == solver-root for both arms).
    Diff-IK consumes the world frame (env-relative + ``env_origin``), same
    as the unified single-arm path — IsaacLab's Jacobian is in world frame.
    """

    cfg: "curobo_ik_cfg.DualCuroboIKActionCfg"

    def __init__(
        self, cfg: "curobo_ik_cfg.DualCuroboIKActionCfg", env: "ManagerBasedEnv"
    ):
        super().__init__(cfg, env)
        if self._L != 2:
            raise ValueError(
                f"DualCuroboIKAction requires exactly 2 tool frames; got L={self._L}. "
                f"For single-arm or fused multi-EEF stacks use CuroboIKAction directly."
            )

    # ------------------------------------------------------------------
    # Init overrides (called by the base class __init__)
    # ------------------------------------------------------------------

    def _initialize_joint_info(self) -> None:
        """Base-class unified joint IDs + extra per-arm slices / Jacobian cols."""
        super()._initialize_joint_info()

        r_ids, _ = self._asset.find_joints(self.cfg.right_joint_names)
        l_ids, _ = self._asset.find_joints(self.cfg.left_joint_names)
        self._right_joint_ids: list[int] = list(r_ids)
        self._left_joint_ids: list[int] = list(l_ids)
        self._right_num_joints = len(r_ids)
        self._left_num_joints = len(l_ids)

        if self._asset.is_fixed_base:
            self._right_jacobi_joint_ids = self._right_joint_ids
            self._left_jacobi_joint_ids = self._left_joint_ids
        else:
            self._right_jacobi_joint_ids = [i + 6 for i in self._right_joint_ids]
            self._left_jacobi_joint_ids = [i + 6 for i in self._left_joint_ids]

        # The cfg's ``joint_names`` is composed as right + left (see
        # DualCuroboIKActionCfg.__post_init__), so super()'s find_joints
        # returned ``_joint_ids`` in that order and ``_processed_actions``
        # slices [0, right_num_joints) = right, [right_num_joints, :) = left.

    def _initialize_diff_ik(self) -> None:
        """Two single-arm diff-IK controllers — one per arm."""
        # Suppress base-class unified diff-IK; we replace it with per-arm.
        self._diff_ik: DifferentialIKController | None = None
        self._effective_decimation = int(max(1, self.cfg.decimation))

        self._right_diff_ik: DifferentialIKController | None = None
        self._left_diff_ik: DifferentialIKController | None = None

        method = self.cfg.diff_ik_method
        if method is None:
            return

        diff_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=method,
        )
        self._right_diff_ik = DifferentialIKController(
            cfg=diff_cfg, num_envs=self.num_envs, device=self.device
        )
        self._left_diff_ik = DifferentialIKController(
            cfg=diff_cfg, num_envs=self.num_envs, device=self.device
        )

    # ------------------------------------------------------------------
    # Per-arm diff-IK
    # ------------------------------------------------------------------

    def _solve_differential_ik(self) -> None:
        """Per-arm world-frame diff-IK. No-op when diff_ik_method is ``None``."""
        if self._right_diff_ik is None or self._left_diff_ik is None:
            return

        env_origins = self._env.scene.env_origins
        # Right arm — slice [0, right_num_joints) of _processed_actions.
        self._solve_arm_diff_ik(
            diff_ik=self._right_diff_ik,
            eef_body_idx=self._eef_body_idx[0],
            joint_ids=self._right_joint_ids,
            jacobi_joint_ids=self._right_jacobi_joint_ids,
            des_pose_env_rel=self._desired_eef_pose[:, 0, :],
            slice_start=0,
            num_joints=self._right_num_joints,
            env_origins=env_origins,
        )
        # Left arm — slice [right_num_joints, :) of _processed_actions.
        self._solve_arm_diff_ik(
            diff_ik=self._left_diff_ik,
            eef_body_idx=self._eef_body_idx[1],
            joint_ids=self._left_joint_ids,
            jacobi_joint_ids=self._left_jacobi_joint_ids,
            des_pose_env_rel=self._desired_eef_pose[:, 1, :],
            slice_start=self._right_num_joints,
            num_joints=self._left_num_joints,
            env_origins=env_origins,
        )

    def _solve_arm_diff_ik(
        self,
        diff_ik: DifferentialIKController,
        eef_body_idx: int,
        joint_ids: list[int],
        jacobi_joint_ids: list[int],
        des_pose_env_rel: torch.Tensor,
        slice_start: int,
        num_joints: int,
        env_origins: torch.Tensor,
    ) -> None:
        ee_pos_w = self._asset.data.body_pos_w[:, eef_body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, eef_body_idx]
        des_pos_w = des_pose_env_rel[:, :3] + env_origins
        des_quat_w = des_pose_env_rel[:, 3:7]

        jacobi_body_idx = (
            eef_body_idx - 1 if self._asset.is_fixed_base else eef_body_idx
        )
        jacobian = self._asset.root_physx_view.get_jacobians()[
            :, jacobi_body_idx, :, jacobi_joint_ids
        ]  # (N, 6, num_joints)

        cur_joint_pos = self._asset.data.joint_pos[:, joint_ids]

        diff_ik.set_command(command=torch.cat([des_pos_w, des_quat_w], dim=1))
        new_joint_pos = diff_ik.compute(
            ee_pos=ee_pos_w,
            ee_quat=ee_quat_w,
            jacobian=jacobian,
            joint_pos=cur_joint_pos,
        )
        self._processed_actions[:, slice_start : slice_start + num_joints].copy_(
            new_joint_pos
        )

    # ------------------------------------------------------------------
    # Reset hook
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._right_diff_ik is not None:
            self._right_diff_ik.reset(env_ids)
        if self._left_diff_ik is not None:
            self._left_diff_ik.reset(env_ids)
