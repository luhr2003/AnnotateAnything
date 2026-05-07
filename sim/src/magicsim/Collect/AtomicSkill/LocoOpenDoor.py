import concurrent.futures
from typing import Any
import torch
from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
from magicsim.Env.Utils.viz import draw_waypoints
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.Env.Planner.PlannerManager import PlannerManager
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class LocoOpenDoor(AtomicSkill):
    """
    Atomic skill for opening a door in LocoOpenDoorEnv (mobile base + arm).
    Uses dexterous hand (pose 7 + hand joints hand_joint_dim); no gripper.

    Phase-list driven. Supports two annotation variants:
      - open-by-handle (pull):  phases = ["approach", "pull"]
      - rotate-and-push:        phases = ["approach", "rotate", "push"]

    Execution phases:
        1. pregrasp         — MobileMoveL to pre-grasp pose
        2. move_to_grasp    — MobileMoveL to phases[0][0] (first approach waypoint)
        3. phases[0..N-1]   — MobileServoL along each phase trajectory, in order
    IK goalset selection picks the best trajectory whose required phases are
    all non-empty. No fallback: fails if no trajectories / IK fails / any
    required phase is missing.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_target_pose = None
        self.current_phase = None
        self.handle_pose = None
        self.pre_grasp_pose = None

        self.pre_grasp_offset = getattr(config, "pre_grasp_offset", 0.15)

        # Trajectories from env: {key: {phase_name: Tensor [N,7]}}
        self.all_raw_trajectories = {}
        self.selected_raw_trajectory = None  # {phase_name: Tensor} after IK

        # Read annotation name, quat_format, and servo phase order from config
        traj_cfg = getattr(config, "trajectory", None)
        self._annotation_name = "dex3_1_open_by_handle_trajectory"
        self._quat_format = "xyzw"
        self._phases: list[str] = ["approach", "pull"]
        if traj_cfg is not None:
            self._annotation_name = getattr(
                traj_cfg, "annotation_name", self._annotation_name
            )
            self._quat_format = getattr(traj_cfg, "quat_format", self._quat_format)
            cfg_phases = getattr(traj_cfg, "phases", None)
            if cfg_phases is not None:
                self._phases = [str(p) for p in cfg_phases]
        if len(self._phases) == 0:
            raise ValueError("LocoOpenDoor: trajectory.phases must be non-empty")

        # Async IK pose selection state (following Grasp.py pattern)
        self._ik_job: dict | None = None
        self._ik_token: int = 0
        self.robot_name = None
        self.hand_id = 0
        self.ik_server = None
        self.planner_manager: PlannerManager = None

        # Track which joint/trajectory was selected by IK
        self.selected_trajectory_key = None  # e.g. "joint_0/0"
        self.selected_joint = None  # e.g. "joint_0"
        self.selected_trajectory_id = None  # e.g. "0"

        # Dexterous hand: pose 7 + N hand joints. Default 7 for dex3_1 (index_0/1,
        # middle_0/1, thumb_0/1/2). Override via atomic_skill config if the robot's
        # EEF action slot expects a different width.
        self._hand_joint_dim = int(getattr(config, "hand_joint_dim", 7))

    # ------------------------------------------------------------------ #
    # IK server helpers (following Grasp.py pattern)
    # ------------------------------------------------------------------ #

    def _get_planner_manager(self):
        planner_manager = getattr(self.env.scene, "planner_manager", None)
        if planner_manager is None:
            raise RuntimeError("PlannerManager is not available in the environment.")
        return planner_manager

    def _resolve_robot_name(self) -> str:
        if self.robot_name is not None:
            return self.robot_name
        robot_manager = getattr(self.env.scene, "robot_manager", None)
        if robot_manager is not None:
            robot_dict = getattr(robot_manager, "robots", None)
            if isinstance(robot_dict, dict) and len(robot_dict) > 0:
                self.robot_name = next(iter(robot_dict.keys()))
                return self.robot_name
        raise RuntimeError("Unable to resolve robot_name for IKServer.")

    def _get_robot_state(self) -> dict:
        robot_states = self.env.scene.robot_manager.get_robot_state(noise_flag=False)[0]
        if isinstance(robot_states, dict):
            robot_name = self._resolve_robot_name()
            if robot_name in robot_states:
                return robot_states[robot_name]
            return next(iter(robot_states.values()))
        return robot_states

    def _submit_goalset(self, grasp_pose_list: torch.Tensor):
        """Submit handle pose candidates as a goalset to IK server.
        For Loco (dual fixed/free base): use free-base IK (lock_base=False).
        """
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # Pack to (1, G, eef_num * 7) with NaN rows for the inactive arm.
        # See src/magicsim/Env/Planner/Services/README.md §5 + §7.
        target = self.pack_single_arm_goalset(grasp_pose_list)
        if self.config.get("debug", False):
            print(
                f"Submitting goalset to IK server (free base): {target.shape}, "
                f"hand_id={self.hand_id}, eef_num="
                f"{getattr(self.ik_server, 'eef_num', 1)}"
            )
        is_dual_ik = getattr(self.ik_server, "dual_mode", False)
        if is_dual_ik:
            req = DualIKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=robot_states_dict,
                mode="goalset",
                lock_base=False,
            )
        else:
            req = IKPlanRequest(
                env_ids=[self.env_id],
                target_pos=target,
                robot_states=robot_states_dict,
                mode="goalset",
            )
        return self.ik_server.submit_ik(req)

    # ------------------------------------------------------------------ #
    # Async handle pose selection via IK goalset
    # ------------------------------------------------------------------ #

    def _start_ik_job(self, candidate_poses: torch.Tensor, candidate_keys: list[str]):
        """Start async IK goalset job for handle pose candidates."""
        self._ik_token += 1
        token = self._ik_token
        poses = candidate_poses.to(device=self.env.device)
        self._ik_job = {
            "token": token,
            "poses": poses,
            "keys": candidate_keys,
            "future": self._submit_goalset(poses),
        }

    def get_handle_pose(self):
        """Async handle pose selection via IK goalset.

        - Never blocks the main loop.
        - While computing: sets state="computing", returns None.
        - On success: sets handle_pose/pre_grasp_pose and selected_raw_trajectory.
        - On failure: sets state="failed" and returns None.
        """
        if (
            self.handle_pose is not None
            and self.pre_grasp_pose is not None
            and self.selected_raw_trajectory is not None
        ):
            return self.handle_pose

        self.current_state = "computing"
        self.current_action = None

        # Initialize job if needed (fetch trajectories from env when needed, same as Grasp.get_grasp_pose)
        if self._ik_job is None:
            if not hasattr(self.env, "get_door_trajectories"):
                self.current_state = (
                    "failed: env has no get_door_trajectories (not LocoOpenDoorEnv?)"
                )
                self.current_action = None
                return None
            try:
                self.all_raw_trajectories = self.env.get_door_trajectories(
                    env_id=self.env_id,
                    annotation_name=self._annotation_name,
                    joint_id=self.joint_id,
                )
            except Exception as e:
                self.current_state = f"failed: get_door_trajectories error: {e}"
                self.current_action = None
                return None
            if len(self.all_raw_trajectories) == 0:
                self.current_state = (
                    f"failed: no door trajectories (env_id={self.env_id}, "
                    f"annotation_name={self._annotation_name}, joint_id={self.joint_id})"
                )
                self.current_action = None
                return None

            # Build candidate handle poses from first waypoint of phases[0].
            # Only keep trajectories where every required phase is non-empty.
            # IK goalset requires (G, 7) pose-only — strip hand-joint columns.
            first_phase = self._phases[0]
            candidate_keys = []
            candidate_world_poses = []
            for key, traj in self.all_raw_trajectories.items():
                if isinstance(traj, dict):
                    first_t = traj.get(first_phase)
                    if not (isinstance(first_t, torch.Tensor) and first_t.shape[0] > 0):
                        continue
                    if not all(
                        isinstance(traj.get(p), torch.Tensor)
                        and traj.get(p).shape[0] > 0
                        for p in self._phases
                    ):
                        continue
                    candidate_keys.append(key)
                    candidate_world_poses.append(first_t[0, :7])
                else:
                    if isinstance(traj, torch.Tensor) and traj.shape[0] > 0:
                        candidate_keys.append(key)
                        candidate_world_poses.append(traj[0, :7])

            if len(candidate_world_poses) == 0:
                self.current_state = (
                    f"failed: no trajectories with all required phases non-empty "
                    f"(phases={self._phases})"
                )
                self.current_action = None
                return None
            candidate_world = torch.stack(candidate_world_poses, dim=0)  # [G, 7]

            self._start_ik_job(candidate_world, candidate_keys)
            return None

        job = self._ik_job
        if job.get("token") != self._ik_token:
            self._ik_job = None
            return None

        fut: concurrent.futures.Future | None = job.get("future")
        if fut is None or not fut.done():
            # Keep waiting for IK result; step() will return None while state is "computing"
            self.current_state = "computing"
            return None

        # Consume completed future
        try:
            success_list, goalset_index_list, returned_env_ids = fut.result()
            assert len(returned_env_ids) == 1
            assert returned_env_ids[0] == self.env_id
        except Exception as ex:
            if getattr(self.config, "debug", False):
                import traceback

                print(f"[LocoOpenDoor] IK future exception: {ex}")
                traceback.print_exc()
            self.current_state = f"failed: ik exception {ex}"
            self.current_action = None
            self._ik_job = None
            return None

        selected_idx = -1
        if goalset_index_list is not None and len(goalset_index_list) >= 1:
            selected_idx = int(goalset_index_list[0])

        if selected_idx < 0 or not (len(success_list) >= 1 and bool(success_list[0])):
            if getattr(self.config, "debug", False):
                print(
                    f"[LocoOpenDoor] IK FAILED for all candidates! success={success_list}, goalset_idx={goalset_index_list}"
                )
            self.current_state = "failed: IK failed for all handle candidates"
            self.current_action = None
            self._ik_job = None
            return None

        if selected_idx >= len(job["keys"]):
            self.current_state = "failed: selected_idx out of bounds"
            self.current_action = None
            self._ik_job = None
            return None

        # Use the selected trajectory
        selected_key = job["keys"][selected_idx]
        selected_pose = job["poses"][selected_idx].to(device=self.env.device)

        if getattr(self.config, "debug", False):
            print(
                f"[LocoOpenDoor] IK SUCCESS: selected_idx={selected_idx}, key={selected_key}"
            )
            print(
                f"  handle_pose pos={selected_pose[:3].tolist()}, quat={selected_pose[3:7].tolist()}"
            )

        self.handle_pose = selected_pose
        self.selected_raw_trajectory = self.all_raw_trajectories[selected_key]
        self.selected_trajectory_key = selected_key
        if isinstance(selected_key, int):
            self.selected_joint = "joint_2"
            self.selected_trajectory_id = str(selected_key)
        else:
            parts = str(selected_key).split("/", 1)
            self.selected_joint = parts[0]
            self.selected_trajectory_id = parts[1] if len(parts) > 1 else "0"
        self.pre_grasp_pose = self._compute_pre_grasp_pose(self.handle_pose)
        if getattr(self.config, "debug", False):
            print(f"  pre_grasp_pose pos={self.pre_grasp_pose[:3].tolist()}")

        # Visualize trajectory and key poses for debugging
        self._visualize_poses()

        self.current_state = "ready"
        self.current_action = None
        self._ik_job = None
        return self.handle_pose

    # ------------------------------------------------------------------ #
    # Pose computation helpers
    # ------------------------------------------------------------------ #

    def _compute_pre_grasp_pose(self, handle_pose):
        """Compute pre-grasp pose by moving backward along the grasp approach direction."""
        handle_pos = handle_pose[:3]
        handle_quat = handle_pose[3:7]

        rot_matrix = quat_to_rot_matrix(handle_quat.unsqueeze(0))[0]
        approach_dir = rot_matrix[:, 2]
        approach_dir = approach_dir / torch.norm(approach_dir)

        pre_grasp_pos = handle_pos - approach_dir * self.pre_grasp_offset
        return torch.cat([pre_grasp_pos, handle_quat], dim=0)

    def _next_phase(self, current: str) -> str | None:
        """Return the next phase to execute, or None if done.

        Sequence: pregrasp → move_to_grasp → phases[0] → ... → phases[-1] → None.
        """
        if current == "pregrasp":
            return "move_to_grasp"
        if current == "move_to_grasp":
            return self._phases[0]
        if current in self._phases:
            idx = self._phases.index(current)
            if idx + 1 < len(self._phases):
                return self._phases[idx + 1]
            return None
        return None

    def _get_phase_trajectory(self, phase_name: str):
        """Return world-frame [N,7] trajectory for a given phase, or None."""
        if self.selected_raw_trajectory is None:
            return None
        if isinstance(self.selected_raw_trajectory, dict):
            t = self.selected_raw_trajectory.get(phase_name)
            # Legacy alias: "trajectory" key was used for "pull" in early annotations
            if t is None and phase_name == "pull":
                t = self.selected_raw_trajectory.get("trajectory")
            return t
        # Flat tensor fallback: used for all phases
        return self.selected_raw_trajectory

    def _extract_hand_joints(self, wp_rest: torch.Tensor) -> torch.Tensor:
        """Fit wp_rest (extra joint cols from waypoint) to self._hand_joint_dim.

        - If wp_rest is empty → hand stays open (all zeros, length hand_joint_dim)
        - If wp_rest length == hand_joint_dim → pass through
        - If wp_rest length  > hand_joint_dim → truncate
        - If wp_rest length  < hand_joint_dim → zero-pad at the TAIL

        Annotation stores the 7 right-hand joints (index_0/1, middle_0/1,
        thumb_0/1/2). If the robot's hand action expects more dims (e.g. the
        EEF slice interleaves left+right), tail-padding keeps the annotated
        joints in the leading slots — callers that need a different layout
        should override via config.
        """
        if wp_rest.ndim == 1:
            k = wp_rest.shape[0]
        else:
            k = wp_rest.shape[-1]
        target = self._hand_joint_dim
        if k == target:
            return wp_rest
        if k > target:
            return wp_rest[..., :target]
        pad_shape = list(wp_rest.shape)
        pad_shape[-1] = target - k
        pad = torch.zeros(pad_shape, device=wp_rest.device, dtype=wp_rest.dtype)
        return torch.cat([wp_rest, pad], dim=-1)

    def _to_eef_action(self, wp: torch.Tensor) -> torch.Tensor:
        """Build EEF action (7 pose + hand_joint_dim) from a waypoint row.

        wp is [7] (pose only) or [7+K] (pose + K annotated hand joints).
        """
        wp = wp.view(-1)
        pose7 = wp[:7]
        if wp.shape[0] > 7:
            hand = self._extract_hand_joints(wp[7:])
        else:
            hand = torch.zeros(self._hand_joint_dim, device=wp.device, dtype=wp.dtype)
        return torch.cat([pose7, hand], dim=0)

    def _to_trajectory_eef(self, traj: torch.Tensor) -> torch.Tensor:
        """Build trajectory [N, 7+hand_joint_dim] from [N, 7] or [N, 7+K]."""
        pose = traj[:, :7]
        if traj.shape[1] > 7:
            hand = self._extract_hand_joints(traj[:, 7:])
        else:
            hand = torch.zeros(
                (traj.shape[0], self._hand_joint_dim),
                device=traj.device,
                dtype=traj.dtype,
            )
        return torch.cat([pose, hand], dim=1)

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #

    def _visualize_poses(self):
        """Draw debug points in the viewport after IK selection."""
        if self.selected_raw_trajectory is not None:
            traj_points = []
            # Draw every servo phase, plus the legacy "trajectory" alias if present.
            seen = set()
            viz_keys = list(self._phases) + ["trajectory"]
            for key in viz_keys:
                if key in seen:
                    continue
                seen.add(key)
                traj = (
                    self.selected_raw_trajectory.get(key)
                    if isinstance(self.selected_raw_trajectory, dict)
                    else self.selected_raw_trajectory
                )
                if (
                    traj is not None
                    and isinstance(traj, torch.Tensor)
                    and traj.shape[0] > 0
                ):
                    for i in range(traj.shape[0]):
                        traj_points.append(traj[i, :3].detach().cpu().tolist())
            if traj_points:
                draw_waypoints(
                    traj_points,
                    point_size=8.0,
                    color=(0.0, 1.0, 0.0, 0.8),
                    clear_existing=True,
                )

        if self.handle_pose is not None:
            draw_waypoints(
                [self.handle_pose[:3].detach().cpu().tolist()],
                point_size=15.0,
                color=(0.0, 0.0, 1.0, 1.0),
                clear_existing=False,
            )
        if self.pre_grasp_pose is not None:
            draw_waypoints(
                [self.pre_grasp_pose[:3].detach().cpu().tolist()],
                point_size=15.0,
                color=(0.0, 1.0, 1.0, 1.0),
                clear_existing=False,
            )

    # ------------------------------------------------------------------ #
    # Core interface: reset / step / refresh / update
    # ------------------------------------------------------------------ #

    def reset(self, action: list[Any]):
        # action: [LocoOpenDoor, robot_id, hand_id, obj_type, obj_name, obj_id, optional joint_id]
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]
        self.joint_id = int(action[6]) if len(action) > 6 else -1
        self.current_state = "ready"
        self.current_command = [
            "LocoOpenDoor",
            self.robot_id,
            self.hand_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.joint_id,
        ]
        # pregrasp → move_to_grasp → phases[0] → ... → phases[-1]
        self.current_phase = "pregrasp"
        self.handle_pose = None
        self.pre_grasp_pose = None
        self.selected_raw_trajectory = None
        self.selected_trajectory_key = None
        self.selected_joint = None
        self.selected_trajectory_id = None
        self._ik_token += 1
        self._ik_job = None

        # Resolve IK server by robot_id and hand_id (dual-arm support, same as Grasp)
        self.planner_manager = self._get_planner_manager()
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            raise RuntimeError("IKServer is not available in PlannerManager.")
        robot_name_list = list(ik_dict.keys())
        if self.robot_id < 0 or self.robot_id >= len(robot_name_list):
            raise ValueError(
                f"robot_id {self.robot_id} out of range for robot_name_list={robot_name_list}"
            )
        self.robot_name = robot_name_list[self.robot_id]
        if self.robot_name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{self.robot_name}'.")
        # Single server per robot (MERGE_LEFT_RIGHT §1–§8).
        self.ik_server = ik_dict[self.robot_name]

        # Update obstacles for planners (same as Grasp)
        self.planner_manager.update_obstacles(
            obstacle_avoidance_path_list=["dynamic"],
            env_ids=[self.env_id],
            obstacle_ignore_path_list=[self.obj_name],
        )
        # Kick off async IK goalset selection; trajectories loaded inside get_handle_pose()
        self.get_handle_pose()

    def refresh(self, action: list[Any]):
        # action: [LocoOpenDoor, robot_id, hand_id, obj_type, obj_name, obj_id, optional joint_id]
        new_robot_id = int(action[1])
        new_hand_id = int(action[2])
        new_obj_type = action[3]
        new_obj_name = action[4]
        new_obj_id = action[5]
        new_joint_id = int(action[6]) if len(action) > 6 else -1
        new_command = [
            "LocoOpenDoor",
            new_robot_id,
            new_hand_id,
            new_obj_type,
            new_obj_name,
            new_obj_id,
            new_joint_id,
        ]

        command_changed = (
            self.current_command is None
            or self.current_command[1] != new_robot_id
            or self.current_command[2] != new_hand_id
            or self.current_command[3] != new_obj_type
            or self.current_command[4] != new_obj_name
            or self.current_command[5] != new_obj_id
            or (
                len(self.current_command) > 6
                and self.current_command[6] != new_joint_id
            )
        )

        # If robot or hand changed, re-resolve IKServer before overwriting (same as Grasp)
        if (
            command_changed
            and self.current_command is not None
            and len(self.current_command) >= 3
        ):
            if (
                new_robot_id != self.current_command[1]
                or new_hand_id != self.current_command[2]
            ):
                self.planner_manager = self._get_planner_manager()
                ik_dict = getattr(self.planner_manager, "ik_server", None)
                if ik_dict:
                    robot_name_list = list(ik_dict.keys())
                    if new_robot_id >= 0 and new_robot_id < len(robot_name_list):
                        robot_name = robot_name_list[new_robot_id]
                        if robot_name in ik_dict:
                            # Single server per robot — new_hand_id only
                            # drives downstream target packing.
                            self.robot_name = robot_name
                            self.ik_server = ik_dict[robot_name]

        self.robot_id = new_robot_id
        self.hand_id = new_hand_id
        self.obj_type = new_obj_type
        self.obj_name = new_obj_name
        self.obj_id = new_obj_id
        self.joint_id = new_joint_id
        self.current_command = new_command

        # Same as Grasp: reset phase only if command changed (or phase None); do NOT clear job while computing
        if command_changed or self.current_phase is None:
            self.current_phase = "pregrasp"
            # If still computing IK, do not clear _ik_job — wait for it in step()
            if self.current_state == "computing":
                return
            self.handle_pose = None
            self.pre_grasp_pose = None
            self.selected_raw_trajectory = None
            self.selected_trajectory_key = None
            self.selected_joint = None
            self.selected_trajectory_id = None
            self._ik_token += 1
            self._ik_job = None
            # Reload trajectories from the (possibly new) object
            if hasattr(self.env, "get_door_trajectories"):
                try:
                    self.all_raw_trajectories = self.env.get_door_trajectories(
                        env_id=self.env_id,
                        annotation_name=self._annotation_name,
                        joint_id=self.joint_id,
                    )
                except Exception:
                    self.all_raw_trajectories = {}
            else:
                self.all_raw_trajectories = {}
            self.get_handle_pose()

    _step_count = 0  # debug counter

    def step(self):
        LocoOpenDoor._step_count += 1
        if isinstance(self.current_state, str) and self.current_state.startswith(
            "failed"
        ):
            self.current_action = "Failed"
            return "Failed"

        # Ensure handle pose is computed (async, non-blocking); same logic as Grasp.step()
        if self.handle_pose is None or self.pre_grasp_pose is None:
            self.get_handle_pose()
            if self.current_state == "computing":
                self.current_action = None
                return None
            # If not computing anymore but still missing poses => failed
            if self.handle_pose is None or self.pre_grasp_pose is None:
                if getattr(self.config, "debug", False):
                    print(
                        f"[LocoOpenDoor] step={LocoOpenDoor._step_count}: "
                        f"handle_pose or pre_grasp_pose is None, state={self.current_state}"
                    )
                self.current_state = "failed"
                self.current_action = "Failed"
                return "Failed"

        self.current_state = "running"
        robot_id = self.robot_id
        hand_id = self.hand_id

        if self.current_phase == "pregrasp":
            # Move to pre-grasp pose; dexterous hand open (pose 7 + hand joints zero)
            target_7d = self.pre_grasp_pose.to(device=self.env.device)
            eef_action = self._to_eef_action(target_7d)
            self.current_action = {"MobileMoveL": ((robot_id, hand_id, -1), eef_action)}
            return self.current_action

        if self.current_phase == "move_to_grasp":
            # MoveL to first waypoint of phases[0] trajectory
            first_phase = self._phases[0]
            first_traj = self._get_phase_trajectory(first_phase)
            if first_traj is None or first_traj.shape[0] == 0:
                self.current_state = f"failed: no {first_phase} trajectory"
                self.current_action = "Failed"
                return "Failed"
            first_point = first_traj[0].to(device=self.env.device)
            eef_action = self._to_eef_action(first_point)
            self.current_action = {"MobileMoveL": ((robot_id, hand_id, -1), eef_action)}
            return self.current_action

        if self.current_phase in self._phases:
            # ServoL along the current phase trajectory
            phase_traj = self._get_phase_trajectory(self.current_phase)
            if phase_traj is None or phase_traj.shape[0] == 0:
                self.current_state = f"failed: no {self.current_phase} trajectory"
                self.current_action = "Failed"
                return "Failed"
            traj_eef = self._to_trajectory_eef(phase_traj.to(device=self.env.device))
            self.current_action = {"MobileServoL": ((robot_id, hand_id), traj_eef)}
            return self.current_action

        self.current_state = f"failed: unknown phase {self.current_phase}"
        self.current_action = None
        return None

    def update(self, info):
        result = self._compute_update(info)
        # Always include selected joint/trajectory info for recording
        result["selected_joint"] = self.selected_joint
        result["selected_trajectory_id"] = self.selected_trajectory_id
        return result

    def _compute_update(self, info):
        # During async IK compute, global planner might not exist yet
        if self.current_state == "computing":
            return {
                "atomic_skill_type": "LocoOpenDoor",
                "command": self.current_command,
                "action": None,
                "finished": False,
                "state": "computing",
                "truncated": 0,
                "phase": self.current_phase,
            }

        if isinstance(self.current_state, str) and self.current_state.startswith(
            "failed"
        ):
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "atomic_skill_type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            return {
                "type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state or "ready",
                "truncated": 0,
                "phase": self.current_phase,
            }

        # Check truncated BEFORE finished — the global planner may report
        # both finished=True and truncated>0 when the env terminates mid-phase.
        if gp_info[self.env_id]["truncated"] == 1:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 3:
            if getattr(self.config, "debug", False):
                print(
                    f"[LocoOpenDoor] Env {self.env_id} update: GP truncated=3 (failed to plan), phase={self.current_phase}"
                )
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["finished"]:
            if getattr(self.config, "debug", False):
                print(
                    f"[LocoOpenDoor] Env {self.env_id} update: GP finished, phase={self.current_phase} -> advancing"
                )
            next_phase = self._next_phase(self.current_phase)
            if next_phase is None:
                self.current_state = "finished"
                return {
                    "type": "LocoOpenDoor",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "phase": "completed",
                }
            self.current_phase = next_phase
            return {
                "type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": f"running: {next_phase}",
                "truncated": 0,
                "phase": next_phase,
            }
        else:
            self.current_state = "running"
            return {
                "type": "LocoOpenDoor",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "phase": self.current_phase,
            }
