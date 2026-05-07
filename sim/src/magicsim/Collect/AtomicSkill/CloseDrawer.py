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


class CloseDrawer(AtomicSkill):
    """
    Atomic skill for closing a drawer/articulated object by pushing.

    Uses IK goalset selection to pick the most reachable push pose from all
    available trajectories, then executes:
        1. approach        — MoveL to pre-push pose in front of the push start
        2. push_position   — MoveL to push start pose
        3. close_gripper   — ParallelGripper close (for stable contact)
        4. push            — ServoL to follow the push trajectory
                             (falls back to MoveL if no trajectory configured)
        5. release         — ParallelGripper open
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_target_pose = None
        self.current_phase = None
        self.push_pose = None
        self.pre_push_pose = None
        self.push_trajectory = None  # Cached world-frame trajectory for ServoL

        self.pre_push_offset = getattr(config, "pre_push_offset", 0.15)
        handle_offset_cfg = getattr(config, "handle_offset", [0.0, 0.0, 0.05])
        self.handle_offset = list(handle_offset_cfg)
        self.mobile = bool(getattr(config, "mobile", False))
        self._move_planner_key = "MobileMoveL" if self.mobile else "MoveL"
        self._servo_planner_key = "MobileServoL" if self.mobile else "ServoL"
        # Cached 8D (pose + gripper) push trajectory tensor — built once at
        # the first push step and re-sent on every subsequent tick. See
        # OpenDrawer._pull_traj_8d_cached for the same pattern: identity
        # stability lets MobileServoL.refresh / ServoL.refresh short-circuit
        # instead of re-submitting IK every tick.
        self._push_traj_8d_cached: torch.Tensor | None = None

        # Trajectories are loaded at reset() from the ArticulationObject's
        # own annotations (matching the actual spawned asset).
        self.all_raw_trajectories = {}  # {key: [N, 7] tensor in world frame}
        self.selected_raw_trajectory = None  # The trajectory selected by IK

        # Read annotation name from config
        traj_cfg = getattr(config, "trajectory", None)
        self._annotation_name = "close_by_push_trajectory"
        self._quat_format = "xyzw"
        if traj_cfg is not None:
            self._annotation_name = getattr(
                traj_cfg, "annotation_name", self._annotation_name
            )
            self._quat_format = getattr(traj_cfg, "quat_format", self._quat_format)

        # Async IK pose selection state (following Grasp.py pattern)
        self._ik_job: dict | None = None
        self._ik_token: int = 0
        self.robot_name = None
        self.robot_id = 0
        self.hand_id = 0
        self.ik_server = None
        self.planner_manager: PlannerManager = None

        # Track which joint/trajectory was selected by IK
        self.selected_trajectory_key = None  # e.g. "joint_0/0"
        self.selected_joint = None  # e.g. "joint_0"
        self.selected_trajectory_id = None  # e.g. "0"

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
        """Submit push pose candidates as a goalset to IK server.

        Mobile (``dual_mode=True``) robots use :class:`DualIKPlanRequest`
        with ``lock_base=False`` so the free-base solver searches for a
        feasible base pose alongside the arm IK. Fixed-base robots use
        :class:`IKPlanRequest`. Same dispatch as :meth:`OpenDrawer._submit_goalset`.
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
    # Async push pose selection via IK goalset
    # ------------------------------------------------------------------ #

    def _start_ik_job(self, candidate_poses: torch.Tensor, candidate_keys: list[str]):
        """Start async IK goalset job for push pose candidates."""
        self._ik_token += 1
        token = self._ik_token
        poses = candidate_poses.to(device=self.env.device)
        self._ik_job = {
            "token": token,
            "poses": poses,
            "keys": candidate_keys,
            "future": self._submit_goalset(poses),
        }

    def get_push_pose(self):
        """Async push pose selection via IK goalset.

        - Never blocks the main loop.
        - While computing: sets state="computing", returns None.
        - On success: sets push_pose/pre_push_pose and selected_raw_trajectory.
        - On failure: sets state="failed" and returns None.
        """
        if (
            self.push_pose is not None
            and self.pre_push_pose is not None
            and self.selected_raw_trajectory is not None
        ):
            return self.push_pose

        self.current_state = "computing"
        self.current_action = None

        # Initialize job if needed
        if self._ik_job is None:
            if len(self.all_raw_trajectories) == 0:
                # No trajectories available, use fallback
                self._compute_fallback_push_pose()
                return self.push_pose

            # Build candidate push poses from first waypoint of each trajectory
            # (trajectories are already in world frame from get_trajectory_poses)
            candidate_keys = []
            candidate_world_poses = []
            for key, traj in self.all_raw_trajectories.items():
                candidate_keys.append(key)
                candidate_world_poses.append(traj[0])  # first waypoint [7]

            candidate_world = torch.stack(candidate_world_poses, dim=0)  # [G, 7]

            self._start_ik_job(candidate_world, candidate_keys)
            return None

        job = self._ik_job
        if job.get("token") != self._ik_token:
            self._ik_job = None
            return None

        fut: concurrent.futures.Future | None = job.get("future")
        if fut is None or not fut.done():
            return None

        # Consume completed future
        try:
            success_list, goalset_index_list, returned_env_ids = fut.result()
            assert len(returned_env_ids) == 1
            assert returned_env_ids[0] == self.env_id
        except Exception as ex:
            self.current_state = f"failed: ik exception {ex}"
            self.current_action = None
            self._ik_job = None
            return None

        selected_idx = -1
        if goalset_index_list is not None and len(goalset_index_list) >= 1:
            selected_idx = int(goalset_index_list[0])

        if selected_idx < 0 or not (len(success_list) >= 1 and bool(success_list[0])):
            # IK failed for all candidates, use fallback
            if getattr(self.config, "debug", False):
                print(
                    f"[CloseDrawer] IK FAILED for all candidates! success={success_list}, goalset_idx={goalset_index_list}. Using fallback."
                )
            self._ik_job = None
            self._compute_fallback_push_pose()
            if getattr(self.config, "debug", False):
                print(f"[CloseDrawer] Fallback push_pose={self.push_pose[:3].tolist()}")
            return self.push_pose

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
                f"[CloseDrawer] IK SUCCESS: selected_idx={selected_idx}, key={selected_key}"
            )
            print(
                f"  push_pose pos={selected_pose[:3].tolist()}, quat={selected_pose[3:7].tolist()}"
            )

        self.push_pose = selected_pose
        self.selected_raw_trajectory = self.all_raw_trajectories[selected_key]
        self.selected_trajectory_key = selected_key
        parts = selected_key.split("/", 1)
        self.selected_joint = parts[0]
        self.selected_trajectory_id = parts[1] if len(parts) > 1 else "0"
        self.pre_push_pose = self._compute_pre_push_pose(self.push_pose)
        if getattr(self.config, "debug", False):
            print(f"  pre_push_pose pos={self.pre_push_pose[:3].tolist()}")

        # Visualize trajectory and key poses for debugging
        self._visualize_poses()

        self.current_state = "ready"
        self.current_action = None
        self._ik_job = None
        return self.push_pose

    def _compute_fallback_push_pose(self):
        """Fallback: use object root pose + offset when no trajectory or IK fails."""
        device = self.env.device
        pos, quat, scale = self.env.get_drawer_object_pose(self.env_id)
        pos = pos.to(device=device)
        quat = quat.to(device=device)

        rot_matrix = quat_to_rot_matrix(quat.unsqueeze(0))[0]
        offset_local = torch.tensor(
            self.handle_offset, device=device, dtype=torch.float32
        )
        offset_world = rot_matrix @ offset_local

        push_pos = pos + offset_world
        self.push_pose = torch.cat([push_pos, quat], dim=0)
        self.pre_push_pose = self._compute_pre_push_pose(self.push_pose)
        self.selected_raw_trajectory = None
        self.current_state = "ready"

    # ------------------------------------------------------------------ #
    # Pose computation helpers
    # ------------------------------------------------------------------ #

    def _compute_pre_push_pose(self, push_pose):
        """Compute pre-push pose by moving backward along the approach direction."""
        push_pos = push_pose[:3]
        push_quat = push_pose[3:7]

        rot_matrix = quat_to_rot_matrix(push_quat.unsqueeze(0))[0]
        approach_dir = rot_matrix[:, 2]
        approach_dir = approach_dir / torch.norm(approach_dir)

        pre_push_pos = push_pos - approach_dir * self.pre_push_offset
        return torch.cat([pre_push_pos, push_quat], dim=0)

    def _compute_push_trajectory(self):
        """Return the selected trajectory (already in world frame)."""
        return self.selected_raw_trajectory

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #

    def _visualize_poses(self):
        """Draw debug points in the viewport after IK selection."""
        if self.selected_raw_trajectory is not None:
            traj = self._compute_push_trajectory()
            traj_points = []
            for i in range(traj.shape[0]):
                pos = traj[i, :3].detach().cpu().tolist()
                traj_points.append(pos)
            draw_waypoints(
                traj_points,
                point_size=8.0,
                color=(0.0, 1.0, 0.0, 0.8),
                clear_existing=True,
            )

        if self.push_pose is not None:
            draw_waypoints(
                [self.push_pose[:3].detach().cpu().tolist()],
                point_size=15.0,
                color=(0.0, 0.0, 1.0, 1.0),
                clear_existing=False,
            )
        if self.pre_push_pose is not None:
            draw_waypoints(
                [self.pre_push_pose[:3].detach().cpu().tolist()],
                point_size=15.0,
                color=(0.0, 1.0, 1.0, 1.0),
                clear_existing=False,
            )

    # ------------------------------------------------------------------ #
    # Core interface: reset / step / refresh / update
    # ------------------------------------------------------------------ #

    def reset(self, action: list[Any]):
        # action: [skill_name, robot_id, hand_id, obj_type, obj_name, obj_id, joint_id]
        if len(action) < 7:
            raise ValueError(
                f"CloseDrawer.reset expects action "
                f"[skill_name, robot_id, hand_id, obj_type, obj_name, obj_id, joint_id], "
                f"got length {len(action)}: {action}"
            )
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        if self.hand_id not in (0, 1, -1):
            raise ValueError(
                f"CloseDrawer.reset: hand_id must be 0 (right), 1 (left), or -1 (both); "
                f"got {self.hand_id}"
            )
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]
        self.joint_id = int(action[6])
        robot_name_list = list(self.env.scene.robot_manager.robots.keys())
        if 0 <= self.robot_id < len(robot_name_list):
            self.robot_name = robot_name_list[self.robot_id]
        else:
            self.robot_name = robot_name_list[0] if robot_name_list else None
        self.current_state = "ready"
        self.current_command = [
            "CloseDrawer",
            self.robot_id,
            self.hand_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.joint_id,
        ]
        self.current_phase = "approach"
        self.push_trajectory = None
        self._push_traj_8d_cached = None
        self.push_pose = None
        self.pre_push_pose = None
        self.selected_raw_trajectory = None
        self.selected_trajectory_key = None
        self.selected_joint = None
        self.selected_trajectory_id = None
        self._ik_token += 1
        self._ik_job = None

        # Open the drawer joints so we can push them closed
        self.env.set_drawer_open(self.env_id, self._annotation_name)

        # Load trajectories from the actual spawned object's annotations
        self.all_raw_trajectories = self.env.get_drawer_trajectories(
            self.env_id, self._annotation_name, joint_id=self.joint_id
        )

        # Resolve IK server (support hand_id for dual-arm)
        self.planner_manager = self._get_planner_manager()
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            raise RuntimeError("IKServer is not available in PlannerManager.")
        if self.robot_name is None:
            self.robot_name = next(iter(ik_dict.keys()))
        if self.robot_name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{self.robot_name}'.")
        # Single server per robot (MERGE_LEFT_RIGHT §1–§8).
        self.ik_server = ik_dict[self.robot_name]

        # Update obstacles for planners (keeps world configs fresh)
        self.planner_manager.update_obstacles(
            obstacle_avoidance_path_list=["dynamic"],
            env_ids=[self.env_id],
            obstacle_ignore_path_list=[self.obj_name],
        )
        # Kick off async IK goalset selection
        self.get_push_pose()

    def refresh(self, action: list[Any]):
        # action: [skill_name, robot_id, hand_id, obj_type, obj_name, obj_id, joint_id]
        if len(action) < 7:
            raise ValueError(
                f"CloseDrawer.refresh expects action "
                f"[skill_name, robot_id, hand_id, obj_type, obj_name, obj_id, joint_id], "
                f"got length {len(action)}: {action}"
            )
        new_robot_id = int(action[1])
        new_hand_id = int(action[2])
        if new_hand_id not in (0, 1, -1):
            raise ValueError(
                f"CloseDrawer.refresh: hand_id must be 0 (right), 1 (left), or -1 (both); "
                f"got {new_hand_id}"
            )
        new_obj_type = action[3]
        new_obj_name = action[4]
        new_obj_id = action[5]
        new_joint_id = int(action[6])
        new_command = [
            "CloseDrawer",
            new_robot_id,
            new_hand_id,
            new_obj_type,
            new_obj_name,
            new_obj_id,
            new_joint_id,
        ]

        old_joint_id = (
            int(self.current_command[6])
            if self.current_command and len(self.current_command) > 6
            else -1
        )
        command_changed = (
            self.current_command is None
            or self.current_command[1] != new_robot_id
            or self.current_command[2] != new_hand_id
            or self.current_command[3] != new_obj_type
            or self.current_command[4] != new_obj_name
            or self.current_command[5] != new_obj_id
            or old_joint_id != new_joint_id
        )

        self.robot_id = new_robot_id
        self.hand_id = new_hand_id
        self.obj_type = new_obj_type
        self.obj_name = new_obj_name
        self.obj_id = new_obj_id
        self.joint_id = new_joint_id
        self.current_command = new_command
        if 0 <= self.robot_id < len(self.env.scene.robot_manager.robots):
            robot_name_list = list(self.env.scene.robot_manager.robots.keys())
            self.robot_name = robot_name_list[self.robot_id]

        if command_changed or self.current_phase is None:
            self.current_phase = "approach"
            self.push_trajectory = None
            self._push_traj_8d_cached = None
            self.push_pose = None
            self.pre_push_pose = None
            self.selected_raw_trajectory = None
            self.selected_trajectory_key = None
            self.selected_joint = None
            self.selected_trajectory_id = None
            self._ik_token += 1
            self._ik_job = None
            # Reload trajectories from the (possibly new) object
            self.all_raw_trajectories = self.env.get_drawer_trajectories(
                self.env_id, self._annotation_name, joint_id=self.joint_id
            )
            self.get_push_pose()

    _step_count = 0  # debug counter

    def step(self):
        CloseDrawer._step_count += 1
        if self.current_state == "failed":
            self.current_action = "Failed"
            return "Failed"

        # Ensure push pose is computed (async, non-blocking)
        if self.push_pose is None or self.pre_push_pose is None:
            self.get_push_pose()
            if self.current_state == "computing":
                self.current_action = None
                return None
            if self.push_pose is None or self.pre_push_pose is None:
                if getattr(self.config, "debug", False):
                    print(
                        f"[CloseDrawer] step={CloseDrawer._step_count}: push_pose or pre_push_pose is None, FAILED"
                    )
                self.current_state = "failed"
                self.current_action = "Failed"
                return "Failed"

        self.current_state = "running"
        robot_id = self.robot_id
        hand_id = self.hand_id
        move_key = self._move_planner_key
        servo_key = self._servo_planner_key

        if self.current_phase == "approach":
            # Approach pre-push with gripper open
            target_7d = self.pre_push_pose.to(device=self.env.device)
            target_8d = torch.cat(
                [
                    target_7d,
                    torch.tensor([0.0], device=self.env.device, dtype=torch.float32),
                ],
                dim=0,
            )
            self.current_action = {move_key: ((robot_id, hand_id, -1), target_8d)}
            return self.current_action

        elif self.current_phase == "push_position":
            # Move to push start position with gripper open
            target_7d = self.push_pose.to(device=self.env.device)
            target_8d = torch.cat(
                [
                    target_7d,
                    torch.tensor([0.0], device=self.env.device, dtype=torch.float32),
                ],
                dim=0,
            )
            self.current_action = {move_key: ((robot_id, hand_id, -1), target_8d)}
            return self.current_action

        elif self.current_phase == "close_gripper":
            gripper_target = torch.tensor(
                [1.0], device=self.env.device, dtype=torch.float32
            )
            self.current_action = {
                "ParallelGripper": ((robot_id, hand_id, 0), gripper_target)
            }
            return self.current_action

        elif self.current_phase == "push":
            if self.selected_raw_trajectory is not None:
                # Build 8D push trajectory once and cache; identity stability
                # lets ServoL/MobileServoL.refresh short-circuit instead of
                # re-submitting IK every tick.
                if self._push_traj_8d_cached is None:
                    if self.push_trajectory is None:
                        self.push_trajectory = self._compute_push_trajectory()
                    traj = self.push_trajectory.to(device=self.env.device)
                    gripper_col = torch.ones(
                        (traj.shape[0], 1), device=traj.device, dtype=traj.dtype
                    )
                    self._push_traj_8d_cached = torch.cat(
                        [traj, gripper_col], dim=1
                    )  # [N, 8]
                self.current_action = {
                    servo_key: ((robot_id, hand_id, 0), self._push_traj_8d_cached)
                }
            else:
                # Fallback: no trajectory, just stay at push pose
                target_7d = self.push_pose.to(device=self.env.device)
                target_8d = torch.cat(
                    [
                        target_7d,
                        torch.tensor(
                            [1.0], device=self.env.device, dtype=torch.float32
                        ),
                    ],
                    dim=0,
                )
                self.current_action = {move_key: ((robot_id, hand_id, 0), target_8d)}
            return self.current_action

        elif self.current_phase == "release":
            gripper_target = torch.tensor(
                [0.0], device=self.env.device, dtype=torch.float32
            )
            self.current_action = {
                "ParallelGripper": ((robot_id, hand_id, 0), gripper_target)
            }
            return self.current_action

        else:
            self.current_state = "failed"
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
                "atomic_skill_type": "CloseDrawer",
                "command": self.current_command,
                "action": None,
                "finished": False,
                "state": "computing",
                "truncated": 0,
                "phase": self.current_phase,
            }

        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "atomic_skill_type": "CloseDrawer",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            return {
                "type": "CloseDrawer",
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
                "type": "CloseDrawer",
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
                "type": "CloseDrawer",
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
                    f"[CloseDrawer] Env {self.env_id} update: GP truncated=3 (failed to plan), phase={self.current_phase}"
                )
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "CloseDrawer",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["finished"]:
            if self.current_phase == "approach":
                self.current_phase = "push_position"
                print(f"[CloseDrawer] env_id={self.env_id} phase=push_position")
                return {
                    "type": "CloseDrawer",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: moving to push position",
                    "truncated": 0,
                    "phase": "push_position",
                }
            elif self.current_phase == "push_position":
                self.current_phase = "close_gripper"
                print(f"[CloseDrawer] env_id={self.env_id} phase=close_gripper")
                return {
                    "type": "CloseDrawer",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: closing gripper",
                    "truncated": 0,
                    "phase": "close_gripper",
                }
            elif self.current_phase == "close_gripper":
                self.current_phase = "push"
                print(f"[CloseDrawer] env_id={self.env_id} phase=push")
                return {
                    "type": "CloseDrawer",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: pushing drawer",
                    "truncated": 0,
                    "phase": "push",
                }
            elif self.current_phase == "push":
                self.current_phase = "release"
                print(f"[CloseDrawer] env_id={self.env_id} phase=release")
                return {
                    "type": "CloseDrawer",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: releasing gripper",
                    "truncated": 0,
                    "phase": "release",
                }
            elif self.current_phase == "release":
                self.current_state = "finished"
                print(f"[CloseDrawer] env_id={self.env_id} phase=completed")
                return {
                    "type": "CloseDrawer",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "phase": "completed",
                }
        else:
            self.current_state = "running"
            return {
                "type": "CloseDrawer",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "phase": self.current_phase,
            }
