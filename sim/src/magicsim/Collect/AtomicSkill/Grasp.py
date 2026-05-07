from typing import Any
import torch
from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Planner.Services.IKServer import IKPlanRequest
from magicsim.Env.Planner.Services.DualIKServer import DualIKPlanRequest
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig
from magicsim.Env.Utils.rotations import quat_to_rot_matrix
import concurrent.futures
from magicsim.Env.Planner.PlannerManager import PlannerManager
from magicsim.Task.LocoManip.Env.Test.TestLocoGraspEnv import visualize_grasp_pose
from loguru import logger as log


class Grasp(AtomicSkill):
    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_target_pose = None
        self.current_phase = None  # "pre_grasp", "grasp", "close_gripper", "retrieval"
        self.grasp_pose = None
        self.pre_grasp_pose = None  # Pre-computed pre-grasp pose
        self.retrieval_grasp_pose = None  # Pre-computed retrieval pose
        self.grasp_pose_type = None  # Track which type of grasp pose is used: "functional_grasp" or "grasp"
        self.functional_grasp = False  # Whether to use functional grasp (from action)
        self.part = None  # Part name for grasp (e.g., "handle", "body", "head")
        # Read parameters from config
        self.pre_grasp_offset = getattr(
            config, "pre_grasp_offset", 0.15
        )  # Distance to move backward for pre-grasp
        self.retrieval_offset = getattr(
            config, "retrieval_offset", 0.3
        )  # Distance to move upward for retrieval
        self.update_period = getattr(config, "update_period", 30)
        self.update_count = 0
        self.mobile = getattr(config, "mobile", False)
        self._move_planner_key = "MobileMoveL" if self.mobile else "MoveL"
        # Async grasp pose computation state
        self._grasp_job: dict | None = None
        self._grasp_token: int = 0
        self.robot_id = 0
        self.hand_id = 0
        self.robot_name = None
        self.ik_server = None
        self.planner_manager: PlannerManager = None
        self.debug = getattr(config, "debug", False)
        self.viz_grasp = getattr(config, "viz_grasp", True)
        self.reactive = getattr(config, "reactive", False)
        self._last_viz_phase = None
        self.grasp_pose_id = -1
        self._grasp_pose_updated = False

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
        robot_state = self._get_robot_state()
        robot_states_dict = {
            "base_pos": robot_state["base_pos"],
            "base_quat": robot_state["base_quat"],
            "joint_pos": robot_state["joint_pos"],
            "joint_vel": robot_state["joint_vel"],
        }
        # Pack single-arm goalset into (1, G, eef_num * 7). NaN rows mark
        # the inactive arm for this solve — the IK server's NaN-as-disable
        # preprocessing flips that arm's ToolPoseCriteria to disabled().
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

    def reset(self, action: list[Any]):
        # action: [skill_name, robot_id, hand_id, obj_type, obj_name, obj_id, optional functional_grasp, optional part]
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]
        if len(action) >= 8:
            self.functional_grasp = action[6]
            self.part = action[7]
        else:
            self.functional_grasp = None
            self.part = None
        self.current_state = "ready"
        self.current_command = [
            "Grasp",
            self.robot_id,
            self.hand_id,
            self.obj_type,
            self.obj_name,
            self.obj_id,
            self.functional_grasp,
            self.part,
        ]
        self.current_phase = "pre_grasp"

        # Reset pre-computed poses
        self.grasp_pose = None
        self.pre_grasp_pose = None
        self.retrieval_grasp_pose = None
        self.grasp_pose_type = None
        self._last_viz_phase = None
        self.grasp_pose_id = -1
        self._grasp_pose_updated = False
        # Cancel any previous async job by bumping token and clearing job
        self._grasp_token += 1
        self._grasp_job = None

        # Resolve IKServer for the current robot and hand
        self.planner_manager = self._get_planner_manager()
        ik_dict = getattr(self.planner_manager, "ik_server", None)
        if not ik_dict:
            raise RuntimeError("IKServer is not available in PlannerManager.")
        if self.robot_name is None:
            self.robot_name = next(iter(ik_dict.keys()))
        if self.robot_name not in ik_dict:
            raise RuntimeError(f"IKServer not found for robot '{self.robot_name}'.")
        # Single server per robot (MERGE_LEFT_RIGHT §1–§8). ``self.hand_id``
        # still rides in downstream GlobalPlanner action headers for
        # target packing (NaN for inactive arm).
        self.ik_server = ik_dict[self.robot_name]
        # Update obstacles for planners (keeps world configs fresh)
        if self.obj_name is not None:
            self.planner_manager.update_obstacles(
                obstacle_avoidance_path_list=["dynamic"],
                env_ids=[self.env_id],
                obstacle_ignore_path_list=[self.obj_name],
            )
        # Kick off async computation; will be polled from step()
        self.get_grasp_pose()

    def _combine_poses_from_dict(self, parts_dict: dict) -> torch.Tensor | None:
        """
        Combine poses from all parts in a dictionary.

        Args:
            parts_dict: Dictionary mapping part names to pose tensors

        Returns:
            Combined tensor of all poses [N, 7], or None if no valid poses found
        """
        if not parts_dict or not isinstance(parts_dict, dict):
            return None

        tensor_list = []
        for part_name, poses in parts_dict.items():
            if (
                poses is not None
                and isinstance(poses, torch.Tensor)
                and poses.numel() > 0
            ):
                # Ensure 2D [N, 7]
                if poses.ndim == 1:
                    poses = poses.unsqueeze(0)
                if poses.shape[-1] == 7:
                    tensor_list.append(poses)

        if not tensor_list:
            return None

        return torch.cat(tensor_list, dim=0)

    def _build_pose_tensor_from_grasp_dict(
        self, all_grasp_poses: dict
    ) -> tuple[torch.Tensor | None, str | None]:
        """
        Apply the functional_grasp/part selection logic to an env.get_grasp_pose()
        dict and return (flat_pose_tensor [N, 7], pose_type). Stable ordering so
        a previously selected index remains valid across calls.
        """
        if all_grasp_poses is None:
            return None, None
        functional_grasp_dict = all_grasp_poses.get("functional_grasp", {})
        grasp_dict = all_grasp_poses.get("grasp", {})
        grasp_pose_tensor = None
        pose_type = None

        if self.functional_grasp is True:
            if (
                self.part
                and functional_grasp_dict
                and self.part in functional_grasp_dict
                and functional_grasp_dict[self.part] is not None
            ):
                part_poses = functional_grasp_dict[self.part]
                if isinstance(part_poses, torch.Tensor) and part_poses.numel() > 0:
                    grasp_pose_tensor = part_poses
                    pose_type = "functional_grasp"
            if grasp_pose_tensor is None and functional_grasp_dict:
                grasp_pose_tensor = self._combine_poses_from_dict(functional_grasp_dict)
                if grasp_pose_tensor is not None:
                    pose_type = "functional_grasp"
            if grasp_pose_tensor is None and grasp_dict:
                grasp_pose_tensor = self._combine_poses_from_dict(grasp_dict)
                if grasp_pose_tensor is not None:
                    pose_type = "grasp"
        else:
            if self.part is not None:
                if (
                    functional_grasp_dict
                    and self.part in functional_grasp_dict
                    and functional_grasp_dict[self.part] is not None
                ):
                    part_poses = functional_grasp_dict[self.part]
                    if isinstance(part_poses, torch.Tensor) and part_poses.numel() > 0:
                        grasp_pose_tensor = part_poses
                        pose_type = "functional_grasp"
                if grasp_pose_tensor is None:
                    if (
                        grasp_dict
                        and self.part in grasp_dict
                        and grasp_dict[self.part] is not None
                    ):
                        part_poses = grasp_dict[self.part]
                        if (
                            isinstance(part_poses, torch.Tensor)
                            and part_poses.numel() > 0
                        ):
                            grasp_pose_tensor = part_poses
                            pose_type = "grasp"
            if grasp_pose_tensor is None:
                all_parts_dict = {}
                if functional_grasp_dict:
                    all_parts_dict.update(functional_grasp_dict)
                if grasp_dict:
                    all_parts_dict.update(grasp_dict)
                grasp_pose_tensor = self._combine_poses_from_dict(all_parts_dict)
                if grasp_pose_tensor is not None:
                    pose_type = "grasp"

        if grasp_pose_tensor is not None and grasp_pose_tensor.ndim == 1:
            grasp_pose_tensor = grasp_pose_tensor.unsqueeze(0)
        return grasp_pose_tensor, pose_type

    def _update_grasp_from_object_pose(self) -> bool:
        """
        Refetch grasp poses (env returns world-frame using current object pose)
        and reuse the stored selected index to refresh grasp / pre_grasp / retrieval.
        Used when reactive=True during the grasp phase.
        """
        if self.grasp_pose_id < 0 or not hasattr(self.env, "get_grasp_pose"):
            return False
        all_grasp_poses = self.env.get_grasp_pose(
            env_ids=[self.env_id], obj_name=self.obj_name
        )[0]
        if all_grasp_poses is None:
            return False
        pose_tensor, _ = self._build_pose_tensor_from_grasp_dict(all_grasp_poses)
        if pose_tensor is None or self.grasp_pose_id >= pose_tensor.shape[0]:
            return False
        grasp_pose = pose_tensor[self.grasp_pose_id].to(device=self.env.device)
        self.grasp_pose = grasp_pose
        self.pre_grasp_pose = self._compute_pose_along_grasp_direction(
            grasp_pose, self.pre_grasp_offset, backward=True
        )
        self.retrieval_grasp_pose = self._compute_pose_upward(
            grasp_pose, self.retrieval_offset
        )
        return True

    def _start_job(self, pose_type: str, grasp_pose_tensor: torch.Tensor):
        self._grasp_token += 1
        token = self._grasp_token
        grasp_pose_list = grasp_pose_tensor.to(device=self.env.device)
        if grasp_pose_list.ndim != 2 or grasp_pose_list.shape[-1] != 7:
            raise ValueError(
                f"Expected grasp_pose_list [G,7], got {tuple(grasp_pose_list.shape)}"
            )
        self._grasp_job = {
            "token": token,
            "pose_type": pose_type,  # "functional_grasp" or "grasp"
            "pose_list": grasp_pose_list,
            "future": None,
        }
        # Submit goalset selection
        self._grasp_job["future"] = self._submit_goalset(grasp_pose_list)

    def get_grasp_pose(self):
        """
        Async grasp pose selection.

        Logic:
        If self.functional_grasp is True:
            1. If self.part provided and found, return this part's poses
            2. Else if self.part doesn't exist or is None, combine all parts' poses in functional_grasp
            3. If no parts in functional_grasp, combine all parts' poses in grasp

        If self.functional_grasp is False or None:
            1. If self.part is not None, return this part's poses (from either functional_grasp or grasp)
            2. If self.part is None, return all parts in both functional_grasp and grasp keys

        - Never blocks the main loop.
        - While computing: sets state="computing", action=None, returns None.
        - On success: sets grasp_pose/pre_grasp_pose/retrieval_grasp_pose and returns grasp_pose.
        - On final failure: sets state="failed" and returns None.
        """
        # If already have a valid grasp pose, return it.
        if (
            self.grasp_pose is not None
            and self.pre_grasp_pose is not None
            and self.retrieval_grasp_pose is not None
        ):
            return self.grasp_pose

        self.current_state = "computing"
        self.current_action = None

        # Initialize job if needed
        if self._grasp_job is None:
            # Get all grasp poses from environment
            if hasattr(self.env, "get_grasp_pose"):
                all_grasp_poses = self.env.get_grasp_pose(
                    env_ids=[self.env_id], obj_name=self.obj_name
                )[0]
            else:
                self.current_state = "failed: no grasp pose method available"
                self.current_action = None
                self._grasp_job = None
                return None

            if all_grasp_poses is None:
                log.error(
                    "[Grasp] env.get_grasp_pose returned None for obj_name={}",
                    self.obj_name,
                )
                self.current_state = "failed: no grasp annotation"
                self.current_action = None
                self._grasp_job = None
                return None

            grasp_pose_tensor, pose_type = self._build_pose_tensor_from_grasp_dict(
                all_grasp_poses
            )
            if pose_type is not None:
                self.grasp_pose_type = pose_type

            # If still no poses found, fail
            if grasp_pose_tensor is None:
                self.current_state = "failed: no grasp poses available"
                self.current_action = None
                self._grasp_job = None
                return None

            # Ensure tensor is 2D [N, 7]
            if grasp_pose_tensor.ndim == 1:
                grasp_pose_tensor = grasp_pose_tensor.unsqueeze(0)

            self.grasp_pose_tensor = grasp_pose_tensor
            self.grasp_pose_list = grasp_pose_tensor.to(device=self.env.device)
            self._start_job(self.grasp_pose_type, grasp_pose_tensor)
            return None

        job = self._grasp_job

        # If job token mismatches, treat as stale and restart
        if job.get("token") != self._grasp_token:
            self._grasp_job = None
            return None

        fut: concurrent.futures.Future | None = job.get("future")
        if fut is None or not fut.done():
            return None

        # Consume completed future
        try:
            success_list, goalset_index_list, returned_env_ids = fut.result()
            assert len(returned_env_ids) == 1, (
                f"Expected 1 env_id, got {len(returned_env_ids)}, returned_env_ids: {returned_env_ids}"
            )
            assert returned_env_ids[0] == self.env_id, (
                f"Expected env_id {self.env_id}, got {returned_env_ids[0]}, returned_env_ids: {returned_env_ids}"
            )
        except Exception as ex:
            # Fail fast on solver errors
            log.error("[Grasp] IK exception: {}", ex)
            self.current_state = f"failed: ik exception {ex}"
            self.current_action = None
            self._grasp_job = None
            return None

        # Ensure the result corresponds to this env
        if not returned_env_ids or int(returned_env_ids[0]) != int(self.env_id):
            self.current_state = "failed: ik env mismatch"
            self.current_action = None
            self._grasp_job = None
            return None

        # Goalset selection: get the selected index
        selected_idx = -1
        if goalset_index_list is not None and len(goalset_index_list) >= 1:
            selected_idx = int(goalset_index_list[0])

        # If failed or no valid goal:
        if selected_idx < 0 or not (len(success_list) >= 1 and bool(success_list[0])):
            # Fallback logic: if functional_grasp failed, try other options
            if job["pose_type"] == "functional_grasp":
                # Try to get grasp poses again with fallback logic
                if hasattr(self.env, "get_grasp_pose"):
                    all_grasp_poses = self.env.get_grasp_pose(
                        env_ids=[self.env_id], obj_name=self.obj_name
                    )[0]
                else:
                    all_grasp_poses = None

                if all_grasp_poses is None:
                    self.current_state = "failed"
                    self.current_action = None
                    self._grasp_job = None
                    return None

                functional_grasp_dict = all_grasp_poses.get("functional_grasp", {})
                grasp_dict = all_grasp_poses.get("grasp", {})
                grasp_pose_tensor = None

                # If we were trying a specific part, try other parts in functional_grasp
                if self.part and functional_grasp_dict:
                    # Try combining all other parts in functional_grasp (excluding the failed part)
                    other_parts = {
                        k: v for k, v in functional_grasp_dict.items() if k != self.part
                    }
                    grasp_pose_tensor = self._combine_poses_from_dict(other_parts)
                    if grasp_pose_tensor is not None:
                        self.grasp_pose_type = "functional_grasp"

                # If still no poses, try grasp dict
                if grasp_pose_tensor is None and grasp_dict:
                    if self.part and self.part in grasp_dict:
                        part_poses = grasp_dict[self.part]
                        if (
                            isinstance(part_poses, torch.Tensor)
                            and part_poses.numel() > 0
                        ):
                            grasp_pose_tensor = part_poses
                            self.grasp_pose_type = "grasp"
                    else:
                        # Combine all parts in grasp
                        grasp_pose_tensor = self._combine_poses_from_dict(grasp_dict)
                        if grasp_pose_tensor is not None:
                            self.grasp_pose_type = "grasp"

                if grasp_pose_tensor is not None:
                    if grasp_pose_tensor.ndim == 1:
                        grasp_pose_tensor = grasp_pose_tensor.unsqueeze(0)
                    self.grasp_pose_tensor = grasp_pose_tensor
                    self.grasp_pose_list = grasp_pose_tensor.to(device=self.env.device)
                    self._start_job(self.grasp_pose_type, grasp_pose_tensor)
                    return None

            # No fallback available or fallback also failed
            log.error(
                "[Grasp] IK goalset failed and no fallback available. "
                "pose_type={} selected_idx={} success_list={}",
                job.get("pose_type"),
                selected_idx,
                success_list if "success_list" in dir() else "N/A",
            )
            self.current_state = "failed"
            self.current_action = None
            self._grasp_job = None
            return None

        # Validate selected index
        if selected_idx >= job["pose_list"].shape[0]:
            self.current_state = "failed: selected_idx out of bounds"
            self.current_action = None
            self._grasp_job = None
            return None

        # Get the selected grasp pose
        grasp_pose = job["pose_list"][selected_idx].to(device=self.env.device)
        self.grasp_pose = grasp_pose
        self.grasp_pose_id = selected_idx

        # Compute pre-grasp and retrieval poses directly
        self.pre_grasp_pose = self._compute_pose_along_grasp_direction(
            grasp_pose, self.pre_grasp_offset, backward=True
        )
        self.retrieval_grasp_pose = self._compute_pose_upward(
            grasp_pose, self.retrieval_offset
        )

        # Success: commit poses and clear job
        self.current_state = "ready"
        self.current_action = None
        self._grasp_job = None
        return self.grasp_pose

    def refresh(self, action: list[Any]):
        # action: [Grasp, robot_id, hand_id, obj_type, obj_name, obj_id, optional functional_grasp, optional part]
        new_robot_id = int(action[1])
        new_hand_id = int(action[2])
        new_obj_type = action[3]
        new_obj_name = action[4]
        new_obj_id = action[5]
        if len(action) >= 8:
            new_functional_grasp = action[6]
            new_part = action[7]
        else:
            new_functional_grasp = None
            new_part = None
        new_command = [
            "Grasp",
            new_robot_id,
            new_hand_id,
            new_obj_type,
            new_obj_name,
            new_obj_id,
            new_functional_grasp,
            new_part,
        ]

        # Check if command changed (robot_id, hand_id, obj, functional_grasp, part)
        command_changed = (
            self.current_command is None
            or self.current_command[1] != new_robot_id
            or self.current_command[2] != new_hand_id
            or self.current_command[3] != new_obj_type
            or self.current_command[4] != new_obj_name
            or self.current_command[5] != new_obj_id
            or (
                len(self.current_command) >= 7
                and self.current_command[6] != new_functional_grasp
            )
            or (len(self.current_command) >= 8 and self.current_command[7] != new_part)
        )

        # If robot or hand changed, re-resolve IKServer before overwriting
        if (
            command_changed
            and self.current_command is not None
            and len(self.current_command) >= 3
        ):
            if (
                new_robot_id != self.current_command[1]
                or new_hand_id != self.current_command[2]
            ):
                print("***************** obstacle_ignore_path_list: ", self.obj_name)
                self.planner_manager = self._get_planner_manager()
                self.planner_manager.update_obstacles(
                    obstacle_avoidance_path_list=["dynamic"],
                    env_ids=[self.env_id],
                    obstacle_ignore_path_list=[self.obj_name],
                )
                ik_dict = getattr(self.planner_manager, "ik_server", None)
                if ik_dict:
                    # Resolve robot_name from config or first key if multi-robot
                    robot_name = (
                        next(iter(ik_dict.keys()))
                        if self.robot_name is None
                        else self.robot_name
                    )
                    if robot_name not in ik_dict:
                        raise RuntimeError(
                            f"IKServer not found for robot '{robot_name}'."
                        )
                    # Single server per robot — ``new_hand_id`` only affects
                    # downstream target packing.
                    self.ik_server = ik_dict[robot_name]

        self.robot_id = new_robot_id
        self.hand_id = new_hand_id
        self.obj_type = new_obj_type
        self.obj_name = new_obj_name
        self.obj_id = new_obj_id
        self.functional_grasp = new_functional_grasp
        self.part = new_part
        self.current_command = new_command

        # Reset phase only if command changed, otherwise keep current phase
        if command_changed or self.current_phase is None:
            self.current_phase = "pre_grasp"
            # Cancel any previous async job by bumping token and clearing job
            self._grasp_token += 1
            self._grasp_job = None
            self.grasp_pose = None
            self.pre_grasp_pose = None
            self.retrieval_grasp_pose = None
            self.grasp_pose_type = None
            self._last_viz_phase = None
            self.get_grasp_pose()

    def _compute_pose_along_grasp_direction(
        self, grasp_pose, offset_distance, backward=True
    ):
        """
        Compute a pose by moving along the grasp direction.

        Args:
            grasp_pose: torch.Tensor of 7 elements [x, y, z, qw, qx, qy, qz]
            offset_distance: Distance to move along grasp direction
            backward: If True, move backward (subtract); if False, move forward (add)

        Returns:
            torch.Tensor: New pose [x, y, z, qw, qx, qy, qz]
        """
        device = self.env.device

        # Ensure grasp_pose is a tensor on the correct device
        if not isinstance(grasp_pose, torch.Tensor):
            grasp_pose = torch.tensor(grasp_pose, device=device, dtype=torch.float32)
        else:
            grasp_pose = grasp_pose.to(device=device)

        # Extract position and quaternion
        grasp_pos = grasp_pose[:3]  # [x, y, z]
        grasp_quat = grasp_pose[3:7]  # [qw, qx, qy, qz]

        # Convert quaternion to rotation matrix
        rot_matrix = quat_to_rot_matrix(grasp_quat.unsqueeze(0))  # (1, 3, 3)

        # Extract z-axis direction (grasp direction)
        grasp_direction = rot_matrix[0, :, 2]  # (3,) - third column is z-axis

        # Normalize the direction vector
        grasp_direction_normalized = grasp_direction / torch.norm(grasp_direction)

        # Compute offset
        offset = grasp_direction_normalized * offset_distance

        # Apply offset
        if backward:
            new_pos = grasp_pos - offset
        else:
            new_pos = grasp_pos + offset

        # Return new pose (position + original quaternion)
        return torch.cat([new_pos, grasp_quat], dim=0)

    def _compute_pose_upward(self, grasp_pose, offset_distance):
        """
        Compute a pose by moving upward along the world z-axis.

        Args:
            grasp_pose: torch.Tensor of 7 elements [x, y, z, qw, qx, qy, qz]
            offset_distance: Distance to move upward along world z-axis

        Returns:
            torch.Tensor: New pose [x, y, z, qw, qx, qy, qz]
        """
        device = self.env.device

        # Ensure grasp_pose is a tensor on the correct device
        if not isinstance(grasp_pose, torch.Tensor):
            grasp_pose = torch.tensor(grasp_pose, device=device, dtype=torch.float32)
        else:
            grasp_pose = grasp_pose.to(device=device)

        # Extract position and quaternion
        grasp_pos = grasp_pose[:3]  # [x, y, z]
        grasp_quat = grasp_pose[3:7]  # [qw, qx, qy, qz]

        # World z-axis is [0, 0, 1]
        world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

        # Compute offset along world z-axis
        offset = world_z_axis * offset_distance

        # Apply offset upward
        new_pos = grasp_pos + offset

        # Return new pose (position + original quaternion)
        return torch.cat([new_pos, grasp_quat], dim=0)

    def _get_env_origin(self) -> torch.Tensor:
        return self.env.scene.env_origins[self.env_id].clone()

    def _grasp_pose_world_for_viz(self, pose: torch.Tensor) -> torch.Tensor:
        """Grasp targets are env-local; Isaac debug draw uses world frame."""
        device = self.env.device
        pose = pose.to(device=device)
        origin = self._get_env_origin().to(device=device)
        if origin.ndim > 1:
            origin = origin[0]
        out = pose.clone()
        out[:3] = pose[:3] + origin[:3]
        return out

    def _viz_grasp_for_phase(self, phase: str | None) -> None:
        """Visualize the current phase's target pose (single pose; replaces prior draw)."""
        if not self.viz_grasp or phase is None:
            return
        pose = None
        if phase == "pre_grasp":
            pose = self.pre_grasp_pose
        elif phase == "grasp":
            pose = self.grasp_pose
        elif phase == "retrieval":
            pose = self.retrieval_grasp_pose
        if pose is not None:
            visualize_grasp_pose(
                [self._grasp_pose_world_for_viz(pose.to(self.env.device))]
            )

    def step(self):
        if self.current_state == "failed":
            self.current_state = "failed"
            self.current_action = "Failed"
            return "Failed"

        # Ensure grasp poses are computed (async, non-blocking)
        if (
            self.grasp_pose is None
            or self.pre_grasp_pose is None
            or self.retrieval_grasp_pose is None
        ):
            self.get_grasp_pose()
            if self.current_state == "computing":
                self.current_action = None
                return None
            # If not computing anymore but still missing poses => failed
            if (
                self.grasp_pose is None
                or self.pre_grasp_pose is None
                or self.retrieval_grasp_pose is None
            ):
                self.current_state = "failed"
                self.current_action = "Failed"
                return "Failed"

        self.current_state = "running"
        move_key = self._move_planner_key

        if self.viz_grasp and self.current_phase != self._last_viz_phase:
            print(f"[Grasp] env_id={self.env_id} phase={self.current_phase}")
            self._viz_grasp_for_phase(self.current_phase)
            self._last_viz_phase = self.current_phase

        if self.current_phase == "pre_grasp":
            # Pre-grasp: use pre-computed pre_grasp_pose (gripper open)
            target_pose_7d = self.pre_grasp_pose.to(device=self.env.device)
            # 8D action: pose + gripper (0.0 = open) - this ensures gripper opens at start of new grasp
            target_pose_8d = torch.cat(
                [
                    target_pose_7d,
                    torch.tensor([0.0], device=self.env.device, dtype=torch.float32),
                ],
                dim=0,
            )
            self.current_action = {
                move_key: ((self.robot_id, self.hand_id, -1), target_pose_8d)
            }
            return {move_key: ((self.robot_id, self.hand_id, -1), target_pose_8d)}

        elif self.current_phase == "grasp":
            if self.reactive and not self._grasp_pose_updated:
                self._update_grasp_from_object_pose()
                self._grasp_pose_updated = True
            # Grasp: move to exact grasp pose (gripper open)
            target_pose_7d = self.grasp_pose.to(device=self.env.device)
            self.current_action = {
                move_key: ((self.robot_id, self.hand_id, 0), target_pose_7d)
            }
            return {move_key: ((self.robot_id, self.hand_id, 0), target_pose_7d)}

        elif self.current_phase == "close_gripper":
            # Close gripper: use ParallelGripper to close gripper (1.0 = close)
            gripper_target = torch.tensor(
                [1.0], device=self.env.device, dtype=torch.float32
            )
            self.current_action = {
                "ParallelGripper": ((self.robot_id, self.hand_id, 0), gripper_target)
            }
            return {
                "ParallelGripper": ((self.robot_id, self.hand_id, 0), gripper_target)
            }

        elif self.current_phase == "retrieval":
            # Retrieval: use pre-computed retrieval_grasp_pose (gripper closed)
            target_pose_7d = self.retrieval_grasp_pose.to(device=self.env.device)
            self.current_action = {
                move_key: ((self.robot_id, self.hand_id, 0), target_pose_7d)
            }
            return {move_key: ((self.robot_id, self.hand_id, 0), target_pose_7d)}
        else:
            self.current_state = "failed"
            self.current_action = None
            return None

    def update(self, info):
        # During async compute, global planner might not exist yet (global_planner_info[env_id] == None).
        if self.current_state == "computing":
            return {
                "atomic_skill_type": "Grasp",
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
                "atomic_skill_type": "Grasp",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }

        gp_info = info.get("global_planner_info", None)
        if gp_info is None or gp_info[self.env_id] is None:
            # No global planner update for this env yet (likely still computing or waiting for first action).
            return {
                "type": "Grasp",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state or "ready",
                "truncated": 0,
                "phase": self.current_phase,
            }

        if gp_info[self.env_id]["finished"]:
            # Global planner finished, move to next phase
            gp_state = gp_info[self.env_id].get("state", "?")
            if self.current_phase == "pre_grasp":
                print(f"[Grasp] env_id={self.env_id} phase=grasp")
                self.current_phase = "grasp"
                self._grasp_pose_updated = False
                return {
                    "type": "Grasp",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: moving to grasp",
                    "truncated": 0,
                    "phase": "grasp",
                }
            elif self.current_phase == "grasp":
                print(f"[Grasp] env_id={self.env_id} phase=close_gripper")
                self.current_phase = "close_gripper"
                return {
                    "type": "Grasp",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: closing gripper",
                    "truncated": 0,
                    "phase": "close_gripper",
                }
            elif self.current_phase == "close_gripper":
                print(f"[Grasp] env_id={self.env_id} phase=retrieval")
                self.current_phase = "retrieval"
                return {
                    "type": "Grasp",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": False,
                    "state": "running: retrieving",
                    "truncated": 0,
                    "phase": "retrieval",
                }
            elif self.current_phase == "retrieval":
                # All phases completed
                self.current_state = "finished"
                print(f"[Grasp] env_id={self.env_id} phase=completed")
                return {
                    "type": "Grasp",
                    "command": self.current_command,
                    "action": self.current_action,
                    "finished": True,
                    "state": self.current_state,
                    "truncated": 0,
                    "phase": "completed",
                }
        elif gp_info[self.env_id]["truncated"] == 1:
            # Global planner truncated, grasp finished
            self.current_state = "truncated: env terminated first"
            print(
                f"[Grasp] env_id={self.env_id} truncated=1 phase={self.current_phase}"
            )
            return {
                "type": "Grasp",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 2:
            # Global planner truncated, grasp truncated
            self.current_state = "truncated: env truncated first"
            print(
                f"[Grasp] env_id={self.env_id} truncated=2 phase={self.current_phase}"
            )
            return {
                "type": "Grasp",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
                "phase": self.current_phase,
            }
        elif gp_info[self.env_id]["truncated"] == 3:
            # Global planner failed
            self.current_state = "failed: global planner failed to plan"
            print(
                f"[Grasp] env_id={self.env_id} truncated=3 phase={self.current_phase}"
            )
            return {
                "type": "Grasp",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
                "phase": self.current_phase,
            }
        else:
            # Global planner running, grasp running
            self.current_state = "running"
            return {
                "type": "Grasp",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
                "phase": self.current_phase,
            }
