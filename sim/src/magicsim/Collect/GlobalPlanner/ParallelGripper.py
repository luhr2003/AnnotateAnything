from typing import Any, Dict
import torch
from magicsim.Collect.GlobalPlanner.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class ParallelGripper(GlobalPlanner):
    """
    Global Planner for parallel gripper control.
    Accepts 1D input (gripper target value) and outputs cat(robot joint pose, input).
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.current_target = None  # 1D: gripper target value
        self.current_base_target = None
        self.current_arm_target = None
        self.current_eef_target = None
        self.step_count = 0
        self.previous_gripper_joint_pos = (
            None  # Previous gripper joint positions (last 2 dimensions)
        )
        self.robot_id = -1
        self.hand_id = 0  # 0 = right (default), 1 = left
        self.robot_name = None  # Robot name, will be set during first reset
        super().__init__(config, env, env_id, logger)

    def _get_robot_name_list(self):
        return list(self.env.scene.robot_manager.robots.keys())

    def _set_robot_by_id(self, robot_id: int, hand_id: int = 0) -> bool:
        robot_id = int(robot_id)
        hand_id = int(hand_id)
        if (
            robot_id == self.robot_id
            and hand_id == self.hand_id
            and self.robot_name is not None
        ):
            return False
        robot_name_list = self._get_robot_name_list()
        if robot_id < 0 or robot_id >= len(robot_name_list):
            raise ValueError(
                f"robot_id {robot_id} out of range for robot_name_list={robot_name_list}"
            )
        self.robot_id = robot_id
        self.hand_id = hand_id
        self.robot_name = robot_name_list[robot_id]
        return True

    def _get_planner_manager(self):
        planner_manager = getattr(self.env.scene, "planner_manager", None)
        if planner_manager is None:
            raise RuntimeError("PlannerManager is not available in the environment.")
        return planner_manager

    def _parse_action(self, action: torch.Tensor):
        """Parse action format (same header rules as ``GlobalPlanner.parse_planner_header``).

        Supported formats:
            - target_tensor → default robot/hand; extra header ints ignored for gripper
            - (robot_id, target_tensor) → legacy
            - ((robot_id, hand_id, planner_mode), target_tensor)
        """
        robot_id, hand_id, _mode, target = GlobalPlanner.parse_planner_header(
            action,
            default_robot_id=self.robot_id if self.robot_id >= 0 else 0,
            default_hand_id=self.hand_id,
        )
        return robot_id, hand_id, target

    def _get_action_dims(self) -> tuple[int, int, int]:
        if self.robot_name is None:
            raise RuntimeError("Robot name not set. Call _set_robot_by_id first.")
        planner_manager = self._get_planner_manager()
        info = planner_manager.get_info()
        robot_info = info.get(self.robot_name, {})
        base_dim = int(robot_info.get("base", {}).get("action_dim", 0))
        arm_dim = int(robot_info.get("arm", {}).get("action_dim", 0))
        eef_dim = int(robot_info.get("eef", {}).get("action_dim", 0))
        if base_dim == 0 and arm_dim == 0 and eef_dim == 0:
            # Fallback to config dims for backward compatibility
            input_cfg = getattr(self.config, "input", None)
            base_dim = int(getattr(input_cfg, "base_dim", 0)) if input_cfg else 0
            arm_dim = int(getattr(input_cfg, "arm_dim", 0)) if input_cfg else 0
            eef_dim = int(getattr(input_cfg, "eef_dim", 0)) if input_cfg else 0
        return base_dim, arm_dim, eef_dim

    def _expand_arm_action(self, arm_action: torch.Tensor) -> torch.Tensor:
        """Expand a per-arm command to full ``arm_dim`` (NaN for the other arm). Same as MoveL/ServoL."""
        _, arm_dim, _ = self._get_action_dims()
        arm_action = arm_action.view(-1)
        if arm_action.shape[0] == arm_dim:
            return arm_action
        full_arm = torch.full(
            (arm_dim,), torch.nan, device=arm_action.device, dtype=arm_action.dtype
        )
        if self.hand_id == 0:
            full_arm[: arm_action.shape[0]] = arm_action
        elif self.hand_id == 1:
            full_arm[arm_dim - arm_action.shape[0] :] = arm_action
        else:
            full_arm[: arm_action.shape[0]] = arm_action
        return full_arm

    def _expand_eef_target(self, eef_target: torch.Tensor) -> torch.Tensor:
        """Expand per-EEF gripper command to full ``eef_dim``. Same rules as MoveL/ServoL."""
        _, _, eef_dim = self._get_action_dims()
        if eef_dim == 0:
            return eef_target
        flat = eef_target.view(-1)
        if flat.shape[0] == eef_dim:
            return flat
        full_eef = torch.full(
            (eef_dim,), torch.nan, device=flat.device, dtype=flat.dtype
        )
        if self.hand_id == 0:
            full_eef[: flat.shape[0]] = flat
        elif self.hand_id == 1:
            full_eef[eef_dim - flat.shape[0] :] = flat
        else:
            full_eef[: flat.shape[0]] = flat
        return full_eef

    def _parse_target(self, action: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(
                action, device=self.env.device, dtype=torch.float32
            )
        else:
            action = action.to(self.env.device, dtype=torch.float32)
        if action.ndim != 1:
            action = action.view(-1)

        base_dim, arm_dim, eef_dim = self._get_action_dims()
        max_eef_num, per_eef_dim = self._eef_layout_from_robot()

        if eef_dim > 0:
            if per_eef_dim * max_eef_num != eef_dim:
                raise ValueError(
                    f"eef_dim {eef_dim} inconsistent with RobotManager layout "
                    f"max_eef_num={max_eef_num}, per_eef_dim={per_eef_dim}."
                )
        elif per_eef_dim != 0:
            raise ValueError(
                f"eef_dim is 0 but per_eef_dim={per_eef_dim} (RobotManager.get_info)."
            )
        if per_eef_dim <= 0:
            raise ValueError("ParallelGripper requires eef_dim > 0 (gripper action).")

        active_eef_num = GlobalPlanner.eef_num_from_hand_id(self.hand_id)
        if active_eef_num > max_eef_num:
            raise ValueError(
                f"hand_id={self.hand_id} implies {active_eef_num} active EEFs but "
                f"max_eef_num is {max_eef_num}."
            )
        matched_eef_dim = per_eef_dim * active_eef_num
        arm_dim = active_eef_num * 7
        total_dim = int(action.shape[0])
        base_target = None
        arm_target = None
        eef_target = None

        if total_dim == matched_eef_dim:
            eef_target = action.clone()
        elif base_dim > 0 and total_dim == base_dim + matched_eef_dim:
            base_target = action[:base_dim].clone()
            eef_target = action[base_dim:].clone()
        elif arm_dim > 0 and total_dim == arm_dim + matched_eef_dim:
            arm_target = action[:arm_dim].clone()
            eef_target = action[arm_dim:].clone()
        elif (
            base_dim > 0
            and arm_dim > 0
            and total_dim == base_dim + arm_dim + matched_eef_dim
        ):
            base_target = action[:base_dim].clone()
            arm_target = action[base_dim : base_dim + arm_dim].clone()
            eef_target = action[base_dim + arm_dim :].clone()
        else:
            valid = [str(matched_eef_dim)]
            if base_dim > 0:
                valid.append(f"base({base_dim})+{matched_eef_dim}")
            if arm_dim > 0:
                valid.append(f"arm({arm_dim})+{matched_eef_dim}")
            if base_dim > 0 and arm_dim > 0:
                valid.append(f"base({base_dim})+arm({arm_dim})+{matched_eef_dim}")
            raise ValueError(
                f"ParallelGripper action dim {total_dim} does not match any valid combination: "
                f"{', '.join(valid)} (hand_id={self.hand_id}, active_eef_num={active_eef_num}, "
                f"per_eef_dim={per_eef_dim})."
            )

        if per_eef_dim > 0 and eef_target.numel() == active_eef_num * per_eef_dim:
            eef_target = eef_target.view(active_eef_num, per_eef_dim)

        return base_target, arm_target, eef_target

    def _build_full_action(self, eef_action: torch.Tensor) -> torch.Tensor:
        """Full robot action ``base | arm | eef``; aligns with MoveL stacking (NaN placeholders, expands)."""
        base_dim, arm_dim, eef_dim = self._get_action_dims()

        eef_flat = self._expand_eef_target(eef_action)
        if eef_flat.shape[0] != eef_dim:
            raise ValueError(
                f"eef_action length {eef_flat.shape[0]} does not match eef_dim {eef_dim}."
            )

        device = eef_flat.device
        dtype = eef_flat.dtype
        chunks = []
        if base_dim > 0:
            if self.current_base_target is None:
                chunks.append(
                    torch.tensor(
                        [torch.nan] * base_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                chunks.append(self.current_base_target)
        if arm_dim > 0:
            if self.current_arm_target is None:
                chunks.append(
                    torch.tensor(
                        [torch.nan] * arm_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                arm_flat = self._expand_arm_action(self.current_arm_target.view(-1))
                if arm_flat.shape[0] != arm_dim:
                    raise ValueError(
                        f"arm slice length {arm_flat.shape[0]} does not match arm_dim {arm_dim}."
                    )
                chunks.append(arm_flat)
        chunks.append(eef_flat)
        output_action = torch.cat(chunks, dim=0)
        assert output_action.shape[0] == base_dim + arm_dim + eef_dim, (
            f"Output action shape {output_action.shape[0]} does not match expected "
            f"base_dim={base_dim}, arm_dim={arm_dim}, eef_dim={eef_dim}."
        )
        return output_action

    def _get_robot_name(self):
        """Get the robot name from robot manager."""
        if self.robot_name is None:
            self._set_robot_by_id(0)
        return self.robot_name

    def _get_robot_joint_pos(self):
        """Get current robot joint positions."""
        robot_states = self.env.scene.robot_manager.get_robot_state()[0]
        robot_name = self._get_robot_name()
        joint_pos = robot_states[robot_name]["joint_pos"][self.env_id]
        return joint_pos

    def _get_gripper_joint_pos(self):
        """Get current gripper joint positions (last 2 dimensions)."""
        joint_pos = self._get_robot_joint_pos()
        # Gripper joints are the last 2 dimensions
        gripper_joint_pos = joint_pos[-2:]
        return gripper_joint_pos

    def reset(self, action: torch.Tensor):
        """Reset the global planner with ((robot_id, hand_id, ...), gripper_target)."""
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        (
            self.current_base_target,
            self.current_arm_target,
            self.current_eef_target,
        ) = self._parse_target(target_action)
        self.current_target = self._build_full_action(self.current_eef_target)

        # Get initial gripper joint positions
        self.previous_gripper_joint_pos = self._get_gripper_joint_pos().clone()

        self.step_count = 0
        self.current_state = "ready"
        self.current_command = ["ParallelGripper", self.robot_name, self.current_target]

    def step(self) -> torch.Tensor:
        """Return cat(robot joint pose, input)."""
        if self.current_target is None:
            raise RuntimeError("Current Target Is Not Set, Please Call Reset First")

        self.current_state = "running"

        output_action = self._build_full_action(self.current_eef_target)

        self.current_action = {"ParallelGripper": output_action}
        self.step_count += 1

        return output_action

    def refresh(self, action: torch.Tensor):
        """Refresh the gripper target value."""
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        (
            self.current_base_target,
            self.current_arm_target,
            self.current_eef_target,
        ) = self._parse_target(target_action)
        self.current_target = self._build_full_action(self.current_eef_target)
        self.current_command = ["ParallelGripper", self.robot_name, self.current_target]

    def get_done(self) -> bool:
        """Check if gripper has reached target by comparing consecutive gripper joint positions."""
        current_gripper_joint_pos = self._get_gripper_joint_pos()

        if self.previous_gripper_joint_pos is None:
            # First time checking, update and return False
            self.previous_gripper_joint_pos = current_gripper_joint_pos.clone()
            return False

        # Calculate the difference between current and previous gripper joint positions
        gripper_diff = torch.norm(
            current_gripper_joint_pos - self.previous_gripper_joint_pos
        )

        # Check if difference is below threshold
        gripper_done = gripper_diff < self.config.gripper_threshold

        timeout_steps = int(getattr(self.config, "timeout_steps", 30))
        timeout_reached = self.step_count >= timeout_steps

        # Done only when BOTH timeout reached AND gripper has stopped moving
        if timeout_reached and gripper_done:
            return True

        # Update previous gripper joint positions for next check
        self.previous_gripper_joint_pos = current_gripper_joint_pos.clone()
        return False

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Update the global planner state based on environment info."""
        if self.current_state == "failed":
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "ParallelGripper",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif self.current_state == "finished" or self.get_done():
            self.current_state = "finished"
            return {
                "type": "ParallelGripper",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["env_info"][2][self.env_id]:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "ParallelGripper",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["env_info"][3][self.env_id]:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "ParallelGripper",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        else:
            self.current_state = "running"
            return {
                "type": "ParallelGripper",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
