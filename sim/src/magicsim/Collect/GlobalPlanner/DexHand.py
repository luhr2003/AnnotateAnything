"""
DexHand: Global planner for dexterous hand control.

Similar to ParallelGripper: accepts hand joint targets only, outputs
base(NaN) + arm(NaN) + hand_joints. Base and arm are always filled with NaN.
"""

from typing import Any, Dict
import torch
from magicsim.Collect.GlobalPlanner.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class DexHand(GlobalPlanner):
    """
    Global planner for dexterous hand control.
    Accepts hand joint targets only; base and arm are always NaN.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.current_target = None
        self.current_eef_target = None
        self.step_count = 0
        self.previous_hand_joint_pos = None
        self.robot_id = -1
        self.hand_id = 0
        self.robot_name = None
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

    def _parse_action(self, action):
        """Parse ((robot_id, hand_id, ...), hand_joints) or (robot_id, hand_joints)."""
        robot_id = self.robot_id if self.robot_id >= 0 else 0
        hand_id = self.hand_id
        target = action
        if isinstance(action, (list, tuple)) and len(action) == 2:
            header = action[0]
            target = action[1]
            if isinstance(header, (list, tuple)):
                robot_id = int(header[0])
                hand_id = int(header[1]) if len(header) > 1 else 0
            else:
                robot_id = int(header)
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
            input_cfg = getattr(self.config, "input", None)
            base_dim = int(getattr(input_cfg, "base_dim", 0)) if input_cfg else 0
            arm_dim = int(getattr(input_cfg, "arm_dim", 0)) if input_cfg else 0
            eef_dim = int(getattr(input_cfg, "eef_dim", 0)) if input_cfg else 0
        return base_dim, arm_dim, eef_dim

    def _get_eef_num(self) -> int:
        """Return eef_num (number of hands) from planner_manager servers.

        Post-MERGE_LEFT_RIGHT §1–§8 flatten: one server per robot, indexed
        directly by robot_name (no inner hand_id dict).
        """
        planner_manager = self._get_planner_manager()
        for server_dict in (
            getattr(planner_manager, "ik_server", {}),
            getattr(planner_manager, "motiongen_server", {}),
        ):
            if self.robot_name in server_dict:
                srv = server_dict[self.robot_name]
                if srv is not None and hasattr(srv, "eef_num"):
                    return int(srv.eef_num)
        return 1

    def _get_effective_eef_dim(self) -> int:
        """Per-hand eef dim for hand_id 0/1; full eef_dim for hand_id -1."""
        _, arm_dim, eef_dim = self._get_action_dims()
        eef_num = self._get_eef_num()
        if eef_num <= 1 or self.hand_id == -1:
            return eef_dim
        total_arms = arm_dim // 7 if arm_dim > 0 and arm_dim % 7 == 0 else 1
        return max(eef_dim // total_arms, 1)

    def _expand_eef_action(self, hand_action: torch.Tensor) -> torch.Tensor:
        """Expand single-hand action to full eef_dim, fill other hand(s) with NaN."""
        _, _, eef_dim = self._get_action_dims()
        hand_action = hand_action.view(-1)
        if hand_action.shape[0] == eef_dim:
            return hand_action
        per_hand = hand_action.shape[0]
        full_eef = torch.full(
            (eef_dim,), torch.nan, device=hand_action.device, dtype=hand_action.dtype
        )
        if self.hand_id == 0:
            full_eef[:per_hand] = hand_action
        elif self.hand_id == 1:
            full_eef[eef_dim - per_hand :] = hand_action
        return full_eef

    def _get_hand_joint_dim(self) -> int:
        """Number of hand joints (last N of joint_pos)."""
        _, _, eef_dim = self._get_action_dims()
        return eef_dim

    def _get_hand_joint_pos(self) -> torch.Tensor:
        """Current hand joint positions (last hand_joint_dim of joint_pos)."""
        joint_pos = self.env.scene.robot_manager.get_robot_state()[0][self.robot_name][
            "joint_pos"
        ][self.env_id]
        dim = self._get_hand_joint_dim()
        return joint_pos[-dim:]

    def _build_full_action(self, hand_action: torch.Tensor) -> torch.Tensor:
        """Build base(NaN) + arm(NaN) + hand_action. Expands single-hand to full eef_dim."""
        base_dim, arm_dim, eef_dim = self._get_action_dims()
        hand_action = hand_action.view(-1).to(
            device=self.env.device, dtype=torch.float32
        )
        if hand_action.shape[0] != eef_dim:
            hand_action = self._expand_eef_action(hand_action)
        assert hand_action.shape[0] == eef_dim, (
            f"hand_action dim {hand_action.shape[0]} != eef_dim {eef_dim}"
        )
        device = hand_action.device
        dtype = hand_action.dtype
        chunks = []
        if base_dim > 0:
            chunks.append(
                torch.full((base_dim,), torch.nan, device=device, dtype=dtype)
            )
        if arm_dim > 0:
            chunks.append(torch.full((arm_dim,), torch.nan, device=device, dtype=dtype))
        chunks.append(hand_action)
        return torch.cat(chunks, dim=0)

    def reset(self, action: torch.Tensor):
        """Reset with ((robot_id, hand_id, ...), hand_joints)."""
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        if not isinstance(target_action, torch.Tensor):
            target_action = torch.as_tensor(
                target_action, device=self.env.device, dtype=torch.float32
            )
        else:
            target_action = target_action.to(self.env.device, dtype=torch.float32)
        target_action = target_action.view(-1)
        self.current_eef_target = target_action.clone()
        self.current_target = self._build_full_action(self.current_eef_target)
        self.previous_hand_joint_pos = self._get_hand_joint_pos().clone()
        self.step_count = 0
        self.current_state = "ready"
        self.current_command = ["DexHand", self.robot_name, self.current_target]

    def step(self) -> torch.Tensor:
        if self.current_target is None:
            raise RuntimeError("Current target not set. Call reset first.")
        self.current_state = "running"
        output_action = self._build_full_action(self.current_eef_target)
        self.current_action = {"DexHand": output_action}
        self.step_count += 1
        return output_action

    def refresh(self, action: torch.Tensor):
        """Refresh hand joint target."""
        robot_id, hand_id, target_action = self._parse_action(action)
        self._set_robot_by_id(robot_id, hand_id)
        if not isinstance(target_action, torch.Tensor):
            target_action = torch.as_tensor(
                target_action, device=self.env.device, dtype=torch.float32
            )
        else:
            target_action = target_action.to(self.env.device, dtype=torch.float32)
        self.current_eef_target = target_action.view(-1).clone()
        self.current_target = self._build_full_action(self.current_eef_target)
        self.current_command = ["DexHand", self.robot_name, self.current_target]

    def get_done(self) -> bool:
        """Done when timeout reached or hand joints stopped moving."""
        current_pos = self._get_hand_joint_pos()
        if self.previous_hand_joint_pos is None:
            self.previous_hand_joint_pos = current_pos.clone()
            return False
        diff = torch.norm(current_pos - self.previous_hand_joint_pos)
        hand_done = diff < getattr(self.config, "hand_threshold", 0.01)
        timeout_steps = int(getattr(self.config, "timeout_steps", 50))
        timeout_reached = self.step_count >= timeout_steps
        if timeout_reached and hand_done:
            return True
        self.previous_hand_joint_pos = current_pos.clone()
        return False

    def update(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_state == "failed":
            return {
                "type": "DexHand",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        elif self.current_state == "finished" or self.get_done():
            self.current_state = "finished"
            return {
                "type": "DexHand",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif info["env_info"][2][self.env_id]:
            return {
                "type": "DexHand",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": "truncated: env terminated first",
                "truncated": 1,
            }
        elif info["env_info"][3][self.env_id]:
            return {
                "type": "DexHand",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": "truncated: env truncated first",
                "truncated": 2,
            }
        return {
            "type": "DexHand",
            "command": self.current_command,
            "action": self.current_action,
            "finished": False,
            "state": "running",
            "truncated": 0,
        }
