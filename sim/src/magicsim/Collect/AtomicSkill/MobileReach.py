from typing import Any

import torch
from omegaconf import DictConfig

from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Planner.PlannerManager import PlannerManager
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class MobileReach(AtomicSkill):
    """Reach task for mobile manipulation using MobileMoveL.

    Supports both single-arm and dual-arm modes:
    - Single-arm (hand_id=0 or 1): action=[skill, robot_id, hand_id, obj_type, obj_name, obj_id]
    - Dual-arm (hand_id=-1): action=[skill, robot_id, -1, right_type, right_name, right_id, left_type, left_name, left_id]

    In dual-arm mode, both arms are planned simultaneously via MobileMoveL
    with hand_id=-1 and a 14D target (right_7d + left_7d).
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
        self.current_target_pose = None
        self.robot_id = int(getattr(config, "robot_id", 0))
        self.hand_id = int(getattr(config, "hand_id", 0))

        # Dual-arm fields
        self.is_dual = False
        self.right_obj_type = None
        self.right_obj_name = None
        self.right_obj_id = None
        self.left_obj_type = None
        self.left_obj_name = None
        self.left_obj_id = None

    # ------------------------------------------------------------------
    # Target building
    # ------------------------------------------------------------------

    def _get_object_pose_7d(self, obj_type, obj_name, obj_id) -> torch.Tensor:
        """Get 7D pose (pos + quat) for a scene object."""
        target_pose = self.env.scene.scene_manager.get_category(obj_type)[self.env_id][
            obj_name
        ][obj_id].get_local_pose()
        target_pos, target_quat = target_pose
        return torch.cat([target_pos, target_quat], dim=0)

    def _build_target_action(self) -> torch.Tensor:
        """Build the target pose tensor.

        Returns:
            - Single-arm: 7D tensor
            - Dual-arm: 14D tensor (right_7d + left_7d)
        """
        if self.is_dual:
            return self._build_dual_target_action()

        # Single-arm: if target info is None (during init phase), return NaN placeholder
        if self.obj_type is None or self.obj_name is None or self.obj_id is None:
            return torch.full(
                (7,), torch.nan, device=self.env.device, dtype=torch.float32
            )
        return self._get_object_pose_7d(self.obj_type, self.obj_name, self.obj_id)

    def _build_dual_target_action(self) -> torch.Tensor:
        """Build the 14D target for dual-arm mode: right_7d + left_7d."""
        nan_7d = torch.full(
            (7,), torch.nan, device=self.env.device, dtype=torch.float32
        )
        # Right arm target
        if (
            self.right_obj_type is None
            or self.right_obj_name is None
            or self.right_obj_id is None
        ):
            right_7d = nan_7d
        else:
            right_7d = self._get_object_pose_7d(
                self.right_obj_type, self.right_obj_name, self.right_obj_id
            )
        # Left arm target
        if (
            self.left_obj_type is None
            or self.left_obj_name is None
            or self.left_obj_id is None
        ):
            left_7d = nan_7d.clone()
        else:
            left_7d = self._get_object_pose_7d(
                self.left_obj_type, self.left_obj_name, self.left_obj_id
            )
        return torch.cat([right_7d, left_7d], dim=0)

    # ------------------------------------------------------------------
    # Action parsing helpers
    # ------------------------------------------------------------------

    def _parse_single_action(self, action: list[Any]):
        """Parse standard 6-element single-arm action."""
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        self.is_dual = False
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]

    def _parse_dual_action(self, action: list[Any]):
        """Parse 9-element dual-arm action."""
        self.robot_id = int(action[1])
        self.hand_id = -1
        self.is_dual = True
        self.right_obj_type = action[3]
        self.right_obj_name = action[4]
        self.right_obj_id = action[5]
        self.left_obj_type = action[6]
        self.left_obj_name = action[7]
        self.left_obj_id = action[8]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, action: list[Any]):
        # action format:
        #   6 elements: [skill_name, robot_id, hand_id, obj_type, obj_name, obj_id]
        #   9 elements: [skill_name, robot_id, -1, right_type, right_name, right_id,
        #                left_type, left_name, left_id]
        if len(action) == 9:
            self._parse_dual_action(action)
        else:
            self._parse_single_action(action)

        self.current_state = "ready"
        self.current_command = list(action)
        self.current_target_pose = self._build_target_action()
        self.planner_manager: PlannerManager = self._get_planner_manager()
        self.planner_manager.update_obstacles(
            obstacle_avoidance_path_list=["dynamic"],
            obstacle_ignore_path_list=["cube"],
            env_ids=[self.env_id],
        )

    def _get_planner_manager(self):
        planner_manager = getattr(self.env.scene, "planner_manager", None)
        if planner_manager is None:
            raise RuntimeError("PlannerManager is not available in the environment.")
        return planner_manager

    def refresh(self, action: list[Any]):
        if len(action) == 9:
            self._parse_dual_action(action)
        else:
            self._parse_single_action(action)

        self.current_command = list(action)
        self.current_target_pose = self._build_target_action()

    def step(self):
        self.current_target_pose = self._build_target_action()
        self.current_state = "running"
        # hand_id: -1 for dual-arm (14D target), 0 or 1 for single-arm (7D target)
        self.current_action = {
            "MobileMoveL": (
                (self.robot_id, self.hand_id, -1),
                self.current_target_pose,
            )
        }
        return self.current_action

    def update(self, info):
        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "atomic_skill_type": "MobileReach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        elif info["global_planner_info"][self.env_id]["truncated"] == 1:
            self.current_state = "truncated: env terminated first"
            return {
                "type": "MobileReach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["global_planner_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"
            return {
                "type": "MobileReach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif info["global_planner_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "MobileReach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        else:
            self.current_state = "running"
            return {
                "type": "MobileReach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
