from typing import Any

import torch
from magicsim.Collect.AtomicSkill.AtomicSkill import AtomicSkill
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class Reach(AtomicSkill):
    """Reach for TableTop using MoveL.

    Supports both single-arm and dual-arm modes:
    - Single-arm (hand_id=0 or 1): action=[Reach, robot_id, hand_id, obj_type, obj_name, obj_id]
    - Dual-arm (hand_id=-1): action=[Reach, robot_id, -1, right_type, right_name, right_id,
                                    left_type, left_name, left_id]

    In dual-arm mode, both arms are planned simultaneously via MoveL
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
        self.obj_type = None
        self.obj_name = None
        self.obj_id = None
        self.right_obj_type = None
        self.right_obj_name = None
        self.right_obj_id = None
        self.left_obj_type = None
        self.left_obj_name = None
        self.left_obj_id = None

    def _get_object_pose_7d(self, obj_type, obj_name, obj_id) -> torch.Tensor:
        """Get 7D pose (pos + quat) for a scene object."""
        target_pose = self.env.scene.scene_manager.get_category(obj_type)[self.env_id][
            obj_name
        ][obj_id].get_local_pose()
        target_pos, target_quat = target_pose
        return torch.cat([target_pos, target_quat], dim=0)

    def _build_target_action(self) -> torch.Tensor:
        """Returns 7D for single-arm, 14D (right+left) for dual-arm."""
        if self.is_dual:
            nan_7d = torch.full(
                (7,), torch.nan, device=self.env.device, dtype=torch.float32
            )
            right_7d = (
                nan_7d
                if (
                    self.right_obj_type is None
                    or self.right_obj_name is None
                    or self.right_obj_id is None
                )
                else self._get_object_pose_7d(
                    self.right_obj_type, self.right_obj_name, self.right_obj_id
                )
            )
            left_7d = (
                nan_7d.clone()
                if (
                    self.left_obj_type is None
                    or self.left_obj_name is None
                    or self.left_obj_id is None
                )
                else self._get_object_pose_7d(
                    self.left_obj_type, self.left_obj_name, self.left_obj_id
                )
            )
            return torch.cat([right_7d, left_7d], dim=0)

        if self.obj_type is None or self.obj_name is None or self.obj_id is None:
            return torch.full(
                (7,), torch.nan, device=self.env.device, dtype=torch.float32
            )
        return self._get_object_pose_7d(self.obj_type, self.obj_name, self.obj_id)

    def _parse_single_action(self, action: list[Any]):
        self.robot_id = int(action[1])
        self.hand_id = int(action[2])
        self.is_dual = False
        self.obj_type = action[3]
        self.obj_name = action[4]
        self.obj_id = action[5]

    def _parse_dual_action(self, action: list[Any]):
        self.robot_id = int(action[1])
        self.hand_id = -1
        self.is_dual = True
        self.right_obj_type = action[3]
        self.right_obj_name = action[4]
        self.right_obj_id = action[5]
        self.left_obj_type = action[6]
        self.left_obj_name = action[7]
        self.left_obj_id = action[8]

    def reset(self, action: list[Any]):
        # 6 elements: single-arm; 9 elements: dual-arm
        if len(action) == 9:
            self._parse_dual_action(action)
        else:
            self._parse_single_action(action)

        self.current_state = "ready"
        self.current_command = list(action)
        self.current_target_pose = self._build_target_action()

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
        self.current_action = {
            "MoveL": (
                (self.robot_id, self.hand_id, 1),
                self.current_target_pose,
            ),
        }
        return self.current_action

    def update(self, info):
        if self.current_state == "failed":
            self.current_state = "failed: atomicskill failed to plan"
            return {
                "atomic_skill_type": "Reach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 4,
            }
        if info["global_planner_info"][self.env_id][
            "finished"
        ]:  # global planner finished, reach finished
            self.current_state = "finished"
            return {
                "type": "Reach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 0,
            }
        elif (
            info["global_planner_info"][self.env_id]["truncated"] == 1
        ):  # global planner truncated, reach finished
            self.current_state = "truncated: env terminated first"
            return {
                "type": "Reach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": True,
                "state": self.current_state,
                "truncated": 1,
            }
        elif info["global_planner_info"][self.env_id]["truncated"] == 2:
            self.current_state = "truncated: env truncated first"  # global planner truncated, reach truncated
            return {
                "type": "Reach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 2,
            }
        elif info["global_planner_info"][self.env_id]["truncated"] == 3:
            self.current_state = "failed: global planner failed to plan"
            return {
                "type": "Reach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 3,
            }
        else:
            self.current_state = "running"  # global planner running, reach running
            return {
                "type": "Reach",
                "command": self.current_command,
                "action": self.current_action,
                "finished": False,
                "state": self.current_state,
                "truncated": 0,
            }
