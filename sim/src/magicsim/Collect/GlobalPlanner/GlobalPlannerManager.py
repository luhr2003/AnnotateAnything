from typing import Any, Dict, List, Sequence
import torch
from magicsim.Collect.GlobalPlanner import GlobalPlanner
from magicsim.Env.Utils.file import Logger
from omegaconf import DictConfig
from magicsim.Collect.GlobalPlanner.MoveL import MoveL
from magicsim.Collect.GlobalPlanner.MobileMoveL import MobileMoveL
from magicsim.Collect.GlobalPlanner.RetractMoveL import RetractMoveL
from magicsim.Collect.GlobalPlanner.ParallelGripper import ParallelGripper
from magicsim.Collect.GlobalPlanner.DexHand import DexHand
from magicsim.Collect.GlobalPlanner.NavTo import NavTo
from magicsim.Collect.GlobalPlanner.ServoL import ServoL
from magicsim.Collect.GlobalPlanner.MobileServoL import MobileServoL
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class GlobalPlannerManager:
    """
    Global Planner for all tasks.
    """

    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        global_planner_config: DictConfig,
        device=torch.device("cpu"),
        logger: Logger = None,
    ):
        self.num_envs = num_envs
        self.env = env
        self.global_planner_config = global_planner_config
        self.device = device
        self.logger = logger
        self.global_planner_list: list[GlobalPlanner] = [None] * num_envs
        self.global_planner_type_list: list[str] = [None] * num_envs
        self.info_list: List[Dict[str, Any]] = [None] * self.num_envs

    def create_global_planner(self, global_planner_type: str, env_id: int):
        if global_planner_type == "MoveL":
            self.global_planner_list[env_id] = MoveL(
                self.global_planner_config.MoveL, self.env, env_id, self.logger
            )
            self.global_planner_type_list[env_id] = "MoveL"
        elif global_planner_type == "MobileMoveL":
            self.global_planner_list[env_id] = MobileMoveL(
                self.global_planner_config.MobileMoveL,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "MobileMoveL"
        elif global_planner_type == "ParallelGripper":
            self.global_planner_list[env_id] = ParallelGripper(
                self.global_planner_config.ParallelGripper,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "ParallelGripper"
        elif global_planner_type == "DexHand":
            self.global_planner_list[env_id] = DexHand(
                self.global_planner_config.DexHand,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "DexHand"
        elif global_planner_type == "RetractMoveL":
            self.global_planner_list[env_id] = RetractMoveL(
                self.global_planner_config.RetractMoveL,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "RetractMoveL"
        elif global_planner_type == "NavTo":
            self.global_planner_list[env_id] = NavTo(
                self.global_planner_config.NavTo,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "NavTo"
        elif global_planner_type == "ServoL":
            self.global_planner_list[env_id] = ServoL(
                self.global_planner_config.ServoL,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "ServoL"
        elif global_planner_type == "MobileServoL":
            self.global_planner_list[env_id] = MobileServoL(
                self.global_planner_config.MobileServoL,
                self.env,
                env_id,
                self.logger,
            )
            self.global_planner_type_list[env_id] = "MobileServoL"
        else:
            raise ValueError(
                f"Global planner type {global_planner_type} not supported."
            )

    def step(self, actions: List[Dict[str, torch.Tensor]], env_ids: Sequence[int]):
        output_action = []
        valid_env_ids = []
        failed_env_ids = []
        for i, env_id in enumerate(env_ids):
            if actions[i] is not None:  # New command is given
                if self.global_planner_type_list[env_id] is None:
                    self.create_global_planner(list(actions[i].keys())[0], env_id)
                    self.global_planner_list[env_id].reset(list(actions[i].values())[0])
                else:
                    if (
                        list(actions[i].keys())[0]
                        == self.global_planner_type_list[env_id]
                    ):
                        self.global_planner_list[env_id].refresh(
                            list(actions[i].values())[0]
                        )
                    else:
                        # Swap planner type (e.g. DexGrasp pre_grasp RetractMoveL → MobileMoveL):
                        # drop the old planner and create the new one instead of erroring.
                        self.global_planner_list[env_id] = None
                        self.global_planner_type_list[env_id] = None
                        self.create_global_planner(list(actions[i].keys())[0], env_id)
                        self.global_planner_list[env_id].reset(
                            list(actions[i].values())[0]
                        )
            else:
                raise RuntimeError(
                    f"Global planner action is None for env {env_id}, but global planner list is not None"
                )

            action = self.global_planner_list[env_id].step()
            if action == "Failed":
                failed_env_ids.append(env_id)
            elif action is not None:
                output_action.append(action)
                valid_env_ids.append(env_id)
        assert len(output_action) == len(valid_env_ids), (
            "Output action length should be equal to valid env ids length"
        )
        # print("output_action: ", output_action)
        if len(output_action) == 0:
            return torch.tensor([], device=self.device), valid_env_ids, failed_env_ids

        return torch.stack(output_action), valid_env_ids, failed_env_ids

    def update(self, info: Dict[str, Any]):
        for env_id in range(self.num_envs):
            if self.global_planner_type_list[env_id] is None:
                # Keep last info so atomic skill receives finished signal and can transition
                continue
            self.info_list[env_id] = self.global_planner_list[env_id].update(info)
            if (
                self.info_list[env_id]["finished"]
                or self.info_list[env_id]["truncated"] > 0
            ):
                if self.info_list[env_id]["truncated"] != 5:
                    print(
                        f"Global planner {self.global_planner_type_list[env_id]} of env {env_id} is finished or truncated, truncated: {self.info_list[env_id]['truncated']}"
                    )
                self.global_planner_type_list[env_id] = None
                self.global_planner_list[env_id] = None
        return self.info_list

    def get_manager_info(self):
        return {
            "global_planner_type_list": self.global_planner_type_list,
            "global_planner_list": self.global_planner_list,
        }

    def reset(self):
        """Clear all planners so next episode starts fresh."""
        self.global_planner_list = [None] * self.num_envs
        self.global_planner_type_list = [None] * self.num_envs
        self.info_list = [None] * self.num_envs
        return [None] * self.num_envs
