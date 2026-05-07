from typing import Any, Dict, List, Sequence
import torch
from magicsim.Collect.AtomicSkill import AtomicSkill
from magicsim.Collect.AtomicSkill.Reach import Reach
from magicsim.Collect.AtomicSkill.Grasp import Grasp
from magicsim.Collect.AtomicSkill.MobileReach import MobileReach
from magicsim.Collect.AtomicSkill.RetractReach import RetractReach
from magicsim.Collect.AtomicSkill.NavTo import NavTo
from magicsim.Collect.AtomicSkill.Push import Push
from magicsim.Collect.AtomicSkill.Wave import Wave
from magicsim.Collect.AtomicSkill.DexGrasp import DexGrasp
from magicsim.Collect.AtomicSkill.OpenDrawer import OpenDrawer
from magicsim.Collect.AtomicSkill.DexOpenDrawer import DexOpenDrawer
from magicsim.Collect.AtomicSkill.CloseDrawer import CloseDrawer
from magicsim.Collect.AtomicSkill.LocoOpenDoor import LocoOpenDoor
from magicsim.Collect.AtomicSkill.Dehatch import Dehatch
from magicsim.Collect.AtomicSkill.Fling import Fling
from magicsim.Collect.AtomicSkill.Fold import Fold
from magicsim.Collect.AtomicSkill.Lift import Lift
from magicsim.Collect.AtomicSkill.LocoBox import LocoBox
from magicsim.Collect.AtomicSkill.BiGrasp import BiGrasp
from magicsim.Collect.AtomicSkill.BiDexGrasp import BiDexGrasp
from magicsim.Collect.AtomicSkill.Handover import Handover
from magicsim.Env.Utils.file import Logger
from omegaconf import DictConfig
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv


class AtomicSkillManager:
    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        atomic_skill_config: DictConfig,
        device=torch.device("cpu"),
        logger: Logger = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.atomic_skill_config = atomic_skill_config
        self.device = device
        self.logger = logger
        self.atomic_skill_list: list[AtomicSkill] = [None] * num_envs
        self.atomic_skill_type_list: list[str] = [None] * num_envs
        self.info_list: List[Dict[str, Any]] = [None] * self.num_envs

    def create_atomic_skill(
        self, atomic_skill_type: str, env_id: int, robot_name: str = None
    ):
        print(f"Creating atomic skill: {atomic_skill_type} for env {env_id}")
        if atomic_skill_type == "Reach":
            self.atomic_skill_list[env_id] = Reach(
                self.atomic_skill_config.Reach, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Reach"
        elif atomic_skill_type == "Grasp":
            self.atomic_skill_list[env_id] = Grasp(
                self.atomic_skill_config.Grasp, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Grasp"
        elif atomic_skill_type == "MobileReach":
            self.atomic_skill_list[env_id] = MobileReach(
                self.atomic_skill_config.MobileReach, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "MobileReach"
        elif atomic_skill_type == "RetractReach":
            self.atomic_skill_list[env_id] = RetractReach(
                self.atomic_skill_config.RetractReach, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "RetractReach"
        elif atomic_skill_type == "Push":
            self.atomic_skill_list[env_id] = Push(
                self.atomic_skill_config.Push, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Push"
        elif atomic_skill_type == "NavTo":
            self.atomic_skill_list[env_id] = NavTo(
                self.atomic_skill_config.NavTo, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "NavTo"
        elif atomic_skill_type == "Wave":
            self.atomic_skill_list[env_id] = Wave(
                self.atomic_skill_config.Wave, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Wave"
        elif atomic_skill_type == "DexGrasp":
            self.atomic_skill_list[env_id] = DexGrasp(
                self.atomic_skill_config.DexGrasp, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "DexGrasp"
        elif atomic_skill_type == "OpenDrawer":
            self.atomic_skill_list[env_id] = OpenDrawer(
                self.atomic_skill_config.OpenDrawer, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "OpenDrawer"
        elif atomic_skill_type == "DexOpenDrawer":
            self.atomic_skill_list[env_id] = DexOpenDrawer(
                self.atomic_skill_config.DexOpenDrawer, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "DexOpenDrawer"
        elif atomic_skill_type == "CloseDrawer":
            self.atomic_skill_list[env_id] = CloseDrawer(
                self.atomic_skill_config.CloseDrawer, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "CloseDrawer"
        elif atomic_skill_type == "LocoOpenDoor":
            self.atomic_skill_list[env_id] = LocoOpenDoor(
                self.atomic_skill_config.LocoOpenDoor, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "LocoOpenDoor"
        elif atomic_skill_type == "Dehatch":
            self.atomic_skill_list[env_id] = Dehatch(
                self.atomic_skill_config.Dehatch, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Dehatch"
        elif atomic_skill_type == "Fling":
            self.atomic_skill_list[env_id] = Fling(
                self.atomic_skill_config.Fling, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Fling"
        elif atomic_skill_type == "Fold":
            self.atomic_skill_list[env_id] = Fold(
                self.atomic_skill_config.Fold, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Fold"
        elif atomic_skill_type == "Lift":
            self.atomic_skill_list[env_id] = Lift(
                self.atomic_skill_config.Lift, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Lift"
        elif atomic_skill_type == "LocoBox":
            self.atomic_skill_list[env_id] = LocoBox(
                self.atomic_skill_config.LocoBox, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "LocoBox"
        elif atomic_skill_type == "BiGrasp":
            self.atomic_skill_list[env_id] = BiGrasp(
                self.atomic_skill_config.BiGrasp, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "BiGrasp"
        elif atomic_skill_type == "BiDexGrasp":
            self.atomic_skill_list[env_id] = BiDexGrasp(
                self.atomic_skill_config.BiDexGrasp, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "BiDexGrasp"
        elif atomic_skill_type == "Handover":
            self.atomic_skill_list[env_id] = Handover(
                self.atomic_skill_config.Handover, self.env, env_id, self.logger
            )
            self.atomic_skill_type_list[env_id] = "Handover"
        else:
            raise ValueError(f"Atomic skill type {atomic_skill_type} not supported.")

    def step(self, actions: List[List[str]], env_ids: Sequence[int]):
        # Note here we only return the action for env_ids
        # action is a list of [atomic_skill_type, obj_type, obj_name, obj_id]
        assert len(actions) == len(env_ids), (
            f"Action length should be equal to env_ids length, but got {len(actions)} and {len(env_ids)}"
        )
        output_action = []
        valid_env_ids = []
        failed_env_ids = []
        for i, env_id in enumerate(env_ids):
            if actions[i] is not None:  # New command is given
                if self.atomic_skill_list[env_id] is None:
                    assert self.atomic_skill_type_list[env_id] is None, (
                        f"[AtomicSkillManager] Atomic skill type {self.atomic_skill_type_list[env_id]} is not None, but atomic skill list is None"
                    )
                    self.create_atomic_skill(actions[i][0], env_id)
                    self.atomic_skill_list[env_id].reset(actions[i])
                else:
                    if actions[i][0] == self.atomic_skill_type_list[env_id]:
                        self.atomic_skill_list[env_id].refresh(actions[i])
                    else:
                        raise RuntimeError(
                            f"[AtomicSkillManager] Atomic skill {self.atomic_skill_type_list[env_id]} is running, but new command {actions[i][0]} is given."
                        )
            else:
                raise RuntimeError(
                    f"[AtomicSkillManager] Action is None for env {env_id}, but atomic skill list is not None"
                )
            action = self.atomic_skill_list[
                env_id
            ].step()  # here we still not support update target at realtime、
            if action == "Failed":
                failed_env_ids.append(env_id)
            elif action is not None:
                output_action.append(action)
                valid_env_ids.append(env_id)
        assert len(output_action) == len(valid_env_ids), (
            "Output action length should be equal to valid env ids length"
        )
        return output_action, valid_env_ids, failed_env_ids

    def update(self, info: Dict[str, Any]):
        # Note here we update the info for all environments
        for env_id in range(self.num_envs):
            if self.atomic_skill_type_list[env_id] is None:
                self.info_list[env_id] = None
                continue
            self.info_list[env_id] = self.atomic_skill_list[env_id].update(info)
            if (
                self.info_list[env_id]["finished"]
                or self.info_list[env_id]["truncated"] > 0
            ):
                print(f"Atomic skill of env {env_id} is finished or truncated")
                self.atomic_skill_type_list[env_id] = None
                self.atomic_skill_list[env_id] = None
        return self.info_list

    def get_manager_info(self):
        return {
            "atomic_skill_type_list": self.atomic_skill_type_list,
            "atomic_skill_list": self.atomic_skill_list,
        }

    def reset(self):
        return [None] * self.num_envs
