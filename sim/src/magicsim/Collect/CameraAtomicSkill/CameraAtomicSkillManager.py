from typing import Any, Dict, List, Sequence

import carb
import torch
from magicsim.Collect.CameraAtomicSkill.CameraAtomicSkill import CameraAtomicSkill
from magicsim.Collect.CameraAtomicSkill.GoTo import GoTo
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class CameraAtomicSkillManager:
    """Dispatcher/manager for camera atomic skills (GoTo etc.)."""

    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        camera_atomic_skill_config: DictConfig,
        device: torch.device = torch.device("cpu"),
        logger: Logger | None = None,
    ):
        self.env = env
        self.num_envs = num_envs
        self.camera_atomic_skill_config = camera_atomic_skill_config
        self.device = device
        self.logger = logger
        self.camera_atomic_skill_list: list[CameraAtomicSkill | None] = [
            None
        ] * num_envs
        self.camera_atomic_skill_type_list: list[str | None] = [None] * num_envs
        self.info_list: List[Dict[str, Any] | None] = [None] * num_envs

    def create_camera_atomic_skill(
        self, skill_type: str, env_id: int, camera_name: str
    ):
        if skill_type == "GoTo":
            cfg = self.camera_atomic_skill_config.GoTo
            self.camera_atomic_skill_list[env_id] = GoTo(
                cfg,
                self.env,
                env_id,
                self.logger,
            )
            self.camera_atomic_skill_list[env_id].camera_name = camera_name
            self.camera_atomic_skill_type_list[env_id] = "GoTo"
        else:
            raise ValueError(
                f"Camera atomic skill type {skill_type} is not supported for env {env_id}."
            )

    def step(
        self, actions: List[List[Any]] | None, env_ids: Sequence[int]
    ) -> tuple[List[Dict[str, Any] | None], List[int], List[int]]:
        """Action format: [skill_type, camera_name, obj_type, obj_name, obj_id]."""
        if actions is None:
            actions = [None] * len(env_ids)
        env_ids_list = (
            env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        )
        output_action: list[Dict[str, Any] | None] = [None] * len(env_ids_list)
        valid_env_ids: list[int] = env_ids_list.copy()
        failed_env_ids: list[int] = []

        for i, env_id in enumerate(env_ids_list):
            action_spec = actions[i]
            if action_spec is not None:
                skill_type, camera_name, obj_type, obj_name, obj_id = action_spec
                # Align behavior with robot AtomicSkillManager:
                # - If no skill instance exists for this env, create and reset.
                # - If the same skill type is already running, call refresh to update target.
                # - If a different skill type is running, raise an error.
                if self.camera_atomic_skill_type_list[env_id] is None:
                    self.create_camera_atomic_skill(skill_type, env_id, camera_name)
                    self.camera_atomic_skill_list[env_id].reset(
                        camera_name=camera_name,
                        obj_type=obj_type,
                        obj_name=obj_name,
                        obj_id=obj_id,
                    )
                else:
                    if skill_type == self.camera_atomic_skill_type_list[env_id]:
                        self.camera_atomic_skill_list[env_id].refresh(
                            camera_name=camera_name,
                            obj_type=obj_type,
                            obj_name=obj_name,
                            obj_id=obj_id,
                        )
                    else:
                        raise RuntimeError(
                            f"Camera atomic skill {self.camera_atomic_skill_type_list[env_id]} is running, "
                            f"but new command {skill_type} is given for env {env_id}."
                        )

            skill = self.camera_atomic_skill_list[env_id]
            if skill is None:
                continue

            action = skill.step()
            if action is not None:
                carb.log_info(f"[CameraAtomic] env {env_id} action: {action}")

            if action is None:
                failed_env_ids.append(env_id)
                continue

            navto_payload = action.get("NavTo") if isinstance(action, dict) else action
            if navto_payload is None:
                failed_env_ids.append(env_id)
                continue

            output_action[i] = {"NavTo": navto_payload}

        return output_action, valid_env_ids, failed_env_ids

    def update(self, info: Dict[str, Any]):
        for env_id in range(self.num_envs):
            if self.camera_atomic_skill_type_list[env_id] is None:
                self.info_list[env_id] = None
                continue
            updated_info = self.camera_atomic_skill_list[env_id].update(info)
            self.info_list[env_id] = updated_info
            if updated_info["finished"] or updated_info["truncated"] > 0:
                self.camera_atomic_skill_type_list[env_id] = None
                self.camera_atomic_skill_list[env_id] = None
        return self.info_list

    def reset(self):
        self.camera_atomic_skill_list = [None] * self.num_envs
        self.camera_atomic_skill_type_list = [None] * self.num_envs
        self.info_list = [None] * self.num_envs
        return [None] * self.num_envs
