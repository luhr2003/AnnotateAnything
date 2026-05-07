from typing import Any, Dict, Sequence

import torch
from magicsim.Task.TableTop.Env.GraspEnv import GraspEnv


class WaveEnv(GraspEnv):
    """
    Wave Environment for Robot Tasks.

    Extends GraspEnv to support wave/jitter phase after successful grasp:
    - When object is successfully grasped and lifted, allow additional jitter steps
    - Similar to RandomReachEnv but for grasped objects
    """

    def __init__(self, config, cli_args, logger):
        super().__init__(config, cli_args, logger)

        # Extra jitter steps after successful grasp
        # Default value can be overridden by config.extra_jitter_steps
        # Should match atomic_skill.Wave.jitter_steps for consistency
        self.extra_jitter_steps: int = int(getattr(config, "extra_jitter_steps", 300))

        # Track when grasp was completed (eef close to mug and mug lifted)
        # -1 means grasp not yet completed
        self._grasp_completed_step = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and clear grasp completion tracking."""
        obs, info = super().reset(seed=seed, options=options)
        self._grasp_completed_step.fill_(-1)
        return obs, info

    def reset_idx(
        self,
        env_ids: Sequence[int] = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset specific environments and clear their grasp completion tracking."""
        obs, info = super().reset_idx(env_ids=env_ids, seed=seed, options=options)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._grasp_completed_step[env_ids] = -1
        return obs, info

    def get_termination(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Modified termination logic for Wave:
        - Never terminate early - let Wave AtomicSkill control completion
        - Only truncate if mug falls off table
        - Wave AtomicSkill will signal completion through finished=True
        - This ensures Wave AtomicSkill can complete all phases including wave jittering
        """
        eef_pos = self.get_eef_pose()[:, :3]
        mug_pos = self.get_mug_pose()[:, :3]
        mug_z = mug_pos[:, 2]

        # Never terminate based on grasp success - let Wave AtomicSkill control when to finish
        # Wave AtomicSkill will complete all phases (pre_grasp -> grasp -> close_gripper -> retrieval -> wave)
        # and signal completion through finished=True
        termination = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Truncated: mug falls off table (same as GraspEnv)
        truncated = mug_z < 0.8

        return termination, truncated
