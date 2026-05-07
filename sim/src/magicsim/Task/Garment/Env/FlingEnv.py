from typing import Dict, Sequence

import torch

from magicsim.Task.Garment.Env.GarmentFoldEnv import GarmentFoldEnv


class FlingEnv(GarmentFoldEnv):
    """Dynamic garment flinging environment.

    Inherits dual-Franka + garment scene from GarmentFoldEnv. The task runs
    open-ended (no automatic termination), and external auto-collect skills
    drive the actual fling trajectory.
    """

    def get_done(self, env_id: int) -> dict:
        base = super().get_done(env_id)
        base["is_done"] = False
        return base

    def get_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, truncated

    def get_description(self) -> str:
        return "Fling the garment with dual Franka arms"

    def get_reward(
        self,
        action: torch.Tensor | list[Dict],
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
