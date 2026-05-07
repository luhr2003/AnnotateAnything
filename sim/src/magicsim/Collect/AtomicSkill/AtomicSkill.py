from typing import Any, Dict

import torch

from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class AtomicSkill:
    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        self.config = config
        self.env = env
        self.logger = logger
        self.current_state = (
            None  # should be one of "ready", "running", "failed", "truncated"
        )
        self.current_command = None  # command that atomic skill get
        self.current_action = None  # action that atomic skill output
        self.env_id = env_id

    def reset(self, obj_type: str, obj_name: str, obj_id: int):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def update(self, info: Dict[str, Any]):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # IK goalset packing helper (shared by all single-arm grasp-like skills)
    # ------------------------------------------------------------------

    def pack_single_arm_goalset(self, arm_poses: torch.Tensor) -> torch.Tensor:
        """Pack a single-arm goalset into the unified ``(1, G, eef_num * 7)``
        shape accepted by :class:`IKPlanRequest` / :class:`DualIKPlanRequest`.

        Reads ``self.hand_id`` (set in the skill's ``reset``) and
        ``self.ik_server.eef_num`` (set by PlannerManager from YAML) —
        callers just pass the active arm's candidate poses.

        Inactive arms get NaN-xyz rows so the Server's NaN-as-disable path
        flips their :class:`ToolPoseCriteria` to ``disabled()`` for this
        solve only. See ``src/magicsim/Env/Planner/Services/README.md``
        §5 + §7.

        Args:
            arm_poses: 7-vec grasp candidates for the ACTIVE arm. Accepted
                shapes: ``(G, 7)`` or ``(1, G, 7)``. Values are
                ``[x, y, z, qw, qx, qy, qz]``.

        Returns:
            Goalset target shaped ``(1, G, eef_num * 7)``. Tool frames in
            slot order: right first, left second (MagicSim dual-arm
            convention). NaN rows mark inactive arms for this solve.

        Raises:
            ValueError: ``self.hand_id`` outside ``{0, 1}`` on a multi-EEF
                robot (this helper is for single-arm skills; both-arm
                goalsets should build the tensor directly).
            AttributeError: ``self.ik_server`` / ``self.hand_id`` not yet
                set — call ``reset`` + server resolve before using.
        """
        hand_id = int(self.hand_id)
        eef_num = int(getattr(self.ik_server, "eef_num", 1))

        if arm_poses.ndim == 2:
            arm_poses = arm_poses.unsqueeze(0)  # (1, G, 7)
        if arm_poses.ndim != 3 or arm_poses.shape[-1] != 7:
            raise ValueError(
                f"pack_single_arm_goalset: arm_poses must be (G, 7) or "
                f"(1, G, 7); got {tuple(arm_poses.shape)}"
            )
        N, G, _ = arm_poses.shape
        if eef_num == 1:
            # Single-arm robot: (N, G, 7) is already (N, G, L*7) with L=1.
            return arm_poses.contiguous()
        if hand_id not in (0, 1):
            raise ValueError(
                f"pack_single_arm_goalset: self.hand_id must be 0 (right) "
                f"or 1 (left) on a multi-EEF robot (eef_num={eef_num}); got "
                f"{hand_id}. Both-arm goalsets should build the "
                f"(1, G, eef_num*7) tensor directly."
            )
        # (N, G, eef_num, 7) — NaN-filled, then active arm's poses placed in.
        target = torch.full(
            (N, G, eef_num, 7),
            float("nan"),
            device=arm_poses.device,
            dtype=arm_poses.dtype,
        )
        target[:, :, hand_id, :] = arm_poses
        return target.reshape(N, G, eef_num * 7).contiguous()
