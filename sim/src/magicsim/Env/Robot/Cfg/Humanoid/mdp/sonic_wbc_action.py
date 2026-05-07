"""SonicWBCAction — full-body ActionTerm for the SONIC hybrid IK pipeline.

Per policy tick:
    1. Read the 5D action ``[vx, vy, ang_vel, height, mode]``; NaN entries
       fall back to the previous value.
    2. Pull the latest 17-DOF arm+waist target from ``SonicArmBuffer`` — this
       is the upper-body reference the SONIC encoder consumes.
    3. Pack observations via ``prepare_observations``.
    4. ``SonicPolicy.set_goal / set_observation / get_action`` → 29-DOF body
       target in IL order.
    5. ``apply_actions`` writes **all 29 body DOF** to sim, matching the sonic
       reference ``stage_hybrid_eval_magicsim.py:415-419`` (``target_body_il``
       covers legs + waist + arms). Pink IK is NOT authoritative on sim joint
       targets; its role is to provide the upper-body reference via the buffer.

This is "Scheme A" (no decouple): the decoder outputs a coordinated 29-DOF
action and we apply it whole, preserving the training-time coupling between
lower-body gait and upper-body arm swing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from gymnasium import spaces

from isaaclab.assets.articulation import Articulation

from magicsim.Env.Robot.Cfg.Humanoid.mdp.sonic_arm_buffer import SonicArmBuffer
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Sonic.policy_constants import (
    ACTION_DIM,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Sonic.utils import (
    prepare_observations,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.wbc_policy_factory import (
    get_wbc_policy,
)
from magicsim.Env.Robot.mdp.action_manager import ActionTerm

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv

    from magicsim.Env.Robot.Cfg.Humanoid.mdp.sonic_wbc_action_cfg import (
        SonicWBCActionCfg,
    )


class SonicWBCAction(ActionTerm):
    """SONIC WBC action term — applies all 29 body DOF each tick."""

    cfg: "SonicWBCActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "SonicWBCActionCfg", env: "IsaacRLEnv"):
        self.step_count = 0
        super().__init__(cfg, env)

        # --- Raw actions (5D) ---
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        # Defaults: height = -1 → planner default; mode = -1 → AUTO
        self._raw_actions[:, 3] = -1.0  # height
        self._raw_actions[:, 4] = -1.0  # mode

        # --- WBC policy (SonicPolicy) ---
        self.wbc_policy = get_wbc_policy(
            self.cfg.robot_type, self.cfg.wbc_version, self.num_envs
        )
        # `_asset` is already valid after `super().__init__`; bind here so
        # SonicPolicy can resolve runtime joint indices / default_angles /
        # action_scale and build its ORT sessions.
        self.wbc_policy.bind_articulation(self._asset.data)

        # --- Joint index bookkeeping (reuse SonicPolicy's resolution) ---
        self._leg_idx_il = self.wbc_policy.leg_idx_il  # [12] body-IL index
        self._body_idx_full = self.wbc_policy.body_idx_full  # [29] USD-full index
        self._upper_default_il = self.wbc_policy.upper_default_il  # [17]

        # USD-full joint ids for the 29 body DOFs (Scheme A applies all 29).
        # Order matches body-IL (runtime breadth-first) — same ordering as the
        # 29-dim `target_il` returned by SonicPolicy.get_action().
        self._body_joint_ids_sim: list[int] = self._body_idx_full.tolist()

        # --- Processed action buffer (29-DOF body target) ---
        self._processed_actions = torch.zeros(
            self.num_envs, len(self._body_joint_ids_sim), device=self.device
        )

        # --- SonicArmBuffer (same singleton as SonicPinkInverseKinematicsAction) ---
        # Whichever ActionTerm `__init__`s first creates the buffer.
        self._arm_buffer = SonicArmBuffer.get_or_create(self.num_envs, self.device)
        # Seed with default arm pose so the first SONIC tick doesn't read zeros.
        self._arm_buffer.reset(
            slice(None), self._upper_default_il.unsqueeze(0).expand(self.num_envs, -1)
        )

        # --- Action space ---
        assert self.cfg.action_space.shape[0] == 2, (
            f"Expected action_space shape (2, {self.action_dim}), got {self.cfg.action_space.shape}"
        )
        assert self.cfg.action_space.shape[1] == self.action_dim, (
            f"Expected action_space shape (2, {self.action_dim}), got {self.cfg.action_space.shape}"
        )
        self._action_space = spaces.Box(
            low=self.cfg.action_space[0].cpu().numpy(),
            high=self.cfg.action_space[1].cpu().numpy(),
        )

    # ================================================================
    # Properties
    # ================================================================
    @property
    def action_dim(self) -> int:
        return ACTION_DIM  # 5

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def get_wbc_version(self):
        return self.cfg.wbc_version

    @property
    def get_wbc_policy(self):
        return self.wbc_policy

    # ================================================================
    # Operations
    # ================================================================
    def process_actions(
        self, actions: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Gated by `decimation` — SONIC runs once per policy tick."""
        if self.step_count % self.cfg.decimation != 0:
            return

        # NaN → fall back to the previous action (same policy as HomieWBCAction).
        work = actions[:, : self.action_dim].clone()
        prev = self._raw_actions if env_ids is None else self._raw_actions[env_ids]
        nan_mask = torch.isnan(work)
        if nan_mask.any():
            work[nan_mask] = prev[nan_mask]

        if env_ids is None:
            self._raw_actions[:] = work
        else:
            self._raw_actions[env_ids] = work

        # --- Build goal + observation for SonicPolicy ---
        self.wbc_policy.set_goal({"action": self._raw_actions})

        obs = prepare_observations(self._asset.data, self._body_idx_full)
        obs["arm_targets_17"] = self._arm_buffer.read_latest()  # [N, 17]
        self.wbc_policy.set_observation(obs)

        # --- Run SONIC (planner + encoder + decoder) ---
        target_il = self.wbc_policy.get_action()  # [N, 29] body-IL

        # --- Scheme A: write the full 29-DOF body target ---
        self._processed_actions[:] = target_il

    def apply_actions(self):
        """Apply all 29 body DOF every sim sub-step."""
        self.step_count += 1
        self._asset.set_joint_position_target(
            self._processed_actions, self._body_joint_ids_sim
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Raw action: zeros with height=-1 (planner default) and mode=-1 (AUTO).
        if env_ids is None:
            self._raw_actions.zero_()
            self._raw_actions[:, 3] = -1.0
            self._raw_actions[:, 4] = -1.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._raw_actions[env_ids, 3] = -1.0
            self._raw_actions[env_ids, 4] = -1.0

        # Reset SonicPolicy (planner cache, ring buffers, etc.). We always do a
        # full-policy reset; SonicPolicy itself ignores `env_ids` and rebuilds
        # everything — simple and matches HomiePolicy.reset style.
        env_ids_t = (
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if env_ids is not None
            else None
        )
        self.wbc_policy.reset(env_ids=env_ids_t)

        # Reset SonicArmBuffer to the default arm pose for the requested envs.
        reset_slice = slice(None) if env_ids is None else env_ids_t
        default_arm = self._upper_default_il.unsqueeze(0).expand(
            self.num_envs if env_ids is None else len(env_ids), -1
        )
        self._arm_buffer.reset(reset_slice, default_arm)

        self.step_count = 0
