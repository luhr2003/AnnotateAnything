"""SonicPinkInverseKinematicsAction — Pink IK subclass that only solves and
writes the shared `SonicArmBuffer`; it does NOT write the sim.

In Scheme A (no decouple) the authoritative sim target is produced by SONIC
decoder and written for all 29 body DOF by `SonicWBCAction`. Pink IK's job
here is purely to supply the encoder's upper-body reference (17 waist+arm
DOFs in `PINK_CONTROLLED_JOINTS_IL` order) through `SonicArmBuffer`. This
mirrors the sonic reference `stage_hybrid_eval_magicsim.py:415-419` where
Pink IK solve feeds `joint_pos_future[..., upper_idx_il]` but the final
`set_joint_position_target` writes the decoder's 29-DOF output.

To achieve this, `apply_actions` reimplements the parent's solve stage
(`_compute_ik_solutions` + `_processed_actions` fill) and intentionally
skips `set_joint_position_target` and `_apply_gravity_compensation` (both
would touch sim).

Joint-order invariant (6th in plan.md): `find_joints()` returns joints
sorted by IL index ascending. For the 17 waist+arm joints, that order is:
    waist_yaw(12), waist_roll(13), waist_pitch(14),
    left_shoulder_pitch(15), right_shoulder_pitch(16)(*sic USD has L/R
    interleaved, so actual asc order depends on the spawn*) ...
If the resolved order differs from sonic's canonical `PINK_CONTROLLED_JOINTS_IL`
(left-arm-first), we install a permutation `_sonic_perm` to remap before
writing the buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from magicsim.Env.Robot.mdp.pink_actions_cfg import PinkInverseKinematicsActionCfg
from magicsim.Env.Robot.mdp.pink_task_space_actions import PinkInverseKinematicsAction

from magicsim.Env.Robot.Cfg.Humanoid.mdp.sonic_arm_buffer import SonicArmBuffer
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Sonic.G1.configs import (
    PINK_CONTROLLED_JOINTS_IL,
)

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv


class SonicPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink IK solver that feeds `SonicArmBuffer` and does **not** write sim."""

    def __init__(
        self,
        cfg: "SonicPinkInverseKinematicsActionCfg",
        env: "IsaacRLEnv",
    ):
        super().__init__(cfg, env)

        # --- Canonical-order permutation (assert or build) ---
        ik_names: list[str] = list(self._isaaclab_controlled_joint_names)
        assert len(ik_names) == 17, (
            f"SonicPinkInverseKinematicsAction expects 17 controlled joints, got {len(ik_names)}"
        )
        if ik_names == PINK_CONTROLLED_JOINTS_IL:
            self._sonic_perm: torch.Tensor | None = None
        else:
            missing = set(PINK_CONTROLLED_JOINTS_IL) - set(ik_names)
            assert not missing, (
                f"Pink IK cfg is missing SONIC joints: {missing}. "
                f"resolved={ik_names}, expected={PINK_CONTROLLED_JOINTS_IL}"
            )
            perm = [ik_names.index(n) for n in PINK_CONTROLLED_JOINTS_IL]
            self._sonic_perm = torch.as_tensor(
                perm, dtype=torch.long, device=self.device
            )
            # Non-error — just flagging the slow path.
            print(
                f"[SonicPinkInverseKinematicsAction] Pink IK output order differs from "
                f"PINK_CONTROLLED_JOINTS_IL; installed permutation of len={len(perm)}."
            )

        # --- SonicArmBuffer (singleton; first to create wins) ---
        self._arm_buffer = SonicArmBuffer.get_or_create(self.num_envs, self.device)

    def apply_actions(self) -> None:
        """Solve IK, write the `SonicArmBuffer`, **skip** sim writes.

        Reimplements the parent's solve block (step_count++, IK solve every
        `decimation` steps, fill `_processed_actions`) but intentionally omits
        `set_joint_position_target` and `_apply_gravity_compensation` — those
        would touch the sim joints that SONIC decoder is responsible for.

        NaN-skipped envs (Pink IK has `fallback_to_current=False`) get their
        `_processed_actions` rows filled with the current sim joint_pos for
        the 17 controlled joints, so the buffer always contains a valid "hold
        current pose" target rather than stale zeros.
        """
        # --- Parent solve block (minus sim write) ---
        self._step_count += 1

        if len(self._env_ids) > 0 and (self._step_count - 1) % self.cfg.decimation == 0:
            ik_joint_positions = self._compute_ik_solutions()
            if self.hand_joint_dim > 0:
                all_joint_positions = torch.cat(
                    (ik_joint_positions, self._target_hand_joint_positions), dim=1
                )
            else:
                all_joint_positions = ik_joint_positions
            self._processed_actions[self._env_ids] = all_joint_positions

        # --- Fill skipped envs with current joint_pos so the buffer is valid ---
        num_ctrl = len(self._isaaclab_controlled_joint_ids)  # == 17 for SONIC G1
        if len(self._env_ids) < self.num_envs:
            skipped_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            skipped_mask[self._env_ids] = False
            current_jp = self._asset.data.joint_pos[
                :, self._isaaclab_controlled_joint_ids
            ]  # [N, 17]
            self._processed_actions[skipped_mask, :num_ctrl] = current_jp[skipped_mask]

        # --- Write SonicArmBuffer (canonical PINK_CONTROLLED_JOINTS_IL order) ---
        if self._sonic_perm is None:
            targets_17 = self._processed_actions[:, :num_ctrl]
        else:
            targets_17 = self._processed_actions[:, :num_ctrl][:, self._sonic_perm]
        self._arm_buffer.write(slice(None), targets_17)

        # NOTE (Scheme A): no `self._asset.set_joint_position_target(...)` call
        # and no `_apply_gravity_compensation()` — the sim body targets (all 29
        # DOF) are owned by SonicWBCAction.


@configclass
class SonicPinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Pink IK cfg variant that selects the buffer-writing subclass."""

    class_type: type[ActionTerm] = SonicPinkInverseKinematicsAction
