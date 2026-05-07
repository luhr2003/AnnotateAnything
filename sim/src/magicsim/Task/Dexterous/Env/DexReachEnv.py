from typing import Dict

import torch
from magicsim.Task.TableTop.Env.ReachEnv import ReachEnv as TableTopReachEnv


class DexReachEnv(TableTopReachEnv):
    """Reach environment for Panda XHand dexterous robot.

    Inherits from TableTop ReachEnv. Overrides process_action to handle XHand action space:
    arm (7D pose) + eef (12D hand joints). When action is 7D (target pose), appends
    default hand pose (zeros = open hand).
    """

    def process_action(self, action: torch.Tensor | list[Dict]):
        """Process action for XHand: 7D arm pose + 12D hand joints.

        When action is 7D (target pose for arm), append 12D zeros for hand (open).
        """
        if action is None:
            return None
        if action.shape[1] == 7:
            # Arm target pose (7D) + default hand joints (12D, open hand)
            hand_default = torch.zeros(
                (action.shape[0], 12), device=action.device, dtype=action.dtype
            )
            action = torch.cat([action, hand_default], dim=1)
        return action
