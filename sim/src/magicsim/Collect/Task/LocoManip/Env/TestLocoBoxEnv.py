"""
Run :class:`LocoBoxEnv` end-to-end through AutoCollect.

Bimanual ground-box squeeze on the rigid-rubber-hand G1
(``g1_fixed_hand``). Atomic skill :class:`LocoBox` drives three phases
through the global planners:

    pre_grasp (RetractMoveL) → squeeze (MoveL) → lift (MoveL)

The pre-grasp planner uses the ``g1_fixed_hand`` move_strategy
(``clip_height=0.4``, ``lock_fwd_offset=-0.15``), so the pelvis stays
crouched while the robot walks up to the box. Squeeze and lift run with
locked base + linear-interp; the box is in the obstacle ignore set so
MotionGen would refuse anyway. No hand DoFs — squeeze is forearm-driven.
"""

from omegaconf import DictConfig
import gymnasium as gym
import hydra
from loguru import logger as log

from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv

# Import to trigger gym.register for LocoBoxEnv-V0.
import magicsim.Task.LocoManip.Env  # noqa: F401


TASK_STRING_DICT = {"LocoBox": 1.0}


@hydra.main(version_base=None, config_path="../Conf", config_name="loco_box")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: AutoCollectEnv = gym.make(
        "AutoCollectEnv-V0",
        task_string=TASK_STRING_DICT,
        config=cfg,
        cli_args=None,
        logger=logger,
    )
    env.start_collect()


if __name__ == "__main__":
    main()
