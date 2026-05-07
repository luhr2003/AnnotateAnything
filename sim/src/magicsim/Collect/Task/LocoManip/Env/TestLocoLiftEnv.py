"""
Test LocoLift with AutoCollect: bimanual forearm squeeze on a bin.

Scene: taller table + narrow bin (``loco_lift_env``) with the wide
mobile robot pos range (``single_g1``). Atomic skill :class:`Lift`
drives three phases via :class:`MobileMoveL` with ``hand_id=-1``:

    pre_grasp → squeeze → lift

Wrist targets come from the bin's local AABB (``LocoLiftEnv.
get_target_bbox_half_extents``) + current world pose. Hands stay fully
closed throughout — the squeeze is forearm-driven. Termination uses
``LocoLiftEnv.get_termination`` (baseline-relative lift threshold).
"""

from omegaconf import DictConfig
import hydra
from loguru import logger as log
from magicsim.Env.Utils.file import Logger
import gymnasium as gym
from magicsim.StardardEnv.Robot.AutoCollectEnv import AutoCollectEnv

# Import to trigger gym.register for LocoLiftEnv-V0.
import magicsim.Task.LocoManip.Env  # noqa: F401


TASK_STRING_DICT = {"LocoLift": 1.0}


@hydra.main(
    version_base=None,
    config_path="../Conf",
    config_name="loco_lift",
)
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
