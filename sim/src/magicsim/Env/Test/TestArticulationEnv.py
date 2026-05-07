import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time
from omegaconf import OmegaConf


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="articulation_config"
)
def main(cfg: DictConfig):
    print(cfg)
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()
    for i in range(50):
        env.sim.sim_step()

    # for i in range(4):
    #     articulation_obj = env.scene_manager.articulation_objects[i][
    #         "articulation_items"
    #     ][0]
    #     upper_joint_pos = articulation_obj.upper_joint_positions
    #     if hasattr(upper_joint_pos, "cpu"):
    #         upper_joint_pos = upper_joint_pos.cpu().tolist()
    #     else:
    #         upper_joint_pos = list(upper_joint_pos)
    #     articulation_obj.set_current_joint_positions(upper_joint_pos)
    #     current_joint = articulation_obj.get_current_joint_positions()
    #     print(f"current_joint: {current_joint}")
    #     for i in range(50):
    #         env.sim.sim_step()

    for j in range(5):
        print(f"Env Reset {j + 1}")
        for _ in range(50):
            env.sim.sim_step()
        env.reset_idx()

    while True:
        env.sim.sim_step()


if __name__ == "__main__":
    main()
