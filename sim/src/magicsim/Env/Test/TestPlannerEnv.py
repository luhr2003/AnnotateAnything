import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="robot_config")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(batched=True)
        env.step(action=rand_action_batched)

    env.reset_idx()
    # _print_joint_states(env, label="After env.reset_idx() (all envs)")

    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 2, 3]
        )
        env.step(action=rand_action_batched, env_ids=[0, 2, 3])

    env.reset_idx([0, 1], seed=[50, 60])
    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 1]
        )
        env.step(action=rand_action_batched, env_ids=[0, 1])

    env.reset_idx([2, 3], seed=[50, 80])

    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[2, 3]
        )
        env.step(action=rand_action_batched, env_ids=[2, 3])

    env.reset_idx([0, 3])

    for i in range(50):
        env.sim.sim_step()
    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 3]
        )
        env.step(action=rand_action_batched, env_ids=[0, 3])

    while True:
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=[0, 3]
        )
        env.step(action=rand_action_batched, env_ids=[0, 3])


if __name__ == "__main__":
    main()
