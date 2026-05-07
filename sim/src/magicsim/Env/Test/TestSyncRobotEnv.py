import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="robot_config")
def main(cfg: DictConfig):
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # Run some simulation steps
    for i in range(3):
        for i in range(50):
            rand_action_batched = env.robot_manager.sample_actions(batched=True)
            env.step(action=rand_action_batched)

        env.reset_idx()

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

    # Collect and save robot data
    robot_names = list(env.robot_manager.robots.keys())
    if not robot_names:
        raise ValueError("No robots found in RobotManager!")

    robot_name = robot_names[0]

    target_envs = [0, 3]

    for i in range(50):
        env.sim.sim_step()

    for i in range(150):
        rand_action_batched = env.robot_manager.sample_actions(
            batched=True, env_ids=target_envs
        )
        env.step(action=rand_action_batched, env_ids=target_envs)


if __name__ == "__main__":
    main()
