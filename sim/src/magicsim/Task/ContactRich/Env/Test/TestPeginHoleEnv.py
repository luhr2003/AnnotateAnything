from magicsim.Task.ContactRich.Env.ContactRichEnv import ContactRichEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(version_base=None, config_path="../../Conf", config_name="task_peg_in_hole")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: ContactRichEnv = gym.make(
        "ContactRichEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    for i in range(150):
        rand_action_batched = env.sample_actions(batched=True)
        env.step(action=rand_action_batched)

    while True:
        rand_action_batched = env.sample_actions(batched=True, env_ids=[0, 3])
        env.step(action=rand_action_batched, env_ids=[0, 3])


if __name__ == "__main__":
    main()
