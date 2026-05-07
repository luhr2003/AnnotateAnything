from magicsim.Task.Camera.Env.GoToEnv import GoToEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from loguru import logger as log


@hydra.main(version_base=None, config_path="../../Conf", config_name="goto_env")
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: GoToEnv = gym.make("GoToEnv-V0", config=cfg, cli_args=None, logger=logger)
    obs, info = env.reset()

    while 1:
        target_pose = obs["privilege_obs"]["target_pose"]
        # Convert target_pose tensor to dict format expected by step()
        camera_name = env._get_primary_camera_name()
        camera_action = {camera_name: target_pose}
        obs, reward, terminated, truncated, info = env.step(camera_action=camera_action)


if __name__ == "__main__":
    main()
