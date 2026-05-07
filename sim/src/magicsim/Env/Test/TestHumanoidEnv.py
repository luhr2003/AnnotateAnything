import gymnasium as gym
from omegaconf import DictConfig
import hydra
import torch
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="g1_fixed_hand_config"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # G1_FixedHand raw action layout (planner=default for body/arm/eef):
    #   7  Homie WBC: [vx, vy, ang_vel, base_height, roll, pitch, yaw]
    #  14  Pink IK: 2 × (xyz + xyzw) — NaN holds current pose
    #   2  placeholder eef (waist NaN no-op — divisible by max_eef_num=2)
    base_cmd = torch.zeros(env.num_envs, 7)
    base_cmd[:, 0] = 0.3  # vx — walk forward at 0.3 m/s
    base_cmd[:, 3] = 0.78  # base_height (matches init_state pelvis z)

    while True:
        action = torch.cat(
            [
                base_cmd,  # 7  Homie WBC
                torch.full((env.num_envs, 14), torch.nan),  # 14 Pink IK (NaN = hold)
                torch.full((env.num_envs, 2), torch.nan),  # 2  eef placeholder (NaN)
            ],
            dim=1,
        )
        env.step(action=action)


if __name__ == "__main__":
    main()
