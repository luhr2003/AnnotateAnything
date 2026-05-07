import gymnasium as gym
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time
import torch


def generate_global_paths(start_xy, end_points, num_points=20, device="cuda"):
    """
    start_xy: tuple/list (x0, y0)
    end_points: list[(x,y)] for each env
    returns: tensor [B, num_points, 2]
    """
    start = torch.tensor(start_xy, dtype=torch.float32, device=device)  # [2]

    all_paths = []
    for end in end_points:
        end_xy = torch.tensor(end, dtype=torch.float32, device=device)  # [2]

        # t: [num_points]
        t = torch.linspace(0.0, 1.0, num_points, device=device)

        # 插值： path[k] = start*(1−t[k]) + end*t[k]
        path = start * (1 - t).unsqueeze(1) + end_xy * t.unsqueeze(1)  # [20,2]

        all_paths.append(path)

    # stack -> [B, 20, 2]
    return torch.stack(all_paths, dim=0)


@hydra.main(version_base=None, config_path=MAGICSIM_CONF, config_name="dwb_config")
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    OmegaConf.set_struct(cfg, False)
    cfg.sim.seed = new_seed
    OmegaConf.set_struct(cfg, True)

    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # 四个并行动环境
    target_env_ids = [0, 1, 2, 3]

    # 四个不同终点
    end_points = [
        (3.0, 0.0),  # env0
        (3.0, -1.0),  # env1
        (3.0, 0.5),  # env2
        (3.0, 1.0),  # env3
    ]
    # 全局路径插值（4 个 env，20 个点）
    path_tensor = generate_global_paths(
        start_xy=(-2.0, 0.0),
        end_points=end_points,
        num_points=20,
        device=env.device,
    )

    # 主循环
    for i in range(1000):
        env.step(action=path_tensor, env_ids=target_env_ids)
        # for j in range(5):
        #     env.sim.sim_step()


if __name__ == "__main__":
    main()
