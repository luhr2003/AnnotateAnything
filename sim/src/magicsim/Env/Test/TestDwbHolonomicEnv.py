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


def interpolate_path(start, end, num_points):
    """线性插值路径，返回 [num_points, 2]"""
    start = torch.tensor(start, dtype=torch.float32)
    end = torch.tensor(end, dtype=torch.float32)

    t = torch.linspace(0, 1, num_points).unsqueeze(1)  # [N,1]
    path = (1 - t) * start + t * end  # [N,2]
    return path


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="dwb_holonomic_config"
)
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

    # 四个并行环境
    target_env_ids = [0, 1, 2, 3]

    # 四个不同终点
    end_points = [
        (3.0, 0.0),  # env0
        (3.0, -1.0),  # env1
        (3.0, 0.5),  # env2
        (3.0, 1.0),  # env3
    ]

    # 起点
    start = (-2.0, 0.0)

    # 固定的后八个参数
    tail = torch.tensor(
        [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    num_points = 20

    # 每个 env 生成 [20,10] 的路径
    path_tensor_list = []

    for end in end_points:
        # 插值前两个维度 (x,y)
        xy_path = interpolate_path(start, end, num_points)  # [20,2]

        # 把 tail 扩展成 [20,8]
        tail_rep = tail.unsqueeze(0).repeat(num_points, 1)  # [20,8]

        # 拼成 [20,10]
        full_path = torch.cat([xy_path, tail_rep], dim=1)  # [20,10]

        path_tensor_list.append(xy_path)

    # 最终堆成 [4,20,10]
    path_tensor = torch.stack(path_tensor_list, dim=0).to(env.device)

    print("Final path_tensor shape:", path_tensor.shape)

    # 主循环
    for i in range(1000):
        env.step(action=path_tensor, env_ids=target_env_ids)


if __name__ == "__main__":
    main()
