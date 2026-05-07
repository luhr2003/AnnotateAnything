import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from isaacsim.core.api.objects import FixedCuboid
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import time
import torch


def build_straight_path_4d(
    start_xy=(-2.0, 0.0),
    goal_xy=(4.0, 0.0),
    heading=0.0,
    height=0.75,
    num_waypoints=30,
    num_envs=1,
    device="cpu",
    dtype=torch.float32,
):
    """
    构造一条直线路径，4 维：(x, y, heading, height)。
    与 dwb_g1.yaml 中起始位置 (0, 0, 0.75) 一致，终点 (3, 0)。

    Returns:
        target_pose: [num_envs, num_waypoints, 4]
    """
    x0, y0 = start_xy
    xg, yg = goal_xy

    xs = torch.linspace(x0, xg, num_waypoints, device=device, dtype=dtype)
    ys = torch.linspace(y0, yg, num_waypoints, device=device, dtype=dtype)
    yaws = torch.full((num_waypoints,), heading, device=device, dtype=dtype)
    heights = torch.full((num_waypoints,), height, device=device, dtype=dtype)

    # [Tp, 4]
    path = torch.stack([xs, ys, yaws, heights], dim=-1)
    # [B, Tp, 4]
    target_pose = path.unsqueeze(0).expand(num_envs, -1, -1).clone()
    return target_pose


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="dwb_humanoid_config"
)
def main(cfg: DictConfig):
    new_seed = int(time.time())
    print(f"Initializing with seed: {new_seed}")

    cfg.sim.seed = new_seed

    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    # 与 dwb_g1.yaml 一致：起点 (0, 0, 0.75)，终点 (3, 0)，4 维 (x, y, heading, height)
    num_envs = getattr(env, "num_envs", 4) or 4
    target_env_ids = list(range(num_envs))  # [0] 当 num_envs=1
    path_tensor = build_straight_path_4d(
        start_xy=(-2, 0.0),
        goal_xy=(3.0, 0.0),
        heading=0.0,
        height=0.75,
        num_waypoints=20,
        num_envs=num_envs,
    )
    action_padding = torch.full(
        (num_envs, path_tensor.shape[1], 10),
        float("nan"),
        device=env.device,
        dtype=torch.float32,
    )
    action_padding = torch.cat(
        [
            action_padding,
            torch.ones(
                (num_envs, path_tensor.shape[1], 1),
                device=env.device,
                dtype=torch.float32,
            )
            * 2,
        ],
        dim=-1,
    )
    action_padding = torch.cat(
        [
            action_padding,
            torch.full(
                (num_envs, path_tensor.shape[1], 28),
                torch.nan,
                device=env.device,
                dtype=torch.float32,
            ),
        ],
        dim=-1,
    )
    action = torch.cat([path_tensor, action_padding], dim=-1)
    # 主循环
    for i in range(5):
        env.step(action=action, env_ids=target_env_ids)

    cube = FixedCuboid(
        prim_path="/World/Test/Cube",
        scale=(0.2, 0.2, 10.3),
        position=(1.5, 0.0, 0.15),
    )
    # wall = FixedCuboid(
    #     prim_path="/World/Test/Cube",
    #     scale=(0.3, 2.3, 10.3),
    #     position=(1.0, 0.0, 0.15),
    # )
    for i in range(10000):
        env.step(action=action, env_ids=target_env_ids)


if __name__ == "__main__":
    main()
