"""Drive Vega 1P + Sharpa forward at vx=1 m/s; arm/hand actions = NaN."""

import gymnasium as gym
import torch
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


def build_forward_action(env: SyncRobotEnv) -> dict:
    device = env.sim.device
    n = env.num_envs
    actions = {}
    for robot_name, am in env.robot_manager.action_managers.items():
        per_robot = {}
        for term_name, term in am._terms.items():
            dim = term.action_dim
            if term_name == "base_action":
                vec = torch.zeros((n, dim), device=device, dtype=torch.float32)
                vec[:, 0] = 0.0  # base-frame [vx_b, vy_b, wz_b]
            else:
                vec = torch.full(
                    (n, dim), float("nan"), device=device, dtype=torch.float32
                )
            per_robot[term_name] = vec
        actions[robot_name] = per_robot
    print(actions)
    return actions


def print_state(env: SyncRobotEnv, step_idx: int):
    for state_dict in env.robot_manager.get_robot_state():
        for robot_name, state in state_dict.items():
            print(f"=== Step {step_idx}: {robot_name} ===")
            print(f"  base_pos     : {state['base_pos'][0].cpu().numpy()}")
            print(f"  base_lin_vel : {state['base_lin_vel'][0].cpu().numpy()}")
            print(f"  base joints  : {state['joint_pos'][0, :3].cpu().numpy()}")


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="vega1pSharpabase"
)
def main(cfg: DictConfig):
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=Logger("Env", log)
    )
    env.reset()
    print_state(env, -1)
    i = 0
    while 1:
        env.step(action=build_forward_action(env))
        i += 1
        if i % 30 == 0:
            print_state(env, i)
    env.close()


if __name__ == "__main__":
    main()
