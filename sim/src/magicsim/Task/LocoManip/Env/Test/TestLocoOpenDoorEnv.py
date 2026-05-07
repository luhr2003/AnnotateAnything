"""Test LocoOpenDoorEnv: g1 at (0,0,0), door at (0, 1.5, 0). Get door trajectories and visualize one (approach + pull) via draw_grasp_samples_as_axes."""

import torch
from magicsim.Task.LocoManip.Env.LocoOpenDoorEnv import LocoOpenDoorEnv
import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Utils.file import Logger
from magicsim.Env.Utils.viz import draw_grasp_samples_as_axes
from loguru import logger as log


@hydra.main(
    version_base=None, config_path="../../Conf", config_name="loco_open_door_env"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: LocoOpenDoorEnv = gym.make(
        "LocoOpenDoorEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    obs, info = env.reset()

    trajs = env.get_door_trajectories(
        env_id=0,
        annotation_name="dex3_1_open_by_handle_trajectory",
        joint_id=-1,
    )
    print(f"Door trajectory keys: {list(trajs.keys())}")
    if trajs:
        seg = trajs[1]
        clear_existing = True
        for phase in ("approach", "pull"):
            t = seg.get(phase)
            if t is not None and t.shape[0] > 0:
                print(f"  {1} {phase}: {t.shape[0]} waypoints")
                draw_grasp_samples_as_axes(
                    grasp_poses=t,
                    axis_length=0.03,
                    line_thickness=3,
                    line_opacity=0.8,
                    clear_existing=clear_existing,
                )
                clear_existing = False

    while True:
        env.step(torch.full((env.num_envs, 15 + 14 + 14), torch.nan))


if __name__ == "__main__":
    main()
