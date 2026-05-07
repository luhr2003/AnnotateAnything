import gymnasium as gym
from omegaconf import DictConfig
import hydra
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF
import torch


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="curobo_dual_arm_config"
)
def main(cfg: DictConfig):
    print(cfg)
    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )
    env.reset()

    translation_0, orientation_0 = env.scene_manager.geometry_objects[0]["Cube"][
        0
    ].get_local_pose()
    translation_1, orientation_1 = env.scene_manager.geometry_objects[1]["Cube"][
        0
    ].get_local_pose()
    translation_2, orientation_2 = env.scene_manager.geometry_objects[2]["Cube"][
        0
    ].get_local_pose()
    translation_3, orientation_3 = env.scene_manager.geometry_objects[3]["Cube"][
        0
    ].get_local_pose()

    target_pos = [
        translation_0.tolist() + orientation_0.tolist() + [1, 1],
        translation_1.tolist() + orientation_1.tolist() + [1, 1],
        translation_2.tolist() + orientation_2.tolist() + [1, 1],
        translation_3.tolist() + orientation_3.tolist() + [1, 1],
    ]

    target_env_ids = [0, 1, 2, 3]

    for i in range(80):
        target_action = torch.tensor(target_pos, device="cuda:0")
        env.step(action=target_action, env_ids=target_env_ids)
        for _ in range(2):
            env.sim.sim_step()

    env.reset_idx([0, 1, 2, 3])
    # execution

    translation_0, orientation_0 = env.scene_manager.geometry_objects[0]["Cube"][
        0
    ].get_local_pose()
    translation_1, orientation_1 = env.scene_manager.geometry_objects[1]["Cube"][
        0
    ].get_local_pose()
    translation_2, orientation_2 = env.scene_manager.geometry_objects[2]["Cube"][
        0
    ].get_local_pose()
    translation_3, orientation_3 = env.scene_manager.geometry_objects[3]["Cube"][
        0
    ].get_local_pose()

    target_pos = [
        translation_0.tolist() + orientation_0.tolist(),
        translation_1.tolist() + orientation_1.tolist(),
        translation_2.tolist() + orientation_2.tolist(),
        translation_3.tolist() + orientation_3.tolist(),
    ]

    target_env_ids = [0, 1, 2, 3]

    for i in range(50):
        ready_action = {}
        ready_action["Xtrainer"] = {
            "arm_action": target_pos,
            "eef_action": [[1, 1], [1, 1], [1, 1], [1, 1]],
        }
        env.step(action=ready_action, env_ids=target_env_ids)
        for _ in range(2):
            env.sim.sim_step()

    env.reset_idx([0, 1, 2, 3])

    translation_0, orientation_0 = env.scene_manager.geometry_objects[0]["Cube"][
        0
    ].get_local_pose()
    translation_1, orientation_1 = env.scene_manager.geometry_objects[1]["Cube"][
        0
    ].get_local_pose()
    translation_2, orientation_2 = env.scene_manager.geometry_objects[2]["Cube"][
        0
    ].get_local_pose()
    translation_3, orientation_3 = env.scene_manager.geometry_objects[3]["Cube"][
        0
    ].get_local_pose()

    target_pos = [
        translation_0.tolist() + orientation_0.tolist() + [1, 1],
        translation_1.tolist() + orientation_1.tolist() + [1, 1],
        translation_2.tolist() + orientation_2.tolist() + [1, 1],
        translation_3.tolist() + orientation_3.tolist() + [1, 1],
    ]

    target_env_ids = [0, 1, 2, 3]
    for env_id in range(4):
        for i in range(50):
            target_action = torch.tensor(target_pos[env_id], device="cuda:0")

            env.step(action=target_action, env_ids=[env_id])
            for _ in range(2):
                env.sim.sim_step()

    env.reset_idx([0, 1, 2, 3])

    translation_0, orientation_0 = env.scene_manager.geometry_objects[0]["Cube"][
        0
    ].get_local_pose()
    translation_1, orientation_1 = env.scene_manager.geometry_objects[1]["Cube"][
        0
    ].get_local_pose()
    translation_2, orientation_2 = env.scene_manager.geometry_objects[2]["Cube"][
        0
    ].get_local_pose()
    translation_3, orientation_3 = env.scene_manager.geometry_objects[3]["Cube"][
        0
    ].get_local_pose()

    target_pos = [
        translation_0.tolist() + orientation_0.tolist(),
        translation_1.tolist() + orientation_1.tolist(),
        translation_2.tolist() + orientation_2.tolist(),
        translation_3.tolist() + orientation_3.tolist(),
    ]
    for env_id in range(4):
        for i in range(50):
            ready_action = {}
            ready_action["Xtrainer"] = {
                "arm_action": target_pos[env_id],
                "eef_action": [1, 1],
            }
            env.step(action=ready_action, env_ids=[env_id])
            for _ in range(2):
                env.sim.sim_step()

    while 1:
        env.sim.sim_step()


if __name__ == "__main__":
    main()
