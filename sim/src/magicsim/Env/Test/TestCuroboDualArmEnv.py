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

    # Get poses for red and green cubes for each environment
    # Red cube corresponds to left arm (J1_6), Green cube corresponds to right arm (J2_6)
    num_envs = 4
    target_pos_list = []

    for env_id in range(num_envs):
        # Get red cube pose (left arm target)
        red_cube_translation, red_cube_orientation = env.scene_manager.geometry_objects[
            env_id
        ]["RedCube"][0].get_local_pose()
        # Get green cube pose (right arm target)
        green_cube_translation, green_cube_orientation = (
            env.scene_manager.geometry_objects[env_id]["GreenCube"][0].get_local_pose()
        )

        # Format: [left_arm_pose(7), right_arm_pose(7)] = [14] for each env
        # Each pose: [x, y, z, qw, qx, qy, qz]
        left_arm_target = red_cube_translation.tolist() + red_cube_orientation.tolist()
        right_arm_target = (
            green_cube_translation.tolist() + green_cube_orientation.tolist() + [1, 1]
        )

        # Flatten to [14] format: [left_arm(7), right_arm(7)]
        target_pos_list.append(left_arm_target + right_arm_target)

    target_env_ids = [0, 1, 2, 3]

    # Test 1: Batch planning for all environments
    print("Test 1: Batch planning for all environments")
    for i in range(80):
        target_action = torch.tensor(target_pos_list, device="cuda:0")  # [4, 14]
        env.step(action=target_action, env_ids=target_env_ids)
        for _ in range(2):
            env.sim.sim_step()

    env.reset_idx([0, 1, 2, 3])

    # Test 2: Using arm_action format
    print("Test 2: Using arm_action format")
    for env_id in range(num_envs):
        red_cube_translation, red_cube_orientation = env.scene_manager.geometry_objects[
            env_id
        ]["RedCube"][0].get_local_pose()
        green_cube_translation, green_cube_orientation = (
            env.scene_manager.geometry_objects[env_id]["GreenCube"][0].get_local_pose()
        )

        left_arm_target = red_cube_translation.tolist() + red_cube_orientation.tolist()
        right_arm_target = (
            green_cube_translation.tolist() + green_cube_orientation.tolist()
        )
        target_pos_list[env_id] = left_arm_target + right_arm_target

    for i in range(50):
        ready_action = {}
        ready_action["Xtrainer"] = {
            "arm_action": target_pos_list,  # [4, 14] format
            "eef_action": [[1, 1], [1, 1], [1, 1], [1, 1]],  # Two grippers per env
        }
        env.step(action=ready_action, env_ids=target_env_ids)
        for _ in range(2):
            env.sim.sim_step()

    env.reset_idx([0, 1, 2, 3])

    # Test 3: Single environment planning
    print("Test 3: Single environment planning")
    for env_id in range(num_envs):
        red_cube_translation, red_cube_orientation = env.scene_manager.geometry_objects[
            env_id
        ]["RedCube"][0].get_local_pose()
        green_cube_translation, green_cube_orientation = (
            env.scene_manager.geometry_objects[env_id]["GreenCube"][0].get_local_pose()
        )

        left_arm_target = red_cube_translation.tolist() + red_cube_orientation.tolist()
        right_arm_target = (
            green_cube_translation.tolist() + green_cube_orientation.tolist() + [1, 1]
        )
        target_pos = left_arm_target + right_arm_target  # [14]

        for i in range(50):
            target_action = torch.tensor(target_pos, device="cuda:0")  # [14]
            env.step(action=target_action, env_ids=[env_id])
            for _ in range(2):
                env.sim.sim_step()

    env.reset_idx([0, 1, 2, 3])

    # Test 4: Single environment with arm_action format
    print("Test 4: Single environment with arm_action format")
    for env_id in range(num_envs):
        red_cube_translation, red_cube_orientation = env.scene_manager.geometry_objects[
            env_id
        ]["RedCube"][0].get_local_pose()
        green_cube_translation, green_cube_orientation = (
            env.scene_manager.geometry_objects[env_id]["GreenCube"][0].get_local_pose()
        )

        left_arm_target = red_cube_translation.tolist() + red_cube_orientation.tolist()
        right_arm_target = (
            green_cube_translation.tolist() + green_cube_orientation.tolist() + [1, 1]
        )
        target_pos = left_arm_target + right_arm_target  # [14]

        for i in range(50):
            ready_action = {}
            ready_action["Xtrainer"] = {
                "arm_action": target_pos,  # [14] format
                "eef_action": [1, 1],  # Two grippers
            }
            env.step(action=ready_action, env_ids=[env_id])
            for _ in range(2):
                env.sim.sim_step()

    # Keep simulation running
    print("All tests completed. Simulation running...")
    while 1:
        env.sim.sim_step()


if __name__ == "__main__":
    main()
