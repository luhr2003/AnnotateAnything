"""
Test camera movement with planner (FlyingCamera)
Tests moving camera from [0,0,1] to [1,1,1] using planner
"""

import gymnasium as gym
from omegaconf import DictConfig
import hydra
import torch
from magicsim.Env.Environment.SyncRobotEnv import SyncRobotEnv
from magicsim.Env.Utils.file import Logger
from loguru import logger as log
from magicsim import MAGICSIM_CONF


@hydra.main(
    version_base=None, config_path=MAGICSIM_CONF, config_name="test_camera_move_config"
)
def main(cfg: DictConfig):
    print("=" * 80)
    print("Test Camera Movement with Planner")
    print("=" * 80)

    logger = Logger("Env", log)
    env: SyncRobotEnv = gym.make(
        "SyncRobotEnv-V0", config=cfg, cli_args=None, logger=logger
    )

    # Reset environment
    env.reset()

    # Use camera manager config (already sanitized) to get camera name
    manager_camera_cfg = getattr(env.camera_manager, "camera_config", None)
    if manager_camera_cfg is None or len(manager_camera_cfg) == 0:
        raise ValueError("CameraManager has no camera configuration loaded.")
    camera_name = list(manager_camera_cfg.keys())[0]
    print(f"Camera name: {camera_name}")

    # Initial position: [0, 0, 1]
    # Target position: [1, 1, 1]
    # Orientation: keep same [1, 0, 0, 0] (w, x, y, z)
    initial_pos = torch.tensor([0.0, 0.0, 2.0], device=env.device)
    target_pos = torch.tensor([1.0, 1.0, 2.0], device=env.device)
    initial_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)  # w, x, y, z

    # Set initial camera pose
    initial_pose = torch.cat(
        [initial_pos, initial_quat], dim=0
    )  # [x, y, z, w, x, y, z]
    env.camera_manager.set_camera_pose(
        camera_name, initial_pose.unsqueeze(0), env_ids=[0]
    )

    # Get initial state
    initial_state = env.camera_manager.get_camera_state(camera_name, env_ids=[0])
    print(f"\nInitial camera position: {initial_state['pos'][0].cpu().tolist()}")
    print(f"Initial camera quaternion: {initial_state['quat'][0].cpu().tolist()}")

    # Create target pose: [x, y, z, w, x, y, z]
    target_pose = torch.cat([target_pos, initial_quat], dim=0).unsqueeze(0)  # [1, 7]

    print(f"\nTarget camera position: {target_pos.cpu().tolist()}")
    print(f"Target camera quaternion: {initial_quat.cpu().tolist()}")
    print("\n" + "=" * 80)
    print("Starting camera movement...")
    print("=" * 80)
    # Simulate movement
    num_steps = 200
    for step in range(num_steps):
        # Use env.step() to process camera action through planner
        # Use planner to smooth the movement via env.step()
        env.step(action=None, camera_action={camera_name: target_pose}, env_ids=[0])
        # Step simulation
        env.sim.sim_step()

        # Get current camera state
        current_state = env.camera_manager.get_camera_state(camera_name, env_ids=[0])
        current_pos = current_state["pos"][0].cpu()
        if step % 10 == 0:
            print(f"step {step}: {current_pos.tolist()}")

    # Final state
    final_state = env.camera_manager.get_camera_state(camera_name, env_ids=[0])
    final_pos = final_state["pos"][0].cpu()
    final_quat = final_state["quat"][0].cpu()

    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(f"Final camera position: {final_pos.tolist()}")
    print(f"Final camera quaternion: {final_quat.tolist()}")
    print(f"Target position: {target_pos.cpu().tolist()}")


if __name__ == "__main__":
    main()
