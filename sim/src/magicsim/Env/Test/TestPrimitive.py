import argparse

import torch
import gymnasium as gym
from isaaclab.app import AppLauncher


# --- Original environment loading logic (preserved) ---
# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run custom DirectRL Env")
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Launch SimulationApp
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab_tasks.direct.franka_cabinet.franka_cabinet_env import (
    FrankaCabinetEnv,
    FrankaCabinetEnvCfg,
)

# 2. Import the new Mesh class from our previously created file
#    Please ensure this import path is correct!
from pxr import Gf  # 1. Import the Gf module to use Gf.Vec3f
from magicsim.Env.Scene.Object.Primitives import (
    CubeMesh,
    PlaneMesh,
    SphereMesh,
    ConeMesh,
    CylinderMesh,
    TorusMesh,
    DiskMesh,
    CapsuleMesh,
)

if __name__ == "__main__":
    # --- Original environment creation logic (preserved) ---
    cfg = FrankaCabinetEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cpu"
    cfg.sim.use_fabric = True

    env: FrankaCabinetEnv = gym.make("Isaac-Franka-Cabinet-Direct-v0", cfg=cfg)

    # --- 3. Add code to create Primitives on top of the existing environment ---
    print("--- Creating Deformable Primitive Meshes in the existing environment ---")

    # Create instances of all basic shapes in the scene and space them out
    # Positions are relative to the world origin

    cube = CubeMesh(prim_path="/World/Test/MyCube", position=Gf.Vec3f(0.0, 0.0, 2.0))
    plane = PlaneMesh(prim_path="/World/Test/MyPlane", position=Gf.Vec3f(1.0, 1.0, 2.0))
    cone = ConeMesh(prim_path="/World/Test/MyCone", position=Gf.Vec3f(2.0, 2.0, 2.0))
    disk = DiskMesh(prim_path="/World/Test/MyDisk", position=Gf.Vec3f(3.0, 3.0, 3.0))
    cylinder = CylinderMesh(
        prim_path="/World/Test/MyCylinder", position=Gf.Vec3f(-1.0, 1.0, 2.0)
    )
    sphere = SphereMesh(
        prim_path="/World/Test/MySphere", position=Gf.Vec3f(-3.0, 3.0, 3.0)
    )
    torus = TorusMesh(
        prim_path="/World/Test/MyTorus", position=Gf.Vec3f(3.0, -3.0, 3.0)
    )
    capsule = CapsuleMesh(
        prim_path="/World/Test/MyCapsule", position=Gf.Vec3f(1.0, -1.0, 3.0)
    )

    print("--- Primitives created. Check the Stage window. ---")

    env.reset()
    # --- Original simulation loop (preserved) ---
    while simulation_app.is_running():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(torch.tensor(action))
        # print(f"obs: {obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")

    env.close()
