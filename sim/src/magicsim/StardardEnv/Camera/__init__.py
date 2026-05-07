import gymnasium as gym

gym.register(
    id="TaskCameraBaseEnv-V0",
    entry_point="magicsim.StardardEnv.Camera.TaskCameraBaseEnv:TaskCameraBaseEnv",
)

gym.register(
    id="AsyncCameraEnv-V0",
    entry_point="magicsim.StardardEnv.Camera.AsyncCameraEnv:AsyncCameraEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="AutoCameraCollectEnv-V0",
    entry_point="magicsim.StardardEnv.Camera.AutoCameraCollectEnv:AutoCameraCollectEnv",
    disable_env_checker=True,
    order_enforce=False,
)
from magicsim.Task.Camera.Env import *  # noqa: F403
