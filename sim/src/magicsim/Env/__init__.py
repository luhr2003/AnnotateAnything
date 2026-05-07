import gymnasium as gym


# Register the environment with OpenAI Gym

gym.register(
    id="BaseEnv-V0",
    entry_point="magicsim.Env.Environment.BaseEnv:BaseEnv",
    disable_env_checker=True,
    order_enforce=False,
)


gym.register(
    id="SyncBaseEnv-V0",
    entry_point="magicsim.Env.Environment.SyncBaseEnv:SyncBaseEnv",
    disable_env_checker=True,
    order_enforce=False,
)


gym.register(
    id="SyncCollectEnv-V0",
    entry_point="magicsim.Env.Environment.SyncCollectEnv:SyncCollectEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SyncRobotEnv-V0",
    entry_point="magicsim.Env.Environment.SyncRobotEnv:SyncRobotEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SyncCameraEnv-V0",
    entry_point="magicsim.Env.Environment.SyncCameraEnv:SyncCameraEnv",
    disable_env_checker=True,
    order_enforce=False,
)
