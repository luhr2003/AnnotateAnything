import gymnasium as gym

gym.register(
    id="GarmentFoldEnv-V0",
    entry_point="magicsim.Task.Garment.Env.GarmentFoldEnv:GarmentFoldEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="FlingEnv-V0",
    entry_point="magicsim.Task.Garment.Env.FlingEnv:FlingEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="HangGarmentEnv-V0",
    entry_point="magicsim.Task.Garment.Env.HangGarmentEnv:HangGarmentEnv",
    disable_env_checker=True,
    order_enforce=False,
)
