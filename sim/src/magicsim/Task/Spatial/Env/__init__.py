import gymnasium as gym

gym.register(
    id="MotionEnv-V0",
    entry_point="magicsim.Task.Spatial.Env.MotionEnv:MotionEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SpatialPlaceEnv-V0",
    entry_point="magicsim.Task.Spatial.Env.SpatialPlaceEnv:SpatialPlaceEnv",
    disable_env_checker=True,
    order_enforce=False,
)
