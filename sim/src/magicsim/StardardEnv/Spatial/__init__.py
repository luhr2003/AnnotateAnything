import gymnasium as gym

gym.register(
    id="SpatialPlaceEnv-V0",
    entry_point="magicsim.StardardEnv.Spatial.SpatialPlaceEnv:SpatialPlaceEnv",
    disable_env_checker=True,
    order_enforce=False,
)
