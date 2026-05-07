import gymnasium as gym


gym.register(
    id="GoToEnv-V0",
    entry_point="magicsim.Task.Camera.Env.GoToEnv:GoToEnv",
    disable_env_checker=True,
    order_enforce=False,
)
