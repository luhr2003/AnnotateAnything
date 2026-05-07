import gymnasium as gym

gym.register(
    id="ContactRichEnv-V0",
    entry_point="magicsim.Task.ContactRich.Env.ContactRichEnv:ContactRichEnv",
    disable_env_checker=True,
    order_enforce=False,
)
