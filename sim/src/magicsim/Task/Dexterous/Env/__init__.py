import gymnasium as gym

gym.register(
    id="DexReachEnv-V0",
    entry_point="magicsim.Task.Dexterous.Env.DexReachEnv:DexReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DexGraspEnv-V0",
    entry_point="magicsim.Task.Dexterous.Env.DexGraspEnv:DexGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BiDexGraspEnv-V0",
    entry_point="magicsim.Task.Dexterous.Env.BiDexGraspEnv:BiDexGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)
