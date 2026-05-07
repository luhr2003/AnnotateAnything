import gymnasium as gym

gym.register(
    id="MobileGraspEnv-V0",
    entry_point="magicsim.Task.MobileManip.Env.MobileGraspEnv:MobileGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MobileDexGraspEnv-V0",
    entry_point="magicsim.Task.MobileManip.Env.MobileDexGraspEnv:MobileDexGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MobileReachEnv-V0",
    entry_point="magicsim.Task.MobileManip.Env.MobileReachEnv:MobileReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MobileDualReachEnv-V0",
    entry_point="magicsim.Task.MobileManip.Env.MobileDualReachEnv:MobileDualReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MobileOpenDrawerEnv-V0",
    entry_point="magicsim.Task.MobileManip.Env.MobileOpenDrawerEnv:MobileOpenDrawerEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MobileCloseDrawerEnv-V0",
    entry_point="magicsim.Task.MobileManip.Env.MobileCloseDrawerEnv:MobileCloseDrawerEnv",
    disable_env_checker=True,
    order_enforce=False,
)
