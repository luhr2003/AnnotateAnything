import gymnasium as gym

gym.register(
    id="LocoGraspEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoGraspEnv:LocoGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LocoBiGraspEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoBiGraspEnv:LocoBiGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LocoLiftEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoLiftEnv:LocoLiftEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LocoBoxEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoBoxEnv:LocoBoxEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SquatGraspEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.SquatGraspEnv:SquatGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LocoReachEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoReachEnv:LocoReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LocoOpenDoorEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoOpenDoorEnv:LocoOpenDoorEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LocoNavEnv-V0",
    entry_point="magicsim.Task.LocoManip.Env.LocoNavEnv:LocoNavEnv",
    disable_env_checker=True,
    order_enforce=False,
)
