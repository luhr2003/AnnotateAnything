import gymnasium as gym

gym.register(
    id="TableTopEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.TableTopEnv:TableTopEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="ReachEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.ReachEnv:ReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DualReachEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.DualReachEnv:DualReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)
gym.register(
    id="GraspEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.GraspEnv:GraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BiGraspEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.BiGraspEnv:BiGraspEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="HandoverEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.HandoverEnv:HandoverEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="WaveEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.WaveEnv:WaveEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RandomReachEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.RandomReachEnv:RandomReachEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="PushEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.PushEnv:PushEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RandomPushEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.RandomPushEnv:RandomPushEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="OpenDrawerEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.OpenDrawerEnv:OpenDrawerEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DexOpenDrawerEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.DexOpenDrawerEnv:DexOpenDrawerEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="CloseDrawerEnv-V0",
    entry_point="magicsim.Task.TableTop.Env.CloseDrawerEnv:CloseDrawerEnv",
    disable_env_checker=True,
    order_enforce=False,
)
