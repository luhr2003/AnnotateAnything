import gymnasium as gym

gym.register(
    id="TaskBaseEnv-V0",
    entry_point="magicsim.StardardEnv.Robot.TaskBaseEnv:TaskBaseEnv",
)

gym.register(
    id="AsycRobotEnv-V0",
    entry_point="magicsim.StardardEnv.Robot.AsyncRobotEnv:AsyncRobotEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="AutoCollectEnv-V0",
    entry_point="magicsim.StardardEnv.Robot.AutoCollectEnv:AutoCollectEnv",
    disable_env_checker=True,
    order_enforce=False,
)
from magicsim.Task.TableTop.Env import *  # noqa: F403
from magicsim.Task.LocoManip.Env import *  # noqa: F403
from magicsim.Task.MobileManip.Env import *  # noqa: F403
from magicsim.Task.Dexterous.Env import *  # noqa: F403
