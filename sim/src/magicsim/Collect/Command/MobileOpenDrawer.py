from magicsim.Collect.Command.OpenDrawer import OpenDrawer
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class MobileOpenDrawer(OpenDrawer):
    """
    MobileOpenDrawer task for mobile manipulation (ridgebackFranka + parallel gripper).

    Reuses :class:`OpenDrawer` logic; the :class:`~magicsim.Collect.AtomicSkill.OpenDrawer`
    atomic skill dispatches to ``MobileMoveL`` / ``MobileServoL`` when its config
    sets ``mobile: true``.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
