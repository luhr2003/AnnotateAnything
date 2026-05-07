from magicsim.Collect.Command.CloseDrawer import CloseDrawer
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class MobileCloseDrawer(CloseDrawer):
    """
    MobileCloseDrawer task for mobile manipulation (ridgebackFranka + parallel gripper).

    Reuses :class:`CloseDrawer` logic; the
    :class:`~magicsim.Collect.AtomicSkill.CloseDrawer` atomic skill dispatches
    to ``MobileMoveL`` / ``MobileServoL`` when its config sets
    ``mobile: true``.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
