from magicsim.Collect.Command.Grasp import Grasp
from magicsim.Env.Utils.file import Logger
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from omegaconf import DictConfig


class MobileGrasp(Grasp):
    """
    MobileGrasp task for mobile manipulation (ridgebackFranka + parallel gripper).
    Reuses Grasp logic; the Grasp atomic skill uses MobileMoveL when mobile=true.
    """

    def __init__(
        self, config: DictConfig, env: TaskBaseEnv, env_id: int, logger: Logger
    ):
        super().__init__(config, env, env_id, logger)
