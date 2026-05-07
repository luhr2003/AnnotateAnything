from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.Go2WBC.Go2.configs import (
    Go2WBCV1Config,
)
from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.wbc_base_policy import (
    WBCPolicy,
)

from magicsim.Env.Robot.Cfg.Quadruped.whole_body_controller.Go2WBC.go2_wbc import (
    Go2WBCPolicy,
)


def get_wbc_policy(robot_type: str, wbc_version: str, num_envs: int = 1) -> WBCPolicy:
    """Get the WBC policy for the given robot type and configuration.

    Args:
        robot_type: The type of robot to get the WBC policy for. Only "go2" is supported.
        wbc_version: The version of the WBC policy. Only "go2_v1" is supported.
        num_envs: The number of environments to use in IsaacLab

    Returns:
        The WBC policy for the given robot type and configuration
    """
    assert num_envs > 0, f"num_envs must be greater than 0, got {num_envs}"
    if robot_type == "go2":
        if wbc_version == "go2_v1":
            wbc_config = Go2WBCV1Config()
            wbc_policy = Go2WBCPolicy(
                wbc_config=wbc_config,
                num_envs=num_envs,
            )
        else:
            raise ValueError(
                f"Invalid WBC policy version: {wbc_version}, Supported WBC policy versions: go2_v1"
            )

    else:
        raise ValueError(
            f"Invalid robot type: {robot_type}. Supported robot types: go2"
        )
    return wbc_policy
