from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Homie.G1.configs import (
    G1Dex1HomieV2Config,
    G1FixedHandHomieV2Config,
    G1HomieV2Config,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.wbc_base_policy import (
    WBCPolicy,
)

from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Homie.homie import (
    HomiePolicy,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Sonic.G1.configs import (
    G1SonicV1Config,
)
from magicsim.Env.Robot.Cfg.Humanoid.whole_body_controller.Sonic.sonic import (
    SonicPolicy,
)


def get_wbc_policy(robot_type: str, wbc_version: str, num_envs: int = 1) -> WBCPolicy:
    """Get the WBC policy for the given robot type and configuration.

    Args:
        robot_type: The type of robot to get the WBC policy for. Only "g1" is supported.
        wbc_config: The configuration for the WBC policy
        num_envs: The number of environments to use in IsaacLab

    Returns:
        The WBC policy for the given robot type and configuration
    """
    assert num_envs > 0, f"num_envs must be greater than 0, got {num_envs}"
    if robot_type == "g1":
        if wbc_version == "homie_v2":
            wbc_config = G1HomieV2Config()
            wbc_policy = HomiePolicy(
                wbc_config=wbc_config,
                num_envs=num_envs,
            )
        elif wbc_version == "sonic_v1":
            wbc_config = G1SonicV1Config()
            wbc_policy = SonicPolicy(
                wbc_config=wbc_config,
                num_envs=num_envs,
            )
        else:
            raise ValueError(
                f"Invalid lower body policy type: {wbc_version}, "
                f"Supported lower body policy types: homie_v2, sonic_v1"
            )

    elif robot_type == "g1_dex1":
        # Same Homie ONNX policy as g1 (legs+waist outputs are identical), but
        # the joint-order table is sized for the 33-joint dex1 articulation.
        if wbc_version != "homie_v2":
            raise ValueError(f"g1_dex1 only supports homie_v2; got {wbc_version}")
        wbc_config = G1Dex1HomieV2Config()
        wbc_policy = HomiePolicy(
            wbc_config=wbc_config,
            num_envs=num_envs,
        )

    elif robot_type == "g1_fixed_hand":
        # Same Homie ONNX policy; 29-joint table (rigid rubber_hand, no fingers).
        if wbc_version != "homie_v2":
            raise ValueError(f"g1_fixed_hand only supports homie_v2; got {wbc_version}")
        wbc_config = G1FixedHandHomieV2Config()
        wbc_policy = HomiePolicy(
            wbc_config=wbc_config,
            num_envs=num_envs,
        )

    else:
        raise ValueError(
            f"Invalid robot type: {robot_type}. Supported robot types: "
            f"g1, g1_dex1, g1_fixed_hand"
        )
    return wbc_policy
