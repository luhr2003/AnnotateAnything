import gymnasium as gym


gym.register(
    id="IsaacRLEnv-V0",
    entry_point="magicsim.Env.Environment.Isaac.IsaacRLEnv:IsaacRLEnv",
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": "magicsim.Env.Environment.Isaac.IsaacRLEnv:IsaacRLEnvCfg"
    },
)
