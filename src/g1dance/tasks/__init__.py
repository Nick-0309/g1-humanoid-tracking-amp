"""Gym registration. Only strings are used here, so importing this is safe before Kit starts."""

import gymnasium as gym

TRACKING_TASK = "G1D-Dance-Tracking-v0"
AMP_TASK = "G1D-Dance-AMP-v0"

gym.register(
    id=TRACKING_TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1dance.tasks.env_cfg:G1DanceTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": "g1dance.tasks.agents:G1DanceTrackingPPORunnerCfg",
    },
)

gym.register(
    id=AMP_TASK,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1dance.tasks.env_cfg:G1DanceAmpEnvCfg",
        "rsl_rl_cfg_entry_point": "g1dance.tasks.agents:G1DanceAmpPPORunnerCfg",
    },
)
