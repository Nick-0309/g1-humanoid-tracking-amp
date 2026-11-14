"""PPO runner configs. Values follow the tutorial's PPO table."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class G1DanceTrackingPPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 10000
    save_interval = 250
    experiment_name = "g1d_tracking"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1DanceAmpPPORunnerCfg(G1DanceTrackingPPORunnerCfg):
    experiment_name = "g1d_amp"
