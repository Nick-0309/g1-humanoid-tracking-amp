"""PPO runner with an AMP style reward.

``learn`` follows ``rsl_rl.runners.OnPolicyRunner.learn`` (rsl-rl-lib 5.x) and adds, per env step,
the AMP window / style-reward bookkeeping and, per iteration, the discriminator update after the
PPO update. The discriminator is saved in and restored from the checkpoint.

The env must provide an ``amp`` observation group (one frame per env) and run with
``compute_final_obs=True`` so terminal frames are available in ``extras["final_obs"]``.
"""

from __future__ import annotations

import importlib.metadata
import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan

from .module import AmpModule

_SUPPORTED_RSL_RL_MAJOR = 5


class AmpOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, amp: AmpModule, log_dir: str | None = None, device: str = "cpu"):
        version = importlib.metadata.version("rsl-rl-lib")
        if int(version.split(".")[0]) != _SUPPORTED_RSL_RL_MAJOR:
            raise RuntimeError(f"AmpOnPolicyRunner.learn is written for rsl-rl-lib 5.x, found {version}")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if self.is_distributed:
            raise NotImplementedError("AMP training supports a single GPU only")
        if "amp" not in self.env.get_observations().keys():
            raise ValueError("the environment has no 'amp' observation group")
        self.amp = amp

    def _next_frame(self, obs, dones: torch.Tensor, extras: dict) -> torch.Tensor:
        """Frame reached by the step: the terminal frame for envs that just finished."""
        final = extras.get("final_obs")
        if final is None:
            if dones.any():
                raise RuntimeError("envs finished but extras has no 'final_obs'; set cfg.compute_final_obs=True")
            return obs["amp"]
        return torch.where(dones.bool().unsqueeze(-1), final["amp"], obs["amp"])

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        self.amp.start(obs["amp"])

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        steps = self.cfg["num_steps_per_env"]
        for it in range(start_it, total_it):
            start = time.time()
            task_reward_sum = torch.zeros((), device=self.device)
            self.amp.begin_rollout()
            with torch.inference_mode():
                for _ in range(steps):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    style = self.amp.step(self._next_frame(obs, dones, extras), dones, obs["amp"])
                    task_reward_sum += rewards.mean()
                    rewards = self.amp.mix(rewards, style)

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            amp_metrics = self.amp.update()
            amp_metrics["task_reward_step_mean"] = float(task_reward_sum) / steps

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )
            self._log_amp(it, amp_metrics)

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def _log_amp(self, it: int, metrics: dict[str, float]) -> None:
        writer = self.logger.writer
        if writer is None:
            return
        for key, value in metrics.items():
            writer.add_scalar(f"AMP/{key}", value, it)
        if it % 10 == 0 and metrics:
            print(
                f"[AMP] it={it} D(expert)={metrics.get('d_expert', float('nan')):+.3f} "
                f"D(policy)={metrics.get('d_policy', float('nan')):+.3f} "
                f"acc={metrics.get('disc_acc', float('nan')):.3f} "
                f"style_r={metrics.get('style_reward', float('nan')):.3f} "
                f"task_r={metrics.get('task_reward_step_mean', float('nan')):.3f}"
                + (" (discriminator warm-up: style reward not used)" if metrics.get("warming_up") else ""),
                flush=True,
            )

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["amp"] = self.amp.state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if self.alg.load(loaded_dict, load_cfg, strict):
            self.current_learning_iteration = loaded_dict["iter"]
        if "amp" in loaded_dict:
            self.amp.load_state_dict(loaded_dict["amp"])
        else:
            print(f"[AMP] {path} has no discriminator state; starting the discriminator from scratch.", flush=True)
        return loaded_dict["infos"]
