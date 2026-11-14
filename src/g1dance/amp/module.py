"""AMP module: expert data, discriminator, rollout window buffer and its update.

Used by :class:`g1dance.amp.runner.AmpOnPolicyRunner`. Pure torch, so it can be tested
without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .discriminator import AmpDiscriminator, discriminator_loss, style_reward
from .features import build_windows
from .window import WindowTracker


@dataclass
class AmpCfg:
    window: int = 3
    """Frames per discriminator input."""
    style_lerp: float = 0.1
    """``r = (1 - w) * r_task + w * r_style``."""
    style_reward: str = "linear"
    """Score-to-reward mapping, see :func:`g1dance.amp.discriminator.style_reward`. ``linear`` is the only one
    that ranked noisier motion lower in the offline test; ``tutorial`` (v1 default) rewarded noise."""
    hidden: tuple[int, ...] = (1024, 512)
    lr: float = 3e-5
    gp_coef: float = 5.0
    gp_mode: str = "expert"
    """``expert`` (AMP paper) or ``interp`` (the tutorial; v1 default, saturates, see the README)."""
    max_grad_norm: float = 1.0
    epochs: int = 1
    minibatches: int = 4
    replay_capacity: int = 0
    """Past policy windows kept for the discriminator. 0 = current rollout only."""
    logit_reg: float = 0.05
    """L2 on the output layer weights."""
    weight_decay: float = 0.0
    """L2 on all discriminator weights."""
    input_noise: float = 0.0
    """Std of Gaussian noise added to normalized discriminator inputs during its update."""
    disc_warmup: int = 0
    """Number of first discriminator updates (= PPO iterations) during which the style reward is not used."""
    seed: int = 0


class AmpModule:
    def __init__(
        self, cfg: AmpCfg, expert_frames: torch.Tensor, num_envs: int, num_steps: int, device: str | torch.device
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.frame_dim = expert_frames.shape[1]
        win_dim = cfg.window * self.frame_dim

        torch.manual_seed(cfg.seed)
        expert_frames = expert_frames.to(self.device)
        self.expert = build_windows(expert_frames, cfg.window)
        self.disc = AmpDiscriminator(self.frame_dim, cfg.window, cfg.hidden).to(self.device)
        self.disc.norm.update(expert_frames)
        self.optim = torch.optim.Adam(self.disc.parameters(), lr=cfg.lr)

        self.tracker = WindowTracker(num_envs, cfg.window, self.frame_dim, self.device)
        self.buf = torch.zeros(num_steps, num_envs, win_dim, device=self.device)
        self.valid = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self._t = 0
        self._style_sum = torch.zeros((), device=self.device)

        self.replay = torch.zeros(cfg.replay_capacity, win_dim, device=self.device)
        self._replay_ptr = 0
        self._replay_size = 0
        self.updates_done = 0

    # -- rollout side -------------------------------------------------------
    def start(self, start_frame: torch.Tensor) -> None:
        self.tracker.reset(start_frame)

    def begin_rollout(self) -> None:
        self._t = 0
        self._style_sum.zero_()

    @torch.no_grad()
    def step(self, next_frame: torch.Tensor, dones: torch.Tensor, start_frame: torch.Tensor) -> torch.Tensor:
        """Record this step's window and return its style reward.

        Envs that were just reset have no complete window yet. They get the mean style reward of the envs that
        do, so mixing the reward neither penalises nor favours the first steps of an episode.
        """
        if self._t >= self.num_steps:
            raise RuntimeError("more steps than the rollout buffer holds; call begin_rollout()")
        windows, valid = self.tracker.step(next_frame, dones, start_frame)
        self.buf[self._t] = windows
        self.valid[self._t] = valid
        self._t += 1
        reward = style_reward(self.disc(windows), self.cfg.style_reward)
        valid_sum = (reward * valid).sum()
        self._style_sum += valid_sum
        fill = valid_sum / valid.sum().clamp(min=1)
        return torch.where(valid, reward, fill)

    @property
    def warming_up(self) -> bool:
        return self.updates_done < self.cfg.disc_warmup

    def mix(self, task_reward: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """During the discriminator warm-up the policy sees the task reward only, unscaled."""
        if self.warming_up:
            return task_reward
        w = self.cfg.style_lerp
        return (1.0 - w) * task_reward + w * style

    @torch.no_grad()
    def score(self, windows: torch.Tensor) -> torch.Tensor:
        return self.disc(windows)

    # -- discriminator update ------------------------------------------------
    def update(self) -> dict[str, float]:
        with torch.inference_mode(False), torch.enable_grad():
            valid = self.valid[: self._t].reshape(-1)
            policy = self.buf[: self._t].reshape(-1, self.buf.shape[-1])[valid].clone()
            n_valid = int(valid.sum())
            if n_valid == 0:
                return {}

            self.disc.norm.update(policy.reshape(-1, self.frame_dim))
            pool = policy
            if self._replay_size > 0:
                pool = torch.cat([policy, self.replay[: self._replay_size]], dim=0)

            sums: dict[str, torch.Tensor] = {}
            steps = 0
            for _ in range(self.cfg.epochs):
                perm = torch.randperm(pool.shape[0], device=self.device)
                for idx in perm.chunk(self.cfg.minibatches):
                    expert_idx = torch.randint(0, self.expert.shape[0], (idx.numel(),), device=self.device)
                    loss, metrics = discriminator_loss(
                        self.disc,
                        self.expert[expert_idx],
                        pool[idx],
                        self.cfg.gp_coef,
                        self.cfg.gp_mode,
                        logit_reg=self.cfg.logit_reg,
                        weight_decay=self.cfg.weight_decay,
                        input_noise=self.cfg.input_noise,
                    )
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.cfg.max_grad_norm)
                    self.optim.step()
                    for k, v in metrics.items():
                        sums[k] = sums.get(k, 0.0) + v
                    steps += 1

            self._push_replay(policy)
            out = {k: float(v) / steps for k, v in sums.items()}
            out["style_reward"] = float(self._style_sum) / n_valid
            out["valid_fraction"] = n_valid / valid.numel()
            out["warming_up"] = float(self.warming_up)
            self.updates_done += 1
            return out

    def _push_replay(self, windows: torch.Tensor) -> None:
        cap = self.cfg.replay_capacity
        if cap == 0:
            return
        windows = windows[-cap:]
        n = windows.shape[0]
        end = self._replay_ptr + n
        if end <= cap:
            self.replay[self._replay_ptr : end] = windows
        else:
            first = cap - self._replay_ptr
            self.replay[self._replay_ptr :] = windows[:first]
            self.replay[: n - first] = windows[first:]
        self._replay_ptr = end % cap
        self._replay_size = min(self._replay_size + n, cap)

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> dict:
        return {"disc": self.disc.state_dict(), "optim": self.optim.state_dict(), "updates_done": self.updates_done}

    def load_state_dict(self, state: dict) -> None:
        self.disc.load_state_dict(state["disc"])
        self.optim.load_state_dict(state["optim"])
        # older checkpoints have no counter: they come from a run that is past any warm-up
        self.updates_done = state.get("updates_done", self.cfg.disc_warmup)
