"""Running mean/std for the discriminator input (state lives in the state_dict)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, clip: float = 10.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """x: (B, dim). Parallel mean/variance merge."""
        n = float(x.shape[0])
        if n == 0:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + n
        new_mean = self.mean + delta * (n / total)
        m2 = self.var * self.count + batch_var * n + delta.pow(2) * (self.count * n / total)
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
