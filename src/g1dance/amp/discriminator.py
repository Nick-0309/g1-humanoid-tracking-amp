"""AMP discriminator, LSGAN loss with gradient penalty, and the style reward."""

from __future__ import annotations

import torch
import torch.nn as nn

from .normalizer import RunningNorm


class AmpDiscriminator(nn.Module):
    """MLP over a window of ``window`` stacked frames; expert -> +1, policy -> -1."""

    def __init__(self, frame_dim: int, window: int, hidden: tuple[int, ...] = (1024, 512)):
        super().__init__()
        self.frame_dim = frame_dim
        self.window = window
        self.norm = RunningNorm(frame_dim)
        layers: list[nn.Module] = []
        last = frame_dim * window
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, window * D) -> normalized (B, window * D); statistics are per frame feature."""
        b = windows.shape[0]
        return self.norm(windows.reshape(b, self.window, self.frame_dim)).reshape(b, -1)

    def score_normalized(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.score_normalized(self.normalize(windows))


def discriminator_loss(
    disc: AmpDiscriminator,
    expert: torch.Tensor,
    policy: torch.Tensor,
    gp_coef: float = 10.0,
    gp_mode: str = "interp",
    logit_reg: float = 0.0,
    weight_decay: float = 0.0,
    input_noise: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """LSGAN loss plus a gradient penalty computed in the normalized input space.

    ``gp_mode="interp"`` is the tutorial's WGAN-GP term on expert/policy interpolations: it only asks for a gradient
    norm of 1, so a discriminator can separate two distant sets perfectly while satisfying it.
    ``gp_mode="expert"`` is the AMP paper's ``(coef / 2) * E_expert ||grad D||^2``: it flattens D around the expert
    data, so policy samples close to it keep similar scores.

    ``logit_reg`` (L2 on the output layer weights), ``weight_decay`` (L2 on all weights) and ``input_noise``
    (Gaussian noise with this std, in normalized units, added to both inputs) are the extra regularizers of common
    AMP implementations; all default to off.
    """
    xe, xp = disc.normalize(expert), disc.normalize(policy)
    if input_noise > 0.0:
        xe = xe + input_noise * torch.randn_like(xe)
        xp = xp + input_noise * torch.randn_like(xp)
    de, dp = disc.score_normalized(xe), disc.score_normalized(xp)
    loss = (de - 1.0).pow(2).mean() + (dp + 1.0).pow(2).mean()

    if gp_mode == "interp":
        n = min(xe.shape[0], xp.shape[0])
        eps = torch.rand(n, 1, device=xe.device)
        xi = (eps * xe[:n] + (1.0 - eps) * xp[:n]).detach().requires_grad_(True)
        grad = torch.autograd.grad(disc.score_normalized(xi).sum(), xi, create_graph=True)[0]
        gp = (grad.norm(dim=-1) - 1.0).pow(2).mean()
        loss = loss + gp_coef * gp
    elif gp_mode == "expert":
        xg = xe.detach().requires_grad_(True)
        grad = torch.autograd.grad(disc.score_normalized(xg).sum(), xg, create_graph=True)[0]
        gp = grad.pow(2).sum(-1).mean()
        loss = loss + 0.5 * gp_coef * gp
    else:
        raise ValueError(f"unknown gp_mode {gp_mode!r}")

    weights = [m.weight for m in disc.mlp if isinstance(m, nn.Linear)]
    if logit_reg > 0.0:
        loss = loss + logit_reg * weights[-1].pow(2).sum()
    if weight_decay > 0.0:
        loss = loss + weight_decay * sum(w.pow(2).sum() for w in weights)

    metrics = {
        "disc_loss": loss.detach(),
        "d_expert": de.detach().mean(),
        "d_policy": dp.detach().mean(),
        "grad_penalty": gp.detach(),
        "disc_acc": 0.5 * ((de.detach() > 0).float().mean() + (dp.detach() < 0).float().mean()),
    }
    return loss, metrics


STYLE_REWARDS = ("tutorial", "paper", "softplus", "linear")


def style_reward(d: torch.Tensor, kind: str = "tutorial", eps: float = 1e-4) -> torch.Tensor:
    """Map discriminator scores (expert -> +1, policy -> -1) to a style reward.

    ``tutorial``: ``-log(1 - 0.25 (D + 1)^2)`` with D clamped to [-1, 1] (the raw expression is not monotonic, it is
    also maximal at D = -3). ``paper``: ``max(0, 1 - 0.25 (D - 1)^2)`` from the AMP paper. Both are exactly 0 for
    every D <= -1, which is where a trained LSGAN discriminator puts the policy, so they stop ranking policy samples
    once the discriminator is confident.

    ``softplus``: ``log(1 + exp(D))`` with D clamped to at most 1, the GAIL reward ``-log(1 - sigmoid(D))``. It is
    increasing everywhere, so better policy samples keep getting more reward even when all of them score below -1.

    ``linear``: ``(D + 1) / 2`` with D clamped to [-3, 1]: 0 at the policy target, 1 at the expert target, negative
    below. The convex or floored mappings above reward the *spread* of D: noisy motion whose windows occasionally
    score high beats steady motion that scores slightly below -1 everywhere (measured offline on a trained tracking
    policy with added action noise). A linear mapping ranks by the mean score instead.
    """
    if kind == "tutorial":
        d = d.clamp(-1.0, 1.0)
        return -torch.log((1.0 - 0.25 * (d + 1.0).pow(2)).clamp(min=eps))
    if kind == "paper":
        return (1.0 - 0.25 * (d - 1.0).pow(2)).clamp(min=0.0)
    if kind == "softplus":
        return torch.nn.functional.softplus(d.clamp(max=1.0))
    if kind == "linear":
        # Mean-preserving: a policy whose windows score more spread out but lower on average gets less reward.
        # Clamped to [-3, 1] only so a few far-off windows cannot dominate a batch.
        return 0.5 * (d.clamp(-3.0, 1.0) + 1.0)
    raise ValueError(f"unknown style reward {kind!r}, expected one of {STYLE_REWARDS}")
