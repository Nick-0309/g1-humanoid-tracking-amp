"""Foot contact labels and action rate limiting. Pure torch."""

from __future__ import annotations

import torch

FLOOR_QUANTILE = 0.02
"""Fraction of the frames used as "the foot is on the floor" height of a clip (the lowest heights)."""


def foot_floor(foot_pos_w: torch.Tensor, quantile: float = FLOOR_QUANTILE) -> torch.Tensor:
    """Height of each foot link origin when the foot is planted. ``(T, F, 3)`` -> ``(F,)``.

    Taken from the clip's own lowest heights, so the offset between the link origin and the sole does not have
    to be known. The 2 % quantile rather than the minimum makes it robust to a few frames of penetration.
    """
    return torch.quantile(foot_pos_w[..., 2], quantile, dim=0)


def foot_speed(foot_pos_w: torch.Tensor, fps: float) -> torch.Tensor:
    """Horizontal speed of each foot in m/s, central differences. ``(T, F, 3)`` -> ``(T, F)``."""
    xy = foot_pos_w[..., :2]
    speed = torch.zeros_like(xy[..., 0])
    if foot_pos_w.shape[0] > 2:
        speed[1:-1] = (xy[2:] - xy[:-2]).norm(dim=-1) * (fps / 2.0)
        speed[0], speed[-1] = speed[1], speed[-2]
    return speed


def median_filter_labels(contact: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Majority filter along time: runs shorter than ``window // 2 + 1`` frames disappear. ``(T, F)`` bool."""
    if window <= 1:
        return contact
    pad = window // 2
    padded = torch.cat([contact[:1].expand(pad, -1), contact, contact[-1:].expand(pad, -1)], dim=0)
    return padded.unfold(0, window, 1).sum(dim=-1) * 2 > window


def reference_foot_contacts(
    foot_pos_w: torch.Tensor,
    fps: float,
    height_margin: float = 0.025,
    speed_limit: float = 0.5,
    smooth_window: int = 1,
) -> torch.Tensor:
    """Stance labels for the feet of a reference clip.

    A foot is in contact when it is within ``height_margin`` of its own lowest height in the clip (which
    absorbs the offset between the link origin and the sole) and moves slower than ``speed_limit`` m/s
    horizontally. ``foot_pos_w``: (T, F, 3) -> (T, F) bool.

    ``smooth_window`` (odd) removes label runs shorter than ``window // 2 + 1`` frames. Off by default:
    ``dance_102`` has 16 / 23 clean stance phases (32 / 46 label changes), so there is nothing to smooth; turn
    it on for a noisier clip, where a contact reward that switches on for a single 50 Hz frame is unlearnable.
    """
    z = foot_pos_w[..., 2]
    floor = foot_floor(foot_pos_w)
    contact = (z < floor + height_margin) & (foot_speed(foot_pos_w, fps) < speed_limit)
    return median_filter_labels(contact, smooth_window)


def touchdowns(contact: torch.Tensor) -> torch.Tensor:
    """Number of swing -> stance transitions per column. ``contact``: (T, F) bool -> (F,)."""
    return (contact[1:] & ~contact[:-1]).sum(dim=0)


def slew_limit(target: torch.Tensor, previous: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
    """Move from ``previous`` towards ``target`` by at most ``max_delta`` per element."""
    return previous + (target - previous).clamp(-max_delta, max_delta)
