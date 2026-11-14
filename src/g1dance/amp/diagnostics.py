"""What does the discriminator look at? Pure torch helpers for ``scripts/diagnose_amp.py``.

Policy windows are compared with expert windows taken at the same reference time steps, so the only
differences are how the robot moved, not which part of the dance it was in.
"""

from __future__ import annotations

import torch


def group_slices(num_joints: int, num_key_bodies: int) -> dict[str, slice]:
    """Column ranges of the feature groups inside one frame (layout of ``compute_amp_frame``)."""
    j, k = num_joints, num_key_bodies
    return {
        "base_ang_vel": slice(0, 3),
        "joint_pos": slice(3, 3 + j),
        "joint_vel": slice(3 + j, 3 + 2 * j),
        "key_body_pos": slice(3 + 2 * j, 3 + 2 * j + 3 * k),
    }


def as_frames(windows: torch.Tensor, window: int) -> torch.Tensor:
    """(M, window * D) -> (M, window, D)."""
    return windows.reshape(windows.shape[0], window, -1)


def swap_group(target: torch.Tensor, source: torch.Tensor, cols: slice, window: int) -> torch.Tensor:
    """Copy of ``target`` windows whose ``cols`` columns come from ``source`` in every frame."""
    out = as_frames(target, window).clone()
    out[..., cols] = as_frames(source, window)[..., cols]
    return out.reshape(target.shape)


def group_rms(x: torch.Tensor, cols: slice, window: int) -> float:
    """Root mean square over all windows, frames and columns of a group."""
    return float(as_frames(x, window)[..., cols].pow(2).mean().sqrt())


def second_difference(windows: torch.Tensor, window: int) -> torch.Tensor:
    """``f[t] - 2 f[t-1] + f[t-2]`` of the last three frames: (M, D). Large for jittery motion."""
    f = as_frames(windows, window)
    return f[:, -1] - 2.0 * f[:, -2] + f[:, -3]


def velocity_mismatch(windows: torch.Tensor, window: int, pos: slice, vel: slice, dt: float) -> float:
    """RMS of ``vel`` minus the central difference of ``pos`` at the middle of the last three frames.

    For the expert both come from the same smooth clip, so this is small. For the simulated robot ``vel`` is the
    instantaneous physics velocity; a large value means velocity carries high-frequency content that positions
    sampled at the control rate do not show.
    """
    f = as_frames(windows, window)
    central = (f[:, -1, pos] - f[:, -3, pos]) / (2.0 * dt)
    return float((f[:, -2, vel] - central).pow(2).mean().sqrt())
