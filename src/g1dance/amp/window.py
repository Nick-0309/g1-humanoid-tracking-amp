"""Per-env sliding window of AMP frames that never straddles an episode reset."""

from __future__ import annotations

import torch


class WindowTracker:
    """Holds the last ``window`` frames of every env.

    ``step`` takes the frame reached by the step (the terminal frame for envs that just
    finished) and returns the window ending there. An env only yields a valid window once
    ``window`` frames of the *same episode* have been seen; after a reset it starts over.
    All updates are in place so the tracker can live outside ``inference_mode`` and be
    updated inside it.
    """

    def __init__(self, num_envs: int, window: int, frame_dim: int, device: str | torch.device):
        self.window = window
        self.frames = torch.zeros(num_envs, window, frame_dim, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, start_frame: torch.Tensor) -> None:
        self.frames.zero_()
        self.frames[:, -1] = start_frame
        self.count.fill_(1)

    def step(
        self, next_frame: torch.Tensor, dones: torch.Tensor, start_frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(windows (N, window * D), valid (N,))``.

        ``start_frame`` is the post-reset frame, used to seed the next episode of done envs.
        """
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = next_frame
        self.count.add_(1).clamp_(max=self.window)
        windows = self.frames.reshape(self.frames.shape[0], -1).clone()
        valid = self.count >= self.window

        done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.frames[done_ids] = 0.0
            self.frames[done_ids, -1] = start_frame[done_ids]
            self.count[done_ids] = 1
        return windows, valid
