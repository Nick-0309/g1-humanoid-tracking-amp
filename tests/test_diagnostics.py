import torch

from g1dance.amp.diagnostics import group_rms, group_slices, second_difference, swap_group, velocity_mismatch
from g1dance.amp.features import frame_dim


def test_group_slices_cover_the_frame():
    groups = group_slices(29, 6)
    cols = sorted(c for s in groups.values() for c in range(s.start, s.stop))
    assert cols == list(range(frame_dim(29, 6)))


def test_swap_group_replaces_the_group_in_every_frame():
    w, d = 3, 5
    target, source = torch.zeros(4, w * d), torch.ones(4, w * d)
    out = swap_group(target, source, slice(1, 3), w).reshape(4, w, d)
    assert torch.all(out[..., 1:3] == 1) and torch.all(out[..., [0, 3, 4]] == 0)
    assert torch.all(target == 0)  # input untouched


def test_second_difference_is_zero_for_linear_motion():
    t = torch.arange(3.0).reshape(1, 3, 1).expand(2, 3, 4)
    windows = (2.0 * t + 1.0).reshape(2, -1)
    assert torch.allclose(second_difference(windows, 3), torch.zeros(2, 4))


def test_velocity_mismatch():
    dt = 0.02
    q = torch.tensor([0.0, 1.0, 2.0]) * 0.1  # 5 rad/s
    consistent = torch.stack([q, torch.full((3,), 5.0)], dim=-1).reshape(1, -1)  # frame = [pos, vel]
    noisy = torch.stack([q, torch.tensor([5.0, 8.0, 5.0])], dim=-1).reshape(1, -1)
    assert velocity_mismatch(consistent, 3, slice(0, 1), slice(1, 2), dt) < 1e-5
    assert abs(velocity_mismatch(noisy, 3, slice(0, 1), slice(1, 2), dt) - 3.0) < 1e-4
    assert abs(group_rms(noisy, slice(1, 2), 3) - float(torch.tensor([25.0, 64, 25]).mean().sqrt())) < 1e-5
