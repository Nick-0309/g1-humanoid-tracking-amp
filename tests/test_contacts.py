import torch

from g1dance.mdp.contacts import (
    foot_floor,
    foot_speed,
    median_filter_labels,
    reference_foot_contacts,
    slew_limit,
    touchdowns,
)

FPS = 50


def make_foot(stance_until=50, swing_until=100, total=150, lift=0.15):
    """One foot: on the ground, a lift-and-step, on the ground again (z of the link origin is 0.03 on the ground)."""
    t = torch.arange(total)
    z = torch.full((total,), 0.03)
    x = torch.zeros(total)
    swing = (t >= stance_until) & (t < swing_until)
    phase = (t[swing] - stance_until) / (swing_until - stance_until)
    z[swing] = 0.03 + lift * torch.sin(torch.pi * phase)
    x[swing] = 0.3 * phase
    x[t >= swing_until] = 0.3
    return torch.stack([x, torch.zeros(total), z], dim=-1)


def test_swing_phase_is_not_contact():
    contact = reference_foot_contacts(make_foot()[:, None, :], FPS)[:, 0]
    assert contact[:45].all()
    assert not contact[55:95].any()
    assert contact[105:].all()


def test_a_foot_that_slides_on_the_floor_is_not_in_contact():
    foot = make_foot(lift=0.0)[:, None, :]  # same horizontal motion, no lift: 0.3 m in 1 s = 0.3 m/s
    assert reference_foot_contacts(foot, FPS, speed_limit=0.2)[60:90].sum() == 0
    assert reference_foot_contacts(foot, FPS, speed_limit=0.5)[60:90].all()


def test_floor_is_relative_to_each_feet_own_minimum():
    foot = make_foot()
    raised = foot.clone()
    raised[:, 2] += 0.2  # a constant offset must not change the labels
    both = torch.stack([foot, raised], dim=1)
    contact = reference_foot_contacts(both, FPS)
    assert torch.equal(contact[:, 0], contact[:, 1])


def test_touchdown_count():
    contact = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1]], dtype=torch.bool)
    assert touchdowns(contact).tolist() == [1, 2]


def test_slew_limit():
    previous = torch.zeros(3)
    target = torch.tensor([1.0, -1.0, 0.05])
    out = slew_limit(target, previous, torch.full((3,), 0.2))
    assert torch.allclose(out, torch.tensor([0.2, -0.2, 0.05]))
    assert torch.allclose(slew_limit(target, target, torch.zeros(3)), target)


def test_foot_floor_and_height():
    foot = make_foot()  # stance at z = 0.03, swing lifts to 0.18 while moving 0.3 m forward
    floor = foot_floor(foot[:, None, :])
    assert torch.allclose(floor, torch.tensor([0.03]))
    speed = foot_speed(foot[:, None, :], FPS)
    assert speed[0, 0].item() == 0.0 and speed[120, 0].item() == 0.0  # still feet
    assert abs(speed[75, 0].item() - 0.3) < 0.02  # mid swing


def test_median_filter_removes_short_runs():
    blip = torch.zeros(21, 1, dtype=torch.bool)
    blip[10] = True  # a single frame of contact in swing
    assert not median_filter_labels(blip, window=3).any()
    gap = torch.ones(21, 1, dtype=torch.bool)
    gap[10] = False  # a single frame without contact in stance
    assert median_filter_labels(gap, window=3).all()
    assert torch.equal(median_filter_labels(blip, window=1), blip)


def test_smoothing_keeps_real_phases_of_the_reference_clip():
    long_stance = torch.zeros(40, 1, dtype=torch.bool)
    long_stance[:20] = True
    labelled = torch.cat([long_stance, ~long_stance, long_stance])
    assert torch.equal(median_filter_labels(labelled, window=3), labelled)
