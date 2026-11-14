import torch

from g1dance.amp.window import WindowTracker


def encode(ep: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    return (ep * 1000 + step).float().unsqueeze(-1)


def test_windows_never_straddle_a_reset():
    n, window = 4, 3
    lengths = torch.tensor([5, 7, 3, 100])
    ep, step = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    tracker = WindowTracker(n, window, 1, "cpu")
    tracker.reset(encode(ep, step))

    valid_count = torch.zeros(n, dtype=torch.long)
    for _ in range(40):
        step += 1
        dones = step >= lengths
        next_frame = encode(ep, step)  # the terminal frame for envs that finish now
        ep_next = ep + dones.long()
        step_next = torch.where(dones, torch.zeros_like(step), step)
        start_frame = encode(ep_next, step_next)  # what the env observes after the reset

        windows, valid = tracker.step(next_frame, dones, start_frame)
        frames = windows.reshape(n, window)
        for i in range(n):
            if valid[i]:
                valid_count[i] += 1
                eps = (frames[i] // 1000).long()
                steps = (frames[i] % 1000).long()
                assert torch.all(eps == ep[i])
                assert torch.equal(steps, torch.arange(step[i] - 2, step[i] + 1))
            else:
                assert step[i] < window - 1

        ep, step = ep_next, step_next

    # env 3 never finishes: a valid window at every step except the first one
    assert valid_count[3] == 39


def test_reset_seeds_next_episode_with_start_frame():
    tracker = WindowTracker(1, 3, 1, "cpu")
    tracker.reset(torch.tensor([[10.0]]))
    tracker.step(torch.tensor([[11.0]]), torch.tensor([0]), torch.tensor([[11.0]]))
    tracker.step(torch.tensor([[99.0]]), torch.tensor([1]), torch.tensor([[50.0]]))  # terminal, then reset
    assert tracker.count.item() == 1
    assert tracker.frames[0, -1].item() == 50.0
    assert torch.all(tracker.frames[0, :-1] == 0)


def test_updates_work_inside_inference_mode():
    tracker = WindowTracker(2, 3, 2, "cpu")
    tracker.reset(torch.zeros(2, 2))
    with torch.inference_mode():
        for _ in range(3):
            tracker.step(torch.ones(2, 2), torch.zeros(2, dtype=torch.long), torch.zeros(2, 2))
    tracker.reset(torch.zeros(2, 2))  # outside again; must not raise
