import pytest
import torch

from g1dance.amp.module import AmpCfg, AmpModule

N, T, D = 8, 4, 6


def make_module(**kw) -> AmpModule:
    torch.manual_seed(0)
    cfg = AmpCfg(hidden=(32, 16), minibatches=2, **kw)
    return AmpModule(cfg, torch.randn(60, D), num_envs=N, num_steps=T, device="cpu")


def rollout(amp: AmpModule, inference: bool = True) -> torch.Tensor:
    amp.start(torch.randn(N, D))
    amp.begin_rollout()
    styles = []
    ctx = torch.inference_mode() if inference else torch.enable_grad()
    with ctx:
        for _ in range(T):
            styles.append(amp.step(torch.randn(N, D), torch.zeros(N, dtype=torch.long), torch.randn(N, D)))
    return torch.stack(styles)


def test_first_rollout_has_incomplete_windows():
    amp = make_module()
    styles = rollout(amp)
    assert torch.all(styles[0] == 0)  # only 2 frames seen so far
    metrics = amp.update()
    assert metrics["valid_fraction"] == pytest.approx((T - 1) / T)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_incomplete_windows_get_the_mean_style_reward_of_the_complete_ones():
    amp = make_module()
    amp.start(torch.randn(N, D))
    amp.begin_rollout()
    with torch.inference_mode():
        for _ in range(2):  # the second step is the first with complete windows
            amp.step(torch.randn(N, D), torch.zeros(N, dtype=torch.long), torch.randn(N, D))
        dones = torch.zeros(N, dtype=torch.long)
        dones[:3] = 1  # these envs end now and start a new episode
        amp.step(torch.randn(N, D), dones, torch.randn(N, D))
        style = amp.step(torch.randn(N, D), torch.zeros(N, dtype=torch.long), torch.randn(N, D))
    valid = amp.valid[3]
    assert not valid[:3].any() and valid[3:].all()  # envs 0-2 were reset one step ago
    assert torch.allclose(style[:3], style[3:].mean().expand(3), atol=1e-5)


def test_update_after_rollout_in_inference_mode_changes_discriminator():
    amp = make_module()
    before = [p.clone() for p in amp.disc.parameters()]
    rollout(amp, inference=True)
    amp.update()
    assert any(not torch.equal(a, b) for a, b in zip(before, amp.disc.parameters()))


def test_mix_is_a_lerp():
    amp = make_module(style_lerp=0.25)
    out = amp.mix(torch.tensor([4.0]), torch.tensor([8.0]))
    assert out.item() == pytest.approx(0.75 * 4.0 + 0.25 * 8.0)


def test_warmup_uses_the_task_reward_only_and_survives_a_checkpoint():
    amp = make_module(style_lerp=0.5, disc_warmup=2)
    task, style = torch.tensor([4.0]), torch.tensor([8.0])
    for expected in (4.0, 4.0, 6.0):  # two warm-up updates, then the 0.5 / 0.5 mix
        assert amp.mix(task, style).item() == pytest.approx(expected)
        rollout(amp)
        metrics = amp.update()
    assert metrics["warming_up"] == 0.0 and amp.updates_done == 3

    fresh = make_module(style_lerp=0.5, disc_warmup=2)
    fresh.load_state_dict(amp.state_dict())
    assert not fresh.warming_up
    legacy = {k: v for k, v in amp.state_dict().items() if k != "updates_done"}
    fresh2 = make_module(style_lerp=0.5, disc_warmup=2)
    fresh2.load_state_dict(legacy)
    assert not fresh2.warming_up


def test_checkpoint_roundtrip():
    a, b = make_module(), make_module(seed=5)
    rollout(a)
    a.update()
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, a.cfg.window * D)
    assert torch.allclose(a.score(x), b.score(x))


def test_replay_buffer_wraps_and_is_used():
    amp = make_module(replay_capacity=10)
    for _ in range(3):
        rollout(amp)
        amp.update()
    assert amp._replay_size == 10
    assert 0 <= amp._replay_ptr < 10
    assert torch.isfinite(amp.replay).all()


def test_rollout_overflow_raises():
    amp = make_module()
    rollout(amp)
    with pytest.raises(RuntimeError):
        amp.step(torch.randn(N, D), torch.zeros(N, dtype=torch.long), torch.randn(N, D))
