import pytest
import torch

from g1dance.amp.discriminator import AmpDiscriminator, discriminator_loss, style_reward
from g1dance.amp.normalizer import RunningNorm


def test_style_reward_range_and_monotonic():
    d = torch.linspace(-6, 6, 241)
    r = style_reward(d)
    assert torch.all(r >= 0)
    assert torch.all(r[1:] >= r[:-1] - 1e-7)
    assert style_reward(torch.tensor(-1.0)) == 0
    assert style_reward(torch.tensor(0.0)) == pytest.approx(0.2877, abs=1e-3)


def test_style_reward_not_exploitable_below_minus_one():
    assert style_reward(torch.tensor(-3.0)) == 0
    assert style_reward(torch.tensor(-10.0)) == 0


def test_other_style_rewards():
    d = torch.linspace(-6, 3, 181)
    paper = style_reward(d, "paper")
    assert paper.min() == 0 and style_reward(torch.tensor(1.0), "paper") == 1
    assert torch.all(paper[d <= -1] == 0)
    soft = style_reward(d, "softplus")
    assert torch.all(soft > 0)
    below = d < 1
    assert torch.all(soft[below][1:] > soft[below][:-1])  # strictly increasing up to the clamp
    assert soft[d >= 1].unique().numel() == 1  # flat above +1
    lin = style_reward(d, "linear")
    assert style_reward(torch.tensor(-1.0), "linear") == 0 and style_reward(torch.tensor(1.0), "linear") == 1
    assert lin.min() == -1.0 and lin.max() == 1.0
    with pytest.raises(ValueError):
        style_reward(d, "nope")


def test_linear_reward_prefers_steady_scores_over_spread_out_lower_ones():
    torch.manual_seed(0)
    steady = -1.0 + 0.05 * torch.randn(10000)
    spread = -1.3 + 1.0 * torch.randn(10000)  # lower on average, but with a high tail
    assert style_reward(spread, "tutorial").mean() > style_reward(steady, "tutorial").mean()  # the problem
    assert style_reward(steady, "linear").mean() > style_reward(spread, "linear").mean()


def test_running_norm_matches_batch_statistics():
    torch.manual_seed(0)
    x = torch.randn(1000, 4) * torch.tensor([1.0, 2.0, 0.5, 3.0]) + torch.tensor([0.0, 5.0, -1.0, 2.0])
    norm = RunningNorm(4)
    for chunk in x.chunk(7):
        norm.update(chunk)
    assert torch.allclose(norm.mean, x.mean(0), atol=1e-2)
    assert torch.allclose(norm.var, x.var(0, unbiased=False), rtol=2e-2)


def make_data(n=256, frame_dim=6, window=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    expert = torch.randn(n, frame_dim * window, generator=g) * 0.3 + 1.0
    policy = torch.randn(n, frame_dim * window, generator=g) * 0.3 - 1.0
    return expert, policy


@pytest.mark.parametrize("gp_mode", ["interp", "expert"])
def test_discriminator_separates_distributions(gp_mode):
    torch.manual_seed(0)
    expert, policy = make_data()
    disc = AmpDiscriminator(6, 2, (64, 32))
    disc.norm.update(torch.cat([expert, policy]).reshape(-1, 6))
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    for _ in range(150):
        loss, metrics = discriminator_loss(disc, expert, policy, gp_coef=10.0, gp_mode=gp_mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert metrics["disc_acc"] > 0.95
    assert metrics["d_expert"] > 0.3 > -0.3 > metrics["d_policy"]
    assert torch.isfinite(metrics["grad_penalty"])


def test_gradient_penalty_reaches_parameters():
    expert, policy = make_data(n=32)
    disc = AmpDiscriminator(6, 2, (16,))
    loss, _ = discriminator_loss(disc, expert, policy, gp_coef=10.0, gp_mode="interp")
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in disc.parameters())


def test_regularizers_add_to_the_loss_and_are_off_by_default():
    torch.manual_seed(0)
    expert, policy = make_data(n=32)
    disc = AmpDiscriminator(6, 2, (16,))
    base, _ = discriminator_loss(disc, expert, policy, gp_coef=0.0)
    logit, _ = discriminator_loss(disc, expert, policy, gp_coef=0.0, logit_reg=1.0)
    decay, _ = discriminator_loss(disc, expert, policy, gp_coef=0.0, weight_decay=1.0)
    out_w = disc.mlp[-1].weight
    assert torch.allclose(logit - base, out_w.pow(2).sum(), atol=1e-5)
    all_w = sum(m.weight.pow(2).sum() for m in disc.mlp if isinstance(m, torch.nn.Linear))
    assert torch.allclose(decay - base, all_w, atol=1e-4)
    torch.manual_seed(1)
    noisy, _ = discriminator_loss(disc, expert, policy, gp_coef=0.0, input_noise=0.5)
    assert not torch.allclose(noisy, base)


def test_unknown_gp_mode_raises():
    expert, policy = make_data(n=8)
    with pytest.raises(ValueError):
        discriminator_loss(AmpDiscriminator(6, 2, (8,)), expert, policy, gp_mode="nope")


def test_discriminator_state_dict_includes_normalizer():
    a, b = AmpDiscriminator(6, 2, (8,)), AmpDiscriminator(6, 2, (8,))
    a.norm.update(torch.randn(100, 6) + 3.0)
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 12)
    assert torch.allclose(a(x), b(x))
