"""Offline discriminator experiment: which settings keep the AMP style reward informative?

    python scripts/offline_disc.py cloud_runs/windows_tracking.pt [--steps 3000]

Trains a fresh discriminator per configuration on expert windows vs. recorded policy windows, the way the AMP
runner does (LSGAN, mini-batches, running normalization), and then scores held-out windows of every noise level.
A useful discriminator gives the noise-free policy a clearly positive style reward and ranks the noise levels in
order; a saturated one puts every level at D = -1 (style reward 0). Pure torch, no Isaac Sim needed.

During training only the "policy" windows of the training noise level(s) are used as negatives, mimicking an
on-policy rollout of a policy of that quality.
"""

from __future__ import annotations

import argparse
import time

import torch

from g1dance.amp.discriminator import STYLE_REWARDS, AmpDiscriminator, discriminator_loss, style_reward

CONFIGS = {
    # name: (gp_mode, gp_coef, lr, logit_reg, weight_decay, input_noise)
    "current (interp, 10)": ("interp", 10.0, 3e-5, 0.0, 0.0, 0.0),
    "paper (expert, 5)": ("expert", 5.0, 3e-5, 0.0, 0.0, 0.0),
    "expert gp 5 + logit 0.05": ("expert", 5.0, 3e-5, 0.05, 0.0, 0.0),
    "expert gp 5 + logit + wd 1e-4": ("expert", 5.0, 3e-5, 0.05, 1e-4, 0.0),
    "expert gp 5 + logit + noise 0.1": ("expert", 5.0, 3e-5, 0.05, 0.0, 0.1),
    "expert gp 20 + logit 0.05": ("expert", 20.0, 3e-5, 0.05, 0.0, 0.0),
    "expert gp 5 + logit + wd + noise": ("expert", 5.0, 3e-5, 0.05, 1e-4, 0.1),
}


def train(cfg, expert, negatives, frame_dim, window, steps, batch, device, seed=0):
    gp_mode, gp_coef, lr, logit_reg, weight_decay, input_noise = cfg
    torch.manual_seed(seed)
    disc = AmpDiscriminator(frame_dim, window, (1024, 512)).to(device)
    disc.norm.update(expert.reshape(-1, frame_dim))
    disc.norm.update(negatives.reshape(-1, frame_dim))
    opt = torch.optim.Adam(disc.parameters(), lr=lr)
    for _ in range(steps):
        e = expert[torch.randint(0, expert.shape[0], (batch,), device=device)]
        p = negatives[torch.randint(0, negatives.shape[0], (batch,), device=device)]
        loss, _ = discriminator_loss(disc, e, p, gp_coef, gp_mode, logit_reg, weight_decay, input_noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt.step()
    disc.eval()
    return disc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("windows")
    p.add_argument("--steps", type=int, default=3000, help="Discriminator updates (the cloud run did ~4 per PPO iteration).")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--train_levels", default="0", help="Noise level indices used as policy negatives.")
    p.add_argument("--only", default=None, help="Comma separated substrings: run only matching configs.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(args.windows, weights_only=False)
    windows, level, ref = data["windows"].to(device), data["level"].to(device), data["ref_step"].to(device)
    expert = data["expert"].to(device)
    window, frame_dim = data["window"], data["frame_dim"]
    levels = data["levels"]

    # Hold out every 4th reference step: test windows come from time steps the discriminator never trained on.
    test = (ref % 4) == 0
    train_levels = [int(x) for x in args.train_levels.split(",")]
    negatives = windows[~test & torch.isin(level, torch.tensor(train_levels, device=device))]
    expert_steps = torch.arange(expert.shape[0], device=device) + (window - 1)
    expert_train, expert_test = expert[(expert_steps % 4) != 0], expert[(expert_steps % 4) == 0]
    print(f"{windows.shape[0]} policy windows, levels {levels}; negatives: levels {[levels[i] for i in train_levels]} "
          f"({negatives.shape[0]}); expert train/test {expert_train.shape[0]}/{expert_test.shape[0]}")

    print("D = mean discriminator score per noise level; r = mean style reward per level and mapping.")
    print("A useful setting: r@0.0 clearly above the others and r decreasing with noise.")
    for name, cfg in CONFIGS.items():
        if args.only and not any(s in name for s in args.only.split(",")):
            continue
        t0 = time.time()
        disc = train(cfg, expert_train, negatives, frame_dim, window, args.steps, args.batch, device)
        with torch.no_grad():
            d_exp = float(disc(expert_test).mean())
            scores = [disc(windows[test & (level == i)]) for i in range(len(levels))]
        print(f"\n{name}  (D expert {d_exp:+.2f}, {time.time() - t0:.0f} s)")
        print("  D        " + " ".join(f"{float(d.mean()):+7.2f}" for d in scores) + "   noise " + str(levels))
        for kind in STYLE_REWARDS:
            r = [float(style_reward(d, kind).mean()) for d in scores]
            order_ok = all(a >= b for a, b in zip(r, r[1:]))
            print(f"  r {kind:9s}" + " ".join(f"{x:7.3f}" for x in r) + ("   monotonic" if order_ok else ""), flush=True)


if __name__ == "__main__":
    main()
