"""Measure every reward term on a deterministic rollout of a checkpoint, to choose weights from data.

    python scripts/calibrate_rewards.py --task G1D-Dance-Tracking-v0 --load_run track_v1 --foot_rewards --kit_args --headless

The tutorial sets the weight of each new term from its measured size on a rollout of a trained policy instead of
copying numbers from papers. This prints, per reward term, the weight, the RMS of the raw value, the RMS of the
weighted value and the share of the summed tracking terms, so a new term can be sized (for example to about 0.5 % of
the main tracking reward) before it is used in training.

``--foot_rewards`` adds the foot terms to the evaluation environment even if the checkpoint was trained without
them (they do not change the observations or actions).
"""

import argparse
import dataclasses
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure reward terms on a policy rollout.")
cli_args.add_task_args(parser, default_task="G1D-Dance-Tracking-v0")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--foot_rewards", action="store_true", help="Also measure the foot reward terms.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from _policy_loader import load_policy  # isort: skip

TRACKING_PREFIX = "motion_"


def main():
    env, _, policy, checkpoint = load_policy(args_cli, args_cli.num_envs or 32, force_foot_rewards=args_cli.foot_rewards)
    unwrapped = env.unwrapped
    manager = unwrapped.reward_manager
    names = list(manager.active_terms)
    weights = {n: manager.get_term_cfg(n).weight for n in names}
    command = unwrapped.command_manager.get_term("motion")
    steps = int(command.motion.time_step_total) - 1
    device = unwrapped.device

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    sum_sq = torch.zeros(len(names), device=device)
    sum_val = torch.zeros(len(names), device=device)
    count = torch.zeros((), device=device)

    obs = env.get_observations()
    with torch.inference_mode():  # warm-up step, see eval.py
        obs, _, dones, _ = env.step(policy(obs))
        policy.reset(dones)
    for _ in range(steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
        alive &= ~dones.bool()
        weighted = manager._step_reward  # (envs, terms): weight * value, per second
        sum_sq += (weighted.square() * alive[:, None]).sum(dim=0)
        sum_val += (weighted * alive[:, None]).sum(dim=0)
        count += alive.sum()

    count = count.clamp(min=1)
    rms_weighted = (sum_sq / count).sqrt()
    mean_weighted = sum_val / count
    tracking = sum(float(rms_weighted[i]) for i, n in enumerate(names) if n.startswith(TRACKING_PREFIX)) or 1.0
    rows = []
    for i, name in enumerate(names):
        w = weights[name]
        rows.append(
            {
                "term": name,
                "weight": w,
                "rms_raw": float(rms_weighted[i]) / abs(w) if w else None,
                "rms_weighted": float(rms_weighted[i]),
                "mean_weighted": float(mean_weighted[i]),
                "share_of_tracking": float(rms_weighted[i]) / tracking,
            }
        )
    rows.sort(key=lambda r: -r["rms_weighted"])

    print(f"{'term':32s} {'weight':>9s} {'rms raw':>9s} {'rms w.':>9s} {'mean w.':>9s} {'share':>8s}")
    for r in rows:
        raw = f"{r['rms_raw']:9.4f}" if r["rms_raw"] is not None else f"{'-':>9s}"
        print(f"{r['term']:32s} {r['weight']:9.4g} {raw} {r['rms_weighted']:9.4f} {r['mean_weighted']:9.4f} {r['share_of_tracking']:8.3%}")
    print(f"(share = RMS of the weighted term / sum of the RMS of the '{TRACKING_PREFIX}*' tracking terms, "
          f"{int(alive.sum())}/{env.num_envs} envs finished the clip)")

    out = os.path.join(os.path.dirname(checkpoint), f"reward_calibration_{os.path.splitext(os.path.basename(checkpoint))[0]}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": checkpoint, "terms": rows}, f, indent=2)
    print(f"[INFO] Wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
