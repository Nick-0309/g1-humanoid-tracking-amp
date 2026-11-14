"""Record AMP windows of a trained policy at several levels of action noise, for offline discriminator tests.

    python scripts/record_amp_windows.py --policy <run>/model_9999.pt --noise 0,0.1,0.2,0.4 --out windows.pt \
        --kit_args --headless

The envs are split into one group per noise level; each group adds Gaussian noise with that std to the
deterministic action (in action units). Noise 0 is the trained motion; larger noise is progressively worse
motion of the same dance. The file holds the windows, the noise level and the reference step of each window, plus
the expert windows of the clip, everything ``scripts/offline_disc.py`` needs.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # noqa: F401  isort: skip

parser = argparse.ArgumentParser(description="Record AMP windows of a policy.")
parser.add_argument("--policy", required=True)
parser.add_argument("--noise", default="0,0.1,0.2,0.4", help="Comma separated action noise std per env group.")
parser.add_argument("--envs_per_level", type=int, default=32)
parser.add_argument("--steps", type=int, default=1400, help="Control steps (kept below the clip length).")
parser.add_argument("--window", type=int, default=3)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import g1dance.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from g1dance.amp.features import build_windows, expert_frames_from_npz
from g1dance.amp.window import WindowTracker
from g1dance.tasks import AMP_TASK
from g1dance.tasks.env_cfg import PELVIS, configure
from g1dance.tasks.features import TaskFeatures
from g1dance.utils.runs import find_saved_motion
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config


@torch.no_grad()
def main():
    levels = [float(x) for x in args_cli.noise.split(",")]
    run_dir = os.path.dirname(args_cli.policy)
    env_cfg, agent_cfg = resolve_task_config(AMP_TASK, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = args_cli.envs_per_level * len(levels)
    configure(env_cfg, motion_file=find_saved_motion(run_dir), features=TaskFeatures.load(run_dir) or TaskFeatures())
    env_cfg.commands.motion.debug_vis = False

    env = RslRlVecEnvWrapper(gym.make(AMP_TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.policy)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    unwrapped = env.unwrapped
    device = unwrapped.device
    command = unwrapped.command_manager.get_term("motion")

    key_bodies = list(env_cfg.observations.amp.frame.params["key_cfg"].body_names)
    expert_frames = expert_frames_from_npz(env_cfg.commands.motion.motion_file, list(unwrapped.scene["robot"].body_names),
                                           PELVIS, key_bodies, device)
    window = args_cli.window
    sigma = torch.tensor(levels, device=device).repeat_interleave(args_cli.envs_per_level)[:, None]
    steps = min(args_cli.steps, int(command.motion.time_step_total) - 2)

    obs = env.get_observations()
    obs, _, dones, _ = env.step(policy(obs))  # warm-up step, see eval.py
    tracker = WindowTracker(env.num_envs, window, expert_frames.shape[1], device)
    tracker.reset(obs["amp"])
    windows, level_idx, ref_steps = [], [], []
    group = torch.arange(len(levels), device=device).repeat_interleave(args_cli.envs_per_level)
    ends = torch.zeros(len(levels), device=device)
    for _ in range(steps):
        action = policy(obs)
        obs, _, dones, extras = env.step(action + sigma * torch.randn_like(action))
        dones = dones.bool()
        final = extras.get("final_obs")
        frame = obs["amp"] if final is None else torch.where(dones[:, None], final["amp"], obs["amp"])
        ref = command.time_steps - 1  # reference step of the post-physics state, see diagnose_amp.py
        w, valid = tracker.step(frame, dones, obs["amp"])
        keep = valid & ~dones & (ref >= window - 1)
        windows.append(w[keep].cpu())
        level_idx.append(group[keep].cpu())
        ref_steps.append(ref[keep].cpu())
        ends += torch.bincount(group[dones], minlength=len(levels)).float()

    data = {
        "levels": levels,
        "windows": torch.cat(windows),
        "level": torch.cat(level_idx),
        "ref_step": torch.cat(ref_steps),
        "expert": build_windows(expert_frames, window).cpu(),
        "window": window,
        "frame_dim": expert_frames.shape[1],
        "num_joints": unwrapped.scene["robot"].num_joints,
        "num_key_bodies": len(key_bodies),
        "policy": os.path.abspath(args_cli.policy),
    }
    torch.save(data, args_cli.out)
    for i, s in enumerate(levels):
        print(f"noise {s:.2f}: {int((data['level'] == i).sum())} windows, {int(ends[i])} episode ends", flush=True)
    print(f"[INFO] Wrote {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
