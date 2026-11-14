"""Find out what the AMP discriminator uses to tell policy motion from the reference.

    python scripts/diagnose_amp.py --disc_checkpoint <amp run>/model_6750.pt --policy <run>/model_9999.pt \
        --label tracking --kit_args --headless

Runs a deterministic rollout of ``--policy`` (from frame 0, no noise, no pushes, no randomisation), records AMP
windows together with the reference time step of each, and builds the expert window that ends at the same step.
Then, per feature group, it reports how the discriminator score changes when that group of the policy windows is
replaced by the expert's (and the reverse), plus statistics that point at the cause: RMS, the frame-to-frame
second difference, and how far joint velocities are from the finite difference of joint positions.

One policy per run (an Isaac Sim process should create only one environment). The report is printed and written
to ``<dir of disc_checkpoint>/amp_diagnosis_<label>.json``.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # noqa: F401  isort: skip  (puts scripts/ on the import path like the other entry points)

parser = argparse.ArgumentParser(description="Diagnose the AMP discriminator.")
parser.add_argument("--disc_checkpoint", required=True, help="Checkpoint of an AMP run (holds the discriminator).")
parser.add_argument("--policy", required=True, help="Policy checkpoint to roll out (tracking or AMP run).")
parser.add_argument("--label", required=True, help="Name of this policy in the report file.")
parser.add_argument("--motion", default=None, help="Motion .npz (default: the one saved with the policy run).")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=600, help="Control steps per rollout (kept below the clip length).")
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
from g1dance.amp.diagnostics import group_rms, group_slices, second_difference, swap_group, velocity_mismatch
from g1dance.amp.discriminator import AmpDiscriminator, style_reward
from g1dance.amp.features import build_windows, expert_frames_from_npz
from g1dance.amp.window import WindowTracker
from g1dance.tasks import AMP_TASK
from g1dance.tasks.env_cfg import PELVIS, configure
from g1dance.tasks.features import TaskFeatures
from g1dance.utils.runs import find_saved_motion
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config


def load_discriminator(path: str, device: str) -> AmpDiscriminator:
    state = torch.load(path, weights_only=True, map_location=device)["amp"]["disc"]
    linear = sorted((k for k in state if k.startswith("mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    hidden = tuple(state[k].shape[0] for k in linear[:-1])
    frame_dim = state["norm.mean"].shape[0]
    window = state[linear[0]].shape[1] // frame_dim
    disc = AmpDiscriminator(frame_dim, window, hidden).to(device)
    disc.load_state_dict(state)
    disc.eval()
    return disc


def summarize(scores: torch.Tensor) -> dict:
    return {
        "D_mean": float(scores.mean()),
        "D_p10": float(torch.quantile(scores, 0.1)),
        "D_p90": float(torch.quantile(scores, 0.9)),
        "style_reward_mean": float(style_reward(scores).mean()),
        "frac_D_above_0": float((scores > 0).float().mean()),
    }


@torch.no_grad()
def main():
    device = "cuda:0"
    disc = load_discriminator(args_cli.disc_checkpoint, device)
    window = disc.window

    run_dir = os.path.dirname(args_cli.policy)
    features = TaskFeatures.load(run_dir) or TaskFeatures()
    motion = args_cli.motion or find_saved_motion(run_dir)
    env_cfg, agent_cfg = resolve_task_config(AMP_TASK, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    configure(env_cfg, motion_file=motion, features=features)
    env_cfg.commands.motion.debug_vis = False

    env = RslRlVecEnvWrapper(gym.make(AMP_TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.policy)  # a tracking checkpoint has no "amp" entry; the PPO part loads the same way
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("motion")

    key_cfg = env_cfg.observations.amp.frame.params["key_cfg"]
    body_names = list(unwrapped.scene["robot"].body_names)
    expert_frames = expert_frames_from_npz(env_cfg.commands.motion.motion_file, body_names, PELVIS,
                                           list(key_cfg.body_names), device)
    if expert_frames.shape[1] != disc.frame_dim:
        raise SystemExit(f"frame size {expert_frames.shape[1]} does not match the discriminator ({disc.frame_dim})")
    expert_all = build_windows(expert_frames, window)
    total = int(command.motion.time_step_total)
    steps = min(args_cli.steps, total - 2)  # the clip wraps around at its end: stay below that

    obs = env.get_observations()
    obs, _, dones, _ = env.step(policy(obs))  # warm-up step, see eval.py
    tracker = WindowTracker(env.num_envs, window, disc.frame_dim, unwrapped.device)
    tracker.reset(obs["amp"])
    policy_windows, ref_steps, episode_ends = [], [], 0
    for _ in range(steps):
        obs, _, dones, extras = env.step(policy(obs))
        dones = dones.bool()
        final = extras.get("final_obs")
        frame = obs["amp"] if final is None else torch.where(dones[:, None], final["amp"], obs["amp"])
        # Rewards compare the post-physics state with reference step t, then the command advances to t + 1
        # before observations are computed: the frame belongs to time_steps - 1.
        ref = command.time_steps - 1
        windows, valid = tracker.step(frame, dones, obs["amp"])
        keep = valid & ~dones & (ref >= window - 1)
        policy_windows.append(windows[keep])
        ref_steps.append(ref[keep])
        episode_ends += int(dones.sum())

    policy_w = torch.cat(policy_windows)
    expert_w = expert_all[torch.cat(ref_steps) - (window - 1)]
    num_joints = unwrapped.scene["robot"].num_joints
    groups = group_slices(num_joints, len(key_cfg.body_names))

    report = {
        "label": args_cli.label,
        "policy_checkpoint": os.path.abspath(args_cli.policy),
        "disc_checkpoint": os.path.abspath(args_cli.disc_checkpoint),
        "windows": int(policy_w.shape[0]),
        "episode_ends": episode_ends,
        "policy": summarize(disc(policy_w)),
        "expert_same_steps": summarize(disc(expert_w)),
        "replace_one_group_of_policy_with_expert": {},
        "replace_one_group_of_expert_with_policy": {},
        "group_stats": {},
    }
    for g, cols in groups.items():
        report["replace_one_group_of_policy_with_expert"][g] = summarize(disc(swap_group(policy_w, expert_w, cols, window)))
        report["replace_one_group_of_expert_with_policy"][g] = summarize(disc(swap_group(expert_w, policy_w, cols, window)))
        report["group_stats"][g] = {
            "rms_policy": group_rms(policy_w, cols, window),
            "rms_expert": group_rms(expert_w, cols, window),
            "second_diff_rms_policy": float(second_difference(policy_w, window)[:, cols].pow(2).mean().sqrt()),
            "second_diff_rms_expert": float(second_difference(expert_w, window)[:, cols].pow(2).mean().sqrt()),
            "rms_policy_minus_expert": group_rms(policy_w - expert_w, cols, window),
        }
    dt = unwrapped.step_dt
    report["joint_vel_minus_finite_difference_rms"] = {
        "policy": velocity_mismatch(policy_w, window, groups["joint_pos"], groups["joint_vel"], dt),
        "expert": velocity_mismatch(expert_w, window, groups["joint_pos"], groups["joint_vel"], dt),
    }
    # Only the velocity groups replaced: if this alone moves D(policy) to the expert side, D keys on velocity noise.
    both = swap_group(swap_group(policy_w, expert_w, groups["joint_vel"], window), expert_w, groups["base_ang_vel"], window)
    report["policy_with_expert_velocities"] = summarize(disc(both))

    print(f"\n===== {args_cli.label}: {report['windows']} windows, {episode_ends} episode ends in {steps} steps")
    print(f"D(policy) {report['policy']['D_mean']:+.3f}   D(expert, same steps) {report['expert_same_steps']['D_mean']:+.3f}"
          f"   D(policy with expert velocities) {report['policy_with_expert_velocities']['D_mean']:+.3f}")
    print(f"{'group':14s} {'pol<-exp':>9s} {'exp<-pol':>9s} {'rms pol':>9s} {'rms exp':>9s} {'d2 pol':>8s} "
          f"{'d2 exp':>8s} {'rms diff':>9s}")
    for g, s in report["group_stats"].items():
        a = report["replace_one_group_of_policy_with_expert"][g]["D_mean"]
        b = report["replace_one_group_of_expert_with_policy"][g]["D_mean"]
        print(f"{g:14s} {a:+9.3f} {b:+9.3f} {s['rms_policy']:9.3f} {s['rms_expert']:9.3f} {s['second_diff_rms_policy']:8.3f} "
              f"{s['second_diff_rms_expert']:8.3f} {s['rms_policy_minus_expert']:9.3f}")
    m = report["joint_vel_minus_finite_difference_rms"]
    print(f"joint_vel - finite difference of joint_pos, RMS rad/s: policy {m['policy']:.3f}  expert {m['expert']:.3f}")

    out = os.path.join(os.path.dirname(args_cli.disc_checkpoint), f"amp_diagnosis_{args_cli.label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
