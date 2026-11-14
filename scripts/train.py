"""Train the G1 dance policy: tracking baseline (G1D-Dance-Tracking-v0) or tracking + AMP (G1D-Dance-AMP-v0).

    python scripts/train.py --task G1D-Dance-AMP-v0 --num_envs 4096 --kit_args --headless
"""

import argparse
import dataclasses
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train a G1 dance policy.")
cli_args.add_task_args(parser, default_task="G1D-Dance-AMP-v0")
cli_args.add_feature_args(parser)
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default from the agent config).")
amp_group = parser.add_argument_group("amp")
amp_group.add_argument("--style_lerp", type=float, default=None, help="r = (1 - w) * r_task + w * r_style.")
amp_group.add_argument("--window", type=int, default=None, help="Frames per discriminator input.")
amp_group.add_argument("--disc_lr", type=float, default=None)
amp_group.add_argument("--gp_coef", type=float, default=None)
amp_group.add_argument("--gp_mode", type=str, default=None, choices=["interp", "expert"])
amp_group.add_argument("--disc_epochs", type=int, default=None)
amp_group.add_argument("--disc_minibatches", type=int, default=None)
amp_group.add_argument("--replay_capacity", type=int, default=None)
amp_group.add_argument("--style_reward", type=str, default=None, choices=["tutorial", "paper", "softplus", "linear"],
                       help="Discriminator score -> style reward mapping.")
amp_group.add_argument("--logit_reg", type=float, default=None, help="L2 on the discriminator output layer.")
amp_group.add_argument("--disc_weight_decay", type=float, default=None, help="L2 on all discriminator weights.")
amp_group.add_argument("--disc_input_noise", type=float, default=None, help="Noise std on normalized D inputs.")
amp_group.add_argument("--disc_warmup", type=int, default=None,
                       help="PPO iterations at the start in which only the discriminator learns (style reward off).")
parser.add_argument("--init_from", type=str, default=None,
                    help="Checkpoint whose actor/critic weights start this run (e.g. a trained tracking policy). "
                         "Iterations, optimizer and discriminator start fresh. Its task features are reused.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below needs the running Kit app."""

import importlib.metadata as metadata
import inspect
import os
import re
import shutil
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

import g1dance.tasks  # noqa: F401  (registers the tasks)
from g1dance.amp.features import expert_frames_from_npz
from g1dance.amp.module import AmpCfg, AmpModule
from g1dance.amp.runner import AmpOnPolicyRunner
from g1dance.tasks.env_cfg import PELVIS, configure
from g1dance.tasks.features import TaskFeatures
from g1dance.utils.runs import find_saved_motion, latest_run_with_checkpoint
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_amp(env, args, motion_file: str, num_steps: int, device: str) -> tuple[AmpCfg, AmpModule]:
    overrides = {
        "style_lerp": args.style_lerp,
        "window": args.window,
        "lr": args.disc_lr,
        "gp_coef": args.gp_coef,
        "gp_mode": args.gp_mode,
        "epochs": args.disc_epochs,
        "minibatches": args.disc_minibatches,
        "replay_capacity": args.replay_capacity,
        "style_reward": args.style_reward,
        "logit_reg": args.logit_reg,
        "weight_decay": args.disc_weight_decay,
        "input_noise": args.disc_input_noise,
        "disc_warmup": args.disc_warmup,
        "seed": args.seed if args.seed not in (None, -1) else 0,
    }
    amp_cfg = AmpCfg(**{k: v for k, v in overrides.items() if v is not None})

    unwrapped = env.unwrapped
    key_bodies = unwrapped.cfg.observations.amp.frame.params["key_cfg"].body_names
    expert = expert_frames_from_npz(motion_file, list(unwrapped.scene["robot"].body_names), PELVIS, key_bodies, device)
    amp = AmpModule(amp_cfg, expert, num_envs=unwrapped.num_envs, num_steps=num_steps, device=device)
    print(
        f"[AMP] frame_dim={amp.frame_dim} window={amp_cfg.window} expert_windows={amp.expert.shape[0]} "
        f"key_bodies={list(key_bodies)}",
        flush=True,
    )
    return amp_cfg, amp


def report_reset_distance(env, expert_frames: torch.Tensor) -> None:
    """Sanity check: frames of freshly reset envs should lie close to the expert frames."""
    frames = env.get_observations()["amp"]
    nearest = torch.cdist(frames, expert_frames.to(frames.device)).min(dim=1).values
    scale = expert_frames.norm(dim=-1).median()
    print(
        f"[AMP] reset frames: median distance to the nearest expert frame {nearest.median():.3f} "
        f"(median expert frame norm {scale:.3f})",
        flush=True,
    )
    if nearest.median() > scale:
        print("[AMP][WARNING] reset frames are far from the expert frames; check body order and units.", flush=True)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_agent_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # -- where to resume from (decided first: a resumed run must keep the task layout of the earlier one)
    log_root_path = cli_args.log_root_for(agent_cfg)
    if args_cli.auto_resume:
        if not agent_cfg.run_name:
            raise SystemExit("--auto_resume needs --run_name to find the earlier run")
        earlier = latest_run_with_checkpoint(log_root_path, agent_cfg.run_name)
        if earlier is not None:
            print(f"[INFO] --auto_resume: continuing run {earlier}", flush=True)
            agent_cfg.resume, agent_cfg.load_run, agent_cfg.load_checkpoint = True, f"^{earlier}$", "model_.*.pt"
        else:
            print("[INFO] --auto_resume: no earlier run, starting fresh", flush=True)
    resume_path = (
        get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint) if agent_cfg.resume else None
    )

    features = cli_args.features_from_args(args_cli)
    motion = args_cli.motion
    if resume_path is not None:
        run_dir = os.path.dirname(resume_path)
        saved = TaskFeatures.load(run_dir)
        if saved is not None and saved != features:
            print(f"[INFO] using the task features of the run being resumed: {saved}", flush=True)
            features = saved
        motion = motion or find_saved_motion(run_dir)
        if args_cli.auto_resume:
            # The log folder is new and --max_iterations is the total; rsl_rl restarts at the saved index.
            done = int(re.search(r"model_(\d+)\.pt", os.path.basename(resume_path)).group(1))
            agent_cfg.max_iterations = max(0, agent_cfg.max_iterations - done)
            print(f"[INFO] {done} iterations done, {agent_cfg.max_iterations} to go", flush=True)
            if agent_cfg.max_iterations == 0:
                print("[INFO] training already finished", flush=True)
                return
    elif args_cli.init_from is not None:
        if not os.path.isfile(args_cli.init_from):
            raise SystemExit(f"--init_from: {args_cli.init_from} does not exist")
        source = os.path.dirname(os.path.abspath(args_cli.init_from))
        saved = TaskFeatures.load(source)
        if saved is not None and saved != features:
            # The actor's input/output sizes depend on the layout, not on the reward terms: keep the source
            # layout (otherwise the weights would not load) and this run's reward/termination features, so a
            # checkpoint trained without the landing rewards can be fine-tuned with them.
            features = features.with_layout_from(saved)
            print(f"[INFO] --init_from: layout of the source run, features of this run: {features}", flush=True)
        motion = motion or find_saved_motion(source)

    key_bodies = args_cli.key_bodies.split(",") if args_cli.key_bodies else None
    configure(
        env_cfg,
        motion_file=motion,
        episode_length_s=args_cli.episode_length_s,
        key_bodies=key_bodies,
        features=features,
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    use_amp = hasattr(env_cfg.observations, "amp")
    motion_file = env_cfg.commands.motion.motion_file
    print(f"[INFO] task features: {features} (tag {features.tag})", flush=True)

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging to {log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # The wrapper resets once without a seed; re-seed so runs are reproducible.
    env.env.reset(seed=agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    train_cfg = agent_cfg.to_dict()
    if use_amp:
        amp_cfg, amp = build_amp(env, args_cli, motion_file, agent_cfg.num_steps_per_env, agent_cfg.device)
        report_reset_distance(env, amp.expert[:, : amp.frame_dim])
        runner = AmpOnPolicyRunner(env, train_cfg, amp, log_dir=log_dir, device=agent_cfg.device)
        dump_yaml(os.path.join(log_dir, "params", "amp.yaml"), dataclasses.asdict(amp_cfg))
    else:
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] Resuming from {resume_path}", flush=True)
        runner.load(resume_path)
    elif args_cli.init_from is not None:
        print(f"[INFO] Initialising actor and critic from {args_cli.init_from}", flush=True)
        runner.alg.load(
            torch.load(args_cli.init_from, weights_only=False, map_location=agent_cfg.device),
            {"actor": True, "critic": True, "optimizer": False, "iteration": False},
            strict=True,
        )

    features.save(log_dir)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )
    shutil.copy(motion_file, os.path.join(log_dir, "params", os.path.basename(motion_file)))

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
