"""Shared command line helpers for the entry scripts."""

from __future__ import annotations

import argparse
import os
import random

from g1dance.tasks.features import TaskFeatures, parse_offsets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_ROOT = os.path.join(PROJECT_ROOT, "logs")


def add_task_args(parser: argparse.ArgumentParser, default_task: str) -> None:
    group = parser.add_argument_group("task")
    group.add_argument("--task", type=str, default=default_task, help="Gym task id.")
    group.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    group.add_argument("--seed", type=int, default=None, help="Random seed (-1 for a random one).")
    group.add_argument("--motion", type=str, default=None, help="Path of the motion .npz (default: motions/dance_102.npz).")
    group.add_argument("--episode_length_s", type=float, default=None, help="Episode length in seconds.")
    group.add_argument(
        "--key_bodies", type=str, default=None, help="Comma separated key body names used in the AMP frame."
    )


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("task features (saved with the run; play/eval restore them)")
    group.add_argument("--future_ref", type=str, default=None,
                       help="Comma separated control-step offsets whose reference pose is also observed, e.g. 5,10.")
    group.add_argument("--residual", action="store_true", help="Action = residual on the next reference pose.")
    group.add_argument("--residual_scale", type=float, default=None, help="Rad per unit action (default: upstream scale).")
    group.add_argument("--slew_limit", action="store_true", help="Rate limit the residual (needs --residual).")
    group.add_argument("--foot_rewards", action="store_true", help="Add the landing reward bundle (contact + heights).")
    group.add_argument("--contact_obs", action="store_true",
                       help="Also observe reference stance labels and measured foot contact (4 more numbers).")
    group.add_argument("--foot_termination", type=float, default=None,
                       help="Terminate when a foot's vertical error to the reference exceeds this many metres.")


def features_from_args(args: argparse.Namespace) -> TaskFeatures:
    return TaskFeatures(
        future_ref=parse_offsets(args.future_ref),
        residual=args.residual,
        residual_scale=args.residual_scale,
        slew_limit=args.slew_limit,
        foot_rewards=args.foot_rewards,
        contact_obs=args.contact_obs,
        foot_termination=args.foot_termination,
    )


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("rsl_rl")
    group.add_argument("--experiment_name", type=str, default=None, help="Log folder name below logs/.")
    group.add_argument("--run_name", type=str, default=None, help="Suffix of the run folder.")
    group.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
    group.add_argument(
        "--auto_resume",
        action="store_true",
        default=False,
        help="Resume from the newest checkpoint of an earlier run with the same --run_name, if there is one. "
        "--max_iterations is then the total, not the number of additional iterations.",
    )
    group.add_argument("--load_run", type=str, default=None, help="Run folder to load from (default: latest).")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file name or path.")
    group.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard", "neptune"])
    group.add_argument("--log_project_name", type=str, default=None)


def update_agent_cfg(agent_cfg, args: argparse.Namespace):
    """Apply the rsl_rl related command line arguments to the agent config."""
    if args.seed is not None:
        agent_cfg.seed = random.randint(0, 10000) if args.seed == -1 else args.seed
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.resume:
        agent_cfg.resume = True
    if args.load_run is not None:
        agent_cfg.load_run = f".*{args.load_run}.*"  # rsl_rl matches from the start; accept any substring
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args.log_project_name:
        agent_cfg.wandb_project = args.log_project_name
        agent_cfg.neptune_project = args.log_project_name
    return agent_cfg


def log_root_for(agent_cfg) -> str:
    return os.path.join(LOG_ROOT, agent_cfg.experiment_name)
