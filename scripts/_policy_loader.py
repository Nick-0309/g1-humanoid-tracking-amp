"""Build a play/eval environment and load a trained policy. Import only after Kit has started."""

from __future__ import annotations

import dataclasses
import importlib.metadata as metadata
import os

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import cli_args
import g1dance.tasks  # noqa: F401  (registers the tasks)
import isaaclab_tasks  # noqa: F401
from g1dance.tasks.env_cfg import configure
from g1dance.tasks.features import TaskFeatures
from g1dance.utils.runs import find_saved_motion
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config


def load_policy(args, num_envs: int, video_length: int = 0, force_foot_rewards: bool = False):
    """Returns ``(env, runner, policy, checkpoint_path)``; the env starts at frame 0 of the clip.

    ``video_length > 0`` records the first steps to ``<run dir>/videos/play``. ``force_foot_rewards`` adds the foot
    reward terms to a run that was trained without them (they do not change observations or actions).
    """
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    agent_cfg = cli_args.update_agent_cfg(agent_cfg, args)
    env_cfg.scene.num_envs = num_envs

    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = retrieve_file_path(args.checkpoint)
    else:
        checkpoint = get_checkpoint_path(cli_args.log_root_for(agent_cfg), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint {checkpoint}", flush=True)

    # The observation and action layout must match the run that wrote the checkpoint.
    run_dir = os.path.dirname(checkpoint)
    features = TaskFeatures.load(run_dir)
    if features is None:
        print("[INFO] no features.json next to the checkpoint; assuming the default task layout", flush=True)
    if force_foot_rewards:
        features = dataclasses.replace(features or TaskFeatures(), foot_rewards=True)
    configure(
        env_cfg,
        motion_file=args.motion or find_saved_motion(run_dir),
        episode_length_s=args.episode_length_s,
        features=features,
    )

    if video_length > 0:
        # Isaac Lab 3.0 records from a visualizer and is driven by env.step(); skip the first steps so the
        # renderer has warmed up.
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab_visualizers.kit import KitVisualizerCfg

        video_dir = os.path.join(os.path.dirname(checkpoint), "videos", "play")
        print(f"[INFO] Recording video to {video_dir}", flush=True)
        env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=(2.6, 2.6, 1.4), lookat=(0.0, 0.0, 0.8))]
        env_cfg.commands.motion.debug_vis = False  # the goal/current frame markers clutter the video
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit", output_dir=video_dir, video_length=video_length, step_offset=5
            )
        ]

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return env, runner, policy, checkpoint
