"""Play a checkpoint from frame 0 of the clip; optionally record a video and export the policy.

    python scripts/play.py --task G1D-Dance-AMP-v0 --checkpoint logs/g1d_amp/<run>/model_5000.pt --kit_args --headless --video
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a G1 dance policy.")
cli_args.add_task_args(parser, default_task="G1D-Dance-AMP-v0")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--steps", type=int, default=None, help="Control steps to run (default: one clip length).")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the first steps.")
parser.add_argument("--video_length", type=int, default=None, help="Video length in steps (default: --steps).")
parser.add_argument("--no_export", action="store_true", default=False, help="Skip the JIT/ONNX export.")
parser.add_argument("--real_time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import shutil
import tempfile
import time

import torch

from _policy_loader import load_policy  # isort: skip


def export_policy(runner, export_dir: str) -> None:
    """Write policy.pt / policy.onnx. ``torch.jit.save`` cannot write below a non-ASCII path on Windows,
    so export into a temp directory and copy."""
    with tempfile.TemporaryDirectory() as tmp:
        runner.export_policy_to_jit(path=tmp, filename="policy.pt")
        runner.export_policy_to_onnx(path=tmp, filename="policy.onnx")
        os.makedirs(export_dir, exist_ok=True)
        for name in os.listdir(tmp):
            shutil.copy2(os.path.join(tmp, name), export_dir)


def main():
    num_envs = args_cli.num_envs or 1
    video_length = (args_cli.video_length or args_cli.steps or 1500) if args_cli.video else 0
    env, runner, policy, checkpoint = load_policy(args_cli, num_envs, video_length=video_length)
    run_dir = os.path.dirname(checkpoint)
    if not args_cli.no_export:
        export_dir = os.path.join(run_dir, "exported")
        export_policy(runner, export_dir)
        print(f"[INFO] Exported policy to {export_dir}", flush=True)

    command = env.unwrapped.command_manager.get_term("motion")
    steps = args_cli.steps or int(command.motion.time_step_total)
    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    # The first step after construction ends every episode (upstream behaviour); take it as a warm-up.
    with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
        policy.reset(dones)
    failures = 0
    for step in range(steps):
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        failures += int(dones.sum())
        if args_cli.real_time:
            time.sleep(max(0.0, dt - (time.time() - start)))
    print(f"[INFO] Played {steps} steps ({steps * dt:.1f} s) with {num_envs} env(s); episodes ended: {failures}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
