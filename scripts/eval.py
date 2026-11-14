"""Evaluate a checkpoint: run every env once through the whole clip from frame 0, no noise, no pushes.

    python scripts/eval.py --task G1D-Dance-AMP-v0 --load_run amp_v1 --num_envs 64 --kit_args --headless

Reports the fraction of envs that finish the clip without a termination, the mean tracking errors
(``MotionCommand.metrics``) and the diagnostics the tutorial used to find the cause of jitter: foot touchdowns
against the reference, how often the foot contact state disagrees with the reference, and the 95th percentile
of the step-to-step change of the action and of the joint target. Errors are averaged over the steps in which an
env was still alive.

The ``landing`` block splits the contact disagreement into the two directions and adds the heights that explain
it: the contact rate of the feet the reference has in stance (``stance_contact_rate``, the number to move) and
in swing, the signed vertical foot error (``+`` = the robot's foot is higher than the reference), the height of
the feet above the reference's planted height in stance, and the vertical torso error. A policy that hovers
shows a low ``stance_contact_rate``, a positive ``foot_z_error_stance_mean_m`` and a ``policy_height_stance``
well above the reference's own ``reference_height_stance``.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a G1 dance policy.")
cli_args.add_task_args(parser, default_task="G1D-Dance-AMP-v0")
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--contact_threshold", type=float, default=10.0, help="Contact force [N] that counts as contact.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from _policy_loader import load_policy  # isort: skip
from g1dance.mdp.commands import FEET
from g1dance.mdp.contacts import touchdowns
from g1dance.mdp.rewards import foot_contacts
from isaaclab.managers import SceneEntityCfg


def p95(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    picked = values[mask]
    return float(torch.quantile(picked, 0.95)) if picked.numel() else None


def ratio(sum_: torch.Tensor, count: torch.Tensor) -> float | None:
    return float(sum_ / count) if float(count) > 0 else None


def main():
    env, _, policy, checkpoint = load_policy(args_cli, args_cli.num_envs or 32)
    unwrapped = env.unwrapped
    device = unwrapped.device
    command = unwrapped.command_manager.get_term("motion")
    action_term = unwrapped.action_manager.get_term("JointPositionAction")
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=list(FEET), preserve_order=True)
    sensor_cfg.resolve(unwrapped.scene)

    foot_indexes = [i for i, name in enumerate(command.cfg.body_names) if name in FEET]
    ref_floor = command.ref_foot_floor  # (2,) height of each ankle link when that foot is planted in the clip

    clip_steps = int(command.motion.time_step_total)
    steps = clip_steps - 1
    n = env.num_envs

    alive = torch.ones(n, dtype=torch.bool, device=device)
    fail_step = torch.full((n,), -1, dtype=torch.long, device=device)
    sums: dict[str, torch.Tensor] = {}
    counts = torch.zeros((), device=device)
    mismatch = torch.zeros(steps, n, device=device)
    action_delta = torch.zeros(steps, n, device=device)
    target_delta = torch.zeros(steps, n, device=device)
    valid = torch.zeros(steps, n, dtype=torch.bool, device=device)
    touch = torch.zeros(n, len(FEET), device=device)
    landing = {k: torch.zeros((), device=device) for k in (
        "stance_n", "stance_contact_n", "swing_n", "swing_contact_n", "z_err_stance", "z_err_stance_abs",
        "z_err_swing", "height_stance", "ref_height_stance", "anchor_z",
    )}
    landing["min_height"] = torch.full((), float("inf"), device=device)
    previous_contact = previous_action = previous_target = None

    obs = env.get_observations()
    # The first step after construction ends every episode: the command's relative body poses are still
    # zero when terminations are checked (upstream behaviour). Take it as a warm-up and ignore its dones.
    with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
        policy.reset(dones)
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            contact = foot_contacts(unwrapped, sensor_cfg, args_cli.contact_threshold)
            reference = command.ref_foot_contact[command.time_steps.clamp(max=clip_steps - 1)]
            target = action_term.processed_actions.clone()
            # Heights, in metres. ``+`` means the robot's foot / torso is higher than the reference.
            foot_z = command.robot_body_pos_w[:, foot_indexes, 2]
            foot_z_error = foot_z - command.body_pos_relative_w[:, foot_indexes, 2]
            foot_height = foot_z - ref_floor  # above the height at which the clip's foot is planted
            reference_height = command.ref_foot_height[command.time_steps.clamp(max=clip_steps - 1)]
            anchor_z_error = command.robot_anchor_pos_w[:, 2] - command.anchor_pos_w[:, 2]
        dones = dones.bool()
        fail_step[dones & alive] = step
        alive &= ~dones
        valid[step] = alive
        alive_col = alive[:, None].float()

        for key, value in command.metrics.items():
            if key.startswith("sampling"):
                continue
            sums[key] = sums.get(key, 0.0) + (value * alive).sum()
        counts += alive.sum()

        mismatch[step] = (contact != reference).float().mean(dim=-1)
        if previous_contact is not None:
            touch += (contact & ~previous_contact) * alive[:, None]
            action_delta[step] = (actions - previous_action).abs().mean(dim=-1)
            target_delta[step] = (target - previous_target).abs().max(dim=-1).values
        previous_contact, previous_action, previous_target = contact, actions.clone(), target

        stance = reference & alive[:, None]
        swing = ~reference & alive[:, None]
        landing["stance_n"] += stance.sum()
        landing["stance_contact_n"] += (contact & stance).sum()
        landing["swing_n"] += swing.sum()
        landing["swing_contact_n"] += (contact & swing).sum()
        landing["z_err_stance"] += (foot_z_error * stance).sum()
        landing["z_err_stance_abs"] += (foot_z_error.abs() * stance).sum()
        landing["z_err_swing"] += (foot_z_error * swing).sum()
        landing["height_stance"] += (foot_height * stance).sum()
        landing["ref_height_stance"] += (reference_height * stance).sum()
        landing["anchor_z"] += (anchor_z_error * alive).sum()
        landing["min_height"] = torch.minimum(
            landing["min_height"], torch.where(alive[:, None], foot_height, torch.inf).min()
        )

    dt = unwrapped.step_dt
    failed = fail_step >= 0
    finished = alive
    reference_touchdowns = touchdowns(command.ref_foot_contact).tolist()
    result = {
        "checkpoint": checkpoint,
        "clip_seconds": clip_steps * dt,
        "num_envs": n,
        "success_rate": float(alive.float().mean()),
        "mean_fail_time_s": float(fail_step[failed].float().mean() * dt) if failed.any() else None,
        "metrics_mean": {k: float(v / counts.clamp(min=1)) for k, v in sums.items()},
        "diagnostics": {
            "feet": list(FEET),
            "touchdowns_policy_mean_over_finished_envs": touch[finished].mean(dim=0).tolist() if finished.any() else None,
            "touchdowns_reference": reference_touchdowns,
            "contact_mismatch_rate": float(mismatch[valid].mean()) if valid.any() else None,
            "action_delta_mean_p95": p95(action_delta, valid),
            "joint_target_delta_max_p95_rad": p95(target_delta, valid),
            "landing": {
                "stance_contact_rate": ratio(landing["stance_contact_n"], landing["stance_n"]),
                "swing_contact_rate": ratio(landing["swing_contact_n"], landing["swing_n"]),
                "foot_z_error_stance_mean_m": ratio(landing["z_err_stance"], landing["stance_n"]),
                "foot_z_error_stance_abs_mean_m": ratio(landing["z_err_stance_abs"], landing["stance_n"]),
                "foot_z_error_swing_mean_m": ratio(landing["z_err_swing"], landing["swing_n"]),
                "policy_height_stance_mean_m": ratio(landing["height_stance"], landing["stance_n"]),
                "reference_height_stance_mean_m": ratio(landing["ref_height_stance"], landing["stance_n"]),
                "anchor_z_error_mean_m": ratio(landing["anchor_z"], counts),
                "min_foot_height_above_floor_m": float(landing["min_height"]),
            },
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    out = os.path.join(os.path.dirname(checkpoint), f"eval_{os.path.splitext(os.path.basename(checkpoint))[0]}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
