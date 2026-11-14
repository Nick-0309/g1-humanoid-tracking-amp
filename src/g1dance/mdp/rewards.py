"""Reward terms added on top of the upstream tracking rewards.

The upstream tracking rewards are exponential kernels over the mean squared error of the anchor (torso) pose,
the 14 tracked body poses and their velocities. They are deliberately forgiving (``std`` 0.3 m for positions),
which is fine for "stay close to the reference" but means a foot that is 2 cm above the ground loses about
0.05 % of the reward -- far less than the action-rate penalty of putting it down. Nothing in the upstream reward
depends on contact, so the policy learns to keep the feet just barely off the floor.

The terms here add that missing pressure:

* :func:`stance_loaded` / :func:`swing_clear` -- contact state must agree with the reference clip's stance labels.
* :func:`foot_height_reward` -- per-foot, tight (``std`` ~3 cm) kernel on the vertical error to the reference, so
  lifting or sinking a foot is no longer free.
* :func:`anchor_height_reward` -- tight kernel on the torso height, which catches the global "stand 3 cm taller
  than the reference" solution that floats both feet at once.

All of them are pure torch over the environment tensors; none needs Isaac Sim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """(N, F) bool: the foot carried more than ``threshold`` N during the last physics steps."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    return torch.linalg.norm(forces, dim=-1).max(dim=1).values > threshold


def reference_foot_contact(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """(N, F) bool stance labels of the reference clip at the current time step."""
    command = env.command_manager.get_term(command_name)
    return command.ref_foot_contact[command.time_steps]


def stance_loaded(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 10.0
) -> torch.Tensor:
    """Fraction of the feet the reference has in stance that carry load. In [0, 1]."""
    reference = reference_foot_contact(env, command_name)
    contact = foot_contacts(env, sensor_cfg, threshold)
    return (contact & reference).sum(dim=-1) / reference.sum(dim=-1).clamp(min=1)


def swing_clear(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 10.0
) -> torch.Tensor:
    """Fraction of the feet the reference has in swing that carry no load. In [0, 1]."""
    reference = reference_foot_contact(env, command_name)
    contact = foot_contacts(env, sensor_cfg, threshold)
    return (~contact & ~reference).sum(dim=-1) / (~reference).sum(dim=-1).clamp(min=1)


def _foot_indexes(command) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if name in FEET]


def foot_height_reward(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Per-foot exponential kernel on the vertical error to the reference, averaged over the feet.

    ``z`` is compared in the same frame for both sides: the command's relative body target keeps the global
    height of the reference, and the robot's bodies are in world coordinates (env origin height 0).
    """
    command = env.command_manager.get_term(command_name)
    indexes = _foot_indexes(command)
    error = command.robot_body_pos_w[:, indexes, 2] - command.body_pos_relative_w[:, indexes, 2]
    return torch.exp(-torch.square(error / std)).mean(dim=-1)


def anchor_height_reward(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Exponential kernel on the vertical error of the anchor (torso) body, metres."""
    command = env.command_manager.get_term(command_name)
    error = command.robot_anchor_pos_w[:, 2] - command.anchor_pos_w[:, 2]
    return torch.exp(-torch.square(error / std))


def feet_slide_world(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float = 10.0
) -> torch.Tensor:
    """Squared horizontal world-frame speed of the feet that touch the ground."""
    speed = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (foot_contacts(env, sensor_cfg, threshold) * speed.square()).sum(dim=-1)


def action_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Squared norm of the raw policy action (the residual when a residual action term is used)."""
    return env.action_manager.get_term(action_name).raw_actions.square().sum(dim=-1)
