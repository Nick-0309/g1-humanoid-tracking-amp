"""Observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g1dance.amp.features import compute_amp_frame
from g1dance.mdp.rewards import foot_contacts

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def amp_frame(env: ManagerBasedRLEnv, pelvis_cfg: SceneEntityCfg, key_cfg: SceneEntityCfg) -> torch.Tensor:
    """One AMP frame per env, see :func:`g1dance.amp.features.compute_amp_frame`."""
    data = env.scene[pelvis_cfg.name].data
    pelvis = pelvis_cfg.body_ids[0]
    body_pos = data.body_pos_w.torch
    return compute_amp_frame(
        body_pos[:, pelvis],
        data.body_quat_w.torch[:, pelvis],
        data.body_ang_vel_w.torch[:, pelvis],
        data.joint_pos.torch,
        data.joint_vel.torch,
        body_pos[:, key_cfg.body_ids],
    )


def motion_future_ref(env: ManagerBasedRLEnv, command_name: str, offsets: tuple[int, ...]) -> torch.Tensor:
    """Reference joint positions and velocities ``offsets`` control steps ahead (clamped at the clip end)."""
    command = env.command_manager.get_term(command_name)
    last = command.motion.time_step_total - 1
    parts = []
    for offset in offsets:
        idx = (command.time_steps + offset).clamp(max=last)
        parts += [command.motion.joint_pos[idx], command.motion.joint_vel[idx]]
    return torch.cat(parts, dim=-1)


def contact_phase(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 10.0
) -> torch.Tensor:
    """Per foot: whether the reference clip has it in stance, then whether it carries load now. (N, 2F).

    The reference half is available on the robot (it is part of the commanded motion); the measured half needs
    foot force sensing, so a deployment without those sensors must drop this term (``--contact_obs`` keeps the
    observation and action layout of the run in ``params/features.json``).
    """
    command = env.command_manager.get_term(command_name)
    reference = command.ref_foot_contact[command.time_steps].float()
    measured = foot_contacts(env, sensor_cfg, threshold).float()
    return torch.cat([reference, measured], dim=-1)
