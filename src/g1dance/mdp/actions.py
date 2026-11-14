"""Residual joint position action: the target is the reference pose one step ahead plus a scaled action."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass, string as string_utils

from g1dance.mdp.contacts import slew_limit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualJointPositionAction(JointPositionAction):
    """``q_target = q_ref(t + 1 step) + scale * a``, optionally with a per-joint rate limit on the residual.

    The upstream startup event ``randomize_joint_default_pos`` writes randomised default positions into
    ``_offset`` to model joint calibration errors. Here that error is the difference to the nominal default pose
    and is added to the target, so the randomisation keeps its meaning.
    """

    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._nominal_offset = self._offset.clone()
        self._previous_residual = torch.zeros_like(self._raw_actions)
        self._max_delta: torch.Tensor | None = None
        if cfg.slew_rates is not None:
            index, _, rates = string_utils.resolve_matching_names_values(cfg.slew_rates, self._joint_names)
            limit = torch.full((self.action_dim,), float("inf"), device=self.device)
            limit[index] = torch.tensor(rates, dtype=torch.float32, device=self.device)
            self._max_delta = limit * env.step_dt

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        residual = self._raw_actions * self._scale
        if self._max_delta is not None:
            residual = slew_limit(residual, self._previous_residual, self._max_delta)
            self._previous_residual = residual
        reference = self._env.command_manager.get_term(self.cfg.command_name).next_joint_pos
        calibration_error = self._offset - self._nominal_offset
        self._processed_actions = reference[:, self._joint_ids] + residual + calibration_error

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._previous_residual[env_ids if env_ids is not None else slice(None)] = 0.0


@configclass
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ResidualJointPositionAction

    command_name: str = "motion"
    slew_rates: dict[str, float] | None = None
    """Maximum change of the residual in rad/s, by joint name regex. ``None`` = no limit."""
