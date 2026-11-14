"""Motion command with the behaviours this project needs on top of the upstream one."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch
from isaaclab.utils import configclass

from g1dance.mdp.contacts import foot_floor, reference_foot_contacts
from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand, MotionCommandCfg

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class G1MotionCommand(MotionCommand):
    cfg: G1MotionCommandCfg

    def _resample_command(self, env_ids):
        # Isaac Lab 3.0 passes ``slice(None)`` for "all envs" on the first reset.
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        super()._resample_command(env_ids)

    def _adaptive_sampling(self, env_ids):
        if self.cfg.start_at_zero:
            self.time_steps[env_ids] = 0
            return
        super()._adaptive_sampling(env_ids)

    @property
    def next_joint_pos(self) -> torch.Tensor:
        """Reference joint positions one control step ahead, in the articulation's joint order."""
        last = self.motion.time_step_total - 1
        return self.motion.joint_pos[(self.time_steps + 1).clamp(max=last)]

    def _load_reference_feet(self) -> None:
        """Foot positions of the clip, its stance labels and the height of each foot when planted."""
        data = np.load(self.cfg.motion_file)
        feet = self.robot.find_bodies(list(FEET), preserve_order=True)[0]
        pos = torch.as_tensor(data["body_pos_w"][:, feet], dtype=torch.float32, device=self.device)
        self._ref_foot_pos = pos
        self._ref_foot_floor = foot_floor(pos)
        self._ref_foot_contact = reference_foot_contacts(pos, float(data["fps"].reshape(-1)[0]))

    @property
    def ref_foot_contact(self) -> torch.Tensor:
        """(frames, 2) bool stance labels of the left and right foot, derived from the clip."""
        if not hasattr(self, "_ref_foot_contact"):
            self._load_reference_feet()
        return self._ref_foot_contact

    @property
    def ref_foot_floor(self) -> torch.Tensor:
        """(2,) height of each ankle link origin when that foot is planted, from the clip itself."""
        if not hasattr(self, "_ref_foot_floor"):
            self._load_reference_feet()
        return self._ref_foot_floor

    @property
    def ref_foot_height(self) -> torch.Tensor:
        """(frames, 2) reference foot height above its own planted height, metres."""
        if not hasattr(self, "_ref_foot_pos"):
            self._load_reference_feet()
        return self._ref_foot_pos[..., 2] - self._ref_foot_floor


@configclass
class G1MotionCommandCfg(MotionCommandCfg):
    class_type: type = G1MotionCommand

    start_at_zero: bool = False
    """Always start (and restart) the clip at frame 0. For play and evaluation."""


def to_g1_command_cfg(cfg: MotionCommandCfg) -> G1MotionCommandCfg:
    """Copy an upstream ``MotionCommandCfg`` into the subclass that uses :class:`G1MotionCommand`."""
    fields = {
        f.name: copy.deepcopy(getattr(cfg, f.name)) for f in dataclasses.fields(cfg) if f.init and f.name != "class_type"
    }
    return G1MotionCommandCfg(**fields)
