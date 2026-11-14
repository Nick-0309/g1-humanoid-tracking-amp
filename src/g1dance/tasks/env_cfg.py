"""Environment configs: the upstream dance-102 tracking task, plus an ``amp`` observation group."""

from __future__ import annotations

import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from g1dance.mdp.actions import ResidualJointPositionActionCfg
from g1dance.mdp.commands import FEET, to_g1_command_cfg
from g1dance.mdp.obs import amp_frame, contact_phase, motion_future_ref
from g1dance.mdp.rewards import (
    action_l2,
    anchor_height_reward,
    feet_slide_world,
    foot_height_reward,
    stance_loaded,
    swing_clear,
)
from g1dance.tasks.features import TaskFeatures
from unitree_rl_lab.tasks.mimic.mdp import bad_motion_body_pos_z_only
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import (
    ObservationsCfg as UpstreamObservationsCfg,
)
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import RobotEnvCfg as UpstreamEnvCfg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MOTION = os.path.join(PROJECT_ROOT, "motions", "dance_102.npz")

SLEW_RATES = {".*(shoulder|elbow|wrist).*": 10.0, ".*(hip|knee|ankle|waist).*": 20.0}
"""Rad/s limits of the residual, from the tutorial (arms 10, legs / waist / ankles 20)."""

LANDING_REWARDS = {
    "stance_loaded": 0.4,
    "swing_clear": 0.6,
    "foot_height": 0.2,
    "anchor_height": 0.1,
    "feet_slide_world": -0.05,
    "action_l2": -0.002,
}
"""Weights of the landing bundle (``--foot_rewards``).

Measured on the tracking baseline (``track_v1``, ``scripts/eval.py`` + ``scripts/calibrate_rewards.py``): the
policy is loaded 99.9 % of the frames in which the reference is in stance, but also 94 % of the frames in which it
is in swing -- it never unloads a foot, so there is nothing to land from and it steps on its own other foot. That
is why ``swing_clear`` carries the most weight. ``stance_loaded`` and ``anchor_height`` are guards (they are
already near 1.0), and ``foot_height`` is a loose kernel that stops the policy from clearing the ground by lifting
the feet much higher than the reference.

Measure the terms again with ``scripts/calibrate_rewards.py --foot_rewards`` on any checkpoint before changing the
weights: the README section "纯 tracking 的着地问题" gives the numbers used here."""

LANDING_STDS = {"foot_height": 0.04, "anchor_height": 0.05}
"""Metres: half-width of the vertical error kernel of the two height terms (upstream uses 0.30 for positions)."""

PELVIS = "pelvis"
DEFAULT_KEY_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
)


@configclass
class AmpObsCfg(ObsGroup):
    """A single AMP frame per env; windows are stacked by the runner."""

    frame = ObsTerm(
        func=amp_frame,
        params={
            "pelvis_cfg": SceneEntityCfg("robot", body_names=[PELVIS], preserve_order=True),
            "key_cfg": SceneEntityCfg("robot", body_names=list(DEFAULT_KEY_BODIES), preserve_order=True),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AmpObservationsCfg(UpstreamObservationsCfg):
    amp: AmpObsCfg = AmpObsCfg()


class G1DanceTrackingEnvCfg(UpstreamEnvCfg):
    """Upstream BeyondMimic-style tracking task with this project's motion command."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.commands.motion = to_g1_command_cfg(self.commands.motion)
        self.commands.motion.motion_file = DEFAULT_MOTION

    def play_mode(self):
        """Deterministic evaluation: start at frame 0, no noise, no pushes, no randomisation."""
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        motion = self.commands.motion
        motion.start_at_zero = True
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)


class G1DanceAmpEnvCfg(G1DanceTrackingEnvCfg):
    """Tracking task plus the ``amp`` observation group and terminal observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = AmpObservationsCfg()
        self.compute_final_obs = True


def configure(
    env_cfg,
    *,
    motion_file: str | None = None,
    episode_length_s: float | None = None,
    key_bodies: list[str] | None = None,
    features: TaskFeatures | None = None,
) -> None:
    """Apply command line overrides and the optional task features."""
    if motion_file is not None:
        env_cfg.commands.motion.motion_file = os.path.abspath(motion_file)
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
    if key_bodies is not None:
        env_cfg.observations.amp.frame.params["key_cfg"].body_names = list(key_bodies)
    if features is not None:
        apply_features(env_cfg, features)


def apply_features(env_cfg, features: TaskFeatures) -> None:
    if features.future_ref:
        for group in (env_cfg.observations.policy, env_cfg.observations.critic):
            group.future_ref = ObsTerm(
                func=motion_future_ref, params={"command_name": "motion", "offsets": tuple(features.future_ref)}
            )

    if features.residual:
        base = env_cfg.actions.JointPositionAction
        env_cfg.actions.JointPositionAction = ResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=features.residual_scale if features.residual_scale is not None else base.scale,
            slew_rates=SLEW_RATES if features.slew_limit else None,
        )

    if features.foot_rewards:
        feet_sensor = SceneEntityCfg("contact_forces", body_names=list(FEET), preserve_order=True)
        feet_body = SceneEntityCfg("robot", body_names=list(FEET), preserve_order=True)
        env_cfg.rewards.stance_loaded = RewTerm(
            func=stance_loaded,
            weight=LANDING_REWARDS["stance_loaded"],
            params={"command_name": "motion", "sensor_cfg": feet_sensor},
        )
        env_cfg.rewards.swing_clear = RewTerm(
            func=swing_clear,
            weight=LANDING_REWARDS["swing_clear"],
            params={"command_name": "motion", "sensor_cfg": feet_sensor},
        )
        env_cfg.rewards.foot_height = RewTerm(
            func=foot_height_reward,
            weight=LANDING_REWARDS["foot_height"],
            params={"command_name": "motion", "std": LANDING_STDS["foot_height"]},
        )
        env_cfg.rewards.anchor_height = RewTerm(
            func=anchor_height_reward,
            weight=LANDING_REWARDS["anchor_height"],
            params={"command_name": "motion", "std": LANDING_STDS["anchor_height"]},
        )
        env_cfg.rewards.feet_slide_world = RewTerm(
            func=feet_slide_world,
            weight=LANDING_REWARDS["feet_slide_world"],
            params={"sensor_cfg": feet_sensor, "asset_cfg": feet_body},
        )
        env_cfg.rewards.action_l2 = RewTerm(
            func=action_l2, weight=LANDING_REWARDS["action_l2"], params={"action_name": "JointPositionAction"}
        )

    if features.contact_obs:
        feet_sensor = SceneEntityCfg("contact_forces", body_names=list(FEET), preserve_order=True)
        for group in (env_cfg.observations.policy, env_cfg.observations.critic):
            group.contact_phase = ObsTerm(
                func=contact_phase, params={"command_name": "motion", "sensor_cfg": feet_sensor}
            )

    if features.foot_termination is not None:
        env_cfg.terminations.foot_height = DoneTerm(
            func=bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": features.foot_termination,
                "body_names": list(FEET),
            },
        )
