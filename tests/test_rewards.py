"""The landing reward terms, on stub objects (no Isaac Sim needed)."""

from types import SimpleNamespace

import torch

from g1dance.mdp.rewards import (
    anchor_height_reward,
    foot_contacts,
    foot_height_reward,
    stance_loaded,
    swing_clear,
)

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
BODY_NAMES = ["pelvis", "waist_roll_link", *FEET]
BODY_IDS = [2, 3]  # the feet inside the contact sensor, which watches every body
NO_CONTACT = torch.zeros(1, 1, len(BODY_NAMES), 3)


class FakeScene:
    def __init__(self, items):
        self.items = items
        self.sensors = items

    def __getitem__(self, key):
        return self.items[key]


class FakeCommand:
    """The parts of ``G1MotionCommand`` the reward terms read."""

    def __init__(self, num_envs: int, reference: torch.Tensor, foot_z=None, ref_z=None, anchor=None):
        self.cfg = SimpleNamespace(body_names=BODY_NAMES)
        self.time_steps = torch.arange(num_envs)  # one reference frame per env
        self.ref_foot_contact = reference.reshape(-1, len(FEET))
        self.robot_body_pos_w = torch.zeros(num_envs, len(BODY_NAMES), 3)
        self.body_pos_relative_w = torch.zeros(num_envs, len(BODY_NAMES), 3)
        if foot_z is not None:
            self.robot_body_pos_w[:, BODY_IDS, 2] = foot_z
        if ref_z is not None:
            self.body_pos_relative_w[:, BODY_IDS, 2] = ref_z
        robot_height, reference_height = anchor if anchor is not None else (0.8, 0.8)
        self.robot_anchor_pos_w = torch.tensor([[0.0, 0.0, robot_height]] * num_envs)
        self.anchor_pos_w = torch.tensor([[0.0, 0.0, reference_height]] * num_envs)


def make_env(forces: torch.Tensor, command: FakeCommand):
    sensor = SimpleNamespace(data=SimpleNamespace(net_normal_forces_w_history=SimpleNamespace(torch=forces)))
    robot = SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=SimpleNamespace(torch=torch.zeros(1, 1, 3))))
    return SimpleNamespace(
        scene=FakeScene({"contact_forces": sensor, "robot": robot}),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(raw_actions=torch.zeros(1, 29))),
    )


def forces_of(loaded: torch.Tensor) -> torch.Tensor:
    """(N, F) bool -> force history (N, 1, bodies, 3) with 50 N on the loaded feet."""
    forces = torch.zeros(loaded.shape[0], 1, len(BODY_NAMES), 3)
    forces[:, 0, BODY_IDS, 2] = loaded.float() * 50.0
    return forces


SENSOR = SimpleNamespace(name="contact_forces", body_ids=BODY_IDS)


def test_foot_contacts_reads_the_sensor_history():
    loaded = torch.tensor([[True, False], [False, True]])
    env = make_env(forces_of(loaded), FakeCommand(2, torch.zeros(1, 2, dtype=torch.bool)))
    assert torch.equal(foot_contacts(env, SENSOR, threshold=10.0), loaded)


def test_stance_loaded_fraction():
    reference = torch.tensor([[True, True], [True, False], [False, False]])
    contact = torch.tensor([[True, True], [True, True], [True, False]])
    env = make_env(forces_of(contact), FakeCommand(3, reference))
    # env 0: 2/2 stance feet loaded; env 1: 1/1 (the swing foot is not counted); env 2: no stance foot -> 0
    assert torch.allclose(stance_loaded(env, "motion", SENSOR), torch.tensor([1.0, 1.0, 0.0]))


def test_swing_clear_fraction():
    reference = torch.tensor([[True, False], [False, True], [True, True]])
    contact = torch.tensor([[True, True], [False, True], [False, False]])
    env = make_env(forces_of(contact), FakeCommand(3, reference))
    # env 0: the swinging right foot touches -> 0; env 1: the swinging left foot is free -> 1; env 2: no swing -> 0
    assert torch.allclose(swing_clear(env, "motion", SENSOR), torch.tensor([0.0, 1.0, 0.0]))


def test_foot_height_reward_is_a_kernel_on_the_two_feet():
    foot_z = torch.tensor([[0.03, 0.06]])  # left matches the reference, right is 3 cm high
    ref_z = torch.tensor([[0.03, 0.03]])
    env = make_env(NO_CONTACT, FakeCommand(1, torch.zeros(1, 2, dtype=torch.bool), foot_z, ref_z))
    expected = (1.0 + float(torch.exp(torch.tensor(-1.0)))) / 2  # std = 3 cm
    assert torch.allclose(foot_height_reward(env, "motion", std=0.03), torch.tensor([expected]))


def test_anchor_height_reward_sign():
    env = make_env(NO_CONTACT, FakeCommand(1, torch.zeros(1, 2, dtype=torch.bool), anchor=(0.75, 0.80)))
    # robot torso 5 cm below the reference, std 5 cm -> exp(-1)
    assert torch.allclose(anchor_height_reward(env, "motion", std=0.05), torch.exp(torch.tensor(-1.0)).reshape(1))
