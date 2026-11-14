"""The pure-array part of the motion cleanup (no simulator, no mujoco)."""

import numpy as np

from g1dance.data.motion_cleanup import (
    contact_mask,
    foot_distance,
    foot_gaps,
    ground_offset,
    runs,
    sole_heights,
    swing_lift,
)


def make_foot(z_values, x_values=None, y=0.0):
    """4 sole corners per frame for one foot; ``z_values`` is the lowest sole height per frame."""
    T = len(z_values)
    if x_values is None:
        x_values = np.zeros(T)
    corners = np.array([[-0.05, 0.025], [-0.05, -0.025], [0.12, 0.03], [0.12, -0.03]])
    out = np.zeros((T, 4, 3))
    for t in range(T):
        out[t, :, 0] = x_values[t] + corners[:, 0]
        out[t, :, 1] = y + corners[:, 1]
        out[t, :, 2] = z_values[t]
    return out


def test_sole_heights_is_the_lowest_corner():
    points = np.zeros((3, 2, 4, 3))
    points[:, :, :, 2] = 0.1
    points[1, 0, 2, 2] = -0.02
    heights = sole_heights(points)
    assert heights.shape == (3, 2)
    assert np.isclose(heights[1, 0], -0.02) and np.isclose(heights[1, 1], 0.1)


def test_contact_mask_threshold():
    heights = np.array([[0.0, 0.05], [0.02, 0.005]])
    assert contact_mask(heights, threshold=0.012).tolist() == [[True, False], [False, True]]


def test_ground_offset_lifts_by_the_penetration_and_holds_flight():
    # 3 cm penetration, then on the floor, then a jump where both feet are 12 cm up
    heights = np.array([[-0.03, -0.02], [0.0, 0.001], [0.12, 0.10], [0.0, 0.0]])
    offset = ground_offset(heights, support_height=0.02, sigma=0.0)
    assert np.isclose(offset[0], 0.03)  # the deepest sole goes to 0
    assert np.isclose(offset[1], 0.0)
    assert np.isclose(offset[3], 0.0)
    assert abs(offset[2]) < 1e-9  # in flight the clip keeps its own root height, it is not glued to the floor


def test_runs_and_gaps():
    mask = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [1, 0]], dtype=bool)
    assert runs(mask, min_frames=1)[0] == [(1, 3), (4, 5)]
    assert runs(mask, min_frames=1)[1] == [(0, 2)]
    contact = np.array([[1, 1], [1, 0], [1, 1], [1, 1]], dtype=bool)
    assert foot_gaps(contact, min_frames=1)[1] == [(1, 2)]
    assert foot_gaps(contact)[1] == []  # a one frame gap is a dropout, not a swing


def test_swing_lift_is_zero_at_the_swing_edges_and_reaches_the_margin():
    heights = np.zeros((24, 1))
    swings = [[(2, 22)]]  # 20 frames, so the middle is exactly pi / 2
    lift = swing_lift(heights, swings, margin=0.03, max_lift=0.08)
    assert np.all(lift[:2] == 0) and np.all(lift[22:] == 0)
    assert np.isclose(lift[2, 0], 0.0) and np.isclose(lift[21, 0], 0.0)
    assert lift[12, 0] > 0.025  # middle of the swing, a sample or two from the exact peak
    assert np.all(np.diff(lift[2:11, 0]) > 0)
    assert np.isclose(lift.max(), 0.03, atol=5e-3)


def test_swing_lift_does_nothing_when_the_foot_already_clears():
    heights = np.full((10, 1), 0.05)
    lift = swing_lift(heights, [[(2, 8)]], margin=0.03)
    assert np.all(lift == 0)


def test_foot_distance_measures_the_gap_between_two_sole_rectangles():
    # both soles are ~5 cm wide, so centres 9 cm apart leave ~4 cm between the patches
    near = np.stack([make_foot(np.zeros(1)), make_foot(np.zeros(1), y=0.09)], axis=1)
    far = np.stack([make_foot(np.zeros(1)), make_foot(np.zeros(1), y=0.30)], axis=1)
    d_near = foot_distance(near)
    d_far = foot_distance(far)
    assert abs(d_near[0] - 0.04) < 0.012  # lattice spacing is ~1.4 cm
    assert abs(d_far[0] - 0.25) < 0.012
    assert d_near[0] < d_far[0]


def test_foot_distance_is_zero_when_the_patches_overlap():
    points = np.stack([make_foot(np.zeros(1)), make_foot(np.zeros(1), y=0.01)], axis=1)
    assert foot_distance(points)[0] < 0.002
