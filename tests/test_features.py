import math

import numpy as np
import pytest
import torch

from g1dance.amp.features import (
    build_windows,
    compute_amp_frame,
    expert_frames_from_npz,
    frame_dim,
    quat_apply_inverse,
    yaw_quat,
)


def rand_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


def quat_to_matrix_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def yaw_rot(angle: float, n: int) -> torch.Tensor:
    half = angle / 2
    return torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)]).repeat(n, 1)


def test_quat_apply_inverse_matches_matrix():
    gen = torch.Generator().manual_seed(0)
    q, v = rand_quat(16, gen), torch.randn(16, 3, generator=gen)
    expected = torch.einsum("nji,nj->ni", quat_to_matrix_xyzw(q), v)  # R^T v
    assert torch.allclose(quat_apply_inverse(q, v), expected, atol=1e-5)


def test_quat_apply_inverse_matches_isaaclab():
    math_utils = pytest.importorskip("isaaclab.utils.math")
    gen = torch.Generator().manual_seed(1)
    q, v = rand_quat(16, gen), torch.randn(16, 3, generator=gen)
    assert torch.allclose(quat_apply_inverse(q, v), math_utils.quat_apply_inverse(q, v), atol=1e-5)


def test_quat_apply_inverse_broadcasts_over_bodies():
    gen = torch.Generator().manual_seed(2)
    q, v = rand_quat(5, gen), torch.randn(5, 7, 3, generator=gen)
    out = quat_apply_inverse(q[:, None, :], v)
    assert out.shape == (5, 7, 3)
    assert torch.allclose(out[:, 3], quat_apply_inverse(q, v[:, 3]), atol=1e-6)


def test_yaw_quat_keeps_only_heading():
    gen = torch.Generator().manual_seed(3)
    q = rand_quat(32, gen)
    yq = yaw_quat(q)
    assert torch.allclose(yq.norm(dim=-1), torch.ones(32), atol=1e-6)
    assert torch.all(yq[:, :2] == 0)
    pure = yaw_rot(0.7, 1)
    assert torch.allclose(yaw_quat(pure), pure, atol=1e-6)


def make_state(n: int, j: int, k: int, seed: int = 4):
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, 3, generator=gen),
        rand_quat(n, gen),
        torch.randn(n, 3, generator=gen),
        torch.randn(n, j, generator=gen),
        torch.randn(n, j, generator=gen),
        torch.randn(n, k, 3, generator=gen),
    )


def test_frame_dim():
    pos, quat, ang, jp, jv, kp = make_state(4, 29, 6)
    frame = compute_amp_frame(pos, quat, ang, jp, jv, kp)
    assert frame.shape == (4, frame_dim(29, 6)) == (4, 79)


def test_frame_is_invariant_to_world_yaw():
    pos, quat, ang, jp, jv, kp = make_state(8, 5, 4)
    rot = yaw_rot(1.1, 8)
    inv = _inv(rot)  # applying inv^-1 == rot rotates the whole world by 1.1 rad about z
    rotated = compute_amp_frame(
        quat_apply_inverse(inv, pos),
        quat_mul_xyzw(rot, quat),
        quat_apply_inverse(inv, ang),
        jp,
        jv,
        quat_apply_inverse(inv[:, None, :], kp),
    )
    assert torch.allclose(rotated, compute_amp_frame(pos, quat, ang, jp, jv, kp), atol=1e-5)


def _inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion (xyzw)."""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def test_key_body_features_carry_tilt_but_angular_velocity_does_not():
    pos, quat, ang, jp, jv, kp = make_state(4, 5, 4)
    roll = torch.tensor([math.sin(0.2), 0.0, 0.0, math.cos(0.2)]).repeat(4, 1)
    inv = _inv(roll)
    base = compute_amp_frame(pos, quat, ang, jp, jv, kp)
    tilted = compute_amp_frame(
        quat_apply_inverse(inv, pos),
        quat_mul_xyzw(roll, quat),
        quat_apply_inverse(inv, ang),
        jp,
        jv,
        quat_apply_inverse(inv[:, None, :], kp),
    )
    assert torch.allclose(base[:, :3], tilted[:, :3], atol=1e-5)  # pelvis-frame angular velocity
    assert not torch.allclose(base[:, -12:], tilted[:, -12:], atol=1e-3)  # heading-frame key bodies


def test_build_windows_order():
    frames = torch.arange(12.0).reshape(6, 2)
    w = build_windows(frames, 3)
    assert w.shape == (4, 6)
    assert torch.equal(w[0], torch.tensor([0.0, 1, 2, 3, 4, 5]))
    assert torch.equal(w[3], torch.tensor([6.0, 7, 8, 9, 10, 11]))
    with pytest.raises(ValueError):
        build_windows(frames[:2], 3)


def test_expert_frames_from_npz(tmp_path):
    t, b, j = 10, 5, 4
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(t, b, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    path = tmp_path / "m.npz"
    np.savez(
        path,
        fps=[50],
        joint_pos=rng.normal(size=(t, j)),
        joint_vel=rng.normal(size=(t, j)),
        body_pos_w=rng.normal(size=(t, b, 3)),
        body_quat_w=quat,
        body_lin_vel_w=rng.normal(size=(t, b, 3)),
        body_ang_vel_w=rng.normal(size=(t, b, 3)),
    )
    names = ["pelvis", "a", "b", "c", "d"]
    frames = expert_frames_from_npz(str(path), names, "pelvis", ["b", "d"])
    assert frames.shape == (t, frame_dim(j, 2))
    with pytest.raises(ValueError):
        expert_frames_from_npz(str(path), names[:3], "pelvis", ["b"])
