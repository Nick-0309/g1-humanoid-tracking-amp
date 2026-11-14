"""AMP frame features. Pure torch; quaternions are (x, y, z, w) as in Isaac Lab 3.0.

One frame (identical layout for the expert clip and the simulated robot):

    base_ang_vel   (3)      pelvis angular velocity in the pelvis frame
    joint_pos      (J)
    joint_vel      (J)
    key_body_pos   (3 * K)  key body positions relative to the pelvis, in the heading frame

The heading frame is the pelvis frame with roll and pitch removed. It makes the features
independent of the world yaw while still carrying the body tilt.
"""

from __future__ import annotations

import numpy as np
import torch


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the inverse of ``quat`` (xyzw). Leading dims broadcast."""
    batch = torch.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    quat, vec = quat.expand(*batch, 4), vec.expand(*batch, 3)
    xyz = quat[..., :3]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2.0
    return vec - quat[..., 3:4] * t + torch.linalg.cross(xyz, t, dim=-1)


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Keep only the rotation about the world z axis. Input and output are xyzw."""
    x, y, z, w = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    zeros = torch.zeros_like(half)
    return torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)


def frame_dim(num_joints: int, num_key_bodies: int) -> int:
    return 3 + 2 * num_joints + 3 * num_key_bodies


def compute_amp_frame(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    key_body_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Shapes: (N,3), (N,4), (N,3), (N,J), (N,J), (N,K,3) -> (N, 3 + 2J + 3K)."""
    ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
    rel_w = key_body_pos_w - root_pos_w[:, None, :]
    rel_h = quat_apply_inverse(yaw_quat(root_quat_w)[:, None, :], rel_w)
    return torch.cat([ang_vel_b, joint_pos, joint_vel, rel_h.reshape(rel_h.shape[0], -1)], dim=-1)


def build_windows(frames: torch.Tensor, window: int) -> torch.Tensor:
    """(T, D) -> (T - window + 1, window * D); oldest frame first."""
    if frames.shape[0] < window:
        raise ValueError(f"need at least {window} frames, got {frames.shape[0]}")
    return frames.unfold(0, window, 1).permute(0, 2, 1).reshape(frames.shape[0] - window + 1, -1)


def expert_frames_from_npz(
    npz_path: str,
    body_names: list[str],
    pelvis_name: str,
    key_body_names: list[str],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Expert AMP frames from a motion ``.npz`` written by ``csv_to_npz.py``.

    ``body_names`` must be the simulated robot's body order, which is the order of the
    ``body_*`` arrays in the file.
    """
    data = np.load(npz_path)

    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(data[key], dtype=torch.float32, device=device)

    if data["body_pos_w"].shape[1] != len(body_names):
        raise ValueError(
            f"{npz_path} has {data['body_pos_w'].shape[1]} bodies but the robot has {len(body_names)}"
        )
    pelvis = body_names.index(pelvis_name)
    keys = [body_names.index(n) for n in key_body_names]
    body_pos, body_quat, body_ang = t("body_pos_w"), t("body_quat_w"), t("body_ang_vel_w")
    return compute_amp_frame(
        body_pos[:, pelvis],
        body_quat[:, pelvis],
        body_ang[:, pelvis],
        t("joint_pos"),
        t("joint_vel"),
        body_pos[:, keys],
    )
