"""Policy observations built from validated rollout state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from alexdoor_xas.action.frames import quat_to_rot_matrix
from alexdoor_xas.dataset.loader import OBS_PRESETS

if TYPE_CHECKING:
    from alexdoor_xas.adapters.base import StepContext

OBS_CLIP = 10.0
"""Bounds normalized observations when near-constant dimensions have tiny variance."""


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_obs_preset(preset: str) -> None:
    if preset not in OBS_PRESETS:
        raise ValueError(f"unknown obs preset {preset!r} (known: {sorted(OBS_PRESETS)})")


def build_rollout_obs(
    ctx: StepContext,
    preset: str,
    door_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Build a dataset-ordered observation from one validated state snapshot."""
    validate_obs_preset(preset)
    parts = [
        np.asarray(ctx.ee_pos_w, dtype=np.float64).reshape(3),
        np.asarray(ctx.ee_quat_w_xyzw, dtype=np.float64).reshape(4),
        np.array([ctx.hinge_angle_rad, ctx.hinge_velocity_rad_s], dtype=np.float64),
    ]
    if preset == "core_contact":
        parts.append(np.array([float(ctx.contact_sensed)], dtype=np.float64))
    if preset == "core_door_pose":
        if door_pose is None:
            raise ValueError("obs preset 'core_door_pose' requires a door-pose observation")
        parts.append(np.asarray(door_pose, dtype=np.float64).reshape(5))
    return np.concatenate(parts)


def read_door_pose_obs(env) -> np.ndarray:
    """Read the static relative door origin and yaw terms after reset."""
    frame_pos, frame_quat = env.door_frame_pose_w()
    position = np.asarray(_numpy(frame_pos), dtype=np.float64).reshape(-1)[:3]
    quaternion = np.asarray(_numpy(frame_quat), dtype=np.float64).reshape(-1)[:4]
    base = np.asarray(_numpy(env.robot_base_pos_w()), dtype=np.float64).reshape(-1)[:3]
    rotation = quat_to_rot_matrix(quaternion)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    relative_position = position - base
    return np.array(
        [*relative_position, np.sin(yaw), np.cos(yaw)],
        dtype=np.float64,
    )
