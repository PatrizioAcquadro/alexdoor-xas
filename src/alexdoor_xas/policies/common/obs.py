"""Live observation readers and rollout-termination wrappers (policy-agnostic).

The env is duck-typed through the benchmark accessor surface
(``ee_pose_w`` / ``hinge_state`` / optional ``contact_sensed``), so the pure
test fakes and the Isaac env work unchanged. No Isaac imports.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from alexdoor_xas.action.frames import quat_to_rot_matrix
from alexdoor_xas.dataset import OBS_PRESETS

OBS_CLIP = 10.0
"""Normalized-observation clip: near-constant training dims have their std
floored at 1e-8, so a small absolute rollout deviation would otherwise map to
an enormous normalized value far outside anything the model saw."""

ROLLOUT_OBS_PRESETS = ("core", "core_contact", "core_door_pose")
"""Presets with a closed-loop env reader. ``alex_full`` training remains
possible offline, but its joint-state/force layout has no verified live
reader yet, so rollout refuses it rather than risk a silent mismatch."""


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def build_env_obs(env, preset: str) -> np.ndarray:
    """Read the frozen observation preset live from the env, dataset-ordered."""
    if preset not in OBS_PRESETS:
        raise ValueError(f"unknown obs preset {preset!r} (known: {sorted(OBS_PRESETS)})")
    if preset not in ROLLOUT_OBS_PRESETS:
        raise ValueError(
            f"obs preset {preset!r} has no closed-loop env reader "
            f"(supported: {list(ROLLOUT_OBS_PRESETS)})"
        )
    ee_pos, ee_quat = env.ee_pose_w()
    angle, velocity = env.hinge_state()
    parts = [
        np.asarray(
            ee_pos.detach().cpu().numpy() if isinstance(ee_pos, torch.Tensor) else ee_pos,
            dtype=np.float64,
        ).reshape(-1)[:3],
        np.asarray(
            ee_quat.detach().cpu().numpy() if isinstance(ee_quat, torch.Tensor) else ee_quat,
            dtype=np.float64,
        ).reshape(-1)[:4],
        np.array([_scalar(angle), _scalar(velocity)], dtype=np.float64),
    ]
    if preset == "core_contact":
        if not hasattr(env, "contact_sensed"):
            raise ValueError(
                "obs preset 'core_contact' needs env.contact_sensed(); "
                "this env does not expose force contact sensing"
            )
        parts.append(np.array([_scalar(env.contact_sensed())], dtype=np.float64))
    if preset == "core_door_pose":
        parts.append(_door_pose_terms(env))
    return np.concatenate(parts)


def _door_pose_terms(env) -> np.ndarray:
    """Door-pose obs block, dataset-ordered: rel origin (3) + sin/cos yaw (2).

    Mirrors the recorder exactly (data_engine.generate): door-frame origin
    relative to the robot base (world origin for base-less envs), yaw from the
    door-frame rotation about +Z.
    """
    frame_pos, frame_quat = env.door_frame_pose_w()
    pos = np.asarray(
        frame_pos.detach().cpu().numpy() if isinstance(frame_pos, torch.Tensor) else frame_pos,
        dtype=np.float64,
    ).reshape(-1)[:3]
    quat = np.asarray(
        frame_quat.detach().cpu().numpy() if isinstance(frame_quat, torch.Tensor) else frame_quat,
        dtype=np.float64,
    ).reshape(-1)[:4]
    base = np.zeros(3)
    if hasattr(env, "robot_base_pos_w"):
        base_pos = env.robot_base_pos_w()
        base = np.asarray(
            base_pos.detach().cpu().numpy() if isinstance(base_pos, torch.Tensor) else base_pos,
            dtype=np.float64,
        ).reshape(-1)[:3]
    rot = quat_to_rot_matrix(quat)
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    rel = pos - base
    return np.array([rel[0], rel[1], rel[2], np.sin(yaw), np.cos(yaw)], dtype=np.float64)


def stop_on_hinge_angle(source: Callable, threshold_rad: float) -> Callable:
    """End the rollout once the door is open past ``threshold_rad``.

    The demos end when the scripted FSM completes, so a learned policy has no
    in-distribution behavior after task completion — left running, the
    extrapolating arm can knock the door shut again. This wrapper terminates
    at the first source query (chunk boundary) where the hinge angle has
    passed the threshold, bounding post-task extrapolation the same way the
    scripted episode termination does.
    """

    def wrapped(ctx):
        if ctx.hinge_angle_rad >= threshold_rad:
            return None
        return source(ctx)

    return wrapped


__all__ = [
    "OBS_CLIP",
    "ROLLOUT_OBS_PRESETS",
    "build_env_obs",
    "stop_on_hinge_angle",
]
