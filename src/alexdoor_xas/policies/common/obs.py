"""Live observation readers and rollout-termination wrappers (policy-agnostic).

The env is duck-typed through the frozen Phase 2 accessor surface
(``proxy_pose_w`` / ``hinge_state`` / optional ``contact_sensed``), so the pure
test fakes and both Isaac envs work unchanged. No Isaac imports.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from alexdoor_xas.dataset import OBS_PRESETS

OBS_CLIP = 10.0
"""Normalized-observation clip: near-constant training dims have their std
floored at 1e-8, so a small absolute rollout deviation would otherwise map to
an enormous normalized value far outside anything the model saw."""

ROLLOUT_OBS_PRESETS = ("core", "core_contact")
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
    ee_pos, ee_quat = env.proxy_pose_w()
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
    return np.concatenate(parts)


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
