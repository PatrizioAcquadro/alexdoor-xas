"""Anti-windup clamping of solved IK joint-position targets (torch-only, no Isaac).

The dls differential-IK solver integrates Cartesian error into joint-position
targets with no knowledge of joint limits: when the commanded EE path leaves
the reachable workspace the solved targets march past the physical limits
while the drives saturate at them — integrator windup, seen in the Phase 3.2
dataset scale-up as 0.17-0.41 rad target overshoot on RIGHT_ELBOW_Y /
RIGHT_WRIST_X (see the adapter limits in the action-representation wiki page).
Clamping the *targets* to the
Isaac-reported position limits each tick removes the windup without touching
the frozen per-tick EE-delta clamps or any sanity threshold; because the IK
compute consumes live joint positions (not the previous target), the clamp
introduces no state that could itself wind up.

This module stays Isaac-free so the clamp math is unit-testable without Kit.
"""

from __future__ import annotations

import torch


def clamp_joint_targets(
    targets: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp joint-position targets to ``[lower, upper]`` per joint.

    Returns ``(clamped, excess)`` where ``excess >= 0`` is the pre-clamp
    overshoot magnitude per entry — exactly ``0`` wherever the raw target was
    already inside its limits (the clamp is a bitwise no-op there), so
    ``excess > 0`` doubles as the "this tick clamped this joint" event flag.
    All three tensors broadcast against each other (typically
    ``(num_envs, n_joints)``).
    """
    if torch.any(lower > upper):
        raise ValueError("joint limit lower bounds must not exceed upper bounds")
    clamped = torch.clamp(targets, min=lower, max=upper)
    excess = (targets - clamped).abs()
    return clamped, excess


__all__ = ["clamp_joint_targets"]
