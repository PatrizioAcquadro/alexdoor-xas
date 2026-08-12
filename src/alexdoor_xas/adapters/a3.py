"""Convert A3 door-frame deltas to A2 world-frame commands.

The hinge-anchored door frame is static per episode and must be a proper rotation.
"""

from __future__ import annotations

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame, frame_delta_to_world
from alexdoor_xas.action.spaces import EE_DELTA_DIM

from .a2 import A2Adapter
from .base import AdapterDecision, AdapterStatus, StepContext

_ROT_ORTHONORMAL_TOL = 1e-5


def validate_object_frame(frame: ObjectFrame | None) -> str:
    """Return the failure reason if the object frame is missing/corrupt, else ''."""
    if frame is None:
        return "object frame is unavailable (no door frame in the step context)"
    origin = np.asarray(frame.origin, dtype=np.float64).reshape(-1)
    rot = np.asarray(frame.rot, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return f"object frame origin is invalid: {origin}"
    if rot.shape != (3, 3) or not np.isfinite(rot).all():
        return "object frame rotation is non-finite or misshaped"
    if not np.allclose(rot @ rot.T, np.eye(3), atol=_ROT_ORTHONORMAL_TOL):
        return "object frame rotation is not orthonormal"
    if not np.isclose(np.linalg.det(rot), 1.0, atol=_ROT_ORTHONORMAL_TOL):
        return "object frame rotation is not proper (determinant must be +1)"
    return ""


class A3Adapter:
    """Convert door-frame EE deltas to world frame and execute through A2."""

    def __init__(self, a2: A2Adapter):
        self.a2 = a2

    @property
    def log(self):
        return self.a2.log

    def process(self, delta_door_frame, ctx: StepContext) -> tuple[np.ndarray, AdapterDecision]:
        """Return the applied world-frame delta and its decision."""
        requested = np.asarray(delta_door_frame, dtype=np.float64).reshape(-1)
        frame_reason = validate_object_frame(ctx.door_frame)
        hinge_ok = bool(np.isfinite(ctx.hinge_angle_rad))
        if frame_reason or not hinge_ok:
            decision = self.log.record(
                AdapterDecision(
                    status=AdapterStatus.REJECTED,
                    reason=frame_reason or "hinge angle is non-finite",
                    checks={"object_frame_trusted": not frame_reason, "hinge_readable": hinge_ok},
                    requested=requested,
                    applied=np.zeros(EE_DELTA_DIM),
                )
            )
            return np.zeros(EE_DELTA_DIM), decision
        if requested.shape != (EE_DELTA_DIM,) or not np.isfinite(requested).all():
            # A2 owns the canonical delta validation.
            return self.a2.process(requested, ctx)

        delta_world = frame_delta_to_world(requested, ctx.door_frame)
        applied, a2_decision = self.a2.process(delta_world, ctx)
        # Preserve the A3 request and static-frame assumption in the decision.
        checks = dict(a2_decision.checks)
        checks["object_frame_trusted"] = True
        checks["door_frame_static_stage_read"] = True
        annotated = AdapterDecision(
            status=a2_decision.status,
            reason=a2_decision.reason,
            checks=checks,
            warnings=a2_decision.warnings,
            warning_records=a2_decision.warning_records,
            requested=requested,
            applied=applied,
        )
        self.log.decisions[-1] = annotated
        return applied, annotated
