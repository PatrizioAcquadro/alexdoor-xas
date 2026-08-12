"""Canonical action-space tags and A4 serialization structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

A1_JOINT_DELTA = "A1_joint_delta"
A2_EE_DELTA = "A2_ee_delta"
A3_OBJ_REL_EE_DELTA = "A3_obj_rel_ee_delta"
A4_OBJ_CENTRIC_CHUNK = "A4_obj_centric_chunk"

ALL_ACTION_SPACES: tuple[str, ...] = (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)

# A2/A3 per-step action layout: (dx, dy, dz, drx, dry, drz) for one end-effector.
EE_DELTA_DIM = 6

# Excludes terminal ``done`` and avoids importing the scripted policy here.
A4_PHASE_VOCAB: tuple[str, ...] = (
    "approach",
    "align",
    "pre_contact",
    "contact",
    "push",
    "hold",
    "release",
)


@dataclass(frozen=True)
class ObjectCentricChunk:
    """A4 intent for one controller phase.

    The contact target uses the moving panel frame; motion is the intended hinge
    delta and duration is the completed phase length.
    """

    phase: str
    contact_target_panel: tuple[float, float, float]
    motion_hinge_delta_rad: float
    duration_ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "contact_target_panel": list(self.contact_target_panel),
            "motion_hinge_delta_rad": self.motion_hinge_delta_rad,
            "duration_ticks": self.duration_ticks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectCentricChunk:
        return cls(
            phase=str(data["phase"]),
            contact_target_panel=tuple(float(v) for v in data["contact_target_panel"]),
            motion_hinge_delta_rad=float(data["motion_hinge_delta_rad"]),
            duration_ticks=int(data["duration_ticks"]),
        )


__all__ = [
    "A1_JOINT_DELTA",
    "A2_EE_DELTA",
    "A3_OBJ_REL_EE_DELTA",
    "A4_OBJ_CENTRIC_CHUNK",
    "A4_PHASE_VOCAB",
    "ALL_ACTION_SPACES",
    "EE_DELTA_DIM",
    "ObjectCentricChunk",
]
