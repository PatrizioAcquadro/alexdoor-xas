"""Adapter decisions, warnings, logs, and per-tick state."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame


class AdapterStatus(enum.StrEnum):
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AdapterWarning:
    """Stable warning family plus JSON-serializable physical evidence."""

    id: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "message": self.message, "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class AdapterDecision:
    """Recorded validation outcome; rejection applies zero motion."""

    status: AdapterStatus
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    warning_records: tuple[AdapterWarning, ...] = ()
    requested: np.ndarray | None = None
    applied: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "reason": self.reason,
            "checks": dict(self.checks),
            "warning_records": [warning.to_dict() for warning in self.warning_records],
            "requested": None if self.requested is None else np.asarray(self.requested).tolist(),
            "applied": None if self.applied is None else np.asarray(self.applied).tolist(),
        }


@dataclass
class AdapterLog:
    """Ordered adapter decisions for one rollout, serializable to JSON."""

    decisions: list[AdapterDecision] = field(default_factory=list)

    def record(self, decision: AdapterDecision) -> AdapterDecision:
        self.decisions.append(decision)
        return decision

    def count(self, status: AdapterStatus) -> int:
        return sum(1 for decision in self.decisions if decision.status is status)

    @property
    def n_accepted(self) -> int:
        return self.count(AdapterStatus.ACCEPTED)

    @property
    def n_corrected(self) -> int:
        return self.count(AdapterStatus.CORRECTED)

    @property
    def n_rejected(self) -> int:
        return self.count(AdapterStatus.REJECTED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_accepted": self.n_accepted,
            "n_corrected": self.n_corrected,
            "n_rejected": self.n_rejected,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


@dataclass(frozen=True)
class StepContext:
    """Validated state used to adapt one command."""

    door_frame: ObjectFrame | None
    hinge_angle_rad: float
    hinge_velocity_rad_s: float
    ee_pos_w: np.ndarray  # (3,)
    ee_quat_w_xyzw: np.ndarray  # (4,); validated but not commanded
    contact_sensed: bool
    contact_force_n: float
    joint_state: dict[str, np.ndarray]
    joint_limits: dict[str, np.ndarray]
    joint_names: tuple[str, ...]
    tick_index: int | None = None
    rollout_phase: str = "unknown"
