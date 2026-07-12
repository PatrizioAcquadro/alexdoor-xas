"""Adapter-v1 core types: decision statuses, per-command decisions, and logs.

Every adapter call resolves to one :class:`AdapterDecision` — the command was
either **accepted** as-is, **corrected** (executed after a bounded fix, e.g. a
per-tick clamp), or **rejected** (not executed; the applied command is zero
motion). Decisions carry the requested and applied commands plus per-check
outcomes, so a run's adapter behavior is reconstructable from its log alone
(guidelines §5: safety/logging are part of the research system).

Pure Python/numpy — no Isaac imports. Environments enter only through the
duck-typed Phase 2 accessor surface (see :mod:`alexdoor_xas.adapters.rollout`).
"""

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
    """Outcome of adapting one command (a per-tick delta or an A4 chunk).

    ``checks`` maps check name -> passed; a failed check either downgraded the
    status (with ``reason`` naming it) or was corrected. ``warnings`` are
    non-blocking observations (e.g. joint-limit drift within the known benign
    band). ``requested``/``applied`` are the command before/after adaptation;
    a rejected command applies zero motion.
    """

    status: AdapterStatus
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    warning_records: tuple[AdapterWarning, ...] = ()
    requested: np.ndarray | None = None
    applied: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "reason": self.reason,
            "checks": dict(self.checks),
            "warnings": list(self.warnings),
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
    """Per-tick world state the adapters check commands against.

    Mirrors the scripted controller's :class:`DoorPushObservation` but stays
    adapter-owned (adapters never import policies). ``joint_state`` /
    ``joint_limits`` carry the env's optional proprio accessors when available
    (Alex); EE orientation is captured even though v1 adapters do not command
    it directly; contact/force fields are ``None`` when sensing is unavailable.
    """

    door_frame: ObjectFrame | None
    hinge_angle_rad: float
    hinge_velocity_rad_s: float
    ee_pos_w: np.ndarray  # (3,)
    ee_quat_w_xyzw: np.ndarray | None = None  # (4,); populated by env snapshots
    contact_sensed: bool | None = None
    contact_force_n: float | None = None
    joint_state: dict[str, np.ndarray] | None = None
    joint_limits: dict[str, np.ndarray] | None = None
    joint_names: tuple[str, ...] | None = None
    tick_index: int | None = None
    rollout_phase: str = "unknown"


__all__ = ["AdapterDecision", "AdapterLog", "AdapterStatus", "AdapterWarning", "StepContext"]
