"""In-memory episode records."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from alexdoor_xas.action.spaces import EE_DELTA_DIM

TERMINATION_REASONS = (
    "controller_done",
    "controller_timeout",
    "tick_budget",
    "environment_terminated",
    "environment_truncated",
    "step_error",
)
LEGACY_TERMINATION_REASON = "not_recorded"


@dataclass(frozen=True)
class EpisodeMeta:
    """Fixed per-episode metadata (schema `meta` table)."""

    episode_id: str
    task: str
    action_space: str
    robot: str
    scene: str
    policy: str
    seed: int
    sim_dt: float
    control_dt: float
    created_utc: str
    robot_asset_id: str
    robot_asset_sha256: str

    @classmethod
    def create(
        cls,
        *,
        task: str,
        action_space: str,
        robot: str,
        scene: str,
        policy: str,
        seed: int,
        sim_dt: float,
        control_dt: float,
        robot_asset_id: str,
        robot_asset_sha256: str,
    ) -> EpisodeMeta:
        return cls(
            episode_id=str(uuid.uuid4()),
            task=task,
            action_space=action_space,
            robot=robot,
            scene=scene,
            policy=policy,
            seed=seed,
            sim_dt=sim_dt,
            control_dt=control_dt,
            created_utc=datetime.now(UTC).isoformat(),
            robot_asset_id=robot_asset_id,
            robot_asset_sha256=robot_asset_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeStep:
    """One control tick (schema `steps[]` table)."""

    t: float
    action: np.ndarray
    proprio: dict[str, np.ndarray]
    object_state: dict[str, float]
    contact: dict[str, Any]
    safety: dict[str, Any]


@dataclass(frozen=True)
class EpisodeOutcome:
    """Fixed per-episode result (schema `outcome` table)."""

    success: bool
    final_door_angle: float
    n_steps: int
    termination_reason: str
    environment_terminated: bool | None
    environment_truncated: bool | None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.termination_reason not in (*TERMINATION_REASONS, LEGACY_TERMINATION_REASON):
            raise ValueError(f"unknown factual termination reason: {self.termination_reason!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeBuffer:
    """One recorded trial."""

    meta: EpisodeMeta
    steps: list[EpisodeStep] = field(default_factory=list)
    outcome: EpisodeOutcome | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: EpisodeStep) -> None:
        if self.outcome is not None:
            raise RuntimeError("cannot add steps after the episode outcome is set")
        action = np.asarray(step.action, dtype=np.float64).reshape(-1)
        if action.shape != (EE_DELTA_DIM,):
            raise ValueError(f"step action must have shape ({EE_DELTA_DIM},), got {action.shape}")
        self.steps.append(step)

    def set_outcome(self, outcome: EpisodeOutcome) -> None:
        if outcome.n_steps != len(self.steps):
            raise ValueError(
                f"outcome.n_steps={outcome.n_steps} does not match recorded steps={len(self.steps)}"
            )
        self.outcome = outcome

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def stacked(self, getter) -> np.ndarray:
        if not self.steps:
            return np.zeros((0,))
        return np.stack([np.asarray(getter(step), dtype=np.float64) for step in self.steps])
