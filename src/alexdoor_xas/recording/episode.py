"""In-memory episode buffer mirroring docs/episode_schema.md (pure Python/numpy).

One episode is ``meta`` (fixed) + ``steps[]`` (one entry per control tick) +
``outcome`` (filled at the end). The buffer also carries ``extras`` for
per-episode payloads the Phase 2 exports need that are not per-step data:
the (static) door frame pose used for A2 <-> A3 conversion, the A4 chunk log,
and the sampled variation of randomized rollouts.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from alexdoor_xas.action.spaces import EE_DELTA_DIM


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
    chunk_len: int
    created_utc: str

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
        chunk_len: int = 1,
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
            chunk_len=chunk_len,
            created_utc=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeStep:
    """One control tick (schema `steps[]` table)."""

    t: float
    action: np.ndarray  # (EE_DELTA_DIM,) in meta.action_space
    obs_ref: dict[str, float]  # inline low-dim state (no image tensors in Phase 2)
    proprio: dict[str, np.ndarray]  # proxy EE pose (stands in for robot state)
    object_state: dict[str, float]  # door angle / angular velocity
    contact: dict[str, Any]  # inferred contact flag + source tag
    safety: dict[str, Any]  # adapter clamp flags


@dataclass(frozen=True)
class EpisodeOutcome:
    """Fixed per-episode result (schema `outcome` table)."""

    success: bool
    final_door_angle: float
    failure_label: str | None
    n_steps: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeBuffer:
    """One recorded trial: meta + steps + outcome (+ Phase 2 extras)."""

    meta: EpisodeMeta
    steps: list[EpisodeStep] = field(default_factory=list)
    outcome: EpisodeOutcome | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: EpisodeStep) -> None:
        if self.outcome is not None:
            raise RuntimeError("cannot add steps after the episode outcome is set")
        action = np.asarray(step.action, dtype=np.float64).reshape(-1)
        if action.shape != (EE_DELTA_DIM,):
            raise ValueError(
                f"step action must have shape ({EE_DELTA_DIM},), got {action.shape}"
            )
        self.steps.append(step)

    def set_outcome(self, outcome: EpisodeOutcome) -> None:
        if outcome.n_steps != len(self.steps):
            raise ValueError(
                f"outcome.n_steps={outcome.n_steps} does not match "
                f"recorded steps={len(self.steps)}"
            )
        self.outcome = outcome

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def stacked(self, getter) -> np.ndarray:
        """Stack a per-step quantity (``getter(step) -> array-like``) into (N, ...)."""
        if not self.steps:
            return np.zeros((0,))
        return np.stack([np.asarray(getter(step), dtype=np.float64) for step in self.steps])


__all__ = ["EpisodeBuffer", "EpisodeMeta", "EpisodeOutcome", "EpisodeStep"]
