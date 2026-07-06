"""Model-agnostic closed-loop rollout through adapter-v1.

This is the execution path learned policies (ACT, Diffusion Policy, later VLA)
share for evaluation: a *chunk source* (any callable — a learned policy, a
replay of recorded actions, a scripted planner) emits ``(H, 6)`` action chunks,
every step passes through an adapter (:class:`A2Adapter` for world-frame
deltas, :class:`A3Adapter` for door-frame deltas), and the adapted command is
what the env executes. Nothing here depends on a specific model.

The env is duck-typed through the frozen Phase 2 accessor surface
(``door_frame_pose_w`` / ``hinge_state`` / ``proxy_pose_w`` and the optional
Phase 2.5 accessors probed via ``hasattr``) — the same protocol the data
engine uses, so the fakes in ``tests/conftest.py`` and both Isaac envs work
unchanged. No Isaac imports; torch only at the ``env.step`` boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from alexdoor_xas.action.frames import ObjectFrame, door_frame_from_body_pose
from alexdoor_xas.action.spaces import EE_DELTA_DIM

from .base import AdapterDecision, AdapterLog, AdapterStatus, StepContext

ChunkSource = Callable[[StepContext], Any]
"""Emits the next ``(H, 6)`` action chunk given the current step context, or
``None`` to end the rollout."""


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_door_frame(env) -> ObjectFrame:
    """Hinge-anchored door frame from the env (stage-read at reset, static)."""
    frame_pos, frame_quat = env.door_frame_pose_w()
    return door_frame_from_body_pose(_numpy(frame_pos)[0], _numpy(frame_quat)[0])


def read_joint_limits(env) -> dict[str, np.ndarray] | None:
    """Isaac-reported joint limits, when the env exposes them (read once)."""
    if not hasattr(env, "robot_joint_limits"):
        return None
    return {
        name: np.asarray(value, dtype=np.float64)
        for name, value in env.robot_joint_limits().items()
    }


def read_step_context(
    env,
    door_frame: ObjectFrame | None,
    joint_limits: dict[str, np.ndarray] | None = None,
) -> StepContext:
    """Snapshot the per-tick world state the adapters check commands against."""
    angle, velocity = env.hinge_state()
    ee_pos, _ = env.proxy_pose_w()
    contact_sensed: bool | None = None
    if hasattr(env, "contact_sensed"):
        contact_sensed = bool(_numpy(env.contact_sensed())[0])
    joint_state: dict[str, np.ndarray] | None = None
    if hasattr(env, "robot_joint_state"):
        joint_state = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in env.robot_joint_state().items()
        }
    return StepContext(
        door_frame=door_frame,
        hinge_angle_rad=float(_numpy(angle)[0]),
        hinge_velocity_rad_s=float(_numpy(velocity)[0]),
        ee_pos_w=_numpy(ee_pos)[0].astype(np.float64),
        contact_sensed=contact_sensed,
        joint_state=joint_state,
        joint_limits=joint_limits,
    )


def step_env(env, delta_world: np.ndarray) -> None:
    """Execute one adapted world-frame EE delta on the env."""
    action = torch.as_tensor(
        np.asarray(delta_world, dtype=np.float64), dtype=torch.float32
    ).reshape(1, -1)
    env.step(action)


@dataclass
class RolloutResult:
    """One adapter-mediated rollout: door motion + the full decision log."""

    n_ticks: int
    initial_angle_rad: float
    final_angle_rad: float
    log: AdapterLog
    notes: str = ""
    decisions_per_tick: list[AdapterDecision] = field(default_factory=list)

    @property
    def door_angle_change_rad(self) -> float:
        return self.final_angle_rad - self.initial_angle_rad

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_ticks": self.n_ticks,
            "initial_angle_rad": self.initial_angle_rad,
            "final_angle_rad": self.final_angle_rad,
            "door_angle_change_rad": self.door_angle_change_rad,
            "notes": self.notes,
            "log": self.log.to_dict(),
        }


def rollout_chunks(
    env,
    chunk_source: ChunkSource,
    adapter,
    max_ticks: int = 600,
    stop_on_reject: bool = False,
) -> RolloutResult:
    """Drive the env with adapter-mediated chunks until exhaustion or budget.

    ``adapter`` is anything with ``process(delta, ctx) -> (applied, decision)``
    (:class:`A2Adapter` for world-frame chunks, :class:`A3Adapter` for
    door-frame chunks). Every emitted step is adapted against a fresh context
    and executed; a rejected step executes zero motion (tick accounting stays
    aligned with the source's chunk clock) unless ``stop_on_reject``.

    The env must already be reset; the door frame is read once up front (the
    stage-read pose is static for the episode in this build).
    """
    door_frame = read_door_frame(env)
    joint_limits = read_joint_limits(env)
    log: AdapterLog = adapter.log
    decisions: list[AdapterDecision] = []

    ctx = read_step_context(env, door_frame, joint_limits)
    initial_angle = ctx.hinge_angle_rad
    ticks = 0
    notes = ""
    while ticks < max_ticks:
        chunk = chunk_source(ctx)
        if chunk is None:
            break
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2 or chunk.shape[1] != EE_DELTA_DIM:
            raise ValueError(
                f"chunk source must emit (H, {EE_DELTA_DIM}) chunks, got {chunk.shape}"
            )
        stop = False
        for delta in chunk:
            applied, decision = adapter.process(delta, ctx)
            decisions.append(decision)
            if decision.status is AdapterStatus.REJECTED and stop_on_reject:
                notes = f"stopped on rejected command: {decision.reason}"
                stop = True
                break
            step_env(env, applied)
            ticks += 1
            ctx = read_step_context(env, door_frame, joint_limits)
            if ticks >= max_ticks:
                notes = notes or f"tick budget exhausted ({max_ticks})"
                stop = True
                break
        if stop:
            break

    return RolloutResult(
        n_ticks=ticks,
        initial_angle_rad=initial_angle,
        final_angle_rad=ctx.hinge_angle_rad,
        log=log,
        notes=notes,
        decisions_per_tick=decisions,
    )


def replay_source(actions) -> ChunkSource:
    """Chunk source that replays a recorded ``(N, 6)`` action sequence.

    The Phase 3.1 gate uses this to prove replay equivalence: a recorded
    episode's actions pushed through the adapters reproduce the same door
    motion (headless physics is deterministic in this build).
    """
    remaining = [np.asarray(action, dtype=np.float64).reshape(EE_DELTA_DIM) for action in actions]
    iterator = iter(remaining)

    def source(ctx: StepContext):
        del ctx
        try:
            return next(iterator).reshape(1, EE_DELTA_DIM)
        except StopIteration:
            return None

    return source


__all__ = [
    "ChunkSource",
    "RolloutResult",
    "read_door_frame",
    "read_joint_limits",
    "read_step_context",
    "replay_source",
    "rollout_chunks",
    "step_env",
]
