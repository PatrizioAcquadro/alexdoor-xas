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


def step_env(env, delta_world: np.ndarray) -> tuple[bool, bool]:
    """Execute one adapted world-frame EE delta; returns ``(terminated, truncated)``.

    A ``DirectRLEnv`` auto-resets *inside* ``env.step`` when either flag is
    set, so any state read after a flagged step is post-reset — callers must
    stop consuming the env immediately.
    """
    action = torch.as_tensor(
        np.asarray(delta_world, dtype=np.float64), dtype=torch.float32
    ).reshape(1, -1)
    result = env.step(action)
    if isinstance(result, tuple) and len(result) >= 4:
        terminated = bool(_numpy(result[2]).reshape(-1)[0])
        truncated = bool(_numpy(result[3]).reshape(-1)[0])
        return terminated, truncated
    return False, False


TERMINATION_REASONS = (
    "success",
    "policy_exhausted",
    "rejection_stop",
    "env_truncated",
    "tick_budget",
)
"""Every rollout ends with exactly one of these:

- ``success`` — the hinge crossed the success threshold (checked after every
  executed control tick, independent of policy chunk size);
- ``policy_exhausted`` — the chunk source returned ``None``;
- ``rejection_stop`` — a rejected command with ``stop_on_reject``;
- ``env_truncated`` — ``env.step`` reported terminated/truncated (the env
  auto-reset internally; no post-reset state is consumed);
- ``tick_budget`` — the rollout's ``max_ticks`` budget ran out.
"""


@dataclass
class RolloutResult:
    """One adapter-mediated rollout: door motion + the full decision log."""

    n_ticks: int
    initial_angle_rad: float
    final_angle_rad: float
    log: AdapterLog
    notes: str = ""
    decisions_per_tick: list[AdapterDecision] = field(default_factory=list)
    contact_per_tick: list[bool | None] = field(default_factory=list)
    """Post-step force-sensed contact flag per executed tick (``None`` when the
    env exposes no contact sensing). Additive: existing consumers ignore it.
    On ``env_truncated`` the final tick has no valid post-step read, so these
    lists are one entry shorter than ``n_ticks``."""
    force_n_per_tick: list[float | None] = field(default_factory=list)
    """Post-step |contact force| in newtons per executed tick (``None`` when
    the env exposes no ``contact_force_w``)."""
    termination_reason: str = "tick_budget"
    """One of :data:`TERMINATION_REASONS`."""
    first_success_tick: int | None = None
    """Executed-tick count at the first success-threshold crossing (0 = the
    reset state already satisfied it); ``None`` = never crossed or no
    threshold was given. Chunk-size independent by construction."""
    success: bool | None = None
    """First-crossing success (``None`` when no threshold was given). A
    cross-then-rebound trajectory stays successful with its original
    crossing tick."""
    env_truncated: bool = False
    """``env.step`` flagged terminated/truncated: the env auto-reset itself and
    ``final_angle_rad`` is the last valid pre-step read."""

    @property
    def door_angle_change_rad(self) -> float:
        return self.final_angle_rad - self.initial_angle_rad

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_ticks": self.n_ticks,
            "initial_angle_rad": self.initial_angle_rad,
            "final_angle_rad": self.final_angle_rad,
            "door_angle_change_rad": self.door_angle_change_rad,
            "termination_reason": self.termination_reason,
            "first_success_tick": self.first_success_tick,
            "success": self.success,
            "env_truncated": self.env_truncated,
            "notes": self.notes,
            "log": self.log.to_dict(),
        }


def rollout_chunks(
    env,
    chunk_source: ChunkSource,
    adapter,
    max_ticks: int = 600,
    stop_on_reject: bool = False,
    success_angle_rad: float | None = None,
    post_success_diagnostic: bool = False,
) -> RolloutResult:
    """Drive the env with adapter-mediated chunks until success/exhaustion/budget.

    ``adapter`` is anything with ``process(delta, ctx) -> (applied, decision)``
    (:class:`A2Adapter` for world-frame chunks, :class:`A3Adapter` for
    door-frame chunks). Every emitted step is adapted against a fresh context
    and executed; a rejected step executes zero motion (tick accounting stays
    aligned with the source's chunk clock) unless ``stop_on_reject``.

    ``success_angle_rad`` enables per-tick success semantics: the hinge
    threshold is checked after **every executed control tick**, so
    ``first_success_tick`` is the exact first crossing independent of the
    policy's chunk size, and the rollout stops there unless
    ``post_success_diagnostic`` explicitly requests post-success execution
    (success and its crossing tick are latched either way — a later rebound
    cannot unlabel it).

    ``env.step`` termination/truncation ends the rollout immediately with
    ``env_truncated``: a ``DirectRLEnv`` auto-resets inside ``step``, so no
    post-reset state is read (the final angle is the last valid pre-step
    read). A defensive episode-counter guard (``env.episode_length_buf``)
    additionally fails loudly if an unreported mid-rollout reset slipped
    through — analogous to the data-engine guard.

    The env must already be reset; the door frame is read once up front (the
    stage-read pose is static for the episode in this build).
    """
    door_frame = read_door_frame(env)
    joint_limits = read_joint_limits(env)
    log: AdapterLog = adapter.log
    decisions: list[AdapterDecision] = []
    has_force = hasattr(env, "contact_force_w")
    contact_per_tick: list[bool | None] = []
    force_n_per_tick: list[float | None] = []

    ctx = read_step_context(env, door_frame, joint_limits)
    initial_angle = ctx.hinge_angle_rad
    ticks = 0
    notes = ""
    reason: str | None = None
    first_success_tick: int | None = None
    env_truncated = False

    def crossed() -> bool:
        return success_angle_rad is not None and ctx.hinge_angle_rad >= success_angle_rad

    if crossed():
        first_success_tick = 0
        if not post_success_diagnostic:
            reason = "success"

    while reason is None and ticks < max_ticks:
        chunk = chunk_source(ctx)
        if chunk is None:
            reason = "policy_exhausted"
            break
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2 or chunk.shape[1] != EE_DELTA_DIM:
            raise ValueError(
                f"chunk source must emit (H, {EE_DELTA_DIM}) chunks, got {chunk.shape}"
            )
        for delta in chunk:
            applied, decision = adapter.process(delta, ctx)
            decisions.append(decision)
            if decision.status is AdapterStatus.REJECTED and stop_on_reject:
                notes = f"stopped on rejected command: {decision.reason}"
                reason = "rejection_stop"
                break
            terminated, truncated = step_env(env, applied)
            ticks += 1
            if terminated or truncated:
                # The env auto-reset inside step: everything readable now is
                # post-reset state. Keep the last valid pre-step context as
                # the final state and stop without any further env reads.
                env_truncated = True
                reason = "env_truncated"
                notes = (
                    f"env reported {'termination' if terminated else 'truncation'} "
                    f"at tick {ticks}; rollout state frozen at the last valid read"
                )
                break
            ctx = read_step_context(env, door_frame, joint_limits)
            contact_per_tick.append(ctx.contact_sensed)
            force_n_per_tick.append(
                float(np.linalg.norm(_numpy(env.contact_force_w())[0])) if has_force else None
            )
            if first_success_tick is None and crossed():
                first_success_tick = ticks
                if not post_success_diagnostic:
                    reason = "success"
                    break
            if ticks >= max_ticks:
                notes = notes or f"tick budget exhausted ({max_ticks})"
                break
        # An exhausted chunk loops back for the next chunk unless a stop
        # reason was latched or the budget ran out (reason set on re-check).

    if reason is None:
        # Diagnostic post-success execution still records why it *stopped*;
        # the first-crossing success/tick are latched separately above.
        reason = "tick_budget"
        notes = notes or f"tick budget exhausted ({max_ticks})"

    if not env_truncated and hasattr(env, "episode_length_buf"):
        env_ticks = int(_numpy(env.episode_length_buf).reshape(-1)[0])
        if env_ticks < ticks:
            raise RuntimeError(
                f"rollout executed {ticks} ticks but the env's episode counter reads "
                f"{env_ticks}: the env auto-reset mid-rollout without reporting "
                "termination — recorded state past the reset would be invalid"
            )

    return RolloutResult(
        n_ticks=ticks,
        initial_angle_rad=initial_angle,
        final_angle_rad=ctx.hinge_angle_rad,
        log=log,
        notes=notes,
        decisions_per_tick=decisions,
        contact_per_tick=contact_per_tick,
        force_n_per_tick=force_n_per_tick,
        termination_reason=reason,
        first_success_tick=first_success_tick,
        success=(first_success_tick is not None) if success_angle_rad is not None else None,
        env_truncated=env_truncated,
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
    "TERMINATION_REASONS",
    "ChunkSource",
    "RolloutResult",
    "read_door_frame",
    "read_joint_limits",
    "read_step_context",
    "replay_source",
    "rollout_chunks",
    "step_env",
]
