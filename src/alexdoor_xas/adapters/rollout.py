"""Model-agnostic closed-loop rollout through adapter-v1.

This is the execution path learned policies (ACT, Diffusion Policy, later VLA)
share for evaluation: a *chunk source* (any callable — a learned policy, a
replay of recorded actions, a scripted planner) emits ``(H, 6)`` action chunks,
every step passes through an adapter (:class:`A2Adapter` for world-frame
deltas, :class:`A3Adapter` for door-frame deltas), and the adapted command is
what the env executes. Nothing here depends on a specific model.

The env is duck-typed through the benchmark accessor surface
(``door_frame_pose_w`` / ``hinge_state`` / ``ee_pose_w`` and optional
accessors probed via ``hasattr``) — the same protocol the data engine uses,
so the fakes in ``tests/conftest.py`` and the Isaac env work
unchanged. No Isaac imports; torch only at the ``env.step`` boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from alexdoor_xas.action.frames import ObjectFrame, door_frame_from_body_pose
from alexdoor_xas.action.spaces import EE_DELTA_DIM

from .base import AdapterDecision, AdapterLog, AdapterStatus, StepContext
from .limits import RobotLimitsCfg

ChunkSource = Callable[[StepContext], Any]
"""Emits the next ``(H, 6)`` action chunk given the current step context, or
``None`` to end the rollout."""


class InvalidSimulatorStateError(RuntimeError):
    """A simulator accessor returned state unsafe for adapter execution."""


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_array(name: str, value) -> np.ndarray:
    """Convert a numeric accessor value and fail closed on NaN/Inf/non-numeric data."""
    try:
        array = np.asarray(_numpy(value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidSimulatorStateError(
            f"invalid simulator state: {name} is not numeric"
        ) from exc
    if array.size == 0:
        raise InvalidSimulatorStateError(f"invalid simulator state: {name} is empty")
    if not np.isfinite(array).all():
        raise InvalidSimulatorStateError(
            f"invalid simulator state: {name} contains NaN or Inf"
        )
    return array


def _first_vector(name: str, value, width: int) -> np.ndarray:
    array = _finite_array(name, value)
    if array.ndim == 1 and array.shape == (width,):
        return array.copy()
    if array.ndim >= 2 and array.shape[-1] == width:
        return array.reshape(-1, width)[0].copy()
    raise InvalidSimulatorStateError(
        f"invalid simulator state: {name} must end in shape ({width},), got {array.shape}"
    )


def _validate_execution_limits(limits: RobotLimitsCfg | None) -> None:
    """Validate every numeric execution-limit input used by adapter checks."""
    if limits is None:
        return
    scalar_rules = (
        ("max_pos_delta_m", limits.max_pos_delta_m, True),
        ("max_rot_delta_rad", limits.max_rot_delta_rad, True),
        ("reach_margin_m", limits.reach_margin_m, False),
    )
    for name, value, strictly_positive in scalar_rules:
        scalar = _finite_array(f"adapter limits {name}", value).reshape(-1)
        if scalar.shape != (1,) or (scalar[0] <= 0.0 if strictly_positive else scalar[0] < 0.0):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise InvalidSimulatorStateError(
                f"invalid simulator state: adapter limits {name} must be finite and {qualifier}"
            )
    for name in (
        "contact_surface_x_m",
        "contact_approach_start_clearance_m",
        "contact_approach_max_step_m",
    ):
        value = getattr(limits, name)
        if value is not None:
            _finite_array(f"adapter limits {name}", value)
    workspace = limits.workspace
    if workspace is None:
        return
    center = _finite_array("workspace center_w", workspace.center_w).reshape(-1)
    if center.shape != (3,):
        raise InvalidSimulatorStateError(
            f"invalid simulator state: workspace center_w must have shape (3,), got {center.shape}"
        )
    min_reach = _finite_array("workspace min_reach_m", workspace.min_reach_m).reshape(-1)
    max_reach = _finite_array("workspace max_reach_m", workspace.max_reach_m).reshape(-1)
    if (
        min_reach.shape != (1,)
        or max_reach.shape != (1,)
        or min_reach[0] < 0.0
        or min_reach[0] >= max_reach[0]
    ):
        raise InvalidSimulatorStateError(
            "invalid simulator state: workspace reach bounds must be finite, non-negative, "
            "and increasing"
        )


def _adapter_limits(adapter) -> RobotLimitsCfg | None:
    limits = getattr(adapter, "limits", None)
    if limits is not None:
        return limits
    return getattr(getattr(adapter, "a2", None), "limits", None)


def read_door_frame(env) -> ObjectFrame:
    """Hinge-anchored door frame from the env (stage-read at reset, static)."""
    frame_pos, frame_quat = env.door_frame_pose_w()
    position = _first_vector("door frame position", frame_pos, 3)
    orientation = _first_vector("door frame orientation", frame_quat, 4)
    orientation_norm = float(np.linalg.norm(orientation))
    if not np.isfinite(orientation_norm) or orientation_norm <= 0.0:
        raise InvalidSimulatorStateError(
            "invalid simulator state: door frame orientation quaternion has invalid norm"
        )
    try:
        return door_frame_from_body_pose(position, orientation)
    except ValueError as exc:
        raise InvalidSimulatorStateError(
            f"invalid simulator state: door frame is invalid ({exc})"
        ) from exc


def read_joint_limits(env) -> dict[str, np.ndarray] | None:
    """Isaac-reported joint limits, when the env exposes them (read once)."""
    if not hasattr(env, "robot_joint_limits"):
        return None
    raw = env.robot_joint_limits()
    if not isinstance(raw, dict):
        raise InvalidSimulatorStateError(
            "invalid simulator state: robot_joint_limits() must return a dict"
        )
    return {name: _finite_array(f"joint limits {name}", value) for name, value in raw.items()}


def read_step_context(
    env,
    door_frame: ObjectFrame | None,
    joint_limits: dict[str, np.ndarray] | None = None,
    execution_limits: RobotLimitsCfg | None = None,
) -> StepContext:
    """Snapshot and validate all physical state used by adapter execution."""
    _validate_execution_limits(execution_limits)
    angle, velocity = env.hinge_state()
    angle_array = _finite_array("hinge angle", angle).reshape(-1)
    velocity_array = _finite_array("hinge velocity", velocity).reshape(-1)
    ee_pos, ee_quat = env.ee_pose_w()
    position = _first_vector("end-effector position", ee_pos, 3)
    orientation = _first_vector("end-effector orientation", ee_quat, 4)
    orientation_norm = float(np.linalg.norm(orientation))
    if not np.isfinite(orientation_norm) or orientation_norm <= 0.0:
        raise InvalidSimulatorStateError(
            "invalid simulator state: end-effector orientation quaternion has invalid norm"
        )
    contact_sensed: bool | None = None
    if hasattr(env, "contact_sensed"):
        raw_contact = _finite_array("contact value", env.contact_sensed()).reshape(-1)[0]
        contact_sensed = bool(raw_contact)
    contact_force_n: float | None = None
    if hasattr(env, "contact_force_w"):
        force = _finite_array("contact force", env.contact_force_w())
        if force.ndim == 0 or force.shape[-1] != 3:
            raise InvalidSimulatorStateError(
                f"invalid simulator state: contact force must end in shape (3,), got {force.shape}"
            )
        contact_force_n = float(np.linalg.norm(force.reshape(-1, 3)[0]))
        if not np.isfinite(contact_force_n):
            raise InvalidSimulatorStateError(
                "invalid simulator state: contact force magnitude is non-finite"
            )
    joint_state: dict[str, np.ndarray] | None = None
    joint_names: tuple[str, ...] | None = None
    if hasattr(env, "robot_joint_state"):
        raw_joint_state = env.robot_joint_state()
        if not isinstance(raw_joint_state, dict):
            raise InvalidSimulatorStateError(
                "invalid simulator state: robot_joint_state() must return a dict"
            )
        required_state = ("joint_pos", "joint_vel", "joint_pos_target")
        missing_state = [name for name in required_state if name not in raw_joint_state]
        if missing_state:
            raise InvalidSimulatorStateError(
                f"invalid simulator state: joint state is missing {missing_state}"
            )
        joint_state = {
            name: _finite_array(f"joint state {name}", raw_joint_state[name]).reshape(-1)
            for name in required_state
        }
        state_shapes = {value.shape for value in joint_state.values()}
        if len(state_shapes) != 1:
            raise InvalidSimulatorStateError(
                "invalid simulator state: joint positions, velocities, and targets "
                "must have matching shapes"
            )
        if hasattr(env, "robot_joint_names"):
            joint_names = tuple(str(name) for name in env.robot_joint_names())
            if len(joint_names) != joint_state["joint_pos"].size:
                raise InvalidSimulatorStateError(
                    "invalid simulator state: joint names and state sizes do not match"
                )
            if len(set(joint_names)) != len(joint_names):
                raise InvalidSimulatorStateError(
                    "invalid simulator state: robot joint names must be unique"
                )
    if joint_limits is not None:
        required_limits = ("joint_pos_limits", "joint_vel_limits")
        missing_limits = [name for name in required_limits if name not in joint_limits]
        if missing_limits:
            raise InvalidSimulatorStateError(
                f"invalid simulator state: joint limits are missing {missing_limits}"
            )
        pos_limits = _finite_array("joint position limits", joint_limits["joint_pos_limits"])
        vel_limits = _finite_array("joint velocity limits", joint_limits["joint_vel_limits"])
        if pos_limits.ndim != 2 or pos_limits.shape[1] != 2:
            raise InvalidSimulatorStateError(
                "invalid simulator state: joint position limits must have shape (N, 2)"
            )
        vel_limits = vel_limits.reshape(-1)
        if joint_state is not None and (
            pos_limits.shape[0] != joint_state["joint_pos"].size
            or vel_limits.size != joint_state["joint_pos"].size
        ):
            raise InvalidSimulatorStateError(
                "invalid simulator state: joint state and limit sizes do not match"
            )
    return StepContext(
        door_frame=door_frame,
        hinge_angle_rad=float(angle_array[0]),
        hinge_velocity_rad_s=float(velocity_array[0]),
        ee_pos_w=position,
        ee_quat_w_xyzw=orientation,
        contact_sensed=contact_sensed,
        contact_force_n=contact_force_n,
        joint_state=joint_state,
        joint_limits=joint_limits,
        joint_names=joint_names,
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
    "environment_terminated",
    "environment_truncated",
    "invalid_simulator_state",
    "tick_budget",
)
"""Every rollout ends with exactly one of these:

- ``success`` — the hinge crossed the success threshold (checked after every
  executed control tick, independent of policy chunk size);
- ``policy_exhausted`` — the chunk source returned ``None``;
- ``rejection_stop`` — a rejected command with ``stop_on_reject``;
- ``environment_terminated`` / ``environment_truncated`` — the corresponding
  factual flag was returned by ``env.step``;
- ``invalid_simulator_state`` — a required numeric simulator state or adapter
  limit was invalid; no command is adapted from that snapshot;
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
    On environment termination the final tick has no valid post-step read, so these
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
    environment_terminated: bool = False
    environment_truncated: bool = False
    """Factual Gymnasium flags; the final angle is the last valid pre-step read."""

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
            "environment_terminated": self.environment_terminated,
            "environment_truncated": self.environment_truncated,
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
    step_hook: Callable[[int], None] | None = None,
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

    ``env.step`` termination/truncation ends the rollout immediately with its
    factual environment reason: a ``DirectRLEnv`` auto-resets inside ``step``, so no
    post-reset state is read (the final angle is the last valid pre-step
    read). A defensive episode-counter guard (``env.episode_length_buf``)
    additionally fails loudly if an unreported mid-rollout reset slipped
    through — analogous to the data-engine guard.

    When provided, ``step_hook`` runs after every completed, non-truncated
    environment step. It receives the one-based tick count and may fail the
    rollout; video capture uses this boundary so a missing frame cannot be
    silently accepted.

    The env must already be reset; the door frame is read once up front (the
    stage-read pose is static for the episode in this build).
    """
    log: AdapterLog = adapter.log
    execution_limits = _adapter_limits(adapter)
    decisions: list[AdapterDecision] = []
    contact_per_tick: list[bool | None] = []
    force_n_per_tick: list[float | None] = []

    try:
        door_frame = read_door_frame(env)
        joint_limits = read_joint_limits(env)
        ctx = read_step_context(env, door_frame, joint_limits, execution_limits)
    except InvalidSimulatorStateError as exc:
        notes = str(exc)
        return RolloutResult(
            n_ticks=0,
            initial_angle_rad=float("nan"),
            final_angle_rad=float("nan"),
            log=log,
            notes=notes,
            termination_reason="invalid_simulator_state",
            success=False if success_angle_rad is not None else None,
        )
    initial_angle = ctx.hinge_angle_rad
    ticks = 0
    notes = ""
    reason: str | None = None
    first_success_tick: int | None = None
    environment_terminated = False
    environment_truncated = False
    contact_ever_sensed = ctx.contact_sensed is True

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
            phase = (
                "post_success"
                if first_success_tick is not None
                else ("contact" if contact_ever_sensed else "pre_contact")
            )
            ctx = replace(ctx, tick_index=ticks, rollout_phase=phase)
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
                environment_terminated = terminated
                environment_truncated = truncated
                reason = "environment_terminated" if terminated else "environment_truncated"
                notes = (
                    f"env reported {'termination' if terminated else 'truncation'} "
                    f"at tick {ticks}; rollout state frozen at the last valid read"
                )
                break
            if step_hook is not None:
                step_hook(ticks)
            try:
                ctx = read_step_context(env, door_frame, joint_limits, execution_limits)
            except InvalidSimulatorStateError as exc:
                reason = "invalid_simulator_state"
                notes = f"{exc} after tick {ticks}"
                break
            contact_per_tick.append(ctx.contact_sensed)
            force_n_per_tick.append(ctx.contact_force_n)
            contact_ever_sensed = contact_ever_sensed or ctx.contact_sensed is True
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

    if not (environment_terminated or environment_truncated) and hasattr(
        env, "episode_length_buf"
    ):
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
        environment_terminated=environment_terminated,
        environment_truncated=environment_truncated,
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
    "InvalidSimulatorStateError",
    "RolloutResult",
    "read_door_frame",
    "read_joint_limits",
    "read_step_context",
    "replay_source",
    "rollout_chunks",
    "step_env",
]
