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
from alexdoor_xas.envs.door_task.contact_force import decode_contact_flag

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


def _read_hinge_state(env) -> tuple[float, float]:
    angle, velocity = env.hinge_state()
    angle_array = _finite_array("hinge angle", angle).reshape(-1)
    velocity_array = _finite_array("hinge velocity", velocity).reshape(-1)
    return float(angle_array[0]), float(velocity_array[0])


def _read_ee_pose(env) -> tuple[np.ndarray, np.ndarray]:
    ee_pos, ee_quat = env.ee_pose_w()
    position = _first_vector("end-effector position", ee_pos, 3)
    orientation = _first_vector("end-effector orientation", ee_quat, 4)
    orientation_norm = float(np.linalg.norm(orientation))
    if not np.isfinite(orientation_norm) or orientation_norm <= 0.0:
        raise InvalidSimulatorStateError(
            "invalid simulator state: end-effector orientation quaternion has invalid norm"
        )
    return position, orientation


def _read_contact_state(env) -> tuple[bool | None, float | None]:
    contact_sensed: bool | None = None
    if hasattr(env, "contact_sensed"):
        try:
            contact_sensed = decode_contact_flag(env.contact_sensed())
        except ValueError as exc:
            raise InvalidSimulatorStateError(
                f"invalid simulator state: contact value is invalid ({exc})"
            ) from exc
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
    return contact_sensed, contact_force_n


def _read_joint_state(env) -> tuple[dict[str, np.ndarray] | None, tuple[str, ...] | None]:
    if not hasattr(env, "robot_joint_state"):
        return None, None
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
    if len({value.shape for value in joint_state.values()}) != 1:
        raise InvalidSimulatorStateError(
            "invalid simulator state: joint positions, velocities, and targets "
            "must have matching shapes"
        )
    if not hasattr(env, "robot_joint_names"):
        return joint_state, None
    joint_names = tuple(str(name) for name in env.robot_joint_names())
    if len(joint_names) != joint_state["joint_pos"].size:
        raise InvalidSimulatorStateError(
            "invalid simulator state: joint names and state sizes do not match"
        )
    if len(set(joint_names)) != len(joint_names):
        raise InvalidSimulatorStateError(
            "invalid simulator state: robot joint names must be unique"
        )
    return joint_state, joint_names


def _validate_joint_state_limits(
    joint_state: dict[str, np.ndarray] | None,
    joint_limits: dict[str, np.ndarray] | None,
) -> None:
    if joint_limits is None:
        return
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
    if joint_state is not None and (
        pos_limits.shape[0] != joint_state["joint_pos"].size
        or vel_limits.reshape(-1).size != joint_state["joint_pos"].size
    ):
        raise InvalidSimulatorStateError(
            "invalid simulator state: joint state and limit sizes do not match"
        )


def read_step_context(
    env,
    door_frame: ObjectFrame | None,
    joint_limits: dict[str, np.ndarray] | None = None,
    execution_limits: RobotLimitsCfg | None = None,
) -> StepContext:
    """Snapshot and validate all physical state used by adapter execution."""
    _validate_execution_limits(execution_limits)
    angle, velocity = _read_hinge_state(env)
    position, orientation = _read_ee_pose(env)
    contact_sensed, contact_force_n = _read_contact_state(env)
    joint_state, joint_names = _read_joint_state(env)
    _validate_joint_state_limits(joint_state, joint_limits)
    return StepContext(
        door_frame=door_frame,
        hinge_angle_rad=angle,
        hinge_velocity_rad_s=velocity,
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


@dataclass(frozen=True)
class _RolloutStart:
    door_frame: ObjectFrame
    joint_limits: dict[str, np.ndarray] | None
    execution_limits: RobotLimitsCfg | None
    ctx: StepContext


@dataclass
class _RolloutState:
    ctx: StepContext
    initial_angle_rad: float
    ticks: int = 0
    reason: str | None = None
    notes: str = ""
    first_success_tick: int | None = None
    environment_terminated: bool = False
    environment_truncated: bool = False
    contact_ever_sensed: bool = False
    decisions: list[AdapterDecision] = field(default_factory=list)
    contact_per_tick: list[bool | None] = field(default_factory=list)
    force_n_per_tick: list[float | None] = field(default_factory=list)

    def to_result(self, log: AdapterLog, success_angle_rad: float | None) -> RolloutResult:
        return RolloutResult(
            n_ticks=self.ticks,
            initial_angle_rad=self.initial_angle_rad,
            final_angle_rad=self.ctx.hinge_angle_rad,
            log=log,
            notes=self.notes,
            decisions_per_tick=self.decisions,
            contact_per_tick=self.contact_per_tick,
            force_n_per_tick=self.force_n_per_tick,
            termination_reason=self.reason or "tick_budget",
            first_success_tick=self.first_success_tick,
            success=(self.first_success_tick is not None)
            if success_angle_rad is not None
            else None,
            environment_terminated=self.environment_terminated,
            environment_truncated=self.environment_truncated,
        )


@dataclass
class _RolloutRuntime:
    env: Any
    adapter: Any
    start: _RolloutStart
    state: _RolloutState
    max_ticks: int
    stop_on_reject: bool
    success_angle_rad: float | None
    post_success_diagnostic: bool
    step_hook: Callable[[int], None] | None

    def crossed_success(self) -> bool:
        return (
            self.success_angle_rad is not None
            and self.state.ctx.hinge_angle_rad >= self.success_angle_rad
        )

    def latch_initial_success(self) -> None:
        if not self.crossed_success():
            return
        self.state.first_success_tick = 0
        if not self.post_success_diagnostic:
            self.state.reason = "success"

    def active(self) -> bool:
        return self.state.reason is None and self.state.ticks < self.max_ticks

    def execute_delta(self, delta: np.ndarray) -> None:
        state = self.state
        phase = (
            "post_success"
            if state.first_success_tick is not None
            else ("contact" if state.contact_ever_sensed else "pre_contact")
        )
        state.ctx = replace(state.ctx, tick_index=state.ticks, rollout_phase=phase)
        applied, decision = self.adapter.process(delta, state.ctx)
        state.decisions.append(decision)
        if decision.status is AdapterStatus.REJECTED and self.stop_on_reject:
            state.notes = f"stopped on rejected command: {decision.reason}"
            state.reason = "rejection_stop"
            return

        terminated, truncated = step_env(self.env, applied)
        state.ticks += 1
        if terminated or truncated:
            self._stop_on_environment_end(terminated, truncated)
            return
        if self.step_hook is not None:
            self.step_hook(state.ticks)
        if not self._refresh_context():
            return
        self._record_post_step_state()

    def _stop_on_environment_end(self, terminated: bool, truncated: bool) -> None:
        state = self.state
        state.environment_terminated = terminated
        state.environment_truncated = truncated
        state.reason = "environment_terminated" if terminated else "environment_truncated"
        state.notes = (
            f"env reported {'termination' if terminated else 'truncation'} "
            f"at tick {state.ticks}; rollout state frozen at the last valid read"
        )

    def _refresh_context(self) -> bool:
        try:
            self.state.ctx = read_step_context(
                self.env,
                self.start.door_frame,
                self.start.joint_limits,
                self.start.execution_limits,
            )
        except InvalidSimulatorStateError as exc:
            self.state.reason = "invalid_simulator_state"
            self.state.notes = f"{exc} after tick {self.state.ticks}"
            return False
        return True

    def _record_post_step_state(self) -> None:
        state = self.state
        state.contact_per_tick.append(state.ctx.contact_sensed)
        state.force_n_per_tick.append(state.ctx.contact_force_n)
        state.contact_ever_sensed = state.contact_ever_sensed or state.ctx.contact_sensed is True
        if state.first_success_tick is None and self.crossed_success():
            state.first_success_tick = state.ticks
            if not self.post_success_diagnostic:
                state.reason = "success"
        if state.ticks >= self.max_ticks and not state.notes:
            state.notes = f"tick budget exhausted ({self.max_ticks})"

    def finish(self) -> None:
        if self.state.reason is None:
            self.state.reason = "tick_budget"
            self.state.notes = self.state.notes or f"tick budget exhausted ({self.max_ticks})"

    def assert_no_silent_reset(self) -> None:
        state = self.state
        if state.environment_terminated or state.environment_truncated:
            return
        if not hasattr(self.env, "episode_length_buf"):
            return
        env_ticks = int(_numpy(self.env.episode_length_buf).reshape(-1)[0])
        if env_ticks < state.ticks:
            raise RuntimeError(
                f"rollout executed {state.ticks} ticks but the env's episode counter reads "
                f"{env_ticks}: the env auto-reset mid-rollout without reporting "
                "termination — recorded state past the reset would be invalid"
            )


def _read_rollout_start(env, adapter) -> _RolloutStart:
    execution_limits = _adapter_limits(adapter)
    door_frame = read_door_frame(env)
    joint_limits = read_joint_limits(env)
    ctx = read_step_context(env, door_frame, joint_limits, execution_limits)
    return _RolloutStart(door_frame, joint_limits, execution_limits, ctx)


def _invalid_start_result(
    log: AdapterLog,
    error: InvalidSimulatorStateError,
    success_angle_rad: float | None,
) -> RolloutResult:
    return RolloutResult(
        n_ticks=0,
        initial_angle_rad=float("nan"),
        final_angle_rad=float("nan"),
        log=log,
        notes=str(error),
        termination_reason="invalid_simulator_state",
        success=False if success_angle_rad is not None else None,
    )


def _normalize_chunk(chunk: Any) -> np.ndarray:
    normalized = np.asarray(chunk, dtype=np.float64)
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    if normalized.ndim != 2 or normalized.shape[1] != EE_DELTA_DIM:
        raise ValueError(
            f"chunk source must emit (H, {EE_DELTA_DIM}) chunks, got {normalized.shape}"
        )
    return normalized


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
    try:
        start = _read_rollout_start(env, adapter)
    except InvalidSimulatorStateError as exc:
        return _invalid_start_result(log, exc, success_angle_rad)

    state = _RolloutState(
        ctx=start.ctx,
        initial_angle_rad=start.ctx.hinge_angle_rad,
        contact_ever_sensed=start.ctx.contact_sensed is True,
    )
    runtime = _RolloutRuntime(
        env=env,
        adapter=adapter,
        start=start,
        state=state,
        max_ticks=max_ticks,
        stop_on_reject=stop_on_reject,
        success_angle_rad=success_angle_rad,
        post_success_diagnostic=post_success_diagnostic,
        step_hook=step_hook,
    )
    runtime.latch_initial_success()

    while runtime.active():
        chunk = chunk_source(state.ctx)
        if chunk is None:
            state.reason = "policy_exhausted"
            break
        for delta in _normalize_chunk(chunk):
            runtime.execute_delta(delta)
            if not runtime.active():
                break

    runtime.finish()
    runtime.assert_no_silent_reset()
    return state.to_result(log, success_angle_rad)


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
