"""A4 adapter: object-centric chunks -> guarded A3/A2 execution.

Adapter-v1 for ``A4_obj_centric_chunk``: interprets an
:class:`~alexdoor_xas.action.spaces.ObjectCentricChunk` sequence (phase,
panel-frame contact target, intended hinge delta, duration), validates it
against door geometry and the robot's workspace, plans the guarded
approach -> pre-contact -> contact -> push sequence, and executes it
closed-loop by emitting per-tick door-frame deltas through the
:class:`A3Adapter` -> :class:`A2Adapter` path.

Semantics (frozen): a chunk's ``motion_hinge_delta_rad`` is **controller
intent** — the Phase 2 chunk logs stay intent labels. The adapter adds the
achieved side: every execution result reports ``requested_hinge_delta_rad``
vs ``achieved_hinge_delta_rad`` plus contact reached/missed, so intent and
outcome are never conflated.

Planning uses only the chunk's contact target ``(y, z)``: the x coordinate is
recomputed from the panel geometry and the phase's clearance (the chunk's x
carries the emitter's EE-at-face convention and is sanity-checked, not
trusted). Push direction: the panel's push face is +X, so positive hinge
deltas (opening) are executable; negative deltas mean pulling, which this
no-grasp scene cannot do — rejected, not corrected.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from alexdoor_xas.action.frames import rot_z
from alexdoor_xas.action.spaces import A4_PHASE_VOCAB, EE_DELTA_DIM, ObjectCentricChunk
from alexdoor_xas.calibration.alex_v2_door import AlexV2DoorCalibration

from .a3 import A3Adapter
from .base import AdapterDecision, AdapterLog, AdapterStatus, AdapterWarning
from .limits import MAX_HINGE_ANGLE_RAD, DoorPanelGeometry, RobotLimitsCfg
from .rollout import read_door_frame, read_joint_limits, read_step_context, step_env

_CONTACT_PHASES = ("contact", "push", "hold")


@dataclass(frozen=True)
class A4AdapterCfg:
    """Guarded-execution geometry and budgets for one calibrated Alex V2 setup."""

    approach_standoff_m: float
    align_standoff_m: float
    pre_contact_clearance_m: float
    contact_clearance_m: float
    release_standoff_m: float

    max_step_m: float = 0.015
    approach_tol_m: float = 0.020
    align_tol_m: float = 0.010
    pre_contact_tol_m: float = 0.005

    budget_headroom: float = 3.0
    """Stage tick budget = max(min_stage_budget, duration_ticks * headroom):
    chunk durations are intent recorded from one executor; another executor
    (or a re-simulated one) legitimately tracks slower."""
    min_stage_budget_ticks: int = 150
    push_stall_ticks: int = 60
    """Consecutive near-zero hinge-velocity ticks in PUSH before declaring the
    push stalled (insufficient door motion)."""
    push_stall_min_vel_rad_s: float = 1e-3
    target_nudge_tol_m: float = 0.02
    """Off-panel contact targets within this distance are corrected (clamped
    onto the panel); beyond it the chunk is rejected."""
    target_x_face_tol_m: float = 0.05
    """Warn when the chunk's x deviates from the EE-at-face convention by more."""


def alex_v2_a4_cfg(calibration: AlexV2DoorCalibration) -> A4AdapterCfg:
    """Build A4 execution geometry from the validated Alex V2 calibration."""

    values = calibration.controller
    return A4AdapterCfg(
        approach_standoff_m=float(values["approach_standoff_m"]),
        align_standoff_m=float(values["align_standoff_m"]),
        pre_contact_clearance_m=float(values["pre_contact_clearance_m"]),
        contact_clearance_m=float(values["contact_clearance_m"]),
        release_standoff_m=float(values["release_standoff_m"]),
    )


_PHASE_CLEARANCE_ATTR = {
    "approach": "approach_standoff_m",
    "align": "align_standoff_m",
    "pre_contact": "pre_contact_clearance_m",
    "contact": "contact_clearance_m",
    "push": "contact_clearance_m",
    "hold": "contact_clearance_m",
    "release": "release_standoff_m",
}


@dataclass(frozen=True)
class _Stage:
    """One executable stage of the guarded plan."""

    phase: str
    target_panel_yz: tuple[float, float]
    clearance_m: float
    budget_ticks: int
    hinge_delta_rad: float = 0.0
    hold_ticks: int = 0
    synthesized: bool = False


class _ChunkValidationError(ValueError):
    """Internal fail-fast signal carrying a user-facing rejection reason."""


@dataclass
class _ChunkValidationState:
    chunk: ObjectCentricChunk
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    warning_records: list[AdapterWarning] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)

    def require(self, name: str, condition: bool, reason: str) -> None:
        self.checks[name] = bool(condition)
        if not condition:
            raise _ChunkValidationError(reason)

    def fail(self, name: str, reason: str) -> None:
        self.checks[name] = False
        raise _ChunkValidationError(reason)


@dataclass
class StageResult:
    phase: str
    completed: bool
    reason: str
    ticks: int
    entry_angle_rad: float
    exit_angle_rad: float
    synthesized: bool = False
    contact_reached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "completed": self.completed,
            "reason": self.reason,
            "ticks": self.ticks,
            "entry_angle_rad": self.entry_angle_rad,
            "exit_angle_rad": self.exit_angle_rad,
            "synthesized": self.synthesized,
            "contact_reached": self.contact_reached,
        }


@dataclass
class A4ExecutionResult:
    """What executing one A4 chunk sequence did to the world.

    ``status`` is the *adaptation* outcome (accepted / corrected / rejected);
    ``completed`` + ``failure`` describe the *execution* outcome. A rejected
    sequence executes nothing.
    """

    status: AdapterStatus
    reason: str
    requested_hinge_delta_rad: float
    achieved_hinge_delta_rad: float
    contact_reached: bool
    initial_angle_rad: float
    final_angle_rad: float
    n_ticks: int
    completed: bool
    failure: str
    environment_terminated: bool = False
    environment_truncated: bool = False
    chunk_decisions: list[AdapterDecision] = field(default_factory=list)
    stages: list[StageResult] = field(default_factory=list)
    log: AdapterLog = field(default_factory=AdapterLog)

    @property
    def final_door_angle_change_rad(self) -> float:
        return self.final_angle_rad - self.initial_angle_rad

    @property
    def contact_missed(self) -> bool:
        return self.failure == "missed_contact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "reason": self.reason,
            "requested_hinge_delta_rad": self.requested_hinge_delta_rad,
            "achieved_hinge_delta_rad": self.achieved_hinge_delta_rad,
            "contact_reached": self.contact_reached,
            "contact_missed": self.contact_missed,
            "initial_angle_rad": self.initial_angle_rad,
            "final_angle_rad": self.final_angle_rad,
            "final_door_angle_change_rad": self.final_door_angle_change_rad,
            "n_ticks": self.n_ticks,
            "completed": self.completed,
            "failure": self.failure,
            "environment_terminated": self.environment_terminated,
            "environment_truncated": self.environment_truncated,
            "chunk_decisions": [decision.to_dict() for decision in self.chunk_decisions],
            "stages": [stage.to_dict() for stage in self.stages],
            "log": self.log.to_dict(),
        }


@dataclass(frozen=True)
class _ValidatedSequence:
    chunks: list[ObjectCentricChunk]
    decisions: list[AdapterDecision]
    status: AdapterStatus
    rejected_reason: str = ""


@dataclass
class _A4ExecutionState:
    ctx: Any
    ticks: int = 0
    stage_results: list[StageResult] = field(default_factory=list)
    contact_reached: bool = False
    push_entry_angle: float | None = None
    push_exit_angle: float | None = None
    failure: str = ""
    environment_terminated: bool = False
    environment_truncated: bool = False


@dataclass
class _StageExecution:
    env: Any
    stage: _Stage
    ctx: Any
    door_frame: Any
    joint_limits: dict[str, np.ndarray] | None
    max_ticks: int
    total_ticks: int
    stage_ticks: int = 0
    stall_ticks: int = 0
    contact_reached: bool = False


@dataclass(frozen=True)
class _StageObservation:
    angle: float
    ee_door: np.ndarray
    in_contact: bool


@dataclass(frozen=True)
class _StageOutcome:
    result: StageResult
    ctx: Any
    total_ticks: int
    environment_terminated: bool = False
    environment_truncated: bool = False


class A4Adapter:
    """Validate, plan, and execute object-centric chunks through A3/A2."""

    def __init__(
        self,
        a3: A3Adapter,
        *,
        cfg: A4AdapterCfg,
        geometry: DoorPanelGeometry | None = None,
    ):
        self.a3 = a3
        self.geometry = geometry or DoorPanelGeometry()
        self.cfg = cfg

    @property
    def limits(self) -> RobotLimitsCfg:
        return self.a3.a2.limits

    @property
    def log(self) -> AdapterLog:
        return self.a3.log

    # -- validation ------------------------------------------------------------

    def validate_chunk(
        self,
        chunk: ObjectCentricChunk,
        entry_angle_rad: float,
        door_frame=None,
    ) -> tuple[ObjectCentricChunk, AdapterDecision]:
        """Validate one chunk at its predicted entry hinge angle.

        Returns the (possibly corrected) chunk and the decision. A rejected
        chunk is returned unchanged; callers must not execute it.
        """
        state = _ChunkValidationState(chunk)
        try:
            target, numeric = self._parse_chunk(state)
            hinge_delta = self._validate_phase_and_hinge(state, numeric)
            target = self._validate_contact_target(state, target)
            hinge_delta = self._fit_hinge_travel(state, hinge_delta, entry_angle_rad)
            self._validate_reach(
                state,
                target,
                entry_angle_rad,
                entry_angle_rad + hinge_delta,
                door_frame,
            )
        except _ChunkValidationError as exc:
            return chunk, self._reject_chunk(
                chunk,
                state.checks,
                str(exc),
                warnings=state.warnings,
                warning_records=state.warning_records,
            )
        corrected_chunk = replace(
            chunk,
            contact_target_panel=tuple(float(v) for v in target),
            motion_hinge_delta_rad=float(hinge_delta),
        )
        if state.corrections:
            decision = AdapterDecision(
                status=AdapterStatus.CORRECTED,
                reason="; ".join(state.corrections),
                checks=state.checks,
                warnings=tuple(state.warnings),
                warning_records=tuple(state.warning_records),
                requested=numeric,
                applied=np.array(
                    [
                        *corrected_chunk.contact_target_panel,
                        corrected_chunk.motion_hinge_delta_rad,
                        float(corrected_chunk.duration_ticks),
                    ]
                ),
            )
        else:
            decision = AdapterDecision(
                status=AdapterStatus.ACCEPTED,
                checks=state.checks,
                warnings=tuple(state.warnings),
                warning_records=tuple(state.warning_records),
                requested=numeric,
                applied=numeric,
            )
        return corrected_chunk, decision

    def _parse_chunk(self, state: _ChunkValidationState) -> tuple[np.ndarray, np.ndarray]:
        chunk = state.chunk
        try:
            target = np.asarray(chunk.contact_target_panel, dtype=np.float64)
        except (TypeError, ValueError):
            state.fail(
                "target_shape",
                "contact_target_panel must be a numeric shape (3,) vector, got "
                f"{type(chunk.contact_target_panel).__name__}",
            )
        state.require(
            "target_shape",
            target.shape == (3,),
            f"contact_target_panel must have shape (3,), got shape {target.shape}",
        )
        try:
            numeric = np.array(
                [*target, chunk.motion_hinge_delta_rad, float(chunk.duration_ticks)],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            state.fail("finite", "chunk contains non-numeric values")
        return target, numeric

    def _validate_phase_and_hinge(self, state: _ChunkValidationState, numeric: np.ndarray) -> float:
        chunk = state.chunk
        hinge_delta = float(numeric[3])
        state.require(
            "phase_known",
            chunk.phase in A4_PHASE_VOCAB,
            f"unknown A4 phase {chunk.phase!r} (vocabulary: {A4_PHASE_VOCAB})",
        )
        state.require(
            "finite", bool(np.isfinite(numeric).all()), "chunk contains non-finite values"
        )
        state.require(
            "duration_positive",
            float(numeric[4]) > 0,
            f"duration_ticks must be positive, got {chunk.duration_ticks}",
        )
        state.require(
            "push_not_pull",
            hinge_delta >= 0.0,
            f"hinge delta {hinge_delta:.4f} rad is negative: pulling the door is physically "
            "invalid for this scene (push face only, no grasp)",
        )
        state.require(
            "hinge_delta_phase_valid",
            chunk.phase == "push" or hinge_delta == 0.0,
            "non-push phase cannot request hinge motion",
        )
        return hinge_delta

    def _validate_contact_target(
        self, state: _ChunkValidationState, target: np.ndarray
    ) -> np.ndarray:
        geo = self.geometry
        state.checks["target_on_panel"] = geo.on_panel(target)
        if not state.checks["target_on_panel"]:
            nudged = geo.clamp_to_panel(target)
            offset = float(np.linalg.norm(nudged - target))
            if offset > self.cfg.target_nudge_tol_m:
                raise _ChunkValidationError(
                    f"contact target {target.round(3).tolist()} is {offset:.3f} m off the "
                    f"panel (correction tolerance {self.cfg.target_nudge_tol_m} m)"
                )
            target = nudged
            state.corrections.append(f"contact target nudged {offset:.3f} m onto the panel")
        state.require(
            "target_clear_of_handle",
            not geo.in_handle_band(target),
            f"contact target {target.round(3).tolist()} lies in the handle band "
            f"y in {list(geo.handle_band_y_m)}, z in {list(geo.handle_band_z_m)}",
        )
        self._record_face_warning(state, target)
        return target

    def _record_face_warning(self, state: _ChunkValidationState, target: np.ndarray) -> None:
        face_x = self.geometry.surface_x_m(0.0)
        deviation = abs(float(target[0] - face_x))
        if deviation <= self.cfg.target_x_face_tol_m:
            return
        message = (
            f"chunk target x={target[0]:.3f} deviates from the EE-at-face convention "
            f"x={face_x:.3f}; planning recomputes x from phase clearances"
        )
        state.warnings.append(message)
        state.warning_records.append(
            AdapterWarning(
                id="a4.target_face_deviation",
                message=message,
                evidence={
                    "target_x_m": float(target[0]),
                    "configured_face_x_m": face_x,
                    "deviation_m": deviation,
                    "phase": state.chunk.phase,
                },
            )
        )

    def _fit_hinge_travel(
        self,
        state: _ChunkValidationState,
        hinge_delta: float,
        entry_angle_rad: float,
    ) -> float:
        state.checks["within_hinge_travel"] = entry_angle_rad + hinge_delta <= MAX_HINGE_ANGLE_RAD
        if state.checks["within_hinge_travel"]:
            return hinge_delta
        capped = max(MAX_HINGE_ANGLE_RAD - entry_angle_rad, 0.0)
        state.corrections.append(
            f"hinge delta capped from {hinge_delta:.4f} to {capped:.4f} rad "
            f"(remaining travel to {MAX_HINGE_ANGLE_RAD:.4f})"
        )
        return capped

    def _validate_reach(
        self,
        state: _ChunkValidationState,
        target: np.ndarray,
        entry_angle: float,
        exit_angle: float,
        door_frame,
    ) -> None:
        state.checks["reachable"] = True
        if self.limits.workspace is None or door_frame is None:
            return
        reason = self._reach_reason(target, entry_angle, exit_angle, state.chunk, door_frame)
        if reason:
            state.fail("reachable", reason)

    def _reach_reason(
        self, target_panel, entry_angle: float, exit_angle: float, chunk, door_frame
    ) -> str:
        """Workspace check across the swept hinge arc (fixed-base robots only)."""
        workspace = self.limits.workspace
        geo = self.geometry
        probes: list[tuple[str, np.ndarray, float]] = []
        for angle_name, angle in (("entry", entry_angle), ("exit", exit_angle)):
            contact = np.array(
                [geo.surface_x_m(0.0), target_panel[1], target_panel[2]], dtype=np.float64
            )
            probes.append((f"contact point at {angle_name} angle", contact, angle))
        # The approach waypoint matters even for contact/push chunks: guarded
        # execution synthesizes the approach prefix from the same target, and
        # the measured Alex failure mode is exactly this waypoint folding
        # inside min reach.
        approach = np.array(
            [geo.surface_x_m(self.cfg.approach_standoff_m), target_panel[1], target_panel[2]],
            dtype=np.float64,
        )
        probes.append(("approach waypoint at entry angle", approach, entry_angle))
        for name, point_panel, angle in probes:
            point_w = door_frame.point_to_world(rot_z(angle) @ point_panel)
            distance = workspace.distance(point_w)
            if workspace.beyond_max_reach(point_w, self.limits.reach_margin_m):
                return (
                    f"{name} is {distance:.3f} m from the shoulder, beyond max reach "
                    f"{workspace.max_reach_m:.3f} m"
                )
            if workspace.within_min_reach(point_w):
                return (
                    f"{name} is {distance:.3f} m from the shoulder, inside min reach "
                    f"{workspace.min_reach_m:.3f} m (near-singular fold region)"
                )
        return ""

    def _reject_chunk(
        self,
        chunk: ObjectCentricChunk,
        checks: dict[str, bool],
        reason: str,
        warnings: list[str] | None = None,
        warning_records: list[AdapterWarning] | None = None,
    ) -> AdapterDecision:
        requested = self._chunk_requested_vector(chunk)
        return AdapterDecision(
            status=AdapterStatus.REJECTED,
            reason=reason,
            checks=checks,
            warnings=tuple(warnings or ()),
            warning_records=tuple(warning_records or ()),
            requested=requested,
            applied=None,
        )

    def _chunk_requested_vector(self, chunk: ObjectCentricChunk) -> np.ndarray | None:
        """Best-effort numeric chunk vector for telemetry; never raises."""
        try:
            target = np.asarray(chunk.contact_target_panel, dtype=np.float64).reshape(-1)
            tail = np.array(
                [chunk.motion_hinge_delta_rad, float(chunk.duration_ticks)], dtype=np.float64
            )
        except (TypeError, ValueError):
            return None
        return np.concatenate([target, tail])

    def _requested_hinge_delta(self, chunks: Sequence[ObjectCentricChunk]) -> float:
        """Best-effort requested hinge-motion summary for rejected malformed chunks."""
        total = 0.0
        for chunk in chunks:
            try:
                value = float(chunk.motion_hinge_delta_rad)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                total += value
        return total

    # -- planning ----------------------------------------------------------------

    def plan(self, chunks: Sequence[ObjectCentricChunk]) -> list[_Stage]:
        """Turn validated chunks into the guarded executable stage sequence.

        Inserts synthesized approach/pre-contact stages when the sequence jumps
        straight to a contact/push/hold chunk (guarded approach: never command
        the contact clearance from far away).
        """
        stages: list[_Stage] = []
        seen_pre_contact = False
        for chunk in chunks:
            yz = (float(chunk.contact_target_panel[1]), float(chunk.contact_target_panel[2]))
            budget = max(
                self.cfg.min_stage_budget_ticks,
                int(math.ceil(chunk.duration_ticks * self.cfg.budget_headroom)),
            )
            if chunk.phase in _CONTACT_PHASES and not seen_pre_contact:
                stages.append(
                    _Stage(
                        phase="approach",
                        target_panel_yz=yz,
                        clearance_m=self.cfg.approach_standoff_m,
                        budget_ticks=max(self.cfg.min_stage_budget_ticks, budget),
                        synthesized=True,
                    )
                )
                stages.append(
                    _Stage(
                        phase="pre_contact",
                        target_panel_yz=yz,
                        clearance_m=self.cfg.pre_contact_clearance_m,
                        budget_ticks=max(self.cfg.min_stage_budget_ticks, budget),
                        synthesized=True,
                    )
                )
                seen_pre_contact = True
            if chunk.phase == "pre_contact":
                seen_pre_contact = True
            stages.append(
                _Stage(
                    phase=chunk.phase,
                    target_panel_yz=yz,
                    clearance_m=getattr(self.cfg, _PHASE_CLEARANCE_ATTR[chunk.phase]),
                    budget_ticks=budget,
                    hinge_delta_rad=chunk.motion_hinge_delta_rad,
                    hold_ticks=chunk.duration_ticks if chunk.phase == "hold" else 0,
                )
            )
        return stages

    # -- execution ---------------------------------------------------------------

    def execute(
        self,
        env,
        chunks: ObjectCentricChunk | Sequence[ObjectCentricChunk],
        max_ticks: int = 1800,
    ) -> A4ExecutionResult:
        """Validate, plan, and execute a chunk (sequence) on a reset env.

        A validation rejection rejects the whole sequence and commands no
        motion (guarded execution never partially runs an invalid plan).
        """
        if isinstance(chunks, ObjectCentricChunk):
            chunks = [chunks]
        chunks = list(chunks)
        if not chunks:
            raise ValueError("cannot execute an empty A4 chunk sequence")

        door_frame = read_door_frame(env)
        joint_limits = read_joint_limits(env)
        ctx = read_step_context(env, door_frame, joint_limits, self.limits)
        initial_angle = ctx.hinge_angle_rad
        validated = self._validate_sequence(chunks, initial_angle, door_frame)
        if validated.rejected_reason:
            return self._rejected_execution_result(chunks, validated, initial_angle)

        state = _A4ExecutionState(ctx=ctx)
        reason = "; ".join(decision.reason for decision in validated.decisions if decision.reason)
        for stage in self.plan(validated.chunks):
            outcome = self._run_stage(
                _StageExecution(
                    env=env,
                    stage=stage,
                    ctx=state.ctx,
                    door_frame=door_frame,
                    joint_limits=joint_limits,
                    max_ticks=max_ticks,
                    total_ticks=state.ticks,
                )
            )
            self._apply_stage_outcome(state, stage, outcome)
            if not outcome.result.completed:
                state.failure = self._classify_stage_failure(stage, outcome)
                failure_reason = f"{stage.phase} stage failed: {outcome.result.reason}"
                reason = f"{reason}; {failure_reason}" if reason else failure_reason
                break

        requested_delta = sum(chunk.motion_hinge_delta_rad for chunk in validated.chunks)
        achieved = self._achieved_push_delta(state)
        return A4ExecutionResult(
            status=validated.status,
            reason=reason,
            requested_hinge_delta_rad=requested_delta,
            achieved_hinge_delta_rad=achieved,
            contact_reached=state.contact_reached,
            initial_angle_rad=initial_angle,
            final_angle_rad=state.ctx.hinge_angle_rad,
            n_ticks=state.ticks,
            completed=not state.failure,
            failure=state.failure,
            environment_terminated=state.environment_terminated,
            environment_truncated=state.environment_truncated,
            chunk_decisions=validated.decisions,
            stages=state.stage_results,
            log=self.log,
        )

    def _validate_sequence(
        self,
        chunks: list[ObjectCentricChunk],
        initial_angle: float,
        door_frame,
    ) -> _ValidatedSequence:
        validated: list[ObjectCentricChunk] = []
        decisions: list[AdapterDecision] = []
        entry_angle = initial_angle
        status = AdapterStatus.ACCEPTED
        for index, chunk in enumerate(chunks):
            fixed, decision = self.validate_chunk(chunk, entry_angle, door_frame)
            decisions.append(decision)
            self.log.record(decision)
            if decision.status is AdapterStatus.REJECTED:
                reason = f"chunk {index} ({chunk.phase}): {decision.reason}"
                return _ValidatedSequence(validated, decisions, AdapterStatus.REJECTED, reason)
            if decision.status is AdapterStatus.CORRECTED:
                status = AdapterStatus.CORRECTED
            validated.append(fixed)
            entry_angle += fixed.motion_hinge_delta_rad
        return _ValidatedSequence(validated, decisions, status)

    def _rejected_execution_result(
        self,
        chunks: list[ObjectCentricChunk],
        validated: _ValidatedSequence,
        initial_angle: float,
    ) -> A4ExecutionResult:
        return A4ExecutionResult(
            status=AdapterStatus.REJECTED,
            reason=validated.rejected_reason,
            requested_hinge_delta_rad=self._requested_hinge_delta(chunks),
            achieved_hinge_delta_rad=0.0,
            contact_reached=False,
            initial_angle_rad=initial_angle,
            final_angle_rad=initial_angle,
            n_ticks=0,
            completed=False,
            failure="",
            chunk_decisions=validated.decisions,
            log=self.log,
        )

    def _run_stage(self, execution: _StageExecution) -> _StageOutcome:
        """Run one guarded stage and freeze the last pre-reset context."""
        stage = execution.stage
        entry_angle = execution.ctx.hinge_angle_rad
        push_target_angle = entry_angle + stage.hinge_delta_rad
        completed = False
        why = ""
        environment_terminated = False
        environment_truncated = False

        while True:
            observation = self._stage_observation(execution)
            done, why = self._stage_done(execution, observation, push_target_angle)
            if done:
                completed = True
                break
            why = self._stage_stop_reason(execution, observation.angle)
            if why:
                break
            delta_door = self._stage_delta(stage, observation.angle, observation.ee_door)
            applied, decision = self.a3.process(delta_door, execution.ctx)
            if decision.status is AdapterStatus.REJECTED:
                why = f"per-tick command rejected: {decision.reason}"
                break
            terminated, truncated = step_env(execution.env, applied)
            execution.total_ticks += 1
            execution.stage_ticks += 1
            if terminated or truncated:
                environment_terminated = terminated
                environment_truncated = truncated
                label = "terminated" if terminated else "truncated"
                why = f"environment {label} at tick {execution.total_ticks}"
                break
            execution.ctx = read_step_context(
                execution.env,
                execution.door_frame,
                execution.joint_limits,
                self.limits,
            )

        result = StageResult(
            phase=stage.phase,
            completed=completed,
            reason=why,
            ticks=execution.stage_ticks,
            entry_angle_rad=entry_angle,
            exit_angle_rad=execution.ctx.hinge_angle_rad,
            synthesized=stage.synthesized,
            contact_reached=execution.contact_reached,
        )
        return _StageOutcome(
            result=result,
            ctx=execution.ctx,
            total_ticks=execution.total_ticks,
            environment_terminated=environment_terminated,
            environment_truncated=environment_truncated,
        )

    def _stage_observation(self, execution: _StageExecution) -> _StageObservation:
        angle = execution.ctx.hinge_angle_rad
        ee_door = execution.door_frame.point_from_world(execution.ctx.ee_pos_w)
        ee_panel = rot_z(angle).T @ ee_door
        sensed = execution.ctx.contact_sensed
        in_contact = (
            bool(sensed) if sensed is not None else self.geometry.geometric_contact(ee_panel)
        )
        execution.contact_reached = execution.contact_reached or in_contact
        return _StageObservation(angle, ee_door, in_contact)

    def _stage_stop_reason(self, execution: _StageExecution, angle: float) -> str:
        stage = execution.stage
        if execution.stage_ticks >= stage.budget_ticks:
            return f"stage tick budget exhausted ({stage.budget_ticks})"
        if execution.total_ticks >= execution.max_ticks:
            return f"rollout tick budget exhausted ({execution.max_ticks})"
        if stage.phase != "push":
            return ""
        if abs(execution.ctx.hinge_velocity_rad_s) < self.cfg.push_stall_min_vel_rad_s:
            execution.stall_ticks += 1
        else:
            execution.stall_ticks = 0
        if execution.stall_ticks < self.cfg.push_stall_ticks:
            return ""
        return (
            f"push stalled: |hinge velocity| < {self.cfg.push_stall_min_vel_rad_s} rad/s "
            f"for {execution.stall_ticks} ticks at angle {angle:.4f} rad"
        )

    def _stage_delta(self, stage: _Stage, angle: float, ee_door: np.ndarray) -> np.ndarray:
        delta_door = np.zeros(EE_DELTA_DIM)
        if stage.phase == "hold":
            return delta_door
        target_panel = np.array(
            [self.geometry.surface_x_m(stage.clearance_m), *stage.target_panel_yz],
            dtype=np.float64,
        )
        error = rot_z(angle) @ target_panel - ee_door
        distance = float(np.linalg.norm(error))
        step = (
            error if distance <= self.cfg.max_step_m else error * (self.cfg.max_step_m / distance)
        )
        delta_door[:3] = step
        return delta_door

    def _apply_stage_outcome(
        self,
        state: _A4ExecutionState,
        stage: _Stage,
        outcome: _StageOutcome,
    ) -> None:
        result = outcome.result
        state.stage_results.append(result)
        state.ctx = outcome.ctx
        state.ticks = outcome.total_ticks
        state.contact_reached = state.contact_reached or result.contact_reached
        state.environment_terminated = outcome.environment_terminated
        state.environment_truncated = outcome.environment_truncated
        if stage.phase == "push":
            if state.push_entry_angle is None:
                state.push_entry_angle = result.entry_angle_rad
            state.push_exit_angle = result.exit_angle_rad

    def _classify_stage_failure(self, stage: _Stage, outcome: _StageOutcome) -> str:
        if outcome.environment_terminated:
            return "environment_terminated"
        if outcome.environment_truncated:
            return "environment_truncated"
        reason = outcome.result.reason
        if stage.phase in ("contact", "pre_contact") and "budget" in reason:
            return "missed_contact"
        if stage.phase == "push" and "stalled" in reason:
            return "push_stalled"
        if "rejected" in reason:
            return "command_rejected"
        return "stage_timeout"

    @staticmethod
    def _achieved_push_delta(state: _A4ExecutionState) -> float:
        if state.push_entry_angle is None or state.push_exit_angle is None:
            return 0.0
        return state.push_exit_angle - state.push_entry_angle

    def _stage_done(
        self,
        execution: _StageExecution,
        observation: _StageObservation,
        push_target_angle: float,
    ) -> tuple[bool, str]:
        stage = execution.stage
        ee_door = observation.ee_door
        angle = observation.angle
        in_contact = observation.in_contact
        geo = self.geometry
        cfg = self.cfg
        target_panel = np.array(
            [geo.surface_x_m(stage.clearance_m), *stage.target_panel_yz], dtype=np.float64
        )
        distance = float(np.linalg.norm(rot_z(angle) @ target_panel - ee_door))
        if stage.phase in ("approach", "release"):
            return distance <= cfg.approach_tol_m, "waypoint reached"
        if stage.phase == "align":
            return distance <= cfg.align_tol_m, "waypoint reached"
        if stage.phase == "pre_contact":
            return (
                distance <= cfg.pre_contact_tol_m or bool(in_contact),
                "pre-contact pose reached",
            )
        if stage.phase == "contact":
            return bool(in_contact), "contact established"
        if stage.phase == "push":
            return angle >= push_target_angle, "requested hinge delta achieved"
        if stage.phase == "hold":
            return execution.stage_ticks >= stage.hold_ticks, "hold duration elapsed"
        raise ValueError(f"unplannable stage phase {stage.phase!r}")


__all__ = [
    "A4Adapter",
    "A4AdapterCfg",
    "A4ExecutionResult",
    "StageResult",
    "alex_v2_a4_cfg",
]
