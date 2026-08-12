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

from .a3 import A3Adapter
from .base import AdapterDecision, AdapterLog, AdapterStatus, AdapterWarning
from .limits import MAX_HINGE_ANGLE_RAD, DoorPanelGeometry, RobotLimitsCfg
from .rollout import read_door_frame, read_joint_limits, read_step_context, step_env

_CONTACT_PHASES = ("contact", "push", "hold")


@dataclass(frozen=True)
class A4AdapterCfg:
    """Guarded-execution geometry and budgets (defaults = the Alex preset).

    Standoffs/clearances are the Phase 2.5 Alex controller values
    (``alex_fixedbase_push_cfg``); they also work for synthetic test doubles, which
    has no reach constraints.
    """

    approach_standoff_m: float = 0.12
    align_standoff_m: float = 0.06
    pre_contact_clearance_m: float = 0.010
    contact_clearance_m: float = -0.005
    release_standoff_m: float = 0.30

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
    failure: str  # "" | "missed_contact" | "push_stalled" | "stage_timeout" | "command_rejected"
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
            "chunk_decisions": [decision.to_dict() for decision in self.chunk_decisions],
            "stages": [stage.to_dict() for stage in self.stages],
            "log": self.log.to_dict(),
        }


class A4Adapter:
    """Validate, plan, and execute object-centric chunks through A3/A2."""

    def __init__(
        self,
        a3: A3Adapter,
        geometry: DoorPanelGeometry | None = None,
        cfg: A4AdapterCfg | None = None,
    ):
        self.a3 = a3
        self.geometry = geometry or DoorPanelGeometry()
        self.cfg = cfg or A4AdapterCfg()

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
        geo = self.geometry
        checks: dict[str, bool] = {}
        warnings: list[str] = []
        warning_records: list[AdapterWarning] = []
        corrections: list[str] = []

        try:
            target = np.asarray(chunk.contact_target_panel, dtype=np.float64)
        except (TypeError, ValueError):
            checks["target_shape"] = False
            return chunk, self._reject_chunk(
                chunk,
                checks,
                f"contact_target_panel must be a numeric shape (3,) vector, got "
                f"{type(chunk.contact_target_panel).__name__}",
            )
        checks["target_shape"] = target.shape == (3,)
        if not checks["target_shape"]:
            return chunk, self._reject_chunk(
                chunk,
                checks,
                f"contact_target_panel must have shape (3,), got shape {target.shape}",
            )

        try:
            numeric = np.array(
                [*target, chunk.motion_hinge_delta_rad, float(chunk.duration_ticks)],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            checks["finite"] = False
            return chunk, self._reject_chunk(chunk, checks, "chunk contains non-numeric values")
        hinge_delta_requested = float(numeric[3])
        duration_ticks = float(numeric[4])

        checks["phase_known"] = chunk.phase in A4_PHASE_VOCAB
        if not checks["phase_known"]:
            return chunk, self._reject_chunk(
                chunk, checks, f"unknown A4 phase {chunk.phase!r} (vocabulary: {A4_PHASE_VOCAB})"
            )
        checks["finite"] = bool(np.isfinite(numeric).all())
        if not checks["finite"]:
            return chunk, self._reject_chunk(chunk, checks, "chunk contains non-finite values")
        checks["duration_positive"] = duration_ticks > 0
        if not checks["duration_positive"]:
            return chunk, self._reject_chunk(
                chunk, checks, f"duration_ticks must be positive, got {chunk.duration_ticks}"
            )

        checks["push_not_pull"] = hinge_delta_requested >= 0.0
        if not checks["push_not_pull"]:
            return chunk, self._reject_chunk(
                chunk,
                checks,
                f"hinge delta {hinge_delta_requested:.4f} rad is negative: pulling the "
                "door is physically invalid for this scene (push face only, no grasp)",
            )

        checks["hinge_delta_phase_valid"] = chunk.phase == "push" or hinge_delta_requested == 0.0
        if not checks["hinge_delta_phase_valid"]:
            return chunk, self._reject_chunk(
                chunk,
                checks,
                "non-push phase cannot request hinge motion",
            )

        checks["target_on_panel"] = geo.on_panel(target)
        if not checks["target_on_panel"]:
            nudged = geo.clamp_to_panel(target)
            offset = float(np.linalg.norm(nudged - target))
            if offset > self.cfg.target_nudge_tol_m:
                return chunk, self._reject_chunk(
                    chunk,
                    checks,
                    f"contact target {target.round(3).tolist()} is {offset:.3f} m off the "
                    f"panel (correction tolerance {self.cfg.target_nudge_tol_m} m)",
                )
            target = nudged
            corrections.append(f"contact target nudged {offset:.3f} m onto the panel")

        checks["target_clear_of_handle"] = not geo.in_handle_band(target)
        if not checks["target_clear_of_handle"]:
            return chunk, self._reject_chunk(
                chunk,
                checks,
                f"contact target {target.round(3).tolist()} lies in the handle band "
                f"y in {list(geo.handle_band_y_m)}, z in {list(geo.handle_band_z_m)}",
            )
        if abs(target[0] - geo.surface_x_m(0.0)) > self.cfg.target_x_face_tol_m:
            warnings.append(
                f"chunk target x={target[0]:.3f} deviates from the EE-at-face convention "
                f"x={geo.surface_x_m(0.0):.3f}; planning recomputes x from phase clearances"
            )
            warning_records.append(
                AdapterWarning(
                    id="a4.target_face_deviation",
                    message=warnings[-1],
                    evidence={
                        "target_x_m": float(target[0]),
                        "configured_face_x_m": geo.surface_x_m(0.0),
                        "deviation_m": abs(float(target[0] - geo.surface_x_m(0.0))),
                        "phase": chunk.phase,
                    },
                )
            )

        hinge_delta = hinge_delta_requested
        exit_angle = entry_angle_rad + hinge_delta
        checks["within_hinge_travel"] = exit_angle <= MAX_HINGE_ANGLE_RAD
        if not checks["within_hinge_travel"]:
            capped = max(MAX_HINGE_ANGLE_RAD - entry_angle_rad, 0.0)
            corrections.append(
                f"hinge delta capped from {hinge_delta:.4f} to {capped:.4f} rad "
                f"(remaining travel to {MAX_HINGE_ANGLE_RAD:.4f})"
            )
            hinge_delta = capped
            exit_angle = entry_angle_rad + hinge_delta

        checks["reachable"] = True
        if self.limits.workspace is not None and door_frame is not None:
            reason = self._reach_reason(target, entry_angle_rad, exit_angle, chunk, door_frame)
            if reason:
                checks["reachable"] = False
                return chunk, self._reject_chunk(
                    chunk,
                    checks,
                    reason,
                    warnings=warnings,
                    warning_records=warning_records,
                )

        corrected_chunk = replace(
            chunk,
            contact_target_panel=tuple(float(v) for v in target),
            motion_hinge_delta_rad=float(hinge_delta),
        )
        if corrections:
            decision = AdapterDecision(
                status=AdapterStatus.CORRECTED,
                reason="; ".join(corrections),
                checks=checks,
                warnings=tuple(warnings),
                warning_records=tuple(warning_records),
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
                checks=checks,
                warnings=tuple(warnings),
                warning_records=tuple(warning_records),
                requested=numeric,
                applied=numeric,
            )
        return corrected_chunk, decision

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

        validated: list[ObjectCentricChunk] = []
        decisions: list[AdapterDecision] = []
        entry_angle = initial_angle
        overall_status = AdapterStatus.ACCEPTED
        for chunk in chunks:
            fixed, decision = self.validate_chunk(chunk, entry_angle, door_frame)
            decisions.append(decision)
            self.log.record(decision)
            if decision.status is AdapterStatus.REJECTED:
                return A4ExecutionResult(
                    status=AdapterStatus.REJECTED,
                    reason=f"chunk {len(decisions) - 1} ({chunk.phase}): {decision.reason}",
                    requested_hinge_delta_rad=self._requested_hinge_delta(chunks),
                    achieved_hinge_delta_rad=0.0,
                    contact_reached=False,
                    initial_angle_rad=initial_angle,
                    final_angle_rad=initial_angle,
                    n_ticks=0,
                    completed=False,
                    failure="",
                    chunk_decisions=decisions,
                    log=self.log,
                )
            if decision.status is AdapterStatus.CORRECTED:
                overall_status = AdapterStatus.CORRECTED
            validated.append(fixed)
            entry_angle += fixed.motion_hinge_delta_rad

        requested_delta = sum(chunk.motion_hinge_delta_rad for chunk in validated)
        stages = self.plan(validated)

        stage_results: list[StageResult] = []
        contact_reached = False
        push_entry_angle: float | None = None
        push_exit_angle: float | None = None
        ticks = 0
        failure = ""
        reason = "; ".join(d.reason for d in decisions if d.reason) or ""

        for stage in stages:
            result, ctx, ticks = self._run_stage(
                env, stage, ctx, door_frame, joint_limits, ticks, max_ticks
            )
            stage_results.append(result)
            if result.contact_reached:
                contact_reached = True
            if stage.phase == "push":
                if push_entry_angle is None:
                    push_entry_angle = result.entry_angle_rad
                push_exit_angle = result.exit_angle_rad
            if not result.completed:
                if stage.phase in ("contact", "pre_contact") and "budget" in result.reason:
                    failure = "missed_contact"
                elif stage.phase == "push" and "stalled" in result.reason:
                    failure = "push_stalled"
                elif "rejected" in result.reason:
                    failure = "command_rejected"
                else:
                    failure = "stage_timeout"
                reason = (reason + "; " if reason else "") + (
                    f"{stage.phase} stage failed: {result.reason}"
                )
                break

        achieved = 0.0
        if push_entry_angle is not None and push_exit_angle is not None:
            achieved = push_exit_angle - push_entry_angle

        return A4ExecutionResult(
            status=overall_status,
            reason=reason,
            requested_hinge_delta_rad=requested_delta,
            achieved_hinge_delta_rad=achieved,
            contact_reached=contact_reached,
            initial_angle_rad=initial_angle,
            final_angle_rad=ctx.hinge_angle_rad,
            n_ticks=ticks,
            completed=not failure,
            failure=failure,
            chunk_decisions=decisions,
            stages=stage_results,
            log=self.log,
        )

    def _run_stage(self, env, stage: _Stage, ctx, door_frame, joint_limits, ticks, max_ticks):
        """Closed-loop stage execution; returns (StageResult, last ctx, ticks)."""
        cfg = self.cfg
        geo = self.geometry
        entry_angle = ctx.hinge_angle_rad
        push_target_angle = entry_angle + stage.hinge_delta_rad
        stage_ticks = 0
        stall_ticks = 0
        completed = False
        contact_reached = False
        why = ""

        while True:
            angle = ctx.hinge_angle_rad
            ee_door = door_frame.point_from_world(ctx.ee_pos_w)
            ee_panel = rot_z(angle).T @ ee_door
            in_contact = (
                ctx.contact_sensed
                if ctx.contact_sensed is not None
                else geo.geometric_contact(ee_panel)
            )
            contact_reached = contact_reached or bool(in_contact)

            done, why = self._stage_done(
                stage, ee_door, angle, in_contact, push_target_angle, stage_ticks
            )
            if done:
                completed = True
                break
            if stage_ticks >= stage.budget_ticks:
                why = f"stage tick budget exhausted ({stage.budget_ticks})"
                break
            if ticks >= max_ticks:
                why = f"rollout tick budget exhausted ({max_ticks})"
                break
            if stage.phase == "push":
                if abs(ctx.hinge_velocity_rad_s) < cfg.push_stall_min_vel_rad_s:
                    stall_ticks += 1
                else:
                    stall_ticks = 0
                if stall_ticks >= cfg.push_stall_ticks:
                    why = (
                        f"push stalled: |hinge velocity| < {cfg.push_stall_min_vel_rad_s} rad/s "
                        f"for {stall_ticks} ticks at angle {angle:.4f} rad"
                    )
                    break

            target_panel = np.array(
                [geo.surface_x_m(stage.clearance_m), *stage.target_panel_yz], dtype=np.float64
            )
            target_door = rot_z(angle) @ target_panel
            delta_door = np.zeros(EE_DELTA_DIM)
            if stage.phase != "hold":
                error = target_door - ee_door
                distance = float(np.linalg.norm(error))
                step = error if distance <= cfg.max_step_m else error * (cfg.max_step_m / distance)
                delta_door[:3] = step

            applied, decision = self.a3.process(delta_door, ctx)
            if decision.status is AdapterStatus.REJECTED:
                why = f"per-tick command rejected: {decision.reason}"
                break
            step_env(env, applied)
            ticks += 1
            stage_ticks += 1
            ctx = read_step_context(env, door_frame, joint_limits, self.limits)

        result = StageResult(
            phase=stage.phase,
            completed=completed,
            reason=why,
            ticks=stage_ticks,
            entry_angle_rad=entry_angle,
            exit_angle_rad=ctx.hinge_angle_rad,
            synthesized=stage.synthesized,
            contact_reached=contact_reached,
        )
        return result, ctx, ticks

    def _stage_done(
        self, stage: _Stage, ee_door, angle, in_contact, push_target_angle, stage_ticks
    ) -> tuple[bool, str]:
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
            return stage_ticks >= stage.hold_ticks, "hold duration elapsed"
        raise ValueError(f"unplannable stage phase {stage.phase!r}")


__all__ = ["A4Adapter", "A4AdapterCfg", "A4ExecutionResult", "StageResult"]
