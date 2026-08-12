"""Deterministic scripted door-push controller (Phase 2 baseline).

A small finite-state machine that drives the Alex right-gripper EE through::

    APPROACH -> ALIGN -> PRE_CONTACT -> CONTACT -> PUSH -> HOLD -> RELEASE

Everything is expressed **door-relative**: waypoints live in the panel frame
(they move with the door), the emitted per-tick command is a 6-dim EE delta in
the hinge-anchored *door frame* (the A3 representation), and the controller
never sees world coordinates except through the door frame it is given. It is
pure Python/numpy — no Isaac imports — so the FSM is unit-testable against
synthetic door kinematics.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.action.spaces import EE_DELTA_DIM, ChunkLog, ObjectCentricChunk


class DoorPushPhase(enum.StrEnum):
    APPROACH = "approach"
    ALIGN = "align"
    PRE_CONTACT = "pre_contact"
    CONTACT = "contact"
    PUSH = "push"
    HOLD = "hold"
    RELEASE = "release"
    DONE = "done"


PHASE_ORDER: tuple[DoorPushPhase, ...] = (
    DoorPushPhase.APPROACH,
    DoorPushPhase.ALIGN,
    DoorPushPhase.PRE_CONTACT,
    DoorPushPhase.CONTACT,
    DoorPushPhase.PUSH,
    DoorPushPhase.HOLD,
    DoorPushPhase.RELEASE,
)


@dataclass(frozen=True)
class DoorPushControllerCfg:
    """Door-relative geometry and per-phase budgets for the scripted push.

    Distances are meters, angles radians. The panel occupies
    ``x in [0, panel_thickness_m]``, ``y in [0, panel_width_m]``, and
    ``z in [-panel_height_m/2, panel_height_m/2]`` in the panel frame; pushing
    the +X face toward -X produces positive hinge torque (opens the door).
    """

    panel_width_m: float = 0.83
    panel_height_m: float = 2.0
    panel_thickness_m: float = 0.036

    push_radius_frac: float = 0.8
    """Push point distance from the hinge, as a fraction of the panel width."""
    push_height_m: float = -0.30
    """Push point height in the door frame (0 = hinge-origin height, i.e. panel
    mid-height). Kept below -0.15 so the tool point stays on the flat panel
    face and clears the handle, which protrudes around door-frame y in
    [0.63, 0.80], z in [0.0, 0.09]."""

    approach_standoff_m: float = 0.35
    align_standoff_m: float = 0.12
    pre_contact_clearance_m: float = 0.010
    contact_clearance_m: float = -0.005
    """Commanded surface clearance while pushing (slightly inside the face)."""
    release_standoff_m: float = 0.30

    approach_tol_m: float = 0.020
    align_tol_m: float = 0.010
    pre_contact_tol_m: float = 0.005
    contact_eps_m: float = 0.002
    """Inferred contact: tool point within this distance of the panel face."""

    target_open_angle_rad: float = math.radians(50.0)
    hold_ticks: int = 30
    max_step_m: float = 0.015
    contact_approach_max_step_m: float | None = None
    """Optional tighter translation limit for PRE_CONTACT and CONTACT.

    ``None`` preserves the general controller's historical ``max_step_m``.
    Collision-heavy robot presets can slow only the final normal approach
    without lengthening free-space motion or the tangential push.
    """

    approach_max_ticks: int = 300
    align_max_ticks: int = 150
    pre_contact_max_ticks: int = 90
    contact_max_ticks: int = 90
    push_max_ticks: int = 400
    release_max_ticks: int = 150

    def phase_budget(self, phase: DoorPushPhase) -> int:
        budgets = {
            DoorPushPhase.APPROACH: self.approach_max_ticks,
            DoorPushPhase.ALIGN: self.align_max_ticks,
            DoorPushPhase.PRE_CONTACT: self.pre_contact_max_ticks,
            DoorPushPhase.CONTACT: self.contact_max_ticks,
            DoorPushPhase.PUSH: self.push_max_ticks,
            DoorPushPhase.HOLD: self.hold_ticks,
            DoorPushPhase.RELEASE: self.release_max_ticks,
        }
        return budgets[phase]

    @property
    def push_point_y_m(self) -> float:
        return self.push_radius_frac * self.panel_width_m

    def surface_x_m(self, clearance_m: float) -> float:
        """Panel-frame x of the Alex V2 tool point off the +X face."""
        return self.panel_thickness_m + clearance_m


@dataclass(frozen=True)
class DoorPushObservation:
    """Per-tick controller input, already reduced to door-relative quantities."""

    door_frame: ObjectFrame
    hinge_angle_rad: float
    hinge_velocity_rad_s: float
    ee_pos_w: np.ndarray  # (3,)
    contact_sensed: bool | None = None
    """Force-sensed contact flag from the env's contact sensor, when available.
    ``None`` (envs without force sensing) falls back to geometric inference for
    the CONTACT-phase transition."""


@dataclass(frozen=True)
class DoorPushCommand:
    """Per-tick controller output."""

    delta_door_frame: np.ndarray  # (EE_DELTA_DIM,) — the A3 step action
    phase: DoorPushPhase
    done: bool
    timed_out: bool
    contact_inferred: bool
    target_door_frame: np.ndarray  # (3,) — waypoint the delta tracks, door frame


@dataclass(frozen=True)
class DoorPushVariation:
    """Seeded, bounded variation for randomized rollouts (off by default)."""

    start_offset_door_frame: tuple[float, float, float]
    push_radius_frac: float
    push_height_m: float

    def apply(self, cfg: DoorPushControllerCfg) -> DoorPushControllerCfg:
        return replace(
            cfg, push_radius_frac=self.push_radius_frac, push_height_m=self.push_height_m
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "start_offset_door_frame": list(self.start_offset_door_frame),
            "push_radius_frac": self.push_radius_frac,
            "push_height_m": self.push_height_m,
        }


@dataclass(frozen=True)
class VariationBounds:
    """Sampling ranges for :func:`sample_variation`.

    Defaults preserve the frozen task ranges and RNG draw order; calibrated
    robot-specific presets may narrow them.
    """

    start_offset_low: tuple[float, float, float] = (-0.05, -0.15, -0.10)
    start_offset_high: tuple[float, float, float] = (0.15, 0.15, 0.10)
    push_radius_frac_range: tuple[float, float] = (0.70, 0.90)
    push_height_m_range: tuple[float, float] = (-0.45, -0.15)


def sample_variation(
    rng: np.random.Generator, bounds: VariationBounds | None = None
) -> DoorPushVariation:
    """Draw one bounded variation; deterministic given the generator state.

    Default bounds keep push heights below the handle band (see
    ``DoorPushControllerCfg``); pass ``bounds`` for robot-specific ranges.
    """
    bounds = bounds or VariationBounds()
    offset = rng.uniform(low=bounds.start_offset_low, high=bounds.start_offset_high)
    return DoorPushVariation(
        start_offset_door_frame=tuple(float(v) for v in offset),
        push_radius_frac=float(rng.uniform(*bounds.push_radius_frac_range)),
        push_height_m=float(rng.uniform(*bounds.push_height_m_range)),
    )

@dataclass
class _FsmState:
    phase: DoorPushPhase = DoorPushPhase.APPROACH
    ticks_in_phase: int = 0
    timed_out: bool = False
    angle_at_phase_entry: float = 0.0


class DoorPushController:
    """Deterministic FSM producing door-frame EE deltas + an A4 chunk log."""

    def __init__(self, cfg: DoorPushControllerCfg | None = None):
        self.cfg = cfg or DoorPushControllerCfg()
        self._state = _FsmState()
        self._chunk_log = ChunkLog()
        self._open_chunk_start_tick = 0
        self._total_ticks = 0

    @property
    def phase(self) -> DoorPushPhase:
        return self._state.phase

    @property
    def chunk_log(self) -> ChunkLog:
        return self._chunk_log

    def reset(self) -> None:
        self._state = _FsmState()
        self._chunk_log = ChunkLog()
        self._open_chunk_start_tick = 0
        self._total_ticks = 0

    def act(self, obs: DoorPushObservation) -> DoorPushCommand:
        cfg = self.cfg
        state = self._state

        ee_door = obs.door_frame.point_from_world(obs.ee_pos_w)
        contact_geometric = self._contact_inferred(ee_door, obs.hinge_angle_rad)
        # Force-sensed contact drives the FSM when available; the geometric
        # inference is always recorded (DoorPushCommand.contact_inferred).
        contact = contact_geometric if obs.contact_sensed is None else obs.contact_sensed

        if state.phase is DoorPushPhase.DONE or state.timed_out:
            return self._command(np.zeros(3), ee_door, contact_geometric)

        target_door = self._phase_target_door(state.phase, obs.hinge_angle_rad)
        transition = self._phase_complete(state.phase, ee_door, target_door, obs, contact)
        if transition:
            self._advance_phase(obs)
            if self._state.phase is DoorPushPhase.DONE:
                return self._command(np.zeros(3), ee_door, contact_geometric)
            target_door = self._phase_target_door(self._state.phase, obs.hinge_angle_rad)

        state = self._state
        state.ticks_in_phase += 1
        self._total_ticks += 1
        if state.ticks_in_phase > cfg.phase_budget(state.phase):
            state.timed_out = True
            self._close_open_chunk()
            return self._command(np.zeros(3), ee_door, contact_geometric)

        if state.phase is DoorPushPhase.HOLD:
            step = np.zeros(3)
        else:
            error = target_door - ee_door
            distance = float(np.linalg.norm(error))
            max_step_m = cfg.max_step_m
            if (
                state.phase in (DoorPushPhase.PRE_CONTACT, DoorPushPhase.CONTACT)
                and cfg.contact_approach_max_step_m is not None
            ):
                max_step_m = cfg.contact_approach_max_step_m
            if distance > max_step_m:
                step = error * (max_step_m / distance)
            else:
                step = error
        return self._command(step, target_door, contact_geometric)

    def finalize(self) -> ChunkLog:
        """Close the open A4 chunk (episode end) and return the chunk log."""
        self._close_open_chunk()
        return self._chunk_log

    # -- FSM internals ---------------------------------------------------------

    def _phase_target_door(self, phase: DoorPushPhase, hinge_angle_rad: float) -> np.ndarray:
        cfg = self.cfg
        clearance = {
            DoorPushPhase.APPROACH: cfg.approach_standoff_m,
            DoorPushPhase.ALIGN: cfg.align_standoff_m,
            DoorPushPhase.PRE_CONTACT: cfg.pre_contact_clearance_m,
            DoorPushPhase.CONTACT: cfg.contact_clearance_m,
            DoorPushPhase.PUSH: cfg.contact_clearance_m,
            DoorPushPhase.HOLD: cfg.contact_clearance_m,
            DoorPushPhase.RELEASE: cfg.release_standoff_m,
        }[phase]
        point_panel = np.array(
            [cfg.surface_x_m(clearance), cfg.push_point_y_m, cfg.push_height_m]
        )
        # Panel-frame waypoints rotate with the hinge angle into the door frame.
        return rot_z(hinge_angle_rad) @ point_panel

    def _phase_complete(
        self,
        phase: DoorPushPhase,
        ee_door: np.ndarray,
        target_door: np.ndarray,
        obs: DoorPushObservation,
        contact: bool,
    ) -> bool:
        cfg = self.cfg
        distance = float(np.linalg.norm(target_door - ee_door))
        if phase is DoorPushPhase.APPROACH:
            return distance <= cfg.approach_tol_m
        if phase is DoorPushPhase.ALIGN:
            return distance <= cfg.align_tol_m
        if phase is DoorPushPhase.PRE_CONTACT:
            # A force-sensed touch short-circuits the distance check: the hand
            # has physically arrived even if the geometric model disagrees
            # (e.g. a lagging arm meeting a moving panel). Sensed-only, so
            # envs without force sensing (contact_sensed=None) are unchanged.
            return distance <= cfg.pre_contact_tol_m or obs.contact_sensed is True
        if phase is DoorPushPhase.CONTACT:
            return contact
        if phase is DoorPushPhase.PUSH:
            return obs.hinge_angle_rad >= cfg.target_open_angle_rad
        if phase is DoorPushPhase.HOLD:
            return self._state.ticks_in_phase >= cfg.hold_ticks
        if phase is DoorPushPhase.RELEASE:
            return distance <= cfg.approach_tol_m
        return False

    def _advance_phase(self, obs: DoorPushObservation) -> None:
        self._close_open_chunk()
        state = self._state
        if state.phase is DoorPushPhase.RELEASE:
            state.phase = DoorPushPhase.DONE
            return
        state.phase = PHASE_ORDER[PHASE_ORDER.index(state.phase) + 1]
        state.ticks_in_phase = 0
        state.angle_at_phase_entry = obs.hinge_angle_rad
        self._open_chunk_start_tick = self._total_ticks

    def _contact_inferred(self, ee_door: np.ndarray, hinge_angle_rad: float) -> bool:
        cfg = self.cfg
        ee_panel = rot_z(hinge_angle_rad).T @ ee_door
        on_face = ee_panel[0] <= cfg.surface_x_m(cfg.contact_eps_m)
        half_height = cfg.panel_height_m / 2.0
        within_panel = (
            0.0 <= ee_panel[1] <= cfg.panel_width_m
            and -half_height <= ee_panel[2] <= half_height
            and ee_panel[0] >= 0.0
        )
        return bool(on_face and within_panel)

    def _close_open_chunk(self) -> None:
        state = self._state
        if state.phase is DoorPushPhase.DONE:
            return
        duration = self._total_ticks - self._open_chunk_start_tick
        if duration <= 0:
            return
        cfg = self.cfg
        motion = 0.0
        if state.phase is DoorPushPhase.PUSH:
            motion = cfg.target_open_angle_rad - state.angle_at_phase_entry
        self._chunk_log.chunks.append(
            ObjectCentricChunk(
                phase=str(state.phase),
                contact_target_panel=(
                    cfg.surface_x_m(0.0),
                    cfg.push_point_y_m,
                    cfg.push_height_m,
                ),
                motion_hinge_delta_rad=motion,
                duration_ticks=duration,
            )
        )
        self._open_chunk_start_tick = self._total_ticks

    def _command(
        self, step_door: np.ndarray, target_door: np.ndarray, contact: bool
    ) -> DoorPushCommand:
        delta = np.zeros(EE_DELTA_DIM)
        delta[:3] = step_door
        state = self._state
        return DoorPushCommand(
            delta_door_frame=delta,
            phase=state.phase,
            done=state.phase is DoorPushPhase.DONE,
            timed_out=state.timed_out,
            contact_inferred=contact,
            target_door_frame=np.asarray(target_door, dtype=np.float64).reshape(3),
        )


__all__ = [
    "PHASE_ORDER",
    "DoorPushCommand",
    "DoorPushController",
    "DoorPushControllerCfg",
    "DoorPushObservation",
    "DoorPushPhase",
    "DoorPushVariation",
    "VariationBounds",
    "sample_variation",
]
