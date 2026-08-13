"""Pure tests for the scripted door-push FSM against synthetic door kinematics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame, frame_delta_to_world, rot_z
from alexdoor_xas.policies.scripted.door_push import (
    PHASE_ORDER,
    DoorPushController,
    DoorPushControllerCfg,
    DoorPushObservation,
    DoorPushPhase,
    sample_variation,
)
from alexdoor_xas.policies.scripted.door_push_alex_v2 import (
    alex_v2_push_cfg,
    alex_v2_variation_bounds,
)
from conftest import SyntheticDoorWorld


def _v2_calibration():
    return SimpleNamespace(
        controller={
            "push_radius_frac": 0.35,
            "push_height_m": 0.15,
            "approach_standoff_m": 0.12,
            "align_standoff_m": 0.10,
            "pre_contact_clearance_m": 0.01,
            "contact_clearance_m": -0.005,
            "contact_approach_max_step_m": 0.005,
            "release_standoff_m": 0.30,
            "contact_force_threshold_n": 2.5,
        },
        randomization_bounds={
            "start_offset_low": (-0.04, -0.06, -0.05),
            "start_offset_high": (0.06, 0.06, 0.05),
            "push_radius_frac_range": (0.32, 0.40),
            "push_height_m_range": (0.05, 0.18),
        },
    )


def _observe(world: SyntheticDoorWorld) -> DoorPushObservation:
    ee_door = world.door_frame.point_from_world(world.ee_pos_w)
    ee_panel = rot_z(world.angle).T @ ee_door
    half_height = world.cfg.panel_height_m / 2.0
    contact_sensed = bool(
        0.0 <= ee_panel[0] <= world.cfg.surface_x_m(world.cfg.contact_eps_m)
        and 0.0 <= ee_panel[1] <= world.cfg.panel_width_m
        and -half_height <= ee_panel[2] <= half_height
    )
    return DoorPushObservation(
        door_frame=world.door_frame,
        hinge_angle_rad=world.angle,
        ee_pos_w=world.ee_pos_w.copy(),
        contact_sensed=contact_sensed,
    )


def _run_episode(
    door_frame: ObjectFrame,
    cfg: DoorPushControllerCfg | None = None,
    start_door: np.ndarray | None = None,
    max_ticks: int = 2000,
) -> tuple[DoorPushController, SyntheticDoorWorld, list[str], list[np.ndarray]]:
    cfg = cfg or DoorPushControllerCfg()
    controller = DoorPushController(cfg)
    world = SyntheticDoorWorld(door_frame=door_frame, cfg=cfg)
    start = start_door if start_door is not None else np.array([0.7, 0.2, 0.0])
    world.ee_pos_w = door_frame.point_to_world(start)

    phases: list[str] = []
    deltas: list[np.ndarray] = []
    for _ in range(max_ticks):
        command = controller.act(_observe(world))
        phases.append(str(command.phase))
        deltas.append(command.delta_door_frame.copy())
        if command.done or command.timed_out:
            break
        world.apply_world(frame_delta_to_world(command.delta_door_frame, door_frame)[:3])
    return controller, world, phases, deltas


def test_fixed_start_visits_all_phases_in_order_and_opens_door() -> None:
    controller, world, phases, _ = _run_episode(ObjectFrame(origin=np.zeros(3), rot=np.eye(3)))

    seen = list(dict.fromkeys(phases))  # ordered unique phases
    expected = [str(phase) for phase in PHASE_ORDER] + [str(DoorPushPhase.DONE)]
    assert seen == expected
    assert world.angle >= controller.cfg.target_open_angle_rad
    assert phases[-1] == str(DoorPushPhase.DONE)


def test_controller_is_deterministic() -> None:
    frame = ObjectFrame(origin=np.array([1.0, -2.0, 0.0]), rot=rot_z(0.3))
    _, _, phases_a, deltas_a = _run_episode(frame)
    _, _, phases_b, deltas_b = _run_episode(frame)
    assert phases_a == phases_b
    np.testing.assert_array_equal(np.stack(deltas_a), np.stack(deltas_b))


def test_door_frame_deltas_are_invariant_to_door_placement() -> None:
    frame_a = ObjectFrame(origin=np.zeros(3), rot=np.eye(3))
    frame_b = ObjectFrame(origin=np.array([3.0, -5.0, 0.0]), rot=rot_z(2.1))
    _, _, phases_a, deltas_a = _run_episode(frame_a)
    _, _, phases_b, deltas_b = _run_episode(frame_b)

    assert phases_a == phases_b
    np.testing.assert_allclose(np.stack(deltas_a), np.stack(deltas_b), atol=1e-9)


def test_step_magnitude_never_exceeds_max_step() -> None:
    _, _, _, deltas = _run_episode(ObjectFrame(origin=np.zeros(3), rot=np.eye(3)))
    cfg = DoorPushControllerCfg()
    step_norms = np.linalg.norm(np.stack(deltas)[:, :3], axis=-1)
    assert float(step_norms.max()) <= cfg.max_step_m + 1e-12


@pytest.mark.parametrize("phase", (DoorPushPhase.PRE_CONTACT, DoorPushPhase.CONTACT))
def test_contact_approach_uses_its_dedicated_step_limit(phase) -> None:
    cfg = DoorPushControllerCfg(max_step_m=0.015, contact_approach_max_step_m=0.005)
    frame = ObjectFrame(origin=np.zeros(3), rot=np.eye(3))
    controller = DoorPushController(cfg)
    controller._state.phase = phase
    obs = DoorPushObservation(
        door_frame=frame,
        hinge_angle_rad=0.0,
        ee_pos_w=np.array([0.25, cfg.push_point_y_m, cfg.push_height_m]),
        contact_sensed=False,
    )

    command = controller.act(obs)

    assert command.phase is phase
    assert np.linalg.norm(command.delta_door_frame[:3]) == pytest.approx(0.005)


def test_alex_v2_preset_keeps_alignment_outside_contact_and_slows_final_approach() -> None:
    cfg = alex_v2_push_cfg(_v2_calibration())

    assert cfg.align_standoff_m == pytest.approx(0.10)
    assert cfg.contact_approach_max_step_m == pytest.approx(0.005)
    assert cfg.max_step_m == pytest.approx(0.015)


def test_chunk_log_covers_all_phases_with_positive_durations() -> None:
    controller, _, _, _ = _run_episode(ObjectFrame(origin=np.zeros(3), rot=np.eye(3)))
    chunks = controller.finalize()
    chunk_phases = [chunk.phase for chunk in chunks]
    assert chunk_phases == [str(phase) for phase in PHASE_ORDER]
    assert all(chunk.duration_ticks > 0 for chunk in chunks)
    push_chunk = chunks[chunk_phases.index(str(DoorPushPhase.PUSH))]
    assert push_chunk.motion_hinge_delta_rad > 0.0
    assert all(chunk.motion_hinge_delta_rad == 0.0 for chunk in chunks if chunk is not push_chunk)


def test_phase_timeout_freezes_controller() -> None:
    # An unreachable approach budget forces a timeout in the first phase.
    cfg = DoorPushControllerCfg(approach_max_ticks=3)
    controller, _, phases, deltas = _run_episode(
        ObjectFrame(origin=np.zeros(3), rot=np.eye(3)), cfg=cfg
    )
    assert phases[-1] == str(DoorPushPhase.APPROACH)
    assert controller.phase is DoorPushPhase.APPROACH
    np.testing.assert_array_equal(deltas[-1], np.zeros(6))


def test_alex_v2_preset_visits_all_phases_and_opens_door() -> None:
    cfg = alex_v2_push_cfg(_v2_calibration())
    controller, world, phases, _ = _run_episode(
        ObjectFrame(origin=np.zeros(3), rot=np.eye(3)),
        cfg=cfg,
        start_door=np.array([0.35, 0.29, 0.15]),
    )
    seen = list(dict.fromkeys(phases))
    expected = [str(phase) for phase in PHASE_ORDER] + [str(DoorPushPhase.DONE)]
    assert seen == expected
    assert world.angle >= cfg.target_open_angle_rad
    assert controller.phase is DoorPushPhase.DONE


def test_contact_sensed_overrides_geometric_inference() -> None:
    cfg = DoorPushControllerCfg()
    frame = ObjectFrame(origin=np.zeros(3), rot=np.eye(3))
    controller = DoorPushController(cfg)
    world = SyntheticDoorWorld(door_frame=frame, cfg=cfg)
    world.ee_pos_w = frame.point_to_world(np.array([0.7, 0.2, 0.0]))

    for _ in range(2000):
        command = controller.act(_observe(world))
        if command.phase is DoorPushPhase.CONTACT:
            break
        world.apply_world(frame_delta_to_world(command.delta_door_frame, frame)[:3])
    assert controller.phase is DoorPushPhase.CONTACT

    away = frame.point_to_world(np.array([cfg.surface_x_m(0.05), cfg.push_point_y_m, 0.0]))
    obs = DoorPushObservation(
        door_frame=frame,
        hinge_angle_rad=world.angle,
        ee_pos_w=away,
        contact_sensed=True,
    )
    command = controller.act(obs)
    assert command.phase is DoorPushPhase.PUSH
    assert command.contact_inferred is False

    controller_b = DoorPushController(cfg)
    controller_b._state.phase = DoorPushPhase.CONTACT
    on_face = frame.point_to_world(np.array([cfg.surface_x_m(0.0), cfg.push_point_y_m, 0.0]))
    obs_b = DoorPushObservation(
        door_frame=frame,
        hinge_angle_rad=0.0,
        ee_pos_w=on_face,
        contact_sensed=False,
    )
    command_b = controller_b.act(obs_b)
    assert command_b.phase is DoorPushPhase.CONTACT
    assert command_b.contact_inferred is True


def test_alex_v2_variation_bounds_are_respected() -> None:
    bounds = alex_v2_variation_bounds(_v2_calibration())
    for seed in range(20):
        variation = sample_variation(np.random.default_rng(seed), bounds)
        low, high = bounds.push_radius_frac_range
        assert low <= variation.push_radius_frac <= high
        low, high = bounds.push_height_m_range
        assert low <= variation.push_height_m <= high
        offset = np.asarray(variation.start_offset_door_frame)
        assert np.all(offset >= bounds.start_offset_low)
        assert np.all(offset <= bounds.start_offset_high)


def test_sample_variation_is_seeded_and_bounded() -> None:
    variation_a = sample_variation(np.random.default_rng(11))
    variation_b = sample_variation(np.random.default_rng(11))
    assert variation_a == variation_b

    for seed in range(20):
        variation = sample_variation(np.random.default_rng(seed))
        assert 0.70 <= variation.push_radius_frac <= 0.90
        assert -0.45 <= variation.push_height_m <= -0.15
        offset = np.asarray(variation.start_offset_door_frame)
        assert np.all(offset >= (-0.05, -0.15, -0.10)) and np.all(offset <= (0.15, 0.15, 0.10))


def test_variation_apply_overrides_push_geometry() -> None:
    variation = sample_variation(np.random.default_rng(3))
    cfg = variation.apply(DoorPushControllerCfg())
    assert cfg.push_radius_frac == variation.push_radius_frac
    assert cfg.push_height_m == variation.push_height_m

    controller, world, phases, _ = _run_episode(
        ObjectFrame(origin=np.zeros(3), rot=np.eye(3)),
        cfg=cfg,
        start_door=np.array([0.7, 0.2, 0.0]) + np.asarray(variation.start_offset_door_frame),
    )
    assert phases[-1] == str(DoorPushPhase.DONE)
    assert world.angle >= controller.cfg.target_open_angle_rad
