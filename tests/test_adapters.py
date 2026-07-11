"""Adapter-v1 unit tests (Phase 3.1): pure Python, no Kit, synthetic worlds."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alexdoor_xas.action.frames import ObjectFrame, frame_delta_to_world, rot_z
from alexdoor_xas.action.spaces import A4_PHASE_VOCAB, EE_DELTA_DIM, ObjectCentricChunk
from alexdoor_xas.adapters import (
    ALEX_V2_ROBOT_TAG,
    MAX_HINGE_ANGLE_RAD,
    PROXY_LIMITS,
    A2Adapter,
    A3Adapter,
    A4Adapter,
    A4AdapterCfg,
    AdapterStatus,
    DoorPanelGeometry,
    RobotLimitsCfg,
    StepContext,
    WorkspaceSphere,
    alex_v2_limits,
    limits_for_robot,
    replay_source,
    rollout_chunks,
    validate_object_frame,
)
from alexdoor_xas.data_engine import DataEngineCfg, plan_episodes, run_episode
from alexdoor_xas.policies.scripted import DoorPushController, DoorPushControllerCfg
from alexdoor_xas.policies.scripted.door_push import PHASE_ORDER, DoorPushPhase
from conftest import FakeDoorPushEnv, FakeForceDoorPushEnv


def _ctx(
    ee_pos_w=(0.0, 0.0, 0.0),
    door_frame: ObjectFrame | None = None,
    hinge_angle: float = 0.0,
    **kwargs,
) -> StepContext:
    return StepContext(
        door_frame=door_frame,
        hinge_angle_rad=hinge_angle,
        hinge_velocity_rad_s=0.0,
        ee_pos_w=np.asarray(ee_pos_w, dtype=np.float64),
        **kwargs,
    )


def _identity_frame(origin=(0.0, 0.0, 0.0)) -> ObjectFrame:
    return ObjectFrame(origin=np.asarray(origin, dtype=np.float64), rot=np.eye(3))


def _v2_limits(center=(1.0, 2.0, 3.0), reach_shell=(0.2, 0.8)):
    calibration = SimpleNamespace(reach_shell_m=reach_shell)
    return alex_v2_limits(calibration, workspace_center_w=center)


def _chunk(
    phase: str = "push",
    target=(0.086, 0.55, -0.30),
    hinge_delta: float = 0.0,
    duration: int = 100,
) -> ObjectCentricChunk:
    return ObjectCentricChunk(
        phase=phase,
        contact_target_panel=tuple(float(v) for v in target),
        motion_hinge_delta_rad=hinge_delta,
        duration_ticks=duration,
    )


# -- A2 -------------------------------------------------------------------------


def test_a2_accepts_in_range_delta():
    adapter = A2Adapter(PROXY_LIMITS)
    delta = np.array([0.01, -0.005, 0.0, 0.0, 0.0, 0.0])
    applied, decision = adapter.process(delta, _ctx())
    assert decision.status is AdapterStatus.ACCEPTED
    assert decision.reason == ""
    np.testing.assert_allclose(applied, delta)
    assert adapter.log.n_accepted == 1 and adapter.log.n_rejected == 0


def test_a2_clamps_and_logs_correction():
    adapter = A2Adapter(PROXY_LIMITS)
    delta = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.2])
    applied, decision = adapter.process(delta, _ctx())
    assert decision.status is AdapterStatus.CORRECTED
    assert "clamp" in decision.reason
    np.testing.assert_allclose(applied[:3], [PROXY_LIMITS.max_pos_delta_m, 0.0, 0.0])
    assert applied[5] == pytest.approx(PROXY_LIMITS.max_rot_delta_rad)
    np.testing.assert_allclose(decision.requested, delta)
    np.testing.assert_allclose(decision.applied, applied)


@pytest.mark.parametrize(
    "bad", [np.full(6, np.nan), np.zeros(3), np.zeros((2, 6)).reshape(-1)]
)
def test_a2_rejects_malformed_deltas(bad):
    adapter = A2Adapter(PROXY_LIMITS)
    applied, decision = adapter.process(bad, _ctx())
    assert decision.status is AdapterStatus.REJECTED
    assert decision.reason
    np.testing.assert_allclose(applied, np.zeros(EE_DELTA_DIM))


def test_a2_rejects_out_of_workspace_command():
    limits = _v2_limits()
    adapter = A2Adapter(limits)
    center = np.asarray(limits.workspace.center_w)
    ee = center + np.array([limits.workspace.max_reach_m + 0.01, 0.0, 0.0])
    applied, decision = adapter.process(
        np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), _ctx(ee_pos_w=ee)
    )
    assert decision.status is AdapterStatus.REJECTED
    assert "beyond max reach" in decision.reason
    assert decision.checks["reachable"] is False
    np.testing.assert_allclose(applied, np.zeros(EE_DELTA_DIM))


def test_a2_warns_near_min_reach_but_accepts():
    limits = _v2_limits(reach_shell=(0.24, 0.8))
    adapter = A2Adapter(limits)
    center = np.asarray(limits.workspace.center_w)
    ee = center + np.array([0.22, 0.0, 0.0])
    applied, decision = adapter.process(np.zeros(6), _ctx(ee_pos_w=ee))
    assert decision.status is AdapterStatus.ACCEPTED
    assert any("min reach" in warning for warning in decision.warnings)


def test_a2_flags_joint_limit_excess_as_warning():
    adapter = A2Adapter(PROXY_LIMITS)
    n = 4
    joint_state = {
        "joint_pos": np.zeros(n),
        "joint_vel": np.array([0.0, 0.0, 0.0, 12.0]),
        "joint_pos_target": np.array([0.0, 2.55, 0.0, 0.0]),
    }
    joint_limits = {
        "joint_pos_limits": np.stack([np.full(n, -2.5), np.full(n, 2.5)], axis=1),
        "joint_vel_limits": np.full(n, 10.0),
    }
    _, decision = adapter.process(
        np.zeros(6), _ctx(joint_state=joint_state, joint_limits=joint_limits)
    )
    assert decision.status is AdapterStatus.ACCEPTED
    assert any("position limit" in warning for warning in decision.warnings)
    assert any("velocity exceeds" in warning for warning in decision.warnings)


def test_a2_chunk_is_cut_at_first_rejection():
    workspace = WorkspaceSphere(center_w=(0.0, 0.0, 0.0), min_reach_m=0.0, max_reach_m=0.05)
    limits = RobotLimitsCfg(robot="test", workspace=workspace, reach_margin_m=0.0)
    adapter = A2Adapter(limits)
    chunk = np.tile(np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), (5, 1))
    applied, decisions = adapter.process_chunk(chunk, _ctx())
    statuses = [d.status for d in decisions]
    assert AdapterStatus.REJECTED in statuses
    first_reject = statuses.index(AdapterStatus.REJECTED)
    assert first_reject == 2  # cumulative 0.02*3 = 0.06 > 0.05 at the 3rd step
    np.testing.assert_allclose(applied[first_reject:], 0.0)
    assert all(s is AdapterStatus.REJECTED for s in statuses[first_reject:])


def test_limits_for_robot_rejects_unknown_tag():
    assert limits_for_robot("proxy_ee_sphere_v0") is PROXY_LIMITS
    with pytest.raises(ValueError, match="workspace_center_w"):
        limits_for_robot(ALEX_V2_ROBOT_TAG)
    with pytest.raises(KeyError, match="no adapter limits"):
        limits_for_robot("robot_from_the_future_v9")


def test_alex_v2_limits_use_calibrated_shell_and_caller_center() -> None:
    calibration = SimpleNamespace(reach_shell_m=(0.31, 0.77))
    center = (4.0, -2.0, 1.25)

    limits = limits_for_robot(
        ALEX_V2_ROBOT_TAG,
        calibration=calibration,
        workspace_center_w=center,
    )

    assert limits.robot == ALEX_V2_ROBOT_TAG
    assert limits.workspace.center_w == center
    assert limits.workspace.min_reach_m == pytest.approx(0.31)
    assert limits.workspace.max_reach_m == pytest.approx(0.77)


@pytest.mark.parametrize("center", [(1.0, 2.0), (1.0, float("nan"), 3.0)])
def test_alex_v2_limits_reject_invalid_caller_center(center) -> None:
    with pytest.raises(ValueError, match="three finite"):
        _v2_limits(center=center)


# -- A3 -------------------------------------------------------------------------


def test_a3_matches_frame_conversion():
    frame = ObjectFrame(origin=np.array([1.0, -2.0, 0.5]), rot=rot_z(0.7))
    adapter = A3Adapter(A2Adapter(PROXY_LIMITS))
    delta_door = np.array([0.01, -0.004, 0.002, 0.0, 0.0, 0.0])
    applied, decision = adapter.process(delta_door, _ctx(door_frame=frame))
    assert decision.status is AdapterStatus.ACCEPTED
    np.testing.assert_allclose(applied, frame_delta_to_world(delta_door, frame))
    assert decision.checks["object_frame_trusted"] is True
    assert decision.checks["door_frame_static_stage_read"] is True
    np.testing.assert_allclose(decision.requested, delta_door)  # A3 view, not world


def test_a3_rejects_missing_or_corrupt_frame():
    adapter = A3Adapter(A2Adapter(PROXY_LIMITS))
    applied, decision = adapter.process(np.zeros(6), _ctx(door_frame=None))
    assert decision.status is AdapterStatus.REJECTED
    assert "unavailable" in decision.reason
    np.testing.assert_allclose(applied, np.zeros(EE_DELTA_DIM))

    skewed = ObjectFrame(origin=np.zeros(3), rot=np.eye(3) + 0.2)
    _, decision = adapter.process(np.zeros(6), _ctx(door_frame=skewed))
    assert decision.status is AdapterStatus.REJECTED
    assert "orthonormal" in decision.reason


def test_validate_object_frame_reasons():
    assert validate_object_frame(None)
    assert validate_object_frame(_identity_frame()) == ""
    bad_origin = ObjectFrame(origin=np.array([np.inf, 0.0, 0.0]), rot=np.eye(3))
    assert "origin" in validate_object_frame(bad_origin)


# -- pins against the scripted controller (adapters must not import policies) ----


def test_panel_geometry_pins_controller_defaults():
    geo = DoorPanelGeometry()
    cfg = DoorPushControllerCfg()
    assert geo.panel_width_m == cfg.panel_width_m
    assert geo.panel_thickness_m == cfg.panel_thickness_m
    assert geo.ee_radius_m == cfg.ee_radius_m
    assert geo.contact_eps_m == cfg.contact_eps_m
    assert geo.surface_x_m(0.01) == cfg.surface_x_m(0.01)


def test_geometric_contact_pins_controller_inference():
    geo = DoorPanelGeometry()
    controller = DoorPushController(DoorPushControllerCfg())
    rng = np.random.default_rng(0)
    for _ in range(200):
        angle = float(rng.uniform(0.0, math.pi / 2))
        ee_door = rng.uniform([-0.2, -0.3, -0.6], [0.4, 1.1, 0.6])
        expected = controller._contact_inferred(ee_door, angle)
        assert geo.geometric_contact(rot_z(angle).T @ ee_door) == expected


def test_a4_phase_vocab_is_shared_with_dataset():
    from alexdoor_xas.dataset import A4_PHASE_VOCAB as dataset_vocab

    assert dataset_vocab is A4_PHASE_VOCAB


def test_a4_phase_vocab_pins_scripted_controller_order():
    assert A4_PHASE_VOCAB == tuple(str(phase) for phase in PHASE_ORDER)
    assert str(DoorPushPhase.DONE) not in A4_PHASE_VOCAB


# -- A4 validation ----------------------------------------------------------------


def _a4(limits=PROXY_LIMITS, cfg: A4AdapterCfg | None = None) -> A4Adapter:
    return A4Adapter(A3Adapter(A2Adapter(limits)), cfg=cfg)


@pytest.mark.parametrize(
    ("chunk", "reason_match"),
    [
        (_chunk(phase="moonwalk"), "unknown A4 phase"),
        (_chunk(target=(0.086, np.nan, -0.3)), "non-finite"),
        (_chunk(duration=0), "duration_ticks"),
        (_chunk(hinge_delta=-0.3), "pulling"),
        (_chunk(target=(0.086, 1.2, -0.3)), "off the"),
        (_chunk(target=(0.086, 0.70, 0.05)), "handle band"),
    ],
)
def test_a4_rejects_invalid_chunks(chunk, reason_match):
    _, decision = _a4().validate_chunk(chunk, entry_angle_rad=0.0)
    assert decision.status is AdapterStatus.REJECTED
    assert reason_match in decision.reason


@pytest.mark.parametrize(
    "phase",
    ["approach", "align", "pre_contact", "contact", "hold", "release"],
)
def test_a4_rejects_non_push_phase_hinge_motion(phase):
    _, decision = _a4().validate_chunk(
        _chunk(phase=phase, hinge_delta=0.1), entry_angle_rad=0.0
    )
    assert decision.status is AdapterStatus.REJECTED
    assert decision.checks["hinge_delta_phase_valid"] is False
    assert "non-push phase cannot request hinge motion" in decision.reason


@pytest.mark.parametrize(
    ("target", "shape_text"),
    [
        ((0.086, 0.55), "(2,)"),
        ((0.086, 0.55, -0.30, 0.0), "(4,)"),
    ],
)
def test_a4_rejects_malformed_contact_target_shape(target, shape_text):
    _, decision = _a4().validate_chunk(_chunk(target=target), entry_angle_rad=0.0)
    assert decision.status is AdapterStatus.REJECTED
    assert decision.checks["target_shape"] is False
    assert "contact_target_panel" in decision.reason
    assert shape_text in decision.reason
    assert decision.applied is None


def test_a4_rejects_non_finite_contact_target_after_shape_check():
    _, decision = _a4().validate_chunk(
        _chunk(target=(0.086, np.inf, -0.30)), entry_angle_rad=0.0
    )
    assert decision.status is AdapterStatus.REJECTED
    assert "non-finite" in decision.reason


def test_a4_corrects_slightly_off_panel_target():
    chunk = _chunk(target=(0.086, 0.84, -0.3))  # 0.01 m past the panel edge
    fixed, decision = _a4().validate_chunk(chunk, entry_angle_rad=0.0)
    assert decision.status is AdapterStatus.CORRECTED
    assert "nudged" in decision.reason
    assert fixed.contact_target_panel[1] == pytest.approx(0.83)


def test_a4_caps_hinge_delta_to_remaining_travel():
    chunk = _chunk(hinge_delta=2.0)
    fixed, decision = _a4().validate_chunk(chunk, entry_angle_rad=0.5)
    assert decision.status is AdapterStatus.CORRECTED
    assert "capped" in decision.reason
    assert fixed.motion_hinge_delta_rad == pytest.approx(MAX_HINGE_ANGLE_RAD - 0.5)


def test_a4_rejects_unreachable_target():
    # Workspace centered 2 m from the door: every probe point is out of reach.
    workspace = WorkspaceSphere(center_w=(2.0, 2.0, 2.0), min_reach_m=0.1, max_reach_m=0.5)
    limits = RobotLimitsCfg(robot="test", workspace=workspace)
    adapter = _a4(limits=limits)
    _, decision = adapter.validate_chunk(
        _chunk(hinge_delta=0.5), entry_angle_rad=0.0, door_frame=_identity_frame()
    )
    assert decision.status is AdapterStatus.REJECTED
    assert "beyond max reach" in decision.reason
    assert decision.checks["reachable"] is False


# -- A4 execution on the synthetic world -------------------------------------------


def _canonical_chunks(push_delta: float = math.radians(50.0)):
    target = (0.086, 0.55, -0.30)
    return [
        _chunk(phase="approach", target=target, duration=120),
        _chunk(phase="align", target=target, duration=60),
        _chunk(phase="pre_contact", target=target, duration=60),
        _chunk(phase="contact", target=target, duration=60),
        _chunk(phase="push", target=target, hinge_delta=push_delta, duration=250),
        _chunk(phase="hold", target=target, duration=30),
        _chunk(phase="release", target=target, duration=60),
    ]


def test_a4_executes_canonical_sequence_on_synthetic_door():
    env = FakeDoorPushEnv(yaw_rad=0.4, origin=(1.0, -0.5, 0.2))
    env.reset(seed=0)
    adapter = _a4()
    push_delta = math.radians(50.0)
    result = adapter.execute(env, _canonical_chunks(push_delta))
    assert result.status is AdapterStatus.ACCEPTED
    assert result.completed and result.failure == ""
    assert result.contact_reached is True
    assert result.requested_hinge_delta_rad == pytest.approx(push_delta)
    assert result.achieved_hinge_delta_rad >= push_delta - 1e-9
    assert result.final_door_angle_change_rad >= push_delta - 1e-9
    assert [s.phase for s in result.stages][:2] == ["approach", "align"]
    assert result.log.n_rejected == 0
    assert any(stage.contact_reached for stage in result.stages)
    json.dumps(result.to_dict())  # the full result must be JSON-serializable


def test_a4_hold_only_without_contact_does_not_report_contact_reached():
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    result = _a4().execute(env, _chunk(phase="hold", duration=3))
    assert result.completed, result.reason
    assert result.contact_reached is False
    assert all(not stage.contact_reached for stage in result.stages)


def test_a4_lone_push_chunk_synthesizes_guarded_prefix():
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    result = _a4().execute(env, _chunk(phase="push", hinge_delta=0.3, duration=250))
    assert result.completed, result.reason
    phases = [s.phase for s in result.stages]
    assert phases == ["approach", "pre_contact", "push"]
    assert result.stages[0].synthesized and result.stages[1].synthesized
    assert result.achieved_hinge_delta_rad >= 0.3 - 1e-9


class _NeverSensesContactEnv(FakeDoorPushEnv):
    """Force-sensing surface that never trips: forces the missed-contact path."""

    def contact_sensed(self):
        return torch.tensor([False])

    def contact_force_w(self):
        return torch.zeros((1, 3), dtype=torch.float64)


def test_a4_reports_missed_contact():
    env = _NeverSensesContactEnv()
    env.reset(seed=0)
    cfg = A4AdapterCfg(min_stage_budget_ticks=200)
    result = _a4(cfg=cfg).execute(env, _chunk(phase="contact", duration=1))
    assert not result.completed
    assert result.failure == "missed_contact"
    assert result.contact_missed is True
    assert result.contact_reached is False


class _StuckDoorEnv(FakeDoorPushEnv):
    """Door that never moves: forces the push-stall (insufficient motion) path."""

    def reset(self, seed=None):
        out = super().reset(seed)
        self.world.gain = 0.0
        return out


def test_a4_reports_push_stall():
    env = _StuckDoorEnv()
    env.reset(seed=0)
    cfg = A4AdapterCfg(push_stall_ticks=15, min_stage_budget_ticks=400)
    result = _a4(cfg=cfg).execute(env, _chunk(phase="push", hinge_delta=0.5, duration=100))
    assert not result.completed
    assert result.failure == "push_stalled"
    assert result.achieved_hinge_delta_rad == pytest.approx(0.0)


def test_a4_rejected_sequence_commands_no_motion():
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    start = env.world.ee_pos_w.copy()
    result = _a4().execute(env, _chunk(hinge_delta=-0.5))  # pull -> reject
    assert result.status is AdapterStatus.REJECTED
    assert result.n_ticks == 0 and not result.stages
    assert "pulling" in result.reason
    assert result.log.n_rejected == 1
    np.testing.assert_allclose(env.world.ee_pos_w, start)
    assert env.world.angle == 0.0


def test_a4_mixed_sequence_rejects_non_push_hinge_motion_without_motion():
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    start = env.world.ee_pos_w.copy()
    chunks = [
        _chunk(phase="approach", duration=20),
        _chunk(phase="hold", hinge_delta=0.2, duration=20),
        _chunk(phase="push", hinge_delta=0.3, duration=20),
    ]
    result = _a4().execute(env, chunks)
    assert result.status is AdapterStatus.REJECTED
    assert not result.completed
    assert result.n_ticks == 0 and not result.stages
    assert result.requested_hinge_delta_rad == pytest.approx(0.5)
    assert result.achieved_hinge_delta_rad == pytest.approx(0.0)
    assert result.contact_reached is False
    assert "non-push phase cannot request hinge motion" in result.reason
    assert result.log.n_rejected == 1
    np.testing.assert_allclose(env.world.ee_pos_w, start)
    assert env.world.angle == 0.0


def test_a4_malformed_target_rejection_commands_no_motion():
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    start = env.world.ee_pos_w.copy()
    result = _a4().execute(env, _chunk(target=(0.086, 0.55)))
    assert result.status is AdapterStatus.REJECTED
    assert not result.completed
    assert result.n_ticks == 0 and not result.stages
    assert "contact_target_panel" in result.reason
    assert result.log.n_rejected == 1
    np.testing.assert_allclose(env.world.ee_pos_w, start)
    assert env.world.angle == 0.0


def test_a4_non_numeric_hinge_delta_rejection_commands_no_motion():
    env = FakeDoorPushEnv()
    env.reset(seed=0)
    start = env.world.ee_pos_w.copy()
    result = _a4().execute(env, _chunk(hinge_delta="bad"))
    assert result.status is AdapterStatus.REJECTED
    assert not result.completed
    assert result.n_ticks == 0 and not result.stages
    assert result.requested_hinge_delta_rad == pytest.approx(0.0)
    assert result.achieved_hinge_delta_rad == pytest.approx(0.0)
    assert "non-numeric" in result.reason
    assert result.log.n_rejected == 1
    np.testing.assert_allclose(env.world.ee_pos_w, start)
    assert env.world.angle == 0.0


# -- replay equivalence through the rollout driver ----------------------------------


def test_a2_replay_reproduces_scripted_episode():
    item = plan_episodes(1, 0, base_seed=3)[0]
    env = FakeDoorPushEnv(yaw_rad=0.3, origin=(0.5, 0.5, 0.0))
    episode = run_episode(env, item, DataEngineCfg())
    assert episode.outcome.success

    replay_env = FakeDoorPushEnv(yaw_rad=0.3, origin=(0.5, 0.5, 0.0))
    replay_env.reset(seed=item.seed)
    actions = [step.action for step in episode.steps]
    result = rollout_chunks(replay_env, replay_source(actions), A2Adapter(PROXY_LIMITS))
    assert result.n_ticks == episode.n_steps
    assert result.log.n_rejected == 0
    assert result.final_angle_rad == pytest.approx(episode.outcome.final_door_angle, abs=1e-9)


def test_a3_replay_matches_a2_replay():
    item = plan_episodes(1, 0, base_seed=3)[0]
    env = FakeForceDoorPushEnv()
    episode = run_episode(env, item, DataEngineCfg())
    actions_door = np.asarray(episode.extras["action_door_frame"])

    replay_env = FakeForceDoorPushEnv()
    replay_env.reset(seed=item.seed)
    adapter = A3Adapter(A2Adapter(PROXY_LIMITS))
    result = rollout_chunks(replay_env, replay_source(actions_door), adapter)
    assert result.n_ticks == episode.n_steps
    assert result.final_angle_rad == pytest.approx(episode.outcome.final_door_angle, abs=1e-9)
