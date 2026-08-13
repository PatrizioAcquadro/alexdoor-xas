"""Deterministic scripted episode generation for the Alex V2 benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from alexdoor_xas.action.frames import ObjectFrame, door_frame_from_body_pose, frame_delta_to_world
from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.assets.door_task import DEFAULT_DOOR_POSE_ID, canonical_door_pose
from alexdoor_xas.policies.scripted import (
    DoorPushController,
    DoorPushControllerCfg,
    DoorPushObservation,
    DoorPushVariation,
    VariationBounds,
    sample_variation,
)
from alexdoor_xas.recording import EpisodeBuffer, EpisodeMeta, EpisodeOutcome, EpisodeStep

CONTACT_SOURCE_FORCE = "force_sensor+geometric"
DEFAULT_SUCCESS_ANGLE_RAD = math.pi / 4.0
DEFAULT_MAX_TICKS = 600


@dataclass(frozen=True)
class DataEngineCfg:
    """Engine-level settings shared by all episodes of one run."""

    task: str
    robot: str
    limitations: tuple[str, ...]
    success_angle_rad: float = DEFAULT_SUCCESS_ANGLE_RAD
    max_ticks: int = DEFAULT_MAX_TICKS
    door_pose_id: str = DEFAULT_DOOR_POSE_ID

    def __post_init__(self) -> None:
        canonical_door_pose(self.door_pose_id)

    @property
    def scene(self) -> str:
        return f"outputs/door_scene/{self.door_pose_id}.usda"

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        pose = canonical_door_pose(self.door_pose_id)
        values["door_yaw_rad"] = pose.yaw_rad
        values["door_offset_xy"] = list(pose.xy_offset_m)
        return values


@dataclass(frozen=True)
class EpisodePlanItem:
    """One planned episode: a seed and (for randomized rollouts) a variation."""

    seed: int
    variation: DoorPushVariation | None = None


def plan_episodes(
    n_fixed: int,
    n_randomized: int,
    base_seed: int,
    bounds: VariationBounds | None = None,
) -> list[EpisodePlanItem]:
    """Fixed-start episodes first, then seeded bounded variations."""
    items = [EpisodePlanItem(seed=base_seed + i) for i in range(n_fixed)]
    for i in range(n_randomized):
        seed = base_seed + n_fixed + i
        items.append(
            EpisodePlanItem(
                seed=seed, variation=sample_variation(np.random.default_rng(seed), bounds)
            )
        )
    return items


def plan_randomized_seeds(
    seeds: list[int] | tuple[int, ...],
    bounds: VariationBounds | None = None,
) -> list[EpisodePlanItem]:
    """Build randomized plan items from an explicit deterministic seed list."""
    normalized = [int(seed) for seed in seeds]
    if len(normalized) != len(set(normalized)):
        raise ValueError("explicit randomized seed plan contains duplicates")
    if any(seed < 0 for seed in normalized):
        raise ValueError("explicit randomized seeds must be non-negative")
    return [
        EpisodePlanItem(
            seed=seed,
            variation=sample_variation(np.random.default_rng(seed), bounds),
        )
        for seed in normalized
    ]


@dataclass(frozen=True)
class _EpisodeSetup:
    buffer: EpisodeBuffer
    controller: DoorPushController
    controller_cfg: DoorPushControllerCfg
    door_frame: ObjectFrame
    door_pose_obs: dict[str, float]
    control_dt: float
    settle_report: dict | None


@dataclass(frozen=True)
class _TickSnapshot:
    angle: float
    velocity: float
    ee_pos_w: np.ndarray
    ee_quat_w: np.ndarray
    contact_sensed: bool
    contact_force_n: float


@dataclass(frozen=True)
class _EnvStepOutcome:
    termination_reason: str = ""
    notes: str = ""
    environment_terminated: bool = False
    environment_truncated: bool = False


@dataclass
class _EpisodeRuntime:
    actions_door_frame: list[np.ndarray] = field(default_factory=list)
    notes: str = ""
    termination_reason: str = "tick_budget"
    environment_terminated: bool = False
    environment_truncated: bool = False
    final_angle: float = float("nan")
    last_command: Any = None


def run_episode(
    env,
    item: EpisodePlanItem,
    engine_cfg: DataEngineCfg,
    controller_cfg: DoorPushControllerCfg | None = None,
    render_hook=None,
) -> EpisodeBuffer:
    """Roll out and record one episode; deterministic given (env state, item)."""
    setup = _prepare_episode(env, item, engine_cfg, controller_cfg)
    runtime = _record_episode_ticks(env, engine_cfg, setup, render_hook)
    _assert_no_silent_episode_reset(env, item, setup.buffer, runtime)
    _record_terminal_contact(env, setup, runtime)
    _finalize_episode(env, item, engine_cfg, setup, runtime)
    return setup.buffer


def _prepare_episode(
    env,
    item: EpisodePlanItem,
    engine_cfg: DataEngineCfg,
    controller_cfg: DoorPushControllerCfg | None,
) -> _EpisodeSetup:
    base_controller_cfg = controller_cfg or DoorPushControllerCfg()
    robot_asset = env.robot_asset_provenance()
    env.reset(seed=item.seed)
    door_frame = _read_door_frame(env)

    active_cfg = base_controller_cfg
    settle_report = env.start_pose_settle_report()
    if item.variation is not None:
        active_cfg = item.variation.apply(base_controller_cfg)
        settle_report = apply_start_offset(env, door_frame, item.variation)

    sim_dt = float(env.cfg.sim.dt)
    control_dt = sim_dt * int(env.cfg.decimation)
    meta = EpisodeMeta.create(
        task=engine_cfg.task,
        action_space=A2_EE_DELTA,
        robot=engine_cfg.robot,
        scene=engine_cfg.scene,
        policy="scripted",
        seed=item.seed,
        sim_dt=sim_dt,
        control_dt=control_dt,
        robot_asset_id=str(robot_asset["id"]),
        robot_asset_sha256=str(robot_asset["sha256"]),
    )
    buffer = EpisodeBuffer(meta=meta)
    buffer.extras["robot_asset_manifest"] = robot_asset["manifest"]
    return _EpisodeSetup(
        buffer=buffer,
        controller=DoorPushController(active_cfg),
        controller_cfg=active_cfg,
        door_frame=door_frame,
        door_pose_obs=_door_pose_observation(env, door_frame),
        control_dt=control_dt,
        settle_report=settle_report,
    )


def _door_pose_observation(env, door_frame: ObjectFrame) -> dict[str, float]:
    """Constant episode-level door pose expressed relative to the robot base."""
    door_yaw_rad = float(math.atan2(door_frame.rot[1, 0], door_frame.rot[0, 0]))
    base_pos_w = np.asarray(_numpy(env.robot_base_pos_w())[0], dtype=np.float64)
    door_rel_pos = door_frame.origin - base_pos_w
    return {
        "door_yaw_rad": door_yaw_rad,
        "door_rel_pos_x": float(door_rel_pos[0]),
        "door_rel_pos_y": float(door_rel_pos[1]),
        "door_rel_pos_z": float(door_rel_pos[2]),
    }


def _record_episode_ticks(
    env,
    engine_cfg: DataEngineCfg,
    setup: _EpisodeSetup,
    render_hook,
) -> _EpisodeRuntime:
    runtime = _EpisodeRuntime()
    for tick in range(engine_cfg.max_ticks):
        snapshot = _read_tick_snapshot(env)
        runtime.final_angle = snapshot.angle
        command = setup.controller.act(
            DoorPushObservation(
                door_frame=setup.door_frame,
                hinge_angle_rad=snapshot.angle,
                hinge_velocity_rad_s=snapshot.velocity,
                ee_pos_w=snapshot.ee_pos_w,
                contact_sensed=snapshot.contact_sensed,
            )
        )
        runtime.last_command = command
        controller_reason = _controller_stop_reason(command)
        if controller_reason:
            runtime.termination_reason = controller_reason
            break

        delta_world = frame_delta_to_world(command.delta_door_frame, setup.door_frame)
        setup.buffer.add_step(_build_episode_step(env, setup, snapshot, command, delta_world))
        runtime.actions_door_frame.append(np.asarray(command.delta_door_frame, dtype=np.float64))
        step_outcome = _step_episode_env(env, delta_world, tick, render_hook)
        if step_outcome.termination_reason:
            _apply_env_step_outcome(runtime, step_outcome)
            break
    return runtime


def _read_tick_snapshot(env) -> _TickSnapshot:
    angle, velocity = _hinge_state(env)
    ee_pos_w, ee_quat_w = _ee_pose(env)
    contact_sensed, contact_force_n = _read_force_contact(env)
    return _TickSnapshot(
        angle,
        velocity,
        ee_pos_w,
        ee_quat_w,
        contact_sensed,
        contact_force_n,
    )


def _read_force_contact(env) -> tuple[bool, float]:
    force_w, sensed = env.contact_state()
    return bool(_numpy(sensed).reshape(-1)[0]), float(np.linalg.norm(_numpy(force_w)[0]))


def _controller_stop_reason(command) -> str:
    if command.done:
        return "controller_done"
    if command.timed_out:
        return "controller_timeout"
    return ""


def _build_episode_step(
    env,
    setup: _EpisodeSetup,
    snapshot: _TickSnapshot,
    command,
    delta_world: np.ndarray,
) -> EpisodeStep:
    proprio: dict[str, np.ndarray] = {
        "ee_pos_w": snapshot.ee_pos_w,
        "ee_quat_w_xyzw": snapshot.ee_quat_w,
    }
    proprio.update(env.robot_joint_state())
    return EpisodeStep(
        t=setup.buffer.n_steps * setup.control_dt,
        action=delta_world,
        obs_ref={
            "door_angle_rad": snapshot.angle,
            "door_angular_velocity_rad_s": snapshot.velocity,
            "ee_pos_x_m": float(snapshot.ee_pos_w[0]),
            "ee_pos_y_m": float(snapshot.ee_pos_w[1]),
            "ee_pos_z_m": float(snapshot.ee_pos_w[2]),
        },
        proprio=proprio,
        object_state={
            "door_angle_rad": snapshot.angle,
            "door_angular_velocity_rad_s": snapshot.velocity,
            **setup.door_pose_obs,
        },
        contact={
            "inferred": command.contact_inferred,
            "sensed": snapshot.contact_sensed,
            "force_n": snapshot.contact_force_n,
            "source": CONTACT_SOURCE_FORCE,
        },
        safety={
            "controller_phase": str(command.phase),
            "pos_clamped": bool(np.any(np.abs(delta_world[:3]) > env.cfg.max_pos_delta_m + 1e-12)),
            "rot_clamped": bool(
                np.any(np.abs(delta_world[3:]) > env.cfg.max_rot_delta_rad + 1e-12)
            ),
        },
    )


def _step_episode_env(env, delta_world: np.ndarray, tick: int, render_hook) -> _EnvStepOutcome:
    action = torch.as_tensor(delta_world, dtype=torch.float32).reshape(1, -1)
    try:
        step_result = env.step(action)
    except RuntimeError as error:
        return _EnvStepOutcome("step_error", f"env.step failed: {error}")
    terminated, truncated = _step_termination_flags(step_result)
    if terminated:
        return _EnvStepOutcome("environment_terminated", environment_terminated=True)
    if truncated:
        return _EnvStepOutcome("environment_truncated", environment_truncated=True)
    if render_hook is not None:
        render_hook(tick)
    return _EnvStepOutcome()


def _apply_env_step_outcome(runtime: _EpisodeRuntime, outcome: _EnvStepOutcome) -> None:
    runtime.notes = outcome.notes
    runtime.termination_reason = outcome.termination_reason
    runtime.environment_terminated = outcome.environment_terminated
    runtime.environment_truncated = outcome.environment_truncated


def _assert_no_silent_episode_reset(
    env,
    item: EpisodePlanItem,
    buffer: EpisodeBuffer,
    runtime: _EpisodeRuntime,
) -> None:
    if not buffer.n_steps or runtime.environment_terminated or runtime.environment_truncated:
        return
    env_ticks = int(_numpy(env.episode_length_buf)[0])
    if env_ticks < buffer.n_steps:
        raise RuntimeError(
            f"episode seed {item.seed} hit the env's auto-reset after "
            f"{buffer.n_steps} executed steps (episode counter {env_ticks}); "
            "the recorded final state would be invalid — lower engine "
            "max_ticks or raise the env's episode_length_s"
        )


def _record_terminal_contact(env, setup: _EpisodeSetup, runtime: _EpisodeRuntime) -> None:
    if not setup.buffer.n_steps:
        return
    if runtime.environment_terminated or runtime.environment_truncated:
        return
    sensed, force_n = _read_force_contact(env)
    setup.buffer.extras["terminal_contact"] = {
        "sensed": sensed,
        "force_n": force_n,
        "t": setup.buffer.n_steps * setup.control_dt,
        "alignment": (
            "post-step env state at loop exit: the contact/force response to "
            "the final executed action (steps[t].contact is pre-action, i.e. "
            "the response to action t-1)"
        ),
    }


def _finalize_episode(
    env,
    item: EpisodePlanItem,
    engine_cfg: DataEngineCfg,
    setup: _EpisodeSetup,
    runtime: _EpisodeRuntime,
) -> None:
    if (
        not runtime.environment_terminated
        and not runtime.environment_truncated
        and runtime.termination_reason != "step_error"
    ):
        runtime.final_angle, _ = _hinge_state(env)
    chunks = setup.controller.finalize()
    timed_out = bool(runtime.last_command is not None and runtime.last_command.timed_out)
    controller_done = bool(runtime.last_command is not None and runtime.last_command.done)
    setup.buffer.extras.update(
        {
            "action_door_frame": np.stack(runtime.actions_door_frame)
            if runtime.actions_door_frame
            else np.zeros((0, 6)),
            "door_frame_pos_w": setup.door_frame.origin.copy(),
            "door_frame_quat_w_xyzw": _door_frame_quat(env),
            "a4_chunks": [chunk.to_dict() for chunk in chunks],
            "variation": item.variation.to_dict() if item.variation is not None else None,
            "start_pose_settle": setup.settle_report,
            "controller_cfg": asdict(setup.controller_cfg),
            "engine_cfg": engine_cfg.to_dict(),
            "door_pose_id": engine_cfg.door_pose_id,
            "controller_done": controller_done,
            "controller_timed_out": timed_out,
            "last_phase": str(runtime.last_command.phase)
            if runtime.last_command is not None
            else "",
        }
    )
    _record_final_state(env, setup)
    success = (
        math.isfinite(runtime.final_angle) and runtime.final_angle >= engine_cfg.success_angle_rad
    )
    setup.buffer.set_outcome(
        EpisodeOutcome(
            success=success,
            final_door_angle=runtime.final_angle,
            n_steps=setup.buffer.n_steps,
            termination_reason=runtime.termination_reason,
            environment_terminated=runtime.environment_terminated,
            environment_truncated=runtime.environment_truncated,
            notes=runtime.notes,
        )
    )


def _record_final_state(env, setup: _EpisodeSetup) -> None:
    buffer = setup.buffer
    buffer.extras["joint_names"] = list(env.robot_joint_names())
    buffer.extras["arm_joint_ids"] = [int(index) for index in env.arm_joint_ids()]
    buffer.extras["final_joint_pos_target"] = np.asarray(
        env.robot_joint_state()["joint_pos_target"], dtype=np.float64
    )
    for name, value in env.robot_joint_limits().items():
        buffer.extras[name] = np.asarray(value, dtype=np.float64)
    buffer.extras["ik_clamp_telemetry"] = env.ik_clamp_telemetry()


def _step_termination_flags(step_result: Any) -> tuple[bool, bool]:
    """Extract Gymnasium termination flags from a duck-typed env step result."""
    if not isinstance(step_result, tuple) or len(step_result) < 5:
        return False, False
    return bool(_numpy(step_result[2]).reshape(-1)[0]), bool(_numpy(step_result[3]).reshape(-1)[0])


def traces_equal(
    first: EpisodeBuffer, second: EpisodeBuffer, tol: float = 1e-6, force_tol: float | None = None
) -> float:
    """Compare deterministic action, state, contact, and phase traces."""
    if first.n_steps != second.n_steps:
        raise AssertionError(f"step-count mismatch: {first.n_steps} != {second.n_steps}")
    force_tol = tol if force_tol is None else force_tol

    numeric_traces: list[tuple[str, float, Any]] = [
        ("action", tol, lambda s: s.action),
        ("proprio.ee_pos_w", tol, lambda s: s.proprio["ee_pos_w"]),
        ("object_state.door_angle_rad", tol, lambda s: [s.object_state["door_angle_rad"]]),
    ]
    for key in ("joint_pos", "joint_vel", "joint_pos_target"):
        if _both_have(first, second, lambda s, k=key: k in s.proprio):
            numeric_traces.append((f"proprio.{key}", tol, lambda s, k=key: s.proprio[k]))
    if _both_have(first, second, lambda s: "force_n" in s.contact):
        numeric_traces.append(
            ("contact.force_n", force_tol, lambda s: [float(s.contact["force_n"])])
        )

    max_diff = 0.0
    for name, trace_tol, getter in numeric_traces:
        a = first.stacked(getter)
        b = second.stacked(getter)
        diff = 0.0 if a.size == 0 else float(np.max(np.abs(a - b)))
        max_diff = max(max_diff, diff)
        if diff > trace_tol:
            raise AssertionError(f"episode {name!r} traces differ by {diff:.3g} > {trace_tol:.3g}")

    exact_traces = [("safety.controller_phase", lambda s: s.safety["controller_phase"])]
    if _both_have(first, second, lambda s: s.contact.get("sensed") is not None):
        exact_traces.append(("contact.sensed", lambda s: bool(s.contact["sensed"])))
    for name, getter in exact_traces:
        for tick, (step_a, step_b) in enumerate(zip(first.steps, second.steps, strict=True)):
            if getter(step_a) != getter(step_b):
                raise AssertionError(
                    f"episode {name!r} traces differ at tick {tick}: "
                    f"{getter(step_a)!r} != {getter(step_b)!r}"
                )
    return max_diff


def _both_have(first: EpisodeBuffer, second: EpisodeBuffer, predicate) -> bool:
    return (
        bool(first.steps)
        and bool(second.steps)
        and predicate(first.steps[0])
        and predicate(second.steps[0])
    )


def _read_door_frame(env) -> ObjectFrame:
    frame_pos, frame_quat = env.door_frame_pose_w()
    return door_frame_from_body_pose(_numpy(frame_pos)[0], _numpy(frame_quat)[0])


def _door_frame_quat(env) -> np.ndarray:
    _, frame_quat = env.door_frame_pose_w()
    return _numpy(frame_quat)[0].astype(np.float64)


def apply_start_offset(env, door_frame: ObjectFrame, variation: DoorPushVariation) -> dict:
    """Apply a door-frame start offset and return its settle evidence."""
    pos, quat = env.ee_pose_w()
    offset_w = door_frame.vector_to_world(np.asarray(variation.start_offset_door_frame))
    new_pos = _numpy(pos)[0] + offset_w
    env.set_ee_pose_w(
        torch.as_tensor(new_pos, dtype=torch.float32).reshape(1, 3), quat.reshape(1, 4)
    )
    report = env.start_pose_settle_report()
    if report is None:
        raise RuntimeError("environment did not record start-pose settle evidence")
    return report


def _hinge_state(env) -> tuple[float, float]:
    angle, velocity = env.hinge_state()
    return float(_numpy(angle)[0]), float(_numpy(velocity)[0])


def _ee_pose(env) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = env.ee_pose_w()
    return _numpy(pos)[0].astype(np.float64), _numpy(quat)[0].astype(np.float64)


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
