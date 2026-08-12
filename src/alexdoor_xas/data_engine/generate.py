"""Deterministic scripted-episode generation for the Alex V2 benchmark.

The engine drives a door-push environment with the scripted
controller and records every control tick to the episode schema. The env is
duck-typed through its benchmark state accessors (``door_frame_pose_w``,
``hinge_state``, ``ee_pose_w``, ``set_ee_pose_w``, ``step``, ``reset``), so
the loop itself has no Isaac imports and is testable against a synthetic env.

Recorded actions are the executed world-frame EE deltas (A2). The controller's
native door-frame deltas (A3) are stored per step in ``extras`` so the A3
export is a relabeling, not a recomputation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from alexdoor_xas.action.frames import ObjectFrame, door_frame_from_body_pose, frame_delta_to_world
from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.assets.alex_v2_contract import (
    AlexV2ContractError,
    RobotAssetRef,
    validate_alex_v2_manifest,
)
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

CONTACT_SOURCE = "inferred_geometric"
CONTACT_SOURCE_FORCE = "force_sensor+geometric"
DEFAULT_SUCCESS_ANGLE_RAD = math.pi / 4.0
DEFAULT_MAX_TICKS = 600

_MISSING = object()


@dataclass(frozen=True)
class DataEngineCfg:
    """Engine-level settings shared by all episodes of one run."""

    task: str
    robot: str
    limitations: tuple[str, ...]
    """Known limitations of the run setup, surfaced in the run report."""
    scene: str = ""
    policy: str = "scripted"
    success_angle_rad: float = DEFAULT_SUCCESS_ANGLE_RAD
    max_ticks: int = DEFAULT_MAX_TICKS
    door_pose_id: str = DEFAULT_DOOR_POSE_ID
    """Canonical door pose selected by ID; routine generation never accepts raw transforms."""

    def __post_init__(self) -> None:
        canonical_door_pose(self.door_pose_id)
        expected_scene = f"outputs/door_scene/{self.door_pose_id}.usda"
        if self.scene and self.scene != expected_scene:
            raise ValueError(
                f"scene {self.scene!r} conflicts with canonical pose {self.door_pose_id!r}; "
                f"expected {expected_scene!r}"
            )
        object.__setattr__(self, "scene", expected_scene)

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


def _validated_robot_asset_provenance(
    env: Any,
) -> tuple[RobotAssetRef | None, dict[str, Any] | None]:
    """Validate an optional env-provided robot asset before executing it."""
    accessor = getattr(env, "robot_asset_provenance", _MISSING)
    if accessor is _MISSING:
        return None, None
    if not callable(accessor):
        raise AlexV2ContractError("env robot_asset_provenance must be callable")

    payload = accessor()
    if not isinstance(payload, Mapping):
        raise AlexV2ContractError("env robot_asset_provenance must return an object")
    required = {"id", "sha256", "manifest_fingerprint", "manifest"}
    missing = required.difference(payload)
    if missing:
        raise AlexV2ContractError(
            "env robot_asset_provenance is missing required keys: "
            + ", ".join(sorted(missing))
        )

    manifest_value = payload["manifest"]
    if not isinstance(manifest_value, Mapping):
        raise AlexV2ContractError("env robot_asset_provenance manifest must be an object")
    manifest = deepcopy(dict(manifest_value))
    validated = validate_alex_v2_manifest(manifest)
    provided = RobotAssetRef(
        asset_id=str(payload["id"]),
        sha256=str(payload["sha256"]),
        manifest_fingerprint=str(payload["manifest_fingerprint"]),
    )
    if provided != validated:
        raise AlexV2ContractError(
            "env robot asset reference does not match its canonical manifest"
        )
    return validated, manifest


def run_episode(
    env,
    item: EpisodePlanItem,
    engine_cfg: DataEngineCfg,
    controller_cfg: DoorPushControllerCfg | None = None,
    render_hook=None,
) -> EpisodeBuffer:
    """Roll out and record one episode; deterministic given (env state, item)."""
    base_controller_cfg = controller_cfg or DoorPushControllerCfg()
    robot_asset_ref, robot_asset_manifest = _validated_robot_asset_provenance(env)

    env.reset(seed=item.seed)
    door_frame = _read_door_frame(env)

    active_cfg = base_controller_cfg
    settle_report: dict | None = None
    if item.variation is not None:
        active_cfg = item.variation.apply(base_controller_cfg)
        settle_report = apply_start_offset(env, door_frame, item.variation)
    controller = DoorPushController(active_cfg)

    sim_dt = float(env.cfg.sim.dt)
    decimation = int(env.cfg.decimation)
    control_dt = sim_dt * decimation

    meta = EpisodeMeta.create(
        task=engine_cfg.task,
        action_space=A2_EE_DELTA,
        robot=engine_cfg.robot,
        scene=engine_cfg.scene,
        policy=engine_cfg.policy,
        seed=item.seed,
        sim_dt=sim_dt,
        control_dt=control_dt,
        robot_asset_id=robot_asset_ref.asset_id if robot_asset_ref is not None else "",
        robot_asset_sha256=robot_asset_ref.sha256 if robot_asset_ref is not None else "",
    )
    buffer = EpisodeBuffer(meta=meta)
    if robot_asset_manifest is not None:
        buffer.extras["robot_asset_manifest"] = robot_asset_manifest
    actions_door_frame: list[np.ndarray] = []
    has_force_contact = hasattr(env, "contact_sensed") and hasattr(env, "contact_force_w")
    has_joint_state = hasattr(env, "robot_joint_state")

    # Door-pose observation terms (constant per episode: the pose is authored
    # into the scene USD per process, and the frame is static within a rollout).
    # Yaw is derived from the recorded door-frame rotation; the translation is
    # the door-frame origin relative to the robot base (world origin for the
    # origin for base-less test doubles) so it stays meaningful if re-based.
    door_yaw_rad = float(math.atan2(door_frame.rot[1, 0], door_frame.rot[0, 0]))
    base_pos_w = np.zeros(3)
    if hasattr(env, "robot_base_pos_w"):
        base_pos_w = np.asarray(_numpy(env.robot_base_pos_w())[0], dtype=np.float64)
    door_rel_pos = door_frame.origin - base_pos_w
    door_pose_obs = {
        "door_yaw_rad": door_yaw_rad,
        "door_rel_pos_x": float(door_rel_pos[0]),
        "door_rel_pos_y": float(door_rel_pos[1]),
        "door_rel_pos_z": float(door_rel_pos[2]),
    }

    notes = ""
    termination_reason = "tick_budget"
    environment_terminated = False
    environment_truncated = False
    final_angle = float("nan")
    last_command = None
    for tick in range(engine_cfg.max_ticks):
        angle, velocity = _hinge_state(env)
        final_angle = angle
        ee_pos_w, ee_quat_w = _ee_pose(env)
        contact_sensed: bool | None = None
        contact_force_n = 0.0
        if has_force_contact:
            contact_sensed = bool(_numpy(env.contact_sensed())[0])
            contact_force_n = float(np.linalg.norm(_numpy(env.contact_force_w())[0]))
        command = controller.act(
            DoorPushObservation(
                door_frame=door_frame,
                hinge_angle_rad=angle,
                hinge_velocity_rad_s=velocity,
                ee_pos_w=ee_pos_w,
                contact_sensed=contact_sensed,
            )
        )
        last_command = command
        if command.done:
            termination_reason = "controller_done"
            break
        if command.timed_out:
            termination_reason = "controller_timeout"
            break

        proprio: dict[str, np.ndarray] = {"ee_pos_w": ee_pos_w, "ee_quat_w_xyzw": ee_quat_w}
        if has_joint_state:
            proprio.update(env.robot_joint_state())
        if has_force_contact:
            contact = {
                "inferred": command.contact_inferred,
                "sensed": contact_sensed,
                "force_n": contact_force_n,
                "source": CONTACT_SOURCE_FORCE,
            }
        else:
            contact = {"inferred": command.contact_inferred, "source": CONTACT_SOURCE}

        delta_world = frame_delta_to_world(command.delta_door_frame, door_frame)
        buffer.add_step(
            EpisodeStep(
                t=tick * control_dt,
                action=delta_world,
                obs_ref={
                    "door_angle_rad": angle,
                    "door_angular_velocity_rad_s": velocity,
                    "ee_pos_x_m": float(ee_pos_w[0]),
                    "ee_pos_y_m": float(ee_pos_w[1]),
                    "ee_pos_z_m": float(ee_pos_w[2]),
                },
                proprio=proprio,
                object_state={
                    "door_angle_rad": angle,
                    "door_angular_velocity_rad_s": velocity,
                    **door_pose_obs,
                },
                contact=contact,
                safety={
                    "controller_phase": str(command.phase),
                    "pos_clamped": bool(
                        np.any(np.abs(delta_world[:3]) > env.cfg.max_pos_delta_m + 1e-12)
                    ),
                    "rot_clamped": bool(
                        np.any(np.abs(delta_world[3:]) > env.cfg.max_rot_delta_rad + 1e-12)
                    ),
                },
            )
        )
        actions_door_frame.append(np.asarray(command.delta_door_frame, dtype=np.float64))

        action = torch.as_tensor(delta_world, dtype=torch.float32).reshape(1, -1)
        try:
            step_result = env.step(action)
        except RuntimeError as error:  # non-finite sim state raised by the env
            notes = f"env.step failed: {error}"
            termination_reason = "step_error"
            break
        environment_terminated, environment_truncated = _step_termination_flags(step_result)
        if environment_terminated:
            termination_reason = "environment_terminated"
            break
        if environment_truncated:
            termination_reason = "environment_truncated"
            break
        if render_hook is not None:
            render_hook(tick)

    # A DirectRLEnv auto-resets *inside* env.step when the episode budget is
    # reached; everything read after that (final angle, final joint targets,
    # clamp telemetry) would silently be post-reset state. The env's episode
    # counter zeroes on reset, so a counter smaller than the executed step
    # count is unambiguous evidence of a mid-episode reset. Fail loudly.
    if (
        buffer.n_steps
        and not environment_terminated
        and not environment_truncated
        and hasattr(env, "episode_length_buf")
    ):
        env_ticks = int(_numpy(env.episode_length_buf)[0])
        if env_ticks < buffer.n_steps:
            raise RuntimeError(
                f"episode seed {item.seed} hit the env's auto-reset after "
                f"{buffer.n_steps} executed steps (episode counter {env_ticks}); "
                "the recorded final state would be invalid — lower engine "
                "max_ticks or raise the env's episode_length_s"
            )

    # Terminal post-action safety sample (additive, phase2.v1-compatible):
    # per-step contact samples are read *before* each tick's action, so the
    # force/contact response to the final executed action is only visible in
    # the env state at loop exit — capture it so the dataset admission bound
    # covers the response to every executed action, including the last one.
    if (
        buffer.n_steps
        and has_force_contact
        and not (environment_terminated or environment_truncated)
    ):
        buffer.extras["terminal_contact"] = {
            "sensed": bool(_numpy(env.contact_sensed())[0]),
            "force_n": float(np.linalg.norm(_numpy(env.contact_force_w())[0])),
            "t": buffer.n_steps * control_dt,
            "alignment": (
                "post-step env state at loop exit: the contact/force response to "
                "the final executed action (steps[t].contact is pre-action, i.e. "
                "the response to action t-1)"
            ),
        }

    if not (environment_terminated or environment_truncated) and termination_reason != "step_error":
        final_angle, _ = _hinge_state(env)
    chunk_log = controller.finalize()
    timed_out = bool(last_command is not None and last_command.timed_out)
    controller_done = bool(last_command is not None and last_command.done)
    success = math.isfinite(final_angle) and final_angle >= engine_cfg.success_angle_rad

    buffer.extras.update(
        {
            "action_door_frame": np.stack(actions_door_frame)
            if actions_door_frame
            else np.zeros((0, 6)),
            "door_frame_pos_w": door_frame.origin.copy(),
            "door_frame_quat_w_xyzw": _door_frame_quat(env),
            "a4_chunks": chunk_log.to_list(),
            "variation": item.variation.to_dict() if item.variation is not None else None,
            "start_pose_settle": settle_report,
            "controller_cfg": asdict(active_cfg),
            "engine_cfg": engine_cfg.to_dict(),
            "door_pose_id": engine_cfg.door_pose_id,
            "controller_done": controller_done,
            "controller_timed_out": timed_out,
            "last_phase": str(last_command.phase) if last_command is not None else "",
        }
    )
    if hasattr(env, "robot_joint_names"):
        buffer.extras["joint_names"] = list(env.robot_joint_names())
    if hasattr(env, "arm_joint_ids"):
        buffer.extras["arm_joint_ids"] = [int(i) for i in env.arm_joint_ids()]
    if has_joint_state:
        # Applied target after the last executed tick: per-step proprio targets
        # are captured pre-step, so the A1 (joint-target delta) relabel of the
        # final step needs this one extra sample (see export._relabel_to_joint_delta).
        buffer.extras["final_joint_pos_target"] = np.asarray(
            env.robot_joint_state()["joint_pos_target"], dtype=np.float64
        )
    if hasattr(env, "robot_joint_limits"):
        for name, value in env.robot_joint_limits().items():
            buffer.extras[name] = np.asarray(value, dtype=np.float64)
    if hasattr(env, "ik_clamp_telemetry"):
        # Raw pre-clamp diff-IK excess per joint (anti-windup telemetry): how
        # often and how far the solver ran past the position limits before the
        # executor clamped the targets. JSON-able → lands in the sidecar-style
        # /extras_json group.
        buffer.extras["ik_clamp_telemetry"] = env.ik_clamp_telemetry()
    buffer.set_outcome(
        EpisodeOutcome(
            success=success,
            final_door_angle=final_angle,
            n_steps=buffer.n_steps,
            termination_reason=termination_reason,
            environment_terminated=environment_terminated,
            environment_truncated=environment_truncated,
            notes=notes,
        )
    )
    return buffer


def _step_termination_flags(step_result: Any) -> tuple[bool, bool]:
    """Extract Gymnasium termination flags from a duck-typed env step result."""
    if not isinstance(step_result, tuple) or len(step_result) < 5:
        return False, False
    return bool(_numpy(step_result[2]).reshape(-1)[0]), bool(
        _numpy(step_result[3]).reshape(-1)[0]
    )


def traces_equal(
    first: EpisodeBuffer, second: EpisodeBuffer, tol: float = 1e-6, force_tol: float | None = None
) -> float:
    """Max abs difference between two episodes' action/state traces (determinism check).

    Always compares the action, EE position, and door angle. When both episodes
    recorded them, also compares joint positions/velocities/targets (at ``tol``),
    the sensed contact force (at ``force_tol``, default ``tol`` — headless physics
    is deterministic in this build, so the contact force is too), and the exact
    per-tick sensed-contact flags and controller phases.
    """
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
            raise AssertionError(
                f"episode {name!r} traces differ by {diff:.3g} > {trace_tol:.3g}"
            )

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
    return door_frame_from_body_pose(
        _numpy(frame_pos)[0], _numpy(frame_quat)[0]
    )


def _door_frame_quat(env) -> np.ndarray:
    _, frame_quat = env.door_frame_pose_w()
    return _numpy(frame_quat)[0].astype(np.float64)


def apply_start_offset(
    env, door_frame: ObjectFrame, variation: DoorPushVariation
) -> dict | None:
    """Shift the EE start pose by the variation's door-frame offset (shared with eval).

    Returns the env's realized-state settle report when it exposes one
    (``start_pose_settle_report``): requested/realized position, residual,
    settle ticks, and pass/fail — the fail-closed postcondition itself lives
    in the env's ``set_ee_pose_w``. Teleporting test fakes realize the request
    exactly and return ``None``.
    """
    pos, quat = env.ee_pose_w()
    offset_w = door_frame.vector_to_world(np.asarray(variation.start_offset_door_frame))
    new_pos = _numpy(pos)[0] + offset_w
    env.set_ee_pose_w(
        torch.as_tensor(new_pos, dtype=torch.float32).reshape(1, 3), quat.reshape(1, 4)
    )
    if hasattr(env, "start_pose_settle_report"):
        return env.start_pose_settle_report()
    return None


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


__all__ = [
    "CONTACT_SOURCE",
    "CONTACT_SOURCE_FORCE",
    "DEFAULT_MAX_TICKS",
    "DEFAULT_SUCCESS_ANGLE_RAD",
    "DataEngineCfg",
    "EpisodePlanItem",
    "plan_episodes",
    "plan_randomized_seeds",
    "run_episode",
    "traces_equal",
]
