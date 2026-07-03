"""Deterministic scripted-episode generation (Phase 2 data engine).

The engine drives a :class:`DoorPushEnv`-shaped environment with the scripted
controller and records every control tick to the episode schema. The env is
duck-typed through its Phase 2 state accessors (``door_frame_pose_w``,
``hinge_state``, ``proxy_pose_w``, ``set_proxy_pose``, ``step``, ``reset``), so
the loop itself has no Isaac imports and is testable against a synthetic env.

Recorded actions are the executed world-frame EE deltas (A2). The controller's
native door-frame deltas (A3) are stored per step in ``extras`` so the A3
export is a relabeling, not a recomputation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from alexdoor_xas.action.frames import ObjectFrame, door_frame_from_body_pose, frame_delta_to_world
from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.eval.failures import label_episode
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

PROXY_LIMITATIONS = (
    "Robot is a velocity-driven proxy sphere (`proxy_ee_sphere_v0`), not Alex; "
    "A2 rotation components are accepted but physically inert for the sphere.",
    "`A1_joint_delta` is not exported: the proxy has no joints.",
    "Contact flags are geometric inference (`inferred_geometric`), not force sensing.",
    "The door frame pose is read from the USD stage at reset (world-fixed by the "
    "task layer); live articulation pose reads return zeros in this Isaac Lab build.",
)

ALEX_LIMITATIONS = (
    "Alex is fixed-base (pelvis welded to the world): no stepping, balancing, or "
    "regrasping; legs/torso/left arm are position-held at the standing pose.",
    "A2 rotation deltas are clamped and recorded but not actuated (position-only "
    "differential IK); same rotation contract as the proxy sphere.",
    "A1 is exported as 29-wide full-body joint-position-target deltas relabeled "
    "from the recorded per-tick targets; only the 6 right-arm IK joints move "
    "(the rest are position-held, so their deltas are zero).",
    "Contact force is the EE link's net contact force (PhysX could not build the "
    "filtered gripper<->door pair view for the referenced door USD); in this scene "
    "the gripper can only touch the door assembly.",
    "The EE contact point is the gripper link's 0.05 m collision sphere; there is "
    "no articulated hand.",
    "The Alex env adds passive hinge damping (4 N*m*s/rad) so the door moves only "
    "while pushed; the proxy env keeps the frozen undamped hinge.",
)


@dataclass(frozen=True)
class DataEngineCfg:
    """Engine-level settings shared by all episodes of one run."""

    task: str = "door_push"
    scene: str = "outputs/door_task/door_task.usda"
    robot: str = "proxy_ee_sphere_v0"
    policy: str = "scripted"
    success_angle_rad: float = math.pi / 4.0
    max_ticks: int = 600
    limitations: tuple[str, ...] = PROXY_LIMITATIONS
    """Known limitations of the run setup, surfaced in the run report."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def run_episode(
    env,
    item: EpisodePlanItem,
    engine_cfg: DataEngineCfg | None = None,
    controller_cfg: DoorPushControllerCfg | None = None,
    render_hook=None,
) -> EpisodeBuffer:
    """Roll out and record one episode; deterministic given (env state, item)."""
    engine_cfg = engine_cfg or DataEngineCfg()
    base_controller_cfg = controller_cfg or DoorPushControllerCfg()

    env.reset(seed=item.seed)
    door_frame = _read_door_frame(env)

    active_cfg = base_controller_cfg
    if item.variation is not None:
        active_cfg = item.variation.apply(base_controller_cfg)
        _apply_start_offset(env, door_frame, item.variation)
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
    )
    buffer = EpisodeBuffer(meta=meta)
    actions_door_frame: list[np.ndarray] = []
    has_force_contact = hasattr(env, "contact_sensed") and hasattr(env, "contact_force_w")
    has_joint_state = hasattr(env, "robot_joint_state")

    notes = ""
    last_command = None
    for tick in range(engine_cfg.max_ticks):
        angle, velocity = _hinge_state(env)
        ee_pos_w, ee_quat_w = _proxy_pose(env)
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
        if command.done or command.timed_out:
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
            env.step(action)
        except RuntimeError as error:  # non-finite sim state raised by the env
            notes = f"env.step failed: {error}"
            break
        if render_hook is not None:
            render_hook(tick)

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
            "controller_cfg": asdict(active_cfg),
            "engine_cfg": engine_cfg.to_dict(),
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
    buffer.set_outcome(
        EpisodeOutcome(
            success=success,
            final_door_angle=final_angle,
            failure_label=label_episode(
                final_angle_rad=final_angle,
                success_angle_rad=engine_cfg.success_angle_rad,
                controller_done=controller_done,
                timed_out=timed_out,
                last_phase=str(last_command.phase) if last_command is not None else "unknown",
                notes=notes,
            ),
            n_steps=buffer.n_steps,
            notes=notes,
        )
    )
    return buffer


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


def _apply_start_offset(env, door_frame: ObjectFrame, variation: DoorPushVariation) -> None:
    pos, quat = env.proxy_pose_w()
    offset_w = door_frame.vector_to_world(np.asarray(variation.start_offset_door_frame))
    new_pos = _numpy(pos)[0] + offset_w
    env.set_proxy_pose(
        torch.as_tensor(new_pos, dtype=torch.float32).reshape(1, 3), quat.reshape(1, 4)
    )


def _hinge_state(env) -> tuple[float, float]:
    angle, velocity = env.hinge_state()
    return float(_numpy(angle)[0]), float(_numpy(velocity)[0])


def _proxy_pose(env) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = env.proxy_pose_w()
    return _numpy(pos)[0].astype(np.float64), _numpy(quat)[0].astype(np.float64)


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "ALEX_LIMITATIONS",
    "CONTACT_SOURCE",
    "CONTACT_SOURCE_FORCE",
    "PROXY_LIMITATIONS",
    "DataEngineCfg",
    "EpisodePlanItem",
    "plan_episodes",
    "run_episode",
    "traces_equal",
]
