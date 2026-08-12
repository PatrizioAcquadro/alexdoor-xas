#!/usr/bin/env python
"""Phase 3.1 verification gate: adapter-v1 executes A2/A3/A4 on the Alex env.

Proves the adapter layer (``src/alexdoor_xas/adapters/``,
knowledge/wiki/topics/action-representations-and-adapters.md) on
``AlexDoor-DoorPush-AlexV2-v0``. Fails non-zero unless:

- a fixed-seed scripted reference episode succeeds (door past the success
  threshold),
- **A2 replay**: the reference episode's recorded world-frame EE deltas pushed
  through ``A2Adapter`` + the rollout driver reproduce the reference final
  door angle (headless physics is deterministic), with zero rejections or
  corrections,
- **A3 replay**: the recorded door-frame deltas pushed through ``A3Adapter``
  (object-frame transform -> A2) reproduce the same final angle,
- **A4 execution**: the reference episode's object-centric chunk sequence
  executed by ``A4Adapter`` re-opens the door past the success threshold with
  contact reached, and the result logs requested vs achieved hinge delta,
  contact, status, and final door angle change,
- **rejection path**: a physically invalid chunk (negative hinge delta = pull)
  is rejected with a reason and commands zero motion,
- adapter logs and the A4 execution result are written as JSON artifacts.

Gate artifacts go under ``~/.cache/alexdoor-xas/verification/verify_adapters/``.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_adapters.py --viz none --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

# -- AppLauncher must be configured before any other Isaac import.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS adapter-v1 verification gate")
parser.add_argument("--seed", type=int, default=0, help="Seed for the reference episode.")
parser.add_argument(
    "--replay-tol",
    type=float,
    default=1e-6,
    help="Maximum allowed final-door-angle difference between reference and adapter replay.",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting; useful for debugging Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.action.spaces import ObjectCentricChunk  # noqa: E402
from alexdoor_xas.adapters.a2 import A2Adapter  # noqa: E402
from alexdoor_xas.adapters.a3 import A3Adapter  # noqa: E402
from alexdoor_xas.adapters.a4 import A4Adapter, alex_v2_a4_cfg  # noqa: E402
from alexdoor_xas.adapters.base import AdapterStatus  # noqa: E402
from alexdoor_xas.adapters.limits import DoorPanelGeometry, limits_for_robot  # noqa: E402
from alexdoor_xas.adapters.rollout import replay_source, rollout_chunks  # noqa: E402
from alexdoor_xas.data_engine import (  # noqa: E402
    DEFAULT_SUCCESS_ANGLE_RAD,
    DataEngineCfg,
    plan_episodes,
    run_episode,
)
from alexdoor_xas.envs.door_task.alex_v2_runtime import ALEX_V2_LIMITATIONS  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_ROBOT_TAG,
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.policies.scripted import alex_v2_push_cfg  # noqa: E402

EXPERIMENT = "verify_adapters"
A4_MAX_TICKS = 2400


def _make_env():
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped


def _reference_episode(env):
    engine_cfg = DataEngineCfg(
        task=paths.ALEX_V2_TASK,
        robot=ALEX_V2_ROBOT_TAG,
        limitations=ALEX_V2_LIMITATIONS,
    )
    item = plan_episodes(1, 0, args.seed)[0]
    controller_cfg = alex_v2_push_cfg(env.alex_v2_calibration())
    episode = run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
    if not episode.outcome.success:
        raise RuntimeError(
            f"reference scripted episode must succeed; "
            f"termination={episode.outcome.termination_reason!r} "
            f"final_angle={episode.outcome.final_door_angle:.4f} rad"
        )
    return episode


def _fresh_adapters(env):
    center_w = env.shoulder_position_world_m()[0].detach().cpu().numpy()
    limits = limits_for_robot(
        ALEX_V2_ROBOT_TAG,
        calibration=env.alex_v2_calibration(),
        workspace_center_w=center_w,
    )
    a2 = A2Adapter(limits)
    a3 = A3Adapter(a2)
    return a2, a3, A4Adapter(a3, cfg=alex_v2_a4_cfg(env.alex_v2_calibration()))


def _assert_replay(env, episode, actions, adapter, label: str, out_dir):
    env.reset(seed=args.seed)
    result = rollout_chunks(env, replay_source(actions), adapter, max_ticks=episode.n_steps + 1)
    result.log.write_json(out_dir / f"{label}_log.json")
    (out_dir / f"{label}_result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    if result.n_ticks != episode.n_steps:
        raise RuntimeError(
            f"{label}: replay executed {result.n_ticks} ticks, reference has {episode.n_steps}"
        )
    if result.log.n_rejected or result.log.n_corrected:
        raise RuntimeError(
            f"{label}: recorded actions must replay untouched, got "
            f"{result.log.n_rejected} rejected / {result.log.n_corrected} corrected"
        )
    diff = abs(result.final_angle_rad - episode.outcome.final_door_angle)
    if diff > args.replay_tol:
        raise RuntimeError(
            f"{label}: final angle {result.final_angle_rad:.6f} differs from reference "
            f"{episode.outcome.final_door_angle:.6f} by {diff:.3g} > {args.replay_tol:.3g}"
        )
    print(
        f"[{label}] ticks={result.n_ticks} final_angle={math.degrees(result.final_angle_rad):.2f} "
        f"deg (reference diff {diff:.3g} rad), all {result.log.n_accepted} commands accepted",
        flush=True,
    )
    return result


def _assert_a4_execution(env, episode, success_angle_rad: float, out_dir):
    chunks = [ObjectCentricChunk.from_dict(entry) for entry in episode.extras["a4_chunks"]]
    if not any(chunk.phase == "push" and chunk.motion_hinge_delta_rad > 0 for chunk in chunks):
        raise RuntimeError("reference chunk log has no push chunk with positive hinge delta")

    env.reset(seed=args.seed)
    _, _, a4 = _fresh_adapters(env)
    result = a4.execute(env, chunks, max_ticks=A4_MAX_TICKS)
    (out_dir / "a4_execution.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    if result.status is AdapterStatus.REJECTED:
        raise RuntimeError(f"A4 execution was rejected: {result.reason}")
    if not result.completed:
        raise RuntimeError(
            f"A4 execution did not complete: failure={result.failure!r} reason={result.reason!r}"
        )
    if not result.contact_reached:
        raise RuntimeError("A4 execution never reached contact")
    if result.final_angle_rad < success_angle_rad:
        raise RuntimeError(
            f"A4 execution final angle {result.final_angle_rad:.4f} rad is below the "
            f"success threshold {success_angle_rad:.4f} rad"
        )
    if result.achieved_hinge_delta_rad < result.requested_hinge_delta_rad - 1e-6:
        raise RuntimeError(
            f"A4 achieved hinge delta {result.achieved_hinge_delta_rad:.4f} rad is below "
            f"the requested {result.requested_hinge_delta_rad:.4f} rad"
        )
    # The logged result must carry every field required by the adapter wiki contract.
    payload = result.to_dict()
    for key in (
        "requested_hinge_delta_rad",
        "achieved_hinge_delta_rad",
        "contact_reached",
        "contact_missed",
        "status",
        "reason",
        "final_door_angle_change_rad",
    ):
        if key not in payload:
            raise RuntimeError(f"A4 execution result is missing the {key!r} field")
    print(
        f"[a4] status={payload['status']} requested={result.requested_hinge_delta_rad:.4f} rad "
        f"achieved={result.achieved_hinge_delta_rad:.4f} rad "
        f"contact_reached={result.contact_reached} ticks={result.n_ticks} "
        f"final_angle={math.degrees(result.final_angle_rad):.2f} deg",
        flush=True,
    )
    return result


def _hinge_angle(env) -> float:
    return float(np.asarray(env.hinge_state()[0].cpu())[0])


def _ee_pos_w(env) -> np.ndarray:
    return np.asarray(env.ee_pose_w()[0].cpu())[0]


def _assert_rejected_a4_case(env, out_dir, artifact_name: str, chunk, reason_substring: str):
    env.reset(seed=args.seed)
    angle_before = _hinge_angle(env)
    ee_before = _ee_pos_w(env)
    _, _, a4 = _fresh_adapters(env)
    result = a4.execute(env, chunk)
    (out_dir / f"{artifact_name}.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")

    if result.status is not AdapterStatus.REJECTED:
        raise RuntimeError(f"{artifact_name}: chunk must be rejected, got {result.status}")
    if not result.reason:
        raise RuntimeError(f"{artifact_name}: rejected A4 execution must carry a reason")
    if reason_substring not in result.reason:
        raise RuntimeError(
            f"{artifact_name}: rejection reason {result.reason!r} does not contain "
            f"{reason_substring!r}"
        )
    if result.n_ticks != 0 or result.stages:
        raise RuntimeError(f"{artifact_name}: rejected A4 execution must command zero motion")
    if result.achieved_hinge_delta_rad != 0.0:
        raise RuntimeError(f"{artifact_name}: rejected A4 execution reported achieved motion")
    if result.contact_reached:
        raise RuntimeError(f"{artifact_name}: rejected A4 execution reported contact")
    if result.log.n_rejected != 1:
        raise RuntimeError(
            f"{artifact_name}: rejected chunk must be reflected in AdapterLog, got "
            f"{result.log.n_rejected} rejected decisions"
        )
    angle_after = _hinge_angle(env)
    if abs(angle_after - angle_before) > 1e-9:
        raise RuntimeError(f"{artifact_name}: door moved during a rejected A4 execution")
    ee_after = _ee_pos_w(env)
    if np.linalg.norm(ee_after - ee_before) > 1e-9:
        raise RuntimeError(f"{artifact_name}: EE moved during a rejected A4 execution")
    print(
        f"[a4-reject:{artifact_name}] status={result.status} reason={result.reason!r}",
        flush=True,
    )
    return result


def _assert_a4_rejections(env, out_dir):
    face_x = DoorPanelGeometry().surface_x_m(0.0)
    cases = [
        (
            "a4_rejected_pull",
            ObjectCentricChunk(
                phase="push",
                contact_target_panel=(face_x, 0.29, 0.15),
                motion_hinge_delta_rad=-0.3,
                duration_ticks=100,
            ),
            "pulling",
        ),
        (
            "a4_rejected_approach_hinge_motion",
            ObjectCentricChunk(
                phase="approach",
                contact_target_panel=(face_x, 0.29, 0.15),
                motion_hinge_delta_rad=0.1,
                duration_ticks=100,
            ),
            "non-push phase cannot request hinge motion",
        ),
        (
            "a4_rejected_hold_hinge_motion",
            ObjectCentricChunk(
                phase="hold",
                contact_target_panel=(face_x, 0.29, 0.15),
                motion_hinge_delta_rad=0.1,
                duration_ticks=100,
            ),
            "non-push phase cannot request hinge motion",
        ),
        (
            "a4_rejected_malformed_target",
            ObjectCentricChunk(
                phase="push",
                contact_target_panel=(face_x, 0.29),
                motion_hinge_delta_rad=0.0,
                duration_ticks=100,
            ),
            "contact_target_panel",
        ),
    ]
    for artifact_name, chunk, reason_substring in cases:
        _assert_rejected_a4_case(env, out_dir, artifact_name, chunk, reason_substring)


def main() -> int:
    rc = 0
    env = None
    try:
        out_dir = paths.VERIFICATION_CACHE_DIR / EXPERIMENT / "gate"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = _make_env()

        episode = _reference_episode(env)
        print(
            f"[reference] steps={episode.n_steps} "
            f"final_angle={math.degrees(episode.outcome.final_door_angle):.2f} deg",
            flush=True,
        )

        actions_world = [step.action for step in episode.steps]
        a2, _, _ = _fresh_adapters(env)
        _assert_replay(env, episode, actions_world, a2, "a2_replay", out_dir)

        actions_door = list(np.asarray(episode.extras["action_door_frame"]))
        _, a3, _ = _fresh_adapters(env)
        _assert_replay(env, episode, actions_door, a3, "a3_replay", out_dir)

        success_angle = DEFAULT_SUCCESS_ANGLE_RAD
        _assert_a4_execution(env, episode, success_angle, out_dir)
        _assert_a4_rejections(env, out_dir)

        print(f"[artifacts] {out_dir}", flush=True)
        print("PASS: adapter-v1 gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: adapter-v1 gate failed.", flush=True)
        rc = 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
        if args.clean_shutdown:
            try:
                print("[shutdown] closing SimulationApp (--clean-shutdown)", flush=True)
                simulation_app.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
    return rc


if __name__ == "__main__":
    # os._exit avoids Kit shutdown masking the verification exit code.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
