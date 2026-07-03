#!/usr/bin/env python
"""Phase 2.5 verification gate: Alex fixed-base scripted rollout + data export.

Mirrors ``scripts/verify_scripted_baseline.py`` but on the Alex env
(``AlexDoor-DoorPush-Alex-v0``): the IHMC Alex humanoid, fixed-base, opens the
door with its right arm via differential IK, with force-sensed contact. Fails
non-zero unless:

- the fixed-start scripted episode succeeds (door past the success threshold),
- a repeated rollout of the same seed reproduces identical traces (headless),
  including joint state/targets, sensed contact flags, forces, and phases,
- episode meta carries the Alex robot tag and 29-joint proprio,
- force-sensed contact ticks with positive force were recorded,
- every episode passes the rollout sanity checks (finite joint data, targets
  within Isaac-reported limits, sane velocities, force-sensing contact source),
- A1/A2/A3/A4 dataset exports exist and satisfy the episode schema (A1 actions
  are the recorded joint-position-target deltas).

Gate artifacts go under ``outputs/verify_alex_door_baseline/``.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_alex_door_baseline.py --viz none --device cpu
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

parser = argparse.ArgumentParser(description="AlexDoor-XAS Alex door baseline verification gate")
parser.add_argument("--seed", type=int, default=0, help="Base seed for the gate episodes.")
parser.add_argument(
    "--determinism-tol",
    type=float,
    default=1e-6,
    help="Maximum allowed absolute difference between repeated episode traces.",
)
parser.add_argument(
    "--determinism-force-tol",
    type=float,
    default=None,
    help=(
        "Separate tolerance for the sensed contact-force trace (N); defaults to "
        "--determinism-tol (headless physics is deterministic in this build, so "
        "repeated forces match to the same tolerance as the kinematic traces)."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Also exercise the video hook (requires --enable_cameras).",
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
from alexdoor_xas.action.frames import door_frame_from_body_pose, world_delta_to_frame  # noqa: E402
from alexdoor_xas.action.spaces import (  # noqa: E402
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)
from alexdoor_xas.data_engine import (  # noqa: E402
    DataEngineCfg,
    plan_episodes,
    run_baseline,
    run_episode,
    traces_equal,
)
from alexdoor_xas.data_engine.generate import ALEX_LIMITATIONS, CONTACT_SOURCE_FORCE  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_env_cfg import (  # noqa: E402
    ALEX_ROBOT_TAG,
    DoorPushAlexEnvCfg,
)
from alexdoor_xas.eval import check_alex_episode  # noqa: E402
from alexdoor_xas.policies.scripted import (  # noqa: E402
    ALEX_VARIATION_BOUNDS,
    alex_fixedbase_push_cfg,
)
from alexdoor_xas.recording import read_episode  # noqa: E402

EXPERIMENT = "verify_alex_door_baseline"
GATE_N_FIXED = 1
GATE_N_RANDOMIZED = 1
N_ALEX_JOINTS = 29


def _make_env():
    cfg = DoorPushAlexEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    render_mode = "rgb_array" if args.video else None
    return gym.make(door_task.DOOR_PUSH_ALEX_ENV_ID, cfg=cfg, render_mode=render_mode).unwrapped


def _engine_cfg() -> DataEngineCfg:
    return DataEngineCfg(
        task="door_push_alex",
        robot=ALEX_ROBOT_TAG,
        limitations=ALEX_LIMITATIONS,
    )


def _assert_determinism(env, engine_cfg: DataEngineCfg) -> float:
    item = plan_episodes(GATE_N_FIXED, 0, args.seed)[0]
    controller_cfg = alex_fixedbase_push_cfg()
    first = run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
    second = run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
    max_diff = traces_equal(
        first, second, tol=args.determinism_tol, force_tol=args.determinism_force_tol
    )
    if not first.outcome.success:
        raise RuntimeError(
            f"fixed-start Alex episode must succeed; failure={first.outcome.failure_label!r} "
            f"final_angle={first.outcome.final_door_angle:.4f} rad "
            f"last_phase={first.extras.get('last_phase')!r}"
        )
    return max_diff


def _assert_alex_episode_schema(artifacts) -> None:
    fixed = artifacts.episodes[0]
    if fixed.meta.robot != ALEX_ROBOT_TAG:
        raise RuntimeError(
            f"episode robot tag must be {ALEX_ROBOT_TAG!r}, got {fixed.meta.robot!r}"
        )

    step = fixed.steps[0]
    for key in ("joint_pos", "joint_vel", "joint_pos_target"):
        if key not in step.proprio:
            raise RuntimeError(f"Alex episode proprio is missing {key!r}")
        width = np.asarray(step.proprio[key]).shape
        if width != (N_ALEX_JOINTS,):
            raise RuntimeError(f"proprio {key!r} must have shape ({N_ALEX_JOINTS},), got {width}")
    if len(fixed.extras.get("joint_names", [])) != N_ALEX_JOINTS:
        raise RuntimeError("episode extras must record the 29 Alex joint names")

    sensed_forces = [
        float(s.contact["force_n"])
        for s in fixed.steps
        if s.contact.get("sensed") and s.safety["controller_phase"] in ("contact", "push", "hold")
    ]
    if not sensed_forces:
        raise RuntimeError("no force-sensed contact ticks during the contact/push/hold phases")
    if max(sensed_forces) <= 0.0:
        raise RuntimeError("sensed contact ticks must carry positive force")
    if step.contact["source"] != CONTACT_SOURCE_FORCE:
        raise RuntimeError(
            f"contact source must be {CONTACT_SOURCE_FORCE!r}, got {step.contact['source']!r}"
        )


def _assert_sanity(artifacts) -> None:
    """Run the rollout sanity checks on every recorded episode (silent-bad-data net)."""
    for episode in artifacts.episodes:
        if "joint_pos_limits" not in episode.extras:
            raise RuntimeError("Alex episodes must record Isaac-reported joint limits in extras")
        result = check_alex_episode(episode)
        for warning in result.warnings:
            print(f"[sanity] WARNING: {warning}", flush=True)
        if not result.ok:
            raise RuntimeError("rollout sanity checks failed:\n" + "\n".join(result.errors))


def _assert_artifacts(artifacts) -> None:
    for episode in artifacts.episodes:
        if episode.outcome is None:
            raise RuntimeError("every recorded episode must have an outcome")
    fixed = artifacts.episodes[0]
    if not fixed.outcome.success or fixed.outcome.failure_label is not None:
        raise RuntimeError(
            f"gate fixed-start episode failed: label={fixed.outcome.failure_label!r}"
        )

    for name, path in artifacts.plots.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"plot {name!r} was not written: {path}")
    if not artifacts.report_path.is_file():
        raise RuntimeError(f"run report was not written: {artifacts.report_path}")
    metrics_path = artifacts.run_dir / "metrics" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics["aggregate"]["n_episodes"] != len(artifacts.episodes):
        raise RuntimeError(f"metrics.json episode count mismatch: {metrics_path}")
    fixed_metrics = metrics["episodes"][0]
    for key in ("mean_contact_force_n", "max_contact_force_n", "p95_contact_force_n"):
        if not fixed_metrics.get(key):
            raise RuntimeError(f"fixed episode metrics must report a positive {key}")
    if not metrics["aggregate"].get("contact_force_n", {}).get("max"):
        raise RuntimeError("aggregate metrics must include the contact_force_n block")


def _assert_exports(artifacts) -> None:
    expected_spaces = {A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA, A4_OBJ_CENTRIC_CHUNK}
    if set(artifacts.exports) != expected_spaces:
        raise RuntimeError(f"exports must cover {expected_spaces}, got {set(artifacts.exports)}")

    n_episodes = len(artifacts.episodes)
    for action_space, directory in artifacts.exports.items():
        meta = json.loads((directory / "meta.json").read_text())
        if meta["n_episodes"] != n_episodes or meta["action_space"] != action_space:
            raise RuntimeError(f"dataset meta.json is inconsistent: {directory}")

    a2_files = sorted(artifacts.exports[A2_EE_DELTA].glob("episode_*.hdf5"))
    a3_files = sorted(artifacts.exports[A3_OBJ_REL_EE_DELTA].glob("episode_*.hdf5"))
    if len(a2_files) != n_episodes or len(a3_files) != n_episodes:
        raise RuntimeError("A2/A3 exports are missing episode files")

    a2 = read_episode(a2_files[0])
    a3 = read_episode(a3_files[0])
    if a2.meta.action_space != A2_EE_DELTA or a3.meta.action_space != A3_OBJ_REL_EE_DELTA:
        raise RuntimeError("exported episodes carry wrong action-space tags")
    frame = door_frame_from_body_pose(
        np.asarray(a2.extras["door_frame_pos_w"]),
        np.asarray(a2.extras["door_frame_quat_w_xyzw"]),
    )
    for step_a2, step_a3 in zip(a2.steps, a3.steps, strict=True):
        expected = world_delta_to_frame(step_a2.action, frame)
        if not np.allclose(step_a3.action, expected, atol=1e-6):
            raise RuntimeError("A3 export does not match door-frame conversion of A2 actions")

    a1_files = sorted(artifacts.exports[A1_JOINT_DELTA].glob("episode_*.hdf5"))
    if len(a1_files) != n_episodes:
        raise RuntimeError("A1 export is missing episode files")
    a1 = read_episode(a1_files[0])
    if a1.meta.action_space != A1_JOINT_DELTA:
        raise RuntimeError("A1 export carries the wrong action-space tag")
    a1_actions = np.stack([step.action for step in a1.steps])
    if a1_actions.shape != (len(a2.steps), N_ALEX_JOINTS):
        raise RuntimeError(
            f"A1 actions must have shape (n_steps, {N_ALEX_JOINTS}), got {a1_actions.shape}"
        )
    # A1 actions are the recorded joint-position-target diffs (pre-step capture,
    # so action[t] = target[t+1] - target[t]; the last diff uses the post-loop
    # target from extras["final_joint_pos_target"]).
    targets = np.stack([step.proprio["joint_pos_target"] for step in a2.steps])
    expected_a1 = np.diff(
        np.concatenate(
            [targets, np.asarray(a2.extras["final_joint_pos_target"]).reshape(1, -1)], axis=0
        ),
        axis=0,
    )
    if not np.allclose(a1_actions, expected_a1, atol=1e-9):
        raise RuntimeError("A1 export does not match the recorded joint-target deltas")
    if not np.any(np.abs(a1_actions) > 0.0):
        raise RuntimeError("A1 actions are all zero — the arm never received a target update")

    a4_lines = (
        (artifacts.exports[A4_OBJ_CENTRIC_CHUNK] / "episodes.jsonl").read_text().splitlines()
    )
    if len(a4_lines) != n_episodes:
        raise RuntimeError("A4 export is missing episode records")
    record = json.loads(a4_lines[0])
    chunk_phases = [chunk["phase"] for chunk in record["chunks"]]
    if chunk_phases[:2] != ["approach", "align"] or "push" not in chunk_phases:
        raise RuntimeError(f"A4 chunk sequence is malformed: {chunk_phases}")


def main() -> int:
    rc = 0
    env = None
    try:
        env = _make_env()
        engine_cfg = _engine_cfg()
        max_diff = _assert_determinism(env, engine_cfg)

        artifacts = run_baseline(
            env,
            outputs_root=paths.OUTPUTS_DIR,
            datasets_root=paths.OUTPUTS_DIR / EXPERIMENT / "gate_datasets",
            experiment=EXPERIMENT,
            run_id="gate",
            n_fixed=GATE_N_FIXED,
            n_randomized=GATE_N_RANDOMIZED,
            base_seed=args.seed,
            engine_cfg=engine_cfg,
            controller_cfg=alex_fixedbase_push_cfg(),
            variation_bounds=ALEX_VARIATION_BOUNDS,
            video=args.video,
        )
        _assert_artifacts(artifacts)
        _assert_alex_episode_schema(artifacts)
        _assert_sanity(artifacts)
        _assert_exports(artifacts)

        fixed = artifacts.episodes[0]
        contact_ticks = sum(1 for s in fixed.steps if s.contact.get("sensed"))
        peak_force = max(float(s.contact["force_n"]) for s in fixed.steps)
        print(f"[gate] run_dir={artifacts.run_dir}", flush=True)
        print(
            f"[gate] fixed episode: steps={fixed.n_steps} "
            f"final_angle={math.degrees(fixed.outcome.final_door_angle):.1f} deg "
            f"success={fixed.outcome.success}",
            flush=True,
        )
        print(
            f"[gate] force contact: sensed_ticks={contact_ticks} peak_force={peak_force:.1f} N",
            flush=True,
        )
        print(
            f"[gate] randomized episode: success={artifacts.episodes[1].outcome.success} "
            f"label={artifacts.episodes[1].outcome.failure_label!r}",
            flush=True,
        )
        print(f"[determinism] max_episode_trace_diff={max_diff:.9g}", flush=True)
        print(f"[videos] status={artifacts.videos['status']}", flush=True)
        print("PASS: Alex door baseline gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: Alex door baseline gate failed.", flush=True)
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
