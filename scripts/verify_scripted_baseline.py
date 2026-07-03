#!/usr/bin/env python
"""Phase 2 verification gate: scripted rollout + deterministic data export.

Runs a small fixed configuration through the same code path as
``scripts/run_scripted_baseline.py`` and fails non-zero unless:

- the fixed-start scripted episode succeeds (door past the success threshold),
- a repeated rollout of the same seed reproduces identical traces,
- A2/A3/A4 dataset exports exist and satisfy the episode schema
  (including the A2 -> A3 door-frame relabeling),
- metrics, plots, and the run report are written.

Gate artifacts go under ``outputs/verify_scripted_baseline/`` (both the run
artifacts and the gate's dataset exports — the reusable ``datasets/`` tree is
left to real engine runs).

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_scripted_baseline.py --viz none --device cpu
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

parser = argparse.ArgumentParser(description="AlexDoor-XAS scripted baseline verification gate")
parser.add_argument("--seed", type=int, default=0, help="Base seed for the gate episodes.")
parser.add_argument(
    "--determinism-tol",
    type=float,
    default=1e-6,
    help="Maximum allowed absolute difference between repeated episode traces.",
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
from alexdoor_xas.envs.door_task.door_push_env_cfg import DoorPushEnvCfg  # noqa: E402
from alexdoor_xas.recording import read_episode  # noqa: E402

EXPERIMENT = "verify_scripted_baseline"
GATE_N_FIXED = 1
GATE_N_RANDOMIZED = 1


def _make_env():
    cfg = DoorPushEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    render_mode = "rgb_array" if args.video else None
    return gym.make(door_task.DOOR_PUSH_ENV_ID, cfg=cfg, render_mode=render_mode).unwrapped


def _assert_determinism(env, engine_cfg: DataEngineCfg) -> float:
    item = plan_episodes(GATE_N_FIXED, 0, args.seed)[0]
    first = run_episode(env, item, engine_cfg)
    second = run_episode(env, item, engine_cfg)
    max_diff = traces_equal(first, second, tol=args.determinism_tol)
    if not first.outcome.success:
        raise RuntimeError(
            f"fixed-start episode must succeed; failure={first.outcome.failure_label!r} "
            f"final_angle={first.outcome.final_door_angle:.4f} rad"
        )
    return max_diff


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


def _assert_exports(artifacts) -> None:
    expected_spaces = {A2_EE_DELTA, A3_OBJ_REL_EE_DELTA, A4_OBJ_CENTRIC_CHUNK}
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
        engine_cfg = DataEngineCfg()
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
            video=args.video,
        )
        _assert_artifacts(artifacts)
        _assert_exports(artifacts)

        fixed = artifacts.episodes[0]
        print(f"[gate] run_dir={artifacts.run_dir}", flush=True)
        print(
            f"[gate] fixed episode: steps={fixed.n_steps} "
            f"final_angle={math.degrees(fixed.outcome.final_door_angle):.1f} deg "
            f"success={fixed.outcome.success}",
            flush=True,
        )
        print(
            f"[gate] randomized episode: success={artifacts.episodes[1].outcome.success} "
            f"label={artifacts.episodes[1].outcome.failure_label!r}",
            flush=True,
        )
        print(f"[determinism] max_episode_trace_diff={max_diff:.9g}", flush=True)
        print(f"[videos] status={artifacts.videos['status']}", flush=True)
        print("PASS: scripted baseline gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: scripted baseline gate failed.", flush=True)
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
