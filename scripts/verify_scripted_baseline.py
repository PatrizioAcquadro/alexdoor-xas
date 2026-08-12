#!/usr/bin/env python
"""Alex V2 scripted rollout and A1-A4 export verification gate.

The gate exercises the same calibrated environment, controller, data engine,
and exporter as ``run_scripted_baseline.py``. Its disposable artifacts live
under ``~/.cache/alexdoor-xas/verification/verify_scripted_baseline/``;
reusable datasets are untouched.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_scripted_baseline.py --viz none --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS Alex V2 scripted baseline gate")
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

# Runtime imports after AppLauncher.
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
from alexdoor_xas.assets.alex_v2_contract import EXPECTED_RUNTIME_JOINTS  # noqa: E402
from alexdoor_xas.data_engine import (  # noqa: E402
    DataEngineCfg,
    plan_episodes,
    run_baseline,
    run_episode,
    traces_equal,
)
from alexdoor_xas.envs.door_task.alex_v2_runtime import ALEX_V2_LIMITATIONS  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.policies.scripted import (  # noqa: E402
    alex_v2_push_cfg,
    alex_v2_variation_bounds,
)
from alexdoor_xas.recording import read_episode  # noqa: E402

EXPERIMENT = "verify_scripted_baseline"
GATE_N_FIXED = 1
GATE_N_RANDOMIZED = 1
EXPECTED_SPACES = {
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
}


def _make_env():
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    render_mode = "rgb_array" if args.video else None
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg, render_mode=render_mode).unwrapped


def _engine_cfg(env) -> DataEngineCfg:
    return DataEngineCfg(
        task=paths.ALEX_V2_TASK,
        robot=paths.ALEX_V2_ROBOT_TAG,
        limitations=ALEX_V2_LIMITATIONS,
    )


def _assert_determinism(env, engine_cfg: DataEngineCfg, controller_cfg) -> float:
    item = plan_episodes(GATE_N_FIXED, 0, args.seed)[0]
    first = run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
    second = run_episode(env, item, engine_cfg, controller_cfg=controller_cfg)
    max_diff = traces_equal(first, second, tol=args.determinism_tol)
    if not first.outcome.success:
        raise RuntimeError(
            f"fixed-start episode must succeed; termination={first.outcome.termination_reason!r} "
            f"final_angle={first.outcome.final_door_angle:.4f} rad"
        )
    return max_diff


def _assert_artifacts(artifacts) -> None:
    if len(artifacts.episodes) != GATE_N_FIXED + GATE_N_RANDOMIZED:
        raise RuntimeError("scripted gate produced the wrong episode count")
    for label, episode in zip(("fixed", "randomized"), artifacts.episodes, strict=True):
        if episode.outcome is None or not episode.outcome.success:
            termination = None if episode.outcome is None else episode.outcome.termination_reason
            raise RuntimeError(f"{label} Alex V2 episode failed: {termination!r}")
        if episode.meta.task != paths.ALEX_V2_TASK:
            raise RuntimeError(f"{label} episode carries task {episode.meta.task!r}")
        if episode.meta.robot != paths.ALEX_V2_ROBOT_TAG:
            raise RuntimeError(f"{label} episode carries robot {episode.meta.robot!r}")
        if tuple(episode.extras["engine_cfg"]["limitations"]) != tuple(ALEX_V2_LIMITATIONS):
            raise RuntimeError(f"{label} episode carries incorrect Alex limitations")

    for name, path in artifacts.plots.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"plot {name!r} was not written: {path}")
    if not artifacts.report_path.is_file():
        raise RuntimeError(f"run report was not written: {artifacts.report_path}")
    metrics_path = artifacts.run_dir / "metrics" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics["aggregate"]["n_episodes"] != len(artifacts.episodes):
        raise RuntimeError(f"metrics.json episode count mismatch: {metrics_path}")
    if artifacts.sanity is None or artifacts.sanity["n_episodes_with_errors"]:
        raise RuntimeError("Alex V2 sanity gate is missing or contains errors")


def _assert_exports(artifacts, env) -> None:
    if set(artifacts.exports) != EXPECTED_SPACES:
        raise RuntimeError(f"exports must cover {EXPECTED_SPACES}, got {set(artifacts.exports)}")

    n_episodes = len(artifacts.episodes)
    runtime_asset = env.robot_asset_provenance()
    for action_space, directory in artifacts.exports.items():
        if directory.name != paths.ALEX_V2_DATASET_VERSION:
            raise RuntimeError(
                f"dataset version is not {paths.ALEX_V2_DATASET_VERSION}: {directory}"
            )
        meta = json.loads((directory / "meta.json").read_text())
        engine_cfg = meta.get("generator", {}).get("engine_cfg", {})
        asset = meta.get("robot_asset") or {}
        if meta["n_episodes"] != n_episodes or meta["action_space"] != action_space:
            raise RuntimeError(f"dataset meta.json is inconsistent: {directory}")
        if meta["task"] != paths.ALEX_V2_TASK or meta["robot"] != paths.ALEX_V2_ROBOT_TAG:
            raise RuntimeError(f"dataset metadata is not Alex V2: {directory}")
        if tuple(engine_cfg.get("limitations", ())) != tuple(ALEX_V2_LIMITATIONS):
            raise RuntimeError(f"dataset limitations are not the Alex V2 contract: {directory}")
        for key in ("id", "sha256", "manifest_fingerprint"):
            if asset.get(key) != runtime_asset[key]:
                raise RuntimeError(f"dataset robot asset {key} differs from runtime: {directory}")

    hdf5_spaces = (A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA)
    files = {
        space: sorted(artifacts.exports[space].glob("episode_*.hdf5")) for space in hdf5_spaces
    }
    if any(len(items) != n_episodes for items in files.values()):
        raise RuntimeError("A1/A2/A3 exports are missing episode files")

    a1 = read_episode(files[A1_JOINT_DELTA][0])
    a2 = read_episode(files[A2_EE_DELTA][0])
    a3 = read_episode(files[A3_OBJ_REL_EE_DELTA][0])
    if a1.meta.action_space != A1_JOINT_DELTA:
        raise RuntimeError("A1 export carries the wrong action-space tag")
    if tuple(a1.extras["joint_names"]) != EXPECTED_RUNTIME_JOINTS:
        raise RuntimeError("A1 export does not carry the canonical Alex V2 joint order")
    if any(step.action.shape != (len(EXPECTED_RUNTIME_JOINTS),) for step in a1.steps):
        raise RuntimeError("A1 actions do not match the Alex V2 joint dimension")

    frame = door_frame_from_body_pose(
        np.asarray(a2.extras["door_frame_pos_w"]),
        np.asarray(a2.extras["door_frame_quat_w_xyzw"]),
    )
    for step_a2, step_a3 in zip(a2.steps, a3.steps, strict=True):
        expected = world_delta_to_frame(step_a2.action, frame)
        if not np.allclose(step_a3.action, expected, atol=1e-6):
            raise RuntimeError("A3 export does not match door-frame conversion of A2 actions")

    a4_lines = (artifacts.exports[A4_OBJ_CENTRIC_CHUNK] / "episodes.jsonl").read_text().splitlines()
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
        engine_cfg = _engine_cfg(env)
        calibration = env.alex_v2_calibration()
        controller_cfg = alex_v2_push_cfg(calibration)
        variation_bounds = alex_v2_variation_bounds(calibration)
        max_diff = _assert_determinism(env, engine_cfg, controller_cfg)

        artifacts = run_baseline(
            env,
            outputs_root=paths.VERIFICATION_CACHE_DIR,
            datasets_root=paths.VERIFICATION_CACHE_DIR / EXPERIMENT / "gate_datasets",
            experiment=EXPERIMENT,
            run_id="gate",
            n_fixed=GATE_N_FIXED,
            n_randomized=GATE_N_RANDOMIZED,
            base_seed=args.seed,
            engine_cfg=engine_cfg,
            controller_cfg=controller_cfg,
            variation_bounds=variation_bounds,
            video=args.video,
            dataset_version=paths.ALEX_V2_DATASET_VERSION,
        )
        _assert_artifacts(artifacts)
        _assert_exports(artifacts, env)

        fixed, randomized = artifacts.episodes
        print(f"[gate] run_dir={artifacts.run_dir}", flush=True)
        print(
            f"[gate] fixed: steps={fixed.n_steps} "
            f"final={math.degrees(fixed.outcome.final_door_angle):.1f} deg; "
            f"randomized: steps={randomized.n_steps} "
            f"final={math.degrees(randomized.outcome.final_door_angle):.1f} deg",
            flush=True,
        )
        print(f"[determinism] max_episode_trace_diff={max_diff:.9g}", flush=True)
        print(f"[export] version={paths.ALEX_V2_DATASET_VERSION} spaces={sorted(EXPECTED_SPACES)}")
        print("PASS: Alex V2 scripted baseline gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: Alex V2 scripted baseline gate failed.", flush=True)
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
                simulation_app.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
    return rc


if __name__ == "__main__":
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
