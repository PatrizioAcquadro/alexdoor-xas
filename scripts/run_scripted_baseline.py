#!/usr/bin/env python
"""Scripted door-push data engine CLI for the proxy and calibrated Alex V2.

Rolls out the deterministic scripted controller in the door-push env, records
episodes to the schema, exports A2/A3/A4 datasets under ``datasets/``, and
writes metrics/plots/videos/report under ``outputs/<experiment>/<run_id>/``.
``--robot alex_v2`` runs the calibrated fixed-base V2 humanoid instead of the
proxy sphere. V2 datasets use the frozen ``door_push_alex_v2`` task identity.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/run_scripted_baseline.py --viz none --device cpu \
        --episodes 5 --randomized 3 [--robot alex_v2]

Add ``--video --enable_cameras`` to also record per-episode rollout videos.
Run settings can also be supplied as Hydra-style overrides, for example
``run.robot=alex_v2 run.episodes=5 run.randomized=3``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# -- AppLauncher must be configured before any other Isaac import.
from isaaclab.app import AppLauncher

from alexdoor_xas.scripted_baseline_config import (
    ScriptedBaselineConfigError,
    apply_controller_overrides,
    load_scripted_baseline_config,
)

parser = argparse.ArgumentParser(description="AlexDoor-XAS scripted door-push data engine")
parser.add_argument(
    "--robot",
    choices=("proxy", "alex_v2"),
    default=None,
    help="Executor: the proxy sphere (default) or calibrated fixed-base Alex V2.",
)
parser.add_argument(
    "--randomized-seed-plan",
    type=Path,
    default=None,
    help="JSON list of explicit randomized seeds (scale candidate generation only).",
)
parser.add_argument(
    "--candidate-pool",
    action="store_true",
    help=(
        "Preserve failed candidate evidence for later fail-closed selection; "
        "requires --no-export."
    ),
)
parser.add_argument("--episodes", type=int, default=None, help="Fixed-start episodes.")
parser.add_argument("--randomized", type=int, default=None, help="Seeded randomized episodes.")
parser.add_argument("--seed", type=int, default=None, help="Base seed for the episode plan.")
parser.add_argument(
    "--experiment",
    type=str,
    default=None,
    help="outputs/<experiment>/ name (default: scripted_door_push or alex_v2_door_push).",
)
parser.add_argument(
    "--run-id", type=str, default=None, help="Run id (default: <UTC date>_seed<seed>)."
)
parser.add_argument(
    "--success-angle-deg", type=float, default=None, help="Success threshold in degrees."
)
parser.add_argument("--max-ticks", type=int, default=None, help="Per-episode tick budget.")
parser.add_argument(
    "--video",
    action="store_true",
    default=None,
    help="Record rollout videos (requires --enable_cameras for offscreen rendering).",
)
parser.add_argument(
    "--no-export",
    action="store_true",
    default=None,
    help="Record the run under outputs/ only; never write datasets/ (multi-pose passes).",
)
parser.add_argument(
    "--door-yaw-deg",
    type=float,
    default=None,
    help="Door-task pose variation: yaw about the hinge axis, degrees (default 0).",
)
parser.add_argument(
    "--door-offset-x",
    type=float,
    default=None,
    help="Door-task pose variation: world-x translation in meters (default 0).",
)
parser.add_argument(
    "--door-offset-y",
    type=float,
    default=None,
    help="Door-task pose variation: world-y translation in meters (default 0).",
)
parser.add_argument(
    "--door-pose-id",
    type=str,
    default=None,
    help="Pose label recorded into episodes and manifests (e.g. D0).",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    default=None,
    help="Call SimulationApp.close() before exiting; useful for debugging Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_overrides = parser.parse_known_args()

try:
    run_config = load_scripted_baseline_config(
        hydra_overrides,
        cli_overrides={
            "robot": args.robot,
            "episodes": args.episodes,
            "randomized": args.randomized,
            "seed": args.seed,
            "experiment": args.experiment,
            "run_id": args.run_id,
            "success_angle_deg": args.success_angle_deg,
            "max_ticks": args.max_ticks,
            "video": args.video,
            "clean_shutdown": args.clean_shutdown,
            "export": False if args.no_export else None,
            "door_yaw_deg": args.door_yaw_deg,
            "door_offset_x": args.door_offset_x,
            "door_offset_y": args.door_offset_y,
            "door_pose_id": args.door_pose_id,
        },
    )
except ScriptedBaselineConfigError as error:
    parser.error(str(error))

if run_config.run.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.data_engine import (  # noqa: E402
    DataEngineCfg,
    plan_randomized_seeds,
    run_baseline,
)
from alexdoor_xas.envs.door_task.alex_v2_runtime import ALEX_V2_LIMITATIONS  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_ROBOT_TAG,
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.envs.door_task.door_push_env_cfg import DoorPushEnvCfg  # noqa: E402
from alexdoor_xas.policies.scripted import (  # noqa: E402
    DoorPushControllerCfg,
    alex_v2_push_cfg,
    alex_v2_variation_bounds,
)

DEFAULT_EXPERIMENTS = {
    "proxy": "scripted_door_push",
    "alex_v2": "alex_v2_door_push",
}


def _make_env():
    render_mode = "rgb_array" if run_config.run.video else None
    if run_config.run.robot == "alex_v2":
        cfg = DoorPushAlexV2EnvCfg()
        env_id = door_task.DOOR_PUSH_ALEX_V2_ENV_ID
    else:
        cfg = DoorPushEnvCfg()
        env_id = door_task.DOOR_PUSH_ENV_ID
    cfg.seed = run_config.run.seed
    cfg.sim.device = args.device
    cfg.door_yaw_rad = math.radians(run_config.run.door_yaw_deg)
    cfg.door_offset_xy = (run_config.run.door_offset_x, run_config.run.door_offset_y)
    return gym.make(env_id, cfg=cfg, render_mode=render_mode).unwrapped


def main() -> int:
    rc = 0
    env = None
    try:
        env = _make_env()
        run_id = run_config.run.run_id or (
            f"{datetime.now(UTC).date().isoformat()}_seed{run_config.run.seed}"
        )
        experiment = run_config.run.experiment or DEFAULT_EXPERIMENTS[run_config.run.robot]
        door_pose_kwargs = {
            "door_pose_id": run_config.run.door_pose_id,
            "door_yaw_rad": math.radians(run_config.run.door_yaw_deg),
            "door_offset_xy": (run_config.run.door_offset_x, run_config.run.door_offset_y),
            "scene": str(env.cfg.door_task_scene.spawn.usd_path),
        }
        if run_config.run.robot == "alex_v2":
            calibration = env.alex_v2_calibration()
            engine_cfg = DataEngineCfg(
                task=paths.ALEX_V2_TASK,
                robot=ALEX_V2_ROBOT_TAG,
                limitations=ALEX_V2_LIMITATIONS,
                success_angle_rad=math.radians(run_config.run.success_angle_deg),
                max_ticks=run_config.run.max_ticks,
                **door_pose_kwargs,
            )
            controller_cfg = apply_controller_overrides(
                alex_v2_push_cfg(calibration), run_config.controller_overrides
            )
            variation_bounds = alex_v2_variation_bounds(calibration)
        else:
            engine_cfg = DataEngineCfg(
                success_angle_rad=math.radians(run_config.run.success_angle_deg),
                max_ticks=run_config.run.max_ticks,
                **door_pose_kwargs,
            )
            controller_cfg = (
                apply_controller_overrides(
                    DoorPushControllerCfg(), run_config.controller_overrides
                )
                if run_config.controller_overrides
                else None
            )
            variation_bounds = None
        explicit_plan = None
        if args.randomized_seed_plan is not None:
            import json

            seed_payload = json.loads(args.randomized_seed_plan.read_text())
            if not isinstance(seed_payload, list):
                raise ValueError("--randomized-seed-plan must contain a JSON list")
            explicit_plan = plan_randomized_seeds(seed_payload, variation_bounds)
        artifacts = run_baseline(
            env,
            outputs_root=paths.OUTPUTS_DIR,
            datasets_root=paths.DATASETS_DIR,
            experiment=experiment,
            run_id=run_id,
            n_fixed=run_config.run.episodes,
            n_randomized=run_config.run.randomized,
            base_seed=run_config.run.seed,
            engine_cfg=engine_cfg,
            controller_cfg=controller_cfg,
            variation_bounds=variation_bounds,
            video=run_config.run.video,
            export=run_config.run.export,
            episode_plan=explicit_plan,
            preserve_candidate_failures=args.candidate_pool,
        )

        print(f"[run] dir={artifacts.run_dir}", flush=True)
        for action_space, directory in artifacts.exports.items():
            print(f"[export] {action_space} -> {directory}", flush=True)
        if not artifacts.exports:
            print("[export] skipped (run.export=false); episodes stay under outputs/", flush=True)
        if artifacts.sanity is not None:
            print(
                f"[sanity] episodes_checked={artifacts.sanity['n_episodes_checked']} "
                f"warnings={artifacts.sanity['n_episodes_with_warnings']} "
                f"errors={artifacts.sanity['n_episodes_with_errors']}",
                flush=True,
            )
        agg = artifacts.aggregate
        print(
            f"[metrics] episodes={agg['n_episodes']} "
            f"(fixed={agg['n_fixed']} randomized={agg['n_randomized']}) "
            f"success_rate={agg['success_rate']:.2f} "
            f"final_angle_mean={math.degrees(agg['final_door_angle_rad']['mean']):.1f} deg",
            flush=True,
        )
        if agg["failure_labels"]:
            print(f"[metrics] failures={agg['failure_labels']}", flush=True)
        print(f"[videos] status={artifacts.videos['status']}", flush=True)
        for video in artifacts.videos["files"]:
            print(f"[videos] {video}", flush=True)
        print(f"[report] {artifacts.report_path}", flush=True)
        print("DONE: scripted baseline run complete.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: scripted baseline run failed.", flush=True)
        rc = 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
        if run_config.run.clean_shutdown:
            try:
                print("[shutdown] closing SimulationApp (--clean-shutdown)", flush=True)
                simulation_app.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = 1 if rc == 0 else rc
    return rc


if __name__ == "__main__":
    # os._exit avoids Kit shutdown masking the exit code.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
