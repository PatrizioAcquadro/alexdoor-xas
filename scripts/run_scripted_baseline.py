#!/usr/bin/env python
"""Phase 2/2.5 data engine CLI: scripted door-push episodes -> datasets + artifacts.

Rolls out the deterministic scripted controller in the door-push env, records
episodes to the schema, exports A2/A3/A4 datasets under ``datasets/``, and
writes metrics/plots/videos/report under ``outputs/<experiment>/<run_id>/``.
``--robot alex`` runs the fixed-base Alex humanoid (diff-IK right arm, force
contact sensing) instead of the Phase 2 proxy sphere; Alex datasets go to
``datasets/door_push_alex/`` so the frozen proxy datasets are never replaced.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/run_scripted_baseline.py --viz none --device cpu \
        --episodes 5 --randomized 3 [--robot alex]

Add ``--video --enable_cameras`` to also record per-episode rollout videos.
Run settings can also be supplied as Hydra-style overrides, for example
``run.robot=alex run.episodes=5 run.randomized=3``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from datetime import UTC, datetime

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
    choices=("proxy", "alex"),
    default=None,
    help="Executor: the Phase 2 proxy sphere (default) or the fixed-base Alex humanoid.",
)
parser.add_argument("--episodes", type=int, default=None, help="Fixed-start episodes.")
parser.add_argument("--randomized", type=int, default=None, help="Seeded randomized episodes.")
parser.add_argument("--seed", type=int, default=None, help="Base seed for the episode plan.")
parser.add_argument(
    "--experiment",
    type=str,
    default=None,
    help="outputs/<experiment>/ name (default: scripted_door_push or alex_door_push).",
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
from alexdoor_xas.data_engine import DataEngineCfg, run_baseline  # noqa: E402
from alexdoor_xas.data_engine.generate import ALEX_LIMITATIONS  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_env_cfg import (  # noqa: E402
    ALEX_ROBOT_TAG,
    DoorPushAlexEnvCfg,
)
from alexdoor_xas.envs.door_task.door_push_env_cfg import DoorPushEnvCfg  # noqa: E402
from alexdoor_xas.policies.scripted import (  # noqa: E402
    ALEX_VARIATION_BOUNDS,
    DoorPushControllerCfg,
    alex_fixedbase_push_cfg,
)

DEFAULT_EXPERIMENTS = {"proxy": "scripted_door_push", "alex": "alex_door_push"}


def _make_env():
    render_mode = "rgb_array" if run_config.run.video else None
    if run_config.run.robot == "alex":
        cfg = DoorPushAlexEnvCfg()
        env_id = door_task.DOOR_PUSH_ALEX_ENV_ID
    else:
        cfg = DoorPushEnvCfg()
        env_id = door_task.DOOR_PUSH_ENV_ID
    cfg.seed = run_config.run.seed
    cfg.sim.device = args.device
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
        if run_config.run.robot == "alex":
            # Distinct task tag => Alex datasets land in datasets/door_push_alex/
            # and never replace the frozen Phase 2 proxy datasets.
            engine_cfg = DataEngineCfg(
                task="door_push_alex",
                robot=ALEX_ROBOT_TAG,
                limitations=ALEX_LIMITATIONS,
                success_angle_rad=math.radians(run_config.run.success_angle_deg),
                max_ticks=run_config.run.max_ticks,
            )
            controller_cfg = apply_controller_overrides(
                alex_fixedbase_push_cfg(), run_config.controller_overrides
            )
            variation_bounds = ALEX_VARIATION_BOUNDS
        else:
            engine_cfg = DataEngineCfg(
                success_angle_rad=math.radians(run_config.run.success_angle_deg),
                max_ticks=run_config.run.max_ticks,
            )
            controller_cfg = (
                apply_controller_overrides(
                    DoorPushControllerCfg(), run_config.controller_overrides
                )
                if run_config.controller_overrides
                else None
            )
            variation_bounds = None
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
        )

        print(f"[run] dir={artifacts.run_dir}", flush=True)
        for action_space, directory in artifacts.exports.items():
            print(f"[export] {action_space} -> {directory}", flush=True)
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
