#!/usr/bin/env python
"""Closed-loop ACT evaluation through adapter-v1 on the Alex env (Phase 3.2).

Loads a trained ACT checkpoint, rebuilds the policy (norm stats embedded), and
rolls it out on ``AlexDoor-DoorPush-Alex-v0`` through the adapter matching the
checkpoint's action space (A2: world-frame deltas, A3: door-frame deltas).
Runs ``rollout.episodes_fixed`` fixed-reset rollouts (deterministic headless
physics makes this block a determinism probe) plus
``rollout.episodes_randomized`` rollouts with seeded EE start-offset
variations on held-out seeds. Writes per-rollout rows and aggregates
(success vs. the door-angle threshold, adapter accept/correct/reject counts)
to ``metrics/act_eval.json`` next to the checkpoint, with the scripted
baseline's aggregate embedded for side-by-side comparison when
``rollout.reference_metrics`` is set::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py \
        --viz none --device cpu \
        rollout.checkpoint=outputs/act_door_push/<run_id>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

# -- AppLauncher must be configured before any other Isaac import; the ACT
# config layer is torch/Isaac-free, so it resolves first (Hydra precedent).
from isaaclab.app import AppLauncher

from alexdoor_xas.policies.act import ActConfigError, load_act_config

parser = argparse.ArgumentParser(description="AlexDoor-XAS ACT closed-loop evaluation")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Trained ACT checkpoint (.pt) to evaluate."
)
parser.add_argument(
    "--reference-metrics",
    type=str,
    default=None,
    help="Scripted-baseline metrics.json to embed for comparison.",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting; useful for debugging Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_overrides = parser.parse_known_args()

try:
    act_cfg = load_act_config(
        hydra_overrides,
        cli_overrides={
            "rollout.checkpoint": args.checkpoint,
            "rollout.reference_metrics": args.reference_metrics,
        },
    )
except ActConfigError as error:
    parser.error(str(error))
if act_cfg.rollout.checkpoint is None:
    parser.error("rollout.checkpoint is required (--checkpoint or rollout.checkpoint=...)")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA  # noqa: E402
from alexdoor_xas.adapters import (  # noqa: E402
    A2Adapter,
    A3Adapter,
    limits_for_robot,
    read_door_frame,
    rollout_chunks,
)
from alexdoor_xas.data_engine import apply_start_offset, plan_episodes  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_env_cfg import (  # noqa: E402
    ALEX_ROBOT_TAG,
    DoorPushAlexEnvCfg,
)
from alexdoor_xas.policies.act.policy import (  # noqa: E402
    ActPolicy,
    act_chunk_source,
    stop_on_hinge_angle,
)
from alexdoor_xas.policies.scripted import ALEX_VARIATION_BOUNDS  # noqa: E402
from alexdoor_xas.tracking import load_wandb_config, start_wandb_run  # noqa: E402


def _make_env():
    cfg = DoorPushAlexEnvCfg()
    cfg.seed = act_cfg.rollout.base_seed
    cfg.sim.device = args.device
    return gym.make(door_task.DOOR_PUSH_ALEX_ENV_ID, cfg=cfg).unwrapped


def _fresh_adapter(action_space: str):
    a2 = A2Adapter(limits_for_robot(ALEX_ROBOT_TAG))
    if action_space == A2_EE_DELTA:
        return a2
    if action_space == A3_OBJ_REL_EE_DELTA:
        return A3Adapter(a2)
    raise ValueError(f"no adapter path for action space {action_space!r}")


def _run_rollout(env, policy, seed: int, variation, success_angle_rad: float) -> dict:
    env.reset(seed=seed)
    if variation is not None:
        apply_start_offset(env, read_door_frame(env), variation)
    adapter = _fresh_adapter(policy.action_space)
    # Rollouts end at the first chunk boundary past the success angle: the
    # demos terminate with the FSM, so post-task extrapolation is unbounded
    # (a wandering arm can knock the open door shut again).
    source = stop_on_hinge_angle(
        act_chunk_source(
            policy,
            env,
            temporal_ensemble=act_cfg.rollout.temporal_ensemble,
            ensemble_m=act_cfg.rollout.ensemble_m,
        ),
        success_angle_rad,
    )
    result = rollout_chunks(env, source, adapter, max_ticks=act_cfg.rollout.max_ticks)
    return {
        "seed": seed,
        "randomized": variation is not None,
        "success": bool(result.final_angle_rad >= success_angle_rad),
        "initial_angle_rad": result.initial_angle_rad,
        "final_angle_rad": result.final_angle_rad,
        "door_angle_change_rad": result.door_angle_change_rad,
        "n_ticks": result.n_ticks,
        "n_accepted": result.log.n_accepted,
        "n_corrected": result.log.n_corrected,
        "n_rejected": result.log.n_rejected,
        "notes": result.notes,
    }


def _aggregate(rows: list[dict]) -> dict:
    finals = [row["final_angle_rad"] for row in rows]
    fixed_finals = [row["final_angle_rad"] for row in rows if not row["randomized"]]
    return {
        "n_rollouts": len(rows),
        "n_fixed": sum(1 for row in rows if not row["randomized"]),
        "n_randomized": sum(1 for row in rows if row["randomized"]),
        "n_success": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "final_angle_rad": {
            "mean": float(np.mean(finals)),
            "min": float(np.min(finals)),
            "max": float(np.max(finals)),
        },
        "mean_ticks": float(np.mean([row["n_ticks"] for row in rows])),
        "adapter": {
            "n_accepted": sum(row["n_accepted"] for row in rows),
            "n_corrected": sum(row["n_corrected"] for row in rows),
            "n_rejected": sum(row["n_rejected"] for row in rows),
        },
        "fixed_determinism_spread_rad": (
            float(np.max(fixed_finals) - np.min(fixed_finals)) if fixed_finals else None
        ),
    }


def _reference_aggregate() -> dict | None:
    if act_cfg.rollout.reference_metrics is None:
        return None
    path = paths.REPO_ROOT / act_cfg.rollout.reference_metrics
    payload = json.loads(path.read_text())
    return {"path": str(path), "aggregate": payload.get("aggregate", payload)}


def main() -> int:
    rc = 0
    env = None
    try:
        checkpoint_path = paths.REPO_ROOT / act_cfg.rollout.checkpoint
        policy = ActPolicy.from_checkpoint(checkpoint_path)
        run_dir = checkpoint_path.parent.parent  # outputs/<experiment>/<run_id>/
        success_angle_rad = math.radians(act_cfg.rollout.success_angle_deg)
        print(
            f"[eval_act] checkpoint={checkpoint_path} space={policy.action_space} "
            f"obs={policy.obs_preset} chunk={policy.chunk_size} "
            f"ensemble={act_cfg.rollout.temporal_ensemble}",
            flush=True,
        )

        env = _make_env()
        rows: list[dict] = []
        for i in range(act_cfg.rollout.episodes_fixed):
            row = _run_rollout(
                env, policy, act_cfg.rollout.base_seed + i, None, success_angle_rad
            )
            rows.append(row)
            print(f"[fixed {i}] {_row_line(row)}", flush=True)
        plan = plan_episodes(
            0,
            act_cfg.rollout.episodes_randomized,
            act_cfg.rollout.base_seed + act_cfg.rollout.episodes_fixed,
            ALEX_VARIATION_BOUNDS,
        )
        for i, item in enumerate(plan):
            row = _run_rollout(env, policy, item.seed, item.variation, success_angle_rad)
            rows.append(row)
            print(f"[rand {i}] {_row_line(row)}", flush=True)

        aggregate = _aggregate(rows)
        payload = {
            "checkpoint": str(checkpoint_path),
            "action_space": policy.action_space,
            "obs_preset": policy.obs_preset,
            "chunk_size": policy.chunk_size,
            "temporal_ensemble": act_cfg.rollout.temporal_ensemble,
            "max_ticks": act_cfg.rollout.max_ticks,
            "success_angle_deg": act_cfg.rollout.success_angle_deg,
            "base_seed": act_cfg.rollout.base_seed,
            "rollouts": rows,
            "aggregate": aggregate,
            "scripted_reference": _reference_aggregate(),
        }
        metrics_path = run_dir / "metrics" / "act_eval.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

        wandb_cfg = load_wandb_config(
            overrides={
                "group": act_cfg.run.experiment,
                "name": f"{run_dir.name}_eval",
                "job_type": "eval",
                **act_cfg.wandb_overrides,
            }
        )
        with start_wandb_run(wandb_cfg, config=payload | {"rollouts": None}) as run:
            run.log(
                {
                    "eval/success_rate": aggregate["success_rate"],
                    "eval/final_angle_mean_rad": aggregate["final_angle_rad"]["mean"],
                    "eval/n_corrected": aggregate["adapter"]["n_corrected"],
                    "eval/n_rejected": aggregate["adapter"]["n_rejected"],
                }
            )

        print(
            f"[eval_act] success_rate={aggregate['success_rate']:.2f} "
            f"({aggregate['n_success']}/{aggregate['n_rollouts']}) "
            f"final_angle_mean={math.degrees(aggregate['final_angle_rad']['mean']):.1f} deg "
            f"adapter accepted/corrected/rejected="
            f"{aggregate['adapter']['n_accepted']}/{aggregate['adapter']['n_corrected']}/"
            f"{aggregate['adapter']['n_rejected']}",
            flush=True,
        )
        print(f"[eval_act] metrics: {metrics_path}", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: ACT evaluation failed.", flush=True)
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


def _row_line(row: dict) -> str:
    return (
        f"seed={row['seed']} success={row['success']} "
        f"final={math.degrees(row['final_angle_rad']):.1f} deg ticks={row['n_ticks']} "
        f"a/c/r={row['n_accepted']}/{row['n_corrected']}/{row['n_rejected']}"
    )


if __name__ == "__main__":
    # os._exit avoids Kit shutdown masking the exit code.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
