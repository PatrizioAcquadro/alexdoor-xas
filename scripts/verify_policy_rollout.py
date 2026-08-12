#!/usr/bin/env python
"""Verify an ACT or Diffusion checkpoint through adapter-v1 in Isaac Lab.

The selected policy and ``DoorPushAlexV2Env`` run on the same ``--device``.
For A2 (required) and A3 (optional), the gate executes one fixed and one
randomized rollout, requires complete adapter logs, and requires the fixed
rollout to pass the door-angle success threshold. Artifacts are written to
``~/.cache/alexdoor-xas/verification/verify_policy_rollout/<policy>/gate/``.

Example::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_policy_rollout.py --policy act --viz none --device cuda:0 \
        --checkpoint-a2 outputs/door_push_alex_v2/act/<run_id>/checkpoints/best.pt \
        --checkpoint-a3 outputs/door_push_alex_v2/act/<run_id>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS policy rollout gate")
parser.add_argument("--policy", choices=("act", "diffusion"), required=True)
parser.add_argument("--checkpoint-a2", required=True, help="A2 checkpoint (.pt).")
parser.add_argument("--checkpoint-a3", default=None, help="Optional A3 checkpoint (.pt).")
parser.add_argument("--seed", type=int, default=0, help="Fixed-rollout reset seed.")
parser.add_argument("--max-ticks", type=int, default=600, help="Per-rollout tick budget.")
parser.add_argument(
    "--sampler",
    choices=("ddpm", "ddim"),
    default=None,
    help="Diffusion only (default: ddpm).",
)
parser.add_argument(
    "--inference-steps",
    type=int,
    default=None,
    help="Diffusion only (default: 100).",
)
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="Diffusion only; sampled-chunk prefix to execute (default: 8).",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_diffusion_options = (args.sampler, args.inference_steps, args.n_action_steps)
if args.policy == "act" and any(value is not None for value in _diffusion_options):
    parser.error("--sampler, --inference-steps, and --n-action-steps are Diffusion-only")
if args.policy == "diffusion":
    args.sampler = args.sampler or "ddpm"
    args.inference_steps = args.inference_steps or 100
    args.n_action_steps = args.n_action_steps or 8

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Runtime imports after AppLauncher.
import gymnasium as gym  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas import paths  # noqa: E402
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA  # noqa: E402
from alexdoor_xas.adapters.a2 import A2Adapter  # noqa: E402
from alexdoor_xas.adapters.a3 import A3Adapter  # noqa: E402
from alexdoor_xas.adapters.limits import limits_for_robot  # noqa: E402
from alexdoor_xas.adapters.rollout import read_door_frame, rollout_chunks  # noqa: E402
from alexdoor_xas.assets.alex_v2_contract import RobotAssetRef  # noqa: E402
from alexdoor_xas.data_engine import (  # noqa: E402
    DEFAULT_SUCCESS_ANGLE_RAD,
    apply_start_offset,
    plan_episodes,
)
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    ALEX_V2_ROBOT_TAG,
    DoorPushAlexV2EnvCfg,
)
from alexdoor_xas.policies.act.policy import ActPolicy, act_chunk_source  # noqa: E402
from alexdoor_xas.policies.diffusion.policy import (  # noqa: E402
    DiffusionPolicy,
    diffusion_chunk_source,
)
from alexdoor_xas.policies.scripted import alex_v2_variation_bounds  # noqa: E402

EXPECTED_SPACES = {"a2": A2_EE_DELTA, "a3": A3_OBJ_REL_EE_DELTA}


def _load_act(checkpoint, runtime_asset: RobotAssetRef):
    return ActPolicy.from_checkpoint(
        paths.REPO_ROOT / checkpoint,
        device=args.device,
        runtime_asset=runtime_asset,
    )


def _load_diffusion(checkpoint, runtime_asset: RobotAssetRef):
    return DiffusionPolicy.from_checkpoint(
        paths.REPO_ROOT / checkpoint,
        device=args.device,
        sampler=args.sampler,
        num_inference_steps=args.inference_steps,
        runtime_asset=runtime_asset,
    )


def _act_chunks(policy, env):
    return act_chunk_source(policy, env)


def _diffusion_chunks(policy, env):
    return diffusion_chunk_source(policy, env, n_action_steps=args.n_action_steps)


def _no_seed(policy, seed: int) -> None:
    del policy, seed


def _seed_diffusion(policy, seed: int) -> None:
    policy.seed(seed)


POLICY_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "act": {"load": _load_act, "chunks": _act_chunks, "seed": _no_seed},
    "diffusion": {
        "load": _load_diffusion,
        "chunks": _diffusion_chunks,
        "seed": _seed_diffusion,
    },
}


def _make_env():
    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    return gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped


def _adapter_for(action_space: str, env):
    center_w = env.shoulder_position_world_m()[0].detach().cpu().numpy()
    limits = limits_for_robot(
        ALEX_V2_ROBOT_TAG,
        calibration=env.alex_v2_calibration(),
        workspace_center_w=center_w,
    )
    a2 = A2Adapter(limits, contact_entry_shaping=True)
    return a2 if action_space == A2_EE_DELTA else A3Adapter(a2)


def _rollout(env, policy, seed: int, variation) -> dict[str, Any]:
    env.reset(seed=seed)
    if variation is not None:
        apply_start_offset(env, read_door_frame(env), variation)
    adapter = _adapter_for(policy.action_space, env)
    runtime = POLICY_REGISTRY[args.policy]
    runtime["seed"](policy, seed)
    result = rollout_chunks(
        env,
        runtime["chunks"](policy, env),
        adapter,
        max_ticks=args.max_ticks,
        success_angle_rad=DEFAULT_SUCCESS_ANGLE_RAD,
    )
    if result.environment_terminated or result.environment_truncated:
        raise RuntimeError(f"rollout hit env truncation at tick {result.n_ticks}")
    if result.n_ticks == 0 or result.n_ticks > args.max_ticks:
        raise RuntimeError(f"rollout ran {result.n_ticks} ticks (budget {args.max_ticks})")
    if len(result.decisions_per_tick) != result.n_ticks:
        raise RuntimeError(
            f"{len(result.decisions_per_tick)} decisions logged for {result.n_ticks} ticks"
        )
    if not (math.isfinite(result.final_angle_rad) and math.isfinite(result.initial_angle_rad)):
        raise RuntimeError("rollout produced non-finite door angles")
    return {
        "policy": args.policy,
        "device": args.device,
        "seed": seed,
        "randomized": variation is not None,
        "robot_compatibility_label": policy.robot_compatibility_label,
        "final_angle_rad": result.final_angle_rad,
        "door_angle_change_rad": result.door_angle_change_rad,
        "n_ticks": result.n_ticks,
        "n_accepted": result.log.n_accepted,
        "n_corrected": result.log.n_corrected,
        "n_rejected": result.log.n_rejected,
        "log": result.log.to_dict(),
    }


def _check_checkpoint(env, label: str, checkpoint: str, out_dir) -> None:
    runtime_asset = RobotAssetRef.from_dict(env.robot_asset_provenance())
    policy = POLICY_REGISTRY[args.policy]["load"](checkpoint, runtime_asset)
    expected_space = EXPECTED_SPACES[label]
    if policy.action_space != expected_space:
        raise RuntimeError(
            f"--checkpoint-{label} is a {policy.action_space} model, expected {expected_space}"
        )

    fixed = _rollout(env, policy, args.seed, None)
    (out_dir / f"{label}_fixed.json").write_text(json.dumps(fixed, indent=2) + "\n")
    if fixed["final_angle_rad"] < DEFAULT_SUCCESS_ANGLE_RAD:
        raise RuntimeError(
            f"{label}: fixed rollout final angle "
            f"{math.degrees(fixed['final_angle_rad']):.1f} deg is below "
            f"{math.degrees(DEFAULT_SUCCESS_ANGLE_RAD):.1f} deg"
        )

    bounds = alex_v2_variation_bounds(env.alex_v2_calibration())
    item = plan_episodes(0, 1, args.seed + 1, bounds)[0]
    randomized = _rollout(env, policy, item.seed, item.variation)
    (out_dir / f"{label}_randomized.json").write_text(json.dumps(randomized, indent=2) + "\n")
    print(
        f"[{label}] fixed={math.degrees(fixed['final_angle_rad']):.1f} deg/"
        f"{fixed['n_ticks']} ticks a/c/r={fixed['n_accepted']}/{fixed['n_corrected']}/"
        f"{fixed['n_rejected']} randomized="
        f"{math.degrees(randomized['final_angle_rad']):.1f} deg/"
        f"{randomized['n_ticks']} ticks",
        flush=True,
    )


def main() -> int:
    rc = 0
    env = None
    try:
        out_dir = paths.VERIFICATION_CACHE_DIR / "verify_policy_rollout" / args.policy / "gate"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = _make_env()
        _check_checkpoint(env, "a2", args.checkpoint_a2, out_dir)
        if args.checkpoint_a3 is not None:
            _check_checkpoint(env, "a3", args.checkpoint_a3, out_dir)
        else:
            print("[a3] skipped (no --checkpoint-a3 given)", flush=True)
        print(f"[artifacts] {out_dir}", flush=True)
        print(f"PASS: {args.policy} policy rollout gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"FAIL: {args.policy} policy rollout gate failed.", flush=True)
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
