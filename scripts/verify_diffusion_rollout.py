#!/usr/bin/env python
"""Phase 3.3 rollout gate: sampled diffusion chunks execute through adapter-v1 in Isaac.

Requires trained checkpoints (from ``scripts/train_diffusion.py``) the same
way the dataset gate requires exported datasets. Per checkpoint (A2 required,
A3 optional) on ``AlexDoor-DoorPush-Alex-v0``: one fixed-seed rollout and one
randomized-start rollout through the matching adapter must complete within
the tick budget with every command logged and no crash, and the fixed-seed
rollout must open the door past the success threshold — the Phase 3.3 claim
that sampled diffusion chunks are adapter-executable. Execution is
receding-horizon (Ta of each sampled Tp-chunk) with per-rollout seeded
sampling. Artifacts go under ``outputs/verify_diffusion_rollout/gate/``::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_diffusion_rollout.py --viz none --device cpu \
        --checkpoint-a2 outputs/diffusion_door_push/<run_id>/checkpoints/best.pt \
        --checkpoint-a3 outputs/diffusion_door_push/<run_id>/checkpoints/best.pt
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

parser = argparse.ArgumentParser(description="AlexDoor-XAS diffusion rollout verification gate")
parser.add_argument(
    "--checkpoint-a2", type=str, required=True, help="Trained DP-A2 checkpoint (.pt)."
)
parser.add_argument(
    "--checkpoint-a3", type=str, default=None, help="Trained DP-A3 checkpoint (.pt)."
)
parser.add_argument("--seed", type=int, default=0, help="Fixed-rollout reset seed.")
parser.add_argument("--max-ticks", type=int, default=600, help="Per-rollout tick budget.")
parser.add_argument(
    "--sampler", type=str, default="ddpm", choices=("ddpm", "ddim"), help="Rollout sampler."
)
parser.add_argument(
    "--inference-steps", type=int, default=100, help="Denoising steps at rollout."
)
parser.add_argument(
    "--n-action-steps", type=int, default=8, help="Executed prefix of each sampled chunk (Ta)."
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
from alexdoor_xas.data_engine import DataEngineCfg, apply_start_offset, plan_episodes  # noqa: E402
from alexdoor_xas.envs.door_task.door_push_alex_env_cfg import (  # noqa: E402
    ALEX_ROBOT_TAG,
    DoorPushAlexEnvCfg,
)
from alexdoor_xas.policies.common.obs import stop_on_hinge_angle  # noqa: E402
from alexdoor_xas.policies.diffusion.policy import (  # noqa: E402
    DiffusionPolicy,
    diffusion_chunk_source,
)
from alexdoor_xas.policies.scripted import ALEX_VARIATION_BOUNDS  # noqa: E402

EXPERIMENT = "verify_diffusion_rollout"
EXPECTED_SPACES = {"a2": A2_EE_DELTA, "a3": A3_OBJ_REL_EE_DELTA}


def _make_env():
    cfg = DoorPushAlexEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    return gym.make(door_task.DOOR_PUSH_ALEX_ENV_ID, cfg=cfg).unwrapped


def _adapter_for(action_space: str):
    a2 = A2Adapter(limits_for_robot(ALEX_ROBOT_TAG))
    return a2 if action_space == A2_EE_DELTA else A3Adapter(a2)


def _rollout(env, policy, seed: int, variation) -> dict:
    env.reset(seed=seed)
    if variation is not None:
        apply_start_offset(env, read_door_frame(env), variation)
    adapter = _adapter_for(policy.action_space)
    policy.seed(seed)  # deterministic sampling per rollout
    # Rollouts end at the first chunk boundary past the success angle: the
    # demos terminate with the FSM, so post-task extrapolation is unbounded
    # (a wandering arm can knock the open door shut again).
    source = stop_on_hinge_angle(
        diffusion_chunk_source(policy, env, n_action_steps=args.n_action_steps),
        DataEngineCfg().success_angle_rad,
    )
    result = rollout_chunks(env, source, adapter, max_ticks=args.max_ticks)

    if result.n_ticks == 0 or result.n_ticks > args.max_ticks:
        raise RuntimeError(f"rollout ran {result.n_ticks} ticks (budget {args.max_ticks})")
    if len(result.decisions_per_tick) != result.n_ticks:
        raise RuntimeError(
            f"{len(result.decisions_per_tick)} decisions logged for {result.n_ticks} ticks"
        )
    if not (math.isfinite(result.final_angle_rad) and math.isfinite(result.initial_angle_rad)):
        raise RuntimeError("rollout produced non-finite door angles")
    return {
        "seed": seed,
        "randomized": variation is not None,
        "final_angle_rad": result.final_angle_rad,
        "door_angle_change_rad": result.door_angle_change_rad,
        "n_ticks": result.n_ticks,
        "n_accepted": result.log.n_accepted,
        "n_corrected": result.log.n_corrected,
        "n_rejected": result.log.n_rejected,
        "log": result.log.to_dict(),
    }


def _check_checkpoint(env, label: str, checkpoint: str, out_dir) -> None:
    policy = DiffusionPolicy.from_checkpoint(
        paths.REPO_ROOT / checkpoint,
        sampler=args.sampler,
        num_inference_steps=args.inference_steps,
    )
    expected_space = EXPECTED_SPACES[label]
    if policy.action_space != expected_space:
        raise RuntimeError(
            f"--checkpoint-{label} is a {policy.action_space} model, expected {expected_space}"
        )
    success_angle = DataEngineCfg().success_angle_rad

    fixed = _rollout(env, policy, args.seed, None)
    (out_dir / f"{label}_fixed.json").write_text(json.dumps(fixed, indent=2) + "\n")
    if fixed["final_angle_rad"] < success_angle:
        raise RuntimeError(
            f"{label}: fixed-seed rollout final angle "
            f"{math.degrees(fixed['final_angle_rad']):.1f} deg is below the success "
            f"threshold {math.degrees(success_angle):.1f} deg"
        )

    item = plan_episodes(0, 1, args.seed + 1, ALEX_VARIATION_BOUNDS)[0]
    randomized = _rollout(env, policy, item.seed, item.variation)
    (out_dir / f"{label}_randomized.json").write_text(json.dumps(randomized, indent=2) + "\n")

    print(
        f"[{label}] fixed: final={math.degrees(fixed['final_angle_rad']):.1f} deg "
        f"ticks={fixed['n_ticks']} a/c/r={fixed['n_accepted']}/{fixed['n_corrected']}/"
        f"{fixed['n_rejected']} | randomized: "
        f"final={math.degrees(randomized['final_angle_rad']):.1f} deg "
        f"ticks={randomized['n_ticks']}",
        flush=True,
    )


def main() -> int:
    rc = 0
    env = None
    try:
        out_dir = paths.OUTPUTS_DIR / EXPERIMENT / "gate"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = _make_env()

        _check_checkpoint(env, "a2", args.checkpoint_a2, out_dir)
        if args.checkpoint_a3 is not None:
            _check_checkpoint(env, "a3", args.checkpoint_a3, out_dir)
        else:
            print("[a3] skipped (no --checkpoint-a3 given)", flush=True)

        print(f"[artifacts] {out_dir}", flush=True)
        print("PASS: Diffusion Policy rollout gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: Diffusion Policy rollout gate failed.", flush=True)
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
    # os._exit avoids Kit shutdown masking the verification exit code.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
