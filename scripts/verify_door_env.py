#!/usr/bin/env python
"""Isaac Lab verification gate for the single-door DirectRLEnv.

This script is an integration gate, not a training entrypoint. It launches
Isaac Lab, creates the registered door env, resets it, rolls out deterministic
actions, and fails non-zero if the env is not ready for future scripted
interactions.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_door_env.py --viz none --device cpu --steps 100
"""

from __future__ import annotations

import argparse
import math
import numbers
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# -- AppLauncher must be configured before any other Isaac import.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS door env verification gate")
parser.add_argument("--steps", type=int, default=100, help="Environment steps per rollout.")
parser.add_argument("--seed", type=int, default=1234, help="Seed used for env creation and reset.")
parser.add_argument(
    "--determinism-tol",
    type=float,
    default=1e-6,
    help="Maximum allowed absolute difference between repeated rollout traces.",
)
parser.add_argument(
    "--angle-bound-tol",
    type=float,
    default=1e-6,
    help="Tolerance applied around cooked hinge position limits, in radians.",
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
import torch  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas.assets.door_task import ensure_door_task_usd, validate_door_task_usd  # noqa: E402
from alexdoor_xas.envs.door_task.door_env_cfg import (  # noqa: E402
    OBSERVATION_TERMS,
    DoorTaskEnvCfg,
)

EXPECTED_HINGE_LOWER_RAD = 0.0
EXPECTED_HINGE_UPPER_RAD = math.pi / 2.0
FORBIDDEN_SCENE_TOKENS = (
    "combinedhallwayscene",
    "floorplan",
    "objects/thor",
    "/thor/",
    "\\thor\\",
    "file:/c:",
)


def _assert_static_asset_contract(usd_path: Path) -> None:
    """Validate generated task USD and prove it does not reference the full scene."""
    from pxr import Usd, UsdUtils  # noqa: PLC0415

    validate_door_task_usd(usd_path)

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open door task USD: {usd_path}")

    layers, resolved_assets, unresolved_assets = UsdUtils.ComputeAllDependencies(str(usd_path))
    scan_text = "\n".join(
        [
            str(usd_path),
            usd_path.read_text(),
            *(str(asset) for asset in resolved_assets),
            *(str(asset) for asset in unresolved_assets),
            *(_layer_text(layer) for layer in layers),
            *(_reference_text(stage)),
        ]
    ).lower()
    offenders = [token for token in FORBIDDEN_SCENE_TOKENS if token in scan_text]
    if offenders:
        raise RuntimeError(f"door env task USD contains forbidden scene references: {offenders}")


def _make_env(usd_path: Path):
    cfg = DoorTaskEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device

    env = gym.make(door_task.DOOR_TASK_ENV_ID, cfg=cfg).unwrapped
    spawn_path = Path(env.cfg.door_task_scene.spawn.usd_path).expanduser().resolve()
    expected_path = usd_path.expanduser().resolve()
    if spawn_path != expected_path:
        raise RuntimeError(
            f"env is using unexpected door task USD: {spawn_path} != {expected_path}"
        )
    if "combinedhallwayscene" in str(spawn_path).lower():
        raise RuntimeError(f"env spawn path uses forbidden CombinedHallwayScene path: {spawn_path}")

    return env


def _rollout(env, seed: int, steps: int) -> dict[str, Any]:
    if steps < 0:
        raise ValueError(f"--steps must be non-negative, got {steps}")

    reset_result = env.reset(seed=seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise RuntimeError(f"reset() must return (obs, info), got: {type(reset_result).__name__}")
    obs, info = reset_result
    _assert_info(info, "reset info")
    policy_obs = _assert_policy_observation(env, obs, "reset observation")
    _assert_angle_bounds(policy_obs, _hinge_limits(env), "reset observation")

    obs_trace = [policy_obs.detach().cpu()]
    reward_trace: list[torch.Tensor] = []
    terminated_trace: list[torch.Tensor] = []
    truncated_trace: list[torch.Tensor] = []
    action_trace: list[torch.Tensor] = []

    for step in range(steps):
        action = _deterministic_action(env, step)
        step_result = env.step(action)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise RuntimeError(
                "step() must return (obs, reward, terminated, truncated, info), "
                f"got {type(step_result).__name__} with length "
                f"{len(step_result) if isinstance(step_result, tuple) else 'n/a'}"
            )

        obs, reward, terminated, truncated, info = step_result
        _assert_info(info, f"step {step} info")
        policy_obs = _assert_policy_observation(env, obs, f"step {step} observation")
        reward = _assert_reward(env, reward, step)
        terminated = _assert_bool_flags(env, terminated, "terminated", step)
        truncated = _assert_bool_flags(env, truncated, "truncated", step)
        _assert_angle_bounds(policy_obs, _hinge_limits(env), f"step {step} observation")
        _assert_stored_action(env, action, step)

        obs_trace.append(policy_obs.detach().cpu())
        reward_trace.append(reward.detach().cpu())
        terminated_trace.append(terminated.detach().cpu())
        truncated_trace.append(truncated.detach().cpu())
        action_trace.append(action.detach().cpu())

    return {
        "observations": torch.stack(obs_trace),
        "rewards": torch.stack(reward_trace) if reward_trace else torch.empty(0),
        "terminated": torch.stack(terminated_trace) if terminated_trace else torch.empty(0),
        "truncated": torch.stack(truncated_trace) if truncated_trace else torch.empty(0),
        "actions": torch.stack(action_trace) if action_trace else torch.empty(0),
    }


def _deterministic_action(env, step: int) -> torch.Tensor:
    action_dim = int(env.cfg.action_space)
    base = torch.arange(action_dim, dtype=torch.float32, device=env.device)
    values = torch.sin(base + float(step) * 0.173) * float(env.cfg.action_clip)
    return values.repeat(env.num_envs, 1)


def _assert_policy_observation(env, obs: Any, name: str) -> torch.Tensor:
    if not isinstance(obs, dict):
        raise RuntimeError(f"{name} must be a dict, got {type(obs).__name__}")
    if "policy" not in obs:
        raise RuntimeError(f"{name} is missing 'policy': keys={list(obs.keys())}")

    policy_obs = obs["policy"]
    if not isinstance(policy_obs, torch.Tensor):
        raise RuntimeError(
            f"{name}['policy'] must be a torch.Tensor, got {type(policy_obs).__name__}"
        )
    expected_shape = (env.num_envs, len(OBSERVATION_TERMS))
    if tuple(policy_obs.shape) != expected_shape:
        raise RuntimeError(
            f"{name}['policy'] has shape {tuple(policy_obs.shape)}, expected {expected_shape}"
        )
    _require_finite_tensor(f"{name}['policy']", policy_obs)
    return policy_obs


def _assert_reward(env, reward: Any, step: int) -> torch.Tensor:
    if not isinstance(reward, torch.Tensor):
        raise RuntimeError(
            f"step {step} reward must be a torch.Tensor, got {type(reward).__name__}"
        )
    expected_shape = (env.num_envs,)
    if tuple(reward.shape) != expected_shape:
        raise RuntimeError(f"step {step} reward shape {tuple(reward.shape)} != {expected_shape}")
    _require_finite_tensor(f"step {step} reward", reward)
    return reward


def _assert_bool_flags(env, flags: Any, name: str, step: int) -> torch.Tensor:
    if not isinstance(flags, torch.Tensor):
        raise RuntimeError(
            f"step {step} {name} flags must be a torch.Tensor, got {type(flags).__name__}"
        )
    expected_shape = (env.num_envs,)
    if tuple(flags.shape) != expected_shape:
        raise RuntimeError(
            f"step {step} {name} flags shape {tuple(flags.shape)} != {expected_shape}"
        )
    if flags.dtype != torch.bool:
        raise RuntimeError(f"step {step} {name} flags must be bool, got {flags.dtype}")
    return flags


def _assert_info(info: Any, name: str) -> None:
    if not isinstance(info, dict):
        raise RuntimeError(f"{name} must be a dict, got {type(info).__name__}")
    _assert_finite_value(name, info)


def _assert_finite_value(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _require_finite_tensor(name, value)
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values: {value}")
    elif isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{name} is non-finite: {value}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_value(f"{name}.{key}", child)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_value(f"{name}[{index}]", child)


def _assert_stored_action(env, action: torch.Tensor, step: int) -> None:
    stored_action = getattr(env, "actions", None)
    if stored_action is None:
        return
    if not isinstance(stored_action, torch.Tensor):
        raise RuntimeError(
            f"env.actions must be a torch.Tensor, got {type(stored_action).__name__}"
        )
    expected = action.clamp(-env.cfg.action_clip, env.cfg.action_clip)
    if not torch.allclose(stored_action, expected, atol=0.0, rtol=0.0):
        raise RuntimeError(
            f"step {step} env.actions does not match clipped deterministic action: "
            f"{stored_action.detach().cpu()} != {expected.detach().cpu()}"
        )


def _hinge_limits(env) -> tuple[float, float]:
    door = getattr(env, "_door", None)
    hinge_joint_id = getattr(env, "_hinge_joint_id", None)
    if door is None or hinge_joint_id is None:
        raise RuntimeError("env must expose _door and _hinge_joint_id for door-angle verification")

    limits = door.data.joint_pos_limits.torch
    _require_finite_tensor("cooked joint position limits", limits)
    lower = float(limits[0, hinge_joint_id, 0].detach().cpu().item())
    upper = float(limits[0, hinge_joint_id, 1].detach().cpu().item())
    if not math.isclose(lower, EXPECTED_HINGE_LOWER_RAD, abs_tol=args.angle_bound_tol):
        raise RuntimeError(
            f"hinge lower limit must be {EXPECTED_HINGE_LOWER_RAD:.9g} rad, got {lower:.9g}"
        )
    if not math.isclose(upper, EXPECTED_HINGE_UPPER_RAD, abs_tol=args.angle_bound_tol):
        raise RuntimeError(
            f"hinge upper limit must be {EXPECTED_HINGE_UPPER_RAD:.9g} rad, got {upper:.9g}"
        )
    return lower, upper


def _assert_angle_bounds(policy_obs: torch.Tensor, limits: tuple[float, float], name: str) -> None:
    lower, upper = limits
    angles = policy_obs[:, 0]
    below_lower = torch.any(angles < lower - args.angle_bound_tol)
    above_upper = torch.any(angles > upper + args.angle_bound_tol)
    if below_lower or above_upper:
        raise RuntimeError(
            f"{name} door angle is outside radian hinge bounds "
            f"[{lower:.9g}, {upper:.9g}] rad: {angles.detach().cpu()}"
        )


def _assert_deterministic(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float:
    max_diff = 0.0
    for key in ("observations", "rewards", "terminated", "truncated", "actions"):
        a = _trace_as_float(first[key])
        b = _trace_as_float(second[key])
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f"determinism trace {key!r} shape mismatch: {a.shape} != {b.shape}")
        diff = 0.0 if a.numel() == 0 else float(torch.max(torch.abs(a - b)).item())
        max_diff = max(max_diff, diff)
        if diff > args.determinism_tol:
            raise RuntimeError(
                f"determinism trace {key!r} exceeded tolerance: "
                f"{diff:.9g} > {args.determinism_tol:.9g}"
            )
    return max_diff


def _trace_as_float(value: torch.Tensor) -> torch.Tensor:
    if value.dtype == torch.bool:
        return value.to(torch.float32)
    return value.to(torch.float32)


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values: {tensor.detach().cpu()}")


def _iter_authored_references(stage):
    for prim in stage.TraverseAll():
        refs = prim.GetMetadata("references")
        if refs is None:
            continue
        yield from refs.explicitItems
        yield from refs.addedItems
        yield from refs.prependedItems
        yield from refs.appendedItems


def _reference_text(stage) -> list[str]:
    values: list[str] = []
    for ref in _iter_authored_references(stage):
        values.append(str(ref.assetPath))
        values.append(str(ref.primPath))
    return values


def _layer_text(layer) -> str:
    try:
        return layer.ExportToString()
    except Exception:  # noqa: BLE001 - validation should still inspect layer identifiers.
        return str(layer.identifier)


def _print_summary(env, usd_path: Path, first: dict[str, torch.Tensor], max_diff: float) -> None:
    lower, upper = _hinge_limits(env)
    final_obs = first["observations"][-1]
    final_angle = float(final_obs[0, 0].item())
    final_velocity = float(final_obs[0, 1].item())
    print(f"[env] id={door_task.DOOR_TASK_ENV_ID}", flush=True)
    print(
        f"[env] seed={args.seed} steps={args.steps} device={env.device} "
        f"num_envs={env.num_envs}",
        flush=True,
    )
    print(f"[env] task_usd={usd_path}", flush=True)
    print(
        f"[door] observation_terms={OBSERVATION_TERMS} "
        f"angle_bounds=[{lower:.9g}, {upper:.9g}] rad",
        flush=True,
    )
    print(
        f"[door] final_angle={final_angle:.9g} rad "
        f"final_velocity={final_velocity:.9g} rad/s",
        flush=True,
    )
    print(f"[determinism] max_rollout_diff={max_diff:.9g}", flush=True)


def main() -> int:
    rc = 0
    env = None
    try:
        usd_path = ensure_door_task_usd()
        _assert_static_asset_contract(usd_path)
        env = _make_env(usd_path)
        first = _rollout(env, args.seed, args.steps)
        second = _rollout(env, args.seed, args.steps)
        max_diff = _assert_deterministic(first, second)
        _print_summary(env, usd_path, first, max_diff)
        print("PASS: door env verification gate passed.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: door env verification gate failed.", flush=True)
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
