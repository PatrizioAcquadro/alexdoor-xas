#!/usr/bin/env python
"""Verify the complete Door + Alex V2 benchmark scene in Isaac Lab.

The gate validates the pinned Alex V2 manifest and generated door-task USD,
then loads the production environment, resets it, and checks that Alex, the
hinge, door frame, and panel remain finite and stable during zero-action steps.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_benchmark_scene.py --viz none --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS benchmark scene gate")
parser.add_argument("--steps", type=int, default=100, help="Zero-action steps to run.")
parser.add_argument(
    "--frame-drift-tol",
    type=float,
    default=1e-4,
    help="Maximum allowed door-frame drift in meters.",
)
parser.add_argument(
    "--door-bound",
    type=float,
    default=5.0,
    help="Maximum allowed door-panel world-position norm in meters.",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac/runtime imports must follow AppLauncher construction.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import alexdoor_xas.envs.door_task as door_task  # noqa: E402
from alexdoor_xas.assets.alex_v2 import build_alex_v2_door_asset  # noqa: E402
from alexdoor_xas.assets.alex_v2_contract import (  # noqa: E402
    EXPECTED_RUNTIME_JOINTS,
    RobotAssetRef,
)
from alexdoor_xas.assets.door_task import (  # noqa: E402
    ensure_door_task_usd,
    validate_door_task_usd,
)
from alexdoor_xas.envs.door_task.door_push_alex_v2_env_cfg import (  # noqa: E402
    DoorPushAlexV2EnvCfg,
)

EXPECTED_HINGE_PRIM_PATH = "/World/DoorTaskDoor/Doorframe/Hinge"
DOOR_BODY_NAMES = ("Doorframe", "Door", "Handle")
USD_EXTENSIONS = (".usd", ".usda", ".usdc")


def _as_torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _check_static_usd(usd_path: Path) -> str:
    """Validate generation, dependencies, hinge, and authored mass/inertia."""
    from pxr import Usd, UsdPhysics, UsdUtils  # noqa: PLC0415

    validate_door_task_usd(usd_path)
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open door task USD: {usd_path}")

    layers, resolved, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
    scan_text = "\n".join(
        [
            str(usd_path),
            *(str(asset) for asset in resolved),
            *(str(asset) for asset in unresolved),
            *(_layer_text(layer) for layer in layers),
            *(_reference_text(stage)),
        ]
    ).lower()
    if "file:/c:" in scan_text:
        raise RuntimeError("door task USD contains a forbidden file:/C:/ reference")
    unresolved_usd = [asset for asset in unresolved if _is_usd_like_asset(asset)]
    if unresolved_usd:
        raise RuntimeError(f"door task USD has unresolved USD references: {unresolved_usd}")
    unresolved_non_usd = [asset for asset in unresolved if not _is_usd_like_asset(asset)]
    if unresolved_non_usd:
        print(f"[door] ignored unresolved non-USD assets: {unresolved_non_usd}", flush=True)

    hinges = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    if hinges != [EXPECTED_HINGE_PRIM_PATH]:
        raise RuntimeError(
            f"expected exactly one hinge at {EXPECTED_HINGE_PRIM_PATH}; found {hinges}"
        )
    for prim_path in ("/World/DoorTaskDoor/Door", "/World/DoorTaskDoor/Handle"):
        prim = stage.GetPrimAtPath(prim_path)
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if mass is None or not math.isfinite(float(mass)) or float(mass) <= 0.0:
            raise RuntimeError(f"{prim_path} has invalid authored mass: {mass}")
        if inertia is None or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in inertia
        ):
            raise RuntimeError(f"{prim_path} has invalid authored inertia: {inertia}")
    return hinges[0]


def _run_live_environment(asset_ref: RobotAssetRef, hinge_prim_path: str) -> dict[str, object]:
    if args.steps < 0:
        raise ValueError(f"--steps must be non-negative, got {args.steps}")

    cfg = DoorPushAlexV2EnvCfg()
    cfg.seed = 0
    cfg.sim.device = args.device
    env = gym.make(door_task.DOOR_PUSH_ALEX_V2_ENV_ID, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=0)
        policy_obs = obs["policy"]
        _require_finite("reset observation", policy_obs)
        joint_names = env.robot_joint_names()
        if tuple(joint_names) != EXPECTED_RUNTIME_JOINTS:
            raise RuntimeError(
                "Alex V2 joint names/order differ from the frozen manifest: "
                f"expected={list(EXPECTED_RUNTIME_JOINTS)!r}, actual={joint_names!r}"
            )
        live_ref = RobotAssetRef.from_dict(env.robot_asset_provenance())
        if live_ref != asset_ref:
            raise RuntimeError(f"live Alex asset identity differs: {live_ref} != {asset_ref}")

        door = env._door  # noqa: SLF001 - verifier intentionally inspects cooked PhysX state.
        body_names = list(door.body_names)
        body_ids = {name: _resolve_body_id(body_names, name) for name in DOOR_BODY_NAMES}
        masses = _as_torch(door.data.body_mass).detach().cpu()
        inertias = _as_torch(door.data.body_inertia).detach().cpu()
        _require_finite("cooked body masses", masses)
        _require_finite("cooked body inertias", inertias)
        if not torch.all(masses > 0.0):
            raise RuntimeError(f"cooked door masses must be positive: {masses}")
        if not torch.all(inertias[..., [0, 4, 8]] > 0.0):
            raise RuntimeError(f"cooked door inertia diagonals must be positive: {inertias}")

        hinge_pos, hinge_vel = env.hinge_state()
        _require_finite("reset hinge position", hinge_pos)
        _require_finite("reset hinge velocity", hinge_vel)
        frame_id = body_ids["Doorframe"]
        door_id = body_ids["Door"]
        frame_pos_0 = _as_torch(door.data.body_pos_w)[0, frame_id].detach().clone()
        max_frame_drift = 0.0
        max_door_bound = _door_position_bound(door, door_id)
        zero_action = torch.zeros((env.num_envs, cfg.action_space), device=env.device)

        for step in range(args.steps):
            obs, _, _, _, _ = env.step(zero_action)
            _require_finite(f"step {step}: observation", obs["policy"])
            hinge_pos, hinge_vel = env.hinge_state()
            _require_finite(f"step {step}: hinge position", hinge_pos)
            _require_finite(f"step {step}: hinge velocity", hinge_vel)
            body_pos = _as_torch(door.data.body_pos_w)
            body_quat = _as_torch(door.data.body_quat_w)
            _require_finite(f"step {step}: door position", body_pos[0, door_id])
            _require_finite(f"step {step}: door quaternion", body_quat[0, door_id])

            drift = float(torch.linalg.vector_norm(body_pos[0, frame_id] - frame_pos_0).item())
            max_frame_drift = max(max_frame_drift, drift)
            if drift > args.frame_drift_tol:
                raise RuntimeError(
                    f"door frame drift at step {step}: {drift:.6g} m > "
                    f"{args.frame_drift_tol:.6g} m"
                )
            bound = _door_position_bound(door, door_id)
            max_door_bound = max(max_door_bound, bound)
            if bound > args.door_bound:
                raise RuntimeError(
                    f"door panel bound at step {step}: {bound:.6g} m > {args.door_bound:.6g} m"
                )

        final_hinge, _ = env.hinge_state()
        return {
            "hinge_prim_path": hinge_prim_path,
            "hinge_angle": float(final_hinge[0].detach().cpu().item()),
            "joint_names": joint_names,
            "body_names": body_names,
            "masses": masses,
            "inertias": inertias,
            "max_frame_drift": max_frame_drift,
            "max_door_bound": max_door_bound,
        }
    finally:
        env.close()


def _resolve_body_id(body_names: list[str], target: str) -> int:
    matches = [
        idx
        for idx, name in enumerate(body_names)
        if name == target or name.rsplit("/", 1)[-1] == target
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one body named {target!r}; found {matches} in {body_names}")
    return matches[0]


def _door_position_bound(door, door_id: int) -> float:
    value = _as_torch(door.data.body_pos_w)[0, door_id]
    return float(torch.linalg.vector_norm(value).detach().cpu().item())


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values: {tensor.detach().cpu()}")


def _is_usd_like_asset(asset: object) -> bool:
    return str(asset).lower().endswith(USD_EXTENSIONS)


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
    return [
        str(value)
        for ref in _iter_authored_references(stage)
        for value in (ref.assetPath, ref.primPath)
    ]


def _layer_text(layer) -> str:
    try:
        return layer.ExportToString()
    except Exception:  # noqa: BLE001
        return str(layer.identifier)


def _print_summary(usd_path: Path, asset, ref: RobotAssetRef, evidence: dict[str, object]) -> None:
    print(
        f"[alex] model=v2 urdf={asset.urdf_path} asset_id={ref.asset_id} "
        f"asset_sha256={ref.sha256} manifest_fingerprint={ref.manifest_fingerprint}",
        flush=True,
    )
    print(f"[alex] runtime_joints={len(evidence['joint_names'])}", flush=True)
    print(f"[door] usd={usd_path} hinge={evidence['hinge_prim_path']}", flush=True)
    print(
        f"[door] bodies={evidence['body_names']} hinge_angle={evidence['hinge_angle']:.9g} rad",
        flush=True,
    )
    print(
        f"[door] stability=max_frame_drift={evidence['max_frame_drift']:.9g} m "
        f"max_door_bound={evidence['max_door_bound']:.9g} m",
        flush=True,
    )


def main() -> int:
    rc = 0
    try:
        asset, ref = build_alex_v2_door_asset()
        usd_path = ensure_door_task_usd()
        hinge_prim_path = _check_static_usd(usd_path)
        evidence = _run_live_environment(ref, hinge_prim_path)
        _print_summary(usd_path, asset, ref, evidence)
        print("PASS: Door + Alex V2 benchmark scene is valid and stable.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: benchmark scene verification failed.", flush=True)
        rc = 1
    finally:
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
