#!/usr/bin/env python
"""Door-only Isaac Lab verification for the single-door task scene.

This is a focused integration gate before adding Alex. It launches Isaac Lab,
opens the generated door task USD, binds only the door articulation, steps the
simulation, and fails non-zero if the door fixture is not stable enough for
later scripted interactions.

Run through the official Isaac Lab launcher::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_door_task_scene.py --viz none --device cpu --steps 100
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

# -- AppLauncher must be configured before any other Isaac import.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS door-only task scene check")
parser.add_argument("--steps", type=int, default=100, help="Simulation steps to run.")
parser.add_argument(
    "--frame-drift-tol",
    type=float,
    default=1e-4,
    help="Maximum allowed static frame drift in meters.",
)
parser.add_argument(
    "--door-bound",
    type=float,
    default=5.0,
    help="Maximum allowed door panel world-position norm in meters.",
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

# -- Isaac imports after AppLauncher.
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from alexdoor_xas.assets.door_task import (  # noqa: E402
    ensure_door_task_usd,
    validate_door_task_usd,
)

ARTICULATION_PRIM_PATH = "/World/DoorTaskDoor"
EXPECTED_HINGE_PRIM_PATH = "/World/DoorTaskDoor/Doorframe/Hinge"
DOOR_BODY_NAMES = ("Doorframe", "Door", "Handle")
AUTHORED_MASS_PRIMS = (
    "/World/DoorTaskDoor/Door",
    "/World/DoorTaskDoor/Handle",
)
USD_EXTENSIONS = (".usd", ".usda", ".usdc")


def _check_static_usd(usd_path: Path) -> str:
    """Validate USD composition and authored door physics before launching PhysX."""
    from pxr import Usd, UsdPhysics, UsdUtils  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open door task USD: {usd_path}")

    layers, resolved_assets, unresolved_assets = UsdUtils.ComputeAllDependencies(str(usd_path))
    scan_text = "\n".join(
        [
            str(usd_path),
            *(str(asset) for asset in resolved_assets),
            *(str(asset) for asset in unresolved_assets),
            *(_layer_text(layer) for layer in layers),
            *(_reference_text(stage)),
        ]
    ).lower()
    if "file:/c:" in scan_text:
        raise RuntimeError("door task USD contains a forbidden file:/C:/ reference")

    unresolved_usd = [asset for asset in unresolved_assets if _is_usd_like_asset(asset)]
    if unresolved_usd:
        raise RuntimeError(f"door task USD has unresolved USD references: {unresolved_usd}")

    unresolved_non_usd = [asset for asset in unresolved_assets if not _is_usd_like_asset(asset)]
    if unresolved_non_usd:
        print(f"[usd] non-USD unresolved assets ignored: {unresolved_non_usd}", flush=True)

    hinges = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    hinge_paths = [prim.GetPath().pathString for prim in hinges]
    if hinge_paths != [EXPECTED_HINGE_PRIM_PATH]:
        raise RuntimeError(
            "door task USD must contain exactly one revolute hinge at "
            f"{EXPECTED_HINGE_PRIM_PATH}; found {hinge_paths}"
        )

    _check_authored_mass_inertia(stage)
    return hinge_paths[0]


def _check_authored_mass_inertia(stage) -> None:
    from pxr import UsdPhysics  # noqa: PLC0415

    for prim_path in AUTHORED_MASS_PRIMS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"authored mass prim is missing: {prim_path}")

        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if mass is None or not math.isfinite(float(mass)) or float(mass) <= 0.0:
            raise RuntimeError(f"{prim_path} has invalid authored mass: {mass}")
        if inertia is None:
            raise RuntimeError(f"{prim_path} has no authored diagonal inertia")
        bad_inertia = [
            float(value)
            for value in inertia
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if bad_inertia:
            raise RuntimeError(f"{prim_path} has invalid authored diagonal inertia: {inertia}")


def _run_live_sim(usd_path: Path, hinge_prim_path: str) -> dict[str, object]:
    """Step the door-only articulation and return measured verification evidence."""
    if args.steps < 0:
        raise ValueError(f"--steps must be non-negative, got {args.steps}")

    sim_utils.open_stage(str(usd_path))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    door = Articulation(ArticulationCfg(prim_path=ARTICULATION_PRIM_PATH, actuators={}))

    sim.reset()
    sim_dt = sim.get_physics_dt()
    door.update(0.0)

    joint_names = list(door.joint_names)
    if not joint_names:
        raise RuntimeError("door articulation exposes no joints")

    hinge_joint_id, hinge_joint_name = _resolve_hinge_joint(joint_names)
    body_names = list(door.body_names)
    body_ids = _resolve_required_body_ids(body_names)

    _check_cooked_mass_inertia(door, body_names)

    joint_pos = door.data.joint_pos.torch
    _require_finite("initial joint positions", joint_pos)
    hinge_angle = float(joint_pos[0, hinge_joint_id].detach().cpu().item())
    if not math.isfinite(hinge_angle):
        raise RuntimeError(f"hinge angle is not finite: {hinge_angle}")

    frame_id = body_ids["Doorframe"]
    door_id = body_ids["Door"]
    frame_pos_0 = door.data.body_pos_w.torch[0, frame_id].detach().clone()
    max_frame_drift = 0.0
    max_door_bound = _door_position_bound(door, door_id)

    for step in range(args.steps):
        sim.step()
        door.update(sim_dt)

        _require_finite(f"step {step}: joint positions", door.data.joint_pos.torch)
        _require_finite(f"step {step}: door position", door.data.body_pos_w.torch[0, door_id])
        _require_finite(f"step {step}: door quaternion", door.data.body_quat_w.torch[0, door_id])

        frame_pos = door.data.body_pos_w.torch[0, frame_id]
        frame_drift = float(torch.linalg.vector_norm(frame_pos - frame_pos_0).detach().cpu().item())
        max_frame_drift = max(max_frame_drift, frame_drift)
        if frame_drift > args.frame_drift_tol:
            raise RuntimeError(
                f"door frame drift exceeded tolerance at step {step}: "
                f"{frame_drift:.6g} m > {args.frame_drift_tol:.6g} m"
            )

        door_bound = _door_position_bound(door, door_id)
        max_door_bound = max(max_door_bound, door_bound)
        if door_bound > args.door_bound:
            raise RuntimeError(
                f"door panel exceeded world-position bound at step {step}: "
                f"{door_bound:.6g} m > {args.door_bound:.6g} m"
            )

    hinge_angle = float(door.data.joint_pos.torch[0, hinge_joint_id].detach().cpu().item())
    return {
        "hinge_prim_path": hinge_prim_path,
        "hinge_joint_name": hinge_joint_name,
        "hinge_joint_id": hinge_joint_id,
        "hinge_angle": hinge_angle,
        "body_names": body_names,
        "body_ids": body_ids,
        "masses": door.data.body_mass.torch.detach().cpu(),
        "inertias": door.data.body_inertia.torch.detach().cpu(),
        "max_frame_drift": max_frame_drift,
        "max_door_bound": max_door_bound,
    }


def _check_cooked_mass_inertia(door: Articulation, body_names: list[str]) -> None:
    masses = door.data.body_mass.torch.detach().cpu()
    inertias = door.data.body_inertia.torch.detach().cpu()
    inertia_diag = inertias[..., [0, 4, 8]]

    _require_finite("cooked body masses", masses)
    _require_finite("cooked body inertias", inertias)
    if not torch.all(masses > 0.0):
        raise RuntimeError(
            f"cooked body masses must be positive: {_body_table(body_names, masses)}"
        )
    if not torch.all(inertia_diag > 0.0):
        raise RuntimeError(
            "cooked body inertia diagonals must be positive: "
            f"{_body_table(body_names, inertia_diag)}"
        )


def _resolve_hinge_joint(joint_names: list[str]) -> tuple[int, str]:
    matches = [
        idx
        for idx, name in enumerate(joint_names)
        if name.lower() == "hinge" or name.rsplit("/", 1)[-1].lower() == "hinge"
    ]
    if len(matches) == 1:
        joint_id = matches[0]
        return joint_id, joint_names[joint_id]
    if not matches and len(joint_names) == 1:
        return 0, joint_names[0]
    raise RuntimeError(f"could not identify a unique hinge joint from joints: {joint_names}")


def _resolve_required_body_ids(body_names: list[str]) -> dict[str, int]:
    return {name: _resolve_body_id(body_names, name) for name in DOOR_BODY_NAMES}


def _resolve_body_id(body_names: list[str], target: str) -> int:
    matches = [
        idx
        for idx, name in enumerate(body_names)
        if name == target or name.rsplit("/", 1)[-1] == target
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one body named {target!r}; found {matches} in {body_names}")
    return matches[0]


def _door_position_bound(door: Articulation, door_id: int) -> float:
    return float(
        torch.linalg.vector_norm(door.data.body_pos_w.torch[0, door_id]).detach().cpu().item()
    )


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
    values: list[str] = []
    for ref in _iter_authored_references(stage):
        values.append(str(ref.assetPath))
        values.append(str(ref.primPath))
    return values


def _layer_text(layer) -> str:
    try:
        return layer.ExportToString()
    except Exception:  # noqa: BLE001 - validation should still report dependency identifiers.
        return str(layer.identifier)


def _body_table(body_names: list[str], values: torch.Tensor) -> list[tuple[str, object]]:
    rows = []
    for idx, name in enumerate(body_names):
        value = values[0, idx]
        rows.append((name, value.tolist() if value.ndim else float(value.item())))
    return rows


def _print_summary(usd_path: Path, evidence: dict[str, object]) -> None:
    body_names = evidence["body_names"]
    masses = evidence["masses"]
    inertias = evidence["inertias"]
    assert isinstance(body_names, list)
    assert isinstance(masses, torch.Tensor)
    assert isinstance(inertias, torch.Tensor)

    print(f"[door] usd={usd_path}", flush=True)
    print(
        "[door] hinge="
        f"{evidence['hinge_prim_path']}  joint={evidence['hinge_joint_name']} "
        f"(id={evidence['hinge_joint_id']})  angle={float(evidence['hinge_angle']):.9g} rad",
        flush=True,
    )
    print(f"[door] bodies={body_names}", flush=True)
    print("[door] cooked mass/inertia diagonal:", flush=True)
    for idx, name in enumerate(body_names):
        diag = inertias[0, idx, [0, 4, 8]].tolist()
        print(f"       {name}: mass={float(masses[0, idx]):.9g} diag={diag}", flush=True)
    print(
        "[door] stability="
        f"max_frame_drift={float(evidence['max_frame_drift']):.9g} m "
        f"max_door_bound={float(evidence['max_door_bound']):.9g} m",
        flush=True,
    )


def main() -> int:
    rc = 0
    try:
        usd_path = ensure_door_task_usd()
        validate_door_task_usd(usd_path)
        hinge_prim_path = _check_static_usd(usd_path)
        evidence = _run_live_sim(usd_path, hinge_prim_path)
        _print_summary(usd_path, evidence)
        print("PASS: door task scene stable", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: door task scene verification failed.", flush=True)
        rc = 1
    finally:
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
