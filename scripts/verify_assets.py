#!/usr/bin/env python
"""Phase-1 asset load verification (headless Isaac Sim).

Proves the two local assets load in Isaac Lab, as a *fast* smoke test:

1. **Robot** — spawn the Alex URDF on a ground plane, ``sim.reset()`` (cooks only
   Alex), step a few frames, and report the joint set. This exercises URDF -> USD
   conversion, mesh resolution, and articulation init.
2. **Scene** — reference the combined corridor scene onto the stage and count its
   prims. This exercises USD composition + reference resolution (the iThor object
   refs resolve via ``~/objects/thor``). It deliberately does **not** physics-cook
   the furnished multi-room scene (that takes many minutes and is unnecessary to
   prove the asset loads).

Exits 0 only if both checks pass.

Run through the official Isaac Lab launcher (see docs/environment.md)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_assets.py --viz none --device cpu --steps 1

For GUI inspection, use::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
        scripts/verify_assets.py --viz kit --device cpu --steps 1

Flags: ``--variant {fullbody,nub}``  ``--scene {combined,none}``  ``--steps N``.
By default the script exits immediately after flushing the result because Kit
shutdown can be slow; pass ``--clean-shutdown`` to debug graceful shutdown.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# ── AppLauncher must be configured before any other Isaac import ─────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AlexDoor-XAS Phase-1 asset load check")
parser.add_argument("--variant", choices=["fullbody", "nub"], default="fullbody",
                    help="Alex configuration to spawn (default: fullbody).")
parser.add_argument("--scene", choices=["combined", "none"], default="combined",
                    help="Also compose the combined corridor scene (default: combined).")
parser.add_argument("--steps", type=int, default=8, help="Robot simulation steps to run.")
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Call SimulationApp.close() before exiting; useful for debugging Kit shutdown hangs.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac imports (after AppLauncher) ────────────────────────────────────────
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from alexdoor_xas.assets import scenes  # noqa: E402
from alexdoor_xas.assets.alex import load_alex_articulation_cfg  # noqa: E402


def _check_robot() -> list[str]:
    """Spawn Alex, step a few frames, and return its joint names.

    No ground plane: ``GroundPlaneCfg`` fetches its USD from the Isaac remote
    asset root (unreachable/misconfigured on this machine), and a load check does
    not need one — over a handful of steps Alex free-falls a negligible amount.
    """
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
    sim.set_camera_view(eye=(4.0, 4.0, 3.0), target=(0.0, 0.0, 1.0))
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    robot_cfg = load_alex_articulation_cfg(args.variant)
    robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
    print(f"[alex] variant={args.variant}  urdf={robot_cfg.spawn.asset_path}")
    robot = Articulation(robot_cfg.replace(prim_path="/World/Alex"))

    sim.reset()  # cooks only Alex + ground → fast
    for _ in range(args.steps):
        sim.step()
        robot.update(0.005)

    joint_names = list(robot.data.joint_names)
    print(f"[alex] spawned OK — {len(joint_names)} joints:")
    print("       " + ", ".join(joint_names))
    return joint_names


def _check_scene() -> int:
    """Compose the combined scene with pure USD (no PhysX) and count prims.

    Uses ``pxr`` to open the stage with all payloads loaded, so every referenced
    room + object USD is actually resolved and opened (the iThor objects resolve
    via the ``~/objects/thor`` symlink) — without physics-cooking the scene, which
    is what makes this a fast smoke test rather than a multi-minute one.
    """
    scene_usd = scenes.combined_scene_usd()
    if not scene_usd.is_file():
        raise FileNotFoundError(f"combined scene not found: {scene_usd}")
    stage = scenes.open_stage(scene_usd, load_all=True)
    n_prims = sum(1 for _ in stage.Traverse())
    print(f"[scene] composed {scene_usd.name} — {n_prims} prims (payloads loaded)", flush=True)
    return n_prims


def main() -> int:
    rc = 0
    try:
        _check_robot()
        if args.scene == "combined":
            _check_scene()
        else:
            print("[scene] skipped (--scene none)")
        print("PASS: assets loaded.", flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: asset load verification failed.", flush=True)
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
    # os._exit avoids Kit's shutdown masking the process exit code.
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
