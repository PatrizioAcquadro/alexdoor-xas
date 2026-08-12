# Alex V2 Benchmark

The current benchmark is a fixed-base IHMC Alex V2 torso pushing one simulated hinged door through a six-joint right-arm controller.

## Asset and calibration

The autonomous `~/Desktop/Alex` repository owns the external URDF and the
generic `ihmc_alex_isaaclab` extension. DoorManipulation imports
`make_alex_v2_cfg` from `ihmc_alex_isaaclab.robots.alex_v2`, then applies its
fixed-base runtime identity, right-arm PD gains, non-right-arm damping, and
task calibration locally. `assets/alex_v2_contract.py` validates the robot
asset and runtime identity. `configs/alex_v2_door.json` remains the single
active task calibration and contains the task name, base pose, six initial
joints, operational tool frame, reach shell, controller parameters, and
randomization limits.

`scripts/verify_benchmark_scene.py` is the non-mutating scene gate. It validates the Alex asset and runtime manifest, canonical door USD dependencies, hinge, mass and inertia, exact 29-joint runtime order, reset state, and zero-action frame/door stability on the requested device. The optional combined hallway path remains registered for manual composition, but it is neither benchmark physics nor a `check_env.py` requirement.

`configs/alex_v2_door.json` is edited directly when the operational calibration changes. Before committing it, run `verify_benchmark_scene.py`, `verify_scripted_baseline.py`, and `verify_adapters.py`; construction also rejects a malformed config or a tool frame that no longer matches the collision geometry.

## Canonical scenes

`src/alexdoor_xas/assets/door_task.py` owns one immutable pose registry and `outputs/door_scene/` contains exactly its five generated layers: D0 yaw `0.00`, XY `(0.00, 0.00)`; D1 yaw `+0.05`, XY `(+0.02, 0.00)`; D2 yaw `-0.05`, XY `(0.00, -0.02)`; D3 yaw `+0.10`, XY `(+0.02, +0.02)`; D4 yaw `-0.10`, XY `(+0.02, -0.02)`. D0 is the default. Runtime and data APIs accept only these pose IDs.

## Control and sensing

The executor uses position-only differential IK over the six right-arm joints. The Jacobian is shifted from the gripper-link origin to the calibrated collision tool point. Simulation runs at 120 Hz with decimation 2; A2 translation is limited to 0.02 m per control tick. Rotation remains represented but is not actuated.

Success is the first hinge crossing at 45 degrees. Contact force comes from PhysX raw GPU contact buffers and is selected by the exact door actor ID; it does not represent arbitrary robot/environment contacts.

## Limits

The benchmark is simulation-only. It has no physical robot command path, hardware calibration workflow, or hardware safety layer.

## Version Notes

- 2026-08-12 — Removed duplicate calibration authoring; the active JSON is the single source and changes require the maintained runtime gates.
- 2026-08-12 — Removed unused asset evidence fields and noncanonical scene-generation APIs while preserving runtime identity, D0-D4 validation, and physical scene checks.
- 2026-08-12 — Limited environment readiness to benchmark-required assets while retaining the optional combined hallway path for manual composition.
- 2026-08-12 — Moved generic Alex V2 configuration to the autonomous external extension while retaining every Door-specific runtime contract in this consumer.
- 2026-08-12 — Replaced transform-derived scene names with the exact D0-D4 registry and moved noncanonical generation to the runtime cache.
- 2026-08-12 — Consolidated asset and isolated-door checks into the production Alex V2 benchmark scene gate and separated calibration authoring by name.
- 2026-08-11 — Replaced the unsupported GPU shape-level contact filter with exact door-actor selection over raw GPU contact buffers.
- 2026-08-11 — Replaced the self-fingerprinted candidate/validated calibration pair with one minimal active calibration plus the existing live verifier.
