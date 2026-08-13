# Alex V2 Benchmark

The maintained benchmark is simulated door pushing with the fixed-base IHMC Alex V2 torso and a six-joint right-arm controller.

## Asset and Calibration

The external `~/Desktop/Alex` repository owns the Alex V2 URDF and the generic `ihmc_alex_isaaclab` extension. DoorManipulation imports `make_alex_v2_cfg` and applies the door-task calibration locally.

`configs/alex_v2_door.json` is the only active calibration. It defines the task and robot identities, base pose, six initial right-arm joints, operational tool frame, reach shell, control parameters, and randomization bounds. There is no separate calibration-authoring workflow.

`src/alexdoor_xas/assets/alex_v2_contract.py` and the benchmark gate validate the external asset, fixed-base identity, joint order, tool frame, and runtime manifest.

## Canonical Door Scenes

`src/alexdoor_xas/assets/door_scene.py` defines the only accepted pose registry:

| Pose | Yaw | XY offset |
|---|---:|---:|
| D0 | 0.00 | (0.00, 0.00) |
| D1 | +0.05 | (+0.02, 0.00) |
| D2 | -0.05 | (0.00, -0.02) |
| D3 | +0.10 | (+0.02, +0.02) |
| D4 | -0.10 | (+0.02, -0.02) |

`outputs/door_scene/` contains exactly one USD layer per pose. D0 is the default. Noncanonical scenes must use an explicit path under the runtime cache and are not benchmark poses.

## Control and Sensing

`DoorPushAlexV2Env` is a single-environment Isaac Lab `DirectRLEnv`; construction rejects any other environment count. It resolves the calibrated six-joint right arm, gripper link, shoulder link, hinge, and exact door actor.

Control runs at 120 Hz with decimation 2. Position-only differential IK commands the collision-derived tool point. Translation is limited to 0.02 m per control tick. Rotational components remain represented in A2/A3 data and adapter decisions but are not actuated.

Success is the first 45-degree hinge crossing. Each runtime snapshot reads the raw GPU contact buffer once and retains only contacts whose opposite actor is the exact door panel. Unfiltered net force and geometric contact are not accepted as task-force substitutes; geometry remains diagnostic only.

## Verification

`scripts/verify_benchmark_scene.py` checks the external robot, canonical door dependencies and dynamics, runtime joint order, reset state, and zero-action stability. Calibration changes also require the maintained scripted-baseline and adapter gates.

## Limits

The benchmark is simulation-only, fixed-base, single-environment, and limited to one door family. It has no physical-robot command, hardware-calibration, or hardware-safety layer.

## Version Notes

- 2026-08-13 — Documented only the external Alex asset, one active calibration, D0-D4 scenes, and the concrete single-environment runtime.
