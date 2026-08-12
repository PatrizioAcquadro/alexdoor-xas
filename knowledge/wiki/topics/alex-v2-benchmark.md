# Alex V2 Benchmark

The current benchmark is a fixed-base IHMC Alex V2 torso pushing one simulated
hinged door through a six-joint right-arm controller.

## Asset and calibration

The external URDF defaults to `~/Desktop/Alex/urdf/alex_v2.urdf`.
`assets/alex_v2_contract.py` validates the robot asset and runtime identity.

`configs/alex_v2_door.json` is the single active calibration. It contains only
the task name, base pose, six initial joints, operational tool frame, reach
shell, controller parameters, and randomization limits. The production loader
validates structure and numerical safety.

`scripts/verify_benchmark_scene.py` is the non-mutating scene gate. It validates
the Alex asset and runtime manifest, generated door USD dependencies, hinge,
mass and inertia, exact 29-joint Alex runtime order, reset state, and zero-action
frame/door stability on the requested device. The combined hallway remains an
availability check in `scripts/check_env.py`; it is not benchmark physics.

`scripts/author_alex_v2_door_calibration.py` is a separate mutating maintenance
command with internal asset, tool-frame, reset, Jacobian, contact, and scripted
checks. It is not one of the routine verifiers.

## Control and sensing

The executor uses position-only differential IK over the six right-arm joints.
The Jacobian is shifted from the gripper-link origin to the calibrated collision
tool point. Simulation runs at 120 Hz with decimation 2; A2 translation is
limited to 0.02 m per control tick. Rotation remains represented but is not
actuated.

Success is the first hinge crossing at 45 degrees. Door poses D0-D4 move the
fixture around its hinge. Contact force comes from PhysX raw GPU contact buffers
and is selected by the exact door actor ID; it does not represent arbitrary
robot/environment contacts.

## Limits

The benchmark is simulation-only. It has no physical robot command path,
hardware calibration workflow, or hardware safety layer.

## Version Notes

- 2026-08-12 — Consolidated asset and isolated-door checks into the production
  Alex V2 benchmark scene gate and separated calibration authoring by name.

- 2026-08-11 — Replaced the unsupported GPU shape-level contact filter with
  exact door-actor selection over raw GPU contact buffers.
- 2026-08-11 — Replaced the self-fingerprinted candidate/validated calibration
  pair with one minimal active calibration plus the existing live verifier.
