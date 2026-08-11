# Alex V2 Benchmark

The current benchmark is a fixed-base IHMC Alex V2 torso pushing one simulated
hinged door through a six-joint right-arm controller.

## Asset and calibration

The external URDF defaults to `~/Desktop/Alex/urdf/alex_v2.urdf`.
`assets/alex_v2_contract.py` validates the robot asset and runtime identity.

`configs/alex_v2_door.json` is the single active calibration. It contains only
the task name, base pose, six initial joints, operational tool frame, reach
shell, controller parameters, and randomization limits. The production loader
validates structure and numerical safety. The live verifier separately checks
the current asset, collision-derived tool frame, reset behavior, Jacobians,
contact behavior, and scripted performance before writing the calibration.

## Control and sensing

The executor uses position-only differential IK over the six right-arm joints.
The Jacobian is shifted from the gripper-link origin to the calibrated collision
tool point. Simulation runs at 120 Hz with decimation 2; A2 translation is
limited to 0.02 m per control tick. Rotation remains represented but is not
actuated.

Success is the first hinge crossing at 45 degrees. Door poses D0-D4 move the
fixture around its hinge. Contact force is filtered to the door panel and does
not represent all robot/environment contacts.

## Limits

The benchmark is simulation-only. It has no physical robot command path,
hardware calibration workflow, or hardware safety layer.

## Version Notes

- 2026-08-11 — Replaced the self-fingerprinted candidate/validated calibration
  pair with one minimal active calibration plus the existing live verifier.
