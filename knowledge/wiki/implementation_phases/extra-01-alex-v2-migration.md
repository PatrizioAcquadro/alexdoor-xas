# Extra 01 — Alex V2 Migration

## Objective

Replace provisional robot assumptions with the fixed-base IHMC Alex V2 door
benchmark.

## Focus

### Subphase E1.1 — Asset, calibration, and execution

#### Implementation

Validated the Alex V2 asset and joint order, derived the gripper collision tool
point, implemented offset-point Jacobians, six-joint position-only IK, and
door-panel force sensing. The active calibration is now
`configs/alex_v2_door.json`.

#### Key Decisions and Problems

- Requested rotation remains represented but is not actuated.
- Live verification gates calibration authoring; gates and runtime history are
  not stored in the active config.
- Simulation results are not physical-robot safety evidence.

#### Tests

Asset, tool-frame, calibration, runtime injection, executor, and live scripted
smoke checks cover the maintained path.

## Version Notes

- 2026-08-11 — Replaced candidate/validated calibration contracts with one
  minimal operational calibration.
