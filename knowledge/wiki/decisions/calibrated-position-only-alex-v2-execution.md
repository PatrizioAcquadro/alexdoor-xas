# Decision — Calibrated Position-Only Alex V2 Execution

## Context

The physical contact point is offset from the gripper-link origin, while full
pose control would add uncalibrated orientation behavior.

## Decision

Use the fixed-base Alex V2 runtime, one minimal door calibration, and
position-only differential IK over six right-arm joints at the
collision-derived tool point. Keep the six-dimensional A2/A3 interface but do
not actuate requested rotation. Filter task force to the door panel.

The live calibration verifier must reproduce the asset, tool frame, reset,
Jacobian, contact, and scripted behavior checks before writing
`configs/alex_v2_door.json`.

## Consequences

- The controller stays small and directly testable.
- Learned rotation values do not influence current motion.
- Asset and robot compatibility remain explicit where they affect execution.
- Simulator evidence does not authorize physical Alex operation.

See [[topics/alex-v2-benchmark|Alex V2 Benchmark]].

## Version Notes

- 2026-08-11 — Calibration state was reduced to operational parameters; live
  verification remains the authoring gate.
