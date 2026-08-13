# Decision — Calibrated Position-Only Alex V2 Execution

## Context

The Alex V2 benchmark needs a reproducible controller at the physical gripper contact point while preserving the A2/A3 representation contract.

## Decision

Use the fixed-base Alex V2 torso, the six calibrated right-arm joints, and position-only differential IK at the collision-derived tool point. Keep rotational A2/A3 components in data and adapter decisions, but do not command them to the robot.

Use `configs/alex_v2_door.json` as the only task calibration. Generic Alex construction remains in the external `ihmc_alex_isaaclab` extension; door calibration, IK, contact selection, and safety semantics remain local.

Accept task force only from exact-door raw PhysX contact selection. Geometric contact may be recorded for diagnosis but cannot replace sensed force.

## Consequences

- The current action execution is intentionally translation-only.
- Calibration changes require the benchmark, scripted-baseline, and adapter gates.
- The controller is specialized to the single-environment simulated Alex V2 benchmark.
- Simulator force thresholds and success do not establish physical-robot safety.

The retired calibration-authoring, generic executor, sensorless, and surrogate-robot paths are not part of this decision.

## Version Notes

- 2026-08-13 — Restated the active decision around one calibration, position-only tool-point IK, and exact-door contact sensing.
