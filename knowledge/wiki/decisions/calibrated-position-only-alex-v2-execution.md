# Decision — Calibrated Position-Only Alex V2 Execution

## Context

The provisional Alex V1 path did not match the actual IHMC Alex V2 asset, and
the collision contact point is offset from the gripper link origin. A
full-pose controller would add orientation-control behavior that had not been
calibrated or validated for this benchmark.

## Decision

Bind the benchmark to one exact fixed-base Alex V2 runtime identity and one
self-fingerprinted door calibration. Execute A2 translation with position-only
differential IK over six right-arm joints at the collision-derived tool point.
Use collision-derived offset-point Jacobians and panel-filtered force evidence.

Retain the canonical six-dimensional A2/A3 representation and enforce rotation
limits, but do not actuate requested rotation in the current controller. Treat
the raw URDF identity, converted runtime identity, and calibration identity as
separate bindings.

## Consequences

- Kinematics, contact geometry, dataset provenance, and checkpoint provenance
  refer to the same calibrated robot.
- The controller is smaller and more verifiable than an uncalibrated full-pose
  path.
- Learned rotation values cannot influence current robot motion; representation
  conclusions must preserve that limitation.
- Asset, joint, actuator, PD, tool-frame, or calibration drift fails closed.
- Simulator validation does not authorize physical Alex execution.

## Evidence

- `configs/alex_v2_door_calibration.v0.json`
- `src/alexdoor_xas/assets/alex_v2_contract.py`
- `src/alexdoor_xas/assets/alex_v2_tool_frame.py`
- `src/alexdoor_xas/envs/door_task/door_push_robot_env.py`
- `src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py`
- `tests/test_alex_v2_door_calibration.py`
- `tests/test_alex_v2_executor_contract.py`

See [[topics/alex-v2-benchmark|Alex V2 Benchmark]].

## Version Notes

- 2026-07-08 — The calibrated position-only Alex V2 controller replaced the
  provisional V1 execution path.
- 2026-07-11 — Settle, terminal-force, and provenance checks tightened its
  evaluation boundary.
