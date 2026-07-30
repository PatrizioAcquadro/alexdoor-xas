# Alex V2 Benchmark

The current benchmark is a fixed-base IHMC Alex V2 torso pushing one simulated
hinged door through a calibrated six-joint right-arm controller. It replaced
the provisional Alex V1 path and is the only current robot execution contract.

## Asset Identity

The external URDF defaults to `~/Desktop/Alex/urdf/alex_v2.urdf` and can be
overridden with `ALEX_V2_ASSET_ROOT`. `src/alexdoor_xas/assets/alex_v2_manifest.py`
binds the raw file hash, 29 movable joints, and 32 primitive collision elements.

`src/alexdoor_xas/assets/alex_v2_contract.py` derives a separate runtime
fingerprint from the raw asset plus fixed-base conversion, joint order,
actuators, and PD settings. Dataset and checkpoint identity uses this derived
runtime contract; the raw and runtime hashes must not be conflated.

## Calibration

`configs/alex_v2_door_calibration.v0.json` binds:

- the robot runtime identity and base pose;
- six right-arm ready joints;
- the collision-derived tool point on `RIGHT_GRIPPER_Z_LINK`;
- a 0.2–0.8 m reach envelope;
- controller geometry and seven calibration gates;
- the calibration's own content fingerprint.

`src/alexdoor_xas/calibration/alex_v2_door.py` validates this immutable contract.
Authoring and probe code are separate so an ordinary run cannot silently
rewrite calibration to match a changed environment.

## Control

`src/alexdoor_xas/envs/door_task/door_push_robot_env.py` uses position-only
differential IK on the six right-arm joints. `src/alexdoor_xas/kinematics/offset_point.py`
transforms the Jacobian from the link origin to the collision-derived tool
point. Anti-windup logic keeps generated joint targets within the configured
limits.

The current PD gains are 600/15 for shoulder and elbow stiffness/damping and
150/4 for wrists. Simulation runs at 120 Hz with decimation 2, so the controller
runs at 60 Hz. A2 translation is limited to 0.02 m per tick. Rotation is present
in the six-dimensional request contract but is not actuated.

## Door and Sensing

The isolated task fixture is authored by
`src/alexdoor_xas/assets/door_task.py`. Success is the first hinge crossing at
`pi/4`. Door poses D0–D4 rotate/place the fixture around its hinge so the task
geometry remains consistent.

`src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py` exposes live
tool-point position/Jacobian and filters contact force to the door panel shape
prim `.../Door/Cylinder_001`. Non-panel robot collisions are not counted as
task-force evidence.

## Runtime Authority and Limits

The supported workstation stack is Isaac Sim 6.0.1 at
`/home/pacquadr/isaacsim` with the Alex-enabled Isaac Lab branch
`pacquadr/alex-v2-asset`. Validated simulation uses CPU physics; CUDA is used
for training and inference where configured.

This is simulation-only evidence. The repository has no physical Alex command,
hardware safety layer, or hardware calibration workflow. A physical progression
requires a separate, safety-reviewed scope.

## Primary References

- `src/alexdoor_xas/assets/alex_v2_manifest.py`
- `src/alexdoor_xas/assets/alex_v2_contract.py`
- `src/alexdoor_xas/calibration/alex_v2_door.py`
- `src/alexdoor_xas/envs/door_task/door_push_alex_v2_env.py`
- `src/alexdoor_xas/envs/door_task/door_push_robot_env.py`
- `src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py`
- `tests/test_alex_v2_manifest.py`
- `tests/test_alex_v2_door_calibration.py`
- `tests/test_alex_v2_executor_contract.py`

## Version Notes

- 2026-07-08 — The calibrated Alex V2 benchmark replaced the provisional V1
  asset, execution, data, and checkpoint identity.
- 2026-07-11 — Terminal-force, settle, and provenance semantics were hardened
  around the same robot/calibration contract.
