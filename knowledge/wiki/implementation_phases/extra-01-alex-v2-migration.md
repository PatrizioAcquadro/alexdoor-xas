# Extra 01 — Alex V2 Migration

## Objective

Replace every provisional Alex V1 dependency with the actual fixed-base IHMC
Alex V2 asset, establish a measured calibration and execution contract, and
regenerate the matched dataset and learned baselines against that contract.

## Focus

### Subphase Extra 01.1 — Asset Identity and Contract

#### Implementation

`src/alexdoor_xas/assets/alex_v2_manifest.py` inventories the external Alex V2
URDF by exact content hash, 29 movable joints, and 32 primitive collision
elements. `src/alexdoor_xas/assets/alex_v2_contract.py` derives the runtime
identity from the source asset plus fixed-base conversion, joint ordering,
actuator mapping, and PD configuration. The raw URDF hash and the derived
runtime fingerprint are intentionally different identities.

`src/alexdoor_xas/assets/alex_v2.py` converts and configures the articulation,
and `src/alexdoor_xas/envs/door_task/alex_v2_runtime.py` verifies the live
articulation against the contract. The migration removed the V1 executor and
old asset assumptions rather than maintaining parallel robot paths.

#### Key Decisions and Problems

- The benchmark fails closed if the URDF, movable-joint order, collision
  inventory, actuator layout, or fixed-base settings drift.
- External robot assets remain machine-local and are never copied into the
  repository.
- A dataset or checkpoint robot binding uses the derived runtime identity, not
  merely the raw URDF hash.

#### Tests

- `tests/test_alex_v2_manifest.py`, `tests/test_alex_v2_infrastructure.py`, and
  `tests/test_alex_v2_task_registration.py` verify the source inventory,
  runtime contract, and environment registration.
- `scripts/check_env.py` and `scripts/verify_assets.py` were updated to reject
  stale V1 paths and validate the supported Alex-enabled Isaac Lab checkout.

### Subphase Extra 01.2 — Calibration and Tool Geometry

#### Implementation

`configs/alex_v2_door_calibration.v0.json` is a self-fingerprinted calibration
contract. It binds the robot identity, base pose, six ready right-arm joints,
reach envelope, controller geometry, collision-derived tool point on
`RIGHT_GRIPPER_Z_LINK`, and seven calibration gates.

`src/alexdoor_xas/calibration/alex_v2_door.py` validates the committed contract;
`alex_v2_door_authoring.py` supports measured authoring and probe results.
`src/alexdoor_xas/assets/alex_v2_tool_frame.py` and
`src/alexdoor_xas/kinematics/offset_point.py` derive the commanded tool point
and its Jacobian from collision geometry rather than assuming the link origin
is the contact point.

The current PD values are 600 stiffness / 15 damping for shoulder and elbow
joints and 150 / 4 for wrists. The verified reach envelope is 0.2–0.8 m. These
values are part of the calibrated benchmark rather than generic Alex V2
properties.

#### Key Decisions and Problems

- Calibration is immutable input to ordinary runs. An unexplained fingerprint
  mismatch is an error, not a reason to refresh the expected value.
- The collision-derived tool point keeps kinematics, contact geometry, and
  controller targets aligned.
- The workstation still contains an unrelated dangling `thor` asset reference;
  the isolated Alex-door benchmark does not require it.

#### Tests

- `tests/test_alex_v2_door_calibration.py`,
  `tests/test_alex_v2_door_calibration_authoring.py`,
  `tests/test_alex_v2_tool_frame.py`, and
  `tests/test_offset_point_kinematics.py` cover self-fingerprints, authoring
  constraints, collision offsets, and Jacobians.
- `scripts/probe_alex_v2_door_calibration.py` and the closeout calibration
  evidence verified all seven gates on CPU simulation.

### Subphase Extra 01.3 — Position-Only Execution

#### Implementation

`src/alexdoor_xas/envs/door_task/door_push_robot_env.py::DoorPushRobotEnv`
implements position-only differential IK for the six right-arm joints. It
maps the A2 translational command at the collision-derived tool point into
bounded joint targets and applies anti-windup clamping. Requested rotational
components remain in the six-dimensional contract for representation
compatibility, but the current robot controller does not actuate them.

`door_push_alex_v2_env.py::DoorPushAlexV2Env` loads the exact asset and validated
calibration. `door_push_alex_v2_executor.py::DoorPushAlexV2Executor` supplies
live tool-point kinematics and force evidence filtered to the door panel shape
prim `.../Door/Cylinder_001`. Generic or non-panel contact does not count as the
task force signal.

The complete current contract is maintained in [[topics/alex-v2-benchmark|Alex V2 Benchmark]],
and its architectural rationale is recorded in
[[decisions/calibrated-position-only-alex-v2-execution|Calibrated Position-Only Alex V2 Execution]].

#### Key Decisions and Problems

- Validated simulation uses CPU physics; CUDA is used for training and policy
  inference where configured.
- Force evidence is door-panel-filtered to avoid treating unrelated robot
  collision as task contact.
- This repository provides no physical Alex execution command. Simulator
  validation must not be presented as hardware validation.

#### Tests

- `tests/test_alex_v2_executor_contract.py`, `tests/test_joint_limits.py`, and
  `tests/test_alex_v2_episode_provenance.py` verify executor state, filtering,
  bounds, and recorded identity.
- `scripts/verify_alex_v2_door_baseline.py` is the full CPU simulator gate for
  the calibrated reference execution.

### Subphase Extra 01.4 — Matched V2 Data and Retraining

#### Implementation

`configs/door_pose_plan_v2_pose.json` defines five door poses, D0–D4. The
migration generated 50 successful physical episodes and published matched A2
and A3 `v2_pose` products with shared identities, splits, training-only
normalization, and distinct action values. ACT and Diffusion checkpoints were
then retrained and rebound to the new dataset and robot fingerprints.

The resulting dataset was a correctness and local-smoke foundation, not the
later scale master. [[extra-04-scale-dataset|Extra 04]] introduced the 550-
episode `v3_scale` family.

#### Key Decisions and Problems

- A2 and A3 share physical trajectories but remain numerically and semantically
  distinct; one representation is not relabeled as the other without the
  explicit frame transform.
- V1 datasets and checkpoints are historical only and cannot satisfy current
  Alex V2 provenance.

#### Tests

- `tests/test_alex_v2_dataset_fingerprint.py` verifies robot and dataset
  binding, and `scripts/verify_a2_a3_distinct.py` rejects collapsed action
  representations.
- The phase closeout recorded 50/50 successful physical-master episodes across
  D0–D4 and successful retraining/closed-loop gates for both policy families.

## Version Notes

- 2026-07-08 — Alex V2 asset identity, calibration, position-only execution,
  matched `v2_pose` data, and rebound policies replaced the V1 stack.
- 2026-07-09 onward — Stabilization tightened evaluation and provenance around
  the same calibrated benchmark.
