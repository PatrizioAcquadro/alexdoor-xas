# Phase 2 — Scripted Baseline and Data Engine

## Objective

Build a deterministic door-pushing reference system that can execute the task,
record authoritative trajectories, export matched action representations, and
measure correctness before learned policies are introduced.

## Focus

### Subphase 2.1 — Door Task and Scripted Controller

#### Implementation

`src/alexdoor_xas/assets/door_task.py` constructs the isolated door fixture,
including the panel, hinge, world anchor, collision geometry, and posed variants
whose placement pivots about the hinge. `src/alexdoor_xas/envs/door_task/door_push_env.py`
defines the simulator timing and task state: physics runs at 120 Hz, commands
are applied at 60 Hz through decimation 2, and success is the first hinge-angle
crossing at `pi/4`.

`src/alexdoor_xas/policies/scripted/door_push.py::DoorPushController` is a
pure-NumPy finite-state controller with APPROACH, ALIGN, PRE_CONTACT, CONTACT,
PUSH, HOLD, and RELEASE states. It consumes task observations and emits
six-dimensional A3 translation/rotation deltas plus A4 intent records. The
controller is a deterministic data and evaluation reference, not an optimal
policy claim.

The originally landed physical executor targeted a provisional Alex V1 model.
[[extra-01-alex-v2-migration|Extra 01]] deleted that path and introduced the
current calibrated Alex V2 executor. Current action and frame meanings are
canonicalized in [[topics/action-representations-and-adapters|Action Representations and Adapters]].

#### Key Decisions and Problems

- The isolated door fixture avoids unrelated CombinedScene composition failures
  and gives the task an explicit world anchor and validated hinge.
- Per-tick A2/A3 bounds are 0.02 m translation and 0.05 rad rotation. These are
  control-period limits, not episode-level workspace bounds.
- Current geometric contact checks do not enforce the panel Z extent. The
  approved code-quality roadmap identifies this as unresolved work; it is not
  silently treated as fixed.

#### Tests

- `tests/test_door_task_assets.py` and `tests/test_door_task_env.py` verify
  fixture construction contracts, timing, state, and success semantics.
- `tests/test_scripted_door_push.py` covers controller transitions,
  deterministic outputs, and action limits.
- `scripts/verify_door_task_scene.py`, `scripts/verify_door_env.py`, and
  `scripts/verify_scripted_baseline.py` provide progressively broader CPU
  simulator gates.

### Subphase 2.2 — Episode Execution and Serialization

#### Implementation

`src/alexdoor_xas/recording/episode.py` defines episode metadata, per-tick
steps, outcomes, and the in-memory buffer. A step stores the pre-action
observation and force/contact sample together with the command selected from
that state. `terminal_contact` separately records the response to the final
executed action, preserving the temporal contract.

`src/alexdoor_xas/data_engine/generate.py::run_episode` performs reset,
observation, policy selection, recording, environment stepping, and terminal
outcome assembly. `src/alexdoor_xas/recording/writer.py` publishes the
`phase2.v1` format: one HDF5 trajectory plus one JSON sidecar for A1/A2/A3, and
JSON Lines for A4 chunks. The complete data contract is documented in
[[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

Two current reliability gaps are documented rather than obscured. The standard
writer publishes HDF5 and sidecar files sequentially rather than as one
transaction, and `run_episode` appends the pending step before confirming that
`env.step` returned successfully. The code-quality roadmap proposes fixes, but
those fixes are not in the current implementation.

#### Key Decisions and Problems

- Observations and force are pre-action by contract. Reinterpreting them as
  post-action would misalign policy inputs, actions, and safety evidence.
- Serialization is versioned and validated instead of exposing raw HDF5 keys
  as the model API.
- Runtime output remains ignored; only deliberately curated small evidence is
  tracked under `outputs/curated/`.

#### Tests

- `tests/test_recording.py` verifies schema fields, shape checks, sidecars, and
  round-trip behavior.
- `tests/test_data_engine.py` checks episode execution, terminal state
  accounting, exports, and deterministic reference behavior.
- Phase closeout recorded eight of eight successful reference episodes with
  exact repeatability in the same execution mode.

### Subphase 2.3 — Matched Exports and Evaluation

#### Implementation

`src/alexdoor_xas/data_engine/export.py` converts a physical episode identity
into representation-specific products. A2 retains world-frame end-effector
deltas; A3 stores the equivalent command in the static hinge-anchored door
frame; A4 stores guarded object-centric intent chunks in the moving panel
frame; Alex executions can derive A1 joint-target deltas. Matched representations
therefore share physical trajectories rather than independently generated task
instances.

`src/alexdoor_xas/eval/metrics.py`, `failures.py`, `sanity.py`, `plots.py`, and
`report.py` define success, force and efficiency summaries, failure categories,
sanity checks, and review artifacts. Later stabilization work tightened
first-crossing, terminal-force, provenance, and matched-seed semantics; see
[[extra-02-local-stabilization|Extra 02]].

#### Key Decisions and Problems

- The phase compares representations through matched episode identities to
  reduce task-distribution confounding. See [[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]].
- Ordinary version export replaces the owned target version. Changed generation
  must use a new version and regenerate splits and training-only normalization.
- Historical V1 physical results establish implementation progress only; they
  do not validate the current Alex V2 benchmark.

#### Tests

- `tests/test_action_spaces.py` verifies canonical tags, dimensions, and frame
  transforms.
- `tests/test_eval.py` checks metric aggregation, failure taxonomy, sanity
  checks, and report inputs.
- `scripts/verify_dataset_interface.py` and later
  `scripts/verify_a2_a3_distinct.py` verify representation and dataset
  contracts on published data.

## Version Notes

- 2026-07-03 — Door task, scripted controller, data engine, serialization,
  matched exports, and evaluation framework landed.
- 2026-07-08 onward — Alex V2 execution and later evaluation hardening replaced
  provisional robot behavior without changing the core pre-action episode
  contract.
