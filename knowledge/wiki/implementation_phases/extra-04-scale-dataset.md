# Extra 04 — Scale Dataset

## Objective

Create one larger, deterministic Alex V2 physical-trajectory master and publish
matched A2/A3 datasets with nested training views so data-scale comparisons use
the same task distribution, validation set, and test set.

## Focus

### Subphase Extra 04.1 — Scale Pose Plan and Physical Master

#### Implementation

`configs/door_pose_plan_v3_scale.json` defines a five-pose D0–D4 generation plan
with 750 deterministic candidate identities. `scripts/build_scale_dataset.py`
drives generation, publication, and verification through explicit subcommands.
The accepted physical master contains 550 successful Alex V2 episodes; the
remaining 200 candidates are retained as `NOT_NEEDED_OVERDRAW`, not silently
discarded failures.

Each physical episode is generated once and carries robot, calibration,
simulator, pose-plan, and source identity. A2 and A3 are then exported from the
same master, preserving episode IDs and physical outcomes. The resulting
`v3_scale` family still represents one simulated door family with state-only
observations; scale does not broaden embodiment, task, or modality coverage.

#### Key Decisions and Problems

- One physical master prevents A2/A3 comparison from being confounded by
  independently sampled rollouts.
- Candidate overdraw is declared before generation so surplus successful
  candidates have a deterministic status rather than an outcome-dependent
  inclusion rule.
- The scale dataset is reusable training data under `datasets/`, not a tracked
  experiment artifact.

#### Tests

- `tests/test_scale_dataset.py` verifies candidate selection, generation
  accounting, physical-master identity, paired publication, and fail-closed
  verification.
- The completed run recorded 550 selected episodes and exactly 200
  `NOT_NEEDED_OVERDRAW` candidates across D0–D4.

### Subphase Extra 04.2 — Atomic Paired Publication

#### Implementation

`src/alexdoor_xas/data_engine/export.py` publishes the A2 and A3 dataset roots
as one owned pair. Both views receive the same episode identities, shared
split membership, representation-appropriate action arrays, and a binding to
the one physical master. Publication stages complete products before replacing
the owned targets, avoiding a mixed A2/A3 generation.

The A2 and A3 master fingerprints are distinct because their action values and
frame semantics differ, while the shared physical-source fingerprint proves
common origin. `src/alexdoor_xas/dataset/validate.py` rejects inconsistent
schema, provenance, pairing, or normalization rather than refreshing expected
fingerprints.

#### Key Decisions and Problems

- A generation pass owns a dataset version. Changed generation requires a new
  version and refreshed shared split/train-only normalization artifacts.
- Atomicity is implemented for this paired scale publication path. The ordinary
  per-episode HDF5/JSON writer still has the non-transactional gap documented
  in [[phase-2-scripted-baseline-and-data-engine|Phase 2]].

#### Tests

- `tests/test_scale_dataset.py` injects publication and validation failures to
  verify that an incomplete pair does not become authoritative.
- `scripts/build_scale_dataset.py verify` checks exact episode counts, matched
  IDs, representation distinction, fingerprints, and required metadata.

### Subphase Extra 04.3 — Nested Views and Normalization

#### Implementation

`src/alexdoor_xas/dataset/views.py` constructs four views: N50, N100, N250, and
N500. All use the same 25 validation and 25 test episodes; only the training
set grows. Selection is deterministic, pose-aware, and nested, so every smaller
training set is a subset of every larger applicable set.

Each action space and training size has its own training-only normalization
artifact, producing eight normalization products. View definitions contain
ordered IDs and a selection SHA-256 so a name such as N100 cannot be rebound to
different episodes unnoticed. The rationale is recorded in
[[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]].

#### Key Decisions and Problems

- Fixed holdouts make performance changes attributable to training data scale
  rather than evaluation-set drift.
- Normalization is recomputed per training view; using N500 statistics for N50
  would leak information across the intended comparison.
- View identity includes exact membership and ordering, not only a count.

#### Tests

- `tests/test_dataset_views.py` verifies nesting, fixed holdouts, deterministic
  selection hashes, pose balance, and fingerprint failures.
- The phase closeout verified all four views for both A2 and A3, including the
  eight train-only normalization bindings.

## Version Notes

- 2026-07-15 — The 550-episode paired `v3_scale` master, fixed holdouts, four
  nested views, and eight normalization artifacts were published and verified.
- 2026-07-16 onward — Cluster sweep and unified evaluation consumed these
  immutable dataset/view identities without regenerating them.
