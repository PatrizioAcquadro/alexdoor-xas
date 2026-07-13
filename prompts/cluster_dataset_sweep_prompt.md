# TASK: Prepare the Full Nested Dataset-Scale Sweep on Ubuntu

## Objective

Implement, generate, and validate the complete Ubuntu-side package for the later Gilbreth
dataset-scale sweep.

This task prepares the datasets, reproducibility contracts, transfer/return tooling, and 16-cell
Slurm renderer. It must **not** transfer files to Gilbreth, call `sbatch`, run the full sweep, start
closed-loop sweep evaluation, or begin Phase 4/VLA work.

**DO NOT USE SUBAGENTS.** Work single-agent from inspection through validation.

Use the `robotics-test` workflow first to freeze regression tests and acceptance criteria, then use
`robotics-implementation` to implement them. Keep a local-only `TODO.md` and mark each gate complete
as it passes. Do not commit `TODO.md`.

## Verified Starting State

- Repository: `/home/pacquadr/Desktop/DoorManipulation`
- Expected starting `main`: `252fb91d370ddeb5e391306aed6e504e2dbf710a`
- Local stabilization is complete.
- Gilbreth compatibility pilot `11279452` completed and returned successfully.
- W&B durable-publication canary `11279800` completed successfully:
  - ACT-A2 and Diffusion-A3 both completed;
  - durable W&B symlink count was zero;
  - direct return packaging passed without manual staging;
  - Ubuntu verified all returned hashes and loaded both checkpoints on CPU.
- The existing `v2_pose` dataset is compatibility/stabilization evidence only.
- `configs/cluster_pilot_n50.v1.json` records the future sweep contract but intentionally does not
  implement or authorize it.

Before editing, read completely:

- `CLAUDE.md`
- `docs/PROJECT_GUIDELINES.md`
- `docs/architecture.md`
- `docs/development.md`
- `docs/status.md`
- `docs/cluster.md`
- `configs/cluster_pilot_n50.v1.json`
- the dataset generation, split, normalization, training, manifest, Slurm, W&B publication, and
  return implementations and tests they reference.

The only allowed tracked change at task start is this prompt file. If that is the only change,
review its diff and commit it alone on `main` with the message `Define full cluster sweep task`.
Do not push it yet and do not mix it with implementation changes. If any other tracked or untracked
change exists, stop and report it.

Require the prompt commit to sit directly on the synchronized expected `origin/main`, with no other
local commits or changes. Preserve the ignored local `CLAUDE.md`, existing datasets, pilot
manifests, returned attempts, cluster evidence, and unrelated user files. Create a task branch such
as `impl/full-cluster-dataset-scale-sweep` from the local prompt commit.

If the starting commit differs, inspect the additional commits and continue only if they do not
change this contract. Otherwise stop and report the conflict.

## Frozen Scientific Contract

### Dataset identity

Create one new deterministic master dataset version:

```text
task: door_push_alex_v2
master version: v3_scale_master
observation preset: core_door_pose
door poses: D0, D1, D2, D3, D4
source episodes: 550 successful, sanity-clean, content-distinct episodes
episodes per pose: 110
```

Reuse the exact validated D0-D4 door geometry and Alex V2 calibration. Do not modify `v2_pose`, the
robot asset, controller safety limits, force admission, action definitions, observation definitions,
or adapter behavior.

Generate the master pool once on Ubuntu with the official IsaacLab launcher. Do not independently
generate N50, N100, N250, and N500 datasets.

The 550 master episodes must be randomized, independently grouped trajectories. Fixed-start pose
probes may be run before generation but must not be included as duplicated source episodes. Use
explicit deterministic, non-overlapping seed and overdraw namespaces. Record every failed/skipped
candidate and its deterministic replacement; never hide failures or admit partial/unsafe episodes.

### Fixed holdouts and nested training views

Create four deterministic views:

| View | Training | Validation | Test | Total referenced |
|---|---:|---:|---:|---:|
| N50 | 50 | 25 | 25 | 100 |
| N100 | 100 | 25 | 25 | 150 |
| N250 | 250 | 25 | 25 | 300 |
| N500 | 500 | 25 | 25 | 550 |

`N` means the exact number of **training episodes**.

Every view must satisfy:

- validation IDs are identical across all four views;
- test IDs are identical across all four views;
- validation and test contain exactly five episodes from each pose;
- training is pose-balanced:
  - N50: 10 per pose;
  - N100: 20 per pose;
  - N250: 50 per pose;
  - N500: 100 per pose;
- `train(N50) ⊂ train(N100) ⊂ train(N250) ⊂ train(N500)`;
- train, validation, and test are pairwise disjoint;
- no trajectory-content group crosses a split;
- all selected source episodes are content-distinct;
- A2 and A3 use the exact same source episode IDs and split membership.

Use deterministic per-pose ordering and record its seed/algorithm. Holdouts must be selected once
from the master pool and excluded before constructing nested training prefixes.

### Efficient dataset views

Do not duplicate or regenerate HDF5 episode payloads for every N.

Keep one physical master export per action space:

```text
datasets/door_push_alex_v2/A2_ee_delta/v3_scale_master/
datasets/door_push_alex_v2/A3_obj_rel_ee_delta/v3_scale_master/
```

Add a backward-compatible dataset-view contract so training can select a split independently from
the physical dataset version. Use canonical view IDs:

```text
v3_scale_n50
v3_scale_n100
v3_scale_n250
v3_scale_n500
```

Existing configurations without an explicit view/split ID must continue resolving splits and
normalization exactly as before. Do not break `v2_pose`, its checkpoints, or the pilot.

Store one shared split file per view and one train-only normalization file per action-space/view.
Normalization statistics must use only that view's training IDs. They must bind the master dataset
fingerprint, view/split fingerprint, observation preset, action space, exact training IDs, and
normalization-file hash. Validation/test data must never influence normalization.

### Training matrix

Create exactly 16 cells:

```text
4 dataset views × 2 policies × 2 action spaces
```

For every N, include:

- ACT-A2
- ACT-A3
- Diffusion-A2
- Diffusion-A3

Use seed `0`, `core_door_pose`, `train.device=cuda`, `train.overfit_episodes=null`, and explicit W&B
offline mode. Use the committed non-pilot ACT and Diffusion training defaults unless a live code
constraint proves they are invalid:

- ACT: 100 epochs, validation every 5 epochs;
- Diffusion: 300 epochs, validation every 10 epochs, EMA enabled, 10-step DDIM validation metric;
- preserve the existing model, optimizer, horizon, chunk, and scheduler defaults;
- do not tune hyperparameters separately by N or action space;
- do not reuse the two-epoch/two-episode compatibility-pilot limits.

Use deterministic run IDs:

```text
sweep_act_a2_n50_seed0
sweep_act_a3_n50_seed0
sweep_diffusion_a2_n50_seed0
sweep_diffusion_a3_n50_seed0
...
sweep_diffusion_a3_n500_seed0
```

Each cell is an independent **single-GPU** training job. Do not add DeepSpeed, ZeRO, DDP, or any
multi-GPU model sharding. Render a stable `0-15` Slurm array with configurable concurrency and a
conservative default of `%2`.

Gilbreth remains non-Isaac:

- no Isaac Sim/Lab installation or import;
- no dataset generation;
- no closed-loop simulation evaluation;
- only package verification, preflight, GPU training, open-loop metrics, durable publication, and
  return packaging.

## Required Implementation

### 1. Tests and versioned configuration

Write failing regressions first, then implement a versioned sweep configuration such as:

```text
configs/cluster_sweep.v1.json
configs/door_pose_plan_v3_scale.json
```

The configuration must freeze:

- master dataset/version and five poses;
- source count and per-pose count;
- view IDs and exact train/validation/test counts;
- deterministic selection/seed contract;
- A2/A3 pairing;
- `core_door_pose`;
- 16-cell identity and stable task-index mapping;
- training defaults and seed;
- W&B offline mode;
- one GPU per task and configurable array concurrency;
- required source/dependency inventory;
- scratch and durable attempt layout;
- no-Isaac cluster boundary.

Reject missing keys, unknown keys, duplicate cells, drifted counts, unequal pose balance, changed
holdouts, non-nested training IDs, pilot overfit settings, online W&B, or distributed training.

### 2. Master generation and deterministic view builder

Extend the existing pose-plan/generation/export flow instead of building a second data engine.

Provide a deterministic orchestration and verification surface that:

- runs one Isaac process per pose using the official launcher;
- resumes safely without silently accepting incomplete output;
- records candidate, skipped, replacement, and selected episode provenance;
- selects exactly 110 clean, successful, content-distinct episodes per pose;
- exports A2 and A3 once from the same 550 source episodes;
- verifies force/sanity gates and Alex V2 asset provenance;
- builds the four shared split/view files;
- computes eight train-only normalization artifacts;
- writes a master manifest plus per-view fingerprints/hashes;
- fails closed before publishing an incomplete or inconsistent official dataset.

Use atomic publication for the official master/view metadata. Preserve failed generation output for
diagnosis, but never let it become an official dataset.

### 3. Backward-compatible data loading and provenance

Add the minimum backward-compatible configuration needed to select `dataset.version` and a separate
view/split ID.

Bind every training run and checkpoint to:

- master dataset fingerprint;
- view/split ID and fingerprint;
- exact train/validation/test IDs and counts;
- normalization artifact and SHA-256;
- action space and `core_door_pose`;
- source Git commit;
- resolved training configuration.

Checkpoint loading, evaluation metadata, and return verification must fail closed on any mismatch.
Existing `v2_pose` loading and checkpoint tests must remain unchanged and pass.

### 4. Sweep transfer manifest

Implement a sweep-specific clean-tree manifest builder/verifier, reusing hardened pilot helpers
where appropriate rather than copying them.

The manifest must include exactly the required:

- master A2/A3 datasets and metadata;
- four shared split/view files;
- eight normalization artifacts;
- sweep configuration and source files;
- pinned non-Isaac Python environment specification;
- Slurm/preflight/publication/return tooling;
- Alex V2 URDF provenance hash, without transferring Isaac.

It must bind the clean source commit, exact inventory, sizes, SHA-256 hashes, master/view/dataset
fingerprints, pose/count/nesting invariants, and secret scan. Generate checksum-based outbound
`rsync-files.txt` and command artifacts. Never include credentials, W&B keys, `.netrc`, private
keys, authenticated URLs, or unrelated outputs.

Preserve all historical pilot and stabilization manifests unchanged.

### 5. Gilbreth preflight and 16-cell Slurm renderer

Reuse the verified Python 3.11/CUDA non-Isaac environment contract and the successful pilot/canary
launcher patterns.

The renderer must:

- accept account, partition, optional QOS, depot root, scratch root, durable-results root, memory,
  CPUs, wall time, and concurrency at render time;
- render exactly 16 stable task mappings;
- use one GPU per task and optionally enforce A100 80GB;
- invoke only `$CONDA_PREFIX/bin/python`, never environment activation or inherited tool shims;
- verify the sweep manifest before training;
- run pure and live-CUDA preflight;
- use attempt-specific scratch and durable paths keyed by array job ID/task ID/run ID;
- preserve all resolved configs, checkpoints, logs, open-loop metrics, environment reports, and
  Slurm evidence;
- reuse the verified symlink-free W&B publisher before the final atomic durable move;
- fail closed on publication/status errors, duplicates, unsafe W&B entries, source mismatch, or
  missing completion evidence;
- never overwrite or silently reuse an earlier attempt.

Render and syntax-check the script locally with example paths. Do not call `salloc`, `srun`, or
`sbatch` in this task.

### 6. Sweep return tooling

Implement exact-attempt return build/verify tooling for all 16 cells.

Require:

- exactly the configured 16 task identities;
- completion status for every cell;
- no mixed attempt/job/task/source identities;
- no missing or duplicate cells;
- no failure records;
- symlink-free durable results and W&B publication reports;
- all expected checkpoints/configs/logs/metrics/environment evidence;
- SHA-256 verification of every returned payload;
- CPU loading of every returned best checkpoint without Isaac;
- checksum-based return file list and command.

No staging workaround is allowed. A failed or partial sweep must not produce a verified return
package. Do not automatically resubmit failed cells.

### 7. Documentation

Update only durable, relevant documentation:

- `docs/architecture.md`: implemented dataset-view, sweep, provenance, and publication contracts;
- `docs/development.md`: exact Ubuntu build/generation/verification commands and later transfer
  workflow;
- `docs/status.md`: datasets/package prepared locally, full sweep not transferred or launched;
- `docs/cluster.md`: concise later Gilbreth verification/render/submit/return sequence.

Keep historical job IDs `11279452` and `11279800` as evidence. Do not claim the full sweep ran or
produced scientific conclusions.

## Required Regression Coverage

At minimum, tests must prove:

- exact 550/110-per-pose master contract;
- exact N and per-pose counts for all views;
- identical fixed validation/test IDs across views;
- strict training nesting and split disjointness;
- content-group non-leakage and duplicate rejection;
- exact A2/A3 source-ID pairing and numerical distinctness;
- deterministic replay of view generation and fingerprints;
- train-only, per-view normalization with provenance/hash validation;
- backward compatibility for `v2_pose` configs/checkpoints;
- complete, unique 16-cell mapping and stable run IDs;
- no pilot overfit settings, online W&B, Isaac imports, activation, or distributed training;
- exact manifest inventory/hash/secret checks and clean-tree binding;
- polluted-`PATH` generated-script execution through `$CONDA_PREFIX/bin/python`;
- one-GPU Slurm directives, configurable `%2` default concurrency, and `bash -n`;
- attempt isolation, retry/no-overwrite behavior, durable failure publication, atomic success
  publication, and W&B sanitization;
- exact 16-cell return selection, no-symlink rule, hash verification, and CPU checkpoint loading;
- failure on missing, duplicate, mixed, partial, stale, or tampered evidence.

Use small synthetic fixtures for unit/integration tests; do not require 550 real episodes in the
ordinary pytest suite.

## Execution and Validation Order

1. Create/update local-only `TODO.md` with all gates.
2. Inspect current contracts and write regression tests first.
3. Implement configuration, dataset views, provenance, sweep tooling, and documentation.
4. Run focused tests until green.
5. Run the full existing suite and static checks before expensive generation.
6. Generate the real 550-episode master pool once on Ubuntu.
7. Build and verify all four views, eight normalization artifacts, and fingerprints.
8. Rerun dataset/sweep focused verification against the real artifacts.
9. Commit the validated code/config/docs on the task branch.
10. From the clean committed checkout, build and verify the final sweep transfer manifest and
    checksum file list.
11. Stop and report. Do not push, merge, transfer, allocate a GPU, submit, or train the full sweep
    without a separate explicit user instruction.

Use the official launcher for all repo Python/Isaac validation:

```text
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script-or--m-module>
```

Also require:

```text
/home/pacquadr/.local/bin/ruff check .
git diff --check
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
bash -n <every generated/bootstrap Slurm or shell script>
```

Run existing dataset, ACT, Diffusion, stabilization, pilot, W&B publication, returned-checkpoint,
and historical-manifest regressions. Historical hashes and returned pilot evidence must remain
unchanged.

## Stop Conditions

Stop and report with exact evidence if:

- the starting checkout is dirty or contains unrelated user changes that overlap this task;
- the five frozen door poses or Alex V2 calibration no longer pass their existing gates;
- 110 safe, successful, content-distinct episodes cannot be obtained for any pose;
- generation requires weakening force, sanity, success, adapter, or provenance gates;
- any view is imbalanced, non-nested, overlapping, or changes holdout IDs;
- A2/A3 source IDs differ or their actions become numerically identical;
- normalization includes validation/test data;
- existing `v2_pose`, pilot, stabilization, or checkpoint compatibility regresses;
- the final manifest cannot be built from a clean committed checkout;
- generated artifacts cannot be atomically preserved and hash-verified.

Do not reinterpret a stop condition as permission to weaken the contract, delete evidence, edit
generated results manually, or continue with partial data.

## Final Report

Report only decision-relevant evidence:

- branch and implementation commit(s);
- files/contracts added or changed;
- master dataset path, 550 total count, 110-per-pose counts, generation seed/overdraw plan, skipped
  candidates, and master fingerprint;
- each view's train/validation/test counts, per-pose counts, split fingerprint, and nesting proof;
- A2/A3 fingerprints and pairing proof;
- eight normalization paths and hashes;
- 16-cell mapping and rendered-script hash;
- focused/full test, Ruff, `git diff --check`, manifest verification, and shell-syntax results;
- final transfer-manifest path/hash, payload count/bytes, and rsync-file-list path/hash;
- remaining limitations or blockers.

End explicitly with:

```text
Full sweep transferred: NO
Full sweep submitted: NO
Phase 4 started: NO
```
