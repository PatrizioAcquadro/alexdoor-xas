# Episode and Dataset Contracts

The maintained data path records one physical Alex V2 door episode and exports matched A1-A4 representations under the `door_push_alex_v2/v2_pose` identity.

## Recording Schema

New A1-A3 HDF5 episodes use `phase2.v2`. Each step contains the non-duplicated pre-action state, requested and applied action, adapter decision, controller phase, and contact information. The terminal response is stored separately so the final action is not left without an outcome.

`EpisodeOutcome` records success, final door angle, step count, notes, factual termination reason, environment termination/truncation flags, and controller completion/timeout state. The writer does not invent failure labels from those facts.

`phase2.v1` HDF5 files remain readable and are upgraded in memory where necessary. The repository never rewrites them in place and no longer emits v1.

## Matched Exports

One recorded physical episode can produce:

- A1, A2, and A3 as one HDF5 file per episode;
- A4 as structured chunks in `episodes.jsonl`.

All representations share the physical episode ID and outcome. Representation-specific actions and metadata remain separate. `scripts/verify_dataset_interface.py` checks the four exports and the numerical distinction between A2 and A3.

## Dataset Layout

The active path is `datasets/<task>/<action_space>/<version>/`. For this benchmark, task is `door_push_alex_v2` and version is `v2_pose`.

`datasets/door_push_alex_v2/splits/v2_pose.json` assigns shared episode IDs to disjoint train, validation, and test sets. Retained view files select nested training subsets without changing validation or test membership. Normalization is computed from the selected training IDs and validated by direct recomputation.

The repository loads existing splits, views, and normalization. Completed scale-dataset construction, pose-plan, merge, ledger, and publication workflows are not maintained.

## Model-Facing Data

`EpisodeDataset`, `A4ChunkDataset`, and `ChunkSampler` expose validated records. Learned policies support `core`, `core_contact`, and `core_door_pose` observation presets. Training batches contain only `obs`, `actions`, and `is_pad`.

ACT and Diffusion train on A2 or A3. A1 remains export-only and A4 remains non-learned.

## Version Notes

- 2026-08-13 — Documented `phase2.v2`, read-only v1 compatibility, matched `v2_pose` exports, and the minimal model-facing dataset path.
