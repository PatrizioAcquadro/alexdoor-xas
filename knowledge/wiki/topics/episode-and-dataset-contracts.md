# Episode and Dataset Contracts

The data system separates factual physical execution records from the action representations consumed by models.

## Episode schema

New episodes use `phase2.v2`. `EpisodeOutcome` stores `success`, `final_door_angle`, `n_steps`, `termination_reason`, `environment_terminated`, `environment_truncated`, and notes. Stop reasons are factual: `controller_done`, `controller_timeout`, `tick_budget`, `environment_terminated`, `environment_truncated`, or `step_error`. The schema does not generate, validate, aggregate, or expose interpreted failure labels.

Readers continue to accept `phase2.v0` and `phase2.v1`, including A4 JSON Lines records. They silently discard the obsolete `failure_label`, expose `termination_reason: not_recorded`, and use unknown environment flags. Existing datasets are not rewritten, so historical scene paths and legacy outcome fields remain provenance rather than current output contracts.

`src/alexdoor_xas/recording/episode.py` records observation and contact state before each action, the requested/applied command for that state, and the terminal response to the final action. A1/A2/A3 use HDF5 plus JSON sidecars; A4 uses JSON Lines.

## Matched exports

`src/alexdoor_xas/data_engine/export.py` derives matched products from one physical episode. A2 keeps world-frame end-effector deltas, A3 expresses the equivalent command in the static door frame, A4 stores object-centric guarded chunks, and Alex episodes derive A1 joint deltas from recorded targets. New A1 exports require the final applied target; legacy v0/v1 support remains read-only. The active operational version is `v2_pose` under `door_push_alex_v2`.

`scripts/verify_dataset_interface.py` requires all A1-A4 products, validates the existing split and normalization artifacts without writing, and checks the paired A2/A3 conversion. `--write-artifacts` is the explicit regeneration mode.

## Model-facing access

`EpisodeDataset` and `A4ChunkDataset` are the supported readers. A4 remains structured and has no unused numeric feature encoding. `ChunkSampler` pairs observation at time t with the following action horizon; batches contain only `obs`, `actions`, and `is_pad`. Supported observations are `core`, `core_contact`, and `core_door_pose`.

## Splits, views, and normalization

`dataset/splits.py` hashes trajectory content only to keep equivalent episodes inside one split and prevent exact-content leakage. A retained view is resolved from `datasets/<task>/splits/<view_id>.json`; its normalization comes from the selected action-space version's `views/<view_id>/norm_stats.json`. Loaders verify schema, action space, dimensions, finite values, membership, and direct recomputation over declared train IDs.

The retained `v3_scale_master` contains 550 episodes across D0-D4. Its N50, N100, N250, and N500 split files remain usable with fixed validation and test memberships; the generation workflow is historical.

## Version Notes

- 2026-08-12 — Removed unused dataset APIs and A4 numeric encoding, narrowed batch and observation contracts, and made the verifier read committed artifacts by default.
- 2026-08-12 — Removed retired candidate-generation and paired-publication hooks; current A1 writes require the recorded final target.
- 2026-08-12 — Introduced `phase2.v2` factual outcomes and legacy v0/v1 reads without failure labels; existing datasets remain unchanged.
- 2026-08-12 — Made `door_push_alex_v2/v2_pose` the verifier default and folded the A2/A3 posed-door distinction check into the dataset interface gate.
- 2026-08-11 — Dataset loading moved from publication/fingerprint orchestration to direct split membership and numerical validation.
