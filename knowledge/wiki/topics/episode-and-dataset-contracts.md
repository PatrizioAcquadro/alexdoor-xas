# Episode and Dataset Contracts

The data system separates physical execution records from the action
representations consumed by models.

## Episodes and exports

`src/alexdoor_xas/recording/episode.py` records observation and contact state
before each action, the requested/applied command for that state, and the
terminal response to the final action. A1/A2/A3 use HDF5 plus JSON sidecars;
A4 uses JSON Lines.

`src/alexdoor_xas/data_engine/export.py` derives matched products from one
physical episode. A2 keeps world-frame end-effector deltas, A3 expresses the
equivalent command in the static door frame, A4 stores object-centric guarded
chunks, and Alex episodes can derive A1 joint deltas.

## Model-facing access

`EpisodeDataset` and `A4ChunkDataset` are the supported readers.
`ChunkSampler` pairs observation at time t with the following action horizon
and padding mask. Model code does not interpret raw storage keys directly.

## Splits, views, and normalization

`dataset/splits.py` hashes trajectory content only to keep equivalent episodes
inside one split and prevent exact-content leakage. Train, validation, and test
membership must be disjoint.

A retained view is resolved directly from
`datasets/<task>/splits/<view_id>.json`; its normalization comes from the
selected action-space version's `views/<view_id>/norm_stats.json`. Loaders
verify schema, action space, dimensions, finite values, membership, and direct
recomputation of mean, standard deviation, minimum, maximum, and count over the
declared train IDs.

Legacy local datasets may contain additional fingerprints or administrative
metadata. These fields are tolerated and ignored. Newly written normalization
artifacts contain only active fields.

The retained `v3_scale_master` contains 550 episodes across D0-D4. Its N50,
N100, N250, and N500 split files remain usable with fixed validation and test
memberships; the workflow that originally generated them is historical.

## Version Notes

- 2026-08-11 — Dataset loading moved from publication/fingerprint orchestration
  to direct split membership and numerical validation.
