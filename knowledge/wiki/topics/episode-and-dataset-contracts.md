# Episode and Dataset Contracts

The data system separates a physical execution record from the representation-
specific datasets consumed by models. It uses versioned schemas, shared
physical identities, content-aware splits, and training-only normalization.

## Episode Timing and Schema

`src/alexdoor_xas/recording/episode.py` defines `EpisodeMeta`, `EpisodeStep`,
`EpisodeOutcome`, and `EpisodeBuffer`. Each step records:

- the observation and force/contact state before action;
- the requested and applied action information for that state;
- task and provenance metadata needed to interpret the tick.

`terminal_contact` stores the response to the final executed action. Treating a
step's contact as post-action would violate the schema's temporal alignment.

`src/alexdoor_xas/recording/writer.py` implements container version
`phase2.v1`. A1/A2/A3 use one HDF5 trajectory and one JSON sidecar per episode;
A4 uses JSON Lines. The sidecar carries schema, representation, outcome, and
provenance that should not be inferred from HDF5 path names.

The ordinary episode writer currently creates the HDF5 and JSON products
sequentially, not as one transaction. `run_episode` also appends a pending step
before confirming a successful `env.step`. Both are known current reliability
gaps and planned quality work, not completed guarantees.

## Model-Facing Access

`src/alexdoor_xas/dataset/loader.py::EpisodeDataset` is the A1/A2/A3 model
interface; `A4ChunkDataset` serves A4 JSONL. Callers receive validated records
and named observation presets instead of reading raw HDF5 keys.

Current `EpisodeDataset` construction loads its validated records eagerly, and
`by_id` is a linear lookup. An older closeout's “lazy loading” wording does not
describe current behavior.

`src/alexdoor_xas/dataset/sampling.py::ChunkSampler` samples observation at
time `t`, the following action horizon, and a padding mask. ACT and Diffusion
adapt this shared contract rather than defining independent raw-data readers.

## Splits and Normalization

`src/alexdoor_xas/dataset/splits.py` hashes physical trajectory content, groups
equivalent episodes, and assigns whole groups to pose-stratified
train/validation/test splits. This prevents exact-content leakage even when
episode IDs differ.

`src/alexdoor_xas/dataset/normalize.py` computes statistics only from training
IDs. Each normalization artifact binds the dataset, action representation,
observation preset, split, and training membership. Checkpoints carry these
identities and fail closed on mismatch.

## Matched Exports

`src/alexdoor_xas/data_engine/export.py` derives representation-specific
products from one physical episode:

- A2 keeps the world-frame end-effector command.
- A3 stores the equivalent static door-frame command.
- A4 stores object-centric guarded intent chunks.
- Alex episodes can derive A1 joint-target deltas.

The products share physical identity and outcome but have different action
arrays and representation fingerprints. See
[[action-representations-and-adapters|Action Representations and Adapters]].

## Scale Dataset and Views

The `v3_scale` physical master contains 550 successful episodes across five
door poses. A2 and A3 are published as one owned pair.
`src/alexdoor_xas/dataset/views.py` defines N50, N100, N250, and N500 nested
training subsets with fixed 25-episode validation and test sets. Each view has
exact ordered IDs, a selection hash, and its own training-only normalization
for each representation.

Changing generation requires a new dataset version and regenerated splits and
normalization. A known fingerprint must not be refreshed merely to absorb
unexplained content drift.

## Primary References

- `src/alexdoor_xas/recording/episode.py`
- `src/alexdoor_xas/recording/writer.py`
- `src/alexdoor_xas/data_engine/generate.py`
- `src/alexdoor_xas/data_engine/export.py`
- `src/alexdoor_xas/dataset/loader.py`
- `src/alexdoor_xas/dataset/splits.py`
- `src/alexdoor_xas/dataset/normalize.py`
- `src/alexdoor_xas/dataset/views.py`
- `tests/test_dataset_interface.py`
- `tests/test_dataset_views.py`

## Version Notes

- 2026-07-03 — `phase2.v1`, pre-action timing, matched export, and outcome
  contracts landed.
- 2026-07-15 — The paired 550-episode master and fingerprinted nested views
  extended the same contract to scale comparisons.
