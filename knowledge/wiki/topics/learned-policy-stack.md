# Learned Policy Stack

AlexDoor-XAS maintains state-only ACT and Diffusion policies for A2 and A3.
Both use the same dataset, normalization, adapter, and rollout boundaries.

## Shared data and execution

`policies/common/data.py` loads one dataset version and either its normal split
or a retained view. It validates stored train-only normalization by direct
recomputation. Online observations use the same named preset as training.

Policies never bypass adapters. A2 is validated before execution; A3 is first
transformed through the door frame and then follows the A2 safety path.

## ACT

`ACTModel` is a conditional variational transformer over action chunks. It
optimizes masked L1 reconstruction plus KL regularization and uses a zero
latent for deterministic inference. The default active configuration is
`configs/act.yaml`.

## Diffusion

`DiffusionTransformer` predicts epsilon over an action horizon with a causal
transformer. Training uses DDPM noise and optional EMA weights; inference
supports DDPM and DDIM. Observations use z-score normalization and actions use
train-range min-max normalization. The default active configuration is
`configs/diffusion.yaml`.

## Checkpoint contract

New ACT and Diffusion checkpoints use v2 formats. Each stores model weights,
dimensions, model configuration, an essential dataset descriptor,
normalization statistics, required robot identity, and optional run metadata.
Save and load validate dimensions, finite weights/statistics, action space,
observation preset, and robot compatibility.

Legacy Phase 3 v1 checkpoints remain loadable. Their full split lists, source
commit, configuration hashes, dataset/view fingerprints, and provenance blocks
are ignored; files are not rewritten or migrated. Closed-loop evaluation uses
checkpoint-owned metadata and does not consult the former training dataset.

## Limits

- Only A2 and A3 have learned policies.
- Models are state-only and benchmark-specific, not VLA systems.
- Training loss is not a closed-loop success or safety result.
- The saturated Phase 3 result does not establish equivalence or a winner.

## Version Notes

- 2026-08-11 — Introduced compact checkpoint v2 and dataset-independent
  evaluation while retaining read compatibility with Phase 3 v1 files.
