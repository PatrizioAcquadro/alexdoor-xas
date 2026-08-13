# Phase 3 — Non-VLA Learned Baselines

## Objective

Train and evaluate state-only ACT and Diffusion policies through shared dataset, adapter, and artifact contracts.

## Focus

### Subphase 3.1 — Data, adapters, policies, and canonical runs

#### Implementation

Added validated dataset readers, content-grouped splits, train-only normalization, chunk sampling, A2/A3/A4 adapters, ACT, Diffusion, and closed-loop evaluation. Inference uses compact self-contained checkpoint v2 contracts and requires the checkpoint robot identity to exactly match the active Alex V2 runtime.

Training now allocates collision-safe UTC runs under `outputs/door_push_alex_v2/{act,diffusion}/`, freezes the complete evaluation protocol in immutable `resolved_config.json`, writes consolidated atomic resume state before epoch 0 and after each epoch, and publishes compact training/open-loop artifacts plus one narrative report. Successful completion removes the resume checkpoint; errors retain it and create `error.log`.

Closed-loop evaluation runs the frozen 36-rollout D0-D4 protocol with a fresh environment per pose and publishes factual success, termination, time, force, adapter, and warning-family results. Exact protocol matches may complete the source training run; changed protocols create checkpoint-free sibling evaluation runs. Optional traces and media are selective and empty directories are never created.

#### Key Decisions and Problems

- Learned policies cover A2 and A3 only.
- Evaluation uses the source run's frozen configuration and self-contained checkpoint, not the live training dataset.
- Training and open-loop metrics do not substitute for closed-loop results.
- Completed runs and completed closed-loop results are never overwritten.

#### Tests

Deterministic tests cover data validation, normalization, checkpoint v2 loading and rejection of older formats, ACT/Diffusion behavior, uninterrupted-versus-resumed equivalence including RNG/EMA, exclusive run allocation, output schemas, lifecycle rules, 36-rollout aggregation, protocol routing, selective artifacts, adapters, and rollout semantics.

## Version Notes

- 2026-08-12 — Reduced the dataset layer to current A1-A4 validation and the three observation presets supported by ACT/Diffusion rollout.
- 2026-08-12 — Made checkpoint loading Alex V2-only: older formats, unfingerprinted checkpoints, and cross-model transfer are rejected.
- 2026-08-12 — Added canonical learned runs, complete resume state, compact output schemas, frozen multi-pose evaluation, and protocol-aware evaluation-only siblings.
- 2026-08-11 — Removed Phase 3 provenance orchestration and introduced compact checkpoint v2.
