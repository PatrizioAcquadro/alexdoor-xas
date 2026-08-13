# Phase 3 — Non-VLA Learned Baselines

## Objective

Train and evaluate state-only ACT and Diffusion policies through shared dataset, adapter, and artifact contracts.

## Focus

### Subphase 3.1 — Data, adapters, policies, and canonical runs

#### Implementation

Added validated dataset readers, content-grouped splits, train-only normalization, chunk sampling, A2/A3/A4 adapters, ACT, Diffusion, and closed-loop evaluation. ACT and Diffusion share configuration types, OmegaConf loading, sinusoidal tables, and checkpoint serialization; each policy reconstructs its own model from the unchanged v2 payload. Inference requires the checkpoint robot identity to exactly match the active Alex V2 runtime.

Training now allocates collision-safe UTC runs under `outputs/door_push_alex_v2/{act,diffusion}/`, freezes the complete evaluation protocol in immutable `resolved_config.json`, writes consolidated atomic resume state before epoch 0 and after each epoch, and publishes compact training/open-loop artifacts plus one narrative report. Successful completion removes the resume checkpoint; errors retain it and create `error.log`.

Closed-loop evaluation runs the frozen 36-rollout D0-D4 protocol with a fresh environment per pose and publishes factual success, termination, time, force, adapter, and warning-family results. Every invocation creates an exclusive `<training-run>/closed_loop/<UTC-id>[_rN]/` child containing resolved config, metrics, one summary, and one report without modifying the training run. Traces are generated only for failures, force-limit exceedances, or explicit rollout keys.

#### Key Decisions and Problems

- Learned policies cover A2 and A3 only.
- Evaluation uses the source run's frozen configuration and self-contained checkpoint, not the live training dataset.
- Training and open-loop metrics do not substitute for closed-loop results.
- Completed runs and completed closed-loop results are never overwritten.

#### Tests

Deterministic tests cover data validation, normalization, checkpoint v2 and robot identity, ACT/Diffusion forward and loss behavior, Diffusion causality, uninterrupted-versus-resumed equivalence including RNG/EMA, immutable evaluation allocation, factual aggregation, selective traces, adapters, scripted FSM safety, and rollout semantics.

## Version Notes

- 2026-08-12 — Consolidated policy configuration and runtime internals, removed implementation-only tests, and made every evaluation an immutable child of its training run.
- 2026-08-12 — Reduced the dataset layer to current A1-A4 validation and the three observation presets supported by ACT/Diffusion rollout.
- 2026-08-12 — Made checkpoint loading Alex V2-only: older formats, unfingerprinted checkpoints, and cross-model transfer are rejected.
- 2026-08-12 — Added canonical learned runs, complete resume state, compact output schemas, frozen multi-pose evaluation, and protocol-aware evaluation-only siblings.
- 2026-08-11 — Removed Phase 3 provenance orchestration and introduced compact checkpoint v2.
