# Phase 3 — Non-VLA Learned Baselines

## Objective

Train and evaluate state-only ACT and Diffusion policies through shared dataset
and adapter contracts.

## Focus

### Subphase 3.1 — Data, adapters, and policies

#### Implementation

Added validated dataset readers, content-grouped splits, train-only
normalization, chunk sampling, A2/A3/A4 adapters, ACT, Diffusion, and
closed-loop evaluation. New checkpoints use compact v2 contracts; Phase 3 v1
checkpoints remain loadable without their former administrative fields.

#### Key Decisions and Problems

- Learned policies cover A2 and A3 only.
- Evaluation uses checkpoint-owned model and normalization state, not the live
  training dataset.
- Training metrics do not substitute for closed-loop results.

#### Tests

Deterministic tests cover data validation, normalization recomputation,
checkpoint v1/v2 loading, model behavior, CPU overfit, adapters, and rollout
semantics.

## Version Notes

- 2026-08-11 — Removed Phase 3 provenance orchestration and introduced compact
  checkpoint v2 while preserving legacy loading.
