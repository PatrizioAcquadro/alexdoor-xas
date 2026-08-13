# Phase 2 — Scripted Baseline and Data Engine

> Historical phase record. Current contracts are documented in [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

## Objective

Create a deterministic door-push generator and matched A1-A4 episode exports.

## Subphase 2.1 — Scripted Execution and Recording

#### Implementation

This phase established the approach/contact/push/release controller, pre-action recording alignment, terminal-response capture, and matched representation export from one physical episode.

The maintained successor writes compact `phase2.v2` A1-A3 HDF5 episodes and A4 JSONL chunks under the `door_push_alex_v2/v2_pose` contract. It retains read-only compatibility with existing v1 records.

#### Key Decisions

- Door-relative frames are explicit and hinge anchored.
- Representation products share physical identity and factual outcome.
- Reusable exports remain in `datasets/`; scripted staging lives in the runtime cache.

#### Problems / Limitations

- A1 is not learned and A4 has no learned policy.
- Scale-candidate, paired-master publication, and legacy sidecar workflows were removed.

## Artifacts

Reusable `v2_pose` datasets and their split/normalization artifacts remain valid. Historical generation workspaces and evidence bundles are not active repository artifacts.

## Files

- `src/alexdoor_xas/policies/scripted/`
- `src/alexdoor_xas/recording/`
- `src/alexdoor_xas/data_engine/`
