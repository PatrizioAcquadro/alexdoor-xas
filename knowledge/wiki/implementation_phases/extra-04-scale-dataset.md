# Extra 04 — Scale Dataset

## Objective

Produce one matched A2/A3 dataset supporting controlled N50-N500 comparisons.

## Focus

### Subphase E4.1 — Master and retained views

#### Implementation

Published `v3_scale_master` with 550 episodes, 110 per D0-D4 pose, plus retained
N50, N100, N250, and N500 train memberships and fixed 25-episode validation and
test sets. Each action-space/view pair has train-only normalization.

#### Key Decisions and Problems

- Content grouping prevents exact-trajectory leakage across splits.
- Current loaders validate split membership and recompute numerical statistics.
- Scale generation, pose-plan, merge, ledger, and publication tooling is retired.

#### Tests

Dataset, split, view, and normalization tests cover the retained products.

## Version Notes

- 2026-08-11 — Preserved dataset usability while removing its completed build workflow.
