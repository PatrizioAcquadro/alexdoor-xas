# Extra 06 — Phase 3 Unified Evaluation

## Objective

Evaluate all sixteen Phase 3 checkpoints under one matched Alex V2 protocol.

## Focus

### Subphase E6.1 — Matched closed-loop matrix

#### Implementation

Ran 36 D0-D4 rollouts per cell, 576 total. Every rollout succeeded and none was adapter-rejected. Success was saturated, secondary metrics were heterogeneous, and one ACT-A3-N50 seed-112 force event remained `REVIEW_REQUIRED`.

#### Key Decisions and Problems

- The result does not select a policy, representation, or dataset size.
- Simulator force evidence is not hardware safety evidence.
- The executable matrix runner and intermediate artifacts are retired; canonical experiment pages preserve the conclusions and Git preserves the removed compact artifacts.

#### Tests

Current factual aggregation, frozen-protocol, and evaluation-routing primitives remain covered. The former compact evidence package is recoverable from Git history through commit `7f1fc8c` and is no longer part of the active output structure.

## Version Notes

- 2026-08-12 — Removed the curated-output package after retaining its conclusions in the canonical wiki and its files in Git history.
- 2026-08-11 — Reduced the phase to scientific findings and retained evidence.
