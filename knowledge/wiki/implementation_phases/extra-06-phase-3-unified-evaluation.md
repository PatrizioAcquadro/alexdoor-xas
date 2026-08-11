# Extra 06 — Phase 3 Unified Evaluation

## Objective

Evaluate all sixteen Phase 3 checkpoints under one matched Alex V2 protocol.

## Focus

### Subphase E6.1 — Matched closed-loop matrix

#### Implementation

Ran 36 D0-D4 rollouts per cell, 576 total. Every rollout succeeded and none was
adapter-rejected. Success was saturated, secondary metrics were heterogeneous,
and one ACT-A3-N50 seed-112 force event remained `REVIEW_REQUIRED`.

#### Key Decisions and Problems

- The result does not select a policy, representation, or dataset size.
- Simulator force evidence is not hardware safety evidence.
- Only the compact report and aggregate summary remain curated; the executable
  matrix runner and intermediate artifacts are retired.

#### Tests

Current evaluation primitives remain covered. The completed aggregate is
preserved in `outputs/curated/phase3_unified_evaluation/`.

## Version Notes

- 2026-08-11 — Reduced the phase to scientific findings and retained evidence.
