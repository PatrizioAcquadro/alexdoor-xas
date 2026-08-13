# Extra 06 — Phase 3 Unified Evaluation

> Historical phase record. The scientific result is summarized in [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]].

## Objective

Evaluate all sixteen Phase 3 checkpoints under one matched Alex V2 protocol.

## Subphase E6.1 — Matched Closed-Loop Matrix

#### Implementation

Each ACT/Diffusion x A2/A3 x N50/N100/N250/N500 cell ran 36 D0-D4 rollouts, for 576 total. Every rollout succeeded and none was adapter-rejected.

#### Key Decisions

- Saturated success cannot select a policy, representation, or data size.
- Simulator force evidence cannot be interpreted as hardware-safety evidence.

#### Problems / Limitations

- Secondary metrics were heterogeneous.
- One ACT-A3-N50 seed-112 force event remains `REVIEW_REQUIRED`.
- The matrix runner and intermediate packages were removed after closeout.

## Artifacts

Canonical experiment pages retain the conclusions. Detailed removed files remain recoverable from Git history through commit `7f1fc8c`.

## Files

No unified-matrix runner or result package remains in the active tree.
