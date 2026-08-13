# Experiment — Phase 3 Unified Evaluation

> Historical experiment record. The matrix runner and result package are not maintained.

## Question

Did policy family, action representation, or retained training size produce a clear winner under one matched Alex V2 closed-loop protocol?

## Method and Result

Sixteen ACT/Diffusion x A2/A3 x N50/N100/N250/N500 checkpoints ran 36 D0-D4 rollouts each, for 576 total. Every cell achieved 36/36 success. Across 57,678 adapter decisions, 54,183 were accepted, 3,495 corrected, and none rejected.

Success therefore selected neither A2 versus A3, ACT versus Diffusion, nor dataset size. Secondary timing, force, and correction metrics were heterogeneous.

One ACT-A3-N50 D0 randomized rollout at seed 112 peaked at 219.95 N for one tick and remains `REVIEW_REQUIRED`. See [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]].

## Limits

The result covers one simulated door family, state-only policies, and seed-0 training. It is not hardware, generalization, sim-to-real, or VLA evidence.

The former report and aggregate remain recoverable from Git history through commit `7f1fc8c`.

## Version Notes

- 2026-08-13 — Kept only the closed-loop design, scientific conclusion, force-review exception, and limits.
