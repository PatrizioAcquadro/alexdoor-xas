# Experiment — Phase 3 Unified Evaluation

## Question

Do policy family, action representation, or retained training size produce a
clear winner under one matched Alex V2 closed-loop protocol?

## Method

Sixteen ACT/Diffusion x A2/A3 x N50/N100/N250/N500 checkpoints ran 36 rollouts
each across D0-D4, for 576 total. Success was the first 45-degree hinge crossing
within 600 control ticks. ACT used horizon 40; Diffusion used DDIM-10, Tp16,
Ta8. Adapter decisions and panel-filtered force were recorded.

## Result

Every cell achieved 36/36 successes. Across 57,678 adapter decisions, 54,183
were accepted, 3,495 corrected, and zero rejected. Success therefore did not
select A2 versus A3, ACT versus Diffusion, or dataset size. Secondary timing,
force, and correction directions were heterogeneous.

One ACT-A3-N50 D0 randomized rollout at seed 112 peaked at 219.95 N for one
tick and remains `REVIEW_REQUIRED`. See
[[act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]].

The retained scientific evidence is:

- `outputs/curated/phase3_unified_evaluation/report.md`
- `outputs/curated/phase3_unified_evaluation/aggregate_summary.json`

The matrix runner and its resolved-plan/inventory/rollout-table artifacts are
no longer maintained. Git retains the former implementation.

## Limits

The result covers one simulated door family and seed-0-trained state-only
policies. It is not hardware, broader generalization, or VLA evidence.

## Version Notes

- 2026-08-11 — Retained the scientific outcome and compact aggregate while
  retiring the executable unified-evaluation workflow.
