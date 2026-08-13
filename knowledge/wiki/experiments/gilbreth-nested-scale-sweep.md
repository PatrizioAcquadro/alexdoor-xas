# Experiment — Nested Scale Sweep

> Historical experiment record. Its cluster workflow is not maintained.

## Question

Can ACT/Diffusion, A2/A3, and N50/N100/N250/N500 be compared from one matched dataset with fixed holdouts?

## Method and Result

All sixteen seed-0 cells trained on the retained `v3_scale_master` views using Gilbreth A100 GPUs. Every checkpoint returned and loaded on the workstation, establishing a complete matrix for closed-loop evaluation.

Training loss alone did not identify a winner. The later [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] also found no monotonic data-scale advantage under its saturated protocol.

## Limits

The study used one simulated door family and one training seed. Cluster, transfer, Slurm, return, and sweep tooling were removed after completion; Git retains their history.

## Version Notes

- 2026-08-13 — Reduced the page to the historical question, result, and limits.
