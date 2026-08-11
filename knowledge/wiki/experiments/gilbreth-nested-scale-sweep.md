# Experiment — Nested Scale Sweep

## Question

Train a complete ACT/Diffusion x A2/A3 x N50/N100/N250/N500 matrix so data
scale, policy family, and action representation could be compared under fixed
holdouts.

## Method

All 16 cells used training seed 0 and the retained `v3_scale_master` views.
ACT trained for 100 epochs; Diffusion trained for 300 epochs with EMA. Training
ran on Purdue Gilbreth A100 GPUs because it required PyTorch but not Isaac.

## Result

All 16 cells completed and all checkpoints loaded on the workstation. This
established a complete policy matrix for closed-loop evaluation. Training loss
alone did not identify a winner.

The later [[phase-3-unified-evaluation|Phase 3 Unified Evaluation]] found
success saturation and no monotonic N50-to-N500 advantage under the tested
protocol.

The cluster, transfer, Slurm, return, and sweep tooling used for this completed
run is no longer maintained. Git retains its implementation history.

## Version Notes

- 2026-08-11 — Reduced this page to the scientific design, result, and limits
  after removal of the completed orchestration.
