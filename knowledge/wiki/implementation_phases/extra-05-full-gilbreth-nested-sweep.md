# Extra 05 — Full Gilbreth Nested Sweep

## Objective

Train the complete ACT/Diffusion x A2/A3 x N50/N100/N250/N500 matrix.

## Focus

### Subphase E5.1 — Sixteen-cell training result

#### Implementation

All 16 seed-0 cells completed on Gilbreth and returned checkpoints that loaded
on the workstation. The matrix became the input to the unified closed-loop
evaluation.

#### Key Decisions and Problems

- Completion established checkpoint availability, not comparative policy quality.
- Cluster, transfer, Slurm, return, and W&B publication orchestration was
  specific to the finished run and has been removed.

#### Tests

Current ACT/Diffusion checkpoint tests validate checkpoint v2 loading and reject
older checkpoint formats.

## Version Notes

- 2026-08-11 — Removed completed sweep infrastructure; scientific results remain
  in [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]].
