# Extra 05 — Full Gilbreth Nested Sweep

> Historical phase record. The result is summarized in [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]].

## Objective

Train the complete ACT/Diffusion x A2/A3 x N50/N100/N250/N500 matrix.

## Subphase E5.1 — Sixteen-Cell Training Result

#### Implementation

All sixteen seed-0 cells completed on Gilbreth and returned checkpoints that loaded on the workstation. The matrix became the input to the matched closed-loop evaluation.

#### Key Decisions

- Completion established checkpoint availability, not comparative policy quality.
- Training remained Isaac-free; simulator evaluation remained local.

#### Problems / Limitations

- Training loss alone did not identify a winner.
- Cluster, transfer, Slurm, return, and sweep orchestration were specific to the completed run and were removed.

## Artifacts

The historical checkpoints and run packages are not part of the active repository tree. Their code history remains in Git.

## Files

No sweep-specific source or configuration file remains active.
