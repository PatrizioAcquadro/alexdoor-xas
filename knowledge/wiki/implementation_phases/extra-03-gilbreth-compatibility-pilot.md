# Extra 03 — Gilbreth Compatibility Pilot

## Objective

Confirm that non-Isaac ACT and Diffusion training could run on Gilbreth A100
GPUs and return loadable checkpoints.

## Focus

### Subphase E3.1 — Two-cell compatibility result

#### Implementation

One ACT-A2 and one Diffusion-A3 N50 cell completed and their checkpoints loaded
on the workstation. This qualified the environment for the later scale sweep.

#### Key Decisions and Problems

- Isaac simulation remained workstation-only.
- The pilot environment, transfer, Slurm, return, and preflight workflows were
  run-specific and are no longer part of the repository product.

#### Tests

Historical pilot verification is retained in Git history; current checkpoint
load tests cover the durable interoperability requirement.

## Version Notes

- 2026-08-11 — Retired the completed pilot implementation and retained its result.
