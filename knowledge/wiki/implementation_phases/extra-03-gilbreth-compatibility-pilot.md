# Extra 03 — Gilbreth Compatibility Pilot

> Historical phase record. Gilbreth orchestration is not a maintained repository capability.

## Objective

Confirm that Isaac-free ACT and Diffusion training could run on Gilbreth A100 GPUs and return loadable checkpoints.

## Subphase E3.1 — Two-Cell Compatibility Result

#### Implementation

One ACT-A2 and one Diffusion-A3 N50 cell completed on Gilbreth, and their checkpoints loaded on the workstation. This qualified the training environment for the later scale sweep.

#### Key Decisions

- Isaac simulation remained workstation-only.
- Cluster training consumed the same model-facing dataset contract as local training.

#### Problems / Limitations

- Two cells established compatibility, not policy quality or scale behavior.
- Environment, transfer, Slurm, return, and preflight tooling were run-specific and were removed.

## Artifacts

Pilot packages are historical and are not present in the active repository. Git retains their implementation history.

## Files

No pilot-specific source or configuration file remains active.
