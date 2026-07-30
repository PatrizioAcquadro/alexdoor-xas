# Experiment — Gilbreth Nested Scale Sweep

## Purpose

Train a complete, controlled checkpoint matrix across policy family, action
representation, and nested data scale. The run's durable value is the exact,
verified set of sixteen training products later used by the unified workstation
evaluation.

## Matrix

| Dimension | Values |
|---|---|
| Policy | ACT, Diffusion |
| Action space | A2, A3 |
| Training view | N50, N100, N250, N500 |
| Seed | 0 |
| Total cells | 16 |

ACT trained for 100 epochs. Diffusion trained for 300 epochs with EMA. All cells
used the `v3_scale` physical-master family, fixed 25-episode validation and test
sets, and the view-specific training-only normalization described in
[[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

## Execution

- Location: Purdue Gilbreth, non-Isaac Python/PyTorch environment.
- Attempt: `11281591`.
- Source commit: `efa39434…`.
- Resources: one GPU per cell, at most two concurrent cells under the rendered
  contract.
- Inputs and outputs: exact SHA-256 inventories, attempt-specific paths, and
  symlink-free portable W&B publication.
- Configuration: `configs/cluster_sweep.v1.json`.

## Results

All 16/16 cells completed. The exact cluster return contained 736 payload files;
the workstation-side inventory contained 738 entries including enclosing
manifest material. All sixteen checkpoints passed hash and provenance
verification and loaded successfully on CPU.

This result establishes complete training, portable serialization, and an
auditable artifact return for the declared matrix. Training loss alone is not
a closed-loop outcome, and the sweep did not select a winner.

## Interpretation

The sweep qualified every declared policy/space/scale checkpoint for matched
workstation simulation. Its main comparative consequence is the availability
of a complete matrix with controlled data memberships, not a conclusion about
policy quality.

The later [[phase-3-unified-evaluation|Phase 3 Unified Evaluation]] found
success saturation across these checkpoints and no reliable monotonic
data-scale advantage under the tested protocol.

## Provenance

- Phase: [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]]
- Scale decision: [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]]
- Cluster boundary: [[decisions/workstation-simulation-and-non-isaac-cluster-training|Workstation Simulation and Non-Isaac Cluster Training]]
- Verification: `scripts/verify_returned_cluster_sweep.py`,
  `tests/test_cluster_sweep.py`, `tests/test_wandb_publication.py`

## Version Notes

- 2026-07-16 — Attempt `11281591` completed all sixteen cells and passed exact
  return/checkpoint verification.
