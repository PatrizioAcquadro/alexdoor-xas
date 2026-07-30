# Extra 05 — Full Gilbreth Nested Sweep

## Objective

Train the complete Phase 3 matrix across two policy families, two action
representations, and four nested dataset sizes on Gilbreth, then return every
checkpoint and training artifact through an exact, attempt-scoped package.

## Focus

### Subphase Extra 05.1 — Sixteen-Cell Sweep Contract

#### Implementation

`configs/cluster_sweep.v1.json` declares 16 cells: ACT and Diffusion over A2 and
A3 at N50, N100, N250, and N500, all with seed 0. ACT trains for 100 epochs;
Diffusion trains for 300 epochs with EMA. Every cell resolves its dataset view,
split, normalization, model configuration, and expected provenance through
`src/alexdoor_xas/cluster_sweep/config.py`.

The cell resolver is the single source for preflight, Slurm rendering,
training-command construction, and return verification. This avoids independent
interpretations of a cell name in different scripts.

#### Key Decisions and Problems

- The matrix changes one of policy family, representation, or nested training
  size while retaining fixed holdouts and the common physical master.
- One GPU is requested per cell, with no more than two concurrent cells in the
  supported schedule. Live account, partition, QOS, and allocation remain
  operator-provided.
- The sweep is training evidence. Checkpoint completion does not imply
  closed-loop success.

#### Tests

- `tests/test_cluster_sweep.py` verifies exact cell expansion, training
  commands, view/normalization binding, scheduler resources, and invalid
  configurations.
- `scripts/preflight_cluster_sweep.py` validates source, all scale views,
  normalization artifacts, environment, and expected cell identities before
  rendering jobs.

### Subphase Extra 05.2 — Transfer, Slurm, and Portable Publication

#### Implementation

`src/alexdoor_xas/cluster_sweep/transfer.py` and
`scripts/build_cluster_sweep_manifest.py` create an exact source/data package.
`cluster_sweep/slurm.py` renders attempt-isolated jobs using an absolute
environment Python path. Each cell writes checkpoints, metrics, logs, runtime
metadata, and a portable W&B export to its own attempt path.

`src/alexdoor_xas/cluster_pilot/wandb_publication.py` materializes W&B
directories without symlinks. This full sweep is the first live evidence that
the automated publisher works across the complete matrix; earlier pilot
manual cleanup is historical.

`cluster_sweep/returns.py` and
`scripts/build_cluster_sweep_return_manifest.py` build the exact return
inventory. `scripts/verify_returned_cluster_sweep.py` checks hashes, attempt
identity, all expected cells, model/config bindings, and CPU checkpoint loads.

#### Key Decisions and Problems

- Source and data are content-addressed at transfer time; the cluster does not
  fetch a moving branch as its authoritative input.
- Attempts never share output directories, and returned files from separate
  attempts must not be mixed.
- W&B is supplementary tracking. Exact repository-owned metadata and
  inventories remain the reproducibility authority.

#### Tests

- Transfer, publication, and return failure paths are covered by
  `tests/test_cluster_transfer.py`, `tests/test_wandb_publication.py`, and
  `tests/test_cluster_sweep.py`.
- Post-return verification loaded all sixteen checkpoints on CPU and reconciled
  the complete inventory.

### Subphase Extra 05.3 — Completed Sweep

#### Implementation

Gilbreth attempt `11281591` trained all 16 declared cells from exact source
commit `efa39434…`. The cluster return contained 736 payload files; the
workstation-side exact inventory contained 738 entries including its enclosing
manifest material. All sixteen checkpoints loaded successfully on CPU.

The durable run interpretation is maintained in
[[experiments/gilbreth-nested-scale-sweep|Gilbreth Nested Scale Sweep]].
Training metrics and checkpoint integrity qualified the models for the
workstation evaluation in [[extra-06-phase-3-unified-evaluation|Extra 06]];
they did not select a winning policy.

#### Key Decisions and Problems

- A completed cell means the declared training and artifact contract passed.
  Behavioral comparison is deferred to matched closed-loop simulation.
- The full sweep supersedes stale README and cluster-runbook statements that
  describe the sweep as unsubmitted or the portable publisher as unexercised.

#### Tests

- The authoritative attempt completed 16/16 cells and returned the expected
  736 payload files.
- All sixteen checkpoints passed exact-inventory checks, provenance
  reconciliation, and workstation CPU loading.

## Version Notes

- 2026-07-16 — The sixteen-cell Gilbreth sweep completed and its exact return
  package passed local verification.
- 2026-07-16 — Automatic symlink-free W&B publication became live-validated
  across the full matrix.
