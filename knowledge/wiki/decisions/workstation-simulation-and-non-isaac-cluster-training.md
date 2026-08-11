# Decision — Workstation Simulation and Non-Isaac Cluster Training

## Context

Isaac Sim, Isaac Lab, external robot assets, and measured calibration are tied
to the configured Ubuntu workstation. Gilbreth provides useful A100 capacity
for training but is not the calibrated simulator authority and should not
require the Isaac stack.

## Decision

Keep asset verification, calibration, physical-master generation, and
closed-loop evaluation on the local Isaac workstation. Package only portable
Python/PyTorch data and model training for Gilbreth. Require Isaac modules to
be absent from the cluster environment and move inputs/returns through exact
inventories.

Select the explicit PyTorch/CUDA build from the live cluster driver. Treat
account, partition, QOS, and allocation as operator-supplied live state rather
than hard-coded project facts.

## Consequences

- Simulator semantics have one authority and are not duplicated on the
  training cluster.
- Core datasets, models, configs, and checkpoint loading must remain Isaac-free.
- Cluster completion proves training compatibility and artifact integrity, not
  closed-loop task behavior.
- Returned checkpoints must be verified locally before simulator evaluation.
- The workflow gains transfer/return complexity in exchange for portable and
  auditable compute use.

## Evidence

- `environment/gilbreth_pilot_py311.yml`
- `scripts/bootstrap_gilbreth_pilot.sh`
- `src/alexdoor_xas/cluster_pilot/preflight.py`
- `src/alexdoor_xas/cluster_sweep/slurm.py`
- `scripts/verify_returned_cluster_sweep.py`

See [[topics/system-architecture|System Architecture]] and
[[topics/provenance-and-artifact-lifecycle|Provenance and Artifact Lifecycle]],
plus [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03]]
and [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05]] for
the verified pilot and sweep workflows.

## Version Notes

- 2026-07-14 — The two-cell pilot qualified the non-Isaac cluster boundary.
- 2026-07-16 — The same boundary completed the sixteen-cell nested sweep.
