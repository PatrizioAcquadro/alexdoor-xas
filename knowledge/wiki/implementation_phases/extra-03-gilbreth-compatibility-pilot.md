# Extra 03 — Gilbreth Compatibility Pilot

## Objective

Prove that the state-only training stack can run on Purdue Gilbreth without
Isaac dependencies, while preserving exact source/data provenance, isolated
Slurm attempts, and verifiable returned artifacts.

## Focus

### Subphase Extra 03.1 — Portable Training Environment

#### Implementation

`environment/gilbreth_pilot_py311.yml` defines the portable Python 3.11
environment. The live NVIDIA driver determines the explicit compatible
PyTorch/CUDA build at bootstrap time; Isaac Sim and Isaac Lab must remain
absent. `scripts/bootstrap_gilbreth_pilot.sh` creates the environment and
captures installed versions for provenance.

`src/alexdoor_xas/cluster_pilot/config.py` parses
`configs/cluster_pilot_n50.v1.json`, which selects one ACT-A2 and one
Diffusion-A2 N50 cell. Preflight checks source, dataset, split, normalization,
configuration, Python, and dependency assumptions before Slurm rendering.

#### Key Decisions and Problems

- Gilbreth is a training target only. Simulator calibration, data generation,
  and closed-loop evaluation remain authoritative on the workstation.
- The environment excludes Isaac modules to keep the cluster package portable
  and to expose accidental simulator coupling.
- Account, partition, QOS, and live allocation are operator inputs because
  they are cluster state, not stable repository facts.

#### Tests

- `tests/test_cluster_pilot.py` verifies configuration and Slurm contract
  rendering.
- `scripts/preflight_cluster_pilot.py` rejects missing data, incompatible
  fingerprints, unsupported Python, and accidental Isaac availability.

### Subphase Extra 03.2 — Exact Transfer and Attempt Isolation

#### Implementation

`src/alexdoor_xas/cluster_pilot/transfer.py` and
`scripts/build_cluster_pilot_manifest.py` build a transfer package with an
exact SHA-256 inventory. The rendered Slurm job resolves a specific source
commit and creates attempt-specific directories so retries cannot mix logs,
checkpoints, or metadata.

`src/alexdoor_xas/cluster_pilot/returns.py` and
`scripts/build_cluster_pilot_return_manifest.py` produce a symlink-free return
inventory. `scripts/verify_returned_cluster_pilot.py` verifies checksums,
expected cells, attempt identity, checkpoint portability, and provenance after
return to the workstation. This workflow is a concrete application of
[[topics/provenance-and-artifact-lifecycle|Provenance and Artifact Lifecycle]].

#### Key Decisions and Problems

- Transfer and return inventories are exact; extra or missing files fail
  verification.
- Every attempt owns its output path. A retry is new evidence, not an in-place
  repair of an earlier attempt.
- Historical W&B run directories included symlinks that required manual
  cleanup for this pilot. Automated symlink-free publication was completed and
  exercised by the later full sweep, not retroactively by the pilot.

#### Tests

- `tests/test_cluster_transfer.py`, `tests/test_wandb_publication.py`, and
  `tests/test_wandb_tracking.py` cover inventories, portable publication, and
  tracking metadata.
- Returned checkpoints were loaded on CPU after hash verification, proving
  serialization portability independently of cluster GPU availability.

### Subphase Extra 03.3 — Two-Cell Compatibility Run

#### Implementation

Gilbreth attempt `11279452` trained the two declared N50 cells from exact source
commit `10ba63e…` on an A100 80 GB GPU. The recorded environment was Python
3.11.15, NumPy 2.4.6, PyTorch 2.12.1+cu126, and CUDA 12.6. The returned package
contained 52 payload files; both checkpoints passed workstation CPU loading and
provenance verification.

This was a compatibility pilot, not a durable comparative experiment. Its
purpose was to qualify the transport, environment, scheduler, training, and
return path before the full matrix in [[extra-05-full-gilbreth-nested-sweep|Extra 05]].

#### Key Decisions and Problems

- Training completion and checkpoint loadability are the pilot acceptance
  criteria. Closed-loop task performance is outside this phase.
- Only A2 cells were needed to prove compatibility; expanding to all action
  spaces before the workflow was qualified would not add useful evidence.

#### Tests

- Both declared Slurm cells completed and the 52-file return inventory matched
  exactly.
- Both checkpoints loaded successfully on CPU with the expected model,
  dataset, normalization, split, source, and robot bindings.

## Version Notes

- 2026-07-12 to 2026-07-14 — Portable environment, exact packaging, isolated
  Slurm execution, and verified return workflow landed and passed the pilot.
- 2026-07-16 — The later full sweep exercised automatic portable W&B
  publication, replacing the pilot's manual symlink cleanup as the current
  operational path.
