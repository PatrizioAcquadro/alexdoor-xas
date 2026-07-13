# AlexDoor-XAS

AlexDoor-XAS is a research codebase for comparing action representations in
humanoid articulated-object manipulation. Its first benchmark is door pushing
with the fixed-base IHMC Alex V2 torso in NVIDIA Isaac Sim and Isaac Lab.

The main research variable is the action interface: joint deltas (A1),
end-effector deltas (A2), object-relative deltas (A3), and object-centric
chunks (A4). Policies execute through a shared adapter and produce explicit
safety and provenance evidence.

## Current state

Phases 1–3 are implemented through local post-Phase 3.3 stabilization:

- deterministic proxy and calibrated Alex V2 door-push environments;
- scripted generation and A1–A4 dataset export;
- fail-closed dataset, split, normalization, and provenance contracts;
- A2/A3/A4 adapter execution;
- state-only ACT and Diffusion Policy training and evaluation;
- a validated five-pose, 50-episode Alex V2 smoke dataset;
- tooling for a two-cell Gilbreth compatibility pilot.

The Gilbreth pilot has not run, the full cluster sweep has not started, and
Phase 4 VLA work has not started. See [project status](docs/status.md) for the
evidence and boundaries.

## Documentation

- [Project guidelines](docs/PROJECT_GUIDELINES.md) — research intent, phases,
  evaluation principles, and hardware boundary.
- [System architecture](docs/architecture.md) — implemented components and
  technical contracts.
- [Development guide](docs/development.md) — runtime, workflows, and exact
  verification commands.
- [Project status](docs/status.md) — completed work, evidence, limitations, and
  next steps.
- [Gilbreth pilot runbook](docs/cluster.md) — contract-bound transfer, execution,
  and return procedure.

## Quick start

Isaac Sim and Isaac Lab are supplied by the workstation runtime; they are not
installed as Python package dependencies. From the repository root:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Do not use bare system `python3` for Isaac code. The complete gate list and
data/training workflows are in the [development guide](docs/development.md).

## Repository layout

```text
src/alexdoor_xas/   package: assets, envs, actions, adapters, policies, data, eval
scripts/            verification, generation, training, evaluation, cluster tools
configs/            calibration, data, policy, tracking, and pilot contracts
tests/              pure-Python regression and contract tests
docs/               four canonical project docs plus the cluster runbook
datasets/           reusable generated datasets (ignored except README)
outputs/            per-run artifacts (ignored except README/curated evidence)
```

Machine-local assets are referenced in place. The sole robot lineage is
`~/Desktop/Alex/urdf/alex_v2.urdf`; generated door assets derive from the local
CombinedScene checkout. Generated datasets, checkpoints, videos, logs, and raw
simulator outputs stay out of Git.

## License

This repository is proprietary. No license grant is provided unless a separate
license file or written agreement states otherwise.
