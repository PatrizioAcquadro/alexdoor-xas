# AlexDoor-XAS

AlexDoor-XAS is a research codebase for comparing action representations in
humanoid articulated-object manipulation. Its first benchmark is door pushing
with the fixed-base IHMC Alex V2 torso in NVIDIA Isaac Sim and Isaac Lab.

The main research variable is the action interface: joint deltas (A1),
end-effector deltas (A2), object-relative deltas (A3), and object-centric
chunks (A4). Policies execute through a shared adapter and produce explicit
safety and provenance evidence.

## Current state

Phases 1–3 are implemented through the completed Phase 3 unified evaluation:

- deterministic proxy and calibrated Alex V2 door-push environments;
- scripted generation and A1–A4 dataset export;
- fail-closed dataset, split, normalization, and provenance contracts;
- A2/A3/A4 adapter execution;
- state-only ACT and Diffusion Policy training and evaluation;
- a validated 550-episode paired A2/A3 scale dataset with four nested views;
- a completed two-cell Gilbreth compatibility pilot and sixteen-cell sweep;
- a matched 576-rollout closed-loop evaluation with curated evidence.

The all-success primary matrix is saturated and does not select a winning
representation, policy family, or dataset size. One ACT-A3-N50 rollout remains
`REVIEW_REQUIRED` after its reproducible force-watch event. Phase 4 VLA work
has not started. See [Project Status](knowledge/wiki/status.md) for the evidence
and boundaries.

## Documentation

- [Technical wiki](knowledge/wiki/index.md) — navigation root for the official
  repository documentation.
- [Project Status](knowledge/wiki/status.md) — completed work, evidence,
  limitations, boundaries, and next steps.
- [System Architecture](knowledge/wiki/topics/system-architecture.md) —
  implemented components, responsibilities, and data/control flows.
- [Action Representations and Adapters](knowledge/wiki/topics/action-representations-and-adapters.md)
  — canonical A1–A4 meanings and execution semantics.
- [Alex V2 Benchmark](knowledge/wiki/topics/alex-v2-benchmark.md) — calibrated
  task geometry, control, sensing, and runtime limits.
- [Provenance and Artifact Lifecycle](knowledge/wiki/topics/provenance-and-artifact-lifecycle.md)
  — fail-closed identities, cluster packages, and curated evidence.

## Quick start

Isaac Sim and Isaac Lab are supplied by the workstation runtime; they are not
installed as Python package dependencies. From the repository root:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Do not use bare system `python3` for Isaac code. Runtime and verification
responsibilities are documented in
[Phase 1](knowledge/wiki/implementation_phases/phase-1-project-and-simulation-readiness.md)
and the [technical wiki](knowledge/wiki/index.md).

## Repository layout

```text
src/alexdoor_xas/   package: assets, envs, actions, adapters, policies, data, eval
scripts/            verification, generation, training, evaluation, cluster tools
configs/            calibration, data, policy, tracking, and pilot contracts
tests/              pure-Python regression and contract tests
knowledge/          user-owned raw research plus the official technical wiki
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
