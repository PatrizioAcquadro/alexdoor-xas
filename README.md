# AlexDoor-XAS

AlexDoor-XAS compares action representations for contact-rich humanoid
manipulation. Its first benchmark is simulated door pushing with the fixed-base
IHMC Alex V2 torso in NVIDIA Isaac Sim and Isaac Lab.

## Current product

The maintained path includes:

- the calibrated Alex V2 door benchmark and scripted baseline;
- matched A1-A4 episode export, with learned-policy datasets for A2 and A3;
- content-grouped splits and train-only normalization;
- state-only ACT and Diffusion training and closed-loop evaluation;
- A2/A3/A4 adapters, force checks, and execution safety controls;
- W&B as optional run tracking.

The completed Gilbreth pilot, cluster sweep, smoke-matrix aggregation, and
unified-evaluation orchestration are no longer executable product workflows.
Their scientific conclusions remain in the wiki and curated reports; Git owns
their source history.

The Phase 3 matrix was success-saturated and did not identify a winning policy,
representation, or dataset size. One ACT-A3-N50 rollout remains
`REVIEW_REQUIRED` after a reproducible force-watch event. Phase 4 VLA work has
not started. See [Project Status](knowledge/wiki/status.md).

## Quick start

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Do not use bare system `python3` for Isaac code.

## Repository layout

```text
src/alexdoor_xas/   assets, environments, actions, adapters, data, policies, evaluation
scripts/            supported verification, generation, inspection, training, evaluation
configs/            five active calibration, policy, scripted, and tracking configs
tests/              deterministic regression and contract tests
knowledge/          user-owned raw research and the official technical wiki
datasets/           reusable local datasets (ignored except README)
outputs/            local runs plus the small tracked curated evidence set
```

Machine-local assets, datasets, checkpoints, videos, logs, and ordinary run
outputs stay out of Git.

## Documentation

- [Technical wiki](knowledge/wiki/index.md)
- [System Architecture](knowledge/wiki/topics/system-architecture.md)
- [Episode and Dataset Contracts](knowledge/wiki/topics/episode-and-dataset-contracts.md)
- [Learned Policy Stack](knowledge/wiki/topics/learned-policy-stack.md)
- [Alex V2 Benchmark](knowledge/wiki/topics/alex-v2-benchmark.md)

## License

This repository is proprietary. No license grant is provided unless a separate
license file or written agreement states otherwise.
