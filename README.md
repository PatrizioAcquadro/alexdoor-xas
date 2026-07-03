# AlexDoor-XAS

## About

AlexDoor-XAS is a lightweight Isaac Sim / Isaac Lab research package for
studying humanoid door manipulation with the IHMC Alex torso. It provides asset
loaders, door-task environments, scripted rollout tools, recording and export
utilities, and evaluation helpers for comparing manipulation action
representations.

The project is intentionally organized as a lightweight Python package. Isaac
Sim, Isaac Lab, CUDA, and PyTorch are supplied by the local simulator
installation rather than installed as package dependencies.

## Overview

The repository provides a controlled door-manipulation benchmark around three
core concerns:

- loading local Alex and door scene assets through a stable path registry
- running Isaac Lab door-task environments for proxy and Alex-based execution
- generating, recording, exporting, and evaluating door-push rollouts

Generated datasets, videos, checkpoints, logs, and simulator outputs are kept
out of git by default. Only source code, tests, verification scripts, packaging
metadata, and placeholder README files for artifact directories are tracked.

## Capabilities

- Local asset discovery for the Alex model and door scenes.
- Single-door task fixture generation from machine-local USD assets.
- Isaac Lab environments for reset, step, scripted interaction, and Alex-based
  execution.
- Action-space utilities for end-effector, object-relative, and object-centric
  representations.
- Deterministic scripted baseline controller for door-push rollouts.
- Episode recording and dataset export utilities for HDF5, JSON, and JSONL
  outputs.
- Evaluation utilities for metrics, failure labels, plots, reports, and sanity
  checks.
- Verification scripts for environment readiness, asset loading, door-task
  setup, scripted rollout export, IK probing, and Alex door interaction.

## Repository Structure

```text
.
|-- README.md
|-- pyproject.toml
|-- .gitignore
|-- src/alexdoor_xas/
|   |-- paths.py
|   |-- assets/
|   |-- action/
|   |-- envs/
|   |-- policies/
|   |-- recording/
|   |-- data_engine/
|   `-- eval/
|-- scripts/
|   |-- check_env.py
|   |-- verify_assets.py
|   |-- verify_door_task_scene.py
|   |-- verify_door_env.py
|   |-- verify_scripted_baseline.py
|   |-- verify_alex_ik_probe.py
|   |-- verify_alex_door_baseline.py
|   `-- run_scripted_baseline.py
|-- tests/
|-- datasets/
|   `-- README.md
`-- outputs/
    `-- README.md
```

The tracked source tree is divided by responsibility:

- `src/alexdoor_xas/paths.py` defines canonical local paths and environment
  overrides.
- `src/alexdoor_xas/assets/` contains Alex, scene, and door-task asset helpers.
- `src/alexdoor_xas/action/` contains action tags, structures, and frame
  conversion utilities.
- `src/alexdoor_xas/envs/` contains Isaac Lab door-task environments.
- `src/alexdoor_xas/policies/` contains scripted controllers.
- `src/alexdoor_xas/recording/` contains episode buffers and writers.
- `src/alexdoor_xas/data_engine/` contains generation, export, and run
  orchestration.
- `src/alexdoor_xas/eval/` contains metrics, plots, reports, and sanity checks.
- `scripts/` contains executable verification and data-generation entrypoints.
- `tests/` contains pure-Python regression and contract tests.
- `datasets/README.md` and `outputs/README.md` document local artifact
  conventions while generated contents remain ignored.

## Requirements

- Python 3.11 or newer.
- NVIDIA Isaac Sim and Isaac Lab installed locally.
- A Python environment supplied by Isaac Lab for simulator-backed commands.
- Local Alex and scene assets available on the workstation:
  - `~/Desktop/Alex-robot/alex_models/`
  - `~/Desktop/CombinedScene/`
- Python package dependencies declared in `pyproject.toml`.

Set `ALEXDOOR_ASSETS_ROOT` if the local asset root is different from the default
desktop layout.

## Installation

From the repository root:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
```

For development tools:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e ".[dev]"
```

Isaac-backed commands should be run through the Isaac Lab launcher. System
`python3` is not expected to import Isaac Sim, Isaac Lab, Omniverse, or USD
runtime modules in this project.

## Verification

Run the pure-Python test suite:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
```

Run fast environment checks:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Run Isaac-backed smoke and task checks:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz none --device cpu --steps 1
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_task_scene.py --viz none --device cpu --steps 100
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_env.py --viz none --device cpu --steps 100
```

Run scripted rollout and Alex-specific verification:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_scripted_baseline.py --viz none --device cpu
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_ik_probe.py --viz none --device cpu --contact
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_door_baseline.py --viz none --device cpu
```

## Usage

Generate scripted door-push rollouts with the proxy end-effector backend:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --robot proxy --episodes 5 --randomized 3
```

Generate scripted door-push rollouts with the Alex backend:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --robot alex --episodes 5 --randomized 3 --video --enable_cameras
```

Use `--help` on any script to inspect supported flags:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py --help
```

## Data and Artifacts

`datasets/` is reserved for reusable exported episode datasets. `outputs/` is
reserved for per-run artifacts such as metrics, plots, reports, videos,
checkpoints, logs, and temporary episode captures.

Both directories are ignored by default except for their README files. This
keeps large generated files and machine-specific simulator artifacts out of the
repository while preserving the expected local layout.

Small, deliberately curated review artifacts may be placed under
`outputs/curated/` according to the repository ignore rules.

## Development Notes

- Keep simulator dependencies out of `pyproject.toml`; they are provided by the
  Isaac installation.
- Prefer repository-local path helpers instead of hard-coded absolute paths in
  source modules.
- Keep generated data, videos, logs, and binary simulator artifacts out of git.
- Use explicit script entrypoints for verification rather than relying on manual
  simulator state.
- Treat asset-load checks, environment checks, scripted rollout checks, and
  Alex-specific checks as separate validation layers.

## License

This repository is proprietary. No license grant is provided unless a separate
license file or written agreement states otherwise.
