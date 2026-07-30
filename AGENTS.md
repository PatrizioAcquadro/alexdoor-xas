# AlexDoor-XAS Repository Guidance

## Project Scope

- AlexDoor-XAS compares A1–A4 action representations for contact-rich humanoid
  manipulation, using door pushing with the fixed-base IHMC Alex V2 torso as
  the first Isaac Sim/Isaac Lab benchmark.
- This repository owns simulation task construction, scripted data generation,
  dataset contracts, adapters, state-only ACT/Diffusion training, evaluation,
  and provenance. It does not provide a live physical-robot control layer.
- The system boundary is observation → policy → explicit action representation
  → adapter → environment; safety decisions and trial evidence are recorded
  alongside execution.

## Repository Map

- `src/alexdoor_xas/` — package code for assets, environments, action spaces,
  adapters, datasets, policies, evaluation, recording, and cluster contracts.
- `scripts/` — supported verification, generation, training, evaluation,
  packaging, and cluster entry points.
- `configs/` — versioned calibration, pose-plan, policy, tracking, pilot, sweep,
  and unified-evaluation contracts.
- `tests/` — deterministic pure-Python regression and contract suite.
- `docs/` — canonical intent (`PROJECT_GUIDELINES.md`), technical contracts
  (`architecture.md`), workflows (`development.md`), status, and cluster runbook.
- `environment/` — the reproducible non-Isaac Gilbreth Python environment.
- `datasets/` — reusable exported episodes and shared split/normalization
  artifacts; generated contents are ignored.
- `outputs/` — per-run outputs and the only tracked evidence area,
  `outputs/curated/`.

## Environment

- The package requires Python 3.11 or newer. The workstation uses the Python
  supplied by `/home/pacquadr/IsaacLab/isaaclab.sh`; do not use bare system
  Python for Isaac workflows.
- The supported workstation stack is Isaac Sim 6.0.1 at
  `/home/pacquadr/isaacsim` and the Alex-enabled Isaac Lab checkout at
  `/home/pacquadr/IsaacLab` on branch `pacquadr/alex-v2-asset`. Isaac Sim,
  Isaac Lab, PyTorch, and CUDA are external runtime dependencies, not package
  dependencies.
- Required machine-local assets default to
  `~/Desktop/Alex/urdf/alex_v2.urdf`,
  `~/Desktop/CombinedScene/CombinedHallwayScene/combinedScene.usda`, and
  `~/Desktop/CombinedScene/Door.usd`.
- Asset locations may be overridden with `ALEX_V2_ASSET_ROOT` and
  `ALEXDOOR_ASSETS_ROOT`; `ALEXDOOR_V2_RUNTIME_CACHE_ROOT` overrides the
  generated Alex V2 runtime cache.
- Gilbreth uses Python 3.11 from
  `environment/gilbreth_pilot_py311.yml` plus an explicit PyTorch/CUDA build
  selected from the live driver. Isaac modules must be absent there.

## Commands

- Editable install:
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .`
- Deterministic test suite:
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q`
- Lint:
  `ruff check .`
- Workstation environment/assets:
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py`
- CPU simulator gates, in increasing scope:
  `scripts/verify_assets.py --viz none --device cpu --steps 1`,
  `scripts/verify_door_task_scene.py --viz none --device cpu --steps 100`,
  `scripts/verify_door_env.py --viz none --device cpu --steps 100`,
  `scripts/verify_scripted_baseline.py --viz none --device cpu`, and
  `scripts/verify_alex_v2_door_baseline.py --viz none --device cpu`; prefix
  each with
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p`.
- Dataset and adapter contracts:
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py`
  and
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_adapters.py --viz none --device cpu`.
- Training uses Hydra overrides after script arguments, for example:
  `PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py dataset.space=A2_ee_delta dataset.version=v2_pose`.
  Diffusion uses `scripts/train_diffusion.py` with the same override pattern.

## Project Invariants

- Preserve the canonical action tags and meanings:
  `A1_joint_delta`, `A2_ee_delta`, `A3_obj_rel_ee_delta`, and
  `A4_obj_centric_chunk`. A2/A3 are six-dimensional
  `(dx, dy, dz, drx, dry, drz)` deltas.
- A2 is world-frame; A3 is expressed in the static hinge-anchored door frame
  whose +Z axis is the hinge; A4 contact targets are in the moving panel frame.
  Frames are Z-up, distances are meters, angles are radians, and quaternions
  are `(x, y, z, w)`.
- Policies do not import or bypass adapters, adapters do not import policies,
  and reusable core modules stay Isaac-free where possible. Isaac scripts must
  initialize `AppLauncher` before importing `isaaclab`, `omni`, or `pxr`.
- The calibrated Alex V2 benchmark uses position-only differential IK on the
  six right-arm joints, a collision-derived tool point on
  `RIGHT_GRIPPER_Z_LINK`, and door-panel-filtered force evidence.
- Simulator timing is `sim.dt = 1/120` with decimation 2
  (`control_dt = 1/60`). Per-tick A2 limits are 0.02 m and 0.05 rad; success is
  the first hinge-angle crossing at `pi/4`.
- The current episode container is `phase2.v1`: A1/A2/A3 use one HDF5 file plus
  one JSON sidecar per episode, while A4 uses JSON Lines. Observations and force
  samples are pre-action; `terminal_contact` records the response to the last
  executed action.
- Models consume data through `EpisodeDataset` or `A4ChunkDataset`, never by
  reading raw HDF5 keys. Matched action spaces share episode identities and
  splits; normalization uses only training IDs.
- Calibration, datasets, views, normalization, checkpoints, evaluations, and
  cluster packages are fail-closed and fingerprint-bound. Do not refresh a
  fingerprint or expected artifact merely to absorb unexplained drift.

## Hardware, Simulation and GPU

- The current workstation has one NVIDIA GeForce RTX 4090 with 24 GiB VRAM.
  Validated simulation uses CPU physics; training and policy inference use
  CUDA where configured.
- Local Ubuntu is authoritative for Isaac simulation, calibration, dataset
  generation, and closed-loop evaluation. Simulator validation is not
  physical-hardware validation.
- Gilbreth is a non-Isaac training target. Historical repository evidence
  validates A100 80GB training; current pilot/sweep contracts request one GPU
  per cell and at most two concurrent cells. Account, partition, QOS, and live
  allocation are intentionally not fixed in the repository and must be checked
  before rendering or submitting Slurm jobs.
- No physical Alex execution command exists. Hardware progression and
  calibration changes require a separately defined, safety-reviewed scope.

## Artifacts

- Store reusable data under
  `datasets/<task>/<action_space>/<version>/`. A generation pass owns a version;
  re-exporting that version replaces it, so use a new version for changed
  generation and regenerate shared splits and train-only normalization.
- Store each run under `outputs/<experiment>/<run_id>/` with its metrics,
  plots, videos, checkpoints, logs, and captures. Raw runs, model weights,
  videos, HDF5 files, and logs remain ignored.
- Promote only small review artifacts (`md`, `json`, `csv`, `png`, `svg`, or
  `txt`) to `outputs/curated/`. Existing curated Phase 3 evidence is historical
  evidence and must not be regenerated or replaced without an explicitly
  authorized artifact refresh.
- Cluster transfer and return packages use exact SHA-256 inventories and
  attempt-specific paths; never mix files from different Slurm attempts or
  rewrite retained historical packages in place.
