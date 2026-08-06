# Development Guide

## Runtime

The supported workstation runtime is Isaac Sim 6.0.1 at
`/home/pacquadr/isaacsim` and Isaac Lab `release/3.0.0-beta2` at
`/home/pacquadr/IsaacLab`. Do not install or upgrade either stack for this
project.

Run repository Python through the Isaac Lab launcher, including pure tests:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script-or--m-module>
```

Do not use bare system `python3` for Isaac code. Scripts that require Kit must
create `AppLauncher` before importing `isaaclab`, `omni`, or `pxr` modules.

Install the package once in editable mode:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
```

Optional extras are `.[dev]`, `.[tracking]`, and `.[diffusion]`. Isaac Sim,
Isaac Lab, CUDA, and PyTorch are supplied by the runtime and are not package
dependencies.

## Standard verification

Run the pure suite and fast environment gate:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Run simulator and task gates separately:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz none --device cpu --steps 1
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_task_scene.py --viz none --device cpu --steps 100
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_env.py --viz none --device cpu --steps 100
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_scripted_baseline.py --viz none --device cpu
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_v2_door_baseline.py --viz none --device cpu
```

Run dataset, adapter, and learned-policy gates as needed:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_a2_a3_distinct.py --task door_push_alex_v2 --version v2_pose
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_adapters.py --viz none --device cpu
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_training.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_diffusion_training.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_stabilization_doc.py
```

The ACT and Diffusion rollout gates also require compatible A2 and A3
checkpoints; inspect `--help` and provide `--checkpoint-a2` and
`--checkpoint-a3`. Do not substitute old Alex V1 gate names.

## Data generation

Generate proxy or calibrated Alex V2 episodes with the same entry point:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
  --viz none --device cpu --robot alex_v2 --episodes 5 --randomized 3
```

Add `--video --enable_cameras` only when video is required. Camera-enabled and
headless physics traces must be compared within their own mode. Non-default
door-pose runs cannot export directly; create per-pose no-export runs and merge
them once with `scripts/export_merged_dataset.py` and a tracked pose plan.

Record one learned ACT rollout to a new MP4 under `outputs/` with:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py \
  --viz none --device cpu --enable_cameras \
  --checkpoint outputs/act_door_push/RUN_ID/checkpoints/best.pt \
  --video-output outputs/demo_videos/act_rollout.mp4 \
  rollout.policy_device=cuda rollout.episodes_fixed=0 \
  rollout.episodes_randomized=1 rollout.base_seed=106
```

Video mode is fail-closed: it requires exactly one rollout, refuses paths
outside `outputs/` or existing MP4/JSON targets, publishes only a successful
door opening, and writes the camera-mode evaluation evidence beside the video.
It does not modify the checkpoint's original metrics or curated Phase 3
evidence.

The full nested scale master has a dedicated fail-closed orchestrator. It
requires a clean committed checkout and launches one fresh simulator process
per pose. Generation state is commit/plan-hash bound and resumes only after
re-verifying a completed pose attempt:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/build_scale_dataset.py generate
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/build_scale_dataset.py publish
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/build_scale_dataset.py verify
```

`generate` preserves every candidate and failure under ignored outputs.
`publish` admits exactly the configured successful, safe, content-distinct
episodes, uses overdraw only to replace rejected source seeds, publishes the
paired A2/A3 master atomically, builds the four shared nested views, and writes
eight train-only norm files. Official version directories are never silently
replaced or accumulated.

`verify` independently revalidates the canonical D0-D4 geometry, the validated
calibration schema/content/self-fingerprint and master robot/runtime identity,
exact disjoint source/overdraw seed inventories, all 750 raw candidate fields,
decisions, reasons and replacement links, the common source fingerprint, both
action-export fingerprints, all four deterministic view memberships, and every
numerical field of all eight norms recomputed from the exact training IDs. The
transferable report binds the successful Ubuntu raw replay without requiring
Ubuntu-only paths on Gilbreth. Refreshed inner or outer hashes cannot make
altered candidate, calibration, or normalization evidence pass.

`scripts/verify_dataset_interface.py` is read-only by default. Use
`--write-artifacts` only when intentionally refreshing official split and
normalization files after a dataset regeneration.

## Training and evaluation

ACT and Diffusion use Hydra overrides after their script arguments:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py \
  dataset.space=A2_ee_delta dataset.version=v2_pose

PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_diffusion.py \
  dataset.space=A3_obj_rel_ee_delta dataset.version=v2_pose
```

For a scale cell, keep the physical version and logical view independent:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py \
  dataset.space=A2_ee_delta dataset.version=v3_scale_master \
  dataset.view_id=v3_scale_n100 dataset.obs_preset=core_door_pose
```

The view loader requires the final publication marker, exact shared split
payload, and the matching per-view norm. It never recomputes a scale norm on
the cluster or includes validation/test records in normalization.

Diffusion training is CUDA-first and fails rather than silently falling back
to CPU. Simulator evaluation remains on CPU under the validated calibration
contract. Evaluation commands are `scripts/eval_act.py` and
`scripts/eval_diffusion.py`; the checkpoint action space, observation preset,
dataset fingerprint, and split provenance must match the live dataset.

## Experiment tracking

W&B is optional and disabled by default through `configs/wandb.yaml`.
Use `wandb.mode=offline` for durable local/cluster capture or
`wandb.mode=online` only after authentication. Never store API keys or other
credentials in the repository, configs, manifests, rendered jobs, or logs.

## Local and cluster boundary

Ubuntu is authoritative for Isaac simulation, dataset generation, calibration,
and closed-loop evaluation. Gilbreth is a non-Isaac training environment.

The compatibility-pilot workflow is documented in [`cluster.md`](cluster.md).
It transfers only the existing N50 dataset and two short training cells. It is
not the full scientific dataset-scale sweep. Do not build a transfer manifest
from a dirty tree, guess Gilbreth account/runtime values, submit `sbatch`, or
start the later sweep without explicit authorization.

The prepared full-sweep workflow is also documented there. Local preparation
may build and verify its ignored transfer artifacts and render/syntax-check an
example array script. Transfer, `sbatch`, training, and Phase 4 remain separate
explicitly authorized actions.

Build the sweep manifest only from a clean committed checkout and immediately
run its `verify` subcommand. The manifest builder reruns the same transferable
generation and normalization invariants; do not copy forward an older manifest
after code, config, pose-plan, calibration, or dataset metadata changes.

## Change discipline

- Preserve action, frame, timing, calibration, dataset, and provenance
  contracts unless a reviewed spec explicitly changes them.
- Keep generated datasets, checkpoints, logs, videos, and raw outputs out of
  Git.
- Use explicit verification scripts instead of manual simulator state.
- Treat restricted-shell GPU/display failures as inconclusive until rerun in a
  host-visible shell.
- Keep project status current in [`status.md`](status.md); do not create a new
  phase report for every implementation increment.
