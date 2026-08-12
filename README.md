# AlexDoor-XAS

AlexDoor-XAS compares action representations for contact-rich humanoid manipulation. Its first benchmark is simulated door pushing with the fixed-base IHMC Alex V2 torso in NVIDIA Isaac Sim and Isaac Lab.

## Current product

The maintained path includes:

- the calibrated Alex V2 door benchmark and scripted baseline;
- matched A1-A4 episode export, with learned-policy datasets for A2 and A3;
- content-grouped splits and train-only normalization;
- state-only ACT and Diffusion training and closed-loop evaluation;
- A2/A3/A4 adapters, force checks, and execution safety controls;
- optional vanilla W&B tracking for training and evaluation metrics.

The only operational path is:

`Door + Alex V2 -> v2_pose A1-A4 -> training -> adapter-v1 -> evaluation`

The completed Gilbreth pilot, cluster sweep, smoke-matrix aggregation, and unified-evaluation orchestration are no longer executable product workflows. Their scientific conclusions remain in the wiki; Git owns their source and artifact history.

The Phase 3 matrix was success-saturated and did not identify a winning policy, representation, or dataset size. One ACT-A3-N50 rollout remains `REVIEW_REQUIRED` after a reproducible force-watch event. Phase 4 VLA work has not started. See [Project Status](knowledge/wiki/status.md).

## Quick start

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
```

Do not use bare system `python3` for Isaac code.

W&B is disabled by default and remains optional. Install it only when tracking is needed:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e ".[tracking]"
```

## Operational workflow

Generate matched Alex V2 episodes and export A1-A4 under the active
`v2_pose` dataset version:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/run_scripted_baseline.py --viz none --device cuda:0
```

Train ACT or Diffusion on A2 or A3; both active configs default to
`door_push_alex_v2/v2_pose` and GPU training:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py --space A2_ee_delta
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_diffusion.py --space A3_obj_rel_ee_delta
```

Set the official `WANDB_MODE=offline` or `WANDB_MODE=online` environment variable to enable
tracking; `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME`, `WANDB_RUN_GROUP`,
`WANDB_JOB_TYPE`, `WANDB_TAGS`, `WANDB_API_KEY`, `WANDB_RUN_ID`, and `WANDB_RESUME`
are passed through to the SDK. Unset or `WANDB_MODE=disabled` performs no W&B import or
write. Enabled runs use `outputs/wandb/` and log one aggregate per epoch plus one final
training or evaluation aggregate; no artifacts or media are uploaded automatically.

Training runs are allocated under `outputs/door_push_alex_v2/{act,diffusion}/<run_id>/`. Resume only an incomplete run with `--resume <run-directory>`. Evaluate a completed self-contained checkpoint with its frozen protocol:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py --checkpoint outputs/door_push_alex_v2/act/<run_id>/checkpoints/best.pt --device cuda:0
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_diffusion.py --checkpoint outputs/door_push_alex_v2/diffusion/<run_id>/checkpoints/best.pt --device cuda:0
```

The default scene is `outputs/door_scene/D0.usda`; D0-D4 are the only scene layers allowed under `outputs/door_scene/`. See [the output contract](outputs/README.md).

The supported verification surface contains exactly five gates:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_benchmark_scene.py --viz none --device cuda:0
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_scripted_baseline.py --viz none --device cuda:0
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_adapters.py --viz none --device cuda:0
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_policy_rollout.py --policy act --viz none --device cuda:0 --checkpoint-a2 <a2.pt> --checkpoint-a3 <a3.pt>
```

Use `--policy diffusion` for Diffusion checkpoints; only that selection accepts `--sampler`, `--inference-steps`, and `--n-action-steps`. Verifier artifacts are written under `~/.cache/alexdoor-xas/verification/`.

`scripts/author_alex_v2_door_calibration.py` is a mutating maintenance command,
not a verifier, and is never part of routine validation.

## Repository layout

```text
src/alexdoor_xas/   assets, environments, actions, adapters, data, policies, evaluation
scripts/            supported verification, generation, inspection, training, evaluation
configs/            four active calibration, policy, and scripted configs
tests/              deterministic regression and contract tests
knowledge/          user-owned raw research and the official technical wiki
datasets/           reusable local datasets (ignored except README)
outputs/            canonical D0-D4 scenes, learned-policy runs, optional W&B state
```

Machine-local assets, datasets, learned checkpoints, videos, logs, W&B state, and runtime caches stay out of Git. Historical Phase 3 conclusions are preserved in the wiki and detailed removed artifacts remain available through Git history.

## Documentation

- [Technical wiki](knowledge/wiki/index.md)
- [System Architecture](knowledge/wiki/topics/system-architecture.md)
- [Episode and Dataset Contracts](knowledge/wiki/topics/episode-and-dataset-contracts.md)
- [Learned Policy Stack](knowledge/wiki/topics/learned-policy-stack.md)
- [Alex V2 Benchmark](knowledge/wiki/topics/alex-v2-benchmark.md)

## License

This repository is proprietary. No license grant is provided unless a separate
license file or written agreement states otherwise.
