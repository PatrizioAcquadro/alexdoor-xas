# System Architecture

AlexDoor-XAS compares action representations through one maintained boundary:

`observation -> policy or scripted controller -> A1-A4 representation -> adapter -> Alex V2 door environment`

The operational learned-policy path is `Door + Alex V2 -> v2_pose A2/A3 -> ACT or Diffusion -> adapter-v1 -> closed-loop evaluation`.

## Runtime

`src/alexdoor_xas/envs/door_task/` contains one registered simulator environment, `AlexDoor-DoorPush-AlexV2-v0`. `DoorPushAlexV2EnvCfg` and `DoorPushAlexV2Env` own the single-environment Alex V2 scene, reset, calibrated tool-point IK, task state, termination, and telemetry. The only separate environment helper aggregates raw PhysX contacts for the exact door actor.

Generic Alex V2 asset construction comes from the external `ihmc_alex_isaaclab` extension. Door-specific calibration, scene composition, control, contact selection, and task semantics remain in this repository.

No generic robot runtime, surrogate robot, sensorless path, multi-environment path, or alternate simulator task is maintained.

## Data Path

`policies/scripted/` produces deterministic door-push actions. `recording/` aligns pre-action state with requested and applied actions and stores the terminal response. `data_engine/` exports matched A1-A4 products from the same physical episode.

`dataset/` loads A1-A4 records, validates content-grouped splits and train-only normalization, exposes the three supported observation presets, and samples model chunks. A2 and A3 are the learned-policy inputs; A1 and A4 remain non-learned products.

## Policy and Evaluation Path

`policies/common/config.py` provides strict OmegaConf-backed configuration loading. `policies/common/checkpoint.py` stores the minimum self-contained inference contract, including model shape/configuration, normalization, dataset identity, and Alex V2 robot identity.

`scripts/train_policy.py` allocates exclusive ACT or Diffusion runs, writes immutable resolved configuration, maintains one resumable `last.pt`, and publishes the best inference checkpoint and compact training/open-loop summaries. `scripts/eval_policy.py` loads the source run and checkpoint, executes the frozen D0-D4 protocol through adapter-v1, and writes a separate immutable evaluation result.

## Storage Boundary

- `configs/` contains only the four active task, scripted, ACT, and Diffusion configurations.
- `datasets/` contains reusable episodes, shared splits, retained views, and normalization.
- `outputs/door_scene/` contains the canonical D0-D4 USD layers.
- `outputs/door_push_alex_v2/` contains learned ACT and Diffusion runs.
- `outputs/wandb/` contains optional standard SDK state.
- `~/.cache/alexdoor-xas/` contains verification, scripted staging, and noncanonical scene artifacts.

## Deployment Boundary

The workstation is authoritative for Isaac asset checks, data generation, and closed-loop evaluation. ACT and Diffusion training do not import Isaac and can run in a compatible PyTorch environment. The repository has no maintained cluster orchestration and no physical-robot command path.

## Version Notes

- 2026-08-13 — Reduced the architecture description to the single maintained Alex V2 data, policy, adapter, and evaluation path.
