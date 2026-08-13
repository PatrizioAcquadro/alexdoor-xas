# System Architecture

AlexDoor-XAS compares explicit action representations through one maintained control boundary: `observation -> policy -> action representation -> adapter -> environment`.

The end-to-end operational path is `Door + Alex V2 -> v2_pose A1-A4 -> training -> adapter-v1 -> evaluation`.

## Components

- `src/alexdoor_xas/assets/` and `envs/door_task/` construct the calibrated Alex V2 benchmark and expose task state.
- `policies/scripted/` generates deterministic demonstrations.
- `recording/` and `data_engine/` record one physical episode and export matched A1-A4 representations.
- `dataset/` validates episodes, content-grouped splits, retained views, train-only normalization, and chunk sampling.
- `policies/act/` and `policies/diffusion/` train and run state-only A2/A3 policies from self-contained checkpoints.
- `action/` and `adapters/` validate, transform, correct, reject, and execute requested actions.
- `policies/common/runs.py` and `policies/common/closed_loop.py` own training/evaluation allocation, resume state, artifact schemas, frozen protocols, aggregation, and plotting.

There is one registered simulator environment, `AlexDoor-DoorPush-AlexV2-v0`. `DoorPushAlexV2EnvCfg` and `DoorPushAlexV2Env` own the complete single-environment runtime; the only separate environment helper aggregates raw PhysX contacts for one exact door actor. No generic robot layer, compatibility shim, sensorless runtime, multi-environment path, or alternate simulator task is maintained.

## Main flows

Scripted generation records pre-action state and the matching requested/applied action, then exports representation-specific siblings from that same episode. Training selects a split or retained view, validates stored normalization, allocates an exclusive learned run, and maintains both a self-contained best inference checkpoint and a consolidated resumable state until successful completion. Optional W&B tracking calls the SDK directly from the training and evaluation scripts, remains disabled unless `WANDB_MODE` enables it, and records only compact configuration and aggregate scalar metrics.

Generation and rollout consume one strict runtime snapshot contract: contact force and sensed state from one `contact_state()` read, robot provenance and base pose, full joint state/names/limits, episode counter, settle evidence, and IK-clamp telemetry. Geometric contact remains recorded for analysis but never replaces the PhysX sensor.

Closed-loop evaluation loads the source run's frozen configuration and self-contained `best.pt`, creates a fresh environment for each canonical pose, executes every requested rollout through adapter-v1, and writes factual results to an immutable child under the training run's `closed_loop/` directory.

## Storage boundary

`outputs/` is limited to `README.md`, canonical `door_scene/D0.usda`-`D4.usda`, learned runs under `door_push_alex_v2/{act,diffusion}/`, and optional standard SDK state under `wandb/`. Reusable episodes remain under `datasets/`. Scripted-run staging, verification evidence, and arbitrary scenes live under `~/.cache/alexdoor-xas/`.

## Runtime boundary

The workstation is authoritative for Isaac asset validation, calibration, dataset generation, and closed-loop evaluation. ACT and Diffusion training are Isaac-free and can run wherever the configured PyTorch environment is available. The repository no longer maintains cluster transfer, Slurm, pilot, sweep, smoke-matrix, or unified-matrix orchestration.

## Limits

- Learned observations are state-only; there are no image or language inputs.
- Learned policies cover A2 and A3; A4 is recorded and adapter-executable only.
- Only six right-arm joints are position-controlled; requested rotation is not actuated.
- No repository command controls a physical Alex robot.

## Version Notes

- 2026-08-12 — Made every learned-policy evaluation an immutable child of its source training run.
- 2026-08-12 — Collapsed the environment runtime to one Alex V2 config, one concrete environment, and one exact-actor contact helper; removed generic, legacy, sensorless, and multi-environment paths.
- 2026-08-12 — Removed calibration authoring and retained one directly validated runtime config.
- 2026-08-12 — Replaced the custom W&B wrapper and configuration with direct, environment-controlled SDK logging in the four learned-policy scripts.
- 2026-08-12 — Added the canonical scene/output boundary, exclusive learned runs, resumable training state, and frozen-protocol evaluation routing.
- 2026-08-12 — Removed alternate simulator runtimes and consolidated routine validation into five Alex V2 gates.
- 2026-08-11 — Removed completed Phase 3 orchestration and reduced the active architecture to benchmark, data, policy, adapter, and evaluation primitives.
