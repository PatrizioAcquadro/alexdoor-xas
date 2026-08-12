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
- `policies/common/runs.py` and `policies/common/closed_loop.py` own learned-run allocation, resume state, artifact schemas, frozen evaluation protocols, aggregation, plotting, and protocol routing.

There is one registered simulator environment, `AlexDoor-DoorPush-AlexV2-v0`. Neutral door contract/runtime helpers own the observation/action terms, hinge resolution, and door-frame stage reads used by the Alex executor; no alternate task or surrogate-robot runtime is maintained.

## Main flows

Scripted generation records pre-action state and the matching requested/applied action, then exports representation-specific siblings from that same episode. Training selects a split or retained view, validates stored normalization, allocates an exclusive learned run, and maintains both a self-contained best inference checkpoint and a consolidated resumable state until successful completion.

Closed-loop evaluation loads the source run's frozen configuration and self-contained `best.pt`, creates a fresh environment for each canonical pose, executes every requested rollout through adapter-v1, and writes factual aggregate results. Exact protocol matches may complete the source training run; changed protocols create checkpoint-free sibling evaluation runs.

## Storage boundary

`outputs/` is limited to `README.md`, canonical `door_scene/D0.usda`-`D4.usda`, and learned runs under `door_push_alex_v2/{act,diffusion}/`. Reusable episodes remain under `datasets/`. Scripted-run staging, calibration probes, verification evidence, W&B state, arbitrary scenes, and inspection figures live under `~/.cache/alexdoor-xas/`.

## Runtime boundary

The workstation is authoritative for Isaac asset validation, calibration, dataset generation, and closed-loop evaluation. ACT and Diffusion training are Isaac-free and can run wherever the configured PyTorch environment is available. The repository no longer maintains cluster transfer, Slurm, pilot, sweep, smoke-matrix, or unified-matrix orchestration.

## Limits

- Learned observations are state-only; there are no image or language inputs.
- Learned policies cover A2 and A3; A4 is recorded and adapter-executable only.
- Only six right-arm joints are position-controlled; requested rotation is not actuated.
- No repository command controls a physical Alex robot.

## Version Notes

- 2026-08-12 — Added the canonical scene/output boundary, exclusive learned runs, resumable training state, and frozen-protocol evaluation routing.
- 2026-08-12 — Removed alternate simulator runtimes and consolidated routine validation into five Alex V2 gates.
- 2026-08-11 — Removed completed Phase 3 orchestration and reduced the active architecture to benchmark, data, policy, adapter, and evaluation primitives.
