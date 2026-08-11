# System Architecture

AlexDoor-XAS compares explicit action representations through one maintained
control boundary:

`observation -> policy -> action representation -> adapter -> environment`

## Components

- `src/alexdoor_xas/assets/` and `envs/door_task/` construct the calibrated
  Alex V2 benchmark and expose task state.
- `policies/scripted/` generates deterministic demonstrations.
- `recording/` and `data_engine/` record one physical episode and export
  matched A1-A4 representations.
- `dataset/` validates episodes, content-grouped splits, retained views,
  train-only normalization, and chunk sampling.
- `policies/act/` and `policies/diffusion/` train and run state-only A2/A3
  policies from self-contained checkpoints.
- `action/` and `adapters/` validate, transform, correct, reject, and execute
  requested actions.
- `eval/` supplies metrics, force checks, plots, video, and report utilities.

## Main flows

Scripted generation records pre-action state and the matching requested and
applied action, then exports representation-specific siblings from that same
episode. Training selects a split or retained view, recomputes the stored
normalization statistics, and saves a compact checkpoint with weights, model
configuration, dataset descriptor, normalization, and robot identity.

Evaluation loads only the checkpoint and current simulator runtime. It does not
require the training dataset. The policy produces A2 or A3, the adapter mediates
the request, and the Alex V2 executor applies position-only differential IK.

## Runtime boundary

The workstation is authoritative for Isaac asset validation, calibration,
dataset generation, and closed-loop evaluation. ACT and Diffusion training are
Isaac-free and can run wherever the configured PyTorch environment is
available. The repository no longer maintains cluster transfer, Slurm, pilot,
sweep, smoke-matrix, or unified-matrix orchestration.

## Limits

- Learned observations are state-only; there are no image or language inputs.
- Learned policies cover A2 and A3; A4 is recorded and adapter-executable only.
- Only six right-arm joints are position-controlled; requested rotation is not
  actuated.
- No repository command controls a physical Alex robot.

## Version Notes

- 2026-08-11 — Removed completed Phase 3 orchestration and reduced the active
  architecture to benchmark, data, policy, adapter, and evaluation primitives.
