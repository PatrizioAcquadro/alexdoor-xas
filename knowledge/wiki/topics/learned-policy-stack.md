# Learned Policy Stack

AlexDoor-XAS maintains state-only ACT and Diffusion policies for A2 and A3. Both use the same dataset, normalization, adapter, run-output, and evaluation boundaries.

## Shared data and execution

`policies/common/data.py` loads one dataset version and either its normal split or a retained view, then validates train-only normalization by direct recomputation. Policies never bypass adapters: A2 is validated before execution, while A3 is transformed through the door frame and then follows the A2 safety path.

## ACT and Diffusion

`ACTModel` is a conditional variational transformer over action chunks. It optimizes masked L1 reconstruction plus KL regularization and uses a zero latent for deterministic inference. `DiffusionTransformer` predicts epsilon over an action horizon with a causal transformer, DDPM training noise, optional EMA, and DDPM/DDIM inference. Active defaults are in `configs/act.yaml` and `configs/diffusion.yaml`.

## Training-run contract

New runs are exclusively allocated under `outputs/door_push_alex_v2/{act,diffusion}/<run_id>/` with full UTC IDs such as `20260812T153045Z_a3_v3n500_seed0`; same-second collisions append `_r2`, `_r3`, and so on. `resolved_config.json` is immutable and freezes the policy configuration plus the complete evaluation protocol. Reuse requires explicit `--resume <run-directory>` and a resumable `last.pt`; completed runs are never overwritten.

`best.pt` is the best validated self-contained inference checkpoint. `last.pt` is written atomically before epoch 0 and after each completed epoch with model, optimizer, scheduler or explicit null, next epoch, global step, complete history, Python/NumPy/Torch CPU/CUDA RNG states, and Diffusion EMA when used. Successful completion removes `last.pt` only after all required artifacts and the report are complete. Interrupted runs retain `last.pt`, preserve any `best.pt`, and create `error.log`.

ACT history records train L1, KL, total loss, validation L1, batch counts, best epoch/value, and durations. Diffusion history records train MSE, sampled validation L1, learning rate, batch counts, best epoch/value, and durations. Open-loop evaluation reports translation-only aggregate and dx/dy/dz L1, per-episode L1, evaluated steps, and one deterministic worst-episode plot; it does not publish MSE, constant-zero rotation metrics, or per-episode plot sets.

W&B is an optional vanilla SDK integration installed through `.[tracking]`. The four train/evaluate scripts read official `WANDB_*` environment variables and skip the import entirely when `WANDB_MODE` is unset or `disabled`. Enabled training runs log one aggregate dictionary per epoch and one final open-loop dictionary; enabled evaluators log one final overall closed-loop dictionary. Configuration is limited to policy identity plus the relevant dataset/model/train or evaluation summary. The integration does not watch models or publish artifacts, tables, files, or media, and standard local state stays under `outputs/wandb/`.

## Closed-loop evaluation

Each training run freezes the canonical 36-rollout D0-D4 protocol, thresholds, 200 N force limit, horizon, control settings, and policy execution settings. ACT and Diffusion evaluators create a fresh environment per pose and publish one `closed_loop/metrics.json` with factual rollout rows plus overall, pose, fixed/randomized, and pose-plus-subset aggregates. The single summary plot shows time to success, peak force with its limit, and adapter correction/rejection rates.

An exact protocol match may publish the first closed-loop result into the source training run. Any change to poses, seeds, randomization, thresholds, force limits, horizon, control settings, or policy execution creates a timestamped sibling evaluation run with `run_type: evaluation`, `source_run_id`, and `source_checkpoint`; the checkpoint is not copied and the source run is not modified. `traces/` is retained only for unsuccessful rollouts, force-limit exceedances, or explicitly selected keys, and `media/` only when explicitly requested.

Legacy Phase 3 v1 inference checkpoints remain loadable without rewriting. `scripts/verify_policy_rollout.py` remains the compact A2/A3 runtime gate and writes temporary evidence under `~/.cache/alexdoor-xas/verification/`.

## Limits

- Only A2 and A3 have learned policies.
- Models are state-only and benchmark-specific, not VLA systems.
- Training or open-loop loss is not a closed-loop success or safety result.
- The saturated Phase 3 result does not establish equivalence or a winner.

## Version Notes

- 2026-08-12 — Replaced repository-owned W&B configuration, wrappers, artifact gates, and simulated tests with direct optional SDK initialization and aggregate logging.
- 2026-08-12 — Added exclusive timestamped training runs, full resume state, compact training/open-loop artifacts, frozen multi-pose evaluation, protocol-match routing, factual closed-loop metrics, and selective optional artifacts.
- 2026-08-12 — Unified ACT and Diffusion rollout verification and moved deterministic training gates into pytest.
- 2026-08-11 — Introduced compact checkpoint v2 and dataset-independent evaluation while retaining read compatibility with Phase 3 v1 files.
