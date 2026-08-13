# Learned Policy Stack

AlexDoor-XAS maintains state-only ACT and Diffusion policies for A2 and A3. Both use the same dataset, normalization, adapter, run-output, and evaluation boundaries.

## Shared data and execution

`policies/common/data.py` loads one dataset version and either its normal split or a retained view, then validates train-only normalization by direct recomputation. Policies never bypass adapters: A2 is validated before execution, while A3 is transformed through the door frame and then follows the A2 safety path.

## ACT and Diffusion

`ACTModel` is a conditional variational transformer over action chunks. It optimizes masked L1 reconstruction plus KL regularization and uses a zero latent for deterministic inference. `DiffusionTransformer` predicts epsilon over an action horizon with a causal transformer, DDPM training noise, optional EMA, and DDPM/DDIM inference. Active defaults are in `configs/act.yaml` and `configs/diffusion.yaml`.

Both policy configs use one OmegaConf loader. YAML values are overridden first by CLI `key=value` tokens and then by explicit argparse options; no global Hydra state is used.

## Training-run contract

New runs are exclusively allocated under `outputs/door_push_alex_v2/{act,diffusion}/<run_id>/` with full UTC IDs such as `20260812T153045Z_a3_v3n500_seed0`; same-second collisions append `_r2`, `_r3`, and so on. `resolved_config.json` is immutable and freezes the policy configuration plus the complete evaluation protocol. Reuse requires explicit `--resume <run-directory>` and a resumable `last.pt`; completed runs are never overwritten.

`scripts/train_policy.py --policy {act,diffusion}` is the only training entry point. The policy selector remains mandatory during resume and must match the frozen run. `scripts/eval_policy.py --checkpoint ...` is the only learned-policy evaluator and detects ACT or Diffusion from the source `resolved_config.json`.

`best.pt` is the best validated self-contained inference checkpoint. `last.pt` is written atomically before epoch 0 and after each completed epoch with model, optimizer, scheduler or explicit null, next epoch, global step, complete history, Python/NumPy/Torch CPU/CUDA RNG states, and Diffusion EMA when used. Successful completion removes `last.pt` only after all required artifacts and the report are complete. Interrupted runs retain `last.pt`, preserve any `best.pt`, and create `error.log`.

ACT history records train L1, KL, total loss, validation L1, batch counts, best epoch/value, and durations. Diffusion history records train MSE, sampled validation L1, learning rate, batch counts, best epoch/value, and durations. Open-loop evaluation reports translation-only aggregate and dx/dy/dz L1, per-episode L1, evaluated steps, and one deterministic worst-episode plot; it does not publish MSE, constant-zero rotation metrics, or per-episode plot sets.

W&B is an optional vanilla SDK integration installed through `.[tracking]`. The unified train/evaluate scripts read official `WANDB_*` environment variables and skip the import entirely when `WANDB_MODE` is unset or `disabled`. Enabled training runs log one aggregate dictionary per epoch and one final open-loop dictionary; enabled evaluators log one final overall closed-loop dictionary. Configuration is limited to policy identity plus the relevant dataset/model/train or evaluation summary. The integration does not watch models or publish artifacts, tables, files, or media, and standard local state stays under `outputs/wandb/`.

## Closed-loop evaluation

Each training run freezes the canonical 36-rollout D0-D4 protocol, thresholds, 200 N force limit, horizon, control settings, and policy execution settings. The evaluator creates a fresh environment per pose and allocates every result under `<training-run>/closed_loop/<UTC-id>[_rN]/`; legacy files already under `closed_loop/` are left untouched.

Each evaluation directory contains immutable `resolved_config.json`, `metrics.json`, `summary.png`, and `report.md`. The resolved config records `run_type: evaluation`, `source_run_id`, checkpoint path, frozen training config, and the requested protocol. Metrics contain factual rollout rows plus overall, pose, fixed/randomized, and pose-plus-subset aggregates; the summary shows time to success, peak force with its limit, and adapter correction/rejection rates. `traces/` exists only for unsuccessful rollouts, force-limit exceedances, or explicit `--trace-rollout` keys. Evaluation never changes the training report and never creates top-level sibling runs.

Preflight requires only a valid `checkpoints/best.pt` location, training `resolved_config.json`, and evaluation protocol. Training plots, reports, open-loop output, and the absence of `last.pt` are not evaluation prerequisites.

ACT and Diffusion accept only self-contained checkpoint v2 files whose robot-asset identity exactly matches the active Alex V2 runtime. Older or unfingerprinted checkpoint formats and cross-model transfer evaluation are unsupported. `scripts/verify_policy_rollout.py` remains the compact A2/A3 runtime gate and writes temporary evidence under `~/.cache/alexdoor-xas/verification/`.

Both policy families use `policies/common/checkpoint.py` for payload serialization,
normalization, robot identity, and atomic I/O. Each policy loader reconstructs
its own model directly without policy-specific checkpoint modules. The
`alexdoor_xas.{act,diffusion}.v2` formats remain unchanged.

## Limits

- Only A2 and A3 have learned policies.
- Models are state-only and benchmark-specific, not VLA systems.
- Training or open-loop loss is not a closed-loop success or safety result.
- The saturated Phase 3 result does not establish equivalence or a winner.

## Version Notes

- 2026-08-12 — Replaced protocol-match routing with exclusive immutable evaluation children under each training run and reduced preflight to its operational inputs.
- 2026-08-12 — Unified OmegaConf loading, shared model/checkpoint primitives, and policy-owned model reconstruction while reducing internal APIs and tests.
- 2026-08-12 — Consolidated model-neutral checkpoint v2 serialization and
  validation while preserving the ACT and Diffusion public loaders and formats.
- 2026-08-12 — Consolidated policy execution into `train_policy.py` and `eval_policy.py`; evaluation now detects the policy family from the frozen source run.
- 2026-08-12 — Removed pre-v2 checkpoint loading and cross-model transfer; runtime evaluation now requires an exact Alex V2 robot-asset identity match.
- 2026-08-12 — Replaced repository-owned W&B configuration, wrappers, artifact gates, and simulated tests with direct optional SDK initialization and aggregate logging.
- 2026-08-12 — Added exclusive timestamped training runs, full resume state, compact training/open-loop artifacts, frozen multi-pose evaluation, protocol-match routing, factual closed-loop metrics, and selective optional artifacts.
- 2026-08-12 — Unified ACT and Diffusion rollout verification and moved deterministic training gates into pytest.
- 2026-08-11 — Introduced compact checkpoint v2 and dataset-independent evaluation.
