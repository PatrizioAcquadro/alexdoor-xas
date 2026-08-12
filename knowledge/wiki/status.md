# Project Status

Current as of 2026-08-12. Code and deterministic tests are the source of truth for executable behavior; the wiki preserves historical scientific conclusions and Git preserves removed artifacts and workflow history.

## Maintained capabilities

- Calibrated fixed-base Alex V2 benchmark with exact canonical D0-D4 scene layers and cache-only noncanonical generation.
- Deterministic scripted baseline and matched `phase2.v2`/`v2_pose` A1-A4 episode export with factual termination fields and legacy v0/v1 reads.
- A2/A3 model datasets, content-grouped splits, retained views, and directly recomputed train-only normalization.
- A2/A3/A4 adapters and simulator safety controls.
- State-only ACT and Diffusion training with collision-safe UTC run IDs, immutable resolved configuration, atomic full-state resume checkpoints, compact histories, open-loop summaries, and self-contained best checkpoints.
- Frozen 36-rollout D0-D4 evaluation with factual force/adapter/termination metrics, protocol-match routing, checkpoint-free sibling evaluation runs, and selective traces/media.
- Optional vanilla W&B metric tracking, disabled by default and stored under `outputs/wandb/` only when enabled.

The only maintained execution path is Door + Alex V2 to `v2_pose` A1-A4, training, adapter-v1, and evaluation. The five verification gates are `verify_benchmark_scene.py`, `verify_scripted_baseline.py`, `verify_dataset_interface.py`, `verify_adapters.py`, and `verify_policy_rollout.py`.

The active `configs/` surface is exactly `act.yaml`, `diffusion.yaml`, `scripted_baseline.yaml`, and `alex_v2_door.json`.

## Completed scientific results

The retained `v3_scale_master` has 550 matched A2/A3 episodes across five door poses and usable N50, N100, N250, and N500 train views.

The completed Phase 3 matrix evaluated 16 ACT/Diffusion x A2/A3 x data-size cells over 576 matched rollouts. Every rollout succeeded; across 57,678 adapter decisions, 54,183 were accepted, 3,495 corrected, and zero rejected. Success therefore did not select a policy family, representation, or dataset size, and secondary metrics did not produce a consistent winner.

One ACT-A3-N50 D0 randomized rollout at seed 112 peaked at 219.95 N for one tick. Exact replay reproduced the event; changing only the initial door-frame X position by -1 mm and +1 mm reduced the peak to 66.00 N and 86.44 N. The original cell remains `REVIEW_REQUIRED`.

The canonical experiment pages preserve these conclusions. The former compact evidence packages were removed from the active output structure and remain recoverable from Git history through commit `7f1fc8c`.

## Output boundary

`outputs/` contains only `README.md`, `door_scene/D0.usda`-`D4.usda`, learned ACT/Diffusion runs under `door_push_alex_v2/{act,diffusion}/<run_id>/`, and optional standard W&B state under `wandb/`. Reusable datasets remain in `datasets/`; verifier evidence, calibration authoring evidence, scripted staging, and arbitrary scenes belong under `~/.cache/alexdoor-xas/`.

## Retired workflows

Gilbreth pilot and sweep orchestration, cluster environments, Slurm, transfer and return packaging, scale-dataset construction, multi-pose merge, smoke matrix aggregation, and unified-evaluation orchestration are no longer part of the executable repository. Existing local datasets and checkpoints remain usable and are not rewritten.

The former generic door-task and surrogate-robot simulator runtimes are also retired. Test-only fake environments remain deterministic software doubles, not supported runtime paths.

## Boundaries

- Phase 4 VLA work has not started.
- A4 is recorded and adapter-executable but has no learned policy.
- Policies are state-only; image and language inputs are absent.
- Results cover one simulated door family and seed-0 training only.
- Simulator force and success results do not establish hardware safety, broader generalization, or sim-to-real readiness.
- No command in this repository controls a physical Alex robot.

## Next decision

Any new dataset, harder benchmark, training-seed study, VLA/A4 learning, hardware, or sim-to-real phase requires a separately authorized scope. The current Phase 3 evidence is pipeline validation on a saturated benchmark, not winner selection.
