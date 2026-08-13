# Project Status

Current as of 2026-08-13. Code, configurations, and deterministic tests define executable behavior. This page describes only the maintained repository; completed research is summarized separately as historical evidence.

## Current System

AlexDoor-XAS maintains one simulation workflow:

`scripted Alex V2 door push -> matched v2_pose A1-A4 exports -> A2/A3 ACT or Diffusion -> adapter-v1 -> closed-loop evaluation`

- The simulator runtime is the fixed-base IHMC Alex V2 torso in the single registered `AlexDoor-DoorPush-AlexV2-v0` environment.
- D0-D4 are the only accepted door poses. The runtime uses one calibrated six-joint, position-only right-arm controller and exact-door raw PhysX contact sensing.
- New recordings use `phase2.v2`. Existing `phase2.v1` episodes and legacy A4 records remain readable, but the repository does not write them.
- A1 is export-only. A2 and A3 support scripted execution and state-only learned policies. A4 is exported and adapter-executable but has no learned policy.
- ACT and Diffusion share dataset, configuration, checkpoint, run-allocation, and closed-loop reporting primitives while retaining separate models and training logic.

See [[topics/system-architecture|System Architecture]] for the current data flow and [[topics/alex-v2-benchmark|Alex V2 Benchmark]] for the simulator contract.

## Maintained Entry Points

`scripts/check_env.py` checks the supported workstation, Isaac, and external Alex dependencies. The maintained behavior gates are:

- `scripts/verify_benchmark_scene.py`
- `scripts/verify_scripted_baseline.py`
- `scripts/verify_dataset_interface.py`
- `scripts/verify_adapters.py`
- `scripts/verify_policy_rollout.py`

Generation, training, and evaluation use `scripts/run_scripted_baseline.py`, `scripts/train_policy.py`, and `scripts/eval_policy.py`.

## Active Configuration and Storage

The complete active `configs/` surface is `alex_v2_door.json`, `scripted_baseline.yaml`, `act.yaml`, and `diffusion.yaml`.

- `datasets/` stores reusable task/action-space/version exports, shared split files, retained views, and normalization artifacts.
- `outputs/door_scene/` stores exactly the five canonical D0-D4 layers.
- `outputs/door_push_alex_v2/{act,diffusion}/` is reserved for learned-policy runs.
- `outputs/wandb/` exists only when optional W&B tracking is enabled.
- Verification evidence, arbitrary scenes, and scripted-run staging belong under `~/.cache/alexdoor-xas/`.

The repository does not maintain run-specific cluster packages, dataset-construction workspaces, or historical result bundles in the active output tree.

## Historical Results

The completed scale study used 550 matched A2/A3 episodes across D0-D4 with nested N50, N100, N250, and N500 training views. Sixteen ACT/Diffusion x A2/A3 x data-size cells were evaluated over 576 rollouts. Every rollout succeeded, so the benchmark did not select a policy family, representation, or dataset size.

One ACT-A3-N50 rollout at seed 112 produced a reproducible 219.95 N one-tick peak. Two +/-1 mm door-position perturbations reduced the peak, but the original cell remains `REVIEW_REQUIRED`.

These are historical scientific conclusions, not active workflows. See [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] and [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]. Git retains removed implementation and evidence files.

## Retired Surface

The repository no longer maintains surrogate robots, generic door-task layers, sensorless execution, multi-environment runtime support, calibration authoring, scale-dataset construction, cluster/Slurm transfer, pilot and sweep orchestration, smoke or unified matrix runners, or compatibility shims for removed source APIs.

One deterministic fake environment remains for software tests. It mirrors the production state contract but is not a supported simulator runtime or physics result.

## Boundaries

- Phase 4 and VLA work have not started.
- Learned policies are state-only; image and language inputs are absent.
- Results cover one simulated door family and seed-0 training.
- Simulator success and force measurements do not establish hardware safety, sim-to-real readiness, or broader generalization.
- No repository command controls a physical Alex robot.

Any new dataset, benchmark, seed study, learned A4/VLA path, physical-robot work, or sim-to-real work requires a separately authorized scope.

## Version Notes

- 2026-08-13 — Reconciled the wiki with the simplified current repository and separated maintained behavior from concise historical evidence.
