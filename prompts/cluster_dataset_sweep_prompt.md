# TASK: Cluster dataset-scale training and evaluation sweep

Before doing anything, read `PROJECT_GUIDELINES.md` from the repo root. If it is not present, read `docs/PROJECT_GUIDELINES.md`. Treat it as binding project guidance.

This task happens only after the local post-Phase 3.3 stabilization task has passed and been committed.

Do not start Phase 4, VLA, RL, WAM-lite, hardware-transfer work, post-processing comparison, or final scientific analysis.

The goal is to use the cluster to generate/train/evaluate the full dataset-size matrix for ACT and Diffusion Policy. Produce clean artifacts for later comparison, but do not perform the comparison in this task.

Use robotics workflows and subagents where useful:

- one subagent for cluster environment / reproducibility checks;
- one subagent for dataset-generation job planning;
- one subagent for ACT training/evaluation jobs;
- one subagent for Diffusion Policy training/evaluation jobs;
- one subagent for Hydra/W&B run-configuration review;
- one subagent for final artifact/readiness review.

## Cluster Assumptions

The cluster has 8 A100 GPUs.

Use the GPUs primarily for training and parallel sweeps.

Do not assume the A100s automatically fix Isaac/data-engine issues. If Isaac Sim / Isaac Lab headless simulation is not stable on the cluster, do not force dataset generation there. In that case:

- generate datasets locally or on the validated machine;
- transfer validated datasets to the cluster;
- use the cluster for training and evaluation only where supported.

Before launching the full sweep, verify:

- repo commit hash matches the local stabilized commit;
- dependencies are installed;
- Isaac/IsaacLab launcher works if dataset generation or closed-loop eval will run on the cluster;
- dataset paths are correct;
- output paths are writable;
- W&B credentials/network are available if online tracking will be used;
- W&B behavior is explicit through Hydra overrides, never implicit or hard-coded;
- GPU visibility works for training.

## Dataset Sizes

Run the full sweep for:

- 50 episodes
- 100 episodes
- 250 episodes
- 500 episodes

Do not run 1000+ episodes unless the user explicitly approves it later.

## Action Spaces

For every dataset size, cover:

- A2: `A2_ee_delta`
- A3: `A3_obj_rel_ee_delta`

A2 and A3 must use matched seed plans and matched door-pose/orientation variation.

## Policies

For every dataset size and action space, train and evaluate:

- ACT
- Diffusion Policy

Expected matrix:

| dataset size | ACT-A2 | ACT-A3 | Diffusion-A2 | Diffusion-A3 |
|---:|---|---|---|---|
| 50 | train + eval | train + eval | train + eval | train + eval |
| 100 | train + eval | train + eval | train + eval | train + eval |
| 250 | train + eval | train + eval | train + eval | train + eval |
| 500 | train + eval | train + eval | train + eval | train + eval |

## GPU Scheduling Guidance

Prefer one GPU per training job.

Do not add distributed multi-GPU training unless the repo already supports it cleanly and it is validated.

A reasonable first cluster schedule is:

```text
GPU 0: Diffusion-A2-50
GPU 1: Diffusion-A3-50
GPU 2: Diffusion-A2-100
GPU 3: Diffusion-A3-100
GPU 4: Diffusion-A2-250
GPU 5: Diffusion-A3-250
GPU 6: Diffusion-A2-500
GPU 7: Diffusion-A3-500
```

ACT jobs are smaller and may be run before/after the Diffusion jobs, or in parallel where resources allow.

## Hydra and W&B Requirements

Use the repo's Hydra/OmegaConf config pattern for all dataset-generation, training, and evaluation jobs.

Do not hard-code dataset size, action space, seed, checkpoint path, device, run id, or W&B settings inside scripts unless the repo already requires it. Prefer Hydra overrides such as:

```bash
dataset.space=A2_ee_delta
dataset.version=<dataset_version>
train.seed=0
train.device=cuda
run.run_id=<clear_run_id>
+wandb.mode=online
+wandb.project=alexdoor-xas
+wandb.group=<sweep_group>
+wandb.name=<run_name>
```

For cluster runs:

- use W&B online where credentials and network are available, especially for long ACT/Diffusion training sweeps;
- keep W&B explicit through Hydra overrides, never implicit or hard-coded;
- each training run should log losses, validation metrics, config, dataset fingerprint, checkpoint paths, and run id;
- each evaluation run should log success rate, final door angle, ticks/time to success, adapter accepted/corrected/rejected counts, adapter warning counts, eval JSON path, and dataset fingerprint;
- record W&B run URLs or run ids in the final report;
- never store W&B credentials, API keys, `.netrc` contents, or secrets in the repo.

If W&B online is unavailable on the cluster:

- do not fail the entire task only for missing W&B connectivity;
- fall back to offline/local logging under the repo's established `outputs/wandb` or run-output paths;
- clearly report that online tracking was unavailable and include the reason;
- keep all local/offline artifacts sufficient for later comparison.

Hydra must remain a config/override layer, not a replacement for Isaac `AppLauncher`. Keep all Isaac-backed commands using the canonical launcher pattern:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script> <hydra_overrides>
```

## Dataset Generation

For each dataset size:

- generate or load the validated dataset;
- regenerate splits and norm stats;
- verify dataset interface;
- record dataset fingerprint;
- record seed range and variation protocol;
- store dataset manifests.

If generation fails at any size:

- stop at that size;
- report exact failing seeds;
- report failure type;
- do not continue to train on partial or dirty data unless explicitly marked as a failed/debug artifact.

## Training Requirements

For every policy/action-space/dataset-size combination:

- use deterministic seeds;
- record full config;
- record run id;
- record dataset fingerprint;
- save best and last checkpoints where supported;
- save train/val logs;
- save open-loop diagnostics where supported;
- use W&B online when available and useful for the cluster sweep;
- fall back to offline/disabled W&B only when online tracking is unavailable or inappropriate.

Use clear run ids, for example:

```text
act_a2_n50_seed0
act_a3_n50_seed0
diffusion_a2_n50_seed0
diffusion_a3_n50_seed0
...
diffusion_a3_n500_seed0
```

## Evaluation Requirements

For every trained checkpoint, run the same matched evaluation protocol:

- same fixed seeds;
- same randomized seeds;
- same door-pose variation plan;
- same success threshold;
- same max ticks;
- same adapter-v1 path;
- same warning/correction/rejection reporting.

Each evaluation output must include:

- success rate;
- final door angle;
- time/ticks to success;
- per-rollout rows with seed, fixed/randomized flag, door-pose/orientation id, success flag, final angle, ticks, failure label, and notes;
- contact ticks and contact source;
- sensed-force summaries when available, at minimum mean/max/p95 contact force and impulse; if unavailable, record why;
- adapter accepted/corrected/rejected counts;
- adapter warning counts and warning messages;
- fixed-seed determinism evidence;
- run config;
- dataset id/fingerprint;
- checkpoint path;
- train/val/test split metadata;
- policy-specific evaluation metadata, including ACT chunk horizon/settings or Diffusion sampler, inference steps, Tp/Ta horizon, and checkpoint horizon;
- W&B run id/URL when online tracking is enabled.

For Diffusion Policy, keep sampler and horizon effects separable from action-space effects:

- choose one primary deployment evaluation setting before running the matrix, based on the validated local stabilization result;
- record the sampler (`ddpm` or `ddim`), number of inference steps, `model.horizon`, and `rollout.n_action_steps` for every run;
- do not mix sampler/horizon settings inside the main ACT-vs-Diffusion or A2-vs-A3 matrix;
- if compute allows, run a small diagnostic-only Diffusion panel on the hardest known seeds or a fixed stress subset with alternate sampler/horizon settings, and label those outputs as diagnostics rather than primary comparison artifacts.

## Outputs

Produce structured outputs for every dataset size and run:

- dataset manifest;
- split file;
- norm stats;
- dataset verification report;
- ACT run directories;
- Diffusion Policy run directories;
- checkpoint paths;
- train logs;
- open-loop reports;
- closed-loop eval JSON files;
- per-rollout normalized metrics tables suitable for Unified Phase 3 scientific evaluation;
- policy metadata tables covering sampler/horizon/chunk settings;
- W&B run ids/URLs when online tracking is enabled;
- curated summary JSON/MD under `outputs/curated/`.

The curated summary should organize artifacts and metrics only.

Allowed:

- "Dataset size X generated successfully."
- "Training completed."
- "Evaluation completed."
- "This run produced N warnings / corrections / rejections."
- "W&B run id/URL: ..."

Not allowed:

- "A3 is better than A2."
- "Diffusion is better than ACT."
- "This proves generalization."
- "This is ready for VLA."
- Any final comparison or scientific conclusion.

## Required Validation

Run and report before or after the sweep as appropriate:

```bash
ruff check .
git diff --check
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_training.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_rollout.py --viz none --device cpu
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_adapters.py --viz none --device cpu
```

Also run all Phase 3.3 Diffusion Policy gates that exist after Phase 3.3 is implemented.

If some Isaac closed-loop validation cannot run on the cluster, report why and run it on the validated local machine instead. Do not silently skip it.

## Constraints

- This task is cluster execution only after local stabilization.
- Do not perform post-processing comparison or final scientific analysis.
- Do not start Phase 4/VLA/RL/WAM-lite work.
- Do not implement hardware-control code.
- Do not weaken frozen gates.
- Use W&B online for cluster training/evaluation sweeps when credentials and network are available.
- Keep W&B explicit through Hydra overrides, never implicit or hard-coded.
- Keep Isaac entrypoints using the canonical launcher pattern.
- Preserve adapter warning reporting.
- Do not hide warnings; reduce them only through real behavior/safety improvements.
- Keep all runs deterministic and reproducible.
- Do not train on dirty or partially failed datasets.
- Commit only code/config/script changes that are required and validated.
- Large generated datasets/checkpoints should remain in the repo's established artifact/output locations, not blindly committed.

## Stop Conditions

Stop and report instead of pushing forward if:

- cluster environment differs enough to break determinism or Isaac behavior;
- 50 episodes fail despite passing locally;
- any larger dataset size fails generation;
- A2 and A3 become numerically identical again;
- primary matrix runs use inconsistent Diffusion sampler/horizon settings;
- per-rollout evaluation outputs are missing seed, variation, failure-label, contact, adapter, or policy-metadata fields needed for Unified Phase 3 scientific evaluation;
- systematic adapter rejections appear;
- warnings indicate unsafe or invalid behavior;
- training succeeds only because evaluation was weakened;
- any frozen gate would need to be weakened;
- output artifacts cannot be reliably stored or recovered;
- W&B online is unavailable and no offline/local artifact path can preserve the run evidence.

## Final Report

Provide:

1. Cluster machine/environment summary.
2. Repo commit hash used.
3. Dataset sizes completed.
4. Seed ranges and variation protocols.
5. Dataset fingerprints/manifests.
6. Training run ids for every ACT and Diffusion run.
7. Checkpoint paths.
8. Evaluation JSON paths.
9. Per-rollout normalized metrics table paths.
10. Policy metadata table paths, including Diffusion sampler/horizon settings.
11. W&B run ids/URLs where online tracking was used.
12. Adapter warning/correction/rejection summary for every run.
13. Failed runs, if any, with exact reason.
14. Exact validation commands and results.
15. Remaining caveats.
16. Commit hash, if code/config changes were made.

Do not perform post-processing comparison or scientific interpretation.
