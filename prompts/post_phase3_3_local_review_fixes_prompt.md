# TASK: Fix Phase 3.3 and local post-Phase 3.3 review findings

Before doing anything, read `CLAUDE.md` and `docs/PROJECT_GUIDELINES.md`. Treat
them as binding project guidance. Then invoke and follow the
`robotics-implementation` skill at
`/home/pacquadr/.codex/skills/robotics-implementation/SKILL.md`.

This file is the approved implementation specification for the task. Do not
expand its scope while implementing it. Write or update focused acceptance
tests before changing production behavior, then implement only what is needed
to make those tests and the repository gates pass.

The task covers:

- the Phase 3.3 Diffusion Policy implementation;
- the local post-Phase 3.3 stabilization pipeline and its 50-episode
  `door_push_alex_v2/v2_pose` artifacts;
- the correctness, safety, provenance, split, evaluation, and reproducibility
  defects listed below.

This task does **not** include the cluster sweep, the **Unified Phase 3 scientific
evaluation**, Phase 4, VLA, RL, WAM-lite, hardware transfer, or new scientific
claims.

## Mandatory workflow

Use the `robotics-implementation` workflow exactly:

1. Inspect `git status`, the current branch, unpublished history, and the live
   artifacts before editing. Preserve unrelated user work.
2. Treat this prompt as the approved spec. Map every acceptance criterion to
   one or more tests before implementation.
3. Use independent subagents for:
   - dataset split/leakage and artifact-regeneration review;
   - rollout lifecycle, timing, force, and safety review;
   - checkpoint/evaluation provenance and summary-gate review;
   - final diff and acceptance-test review.
4. Keep all edits owned by the main agent. Subagents must not make commits.
5. Implement in small validated blocks and make concise commits only after the
   focused tests for that block pass.
6. After implementation, invoke `robotics-review` for an independent final
   review against this prompt. Resolve verified blocking findings before
   declaring completion.

Use an existing directly related implementation branch if appropriate;
otherwise create a concise branch such as
`impl/phase3-3-local-review-fixes`. Do not reset or overwrite unrelated work.

## Current blocking findings

The following findings are already validated and must all be addressed. Do not
spend the task debating whether they exist.

### 1. Train/test leakage and missing pose coverage

The current shared split is made by shuffling episode IDs without grouping
equivalent trajectories. In `v2_pose`, three of the six test episodes are
numeric duplicates of training episodes:

- D0 seeds 0 and 1;
- D1 seeds 10 and 11;
- D2 seeds 20 and 21.

All 33 numeric HDF5 datasets in each pair match with maximum absolute
difference `0.0`. The current test set also has no D3 or D4 episodes.

Implement a deterministic split contract that:

- groups content-equivalent trajectories so a group can never cross
  train/validation/test boundaries;
- detects equivalence from stable trajectory content, not from a hard-coded
  seed-pair rule;
- excludes episode IDs, file paths, creation times, split labels, and other
  provenance-only fields from the equivalence identity;
- is pose-stratified so every configured pose appears in validation and test
  whenever the dataset has enough independent groups;
- preserves disjoint, exhaustive, deterministic splits shared by A1-A4;
- keeps the configured split fractions/sizes as closely as feasible without
  breaking grouping or pose coverage;
- fails loudly with a useful explanation when grouping and requested coverage
  cannot both be satisfied;
- records enough split metadata to audit grouping strategy, seed, pose counts,
  group counts, and any fraction deviation.

Do not merely reshuffle until the present dataset happens to look acceptable.
The generic split implementation and its tests must enforce these invariants.

### 2. Success timing is chunk-boundary dependent

The current success wrapper checks the hinge threshold only when a new policy
chunk is requested. ACT can therefore execute up to 39 post-success ticks with
H=40, while Diffusion can execute up to 7 with Ta=8.

Change the shared rollout path so that:

- the hinge threshold is checked after every executed control tick;
- `first_success_tick` and `time_to_success_s` mean the first threshold
  crossing, independent of policy chunk size;
- the rollout stops immediately on success unless an explicitly named
  diagnostic mode requests post-success execution;
- final-state metrics and first-crossing metrics remain unambiguous;
- a cross-then-rebound trajectory cannot be mislabeled as never successful;
- ACT and Diffusion use the same termination semantics.

Preserve the policy chunking behavior itself; only execution termination and
metric semantics should change.

### 3. Learned-policy rollouts ignore truncation and auto-reset

ACT and Diffusion use a 600-tick evaluation budget, equal to the Alex V2
environment episode budget. The DirectRLEnv can auto-reset inside `env.step`,
but the learned rollout driver currently ignores step termination/truncation
and may report post-reset state.

Make learned rollouts fail loudly or terminate with an explicit truncation
result before post-reset state can contaminate:

- final door angle and EE state;
- contact/force histories;
- success and failure labels;
- adapter counts and warnings.

Use the actual `env.step` result where supported and retain an episode-counter
guard analogous to the data-engine guard as a defensive invariant. Record an
explicit termination reason in every rollout row.

### 4. Evaluation provenance is not bound to the exact trained dataset

The current eval payload reports the shared source-manifest fingerprint, while
checkpoints embed an action-space-specific dataset fingerprint. A2 and A3 have
different exact content fingerprints even though they share the same source
episodes.

Extend provenance without losing existing fields:

- retain the shared source fingerprint as a separately named field;
- record the checkpoint-embedded exact dataset/content fingerprint;
- compute/read the current exact fingerprint for the selected action-space
  dataset;
- fail evaluation if the checkpoint fingerprint and live dataset fingerprint
  differ;
- validate that checkpoint train/validation IDs match the live split contract;
- keep provenance self-contained enough to audit an eval JSON after the
  dataset directory has changed.

Do not silently reinterpret the old source fingerprint as the exact dataset
fingerprint.

### 5. Smoke-summary coverage accepts inconsistent per-pose files

Strengthen `scripts/summarize_smoke_eval.py` so `metadata_coverage: PASS`
requires semantic consistency, not only field presence.

For every primary D0-D4 file in one run, verify at minimum:

- checkpoint path and a stable checkpoint identity/hash;
- action space and observation preset;
- exact dataset fingerprint and shared source fingerprint;
- split identity and train/validation/test membership;
- success threshold, max ticks, control timestep, and termination semantics;
- seed protocol and fixed/randomized counts expected for that pose;
- ACT chunk/ensemble settings, or Diffusion sampler, inference steps, Tp and Ta;
- top-level policy metadata agrees with every rollout row;
- door-pose payload agrees with the expected pose plan;
- no duplicate seeds or duplicate primary pose files;
- diagnostics remain separate from the main matrix.

Any mismatch must produce `metadata_coverage: FAIL`, list the exact files and
fields that disagree, and return a non-zero exit code. Add focused negative
fixtures for every invariant family.

### 6. Fixed determinism evidence does not repeat the same seed

Replace the misleading determinism contract:

- use `fixed_reset` or similarly precise language for rollouts that disable
  environment randomization but use different seeds;
- report across-seed policy/output spread separately;
- add a real repeat-same-seed determinism probe using the same environment
  reset seed, policy sampling seed, pose, checkpoint, and configuration;
- compare at least action/adapted-command traces, success crossing tick, final
  state, adapter decisions, and force/contact traces within explicit
  tolerances;
- store repeat count, tolerances, trace hashes, and pass/fail status in the eval
  artifact;
- make the summary gate require genuine same-seed evidence for every primary
  run.

Do not claim simulator or policy determinism from a spread across distinct
seeds.

### 7. Terminal force response can escape dataset admission

The data engine records the pre-action observation/force and may break on the
next controller query without persisting the response to the final executed
action.

Preserve the established observation/action schema, but add an explicit
terminal/post-action safety sample or equivalent additive structure so that:

- every executed action has its resulting force/contact response checked;
- the 0-200 N dataset admission bound covers the response to the final action;
- non-finite or over-bound terminal force aborts export;
- existing v0/`phase2.v1` datasets remain readable;
- no action/observation temporal alignment is silently shifted.

Document the exact time alignment of pre-action samples, actions, and terminal
post-action safety data.

### 8. Readiness status and safety evidence are conflated

The current smoke summary can report metadata coverage `PASS` while runs still
contain force-bound exceedances and many joint-velocity warnings. Separate:

- schema/metadata coverage;
- protocol consistency;
- safety/readiness disposition.

Do not automatically call every warning unsafe, but require a machine-readable
readiness result such as `PASS`, `REVIEW_REQUIRED`, or `FAIL`, with explicit
reasons and counts. Unsafe/invalid warnings or systematic adapter rejections
must fail. Unresolved force-bound exceedances must at least produce
`REVIEW_REQUIRED` and cannot be hidden by a metadata `PASS`.

Correct the local report's peak-force range from the artifacts; it currently
omits the observed approximately 272.2 N maximum. Do not weaken the 200 N
dataset-admission bound to make learned-policy evals look clean.

### 9. Requested randomized start offsets need a realized-state postcondition

`set_proxy_pose()` currently has a bounded IK settle loop but no explicit
postcondition proving that the requested start pose was realized.

Add a fail-closed postcondition and provenance:

- measure final position residual, and orientation residual if orientation is
  part of the request;
- compare against explicit, documented tolerances;
- abort or reject the episode/evaluation pose when the residual is excessive;
- record requested pose, realized pose, residual, settle ticks, and result;
- ensure pose-stratified dataset/eval claims refer to realized poses, not only
  requested configuration values.

Do not loosen the reachability, IK, joint-limit, force, or adapter gates.

## Required tests and acceptance criteria

Add or extend tests that prove all of the following.

### Split and dataset tests

- Exact duplicate trajectories cannot cross splits even when episode IDs and
  seeds differ.
- Near-but-not-identical trajectories are not collapsed accidentally.
- A five-pose, ten-episode-per-pose fixture gives validation and test coverage
  for D0-D4.
- Split generation is deterministic for the same inputs and seed.
- Impossible grouping/coverage requests fail loudly.
- Split metadata records group and per-pose counts.
- The real regenerated `v2_pose` split has no cross-split content duplicates
  and contains D0-D4 in both validation and test.

### Rollout tests

- First success is captured on the exact control tick for both H=40 and Ta=8.
- Cross-then-rebound remains a success with the original crossing tick.
- A full-budget truncation cannot report post-reset state.
- Termination reason distinguishes success, policy exhaustion, rejection,
  truncation, and tick-budget timeout.
- ACT and Diffusion rows expose the same timing and termination fields.

### Provenance and summary tests

- A2 and A3 retain different exact dataset fingerprints while sharing the same
  source fingerprint.
- Checkpoint/live dataset fingerprint mismatch fails evaluation.
- Changed split membership fails evaluation.
- The summary rejects mixed checkpoints, fingerprints, splits, thresholds,
  tick budgets, poses, samplers, horizons, or row/top metadata.
- Diagnostics cannot enter primary aggregates.
- Genuine repeat-same-seed evidence is required and reproducible.

### Safety tests

- A force spike caused by the final executed action fails dataset admission.
- Terminal force/contact alignment is explicit and backward compatible.
- Readiness becomes `REVIEW_REQUIRED` or `FAIL` for the configured warning and
  force cases while metadata coverage can remain independently reported.
- Excessive IK settle residual fails closed and is recorded.

## Artifact regeneration

After the implementation and focused tests pass:

1. Regenerate the shared `v2_pose` split with grouping and pose stratification.
2. Regenerate A1-A4 norm stats and validate their fingerprints.
3. Re-run the 50-episode dataset/interface and A2/A3-distinctness gates.
4. Retrain the four affected local smoke checkpoints from the corrected split:
   - ACT-A2;
   - ACT-A3;
   - Diffusion-A2;
   - Diffusion-A3.
5. Re-run matched D0-D4 local smoke evaluations with corrected per-tick success,
   truncation, provenance, determinism, and safety metadata.
6. Regenerate `outputs/local_smoke_n50/summary.json` and the local stabilization
   report.

Use CUDA for real training and policy inference when host-visible and compatible;
keep simulator device and official gate overrides consistent with the frozen
repo contract. Keep W&B disabled or offline unless the user explicitly requests
online tracking. Never store credentials or secrets.

Do not reuse old checkpoints as evidence after changing splits or norm stats.
Do not edit generated files by hand to make gates pass.

## Required validation

Use the official launcher for repository Python/Isaac checks:

```bash
ruff check .
git diff --check
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py --task door_push_alex_v2 --version v2_pose
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_a2_a3_distinct.py --task door_push_alex_v2 --version v2_pose
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_training.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_rollout.py --viz none --device cpu \
    --checkpoint-a2 <corrected-act-a2-best.pt> \
    --checkpoint-a3 <corrected-act-a3-best.pt>
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_diffusion_training.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_diffusion_rollout.py --viz none --device cpu \
    --checkpoint-a2 <corrected-diffusion-a2-best.pt> \
    --checkpoint-a3 <corrected-diffusion-a3-best.pt>
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_adapters.py --viz none --device cpu
```

Also run focused regression tests for split leakage, pose coverage, per-tick
success, truncation, terminal force, provenance, summary consistency,
determinism, and IK-settle postconditions. Report exact commands and results.

## Stop conditions

Stop and report instead of weakening or bypassing a gate if:

- any content-equivalent trajectory still crosses splits;
- D0-D4 cannot all be represented in validation and test;
- the corrected split invalidates artifacts that have not yet been regenerated;
- a checkpoint/live dataset or split mismatch is detected;
- learned rollouts can still consume post-reset state;
- first-success timing remains chunk-size dependent;
- terminal force response remains unchecked;
- repeat-same-seed determinism fails beyond the documented tolerance;
- IK start-pose residual exceeds tolerance;
- systematic adapter rejections or unsafe/invalid warnings occur;
- a frozen gate would need to be weakened;
- completion would require starting the cluster sweep or Unified Phase 3 scientific
  analysis.

## Constraints

- Phase 3.3 plus local stabilization fixes only.
- No 100/250/500 dataset sweep.
- No cluster job submission or cluster artifact claims.
- No ACT-vs-Diffusion or A2-vs-A3 performance conclusions.
- No Phase 4, VLA, RL, WAM-lite, or hardware-control work.
- No simulator-stack installation or upgrade.
- No hidden fallback to stale datasets, splits, checkpoints, or eval JSONs.
- No weakening safety thresholds, force admission, adapter checks, or frozen
  verification gates.
- Preserve backward readability of existing dataset/checkpoint formats through
  additive migration where necessary.
- Keep Hydra as the config/override layer and Isaac `AppLauncher` as the
  simulator entrypoint.
- Keep W&B opt-in and disabled/offline by default.
- Do not stage, commit, or alter unrelated user files.

## Commit guidance

Prefer separate concise commits after validated blocks, for example:

1. `Harden grouped pose-aware dataset splits`
2. `Fix learned rollout termination semantics`
3. `Bind eval provenance to checkpoint data`
4. `Strengthen smoke readiness validation`
5. `Regenerate local Phase 3.3 artifacts`

Do not commit failing work. Stage only intentional files, and do not add
Claude/Anthropic co-author trailers.

## Final report

Report:

1. Spec and guidance files read.
2. Branch and starting commit.
3. Files changed, grouped by work package.
4. New split algorithm and exact real `v2_pose` train/validation/test counts by
   pose and equivalence group.
5. Proof that no content duplicates cross splits.
6. New timing, termination, truncation, force, IK, determinism, and provenance
   semantics.
7. Corrected dataset fingerprints and split identity.
8. Regenerated checkpoint/run IDs and eval JSON paths.
9. Metadata coverage, protocol consistency, and safety/readiness results as
   separate statuses.
10. Adapter warnings, corrections, rejections, force exceedances, and IK
    residual summary without hiding any warning.
11. Exact validation commands and results.
12. Independent `robotics-review` findings and their resolution.
13. Commits created.
14. Remaining caveats and any explicit deviations from this spec.

Do not claim cluster readiness, scientific comparison readiness, or Unified Phase 3
completion unless every acceptance criterion above passes and the user
separately authorizes the next phase.
