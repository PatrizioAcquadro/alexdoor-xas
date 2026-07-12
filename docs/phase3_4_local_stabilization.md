# Local post-Phase 3.3 stabilization (smoke pass before the cluster sweep)

> Historical path note: this file retains its original filename to preserve
> local references. “Phase 3.4” is not a milestone name here. The later report
> is the **Unified Phase 3 scientific evaluation**.

**Status: COMPLETE; SAFETY READINESS PASS.** Phase 3.3 software and local
protocol validation pass. The primary matrix was regenerated after the
calibrated learned-policy contact-entry correction; schema, protocol, and safety
readiness now all pass. All results are **pipeline
validation on a 50-episode smoke dataset — not performance evidence**. No ACT-vs-Diffusion or
A2-vs-A3 claims are made or implied; `v2_pose` is the definitive local
stabilization contract (schema + generation protocol) that cluster/VLA/WAM
scaling inherits, nothing more.

## 1. Problem 1 — clean 50-episode generation (fixed earlier in this pass)

The diff-IK joint-limit windup blocker was fixed by the anti-windup clamp on
solved IK targets (executor-level, with mandatory pre-clamp telemetry: per-joint
max excess + clamp tick counts recorded into `extras["ik_clamp_telemetry"]`),
plus loud generation-time sanity enforcement in the runner (per-run
`metrics/sanity.json`, hard abort before export on any sanity error). The exact
previously-failing plan (10 fixed + 40 randomized, base seed 0) regenerates with
**0 sanity errors on all 50 episodes**; the historical windup seeds absorbed
0.28–0.41 rad of raw pre-clamp excess. `eval/sanity.py` thresholds unchanged.

Additional hardening landed at the end of this pass:
- `run_baseline` **refuses `export=True` for runs at a non-default door pose**
  (a posed re-export would replace the official default-pose version dir);
  multi-pose data flows only through `scripts/export_merged_dataset.py`.
- `run_baseline` rejects `max_ticks` above the env's episode budget, and
  `run_episode` hard-fails if an episode actually hits the env's mid-episode
  auto-reset (post-reset state can no longer be recorded silently).
- Run config provenance (`logs/run_config.json`) is written before the sanity
  gate so aborted runs keep their provenance.
- Zero-step Alex episodes now reach the sanity checker (identified via the
  joint-limit extras) instead of bypassing it.

## 2. Problem 2 — A2/A3 distinguishability (v2_pose)

`datasets/door_push_alex_v2/{A1,A2,A3,A4}/v2_pose`: 50 episodes, 5 probe-gated
door poses (`configs/door_pose_plan_v2_pose.json`; D0 default, D1–D4 = ±0.05 /
±0.10 rad yaw about the hinge axis + ≤2 cm world-XY offsets), 2 fixed + 8
randomized per pose in disjoint seed blocks, single-writer merge with
contract-grade `manifest.json` (pose plan + probe results, object metadata,
per-episode contact/safety + IK-clamp telemetry, `task_instruction` placeholder,
camera/calibration slots). The regenerated shared split is a deterministic,
content-hash-grouped, pose-stratified 38/6/6 split over 46 equivalence groups.
Equivalent trajectories never cross splits. Its exact per-pose episode counts
(train/validation/test) are D0 8/1/1, D1 8/1/1, D2 7/2/1, D3 8/1/1, and D4
7/1/2; group counts are D0–D3 9 each and D4 10. Requested and actual split
sizes match with zero fraction deviation. Split fingerprint:
`9842921abccfc4159da912ca69438d7b498c239c4beed1a3b097ac8be29497f5`.

The shared source fingerprint remains
`8278dbcacd9af372451ccd9940541e3fbbc422b9e65d82793062596eef7ccdb0`,
while exact action-space dataset fingerprints are deliberately distinct:
A2 `ba658fed2b49311373685c20c6a40045f6955bcccf11e6e5490cfbb753192ff3`
and A3 `00149827e4bd5e66446701ea92cddf7fdf3857a04a00ac55df3ea22ee5ebcfbb`.

`scripts/verify_a2_a3_distinct.py --task door_push_alex_v2 --version v2_pose`:
**PASS** — D0 exports identical; yawed poses differ by 7.5e-4 (±0.05 rad) /
1.5e-3 (±0.10 rad), exactly linear in yaw; conversion error 1.7e-18.

## 3. Eval-schema extensions (this pass)

The shared rollout driver now checks success after every executed control tick,
records the first threshold crossing, and stops immediately. ACT H=40 and
Diffusion Ta=8 therefore share `per_tick_first_crossing_stop` semantics. Every
row records an explicit termination reason; environment termination/truncation
and the defensive episode counter are checked before post-reset state can enter
angles, force/contact histories, adapter counts, or labels.

Per-rollout rows carry seed, fixed/randomized flag, requested and realized door
pose, first-success tick/time, final state, termination reason, contact/force
metrics, adapter decisions, warnings, and start-pose settle provenance. All 108
randomized primary rows passed the realized-position postcondition (maximum
residual 0.004983 m versus the 0.010 m tolerance, at most 10 settle ticks);
orientation is not part of these translation-only requests. Dataset episodes
also retain an additive terminal post-action safety sample, so the response to
the final executed action is checked without shifting the established
pre-action observation/action alignment.

Evaluation provenance retains the shared source fingerprint separately from
the checkpoint-embedded exact dataset fingerprint and the live exact dataset
fingerprint. Evaluation fails on exact-fingerprint or train/validation split
mismatch; missing companion training split provenance also fails rather than
silently skipping validation. Per-pose files remain pose-qualified, and the summary checks
checkpoint hash, dataset/split identity, thresholds, timing, seed protocol,
pose plan, policy metadata, row/top-level agreement, and diagnostic separation.

## 4. Smoke matrix (n=50, seed 0; GPU training + GPU policy inference; CPU sim per frozen gate contract)

Training (`dataset.version=v2_pose`, `dataset.obs_preset=core_door_pose`
(14-dim), `train.seed=0`, `train.device=cuda`, W&B **offline**, group
`local_smoke_n50_gpu`): run ids `local_smoke_act_a2_n50_seed0`,
`local_smoke_act_a3_n50_seed0`, `local_smoke_diffusion_a2_n50_seed0`,
`local_smoke_diffusion_a3_n50_seed0` under
`outputs/door_push_alex_v2/{act,diffusion}_door_push/`. ACT val L1 ≈
0.038/0.035 (A2/A3); a repeated ACT-A3 run reproduced bit-identical metrics
(seeded-training determinism).

Evaluation (matched protocol for all four checkpoints; success ≥45°, 600 max
ticks, per-tick success-stop, adapter-v1; D0 = 5 fixed + 15 randomized @ seed
100; D1–D4 = 1 fixed + 3 randomized @ seeds 200/210/220/230;
diffusion primary sampler **DDIM-10**, Tp=16/Ta=8):

The per-pose seeds and fixed/randomized counts are bound to the tracked
`configs/local_smoke_eval_plan_n50.json`; eval files cannot self-declare a
different or easier protocol.

| run | success | corrected commands/rejected | warnings | peak force | readiness |
|---|---|---|---|---|---|
| ACT-A2 | 36/36 | 296/0 | 160 | 135.8 N | PASS |
| ACT-A3 | 36/36 | 233/0 | 160 | 129.4 N | PASS |
| DP-A2 (DDIM-10) | 36/36 | 195/0 | 160 | 145.5 N | PASS |
| DP-A3 (DDIM-10) | 36/36 | 210/0 | 160 | 143.7 N | PASS |

The final corrected matrix has no 200 N dataset-admission-bound exceedance in
any cell. The two formerly blocking seed-112 Diffusion entries now peak at
116.4 N (A2) and 119.2 N (A3); their force-producing commands are recorded as
`corrected`, with requested commands preserved and applied translation bounded
to the calibrated 5 mm approach step. The bound remains unchanged. The pre-fix
local artifact set contained peaks of 230.7 N, 201.9 N, and an older observed
maximum of approximately 272.2 N; those artifacts are superseded, not hidden.
All warnings are
warn-level arm joint-velocity flags (0.4–2.3 rad/s over task-level caps on
wrist/elbow joints, same class the scripted baseline shows); reported verbatim
in every eval JSON, none suppressed. Across-seed fixed-reset output spread is
reported separately (maximum 0.01014 rad); it is not determinism evidence.

Genuine same-seed evidence is a fresh-process replay of the first fixed rollout
for each policy/pose file with identical reset seed, policy sampling seed,
checkpoint, pose, and configuration. All 20 probes passed trace-by-trace over
requested/applied commands, adapter statuses, success tick, termination, final
state, contact, and force within their stored tolerances.

**Diagnostic only** (excluded from the matrix): DDPM-100 on hard seed 105 @ D0
succeeded for both DP checkpoints (final 0.796 rad). The Phase 3.3 DDPM-100
seed-105 failure does not reproduce on the v2_pose-trained checkpoints; DDIM-10
remains the deployment default. This is a sampler-diagnostic data point, not a
comparison result.

Summary: `outputs/local_smoke_n50/summary.json` — **metadata coverage PASS**,
**protocol consistency PASS**, and **safety readiness PASS** across all 144
primary rollouts. Diagnostics remain separate from primary aggregates.

## 5. Validation battery (all PASS)

Fresh post-review-fix validation: ruff, `git diff --check`, pytest (537),
`verify_dataset_interface` (`v2_pose`, 50 episodes, 46 groups, D0–D4 in
validation and test), `verify_a2_a3_distinct`, `verify_act_training`,
`verify_diffusion_training`, `verify_act_rollout`, `verify_diffusion_rollout`,
and `verify_adapters`. The earlier stabilization battery also passed `check_env`,
`verify_assets`,
`verify_door_task_scene`, `verify_door_env`, `verify_scripted_baseline`,
`verify_alex_v2_door_baseline` (re-certified after the runner changes),
`verify_dataset_interface` (v0 proxy + v2_pose), `verify_a2_a3_distinct`,
`verify_adapters`, `verify_act_training` (now pins `train.device=cpu` like the
diffusion gate — GPU-first defaults), `verify_act_rollout` +
`verify_diffusion_rollout` (against the smoke checkpoints),
`verify_diffusion_training`.

## 6. Review trail

Five independent review passes: (1) data-engine/IK/safety (clamp verified
correct; its two moderate findings — posed-export guard, gate re-runs — are
fixed/done in this pass), (2) A2/A3 frame semantics (MERGE-READY; its minor
labeling findings — pose-id requirement, failure-label precedence,
contact-ticks unification, anchor-rotation validation — all fixed), (3) ACT
smoke (MERGE-READY; its force-admission-flag finding implemented, exceedances
now visible), (4) Diffusion smoke (MERGE-READY; summary pose-matrix gate +
duplicate rejection + non-diag metadata sampling implemented), (5) final
readiness (see commit).

The required final `robotics-review` then identified three material gate gaps:
incomplete determinism-evidence validation, optional validation-split binding,
and a self-declared per-pose seed protocol. All three were resolved with
negative regression tests; the repeated review found no remaining critical or
moderate correctness issue and recommends approval. No subagents were used, per
the task instruction.

## 7. Caveats carried forward

- Smoke-scale evidence only (50 episodes; 36 eval rollouts per cell).
- Sim runs on CPU per the frozen calibration/gate contract; GPU-sim physics
  would be a recalibration-level change (cluster-scope decision).
- Force sensing at eval is the door-panel-filtered EE force; impulse integrates
  |F|·control_dt over the whole rollout.
- Local smoke safety readiness is not hardware-readiness evidence; the unchanged
  200 N bound and calibrated contact-entry correction remain simulation-local.
