# TASK: Fix Full-Sweep Readiness Locally on Ubuntu

## Objective

Fix every confirmed blocker from the read-only review of the completed nested dataset-scale sweep
implementation. Work only in the Ubuntu checkout and finish with a clean, committed, independently
verified transfer package that is safe to hand to Gilbreth later.

Do **not** regenerate the 550 episodes unless a corrected verifier proves the published dataset is
actually invalid. The current real dataset, views, A2/A3 conversion, and eight normalization values
were independently verified as correct. This task fixes fail-open contracts, provenance, environment
selection, return packaging, and missing integration coverage.

Do **not** push, merge, transfer files, connect to Gilbreth, request an allocation, call `sbatch`, run
the full sweep, start closed-loop evaluation, or begin Phase 4.

Use the `robotics-test` workflow first to add regressions that reproduce the defects, then use
`robotics-implementation` to make the smallest fixes. Replace the completed local-only `TODO.md`
with a checklist for this fix task, keep it ignored, and update it as each gate passes.

## Verified Starting State

- Repository: `/home/pacquadr/Desktop/DoorManipulation`
- Branch: `impl/full-cluster-dataset-scale-sweep`
- Reviewed implementation commit: `bfa79f2ef866cb20b7cbf46a360c48e1f58021aa`
- Review baseline: `origin/main` at `252fb91d370ddeb5e391306aed6e504e2dbf710a`
- Expected only initial tracked change: this prompt file.
- Baseline full suite: 701 passed, with two existing IsaacLab deprecation warnings.
- Real transfer manifest before fixes:
  - SHA-256: `41bb2659519ce51b82af768917e56c64c3ee0c1e5a217236b1949b9e4328cc81`
  - 2,292 payload files
  - 235,585,511 bytes
  - 2,294 rsync paths
- Real source-master fingerprint:
  `79dd3e819c2fbb2d21b9cf3848df7942bd6e69f163a9d82ef892710f5e39d27b`
- Real action-export fingerprints:
  - A2: `5179595aeeb7a69898456750658feb59b6a2aaff1821f6ce8af36739f04f9e06`
  - A3: `3b528ceb451e9547c2aec76d47dafbf7129f870bdfaa6c691769304487ca615f`
- Proven Gilbreth environment prefix:
  `envs/alexdoor-gilbreth-pilot-py311`

Before editing, read completely:

- `CLAUDE.md`
- `docs/PROJECT_GUIDELINES.md`
- `TODO.md`
- this prompt
- all files named in the findings below
- the related ACT, Diffusion, dataset-view, normalization, pilot-return, sweep-return, Slurm,
  transfer, and checkpoint tests.

If this prompt is the only worktree change, review it and commit it alone on the current branch with
the message `Define full sweep readiness fixes`. If any unrelated change exists, preserve it and
stop if it overlaps this task. Do not reset, clean, or discard user work.

## Required Fixes

### 1. Bind every returned checkpoint to the exact configured sweep cell

The current sweep return verifier checks dataset provenance but does not prove that a returned
checkpoint used the intended training configuration.

Fix the contract so each of the 16 returned cells is validated against its exact expected resolved
configuration, including:

- policy and stable run ID;
- dataset task, physical version, view ID, action space, and `core_door_pose`;
- seed `0`;
- `train.device=cuda`;
- `train.overfit_episodes=null`;
- ACT epochs `100` and validation cadence `5`;
- Diffusion epochs `300`, validation cadence `10`, EMA enabled, and 10 validation inference steps;
- W&B offline mode;
- all other resolved model, optimizer, horizon, chunk, and scheduler values inherited from the
  committed base policy configuration.

For each returned cell:

1. Parse its durable `resolved_config.json`.
2. Canonicalize and SHA-256 hash it using the same algorithm used during training.
3. Require that hash to equal checkpoint provenance `resolved_training_config_sha256`.
4. Require the loaded checkpoint configuration to equal the durable resolved configuration.
5. Reconstruct the expected cell configuration through the normal config/override path and require
   exact equality for all scientifically relevant fields, not only the dataset block.
6. Reject wrong epochs, validation cadence, EMA, inference steps, seed, device, overfit value,
   policy, run ID, W&B mode, or resolved-config content.

Do not create a second permissive validator. Keep one authoritative cell-configuration contract
shared by rendering, training, and return verification.

### 2. Separate source-master and action-export fingerprints

The current checkpoint field named `master_dataset_fingerprint_sha256` contains the action-export
fingerprint from normalization instead of the common 550-source fingerprint.

Correct the provenance model:

- carry the view's real `master_dataset_fingerprint_sha256` through `PolicyData`;
- validate it against the physical dataset manifest's `source_fingerprint_sha256`;
- embed it in every view-selected ACT and Diffusion checkpoint as
  `master_dataset_fingerprint_sha256`;
- add a separate `action_dataset_fingerprint_sha256` containing the A2 or A3 export fingerprint;
- validate both fields during checkpoint load, evaluation metadata construction, sweep return
  verification, and any other scale-checkpoint compatibility gate;
- preserve loading of existing `v2_pose` and compatibility-pilot checkpoints that do not use a
  scale view.

Use a backward-compatible provenance schema evolution. Do not silently reinterpret historical
checkpoint fields.

Required real values are:

- source master: `79dd3e819c2fbb2d21b9cf3848df7942bd6e69f163a9d82ef892710f5e39d27b`;
- A2 export: `5179595aeeb7a69898456750658feb59b6a2aaff1821f6ce8af36739f04f9e06`;
- A3 export: `3b528ceb451e9547c2aec76d47dafbf7129f870bdfaa6c691769304487ca615f`.

### 3. Recompute normalization values from the declared training episodes

The existing validator accepts numerically wrong means/std/min/max when their self-fingerprint and
outer manifest hash are refreshed.

Make numerical correctness authoritative:

- recompute action and observation count/mean/std/min/max from exactly the view's declared training
  IDs;
- compare every recomputed value with the stored artifact;
- run this comparison when publication encounters an existing normalization file;
- run it in local scale-dataset verification;
- run it while reconstructing/building the sweep transfer contract;
- ensure the committed verifier on Gilbreth can validate the transferred artifact against the
  transferred dataset without Isaac;
- retain validation of action space, observation preset, dataset IDs/fingerprint, view ID and
  fingerprint, exact train IDs, normalization fingerprint, and file SHA-256.

The comparison must reject a modified numeric value even if every self-hash and manifest hash is
recomputed. Use deterministic recomputation and explicit finite/count/shape checks. Validation and
test episodes must remain excluded.

### 4. Make generation provenance machine-consistent and fail closed

The checked-in seed namespace descriptions do not match the actual pose plan. The current dataset
verifier also does not validate the complete candidate ledger or bind the live pose plan and
calibration to the published master.

Create one authoritative, machine-readable generation contract:

- remove or replace the misleading namespace formulas in `configs/cluster_sweep.v1.json`;
- encode or reference the exact per-pose source and overdraw ranges from
  `configs/door_pose_plan_v3_scale.json` without maintaining contradictory duplicate truth;
- require exact non-overlapping seed inventories for D0-D4;
- bind the tracked pose-plan path and SHA-256 to the master manifest and transfer contract;
- validate the calibration fingerprint against the canonical Alex V2 calibration artifact;
- validate D0-D4 yaw/offset geometry against the already validated canonical pose definition;
- reject changed seed ranges, plan hash, calibration fingerprint, or geometry even when counts still
  look valid.

Validate the complete candidate-provenance ledger locally on Ubuntu:

- exactly 750 unique pose/seed candidates for the current artifact;
- exact source and overdraw membership per pose;
- every selected, skipped, failed, replacement, and not-needed decision has valid reasons and
  deterministic replacement linkage;
- selected episode IDs and pose IDs match the paired exports;
- selected content-group hashes match the raw generated candidates;
- selected source paths exist during Ubuntu publication/verification;
- the recomputed selected-source fingerprint matches the master manifest;
- all 550 selected episodes are safe, successful, content-distinct, and balanced 110 per pose.

The cluster verifier cannot depend on Ubuntu-only raw candidate paths. Bind the verified Ubuntu
ledger/report into the transfer manifest by hash and verify all transferable invariants from the
received files. Do not weaken the clean-tree or exact-inventory gates.

The existing real ledger is expected to remain:

- 550 `SELECTED` source candidates;
- 200 `NOT_NEEDED_OVERDRAW` candidates;
- no replacements required.

If the corrected contract matches these artifacts, do not regenerate HDF5 files or change episode
IDs, views, holdouts, or normalization values.

### 5. Make the sweep return package directly transferable and verifiable

Bring sweep return packaging to parity with the proven pilot contract.

The written `return-files.txt` must contain exactly once:

- every manifest payload path;
- `.sweep_return/attempts/<ARRAY_JOB_ID>/return_manifest.json`;
- `.sweep_return/attempts/<ARRAY_JOB_ID>/return-files.txt`.

The checksum rsync command must use that remote file list and transfer the controls to the Ubuntu
return root together with the 16-cell payload. After transfer, the documented Ubuntu verifier must
find the manifest at the transferred path and verify all hashes plus all 16 CPU checkpoint loads.

Reject malformed, missing, duplicated, mixed-attempt, or self-inconsistent control paths. Do not
include the rsync command itself unless the existing pilot contract requires it.

### 6. Reuse the proven Gilbreth environment path

Do not create an unsupported second environment implicitly.

- Change the sweep contract to use `envs/alexdoor-gilbreth-pilot-py311`, the environment proven by
  pilot and canary jobs.
- Ensure the renderer, preflight, documentation, example Slurm script, and transfer metadata all
  agree on this prefix.
- Bind and preflight the expected non-Isaac runtime evidence: Python 3.11, NumPy 2.4.6, PyTorch
  2.12.1+cu126, Torch CUDA 12.6, and no Isaac imports.
- Reuse the existing explicit prefix-Python launcher pattern; never activate Conda or depend on
  inherited Python/Ruff shims.
- If a bootstrap/update path is retained, it must target the same configured prefix and remain
  credential-free. Do not require recreating the already verified environment.

### 7. Add the missing generated-Slurm integration coverage

Execute a rendered sweep script with synthetic fixtures and a fake prefix environment. No GPU,
Isaac, network, or real training is allowed in ordinary tests.

At minimum prove:

- polluted host `PATH` still invokes only `$CONDA_PREFIX/bin/python` and prefix-owned tools;
- exact `0-15%2` default array and one GPU per task;
- stable task-index/cell/run-ID mapping;
- successful scratch execution and atomic durable publication;
- durable failure evidence when preflight or training fails;
- W&B sanitization before final publication;
- duplicate final destination is rejected;
- retry uses a new attempt ID and never overwrites earlier evidence;
- no Conda activation, Isaac import, `torchrun`, DeepSpeed, ZeRO, or DDP;
- rendered shell passes `bash -n`.

## Regression Requirements

Write failing tests before implementation for every confirmed defect:

1. A checkpoint whose resolved config has the wrong epoch/EMA/run ID must fail return verification.
2. A tampered `resolved_config.json`, with or without an updated checkpoint hash, must fail.
3. A scale view where source-master and action-export fingerprints differ must preserve and verify
   both exact values; tampering either must fail.
4. A normalization mean modified by `+123`, with refreshed normalization and manifest hashes, must
   still fail numerical recomputation.
5. A changed pose seed range, yaw, calibration fingerprint, or pose-plan hash must fail dataset and
   transfer verification.
6. A deleted/falsified candidate-provenance row or replacement link must fail.
7. The actual written return file list must contain all payload and two control files exactly once.
8. A polluted-`PATH` rendered-script smoke must exercise success, failure, duplicate prevention,
   and new-attempt retry.
9. The configured/rendered Conda prefix must equal the supported bootstrap/proven environment
   prefix.
10. Existing `v2_pose`, pilot, W&B canary, historical manifests, checkpoint loading, and returned
    pilot evidence must remain unchanged and pass.

Use small synthetic fixtures for unit and integration tests. Do not make the ordinary test suite
read all 550 real episodes unless it is an explicitly separate real-artifact gate.

## Implementation Boundaries

- Do not change the scientific matrix, N definitions, fixed holdouts, pose balance, seed 0,
  training epochs, policy defaults, action/observation semantics, controller limits, or force gates.
- Do not change the Alex V2 URDF or calibration.
- Do not modify `v2_pose`, pilot datasets, historical job evidence, returned checkpoints, or
  historical manifests.
- Do not hand-edit generated HDF5, checkpoint, normalization, or manifest evidence to force a pass.
- Prefer shared authoritative helpers over parallel validators or copied contracts.
- Keep Isaac imports out of cluster modules and returned-checkpoint verification.
- Preserve exact-attempt, no-overwrite, atomic-publication, symlink-free W&B, and secret-scan
  behavior.

## Execution Order

1. Record the findings and gates in ignored `TODO.md`.
2. Commit this prompt alone if it is the only initial change.
3. Add focused failing regressions for all seven required-fix sections.
4. Implement checkpoint/config and dual-fingerprint binding.
5. Implement normalization recomputation.
6. Implement generation-plan, calibration, and candidate-ledger verification.
7. Fix return controls and the environment prefix.
8. Add generated-Slurm integration coverage.
9. Run focused tests after each block and make concise commits only when green.
10. Run the full existing suite and static/shell checks.
11. Run the real local dataset, A2/A3, view, normalization, generation-provenance, and transfer
    verifiers against the existing artifacts.
12. Confirm the existing 550 HDF5 files, episode IDs, view memberships, and numerical normalization
    values did not change.
13. Update only durable relevant documentation.
14. From a clean committed checkout, rebuild and verify the final sweep transfer manifest,
    checksum file list, command artifact, and example Slurm script.
15. Commit final generated metadata if required by repository policy, re-run affected verification,
    and stop.

Do not push or merge.

## Required Validation

Use the official launcher for repository Python validation:

```text
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script-or--m-module>
```

Run at minimum:

```text
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
/home/pacquadr/.local/bin/ruff check .
git diff --check
bash -n <every generated/bootstrap Slurm or shell script>
```

Also require:

- focused regressions for every finding;
- real `build_scale_dataset.py verify` success with strengthened checks;
- real A2/A3 distinctness and exact-conversion success;
- independent deterministic replay of all four views;
- exact recomputation of all eight normalization artifacts;
- exact candidate-ledger and selected-source-fingerprint verification;
- sweep transfer-manifest build and verify from a clean committed checkout;
- exact 16-cell config/render consistency;
- generated-script integration smoke under polluted `PATH`;
- return-package control-file integration test;
- historical pilot/stabilization manifest hashes unchanged;
- final worktree clean.

The final full suite must not regress below the 701-test baseline except for deliberate additions;
the two existing IsaacLab deprecation warnings are allowed.

## Stop Conditions

Stop and report exact evidence if:

- unrelated user changes overlap the task;
- a fix requires changing the frozen scientific matrix or safety contracts;
- the corrected verifier proves any of the 550 published source episodes, views, or normalization
  values are invalid;
- preserving backward compatibility would require silently reinterpreting historical checkpoints;
- the proven Gilbreth environment contract cannot support the full-sweep runtime;
- any focused/full test, Ruff, diff, shell, real-artifact, historical-hash, or clean-tree gate fails;
- the final transfer manifest cannot be rebuilt from the exact clean implementation commit.

Do not weaken a validator, regenerate evidence unnecessarily, delete historical artifacts, or
continue with partial success.

## Documentation

Update only key durable facts in:

- `docs/architecture.md` for dual dataset fingerprints and exact resolved-config provenance;
- `docs/development.md` for strengthened local verification and manifest rebuilding;
- `docs/cluster.md` for the proven environment prefix and directly transferable return package;
- `docs/status.md` for local readiness status only.

Keep documentation concise. Do not claim that files were transferred, the full sweep ran, or any
scientific result exists.

## Final Report

Report only decision-relevant evidence:

- branch and commits;
- files/contracts changed;
- focused and full test results;
- dual fingerprint values and tamper-test results;
- normalization recomputation results for all eight artifacts;
- pose-plan, calibration, seed-range, candidate-ledger, and source-fingerprint verification;
- supported Conda prefix and rendered Slurm hash;
- return-file-list/control-file proof;
- final transfer-manifest path/hash, payload count/bytes, and rsync-list path/hash;
- confirmation that real episodes, IDs, views, normalization values, pilot evidence, and historical
  manifests remained unchanged;
- remaining limitations or blockers.

End exactly with:

```text
Ready to transfer full sweep: YES or NO
Full sweep transferred: NO
Full sweep submitted: NO
Phase 4 started: NO
```
