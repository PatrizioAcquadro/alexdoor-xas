# TASK: Close the Final Full-Sweep Provenance Gaps on Ubuntu

## Objective

Fix the two remaining fail-open provenance defects found by the read-only `robotics-review` of the
completed full-sweep implementation. Work only in the Ubuntu checkout. Finish with a clean,
committed, independently verified transfer package that can later be handed to Gilbreth.

The current selected dataset, A2/A3 exports, nested views, normalization artifacts, sweep-cell
configuration, Slurm execution contract, environment contract, and return tooling already passed
review. Preserve them. Do not broaden this task into another sweep implementation or regenerate
episodes unless the strengthened verifier proves that the existing artifact is invalid.

Do **not** use subagents. Use `robotics-test` first to add regressions that reproduce both defects,
then use `robotics-implementation` for the smallest shared-contract fixes.

Do **not** push, merge, transfer files, connect to Gilbreth, request an allocation, call `sbatch`,
launch the full sweep, start closed-loop evaluation, or begin Phase 4.

## Verified Starting State

- Repository: `/home/pacquadr/Desktop/DoorManipulation`
- Branch: `impl/full-cluster-dataset-scale-sweep`
- Reviewed HEAD: `5b8670c57d421e6575c16731eaf9f683d562ec4c`
- Expected only initial tracked change: this prompt file.
- Worktree before this prompt replacement: clean.
- Full official-launcher suite: 719 passed with two allowed IsaacLab deprecation warnings.
- Focused sweep/dataset regressions: 49 passed.
- Real dataset verification passed for 550 paired episodes, four nested views, and eight
  normalization artifacts.
- Existing candidate ledger: 750 rows, comprising 550 `SELECTED` source candidates and 200
  `NOT_NEEDED_OVERDRAW` candidates, with no skipped candidates or replacements.

Stable dataset fingerprints:

- Source master: `79dd3e819c2fbb2d21b9cf3848df7942bd6e69f163a9d82ef892710f5e39d27b`
- A2 export: `5179595aeeb7a69898456750658feb59b6a2aaff1821f6ce8af36739f04f9e06`
- A3 export: `3b528ceb451e9547c2aec76d47dafbf7129f870bdfaa6c691769304487ca615f`
- Master manifest: `eaead0e1626bb15403365db9a122818ae61a331fb9d9dc461d0bdce0681e0ccf`
- Candidate ledger: `41991acbe90a3a559720e02b7a34d71c2ccf90c4c0e2fb45c26f37eeb97b0000`
- Pose plan: `adec8222928a7bfa35776f497fed258bce9047bb21b202984e77da1910588e40`
- Calibration fingerprint: `066e0a2d0157549a331b96702e643fd7626eed58d33bbe701005e111ed358948`

Current local handoff artifacts, expected to be rebuilt after the fixes:

- Transfer manifest: `afa8be07065aec4ad805948e10feafae78ed63214a7c9d6774c5be981b73f7d8`
- Rsync file list: `42a11b8c827e995f633f5d6365b0b6a2a993c4f7d26f44193fe7bd2d96ed2b02`
- Example Slurm script: `7620a795ea31b57c9ab16343fb969e64fee066aed835f9d13cc74b2a5410ceb7`
- Transfer inventory: 2,295 payload files, 235,768,999 bytes, and 2,297 rsync paths.

Before editing, read completely:

- `CLAUDE.md`
- `docs/PROJECT_GUIDELINES.md`
- `TODO.md`
- this prompt
- `scripts/build_scale_dataset.py`
- `src/alexdoor_xas/calibration/alex_v2_door.py`
- `src/alexdoor_xas/cluster_sweep/transfer.py`
- `tests/test_scale_dataset.py`
- `tests/test_cluster_sweep.py`
- the master manifest, pose plan, calibration, generation report, and transfer manifest.

Replace the completed ignored `TODO.md` with a checklist for this task and update it as each gate
passes. If this prompt is the only worktree change, commit it alone with the message
`Define final sweep provenance fixes`. Preserve and stop on any unrelated overlapping change. Do not
reset, clean, or discard user work.

## Confirmed Finding 1: Unselected Candidate Rows Are Not Authenticated

`_validate_candidate_provenance()` validates raw paths, episode identity, pose, seed, content hash,
safety, and outcome only after this condition:

```text
if decision != "SELECTED":
    continue
```

The real verifier was run after changing a `NOT_NEEDED_OVERDRAW` row to contain:

- a false episode ID;
- a nonexistent source path;
- a fabricated content-group SHA-256.

With `require_source_paths=True`, it still returned:

```text
FAIL_OPEN D0 110000 PASS 750
```

This contradicts the binding requirement to validate the complete 750-row candidate ledger and to
reject any falsified candidate-provenance row.

### Required fix

Create one authoritative deterministic candidate-evaluation and selection contract shared by
publication and verification. During Ubuntu publication/verification, reconstruct and compare the
complete ledger from all 750 raw candidates—not only the selected 550.

For every row, require:

- the exact canonical pose/namespace/seed inventory and deterministic row ordering;
- an existing regular source HDF5 path contained in the expected candidate-run evidence;
- exact raw episode ID, pose ID, and seed agreement;
- a recomputed content-group SHA-256 matching the row;
- recomputed safety, task outcome, duplicate-content, and pose-provenance results;
- an exact decision and exact deterministic reasons derived from those results;
- exact deterministic replacement linkage when replacement is necessary.

Additional decision requirements:

- `SELECTED` rows must be safe, successful, unique, and selected by the frozen algorithm.
- `NOT_NEEDED_OVERDRAW` rows must be otherwise eligible and unused only because the per-pose quota
  was already satisfied. They may not bypass raw-source validation.
- `SKIPPED` rows must carry the exact reasons recomputed from raw evidence.
- Selected overdraw replacements must map deterministically to the correct skipped source.
- The final selected set must remain 550 unique episodes, balanced 110 per pose, and must retain the
  existing selected-source fingerprint.

Do not add a second validator that can drift from `_select_master()`. Factor the shared deterministic
logic so the producer and verifier cannot disagree.

The Gilbreth verifier must remain independent of Ubuntu-only raw paths. Perform the full raw replay
locally, bind its canonical report and hashes into the transferable contract, and continue verifying
all transferable invariants on Gilbreth.

### Required regressions

Tests must fail before the implementation and pass afterward for at least:

1. A `NOT_NEEDED_OVERDRAW` row with a false episode ID.
2. A `NOT_NEEDED_OVERDRAW` row with a missing or escaping source path.
3. A `NOT_NEEDED_OVERDRAW` row with a fabricated content-group hash.
4. A changed `NOT_NEEDED_OVERDRAW` decision or invented reason.
5. A `SKIPPED` row whose raw evidence does not produce its stored reasons.
6. A false, duplicated, missing, or nondeterministically swapped replacement link.
7. Tampering that refreshes the ledger/report/outer manifest hashes; raw replay must still reject it.
8. The unchanged real 750-row ledger passing exact reconstruction.

## Confirmed Finding 2: Calibration Self-Fingerprint Is Not Recomputed

`_load_plan()` currently parses the calibration JSON and only compares:

```text
plan["calibration_fingerprint"] == calibration["fingerprint"]
```

It does not recompute the fingerprint from calibration contents or use the repository's validated
calibration loader. The review changed `reach_shell_m` while retaining the old stored fingerprint.
The sweep plan loader accepted the altered artifact even though:

```text
stored fingerprint:     066e0a2d0157549a331b96702e643fd7626eed58d33bbe701005e111ed358948
recomputed fingerprint: ad2236e634ac1f74bd17ab9a7a77caa5792e297cc530d73a9c9e0366fefec123
```

If outer transfer hashes are rebuilt, this produces a self-consistent transfer package that falsely
claims the old calibration provenance.

### Required fix

Use the existing authoritative calibration contract in
`src/alexdoor_xas/calibration/alex_v2_door.py`; do not create another partial calibration parser.

At minimum, require:

- valid calibration schema, task, validated status, complete gates, robot identity, runtime
  versions, controller fields, poses, bounds, and finite values through the shared loader;
- recomputation of `calibration_fingerprint(payload)` and exact agreement with the stored field;
- exact agreement between the recomputed calibration fingerprint and the pose-plan fingerprint;
- exact robot-asset agreement with the master dataset and transfer robot-asset contract;
- the existing calibration file SHA-256 binding in the transfer manifest.

Dataset and transfer verification must reject changed calibration contents even if the stored
fingerprint, verification-report hash, file hash, and transfer-manifest hashes are refreshed.

### Required regressions

Tests must fail before the implementation and pass afterward for at least:

1. Calibration content changed while the stored fingerprint remains stale.
2. Calibration content and stored fingerprint changed together while the pose plan remains bound to
   the canonical fingerprint.
3. Calibration and pose-plan fingerprints changed together but the calibration no longer matches
   the validated robot/runtime/master contract.
4. Refreshed verification and transfer hashes cannot hide any of the above.
5. The unchanged canonical calibration passes through the shared validated loader.

## Preservation Requirements

Do not change:

- the 550 selected episode identities or their raw/generated HDF5 content;
- the 1,100 A2/A3 exported HDF5 files;
- the source-master, A2, or A3 fingerprints;
- N50/N100/N250/N500 train, validation, or test membership;
- the eight normalization numeric values or their train-only definitions;
- the 16-cell scientific matrix, seeds, epochs, validation cadence, policy defaults, or W&B mode;
- the proven Gilbreth Conda prefix or dependency pins;
- pilot/canary evidence, historical manifests, returned checkpoints, or durable results;
- Alex V2 URDF or calibration contents.

Preserve exact resolved-config/checkpoint binding, dual fingerprints, numerical normalization
recomputation, one-GPU-per-cell Slurm execution, atomic durable publication, symlink-free W&B,
directly transferable return controls, and exact-attempt/no-overwrite behavior.

Only update documentation if the durable contract actually changes and only with key facts. Do not
claim that the sweep was transferred or run.

## Required Validation

Use the official launcher for repository Python validation:

```text
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script-or--m-module>
```

Run at minimum:

```text
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q tests/test_scale_dataset.py tests/test_cluster_sweep.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
/home/pacquadr/.local/bin/ruff check .
git diff --check
```

Also require:

- both exact adversarial reproductions to fail closed after the fix;
- real `build_scale_dataset.py verify` success;
- real A2/A3 distinctness and exact-conversion success;
- all four nested views and eight normalization recomputations to pass;
- exact replay of all 750 raw candidates and the unchanged selected-source fingerprint;
- canonical calibration validation and recomputed fingerprint agreement;
- sweep transfer-manifest build and verification from the exact clean final commit;
- `bash -n` for every generated/bootstrap Slurm or shell script;
- the generated `0-15%2` one-GPU Slurm array to remain unchanged scientifically;
- historical pilot/stabilization manifests and evidence to remain unchanged;
- final worktree clean.

The full suite must exceed the 719-test reviewed baseline by the new regressions. The two existing
IsaacLab deprecation warnings are allowed.

Rebuild the final local sweep manifest, rsync file list/command, and example Slurm only after all
code and tests are committed and the clean-tree gate passes. Commit generated metadata if required
by repository policy, then rerun affected verification from the exact final commit.

## Stop Conditions

Stop and report exact evidence if:

- unrelated work overlaps this task;
- either adversarial mutation still passes;
- the corrected verifier invalidates the existing canonical calibration or any real candidate;
- any of the 550 selected episodes, paired exports, views, or normalization values changes;
- a fix requires changing the scientific matrix, safety limits, action semantics, or calibration;
- backward compatibility with `v2_pose`, pilot checkpoints, or historical evidence regresses;
- any focused/full, Ruff, diff, shell, real-artifact, historical-hash, or clean-tree gate fails;
- the transfer package cannot be rebuilt from the exact final clean commit.

Do not weaken the validator, hand-edit evidence, regenerate data unnecessarily, or continue with
partial success.

## Commit and Final Report

Make concise commits after each validated block. Do not push or merge.

Report only decision-relevant evidence:

- final branch and commits;
- files and shared contracts changed;
- failing-before/passing-after adversarial regressions;
- focused and full test results;
- complete 750-row replay and calibration self-fingerprint results;
- preservation hashes for datasets, views, normalization, and historical evidence;
- final transfer-manifest path/hash, payload count/bytes;
- final rsync-list path/count/hash and example Slurm hash;
- remaining limitations or blockers.

End exactly with:

```text
Ready to transfer full sweep: YES or NO
Full sweep transferred: NO
Full sweep submitted: NO
Phase 4 started: NO
```
