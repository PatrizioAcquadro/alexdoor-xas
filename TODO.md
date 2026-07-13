# TODO — Gilbreth N50 compatibility pilot preparation

Source spec: `prompts/prepare_gilbreth_compatibility_pilot_prompt.md`.
Execution model: hybrid local/cluster workflow. Complete and verify all local package,
configuration, transfer, preflight, Slurm, and return-verification tooling here; then stop.
Gilbreth execution and the later scientific sweep remain explicit follow-on work.

Baseline: branch `impl/phase3-3-local-review-fixes` @
`a313f602cc911aaa4cc5dbdc0ee0aca52da2e1d8`.

Pre-existing user changes to preserve:

- `CLUSTER.md` — untracked Gilbreth notes that this task must incorporate and track intentionally.
- `prompts/post_phase3_3_local_review_fixes_prompt.md` — deleted before this task; preserve the
  deletion without silently staging or restoring it.

## Local work in this task

- [x] 0. Initialize task tracking with the baseline and pre-existing worktree changes.
- [x] 1. Add the versioned two-cell pilot configuration and record the future nested-sweep
  contract without generating datasets.
- [x] 2. Add the pinned, non-Isaac Python 3.11 environment specification and fail-loud Gilbreth
  bootstrap script with configurable PyTorch/CUDA selection.
- [x] 3. Add tests-first coverage for the pilot inventory, fingerprints, file/hash/secret/source
  rejection, deterministic transfer outputs, pure preflight, Slurm contract, return artifacts,
  and future-sweep invariants.
- [x] 4. Implement and locally validate the pilot transfer manifest builder/verifier and
  repository-relative resumable rsync file list/template without changing the historical
  stabilization manifest.
- [x] 5. Implement the split pure/live non-Isaac preflight, including dataset and checkpoint I/O
  validation locally and fail-closed CUDA/allocation/A100 probing for Gilbreth.
- [x] 6. Implement and locally validate the deterministic two-cell Slurm renderer, atomic
  per-cell status publication, and durable result-copy contract.
- [x] 7. Implement and locally validate return-manifest construction/verification, resumable
  return rsync templating, and non-Isaac ACT/Diffusion checkpoint loading.
- [x] 8. Update tracked cluster documentation with exact configurable bootstrap, preparation,
  preflight, render, submit, and return commands; do not access or submit to Gilbreth.
- [x] 9. Pass focused pilot tests, shell syntax checks, Ruff, `git diff --check`, the full test
  suite, and the three required repository verification scripts using the official launcher.
- [ ] 10. Review the full diff, preserve unrelated work, create a concise source commit containing
  only the intended pilot preparation, and reach the required clean source state.
- [ ] 11. Build and verify the final ignored pilot-transfer artifacts from the clean committed
  checkout; confirm commit binding, exact inventory, hashes/fingerprints, and clean tree.
- [ ] 12. Perform a final local spec/diff/test audit and report the ready-to-transfer package,
  exact commands/templates, remaining Gilbreth values, and explicit non-execution boundaries.

## Gilbreth follow-on work — leave unchecked in this task

- [ ] G1. Fill in the Gilbreth depot/scratch/account/partition/QOS/resource values and requested
  PyTorch/CUDA build using live Gilbreth driver/module evidence.
- [ ] G2. Transfer the committed pilot package to Gilbreth and verify the incoming manifest.
- [ ] G3. Bootstrap the persistent Gilbreth Conda environment and capture the resolved
  environment inventory/lock.
- [ ] G4. Run the live Gilbreth preflight with the allocated CUDA device and A100-80GB check when
  that partition is requested.
- [ ] G5. Submit and complete the two-cell ACT-on-A2 / Diffusion-on-A3 compatibility pilot.
- [ ] G6. Build the return manifest and transfer durable pilot artifacts back to Ubuntu.
- [ ] G7. Verify returned hashes and Ubuntu compatibility by loading both checkpoints without
  Isaac; do not run closed-loop evaluation as part of the pilot task.

## Later scientific sweep — leave unchecked in this task

- [ ] S1. Generate one deterministic master episode pool with equal pose balance and paired A2/A3
  exports from the same source episodes.
- [ ] S2. Materialize nested N50/N100/N250/N500 training subsets, where N counts training
  episodes, with fixed shared validation/test episodes.
- [ ] S3. Run the full 16-cell cluster sweep only after the pilot and returned-environment/
  checkpoint compatibility gates pass.
