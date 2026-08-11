# Extra 06 — Phase 3 Unified Evaluation

## Objective

Evaluate all sixteen returned Phase 3 checkpoints under one matched,
fingerprint-bound workstation protocol, package durable review evidence, and
state what the saturated results do and do not support.

## Focus

### Subphase Extra 06.1 — Unified Evaluation Protocol

#### Implementation

`configs/phase3_unified_eval.v1.json` defines one source attempt, all sixteen
cells, 36 matched rollout seeds per cell, and the fixed simulator protocol.
Each cell receives 20 D0 rollouts and four from each of D1–D4, for 576 total
rollouts. Success is the first 45-degree hinge crossing within 600 control
ticks.

ACT uses a 40-step action chunk without temporal ensembling. Diffusion uses a
16-step prediction horizon, eight executed steps, and deterministic DDIM with
10 inference steps. CPU physics remains authoritative; policy inference uses
CUDA where configured.

`src/alexdoor_xas/eval/phase3_unified.py` resolves checkpoints, datasets, views,
normalization, robot/calibration/runtime identity, matched seeds, rollout
execution, aggregation, plots, and report inputs. It fails before execution on
mixed or stale provenance and records per-rollout and per-decision evidence.
`scripts/run_phase3_unified_evaluation.py` is the supported entry point.

#### Key Decisions and Problems

- The evaluation changes no policy hyperparameters or checkpoint contents
  after inspecting results.
- All cells use the same physical master, fixed holdouts, seed allocation,
  simulator, success threshold, and force measurement path.
- Thirty-six rollouts per cell give a useful matched engineering comparison but
  limited power once every cell saturates at 100% observed success.

#### Tests

- `tests/test_phase3_unified_evaluation.py` covers configuration expansion,
  provenance rejection, matched seeds, aggregation, force review, report
  generation, and deterministic packaging.
- The runner validated all required checkpoints, datasets, views,
  normalization, robot/calibration, and simulator identities before rollout.

### Subphase Extra 06.2 — Results and Safety Review

#### Implementation

All sixteen cells completed 36/36 successful rollouts. For each cell, the 95%
Wilson interval is approximately 90.4%–100%. Across 57,678 adapter decisions,
54,183 were accepted, 3,495 were corrected, and none were rejected. The
evaluation recorded 3,506 warnings in the expected family.

Fifteen cells remained within the configured force-watch envelope. One cell,
ACT-A3-N50 at seed 112, reached a 219.95 N peak panel-filtered force and is
marked `REVIEW_REQUIRED`. The run itself completed; the review state prevents a
blanket safety-clear claim. Full tables and interpretation are maintained in
[[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]].

#### Key Decisions and Problems

- Equal observed success is saturation, not proof that all policies are
  equivalent or that a winner exists.
- The results show no reliable monotonic benefit from N50 to N500 under this
  test. They do not show that additional data is generally useless.
- Simulator force is an engineering signal, not physical-hardware force or
  safety validation.

#### Tests

- The actual run completed 576/576 rollouts, 36/36 for every cell, with zero
  adapter rejections.
- Force-policy validation correctly classified fifteen cells as watch-pass and
  one cell as review-required without invalidating unrelated cells.

### Subphase Extra 06.3 — Curated Evidence Package

#### Implementation

Small durable outputs were promoted to
`outputs/curated/phase3_unified_evaluation/`: the report, summary tables, selected
plots, machine-readable metadata, and exact hashes needed for review. Raw
rollouts, checkpoints, videos, and logs remain ignored in their run-specific
locations.

The curated report is historical evidence and must not be regenerated in place
to absorb later code or interpretation changes. A changed protocol or rerun
requires a new run and separately identified evidence package, consistent with
[[decisions/fail-closed-provenance-and-immutable-artifacts|Fail-Closed Provenance and Immutable Artifacts]].

#### Key Decisions and Problems

- Curated evidence is intentionally small and reviewable; it is not a second
  operational dataset.
- Git records detailed evolution, while the wiki retains only the latest
  verified interpretation and concise version notes.
- [[status|Project Status]], README, and the curated report reflect this
  completed phase; the wiki status and curated evidence remain authoritative
  for current completion claims.

#### Tests

- Package validation checked expected files, internal identities, summary
  counts, report consistency, and SHA-256 inventories.
- `git diff --check` and the deterministic repository suite were included in
  phase closeout alongside the evaluation-specific tests.

## Version Notes

- 2026-07-18 — The matched 576-rollout unified evaluation completed and its
  curated evidence package was verified.
- 2026-07-18 — Interpretation was closed as success-saturated with one
  force-review cell; no representation, policy, or data-scale winner was
  claimed.
