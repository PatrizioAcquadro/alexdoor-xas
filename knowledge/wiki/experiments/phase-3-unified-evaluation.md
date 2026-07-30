# Experiment — Phase 3 Unified Evaluation

## Purpose

Compare all sixteen returned Phase 3 checkpoints under one matched,
fingerprint-bound closed-loop protocol and determine whether the evidence
supports a policy, representation, or data-scale conclusion.

## Protocol

- Checkpoints: 16 cells from Gilbreth attempt `11281591`.
- Matrix: ACT/Diffusion × A2/A3 × N50/N100/N250/N500.
- Rollouts: 36 per cell, 576 total.
- Pose allocation per cell: D0 × 20; D1–D4 × 4 each.
- Success: first hinge-angle crossing at 45 degrees within 600 control ticks.
- Runtime: CPU physics and CUDA policy inference on the calibrated workstation.
- ACT: chunk size 40, no temporal ensembling.
- Diffusion: prediction horizon 16, eight executed actions, DDIM with 10
  inference steps.
- Execution: explicit adapter decisions, panel-filtered force, online
  first-crossing, terminal evidence, matched seeds, and fail-closed provenance.

The protocol is fixed in `configs/phase3_unified_eval.v1.json` and implemented
by `src/alexdoor_xas/eval/phase3_unified.py` and
`scripts/run_phase3_unified_evaluation.py`.

## Results

Every cell completed 36/36 successful rollouts, for 576/576 total. Each cell's
95% Wilson interval is approximately 90.4%–100%.

| Adapter outcome | Count |
|---|---:|
| Accepted | 54,183 |
| Corrected | 3,495 |
| Rejected | 0 |
| Total decisions | 57,678 |

The evaluation recorded 3,506 warnings in the expected family. Fifteen cells
remained within the configured force-watch envelope. ACT-A3-N50 at seed 112
reached 219.95 N peak panel-filtered force and is `REVIEW_REQUIRED`.

The later
[[act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]
reproduced that target row exactly and found local sensitivity to ±1 mm changes
in initial door-normal position. This follow-up does not alter the historical
matrix or clear the review status.

## Interpretation

The benchmark is success-saturated under this protocol. It does not identify a
winning policy family, action representation, or training size. It also shows
no reliable monotonic N50→N500 improvement, but this does not imply that data
scale is unimportant in broader tasks or less saturated evaluations.

The force outlier does not invalidate the other cells or the run's completion,
but it prevents a blanket “all cells cleared” safety statement. All forces are
simulation engineering signals and do not constitute physical-robot safety
evidence.

These results support treating a harder or more discriminative benchmark as a
future bridge before making stronger VLA or scaling claims. Such a bridge is
not implemented by this phase.

## Evidence and Provenance

- Phase: [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]]
- Training inputs: [[gilbreth-nested-scale-sweep|Gilbreth Nested Scale Sweep]]
- Curated package: `outputs/curated/phase3_unified_evaluation/`
- Report: `outputs/curated/phase3_unified_evaluation/report.md`
- Tests: `tests/test_phase3_unified_evaluation.py`
- Force follow-up:
  [[act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]]

The curated package is immutable historical evidence. A changed protocol or
rerun must receive a new identity rather than replacing this package.

## Version Notes

- 2026-07-18 — All sixteen cells completed 36/36 rollouts; interpretation
  closed as success-saturated with one force-review cell and no winner claim.
