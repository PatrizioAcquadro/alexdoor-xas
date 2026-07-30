# Experiment — ACT-A3-N50 Seed-112 Force Diagnostic

## Purpose

Determine whether the `219.95 N` watch event from the
[[phase-3-unified-evaluation|Phase 3 Unified Evaluation]] is reproducible and
whether a minimal change in initial contact geometry affects it. The result is
an input to future dataset design, not a dataset specification or adapter fix.

## Protocol

- Target: ACT, `A3_obj_rel_ee_delta`, `v3_scale_n50`, training seed 0.
- Condition: D0 randomized rollout seed 112.
- Runtime contract: CPU physics, CUDA inference, ACT horizon 40, no temporal
  ensemble, adapter-v1, 45-degree success, and the 200 N force-watch bound.
- Exact replay: fresh process containing the original seed prefix 100–112 so
  seed 112 retains its original same-process physics history.
- Decision rule: recurrence authorizes only two cases changing seed 112's
  initial door-frame X offset by −1 mm and +1 mm. All other variables and the
  100–111 prefix remain fixed.

## Results

The exact replay reproduced the historical target row byte-for-byte after
canonical JSON sorting. It retained success at tick 93, first contact and a
single `219.953 N` peak at tick 55, and the original trace hash.

| Case | Peak force | First contact / peak | >200 N ticks | Success | Corrections |
|---|---:|---:|---:|---:|---:|
| Exact replay | 219.953 N | 55 / 55 | 1 | tick 93 | 0 |
| X −1 mm | 66.002 N | 56 / 57 | 0 | tick 94 | 2 |
| X +1 mm | 86.442 N | 55 / 55 | 0 | tick 94 | 0 |

Seeds 100–111 preserved their historical trace hashes in every execution. All
three diagnostic cases succeeded and none produced an adapter rejection.

## Interpretation

The force event is reproducible under the frozen replay and is locally
sensitive to initial door-normal position and contact-entry motion. In the
negative perturbation, the policy request crossed the existing contact-entry
shaping trigger and two commands were corrected. In the positive perturbation,
the policy produced a smaller accepted first-contact command.

This evidence does not prove root cause or justify changing the adapter
threshold. It supports explicit contact-start coverage in the next dataset,
followed by multi-training-seed closed-loop evaluation and a small perturbation
gate. Dataset admission and post-training policy safety remain separate
controls; neither is hardware-safety evidence.

## Evidence and Provenance

- Curated report:
  `outputs/curated/phase3_seed112_force_diagnostic/report.md`
- Structured results:
  `outputs/curated/phase3_seed112_force_diagnostic/results.json`
- Provenance and hashes:
  `outputs/curated/phase3_seed112_force_diagnostic/provenance.json` and
  `SHA256SUMS.txt`
- Related concepts:
  [[topics/action-representations-and-adapters|Action Representations and Adapters]]
  and [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]]

The original `outputs/curated/phase3_unified_evaluation/` package remains
immutable.

## Version Notes

- 2026-07-30 — Exact recurrence confirmed; two ±1 mm door-normal
  perturbations removed the force-watch exceedance without changing success.
