# ACT-A3-N50 Seed-112 Force Diagnostic

## Purpose

Review the single `219.95 N` force-watch event from the frozen Phase 3 unified
evaluation before designing the next dataset. The target was ACT,
`A3_obj_rel_ee_delta`, `v3_scale_n50`, training seed 0, D0 randomized rollout
seed 112.

## Result

The exact fresh-process replay preserved the original 100–112 rollout prefix
and reproduced the target row exactly: success at tick 93, first contact and a
single `219.953 N` peak at tick 55, and trace SHA-256
`56a2b1a40dea614685385a5057d2c47aaec1c7bee4dac523ceb110f6947f27d2`.

The recurrence rule authorized two cases that changed only seed 112's initial
door-frame X offset:

| Case | Peak force | First contact / peak | >200 N ticks | Success | Adapter corrections |
|---|---:|---:|---:|---:|---:|
| Exact replay | 219.953 N | 55 / 55 | 1 | tick 93 | 0 |
| X −1 mm | 66.002 N | 56 / 57 | 0 | tick 94 | 2 |
| X +1 mm | 86.442 N | 55 / 55 | 0 | tick 94 | 0 |

Seeds 100–111 retained their original trace hashes in all three executions.
All cases succeeded and none produced an adapter rejection.

## Interpretation and Future Use

The event is reproducible, not an unexplained one-off under the frozen replay.
Both 1 mm perturbations removed the force-watch exceedance, supporting local
sensitivity to initial door-normal position and contact-entry motion. The
negative perturbation crossed the existing contact-entry shaping trigger and
was corrected; the positive perturbation produced a smaller accepted
first-contact command.

This small diagnostic does not establish root cause, validate an adapter
change, or prove hardware safety. It should guide the next dataset design
toward explicit coverage of initial contact geometry and entry motion, followed
by multi-training-seed closed-loop evaluation and a small contact-start
robustness gate. A new dataset must not be assumed to resolve the event without
that post-training evidence.

## Evidence

- Structured results: `outputs/curated/phase3_seed112_force_diagnostic/results.json`
- Provenance: `outputs/curated/phase3_seed112_force_diagnostic/provenance.json`
- Integrity inventory: `outputs/curated/phase3_seed112_force_diagnostic/SHA256SUMS.txt`
- Historical Phase 3 package: `outputs/curated/phase3_unified_evaluation/`

The historical Phase 3 package was not modified.
