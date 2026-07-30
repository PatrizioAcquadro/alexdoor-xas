# Experiment — Local N50 Stabilization Matrix

## Purpose

Validate the hardened closed-loop evaluation, adapter accounting, force,
warning, provenance, and fresh-process determinism semantics across the four
local Alex V2 learned-policy cells. This was an engineering smoke matrix, not a
policy-selection experiment.

## Protocol

- Policy families: ACT and Diffusion.
- Action representations: A2 and A3.
- Dataset: Alex V2 `v2_pose`, N50 training scale.
- Rollouts: 36 matched episodes/seeds per cell, 144 total.
- Simulator: calibrated [[topics/alex-v2-benchmark|Alex V2 Benchmark]] with
  CPU physics.
- Execution: checkpoint-bound state-only inference through the corresponding
  adapter; online first-crossing and terminal-force capture.
- Additional determinism: 20 probes launched in fresh Python processes.

The plan is encoded in `configs/local_smoke_eval_plan_n50.json`; evaluation and
aggregation use `scripts/eval_act.py`, `scripts/eval_diffusion.py`, and
`scripts/summarize_smoke_eval.py`.

## Results

| Cell | Successful rollouts | Corrected | Rejected | Peak panel force |
|---|---:|---:|---:|---:|
| ACT-A2 | 36/36 | 296 | 0 | 135.8 N |
| ACT-A3 | 36/36 | 233 | 0 | 129.4 N |
| Diffusion-A2 | 36/36 | 195 | 0 | 145.5 N |
| Diffusion-A3 | 36/36 | 210 | 0 | 143.7 N |

Each cell emitted 219 warnings, for 876 total. All belonged to the declared
bounded reset-transient family. All twenty fresh-process determinism probes
matched their expected outputs.

## Interpretation

The matrix established that the four tested cells could complete the declared
smoke protocol with no rejected actions, stable warning classification,
bounded recorded forces, and process-independent determinism. Correction counts
remain meaningful: success did not mean every policy request passed through
unchanged.

The run does not establish a best policy, a best representation, robustness
beyond the tested seeds/poses, or physical-robot safety. Later scale training
and the matched 576-rollout evaluation provide the current comparative
evidence.

## Provenance

- Phase: [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]]
- Dataset family: matched A2/A3 `v2_pose` products; generated dataset contents
  are ignored and are not present in the current checkout
- Plan: `configs/local_smoke_eval_plan_n50.json`
- Verification: `tests/test_summarize_smoke_eval.py`,
  `tests/test_stabilization_doc.py`,
  `scripts/verify_stabilization_doc.py`

## Version Notes

- 2026-07-11 — Four cells completed 144/144 smoke rollouts and all twenty
  fresh-process determinism probes.
