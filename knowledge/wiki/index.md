# AlexDoor-XAS Technical Wiki

## Project Status

- [[status|Project Status]] — Maintained capabilities, completed evidence, retired workflows, and boundaries.

## Implementation Phases

- [[implementation_phases/phase-1-project-and-simulation-readiness|Phase 1 — Project and Simulation Readiness]] — Package, runtime, assets, and simulator readiness.
- [[implementation_phases/phase-2-scripted-baseline-and-data-engine|Phase 2 — Scripted Baseline and Data Engine]] — Scripted control, recording, and matched exports.
- [[implementation_phases/phase-3-non-vla-learned-baselines|Phase 3 — Non-VLA Learned Baselines]] — Dataset contracts, adapters, ACT, Diffusion, and evaluation.
- [[implementation_phases/extra-01-alex-v2-migration|Extra 01 — Alex V2 Migration]] — Fixed-base Alex V2 calibration and execution.
- [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]] — Closed-loop, warning, and force semantics.
- [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03 — Gilbreth Compatibility Pilot]] — Completed two-cell training compatibility result.
- [[implementation_phases/extra-04-scale-dataset|Extra 04 — Scale Dataset]] — Retained 550-episode master and nested views.
- [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]] — Completed sixteen-cell training result.
- [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]] — Completed 576-rollout matched evaluation.

## Topics

- [[topics/system-architecture|System Architecture]] — Current components, flows, and runtime boundaries.
- [[topics/action-representations-and-adapters|Action Representations and Adapters]] — A1-A4 meanings, frames, and execution semantics.
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]] — Storage, splits, views, normalization, and sampling.
- [[topics/alex-v2-benchmark|Alex V2 Benchmark]] — Calibration, control, sensing, and limits.
- [[topics/learned-policy-stack|Learned Policy Stack]] — ACT, Diffusion, checkpoint v2, and evaluation.

## Key Decisions

- [[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]] — Compare representations on shared physical episodes.
- [[decisions/calibrated-position-only-alex-v2-execution|Calibrated Position-Only Alex V2 Execution]] — Use six-joint IK at the collision-derived tool point.
- [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]] — Compare data scale with fixed holdouts.

## Experiments

- [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]] — Complete sixteen-cell training matrix.
- [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] — Saturated 576-rollout evaluation with one force-review cell.
- [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]] — Reproducible force event and two position perturbations.

## Sources

No user-owned raw source has been ingested.
