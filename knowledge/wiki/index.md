# AlexDoor-XAS Technical Wiki

Current pages describe only the maintained repository. Implementation-phase and experiment pages are concise historical records; they do not define active workflows.

## Project Status

- [[status|Project Status]] — Current capabilities, entry points, storage boundaries, historical conclusions, and limits.

## Implementation Phases

Historical development records:

- [[implementation_phases/phase-1-project-and-simulation-readiness|Phase 1 — Project and Simulation Readiness]] — Initial package, dependency, and simulator foundation.
- [[implementation_phases/phase-2-scripted-baseline-and-data-engine|Phase 2 — Scripted Baseline and Data Engine]] — Scripted execution, recording, and matched export foundation.
- [[implementation_phases/phase-3-non-vla-learned-baselines|Phase 3 — Non-VLA Learned Baselines]] — State-only policy, adapter, and evaluation foundation.
- [[implementation_phases/extra-01-alex-v2-migration|Extra 01 — Alex V2 Migration]] — Migration from provisional assumptions to fixed-base Alex V2.
- [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]] — Closed-loop and force-semantics stabilization.
- [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03 — Gilbreth Compatibility Pilot]] — Completed two-cell A100 compatibility check.
- [[implementation_phases/extra-04-scale-dataset|Extra 04 — Scale Dataset]] — Completed 550-episode master and nested-view construction.
- [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]] — Completed sixteen-cell training matrix.
- [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]] — Completed 576-rollout matched evaluation.

## Topics

Current technical behavior:

- [[topics/system-architecture|System Architecture]] — Maintained runtime, data, policy, evaluation, and storage flow.
- [[topics/alex-v2-benchmark|Alex V2 Benchmark]] — External asset, calibration, canonical scenes, control, sensing, and limits.
- [[topics/action-representations-and-adapters|Action Representations and Adapters]] — A1-A4 meanings and maintained execution boundaries.
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]] — `phase2.v2`, matched exports, splits, views, normalization, and model data.
- [[topics/learned-policy-stack|Learned Policy Stack]] — ACT/Diffusion configuration, training, checkpoint/resume, outputs, and evaluation.

## Key Decisions

Current architectural and scientific contracts:

- [[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]] — Hold physical experience and task geometry aligned across A1-A4.
- [[decisions/calibrated-position-only-alex-v2-execution|Calibrated Position-Only Alex V2 Execution]] — Use one calibration, tool-point IK, and exact-door contact sensing.
- [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]] — Reuse fixed holdouts and nested training memberships for the completed scale data.

## Experiments

Historical scientific records:

- [[experiments/gilbreth-nested-scale-sweep|Nested Scale Sweep]] — Completed sixteen-cell training result and limits.
- [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] — Saturated matched evaluation with no selected winner.
- [[experiments/act-a3-n50-seed-112-force-diagnostic|ACT-A3-N50 Seed-112 Force Diagnostic]] — Reproducible force event and bounded perturbation result.

## Sources

No user-owned raw source has been ingested.
