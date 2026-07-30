# AlexDoor-XAS Technical Wiki

This wiki describes the latest verified state of AlexDoor-XAS while preserving
phase attribution through concise derived documentation. Code and tests remain
the source of truth for executable behavior.

## Implementation Phases

- [[implementation_phases/phase-1-project-and-simulation-readiness|Phase 1 — Project and Simulation Readiness]] — Established the package, runtime boundary, path registry, asset checks, and simulator smoke gate.
- [[implementation_phases/phase-2-scripted-baseline-and-data-engine|Phase 2 — Scripted Baseline and Data Engine]] — Built the door task, deterministic controller, execution/recording pipeline, action-space exports, and reference evaluations.
- [[implementation_phases/phase-3-non-vla-learned-baselines|Phase 3 — Non-VLA Learned Baselines]] — Added dataset contracts, safety adapters, ACT and Diffusion policies, training, and closed-loop evaluation support.
- [[implementation_phases/extra-01-alex-v2-migration|Extra 01 — Alex V2 Migration]] — Replaced the provisional V1 path with the calibrated fixed-base Alex V2 benchmark and regenerated matched data.
- [[implementation_phases/extra-02-local-stabilization|Extra 02 — Local Stabilization]] — Hardened evaluation semantics, provenance, force handling, and the local N50 learned-policy matrix.
- [[implementation_phases/extra-03-gilbreth-compatibility-pilot|Extra 03 — Gilbreth Compatibility Pilot]] — Established the non-Isaac cluster environment and validated two-cell training and artifact return.
- [[implementation_phases/extra-04-scale-dataset|Extra 04 — Scale Dataset]] — Produced one 550-episode paired master and four fingerprinted nested training views.
- [[implementation_phases/extra-05-full-gilbreth-nested-sweep|Extra 05 — Full Gilbreth Nested Sweep]] — Trained the complete sixteen-cell policy/action-space/data-scale matrix on Gilbreth.
- [[implementation_phases/extra-06-phase-3-unified-evaluation|Extra 06 — Phase 3 Unified Evaluation]] — Ran and packaged the matched 576-rollout closed-loop evaluation across all returned checkpoints.

## Topics

- [[topics/system-architecture|System Architecture]] — System boundaries, runtime separation, main control/data flows, and repository responsibility map.
- [[topics/action-representations-and-adapters|Action Representations and Adapters]] — Canonical A1–A4 meanings, frame semantics, conversion path, and safety mediation.
- [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]] — Episode serialization, matched exports, splits, normalization, chunking, and scale-view identity.
- [[topics/alex-v2-benchmark|Alex V2 Benchmark]] — Calibrated asset identity, task geometry, control, sensing, runtime constraints, and limitations.
- [[topics/learned-policy-stack|Learned Policy Stack]] — Shared state-only policy interface and the implemented ACT and Diffusion models.
- [[topics/provenance-and-artifact-lifecycle|Provenance and Artifact Lifecycle]] — Fingerprints, fail-closed bindings, cluster packages, run outputs, and curated evidence.

## Key Decisions

- [[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]] — Uses one physical trajectory source to compare canonical action representations without changing the task.
- [[decisions/calibrated-position-only-alex-v2-execution|Calibrated Position-Only Alex V2 Execution]] — Grounds the benchmark in six-arm-joint differential IK and a collision-derived tool point.
- [[decisions/fail-closed-provenance-and-immutable-artifacts|Fail-Closed Provenance and Immutable Artifacts]] — Rejects unexplained drift and preserves attempt-specific and curated historical evidence.
- [[decisions/workstation-simulation-and-non-isaac-cluster-training|Workstation Simulation and Non-Isaac Cluster Training]] — Keeps Isaac authority local while using Gilbreth only for portable PyTorch training.
- [[decisions/one-scale-master-with-nested-views|One Scale Master with Nested Views]] — Publishes one paired physical master and deterministic nested train subsets with fixed holdouts.

## Experiments

- [[experiments/local-n50-stabilization|Local N50 Stabilization Matrix]] — Four-cell, 144-rollout smoke matrix used to validate closed-loop semantics and bounded safety corrections.
- [[experiments/gilbreth-nested-scale-sweep|Gilbreth Nested Scale Sweep]] — Sixteen-cell cluster training run across policy family, action space, and nested training size.
- [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] — Matched 576-rollout evaluation that found saturated success and one force-review cell.

## Sources

No user-provided raw sources have been ingested. The setup guidance used to
establish this wiki is intentionally excluded from project knowledge.
