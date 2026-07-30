# Decision — Fail-Closed Provenance and Immutable Artifacts

## Context

Robot assets, calibration, physical datasets, representation exports, splits,
normalization, checkpoints, cluster attempts, and evaluations can drift
independently. Accepting a “close enough” identity would make a successful run
look comparable while silently changing its inputs.

## Decision

Content-bind each material layer and reject missing, mixed, unexpected, or
mismatched identities. Preserve attempts and curated evidence as immutable
historical records. Diagnose drift instead of refreshing an expected
fingerprint to absorb it.

Cluster transfers and returns use exact SHA-256 inventories and attempt-specific
paths. Checkpoints are self-contained and tied to source, dataset, view/split,
normalization, robot, and calibration. Evaluation packages bind the complete
checkpoint set, protocol, simulator/runtime, seed plan, and result inventory.

## Consequences

- Failures occur early and explain which identity changed.
- Retries create new attempt evidence and never overwrite or mix the previous
  attempt.
- Behavior-changing dataset generation uses a new version and new downstream
  bindings.
- Curated Phase 3 evidence cannot be regenerated in place to match later code.
- More metadata and validation are required, but the resulting comparisons are
  auditable.

## Evidence

- `src/alexdoor_xas/dataset/validate.py`
- `src/alexdoor_xas/policies/act/checkpoint.py`
- `src/alexdoor_xas/policies/diffusion/checkpoint.py`
- `src/alexdoor_xas/cluster_sweep/transfer.py`
- `src/alexdoor_xas/cluster_sweep/returns.py`
- `src/alexdoor_xas/eval/phase3_unified.py`
- `outputs/curated/phase3_unified_evaluation/`

See [[topics/provenance-and-artifact-lifecycle|Provenance and Artifact Lifecycle]].

## Version Notes

- 2026-07-08 — Robot/calibration/dataset/checkpoint bindings became one
  fail-closed Alex V2 chain.
- 2026-07-16 to 2026-07-18 — Exact cluster return and curated unified-evaluation
  evidence completed the lifecycle.
