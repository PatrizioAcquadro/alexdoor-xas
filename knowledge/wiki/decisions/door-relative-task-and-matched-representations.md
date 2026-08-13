# Decision — Door-Relative Task and Matched Representations

## Context

Comparing action representations is meaningful only when task geometry, physical experience, data membership, and evaluation conditions remain aligned.

## Decision

Represent the door task around an explicit hinge frame and derive matched A1-A4 products from the same physical episode. A2 uses world-frame end-effector deltas; A3 uses a static hinge-anchored frame; A4 uses contact targets in the moving panel frame.

Use only the D0-D4 registry. Door variants rotate and translate the same task around the hinge rather than introducing independent scene definitions. Runtime APIs reject other pose IDs.

Matched A2/A3 products share physical outcome, episode identity, split, pose distribution, and evaluation seeds. Their action arrays and train-only normalization remain representation-specific.

## Consequences

- A3 must be validated and transformed before A2 execution.
- A4 requires guarded staged execution rather than direct application.
- A changed physical generation pass requires a new dataset version, shared split, and regenerated normalization.
- The design reduces major task-distribution confounds but does not prove perfect causal isolation of representation.

Completed pose-plan, dataset-publication, and unified-matrix orchestration are historical workflows and are not required by this active decision.

## Evidence

- `src/alexdoor_xas/action/frames.py`
- `src/alexdoor_xas/assets/door_scene.py`
- `src/alexdoor_xas/data_engine/export.py`
- `scripts/verify_dataset_interface.py`

## Version Notes

- 2026-08-13 — Limited the active decision to hinge-relative geometry, D0-D4, and matched physical identity across representations.
