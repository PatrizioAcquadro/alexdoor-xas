# Decision — Door-Relative Task and Matched Representations

## Context

The project aims to compare action representation rather than compare different
tasks or independently sampled trajectory distributions. World-frame and
object-relative actions can appear different simply because generation,
initial pose, or outcome differs.

## Decision

Use one isolated hinge-anchored door task and derive A1–A4 products from the
same physical episode identity. A2 is world-frame end-effector delta; A3 is the
equivalent delta in a static hinge-anchored door frame; A4 contact targets are
in the moving panel frame. Matched A2/A3 products share physical outcome,
episode identity, pose allocation, split, and evaluation seeds while retaining
distinct action arrays.

The door-frame origin and +Z hinge axis are explicit. Door pose variants pivot
around the hinge instead of redefining the task geometry independently.

## Consequences

- Representation comparisons have less task-distribution confounding.
- A3 must be transformed and validated before A2 execution; frame identity is
  part of the data and adapter contract.
- A4 needs guarded staged execution rather than direct simulator application.
- Changed physical generation requires a new dataset version and regenerated
  matched products.
- The design does not prove causal isolation of representation from every
  learning interaction; it provides a controlled benchmark contract.

## Evidence

- `src/alexdoor_xas/action/spaces.py`
- `src/alexdoor_xas/action/frames.py`
- `src/alexdoor_xas/data_engine/export.py`
- `src/alexdoor_xas/assets/door_task.py`
- `tests/test_action_spaces.py`
- `scripts/verify_dataset_interface.py`

See [[topics/action-representations-and-adapters|Action Representations and Adapters]]
and [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

## Version Notes

- 2026-08-12 — The A2/A3 distinguishability evidence moved into the canonical
  dataset interface gate.

- 2026-07-03 — Matched representation export and hinge-relative frame semantics
  were established.
- 2026-07-15 — The paired scale master and nested views applied the decision to
  the full data-scale study.
