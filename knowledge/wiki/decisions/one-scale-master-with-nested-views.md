# Decision — One Scale Master with Nested Views

## Context

Independent N50, N100, N250, and N500 datasets would change episode
composition and holdouts alongside training size.

## Decision

Use one 550-episode physical master with matched A2/A3 exports. Retain nested
N50, N100, N250, and N500 train memberships with fixed 25-episode validation
and test sets. Keep separate train-only normalization for every action-space
and view pair.

Current code loads these retained split files directly. It does not maintain a
view-generation or sweep-publication workflow.

## Consequences

- Larger views contain all training episodes from smaller views.
- Validation and test membership stay fixed across sizes and representations.
- Results remain limited to one simulated door family and five poses.
- Split membership and recomputed normalization, rather than administrative
  fingerprints, define current validity.

See [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

## Version Notes

- 2026-08-11 — Retained the scientific view contract while retiring its
  generation and publication infrastructure.
