# Phase 2 — Scripted Baseline and Data Engine

## Objective

Create a deterministic door-push generator and matched A1-A4 episode exports.

## Focus

### Subphase 2.1 — Scripted execution and recording

#### Implementation

The scripted controller approaches, contacts, pushes, and releases the door.
The recorder aligns pre-action observation with requested/applied action and
stores the terminal response separately. One physical episode can be exported
as matched A1, A2, A3, and A4 products.

#### Key Decisions and Problems

- Door-relative frames are explicit and hinge anchored.
- Representation products share physical identity and outcome.
- Generated data stays outside Git.

#### Tests

Pure tests cover controller transitions, recording semantics, matched export,
metrics, and failure labels; Isaac smoke checks cover live execution.

## Version Notes

- 2026-08-11 — The scripted baseline and data engine remain active unchanged.
