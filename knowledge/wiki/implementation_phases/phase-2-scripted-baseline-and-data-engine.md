# Phase 2 — Scripted Baseline and Data Engine

## Objective

Create a deterministic door-push generator and matched A1-A4 episode exports.

## Focus

### Subphase 2.1 — Scripted execution and recording

#### Implementation

The scripted controller approaches, contacts, pushes, and releases the door. The recorder aligns pre-action observation with requested/applied action and stores the terminal response separately. One physical episode can be exported as matched A1, A2, A3, and A4 products.

New writes use `phase2.v2`: outcomes retain success, final door angle, step count, notes, factual termination reason, and environment termination/truncation flags. The current contract does not produce interpreted failure labels. Legacy `phase2.v0/v1` and A4 records remain readable without rewriting; obsolete failure labels are discarded and unavailable termination fields are marked unknown/not recorded.

Scripted-run staging moved from `outputs/` to `~/.cache/alexdoor-xas/scripted_runs/`; reusable exports remain under `datasets/`.

#### Key Decisions and Problems

- Door-relative frames are explicit and hinge anchored.
- Representation products share physical identity and outcome.
- Generated data and staging stay outside Git.
- Environment stop information remains factual; failure interpretation is not part of the schema.

#### Tests

Pure tests cover controller transitions, `phase2.v2` recording, legacy v0/v1 reads, matched export, metrics, and factual termination semantics; Isaac smoke checks cover live execution.

## Version Notes

- 2026-08-12 — Introduced factual `phase2.v2`, retained legacy read compatibility, and moved scripted staging to the runtime cache.
- 2026-08-11 — The scripted baseline and data engine remain active.
