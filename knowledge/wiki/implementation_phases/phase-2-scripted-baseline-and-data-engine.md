# Phase 2 — Scripted Baseline and Data Engine

## Objective

Create a deterministic door-push generator and matched A1-A4 episode exports.

## Focus

### Subphase 2.1 — Scripted execution and recording

#### Implementation

The scripted controller approaches, contacts, pushes, and releases the door. The recorder aligns pre-action observation with requested/applied action and stores the terminal response separately. One physical episode can be exported as matched A1, A2, A3, and A4 products.

New writes use `phase2.v2`: outcomes retain success, final door angle, step count, notes, factual termination reason, and environment termination/truncation flags. Each A1-A3 episode is one HDF5 file. Duplicate observation references, unused clamp flags, interpreted failure labels, and per-episode sidecars are absent. Active `phase2.v1` and legacy A4 records remain readable without rewriting.

Scripted-run staging moved from `outputs/` to `~/.cache/alexdoor-xas/scripted_runs/`; reusable exports remain under `datasets/`.

The maintained runner uses count-and-seed planning only. Retired scale-candidate and paired-master publication paths are not part of the current engine.

Each run writes compact factual episode and aggregate metrics, a compact joint/contact sanity summary, two door-angle plots, and one Markdown report. The sanity gate always rejects non-finite or negative force and force above 200 N, including the terminal response to the final action. Legacy episodes without that terminal sample remain readable.

#### Key Decisions and Problems

- Door-relative frames are explicit and hinge anchored.
- Representation products share physical identity and outcome.
- Generated data and staging stay outside Git.
- Environment stop information remains factual; failure interpretation is not part of the schema.

#### Tests

Pure tests cover controller transitions, `phase2.v2` recording, active v1 reads, matched export, compact metrics and reports, factual termination semantics, and the fixed force gate. The Isaac scripted-baseline gate verifies the same artifacts and A1-A4 export on `cuda:0`.

## Version Notes

- 2026-08-12 — Reduced recording to non-duplicated state, action, controller phase, outcome, and active v1 compatibility.
- 2026-08-12 — Reduced evaluation to direct module imports, compact metrics and force evidence, one fixed 200 N admission gate, essential tests, and unchanged report/plot outputs.
- 2026-08-12 — Reduced the engine to maintained count-and-seed generation and current A1-A4 export contracts.
- 2026-08-12 — Introduced factual `phase2.v2`, retained legacy read compatibility, and moved scripted staging to the runtime cache.
- 2026-08-11 — The scripted baseline and data engine remain active.
