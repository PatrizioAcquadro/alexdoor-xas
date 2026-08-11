# Experiment — ACT-A3-N50 Seed-112 Force Diagnostic

## Question

Is the 219.95 N watch event from the
[[phase-3-unified-evaluation|Phase 3 Unified Evaluation]] reproducible and
sensitive to a minimal initial contact-position change?

## Method and result

The exact seed-112 replay reproduced success at tick 93 and a 219.953 N peak at
tick 55. Changing only the initial door-frame X position by -1 mm and +1 mm
reduced the peak to 66.002 N and 86.442 N, respectively. All runs succeeded
and none produced an adapter rejection.

## Interpretation

The event is reproducible and locally sensitive to initial door-normal
position and contact-entry motion. The diagnostic does not prove root cause,
validate a controller change, clear the original review status, or establish
hardware safety.

The retained evidence is:

- `outputs/curated/phase3_seed112_force_diagnostic/report.md`
- `outputs/curated/phase3_seed112_force_diagnostic/results.json`

## Version Notes

- 2026-08-11 — Removed provenance and checksum packaging while preserving the
  measured cases and bounded interpretation.
