# Experiment — ACT-A3-N50 Seed-112 Force Diagnostic

> Historical experiment record. Its diagnostic runner and package are not maintained.

## Question

Was the 219.95 N event from the [[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]] reproducible and sensitive to a minimal initial door-position change?

## Method and Result

Exact replay reproduced success at tick 93 and a 219.953 N peak at tick 55. Changing only the initial door-frame X position by -1 mm and +1 mm reduced the peak to 66.002 N and 86.442 N. All three runs succeeded without adapter rejection.

## Interpretation and Limits

The event is reproducible and locally sensitive to door-normal position and contact-entry motion. The diagnostic does not establish root cause, validate a controller change, clear the original `REVIEW_REQUIRED` status, or provide hardware-safety evidence.

The former structured results remain recoverable from Git history through commit `7f1fc8c`.

## Version Notes

- 2026-08-13 — Reduced the page to the reproducible measurement and bounded interpretation.
