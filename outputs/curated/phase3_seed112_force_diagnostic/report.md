# ACT-A3-N50 Seed-112 Force Diagnostic

## Question

Why did the ACT-A3-N50 D0 randomized rollout at seed 112 cross the 200 N
simulation force-watch bound during the Phase 3 evaluation?

## Result

The exact replay reproduced success at tick 93 and a single 219.953 N peak at
tick 55. Two controlled runs changed only the initial door-frame X position:

| Case | Peak force | First contact / peak | >200 N ticks | Success | Corrections |
|---|---:|---:|---:|---:|---:|
| Exact replay | 219.953 N | 55 / 55 | 1 | tick 93 | 0 |
| X -1 mm | 66.002 N | 56 / 57 | 0 | tick 94 | 2 |
| X +1 mm | 86.442 N | 55 / 55 | 0 | tick 94 | 0 |

All cases succeeded and none produced an adapter rejection.

## Interpretation

The event is reproducible and locally sensitive to initial door-normal
position and contact-entry motion. This small diagnostic does not prove root
cause, validate a controller change, or establish hardware safety. The
original Phase 3 cell therefore remains `REVIEW_REQUIRED`.

Structured values are retained in `results.json`.
