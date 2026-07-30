# Decision — One Scale Master with Nested Views

## Context

Independent data generation at N50, N100, N250, and N500 would change episode
composition, holdouts, and normalization alongside training size. It would
also allow A2 and A3 to diverge in physical trajectory identity.

## Decision

Generate one deterministic 550-episode physical master, export A2 and A3 as an
atomic matched pair, and define N50, N100, N250, and N500 as deterministic
nested training subsets. Keep one fixed 25-episode validation set and one fixed
25-episode test set. Compute separate training-only normalization for every
action-space/view pair.

Bind every view to ordered IDs and a selection SHA-256. Retain predeclared
surplus successful candidates as `NOT_NEEDED_OVERDRAW` so inclusion is not
chosen after observing performance.

## Consequences

- Larger views contain all episodes from smaller views, so data scale changes
  without replacing examples.
- Validation/test comparison remains fixed across sizes and representations.
- Eight normalization artifacts are required because training membership and
  action representation both matter.
- Results remain specific to one simulated door family and five poses; more
  episodes do not increase modality or task diversity.
- Any membership or ordering drift invalidates the named view.

## Evidence

- `configs/door_pose_plan_v3_scale.json`
- `scripts/build_scale_dataset.py`
- `src/alexdoor_xas/dataset/views.py`
- `src/alexdoor_xas/data_engine/export.py`
- `tests/test_dataset_views.py`
- `tests/test_scale_dataset.py`

See [[topics/episode-and-dataset-contracts|Episode and Dataset Contracts]].

## Version Notes

- 2026-07-15 — The 550-episode paired master, fixed holdouts, and four nested
  training views were published.
- 2026-07-16 — All sixteen cluster cells consumed the immutable view and
  normalization identities.
