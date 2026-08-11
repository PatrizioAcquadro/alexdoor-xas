# outputs/

Ordinary run outputs are ignored by Git. Keep each run self-contained under
`outputs/<experiment>/<run_id>/` with optional `metrics/`, `plots/`, `videos/`,
`checkpoints/`, `logs/`, and `episodes/` subdirectories.

Reusable training datasets belong in `datasets/`, not here.

`outputs/curated/` is the only tracked evidence area. The retained Phase 3
packages are intentionally small:

- unified evaluation: `report.md` and `aggregate_summary.json`;
- seed-112 force diagnostic: `report.md` and `results.json`.

Raw rollouts, checkpoint copies, resolved plans, inventories, transfer
manifests, and checksums remain local or in Git history rather than becoming a
second archive inside the repository.
