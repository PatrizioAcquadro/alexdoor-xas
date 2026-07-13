# outputs/

Per-run artifacts. Everything a single run produces lives under one folder so a
run is easy to reproduce, compare, and delete. Reusable training data does **not**
go here - it lives in `../datasets/`.

**Gitignored by default:** raw run folders, episodes, videos, checkpoints, logs,
and generated metrics/plots/reports. Keep these local unless a result has been
deliberately promoted for review.

**Curated evidence:** small, stable artifacts may be committed under
`outputs/curated/`. Use this only for review material such as `*.md`, `*.json`,
`*.csv`, `*.png`, `*.svg`, or `*.txt`. Do not place HDF5 episodes, videos,
checkpoints, or full raw runs there.

## Layout convention

```
outputs/<experiment>/<run_id>/
    metrics/       # scalar/tabular results (json, csv)
    plots/         # figures
    videos/        # rollouts / renders
    checkpoints/   # model weights
    logs/          # stdout, config snapshot, git commit
    episodes/      # trial captures for this run (see docs/architecture.md)
```

- `<experiment>` — a named comparison, e.g. `A2_vs_A4_door_push`
- `<run_id>` — a unique run, e.g. `2026-07-01_seed0` (timestamp + variant/seed)

Grouping by experiment → run makes action-space comparisons (the project's core
question) straightforward: sibling runs under one experiment share task and
evaluation protocol and differ only in the variable under study.
