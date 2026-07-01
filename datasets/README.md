# datasets/

Reusable, exported episode datasets — the output of the deterministic data engine
(Phase 2+) and the input to learned baselines (Phase 3+). Kept separate from
per-run results (`../outputs/`) because a dataset is shared across many runs.

**Gitignored:** everything here except this file. Large data never lands in git.

## Layout convention

```
datasets/<task>/<action_space>/<version>/
    episodes.<ext>     # episodes conforming to docs/episode_schema.md
    meta.json          # counts, seeds, generator + git commit, creation time
```

- `<task>` — e.g. `door_push`
- `<action_space>` — a tag from [docs/action_spaces.md](../docs/action_spaces.md),
  e.g. `A2_ee_delta` (a dataset holds one action space; re-export produces siblings)
- `<version>` — `v0`, `v1`, … (bump on any generation change)

Example: `datasets/door_push/A4_obj_centric_chunk/v0/`.

The concrete file format is chosen in Phase 2; the schema is container-agnostic.
