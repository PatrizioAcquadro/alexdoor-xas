# datasets/

Reusable, exported episode datasets — the output of the deterministic data engine
(Phase 2+) and the input to learned baselines (Phase 3+). Kept separate from
per-run results (`../outputs/`) because a dataset is shared across many runs.

**Gitignored:** everything here except this file. Large data never lands in git.

## Layout convention

```
datasets/<task>/<action_space>/<version>/
    episode_<id8>.hdf5       # one episode per file (docs/episode_schema.md)
    episode_<id8>.meta.json  # human-readable meta + outcome sidecar
    episodes.jsonl           # A4 only: struct chunks as JSON lines
    meta.json                # counts, seeds, generator + git commit, creation time
    norm_stats.json          # Phase 3.0: train-split action/obs normalization stats
datasets/<task>/splits/<version>.json  # Phase 3.0: train/val/test episode ids,
                                       # shared across the task's action spaces
```

A version directory is **one generation pass**: re-exporting the same version
replaces it (`data_engine/export.py`). Bump `<version>` on any generation change.
Splits and norm stats describe one pass too: a re-export mints fresh episode
ids, so regenerate them with
`scripts/verify_dataset_interface.py --write-artifacts` when you intentionally
refresh official artifacts.

- `<task>` — `door_push` (Phase 2 proxy-sphere episodes) or `door_push_alex_v2`
  (Alex V2 episodes with force-sensed contact and joint proprio;
  a distinct task dir so Alex runs never replace the frozen proxy datasets)
- `<action_space>` — a tag from [docs/action_spaces.md](../docs/action_spaces.md),
  e.g. `A2_ee_delta` (a dataset holds one action space; re-export produces siblings)
- `<version>` — `v0`, `v1`, … (bump on any generation change)

Example: `datasets/door_push/A4_obj_centric_chunk/v0/`.

Phase 2 chose HDF5 + JSON sidecar per episode (A4: JSON lines); generate with
`scripts/run_scripted_baseline.py` (`--robot alex_v2` for Alex V2 episodes). Consume
via `src/alexdoor_xas/dataset/` (docs/dataset_interface.md) — never read the
HDF5 layout directly from model code.

Phase 3.0 validation is fail-closed for malformed metadata, action tensor ranks,
timing/control-rate mismatches, contact flags/sources, A3 door-frame action
relabels, A4 outcome/chunk fields, matched action-space provenance, and stale
normalization stats. `scripts/verify_dataset_interface.py` is read-only by
default: it writes temporary splits and stats for validation, then discards
them. Passing `--write-artifacts` explicitly writes the shared split file and
per-space `norm_stats.json` into this tree.
