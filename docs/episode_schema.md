# Episode Schema (minimal, Phase 1)

An **episode** is one door-interaction trial. This document defines the minimal,
forward-compatible shape of a logged episode so later phases (data engine,
imitation/diffusion/VLA training, evaluation) share one contract. It is a
**specification only** — no serialization code is written in Phase 1. The
recording implementation lands in Phase 2 (`src/alexdoor_xas/recording/`).

Design constraints:
- Actions are stored in an **explicit action space** (see [action_spaces.md](action_spaces.md)),
  so the same behavior can be re-exported into other spaces and compared under matched conditions.
- The schema mirrors the five roles (Observation / Policy / Action / Adapter / Safety+Logging).
- Keep it small; add fields only when a phase needs them.

## Structure

```
episode
├── meta            # fixed per episode
├── steps[]         # one entry per control tick
└── outcome         # fixed per episode, filled at the end
```

### `meta`
| field | type | notes |
|-------|------|-------|
| `episode_id` | str (uuid) | unique |
| `task` | str | e.g. `"door_push"` (first benchmark) |
| `action_space` | str | conditioning tag: `A1_joint_delta` … `A4_obj_centric_chunk` |
| `robot` | str | `"alex_v1_fullbody"` \| `"alex_v1_nub"` |
| `scene` | str | scene id / USD ref (e.g. `combinedScene`) |
| `policy` | str | producer: `scripted` \| `act` \| `diffusion` \| `vla` \| … |
| `seed` | int | RNG seed for reproducibility |
| `sim_dt` | float | physics step (s), e.g. `0.005` |
| `control_dt` | float | action rate (s), e.g. `0.02` |
| `chunk_len` | int | actions per policy call (1 for non-chunked) |
| `created_utc` | str | ISO-8601 timestamp |

### `steps[]` (per control tick)
| field | type | notes |
|-------|------|-------|
| `t` | float | seconds from episode start |
| `obs_ref` | dict | references/paths to observation tensors (images, depth) + inline low-dim state; keeps the record small |
| `action` | array | in the declared `action_space` (shape per [action_spaces.md](action_spaces.md)) |
| `proprio` | dict | robot joint pos/vel, base pose, end-effector pose(s) |
| `object_state` | dict | door angle, hinge/handle pose, panel frame |
| `contact` | dict | contact flags/forces relevant to the hand(s)/door |
| `safety` | dict | supervisor flags (limit hits, clamps, aborts) — see Adapter/Safety roles |

### `outcome`
| field | type | notes |
|-------|------|-------|
| `success` | bool | task-defined (e.g. door opened past a threshold) |
| `final_door_angle` | float | radians |
| `failure_label` | str \| null | categorical failure mode (Phase 2 defines the vocabulary) |
| `n_steps` | int | length of `steps` |
| `notes` | str | free-form |

## Storage (convention, implemented later)
- Reusable datasets: `datasets/<task>/<action_space>/<version>/` (see
  [../datasets/README.md](../datasets/README.md)).
- Per-run captures: `outputs/<experiment>/<run_id>/episodes/` (see
  [../outputs/README.md](../outputs/README.md)).
- Concrete container (HDF5 / parquet / npz + json sidecar) is chosen in Phase 2 to match the first
  learned baseline's loader; this schema is container-agnostic.
