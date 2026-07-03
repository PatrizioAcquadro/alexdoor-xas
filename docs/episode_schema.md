# Episode Schema

An **episode** is one door-interaction trial. This document defines the minimal,
forward-compatible shape of a logged episode so later phases (data engine,
imitation/diffusion/VLA training, evaluation) share one contract. The recording
implementation lives in `src/alexdoor_xas/recording/` (Phase 2):
`EpisodeBuffer`/`EpisodeMeta`/`EpisodeStep`/`EpisodeOutcome` mirror the tables
below, and `write_episode`/`read_episode` round-trip the container.

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
| `robot` | str | `"alex_v1_fullbody"` \| `"alex_v1_nub"` \| `"proxy_ee_sphere_v0"` (Phase 2 proxy end-effector — a velocity-driven sphere standing in for an Alex hand) |
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
| `failure_label` | str \| null | categorical failure mode; vocabulary in `src/alexdoor_xas/eval/failures.py` (`non_finite_state`, `phase_timeout_<phase>`, `env_truncated_before_completion`, `insufficient_final_angle`) |
| `n_steps` | int | length of `steps` |
| `notes` | str | free-form |

## Storage (Phase 2 container choice)
- Reusable datasets: `datasets/<task>/<action_space>/<version>/` (see
  [../datasets/README.md](../datasets/README.md)).
- Per-run captures: `outputs/<experiment>/<run_id>/episodes/` (see
  [../outputs/README.md](../outputs/README.md)).
- **Container (chosen in Phase 2): one HDF5 file per episode + a JSON sidecar**
  (`episode_<id8>.hdf5` + `episode_<id8>.meta.json`), written by
  `recording/writer.py`. HDF5 keeps episodes loadable by ACT/robomimic-style
  dataloaders in Phase 3; the sidecar keeps meta/outcome human-inspectable.
  A4 datasets are JSON lines (`episodes.jsonl`) since chunks are struct/variable.
- Phase 2 notes: `obs_ref` is inline low-dim state (no image tensors yet);
  `contact` carries `source: "inferred_geometric"` (no force sensing);
  `safety` records the per-tick clamp flags and the controller phase; episode
  `extras` hold the recorded door frame, per-step door-frame (A3) actions, the
  A4 chunk log, and the sampled variation.
- **`phase2.v1` (additive superset of v0; v0 files stay readable, the reader
  never branches on the version):** episodes from force-sensing robot envs
  (robot tag `alex_v1_fullbody_fixedbase_v0`) additionally record
  - `proprio.joint_pos` / `proprio.joint_vel` / `proprio.joint_pos_target`
    (29-wide full-body joint state + applied targets; the A1 dataset is
    exported from these as joint-position-target deltas — see
    [action_spaces.md](action_spaces.md)),
  - `contact.sensed` (force-sensor flag, threshold 1 N), `contact.force_n`
    (net EE contact force norm), with
    `contact.source = "force_sensor+geometric"` — `contact.inferred` keeps the
    geometric value as a recorded fallback,
  - extras `joint_names` (29 names), `arm_joint_ids` (the 6 IK joints),
    `final_joint_pos_target` (post-loop applied target, closes the A1 diff),
    and `joint_pos_limits` (J, 2) / `joint_vel_limits` (J,) (Isaac-reported
    limits, consumed by the rollout sanity checks in `eval/sanity.py`).
