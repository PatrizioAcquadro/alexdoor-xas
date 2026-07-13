# System Architecture

AlexDoor-XAS keeps observations, policies, action representations, adapters,
and safety/logging separate so action spaces can be compared without changing
the task definition.

```text
observation -> policy -> A1/A2/A3/A4 -> adapter -> environment
                                      |             |
                                      +-> decisions +-> episode/evaluation evidence
```

## Package map and dependency direction

- `assets/`, `paths.py` — local Alex V2 and door-scene discovery and fixture
  construction.
- `envs/door_task/` — task shell, proxy executor, calibrated Alex V2 executor,
  and environment registration.
- `action/` — canonical action tags, A4 structures, and A2/A3 frame conversion.
- `policies/scripted/` — deterministic door-relative finite-state controller.
- `policies/act/`, `policies/diffusion/` — state-only learned chunk policies.
- `policies/common/` — shared observation, data, and rollout-evaluation helpers.
- `adapters/` — model-independent validation, correction, rejection, and
  closed-loop execution.
- `recording/`, `data_engine/` — episode capture, generation, export, and run
  orchestration.
- `dataset/` — fail-closed loading, validation, splits, normalization, and
  chunk sampling.
- `eval/` — metrics, failure labels, plots, reports, and safety diagnostics.
- `cluster_pilot/` — non-Isaac transfer, preflight, Slurm, publication, and
  return-manifest contracts.

Core modules remain dependency-light and Isaac-free where possible. Scripts
compose policies, adapters, and environments. Policies never import adapters;
adapters never import policies. Isaac runtime imports stay at environment and
script boundaries after `AppLauncher` initialization.

## Environments and robot contracts

| Environment | Purpose | Execution |
|---|---|---|
| `AlexDoor-DoorTask-Direct-v0` | reset/step verification shell | no learned policy |
| `AlexDoor-DoorPush-Proxy-v0` | scripted proxy baseline | gravity-free dynamic EE sphere |
| `AlexDoor-DoorPush-AlexV2-v0` | calibrated benchmark | fixed-base Alex V2 right-arm differential IK |

The sole robot asset is `~/Desktop/Alex/urdf/alex_v2.urdf`, tagged
`alex_v2_fullbody_fixedbase_standard_forearm_v0` and SHA-256 fingerprinted.
The Alex V2 environment constructs only when
`configs/alex_v2_door_calibration.v0.json` is validated and matches the runtime
asset. `scripts/verify_alex_v2_door_baseline.py` is the only calibration writer.

The benchmark uses a collision-derived tool point on
`RIGHT_GRIPPER_Z_LINK`, position-mode differential IK, joint-target
anti-windup clamps, and door-panel-filtered force sensing. Simulator timing is
`sim.dt = 1/120` with decimation 2 (`control_dt = 1/60`). The shared A2 command
is six-dimensional Cartesian delta motion, clamped to 0.02 m and 0.05 rad per
control tick. Success is final hinge angle at least pi/4.

## Action representations

| ID | Canonical tag | Operational form | Current role |
|---|---|---|---|
| A1 | `A1_joint_delta` | Alex full-body joint-position-target deltas | robot-specific baseline; Alex only |
| A2 | `A2_ee_delta` | 6-D world-frame EE delta | practical learned baseline |
| A3 | `A3_obj_rel_ee_delta` | 6-D hinge-anchored door-frame EE delta | transfer-oriented baseline |
| A4 | `A4_obj_centric_chunk` | phase, panel contact target, intended hinge delta, duration | symbolic object-centric representation |

A3 uses the static hinge-anchored `Doorframe` pose with +Z along the hinge.
Quaternions are `(x, y, z, w)`. A4 logs controller intent; achieved hinge
motion is reported by execution results and never written back into the label.

Proxy episodes export A2/A3/A4. Alex episodes export A1/A2/A3/A4; only the six
right-arm IK joints move in A1 while held joints carry zero deltas.

## Adapter-v1 and safety

- `A2Adapter` checks shape and finiteness, applies per-tick clamps, enforces the
  calibrated workspace, and records joint-limit warnings.
- `A3Adapter` validates the object frame, converts into A2, and delegates to
  the same execution checks.
- `A4Adapter` validates door geometry and push semantics, creates guarded
  approach/contact/push stages, and detects missed contact, stalls, and stage
  timeouts.
- `rollout_chunks` is the shared ACT/Diffusion/A4 execution driver. Invalid
  simulator state terminates as `invalid_simulator_state` before a command is
  adapted.

Every command produces an accepted, corrected, or rejected decision with the
requested and applied values. The environment clamps remain a final backstop.
Acceleration limits, general self/environment collision queries, and slip
detection are not implemented.

## Policies

- The scripted controller is deterministic and door-relative. It drives data
  generation and reference rollouts.
- ACT is a state-only CVAE chunk model. Checkpoints include weights, resolved
  configuration, normalization statistics, and dataset/split provenance.
- Diffusion Policy is a state-only causal diffusion transformer with EMA
  weights, min-max action normalization, and DDPM/DDIM sampling. The local
  primary protocol uses DDIM-10 with prediction horizon 16 and eight executed
  actions per replan.
- Live learned-policy observations currently support low-dimensional presets;
  camera/VLA observation pipelines are not implemented.

Both learned policies are evaluated closed-loop through adapter-v1 and use
per-tick first-crossing success termination.

## Episode and dataset contract

An episode contains immutable metadata, per-control-tick steps, an outcome,
and additive extras. The current container schema is `phase2.v1`:

- one HDF5 file plus a human-readable JSON sidecar per A1/A2/A3 episode;
- JSON Lines for variable-length A4 chunks;
- pre-action observations and force samples per tick;
- `terminal_contact` for the response to the final executed action;
- Alex joint state/targets, calibration fingerprint, contact provenance, and
  start-pose-settle evidence when applicable.

Reusable data lives at `datasets/<task>/<action_space>/<version>/`. Models
consume it only through `EpisodeDataset` or `A4ChunkDataset`, never through raw
HDF5 keys. Splits are shared across action spaces from the same generation
pass. Normalization statistics use only the training split and bind both the
source-master fingerprint and the action-export fingerprint, plus the action
space and observation preset. Re-exporting a version replaces that generation
and requires regenerated splits/statistics.

The stabilization Alex dataset is `door_push_alex_v2/v2_pose`: 50 episodes over
five door poses with a grouped, pose-stratified 38/6/6 split. Dataset generation
and merged export reject non-finite force or any sample outside the unchanged
0–200 N admission range.

The scale-sweep contract adds a separate physical master version,
`v3_scale_master`, without replacing `v2_pose`. Its local official publication is a
paired A2/A3 payload built once from 550 randomized source episodes: exactly
110 safe, successful, trajectory-content-distinct episodes for each of D0–D4.
Source and overdraw seeds occupy explicit disjoint namespaces, and every
candidate records its admission decision and any replacement relationship.
The A2 and A3 exports retain identical episode identities but must be
numerically distinct.

Four logical views select data from that one physical master:
`v3_scale_n50`, `v3_scale_n100`, `v3_scale_n250`, and `v3_scale_n500`.
Validation and test are fixed at 25 episodes each, balanced five per pose.
Training is a balanced, strictly nested prefix of 10, 20, 50, or 100 episodes
per pose. Views are shared across action spaces and fingerprint the master,
selection seed, split membership, and content groups. Each action-space/view
pair owns one train-only normalization artifact, for eight total. Scale views
carry two distinct provenance values: the common source-master fingerprint and
the action-specific A2 or A3 export fingerprint. A view checkpoint binds both,
the view fingerprint, exact split IDs, norm file hash and semantic fingerprint,
action space, observation preset, source commit, and the canonical SHA-256 of
the complete resolved training config. Legacy version-only checkpoints and
`v2_pose` loading remain readable through the unchanged path.

## Artifacts and provenance

- `datasets/` holds reusable exported episodes; see
  [`../datasets/README.md`](../datasets/README.md).
- `outputs/` holds per-run metrics, plots, videos, checkpoints, logs, and
  captures; see [`../outputs/README.md`](../outputs/README.md).
- `outputs/curated/` is the only location intended for small tracked evidence.

Generated data and raw run artifacts remain ignored. Every scientific result
must bind its source commit, dataset fingerprint, observation preset, split,
configuration, checkpoint, seed protocol, and evaluation protocol.

The full-sweep orchestration lives in `cluster_sweep/`. Its versioned config
defines a stable 16-cell array: ACT then Diffusion across A2/A3 for each nested
training view, seed 0, normal non-pilot epochs, offline W&B, one visible GPU,
and no distributed or Isaac runtime. One authoritative cell resolver supplies
the renderer, trainer, checkpoint provenance, and return verifier; every
returned durable config must equal that exact resolved cell and match its
checkpoint hash. Transfer and return manifests are exact SHA-256 inventories.
Scratch and durable results are isolated by numeric array job ID, task ID, and
run ID; only one complete 16-cell attempt can be returned.
