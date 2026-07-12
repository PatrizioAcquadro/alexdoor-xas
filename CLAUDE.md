# CLAUDE.md — AlexDoor-XAS

Cross-action-space VLA learning for humanoid door manipulation (IHMC Alex, Isaac
Lab). Source of truth for scope/phases: `docs/PROJECT_GUIDELINES.md`. Currently
at **local post-Phase 3.3 stabilization closeout** (the historical documentation
path is `docs/phase3_4_local_stabilization.md`). The existing local software and
protocol gates pass, but cluster authorization requires regenerated safety
evidence with `safety_readiness: PASS`. The only robot lineage is the static
`~/Desktop/Alex/urdf/alex_v2.urdf`, calibration-gated through
`AlexDoor-DoorPush-AlexV2-v0`. The official `door_push_alex_v2/v2_pose`
dataset contains 50 episodes across five door poses. ACT and Diffusion were
retrained for A2/A3 with Alex V2-fingerprinted checkpoints and passed their
training and closed-loop rollout gates. VLA (project Phase 4) has not started.

## Launcher policy (never bare `python3` for Isaac code)

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script>   # Isaac Lab scripts + pytest
```

- Isaac Sim 6.0.1 at `/home/pacquadr/isaacsim`; Isaac Lab `release/3.0.0-beta2`
  at `/home/pacquadr/IsaacLab`. Never install/upgrade the sim stack.
- Scripts that need Kit create `AppLauncher` **before** any Isaac import.
- `ruff` runs standalone (`~/.local/bin/ruff`, line length 100, rules E/F/I/UP/B).

## Verification gates (all must PASS; run via the launcher above)

```bash
isaaclab.sh -p -m pytest -q                                             # pure tests, no Kit
isaaclab.sh -p scripts/check_env.py                                     # readiness, no Kit
isaaclab.sh -p scripts/verify_assets.py --viz none --device cpu --steps 1
isaaclab.sh -p scripts/verify_door_task_scene.py --viz none --device cpu --steps 100
isaaclab.sh -p scripts/verify_door_env.py --viz none --device cpu --steps 100
isaaclab.sh -p scripts/verify_scripted_baseline.py --viz none --device cpu   # Phase 2 gate
isaaclab.sh -p scripts/verify_alex_v2_door_baseline.py --viz none --device cpu  # Alex V2 calibration gate (sole writer of configs/alex_v2_door_calibration.v0.json)
isaaclab.sh -p scripts/verify_dataset_interface.py                      # Phase 3.0 gate, no Kit
isaaclab.sh -p scripts/verify_adapters.py --viz none --device cpu       # Phase 3.1 gate
isaaclab.sh -p scripts/verify_act_training.py                           # Phase 3.2 training gate, no Kit
isaaclab.sh -p scripts/verify_act_rollout.py --viz none --device cpu \
    --checkpoint-a2 <best.pt> --checkpoint-a3 <best.pt>                 # Phase 3.2 rollout gate (needs trained ckpts)
isaaclab.sh -p scripts/verify_diffusion_training.py                     # Phase 3.3 training gate, no Kit
isaaclab.sh -p scripts/verify_diffusion_rollout.py --viz none --device cpu \
    --checkpoint-a2 <best.pt> --checkpoint-a3 <best.pt>                 # Phase 3.3 rollout gate (needs trained ckpts)
isaaclab.sh -p scripts/verify_a2_a3_distinct.py --task door_push_alex_v2 --version v2_pose  # pose-separation gate, no Kit
```

Data engine CLI: `scripts/run_scripted_baseline.py --episodes N --randomized M`
(add `--robot alex_v2` for the calibrated V2 executor — fail-closed without the
validated calibration —, `--video --enable_cameras` for mp4s). Multi-pose
datasets: one `--no-export --door-pose-id Dk --door-yaw-deg … --door-offset-…`
run per pose of `configs/door_pose_plan_v2_pose.json`, merged once by
`scripts/export_merged_dataset.py --pose-plan … --experiment …` (single writer
of `datasets/door_push_alex_v2/<space>/v0`; enforces the 0–200 N force
admission policy; `scripts/verify_a2_a3_distinct.py --task door_push_alex_v2`
checks pose-induced A2/A3 separation). The v1 diff-IK windup scale blocker does
not apply to v2 (IK40 gains + per-solve joint-limit clamps; zero clamp ticks in
the calibration gate).
ACT: train `scripts/train_act.py dataset.space=A2_ee_delta` (no Kit; Hydra
`configs/act.yaml`; W&B via the `tracking/` scaffold, disabled by default),
eval `scripts/eval_act.py --viz none --device cpu rollout.checkpoint=<best.pt>`.
Diffusion: train `scripts/train_diffusion.py dataset.space=A2_ee_delta` (no
Kit; GPU by default — `train.device=cpu` for smoke runs; Hydra
`configs/diffusion.yaml`; needs `diffusers`, installed in the Isaac env), eval
`scripts/eval_diffusion.py --viz none --device cpu rollout.checkpoint=<best.pt>`
(pass `model.horizon=<H>` when the checkpoint's horizon differs from the yaml).
Local post-Phase 3.3 stabilization (started 2026-07-11): training/eval are **GPU-first**
(`train.device=cuda`, `rollout.policy_device=cuda`; the training gates pin
`train.device=cpu`; sim stays `--device cpu` per the frozen calibration
contract). Eval rows carry door-pose id/yaw/offset, `failure_label`, contact
ticks/source, force mean/max/p95 + impulse, and payloads carry
`dataset_provenance` (manifest fingerprint + splits) and policy
sampler/horizon metadata (fail-loud checkpoint-horizon check). Per-pose evals
need `rollout.door_pose_id` (pose-qualified `*_eval_<pose>.json`);
`scripts/summarize_smoke_eval.py` merges them and gates metadata coverage.
Smoke matrix `local_smoke_{act,diffusion}_{a2,a3}_n50_seed0` (obs preset
`core_door_pose`, W&B offline group `local_smoke_n50_gpu`): 36/36 per cell,
0 rejections; DDPM-100 seed-105 diagnostic passed (diagnostic only). Runner
refuses direct export from posed runs and detects env mid-episode auto-reset.

## Frozen contracts

- Env IDs: `AlexDoor-DoorTask-Direct-v0` (gate shell), `AlexDoor-DoorPush-Proxy-v0`
  (Phase 2 push task; action = 6-dim EE delta, clamps 0.02 m / 0.05 rad per tick),
  `AlexDoor-DoorPush-AlexV2-v0` (same action contract executed by the fixed-base
  Alex V2 right arm via position-mode differential IK on the collision-derived
  tool point + door-panel-filtered force sensing; adds passive hinge damping —
  the proxy hinge stays undamped). The V2 env is **calibration-gated**: it
  refuses to construct unless `configs/alex_v2_door_calibration.v0.json` has
  `status="validated"`, a matching fingerprint, and the exact runtime robot
  asset — only `scripts/verify_alex_v2_door_baseline.py` may write that file.
  The unregistered `DoorPushAlexV2CalibrationEnv` executes candidates.
- Action-space tags (`action/spaces.py`): `A1_joint_delta`, `A2_ee_delta`,
  `A3_obj_rel_ee_delta`, `A4_obj_centric_chunk`. Exports cover A2/A3/A4 for
  every robot; Alex episodes additionally export A1 as 29-wide full-body
  joint-position-target deltas (`target[t+1] - target[t]`; last diff closed by
  extras `final_joint_pos_target`). Proxy runs stay A1-less (no joints).
- Robot tags: `proxy_ee_sphere_v0` (dynamic gravity-free sphere, velocity-driven;
  **never kinematic** — kinematic-target writes sweep through the scene and
  destabilize the door articulation) and
  `alex_v2_fullbody_fixedbase_standard_forearm_v0` (static
  `~/Desktop/Alex/urdf/alex_v2.urdf`, sha256-pinned in
  `assets/alex_v2_manifest.py`; pelvis welded, right-arm IK on the
  collision-derived tool point of `RIGHT_GRIPPER_Z_LINK`; articulation cfg from
  `isaaclab_assets.robots.alex.make_alex_v2_cfg` — requires the IsaacLab
  checkout on branch `pacquadr/alex-v2-asset`, enforced by `check_env.py`).
  Right-arm PD = IK40 (`600/15` shoulders+elbow, `150/4` wrists,
  `door-alex-v2-right-arm-ik40-pd-v2`); gains are fingerprint inputs — every
  episode/checkpoint carries the runtime asset fingerprint and
  `assert_checkpoint_runtime_compatible` fails closed on mismatch.
- A3 frame = hinge-anchored door frame: origin at the `Doorframe` body, +Z hinge
  axis; panel frame = door frame rotated by the hinge angle (`action/frames.py`).
- Quaternions are **(x, y, z, w)** everywhere (this Isaac Lab build's data layout).
- Episode container: HDF5 + JSON sidecar per episode (`recording/writer.py`,
  `schema_version = "phase2.v1"` — additive superset of v0: optional proprio
  `joint_pos/joint_vel/joint_pos_target` and contact `sensed/force_n` keys; v0
  files stay readable); A4 datasets are JSON lines. A dataset
  `datasets/<task>/<space>/<version>/` dir is one generation pass (re-export
  replaces); Alex V2 runs use task `door_push_alex_v2` (current version
  `v2_pose`: 50 episodes, 5 probe-gated door poses,
  `configs/door_pose_plan_v2_pose.json`) so proxy datasets are never replaced.
  Dataset force admission: every recorded control-tick force sample must be
  finite and within [0, 200] N — hard sanity error at generation, re-checked by
  `export_merged_dataset.py` before the merged export.
- Timing: `sim.dt = 1/120`, decimation 2 → `control_dt = 1/60`. Success = final
  door angle ≥ π/4 (configurable in `DataEngineCfg`).
- Dataset/model interface (Phase 3.0, `dataset/`, `docs/dataset_interface.md`):
  models consume episodes only through `EpisodeDataset`/`A4ChunkDataset` — never
  raw HDF5. Frozen obs presets `core` (9-dim, every episode), `core_contact`,
  `alex_full` (Alex episodes with proprio/contact only; the proxy has neither).
  Splits live at `datasets/<task>/splits/<version>.json`
  and are **shared across action spaces** (same episode ids per pass); norm stats
  (train-split only, std floored 1e-8 — A2/A3 rotation dims are constant zero) at
  `<version>/norm_stats.json`; both regenerate on re-export. Chunk samples are
  ACT-style (obs@t + H actions + `is_pad`). A4 stays symbolic per-phase chunks
  (12-dim numeric encoding available; hinge delta = controller intent, not
  achieved motion).
- Adapter-v1 (Phase 3.1, `adapters/`, `docs/adapters.md`): predicted actions
  execute only through the adapter layer — `A2Adapter` (clamp + measured Alex
  workspace sphere, warn-level joint-limit flags), `A3Adapter` (object-frame
  trust check + frozen A2<->A3 conversion), `A4Adapter` (chunk validation,
  guarded approach/contact/push planning, closed-loop execution with missed-
  contact and push-stall detection). Every command logs
  accepted/corrected/rejected; `A4ExecutionResult` reports requested vs
  achieved hinge delta (chunk *logs* stay intent labels — never relabel them).
  `rollout_chunks` (`adapters/rollout.py`) is the shared closed-loop driver
  for ACT/Diffusion/VLA eval. `A4_PHASE_VOCAB` lives in `action/spaces.py`
  (dataset re-exports it). Adapters never import policies: panel geometry
  (`DoorPanelGeometry`) and the vocab are pinned by unit tests instead. V2
  workspace limits are never hardcoded: `alex_v2_limits(calibration,
  workspace_center_w=…)` builds the sphere from the validated calibration's
  `reach_shell_m` (0.2/0.8 m) plus a live-measured shoulder center
  (`env.shoulder_position_world_m()`); `limits_for_robot` requires both for
  the v2 tag.
- ACT baseline (Phase 3.2, `policies/act/`, `docs/act.md`): state-only ACT CVAE
  (~1.3M params; single obs token, masked L1 + β·KL, z=0 at test). Checkpoints
  are self-contained (weights + resolved config + norm stats) and
  `torch.load(weights_only=True)`-safe. `ActPolicy` clips normalized obs to ±10
  (floored-std dims); live obs readers exist for `core`/`core_contact` only.
  Closed-loop eval uses success-stop termination (`stop_on_hinge_angle`):
  rollouts end at the first chunk boundary past the success angle — post-task
  extrapolation is out of distribution and can knock the door shut. The A3
  dataset equals A2 numerically while door poses are unrandomized (door frame
  is world-axis-aligned).
- Shared chunk-policy helpers (Phase 3.3, `policies/common/`): obs readers
  (`build_env_obs`, `OBS_CLIP`), `stop_on_hinge_angle`, dataset plumbing
  (`load_policy_data`, batch factories with a pluggable `normalize=` hook),
  `open_loop_report` (stride param), and rollout-eval aggregation. ACT
  re-exports them (`ActData = PolicyData`, `load_act_data = load_policy_data`,
  …) so its public surface is unchanged; new policies import from common.
- Diffusion baseline (Phase 3.3, `policies/diffusion/`, `docs/diffusion.md`):
  state-only time-series diffusion transformer (~1.1M params, To=1, causal
  decoder over noisy action tokens; epsilon prediction, squaredcos T=100,
  `clip_sample=True`), `diffusers` schedulers (0.39.0, the one approved
  install into the Isaac env; `schedulers.py` is the single construction
  point). Actions are **min-max normalized to [-1, 1]** (never z-score — DDPM
  clips samples), constant rotation dims shift-to-zero without scaling, so
  denormalized |dpos| ≤ train extrema < the 0.02 m clamp by construction.
  EMA weights are what checkpoints ship (`alexdoor_xas.diffusion.v1`).
  Receding horizon: sample Tp=16, execute Ta=8 (`rollout.n_action_steps`).
  `DiffusionPolicy.seed(seed)` per rollout keeps sampling deterministic;
  fixed-block spread in evals is sampling variance across seeds, not physics.
  Best-checkpoint selection = seeded DDIM-10 sampled-chunk L1 (denoise MSE is
  a noisy selector). Real training runs are GPU (`train.device=cuda`
  default); the script hard-fails without CUDA rather than silently using
  CPU. Eval of a non-default-horizon checkpoint needs `model.horizon=<H>`
  (config-level Ta<=Tp check runs before the checkpoint loads). Phase 3.3
  results are pipeline validation only (22 episodes) — performance claims
  deferred; DDIM-10 beat DDPM-100 closed-loop (20/20 vs 19/20, seed-105
  sampler-noise failure), retest when the dataset scales.

## Isaac Lab 3.0.0-beta2 gotchas (hard-won)

- Articulation pose reads (`data.body_pos_w`, `data.root_pos_w`) return zeros in
  the DirectRLEnv context **for the USD-referenced door**; read static prim poses
  from the USD stage (`omni.usd` + `UsdGeom.XformCache`) instead. Joint state
  reads are fine. URDF-spawned articulations (Alex) read live body poses and
  jacobians correctly — but the stage keeps only the authored rest pose for
  dynamic bodies (no physics→USD sync headless), so never compare live reads
  against stage reads for moving bodies.
- The URDF importer nests link rigid bodies by kinematic chain under a
  `Geometry` scope, and `activate_contact_sensors` stops at the first rigid body
  (the root link) — apply the contact-report API to non-root links explicitly
  (`isaaclab.sim.schemas.activate_contact_sensors` on the link prim path).
  Filtered contact views: PhysX filter globs **prefix-match path strings**, so
  a body-path filter like `…/Door` also captures the sibling `…/Doorframe`
  collider ("expected 1, found 2") and `force_matrix_w` silently never builds
  (`contact_view.filter_count` AttributeError on `None`). Filter on the exact
  collision-shape prim instead (`…/Door/Cylinder_001`) — this is why Phase 2.5
  fell back to net force; v2 requires the filtered matrix and now gets it.
- High PD gains on `IdealPDActuatorCfg` (explicit software PD) are numerically
  unstable on low-inertia joints (wrists peg at their velocity limits within
  ticks); use `ImplicitActuatorCfg` for stiff IK tracking. Per-tick tracking
  ratio ≈ (stiffness/damping)·dt — keep k/c ≈ 40/s for ~50% per control tick.
  Hold-stability probes do NOT validate tracking: the SDK-derived V2 gains
  (k/c ≈ 0.33/s) held pose perfectly but tracked diff-IK at ~0.3%/tick — the
  arm crawled millimetres while every episode timed out. Measure tracking with
  a commanded delta, not just settle velocity, before freezing gains.
- Pose-mode (6-DoF) differential IK stalls near the arm's hanging pose
  (near-null joint swings, no net motion); use `command_type="position"` for
  translation-only tasks. The production Alex V2 standard URDF includes the
  primitive arm and gripper collision geometry used by the tool-frame gate.
- World-anchored USD joints need **explicitly authored** local frames
  (`assets/door_task.py` authors `FixDoorframe`'s world-side anchor); unauthored
  frames default to the world origin and drag the articulation there when the
  USD is referenced under an env namespace.
- Do not write cfg-default root poses for world-anchored articulations in
  `_reset_idx` — reset joint state only.
- Physics is deterministic per rendering mode; `--enable_cameras` runs differ
  numerically from headless runs of the same seed.
- Data accessors use `.torch` proxies (`data.joint_pos.torch`); writers are
  `write_*_to_sim_index(..., env_ids=...)`.

## Architecture rules

- Dependency direction (see `docs/architecture.md`): `action` → `policies/scripted`,
  `recording` → `eval` → `data_engine` → `scripts`; Phase 3.0 adds
  {`action`, `recording`, `eval`} → `dataset` → `scripts`; Phase 3.1 adds
  `action` → `adapters` → `scripts` (adapters never import policies); Phase 3.2
  adds {`action`, `dataset`} → `policies/act` → `scripts` (the ACT package never
  imports adapters or envs — scripts compose policy + adapter + env; its
  `config.py` is torch-free so it resolves before AppLauncher); Phase 3.3
  adds `policies/common` (policy-agnostic, ACT re-exports it) and
  {`action`, `dataset`, `policies/common`} → `policies/diffusion` → `scripts`
  (same rules; never imports `policies/act`; `config.py` also diffusers-free). The
  scripted controller, data engine, dataset interface, and adapters have
  **no Isaac imports** (env is duck-typed, h5py lazy, torch optional) — keep it
  that way so logic stays unit-testable; Isaac-dependent behavior is covered by
  gate scripts, not unit tests.
- Assets are referenced in place: scenes under `~/Desktop` (override
  `ALEXDOOR_ASSETS_ROOT`), the Alex V2 model at `~/Desktop/Alex` (override
  `ALEX_V2_ASSET_ROOT`, name-aligned with the IsaacLab module); paths only via
  `alexdoor_xas.paths`. The `calibration/` package (pure Python, fail-closed
  loaders) sits between `assets` and the v2 env/adapters:
  {`assets`} → `calibration` → {`envs`, `adapters`, `policies/scripted`} —
  adapters still never import policies or Isaac.
- No empty future packages; `safety/`, `observation/` start when their phase
  needs them (`adapters/` started in Phase 3.1).
