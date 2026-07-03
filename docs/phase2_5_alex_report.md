# Phase 2.5 report — Alex fixed-base executor + force contact sensing

**Status: complete (2026-07-02).** Extends the frozen Phase 2 proxy baseline
(`docs/phase2_report.md`) with (a) fixes for all issues raised in the external
Phase 2 review, (b) a new env where the **IHMC Alex humanoid physically opens
the door** with its right arm, and (c) **force-based contact sensing** replacing
geometric inference as the primary contact signal. The proxy env, its gate, and
its datasets are untouched (regression baseline).

## 1. External review fixes (all claims verified true)

| Issue | Fix |
| --- | --- |
| `run_baseline` accumulated stale episode files on same-`run_id` reruns (reference dir held 16 pairs vs the report's 8) | `_fresh_run_dir` removes the run-owned artifacts (`episodes/videos/metrics/plots/logs`, `report.md`) before writing; targeted, so sibling `gate_datasets` survive. Regression test + double-gate-run verification (exactly 4 files after 2 runs). |
| `check_env.py` passed with `isaacsim` metadata MISSING and never pinned versions | New `-- provenance --` section: Isaac Sim `VERSION` file (FAIL if missing, WARN if not 6.0.1\*), Isaac Lab git branch/describe (WARN), `_isaac_sim` symlink target (FAIL on mismatch). Verified output: `6.0.1-rc.7+release.42383.32955d8d.gl`, branch `release/3.0.0-beta2`, symlink → `/home/pacquadr/isaacsim`. |
| `door_push_env_cfg.py` said "kinematic sphere proxy" while the config is dynamic/non-kinematic | Docstrings corrected (behavior unchanged). |
| Zero-step episodes crashed `write_episode` (`np.stack` on an empty list) | `EpisodeBuffer.stacked` and the writer handle N=0 (`(0, 6)` action dataset); round-trip regression test. |

## 2. `AlexDoor-DoorPush-Alex-v0`

Same frozen action contract as the proxy (6-dim A2 EE delta, clamps
0.02 m / 0.05 rad per tick, `sim.dt=1/120`, decimation 2, 9-term observation),
executed by Alex instead of a sphere:

- **Asset**: `alex_v1.rlModel_fullBody_robotAccurate_fullCollisions.urdf` — the
  default fullbody URDF has **no arm collision geometry**; the fullCollisions
  variant gives `RIGHT_GRIPPER_Z_LINK` a 0.05 m collision sphere (same radius
  as the proxy). `fix_base=True` (pelvis welded; no balance controller in scope).
  Standing pose `(-0.45, -0.38, 0.93)`, yaw π — the right arm (which hangs
  ~0.31 m to the robot's right) lines up with the push corridor.
- **Control**: per tick, the clamped EE delta becomes a relative
  **position-mode** `DifferentialIKController` command (`dls`); the solved
  targets for the 6 right-arm joints are tracked by stiff `ImplicitActuatorCfg`
  drives (shoulder/elbow 600/15, wrist 150/4). Rotation deltas are clamped and
  recorded but not actuated — the same rotation contract as the proxy sphere.
  All other joints (legs, torso, left arm, grippers) are position-held.
- **Geometry** (`alex_fixedbase_push_cfg()`): the measured arm workspace
  (shoulder at `(-0.43, -0.10, 1.39)`, full reach ~0.584 m) cannot reach the
  proxy's push point (80 % width, −0.30 m). The preset pushes at 35 % width,
  +0.15 m: the whole 0–50° push arc stays 0.31–0.52 m from the shoulder
  (probe-verified numerically). Randomized variations sample Alex-specific
  bounds (`ALEX_VARIATION_BOUNDS`).
- **Door**: the Alex env adds a passive hinge damper (4 N·m·s/rad, zero
  stiffness) — with the frozen frictionless hinge, a single arm tap sent the
  door coasting to full-open ahead of the pusher and the FSM never got a
  sustained push. The proxy env keeps the undamped hinge.
- **Engine compatibility**: the env exposes the same duck-typed accessors
  (`proxy_pose_w` now reads the gripper EE pose; `set_proxy_pose` runs a
  bounded IK-settle toward requested starts), so `generate.py`/`runner.py`
  needed only additive changes.

## 3. Force contact sensing

- Isaac Lab `ContactSensor` on `RIGHT_GRIPPER_Z_LINK`. Two backend gotchas:
  the URDF importer nests link rigid bodies so the spawner's
  `activate_contact_sensors` reached only the pelvis (the env applies the
  contact-report API to the EE link explicitly), and PhysX could not build the
  filtered gripper↔door pair view against the referenced door USD — the env
  records the EE's **net** contact force, unambiguous here because the gripper
  can only touch the door assembly.
- Per-step contact record became
  `{inferred, sensed, force_n, source="force_sensor+geometric"}`: `sensed` is
  the force flag (threshold 1 N), `inferred` keeps the geometric value as a
  recorded fallback. Proxy episodes are byte-identical to before.
- The scripted FSM consumes the sensed flag when present: the CONTACT phase
  advances on real force, and a sensed touch short-circuits PRE_CONTACT's
  distance check (a lagging arm meeting a moving panel is "arrived" the moment
  force appears). Envs without force sensing (`contact_sensed=None`) behave
  exactly as before — the controller stays Isaac-free.
- `eval/metrics.py` prefers the sensed signal for `contact_ticks` and adds
  `mean_contact_force_n`.
- Schema `phase2.v1` (additive superset of v0): optional proprio keys
  `joint_pos/joint_vel/joint_pos_target` (29-wide → **A1 becomes relabelable**;
  the A1 dataset export itself is still pending) and the contact keys above.

## 4. Backend probe (`scripts/verify_alex_ik_probe.py`)

Run before the env work relied on any backend read; findings:

- **Live pose/jacobian reads work for URDF-spawned Alex** (the Phase 2
  zeros-gotcha is specific to the USD-referenced door). The USD stage keeps
  only the authored rest pose for dynamic bodies — never compare live reads
  against stage reads for moving links.
- Standing pose settles at max 0.15 rad/s under the implicit drives. Two
  failure modes were measured and fixed on the way: high-gain
  `IdealPDActuatorCfg` (explicit PD) pegged the wrists at their velocity
  limits within 2 ticks (→ implicit drives), and damping 30 vs stiffness 300
  tracked only ~10 %/tick (→ k/c ≈ 40/s: 600/15, 150/4).
- Pose-mode (6-DoF) IK stalled from the ready pose (near-null joint swings,
  ~0 net motion); position-mode IK tracks 37 % of a commanded straight-line
  move open-loop, which the per-tick closed loop converts into reliable
  waypoint convergence.
- `--contact` mode measured 124 N on a commanded push into the panel.
- Self-collisions are disabled (`enabled_self_collisions=False`): the
  fullCollisions geometry overlaps between neighboring links at the zero pose
  and made the stance thrash; only the right arm moves, along a corridor clear
  of the torso.

## 5. Gate results (`scripts/verify_alex_door_baseline.py`)

- Fixed-start episode: **success, 61.0° final angle, 184 steps**, full phase
  sequence `approach(46) → align(8) → pre_contact(7) → contact(1) → push(47) →
  hold(30) → release(45)`.
- Force contact: **23 sensed ticks, 27.1 N peak** during contact/push/hold;
  force decays through hold and is zero in release — a physically sensible
  profile.
- Randomized episode (Alex variation bounds): success.
- Determinism: repeated same-seed rollout trace diff **0** (headless).
- Episode meta `robot=alex_v1_fullbody_fixedbase_v0`; 29-wide joint proprio;
  A2/A3/A4 exports pass the same relabeling checks as the proxy gate.

## 6. Reference run

`scripts/run_scripted_baseline.py --robot alex --episodes 5 --randomized 3
--video --enable_cameras --viz none --device cpu` →
`outputs/alex_door_push/2026-07-02_seed0/`:

- **8/8 episodes successful** (5 fixed seeds 0–4, 3 randomized 5–7),
  **mean final door angle 60.2°**, zero failure labels.
- 8 per-episode mp4s of Alex pushing the door open (`videos/episode_seed*.mp4`),
  metrics, door-angle/final-angle plots, and the run report.
- Datasets under `datasets/door_push_alex/{A1,A2,A3,A4}.../v0/` (a distinct task
  dir so the frozen Phase 2 proxy datasets are never replaced; A1 was added in
  the post-2.5 hardening pass).
- One tuning finding along the way: with the variation height band reaching
  0.22 m, one randomized episode timed out in APPROACH — the waypoint folds to
  within ~0.18 m of the shoulder, outside the arm's reachable band. The
  `ALEX_VARIATION_BOUNDS` height cap is 0.18 m for exactly this reason.

## Limitations

- Fixed base: no stepping, balancing, or regrasping; legs/torso/left arm are
  position-held. Whole-body control is Phase 5 territory.
- Rotation action components are not actuated (position-mode IK); the wrist
  orientation floats. Matches the proxy contract but means A2's rotation
  channels remain untested against a real actuator.
- Contact force is the EE link's net force, not a filtered gripper↔door pair;
  fine in this single-object scene, revisit for cluttered scenes.
- No articulated hand: the contact body is the gripper link's collision sphere.
- The Alex env's hinge damping (4 N·m·s/rad) makes its door dynamics
  intentionally different from the frozen proxy env.
- A1 is exported as 29-wide full-body joint-position-target deltas (hardening
  pass after 2.5); only the 6 right-arm IK joints carry non-zero deltas.
- Determinism remains per rendering mode (`--enable_cameras` runs differ
  numerically from headless runs of the same seed).
