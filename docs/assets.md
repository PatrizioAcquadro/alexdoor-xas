# Local Assets Inventory

All assets are **referenced in place** — nothing is copied into this repo. The
canonical paths live in [`src/alexdoor_xas/paths.py`](../src/alexdoor_xas/paths.py)
and resolve from `ASSETS_ROOT` (default `~/Desktop`, override with
`ALEXDOOR_ASSETS_ROOT`). Run [`scripts/check_env.py`](../scripts/check_env.py) to
confirm every path below exists on your machine.

Legend — **Load status:** ✅ loadable · ⚠️ loadable with the noted handling · ❌ broken here.

## Alex model — `~/Desktop/Alex-robot/alex_models/`

| Asset | Path (relative to `Alex-robot/alex_models`) | Kind | Likely contents | Load status | How to load |
|-------|---------------------------------------------|------|-----------------|-------------|-------------|
| Full-body URDF **(default)** | `alex_V1_description/rl_urdf/alex_v1.rlModel_fullBody_robotAccurate_torsoFootCollisions.urdf` | URDF | 29-DoF humanoid: legs + spine + neck + arms + **wrist/gripper**; minimal (torso+foot) collisions | ✅ | load-proven by `scripts/verify_assets.py` (29 joints); `package://` meshes → rewrite to abs paths (done by `resolve_alex_urdf`) |
| Nub URDF (fallback) | `alex_V1_description/rl_urdf/alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions.urdf` | URDF | 23-DoF; forearms replaced by a nub (no hand joints). **Proven to load** (walking script) | ✅ | same `package://` handling |
| Other URDF variants | `alex_V1_description/rl_urdf/*fullCollisions*.urdf`, `*legCollisions*`, `*_hanging*` | URDF | full-collision / leg-collision / hanging (sim rig) variants | ⚠️ | same handling; `*_hanging` uses a broken `../alex-models/` prefix |
| **abs-paths URDF** | `alex_V1_description/rl_urdf/*_abs_paths.urdf` | URDF | pre-flattened mesh paths | ❌ | hardcoded to `/home/sravani/...` — **needs regen**; do not use |
| IsaacLab config | `alex_V1_isaacsim/alex.py` | Python | `ALEX_V1_FULLBODY_DEFAULT_CFG`, `ALEX_V1_NUBS_DEFAULT_CFG` — actuator groups (legs/torso/arms), PD gains, effort/velocity limits, armature | ✅ | imported by abs path via `load_alex_articulation_cfg` (no IHMC shim needed) |
| Meshes | `alex_V1_description/meshes/` (`legs/`, `cycloidal_arm/`, `*.obj/.glb`) | Meshes | visual + collision geometry the URDFs reference | ✅ | resolved automatically once `package://` is rewritten |
| MJCF | `alex_V1_description/mjcf/*.xml` | MuJoCo | MJX / training XMLs (MuJoCo side of the hybrid repo) | n/a for Isaac | not used by this project's Isaac path |

**Not present:** the ONNX walking policy (`isaac-sim-rl-bringup/models/.../policy.onnx`) is not
committed to Alex-robot and must be copied from the lab machine. **Not needed for Phase 1** — this
project spawns Alex statically for a load check and does not run locomotion.

## Scenes — `~/Desktop/CombinedScene/`

| Asset | Path (relative to `CombinedScene`) | Kind | Likely contents | Load status | How to load |
|-------|-----------------------------------|------|-----------------|-------------|-------------|
| **Combined corridor (default)** | `CombinedHallwayScene/combinedScene.usda` | USD (ascii) | `defaultPrim="World"`, Z-up, meters. References a hallway + 4 iThor floorplans as `scene`/`scene_01..03`/`Hallway` | ✅ | `UsdFileCfg(usd_path=...).func("/World/Scene", cfg)` — rooms/hallway + props load once the `~/objects/thor` symlink exists (see issue 4) |
| Door | `Door.usd` | USD (crate) | Articulated door: `Handle` + `Hinge` — the door benchmark object | ✅ | referenced by the generated single-door task scene below |
| Hallway only | `Hallway/Hallway.usd` (+ `Textures/`, `untitled.usdc`) | USD | corridor geometry referenced by the combined scene | ✅ | reference or sublayer |
| FloorPlan rooms | `FloorPlan1_updated_physics/`, `FloorPlan212_physics/`, `FloorPlan315_updated_physics/`, `FloorPlan401_updated_physics/` (`scene.usda` + `Payload/`) | USD | individual iThor rooms with physics; referenced by the combined scene | ✅ | reference `scene.usda` |

The combined scene uses **relative** references (`@../FloorPlan1_updated_physics/scene.usda@` …),
so it must stay alongside its sibling folders (it currently does) and be opened by its absolute
path so USD resolves the siblings.

## Pre-Phase-2 single-door task scene

The first scripted interactions do **not** use `CombinedHallwayScene`. They use a
small generated fixture:

| Asset | Path | Kind | Contents | Load status | How to verify |
|-------|------|------|----------|-------------|---------------|
| Single-door task scene | `outputs/door_task/door_task.usda` | generated USD (ignored output) | `/World`, floor, light, one reference to `~/Desktop/CombinedScene/Door.usd` at `/World/DoorTaskDoor`, fixed door frame, authored positive mass/inertia for door and handle | ✅ | `scripts/verify_door_task_scene.py --viz none --device cpu --steps 100` |

The fixture is authored by `src/alexdoor_xas/assets/door_task.py` so the control
and env checks start from one door articulation only. It deliberately excludes
`CombinedHallwayScene`, `FloorPlan*` payloads, `objects/thor` references, and
foreign `file:/C:/.../Door.usd` references. This keeps the first hinge, mass,
inertia, and reset/step checks independent from room props and floorplan door
physics.

`CombinedHallwayScene` is still the Phase 1 full-scene readiness asset. The
single-door task scene is the pre-Phase-2 gate for first scripted interactions;
it is not the scripted baseline, data logger, or data engine.

## Loading from code (the intended path)

```python
# after AppLauncher (see scripts/verify_assets.py)
from alexdoor_xas.assets import scenes
from alexdoor_xas.assets.alex import load_alex_articulation_cfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

cfg = scenes.scene_spawn_cfg(scenes.combined_scene_usd()); cfg.func("/World/Scene", cfg)
robot = Articulation(load_alex_articulation_cfg("fullbody").replace(prim_path="/World/Alex"))
```

## Known issues and prerequisites (carried forward, not fixed in Phase 1)

1. **`*_abs_paths.urdf` is broken on this machine** (paths point at `/home/sravani/...`). Use the
   `package://` URDFs; `resolve_alex_urdf` flattens their mesh paths correctly. Regenerate the
   abs-paths variant only if a tool specifically needs it.
2. **Full-body config is now load-proven here.** `scripts/verify_assets.py --viz none --device
   cpu --steps 1` spawns the default full-body Alex with 29 joints. Keep `--variant nub` as a
   debugging fallback only if the full-body path regresses.
3. **Legacy-only IHMC IsaacLab shim drift.** The original per-project `IsaacLab-alex` clone and
   `/home/pacquadr/Desktop/isaac_suitcase` runtime are historical references only. This project
   now runs through the official `/home/pacquadr/IsaacLab` install and deliberately does **not**
   rely on the IHMC shim (it imports `alex.py` by absolute path). See
   [environment.md](environment.md).
4. **Combined scene props — resolved via external `~/objects/thor` symlink.** The 86 room objects
   are referenced *relatively* (`@../../../../objects/thor/<Obj>/<Obj>.usda@` from each
   `FloorPlan*/Payload/Geometry.usda`), which resolves to `~/objects/thor/`. The full library
   (5004 entries, covering all 86) lives at `Alex-robot/assets/usd/objects/thor/`, so this machine
   uses:
   ```
   ln -s ~/Desktop/Alex-robot/assets/usd/objects/thor ~/objects/thor
   ```
   All 86/86 now resolve. Treat this symlink as a **prerequisite** for a fully-dressed combined
   scene (recreate it if the link or Alex-robot moves), not an active blocker. Note:
   `~/Desktop/HallwayScene/Objects` holds only ~9 *hallway* furniture pieces (covers 1 of the 86)
   — it is not the room-object source.
5. **Combined scene door physics is excluded from the first door task gate.** `FloorPlan1` references a door at a
   Windows path (`file:/C:/Users/rainb/.../Door.usd`) → harmless PhysX "no bodies defined" warning
   for `/World/Scene/scene/Door_01/FixDoorframe`. Each floorplan's `Door/Handle` rigid body also
   warns about negative mass / invalid inertia. The pre-Phase-2 gate uses the repo's own
   `Door.usd` through `outputs/door_task/door_task.usda` with task-layer mass/inertia overrides
   instead of the combined scene floorplan doors. Return to the fully dressed scene later; it is
   not required for the first scripted baseline/data-logging step after the env verifier passes.
