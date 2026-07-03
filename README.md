# AlexDoor-XAS

**Cross-Action-Space VLA/WAM Learning for Humanoid Articulated-Object Manipulation.**
A research benchmark for studying how humanoid manipulation *actions* should be
represented, learned, evaluated, and safely transferred — using door interaction
with the IHMC **Alex** torso in Isaac Sim / Isaac Lab as the first controlled task.

Scope, thesis, roles, action spaces, and phases are defined in
[`docs/PROJECT_GUIDELINES.md`](docs/PROJECT_GUIDELINES.md) (the source of truth). This repo is
currently at **Phase 2: scripted baseline and deterministic data engine** (complete —
see [`docs/phase2_report.md`](docs/phase2_report.md)).

## Repository map

```
src/alexdoor_xas/
  paths.py              # canonical path registry — every asset referenced in place
  assets/               # load Alex / scene assets and generate the single-door task fixture
  action/               # Phase 2: A1–A4 tags/structs + door-frame math (A2<->A3 converters)
  policies/scripted/    # Phase 2: deterministic door-push FSM controller (door-relative)
  envs/door_task/       # door env shell (gate) + Phase 2 door-push env with proxy EE
  recording/            # Phase 2: episode schema buffer + HDF5/JSON episode container
  data_engine/          # Phase 2: deterministic generation, A2/A3/A4 export, run orchestration
  eval/                 # Phase 2: metrics, failure labels, plots, run reports
scripts/
  check_env.py          # fast readiness check (versions, CUDA, assets) — no Isaac launch
  verify_assets.py      # headless Isaac: spawn Alex + open combined scene
  verify_door_task_scene.py # headless Isaac: validate the single-door task scene
  verify_door_env.py    # headless Isaac: reset/step the door DirectRLEnv gate
  run_scripted_baseline.py  # engine CLI: episodes -> datasets + metrics/plots/videos (--robot proxy|alex)
  verify_scripted_baseline.py # Phase 2 gate: scripted rollout + deterministic data export
  verify_alex_ik_probe.py   # Phase 2.5 backend probe: Alex pose/jacobian/IK/contact
  verify_alex_door_baseline.py # Phase 2.5 gate: Alex rollout + force contact + export
docs/
  PROJECT_GUIDELINES.md # project identity, scope, action spaces, phases (read first)
  architecture.md       # map of the whole project (roles → modules → phases)
  assets.md             # inventory: where each asset is, what it holds, how it loads
  environment.md        # official Isaac Sim / Isaac Lab launchers + verification
  episode_schema.md     # trial/episode schema (HDF5 + JSON container since Phase 2)
  action_spaces.md      # A1–A4 taxonomy + conditioning tags (A2/A3/A4 exported in Phase 2)
  phase2_report.md      # Phase 2 report: scene, controller, episodes, metrics, placeholders
datasets/               # (gitignored) reusable exported episodes
outputs/                # (gitignored) per-run artifacts
tests/                  # pure-Python tests (paths, action math, FSM, recording, engine, eval)
```

The full intended source tree (policies, adapters, safety, data engine, eval, …) is
described in [`docs/architecture.md`](docs/architecture.md) and grown one phase at a
time — no empty packages are created ahead of need.

## Quickstart (Phase 1 verification)

```bash
# 1. Install this package with the official Isaac Lab Python (no sim stack reinstall)
cd /home/pacquadr/Desktop/DoorManipulation
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .

# 2. Pure Python tests (no Kit launch)
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q

# 3. Isaac Lab smoke checks
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz none --device cpu --steps 1

# Optional GUI smoke check
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz kit --device cpu --steps 1
```

See [`docs/environment.md`](docs/environment.md) for details and troubleshooting.

## Local assets (referenced in place)
- **Alex model** → `~/Desktop/Alex-robot/alex_models/`
- **Scenes** → `~/Desktop/CombinedScene/` (corridor + rooms in `CombinedHallwayScene/combinedScene.usda`; door in `Door.usd`)

Override the root with `ALEXDOOR_ASSETS_ROOT` if these folders move. Full inventory
and load status in [`docs/assets.md`](docs/assets.md).

## Phase 1 status
- [x] Repo skeleton (lean `src/`, `datasets/` + `outputs/` conventions)
- [x] `README` / `.gitignore` / `pyproject.toml`
- [x] Docs: architecture, assets, environment, episode schema, action spaces
- [x] `paths.py` + `assets/` (self-contained Alex loader — no IHMC shim dependency)
- [x] `check_env.py` passes with the official Isaac Sim 6.0.1 / Isaac Lab release/3.0.0-beta2 install
- [x] `verify_assets.py` — full-body Alex loads (29 joints) + combined scene composes (5979 prims)

## Pre-Phase-2 door task gate

Before Phase 2 scripted interactions or data logging begin, this repo now has a
door-only gate that validates the benchmark object in isolation. It uses a
generated single-door task scene at `outputs/door_task/door_task.usda`, sourced
from `~/Desktop/CombinedScene/Door.usd`, plus a minimal registered
`DirectRLEnv` shell that can reset and step deterministic no-op actions.

Run the gate with:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_task_scene.py --viz none --device cpu --steps 100
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_env.py --viz none --device cpu --steps 100
```

`CombinedHallwayScene` remains the Phase 1 full-scene asset-readiness check, but
it is not used for the first scripted interactions. The combined hallway brings
room payloads, `objects/thor` references, and floorplan door-physics issues into
the first control/debug loop; the single-door task scene keeps the hinge, mass,
inertia, and env reset/step contract isolated.

## Phase 2: scripted baseline + deterministic data engine

Phase 2 is implemented and verified. A deterministic, **door-relative** scripted
controller (`approach → align → pre-contact → contact → push → hold → release`)
drives a velocity-controlled **proxy end-effector sphere** (`proxy_ee_sphere_v0`
— Alex is not yet in the loop; the env's 6-dim EE-delta action interface is what
the later Alex adapter must implement). Episodes are recorded to
[`docs/episode_schema.md`](docs/episode_schema.md) (HDF5 + JSON sidecar) and
exported per action space to `datasets/door_push/{A2,A3,A4}.../v0/`; A1 is a
documented placeholder (the proxy has no joints).

```bash
# Phase 2 gate: scripted rollout + deterministic data export (PASS/FAIL).
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_scripted_baseline.py --viz none --device cpu

# Data engine run: 5 fixed + 3 randomized episodes, datasets + metrics/plots/videos/report.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --episodes 5 --randomized 3 --video --enable_cameras
```

Results, limitations, and placeholders are summarized in
[`docs/phase2_report.md`](docs/phase2_report.md); each run also writes its own
`outputs/<experiment>/<run_id>/report.md`.

## Phase 2.5: Alex fixed-base executor + force contact sensing

The same scripted task is now also executed by the **IHMC Alex humanoid**
(`AlexDoor-DoorPush-Alex-v0`, robot tag `alex_v1_fullbody_fixedbase_v0`): fixed
base, right arm driven by position-mode differential IK, and contact measured by
a force sensor on the gripper link instead of geometric inference
(`contact.source = "force_sensor+geometric"`, schema `phase2.v1`). Alex episodes
record full joint state + applied targets, so A1 becomes relabelable. See
[`docs/phase2_5_alex_report.md`](docs/phase2_5_alex_report.md).

```bash
# Phase 2.5 backend probe: pose/jacobian reads, stance, IK tracking, contact force.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_ik_probe.py --viz none --device cpu --contact

# Phase 2.5 gate: Alex scripted rollout + force-sensed contact + data export (PASS/FAIL).
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_door_baseline.py --viz none --device cpu

# Alex data engine run with videos of Alex pushing the door open.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --robot alex --episodes 5 --randomized 3 --video --enable_cameras
```

> Isaac Lab workflows use `/home/pacquadr/IsaacLab/isaaclab.sh -p`; Isaac Sim-only
> Python workflows use `/home/pacquadr/isaacsim/python.sh`. Bare system `python3`
> is not expected to import `isaacsim`, `isaaclab`, `omni`, or `pxr`; runtime imports
> belong after `AppLauncher` initializes Kit. If GPU/display checks fail in a restricted
> shell, rerun them from a normal host-visible shell before diagnosing drivers.

> Fully-dressed combined scene requires the `~/objects/thor` symlink (see
> [`docs/assets.md`](docs/assets.md) issue 4): `ln -s ~/Desktop/Alex-robot/assets/usd/objects/thor ~/objects/thor`

> Phase 2 (scripted baseline + deterministic data engine) and Phase 2.5 (Alex
> fixed-base executor + force contact sensing) are implemented.
> Phases 3–5 (imitation/diffusion, VLA, hardware transfer) are **not**.
> See [`docs/PROJECT_GUIDELINES.md`](docs/PROJECT_GUIDELINES.md) §10.
