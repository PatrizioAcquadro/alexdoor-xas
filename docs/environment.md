# Environment & Simulation Readiness

Phase 1 **verifies** the simulator is usable; it does **not** install or upgrade
anything. Isaac Sim / Isaac Lab / PyTorch come from the official local installs.

## Official Isaac installs

| | |
|---|---|
| Isaac Sim | `/home/pacquadr/isaacsim` |
| Isaac Lab | `/home/pacquadr/IsaacLab` |
| Isaac Sim version | 6.0.1 |
| Isaac Lab branch/version | `release/3.0.0-beta2` |

## Launch command policy

```bash
# Isaac Lab script
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p <script>

# Isaac Sim-only Python script
PYTHONPATH=$PWD /home/pacquadr/isaacsim/python.sh <script>
```

Do not use bare system `python3` for Isaac code. It is not expected to import
`isaacsim`, `isaaclab`, `omni`, or `pxr`. Scripts that need Kit/Isaac runtime must
create `AppLauncher` before importing Isaac runtime modules such as `isaaclab.sim`,
`omni`, or `pxr`.

Headless and GUI workflows are intentionally documented separately:

```bash
# Headless smoke
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz none --device cpu --steps 1

# GUI smoke
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz kit --device cpu --steps 1
```

## One-time: install this package (editable, light)

```bash
cd /home/pacquadr/Desktop/DoorManipulation
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e .
```

## Verify readiness

```bash
# Pure Python tests: no Kit launch, no Isaac runtime imports.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q

# Environment/package/assets check: no Kit launch.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py

# Isaac Lab headless asset smoke.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_assets.py --viz none --device cpu --steps 1
```

`verify_assets.py` should print the full-body Alex joint set and a non-trivial scene prim count.
If the full-body URDF fails to convert, retry with `--variant nub` and note the error (see
[assets.md](assets.md)).

## Pre-Phase-2 door task gate

After Phase 1 asset/environment readiness passes, run the door-only gate before
starting Phase 2 scripted baseline or data logging work:

```bash
# Door-only USD/articulation stability gate.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_task_scene.py --viz none --device cpu --steps 100

# Registered DirectRLEnv reset/step determinism gate.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_door_env.py --viz none --device cpu --steps 100
```

`scripts/verify_door_task_scene.py` validates the generated
`outputs/door_task/door_task.usda` fixture, including the single revolute hinge,
positive door/handle mass and inertia, fixed frame behavior, finite simulated
state, and bounded frame/door motion. `scripts/verify_door_env.py` creates the
registered door `DirectRLEnv`, confirms it uses that same single-door fixture,
resets it, steps deterministic no-op actions, checks finite observations/rewards,
and compares repeated rollout traces.

These commands are a gate into Phase 2, not Phase 2 itself.

## Phase 2 scripted baseline + data engine

```bash
# Gate: scripted rollout + deterministic data export (PASS/FAIL, non-zero on failure).
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_scripted_baseline.py --viz none --device cpu

# Engine run: fixed + randomized episodes -> datasets/ + outputs/<experiment>/<run_id>/.
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --episodes 5 --randomized 3

# Same run with per-episode rollout videos (offscreen rendering).
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --episodes 5 --randomized 3 --video --enable_cameras
```

`verify_scripted_baseline.py` asserts: the fixed-start scripted episode opens the
door past the success threshold, a repeated same-seed rollout reproduces
identical traces, A2/A3/A4 exports exist and satisfy the episode schema
(including the A2→A3 door-frame relabeling), and metrics/plots/report are
written. Gate artifacts live under `outputs/verify_scripted_baseline/`; real
dataset exports only come from `run_scripted_baseline.py`.

Note: physics traces are deterministic per mode, but a `--video --enable_cameras`
run differs numerically from a headless run of the same seed (camera pipeline
changes Kit's update path). Compare rollouts within one mode.

## Phase 2.5 Alex executor (fixed base, force contact sensing)

```bash
# Backend probe: live pose/jacobian reads, stance stability, IK tracking,
# push-arc reachability, contact force (run before touching the Alex env).
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_ik_probe.py --viz none --device cpu --contact

# Gate: Alex scripted rollout + force-sensed contact + data export (PASS/FAIL).
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_alex_door_baseline.py --viz none --device cpu

# Engine run on Alex (videos of Alex pushing the door open):
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/run_scripted_baseline.py \
    --viz none --device cpu --robot alex --episodes 5 --randomized 3 --video --enable_cameras
```

`AlexDoor-DoorPush-Alex-v0` keeps the proxy env's action contract (6-dim A2 EE
delta, same clamps and timing) but executes it with the IHMC Alex humanoid:
fixed base (pelvis welded at `(-0.45, -0.38, 0.93)`, yaw π), right arm driven by
position-mode differential IK (`dls`) into stiff implicit PD drives, contact
measured by a `ContactSensor` on `RIGHT_GRIPPER_Z_LINK` (fullCollisions URDF —
the default fullbody URDF has no arm collision geometry). The Alex env adds
passive hinge damping (4 N·m·s/rad) so the door moves only while pushed; the
proxy env keeps the frozen undamped hinge. Design details and measured
gate/probe results: [phase2_5_alex_report.md](phase2_5_alex_report.md).

## Constraints & notes

- **Do not install or upgrade** Isaac Sim / Isaac Lab. Phase 1 only verifies.
- **`isaacsim` / `isaaclab` are not pip dependencies** of this package — they are provided by
  the official Isaac installs. Installing `alexdoor_xas` never pulls them.
- **Restricted-shell GPU/display failures are not driver proof.** If `nvidia-smi`,
  Torch CUDA/NVML, X11, GLX, or Vulkan fail inside a sandboxed IDE/Codex shell, rerun
  the verification commands from a normal host-visible shell before diagnosing the host driver.
- **Guard Isaac imports.** `pxr`, `omni`, `isaaclab_tasks`, and Isaac Lab assets may require
  `AppLauncher` to initialize Kit first.
- **Self-contained by design.** The Alex config is imported from Alex-robot by absolute path, so
  this project does **not** depend on the IHMC IsaacLab shim.
- **Legacy-only custom runtime.** `/home/pacquadr/Desktop/isaac_suitcase`, `env_alex`,
  `isaaclab-run`, and `isaaclab-alex-run` are historical references only. Do not delete the
  suitcase folder, but do not use it as the primary runtime for this repo.
