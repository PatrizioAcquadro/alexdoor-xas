# Architecture Map

This is the **map of the whole project**. It exists so development stays oriented
even though only Phases 1–2 exist today. It describes the target structure, which
of the five system roles each part plays, and the phase that creates it.

> Principle (lean growth): we do **not** create empty code packages ahead of time.
> Each directory below is added when its phase begins. Until then, this document is
> the contract for where things will go. Source of truth for scope + phases is
> [`PROJECT_GUIDELINES.md`](PROJECT_GUIDELINES.md).

## The five roles (from guidelines §5)

Every runtime component maps to exactly one role. Keeping them separate is what
lets the *action interface* — the project's main research variable — be studied
independently of any one robot or policy.

| Role | Responsibility | Target module |
|------|----------------|---------------|
| **Observation** | Build what a policy sees: vision, robot state, object/door state, language, action-space context. | `src/alexdoor_xas/observation/` |
| **Policy** | Predict an action *representation* from observations. | `src/alexdoor_xas/policies/` |
| **Action Representation** | The format of the predicted action (A1–A4). The main research object. | `src/alexdoor_xas/action/` |
| **Adapter** | Convert an action representation into safe, executable Alex commands. | `src/alexdoor_xas/adapters/` |
| **Safety & Logging** | Enforce execution constraints; record every trial to the episode schema. | `src/alexdoor_xas/safety/`, `src/alexdoor_xas/recording/` |

## Target source tree (grown one phase at a time)

```
src/alexdoor_xas/
├── paths.py            # ✅ Phase 1 — canonical path registry (assets referenced in place)
├── assets/             # ✅ Phase 1 — load Alex/scene assets; single-door task fixture
├── observation/        # ⬜ Phase 3+ — Observation role: obs builders (Phase 2 obs are inline low-dim)
├── action/             # ✅ Phase 2 — A1–A4 tags/structs; door-frame math; A2<->A3 converters
├── envs/               # ✅ door env shell (gate) + Phase 2 DoorPushEnv (proxy EE) +
│                       #    Phase 2.5 DoorPushAlexEnv (fixed-base Alex, diff-IK arm, force contact)
├── policies/           # Policy role (model hierarchy, guidelines §7):
│   ├── scripted/       # ✅ Phase 2 — deterministic door-relative push FSM + seeded variations
│   ├── act/            #    Phase 3 — ACT-style imitation baseline
│   ├── diffusion/      #    Phase 3 — diffusion policy baseline
│   ├── vla/            #    Phase 4 — OpenVLA-OFT + mixed-action-space conditioning
│   └── wam_lite/       #    later   — action-conditioned future predictor (guidelines §8)
├── adapters/           # ⬜ Phase 3+ — Adapter role: action representation → Alex commands
│                       #    (Phase 2 executes A2 directly on the proxy EE; clamps live in the env)
├── safety/             # ⬜ Phase 3+ — Safety role: execution constraints, staged HW gates (§12)
│                       #    (Phase 2 safety = per-tick action clamps, recorded per step)
├── recording/          # ✅ Phase 2 — episode schema buffer + HDF5/JSON container
├── data_engine/        # ✅ Phase 2 — deterministic generation, A2/A3/A4 export, run orchestration
└── eval/               # ✅ Phase 2 — metrics, failure taxonomy, plots, run reports
```

Registered envs: `AlexDoor-DoorTask-Direct-v0` (verification shell),
`AlexDoor-DoorPush-Proxy-v0` (Phase 2 push task), and
`AlexDoor-DoorPush-Alex-v0` (Phase 2.5: same action contract executed by the
fixed-base Alex right arm with force contact sensing). Top-level `configs/`
(experiment/Hydra configs) stayed unnecessary in Phase 2 — engine/controller
settings are frozen dataclasses (`DataEngineCfg`, `DoorPushControllerCfg`)
snapshotted into each run's `logs/run_config.json`.

## What exists today (Phases 1–2)

```
DoorManipulation/
├── README.md                 # entry point
├── pyproject.toml            # package `alexdoor_xas` (src layout); sim stack is env-provided
├── src/alexdoor_xas/
│   ├── paths.py              # where every asset + artifact lives
│   ├── assets/{alex,scenes,door_task}.py
│   ├── action/               # A1–A4 tags/structs + door-frame math (A2<->A3)
│   ├── policies/scripted/    # deterministic door-relative push FSM
│   ├── envs/door_task/       # DoorTaskEnv (gate shell) + DoorPushEnv (proxy EE)
│   │                         #   + DoorPushAlexEnv (fixed-base Alex, diff-IK, contact sensor)
│   ├── recording/            # episode buffer + HDF5/JSON writer (episode_schema.md)
│   ├── data_engine/          # generation, A2/A3/A4 export, run orchestration
│   └── eval/                 # metrics, failure labels, plots, run reports
├── scripts/
│   ├── check_env.py          # fast readiness check (no Isaac launch)
│   ├── verify_assets.py      # headless Isaac: load Alex + combined scene
│   ├── verify_door_task_scene.py # headless Isaac: generated single-door scene gate
│   ├── verify_door_env.py    # headless Isaac: door env reset/step gate
│   ├── run_scripted_baseline.py  # engine CLI (episodes → datasets + artifacts; --robot proxy|alex)
│   ├── verify_scripted_baseline.py # Phase 2 gate (rollout + deterministic export)
│   ├── verify_alex_ik_probe.py     # Phase 2.5 backend probe (pose/jacobian/IK/contact)
│   └── verify_alex_door_baseline.py # Phase 2.5 gate (Alex rollout + force contact + export)
├── docs/                     # PROJECT_GUIDELINES (source of truth) + architecture, assets,
│                             # environment, episode_schema, action_spaces, phase2_report
├── tests/                    # pure-Python tests (no Kit launch)
├── datasets/                 # (gitignored) reusable exported episodes
└── outputs/                  # (gitignored) per-run artifacts
```

Phase 2 executes the scripted policy through the env's clamped EE-delta action
interface on a **proxy end-effector** (`proxy_ee_sphere_v0`); Phase 2.5 executes
the same interface on the **fixed-base Alex humanoid**
(`alex_v1_fullbody_fixedbase_v0`, right arm via position-mode differential IK,
force-sensed contact). The dependency direction is preserved: the controller and
data engine stay Isaac-free, the Alex env only *grew* optional duck-typed
accessors (`robot_joint_state`, `contact_force_w`, `contact_sensed`) that the
engine probes via `hasattr`. The dedicated Adapter and Safety packages still
start in Phase 3+; until then the env's per-tick clamps play the safety role and
are recorded per step.

`CombinedHallwayScene` remains the full-scene Phase 1 readiness asset. The first
scripted interactions use the single-door fixture instead, so room payloads,
THOR object references, and problematic floorplan door physics do not enter the
initial control/debug loop.

## Dependency direction

`paths` -> `assets` -> `envs` / `scripts`, and for Phase 2:
`action` -> {`policies/scripted`, `recording`} -> `eval` -> `data_engine` -> `scripts`.
Roles depend *inward*: policies and adapters may import `action` + `observation`;
nothing may import a policy from inside the adapter or safety layers. The
scripted policy and the data engine have **no Isaac imports** — the env is passed
in and duck-typed, so their logic is unit-tested without Kit. The adapter is the
only component allowed to emit low-level Alex commands (guidelines §5, §12).

## Data & artifacts

- **`datasets/<task>/<action_space>/<version>/`** — reusable training data emitted
  by the data engine; shared across many runs. See [`../datasets/README.md`](../datasets/README.md).
- **`outputs/<experiment>/<run_id>/{metrics,plots,videos,checkpoints,logs,episodes}/`**
  — everything a single run produces. See [`../outputs/README.md`](../outputs/README.md).

Both are gitignored (only their READMEs are tracked). Episodes conform to
[`episode_schema.md`](episode_schema.md); actions are tagged with a space from
[`action_spaces.md`](action_spaces.md).
