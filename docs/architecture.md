# Architecture Map

This is the **map of the whole project**. It exists so development stays oriented
even though only Phase 1 is implemented today. It describes the target structure,
which of the five system roles each part plays, and the phase that creates it.

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
├── assets/             # ✅ Phase 1 — load Alex articulation cfg + scene/door USDs
├── observation/        # ⬜ Phase 2+  — Observation role: obs builders
├── action/             # ⬜ Phase 2   — Action Representation role: A1–A4 types, encoders, converters
│                       #                (taxonomy defined now in docs/action_spaces.md)
├── envs/               # ⬜ Phase 2   — Isaac Lab door task(s): scene wiring + door benchmark
├── policies/           # ⬜ Phase 2+  — Policy role (model hierarchy, guidelines §7):
│   ├── scripted/       #    Phase 2   — deterministic controller + data generation
│   ├── act/            #    Phase 3   — ACT-style imitation baseline
│   ├── diffusion/      #    Phase 3   — diffusion policy baseline
│   ├── vla/            #    Phase 4   — OpenVLA-OFT + mixed-action-space conditioning
│   └── wam_lite/       #    later     — action-conditioned future predictor (guidelines §8)
├── adapters/           # ⬜ Phase 2+  — Adapter role: action representation → Alex commands
├── safety/             # ⬜ Phase 2+  — Safety role: execution constraints, staged HW gates (§12)
├── recording/          # ⬜ Phase 2   — Logging role: write trials to the episode schema
├── data_engine/        # ⬜ Phase 2   — deterministic episode generation + multi-action-space export
└── eval/               # ⬜ Phase 2+  — metrics, failure labels, plots, reports (§11)
```

Top-level `configs/` (experiment/Hydra configs) is likewise deferred to Phase 2.

## What exists today (Phase 1)

```
DoorManipulation/
├── README.md                 # entry point
├── pyproject.toml            # package `alexdoor_xas` (src layout); sim stack is env-provided
├── src/alexdoor_xas/
│   ├── paths.py              # where every asset + artifact lives
│   └── assets/{alex,scenes}.py
├── scripts/
│   ├── check_env.py          # fast readiness check (no Isaac launch)
│   └── verify_assets.py      # headless Isaac: load Alex + combined scene
├── docs/                     # PROJECT_GUIDELINES (source of truth) + architecture, assets, environment, episode_schema, action_spaces
├── tests/                    # light path/import tests
├── datasets/                 # (gitignored) reusable exported episodes
└── outputs/                  # (gitignored) per-run artifacts
```

## Dependency direction

`paths` → `assets` → `scripts` today. As roles are added they must depend
*inward*: policies and adapters may import `action` + `observation`; nothing may
import a policy from inside the adapter or safety layers. The adapter is the only
component allowed to emit low-level Alex commands (guidelines §5, §12).

## Data & artifacts

- **`datasets/<task>/<action_space>/<version>/`** — reusable training data emitted
  by the data engine; shared across many runs. See [`../datasets/README.md`](../datasets/README.md).
- **`outputs/<experiment>/<run_id>/{metrics,plots,videos,checkpoints,logs,episodes}/`**
  — everything a single run produces. See [`../outputs/README.md`](../outputs/README.md).

Both are gitignored (only their READMEs are tracked). Episodes conform to
[`episode_schema.md`](episode_schema.md); actions are tagged with a space from
[`action_spaces.md`](action_spaces.md).
