# AlexDoor-XAS

**Cross-Action-Space VLA/WAM Learning for Humanoid Articulated-Object Manipulation.**
A research benchmark for studying how humanoid manipulation *actions* should be
represented, learned, evaluated, and safely transferred — using door interaction
with the IHMC **Alex** torso in Isaac Sim / Isaac Lab as the first controlled task.

Scope, thesis, roles, action spaces, and phases are defined in
[`docs/PROJECT_GUIDELINES.md`](docs/PROJECT_GUIDELINES.md) (the source of truth). This repo is
currently at **Phase 1: project definition, asset organization, and simulation readiness.**

## Repository map

```
src/alexdoor_xas/
  paths.py              # canonical path registry — every asset referenced in place
  assets/               # load the Alex articulation cfg + the corridor scene / door
scripts/
  check_env.py          # fast readiness check (versions, CUDA, assets) — no Isaac launch
  verify_assets.py      # headless Isaac: spawn Alex + open combined scene
docs/
  PROJECT_GUIDELINES.md # project identity, scope, action spaces, phases (read first)
  architecture.md       # map of the whole project (roles → modules → phases)
  assets.md             # inventory: where each asset is, what it holds, how it loads
  environment.md        # env_alex activation + how to verify readiness
  episode_schema.md     # minimal trial/episode schema
  action_spaces.md      # A1–A4 taxonomy + conditioning tags
datasets/               # (gitignored) reusable exported episodes
outputs/                # (gitignored) per-run artifacts
tests/                  # light path/import tests
```

The full intended source tree (policies, adapters, safety, data engine, eval, …) is
described in [`docs/architecture.md`](docs/architecture.md) and grown one phase at a
time — no empty packages are created ahead of need.

## Quickstart (Phase 1 verification)

```bash
# 1. Activate the ready-made Isaac environment (nothing is installed/upgraded)
source /home/pacquadr/Desktop/isaac_suitcase/miniforge3/etc/profile.d/conda.sh
conda activate env_alex

# 2. Install this package (light; the sim stack is provided by env_alex)
cd /home/pacquadr/Desktop/DoorManipulation
python -m pip install -e .

# 3. Verify
python -m pytest -q                          # path/import sanity
python scripts/check_env.py                  # versions + CUDA + asset existence
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Yes PRIVACY_CONSENT=Y
python scripts/verify_assets.py --headless   # load Alex + combined scene in Isaac
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
- [x] `check_env.py` passes (Isaac Sim 5.1.0.0, Isaac Lab, torch 2.7.0+cu128, RTX 4090)
- [x] `verify_assets.py` — full-body Alex loads (29 joints) + combined scene composes (5979 prims)

> Fully-dressed combined scene requires the `~/objects/thor` symlink (see
> [`docs/assets.md`](docs/assets.md) issue 4): `ln -s ~/Desktop/Alex-robot/assets/usd/objects/thor ~/objects/thor`

> Phases 2–6 (scripted baseline, data engine, imitation/diffusion, VLA, hardware transfer)
> are **not** implemented. See [`docs/PROJECT_GUIDELINES.md`](docs/PROJECT_GUIDELINES.md) §10.
