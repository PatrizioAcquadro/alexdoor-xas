# Environment & Simulation Readiness

Phase 1 **verifies** the simulator is usable; it does **not** install or upgrade
anything. Isaac Sim / Isaac Lab / PyTorch come from the existing `env_alex` conda
environment.

## The environment: `env_alex`

| | |
|---|---|
| Location | `/home/pacquadr/Desktop/isaac_suitcase/miniforge3/envs/env_alex` |
| Python | 3.11.15 |
| Isaac Sim | `isaacsim` 5.1.0.0 |
| Isaac Lab | `isaaclab` + `isaaclab_assets` (editable) |
| PyTorch | 2.7.0+cu128 |

Activate:

```bash
source /home/pacquadr/Desktop/isaac_suitcase/miniforge3/etc/profile.d/conda.sh
conda activate env_alex
```

Isaac scripts additionally require the EULA / privacy env vars:

```bash
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Yes PRIVACY_CONSENT=Y
```

## One-time: install this package (editable, light)

```bash
cd /home/pacquadr/Desktop/DoorManipulation
python -m pip install -e .        # installs `alexdoor_xas`; does NOT touch the sim stack
```

## Verify readiness

```bash
python -m pytest -q               # light path/import tests (no Isaac launch)
python scripts/check_env.py       # versions + CUDA + asset existence → PASS/exit 0
python scripts/verify_assets.py --headless   # spawn Alex + open combined scene → PASS/exit 0
```

`verify_assets.py` should print the full-body Alex joint set and a non-trivial scene prim count.
If the full-body URDF fails to convert, retry with `--variant nub` and note the error (see
[assets.md](assets.md)).

## Constraints & notes

- **Do not install or upgrade** Isaac Sim / Isaac Lab. Phase 1 only verifies.
- **`isaacsim` / `isaaclab` are not pip dependencies** of this package — they are provided by
  `env_alex`. Installing `alexdoor_xas` never pulls them.
- **Self-contained by design.** The Alex config is imported from Alex-robot by absolute path, so
  this project does **not** depend on the IHMC IsaacLab shim.
- **Suitcase-invariant drift (informational).** Historically Alex work used a separate
  `IsaacLab-alex` clone to keep the frozen `isaac_suitcase` untouched. That clone is now gone;
  `env_alex` points its editable IsaacLab at `isaac_suitcase/IsaacLab/` and the IHMC shim was added
  inside that tree. The separate `env_isaaclab` is unaffected (the change is additive), and this
  project does not depend on the shim — but the original "never write inside the suitcase IsaacLab"
  rule has been bent. Flagged, not acted on.
