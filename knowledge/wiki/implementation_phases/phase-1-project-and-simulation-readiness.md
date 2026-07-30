# Phase 1 — Project and Simulation Readiness

## Objective

Establish a small, installable AlexDoor-XAS package and a fail-fast workstation
gate before implementing the manipulation task. This phase defined the external
Isaac dependency boundary, machine-local asset discovery, and the first
simulator smoke test.

## Focus

### Subphase 1.1 — Package and Runtime Boundary

#### Implementation

Phase 1 created the `alexdoor_xas` package, its test layout, and supported
workstation commands. `pyproject.toml` deliberately contains only portable
package dependencies; Isaac Sim, Isaac Lab, PyTorch, and CUDA remain external
runtime dependencies supplied by the supported workstation stack.

`src/alexdoor_xas/paths.py` centralizes machine-local roots and environment
overrides so scripts do not scatter absolute paths. The current registry points
to Alex V2 and the CombinedScene assets because [[extra-01-alex-v2-migration|Extra 01]]
replaced the original provisional Alex V1 contract. This later update is
current behavior, not a Phase 1 Alex V2 deliverable.

#### Key Decisions and Problems

- Isaac modules are intentionally absent from package dependencies. Importable
  core modules stay usable in ordinary Python, while simulator entry points use
  `/home/pacquadr/IsaacLab/isaaclab.sh`.
- Large USD and robot assets are referenced in place rather than copied into
  Git. Environment variables can override the default local locations.
- The original Alex V1 assumptions from this phase were superseded by the
  calibrated fixed-base [[topics/alex-v2-benchmark|Alex V2 Benchmark]].

#### Tests

- `tests/test_paths.py` verifies default and overridden path resolution without
  requiring Isaac.
- `tests/test_check_env.py` exercises environment-check failure reporting and
  version/path contracts with test doubles.

### Subphase 1.2 — Environment and Asset Gates

#### Implementation

`scripts/check_env.py` performs the non-simulator preflight. It validates the
official Isaac Sim and Isaac Lab roots, supported versions, the Alex-enabled
Isaac Lab branch, required asset paths, and the expected runtime imports. It
reports independent checks so a missing asset is distinguishable from a broken
Python or simulator installation.

`scripts/verify_assets.py` is the simulator smoke gate. It initializes
`AppLauncher` before importing Isaac modules, creates a small world, loads the
current Alex V2 articulation and combined stage through
`src/alexdoor_xas/assets/scenes.py`, steps simulation, and verifies that the
expected prims are present. This is an asset-load and stepping check, not a
door-task or learned-policy validation.

#### Key Decisions and Problems

- Verification is two-tiered: a fast environment/path preflight followed by an
  actual simulator load. Passing only the first tier does not establish that
  USD composition or physics initialization works.
- The optional CombinedScene composition can encounter a dangling local
  `~/objects/thor` reference. The isolated door benchmark introduced later
  does not depend on that scene reference.
- No physical-robot control surface was created. Simulator readiness is not
  hardware readiness.

#### Tests

- The phase closeout recorded a successful editable install, deterministic
  unit suite, environment preflight, and simulator asset smoke run on the
  then-supported stack.
- Current regression coverage includes `tests/test_paths.py`,
  `tests/test_check_env.py`, and asset-specific tests added by later phases;
  current full runtime validation still requires the configured Isaac stack.

## Version Notes

- 2026-07-01 — Initial package, path registry, workstation checks, and asset
  smoke workflow landed.
- 2026-07-08 — Alex V2 migration replaced the provisional V1 asset and runtime
  assumptions while preserving the Phase 1 verification boundary.
