# Extra 01 — Alex V2 Migration

> Historical phase record. The current benchmark is documented in [[topics/alex-v2-benchmark|Alex V2 Benchmark]].

## Objective

Replace provisional robot assumptions with the fixed-base IHMC Alex V2 door benchmark.

## Subphase E1.1 — Asset, Calibration, and Execution

#### Implementation

This work validated the Alex V2 asset and joint order, derived the collision tool point, introduced offset-point Jacobian control, and added door-panel force sensing.

Generic Alex construction later moved to the external `ihmc_alex_isaaclab` extension. The current repository retains only door-specific calibration, position-only IK, scene, contact, and task behavior.

#### Key Decisions

- Requested rotation remains represented but is not actuated.
- `configs/alex_v2_door.json` is the only active task calibration.

#### Problems / Limitations

- Calibration and force evidence are simulation-specific.
- The completed calibration-authoring and generic executor layers were removed.

## Artifacts

The current calibration JSON and canonical D0-D4 layers are operational inputs, not historical phase packages.

## Files

- `configs/alex_v2_door.json`
- `src/alexdoor_xas/assets/alex_v2.py`
- `src/alexdoor_xas/envs/door_task/`
