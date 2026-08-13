# Phase 1 — Project and Simulation Readiness

> Historical phase record. Current operational behavior is documented in [[topics/system-architecture|System Architecture]] and [[topics/alex-v2-benchmark|Alex V2 Benchmark]].

## Objective

Establish a runnable Python package, external asset boundary, and deterministic Isaac environment smoke path.

## Subphase 1.1 — Repository and Runtime Foundation

#### Implementation

This phase created the package layout, path conventions, host checks, simulator entry-point pattern, and initial door environment. Isaac imports were kept behind `AppLauncher` so pure data and policy modules could remain Isaac-free.

The maintained successor is now the single Alex V2 environment and compact path registry; earlier provisional robot and generic runtime layers were removed.

#### Key Decisions

- Isaac Sim, Isaac Lab, CUDA, and machine-local assets remain external dependencies.
- Simulator startup and pure-Python imports remain separated.

#### Problems / Limitations

- The original foundation did not define the final Alex V2 benchmark.
- Historical compatibility and alternate runtime paths are not maintained.

## Artifacts

No run-specific Phase 1 evidence package remains active. Current readiness is checked directly by the environment and benchmark verification commands.

## Files

- `src/alexdoor_xas/paths.py`
- `scripts/check_env.py`
- `src/alexdoor_xas/envs/door_task/`
