# Phase 1 — Project and Simulation Readiness

## Objective

Establish a runnable Python package, local asset boundary, and deterministic
Isaac environment smoke path.

## Focus

### Subphase 1.1 — Repository and runtime foundation

#### Implementation

Created the package layout, path registry, environment checks, door fixture,
and simulator entry-point conventions. Isaac imports occur only after
`AppLauncher`; pure data and policy code remains importable without Isaac.

#### Key Decisions and Problems

- Isaac Sim/Lab and robot assets remain external workstation dependencies.
- The repository does not distribute machine-local scene or robot files.

#### Tests

Deterministic path, asset, import, reset, and step checks cover the maintained
foundation.

## Version Notes

- 2026-08-11 — Historical run detail removed; current runtime boundary retained.
