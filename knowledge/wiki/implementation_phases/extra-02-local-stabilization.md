# Extra 02 — Local Stabilization

> Historical phase record. Current safety and execution semantics are documented in [[topics/action-representations-and-adapters|Action Representations and Adapters]].

## Objective

Stabilize learned-policy rollout, contact entry, force reporting, and terminal-state handling on the five-pose Alex V2 benchmark.

## Subphase E2.1 — Closed-Loop Safety Semantics

#### Implementation

This work hardened first-crossing success, settle behavior, adapter correction/rejection, terminal-state preservation, and force admission. A local four-cell N50 smoke matrix completed 144 successful rollouts with no adapter rejection.

The smoke-matrix runner was removed after its behavior-level contracts moved into the maintained adapter, rollout, evaluation, and test paths.

#### Key Decisions

- Adapter corrections and warnings remain explicit.
- Simulator force limits are evaluation signals, not hardware limits.

#### Problems / Limitations

- The local matrix was a stabilization check, not a comparative scientific result.
- Its run-specific plan and summarizer are no longer maintained.

## Artifacts

No local smoke-matrix artifact is part of the active output contract. Git retains the historical implementation.

## Files

- `src/alexdoor_xas/adapters/rollout.py`
- `src/alexdoor_xas/policies/common/closed_loop.py`
- `tests/test_rollout_semantics.py`
