# Extra 02 — Local Stabilization

## Objective

Turn the four local Alex V2 learned-policy cells into a reproducible,
reviewable smoke-evaluation gate by hardening termination, force, provenance,
warning, reporting, and process-level determinism semantics.

## Focus

### Subphase Extra 02.1 — Evaluation Semantics and Provenance

#### Implementation

`src/alexdoor_xas/policies/common/rollout_eval.py` and
`src/alexdoor_xas/adapters/rollout.py` preserve the last pre-reset state,
recognize simulator termination and truncation after every executed tick, and
separate terminal force from the pre-action series. Success is recorded at the
first hinge-angle crossing rather than reconstructed from a reset observation.

`src/alexdoor_xas/policies/common/eval_metadata.py` assembles run identity from
checkpoint, dataset, split, normalization, robot, calibration, simulator, and
source bindings. `src/alexdoor_xas/kinematics/settle.py` verifies the reset and
settle postcondition before a rollout. The runner rejects stale, mixed, or
missing provenance rather than producing a superficially comparable report.

#### Key Decisions and Problems

- Evaluation is fail-closed: all cells must use the intended physical dataset,
  matched episode/seed plan, and compatible runtime identities.
- First-crossing and terminal-force evidence are captured online because a
  post-reset observation cannot reconstruct them reliably.
- The local matrix is a smoke gate at N50, not a policy-quality ranking or
  data-scaling study.

#### Tests

- `tests/test_rollout_semantics.py`, `tests/test_eval_metadata.py`,
  `tests/test_terminal_force.py`, and `tests/test_settle_postcondition.py`
  exercise the hardened boundaries and failure cases.
- `tests/test_stabilization_doc.py` and `scripts/verify_stabilization_doc.py`
  keep recorded stabilization claims consistent with the machine-readable
  evidence.

### Subphase Extra 02.2 — Safety Corrections and Warning Envelope

#### Implementation

The adapter/evaluation path records every accepted, corrected, and rejected
decision. Contact-entry shaping and joint/workspace limiting are therefore
visible as correction counts rather than hidden control behavior. Warnings are
classified and checked against an explicit envelope.

The smoke matrix produced 876 warnings—219 per cell—and all were the documented
bounded reset-transient family. No action was rejected. This established that
the warning path was stable and interpretable for the tested matrix, not that
all future warnings are benign.

#### Key Decisions and Problems

- A corrected action remains executable evidence but is not equivalent to a
  policy request being accepted unchanged.
- The warning envelope is explicit and versioned; unknown warning families
  require review.
- Position-only execution continues to leave requested rotation unactuated,
  even when the six-dimensional adapter request is accepted or corrected.

#### Tests

- `tests/test_adapters.py` verifies correction records and warning generation.
- `tests/test_summarize_smoke_eval.py` verifies aggregation and rejects
  inconsistent cell counts or undocumented warning families.

### Subphase Extra 02.3 — Four-Cell Local Matrix

#### Implementation

`configs/local_smoke_eval_plan_n50.json` defines matched seeds and cells for ACT
and Diffusion over A2 and A3. `scripts/eval_act.py`,
`scripts/eval_diffusion.py`, and `scripts/summarize_smoke_eval.py` execute and
summarize the matrix. The durable results are maintained in
[[experiments/local-n50-stabilization|Local N50 Stabilization Matrix]].

All four cells completed 36/36 rollouts: ACT-A2 had 296 corrected actions,
ACT-A3 233, Diffusion-A2 195, and Diffusion-A3 210, with zero rejected actions.
Peak panel-filtered forces ranged from 129.4 N to 145.5 N. Twenty fresh-process
determinism probes also matched. These results validated the stabilized
workflow, not general policy superiority.

#### Key Decisions and Problems

- Thirty-six matched rollouts per cell balance local smoke coverage with
  workstation cost.
- The results use simulation-only force and success evidence; they do not
  establish physical-robot safety.

#### Tests

- The four-cell execution produced 144/144 successful rollouts and passed
  summary validation.
- Twenty fresh-process probes verified that determinism was not an artifact of
  state retained inside one Python process.

## Version Notes

- 2026-07-09 to 2026-07-11 — Termination, terminal-force, settle, provenance,
  warning, and reporting semantics were hardened and reviewed.
- 2026-07-11 — The complete N50 local smoke matrix and fresh-process probes
  passed; the phase was closed as a stabilization gate.
