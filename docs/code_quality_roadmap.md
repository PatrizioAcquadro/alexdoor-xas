# AlexDoor-XAS Production Code Quality Roadmap

## Document status

- **Roadmap status:** approved; implementation not started
- **Scope:** the complete tracked repository
- **Baseline commit:** `d22c5b8a714b85f44ecd1b7fd0a724a48f1f4f88`
- **Baseline date:** 2026-07-16
- **Compatibility policy:** preserve external contracts; fix validated defects when
  required to restore documented fail-closed behavior
- **Quality tools:** Ruff and pytest only
- **Final qualification:** one frozen 16-cell, 576-rollout rerun after all
  runtime/evaluation changes

This document governs code-quality work. It does not replace:

- [`PROJECT_GUIDELINES.md`](PROJECT_GUIDELINES.md) for research intent, phase
  boundaries, and safety principles;
- [`architecture.md`](architecture.md) for implemented technical contracts;
- [`development.md`](development.md) for supported workflows and commands; or
- [`status.md`](status.md) for completed evidence, limitations, and next steps.

If this roadmap conflicts with those sources, stop and resolve the discrepancy
before changing code.

## Progress

| Phase | Name | Active unit | Status |
|---:|---|---|---|
| 0 | Restore a trusted baseline | - | Not started |
| 1 | Correct validated robotics boundary defects | - | Not started |
| 2 | Make data and resource lifecycles transactional | - | Not started |
| 3 | Establish consistency and automated gates | - | Not started |
| 4 | Strengthen core design contracts | - | Not started |
| 5 | Consolidate data, configuration, and policy infrastructure | - | Not started |
| 6 | Decompose evaluation, cluster, and CLI monoliths | - | Not started |
| 7 | Finish packaging, runtime, documentation, and efficiency hygiene | - | Not started |
| 8 | Final qualification | - | Not started |

Allowed status values are `Not started`, `In progress`, `Blocked`, and
`Complete`. Mark a phase complete only after every exit criterion passes.

## 1. Quality standard

Production quality for this repository means:

- **Correctness:** physical units, frames, action semantics, state transitions,
  and scientific computations are explicit and tested.
- **Fail-closed robotics behavior:** invalid commands, state, calibration, or
  provenance stop execution or publication before unsafe or misleading output
  is produced.
- **Simple design:** use small cohesive components, direct control flow, and
  stable interfaces. Do not add an abstraction without a concrete second use.
- **Clear boundaries:** policies predict representations; adapters validate and
  convert them; environments execute them; recording and evaluation preserve
  evidence.
- **Reproducibility:** seeds, schemas, fingerprints, configurations, checkpoints,
  and evaluation protocols remain deterministic and auditable.
- **Resource safety:** files, figures, temporary directories, subprocesses, and
  simulator lifecycles close or roll back correctly on every path.
- **Measured efficiency:** optimize demonstrated bottlenecks, not hypothetical
  ones. Every optimization needs before/after evidence and compatibility tests.
- **Concise communication:** comments and docstrings explain non-obvious domain
  contracts. They do not repeat the code. Type annotations are added where they
  clarify an interface, not as decoration.

The target is not maximum abstraction, minimum line count, or blanket adherence
to style slogans. The target is code whose behavior is difficult to misuse and
straightforward to verify.

## 2. Non-negotiable compatibility contract

### Must remain compatible

- Public import paths, re-exported symbols, and package version behavior.
- Script names, CLI arguments, defaults, standard output/error conventions, and
  exit codes.
- Gym environment IDs, entry points, and registration side effects.
- Config filenames, keys, schemas, precedence, and current defaults.
- A1-A4 tags, shapes, meanings, units, frames, and conversion semantics.
- XYZW quaternion order; world, door, panel, tool, and joint conventions.
- Simulator timing, per-tick clamps, joint order, position-only Alex V2 IK, and
  force-evidence semantics.
- Stable adapter statuses, warning identifiers, and established failure labels.
- `phase2.v1` HDF5/JSON/A4 layouts and legacy readable versions.
- Dataset, split, view, normalization, calibration, checkpoint, and artifact
  fingerprint contracts.
- Evaluation plan, rollout, normalized-row, aggregate, CSV, JSON, and report
  schemas.
- Existing artifact readers and the ability to verify historical artifacts in
  their original source context.
- AppLauncher-before-Isaac import ordering and current workstation/Gilbreth
  runtime boundaries.

### May change

- Private helpers and internal module layout.
- Internal data structures behind compatibility facades.
- Tests, fixtures, and developer automation.
- Additive public APIs whose defaults preserve existing behavior.
- Behavior proven to violate the documented fail-closed contract, provided the
  correction has a focused regression test and is recorded as a defect fix.

### Requires separate approval

- Removing or renaming any external interface.
- A schema or checkpoint-format migration.
- New research features, Phase 4/VLA work, or a new action representation.
- Training, dataset generation, official artifact replacement, or threshold
  tuning.
- Hardware, fake-door, real-door, WAM, or other robot execution.
- Changing scientific claims or refreshing historical curated evidence.

## 3. Verified starting point

The baseline was measured from a clean tracked worktree on the commit above.

| Check | Result |
|---|---|
| `ruff check .` | Pass |
| `ruff format --check .` | Fail: 112 files would be reformatted |
| Supported-runtime pytest | 758 passed, 1 failed, 2 dependency deprecation warnings |
| Failing test | Clean-tree sweep-manifest test depends on an ignored stale local artifact |
| Hosted CI | None |
| Pytest marker taxonomy | None |
| Package code | Approximately 24,800 lines |
| Scripts | Approximately 10,400 lines |
| Tests | Approximately 14,700 lines |

The current Ruff selection is `E`, `F`, `I`, `UP`, and `B`. Complexity probes
show concentrated risk rather than uniform disorder. The main hotspots are:

- unified Phase 3 evaluation and smoke-summary validation;
- cluster transfer, return, provenance, and manifest verification;
- A4 and shared rollout execution;
- dataset validation/normalization and checkpoint provenance; and
- training/evaluation/orchestration scripts.

The tracked curated Phase 3 evidence aggregate is:

```text
a46d0c73c26c8fd35a1583ce7a65774c25dd8f521d1fffd41b06f23526a64bdb
```

It is the SHA-256 of the sorted per-file `sha256sum` inventory produced by:

```bash
git ls-files outputs/curated/phase3_unified_evaluation \
  | sort \
  | xargs sha256sum \
  | sha256sum
```

This aggregate must remain unchanged through Phases 0-7. If implementation
starts from a different commit, append a new baseline entry; do not erase this
historical snapshot.

## 4. Execution rules for every phase

1. Start from a clean tracked worktree and record unrelated user changes.
2. Add characterization or regression tests before changing observable behavior.
3. Keep one coherent concern per change set. Do not mix formatting, behavior
   fixes, structural refactors, and new features.
4. Preserve public behavior through a facade before moving implementation.
5. Run focused tests while iterating, then the complete hermetic suite.
6. Run simulator or artifact gates only when the touched boundary requires them.
7. Inspect the diff for schema, config, fingerprint, warning, and output drift.
8. Stop on unexplained drift. Do not update expected output merely to make a
   failing test pass.
9. Update this document's phase status and record any approved exception.

Minimum gate after every change set:

```bash
ruff check .
ruff format --check .
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
git diff --check
```

`ruff format --check .` becomes required after the Phase 3 formatting commit.
Before then, use it to prevent newly touched files from increasing the formatting
debt.

## Agent execution protocol

The subphases below are the implementation units. Give an implementation agent
one unit ID at a time, together with this document. A unit is deliberately
smaller than a phase and must produce one reviewable outcome.

### Rules for assigning a unit

1. Run units in the listed dependency order. Keep only one implementation unit
   active at a time.
2. Before editing, the agent must read the parent phase's objective, work,
   protected contracts, tests, exit criteria, and deferred scope.
3. The agent may change only what the assigned unit requires. Discovering nearby
   debt does not authorize fixing it.
4. A defect unit starts with a failing regression test. A refactor unit starts
   with passing characterization tests. A mechanical unit may not change tests
   or expected behavior.
5. Run the unit's focused gate, the current repository-wide gate, and
   `git diff --check` before reporting completion.
6. Do not update a golden file, fingerprint, schema, warning ID, or curated
   artifact to absorb unexplained drift.
7. Stop and mark the unit `Blocked` if completion requires an unapproved schema,
   API, dependency, research, dataset, training, simulator-protocol, or hardware
   change.
8. Qualification units are validation-only. If a qualification unit fails,
   reopen the responsible implementation unit; do not fix code inside the
   qualification unit.
9. After review, update the unit status and the phase's `Active unit`. Mark the
   phase `Complete` only when every unit in that phase is complete and the phase
   exit criteria pass.

### Required agent handoff

Every unit completion report must contain:

```text
Subphase: <ID and name>
Status: Complete | Blocked
Intent: <one-sentence result>
Files changed: <tracked files only>
External contracts: <preserved or intentionally additive>
Tests and gates: <commands and exact results>
Artifacts: <hash/inventory result or Not applicable>
Behavior differences: <approved difference or None>
Residual risks/deferred work: <items not addressed>
Commit: <hash, or Not committed>
```

### Phase 0 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 0A | Reconfirm commit/worktree, Ruff/test/format results, expected warnings/skips, and curated hash | None | Appended baseline entry with exact commands/results | Not started |
| 0B | Replace the ignored-manifest test with an isolated `tmp_path` manifest and retain a separate local-artifact gate | 0A | Focused manifest tests plus clean-clone-style pytest pass | Not started |
| 0C | Add small compatibility fixtures for imports, CLIs, Gym IDs, configs, serialization, fingerprints, warnings/failures, and evaluation output | 0B | New compatibility suite passes without ignored files | Not started |
| 0D | Qualify Phase 0 without implementation changes | 0C | Full supported suite green; curated hash unchanged | Not started |

### Phase 1 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 1A | Fix A4 termination/truncation handling and freeze the last valid state | 0D | Regression tests prove no post-reset read and stable new failure values | Not started |
| 1B | Enforce quaternion norm and proper-rotation validation | 1A | Unit/round-trip tests cover drift, non-finite input, and reflections | Not started |
| 1C | Enforce panel height, contact Boolean, joint-limit, and final action-boundary validation | 1B | Focused invalid/boundary tests pass; nominal traces remain exact | Not started |
| 1D | Add calibrated Alex A4 construction and migrate only Alex call sites | 1C | Factory/calibration tests plus Alex adapter smoke gate | Not started |
| 1E | Add rotation-actuation capability and non-actuated-rotation warning | 1D | Warning tests pass; zero-rotation rollout output is unchanged | Not started |
| 1F | Qualify Phase 1 without implementation changes | 1E | Adapter, rollout, calibration, executor, and relevant Isaac gates pass | Not started |

### Phase 2 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 2A | Make episode-step recording commit only after successful execution | 1F | Step-exception/auto-reset tests prove no invalid trainable tick | Not started |
| 2B | Make HDF5/sidecar publication transactional with HDF5 as commit point | 2A | Failure-injection and unchanged round-trip tests pass | Not started |
| 2C | Close figures and harden temporary/subprocess/simulator cleanup paths | 2B | Repetition, timeout, nonzero-exit, and cleanup tests pass | Not started |
| 2D | Qualify Phase 2 without implementation changes | 2C | Recording, export, evaluation, and full supported tests pass; hashes unchanged | Not started |

### Phase 3 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 3A | Apply repository-wide Ruff formatting in a mechanical-only change | 2D | Format diff reviewed; behavior tests unchanged; format check passes | Not started |
| 3B | Enable/fix `C4`, `SIM`, `RET`, and `PIE` in isolated green batches | 3A | Selected Ruff rules and full tests pass | Not started |
| 3C | Enable/fix `PERF`, `RUF`, `N`, `A`, and `S` with narrow documented ignores | 3B | Expanded Ruff configuration and full tests pass | Not started |
| 3D | Add pytest marker taxonomy and collection-safe optional ML tests | 3C | Hermetic selector collects/runs without Torch, Isaac, or local artifacts | Not started |
| 3E | Add hosted Python 3.11/3.12 Ruff and hermetic pytest workflow | 3D | Workflow definition passes from a clean checkout | Not started |
| 3F | Qualify Phase 3 without implementation changes | 3E | Hosted and supported-runtime gates are green; skips are explicit | Not started |

### Phase 4 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 4A | Add `ActionSpace` and minimal environment/policy protocols behind existing APIs | 3F | Public import/signature fixtures and fake-env tests pass | Not started |
| 4B | Type stable rollout, provenance, evaluation-row, and manifest boundaries only | 4A | Runtime validation/serialization tests remain exact | Not started |
| 4C | Consolidate shared step/termination/context-freezing state machine | 4B | A2/A3/A4 characterization and termination tests pass | Not started |
| 4D | Add executable import-direction and AppLauncher-order architecture tests | 4C | Known-good graph passes and injected violations fail clearly | Not started |
| 4E | Qualify facades and Phase 4 without implementation changes | 4D | All legacy imports/signatures pass; full suite and relevant simulator gates pass | Not started |

### Phase 5 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 5A | Extract canonical JSON, hashing, inventory, and atomic-publication primitives behind facades | 4E | Old/new byte and hash equivalence tests pass | Not started |
| 5B | Centralize Hydra composition and legacy override precedence | 5A | Resolved-config matrix is identical | Not started |
| 5C | Share checkpoint envelope/provenance validation without moving policy-specific state | 5B | ACT/Diffusion legacy/current checkpoint tests pass | Not started |
| 5D | Add O(1) internal episode-ID indexing without changing dataset ordering/API | 5C | Lookup/error/fingerprint equivalence tests pass | Not started |
| 5E | Profile the 550-episode path and make the explicit lazy-loading decision | 5D | Recorded time/memory evidence; default decision is no lazy rewrite unless material benefit is proven | Not started |
| 5F | Qualify Phase 5 without implementation changes | 5E | Dataset, config, checkpoint, fingerprint, and full supported gates pass | Not started |

### Phase 6 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 6A | Extract Phase 3 plan, evidence, execution, statistics, aggregation, and reporting behind `phase3_unified` | 5F | Facade imports and curated-output golden tests pass | Not started |
| 6B | Move smoke-summary domain logic into the package and retain a thin CLI | 6A | Existing CLI and summary golden tests pass | Not started |
| 6C | Extract policy-agnostic evaluation flow with explicit ACT/Diffusion strategies | 6B | Rollout request/metadata/JSON equivalence tests pass | Not started |
| 6D | Extract stable cluster-common primitives while keeping pilot/sweep validators separate | 6C | Transfer/return/tamper suites pass with identical outputs | Not started |
| 6E | Thin remaining scripts to parsing, AppLauncher, orchestration, and exit mapping | 6D | CLI help/output/exit golden tests pass | Not started |
| 6F | Enforce C90 complexity 15 and qualify Phase 6 without implementation changes | 6E | C90, full tests, and exact curated reprocessing pass | Not started |

### Phase 7 execution units

| ID | Bounded implementation task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 7A | Establish one version source and supported Python/dependency bounds | 6F | Source/install version and Python 3.11/3.12 smoke tests pass | Not started |
| 7B | Add `data`/`plots` extras and extend hosted CI to exercise them | 7A | Extras import tests and expanded CI pass | Not started |
| 7C | Centralize official runtime/path defaults without changing frozen files | 7B | Default/override/path-provenance tests pass | Not started |
| 7D | Reconcile README, development, status, package, calibration, and environment documentation | 7C | Documentation contract test/manual cross-check passes | Not started |
| 7E | Record performance evidence for every accepted optimization and reject unsupported ones | 7D | Before/after table plus output-equivalence evidence | Not started |
| 7F | Qualify Phase 7 without implementation changes | 7E | Clean installs, extras, CI, full runtime tests, and curated hash pass | Not started |

### Phase 8 execution units

| ID | Bounded qualification task | Depends on | Completion evidence | Status |
|---|---|---|---|---|
| 8A | Run final format, lint, hosted hermetic, full supported pytest, and diff gates | 7F | All static/hermetic results recorded and green | Not started |
| 8B | Run documented asset, scene, environment, scripted, dataset, adapter, ACT, and Diffusion gates | 8A | Exact runtime gate log with no unexplained warning/regression | Not started |
| 8C | Re-hash curated evidence and reprocess it through the refactored evaluator | 8B | Historical hash and canonical output bytes are exact | Not started |
| 8D | Run the frozen 16-cell, 576-rollout qualification in a new ignored workspace | 8C | 16 complete cells, 576 valid rollouts, zero exclusions, approved tolerances | Not started |
| 8E | Perform final compatibility audit, document residual limitations, and hand off | 8D | Every phase/unit complete; final evidence and commit recorded | Not started |

The detailed phase sections below are the specification for these units. The
tables define assignment boundaries and order; they do not replace the tests,
protected contracts, or exit criteria in the parent phase.

## Phase 0 - Restore a trusted baseline

### Objective

Make the default test suite hermetic and establish compatibility evidence before
any broad refactor.

### Work

- [ ] Replace the sweep-manifest test's dependency on
  `outputs/cluster_sweep/sweep_transfer_manifest.json` with a manifest generated
  entirely under `tmp_path`.
- [ ] Preserve the test's real purpose: the CLI must run without the repository
  root on `PYTHONPATH` and must verify a valid manifest.
- [ ] Keep exact verification of machine-local manifests as an explicit artifact
  integration command, never as a default unit test.
- [ ] Add compact compatibility tests for public imports, package exports, Gym
  registrations, CLI parsers, config loading, serialization, fingerprinting,
  warning/failure identifiers, and canonical evaluation outputs.
- [ ] Store only small synthetic compatibility fixtures. Do not copy datasets,
  checkpoints, or Phase 3 output trees into the test suite.
- [ ] Record the complete supported-runtime test count, expected skips, and the
  exact two Isaac dependency warnings.
- [ ] Recompute and verify the curated-evidence aggregate before merging.

### Protected contracts

The local artifact verifier, manifest schema, CLI output, and exit status remain
unchanged. Historical manifests are verified against their source snapshot, not
against the current `HEAD` by assumption.

### Required tests

- Valid temporary manifest passes through the isolated CLI entry point.
- Missing, stale, extra, and tampered manifest entries fail closed.
- A clean clone without ignored outputs passes the default suite.
- Dirty and clean Git states do not change hermetic test results.
- Compatibility fixtures pass on Python 3.11 and 3.12.

### Exit criteria

- Full supported-runtime pytest has zero failures.
- Default tests require no ignored artifacts, network, GPU, display, or simulator
  launch.
- The curated Phase 3 aggregate is unchanged.
- Any expected skip or warning has an explicit reason.

### Deferred

No production refactor, formatting sweep, or defect correction belongs in this
phase.

## Phase 1 - Correct validated robotics boundary defects

### Objective

Restore documented fail-closed semantics at action, frame, environment, and A4
execution boundaries before restructuring those paths.

### Work

- [ ] In A4 execution, inspect `(terminated, truncated)` immediately after every
  environment step. Freeze the last valid pre-step context and perform no
  post-reset read.
- [ ] Add `environment_terminated` and `environment_truncated` to A4 execution
  failures without changing `A4ExecutionResult` fields.
- [ ] Accept a quaternion only when it is finite, nonzero, and
  `abs(norm - 1.0) <= 1e-3`. Normalize only within that tolerance.
- [ ] Require every trusted rotation matrix to be finite, orthonormal, and have
  `det(R)` within `1e-5` of `+1`.
- [ ] Enforce panel Y and Z bounds in both scripted and adapter geometric-contact
  inference.
- [ ] Accept contact state only as a Boolean or exact scalar `0/1`; reject other
  numeric values.
- [ ] Validate joint-limit shapes, finite values, matching joint order, and
  `lower <= upper` before using them.
- [ ] At the final environment boundary, reject wrong-shaped, non-floating,
  non-finite, or wrong-device actions before clamp or IK.
- [ ] Add and export
  `build_alex_v2_a4_adapter(a3, calibration) -> A4Adapter`. It must use validated
  controller clearances, the collision-derived tool convention, and
  `ee_radius_m=0`.
- [ ] Keep the existing A4 constructor/defaults for proxy and legacy callers.
  All Alex V2 call sites must use the calibrated factory.
- [ ] Add `RobotLimitsCfg.actuates_rotation: bool = False` as a trailing defaulted
  field. Preserve `AdapterDecision.applied` as the adapter output.
- [ ] Emit warning record `a2.rotation_not_actuated` when a nonzero rotation is
  passed to a translation-only executor.

### Protected contracts

- A2/A3 remain six-dimensional `(dpos_m, drot_axis_angle_rad)` representations.
- A2 remains world-frame; A3 remains static hinge-anchored door-frame.
- Current clamps remain `0.02 m` and `0.05 rad` per control tick.
- Alex remains position-only DLS IK with the exact six right-arm joints.
- Existing zero-rotation learned rollouts must not change adapter decisions.

### Required tests

- A4 truncation and termination stop at the exact step and never read auto-reset
  state.
- Non-unit quaternions outside tolerance, NaN/Inf quaternions, and reflections
  fail closed.
- Small float32 quaternion drift inside tolerance remains accepted.
- Contact above or below the panel is false; boundary contact remains unchanged.
- Contact values `0`, `1`, `False`, and `True` pass; other scalars fail.
- Reversed or mismatched joint limits fail before action processing.
- NaN/Inf, integer, and wrong-device environment actions fail before IK.
- The Alex A4 factory reproduces every calibrated clearance and tool convention.
- Nonzero rotation produces `a2.rotation_not_actuated`; zero rotation does not.
- Existing A2/A3 round trips, clamps, warning families, and same-seed synthetic
  traces remain exact.

### Exit criteria

- Every listed defect has a focused regression test.
- All adapter, rollout-semantics, calibration, joint-limit, and executor-contract
  tests pass.
- Relevant CPU Isaac adapter and Alex V2 gates pass.
- No existing public field or import is removed.

### Deferred

Acceleration limits, general collision checking, slip detection, rotational IK,
and hardware readiness remain outside this roadmap.

## Phase 2 - Make data and resource lifecycles transactional

### Objective

Ensure failures cannot publish partial evidence, count unexecuted actions, or
leak runtime resources.

### Work

- [ ] Build an episode step as pending data, call `env.step`, and append it to the
  trainable buffer only after the call returns successfully.
- [ ] If execution may have partially progressed before an exception, invalidate
  the complete episode and prevent export rather than guessing what executed.
- [ ] Detect termination/truncation immediately and never derive terminal evidence
  from an auto-reset state.
- [ ] Write HDF5 and JSON sidecars to same-filesystem temporary paths.
- [ ] Read back and validate both temporary artifacts before publication.
- [ ] Publish the sidecar first and atomically replace the HDF5 file last as the
  visible commit point. Remove temporary paths in `finally` blocks.
- [ ] Preserve filenames, `phase2.v1`, HDF5 groups, JSON keys, and reader behavior.
- [ ] Close every Matplotlib figure in a `finally` block after saving.
- [ ] Audit file handles, temporary directories, subprocesses, and simulator
  shutdown paths. Require explicit return-code handling and bounded timeouts for
  external probes.

### Protected contracts

Pre-action observations remain paired with the action that actually executed.
Terminal contact remains the post-action response to the final committed action.
The existing episode container and discovery rules do not change.

### Required tests

- An `env.step` exception commits no trainable tick for the attempted action.
- A partial or failed episode cannot be exported or admitted to a dataset.
- Failure during HDF5 creation, sidecar creation, validation, or either rename
  leaves no discoverable partial episode.
- Successful output round-trips identically through the existing reader.
- Repeated plotting leaves no open figures.
- Subprocess timeout and nonzero-exit paths clean temporary state and report the
  original cause.

### Exit criteria

- Interrupted-step, interrupted-write, cleanup, and repeated-plot tests pass.
- Existing dataset fingerprints and serialized field values are unchanged for
  successful fixtures.
- No raw exception is swallowed at a publication boundary.

### Deferred

Changing the episode schema, adding a transaction database, or rewriting HDF5
storage is not authorized.

## Phase 3 - Establish consistency and automated gates

### Objective

Create one enforced style and a fast hosted signal while keeping the supported
Torch/Isaac runtime authoritative for the complete suite.

### Work

- [ ] Run `ruff format .` once and commit only the mechanical formatting diff.
- [ ] Require `ruff format --check .` thereafter.
- [ ] Enable Ruff families in small green batches: `C4`, `SIM`, `RET`, `PIE`,
  `PERF`, `RUF`, `N`, `A`, and `S`.
- [ ] Use narrow per-file ignores for test assertions, intentional CLI output,
  and reviewed subprocess boundaries. Each ignore must state why it is safe.
- [ ] Do not enable blanket docstring or annotation rules.
- [ ] Define strict pytest markers: `ml`, `runtime`, `artifact`, and `slow`.
- [ ] Make Torch and Diffusers test modules collection-safe through explicit
  `pytest.importorskip` guards.
- [ ] Add `.github/workflows/quality.yml` with Python 3.11 and 3.12 jobs.
- [ ] Hosted CI runs Ruff plus `pytest -m "not ml and not runtime and not artifact
  and not slow"` using `.[dev]`.
- [ ] Record expected data/plot skips in hosted CI until Phase 7 declares their
  optional extras. The supported workstation gate must continue to exercise them.
- [ ] Keep the existing Isaac launcher as the sole authority for the complete
  local suite and all simulator gates.

### Protected contracts

Formatting and lint fixes must not change runtime output, schemas, or numerical
behavior. CLI `print` calls remain valid at CLI boundaries.

### Required tests and gates

```bash
ruff check .
ruff format --check .
pytest -q -m "not ml and not runtime and not artifact and not slow"
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
```

CI must also run from a clean checkout containing no ignored local artifacts.

### Exit criteria

- Formatting and expanded lint are green.
- Hosted hermetic CI passes on Python 3.11 and 3.12.
- The complete supported-runtime suite remains green.
- CI and local gate responsibilities are explicit; skips are not silent.

### Deferred

Static type checkers, coverage tools, pre-commit frameworks, and security scanners
are intentionally excluded. Ruff and pytest remain the only quality dependencies.

## Phase 4 - Strengthen core design contracts

### Objective

Make architectural and data-flow boundaries explicit without replacing efficient
NumPy/Torch APIs or breaking imports.

### Work

- [ ] Add and export an `ActionSpace` `Literal` covering the four canonical tags.
- [ ] Define small `Protocol` interfaces for the environment accessors and chunk
  policy call surfaces actually consumed by adapters and evaluation.
- [ ] Use dataclasses or `TypedDict` definitions at stable boundaries such as
  rollout state, evaluation rows, provenance, and manifest entries.
- [ ] Keep numerical arrays as NumPy/Torch values; do not wrap every vector in a
  class.
- [ ] Consolidate step execution, termination detection, context refresh, and
  final-state freezing into one shared state machine used by A2/A3/A4 paths.
- [ ] Keep the existing exported `step_env` behavior through a compatibility
  wrapper if a richer internal result type is introduced.
- [ ] Add AST-based architecture tests enforcing:
  - policies do not import adapters;
  - adapters do not import policies;
  - dependency-light action/dataset/adapter modules do not import Isaac; and
  - scripts requiring Kit initialize AppLauncher before Isaac imports.
- [ ] Re-export every moved public symbol from its original module.

### Protected contracts

Duck-typed fake environments, public ndarray arguments/results, import paths, and
policy/adapter separation remain supported. Protocols improve clarity; they do
not introduce a new static-analysis dependency.

### Required tests

- Architecture violations fail with the importing modules named.
- All current fake environments satisfy the runtime contract used by their tests.
- A2/A3/A4 use identical termination semantics.
- Public import and signature fixtures remain unchanged.
- Invalid payloads still fail at their original boundary with equivalent error
  meaning.

### Exit criteria

- Architecture rules are executable tests.
- Shared execution semantics replace duplicated state-transition code.
- No compatibility fixture changes are required.

### Deferred

Do not build a generic framework, dependency-injection container, or universal
robot abstraction.

## Phase 5 - Consolidate data, configuration, and policy infrastructure

### Objective

Remove stable duplication where it creates drift while preserving policy- and
protocol-specific logic.

### Work

- [ ] Create a focused artifact-support package for canonical JSON encoding,
  SHA-256 hashing, file inventories, and atomic JSON/text publication.
- [ ] Replace duplicate implementations incrementally; keep current public
  functions as facades until all callers and fixtures pass.
- [ ] Centralize Hydra initialization/composition and legacy override application.
  Preserve all scripted, ACT, and Diffusion keys and precedence.
- [ ] Extract shared checkpoint-envelope and provenance validation into
  policy-common code. Keep model state, optimizer state, and policy-specific
  metadata in their existing policy modules.
- [ ] Add an internal episode-ID dictionary when an `EpisodeDataset` is built so
  `by_id` is O(1) without changing ordering or the external API.
- [ ] Profile the 550-episode master path using `cProfile`, `time.perf_counter`,
  and process memory evidence under the supported runtime.
- [ ] Introduce lazy loading only if the profile demonstrates a material memory
  or latency bottleneck and the lazy implementation preserves samples, order,
  exceptions, and fingerprints exactly.

### Protected contracts

Canonical bytes, hashes, config precedence, dataset ordering, normalization,
checkpoint keys, and legacy loading remain unchanged.

### Required tests

- Old and new helpers produce identical canonical bytes and fingerprints.
- All config permutations resolve identically before and after consolidation.
- ACT and Diffusion legacy/current checkpoints load with the same validation
  outcomes.
- Episode lookup returns the same object/error for every fixture ID.
- Any accepted optimization includes deterministic equivalence tests and recorded
  before/after measurements.

### Exit criteria

- Duplicated stable primitives are removed from callers.
- Policy-specific behavior remains separate and readable.
- Deterministic outputs and performance evidence are documented.

### Deferred

No lazy dataset rewrite is performed without measured justification. No database,
distributed loader, or new serialization format is introduced.

## Phase 6 - Decompose evaluation, cluster, and CLI monoliths

### Objective

Split high-risk orchestration and validation modules into cohesive components
while retaining their existing commands and output contracts.

### Work

- [ ] Keep `alexdoor_xas.eval.phase3_unified` as the public compatibility facade.
- [ ] Extract Phase 3 plan parsing, evidence validation, execution, statistics,
  aggregation, and reporting into a focused `alexdoor_xas.eval.phase3` package.
- [ ] Move smoke-summary domain logic into `alexdoor_xas.eval.smoke_summary`; keep
  the existing script as a thin CLI.
- [ ] Add a policy-agnostic evaluation runner with explicit ACT and Diffusion
  strategies. Preserve script filenames, arguments, defaults, rollout ordering,
  metadata, and JSON output.
- [ ] Extract stable inventory, hashing, secret scanning, transfer, and atomic
  publication primitives into cluster-common code.
- [ ] Keep pilot/sweep schemas and protocol-specific validators in their current
  domains; do not force them through a generic manifest abstraction.
- [ ] Reduce scripts to argument parsing, required AppLauncher initialization,
  orchestration, and mapping domain errors to the established PASS/FAIL output and
  exit codes.
- [ ] Enable Ruff `C90` with `max-complexity = 15` after decomposition.
- [ ] Permit an exception only for a reviewed boundary validator that remains
  clearer as one function; scope and justify it beside the Ruff configuration.

### Protected contracts

All current CLI, import, subprocess, JSON/CSV/report, bootstrap, comparison,
exclusion, and provenance semantics remain exact.

### Required tests

- Original imports resolve through the facade.
- CLI help, parsing, exit codes, and PASS/FAIL output match golden fixtures.
- Existing curated inputs reproduce byte-identical normalized CSV, aggregate
  JSON, exclusions, and report output in a temporary directory.
- ACT and Diffusion runner strategies produce the same rollout requests and
  metadata as their current scripts.
- Pilot and sweep transfer/return tampering tests remain fail-closed.
- `ruff check . --select C90` passes under the approved threshold/exception list.

### Exit criteria

- Former monoliths are facades or cohesive components.
- No public caller requires migration.
- Golden scientific and operational outputs are exact.

### Deferred

Do not redesign the scientific protocol, merge ACT and Diffusion model logic, or
generalize cluster tooling beyond demonstrated shared primitives.

## Phase 7 - Finish packaging, runtime, documentation, and efficiency hygiene

### Objective

Make supported environments and optional capabilities explicit, remove metadata
drift, and align documentation with executable reality.

### Work

- [ ] Add `src/alexdoor_xas/_version.py` containing `0.1.0` and configure Hatch to
  read it as the single version source. Re-export it from the package root.
- [ ] Set supported Python to `>=3.11,<3.13`.
- [ ] Bound NumPy as `>=1.24,<3`; retain existing Hydra/OmegaConf ranges.
- [ ] Bound development tools as `pytest>=9.1,<10` and `ruff>=0.15,<0.16`.
- [ ] Add `data = ["h5py>=3.16,<4"]` and
  `plots = ["matplotlib>=3.10,<4"]` optional extras.
- [ ] Preserve the current `tracking` and `diffusion` extras. Keep Torch, Isaac
  Sim, and Isaac Lab supplied by the official environments.
- [ ] Update hosted CI to install `.[dev,data,plots]` and remove the corresponding
  expected skips.
- [ ] Centralize official Isaac Sim, Isaac Lab, Alex asset, and scene path defaults
  in the path/runtime contract. Explicit CLI/config values retain precedence.
- [ ] Do not alter frozen Phase 3 protocol files or calibration fingerprints merely
  to remove an absolute path.
- [ ] Correct README status drift and align development/status documentation with
  Isaac Sim `6.0.1-rc.7`, Isaac Lab `3.0.0`, workstation Python 3.12, and Gilbreth
  Python 3.11.
- [ ] Record before/after wall time, memory, and output equivalence for every
  accepted optimization.

### Protected contracts

Package name/version, editable installation, optional-runtime behavior, current
official paths, and all frozen environment/provenance identities remain valid.

### Required tests

- Core editable installation and import succeed on Python 3.11 and 3.12.
- `data`, `plots`, `tracking`, and `diffusion` extras have explicit import smoke
  tests appropriate to their supported environment.
- Source and installed package report version `0.1.0`.
- Runtime/path resolution preserves current defaults and explicit overrides.
- Canonical docs agree with `check_env.py`, calibration, and package metadata.

### Exit criteria

- Packaging has one version source and bounded supported ranges.
- Hosted CI exercises all hermetic non-ML data/plot tests.
- Canonical documentation and executable checks agree.
- Every optimization has evidence; unproven optimization ideas remain deferred.

### Deferred

Do not publish the package, distribute machine-local assets, install/upgrade the
Isaac stack, or claim compatibility with untested Python/runtime versions.

## Phase 8 - Final qualification

### Objective

Prove that the completed improvement program preserved the pipeline, scientific
evidence, and runtime behavior within the approved defect corrections.

### Qualification sequence

- [ ] Require a clean tracked worktree and record the final commit.
- [ ] Run Ruff lint, Ruff format check, the hosted hermetic matrix, and the full
  supported-runtime pytest suite.
- [ ] Run every relevant asset, scene, environment, scripted baseline, dataset,
  adapter, ACT, and Diffusion verification gate from `development.md`.
- [ ] Recompute the tracked curated-evidence aggregate and require the baseline
  value in this document.
- [ ] Reprocess existing Phase 3 evidence through the refactored audit/report path
  in a temporary directory; require exact canonical JSON, CSV, and report bytes.
- [ ] Create a new ignored qualification workspace. Do not write into the returned
  cluster package or tracked curated directory.
- [ ] Reuse the frozen 16 checkpoints, datasets, views, normalization, seeds, and
  36-rollout-per-cell plan. Do not train or generate data.
- [ ] Run the existing prepare, preflight, 16 cell, audit, report, and immutable
  verification sequence for all 576 primary rollouts.
- [ ] Require exact cell IDs, rollout IDs, seeds, checkpoint hashes, provenance,
  success states, termination reasons, warning families, adapter counts, and zero
  exclusions.
- [ ] Require exact tick counts. Compare runtime angles/positions with absolute
  tolerance `1e-4` and forces with absolute tolerance `0.05 N`.
- [ ] Require zero new adapter rejections and zero new force-watch events.
- [ ] Preserve explicit `REVIEW_REQUIRED` treatment of the existing ACT-A3-N50
  D0 seed-112 event near `219.95 N`.
- [ ] Record all approved behavior differences caused by Phase 1 defect fixes.
  Unexplained drift blocks completion.

### Exit criteria

- Every phase is `Complete`.
- All static, hermetic, supported-runtime, simulator, artifact, and final matrix
  gates pass.
- Historical curated evidence remains byte-identical.
- The final 576-rollout package has 16 complete cells, 576 valid rollouts, and
  zero exclusions.
- Remaining limitations are explicit in `status.md`; none are hidden behind a
  passing test or refreshed expected output.

### Deferred

Any failure suggesting a protocol, schema, model, dataset, or hardware change
becomes a separately approved project. Phase 8 does not broaden this roadmap.

## 5. Planned additive public interfaces

The roadmap authorizes only these additive interfaces:

- `ActionSpace`, covering the four canonical action tags.
- Environment and chunk-policy `Protocol` definitions.
- `build_alex_v2_a4_adapter(a3, calibration)`.
- `RobotLimitsCfg.actuates_rotation`, defaulting to `False`.
- A4 failure values `environment_terminated` and `environment_truncated`.
- Warning identifier `a2.rotation_not_actuated` through the existing warning
  record schema.
- Optional installation extras `data` and `plots`.
- Compatibility facades for relocated implementation.

No existing import, field, CLI, config key, schema version, warning/failure value,
or artifact reader may be removed. If an additive change would still break an
existing consumer or canonical payload, stop and request a separately versioned
migration.

## 6. Test-to-contract matrix

Every future change must identify the affected rows before implementation.

| Contract | Minimum evidence |
|---|---|
| Public imports and CLIs | Import fixtures, parser/help snapshots, subprocess exit tests |
| Action units/frames | Known vectors, round trips, invalid rotations, physical invariants |
| Adapter behavior | Requested/applied decisions, warnings, rejection and termination tests |
| Environment execution | Boundary validation, clamps, no post-reset reads, targeted Isaac gate |
| Recording | Pre-action alignment, transactional failures, HDF5/JSON round trip |
| Dataset/splits/views | Determinism, leakage guards, exact fingerprints, tamper rejection |
| Policies/checkpoints | Legacy/current loading, provenance equality, tiny deterministic training |
| Evaluation | Golden rows, exact aggregation, exclusions, deterministic bootstrap/report |
| Cluster operations | Inventory/hash equality, secret checks, atomic publication, tamper tests |
| Packaging/runtime | Clean install/import, supported versions, optional extras, path precedence |
| Performance | Before/after time and memory plus exact output-equivalence tests |

Passing tests are necessary but insufficient when physical semantics, provenance,
or scientific claims are untested. Each phase must also verify the relevant
invariants from `PROJECT_GUIDELINES.md` and `architecture.md`.

## 7. Change discipline and rollback

- Use small reviewable changes, normally one defect or one structural boundary at
  a time.
- Keep characterization tests in the same change as the refactor they protect.
- Put repository-wide formatting in its own commit.
- Preserve old implementations behind a facade until compatibility tests pass;
  remove private dead code only in a later green change.
- Never repair a failing fingerprint or golden test by regenerating evidence
  before identifying the semantic difference.
- If a runtime phase regresses, revert that coherent change rather than layering
  compensating special cases.
- Record approved exceptions below with owner, reason, affected contract, and
  removal condition.

## 8. Approved exceptions

None.

## 9. Roadmap change log

| Date | Change | Evidence |
|---|---|---|
| 2026-07-16 | Initial approved roadmap and baseline | Clean commit `d22c5b8`; Ruff/test/format audit; curated aggregate recorded above |
| 2026-07-16 | Decomposed all phases into agent-executable units | 48 ordered units with dependencies, bounded deliverables, gates, status, and handoff requirements |
