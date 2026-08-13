# Phase 4 — Visuoproprioceptive Generalization Benchmark

> Planned phase. None of the implementation or result claims below are complete.
> The scientific contract is defined in
> [[decisions/visuoproprioceptive-generalization-benchmark|Visuoproprioceptive Generalization Benchmark]].

## Objective

Build and execute B1: a reproducible visuoproprioceptive benchmark that compares
A1-A4 x ACT/Diffusion on geometrically unseen push doors without retraining.

## Subphase 4.1 — Asset Collection and Qualification

#### Implementation

Execute the following bounded acquisition and qualification loop:

0. Freeze the admission, normalization, and qualification criteria below.
1. Implement only the local scripts required to check, normalize, and test candidates.
2. Use an online-search agent to find 24 candidates, applying every inexpensive remote check available before download and preferring USD-family sources.
3. Download the candidates, run checks that require local files, and replace failures through another targeted online search until 24 remain.
4. Normalize the 24 candidates to the common door contract.
5. Run static and physics tests, replacing failures by repeating Steps 2-4 until 24 pass.
6. Run one nominal expert rollout per door to verify that it opens.
7. Reset and repeat the identical rollout. Compare outcome, angle curve, maximum sustained angle, and errors. If the pair is outside tolerance, run three additional identical rollouts and diagnose the source of variation.
8. Produce the candidate manifest and qualification report, recommend the common Subphase 4.2 rollout count from the observed repeatability, and update the canonical documentation.

The online gate checks the per-asset license, source, available format, archive size, reported triangle count, visible frame/panel separability, dimensions when stated, and duplicates before download. Values unavailable or unreliable online are checked locally. Search and replacement continue only until 24 assets pass; no unused reserve set is collected.

#### Key Decisions

- Accept only CC0 or CC BY 4.0 covering the model and every redistributed dependency. Record source URL, author, retrieval date, license evidence, attribution, modifications, identifier, and checksums. Reject unclear, custom, royalty-free, NC, ND, and SA terms.
- Use realistic full-size single-leaf interior, exterior, or industrial push doors. Exclude gates, double or sliding doors, cabinet doors, and fantasy doors. A non-actuated handle may remain rigidly attached to the panel.
- Require a separable frame and panel, one vertical revolute hinge, no latch operation, and a plausible panel size: `0.65-1.20 m` wide, `1.80-2.40 m` high, and `0.025-0.10 m` thick.
- Qualify exactly 12 left-hinged and 12 right-hinged doors. Recolors, mirrors, or variants of one base mesh are not separate identities. There is no per-author or per-pack quota.
- Prefer USD/USDZ, followed by GLB/glTF, Blend/FBX, and OBJ. A non-USD source is admissible only when it can be converted without substantial remodeling.
- Permit format conversion, uniform scaling, simple frame/panel separation, pivot correction, materials, colliders, and articulation. Do not deform or substantially remodel source geometry.
- Limit normalized assets to 250,000 visual triangles and 4K textures. Reject candidates that require major decimation.
- Normalize every asset to meters, Z-up, closed angle `0`, positive opening direction, canonical frame/panel semantics, local redistributable dependencies, and one common nominal physics template. Mass, hinge damping, friction, joint limits, and collision policy are shared; inertia is derived consistently from each normalized geometry.
- Static tests verify a loadable dependency-complete USD, canonical units and prims, dimensions and complexity, one valid vertical hinge, local textures, colliders, and positive mass properties. Physics tests verify a fixed frame, one stable door degree of freedom, closed reset, no passive drift, interpenetration, NaN, or simulation explosion, and unobstructed mechanical opening.
- A nominal expert rollout passes when it reaches at least 45 degrees for 0.5 seconds without runtime errors, invalid physics, or a force-admission violation.
- Reset repeatability passes when the two identical rollouts have the same outcome and termination, maximum sustained angles within 2 degrees, and a maximum time-aligned angle-curve error of 3 degrees. A failure triggers three additional diagnostic repeats rather than immediate rejection.
- Asset-specific physical instability causes replacement. Otherwise, the bounded variation observed across the 24 accepted assets determines one common `n_qual` for Subphase 4.2; 20 is not fixed in advance.

#### Problems / Limitations

This subphase is complete only when exactly 24 legally eligible, normalized, technically valid, nominally openable assets pass the repeatability gate and the Subphase 4.2 qualification protocol has been calibrated. These assets remain provisional: a later reliable-angle failure reopens Subphase 4.1 and replaces that asset.

## Subphase 4.2 — Benchmark and Split Freeze

#### Implementation

Run the frozen common `n_qual` expert trials on the 24 provisional doors. Compute
each door's reliable angle as the 10th percentile of its maximum sustained angle,
replace any failure through Subphase 4.1, then assign 12 train, 4 development,
and 8 sealed test identities and freeze the asset manifest. Set the common
primary threshold to the minimum reliable angle minus 5 degrees, rounded down to
5 degrees.

#### Key Decisions

- Replace any door whose margin-adjusted reliable angle is below 45 degrees.
- Freeze one common `n_qual` before official qualification; do not adapt the count per door.
- Test qualification data prove feasibility only and never train a learned component.
- Freeze the split and primary threshold before learned-system development.

#### Problems / Limitations

After this gate, test identities cannot influence perception training, policy
training, model selection, randomization design, or benchmark tuning.

## Subphase 4.3 — Multi-Door Environment and Visual Capture

#### Implementation

Build one asset-indexed Isaac Lab environment with consistent door-frame and
articulation conventions. Add synchronized front RGB, wrist RGB, and
proprioception, then use vanilla Isaac Sim Replicator APIs for visual
randomization, annotations, and recording.

#### Key Decisions

- Preserve the fixed Alex V2 base and six-joint right-arm task boundary.
- Keep simulator state, depth, segmentation, and privileged contacts out of policy observations.
- Make asset identity, scene condition, and randomization seed explicit and reproducible.

#### Problems / Limitations

All 24 assets must pass the same reset, stepping, sensing, and expert-execution
contracts before perception or dataset work begins.

## Subphase 4.4 — Frozen Perception Stack

#### Implementation

Compare DINOv3 ViT-B/16 and SigLIP 2 Base Patch16 using train/development assets,
select one frozen visual backbone, and train the door-frame and
articulation-state estimator. Freeze its preprocessing, observation history,
features, checkpoint, and output contract before policy comparison.

#### Key Decisions

- Train only on training-door data and select only on development doors.
- Supply estimated geometry to A3/A4 in the main benchmark.
- Use oracle simulator geometry only as a separately labeled diagnostic.

#### Problems / Limitations

The estimator must meet a development-set accuracy and closed-loop usability
gate defined before training. Test-door perception cannot tune that gate.

## Subphase 4.5 — Complete A1 and A4

#### Implementation

Add full ACT/Diffusion dataset, model-output, normalization, adapter, and rollout
support for A1 and A4. A1 predicts six active right-arm joint-target deltas and
executes them directly without differential IK. A4 predicts the complete
object-centric approach/contact/push sequence; its adapter validates,
transforms, and executes the sequence without inventing missing phases.

#### Key Decisions

- Keep A2/A3 behavior stable except for the shared visuoproprioceptive input path.
- Make the A4 tensor encoding and sequence boundary explicit and testable.
- Require all eight ACT/Diffusion x A1-A4 paths to pass the same smoke contract.

#### Problems / Limitations

A1/A4 must be executable before the data pilot. A3/A4 execution also depends on
the frozen estimator contract from Subphase 4.4.

## Subphase 4.6 — Data-Generation Pilot

#### Implementation

Run a small train/development-only pilot through the complete recording,
matched-export, loading, training-smoke, and rollout path. Use the scripted
expert as the primary teacher, Replicator for visual variation, a gated Isaac
Lab Mimic trial for spatial diversity, and targeted teleoperation only for
identified coverage gaps.

#### Key Decisions

- Export the same accepted physical episodes and observations to A1-A4.
- Keep teacher source explicit; defer an RL teacher until later task families require it.
- Calibrate dataset size, A4 sequence length, teacher mix, and OOD ranges without test data.

#### Problems / Limitations

Pilot data are engineering evidence, not final benchmark data. Production begins
only after validity, synchronization, coverage, and learnability gates pass.

## Subphase 4.7 — Protocol Freeze and Final Dataset

#### Implementation

Freeze the remaining B1 values: dataset size, accepted teacher mix, observation
history, A4 encoding, randomization ranges, normalization, training budget, seed
list, checkpoint-selection rule, and evaluation configuration. Then generate
the final matched A1-A4 dataset from training doors and publish its manifests and
split artifacts.

#### Key Decisions

- Use one fixed accepted episode set and dataset size for every matrix cell.
- Generate development/test evaluation conditions but no demonstrations from test doors.
- Make the frozen configuration the only input to full training and evaluation.

#### Problems / Limitations

Any later protocol change invalidates comparability and requires a new benchmark
version. Final data must pass schema, pairing, split-leakage, and replay checks.

## Subphase 4.8 — Forty-Checkpoint Training Matrix

#### Implementation

Train ACT and Diffusion for A1-A4 with five independent seeds per cell, producing
`2 x 4 x 5 = 40` selected checkpoints. Hold dataset membership, observation
inputs, optimization budget, and checkpoint-selection procedure fixed across
the matrix.

#### Key Decisions

- Use development doors only for checkpoint selection and permitted tuning.
- Keep each seed independent and report variation instead of only the best run.
- Use the workstation GPU for Isaac workloads and Gilbreth A100s for non-Isaac training.

#### Problems / Limitations

All 40 checkpoints must be complete and loadable before sealed evaluation.
Training failures are reported, not silently replaced with extra favorable runs.

## Subphase 4.9 — Generalization Evaluation

#### Implementation

Evaluate every selected checkpoint with 20 paired rollouts per evaluated door in
ID, GEO, POSE, LIGHT, DYN, and COMPOUND conditions. Match initial conditions and
evaluation seeds across cells and keep the sealed protocol unchanged.

#### Key Decisions

- Rank the main study by GEO success at the frozen primary angle.
- Also report COMPOUND robustness, isolated stress tiers, `success@45/60/75`,
  maximum angle, expert-normalized progress, seen-to-held-out gap, per-door
  results, and variation across training seeds.
- Treat safety checks as validity gates and diagnostics, not the primary ranking.

#### Problems / Limitations

Results support conclusions about unseen push-door instances only. They do not
establish transfer to handles, drawers, other objects, hardware, or sim-to-real.

## Artifacts

- Licensed asset manifest, qualification evidence, frozen 12/4/8 split, and threshold.
- Multi-door environment, visual-capture configuration, and frozen estimator.
- Complete A1-A4 contracts and matched final dataset.
- Forty ACT/Diffusion checkpoints and sealed evaluation results.
- Public-release recipe, configurations, attribution, and aggregate/per-door report.

## Files

Planned main repository surfaces:

- `src/alexdoor_xas/assets/`
- `src/alexdoor_xas/envs/`
- `src/alexdoor_xas/action/`
- `src/alexdoor_xas/adapters/`
- `src/alexdoor_xas/recording/`
- `src/alexdoor_xas/data_engine/`
- `src/alexdoor_xas/dataset/`
- `src/alexdoor_xas/policies/`
- `src/alexdoor_xas/eval/`
- `configs/`
- `scripts/`
