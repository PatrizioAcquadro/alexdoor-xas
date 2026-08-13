# Visuoproprioceptive Generalization Benchmark

## Context

The completed state-only Phase 3 study covered A2/A3, ACT/Diffusion, one simulated
door family, and one training seed. All 576 closed-loop rollouts succeeded. It
validated the pipeline, but its saturated success rate could not identify a
better action representation or policy model.

The next benchmark must therefore make geometric generalization, rather than
single-scene task completion, the main source of difficulty.

## Scientific Question

Holding Alex V2, observations, physical demonstrations, dataset size, training
budget, and evaluation protocol fixed, how do A1-A4 action representations and
ACT/Diffusion policy models affect zero-shot success on geometrically unseen
push doors?

Zero-shot means that a policy acts on complete USD door assets excluded from its
training data, without retraining or adaptation. The primary comparison is the
complete `4 action representations x 2 policy models` matrix. Future VLA models
will enter the same benchmark as an extension, not change its central question.

## Benchmark Boundary

The current benchmark becomes **B0**, a state-only single-door regression test.
It remains useful for software verification but not for scientific ranking.

The new study is **B1**, a visuoproprioceptive multi-door generalization
benchmark:

- fixed-base Alex V2 torso and the six-joint right arm;
- push-door opening without a handle-operation subtask;
- front RGB, wrist RGB, and robot proprioception as policy observations;
- 24 qualified door assets split into 12 train, 4 development, and 8 sealed test
  identities;
- A1-A4 x ACT/Diffusion, five independent training seeds per cell;
- no test-door data for policy or estimator training, model selection, or
  hyperparameter tuning.

The robot base does not move. Pose variation changes the door frame relative to
the robot, within an expert-qualified reachable region.

## Assets and Split

Each door identity is a complete articulated asset, not another pose, material,
or randomization of a training door. Assets must be suitable for a future public
release and are accepted only under CC0, CC BY, or an explicitly equivalent
license. Assets with unclear, non-commercial, no-derivatives, or share-alike
terms are excluded.

Candidate assets are first normalized and qualified with the privileged expert.
Only then are the 12/4/8 identities and all benchmark parameters frozen. Every
accepted asset records its source, license, attribution, modifications, stable
identifier, and checksum in a release manifest.

After the split is frozen, test assets are sealed from learned-system
development. Their only prior use is the automated expert qualification needed
to prove task feasibility and set the common task threshold; qualification
trajectories and images never enter a learned component.

## Observations and Perception

All eight main comparison cells receive the same synchronized front RGB, wrist
RGB, and proprioceptive history. Simulator door pose, articulation state, depth,
segmentation, and privileged contacts are not policy inputs.

A frozen visual-backbone pilot will choose between DINOv3 ViT-B/16 and SigLIP 2
Base Patch16 using only train/development assets. The selected backbone,
preprocessing, feature dimensions, and observation history are then fixed for
every action representation and policy model.

A separate door-frame and articulation-state estimator supplies the geometric
state required to execute A3 and A4. It is trained only on training assets,
selected only on development assets, and frozen before test evaluation. Main
results are end-to-end with estimated geometry. An oracle simulator-state run is
reported only as a diagnostic to separate perception failures from action-policy
failures; it is not a benchmark result or model-selection signal.

## Action-Representation Contract

Every representation must be trainable and executable under both ACT and
Diffusion before B1 data production begins.

| Representation | B1 policy output and execution contract |
|---|---|
| A1 | Deltas for the six active right-arm joint targets, executed directly in joint space without differential IK. |
| A2 | World-frame 6D end-effector deltas, executed through the maintained robot controller. |
| A3 | Door-frame-relative 6D end-effector deltas, transformed using the frozen estimated door frame and then executed through A2. |
| A4 | A complete object-centric approach/contact/push sequence. The adapter validates, transforms, and executes the predicted sequence but never invents omitted phases. |

All four exports come from the same accepted physical episode and synchronized
observations. They share episode identity, split, outcome, and rollout seeds,
while retaining representation-specific action arrays and normalization.

## Data Strategy

The dataset is simulation-first and progressively hybrid:

- a privileged scripted expert is the primary trajectory source;
- vanilla Isaac Sim Replicator APIs provide visual randomization, annotations,
  and synchronized capture;
- an Isaac Lab Mimic pilot decides whether transformed demonstrations add valid
  spatial diversity before Mimic is admitted to production;
- targeted teleoperation covers recoveries or qualified doors the scripted
  expert cannot cover well;
- an RL teacher is deferred until handles or broader articulated objects expose
  coverage that scripted, Mimic, and teleoperated data cannot provide.

Teacher source is explicit for every episode. Representation comparisons use
the same accepted underlying episodes, the same fixed dataset size, and the same
sampling rule. Test doors produce evaluation rollouts only, never demonstrations.

## Door Qualification and Success

The fixed-base setup must not impose an arbitrary success angle that some doors
cannot reach. For every candidate door, the privileged expert runs 20 trials.
For door `d`, `q_d` is the 10th percentile of the maximum opening angle sustained
for at least 0.5 seconds.

The primary angle is frozen before learned-policy training:

`theta_primary = floor_to_5deg(min_d(q_d) - 5deg)`

The subtraction provides a fixed margin and rounding makes the criterion easy
to interpret. A candidate is replaced if its margin-adjusted reliable angle is
below 45 degrees. A rollout succeeds when it reaches and sustains
`theta_primary` for at least 0.5 seconds within the fixed episode horizon.

## Evaluation Conditions

B1 separates different kinds of distribution shift:

| Tier | Purpose |
|---|---|
| ID | Seen training-door identities under nominal pose, lighting, and dynamics. |
| GEO | Sealed door identities with nominal pose, lighting, and dynamics; the primary generalization result. |
| POSE | Seen identities under held-out reachable door-frame poses. |
| LIGHT | Seen identities under held-out illumination conditions. |
| DYN | Seen identities under held-out friction, damping, and inertial conditions. |
| COMPOUND | Sealed identities with simultaneous held-out pose, lighting, and dynamics. |

POSE, LIGHT, and DYN isolate individual stressors before COMPOUND combines them.
The exact ranges are derived from the train/development pilot and frozen before
test evaluation.

Each of the eight model/representation cells uses five independent training
seeds, producing 40 selected checkpoints. Checkpoint selection uses development
assets only. Every checkpoint receives 20 paired rollouts per evaluated door;
initial conditions and evaluation seeds are matched across cells.

## Metrics

The scientific ranking is based on generalization, not force or motion quality.
The benchmark reports:

- primary success rate at `theta_primary`;
- `success@45deg`, `success@60deg`, and `success@75deg`;
- maximum door angle reached;
- expert-normalized progress per door;
- seen-to-held-out generalization gap;
- per-door results and variation across training seeds.

The primary result is GEO success. COMPOUND success measures robustness beyond
geometry alone. ID and the isolated OOD tiers explain where performance is lost.
Expert-normalized progress compares the policy's maximum angle with `q_d`, so
policy quality can be distinguished from a door's fixed-base reachability limit.
Safety checks remain validity gates and diagnostics, not the main ranking target.

## Pilot-Calibrated Values

The benchmark structure above is fixed. A train/development-only pilot must set
the following once before the sealed evaluation:

- the selected frozen visual backbone and observation-history length;
- the fixed number of accepted training episodes used by every matrix cell;
- the A4 sequence length and fixed tensor encoding;
- nominal and OOD randomization ranges;
- whether Mimic data passes validity and usefulness gates;
- the final expert-qualified assets and `theta_primary`.

These values and acceptance rules are recorded before full training. They cannot
be changed after inspecting test-policy results.

## Public-Release Contract

B1 is designed from the start for publication. A release includes the eligible
asset manifest, fixed splits, benchmark configuration, observation and action
schemas, estimator contract, dataset-generation recipe, training/evaluation
commands, aggregate and per-door results, and required attribution. Release
artifacts must be sufficient to reproduce the protocol without exposing test
data to training code.

## Progression

1. B1: different push doors, followed by POSE, LIGHT, DYN, and COMPOUND tests.
2. Handle operation and generalization across different handles.
3. Doors, cabinets, drawers, and other articulated objects.

Each stage first tests unseen instances of the same task family. B1 does not
claim transfer from pushing directly to handle operation or other objects.

## Implementation Order

1. Collect and qualify candidate door assets.
2. Freeze the 24 accepted assets, 12/4/8 split, sealed test set, and primary
   success angle.
3. Build the multi-door environment, cameras, and Replicator path.
4. Select the visual backbone and train and freeze the door-frame and
   articulation-state estimator.
5. Correct and complete the learned A1/A4 action, dataset, and execution
   contracts.
6. Run the train/development data-generation pilot.
7. Freeze the remaining dataset and protocol values and generate the final
   matched B1 dataset.
8. Train the 40 ACT/Diffusion checkpoints.
9. Run ID, GEO, POSE, LIGHT, DYN, and COMPOUND evaluation.

The execution plan and the gate for each step are recorded in
[[implementation_phases/phase-4-visuoproprioceptive-generalization-benchmark|Phase 4 — Visuoproprioceptive Generalization Benchmark]].

## Status

This page records the approved scientific and benchmark design. B1, learned
A1/A4 support, visual observations, multi-door assets, the perception estimator,
the new dataset, and multi-seed evaluation have not yet been implemented.
