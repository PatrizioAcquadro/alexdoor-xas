# System Architecture

AlexDoor-XAS compares explicit action representations for contact-rich humanoid
door manipulation. Its executable boundary is:

`observation → policy → action representation → adapter → environment`

The repository owns simulator task construction, scripted data generation,
dataset contracts, state-only ACT and Diffusion training, closed-loop
evaluation, and provenance. It does not own a live physical-robot controller.

## Components and Responsibilities

- **Assets and environments** — `src/alexdoor_xas/assets/` and
  `src/alexdoor_xas/envs/door_task/` load the exact robot/door geometry, apply
  the calibrated task configuration, advance CPU simulation, and expose task
  state. See [[alex-v2-benchmark|Alex V2 Benchmark]].
- **Policies** — `src/alexdoor_xas/policies/` converts observations into
  requested actions. Policies do not import adapters or directly step the
  environment.
- **Action and adapters** — `src/alexdoor_xas/action/` defines canonical spaces
  and frames. `src/alexdoor_xas/adapters/` validates, corrects, rejects,
  transforms, and executes requests. See [[action-representations-and-adapters|Action Representations and Adapters]].
- **Recording and data engine** — `src/alexdoor_xas/recording/` defines the
  episode schema; `data_engine/` generates and exports matched representations.
  See [[episode-and-dataset-contracts|Episode and Dataset Contracts]].
- **Learned policies** — `src/alexdoor_xas/policies/act/` and `diffusion/`
  train and infer through the canonical dataset and checkpoint interfaces. See
  [[learned-policy-stack|Learned Policy Stack]].
- **Evaluation** — `src/alexdoor_xas/eval/` defines success, failure, force,
  aggregation, reporting, and the unified matrix runner.
- **Cluster contracts** — `cluster_pilot/` and `cluster_sweep/` package
  non-Isaac training inputs and returns. See
  [[provenance-and-artifact-lifecycle|Provenance and Artifact Lifecycle]].

## Main Flows

### Scripted Data Generation

The environment is reset and settled, the scripted policy observes pre-action
state, the recorder stores that state and requested command, the adapter/executor
applies the bounded command, and the environment advances. Terminal contact is
recorded separately as the response to the last action. A physical episode is
then exported into matched A2, A3, A4, and, where available, A1 products.

### Learned Training

`EpisodeDataset` validates a published representation, applies a named
observation preset, split, and training-only normalization, and feeds an ACT or
Diffusion training adapter. A checkpoint packages weights together with model,
dataset, split, normalization, source, and robot identities. Training metrics
do not substitute for closed-loop evaluation.

### Learned Rollout

A checkpoint-bound policy normalizes the current state and requests A2 or A3.
The matching adapter validates and possibly corrects it; A3 is transformed to
A2. The Alex V2 executor maps the A2 translation through position-only
differential IK and steps simulation. Every requested/applied action, decision
status, warning, task state, and force sample becomes evaluation evidence.

## Runtime Separation

The workstation is authoritative for Isaac asset validation, calibration,
physical-master generation, and closed-loop evaluation. Isaac entry points
must initialize `AppLauncher` before importing `isaaclab`, `omni`, or `pxr`.
Core data, model, configuration, and cluster modules stay Isaac-free where
possible.

Gilbreth runs portable Python/PyTorch training only. Source, datasets,
normalization, and returned checkpoints cross this boundary through exact
inventories. This separation is a deliberate decision; see
[[decisions/workstation-simulation-and-non-isaac-cluster-training|Workstation Simulation and Non-Isaac Cluster Training]].

## Current Limits

- Observations are state-only; there is no visual or language policy input.
- Learned policies cover A2 and A3. There is no learned A1 adapter or learned
  A4 policy.
- Robot control is position-only for six right-arm joints; requested rotations
  are not actuated.
- No command in this repository executes a physical Alex robot.
- Simulator results and force signals do not establish physical safety.

## Primary References

- `README.md`
- `knowledge/wiki/status.md`
- `src/alexdoor_xas/data_engine/generate.py`
- `src/alexdoor_xas/adapters/rollout.py`
- `src/alexdoor_xas/policies/common/rollout_eval.py`
- `src/alexdoor_xas/eval/phase3_unified.py`

## Version Notes

- 2026-07-08 — The architecture's robot execution layer moved from provisional
  Alex V1 assumptions to the calibrated Alex V2 contract.
- 2026-07-18 — The complete cluster-training-to-workstation-evaluation path was
  verified across all sixteen Phase 3 cells.
