# Phase 3 — Non-VLA Learned Baselines

## Objective

Introduce reproducible state-only learned baselines for A2 and A3 while keeping
all execution behind explicit adapters. The phase establishes the model-facing
dataset API, ACT and Diffusion training/inference, and closed-loop rollout
plumbing without claiming VLA capability.

## Focus

### Subphase 3.0 — Dataset Interface

#### Implementation

`src/alexdoor_xas/dataset/loader.py::EpisodeDataset` and
`A4ChunkDataset` are the supported model-facing access layer. They validate
`phase2.v1` products, assemble named observation presets, and return structured
records rather than exposing raw HDF5 fields. Presets include compact task
state, contact-augmented state, Alex joint state, and door-pose-aware inputs.

`src/alexdoor_xas/dataset/splits.py` groups exact trajectory-content hashes
before producing pose-stratified train/validation/test splits. This prevents
equivalent trajectories from crossing split boundaries.
`src/alexdoor_xas/dataset/normalize.py` computes statistics from training IDs
only and binds them to dataset and split fingerprints.
`src/alexdoor_xas/dataset/sampling.py::ChunkSampler` returns the observation at
time `t`, an action horizon beginning at `t`, and a padding mask.

The phase closeout described lazy episode access. Current
`EpisodeDataset` construction instead loads validated records eagerly, and
`by_id` performs a linear search. The current code is authoritative; the
approved quality roadmap treats O(1) indexing and evidence-driven lazy loading
as future optimization rather than completed behavior.

#### Key Decisions and Problems

- Models consume only `EpisodeDataset` or `A4ChunkDataset`; raw storage layout
  is not a training interface.
- Exact-content grouping, train-only normalization, and fingerprint binding
  make split and preprocessing provenance explicit.
- Observation modalities are state-only. No image, language, or VLA input is
  implemented.

#### Tests

- `tests/test_dataset_interface.py`, `tests/test_splits.py`, and
  `tests/test_alex_v2_dataset_fingerprint.py` cover schema validation,
  content-grouped splits, training-only statistics, and provenance failures.
- `scripts/verify_dataset_interface.py` exercises the published interface from
  a supported dataset version.

### Subphase 3.1 — Adapter v1

#### Implementation

`src/alexdoor_xas/adapters/base.py` defines the common contract: every request
is classified as accepted, corrected, or rejected and records both requested
and applied actions plus warnings. `a2.py` validates world-frame commands,
finite values, tick limits, workspace constraints, contact-entry shaping, and
joint warnings. `a3.py` validates the static door frame and transforms an A3
command into A2 before reusing the A2 safety path. `a4.py` executes a guarded
A4 chunk through A4-to-A3-to-A2 stages with timeouts and stall detection.

`src/alexdoor_xas/adapters/rollout.py` applies learned-policy actions, captures
per-tick decisions, stops on simulator termination or truncation, preserves the
pre-reset terminal state, and rejects invalid simulator state. This enforces
the architecture boundary described in [[topics/system-architecture|System Architecture]].

Three current validation gaps remain. A3 verifies orthonormality but not a
proper-rotation determinant of +1; finite scalar contact values are coerced to
Boolean rather than restricted to exact Boolean/0/1 values; and the A4
`_run_stage` helper ignores terminated/truncated values returned by its
environment-step callback. These are recorded in the approved quality roadmap
and are not claimed as repaired.

#### Key Decisions and Problems

- Policies never import or bypass adapters; corrections are observable
  evidence, not hidden policy behavior.
- A3 is door-frame data and A2 is the execution currency. A4 is a guarded
  intent sequence, not a raw simulator command.
- Position-only robot control means accepted rotational request components are
  recorded and bounded but are not actuated on the current Alex V2 benchmark.

#### Tests

- `tests/test_adapters.py` verifies validation, clamping, transforms, warnings,
  rejection paths, and staged A4 behavior.
- `tests/test_rollout_semantics.py`, `tests/test_terminal_force.py`, and
  `tests/test_settle_postcondition.py` cover later-hardened termination,
  terminal evidence, and post-rollout state semantics.
- `scripts/verify_adapters.py` supplies an Isaac-backed integration gate.

### Subphase 3.2 — ACT Baseline

#### Implementation

The ACT stack under `src/alexdoor_xas/policies/act/` implements a state-only
conditional variational transformer. Training uses the current observation as
one token, an action chunk, a latent variable, masked L1 reconstruction, and KL
regularization. Inference uses the deterministic zero latent and emits a fixed
chunk for the adapter-mediated rollout path.

`model.py::ACTModel` owns the network, `data.py` adapts the canonical dataset,
`train.py` owns optimization and validation, `policy.py` owns normalized
inference, and `checkpoint.py` stores model weights with configuration,
normalization, split, dataset, provenance, and robot-asset bindings. The
default training contract is in `configs/act.yaml`; Hydra overrides select
dataset space and version.

The later `scripts/eval_act.py --video-output` path records one successful
camera-enabled rollout to a new MP4 under `outputs/` and writes its evaluation
JSON alongside it. The path is overwrite-safe, frames the complete robot and
door, and records capture timing separately from configurable playback speed
and presentation holds. Camera-mode demo evidence stays separate from the
original checkpoint metrics and frozen unified evaluation.

The optional `--visual-room floorplan212_living_room` presentation profile
references the combined-scene living room and hallway, aligns their doorway
around the unchanged D0 task door, removes the duplicate room door, and
disables all imported physics. Its source hash, camera placement, and
disabled-physics counts are added to the sidecar. The profile changes pixels
only: ACT observations, door/contact physics, adapters, and success semantics
remain unchanged.

#### Key Decisions and Problems

- ACT predicts action chunks to model short-horizon temporal structure while
  retaining an explicit per-tick safety adapter.
- Checkpoints are self-describing and fail closed on incompatible dataset,
  split, normalization, or robot identity.
- Early V1 metrics were superseded by Alex V2 retraining and are not evidence
  for current comparative performance.

#### Tests

- `tests/test_act.py`, `tests/test_act_config.py`, and
  `tests/test_hydra_config.py` verify shapes, losses, masks, configuration, and
  checkpoint round trips.
- `scripts/verify_act_training.py` and `scripts/verify_act_rollout.py` provide
  training and simulator rollout gates.

### Subphase 3.3 — Diffusion Baseline

#### Implementation

The Diffusion stack under `src/alexdoor_xas/policies/diffusion/` implements a
state-only causal transformer that predicts noise over an action horizon.
Training uses DDPM noise injection and epsilon prediction; inference supports
DDPM or deterministic DDIM denoising. Actions are min-max mapped to `[-1, 1]`,
observations are z-score normalized, and an exponential moving average of model
weights is maintained for evaluation.

`model.py::DiffusionTransformer` owns the denoiser, `schedulers.py` implements
the diffusion schedules, `train.py` owns sampled validation and optimization,
`policy.py` owns iterative sampling, and `checkpoint.py` preserves the same
provenance classes used by ACT. `diffusers` is optional; the implemented core
does not require it.

#### Key Decisions and Problems

- ACT and Diffusion share the same canonical data and rollout contracts so
  architecture comparisons do not change the execution boundary.
- Sampled validation L1 is diagnostic, not closed-loop task success.
- No A1 learned adapter or A4 learned policy was introduced in this phase.

#### Tests

- `tests/test_diffusion.py`, `tests/test_diffusion_config.py`, and
  `tests/test_hydra_config.py` verify schedules, shapes, deterministic DDIM
  behavior, normalization, masks, and checkpoints.
- `scripts/verify_diffusion_training.py` and
  `scripts/verify_diffusion_rollout.py` exercise end-to-end training and
  adapter-mediated rollout paths.

## Version Notes

- 2026-07-05 to 2026-07-07 — Dataset interface, adapter-v1, ACT, and Diffusion
  baselines landed in staged commits.
- 2026-07-08 onward — Alex V2 migration and stabilization refreshed datasets,
  robot bindings, checkpoints, and rollout semantics while preserving the
  state-only Phase 3 model boundary.
- 2026-08-06 — Added fail-closed single-rollout ACT video capture, full-body
  framing, explicit presentation timing, and colocated camera-mode evidence.
- 2026-08-06 — Added an optional visual-only FloorPlan212 living-room and
  hallway context for ACT presentation videos without changing benchmark
  physics.
