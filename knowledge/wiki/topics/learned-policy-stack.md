# Learned Policy Stack

AlexDoor-XAS implements two state-only policy families—ACT and Diffusion—for A2
and A3. Both consume the same canonical dataset interface, package complete
provenance in checkpoints, and execute only through explicit adapters.

## Shared Contract

`src/alexdoor_xas/policies/common/data.py` and
`src/alexdoor_xas/dataset/loader.py` supply validated observations and actions.
Named observation presets define the state vector; there are no images or
language tokens. Splits are content-grouped and normalization uses training IDs
only.

`src/alexdoor_xas/policies/common/obs.py` assembles online observations using
the same preset contract as training. `common/rollout_eval.py` manages reset,
settle, policy inference, adapter decisions, environment stepping, first
success crossing, terminal state, and force evidence.

Checkpoint modules store:

- model weights and model configuration;
- action space and observation preset;
- normalization values and identity;
- dataset, split, and view fingerprints;
- source and training metadata;
- robot asset and calibration provenance.

A mismatch is an error, not an instruction to reinterpret the checkpoint.

## ACT

`src/alexdoor_xas/policies/act/model.py::ACTModel` is a conditional
variational transformer for action chunks. Training encodes the action chunk
into a latent variable, conditions on the current observation token, and
optimizes masked L1 reconstruction plus KL regularization. Inference uses the
zero latent for deterministic chunk prediction.

The default `configs/act.yaml` uses a 40-step chunk. The unified evaluation
executes ACT without temporal ensembling. `act/train.py` owns optimization and
validation; `act/policy.py` normalizes observations and denormalizes predicted
actions; `act/checkpoint.py` enforces self-contained serialization.

`scripts/eval_act.py --video-output` can record exactly one camera-enabled
closed-loop rollout to a new MP4 under `outputs/`. The same run writes a JSON
sidecar with checkpoint identity, rollout metrics, and video metadata. Capture
fails rather than publishing a missing-frame, unsuccessful, or overwriting
result. The presentation path uses full-robot/door framing, records at the
control rate, and explicitly records a configurable playback rate plus
intro/outro holds; camera-mode evidence remains separate from the frozen
headless evaluation.

Video mode can additionally compose the FloorPlan212 living room and hallway
from the registered combined scene around the unchanged D0 task door. The
presentation layer removes the duplicate room door, disables every imported
collider/rigid-body behavior, aligns the rotated source and task doorframe
bounds, and warms the renderer before capture. It records the source hash,
camera profile, alignment errors, warm-up count, and disabled-physics counts.
It changes rendered pixels only; ACT remains state-only and the calibrated
task, contact filter, and rollout evidence are unchanged.

## Diffusion

`src/alexdoor_xas/policies/diffusion/model.py::DiffusionTransformer` is a
causal transformer epsilon predictor over an action horizon. Training injects
DDPM noise and predicts it; an exponential moving average of weights is used
for evaluation. Observations use z-score normalization and actions are min-max
mapped to `[-1, 1]`.

`diffusion/schedulers.py` implements DDPM and deterministic DDIM schedules.
The unified protocol predicts 16 steps, executes eight, and samples with 10
DDIM inference steps. Sampled validation L1 diagnoses denoising behavior but is
not a task-success metric.

## Training and Evaluation Boundary

Training can run locally or in the non-Isaac Gilbreth environment. Closed-loop
evaluation runs on the authoritative Isaac workstation. The sixteen returned
scale-sweep checkpoints were all CPU-load verified before evaluation; see
[[experiments/gilbreth-nested-scale-sweep|Gilbreth Nested Scale Sweep]] and
[[experiments/phase-3-unified-evaluation|Phase 3 Unified Evaluation]].

ACT and Diffusion do not import or bypass adapters. A2 executes after
validation; A3 is transformed through the verified door frame and then passed
through A2. Requested, corrected, rejected, and applied actions remain part of
the evaluation record.

## Limits

- Only A2 and A3 have learned policies.
- Models are state-only and benchmark-specific; they are not VLA systems.
- Current success saturation does not establish policy equivalence or a winner.
- Training loss and successful checkpoint return do not establish closed-loop
  behavior.

## Primary References

- `configs/act.yaml`
- `configs/diffusion.yaml`
- `src/alexdoor_xas/policies/common/rollout_eval.py`
- `src/alexdoor_xas/policies/act/model.py`
- `src/alexdoor_xas/policies/act/train.py`
- `src/alexdoor_xas/policies/diffusion/model.py`
- `src/alexdoor_xas/policies/diffusion/train.py`
- `tests/test_act.py`
- `tests/test_diffusion.py`

## Version Notes

- 2026-07-06 to 2026-07-07 — ACT and Diffusion state-only baselines landed.
- 2026-07-16 to 2026-07-18 — Sixteen scale checkpoints trained on Gilbreth and
  completed the matched workstation evaluation.
- 2026-08-06 — ACT evaluation gained fail-closed single-rollout MP4 capture,
  full-body framing, explicit playback timing and holds, and a colocated
  camera-mode evidence sidecar.
- 2026-08-06 — ACT video mode gained an auditable, visual-only FloorPlan212
  living-room and hallway context around the unchanged task door.
- 2026-08-06 — Rotated doorway placement switched from hinge origins to full
  doorframe bounds, with renderer warm-up before capture.
