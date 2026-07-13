# Project Status

Documentation refreshed 2026-07-13. This page records durable development
state; raw run detail remains under ignored `outputs/` artifacts and Git
history.

## Completed

- **Phase 1:** package scaffold, local asset registry, simulator readiness,
  isolated door fixture, and deterministic reset/step gates.
- **Phase 2:** deterministic scripted push controller, proxy and calibrated
  Alex V2 execution, episode recording, A1–A4 export, metrics, failure labels,
  and reproducibility checks.
- **Phase 3.0:** fail-closed dataset/model interface, shared grouped splits,
  normalization, provenance fingerprints, and chunk sampling.
- **Phase 3.1:** adapter-v1 for A2, A3, and A4 with structured decisions,
  guarded execution, and invalid-state termination.
- **Phase 3.2:** state-only ACT training, self-contained checkpoints, and
  closed-loop adapter evaluation.
- **Phase 3.3:** state-only Diffusion Policy training, EMA checkpoints,
  DDPM/DDIM sampling, and closed-loop adapter evaluation.
- **Alex V2 migration and local stabilization:** the V2-only calibrated robot,
  five-pose N50 dataset, pose-aware A2/A3 provenance, terminal force admission,
  learned-policy contact-entry correction, and exhaustive warning evidence are
  integrated on `main`.
- **Gilbreth compatibility pilot:** ACT-A2 and Diffusion-A3 completed on A100
  80GB GPUs, the return package passed its hash checks, and Ubuntu loaded both
  returned checkpoints on CPU. Automatic symlink-free W&B publication is
  implemented and locally tested for later jobs.
- **Full nested-sweep implementation:** strict scale pose/sweep configs,
  resumable candidate generation and atomic paired publication, shared nested
  views, train-only per-view normalization, view-bound checkpoint provenance,
  exact transfer/preflight/16-cell Slurm tooling, and exact-attempt return/CPU
  checkpoint verification are implemented. The real `v3_scale_master` is
  locally published and verified at 550 episodes (110 per D0–D4), with paired
  A2/A3 exports, four nested views, and eight train-only norm files. The final
  clean-tree transfer package remains a local preparation gate.

## Local stabilization evidence

The official `door_push_alex_v2/v2_pose` dataset contains 50 episodes across
five poses with a deterministic grouped 38/6/6 train/validation/test split.
The local matrix used `core_door_pose`, seed 0, GPU training and policy
inference, CPU simulation, and DDIM-10 for the primary Diffusion cells.

| Cell | Rollouts | Result | Rejections | Peak force |
|---|---:|---:|---:|---:|
| ACT-A2 | 36 | 36/36 | 0 | 135.8 N |
| ACT-A3 | 36 | 36/36 | 0 | 129.4 N |
| Diffusion-A2 | 36 | 36/36 | 0 | 145.5 N |
| Diffusion-A3 | 36 | 36/36 | 0 | 143.7 N |

The exact A2 dataset fingerprint is
`b703af983a4bef98b73219f28b81046e564400a3a0368aeb746433b8191c29e6`;
the exact A3 fingerprint is
`01172f9a266c86d5bddd0b03e87f2835eb597e18abf5dc5b2ea6668dffb136f2`.
ACT best validation L1 was `0.03083649` for A2 and `0.03229217` for A3.

Metadata coverage, protocol consistency, and safety readiness pass across all
144 primary rollouts. Same-seed fresh-process replay passed for all 20
policy/pose files. No cell exceeded the unchanged 200 N simulation admission
bound.

All 876 warnings are structured warn-level lower-body reset-transient records,
219 in each matrix cell. The v3 adjudication envelope allows only the five
known passive knee/ankle joints during the initial reset window, with
≤4 events for one joint and ≤11 total records per rollout. Warnings remain present in the
evaluation evidence; unknown or out-of-envelope records require review.

These numbers validate the pipeline and safety-evidence contract at smoke
scale. They are not ACT-versus-Diffusion, A2-versus-A3, generalization, or
hardware-performance claims. `scripts/verify_stabilization_doc.py` checks the
artifact-bound values above against the local evidence.

## Current boundaries

- The **Gilbreth N50 compatibility pilot is complete and return-verified**.
  Its automatic W&B durable-publication fix was merged afterward and still
  requires one live two-cell canary before full-sweep execution.
- The **full cluster dataset-scale sweep has not started**. Its nested
  N50/N100/N250/N500 implementation is local-only preparation. No sweep package
  has been transferred, no 16-cell array has been submitted, and no full-sweep
  training result exists.
- **Phase 4 VLA work has not started**. There is no image/VLA observation
  pipeline, OpenVLA fine-tuning, or mixed-action-space model yet.
- A4 is recorded and adapter-executable but has no learned A4 policy.
- Local simulation safety readiness is not Alex hardware-readiness evidence.
  No real-door trial is authorized.

## Known limitations

- The `v2_pose` stabilization dataset is smoke-scale. The 550-episode scale
  master still covers only one door family in simulation.
- Learned baselines are state-only; camera observations and language inputs are
  not implemented.
- Simulator evaluation stays on CPU under the frozen calibration contract.
- Force evidence is the door-panel-filtered EE force; broader collision and
  slip sensing are unavailable.
- Adapter-v1 does not implement acceleration limits, general collision
  checking, or tangential slip detection.
- Machine-local Alex and scene assets are required and are not distributed by
  this repository.

## Next development sequence

1. Run and return-verify one two-cell Gilbreth canary using the automatic W&B
   durable-publication path, without manual return staging.
2. Build and validate the final clean-tree transfer manifest and 16-cell Slurm
   matrix locally from the verified scale master.
3. Transfer and hash-verify the committed sweep package on Gilbreth only after
   separate authorization.
4. Launch the full sweep only after every prior gate passes and submission is
   separately authorized.
