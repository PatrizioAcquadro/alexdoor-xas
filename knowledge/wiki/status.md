# Project Status

Documentation aligned with the technical wiki on 2026-08-11. This page records durable development
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
  views, exact train-only normalization recomputation, dual source/action
  fingerprints, exact resolved-cell checkpoint provenance, complete candidate
  ledger verification with full 750-row raw replay, calibration
  self-fingerprint/master-identity validation, exact transfer/preflight/16-cell
  Slurm tooling, and directly transferable exact-attempt return controls are
  implemented. The real `v3_scale_master` is locally published and verified at
  550 episodes (110 per D0-D4), with paired A2/A3 exports, four nested views,
  and eight train-only norm files.
- **Full Gilbreth nested sweep:** attempt `11281591` completed all 16 ACT and
  Diffusion cells from source commit
  `efa39434a123dab4d029f5f4ffdb122844892a6d`, with zero failed cells and 16
  best checkpoints. The 736-file payload inventory, both return-control hashes,
  complete provenance, and all 16 CPU checkpoint loads passed on Ubuntu. The
  returned package is retained byte-for-byte with a separate 738-file local
  hash/size inventory.
- **Phase 3 unified evaluation:** all 16 returned checkpoints completed the
  frozen D0-D4 primary protocol: 36 rollouts per cell and 576 total. Every pose
  artifact passed schema and provenance validation, and the evidence audit
  excluded zero cells. Evaluation used CPU simulation, CUDA policy inference,
  adapter-v1, a 45-degree first-crossing success threshold, 600 ticks, ACT-40
  without temporal ensembling, and Diffusion DDIM-10/Tp16/Ta8.

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
≤4 events for one joint and ≤11 total records per rollout. Warnings remain
present in the evaluation evidence; unknown or out-of-envelope records require
review.

These numbers validate the pipeline and safety-evidence contract at smoke
scale. They are not ACT-versus-Diffusion, A2-versus-A3, generalization, or
hardware-performance claims. `scripts/verify_stabilization_doc.py` checks the
artifact-bound values above against the local evidence.

## Phase 3 unified evaluation evidence

The returned scale sweep and Ubuntu closed-loop matrix are complete and
provenance-valid. Every cell achieved 36/36 successes; the per-cell 95% Wilson
interval is `[90.4%, 100.0%]`. All 36 matched success outcomes tie for every
A3-versus-A2 and Diffusion-versus-ACT comparison. The benchmark therefore does
not select an action space, policy family, or N50/N100/N250/N500 dataset size
on success rate.

Secondary behavior is heterogeneous. Mean ticks to success do not improve
monotonically with dataset size, and paired timing, force, and adapter-correction
directions change across policy/action-space/size cells. This does not support
a general A3-over-A2 or Diffusion-over-ACT claim. Exact continuous summaries,
95% deterministic bootstrap intervals, and all matched comparisons are under
`outputs/curated/phase3_unified_evaluation/aggregate_summary.json`.

All 576 rollouts recorded force-sensor contact. Across 57,678 adapter decisions,
54,183 were accepted, 3,495 corrected, and zero rejected. All 3,506 warning
records are in the established `a2.joint_velocity_limit` family. One
ACT-A3-N50 D0 randomized rollout (seed 112) exceeded the 200 N force watch
bound for one tick and peaked at 219.95 N; that cell is `REVIEW_REQUIRED`.
The remaining 15 cells pass the bounded simulation audit. This isolated watch
event must not be hidden or interpreted as hardware-safe behavior.

A minimal follow-up diagnostic replayed the original seed prefix 100–112 in a
fresh process and reproduced the seed-112 target row exactly, including the
tick-55 peak and trace hash. Changing only seed 112's initial door-frame X
offset by −1 mm and +1 mm reduced the peak to 66.00 N and 86.44 N,
respectively, with no force-watch exceedance and no rejection. This supports
local sensitivity to initial door-normal position and contact-entry motion; it
does not prove root cause, validate an adapter change, or clear the original
review status. The compact evidence is under
`outputs/curated/phase3_seed112_force_diagnostic/`.

Diffusion diagnostic evidence remains incomplete. Training binds horizon 16
and uses a 10-step DDIM sampled validation metric, while the primary matrix
freezes DDIM-10/Tp16/Ta8, but no controlled closed-loop sampler or horizon
comparison exists. Training used only seed 0; matched rollout seeds enable
paired evaluation but do not establish robustness across training seeds.

## Current boundaries

- The **full Gilbreth dataset-scale sweep and Ubuntu Phase 3 unified evaluation
  are complete** for attempt `11281591`; no additional cluster run, training,
  or dataset generation is authorized by this evidence.
- **Phase 4 VLA work has not started**. There is no image/VLA observation
  pipeline, OpenVLA fine-tuning, or mixed-action-space model yet.
- A4 is recorded and adapter-executable but has no learned A4 policy.
- Local simulation safety readiness is not Alex hardware-readiness evidence.
  No real-door trial is authorized.
- The 576-rollout result does not establish general geometry, viewpoint,
  language, door-family, training-seed, or sim-to-real generalization.

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

## Phase 4 planning recommendations

1. Treat the all-success matrix as pipeline validation and a saturated
   benchmark, not proof that A3, Diffusion, or larger datasets are superior.
2. Treat the ACT-A3-N50 force review as investigated but not cleared. Use its
   demonstrated contact-start sensitivity when designing the next dataset and
   require a post-training multi-seed and contact-start robustness gate.
3. If sampler or horizon choice matters to a later decision, design a small,
   separately authorized controlled diagnostic; current evidence is
   inconclusive.
4. Any Phase 4, VLA, A4-learning, RL, WAM-lite, hardware, fake-door, or
   sim-to-real execution requires a new scope and explicit authorization.
