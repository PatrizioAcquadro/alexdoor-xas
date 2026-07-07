# Phase 3.2 ACT Hardening Handoff

Date: 2026-07-07. Branch: `impl/phase3-2-act`.

Root `PROJECT_GUIDELINES.md` was not present, so this pass read
`docs/PROJECT_GUIDELINES.md` per the objective. The docs tree remains local
and ignored; this tracked handoff lives under `outputs/curated/phase3_2_act/`.

## Scope

Completed Phase 3.2 hardening only:

- `train.seed` now controls ACT model initialization before training starts.
- ACT eval can run an opt-in matched scripted reference on the same fixed and
  randomized seed plan as ACT.
- ACT eval now reports adapter warnings per rollout and in aggregate, including
  exact warning message counts in each generated `act_eval.json`.
- W&B logging remains disabled by default and now includes aggregate eval
  warning count when tracking is enabled.

No Phase 3.3, Diffusion Policy, VLA, WAM-lite, RL, or hardware-control code was
implemented.

## Dataset

Current Phase 3.2 training data is still the clean 22-episode Alex prefix:

- Task/version: `door_push_alex/v0`.
- Seeds: `0-21`.
- Split counts: train 16, validation 3, test 3.
- Scripted source run: `outputs/alex_door_push/2026-07-07_seed0_n22/`.
- Scripted source success: 21/22, 95.45 percent.

The 50-episode data-engine blocker remains open. Randomized seeds 22, 26, 30,
and 43 still hit Alex IK joint-limit windup in the larger attempted pass. That
belongs to the data engine or variation bounds, not this ACT hardening pass.

## Seedfix Training

Both runs used `train.seed=0` and the corrected seeded model initialization.

| action space | run id | best epoch | best val L1 | open-loop val L1 |
|---|---:|---:|---:|---:|
| A2 | `20260707_a2_seed0_seedfix` | 54 | 0.076795 | 0.000389 |
| A3 | `20260707_a3_seed0_seedfix` | 54 | 0.076795 | 0.000389 |

Checkpoints:

- `outputs/act_door_push/20260707_a2_seed0_seedfix/checkpoints/best.pt`
- `outputs/act_door_push/20260707_a3_seed0_seedfix/checkpoints/best.pt`

A2 and A3 remain numerically identical in this scene because the current door
frame is aligned with world axes. Door-pose randomization is still required for
an informative A2-vs-A3 comparison.

## Matched Evaluation

ACT and the matched scripted reference used the same evaluation protocol:

- Fixed rollouts: 5, seeds `100-104`.
- Randomized rollouts: 15, seeds `105-119`.
- Variation bounds: `ALEX_VARIATION_BOUNDS`.
- Success angle: 45 degrees.
- Max ticks: 600.
- Legacy embedded reference preserved:
  `outputs/alex_door_push/2026-07-07_seed0_n22/metrics/metrics.json`.

| policy | success | final angle mean rad | mean ticks | adapter accepted/corrected/rejected | warnings |
|---|---:|---:|---:|---:|---:|
| ACT-A2 | 20/20 | 0.956664 | 134 | 2680/0/0 | 319 |
| ACT-A3 | 20/20 | 0.956664 | 134 | 2680/0/0 | 319 |
| matched scripted | 20/20 | 1.048785 | n/a | n/a | n/a |

Full rows, seed protocol, legacy reference, matched reference, and exact warning
message counts are in:

- `outputs/act_door_push/20260707_a2_seed0_seedfix/metrics/act_eval.json`
- `outputs/act_door_push/20260707_a3_seed0_seedfix/metrics/act_eval.json`

Warning groups in each ACT eval:

- `joint_target_12_position_limit`: 145 warnings across 105 unique messages.
- `joint_target_20_position_limit`: 63 warnings across 56 unique messages.
- `joint_target_16_position_limit`: 1 warning.
- `joint_14_velocity_limit`: 46 warnings across 4 unique messages.
- `joint_22_velocity_limit`: 64 warnings across 6 unique messages.

These warnings are visible evidence from adapter decisions. They are not hidden,
and this pass does not weaken frozen gates or thresholds.

## Validation

All required commands passed:

```bash
ruff check .
git diff --check
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_training.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_act_rollout.py --viz none --device cpu --checkpoint-a2 outputs/act_door_push/20260707_a2_seed0_seedfix/checkpoints/best.pt --checkpoint-a3 outputs/act_door_push/20260707_a3_seed0_seedfix/checkpoints/best.pt
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_adapters.py --viz none --device cpu
```

Training and eval commands also passed:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py dataset.space=A2_ee_delta train.seed=0 run.run_id=20260707_a2_seed0_seedfix
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py dataset.space=A3_obj_rel_ee_delta train.seed=0 run.run_id=20260707_a3_seed0_seedfix
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py --viz none --device cpu --matched-scripted-reference rollout.checkpoint=outputs/act_door_push/20260707_a2_seed0_seedfix/checkpoints/best.pt rollout.reference_metrics=outputs/alex_door_push/2026-07-07_seed0_n22/metrics/metrics.json
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py --viz none --device cpu --matched-scripted-reference rollout.checkpoint=outputs/act_door_push/20260707_a3_seed0_seedfix/checkpoints/best.pt rollout.reference_metrics=outputs/alex_door_push/2026-07-07_seed0_n22/metrics/metrics.json
```

Isaac emitted expected sandbox/runtime warnings about OmniHub, display, CUDA,
and user config writes, but the CPU `--viz none` gates completed successfully.

## Caveats

- Dataset size is still small: 22 clean Alex episodes.
- A3 currently equals A2 numerically due to the aligned door frame.
- The 50-episode data-engine blocker remains deferred.
- Success-stop termination still bounds ACT post-task extrapolation at chunk
  boundaries, so ACT final-angle means are structurally lower than scripted
  final-angle means.
