# Unified Phase 3 Scientific Evaluation

## Confirmed artifact facts

- Gilbreth attempt `11281591` used source commit `efa39434a123dab4d029f5f4ffdb122844892a6d`.
- Completed primary cells: 16/16.
- Included primary rollouts: 576/576.
- Excluded cells: 0.
- All included rows use the frozen D0-D4 seed/pose protocol, adapter-v1, CPU simulation, CUDA inference, a 45 degree threshold, and 600 ticks.

## Methods

Each checkpoint was evaluated on 36 matched rollouts: 20 at D0 and four each at D1-D4. ACT uses horizon 40 without temporal ensembling. Diffusion uses DDIM-10 with Tp=16 and Ta=8. Success intervals are 95% Wilson intervals; continuous and paired mean-difference intervals use 10,000 deterministic bootstrap resamples with seed 3407. No missing value is imputed.

## Main findings

- Every cell achieved 36/36 successes. The per-cell 95% Wilson interval is [90.4%, 100.0%], so the finite benchmark supports high success under this protocol but does not establish a true 100% rate.
- Because all 576 matched outcomes were successes, every A3-versus-A2 and Diffusion-versus-ACT success difference is exactly zero (36 ties, no discordant pairs). Success therefore does not select an action space, policy family, or dataset size in this matrix.
- Continuous behavior does differ, but directions change with dataset size. No policy family or action space is uniformly faster, lower-force, or less intervention-prone across N50/N100/N250/N500.
- All 576 rollouts recorded force-sensor contact and none was adapter-rejected. One rollout exceeded the 200 N force watch bound: ACT-A3-N50, D0 randomized seed 112, peaked at 219.95 N for one tick. That cell is `REVIEW_REQUIRED`; the other 15 cells pass the bounded safety audit.
- These conclusions concern one matched simulator benchmark and seed-0-trained checkpoints. They are not evidence of training-seed robustness, broader generalization, hardware safety, or sim-to-real readiness.

## Cell results

| Run | Success | 95% Wilson CI | Mean final angle (deg) | Peak force (N) | Safety |
|---|---:|---:|---:|---:|---|
| `sweep_act_a2_n50_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.41 | 167.58 | PASS |
| `sweep_act_a3_n50_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.48 | 219.95 | REVIEW_REQUIRED |
| `sweep_diffusion_a2_n50_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.40 | 173.06 | PASS |
| `sweep_diffusion_a3_n50_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.46 | 156.66 | PASS |
| `sweep_act_a2_n100_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.36 | 129.02 | PASS |
| `sweep_act_a3_n100_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.30 | 134.00 | PASS |
| `sweep_diffusion_a2_n100_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.41 | 152.80 | PASS |
| `sweep_diffusion_a3_n100_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.40 | 138.56 | PASS |
| `sweep_act_a2_n250_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.48 | 132.41 | PASS |
| `sweep_act_a3_n250_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.47 | 128.58 | PASS |
| `sweep_diffusion_a2_n250_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.49 | 145.10 | PASS |
| `sweep_diffusion_a3_n250_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.39 | 121.50 | PASS |
| `sweep_act_a2_n500_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.41 | 134.12 | PASS |
| `sweep_act_a3_n500_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.43 | 123.74 | PASS |
| `sweep_diffusion_a2_n500_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.42 | 131.36 | PASS |
| `sweep_diffusion_a3_n500_seed0` | 36/36 (100.0%) | [90.4%, 100.0%] | 45.40 | 148.38 | PASS |

## Matched comparisons

Positive values mean the named right-hand method/action space was higher.

| Comparison | Success difference | 95% paired bootstrap CI | Right wins | Left wins |
|---|---:|---:|---:|---:|
| A3 minus A2: act N50 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: act N100 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: act N250 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: act N500 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: diffusion N50 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: diffusion N100 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: diffusion N250 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| A3 minus A2: diffusion N500 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A2_ee_delta N50 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A2_ee_delta N100 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A2_ee_delta N250 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A2_ee_delta N500 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A3_obj_rel_ee_delta N50 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A3_obj_rel_ee_delta N100 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A3_obj_rel_ee_delta N250 | +0.000 | [+0.000, +0.000] | 0 | 0 |
| Diffusion minus ACT: A3_obj_rel_ee_delta N500 | +0.000 | [+0.000, +0.000] | 0 | 0 |

## Dataset-size trends

All four dataset sizes saturated the observed success metric at 36/36 for every policy/action-space combination. Mean ticks to success were:

| Family / space | N50 | N100 | N250 | N500 | Interpretation |
|---|---:|---:|---:|---:|---|
| ACT / A2 | 102.19 | 99.50 | 98.69 | 99.28 | Improves through N250, then slightly reverses |
| ACT / A3 | 100.92 | 99.39 | 99.83 | 100.00 | Non-monotonic |
| Diffusion / A2 | 99.14 | 100.44 | 100.94 | 100.92 | No increasing-size speed benefit |
| Diffusion / A3 | 99.50 | 100.06 | 101.03 | 100.33 | Non-monotonic |

Thus the matrix does not support a monotonic N50-to-N500 improvement. Since success is saturated and final angle is measured at threshold termination (cell means 45.30–45.49 degrees), neither metric resolves a dataset-size winner.

## Continuous matched differences

The following examples are descriptive paired mean differences whose unadjusted 95% bootstrap intervals exclude zero; all 16 comparisons and all metrics are preserved in `aggregate_summary.json`.

- For A3 minus A2, ACT-N50 reached success 1.28 ticks sooner [−2.39, −0.22], while ACT-N250 took 1.14 ticks longer [+0.22, +2.17]. At ACT-N500, A3 reduced per-rollout maximum contact force by 10.74 N [−21.56, −0.37] and mean contact force by 2.30 N [−3.46, −1.16]. These changing directions do not support a general A3 advantage.
- For Diffusion minus ACT in A2, N50 was 3.06 ticks faster [−4.81, −1.33], but N250 and N500 were respectively 2.25 [+1.25, +3.33] and 1.64 [+0.72, +2.61] ticks slower. At N500, Diffusion had a 9.09 N lower maximum-force mean [−18.12, −0.11] but a 0.89 N higher mean contact force [+0.19, +1.62].
- For Diffusion minus ACT in A3, N250 was 1.19 ticks slower [+0.19, +2.28] and had 0.83 N higher mean contact force [+0.06, +1.62]; at N500 its mean contact force was 2.89 N higher [+1.25, +4.52]. Other intervals often include zero.

These intervals are not multiplicity-adjusted and should not be read as confirmatory hypothesis tests. Their main value is showing that secondary behavior is heterogeneous despite identical success outcomes.

## Contact, adapter safety, and failure modes

All 576 rollouts had contact evidence. Across 57,678 policy decisions, adapter-v1 accepted 54,183 (93.94%), corrected 3,495 (6.06%), and rejected zero. The artifacts contain 3,506 structured warnings, all in the `a2.joint_velocity_limit` family. Aside from the single force-watch exceedance described above, warning evidence remained inside the established v3 audit envelope; it is still evidence of bounded adapter intervention, not evidence that warnings are harmless on hardware.

All failure labels are `success`; there were no timeouts, environment truncations, or adapter rejection failures. The maximum observed force was 219.95 N. Cell maximum forces otherwise ranged from 121.50 N to 219.95 N, while cell-level mean contact forces ranged from 17.02 N to 20.21 N. Adapter evidence consists of decision totals plus structured warnings and force-peak windows; it is not a complete per-tick adapter trace, and the simulator lacks general collision/slip sensing.

## Diffusion diagnostics

Returned training evidence uses a 10-step DDIM validation metric, and the primary matrix freezes DDIM-10/Tp16/Ta8. No separate closed-loop sampler or horizon diagnostic sweep is available or authorized.

## Supported, inconclusive, and unsupported claims

Supported findings are limited to the complete matched cells and the frozen door-pose/orientation benchmark: all 16 cells succeeded on all 36 rollouts; success did not distinguish A3 from A2, Diffusion from ACT, or dataset sizes; secondary timing, force, and correction effects were heterogeneous; and one ACT-A3-N50 rollout requires force review. Exact continuous summaries and uncertainty are recorded in `aggregate_summary.json`.

Diffusion sampler/horizon sensitivity and robustness across training seeds are inconclusive. Training used seed 0 only; matched evaluation seeds do not prove training-seed robustness.

This evidence does not support VLA readiness, A4 learning, hardware readiness, general geometry/viewpoint/language generalization, sim-to-real transfer, RL, WAM-lite, or fake-door claims.

## Phase 4 planning recommendation

Use only complete, provenance-valid Phase 3 cells to choose candidate action representations or baseline families. Treat safety review items and diagnostic gaps as planning inputs, not readiness evidence. Any Phase 4 execution requires separate authorization.

## Artifact paths

- Artifact completeness: `/home/pacquadr/Desktop/DoorManipulation/outputs/curated/phase3_unified_evaluation/artifact_completeness.csv`
- Normalized rollouts: `/home/pacquadr/Desktop/DoorManipulation/outputs/curated/phase3_unified_evaluation/normalized_rollouts.csv`
- Aggregate summary: `/home/pacquadr/Desktop/DoorManipulation/outputs/curated/phase3_unified_evaluation/aggregate_summary.json`
- Exclusions: `/home/pacquadr/Desktop/DoorManipulation/outputs/curated/phase3_unified_evaluation/exclusions.json`
- Resolved plan: `/home/pacquadr/Desktop/DoorManipulation/outputs/curated/phase3_unified_evaluation/evaluation_plan.resolved.json`

## Remaining caveats

One simulated door family, state-only policies, CPU simulation, limited force sensing, no general collision/slip sensing, and no independent training-seed replication bound every conclusion.
