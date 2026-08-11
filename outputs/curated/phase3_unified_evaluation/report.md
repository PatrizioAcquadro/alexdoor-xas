# Unified Phase 3 Scientific Evaluation

## Protocol

Sixteen ACT/Diffusion, A2/A3, and N50/N100/N250/N500 cells were evaluated on
36 matched rollouts each: 20 at D0 and four each at D1-D4. ACT used horizon 40
without temporal ensembling. Diffusion used DDIM-10 with Tp=16 and Ta=8.
Success used a 45 degree first-crossing threshold and a 600-tick limit.

## Main findings

- All 16 cells completed and all 576 rollouts succeeded. Each cell's 36/36
  result has a 95% Wilson interval of [90.4%, 100.0%].
- Success rate therefore does not select A2 versus A3, ACT versus Diffusion,
  or N50/N100/N250/N500 in this saturated benchmark.
- Timing, force, and adapter-correction directions change across cells and do
  not support a uniform secondary-metric winner.
- All rollouts recorded contact and none was adapter-rejected.
- One ACT-A3-N50 D0 randomized rollout at seed 112 peaked at 219.95 N for one
  tick. That cell remains `REVIEW_REQUIRED`; the other 15 pass the bounded
  simulation audit.

## Cell results

| Run | Success | Mean final angle | Peak force | Safety |
|---|---:|---:|---:|---|
| ACT / A2 / N50 | 36/36 | 45.41 deg | 167.58 N | PASS |
| ACT / A3 / N50 | 36/36 | 45.48 deg | 219.95 N | REVIEW_REQUIRED |
| Diffusion / A2 / N50 | 36/36 | 45.40 deg | 173.06 N | PASS |
| Diffusion / A3 / N50 | 36/36 | 45.46 deg | 156.66 N | PASS |
| ACT / A2 / N100 | 36/36 | 45.36 deg | 129.02 N | PASS |
| ACT / A3 / N100 | 36/36 | 45.30 deg | 134.00 N | PASS |
| Diffusion / A2 / N100 | 36/36 | 45.41 deg | 152.80 N | PASS |
| Diffusion / A3 / N100 | 36/36 | 45.40 deg | 138.56 N | PASS |
| ACT / A2 / N250 | 36/36 | 45.48 deg | 132.41 N | PASS |
| ACT / A3 / N250 | 36/36 | 45.47 deg | 128.58 N | PASS |
| Diffusion / A2 / N250 | 36/36 | 45.49 deg | 145.10 N | PASS |
| Diffusion / A3 / N250 | 36/36 | 45.39 deg | 121.50 N | PASS |
| ACT / A2 / N500 | 36/36 | 45.41 deg | 134.12 N | PASS |
| ACT / A3 / N500 | 36/36 | 45.43 deg | 123.74 N | PASS |
| Diffusion / A2 / N500 | 36/36 | 45.42 deg | 131.36 N | PASS |
| Diffusion / A3 / N500 | 36/36 | 45.40 deg | 148.38 N | PASS |

## Boundaries

These results cover one simulated door family and seed-0-trained, state-only
policies. They do not establish training-seed robustness, broader
generalization, hardware safety, VLA readiness, or sim-to-real performance.
Exact cell summaries and paired comparisons remain in `aggregate_summary.json`.
