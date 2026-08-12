# outputs/

`outputs/` contains only this guide, the five canonical door scene layers, ACT/Diffusion learned-policy runs, and optional W&B state:

```text
outputs/
├── README.md
├── door_scene/
│   ├── D0.usda
│   ├── D1.usda
│   ├── D2.usda
│   ├── D3.usda
│   └── D4.usda
├── door_push_alex_v2/
│   ├── act/<run_id>/
│   └── diffusion/<run_id>/
└── wandb/                     # only when WANDB_MODE enables tracking
```

The default scene is `door_scene/D0.usda`. D0 is yaw `0.00` with XY `(0.00, 0.00)`; D1 is yaw `+0.05` with `(+0.02, 0.00)`; D2 is yaw `-0.05` with `(0.00, -0.02)`; D3 is yaw `+0.10` with `(+0.02, +0.02)`; D4 is yaw `-0.10` with `(+0.02, -0.02)`. `door_scene/` must contain exactly these files. Noncanonical scenes require an explicit path under `~/.cache/alexdoor-xas/door_scenes/`.

Training run IDs use collision-safe UTC timestamps: `<YYYYMMDDTHHMMSSZ>_<a2|a3>_<dataset-token>_seed<N>`, for example `20260812T153045Z_a3_v3n500_seed0`. Exclusive directory creation prevents overwrites; same-second collisions append `_r2`, `_r3`, and so on. A completed run is never reused. Resume only an incomplete run with `--resume <run-directory>`.

```text
<run_id>/
├── resolved_config.json
├── report.md
├── checkpoints/
│   ├── best.pt
│   └── last.pt              # incomplete/interrupted run only
├── training/
│   ├── history.json
│   └── summary.png
├── open_loop/
│   ├── metrics.json
│   └── summary.png
├── closed_loop/             # after evaluation
│   ├── metrics.json
│   └── summary.png
├── media/                   # optional, explicitly selected
├── traces/                  # optional, selectively retained
└── error.log                # optional, interruption/error only
```

`resolved_config.json` is immutable and freezes the run identity, policy configuration, and complete evaluation protocol. `report.md` is the only narrative summary. `best.pt` is the best validated self-contained inference checkpoint. `last.pt` is an atomic resumable state written before epoch 0 and after each completed epoch; it contains the model, optimizer, scheduler or explicit null, next epoch, global step, full history, Python/NumPy/Torch CPU/CUDA RNG states, and Diffusion EMA when used. Successful completion removes `last.pt` only after training, open-loop evaluation, plots, and report succeed. An interruption retains `last.pt`, preserves any `best.pt`, and creates `error.log`; a successful resume may retain the historical error log as an anomaly.

`training/history.json` contains policy-specific epoch losses, validation values, batch counts, best epoch/value, and durations. `training/summary.png` separates metrics with different meanings. `open_loop/metrics.json` contains translation L1 mean, dx/dy/dz L1, per-episode L1, and evaluated step counts. Its single plot shows per-dimension L1 and recorded-versus-predicted actions for the deterministic worst-L1 episode with episode-ID tie-breaking.

`closed_loop/metrics.json` records each rollout's pose, seed, fixed/randomized status, success, factual termination data, steps, time to success, force statistics and impulse, force-limit exceedances, adapter counts/rates, and warning-family counts. Aggregates cover overall, pose, fixed/randomized subset, and pose-plus-subset. Its single plot shows time to success, peak force with the frozen limit, and adapter correction/rejection rates. It does not create failure labels, failure interpretations, force windows, final-angle headlines, or per-rollout plots.

`closed_loop/` is written into a training run only when the requested protocol exactly matches the protocol frozen in its `resolved_config.json` and no completed closed-loop result exists. Any change to poses, seeds, randomization, thresholds, force limits, horizon, control settings, or policy execution creates a new timestamped sibling evaluation run. An evaluation-only run contains `resolved_config.json`, `report.md`, and `closed_loop/{metrics.json,summary.png}`; it records `run_type: evaluation`, `source_run_id`, and `source_checkpoint`, does not copy the checkpoint, and does not modify the source run.

`traces/` exists only for unsuccessful rollouts, force-limit exceedances, or explicitly selected rollout keys. `media/` exists only when explicitly requested or selected as useful. Empty optional directories are not created. W&B is an optional direct SDK integration: unset or `WANDB_MODE=disabled` creates nothing, while enabled modes write standard SDK state only under `outputs/wandb/`. The scripts log scalar aggregates only and never automatically publish artifacts or media. Reusable datasets stay in `datasets/`; calibration probes, verifier artifacts, scripted-run staging, arbitrary scenes, and inspection figures stay under `~/.cache/alexdoor-xas/`. Do not add generic `metrics/`, `plots/`, `logs/`, `evaluations/`, or `curated/` directories under `outputs/`.
