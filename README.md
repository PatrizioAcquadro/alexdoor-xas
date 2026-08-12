# AlexDoor-XAS

AlexDoor-XAS studies how action representation affects learning and execution in contact-rich humanoid manipulation. 
Its first benchmark is simulated door pushing with the fixed-base IHMC Alex V2 torso in NVIDIA Isaac Sim and Isaac Lab.

The benchmark records matched episodes in four action representations (A1-A4), allowing the representation to change while the robot, task, and underlying experience remain fixed.

## Current scope

The maintained workflows are:

```text
scripted Alex V2 door baseline -> matched v2_pose A1-A4 exports
A2/A3 datasets -> ACT or Diffusion -> adapter-v1 -> closed-loop evaluation
```

- A1 is export-only.
- A2 and A3 support learned policies.
- A4 is recorded and adapter-executable but does not have a learned policy.
- Policies are state-only, the benchmark is simulation-only, and no command controls a physical Alex robot.

The completed evaluation was success-saturated and did not identify a winning policy, representation, or dataset size. 
See [Project Status](knowledge/wiki/status.md) for maintained capabilities, results, and current boundaries.

## Requirements

- Python 3.11 or newer through the supported Isaac Lab runtime.
- Isaac Sim 6.0.1 and Isaac Lab `release/3.0.0-beta2`.
- The external Alex extension, the machine-local Alex V2, door, and hallway assets.

Do not use bare system `python3` for Isaac code.

## Installation and validation

```bash
/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e \
  /home/pacquadr/Desktop/Alex/source/ihmc_alex_isaaclab
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e ".[dev]"
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p -m pytest -q
```

The complete simulator, dataset, adapter, and policy verification surface is listed in [Project Status](knowledge/wiki/status.md).

## Minimal workflow

Generate matched Alex V2 episodes and A1-A4 exports:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p \
  scripts/run_scripted_baseline.py --viz none --device cuda:0
```

Train ACT on A2 or Diffusion on A3:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_act.py --space A2_ee_delta
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/train_diffusion.py --space A3_obj_rel_ee_delta
```

Evaluate a completed self-contained checkpoint:

```bash
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_act.py --checkpoint outputs/door_push_alex_v2/act/<run_id>/checkpoints/best.pt --device cuda:0
PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/eval_diffusion.py --checkpoint outputs/door_push_alex_v2/diffusion/<run_id>/checkpoints/best.pt --device cuda:0
```

## Repository layout

```text
src/alexdoor_xas/   package code for the benchmark, data, policies, and evaluation
scripts/            supported generation, training, evaluation, and verification entry points
configs/            active calibration, scripted-baseline, and policy configuration
tests/              deterministic regression and contract tests
knowledge/          user-owned raw research and the official technical wiki
datasets/           reusable local episodes, splits, and normalization artifacts
outputs/            canonical D0-D4 scenes and learned-policy runs
```

Machine-local assets, datasets, checkpoints, videos, logs, and runtime caches remain outside Git.

## Documentation

- [Technical Wiki](knowledge/wiki/index.md)
- [Project Status](knowledge/wiki/status.md)
- [System Architecture](knowledge/wiki/topics/system-architecture.md)
- [Action Representations and Adapters](knowledge/wiki/topics/action-representations-and-adapters.md)
- [Episode and Dataset Contracts](knowledge/wiki/topics/episode-and-dataset-contracts.md)
- [Learned Policy Stack](knowledge/wiki/topics/learned-policy-stack.md)
- [Alex V2 Benchmark](knowledge/wiki/topics/alex-v2-benchmark.md)
- [Output Contract](outputs/README.md)

## License

This repository is proprietary. 
No license grant is provided unless a separate license file or written agreement states otherwise.
