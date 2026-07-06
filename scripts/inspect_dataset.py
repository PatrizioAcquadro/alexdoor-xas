#!/usr/bin/env python
"""Inspect one exported dataset as a learned policy would see it (Phase 3.0).

Prints a seeded sample batch (shapes, dtypes, per-key stats, provenance) and
saves an action-channel trace + histogram figure. For A4 datasets it prints the
structured chunk log and its numeric encoding instead (A4 is symbolic per-phase
data, not a per-tick tensor stream). No Kit launch needed::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/inspect_dataset.py \\
        --dataset datasets/door_push_alex/A2_ee_delta/v0 --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.dataset import (
    A4ChunkDataset,
    BatchIterator,
    ChunkSampler,
    EpisodeDataset,
    episode_chunk_features,
    load_splits,
    open_dataset,
    splits_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="dataset version dir, e.g. datasets/door_push/A2_ee_delta/v0",
    )
    parser.add_argument(
        "--split", default=None, choices=("train", "val", "test"),
        help="restrict to a split (needs the task's split file); default: all episodes",
    )
    parser.add_argument("--obs_preset", default="core")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path, default=paths.OUTPUTS_DIR / "dataset_inspect",
        help="where the inspection figure is written",
    )
    return parser.parse_args()


def _stat_line(name: str, array: np.ndarray) -> str:
    return (
        f"  {name:<8} shape={tuple(array.shape)!s:<14} dtype={array.dtype!s:<8} "
        f"min={array.min():+.4f} mean={float(array.mean()):+.4f} max={array.max():+.4f}"
    )


def inspect_a4(dataset: A4ChunkDataset) -> int:
    print(f"A4 dataset: {len(dataset)} episodes, task {dataset.task}")
    record = dataset[0]
    print(f"episode {record.episode_id[:8]} (success={record.success}, "
          f"final angle {record.final_door_angle:.3f} rad):")
    for chunk in record.chunks:
        print(f"  {chunk.phase:<12} target_panel={chunk.contact_target_panel} "
              f"hinge_delta={chunk.motion_hinge_delta_rad:+.3f} rad "
              f"duration={chunk.duration_ticks} ticks")
    features = episode_chunk_features(record)
    print(_stat_line("features", features))
    print("NOTE: A4 is symbolic per-phase data (docs/dataset_interface.md); "
          "there is no per-tick batch view.")
    return 0


def main() -> int:
    args = parse_args()
    dataset = open_dataset(args.dataset)
    if isinstance(dataset, A4ChunkDataset):
        return inspect_a4(dataset)
    assert isinstance(dataset, EpisodeDataset)

    episode_ids = None
    if args.split is not None:
        # datasets/<task>/<space>/<version> -> the task's shared split file.
        version_dir = args.dataset.resolve()
        split_file = splits_path(
            version_dir.parents[2], version_dir.parents[1].name, version_dir.name
        )
        episode_ids = load_splits(split_file, episode_ids=dataset.episode_ids)[args.split]

    sampler = ChunkSampler(
        dataset, horizon=args.horizon, obs_preset=args.obs_preset, episode_ids=episode_ids
    )
    batch = next(iter(BatchIterator(sampler, batch_size=args.batch_size, seed=args.seed)))

    print(f"dataset : {args.dataset} ({dataset.action_space}, task {dataset.task})")
    print(f"episodes: {len(dataset)} total"
          + (f", {len(episode_ids)} in split {args.split!r}" if episode_ids else ""))
    print(f"samples : {len(sampler)} (horizon {args.horizon}, obs preset {args.obs_preset!r})")
    print("batch:")
    for key in ("obs", "actions", "is_pad", "t"):
        print(_stat_line(key, batch[key]))
    print(f"  action_space tag: {batch['action_space']}")
    print(f"  episodes in batch: {[e[:8] for e in batch['episode_ids']]}")

    figure = _save_figure(args, dataset, batch)
    print(f"figure  : {figure}")
    return 0


def _save_figure(args: argparse.Namespace, dataset: EpisodeDataset, batch) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    record = dataset[0]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7))
    for dim in range(record.action_dim):
        top.plot(record.t, record.actions[:, dim], lw=1.0, label=f"dim {dim}")
    top.set_xlabel("time (s)")
    top.set_ylabel(f"action ({dataset.action_space})")
    top.set_title(f"episode {record.episode_id[:8]}: action channels")
    top.legend(fontsize=7, ncol=3)

    bottom.hist(batch["actions"].reshape(-1, record.action_dim), bins=30, stacked=True)
    bottom.set_xlabel(f"batch action values ({dataset.action_space})")
    bottom.set_ylabel("count")
    bottom.set_title(f"sampled batch (B={batch['obs'].shape[0]}, H={args.horizon})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{dataset.task}_{dataset.action_space}_batch.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    sys.exit(main())
