#!/usr/bin/env python
"""Phase 3.0 gate: the dataset/model interface consumes Phase 2 episodes.

For every exported dataset (task x action space) this verifies, without any
policy code: episodes load and validate against the frozen schema; episode ids
agree across action spaces; train/val/test splits generate deterministically
and save/reload; action + observation normalization stats compute and
round-trip; seeded chunk batches sample reproducibly; and a dummy linear
"policy" (numpy, plus a torch Linear when torch is importable) consumes a
batch end-to-end — proving an ACT/Diffusion-style trainer can train on the
data shapes. No Kit launch needed::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import (
    A4_OBJ_CENTRIC_CHUNK,
    ALL_ACTION_SPACES,
)
from alexdoor_xas.dataset import (
    A4_FEATURE_DIM,
    A4ChunkDataset,
    BatchIterator,
    ChunkSampler,
    EpisodeDataset,
    collate_torch,
    compute_norm_stats,
    episode_chunk_features,
    load_norm_stats,
    load_splits,
    make_splits,
    norm_stats_path,
    save_norm_stats,
    save_splits,
    splits_path,
    validate_a4_dataset,
    validate_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets_root", type=Path, default=paths.DATASETS_DIR,
        help="datasets root (default: repo datasets/)",
    )
    parser.add_argument("--version", default="v0", help="dataset version to verify")
    parser.add_argument("--horizon", type=int, default=20, help="action chunk horizon")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="split + batch seed")
    return parser.parse_args()


def discover_tasks(datasets_root: Path, version: str) -> list[str]:
    tasks = []
    for task_dir in sorted(datasets_root.iterdir()) if datasets_root.is_dir() else []:
        if any(
            (task_dir / space / version).is_dir() for space in ALL_ACTION_SPACES
        ):
            tasks.append(task_dir.name)
    return tasks


def verify_task(args: argparse.Namespace, task: str) -> list[str]:
    """Verify one task's datasets; returns a list of failure strings."""
    failures: list[str] = []
    root: Path = args.datasets_root
    print(f"== task {task} (version {args.version}) ==")

    hdf5_datasets: dict[str, EpisodeDataset] = {}
    a4_dataset: A4ChunkDataset | None = None
    for space in ALL_ACTION_SPACES:
        dataset_dir = root / task / space / args.version
        if not dataset_dir.is_dir():
            continue
        try:
            if space == A4_OBJ_CENTRIC_CHUNK:
                a4_dataset = A4ChunkDataset(dataset_dir)
            else:
                hdf5_datasets[space] = EpisodeDataset(dataset_dir)
        except Exception as exc:  # noqa: BLE001 - gate reports, never crashes
            failures.append(f"{task}/{space}: failed to load: {exc}")
    if not hdf5_datasets:
        failures.append(f"{task}: no HDF5 action-space datasets found")
        return failures

    # -- validation, per action space --------------------------------------
    for space, dataset in hdf5_datasets.items():
        result = validate_dataset(dataset)
        for warning in result.warnings:
            print(f"  WARN [{space}] {warning}")
        for error in result.errors:
            failures.append(f"{task}/{space}: {error}")
        print(f"  [{'ok ' if result.ok else 'ERR'}] {space}: "
              f"{len(dataset)} episodes validated, action_dim={dataset.action_dim}")
    if a4_dataset is not None:
        result = validate_a4_dataset(a4_dataset)
        for warning in result.warnings:
            print(f"  WARN [A4] {warning}")
        for error in result.errors:
            failures.append(f"{task}/A4: {error}")
        features = [episode_chunk_features(record) for record in a4_dataset.records]
        bad = [f.shape for f in features if f.ndim != 2 or f.shape[1] != A4_FEATURE_DIM]
        if bad:
            failures.append(f"{task}/A4: bad chunk-feature shapes {bad}")
        print(f"  [{'ok ' if result.ok else 'ERR'}] {A4_OBJ_CENTRIC_CHUNK}: "
              f"{len(a4_dataset)} episodes, "
              f"{sum(f.shape[0] for f in features)} chunks x {A4_FEATURE_DIM} features")

    # -- episode ids must agree across action spaces ------------------------
    reference_space = next(iter(hdf5_datasets))
    reference_ids = set(hdf5_datasets[reference_space].episode_ids)
    for space, dataset in hdf5_datasets.items():
        if set(dataset.episode_ids) != reference_ids:
            failures.append(f"{task}: episode ids of {space} differ from {reference_space}")
    if a4_dataset is not None and set(a4_dataset.episode_ids) != reference_ids:
        failures.append(f"{task}: episode ids of A4 differ from {reference_space}")

    # -- splits: shared per task, deterministic, reload-verified ------------
    ids = hdf5_datasets[reference_space].episode_ids
    splits = make_splits(ids, seed=args.seed)
    if make_splits(ids, seed=args.seed) != splits:
        failures.append(f"{task}: make_splits is not deterministic")
    path = splits_path(root, task, args.version)
    save_splits(path, splits, seed=args.seed)
    try:
        reloaded = load_splits(path, episode_ids=ids)
    except ValueError as exc:
        failures.append(f"{task}: split reload failed: {exc}")
        reloaded = splits
    if reloaded != splits:
        failures.append(f"{task}: reloaded splits differ")
    print(f"  [ok ] splits: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} -> {path.relative_to(root)}")

    # -- per space: norm stats + batches + model consumption ----------------
    for space, dataset in hdf5_datasets.items():
        try:
            failures.extend(_verify_space_consumption(args, task, space, dataset, splits))
        except Exception as exc:  # noqa: BLE001 - gate reports, never crashes
            failures.append(f"{task}/{space}: consumption check crashed: {exc}")
    return failures


def _verify_space_consumption(
    args: argparse.Namespace,
    task: str,
    space: str,
    dataset: EpisodeDataset,
    splits: dict[str, list[str]],
) -> list[str]:
    failures: list[str] = []
    label = f"{task}/{space}"

    stats = compute_norm_stats(dataset, splits["train"])
    stats_path = save_norm_stats(norm_stats_path(dataset.dataset_dir), stats)
    loaded = load_norm_stats(stats_path)
    sample_actions = dataset[0].actions
    roundtrip = loaded.action.denormalize(loaded.action.normalize(sample_actions))
    if not np.allclose(roundtrip, sample_actions, atol=1e-9):
        failures.append(f"{label}: normalization round-trip failed")

    sampler = ChunkSampler(dataset, horizon=args.horizon, episode_ids=splits["train"])
    iterator = BatchIterator(sampler, batch_size=args.batch_size, seed=args.seed)
    first = next(iter(iterator))
    again = next(iter(BatchIterator(sampler, batch_size=args.batch_size, seed=args.seed)))
    if not (
        np.array_equal(first["obs"], again["obs"])
        and np.array_equal(first["actions"], again["actions"])
    ):
        failures.append(f"{label}: seeded batches are not reproducible")

    batch_size = first["obs"].shape[0]
    expected = {
        "obs": (batch_size, sampler.obs_dim),
        "actions": (batch_size, args.horizon, dataset.action_dim),
        "is_pad": (batch_size, args.horizon),
    }
    for key, shape in expected.items():
        if first[key].shape != shape:
            failures.append(f"{label}: batch {key} shape {first[key].shape} != {shape}")

    # Dummy numpy "policy": obs -> (H, D) chunk; trains on normalized actions.
    rng = np.random.default_rng(args.seed)
    weights = rng.standard_normal((sampler.obs_dim, args.horizon * dataset.action_dim))
    predicted = (loaded.obs.normalize(first["obs"]) @ weights).reshape(
        batch_size, args.horizon, dataset.action_dim
    )
    error = predicted - loaded.action.normalize(first["actions"])
    loss = float((error**2 * ~first["is_pad"][..., None]).mean())
    if predicted.shape != first["actions"].shape or not np.isfinite(loss):
        failures.append(f"{label}: dummy policy could not consume the batch")

    torch_note = "torch skipped"
    try:
        import torch

        tensors = collate_torch(first)
        linear = torch.nn.Linear(sampler.obs_dim, args.horizon * dataset.action_dim)
        with torch.no_grad():
            out = linear(tensors["obs"]).reshape(
                batch_size, args.horizon, dataset.action_dim
            )
        if out.shape != tensors["actions"].shape:
            failures.append(f"{label}: torch forward shape mismatch")
        torch_note = "torch ok"
    except ImportError:
        pass

    print(f"  [ok ] {space}: {len(sampler)} samples, batch obs{first['obs'].shape} "
          f"actions{first['actions'].shape}, dummy loss {loss:.3f}, {torch_note}")
    return failures


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args.datasets_root, args.version)
    if not tasks:
        print(f"FAIL: no datasets under {args.datasets_root} (version {args.version})")
        return 1

    failures: list[str] = []
    for task in tasks:
        failures.extend(verify_task(args, task))

    print("-- result --")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print("FAIL")
        return 1
    print(f"PASS ({len(tasks)} tasks: {', '.join(tasks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
