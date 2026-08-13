#!/usr/bin/env python
"""Verify matched A1-A4 datasets, splits, normalization, and model-facing batches."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from alexdoor_xas import paths
from alexdoor_xas.action.frames import door_frame_from_body_pose
from alexdoor_xas.action.spaces import (
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
    ALL_ACTION_SPACES,
)
from alexdoor_xas.dataset.loader import A4ChunkDataset, EpisodeDataset
from alexdoor_xas.dataset.normalize import (
    compute_norm_stats,
    load_norm_stats,
    norm_stats_path,
    save_norm_stats,
    validate_norm_stats,
)
from alexdoor_xas.dataset.robot_asset import (
    load_dataset_robot_asset,
    validate_dataset_episode_robot_asset,
)
from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler
from alexdoor_xas.dataset.splits import (
    assert_no_cross_split_duplicates,
    load_splits,
    make_grouped_splits,
    save_splits,
    split_entries,
    splits_path,
)
from alexdoor_xas.dataset.validate import (
    validate_a4_dataset,
    validate_dataset,
    validate_matched_action_space_datasets,
)

YAW_IDENTITY_TOL_RAD = 1e-9
CONVERSION_TOL = 1e-9
MIN_DISTINCT_DELTA = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets_root", type=Path, default=paths.DATASETS_DIR)
    parser.add_argument("--task", default=paths.ALEX_V2_TASK)
    parser.add_argument("--version", default=paths.ALEX_V2_DATASET_VERSION)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="replace the task split and per-space normalization artifacts",
    )
    return parser.parse_args()


def verify_task(args: argparse.Namespace, task: str) -> list[str]:
    failures: list[str] = []
    print(f"== task {task} (version {args.version}) ==")

    datasets: dict[str, EpisodeDataset | A4ChunkDataset] = {}
    for space in ALL_ACTION_SPACES:
        dataset_dir = args.datasets_root / task / space / args.version
        if not dataset_dir.is_dir():
            failures.append(f"{task}: missing required {space} dataset")
            continue
        try:
            dataset = (
                A4ChunkDataset(dataset_dir)
                if space == A4_OBJ_CENTRIC_CHUNK
                else EpisodeDataset(dataset_dir)
            )
            if task == paths.ALEX_V2_TASK:
                ref, _ = load_dataset_robot_asset(dataset_dir, require=True)
                if ref is None:
                    raise ValueError("required Alex V2 robot asset is missing")
                validate_dataset_episode_robot_asset(dataset, ref)
        except (OSError, KeyError, TypeError, ValueError) as error:
            failures.append(f"{task}/{space}: failed to load: {error}")
        else:
            datasets[space] = dataset

    if failures:
        return failures

    hdf5_datasets = {
        space: dataset for space, dataset in datasets.items() if isinstance(dataset, EpisodeDataset)
    }
    a4_dataset = datasets[A4_OBJ_CENTRIC_CHUNK]
    if not isinstance(a4_dataset, A4ChunkDataset):
        return [f"{task}: A4 dataset used the wrong loader"]

    for space, dataset in hdf5_datasets.items():
        result = validate_dataset(dataset)
        _record_result(space, result, failures)
        print(
            f"  [{'ok ' if result.ok else 'ERR'}] {space}: "
            f"{len(dataset)} episodes, action_dim={dataset.action_dim}"
        )

    a4_result = validate_a4_dataset(a4_dataset)
    _record_result("A4", a4_result, failures)
    print(
        f"  [{'ok ' if a4_result.ok else 'ERR'}] {A4_OBJ_CENTRIC_CHUNK}: "
        f"{len(a4_dataset)} episodes, "
        f"{sum(len(record.chunks) for record in a4_dataset.records)} chunks"
    )
    if failures:
        return failures

    matched = validate_matched_action_space_datasets(hdf5_datasets, a4_dataset)
    _record_result("matched", matched, failures)
    failures.extend(
        _verify_a2_a3_distinct(
            task,
            hdf5_datasets[A2_EE_DELTA],
            hdf5_datasets[A3_OBJ_REL_EE_DELTA],
        )
    )
    if failures:
        return failures

    reference = next(iter(hdf5_datasets.values()))
    split_file = splits_path(args.datasets_root, task, args.version)
    try:
        if args.write_artifacts:
            splits, metadata = make_grouped_splits(split_entries(reference), seed=args.seed)
            save_splits(split_file, splits, seed=args.seed, metadata=metadata)
            split_mode = "wrote"
        else:
            splits = load_splits(split_file, episode_ids=reference.episode_ids)
            split_mode = "read"
        for dataset in hdf5_datasets.values():
            assert_no_cross_split_duplicates(split_entries(dataset), splits)
    except (OSError, KeyError, TypeError, ValueError) as error:
        failures.append(f"{task}: invalid splits {split_file}: {error}")
        return failures

    print(
        f"  [ok ] splits: train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} ({split_mode} {split_file})"
    )
    for space, dataset in hdf5_datasets.items():
        failures.extend(_verify_space(args, task, space, dataset, splits))
    return failures


def _record_result(label, result, failures: list[str]) -> None:
    for warning in result.warnings:
        print(f"  WARN [{label}] {warning}")
    failures.extend(f"{label}: {error}" for error in result.errors)


def _verify_a2_a3_distinct(
    task: str,
    a2: EpisodeDataset,
    a3: EpisodeDataset,
) -> list[str]:
    failures: list[str] = []
    if sorted(a2.episode_ids) != sorted(a3.episode_ids):
        return [f"{task}: A2 and A3 exports do not share episode ids"]

    per_pose: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"n": 0, "max_pair_diff": 0.0, "max_conversion_err": 0.0}
    )
    yawed_episodes = 0
    for episode_id in a2.episode_ids:
        rec2 = a2.by_id(episode_id)
        rec3 = a3.by_id(episode_id)
        frame = door_frame_from_body_pose(
            np.asarray(rec2.extras["door_frame_pos_w"], dtype=np.float64),
            np.asarray(rec2.extras["door_frame_quat_w_xyzw"], dtype=np.float64),
        )
        yaw = float(np.arctan2(frame.rot[1, 0], frame.rot[0, 0]))
        pose_id = str(rec2.extras.get("door_pose_id"))
        label = f"episode {episode_id[:8]} (pose {pose_id}, seed {rec2.meta['seed']})"
        pair_diff = float(np.abs(rec2.actions - rec3.actions).max())
        expected_a3 = np.concatenate(
            [rec2.actions[:, :3] @ frame.rot, rec2.actions[:, 3:] @ frame.rot],
            axis=1,
        )
        conversion_err = float(np.abs(rec3.actions - expected_a3).max())
        stats = per_pose[pose_id]
        stats["n"] += 1
        stats["max_pair_diff"] = max(stats["max_pair_diff"], pair_diff)
        stats["max_conversion_err"] = max(stats["max_conversion_err"], conversion_err)

        if conversion_err > CONVERSION_TOL:
            failures.append(f"{task}/{label}: A2/A3 conversion error {conversion_err:.3e}")
        if abs(yaw) <= YAW_IDENTITY_TOL_RAD:
            if pair_diff > CONVERSION_TOL:
                failures.append(f"{task}/{label}: yaw=0 exports differ by {pair_diff:.3e}")
        else:
            yawed_episodes += 1
            if pair_diff < MIN_DISTINCT_DELTA:
                failures.append(
                    f"{task}/{label}: yaw={yaw:+.4f} but A2/A3 max diff is {pair_diff:.3e}"
                )

    if task == paths.ALEX_V2_TASK and yawed_episodes == 0:
        failures.append(f"{task}: A2/A3 gate found no yawed door episodes")
    for pose_id, stats in sorted(per_pose.items()):
        print(
            f"  [{'ok ' if not failures else 'CHK'}] A2/A3 pose {pose_id}: "
            f"episodes={stats['n']} max|A2-A3|={stats['max_pair_diff']:.6f} "
            f"max_conversion_err={stats['max_conversion_err']:.2e}"
        )
    return failures


def _verify_space(
    args: argparse.Namespace,
    task: str,
    space: str,
    dataset: EpisodeDataset,
    splits: dict[str, list[str]],
) -> list[str]:
    label = f"{task}/{space}"
    stats_file = norm_stats_path(dataset.dataset_dir)
    try:
        if args.write_artifacts:
            stats = compute_norm_stats(dataset, splits["train"])
            save_norm_stats(stats_file, stats)
            stats_mode = "wrote"
        else:
            stats = load_norm_stats(stats_file)
            stats_mode = "read"
        errors = validate_norm_stats(stats, dataset, splits["train"])
        if errors:
            return [f"{label}: {error}" for error in errors]

        sampler = ChunkSampler(dataset, horizon=args.horizon, episode_ids=splits["train"])
        batch = next(iter(BatchIterator(sampler, batch_size=args.batch_size, seed=args.seed)))
    except (OSError, KeyError, TypeError, ValueError) as error:
        return [f"{label}: model-facing interface failed: {error}"]

    batch_size = min(args.batch_size, len(sampler))
    expected = {
        "obs": (batch_size, sampler.obs_dim),
        "actions": (batch_size, args.horizon, dataset.action_dim),
        "is_pad": (batch_size, args.horizon),
    }
    if set(batch) != set(expected):
        return [f"{label}: batch keys {sorted(batch)} != {sorted(expected)}"]
    failures = [
        f"{label}: batch {key} shape {batch[key].shape} != {shape}"
        for key, shape in expected.items()
        if batch[key].shape != shape
    ]
    if not np.isfinite(batch["obs"]).all() or not np.isfinite(batch["actions"]).all():
        failures.append(f"{label}: batch contains non-finite values")
    print(
        f"  [{'ok ' if not failures else 'ERR'}] {space}: "
        f"batch obs{batch['obs'].shape} actions{batch['actions'].shape} ({stats_mode} {stats_file})"
    )
    return failures


def main() -> int:
    args = parse_args()
    failures = verify_task(args, args.task)
    print("-- result --")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print("FAIL")
        return 1
    print(f"PASS ({args.task})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
