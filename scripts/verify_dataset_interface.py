#!/usr/bin/env python
"""Phase 3.0 gate: the dataset/model interface consumes Phase 2 episodes.

For the active Alex V2 dataset this verifies, without policy code: exact A1-A4
availability and schemas; matched episode provenance/content; exact A2/A3
door-frame conversion and non-identity for yawed poses; deterministic grouped
splits; normalization round-trips; reproducible chunk batches; and end-to-end
consumption by a dummy linear model.

By default, split/stat artifacts are written to a temporary directory. Pass
``--write-artifacts`` to refresh official files under ``datasets/``. No Kit
launch needed::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/verify_dataset_interface.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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
from alexdoor_xas.dataset import (
    A4_FEATURE_DIM,
    A4ChunkDataset,
    BatchIterator,
    ChunkSampler,
    EpisodeDataset,
    assert_no_cross_split_duplicates,
    collate_torch,
    compute_norm_stats,
    episode_chunk_features,
    load_norm_stats,
    load_splits,
    make_grouped_splits,
    norm_stats_path,
    save_norm_stats,
    save_splits,
    split_entries,
    splits_path,
    validate_a4_dataset,
    validate_dataset,
    validate_matched_action_space_datasets,
    validate_norm_stats,
)
from alexdoor_xas.dataset.robot_asset import (
    load_dataset_robot_asset,
    validate_dataset_episode_robot_asset,
)

YAW_IDENTITY_TOL_RAD = 1e-9
CONVERSION_TOL = 1e-9
MIN_DISTINCT_DELTA = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=paths.DATASETS_DIR,
        help="datasets root (default: repo datasets/)",
    )
    parser.add_argument(
        "--task",
        default=paths.ALEX_V2_TASK,
        help=f"task to verify (default: {paths.ALEX_V2_TASK})",
    )
    parser.add_argument(
        "--version",
        default=paths.ALEX_V2_DATASET_VERSION,
        help=f"dataset version (default: {paths.ALEX_V2_DATASET_VERSION})",
    )
    parser.add_argument("--horizon", type=int, default=20, help="action chunk horizon")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="split + batch seed")
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="write splits/<version>.json and norm_stats.json into datasets/",
    )
    return parser.parse_args()


def discover_tasks(datasets_root: Path, version: str) -> list[str]:
    tasks = []
    for task_dir in sorted(datasets_root.iterdir()) if datasets_root.is_dir() else []:
        if any((task_dir / space / version).is_dir() for space in ALL_ACTION_SPACES):
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
            failures.append(f"{task}: missing required {space} dataset")
            continue
        try:
            if space == A4_OBJ_CENTRIC_CHUNK:
                dataset = A4ChunkDataset(dataset_dir)
                if task == paths.ALEX_V2_TASK:
                    ref, _ = load_dataset_robot_asset(dataset_dir, require=True)
                    if ref is None:
                        raise ValueError("required Alex V2 robot asset is missing")
                    validate_dataset_episode_robot_asset(dataset, ref)
                a4_dataset = dataset
            else:
                dataset = EpisodeDataset(dataset_dir)
                if task == paths.ALEX_V2_TASK:
                    ref, _ = load_dataset_robot_asset(dataset_dir, require=True)
                    if ref is None:
                        raise ValueError("required Alex V2 robot asset is missing")
                    validate_dataset_episode_robot_asset(dataset, ref)
                hdf5_datasets[space] = dataset
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
        action_dim = _safe_action_dim(dataset)
        print(
            f"  [{'ok ' if result.ok else 'ERR'}] {space}: "
            f"{len(dataset)} episodes validated, action_dim={action_dim}"
        )
    if a4_dataset is not None:
        result = validate_a4_dataset(a4_dataset)
        for warning in result.warnings:
            print(f"  WARN [A4] {warning}")
        for error in result.errors:
            failures.append(f"{task}/A4: {error}")
        try:
            features = [episode_chunk_features(record) for record in a4_dataset.records]
        except ValueError as exc:
            failures.append(f"{task}/A4: chunk-feature encoding failed: {exc}")
            features = []
        bad = [f.shape for f in features if f.ndim != 2 or f.shape[1] != A4_FEATURE_DIM]
        if bad:
            failures.append(f"{task}/A4: bad chunk-feature shapes {bad}")
        n_chunks = sum(f.shape[0] for f in features)
        print(
            f"  [{'ok ' if result.ok else 'ERR'}] {A4_OBJ_CENTRIC_CHUNK}: "
            f"{len(a4_dataset)} episodes, "
            f"{n_chunks} chunks x {A4_FEATURE_DIM} features"
        )

    # -- matched-condition checks across action spaces ----------------------
    reference_space = next(iter(hdf5_datasets))
    matched = validate_matched_action_space_datasets(hdf5_datasets, a4_dataset)
    for warning in matched.warnings:
        print(f"  WARN [matched] {warning}")
    for error in matched.errors:
        failures.append(f"{task}: {error}")
    if A2_EE_DELTA in hdf5_datasets and A3_OBJ_REL_EE_DELTA in hdf5_datasets:
        failures.extend(
            _verify_a2_a3_distinct(
                task,
                hdf5_datasets[A2_EE_DELTA],
                hdf5_datasets[A3_OBJ_REL_EE_DELTA],
            )
        )

    # -- splits: shared per task, grouped + pose-stratified, deterministic --
    ids = hdf5_datasets[reference_space].episode_ids
    entries = split_entries(hdf5_datasets[reference_space])
    splits, split_meta = make_grouped_splits(entries, seed=args.seed)
    if make_grouped_splits(entries, seed=args.seed)[0] != splits:
        failures.append(f"{task}: make_grouped_splits is not deterministic")
    # Leakage invariant, checked against every action space's own content keys
    # (relabelings share episode ids, so one shared split covers all spaces).
    for space, dataset in hdf5_datasets.items():
        try:
            assert_no_cross_split_duplicates(split_entries(dataset), splits)
        except ValueError as exc:
            failures.append(f"{task}/{space}: split leakage: {exc}")
    path = splits_path(_artifact_root(args), task, args.version)
    save_splits(path, splits, seed=args.seed, metadata=split_meta)
    try:
        reloaded = load_splits(path, episode_ids=ids)
    except ValueError as exc:
        failures.append(f"{task}: split reload failed: {exc}")
        reloaded = splits
    if reloaded != splits:
        failures.append(f"{task}: reloaded splits differ")
    artifact_mode = "datasets/" if args.write_artifacts else "temp"
    pose_note = ", ".join(
        f"{pose}:{info['episodes_per_split']['train']}/"
        f"{info['episodes_per_split']['val']}/{info['episodes_per_split']['test']}"
        for pose, info in split_meta["per_pose"].items()
    )
    print(
        f"  [ok ] splits: train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} ({split_meta['n_groups']} content groups; "
        f"per-pose train/val/test {pose_note}) "
        f"-> {artifact_mode}:{path.relative_to(_artifact_root(args))}"
    )

    # -- per space: norm stats + batches + model consumption ----------------
    for space, dataset in hdf5_datasets.items():
        try:
            failures.extend(_verify_space_consumption(args, task, space, dataset, splits))
        except Exception as exc:  # noqa: BLE001 - gate reports, never crashes
            failures.append(f"{task}/{space}: consumption check crashed: {exc}")
    return failures


def _verify_a2_a3_distinct(
    task: str,
    a2: EpisodeDataset,
    a3: EpisodeDataset,
) -> list[str]:
    """Check the exact posed-door conversion ``A3 = R_door^T A2``."""
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
                    f"{task}/{label}: yaw={yaw:+.4f} but A2/A3 max diff is only {pair_diff:.3e}"
                )

    if task == paths.ALEX_V2_TASK and yawed_episodes == 0:
        failures.append(f"{task}: A2/A3 gate found no yawed door episodes")
    for pose_id in sorted(per_pose):
        stats = per_pose[pose_id]
        print(
            f"  [{'ok ' if not failures else 'CHK'}] A2/A3 pose {pose_id}: "
            f"episodes={stats['n']} max|A2-A3|={stats['max_pair_diff']:.6f} "
            f"max_conversion_err={stats['max_conversion_err']:.2e}"
        )
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
    stats_path = save_norm_stats(norm_stats_path(_artifact_dataset_dir(args, dataset)), stats)
    loaded = load_norm_stats(stats_path)
    for error in validate_norm_stats(loaded, dataset, splits["train"]):
        failures.append(f"{label}: {error}")
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
            out = linear(tensors["obs"]).reshape(batch_size, args.horizon, dataset.action_dim)
        if out.shape != tensors["actions"].shape:
            failures.append(f"{label}: torch forward shape mismatch")
        torch_note = "torch ok"
    except ImportError:
        pass

    print(
        f"  [ok ] {space}: {len(sampler)} samples, batch obs{first['obs'].shape} "
        f"actions{first['actions'].shape}, dummy loss {loss:.3f}, {torch_note}"
    )
    return failures


def _artifact_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "artifacts_root", args.datasets_root))


def _artifact_dataset_dir(args: argparse.Namespace, dataset: EpisodeDataset) -> Path:
    if args.write_artifacts:
        return dataset.dataset_dir
    try:
        relative = dataset.dataset_dir.resolve().relative_to(args.datasets_root.resolve())
    except ValueError:
        relative = Path(dataset.task) / dataset.action_space / args.version
    return _artifact_root(args) / relative


def _safe_action_dim(dataset: EpisodeDataset) -> int | str:
    actions = np.asarray(dataset[0].actions)
    return int(actions.shape[1]) if actions.ndim == 2 else "invalid"


def main() -> int:
    args = parse_args()
    tasks = [args.task] if args.task else discover_tasks(args.datasets_root, args.version)
    if not tasks:
        print(f"FAIL: no datasets under {args.datasets_root} (version {args.version})")
        return 1

    failures: list[str] = []
    if args.write_artifacts:
        args.artifacts_root = args.datasets_root
        for task in tasks:
            failures.extend(verify_task(args, task))
    else:
        with tempfile.TemporaryDirectory(prefix="alexdoor_dataset_gate_") as tmp:
            args.artifacts_root = Path(tmp) / "datasets"
            print(
                "[artifacts] read-only mode: writing temporary gate files under "
                f"{args.artifacts_root}"
            )
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
