"""Dataset/split provenance for closed-loop eval payloads (pure file reads).

The later unified report needs every eval JSON to be self-describing: which
dataset (task/space/version/obs preset) the checkpoint was trained on, the
dataset content fingerprints, and the train/val/test split membership. Two
fingerprints exist and are never conflated:

- ``source_fingerprint_sha256`` — the *shared source* fingerprint from the
  version's ``manifest.json`` (sha256 over the source episode HDF5 files;
  identical for A2/A3 exports of one generation pass);
- ``checkpoint_dataset_fingerprint_sha256`` / ``live_dataset_fingerprint_sha256``
  — the *exact per-action-space* content fingerprint
  (:func:`alexdoor_xas.dataset.dataset_fingerprint`), embedded in the
  checkpoint's norm stats at train time and recomputed from the on-disk
  dataset at eval time. A2 and A3 legitimately differ here while sharing the
  same source fingerprint.

:func:`verify_checkpoint_dataset_binding` fails evaluation loudly when the
checkpoint and the live dataset disagree (content fingerprint, episode ids,
or split membership), so an eval JSON can never silently describe a
checkpoint evaluated against data it was not trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from alexdoor_xas.dataset import (
    EpisodeDataset,
    dataset_fingerprint,
    load_split_payload,
    split_fingerprint,
)


class EvalProvenanceError(RuntimeError):
    """Checkpoint/live dataset or split mismatch detected at evaluation time."""


def file_sha256(path: str | Path) -> str:
    """sha256 of one file (stable checkpoint identity for eval payloads)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_provenance(
    checkpoint_config: dict[str, Any] | None,
    run_dir: Path,
    datasets_root: Path,
) -> dict[str, Any]:
    """Dataset + split metadata block for an eval payload (gather-only).

    ``fingerprint_sha256`` is retained for existing consumers and is the same
    value as ``source_fingerprint_sha256`` (the shared source-manifest
    fingerprint) — it has never been the exact per-space dataset fingerprint;
    the exact fingerprints come from :func:`verify_checkpoint_dataset_binding`.
    """
    notes: list[str] = []
    dataset = dict((checkpoint_config or {}).get("dataset") or {})
    out: dict[str, Any] = {
        "dataset": dataset or None,
        "fingerprint_sha256": None,
        "source_fingerprint_sha256": None,
        "manifest_path": None,
        "splits_path": None,
        "split_fingerprint_sha256": None,
        "split_summary": None,
        "split_episode_ids": None,
        "norm_stats_source": None,
        "notes": notes,
    }
    if not dataset:
        notes.append("checkpoint carries no embedded dataset config")

    task = dataset.get("task")
    space = dataset.get("space")
    version = dataset.get("version")
    if task and space and version:
        manifest_path = datasets_root / task / space / version / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            source = manifest.get("source_fingerprint_sha256")
            out["fingerprint_sha256"] = source
            out["source_fingerprint_sha256"] = source
            out["manifest_path"] = str(manifest_path)
        else:
            notes.append(f"dataset version has no manifest.json ({manifest_path})")
        splits_path = datasets_root / task / "splits" / f"{version}.json"
        if splits_path.is_file():
            splits = load_split_payload(splits_path)
            out["splits_path"] = str(splits_path)
            split_ids = splits.get("splits") or {}
            out["split_fingerprint_sha256"] = splits.get(
                "split_fingerprint_sha256"
            ) or split_fingerprint({name: list(ids) for name, ids in split_ids.items()})
            out["split_summary"] = {
                "n_episodes": splits.get("n_episodes"),
                "fractions": splits.get("fractions"),
                "seed": splits.get("seed"),
                "sizes": {name: len(ids) for name, ids in split_ids.items()},
            }
            out["split_episode_ids"] = split_ids
        else:
            notes.append(f"no shared splits file ({splits_path})")

    train_log_path = run_dir / "logs" / "train_log.json"
    if train_log_path.is_file():
        train_log = json.loads(train_log_path.read_text())
        out["norm_stats_source"] = train_log.get("stats_source")
        out["train_log_split_ids"] = {
            "train": train_log.get("train_episode_ids"),
            "val": train_log.get("val_episode_ids"),
        }
    else:
        notes.append(f"run dir has no train_log.json ({train_log_path})")
    return out


def verify_checkpoint_dataset_binding(
    checkpoint_stats,
    provenance: dict[str, Any],
    datasets_root: Path,
) -> dict[str, Any]:
    """Bind an eval to the exact dataset the checkpoint was trained on.

    ``checkpoint_stats`` is the checkpoint-embedded ``DatasetNormStats``
    (exact dataset fingerprint + episode ids + train split). The live dataset
    named by the checkpoint's config is loaded from disk, its exact content
    fingerprint recomputed, and the shared split file cross-checked. Any
    mismatch raises :class:`EvalProvenanceError` — evaluation must not
    proceed. On success the returned block is merged into the eval payload's
    ``dataset_provenance`` so the JSON stays auditable after the dataset
    directory changes.
    """
    dataset_cfg = provenance.get("dataset") or {}
    task, space, version = (
        dataset_cfg.get("task"),
        dataset_cfg.get("space"),
        dataset_cfg.get("version"),
    )
    if not (task and space and version):
        raise EvalProvenanceError(
            "checkpoint has no embedded dataset task/space/version — cannot bind "
            "the evaluation to its training dataset"
        )
    dataset_dir = Path(datasets_root) / task / space / version
    try:
        dataset = EpisodeDataset(dataset_dir)
    except (FileNotFoundError, ValueError) as error:
        raise EvalProvenanceError(
            f"live dataset {dataset_dir} is unavailable: {error}"
        ) from error

    checkpoint_fp = str(checkpoint_stats.dataset_fingerprint)
    live_fp = dataset_fingerprint(dataset)
    if not checkpoint_fp:
        raise EvalProvenanceError("checkpoint norm stats carry no dataset fingerprint")
    if checkpoint_fp != live_fp:
        raise EvalProvenanceError(
            f"checkpoint dataset fingerprint {checkpoint_fp[:16]}… does not match the "
            f"live {space} dataset {live_fp[:16]}… — the dataset was re-exported or "
            "modified after training; retrain or restore the exact dataset"
        )
    if tuple(checkpoint_stats.dataset_episode_ids) and tuple(
        checkpoint_stats.dataset_episode_ids
    ) != tuple(dataset.episode_ids):
        raise EvalProvenanceError(
            "checkpoint dataset episode ids do not match the live dataset"
        )

    split_ids = provenance.get("split_episode_ids")
    if not split_ids:
        raise EvalProvenanceError(
            f"no shared splits file for {task}/{version} — cannot validate the "
            "checkpoint's split membership"
        )
    live_splits = {
        name: list(split_ids.get(name) or ()) for name in ("train", "val", "test")
    }
    all_split_ids = [episode_id for ids in live_splits.values() for episode_id in ids]
    if len(all_split_ids) != len(set(all_split_ids)):
        raise EvalProvenanceError("live split contract has overlapping memberships")
    if set(all_split_ids) != set(dataset.episode_ids):
        raise EvalProvenanceError(
            "live split contract is not exhaustive over the exact dataset episodes"
        )
    computed_split_fp = split_fingerprint(live_splits)
    if provenance.get("split_fingerprint_sha256") != computed_split_fp:
        raise EvalProvenanceError(
            "live split fingerprint does not match its train/validation/test membership"
        )

    live_train = tuple(live_splits["train"])
    checkpoint_train = tuple(checkpoint_stats.train_episode_ids)
    if sorted(checkpoint_train) != sorted(live_train):
        raise EvalProvenanceError(
            "checkpoint train split does not match the live split contract "
            f"({len(checkpoint_train)} checkpoint train ids vs {len(live_train)} live)"
        )
    companion_ids = provenance.get("train_log_split_ids") or {}
    companion_train = companion_ids.get("train")
    companion_val = companion_ids.get("val")
    if companion_train is None or companion_val is None:
        raise EvalProvenanceError(
            "run train_log is missing train/validation split ids — cannot validate "
            "the checkpoint's complete training split provenance"
        )
    if sorted(companion_train) != sorted(live_train):
        raise EvalProvenanceError(
            "run train_log train split does not match the live split contract"
        )
    live_val = tuple(live_splits["val"])
    if sorted(companion_val) != sorted(live_val):
        raise EvalProvenanceError(
            "run train_log val split does not match the live split contract"
        )

    return {
        "checkpoint_dataset_fingerprint_sha256": checkpoint_fp,
        "live_dataset_fingerprint_sha256": live_fp,
        "dataset_fingerprint_match": True,
        "train_split_match": True,
        "val_split_checked": True,
        "n_live_episodes": len(dataset),
    }


__all__ = [
    "EvalProvenanceError",
    "dataset_provenance",
    "file_sha256",
    "verify_checkpoint_dataset_binding",
]
