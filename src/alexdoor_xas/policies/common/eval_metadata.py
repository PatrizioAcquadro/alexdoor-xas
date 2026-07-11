"""Dataset/split provenance for closed-loop eval payloads (pure file reads).

The later unified report needs every eval JSON to be self-describing: which
dataset (task/space/version/obs preset) the checkpoint was trained on, the
dataset content fingerprint, and the train/val/test split membership. All of
it is recoverable from the checkpoint's embedded config, the run dir's
``train_log.json``, and the dataset version's ``manifest.json`` / shared
splits file — this helper gathers them and records *why* a field is missing
instead of silently omitting it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dataset_provenance(
    checkpoint_config: dict[str, Any] | None,
    run_dir: Path,
    datasets_root: Path,
) -> dict[str, Any]:
    """Dataset + split metadata block for an eval payload."""
    notes: list[str] = []
    dataset = dict((checkpoint_config or {}).get("dataset") or {})
    out: dict[str, Any] = {
        "dataset": dataset or None,
        "fingerprint_sha256": None,
        "manifest_path": None,
        "splits_path": None,
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
            out["fingerprint_sha256"] = manifest.get("source_fingerprint_sha256")
            out["manifest_path"] = str(manifest_path)
        else:
            notes.append(f"dataset version has no manifest.json ({manifest_path})")
        splits_path = datasets_root / task / "splits" / f"{version}.json"
        if splits_path.is_file():
            splits = json.loads(splits_path.read_text())
            out["splits_path"] = str(splits_path)
            split_ids = splits.get("splits") or {}
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


__all__ = ["dataset_provenance"]
