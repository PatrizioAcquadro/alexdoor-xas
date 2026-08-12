"""Robot-asset provenance shared by dataset export and policy loading."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from alexdoor_xas import paths
from alexdoor_xas.assets.alex_v2_contract import (
    AlexV2ContractError,
    RobotAssetRef,
    validate_alex_v2_manifest,
)


def dataset_robot_asset_payload(episodes: Iterable[Any]) -> dict[str, Any] | None:
    """Validate episode provenance and build the dataset-level payload.

    An Alex V2 export must carry one identical reference on every episode and the
    complete generated manifest in ``episode.extras['robot_asset_manifest']``.
    Synthetic non-Alex test exports may omit the robot-asset payload.
    """
    values = list(episodes)
    if not values:
        raise AlexV2ContractError("cannot derive robot provenance from no episodes")
    tasks = {str(item.meta.task) for item in values}
    if len(tasks) != 1:
        raise AlexV2ContractError(f"dataset export cannot mix episode tasks: {sorted(tasks)}")
    is_v2 = any(str(item.meta.task) == paths.ALEX_V2_TASK for item in values)
    raw_refs = {
        (str(item.meta.robot_asset_id), str(item.meta.robot_asset_sha256)) for item in values
    }
    if len(raw_refs) != 1:
        raise AlexV2ContractError("episodes do not share one robot asset id and fingerprint")
    asset_id, sha256 = raw_refs.pop()
    if not asset_id and not sha256:
        if is_v2:
            raise AlexV2ContractError("Alex V2 episodes require robot asset provenance")
        return None
    if not asset_id or not sha256:
        raise AlexV2ContractError("robot asset id and sha256 must be set together")
    ref = RobotAssetRef(asset_id=asset_id, sha256=sha256)

    manifests = [item.extras.get("robot_asset_manifest") for item in values]
    present = [value for value in manifests if value is not None]
    if not present:
        if is_v2:
            raise AlexV2ContractError("Alex V2 dataset export requires the full asset manifest")
        return ref.to_dict()
    first = present[0]
    if not isinstance(first, Mapping):
        raise AlexV2ContractError("robot_asset_manifest must be a mapping")
    if len(present) != len(values) or any(value != first for value in present[1:]):
        raise AlexV2ContractError("episodes do not carry one identical robot asset manifest")
    manifest = dict(first)
    validated = validate_alex_v2_manifest(manifest)
    if (validated.asset_id, validated.sha256) != (ref.asset_id, ref.sha256):
        raise AlexV2ContractError(
            "episode robot asset reference does not match its canonical manifest fingerprint"
        )
    return {**validated.to_dict(), "manifest": manifest}


def load_dataset_robot_asset(
    dataset_dir: str | Path,
    *,
    require: bool = False,
) -> tuple[RobotAssetRef | None, dict[str, Any] | None]:
    """Read and validate ``meta.json`` robot provenance."""
    meta_path = Path(dataset_dir) / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AlexV2ContractError(f"cannot read dataset metadata {meta_path}: {error}") from error
    payload = meta.get("robot_asset")
    if payload is None:
        if require:
            raise AlexV2ContractError(f"dataset {dataset_dir} has no robot asset provenance")
        return None, None
    if not isinstance(payload, Mapping):
        raise AlexV2ContractError("dataset robot_asset metadata must be an object")
    ref = RobotAssetRef.from_dict(payload)
    manifest_value = payload.get("manifest")
    if manifest_value is None:
        if require:
            raise AlexV2ContractError("Alex V2 dataset metadata does not embed its manifest")
        return ref, None
    if not isinstance(manifest_value, Mapping):
        raise AlexV2ContractError("dataset robot asset manifest must be an object")
    manifest = dict(manifest_value)
    if validate_alex_v2_manifest(manifest) != ref:
        raise AlexV2ContractError("dataset robot asset fingerprint does not match its manifest")
    return ref, manifest


def validate_dataset_episode_robot_asset(dataset: Any, ref: RobotAssetRef) -> None:
    """Require every loaded V2 episode to match dataset-level provenance."""
    for record in dataset.records:
        meta = record.meta
        if str(meta.get("robot", "")) != paths.ALEX_V2_ROBOT_TAG:
            raise AlexV2ContractError(
                f"episode {record.episode_id} has robot tag {meta.get('robot')!r}; "
                f"expected {paths.ALEX_V2_ROBOT_TAG!r}"
            )
        if (
            str(meta.get("robot_asset_id", "")),
            str(meta.get("robot_asset_sha256", "")),
        ) != (ref.asset_id, ref.sha256):
            raise AlexV2ContractError(
                f"episode {record.episode_id} robot asset identity differs from meta.json"
            )


__all__ = [
    "dataset_robot_asset_payload",
    "load_dataset_robot_asset",
    "validate_dataset_episode_robot_asset",
]
