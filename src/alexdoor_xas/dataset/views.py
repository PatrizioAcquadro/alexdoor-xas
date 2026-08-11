"""Direct loading of retained train/validation/test dataset views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .splits import SPLIT_NAMES

VIEW_SCHEMA = "alexdoor_xas.dataset_view.v1"


def load_view_payload(path: str | Path) -> dict[str, Any]:
    """Load required view fields while tolerating legacy administrative metadata."""

    target = Path(path)
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != VIEW_SCHEMA:
        raise ValueError(f"invalid dataset view schema: {target}")
    if not isinstance(payload.get("splits"), dict):
        raise ValueError(f"dataset view has no split mapping: {target}")
    try:
        for name in SPLIT_NAMES:
            values = payload["splits"][name]
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item for item in values
            ):
                raise ValueError(f"dataset view {name} split must be a list of IDs")
    except KeyError as error:
        raise ValueError(f"dataset view is missing split {error.args[0]!r}") from error
    return payload


def view_path(datasets_root: str | Path, task: str, view_id: str) -> Path:
    if not isinstance(view_id, str) or not view_id or "/" in view_id or ".." in view_id:
        raise ValueError("view_id must be a safe single path component")
    return Path(datasets_root) / task / "splits" / f"{view_id}.json"


__all__ = ["VIEW_SCHEMA", "load_view_payload", "view_path"]
