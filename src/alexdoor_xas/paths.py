"""Canonical path registry for AlexDoor-XAS (single source of truth).

Assets are **referenced in place** — the Alex V2 model lives in ``~/Desktop/Alex``
and the scenes in ``~/Desktop/CombinedScene``; nothing is copied into this repo.
The Alex root is overridden with ``ALEX_V2_ASSET_ROOT`` to match Isaac Lab.  The
scene root remains controlled by ``ALEXDOOR_ASSETS_ROOT``.

This module is pure-Python and imports nothing from Isaac — it is safe to import
anywhere (tests, the light env check, and inside Isaac scripts alike).
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Repository ────────────────────────────────────────────────────────────────
# paths.py lives at <repo>/src/alexdoor_xas/paths.py → parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATASETS_DIR: Path = REPO_ROOT / "datasets"  # reusable exported episodes (Phase 2+)
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"  # canonical scenes + learned-policy runs
KNOWLEDGE_DIR: Path = REPO_ROOT / "knowledge"
WIKI_DIR: Path = KNOWLEDGE_DIR / "wiki"
# Backward-compatible name for the canonical documentation directory.
DOCS_DIR: Path = WIKI_DIR

# Runtime caches and arbitrary generated artifacts never belong in outputs/.
RUNTIME_CACHE_ROOT: Path = Path(
    os.environ.get("ALEXDOOR_CACHE_ROOT", str(Path.home() / ".cache" / "alexdoor-xas"))
).expanduser()
DIAGNOSTIC_SCENES_DIR: Path = RUNTIME_CACHE_ROOT / "door_scenes"
CALIBRATION_CACHE_DIR: Path = RUNTIME_CACHE_ROOT / "calibration"
VERIFICATION_CACHE_DIR: Path = RUNTIME_CACHE_ROOT / "verification"
SCRIPTED_RUNS_CACHE_DIR: Path = RUNTIME_CACHE_ROOT / "scripted_runs"
DATASET_INSPECTION_CACHE_DIR: Path = RUNTIME_CACHE_ROOT / "dataset_inspection"
LEGACY_RUNS_CACHE_DIR: Path = RUNTIME_CACHE_ROOT / "legacy_runs"

# ── External asset root (referenced in place, overridable) ───────────────────
ASSETS_ROOT: Path = Path(
    os.environ.get("ALEXDOOR_ASSETS_ROOT", str(Path.home() / "Desktop"))
).expanduser()

# ── Alex V2 simulator-neutral asset ──────────────────────────────────────────
ALEX_V2_ASSET_ROOT: Path = (
    Path(os.environ.get("ALEX_V2_ASSET_ROOT", str(Path.home() / "Desktop" / "Alex")))
    .expanduser()
    .resolve()
)
ALEX_V2_URDF: Path = ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2.urdf"
ALEX_V2_RUNTIME_CACHE_ROOT: Path = Path(
    os.environ.get(
        "ALEXDOOR_V2_RUNTIME_CACHE_ROOT",
        str(Path.home() / ".cache" / "alexdoor-xas" / "alex-v2"),
    )
).expanduser()

# Canonical Alex V2 runtime, dataset, and output identifiers.
ALEX_V2_TASK = "door_push_alex_v2"
ALEX_V2_DATASET_VERSION = "v2_pose"
ALEX_V2_ROBOT_TAG = "alex_v2_fullbody_fixedbase_standard_forearm_v0"
ALEX_V2_DATASETS_DIR: Path = DATASETS_DIR / ALEX_V2_TASK
ALEX_V2_OUTPUTS_DIR: Path = OUTPUTS_DIR / ALEX_V2_TASK
DOOR_SCENE_DIR: Path = OUTPUTS_DIR / "door_scene"

# ── Scenes (CombinedScene) ───────────────────────────────────────────────────
SCENES_ROOT: Path = ASSETS_ROOT / "CombinedScene"
# The "corridor with many rooms": a hallway + 4 iThor floorplans.
COMBINED_SCENE_USD: Path = SCENES_ROOT / "CombinedHallwayScene" / "combinedScene.usda"
# Standalone articulated door (handle + hinge) for the door benchmark.
DOOR_USD: Path = SCENES_ROOT / "Door.usd"


def iter_assets() -> list[tuple[str, Path, bool]]:
    """Registered external assets as ``(name, path, required)`` triples.

    Used by ``scripts/check_env.py`` and the path tests to confirm every
    referenced asset actually exists on this machine.
    """
    return [
        *iter_alex_v2_assets(),
        ("Combined scene USD", COMBINED_SCENE_USD, True),
        ("Door USD", DOOR_USD, True),
    ]


def iter_alex_v2_assets() -> list[tuple[str, Path, bool]]:
    """Static external assets required by the Alex V2 lineage."""
    return [
        ("Alex V2 asset root", ALEX_V2_ASSET_ROOT, True),
        ("Alex V2 URDF", ALEX_V2_URDF, True),
    ]
