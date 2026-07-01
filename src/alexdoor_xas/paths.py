"""Canonical path registry for AlexDoor-XAS (single source of truth).

Assets are **referenced in place** — the Alex model lives in ``~/Desktop/Alex-robot``
and the scenes in ``~/Desktop/CombinedScene``; nothing is copied into this repo.
The root of those peer folders is ``ASSETS_ROOT`` (default ``~/Desktop``), which
can be overridden with the ``ALEXDOOR_ASSETS_ROOT`` environment variable if the
folders ever move.

This module is pure-Python and imports nothing from Isaac — it is safe to import
anywhere (tests, the light env check, and inside Isaac scripts alike).
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Repository ────────────────────────────────────────────────────────────────
# paths.py lives at <repo>/src/alexdoor_xas/paths.py → parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATASETS_DIR: Path = REPO_ROOT / "datasets"   # reusable exported episodes (Phase 2+)
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"     # per-run artifacts (metrics/plots/…)
DOCS_DIR: Path = REPO_ROOT / "docs"

# ── External asset root (referenced in place, overridable) ───────────────────
ASSETS_ROOT: Path = Path(
    os.environ.get("ALEXDOOR_ASSETS_ROOT", str(Path.home() / "Desktop"))
).expanduser()

# ── Alex model (IHMC Alex V1) ────────────────────────────────────────────────
ALEX_REPO: Path = ASSETS_ROOT / "Alex-robot"
ALEX_MODELS: Path = ALEX_REPO / "alex_models"
ALEX_DESCRIPTION: Path = ALEX_MODELS / "alex_V1_description"
# Root that the URDF's ``package://alex_V1_description/`` refs resolve against.
ALEX_MESH_ROOT: Path = ALEX_DESCRIPTION
ALEX_URDF_DIR: Path = ALEX_DESCRIPTION / "rl_urdf"
# Vendored IsaacLab articulation/actuator config (imported by absolute path).
ALEX_ISAAC_CFG_PY: Path = ALEX_MODELS / "alex_V1_isaacsim" / "alex.py"

# Default (full body, adds wrist/gripper — needed for handle/latch manipulation).
ALEX_URDF: Path = ALEX_URDF_DIR / "alex_v1.rlModel_fullBody_robotAccurate_torsoFootCollisions.urdf"
# Fallback (nub forearms, 23 DoF — proven to load; sufficient for door pushing).
ALEX_URDF_NUB: Path = (
    ALEX_URDF_DIR / "alex_v1.rlModel_nubForearms_robotAccurate_torsoFootCollisions.urdf"
)

# ── Scenes (CombinedScene) ───────────────────────────────────────────────────
SCENES_ROOT: Path = ASSETS_ROOT / "CombinedScene"
# The "corridor with many rooms": a hallway + 4 iThor floorplans.
COMBINED_SCENE_USD: Path = SCENES_ROOT / "CombinedHallwayScene" / "combinedScene.usda"
# Standalone articulated door (handle + hinge) for the door benchmark.
DOOR_USD: Path = SCENES_ROOT / "Door.usd"


def urdf_for(variant: str = "fullbody") -> Path:
    """Return the source URDF for ``variant`` (``"fullbody"`` or ``"nub"``)."""
    if variant == "fullbody":
        return ALEX_URDF
    if variant == "nub":
        return ALEX_URDF_NUB
    raise ValueError(f"unknown Alex variant {variant!r} (expected 'fullbody' or 'nub')")


def iter_assets() -> list[tuple[str, Path, bool]]:
    """Registered external assets as ``(name, path, required)`` triples.

    Used by ``scripts/check_env.py`` and the path tests to confirm every
    referenced asset actually exists on this machine.
    """
    return [
        ("Alex repo", ALEX_REPO, True),
        ("Alex IsaacLab config", ALEX_ISAAC_CFG_PY, True),
        ("Alex mesh root", ALEX_MESH_ROOT, True),
        ("Alex URDF (fullbody)", ALEX_URDF, True),
        ("Alex URDF (nub)", ALEX_URDF_NUB, True),
        ("Combined scene USD", COMBINED_SCENE_USD, True),
        ("Door USD", DOOR_USD, True),
    ]
