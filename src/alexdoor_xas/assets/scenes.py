"""Scene asset helpers for AlexDoor-XAS.

Thin wrappers around the referenced-in-place USD scenes. Path getters and the
``pxr``-based :func:`open_stage` work without a running Isaac app;
:func:`scene_spawn_cfg` returns an Isaac Lab ``UsdFileCfg`` and therefore
requires Isaac Lab (call after ``AppLauncher``).
"""

from __future__ import annotations

from pathlib import Path

from alexdoor_xas import paths


def combined_scene_usd() -> Path:
    """The corridor-with-many-rooms scene (hallway + 4 iThor floorplans)."""
    return paths.COMBINED_SCENE_USD


def door_usd() -> Path:
    """The standalone articulated door (handle + hinge)."""
    return paths.DOOR_USD


def open_stage(usd_path: str | Path, load_all: bool = True):
    """Open a USD stage with ``pxr`` (no Isaac app / no PhysX). Returns the stage.

    ``load_all=True`` loads every payload, so all referenced sub-scenes and
    objects are actually opened — a real "does it compose + resolve" check —
    without any physics cooking.
    """
    from pxr import Usd

    usd_path = Path(usd_path)
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD not found: {usd_path}")
    load_rule = Usd.Stage.LoadAll if load_all else Usd.Stage.LoadNone
    stage = Usd.Stage.Open(str(usd_path), load_rule)
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {usd_path}")
    return stage


def scene_spawn_cfg(usd_path: str | Path):
    """Return an Isaac Lab ``UsdFileCfg`` referencing ``usd_path``.

    Requires Isaac Lab. Spawn with e.g.::

        cfg = scene_spawn_cfg(combined_scene_usd())
        cfg.func("/World/Scene", cfg)
    """
    import isaaclab.sim as sim_utils

    return sim_utils.UsdFileCfg(usd_path=str(usd_path))
