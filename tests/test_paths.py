"""Light Phase-1 tests: path resolution + pure-Python imports (no Isaac launch).

These keep the scaffold honest — that the package imports, the path registry
resolves, and the referenced assets exist on this machine. They intentionally do
NOT import Isaac Lab; the full sim load is exercised by scripts/verify_assets.py.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from alexdoor_xas import paths


def test_repo_root_resolves() -> None:
    assert paths.REPO_ROOT.is_dir()
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()


def test_required_assets_exist() -> None:
    missing = [
        name
        for name, path, required in paths.iter_assets()
        if required and not path.exists()
    ]
    assert not missing, f"missing required assets: {missing}"


def test_alex_v2_path_surface_uses_the_static_standard_asset() -> None:
    assert paths.ALEX_V2_URDF == paths.ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2.urdf"
    assert paths.iter_alex_v2_assets() == [
        ("Alex V2 asset root", paths.ALEX_V2_ASSET_ROOT, True),
        ("Alex V2 URDF", paths.ALEX_V2_URDF, True),
    ]
    assert not hasattr(paths, "ALEX_V2_BRIDGE_ROOT")
    assert not hasattr(paths, "IHMC_ALEX_SDK_ROOT")


def test_assets_module_imports_without_isaac() -> None:
    mod = importlib.import_module("alexdoor_xas.assets")
    assert hasattr(mod, "build_alex_v2_door_asset")
    assert hasattr(mod, "load_alex_v2_articulation_cfg")


def test_alex_v2_urdf_is_the_required_registered_asset() -> None:
    registered = {path for _name, path, required in paths.iter_assets() if required}
    assert paths.ALEX_V2_ASSET_ROOT in registered
    assert paths.ALEX_V2_URDF in registered
    assert Path(paths.ALEX_V2_URDF).is_file()
