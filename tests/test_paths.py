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


def test_urdf_for_variants() -> None:
    assert paths.urdf_for("fullbody") == paths.ALEX_URDF
    assert paths.urdf_for("fullbody_fullcollisions") == paths.ALEX_URDF_FULLBODY_FULLCOLL
    assert paths.urdf_for("nub") == paths.ALEX_URDF_NUB


def test_assets_module_imports_without_isaac() -> None:
    mod = importlib.import_module("alexdoor_xas.assets.alex")
    assert hasattr(mod, "resolve_alex_urdf")
    assert hasattr(mod, "load_alex_articulation_cfg")


def test_resolve_alex_urdf_flattens_package_paths() -> None:
    from alexdoor_xas.assets.alex import resolve_alex_urdf

    out = Path(resolve_alex_urdf("fullbody"))
    assert out.is_file()
    assert "package://" not in out.read_text()
