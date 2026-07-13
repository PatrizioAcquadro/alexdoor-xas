"""No-Kit checks for the Alex V2 environment readiness helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _check_env_module():
    script = Path(__file__).parents[1] / "scripts" / "check_env.py"
    spec = importlib.util.spec_from_file_location("alexdoor_check_env", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alex_v2_factory_check_fails_loudly_with_branch_action(tmp_path) -> None:
    check_env = _check_env_module()

    failure = check_env._alex_v2_module_failure(
        find_spec=lambda _name: None,
        module_file=tmp_path / "missing" / "alex.py",
    )

    assert "isaaclab_assets.robots.alex is not importable" in failure
    assert "pacquadr/alex-v2-asset" in failure
    assert "isaaclab.sh" in failure


def test_alex_v2_factory_check_accepts_discoverable_module(tmp_path) -> None:
    check_env = _check_env_module()

    failure = check_env._alex_v2_module_failure(
        find_spec=lambda _name: SimpleNamespace(origin="alex.py"),
        module_file=tmp_path / "unused.py",
    )

    assert failure is None


def test_missing_alex_v2_asset_root_is_a_required_failure(tmp_path) -> None:
    check_env = _check_env_module()
    missing_root = tmp_path / "Desktop" / "Alex"

    missing = check_env._missing_required_assets(
        [("Alex V2 asset root", missing_root, True)]
    )

    assert missing == ["Alex V2 asset root"]
