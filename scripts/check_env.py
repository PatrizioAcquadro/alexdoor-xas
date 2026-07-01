#!/usr/bin/env python
"""Phase-1 environment readiness check (fast, no Isaac app launch).

Reports the Python / Isaac Sim / Isaac Lab / PyTorch versions and CUDA
availability, then confirms every registered external asset exists. Exits
non-zero if a required package or asset is missing.

Run inside ``env_alex`` (see docs/environment.md)::

    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def main() -> int:
    from alexdoor_xas import paths

    print("== AlexDoor-XAS environment check ==")
    print(f"python      : {platform.python_version()}  ({sys.executable})")

    versions = {name: _pkg_version(name) for name in ("isaacsim", "isaaclab", "torch", "numpy")}
    for name, ver in versions.items():
        print(f"{name:<12}: {ver}")

    # CUDA (importing torch is cheap enough and is the only reliable probe).
    try:
        import torch

        if torch.cuda.is_available():
            print(f"cuda        : available ({torch.cuda.get_device_name(0)})")
        else:
            print("cuda        : NOT available")
    except Exception as exc:  # noqa: BLE001
        print(f"cuda        : could not probe ({exc.__class__.__name__}: {exc})")

    print(f"assets_root : {paths.ASSETS_ROOT}")
    print("-- assets --")
    missing_pkgs = [n for n, v in versions.items() if v == "MISSING"]
    missing_assets = []
    for name, path, required in paths.iter_assets():
        ok = path.exists()
        flag = "ok " if ok else ("ERR" if required else "opt")
        print(f"  [{flag}] {name}: {path}")
        if required and not ok:
            missing_assets.append(name)

    print("-- result --")
    if missing_pkgs:
        print(f"MISSING packages: {missing_pkgs}")
    if missing_assets:
        print(f"MISSING required assets: {missing_assets}")
    if missing_pkgs or missing_assets:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
