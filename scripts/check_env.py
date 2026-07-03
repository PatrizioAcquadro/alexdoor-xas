#!/usr/bin/env python
"""Phase-1 environment readiness check (fast, no Isaac app launch).

Reports the official Isaac install paths, Python / package versions, and CUDA
availability, then confirms every registered external asset exists. Exits
non-zero if a required package, install path, or asset is missing.

Run through the official Isaac Lab Python (see docs/environment.md)::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

OFFICIAL_ISAAC_SIM_ROOT = Path("/home/pacquadr/isaacsim")
OFFICIAL_ISAAC_LAB_ROOT = Path("/home/pacquadr/IsaacLab")
EXPECTED_ISAAC_SIM_VERSION_PREFIX = "6.0.1"
EXPECTED_ISAAC_LAB_BRANCH = "release/3.0.0-beta2"


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _check_provenance() -> tuple[list[str], list[str]]:
    """Verify exact runtime provenance; returns (failures, warnings)."""
    failures: list[str] = []
    warnings: list[str] = []

    version_file = OFFICIAL_ISAAC_SIM_ROOT / "VERSION"
    if version_file.is_file():
        version = version_file.read_text().splitlines()[0].strip()
        print(f"  isaac sim VERSION : {version}")
        if not version.startswith(EXPECTED_ISAAC_SIM_VERSION_PREFIX):
            warnings.append(
                f"Isaac Sim VERSION is {version!r}, expected "
                f"{EXPECTED_ISAAC_SIM_VERSION_PREFIX}*"
            )
    else:
        print(f"  isaac sim VERSION : MISSING ({version_file})")
        failures.append(f"Isaac Sim VERSION file missing: {version_file}")

    branch = _git(OFFICIAL_ISAAC_LAB_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    describe = _git(OFFICIAL_ISAAC_LAB_ROOT, "describe", "--always", "--dirty")
    print(f"  isaac lab branch  : {branch or 'unavailable'}")
    print(f"  isaac lab describe: {describe or 'unavailable'}")
    if branch is None:
        warnings.append("Isaac Lab git state unavailable (not a git checkout?)")
    elif branch != EXPECTED_ISAAC_LAB_BRANCH:
        warnings.append(
            f"Isaac Lab branch is {branch!r}, expected {EXPECTED_ISAAC_LAB_BRANCH!r}"
        )

    symlink = OFFICIAL_ISAAC_LAB_ROOT / "_isaac_sim"
    if symlink.exists():
        target = symlink.resolve()
        print(f"  _isaac_sim -> {target}")
        if target != OFFICIAL_ISAAC_SIM_ROOT.resolve():
            failures.append(
                f"IsaacLab/_isaac_sim resolves to {target}, "
                f"not {OFFICIAL_ISAAC_SIM_ROOT} (launcher would run a different sim)"
            )
    else:
        print("  _isaac_sim        : absent")
        warnings.append("IsaacLab/_isaac_sim symlink absent (Isaac Lab not linked?)")

    return failures, warnings


def main() -> int:
    from alexdoor_xas import paths

    print("== AlexDoor-XAS environment check ==")
    print(f"python      : {platform.python_version()}  ({sys.executable})")

    versions = {name: _pkg_version(name) for name in ("isaacsim", "isaaclab", "torch", "numpy")}
    for name, ver in versions.items():
        print(f"{name:<12}: {ver}")
    print(f"isaacsim dir: {OFFICIAL_ISAAC_SIM_ROOT}")
    print(f"isaaclab dir: {OFFICIAL_ISAAC_LAB_ROOT}")

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
    print("-- install paths --")
    missing_paths = []
    for name, path in (
        ("official Isaac Sim", OFFICIAL_ISAAC_SIM_ROOT),
        ("official Isaac Lab", OFFICIAL_ISAAC_LAB_ROOT),
    ):
        ok = path.exists()
        print(f"  [{'ok ' if ok else 'ERR'}] {name}: {path}")
        if not ok:
            missing_paths.append(name)

    print("-- provenance --")
    provenance_failures, provenance_warnings = _check_provenance()

    print("-- assets --")
    required_pkgs = ("isaaclab", "torch", "numpy")
    missing_pkgs = [n for n in required_pkgs if versions[n] == "MISSING"]
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
    if versions["isaacsim"] == "MISSING":
        print(
            "NOTE: isaacsim package metadata absent; "
            "the install-path and VERSION-file checks verify the sim instead."
        )
    if missing_paths:
        print(f"MISSING install paths: {missing_paths}")
    if missing_assets:
        print(f"MISSING required assets: {missing_assets}")
    for warning in provenance_warnings:
        print(f"WARN: {warning}")
    for failure in provenance_failures:
        print(f"PROVENANCE FAIL: {failure}")
    if missing_pkgs or missing_paths or missing_assets or provenance_failures:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
