#!/usr/bin/env python
"""Environment readiness preflight (fast, no Isaac app launch).

Reports the official Isaac install paths, Python / package versions, and CUDA
availability, then confirms every registered external asset exists. Exits
non-zero if CUDA, a required package, install path, or asset is unavailable.

Run through the official Isaac Lab Python::

    PYTHONPATH=$PWD /home/pacquadr/IsaacLab/isaaclab.sh -p scripts/check_env.py
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tomllib
from collections.abc import Callable
from importlib import metadata, util
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

OFFICIAL_ISAAC_SIM_ROOT = Path("/home/pacquadr/isaacsim")
OFFICIAL_ISAAC_LAB_ROOT = Path("/home/pacquadr/IsaacLab")
EXPECTED_ISAAC_SIM_VERSION = "6.0.1"
EXPECTED_ISAAC_SIM_BUILD = "6.0.1-rc.7+release.42383.32955d8d.gl"
ISAAC_SIM_APP_FILE = OFFICIAL_ISAAC_SIM_ROOT / "apps" / "isaacsim.exp.full.kit"
EXPECTED_ISAAC_LAB_BRANCH = "release/3.0.0-beta2"
EXPECTED_ISAAC_LAB_REMOTE = "origin/release/3.0.0-beta2"
OFFICIAL_ISAAC_LAB_REMOTES = {
    "git@github.com:isaac-sim/IsaacLab.git",
    "https://github.com/isaac-sim/IsaacLab.git",
}
ALEX_V2_EXTENSION_MODULE = "ihmc_alex_isaaclab.robots.alex_v2"
ALEX_V2_EXTENSION_MODULE_FILE = (
    Path("/home/pacquadr/Desktop/Alex")
    / "source"
    / "ihmc_alex_isaaclab"
    / "ihmc_alex_isaaclab"
    / "robots"
    / "alex_v2.py"
)


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


def _isaac_sim_version_failure(build: str, app_version: str) -> str | None:
    """Require NVIDIA's exact 6.0.1 GA archive identity and app version."""

    if build == EXPECTED_ISAAC_SIM_BUILD and app_version == EXPECTED_ISAAC_SIM_VERSION:
        return None
    return (
        f"Isaac Sim identity is build={build!r}, app={app_version!r}; expected "
        f"official GA build={EXPECTED_ISAAC_SIM_BUILD!r}, app={EXPECTED_ISAAC_SIM_VERSION!r}"
    )


def _cuda_failure(available: bool, probe_error: BaseException | None = None) -> str | None:
    """Require a visible CUDA device for the supported simulator and policy workflows."""

    if probe_error is not None:
        return f"CUDA probe failed: {probe_error.__class__.__name__}: {probe_error}"
    if not available:
        return "CUDA is not available from the supported Isaac Lab Python"
    return None


def _check_provenance() -> tuple[list[str], list[str]]:
    """Verify exact runtime provenance; returns (failures, warnings)."""
    failures: list[str] = []
    warnings: list[str] = []

    version_file = OFFICIAL_ISAAC_SIM_ROOT / "VERSION"
    if version_file.is_file():
        build = version_file.read_text().splitlines()[0].strip()
        try:
            app_version = str(tomllib.loads(ISAAC_SIM_APP_FILE.read_text())["package"]["version"])
        except (KeyError, OSError, tomllib.TOMLDecodeError):
            app_version = "unavailable"
        print(f"  isaac sim build   : {build}")
        print(f"  isaac sim app     : {app_version}")
        version_failure = _isaac_sim_version_failure(build, app_version)
        if version_failure is not None:
            failures.append(version_failure)
    else:
        print(f"  isaac sim VERSION : MISSING ({version_file})")
        failures.append(f"Isaac Sim VERSION file missing: {version_file}")

    branch = _git(OFFICIAL_ISAAC_LAB_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    describe = _git(OFFICIAL_ISAAC_LAB_ROOT, "describe", "--always", "--dirty")
    print(f"  isaac lab branch  : {branch or 'unavailable'}")
    print(f"  isaac lab describe: {describe or 'unavailable'}")
    if branch is None:
        failures.append("Isaac Lab git state unavailable (not a git checkout?)")
    elif branch != EXPECTED_ISAAC_LAB_BRANCH:
        failures.append(f"Isaac Lab branch is {branch!r}, expected {EXPECTED_ISAAC_LAB_BRANCH!r}")
    status = _git(OFFICIAL_ISAAC_LAB_ROOT, "status", "--porcelain")
    if status is None:
        failures.append("Isaac Lab worktree status is unavailable")
    elif status:
        failures.append("Isaac Lab worktree is not clean")
    head = _git(OFFICIAL_ISAAC_LAB_ROOT, "rev-parse", "HEAD")
    remote_head = _git(OFFICIAL_ISAAC_LAB_ROOT, "rev-parse", EXPECTED_ISAAC_LAB_REMOTE)
    if head is None or remote_head is None:
        failures.append(f"Isaac Lab cannot resolve HEAD and {EXPECTED_ISAAC_LAB_REMOTE}")
    elif head != remote_head:
        failures.append(
            f"Isaac Lab HEAD {head} differs from {EXPECTED_ISAAC_LAB_REMOTE} {remote_head}"
        )
    remote_url = _git(OFFICIAL_ISAAC_LAB_ROOT, "remote", "get-url", "origin")
    if remote_url not in OFFICIAL_ISAAC_LAB_REMOTES:
        failures.append(f"Isaac Lab origin is not the official NVIDIA repository: {remote_url!r}")

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


def _alex_v2_module_failure(
    find_spec: Callable[[str], Any] = util.find_spec,
    module_file: Path = ALEX_V2_EXTENSION_MODULE_FILE,
) -> str | None:
    """Return an actionable failure when the external Alex factory is unavailable."""

    try:
        spec = find_spec(ALEX_V2_EXTENSION_MODULE)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as error:
        detail = f"find_spec raised {error.__class__.__name__}: {error}"
    else:
        origin = getattr(spec, "origin", None) if spec is not None else None
        if origin is not None:
            try:
                resolved_origin = Path(origin).resolve()
            except (OSError, RuntimeError):
                resolved_origin = None
            if resolved_origin == module_file.resolve():
                return None
            detail = f"module origin is {resolved_origin}, expected {module_file.resolve()}"
        else:
            detail = "find_spec returned no module origin"
    file_detail = "present" if module_file.is_file() else "missing"
    return (
        f"{ALEX_V2_EXTENSION_MODULE} is not the installed external extension "
        f"({detail}; module file {file_detail}: {module_file}). Install it with "
        f"/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e "
        f"/home/pacquadr/Desktop/Alex/source/ihmc_alex_isaaclab."
    )


def _missing_required_assets(assets: list[tuple[str, Path, bool]]) -> list[str]:
    """Return required asset names whose paths do not exist."""

    return [name for name, path, required in assets if required and not path.exists()]


def main() -> int:
    from alexdoor_xas import paths

    print("== AlexDoor-XAS environment check ==")
    print(f"python      : {platform.python_version()}  ({sys.executable})")

    versions = {
        name: _pkg_version(name)
        for name in ("isaacsim", "isaaclab", "ihmc-alex-isaaclab", "torch", "numpy")
    }
    for name, ver in versions.items():
        print(f"{name:<12}: {ver}")
    print(f"isaacsim dir: {OFFICIAL_ISAAC_SIM_ROOT}")
    print(f"isaaclab dir: {OFFICIAL_ISAAC_LAB_ROOT}")

    # CUDA (importing torch is cheap enough and is the only reliable probe).
    cuda_failure = None
    try:
        import torch

        if torch.cuda.is_available():
            print(f"cuda        : available ({torch.cuda.get_device_name(0)})")
        else:
            print("cuda        : NOT available")
            cuda_failure = _cuda_failure(False)
    except Exception as error:  # noqa: BLE001
        print(f"cuda        : could not probe ({error.__class__.__name__}: {error})")
        cuda_failure = _cuda_failure(False, error)

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
    alex_v2_module_failure = _alex_v2_module_failure()
    if alex_v2_module_failure is None:
        print(f"  [ok ] {ALEX_V2_EXTENSION_MODULE}")
    else:
        print(f"  [ERR] {ALEX_V2_EXTENSION_MODULE}")
        provenance_failures.append(alex_v2_module_failure)

    print("-- assets --")
    required_pkgs = ("isaaclab", "ihmc-alex-isaaclab", "torch", "numpy")
    missing_pkgs = [n for n in required_pkgs if versions[n] == "MISSING"]
    assets = paths.iter_assets()
    missing_assets = _missing_required_assets(assets)
    for name, path, required in assets:
        ok = path.exists()
        flag = "ok " if ok else ("ERR" if required else "opt")
        print(f"  [{flag}] {name}: {path}")

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
    if cuda_failure is not None:
        print(f"CUDA FAIL: {cuda_failure}")
    for warning in provenance_warnings:
        print(f"WARN: {warning}")
    for failure in provenance_failures:
        print(f"PROVENANCE FAIL: {failure}")
    if cuda_failure or missing_pkgs or missing_paths or missing_assets or provenance_failures:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
