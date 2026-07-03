"""Load the IHMC Alex articulation config for AlexDoor-XAS.

Two responsibilities, kept self-contained so this project never depends on the
legacy-only custom Isaac suitcase or its IHMC IsaacLab shim:

1. :func:`resolve_alex_urdf` — rewrite the URDF's ``package://alex_V1_description/``
   mesh references to absolute paths (into a temp cache) so the Isaac URDF
   importer resolves meshes out-of-tree. Pure-Python; safe to call anytime. This
   mirrors the proven approach in the Alex-robot walking entrypoint.
2. :func:`load_alex_articulation_cfg` — import the vendored actuator/articulation
   config (``alex_models/alex_V1_isaacsim/alex.py``) by absolute path, deep-copy
   the requested variant, and point its ``spawn.asset_path`` at the resolved URDF.
   This imports Isaac Lab, so it must be called **after** ``AppLauncher``.

Variants:
  - ``"fullbody"`` → ``ALEX_V1_FULLBODY_DEFAULT_CFG`` (adds wrist/gripper; default).
  - ``"fullbody_fullcollisions"`` → same cfg, but the URDF authors collision
    geometry on every link (the default fullbody URDF has NO arm collisions —
    required whenever the arm must contact the world, e.g. door pushing).
  - ``"nub"``      → ``ALEX_V1_NUBS_DEFAULT_CFG`` (nub forearms, 23 DoF).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from alexdoor_xas import paths

_PKG_PREFIX = "package://alex_V1_description/"
_URDF_CACHE_DIR = Path("/tmp/alexdoor-xas-urdf")

_CFG_ATTR = {
    "fullbody": "ALEX_V1_FULLBODY_DEFAULT_CFG",
    "fullbody_fullcollisions": "ALEX_V1_FULLBODY_DEFAULT_CFG",
    "nub": "ALEX_V1_NUBS_DEFAULT_CFG",
}


def resolve_alex_urdf(variant: str = "fullbody") -> str:
    """Return a path to an Alex URDF whose meshes resolve on this machine.

    If the source URDF uses ``package://`` mesh refs, they are rewritten to
    absolute paths (rooted at :data:`alexdoor_xas.paths.ALEX_MESH_ROOT`) into a
    cached copy under ``/tmp``. The cache is rebuilt when the source changes.
    """
    src = paths.urdf_for(variant)
    if not src.is_file():
        raise FileNotFoundError(f"Alex URDF not found: {src}")

    text = src.read_text()
    if _PKG_PREFIX not in text:
        return str(src)  # already absolute / no rewrite needed

    _URDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = _URDF_CACHE_DIR / f"{src.stem}_abs.urdf"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        abs_prefix = str(paths.ALEX_MESH_ROOT) + "/"
        dst.write_text(text.replace(_PKG_PREFIX, abs_prefix))
    return str(dst)


def _load_vendored_alex_module() -> ModuleType:
    """Import ``alex_V1_isaacsim/alex.py`` by absolute path (imports Isaac Lab)."""
    cfg_py = paths.ALEX_ISAAC_CFG_PY
    if not cfg_py.is_file():
        raise FileNotFoundError(f"Alex IsaacLab config not found: {cfg_py}")
    spec = importlib.util.spec_from_file_location("alexdoor_xas._vendored_alex_cfg", cfg_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {cfg_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_alex_articulation_cfg(variant: str = "fullbody", *, fix_base: bool = False):
    """Return a ready-to-spawn ``ArticulationCfg`` for the given Alex variant.

    Deep-copies the vendored config and sets ``spawn.asset_path`` to the resolved
    URDF. ``fix_base=True`` welds the pelvis to the world (the vendored cfg is a
    free-floating biped). Requires Isaac Lab (call after ``AppLauncher``). The
    returned cfg still needs a ``prim_path`` — e.g.
    ``cfg.replace(prim_path="/World/Alex")``.
    """
    import copy

    if variant not in _CFG_ATTR:
        raise ValueError(f"unknown Alex variant {variant!r} (expected {list(_CFG_ATTR)})")

    module = _load_vendored_alex_module()
    cfg = copy.deepcopy(getattr(module, _CFG_ATTR[variant]))
    cfg.spawn.asset_path = resolve_alex_urdf(variant)
    cfg.spawn.fix_base = fix_base
    return cfg
