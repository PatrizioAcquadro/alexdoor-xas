"""Asset loaders for AlexDoor-XAS.

- :mod:`alexdoor_xas.assets.alex` — load the Alex articulation config + URDF.
- :mod:`alexdoor_xas.assets.scenes` — locate / open the corridor scene and door.

The pure-Python helpers (path resolution, ``package://`` rewriting, opening a USD
stage with ``pxr``) work without a running Isaac app. Anything that builds Isaac
Lab config objects must be called *after* ``AppLauncher`` (see the module docs).
"""

from alexdoor_xas.assets.alex import load_alex_articulation_cfg, resolve_alex_urdf

__all__ = ["load_alex_articulation_cfg", "resolve_alex_urdf"]
