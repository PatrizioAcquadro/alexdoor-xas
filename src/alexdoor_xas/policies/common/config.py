"""Shared OmegaConf loading for policy configuration dataclasses."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

ConfigT = TypeVar("ConfigT")


def load_config(
    path: str | Path,
    config_type: type[ConfigT],
    error_type: type[ValueError],
    overrides: list[str] | tuple[str, ...] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ConfigT:
    """Load YAML, apply key=value then argparse overrides, and build dataclasses."""
    tokens = list(overrides or ())
    invalid = [token for token in tokens if "=" not in token]
    if invalid:
        raise error_type("overrides must use key=value syntax: " + ", ".join(invalid))

    try:
        config = OmegaConf.load(Path(path))
        OmegaConf.set_struct(config, True)
        normalized = [token[1:] if token.startswith("+") else token for token in tokens]
        if normalized:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist(normalized))
        for key, value in (cli_overrides or {}).items():
            if not _path_exists(config, key):
                raise KeyError(f"unknown CLI override: {key}")
            if value is not None:
                OmegaConf.update(config, key, value, merge=False)
        payload = OmegaConf.to_container(config, resolve=True)
    except KeyError as error:
        raise error_type(str(error).strip("'")) from error
    except Exception as error:  # OmegaConf errors become one stable CLI-facing type.
        raise error_type(str(error)) from error
    return config_from_mapping(payload, config_type, error_type)


def _path_exists(config: Any, path: str) -> bool:
    node = config
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def config_from_mapping(
    payload: Any,
    config_type: type[ConfigT],
    error_type: type[ValueError],
) -> ConfigT:
    if not isinstance(payload, Mapping):
        raise error_type("configuration must be a mapping")
    try:
        return _build_dataclass(config_type, payload, "")
    except (TypeError, ValueError) as error:
        raise error_type(str(error)) from error


def _build_dataclass(cls, payload: Mapping[str, Any], prefix: str):
    field_map = {field.name: field for field in fields(cls)}
    unknown = sorted(set(payload) - set(field_map))
    if unknown:
        location = prefix.rstrip(".") or "config"
        raise ValueError(f"unknown {location} field(s): {', '.join(unknown)}")

    hints = get_type_hints(cls)
    values = {}
    for name, field in field_map.items():
        if name in payload:
            values[name] = _coerce(payload[name], hints[name], f"{prefix}{name}")
        elif field.default is MISSING and field.default_factory is MISSING:
            raise ValueError(f"missing config field: {prefix}{name}")
    return cls(**values)


def _coerce(value: Any, expected: Any, name: str) -> Any:
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        expected = next(arg for arg in args if arg is not type(None))

    if is_dataclass(expected):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        return _build_dataclass(expected, value, f"{name}.")
    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value
    if expected is int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        converted = int(value)
        if isinstance(value, float) and not value.is_integer():
            raise TypeError(f"{name} must be an integer")
        return converted
    if expected is float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        return value
    return value
