"""Small, opt-in W&B wrapper for Phase 3 training/evaluation code.

Default mode is ``disabled`` so tests and ordinary repo commands do not talk to
W&B or write local W&B metadata unless a script explicitly enables it.
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from alexdoor_xas import paths

CONFIG_PATH = paths.REPO_ROOT / "configs" / "wandb.yaml"
VALID_MODES = frozenset({"disabled", "offline", "online"})
SENSITIVE_KEY_PARTS = ("api_key", "apikey", "password", "secret", "token", "credential")


class WandbConfigError(ValueError):
    """Raised when W&B tracking config is malformed."""


@dataclass(frozen=True)
class WandbConfig:
    """W&B run settings.

    The API key is deliberately absent: authenticate with ``wandb login`` or
    shell environment variables outside the repo.
    """

    mode: str = "disabled"
    project: str = "alexdoor-xas"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    job_type: str | None = None
    tags: tuple[str, ...] = ()
    dir: Path = paths.OUTPUTS_DIR / "wandb"
    log_artifacts: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> WandbConfig:
        data = dict(values or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise WandbConfigError("unknown W&B config field(s): " + ", ".join(unknown))
        cfg = cls(
            mode=str(data.get("mode", cls.mode)).lower(),
            project=_required_str("project", data.get("project", cls.project)),
            entity=_optional_str("entity", data.get("entity")),
            group=_optional_str("group", data.get("group")),
            name=_optional_str("name", data.get("name")),
            job_type=_optional_str("job_type", data.get("job_type")),
            tags=_normalize_tags(data.get("tags", ())),
            dir=_coerce_path(data.get("dir", cls.dir)),
            log_artifacts=_coerce_bool("log_artifacts", data.get("log_artifacts", False)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise WandbConfigError(
                f"mode must be one of {sorted(VALID_MODES)}, got {self.mode!r}"
            )
        _required_str("project", self.project)
        for name, value in (
            ("entity", self.entity),
            ("group", self.group),
            ("name", self.name),
            ("job_type", self.job_type),
        ):
            _optional_str(name, value)
        if any(not isinstance(tag, str) or not tag for tag in self.tags):
            raise WandbConfigError("tags must be non-empty strings")
        if not isinstance(self.dir, Path):
            raise WandbConfigError("dir must be a pathlib.Path")
        if not isinstance(self.log_artifacts, bool):
            raise WandbConfigError("log_artifacts must be a boolean")

    def replace(self, **kwargs: Any) -> WandbConfig:
        cfg = replace(self, **kwargs)
        cfg.validate()
        return cfg


class NoOpWandbRun:
    """No-op run object returned when tracking is disabled."""

    disabled = True
    mode = "disabled"
    url = None

    def __enter__(self) -> NoOpWandbRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None

    def log_artifact(self, artifact: Any, *args: Any, **kwargs: Any) -> None:
        return None

    def log_file_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        type: str = "dataset",
        metadata: Mapping[str, Any] | None = None,
        allow: bool = False,
    ) -> None:
        return None


class WandbRun:
    """Thin wrapper around a real W&B run with artifact opt-in enforcement."""

    disabled = False

    def __init__(self, run: Any, wandb_module: Any, cfg: WandbConfig):
        self._run = run
        self._wandb = wandb_module
        self.cfg = cfg
        self.mode = cfg.mode

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None)

    def __enter__(self) -> WandbRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        if step is None:
            self._run.log(dict(data))
        else:
            self._run.log(dict(data), step=step)

    def finish(self) -> None:
        self._run.finish()

    def log_artifact(self, artifact: Any, *args: Any, allow: bool = False, **kwargs: Any):
        if not (self.cfg.log_artifacts or allow):
            return None
        return self._run.log_artifact(artifact, *args, **kwargs)

    def log_file_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        type: str = "dataset",
        metadata: Mapping[str, Any] | None = None,
        allow: bool = False,
    ):
        if not (self.cfg.log_artifacts or allow):
            return None
        artifact = self._wandb.Artifact(name=name, type=type, metadata=dict(metadata or {}))
        artifact.add_file(str(path))
        return self._run.log_artifact(artifact)


def load_wandb_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> WandbConfig:
    """Load ``configs/wandb.yaml`` and apply explicit overrides."""

    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.is_file():
        raise WandbConfigError(f"W&B config not found: {path}")
    try:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as error:  # noqa: BLE001 - normalize OmegaConf errors for scripts.
        raise WandbConfigError(str(error)) from error
    if not isinstance(loaded, dict):
        raise WandbConfigError(f"W&B config must be a mapping: {path}")
    data = dict(loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    return WandbConfig.from_mapping(data)


def start_wandb_run(
    cfg: WandbConfig | Mapping[str, Any] | None = None,
    *,
    config: Mapping[str, Any] | None = None,
) -> NoOpWandbRun | WandbRun:
    """Start a W&B run or return a no-op object when disabled."""

    wandb_cfg = cfg if isinstance(cfg, WandbConfig) else WandbConfig.from_mapping(cfg)
    wandb_cfg.validate()
    if wandb_cfg.mode == "disabled":
        return NoOpWandbRun()

    _prepare_wandb_local_dirs(wandb_cfg.dir)
    wandb = _import_wandb()
    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.name,
        group=wandb_cfg.group,
        job_type=wandb_cfg.job_type,
        tags=list(wandb_cfg.tags) if wandb_cfg.tags else None,
        dir=str(wandb_cfg.dir),
        mode=wandb_cfg.mode,
        config=sanitize_wandb_config(config or {}),
    )
    return WandbRun(run, wandb, wandb_cfg)


def sanitize_wandb_config(value: Any) -> Any:
    """Return a W&B-safe config value without obvious secret-like keys."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key).replace(".", "_")
            if _is_sensitive_key(key_str):
                continue
            clean[key_str] = sanitize_wandb_config(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitize_wandb_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _prepare_wandb_local_dirs(root: Path) -> None:
    """Keep W&B local state under the configured ignored output directory."""

    root.mkdir(parents=True, exist_ok=True)
    local_dirs = {
        "WANDB_DIR": root,
        "WANDB_CACHE_DIR": root / "cache",
        "WANDB_CONFIG_DIR": root / "config",
        "WANDB_DATA_DIR": root / "data",
        "WANDB_ARTIFACT_DIR": root / "artifacts",
    }
    for env_name, directory in local_dirs.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(directory))


def _import_wandb():
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking requires the optional dependency: "
            "`/home/pacquadr/IsaacLab/isaaclab.sh -p -m pip install -e \".[tracking]\"`"
        ) from error


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def _required_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WandbConfigError(f"{name} must be a non-empty string")
    return value


def _optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise WandbConfigError(f"{name} must be a string or null")


def _normalize_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        items = [str(item).strip() for item in value]
    else:
        raise WandbConfigError("tags must be a list of strings")
    tags = tuple(item for item in items if item)
    if len(tags) != len(set(tags)):
        raise WandbConfigError("tags must not contain duplicates")
    return tags


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        path = Path(value)
        if not path.is_absolute():
            path = paths.REPO_ROOT / path
        return path
    raise WandbConfigError("dir must be a path string")


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise WandbConfigError(f"{name} must be a boolean")


__all__ = [
    "CONFIG_PATH",
    "NoOpWandbRun",
    "VALID_MODES",
    "WandbConfig",
    "WandbConfigError",
    "WandbRun",
    "load_wandb_config",
    "sanitize_wandb_config",
    "start_wandb_run",
]
