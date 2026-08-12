"""Pure tests for the optional W&B tracking scaffold."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas import paths
from alexdoor_xas.tracking import (
    NoOpWandbRun,
    WandbConfig,
    load_wandb_config,
    sanitize_wandb_config,
    start_wandb_run,
)
from alexdoor_xas.tracking.wandb import WandbConfigError


def test_default_wandb_config_is_disabled_and_safe() -> None:
    cfg = load_wandb_config()

    assert cfg.mode == "disabled"
    assert cfg.project == "alexdoor-xas"
    assert cfg.entity is None
    assert cfg.tags == ()
    assert cfg.log_artifacts is False
    assert cfg.dir == paths.WANDB_CACHE_DIR


def test_disabled_run_is_noop_and_does_not_import_wandb(monkeypatch) -> None:
    def fail_import(name: str):
        if name == "wandb":
            raise AssertionError("disabled mode must not import wandb")
        raise ImportError(name)

    monkeypatch.setattr("alexdoor_xas.tracking.wandb.importlib.import_module", fail_import)

    with start_wandb_run(WandbConfig(mode="disabled")) as run:
        assert isinstance(run, NoOpWandbRun)
        assert run.disabled
        assert run.log({"metric": 1}) is None
        assert run.log_artifact(object()) is None


@pytest.mark.parametrize(
    "values, message",
    [
        ({"mode": "bad"}, "mode"),
        ({"project": ""}, "project"),
        ({"tags": ["phase3", "phase3"]}, "duplicates"),
        ({"log_artifacts": "yes"}, "log_artifacts"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_invalid_config_is_rejected(values, message: str) -> None:
    with pytest.raises(WandbConfigError, match=message):
        WandbConfig.from_mapping(values)


def test_sanitize_config_drops_secret_like_keys_and_normalizes_values() -> None:
    clean = sanitize_wandb_config(
        {
            "train.loss": 0.1,
            "WANDB_API_KEY": "do-not-keep",
            "nested": {"password": "do-not-keep", "path": Path("datasets/x")},
            "bad_float": float("inf"),
        }
    )

    assert clean == {
        "train_loss": 0.1,
        "nested": {"path": "datasets/x"},
        "bad_float": "inf",
    }


def test_offline_run_lazy_imports_wandb_and_logs(tmp_path, monkeypatch) -> None:
    calls = {"init": None, "log": [], "finish": 0, "artifacts": []}

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name
            self.type = type
            self.metadata = metadata or {}
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    class FakeRun:
        url = "offline-url"

        def log(self, data, step=None):
            calls["log"].append((data, step))

        def finish(self):
            calls["finish"] += 1

        def log_artifact(self, artifact):
            calls["artifacts"].append(artifact)
            return artifact

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.update(init=kwargs) or FakeRun(),
        Artifact=FakeArtifact,
    )
    monkeypatch.setattr(
        "alexdoor_xas.tracking.wandb.importlib.import_module",
        lambda name: fake_wandb if name == "wandb" else None,
    )

    cfg = WandbConfig(mode="offline", dir=tmp_path, log_artifacts=False)
    artifact_path = tmp_path / "tiny.json"
    artifact_path.write_text("{}\n")
    with start_wandb_run(cfg, config={"secret_token": "drop", "epochs": 1}) as run:
        assert not run.disabled
        run.log({"loss": 1.0}, step=2)
        assert run.log_file_artifact(artifact_path, name="blocked") is None
        logged = run.log_file_artifact(artifact_path, name="allowed", allow=True)

    assert calls["init"]["project"] == "alexdoor-xas"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["dir"] == str(tmp_path)
    assert calls["init"]["config"] == {"epochs": 1}
    assert calls["log"] == [({"loss": 1.0}, 2)]
    assert calls["finish"] == 1
    assert logged.name == "allowed"
    assert calls["artifacts"] == [logged]


def test_missing_wandb_dependency_has_actionable_error(tmp_path, monkeypatch) -> None:
    def raise_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("alexdoor_xas.tracking.wandb.importlib.import_module", raise_import)

    with pytest.raises(RuntimeError, match=r"\.\[tracking\]"):
        start_wandb_run(WandbConfig(mode="offline", dir=tmp_path))
