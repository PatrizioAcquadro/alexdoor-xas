"""Pure tests for the scripted baseline Hydra/OmegaConf config layer."""

from __future__ import annotations

import pytest

from alexdoor_xas.policies.scripted import DoorPushControllerCfg
from alexdoor_xas.scripted_baseline_config import (
    ScriptedBaselineConfigError,
    apply_controller_overrides,
    load_scripted_baseline_config,
)


def test_default_config_matches_legacy_script_defaults() -> None:
    cfg = load_scripted_baseline_config()

    assert cfg.run.robot == "proxy"
    assert cfg.run.episodes == 5
    assert cfg.run.randomized == 0
    assert cfg.run.seed == 0
    assert cfg.run.experiment is None
    assert cfg.run.run_id is None
    assert cfg.run.success_angle_deg == pytest.approx(45.0)
    assert cfg.run.max_ticks == 600
    assert cfg.run.video is False
    assert cfg.run.clean_shutdown is False
    assert cfg.controller_overrides == {}


def test_hydra_overrides_update_run_and_controller_settings() -> None:
    cfg = load_scripted_baseline_config(
        [
            "run.robot=alex_v2",
            "run.episodes=7",
            "run.randomized=2",
            "run.seed=11",
            "run.experiment=hydra_test",
            "run.run_id=seed11",
            "run.video=true",
            "controller.overrides.push_height_m=-0.25",
            "controller.overrides.contact_max_ticks=12",
        ]
    )

    assert cfg.run.robot == "alex_v2"
    assert cfg.run.episodes == 7
    assert cfg.run.randomized == 2
    assert cfg.run.seed == 11
    assert cfg.run.experiment == "hydra_test"
    assert cfg.run.run_id == "seed11"
    assert cfg.run.video is True
    assert cfg.controller_overrides == {
        "push_height_m": pytest.approx(-0.25),
        "contact_max_ticks": 12,
    }


def test_legacy_cli_overrides_take_precedence_over_hydra() -> None:
    cfg = load_scripted_baseline_config(
        ["run.robot=proxy", "run.episodes=2", "run.video=false"],
        cli_overrides={"robot": "alex_v2", "episodes": 9, "video": True},
    )

    assert cfg.run.robot == "alex_v2"
    assert cfg.run.episodes == 9
    assert cfg.run.video is True


@pytest.mark.parametrize(
    "override, message",
    [
        ("run.robot=bad", "run.robot"),
        ("run.episodes=-1", "run.episodes"),
        ("run.randomized=-1", "run.randomized"),
        ("run.max_ticks=0", "run.max_ticks"),
    ],
)
def test_invalid_run_config_is_rejected(override: str, message: str) -> None:
    with pytest.raises(ScriptedBaselineConfigError, match=message):
        load_scripted_baseline_config([override])


def test_unknown_controller_override_is_rejected() -> None:
    with pytest.raises(ScriptedBaselineConfigError, match="not_in_controller"):
        load_scripted_baseline_config(["+controller.overrides.not_in_controller=1.0"])


def test_non_key_value_hydra_tokens_are_rejected() -> None:
    with pytest.raises(ScriptedBaselineConfigError, match="key=value"):
        load_scripted_baseline_config(["--not-a-hydra-token"])


def test_apply_controller_overrides_preserves_controller_type() -> None:
    base = DoorPushControllerCfg()
    cfg = apply_controller_overrides(
        base,
        {"push_height_m": -0.25, "contact_max_ticks": 12},
    )

    assert isinstance(cfg, DoorPushControllerCfg)
    assert cfg.push_height_m == pytest.approx(-0.25)
    assert cfg.contact_max_ticks == 12
    assert base.push_height_m != cfg.push_height_m
