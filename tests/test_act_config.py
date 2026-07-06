"""Pure tests for the ACT Hydra/OmegaConf config layer (Phase 3.2)."""

from __future__ import annotations

import pytest

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.policies.act import ActConfigError, load_act_config


def test_default_config_matches_yaml_defaults() -> None:
    cfg = load_act_config()

    assert cfg.dataset.task == "door_push_alex"
    assert cfg.dataset.space == A2_EE_DELTA
    assert cfg.dataset.version == "v0"
    assert cfg.dataset.obs_preset == "core"
    assert cfg.dataset.dataset_dir == (
        paths.DATASETS_DIR / "door_push_alex" / A2_EE_DELTA / "v0"
    )

    assert cfg.model.chunk_size == 40
    assert cfg.model.d_model == 128
    assert cfg.model.n_heads == 4
    assert cfg.model.dim_feedforward == 512
    assert cfg.model.z_dim == 16
    assert cfg.model.cvae_encoder_layers == 2
    assert cfg.model.encoder_layers == 2
    assert cfg.model.decoder_layers == 2
    assert cfg.model.dropout == pytest.approx(0.1)

    assert cfg.train.epochs == 100
    assert cfg.train.batch_size == 64
    assert cfg.train.lr == pytest.approx(1.0e-4)
    assert cfg.train.weight_decay == pytest.approx(1.0e-4)
    assert cfg.train.kl_weight == pytest.approx(10.0)
    assert cfg.train.grad_clip == pytest.approx(1.0)
    assert cfg.train.seed == 0
    assert cfg.train.device == "cpu"
    assert cfg.train.val_every == 5
    assert cfg.train.overfit_episodes is None

    assert cfg.run.experiment == "act_door_push"
    assert cfg.run.run_id is None

    assert cfg.rollout.checkpoint is None
    assert cfg.rollout.episodes_fixed == 5
    assert cfg.rollout.episodes_randomized == 15
    assert cfg.rollout.base_seed == 100
    assert cfg.rollout.max_ticks == 600
    assert cfg.rollout.success_angle_deg == pytest.approx(45.0)
    assert cfg.rollout.temporal_ensemble is False
    assert cfg.rollout.ensemble_m == pytest.approx(0.01)
    assert cfg.rollout.reference_metrics is None

    assert cfg.wandb_overrides == {}


def test_hydra_overrides_flow_through_all_sections() -> None:
    cfg = load_act_config(
        [
            "dataset.space=A3_obj_rel_ee_delta",
            "dataset.obs_preset=core_contact",
            "model.chunk_size=8",
            "model.d_model=64",
            "train.epochs=3",
            "train.overfit_episodes=2",
            "run.run_id=test_run",
            "rollout.temporal_ensemble=true",
            "+wandb.mode=offline",
        ]
    )

    assert cfg.dataset.space == A3_OBJ_REL_EE_DELTA
    assert cfg.dataset.obs_preset == "core_contact"
    assert cfg.model.chunk_size == 8
    assert cfg.model.d_model == 64
    assert cfg.train.epochs == 3
    assert cfg.train.overfit_episodes == 2
    assert cfg.run.run_id == "test_run"
    assert cfg.resolved_run_id() == "test_run"
    assert cfg.rollout.temporal_ensemble is True
    assert cfg.wandb_overrides == {"mode": "offline"}


def test_cli_overrides_take_precedence_over_hydra() -> None:
    cfg = load_act_config(
        ["train.epochs=3", "rollout.max_ticks=100"],
        cli_overrides={
            "train.epochs": 9,
            "rollout.checkpoint": "outputs/x/best.pt",
            "rollout.max_ticks": None,
        },
    )

    assert cfg.train.epochs == 9
    assert cfg.rollout.checkpoint == "outputs/x/best.pt"
    assert cfg.rollout.max_ticks == 100


def test_default_run_id_names_space_and_seed() -> None:
    cfg = load_act_config(["train.seed=7"])
    run_id = cfg.resolved_run_id()
    assert run_id.endswith(f"_{A2_EE_DELTA}_seed7")


@pytest.mark.parametrize(
    "override, message",
    [
        ("dataset.space=A1_joint_delta", "dataset.space"),
        ("dataset.space=A4_obj_centric_chunk", "dataset.space"),
        ("dataset.obs_preset=bogus", "dataset.obs_preset"),
        ("model.n_heads=3", "divisible"),
        ("model.chunk_size=0", "model.chunk_size"),
        ("model.dropout=1.5", "model.dropout"),
        ("train.epochs=0", "train.epochs"),
        ("train.lr=-1.0", "train.lr"),
        ("train.overfit_episodes=0", "train.overfit_episodes"),
        ("rollout.max_ticks=0", "rollout.max_ticks"),
        ("rollout.success_angle_deg=0.0", "rollout.success_angle_deg"),
        ("rollout.ensemble_m=0.0", "rollout.ensemble_m"),
    ],
)
def test_invalid_config_is_rejected(override: str, message: str) -> None:
    with pytest.raises(ActConfigError, match=message):
        load_act_config([override])


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ActConfigError, match="not_a_field"):
        load_act_config(["+model.not_a_field=1"])
    with pytest.raises(ActConfigError, match="unknown CLI override"):
        load_act_config(cli_overrides={"nosection": 1})


def test_non_key_value_hydra_tokens_are_rejected() -> None:
    with pytest.raises(ActConfigError, match="key=value"):
        load_act_config(["--not-a-hydra-token"])
