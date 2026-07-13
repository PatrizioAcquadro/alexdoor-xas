"""Pure tests for the Diffusion Policy Hydra/OmegaConf config layer (Phase 3.3)."""

from __future__ import annotations

import pytest

from alexdoor_xas import paths
from alexdoor_xas.action.spaces import A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.policies.diffusion import DiffusionConfigError, load_diffusion_config


def test_default_config_matches_yaml_defaults() -> None:
    cfg = load_diffusion_config()

    assert cfg.dataset.task == "door_push_alex_v2"
    assert cfg.dataset.space == A2_EE_DELTA
    assert cfg.dataset.version == "v2_pose"
    assert cfg.dataset.obs_preset == "core"
    assert cfg.dataset.dataset_dir == (
        paths.DATASETS_DIR / "door_push_alex_v2" / A2_EE_DELTA / "v2_pose"
    )

    assert cfg.model.horizon == 16
    assert cfg.model.d_model == 128
    assert cfg.model.n_heads == 4
    assert cfg.model.n_decoder_layers == 4
    assert cfg.model.dim_feedforward == 512
    assert cfg.model.dropout == pytest.approx(0.1)
    assert cfg.model.num_train_timesteps == 100
    assert cfg.model.beta_schedule == "squaredcos_cap_v2"
    assert cfg.model.prediction_type == "epsilon"

    assert cfg.train.epochs == 300
    assert cfg.train.batch_size == 64
    assert cfg.train.lr == pytest.approx(1.0e-4)
    assert cfg.train.weight_decay == pytest.approx(1.0e-3)
    assert cfg.train.grad_clip == pytest.approx(1.0)
    assert cfg.train.lr_schedule == "cosine"
    assert cfg.train.lr_warmup_steps == 500
    assert cfg.train.use_ema is True
    assert cfg.train.ema_decay == pytest.approx(0.999)
    assert cfg.train.seed == 0
    assert cfg.train.device == "cuda"
    assert cfg.train.val_every == 10
    assert cfg.train.val_inference_steps == 10
    assert cfg.train.overfit_episodes is None

    assert cfg.run.experiment == "diffusion_door_push"
    assert cfg.run.run_id is None
    assert cfg.run.output_root is None

    assert cfg.rollout.checkpoint is None
    assert cfg.rollout.episodes_fixed == 5
    assert cfg.rollout.episodes_randomized == 15
    assert cfg.rollout.base_seed == 100
    assert cfg.rollout.max_ticks == 600
    assert cfg.rollout.success_angle_deg == pytest.approx(45.0)
    assert cfg.rollout.n_action_steps == 8
    assert cfg.rollout.sampler == "ddpm"
    assert cfg.rollout.num_inference_steps == 100
    assert cfg.rollout.policy_device == "cuda"
    assert cfg.rollout.reference_metrics is None
    assert cfg.rollout.matched_scripted_reference is False

    assert cfg.wandb_overrides == {}


def test_hydra_overrides_flow_through_all_sections() -> None:
    cfg = load_diffusion_config(
        [
            "dataset.space=A3_obj_rel_ee_delta",
            "dataset.obs_preset=core_contact",
            "model.horizon=8",
            "model.d_model=64",
            "model.num_train_timesteps=25",
            "train.epochs=3",
            "train.lr_schedule=constant",
            "train.use_ema=false",
            "train.overfit_episodes=2",
            "run.run_id=test_run",
            "run.output_root=/tmp/diffusion-pilot",
            "rollout.n_action_steps=4",
            "rollout.sampler=ddim",
            "rollout.num_inference_steps=10",
            "rollout.matched_scripted_reference=true",
            "+wandb.mode=offline",
        ]
    )

    assert cfg.dataset.space == A3_OBJ_REL_EE_DELTA
    assert cfg.dataset.obs_preset == "core_contact"
    assert cfg.model.horizon == 8
    assert cfg.model.d_model == 64
    assert cfg.model.num_train_timesteps == 25
    assert cfg.train.epochs == 3
    assert cfg.train.lr_schedule == "constant"
    assert cfg.train.use_ema is False
    assert cfg.train.overfit_episodes == 2
    assert cfg.run.run_id == "test_run"
    assert cfg.run.output_root == "/tmp/diffusion-pilot"
    assert cfg.resolved_run_id() == "test_run"
    assert cfg.rollout.n_action_steps == 4
    assert cfg.rollout.sampler == "ddim"
    assert cfg.rollout.num_inference_steps == 10
    assert cfg.rollout.matched_scripted_reference is True
    assert cfg.wandb_overrides == {"mode": "offline"}


def test_cli_overrides_take_precedence_over_hydra() -> None:
    cfg = load_diffusion_config(
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
    cfg = load_diffusion_config(["train.seed=7"])
    run_id = cfg.resolved_run_id()
    assert run_id.endswith(f"_{A2_EE_DELTA}_seed7")


@pytest.mark.parametrize(
    "override, message",
    [
        ("dataset.space=A1_joint_delta", "dataset.space"),
        ("dataset.space=A4_obj_centric_chunk", "dataset.space"),
        ("dataset.obs_preset=bogus", "dataset.obs_preset"),
        ("model.n_heads=3", "divisible"),
        ("model.horizon=0", "model.horizon"),
        ("model.dropout=1.5", "model.dropout"),
        ("model.beta_schedule=linear", "model.beta_schedule"),
        ("model.prediction_type=sample", "model.prediction_type"),
        ("train.epochs=0", "train.epochs"),
        ("train.lr=-1.0", "train.lr"),
        ("train.ema_decay=1.0", "train.ema_decay"),
        ("train.lr_schedule=step", "train.lr_schedule"),
        ("train.overfit_episodes=0", "train.overfit_episodes"),
        ("rollout.max_ticks=0", "rollout.max_ticks"),
        ("rollout.success_angle_deg=0.0", "rollout.success_angle_deg"),
        ("rollout.n_action_steps=0", "rollout.n_action_steps"),
        ("rollout.sampler=heun", "rollout.sampler"),
    ],
)
def test_invalid_config_is_rejected(override: str, message: str) -> None:
    with pytest.raises(DiffusionConfigError, match=message):
        load_diffusion_config([override])


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["rollout.n_action_steps=32"], "n_action_steps"),
        (["rollout.num_inference_steps=200"], "num_inference_steps"),
        (["train.val_inference_steps=200"], "val_inference_steps"),
    ],
)
def test_cross_section_constraints_are_enforced(overrides: list[str], message: str) -> None:
    with pytest.raises(DiffusionConfigError, match=message):
        load_diffusion_config(overrides)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(DiffusionConfigError, match="not_a_field"):
        load_diffusion_config(["+model.not_a_field=1"])
    with pytest.raises(DiffusionConfigError, match="unknown CLI override"):
        load_diffusion_config(cli_overrides={"nosection": 1})


def test_non_key_value_hydra_tokens_are_rejected() -> None:
    with pytest.raises(DiffusionConfigError, match="key=value"):
        load_diffusion_config(["--not-a-hydra-token"])
