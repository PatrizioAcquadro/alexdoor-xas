"""Noise-scheduler factories and the shared sampling loop (diffusers-backed).

This is the only place the ``diffusers`` schedulers are constructed, so the
train/inference schedules and the checkpoint payload cannot drift apart. The
model is duck-typed as ``model(noisy_actions, timesteps, obs) -> eps_hat``.
"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from alexdoor_xas.policies.diffusion.config import DiffusionConfigError, DiffusionModelCfg


def scheduler_config_payload(cfg: DiffusionModelCfg) -> dict:
    """The plain-dict schedule contract embedded in checkpoints."""
    return {
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_schedule": cfg.beta_schedule,
        "prediction_type": cfg.prediction_type,
        "clip_sample": True,
    }


def make_train_scheduler(cfg: DiffusionModelCfg) -> DDPMScheduler:
    """The DDPM schedule used both to corrupt training targets and to sample."""
    return DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
        clip_sample=True,  # actions are min-max normalized to [-1, 1]
    )


def make_inference_scheduler(
    cfg: DiffusionModelCfg, sampler: str, num_inference_steps: int
) -> DDPMScheduler | DDIMScheduler:
    """A sampler with its timestep subsequence already configured."""
    if not 1 <= num_inference_steps <= cfg.num_train_timesteps:
        raise DiffusionConfigError(
            f"num_inference_steps must be in [1, {cfg.num_train_timesteps}], "
            f"got {num_inference_steps}"
        )
    if sampler == "ddpm":
        scheduler = make_train_scheduler(cfg)
    elif sampler == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
            clip_sample=True,
        )
    else:
        raise DiffusionConfigError(f"unknown sampler {sampler!r} (expected ddpm or ddim)")
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


@torch.no_grad()
def sample_actions(
    model,
    scheduler,
    obs: torch.Tensor,
    horizon: int,
    action_dim: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reverse-diffuse one normalized action chunk per obs row: ``(B, H, A)``.

    ``scheduler`` must already have its timesteps set
    (:func:`make_inference_scheduler`). DDIM runs with its default eta=0, so a
    fixed ``generator`` (which also seeds the initial noise) makes the sample
    fully deterministic; DDPM additionally draws its per-step noise from the
    same generator.
    """
    device = obs.device
    x = torch.randn(
        (obs.shape[0], horizon, action_dim),
        generator=generator,
        device=generator.device if generator is not None else device,
        dtype=torch.float32,
    ).to(device)
    for t in scheduler.timesteps:
        t_batch = torch.full((obs.shape[0],), int(t), device=device, dtype=torch.long)
        eps_hat = model(x, t_batch, obs)
        x = scheduler.step(eps_hat, t, x, generator=generator).prev_sample
    return x


__all__ = [
    "make_inference_scheduler",
    "make_train_scheduler",
    "sample_actions",
    "scheduler_config_payload",
]
