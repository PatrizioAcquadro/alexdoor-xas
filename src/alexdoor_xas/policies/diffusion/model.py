"""State-only time-series diffusion transformer (Chi et al. 2023, section 3.1).

The noise-prediction network eps_theta(A_t^k, k, O_t): noisy action tokens go
through a causally masked transformer decoder whose cross-attention memory
holds the diffusion-timestep embedding and the projected state observation
(To = 1 — matching the frozen ``ChunkSampler``; the paper reports state-based
insensitivity to the observation horizon). The transformer backbone was chosen
over the paper's CNN default because the project is moving toward
transformer-based VLA policies. No Isaac imports; torch only.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg


def sinusoidal_table(n_positions: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal position embeddings, shape ``(n_positions, d_model)``."""
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    table = torch.zeros(n_positions, d_model)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return table


class DiffusionTransformer(nn.Module):
    """Epsilon-prediction transformer over one noisy action chunk."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: DiffusionModelCfg) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

        d = cfg.d_model
        self.action_proj = nn.Linear(action_dim, d)
        self.obs_mlp = nn.Sequential(nn.Linear(obs_dim, d), nn.Mish(), nn.Linear(d, d))
        # Timestep k is an integer in [0, T): a fixed sinusoidal lookup + MLP.
        self.register_buffer(
            "timestep_table", sinusoidal_table(cfg.num_train_timesteps, d), persistent=False
        )
        self.timestep_mlp = nn.Sequential(nn.Linear(d, d), nn.Mish(), nn.Linear(d, d))
        self.memory_pos = nn.Parameter(torch.zeros(2, d))

        self.register_buffer("query_pos", sinusoidal_table(cfg.horizon, d), persistent=False)
        causal = nn.Transformer.generate_square_subsequent_mask(cfg.horizon)
        self.register_buffer("causal_mask", causal, persistent=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.n_decoder_layers,
        )
        self.eps_head = nn.Linear(d, action_dim)

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:
        """``(B, H, A), (B,), (B, obs_dim) -> (B, H, A)`` predicted noise."""
        batch = noisy_actions.shape[0]
        expected = (batch, self.cfg.horizon, self.action_dim)
        if tuple(noisy_actions.shape) != expected:
            raise ValueError(
                f"expected noisy actions of shape {expected}, got {tuple(noisy_actions.shape)}"
            )
        if tuple(obs.shape) != (batch, self.obs_dim):
            raise ValueError(
                f"expected obs of shape {(batch, self.obs_dim)}, got {tuple(obs.shape)}"
            )

        t_token = self.timestep_mlp(self.timestep_table[timesteps.to(torch.long)])
        obs_token = self.obs_mlp(obs)
        memory = torch.stack([t_token, obs_token], dim=1) + self.memory_pos

        tokens = self.action_proj(noisy_actions) + self.query_pos
        decoded = self.decoder(tokens, memory, tgt_mask=self.causal_mask)
        return self.eps_head(decoded)


def diffusion_loss(
    model: DiffusionTransformer,
    scheduler,
    actions: torch.Tensor,
    obs: torch.Tensor,
    is_pad: torch.Tensor,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Masked epsilon-prediction MSE over one normalized batch.

    Samples ``t ~ U[0, T)`` and Gaussian noise, corrupts the actions with the
    training schedule, and regresses the model output onto the drawn noise.
    Padded chunk steps are masked out of the loss exactly like ``act_loss``
    masks its L1 (they are never executed at rollout).
    """
    valid = ~is_pad.to(torch.bool)
    n_valid = valid.sum()
    if int(n_valid) == 0:
        raise ValueError("cannot compute diffusion loss on an all-padded batch")

    batch = actions.shape[0]
    draw_device = generator.device if generator is not None else actions.device
    t = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch,),
        generator=generator,
        device=draw_device,
    ).to(actions.device)
    noise = torch.randn(
        actions.shape, generator=generator, device=draw_device, dtype=actions.dtype
    ).to(actions.device)

    noisy = scheduler.add_noise(actions, noise, t)
    eps_hat = model(noisy, t, obs)
    per_step_mse = (eps_hat - noise).pow(2).mean(dim=-1)
    mse = (per_step_mse * valid).sum() / n_valid
    return {"mse": mse, "loss": mse}


__all__ = ["DiffusionTransformer", "diffusion_loss", "sinusoidal_table"]
