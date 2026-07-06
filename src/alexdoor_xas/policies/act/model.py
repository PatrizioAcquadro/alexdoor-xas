"""State-only ACT model: a CVAE action-chunk transformer (Zhao et al. 2023).

Phase 2 episodes carry no images, so the paper's ResNet feature sequence is
replaced by a single projected observation token; everything else follows the
ACT recipe — CVAE encoder over (obs, action chunk) -> style variable z,
transformer encoder/decoder policy with fixed sinusoidal chunk queries, z = 0
at inference. No Isaac imports; torch only.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from alexdoor_xas.policies.act.config import ActModelCfg


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


class ACTModel(nn.Module):
    """CVAE action-chunk policy over a single state-observation token."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: ActModelCfg) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

        d = cfg.d_model
        self.obs_proj = nn.Linear(obs_dim, d)
        self.action_proj = nn.Linear(action_dim, d)

        # CVAE encoder (training only): [CLS, obs, a_1..a_H] -> z mean/logvar.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.register_buffer(
            "cvae_pos", sinusoidal_table(cfg.chunk_size + 2, d), persistent=False
        )
        self.cvae_encoder = nn.TransformerEncoder(
            self._encoder_layer(cfg), num_layers=cfg.cvae_encoder_layers
        )
        self.latent_head = nn.Linear(d, 2 * cfg.z_dim)

        # Policy decoder: memory [obs, z] -> chunk queries -> (H, action_dim).
        self.z_proj = nn.Linear(cfg.z_dim, d)
        self.memory_pos = nn.Parameter(torch.zeros(2, d))
        self.encoder = nn.TransformerEncoder(
            self._encoder_layer(cfg), num_layers=cfg.encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.decoder_layers,
        )
        self.register_buffer(
            "query_pos", sinusoidal_table(cfg.chunk_size, d), persistent=False
        )
        self.action_head = nn.Linear(d, action_dim)

    @staticmethod
    def _encoder_layer(cfg: ActModelCfg) -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(
        self, obs: torch.Tensor, actions: torch.Tensor, is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CVAE encoder: ``(B, obs_dim), (B, H, A), (B, H)`` -> ``(mu, logvar)``."""
        batch = obs.shape[0]
        if actions.shape != (batch, self.cfg.chunk_size, self.action_dim):
            raise ValueError(
                f"expected actions of shape {(batch, self.cfg.chunk_size, self.action_dim)}, "
                f"got {tuple(actions.shape)}"
            )
        cls = self.cls_token.expand(batch, -1, -1)
        obs_token = self.obs_proj(obs).unsqueeze(1)
        action_tokens = self.action_proj(actions)
        sequence = torch.cat([cls, obs_token, action_tokens], dim=1) + self.cvae_pos
        prefix_mask = torch.zeros(batch, 2, dtype=torch.bool, device=obs.device)
        padding_mask = torch.cat([prefix_mask, is_pad.to(torch.bool)], dim=1)
        encoded = self.cvae_encoder(sequence, src_key_padding_mask=padding_mask)
        mu, logvar = self.latent_head(encoded[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Policy decoder: ``(B, obs_dim), (B, z_dim)`` -> ``(B, H, action_dim)``."""
        obs_token = self.obs_proj(obs).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([obs_token, z_token], dim=1) + self.memory_pos
        memory = self.encoder(memory)
        queries = self.query_pos.unsqueeze(0).expand(obs.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training pass: returns ``(a_hat, mu, logvar)`` with reparameterized z."""
        mu, logvar = self.encode(obs, actions, is_pad)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(obs, z), mu, logvar

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference pass with z = 0 (prior mean): ``(B, obs_dim) -> (B, H, A)``."""
        z = torch.zeros(obs.shape[0], self.cfg.z_dim, dtype=obs.dtype, device=obs.device)
        return self.decode(obs, z)


def act_loss(
    a_hat: torch.Tensor,
    actions: torch.Tensor,
    is_pad: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    """Masked-L1 reconstruction + beta-weighted KL, per the ACT objective."""
    valid = ~is_pad.to(torch.bool)
    n_valid = valid.sum()
    if int(n_valid) == 0:
        raise ValueError("cannot compute ACT loss on an all-padded batch")
    per_step_l1 = (a_hat - actions).abs().mean(dim=-1)
    l1 = (per_step_l1 * valid).sum() / n_valid
    kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()
    return {"l1": l1, "kl": kl, "loss": l1 + kl_weight * kl}


__all__ = ["ACTModel", "act_loss", "sinusoidal_table"]
