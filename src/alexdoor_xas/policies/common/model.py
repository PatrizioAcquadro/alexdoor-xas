"""Shared neural-network building blocks."""

from __future__ import annotations

import math

import torch


def sinusoidal_table(n_positions: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
    )
    table = torch.zeros(n_positions, d_model)
    table[:, 0::2] = torch.sin(positions * frequencies)
    table[:, 1::2] = torch.cos(positions * frequencies[: d_model // 2])
    return table
