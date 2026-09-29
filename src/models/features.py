"""Interchangeable complex-to-real network feature transforms."""
from __future__ import annotations
import torch
from torch import nn


class SpectralFeatures(nn.Module):
    def __init__(self, num_sources: int, log_magnitude: bool = True):
        super().__init__()
        self.num_sources = num_sources
        self.log_magnitude = log_magnitude
        self.out_channels = (3 if log_magnitude else 2) * (num_sources + 1)

    def forward(self, x, y):
        if self.log_magnitude:
            return torch.cat([x.real, x.imag, torch.log1p(x.abs()),
                torch.stack([y.real, y.imag, torch.log1p(y.abs())], 1)], 1)
        return torch.cat([x.real, x.imag, torch.stack([y.real, y.imag], 1)], 1)
