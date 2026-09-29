"""Fixed time embeddings; the default is unchanged."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_size=256, max_period=10000.0):
        super().__init__()
        if embedding_size < 2 or embedding_size % 2 or max_period <= 1:
            raise ValueError("Require positive even embedding_size>=2 and max_period>1")
        exponent = torch.arange(embedding_size // 2, dtype=torch.float32) / max(embedding_size // 2 - 1, 1)
        self.register_buffer("frequencies", torch.exp(-math.log(max_period) * exponent), persistent=True)

    def forward(self, t: Tensor):
        if t.ndim != 1 or not t.is_floating_point():
            raise ValueError("t must be a real floating tensor [B]")
        angles = 2 * math.pi * t[:, None] * self.frequencies[None].to(t.dtype)
        return torch.cat([angles.sin(), angles.cos()], -1)



class GaussianFourierProjection(nn.Module):
    """Fixed, seeded random frequencies. No learnable embedding parameters."""
    def __init__(self, embedding_size=256, scale=16.0, seed=0):
        super().__init__()
        if embedding_size < 2 or embedding_size % 2 or scale <= 0:
            raise ValueError("Require even embedding_size>=2 and scale>0")
        generator = torch.Generator().manual_seed(int(seed))
        self.register_buffer("frequencies", torch.randn(embedding_size // 2, generator=generator) * scale)

    def forward(self, t):
        if t.ndim != 1 or not t.is_floating_point():
            raise ValueError("t must be a real floating tensor [B]")
        angles = 2 * math.pi * t[:, None] * self.frequencies[None].to(t.dtype)
        return torch.cat([angles.sin(), angles.cos()], -1)
