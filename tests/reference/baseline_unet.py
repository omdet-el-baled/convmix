"""The enhanced ScoreUNet from the conversation, with a fixed time embedding.

No time-embedding MLP/learnable frequencies. FiLM projections are trainable:
they are feature-conditioning weights, not part of the fixed embedding.
The output uses out-of-place endpoint masking; odd FFT lengths are supported.
"""
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


def valid_num_groups(channels, requested_groups=8):
    if channels < 1 or requested_groups < 1:
        raise ValueError("GroupNorm dimensions must be positive")
    return next(g for g in range(min(channels, requested_groups), 0, -1) if channels % g == 0)


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dilation=1, dropout=0.0, groups=8):
        super().__init__()
        if dilation < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid dilation/dropout")
        self.norm1 = nn.GroupNorm(valid_num_groups(in_channels, groups), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.time_film = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * out_channels))
        self.norm2 = nn.GroupNorm(valid_num_groups(out_channels, groups), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_film(time_embedding).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None]) + shift[..., None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (self.skip(x) + h) / math.sqrt(2)


class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4, groups=8):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError("channels must be divisible by positive num_heads")
        self.norm = nn.GroupNorm(valid_num_groups(channels, groups), channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.output = nn.Linear(channels, channels)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attention(h, h, h, need_weights=False)
        return (x + self.output(h).transpose(1, 2)) / math.sqrt(2)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x, target_length):
        return self.conv(F.interpolate(x, size=target_length, mode="nearest"))


class ScoreUNet(nn.Module):
    def __init__(self, num_sources, base_channels=64, time_dim=256, num_heads=4,
                 dropout=0.05, *, n_fft=None):
        super().__init__()
        if num_sources < 1 or base_channels < 1:
            raise ValueError("num_sources/base_channels must be positive")
        self.num_sources, self.time_dim, self.n_fft = int(num_sources), int(time_dim), n_fft
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels
        self.input_conv = nn.Conv1d(3 * num_sources + 3, c1, 5, padding=2)
        def block(ci, co, dilation=1):
            return FiLMResidualBlock(ci, co, time_dim, dilation, dropout)
        self.enc1a, self.enc1b, self.down1 = block(c1, c1), block(c1, c1, 2), Downsample(c1)
        self.enc2a, self.enc2b, self.down2 = block(c1, c2), block(c2, c2, 2), Downsample(c2)
        self.enc3a, self.enc3b, self.down3 = block(c2, c3), block(c3, c3, 4), Downsample(c3)
        self.mid1, self.mid2 = block(c3, c3), block(c3, c3, 2)
        self.attention = SelfAttention1D(c3, num_heads)
        self.mid3 = block(c3, c3, 4)
        self.up3, self.dec3a, self.dec3b = Upsample(c3), block(2 * c3, c3), block(c3, c3, 2)
        self.up2, self.dec2a, self.dec2b = Upsample(c3), block(c3 + c2, c2), block(c2, c2, 2)
        self.up1, self.dec1a, self.dec1b = Upsample(c2), block(c2 + c1, c1), block(c1, c1)
        self.output_norm = nn.GroupNorm(valid_num_groups(c1), c1)
        self.output_conv = nn.Conv1d(c1, 2 * num_sources, 3, padding=1)
        nn.init.zeros_(self.output_conv.bias)
        nn.init.normal_(self.output_conv.weight, std=1e-3)

    def _prepare_input(self, x, y):
        if x.ndim != 3 or y.ndim != 2 or y.shape != (x.shape[0], x.shape[-1]):
            raise ValueError("Expected x [B,K,F], y [B,F]")
        if x.shape[1] != self.num_sources or not x.is_complex() or not y.is_complex():
            raise ValueError("Unexpected source count or non-complex inputs")
        if x.dtype != y.dtype or x.device != y.device:
            raise ValueError("x/y must have matching dtype and device")
        if self.n_fft is not None and x.shape[-1] != self.n_fft // 2 + 1:
            raise ValueError("Frequency dimension does not match n_fft")
        return torch.cat([x.real, x.imag, torch.log1p(x.abs()),
            torch.stack([y.real, y.imag, torch.log1p(y.abs())], 1)], 1)

    def _constrain_rfft_imaginary(self, imag):
        index = torch.arange(imag.shape[-1], device=imag.device)
        mask = index != 0
        # None keeps the old even-n_fft behavior for old construction calls.
        if self.n_fft is None or self.n_fft % 2 == 0:
            mask = mask & (index != imag.shape[-1] - 1)
        return imag * mask.to(imag.dtype)[None, None]

    def forward(self, x, y, t):
        features = self._prepare_input(x, y)
        if t.shape != (x.shape[0],) or t.device != x.device:
            raise ValueError("t must be [B] on the same device as x")
        original_length = x.shape[-1]
        # Three strided convolutions need at least 8 bins. Support tiny unit
        # tests explicitly rather than failing in Conv1d. Real runs have 1007.
        if original_length < 8:
            features = F.pad(features, (0, 8 - original_length))
        te = self.time_embedding(t.to(self.time_embedding.frequencies.dtype))
        h = self.input_conv(features)
        h1 = self.enc1b(self.enc1a(h, te), te)
        h2 = self.enc2b(self.enc2a(self.down1(h1), te), te)
        h3 = self.enc3b(self.enc3a(self.down2(h2), te), te)
        h = self.mid2(self.mid1(self.down3(h3), te), te)
        h = self.mid3(self.attention(h), te)
        h = self.dec3b(self.dec3a(torch.cat([self.up3(h, h3.shape[-1]), h3], 1), te), te)
        h = self.dec2b(self.dec2a(torch.cat([self.up2(h, h2.shape[-1]), h2], 1), te), te)
        h = self.dec1b(self.dec1a(torch.cat([self.up1(h, h1.shape[-1]), h1], 1), te), te)
        output = self.output_conv(F.silu(self.output_norm(h)))[..., :original_length]
        # torch.complex does not accept bf16. Keep SDE/score output in the input
        # real dtype even when only the network runs with bf16 autocast.
        real, imag = output.to(x.real.dtype).chunk(2, dim=1)
        return torch.complex(real, self._constrain_rfft_imaginary(imag))


class BasicScoreUNet(ScoreUNet):
    """Lightweight controlled architecture ablation, NOT an exact old checkpoint replica.

    Same fixed embedding, input features, FiLM and bottleneck attention. Disable
    the second block at each resolution and two extra bottleneck blocks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("enc1b", "enc2b", "enc3b", "mid2", "mid3", "dec3b", "dec2b", "dec1b"):
            setattr(self, name, _TimeIdentity())


class _TimeIdentity(nn.Module):
    def forward(self, x, time_embedding):
        return x
