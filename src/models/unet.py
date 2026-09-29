"""Configurable ScoreUNet with the latest network as the exact default graph.

Swap the entire class or any component through _target_ YAML. The default
module names, weight shapes, initialization order, and numerical operations
match the previously packaged ScoreUNet (see regression tests).
"""
from __future__ import annotations

import string
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.components import build_component
from .embeddings import SinusoidalTimeEmbedding, GaussianFourierProjection
from .normalization import valid_num_groups, adaptive_group_norm
from .blocks import FiLMResidualBlock, TimeIdentity
from .attention import SelfAttention1D
from .resampling import Downsample, Upsample
from .features import SpectralFeatures


class ScoreUNet(nn.Module):
    def __init__(self, num_sources, base_channels=64, time_dim=256, num_heads=4,
                 dropout=0.05, *, n_fft=None,
                 channel_multipliers=(1, 2, 4),
                 encoder_dilations=((1, 2), (1, 2), (1, 4)),
                 decoder_dilations=((1, 1), (1, 2), (1, 2)),
                 bottleneck_dilations=(1, 2, 4), attention_after=2,
                 time_embedding=None, block=None, attention=None,
                 downsample=None, upsample=None, features=None,
                 input_projection=None, output_projection=None,
                 output_norm=None, output_activation=None,
                 output_init_std=1e-3):
        super().__init__()
        if num_sources < 1 or base_channels < 1 or time_dim < 2 or time_dim % 2:
            raise ValueError('Invalid source count, width, or time dimension')
        if not channel_multipliers or any(int(m) != m or m < 1 for m in channel_multipliers):
            raise ValueError('channel_multipliers must be positive integers')
        depth = len(channel_multipliers)
        if len(encoder_dilations) != depth or len(decoder_dilations) != depth:
            raise ValueError('Supply one encoder_dilations and decoder_dilations list per resolution')
        for values in (*encoder_dilations, *decoder_dilations, bottleneck_dilations):
            if isinstance(values, int):
                values = (values,)
            if not values or any(int(d) != d or d < 1 for d in values):
                raise ValueError('Dilations must be nonempty positive-integer lists')
        if not 0 <= attention_after <= len(bottleneck_dilations):
            raise ValueError('attention_after must be in [0, number of bottleneck blocks]')
        if output_init_std is not None and output_init_std < 0:
            raise ValueError('output_init_std must be nonnegative')
        self.num_sources, self.time_dim, self.n_fft = int(num_sources), int(time_dim), n_fft
        self.depth, self.attention_after = depth, int(attention_after)
        self.encoder_names, self.decoder_names = [], []
        self.middle_names = []
        self.time_embedding = build_component(time_embedding, SinusoidalTimeEmbedding, embedding_size=time_dim)
        if any(p.requires_grad for p in self.time_embedding.parameters()):
            raise ValueError('The time embedding must be fixed, not learnable. FiLM projections may learn.')
        self.features = build_component(features, SpectralFeatures, num_sources=num_sources)
        if not hasattr(self.features, 'out_channels'):
            raise TypeError('Feature builder must expose out_channels')
        widths = [int(base_channels * m) for m in channel_multipliers]
        self.input_conv = build_component(input_projection,
            lambda **kw: nn.Conv1d(**kw, kernel_size=5, padding=2),
            in_channels=self.features.out_channels, out_channels=widths[0])

        def residual(ci, co, dilation):
            return build_component(block, FiLMResidualBlock, in_channels=ci,
                                   out_channels=co, time_dim=time_dim,
                                   dilation=int(dilation), dropout=dropout)

        def suffix(i):
            return string.ascii_lowercase[i] if i < 26 else '_' + str(i)

        current = widths[0]
        for level, (width, dilations) in enumerate(zip(widths, encoder_dilations), 1):
            names = []
            for j, dilation in enumerate(dilations):
                name = f'enc{level}{suffix(j)}'
                self.add_module(name, residual(current, width, dilation))
                names.append(name)
                current = width
            self.encoder_names.append(names)
            self.add_module(f'down{level}', build_component(downsample, Downsample, channels=width))

        for j in range(len(bottleneck_dilations) + 1):
            if j == attention_after:
                self.attention = build_component(attention, SelfAttention1D,
                                                  channels=current, num_heads=num_heads)
            if j < len(bottleneck_dilations):
                name = f'mid{j+1}'
                self.add_module(name, residual(current, current, bottleneck_dilations[j]))
                self.middle_names.append(name)

        for level in range(depth, 0, -1):
            width = widths[level-1]
            self.add_module(f'up{level}', build_component(upsample, Upsample, channels=current))
            names = []
            for j, dilation in enumerate(decoder_dilations[level-1]):
                name = f'dec{level}{suffix(j)}'
                self.add_module(name, residual(current + width if j == 0 else current, width, dilation))
                names.append(name)
                current = width
            self.decoder_names.append(names)
        self.output_norm = build_component(output_norm, adaptive_group_norm, channels=current)
        self.output_conv = build_component(output_projection,
            lambda **kw: nn.Conv1d(**kw, kernel_size=3, padding=1),
            in_channels=current, out_channels=2*num_sources)
        self.output_activation = build_component(output_activation, nn.SiLU)
        # Same default initialization as the latest packaged network.
        if output_init_std is not None:
            if not hasattr(self.output_conv, 'weight'):
                raise TypeError('output_projection must expose weight, or set output_init_std=null')
            if getattr(self.output_conv, 'bias', None) is not None:
                nn.init.zeros_(self.output_conv.bias)
            nn.init.normal_(self.output_conv.weight, std=output_init_std)

    def _prepare_input(self, x, y):
        if x.ndim != 3 or y.ndim != 2 or y.shape != (x.shape[0], x.shape[-1]):
            raise ValueError('Expected x [B,K,F], y [B,F]')
        if x.shape[1] != self.num_sources or not x.is_complex() or not y.is_complex():
            raise ValueError('Unexpected source count or non-complex inputs')
        if x.dtype != y.dtype or x.device != y.device:
            raise ValueError('x/y must have matching dtype and device')
        if self.n_fft is not None and x.shape[-1] != self.n_fft // 2 + 1:
            raise ValueError('Frequency dimension does not match n_fft')
        return self.features(x, y)

    def _constrain_rfft_imaginary(self, imag):
        index = torch.arange(imag.shape[-1], device=imag.device)
        mask = index != 0
        if self.n_fft is None or self.n_fft % 2 == 0:
            mask = mask & (index != imag.shape[-1] - 1)
        return imag * mask.to(imag.dtype)[None, None]

    def forward(self, x, y, t):
        features = self._prepare_input(x, y)
        if t.shape != (x.shape[0],) or t.device != x.device:
            raise ValueError('t must be [B] on the same device as x')
        original_length = x.shape[-1]
        minimum = 2**self.depth
        if original_length < minimum:
            features = F.pad(features, (0, minimum - original_length))
        te = self.time_embedding(t.to(features.dtype))
        if te.shape != (x.shape[0], self.time_dim):
            raise ValueError('Time embedding must return [B,time_dim]')
        h = self.input_conv(features)
        skips = []
        for level, names in enumerate(self.encoder_names, 1):
            for name in names:
                h = getattr(self, name)(h, te)
            skips.append(h)
            h = getattr(self, f'down{level}')(h)
        for j in range(len(self.middle_names) + 1):
            if j == self.attention_after:
                h = self.attention(h)
            if j < len(self.middle_names):
                h = getattr(self, self.middle_names[j])(h, te)
        for level, names in zip(range(self.depth, 0, -1), self.decoder_names):
            skip = skips[level-1]
            h = getattr(self, f'up{level}')(h, target_length=skip.shape[-1])
            h = torch.cat([h, skip], 1)
            for name in names:
                h = getattr(self, name)(h, te)
        output = self.output_conv(self.output_activation(self.output_norm(h)))[..., :original_length]
        if output.shape != (x.shape[0], 2*self.num_sources, original_length):
            raise ValueError('Configured output head must return [B,2K,F] with the original length')
        real, imag = output.to(x.real.dtype).chunk(2, dim=1)
        return torch.complex(real, self._constrain_rfft_imaginary(imag))


class BasicScoreUNet(ScoreUNet):
    """The same lighter ablation as the previous ZIP; keys/behavior are preserved."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ('enc1b','enc2b','enc3b','mid2','mid3','dec3b','dec2b','dec1b'):
            if hasattr(self, name):
                setattr(self, name, TimeIdentity())


_TimeIdentity = TimeIdentity  # compatibility with the earlier module
