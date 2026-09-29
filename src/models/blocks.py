"""Time-conditioned residual blocks. Default algebra and parameter keys preserved."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.components import build_component
from .normalization import valid_num_groups, adaptive_group_norm

class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dilation=1, dropout=0.0, groups=8,
                 normalization=None, activation=None, convolution=None, time_projection=None,
                 kernel_size=3, skip=None):
        super().__init__()
        if dilation < 1 or kernel_size < 1 or kernel_size % 2 == 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid dilation/dropout")
        self.norm1 = build_component(normalization, adaptive_group_norm, channels=in_channels, groups=groups)
        self.conv1 = build_component(convolution, nn.Conv1d, in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, padding=dilation*(kernel_size//2), dilation=dilation)
        self.time_film = nn.Sequential(build_component(activation, nn.SiLU),
            build_component(time_projection, nn.Linear, in_features=time_dim, out_features=2*out_channels))
        self.norm2 = build_component(normalization, adaptive_group_norm, channels=out_channels, groups=groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = build_component(convolution, nn.Conv1d, in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, padding=dilation*(kernel_size//2), dilation=dilation)
        self.skip = (build_component(skip, nn.Conv1d, in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=1) if in_channels != out_channels else nn.Identity())
        self.activation1 = build_component(activation, nn.SiLU)
        self.activation2 = build_component(activation, nn.SiLU)

    def forward(self, x, time_embedding):
        h = self.conv1(self.activation1(self.norm1(x)))
        scale, shift = self.time_film(time_embedding).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[..., None]) + shift[..., None]
        h = self.conv2(self.dropout(self.activation2(h)))
        return (self.skip(x) + h) / math.sqrt(2)



class TimeIdentity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, x, time_embedding):
        return x
