"""Swappable resolution changes; default operations/parameter names unchanged."""
from torch import nn
import torch.nn.functional as F
from src.components import build_component


class Downsample(nn.Module):
    def __init__(self, channels, kernel_size=4, stride=2, padding=1, convolution=None):
        super().__init__()
        self.conv = build_component(convolution, nn.Conv1d, in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, mode='nearest', convolution=None):
        super().__init__()
        if mode not in ('nearest','nearest-exact','linear'):
            raise ValueError('Unsupported 1D interpolation mode')
        self.mode = mode
        self.conv = build_component(convolution, nn.Conv1d, in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, target_length):
        kwargs = {'align_corners': False} if self.mode == 'linear' else {}
        return self.conv(F.interpolate(x, size=target_length, mode=self.mode, **kwargs))
