"""Self-attention with configurable internal components; default graph unchanged."""
import math
from torch import nn
from src.components import build_component
from .normalization import adaptive_group_norm


class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4, groups=8, normalization=None,
                 attention=None, output_projection=None):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError('channels must be divisible by positive num_heads')
        self.norm = build_component(normalization, adaptive_group_norm, channels=channels, groups=groups)
        self.attention = build_component(attention, nn.MultiheadAttention,
            embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.output = build_component(output_projection, nn.Linear, in_features=channels, out_features=channels)

    def forward(self, x):
        h = self.norm(x).transpose(1,2)
        h,_ = self.attention(h,h,h,need_weights=False)
        h = self.output(h).transpose(1,2)
        return (x+h)/math.sqrt(2)


class IdentityAttention(nn.Module):
    def __init__(self, channels, num_heads=4, groups=8):
        super().__init__()
    def forward(self, x):
        return x
