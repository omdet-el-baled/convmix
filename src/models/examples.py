"""A tiny alternative architecture demonstrating the external component contract.

This is an integration example, not a recommended research baseline.
Copy this file to your own importable package and select its _target_ in YAML.
"""
import torch
from torch import nn
from .embeddings import SinusoidalTimeEmbedding
from src.sde.rfft_noise import project_rfft


class TinyScoreNet(nn.Module):
    def __init__(self, num_sources, n_fft, channels=16, time_dim=32):
        super().__init__()
        self.num_sources, self.n_fft = num_sources, n_fft
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input = nn.Conv1d(2*num_sources+2, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, channels)
        self.output = nn.Conv1d(channels, 2*num_sources, 3, padding=1)

    def forward(self, x, y, t):
        features = torch.cat([x.real,x.imag,torch.stack([y.real,y.imag],1)],1)
        h = self.input(features) + self.time_projection(self.time_embedding(t))[...,None]
        real, imag = self.output(torch.nn.functional.silu(h)).to(x.real.dtype).chunk(2,1)
        return project_rfft(torch.complex(real,imag), self.n_fft)
