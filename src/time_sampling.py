"""Interchangeable time samplers, retaining the previous draws and equations."""
from __future__ import annotations
import math
import torch


class TimeDistribution:
    """Compatibility selector for direct construction / older YAML files."""
    def __init__(self, name=None):
        self.name = name

    def __call__(self, batch_size, device, dtype, *, t_eps, T, log_fraction=0.5,
                 generator=None, strategy='uniform'):
        u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        uniform = t_eps + (T - t_eps) * u
        if strategy == 'uniform':
            return uniform
        log_t = torch.exp(math.log(t_eps) + u * math.log(T / t_eps))
        if strategy == 'log_uniform':
            return log_t
        if strategy == 'mixed':
            choose = torch.rand(batch_size, device=device, generator=generator) < log_fraction
            return torch.where(choose, log_t, uniform)
        raise ValueError(f'Unknown strategy {strategy}')


class UniformTimeSampler(TimeDistribution):
    def __call__(self, *args, **kwargs):
        kwargs['strategy'] = 'uniform'
        return super().__call__(*args, **kwargs)


class LogUniformTimeSampler(TimeDistribution):
    def __call__(self, *args, **kwargs):
        kwargs['strategy'] = 'log_uniform'
        return super().__call__(*args, **kwargs)


class MixedTimeSampler(TimeDistribution):
    def __call__(self, *args, **kwargs):
        kwargs['strategy'] = 'mixed'
        return super().__call__(*args, **kwargs)
