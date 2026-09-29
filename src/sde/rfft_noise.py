from __future__ import annotations

import math
import torch
from torch import Tensor


def project_rfft(x: Tensor, n_fft: int) -> Tensor:
    if not x.is_complex() or x.shape[-1] != n_fft // 2 + 1:
        raise ValueError("Expected complex rFFT tensor")
    idx = torch.arange(x.shape[-1], device=x.device)
    mask = idx != 0
    if n_fft % 2 == 0:
        mask = mask & (idx != x.shape[-1] - 1)
    return torch.complex(x.real, x.imag * mask.to(x.real.dtype))


def standard_rfft_noise_like(x: Tensor, *, n_fft: int,
                             generator: torch.Generator | None = None) -> Tensor:
    """CN(0,1) interior bins; real N(0,1) at DC and even-length Nyquist."""
    if not x.is_complex() or x.ndim != 3 or x.shape[-1] != n_fft // 2 + 1:
        raise ValueError("Expected complex [B,K,n_fft//2+1]")
    re = torch.randn(x.shape, device=x.device, dtype=x.real.dtype, generator=generator)
    im = torch.randn(x.shape, device=x.device, dtype=x.real.dtype, generator=generator)
    idx = torch.arange(x.shape[-1], device=x.device)
    endpoints = idx == 0
    if n_fft % 2 == 0:
        endpoints = endpoints | (idx == x.shape[-1] - 1)
    real_scale = torch.where(endpoints, 1.0, 1 / math.sqrt(2)).to(x.real.dtype)
    imag_scale = (~endpoints).to(x.real.dtype) / math.sqrt(2)
    return torch.complex(re * real_scale, im * imag_scale)
