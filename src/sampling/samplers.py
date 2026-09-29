"""Single source of truth for all reverse samplers. No hidden endpoint denoise.

Euler/Heun/RK4 solve the PF-ODE; EM and PC solve the reverse SDE.
All time derivatives use original forward time t; the numerical dt is negative.
Langevin corrector uses the same complex/real-endpoint convention as DSM.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor

from src.sde.rfft_noise import standard_rfft_noise_like, project_rfft


@dataclass(frozen=True)
class SamplingConfig:
    solver: str = "euler"
    num_steps: int = 50
    add_terminal_noise: bool = True
    corrector_steps: int = 1
    corrector_snr: float = 0.16
    corrector_geometry: str = "diffusion"
    corrector_step_size: float | None = None
    corrector_max_step: float | None = 1.0
    weights: str = "ema"
    seed: int = 11001

    def __post_init__(self):
        if self.solver not in {"euler", "heun", "rk4", "em", "pc"}:
            raise ValueError("Unknown solver")
        if self.num_steps < 1 or self.corrector_steps < 0 or self.corrector_snr <= 0:
            raise ValueError("Invalid sampler counts or SNR")
        if self.corrector_geometry not in {"diffusion", "identity"}:
            raise ValueError("corrector_geometry must be diffusion or identity")
        if self.weights not in {"online", "ema"}:
            raise ValueError("weights must be online or ema")
        for value in (self.corrector_step_size, self.corrector_max_step):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Corrector step sizes must be positive or null")


@dataclass
class SamplerResult:
    x: Tensor
    nfe: int
    final_time: float
    times: Tensor
    initial_x: Tensor
    score_times: list[float]
    corrector_clipped_steps: int = 0


class PairedNoise:
    """One independent generator per example; invariant to evaluation batch size.

    ODE comparisons share the exact x_T. Stochastic solvers share initial noise,
    not a Brownian path under different grids; do not claim strong path coupling.
    """
    def __init__(self, example_ids, seed, device, n_fft):
        self.n_fft = n_fft
        self.generators = [torch.Generator(device=device).manual_seed(
            (int(seed) + 1000003 * int(index)) % (2**63 - 1)) for index in example_ids]

    def __call__(self, x):
        if len(self.generators) != x.shape[0]:
            raise ValueError("Noise source example count mismatch")
        return torch.cat([standard_rfft_noise_like(x[i:i+1], n_fft=self.n_fft, generator=g)
                          for i, g in enumerate(self.generators)], dim=0)


def apply_drift(sde, x, t):
    return sde.apply_drift(x, t)


def diffusion_g2(sde, t):
    return sde.g_squared(t)


def apply_diffusion_matrix(sde, noise, t):
    return sde.apply_B(noise, t)


def apply_diffusion_covariance(sde, score, t):
    return sde.apply_BBH(score, t)


@torch.no_grad()
def initialize_terminal_state(sde, y, K, n_fft, add_noise=True, *, noise_source=None):
    """Approximate terminal law y_rep + S_T z, NOT the exact finite-T conditional law."""
    B, F = y.shape
    sde.send_to(y.device)
    mean = y[:, None, :].expand(B, K, F).clone()
    if not add_noise:
        return mean
    noise_source = noise_source or (lambda x: standard_rfft_noise_like(x, n_fft=n_fft))
    t = y.real.new_full((B,), sde.T)
    std = sde.fast_factor(t)
    perturbation = sde.apply_std(std, noise_source(mean).reshape(B, -1))
    return project_rfft(mean + perturbation.reshape_as(mean), n_fft)


@torch.no_grad()
def sample(model, sde, y: Tensor, *, K: int, n_fft: int, t_eps: float,
           config: SamplingConfig, example_ids=None, x_T: Tensor | None = None,
           noise_source: Callable | None = None) -> SamplerResult:
    if y.ndim != 2 or not y.is_complex() or y.shape[-1] != n_fft // 2 + 1:
        raise ValueError("Expected complex observed y [B,F]")
    if not 0 < t_eps < sde.T or K * y.shape[-1] != sde.ndim:
        raise ValueError("Invalid endpoint/state dimensions")
    if getattr(model, "training", False):
        raise RuntimeError("Call model.eval() before sampling (especially with dropout)")
    sde.send_to(y.device)
    B, F = y.shape
    if example_ids is None:
        example_ids = list(range(B))
    noise_source = noise_source or PairedNoise(example_ids, config.seed, y.device, n_fft)
    if x_T is None:
        x = initialize_terminal_state(sde, y, K, n_fft, config.add_terminal_noise,
                                     noise_source=noise_source)
    else:
        if x_T.shape != (B, K, F) or x_T.device != y.device or x_T.dtype != y.dtype:
            raise ValueError("x_T must match [B,K,F], dtype and device")
        x = x_T.clone()
    initial = x.clone()
    # Host double grid: exact shared endpoints; no accidental interval beyond eps.
    times = torch.linspace(float(sde.T), float(t_eps), config.num_steps + 1, dtype=torch.float64)
    times[0], times[-1] = float(sde.T), float(t_eps)
    score_times = []
    nfe = 0
    clipped = 0

    def network(state, time):
        nonlocal nfe
        if not t_eps - 1e-12 <= time <= sde.T + 1e-12:
            raise RuntimeError("Attempt to evaluate score outside [t_eps,T]")
        t = y.real.new_full((B,), time)
        output = model(state, y, t, use_ema=config.weights == "ema")
        nfe += 1
        score_times.append(time)
        if output.shape != state.shape or not torch.isfinite(output).all():
            raise FloatingPointError("Score has invalid shape or nonfinite values")
        return output, t

    def field(state, time, probability_flow):
        score, t = network(state, time)
        drift = sde.apply_drift(state.reshape(B, -1), t)
        covariance_score = sde.apply_BBH(score.reshape(B, -1), t)
        return (drift - (0.5 if probability_flow else 1.0) * covariance_score).reshape_as(state)

    for i in range(config.num_steps):
        current, following = float(times[i]), float(times[i+1])
        dt = following - current
        if config.solver == "pc":
            for _ in range(config.corrector_steps):
                score, t = network(x, current)
                z = noise_source(x).reshape(B, -1)
                if config.corrector_geometry == "diffusion":
                    direction = sde.apply_BBH(score.reshape(B, -1), t)
                    noise = sde.apply_B(z, t)
                else:
                    direction, noise = score.reshape(B, -1), z
                if config.corrector_step_size is None:
                    dnorm = torch.linalg.vector_norm(direction, dim=-1)
                    znorm = torch.linalg.vector_norm(noise, dim=-1)
                    eta = 2 * (config.corrector_snr * znorm / dnorm.clamp_min(1e-12)).square()
                    # A zero/near-zero score cannot define a meaningful adaptive
                    # SNR step. Skip rather than inject unbounded pure noise.
                    eta = torch.where(dnorm > 1e-12, eta, torch.zeros_like(eta))
                else:
                    eta = y.real.new_full((B,), config.corrector_step_size)
                if config.corrector_max_step is not None:
                    clipped += int((eta > config.corrector_max_step).sum())
                    eta = eta.clamp_max(config.corrector_max_step)
                x = (x.reshape(B, -1) + eta[:, None] * direction
                     + (2 * eta).sqrt()[:, None] * noise).reshape_as(x)
                x = project_rfft(x, n_fft)

        if config.solver == "euler":
            x = x + dt * field(x, current, True)
        elif config.solver == "heun":
            k1 = field(x, current, True)
            k2 = field(x + dt * k1, following, True)
            x = x + 0.5 * dt * (k1 + k2)
        elif config.solver == "rk4":
            middle = (current + following) / 2
            k1 = field(x, current, True)
            k2 = field(x + 0.5 * dt * k1, middle, True)
            k3 = field(x + 0.5 * dt * k2, middle, True)
            k4 = field(x + dt * k3, following, True)
            x = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        else:  # EM, including the PC predictor
            drift = field(x, current, False)
            t = y.real.new_full((B,), current)
            noise = sde.apply_B(noise_source(x).reshape(B, -1), t).reshape_as(x)
            x = x + dt * drift + math.sqrt(-dt) * noise
        x = project_rfft(x, n_fft)
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite state at interval {i}, t={following}")
    return SamplerResult(x, nfe, float(times[-1]), times, initial, score_times, clipped)


# Compatibility names used by the previous evaluate_test_set.py.
def _wrapper(solver, model, sde, y, *, K, n_fft, num_steps, t_eps,
             add_terminal_noise=True, **kwargs):
    config = SamplingConfig(solver=solver, num_steps=num_steps,
                            add_terminal_noise=add_terminal_noise, **kwargs)
    return sample(model, sde, y, K=K, n_fft=n_fft, t_eps=t_eps, config=config).x


def sample_euler(model, sde, y, **kwargs):
    return _wrapper("euler", model, sde, y, **kwargs)


def sample_heun(model, sde, y, **kwargs):
    return _wrapper("heun", model, sde, y, **kwargs)


def sample_pc(model, sde, y, *, preconditioned_corrector=True, **kwargs):
    kwargs["corrector_geometry"] = "diffusion" if preconditioned_corrector else "identity"
    return _wrapper("pc", model, sde, y, **kwargs)
