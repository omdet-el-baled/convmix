"""Component-driven sampler. No solver-name dispatch in the integration loop.

The default grid, initialization, step formulas, noise, and masking match the
previous sampler. New predictor/corrector classes only need the contracts below.
"""
from __future__ import annotations
import torch
from src.components import build_component
from src.sde.rfft_noise import project_rfft
from .samplers import PairedNoise, SamplerResult
from .predictors import EulerProbabilityFlow
from .correctors import NoCorrector
from .grids import LinearTimeGrid
from .initialization import RepeatedObservationInitialization


class SamplingContext:
    def __init__(self, model, sde, y, n_fft, t_eps, weights, noise_source):
        self.model, self.sde, self.y = model, sde, y
        self.B, self.n_fft, self.t_eps = y.shape[0], n_fft, t_eps
        self.weights, self.noise_source = weights, noise_source
        self.nfe, self.score_times = 0, []

    def network(self, state, time):
        if not self.t_eps - 1e-12 <= time <= self.sde.T + 1e-12:
            raise RuntimeError('Attempt to evaluate score outside [t_eps,T]')
        t = self.y.real.new_full((self.B,), time)
        output = self.model(state, self.y, t, use_ema=self.weights == 'ema')
        self.nfe += 1
        self.score_times.append(time)
        if output.shape != state.shape or not torch.isfinite(output).all():
            raise FloatingPointError('Score has invalid shape or nonfinite values')
        return output, t

    def field(self, state, time, probability_flow):
        score, t = self.network(state, time)
        drift = self.sde.apply_drift(state.reshape(self.B, -1), t)
        covariance_score = self.sde.apply_BBH(score.reshape(self.B, -1), t)
        return (drift - (0.5 if probability_flow else 1.0)*covariance_score).reshape_as(state)


class PredictorCorrectorSampler:
    def __init__(self, predictor=None, corrector=None, grid=None, initialization=None,
                 noise=None, num_steps=50, add_terminal_noise=True, weights='ema', seed=11001,
                 solver='euler'):
        if num_steps < 1 or int(num_steps) != num_steps or weights not in ('ema', 'online'):
            raise ValueError('Invalid sampler count/weights')
        self.predictor = build_component(predictor, EulerProbabilityFlow)
        self.corrector = build_component(corrector, NoCorrector)
        self.grid = build_component(grid, LinearTimeGrid)
        self.initialization = build_component(initialization, RepeatedObservationInitialization)
        self.noise = noise
        self.num_steps, self.add_terminal_noise = int(num_steps), bool(add_terminal_noise)
        self.weights, self.seed = weights, int(seed)
        self.solver = solver  # label only; the predictor/corrector objects define behavior

    @torch.no_grad()
    def __call__(self, model, sde, y, *, K, n_fft, t_eps, example_ids=None,
                 x_T=None, noise_source=None):
        if y.ndim != 2 or not y.is_complex() or y.shape[-1] != n_fft//2+1:
            raise ValueError('Expected complex observed y [B,F]')
        if not 0 < t_eps < sde.T or K*y.shape[-1] != sde.ndim:
            raise ValueError('Invalid endpoint/state dimensions')
        if getattr(model, 'training', False):
            raise RuntimeError('Call model.eval() before sampling')
        sde.send_to(y.device)
        B, F = y.shape
        ids = list(range(B)) if example_ids is None else example_ids
        if noise_source is None:
            noise_source = build_component(self.noise, PairedNoise,
                example_ids=ids, seed=self.seed, device=y.device, n_fft=n_fft)
        if x_T is None:
            x = self.initialization(sde, y, K, n_fft, self.add_terminal_noise, noise_source=noise_source)
        else:
            if x_T.shape != (B,K,F) or x_T.device != y.device or x_T.dtype != y.dtype:
                raise ValueError('x_T must match [B,K,F], dtype and device')
            x = x_T.clone()
        initial = x.clone()
        times = self.grid(sde.T, t_eps, self.num_steps)
        if times.shape != (self.num_steps+1,) or not torch.isfinite(times).all():
            raise ValueError('Grid must return num_steps+1 finite times')
        if float(times[0]) != float(sde.T) or float(times[-1]) != float(t_eps) or not (times[1:] < times[:-1]).all():
            raise ValueError('Grid must strictly decrease from T to exactly t_eps')
        ctx = SamplingContext(model, sde, y, n_fft, t_eps, self.weights, noise_source)
        clipped = 0
        for i in range(self.num_steps):
            current, following = float(times[i]), float(times[i+1])
            x, count = self.corrector(x, current, ctx)
            clipped += count
            x = self.predictor(x, current, following, ctx)
            x = project_rfft(x, n_fft)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f'Nonfinite state at interval {i}, t={following}')
        return SamplerResult(x, ctx.nfe, float(times[-1]), times, initial, ctx.score_times, clipped)
