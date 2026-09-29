"""Fixed-time correction steps. Adaptive step sizes remain an explicit heuristic."""
import math
import torch
from src.sde.rfft_noise import project_rfft


class NoCorrector:
    def __call__(self, x, time, ctx):
        return x, 0


class LangevinCorrector:
    def __init__(self, num_steps=1, snr=0.16, geometry='diffusion',
                 step_size=None, max_step=1.0):
        if num_steps < 0 or int(num_steps) != num_steps or snr <= 0:
            raise ValueError('Invalid corrector step count or SNR')
        if geometry not in ('diffusion', 'identity'):
            raise ValueError('geometry must be diffusion or identity')
        for value in (step_size, max_step):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Corrector step sizes must be positive or null')
        self.num_steps, self.snr, self.geometry = int(num_steps), snr, geometry
        self.step_size, self.max_step = step_size, max_step

    def __call__(self, x, time, ctx):
        B, clipped = ctx.B, 0
        for _ in range(self.num_steps):
            score, t = ctx.network(x, time)
            z = ctx.noise_source(x).reshape(B, -1)
            if self.geometry == 'diffusion':
                direction = ctx.sde.apply_BBH(score.reshape(B, -1), t)
                noise = ctx.sde.apply_B(z, t)
            else:
                direction, noise = score.reshape(B, -1), z
            if self.step_size is None:
                dnorm = torch.linalg.vector_norm(direction, dim=-1)
                znorm = torch.linalg.vector_norm(noise, dim=-1)
                eta = 2 * (self.snr * znorm / dnorm.clamp_min(1e-12)).square()
                eta = torch.where(dnorm > 1e-12, eta, torch.zeros_like(eta))
            else:
                eta = ctx.y.real.new_full((B,), self.step_size)
            if self.max_step is not None:
                clipped += int((eta > self.max_step).sum())
                eta = eta.clamp_max(self.max_step)
            x = (x.reshape(B, -1) + eta[:, None] * direction
                 + (2 * eta).sqrt()[:, None] * noise).reshape_as(x)
            x = project_rfft(x, ctx.n_fft)
        return x, clipped
