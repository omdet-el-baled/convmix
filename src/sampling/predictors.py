"""Interchangeable integration steps, with the same formulas as samplers.py."""
import math


class EulerProbabilityFlow:
    def __call__(self, x, current, following, ctx):
        return x + (following-current) * ctx.field(x, current, True)


class HeunProbabilityFlow:
    def __call__(self, x, current, following, ctx):
        dt = following-current
        k1 = ctx.field(x, current, True)
        k2 = ctx.field(x + dt*k1, following, True)
        return x + 0.5*dt*(k1+k2)


class RK4ProbabilityFlow:
    def __call__(self, x, current, following, ctx):
        dt = following-current
        middle = (current+following)/2
        k1 = ctx.field(x, current, True)
        k2 = ctx.field(x + 0.5*dt*k1, middle, True)
        k3 = ctx.field(x + 0.5*dt*k2, middle, True)
        k4 = ctx.field(x + dt*k3, following, True)
        return x + dt/6*(k1+2*k2+2*k3+k4)


class EulerMaruyama:
    def __call__(self, x, current, following, ctx):
        dt = following-current
        drift = ctx.field(x, current, False)
        t = ctx.y.real.new_full((ctx.B,), current)
        noise = ctx.sde.apply_B(ctx.noise_source(x).reshape(ctx.B, -1), t).reshape_as(x)
        return x + dt*drift + math.sqrt(-dt)*noise
