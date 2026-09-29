"""Shared endpoints are exact host-double values. No integration below t_eps."""
import torch


class LinearTimeGrid:
    def __call__(self, T, t_eps, num_steps):
        times = torch.linspace(float(T), float(t_eps), num_steps+1, dtype=torch.float64)
        times[0], times[-1] = float(T), float(t_eps)
        return times


class LogTimeGrid:
    """An explicit new grid ablation, not the default numerical method."""
    def __call__(self, T, t_eps, num_steps):
        times = torch.exp(torch.linspace(torch.tensor(float(T)).double().log(),
            torch.tensor(float(t_eps)).double().log(), num_steps+1, dtype=torch.float64))
        times[0], times[-1] = float(T), float(t_eps)
        return times
