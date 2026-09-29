"""EMA rule extracted without changing updates or checkpoint tensors."""
from __future__ import annotations
import torch


class ExponentialMovingAverage:
    @torch.no_grad()
    def __call__(self, online_model, ema_model, decay):
        ema_params = dict(ema_model.named_parameters())
        for name, online in online_model.named_parameters():
            if online.requires_grad:
                ema_params[name].mul_(decay).add_(online, alpha=1-decay)
            else:
                ema_params[name].copy_(online)
        ema_buffers = dict(ema_model.named_buffers())
        for name, buffer in online_model.named_buffers():
            ema_buffers[name].copy_(buffer)
