"""Covariance-weighted DSM and terminal mismatch objective, unchanged algebra.

The class is a callable, not a LightningModule; it holds no network/SDE state.
Tests compare inputs, targets, residuals and gradients to the previous module.
"""
from __future__ import annotations
import torch
from src.components import build_component


class ComplexSquaredResidual:
    def __call__(self, residual):
        return residual.abs().square().mean(-1)


class CovarianceWeightedDSM:
    def __init__(self, reduction=None):
        self.reduction = build_component(reduction, ComplexSquaredResidual)

    def __call__(self, model, x0, y, t, *, mismatch_mask=None, use_ema=False,
                 generator=None, epsilon=None):
        model._check_xy(x0, y)
        B, K, _ = x0.shape
        if mismatch_mask is None:
            mismatch_mask = torch.zeros(B, dtype=torch.bool, device=x0.device)
        if mismatch_mask.shape != (B,):
            raise ValueError("mismatch_mask must be [B]")
        t = torch.where(mismatch_mask, torch.full_like(t, model.sde.T), t)
        mean, std, epsilon, perturbation = model._forward_components(x0, t, generator=generator, epsilon=epsilon)
        y_rep = model._repeat_observation(y, K).reshape(B, -1)
        shift = y_rep - mean
        # Only solve for selected examples. All factors are invertible at t>=eps.
        correction = torch.zeros_like(mean)
        if mismatch_mask.any():
            correction[mismatch_mask] = model.sde.apply_std_inv(
                std[mismatch_mask], shift[mismatch_mask])
        if model.mismatch_input == "terminal":
            center = torch.where(mismatch_mask[:, None], y_rep, mean)
        else:
            center = mean  # explicitly reproduces the earlier incorrect input-centering variant
        xt = (center + perturbation).reshape_as(x0)
        score = model(xt, y, t, use_ema=use_ema).reshape(B, -1)
        weighted_score = model.sde.apply_std_h(std, score)
        target_noise = epsilon.reshape(B, -1) + correction
        residual = weighted_score + target_noise
        per_example = self.reduction(residual)
        if not torch.isfinite(per_example).all():
            raise FloatingPointError("Nonfinite DSM loss; inspect SDE covariance, inputs, and score")
        return {"loss": per_example.mean(), "per_example": per_example,
                "time": t, "residual": residual, "target_noise": target_noise,
                "xt": xt, "mean": mean, "std": std, "epsilon": epsilon,
                "mismatch_mask": mismatch_mask}



class BernoulliSelection:
    def __call__(self, batch_size, device, probability, mode="batch"):
        # Same draw even when p=0, preserving the matched RNG policy.
        size = 1 if mode == "batch" else batch_size
        return (torch.rand(size, device=device) < probability).expand(batch_size)
