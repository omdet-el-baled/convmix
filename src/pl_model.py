"""Lightning module: ordinary DSM, optional terminal mismatch branch, EMA.

Residual: S_T^H score + epsilon + S_T^{-1}(y_rep - mu_T).
The terminal branch uses INPUT y_rep + S_T epsilon. The prior message's
mu_T-centred input is available explicitly as `forward_legacy`; see CHANGELOG.
"""
from __future__ import annotations

import copy
import math
import random

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from src.sde.freqsde import ConvMixSDE
from src.sde.rfft_noise import standard_rfft_noise_like
from src.components import build_component
from src.losses import CovarianceWeightedDSM, BernoulliSelection
from src.time_sampling import TimeDistribution
from src.ema import ExponentialMovingAverage


class DCSModel(pl.LightningModule):
    def __init__(self, score_model: nn.Module, sde: ConvMixSDE, *, n_fft: int,
                 learning_rate=1e-4, weight_decay=1e-4, t_eps=0.005,
                 time_sampling_strategy="uniform", ema_decay=0.999,
                 mismatch_probability=0.0, mismatch_input="terminal",
                 mismatch_selection="batch", time_sampling_log_fraction=0.5,
                 validation_seed=11001, optimizer=None, scheduler=None, loss=None,
                 time_sampler=None, validation_time_sampler=None, noise_sampler=None,
                 ema_update=None, loss_selection=None):
        super().__init__()
        self.save_hyperparameters(ignore=["score_model", "sde", "optimizer", "scheduler", "loss",
            "time_sampler", "validation_time_sampler", "noise_sampler", "ema_update", "loss_selection"])
        if n_fft < 1 or learning_rate <= 0 or weight_decay < 0 or not 0 < t_eps < sde.T:
            raise ValueError("Invalid FFT/optimization/time parameters")
        if time_sampler is None and time_sampling_strategy not in {"uniform", "log_uniform", "mixed"}:
            raise ValueError("Unknown time sampling strategy")
        if ema_decay is not None and not 0 < ema_decay < 1:
            raise ValueError("ema_decay must be None or in (0,1)")
        if not 0 <= mismatch_probability <= 1 or not 0 <= time_sampling_log_fraction <= 1:
            raise ValueError("Probabilities must lie in [0,1]")
        if mismatch_input not in {"terminal", "forward_legacy"}:
            raise ValueError("mismatch_input must be terminal or forward_legacy")
        if mismatch_selection not in {"batch", "per_sample"}:
            raise ValueError("mismatch_selection must be batch or per_sample")
        self.score_model, self.sde = score_model, sde
        self.objective = build_component(loss, CovarianceWeightedDSM)
        self.time_sampler = build_component(time_sampler, TimeDistribution)
        self.validation_time_sampler = build_component(validation_time_sampler, TimeDistribution)
        self.noise_sampler = standard_rfft_noise_like if noise_sampler is None else noise_sampler
        self.ema_update_rule = build_component(ema_update, ExponentialMovingAverage)
        self.loss_selection_rule = build_component(loss_selection, BernoulliSelection)
        self.optimizer_factory = optimizer
        self.scheduler_config = scheduler
        self.n_fft, self.t_eps = int(n_fft), float(t_eps)
        self.learning_rate, self.weight_decay = float(learning_rate), float(weight_decay)
        self.time_sampling_strategy = time_sampling_strategy
        self.ema_decay, self.mismatch_probability = ema_decay, float(mismatch_probability)
        self.mismatch_input, self.mismatch_selection = mismatch_input, mismatch_selection
        self.time_sampling_log_fraction, self.validation_seed = float(time_sampling_log_fraction), int(validation_seed)
        self.ema_score_model = copy.deepcopy(score_model).requires_grad_(False).eval() if ema_decay is not None else None
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        self.run_config = None
        self.data_metadata = None
        self._pending_rng_state = None
        if self.sde.kernels is not None:
            self.sde.set_rfft(self.n_fft)

    def train(self, mode=True):
        super().train(mode)
        if self.ema_score_model is not None:
            self.ema_score_model.eval()  # never turn EMA dropout back on
        return self

    def forward(self, x: Tensor, y: Tensor, t: Tensor, *, use_ema=False):
        self._check_xy(x, y)
        if t.shape != (x.shape[0],) or t.device != x.device:
            raise ValueError("t must be [B] on x.device")
        network = self.ema_score_model if use_ema and self.ema_score_model is not None else self.score_model
        output = network(x, y, t)
        if output.shape != x.shape or not output.is_complex():
            raise ValueError("Score network must return complex [B,K,F]")
        return output.to(x.dtype)

    def sample_time(self, batch_size, device, dtype, *, generator=None, strategy=None):
        sampler = self.time_sampler if strategy is None else self.validation_time_sampler
        return sampler(batch_size, device, dtype, generator=generator,
                       strategy=self.time_sampling_strategy if strategy is None else strategy,
                       t_eps=self.t_eps, T=self.sde.T,
                       log_fraction=self.time_sampling_log_fraction)

    @staticmethod
    def _repeat_observation(y, K):
        return y[:, None].expand(-1, K, -1)

    def _forward_components(self, x0, t, *, generator=None, epsilon=None):
        B = x0.shape[0]
        self.sde.send_to(x0.device)
        if epsilon is None:
            epsilon = self.noise_sampler(x0, n_fft=self.n_fft, generator=generator)
        if epsilon.shape != x0.shape:
            raise ValueError("epsilon and x0 shapes must match")
        mean = self.sde._mean(x0.reshape(B, -1), t)
        std = self.sde.fast_factor(t)
        perturbation = self.sde.apply_std(std, epsilon.reshape(B, -1))
        return mean, std, epsilon, perturbation

    def sample_forward(self, x0, t, *, generator=None, epsilon=None):
        """Old four-result interface preserved for your small integration tests.

        This explicitly returns a dense factor. Training uses _forward_components
        instead, so no [B,KF,KF] matrix is allocated in the normal loss path.
        """
        mean, std, noise, perturbation = self._forward_components(x0, t, generator=generator, epsilon=epsilon)
        if std.ndim == 2:
            blocks = self.sde.V[None] * std.reshape(-1, self.sde.K, self.sde.N_fft).transpose(1, 2)[:, :, None, :]
        else:
            blocks = std
        dense = self.sde._blocks_to_dense(blocks)
        return (mean + perturbation).reshape_as(x0), noise, mean, dense

    def loss_terms(self, x0, y, t, *, mismatch_mask=None, use_ema=False,
                   generator=None, epsilon=None):
        return self.objective(self, x0, y, t, mismatch_mask=mismatch_mask,
                              use_ema=use_ema, generator=generator, epsilon=epsilon)

    def compute_score_loss(self, x0, y, *, use_ema=False, log_time_statistics=False,
                           generator=None, strategy=None):
        t = self.sample_time(x0.shape[0], x0.device, x0.real.dtype, generator=generator, strategy=strategy)
        terms = self.loss_terms(x0, y, t, use_ema=use_ema, generator=generator)
        if log_time_statistics:
            self._log_times(terms["time"], x0.shape[0])
        return terms["loss"]

    def compute_mismatch_loss(self, x0, y, *, use_ema=False, generator=None):
        B = x0.shape[0]
        t = torch.full((B,), self.sde.T, dtype=x0.real.dtype, device=x0.device)
        return self.loss_terms(x0, y, t,
            mismatch_mask=torch.ones(B, device=x0.device, dtype=torch.bool),
            use_ema=use_ema, generator=generator)["loss"]

    def training_step(self, batch, batch_idx):
        del batch_idx
        x0, y = batch
        B = x0.shape[0]
        t = self.sample_time(B, x0.device, x0.real.dtype)
        # Always draw this random variable, even for p=0, to reduce unnecessary
        # RNG differences between paired training variants.
        mask = self.loss_selection_rule(B, x0.device, self.mismatch_probability, self.mismatch_selection)
        terms = self.loss_terms(x0, y, t, mismatch_mask=mask)
        loss = terms["loss"]
        self._log("train/loss", loss, B, on_step=True, prog_bar=True)
        self._log("train/score_loss", loss, B, on_step=True)
        self._log("train/mismatch_selected", mask.float().mean(), B)
        # Contribution, NOT conditional branch mean. All ranks log the same keys.
        self._log("train/dsm_contribution", (terms["per_example"] * (~mask)).mean(), B)
        self._log("train/mismatch_contribution", (terms["per_example"] * mask).mean(), B)
        self._log_times(terms["time"], B)
        return loss

    def _log(self, name, value, B, **kwargs):
        if self._trainer is not None:
            self.log(name, value, on_epoch=True, sync_dist=True, batch_size=B, **kwargs)

    def _log_times(self, t, B):
        self._log("train/time_mean", t.mean(), B)
        self._log("train/time_frac_lt_0p05", (t < 0.05).float().mean(), B)
        self._log("train/time_frac_lt_0p02", (t < 0.02).float().mean(), B)

    def _eval_losses(self, batch, batch_idx, prefix):
        x0, y = batch
        # Fixed batch ordering gives identical validation draws across epochs and
        # between training variants. Independent from training RNG/dropout.
        offset = 1000000 if prefix == "test" else 0
        gen = torch.Generator(device=x0.device).manual_seed(self.validation_seed + offset + batch_idx)
        dsm = self.compute_score_loss(x0, y, use_ema=True, generator=gen, strategy="uniform")
        mismatch = self.compute_mismatch_loss(x0, y, use_ema=True, generator=gen)
        self._log(prefix + "/score_loss", dsm, x0.shape[0], prog_bar=True)
        self._log(prefix + "/mismatch_loss", mismatch, x0.shape[0])
        return {"score_loss": dsm, "mismatch_loss": mismatch}

    def validation_step(self, batch, batch_idx):
        return self._eval_losses(batch, batch_idx, "val")["score_loss"]

    def test_step(self, batch, batch_idx):
        return self._eval_losses(batch, batch_idx, "test")

    def configure_optimizers(self):
        parameters = [p for p in self.score_model.parameters() if p.requires_grad]
        if self.optimizer_factory is None:
            optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif callable(self.optimizer_factory):
            optimizer = self.optimizer_factory(params=parameters)
        else:
            optimizer = build_component(self.optimizer_factory, params=parameters)
        # Actual optimizer post-step hook, as before: AMP-skipped steps do not update EMA.
        optimizer.register_step_post_hook(lambda opt, args, kwargs: self.update_ema())
        if self.scheduler_config is None:
            return optimizer
        cfg = dict(self.scheduler_config)
        scheduler = build_component(cfg.pop("scheduler"), optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, **cfg}}

    @torch.no_grad()
    def update_ema(self):
        if self.ema_score_model is None:
            return
        self.ema_update_rule(self.score_model, self.ema_score_model, self.ema_decay)
        self.ema_updates.add_(1)

    def on_fit_start(self):
        self.sde.send_to(self.device)

    def on_train_start(self):
        if self._pending_rng_state is not None:
            state = self._pending_rng_state
            torch.set_rng_state(state["cpu"])
            random.setstate(state["python"])
            if torch.cuda.is_available() and state["cuda"]:
                torch.cuda.set_rng_state_all(state["cuda"])
            self._pending_rng_state = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint["convmix"] = {
            "schema_version": 2, "config": copy.deepcopy(self.run_config),
            "metadata": copy.deepcopy(self.data_metadata),
            "kernels": self.sde.kernels.detach().cpu(),
        }
        checkpoint["convmix_rng"] = {"cpu": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def on_load_checkpoint(self, checkpoint):
        self._pending_rng_state = checkpoint.get("convmix_rng")

    def _check_xy(self, x, y):
        if x.ndim != 3 or y.ndim != 2 or y.shape != (x.shape[0], x.shape[-1]):
            raise ValueError("Expected x [B,K,F] and y [B,F]")
        if not x.is_complex() or not y.is_complex() or x.dtype != y.dtype or x.device != y.device:
            raise ValueError("x/y must be complex tensors with matching dtype/device")
        if x.shape[-1] != self.n_fft // 2 + 1 or x.shape[1] * x.shape[-1] != self.sde.ndim:
            raise ValueError("Dataset/SDE FFT dimensions do not match")
