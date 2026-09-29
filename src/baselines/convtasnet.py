"""Conv-TasNet + differentiable Wiener deconvolution on existing NPZ batches.

The data module is unchanged: it yields latent source spectra [B,K,F] and
observed mixture spectra [B,F]. Conversion to real waveforms happens here.
No resampling, source permutation, or mixture normalization across examples
is performed by the default recipe.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.components import build_component


def make_convtasnet(n_src: int, sample_rate: int, **kwargs) -> nn.Module:
    """Instantiate Asteroid's actual ConvTasNet, importing it only when selected.

    Lazy import keeps the existing diffusion-only tests usable without the
    optional Asteroid dependency. All constructor kwargs come from YAML.
    """
    try:
        from asteroid.models import ConvTasNet
    except ModuleNotFoundError as exc:
        if exc.name == "asteroid":
            raise ModuleNotFoundError(
                "Asteroid is missing. See docs/CONVTASNET.md for installation "
                "without downgrading the existing TorchMetrics/CUDA environment."
            ) from exc
        raise
    return ConvTasNet(n_src=n_src, sample_rate=sample_rate, **kwargs)


def _filter_spectra(filters: Sequence, n_fft: int) -> Tensor:
    if not isinstance(n_fft, int) or isinstance(n_fft, bool) or n_fft < 2:
        raise ValueError("n_fft must be an integer >= 2")
    if len(filters) < 1:
        raise ValueError("The filter bank is empty")
    bank = []
    for h in filters:
        # Python float lists must not be rounded through float32 before FFT.
        if not isinstance(h, Tensor):
            h = np.asarray(h)
        h = torch.as_tensor(h).detach().cpu()
        if h.ndim != 1 or h.is_complex() or not 1 <= h.numel() <= n_fft:
            raise ValueError("Each filter must be real, 1D, nonempty and no longer than n_fft")
        if not torch.isfinite(h).all() or h.square().sum() == 0:
            raise ValueError("Filters must be finite and nonzero")
        bank.append(torch.fft.rfft(h.to(torch.float64), n=n_fft))
    return torch.stack(bank)


def wiener_deconvolution(model_output: Tensor, filters, snr: float = 1e-2) -> Tensor:
    """Your original differentiable operation, with the same parameter names.

    model_output: real [...,K,n_fft]. snr is the additive inverse-SNR
    regularizer, NOT the measured dataset SNR in dB. There is no signal crop.
    For repeated training calls use WienerDeconvolution, which caches H.
    """
    return WienerDeconvolution(filters, model_output.shape[-1], snr).to(
        model_output.device)(model_output)


class WienerDeconvolution(nn.Module):
    """Fixed H*/(|H|^2 + snr); registered buffers move with the module."""
    def __init__(self, filters, n_fft: int, snr: float = 1e-2):
        super().__init__()
        if not math.isfinite(snr) or snr < 0:
            raise ValueError("snr must be finite and >= 0")
        self.n_fft, self.snr = int(n_fft), float(snr)
        H = _filter_spectra(filters, n_fft)
        if snr == 0 and torch.any(H.abs() == 0):
            raise ValueError("Unregularized inversion requires nonzero transfer functions")
        # Store real/imag separately so .float(), .double() and .to() behave
        # like ordinary PyTorch modules; no complex buffer cast surprises.
        self.register_buffer("H_real", H.real)
        self.register_buffer("H_imag", H.imag)

    def forward(self, model_output: Tensor) -> Tensor:
        if model_output.is_complex() or not model_output.is_floating_point():
            raise TypeError("WienerDeconvolution expects real floating-point waveforms")
        if model_output.ndim < 2 or model_output.shape[-2:] != (self.H_real.shape[0], self.n_fft):
            raise ValueError(f"Expected [...,{self.H_real.shape[0]},{self.n_fft}], got {tuple(model_output.shape)}")
        # FFTs at arbitrary lengths are not performed in half precision.
        dtype = torch.float64 if model_output.dtype == torch.float64 else torch.float32
        output = model_output.to(dtype)
        H = torch.complex(self.H_real.to(device=output.device, dtype=dtype),
                          self.H_imag.to(device=output.device, dtype=dtype))
        G = H.conj() / (H.abs().square() + self.snr)
        Y = torch.fft.rfft(output, n=self.n_fft, dim=-1)
        return torch.fft.irfft(Y * G, n=self.n_fft, dim=-1)


class IdentityDeconvolution(nn.Module):
    """Explicit no-Wiener ablation. Network output is a latent estimate."""
    def __init__(self, filters=None, n_fft=None):
        super().__init__()

    def forward(self, model_output: Tensor) -> Tensor:
        return model_output


class PeakNormalizer(nn.Module):
    """Return (normalized mixture, gain[B,1]). No target information is used.

    mode='example': peak per recording, invariant to evaluation batch size.
    mode='batch': reproduces the input scaling in the user's original loop.
    mode='none': identity. Scale restoration is controlled by the owner.
    """
    def __init__(self, mode: str = "example", eps: float = 1e-8):
        super().__init__()
        if mode not in {"example", "batch", "none"} or eps <= 0:
            raise ValueError("Invalid normalization mode/eps")
        self.mode, self.eps = mode, float(eps)

    def forward(self, mixture: Tensor) -> tuple[Tensor, Tensor]:
        if mixture.ndim != 2:
            raise ValueError("Expected mixture [B,T]")
        if self.mode == "none":
            gain = mixture.new_ones(mixture.shape[0], 1)
        elif self.mode == "batch":
            gain = mixture.abs().amax().clamp_min(self.eps).expand(mixture.shape[0], 1)
        else:
            gain = mixture.abs().amax(-1, keepdim=True).clamp_min(self.eps)
        return mixture / gain, gain


class NegativeSISDR(nn.Module):
    """Stateless differentiable ordered-source loss (no PIT).

    zero_mean=False preserves the TorchMetrics default in the supplied code.
    The shared project evaluator reports its existing centered SI-SDR separately.
    """
    def __init__(self, zero_mean: bool = False):
        super().__init__()
        self.zero_mean = bool(zero_mean)

    def forward(self, estimate: Tensor, reference: Tensor) -> Tensor:
        if estimate.shape != reference.shape or estimate.ndim != 3:
            raise ValueError("Loss expects matching real [B,K,source_length] tensors")
        ref = reference - reference.mean(-1, keepdim=True) if self.zero_mean else reference
        if not torch.isfinite(reference).all() or torch.any(ref.square().sum(-1) == 0):
            raise ValueError("SI-SDR loss requires finite non-silent references")
        return -scale_invariant_signal_distortion_ratio(
            estimate, reference, zero_mean=self.zero_mean).mean()


class ConvTasNetModule(pl.LightningModule):
    """Independent supervised Lightning module, reusing MixData_Module unchanged.

    forward(mixture_time) -> deconvolved latent estimates [B,K,n_fft].
    predict_spectra(y) -> (latent_time_full, latent_fft).
    Dataset x0 is used only for the training/evaluation target, never as input.
    """
    def __init__(self, network: Any, *, filters, num_sources: int, n_fft: int,
                 source_length: int, sample_rate: int, fft_norm: str = "ortho",
                 postprocess=None, normalization=None, loss=None, optimizer=None,
                 scheduler=None, restore_scale: bool = True):
        super().__init__()
        if num_sources < 2 or sample_rate <= 0 or not 1 <= source_length <= n_fft:
            raise ValueError("Invalid dataset dimensions/sample rate")
        if fft_norm not in {"ortho", "backward", "forward"}:
            raise ValueError("Invalid fft_norm")
        if len(filters) != num_sources or source_length + max(map(len, filters)) - 1 > n_fft:
            raise ValueError("Filter count or full convolution length does not match dataset")
        self.num_sources, self.n_fft = int(num_sources), int(n_fft)
        self.source_length, self.sample_rate = int(source_length), int(sample_rate)
        self.fft_norm, self.restore_scale = fft_norm, bool(restore_scale)
        self.network = build_component(network, n_src=num_sources, sample_rate=sample_rate)
        if not isinstance(self.network, nn.Module):
            raise TypeError("network must construct an nn.Module")
        self.postprocess = build_component(postprocess, WienerDeconvolution,
                                          filters=filters, n_fft=n_fft)
        self.normalization = build_component(normalization, PeakNormalizer)
        self.loss = build_component(loss, NegativeSISDR)
        self.optimizer_config, self.scheduler_config = optimizer, scheduler
        self.run_config = self.data_metadata = None
        self.filters = [torch.as_tensor(h, dtype=torch.float64).cpu().clone() for h in filters]
        self.save_hyperparameters({"num_sources": num_sources, "n_fft": n_fft,
            "source_length": source_length, "sample_rate": sample_rate,
            "fft_norm": fft_norm, "restore_scale": restore_scale})

    def forward(self, mixture: Tensor) -> Tensor:
        if mixture.ndim != 2 or mixture.shape[-1] != self.n_fft or mixture.is_complex():
            raise ValueError(f"Expected real mixture [B,{self.n_fft}]")
        normalized, gain = self.normalization(mixture)
        output = self.network(normalized)
        expected = (mixture.shape[0], self.num_sources, self.n_fft)
        if output.shape != expected:
            raise ValueError(f"Separator must preserve the full length: expected {expected}, got {tuple(output.shape)}")
        output = self.postprocess(output)
        if self.restore_scale:
            output = output * gain[:, None, :]
        return output

    def predict_time(self, y: Tensor) -> Tensor:
        if not y.is_complex() or y.ndim != 2 or y.shape[-1] != self.n_fft // 2 + 1:
            raise ValueError("Expected observed mixture spectra [B,F]")
        mixture = torch.fft.irfft(y, n=self.n_fft, dim=-1, norm=self.fft_norm)
        return self(mixture)

    def predict_spectra(self, y: Tensor) -> tuple[Tensor, Tensor]:
        full = self.predict_time(y)
        spectra = torch.fft.rfft(full, n=self.n_fft, dim=-1, norm=self.fft_norm)
        return full, spectra

    def compute_loss(self, batch) -> Tensor:
        x0, y = batch  # exact order returned by your NPZ data module
        if not x0.is_complex() or x0.shape != (y.shape[0], self.num_sources, self.n_fft // 2 + 1):
            raise ValueError("Expected latent-source spectra [B,K,F] followed by mixture spectra [B,F]")
        reference = torch.fft.irfft(x0, n=self.n_fft, dim=-1,
                                    norm=self.fft_norm)[..., :self.source_length]
        estimate_full = self.predict_time(y)
        # Do not truncate the observed convolution before the network/Wiener.
        # Crop only the latent output and the latent training reference.
        value = self.loss(estimate_full[..., :self.source_length], reference)
        if value.ndim != 0 or not torch.isfinite(value):
            raise FloatingPointError("Nonfinite/non-scalar baseline loss")
        return value

    def _step(self, batch, stage):
        value = self.compute_loss(batch)
        self.log(f"{stage}/loss", value, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=batch[0].shape[0])
        return value

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        default = lambda params: torch.optim.Adam(params, lr=2e-4)
        optimizer = build_component(self.optimizer_config, default, params=parameters)
        spec = copy.deepcopy(self.scheduler_config)
        if not spec or not spec.pop("enabled", True):
            return optimizer
        scheduler = build_component(spec.pop("scheduler"), optimizer=optimizer)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            spec.setdefault("monitor", "val/loss")
            spec.setdefault("strict", True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, **spec}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["convmix_convtasnet"] = {
            "format_version": 1, "config": copy.deepcopy(self.run_config),
            "metadata": copy.deepcopy(self.data_metadata),
            "filters": [h.clone() for h in self.filters],
        }
