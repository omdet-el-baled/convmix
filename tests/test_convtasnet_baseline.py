"""Baseline adapter tests. No Asteroid/Hydra substitute is used in these tests."""
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from src.baselines.convtasnet import (ConvTasNetModule, IdentityDeconvolution,
    NegativeSISDR, PeakNormalizer, WienerDeconvolution, wiener_deconvolution)
from src.dataset import MixData_Module


class TinyWaveformNetwork(nn.Module):
    """Test double for the waveform contract; not a Conv-TasNet implementation."""
    def __init__(self, n_src=2, sample_rate=2000):
        super().__init__()
        self.conv = nn.Conv1d(1, n_src, 3, padding=1)

    def forward(self, x):
        return self.conv(x[:, None])


def _data(root, fft_norm="ortho", n_fft=34):
    rng = np.random.default_rng(123)
    filters = [np.array([1.0, 0.1]), np.array([0.8, -0.1])]
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "filters.npz", filters=np.stack(filters), filter_lengths=[2, 2])
    for name, count in [("train", 6), ("cv", 4), ("test", 4)]:
        sources = rng.normal(size=(count, 2, 32)).astype("float32")
        images = np.zeros((count, 2, n_fft), dtype="float32")
        for b in range(count):
            for k, h in enumerate(filters):
                image = np.convolve(sources[b, k], h)
                images[b, k, :len(image)] = image
        mixtures = images.sum(1) + 0.1*rng.normal(size=(count, n_fft)).astype("float32")
        np.savez_compressed(root / f"{name}.npz", sources=sources,
            source_images=images, mixtures=mixtures,
            sources_fft=np.fft.rfft(sources, n=n_fft, norm=fft_norm).astype("complex64"),
            mixtures_fft=np.fft.rfft(mixtures, n=n_fft, norm=fft_norm).astype("complex64"),
            n_fft=n_fft, source_length=32, sample_rate=2000, fft_norm=fft_norm)
    dm = MixData_Module(root, batch_size=2, num_workers=0, fft_norm=fft_norm)
    dm.setup("fit")
    return dm, filters


def _model(filters, n_fft=34, fft_norm="ortho", **kwargs):
    return ConvTasNetModule(TinyWaveformNetwork(), filters=filters, num_sources=2,
        n_fft=n_fft, source_length=32, sample_rate=2000, fft_norm=fft_norm, **kwargs)


@pytest.mark.parametrize("n_fft", [33, 34])
@pytest.mark.parametrize("fft_norm", ["ortho", "backward", "forward"])
def test_existing_npz_loader_to_waveform_adapter(tmp_path, n_fft, fft_norm):
    dm, filters = _data(tmp_path / "data", fft_norm, n_fft)
    model = _model(filters, n_fft, fft_norm)
    batch = next(iter(dm.train_dataloader()))
    assert batch[0].shape == (2, 2, n_fft//2+1)
    assert batch[1].shape == (2, n_fft//2+1)
    full, spectra = model.predict_spectra(batch[1])
    assert full.shape == (2, 2, n_fft)
    assert spectra.shape == batch[0].shape
    value = model.compute_loss(batch)
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.network.parameters())
    restored = torch.fft.irfft(batch[0], n=n_fft, norm=fft_norm)[..., :32]
    assert restored.shape[-1] == 32


def test_wiener_matches_supplied_formula():
    torch.manual_seed(1)
    output = torch.randn(3, 2, 34, dtype=torch.float64)
    filters = [torch.tensor([1.0, 0.2], dtype=torch.float64),
               torch.tensor([0.7, -0.1, 0.05], dtype=torch.float64)]
    H = torch.stack([torch.fft.rfft(h, n=34) for h in filters])
    G = H.conj() / (H.abs().square() + 1e-2)
    expected = torch.fft.irfft(torch.fft.rfft(output) * G, n=34)
    result = wiener_deconvolution(output, filters, snr=1e-2)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)


def test_wiener_gradient():
    module = WienerDeconvolution([[1.0, 0.1], [0.8, -0.1]], n_fft=8)
    x = torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,), eps=1e-6, atol=1e-4)


def test_unregularized_inverse_of_full_linear_convolution():
    torch.manual_seed(2)
    L, n_fft = 20, 24
    s = torch.randn(2, 2, L, dtype=torch.float64)
    filters = [[1.0, 0.1], [0.9, -0.2, 0.05]]
    images = np.zeros((2, 2, n_fft))
    for b in range(2):
        for k in range(2):
            conv = np.convolve(s[b, k].numpy(), filters[k])
            images[b, k, :len(conv)] = conv
    recovered = WienerDeconvolution(filters, n_fft, snr=0)(torch.from_numpy(images))
    torch.testing.assert_close(recovered[..., :L], s, rtol=1e-10, atol=1e-10)
    assert recovered[..., L:].abs().max() < 1e-10


@pytest.mark.parametrize("zero_mean", [False, True])
def test_loss_matches_functional_torchmetrics_and_has_no_running_state(zero_mean):
    loss = NegativeSISDR(zero_mean=zero_mean)
    reference = torch.randn(3, 2, 32) + 0.2
    estimate = reference + 0.3*torch.randn_like(reference)
    expected = -scale_invariant_signal_distortion_ratio(estimate, reference, zero_mean=zero_mean).mean()
    torch.testing.assert_close(loss(estimate, reference), expected)
    loss(torch.randn_like(reference), reference)
    torch.testing.assert_close(loss(estimate, reference), expected)
    assert not list(loss.named_buffers())


def test_peak_normalization_batch_independence_and_silent_input():
    normalizer = PeakNormalizer()
    x = torch.randn(4, 34)
    x[0].zero_()
    n, g = normalizer(x)
    assert torch.isfinite(n).all()
    for i in range(4):
        ni, gi = normalizer(x[i:i+1])
        torch.testing.assert_close(ni, n[i:i+1])
        torch.testing.assert_close(gi, g[i:i+1])
    torch.testing.assert_close(n*g, x)


def test_restored_gain_and_legacy_option():
    class Duplicate(nn.Module):
        def forward(self, x):
            return x[:, None].repeat(1, 2, 1)
    filters = [[1.0], [1.0]]
    model = ConvTasNetModule(Duplicate(), filters=filters, num_sources=2, n_fft=34,
        source_length=32, sample_rate=2000, postprocess=IdentityDeconvolution())
    x = 7*torch.randn(3, 34)
    torch.testing.assert_close(model(x), x[:, None].expand(-1, 2, -1))
    model.normalization = PeakNormalizer(mode="batch")
    model.restore_scale = False
    expected = (x/x.abs().max())[:, None].expand(-1, 2, -1)
    torch.testing.assert_close(model(x), expected)


def test_wrong_lengths_and_regularizer_rejected():
    with pytest.raises(ValueError):
        WienerDeconvolution([[1.0], [1.0]], 10, snr=-1)
    with pytest.raises(ValueError):
        WienerDeconvolution([[1.0], [1.0]], 10)(torch.randn(1, 2, 9))
    with pytest.raises(ValueError):
        _model([[1.0]*8, [1.0]*8], n_fft=34)


@pytest.mark.integration
def test_lightning_npz_training_and_checkpoint_state(tmp_path):
    torch.manual_seed(4)
    dm, filters = _data(tmp_path / "data")
    model = _model(filters)
    before = {k: v.clone() for k, v in model.network.state_dict().items()}
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=1,
        limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False)
    trainer.fit(model, datamodule=dm)
    assert trainer.global_step == 2
    assert any(not torch.equal(before[k], v) for k, v in model.network.state_dict().items())
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert "optimizer_states" in payload and "convmix_convtasnet" in payload
    restored = _model(filters).eval()
    restored.load_state_dict(payload["state_dict"], strict=True)
    y = next(iter(dm.val_dataloader()))[1]
    model.eval()
    torch.testing.assert_close(model.predict_spectra(y)[0], restored.predict_spectra(y)[0])
