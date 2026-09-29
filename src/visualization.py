from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config, project_path
from src.data.filters import load_filters
from src.dataset import MixDataset
from src.factory import make_dataset
from src.components import build_component


def save_lines(path: Path, x, series, *, xlabel, ylabel, title, log_y=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, values in series:
        ax.plot(x, values, label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def visualize_dataset(cfg, index=0, out=None):
    data_dir = project_path(cfg["data"]["data_dir"])
    ds = make_dataset(cfg, "cv", load_time_domain=True)
    if not 0 <= index < len(ds):
        raise IndexError(index)
    filters = load_filters(project_path(cfg["data"]["filters"]) if cfg["data"]["filters"] else data_dir / "filters.npz")
    out = project_path(out or f"figures/{data_dir.name}/example_{index:04d}")
    item = ds.get_time_domain(index)
    source_t = np.arange(ds.source_length) / ds.sample_rate
    mix_t = np.arange(ds.n_fft) / ds.sample_rate
    freq = np.fft.rfftfreq(ds.n_fft, d=1 / ds.sample_rate)
    series = [(f"s{k+1}", s) for k, s in enumerate(item["sources"])]
    save_lines(out / "sources", source_t, series, xlabel="Signal time (s)", ylabel="Amplitude", title="Latent sources")
    save_lines(out / "images", mix_t, [(f"h{k+1} * s{k+1}", s) for k, s in enumerate(item["source_images"])],
               xlabel="Signal time (s)", ylabel="Amplitude", title="Convolutive source images")
    clean = item["source_images"].sum(0)
    save_lines(out / "mixture", mix_t, [("clean", clean), ("observed", item["mixture"])],
               xlabel="Signal time (s)", ylabel="Amplitude", title="Mixture")
    save_lines(out / "noise", mix_t, [("noise", item["mixture"] - clean)],
               xlabel="Signal time (s)", ylabel="Amplitude", title="Measurement noise")
    length = max(map(len, filters))
    padded = [np.pad(h, (0, length-len(h))) for h in filters]
    save_lines(out / "filters", np.arange(length), [(f"h{k+1}", h) for k, h in enumerate(padded)],
               xlabel="Tap", ylabel="Amplitude", title="FIR filters")
    H = np.stack([np.fft.rfft(h, n=ds.n_fft) for h in filters])
    save_lines(out / "filter_responses", freq, [(f"H{k+1}", 20*np.log10(np.maximum(abs(h), 1e-12))) for k, h in enumerate(H)],
               xlabel="Frequency (Hz)", ylabel="Magnitude (dB, no peak normalization)", title="Filter transfer functions")
    S = ds.sources_fft[index]
    save_lines(out / "source_spectra", freq, [(f"S{k+1}", abs(s)) for k, s in enumerate(S)],
               xlabel="Frequency (Hz)", ylabel="Absolute rFFT magnitude", title="Latent source spectra")
    independent = np.zeros((ds.num_sources, ds.n_fft))
    for k in range(ds.num_sources):
        conv = np.convolve(item["sources"][k].astype(float), filters[k], mode="full")
        independent[k, :len(conv)] = conv
    target = np.fft.rfft(independent, norm=ds.fft_norm)
    print("Independent convolution vs cached spectral product, max abs:", np.max(abs(target-H*S)))
    print("Saved figures:", out)


def plot_separation(out: Path, cfg: dict, index: int):
    with np.load(out / "per_example.npz", allow_pickle=False) as z:
        if "estimates" not in z:
            return
        estimates, sisdr = z["estimates"][0], z["si_sdr"][0]
    # The protocol identifies the selected split independently of config defaults.
    import json
    protocol = json.loads((out / "protocol.json").read_text())
    dataset_path = Path(protocol["dataset_path"])
    ds = build_component(cfg.get("components", {}).get("datamodule", {}).get("dataset"), MixDataset,
                         path=dataset_path, fft_norm=cfg["data"]["fft_norm"], load_time_domain=True)
    reference = ds.sources[index]
    t = np.arange(ds.source_length) / ds.sample_rate
    freq = np.fft.rfftfreq(ds.n_fft, d=1/ds.sample_rate)
    for k in range(ds.num_sources):
        save_lines(out / f"source_{k+1}", t, [("Reference", reference[k]), ("Estimate", estimates[k])],
            xlabel="Signal time (s)", ylabel="Amplitude", title=f"Source {k+1}; SI-SDR {sisdr[k]:.2f} dB")
        spectra = [np.fft.rfft(a, n=ds.n_fft, norm=ds.fft_norm) for a in (reference[k], estimates[k])]
        save_lines(out / f"source_{k+1}_spectrum", freq,
            [(name, abs(s)) for name, s in zip(("Reference", "Estimate"), spectra)],
            xlabel="Frequency (Hz)", ylabel="Absolute rFFT magnitude", title=f"Source {k+1} spectrum")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    visualize_dataset(load_config(a.config, a.set), a.index, a.out)
