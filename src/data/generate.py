"""Reproducible latent-source NPZ data with full time-domain convolution.

These are synthetic, paper-like signals, not measured mechanical data. Music
uses fixed harmonic stacks. Finite-record RMS normalization means we do NOT
claim strict stationarity of the normalized ensemble.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.config import load_config, project_path
from src.data.filters import generate_filter_bank, save_filter_bank
from src.utils import atomic_json, atomic_npz, file_hash


def normalize_rms(x: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    power = float(np.mean(x**2))
    if not np.isfinite(power) or power <= 1e-24:
        raise ValueError("Source is zero/nonfinite; check frequencies against sample rate")
    return x * target_rms / np.sqrt(power + 1e-12)


def add_noise_at_snr(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(clean**2))
    if power <= 0:
        raise ValueError("Cannot define SNR for a zero mixture")
    noise = rng.standard_normal(clean.shape)
    noise *= np.sqrt(power / (10**(snr_db / 10)) / (np.mean(noise**2) + 1e-12))
    return clean + noise


def generate_spur_gear_source(num_samples: int, fs: float, *, rpm_range,
                              teeth_range, harmonic_range, sideband_range, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples, dtype=np.float64) / fs
    rpm = rng.uniform(*rpm_range)
    teeth = int(rng.integers(teeth_range[0], teeth_range[1] + 1))
    num_harmonics = int(rng.integers(harmonic_range[0], harmonic_range[1] + 1))
    num_sidebands = int(rng.integers(sideband_range[0], sideband_range[1] + 1))
    shaft_frequency = rpm / 60
    mesh_frequency = shaft_frequency * teeth
    x = np.zeros(num_samples, dtype=np.float64)
    for harmonic in range(1, num_harmonics + 1):
        frequency = harmonic * mesh_frequency
        if frequency >= fs / 2:
            continue
        amplitude = rng.uniform(0.6, 1.0) / harmonic
        x += amplitude * np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
        for sideband in range(1, num_sidebands + 1):
            offset = sideband * shaft_frequency
            sb_amplitude = amplitude * rng.uniform(0.05, 0.20) / sideband
            for sign in (-1, 1):
                f_sb = frequency + sign * offset
                if 0 < f_sb < fs / 2:
                    x += sb_amplitude * np.sin(2 * np.pi * f_sb * t + rng.uniform(0, 2 * np.pi))
    depth = rng.uniform(0.05, 0.20)
    phase = rng.uniform(0, 2 * np.pi)
    x *= 1 + depth * np.sin(2 * np.pi * shaft_frequency * t + phase)
    if rng.random() < 0.05:
        period = max(int(fs / max(shaft_frequency, 1)), 1)
        start = int(rng.integers(0, period))
        x[np.arange(start, num_samples, period)] += rng.uniform(0.3, 0.8)
    return normalize_rms(x)


def generate_epicyclic_source(num_samples: int, fs: float, seed: int) -> np.ndarray:
    # Synthetic epicyclic-like family, not a complete physical gearbox model.
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples, dtype=np.float64) / fs
    carrier = rng.uniform(30, 80) / 60
    z_sun = rng.integers(18, 25)
    z_planet = rng.integers(25, 36)
    z_ring = rng.integers(70, 91)
    x = np.zeros(num_samples)
    for f0 in (carrier * z_sun, carrier * z_planet, carrier * z_ring / 3):
        for harmonic in range(1, 5):
            freq = f0 * harmonic
            if freq < fs / 2:
                amp = rng.uniform(0.4, 1.0) / harmonic
                x += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return normalize_rms(x)


def generate_stationary_music_source(num_samples: int, fs: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples, dtype=np.float64) / fs
    fundamental = rng.uniform(40, 220)
    num_harmonics = int(rng.integers(3, 9))
    spectral_decay = rng.uniform(0.7, 1.5)
    x = np.zeros(num_samples)
    for harmonic in range(1, num_harmonics + 1):
        freq = harmonic * fundamental
        if freq >= fs / 2:
            break
        amplitude = rng.uniform(0.7, 1.3) / harmonic**spectral_decay
        x += amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return normalize_rms(x)


def generate_source(source_index: int, *, kind: str, num_samples: int,
                    fs: float, seed: int) -> np.ndarray:
    if kind == "music":
        return generate_stationary_music_source(num_samples, fs, seed)
    if kind != "mechanical":
        raise ValueError(f"Unknown data kind {kind!r}")
    if source_index == 0:
        return generate_spur_gear_source(num_samples, fs, rpm_range=(200, 400),
            teeth_range=(15, 24), harmonic_range=(4, 7), sideband_range=(2, 5), seed=seed)
    if source_index == 1:
        return generate_spur_gear_source(num_samples, fs, rpm_range=(500, 1000),
            teeth_range=(25, 34), harmonic_range=(2, 4), sideband_range=(3, 6), seed=seed)
    return generate_epicyclic_source(num_samples, fs, seed)


def generate_split(*, num_examples: int, num_sources: int, num_samples: int,
                   fs: float, filters: list[np.ndarray], snr_range, seed: int,
                   kind: str, cache_fft: bool = True, fft_norm: str = "ortho",
                   source_factory=None, noise_factory=None) -> dict:
    if num_examples < 1 or num_sources != len(filters) or num_samples < 2 or fs <= 0:
        raise ValueError("Invalid data dimensions / filter count")
    if len(snr_range) != 2 or snr_range[0] > snr_range[1]:
        raise ValueError("Invalid snr_range")
    source_factory = source_factory or generate_source
    noise_factory = noise_factory or add_noise_at_snr
    rng = np.random.default_rng(seed)
    n_fft = num_samples + max(map(len, filters)) - 1
    sources = np.empty((num_examples, num_sources, num_samples), np.float32)
    images = np.zeros((num_examples, num_sources, n_fft), np.float32)
    mixtures = np.empty((num_examples, n_fft), np.float32)
    snrs = np.empty(num_examples, np.float32)
    seeds = np.empty((num_examples, num_sources), np.int64)
    for i in range(num_examples):
        for k in range(num_sources):
            seeds[i, k] = rng.integers(0, 2**31 - 1)
            sources[i, k] = source_factory(k, kind=kind, num_samples=num_samples,
                                            fs=fs, seed=int(seeds[i, k]))
            # Convolve the STORED latent source so targets/cached FFTs agree
            # up to their explicitly chosen float32 rounding.
            image = np.convolve(sources[i, k].astype(np.float64), filters[k], mode="full")
            images[i, k, :len(image)] = image
        clean = images[i].astype(np.float64).sum(axis=0)
        snrs[i] = rng.uniform(*snr_range)
        mixtures[i] = noise_factory(clean, float(snrs[i]), rng)
    output = dict(sources=sources, source_images=images, mixtures=mixtures,
                  snr_db=snrs, source_seeds=seeds, sample_rate=np.int64(fs),
                  source_length=np.int64(num_samples), n_fft=np.int64(n_fft),
                  fft_norm=np.asarray(fft_norm), kind=np.asarray(kind),
                  seed=np.int64(seed), schema_version=np.int64(1))
    if cache_fft:
        output["sources_fft"] = np.fft.rfft(sources, n=n_fft, axis=-1, norm=fft_norm).astype(np.complex64)
        output["mixtures_fft"] = np.fft.rfft(mixtures, n=n_fft, axis=-1, norm=fft_norm).astype(np.complex64)
    return output


def generate_dataset(config: dict, *, force: bool = False, filter_factory=None,
                     source_factory=None, noise_factory=None) -> Path:
    d = config["data"]
    out = project_path(d["data_dir"])
    paths = [out / n for n in ("train.npz", "cv.npz", "test.npz", "filters.npz")]
    if not force and any(p.exists() for p in paths):
        raise FileExistsError(f"Refusing to overwrite dataset in {out}; use --force explicitly.")
    filter_factory = filter_factory or generate_filter_bank
    filters = filter_factory(d["num_sources"], d["filter_length"], d["filter_model"], d["seed"])
    save_filter_bank(filters, out / "filters.npz")
    n_samples = round(d["duration"] * d["sample_rate"])
    for name, count, snr, seed_offset in (
        ("train", d["train"], d["train_snr"], 1000),
        ("cv", d["val"], d["eval_snr"], 2000),
        ("test", d["test"], d["eval_snr"], 3000),
    ):
        print(f"Generating {name}: {count} examples", flush=True)
        values = generate_split(num_examples=count, num_sources=d["num_sources"], num_samples=n_samples,
            fs=d["sample_rate"], filters=filters, snr_range=snr, seed=d["seed"] + seed_offset,
            kind=d["kind"], cache_fft=d["cache_fft"], fft_norm=d["fft_norm"],
            source_factory=source_factory, noise_factory=noise_factory)
        atomic_npz(out / f"{name}.npz", compressed=d["compressed"], **values)
    atomic_json(out / "metadata.json", {"data_config": d, "filter_sha256": file_hash(out / "filters.npz"),
                "target": "latent_sources", "convolution": "full_linear", "schema_version": 1})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print("Saved dataset:", generate_dataset(load_config(args.config, args.set), force=args.force))


if __name__ == "__main__":
    main()
