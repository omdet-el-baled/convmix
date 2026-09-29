"""FIR families from the last shared filter_generator.py (no new filter model)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.utils import atomic_npz


def normalize_filter(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1 or not np.isfinite(h).all():
        raise ValueError("h must be a finite 1D array")
    norm = np.linalg.norm(h)
    if norm < eps:
        raise ValueError("Cannot normalize a zero filter")
    return h / norm


def generate_exponential_filter(length: int = 13, decay: float = 0.25,
                                cutoff: float = 0.25, seed: int | None = None) -> np.ndarray:
    if length < 3 or decay <= 0 or not 0 < cutoff < 0.5:
        raise ValueError("Require length>=3, decay>0, 0<cutoff<0.5")
    rng = np.random.default_rng(seed)
    n = np.arange(length, dtype=np.float64)
    lowpass = 2 * cutoff * np.sinc(2 * cutoff * (n - (length - 1) / 2))
    h = lowpass * np.hanning(length) * np.exp(-decay * n)
    h *= 1 + 0.03 * rng.standard_normal(length)
    h[0] += 0.5
    return normalize_filter(h)


def generate_reflection_filter(length: int = 13, decay: float = 0.25,
                               cutoff: float = 0.25, reflection_strength: float = 0.20,
                               num_reflections: int = 2, seed: int | None = None) -> np.ndarray:
    if num_reflections < 0 or reflection_strength < 0.05:
        raise ValueError("Require num_reflections>=0 and reflection_strength>=0.05")
    rng = np.random.default_rng(seed)
    h = generate_exponential_filter(length, decay, cutoff, seed)
    delays = rng.choice(np.arange(2, length), min(num_reflections, length - 2), replace=False)
    for delay in delays:
        h[delay] += rng.choice([-1.0, 1.0]) * rng.uniform(0.05, reflection_strength)
    return normalize_filter(h)


def generate_filter_bank(num_sources: int, length: int = 13,
                         model: str = "exponential", seed: int = 42) -> list[np.ndarray]:
    if num_sources < 2:
        raise ValueError("num_sources must be >=2 for the underdetermined single-sensor setup")
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(num_sources):
        decay, cutoff = rng.uniform(0.15, 0.35), rng.uniform(0.12, 0.35)
        source_seed = int(rng.integers(0, 2**31 - 1))
        if model == "exponential":
            h = generate_exponential_filter(length, decay, cutoff, source_seed)
        elif model == "reflections":
            h = generate_reflection_filter(length, decay, cutoff, seed=source_seed)
        else:
            raise ValueError(f"Unknown filter model {model!r}")
        filters.append(h)
    return filters


def save_filter_bank(filters: list[np.ndarray], path: str | Path) -> None:
    if not filters:
        raise ValueError("Empty filter bank")
    lengths = np.asarray([len(h) for h in filters], dtype=np.int64)
    padded = np.zeros((len(filters), lengths.max()), dtype=np.float64)
    for k, h in enumerate(filters):
        padded[k, :len(h)] = h
    atomic_npz(path, compressed=False, filters=padded, filter_lengths=lengths)


def load_filters(path: str | Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arr = np.asarray(data["filters"], dtype=np.float64)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError("filters must be finite [K,Lh]")
        lengths = np.asarray(data["filter_lengths"] if "filter_lengths" in data
                             else [arr.shape[1]] * arr.shape[0], dtype=np.int64)
        if lengths.shape != (arr.shape[0],) or np.any(lengths < 1) or np.any(lengths > arr.shape[1]):
            raise ValueError("Invalid filter_lengths")
        return [arr[k, :int(n)].copy() for k, n in enumerate(lengths)]
