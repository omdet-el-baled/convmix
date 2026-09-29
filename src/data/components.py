"""Composable generation with the same source and filter functions as before."""
from __future__ import annotations
from functools import partial
from src.components import build_component
from src.data.generate import generate_source, add_noise_at_snr
from src.data.filters import generate_filter_bank


class SourceBank:
    """Select per-source callables from YAML, plus a fallback for K>len(sources)."""
    def __init__(self, sources=None, fallback=None):
        self.sources, self.fallback = sources or [], fallback

    def __call__(self, source_index, *, kind, num_samples, fs, seed):
        spec = self.sources[source_index] if source_index < len(self.sources) else self.fallback
        if spec is None:
            return generate_source(source_index, kind=kind, num_samples=num_samples, fs=fs, seed=seed)
        return build_component(spec, num_samples=num_samples, fs=fs, seed=seed)


class DatasetGenerator:
    def __init__(self, filters=None, sources=None, measurement_noise=None):
        self.filters = filters
        self.sources = build_component(sources, SourceBank)
        self.measurement_noise = add_noise_at_snr if measurement_noise is None else build_component(measurement_noise)

    def __call__(self, config, *, force=False):
        from src.data.generate import generate_dataset
        def filters_fn(num_sources, length, model, seed):
            return build_component(self.filters, generate_filter_bank,
                num_sources=num_sources, length=length, model=model, seed=seed)
        return generate_dataset(config, force=force, filter_factory=filters_fn,
                                source_factory=self.sources, noise_factory=self.measurement_noise)
