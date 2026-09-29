"""Config-selected metric functions; definitions remain in src.metrics."""
from functools import partial
from src.metrics import si_sdr, nmse, nmae
from src.components import build_component


class OrderedSourceMetrics:
    def __init__(self, si_sdr_fn=None, nmse_fn=None, nmae_fn=None):
        self.si_sdr = si_sdr if si_sdr_fn is None else build_component(si_sdr_fn)
        self.nmse = nmse if nmse_fn is None else build_component(nmse_fn)
        self.nmae = nmae if nmae_fn is None else build_component(nmae_fn)
