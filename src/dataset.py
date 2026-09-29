"""Eager NPZ loading: no archive/file descriptor survives __init__."""
from __future__ import annotations

import random
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from src.components import build_component


class MixDataset(Dataset):
    def __init__(self, path: str | Path, *, fft_norm: str = "ortho",
                 dtype: torch.dtype = torch.complex64, load_time_domain: bool = False):
        self.path = Path(path)
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be complex64 or complex128")
        if fft_norm not in ("ortho", "backward", "forward"):
            raise ValueError("Invalid FFT normalization")
        self.fft_norm, self.dtype = fft_norm, dtype
        real_dtype = np.float64 if dtype == torch.complex128 else np.float32
        complex_dtype = np.complex128 if dtype == torch.complex128 else np.complex64
        with np.load(self.path, allow_pickle=False) as data:
            # Each member is read at most once; .npz entries are not memory maps.
            sources = np.asarray(data["sources"], dtype=real_dtype)
            mixtures = np.asarray(data["mixtures"], dtype=real_dtype)
            if sources.ndim != 3 or mixtures.ndim != 2:
                raise ValueError("Expected sources [N,K,L] and mixtures [N,n_fft]")
            self.num_examples, self.num_sources, L = sources.shape
            self.n_fft = int(data["n_fft"].item()) if "n_fft" in data else mixtures.shape[-1]
            self.source_length = int(data["source_length"].item()) if "source_length" in data else L
            self.sample_rate = int(data["sample_rate"].item()) if "sample_rate" in data else None
            self.num_frequency_bins = self.n_fft // 2 + 1
            if L != self.source_length or self.n_fft < L or mixtures.shape != (self.num_examples, self.n_fft):
                raise ValueError("Inconsistent signal lengths/metadata")
            if "fft_norm" in data:
                stored_norm = str(data["fft_norm"].item())
                if stored_norm != fft_norm:
                    raise ValueError(f"FFT normalization mismatch: file={stored_norm}, requested={fft_norm}")
            elif "sources_fft" in data or "mixtures_fft" in data:
                warnings.warn(f"{self.path}: no FFT normalization metadata; assuming {fft_norm} for cached arrays.")
            self.sources_fft = np.ascontiguousarray(data["sources_fft"] if "sources_fft" in data else
                np.fft.rfft(sources, n=self.n_fft, axis=-1, norm=fft_norm), dtype=complex_dtype)
            self.mixtures_fft = np.ascontiguousarray(data["mixtures_fft"] if "mixtures_fft" in data else
                np.fft.rfft(mixtures, n=self.n_fft, axis=-1, norm=fft_norm), dtype=complex_dtype)
            self.snr_db = np.array(data["snr_db"]) if "snr_db" in data else None
            self.source_seeds = np.array(data["source_seeds"]) if "source_seeds" in data else None
            self.sources = sources if load_time_domain else None
            self.mixtures = mixtures if load_time_domain else None
            self.source_images = np.array(data["source_images"], dtype=real_dtype) if load_time_domain and "source_images" in data else None
        for arr, expected in ((self.sources_fft, (self.num_examples, self.num_sources, self.num_frequency_bins)),
                              (self.mixtures_fft, (self.num_examples, self.num_frequency_bins))):
            if arr.shape != expected or not np.isfinite(arr).all() or not np.iscomplexobj(arr):
                raise ValueError("Invalid cached FFT array")
        if self.num_examples == 0:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        # Returned tensors share read-only-by-convention NumPy storage.
        # Neither the training module nor collator mutates the source arrays.
        return torch.from_numpy(self.sources_fft[index]), torch.from_numpy(self.mixtures_fft[index])

    def get_snr(self, index):
        return None if self.snr_db is None else float(self.snr_db[index])

    def get_source_seeds(self, index):
        return None if self.source_seeds is None else self.source_seeds[index].copy()

    def get_time_domain(self, index):
        if self.sources is None:
            raise RuntimeError("Construct with load_time_domain=True")
        result = {"sources": self.sources[index].copy(), "mixture": self.mixtures[index].copy()}
        if self.source_images is not None:
            result["source_images"] = self.source_images[index].copy()
        return result


def seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class MixData_Module(pl.LightningDataModule):
    def __init__(self, data_dir: str | Path, *, batch_size=4, num_workers=0,
                 fft_norm="ortho", dtype=torch.complex64, pin_memory=None,
                 persistent_workers=None, seed=42, drop_last=False, train_subset=None,
                 dataset=None, loader=None, train_loader=None, val_loader=None, test_loader=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_config, self.loader_config = dataset, loader
        self.split_loader_configs = {"train":train_loader, "val":val_loader, "test":test_loader}
        self.batch_size, self.num_workers = int(batch_size), int(num_workers)
        self.fft_norm, self.dtype = fft_norm, dtype
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.persistent_workers = (num_workers > 0) if persistent_workers is None else bool(persistent_workers)
        if batch_size < 1 or num_workers < 0 or (self.persistent_workers and num_workers == 0):
            raise ValueError("Invalid DataLoader arguments")
        self.seed, self.drop_last, self.train_subset = int(seed), bool(drop_last), train_subset
        self._train_generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def setup(self, stage=None):
        def load(split):
            return build_component(self.dataset_config, MixDataset, path=self.data_dir / f"{split}.npz", fft_norm=self.fft_norm, dtype=self.dtype)
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = load("train")
            if self.val_dataset is None:
                self.val_dataset = load("cv")
        if stage == "validate" and self.val_dataset is None:
            self.val_dataset = load("cv")
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = load("test")
        datasets = [ds for ds in (self.train_dataset, self.val_dataset, self.test_dataset) if ds is not None]
        signatures = {(d.num_sources, d.n_fft, d.source_length, d.sample_rate, d.fft_norm) for d in datasets}
        if len(signatures) > 1:
            raise ValueError("Train/cv/test metadata do not match")

    def _loader(self, ds, train=False, split="train"):
        if ds is None:
            raise RuntimeError("Call setup(stage) before requesting this split")
        if train and self.train_subset is not None:
            n = min(int(self.train_subset), len(ds))
            if n < 1:
                raise ValueError("train_subset must be positive")
            ds = Subset(ds, range(n))
        if train and self.drop_last and len(ds) < self.batch_size:
            raise ValueError("drop_last would discard the entire training dataset")
        return build_component(self.split_loader_configs.get(split) or self.loader_config, DataLoader, dataset=ds, batch_size=self.batch_size, shuffle=train,
            generator=self._train_generator if train else torch.Generator().manual_seed(self.seed + 1),
            num_workers=self.num_workers, worker_init_fn=seed_worker, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, drop_last=self.drop_last if train else False)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, split="val")

    def test_dataloader(self):
        return self._loader(self.test_dataset, split="test")

    def state_dict(self):
        return {"train_generator_state": self._train_generator.get_state()}

    def load_state_dict(self, state):
        if "train_generator_state" in state:
            self._train_generator.set_state(state["train_generator_state"])
