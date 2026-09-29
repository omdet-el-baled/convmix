"""Train/evaluate the supervised waveform baseline without touching diffusion."""
from __future__ import annotations

import copy
import csv
import inspect
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

from src.components import build_component
from src.config import config_hash, project_path
from src.data.filters import load_filters
from src.dataset import MixDataset
from src.metric_suite import OrderedSourceMetrics
from src.utils import (atomic_json, atomic_npz, atomic_yaml, choose_device,
                       file_hash, load_checkpoint, source_tree_hash)


def _versions():
    packages = {}
    for name in ("torch", "torchaudio", "asteroid", "asteroid-filterbanks", "torchmetrics",
                 "pytorch-lightning", "numpy", "hydra-core"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda}


def _metadata(ds):
    if ds.sample_rate is None:
        raise ValueError("This workflow requires sample_rate metadata in the NPZ")
    return {"n_fft": int(ds.n_fft), "source_length": int(ds.source_length),
            "num_sources": int(ds.num_sources), "sample_rate": int(ds.sample_rate),
            "fft_norm": ds.fft_norm}


def _build_model(config, metadata, filters):
    model = build_component(config["separator"], filters=filters, **{
        k: metadata[k] for k in ("num_sources", "n_fft", "source_length", "sample_rate", "fft_norm")})
    model.run_config = copy.deepcopy(config)
    model.data_metadata = copy.deepcopy(metadata)
    return model


def _check_scheduler(config):
    spec = config["separator"].get("scheduler") or {}
    if not spec.get("enabled", False):
        return
    target = spec.get("scheduler", {}).get("_target_", "")
    if target.endswith("ReduceLROnPlateau"):
        if spec.get("interval", "epoch") != "epoch":
            raise ValueError("The plateau recipe must use interval=epoch")
        every = config["trainer"]["check_val_every_n_epoch"]
        frequency = int(spec.get("frequency", 1))
        if frequency < 1 or frequency % every != 0:
            raise ValueError("Plateau scheduler frequency must be a multiple of validation frequency")


def train(config: dict):
    """Run with the existing MixData_Module; save full model+optimizer checkpoints."""
    config = copy.deepcopy(config)
    if config["trainer"].get("devices", 1) != 1:
        raise ValueError("This baseline workflow is currently single-device")
    if str(config["trainer"].get("precision", "32-true")) not in ("32", "32-true"):
        raise ValueError("Use precision=32-true for this FFT-based baseline workflow")
    _check_scheduler(config)
    pl.seed_everything(int(config["seed"]), workers=True)
    torch.set_float32_matmul_precision("highest")
    run_dir = project_path(config["run"]["output_dir"]) / config["run"]["name"]
    resume = config["run"].get("resume")
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run exists: {run_dir}. Choose run.name or explicitly resume.")
    data_dir = project_path(config["data"]["data_dir"])
    dm = build_component(config["datamodule"], data_dir=data_dir, seed=config["seed"])
    dm.setup("fit")
    ds = dm.train_dataset
    metadata = _metadata(ds)
    if ds.num_sources != int(config["data"]["num_sources"]):
        raise ValueError("data.num_sources does not match NPZ metadata")
    filter_path = project_path(config["data"]["filters"]) if config["data"].get("filters") else data_dir / "filters.npz"
    filters = load_filters(filter_path)
    metadata.update(filters_sha256=file_hash(filter_path), train_sha256=file_hash(ds.path),
                    cv_sha256=file_hash(dm.val_dataset.path))
    model = _build_model(config, metadata, filters)
    ckpt_path = None
    if resume is not None:
        ckpt_path = project_path(resume)
        old = load_checkpoint(ckpt_path, trust=config["run"].get("trust_checkpoint", False))
        saved = old.get("convmix_convtasnet") or {}
        previous = saved.get("config")
        if not previous:
            raise ValueError("Resume requires a checkpoint made by train_convtasnet.py")
        if previous["separator"] != config["separator"] or previous["seed"] != config["seed"]:
            raise ValueError("Cannot change separator, postprocessor, optimizer/loss or seed on resume")
        if saved.get("metadata") != metadata:
            raise ValueError("Dataset/filter identity differs from checkpoint")
        for key in ("batch_size", "drop_last", "train_subset"):
            if previous["datamodule"].get(key) != config["datamodule"].get(key):
                raise ValueError(f"Cannot change datamodule.{key} during resume")
        if previous["trainer"].get("accumulate_grad_batches", 1) != config["trainer"].get("accumulate_grad_batches", 1):
            raise ValueError("Cannot change gradient accumulation on resume")
        if int(config["trainer"]["max_epochs"]) <= int(old.get("epoch", -1)) + 1:
            raise ValueError("max_epochs must exceed the completed checkpoint epoch count")

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_yaml(run_dir / "resolved.yaml", config)
    atomic_json(run_dir / "environment.json", _versions())
    atomic_json(run_dir / "manifest.json", {"metadata": metadata, "config_hash": config_hash(config),
                "source_tree_sha256": source_tree_hash()})
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    atomic_yaml(run_dir / "sessions" / (session + ".yaml"), config)
    loggers = []
    for name, spec in config["logger"].items():
        if spec is None:
            continue
        kwargs = {"save_dir": str(run_dir)} if name in ("csv", "tensorboard") else {}
        if name == "csv":
            kwargs["version"] = "" if resume is None else "resume_" + session
        loggers.append(build_component(spec, **kwargs))
    callbacks, checkpoint = [], None
    for name, spec in config["callbacks"].items():
        if spec is None:
            continue
        kwargs = {"dirpath": str(run_dir / "checkpoints")} if name in ("checkpoint", "last_checkpoint") else {}
        callback = build_component(spec, **kwargs)
        callbacks.append(callback)
        if name == "checkpoint":
            checkpoint = callback
    if not isinstance(checkpoint, ModelCheckpoint):
        raise ValueError("callbacks.checkpoint must be ModelCheckpoint-compatible")
    trainer = build_component(config["trainer"], callbacks=callbacks, logger=loggers or False,
                              default_root_dir=str(run_dir))
    (run_dir / "model_structure.txt").write_text(str(model), encoding="utf-8")
    print(f"Run: {run_dir}\nNPZ batches: (latent spectra, mixture spectra)\n"
          f"Conv-TasNet input length={ds.n_fft}; latent loss length={ds.source_length}", flush=True)
    fit_options = {}
    if "weights_only" in inspect.signature(trainer.fit).parameters:
        fit_options["weights_only"] = not config["run"].get("trust_checkpoint", False)
    trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt_path) if ckpt_path else None, **fit_options)
    # fast_dev_run intentionally does not create a production checkpoint.
    last = run_dir / "checkpoints" / "last.ckpt"
    if not trainer.fast_dev_run:
        trainer.save_checkpoint(last)
    score = checkpoint.best_model_score
    artifacts = {"best": checkpoint.best_model_path, "last": str(last) if last.exists() else "",
                 "best_val_loss": None if score is None else float(score),
                 "global_step": int(trainer.global_step), "completed_epochs": int(trainer.current_epoch),
                 "config_hash": config_hash(config)}
    atomic_json(run_dir / "artifacts.json", artifacts)
    print("Training artifacts:", artifacts, flush=True)
    if config.get("actions", {}).get("evaluate_after_training", False):
        path = artifacts["best"] or artifacts["last"]
        if not path:
            raise ValueError("No checkpoint available for post-training evaluation")
        evaluate(path, trust=True, split=config["evaluation"]["split"],
                 batch_size=config["evaluation"]["batch_size"],
                 max_examples=config["evaluation"]["max_examples"],
                 save_estimates=config["evaluation"]["save_estimates"],
                 output_dir=config["evaluation"]["output_dir"])
    return run_dir


def restore(checkpoint, *, device="cpu", trust=False):
    """Strict reconstruction from saved target specs; never guess an architecture."""
    if not trust:
        raise ValueError("Checkpoint configuration contains executable _target_ paths. "
                         "Set evaluation.trust_checkpoint=true only for a checkpoint you trust.")
    payload = load_checkpoint(project_path(checkpoint), trust=trust)
    embedded = payload.get("convmix_convtasnet") or {}
    if embedded.get("format_version") != 1 or not embedded.get("config"):
        raise ValueError("Expected a train_convtasnet.py checkpoint, not diffusion weights or a raw old state_dict")
    config, metadata = embedded["config"], embedded["metadata"]
    model = _build_model(config, metadata, embedded["filters"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(choose_device(device)).eval()
    return model, copy.deepcopy(config), metadata


@torch.inference_mode()
def evaluate(checkpoint, *, trust=False, device="auto", data_dir=None, split="test",
             batch_size=16, max_examples=None, save_estimates=True, output_dir="results/convtasnet",
             out=None, force=False):
    """All selected NPZ examples; reuse the project's ordered-source metrics."""
    if split not in {"train", "cv", "test"} or batch_size < 1:
        raise ValueError("Invalid evaluation split/batch_size")
    model, config, metadata = restore(checkpoint, device=device, trust=trust)
    directory = project_path(data_dir or config["data"]["data_dir"])
    spec = config["datamodule"].get("dataset")
    ds = build_component(spec, MixDataset, path=directory / f"{split}.npz",
                         fft_norm=metadata["fft_norm"], load_time_domain=True)
    if _metadata(ds) != {k: metadata[k] for k in _metadata(ds)}:
        raise ValueError("Test-set source count/sample rate/FFT/length differs from training")
    filter_path = project_path(config["data"]["filters"]) if config["data"].get("filters") else directory / "filters.npz"
    filters = load_filters(filter_path)
    if len(filters) != len(model.filters) or any(
        np.asarray(h).shape != tuple(old.shape) or not np.allclose(h, old.numpy(), rtol=1e-6, atol=1e-8)
        for h, old in zip(filters, model.filters)):
        raise ValueError("Evaluation filters differ from the checkpoint")
    if ds.source_images is None:
        raise ValueError("Source-image metrics require source_images in the NPZ")
    count = len(ds) if max_examples is None else min(len(ds), int(max_examples))
    if count < 1:
        raise ValueError("Evaluation subset must be nonempty")
    # DataLoader yields exactly the same (x0_fft,y_fft) interface as training.
    loader = DataLoader(Subset(ds, range(count)), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=model.device.type == "cuda")
    protocol = {"checkpoint_sha256": file_hash(project_path(checkpoint)), "split_sha256": file_hash(ds.path),
                "source_tree_sha256": source_tree_hash(), "indices": list(range(count)), "batch_size": batch_size,
                "ordered_sources": True, "centered_evaluation_si_sdr": True,
                "fft_norm": ds.fft_norm, "separator": config["separator"], "versions": _versions()}
    protocol_id = config_hash(protocol)[:12]
    destination = project_path(out) if out else project_path(output_dir) / config["run"]["name"] / f"{split}_{protocol_id}"
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(f"Evaluation exists: {destination}; use force=true to overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "protocol.json", protocol)
    metrics = build_component(config.get("metrics"), OrderedSourceMetrics)
    results = {key: [] for key in ("si_sdr", "source_image_si_sdr", "nmse", "nmae",
        "measurement_residual", "cropped_linear_measurement_residual", "tail_energy_ratio")}
    estimates, full_estimates = [], []
    H = np.stack([np.fft.rfft(h, n=ds.n_fft) for h in filters])
    start = 0
    for x0, y in loader:
        full, X = model.predict_spectra(y.to(model.device))
        if not torch.isfinite(full).all():
            raise FloatingPointError("Nonfinite separation output")
        full, X = full.cpu().numpy(), X.cpu().numpy()
        size = len(full)
        estimate = full[..., :ds.source_length]
        reference = ds.sources[start:start + size]
        results["si_sdr"].append(metrics.si_sdr(estimate, reference))
        results["nmse"].append(metrics.nmse(X, x0.numpy()))
        results["nmae"].append(metrics.nmae(X, x0.numpy()))
        images = np.zeros((size, ds.num_sources, ds.n_fft), dtype=np.float64)
        for b in range(size):
            for k, h in enumerate(filters):
                value = np.convolve(estimate[b, k], h, mode="full")
                images[b, k, :len(value)] = value
        results["source_image_si_sdr"].append(metrics.si_sdr(images, ds.source_images[start:start + size]))
        y_np = y.numpy()
        predicted_y = (H[None] * X).sum(1)
        results["measurement_residual"].append(np.linalg.norm(predicted_y-y_np, axis=-1) /
                                               np.maximum(np.linalg.norm(y_np, axis=-1), 1e-12))
        observation = ds.mixtures[start:start + size]
        results["cropped_linear_measurement_residual"].append(np.linalg.norm(images.sum(1)-observation, axis=-1) /
                                                              np.maximum(np.linalg.norm(observation, axis=-1), 1e-12))
        results["tail_energy_ratio"].append((full[..., ds.source_length:]**2).sum(-1) /
                                             np.maximum((full**2).sum(-1), 1e-12))
        if save_estimates:
            estimates.append(estimate.astype(np.float32))
            full_estimates.append(full.astype(np.float32))
        start += size
        print(f"Evaluated {start}/{count}", flush=True)
    arrays = {key: np.concatenate(value) for key, value in results.items()}
    arrays["indices"] = np.arange(count)
    if save_estimates:
        arrays.update(estimates=np.concatenate(estimates), estimates_full=np.concatenate(full_estimates))
    S, I = arrays["si_sdr"], arrays["source_image_si_sdr"]
    summary = {"num_examples": count, "num_sources": ds.num_sources,
        "latent_si_sdr_db_mean": float(S.mean()), "source_image_si_sdr_db_mean": float(I.mean()),
        "nmse_mean": float(arrays["nmse"].mean()), "nmae_mean": float(arrays["nmae"].mean()),
        "failure_rate_lt_5db": float((S < 5).mean()),
        "measurement_residual_mean": float(arrays["measurement_residual"].mean()),
        "cropped_linear_measurement_residual_mean": float(arrays["cropped_linear_measurement_residual"].mean()),
        "mean_tail_energy_ratio": float(arrays["tail_energy_ratio"].mean()),
        "per_source_si_sdr_mean": S.mean(0).tolist(), "per_source_si_sdr_std": S.std(0).tolist(),
        "per_source_si_sdr_min": S.min(0).tolist(), "per_source_si_sdr_max": S.max(0).tolist(),
        "checkpoint": str(project_path(checkpoint)), "protocol_id": protocol_id,
        "network_passes_per_example": 1, "training_config": config}
    atomic_json(destination / "summary.json", summary)
    atomic_npz(destination / "per_example.npz", **arrays)
    rows = []
    for i in range(count):
        row = {"index": i, "si_sdr_mean": float(S[i].mean()),
               "image_si_sdr_mean": float(I[i].mean()),
               "measurement_residual": float(arrays["measurement_residual"][i])}
        for k in range(ds.num_sources):
            row[f"si_sdr_source_{k+1}"] = float(S[i, k])
            row[f"nmse_source_{k+1}"] = float(arrays["nmse"][i, k])
        rows.append(row)
    with (destination / "per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Latent SI-SDR={S.mean():.4f} dB; image SI-SDR={I.mean():.4f} dB; "
          f"failure<5 dB={(S<5).mean():.2%}\nSaved: {destination}", flush=True)
    return destination, summary
