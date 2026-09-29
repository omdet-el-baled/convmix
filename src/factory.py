"""Construct components once for train/evaluate/diagnostics; no duplicated formulas."""
from __future__ import annotations

import copy
import numpy as np
import torch

from src.config import load_config, project_path, validate_config
from src.data.filters import load_filters
from src.models.unet import ScoreUNet, BasicScoreUNet
from src.pl_model import DCSModel
from src.sde.freqsde import ConvMixSDE
from src.utils import load_checkpoint
from src.components import build_component


def make_kernels(filters, n_fft, dtype=torch.complex64):
    return torch.as_tensor(np.stack([np.fft.rfft(h, n=n_fft) for h in filters]), dtype=dtype)


def make_model(config: dict, *, n_fft: int, kernels: torch.Tensor) -> DCSModel:
    validate_config(config)
    K, F = kernels.shape
    if K != config["data"]["num_sources"] or F != n_fft // 2 + 1:
        raise ValueError("Configured source count / FFT length differ from dataset")
    components = config.get("components", {})
    sde = make_sde(config, n_fft=n_fft, kernels=kernels)
    net_cfg = dict(config["network"])
    architecture = net_cfg.pop("architecture", "enhanced")
    if components:
        net_spec = components["score_model"]
        network = build_component(net_spec, num_sources=K, n_fft=n_fft, **net_cfg)
        extra = {name: components.get(name) for name in (
            "optimizer", "scheduler", "loss", "time_sampler", "validation_time_sampler",
            "noise_sampler", "ema_update", "loss_selection")}
        # The noise function is a deferred function call (_partial_: true).
        if extra["noise_sampler"] is not None:
            extra["noise_sampler"] = build_component(extra["noise_sampler"])
        model = build_component(components["model"], score_model=network, sde=sde,
                                n_fft=n_fft, **config["model"], **extra)
    else:
        # Compatibility only; new Hydra workflows always specify _target_.
        legacy_targets = {"enhanced": ScoreUNet, "basic": BasicScoreUNet}
        if architecture not in legacy_targets:
            raise ValueError("Legacy network.architecture must be enhanced or basic; use Hydra _target_ for new models")
        network = legacy_targets[architecture](num_sources=K, n_fft=n_fft, **net_cfg)
        model = DCSModel(network, sde, n_fft=n_fft, **config["model"])
    model.run_config = copy.deepcopy(config)
    return model


def load_model_from_checkpoint(checkpoint_path, *, device="cpu", trust=False,
                               legacy_config=None, data_dir=None):
    payload = load_checkpoint(checkpoint_path, trust=trust)
    embedded = payload.get("convmix", {})
    cfg = embedded.get("config")
    metadata = embedded.get("metadata")
    if cfg is None:
        if legacy_config is None:
            raise ValueError("This older checkpoint has no full model/SDE config. Supply --legacy-config "
                             "with the EXACT config used for that checkpoint; do not guess SDE parameters.")
        cfg = load_config(legacy_config)
        from src.dataset import MixDataset
        ds = MixDataset(project_path(data_dir or cfg["data"]["data_dir"]) / "test.npz",
                        fft_norm=cfg["data"]["fft_norm"])
        filters_path = project_path(cfg["data"]["filters"]) if cfg["data"]["filters"] else ds.path.parent / "filters.npz"
        kernels = make_kernels(load_filters(filters_path), ds.n_fft)
        n_fft = ds.n_fft
    else:
        cfg = copy.deepcopy(cfg)
        kernels = embedded["kernels"]
        n_fft = int(metadata["n_fft"])
    if cfg.get("components") and not trust:
        raise ValueError("This checkpoint contains Python _target_ configuration. Pass --trust-checkpoint "
                         "(or evaluation.trust_checkpoint=true) only for checkpoints you created and trust.")
    model = make_model(cfg, n_fft=n_fft, kernels=kernels)
    state = dict(payload["state_dict"])
    if legacy_config is not None and "ema_updates" not in state:
        state["ema_updates"] = torch.tensor(0, dtype=torch.long)
    model.load_state_dict(state, strict=True)  # architecture mismatches fail loudly
    model.data_metadata = metadata
    model.to(device).eval()
    model.sde.send_to(device)
    return model, cfg, metadata


def make_sde(config, *, n_fft, kernels):
    K, F = kernels.shape
    spec = config.get("components", {}).get("sde")
    sde = build_component(spec, ConvMixSDE, ndim=K*F, **config["sde"])
    sde.get_mix_mat(kernels, N_fft=F)
    sde.set_rfft(n_fft)
    return sde


def make_datamodule(config):
    from src.dataset import MixData_Module
    d = config["data"]
    spec = config.get("components", {}).get("datamodule")
    if spec:
        return build_component(spec, data_dir=project_path(d["data_dir"]), seed=config["seed"])
    return MixData_Module(project_path(d["data_dir"]), batch_size=d["batch_size"], num_workers=d["num_workers"],
        fft_norm=d["fft_norm"], pin_memory=d["pin_memory"], persistent_workers=d["persistent_workers"],
        seed=config["seed"], drop_last=d["drop_last"], train_subset=d["train_subset"])


def make_dataset(config, split, *, load_time_domain=False):
    from src.dataset import MixDataset
    spec = config.get("components", {}).get("datamodule", {}).get("dataset")
    return build_component(spec, MixDataset, path=project_path(config["data"]["data_dir"])/f"{split}.npz",
        fft_norm=config["data"]["fft_norm"], load_time_domain=load_time_domain)


def make_sampler(config):
    from functools import partial
    from src.sampling.samplers import SamplingConfig, sample
    spec = copy.deepcopy(config.get("components", {}).get("sampler"))
    if spec is None:
        return partial(sample, config=SamplingConfig(**config["sampling"]))
    scalar = {k:v for k,v in config["sampling"].items() if not k.startswith("corrector_")}
    # Scalar reporting controls stay synchronized with the corrector factory.
    if spec.get("corrector", {}).get("_target_", "").endswith("LangevinCorrector"):
        names = {"corrector_steps":"num_steps", "corrector_snr":"snr", "corrector_geometry":"geometry",
                 "corrector_step_size":"step_size", "corrector_max_step":"max_step"}
        for old,new in names.items(): spec["corrector"][new] = config["sampling"][old]
    return build_component(spec, **scalar)


def make_metrics(config):
    spec = config.get("components", {}).get("metrics")
    from src.metric_suite import OrderedSourceMetrics
    return build_component(spec, OrderedSourceMetrics)


def edit_sampler(config, edits):
    """Apply explicit native eval overrides to the checkpoint's sampler recipe."""
    from src.hydra_config import compose_config, to_runtime_config, split_sampler
    spec = copy.deepcopy(config.get("components", {}).get("sampler"))
    if spec is None:
        # Compatibility bridge for pre-Hydra checkpoints, not model reconstruction.
        recipe = to_runtime_config(compose_config([f"model/sampler={config['sampling']['solver']}"]))
        spec = copy.deepcopy(recipe["components"]["sampler"])
    spec.update({k:v for k,v in config["sampling"].items() if not k.startswith("corrector_")})
    if spec.get("corrector", {}).get("_target_", "").endswith("LangevinCorrector"):
        names = {"corrector_steps":"num_steps", "corrector_snr":"snr", "corrector_geometry":"geometry",
                 "corrector_step_size":"step_size", "corrector_max_step":"max_step"}
        for old,new in names.items():
            if old in config["sampling"]:
                spec["corrector"][new] = config["sampling"][old]
    for key, value in edits:
        parts = key.split(".")
        node = spec
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"Sampler override has no configured parent: {key}")
            node = node[part]
        node[parts[-1]] = copy.deepcopy(value)
    component, sampling = split_sampler(spec)
    config.setdefault("components", {})["sampler"] = component
    config["sampling"] = sampling
