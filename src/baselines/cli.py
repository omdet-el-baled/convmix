"""Native Hydra commands with absolute filesystem configuration paths."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from src.config import config_hash
from src.baselines.workflow import train, evaluate

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="convtasnet")
def train_main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if HydraConfig.initialized() and HydraConfig.get().mode == RunMode.MULTIRUN:
        job = HydraConfig.get().job.num
        config["run"]["name"] += f"_job{job}_{config_hash(config)[:8]}"
    train(config)


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="convtasnet")
def evaluate_main(cfg: DictConfig):
    if cfg.checkpoint is None:
        raise ValueError("Specify checkpoint=/path/to/checkpoints/best.ckpt")
    # Architecture, loss, normalization and Wiener regularizer come from the
    # checkpoint, NOT from this command's default training configuration.
    if HydraConfig.initialized():
        for item in HydraConfig.get().overrides.task:
            key = item.split("=", 1)[0].lstrip("+~")
            allowed = key in {"checkpoint", "device", "out", "force", "data.data_dir"} or key.startswith("evaluation.") or key.startswith("hydra.")
            if not allowed:
                raise ValueError(f"Unsupported evaluation override {key!r}. The trained separator "
                                 "and FFT convention are restored from the checkpoint.")
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    override_dir = None
    if HydraConfig.initialized() and any(
        v.split("=", 1)[0].lstrip("+") == "data.data_dir"
        for v in HydraConfig.get().overrides.task):
        override_dir = config["data"]["data_dir"]
    ev = config["evaluation"]
    evaluate(config["checkpoint"], trust=ev["trust_checkpoint"], device=config["device"],
             data_dir=override_dir, split=ev["split"], batch_size=ev["batch_size"],
             max_examples=ev["max_examples"], save_estimates=ev["save_estimates"],
             output_dir=ev["output_dir"], out=config["out"], force=config["force"])
