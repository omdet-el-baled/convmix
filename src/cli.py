"""Native Hydra entry points; one composition mechanism for every workflow.

The old --config/--set entry points are retained separately for exact legacy
configs. This module requires hydra-core and does not implement a substitute
configuration parser. Instantiate only trusted target-bearing configs.
"""
from __future__ import annotations

import copy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml

from src.config import config_hash
from src.hydra_config import to_runtime_config


# The decorated functions are imported from ``src.cli`` by the root scripts.
# A relative config_path is consequently resolved by Hydra as a Python package
# (pkg://config), whereas this project keeps YAML in a filesystem directory.
# Derive an absolute directory from this file, not from the process working dir.
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def _runtime(cfg: DictConfig, *, unique_multirun: bool = False) -> dict:
    cfg = copy.deepcopy(cfg)
    if unique_multirun and HydraConfig.initialized():
        hc = HydraConfig.get()
        if hc.mode == RunMode.MULTIRUN:
            plain = OmegaConf.to_container(cfg, resolve=True)
            suffix = config_hash(plain)[:8]
            with open_dict(cfg):
                cfg.run.name = f"{cfg.run.name}_job{hc.job.num}_seed{cfg.seed}_{suffix}"
    return to_runtime_config(cfg)


def _task_override_keys() -> list[str]:
    """Hydra has already parsed/applied the overrides; inspect only their keys."""
    if not HydraConfig.initialized():
        return []
    return [item.split('=', 1)[0].lstrip('+~')
            for item in HydraConfig.get().overrides.task]


def _changed(keys, prefix):
    return any(k == prefix or k.startswith(prefix + '.') or k.startswith(prefix + '/') for k in keys)


def evaluation_options(cfg: DictConfig, keys: list[str]) -> dict:
    """Do not silently replace a checkpoint's learned model with CLI defaults.

    Only explicitly requested sampling/metrics/evaluation controls are applied.
    Full model/SDE constructor specs and kernels are restored from checkpoint.
    """
    protected = ('experiment', 'model/score_model', 'model.score_model', 'model/sde', 'model.sde',
                 'model/optimizer', 'model.optimizer', 'model/loss', 'model.loss',
                 'model.t_eps', 'datamodule')
    if any(_changed(keys, key) for key in protected):
        raise ValueError('Evaluation restores model/SDE/data-loader identity from the checkpoint. '
                         'Do not set experiment/model/score_model/model/sde here; select only '
                         'model/sampler, sampling controls, metrics, evaluation.*, data.data_dir, out, device.')
    c = _runtime(cfg)
    allowed_model = ('model.sampler.', 'model/sampler')
    for key in keys:
        if key.startswith('model') and not any(key == prefix.rstrip('.') or key.startswith(prefix) for prefix in allowed_model):
            raise ValueError(f'Unsupported evaluation model override: {key}')
        if key.startswith('data.') and key != 'data.data_dir':
            raise ValueError('Only data.data_dir can change at evaluation; FFT/source count come from checkpoint.')
    scalar_overrides = []
    for key in keys:
        if key.startswith('evaluation.') or key == 'data.data_dir':
            value = OmegaConf.select(cfg, key)
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            encoded = yaml.safe_dump(value, default_flow_style=True).replace('\n...', '').strip()
            scalar_overrides.append(f'{key}={encoded}')
    components = {}
    if 'model/sampler' in keys:
        # Explicit full recipe replacement: all sampler settings come from this group.
        components.update(sampler=c['components']['sampler'], sampling=c['sampling'])
    else:
        # An NFE-only override must NOT reset a checkpoint's PC/Heun/custom sampler
        # to the CLI's default Euler recipe. Patch only explicitly selected paths.
        edits = []
        for key in keys:
            if key.startswith('model.sampler.'):
                relative = key[len('model.sampler.'):]
            elif key.startswith('model/sampler/'):
                relative = key[len('model/sampler/'):].replace('/', '.')
            else:
                continue
            value = OmegaConf.select(cfg, 'model.sampler.' + relative)
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            edits.append((relative, value))
        if edits:
            components['sampler_edits'] = edits
    if _changed(keys, 'metrics'):
        components['metrics'] = c['components']['metrics']
    if not cfg.checkpoint:
        raise ValueError('Supply checkpoint=/path/to/checkpoint.ckpt')
    return dict(checkpoint_path=cfg.checkpoint, overrides=scalar_overrides,
                out=cfg.out, device=cfg.device, force=cfg.force,
                trust=cfg.evaluation.trust_checkpoint,
                component_overrides=components or None)


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def train_main(cfg: DictConfig) -> None:
    from src.training import train
    runtime = _runtime(cfg, unique_multirun=True)
    run_dir = train(runtime)
    if cfg.actions.evaluate_after_training:
        from src.evaluation import evaluate
        # This exact checkpoint was just written by this call from the user-supplied
        # config, so its target metadata are not a new untrusted source.
        evaluate(run_dir / 'checkpoints/last.ckpt', device=cfg.device, trust=True)


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def generate_main(cfg: DictConfig) -> None:
    from src.components import build_component
    runtime = _runtime(cfg)
    generator = build_component(runtime['components']['generation'])
    print('Saved dataset:', generator(runtime, force=cfg.force))


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def evaluate_main(cfg: DictConfig) -> None:
    from src.evaluation import evaluate
    evaluate(**evaluation_options(cfg, _task_override_keys()))


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def separate_main(cfg: DictConfig) -> None:
    from src.evaluation import evaluate
    from src.visualization import plot_separation
    options = evaluation_options(cfg, _task_override_keys())
    options['overrides'] = [o for o in options['overrides'] if not o.startswith('evaluation.save_estimates=')]
    options['overrides'].append('evaluation.save_estimates=true')
    out, summary = evaluate(**options, indices=[int(cfg.index)])
    plot_separation(out, summary['training_config'], int(cfg.index))


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def visualize_main(cfg: DictConfig) -> None:
    from src.visualization import visualize_dataset
    visualize_dataset(_runtime(cfg), index=cfg.index, out=cfg.out)


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def mean_main(cfg: DictConfig) -> None:
    from src.diagnostics import forward_mean
    forward_mean(_runtime(cfg), index=cfg.index, out=cfg.out or 'figures/forward_mean',
                 num_times=cfg.diagnostics.num_times, time_max=cfg.diagnostics.time_max)


@hydra.main(version_base='1.3', config_path=CONFIG_DIR, config_name='config')
def diagnose_main(cfg: DictConfig) -> None:
    from src.diagnostics import forward_mean, fixed_time, covariance_benchmark
    if cfg.diagnostics.task == 'score':
        if not cfg.checkpoint:
            raise ValueError('Supply checkpoint=/path/to/model.ckpt for score diagnostics')
        fixed_time(cfg.checkpoint, times=list(cfg.diagnostics.times), max_examples=cfg.diagnostics.max_examples,
                   out=cfg.out or 'results/score_time', device=cfg.device, trust=cfg.evaluation.trust_checkpoint)
    elif cfg.diagnostics.task == 'mean':
        forward_mean(_runtime(cfg), index=cfg.index, out=cfg.out or 'figures/forward_mean',
                     num_times=cfg.diagnostics.num_times, time_max=cfg.diagnostics.time_max)
    elif cfg.diagnostics.task == 'covariance':
        # Explicit mathematical benchmark, not a trainable component.
        covariance_benchmark(out=cfg.out or 'results/covariance', bins=tuple(cfg.diagnostics.bins),
                             steps=cfg.diagnostics.covariance_steps)
    else:
        raise ValueError('diagnostics.task must be mean, score, or covariance')
