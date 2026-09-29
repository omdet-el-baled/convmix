"""Hydra composition -> the existing workflow's plain-dict config contract.

This adapter changes object construction, not numerical routines. All targets
and constructor arguments survive into checkpoints. Legacy configs remain valid.
"""
from __future__ import annotations
import copy
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from src.config import PROJECT_ROOT, validate_config


def compose_config(overrides=None, config_name='config', config_dir=None):
    from hydra import compose, initialize_config_dir
    directory = Path(config_dir or PROJECT_ROOT / 'config').resolve()
    with initialize_config_dir(version_base='1.3', config_dir=str(directory)):
        return compose(config_name=config_name, overrides=list(overrides or []))


def _split(spec):
    if not isinstance(spec, dict) or '_target_' not in spec:
        raise ValueError('Each constructible component must declare _target_')
    return ({k:v for k,v in spec.items() if k.startswith('_')},
            {k:v for k,v in spec.items() if not k.startswith('_')})



def split_sampler(sampler):
    """Separate nested factories from scalar reporting/control fields."""
    meta, _ = _split(sampler)
    nested = ('predictor', 'corrector', 'grid', 'initialization', 'noise')
    meta.update({k:sampler[k] for k in nested if k in sampler})
    sampling = {k:v for k,v in sampler.items() if not k.startswith('_') and k not in nested}
    corrector = sampler.get('corrector') or {}
    for name, value in {
        'corrector_steps': corrector.get('num_steps',0),
        'corrector_snr': corrector.get('snr',0.16),
        'corrector_geometry': corrector.get('geometry','diffusion'),
        'corrector_step_size': corrector.get('step_size'),
        'corrector_max_step': corrector.get('max_step',1.0),
    }.items(): sampling.setdefault(name,value)
    return meta, sampling

def to_runtime_config(cfg):
    """Keep proven workflow functions, instantiate their components from Hydra."""
    c = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) if OmegaConf.is_config(cfg) else copy.deepcopy(cfg)
    model = dict(c['model'])
    model_meta, _ = _split(model)
    score = model.pop('score_model'); sde = model.pop('sde')
    sampler = model.pop('sampler')
    components = {'model': model_meta}
    for name in ('optimizer','scheduler','loss','time_sampler','validation_time_sampler','noise_sampler','ema_update','loss_selection'):
        components[name] = model.pop(name, None)
    scheduler = components.get('scheduler')
    if scheduler is not None:
        scheduler = dict(scheduler)
        components['scheduler'] = scheduler if scheduler.pop('enabled', True) else None
    components['score_model'], network = _split(score)
    components['sde'], sde_values = _split(sde)
    components['sampler'], sampling = split_sampler(sampler)
    model_values = {k:v for k,v in model.items() if not k.startswith('_')}
    components.update({k:c[k] for k in ('datamodule','callbacks','logger','generation','metrics')})
    trainer_meta, trainer_values = _split(c['trainer'])
    components['trainer'] = trainer_meta
    # Budget/checkpoint fields used by the existing explicit experiment runner.
    trainer_values.setdefault('save_top_k', (c['callbacks'].get('checkpoint') or {}).get('save_top_k',1))
    trainer_values.setdefault('early_stopping_patience',0)
    result = {
        'seed':c['seed'], 'run':c['run'], 'data':c['data'],
        'network':{'architecture':score['_target_'], **network},
        'sde':sde_values, 'model':model_values, 'trainer':trainer_values,
        'sampling':sampling, 'evaluation':c['evaluation'], 'components':components, 'hydra_config':copy.deepcopy(c),
    }
    validate_config(result)
    return result
