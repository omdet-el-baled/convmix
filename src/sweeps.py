"""Explicit experiment plans. Print commands by default; run only with --execute.

A changed epoch budget creates a different run identity. Existing evaluation
outputs are reused only after checking the checkpoint/data/code/protocol hashes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path

import yaml

from src.config import load_config, merge_dict, project_path, set_value, config_hash
from src.training import train
from src.evaluation import evaluate
from src.utils import atomic_json, file_hash, source_tree_hash

METRICS=("latent_si_sdr_db_mean","source_image_si_sdr_db_mean","nmse_mean","nmae_mean",
         "failure_rate_lt_5db","measurement_residual_mean","mean_nfe")


def grid_updates(spec):
    """Cartesian grid, or a list of exact cases; do not multiply coupled cases."""
    if 'cases' in spec:
        yield from spec['cases']
    else:
        grid=spec.get('grid',{})
        for values in itertools.product(*grid.values()):
            yield dict(zip(grid.keys(),values))


def training_identity(cfg):
    result=copy.deepcopy(cfg)
    for key in ('resume','trust_checkpoint'):
        result['run'][key]=None
    # Evaluation settings are independent of a checkpoint's training identity.
    result.pop('sampling',None);result.pop('evaluation',None)
    return config_hash(result)


def write_rows(rows,path):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def collect_row(summary, sampling, output):
    cfg=summary['training_config']
    return dict(seed=cfg['seed'],alpha=cfg['sde']['alpha'],beta=cfg['sde']['beta'],
        mismatch_probability=cfg['model']['mismatch_probability'],mismatch_input=cfg['model']['mismatch_input'],
        drift_mode=cfg['sde']['drift_mode'],time_sampling_strategy=cfg['model']['time_sampling_strategy'],
        architecture=cfg['network']['architecture'],num_sources=cfg['data']['num_sources'],kind=cfg['data']['kind'],
        solver=sampling['solver'],steps=sampling['num_steps'],weights=sampling['weights'],
        corrector_steps=sampling['corrector_steps'],corrector_snr=sampling['corrector_snr'],
        **{k:summary[k] for k in METRICS},output=str(output))


def aggregate(rows):
    import numpy as np
    identity_keys=[k for k in rows[0] if k not in set(METRICS)|{'seed','output'}] if rows else []
    groups={}
    for row in rows:
        key=tuple(row[k] for k in identity_keys)
        groups.setdefault(key,[]).append(row)
    output=[]
    for key,group in groups.items():
        value=dict(zip(identity_keys,key));value['training_runs']=len(group)
        for metric in METRICS:
            arr=np.array([r[metric] for r in group],float)
            value[metric]=float(arr.mean())
            value[metric+'_seed_std']=float(arr.std(ddof=1)) if len(arr)>1 else None
        output.append(value)
    return output


def run_sweep(path, *, execute=False, checkpoints=None, resume_incomplete=False, force_eval=False):
    path=project_path(path)
    spec=yaml.safe_load(path.read_text())
    root=project_path(spec.get('output_dir','sweeps/'+path.stem))
    mode=spec.get('mode','train')
    cases=spec.get('evaluation_cases',[{}])
    jobs=[]
    if mode=='train':
        base=load_config(path.parent/spec['base_config'])
        base=merge_dict(base,spec.get('common',{}))
        for update in grid_updates(spec):
            for seed in spec.get('seeds',[base['seed']]):
                cfg=copy.deepcopy(base)
                for k,v in update.items():set_value(cfg,k,v)
                cfg['seed']=int(seed)
                cfg['run']['resume']=None
                cfg['run']['output_dir']=str(root/'runs')
                cfg['run']['name']='pending'
                short=training_identity(cfg)[:10]
                cfg['run']['name']=(f"a{cfg['sde']['alpha']:g}_b{cfg['sde']['beta']:g}"
                    f"_p{cfg['model']['mismatch_probability']:g}_seed{seed}_{short}")
                run_dir=project_path(cfg['run']['output_dir'])/cfg['run']['name']
                jobs.append((cfg,run_dir,None))
    elif mode=='evaluate':
        for checkpoint in checkpoints or spec.get('checkpoints',[]):
            checkpoint=project_path(checkpoint)
            if not checkpoint.exists():raise FileNotFoundError(checkpoint)
            jobs.append((None,None,checkpoint))
        if not jobs:raise ValueError('Supply --checkpoints CKPT [CKPT ...] for an evaluation-only sweep')
    else:raise ValueError('mode must be train or evaluate')
    print(f"Plan: {len(jobs)} model(s), {len(cases)} sampler case(s) each")
    for cfg,run_dir,checkpoint in jobs:
        print('TRAIN '+str(run_dir) if cfg else 'CHECKPOINT '+str(checkpoint))
        for case in cases:print('  EVALUATE',case)
    if not execute:
        print('Dry run only. Add --execute to run this plan.');return
    root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'plan.json',{'spec':spec,'source_tree_sha256':source_tree_hash(),
                                'checkpoints':[str(p) for p in checkpoints or []]})
    rows=[]
    for cfg,run_dir,checkpoint in jobs:
        if cfg is not None:
            resolved=run_dir/'resolved.yaml'
            if resolved.exists():
                previous=yaml.safe_load(resolved.read_text())
                if training_identity(previous)!=training_identity(cfg):
                    raise RuntimeError(f'Training config mismatch for existing run {run_dir}')
            done=run_dir/'artifacts.json'
            if done.exists():
                manifest=json.loads((run_dir/'manifest.json').read_text())
                if manifest.get('source_tree_sha256') != source_tree_hash():
                    raise RuntimeError('Training source changed. Use a new sweep output_dir, or an evaluation-only sweep to reuse weights deliberately.')
                print('Reuse completed, matching run:',run_dir)
            elif (run_dir/'checkpoints/last.ckpt').exists():
                if not resume_incomplete:raise RuntimeError(f'Incomplete run {run_dir}; pass --resume-incomplete')
                cfg['run']['resume']=str(run_dir/'checkpoints/last.ckpt')
                train(cfg)
            else:
                train(cfg)
            artifacts=json.loads(done.read_text())
            policy=spec.get('checkpoint_policy','last')
            candidate=Path(artifacts[policy])
            checkpoint=candidate if candidate.exists() else run_dir/'checkpoints'/candidate.name
        checkpoint_sha=file_hash(checkpoint)
        for case in cases:
            overrides=[f"{key}={yaml.safe_dump(value,default_flow_style=True).strip().removesuffix('...').strip()}"
                       for key,value in case.items()]
            case_hash=config_hash({'checkpoint':checkpoint_sha,'case':case,'code':source_tree_hash()})[:12]
            output=root/'evaluations'/(checkpoint.parent.parent.name+'_'+case_hash)
            summary_path=output/'summary.json'
            if summary_path.exists() and not force_eval:
                summary=json.loads(summary_path.read_text())
                protocol=json.loads((output/'protocol.json').read_text())
                if protocol['checkpoint_sha256']!=checkpoint_sha or protocol['source_tree_sha256']!=source_tree_hash():
                    raise RuntimeError('Evaluation provenance changed; use --force-eval')
                if file_hash(protocol['dataset_path'])!=protocol['split_sha256']:
                    raise RuntimeError('Evaluation data changed; use a new experiment directory')
                sampling=protocol['sampling']
            else:
                output,summary=evaluate(checkpoint,overrides=overrides,out=output,force=force_eval)
                sampling=json.loads((output/'protocol.json').read_text())['sampling']
            rows.append(collect_row(summary,sampling,output))
            write_rows(rows,root/'results_per_run.csv')
            write_rows(aggregate(rows),root/'results_aggregate.csv')
    from src.plotting import plot_sweep
    plot_sweep(root/'results_per_run.csv')
    print('Saved:',root/'results_per_run.csv',root/'results_aggregate.csv')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--checkpoints',nargs='*')
    p.add_argument('--resume-incomplete',action='store_true')
    p.add_argument('--force-eval',action='store_true')
    a=p.parse_args()
    run_sweep(a.config,execute=a.execute,checkpoints=a.checkpoints,
              resume_incomplete=a.resume_incomplete,force_eval=a.force_eval)
