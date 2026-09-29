"""Collect native Hydra evaluations without averaging different architectures.

Usage:
    python collect_results.py --root results --out sweeps/native_summary
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from src.config import config_hash, project_path
from src.sweeps import collect_row, write_rows, aggregate


def collect(root, out):
    rows, seen = [], set()
    for file in sorted(project_path(root).rglob('summary.json')):
        protocol_file=file.parent/'protocol.json'
        if not protocol_file.exists():
            continue
        summary=json.loads(file.read_text());protocol=json.loads(protocol_file.read_text())
        if 'training_config' not in summary:
            continue
        uid=summary['protocol_id']
        if uid in seen:
            continue
        seen.add(uid)
        cfg=summary['training_config']
        # Group over seeds, but never merge different network constructors,
        # losses, time samplers, optimization budgets, data, or sampler recipes.
        identity={k:copy.deepcopy(cfg[k]) for k in ('sde','network','model','data','trainer')}
        identity['components']={k:v for k,v in cfg.get('components',{}).items()
            if k not in ('logger','callbacks','trainer','sampler','metrics')}
        if 'datamodule' in identity['components']:
            identity['components']['datamodule'] = copy.deepcopy(identity['components']['datamodule'])
            identity['components']['datamodule'].pop('seed', None)
        identity['sampling']=protocol['sampling']
        identity['sampler_components']=protocol.get('sampler_components')
        identity['metric_components']=protocol.get('metric_components')
        identity['test_sha256']=protocol['split_sha256']
        identity['indices']=protocol['indices']
        row=collect_row(summary,protocol['sampling'],file.parent)
        row['experiment_id']=config_hash(identity)[:16]
        row['sampling_seed']=protocol['sampling']['seed']
        rows.append(row)
    if not rows:
        raise FileNotFoundError('No completed evaluation summary.json/protocol.json pairs found')
    out=project_path(out)
    write_rows(rows,out/'results_per_run.csv')
    write_rows(aggregate(rows),out/'results_aggregate.csv')
    print(f'Collected {len(rows)} evaluations into {out}')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='results')
    parser.add_argument('--out',default='sweeps/native_summary')
    args=parser.parse_args()
    collect(args.root,args.out)
