"""Aggregate sweep CSVs into per-metric figures; no reference/energy normalization."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_sweep(results, out=None):
    results=Path(results)
    with results.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:return
    out=Path(out) if out else results.parent/'plots'
    out.mkdir(parents=True,exist_ok=True)
    group_fields=['alpha','beta','mismatch_probability','mismatch_input','drift_mode',
                  'time_sampling_strategy','architecture','num_sources','kind','solver','weights',
                  'corrector_steps','corrector_snr']
    varying=[k for k in group_fields if len({r.get(k,'') for r in rows})>1]
    short={'mismatch_probability':'p','mismatch_input':'input','time_sampling_strategy':'time',
           'architecture':'net','num_sources':'K','corrector_steps':'C','corrector_snr':'snr'}
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        label=', '.join(f'{short.get(k,k)}={row[k]}' for k in varying) or 'configured model'
        groups[label][float(row['mean_nfe'])].append(row)
    metrics=[('latent_si_sdr_db_mean','Latent SI-SDR (dB)'),
             ('source_image_si_sdr_db_mean','Source-image SI-SDR (dB)'),
             ('measurement_residual_mean','Measurement residual'),('nmse_mean','nMSE')]
    for metric,ylabel in metrics:
        fig,ax=plt.subplots(figsize=(9,5))
        if any(len(by_nfe)>1 for by_nfe in groups.values()):
            for label,by_nfe in sorted(groups.items()):
                xs=sorted(by_nfe)
                values=[np.array([float(r[metric]) for r in by_nfe[n]]) for n in xs]
                ys=[v.mean() for v in values]
                errs=[v.std(ddof=1) if len(v)>1 else 0 for v in values]
                ax.errorbar(xs,ys,yerr=errs,marker='o',label=label)
            ax.set(xlabel='Actual network evaluations (NFE)',ylabel=ylabel)
            ax.legend(fontsize=7)
        else:
            labels=[];means=[];errs=[]
            for label,by_nfe in sorted(groups.items()):
                n,items=next(iter(by_nfe.items()))
                vals=np.array([float(r[metric]) for r in items])
                labels.append(f'{label}; NFE={n:g}')
                means.append(vals.mean());errs.append(vals.std(ddof=1) if len(vals)>1 else 0)
            ax.barh(np.arange(len(labels)),means,xerr=errs)
            ax.set_yticks(np.arange(len(labels)),labels,fontsize=7)
            ax.set_xlabel(ylabel)
        ax.grid(True,alpha=.25)
        ax.set_title('Sweep comparison (error bars: training-seed SD where repeated)')
        fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=160);plt.close(fig)
    print('Saved sweep figures:',out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',default=None)
    a=p.parse_args();plot_sweep(a.results,a.out)
