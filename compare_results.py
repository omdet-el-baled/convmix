"""Paired differences, conditional on these trained weights (not seed uncertainty)."""
from pathlib import Path
import argparse
import numpy as np
from src.utils import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--left',type=Path,required=True)
    p.add_argument('--right',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('results/paired_comparison.json'))
    a=p.parse_args()
    with np.load(a.left/'per_example.npz',allow_pickle=False) as z:
        li=z['indices']; left=z['si_sdr']; linit=z['initial_state_sha256']
    with np.load(a.right/'per_example.npz',allow_pickle=False) as z:
        ri=z['indices']; right=z['si_sdr']; rinit=z['initial_state_sha256']
    if not np.array_equal(li,ri):raise ValueError('Example IDs/order differ')
    if left.shape!=right.shape:raise ValueError('Source dimensions differ')
    delta=(right-left).mean(1)
    rng=np.random.default_rng(11001)
    means=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
    result={'definition':'right minus left, mean SI-SDR over ordered sources per example',
        'num_examples':len(delta),'mean_difference_db':float(delta.mean()),
        'median_difference_db':float(np.median(delta)),
        'fraction_positive':float((delta>0).mean()),
        'example_bootstrap_95_percent_interval_db':np.quantile(means,[.025,.975]).tolist(),
        'identical_initial_state_hashes':bool(np.array_equal(linit,rinit)),
        'uncertainty_scope':'example resampling conditional on these weights, not training-seed uncertainty'}
    atomic_json(a.out,result);print(result)

if __name__=='__main__':main()
