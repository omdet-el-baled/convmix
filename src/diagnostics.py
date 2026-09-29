"""Forward mean, fixed-time DSM, and covariance benchmarks (not quality claims)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from src.config import load_config, project_path
from src.data.filters import load_filters
from src.dataset import MixDataset
from src.factory import make_kernels, make_model, load_model_from_checkpoint, make_dataset, make_sde
from src.sde.freqsde import ConvMixSDE
from src.sde.rfft_noise import standard_rfft_noise_like
from src.utils import atomic_json, atomic_npz, choose_device
from src.visualization import save_lines


def forward_mean(cfg, index=0, out="figures/forward_mean", num_times=101, time_max=None):
    data_dir = project_path(cfg["data"]["data_dir"])
    ds = make_dataset(cfg, "cv", load_time_domain=True)
    if not 0 <= index < len(ds):
        raise IndexError(index)
    filters = load_filters(project_path(cfg["data"]["filters"]) if cfg["data"]["filters"] else data_dir / "filters.npz")
    kernels = make_kernels(filters, ds.n_fft, torch.complex128)
    sde = make_sde(cfg, n_fft=ds.n_fft, kernels=kernels)
    sources = ds.sources[index].astype(np.float64)
    # Independent target: actual linear time convolution, NOT apply_A(x0).
    clean = np.zeros(ds.n_fft)
    for s, h in zip(sources, filters):
        conv = np.convolve(s, h, mode="full")
        clean[:len(conv)] += conv
    x0 = torch.as_tensor(np.fft.rfft(sources, n=ds.n_fft, norm=ds.fft_norm)).reshape(1, -1)
    target = torch.as_tensor(np.fft.rfft(clean, norm=ds.fft_norm))[None].expand(ds.num_sources, -1).reshape(1, -1)
    # Mean equation is defined for all t>=0; extending time does not change T
    # or the noise schedule. We do not claim stochastic x_t converges to y.
    times = np.linspace(0, sde.T if time_max is None else time_max, num_times)
    means = torch.cat([sde._mean(x0, float(t)) for t in times]).reshape(num_times, ds.num_sources, -1)
    error = torch.linalg.vector_norm(means.reshape(num_times, -1)-target, dim=-1).numpy()
    convolution_error = float((sde.A @ x0[..., None]).squeeze(-1).sub(target).abs().max())
    out = project_path(out)
    save_lines(out / "distance", times, [("raw Euclidean distance", error)],
        xlabel="Diffusion time t", ylabel="Raw ||mean - FFT(time-convolved mixture)||", title="Forward mean convergence")
    signal_t = np.arange(ds.n_fft) / ds.sample_rate
    for j in np.unique(np.linspace(0, num_times-1, 6).astype(int)):
        wave = np.fft.irfft(means[j].numpy(), n=ds.n_fft, norm=ds.fft_norm)
        save_lines(out / f"waveform_t{times[j]:.4f}", signal_t,
            [("Clean mixture", clean)] + [(f"mean channel {k+1}", wave[k]) for k in range(ds.num_sources)],
            xlabel="Signal time (s)", ylabel="Amplitude", title=f"Mean at diffusion time {times[j]:.4f}")
    atomic_npz(out / "trajectory.npz", times=times, mean=means.numpy(), raw_error=error,
               y_true_time=clean, target_fft=target.numpy(), convolution_error=np.asarray(convolution_error))
    print(f"Time-convolution/operator discrepancy: {convolution_error:.4e}; terminal mean error {error[-1]:.4e}")


@torch.no_grad()
def fixed_time(checkpoint, *, times, max_examples=64, out="results/score_time", device="auto", trust=False):
    model, cfg, meta = load_model_from_checkpoint(project_path(checkpoint), device=choose_device(device), trust=trust)
    if model.sde.drift_mode == "instantaneous":
        raise NotImplementedError("This modal diagnostic is for mixing/none covariance factors")
    ds = make_dataset(cfg, "cv")
    count = min(len(ds), max_examples)
    rows = []
    for ti, t_value in enumerate(times):
        if not model.t_eps <= t_value <= model.sde.T:
            raise ValueError("Diagnostic time outside training interval")
        sums = np.zeros(7)
        elements = 0
        for start in range(0, count, 4):
            ids = list(range(start, min(start+4, count)))
            x = torch.stack([ds[i][0] for i in ids]).to(model.device)
            y = torch.stack([ds[i][1] for i in ids]).to(model.device)
            generator = torch.Generator(device=model.device).manual_seed(12345 + start + ti*100003)
            t = x.real.new_full((len(ids),), t_value)
            terms = model.loss_terms(x, y, t, use_ema=True, generator=generator)
            residual = terms["residual"].reshape_as(x)
            std = terms["std"].reshape_as(x)
            # This is V^H(score - conditional kernel target), NOT error against
            # the unknown marginal score. Its small-t growth has a Bayes floor.
            target_error = residual / std.conj()
            pf_proxy = 0.5 * model.sde.g_squared(t)[:, None, None] * target_error
            values = (residual.abs().square().mean((1,2)), residual[:,0].abs().square().mean(-1),
                residual[:,1:].abs().square().mean((1,2)), target_error[:,0].abs().square().mean(-1),
                target_error[:,1:].abs().square().mean((1,2)), pf_proxy[:,0].abs().square().mean(-1),
                pf_proxy[:,1:].abs().square().mean((1,2)))
            sums += np.array([float(v.sum()) for v in values])
            elements += len(ids)
        names = ["dsm_total", "dsm_nonzero", "dsm_null", "score_target_nonzero", "score_target_null",
                 "pf_target_proxy_nonzero", "pf_target_proxy_null"]
        rows.append({"t": t_value, "num_examples": elements, **dict(zip(names, (sums/elements).tolist()))})
    out = project_path(out); out.mkdir(parents=True, exist_ok=True)
    with (out / "fixed_time.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    for key in names:
        save_lines(out / key, times, [(key, [r[key] for r in rows])], xlabel="Diffusion time",
                   ylabel=key, title="Conditional-kernel target diagnostic (not exact score error)", log_y=True)
    print("Saved:", out)
    return rows


def covariance_benchmark(out="results/covariance", bins=(8, 16, 32), steps=1000):
    """CPU structured quadrature vs block/dense adaptive Lyapunov integration, with errors reported.

    The time integration methods differ; report accuracy with timing rather than
    interpreting a timing-only comparison as an equal-error speedup.
    """
    from scipy.integrate import solve_ivp
    results = []
    torch.set_num_threads(1)
    for F in bins:
        if F > 64:
            raise ValueError("Dense benchmark capped at 64 bins; use frequency blocks at larger sizes")
        omega = torch.linspace(0, 1, F, dtype=torch.float64)
        H = torch.stack([1.2 + .1j*omega, .7 - .07j*omega]).to(torch.complex128)
        sde = ConvMixSDE(2*F, 3, 2, .2, 5, steps, 1); sde.get_mix_mat(H, F)
        t0 = time.perf_counter(); factors = sde.factor_blocks(1.)[0]; computed = factors @ factors.mH
        structured_time = time.perf_counter()-t0
        V, Vinv = sde.V.numpy(), sde.V_inv.numpy()
        def drift_blocks(t):
            rates=sde.get_Gt(float(t))[0].reshape(2,F).T.numpy()
            return (V * rates[:,None,:]) @ Vinv
        diffusion = V @ V.conj().transpose(0,2,1)
        def rhs_blocks(t, q):
            q=q.reshape(F,2,2); drift=drift_blocks(t)
            return (drift@q+q@drift.conj().transpose(0,2,1)+float(sde.g_squared(t)[0])*diffusion).ravel()
        t0=time.perf_counter()
        ref=solve_ivp(rhs_blocks,(0,1),np.zeros(F*4,complex),rtol=1e-9,atol=1e-11).y[:,-1].reshape(F,2,2)
        block_time=time.perf_counter()-t0
        A_V, A_inv=sde.eigvecs.numpy(),sde.eigvecs_inv.numpy(); C=A_V@A_V.conj().T
        def rhs_dense(t,q):
            q=q.reshape(2*F,2*F)
            drift=(A_V*sde.get_Gt(t)[0].numpy()[None,:])@A_inv
            return (drift@q+q@drift.conj().T+float(sde.g_squared(t)[0])*C).ravel()
        t0=time.perf_counter()
        dense=solve_ivp(rhs_dense,(0,1),np.zeros((2*F)**2,complex),rtol=1e-9,atol=1e-11).y[:,-1]
        dense_time=time.perf_counter()-t0
        dense_ref=sde._blocks_to_dense(torch.tensor(ref)).numpy().ravel()
        results.append({"bins":F,"structured_seconds":structured_time,"block_lyapunov_seconds":block_time,
            "dense_lyapunov_seconds":dense_time,"quadrature_relative_error":float(np.linalg.norm(computed.numpy()-ref)/np.linalg.norm(ref)),
            "dense_block_relative_error":float(np.linalg.norm(dense-dense_ref)/np.linalg.norm(dense_ref))})
    atomic_json(project_path(out)/"benchmark.json", results)
    print(results)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    mean=sub.add_parser("mean"); mean.add_argument("--config",default="configs/base.yaml"); mean.add_argument("--set",nargs="*",default=[])
    mean.add_argument("--index",type=int,default=0); mean.add_argument("--out",default="figures/forward_mean")
    mean.add_argument("--num-times",type=int,default=101); mean.add_argument("--time-max",type=float,default=None)
    score=sub.add_parser("score"); score.add_argument("--checkpoint",required=True); score.add_argument("--times",default="0.005,0.01,0.02,0.05,0.1,0.2,0.5,1")
    score.add_argument("--max-examples",type=int,default=64); score.add_argument("--out",default="results/score_time")
    score.add_argument("--device",default="auto"); score.add_argument("--trust-checkpoint",action="store_true")
    cov=sub.add_parser("covariance"); cov.add_argument("--out",default="results/covariance"); cov.add_argument("--bins",default="8,16,32")
    a=p.parse_args()
    if a.command=="mean": forward_mean(load_config(a.config,a.set),a.index,a.out,a.num_times,a.time_max)
    elif a.command=="score": fixed_time(a.checkpoint,times=[float(x) for x in a.times.split(",")],max_examples=a.max_examples,out=a.out,device=a.device,trust=a.trust_checkpoint)
    else: covariance_benchmark(a.out,tuple(int(x) for x in a.bins.split(",")))
