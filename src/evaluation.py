from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.config import project_path, set_value, config_hash
from src.data.filters import load_filters
from src.dataset import MixDataset
from src.factory import load_model_from_checkpoint, make_kernels, make_dataset, make_sampler, make_metrics, edit_sampler
from src.metrics import si_sdr, nmse, nmae
from src.sampling.samplers import SamplingConfig, sample
from src.utils import atomic_json, atomic_npz, file_hash, choose_device, source_tree_hash


def evaluate(checkpoint_path, *, overrides=None, out=None, device="auto", force=False,
             indices=None, trust=False, legacy_config=None, component_overrides=None):
    checkpoint_path = project_path(checkpoint_path)
    model, cfg, metadata = load_model_from_checkpoint(checkpoint_path, device=choose_device(device),
                                                      trust=trust, legacy_config=legacy_config)
    for override in overrides or []:
        key, text = override.split("=", 1)
        if key == "sampling.solver" and cfg.get("components"):
            raise ValueError("This is a component-configured checkpoint. Select model/sampler=pc "
                             "(or another group) with the Hydra CLI; changing only a label "
                             "does not replace the predictor/corrector.")
        if not (key.startswith("sampling.") or key.startswith("evaluation.") or key == "data.data_dir"):
            raise ValueError("Evaluation may override only sampling.*, evaluation.*, data.data_dir. "
                             "SDE/network settings are restored from the training checkpoint.")
        import yaml
        set_value(cfg, key, yaml.safe_load(text))
    if component_overrides:
        # This is an explicit trusted evaluation-only component selection.
        forbidden = set(component_overrides) - {"sampler", "sampling", "metrics", "sampler_edits"}
        if forbidden:
            raise ValueError(f"Cannot replace trained components during evaluation: {forbidden}")
        if "sampler" in component_overrides:
            cfg.setdefault("components", {})["sampler"] = component_overrides["sampler"]
        if "sampling" in component_overrides:
            cfg["sampling"] = component_overrides["sampling"]
        if "metrics" in component_overrides:
            cfg.setdefault("components", {})["metrics"] = component_overrides["metrics"]
        if "sampler_edits" in component_overrides:
            edit_sampler(cfg, component_overrides["sampler_edits"])
    data_dir = project_path(cfg["data"]["data_dir"])
    ev = cfg["evaluation"]
    ds = make_dataset(cfg, ev["split"], load_time_domain=True)
    if ds.n_fft != model.n_fft or ds.num_sources * ds.num_frequency_bins != model.sde.ndim:
        raise ValueError("Evaluation dimensions do not match trained model")
    if metadata and (ds.sample_rate != metadata["sample_rate"] or ds.source_length != metadata["source_length"]):
        raise ValueError("Sample rate/source length differs from training")
    filters_path = project_path(cfg["data"]["filters"]) if cfg["data"]["filters"] else data_dir / "filters.npz"
    filters = load_filters(filters_path)
    actual_kernels = make_kernels(filters, ds.n_fft)
    if not torch.allclose(actual_kernels, model.sde.kernels.cpu(), rtol=1e-6, atol=1e-7):
        raise ValueError("Evaluation filter bank differs from the checkpoint. Kernel-mismatch testing "
                         "must be an explicitly designed experiment, not an accidental override.")
    from types import SimpleNamespace
    sampling = SimpleNamespace(**cfg["sampling"])
    sampler = make_sampler(cfg)
    metrics = make_metrics(cfg)
    if indices is None:
        count = len(ds) if ev["max_examples"] is None else min(len(ds), int(ev["max_examples"]))
        indices = list(range(count))
    if not indices or any(i < 0 or i >= len(ds) for i in indices):
        raise ValueError("Invalid/empty evaluation indices")
    if ev["batch_size"] < 1:
        raise ValueError("evaluation.batch_size must be positive")
    checkpoint_sha = file_hash(checkpoint_path)
    protocol = {"source_tree_sha256": source_tree_hash(), "checkpoint_sha256": checkpoint_sha, "split_sha256": file_hash(ds.path),
                "sampling": cfg["sampling"], "sampler_components":cfg.get("components",{}).get("sampler"),
                "metric_components":cfg.get("components",{}).get("metrics"), "indices": indices, "t_eps": model.t_eps,
                "ordered_sources": True, "fft_norm": ds.fft_norm, "dataset_path": str(ds.path)}
    protocol_id = config_hash(protocol)[:12]
    if out is None:
        tag = f"{ev['split']}_{sampling.solver}_{sampling.num_steps}steps_{protocol_id}"
        out = project_path(ev["output_dir"]) / cfg["run"]["name"] / tag
    else:
        out = project_path(out)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output already exists: {out}. Use --force to rerun explicitly.")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "protocol.json", protocol)
    K, L, n_fft = ds.num_sources, ds.source_length, ds.n_fft
    sisdr, image_sisdr, nmses, nmaes, residuals, time_residuals, tail_ratios = [], [], [], [], [], [], []
    all_estimates, all_full = [], []
    nfes, init_hashes, score_minima, score_maxima = [], [], [], []
    final_times, clips = [], []
    for start in range(0, len(indices), ev["batch_size"]):
        batch_ids = indices[start:start + ev["batch_size"]]
        x_true = torch.stack([ds[i][0] for i in batch_ids])
        y = torch.stack([ds[i][1] for i in batch_ids]).to(model.device)
        result = sampler(model, model.sde, y, K=K, n_fft=n_fft, t_eps=model.t_eps,
                         example_ids=batch_ids)
        x_est = result.x.cpu().numpy()
        initial = result.initial_x.cpu().numpy()
        estimate_full = np.fft.irfft(x_est, n=n_fft, axis=-1, norm=ds.fft_norm)
        estimate = estimate_full[..., :L]
        refs = ds.sources[batch_ids].astype(np.float64)
        sisdr.append(metrics.si_sdr(estimate, refs))
        nmses.append(metrics.nmse(x_est, x_true.numpy()))
        nmaes.append(metrics.nmae(x_est, x_true.numpy()))
        images = np.zeros((len(batch_ids), K, n_fft), dtype=np.float64)
        for b in range(len(batch_ids)):
            for k in range(K):
                convolved = np.convolve(estimate[b, k], filters[k], mode="full")
                images[b, k, :len(convolved)] = convolved
        image_sisdr.append(metrics.si_sdr(images, ds.source_images[batch_ids]))
        predicted_y = (actual_kernels.numpy()[None] * x_est).sum(1)
        y_np = y.cpu().numpy()
        residuals.append(np.linalg.norm(predicted_y-y_np, axis=-1) / np.maximum(np.linalg.norm(y_np, axis=-1), 1e-12))
        clean_est = images.sum(1)
        obs_time = ds.mixtures[batch_ids]
        time_residuals.append(np.linalg.norm(clean_est-obs_time, axis=-1) / np.maximum(np.linalg.norm(obs_time, axis=-1), 1e-12))
        tail_ratios.append(np.sum(estimate_full[..., L:]**2, axis=-1) / np.maximum(np.sum(estimate_full**2, axis=-1), 1e-12))
        if ev["save_estimates"]:
            all_estimates.append(estimate.astype(np.float32))
            all_full.append(estimate_full.astype(np.float32))
        import hashlib
        init_hashes.extend(hashlib.sha256(initial[b].tobytes()).hexdigest() for b in range(len(batch_ids)))
        nfes.extend([result.nfe] * len(batch_ids))
        final_times.extend([result.final_time] * len(batch_ids))
        score_minima.extend([min(result.score_times)] * len(batch_ids))
        score_maxima.extend([max(result.score_times)] * len(batch_ids))
        clips.append(result.corrector_clipped_steps)
        print(f"Separated {min(start + len(batch_ids), len(indices))}/{len(indices)}", flush=True)
    S, I, M, A = map(np.concatenate, (sisdr, image_sisdr, nmses, nmaes))
    residuals, time_residuals, tail_ratios = map(np.concatenate, (residuals, time_residuals, tail_ratios))
    summary = {
        "num_examples": len(indices), "num_sources": K,
        "latent_si_sdr_db_mean": float(S.mean()), "source_image_si_sdr_db_mean": float(I.mean()),
        "nmse_mean": float(M.mean()), "nmae_mean": float(A.mean()),
        "failure_rate_lt_5db": float((S < 5.0).mean()),
        "failure_rate": float((S < ev["failure_threshold"]).mean()),
        "failure_threshold_db": ev["failure_threshold"],
        "measurement_residual_mean": float(residuals.mean()),
        "cropped_linear_measurement_residual_mean": float(time_residuals.mean()),
        "mean_tail_energy_ratio": float(tail_ratios.mean()),
        "mean_nfe": float(np.mean(nfes)), "reverse_t_epsilon": model.t_eps,
        "reverse_final_time": float(final_times[0]), "score_time_min": min(score_minima),
        "score_time_max": max(score_maxima), "corrector_clipped_example_steps": int(sum(clips)),
        "per_source_si_sdr_mean": S.mean(0).tolist(), "per_source_si_sdr_std": S.std(0).tolist(),
        "protocol_id": protocol_id, "checkpoint": str(checkpoint_path),
        "training_config": model.run_config,
    }
    atomic_json(out / "summary.json", summary)
    arrays = dict(indices=np.asarray(indices), si_sdr=S, source_image_si_sdr=I, nmse=M, nmae=A,
                  measurement_residual=residuals, cropped_linear_measurement_residual=time_residuals,
                  tail_energy_ratio=tail_ratios, initial_state_sha256=np.asarray(init_hashes),
                  nfe=np.asarray(nfes), final_time=np.asarray(final_times))
    if ev["save_estimates"]:
        arrays.update(estimates=np.concatenate(all_estimates), estimates_full=np.concatenate(all_full))
    atomic_npz(out / "per_example.npz", **arrays)
    rows = []
    for b, index in enumerate(indices):
        row = {"index": index, "si_sdr_mean": float(S[b].mean()), "image_si_sdr_mean": float(I[b].mean()),
               "measurement_residual": float(residuals[b]), "nfe": nfes[b]}
        row.update({f"si_sdr_source_{k+1}": float(S[b,k]) for k in range(K)})
        rows.append(row)
    with (out / "per_example.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Latent SI-SDR: {S.mean():.4f} dB; image SI-SDR: {I.mean():.4f} dB; NFE: {nfes[0]}")
    print("Saved:", out)
    return out, summary


def main(single=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--legacy-config", default=None)
    parser.add_argument("--index", type=int, default=0 if single else None)
    args = parser.parse_args()
    out, summary = evaluate(args.checkpoint, overrides=args.set, out=args.out, device=args.device,
        force=args.force, trust=args.trust_checkpoint, legacy_config=args.legacy_config,
        indices=[args.index] if args.index is not None else None)
    if single:
        from src.visualization import plot_separation
        plot_separation(out, summary["training_config"], args.index)
