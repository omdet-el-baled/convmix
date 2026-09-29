# ConvMix — modular research project

This is the **latest file-by-file, single-sensor convolutive pipeline** assembled
from our conversation. It is not another copy of the older all-in-one ZIPs.
The class names `ConvMixSDE`, `DCSModel`, `ScoreUNet`, `MixDataset`, and
`MixData_Module`, and SDE names `alpha`, `beta`, `sigma_min`, `sigma_max`, `T`,
`N`, `steps`, `get_Ft`, and `get_Gt`, are retained.

The primary target is the **latent source** `s_k`. Convolutive source images
`h_k * s_k` are separate arrays and secondary evaluation targets.

## 1. Install in your existing working Conda environment

Do not replace your working CUDA-enabled PyTorch build.

```bash
conda activate fourier_diffusion  # or your existing diff-sep environment
cd convmix_modular
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

The editable install makes `src.*` imports work from tests and scripts. `src` is
intentionally retained as the package name for compatibility with the code you
already ran. Python 3.10+ is required. An optional `environment.yml` creates a
Python-only environment; it deliberately does not guess a compatible CUDA wheel.

Test scope and actual local results: `verification/REPORT.md`.

## 2. Configuration layout

Everything has one source of truth. YAML configs inherit from `configs/base.yaml`.
For example:

```yaml
# configs/experiments/mismatch_p005.yaml
extends: ../base.yaml
run:
  name: mismatch_p005
model:
  mismatch_probability: 0.05
  mismatch_input: terminal
```

Any known field can be overridden on the command line:

```bash
python train.py --config configs/experiments/mechanical_k2.yaml \
  --set sde.alpha=3.0 sde.beta=1.0 trainer.max_epochs=100 run.name=k2_a3_b1
```

Unknown keys fail with an error rather than being ignored. Relative data/output
paths are resolved against the **project root**, not an implicit changed working
directory. Relative `extends` paths are resolved against their YAML file.

`steps` has two distinct meanings, kept explicit:

```yaml
sde:
  steps: 512          # covariance quadrature points
  N: 1000             # retained legacy default discretization count
sampling:
  num_steps: 50       # actual reverse solver intervals
```

The solver always stops at `model.t_eps`, never zero. There is no hidden final
denoising or extra predictor step. Euler N intervals uses N network calls; Heun
uses 2N, RK4 4N, EM N, and PC N(1+corrector_steps).

## 3. Generate or reuse data

Your existing NPZ files can be reused. Set `data.data_dir` to their directory.
Files are `train.npz`, `cv.npz`, `test.npz`, and `filters.npz`.

```bash
python generate_data.py --config configs/experiments/mechanical_k2.yaml
python visualize_dataset.py --config configs/experiments/mechanical_k2.yaml --index 0
python visualize_forward_mean.py --config configs/experiments/mechanical_k2.yaml \
  --time-max 5 --out figures/k2_mean
```

Generation refuses to overwrite existing data unless `--force` is explicitly
passed. **Do not regenerate datasets just to run a new training ablation.**

Each split stores:

```text
sources          [N,K,source_length]   latent signals
source_images    [N,K,n_fft]           full time-domain convolution
mixtures         [N,n_fft]             sum of images + measurement noise
sources_fft      [N,K,F]               optional cache
mixtures_fft     [N,F]                 optional cache
snr_db           [N]
source_seeds     [N,K]
n_fft, sample_rate, source_length, fft_norm, kind, seed
```

Default dimensions: `source_length=2000`, `filter_length=13`, `n_fft=2012`,
`F=1007`, `K=2`. Default SNR is 1–4 dB for training and 1–2 dB for cv/test.
Filters are fixed across examples/splits. The filter generator retains the
windowed-sinc/exponential/direct-tap family from the latest code; it is not
claimed to exactly reproduce measured transmission paths.

Cached FFTs are computed from the stored arrays. Signal FFT normalization is
explicit (`ortho` by default); filter transfer functions use the ordinary
unscaled forward FFT. Older NPZs lacking FFT-normalization metadata are accepted
with a warning, using the configured normalization. The loader closes NPZs
before DataLoader workers are created. For datasets larger than RAM, this eager
backend is not appropriate; sharded/memory-mapped storage is not implemented here.

### Stationary music-like and more sources

```bash
python generate_data.py --config configs/experiments/music_k2.yaml
python generate_data.py --config configs/experiments/music_k3.yaml
python generate_data.py --config configs/experiments/mechanical_k3.yaml
```

Mechanical K=4,5,6 configs are also included. Sources >=3 reuse the synthetic
epicyclic-like family: this is a stress-test extension, not new measured machines.
Music uses fixed harmonic stacks, no changing notes/envelopes. Because the
shared generator RMS-normalizes each finite record, strict stationarity of the
normalized random process is **not claimed**. Exact padded Fourier convolution
requires a time-invariant filter, not a stationary source.

## 4. Smoke run, then training

A short end-to-end test has its own dataset and output directories:

```bash
python generate_data.py --config configs/experiments/smoke.yaml
python train.py --config configs/experiments/smoke.yaml
python evaluate_test_set.py --checkpoint runs/smoke/checkpoints/last.ckpt --device cpu
```

Train the actual K=2 model:

```bash
python train.py --config configs/experiments/mechanical_k2.yaml
```

Outputs:

```text
runs/mechanical_k2/
  resolved.yaml
  manifest.json
  environment.json
  artifacts.json
  checkpoints/last.ckpt
  checkpoints/epoch=....ckpt
  csv/metrics.csv
  tb/events.out.tfevents...
```

Training does not overwrite an existing nonempty run. Change `run.name` for a
new experiment. `artifacts.json` identifies the best and final checkpoints.
On resume, prior CSV logs are retained and new CSV logs go to a separate resume session.
Model selection uses fixed-seed standard DSM validation on **both** mismatch
variants. This is not the same as selecting by validation separation SI-SDR.
Early stopping is disabled by default for equal-budget ablations.

```bash
python train.py --config configs/experiments/mechanical_k2.yaml \
  --resume runs/mechanical_k2/checkpoints/last.ckpt \
  --set trainer.max_epochs=200
```

Resume restores model, optimizer, EMA, Lightning loop state and DataLoader RNG.
Changing model/SDE/data identity is rejected. Bitwise resumed-vs-uninterrupted
identity is not promised for all accelerators/worker configurations.

```bash
tensorboard --logdir runs --port 6006
```

Useful tags: `train/loss`, `train/mismatch_selected`, `train/dsm_contribution`,
`train/mismatch_contribution`, `train/time_frac_lt_0p05`, `val/score_loss`, and
`val/mismatch_loss`. Branch contribution tags include zeros when not selected;
they are not conditional branch-only averages.

## 5. Single example and complete test set

New checkpoints contain the network config, complete SDE config, FFT metadata,
and the actual spectral kernels. Evaluation reconstructs these automatically;
you do not retype alpha/beta and accidentally use a different SDE.

```bash
python separate.py --checkpoint runs/mechanical_k2/checkpoints/last.ckpt \
  --index 0 --set sampling.solver=euler sampling.num_steps=50

python evaluate_test_set.py --checkpoint runs/mechanical_k2/checkpoints/last.ckpt \
  --set sampling.solver=heun sampling.num_steps=25

python evaluate_test_set.py --checkpoint runs/mechanical_k2/checkpoints/last.ckpt \
  --set sampling.solver=pc sampling.num_steps=25 sampling.corrector_steps=1 sampling.corrector_snr=0.10
```

These examples use 50 score evaluations each. EM is available directly as
`sampling.solver=em`; PC with zero corrector steps is equivalent to EM.
RK4 is also available. A PC corrector is a finite-step Langevin heuristic:
small-step invariance theory is not a guarantee for adaptive, state-dependent
steps. `sampling.corrector_max_step=1.0` is an explicit safety cap; set it to
`null` to reproduce the earlier uncapped rule. Clip counts are reported.

Each evaluation writes `summary.json`, `per_example.csv`, `per_example.npz`,
and `protocol.json`. Protocol identity includes checkpoint, test-file, code,
solver, seed, endpoint and example indices. Output names include this identity.
An explicit existing output requires `--force`; there is no silent cache reuse.

Primary metrics are ordered-source latent SI-SDR, unweighted rFFT nMSE/nMAE,
failure rate below 5 dB and spectral measurement residual. Source-image SI-SDR
is computed by **full time convolution of the cropped latent estimates**, not
by confusing source images with latent sources. Residual energy outside the
nominal latent source interval is saved/reported separately. No permutation,
scale adjustment for plotting, or time-shift alignment is silently introduced.

Same example ID/seed gives the same standardized initial noise. Actual initial
states match only when the SDE/init settings also match. Initial state hashes
are saved. Different stochastic grids do not share a coupled Brownian path.

**Old checkpoints:** provide `--legacy-config path/to/exact_config.yaml` when
there is no embedded config, and `--trust-checkpoint` only for files you created
and trust. A different U-Net architecture still requires its matching code/config;
strict state loading does not invent a weight conversion. There are no trained
weights or user datasets inside this ZIP.

## 6. Mismatch ablation

```bash
python train.py --config configs/experiments/no_mismatch.yaml
python train.py --config configs/experiments/mismatch_p005.yaml
# Optional p=0.5 comparison, distinct from p=0.05:
python train.py --config configs/experiments/mismatch_p05.yaml
```

The standard default is `mismatch_probability=0.0`. The p=0.05 and p=0.5 configs
are separate; neither is assumed optimal. Source ordering stays fixed.

**Important correction:** the intended terminal augmentation evaluates the network
at `y_rep + S_T epsilon`, not at `mu_T + S_T epsilon` while shifting only the
target. `model.mismatch_input: terminal` implements that recentered branch.
`forward_legacy` reproduces the last shared branch for an explicit comparison.
The formula, reasoning and limitations are in `docs/MATHEMATICS.md`; this is
still an empirical correction, not a proof of exact posterior sampling.

Default selection is one Bernoulli decision per mini-batch, matching the latest
module. Set `model.mismatch_selection=per_sample` for sample-level selection.
Do not mix these choices inside an ostensibly matched ablation.

## 7. Sweep configurations

Sweeps print a plan first. Add `--execute` deliberately:

```bash
python run_sweep.py --config configs/sweeps/mismatch.yaml
python run_sweep.py --config configs/sweeps/mismatch.yaml --execute
```

Included plans:

| File in configs/sweeps | Experiment |
|---|---|
| mismatch.yaml | p=0,0.05,0.5, seeds 0,1,2 |
| factorial.yaml | mixing/no drift x mismatch off/on |
| alpha_beta.yaml | (1,1),(1,3),(3,1),(3,3),(4,3), multiple NFEs |
| time_sampling.yaml | uniform/log_uniform/mixed |
| architecture.yaml | basic-fixed-time vs enhanced-fixed-time U-Net |
| ema.yaml | EMA vs online weights from the same training run |
| source_count.yaml | mechanical K=2,3,4,5,6; generate each dataset first |
| sampler_nfe.yaml | existing checkpoints at Euler 5–500 steps |
| samplers_equal_nfe.yaml | Euler/Heun/EM/PC at 50 NFE |
| pc_snr.yaml | PC corrector SNR 0.05,0.10,0.16,0.25 |

The `basic` network is a controlled lighter version of the enhanced architecture,
not a claimed byte-identical reproduction of the first Gaussian-embedding U-Net.
Both variants have a completely fixed time embedding.

Evaluation-only sweep:

```bash
python run_sweep.py --config configs/sweeps/sampler_nfe.yaml \
  --checkpoints runs/mechanical_k2/checkpoints/last.ckpt --execute
```

Edit the sweep YAML to change epochs, cases or seeds. Run identity includes the
training budget. A 10-epoch checkpoint is never silently labeled 100 epochs.
Incomplete runs require `--resume-incomplete`. Existing completed runs must
match both their config and source fingerprint. To reuse weights after editing
sampling code, run an **evaluation-only** sweep explicitly.

Sweep output includes a plan, per-run results, comparison plots, and an aggregate table with
training-seed standard deviations. One seed is reported without pretending it
estimates training uncertainty. NFE gaps alone do not diagnose stiffness.

Regenerate plots from an existing sweep table:

```bash
python plot_sweep.py --results sweeps/mismatch/results_per_run.csv
```

Paired example differences between two evaluations:

```bash
python compare_results.py --left path/to/no_mismatch/evaluation \
  --right path/to/mismatch/evaluation --out results/mismatch_paired.json
```

The confidence interval resamples examples conditional on those trained models;
it is not a substitute for repeating training seeds.

## 8. Diagnostics

```bash
python diagnose.py mean --config configs/experiments/mechanical_k2.yaml --time-max 5
python diagnose.py score --checkpoint runs/mechanical_k2/checkpoints/last.ckpt
python diagnose.py covariance --bins 8,16,32
```

Mean figures use independently recomputed **time convolution** for the reference.
Fixed-time score diagnostics compare to the conditional perturbation-kernel
training target; they do **not** know the exact marginal score. Large low-time
DSM values can include irreducible target variance. Numerical-solver accuracy
and final source SI-SDR are different quantities.

The covariance benchmark compares structured quadrature, frequency-block
Lyapunov integration and global dense Lyapunov integration on small controlled
systems. Timing and numerical error are reported together. It is not an
already-completed paper performance benchmark.

## 9. Source map

```text
src/config.py               YAML inheritance and explicit overrides
src/data/filters.py         FIR generation/loading
src/data/generate.py        mechanical/music-like NPZ generation
src/dataset.py              MixDataset + Lightning MixData_Module
src/sde/freqsde.py           ConvMixSDE, structured operations + dense compatibility
src/sde/rfft_noise.py        endpoint-safe complex noise
src/models/unet.py          fixed-time ScoreUNet and basic ablation
src/pl_model.py             DSM/mismatch training and EMA
src/sampling/samplers.py    Euler, Heun, RK4, EM, PC (shared implementation)
src/factory.py              consistent model reconstruction
src/training.py             Trainer, logs, checkpoints and provenance
src/evaluation.py           single/full-set sampling and metrics
src/visualization.py        separate waveform/spectrum figures
src/diagnostics.py          mean/score/covariance diagnostics
src/sweeps.py               explicit experiment orchestration
src/metrics.py              ordered SI-SDR, nMSE, nMAE
```

The root files are thin entry points, not duplicated implementations.
`data/create_dataset_npz.py` and `data/filter_generator.py` are compatibility
entry points after editable installation.

### Scope

This ZIP packages the working **single-sensor** reconstruction with configs and
tests. It does not implement or claim results for DPS, Conv-TasNet, SepReformer,
real measured datasets, or the distinct multi-sensor project from earlier in
the conversation. No tests establish trained separation quality; run the actual
experiments before drawing that conclusion.
