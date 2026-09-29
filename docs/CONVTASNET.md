# Conv-TasNet baseline on the existing NPZ data module

This add-on adapts the supplied ConvTasNet + Wiener deconvolution + negative
SI-SDR training loop. It does not alter the diffusion model, SDE, dataset loader,
or existing experiment configs. Asteroid's actual ConvTasNet is used: it is not
reimplemented. A lazy factory imports it only when the baseline is selected.

## Install into the existing project

Extract `convtasnet_npz_addon.zip` in the root of `convmix_configurable` (the root
that contains `config/`, `src/`, and `train.py`). It adds files only. Alternatively,
use `convmix_configurable_with_convtasnet.zip`, which already contains the add-on.

Use the same working Conda environment and matched PyTorch/torchaudio builds.

```bash
conda activate fourier_diffusion
# Your original script already imports Asteroid. Reuse that installation:
python -c "from asteroid.models import ConvTasNet; print('Asteroid import OK')"
python -m pip install -e . --no-deps
python -m pytest tests/test_convtasnet_baseline.py tests/test_convtasnet_hydra.py -q
```

The project already needs Hydra, Lightning, NumPy, and TorchMetrics. No resampling
or audio file reading is used by the adapter. Keep a working Asteroid installation
rather than reinstalling it automatically.

When Asteroid is genuinely missing, note that its 0.7.0 package metadata pins
`torchmetrics<=0.11.4`, even though this adapter uses only its waveform model and
not Asteroid's metric/training utilities. Do not install that legacy pin over your
working Lightning environment. The optional installation is deliberately split:

```bash
python -m pip install -r requirements-convtasnet-runtime.txt
python -m pip install --no-deps -r requirements-convtasnet.txt
python -c "from asteroid.models import ConvTasNet; print('Asteroid import OK')"
```

This explicitly bypasses Asteroid's legacy metadata constraint; `pip check` may
still report that constraint as unsatisfied. It is not a claim that all Asteroid
APIs work with newer TorchMetrics. Run the included real-Asteroid integration tests
and smoke configuration in your environment. That dependency combination could not
be executed here. Keep matched torch/torchaudio binaries rather than upgrading one
of them independently.

Metadata reference: https://github.com/asteroid-team/asteroid/blob/v0.7.0/setup.py

## Pipeline and targets

`MixData_Module` is reused without editing it. Each batch remains:

```text
x0: [B,K,F] complex latent-source spectra
y:  [B,F]   complex observed-mixture spectra
```

The adapter performs:

```python
mixture = torch.fft.irfft(y, n=n_fft, norm=fft_norm)       # [B,n_fft]
reference = torch.fft.irfft(x0, n=n_fft, norm=fft_norm)
reference = reference[..., :source_length]                # [B,K,source_length]
normalized_mixture, gain = normalization(mixture)
raw_output = network(normalized_mixture)                 # [B,K,n_fft]
estimate_full = postprocess(raw_output)                  # Wiener or identity
estimate_full = estimate_full * gain[:, None, :]         # default: restore_scale=true
loss = negative_si_sdr(estimate_full[..., :source_length], reference)
```

For the current dataset, n_fft=2012 and source_length=2000. The convolution tail
is retained through the network and Wiener step. Only the latent outputs and
references are cropped for the training loss. The filter transfer functions use
an unscaled rFFT; data inverse FFTs use the dataset's recorded FFT normalization.
No source permutations are introduced. No targets enter the network input.

The raw network output is passed through Wiener inversion, as in the supplied
script. It has **no separate source-image supervision**. Evaluation source images
are constructed independently by convolving the final cropped latent estimates
with the known filters, not by treating the raw intermediate output as truth.

## Preserved choices and explicit changes

Preserved: Asteroid ConvTasNet default architecture parameters, Adam lr=2e-4,
weight_decay=0, batch_size=16, max_epochs=500, validation every 5 epochs, patience
3 validation checks, fixed source ordering, and differentiable Wiener filter
`H.conj() / (abs(H)**2 + snr)` with snr=0.01. No scheduler or EMA is enabled by
default. Gradient clipping defaults to 0, preserving the supplied optimizer loop.

Changes made explicit:

* The same NPZ loader as the diffusion model replaces the unspecified WAV loader.
* Loss is a stateless TorchMetrics functional call. `zero_mean=false` preserves
  the default of the user's `ScaleInvariantSignalDistortionRatio()` construction.
  It is negative SI-SDR, not MSE, despite the supplied filename.
* Default normalization is **per-example peak** and output gain is restored.
  The original `mixture / mixture.abs().max()` uses a whole-batch peak, so one
  example's preprocessing depends on other examples in the batch. The explicit
  `legacy_normalization` recipe reproduces that policy and omits gain restoration.
* `snr` is still the Wiener inverse-SNR regularizer, not the measured SNR in dB.
* H is computed once and stored as registered buffers. No FFT of the filters is
  repeated for each training batch. FFT deconvolution stays differentiable.
* Epoch losses are weighted by example count, including the final partial batch.
* Best and last checkpoints include optimizer/loop state, config, filters and
  data metadata. Resume refuses a changed model or changed dataset identity.
* Optional ReduceLROnPlateau steps only at a fresh validation epoch, not on the
  same stale validation result during each intervening epoch.

SI-SDR is scale invariant. Restoring the input gain keeps the I/O scale convention
explicit but does not establish absolute amplitude identifiability from an SI-SDR
loss. Shared evaluation nMSE/nMAE use the output's native amplitude, with no fitting
to the reference.

## Train

First inspect the config and run the real tiny Asteroid smoke test on a few
batches of your existing dataset; no regenerated data are required:

```bash
python train_convtasnet.py --cfg job --resolve
python train_convtasnet.py baseline_experiment=smoke \
  data.data_dir=/absolute/path/to/mechanical_k2_npz
```

Then the supplied baseline recipe:

```bash
python train_convtasnet.py \
  data.data_dir=/absolute/path/to/mechanical_k2_npz \
  run.name=convtasnet_wiener_seed42
```

Output example:

```text
runs/convtasnet_wiener_seed42/
  resolved.yaml
  manifest.json
  environment.json
  model_structure.txt
  artifacts.json
  checkpoints/best-epoch=....ckpt
  checkpoints/last.ckpt
  tb/events.out.tfevents...
  csv/metrics.csv
```

```bash
tensorboard --logdir runs/convtasnet_wiener_seed42 --port 6006
```

`val/loss` is negative training-definition SI-SDR; lower is better. The test
summary also reports the existing project's centered SI-SDR, so both baselines
and diffusion use the same evaluation metric. The zero-mean training option is
explicitly changeable: `separator.loss.zero_mean=true`.

Resume a trusted checkpoint with a longer total budget:

```bash
python train_convtasnet.py \
  data.data_dir=/absolute/path/to/mechanical_k2_npz \
  run.name=convtasnet_wiener_seed42 \
  run.resume=runs/convtasnet_wiener_seed42/checkpoints/last.ckpt \
  run.trust_checkpoint=true trainer.max_epochs=700
```

## Modular configs and ablations

All components are targets:

```text
config/convtasnet.yaml
config/separator/convtasnet.yaml
config/separator/network/convtasnet.yaml
config/separator/postprocess/{wiener,none}.yaml
config/separator/normalization/{peak,batch_peak,none}.yaml
config/separator/loss/negative_si_sdr.yaml
config/separator/optimizer/{adam,adamw}.yaml
config/separator/scheduler/{none,plateau}.yaml
config/trainer/convtasnet.yaml
config/callbacks/convtasnet.yaml
```

The lazy `make_convtasnet` factory passes its kwargs directly to
`asteroid.models.ConvTasNet`. The network spec can instead name another waveform
network whose constructor accepts `n_src` and `sample_rate`, and whose output is
[B,K,n_fft]. There is no registry to update. Normalization returns `(waveform,
gain[B,1])`; a postprocessor receives [B,K,n_fft] and must return that same shape.
The default loss receives two cropped [B,K,source_length] tensors and returns a
scalar. Optimizer construction injects the trainable parameter iterator.

Examples:

```bash
# Direct latent prediction (no Wiener), separate training from scratch
python train_convtasnet.py baseline_experiment=direct

# Regularizer change (must retrain to compare the same end-to-end recipe)
python train_convtasnet.py separator.postprocess.snr=0.001 run.name=wiener_1e3

# Smaller architecture
python train_convtasnet.py separator.network.n_blocks=4 \
  separator.network.hid_chan=256 run.name=convtasnet_smaller

# Reproduce the original batch-wide peak normalization policy
python train_convtasnet.py baseline_experiment=legacy_normalization

# Fresh-validation plateau schedule
python train_convtasnet.py separator/scheduler=plateau run.name=convtasnet_plateau

# Three sources using an existing matching NPZ dataset
python train_convtasnet.py data=mechanical_k3 run.name=convtasnet_k3

# Matched Wiener on/off experiment, fixed epoch budget and three seeds
python train_convtasnet.py -m separator/postprocess=wiener,none seed=0,1,2 \
  trainer.max_epochs=100 callbacks.early_stopping=null \
  run.name=convtasnet_wiener_ablation
```

Hydra multiruns append unique job/config identifiers. They never silently reuse a
short training checkpoint for a longer requested run. Both models in the on/off
Wiener ablation receive their own training; switching postprocessors only during
test is a different experiment and is deliberately not allowed by this CLI.

## Evaluate the full test set

Read `artifacts.json` to obtain the actual best checkpoint filename; do not assume
it is literally `best.ckpt`.

```bash
python evaluate_convtasnet.py \
  checkpoint=runs/convtasnet_wiener_seed42/checkpoints/last.ckpt \
  evaluation.trust_checkpoint=true
```

Only use the trust flag for checkpoints whose source/config you trust. Architecture,
normalization, Wiener regularizer, filters and FFT conventions are restored strictly
from the training checkpoint. This flag is not permission to load unknown pickle
files. Old raw `model.state_dict()` files are not silently converted.

Useful overrides:

```bash
python evaluate_convtasnet.py checkpoint=/path/to/checkpoint.ckpt \
  evaluation.trust_checkpoint=true evaluation.batch_size=16 \
  evaluation.max_examples=20 out=results/convtasnet_debug
```

Omit max_examples (default null) to evaluate the complete test set. `data.data_dir`
can relocate the dataset but dimensions and filters must still agree. Use
`evaluation.split=cv` for sampler/model-selection experiments; do not select the
best Wiener regularizer using the test set.

Outputs include per-example/per-source SI-SDR and spectral nMSE/nMAE, source-image
SI-SDR, failure rate below 5 dB, full-state spectral measurement residual, cropped
linear-convolution residual, and latent-tail energy ratio. Files:

```text
summary.json
protocol.json
per_example.csv
per_example.npz
```

When `evaluation.save_estimates=true`, the NPZ includes both cropped latent estimates
and full outputs. There is one deterministic network pass per example; diffusion
NFE has no iterative interpretation here. No scale/permutation/time alignment is
silently applied. SI-SDR and nMSE measure different properties.

## Test scope

See `verification/CONVTASNET_REPORT.md`. CPU adapter/mathematical tests and a short
Lightning run with a small waveform test double were executed. This is NOT a claim
that a real Asteroid ConvTasNet was trained in this environment. Native Hydra and
real Asteroid integration tests are included, but those dependencies were unavailable
here. No user-dataset training, CUDA run, or separation-performance result is claimed.

## Primary API references

* Asteroid model parameters and waveform interface:
  https://asteroid-team.github.io/asteroid/package_reference/models.html
* TorchMetrics functional SI-SDR implementation and zero_mean default:
  https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/sdr.py
* Hydra target instantiation:
  https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/
