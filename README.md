# ConvMix — component-driven Hydra project

This revision changes **how components are constructed**, not the mathematical
model in `convmix_modular.zip`. It keeps `ConvMixSDE`, `DCSModel`, `ScoreUNet`,
`MixDataset`, `MixData_Module`, `alpha`, `beta`, `sigma_min`, `sigma_max`, `T`,
`N`, `steps`, `get_Ft`, and `get_Gt`.

The main configuration tree is now **`config/`**, using native Hydra defaults,
config groups and `_target_` instantiation. The older **`configs/`** tree and
`--config ... --set ...` commands are retained as a compatibility path. They are
not the new component-configuration interface.

## Install in the working environment

Keep your working PyTorch/CUDA build:

```bash
conda activate fourier_diffusion  # or your existing diff-sep environment
cd convmix_configurable
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

`requirements.txt` deliberately does not install torch. It adds `hydra-core` and
`omegaconf`. Use the editable source checkout; root scripts/config files belong
to this checkout. Python 3.10+ and one CPU/GPU device are supported.

**Local verification:** 48 CPU tests passed. The native Hydra integration test
module was skipped because `hydra-core` was not installed and external package
installation was unavailable in the execution environment. Those tests are
included and will run with the dependencies installed. No CUDA or long training
run was executed. See `verification/REPORT.md` for exact scope; this is not an
assertion that all native Hydra CLI combinations were executed locally.

Use only trusted target-bearing YAML and checkpoints: `_target_` selects Python
code. Evaluating a newly produced target-configured checkpoint requires the
explicit `evaluation.trust_checkpoint=true` flag for checkpoints you created and
trust. This flag also allows full PyTorch checkpoint deserialization.

## 1. Change the complete network from configuration

```bash
python train.py experiment=mechanical_k2 model/score_model=unet_enhanced
python train.py experiment=mechanical_k2 model/score_model=unet_deep run.name=k2_deep
python train.py experiment=mechanical_k2 model/score_model=unet_basic run.name=k2_basic
```

The selected file contains the actual class to instantiate:

```yaml
# config/model/score_model/unet_enhanced.yaml
_target_: src.models.unet.ScoreUNet
_recursive_: false
base_channels: 64
time_dim: 256
num_heads: 4
dropout: 0.05
channel_multipliers: [1, 2, 4]
encoder_dilations: [[1, 2], [1, 2], [1, 4]]
decoder_dilations: [[1, 1], [1, 2], [1, 2]]
bottleneck_dilations: [1, 2, 4]
attention_after: 2
```

This snippet omits the file's `defaults` and component declarations; the actual
file is complete. Each inner dilation list specifies **the number and dilation
of blocks at that resolution**. Both encoder and decoder lists are ordered from
low to high channel resolution; execution mirrors the decoder. Supply one list
per `channel_multipliers` entry. The bottleneck width is the final stage width.

To add a different architecture, create its importable class and one config,
not an `if architecture == ...` branch in `train.py` or the Lightning module:

```yaml
# config/model/score_model/my_model.yaml
_target_: my_package.networks.MyScoreNetwork
_recursive_: false
width: 128
num_layers: 8
```

```bash
python train.py model/score_model=my_model run.name=my_model_trial
```

The class must accept `num_sources` and `n_fft` supplied from the dataset, and
implement `forward(x, y, t) -> complex [B,K,F]`. Its additional constructor
parameters come directly from YAML. `src/models/examples.py: TinyScoreNet`
provides a small executable example with a different constructor and graph:

```bash
python train.py model/score_model=tiny_example run.name=tiny_architecture
```

The tiny model is a component-interface example, not a recommended research model.
Changing architecture requires new compatible weights. There is no silent
conversion of checkpoints between unrelated network graphs.

## 2. Change components *inside* the U-Net

```bash
# Fixed Gaussian features instead of fixed sinusoidal features
python train.py experiment=gaussian_time

# Remove attention without modifying the U-Net forward code
python train.py experiment=no_attention

# Real/imaginary inputs only, instead of adding log-magnitude features
python train.py experiment=real_imag_features

# A block's convolution size, activation, and FiLM projection remain configurable
python train.py model.score_model.block.kernel_size=5 run.name=kernel5
python train.py model.score_model.block.activation._target_=torch.nn.GELU run.name=gelu
```

Or select nested groups explicitly:

```bash
python train.py model/score_model/attention=none run.name=no_attention
python train.py model/score_model/time_embedding=gaussian run.name=gaussian_time
python train.py model/score_model/features=real_imag run.name=real_imag
```

The supplied time embeddings have **no trainable parameters**; their frequencies
are checkpointed buffers. A custom U-Net time embedding with trainable parameters
is rejected. FiLM projections in the residual blocks remain trainable, as before.

The U-Net also accepts configurable block, normalization, activation,
convolution, skip projection, downsampling, upsampling, feature encoder,
input/output projection, output normalization, output activation, and attention
submodules. `num_sources`, `n_fft`, channel dimensions and projection dimensions
are injected by the owner to maintain shape compatibility. For a convolution's
kernel size use the owning block's `kernel_size`; padding is derived to preserve
length. The default nearest upsampling remains unchanged.

Full component contracts: `docs/COMPONENTS.md`.

## 3. Optimizer, scheduler, objective, sampling and data

```bash
# Replace optimizer and attach a scheduler
python train.py model/optimizer=adam model/scheduler=cosine run.name=adam_cosine

# Same mismatch objective, probability changed from configuration
python train.py experiment=mismatch_p005
python train.py experiment=no_mismatch

# Same SDE, different alpha/beta
python train.py model.sde.alpha=3.0 model.sde.beta=1.0 run.name=a3_b1

# Time-distribution and drift ablations remain explicit
python train.py model/time_sampler=mixed run.name=mixed_time
python train.py model/sde=none run.name=no_drift
```

`model.sde`, `model.loss`, `model.optimizer`, `model.scheduler`,
`model.time_sampler`, `model.validation_time_sampler`, `model.noise_sampler`,
`model.ema_update`, `model.loss_selection`, `datamodule`, `trainer`, `callbacks`,
`logger`, `generation`, and `metrics` all contain target specs or nested target
specs. The objective interface returns the same diagnostics dictionary used by
`DCSModel`. Replacing a loss is an explicit experiment, not a change bundled
silently with this refactor.

The data module separately configures its Dataset class and DataLoader factory,
plus optional `train_loader`, `val_loader`, and `test_loader` replacements.
A leaf loader can enable `_recursive_: true` for a configured `collate_fn`.
Most composite objects deliberately use `_recursive_: false` because they
instantiate children after computing channel dimensions/runtime metadata.
`src/components/instantiate.py` honors an explicit recursive setting.

### Existing data need not be regenerated

```bash
python train.py data.data_dir=/absolute/path/to/mechanical_k2_npz run.name=k2_existing
```

The files remain `filters.npz`, `train.npz`, `cv.npz`, `test.npz`. The primary
target remains latent sources; source images are separate. No signal/filter
normalization convention is changed by the component refactor.

For new data:

```bash
python generate_data.py experiment=mechanical_k2
python visualize_dataset.py experiment=mechanical_k2 index=0
python visualize_forward_mean.py experiment=mechanical_k2 diagnostics.time_max=5
```

`generation=mechanical_explicit` exposes each source generator and all its
original arguments in YAML. `generation/default.yaml` preserves the existing
dispatcher. Both delegate to the same source/noise/filter functions. Use a new
`data.data_dir` for changes to data-generating distributions. Existing data
require `force=true` to overwrite.

## 4. Inspect config and run a short smoke experiment

```bash
# Compose/resolve only: no training
python train.py --cfg job --resolve
python train.py experiment=no_attention --cfg job --resolve

# Inspect available groups
python train.py --help

# Small, separate dataset and run
python generate_data.py experiment=smoke
python train.py experiment=smoke
python evaluate_test_set.py checkpoint=runs/smoke/checkpoints/last.ckpt \
  evaluation.trust_checkpoint=true device=cpu
```

A run refuses to overwrite a nonempty directory. Change `run.name` for a new run.
Artifacts include `hydra_config.yaml`, `resolved.yaml`, `components.json`,
`model_structure.txt`, numerical/data metadata, checkpoints and logs. Hydra's
own config/override logs live in `hydra_logs/`; model checkpoints stay under
`runs/<run.name>/checkpoints/`.

```bash
# Resume the same graph/physics/data, with a longer epoch budget
python train.py experiment=mechanical_k2 run.name=mechanical_k2 \
  run.resume=runs/mechanical_k2/checkpoints/last.ckpt \
  run.trust_checkpoint=true trainer.max_epochs=200

tensorboard --logdir runs --port 6006
```

## 5. Separation and swappable predictor/corrector components

```bash
python separate.py checkpoint=runs/mechanical_k2/checkpoints/last.ckpt \
  evaluation.trust_checkpoint=true index=0

python evaluate_test_set.py checkpoint=runs/mechanical_k2/checkpoints/last.ckpt \
  evaluation.trust_checkpoint=true model/sampler=pc \
  model.sampler.num_steps=25 model.sampler.corrector.num_steps=1 \
  model.sampler.corrector.snr=0.10
```

The model, network graph, SDE, kernels and endpoint are restored from the
checkpoint. Evaluation does **not** replace them with the CLI's default model.
Do not pass `experiment=...`, `model/score_model=...`, or `model/sde=...` when
evaluating an existing checkpoint.

A sampler has independent target specs for `predictor`, `corrector`, `grid`,
`initialization`, and `noise`. The integration loop does not dispatch on a
hardcoded solver-name list. `solver` is a recorded label; replace its components
or select a sampler group to change the algorithm.

```yaml
# Components selected by config/model/sampler/pc.yaml
_target_: src.sampling.pipeline.PredictorCorrectorSampler
_recursive_: false
predictor:
  _target_: src.sampling.predictors.EulerMaruyama
corrector:
  _target_: src.sampling.correctors.LangevinCorrector
  num_steps: 1
  snr: 0.16
  geometry: diffusion
  step_size: null
  max_step: 1.0
```

`model.sampler.num_steps=100` alone patches the checkpoint's sampler; it does
not reset a saved PC sampler to the default Euler algorithm. Selecting
`model/sampler=heun` intentionally replaces the whole recipe. No step or
corrector is applied after `model.t_eps`. NFE is counted from actual score calls.

Examples at exactly 50 NFE:

```bash
python evaluate_test_set.py -m \
  checkpoint=runs/mechanical_k2/checkpoints/last.ckpt \
  evaluation.trust_checkpoint=true \
  model/sampler=euler_nfe50,heun_nfe50,em_nfe50,pc_nfe50
```

The recipes use Euler 50 intervals, Heun 25, EM 50, or PC 25 with one corrector.
Changing a stochastic solver/grid does not imply coupled Brownian increments;
the existing per-example paired initial-noise policy is retained.

## 6. Native experiment sweeps

```bash
# Same loss code, probabilities 0 / .05 / .5, three training seeds
python train.py -m experiment=mismatch_sweep

# U-Net structure is selected from YAML
python train.py -m experiment=architecture_sweep

# alpha in {1,3,4}, beta in {1,3}, three seeds
python train.py -m experiment=alpha_beta_sweep

# Or a custom grid without another Python script
python train.py -m experiment=mechanical_k2 \
  model/score_model=unet_basic,unet_enhanced \
  model.mismatch_probability=0.0,0.05 seed=0,1,2 \
  actions.evaluate_after_training=true
```

Native multiruns get distinct run names containing job number, seed and a
configuration hash. Early stopping is disabled by default for fixed-budget
ablations. The supplied training sweep experiments evaluate after training;
set `actions.evaluate_after_training=false` to train only.

```bash
# Evaluate an existing model at different NFE, without retraining
python evaluate_test_set.py -m \
  checkpoint=runs/mechanical_k2/checkpoints/last.ckpt \
  evaluation.trust_checkpoint=true model.sampler.num_steps=5,10,20,50,100

python collect_results.py --root results --out sweeps/native_summary
```

The collector includes the full architecture/component identity in its grouping:
changing width, blocks, optimizer, or sampler cannot accidentally merge results
merely because the class is still named `ScoreUNet`. Seed-level statistics are
not evidence of improvement until the runs have actually been completed.

`run_sweep.py --config configs/sweeps/...` remains the **legacy** planned-sweep
entry point. Native component sweeps use Hydra's `-m`, as above. Existing legacy
sweep/config files are not silently rewritten into a different experiment.

## 7. What was preserved and what was tested

The SDE, rFFT noise, original sampler reference, filter functions, and metric
files are byte-identical to the preceding ZIP. Their hashes are recorded in
`verification/preserved_core.json`. The new component sampler is tested against
that reference for Euler, Heun, RK4, EM, fixed-step PC, and adaptive PC.

The default refactored U-Net retains the previous parameter keys and shapes,
initialization order, forward values and gradients in the tested CPU cases.
Its extracted loss likewise matches prior standard/mismatch residuals, inputs,
targets, losses and gradients. New options are explicit ablations, not claims
that a deeper model or a different sampler performs better.

Native Hydra tests are in `tests/test_hydra_workflow.py`. With dependencies
installed they compose experiment groups, instantiate entire and nested network
replacements, select optimizer/scheduler/sampler classes, protect checkpoint
identity, and execute short CLI training/evaluation. They were not executable
in the build environment; do not interpret their inclusion as a local pass.

The previous numerical limitations remain: approximate finite-T terminal law,
conditional-kernel DSM diagnostics rather than known marginal-score error,
finite-step PC heuristics, eager NPZ loading rather than a large-data backend,
and no validation of GPU bitwise parity or long-run separation quality.
