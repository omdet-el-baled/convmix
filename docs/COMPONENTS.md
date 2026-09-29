# Component contracts

All `_target_` strings name importable Python classes/functions. There is no
new-architecture registry to edit. Composite objects choose their child classes
through `build_component`; Hydra performs target resolution/instantiation.
Default runtime dimensions override conflicting user dimensions deliberately.

## Network

`model.score_model` -> a `torch.nn.Module`.

Constructor: accepts `num_sources`, `n_fft`, and all additional YAML parameters.
Forward: `(x: complex[B,K,F], y: complex[B,F], t: real[B]) -> complex[B,K,F]`.
Ordinary `state_dict`, `.to`, `.train`, `.eval`, and `.parameters` behavior is
required. `DCSModel` owns an EMA copy. Handle actual FFT parity; do not silently
project the final bin to real for odd FFT length.

### Internal U-Net components

| Spec | Constructor parameters injected by owner | Call / requirement |
|---|---|---|
| `time_embedding` | `embedding_size=time_dim` | `(t) -> [B,time_dim]`; fixed, no trainable parameters |
| `features` | `num_sources` | `(x,y) -> [B,C,F]`; exposes integer `out_channels` |
| `block` | `in_channels,out_channels,time_dim,dilation,dropout` | `(h,time_embedding) -> [B,out_channels,F]` |
| `attention` | `channels,num_heads` | `(h) -> same shape` |
| `downsample` | `channels` | `(h) -> lower-resolution [B,C,F']` |
| `upsample` | `channels` | `(h,target_length) -> [B,C,target_length]` |
| `input_projection` | `in_channels,out_channels` | Conv1d-like feature projection |
| `output_projection` | `in_channels,out_channels=2K` | Returns 2K real channels, same resolution |
| `output_norm` | `channels` | Shape-preserving normalization |
| `output_activation` | none | Shape-preserving activation |

`encoder_dilations` and `decoder_dilations` contain one list per resolution.
A list's length defines the number of blocks, its values the individual
dilations. `attention_after` is the insertion index in `bottleneck_dilations`.
This supports a deeper/narrower/wider graph without changing `forward`.

The default output projection is initialized exactly as before. Set
`output_init_std: null` to retain a custom head's own initialization, including
heads that do not expose a single `.weight`. This is an explicit model change.

`FiLMResidualBlock` accepts nested `normalization`, `activation`, `convolution`,
`time_projection`, and `skip`. It injects shape parameters and a same-length
kernel/padding/dilation. Use `block.kernel_size`, not a conflicting nested
convolution kernel, to change the receptive field. The skip branch uses a 1x1
projection when channel counts differ. Attention and resampling also expose
their underlying modules. A replacement must respect the caller's interface.

## Lightning / objectives

`model._target_` defaults to `src.pl_model.DCSModel`. The factory injects an
already-built `score_model`, `sde`, and `n_fft`.

`model.loss` is a callable:

```python
loss(model, x0, y, t, *, mismatch_mask=None,
     use_ema=False, generator=None, epsilon=None) -> dict
```

The standard objective's dictionary has `loss` (scalar), `per_example` [B],
`time`, `residual`, `target_noise`, `xt`, `mean`, `std`, `epsilon`,
`mismatch_mask`. Training/diagnostics use those keys. Its nested `reduction`
returns a per-example residual loss. A new objective should retain these
contracts or be paired with an explicitly different Lightning module.

`model.time_sampler` and `model.validation_time_sampler` are independent
callables accepting batch_size/device/dtype and keyword t_eps, T, log_fraction,
generator, strategy. The fixed standard validation distribution remains uniform.
The supplied time sampler `name` is a reporting label, not the selector; its
`_target_` determines behavior. A custom time sampler without a `name` can set
`model.time_sampling_strategy=custom` explicitly.

`model.noise_sampler` is a function/partial `(x,n_fft=...,generator=...)`.
The default is the unchanged standardized physical-rFFT noise helper.

`model.loss_selection`: `(B,device,probability,mode) -> bool[B]`.
`model.ema_update`: `(online_model,ema_model,decay) -> None`; the default exact
update is called only by the real optimizer post-step hook. Checkpointed
stateful strategies should be nn.Module instances with registered buffers.

## Optimizer / scheduler

An optimizer target receives `params` at `configure_optimizers`; it is not
constructed before model parameters exist. Other kwargs come from YAML.
`model.learning_rate` and `model.weight_decay` are retained through interpolation
in the supplied optimizer recipes; explicit optimizer kwargs take precedence.

A scheduler configuration wraps a target plus Lightning metadata:

```yaml
enabled: true
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: ${trainer.max_epochs}
  eta_min: 1.0e-6
interval: epoch
frequency: 1
```

The owner injects `optimizer`. `enabled: false` gives the previous no-scheduler
behavior. A plateau scheduler includes its monitored validation key.

## Sampling

`model.sampler` builds a callable:

```python
sampler(model, sde, y, *, K, n_fft, t_eps, example_ids=None,
        x_T=None, noise_source=None) -> SamplerResult
```

Its component interfaces are:

- Predictor: `(x,current_time,next_time,ctx) -> x_next`.
- Corrector: `(x,current_time,ctx) -> (x_corrected,clipped_example_count)`.
- Grid: `(T,t_eps,num_steps) -> torch.float64[num_steps+1]`, strictly decreasing,
  first T and last exactly t_eps.
- Initialization: `(sde,y,K,n_fft,add_noise,noise_source=...) -> [B,K,F]`.
- Noise factory: constructor `(example_ids,seed,device,n_fft)`; resulting
  callable `(like_tensor) -> standardized_noise`.

`SamplingContext.network` is the score-call entry point and counts actual NFE.
`SamplingContext.field` preserves the previous probability-flow/reverse-SDE
formula. Custom predictors/correctors must use these interfaces to preserve
counting. Neither stage may update after the final integration endpoint.
The loop never interprets a solver label to choose an algorithm.

## Data / generation / metrics / infrastructure

The configured DataModule must provide setup/train_dataloader/val_dataloader/
test_dataloader and dataset metadata used by the workflow. The NPZ Dataset has
the same fields and methods as before. Its loader spec can use recursive Hydra
construction for collate functions or other leaf subcomponents.

Generation selects DatasetGenerator -> filter bank callable + SourceBank +
measurement-noise callable. Default source distributions and random draw order
are unchanged. When defining a different filter distribution, keep the SDE's
existing supported physical-kernel conditions or deliberately implement a
separate SDE experiment; do not silently regularize a singular operator.

The metric suite exposes `si_sdr`, `nmse`, `nmae` vectorized callables. Returned
arrays must be per-example/per-source [B,K] for source metrics. New definitions
should be separately labelled in scientific reporting.

Callbacks and loggers are dictionaries of `_target_` specs (null disables an
entry). The `checkpoint` callback is expected to be ModelCheckpoint-compatible;
its directory is supplied by the run. Reserved logger entries `csv` and
`tensorboard` receive the run directory. Trainer is instantiated from its target
and scalar config. The existing workflow remains single-device.

## Trust and checkpoint contracts

Targets are code, not safe declarative data. Only load/instantiate trusted YAML
and checkpoint config. The checkpoint contains the complete resolved target
specs and parameters. Unknown architectures are not mapped to a nearest known
one. Checkpoint loading is strict; modifying a graph needs compatible weights or
new training. Bundle-local source hashing does not fingerprint every external
third-party plugin package; record external code revisions separately.
