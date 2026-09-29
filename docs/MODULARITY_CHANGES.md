# What this revision changes

## Scope

The preceding ZIP used a custom `extends` config loader and a fixed network
factory choosing only `enhanced` or `basic`. That let users change values but
not freely replace components. This revision adds native Hydra composition and
fully qualified `_target_` specs while keeping the proven numerical core.

## Preserved exactly

`src/sde/freqsde.py`, `src/sde/rfft_noise.py`, `src/sampling/samplers.py`,
`src/metrics.py`, and `src/data/filters.py` are byte-identical to the input ZIP.
`verification/preserved_core.json` records the hashes.

No new diffusion law, score convention, mismatch formula, noise schedule,
endpoint correction, permutation matching or signal scaling was introduced.
The baseline U-Net graph/init/keys are preserved under its default recipe.
The prior basic network remains its existing controlled lighter variant.

## Construction refactor

- The default ScoreUNet's embedding/blocks/attention/resampling/features are
  separate modules with injectable child factories. Stage lists construct
  variable depth, width, block count and dilations.
- Objective, time/noise/selection rules and EMA update are separate callables.
- Optimizer and optional scheduler targets are constructed with live parameters.
- DataModule, Dataset/DataLoader, generation, metrics, Trainer, callbacks and
  loggers are config-selected.
- Predictors, correctors, grid, initialization and noise supply a generic
  sampling loop. The unchanged sampler is retained as an independent reference.
- New target configs are embedded in checkpoints. Evaluating them requires
  explicit trust and reconstructs the exact saved graph, not default CLI values.
- An NFE-only eval override preserves the saved sampler's other components.
- Existing legacy YAML/config commands remain available. Native Hydra is not
  replaced by a hand-rolled imitation.

## Testing scope

Actual local CPU regressions compare the old/new default U-Net weights, outputs
and gradients, old/new objective traces and gradients, and old/new stochastic
and deterministic sampler trajectories. The original 32 tests also pass.

Native Hydra composition/CLI tests are supplied but were skipped locally because
hydra-core could not be installed in the execution environment. This is a
specific remaining verification gap, not a tested Hydra CLI claim. GPU behavior,
long training convergence and improved separation quality are not established.
