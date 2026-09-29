# Actual verification — component refactor

## Executed locally

Command (CPU):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --disable-warnings
```

Result: **48 passed, 1 skipped, 11 warnings**. The exact console output is in
`pytest_output.txt`; environment versions are in `environment.json`.
`compileall` also completed for the source modules and entry points.

This includes the existing 32 tests plus 16 new regression tests. Tests use small
synthetic data, not the user's datasets or trained checkpoints.

### Regression evidence

- Previous/default U-Net parameter keys, initial tensors, forward values and
  gradients match exactly in tested CPU cases: K=2 and K=3, even/odd FFT lengths,
  basic/enhanced variants, with dropout draws paired by seed.
- Refactored sampler outputs match the unchanged reference exactly for Euler,
  Heun, RK4, reverse-SDE Euler-Maruyama, fixed-step PC, and adaptive PC in both
  provided correction geometries. Initial states, NFE and score times match.
- Extracted standard/mismatch objective inputs, means/factors, targets, residuals,
  loss and gradients match the previous Lightning implementation. Both terminal
  and named legacy input-centering variants are covered.
- A deeper graph with different stage widths/block counts, no attention,
  alternative fixed time features, and real/imaginary input features runs forward
  and backward. An embedding with trainable parameters is rejected.
- Optimizer construction is replaceable via a factory. Every `_target_` in the
  supplied YAML tree resolves to an importable callable (static discovery).
- Existing numerical tests cover drift/mean/covariance consistency, physical
  rFFT noise, NPZ/DataLoader behavior, actual short Lightning training, optimizer
  and EMA updates, checkpoint restore/resume, and exact-Gaussian-score sampling.

`preserved_core.json` records byte-for-byte identity with the supplied input ZIP
for SDE, rFFT noise, sampler reference, metrics, and filter implementations.

## Not executed locally

**Native Hydra composition and CLI integration tests were skipped.** `hydra-core`
was unavailable, and attempts to install it could not reach the external
package repository. There is no mocked Hydra implementation or substituted
config parser in production. Tests in `tests/test_hydra_workflow.py` use real
Hydra and will execute when requirements are installed. They cover group
composition, nested/whole-network replacements, optimizer/scheduler selection,
sampler selection, checkpoint identity, and short native CLI training/evaluation.

Therefore the 48 passing tests are **not** a claim that the native Hydra CLI was
successfully run here. Run the full suite in the user's environment before a
long experiment. A skipped native module there means Hydra is still missing.

No CUDA tests, AMP tests, long training runs, user-checkpoint migration tests,
real-data results, separation-quality comparisons, or cross-device bitwise
reproducibility claims are included. TensorBoardLogger construction was not
exercised; local short training used CSV logging.

## Reading the tests

The old numerical code is retained as the sampler oracle; it was not deleted
and rewritten alongside the tests. U-Net and Lightning loss regression
references are stored under `tests/reference/`, sourced from the previous ZIP.
Passing the tests establishes the tested equivalences/identities, not a global
proof of model correctness or improved scientific results.
