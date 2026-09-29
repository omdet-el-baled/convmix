# Conv-TasNet NPZ adapter verification

## Executed in this session

Environment: Python 3.13, PyTorch 2.10.0+cpu, PyTorch Lightning 2.6.5,
TorchMetrics 1.9.0. CUDA unavailable. Hydra and Asteroid unavailable;
package installation/download attempts failed. The user's Python 3.10/3.11
GPU environment was not used.

`python -m pytest -q`: **63 passed, 2 skipped**, 14 warnings.

The 63 tests include the existing 48 tests and **15 new baseline tests**:

* NPZ loader -> real waveform conversion for all three FFT normalizations,
  odd and even FFT lengths, finite loss and gradients;
* Wiener output compared against the formula in the supplied user script;
* double-precision autograd gradcheck through Wiener deconvolution;
* independent full linear time convolution followed by unregularized inversion;
* zero_mean False/True SI-SDR equivalence to TorchMetrics functional API;
* no running metric state contaminating repeated functional calls;
* per-example normalization independence from batch companions and finite silent
  mixture normalization;
* output-gain restoration and explicit legacy batch-peak behavior;
* invalid shape, filter length and regularizer checks;
* two real Lightning optimizer updates with the actual NPZ data module, using a
  small explicitly named waveform test double; checkpoint state round trip.

The two skipped modules are native Hydra integration tests. The added native
baseline module includes real Asteroid training/resume/evaluation and CLI tests
which run when Hydra and Asteroid are installed. These were **not executed here**.
Asteroid's ConvTasNet is used through a lazy constructor, not a substitute network.
Only the tests use a small test double to exercise the surrounding adapter.

No GPU tests, long training, user-dataset runs or quality metrics were executed.
All pre-existing project files remain unchanged; the add-on only adds new files.
